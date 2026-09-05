"""SOURCE_DEV(합성 val) 에서만 하는 사전 보정 — real GT 를 쓰지 않는다.

두 가지를 정한다.
  A. R0 의 coarse residual 분포  -> Gate C 의 crop jitter 분포
  B. 고전 CV 코너 후보 생성 파라미터 -> Gate B 에서 그대로 동결

DEV_EVAL 은 열지 않는다.  여기서 정한 값은 METHOD_LOCK 에 박고 결과를 본 뒤 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
import mtcd_teachers as T

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"


def syn_frame(stem):
    img = cv2.imread(str(SYN / "images/val" / f"{stem}.png"))
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    values = list(map(float, (SYN / "labels/val" / f"{stem}.txt")
                      .read_text().split("\n")[0].split()[5:]))
    xy = np.array([[values[3 * i] * w, values[3 * i + 1] * h] for i in range(9)])
    vis = np.array([values[3 * i + 2] for i in range(9)])
    return img, xy, vis


# ------------------------------------------------- classical corner candidates
def candidates_shi_tomasi(patch_gray, quality, min_dist, max_corners):
    pts = cv2.goodFeaturesToTrack(patch_gray, maxCorners=max_corners,
                                  qualityLevel=quality, minDistance=min_dist)
    return [] if pts is None else [tuple(map(float, p[0])) for p in pts]


def candidates_harris(patch_gray, block, ksize, k, rel_threshold):
    resp = cv2.cornerHarris(np.float32(patch_gray), block, ksize, k)
    if resp.max() <= 0:
        return []
    mask = resp > rel_threshold * resp.max()
    dil = cv2.dilate(resp, None)
    peaks = mask & (resp >= dil)
    ys, xs = np.nonzero(peaks)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def candidates_lsd_intersections(patch_gray, min_len, min_angle_deg, max_lines):
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(patch_gray)[0]
    if lines is None:
        return []
    segs = [l[0] for l in lines]
    segs = [s for s in segs if np.hypot(s[2] - s[0], s[3] - s[1]) >= min_len]
    segs = sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))[:max_lines]
    out = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            d1 = np.array([a[2] - a[0], a[3] - a[1]], float)
            d2 = np.array([b[2] - b[0], b[3] - b[1]], float)
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cosang = abs(float(d1 @ d2) / (n1 * n2))
            if cosang > np.cos(np.deg2rad(min_angle_deg)):
                continue
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                continue
            t = ((b[0] - a[0]) * d2[1] - (b[1] - a[1]) * d2[0]) / den
            out.append((float(a[0] + t * d1[0]), float(a[1] + t * d1[1])))
    return out


def candidates_gradient_junction(patch_gray, sigma, rel_threshold):
    """구조텐서의 작은 고유값 국소최대 — 두 방향 에지가 만나는 곳."""
    g = np.float32(patch_gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    tr, det = jxx + jyy, jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    lam_min = tr / 2.0 - disc
    if lam_min.max() <= 0:
        return []
    dil = cv2.dilate(lam_min, None)
    peaks = (lam_min >= dil) & (lam_min > rel_threshold * lam_min.max())
    ys, xs = np.nonzero(peaks)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


GRID = {
    "shi_tomasi": [{"quality": q, "min_dist": d, "max_corners": 20}
                   for q in (0.01, 0.03, 0.10) for d in (2, 3)],
    "harris": [{"block": b, "ksize": 3, "k": 0.04, "rel_threshold": r}
               for b in (2, 3) for r in (0.01, 0.05, 0.20)],
    "lsd": [{"min_len": L, "min_angle_deg": a, "max_lines": 8}
            for L in (5, 8) for a in (15, 30)],
    "junction": [{"sigma": s, "rel_threshold": r}
                 for s in (1.0, 1.5) for r in (0.05, 0.20)],
}
FUNCS = {"shi_tomasi": candidates_shi_tomasi, "harris": candidates_harris,
         "lsd": candidates_lsd_intersections, "junction": candidates_gradient_junction}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-residual", type=int, default=400)
    parser.add_argument("--n-candidate", type=int, default=120)
    parser.add_argument("--radius", type=int, default=12)
    args = parser.parse_args()

    stems = sorted(p.stem for p in (SYN / "images/val").glob("*.png"))
    random.Random(20260905).shuffle(stems)

    # ---------------------------------------------- A. R0 coarse residual -----
    reg = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    r0 = T.load_yolo(M.REPO_ROOT / reg["T0_R0_YOLO26N_G38LEGACY"]["checkpoint"])
    res_xy, res_norm, res_idx = [], [], []
    for stem in stems[:args.n_residual]:
        img, gt, vis = syn_frame(stem)
        if img is None:
            continue
        out = T.infer_yolo(r0, img, already_padded=True)
        if out.get("status") != "OK" or not out.get("keypoints_xy"):
            continue
        pred = np.asarray(out["keypoints_xy"], float)
        for i in range(8):
            if vis[i] > 0 and np.isfinite(pred[i]).all():
                res_xy.append((pred[i] - gt[i]).tolist())
                res_norm.append(float(np.linalg.norm(pred[i] - gt[i])))
                res_idx.append(i)
    res_xy = np.asarray(res_xy)
    res_norm = np.asarray(res_norm)
    # Gate C 의 crop jitter 는 이 벡터를 복원추출한다 — 가우시안 가정을 쓰지 않는다.
    np.save(M.AUDIT / "R0_SOURCE_COARSE_RESIDUALS.npy",
            np.concatenate([np.asarray(res_idx, np.float32)[:, None],
                            res_xy.astype(np.float32)], axis=1))
    residual = {
        "n": int(res_norm.size),
        "median_px": float(np.median(res_norm)),
        "p90_px": float(np.percentile(res_norm, 90)),
        "p99_px": float(np.percentile(res_norm, 99)),
        "std_x_px": float(res_xy[:, 0].std()),
        "std_y_px": float(res_xy[:, 1].std()),
        "mean_x_px": float(res_xy[:, 0].mean()),
        "mean_y_px": float(res_xy[:, 1].mean()),
    }
    del r0
    print("R0 coarse residual on SOURCE_DEV:", json.dumps(residual, indent=None))

    # ---------------------------------------------- B. candidate generators ---
    rng = np.random.default_rng(20260905)
    jitter_sigma = float(np.median([residual["std_x_px"], residual["std_y_px"]]))
    results = {}
    for family, configs in GRID.items():
        results[family] = []
        for cfg in configs:
            hits3 = hits5 = hits10 = total = 0
            counts = []
            for stem in stems[:args.n_candidate]:
                img, gt, vis = syn_frame(stem)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                for i in range(8):
                    if vis[i] <= 0:
                        continue
                    centre = gt[i] + rng.normal(0.0, jitter_sigma, 2)
                    cx, cy = int(round(centre[0])), int(round(centre[1]))
                    r = args.radius
                    x0, y0 = max(0, cx - r), max(0, cy - r)
                    x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
                    if x1 - x0 < 5 or y1 - y0 < 5:
                        continue
                    patch = gray[y0:y1, x0:x1]
                    try:
                        cands = FUNCS[family](patch, **cfg)
                    except Exception:
                        cands = []
                    total += 1
                    counts.append(len(cands))
                    if not cands:
                        continue
                    pts = np.asarray(cands, float) + np.array([x0, y0], float)
                    best = float(np.linalg.norm(pts - gt[i], axis=1).min())
                    hits3 += best <= 3.0
                    hits5 += best <= 5.0
                    hits10 += best <= 10.0
            results[family].append({
                "config": cfg, "n_patches": total,
                "coverage_3px": hits3 / total if total else 0.0,
                "coverage_5px": hits5 / total if total else 0.0,
                "coverage_10px": hits10 / total if total else 0.0,
                "candidates_per_patch_median": float(np.median(counts)) if counts else 0.0,
                "candidates_per_patch_p95": float(np.percentile(counts, 95)) if counts else 0.0,
            })
        best = max(results[family], key=lambda r: (r["coverage_5px"],
                                                   -r["candidates_per_patch_median"]))
        print(f"{family:12} best cov5 {best['coverage_5px']:.3f} "
              f"cov3 {best['coverage_3px']:.3f} cov10 {best['coverage_10px']:.3f} "
              f"n/patch med {best['candidates_per_patch_median']:.0f} "
              f"p95 {best['candidates_per_patch_p95']:.0f}  {best['config']}")

    report = {
        "population": "SOURCE_DEV synthetic val — real GT 미사용",
        "radius_px": args.radius,
        "jitter_sigma_px": jitter_sigma,
        "r0_coarse_residual": residual,
        "candidate_grid": results,
    }
    out = M.AUDIT / "SOURCE_CALIBRATION.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
