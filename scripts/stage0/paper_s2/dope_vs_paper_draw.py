"""DOPE vs 논문용 YOLO — 08-16 그림 두 형식을 그대로 재현한다.

`dope_vs_paper_dump.py` 가 남긴 json 만 읽는다.  모델을 로드하지 않으므로 env 를 안 탄다.
바뀐 것은 오른쪽 모델 하나뿐이고, 프레임·색·라벨 규약은 08-16 과 같게 둔다.

    형식 A   det/peak 만.  예측 큐보이드(DOPE 빨강 / YOLO 노랑) + GT 초록 빈 원.
             밴드 없이 이미지 위에 직접 인쇄.
    형식 B   det/peak/kp_med.  예측 kp 파란 점 + 번호, 그 kp 로 푼 PnP 큐보이드 빨강,
             GT 초록 빈 원.  44px 검은 밴드.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/scripts/annotate")

import annotate_pnp as APNP                                    # noqa: E402

SRC = f"{ROOT}/data/pallet/eval_results/dope_vs_paper"
LEFT, RIGHT = "paper_s2_stageB", "paper_yolo26n_joint"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
FMT_A = ["1779448868035222528", "1779448633156790272"]
FMT_B = ["1779449266426633216", "1778651530557153024"]

RED, YELLOW, GREEN, BLUE = (80, 80, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)
F = cv2.FONT_HERSHEY_SIMPLEX


def lab(im, text, xy, colour, scale=0.6):
    """검은 외곽선 위에 컬러 — 밴드 없이 이미지 위에 직접 찍는다."""
    cv2.putText(im, text, xy, F, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, text, xy, F, scale, colour, 1, cv2.LINE_AA)


def cuboid(im, pts, colour, filled_r=4):
    ok = [p is not None for p in pts]
    for a, b in EDGES:
        if a < len(pts) and b < len(pts) and ok[a] and ok[b]:
            cv2.line(im, tuple(map(int, pts[a])), tuple(map(int, pts[b])),
                     colour, 2, cv2.LINE_AA)
    if filled_r:
        for p in pts:
            if p is not None:
                cv2.circle(im, tuple(map(int, p)), filled_r, colour, -1,
                           cv2.LINE_AA)


def gt_rings(im, gt8, r=4):
    for p in gt8:
        cv2.circle(im, (int(p[0]), int(p[1])), r, GREEN, 1, cv2.LINE_AA)


def panel_a(ip, r, name, colour):
    im = cv2.imread(ip).copy()
    cuboid(im, r["pred8"], colour)
    gt_rings(im, r["gt8"])
    n = r["n_det"]
    lab(im, f"{name}   det{n}/8   peak{r['peak']:.2f}", (8, 20),
        GREEN if n >= 6 else (0, 0, 255))
    lab(im, f"{r['sess']}   green=GT", (8, 40), (255, 255, 255), 0.5)
    return im


def panel_b(ip, r, name):
    im = cv2.imread(ip).copy()
    K = None
    jp = None
    # PnP 큐보이드 — 예측 kp 로 푼 pose 를 재투영한다 (GT pose 가 아니다).
    kps9 = list(r["pred8"]) + [r["pred_c"]]
    dims = tuple(r["dims"])
    if r["n_det"] >= 6:
        import glob
        fid = r["_fid"]
        jp = sorted(glob.glob(
            f"{ROOT}/challenge/data/01_real/manual_gt/*_manual_gt/{fid}.json")
            + glob.glob(f"{ROOT}/challenge/data/01_real/*/*_manual_gt/{fid}.json"))[0]
        d = json.load(open(jp))
        K = np.array(d["camera_data"]["intrinsics"], float) if isinstance(
            d["camera_data"].get("intrinsics"), list) else None
        if K is None:
            it = d["camera_data"]["intrinsics"]
            K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]],
                          [0, 0, 1]], float)
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=im.shape)
            if pose is not None:
                P3 = APNP.make_pallet_keypoints_3d(*pose.get("dims", dims))[:8]
                R = np.asarray(pose["R"], float)
                t = np.asarray(pose["t"], float).reshape(3, 1)
                proj = (K @ (R @ np.asarray(P3, float).T + t))
                proj = (proj[:2] / proj[2]).T
                cuboid(im, [tuple(p) for p in proj], (0, 0, 255), filled_r=0)
        except Exception as e:
            print(f"    PnP 실패 {r['_fid']}: {e}", flush=True)

    gt_rings(im, r["gt8"], r=5)
    for i, p in enumerate(r["pred8"]):
        if p is not None:
            cv2.circle(im, tuple(map(int, p)), 4, BLUE, -1, cv2.LINE_AA)
            cv2.putText(im, str(i), (int(p[0]) + 5, int(p[1]) - 5), F, 0.45,
                        BLUE, 1, cv2.LINE_AA)

    band = np.zeros((44, im.shape[1], 3), np.uint8)
    n, err = r["n_det"], r["kp_err"]
    col = GREEN if (n >= 6 and err < 20) else (0, 0, 255)
    e = "inf" if not np.isfinite(err) else f"{err:.1f}px"
    cv2.putText(band, f"{name}   det={n}/8   peak={r['peak']:.2f}   kp_med={e}",
                (8, 17), F, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(band, f"{r['sess']}  {r['_fid']}   pred=blue  PnP-cuboid=red  "
                      f"GT=green  dims=({dims[0]:.2f},{dims[1]:.2f},{dims[2]:.2f})",
                (8, 36), F, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, im])


def frame_png(fid):
    import glob
    for pat in (f"{ROOT}/challenge/data/01_real/manual_gt/*_manual_gt/{fid}.png",
                f"{ROOT}/challenge/data/01_real/*/*_manual_gt/{fid}.png"):
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    raise FileNotFoundError(fid)


def main():
    pd = json.load(open(f"{SRC}/pred_dope.json"))
    py = json.load(open(f"{SRC}/pred_yolo.json"))
    for src in (pd, py):
        for fid, r in src["pred"].items():
            r["_fid"] = fid

    for tag, fids, maker in (("A", FMT_A, "a"), ("B", FMT_B, "b")):
        rows = []
        for fid in fids:
            ip = frame_png(fid)
            if maker == "a":
                l = panel_a(ip, pd["pred"][fid], LEFT, RED)
                r_ = panel_a(ip, py["pred"][fid], RIGHT, YELLOW)
            else:
                l = panel_b(ip, pd["pred"][fid], LEFT)
                r_ = panel_b(ip, py["pred"][fid], RIGHT)
            rows.append(np.hstack([l, r_]))
        out = f"{SRC}/FORMAT_{tag}.png"
        cv2.imwrite(out, np.vstack(rows))
        print(f"  형식 {tag} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
