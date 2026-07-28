"""internet_pallet_infer.py — internet_pallet_data(처음 본 인터넷 파렛트)에 현재 최선
모델 s2(Stage B) 추론 + 파일명 치수(mm)를 dims 입력으로 PnP 큐보이드 + 배포필터 판정.

논문용 트랙(v1/v2 제외, 처음 본 파렛트 일반화) 정성 확인용.

- dims: 파일명에서 3~4자리 정수(100~2500mm)만 치수 후보로 추출(무게 kg=2자리 자동 제외),
  첫 3개=W,D,H(mm)→m. H 없으면 0.12m 기본(flag).
- 추론: s2 net_epoch_0057, squash-parity (T.infer_squash).
- ★K(카메라 내부파라미터) 미지 = 인터넷 사진 → fx-search 로 추정: HFOV 후보를 스윕하며
  각 K 로 solve_pose 를 돌려 reproj 최소인 focal 을 채택(고정 HFOV=60° 가정 폐기 — 그 근사가
  너무 넓어 PnP 가 K 오차를 포즈 기울임으로 흡수 → 큐보이드 깔때기 왜곡 발생했음).
  PnP 포즈/큐보이드·f7(posdepth) 는 이 추정 K 기준(절대 스케일/깊이는 여전히 미지 → 정성만).
  키포인트 검출과 belief 필터(f1,f2,f4,f5,f6)는 K 무관(불변).
- 배포필터 = f1&f2&f4&f5&f6&f7 (f3=L-R broken 제외, f8/f9=GT 필요 제외). GT 없음(정성만).
- overlay: 키포인트=파랑, PnP 큐보이드=빨강. 헤더=파일명·dims·검출·필터 PASS/FAIL(+실패이유).

산출: data/pallet/eval_results/paper_s2_scratch_diffpnp/internet_pallet_infer*.jpg

Usage: conda activate pallet-pose; python scripts/stage0/internet_pallet_infer.py
"""
from __future__ import annotations
import glob
import math
import os
import re
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_testset17_9filters as T   # noqa: E402  (infer_squash, M, TAU, ...)
import annotate_pnp as APNP               # noqa: E402
import cv2      # noqa: E402
import torch    # noqa: E402

M = T.M
TAU = T.TAU
E = T.E
N_DET_MIN = T.N_DET_MIN
EDGES = M.EDGES
SRC = os.path.join(ROOT, "data", "pallet", "raw_data", "internet_pallet_data")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "paper_s2_scratch_diffpnp")
OUT_BASE = os.path.join(OUT_DIR, "internet_pallet_infer.jpg")
DEPLOY = ["f1_peak", "f2_peak_ratio", "f4_tta_stab", "f5_rear_conf",
          "f6_frsep", "f7_posdepth"]
FSHORT = {"f1_peak": "corner-conf", "f2_peak_ratio": "peak-sharp",
          "f4_tta_stab": "TTA-stable", "f5_rear_conf": "rear-conf",
          "f6_frsep": "depth-sep", "f7_posdepth": "pose-z>0(approxK)"}
ENV = {"size_lo": 0.0, "size_hi": 1.0, "asp_lo": 0.0, "asp_hi": 10.0}
REAR = (4, 5, 6, 7)
# fx-search: K 미지 → HFOV 스윕 중 reproj 최소 focal 채택. coarse→fine 2단.
HFOV_COARSE = np.arange(25.0, 75.1, 5.0)   # 25~75° 5° 격자
HFOV_FINE_STEP = 1.0                        # best±5° 를 1° 로 재탐색
HFOV_FALLBACK = 60.0                        # under-det(큐보이드 안 그림) 시 무의미 K 용
PAD = 150          # reflect-pad (memory: squash가 near-field/truncation 과소검출 → pad 검출 회복)
THRESH = 0.3
PANEL_W = 560
MAX_PER_IMG = 12
BLUE, RED = (255, 0, 0), (0, 0, 255)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_dims_m(name):
    """파일명 -> (W,D,H) meters, h_default flag. 실패시 None."""
    nums = [int(n) for n in re.findall(r"\d+", name)]
    dims = [n for n in nums if 100 <= n <= 2500]
    if len(dims) < 2:
        return None
    W, D = dims[0], dims[1]
    if len(dims) >= 3:
        H, hdef = dims[2], False
    else:
        H, hdef = 120, True
    return (W / 1000.0, D / 1000.0, H / 1000.0), hdef


def K_from_hfov(w, h, hfov):
    fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], np.float64)


def _kps9_from_pred(pred8, pred_c):
    kps9 = [None if np.isnan(pred8[i, 0]) else [float(pred8[i, 0]), float(pred8[i, 1])]
            for i in range(8)]
    kps9.append(list(pred_c) if pred_c is not None else None)
    return kps9


def fit_K(pred8, pred_c, dims, shape):
    """fx-search: HFOV 스윕 중 solve_pose reproj 최소인 K 채택.

    고정 HFOV 근사는 K 오차를 포즈 기울임(큐보이드 깔때기 왜곡)으로 흡수시킴 → 대신
    HFOV(=focal) 를 원리적으로 탐색. coarse(5°) 로 최소 근방 잡고 fine(1°) 로 정밀화.
    반환 (K, hfov, reproj). 검출점 < N_DET_MIN 이면 None (큐보이드 안 그림).
    """
    kps9 = _kps9_from_pred(pred8, pred_c)
    if sum(1 for k in kps9 if k is not None) < N_DET_MIN:
        return None
    h, w = shape[0], shape[1]

    def reproj_at(hfov):
        K = K_from_hfov(w, h, hfov)
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=shape)
        except Exception:
            return None
        return None if pose is None else float(pose["reproj_error_px"])

    best_hfov, best_r = None, None
    for hf in HFOV_COARSE:
        r = reproj_at(hf)
        if r is not None and (best_r is None or r < best_r):
            best_hfov, best_r = float(hf), r
    if best_hfov is None:
        return None
    # fine: coarse 최소 ±5° 를 1° 로 (격자 경계 clamp)
    lo, hi = max(15.0, best_hfov - 5.0), min(85.0, best_hfov + 5.0)
    hf = lo
    while hf <= hi + 1e-6:
        r = reproj_at(hf)
        if r is not None and r < best_r:
            best_hfov, best_r = float(hf), r
        hf += HFOV_FINE_STEP
    return K_from_hfov(w, h, best_hfov), best_hfov, best_r


def deploy_verdict(scores, f7, n_det):
    if n_det < N_DET_MIN:
        return False, ["under-det"]
    row = {"n_det": n_det, "scores": scores, "f7_posdepth": bool(f7),
           "pred_sr": None, "pred_asp": None}
    failed = [FSHORT[f] for f in DEPLOY if not M.apply_filter(f, row, TAU.get(f), ENV)]
    return (len(failed) == 0), failed


def pnp_cuboid(pred8, pred_c, K, shape):
    kps9 = [None if np.isnan(pred8[i, 0]) else [float(pred8[i, 0]), float(pred8[i, 1])]
            for i in range(8)]
    kps9.append(list(pred_c) if pred_c is not None else None)
    if sum(1 for k in kps9 if k is not None) < N_DET_MIN:
        return None
    try:
        pose = APNP.solve_pose(kps9, K, dims=APNP.PALLET_DIMS, img_shape=shape)
    except Exception:
        return None
    if pose is None:
        return None
    pa = np.array(pose["projected_all"], float)[:8]
    bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
    pa[bad] = np.nan
    return pa


def panel(img, pred8, pred_c, proj, name, dims, hdef, n_det, passed, failed,
          hfov_used=None, reproj_used=None):
    if proj is not None:
        for a, b in EDGES:
            if not (np.isnan(proj[a, 0]) or np.isnan(proj[b, 0])):
                cv2.line(img, (int(proj[a, 0]), int(proj[a, 1])),
                         (int(proj[b, 0]), int(proj[b, 1])), RED, 2, cv2.LINE_AA)
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            p = (int(pred8[i, 0]), int(pred8[i, 1]))
            cv2.circle(img, p, 5, BLUE, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, BLUE, 1, cv2.LINE_AA)
    if pred_c is not None:
        cv2.drawMarker(img, (int(pred_c[0]), int(pred_c[1])), BLUE, cv2.MARKER_CROSS, 12, 2)
    H, W = img.shape[:2]
    img = cv2.resize(img, (PANEL_W, int(H * PANEL_W / W)))
    bcol = (0, 200, 0) if passed else (0, 0, 230)
    img = cv2.copyMakeBorder(img, 56, 6, 6, 6, cv2.BORDER_CONSTANT, value=bcol)
    dstr = (f"dims={dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m"
            + ("(H=default)" if hdef else ""))
    v = "FILTER PASS" if passed else "filter FAIL"
    kstr = (f"  K:fov={hfov_used:.0f}deg reproj={reproj_used:.1f}px"
            if hfov_used is not None else "")
    cv2.putText(img, f"det={n_det}/8  {v}{kstr}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    sub = "passes all 6" if passed else "fail: " + ", ".join(failed)
    cv2.putText(img, sub, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230),
                1, cv2.LINE_AA)
    # 파일명 캡션(위)
    cap = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(cap, f"{name[:34]}  {dstr}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (200, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([cap, img])


def hcat(a, b):
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return np.hstack([pad(a), pad(b)])


def vcat(imgs):
    w = max(x.shape[1] for x in imgs)
    pad = lambda x: cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return np.vstack([pad(x) for x in imgs])


def process(model, fp):
    img = cv2.imread(fp)
    if img is None:
        return None, "imread 실패(webp 등)"
    parsed = parse_dims_m(os.path.basename(fp))
    if parsed is None:
        return None, "치수 파싱 실패"
    dims, hdef = parsed
    APNP.PALLET_DIMS = dims                    # solve_pose call-time
    # reflect-pad 추론 (near-field/truncation 검출 회복; squash 대비 det ↑ 확인됨)
    belief, geom, wh = M.infer_belief(model, img, DEV, PAD)
    pred8, pred_c, peaks, ratios = M.belief_to_pred(belief, geom, wh, PAD, THRESH)
    n_det = int((~np.isnan(pred8[:, 0])).sum())
    # K 미지 → fx-search (검출점으로 reproj 최소 focal 추정). under-det 이면 무의미 K.
    fit = fit_K(pred8, pred_c, dims, img.shape) if n_det >= N_DET_MIN else None
    if fit is not None:
        K, hfov_used, reproj_used = fit
    else:
        K, hfov_used, reproj_used = K_from_hfov(img.shape[1], img.shape[0],
                                                HFOV_FALLBACK), None, None
    scores, f7 = {}, False
    if n_det >= N_DET_MIN:
        det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
        scores["f1_peak"] = float(min(peaks[i] for i in det8))
        scores["f2_peak_ratio"] = float(min(ratios[i] for i in det8))
        scores["f4_tta_stab"] = M.tta_stability(model, img, DEV, PAD, THRESH)
        rear = [i for i in REAR if not np.isnan(pred8[i, 0])]
        scores["f5_rear_conf"] = float(min(peaks[i] for i in rear)) if len(rear) == 4 else None
        scores["f6_frsep"] = M.frsep_frac(pred8)
        f7, _ = M.posdepth_ok(pred8, pred_c, K, img.shape)
    passed, failed = deploy_verdict(scores, f7, n_det)
    proj = pnp_cuboid(pred8, pred_c, K, img.shape) if n_det >= N_DET_MIN else None
    cell = panel(img, pred8, pred_c, proj, os.path.basename(fp), dims, hdef,
                 n_det, passed, failed, hfov_used, reproj_used)
    return (cell, n_det, passed, failed, dims, hdef, hfov_used, reproj_used), None


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*")))
    model = E.load_model(T.WEIGHTS, DEV)
    print(f"[internet_pallet] {len(files)} files in folder; s2 Stage B reflect-pad; "
          f"K=fx-search(HFOV {HFOV_COARSE[0]:.0f}~{HFOV_COARSE[-1]:.0f}deg, reproj-min)")
    cell_dir = os.path.join(OUT_DIR, "internet_pallet_cells")
    os.makedirs(cell_dir, exist_ok=True)
    idx, npass, saved = 0, 0, []
    for fp in files:
        res, err = process(model, fp)
        b = os.path.basename(fp)
        if res is None:
            print(f"  SKIP {b}  ({err})")
            continue
        cell, n_det, passed, failed, dims, hdef, hfov_used, reproj_used = res
        idx += 1
        npass += int(passed)
        wmm, dmm, hmm = (int(round(x * 1000)) for x in dims)
        tag = "PASS" if passed else "FAIL"
        out_name = f"{idx:02d}_{tag}_{wmm}x{dmm}x{hmm}.jpg"
        p = os.path.join(cell_dir, out_name)
        cv2.imwrite(p, cell, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(p)
        kstr = (f" K:fov={hfov_used:.0f}deg/reproj={reproj_used:.1f}px"
                if hfov_used is not None else " K:n/a(under-det)")
        print(f"  {b[:40]:<42} dims={dims} det={n_det}{kstr} "
              f"{'PASS' if passed else 'fail:'+','.join(failed)} -> {out_name}")
    print(f"\n[done] {len(saved)} inferred, {npass} pass filter -> {cell_dir}/ ({len(saved)} images)")


if __name__ == "__main__":
    main()
