"""paper_s2_filterval_passfail_withf3_overlays.py — Stage B (net_epoch_0057) on
REAL filterval (outside44 + night43 + manual36 = 123), split each frame into
pass / fail / underdet by the FULL GT-free filter AND (INCLUDING f3_flip), and
save a per-frame overlay image (pred kp = blue, PnP cuboid = red) into per-domain
pass/fail/underdet folders. Separate output folder from the f3-excluded version.

pass = f1_peak & f2_peak_ratio & f3_flip & f4_tta_stab & f5_rear_conf & f6_frsep
       & f7_posdepth  (ALL 7 GT-free filters AND, TAU = existing s1 values;
       f3_flip=10px, flip un-flip=(W-1)-x). f8_size_env/f9_bbox_iou EXCLUDED
       (need GT). NO GT used for filtering or overlay.
underdet = n_det < 6 (pre-filter stage).

REUSE (no filter/decode logic reimplemented):
  F = paper_s2_filterval_9filters : imported at top so its (W-1)-x flip override
      (T.flip_infer_squash = _flip_infer_squash_wm1) is applied to build_rec's f3.
  T = paper_s2_testset17_9filters : build_rec / infer_squash / M / TAU / N_DET_MIN /
      E / WEIGHTS.
  S = stage25_paperbase_eval : frames_filterval() -> (dom, fid, jp, ip).
  APNP.solve_pose(dims=(1.1,1.3,0.12)) for the red PnP cuboid.

Usage: conda activate pallet-pose;
       python scripts/stage0/paper_s2/paper_s2_filterval_passfail_withf3_overlays.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

# challenge/scripts 도 계열 폴더로 나뉘었다. 예전엔 먼저 로드된 다른 모듈이
# 넣어 주는 데 기대고 있었는데(아래 import annotate_pnp), 그건 로드 순서 의존이다.
_CS = _os.path.join(_os.path.dirname(_os.path.dirname(_S0)), "challenge", "scripts")
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in ("annotate", "infer", "live")]

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_9filters as F  # noqa: E402,F401 (applies (W-1)-x flip)
import paper_s2_testset17_9filters as T  # noqa: E402
import stage25_paperbase_eval as S       # noqa: E402
import annotate_pnp as APNP              # noqa: E402
import cv2                               # noqa: E402
import torch                             # noqa: E402

M = T.M
TAU = T.TAU
E = T.E
WEIGHTS = T.WEIGHTS
N_DET_MIN = T.N_DET_MIN
EDGES = M.EDGES

DIMS = (1.1, 1.3, 0.12)
APNP.PALLET_DIMS = DIMS
PASS_FILTERS = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
                "f5_rear_conf", "f6_frsep", "f7_posdepth"]
DOMAINS = ["outside", "night", "manual"]
OUT_ROOT = os.path.join(ROOT, "data", "pallet", "eval_results",
                        "paper_s2_scratch_diffpnp", "filterval_passfail_withf3")

BLUE = (255, 0, 0)
RED = (0, 0, 255)


def accept(r):
    """full GT-free AND over PASS_FILTERS (includes f3_flip; env-free filters)."""
    row = {"n_det": r["n_det"], "scores": r["scores"],
           "f7_posdepth": r["f7_posdepth"],
           "pred_sr": r["pred_sr"], "pred_asp": r["pred_asp"]}
    passed, failed = [], []
    for f in PASS_FILTERS:
        (passed if M.apply_filter(f, row, TAU.get(f), None) else failed).append(f)
    return (len(failed) == 0), failed


def pnp_cuboid(pred8, pred_c, K, img_shape):
    """solve_pose (dims=DIMS) -> proj_all(8,2 nan-masked) for red cuboid."""
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(list(pred_c) if pred_c is not None else None)
    if sum(1 for k in kps9 if k is not None) < N_DET_MIN:
        return None
    try:
        pose = APNP.solve_pose(kps9, K, dims=DIMS, img_shape=img_shape)
    except Exception:
        return None
    if pose is None:
        return None
    pa = np.array(pose["projected_all"], float)[:8]
    bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
    pa[bad] = np.nan
    return pa


def save_overlay(ip, pred8, pred_c, proj_all, verdict, fid, n_det, failed, path):
    img = cv2.imread(ip)
    if img is None:
        return False
    if proj_all is not None:
        for a, b in EDGES:
            if not (np.isnan(proj_all[a, 0]) or np.isnan(proj_all[b, 0])):
                cv2.line(img, (int(proj_all[a, 0]), int(proj_all[a, 1])),
                         (int(proj_all[b, 0]), int(proj_all[b, 1])), RED, 2, cv2.LINE_AA)
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            p = (int(pred8[i, 0]), int(pred8[i, 1]))
            cv2.circle(img, p, 5, BLUE, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)
    if pred_c is not None:
        cv2.drawMarker(img, (int(pred_c[0]), int(pred_c[1])), BLUE,
                       cv2.MARKER_CROSS, 12, 2)
    fx = "-" if verdict != "FAIL" else ("|".join(f.split("_")[0] for f in failed) or "-")
    h1 = f"{verdict}  {fid}  det={n_det}/8  fail=[{fx}]"
    h2 = "pred=blue  PnP-cuboid=red  dims=(1.1,1.3,0.12)  GT-free AND incl. f3_flip"
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, h1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, h2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dm in DOMAINS:
        for v in ("pass", "fail", "underdet"):
            os.makedirs(os.path.join(OUT_ROOT, dm, v), exist_ok=True)

    frames = S.frames_filterval()
    print(f"[filterval] {len(frames)} frames; dims={DIMS}; weights=Stage B ep0057; "
          f"squash-parity; flip=(W-1)-x; pass=AND{PASS_FILTERS}")
    model = E.load_model(WEIGHTS, device)

    counts = {dm: {"pass": 0, "fail": 0, "underdet": 0} for dm in DOMAINS}
    saved = 0
    for dom, fid, jp, ip in frames:
        r = T.build_rec(model, dom, fid, jp, ip, device)
        if r is None:
            print(f"  {dom:<8}{fid}  build_rec=None (skip)")
            continue
        img = cv2.imread(ip)
        _, pred8, pred_c, _, _ = T.infer_squash(model, img, device)
        n_det = r["n_det"]

        if n_det < N_DET_MIN:
            verdict, sub, failed, proj = "UNDERDET", "underdet", [], None
        else:
            ok, failed = accept(r)
            verdict, sub = ("PASS", "pass") if ok else ("FAIL", "fail")
            K = E.K_from_json(json.load(open(jp)))
            proj = pnp_cuboid(pred8, pred_c, K, img.shape)
        counts[dom][sub] += 1
        path = os.path.join(OUT_ROOT, dom, sub, f"{fid}.jpg")
        if save_overlay(ip, pred8, pred_c, proj, verdict, fid, n_det, failed, path):
            saved += 1
        cm = f"{r['corner_med']:.1f}" if r["corner_med"] is not None else "-"
        print(f"  {dom:<8}{fid}  det={n_det} {verdict:<8} cm={cm} "
              f"fail={failed if verdict=='FAIL' else ''}")

    print("\n=== per-domain counts (WITH f3) ===")
    print(f"{'domain':<10}{'pass':>6}{'fail':>6}{'underdet':>10}{'total':>7}")
    print("-" * 39)
    tot = {"pass": 0, "fail": 0, "underdet": 0}
    for dm in DOMAINS:
        c = counts[dm]
        n = c["pass"] + c["fail"] + c["underdet"]
        for k in tot:
            tot[k] += c[k]
        print(f"{dm:<10}{c['pass']:>6}{c['fail']:>6}{c['underdet']:>10}{n:>7}")
    print("-" * 39)
    ntot = tot["pass"] + tot["fail"] + tot["underdet"]
    print(f"{'ALL':<10}{tot['pass']:>6}{tot['fail']:>6}{tot['underdet']:>10}{ntot:>7}")
    print(f"\n[save] {saved} overlays -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
