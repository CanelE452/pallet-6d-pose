"""_diag_kfov_sweep.py — DIAGNOSTIC ONLY. HFOV(=K) sweep로 wrong-K 가설 검증.

각 이미지에서 HFOV 를 30~90° 로 바꿔가며 solve_pose reproj(선택 후보) + tilt 를 측정.
reproj 가 특정 HFOV 에서 급락하면 => 깔때기 원인 = 잘못된 K(HFOV=60 근사). 그리고
그 HFOV 에서 tilt 가 1.0(flat slab)에 가까워지는지 확인.

Usage: conda activate pallet-pose; python scripts/stage0/diag/_diag_kfov_sweep.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import math
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

import paper_s2_testset17_9filters as T   # noqa: E402
import internet_pallet_infer as IPI       # noqa: E402
import annotate_pnp as APNP               # noqa: E402
import cv2                                # noqa: E402
import torch                              # noqa: E402

M = T.M
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = {"09": "중고파렛트(18kg) [1100*1100*150mm] .png",
           "08": "수출용파렛트(1000×1000×120)4.jpg"}


def K_for(w, h, hfov):
    fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], np.float64)


def main():
    model = T.E.load_model(T.WEIGHTS, DEV)
    for tag, name in TARGETS.items():
        fp = os.path.join(IPI.SRC, name)
        img = cv2.imread(fp)
        dims, _ = IPI.parse_dims_m(name)
        APNP.PALLET_DIMS = dims
        belief, geom, wh = M.infer_belief(model, img, DEV, IPI.PAD)
        pred8, pred_c, _, _ = M.belief_to_pred(belief, geom, wh, IPI.PAD, IPI.THRESH)
        kps9 = [None if np.isnan(pred8[i, 0]) else [float(pred8[i, 0]), float(pred8[i, 1])]
                for i in range(8)]
        kps9.append(list(pred_c) if pred_c is not None else None)
        print(f"\n[{tag}] {name}  dims={dims}  img={img.shape[1]}x{img.shape[0]}")
        print("  HFOV   fx     reproj   tilt   deckz   strict")
        for hfov in (30, 35, 40, 45, 50, 55, 60, 70, 80, 90):
            K = K_for(img.shape[1], img.shape[0], hfov)
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=img.shape)
            if pose is None:
                print(f"  {hfov:4d}   -      None")
                continue
            nz = (pose["R"] @ np.array([0.0, -1.0, 0.0]))[2]
            print(f"  {hfov:4d}  {K[0,0]:6.1f}  {pose['reproj_error_px']:6.2f}  "
                  f"{pose['_v8_tilt']:.3f}  {nz:+.3f}  {pose.get('_v6_strict_passed')}")


if __name__ == "__main__":
    main()
