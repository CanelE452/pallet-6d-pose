"""DOPE 모델로 mp4 frame-by-frame 추론 → annotated mp4 저장.

usage:
    python challenge/scripts/infer/dope_predict_mp4.py \
        --weights weights/challenge_track/challengenight/final_net_epoch_0120.pth \
        --mp4 data/outside/forklift_raw_20260528_163408.mp4 \
        --out runs/dope_forklift.mp4

camera intrinsics 는 challenge/config/task.yaml 의 (614.18, 614.31, 329.28, 234.53) 사용.
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# scripts/data_prep/ 의 shared visualize_inference 재사용
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "scripts", "data_prep"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "self_training"))
sys.path.insert(0, os.path.join(_REPO, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_REPO, "Deep_Object_Pose", "train"))

from visualize_inference import (
    load_model, infer, extract_keypoints, draw_overlay,
)
from pnp_solver import PalletPnPSolver, make_camera_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.3, help="belief peak threshold")
    ap.add_argument("--fx", type=float, default=614.18)
    ap.add_argument("--fy", type=float, default=614.31)
    ap.add_argument("--cx", type=float, default=329.28)
    ap.add_argument("--cy", type=float, default=234.53)
    ap.add_argument("--width",  type=float, default=1.1)
    ap.add_argument("--depth",  type=float, default=1.3)
    ap.add_argument("--height", type=float, default=0.11)
    ap.add_argument("--label", default="DOPE")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[weights] {args.weights}")
    model = load_model(args.weights, device)

    K = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(K, pallet_dims=(args.width, args.depth, args.height))

    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        print(f"failed to open: {args.mp4}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[mp4] {w}x{h} @ {fps:.1f}fps, {n} frames")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    t0 = time.time()
    f = 0
    n_det = 0
    n_pnp = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        belief = infer(model, img, device)
        kps = extract_keypoints(belief, args.threshold)
        vis = draw_overlay(img, kps, None, belief, pnp, f"{args.label} f{f}")
        writer.write(vis)
        detected = sum(1 for kp in kps if kp is not None)
        if detected >= 4:
            n_det += 1
        # 진행 print 매 50 frame
        if f % 50 == 0:
            elapsed = time.time() - t0
            fps_cur = (f + 1) / max(1e-3, elapsed)
            eta = (n - f - 1) / max(1e-3, fps_cur)
            print(f"  frame {f:4d}/{n}  det={detected}/9  ({fps_cur:.1f} FPS, ETA {eta:.0f}s)")
        f += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print()
    print(f"[done] {f} frames processed in {elapsed:.1f}s  ({f/elapsed:.1f} FPS)")
    print(f"[done] detected (≥4 kps) : {n_det}/{f} ({100*n_det/f:.1f}%)")
    print(f"[done] saved: {args.out}")


if __name__ == "__main__":
    main()
