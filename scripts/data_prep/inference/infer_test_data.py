"""test_data (capture*/rgb/) 구조에 대한 DOPE 추론 + 오버레이 시각화.

사용법:
    python scripts/data_prep/infer_test_data.py \
        --weights weights/misc/pallet_v11_far/final_net_epoch_0121.pth \
        --test_dir data/pallet/test_data \
        --output_dir data/pallet/test_data_results \
        --num_per_capture 20
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "self_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # data_prep for shared libs

from visualize_inference import (
    load_model, infer, extract_keypoints, draw_overlay,
)
from pnp_solver import PalletPnPSolver, make_camera_matrix


def parse_cam_K(cam_K_path):
    """cam_K.txt (3x3 행렬) 파싱 → fx, fy, cx, cy."""
    K = np.loadtxt(cam_K_path)
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test_dir", default="data/pallet/test_data")
    parser.add_argument("--output_dir", default="data/pallet/test_data_results")
    parser.add_argument("--num_per_capture", type=int, default=0,
                        help="capture당 샘플 수 (0=전체)")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    print(f"Model loaded: {args.weights} (device: {device})")

    # capture*/rgb/ 구조 탐색
    captures = sorted([
        d for d in os.listdir(args.test_dir)
        if os.path.isdir(os.path.join(args.test_dir, d))
        and os.path.exists(os.path.join(args.test_dir, d, "rgb"))
    ])

    # 플랫 디렉토리 (*.png가 직접 있는 경우) → 단일 capture로 처리
    if not captures:
        flat_imgs = glob.glob(os.path.join(args.test_dir, "*.png")) + \
                    glob.glob(os.path.join(args.test_dir, "*.jpg"))
        if flat_imgs:
            captures = ["."]
            print(f"Flat directory mode: {len(flat_imgs)} images")
        else:
            print("No images found")
            return
    else:
        print(f"Found {len(captures)} captures: {captures}")

    total_processed = 0
    total_detected = 0
    total_pnp_ok = 0

    for cap in captures:
        cap_dir = os.path.join(args.test_dir, cap) if cap != "." else args.test_dir
        rgb_dir_candidate = os.path.join(cap_dir, "rgb")
        rgb_dir = rgb_dir_candidate if os.path.isdir(rgb_dir_candidate) else cap_dir
        cam_K_path = os.path.join(cap_dir, "cam_K.txt")

        if os.path.exists(cam_K_path):
            fx, fy, cx, cy = parse_cam_K(cam_K_path)
        else:
            fx, fy, cx, cy = 614.18, 614.31, 329.28, 234.53
        cam = make_camera_matrix(fx, fy, cx, cy)
        pnp = PalletPnPSolver(cam)
        print(f"\n=== {cap} (fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}) ===")

        imgs = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        if not imgs:
            imgs = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        print(f"  {len(imgs)} images found")

        if args.num_per_capture > 0 and len(imgs) > args.num_per_capture:
            indices = np.linspace(0, len(imgs) - 1, args.num_per_capture, dtype=int)
            imgs = [imgs[i] for i in indices]
            print(f"  Sampling {len(imgs)} images")

        out_dir = args.output_dir if cap == "." else os.path.join(args.output_dir, cap)
        os.makedirs(out_dir, exist_ok=True)

        for i, img_path in enumerate(imgs):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [SKIP] Cannot read: {img_path}")
                continue

            belief = infer(model, img, device)
            pred_kps = extract_keypoints(belief, args.threshold)
            vis = draw_overlay(img, pred_kps, None, belief, pnp,
                               f"{cap}/{os.path.basename(img_path)}")

            basename = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{basename}_overlay.jpg")
            cv2.imwrite(out_path, vis)

            detected = sum(1 for kp in pred_kps if kp is not None)
            pred_orig = []
            h, w = img.shape[:2]
            bh, bw = belief.shape[1], belief.shape[2]
            sx, sy = w / bw, h / bh
            for kp in pred_kps:
                if kp is None:
                    pred_orig.append(None)
                else:
                    pred_orig.append((kp[0] * sx, kp[1] * sy))
            success, _, _, _ = pnp.solve(pred_orig)

            total_processed += 1
            total_detected += detected
            if success:
                total_pnp_ok += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(imgs):
                print(f"  [{i+1}/{len(imgs)}] {detected}/9 kps, PnP: {'OK' if success else 'FAIL'}")

        print(f"  Results saved to: {out_dir}")

    print(f"\n{'='*50}")
    print(f"Total: {total_processed} images processed")
    print(f"Avg keypoints detected: {total_detected / max(total_processed, 1):.1f}/9")
    print(f"PnP success rate: {total_pnp_ok}/{total_processed} ({100*total_pnp_ok/max(total_processed,1):.1f}%)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
