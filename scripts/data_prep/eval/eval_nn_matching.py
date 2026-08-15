"""NN matching 평가 — raw keypoint vs GT projected_cuboid, Hungarian assignment.

사용법:
    python scripts/data_prep/eval/eval_nn_matching.py \
        --weights weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth \
        --test_dir data/pallet/raw_data/capture0403middle \
        --gt_dir data/pallet/raw_data/capture0403middle/gt_final_isaac
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Deep_Object_Pose", "train"))

from models import DopeNetwork


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_keypoints_from_belief(belief_maps, threshold=0.3):
    OFFSET = 0.4395
    WIN = 11
    RAN = WIN // 2
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap_ori = belief_maps[i]
        max_val = bmap_ori.max()
        if max_val < threshold:
            keypoints.append(None)
            continue
        bmap_smooth = gaussian_filter(bmap_ori, sigma=2)
        p = 1
        pad_l = np.zeros_like(bmap_smooth); pad_l[p:, :] = bmap_smooth[:-p, :]
        pad_r = np.zeros_like(bmap_smooth); pad_r[:-p, :] = bmap_smooth[p:, :]
        pad_u = np.zeros_like(bmap_smooth); pad_u[:, p:] = bmap_smooth[:, :-p]
        pad_d = np.zeros_like(bmap_smooth); pad_d[:, :-p] = bmap_smooth[:, p:]
        peaks_binary = (
            (bmap_smooth >= pad_l) & (bmap_smooth >= pad_r) &
            (bmap_smooth >= pad_u) & (bmap_smooth >= pad_d) &
            (bmap_smooth > threshold)
        )
        peak_ys, peak_xs = np.nonzero(peaks_binary)
        if len(peak_xs) == 0:
            keypoints.append(None)
            continue
        peak_vals = [bmap_ori[py, px] for py, px in zip(peak_ys, peak_xs)]
        best_idx = np.argmax(peak_vals)
        px, py = int(peak_xs[best_idx]), int(peak_ys[best_idx])
        y_lo = max(0, py - RAN); y_hi = min(bmap_ori.shape[0], py + RAN + 1)
        x_lo = max(0, px - RAN); x_hi = min(bmap_ori.shape[1], px + RAN + 1)
        patch = bmap_ori[y_lo:y_hi, x_lo:x_hi]
        if patch.sum() > 0:
            ys = np.arange(y_lo, y_hi)
            xs = np.arange(x_lo, x_hi)
            xg, yg = np.meshgrid(xs, ys)
            wx = np.average(xg, weights=patch) + OFFSET
            wy = np.average(yg, weights=patch) + OFFSET
        else:
            wx, wy = float(px), float(py)
        keypoints.append((wx, wy))
    return keypoints


def nn_matching_error(pred_kps, gt_cuboid):
    """Hungarian assignment between pred corners (up to 8) and GT projected_cuboid (8).
    Returns per-keypoint distances (only matched valid ones)."""
    valid_pred = []
    valid_pred_idx = []
    for i, kp in enumerate(pred_kps[:8]):
        if kp is not None:
            valid_pred.append(kp)
            valid_pred_idx.append(i)

    if len(valid_pred) == 0:
        return None, 0

    pred_arr = np.array(valid_pred)  # (n, 2)
    gt_arr = np.array(gt_cuboid[:8])  # (8, 2)

    # Cost matrix: (n_pred, 8)
    cost = np.linalg.norm(pred_arr[:, None, :] - gt_arr[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    dists = cost[row_ind, col_ind]
    return dists, len(valid_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, nargs="+")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--gt_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.gt_dir is None:
        args.gt_dir = os.path.join(args.test_dir, "gt_final_isaac")

    rgb_dir = os.path.join(args.test_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        rgb_dir = args.test_dir

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.json")))
    print(f"GT files: {len(gt_files)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for weights_path in args.weights:
        print(f"\n{'='*60}")
        print(f"Model: {weights_path}")
        print(f"{'='*60}")

        model = load_model(weights_path, device)

        all_dists = []
        frame_means = []  # per-frame mean error
        n_frames_with_pred = 0

        for ji, gt_path in enumerate(gt_files):
            basename = os.path.splitext(os.path.basename(gt_path))[0]

            img_path = None
            for ext in [".png", ".jpg"]:
                candidate = os.path.join(rgb_dir, basename + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            with open(gt_path) as f:
                gt_data = json.load(f)
            gt_cuboid = gt_data["objects"][0]["projected_cuboid"]

            img = cv2.imread(img_path)
            h_orig, w_orig = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (448, 448))
            img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            with torch.no_grad():
                out_bel, out_aff = model(tensor)

            belief = out_bel[-1][0].cpu().numpy()
            pred_kps = extract_keypoints_from_belief(belief, args.threshold)

            bh, bw = belief.shape[1], belief.shape[2]
            sx, sy = bw / w_orig, bh / h_orig

            pred_kps_orig = []
            for kp in pred_kps[:8]:
                if kp is None:
                    pred_kps_orig.append(None)
                else:
                    pred_kps_orig.append((kp[0] / sx, kp[1] / sy))

            dists, n_valid = nn_matching_error(pred_kps_orig, gt_cuboid)
            if dists is not None:
                all_dists.extend(dists.tolist())
                frame_means.append(dists.mean())
                n_frames_with_pred += 1

            if (ji + 1) % 100 == 0:
                print(f"  [{ji+1}/{len(gt_files)}] processed")

        all_dists = np.array(all_dists)
        frame_means = np.array(frame_means)
        n_total = len(gt_files)

        print(f"\n  Frames with predictions: {n_frames_with_pred}/{n_total}")
        print(f"  Total keypoint matches: {len(all_dists)}")

        if len(all_dists) > 0:
            print(f"\n  --- Per-keypoint ---")
            print(f"  Mean dist:   {all_dists.mean():.1f} px")
            print(f"  Median dist: {np.median(all_dists):.1f} px")
            for thr in [5, 10, 20, 50, 100]:
                pct = (all_dists < thr).mean() * 100
                print(f"  <{thr}px:     {pct:.1f}%  ({(all_dists < thr).sum()}/{len(all_dists)})")

        if len(frame_means) > 0:
            print(f"\n  --- Per-frame (mean error per frame, denom={n_total}) ---")
            print(f"  Mean:   {frame_means.mean():.1f} px")
            print(f"  Median: {np.median(frame_means):.1f} px")
            for thr in [5, 10, 20, 50, 100]:
                n_pass = (frame_means < thr).sum()
                pct = n_pass / n_total * 100
                print(f"  <{thr}px:     {pct:.1f}%  ({n_pass}/{n_total})")


if __name__ == "__main__":
    main()
