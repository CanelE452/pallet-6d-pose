"""Phase 1 정성 시각화 — R0 (ep65) vs R1_outside_loo 비교 panel.

각 도메인 (indoor / outside / night) 에서 manual_gt frame 일부 선택 →
GT cuboid 오버레이 + R0 prediction + R1 prediction 옆에 나란히 표시.

사용:
    python scripts/data_prep/eval/qualitative_panel.py \\
        --output _docs/figures/phase1_qualitative.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_ROOT, "Deep_Object_Pose", "train"))

from models import DopeNetwork


# cuboid edges (front face / back face / connecting)
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # front
    (4, 5), (5, 6), (6, 7), (7, 4),  # back
    (0, 4), (1, 5), (2, 6), (3, 7),  # connect
]


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def extract_keypoints(belief_maps, threshold=0.3):
    OFFSET = 0.4395
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap = belief_maps[i]
        if bmap.max() < threshold:
            keypoints.append(None)
            continue
        smooth = gaussian_filter(bmap, sigma=2)
        ys, xs = np.unravel_index(smooth.argmax(), smooth.shape)
        win = 5; half = win // 2
        y0 = max(0, ys - half); y1 = min(bmap.shape[0], ys + half + 1)
        x0 = max(0, xs - half); x1 = min(bmap.shape[1], xs + half + 1)
        patch = bmap[y0:y1, x0:x1]
        if patch.sum() < 1e-6:
            keypoints.append((float(xs), float(ys)))
            continue
        yg, xg = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        wx = np.average(xg, weights=patch) + OFFSET
        wy = np.average(yg, weights=patch) + OFFSET
        keypoints.append((float(wx), float(wy)))
    return keypoints


def infer_image(model, img_bgr, device, image_size=448):
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = w0 / bw, h0 / bh
    raw_kps = extract_keypoints(belief)
    kps = []
    for kp in raw_kps[:8]:
        if kp is None:
            kps.append(None)
        else:
            kps.append((kp[0] * sx, kp[1] * sy))
    return kps


def draw_kps(ax, img, kps, color, label, gt_kps=None):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if gt_kps is not None:
        for (a, b) in CUBOID_EDGES:
            if a < len(gt_kps) and b < len(gt_kps):
                pa, pb = gt_kps[a], gt_kps[b]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="lime", linewidth=1.5, alpha=0.7)
    for i, kp in enumerate(kps):
        if kp is None:
            continue
        ax.scatter(kp[0], kp[1], c=color, s=35, edgecolor="black", linewidth=0.6, zorder=5)
        ax.text(kp[0] + 4, kp[1] - 4, str(i), color=color, fontsize=8, fontweight="bold")
    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--r0_weights", default="weights/v8_ablation_A_coord/final_net_epoch_0065.pth")
    p.add_argument("--r1_indoor", default="weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth")
    p.add_argument("--r1_outside", default="weights/selftrain/r1_outside_loo/final_net_epoch_0096.pth")
    p.add_argument("--r1_night", default="weights/selftrain/r1_outside_loo/final_net_epoch_0096.pth")
    p.add_argument("--n_per_domain", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading R0 model...")
    r0 = load_model(args.r0_weights, device)
    print("Loading R1 indoor (F5)...")
    r1_in = load_model(args.r1_indoor, device)
    print("Loading R1 outside_loo...")
    r1_out = load_model(args.r1_outside, device)
    print("Loading R1 night (outside_loo)...")
    r1_ni = load_model(args.r1_night, device)
    r1_map = {"indoor": r1_in, "outside": r1_out, "night": r1_ni}
    r1_label = {"indoor": "R1_indoor (F5)",
                "outside": "R1_outside_loo",
                "night": "R1_outside_loo (best for night)"}

    # 도메인별 frame 선택
    domains = {
        "indoor": "data/pallet/raw_data/capture0403middle",
        "outside": "data/_eval_sets/outside_combined",
        "night": "data/_eval_sets/night_combined",
    }

    n = args.n_per_domain
    fig, axes = plt.subplots(3 * n, 2, figsize=(8, 4 * n * 3))
    if 3 * n == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for d_name, d_path in domains.items():
        if d_name == "indoor":
            rgb_dir = os.path.join(d_path, "rgb")
            gt_dir = os.path.join(d_path, "gt_final_isaac")
            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
        else:
            rgb_dir = d_path
            gt_dir = d_path
            gt_files = sorted(glob.glob(os.path.join(d_path, "*.json")))

        # --- frame selection: pick top-N frames by R1 detection quality ---
        # quality = (number of detected kps) + 0.01 * spatial spread (pixel std)
        r1_model = r1_map[d_name]
        scored = []  # (score, img_path, gt_path, kps_r0, kps_r1, gt_cuboid)
        for gt_path in gt_files:
            base = os.path.splitext(os.path.basename(gt_path))[0]
            img_path = None
            for ext in [".png", ".jpg"]:
                cand = os.path.join(rgb_dir, base + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is None:
                continue
            img = cv2.imread(img_path)
            with open(gt_path) as f:
                gt = json.load(f)
            gt_cuboid = gt["objects"][0]["projected_cuboid"]

            kps_r1 = infer_image(r1_model, img, device)
            valid = [k for k in kps_r1[:8] if k is not None]
            if len(valid) < 6:
                continue  # need most corners detected
            arr = np.array(valid)
            spread = float(arr.std(axis=0).sum())  # x-std + y-std
            score = len(valid) * 1000 + spread  # primary: count, secondary: spread
            kps_r0 = infer_image(r0, img, device)
            scored.append((score, img, kps_r0, kps_r1, gt_cuboid, base))

        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:n]

        for score, img, kps_r0, kps_r1, gt_cuboid, base in picks:
            n_kp = len([k for k in kps_r1[:8] if k is not None])
            draw_kps(axes[row][0], img, kps_r0, "red",
                     f"{d_name} — R0 (ep65)", gt_kps=gt_cuboid)
            draw_kps(axes[row][1], img, kps_r1, "blue",
                     f"{d_name} — {r1_label[d_name]} ({n_kp}/8 kp detected)",
                     gt_kps=gt_cuboid)
            row += 1

    plt.suptitle("Phase 1 — R0 vs R1 (best) keypoint prediction\n(green: GT cuboid, red: R0, blue: R1)",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
