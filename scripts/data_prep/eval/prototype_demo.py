"""Phase 1 프로토타입 데모 panel —
이미지 → keypoint detection → PnP → 6D cuboid overlay (1 cycle).

R1_outside_loo 모델로 outside cp09 시퀀스 일부 frame 추론 → wireframe overlay.

사용:
    python scripts/data_prep/eval/prototype_demo.py \\
        --output _docs/figures/phase1_prototype_demo.png
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
sys.path.insert(0, os.path.join(_ROOT, "scripts", "self_training"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix, make_pallet_keypoints_3d_isaac


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def extract_kps(belief, threshold=0.3):
    OFFSET = 0.4395
    kps = []
    for i in range(belief.shape[0]):
        bmap = belief[i]
        if bmap.max() < threshold:
            kps.append(None); continue
        smooth = gaussian_filter(bmap, sigma=2)
        ys, xs = np.unravel_index(smooth.argmax(), smooth.shape)
        win = 5; half = win // 2
        y0 = max(0, ys - half); y1 = min(bmap.shape[0], ys + half + 1)
        x0 = max(0, xs - half); x1 = min(bmap.shape[1], xs + half + 1)
        patch = bmap[y0:y1, x0:x1]
        if patch.sum() < 1e-6:
            kps.append((float(xs), float(ys))); continue
        yg, xg = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        kps.append((float(np.average(xg, weights=patch) + OFFSET),
                    float(np.average(yg, weights=patch) + OFFSET)))
    return kps


def infer(model, img_bgr, device):
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448))
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = w0 / bw, h0 / bh
    raw = extract_kps(belief)
    kps = []
    for kp in raw:
        if kp is None:
            kps.append(None)
        else:
            kps.append((kp[0] * sx, kp[1] * sy))
    return kps


def draw_panel(ax, img, kps, R, t, cam_matrix, kp3d, title):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 1) keypoint dots
    for i, kp in enumerate(kps[:8]):
        if kp is None:
            continue
        ax.scatter(kp[0], kp[1], c="cyan", s=30, edgecolor="black", linewidth=0.5, zorder=5)

    # 2) PnP cuboid wireframe (use 3D cuboid → reproject)
    if R is not None and t is not None:
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(kp3d[:8], rvec, t.reshape(3, 1),
                                     cam_matrix, np.zeros(4))
        proj = proj.reshape(-1, 2)
        for (a, b) in CUBOID_EDGES:
            pa, pb = proj[a], proj[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="yellow", linewidth=2, alpha=0.85)
        # centroid (mean of 8 corners)
        cx = proj[:, 0].mean(); cy = proj[:, 1].mean()
        ax.scatter(cx, cy, c="red", s=80, marker="x", linewidth=2.5, zorder=6)
        # depth annotation
        dist = float(np.linalg.norm(t))
        ax.text(10, 30, f"depth: {dist:.2f} m", color="white", fontsize=10,
                bbox=dict(facecolor="black", alpha=0.6, pad=2))

    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default="weights/selftrain/r1_outside_loo/final_net_epoch_0096.pth")
    p.add_argument("--n_frames", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    # outside cp09 시퀀스에서 균등 간격 N frame
    rgb_dir = "data/outside/capturepallet09/rgb"
    all_imgs = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    step = max(1, len(all_imgs) // (args.n_frames + 1))
    picks = [all_imgs[step * (i + 1)] for i in range(args.n_frames)]

    fx = 605.91; fy = 605.97; cx = 317.60; cy = 256.29
    cam_matrix = make_camera_matrix(fx, fy, cx, cy)
    kp3d = make_pallet_keypoints_3d_isaac(1.1, 1.3, 0.11)
    solver = PalletPnPSolver(cam_matrix, keypoints_3d=kp3d)

    cols = 3
    rows = (args.n_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols) if rows > 1 else np.array([axes])

    for i, img_path in enumerate(picks):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        img = cv2.imread(img_path)
        kps = infer(model, img, device)
        ok, R, t, _ = solver.solve(kps)
        base = os.path.basename(img_path)
        title = f"{base[:12]}..  {'OK' if R is not None else 'PnP fail'}"
        draw_panel(ax, img, kps, R, t, cam_matrix, kp3d, title)

    # remove empty axes
    for i in range(args.n_frames, rows * cols):
        r, c = divmod(i, cols)
        fig.delaxes(axes[r][c])

    fig.suptitle("Phase 1 — Prototype demo cycle\n"
                 "image → DOPE keypoint (cyan) → EPnP+RANSAC → cuboid overlay (yellow) + depth\n"
                 "model: R1_outside_loo  scene: outside/capturepallet09",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
