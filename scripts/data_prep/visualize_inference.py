"""DOPE 추론 결과 시각화 — 합성(val) + 실제(real) 이미지.

Belief map heatmap overlay + detected keypoint + GT keypoint (합성만) +
cuboid wireframe을 그려서 저장.

사용법:
    python scripts/data_prep/visualize_inference.py \
        --weights weights/misc/pallet_category/final_net_epoch_0060.pth \
        --val_dir data/pallet/training_data/val \
        --real_dir data/pallet/real_data \
        --output_dir data/pallet/eval_results/vis \
        --num_syn 10 --num_real 10
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_training"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix

# Keypoint colors (BGR): 0-7 corners + 8 centroid
KP_COLORS = [
    (0, 0, 255),    # 0: red
    (0, 128, 255),  # 1: orange
    (0, 255, 255),  # 2: yellow
    (0, 255, 0),    # 3: green
    (255, 255, 0),  # 4: cyan
    (255, 0, 0),    # 5: blue
    (255, 0, 128),  # 6: purple
    (128, 0, 255),  # 7: magenta
    (255, 255, 255),# 8: centroid (white)
]

# Cuboid edges (pairs of corner indices)
CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # front face
    (4, 5), (5, 6), (6, 7), (7, 4),  # rear face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
]


def load_model(weights_path, device):
    model = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_keypoints(belief_maps, threshold=0.3):
    """Sub-pixel keypoint extraction (DOPE style)."""
    OFFSET = 0.4395
    RAN = 5
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap_ori = belief_maps[i]
        if bmap_ori.max() < threshold:
            keypoints.append(None)
            continue
        bmap_smooth = gaussian_filter(bmap_ori, sigma=2)
        p = 1
        pad_l = np.zeros_like(bmap_smooth); pad_l[p:, :] = bmap_smooth[:-p, :]
        pad_r = np.zeros_like(bmap_smooth); pad_r[:-p, :] = bmap_smooth[p:, :]
        pad_u = np.zeros_like(bmap_smooth); pad_u[:, p:] = bmap_smooth[:, :-p]
        pad_d = np.zeros_like(bmap_smooth); pad_d[:, :-p] = bmap_smooth[:, p:]
        peaks = (
            (bmap_smooth >= pad_l) & (bmap_smooth >= pad_r) &
            (bmap_smooth >= pad_u) & (bmap_smooth >= pad_d) &
            (bmap_smooth > threshold)
        )
        pys, pxs = np.nonzero(peaks)
        if len(pxs) == 0:
            keypoints.append(None)
            continue
        vals = [bmap_ori[py, px] for py, px in zip(pys, pxs)]
        bi = np.argmax(vals)
        px, py = int(pxs[bi]), int(pys[bi])
        y0 = max(0, py - RAN); y1 = min(bmap_ori.shape[0], py + RAN + 1)
        x0 = max(0, px - RAN); x1 = min(bmap_ori.shape[1], px + RAN + 1)
        patch = bmap_ori[y0:y1, x0:x1]
        if patch.sum() > 0:
            ys = np.arange(y0, y1); xs = np.arange(x0, x1)
            xg, yg = np.meshgrid(xs, ys)
            wx = np.average(xg, weights=patch) + OFFSET
            wy = np.average(yg, weights=patch) + OFFSET
        else:
            wx, wy = float(px), float(py)
        keypoints.append((wx, wy, float(bmap_ori.max())))
    return keypoints


def belief_to_heatmap(belief_maps, img_shape):
    """9채널 belief map을 컬러 heatmap으로 합성."""
    combined = belief_maps.max(axis=0)
    combined = np.clip(combined, 0, 1)
    combined = (combined * 255).astype(np.uint8)
    combined = cv2.resize(combined, (img_shape[1], img_shape[0]))
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    return heatmap


def draw_overlay(img, pred_kps, gt_kps_orig, belief_maps, pnp_solver, label):
    """이미지에 heatmap + keypoints + cuboid overlay."""
    h, w = img.shape[:2]
    bh, bw = belief_maps.shape[1], belief_maps.shape[2]
    sx, sy = w / bw, h / bh

    vis = img.copy()

    # Belief heatmap overlay (30% opacity)
    heatmap = belief_to_heatmap(belief_maps, img.shape[:2])
    vis = cv2.addWeighted(vis, 0.7, heatmap, 0.3, 0)

    # Predicted keypoints (원본 해상도로 변환)
    pred_orig = []
    for i, kp in enumerate(pred_kps):
        if kp is None:
            pred_orig.append(None)
            continue
        x_orig = kp[0] * sx
        y_orig = kp[1] * sy
        conf = kp[2] if len(kp) > 2 else 0
        pred_orig.append((x_orig, y_orig))
        pt = (int(x_orig), int(y_orig))
        cv2.circle(vis, pt, 6, KP_COLORS[i], -1)
        cv2.circle(vis, pt, 7, (0, 0, 0), 1)
        cv2.putText(vis, f"{i}", (pt[0]+8, pt[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, KP_COLORS[i], 1)

    # GT keypoints (합성만 — 녹색 십자)
    if gt_kps_orig is not None:
        for i, (gx, gy) in enumerate(gt_kps_orig):
            pt = (int(gx), int(gy))
            cv2.drawMarker(vis, pt, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

    # PnP cuboid wireframe
    success, R, t, _ = pnp_solver.solve(pred_orig)
    if success:
        reproj = pnp_solver.reproject(R, t)
        for i0, i1 in CUBOID_EDGES:
            p0 = tuple(reproj[i0].astype(int))
            p1 = tuple(reproj[i1].astype(int))
            cv2.line(vis, p0, p1, (0, 255, 255), 2)

    # 정보 텍스트
    detected = sum(1 for kp in pred_kps if kp is not None)
    info = f"{label} | {detected}/9 kps | PnP: {'OK' if success else 'FAIL'}"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return vis


def aspect_resize(img, target=400):
    """비율유지 resize: 짧은 변=target, 긴 변=비율 그대로(8의 배수). DOPE는 fully-conv라 정사각 불필요.
    run_dope_live의 400/h는 가로영상(640×480) 전용 — 세로(portrait)면 너비가 짜부됨. 일반화하면
    short-side→target 라야 가로/세로 모두 객체 크기 유지. belief→원본 역매핑은 belief.shape 기반이라 그대로 성립."""
    h, w = img.shape[:2]
    s = target / float(min(h, w))
    nw = max(8, int(round(w * s)) & ~7)
    nh = max(8, int(round(h * s)) & ~7)
    return cv2.resize(img, (nw, nh))


def infer(model, img_bgr, device):
    """DOPE 추론 → belief maps."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = aspect_resize(img_rgb, 400)  # 비율유지(실배포 방식), squash 금지
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    return out_bel[-1][0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--val_dir", default="data/pallet/training_data/val")
    parser.add_argument("--real_dir", default="data/pallet/real_data")
    parser.add_argument("--output_dir", default="data/pallet/eval_results/vis")
    parser.add_argument("--num_syn", type=int, default=10)
    parser.add_argument("--num_real", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    print(f"Model loaded: {args.weights}")

    # === 합성 Val 이미지 ===
    syn_pngs = sorted(glob.glob(os.path.join(args.val_dir, "*.png")))
    # 균등 샘플링
    if len(syn_pngs) > args.num_syn:
        indices = np.linspace(0, len(syn_pngs)-1, args.num_syn, dtype=int)
        syn_pngs = [syn_pngs[i] for i in indices]

    print(f"\n=== Synthetic Val ({len(syn_pngs)} images) ===")
    for i, png_path in enumerate(syn_pngs):
        img = cv2.imread(png_path)
        belief = infer(model, img, device)
        pred_kps = extract_keypoints(belief, args.threshold)

        # GT 로드
        basename = os.path.splitext(os.path.basename(png_path))[0]
        json_path = os.path.join(args.val_dir, basename + ".json")
        gt_kps = None
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            obj = data["objects"][0]
            gt_kps = obj["projected_cuboid"] + [obj["projected_cuboid_centroid"]]

        vis = draw_overlay(img, pred_kps, gt_kps, belief, pnp, f"SYN {basename}")
        out_path = os.path.join(args.output_dir, f"syn_{i:02d}_{basename}.jpg")
        cv2.imwrite(out_path, vis)
        detected = sum(1 for kp in pred_kps if kp is not None)
        print(f"  [{i+1}] {basename}: {detected}/9 kps → {out_path}")

    # === Real 이미지 ===
    real_imgs = sorted(glob.glob(os.path.join(args.real_dir, "*.jpg")))
    if not real_imgs:
        real_imgs = sorted(glob.glob(os.path.join(args.real_dir, "*.png")))
    if len(real_imgs) > args.num_real:
        indices = np.linspace(0, len(real_imgs)-1, args.num_real, dtype=int)
        real_imgs = [real_imgs[i] for i in indices]

    print(f"\n=== Real Images ({len(real_imgs)} images) ===")
    for i, img_path in enumerate(real_imgs):
        img = cv2.imread(img_path)
        belief = infer(model, img, device)
        pred_kps = extract_keypoints(belief, args.threshold)

        basename = os.path.splitext(os.path.basename(img_path))[0]
        vis = draw_overlay(img, pred_kps, None, belief, pnp, f"REAL {basename}")
        out_path = os.path.join(args.output_dir, f"real_{i:02d}_{basename}.jpg")
        cv2.imwrite(out_path, vis)
        detected = sum(1 for kp in pred_kps if kp is not None)
        print(f"  [{i+1}] {basename}: {detected}/9 kps → {out_path}")

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
