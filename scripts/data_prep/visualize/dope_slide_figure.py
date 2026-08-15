"""DOPE 발표 슬라이드용 이미지 2장 생성.

1) 팔레트 RGB 이미지 (원본)
2) Belief map heatmap overlay + cuboid edges

자동으로 여러 프레임을 추론해서 '잘된' 프레임을 선택 (NN matching error 최소).

사용법:
    python scripts/data_prep/visualize/dope_slide_figure.py \
        --weights weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth \
        --test_dir data/pallet/raw_data/capture0403middle \
        --out_dir _docs/figures \
        --topk 3
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
    model.to(device).eval()
    return model


def extract_peaks(belief_maps, threshold=0.3):
    OFFSET = 0.4395
    WIN = 11
    RAN = WIN // 2
    kps = []
    for i in range(belief_maps.shape[0]):
        bmap = belief_maps[i]
        if bmap.max() < threshold:
            kps.append(None); continue
        sm = gaussian_filter(bmap, sigma=2)
        p = 1
        pl = np.zeros_like(sm); pl[p:, :] = sm[:-p, :]
        pr = np.zeros_like(sm); pr[:-p, :] = sm[p:, :]
        pu = np.zeros_like(sm); pu[:, p:] = sm[:, :-p]
        pd = np.zeros_like(sm); pd[:, :-p] = sm[:, p:]
        peaks = (sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) & (sm > threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            kps.append(None); continue
        vals = [bmap[y, x] for y, x in zip(ys, xs)]
        bi = int(np.argmax(vals))
        px, py = int(xs[bi]), int(ys[bi])
        y_lo, y_hi = max(0, py - RAN), min(bmap.shape[0], py + RAN + 1)
        x_lo, x_hi = max(0, px - RAN), min(bmap.shape[1], px + RAN + 1)
        patch = bmap[y_lo:y_hi, x_lo:x_hi]
        if patch.sum() > 0:
            yy = np.arange(y_lo, y_hi); xx = np.arange(x_lo, x_hi)
            xg, yg = np.meshgrid(xx, yy)
            wx = np.average(xg, weights=patch) + OFFSET
            wy = np.average(yg, weights=patch) + OFFSET
        else:
            wx, wy = float(px), float(py)
        kps.append((wx, wy))
    return kps


def nn_error(pred_kps, gt_cuboid):
    vp = [(i, k) for i, k in enumerate(pred_kps[:8]) if k is not None]
    if len(vp) < 8:
        return 1e9
    pred_arr = np.array([k for _, k in vp])
    gt_arr = np.array(gt_cuboid[:8])
    cost = np.linalg.norm(pred_arr[:, None, :] - gt_arr[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    return cost[row, col].mean()


def infer_one(model, img_bgr, device, threshold=0.3):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    resized = cv2.resize(rgb, (448, 448))
    norm = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps = extract_peaks(belief, threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w, bh / h
    kps_img = [(k[0] / sx, k[1] / sy) if k is not None else None for k in kps]
    return rgb, belief, kps_img


def render_heatmap_overlay(rgb, belief, kps_img, alpha=0.55):
    h, w = rgb.shape[:2]
    # combine 9 belief maps (max across channels) to single 50x50 then upsample
    combined = belief.max(axis=0)  # (50, 50)
    combined = np.clip(combined, 0, None)
    if combined.max() > 0:
        combined = combined / combined.max()
    heat_large = cv2.resize(combined, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap((heat_large * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    # blend weighted by heat intensity (transparent where heat is low)
    mask = heat_large[..., None]  # (H, W, 1)
    blended = rgb.astype(np.float32) * (1 - mask * alpha) + heat_color.astype(np.float32) * (mask * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 9 keypoint colors (RGB): 8 corners + center
    kp_colors = [
        (230,  30,  30),  # C1 red
        ( 30, 200,  30),  # C2 green
        ( 30, 200, 230),  # C3 cyan
        ( 30,  80, 230),  # C4 blue
        (180,  40, 220),  # C5 purple
        (240, 220,  40),  # C6 yellow
        (245, 130,  30),  # C7 orange
        (235,  80, 170),  # C8 pink
        (255, 255, 255),  # center white
    ]
    overlay = blended.copy()
    for i, kp in enumerate(kps_img[:9]):
        if kp is None:
            continue
        col = kp_colors[i]
        cx, cy = int(kp[0]), int(kp[1])
        cv2.circle(overlay, (cx, cy), 9, col, -1)
        cv2.circle(overlay, (cx, cy), 9, (20, 20, 20), 2)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=3, help="저장할 best 프레임 수")
    ap.add_argument("--max_scan", type=int, default=440, help="스캔할 프레임 수")
    args = ap.parse_args()

    if args.gt_dir is None:
        args.gt_dir = os.path.join(args.test_dir, "gt_final_isaac")
    os.makedirs(args.out_dir, exist_ok=True)

    rgb_dir = os.path.join(args.test_dir, "rgb")
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.json")))[: args.max_scan]
    print(f"Scanning {len(gt_files)} frames...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    scored = []  # (err, gt_path, img_path)
    for gi, gt_path in enumerate(gt_files):
        base = os.path.splitext(os.path.basename(gt_path))[0]
        img_path = None
        for ext in [".png", ".jpg"]:
            cand = os.path.join(rgb_dir, base + ext)
            if os.path.exists(cand):
                img_path = cand; break
        if img_path is None:
            continue
        with open(gt_path) as f:
            gt_data = json.load(f)
        gt_cuboid = gt_data["objects"][0]["projected_cuboid"]
        img_bgr = cv2.imread(img_path)
        _, belief, kps_img = infer_one(model, img_bgr, device, args.threshold)
        err = nn_error(kps_img, gt_cuboid)
        scored.append((err, gt_path, img_path))
        if (gi + 1) % 50 == 0:
            print(f"  [{gi+1}/{len(gt_files)}]")

    scored.sort(key=lambda x: x[0])
    print(f"\nTop-{args.topk} best frames (lowest NN error):")

    for rank, (err, gt_path, img_path) in enumerate(scored[: args.topk]):
        base = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  #{rank+1}  err={err:.2f}px  {base}")

        img_bgr = cv2.imread(img_path)
        rgb, belief, kps_img = infer_one(model, img_bgr, device, args.threshold)

        # 1) RGB 원본
        out_rgb = os.path.join(args.out_dir, f"dope_rank{rank+1}_rgb.png")
        cv2.imwrite(out_rgb, img_bgr)
        print(f"    저장: {out_rgb}")

        # 2) Heatmap overlay + cuboid
        overlay = render_heatmap_overlay(rgb, belief, kps_img)
        out_heat = os.path.join(args.out_dir, f"dope_rank{rank+1}_heatmap.png")
        cv2.imwrite(out_heat, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"    저장: {out_heat}")


if __name__ == "__main__":
    main()
