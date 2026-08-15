"""단일 이미지에 대한 DOPE heatmap overlay 생성.

사용법:
    python scripts/data_prep/visualize/dope_single_image.py \
        --weights weights/misc/f5_noapril_ransac_loo_realonly/final_net_epoch_0096.pth \
        --image data/pallet/raw_data/real_data/color_20250908_165311_662.jpg \
        --out_dir _docs/figures
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

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
    RAN = 5
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--tag", default="realdata")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    img_bgr = cv2.imread(args.image)
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    resized = cv2.resize(rgb, (448, 448))
    norm = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()  # (9, 50, 50)

    kps = extract_peaks(belief, args.threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w, bh / h
    kps_img = [(k[0] / sx, k[1] / sy) if k is not None else None for k in kps]
    n_det = sum(1 for k in kps if k is not None)
    print(f"Detected {n_det}/9 keypoints, image {w}x{h}")

    # RGB 원본 저장
    out_rgb = os.path.join(args.out_dir, f"dope_{args.tag}_rgb.png")
    cv2.imwrite(out_rgb, img_bgr)
    print(f"저장: {out_rgb}")

    # Heatmap overlay
    combined = belief.max(axis=0)
    combined = np.clip(combined, 0, None)
    if combined.max() > 0:
        combined = combined / combined.max()
    heat_large = cv2.resize(combined, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap((heat_large * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    mask = heat_large[..., None]
    blended = rgb.astype(np.float32) * (1 - mask * args.alpha) + heat_color.astype(np.float32) * (mask * args.alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out_heat = os.path.join(args.out_dir, f"dope_{args.tag}_heatmap.png")
    cv2.imwrite(out_heat, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # 추가: 작은 균일 색상 점이 찍힌 버전
    overlay = blended.copy()
    dot_color = (255, 255, 255)  # white
    for kp in kps_img[:9]:
        if kp is None:
            continue
        cx, cy = int(kp[0]), int(kp[1])
        cv2.circle(overlay, (cx, cy), 3, dot_color, -1)
        cv2.circle(overlay, (cx, cy), 3, (20, 20, 20), 1)
    out_dots = os.path.join(args.out_dir, f"dope_{args.tag}_heatmap_dots.png")
    cv2.imwrite(out_dots, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"저장: {out_dots}")
    print(f"저장: {out_heat}")


if __name__ == "__main__":
    main()
