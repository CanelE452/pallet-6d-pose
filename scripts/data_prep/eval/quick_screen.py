"""Quick screening of a DOPE checkpoint: ~2 min pass/fail signal.

하루에 많은 ablation 스크리닝용. 빠른 judgement를 위해
  - synthetic val 20장 (PnP rate + avg detected kp)
  - noapril 20장 (PnP rate + avg detected kp, real domain)
만 측정. 전체 188장 / val 전체는 final eval에서 돌릴 것.

사용법:
    python scripts/data_prep/eval/quick_screen.py \
        --weights weights/v9_ablation_B2/final_net_epoch_0065.pth \
        --tag v8_B2
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "self_training"))

from models import DopeNetwork
from pnp_solver import PalletPnPSolver, make_camera_matrix, make_pallet_keypoints_3d


def load_model(weights_path, device):
    net = DopeNetwork()
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    net.to(device).eval()
    return net


def extract_keypoints(belief_maps, threshold=0.3):
    OFFSET = 0.4395
    WIN = 11
    RAN = WIN // 2
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap = belief_maps[i]
        mx = bmap.max()
        if mx < threshold:
            keypoints.append((-1.0, -1.0, float(mx)))
            continue
        sm = gaussian_filter(bmap, sigma=2)
        pl = np.zeros_like(sm); pl[1:, :] = sm[:-1, :]
        pr = np.zeros_like(sm); pr[:-1, :] = sm[1:, :]
        pu = np.zeros_like(sm); pu[:, 1:] = sm[:, :-1]
        pd = np.zeros_like(sm); pd[:, :-1] = sm[:, 1:]
        peaks = (sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) & (sm > threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            keypoints.append((-1.0, -1.0, float(mx)))
            continue
        vals = [bmap[yy, xx] for yy, xx in zip(ys, xs)]
        bi = int(np.argmax(vals))
        px, py = int(xs[bi]), int(ys[bi])
        yl = max(0, py - RAN); yh = min(bmap.shape[0], py + RAN + 1)
        xl = max(0, px - RAN); xh = min(bmap.shape[1], px + RAN + 1)
        patch = bmap[yl:yh, xl:xh]
        if patch.sum() > 0:
            yg, xg = np.meshgrid(np.arange(yl, yh), np.arange(xl, xh), indexing="ij")
            wx = float(np.average(xg, weights=patch)) + OFFSET
            wy = float(np.average(yg, weights=patch)) + OFFSET
        else:
            wx, wy = float(px), float(py)
        keypoints.append((wx, wy, float(mx)))
    return keypoints


def run_inference(net, img_bgr, device, threshold):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (400, 400))  # 학습 입력(400×400)과 일치
    normed = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        ob, _ = net(tensor)
    belief = ob[-1][0].cpu().numpy()
    kps_bel = extract_keypoints(belief, threshold)
    # scale to original resolution
    h, w = img_bgr.shape[:2]
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w, bh / h
    kps_orig = []
    for kp in kps_bel:
        if kp[0] < 0:
            kps_orig.append(None)
        else:
            kps_orig.append((kp[0] / sx, kp[1] / sy))
    return belief, kps_bel, kps_orig


def eval_synthetic(net, val_dir, pnp_solver, device, n=20, threshold=0.3):
    jsons = sorted(glob.glob(os.path.join(val_dir, "*.json")))[:n]
    pnp_ok = 0
    det_counts = []
    pck5 = []
    for jp in jsons:
        base = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(val_dir, base + ".png")
        if not os.path.exists(ip):
            continue
        img = cv2.imread(ip)
        with open(jp) as f:
            gt = json.load(f)
        if not gt.get("objects"):
            continue
        gt_kps = gt["objects"][0]["projected_cuboid"] + [gt["objects"][0]["projected_cuboid_centroid"]]
        gt_kps = np.array(gt_kps, dtype=np.float64)  # (9, 2)

        _, _, pred_orig = run_inference(net, img, device, threshold)
        det_counts.append(sum(1 for k in pred_orig if k is not None))

        # PnP
        success, _, _, _ = pnp_solver.solve(pred_orig)
        if success:
            pnp_ok += 1

        # PCK@5px on detected kps only
        hits = 0; total = 0
        for i in range(9):
            if pred_orig[i] is None:
                continue
            d = np.linalg.norm(np.array(pred_orig[i]) - gt_kps[i])
            total += 1
            if d < 5.0:
                hits += 1
        if total > 0:
            pck5.append(hits / total)
    n_run = len(det_counts)
    return {
        "n": n_run,
        "pnp_rate": pnp_ok / max(n_run, 1),
        "avg_det": float(np.mean(det_counts)) if det_counts else 0.0,
        "pck5": float(np.mean(pck5)) if pck5 else 0.0,
    }


def eval_real_noapril(net, rgb_dir, filelist_path, pnp_solver, device, n=20, threshold=0.3):
    with open(filelist_path) as f:
        names = [ln.strip() for ln in f if ln.strip()][:n]
    pnp_ok = 0
    det_counts = []
    for nm in names:
        p = os.path.join(rgb_dir, nm)
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        _, _, pred_orig = run_inference(net, img, device, threshold)
        det_counts.append(sum(1 for k in pred_orig if k is not None))
        success, _, _, _ = pnp_solver.solve(pred_orig)
        if success:
            pnp_ok += 1
    n_run = len(det_counts)
    return {
        "n": n_run,
        "pnp_rate": pnp_ok / max(n_run, 1),
        "avg_det": float(np.mean(det_counts)) if det_counts else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick screening of DOPE checkpoint")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tag", default="ckpt")
    parser.add_argument("--val_dir", default=os.path.join(ROOT, "data/pallet/training_data/val"))
    parser.add_argument("--noapril_rgb", default=os.path.join(ROOT, "data/pallet/raw_data/capture0403noapril/rgb"))
    parser.add_argument("--noapril_filelist",
                        default=os.path.join(ROOT, "data/pallet/raw_data/capture0403noapril/noapril_eval/filelist.txt"))
    parser.add_argument("--n_syn", type=int, default=20)
    parser.add_argument("--n_real", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--fx", type=float, default=614.18)
    parser.add_argument("--fy", type=float, default=614.31)
    parser.add_argument("--cx", type=float, default=329.28)
    parser.add_argument("--cy", type=float, default=234.53)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[quick_screen] weights = {args.weights}")
    net = load_model(args.weights, device)
    cam = make_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    pnp = PalletPnPSolver(cam)

    print(f"[quick_screen] synthetic val {args.n_syn}...")
    syn = eval_synthetic(net, args.val_dir, pnp, device, args.n_syn, args.threshold)

    print(f"[quick_screen] noapril real {args.n_real}...")
    real = eval_real_noapril(net, args.noapril_rgb, args.noapril_filelist, pnp, device,
                             args.n_real, args.threshold)

    print()
    print("=" * 60)
    print(f" Quick Screen: {args.tag}")
    print("=" * 60)
    print(f"  Synthetic val ({syn['n']} imgs):")
    print(f"    PnP rate : {syn['pnp_rate']*100:6.1f}%")
    print(f"    Avg kp   : {syn['avg_det']:5.2f} / 9")
    print(f"    PCK@5px  : {syn['pck5']*100:6.1f}%")
    print(f"  noapril real ({real['n']} imgs):")
    print(f"    PnP rate : {real['pnp_rate']*100:6.1f}%")
    print(f"    Avg kp   : {real['avg_det']:5.2f} / 9")
    print("=" * 60)

    return {"tag": args.tag, "weights": args.weights, "syn": syn, "real": real}


if __name__ == "__main__":
    res = main()
