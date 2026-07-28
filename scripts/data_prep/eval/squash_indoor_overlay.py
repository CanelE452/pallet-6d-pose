"""squash_indoor_overlay.py — side-by-side indoor overlay: paper_base(squash)
vs dope_cropaug_pretrain(no-squash), same frame, for visual corner inspection.

Left = paper_base(squash), Right = pretrain(no-squash).
GREEN dots + index 0..8 = prediction (cyan cuboid edges), MAGENTA = GT cuboid.
Badge = order-free Hungarian 8-corner mean reproj px.
Picks frames where BOTH models detect >=6 corners. Inference only.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, HERE)
from filter_pr_camfacing import extract_keypoints_from_belief, load_model  # noqa

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
GREEN = (0, 255, 0); CYAN = (255, 255, 0); MAGENTA = (255, 0, 255)
THRESHOLD = 0.3
GT_DIR = os.path.join(ROOT, "data", "pallet", "raw_data", "capture0403middle", "gt_final")
RGB_DIR = os.path.join(ROOT, "data", "pallet", "raw_data", "capture0403middle", "rgb")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "squash_vs_nosquash", "indoor_overlay")
MODELS = {
    "squash": "weights/paper_base/final_net_epoch_0060.pth",
    "nosquash": "weights/dope_cropaug_pretrain/final_net_epoch_0060.pth",
}
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
N_SAVE = 16


def infer(model, device, img):
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = ((cv2.resize(rgb, (448, 448)).astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps = extract_keypoints_from_belief(belief, THRESHOLD)
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w0, bh / h0
    return [None if k[0] < 0 else (k[0] / sx, k[1] / sy) for k in kps]


def hung8(kp, gt8):
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    if valid.sum() < 6:
        return float("inf"), int(valid.sum())
    P = pred8[valid]
    cost = np.linalg.norm(P[:, None] - gt8[None], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].mean()), int(valid.sum())


def draw(img, kp, gt8, title, err, nval):
    im = img.copy()
    for a, b in EDGES:
        if int(gt8.shape[0]) > max(a, b):
            cv2.line(im, tuple(gt8[a].astype(int)), tuple(gt8[b].astype(int)), MAGENTA, 2)
    pts = [None if k is None else np.asarray(k, float) for k in kp[:8]]
    for a, b in EDGES:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(im, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), CYAN, 1)
    for i in range(9):
        if kp[i] is None:
            continue
        p = tuple(np.asarray(kp[i], float).astype(int))
        cv2.circle(im, p, 4, GREEN, -1)
        cv2.putText(im, str(i), (p[0] + 4, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, f"{title}  reproj8={err:.1f}px  ncorner={nval}",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return im


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {k: load_model(os.path.join(ROOT, v), device) for k, v in MODELS.items()}
    files = sorted(glob.glob(os.path.join(GT_DIR, "*.json")))
    rows = []
    for jp in files:
        base = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(RGB_DIR, base + ".png")
        if not os.path.exists(ip):
            continue
        gt = json.load(open(jp))
        gt8 = np.array(gt["objects"][0]["projected_cuboid"], float)[:8]
        img = cv2.imread(ip)
        kp_s = infer(models["squash"], device, img)
        kp_n = infer(models["nosquash"], device, img)
        e_s, n_s = hung8(kp_s, gt8)
        e_n, n_n = hung8(kp_n, gt8)
        if n_s < 6 or n_n < 6:
            continue
        rows.append((base, ip, gt8, kp_s, kp_n, e_s, n_s, e_n, n_n))
    # sort by where models disagree most (|e_n - e_s| desc) to highlight squash effect
    rows.sort(key=lambda r: abs(r[7] - r[5]), reverse=True)
    print(f"[indoor] {len(rows)} frames both-detected; saving {min(N_SAVE, len(rows))}")
    saved = []
    for base, ip, gt8, kp_s, kp_n, e_s, n_s, e_n, n_n in rows[:N_SAVE]:
        img = cv2.imread(ip)
        L = draw(img, kp_s, gt8, "squash", e_s, n_s)
        R = draw(img, kp_n, gt8, "no-squash", e_n, n_n)
        combo = np.hstack([L, R])
        op = os.path.join(OUT_DIR, f"{base}_d{abs(e_n-e_s):05.1f}.jpg")
        cv2.imwrite(op, combo)
        saved.append(op)
    # contact sheet
    if saved:
        thumbs = [cv2.resize(cv2.imread(p), (640, 240)) for p in saved[:12]]
        while len(thumbs) % 3:
            thumbs.append(np.zeros((240, 640, 3), np.uint8))
        grid = np.vstack([np.hstack(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)])
        cs = os.path.join(OUT_DIR, "_contact_sheet.jpg")
        cv2.imwrite(cs, grid)
        print(f"[save] contact sheet: {cs}")
    print(f"[save dir] {OUT_DIR}")


if __name__ == "__main__":
    main()
