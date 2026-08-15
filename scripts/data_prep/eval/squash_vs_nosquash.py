"""squash_vs_nosquash.py — does squash aug collapse indoor corners?

Objective: paper_base (mixed_v8 + squash + scale + trunc) corners
collapse/shrink in indoor front views. Hypothesis: squash aug hurt corner
precision (esp. indoor frontal depth). Compare against dope_cropaug_pretrain
(mixed_v8 + trunc, NO squash, NO scale) — the "old paper model".

Judge metric (memory: filter-goal / evaluate-on-val-convention-bug):
  order-free 9kp reprojection error.
    * 8 corners: Hungarian-matched to GT projected_cuboid[:8]
    * centroid (kp 8): matched to GT projected_cuboid_centroid
  good% = frames with 8-corner Hungarian mean < 10px.
  Per-keypoint group err = mean dist of preds whose Hungarian-matched
  GT index falls in front(0-3) / back(4-7); centroid separately.
  (GT = object-frame canonical: 0-3 front face, 4-7 back face, 8 centroid.)

NO training. Inference only. Same conditions for both models.
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

from filter_pr_camfacing import (  # noqa: E402
    extract_keypoints_from_belief,
    load_model,
)

EXCLUDE_FILE = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
GOOD_PX = 10.0
THRESHOLD = 0.3

MODELS = {
    "paper_base(squash)": "weights/paper_base/paper_base/final_net_epoch_0060.pth",
    "pretrain(no-squash)": "weights/dope/dope_cropaug_pretrain/final_net_epoch_0060.pth",
}

SETS = {
    "indoor": (os.path.join(ROOT, "data", "pallet", "raw_data",
                            "capture0403middle", "gt_final"),
               os.path.join(ROOT, "data", "pallet", "raw_data",
                            "capture0403middle", "rgb")),
    "outside": (os.path.join(ROOT, "data", "_eval_sets", "outside_combined"), None),
    "night": (os.path.join(ROOT, "data", "_eval_sets", "night_combined"), None),
}


def load_exclude():
    ex = set()
    if os.path.exists(EXCLUDE_FILE):
        for ln in open(EXCLUDE_FILE):
            ln = ln.split("#")[0].strip()
            if ln:
                ex.add(ln)
    return ex


def infer_frame(model, device, mean, std, img):
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = ((cv2.resize(rgb, (448, 448)).astype(np.float32) / 255.0 - mean) / std)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps = extract_keypoints_from_belief(belief, THRESHOLD)
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w0, bh / h0
    kp = []
    for k in kps:
        kp.append(None if k[0] < 0 else (k[0] / sx, k[1] / sy))
    return kp


def hungarian_assign(pred8, gt8):
    """Return (mean8, matches) where matches = list of (pred_idx, gt_idx, dist).
    pred8 (8,2) NaN-missing; gt8 (8,2). None if <6 valid."""
    valid = np.where(~np.isnan(pred8[:, 0]))[0]
    if len(valid) < 6:
        return None, None
    P = pred8[valid]
    cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    matches = [(int(valid[r]), int(c), float(cost[r, c])) for r, c in zip(ri, ci)]
    return float(np.mean([m[2] for m in matches])), matches


def eval_model_on_set(model, device, mean, std, gt_dir, img_dir, exclude):
    idir = img_dir or gt_dir
    files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    n_total = n_det = n_good = 0
    means8, means9 = [], []
    front_d, back_d, ctr_d = [], [], []  # per-corner distances pooled
    for jp in files:
        base = os.path.splitext(os.path.basename(jp))[0]
        if base in exclude:
            continue
        ip = None
        for ext in (".png", ".jpg"):
            c = os.path.join(idir, base + ext)
            if os.path.exists(c):
                ip = c
                break
        if ip is None:
            continue
        gt = json.load(open(jp))
        obj = gt["objects"][0]
        gt8 = np.array(obj["projected_cuboid"], float)[:8]
        gt_c = obj.get("projected_cuboid_centroid")
        n_total += 1

        img = cv2.imread(ip)
        kp = infer_frame(model, device, mean, std, img)
        pred8 = np.full((8, 2), np.nan)
        for i in range(8):
            if kp[i] is not None:
                pred8[i] = kp[i]
        if int((~np.isnan(pred8[:, 0])).sum()) < 6:
            continue
        m8, matches = hungarian_assign(pred8, gt8)
        if m8 is None:
            continue
        n_det += 1
        means8.append(m8)
        if m8 < GOOD_PX:
            n_good += 1
        for _, gi, d in matches:
            (front_d if gi < 4 else back_d).append(d)
        if kp[8] is not None and gt_c is not None:
            dc = float(np.linalg.norm(np.asarray(kp[8], float) -
                                      np.asarray(gt_c, float)))
            ctr_d.append(dc)
            means9.append((m8 * 8 + dc) / 9.0)

    def med(a):
        return float(np.median(a)) if len(a) else float("nan")

    means8 = np.array(means8)
    means9 = np.array(means9)
    return {
        "n_total": n_total, "n_det": n_det,
        "det_rate": n_det / n_total if n_total else 0.0,
        "reproj9_median": med(means9), "reproj9_mean":
            float(means9.mean()) if len(means9) else float("nan"),
        "reproj8_median": med(means8),
        "good_pct": n_good / n_det if n_det else 0.0, "n_good": n_good,
        "front_med": med(front_d), "back_med": med(back_d), "ctr_med": med(ctr_d),
        "front_mean": float(np.mean(front_d)) if front_d else float("nan"),
        "back_mean": float(np.mean(back_d)) if back_d else float("nan"),
        "ctr_mean": float(np.mean(ctr_d)) if ctr_d else float("nan"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    exclude = load_exclude()
    print(f"[device] {device}  [exclude] {exclude}")

    results = {}
    for mname, wpath in MODELS.items():
        print(f"\n[load] {mname}: {wpath}")
        model = load_model(os.path.join(ROOT, wpath), device)
        results[mname] = {}
        for sname, (gt_dir, img_dir) in SETS.items():
            r = eval_model_on_set(model, device, mean, std, gt_dir, img_dir, exclude)
            results[mname][sname] = r
            print(f"  [{mname} x {sname}] n={r['n_total']} det={r['det_rate']*100:5.1f}% "
                  f"9kp_med={r['reproj9_median']:.2f} good={r['good_pct']*100:.1f}% "
                  f"F/B/C={r['front_med']:.1f}/{r['back_med']:.1f}/{r['ctr_med']:.1f}")

    out = os.path.join(HERE, "squash_vs_nosquash_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[save] {out}")

    print("\n" + "=" * 92)
    print("SQUASH vs NO-SQUASH  (order-free 9kp reproj; F/B/C = front/back/centroid median px)")
    print("=" * 92)
    hdr = (f"{'model':<22}{'domain':<9}{'n':>5}{'det%':>7}"
           f"{'9kp_med':>9}{'9kp_mn':>8}{'good%':>7}"
           f"{'front':>8}{'back':>8}{'ctr':>8}")
    print(hdr)
    print("-" * len(hdr))
    for mname in MODELS:
        for sname in SETS:
            r = results[mname][sname]
            print(f"{mname:<22}{sname:<9}{r['n_total']:>5}{r['det_rate']*100:>7.1f}"
                  f"{r['reproj9_median']:>9.2f}{r['reproj9_mean']:>8.2f}"
                  f"{r['good_pct']*100:>7.1f}"
                  f"{r['front_med']:>8.1f}{r['back_med']:>8.1f}{r['ctr_med']:>8.1f}")
    return results


if __name__ == "__main__":
    main()
