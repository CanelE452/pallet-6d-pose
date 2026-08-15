"""c1_r0_vs_r1.py — C1 self-training round R0 vs R1 evaluation matrix.

Objective (paper C1): did diag / diag&ratio pseudo-label finetuning (R1)
improve REAL-domain accuracy over R0 (paper_base)?

Judge metric (per CLAUDE.md memory filter-goal-reliable-pl-full-keypoints,
evaluate-on-val-convention-bug):
    - detection rate (>=6 of 8 corners detected)
    - full 9-keypoint reprojection error, ORDER-FREE:
        * 8 cuboid corners: Hungarian-matched to GT projected_cuboid 8 corners
        * centroid(kp 8): matched to GT projected_cuboid_centroid
      mean over the 9 matched distances per frame (frames with >=6 corners).
    - good% = frames with 8-corner Hungarian mean < 10px.

NO training. Inference only. Reuses belief extraction + Hungarian from
filter_pr_camfacing.py.
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

from models import DopeNetwork  # noqa: E402
from filter_pr_camfacing import (  # noqa: E402
    extract_keypoints_from_belief,
    load_model,
    hungarian_mean_dist,
)

EXCLUDE_FILE = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
GOOD_PX = 10.0
THRESHOLD = 0.3

MODELS = {
    "R0_base": "weights/paper_base/paper_base/final_net_epoch_0060.pth",
    "R1_outside": "weights/paper_s1/paper_r1_outside/final_net_epoch_0091.pth",
    "R1_night": "weights/paper_s1/paper_r1_night/final_net_epoch_0091.pth",
}

# (gt_dir, img_dir_or_None)
SETS = {
    "outside": (os.path.join(ROOT, "data", "_eval_sets", "outside_combined"), None),
    "night": (os.path.join(ROOT, "data", "_eval_sets", "night_combined"), None),
    "indoor": (os.path.join(ROOT, "data", "pallet", "raw_data",
                            "capture0403middle", "gt_final"),
               os.path.join(ROOT, "data", "pallet", "raw_data",
                            "capture0403middle", "rgb")),
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


def eval_model_on_set(model, device, mean, std, gt_dir, img_dir, exclude):
    idir = img_dir or gt_dir
    files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    n_total = 0
    n_det = 0          # frames with >=6 corners
    means8 = []        # 8-corner Hungarian mean (per detected frame)
    means9 = []        # 9-kp (8 corners matched + centroid) mean
    n_good = 0
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
        nvalid = int((~np.isnan(pred8[:, 0])).sum())
        if nvalid < 6:
            continue
        n_det += 1
        m8, _ = hungarian_mean_dist(pred8, gt8)
        if not np.isfinite(m8):
            continue
        means8.append(m8)
        if m8 < GOOD_PX:
            n_good += 1

        # 9-kp: 8-corner matched mean already (m8); add centroid distance
        if kp[8] is not None and gt_c is not None:
            dc = float(np.linalg.norm(np.asarray(kp[8], float) -
                                      np.asarray(gt_c, float)))
            # weight 8 corners + 1 centroid -> mean of 9
            m9 = (m8 * 8 + dc) / 9.0
            means9.append(m9)

    means8 = np.array(means8)
    means9 = np.array(means9)
    return {
        "n_total": n_total,
        "n_det": n_det,
        "det_rate": n_det / n_total if n_total else 0.0,
        "reproj8_mean": float(means8.mean()) if len(means8) else float("nan"),
        "reproj8_median": float(np.median(means8)) if len(means8) else float("nan"),
        "reproj9_mean": float(means9.mean()) if len(means9) else float("nan"),
        "reproj9_median": float(np.median(means9)) if len(means9) else float("nan"),
        "good_pct": n_good / n_det if n_det else 0.0,
        "n_good": n_good,
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
            # indoor: only R0 (R1 not trained on indoor PL)
            if sname == "indoor" and mname != "R0_base":
                continue
            r = eval_model_on_set(model, device, mean, std,
                                  gt_dir, img_dir, exclude)
            results[mname][sname] = r
            print(f"  [{mname} x {sname}] det={r['det_rate']*100:5.1f}% "
                  f"reproj8_med={r['reproj8_median']:.2f} "
                  f"reproj9_med={r['reproj9_median']:.2f} "
                  f"good={r['good_pct']*100:.1f}%")

    out = os.path.join(HERE, "c1_r0_vs_r1_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[save] {out}")

    # ── matrix print ──
    print("\n" + "=" * 78)
    print("C1  R0 vs R1  (detection rate / 9kp reproj order-free / good<10px)")
    print("=" * 78)
    hdr = (f"{'model':<12}{'domain':<9}{'n':>5}{'det%':>7}"
           f"{'rep8_med':>10}{'rep9_med':>10}{'rep9_mean':>10}{'good%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for mname in MODELS:
        for sname in SETS:
            if sname not in results[mname]:
                continue
            r = results[mname][sname]
            print(f"{mname:<12}{sname:<9}{r['n_total']:>5}"
                  f"{r['det_rate']*100:>7.1f}"
                  f"{r['reproj8_median']:>10.2f}"
                  f"{r['reproj9_median']:>10.2f}"
                  f"{r['reproj9_mean']:>10.2f}"
                  f"{r['good_pct']*100:>7.1f}")
    return results


if __name__ == "__main__":
    main()
