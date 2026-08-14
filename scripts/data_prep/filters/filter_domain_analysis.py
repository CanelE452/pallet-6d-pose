"""filter_domain_analysis.py — per-DOMAIN filter pass/good analysis.

Domains: indoor (capture0403middle, AprilTag GT, 440),
         outside (outside_combined, 129), night (night_combined, 90).

Filters (user-specified):
  fullkp     : all 9 keypoints detected
  diag       : spatial diagonals (0-6,1-7,2-4,3-5) intersect near centroid(8)
  ratio      : width-edge group {0-1,3-2,4-5,7-6} consistent length AND
               depth-edge group {0-4,1-5,2-6,3-7} consistent length
  ransac_loo : RANSAC subset consensus AND leave-one-out PnP stability
  combo      : fullkp AND diag AND ratio AND ransac_loo  (4-way AND)

good judgment: order-free Hungarian mean reproj of predicted 8 corners vs GT
projected_cuboid < good_thresh_px (convention-agnostic; absorbs W/D swap).

Outputs domain x filter pass-rate + good-of-passed + qualitative buckets +
representative overlays.  Inference only.
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

# reuse all building blocks from the existing screening script
from filter_pr_camfacing import (  # noqa: E402
    canonical_kp3d, load_model, extract_keypoints_from_belief,
    hungarian_mean_dist, filt_diag, filt_ratio, filt_fullkp,
    sqpnp, ransac_consensus, loo_stability,
)

DOMAINS = {
    "indoor": os.path.join(ROOT, "data", "pallet", "raw_data",
                           "capture0403middle", "gt_final"),
    "outside": os.path.join(ROOT, "data", "_eval_sets", "outside_combined"),
    "night": os.path.join(ROOT, "data", "_eval_sets", "night_combined"),
}
IMG_SUBDIR = {  # where rgb lives relative to gt dir (None = same dir)
    "indoor": os.path.join(ROOT, "data", "pallet", "raw_data",
                           "capture0403middle", "rgb"),
    "outside": None,
    "night": None,
}
DEFAULT_OUT = os.path.join(ROOT, "data", "pallet", "eval_results",
                           "filter_domain_analysis")

FILTERS = ["fullkp", "diag", "ratio", "ransac_loo", "combo"]
RANSAC_C = 6


def run_frame(jp, idir, model, device, mean, std, threshold):
    base = os.path.splitext(os.path.basename(jp))[0]
    ip = None
    for ext in (".png", ".jpg"):
        c = os.path.join(idir, base + ext)
        if os.path.exists(c):
            ip = c
            break
    if ip is None:
        return None
    gt = json.load(open(jp))
    obj = gt["objects"][0]
    gt8 = np.array(obj["projected_cuboid"], float)[:8]
    dm = obj.get("dimensions_m", {"width": 1.3, "depth": 1.1, "height": 0.11})
    kp3d = canonical_kp3d(dm["width"], dm["depth"], dm["height"])
    intr = gt["camera_data"]["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    dist = np.zeros((5, 1))

    img = cv2.imread(ip)
    h0, w0 = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = ((cv2.resize(rgb, (448, 448)).astype(np.float32) / 255.0 - mean) / std)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps_bel = extract_keypoints_from_belief(belief, threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / w0, bh / h0

    kp, confs = [], []
    for k in kps_bel:
        if k[0] < 0:
            kp.append(None)
            confs.append(0.0)
        else:
            kp.append((k[0] / sx, k[1] / sy))
            confs.append(k[2])
    n_det = sum(1 for x in kp[:8] if x is not None)

    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    mean_match, n_match = hungarian_mean_dist(pred8, gt8)

    # filters
    d_ok, d_sc = filt_diag(kp)
    r_ok, r_sc = filt_ratio(kp)
    f_ok, _ = filt_fullkp(kp)
    n_cons, R_rs, t_rs = ransac_consensus(kp, kp3d, K, dist)
    ransac_pass = n_cons >= RANSAC_C
    loo_pass = (loo_stability(kp, kp3d, K, dist, R_rs, t_rs)
                if (ransac_pass and R_rs is not None) else False)
    rl_ok = bool(ransac_pass and loo_pass)
    combo_ok = bool(f_ok and d_ok and r_ok and rl_ok)

    return {
        "frame": base, "img": ip,
        "n_detected": n_det,
        "mean_match_px": (round(mean_match, 2) if np.isfinite(mean_match)
                          else None),
        "n_match": n_match,
        "diag_score": (round(d_sc, 4) if np.isfinite(d_sc) else None),
        "ratio_score": (round(r_sc, 4) if np.isfinite(r_sc) else None),
        "ransac_consensus": int(n_cons),
        "kp": [list(map(float, p)) if p is not None else None for p in kp],
        "gt8": gt8.tolist(),
        "filters": {"fullkp": f_ok, "diag": bool(d_ok), "ratio": bool(r_ok),
                    "ransac_loo": rl_ok, "combo": combo_ok},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--good_thresh_px", type=float, default=10.0)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--indoor_limit", type=int, default=0,
                    help="0 = all indoor frames")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model = load_model(args.weights, device)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    per_frame = {}
    for dom, gt_dir in DOMAINS.items():
        idir = IMG_SUBDIR[dom] or gt_dir
        files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
        if dom == "indoor" and args.indoor_limit > 0:
            files = files[:: max(1, len(files) // args.indoor_limit)]
        print(f"[{dom}] {len(files)} GT frames")
        rows = []
        for fi, jp in enumerate(files):
            r = run_frame(jp, idir, model, device, mean, std, args.threshold)
            if r is None:
                continue
            r["good"] = bool(r["mean_match_px"] is not None and
                             r["mean_match_px"] < args.good_thresh_px)
            rows.append(r)
            if (fi + 1) % 60 == 0:
                print(f"  [{dom} {fi+1}/{len(files)}]")
        per_frame[dom] = rows

    # ── domain x filter table ────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"DOMAIN x FILTER  (good = order-free reproj < {args.good_thresh_px}px)")
    print("=" * 78)
    summary = {}
    for dom, rows in per_frame.items():
        n = len(rows)
        det = [r for r in rows if r["n_detected"] >= 6]
        n_good = sum(1 for r in rows if r["good"])
        base = n_good / len(det) if det else 0.0
        # buckets for qualitative
        gross = [r for r in det if r["mean_match_px"] is not None
                 and r["mean_match_px"] > 20]
        print(f"\n--- {dom}: total={n}  detectable(>=6kp)={len(det)}  "
              f"good={n_good}  base_rate={base:.3f}  gross(>20px)={len(gross)} ---")
        hdr = (f"{'filter':<12}{'pass':>6}{'/total':>8}{'/det':>7}"
               f"{'good_of_pass':>14}{'gross_rej%':>12}")
        print(hdr)
        print("-" * len(hdr))
        summary[dom] = {"total": n, "detectable": len(det), "good": n_good,
                        "base_rate": round(base, 4), "gross": len(gross),
                        "filters": {}}
        for fid in FILTERS:
            passed = [r for r in rows if r["filters"][fid]]
            np_ = len(passed)
            good_pass = sum(1 for r in passed if r["good"])
            gop = good_pass / np_ if np_ else 0.0
            gross_pass = sum(1 for r in passed if r["mean_match_px"] is not None
                             and r["mean_match_px"] > 20)
            gross_rej = (1 - gross_pass / len(gross)) if gross else 0.0
            pct_total = np_ / n if n else 0.0
            pct_det = np_ / len(det) if det else 0.0
            print(f"{fid:<12}{np_:>6}{pct_total*100:>7.0f}%{pct_det*100:>6.0f}%"
                  f"{good_pass:>7}/{np_:<6}{gross_rej*100:>11.0f}%")
            summary[dom]["filters"][fid] = {
                "pass": np_, "pass_pct_of_total": round(pct_total, 4),
                "pass_pct_of_detectable": round(pct_det, 4),
                "good_of_pass": good_pass,
                "good_of_pass_rate": round(gop, 4),
                "gross_pass": gross_pass,
                "gross_reject_rate": round(gross_rej, 4)}

    out_json = os.path.join(args.output_dir, f"summary_{args.tag}.json")
    json.dump({"weights": args.weights, "good_thresh_px": args.good_thresh_px,
               "summary": summary}, open(out_json, "w"), indent=2)
    # strip heavy fields from per_frame dump
    pf_dump = {d: [{k: v for k, v in r.items() if k not in ("kp", "gt8")}
                   for r in rows] for d, rows in per_frame.items()}
    out_pf = os.path.join(args.output_dir, f"per_frame_{args.tag}.json")
    json.dump(pf_dump, open(out_pf, "w"), indent=2, default=str)
    # keep full (with kp) for overlay step
    full_pf = os.path.join(args.output_dir, f"_full_{args.tag}.json")
    json.dump(per_frame, open(full_pf, "w"), default=str)
    print(f"\n[save] {out_json}\n[save] {out_pf}\n[save] {full_pf}")


if __name__ == "__main__":
    main()
