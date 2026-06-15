"""stage0/tau_calibrate.py — calibrate τ_diag + flip weight on FILTER-VAL ONLY.

★ final-test sessions (capturepallet09/07, capturenight09/08) are NEVER touched.
   Calibration uses ONLY the filter-val sessions:
     outside : capturepallet08,02,03,04,05
     night   : capturenight06,07,05
   GT frames are mapped back to their session via the rgb folders; any frame whose
   session is not a filter-val session (final-test, cad, pool) is dropped.

Goal: find the operating point that "drops gross PL without killing good PL":
  sweep τ_diag (diag pass threshold) × flip_tau (flip-consistency cutoff),
  judged by order-free 9-corner GT match (good < good_px, gross > gross_px).

Reuses (NO new geometry):
  filter_pr_camfacing : load_model, extract_keypoints_from_belief,
                        canonical_kp3d, hungarian_mean_dist, filt_diag
  filter_flip_consistency : infer_flip_kp, flip_consistency_score, FLIP_PAIRS

★ GPU script — DO NOT RUN here. CPU-safe: --list_val_frames prints the filter-val
   GT frames it would use (verifies sealing) without loading the model.

Usage (user, GPU):
  conda run -n pallet-pose python scripts/stage0/tau_calibrate.py \
      --weights weights/paper_base/final_net_epoch_0060.pth
Smoke (CPU):
  python scripts/stage0/tau_calibrate.py --list_val_frames
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

# filter-val sessions per split-lock (final-test sessions intentionally absent)
FILTER_VAL_SESSIONS = {
    "outside": {"capturepallet08", "capturepallet02", "capturepallet03",
                "capturepallet04", "capturepallet05"},
    "night": {"capturenight06", "capturenight07", "capturenight05"},
}
FINAL_TEST_SESSIONS = {"capturepallet09", "capturepallet07",
                       "capturenight09", "capturenight08"}
EVAL_SET = {"outside": "outside_combined", "night": "night_combined"}

TAU_DIAG = [0.02, 0.03, 0.05, 0.08, 0.12]   # diag pass thresholds (scale-free)
FLIP_TAUS = [5.0, 8.0, 10.0, 15.0, 1e9]      # 1e9 = flip off
GOOD_PX = 10.0
GROSS_PX = 20.0


def build_session_map():
    m = {}
    for base in ("outside", "night"):
        for d in glob.glob(os.path.join(ROOT, "data", base, "capture*")):
            sess = os.path.basename(d)
            for p in glob.glob(os.path.join(d, "rgb", "*.png")):
                m[os.path.splitext(os.path.basename(p))[0]] = sess
    return m


def collect_val_frames():
    """Return list of (domain, frame_id, gt_json_path, img_path) for filter-val."""
    sm = build_session_map()
    out = []
    for dom, valset in FILTER_VAL_SESSIONS.items():
        gt_dir = os.path.join(ROOT, "data", "_eval_sets", EVAL_SET[dom])
        for jp in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
            fid = os.path.splitext(os.path.basename(jp))[0]
            sess = sm.get(fid)
            if sess is None or sess not in valset:
                continue
            # SEAL guard: refuse any final-test frame
            assert sess not in FINAL_TEST_SESSIONS, f"final-test leak: {fid}"
            ip = os.path.join(gt_dir, fid + ".png")
            if not os.path.exists(ip):
                ip = os.path.join(ROOT, "data", dom, sess, "rgb", fid + ".png")
            out.append((dom, fid, jp, ip))
    return out


def summarize(errs):
    if not errs:
        return {"n": 0, "good": 0, "good_pct": 0.0, "gross": 0, "med": None}
    a = np.asarray(errs)
    return {
        "n": int(a.size),
        "good": int((a < GOOD_PX).sum()),
        "good_pct": round(100 * float((a < GOOD_PX).mean()), 1),
        "gross": int((a > GROSS_PX).sum()),
        "med": round(float(np.median(a)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "paper_base", "final_net_epoch_0060.pth"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--good_px", type=float, default=GOOD_PX)
    ap.add_argument("--gross_px", type=float, default=GROSS_PX)
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "stage0_tau_calibrate"))
    ap.add_argument("--list_val_frames", action="store_true",
                    help="CPU-safe: print filter-val frames and exit (seal check)")
    args = ap.parse_args()

    frames = collect_val_frames()
    by_dom = {}
    for dom, *_ in frames:
        by_dom[dom] = by_dom.get(dom, 0) + 1
    print(f"[filter-val] {len(frames)} GT frames {by_dom}")
    if args.list_val_frames:
        for dom, fid, _, _ in frames:
            print(f"  {dom:<8} {fid}")
        print("[list-only] final-test sessions are NOT in this set (sealed).")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    import cv2
    import torch
    from filter_pr_camfacing import (
        load_model, extract_keypoints_from_belief, canonical_kp3d,
        hungarian_mean_dist, filt_diag,
    )
    from filter_flip_consistency import infer_flip_kp, flip_consistency_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model = load_model(args.weights, device)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # per-frame: gt_err(9-corner order-free), diag_score, flip_score
    recs = []
    for dom, fid, jp, ip in frames:
        gt = json.load(open(jp))
        gt8 = np.array(gt["objects"][0]["projected_cuboid"], float)[:8]
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = ((cv2.resize(rgb, (448, 448)).astype(np.float32) / 255.0 - mean) / std)
        tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_bel, _ = model(tensor)
        belief = out_bel[-1][0].cpu().numpy()
        kps_bel = extract_keypoints_from_belief(belief, args.threshold)
        bh, bw = belief.shape[1], belief.shape[2]
        sx, sy = bw / w0, bh / h0
        kp = []
        for k in kps_bel:
            kp.append(None if k[0] < 0 else (k[0] / sx, k[1] / sy))

        pred8 = np.full((8, 2), np.nan)
        for i in range(8):
            if kp[i] is not None:
                pred8[i] = kp[i]
        gt_err, _ = hungarian_mean_dist(pred8, gt8)  # order-free 8-corner match
        if not np.isfinite(gt_err):
            continue
        _, diag_score = filt_diag(kp)
        kpB = infer_flip_kp(model, img, device, mean, std, args.threshold)
        flip_score, _ = flip_consistency_score(kp, kpB)
        recs.append({"dom": dom, "frame": fid, "gt_err": float(gt_err),
                     "diag_score": (None if not np.isfinite(diag_score) else float(diag_score)),
                     "flip_score": (None if flip_score is None else float(flip_score))})

    # sweep τ_diag × flip_tau on the filter-val pool
    print(f"\n[sweep] {len(recs)} usable frames  "
          f"(good<{args.good_px:g} gross>{args.gross_px:g})")
    grid = []
    hdr = f"{'tau_diag':>9}{'flip_tau':>9}{'N':>5}{'good':>6}{'good%':>7}{'gross':>7}{'med':>6}"
    print(hdr); print("-" * len(hdr))
    for td in TAU_DIAG:
        for ft in FLIP_TAUS:
            passed = [r["gt_err"] for r in recs
                      if r["diag_score"] is not None and r["diag_score"] < td
                      and (ft >= 1e8 or (r["flip_score"] is not None and r["flip_score"] <= ft))]
            s = summarize(passed)
            grid.append({"tau_diag": td, "flip_tau": (None if ft >= 1e8 else ft), **s})
            ftlabel = "off" if ft >= 1e8 else f"{ft:g}"
            print(f"{td:>9.3f}{ftlabel:>9}{s['n']:>5}{s['good']:>6}"
                  f"{s['good_pct']:>6.1f}%{s['gross']:>7}"
                  f"{(s['med'] if s['med'] is not None else float('nan')):>6.1f}")

    out = {"weights": args.weights, "good_px": args.good_px,
           "gross_px": args.gross_px, "filter_val_sessions": {
               k: sorted(v) for k, v in FILTER_VAL_SESSIONS.items()},
           "n_frames": len(recs), "grid": grid, "records": recs}
    fp = os.path.join(args.output_dir, "tau_calibrate.json")
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\n[save] {fp}")
    print("Pick: highest good% with gross small AND N not collapsed.")
    print("final-test was NOT used (sealed).")


if __name__ == "__main__":
    main()
