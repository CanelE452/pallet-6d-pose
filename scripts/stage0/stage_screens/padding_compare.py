"""stage0/padding_compare.py — inference-time padding ablation (training-free).

Compares padding modes (none / reflect / black / white) at INFERENCE ONLY
(no retraining). For each mode and each frame we measure:
  (1) detection count (frames with >=N_DET_MIN corners),
  (2) corner median px (order-free Hungarian match to GT, detected frames),
  (3) good (<GOOD_PX) / gross (>GROSS_PX) counts,
on real filter-val (outside / night) and a synthetic-val sample.

Reuses ONLY verified pieces (no new geometry):
  filter_pr_camfacing : load_model, extract_keypoints_from_belief,
                        hungarian_mean_dist
  dope_predict_mp4_pad: pad_frame (reflect/replicate/black/white)
  tau_calibrate       : collect_val_frames (filter-val GT frames)

Pipeline per (frame, mode):
  1. load BGR img (h0,w0).
  2. proc = pad_frame(img, PAD, mode)   (mode="none" => PAD=0 => img unchanged).
  3. four_arm preprocess (lines 235-258) on proc: RGB, aspect-resize short=400,
     ImageNet normalize -> belief.
  4. belief peak -> proc (=padded-resized canvas, same HxW as orig) coords via
     four_arm's *ux,*uy,/_sc mapping.
  5. undo padding: canvas (cx,cy) -> true original coords
        ox = cx*(w0+2P)/w0 - P ,  oy = cy*(h0+2P)/h0 - P
     (same formula as dope_predict_mp4_pad.belief_peak_to_orig; skipped if PAD<=0).
  6. pred8 vs gt8 -> hungarian_mean_dist -> gt_err.

★ --validate: PAD=0 ("none") gt_err MUST equal four_arm_pl_compare run_inference
  gt_err per-frame (same preprocess). Prints a per-frame diff for a few outside
  frames as a mapping-correctness self-check.

CPU smoke / list:
  conda run -n pallet-pose python scripts/stage0/stage_screens/padding_compare.py --validate
Full ablation (GPU):
  conda run -n pallet-pose python scripts/stage0/stage_screens/padding_compare.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from tau_calibrate import collect_val_frames  # noqa: E402

GOOD_PX = 10.0
GROSS_PX = 20.0
N_DET_MIN = 6
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "paper_base_v2_s2",
                               "net_epoch_0066.pth")
SYN_VAL_DIR = os.path.join(ROOT, "data", "pallet", "training_data", "val")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def collect_syn_frames(n_syn):
    """Evenly sample n_syn synthetic-val frames -> (dom, fid, jp, ip) list."""
    jsons = sorted(glob.glob(os.path.join(SYN_VAL_DIR, "*.json")))
    pairs = []
    for jp in jsons:
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(SYN_VAL_DIR, fid + ".png")
        if os.path.exists(ip):
            pairs.append((fid, jp, ip))
    if not pairs:
        return []
    if n_syn >= len(pairs):
        idx = range(len(pairs))
    else:
        idx = np.linspace(0, len(pairs) - 1, n_syn).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
    return [("synthetic", pairs[i][0], pairs[i][1], pairs[i][2]) for i in idx]


def infer_one(model, extract_keypoints_from_belief, hungarian_mean_dist,
              torch, cv2, img, gt8, pad, mode, threshold, device):
    """One frame, one padding mode -> (gt_err, n_det, pred8). Returns
    (gt_err, n_det) with gt_err possibly inf (<6 corners)."""
    from dope_predict_mp4_pad import pad_frame
    h0, w0 = img.shape[:2]
    use_pad = pad if mode != "none" else 0
    proc = pad_frame(img, use_pad, mode)  # padded-resized canvas, same HxW
    # four_arm preprocess (lines 235-258), applied to proc.
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    ph, pw = proc.shape[:2]
    _sc = 400.0 / min(ph, pw)
    _nw = max(8, int(round(pw * _sc)) & ~7)
    _nh = max(8, int(round(ph * _sc)) & ~7)
    t = ((cv2.resize(rgb, (_nw, _nh)).astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps_bel = extract_keypoints_from_belief(belief, threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    ux, uy = _nw / bw, _nh / bh

    pred8 = np.full((8, 2), np.nan)
    n_det = 0
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        # belief -> proc(=canvas) coords (four_arm mapping)
        cx = (k[0] * ux) / _sc
        cy = (k[1] * uy) / _sc
        # undo padding: canvas -> true original coords (mp4_pad formula)
        if use_pad > 0:
            ox = cx * (w0 + 2 * use_pad) / w0 - use_pad
            oy = cy * (h0 + 2 * use_pad) / h0 - use_pad
        else:
            ox, oy = cx, cy
        pred8[i] = (ox, oy)
        n_det += 1
    gt_err, _ = hungarian_mean_dist(pred8, gt8)
    return gt_err, n_det, pred8


def load_gt8(json_path):
    gt = json.load(open(json_path))
    return np.array(gt["objects"][0]["projected_cuboid"], float)[:8]


def bucket(errs):
    e = np.array([x for x in errs if np.isfinite(x)], float)
    if e.size == 0:
        return {"N_detected": 0, "corner_median": None, "good": 0, "gross": 0}
    return {
        "N_detected": int(e.size),
        "corner_median": round(float(np.median(e)), 1),
        "good": int((e < GOOD_PX).sum()),
        "gross": int((e > GROSS_PX).sum()),
    }


def run_validate(args):
    """PAD=0 ('none') gt_err must match four_arm run_inference gt_err per-frame."""
    import cv2
    import torch
    from filter_pr_camfacing import (
        load_model, extract_keypoints_from_belief, hungarian_mean_dist,
    )
    sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
    from four_arm_pl_compare import run_inference as fa_run_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = collect_val_frames()
    outside = [f for f in frames if f[0] == "outside"][:args.n_validate]
    print(f"[validate] {len(outside)} outside frames (PAD=0 'none' vs four_arm)")

    # four_arm reference (same preprocess, no padding)
    fa = fa_run_inference(outside, args.weights, args.threshold)
    fa_by_fid = {r["frame"]: r["gt_err"] for r in fa}

    model = load_model(args.weights, device)
    print(f"\n  {'frame':<22}{'four_arm':>10}{'pad0':>10}{'|diff|':>10}")
    print("  " + "-" * 52)
    max_diff = 0.0
    n_cmp = 0
    for dom, fid, jp, ip in outside:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = load_gt8(jp)
        err, _, _ = infer_one(model, extract_keypoints_from_belief,
                              hungarian_mean_dist, torch, cv2, img, gt8,
                              0, "none", args.threshold, device)
        ref = fa_by_fid.get(fid)
        if ref is None:
            continue
        diff = abs(err - ref)
        max_diff = max(max_diff, diff)
        n_cmp += 1
        print(f"  {fid:<22}{ref:>10.4f}{err:>10.4f}{diff:>10.6f}")
    print("  " + "-" * 52)
    ok = max_diff < 1e-6
    print(f"\n  compared={n_cmp}  max|diff|={max_diff:.3e}  "
          f"=> {'PASS (PAD=0 ≡ four_arm)' if ok else 'FAIL — mapping mismatch'}")
    return ok


def run_full(args):
    import cv2
    import torch
    from filter_pr_camfacing import (
        load_model, extract_keypoints_from_belief, hungarian_mean_dist,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model = load_model(args.weights, device)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    frames = collect_val_frames()
    if args.n_syn > 0:
        frames = frames + collect_syn_frames(args.n_syn)
    by_dom = {}
    for dom, *_ in frames:
        by_dom[dom] = by_dom.get(dom, 0) + 1
    print(f"[frames] {len(frames)} total {by_dom}  modes={modes}  pad={args.pad}")

    # results[mode][dom] = list of gt_err for detected frames
    results = {m: {} for m in modes}
    for n, (dom, fid, jp, ip) in enumerate(frames):
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = load_gt8(jp)
        for mode in modes:
            err, n_det, _ = infer_one(
                model, extract_keypoints_from_belief, hungarian_mean_dist,
                torch, cv2, img, gt8, args.pad, mode, args.threshold, device)
            if n_det >= N_DET_MIN and np.isfinite(err):
                results[mode].setdefault(dom, []).append(err)
            else:
                results[mode].setdefault(dom, [])
        if (n + 1) % 25 == 0:
            print(f"  [{n+1}/{len(frames)}]")

    doms = sorted({d for d, *_ in frames})
    out_lines = []
    out_lines.append("PADDING-MODE INFERENCE ABLATION (training-free)")
    out_lines.append(f"weights={args.weights}  pad={args.pad}  "
                     f"threshold={args.threshold}")
    out_lines.append(f"detection gate n_det>={N_DET_MIN}; "
                     f"good<{GOOD_PX:g}px gross>{GROSS_PX:g}px; "
                     f"corner err = order-free Hungarian to GT")
    summary = {}
    for dom in doms:
        out_lines.append(f"\n=== {dom} ===")
        hdr = (f"  {'mode':<10}{'N_detected':>11}{'corner_median':>15}"
               f"{'good':>6}{'gross':>7}")
        out_lines.append(hdr)
        out_lines.append("  " + "-" * (len(hdr) - 2))
        summary[dom] = {}
        for mode in modes:
            b = bucket(results[mode].get(dom, []))
            summary[dom][mode] = b
            med = b["corner_median"]
            med_s = f"{med:.1f}" if med is not None else "--"
            out_lines.append(
                f"  {mode:<10}{b['N_detected']:>11}{med_s:>15}"
                f"{b['good']:>6}{b['gross']:>7}")

    print("\n".join(out_lines))
    os.makedirs(args.output_dir, exist_ok=True)
    txt = os.path.join(args.output_dir, "padding_compare.txt")
    open(txt, "w").write("\n".join(out_lines))
    json.dump({
        "weights": args.weights, "pad": args.pad, "threshold": args.threshold,
        "good_px": GOOD_PX, "gross_px": GROSS_PX, "n_det_min": N_DET_MIN,
        "modes": modes, "n_syn": args.n_syn, "summary": summary,
    }, open(os.path.join(args.output_dir, "padding_compare.json"), "w"),
        indent=2, default=str)
    print(f"\n[save] {txt}")
    print(f"[save] {os.path.join(args.output_dir, 'padding_compare.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--pad", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--modes", default="none,reflect,black,white")
    ap.add_argument("--n_syn", type=int, default=200)
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "stage0_padding_compare"))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n_validate", type=int, default=8)
    args = ap.parse_args()

    if args.validate:
        ok = run_validate(args)
        sys.exit(0 if ok else 1)
    run_full(args)


if __name__ == "__main__":
    main()
