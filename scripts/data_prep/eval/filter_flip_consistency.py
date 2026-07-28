"""filter_flip_consistency.py — flip-TTA consistency filter vs 9kp GT error.

Purpose (2026-06-08):
  Test whether a left-right FLIP consistency filter selects reliable PL whose
  FULL 9 keypoints match GT — specifically whether it CATCHES the back-face /
  centroid instability that `diag` (spatial-diagonal incidence, a projective
  invariant on the centroid only) cannot see.

Method per frame (paper_base, camera-facing 0123):
  A = original 9-kp prediction (REUSED from _full_paper_base.json cache).
  B = run inference on the horizontally-flipped image, then:
        un-flip x:  x -> W - x
        camera-facing symmetric swap (FLIP_PAIRS = 0<->1,3<->2,4<->5,7<->6),
        centroid(8) fixed.
  flip_dist = order-free? NO — A and B are the SAME model's predictions for the
        SAME (physical) keypoints, so they ARE index-aligned after the swap.
        We compare per-index ||A_i - B_i|| over the available kp, and use the
        MEAN over available 9 as the consistency score (px).
  pass(flip<=tau) for tau in sweep.

Selection criterion (memory filter-goal-reliable-pl-full-keypoints):
  judge passed PL by OVERALL 9-kp order-free error vs GT (Hungarian 8 corners +
  centroid), via nine_kp_err() — SAME as filter_combo_9kp.py.

Reuses cache for A + GT + img path (inference-free for A); only the FLIP pass is
new inference. Excludes _exclude.txt frames.

Usage:
  conda run -n pallet-pose python scripts/data_prep/eval/filter_flip_consistency.py \
      --tag paper_base
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from filter_pr_camfacing import (  # noqa: E402
    load_model, extract_keypoints_from_belief, filt_diag,
)
from filter_combo_9kp import nine_kp_err, load_exclude  # noqa: E402

import torch  # noqa: E402

DOMAINS = ["indoor", "outside", "night"]
# camera-facing left<->right symmetric corner swap (canonical filter_A pairs).
# centroid (8) maps to itself.
FLIP_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]
TAUS = [5.0, 8.0, 10.0, 15.0]
GOOD_PX = 10.0
GROSS_PX = 20.0


def build_swap():
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return swap  # swap[i] = source index in the flip-back array that maps to i


def infer_flip_kp(model, img_bgr, device, mean, std, threshold, preprocess="squash448"):
    """Run inference on the L-R flipped image; return 9 kp un-flipped + swapped
    back to the canonical camera-facing indexing, in ORIGINAL-image px (or None).

    preprocess:
      "squash448" (default, legacy) — resize flipped img to 448x448, belief→orig
                  via isotropic-per-axis sx,sy = bw/w0, bh/h0.
      "aspect"    — ratio-preserving short-side→400 (matches extract_pl_v1 main
                  path so the flip score is on a consistent preprocessing).
    Both branches return ORIGINAL-image px, so flip_consistency_score (which
    compares against an original-px kpA) is always on the same coordinate scale.
    """
    h0, w0 = img_bgr.shape[:2]
    img_flip = cv2.flip(img_bgr, 1)
    rgb = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)
    if preprocess == "aspect":
        _sc = 400.0 / min(h0, w0)
        _nw = max(8, int(round(w0 * _sc)) & ~7)
        _nh = max(8, int(round(h0 * _sc)) & ~7)
        img_in = cv2.resize(rgb, (_nw, _nh))
    else:
        _nw = _nh = 448
        img_in = cv2.resize(rgb, (448, 448))
    t = ((img_in.astype(np.float32) / 255.0 - mean) / std)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief = out_bel[-1][0].cpu().numpy()
    kps_bel = extract_keypoints_from_belief(belief, threshold)
    bh, bw = belief.shape[1], belief.shape[2]
    # belief → input(_nw,_nh) → flipped-orig px.  For squash448, _nw/bw=448/bw
    # and dividing by (448/w0) collapses to the legacy bw/w0; for aspect the
    # divide is by _sc.  Both yield px on the (flipped) original image.
    if preprocess == "aspect":
        _div_x = _div_y = _sc
        ux, uy = _nw / bw, _nh / bh
    else:
        _div_x, _div_y = _nw / w0, _nh / h0   # = 448/w0, 448/h0
        ux, uy = _nw / bw, _nh / bh

    # belief -> original-image px on the FLIPPED image, then un-flip x.
    flip_px = [None] * 9
    for i, k in enumerate(kps_bel):
        if k[0] < 0:
            continue
        u = (k[0] * ux) / _div_x   # px in flipped image
        v = (k[1] * uy) / _div_y
        flip_px[i] = (w0 - u, v)  # un-flip x back to original orientation

    # symmetric swap: the corner the network calls index i on the flipped image
    # physically corresponds to the mirrored corner.  After un-flipping x, swap
    # the labels so they align to the canonical camera-facing indexing.
    swap = build_swap()
    out = [None] * 9
    for i in range(9):
        out[i] = flip_px[swap[i]]
    return out


def flip_consistency_score(kpA, kpB):
    """Index-aligned mean per-kp distance over available 9 (A=orig, B=flip-back).

    Returns (mean_dist_px, n_compared).  None if < 4 comparable kp.
    """
    dists = []
    for i in range(9):
        if kpA[i] is not None and kpB[i] is not None:
            dists.append(np.linalg.norm(np.asarray(kpA[i], float) -
                                        np.asarray(kpB[i], float)))
    if len(dists) < 4:
        return None, len(dists)
    return float(np.mean(dists)), len(dists)


def summarize(errs9):
    """errs9 = list of overall 9kp GT errors (px) of passed PL."""
    if not errs9:
        return {"n": 0}
    a = np.asarray(errs9)
    n_good = int((a < GOOD_PX).sum())
    n_gross = int((a > GROSS_PX).sum())
    return {
        "n": int(a.size),
        "err9_med": round(float(np.median(a)), 1),
        "err9_mean": round(float(np.mean(a)), 1),
        "good": n_good,
        "good_pct": round(100 * n_good / a.size, 1),
        "gross": n_gross,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "paper_base", "final_net_epoch_0060.pth"))
    ap.add_argument("--tag", default="paper_base")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "filter_flip_consistency"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cache_fp = os.path.join(ROOT, "data", "pallet", "eval_results",
                            "filter_domain_analysis", f"_full_{args.tag}.json")
    data = json.load(open(cache_fp))
    exclude = load_exclude()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    model = load_model(args.weights, device)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # per-frame records: dom, e9 (9kp GT err), flip_score, diag_pass
    records = {d: [] for d in DOMAINS}
    for dom in DOMAINS:
        rows = [r for r in data[dom] if str(r["frame"]) not in exclude]
        det = [r for r in rows if r.get("mean_match_px") is not None]
        print(f"[{dom}] total={len(rows)} detectable={len(det)}  inferring flips...")
        for i, r in enumerate(det):
            kpA = [tuple(p) if p is not None else None for p in r["kp"]]
            e9, _, _ = nine_kp_err(kpA, r["gt8"])
            if e9 is None or not np.isfinite(e9):
                continue
            img = cv2.imread(r["img"])
            if img is None:
                continue
            kpB = infer_flip_kp(model, img, device, mean, std, args.threshold)
            fscore, ncmp = flip_consistency_score(kpA, kpB)
            diag_pass = bool(filt_diag(kpA)[0])
            records[dom].append({
                "frame": r["frame"], "e9": float(e9),
                "flip": fscore, "n_cmp": ncmp, "diag": diag_pass,
            })
            if (i + 1) % 40 == 0:
                print(f"   [{i+1}/{len(det)}]")

    # ── build report ──────────────────────────────────────────────────
    all_rec = []
    for d in DOMAINS:
        all_rec.extend(records[d])
    scopes = [(d, records[d]) for d in DOMAINS] + [("ALL", all_rec)]

    lines = []
    lines.append("FLIP CONSISTENCY FILTER (paper_base, camera-facing 0123)")
    lines.append(f"flip score = index-aligned mean per-kp ||A - B|| (px), "
                 f"A=orig pred, B=flip+unflip+swap{FLIP_PAIRS}")
    lines.append(f"judge = overall 9kp order-free GT err; good = <{GOOD_PX:.0f}px")
    lines.append("")

    # baseline: diag alone, and detectable pool good%
    lines.append("[reference: pool & diag-alone]")
    hdr = f"{'scope':<10}{'det':>5}{'good_pool%':>11}{'diagN':>7}{'diag_med':>9}{'diag_good%':>11}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    diag_stats = {}
    for scope, rec in scopes:
        if not rec:
            continue
        pool_e9 = [x["e9"] for x in rec]
        pool_good = 100 * np.mean(np.asarray(pool_e9) < GOOD_PX)
        diag_e9 = [x["e9"] for x in rec if x["diag"]]
        ds = summarize(diag_e9)
        diag_stats[scope] = ds
        lines.append(f"{scope:<10}{len(rec):>5}{pool_good:>10.1f}%"
                     f"{ds.get('n', 0):>7}{ds.get('err9_med', float('nan')):>9.1f}"
                     f"{ds.get('good_pct', float('nan')):>10.1f}%")

    # flip consistency sweep
    lines.append("")
    lines.append("[flip consistency  tau-sweep]")
    hdr = f"{'scope':<10}{'tau':>5}{'N':>5}{'9kp_med':>8}{'9kp_mn':>8}{'good%':>7}{'gross':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    flip_result = {}
    for scope, rec in scopes:
        if not rec:
            continue
        flip_result[scope] = {}
        for tau in TAUS:
            passed = [x["e9"] for x in rec
                      if x["flip"] is not None and x["flip"] <= tau]
            s = summarize(passed)
            flip_result[scope][tau] = s
            if s["n"] == 0:
                lines.append(f"{scope:<10}{tau:>5.0f}{0:>5}{'--':>8}{'--':>8}{'--':>7}{'--':>6}")
            else:
                lines.append(f"{scope:<10}{tau:>5.0f}{s['n']:>5}{s['err9_med']:>8.1f}"
                             f"{s['err9_mean']:>8.1f}{s['good_pct']:>6.0f}%{s['gross']:>6}")
        lines.append("")

    # diag AND flip combo
    lines.append("[diag AND flip  (does flip prune diag's bad PL?)]")
    hdr = f"{'scope':<10}{'tau':>5}{'N':>5}{'9kp_med':>8}{'good%':>7}{'gross':>6}  vs diag-alone"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    combo_result = {}
    for scope, rec in scopes:
        if not rec:
            continue
        combo_result[scope] = {}
        ds = diag_stats.get(scope, {"n": 0})
        for tau in TAUS:
            passed = [x["e9"] for x in rec
                      if x["diag"] and x["flip"] is not None and x["flip"] <= tau]
            s = summarize(passed)
            combo_result[scope][tau] = s
            # how many diag-pass were dropped by flip, and were they bad?
            diag_only_dropped = [x["e9"] for x in rec
                                 if x["diag"] and (x["flip"] is None or x["flip"] > tau)]
            drop_med = (round(float(np.median(diag_only_dropped)), 1)
                        if diag_only_dropped else None)
            combo_result[scope][tau]["dropped_n"] = len(diag_only_dropped)
            combo_result[scope][tau]["dropped_med"] = drop_med
            if s["n"] == 0:
                lines.append(f"{scope:<10}{tau:>5.0f}{0:>5}{'--':>8}{'--':>7}{'--':>6}"
                             f"  (diag: N={ds.get('n',0)} good%={ds.get('good_pct','--')})")
            else:
                dd = f"drop {len(diag_only_dropped)} (med {drop_med})"
                lines.append(f"{scope:<10}{tau:>5.0f}{s['n']:>5}{s['err9_med']:>8.1f}"
                             f"{s['good_pct']:>6.0f}%{s['gross']:>6}  "
                             f"diag:N={ds.get('n',0)} good%={ds.get('good_pct','--')} | {dd}")
        lines.append("")

    print("\n".join(lines))
    txt = os.path.join(args.output_dir, f"flip_consistency_{args.tag}.txt")
    open(txt, "w").write("\n".join(lines))
    out = {
        "tag": args.tag, "taus": TAUS, "good_px": GOOD_PX, "gross_px": GROSS_PX,
        "flip_pairs": FLIP_PAIRS,
        "diag": diag_stats, "flip": flip_result, "diag_and_flip": combo_result,
        "records": records,
    }
    json.dump(out, open(os.path.join(
        args.output_dir, f"flip_consistency_{args.tag}.json"), "w"), indent=2)
    print(f"\n[save] {txt}")
    print(f"[save] {os.path.join(args.output_dir, f'flip_consistency_{args.tag}.json')}")


if __name__ == "__main__":
    main()
