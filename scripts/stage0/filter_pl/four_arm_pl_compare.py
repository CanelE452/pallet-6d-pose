"""stage0/four_arm_pl_compare.py — PL-level 5-arm comparison at MATCHED pass-count.

★ THESIS TEST (paper_strategy_master §206 "same pass count"):
  Does the geometric `diag` filter suppress GROSS pseudo-labels (catastrophic,
  >gross_px) BETTER than naive size/aspect, confidence, or no-filter — when each
  arm is forced to pass the SAME number of frames (same budget N)?

  The point is NOT that diag passes fewer frames. We sweep EACH arm's threshold
  to a common N grid and compare gross-count (and good%) at matched N. If diag
  has fewer gross at the same N than size/aspect, the thesis (my geometric filter
  beats a strong naive geometric baseline) holds; if comparable, thesis weakens.

ARMS (continuous score, lower = more "accept"; sweep threshold for N):
  A1 none   : detection-only (n_det>=6 pass). score = 0 if detected else inf
              (not sweepable for N — reported as a fixed reference row).
  A2 conf   : -min belief-peak confidence over detected corners (sweep ->
              accept high-conf first).
  A4 size/aspect (★ STRONG naive geometry baseline, see _a4_score):
              penalty = how far the detected-corner 2D bbox (size & aspect)
              falls OUTSIDE the GT cuboid bbox envelope (model-independent).
              Envelope fit on filter-val GT projected_cuboid bboxes to MAXIMIZE
              GT pass (strawman-free). Continuous & sweepable.
  A6 diag   : filt_diag score (centroid incidence of spatial diagonals), AND a
              flip-consistency gate (matches the deployed filter). For the
              matched-N sweep we sweep the diag score; flip is applied as a fixed
              gate at flip_tau (so A6 = diag(score) AND flip<=flip_tau).
  A7 diag+size : A6 AND size/aspect accept (intersection); sweep diag score.

JUDGEMENT (order-free, convention-agnostic — memory evaluate-on-val bug):
  Hungarian-match predicted 8 corners to GT projected_cuboid; mean matched px
  = gt_err.  good = gt_err < good_px ; gross = gt_err > gross_px.

DATA / SEALING:
  filter-val ONLY (outside p08,02,03,04,05 ; night n06,07,05). final-test
  sessions (capturepallet09/07, capturenight09/08) are asserted-absent.
  ★ OUTSIDE-ONLY is the primary thesis read (night depth-collapse is a known
  systematic limit, reported separately & honestly).

PRELIMINARY: usable (detected) GT is small (~tens). Results are a PRELIMINARY
SIGNAL at N=<n>, NOT proof. Reported as such.

Reuses (NO new geometry):
  filter_pr_camfacing : load_model, extract_keypoints_from_belief, filt_diag,
                        hungarian_mean_dist
  filter_flip_consistency : infer_flip_kp, flip_consistency_score
  tau_calibrate : collect_val_frames, FILTER_VAL/FINAL_TEST sessions, aspect
                  preprocess (same scale — squash forbidden)

CPU smoke (no GPU, seal check):
  python scripts/stage0/filter_pl/four_arm_pl_compare.py --list_val_frames

GPU (user / agent — only 87 GT, fast):
  conda run -n pallet-pose python scripts/stage0/filter_pl/four_arm_pl_compare.py \
      --weights weights/paper_base/paper_base_v2_s2/net_epoch_0066.pth
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

from tau_calibrate import (  # noqa: E402
    collect_val_frames, FILTER_VAL_SESSIONS, FINAL_TEST_SESSIONS,
)

GOOD_PX = 10.0
GROSS_PX = 20.0
N_DET_MIN = 6          # detection gate shared by every arm (parity with A1)
FLIP_TAU = 10.0        # fixed flip gate for A6/A7 (deployed operating point)
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "paper_base_v2_s2",
                               "net_epoch_0066.pth")


# ── A4 size/aspect: strong naive geometry baseline ────────────────────
def _bbox_size_aspect(pts_xy):
    """2D bbox (size = diag/img_diag is added by caller, here raw). pts_xy:
    list of (x,y) detected corners. Returns (bbox_diag, aspect=w/h) or None."""
    if len(pts_xy) < N_DET_MIN:
        return None
    a = np.asarray(pts_xy, float)
    w = a[:, 0].max() - a[:, 0].min()
    h = a[:, 1].max() - a[:, 1].min()
    if w <= 1e-6 or h <= 1e-6:
        return None
    return float(np.hypot(w, h)), float(w / h)


def fit_a4_envelope(gt_records):
    """Fit accept envelope (size-ratio & aspect) on filter-val GT cuboid bboxes.

    Model-independent (uses GT projected_cuboid, not predictions) so the
    baseline is NOT tuned to the model under test and NOT circular. We take a
    generous [p2.5, p97.5] envelope so it MAXIMIZES GT pass (strawman-free):
    a frame is "in envelope" iff its size-ratio AND aspect lie inside. The
    continuous A4 score = max relative excursion outside the two intervals
    (0 inside, grows linearly outside), giving a sweepable score where lowering
    the threshold = shrinking the accepted envelope.
    """
    sizes = np.array([g["gt_size_ratio"] for g in gt_records])
    asps = np.array([g["gt_aspect"] for g in gt_records])
    env = {
        "size_lo": float(np.percentile(sizes, 2.5)),
        "size_hi": float(np.percentile(sizes, 97.5)),
        "asp_lo": float(np.percentile(asps, 2.5)),
        "asp_hi": float(np.percentile(asps, 97.5)),
    }
    return env


def _excursion(v, lo, hi):
    """Relative distance OUTSIDE [lo,hi]; 0 inside. Normalized by interval halfwidth."""
    if hi <= lo:
        return 0.0
    half = (hi - lo) / 2.0
    mid = (lo + hi) / 2.0
    d = abs(v - mid) - half
    return max(0.0, d) / half


def a4_score(size_ratio, aspect, env):
    """Continuous size/aspect penalty (0 = perfectly inside envelope). Sweepable."""
    es = _excursion(size_ratio, env["size_lo"], env["size_hi"])
    ea = _excursion(aspect, env["asp_lo"], env["asp_hi"])
    return max(es, ea)


# ── matched-N sweep helpers ───────────────────────────────────────────
def pass_at_threshold(recs, score_key, thr, gate_keys=()):
    """Frames whose score_key <= thr AND all gate_keys True. Returns list of e."""
    out = []
    for r in recs:
        s = r.get(score_key)
        if s is None or not np.isfinite(s) or s > thr:
            continue
        if any(not r.get(g, False) for g in gate_keys):
            continue
        out.append(r)
    return out


def bucket(passed):
    e = np.array([r["gt_err"] for r in passed], float)
    if e.size == 0:
        return {"N": 0, "good": 0, "good_pct": 0.0, "gross": 0, "med": None}
    return {
        "N": int(e.size),
        "good": int((e < GOOD_PX).sum()),
        "good_pct": round(100 * float((e < GOOD_PX).mean()), 1),
        "gross": int((e > GROSS_PX).sum()),
        "med": round(float(np.median(e)), 1),
    }


def sweep_to_N(recs, score_key, target_N, gate_keys=()):
    """Find the threshold making |pass| as close to target_N as possible (>= if
    achievable, else max). Returns (chosen_thr, bucket-dict at that thr)."""
    cand = sorted({r[score_key] for r in recs
                   if r.get(score_key) is not None and np.isfinite(r[score_key])
                   and all(r.get(g, False) for g in gate_keys)})
    if not cand:
        return None, bucket([])
    best = None
    for thr in cand:
        p = pass_at_threshold(recs, score_key, thr, gate_keys)
        n = len(p)
        # prefer the smallest threshold whose N >= target (tightest filter that
        # still meets budget); fall back to the largest N if target unreachable.
        key = (abs(n - target_N), thr)
        if best is None or key < best[0]:
            best = (key, thr, p)
        if n >= target_N:
            # smallest thr reaching target — take it and stop (cand sorted asc)
            return thr, bucket(p)
    return best[1], bucket(best[2])


# ── pool pass-count (quantity, from cached records — A1/diag only) ─────
def pool_pass_counts(records_fp):
    if not os.path.exists(records_fp):
        return None
    recs = json.load(open(records_fp))
    if isinstance(recs, dict) and "records" in recs:
        recs = recs["records"]
    out = {}
    for dom in ("outside", "night"):
        d = [r for r in recs if r.get("domain") == dom]
        det = [r for r in d if r.get("n_detected", 0) >= N_DET_MIN]
        diag = [r for r in det if r.get("diag_pass")]
        # flip gate uses cached flip_score if present
        diagflip = [r for r in diag
                    if r.get("flip_score") is not None and r["flip_score"] <= FLIP_TAU]
        out[dom] = {"total": len(d), "detectable": len(det),
                    "diag": len(diag), "diag_flip": len(diagflip)}
    return out


# ── main GPU pass ─────────────────────────────────────────────────────
def run_inference(frames, weights, threshold):
    import cv2
    import torch
    from filter_pr_camfacing import (
        load_model, extract_keypoints_from_belief, filt_diag, hungarian_mean_dist,
    )
    from filter_flip_consistency import infer_flip_kp, flip_consistency_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {weights} ({device})")
    model = load_model(weights, device)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    recs = []
    for n, (dom, fid, jp, ip) in enumerate(frames):
        gt = json.load(open(jp))
        gt8 = np.array(gt["objects"][0]["projected_cuboid"], float)[:8]
        gw = gt8[:, 0].max() - gt8[:, 0].min()
        gh = gt8[:, 1].max() - gt8[:, 1].min()
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        img_diag = float(np.hypot(w0, h0))
        gt_size_ratio = float(np.hypot(gw, gh) / img_diag)
        gt_aspect = float(gw / gh) if gh > 1e-6 else None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # aspect-preserving preprocess (same as tau_calibrate / extract_pl_v1) —
        # squash forbidden (distorts ratio).
        _sc = 400.0 / min(h0, w0)
        _nw = max(8, int(round(w0 * _sc)) & ~7)
        _nh = max(8, int(round(h0 * _sc)) & ~7)
        t = ((cv2.resize(rgb, (_nw, _nh)).astype(np.float32) / 255.0 - mean) / std)
        tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out_bel, _ = model(tensor)
        belief = out_bel[-1][0].cpu().numpy()
        kps_bel = extract_keypoints_from_belief(belief, threshold)
        bh, bw = belief.shape[1], belief.shape[2]
        ux, uy = _nw / bw, _nh / bh
        kp = []
        confs = []
        for k in kps_bel:
            if k[0] < 0:
                kp.append(None)
                confs.append(0.0)
            else:
                kp.append(((k[0] * ux) / _sc, (k[1] * uy) / _sc))
                confs.append(float(k[2]))

        det_xy = [kp[i] for i in range(8) if kp[i] is not None]
        n_det = len(det_xy)
        detected = n_det >= N_DET_MIN
        min_conf = (min(confs[i] for i in range(8) if kp[i] is not None)
                    if n_det else 0.0)

        pred8 = np.full((8, 2), np.nan)
        for i in range(8):
            if kp[i] is not None:
                pred8[i] = kp[i]
        gt_err, _ = hungarian_mean_dist(pred8, gt8)
        if not np.isfinite(gt_err):
            continue  # < 6 corners -> not a candidate PL for any arm

        diag_ok, diag_score = filt_diag(kp)
        sa = _bbox_size_aspect(det_xy)
        if sa is not None:
            pred_size_ratio = sa[0] / img_diag
            pred_aspect = sa[1]
        else:
            pred_size_ratio = pred_aspect = None

        kpB = infer_flip_kp(model, img, device, mean, std, threshold,
                            preprocess="aspect")
        flip_score, _ = flip_consistency_score(kp, kpB)

        recs.append({
            "dom": dom, "frame": fid, "gt_err": float(gt_err),
            "detected": bool(detected), "n_det": n_det,
            "neg_conf": (-float(min_conf) if detected else None),
            "min_conf": float(min_conf),
            "diag_score": (float(diag_score) if np.isfinite(diag_score) else None),
            "flip_score": (None if flip_score is None else float(flip_score)),
            "pred_size_ratio": pred_size_ratio, "pred_aspect": pred_aspect,
            "gt_size_ratio": gt_size_ratio, "gt_aspect": gt_aspect,
        })
        if (n + 1) % 20 == 0:
            print(f"  [{n+1}/{len(frames)}]")
    return recs


def build_arms(recs, env):
    """Attach per-arm continuous scores + the flip gate to each rec."""
    for r in recs:
        # A4 size/aspect penalty (None if not detected enough)
        if r["pred_size_ratio"] is not None and r["pred_aspect"] is not None:
            r["a4_score"] = a4_score(r["pred_size_ratio"], r["pred_aspect"], env)
        else:
            r["a4_score"] = None
        # flip gate for A6/A7
        r["flip_ok"] = (r["flip_score"] is not None and r["flip_score"] <= FLIP_TAU)
        # diag-with-flip score: only meaningful if flip passes; expose diag score
        r["diag_for_a6"] = r["diag_score"] if r["flip_ok"] else None
    return recs


def report(recs, scope_label, out_lines, env):
    sub = recs  # already scoped by caller
    n_det = sum(1 for r in sub if r["detected"])
    n_good = sum(1 for r in sub if r["gt_err"] < GOOD_PX)
    out_lines.append(f"\n=== {scope_label}  (usable={len(sub)} detected={n_det} "
                     f"good={n_good}; good<{GOOD_PX:g} gross>{GROSS_PX:g}) ===")

    # A1 no-filter reference (all detected pass)
    a1 = bucket([r for r in sub if r["detected"]])
    out_lines.append(f"A1 no-filter (ref): N={a1['N']} good={a1['good']} "
                     f"({a1['good_pct']:.0f}%) gross={a1['gross']} med={a1['med']}")

    # anchor N = diag@0.05 (the deployed operating point), with flip gate
    diag05 = pass_at_threshold(sub, "diag_for_a6", 0.05)
    anchor = bucket(diag05)
    out_lines.append(f"anchor: diag(<=0.05)∧flip(<= {FLIP_TAU:g})  N={anchor['N']} "
                     f"good={anchor['good']} gross={anchor['gross']} med={anchor['med']}")

    # common N grid around the anchor
    base = anchor["N"]
    grid = sorted({max(3, base + d) for d in (-4, -2, 0, 2, 4, 8)} |
                  {max(3, base // 2), base})
    grid = [g for g in grid if g <= n_det]  # cannot exceed detected pool
    if not grid:
        grid = [min(n_det, max(3, base))]

    arms = [
        ("A2 conf", "neg_conf", ()),
        ("A4 size/asp", "a4_score", ()),
        ("A6 diag∧flip", "diag_for_a6", ()),
        ("A7 diag+size", "diag_for_a6", ("flip_ok",)),  # size gate handled below
    ]

    out_lines.append(f"\n  matched-N sweep (★ same N → who has fewer GROSS?):")
    hdr = (f"  {'arm':<14}{'N*':>4}{'good':>5}{'good%':>7}{'gross':>6}"
           f"{'med':>6}   (thr)")
    out_lines.append(hdr)
    out_lines.append("  " + "-" * (len(hdr) - 2))
    table = {}
    for N in grid:
        out_lines.append(f"  -- target N≈{N} " + "-" * 20)
        for label, key, gates in arms:
            if label == "A7 diag+size":
                # A7 = diag∧flip AND size-in-envelope; sweep diag score among
                # frames that ALSO pass the size envelope (a4_score==0 => inside)
                pool = [r for r in sub if r.get("a4_score") == 0.0]
                thr, b = sweep_to_N(pool, "diag_for_a6", N)
            elif label == "A6 diag∧flip":
                thr, b = sweep_to_N(sub, "diag_for_a6", N)
            else:
                thr, b = sweep_to_N(sub, key, N)
            table.setdefault(label, {})[N] = {"thr": thr, **b}
            tlabel = f"{thr:.3f}" if thr is not None else "--"
            out_lines.append(
                f"  {label:<14}{b['N']:>4}{b['good']:>5}{b['good_pct']:>6.0f}%"
                f"{b['gross']:>6}{(b['med'] if b['med'] is not None else float('nan')):>6.1f}"
                f"   ({tlabel})")
    return {"anchor": anchor, "a1": a1, "grid": grid, "table": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--pool_records", default=os.path.join(
        ROOT, "data", "pallet", "pl", "stage0_base_v2_sigma2", "_records.json"))
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "stage0_four_arm"))
    ap.add_argument("--list_val_frames", action="store_true")
    args = ap.parse_args()

    frames = collect_val_frames()
    by_dom = {}
    for dom, *_ in frames:
        by_dom[dom] = by_dom.get(dom, 0) + 1
    print(f"[filter-val] {len(frames)} GT frames {by_dom}")
    # SEAL assertion (defense-in-depth; collect_val_frames already guards)
    sm_final = FINAL_TEST_SESSIONS
    for dom, valset in FILTER_VAL_SESSIONS.items():
        assert not (valset & sm_final), "filter-val/final-test overlap!"
    if args.list_val_frames:
        for dom, fid, _, _ in frames:
            print(f"  {dom:<8} {fid}")
        print("[seal] final-test sessions NOT in this set.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    recs = run_inference(frames, args.weights, args.threshold)

    # fit A4 envelope on GT bboxes (model-independent, max GT pass)
    gt_for_env = [{"gt_size_ratio": r["gt_size_ratio"], "gt_aspect": r["gt_aspect"]}
                  for r in recs if r["gt_aspect"] is not None]
    env = fit_a4_envelope(gt_for_env)
    recs = build_arms(recs, env)

    out_lines = []
    out_lines.append("PL-LEVEL 5-ARM COMPARISON @ MATCHED PASS-COUNT "
                     "(filter-val ONLY, sealed)")
    out_lines.append(f"weights={args.weights}")
    out_lines.append(f"A4 envelope (GT cuboid bbox, model-independent): "
                     f"size∈[{env['size_lo']:.3f},{env['size_hi']:.3f}] "
                     f"aspect∈[{env['asp_lo']:.2f},{env['asp_hi']:.2f}]")
    out_lines.append("★ PRELIMINARY (small N) — signal, not proof.")

    outside = [r for r in recs if r["dom"] == "outside"]
    night = [r for r in recs if r["dom"] == "night"]

    rep_out = report(outside, "OUTSIDE (★ primary thesis read)", out_lines, env)
    rep_night = report(night, "NIGHT (depth-collapse limit — honest separate)",
                       out_lines, env)

    # pool pass-counts (quantity)
    pool = pool_pass_counts(args.pool_records)
    if pool:
        out_lines.append("\n[outside POOL pass-count (quantity; NOT GT — cached "
                         "records, A1/diag only — conf/size need pool inference)]")
        for dom in ("outside", "night"):
            p = pool[dom]
            out_lines.append(f"  {dom:<8} total={p['total']} detectable={p['detectable']} "
                             f"diag={p['diag']} diag∧flip={p['diag_flip']}")

    print("\n".join(out_lines))
    txt = os.path.join(args.output_dir, "four_arm_compare.txt")
    open(txt, "w").write("\n".join(out_lines))
    out = {
        "weights": args.weights, "good_px": GOOD_PX, "gross_px": GROSS_PX,
        "flip_tau": FLIP_TAU, "a4_envelope": env,
        "n_frames": len(recs),
        "filter_val_sessions": {k: sorted(v) for k, v in FILTER_VAL_SESSIONS.items()},
        "outside": rep_out, "night": rep_night,
        "pool_pass_counts": pool, "records": recs,
    }
    json.dump(out, open(os.path.join(args.output_dir, "four_arm_compare.json"),
                        "w"), indent=2, default=str)
    print(f"\n[save] {txt}")
    print(f"[save] {os.path.join(args.output_dir, 'four_arm_compare.json')}")
    print("final-test was NOT used (sealed).")


if __name__ == "__main__":
    main()
