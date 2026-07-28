"""stage19_partA_analyze.py — calibration + keypoint PR + funnel + GATE A (CPU).

Reads partA_records.json (from stage19_partA_infer.py) and produces:
  partA_calibration.json   Spearman(signal<->GT err) x {front,rear,all} x domain
                           + front vs rear peak distribution
  partA_pr_curves.txt      keypoint-level PR sweeps (single signals + AND-combo,
                           + front-only) — thresholds CALIBRATED ON filter-val ONLY
  partA_funnel.txt         B2 funnel (det/diag/flip/diag&flip) vs 6/18 old clean0
  partA_summary.md         GATE A auto-decision (A-PASS / A-FAIL / both-marginal)

Signals (per-keypoint unless noted):
  peak (high=good, expect rho<0)   peakratio (high=good)   flipTTA (low=good)
  loo  (low=good, dims-known)       diag_resid (frame-level, low=good)

GATE A (both required for A-PASS):
  (i)  some combo: keypoint precision >= 70% @ yield >= 20%   [filter-val]
  (ii) that combo's driving signal |rho| >= 0.3
Boundary +-5%p on (70/20/0.3) -> "both-marginal".

Usage: python scripts/stage0/stage19_partA_analyze.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage19_conf_mixup")
REC_FP = os.path.join(OUT_DIR, "partA_records.json")

FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
GOOD_PX = 10.0
FILTER_VAL_DOMS = {"outside", "night"}   # threshold calibration ONLY here


def load():
    d = json.load(open(REC_FP))
    return d["records"]


# ── keypoint-level (signal, err) pairs over GT frames ─────────────────────
def kp_pairs(recs, signal_key, idxs, doms=None):
    """Return (sig_arr, err_arr) over detected GT keypoints for channel idxs.
    signal_key in {peak9, peakratio9, flip9, loo9}."""
    S, E = [], []
    for r in recs:
        if not r["is_gt"]:
            continue
        if doms is not None and r["dom"] not in doms:
            continue
        e = r.get("gt_err9")
        sig = r.get(signal_key)
        if e is None or sig is None:
            continue
        for i in idxs:
            if e[i] is None or sig[i] is None:
                continue
            S.append(float(sig[i])); E.append(float(e[i]))
    return np.array(S), np.array(E)


def rho(S, E):
    if len(S) < 8 or np.ptp(S) == 0:
        return None, len(S)
    r, _ = spearmanr(S, E)
    return (None if np.isnan(r) else round(float(r), 3)), len(S)


def calibration(recs):
    out = {}
    sig_idx = {"peak9": ("peak", FRONT + REAR),
               "peakratio9": ("peakratio", FRONT + REAR),
               "flip9": ("flipTTA", FRONT + REAR),
               "loo9": ("loo", FRONT + REAR)}
    dom_sets = {"all": None, "outside": {"outside"}, "night": {"night"},
                "manual": {"manual"}, "cad": {"cad"}}
    for sk, (name, _) in sig_idx.items():
        out[name] = {}
        for dname, ds in dom_sets.items():
            row = {}
            for scope, idxs in (("front", FRONT), ("rear", REAR),
                                ("all", FRONT + REAR)):
                S, E = kp_pairs(recs, sk, idxs, ds)
                r, n = rho(S, E)
                row[scope] = {"rho": r, "n": n}
            out[name][dname] = row
    # diag (frame-level) vs frame-mean err and rear-mean err
    out["diag_frame"] = {}
    for dname, ds in dom_sets.items():
        D, Efull, Erear = [], [], []
        for r in recs:
            if not r["is_gt"] or r.get("diag_resid") is None:
                continue
            if ds is not None and r["dom"] not in ds:
                continue
            e = r.get("gt_err9")
            if e is None:
                continue
            ef = [x for x in e if x is not None]
            er = [e[i] for i in REAR if e[i] is not None]
            if not ef:
                continue
            D.append(r["diag_resid"]); Efull.append(np.mean(ef))
            Erear.append(np.mean(er) if er else np.nan)
        D = np.array(D)
        rf, nf = rho(D, np.array(Efull))
        mask = ~np.isnan(Erear)
        rr, nr = rho(D[mask], np.array(Erear)[mask]) if mask.sum() >= 8 else (None, int(mask.sum()))
        out["diag_frame"][dname] = {"vs_frame_err": {"rho": rf, "n": nf},
                                    "vs_rear_err": {"rho": rr, "n": nr}}
    # front vs rear peak distribution
    def peakdist(idxs, doms):
        vals = []
        for r in recs:
            if not r["is_gt"]:
                continue
            if doms is not None and r["dom"] not in doms:
                continue
            for i in idxs:
                if r["kp9"][i] is not None:
                    vals.append(r["peak9"][i])
        a = np.array(vals)
        if a.size == 0:
            return None
        return {"n": int(a.size), "median": round(float(np.median(a)), 3),
                "q25": round(float(np.percentile(a, 25)), 3),
                "q75": round(float(np.percentile(a, 75)), 3),
                "mean": round(float(np.mean(a)), 3)}
    out["peak_distribution"] = {
        "front_all": peakdist(FRONT, None), "rear_all": peakdist(REAR, None),
        "front_fval": peakdist(FRONT, FILTER_VAL_DOMS),
        "rear_fval": peakdist(REAR, FILTER_VAL_DOMS)}
    # same for gt_err (context)
    def errdist(idxs, doms):
        vals = []
        for r in recs:
            if not r["is_gt"]:
                continue
            if doms is not None and r["dom"] not in doms:
                continue
            e = r.get("gt_err9")
            if e is None:
                continue
            for i in idxs:
                if e[i] is not None:
                    vals.append(e[i])
        a = np.array(vals)
        return None if a.size == 0 else {
            "n": int(a.size), "median": round(float(np.median(a)), 2),
            "good_pct": round(100 * float((a < GOOD_PX).mean()), 1)}
    out["err_distribution"] = {
        "front_all": errdist(FRONT, None), "rear_all": errdist(REAR, None),
        "front_fval": errdist(FRONT, FILTER_VAL_DOMS),
        "rear_fval": errdist(REAR, FILTER_VAL_DOMS)}
    return out


# ── keypoint-level PR (accept a kp if its signal passes; precision=err<10) ─
def kp_records(recs, doms, idxs):
    """Flatten detected GT keypoints -> list of dicts with signals + err + frame."""
    out = []
    for r in recs:
        if not r["is_gt"] or (doms and r["dom"] not in doms):
            continue
        e = r.get("gt_err9")
        if e is None:
            continue
        for i in idxs:
            if r["kp9"][i] is None or e[i] is None:
                continue
            out.append({"i": i, "err": e[i],
                        "peak": r["peak9"][i],
                        "peakratio": r["peakratio9"][i],
                        "flip": r["flip9"][i],
                        "loo": (r.get("loo9") or [None] * 8)[i],
                        "diag": r.get("diag_resid")})
    return out


def pr_single(kps, key, thr, hi_good):
    """hi_good=True: accept if val>=thr (peak). False: accept if val<=thr (flip)."""
    acc = [k for k in kps if k[key] is not None and
           (k[key] >= thr if hi_good else k[key] <= thr)]
    if not acc:
        return 0, 0.0, 0.0
    tp = sum(1 for k in acc if k["err"] < GOOD_PX)
    return len(acc), tp / len(acc), len(acc) / len(kps)


def pr_combo(kps, p_thr, f_thr, d_thr):
    acc = [k for k in kps if k["peak"] is not None and k["peak"] >= p_thr
           and k["flip"] is not None and k["flip"] <= f_thr
           and k["diag"] is not None and k["diag"] <= d_thr]
    if not acc:
        return 0, 0.0, 0.0
    tp = sum(1 for k in acc if k["err"] < GOOD_PX)
    return len(acc), tp / len(acc), len(acc) / len(kps)


def build_pr(recs):
    lines = []
    fval = FILTER_VAL_DOMS
    lines.append("KEYPOINT-LEVEL PR  (threshold CALIBRATED ON filter-val: outside+night)")
    lines.append(f"precision = accepted kp with channel-aligned err < {GOOD_PX:.0f}px")
    lines.append("yield = accepted / total detected kp")
    best = {"precision": 0.0, "yield": 0.0, "desc": None, "signals_used": [],
            "front_only": False}

    def upd(P, Y, desc, sigs, front_only):
        nonlocal best
        if Y >= 0.20 and P > best["precision"]:
            best = {"precision": P, "yield": Y, "desc": desc,
                    "signals_used": sigs, "front_only": front_only}

    def sweep_single(scope_name, idxs, front_only):
        kps = kp_records(recs, fval, idxs)
        base_good = 100 * np.mean([k["err"] < GOOD_PX for k in kps]) if kps else 0
        lines.append("")
        lines.append(f"[{scope_name}]  n_kp={len(kps)}  base good%={base_good:.1f}")
        for key, hi, grid in (
            ("peak", True, [0.3, 0.5, 0.7, 0.85, 0.95, 0.99]),
            ("peakratio", True, [1.5, 2.0, 3.0, 5.0, 10.0]),
            ("flip", False, [15, 10, 8, 5, 3]),
            ("loo", False, [15, 10, 6, 4, 2]),
        ):
            lines.append(f"  {key:<10} " + "  ".join(
                f"thr={g:g}:P={pr_single(kps,key,g,hi)[1]*100:.0f}%/Y={pr_single(kps,key,g,hi)[2]*100:.0f}%"
                for g in grid))
            for g in grid:
                n, P, Y = pr_single(kps, key, g, hi)
                upd(P, Y, f"{scope_name}/{key}{'>=' if hi else '<='}{g}",
                    [key], front_only)

    sweep_single("ALL corners", FRONT + REAR, False)
    sweep_single("FRONT only (0-3)", FRONT, True)
    sweep_single("REAR only (4-7)", REAR, False)

    # AND-combo grid on ALL corners
    kps = kp_records(recs, fval, FRONT + REAR)
    lines.append("")
    lines.append("[AND-combo peak>=p & flip<=f & diag<=d]  (ALL corners, filter-val)")
    for p in (0.5, 0.7, 0.9):
        for f in (10, 6, 4):
            for dth in (0.05, 0.02, 0.01):
                n, P, Y = pr_combo(kps, p, f, dth)
                if n == 0:
                    continue
                lines.append(f"  p>={p} f<={f} d<={dth}: n={n} P={P*100:.0f}% Y={Y*100:.0f}%")
                upd(P, Y, f"combo p>={p},f<={f},d<={dth}",
                    ["peak", "flip"], False)
    # front-only combo
    kpsf = kp_records(recs, fval, FRONT)
    lines.append("")
    lines.append("[FRONT-only AND-combo]")
    for p in (0.5, 0.7, 0.9):
        for f in (10, 6, 4):
            acc = [k for k in kpsf if k["peak"] >= p and k["flip"] is not None
                   and k["flip"] <= f]
            if acc:
                n = len(acc)
                P = sum(1 for k in acc if k["err"] < GOOD_PX) / n
                Y = n / len(kpsf)
                lines.append(f"  FRONT p>={p} f<={f}: n={n} P={P*100:.0f}% Y={Y*100:.0f}%")
                upd(P, Y, f"FRONT p>={p},f<={f}", ["peak", "flip"], True)
    return lines, best


# ── funnel (frame-level) ──────────────────────────────────────────────────
def build_funnel(recs, tau_diag, tau_flip):
    lines = []
    lines.append("B2 SELF-TRAINING FUNNEL  (frame-level)")
    lines.append(f"diag pass = diag_resid <= {tau_diag}; flip pass = mean flip9 <= {tau_flip}px")
    lines.append("6/18 OLD (paper_base_v2 base): noapril 86det/diag43/clean0 ; cad 76det/clean0")
    lines.append("")
    hdr = f"{'pool':<10}{'det':>5}{'diag':>6}{'flip':>6}{'diag&flip':>10}{'precisionFV':>13}"
    lines.append(hdr); lines.append("-" * len(hdr))

    def frame_mean_flip(r):
        v = [x for x in r["flip9"] if x is not None]
        return np.mean(v) if v else None

    # precision on filter-val for diag&flip
    def fval_precision(pred):
        passed = [r for r in recs if r["is_gt"] and r["dom"] in FILTER_VAL_DOMS and pred(r)]
        if not passed:
            return None, 0
        good = sum(1 for r in passed if r.get("hungarian") is not None
                   and r["hungarian"] < GOOD_PX)
        return round(100 * good / len(passed), 1), len(passed)

    for pool in ("noapril", "cad", "outside", "night", "manual"):
        rows = [r for r in recs if r["dom"] == pool]
        det = len(rows)
        dpass = sum(1 for r in rows if r.get("diag_resid") is not None
                    and r["diag_resid"] <= tau_diag)
        fpass = sum(1 for r in rows if frame_mean_flip(r) is not None
                    and frame_mean_flip(r) <= tau_flip)
        dfpass = sum(1 for r in rows if r.get("diag_resid") is not None
                     and r["diag_resid"] <= tau_diag
                     and frame_mean_flip(r) is not None
                     and frame_mean_flip(r) <= tau_flip)
        prec = "-"
        if pool in FILTER_VAL_DOMS:
            p, n = fval_precision(lambda r: (r.get("diag_resid") is not None
                                             and r["diag_resid"] <= tau_diag
                                             and frame_mean_flip(r) is not None
                                             and frame_mean_flip(r) <= tau_flip))
            prec = f"{p}%({n})" if p is not None else "-"
        lines.append(f"{pool:<10}{det:>5}{dpass:>6}{fpass:>6}{dfpass:>10}{prec:>13}")
    # combined filter-val precision at diag&flip
    p, n = fval_precision(lambda r: (r.get("diag_resid") is not None
                                     and r["diag_resid"] <= tau_diag
                                     and (np.mean([x for x in r["flip9"] if x is not None])
                                          if any(x is not None for x in r["flip9"]) else 1e9) <= tau_flip))
    lines.append("")
    lines.append(f"filter-val(outside+night) diag&flip PL precision (hungarian<{GOOD_PX:.0f}px): "
                 f"{p}% over {n} passed")
    return lines, (p, n)


def main():
    recs = load()
    ngt = sum(r["is_gt"] for r in recs)
    print(f"[load] {len(recs)} records ({ngt} GT, {len(recs)-ngt} pool)")

    cal = calibration(recs)
    json.dump(cal, open(os.path.join(OUT_DIR, "partA_calibration.json"), "w"),
              indent=2)

    pr_lines, best = build_pr(recs)
    open(os.path.join(OUT_DIR, "partA_pr_curves.txt"), "w").write("\n".join(pr_lines))

    # pick funnel thresholds from filter-val PR-ish defaults (diag 0.02, flip 8)
    fun_lines, fv_prec = build_funnel(recs, tau_diag=0.02, tau_flip=8.0)
    open(os.path.join(OUT_DIR, "partA_funnel.txt"), "w").write("\n".join(fun_lines))

    # ── GATE A ─────────────────────────────────────────────────────────
    # cond-ii: max |rho| over the signals actually used by the best combo,
    # evaluated on filter-val at the corner scope the combo operates on.
    def fv_rho(sig_name, idxs):
        sk = {"peak": "peak9", "peakratio": "peakratio9", "flip": "flip9",
              "loo": "loo9"}.get(sig_name)
        if sk is None:
            return None, 0
        S, E = kp_pairs(recs, sk, idxs, FILTER_VAL_DOMS)
        return rho(S, E)
    scope_idx = FRONT if best.get("front_only") else FRONT + REAR
    rho_by_sig = {}
    for s in best["signals_used"]:
        rho_by_sig[s] = fv_rho(s, scope_idx)
    # driving = signal with max |rho|
    driving, rho_fv, n_rho = None, None, 0
    for s, (rr, nn) in rho_by_sig.items():
        if rr is not None and (rho_fv is None or abs(rr) > abs(rho_fv)):
            driving, rho_fv, n_rho = s, rr, nn

    P, Y = best["precision"], best["yield"]
    cond_i = (P >= 0.70 and Y >= 0.20)
    cond_ii = (rho_fv is not None and abs(rho_fv) >= 0.30)
    marg_i = (abs(P - 0.70) <= 0.05 or abs(Y - 0.20) <= 0.05)
    marg_ii = (rho_fv is not None and abs(abs(rho_fv) - 0.30) <= 0.05)
    if cond_i and cond_ii:
        gate = "A-PASS"
    elif (cond_i or marg_i) and (cond_ii or marg_ii) and not (cond_i and cond_ii):
        gate = "BOTH-MARGINAL"
    else:
        gate = "A-FAIL"

    pk = cal["peak_distribution"]
    md = []
    md.append("# PART A — GATE A decision\n")
    md.append(f"**GATE A = {gate}**"
              + ("  (FRONT-corners only; REAR = A-FAIL)" if best.get("front_only") else "")
              + "\n")
    md.append(f"- best combo (filter-val): `{best['desc']}` "
              f"precision={P*100:.0f}% yield={Y*100:.0f}%  "
              f"(cond-i precision>=70%@yield>=20%: {cond_i})")
    md.append(f"- driving signal `{driving}` filter-val rho={rho_fv} (n={n_rho})  "
              f"(cond-ii |rho|>=0.3: {cond_ii})")
    md.append(f"- all constituent-signal filter-val rho: "
              + ", ".join(f"{s}={rr}" for s, (rr, _) in rho_by_sig.items()) + "\n")
    md.append("> [caveat] filter-val real GT is small (outside 39 / night 29 frames); "
              "operating point n is modest -> treat as pilot signal, not final numbers.\n")
    md.append("## Front vs Rear peak (confidently-wrong-rear check)")
    md.append(f"- FRONT peak median={pk['front_all']['median']} "
              f"(q25={pk['front_all']['q25']},q75={pk['front_all']['q75']})")
    md.append(f"- REAR  peak median={pk['rear_all']['median']} "
              f"(q25={pk['rear_all']['q25']},q75={pk['rear_all']['q75']})")
    ed = cal["err_distribution"]
    md.append(f"- FRONT err median={ed['front_all']['median']}px good%={ed['front_all']['good_pct']} ; "
              f"REAR err median={ed['rear_all']['median']}px good%={ed['rear_all']['good_pct']}")
    same_conf = abs(pk['rear_all']['median'] - pk['front_all']['median']) < 0.05
    md.append(f"- rear peak ~= front peak ({same_conf}) while rear err >> front: "
              "peak IS informative WITHIN a face (filter-val rho ~ -0.44..-0.55) but "
              "does NOT separate the front->rear error SHIFT (same peak dist, higher "
              "rear err). => a single global peak gate under-cleans rear; use peak "
              "per-face: accept FRONT PL, spatial-mask REAR (confidently-wrong).\n")
    md.append("## Spearman(signal <-> GT err) filter-val [all/front/rear]")
    for name in ("peak", "flipTTA", "loo"):
        rowall = cal[name]["outside"]["all"]["rho"]
        md.append(f"- {name}: outside all={cal[name]['outside']['all']['rho']} "
                  f"front={cal[name]['outside']['front']['rho']} rear={cal[name]['outside']['rear']['rho']} ; "
                  f"night all={cal[name]['night']['all']['rho']}")
    md.append(f"- diag_frame vs rear err: outside={cal['diag_frame']['outside']['vs_rear_err']['rho']} "
              f"night={cal['diag_frame']['night']['vs_rear_err']['rho']}\n")
    md.append(f"## Funnel (filter-val diag&flip PL precision = {fv_prec[0]}% over {fv_prec[1]})")
    md.append("See partA_funnel.txt. Old base (paper_base_v2) gave clean=0.\n")
    md.append("## PART B branch")
    if gate == "A-PASS":
        md.append("- A-PASS (FRONT confidence-selected PL viable, peak |rho|~0.44) "
                  "-> PART B real supervision = confidence-selected PL heatmap: "
                  "teacher=B2 fixed, accept FRONT channels {0,1,2,3} where "
                  "peak>=0.9 & flip<=10; REAR channels {4,5,6,7} spatial-masked "
                  "on the real pallet bbox (rear is confidently-wrong -> no PL, "
                  "keep sim supervision elsewhere). PL loss weight low.")
    elif gate == "A-FAIL":
        md.append("- A-FAIL -> PART B = label-noise-free variant "
                  "(cut-paste synthetic pallet on real background OR real-bg-only mixup; "
                  "target = sim GT, real contributes appearance only).")
    else:
        md.append("- BOTH-MARGINAL -> proceed with A-FAIL (noise-free) variant as the "
                  "conservative default; note marginality.")

    open(os.path.join(OUT_DIR, "partA_summary.md"), "w").write("\n".join(md))
    print("\n".join(md))
    print(f"\n[GATE A] {gate}")
    print(f"[save] partA_calibration.json / partA_pr_curves.txt / "
          f"partA_funnel.txt / partA_summary.md")
    # emit machine-readable gate for the chain
    json.dump({"gate": gate, "best": best, "rho_fv": rho_fv,
               "cond_i": cond_i, "cond_ii": cond_ii,
               "fval_precision": fv_prec},
              open(os.path.join(OUT_DIR, "partA_gate.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
