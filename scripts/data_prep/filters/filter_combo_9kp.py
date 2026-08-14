"""filter_combo_9kp.py — filter combination x OVERALL 9-keypoint GT error.

User question (corrected 2026-06-04): which filter condition/combination passes
pseudo-labels whose FULL 9 keypoints (8 corners + centroid) match GT best on
AVERAGE?  Selection criterion = OVERALL 9-kp mean order-free error (NOT back-face).
Front/back/centroid breakdown is reference-only.

Reuses data/pallet/eval_results/filter_domain_analysis/_full_s2.json (inference-free):
  kp  = predicted 9 (8 corners + centroid, full-res px or None)
  gt8 = GT projected_cuboid 8 corners px (no centroid -> use mean of 8 corners)
  mean_match_px = 8-corner order-free Hungarian mean (precomputed)

Per passed PL:
  - 8 corners matched order-free (Hungarian) to GT 8 corners -> 8 distances
  - centroid(idx8) vs GT cuboid center (mean of 8 GT corners) -> 1 distance
  - 9kp_err = mean of the available distances among those 9
Report per domain (+ ALL): pass N, 9kp mean/median, good%(<good_px on 9kp),
  gross%(>20px), and (reference) front(0-3)/back(4-7)/centroid medians.

Selection = min 9kp median with pass N >= viability threshold.
Excludes frames in data/_eval_sets/_exclude.txt.
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

from filter_pr_camfacing import (  # noqa: E402
    filt_diag, filt_ratio, filt_fullkp, filt_topbot,
)

DOMAINS = ["indoor", "outside", "night"]
GROSS_PX = 20.0

COMBOS = [
    ("diag",), ("ratio",), ("fullkp",), ("ransac_loo",), ("topbot",),
    ("diag", "ratio"), ("diag", "fullkp"), ("diag", "ransac_loo"),
    ("diag", "topbot"), ("ratio", "fullkp"), ("ratio", "topbot"),
    ("ratio", "ransac_loo"), ("fullkp", "ransac_loo"), ("topbot", "ransac_loo"),
    ("diag", "ratio", "fullkp"), ("diag", "ratio", "topbot"),
    ("diag", "ratio", "ransac_loo"), ("diag", "fullkp", "ransac_loo"),
    ("diag", "fullkp", "topbot"), ("diag", "ratio", "fullkp", "topbot"),
    ("diag", "ratio", "fullkp", "ransac_loo"),
    ("diag", "ratio", "topbot", "ransac_loo"),
    ("diag", "ratio", "fullkp", "topbot", "ransac_loo"),
]


def assign_corners(kp, gt8):
    """order-free Hungarian pred 8 corners -> GT 8 corners. dists[8] by pred slot."""
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    dists = np.full(8, np.nan)
    if valid.sum() < 6:
        return dists
    idx_valid = np.where(valid)[0]
    P = pred8[valid]
    G = np.asarray(gt8, float)
    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        dists[idx_valid[r]] = cost[r, c]
    return dists


def centroid_err(kp, gt8):
    if kp[8] is None:
        return np.nan
    gc = np.asarray(gt8, float).mean(axis=0)
    return float(np.linalg.norm(np.asarray(kp[8], float) - gc))


def nine_kp_err(kp, gt8):
    """mean over available of the 9 (8 Hungarian corner dists + centroid dist)."""
    d8 = assign_corners(kp, gt8)
    ce = centroid_err(kp, gt8)
    all9 = np.concatenate([d8, [ce]])
    if np.all(np.isnan(all9)):
        return np.nan, d8, ce
    return float(np.nanmean(all9)), d8, ce


def load_exclude():
    fp = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
    ex = set()
    if os.path.exists(fp):
        for ln in open(fp):
            ln = ln.split("#")[0].strip()
            if ln:
                ex.add(ln)
    return ex


def recompute_singles(rows):
    for r in rows:
        kp = [tuple(p) if p is not None else None for p in r["kp"]]
        r["_single"] = {
            "diag": bool(filt_diag(kp)[0]),
            "ratio": bool(filt_ratio(kp)[0]),
            "fullkp": bool(filt_fullkp(kp)[0]),
            "topbot": bool(filt_topbot(kp)[0]),
            "ransac_loo": bool(r["filters"].get("ransac_loo", False)),
        }


def nanmed(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return float(np.median(a)) if a else float("nan")


def summarize(passed, good_px):
    """passed = list of (err9, d8, ce, e8corner). returns stats dict."""
    if not passed:
        return {"n": 0}
    err9 = np.array([p[0] for p in passed])
    fronts = [np.nanmedian(p[1][:4]) for p in passed]
    backs = [np.nanmedian(p[1][4:8]) for p in passed]
    ctrs = [p[2] for p in passed]
    n_good = int(np.sum(err9 < good_px))
    n_gross = int(np.sum(err9 > GROSS_PX))
    return {
        "n": len(passed),
        "err9_mean": round(float(np.mean(err9)), 1),
        "err9_med": round(float(np.median(err9)), 1),
        "good": n_good,
        "good_pct": round(100 * n_good / len(passed), 1),
        "gross_pass": n_gross,
        "front_med": round(nanmed(fronts), 1),
        "back_med": round(nanmed(backs), 1),
        "ctr_med": round(nanmed(ctrs), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--good_px", type=float, default=10.0)
    ap.add_argument("--min_pass", type=str, default="indoor:20,outside:20,night:8,ALL:30")
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "filter_combo_9kp"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    min_pass = {}
    for tok in args.min_pass.split(","):
        k, v = tok.split(":")
        min_pass[k] = int(v)

    fp = os.path.join(ROOT, "data", "pallet", "eval_results",
                      "filter_domain_analysis", f"_full_{args.tag}.json")
    data = json.load(open(fp))
    exclude = load_exclude()

    # build per-domain detectable rows with single-filter flags + 9kp err cached
    dom_rows = {}
    all_rows = []
    for dom in DOMAINS:
        rows = [r for r in data[dom] if str(r["frame"]) not in exclude]
        recompute_singles(rows)
        det = [r for r in rows if r["mean_match_px"] is not None]
        for r in det:
            e9, d8, ce = nine_kp_err(r["kp"], r["gt8"])
            r["_e9"] = e9
            r["_d8"] = d8
            r["_ce"] = ce
            r["_dom"] = dom
        det = [r for r in det if r["_e9"] is not None and np.isfinite(r["_e9"])]
        dom_rows[dom] = {"total": len(rows), "det": det}
        all_rows.extend(det)

    scopes = [(d, dom_rows[d]["det"], dom_rows[d]["total"]) for d in DOMAINS]
    scopes.append(("ALL", all_rows, sum(dom_rows[d]["total"] for d in DOMAINS)))

    result = {}
    for scope, det, total in scopes:
        good_overall = sum(1 for r in det if r["_e9"] < args.good_px)
        entry = {"total": total, "detectable": len(det),
                 "good_overall_9kp": good_overall, "combos": {}}
        for combo in COMBOS:
            passed = [(r["_e9"], r["_d8"], r["_ce"])
                      for r in det if all(r["_single"][f] for f in combo)]
            entry["combos"]["+".join(combo)] = summarize(passed, args.good_px)
        result[scope] = entry

    # ── console tables ───────────────────────────────────────────────
    lines = []
    for scope, det, total in scopes:
        R = result[scope]
        mp = min_pass.get(scope, 20)
        lines.append("")
        lines.append("=" * 92)
        lines.append(f"SCOPE={scope}  total={R['total']}  detectable(>=6kp)={R['detectable']}"
                     f"  good_overall_9kp(<{args.good_px:.0f}px)={R['good_overall_9kp']}"
                     f"  [viable N>={mp}]")
        lines.append("=" * 92)
        hdr = (f"{'combo':<34}{'N':>4}{'9kp_med':>8}{'9kp_mn':>8}"
               f"{'good%':>7}{'gross':>6}  {'|(ref) front':>12}{'back':>7}{'ctr':>6}")
        lines.append(hdr)
        lines.append("-" * len(hdr))

        def key(it):
            v = it[1]
            if v["n"] == 0:
                return (2, 1e9)
            viable = 0 if v["n"] >= mp else 1
            return (viable, v["err9_med"])
        for name, v in sorted(R["combos"].items(), key=key):
            if v["n"] == 0:
                lines.append(f"{name:<34}{0:>4}{'  --':>8}{'  --':>8}"
                             f"{'  --':>7}{'  --':>6}  {'--':>12}{'--':>7}{'--':>6}")
                continue
            flag = "" if v["n"] >= mp else " *low"
            lines.append(f"{name:<34}{v['n']:>4}{v['err9_med']:>8.1f}{v['err9_mean']:>8.1f}"
                         f"{v['good_pct']:>6.0f}%{v['gross_pass']:>6}  "
                         f"{v['front_med']:>12.1f}{v['back_med']:>7.1f}{v['ctr_med']:>6.1f}{flag}")
        # best viable
        viable = [(n, v) for n, v in R["combos"].items()
                  if v["n"] >= mp]
        if viable:
            bn, bv = min(viable, key=lambda x: x[1]["err9_med"])
            lines.append(f"--> BEST viable (min 9kp_med, N>={mp}): {bn}"
                         f"  N={bv['n']}  9kp_med={bv['err9_med']}  good%={bv['good_pct']}")
    print("\n".join(lines))

    out = os.path.join(args.output_dir, f"combo_9kp_{args.tag}.json")
    json.dump({"good_px": args.good_px, "min_pass": min_pass,
               "gross_px": GROSS_PX, "result": result},
              open(out, "w"), indent=2)
    txt = os.path.join(args.output_dir, f"combo_9kp_{args.tag}.txt")
    open(txt, "w").write("\n".join(lines))
    print(f"\n[save] {out}\n[save] {txt}")


if __name__ == "__main__":
    main()
