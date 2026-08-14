"""filter_combo_perkp.py — filter COMBINATION x per-keypoint GT error analysis.

User question: which filter combination passes pseudo-labels whose BACK face
corners (4-7) — not just the centroid — match GT well?  Prior diag check only
validates the centroid (spatial-diagonal intersection ~ centroid), so back-face
errors can slip through.

Reuses data/pallet/eval_results/filter_domain_analysis/_full_s2.json which holds
per frame: kp (predicted 9 = 8 corners + centroid, full-res px or None),
gt8 (GT projected_cuboid 8 corners px), mean_match_px (order-free Hungarian),
n_detected.  Inference-free.

Single filters recomputed from kp (so topbot is available, which _full lacks):
  diag, ratio, fullkp, ransac_loo, topbot.
All meaningful AND combinations are then evaluated.

Per passed PL we compute order-free Hungarian per-corner GT error and report:
  pass count, good% (<good_px), overall reproj median,
  FRONT(0-3) median, BACK(4-7) median, CTR(8) median.

Excludes frames in data/_eval_sets/_exclude.txt.
"""
import os as _os, sys as _sys

# --- data_prep 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 먼저 실행돼야 하므로 최상단에 둔다.
_DP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_DP] + [_os.path.join(_DP, _d) for _d in sorted(_os.listdir(_DP))
                         if _os.path.isdir(_os.path.join(_DP, _d)) and not _d.startswith(".")]

import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

from filter_pr_camfacing import (  # noqa: E402
    canonical_kp3d, filt_diag, filt_ratio, filt_fullkp, filt_topbot,
    ransac_consensus, loo_stability,
)

DOMAINS = ["indoor", "outside", "night"]
SINGLES = ["diag", "ratio", "fullkp", "ransac_loo", "topbot"]
GROSS_PX = 20.0
RANSAC_C = 6

# meaningful combinations to evaluate (besides singles).
# focus: combos that could tighten BACK-face accuracy.
COMBOS = [
    ("diag",),
    ("ratio",),
    ("fullkp",),
    ("ransac_loo",),
    ("topbot",),
    # 2-way
    ("diag", "ratio"),
    ("diag", "fullkp"),
    ("diag", "ransac_loo"),
    ("diag", "topbot"),
    ("ratio", "fullkp"),
    ("ratio", "topbot"),
    ("ratio", "ransac_loo"),
    ("fullkp", "ransac_loo"),
    ("topbot", "ransac_loo"),
    # 3-way
    ("diag", "ratio", "fullkp"),
    ("diag", "ratio", "topbot"),
    ("diag", "ratio", "ransac_loo"),
    ("diag", "fullkp", "ransac_loo"),
    ("diag", "ratio", "fullkp", "topbot"),
    # 4-5 way
    ("diag", "ratio", "fullkp", "ransac_loo"),
    ("diag", "ratio", "topbot", "ransac_loo"),
    ("diag", "ratio", "fullkp", "topbot", "ransac_loo"),
]


def assign_corners(kp, gt8):
    """order-free Hungarian: predicted 8 corners -> GT 8 corners.
    returns (mean_px, dists[8] indexed by predicted slot, nan if missing)."""
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i] is not None:
            pred8[i] = kp[i]
    valid = ~np.isnan(pred8[:, 0])
    dists = np.full(8, np.nan)
    if valid.sum() < 6:
        return float("inf"), dists
    idx_valid = np.where(valid)[0]
    P = pred8[valid]
    G = np.asarray(gt8, float)
    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        dists[idx_valid[r]] = cost[r, c]
    return float(cost[ri, ci].mean()), dists


def centroid_err(kp, gt8):
    if kp[8] is None:
        return np.nan
    gc = np.asarray(gt8, float).mean(axis=0)
    return float(np.linalg.norm(np.asarray(kp[8], float) - gc))


def load_exclude():
    fp = os.path.join(ROOT, "data", "_eval_sets", "_exclude.txt")
    ex = set()
    if os.path.exists(fp):
        for ln in open(fp):
            ln = ln.split("#")[0].strip()
            if ln:
                ex.add(ln)
    return ex


def recompute_singles(rows, intr_K_lookup):
    """Add per-row single-filter booleans recomputed from kp (incl topbot).
    ransac_loo needs K; we don't have intrinsics in _full, so reuse the stored
    'ransac_loo' boolean if present, else recompute with a generic K guess.
    Here _full already stores ransac_loo -> trust it; recompute the rest."""
    for r in rows:
        kp = [tuple(p) if p is not None else None for p in r["kp"]]
        r["_single"] = {
            "diag": bool(filt_diag(kp)[0]),
            "ratio": bool(filt_ratio(kp)[0]),
            "fullkp": bool(filt_fullkp(kp)[0]),
            "topbot": bool(filt_topbot(kp)[0]),
            # ransac_loo recomputation requires K (not stored). reuse stored.
            "ransac_loo": bool(r["filters"].get("ransac_loo", False)),
        }


def median(a):
    return float(np.median(a)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--good_px", type=float, default=10.0)
    ap.add_argument("--min_pass", type=int, default=20,
                    help="min pass count to be self-training viable")
    ap.add_argument("--output_dir", default=os.path.join(
        ROOT, "data", "pallet", "eval_results", "filter_combo_perkp"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fp = os.path.join(ROOT, "data", "pallet", "eval_results",
                      "filter_domain_analysis", f"_full_{args.tag}.json")
    data = json.load(open(fp))
    exclude = load_exclude()

    result = {}
    for dom in DOMAINS:
        rows = [r for r in data[dom] if str(r["frame"]) not in exclude]
        recompute_singles(rows, None)
        det = [r for r in rows if r["mean_match_px"] is not None]  # >=6 corners
        result[dom] = {
            "total": len(rows), "detectable": len(det),
            "good_overall": sum(1 for r in det if r["mean_match_px"] < args.good_px),
            "combos": {},
        }
        for combo in COMBOS:
            passed = [r for r in det
                      if all(r["_single"][f] for f in combo)]
            if not passed:
                result[dom]["combos"]["+".join(combo)] = {
                    "n": 0, "good": 0, "good_pct": None,
                    "reproj_med": None, "front_med": None,
                    "back_med": None, "ctr_med": None, "per_kp_med": None,
                    "gross_pass": 0}
                continue
            overall, fronts, backs, ctrs, perkp = [], [], [], [], []
            n_good = n_gross = 0
            for r in passed:
                e = r["mean_match_px"]
                overall.append(e)
                if e < args.good_px:
                    n_good += 1
                if e > GROSS_PX:
                    n_gross += 1
                _, d8 = assign_corners(r["kp"], r["gt8"])
                ce = centroid_err(r["kp"], r["gt8"])
                fronts.append(np.nanmedian(d8[:4]))
                backs.append(np.nanmedian(d8[4:8]))
                ctrs.append(ce)
                perkp.append(np.concatenate([d8, [ce]]))
            perkp = np.array(perkp)
            result[dom]["combos"]["+".join(combo)] = {
                "n": len(passed),
                "good": n_good,
                "good_pct": round(100 * n_good / len(passed), 1),
                "reproj_med": round(median(overall), 1),
                "front_med": round(np.nanmedian(fronts), 1),
                "back_med": round(np.nanmedian(backs), 1),
                "ctr_med": round(np.nanmedian(ctrs), 1),
                "gross_pass": n_gross,
                "per_kp_med": [None if not np.isfinite(v) else round(float(v), 1)
                               for v in np.nanmedian(perkp, axis=0)],
            }

    # ── console tables ───────────────────────────────────────────────
    for dom in DOMAINS:
        R = result[dom]
        print("\n" + "=" * 96)
        print(f"DOMAIN={dom}  total={R['total']}  detectable(>=6kp)={R['detectable']}"
              f"  good_overall(<{args.good_px:.0f}px)={R['good_overall']}")
        print("=" * 96)
        hdr = (f"{'combo':<34}{'N':>4}{'good%':>7}{'reproj':>8}"
               f"{'FRONT':>8}{'BACK':>8}{'ctr':>7}{'gross':>7}")
        print(hdr); print("-" * len(hdr))
        # sort: viable (n>=min_pass) first, then by back_med asc
        items = list(R["combos"].items())
        def key(it):
            v = it[1]
            viable = 0 if (v["n"] >= args.min_pass) else 1
            bm = v["back_med"] if v["back_med"] is not None else 1e9
            return (viable, bm)
        for name, v in sorted(items, key=key):
            if v["n"] == 0:
                print(f"{name:<34}{0:>4}{'   --':>7}{'   --':>8}"
                      f"{'   --':>8}{'   --':>8}{'  --':>7}{'  --':>7}")
                continue
            flag = "" if v["n"] >= args.min_pass else " *low"
            print(f"{name:<34}{v['n']:>4}{v['good_pct']:>6.0f}%"
                  f"{v['reproj_med']:>8.1f}{v['front_med']:>8.1f}"
                  f"{v['back_med']:>8.1f}{v['ctr_med']:>7.1f}"
                  f"{v['gross_pass']:>7}{flag}")

    out = os.path.join(args.output_dir, f"combo_perkp_{args.tag}.json")
    json.dump({"good_px": args.good_px, "min_pass": args.min_pass,
               "gross_px": GROSS_PX, "result": result},
              open(out, "w"), indent=2)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
