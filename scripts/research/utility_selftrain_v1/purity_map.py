"""utility_selftrain_v1 sections 5 and 6 — how GT-free features relate to label purity.

Runs on POLICY_DEV only.  Bin rules come from METHOD_LOCK_UTILITY_ST.json and were
fixed before this script was first run.  GT columns are read here for analysis; they
never enter a selection rule.

This stage answers "where does purity change", not "which threshold to ship".
"""
import csv, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "_docs/experiments/utility_selftrain_v1"
FEATURES = ROOT / "data/pallet/results/utility_selftrain_v1/FEATURES_LABELED.csv"

GROSS, CATASTROPHIC = 20.0, 40.0
CONF_BINS = [(0.0, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
QUANTILE_FEATURES = ["bbox_diag_frac", "max_geom_f4", "pred_elev_deg",
                     "pred_depth_m", "flip_box_iou", "kp_conf_min", "proj_diag_frac"]
TERCILE_FEATURES = ["pred_yaw_c2_deg"]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def purity(rows):
    errs_med = [num(r["gt_corner_median_px"]) for r in rows]
    errs_max = [num(r["gt_corner_max_px"]) for r in rows]
    errs_med = [e for e in errs_med if e is not None]
    errs_max = [e for e in errs_max if e is not None]
    gross = sum(int(r["gt_gross_gt20"]) > 0 for r in rows if r.get("gt_gross_gt20"))
    cat = sum(int(r["gt_catastrophic_gt40"]) > 0 for r in rows if r.get("gt_catastrophic_gt40"))
    acc20 = sum(1 for e in errs_max if e <= 20.0)
    acc10 = sum(1 for e in errs_max if e <= 10.0)
    n = len(rows)
    return {
        "n": n,
        "corner_median_px": st.median(errs_med) if errs_med else None,
        "corner_max_median_px": st.median(errs_max) if errs_max else None,
        "frac_frames_with_gross_gt20": gross / n if n else None,
        "frac_frames_with_catastrophic_gt40": cat / n if n else None,
        "frac_accurate_max_le_20px": acc20 / n if n else None,
        "frac_accurate_max_le_10px": acc10 / n if n else None,
    }


def quantile_edges(values, k):
    v = sorted(values)
    return [v[int(i * (len(v) - 1) / k)] for i in range(1, k)]


def bin_by_edges(rows, key, edges):
    out = {}
    labels = ([f"Q1 <{edges[0]:.4g}"]
              + [f"Q{i+2} {edges[i]:.4g}-{edges[i+1]:.4g}" for i in range(len(edges) - 1)]
              + [f"Q{len(edges)+1} >={edges[-1]:.4g}"])
    for r in rows:
        v = num(r[key])
        if v is None:
            out.setdefault("missing", []).append(r)
            continue
        i = 0
        while i < len(edges) and v >= edges[i]:
            i += 1
        out.setdefault(labels[i], []).append(r)
    return out


def threshold_sweep(rows, taus, confs, tau_fixed, conf_fixed):
    """Section 6.  Purity of the accepted set as each threshold moves, on POLICY_DEV."""
    def cell(sel):
        acc = [r for r in rows if sel(r)]
        rej = [r for r in rows if not sel(r)]
        acc_ok = sum(1 for r in acc if (num(r["gt_corner_max_px"]) or 1e9) <= 20.0)
        rej_ok = sum(1 for r in rej if (num(r["gt_corner_max_px"]) or 1e9) <= 20.0)
        return {
            "accepted": len(acc), "coverage": len(acc) / len(rows) if rows else None,
            "accepted_accurate": acc_ok, "accepted_wrong": len(acc) - acc_ok,
            "rejected_accurate": rej_ok, "rejected_wrong": len(rej) - rej_ok,
            "purity": acc_ok / len(acc) if acc else None,
            "pass": purity(acc),
        }

    def passes(r, tau, conf):
        c = num(r["box_conf"])
        m, f = num(r["s_remove"]), num(r["s_flip"])
        return (c is not None and c >= conf and m is not None and m <= tau
                and f is not None and f <= tau)

    return {
        "geometry_tau": {f"{t:.2f}": cell(lambda r, t=t: passes(r, t, conf_fixed))
                         for t in taus},
        "box_confidence": {f"{c:.2f}": cell(lambda r, c=c: passes(r, tau_fixed, c))
                           for c in confs},
    }


def main() -> int:
    rows = [r for r in csv.DictReader(FEATURES.open()) if r["role"] == "POLICY_DEV"]
    lock = json.loads((ROOT / "data/evaluation/pallet_eval_v1/adaptation/"
                       "PSEUDOLABEL_FILTER_LOCK.json").read_text())
    tau = lock["geometry_thresholds"]["tau_remove"]
    tau_box = lock["TAU_BOX"]

    report = {
        "schema_version": "utility_selftrain_v1_feature_purity_map",
        "role": "POLICY_DEV",
        "n_frames": len(rows),
        "sessions": sorted({r["session"] for r in rows}),
        "gt_used_for": "analysis only",
        "binning_source": "METHOD_LOCK_UTILITY_ST.json",
        "overall": purity(rows),
        "features": {},
    }

    for key in QUANTILE_FEATURES:
        vals = [num(r[key]) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 8:
            report["features"][key] = {"skipped": "fewer than 8 usable values"}
            continue
        edges = quantile_edges(vals, 4)
        report["features"][key] = {
            "bin_rule": "quartiles", "edges": edges,
            "bins": {k: purity(v) for k, v in bin_by_edges(rows, key, edges).items()},
        }

    for key in TERCILE_FEATURES:
        vals = [num(r[key]) for r in rows if num(r[key]) is not None]
        edges = quantile_edges(vals, 3)
        report["features"][key] = {
            "bin_rule": "tertiles", "edges": edges,
            "bins": {k: purity(v) for k, v in bin_by_edges(rows, key, edges).items()},
        }

    conf_bins = {}
    for lo, hi in CONF_BINS:
        sel = [r for r in rows if (num(r["box_conf"]) is not None
                                   and lo <= num(r["box_conf"]) < hi)]
        conf_bins[f"[{lo:.2f},{hi:.2f})"] = purity(sel)
    report["features"]["box_conf"] = {"bin_rule": "fixed", "bins": conf_bins}

    sweep = threshold_sweep(rows, [0.01, 0.02, 0.03, 0.04, 0.05],
                            [0.70, 0.75, 0.80, 0.85, 0.90, 0.95], tau, tau_box)
    sweep_report = {
        "schema_version": "utility_selftrain_v1_threshold_purity_sweep",
        "role": "POLICY_DEV", "n_frames": len(rows),
        "held_fixed": {"box_conf": tau_box, "geometry_tau": tau},
        "note": "purity analysis only.  The highest-purity threshold here is NOT adopted "
                "as the policy; instruction section 6 forbids that.",
        "sweep": sweep,
    }

    (DOCS / "FEATURE_PURITY_MAP.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (DOCS / "THRESHOLD_PURITY_SWEEP.json").write_text(json.dumps(sweep_report, indent=2, ensure_ascii=False))

    o = report["overall"]
    print(f"POLICY_DEV n={o['n']}  corner med={o['corner_median_px']:.2f}px  "
          f"accurate(max<=20px)={o['frac_accurate_max_le_20px']:.3f}  "
          f"gross frames={o['frac_frames_with_gross_gt20']:.3f}")
    for key, d in report["features"].items():
        if "bins" not in d:
            continue
        print(f"\n-- {key} ({d['bin_rule']})")
        print(f"     {'bin':26s}{'n':>4s}{'cornerMed':>11s}{'acc<=20px':>11s}{'gross':>8s}{'cat':>8s}")
        for b, p in d["bins"].items():
            fmt = lambda v, f="{:.3f}": "--" if v is None else f.format(v)
            print(f"     {b:26s}{p['n']:>4d}{fmt(p['corner_median_px'],'{:.2f}'):>11s}"
                  f"{fmt(p['frac_accurate_max_le_20px']):>11s}"
                  f"{fmt(p['frac_frames_with_gross_gt20']):>8s}"
                  f"{fmt(p['frac_frames_with_catastrophic_gt40']):>8s}")

    print("\n=== threshold sweep (POLICY_DEV) ===")
    for axis, cells in sweep.items():
        print(f"-- {axis}")
        print(f"   {'value':>7s}{'accepted':>9s}{'cover':>7s}{'purity':>8s}"
              f"{'acc_ok':>8s}{'acc_bad':>8s}{'rej_ok':>8s}{'passMed':>9s}")
        for v, c in cells.items():
            fmt = lambda x, f="{:.3f}": "--" if x is None else f.format(x)
            print(f"   {v:>7s}{c['accepted']:>9d}{fmt(c['coverage']):>7s}{fmt(c['purity']):>8s}"
                  f"{c['accepted_accurate']:>8d}{c['accepted_wrong']:>8d}"
                  f"{c['rejected_accurate']:>8d}{fmt(c['pass']['corner_median_px'],'{:.2f}'):>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
