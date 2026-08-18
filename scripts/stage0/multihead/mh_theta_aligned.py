"""PHASE 4-5 -- THETA_ONLY_POSE_ALIGNED_SELECTION.

A new, separately named experiment.  It does not overwrite anything: the
historical verdict stays

    THETA_ONLY_LINE_USEFUL = False

and this asks a different question -- whether that failure was the selection rule
rather than the line information.  It is exploratory by construction, because the
rule below was written after seeing the earlier result, which is exactly why it is
confirmed on D4 and not on the populations that produced that result.

## What changes

The old rule minimised rotation median among the survivors.  Rotation is the axis
the line term is good at, so the rule walked seed 1 to the edge of its grid and
translation paid for it.  The new objective scores both axes in relative units:

    J(lambda) = sqrt( (R_lambda / R_point) * (t_lambda / t_point) )

a geometric mean of the two relative improvements, minimised.  Ties go to the
smaller lambda.  The safety filter is unchanged.

## What does not change

The candidate grid is the locked `mh_theta.LAMBDA_GRID`, the solver is
`mh_theta.solve_theta`, and the selection reads the D0 table that PHASE 4 already
measured -- no frame is re-solved to pick the weight, and D0 is train-side, so
nothing here touches a held-out population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_theta as TH                                            # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
CONFIRM = "D4_THETA_CONFIRM512"

# Locked before D4 was read.  Identical thresholds to the earlier screen so the
# only difference between the two experiments is the selection objective.
SAFETY_T_DEGRADE_PCT = 3.0
SAFETY_SUCCESS_DROP_PP = 0.0
SAFETY_SOLVE_DROP_PP = 1.0
GATE = {"ALL_R_gain_pct": 5.0, "ALL_t_degrade_pct": 3.0,
        "Vlt8_R_gain_pct": 10.0, "Vlt8_t_degrade_pct": 5.0}


def log(message):
    print(message, flush=True)


def run_d0(_arguments):
    """Re-score the existing D0 table under the pose-aligned objective."""
    table = json.loads((OUT / "theta_only_solver_d0.json").read_text())
    manifest = (OUT / f"{CONFIRM.lower()}_manifest.json")
    sha = hashlib.sha256(manifest.read_text().encode()).hexdigest()
    result = {"experiment": "THETA_ONLY_POSE_ALIGNED_SELECTION",
              "exploratory": True,
              "historical_verdict_preserved":
                  "THETA_ONLY_LINE_USEFUL = False (unchanged)",
              "grid": table["grid"],
              "objective": "minimise sqrt((R_lam/R_point)*(t_lam/t_point)); "
                           "ties to the smaller lambda",
              "safety": {"t_degrade_pct": SAFETY_T_DEGRADE_PCT,
                         "success_drop_pp": SAFETY_SUCCESS_DROP_PP,
                         "solve_drop_pp": SAFETY_SOLVE_DROP_PP},
              "gate": GATE,
              "confirmation_population": CONFIRM,
              "confirmation_manifest_sha256": sha,
              "seeds": {}}
    for seed in SEEDS:
        block = table["seeds"][f"seed{seed}"]
        base = block["P0_point_only"]
        entry = {"P0_point_only": base, "candidates": {}}
        log(f"seed{seed}  P0  R {base['R_median']:.4f} t {base['t_median']:.5f} "
            f"5cm5 {base['success_5cm5deg']:.4f}")
        survivors = []
        for name, candidate in block["candidates"].items():
            reasons = []
            t_gain = 100.0 * (base["t_median"] - candidate["t_median"]) \
                / base["t_median"]
            if t_gain < -SAFETY_T_DEGRADE_PCT:
                reasons.append("t")
            if (candidate["success_5cm5deg"] - base["success_5cm5deg"]) \
                    < -SAFETY_SUCCESS_DROP_PP / 100.0:
                reasons.append("5cm5deg")
            if (base["solve_rate"] - candidate["solve_rate"]) \
                    > SAFETY_SOLVE_DROP_PP / 100.0:
                reasons.append("solve")
            objective = math.sqrt(
                (candidate["R_median"] / base["R_median"])
                * (candidate["t_median"] / base["t_median"]))
            record = {"R_median": candidate["R_median"],
                      "t_median": candidate["t_median"],
                      "success_5cm5deg": candidate["success_5cm5deg"],
                      "solve_rate": candidate["solve_rate"],
                      "J": round(objective, 6),
                      "rejected_for": reasons}
            entry["candidates"][name] = record
            if not reasons:
                survivors.append((objective, float(name)))
            log(f"  lam {float(name):<5g} R {candidate['R_median']:8.4f} "
                f"t {candidate['t_median']:.5f} J {objective:.5f} "
                f"{'OK' if not reasons else 'reject:' + ','.join(reasons)}")
        if survivors:
            survivors.sort(key=lambda pair: (round(pair[0], 6), pair[1]))
            entry["selected_lambda_theta"] = survivors[0][1]
            entry["NO_SAFE_LAMBDA"] = False
        else:
            entry["selected_lambda_theta"] = None
            entry["NO_SAFE_LAMBDA"] = True
        log(f"  -> selected {entry['selected_lambda_theta']}  "
            f"(old rule chose {block['selected_lambda_theta']})")
        entry["old_rule_selected"] = block["selected_lambda_theta"]
        result["seeds"][f"seed{seed}"] = entry
    path = OUT / "theta_posealigned_d0.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}   D4 sha {sha[:16]}...")


def run_eval(_arguments):
    """One reading of D4 with the weight already locked on D0."""
    d0 = json.loads((OUT / "theta_posealigned_d0.json").read_text())
    full = json.loads(
        (OUT / f"point_line_solver_{TH.RUN}.json").read_text())
    meta = {r["stem"]: r for r in MD.load_split()}
    for seed in SEEDS:
        weight = d0["seeds"][f"seed{seed}"]["selected_lambda_theta"]
        if weight is None:
            log(f"seed{seed}: no safe lambda")
            continue
        data, rows = TH.score_population(
            seed, CONFIRM, weight, arms=("T0", "T1", "T2"),
            full_line_weight=float(full[f"seed{seed}"]["lambda"]))
        subsets = DG._frame_subsets(data, meta)
        entry = {arm: {} for arm in ("T0", "T1", "T2")}
        for label, mask in subsets.items():
            for arm in ("T0", "T1", "T2"):
                entry[arm][label] = TH.summarise(rows[arm], mask)
        block = {"lambda_theta": weight, "population": CONFIRM,
                 "subsets": entry}
        np.savez_compressed(
            OUT / f"theta_posealigned_frames_seed{seed}.npz",
            **{f"{arm}_{key}": rows[arm][key]
               for arm in rows for key in ("R", "t", "solved")},
            **{f"mask_{label}": mask for label, mask in subsets.items()})
        path = OUT / f"theta_posealigned_d4_seed{seed}.json"
        path.write_text(json.dumps(block, indent=1))
        allrow = entry
        log(f"seed{seed} lam {weight}  "
            f"T0 R {allrow['T0']['ALL']['R_median']:7.3f} "
            f"t {allrow['T0']['ALL']['t_median']:.4f} "
            f"5cm5 {allrow['T0']['ALL']['success_5cm5deg']:.4f} | "
            f"T2 R {allrow['T2']['ALL']['R_median']:7.3f} "
            f"t {allrow['T2']['ALL']['t_median']:.4f} "
            f"5cm5 {allrow['T2']['ALL']['success_5cm5deg']:.4f}")
        log(f"-> {path}")


def run_bootstrap(_arguments):
    result = {"resamples": TH.BOOTSTRAP, "population": CONFIRM,
              "note": "paired over frames within one seed; seeds never pooled, "
                      "and D4 is never merged with D2/D3",
              "seeds": {}}
    for seed in SEEDS:
        path = OUT / f"theta_posealigned_frames_seed{seed}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        rng = np.random.default_rng(TH.BOOTSTRAP_SEED + seed)
        entry = {}
        for label in [k[5:] for k in data.files if k.startswith("mask_")]:
            mask = data[f"mask_{label}"]
            for metric in ("R", "t"):
                a, b = data[f"T0_{metric}"], data[f"T2_{metric}"]
                valid = mask & np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 20:
                    continue
                draws = TH._paired_bootstrap(a, b, valid, np.median, rng)
                entry[f"{label}|{metric}"] = {
                    "n": int(valid.sum()),
                    "effect_pct": round(float(np.median(draws)), 3),
                    "ci95": [round(float(np.percentile(draws, 2.5)), 3),
                             round(float(np.percentile(draws, 97.5)), 3)],
                    "P_better": round(float((draws > 0).mean()), 4)}
        result["seeds"][f"seed{seed}"] = entry
    path = OUT / "theta_posealigned_bootstrap.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


def run_verdict(_arguments):
    result = {"gate": GATE, "population": CONFIRM, "per_seed": {}}
    passes = []
    for seed in SEEDS:
        path = OUT / f"theta_posealigned_d4_seed{seed}.json"
        if not path.exists():
            continue
        entry = json.loads(path.read_text())["subsets"]
        allrow, low = entry, entry
        t0, t2 = entry["T0"]["ALL"], entry["T2"]["ALL"]
        l0 = entry["T0"].get("V<8 (off-grid)")
        l2 = entry["T2"].get("V<8 (off-grid)")
        checks = {
            "ALL_R_gain_pct": TH.gain(t0["R_median"], t2["R_median"]),
            "ALL_t_gain_pct": TH.gain(t0["t_median"], t2["t_median"]),
            "ALL_success_delta_pp": 100.0 * (t2["success_5cm5deg"]
                                             - t0["success_5cm5deg"]),
            "Vlt8_R_gain_pct": TH.gain(l0["R_median"], l2["R_median"])
            if l0 else None,
            "Vlt8_t_gain_pct": TH.gain(l0["t_median"], l2["t_median"])
            if l0 else None}
        ok = (checks["ALL_R_gain_pct"] >= GATE["ALL_R_gain_pct"]
              and checks["ALL_t_gain_pct"] >= -GATE["ALL_t_degrade_pct"]
              and checks["ALL_success_delta_pp"] >= 0.0
              and checks["Vlt8_R_gain_pct"] is not None
              and checks["Vlt8_R_gain_pct"] >= GATE["Vlt8_R_gain_pct"]
              and checks["Vlt8_t_gain_pct"] >= -GATE["Vlt8_t_degrade_pct"])
        result["per_seed"][f"seed{seed}"] = {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in checks.items()}
        result["per_seed"][f"seed{seed}"]["PASS"] = bool(ok)
        passes.append(ok)
        log(f"seed{seed} PASS={ok}  ALL R {checks['ALL_R_gain_pct']:+.2f}% "
            f"t {checks['ALL_t_gain_pct']:+.2f}% "
            f"5cm5 {checks['ALL_success_delta_pp']:+.2f}pp | "
            f"V<8 R {checks['Vlt8_R_gain_pct']:+.2f}% "
            f"t {checks['Vlt8_t_gain_pct']:+.2f}%")
    result["THETA_ONLY_POSE_ALIGNED_CONFIRMED"] = bool(passes and all(passes))
    log(f"THETA_ONLY_POSE_ALIGNED_CONFIRMED = "
        f"{result['THETA_ONLY_POSE_ALIGNED_CONFIRMED']}")
    path = OUT / "theta_posealigned_verdict.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["d0", "eval", "bootstrap", "verdict"])
    arguments = parser.parse_args()
    {"d0": run_d0, "eval": run_eval, "bootstrap": run_bootstrap,
     "verdict": run_verdict}[arguments.command](arguments)


if __name__ == "__main__":
    main()
