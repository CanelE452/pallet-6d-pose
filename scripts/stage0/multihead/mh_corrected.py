"""PHASE 8-9 -- does a *predicted* scale correction move the pose, and does the
line branch still earn its place once it is applied?

PHASE 6 found the per-frame isotropic factor only weakly predictable (D2 R^2
0.13-0.17 against a 0.30 bar).  Weak is not the same as useless: the pose cares
about the direction of the correction more than its exactness, so this measures
the thing that actually matters instead of stopping at the regression score.

PHASE 8  C-arms: point-only PnP with the corners rescaled by
         nothing / the D0 constant / each Ridge predictor / ground truth.
PHASE 9  H-arms: the same corrected corners, solved point-only (H0) against
         jointly with the lines (H1).  This is the final gate on the second head:
         if the line branch cannot beat point-only once the corner
         configuration's dominant error mode is corrected, it does not qualify
         as a pose contributor.

Every choice is made on D0 and every number is read once on D2:

    alpha            5-fold CV on D0
    feature set      lowest worst-seed CV MSE on D0 -- never the D2 score, which
                     PHASE 6 ranked on and which must not select anything
    lambda_line      the value PHASE 4 calibrated on D0 for this same run
    scale target     `mh_scale.scale_target`, i.e. run_scaleoracle's definition

The GT-scale arm is a ceiling, reported as a ceiling, never as a method.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_cigm as CG                                             # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_scale as SC                                            # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)

# Pre-registered, both seeds, before anything ran.
POSE_GAIN_PCT = 3.0          # PHASE 8: t must improve by this much
LINE_GAIN_PCT = 3.0          # PHASE 9: R must improve by this much
LINE_COST_PCT = 2.0          # PHASE 9: t may not worsen by more than this
SUCCESS_TOLERANCE = 0.01     # PHASE 9: 5cm5deg may not drop by more than 1pp


def log(message):
    print(message, flush=True)


def solve_all(data, ratios, weight_line, use_lines):
    """Pose for every frame under one scale correction and one solver."""
    rows_R, rows_t = [], []
    for i in range(len(data["pred_corner"])):
        width, height = data["resolution"][i]
        pred = data["pred_corner"][i][:8].copy()
        if ratios is not None:
            pred = SC.apply_scale(pred, float(ratios[i]))
        corner_px = CG.grid_to_pixels(pred, width, height)
        pose = CG.solve(data["model"][i], corner_px, data["K"][i])
        if use_lines and pose is not None:
            lines = DG._line_in_pixels(data["pred_theta"][i],
                                       data["pred_rho"][i], width, height)
            try:
                pose, _ = DG._solve_joint(data["model"][i], data["K"][i],
                                          corner_px, lines, CG.EDGES,
                                          data["support"][i].astype(bool),
                                          weight_line, pose)
            except Exception:
                pass
        error = CG.pose_error(pose, data["R_gt"][i], data["t_gt"][i])
        rows_R.append(error[0] if error else np.nan)
        rows_t.append(error[1] if error else np.nan)
    return np.array(rows_R, float), np.array(rows_t, float)


def summarise(R, t, subsets):
    entry = {}
    for label, mask in subsets.items():
        good = mask & np.isfinite(R) & np.isfinite(t)
        if not good.any():
            continue
        entry[label] = {
            "n": int(mask.sum()),
            "R_median": round(float(np.median(R[good])), 4),
            "t_median": round(float(np.median(t[good])), 5),
            "success_5cm5deg": round(float(((R <= 5.0) & (t <= 0.05)
                                            & good).sum()
                                           / max(int(mask.sum()), 1)), 4)}
    return entry


def rel(reference, value):
    """Per cent improvement of `value` over `reference` (lower is better)."""
    return 100.0 * (reference - value) / abs(reference) if reference else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="e3confirm25k")
    arguments = parser.parse_args()

    lambdas = json.loads(
        (OUT / f"point_line_solver_{arguments.run}.json").read_text())
    meta = {r["stem"]: r for r in MD.load_split()}
    result = {"run": arguments.run,
              "gate": {"pose_gain_pct": POSE_GAIN_PCT,
                       "line_gain_pct": LINE_GAIN_PCT,
                       "line_cost_pct": LINE_COST_PCT,
                       "success_tolerance": SUCCESS_TOLERANCE},
              "selection": "feature set by worst-seed D0 CV MSE, never D2",
              "seeds": {}}

    # ---------------------------------------------------------- selection
    cv_by_set, fitted, loaded = {}, {}, {}
    for seed in SEEDS:
        fit_data = np.load(SC.cache(arguments.run, seed, "D0_MH_SEEN512"),
                           allow_pickle=True)
        eval_data = np.load(SC.cache(arguments.run, seed, "D2_MH_DEV512"),
                            allow_pickle=True)
        loaded[seed] = (fit_data, eval_data)
        y_fit = SC.scale_target(fit_data)
        for name, blocks in SC.FEATURE_SETS.items():
            if not blocks:
                continue
            X_fit = SC.build(fit_data, blocks)
            alpha, table = SC.choose_alpha(X_fit, y_fit)
            model = SC.ridge_fit(X_fit, y_fit, alpha)
            fitted[(seed, name)] = (model, alpha)
            cv_by_set.setdefault(name, []).append(min(table.values()))
    chosen = min(cv_by_set, key=lambda n: max(cv_by_set[n]))
    result["chosen_feature_set"] = chosen
    result["cv_mse_D0_worst_seed"] = {n: round(max(v), 8)
                                      for n, v in cv_by_set.items()}
    log(f"feature set chosen on D0 CV = {chosen}   "
        f"{ {n: round(max(v), 6) for n, v in cv_by_set.items()} }")

    for seed in SEEDS:
        fit_data, eval_data = loaded[seed]
        weight_line = float(lambdas[f"seed{seed}"]["lambda"])
        subsets = DG._frame_subsets(eval_data, meta)
        y_fit = SC.scale_target(fit_data)
        y_eval = SC.scale_target(eval_data)
        constant = float(np.median(y_fit))
        block = {"lambda_line": weight_line, "constant_from_D0": constant,
                 "arms": {}}

        # -------------------------------------------------------- PHASE 8
        ratio_by_arm = {"C0_uncorrected": None,
                        "C_const": np.full(len(y_eval), constant),
                        "C_oracle_GT": y_eval}
        for name in SC.FEATURE_SETS:
            if not SC.FEATURE_SETS[name]:
                continue
            model, _ = fitted[(seed, name)]
            X_eval = SC.build(eval_data, SC.FEATURE_SETS[name])
            ratio_by_arm["C" + name[1:]] = SC.ridge_apply(X_eval, model)

        for name, ratios in ratio_by_arm.items():
            R, t = solve_all(eval_data, ratios, weight_line, use_lines=False)
            block["arms"][name] = summarise(R, t, subsets)
            all_entry = block["arms"][name]["ALL"]
            log(f"  seed{seed} {name:<18} R {all_entry['R_median']:7.3f} "
                f"t {all_entry['t_median']:.4f} "
                f"5cm5 {all_entry['success_5cm5deg']:.4f}")

        # -------------------------------------------------------- PHASE 9
        chosen_ratios = ratio_by_arm["C" + chosen[1:]]
        R0, t0 = solve_all(eval_data, chosen_ratios, weight_line,
                           use_lines=False)
        R1, t1 = solve_all(eval_data, chosen_ratios, weight_line,
                           use_lines=True)
        block["H0_corrected_point"] = summarise(R0, t0, subsets)
        block["H1_corrected_point_line"] = summarise(R1, t1, subsets)
        h0, h1 = block["H0_corrected_point"]["ALL"], \
            block["H1_corrected_point_line"]["ALL"]
        log(f"  seed{seed} H0 point       R {h0['R_median']:7.3f} "
            f"t {h0['t_median']:.4f} 5cm5 {h0['success_5cm5deg']:.4f}")
        log(f"  seed{seed} H1 point+line  R {h1['R_median']:7.3f} "
            f"t {h1['t_median']:.4f} 5cm5 {h1['success_5cm5deg']:.4f}")

        base = block["arms"]["C0_uncorrected"]["ALL"]
        best = block["arms"]["C" + chosen[1:]]["ALL"]
        block["phase8"] = {
            "t_gain_pct": round(rel(base["t_median"], best["t_median"]), 2),
            "R_gain_pct": round(rel(base["R_median"], best["R_median"]), 2),
            "success_delta_pp": round(100.0 * (best["success_5cm5deg"]
                                               - base["success_5cm5deg"]), 2),
            "oracle_t_gain_pct": round(
                rel(base["t_median"],
                    block["arms"]["C_oracle_GT"]["ALL"]["t_median"]), 2),
            "const_t_gain_pct": round(
                rel(base["t_median"],
                    block["arms"]["C_const"]["ALL"]["t_median"]), 2)}
        block["phase9"] = {
            "R_gain_pct": round(rel(h0["R_median"], h1["R_median"]), 2),
            "t_gain_pct": round(rel(h0["t_median"], h1["t_median"]), 2),
            "success_delta_pp": round(100.0 * (h1["success_5cm5deg"]
                                               - h0["success_5cm5deg"]), 2)}
        result["seeds"][f"seed{seed}"] = block

    phase8 = all(
        result["seeds"][f"seed{s}"]["phase8"]["t_gain_pct"] >= POSE_GAIN_PCT
        and result["seeds"][f"seed{s}"]["phase8"]["success_delta_pp"] >= 0.0
        for s in SEEDS)
    phase9 = all(
        result["seeds"][f"seed{s}"]["phase9"]["R_gain_pct"] >= LINE_GAIN_PCT
        and result["seeds"][f"seed{s}"]["phase9"]["t_gain_pct"] >= -LINE_COST_PCT
        and result["seeds"][f"seed{s}"]["phase9"]["success_delta_pp"]
        >= -100.0 * SUCCESS_TOLERANCE
        for s in SEEDS)
    result["PREDICTED_SCALE_HELPS_POSE"] = bool(phase8)
    result["TWO_HEAD_POSE_QUALIFIED"] = bool(phase9)
    log(f"\nPREDICTED_SCALE_HELPS_POSE = {phase8}")
    log(f"TWO_HEAD_POSE_QUALIFIED    = {phase9}")

    path = OUT / f"corrected_point_line_{arguments.run}.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


if __name__ == "__main__":
    main()
