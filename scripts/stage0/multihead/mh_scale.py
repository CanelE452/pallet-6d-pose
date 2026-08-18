"""PHASE 5-6 -- is the per-frame isotropic scale error predictable from the
model's own outputs?

The scale oracle is the largest single lever on translation measured anywhere in
this study (+33-34% translation, 5cm5deg nearly doubled), but it needs ground
truth.  A constant factor calibrated on D0 recovers only 0-14% of it, so the
shrinkage is view-dependent rather than a fixed bias.  This asks the next
question: can the *per-frame* factor be regressed from what the network already
emits, without ground truth?

PHASE 5 -- the target, resolved from `mh_diagnose.run_scaleoracle` rather than
redefined.  Reading that function line by line:

    corners       pred[:8] / gt[:8]        the 8 box corners, centroid excluded
    coordinates   grid frame, i.e. before `mh_cigm.grid_to_pixels`
    spread(x)     mean_j || x_j - mean(x) ||      mean radius about the centroid
    target        s* = spread(gt) / max(spread(pred), 1e-9)
    centre        pred.mean(0)             the PREDICTED centroid, not GT's
    application   pred <- centre + (pred - centre) * s*
    order         applied in the grid frame, before grid_to_pixels and before solve

Nothing here is new; the identical four lines live at
`mh_diagnose.py:1119-1124`.  `spread` is a mean, not an RMS, and the centre is
the predicted centroid -- both are easy to "improve" by accident, which would
silently change what the oracle number means.

PHASE 6 -- predictability.  Ridge, fitted on D0 only, evaluated once on D2.
Ground truth enters as the target and never as an input feature.  Object
dimensions are excluded from the features too: `mh_cigm.solve` already receives
the model points, so a residual scale error is a corner-configuration error, not
a missing-dimension one, and feeding dims in would quietly turn this into a
dims-known method.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_data as MD                                             # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
RUN = "e3confirm25k"

# Locked before the first fit, as the brief requires.
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
CV_FOLDS = 5
CV_SEED = 20260817

# Pre-registered gate.  Both seeds must satisfy both clauses.
MIN_R2 = 0.30
MIN_RESIDUAL_GAIN_VS_CONST = 20.0        # per cent


def log(message):
    print(message, flush=True)


def cache(run, seed, split):
    return OUT / f"mh_predcache_{run}_seed{seed}_{split}.npz"


# ---------------------------------------------------------------- PHASE 5

def spread(points):
    """Mean radius about the centroid -- `run_scaleoracle`'s definition."""
    return np.linalg.norm(points - points.mean(0), axis=1).mean()


def scale_target(data):
    """s* per frame, exactly as the oracle applies it."""
    out = np.empty(len(data["pred_corner"]))
    for i in range(len(out)):
        pred = data["pred_corner"][i][:8]
        truth = data["gt_corner"][i][:8]
        out[i] = spread(truth) / max(spread(pred), 1e-9)
    return out


def apply_scale(pred_corner_8, ratio):
    """The oracle's own two lines, so callers cannot drift from them."""
    centre = pred_corner_8.mean(0)
    return centre + (pred_corner_8 - centre) * ratio


# ---------------------------------------------------------------- PHASE 6

def block_geometry(data):
    """Where and how large the prediction sits.  No confidence, no ground truth."""
    rows = []
    for i in range(len(data["pred_corner"])):
        pred = data["pred_corner"][i][:8]
        centre = pred.mean(0)
        radii = np.linalg.norm(pred - centre, axis=1)
        extent = pred.max(0) - pred.min(0)
        rows.append([
            radii.mean(),
            np.log(max(radii.mean(), 1e-9)),
            radii.std(),
            centre[0], centre[1],
            extent[0], extent[1],
            extent[0] / max(extent[1], 1e-9),
            float(data["in_grid"][i].sum()),
        ])
    return np.asarray(rows, float)


def block_point(data):
    """Corner-head confidence: the nine peaks and their summary."""
    peak = data["corner_peak"].astype(float)
    return np.concatenate(
        [peak, peak.mean(1, keepdims=True), peak.min(1, keepdims=True)], axis=1)


def block_line(data):
    """Line-head outputs: the twelve confidences and the hypothesis spread."""
    peak = data["peak"].astype(float)
    margin = data["margin"].astype(float)
    entropy = data["entropy"].astype(float)
    rho = data["pred_rho"].astype(float)
    support = data["support"].astype(float)
    summary = np.stack([peak.mean(1), margin.mean(1), entropy.mean(1),
                        rho.std(1), support.sum(1)], axis=1)
    return np.concatenate([peak, margin, entropy, summary], axis=1)


FEATURE_SETS = {
    "S0_constant":  (),
    "S1_geometry":  ("geometry",),
    "S2P_point":    ("geometry", "point"),
    "S2L_line":     ("geometry", "line"),
    "S2PL_both":    ("geometry", "point", "line"),
}
BLOCKS = {"geometry": block_geometry, "point": block_point, "line": block_line}


def build(data, names):
    if not names:
        return np.zeros((len(data["pred_corner"]), 0))
    return np.concatenate([BLOCKS[n](data) for n in names], axis=1)


def ridge_fit(X, y, alpha):
    """Closed-form ridge on standardised inputs; intercept never penalised."""
    if X.shape[1] == 0:
        return np.zeros(0), float(y.mean())
    centre, scale = X.mean(0), X.std(0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Z = (X - centre) / scale
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return (w / scale, float(y.mean() - (centre / scale) @ w))


def ridge_apply(X, model):
    w, b = model
    return (X @ w + b) if X.shape[1] else np.full(len(X), b)


def choose_alpha(X, y):
    """K-fold on D0 only.  Ties break toward the stronger penalty."""
    order = np.random.RandomState(CV_SEED).permutation(len(y))
    folds = np.array_split(order, CV_FOLDS)
    best, best_error = ALPHA_GRID[-1], np.inf
    table = {}
    for alpha in ALPHA_GRID:
        errors = []
        for k in range(CV_FOLDS):
            hold = folds[k]
            keep = np.concatenate([folds[j] for j in range(CV_FOLDS) if j != k])
            model = ridge_fit(X[keep], y[keep], alpha)
            errors.append(np.mean((ridge_apply(X[hold], model) - y[hold]) ** 2))
        mean_error = float(np.mean(errors))
        table[f"{alpha:g}"] = round(mean_error, 8)
        if mean_error < best_error - 1e-12:
            best, best_error = alpha, mean_error
    return best, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN)
    arguments = parser.parse_args()

    result = {"run": arguments.run,
              "target": "s* = spread(gt[:8]) / spread(pred[:8]), grid frame, "
                        "centred on the predicted centroid "
                        "(mh_diagnose.run_scaleoracle:1119-1124)",
              "alpha_grid": list(ALPHA_GRID), "cv_folds": CV_FOLDS,
              "gate": {"min_r2": MIN_R2,
                       "min_residual_gain_vs_const_pct":
                           MIN_RESIDUAL_GAIN_VS_CONST},
              "features_exclude": ["object dimensions (would make this "
                                   "dims-known)", "any ground-truth quantity"],
              "seeds": {}}

    for seed in SEEDS:
        fit_data = np.load(cache(arguments.run, seed, "D0_MH_SEEN512"),
                           allow_pickle=True)
        eval_data = np.load(cache(arguments.run, seed, "D2_MH_DEV512"),
                            allow_pickle=True)
        y_fit, y_eval = scale_target(fit_data), scale_target(eval_data)

        constant = float(np.median(y_fit))
        base_residual = float(np.mean((y_eval - constant) ** 2))
        unit_residual = float(np.mean((y_eval - 1.0) ** 2))
        block = {"n_fit": len(y_fit), "n_eval": len(y_eval),
                 "constant_from_D0": constant,
                 "target_D0_median": constant,
                 "target_D2_median": float(np.median(y_eval)),
                 "residual_uncorrected": unit_residual,
                 "residual_constant": base_residual,
                 "arms": {}}
        log(f"seed{seed}  s* median D0 {constant:.4f}  D2 "
            f"{np.median(y_eval):.4f}   n {len(y_fit)}/{len(y_eval)}")

        for name, names in FEATURE_SETS.items():
            X_fit, X_eval = build(fit_data, names), build(eval_data, names)
            alpha, table = choose_alpha(X_fit, y_fit)
            model = ridge_fit(X_fit, y_fit, alpha)
            prediction = ridge_apply(X_eval, model)
            residual = float(np.mean((prediction - y_eval) ** 2))
            variance = float(np.mean((y_eval - y_eval.mean()) ** 2))
            block["arms"][name] = {
                "n_features": int(X_fit.shape[1]), "alpha": alpha,
                "cv_mse_D0": table,
                "residual_D2": residual,
                "r2_D2": round(1.0 - residual / variance, 4),
                "gain_vs_constant_pct":
                    round(100.0 * (base_residual - residual) / base_residual, 2),
                "pred_median": float(np.median(prediction)),
                "pred_std": float(np.std(prediction))}
            entry = block["arms"][name]
            log(f"  {name:<14} k {entry['n_features']:>3}  alpha {alpha:<8g} "
                f"R2 {entry['r2_D2']:+.4f}  vs const "
                f"{entry['gain_vs_constant_pct']:+6.2f}%")
        result["seeds"][f"seed{seed}"] = block

    best_name = max(
        FEATURE_SETS,
        key=lambda n: min(result["seeds"][f"seed{s}"]["arms"][n]["r2_D2"]
                          for s in SEEDS))
    passes = all(
        result["seeds"][f"seed{s}"]["arms"][best_name]["r2_D2"] > MIN_R2
        and result["seeds"][f"seed{s}"]["arms"][best_name]
        ["gain_vs_constant_pct"] > MIN_RESIDUAL_GAIN_VS_CONST
        for s in SEEDS)
    result["best_feature_set"] = best_name
    result["SCALE_PREDICTABLE"] = bool(passes)
    log(f"\nbest by worst-seed R2 = {best_name}")
    log(f"SCALE_PREDICTABLE = {passes}")

    path = OUT / f"scale_ridge_{arguments.run}.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


if __name__ == "__main__":
    main()
