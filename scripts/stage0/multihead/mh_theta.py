"""PHASE 3-5 -- theta-only Point+Line solver.  No training anywhere in this file.

The question is narrow: the full `(theta, rho)` line constraint improved rotation
on both seeds and translation on only one, so is the damage coming from `rho`?
This removes `rho` from the pose objective and keeps the line's orientation.

## How rho is removed -- algebraically, not by rebuilding the geometry

The existing joint residual (`mh_diagnose._joint_residual`) puts both projected
endpoints of each edge on the predicted line:

    da = A*u_a.x + B*u_a.y + C          C carries rho
    db = A*u_b.x + B*u_b.y + C

Split that pair into its two natural halves:

    (da + db)/2 = offset of the edge midpoint from the line     <- carries C, i.e. rho
    (da - db)/2 = A*(u_a.x - u_b.x)/2 + B*(u_a.y - u_b.y)/2     <- C cancels exactly

The second is `rho`-free by construction, and it is the orientation term:
with the line normal `n` unit-length, `n . (u_a - u_b) = L * sin(delta)` where `L`
is the projected edge length and `delta` the undirected angle between the edge and
the line.  So

    r_theta = (da - db) / 2 = (L / 2) * sin(delta)

is the brief's `(L/2) * delta_phi` residual to first order, in pixels, and it is
undirected for free -- swapping the endpoints only flips the sign, and the solver
squares it.  Nothing new needs a wrap function, and no `+pi/2` convention has to be
guessed: the line's pixel-space normal comes from the same `_line_in_pixels` the
full-line solver already uses, and `(A, B)` there is provably independent of rho
(`A = cos(t)*GRID/w`, `B = sin(t)*GRID/h`, both divided by `hypot(A, B)`).

Everything else is the existing contract, unchanged: the point residual, the
per-branch `1/sqrt(n)` normalisation, the Huber scale, `max_nfev`, and the
point-only PnP as the initialisation.
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

OUT = MD.OUT
SEEDS = (1, 2)
RUN = "e3confirm25k"

# The grid the full-line solver was calibrated on.  Reused deliberately: the
# theta residual is a difference of the same pixel quantities the full-line
# residual is built from, so the weights mean the same kind of thing.
LAMBDA_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)

# PHASE 4 selection rule -- locked before D0 was scored, never revisited.
SAFETY_T_DEGRADE_PCT = 3.0
SAFETY_SUCCESS_DROP_PP = 0.0
SAFETY_SOLVE_DROP_PP = 1.0

# PHASE 5 gate -- locked before D2 was read.
GATE = {"ALL_R_gain_pct": 5.0, "ALL_t_degrade_pct": 3.0,
        "Vlt8_R_gain_pct": 10.0, "Vlt8_t_degrade_pct": 5.0,
        "R_p90_degrade_pct": 5.0}

BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260818


def log(message):
    print(message, flush=True)


def cache(seed, population):
    return OUT / f"mh_predcache_{RUN}_seed{seed}_{population}.npz"


# ------------------------------------------------------------------ residual

def theta_residual(params, model, K, corner_px, lines, edges, use_line,
                   weight_theta):
    """Point residual as before; line term keeps orientation and drops rho."""
    import cv2
    rvec, tvec = params[:3], params[3:]
    rotation, _ = cv2.Rodrigues(rvec)
    camera = (rotation @ model.T).T + tvec
    depth = np.clip(camera[:, 2], 1e-6, None)
    projected = (K @ (camera / depth[:, None]).T).T[:, :2]

    point = np.linalg.norm(projected - corner_px, axis=1)
    point = point / np.sqrt(max(len(point), 1))

    values = [point]
    if use_line.any():
        a = np.stack([projected[e[0]] for e in edges])
        b = np.stack([projected[e[1]] for e in edges])
        da = lines[:, 0] * a[:, 0] + lines[:, 1] * a[:, 1] + lines[:, 2]
        db = lines[:, 0] * b[:, 0] + lines[:, 1] * b[:, 1] + lines[:, 2]
        orientation = 0.5 * (da - db)          # rho cancels here, exactly
        stacked = orientation[use_line]
        values.append(weight_theta * stacked / np.sqrt(max(len(stacked), 1)))
    return np.concatenate(values)


def solve_theta(model, K, corner_px, lines, edges, use_line, weight_theta,
                initial):
    from scipy.optimize import least_squares
    import cv2
    rvec, _ = cv2.Rodrigues(initial[0])
    params = np.concatenate([rvec.reshape(3), initial[1]])
    out = least_squares(theta_residual, params, method="trf", loss="huber",
                        f_scale=DG.HUBER_PX, max_nfev=DG.MAX_NFEV,
                        args=(model, K, corner_px, lines, edges, use_line,
                              weight_theta))
    rotation, _ = cv2.Rodrigues(out.x[:3])
    return (rotation, out.x[3:]), float(out.cost)


# ------------------------------------------------------------------ scoring

def score_population(seed, population, weight_theta, arms=("T0", "T2"),
                     full_line_weight=None):
    """Per-frame R and t for the requested arms on one cached population."""
    data = np.load(cache(seed, population), allow_pickle=True)
    n = len(data["pred_corner"])
    out = {arm: {"R": np.full(n, np.nan), "t": np.full(n, np.nan),
                 "solved": np.zeros(n, bool)} for arm in arms}
    for i in range(n):
        width, height = data["resolution"][i]
        model, K = data["model"][i], data["K"][i]
        corner_px = CG.grid_to_pixels(data["pred_corner"][i][:8], width, height)
        base = CG.solve(model, corner_px, K)
        support = data["support"][i].astype(bool)
        lines = DG._line_in_pixels(data["pred_theta"][i], data["pred_rho"][i],
                                   width, height)
        for arm in arms:
            pose = base
            if pose is not None and arm != "T0":
                try:
                    if arm == "T2":
                        pose, _ = solve_theta(model, K, corner_px, lines,
                                              CG.EDGES, support, weight_theta,
                                              pose)
                    elif arm == "T1":
                        pose, _ = DG._solve_joint(model, K, corner_px, lines,
                                                  CG.EDGES, support,
                                                  full_line_weight, pose)
                except Exception:
                    pass
            error = CG.pose_error(pose, data["R_gt"][i], data["t_gt"][i])
            if error:
                out[arm]["R"][i], out[arm]["t"][i] = error[0], error[1]
                out[arm]["solved"][i] = True
    return data, out


def summarise(rows, mask):
    good = mask & np.isfinite(rows["R"]) & np.isfinite(rows["t"])
    if not good.any():
        return None
    R, t = rows["R"][good], rows["t"][good]
    return {"n": int(mask.sum()),
            "solve_rate": round(float(rows["solved"][mask].mean()), 4),
            "R_median": round(float(np.median(R)), 4),
            "R_p90": round(float(np.percentile(R, 90)), 4),
            "t_median": round(float(np.median(t)), 5),
            "t_p90": round(float(np.percentile(t, 90)), 5),
            "success_5cm5deg": round(float(((rows["R"] <= 5.0)
                                            & (rows["t"] <= 0.05)
                                            & good).sum()
                                           / max(int(mask.sum()), 1)), 4)}


def gain(reference, value):
    return 100.0 * (reference - value) / abs(reference) if reference else 0.0


# ------------------------------------------------------------------ PHASE 4

def run_d0(_arguments):
    """Choose lambda_theta on D0 under the locked safety filter."""
    result = {"grid": list(LAMBDA_GRID),
              "rule": "reject if t degrades > 3%, or 5cm5deg drops at all, or "
                      "solve rate drops > 1pp; among survivors take the "
                      "smallest R median, ties to the smaller lambda",
              "safety": {"t_degrade_pct": SAFETY_T_DEGRADE_PCT,
                         "success_drop_pp": SAFETY_SUCCESS_DROP_PP,
                         "solve_drop_pp": SAFETY_SOLVE_DROP_PP},
              "seeds": {}}
    for seed in SEEDS:
        _, rows = score_population(seed, "D0_MH_SEEN512", 0.0, arms=("T0",))
        everything = np.ones(len(rows["T0"]["R"]), bool)
        base = summarise(rows["T0"], everything)
        block = {"P0_point_only": base, "candidates": {}}
        log(f"seed{seed}  P0  R {base['R_median']:.4f}  t {base['t_median']:.5f}"
            f"  5cm5 {base['success_5cm5deg']:.4f}  solve {base['solve_rate']:.4f}")
        survivors = []
        for weight in LAMBDA_GRID:
            _, cand = score_population(seed, "D0_MH_SEEN512", weight,
                                       arms=("T2",))
            entry = summarise(cand["T2"], everything)
            reasons = []
            if gain(base["t_median"], entry["t_median"]) < -SAFETY_T_DEGRADE_PCT:
                reasons.append("t")
            if (entry["success_5cm5deg"] - base["success_5cm5deg"]) < \
                    -SAFETY_SUCCESS_DROP_PP / 100.0:
                reasons.append("5cm5deg")
            if (base["solve_rate"] - entry["solve_rate"]) > \
                    SAFETY_SOLVE_DROP_PP / 100.0:
                reasons.append("solve")
            entry["rejected_for"] = reasons
            block["candidates"][f"{weight:g}"] = entry
            if not reasons:
                survivors.append((entry["R_median"], weight))
            log(f"  lam {weight:<5g} R {entry['R_median']:8.4f} "
                f"t {entry['t_median']:.5f} 5cm5 {entry['success_5cm5deg']:.4f} "
                f"solve {entry['solve_rate']:.4f} "
                f"{'OK' if not reasons else 'reject:' + ','.join(reasons)}")
        if survivors:
            survivors.sort(key=lambda pair: (pair[0], pair[1]))
            block["selected_lambda_theta"] = survivors[0][1]
            block["NO_SAFE_LAMBDA"] = False
        else:
            block["selected_lambda_theta"] = None
            block["NO_SAFE_LAMBDA"] = True
        log(f"  -> selected lambda_theta = {block['selected_lambda_theta']}")
        result["seeds"][f"seed{seed}"] = block
    path = OUT / "theta_only_solver_d0.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


# ------------------------------------------------------------------ PHASE 5

def run_eval(arguments):
    """Score T0/T1/T2 once per population, with lambda already fixed on D0."""
    d0 = json.loads((OUT / "theta_only_solver_d0.json").read_text())
    full = json.loads((OUT / f"point_line_solver_{RUN}.json").read_text())
    meta = {r["stem"]: r for r in MD.load_split()}
    populations = [p for p in arguments.populations.split(",") if p]

    for seed in SEEDS:
        weight = d0["seeds"][f"seed{seed}"]["selected_lambda_theta"]
        if weight is None:
            log(f"seed{seed}: no safe lambda, nothing to evaluate")
            continue
        full_weight = float(full[f"seed{seed}"]["lambda"])
        block = {"lambda_theta": weight, "lambda_full_line": full_weight,
                 "populations": {}}
        for population in populations:
            data, rows = score_population(seed, population, weight,
                                          arms=("T0", "T1", "T2"),
                                          full_line_weight=full_weight)
            subsets = DG._frame_subsets(data, meta)
            entry = {}
            for label, mask in subsets.items():
                entry[label] = {arm: summarise(rows[arm], mask)
                                for arm in ("T0", "T1", "T2")}
            block["populations"][population] = entry
            np.savez_compressed(
                OUT / f"theta_only_frames_seed{seed}_{population}.npz",
                **{f"{arm}_{key}": rows[arm][key]
                   for arm in rows for key in ("R", "t", "solved")},
                **{f"mask_{label}": mask for label, mask in subsets.items()})
            allrow = entry["ALL"]
            log(f"seed{seed} {population:<15} "
                f"T0 R {allrow['T0']['R_median']:7.3f} t {allrow['T0']['t_median']:.4f} "
                f"5cm5 {allrow['T0']['success_5cm5deg']:.4f} | "
                f"T2 R {allrow['T2']['R_median']:7.3f} t {allrow['T2']['t_median']:.4f} "
                f"5cm5 {allrow['T2']['success_5cm5deg']:.4f}")
        path = OUT / f"theta_only_solver_seed{seed}.json"
        path.write_text(json.dumps(block, indent=1))
        log(f"-> {path}")


def _paired_bootstrap(a, b, valid, statistic, rng):
    """Paired frame bootstrap of the improvement of `b` over `a`."""
    index = np.flatnonzero(valid)
    draws = np.empty(BOOTSTRAP)
    for k in range(BOOTSTRAP):
        pick = index[rng.integers(0, len(index), len(index))]
        draws[k] = gain(statistic(a[pick]), statistic(b[pick]))
    return draws


def run_bootstrap(arguments):
    result = {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED,
              "note": "paired over frames, within one seed; seeds are never "
                      "pooled and n=2 seeds stays n=2",
              "seeds": {}}
    populations = [p for p in arguments.populations.split(",") if p]
    for seed in SEEDS:
        block = {}
        for population in populations:
            path = OUT / f"theta_only_frames_seed{seed}_{population}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
            entry = {}
            for label in [k[5:] for k in data.files if k.startswith("mask_")]:
                mask = data[f"mask_{label}"]
                for metric, statistic in (("R", np.median), ("t", np.median)):
                    a, b = data[f"T0_{metric}"], data[f"T2_{metric}"]
                    valid = mask & np.isfinite(a) & np.isfinite(b)
                    if valid.sum() < 20:
                        continue
                    draws = _paired_bootstrap(a, b, valid, statistic, rng)
                    entry[f"{label}|{metric}"] = {
                        "n": int(valid.sum()),
                        "effect_pct": round(float(np.median(draws)), 3),
                        "ci95": [round(float(np.percentile(draws, 2.5)), 3),
                                 round(float(np.percentile(draws, 97.5)), 3)],
                        "P_better": round(float((draws > 0).mean()), 4)}
            block[population] = entry
        result["seeds"][f"seed{seed}"] = block
    path = OUT / "theta_only_solver_bootstrap.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


# ------------------------------------------------------------------ verdict

def run_verdict(arguments):
    seeds = {s: json.loads((OUT / f"theta_only_solver_seed{s}.json").read_text())
             for s in SEEDS}
    populations = [p for p in arguments.populations.split(",") if p]
    result = {"gate": GATE, "populations": {}}
    for population in populations:
        per_seed, passes = {}, []
        for seed in SEEDS:
            entry = seeds[seed]["populations"].get(population)
            if entry is None:
                continue
            allrow, low = entry["ALL"], entry.get("V<8 (off-grid)")
            checks = {
                "ALL_R_gain_pct": gain(allrow["T0"]["R_median"],
                                       allrow["T2"]["R_median"]),
                "ALL_t_gain_pct": gain(allrow["T0"]["t_median"],
                                       allrow["T2"]["t_median"]),
                "ALL_success_delta_pp": 100.0 * (allrow["T2"]["success_5cm5deg"]
                                                 - allrow["T0"]["success_5cm5deg"]),
                "ALL_R_p90_gain_pct": gain(allrow["T0"]["R_p90"],
                                           allrow["T2"]["R_p90"]),
                "Vlt8_R_gain_pct": gain(low["T0"]["R_median"],
                                        low["T2"]["R_median"]) if low else None,
                "Vlt8_t_gain_pct": gain(low["T0"]["t_median"],
                                        low["T2"]["t_median"]) if low else None}
            ok = (checks["ALL_R_gain_pct"] >= GATE["ALL_R_gain_pct"]
                  and checks["ALL_t_gain_pct"] >= -GATE["ALL_t_degrade_pct"]
                  and checks["ALL_success_delta_pp"] >= 0.0
                  and checks["ALL_R_p90_gain_pct"] >= -GATE["R_p90_degrade_pct"]
                  and checks["Vlt8_R_gain_pct"] is not None
                  and checks["Vlt8_R_gain_pct"] >= GATE["Vlt8_R_gain_pct"]
                  and checks["Vlt8_t_gain_pct"] >= -GATE["Vlt8_t_degrade_pct"])
            per_seed[f"seed{seed}"] = {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in checks.items()}
            per_seed[f"seed{seed}"]["PASS"] = bool(ok)
            passes.append(ok)
        result["populations"][population] = {
            "per_seed": per_seed,
            "THETA_ONLY_LINE_USEFUL": bool(passes and all(passes))}
        log(f"{population}: THETA_ONLY_LINE_USEFUL = "
            f"{result['populations'][population]['THETA_ONLY_LINE_USEFUL']}")
        for name, block in per_seed.items():
            log(f"  {name} PASS={block['PASS']}  "
                f"ALL R {block['ALL_R_gain_pct']:+.2f}% "
                f"t {block['ALL_t_gain_pct']:+.2f}% "
                f"5cm5 {block['ALL_success_delta_pp']:+.2f}pp "
                f"Rp90 {block['ALL_R_p90_gain_pct']:+.2f}% | "
                f"V<8 R {block['Vlt8_R_gain_pct']:+.2f}% "
                f"t {block['Vlt8_t_gain_pct']:+.2f}%")
    path = OUT / "theta_only_verdict.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["d0", "eval", "bootstrap", "verdict"])
    parser.add_argument("--populations", default="D2_MH_DEV512")
    main_arguments = parser.parse_args()
    {"d0": run_d0, "eval": run_eval, "bootstrap": run_bootstrap,
     "verdict": run_verdict}[main_arguments.command](main_arguments)


if __name__ == "__main__":
    main()
