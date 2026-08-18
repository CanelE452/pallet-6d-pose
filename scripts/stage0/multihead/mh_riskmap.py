"""PHASE 8 -- where E3's remaining error lives, as a function of elevation and
how much of the pallet is on screen.  No training; the E3 25k caches are reused.

The diagnostic population is the dev side only -- D2, D3 and D4 together, 1,536
frames per seed.  D0 is deliberately excluded: it is train-side, and a risk map
built on frames the model has seen would understate exactly the failures this is
looking for.

Bins are fixed from counts before any metric is read, as the brief requires.
The elevation edges follow the distribution measured on the split itself:

    <8 deg     3,076 frames of 40,000   7.69%
    8-15       3,645                    9.11%
    15-30      9,007                   22.52%
    >=30      24,272                   60.68%

and the real captures sit at about 94% below 8 degrees
(`_docs/history/2026-07-05.md:15`, 92 of 98), which is the mismatch this map is
meant to locate rather than assume.

This stage is diagnostic.  A cell being worse is a correlation, not a cause, and
the write-up says so; PHASE 10's resampling screen is what would test causality.
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
import mh_theta as TH                                            # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
POPULATIONS = ("D2_MH_DEV512", "D3_MH_CONF512", "D4_THETA_CONFIRM512")
ELEV_EDGES = (0.0, 8.0, 15.0, 30.0, 1e9)
ELEV_NAMES = ("<8", "8-15", "15-30", ">=30")
V_NAMES = ("V=8", "V=7", "V<=6")
MIN_CELL = 20                     # below this a cell is reported but not ranked


def log(message):
    print(message, flush=True)


def undirected_angle_error(a, b):
    """Smallest angle between two undirected line orientations, in degrees."""
    difference = a - b
    wrapped = 0.5 * np.arctan2(np.sin(2 * difference), np.cos(2 * difference))
    return np.degrees(np.abs(wrapped))


def frame_rows(seed):
    """One row per dev frame: bins, geometry, pose, line quality, theta gain."""
    meta = {r["stem"]: r for r in MD.load_split()}
    aligned = json.loads((OUT / "theta_posealigned_d0.json").read_text())
    weight = aligned["seeds"][f"seed{seed}"]["selected_lambda_theta"]
    rows = []
    for population in POPULATIONS:
        data, scored = TH.score_population(seed, population, weight,
                                           arms=("T0", "T2"))
        for index in range(len(data["pred_corner"])):
            stem = str(data["stems"][index])
            info = meta.get(stem)
            if info is None:
                continue
            predicted = data["pred_corner"][index][:8]
            truth = data["gt_corner"][index][:8]
            fit = DG._affine_fit(truth, predicted)
            support = data["support"][index].astype(bool)
            theta_error = undirected_angle_error(
                data["pred_theta"][index], data["gt_theta"][index])
            rho_error = np.abs(data["pred_rho"][index]
                               - data["gt_rho"][index])
            rows.append({
                "stem": stem, "population": population,
                "elev": float(info["elev"]), "v": int(info["v"]),
                "corner_error": float(np.linalg.norm(
                    predicted - truth, axis=1).mean()),
                "front_rear_shift": float(np.linalg.norm(
                    (predicted[list(DG.FRONT)].mean(0)
                     - predicted[list(DG.REAR)].mean(0))
                    - (truth[list(DG.FRONT)].mean(0)
                       - truth[list(DG.REAR)].mean(0)))),
                "affine_scale_gap": abs(fit["scale_isotropic"] - 1.0),
                "centroid_shift": float(np.linalg.norm(
                    predicted.mean(0) - truth.mean(0))),
                "nonaffine_rms": float(fit["nonaffine_rms"]),
                "R": float(scored["T0"]["R"][index]),
                "t": float(scored["T0"]["t"][index]),
                "R_theta": float(scored["T2"]["R"][index]),
                "t_theta": float(scored["T2"]["t"][index]),
                "theta_error": float(theta_error[support].mean())
                if support.any() else np.nan,
                "rho_error": float(rho_error[support].mean())
                if support.any() else np.nan,
                "entropy": float(data["entropy"][index][support].mean())
                if support.any() else np.nan,
            })
    return rows, weight


def bin_of(row):
    elevation = np.searchsorted(ELEV_EDGES, row["elev"], side="right") - 1
    elevation = int(np.clip(elevation, 0, len(ELEV_NAMES) - 1))
    v = 0 if row["v"] == 8 else (1 if row["v"] == 7 else 2)
    return ELEV_NAMES[elevation], V_NAMES[v]


MEASURES = ("corner_error", "front_rear_shift", "affine_scale_gap",
            "centroid_shift", "nonaffine_rms", "R", "t",
            "theta_error", "rho_error", "entropy")


def summarise(rows):
    entry = {"n": len(rows)}
    for key in MEASURES:
        values = np.array([r[key] for r in rows], float)
        values = values[np.isfinite(values)]
        entry[key] = round(float(np.median(values)), 5) if len(values) else None
    R0 = np.array([r["R"] for r in rows], float)
    t0 = np.array([r["t"] for r in rows], float)
    R2 = np.array([r["R_theta"] for r in rows], float)
    t2 = np.array([r["t_theta"] for r in rows], float)
    good = np.isfinite(R0) & np.isfinite(t0) & np.isfinite(R2) & np.isfinite(t2)
    entry["success_5cm5deg"] = round(float(
        ((R0 <= 5.0) & (t0 <= 0.05) & good).sum() / max(len(rows), 1)), 4)
    if good.any():
        entry["theta_gain_R_pct"] = round(TH.gain(float(np.median(R0[good])),
                                                  float(np.median(R2[good]))), 2)
        entry["theta_gain_t_pct"] = round(TH.gain(float(np.median(t0[good])),
                                                  float(np.median(t2[good]))), 2)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = {"populations": list(POPULATIONS),
              "note": "dev side only; D0 excluded because it is train-side",
              "elev_edges": list(ELEV_EDGES[:-1]) + ["inf"],
              "min_cell": MIN_CELL,
              "source_distribution": {
                  "synthetic_lt8_pct": 7.69, "synthetic_n": 40000,
                  "real_lt8_pct": 94.0,
                  "real_source": "_docs/history/2026-07-05.md:15 (92/98)"},
              "diagnostic_only": "cells are correlations, not causes",
              "seeds": {}}
    for seed in SEEDS:
        rows, weight = frame_rows(seed)
        cells = {}
        for row in rows:
            elevation, v = bin_of(row)
            cells.setdefault(f"{elevation}|{v}", []).append(row)
        block = {"lambda_theta": weight, "n_frames": len(rows), "cells": {}}
        for name in sorted(cells):
            block["cells"][name] = summarise(cells[name])
        # marginals, so a small cell can still be read against its row/column
        for elevation in ELEV_NAMES:
            subset = [r for r in rows if bin_of(r)[0] == elevation]
            if subset:
                block["cells"][f"{elevation}|ALL_V"] = summarise(subset)
        for v in V_NAMES:
            subset = [r for r in rows if bin_of(r)[1] == v]
            if subset:
                block["cells"][f"ALL_ELEV|{v}"] = summarise(subset)
        block["cells"]["ALL_ELEV|ALL_V"] = summarise(rows)
        result["seeds"][f"seed{seed}"] = block

        log(f"--- seed{seed}  n={len(rows)}  lambda_theta={weight} ---")
        header = (f"{'cell':<14}{'n':>5}{'corner':>8}{'frs':>8}{'scale':>8}"
                  f"{'R':>8}{'t':>8}{'5cm5':>8}{'thR%':>8}{'tht%':>8}")
        log(header)
        for elevation in ELEV_NAMES:
            for v in V_NAMES:
                e = block["cells"].get(f"{elevation}|{v}")
                if not e:
                    continue
                flag = " " if e["n"] >= MIN_CELL else "*"
                log(f"{elevation+'|'+v:<14}{e['n']:>5}{e['corner_error']:>8.3f}"
                    f"{e['front_rear_shift']:>8.3f}{e['affine_scale_gap']:>8.4f}"
                    f"{e['R']:>8.2f}{e['t']:>8.4f}{e['success_5cm5deg']:>8.4f}"
                    f"{e.get('theta_gain_R_pct', float('nan')):>8.1f}"
                    f"{e.get('theta_gain_t_pct', float('nan')):>7.1f}{flag}")
    path = OUT / "data_risk_map.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


if __name__ == "__main__":
    main()
