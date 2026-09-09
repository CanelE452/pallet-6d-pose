"""§13 gate — the Jacobian must match finite differences before any loss is built."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pnp_jacobian import expm_so3, finite_difference_jacobian, pnp_jacobian, project

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1/JACOBIAN_VERIFICATION.json"

GATE_MEDIAN = 1e-4
GATE_P99 = 1e-2


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


def main() -> int:
    rng = np.random.default_rng(20260906)
    rel_errors, conds = [], []
    n_cases = 200
    for _ in range(n_cases):
        across = rng.uniform(0.5, 1.5)
        along = rng.uniform(0.5, 1.5)
        height = rng.uniform(0.08, 0.20)
        X = cuboid(across, height, along)
        f = rng.uniform(400, 1400)
        K = np.array([[f, 0, rng.uniform(280, 500)],
                      [0, f * rng.uniform(0.98, 1.02), rng.uniform(200, 380)],
                      [0, 0, 1.0]])
        R = expm_so3(rng.normal(scale=1.0, size=3))
        t = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                      rng.uniform(1.0, 6.0)])
        if (X @ R.T + t)[:, 2].min() < 0.2:
            continue
        Ja = pnp_jacobian(K, R, t, X)
        Jf = finite_difference_jacobian(K, R, t, X)
        # per-column, so a small rotation block cannot hide behind a large one
        col = np.abs(Ja - Jf).max(axis=0) / np.maximum(np.abs(Jf).max(axis=0), 1e-12)
        rel_errors.append(float(col.max()))
        conds.append(float(np.linalg.cond(Ja)))

    rel = np.array(rel_errors)
    result = {
        "schema_version": "pallet_translation_loss_v1_jacobian_verification_v1",
        "cases_requested": n_cases,
        "cases_used": int(rel.size),
        "max_relative_error": float(rel.max()),
        "median_relative_error": float(np.median(rel)),
        "p99_relative_error": float(np.percentile(rel, 99)),
        "condition_number_median": float(np.median(conds)),
        "condition_number_p99": float(np.percentile(conds, 99)),
        "gate_median_lt": GATE_MEDIAN,
        "gate_p99_lt": GATE_P99,
    }
    result["verdict"] = ("PASS" if result["median_relative_error"] < GATE_MEDIAN
                         and result["p99_relative_error"] < GATE_P99 else "FAIL")

    # zero-residual sanity: at the GT pose the projection reproduces itself
    X = cuboid(1.1, 0.11, 1.3)
    K = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1.0]])
    R, t = expm_so3(np.array([0.2, -0.3, 0.1])), np.array([0.05, -0.1, 2.5])
    u = project(K, R, t, X)
    result["zero_perturbation_reprojection_max_px"] = float(
        np.abs(project(K, expm_so3(np.zeros(3)) @ R, t, X) - u).max())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
