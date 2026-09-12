"""Pre-performance D/H common-initialization and trace parity audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import canonical_sha, immutable_json, sha256

from .model import LocalLineFusionV2
from .runner import sequence, state_sha, utility_sha


def run(manifest: Path, output: Path, steps: int = 2000, batch: int = 8):
    dataset = ObservationDataset(manifest, targets=False); rows = []
    for seed in (1, 2, 3):
        direct = LocalLineFusionV2(arm="direct", common_seed=seed)
        hough = LocalLineFusionV2(arm="hough", common_seed=seed)
        order = sequence(len(dataset), steps * batch, seed)
        d_common, h_common = state_sha(direct), state_sha(hough)
        d_utility, h_utility = utility_sha(direct), utility_sha(hough)
        rows.append({"seed": seed, "direct_common_parameter_sha256": d_common,
                     "hough_common_parameter_sha256": h_common,
                     "common_parameters_byte_identical": d_common == h_common,
                     "direct_initial_utility_head_sha256": d_utility,
                     "hough_initial_utility_head_sha256": h_utility,
                     "utility_head_byte_identical": d_utility == h_utility,
                     "minibatch_order_sha256": canonical_sha(order),
                     "train_manifest_sha256": sha256(manifest)})
    value = {"schema": "symdht_local_init_parity_v2", "PASS": all(
        row["common_parameters_byte_identical"] and row["utility_head_byte_identical"] for row in rows),
        "steps": steps, "batch": batch, "rows": rows}
    immutable_json(output, value); return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); value = run(args.manifest.resolve(), args.output.resolve()); print("PASS" if value["PASS"] else "FAIL")
    if not value["PASS"]: raise SystemExit(1)


if __name__ == "__main__": main()
