"""Recompute how often the frozen-base symmetry target selects a non-identity branch.

This is an evidence audit, not a training stage.  It calls the exact
``select_symmetric_target`` implementation used by ``n3_loss`` and applies it
to the row orders consumed by all six completed fits.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from . import common as C
from .data import _paths, load_arrays
from .model import select_symmetric_target


OUTPUT = C.DOC / "SYMMETRY_ACTIVATION_AUDIT.json"
ARRAY_KEYS = ("points", "point_valid", "boxes", "gt_points", "gt_valid")


def _counts(values: np.ndarray, *, width: int | None = None) -> dict[str, int]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    result = Counter(int(value) for value in values)
    keys = range(width) if width is not None else sorted(result)
    return {str(key): int(result.get(key, 0)) for key in keys}


def select_branches(arrays: Mapping[str, np.ndarray], permutations: np.ndarray,
                    group_valid: np.ndarray, rows: np.ndarray,
                    *, chunk_size: int = 4096) -> np.ndarray:
    """Run the exact training target selector on source row indices, on CPU."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64)
    if rows.min() < 0 or rows.max() >= len(permutations):
        raise ValueError("source row is outside the bound sidecar")
    result = np.empty(len(rows), dtype=np.int64)
    for begin in range(0, len(rows), chunk_size):
        selected = rows[begin:begin + chunk_size]
        box = torch.from_numpy(np.array(arrays["boxes"][selected], copy=True))
        # GenericPointRefiner._evidence, which supplied the training output,
        # computes this same FP32 diagonal and clamps it to one pixel.
        diagonal = torch.linalg.vector_norm(
            box[:, 2:] - box[:, :2], dim=-1).clamp_min(1)
        _, _, branch, _ = select_symmetric_target(
            torch.from_numpy(np.array(arrays["points"][selected], copy=True)),
            torch.from_numpy(np.array(arrays["point_valid"][selected], copy=True)),
            torch.from_numpy(np.array(arrays["gt_points"][selected], copy=True)),
            torch.from_numpy(np.array(arrays["gt_valid"][selected], copy=True)),
            torch.from_numpy(np.array(permutations[selected], copy=True)),
            torch.from_numpy(np.array(group_valid[selected], copy=True)),
            diagonal,
        )
        result[begin:begin + len(selected)] = branch.numpy()
    return result


def summarize(branches: np.ndarray, *, branch_width: int) -> dict:
    branches = np.asarray(branches, dtype=np.int64).reshape(-1)
    total = int(len(branches))
    non_identity = int((branches != 0).sum())
    return {
        "rows_or_exposures": total,
        "branch_counts": _counts(branches, width=branch_width),
        "identity_count": total - non_identity,
        "non_identity_count": non_identity,
        "non_identity_fraction": non_identity / total if total else None,
    }


def audit_backbone(backbone: str, permutations: np.ndarray,
                   group_valid: np.ndarray) -> dict:
    arrays = load_arrays(backbone)
    order_paths = {
        seed: C.RAW / "orders" / f"{backbone}_seed{seed}.npy"
        for seed in C.SEEDS
    }
    orders = {
        seed: np.load(path, mmap_mode="r").reshape(-1)
        for seed, path in order_paths.items()
    }
    expected_exposures = C.STEPS * C.BATCH
    for seed, order in orders.items():
        if len(order) != expected_exposures:
            raise RuntimeError(f"{backbone} seed{seed} order length drift")
    unique_sets = [np.unique(order) for order in orders.values()]
    if not all(np.array_equal(unique_sets[0], other) for other in unique_sets[1:]):
        raise RuntimeError(f"{backbone} seeds do not cover one common usable row set")
    unique_rows = unique_sets[0]
    unique_branches = select_branches(
        arrays, permutations, group_valid, unique_rows)
    branch_by_row = np.full(len(permutations), -1, dtype=np.int64)
    branch_by_row[unique_rows] = unique_branches

    _, cache_receipt = _paths(backbone)
    training_summary = C.DOC / f"TRAINING_{backbone.upper()}_COMPLETE.json"
    per_seed = {}
    for seed, order in orders.items():
        receipt = C.DOC / f"TRAIN_{backbone.upper()}_SEED{seed}.json"
        receipt_payload = C.read(receipt)
        order_binding = C.binding(order_paths[seed])
        if receipt_payload["order"] != order_binding:
            raise RuntimeError(f"{backbone} seed{seed} order binding drift")
        per_seed[str(seed)] = {
            **summarize(branch_by_row[np.asarray(order)],
                        branch_width=permutations.shape[1]),
            "order": order_binding,
            "fit_receipt": C.binding(receipt),
        }
    directory, _ = _paths(backbone)
    return {
        "unique_usable_rows": summarize(
            unique_branches, branch_width=permutations.shape[1]),
        "seeds": per_seed,
        "cache_receipt": C.binding(cache_receipt),
        "source_arrays": {
            key: C.binding(directory / f"{key}.npy") for key in ARRAY_KEYS
        },
        "training_summary": C.binding(training_summary),
    }


def build_audit() -> dict:
    sidecar = np.load(C.SIDECAR, allow_pickle=False)
    try:
        record_index = sidecar["record_index"]
        permutations = sidecar["permutations"]
        group_valid = sidecar["group_valid"]
        if not np.array_equal(record_index, np.arange(len(record_index))):
            raise RuntimeError("sidecar record_index drift")
        if permutations.shape != (len(record_index), 4, 9):
            raise RuntimeError("sidecar permutation shape drift")
        valid_counts = group_valid.sum(axis=1)
        if not group_valid[:, 0].all():
            raise RuntimeError("identity is not valid in every sidecar row")
        identity = np.arange(9, dtype=permutations.dtype)
        if not np.array_equal(permutations[:, 0],
                              np.broadcast_to(identity, permutations[:, 0].shape)):
            raise RuntimeError("identity is not the first sidecar permutation")
        payload = {
            "schema": "pallet_n3_completion_v3_symmetry_activation_audit_v1",
            "complete": True,
            "algorithm": {
                "implementation": "model.select_symmetric_target called on CPU",
                "cost": "mean normalized frozen-initial-point distance over valid corners0..7",
                "missing_initial_prediction_penalty": 1.0,
                "invalid_GT_points_excluded": True,
                "whole_object_permutation": True,
                "identity_first_exact_tie": True,
                "box_diagonal": "FP32 norm(box_max-box_min), clamped to >=1 pixel",
                "inference_use": False,
            },
            "source_sidecar": {
                "binding": C.binding(C.SIDECAR),
                "rows": int(len(record_index)),
                "permutation_slots": int(permutations.shape[1]),
                "valid_permutations_per_row_counts": _counts(valid_counts, width=5),
                "four_way_rows": int((valid_counts == 4).sum()),
            },
            "model_selector_source": C.binding(C.HERE / "model.py"),
            "refiner_evidence_source": C.binding(
                C.ROOT / "scripts/research/pallet_dope_refiner_20261001_v1/refiner.py"),
            "backbones": {
                backbone: audit_backbone(backbone, permutations, group_valid)
                for backbone in C.CONFIGS
            },
            "interpretation": {
                "objective_applied": True,
                "resnet18_non_identity_branch_exercised": False,
                "square_C4_training_target_exercised": bool((valid_counts == 4).any()),
                "claim_boundary": (
                    "The same symmetry-aware objective was applied. Non-identity target "
                    "activation frequency must be reported separately; this audit does not "
                    "establish an incremental symmetry benefit."
                ),
            },
        }
    finally:
        sidecar.close()
    return payload


def main() -> None:
    payload = build_audit()
    C.write(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
