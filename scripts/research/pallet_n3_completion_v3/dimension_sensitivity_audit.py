"""Pre-report audit that each final N3 head responds to W,D,H alone."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import torch

from . import common as C
from .integrity import DIMENSION_A, DIMENSION_B, measure_dimension_sensitivity


OUTPUT = C.DOC / "DIMENSION_SENSITIVITY_AUDIT.json"
SCHEMA = "pallet_n3_completion_v3_dimension_sensitivity_audit_v1"


def summarize(fits: Mapping[str, Mapping]) -> dict:
    statuses = [row.get("status") for row in fits.values()]
    changed = [int(row.get("changed_logit_entries", 0)) for row in fits.values()]
    all_six_changed = len(changed) == 6 and all(value > 0 for value in changed)
    return {
        "status": ("PASS" if statuses == ["PASS"] * 6 and all_six_changed
                   else "BLOCKED_INTEGRITY"),
        "fits_verified": sum(status == "PASS" for status in statuses),
        "changed_logit_entries_per_fit": changed,
        "all_six_changed": all_six_changed,
        "total_changed_logit_entries": sum(changed),
    }


def build(*, measure_fn: Callable = measure_dimension_sensitivity) -> dict:
    normalization = C.read(C.NORMALIZATION)
    fits = {}
    for backbone in C.CONFIGS:
        for seed in C.SEEDS:
            key = f"{backbone}_seed{seed}"
            receipt_path = C.DOC / f"TRAIN_{backbone.upper()}_SEED{seed}.json"
            receipt = C.read(receipt_path)
            checkpoint_path = C.verify(receipt["checkpoint"])
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
            if (receipt.get("complete") is not True or receipt.get("steps") != C.STEPS
                    or checkpoint.get("complete") is not True
                    or checkpoint.get("step") != C.STEPS):
                raise RuntimeError(f"incomplete final fit: {key}")
            result = measure_fn(
                backbone, checkpoint["model_state_dict"], normalization,
                receipt["checkpoint"]["sha256"])
            result.update(
                seed=seed,
                fit_receipt=C.binding(receipt_path),
                checkpoint=receipt["checkpoint"],
            )
            fits[key] = result
    summary = summarize(fits)
    if summary["status"] != "PASS":
        raise RuntimeError("one or more final heads failed dimension sensitivity")
    return {
        "schema": SCHEMA,
        "complete": True,
        "contract": (
            "visual tensors, initial points, box, validity and base logits are "
            "bitwise identical; only registered W,D,H changes"
        ),
        "dimension_a_wdh_m": DIMENSION_A,
        "dimension_b_wdh_m": DIMENSION_B,
        "normalization": C.binding(C.NORMALIZATION),
        "measurement_source": C.binding(C.HERE / "integrity.py"),
        "fits": fits,
        "summary": summary,
        "claim_boundary": (
            "This proves the trained metadata path is active. It does not by "
            "itself identify the causal accuracy gain of dimensions."
        ),
    }


def main() -> None:
    payload = build()
    C.write(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
