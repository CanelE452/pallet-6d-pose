"""Create and verify the immutable N3 v3 execution protocol."""
from __future__ import annotations

import json
from pathlib import Path

from . import common as C
from .resnet_constant_adapter import (
    CHECKPOINT_SHA256 as RESNET_CHECKPOINT_SHA256,
    COMPLETION as RESNET_COMPLETION,
    PROTOCOL as RESNET_PROTOCOL,
    RGB_MODEL_SOURCE,
    load_constant_contract,
    recipe as resnet_recipe,
)


CONTRACT = C.DOC / "contract"
CONTRACT_HASHES = CONTRACT / "PACKAGE_SHA256.json"
DOPE_CACHE_RECEIPT = (
    C.ROOT / "_docs/experiments/pallet_dope_refiner_20261001_v1/"
    "SOURCE_CACHE_COMPLETE.json")
DOPE_ADAPTER = (
    C.ROOT / "scripts/research/pallet_dope_refiner_20261001_v1/dope_adapter.py")
TRAINING_CODE = (
    "common.py", "model.py", "resnet_constant_adapter.py", "adapters.py",
    "data.py", "train.py", "selection.py", "protocol.py",
)


def verify_contract_package() -> list[dict]:
    expected = C.read(CONTRACT_HASHES)
    if set(expected) != {
            "EXECUTE_PALLET_N3_V3_KO.txt", "README_KO.md",
            "source_context/REVISION_NOTES_KO.md",
            "source_context/code_contract_v3.json",
            "source_context/manuscript_ko_v3.md",
            "source_context/number_sources.json"}:
        raise RuntimeError("Attached execution-package inventory changed")
    bindings = []
    for relative, identity in sorted(expected.items()):
        path = CONTRACT / relative
        actual = C.binding(path)
        if (actual["sha256"] != identity["sha256"]
                or actual["bytes"] != identity["bytes"]):
            raise RuntimeError(f"Attached execution-package hash mismatch: {relative}")
        bindings.append(actual)
    bindings.append(C.binding(CONTRACT_HASHES))
    return bindings


def build() -> dict:
    contract_bindings = verify_contract_package()
    source = C.read(C.SOURCE)
    records = source.get("records", [])
    counts = {partition: sum(record.get("partition") == partition for record in records)
              for partition in ("train", "calibration", "selection", "heldout")}
    if counts != {"train": 55980, "calibration": 1004,
                  "selection": 1031, "heldout": 1985}:
        raise RuntimeError(f"Synthetic partition contract drift: {counts}")
    normalization = C.read(C.NORMALIZATION)
    if len(normalization.get("mean", [])) != 5 or len(normalization.get("scale", [])) != 5:
        raise RuntimeError("Dimension normalization is not a locked five-vector")
    dev = C.read(C.DEV)
    dev_records = (dev if isinstance(dev, list)
                   else dev.get("records", dev.get("items", [])))
    if len(dev_records) != 319:
        raise RuntimeError(f"DEV319 manifest drift: {len(dev_records)}")
    dope_receipt = C.read(DOPE_CACHE_RECEIPT)
    if not dope_receipt.get("complete") or dope_receipt.get("rows") != 60000:
        raise RuntimeError("DOPE source cache is not complete")
    load_constant_contract(load_checkpoint=False)

    bindings = []
    bindings.extend(contract_bindings)
    for path in (
            C.SOURCE, C.SIDECAR, C.NORMALIZATION, C.DEV,
            C.DOPE_WEIGHTS, DOPE_CACHE_RECEIPT, DOPE_ADAPTER,
            C.RESNET_CONSTANT, RESNET_PROTOCOL, RESNET_COMPLETION,
            RGB_MODEL_SOURCE):
        bindings.append(C.binding(path))
    bindings.extend(C.binding(C.HERE / filename) for filename in TRAINING_CODE)

    return {
        "schema": "pallet_n3_completion_v3_protocol_v1",
        "contract_package": {
            "status": "read_and_hash_verified",
            "instruction": C.binding(CONTRACT / "EXECUTE_PALLET_N3_V3_KO.txt"),
            "source_context_files": 4,
        },
        "purpose": (
            "Apply the same dimension-conditioned, symmetry-supervised N3 local "
            "correction to frozen DOPE and ResNet-18 estimators; evaluate every "
            "fixed seed and retain no-change, worsening, failure, and missing results."),
        "methods": {
            "proposal": "N3_DIM_SYM",
            "base_inputs": ["RGB"],
            "n3_inputs": [
                "frozen feature taps", "frozen initial nine points",
                "frozen selected box", "registered physical dimensions [W,D,H]",
            ],
            "n3_inference_forbidden_inputs": [
                "ground truth", "visibility", "symmetry branch label",
                "camera pose", "real-domain supervision",
            ],
            "preserved_outputs": [
                "selected instance", "box", "score", "point ordering",
                "center8", "missing-point mask",
            ],
        },
        "backbones": {
            "dope": {
                "checkpoint": C.binding(C.DOPE_WEIGHTS),
                "feature_channels": [256, 128], "feature_strides": [4, 8],
                "frozen": True, "retrained": False, "base_dimension_input": False,
            },
            "resnet18": {
                "source_checkpoint": C.binding(C.RESNET_CONSTANT),
                "source_checkpoint_expected_sha256": RESNET_CHECKPOINT_SHA256,
                "feature_channels": [128, 256], "feature_strides": [8, 16],
                "frozen": True, "retrained": False, "base_dimension_input": False,
                "adapter": resnet_recipe(),
                "deviation": (
                    "Uses the completed 10-epoch synthetic CONSTANT control, with "
                    "its z=0 FiLM algebraically folded into an image-only RGB final "
                    "convolution. It is explicitly not the collapsed 60-epoch RGB "
                    "checkpoint and not a newly trained 60-epoch baseline."),
            },
        },
        "n3": {
            "candidate_directions": 13, "candidate_radii": 17,
            "move_candidates": 221, "no_move_candidates": 1,
            "candidates_per_corner": 222, "refined_corners": list(range(8)),
            "center_index_preserved": 8,
            "candidate_radius_box_diagonal_fraction": .08,
            "stencil_fraction": .1310373991727829,
            "feature_sampling": "4x8 bilinear grid; two 1x1 projections; shared patch encoder; avg+max",
            "visual_hidden": 16, "visual_encoding": 24,
            "dimension_features": [
                "log(W)", "log(D)", "log(H)", "log(W/D)",
                "log(H/sqrt(W*D))",
            ],
            "dimension_encoder": "5->16->16 SiLU",
            "metadata_scorer": "[dimension16, role8, displacement2, null1]=27->32->1 SiLU",
            "metadata_final_layer_initialization": "all zeros",
            "additive_logits": True, "n4_symmetry_onehot": False,
            "decode": "softmax expectation", "lambda": C.LAMBDA,
            "max_move_original_image_diagonal_fraction": C.CAP_FRACTION,
            "inverse_transform": "full per-axis affine; translation excluded for displacements",
        },
        "symmetry_supervision": {
            "source": "approved whole-object permutations in bound sidecar",
            "selection_reference": "raw frozen base predictions",
            "identity_first_exact_tie": True,
            "missing_prediction_normalized_penalty": 1.0,
            "same_permutation_for_points_and_mask": True,
            "center8_preserved": True,
            "learned_branch_output": False,
            "independent_nearest_neighbor_matching": False,
            "target_sigma": ".08 * predicted_box_diagonal / 17",
            "objective": "candidate cross entropy",
            "inference_uses_symmetry_target": False,
        },
        "data": {
            "synthetic": counts, "total": len(records),
            "train_supervision": "synthetic only", "new_real_training_images": 0,
            "dimension_axis_order": ["W", "D", "H"], "dimension_unit": "metre",
            "normalization": C.binding(C.NORMALIZATION),
            "real_primary": {"name": "DEV319", "frames": 319,
                             "manifest": C.binding(C.DEV)},
        },
        "training": {
            "backbones": ["dope", "resnet18"], "seeds": list(C.SEEDS),
            "fits": 6, "steps_per_fit": C.STEPS, "batch": C.BATCH,
            "exposures_per_fit": C.STEPS * C.BATCH,
            "total_updates": 6 * C.STEPS,
            "total_exposures": 6 * C.STEPS * C.BATCH,
            "initialization": "fresh seed-specific visual N3; zero metadata final layer; no P warm start",
            "optimizer": {"name": "AdamW", **C.OPTIMIZER},
            "precision": {"model": "FP32", "AMP": False, "TF32": False},
            "checkpoint_every_updates": 500, "selection": "fixed final step6000 only",
            "smoke": "one discarded two-update run per backbone; never used as evidence",
            "outcome_driven_extension": False,
        },
        "selection": {
            "source": "synthetic calibration1004 only",
            "temperature_grid": list(C.TEMPERATURES),
            "fixed_lambda": C.LAMBDA, "fixed_cap_fraction": C.CAP_FRACTION,
            "no_real_selection": True, "no_seed_selection": True,
            "heldout_opened_after_temperature": True,
        },
        "evaluation": {
            "corner_set": list(range(8)),
            "primary_metrics": [
                "matched pooled median px", "matched pooled P90 px",
                "PCK@5/10/20 over all GT corners", "E_sym",
                "full-penalty median/P90", "translation cm", "rotation deg", "yaw deg",
            ],
            "seed_aggregation": "evaluate each seed; report mean and spread; no ensemble",
            "paired_bootstrap": {"unit": "session", "resamples": 10000,
                                 "seed": 20260917},
            "runtime": {"frames": 26, "warmup": 20, "repeats": 5, "seed": 1},
            "truthfulness": "retain improvement, no-change, worsening, failure, and x",
        },
        "excluded": [
            "new self-training", "actual lifter control", "PDF generation",
            "new manual annotation", "best-DEV checkpoint selection",
        ],
        "bindings": bindings,
    }


def freeze() -> dict:
    payload = build()
    # Convert the only tuple inherited from the optimizer constant to JSON.
    payload["training"]["optimizer"]["betas"] = list(
        payload["training"]["optimizer"]["betas"])
    C.write(C.DOC / "PROTOCOL.json", payload, freeze=True)
    return payload


def verify() -> dict:
    path = C.DOC / "PROTOCOL.json"
    payload = C.read(path)
    expected = json.loads(json.dumps(build(), allow_nan=False))
    if payload != expected:
        raise RuntimeError("Frozen N3 protocol differs from current inputs")
    for binding in payload["bindings"]:
        C.verify(binding)
    return payload


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    arguments = parser.parse_args()
    payload = freeze() if arguments.command == "freeze" else verify()
    print(json.dumps({
        "PASS": True, "schema": payload["schema"],
        "bindings": len(payload["bindings"]),
        "fits": payload["training"]["fits"],
        "total_updates": payload["training"]["total_updates"],
    }, indent=2))


if __name__ == "__main__":
    main()
