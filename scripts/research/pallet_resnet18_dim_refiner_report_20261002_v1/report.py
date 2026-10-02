"""Build a fail-closed Korean report for the frozen FULL ResNet18 refiner.

The writer is intentionally outside the sealed experiment namespace.  It does
not run a model or alter a sealed artifact.  Every production input is first
required and validated; only after that succeeds are report files rendered in
a temporary directory and promoted to the documentation tree.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pallet-pose-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.research.pallet_resnet18_dim_refiner_20261002_v1 import common as C


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / "_docs/experiments/pallet_resnet18_dim_refiner_report_20261002_v3"
SOURCE_DOC = C.DOC
SOURCE_RAW = C.RAW

PROTOCOL = SOURCE_DOC / "PROTOCOL.json"
CACHE = SOURCE_DOC / "SOURCE_CACHE_COMPLETE.json"
TRAINING = SOURCE_DOC / "TRAINING_COMPLETE.json"
VALIDATION = SOURCE_DOC / "VALIDATION_OUTPUTS.json"
SELECTION = SOURCE_DOC / "SELECTION.json"
HELDOUT = SOURCE_DOC / "SYNTHETIC_HELDOUT.json"
INFERENCE = SOURCE_DOC / "DEV_INFERENCE_COMPLETE.json"
PREDICTIONS = SOURCE_RAW / "DEV_PREDICTIONS.json"
EVALUATION_LOCK = SOURCE_DOC / "DEV_EVALUATION_LOCK.json"
DIMENSION_SUPPORT = SOURCE_DOC / "DIMENSION_SUPPORT.json"
RESULTS = SOURCE_DOC / "DEV_RESULTS.json"
PAIRED = SOURCE_DOC / "DEV_PAIRED_RESULTS.json"

DIRECT_DOC = ROOT / "_docs/experiments/pallet_resnet18_dimension_20261002_v1"
DIRECT_REPORT = DIRECT_DOC / "REPORT_KO_V2.md"
DIRECT_FIGURE = DIRECT_DOC / "DSNT_REAL_COMPARISON_V2.png"
DIRECT_MANIFEST = DIRECT_DOC / "DSNT_REPORT_MANIFEST_V2.json"
DIRECT_GALLERY = DIRECT_DOC / "GALLERY.md"
DIRECT_GALLERY_MANIFEST = DIRECT_DOC / "DSNT_GALLERY_MANIFEST.json"

BASELINE = "FULL"
ARMS = ("D0", "P0", "P5", "P5_CONSTANT")
SEEDS = (1, 2, 3)
METHODS = (BASELINE,) + tuple(f"{arm}_S{seed}" for arm in ARMS for seed in SEEDS)
SUBGROUPS = {"plastic": 194, "wood": 125}
SUBGROUP_LABELS = {
    "plastic": "Plastic 110×130×11 cm",
    "wood": "Wood 80×59×14 cm",
}
CONTRASTS = (
    "D0_minus_FULL",
    "P0_minus_FULL",
    "P5_minus_FULL",
    "P5_CONSTANT_minus_FULL",
    "P5_minus_P0",
    "P5_minus_P5_CONSTANT",
)
PRIMARY_CONTRASTS = (
    "P5_minus_FULL",
    "P5_minus_P0",
    "P5_minus_P5_CONSTANT",
)
COLORS = {
    "FULL": "#566573",
    "D0": "#d68910",
    "P0": "#2874a6",
    "P5": "#148f77",
    "P5_CONSTANT": "#884ea0",
}


class ReportContractError(RuntimeError):
    """A completed report cannot be produced from the supplied artifacts."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path.relative_to(ROOT.resolve())),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _binding_for_bytes(path: Path, data: bytes) -> dict:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportContractError(f"{label}: finite number required") from exc
    if not math.isfinite(number):
        raise ReportContractError(f"{label}: finite number required")
    return number


def _same_number(actual, expected, label: str, atol: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ReportContractError(f"{label}: nullability drift")
        return
    if not np.isclose(_finite(actual, label), _finite(expected, label), rtol=0, atol=atol):
        raise ReportContractError(f"{label}: {actual} != {expected}")


def required_inputs() -> tuple[Path, ...]:
    paths = [
        PROTOCOL, CACHE, TRAINING, VALIDATION, SELECTION, HELDOUT, INFERENCE,
        PREDICTIONS, EVALUATION_LOCK, DIMENSION_SUPPORT, RESULTS, PAIRED,
        DIRECT_REPORT, DIRECT_FIGURE, DIRECT_MANIFEST, DIRECT_GALLERY,
        DIRECT_GALLERY_MANIFEST,
    ]
    paths.extend(SOURCE_DOC / f"TRAIN_SEED{seed}.json" for seed in SEEDS)
    return tuple(paths)


def require_complete_inventory(paths: Iterable[Path] | None = None) -> None:
    """Fail before the output directory is touched when a receipt is absent."""
    paths = required_inputs() if paths is None else tuple(Path(path) for path in paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        joined = "\n  - ".join(missing)
        raise ReportContractError(f"Refiner report inputs are incomplete:\n  - {joined}")


def validate_protocol_shape(protocol: dict) -> None:
    expected_partitions = {"train": 55980, "calibration": 1004,
                           "selection": 1031, "heldout": 1985}
    checks = {
        "schema": "resnet18_full_dim_local_refiner_protocol_v1",
        "status": "SEALED_BEFORE_SOURCE_CACHE_OR_HEAD_TRAINING",
        "baseline_arm": BASELINE,
        "baseline_trainable": False,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "steps": 6000,
        "batch": 16,
        "fits": 12,
    }
    for key, expected in checks.items():
        if protocol.get(key) != expected:
            raise ReportContractError(f"Protocol {key} drift: {protocol.get(key)!r}")
    if protocol.get("training", {}).get("partitions") != expected_partitions:
        raise ReportContractError("Protocol partition counts drift")
    if protocol["training"].get("real_training_images") != 0:
        raise ReportContractError("Protocol permits real-image training")
    if protocol.get("calibration", {}).get("real_selection") is not False:
        raise ReportContractError("Calibration role drift")
    if protocol.get("selection", {}).get("real_selection") is not False:
        raise ReportContractError("Selection role drift")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("independent_TEST") is not False:
        raise ReportContractError("Unexpected independent TEST declaration")
    if evaluation.get("physical_6D_ground_truth") is not False:
        raise ReportContractError("Unexpected physical 6D GT declaration")
    interpretation = protocol.get("interpretation", {})
    if interpretation.get("all_arms_share_dimension_conditioned_FULL_baseline") is not True:
        raise ReportContractError("Shared FULL baseline interpretation drift")
    if "P5_minus_P5_CONSTANT" not in interpretation:
        raise ReportContractError("Primary incremental-dimension contrast is absent")


def _verify_binding(value: dict, label: str) -> Path:
    try:
        return C.verify_binding(value)
    except Exception as exc:
        raise ReportContractError(f"Invalid binding: {label}") from exc


def _validate_method_summary(summary: dict, label: str, denominator: int) -> None:
    required = (
        "median_px", "p90_px", "matched_frames", "full_frame_denominator",
        "failed_or_excluded_frames", "supervised_points", "gt_denominator",
        "missing_supervised_keypoints", "ALL_GT_PCK", "pose",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ReportContractError(f"{label}: missing keys {missing}")
    if summary["full_frame_denominator"] != denominator:
        raise ReportContractError(f"{label}: population denominator drift")
    matched = int(summary["matched_frames"])
    failed = int(summary["failed_or_excluded_frames"])
    if matched + failed != denominator:
        raise ReportContractError(f"{label}: matched/failure accounting drift")
    gt_denominator = int(summary["gt_denominator"])
    supervised = int(summary["supervised_points"])
    missing = int(summary["missing_supervised_keypoints"])
    if gt_denominator <= 0 or not 0 <= supervised <= gt_denominator:
        raise ReportContractError(f"{label}: supervised-point denominator drift")
    if not 0 <= missing <= gt_denominator:
        raise ReportContractError(f"{label}: missing-keypoint count drift")
    for key in ("median_px", "p90_px"):
        _finite(summary[key], f"{label}.{key}")
    for threshold in ("5", "10", "20"):
        value = _finite(summary["ALL_GT_PCK"][threshold], f"{label}.PCK{threshold}")
        if not 0 <= value <= 1:
            raise ReportContractError(f"{label}.PCK{threshold}: outside [0,1]")
    pose = summary["pose"]
    if int(pose.get("denominator", -1)) != denominator:
        raise ReportContractError(f"{label}: pose denominator drift")
    if int(pose.get("n", -1)) + int(pose.get("failure_count", -1)) != denominator:
        raise ReportContractError(f"{label}: pose coverage accounting drift")
    _same_number(pose.get("coverage"), pose["n"] / denominator, f"{label}.pose.coverage")
    for key in (
        "rotation_median_deg", "translation_median_cm", "yaw_median_deg",
        "translation_p90_cm", "rotation_p90_deg", "yaw_p90_deg",
        "iou3d_median", "add_sym_auc_conditional", "add_sym_auc_full_population",
    ):
        if pose["n"]:
            _finite(pose[key], f"{label}.pose.{key}")


def _validate_paired_stat(row: dict, label: str, *, seeds_expected: bool = True,
                          expected_units: int = 13) -> None:
    if row.get("status") != "COMPLETE":
        raise ReportContractError(f"{label}: paired statistic is not complete")
    for key in ("delta", "low", "high"):
        _finite(row.get(key), f"{label}.{key}")
    if row["low"] > row["high"]:
        raise ReportContractError(f"{label}: inverted confidence interval")
    if row.get("frames") != 319 or row.get("units") != expected_units:
        raise ReportContractError(f"{label}: reused DEV population drift")
    if row.get("resamples") != 10000 or row.get("random_seed") != 20260914:
        raise ReportContractError(f"{label}: bootstrap contract drift")
    if seeds_expected and row.get("seeds") != 3:
        raise ReportContractError(f"{label}: seed count drift")


def _method_arm(method: str) -> tuple[str, int | None]:
    if method == BASELINE:
        return BASELINE, None
    arm, seed = method.rsplit("_S", 1)
    return arm, int(seed)


def _mean(values: list[float | int | None]) -> float | None:
    if any(value is None for value in values):
        return None
    numbers = [_finite(value, "seed mean") for value in values]
    return float(np.mean(numbers))


def _mean_summary(methods: dict, arm: str, *, subgroup: str | None = None) -> dict:
    rows = []
    for seed in SEEDS:
        item = methods[f"{arm}_S{seed}"]
        if subgroup is not None:
            item = item["by_object"][subgroup]
            pose = methods[f"{arm}_S{seed}"]["pose_by_object"][subgroup]
        else:
            pose = item["pose"]
        rows.append((item, pose))
    return {
        "median_px": _mean([row[0]["median_px"] for row in rows]),
        "p90_px": _mean([row[0]["p90_px"] for row in rows]),
        "matched_frames": _mean([row[0]["matched_frames"] for row in rows]),
        "ALL_GT_PCK": {
            threshold: _mean([row[0]["ALL_GT_PCK"][threshold] for row in rows])
            for threshold in ("5", "10", "20")
        },
        "pose": {
            key: _mean([row[1].get(key) for row in rows])
            for key in (
                "translation_median_cm", "rotation_median_deg", "yaw_median_deg",
                "translation_p90_cm", "rotation_p90_deg", "yaw_p90_deg",
                "iou3d_median", "add_sym_auc_conditional",
                "add_sym_auc_full_population", "coverage", "failure_count",
            )
        },
    }


def _validate_seed_means(results: dict) -> None:
    supplied = results.get("seed_mean", {})
    if set(supplied) != set(ARMS):
        raise ReportContractError("Seed-mean arm set drift")
    for arm in ARMS:
        expected = _mean_summary(results["methods"], arm)
        actual = supplied[arm]
        for key in ("median_px", "p90_px", "matched_frames"):
            _same_number(actual[key], expected[key], f"seed_mean.{arm}.{key}")
        for threshold in ("5", "10", "20"):
            _same_number(actual["ALL_GT_PCK"][threshold], expected["ALL_GT_PCK"][threshold],
                         f"seed_mean.{arm}.PCK{threshold}")
        for key, value in expected["pose"].items():
            _same_number(actual["pose"][key], value, f"seed_mean.{arm}.pose.{key}")


def validate_core_results(results: dict, paired: dict) -> None:
    if (results.get("complete") is not True
            or results.get("schema") != "resnet18_full_dim_refiner_dev_results_v1"
            or results.get("role") != "REUSED_DEV"):
        raise ReportContractError("DEV result role/schema is incomplete")
    if (results.get("positive_frames") != 319 or results.get("sessions") != 13
            or results.get("gt_denominator") != 2818):
        raise ReportContractError("DEV319 counts drift")
    if results.get("object_subgroups") != SUBGROUPS:
        raise ReportContractError("Plastic/wood subgroup counts drift")
    if list(results.get("methods", {})) != list(METHODS):
        raise ReportContractError("Exact FULL plus twelve seeded method order is required")
    if (results.get("same_conditional_support") is not True
            or results.get("negative_images_evaluated") != 0
            or results.get("AP_claim") is not False
            or results.get("new_image_forwards") != 0
            or results.get("geometry_derived_reference_not_independent_physical_metrology") is not True):
        raise ReportContractError("Evaluation interpretation contract drift")
    for method in METHODS:
        value = results["methods"][method]
        _validate_method_summary(value, method, 319)
        if int(value["gt_denominator"]) != 2818:
            raise ReportContractError(f"{method}: full GT denominator drift")
        if set(value.get("by_object", {})) != set(SUBGROUPS):
            raise ReportContractError(f"{method}: subgroup set drift")
        if set(value.get("pose_by_object", {})) != set(SUBGROUPS):
            raise ReportContractError(f"{method}: pose subgroup set drift")
        for subgroup, denominator in SUBGROUPS.items():
            subgroup_value = dict(value["by_object"][subgroup])
            subgroup_value["pose"] = value["pose_by_object"][subgroup]
            _validate_method_summary(subgroup_value, f"{method}/{subgroup}", denominator)
        for key in ("gt_denominator", "supervised_points", "missing_supervised_keypoints",
                    "matched_frames", "failed_or_excluded_frames"):
            subgroup_total = sum(int(value["by_object"][name][key]) for name in SUBGROUPS)
            if subgroup_total != int(value[key]):
                raise ReportContractError(f"{method}: subgroup {key} accounting drift")
        for key in ("n", "failure_count"):
            subgroup_total = sum(int(value["pose_by_object"][name][key]) for name in SUBGROUPS)
            if subgroup_total != int(value["pose"][key]):
                raise ReportContractError(f"{method}: subgroup pose {key} accounting drift")
        for threshold in ("5", "10", "20"):
            weighted = sum(
                float(value["by_object"][name]["ALL_GT_PCK"][threshold])
                * int(value["by_object"][name]["gt_denominator"])
                for name in SUBGROUPS
            ) / 2818
            _same_number(value["ALL_GT_PCK"][threshold], weighted,
                         f"{method}: subgroup PCK{threshold} accounting")
    matched = {results["methods"][method]["matched_frames"] for method in METHODS}
    if len(matched) != 1:
        raise ReportContractError("Conditional support differs despite the fixed-box contract")
    _validate_seed_means(results)

    if (paired.get("complete") is not True
            or paired.get("strict_support_no_silent_intersection") is not True
            or set(paired.get("results", {})) != set(CONTRASTS)):
        raise ReportContractError("Paired comparison set/role drift")
    for contrast in CONTRASTS:
        value = paired["results"][contrast]
        for level in ("session", "frame"):
            row = value["conditional_keypoint_median"][level]
            _validate_paired_stat(row, f"{contrast}.2D.{level}",
                                  expected_units=13 if level == "session" else 319)
            if row.get("level") != level or row.get("units") != (13 if level == "session" else 319):
                raise ReportContractError(f"{contrast}.2D.{level}: resampling unit drift")
        for threshold in ("5", "10", "20"):
            _validate_paired_stat(value["ALL_GT_PCK"][threshold],
                                  f"{contrast}.PCK{threshold}", seeds_expected=False)
        for metric in ("translation_error_cm", "rotation_error_deg", "yaw_error_deg", "iou3d"):
            _validate_paired_stat(value["pose"][metric], f"{contrast}.pose.{metric}")
        _validate_paired_stat(value["pose"]["coverage"],
                              f"{contrast}.pose.coverage", seeds_expected=False)


def _verify_nested_bindings(value, label: str) -> None:
    if isinstance(value, dict):
        if {"path", "sha256"}.issubset(value):
            _verify_binding(value, label)
            return
        for key, child in value.items():
            _verify_nested_bindings(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_nested_bindings(child, f"{label}[{index}]")


def _validate_direct_context() -> dict:
    manifest = _read(DIRECT_MANIFEST)
    if (manifest.get("complete") is not True
            or manifest.get("schema") != "resnet18_dimension_dsnt_three_arm_report_v2"
            or manifest.get("arms") != ["CONSTANT", "SHAPE", "FULL"]):
        raise ReportContractError("Direct-ResNet V2 context manifest drift")
    if manifest.get("outputs", {}).get("report") != _binding(DIRECT_REPORT):
        raise ReportContractError("Direct-ResNet V2 report binding drift")
    if manifest.get("outputs", {}).get("real_figure") != _binding(DIRECT_FIGURE):
        raise ReportContractError("Direct-ResNet V2 figure binding drift")
    _verify_nested_bindings({
        "generator": manifest.get("generator"),
        "protocol": manifest.get("protocol"),
        "inputs": manifest.get("inputs"),
        "outputs": manifest.get("outputs"),
    }, "direct_manifest")
    interpretation = manifest.get("interpretation", {})
    expected = {
        "independent_confirmation": False,
        "physical_pose_GT": False,
        "square_dimension_effect_identifiable": False,
        "latency_measured": False,
        "forced_single_pallet_output": True,
        "one_training_seed_per_arm": True,
    }
    if interpretation != expected:
        raise ReportContractError("Direct-ResNet V2 interpretation drift")
    gallery = _read(DIRECT_GALLERY_MANIFEST)
    if (gallery.get("complete") is not True
            or gallery.get("schema") != "resnet18_dimension_dsnt_rgb_gallery_v1"
            or gallery.get("selection", {}).get("aggregate_evidence") is not False
            or gallery.get("selection", {}).get("outcome_selected") is not True):
        raise ReportContractError("Direct-ResNet gallery interpretation drift")
    _verify_nested_bindings({
        "generator": gallery.get("generator"),
        "datasets": gallery.get("datasets"),
        "symmetry_contract": gallery.get("symmetry_contract"),
        "registry": gallery.get("registry"),
        "inputs": gallery.get("inputs"),
        "outputs": gallery.get("outputs"),
    }, "direct_gallery")
    return {"manifest": manifest, "gallery": gallery}


def load_and_validate() -> dict:
    """Read-only, CPU-only preflight. It never creates a report output."""
    require_complete_inventory()
    protocol = C.verify_protocol(PROTOCOL)
    validate_protocol_shape(protocol)

    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.source_cache import (
        _verify_completion,
    )
    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.selection import (
        validate_selection_payload,
        validate_validation_receipt,
    )
    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.inference import (
        validate_heldout_receipt,
        validate_prediction_payload,
    )

    cache = _verify_completion(_read(CACHE))
    training = _read(TRAINING)
    expected_training = {
        "complete": True,
        "fits": 12,
        "steps_per_fit": 6000,
        "total_head_updates": 72000,
        "nominal_head_exposures": 1152000,
        "physical_shared_backbone_exposures": 288000,
        "real_training_images": 0,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ReportContractError(f"Training {key} drift")
    if (training.get("baseline") != protocol["baseline"]
            or training.get("protocol") != C.binding(PROTOCOL)):
        raise ReportContractError("Training provenance drift")
    receipts = []
    for seed in SEEDS:
        path = SOURCE_DOC / f"TRAIN_SEED{seed}.json"
        receipt = _read(path)
        if (receipt.get("complete") is not True or receipt.get("seed") != seed
                or receipt.get("updates_per_arm") != 6000
                or receipt.get("exposures_per_arm") != 96000
                or receipt.get("arms") != list(ARMS)
                or receipt.get("real_training_images") != 0
                or receipt.get("backbone_retrained") is not False
                or receipt.get("final_checkpoint_only") is not True
                or receipt.get("paired_shared_frozen_features") is not True):
            raise ReportContractError(f"Seed {seed} training receipt drift")
        if set(receipt.get("parameters", {})) != set(ARMS):
            raise ReportContractError(f"Seed {seed} parameter set drift")
        for arm, count in receipt["parameters"].items():
            if not isinstance(count, int) or count <= 0:
                raise ReportContractError(f"Seed {seed}/{arm} invalid parameter count")
        for key in ("checkpoint", "order", "metrics"):
            _verify_binding(receipt[key], f"seed{seed}.{key}")
        metrics = _read(C.verify_binding(receipt["metrics"]))
        updates = metrics.get("updates", [])
        if metrics.get("seed") != seed or len(updates) != 6000:
            raise ReportContractError(f"Seed {seed} metric history length drift")
        if [row.get("step") for row in updates] != list(range(1, 6001)):
            raise ReportContractError(f"Seed {seed} metric step order drift")
        for row in updates:
            if set(row.get("arms", {})) != set(ARMS):
                raise ReportContractError(f"Seed {seed} metric arm set drift")
            for arm in ARMS:
                _finite(row["arms"][arm].get("loss"), f"seed{seed}/{arm}.loss")
        receipts.append(receipt)
    if training.get("seeds") != receipts:
        raise ReportContractError("Training aggregate/seed receipt drift")
    if any(receipt.get("usable_training_rows") != cache["partition_counts"]["train"]["usable"]
           for receipt in receipts):
        raise ReportContractError("Training usable-row count drift")

    # Recover the exact ordered validation row IDs from the completed cache.
    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.source_cache import FeatureDataset
    data = FeatureDataset()
    try:
        validation_rows = data.validation_rows
        validation = validate_validation_receipt(_read(VALIDATION), validation_rows)
    finally:
        data.close()
    selection = validate_selection_payload(
        _read(SELECTION), protocol=protocol, validation_binding=C.binding(VALIDATION))
    heldout = validate_heldout_receipt(_read(HELDOUT))
    predictions = validate_prediction_payload(_read(PREDICTIONS), protocol=protocol)
    inference = _read(INFERENCE)
    if (inference.get("complete") is not True or inference.get("frames") != 319
            or inference.get("methods") != 13 or inference.get("real_accuracy_read") is not False
            or inference.get("predictions") != C.binding(PREDICTIONS)):
        raise ReportContractError("DEV inference receipt drift")

    evaluation_lock = _read(EVALUATION_LOCK)
    if (evaluation_lock.get("complete") is not True
            or evaluation_lock.get("positive_frames") != 319
            or evaluation_lock.get("sessions") != 13
            or evaluation_lock.get("supervised_keypoints") != 2818
            or evaluation_lock.get("object_subgroups") != SUBGROUPS
            or evaluation_lock.get("new_image_forwards") != 0
            or evaluation_lock.get("negative_images_evaluated") != 0
            or evaluation_lock.get("AP_claim") is not False):
        raise ReportContractError("DEV evaluation lock drift")
    for key in ("code", "predictions", "dimension_support"):
        _verify_binding(evaluation_lock[key], f"evaluation_lock.{key}")
    evaluation_code = C.binding(C.HERE / "evaluation.py")
    expected_lock_bindings = {
        "code": evaluation_code,
        "predictions": C.binding(PREDICTIONS),
        "dimension_support": C.binding(DIMENSION_SUPPORT),
    }
    for key, expected in expected_lock_bindings.items():
        if evaluation_lock.get(key) != expected:
            raise ReportContractError(f"Evaluation-lock {key} provenance drift")
    for binding in evaluation_lock.get("sources", []):
        _verify_binding(binding, "evaluation source")

    dimension_support = _read(DIMENSION_SUPPORT)
    if (dimension_support.get("source_train_rows") != 55980
            or set(dimension_support.get("objects", {})) != set(SUBGROUPS)):
        raise ReportContractError("Dimension support population drift")
    for key in ("source_manifest", "dimension_sidecar", "normalization"):
        _verify_binding(dimension_support[key], f"dimension_support.{key}")

    results, paired = _read(RESULTS), _read(PAIRED)
    validate_core_results(results, paired)
    if results.get("predictions") != C.binding(PREDICTIONS):
        raise ReportContractError("Result/prediction binding drift")
    if paired.get("predictions") != results["predictions"]:
        raise ReportContractError("Paired/result prediction binding drift")
    for key in ("code", "evaluation_lock", "predictions", "dimension_support",
                "frame_results", "full_precision_errors", "paired_results"):
        _verify_binding(results[key], f"results.{key}")
    for method, binding in results.get("pose_artifacts", {}).items():
        if method not in METHODS:
            raise ReportContractError("Unexpected pose artifact method")
        _verify_binding(binding, f"pose.{method}")
    if set(results.get("pose_artifacts", {})) != set(METHODS):
        raise ReportContractError("Pose artifact method set drift")
    for binding in results.get("source_bindings", []):
        _verify_binding(binding, "result source")
    for key in ("code", "predictions"):
        _verify_binding(paired[key], f"paired.{key}")
    expected_result_bindings = {
        "code": evaluation_code,
        "evaluation_lock": C.binding(EVALUATION_LOCK),
        "predictions": C.binding(PREDICTIONS),
        "dimension_support": C.binding(DIMENSION_SUPPORT),
        "paired_results": C.binding(PAIRED),
    }
    for key, expected in expected_result_bindings.items():
        if results.get(key) != expected:
            raise ReportContractError(f"Result {key} provenance drift")
    if paired.get("code") != evaluation_code:
        raise ReportContractError("Paired-result code provenance drift")

    direct = _validate_direct_context()
    return {
        "protocol": protocol,
        "cache": cache,
        "training": training,
        "training_receipts": receipts,
        "validation": validation,
        "selection": selection,
        "heldout": heldout,
        "predictions": predictions,
        "inference": inference,
        "evaluation_lock": evaluation_lock,
        "dimension_support": dimension_support,
        "results": results,
        "paired": paired,
        "direct": direct,
    }


def _flat_method_row(method: str, summary: dict, pose: dict | None = None,
                     subgroup: str = "DEV319") -> dict:
    arm, seed = _method_arm(method)
    pose = summary["pose"] if pose is None else pose
    return {
        "subgroup": subgroup,
        "method": method,
        "arm": arm,
        "seed": seed,
        "frames": summary["full_frame_denominator"],
        "matched_frames": summary["matched_frames"],
        "median_px": summary["median_px"],
        "p90_px": summary["p90_px"],
        "PCK5": summary["ALL_GT_PCK"]["5"],
        "PCK10": summary["ALL_GT_PCK"]["10"],
        "PCK20": summary["ALL_GT_PCK"]["20"],
        "pose_coverage": pose["coverage"],
        "translation_median_cm": pose["translation_median_cm"],
        "translation_p90_cm": pose["translation_p90_cm"],
        "rotation_median_deg": pose["rotation_median_deg"],
        "rotation_p90_deg": pose["rotation_p90_deg"],
        "yaw_median_deg": pose["yaw_median_deg"],
        "iou3d_median": pose["iou3d_median"],
        "add_sym_auc_full_population": pose["add_sym_auc_full_population"],
    }


def build_tables(data: dict) -> dict:
    methods = data["results"]["methods"]
    method_rows = [_flat_method_row(method, methods[method]) for method in METHODS]
    arm_means = []
    baseline = _flat_method_row(BASELINE, methods[BASELINE])
    arm_means.append(dict(baseline, row_type="single_frozen_baseline"))
    for arm in ARMS:
        summary = _mean_summary(methods, arm)
        arm_means.append({
            "subgroup": "DEV319", "method": f"{arm}_SEED_MEAN", "arm": arm,
            "seed": None, "frames": 319, "matched_frames": summary["matched_frames"],
            "median_px": summary["median_px"], "p90_px": summary["p90_px"],
            "PCK5": summary["ALL_GT_PCK"]["5"],
            "PCK10": summary["ALL_GT_PCK"]["10"],
            "PCK20": summary["ALL_GT_PCK"]["20"],
            "pose_coverage": summary["pose"]["coverage"],
            "translation_median_cm": summary["pose"]["translation_median_cm"],
            "translation_p90_cm": summary["pose"]["translation_p90_cm"],
            "rotation_median_deg": summary["pose"]["rotation_median_deg"],
            "rotation_p90_deg": summary["pose"]["rotation_p90_deg"],
            "yaw_median_deg": summary["pose"]["yaw_median_deg"],
            "iou3d_median": summary["pose"]["iou3d_median"],
            "add_sym_auc_full_population": summary["pose"]["add_sym_auc_full_population"],
            "row_type": "mean_of_three_seed_statistics_not_ensemble",
        })

    subgroup_rows = []
    for subgroup, denominator in SUBGROUPS.items():
        for method in METHODS:
            summary = methods[method]["by_object"][subgroup]
            pose = methods[method]["pose_by_object"][subgroup]
            subgroup_rows.append(_flat_method_row(method, summary, pose, subgroup))
        for arm in ARMS:
            summary = _mean_summary(methods, arm, subgroup=subgroup)
            subgroup_rows.append({
                "subgroup": subgroup, "method": f"{arm}_SEED_MEAN", "arm": arm,
                "seed": None, "frames": denominator,
                "matched_frames": summary["matched_frames"],
                "median_px": summary["median_px"], "p90_px": summary["p90_px"],
                "PCK5": summary["ALL_GT_PCK"]["5"],
                "PCK10": summary["ALL_GT_PCK"]["10"],
                "PCK20": summary["ALL_GT_PCK"]["20"],
                "pose_coverage": summary["pose"]["coverage"],
                "translation_median_cm": summary["pose"]["translation_median_cm"],
                "translation_p90_cm": summary["pose"]["translation_p90_cm"],
                "rotation_median_deg": summary["pose"]["rotation_median_deg"],
                "rotation_p90_deg": summary["pose"]["rotation_p90_deg"],
                "yaw_median_deg": summary["pose"]["yaw_median_deg"],
                "iou3d_median": summary["pose"]["iou3d_median"],
                "add_sym_auc_full_population": summary["pose"]["add_sym_auc_full_population"],
                "row_type": "mean_of_three_seed_statistics_not_ensemble",
            })

    paired_rows = []
    for contrast in CONTRASTS:
        value = data["paired"]["results"][contrast]
        mapping = {
            "conditional_2D_median_px": value["conditional_keypoint_median"]["session"],
            "ALL_GT_PCK5": value["ALL_GT_PCK"]["5"],
            "ALL_GT_PCK10": value["ALL_GT_PCK"]["10"],
            "ALL_GT_PCK20": value["ALL_GT_PCK"]["20"],
            "translation_median_cm": value["pose"]["translation_error_cm"],
            "rotation_median_deg": value["pose"]["rotation_error_deg"],
            "yaw_median_deg": value["pose"]["yaw_error_deg"],
            "iou3d_median": value["pose"]["iou3d"],
            "pose_coverage": value["pose"]["coverage"],
        }
        for metric, row in mapping.items():
            paired_rows.append({
                "contrast": contrast, "metric": metric,
                "delta": row["delta"], "ci95_low": row["low"],
                "ci95_high": row["high"], "units": row["units"],
                "resamples": row["resamples"], "status": row["status"],
            })

    training_rows = []
    for receipt in data["training_receipts"]:
        for arm in ARMS:
            training_rows.append({
                "seed": receipt["seed"], "arm": arm,
                "updates": receipt["updates_per_arm"],
                "exposures": receipt["exposures_per_arm"],
                "usable_training_rows": receipt["usable_training_rows"],
                "parameters": receipt["parameters"][arm],
                "shared_seed_elapsed_seconds": receipt["elapsed_seconds"],
                "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            })
    selection_rows = []
    for arm in ARMS:
        rule = data["selection"]["rules"][arm]
        selection_rows.append({
            "arm": arm, "lambda": rule["lam"],
            "max_move_image_diagonal_fraction": rule["max_move_image_diagonal_fraction"],
            "synthetic_selection_score": rule["score"],
            "temperature_by_seed": {
                str(seed): data["selection"]["temperatures"][f"{arm}_S{seed}"]
                for seed in SEEDS
            },
        })
    heldout_rows = []
    for method in METHODS:
        row = data["heldout"]["results"][method]
        heldout_rows.append(dict(method=method, **row))
    return {
        "schema": "resnet18_full_dim_refiner_report_tables_v1",
        "complete": True,
        "roles": {
            "training": "synthetic TRAIN only",
            "selection": "synthetic calibration/selection only",
            "heldout": "synthetic heldout; not independent backbone-development test",
            "real": "REUSED_DEV",
        },
        "seed_policy": "arm rows are arithmetic means of three per-seed statistics; no ensemble prediction",
        "method_rows": method_rows,
        "arm_mean_rows": arm_means,
        "subgroup_rows": subgroup_rows,
        "paired_session_rows": paired_rows,
        "training_rows": training_rows,
        "selection_rows": selection_rows,
        "synthetic_heldout_rows": heldout_rows,
    }


def _interval_support(row: dict, lower_is_better: bool) -> bool:
    return bool(row.get("status") == "COMPLETE"
                and (row["high"] < 0 if lower_is_better else row["low"] > 0))


def derive_claim_audit(data: dict) -> dict:
    paired = data["paired"]["results"]
    causal = paired["P5_minus_P5_CONSTANT"]
    correction = paired["P5_minus_FULL"]
    c2d = causal["conditional_keypoint_median"]["session"]
    cpck = causal["ALL_GT_PCK"]["10"]
    ct = causal["pose"]["translation_error_cm"]
    cr = causal["pose"]["rotation_error_deg"]
    ccov = causal["pose"]["coverage"]
    p2d = correction["conditional_keypoint_median"]["session"]
    pt = correction["pose"]["translation_error_cm"]
    pr = correction["pose"]["rotation_error_deg"]
    point_joint = ct["delta"] < 0 and cr["delta"] < 0
    interval_joint = _interval_support(ct, True) and _interval_support(cr, True)
    two_d_joint = _interval_support(c2d, True) and _interval_support(cpck, False)
    wood = data["dimension_support"]["objects"]["wood"]
    claims = {
        "incremental_head_dimension_input_2D_on_reused_DEV": {
            "contrast": "P5_minus_P5_CONSTANT",
            "conditional_median_delta_px": c2d,
            "ALL_GT_PCK10_delta": cpck,
            "both_session_intervals_support_benefit": two_d_joint,
            "status": "SUPPORTED_EXPLORATORY" if two_d_joint else "NOT_JOINTLY_SUPPORTED",
        },
        "incremental_head_dimension_input_TR_on_reused_DEV": {
            "contrast": "P5_minus_P5_CONSTANT",
            "translation_delta_cm": ct,
            "rotation_delta_deg": cr,
            "point_estimates_both_improve": point_joint,
            "both_session_intervals_support_benefit": interval_joint,
            "status": ("SUPPORTED_EXPLORATORY" if interval_joint
                       else "DIRECTION_ONLY" if point_joint else "NOT_JOINTLY_SUPPORTED"),
        },
        "incremental_head_dimension_input_pose_coverage": {
            "contrast": "P5_minus_P5_CONSTANT",
            "coverage_delta": ccov,
            "session_interval_supports_increase": _interval_support(ccov, False),
        },
        "P5_correction_effect_vs_frozen_FULL": {
            "contrast": "P5_minus_FULL",
            "conditional_median_delta_px": p2d,
            "translation_delta_cm": pt,
            "rotation_delta_deg": pr,
            "note": "This changes the correction path/capacity and is not a pure dimension-input contrast.",
        },
    }
    broad_joint = bool(interval_joint
                       and data["protocol"]["evaluation"]["independent_TEST"] is True
                       and data["protocol"]["evaluation"]["physical_6D_ground_truth"] is True)
    return {
        "schema": "resnet18_full_dim_refiner_claim_audit_v1",
        "complete": True,
        "primary_incremental_dimension_contrast": "P5_minus_P5_CONSTANT",
        "architecture_plus_dimension_contrast": "P5_minus_P0",
        "all_arms_share_dimension_conditioned_FULL_baseline": True,
        "claims": claims,
        "broad_claim_gates": {
            "stable_joint_TR_generalization": broad_joint,
            "independent_TEST_present": False,
            "physical_pose_GT_present": False,
            "square_refiner_transfer_evaluated": False,
            "latency_measured_in_this_run": False,
            "multiplicity_adjusted": False,
            "wood_inside_source_TRAIN_axiswise_support": not bool(
                wood["outside_train_axiswise_box"]),
        },
        "allowed_wording": [
            "Frozen FULL ResNet18 outputs were evaluated before and after separately trained source-only correction heads on reused DEV319.",
            "P5 versus P5_CONSTANT isolates the incremental five-value head input within this fixed experiment, conditional on the shared dimension-conditioned FULL baseline.",
            "Session bootstrap intervals are exploratory and unadjusted; report their numeric deltas and intervals.",
        ],
        "prohibited_wording": [
            "Stable T and R improvement on unseen test data.",
            "Physically measured 6D accuracy improvement.",
            "Square-pallet correction transfer from this refiner run.",
            "Backbone-agnostic correction proven by this ResNet run alone.",
            "Dimensions caused the direct ResNet square-set result.",
        ],
        "limitations": {
            "real_role": "REUSED_DEV",
            "pose_reference": "geometry-reconstructed reference, not independent physical metrology",
            "statistics": "13-session bootstrap, 10,000 draws, seed 20260914, unadjusted exploratory intervals",
            "square_context": "linked direct full-image ResNet result only; no correction head inference on GREEN150 or GREEN0918",
            "dimension_support": "wood D=0.59 m is outside source TRAIN axiswise support",
        },
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _pct(value, digits: int = 2) -> str:
    return "NA" if value is None else f"{100 * float(value):.{digits}f}"


def _csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue().encode()


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, format="png", dpi=190, bbox_inches="tight",
                metadata={"Software": "pallet-pose refiner report v2"})
    fig.savefig(pdf, format="pdf", bbox_inches="tight",
                metadata={"Creator": "pallet-pose refiner report v2",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)


def _bar_panel(ax, rows: list[dict], field: str, title: str, scale: float = 1.,
               seed_rows: list[dict] | None = None) -> None:
    labels = ["FULL", "D0", "P0", "P5", "P5-C"]
    arms = ["FULL", "D0", "P0", "P5", "P5_CONSTANT"]
    values = [float(rows[index][field]) * scale for index in range(len(rows))]
    bars = ax.bar(np.arange(len(values)), values, color=[COLORS[arm] for arm in arms], width=.72)
    label_offset = max(max(abs(value) for value in values), 1.) * .018
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + label_offset, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7)
    if seed_rows is not None:
        for index, arm in enumerate(arms[1:], start=1):
            points = sorted((row for row in seed_rows
                             if row.get("arm") == arm and row.get("seed") is not None),
                            key=lambda row: row["seed"])
            if len(points) != 3:
                raise ReportContractError(f"Figure requires three seed rows for {arm}/{field}")
            ax.scatter(index + np.asarray([-.10, 0., .10]),
                       [float(row[field]) * scale for row in points],
                       marker="o", s=17, facecolor="white", edgecolor="#17202a",
                       linewidth=.8, zorder=3)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
    ax.set_title(title)
    ax.grid(axis="y", alpha=.2)
    ax.margins(y=.10)


def render_figures(tables: dict, data: dict, directory: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    means = tables["arm_mean_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    _bar_panel(axes[0], means, "median_px", "Conditional corner median (px) ↓",
               seed_rows=tables["method_rows"])
    _bar_panel(axes[1], means, "p90_px", "Conditional corner P90 (px) ↓",
               seed_rows=tables["method_rows"])
    _bar_panel(axes[2], means, "PCK10", "ALL-GT PCK10 (%) ↑", 100.,
               seed_rows=tables["method_rows"])
    fig.suptitle("Frozen FULL ResNet18 and correction heads on reused DEV319\n"
                 "Correction bars are means of three seed statistics, not ensembles")
    _save_figure(fig, directory, "REFINER_2D_OVERVIEW")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    _bar_panel(axes[0], means, "translation_median_cm", "Translation median (cm) ↓",
               seed_rows=tables["method_rows"])
    _bar_panel(axes[1], means, "rotation_median_deg", "Rotation median (deg) ↓",
               seed_rows=tables["method_rows"])
    _bar_panel(axes[2], means, "add_sym_auc_full_population", "C2-aware ADD AUC ↑",
               seed_rows=tables["method_rows"])
    _bar_panel(axes[3], means, "pose_coverage", "Pose coverage (%) ↑", 100.,
               seed_rows=tables["method_rows"])
    fig.suptitle("Geometry-reconstructed pose reference; denominator and failures retained\n"
                 "These are reused-DEV measurements, not physical metrology")
    _save_figure(fig, directory, "REFINER_POSE_OVERVIEW")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row_index, subgroup in enumerate(("plastic", "wood")):
        rows = [row for row in tables["subgroup_rows"]
                if row["subgroup"] == subgroup
                and row["method"] in ("FULL", "D0_SEED_MEAN", "P0_SEED_MEAN",
                                      "P5_SEED_MEAN", "P5_CONSTANT_SEED_MEAN")]
        seeds = [row for row in tables["subgroup_rows"]
                 if row["subgroup"] == subgroup and row.get("seed") is not None]
        _bar_panel(axes[row_index, 0], rows, "median_px",
                   f"{SUBGROUP_LABELS[subgroup]}: 2D median ↓", seed_rows=seeds)
        _bar_panel(axes[row_index, 1], rows, "PCK10",
                   f"{SUBGROUP_LABELS[subgroup]}: ALL-GT PCK10 ↑", 100., seed_rows=seeds)
    fig.suptitle("DEV319 object subgroups; wood dimensions are outside source TRAIN support")
    _save_figure(fig, directory, "REFINER_SUBGROUP_2D")

    paired = data["paired"]["results"]
    labels = ["P5−FULL", "P5−P0", "P5−P5-C"]
    metric_specs = (
        ("2D median Δpx", lambda item: item["conditional_keypoint_median"]["session"], False),
        ("PCK10 Δpp", lambda item: item["ALL_GT_PCK"]["10"], True),
        ("T median Δcm", lambda item: item["pose"]["translation_error_cm"], False),
        ("R median Δdeg", lambda item: item["pose"]["rotation_error_deg"], False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    for ax, (title, getter, percent) in zip(axes, metric_specs):
        stats = [getter(paired[name]) for name in PRIMARY_CONTRASTS]
        scale = 100. if percent else 1.
        center = np.asarray([row["delta"] * scale for row in stats])
        low = np.asarray([row["low"] * scale for row in stats])
        high = np.asarray([row["high"] * scale for row in stats])
        x = np.arange(3)
        # Percentile intervals need not contain the observed point estimate.
        # Draw their endpoints directly instead of passing negative distances
        # to matplotlib's errorbar API.
        ax.vlines(x, low, high, color="#5d6d7e", linewidth=1.5)
        ax.hlines(low, x - .07, x + .07, color="#5d6d7e", linewidth=1.5)
        ax.hlines(high, x - .07, x + .07, color="#5d6d7e", linewidth=1.5)
        ax.scatter(x, center, color="#17202a", zorder=3)
        ax.axhline(0, color="#7f8c8d", lw=1)
        ax.set_xticks(np.arange(3), labels, rotation=25)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Reused DEV319 paired session bootstrap (95% CI, unadjusted exploratory)\n"
                 "Lower is better for 2D/T/R; higher is better for PCK10")
    _save_figure(fig, directory, "REFINER_PAIRED_CONTRASTS")


def _claim_summary_ko(audit: dict) -> str:
    c2d = audit["claims"]["incremental_head_dimension_input_2D_on_reused_DEV"]
    ctr = audit["claims"]["incremental_head_dimension_input_TR_on_reused_DEV"]
    if c2d["status"] == "SUPPORTED_EXPLORATORY" and ctr["status"] == "SUPPORTED_EXPLORATORY":
        return ("P5−P5_CONSTANT에서 2D 중앙값·PCK10과 T·R 중앙값의 세션 구간이 모두 개선 방향이다. "
                "이 결론은 재사용 DEV의 탐색적 근거로만 제한한다.")
    if c2d["status"] == "SUPPORTED_EXPLORATORY":
        return ("P5−P5_CONSTANT의 2D 근거는 개선 방향을 지지하지만 T·R을 함께 안정적으로 개선했다는 "
                "근거는 충족하지 않는다.")
    if ctr["status"] == "SUPPORTED_EXPLORATORY":
        return ("P5−P5_CONSTANT의 T·R 중앙값 구간은 개선 방향을 지지하지만 2D 지표 전체가 함께 "
                "개선됐다고 말할 수 없다.")
    return ("P5−P5_CONSTANT 비교에서 치수 5값의 추가 입력이 2D와 T·R을 함께 안정적으로 개선했다는 "
            "근거는 확보되지 않았다.")


def render_report(data: dict, tables: dict, claim: dict) -> str:
    means = {row["arm"]: row for row in tables["arm_mean_rows"]}
    paired = data["paired"]["results"]
    lines = [
        "# 치수 입력 ResNet18 보정기 결과",
        "",
        _claim_summary_ko(claim),
        "",
        "이 보고서는 동결된 `FULL` ResNet18 추정기 뒤에 붙인 작은 보정 head의 결과다. "
        "합성 TRAIN만으로 학습한 네 arm을 각각 3 seed로 평가했다. 모든 arm은 동일한 FULL baseline, "
        "동일 seed 내 source 행 순서, 동일한 동결 layer2/layer3 특징을 사용한다.",
        "",
        "![DEV319 2D 결과](REFINER_2D_OVERVIEW.png)",
        "",
        "![DEV319 pose 결과](REFINER_POSE_OVERVIEW.png)",
        "",
        "## 비교가 뜻하는 것",
        "",
        "- `D0`: 동결 특징과 기존 점/box에서 직접 residual을 회귀한다.",
        "- `P0`: 치수 벡터 없이 local probability stencil로 보정한다.",
        "- `P5`: `logW`, `logD`, `logH`, `log(W/D)`, `log(H/√WD)`의 정규화 5값을 보정 head에 넣는다.",
        "- `P5_CONSTANT`: P5와 구조·초기값이 같고 5값만 head 내부에서 0으로 고정한다.",
        "",
        "치수 5값 자체의 증분 효과는 `P5−P5_CONSTANT`가 답한다. `P5−P0`는 경로와 용량도 "
        "함께 바뀌는 package 비교다. `P5−FULL`은 보정기 전체의 전후 비교다. FULL baseline 자체가 이미 "
        "치수 조건부로 학습됐으므로 이 실험은 이미지-only 전체 시스템과 치수 조건부 전체 시스템의 비교가 아니다.",
        "",
        "## 실행 계약",
        "",
        f"- 합성 분할: TRAIN 55,980 / calibration 1,004 / selection 1,031 / heldout 1,985. 실사 학습·선택은 0장이다.",
        f"- 각 arm·seed: 6,000 updates × batch 16 = 96,000 nominal exposures. 총 12 fits와 72,000 head updates를 완료했다.",
        f"- 실제 TRAIN usable 행: {data['cache']['partition_counts']['train']['usable']:,} / 55,980.",
        "- calibration temperature와 correction λ/cap은 합성 calibration/selection에서만 정했고, source heldout은 규칙을 고정한 뒤 열었다.",
        "- 실사 평가는 재사용 DEV319장, 13세션, 감독 landmark 2,818점이다. box·score·instance·center·결측 mask는 보존했다.",
        "",
        "[고정 protocol](../pallet_resnet18_dim_refiner_20261002_v1/PROTOCOL.json) · "
        "[학습 완료](../pallet_resnet18_dim_refiner_20261002_v1/TRAINING_COMPLETE.json) · "
        "[합성 선택](../pallet_resnet18_dim_refiner_20261002_v1/SELECTION.json) · "
        "[합성 heldout](../pallet_resnet18_dim_refiner_20261002_v1/SYNTHETIC_HELDOUT.json) · "
        "[DEV 전체 결과](../pallet_resnet18_dim_refiner_20261002_v1/DEV_RESULTS.json)",
        "",
        "## DEV319 전체",
        "",
        "| 방법 | 2D median px↓ | P90 px↓ | ALL-GT PCK10 %↑ | matched | T median cm↓ | R median °↓ | C2-aware ADD AUC full↑ | pose coverage %↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("FULL", *ARMS):
        row = means[arm]
        label = "FULL baseline" if arm == "FULL" else f"{arm} (3-seed 통계 평균)"
        lines.append(
            f"| {label} | {_fmt(row['median_px'])} | {_fmt(row['p90_px'])} | "
            f"{_pct(row['PCK10'])} | {_fmt(row['matched_frames'], 1)}/319 | "
            f"{_fmt(row['translation_median_cm'])} | {_fmt(row['rotation_median_deg'])} | "
            f"{_fmt(row['add_sym_auc_full_population'], 5)} | {_pct(row['pose_coverage'])} |"
        )
    lines += [
        "",
        "보정 행은 세 seed 예측을 합친 앙상블이 아니라 각 seed에서 계산한 통계의 산술평균이다. "
        "2D median/P90은 고정 baseline box IoU≥0.5이며 9점이 모두 유한한 프레임의 감독점에 조건부다. "
        "ALL-GT PCK는 제외점과 실패점을 실패로 남긴다. Pose 오차는 성공한 PnP에 조건부이고 coverage와 "
        "full-population C2-aware corresponding-point ADD AUC가 실패 질량을 보존한다. 이 지표는 unrestricted nearest-neighbor ADD-S가 아니다.",
        "",
        "## Seed별 결과",
        "",
        "| method | 2D median | P90 | PCK10 % | T cm | R ° | pose coverage % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["method_rows"]:
        lines.append(
            f"| {row['method']} | {_fmt(row['median_px'])} | {_fmt(row['p90_px'])} | "
            f"{_pct(row['PCK10'])} | {_fmt(row['translation_median_cm'])} | "
            f"{_fmt(row['rotation_median_deg'])} | {_pct(row['pose_coverage'])} |"
        )
    lines += [
        "",
        "## 사전에 정한 paired 비교",
        "",
        "![paired contrasts](REFINER_PAIRED_CONTRASTS.png)",
        "",
        "아래 값은 앞 방법−뒤 방법이다. 2D/T/R은 음수가 개선이고 PCK10은 양수가 개선이다. "
        "13세션 단위 10,000 bootstrap, seed 20260914의 95% 구간이며 다중비교 보정은 없다.",
        "",
        "| 비교 | Δ2D median px [95% CI] | ΔPCK10 pp [95% CI] | ΔT cm [95% CI] | ΔR ° [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for contrast in PRIMARY_CONTRASTS:
        value = paired[contrast]
        m = value["conditional_keypoint_median"]["session"]
        p = value["ALL_GT_PCK"]["10"]
        t = value["pose"]["translation_error_cm"]
        r = value["pose"]["rotation_error_deg"]
        lines.append(
            f"| {contrast} | {_fmt(m['delta'])} [{_fmt(m['low'])}, {_fmt(m['high'])}] | "
            f"{_fmt(100*p['delta'])} [{_fmt(100*p['low'])}, {_fmt(100*p['high'])}] | "
            f"{_fmt(t['delta'])} [{_fmt(t['low'])}, {_fmt(t['high'])}] | "
            f"{_fmt(r['delta'])} [{_fmt(r['low'])}, {_fmt(r['high'])}] |"
        )

    lines += [
        "",
        "## 직사각형 물체별 결과",
        "",
        "![plastic and wood subgroups](REFINER_SUBGROUP_2D.png)",
        "",
        "| 집단 | 방법 | 2D median px↓ | P90 px↓ | PCK10 %↑ | T cm↓ | R °↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for subgroup in ("plastic", "wood"):
        rows = [row for row in tables["subgroup_rows"]
                if row["subgroup"] == subgroup
                and row["method"] in ("FULL", "D0_SEED_MEAN", "P0_SEED_MEAN",
                                      "P5_SEED_MEAN", "P5_CONSTANT_SEED_MEAN")]
        for row in rows:
            lines.append(
                f"| {SUBGROUP_LABELS[subgroup]} | {row['method']} | {_fmt(row['median_px'])} | "
                f"{_fmt(row['p90_px'])} | {_pct(row['PCK10'])} | "
                f"{_fmt(row['translation_median_cm'])} | {_fmt(row['rotation_median_deg'])} |"
            )
    support = data["dimension_support"]
    lines += [
        "",
        "Plastic 110×130×11 cm는 source TRAIN의 축별 W·D·H 및 정규화 5D 범위 안이다. "
        "Wood 80×59×14 cm는 D=0.59 m가 source TRAIN D 최소값보다 작고 정규화 문맥도 여러 축에서 "
        "범위를 벗어난다. 이 wood 결과는 외삽이며 직사각형 전체 일반화의 근거로 단독 사용하지 않는다.",
        "",
        "| 집단 | canonical W,D,H m | normalized 5D | raw TRAIN 밖 축 | normalized TRAIN 밖 feature |",
        "|---|---|---|---|---|",
    ]
    for subgroup in ("plastic", "wood"):
        obj = support["objects"][subgroup]
        dims = ", ".join(_fmt(v, 3) for v in obj["canonical_WDH_m"])
        context = ", ".join(f"{v:+.3f}" for v in obj["normalized_5D"])
        axes = ", ".join(obj["outside_raw_axes"]) or "없음"
        features = ", ".join(obj["outside_normalized_features"]) or "없음"
        lines.append(f"| {SUBGROUP_LABELS[subgroup]} | {dims} | {context} | {axes} | {features} |")

    lines += [
        "",
        "## 정사각형 자료는 별도 direct-ResNet 문맥",
        "",
        "이번 보정기 실행은 GREEN150 또는 0918 정사각형 세트에서 correction head를 평가하지 않았다. "
        "아래 그림과 링크는 이미지+치수를 처음부터 입력한 full-image direct ResNet18의 별도 결과다. "
        "GREEN150 150장과 0918 119장은 각각 집단 안에서 W·D·H가 1.10×1.10×0.15 m로 고정돼 "
        "있으므로 정사각형 전이는 볼 수 있어도 프레임별 치수 변화의 인과 효과는 식별할 수 없다.",
        "",
        "![별도 direct ResNet18 정사각형 포함 결과](../pallet_resnet18_dimension_20261002_v1/DSNT_REAL_COMPARISON_V2.png)",
        "",
        "[direct ResNet18 상세 보고서](../pallet_resnet18_dimension_20261002_v1/REPORT_KO_V2.md) · "
        "[RGB 사례 gallery](../pallet_resnet18_dimension_20261002_v1/GALLERY.md) · "
        "[gallery 선택 규칙](../pallet_resnet18_dimension_20261002_v1/DSNT_GALLERY_MANIFEST.json)",
        "",
        "Gallery는 결과를 보고 고른 개선/악화 양끝 사례다. 시각 설명용이며 집계 성능이나 보정기 근거가 아니다.",
        "",
        "## 엄격한 한계와 원고 표현",
        "",
        "- 실사 역할은 `REUSED_DEV`다. 독립 TEST 확인이 없고 반복 개발 사용을 일반화 성능으로 표현할 수 없다.",
        "- T/R reference는 2D 주석·camera K·등록 치수에서 재구성했다. 독립 물리 계측 6D GT가 아니다.",
        "- paired 구간은 탐색적이며 다중비교 보정이 없다. 단일 지표의 유리한 결과만 골라 broad claim을 만들지 않는다.",
        "- `P5−P5_CONSTANT`만 이 head에서 치수 5값의 증분 인과 비교다. FULL baseline 자체의 치수 조건은 모든 arm에 공통이다.",
        "- wood는 source dimension support 밖 외삽이고, square 데이터는 이 correction head로 평가하지 않았다.",
        "- 이 실행에는 동일 장치·동일 범위 latency 측정이 없다. 속도·정확도 주장은 만들지 않는다.",
        "- 이 ResNet 결과 하나로 YOLO·DOPE까지 backbone-agnostic이라고 주장하지 않는다. 각 backbone의 별도 결과와 통합 기준이 필요하다.",
        "- 이 보고서는 안정적 T·R 공동 개선, IEEE Sensors 투고 성공, 또는 논문 목표 완료를 자동 선언하지 않는다.",
        "",
        "[claim audit](CLAIM_AUDIT.json) · [전체 집계표](AGGREGATE_TABLES.json) · "
        "[method CSV](DEV_METHOD_TABLE.csv) · [subgroup CSV](DEV_SUBGROUP_TABLE.csv) · "
        "[paired CSV](DEV_PAIRED_TABLE.csv) · [report manifest](REPORT_MANIFEST.json)",
    ]
    return "\n".join(lines) + "\n"


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _input_bindings(data: dict) -> dict:
    bindings = {path.name: _binding(path) for path in required_inputs()}
    bindings["generator"] = _binding(Path(__file__))
    # The short-name map is safe here because all required receipt basenames are unique.
    if len(bindings) != len(required_inputs()) + 1:
        raise ReportContractError("Input binding basename collision")
    return bindings


def render_all(data: dict, temporary: Path) -> dict[str, bytes]:
    tables = build_tables(data)
    claim = derive_claim_audit(data)
    render_figures(tables, data, temporary)
    report = render_report(data, tables, claim)
    files: dict[str, bytes] = {
        "REPORT_KO.md": report.encode(),
        "AGGREGATE_TABLES.json": _json_bytes(tables),
        "CLAIM_AUDIT.json": _json_bytes(claim),
    }
    method_fields = list(tables["method_rows"][0])
    subgroup_fields = list(tables["subgroup_rows"][0])
    paired_fields = list(tables["paired_session_rows"][0])
    files["DEV_METHOD_TABLE.csv"] = _csv_bytes(tables["method_rows"], method_fields)
    files["DEV_SUBGROUP_TABLE.csv"] = _csv_bytes(tables["subgroup_rows"], subgroup_fields)
    files["DEV_PAIRED_TABLE.csv"] = _csv_bytes(tables["paired_session_rows"], paired_fields)
    for path in sorted(temporary.iterdir()):
        if path.suffix in (".png", ".pdf"):
            files[path.name] = path.read_bytes()
    expected_figures = {
        f"{stem}.{extension}"
        for stem in ("REFINER_2D_OVERVIEW", "REFINER_POSE_OVERVIEW",
                     "REFINER_SUBGROUP_2D", "REFINER_PAIRED_CONTRASTS")
        for extension in ("png", "pdf")
    }
    if not expected_figures.issubset(files):
        raise ReportContractError("Figure rendering incomplete")
    output_bindings = {
        name: _binding_for_bytes(DOC / name, value) for name, value in sorted(files.items())
    }
    manifest = {
        "schema": "resnet18_full_dim_refiner_report_manifest_v2",
        "complete": True,
        "source_experiment": C.NAME,
        "report_role": "REUSED_DEV exploratory report; no new inference",
        "protocol_sha256": C.sha256(PROTOCOL),
        "methods": list(METHODS),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "counts": {"real_frames": 319, "sessions": 13, "supervised_landmarks": 2818,
                   "plastic": 194, "wood": 125, "fits": 12,
                   "updates_per_fit": 6000},
        "primary_incremental_dimension_contrast": "P5_minus_P5_CONSTANT",
        "inputs": _input_bindings(data),
        "outputs": output_bindings,
        "claims": {
            "claim_audit": output_bindings["CLAIM_AUDIT.json"],
            "independent_TEST": False,
            "physical_pose_GT": False,
            "square_refiner_transfer": False,
            "latency_measured": False,
            "stable_joint_TR_generalization_claimed": False,
        },
        "direct_resnet_square_context": {
            "separate_from_refiner_evidence": True,
            "report": _binding(DIRECT_REPORT),
            "figure": _binding(DIRECT_FIGURE),
            "gallery": _binding(DIRECT_GALLERY),
            "gallery_is_outcome_selected_nonaggregate": True,
        },
    }
    files["REPORT_MANIFEST.json"] = _json_bytes(manifest)
    return files


def promote_outputs(files: dict[str, bytes], destination: Path = DOC) -> None:
    """Compare everything first, then atomically promote each immutable file."""
    conflicts = [name for name, value in files.items()
                 if (destination / name).exists() and (destination / name).read_bytes() != value]
    if conflicts:
        raise ReportContractError(f"Immutable report output differs: {conflicts}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in sorted(files.items()):
        target = destination / name
        if target.exists():
            continue
        pending = destination / f".{name}.pending"
        pending.write_bytes(value)
        pending.replace(target)


def build() -> dict:
    data = load_and_validate()
    with tempfile.TemporaryDirectory(prefix="pallet-refiner-report-", dir="/tmp") as name:
        files = render_all(data, Path(name))
    promote_outputs(files)
    manifest = _read(DOC / "REPORT_MANIFEST.json")
    # Verify the promoted bytes named by the manifest; the manifest itself is not self-bound.
    for label, binding in manifest["outputs"].items():
        if _binding(DOC / label) != binding:
            raise ReportContractError(f"Promoted output binding mismatch: {label}")
    return manifest


def main() -> None:
    manifest = build()
    print(json.dumps({
        "PASS": True,
        "report": str((DOC / "REPORT_KO.md").relative_to(ROOT)),
        "outputs": len(manifest["outputs"]),
        "primary_contrast": manifest["primary_incremental_dimension_contrast"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
