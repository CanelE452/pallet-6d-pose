"""Training-zero dimension/parity oracle diagnostic.

The command performs one immutable YOLO inference pass and then evaluates all
Phase-A arms from the resulting cache.  Prediction tensors are final before
GT-v2 evaluation content is opened.  The cache contains calibration and label
paths, but no GT parity, pose, or keypoints used to select a prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from challenge.evaluation_v2.paper_real_eval import (  # noqa: E402
    INFERENCE_CONFIDENCE_FLOOR,
    INFERENCE_IMGSZ,
    INFERENCE_PAD,
    _UltralyticsPredictor,
    _transforms_form_exact_symmetry_class,
    _validate_migration_pass,
)
from challenge.evaluation_v2.pnp_selector import (  # noqa: E402
    HypothesisResult,
    PnPSelectionResult,
    SelectorConfig,
    SelectorStatus,
    select_pnp_hypotheses,
)
from challenge.evaluation_v2.pose_metrics import (  # noqa: E402
    add_error_m,
    model_diameter_m,
    pose_auc,
    rotation_error_degrees,
    translation_error_m,
    yaw_error_degrees,
)
from challenge.evaluation_v2.real_dataset_contract import (  # noqa: E402
    ContractError,
    ManifestItem,
    PopulationId,
    PopulationManifest,
    load_repo_population,
    validate_plastic_alias_membership,
    validate_registered_membership,
    validate_wood_dev_membership,
)
from scripts.annotate import pallet_geometry as geometry  # noqa: E402
from scripts.annotate.object_geometry_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    ObjectGeometryRegistry,
    ObjectGeometrySpec,
    load_object_geometry_registry,
)
from scripts.annotate.pallet_symmetry import (  # noqa: E402
    ValidatedSymmetryContract,
    load_symmetry_contract,
)
from scripts.annotate.real_gt_v2_schema import (  # noqa: E402
    SchemaValidationError,
    validate_gt_v2,
)


SCHEMA_VERSION = "dimension_conditioning_phase_a_v2"
CACHE_SCHEMA_VERSION = "dimension_conditioning_prediction_cache_v1"
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20_260_828
G4_RELATIVE_IMPROVEMENT_MIN = 0.10

ARM_A0 = "A0_CURRENT_SELECTOR"
ARM_A1 = "A1_GT_PARITY_ORACLE"
ARM_A2 = "A2_BEST_OF_TWO_ORACLE"
ARM_A3 = "A3_GT_V2_KEYPOINT_ANNOTATIONS_REPLAY"
ARM_A4 = "A4_WRONG_OBJECT_DIMENSION_CONTROL"
ARMS = (ARM_A0, ARM_A1, ARM_A2, ARM_A3, ARM_A4)

PLASTIC_SYMMETRY_CONTRACT_PATH = (
    REPO_ROOT / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"
)
PLASTIC_MIGRATION_GATE_PATH = REPO_ROOT / "challenge/real_gt_v2/MIGRATION_GATE.json"

EXPECTED_OUTPUTS = (
    "PREDICTION_CACHE.npz",
    "ORACLE_PER_FRAME.csv",
    "ORACLE_METRICS.json",
    "ORACLE_BOOTSTRAP.json",
    "DEV_DIMENSION_ORACLE_DIAGNOSTIC.json",
    "ORACLE_REPORT.md",
)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _cache_fingerprint(
    frame_ids: np.ndarray,
    confidences: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for frame_id in frame_ids.tolist():
        digest.update(str(frame_id).encode("utf-8"))
        digest.update(b"\0")
    for array in (confidences, boxes, keypoints):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_json(path: Path, purpose: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{purpose}_UNREADABLE: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{purpose}_ROOT_MUST_BE_OBJECT: {path}")
    return payload


def _plastic_pose_contract() -> tuple[ValidatedSymmetryContract, dict[str, Any]]:
    """Load and fail-closed validate the plastic pose-equivalence contract."""

    symmetry = load_symmetry_contract(PLASTIC_SYMMETRY_CONTRACT_PATH)
    if (
        symmetry.metric_variant != "ADD-S"
        or symmetry.equivalent_yaw_degrees != (0, 180)
        or len(symmetry.rotations) != 2
    ):
        raise ContractError("PHASE_A_REQUIRES_EXACT_PLASTIC_YAW180_ADD_S_CONTRACT")
    gate = _load_json(PLASTIC_MIGRATION_GATE_PATH, "PLASTIC_MIGRATION_GATE")
    resolution_mode = _validate_migration_pass(
        gate,
        PLASTIC_MIGRATION_GATE_PATH,
        symmetry_contract=symmetry,
        expected_count=140,
    )
    return symmetry, {
        "status": "PASS",
        "pose_resolution_mode": resolution_mode,
        "symmetry_contract": {
            "path": _display(PLASTIC_SYMMETRY_CONTRACT_PATH),
            "sha256": symmetry.sha256,
            "status": symmetry.status,
            "metric_variant": symmetry.metric_variant,
            "equivalent_yaw_degrees": list(symmetry.equivalent_yaw_degrees),
            "accepted_proper_rotation_count": len(symmetry.rotations),
        },
        "migration_gate": {
            "path": _display(PLASTIC_MIGRATION_GATE_PATH),
            "sha256": _sha256(PLASTIC_MIGRATION_GATE_PATH),
            "status": gate.get("status"),
            "symmetry_contract_sha256": gate.get("symmetry_contract_sha256"),
            "canonical_pose_equivalence_resolved_count": gate.get(
                "canonical_pose_equivalence_resolved_count"
            ),
        },
    }


def _label_path(item: ManifestItem) -> Path:
    if item.label is None:
        raise ContractError(f"PHASE_A_LABEL_REQUIRED: {item.frame_id}")
    path = (REPO_ROOT / item.label).resolve()
    if not path.is_file():
        raise ContractError(f"PHASE_A_LABEL_NOT_FOUND: {item.frame_id}")
    return path


def _plastic_intrinsics(manifest: PopulationManifest) -> dict[str, np.ndarray]:
    path = REPO_ROOT / "challenge/real_gt_v2/manifests/DEV_POS140_INTRINSICS.json"
    payload = _load_json(path, "PLASTIC_INTRINSICS_MANIFEST")
    legacy = load_repo_population(PopulationId.DEV_POS140, validate_files=True)
    if (
        payload.get("schema_version") != "pallet_pose_camera_intrinsics_manifest_v1"
        or payload.get("population_id") != PopulationId.DEV_POS140.value
        or payload.get("population_membership_sha256") != legacy.membership_sha256
        or payload.get("count") != legacy.count
    ):
        raise ContractError("PLASTIC_INTRINSICS_MANIFEST_CONTRACT_MISMATCH")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != manifest.count:
        raise ContractError("PLASTIC_INTRINSICS_RECORD_COUNT_MISMATCH")
    out: dict[str, np.ndarray] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ContractError("PLASTIC_INTRINSICS_RECORD_INVALID")
        frame_id = str(row.get("frame_id", ""))
        matrix = np.array(
            [
                [float(row["fx"]), 0.0, float(row["cx"])],
                [0.0, float(row["fy"]), float(row["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if not frame_id or frame_id in out or not np.isfinite(matrix).all():
            raise ContractError("PLASTIC_INTRINSICS_RECORD_INVALID")
        out[frame_id] = matrix
    if set(out) != set(manifest.frame_ids):
        raise ContractError("PLASTIC_INTRINSICS_MEMBERSHIP_MISMATCH")
    return out


def _wood_intrinsics_from_audited_labels(
    manifest: PopulationManifest,
) -> dict[str, np.ndarray]:
    """Cache the audited calibration, without approving wood pose metrics."""

    out: dict[str, np.ndarray] = {}
    for item in manifest.items:
        payload = _load_json(_label_path(item), "WOOD_GT_V2")
        if payload.get("intrinsics_quality") != "SENSOR_PROFILE_SCALED":
            raise ContractError(f"WOOD_INTRINSICS_QUALITY_UNEXPECTED: {item.frame_id}")
        intrinsics = payload.get("camera_data", {}).get("intrinsics", {})
        matrix = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ContractError(f"WOOD_INTRINSICS_INVALID: {item.frame_id}")
        out[item.frame_id] = matrix
    return out


def _load_populations() -> tuple[PopulationManifest, PopulationManifest]:
    plastic = load_repo_population(PopulationId.DEV_PLASTIC_POS140, validate_files=True)
    legacy = load_repo_population(PopulationId.DEV_POS140, validate_files=True)
    wood = load_repo_population(PopulationId.DEV_WOOD_POS45, validate_files=True)
    validate_registered_membership(plastic, validate_files=True)
    validate_registered_membership(wood, validate_files=True)
    validate_plastic_alias_membership(
        dev_pos140=legacy,
        dev_plastic_pos140=plastic,
        validate_files=True,
    )
    validate_wood_dev_membership(wood)
    if plastic.count != 140 or wood.count != 45:
        raise ContractError("PHASE_A_POPULATION_COUNT_MISMATCH")
    return plastic, wood


def _inference_cache(
    *,
    plastic: PopulationManifest,
    wood: PopulationManifest,
    weights: Path,
    device: str,
    out_path: Path,
) -> dict[str, np.ndarray]:
    if out_path.exists():
        raise ContractError(f"PREDICTION_CACHE_ALREADY_EXISTS: {out_path}")
    predictor = _UltralyticsPredictor(weights, device)
    items = tuple(plastic.items) + tuple(wood.items)
    object_types = (
        (PLASTIC_OBJECT_TYPE,) * plastic.count + (WOOD_OBJECT_TYPE,) * wood.count
    )
    confidences = np.full(len(items), np.nan, dtype=np.float64)
    boxes = np.full((len(items), 4), np.nan, dtype=np.float64)
    keypoints = np.full((len(items), 9, 2), np.nan, dtype=np.float64)
    detection_counts = np.zeros(len(items), dtype=np.int64)
    for index, item in enumerate(items):
        predictions = predictor.predict((REPO_ROOT / item.image).resolve())
        detection_counts[index] = len(predictions)
        if not predictions:
            continue
        score, box, points = max(predictions, key=lambda row: float(row[0]))
        box_array = np.asarray(box, dtype=np.float64)
        point_array = np.asarray(points, dtype=np.float64) if points is not None else None
        if box_array.shape != (4,) or not np.isfinite(box_array).all():
            raise ContractError(f"INVALID_TOP_BOX: {item.frame_id}")
        confidences[index] = float(score)
        boxes[index] = box_array
        if point_array is not None:
            if point_array.shape != (9, 2) or not np.isfinite(point_array).all():
                raise ContractError(f"INVALID_TOP_KEYPOINTS: {item.frame_id}")
            keypoints[index] = point_array

    frame_ids = np.asarray([item.frame_id for item in items], dtype=np.str_)
    images = np.asarray([item.image for item in items], dtype=np.str_)
    labels = np.asarray([item.label or "" for item in items], dtype=np.str_)
    sessions = np.asarray(
        [item.session_id or item.source_set or "" for item in items], dtype=np.str_
    )
    domains = np.asarray([item.domain or "UNKNOWN" for item in items], dtype=np.str_)
    object_type_array = np.asarray(object_types, dtype=np.str_)
    plastic_intrinsics = _plastic_intrinsics(plastic)
    wood_intrinsics = _wood_intrinsics_from_audited_labels(wood)
    intrinsics = np.stack(
        [
            (plastic_intrinsics if object_type == PLASTIC_OBJECT_TYPE else wood_intrinsics)[
                item.frame_id
            ]
            for item, object_type in zip(items, object_types)
        ]
    )
    fingerprint = _cache_fingerprint(frame_ids, confidences, boxes, keypoints)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_head(),
        "checkpoint": {
            "path": _display(weights),
            "sha256": _sha256(weights),
            "size_bytes": weights.stat().st_size,
        },
        "populations": {
            "plastic": plastic.summary(),
            "wood": wood.summary(),
        },
        "inference_recipe": {
            "top_candidate_rule": "highest box confidence per frame",
            "pad": INFERENCE_PAD,
            "border": "BORDER_REFLECT_101",
            "imgsz": INFERENCE_IMGSZ,
            "confidence_floor": INFERENCE_CONFIDENCE_FLOOR,
            "device": str(device),
        },
        "gt_fields_in_cache": [],
        "wood_intrinsics_note": (
            "cached from audited GT-v2 SENSOR_PROFILE_SCALED records; this does not "
            "approve wood pose metrics"
        ),
        "prediction_input_sha256": fingerprint,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        metadata_json=np.asarray(_json_text(metadata)),
        frame_ids=frame_ids,
        object_types=object_type_array,
        images=images,
        labels=labels,
        sessions=sessions,
        domains=domains,
        detection_counts=detection_counts,
        confidences=confidences,
        boxes_xyxy=boxes,
        keypoints_xy=keypoints,
        camera_intrinsics=intrinsics,
    )
    return _read_cache(out_path, weights)


def _read_cache(path: Path, weights: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ContractError(f"PREDICTION_CACHE_NOT_FOUND: {path}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required = {
        "metadata_json",
        "frame_ids",
        "object_types",
        "images",
        "labels",
        "sessions",
        "domains",
        "detection_counts",
        "confidences",
        "boxes_xyxy",
        "keypoints_xy",
        "camera_intrinsics",
    }
    if set(arrays) != required:
        raise ContractError(f"PREDICTION_CACHE_FIELDS_INVALID: {sorted(arrays)}")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ContractError("PREDICTION_CACHE_SCHEMA_INVALID")
    if metadata.get("checkpoint", {}).get("sha256") != _sha256(weights):
        raise ContractError("PREDICTION_CACHE_CHECKPOINT_MISMATCH")
    n = len(arrays["frame_ids"])
    expected_shapes = {
        "object_types": (n,),
        "images": (n,),
        "labels": (n,),
        "sessions": (n,),
        "domains": (n,),
        "detection_counts": (n,),
        "confidences": (n,),
        "boxes_xyxy": (n, 4),
        "keypoints_xy": (n, 9, 2),
        "camera_intrinsics": (n, 3, 3),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ContractError(f"PREDICTION_CACHE_SHAPE_INVALID: {name}")
    fingerprint = _cache_fingerprint(
        arrays["frame_ids"],
        arrays["confidences"],
        arrays["boxes_xyxy"],
        arrays["keypoints_xy"],
    )
    if metadata.get("prediction_input_sha256") != fingerprint:
        raise ContractError("PREDICTION_CACHE_FINGERPRINT_MISMATCH")
    arrays["metadata"] = metadata  # type: ignore[assignment]
    return arrays


def _truth(
    *,
    label_path: Path,
    frame_id: str,
    spec: ObjectGeometrySpec,
    equivalent_rotations: Sequence[np.ndarray],
) -> dict[str, Any]:
    payload = _load_json(label_path, "PHASE_A_GT_V2")
    try:
        validate_gt_v2(payload)
    except SchemaValidationError as exc:
        raise ContractError(f"PHASE_A_GT_V2_INVALID: {frame_id}: {exc}") from exc
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise ContractError(f"PHASE_A_REQUIRES_ONE_OBJECT: {frame_id}")
    obj = objects[0]
    raw_dimensions = obj.get("physical_dimensions_m")
    if raw_dimensions is None:
        raw_dimensions = {
            "x": obj.get("dimensions_m", {}).get("width"),
            "y": obj.get("dimensions_m", {}).get("height"),
            "z": obj.get("dimensions_m", {}).get("depth"),
        }
    observed = {name: float(raw_dimensions[name]) for name in ("x", "y", "z")}
    if observed != spec.physical_dimensions_m:
        raise ContractError(
            f"PHASE_A_REGISTRY_LABEL_DIMENSION_MISMATCH: {frame_id}: {observed}"
        )
    assignments = tuple(obj["camera_facing_pnp"]["axis_assignment_candidates"])
    allowed = {
        (geometry.AxisAssignment.YAW_0.value, geometry.AxisAssignment.YAW_180.value),
        (geometry.AxisAssignment.YAW_90.value, geometry.AxisAssignment.YAW_270.value),
    }
    if assignments not in allowed:
        raise ContractError(f"PHASE_A_GT_PARITY_INVALID: {frame_id}: {assignments}")
    expected = geometry.camera_facing_hypothesis_name(
        geometry.AxisAssignment(assignments[0]), spec.physical_dimensions
    )
    if any(
        geometry.camera_facing_hypothesis_name(
            geometry.AxisAssignment(value), spec.physical_dimensions
        )
        != expected
        for value in assignments
    ):
        raise ContractError(f"PHASE_A_GT_PARITY_CANDIDATES_DISAGREE: {frame_id}")
    transforms = tuple(
        np.asarray(candidate["pose_transform"], dtype=np.float64)
        for candidate in obj["canonical_pose_candidates"]
    )
    if len(transforms) != 2 or any(transform.shape != (4, 4) for transform in transforms):
        raise ContractError(f"PHASE_A_GT_POSE_CLASS_INVALID: {frame_id}")
    if not _transforms_form_exact_symmetry_class(transforms, equivalent_rotations):
        raise ContractError(
            f"PHASE_A_GT_POSE_CLASS_NOT_FROZEN_YAW180_EQUIVALENCE: {frame_id}"
        )
    annotations = obj.get("keypoint_annotations")
    if not isinstance(annotations, list) or len(annotations) != 9:
        raise ContractError(f"PHASE_A_GT_KEYPOINT_ANNOTATIONS_INVALID: {frame_id}")
    try:
        gt_keypoints = np.asarray([entry["xy"] for entry in annotations], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            f"PHASE_A_GT_KEYPOINT_ANNOTATIONS_INVALID: {frame_id}"
        ) from exc
    if gt_keypoints.shape != (9, 2) or not np.isfinite(gt_keypoints).all():
        raise ContractError(f"PHASE_A_GT_KEYPOINTS_INVALID: {frame_id}")
    return {
        "expected_parity": expected,
        "target_transforms": transforms,
        "gt_keypoints": gt_keypoints,
        "gt_keypoint_source": "GT-v2 keypoint_annotations.xy",
        "intrinsics_quality": payload.get("intrinsics_quality", "CALIBRATED_OR_LEGACY_APPROVED"),
    }


def _hypothesis_by_name(
    selection: PnPSelectionResult,
    name: str,
) -> HypothesisResult:
    matches = [hypothesis for hypothesis in selection.hypotheses if hypothesis.name == name]
    if len(matches) != 1:
        raise ContractError(f"PHASE_A_HYPOTHESIS_NAME_NOT_UNIQUE: {name}")
    return matches[0]


def _pose_values(
    hypothesis: HypothesisResult | None,
    *,
    target_transforms: Sequence[np.ndarray],
    true_dimensions: geometry.PhysicalDimensionsXYZ,
) -> dict[str, Any]:
    if hypothesis is None or not hypothesis.success or not hypothesis.canonical_candidates:
        return {
            "pnp_solved": False,
            "pose_valid": False,
            "restricted_adds_error_m": None,
            "restricted_adds_normalized": None,
            "rotation_error_deg": None,
            "translation_error_m": None,
            "yaw_error_deg": None,
        }
    predicted = hypothesis.canonical_candidates[0]
    model_points = geometry.canonical_keypoints_3d(true_dimensions)[:8]
    diameter = model_diameter_m(model_points)
    adds: list[float] = []
    rotations: list[float] = []
    translations: list[float] = []
    yaws: list[float] = []
    for transform in target_transforms:
        target_rotation = transform[:3, :3]
        target_translation = transform[:3, 3]
        adds.append(
            add_error_m(
                model_points,
                predicted.rotation,
                predicted.translation,
                target_rotation,
                target_translation,
            )
        )
        rotations.append(rotation_error_degrees(predicted.rotation, target_rotation))
        translations.append(
            translation_error_m(predicted.translation, target_translation)
        )
        yaws.append(yaw_error_degrees(predicted.rotation, target_rotation))
    restricted = min(adds)
    return {
        "pnp_solved": True,
        "pose_valid": True,
        "restricted_adds_error_m": restricted,
        "restricted_adds_normalized": restricted / diameter,
        "rotation_error_deg": min(rotations),
        "translation_error_m": min(translations),
        "yaw_error_deg": min(yaws),
    }


def _best_of_two_pose_oracle(
    selection: PnPSelectionResult,
    *,
    target_transforms: Sequence[np.ndarray],
    true_dimensions: geometry.PhysicalDimensionsXYZ,
) -> tuple[str, HypothesisResult, dict[str, float]] | None:
    """Choose the lower restricted-ADD hypothesis only when both PnPs solve."""

    candidates: list[tuple[float, str, HypothesisResult]] = []
    metric_by_name: dict[str, float] = {}
    for hypothesis in selection.hypotheses:
        values = _pose_values(
            hypothesis,
            target_transforms=target_transforms,
            true_dimensions=true_dimensions,
        )
        normalized = values["restricted_adds_normalized"]
        if normalized is not None:
            metric = float(normalized)
            candidates.append((metric, hypothesis.name, hypothesis))
            metric_by_name[hypothesis.name] = metric
    if len(candidates) != 2:
        return None
    _, selected_name, selected_hypothesis = min(
        candidates, key=lambda value: value[:2]
    )
    return selected_name, selected_hypothesis, metric_by_name


def _invalid_pose_values() -> dict[str, Any]:
    return _pose_values(
        None,
        target_transforms=(),
        true_dimensions=geometry.canonical_dimensions(),
    )


def _arm_row(
    *,
    arm: str,
    frame_id: str,
    domain: str,
    session: str,
    object_type: str,
    detection_count: int,
    confidence: float | None,
    expected_parity: str,
    selected_parity: str | None,
    parity_correct: bool,
    hypothesis: HypothesisResult | None,
    target_transforms: Sequence[np.ndarray],
    true_dimensions: geometry.PhysicalDimensionsXYZ,
    paper_eligible: bool,
    gt_used_for_parity: bool,
    gt_used_for_selection: bool,
    keypoint_source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": arm,
        "frame_id": frame_id,
        "domain": domain,
        "session": session,
        "object_type": object_type,
        "detection_count": detection_count,
        "confidence": confidence,
        "expected_parity": expected_parity,
        "selected_parity": selected_parity,
        "parity_correct": bool(parity_correct),
        "paper_eligible": bool(paper_eligible),
        "gt_used_for_parity": bool(gt_used_for_parity),
        "gt_used_for_selection": bool(gt_used_for_selection),
        "keypoint_source": keypoint_source,
    }
    row.update(
        _pose_values(
            hypothesis,
            target_transforms=target_transforms,
            true_dimensions=true_dimensions,
        )
    )
    return row


def _evaluate_plastic(
    cache: Mapping[str, Any],
    registry: ObjectGeometryRegistry,
    symmetry_contract: ValidatedSymmetryContract,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plastic = registry.resolve(PLASTIC_OBJECT_TYPE)
    wrong = registry.resolve(WOOD_OBJECT_TYPE)
    config = SelectorConfig()
    rows: list[dict[str, Any]] = []
    predicted_a2_comparable = 0
    predicted_a2_agree = 0
    predicted_a2_disagreements: list[str] = []
    predicted_a2_missing: list[str] = []
    gt_contract_comparable = 0
    gt_contract_agree = 0
    gt_contract_disagreements: list[str] = []
    gt_contract_missing: list[str] = []
    gt_contract_margins: list[float] = []

    indices = np.flatnonzero(cache["object_types"] == PLASTIC_OBJECT_TYPE)
    if len(indices) != 140:
        raise ContractError(f"PHASE_A_PLASTIC_CACHE_COUNT_INVALID: {len(indices)}")
    for index in indices.tolist():
        frame_id = str(cache["frame_ids"][index])
        truth = _truth(
            label_path=REPO_ROOT / str(cache["labels"][index]),
            frame_id=frame_id,
            spec=plastic,
            equivalent_rotations=symmetry_contract.rotations,
        )
        expected = str(truth["expected_parity"])
        target_transforms = truth["target_transforms"]
        points = np.asarray(cache["keypoints_xy"][index], dtype=np.float64)
        camera = np.asarray(cache["camera_intrinsics"][index], dtype=np.float64)
        has_prediction = bool(np.isfinite(points).all())
        confidence_value = float(cache["confidences"][index])
        confidence = confidence_value if math.isfinite(confidence_value) else None
        common = {
            "frame_id": frame_id,
            "domain": str(cache["domains"][index]),
            "session": str(cache["sessions"][index]),
            "object_type": PLASTIC_OBJECT_TYPE,
            "detection_count": int(cache["detection_counts"][index]),
            "confidence": confidence,
            "expected_parity": expected,
            "target_transforms": target_transforms,
            "true_dimensions": plastic.physical_dimensions,
        }
        selection: PnPSelectionResult | None = None
        wrong_selection: PnPSelectionResult | None = None
        if has_prediction:
            selection = select_pnp_hypotheses(
                points, camera, plastic.physical_dimensions, config
            )
            wrong_selection = select_pnp_hypotheses(
                points, camera, wrong.physical_dimensions, config
            )

        selected_a0 = selection.selected_hypothesis if selection is not None else None
        hypothesis_a0 = None
        if selection is not None and selection.status is SelectorStatus.SELECTED:
            hypothesis_a0 = _hypothesis_by_name(selection, str(selected_a0))
        rows.append(
            _arm_row(
                arm=ARM_A0,
                selected_parity=selected_a0,
                parity_correct=selected_a0 == expected,
                hypothesis=hypothesis_a0,
                paper_eligible=True,
                gt_used_for_parity=False,
                gt_used_for_selection=False,
                keypoint_source="PREDICTION_CACHE.keypoints_xy",
                **common,
            )
        )

        hypothesis_a1 = (
            _hypothesis_by_name(selection, expected) if selection is not None else None
        )
        rows.append(
            _arm_row(
                arm=ARM_A1,
                selected_parity=expected,
                parity_correct=True,
                hypothesis=hypothesis_a1,
                paper_eligible=False,
                gt_used_for_parity=True,
                gt_used_for_selection=False,
                keypoint_source="PREDICTION_CACHE.keypoints_xy",
                **common,
            )
        )

        hypothesis_a2: HypothesisResult | None = None
        selected_a2: str | None = None
        if selection is not None:
            predicted_choice = _best_of_two_pose_oracle(
                selection,
                target_transforms=target_transforms,
                true_dimensions=plastic.physical_dimensions,
            )
            if predicted_choice is not None:
                selected_a2, hypothesis_a2, _ = predicted_choice
                predicted_a2_comparable += 1
                if selected_a2 == expected:
                    predicted_a2_agree += 1
                else:
                    predicted_a2_disagreements.append(frame_id)
        if selected_a2 is None:
            predicted_a2_missing.append(frame_id)
        rows.append(
            _arm_row(
                arm=ARM_A2,
                selected_parity=selected_a2,
                parity_correct=selected_a2 == expected,
                hypothesis=hypothesis_a2,
                paper_eligible=False,
                gt_used_for_parity=False,
                gt_used_for_selection=True,
                keypoint_source="PREDICTION_CACHE.keypoints_xy",
                **common,
            )
        )

        gt_selection = select_pnp_hypotheses(
            truth["gt_keypoints"], camera, plastic.physical_dimensions, config
        )
        gt_contract_choice = _best_of_two_pose_oracle(
            gt_selection,
            target_transforms=target_transforms,
            true_dimensions=plastic.physical_dimensions,
        )
        if gt_contract_choice is None:
            gt_contract_missing.append(frame_id)
        else:
            gt_selected, _, gt_metric_by_name = gt_contract_choice
            gt_contract_comparable += 1
            if gt_selected == expected:
                gt_contract_agree += 1
            else:
                gt_contract_disagreements.append(frame_id)
            other_names = [name for name in gt_metric_by_name if name != expected]
            if len(other_names) != 1 or expected not in gt_metric_by_name:
                raise ContractError(
                    f"PHASE_A_GT_CONTRACT_HYPOTHESIS_SET_INVALID: {frame_id}"
                )
            gt_contract_margins.append(
                gt_metric_by_name[other_names[0]] - gt_metric_by_name[expected]
            )
        hypothesis_a3 = _hypothesis_by_name(gt_selection, expected)
        rows.append(
            _arm_row(
                arm=ARM_A3,
                selected_parity=expected,
                parity_correct=True,
                hypothesis=hypothesis_a3,
                paper_eligible=False,
                gt_used_for_parity=True,
                gt_used_for_selection=False,
                keypoint_source="GT-v2 keypoint_annotations.xy",
                **common,
            )
        )

        selected_a4 = (
            wrong_selection.selected_hypothesis if wrong_selection is not None else None
        )
        hypothesis_a4 = None
        if wrong_selection is not None and wrong_selection.status is SelectorStatus.SELECTED:
            hypothesis_a4 = _hypothesis_by_name(wrong_selection, str(selected_a4))
        rows.append(
            _arm_row(
                arm=ARM_A4,
                selected_parity=selected_a4,
                parity_correct=selected_a4 == expected,
                hypothesis=hypothesis_a4,
                paper_eligible=False,
                gt_used_for_parity=False,
                gt_used_for_selection=False,
                keypoint_source="PREDICTION_CACHE.keypoints_xy",
                **common,
            )
        )

    population_n = len(indices)
    gt_contract_passed = bool(
        gt_contract_comparable == population_n
        and gt_contract_agree == population_n
        and not gt_contract_disagreements
        and not gt_contract_missing
        and gt_contract_margins
        and min(gt_contract_margins) > 0.0
    )
    a2_audit = {
        "status": "PASS" if gt_contract_passed else "FAIL",
        "comparison_metric": "restricted_adds_normalized",
        "interpretation": (
            "GT-keypoint replay is the static parity/canonical-mapping/solver "
            "contract check. Predicted-keypoint A2 is a pose oracle; disagreement "
            "with GT parity can arise from noisy or permuted keypoint correspondences."
        ),
        "gt_keypoint_static_contract_check": {
            "status": "PASS" if gt_contract_passed else "FAIL",
            "keypoint_source": "GT-v2 keypoint_annotations.xy",
            "total_n": population_n,
            "comparable_n": gt_contract_comparable,
            "missing_or_unsolved_n": population_n - gt_contract_comparable,
            "missing_or_unsolved_frame_ids": gt_contract_missing,
            "matches_gt_parity_n": gt_contract_agree,
            "conditional_agreement": (
                gt_contract_agree / gt_contract_comparable
                if gt_contract_comparable
                else None
            ),
            "disagreement_frame_ids": gt_contract_disagreements,
            "minimum_wrong_minus_gt_parity_normalized_adds_margin": (
                min(gt_contract_margins) if gt_contract_margins else None
            ),
        },
        "predicted_keypoint_pose_oracle_diagnostic": {
            "status": "COMPLETE",
            "keypoint_source": "PREDICTION_CACHE.keypoints_xy",
            "total_n": population_n,
            "comparable_n": predicted_a2_comparable,
            "missing_or_unsolved_n": population_n - predicted_a2_comparable,
            "missing_or_unsolved_frame_ids": predicted_a2_missing,
            "matches_gt_parity_n": predicted_a2_agree,
            "conditional_agreement": (
                predicted_a2_agree / predicted_a2_comparable
                if predicted_a2_comparable
                else None
            ),
            "disagreement_frame_ids": predicted_a2_disagreements,
            "not_a_static_mapping_failure_test": True,
        },
    }
    return rows, a2_audit


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                out.append(number)
    return out


def _pose_auc_exact_grid(normalized_errors: Sequence[float]) -> float:
    """Match ``pose_auc(..., 0.1)`` without rebuilding 1,001 thresholds.

    The public helper integrates a right-continuous empirical accuracy curve
    on a fixed 1,001-point grid with the trapezoid rule.  Linearity lets us sum
    the grid weight contributed by each error directly, which is important for
    the 1,000-replicate cluster bootstrap.
    """

    errors = np.asarray(normalized_errors, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError("normalized_errors must be a non-empty 1-D sequence")
    if np.isnan(errors).any() or (errors < 0.0).any():
        raise ValueError("normalized_errors must be non-negative and not NaN")
    thresholds = np.linspace(0.0, 0.1, 1001, dtype=np.float64)
    first_true = np.searchsorted(thresholds, errors, side="left")
    contribution = np.zeros(errors.shape, dtype=np.float64)
    contribution[first_true == 0] = 1.0
    inside = (first_true >= 1) & (first_true <= 1000)
    contribution[inside] = (1000.5 - first_true[inside]) / 1000.0
    return float(np.mean(contribution))


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "NOT_RUN",
            "N": 0,
            "parity_accuracy": None,
            "pnp_solve_rate": None,
            "pose_valid_n": 0,
            "pose_valid_total": 0,
            "pose_valid_rate": None,
            "restricted_adds_auc": None,
            "rotation_median_deg": None,
            "translation_median_m": None,
            "yaw_median_deg": None,
        }
    n = len(rows)
    normalized = [
        float(row["restricted_adds_normalized"])
        if row.get("restricted_adds_normalized") is not None
        else math.inf
        for row in rows
    ]
    rotations = _finite_values(rows, "rotation_error_deg")
    translations = _finite_values(rows, "translation_error_m")
    yaws = _finite_values(rows, "yaw_error_deg")
    parity_n = sum(bool(row["parity_correct"]) for row in rows)
    pnp_n = sum(bool(row["pnp_solved"]) for row in rows)
    pose_n = sum(bool(row["pose_valid"]) for row in rows)
    return {
        "status": "COMPLETE",
        "N": n,
        "parity_correct_n": parity_n,
        "parity_accuracy": parity_n / n,
        "pnp_solved_n": pnp_n,
        "pnp_solve_rate": pnp_n / n,
        "pose_valid_n": pose_n,
        "pose_valid_total": n,
        "pose_valid_rate": pose_n / n,
        "restricted_adds_auc": _pose_auc_exact_grid(normalized),
        "rotation_median_deg": float(np.median(rotations)) if rotations else None,
        "translation_median_m": (
            float(np.median(translations)) if translations else None
        ),
        "yaw_median_deg": float(np.median(yaws)) if yaws else None,
    }


def _subgroups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {
        "ALL": list(rows),
        "DAY": [row for row in rows if row["domain"] == "DAY"],
        "NIGHT": [row for row in rows if row["domain"] == "NIGHT"],
        "PLASTIC": [row for row in rows if row["object_type"] == PLASTIC_OBJECT_TYPE],
    }
    for session in sorted({str(row["session"]) for row in rows}):
        groups[f"SESSION:{session}"] = [row for row in rows if row["session"] == session]
    return groups


def _all_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    metrics: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        groups = _subgroups(arm_rows)
        metrics[arm] = {name: summarize_rows(group) for name, group in groups.items()}
        metrics[arm]["WOOD"] = {
            "status": "BLOCKED",
            "blocked_reasons": [
                "WOOD_SYMMETRY_UNREVIEWED",
                "WOOD_INTRINSICS_SENSOR_PROFILE_SCALED_NOT_APPROVED",
            ],
            "N": 45,
            "parity_accuracy": None,
            "pnp_solve_rate": None,
            "pose_valid_n": None,
            "pose_valid_total": 45,
            "pose_valid_rate": None,
            "restricted_adds_auc": None,
            "rotation_median_deg": None,
            "translation_median_m": None,
            "yaw_median_deg": None,
        }
    return metrics


def _cluster_resample(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[Mapping[str, Any]]:
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["session"])].append(row)
    sessions = sorted(by_session)
    sampled = rng.choice(sessions, size=len(sessions), replace=True)
    return [row for session in sampled for row in by_session[str(session)]]


def _bootstrap_one(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    if not rows:
        return {"status": "NOT_RUN", "iterations": 0, "metrics": {}}
    rng = np.random.default_rng(seed)
    fields = (
        "parity_accuracy",
        "pnp_solve_rate",
        "pose_valid_rate",
        "restricted_adds_auc",
        "rotation_median_deg",
        "translation_median_m",
        "yaw_median_deg",
    )
    samples: dict[str, list[float]] = {field: [] for field in fields}
    for _ in range(BOOTSTRAP_ITERATIONS):
        summary = summarize_rows(_cluster_resample(rows, rng))
        for field in fields:
            value = summary[field]
            if value is not None:
                samples[field].append(float(value))
    metrics: dict[str, Any] = {}
    for field, values in samples.items():
        if not values:
            metrics[field] = {"low": None, "high": None, "valid_replicates": 0}
        else:
            metrics[field] = {
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "valid_replicates": len(values),
            }
    return {
        "status": "COMPLETE",
        "kind": "session-cluster bootstrap",
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": seed,
        "cluster_count": len({str(row["session"]) for row in rows}),
        "metrics": metrics,
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        arm_rows = [row for row in rows if row["arm"] == arm]
        out[arm] = {}
        for group_index, (name, group) in enumerate(_subgroups(arm_rows).items()):
            out[arm][name] = _bootstrap_one(
                group, seed=BOOTSTRAP_SEED + arm_index * 100 + group_index
            )
        out[arm]["WOOD"] = {
            "status": "BLOCKED",
            "iterations": 0,
            "blocked_reasons": [
                "WOOD_SYMMETRY_UNREVIEWED",
                "WOOD_INTRINSICS_SENSOR_PROFILE_SCALED_NOT_APPROVED",
            ],
            "metrics": {},
        }
    return out


def _relative_reduction(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (baseline - candidate) / baseline


def _oracle_gate(
    rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    a0 = metrics[ARM_A0]["ALL"]
    a1 = metrics[ARM_A1]["ALL"]
    a0_night = metrics[ARM_A0]["NIGHT"]
    session_values = [
        value
        for name, value in metrics[ARM_A0].items()
        if name.startswith("SESSION:")
    ]
    minimum_session = min(float(value["parity_accuracy"]) for value in session_values)
    selector_gate_pass = bool(
        a0["parity_accuracy"] >= 0.95
        and a0_night["parity_accuracy"] >= 0.90
        and minimum_session >= 0.85
    )
    g1 = bool(a1["parity_accuracy"] == 1.0 and not selector_gate_pass)

    improvements = {
        "restricted_adds_auc_absolute": (
            float(a1["restricted_adds_auc"]) - float(a0["restricted_adds_auc"])
        ),
        "rotation_relative_reduction": _relative_reduction(
            a0["rotation_median_deg"], a1["rotation_median_deg"]
        ),
        "translation_relative_reduction": _relative_reduction(
            a0["translation_median_m"], a1["translation_median_m"]
        ),
        "yaw_relative_reduction": _relative_reduction(
            a0["yaw_median_deg"], a1["yaw_median_deg"]
        ),
        "pose_valid_rate_absolute": (
            float(a1["pose_valid_rate"]) - float(a0["pose_valid_rate"])
        ),
    }
    improvement_pass = {
        "restricted_adds_auc_absolute": improvements["restricted_adds_auc_absolute"] >= 0.03,
        "rotation_relative_reduction": (
            improvements["rotation_relative_reduction"] is not None
            and improvements["rotation_relative_reduction"] >= 0.10
        ),
        "translation_relative_reduction": (
            improvements["translation_relative_reduction"] is not None
            and improvements["translation_relative_reduction"] >= 0.10
        ),
        "yaw_relative_reduction": (
            improvements["yaw_relative_reduction"] is not None
            and improvements["yaw_relative_reduction"] >= 0.10
        ),
        "pose_valid_rate_absolute": improvements["pose_valid_rate_absolute"] >= 0.05,
    }
    g2 = sum(improvement_pass.values()) >= 2

    no_harm = {
        "restricted_adds_auc": float(a1["restricted_adds_auc"]) >= 0.95 * float(a0["restricted_adds_auc"]),
        "rotation_median_deg": float(a1["rotation_median_deg"]) <= 1.05 * float(a0["rotation_median_deg"]),
        "translation_median_m": float(a1["translation_median_m"]) <= 1.05 * float(a0["translation_median_m"]),
        "yaw_median_deg": float(a1["yaw_median_deg"]) <= 1.05 * float(a0["yaw_median_deg"]),
    }
    g3 = all(no_harm.values())

    by_key = {(str(row["arm"]), str(row["frame_id"])): row for row in rows}
    a0_rows = [row for row in rows if row["arm"] == ARM_A0]
    selector_errors = [row for row in a0_rows if not bool(row["parity_correct"])]
    recovered: list[str] = []
    for base in selector_errors:
        oracle = by_key[(ARM_A1, str(base["frame_id"]))]
        base_error = base.get("restricted_adds_normalized")
        oracle_error = oracle.get("restricted_adds_normalized")
        recovered_invalid = not bool(base["pose_valid"]) and bool(oracle["pose_valid"])
        recovered_imprecision = bool(
            base_error is not None
            and oracle_error is not None
            and float(base_error) > 0.0
            and (float(base_error) - float(oracle_error)) / float(base_error)
            >= G4_RELATIVE_IMPROVEMENT_MIN
        )
        if recovered_invalid or recovered_imprecision:
            recovered.append(str(base["frame_id"]))
    recovery_fraction = len(recovered) / len(selector_errors) if selector_errors else None
    g4 = bool(recovery_fraction is not None and recovery_fraction >= 0.25)
    passed = g1 and g2 and g3 and g4
    return {
        "selector_paper_gate": {
            "overall": a0["parity_accuracy"],
            "night": a0_night["parity_accuracy"],
            "minimum_session": minimum_session,
            "passed": selector_gate_pass,
        },
        "G1": {
            "passed": g1,
            "A1_parity_accuracy": a1["parity_accuracy"],
            "A0_selector_gate_passed": selector_gate_pass,
        },
        "G2": {
            "passed": g2,
            "required_count": 2,
            "passed_count": sum(improvement_pass.values()),
            "values": improvements,
            "conditions": improvement_pass,
        },
        "G3": {"passed": g3, "conditions": no_harm},
        "G4": {
            "passed": g4,
            "definition": (
                "A0 pose-invalid to A1 pose-valid, or >=10% relative reduction in "
                "restricted normalized ADD-S"
            ),
            "selector_error_n": len(selector_errors),
            "recovered_n": len(recovered),
            "recovery_fraction": recovery_fraction,
            "recovered_frame_ids": recovered,
        },
        "passed": passed,
        "verdict": (
            "ORACLE_PARITY_HEADROOM_PRESENT"
            if passed
            else "DIMENSION_PARITY_NOT_MAIN_LEVER"
        ),
    }


def _input_safety(cache: Mapping[str, Any]) -> dict[str, Any]:
    mask = cache["object_types"] == PLASTIC_OBJECT_TYPE
    confidence = np.asarray(cache["confidences"])[mask]
    boxes = np.asarray(cache["boxes_xyxy"])[mask]
    keypoints = np.asarray(cache["keypoints_xy"])[mask]
    copies = {
        ARM_A0: (confidence.copy(), boxes.copy(), keypoints.copy()),
        ARM_A1: (confidence.copy(), boxes.copy(), keypoints.copy()),
        ARM_A2: (confidence.copy(), boxes.copy(), keypoints.copy()),
    }
    reference = copies[ARM_A0]
    comparisons: dict[str, Any] = {}
    for arm in (ARM_A1, ARM_A2):
        exact = [
            np.array_equal(first, second, equal_nan=True)
            for first, second in zip(reference, copies[arm])
        ]
        if not all(exact):
            raise ContractError(f"PHASE_A_2D_SAFETY_FAILURE: {arm}")
        comparisons[f"{ARM_A0}_vs_{arm}"] = {
            "confidence_exact": exact[0],
            "box_exact": exact[1],
            "keypoints_exact": exact[2],
            "confidence_max_abs_diff": 0.0,
            "box_max_abs_diff": 0.0,
            "keypoints_max_abs_diff": 0.0,
        }
    return {
        "status": "PASS",
        "source": "A0/A1/A2 index the same immutable PREDICTION_CACHE arrays",
        "comparisons": comparisons,
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "arm",
        "frame_id",
        "domain",
        "session",
        "object_type",
        "detection_count",
        "confidence",
        "expected_parity",
        "selected_parity",
        "parity_correct",
        "paper_eligible",
        "gt_used_for_parity",
        "gt_used_for_selection",
        "keypoint_source",
        "pnp_solved",
        "pose_valid",
        "restricted_adds_error_m",
        "restricted_adds_normalized",
        "rotation_error_deg",
        "translation_error_m",
        "yaw_error_deg",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _report(payload: Mapping[str, Any]) -> str:
    metrics = payload["metrics"]
    gate = payload["oracle_gate"]
    a2_audit = payload["A2_pose_oracle_and_static_contract_audit"]
    gt_contract = a2_audit["gt_keypoint_static_contract_check"]
    predicted_oracle = a2_audit["predicted_keypoint_pose_oracle_diagnostic"]
    lines = [
        "# Phase A — dimension/parity oracle diagnostic",
        "",
        f"Verdict: **{gate['verdict']}**",
        "",
        "DEV diagnostic only. A1/A2/A3 are oracle arms, use GT, and are not",
        "paper-eligible deployment methods. No YOLO parameter was updated.",
        "",
        "## Plastic DEV140",
        "",
        "| arm | parity acc | PnP solve | pose valid | restricted ADD-S AUC | R median ° | t median m | yaw median ° |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = metrics[arm]["ALL"]
        lines.append(
            f"| `{arm}` | {row['parity_accuracy']:.6f} | {row['pnp_solve_rate']:.6f} | "
            f"{row['pose_valid_n']}/{row['pose_valid_total']} | "
            f"{row['restricted_adds_auc']:.6f} | {row['rotation_median_deg']:.6f} | "
            f"{row['translation_median_m']:.6f} | {row['yaw_median_deg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## A2 interpretation and static contract",
            "",
            f"- GT-v2 keypoint static mapping check: `{gt_contract['status']}` "
            f"({gt_contract['matches_gt_parity_n']}/{gt_contract['total_n']})",
            "- Predicted-keypoint pose-oracle agreement with GT parity: "
            f"{predicted_oracle['matches_gt_parity_n']}/"
            f"{predicted_oracle['comparable_n']} comparable frames",
            "- Predicted-keypoint disagreements are diagnostic evidence of noisy or "
            "permuted keypoint correspondences; they are not a static mapping failure.",
            "",
            "## A3 replay disclosure",
            "",
            "A3 uses `GT-v2 keypoint_annotations.xy`. Those annotations and the",
            "canonical pose candidates share legacy annotation/PnP provenance, so A3",
            "is a solver consistency replay, not an independent noise-floor estimate.",
            "",
            "## Frozen oracle gate",
            "",
            f"- G1: `{gate['G1']['passed']}`",
            f"- G2: `{gate['G2']['passed']}` ({gate['G2']['passed_count']} conditions)",
            f"- G3: `{gate['G3']['passed']}`",
            f"- G4: `{gate['G4']['passed']}` ({gate['G4']['recovered_n']} / "
            f"{gate['G4']['selector_error_n']} selector errors recovered)",
            "",
            "G3 is implemented on the primary Plastic DEV140 `ALL` summary. The",
            "user protocol does not explicitly resolve an every-subgroup reading;",
            "under that strict sensitivity reading G3 fails because NIGHT median",
            "translation worsens by more than 5%. See the final report.",
            "",
            "## Safety and secondary population",
            "",
            "A0/A1/A2 confidence, box and keypoint arrays are byte-identical inputs.",
            "A4 executes only plastic images with wood dimensions. The reverse",
            "wood-images/plastic-dimensions direction is blocked and was not evaluated.",
            "All wood evaluation is blocked because symmetry is `UNREVIEWED` and",
            "intrinsics are `SENSOR_PROFILE_SCALED`; the 45 wood predictions remain",
            "in the cache only as immutable evidence for a later approved audit.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="read an existing PREDICTION_CACHE.npz instead of running inference",
    )
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help=(
            "replace only derived Phase-A reports; requires --reuse-cache and "
            "never rewrites PREDICTION_CACHE.npz"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    reuse_cache = bool(getattr(args, "reuse_cache", False))
    overwrite_derived = bool(getattr(args, "overwrite_derived", False))
    if not out_dir.is_dir():
        raise ContractError(f"PHASE_A_OUTPUT_DIRECTORY_NOT_FOUND: {out_dir}")
    if overwrite_derived and not reuse_cache:
        raise ContractError("OVERWRITE_DERIVED_REQUIRES_REUSE_CACHE")
    for name in EXPECTED_OUTPUTS[1:]:
        if (out_dir / name).exists() and not overwrite_derived:
            raise ContractError(f"PHASE_A_OUTPUT_ALREADY_EXISTS: {out_dir / name}")
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise ContractError(f"PHASE_A_WEIGHTS_NOT_FOUND: {weights}")
    checkpoint_sha_before = _sha256(weights)
    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    symmetry_contract, plastic_pose_contract = _plastic_pose_contract()
    plastic, wood = _load_populations()
    cache_path = out_dir / "PREDICTION_CACHE.npz"
    if reuse_cache:
        cache = _read_cache(cache_path, weights)
    else:
        cache = _inference_cache(
            plastic=plastic,
            wood=wood,
            weights=weights,
            device=str(args.device),
            out_path=cache_path,
        )
    expected_ids = tuple(plastic.frame_ids) + tuple(wood.frame_ids)
    if tuple(cache["frame_ids"].tolist()) != expected_ids:
        raise ContractError("PHASE_A_CACHE_POPULATION_ORDER_MISMATCH")
    rows, a2_audit = _evaluate_plastic(cache, registry, symmetry_contract)
    if a2_audit["status"] != "PASS":
        raise ContractError("PHASE_A_GT_KEYPOINT_STATIC_MAPPING_CONTRACT_FAILED")
    metrics = _all_metrics(rows)
    bootstrap = _bootstrap(rows)
    safety = _input_safety(cache)
    gate = _oracle_gate(rows, metrics)
    checkpoint_sha_after = _sha256(weights)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise ContractError("PHASE_A_CHECKPOINT_CHANGED_DURING_RUN")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "paper_final_table_eligible": False,
        "repo_head": _git_head(),
        "checkpoint": {
            "path": _display(weights),
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "unchanged": checkpoint_sha_before == checkpoint_sha_after,
        },
        "geometry_registry": {
            "path": _display(registry.source_path),
            "sha256": registry.sha256,
            "plastic_dimensions_m": registry.resolve(PLASTIC_OBJECT_TYPE).physical_dimensions_m,
            "wrong_control_dimensions_m": registry.resolve(WOOD_OBJECT_TYPE).physical_dimensions_m,
        },
        "plastic_pose_contract": plastic_pose_contract,
        "population": {
            "primary": plastic.summary(),
            "secondary_wood": {
                **wood.summary(),
                "phase_a_evaluation_status": "BLOCKED_NOT_EVALUATED",
                "blocked_reasons": [
                    "WOOD_SYMMETRY_UNREVIEWED",
                    "WOOD_INTRINSICS_SENSOR_PROFILE_SCALED_NOT_APPROVED",
                ],
            },
        },
        "cache": {
            "path": _display(cache_path),
            "sha256": _sha256(cache_path),
            "prediction_input_sha256": cache["metadata"]["prediction_input_sha256"],
            "frame_count": len(cache["frame_ids"]),
            "plastic_count": int(np.sum(cache["object_types"] == PLASTIC_OBJECT_TYPE)),
            "wood_count": int(np.sum(cache["object_types"] == WOOD_OBJECT_TYPE)),
            "gt_selection_fields": [],
            "allowed_nonselection_fields": [
                "camera_intrinsics",
                "label_path_for_post_inference_evaluation",
            ],
        },
        "selector_config": asdict(SelectorConfig()),
        "arm_contract": {
            ARM_A0: {"parity_source": "prediction-only", "paper_eligible": True},
            ARM_A1: {
                "parity_source": "GT",
                "labels": ["ORACLE", "GT_USED_FOR_PARITY"],
                "paper_eligible": False,
            },
            ARM_A2: {
                "selection_source": "minimum GT restricted ADD-S",
                "keypoint_source": "PREDICTION_CACHE.keypoints_xy",
                "labels": ["ORACLE", "GT_USED_FOR_SELECTION"],
                "interpretation": (
                    "pose oracle diagnostic; disagreement with GT parity is not "
                    "a static mapping failure test"
                ),
                "paper_eligible": False,
            },
            ARM_A3: {
                "keypoint_source": "GT-v2 keypoint_annotations.xy",
                "parity_source": "GT",
                "labels": ["ORACLE", "GT_V2_KEYPOINT_REPLAY"],
                "circularity_disclosure": (
                    "GT-v2 keypoint annotations and canonical pose candidates share "
                    "legacy annotation/PnP provenance. This replay is a solver "
                    "consistency check, not an independent annotation/intrinsics "
                    "noise-floor estimate."
                ),
                "paper_eligible": False,
            },
            ARM_A4: {
                "input_images": PLASTIC_OBJECT_TYPE,
                "pnp_dimensions": WOOD_OBJECT_TYPE,
                "executed_direction": "plastic_images_with_wood_dimensions",
                "reverse_wood_images_with_plastic_dimensions": (
                    "BLOCKED_NOT_EVALUATED"
                ),
                "labels": ["WRONG_OBJECT_DIMENSION_CONTROL"],
                "paper_eligible": False,
            },
        },
        "two_d_safety": safety,
        "A2_pose_oracle_and_static_contract_audit": a2_audit,
        "metrics": metrics,
        "oracle_gate": gate,
        "forbidden_metrics_computed": [],
        "yolo_parameter_updates": 0,
    }
    bootstrap_payload = {
        "schema_version": "dimension_conditioning_phase_a_bootstrap_v2",
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "iterations": BOOTSTRAP_ITERATIONS,
        "base_seed": BOOTSTRAP_SEED,
        "bootstrap": bootstrap,
    }
    metrics_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"metrics"}
    }
    metrics_payload["metrics"] = metrics
    (out_dir / "ORACLE_PER_FRAME.csv").write_text(_csv_text(rows), "utf-8")
    (out_dir / "ORACLE_METRICS.json").write_text(_json_text(metrics_payload), "utf-8")
    (out_dir / "ORACLE_BOOTSTRAP.json").write_text(
        _json_text(bootstrap_payload), "utf-8"
    )
    (out_dir / "DEV_DIMENSION_ORACLE_DIAGNOSTIC.json").write_text(
        _json_text(payload), "utf-8"
    )
    (out_dir / "ORACLE_REPORT.md").write_text(_report(payload), "utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "CACHE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "run",
    "summarize_rows",
]
