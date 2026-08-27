"""Paper-facing real evaluation with explicit population and pose gates.

Historical evaluators remain untouched.  This command requires caller-supplied
manifests, validates them against the repo registry and refuses to overwrite an
existing output.  ``--dry-run`` performs no model import or weight inference;
it writes a strict contract report whose unavailable metrics are JSON ``null``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # Support ``python challenge/evaluation_v2/paper_real_eval.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from challenge.evaluation_v2.pnp_selector import (  # type: ignore[import-not-found]
        PnPSelectionResult,
        SelectorGateReport,
        SelectorStatus,
        assess_selector_diagnostics,
        select_pnp_hypotheses,
    )
    from challenge.evaluation_v2.pose_metrics import (  # type: ignore[import-not-found]
        PoseErrorRecord,
        PoseMetricGate,
        add_error_m,
        build_pose_metric_gate,
        model_diameter_m,
        rotation_error_degrees,
        summarize_pose_errors,
        translation_error_m,
        yaw_error_degrees,
    )
    from challenge.evaluation_v2.real_dataset_contract import (  # type: ignore[import-not-found]
        ContractError,
        EvaluationPopulationPair,
        ManifestItem,
        PopulationId,
        PopulationManifest,
        PopulationRole,
        REPO_ROOT,
        load_population_manifest,
        load_repo_population,
        validate_common_dev_membership,
        validate_evaluation_pair,
        validate_registered_membership,
    )
    from scripts.annotate import pallet_geometry as geometry  # type: ignore[import-not-found]
    from scripts.annotate.pallet_symmetry import (  # type: ignore[import-not-found]
        SYMMETRY_CONTRACT_SCHEMA_VERSION,
        SymmetryContractError,
        ValidatedSymmetryContract,
        load_symmetry_contract,
    )
    from scripts.annotate.real_gt_v2_schema import (  # type: ignore[import-not-found]
        SchemaValidationError,
        validate_gt_v2,
    )
else:
    from .pnp_selector import (
        PnPSelectionResult,
        SelectorGateReport,
        SelectorStatus,
        assess_selector_diagnostics,
        select_pnp_hypotheses,
    )
    from .pose_metrics import (
        PoseErrorRecord,
        PoseMetricGate,
        add_error_m,
        build_pose_metric_gate,
        model_diameter_m,
        rotation_error_degrees,
        summarize_pose_errors,
        translation_error_m,
        yaw_error_degrees,
    )
    from .real_dataset_contract import (
        ContractError,
        EvaluationPopulationPair,
        ManifestItem,
        PopulationId,
        PopulationManifest,
        PopulationRole,
        REPO_ROOT,
        load_population_manifest,
        load_repo_population,
        validate_common_dev_membership,
        validate_evaluation_pair,
        validate_registered_membership,
    )
    from scripts.annotate import pallet_geometry as geometry
    from scripts.annotate.pallet_symmetry import (
        SYMMETRY_CONTRACT_SCHEMA_VERSION,
        SymmetryContractError,
        ValidatedSymmetryContract,
        load_symmetry_contract,
    )
    from scripts.annotate.real_gt_v2_schema import SchemaValidationError, validate_gt_v2


REPORT_SCHEMA_VERSION = "paper_real_eval_v2"
INFERENCE_PAD = 100
INFERENCE_IMGSZ = 640
INFERENCE_CONFIDENCE_FLOOR = 0.001

MIGRATION_GATE_SCHEMA_VERSION = "real_pallet_gt_v2_migration_gate_v2"
MIGRATION_REQUIRED_COUNT = 140
SIGNED_CANONICAL_POSE = "SIGNED_CANONICAL_POSE"
YAW_180_EQUIVALENCE_CLASS = "YAW_180_EQUIVALENCE_CLASS"
MIGRATION_REQUIRED_CHECKS = (
    "source_sha_and_mtime_unchanged",
    "canonical_physical_dimensions_exact",
    "rotation_orthogonality_within_threshold",
    "rotation_determinant_within_threshold",
    "projection_parity_within_threshold",
    "manual_kps_exact",
    "legacy_fields_preserved",
    "reflection_transform_count_zero",
    "schema_valid",
    "migration_failures_zero",
    "membership_count_exact",
    "output_count_exact",
)


@dataclass(frozen=True)
class PositiveTarget:
    frame_id: str
    box_xyxy: np.ndarray
    keypoints_xy: np.ndarray
    keypoint_xy_present: np.ndarray
    keypoint_supervision_mask: np.ndarray
    visibility: np.ndarray
    camera_intrinsics: np.ndarray
    canonical_pose_transform: np.ndarray | None
    canonical_pose_candidate_transforms: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class DetectionCandidate:
    frame_id: str
    is_positive: bool
    score: float
    box_xyxy: np.ndarray
    keypoints_xy: np.ndarray | None
    target_iou: float | None


@dataclass(frozen=True)
class PoseContractContext:
    gate: PoseMetricGate
    metric_variant: Literal["ADD", "ADD-S"]
    equivalent_rotations: tuple[np.ndarray, ...]
    pose_resolution_mode: str | None
    evidence: Mapping[str, Any]


class PoseEvaluationNotRunnable(ContractError):
    """A passed global gate still cannot identify a per-frame signed pose."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--positive-manifest", required=True)
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--population-role", required=True, choices=["DEV", "FINAL"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate contracts and emit blocked/null metrics without importing a model",
    )
    parser.add_argument("--migration-gate", help="optional machine-readable migration gate JSON")
    parser.add_argument(
        "--selector-diagnostic", help="optional DEV140 per-frame selector diagnostic JSON"
    )
    parser.add_argument("--symmetry-contract", help="optional frozen symmetry contract JSON")
    parser.add_argument("--device", default="0", help="Ultralytics inference device (non-dry-run only)")
    return parser


def _load_json(path: str | Path, purpose: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ContractError(f"{purpose}_NOT_FOUND: {source}")
    try:
        value = json.loads(source.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{purpose}_UNREADABLE: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{purpose}_ROOT_MUST_BE_OBJECT")
    return value


def _evidence_int(payload: Mapping[str, Any], field: str, purpose: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise ContractError(f"{purpose}_{field.upper()}_MUST_BE_NONNEGATIVE_INT")
    return value


def _evidence_number(payload: Mapping[str, Any], field: str, purpose: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ContractError(f"{purpose}_{field.upper()}_MUST_BE_FINITE_NUMBER")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{purpose}_{field.upper()}_MUST_BE_FINITE_NUMBER") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ContractError(f"{purpose}_{field.upper()}_MUST_BE_NONNEGATIVE_FINITE")
    return number


def _read_evidence_csv(path: Path, purpose: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContractError(f"{purpose}_NOT_FOUND: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ContractError(f"{purpose}_HEADER_REQUIRED")
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(f"{purpose}_UNREADABLE: {path}: {exc}") from exc


def _repo_evidence_file(value: Any, purpose: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{purpose}_PATH_REQUIRED")
    raw = Path(value).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ContractError(f"{purpose}_ESCAPES_REPOSITORY: {value!r}") from exc
    if not resolved.is_file():
        raise ContractError(f"{purpose}_NOT_FOUND: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_nonnegative_int(row: Mapping[str, str], field: str, purpose: str) -> int:
    try:
        value = int(row.get(field, ""))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{purpose}_{field.upper()}_MUST_BE_NONNEGATIVE_INT") from exc
    if value < 0:
        raise ContractError(f"{purpose}_{field.upper()}_MUST_BE_NONNEGATIVE_INT")
    return value


def _transforms_form_exact_symmetry_class(
    transforms: Sequence[np.ndarray],
    equivalent_rotations: Sequence[np.ndarray],
    *,
    atol: float = 1e-8,
) -> bool:
    """Return whether two poses are exactly one frozen yaw-equivalence class."""

    if len(transforms) != 2 or len(equivalent_rotations) != 2:
        return False
    normalized: list[np.ndarray] = []
    for value in transforms:
        transform = np.asarray(value, dtype=np.float64)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=atol)
        ):
            return False
        try:
            geometry.validate_proper_rotation(
                transform[:3, :3], name="equivalence_class_rotation", atol=atol
            )
        except (TypeError, ValueError):
            return False
        normalized.append(transform)

    reference = normalized[0]
    if not all(
        np.allclose(
            transform[:3, 3], reference[:3, 3], rtol=0.0, atol=atol
        )
        for transform in normalized
    ):
        return False
    coverage = np.array(
        [
            [
                np.allclose(
                    transform[:3, :3],
                    reference[:3, :3] @ symmetry,
                    rtol=0.0,
                    atol=atol,
                )
                for symmetry in equivalent_rotations
            ]
            for transform in normalized
        ],
        dtype=bool,
    )
    return bool(coverage.any(axis=0).all() and coverage.any(axis=1).all())


def _validate_migration_pass(
    payload: Mapping[str, Any],
    gate_path: Path,
    *,
    symmetry_contract: ValidatedSymmetryContract | None,
) -> str:
    """Revalidate the Phase-F evidence needed by a claimed migration PASS."""

    if payload.get("dry_run") is not False:
        raise ContractError("MIGRATION_PASS_CANNOT_BE_DRY_RUN")
    counts = {
        field: _evidence_int(payload, field, "MIGRATION_GATE")
        for field in (
            "source_count",
            "migrated_count",
            "output_json_count",
            "manual_review_required_count",
            "canonical_pose_resolved_count",
            "canonical_pose_equivalence_resolved_count",
        )
    }
    if any(
        counts[field] != MIGRATION_REQUIRED_COUNT
        for field in ("source_count", "migrated_count", "output_json_count")
    ):
        raise ContractError("MIGRATION_PASS_REQUIRES_140_SOURCE_AND_OUTPUT_LABELS")
    if counts["manual_review_required_count"] != 0:
        raise ContractError("MIGRATION_PASS_REQUIRES_EMPTY_MANUAL_REVIEW_QUEUE")

    resolution_mode = payload.get("pose_resolution_mode")
    if resolution_mode == SIGNED_CANONICAL_POSE:
        if (
            counts["canonical_pose_resolved_count"] != MIGRATION_REQUIRED_COUNT
            or counts["canonical_pose_equivalence_resolved_count"] != 0
        ):
            raise ContractError("MIGRATION_PASS_SIGNED_MODE_REQUIRES_140_SINGULAR_POSES")
    elif resolution_mode == YAW_180_EQUIVALENCE_CLASS:
        if (
            counts["canonical_pose_resolved_count"] != 0
            or counts["canonical_pose_equivalence_resolved_count"]
            != MIGRATION_REQUIRED_COUNT
        ):
            raise ContractError(
                "MIGRATION_PASS_EQUIVALENCE_MODE_REQUIRES_140_CLASSES_AND_ZERO_SINGULAR_POSES"
            )
        if symmetry_contract is None:
            raise ContractError(
                "MIGRATION_PASS_EQUIVALENCE_MODE_REQUIRES_FROZEN_SYMMETRY_CONTRACT"
            )
        if (
            symmetry_contract.metric_variant != "ADD-S"
            or symmetry_contract.equivalent_yaw_degrees != (0, 180)
            or len(symmetry_contract.rotations) != 2
        ):
            raise ContractError(
                "MIGRATION_PASS_EQUIVALENCE_MODE_REQUIRES_EXACT_YAW180_ADD_S_CONTRACT"
            )
        gate_symmetry_path = _repo_evidence_file(
            payload.get("symmetry_contract_path"),
            "MIGRATION_PASS_SYMMETRY_CONTRACT",
        )
        if symmetry_contract.source_path is None or gate_symmetry_path != symmetry_contract.source_path:
            raise ContractError("MIGRATION_PASS_SYMMETRY_CONTRACT_PATH_MISMATCH")
        gate_symmetry_sha = payload.get("symmetry_contract_sha256")
        if gate_symmetry_sha != symmetry_contract.sha256:
            raise ContractError("MIGRATION_PASS_SYMMETRY_CONTRACT_SHA256_MISMATCH")
        if _file_sha256(gate_symmetry_path) != gate_symmetry_sha:
            raise ContractError("MIGRATION_PASS_SYMMETRY_CONTRACT_CURRENT_SHA256_MISMATCH")
    else:
        raise ContractError("MIGRATION_PASS_POSE_RESOLUTION_MODE_INVALID")

    if payload.get("geometry_candidate_checks_pass") is not True:
        raise ContractError("MIGRATION_PASS_REQUIRES_GEOMETRY_CHECKS_PASS")

    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise ContractError("MIGRATION_GATE_CHECKS_REQUIRED")
    missing = [name for name in MIGRATION_REQUIRED_CHECKS if checks.get(name) is not True]
    if missing:
        raise ContractError(f"MIGRATION_PASS_REQUIRED_CHECKS_NOT_TRUE: {missing}")
    if (
        resolution_mode == YAW_180_EQUIVALENCE_CLASS
        and checks.get("yaw180_equivalence_class_exact") is not True
    ):
        raise ContractError("MIGRATION_PASS_YAW180_EQUIVALENCE_CHECK_NOT_TRUE")
    if (
        resolution_mode == YAW_180_EQUIVALENCE_CLASS
        and checks.get("symmetry_contract_sha256_current") is not True
    ):
        raise ContractError("MIGRATION_PASS_SYMMETRY_CONTRACT_CURRENT_CHECK_NOT_TRUE")
    if payload.get("reflection_transform_count") != 0:
        raise ContractError("MIGRATION_PASS_REQUIRES_ZERO_REFLECTIONS")
    if payload.get("failures") != []:
        raise ContractError("MIGRATION_PASS_REQUIRES_EMPTY_FAILURES")

    thresholds = payload.get("thresholds")
    maxima = payload.get("maxima")
    if not isinstance(thresholds, Mapping) or not isinstance(maxima, Mapping):
        raise ContractError("MIGRATION_PASS_REQUIRES_THRESHOLDS_AND_MAXIMA")
    rotation_limit = _evidence_number(thresholds, "rotation_max_error", "MIGRATION_GATE")
    projection_limit = _evidence_number(
        thresholds, "projection_parity_max_px", "MIGRATION_GATE"
    )
    # A PASS artifact may tighten the preregistered limits, never weaken them.
    if rotation_limit > 1e-6 or projection_limit > 1e-4:
        raise ContractError("MIGRATION_GATE_THRESHOLDS_WEAKER_THAN_PHASE_F_CONTRACT")
    orthogonality = _evidence_number(
        maxima, "rotation_orthogonality_max_error", "MIGRATION_GATE"
    )
    determinant = _evidence_number(
        maxima, "rotation_det_max_abs_error", "MIGRATION_GATE"
    )
    projection = _evidence_number(maxima, "projection_parity_max_px", "MIGRATION_GATE")
    if orthogonality > rotation_limit or determinant > rotation_limit:
        raise ContractError("MIGRATION_PASS_ROTATION_MAXIMUM_EXCEEDS_THRESHOLD")
    if projection > projection_limit:
        raise ContractError("MIGRATION_PASS_PROJECTION_MAXIMUM_EXCEEDS_THRESHOLD")

    audit_path = payload.get("source_audit_csv")
    if not isinstance(audit_path, str) or not audit_path.strip():
        raise ContractError("MIGRATION_PASS_SOURCE_AUDIT_CSV_REQUIRED")
    source = _repo_evidence_file(audit_path, "MIGRATION_PASS_SOURCE_AUDIT_CSV")
    audit_rows = _read_evidence_csv(source, "MIGRATION_PASS_SOURCE_AUDIT_CSV")
    audit_ids = [row.get("frame_id", "") for row in audit_rows]
    if (
        len(audit_ids) != MIGRATION_REQUIRED_COUNT
        or not all(audit_ids)
        or len(set(audit_ids)) != len(audit_ids)
    ):
        raise ContractError("MIGRATION_PASS_SOURCE_AUDIT_REQUIRES_140_UNIQUE_FRAMES")
    audit_by_id = {row["frame_id"]: row for row in audit_rows}

    report_rows = _read_evidence_csv(
        gate_path.parent / "MIGRATION_REPORT.csv", "MIGRATION_PASS_REPORT_CSV"
    )
    report_ids = [row.get("frame_id", "") for row in report_rows]
    if len(report_ids) != MIGRATION_REQUIRED_COUNT or set(report_ids) != set(audit_ids):
        raise ContractError("MIGRATION_PASS_REPORT_MUST_MATCH_ALL_140_AUDIT_FRAMES")
    source_paths: set[Path] = set()
    output_paths: set[Path] = set()
    row_orthogonality: list[float] = []
    row_determinant: list[float] = []
    row_projection: list[float] = []
    for row in report_rows:
        frame_id = row["frame_id"]
        audit = audit_by_id[frame_id]
        expected_row_status = (
            "CANONICAL_POSE_CONFIRMED"
            if resolution_mode == SIGNED_CANONICAL_POSE
            else "CANONICAL_POSE_EQUIVALENCE_RESOLVED"
        )
        if row.get("status") != expected_row_status:
            raise ContractError("MIGRATION_PASS_REPORT_POSE_RESOLUTION_STATUS_MISMATCH")
        if (
            resolution_mode == YAW_180_EQUIVALENCE_CLASS
            and row.get("yaw180_equivalence_class_exact") != "True"
        ):
            raise ContractError(
                "MIGRATION_PASS_REPORT_YAW180_EQUIVALENCE_CLASS_NOT_EXACT"
            )
        if row.get("reflection_count") != "0":
            raise ContractError("MIGRATION_PASS_REPORT_HAS_REFLECTION")
        for field in (
            "manual_kps_preserved",
            "legacy_fields_preserved",
            "schema_valid",
            "source_untouched",
        ):
            if row.get(field) != "True":
                raise ContractError(f"MIGRATION_PASS_REPORT_{field.upper()}_NOT_TRUE")

        source_path = _repo_evidence_file(
            row.get("source_label"), "MIGRATION_PASS_REPORT_SOURCE_LABEL"
        )
        audit_source_path = _repo_evidence_file(
            audit.get("label_path"), "MIGRATION_PASS_AUDIT_SOURCE_LABEL"
        )
        if source_path != audit_source_path or source_path in source_paths:
            raise ContractError("MIGRATION_PASS_SOURCE_PATH_MEMBERSHIP_MISMATCH")
        source_paths.add(source_path)

        baseline_sha = audit.get("label_sha256", "")
        report_sha_before = row.get("source_sha_before", "")
        report_sha_after = row.get("source_sha_after", "")
        if (
            len(baseline_sha) != 64
            or any(character not in "0123456789abcdef" for character in baseline_sha)
            or report_sha_before != baseline_sha
            or report_sha_after != baseline_sha
            or _file_sha256(source_path) != baseline_sha
        ):
            raise ContractError("MIGRATION_PASS_SOURCE_SHA_MISMATCH")

        baseline_mtime = _csv_nonnegative_int(
            audit, "label_mtime_ns", "MIGRATION_PASS_AUDIT"
        )
        baseline_size = _csv_nonnegative_int(
            audit, "label_size_bytes", "MIGRATION_PASS_AUDIT"
        )
        report_mtime_before = _csv_nonnegative_int(
            row, "source_mtime_ns_before", "MIGRATION_PASS_REPORT"
        )
        report_mtime_after = _csv_nonnegative_int(
            row, "source_mtime_ns_after", "MIGRATION_PASS_REPORT"
        )
        report_size_before = _csv_nonnegative_int(
            row, "source_size_bytes_before", "MIGRATION_PASS_REPORT"
        )
        report_size_after = _csv_nonnegative_int(
            row, "source_size_bytes_after", "MIGRATION_PASS_REPORT"
        )
        current_stat = source_path.stat()
        if not (
            report_mtime_before
            == report_mtime_after
            == baseline_mtime
            == current_stat.st_mtime_ns
            and report_size_before
            == report_size_after
            == baseline_size
            == current_stat.st_size
        ):
            raise ContractError("MIGRATION_PASS_SOURCE_MTIME_OR_SIZE_MISMATCH")

        row_orthogonality.append(
            _evidence_number(
                row, "rotation_orthogonality_max_error", "MIGRATION_PASS_REPORT"
            )
        )
        row_determinant.append(
            _evidence_number(
                row, "rotation_det_max_abs_error", "MIGRATION_PASS_REPORT"
            )
        )
        row_projection.append(
            _evidence_number(
                row, "projection_parity_max_px", "MIGRATION_PASS_REPORT"
            )
        )
        if (
            row_orthogonality[-1] > rotation_limit
            or row_determinant[-1] > rotation_limit
            or row_projection[-1] > projection_limit
        ):
            raise ContractError("MIGRATION_PASS_REPORT_FRAME_MAXIMUM_EXCEEDS_THRESHOLD")

        output_path = _repo_evidence_file(
            row.get("output_label"), "MIGRATION_PASS_REPORT_OUTPUT_LABEL"
        )
        if output_path in output_paths:
            raise ContractError("MIGRATION_PASS_OUTPUT_LABEL_PATHS_MUST_BE_UNIQUE")
        output_paths.add(output_path)
        output_payload = _load_json(output_path, "MIGRATION_PASS_OUTPUT_GT")
        objects = output_payload.get("objects")
        if not isinstance(objects, list) or len(objects) != 1:
            raise ContractError("MIGRATION_PASS_OUTPUT_GT_REQUIRES_EXACTLY_ONE_OBJECT")
        try:
            validate_gt_v2(output_payload)
        except SchemaValidationError as exc:
            raise ContractError(
                f"MIGRATION_PASS_OUTPUT_GT_SCHEMA_INVALID: {frame_id}: {exc}"
            ) from exc
        output_object = objects[0]
        canonical_pose = output_object.get("canonical_pose")
        camera_facing = output_object.get("camera_facing_pnp")
        if not isinstance(camera_facing, Mapping):
            raise ContractError(
                f"MIGRATION_PASS_OUTPUT_GT_CAMERA_FACING_PNP_REQUIRED: {frame_id}"
            )
        if resolution_mode == SIGNED_CANONICAL_POSE:
            if canonical_pose is None or camera_facing.get("axis_assignment") is None:
                raise ContractError(
                    f"MIGRATION_PASS_OUTPUT_GT_CANONICAL_POSE_REQUIRED: {frame_id}"
                )
        else:
            if canonical_pose is not None or camera_facing.get("axis_assignment") is not None:
                raise ContractError(
                    f"MIGRATION_PASS_EQUIVALENCE_OUTPUT_MUST_NOT_CLAIM_SIGNED_POSE: {frame_id}"
                )
            raw_candidates = output_object.get("canonical_pose_candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
                raise ContractError(
                    f"MIGRATION_PASS_EQUIVALENCE_OUTPUT_REQUIRES_TWO_CANDIDATES: {frame_id}"
                )
            candidate_transforms = tuple(
                np.asarray(candidate.get("pose_transform"), dtype=np.float64)
                for candidate in raw_candidates
                if isinstance(candidate, Mapping)
            )
            if symmetry_contract is None:  # Guard even under ``python -O``.
                raise ContractError(
                    "MIGRATION_PASS_EQUIVALENCE_MODE_LOST_SYMMETRY_CONTEXT"
                )
            if not _transforms_form_exact_symmetry_class(
                candidate_transforms, symmetry_contract.rotations
            ):
                raise ContractError(
                    f"MIGRATION_PASS_OUTPUT_GT_YAW180_EQUIVALENCE_CLASS_INVALID: {frame_id}"
                )

    if len(source_paths) != MIGRATION_REQUIRED_COUNT:
        raise ContractError("MIGRATION_PASS_SOURCE_LABEL_PATHS_MUST_BE_UNIQUE")
    if len(output_paths) != MIGRATION_REQUIRED_COUNT:
        raise ContractError("MIGRATION_PASS_OUTPUT_LABEL_PATHS_MUST_BE_UNIQUE")
    if not (
        math.isclose(max(row_orthogonality), orthogonality, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(max(row_determinant), determinant, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(max(row_projection), projection, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ContractError("MIGRATION_PASS_AGGREGATE_MAXIMA_DO_NOT_MATCH_REPORT")
    review_rows = _read_evidence_csv(
        gate_path.parent / "MANUAL_REVIEW_QUEUE.csv",
        "MIGRATION_PASS_MANUAL_REVIEW_QUEUE",
    )
    if review_rows:
        raise ContractError("MIGRATION_PASS_MANUAL_REVIEW_QUEUE_MUST_BE_EMPTY")
    return resolution_mode


def _migration_status(
    path: str | None,
    *,
    symmetry_contract: ValidatedSymmetryContract | None = None,
) -> tuple[str, dict[str, Any] | None]:
    if path is None:
        return "NOT_RUN", None
    payload = _load_json(path, "MIGRATION_GATE")
    if payload.get("schema_version") != MIGRATION_GATE_SCHEMA_VERSION:
        raise ContractError("MIGRATION_GATE_SCHEMA_VERSION_INVALID")
    status = payload.get("status")
    if status not in {"PASS", "BLOCKED"}:
        raise ContractError("MIGRATION_GATE_STATUS_MUST_BE_PASS_OR_BLOCKED")
    if status == "PASS":
        _validate_migration_pass(
            payload,
            Path(path).expanduser().resolve(),
            symmetry_contract=symmetry_contract,
        )
    return status, payload


def _selector_status(path: str | None) -> tuple[SelectorGateReport, dict[str, Any] | None]:
    if path is None:
        return SelectorGateReport.not_run(), None
    payload = _load_json(path, "SELECTOR_DIAGNOSTIC")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ContractError("SELECTOR_DIAGNOSTIC_RECORDS_REQUIRED")
    report = assess_selector_diagnostics(
        records,
        tail_dominance_assessed=payload.get("tail_dominance_assessed", False),
        tail_dominance_passed=payload.get("tail_dominance_passed"),
        tail_dominance_notes=payload.get("tail_dominance_notes"),
    )
    return report, payload


def _symmetry_status(
    path: str | None,
) -> tuple[
    str,
    str | None,
    tuple[np.ndarray, ...],
    dict[str, Any] | None,
    ValidatedSymmetryContract | None,
]:
    if path is None:
        return "NOT_FROZEN", None, (), None, None
    try:
        contract = load_symmetry_contract(path)
    except SymmetryContractError as exc:
        raise ContractError(f"SYMMETRY_CONTRACT_INVALID: {exc}") from exc
    return (
        contract.status,
        contract.metric_variant,
        contract.rotations,
        contract.payload,
        contract,
    )


def validate_evaluation_request(
    *,
    positive_manifest: str | Path,
    negative_manifest: str | Path,
    population_role: PopulationRole | str,
    allow_unavailable_final: bool,
) -> EvaluationPopulationPair:
    """Validate caller manifests, repo registration and the allowed pair."""

    positive = load_population_manifest(positive_manifest, validate_files=True)
    negative = load_population_manifest(negative_manifest, validate_files=True)
    validate_registered_membership(positive, validate_files=True)
    validate_registered_membership(negative, validate_files=True)
    if positive.population_id is PopulationId.COMMON_DEV_POS128:
        validate_common_dev_membership(
            load_repo_population(PopulationId.DEV_POS140, validate_files=True),
            positive,
            validate_files=True,
        )
    return validate_evaluation_pair(
        positive,
        negative,
        population_role,
        allow_unavailable_final=allow_unavailable_final,
    )


def _output_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise ContractError(f"OUTPUT_ALREADY_EXISTS: {target}")
    if not target.parent.is_dir():
        raise ContractError(f"OUTPUT_PARENT_NOT_FOUND: {target.parent}")
    return target


def _strict_json_text(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ContractError(f"REPORT_NOT_STRICT_JSON: {exc}") from exc


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    text = _strict_json_text(payload)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ContractError(f"OUTPUT_ALREADY_EXISTS: {path}") from exc


def _blocked_2d_metrics(reason: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "blocked_reason": reason,
        "box_ap50_95": None,
        "box_ap50": None,
        "keypoint_location_median_px": None,
        "keypoint_location_p90_px": None,
    }


def _pose_contract(
    args: argparse.Namespace,
    pair: EvaluationPopulationPair,
) -> PoseContractContext:
    # Symmetry is validated first because an equivalence-resolved migration PASS
    # is meaningful only when it is bound to these exact frozen contract bytes.
    (
        symmetry_status,
        metric_variant,
        equivalent_rotations,
        symmetry_payload,
        symmetry_contract,
    ) = _symmetry_status(args.symmetry_contract)
    migration_status, migration_payload = _migration_status(
        args.migration_gate,
        symmetry_contract=symmetry_contract,
    )
    selector_report, selector_payload = _selector_status(args.selector_diagnostic)
    pose_resolution_mode = (
        migration_payload.get("pose_resolution_mode")
        if migration_status == "PASS" and migration_payload is not None
        else None
    )
    final_frozen = bool(
        pair.role is PopulationRole.FINAL
        and pair.ready
        and pair.positive.frozen
        and pair.negative.frozen
    )
    gate = build_pose_metric_gate(
        canonical_migration_status=migration_status,
        selector_report=selector_report,
        symmetry_status=symmetry_status,
        final_manifest_frozen=final_frozen,
    )
    evidence = {
        "migration_gate_supplied": migration_payload is not None,
        "migration_gate_schema_validated": migration_payload is not None,
        "migration_pose_resolution_mode": pose_resolution_mode,
        "selector_diagnostic_supplied": selector_payload is not None,
        "selector_gate_report": selector_report.to_dict(),
        "symmetry_contract_supplied": symmetry_payload is not None,
        "symmetry_contract_evidence_validated": symmetry_status == "FROZEN",
        "symmetry_contract_sha256": (
            symmetry_contract.sha256 if symmetry_contract is not None else None
        ),
        "accepted_symmetry_rotation_count": len(equivalent_rotations),
    }
    return PoseContractContext(
        gate=gate,
        metric_variant=metric_variant or "ADD",
        equivalent_rotations=equivalent_rotations,
        pose_resolution_mode=pose_resolution_mode,
        evidence=evidence,
    )


def _base_report(
    args: argparse.Namespace,
    pair: EvaluationPopulationPair,
    *,
    inference_status: str,
    metrics_2d: Mapping[str, Any],
    pose_context: PoseContractContext,
    pose_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_mode": "DRY_RUN" if args.dry_run else "INFERENCE",
        "population_contract": pair.summary(),
        "weights": {
            "argument": args.weights,
            "loaded": not args.dry_run and inference_status == "COMPLETE",
        },
        "inference": {
            "status": inference_status,
            "recipe": {
                "pad": INFERENCE_PAD,
                "border": "BORDER_REFLECT_101",
                "imgsz": INFERENCE_IMGSZ,
                "confidence_floor": INFERENCE_CONFIDENCE_FLOOR,
            },
        },
        "metrics": {
            "box_and_keypoint_2d": dict(metrics_2d),
            "pose": dict(pose_metrics),
        },
        "pose_contract": pose_context.gate.to_dict(),
        "gate_evidence": dict(pose_context.evidence),
    }


def _legacy_forbidden_target(item: ManifestItem) -> PositiveTarget:
    path = (REPO_ROOT / item.label).resolve()  # item paths were already repo-validated
    payload = _load_json(path, "POSITIVE_GT")
    if payload.get("schema_version") != "real_pallet_gt_v2":
        raise ContractError(
            f"PAPER_EVALUATOR_REQUIRES_GT_V2: {item.frame_id}; legacy fields are forbidden"
        )
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        raise ContractError(f"PAPER_GT_V2_REQUIRES_EXACTLY_ONE_OBJECT: {item.frame_id}")
    try:
        # Paper evaluation deliberately uses the full validator, not a local
        # subset.  This catches fixed-dimension violations, reflections,
        # assignment/permutation mismatches and inconsistent candidate poses.
        validate_gt_v2(payload)
    except SchemaValidationError as exc:
        raise ContractError(f"INVALID_GT_V2_SCHEMA: {item.frame_id}: {exc}") from exc
    obj = objects[0]
    annotations = obj.get("keypoint_annotations")
    if not isinstance(annotations, list) or len(annotations) != 9:
        raise ContractError(f"INVALID_GT_V2_KEYPOINT_ANNOTATIONS: {item.frame_id}")

    xy = np.full((9, 2), np.nan, dtype=np.float64)
    xy_present = np.zeros(9, dtype=bool)
    visibility = np.zeros(9, dtype=np.int64)
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ContractError(f"INVALID_GT_V2_KEYPOINT_{item.frame_id}_{index}")
        value = annotation.get("xy")
        vis = annotation.get("visibility")
        if vis not in {0, 1, 2}:
            raise ContractError(f"INVALID_GT_V2_VISIBILITY_{item.frame_id}_{index}")
        visibility[index] = int(vis)
        if value is not None:
            point = np.asarray(value, dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ContractError(f"INVALID_GT_V2_XY_{item.frame_id}_{index}")
            xy[index] = point
            xy_present[index] = True
        if vis > 0 and not xy_present[index]:
            raise ContractError(
                f"GT_V2_SUPERVISED_KEYPOINT_REQUIRES_XY: {item.frame_id}:{index}"
            )
    if not xy_present[:8].all():
        raise ContractError(f"GT_V2_BOX_REQUIRES_ALL_8_CORNER_LOCATIONS: {item.frame_id}")
    corners = xy[:8]
    box = np.array(
        [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()],
        dtype=np.float64,
    )
    # Visibility 0 explicitly means no supervised location even when migration
    # preserves a legacy projected coordinate for box geometry/provenance.
    supervision_mask = xy_present & (visibility > 0)

    camera_data = payload.get("camera_data")
    if not isinstance(camera_data, Mapping):
        raise ContractError(f"GT_V2_CAMERA_DATA_REQUIRED: {item.frame_id}")
    intrinsics = camera_data.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ContractError(f"GT_V2_CAMERA_INTRINSICS_REQUIRED: {item.frame_id}")
    try:
        camera_matrix = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"GT_V2_CAMERA_INTRINSICS_INVALID: {item.frame_id}") from exc
    if (
        not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0.0
        or camera_matrix[1, 1] <= 0.0
    ):
        raise ContractError(f"GT_V2_CAMERA_INTRINSICS_INVALID: {item.frame_id}")

    canonical_pose = obj["canonical_pose"]
    canonical_transform = (
        np.asarray(canonical_pose["pose_transform"], dtype=np.float64)
        if canonical_pose is not None
        else None
    )
    raw_pose_candidates = obj["canonical_pose_candidates"]
    candidate_transforms = tuple(
        np.asarray(candidate["pose_transform"], dtype=np.float64)
        for candidate in raw_pose_candidates
    )
    return PositiveTarget(
        item.frame_id,
        box,
        xy,
        xy_present,
        supervision_mask,
        visibility,
        camera_matrix,
        canonical_transform,
        candidate_transforms,
    )


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


class _UltralyticsPredictor:
    """Lazy adapter so dry-run and unit tests need no Ultralytics installation."""

    def __init__(self, weights: Path, device: str):
        try:
            import cv2  # type: ignore
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ContractError(f"INFERENCE_DEPENDENCY_UNAVAILABLE: {exc}") from exc
        self.cv2 = cv2
        self.model = YOLO(str(weights), task="pose")
        self.device = device

    def predict(self, image_path: Path) -> list[tuple[float, np.ndarray, np.ndarray | None]]:
        image = self.cv2.imread(str(image_path))
        if image is None:
            raise ContractError(f"IMAGE_DECODE_FAILED: {image_path}")
        padded = self.cv2.copyMakeBorder(
            image,
            INFERENCE_PAD,
            INFERENCE_PAD,
            INFERENCE_PAD,
            INFERENCE_PAD,
            self.cv2.BORDER_REFLECT_101,
        )
        result = self.model.predict(
            padded,
            conf=INFERENCE_CONFIDENCE_FLOOR,
            imgsz=INFERENCE_IMGSZ,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        scores = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy() - INFERENCE_PAD
        keypoints = None
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.detach().cpu().numpy() - INFERENCE_PAD
        return [
            (
                float(scores[index]),
                np.asarray(boxes[index], dtype=np.float64),
                (
                    np.asarray(keypoints[index], dtype=np.float64)
                    if keypoints is not None
                    else None
                ),
            )
            for index in range(len(scores))
        ]


def _average_precision_at_iou(
    candidates: Sequence[DetectionCandidate],
    positive_count: int,
    threshold: float,
) -> float:
    if positive_count <= 0:
        raise ContractError("BOX_AP_REQUIRES_POSITIVE_GROUND_TRUTH")
    ordered = sorted(candidates, key=lambda row: row.score, reverse=True)
    if not ordered:
        return 0.0
    matched: set[str] = set()
    true_positive: list[float] = []
    false_positive: list[float] = []
    for candidate in ordered:
        is_match = bool(
            candidate.is_positive
            and candidate.target_iou is not None
            and candidate.target_iou >= threshold
            and candidate.frame_id not in matched
        )
        if is_match:
            matched.add(candidate.frame_id)
        true_positive.append(1.0 if is_match else 0.0)
        false_positive.append(0.0 if is_match else 1.0)
    tp = np.cumsum(np.asarray(true_positive))
    fp = np.cumsum(np.asarray(false_positive))
    recall = tp / positive_count
    precision = tp / np.maximum(tp + fp, 1e-12)
    # COCO-style 101-point interpolated AP for this IoU threshold.
    samples = np.linspace(0.0, 1.0, 101)
    interpolated = [float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0 for level in samples]
    return float(np.mean(interpolated))


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median_px": None, "p90_px": None}
    if not np.isfinite(array).all():
        raise ContractError("NONFINITE_2D_ERROR")
    return {
        "count": int(array.size),
        "median_px": float(np.median(array)),
        "p90_px": float(np.percentile(array, 90)),
    }


def _collect_predictions(
    pair: EvaluationPopulationPair,
    predictor: Any,
    *,
    validated_targets: Mapping[str, PositiveTarget] | None = None,
) -> tuple[
    dict[str, PositiveTarget],
    list[DetectionCandidate],
    dict[str, DetectionCandidate],
]:
    """Validate every positive GT first, then run inference exactly once."""

    if not pair.ready:
        raise ContractError(pair.blocked_reason or "POPULATION_PAIR_NOT_READY")
    targets = (
        dict(validated_targets)
        if validated_targets is not None
        else {item.frame_id: _legacy_forbidden_target(item) for item in pair.positive.items}
    )
    expected_ids = {item.frame_id for item in pair.positive.items}
    if set(targets) != expected_ids:
        raise ContractError("VALIDATED_TARGET_MEMBERSHIP_MISMATCH")
    candidates: list[DetectionCandidate] = []
    top_by_frame: dict[str, DetectionCandidate] = {}
    for is_positive, items in ((True, pair.positive.items), (False, pair.negative.items)):
        for item in items:
            image_path = (REPO_ROOT / item.image).resolve()
            predictions = predictor.predict(image_path)
            frame_rows: list[DetectionCandidate] = []
            for score, box, keypoints in predictions:
                if not math.isfinite(float(score)):
                    raise ContractError(f"NONFINITE_DETECTION_SCORE: {item.frame_id}")
                box_array = np.asarray(box, dtype=np.float64)
                if box_array.shape != (4,) or not np.isfinite(box_array).all():
                    raise ContractError(f"INVALID_DETECTION_BOX: {item.frame_id}")
                target_iou = (
                    _box_iou(box_array, targets[item.frame_id].box_xyxy)
                    if is_positive
                    else None
                )
                row = DetectionCandidate(
                    frame_id=item.frame_id,
                    is_positive=is_positive,
                    score=float(score),
                    box_xyxy=box_array,
                    keypoints_xy=(
                        np.asarray(keypoints, dtype=np.float64) if keypoints is not None else None
                    ),
                    target_iou=target_iou,
                )
                frame_rows.append(row)
                candidates.append(row)
            if frame_rows:
                top_by_frame[item.frame_id] = max(frame_rows, key=lambda row: row.score)
    return targets, candidates, top_by_frame


def _evaluate_2d_collected(
    pair: EvaluationPopulationPair,
    targets: Mapping[str, PositiveTarget],
    candidates: Sequence[DetectionCandidate],
    top_by_frame: Mapping[str, DetectionCandidate],
) -> dict[str, Any]:
    """Calculate 2-D metrics from one shared inference collection."""

    thresholds = np.arange(0.50, 0.951, 0.05)
    ap_values = [
        _average_precision_at_iou(candidates, len(pair.positive.items), float(threshold))
        for threshold in thresholds
    ]
    errors_all: list[float] = []
    errors_visibility_1: list[float] = []
    errors_visibility_2: list[float] = []
    matched_frames = 0
    for frame_id, target in targets.items():
        prediction = top_by_frame.get(frame_id)
        if (
            prediction is None
            or prediction.target_iou is None
            or prediction.target_iou < 0.5
            or prediction.keypoints_xy is None
            or prediction.keypoints_xy.shape != (9, 2)
            or not np.isfinite(prediction.keypoints_xy).all()
        ):
            continue
        matched_frames += 1
        errors = np.linalg.norm(prediction.keypoints_xy - target.keypoints_xy, axis=1)
        errors_all.extend(errors[target.keypoint_supervision_mask].tolist())
        errors_visibility_1.extend(errors[target.visibility == 1].tolist())
        errors_visibility_2.extend(errors[target.visibility == 2].tolist())

    all_distribution = _distribution(errors_all)
    return {
        "status": "COMPLETE",
        "blocked_reason": None,
        "box_ap50_95": float(np.mean(ap_values)),
        "box_ap50": float(ap_values[0]),
        "box_ap_by_iou": {
            f"{threshold:.2f}": value for threshold, value in zip(thresholds, ap_values)
        },
        "positive_count": len(pair.positive.items),
        "negative_count": len(pair.negative.items),
        "candidate_count": len(candidates),
        "keypoint_matched_frame_count_iou50": matched_frames,
        "keypoint_location_median_px": all_distribution["median_px"],
        "keypoint_location_p90_px": all_distribution["p90_px"],
        "keypoint_all_labeled": all_distribution,
        "keypoint_visibility_1": _distribution(errors_visibility_1),
        "keypoint_visibility_2": _distribution(errors_visibility_2),
        "note": "2-D diagnostics are separate from the blocked canonical pose columns.",
    }


def evaluate_2d_with_predictor(
    pair: EvaluationPopulationPair,
    predictor: Any,
) -> dict[str, Any]:
    """Run box AP and explicitly separate 2-D keypoint diagnostics."""

    targets, candidates, top_by_frame = _collect_predictions(pair, predictor)
    return _evaluate_2d_collected(pair, targets, candidates, top_by_frame)


def _failed_pose_record(object_diameter: float) -> PoseErrorRecord:
    return PoseErrorRecord(
        add_error_m=math.inf,
        adds_error_m=math.inf,
        object_diameter_m=object_diameter,
        rotation_error_deg=math.inf,
        translation_error_m=math.inf,
        yaw_error_deg=math.inf,
    )


def _candidate_is_covered_by_symmetry(
    reference: Any,
    candidate: Any,
    equivalent_rotations: Sequence[np.ndarray],
) -> bool:
    if not np.allclose(candidate.translation, reference.translation, rtol=0.0, atol=1e-8):
        return False
    return any(
        np.allclose(
            candidate.rotation,
            reference.rotation @ symmetry,
            rtol=0.0,
            atol=1e-8,
        )
        for symmetry in equivalent_rotations
    )


def _prediction_only_pose_candidate(
    selection: PnPSelectionResult,
    equivalent_rotations: Sequence[np.ndarray],
) -> Any | None:
    """Choose no candidate using GT; collapse only frozen benchmark equivalents."""

    if selection.status is not SelectorStatus.SELECTED or not selection.canonical_candidates:
        return None
    reference = selection.canonical_candidates[0]
    if not all(
        _candidate_is_covered_by_symmetry(reference, candidate, equivalent_rotations)
        for candidate in selection.canonical_candidates
    ):
        # In particular, an identity/ADD contract cannot collapse the current
        # selector's two signed-yaw candidates.  Looking at GT to pick one is
        # explicitly forbidden.
        raise PoseEvaluationNotRunnable(
            "POSE_SELECTOR_SIGNED_AXIS_AMBIGUITY_NOT_COVERED_BY_FROZEN_SYMMETRY"
        )
    return reference


def _target_pose_reference(
    target: PositiveTarget,
    pose_context: PoseContractContext,
) -> np.ndarray:
    """Return an internal metric representative without asserting a signed GT."""

    if pose_context.pose_resolution_mode == SIGNED_CANONICAL_POSE:
        if target.canonical_pose_transform is None:
            raise ContractError(
                f"POSE_GATE_SIGNED_MODE_REQUIRES_CANONICAL_GT: {target.frame_id}"
            )
        return target.canonical_pose_transform

    if pose_context.pose_resolution_mode == YAW_180_EQUIVALENCE_CLASS:
        if pose_context.metric_variant != "ADD-S":
            raise ContractError("POSE_EQUIVALENCE_CLASS_REQUIRES_ADD_S")
        if target.canonical_pose_transform is not None:
            raise ContractError(
                f"POSE_EQUIVALENCE_CLASS_MUST_NOT_CLAIM_SIGNED_GT: {target.frame_id}"
            )
        if not _transforms_form_exact_symmetry_class(
            target.canonical_pose_candidate_transforms,
            pose_context.equivalent_rotations,
        ):
            raise ContractError(
                f"POSE_GT_YAW180_EQUIVALENCE_CLASS_INVALID: {target.frame_id}"
            )
        # This member is only an internal representative of the quotient class.
        # Every reported orientation/ADD-S error is minimized over the frozen
        # class below, so it is never emitted or interpreted as a signed label.
        return target.canonical_pose_candidate_transforms[0]

    raise ContractError("POSE_GATE_PASS_REQUIRES_KNOWN_RESOLUTION_MODE")


def evaluate_pose_records(
    targets: Mapping[str, PositiveTarget],
    top_by_frame: Mapping[str, DetectionCandidate],
    pose_context: PoseContractContext,
) -> tuple[PoseErrorRecord, ...]:
    """Run prediction-only PnP and compare with canonical GT after all gates.

    This function checks the gate before iterating targets or invoking the
    selector.  The selector call receives only predicted points, intrinsics and
    the fixed named physical dimensions.  Signed candidates are never chosen
    by proximity to GT.
    """

    if not isinstance(pose_context, PoseContractContext):
        raise TypeError("pose_context must be PoseContractContext")
    if not pose_context.gate.passed:
        raise ContractError("POSE_COMPUTATION_FORBIDDEN_BEFORE_ALL_GATES_PASS")
    if not pose_context.equivalent_rotations:
        raise ContractError("POSE_COMPUTATION_REQUIRES_FROZEN_SYMMETRY_ROTATIONS")

    # The explicit physical cuboid corners are used as corresponding model
    # points.  Generic nearest-neighbour matching is intentionally not used:
    # it would grant unreviewed pitch/roll cuboid symmetries.
    model_points = geometry.canonical_keypoints_3d()[:8]
    object_diameter = model_diameter_m(model_points)
    physical_dimensions = geometry.canonical_dimensions()
    records: list[PoseErrorRecord] = []
    for frame_id, target in targets.items():
        target_transform = _target_pose_reference(target, pose_context)
        prediction = top_by_frame.get(frame_id)
        if (
            prediction is None
            or prediction.keypoints_xy is None
            or prediction.keypoints_xy.shape != (9, 2)
            or not np.isfinite(prediction.keypoints_xy).all()
        ):
            records.append(_failed_pose_record(object_diameter))
            continue

        # No label pose, label dimensions, axis answer, frame id or session
        # prior is accepted by this public selector API.
        selection = select_pnp_hypotheses(
            prediction.keypoints_xy,
            target.camera_intrinsics,
            physical_dimensions,
        )
        predicted = _prediction_only_pose_candidate(
            selection, pose_context.equivalent_rotations
        )
        if predicted is None:
            records.append(_failed_pose_record(object_diameter))
            continue

        target_rotation = target_transform[:3, :3]
        target_translation = target_transform[:3, 3]
        direct_add = add_error_m(
            model_points,
            predicted.rotation,
            predicted.translation,
            target_rotation,
            target_translation,
        )
        # "ADD-S" here means min corresponding-point ADD over only the
        # explicit proper rotations in the frozen benchmark contract.  It is
        # not unrestricted nearest-neighbour ADD-S on cuboid corners.
        symmetry_add = min(
            add_error_m(
                model_points,
                predicted.rotation,
                predicted.translation,
                target_rotation @ symmetry,
                target_translation,
            )
            for symmetry in pose_context.equivalent_rotations
        )
        symmetry_rotation = min(
            rotation_error_degrees(
                predicted.rotation, target_rotation @ symmetry
            )
            for symmetry in pose_context.equivalent_rotations
        )
        symmetry_yaw = min(
            yaw_error_degrees(predicted.rotation, target_rotation @ symmetry)
            for symmetry in pose_context.equivalent_rotations
        )
        records.append(
            PoseErrorRecord(
                add_error_m=direct_add,
                adds_error_m=symmetry_add,
                object_diameter_m=object_diameter,
                rotation_error_deg=symmetry_rotation,
                translation_error_m=translation_error_m(
                    predicted.translation, target_translation
                ),
                yaw_error_deg=symmetry_yaw,
            )
        )
    return tuple(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    target = _output_path(args.out)
    role = PopulationRole(args.population_role)
    pair = validate_evaluation_request(
        positive_manifest=args.positive_manifest,
        negative_manifest=args.negative_manifest,
        population_role=role,
        allow_unavailable_final=bool(args.dry_run),
    )
    pose_context = _pose_contract(args, pair)
    # Full GT-v2 validation is part of both dry-run and inference.  For
    # inference-ready populations it happens before model construction, so a
    # malformed label cannot consume inference and then fail late.
    validated_targets = {
        item.frame_id: _legacy_forbidden_target(item) for item in pair.positive.items
    }

    if args.dry_run:
        metrics_2d = _blocked_2d_metrics("DRY_RUN_NO_INFERENCE")
        pose_metrics = summarize_pose_errors(
            (), pose_context.gate, metric_variant=pose_context.metric_variant
        )
        report = _base_report(
            args,
            pair,
            inference_status="NOT_RUN_DRY_RUN",
            metrics_2d=metrics_2d,
            pose_context=pose_context,
            pose_metrics=pose_metrics,
        )
        _write_exclusive(target, report)
        return report

    if not pair.ready:
        raise ContractError(pair.blocked_reason or "POPULATION_PAIR_NOT_READY")
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise ContractError(f"WEIGHTS_NOT_FOUND: {weights}")
    predictor = _UltralyticsPredictor(weights, args.device)
    targets, candidates, top_by_frame = _collect_predictions(
        pair, predictor, validated_targets=validated_targets
    )
    metrics_2d = _evaluate_2d_collected(pair, targets, candidates, top_by_frame)
    if pose_context.gate.passed:
        try:
            pose_records = evaluate_pose_records(targets, top_by_frame, pose_context)
            pose_metrics = summarize_pose_errors(
                pose_records,
                pose_context.gate,
                metric_variant=pose_context.metric_variant,
            )
        except PoseEvaluationNotRunnable as exc:
            pose_metrics = {
                "status": "NOT_RUN",
                "metric_variant": pose_context.metric_variant,
                "blocked_reason": str(exc),
                "blocked_reasons": [str(exc)],
                "add_or_adds_auc": None,
                "rotation_median_deg": None,
                "translation_median_m": None,
                "yaw_median_deg": None,
            }
    else:
        # summarize_pose_errors returns before touching records for a blocked
        # gate.  More importantly, the selector/PnP path above is never called.
        pose_metrics = summarize_pose_errors(
            (), pose_context.gate, metric_variant=pose_context.metric_variant
        )
    report = _base_report(
        args,
        pair,
        inference_status="COMPLETE",
        metrics_2d=metrics_2d,
        pose_context=pose_context,
        pose_metrics=pose_metrics,
    )
    _write_exclusive(target, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_parser",
    "evaluate_2d_with_predictor",
    "main",
    "run",
    "validate_evaluation_request",
]
