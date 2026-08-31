"""Migrate audited legacy real-pallet GT into the additive v2 namespace.

The source labels are never edited.  The Phase-A per-frame CSV is both the
membership list and the immutable SHA/mtime baseline.  Legacy W/D resolves
only yaw parity, so every migrated document keeps ``canonical_pose = null``
and exposes the two proper signed-yaw candidates.  Without a symmetry
contract those candidates require manual review.  With the frozen benchmark
contract they form one explicit yaw-180 equivalence class; no arbitrary signed
representative is written into the GT.

Default outputs::

    challenge/real_gt_v2/migrated_gt/<legacy session structure>/*.json
    challenge/real_gt_v2/MIGRATION_REPORT.csv
    challenge/real_gt_v2/MIGRATION_GATE.json
    challenge/real_gt_v2/MANUAL_REVIEW_QUEUE.csv
    challenge/real_gt_v2/VISIBILITY_REVIEW_QUEUE.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:  # Package import (tests and library callers).
    from .pallet_geometry import (
        AxisAssignment,
        CameraFacingDimensionsWHD,
        axis_assignment_candidates_from_camera_facing_dimensions,
        camera_facing_keypoints_3d,
        camera_facing_to_canonical_pose,
        canonical_keypoints_3d,
        canonical_to_camera_facing_keypoint_permutation,
        canonical_to_camera_facing_transform,
        make_pose_transform,
        validate_proper_rotation,
    )
    from .object_geometry_registry import (
        DEFAULT_REGISTRY_PATH,
        PLASTIC_OBJECT_TYPE,
        ObjectGeometrySpec,
        load_object_geometry_registry,
    )
    from .real_gt_v2_schema import (
        KEYPOINT_COUNT,
        KEYPOINT_FRAME,
        SCHEMA_VERSION,
        validate_gt_v2,
    )
    from .pallet_symmetry import (
        SymmetryContractError,
        ValidatedSymmetryContract,
        load_symmetry_contract,
    )
except ImportError:  # Direct ``python scripts/annotate/...`` execution.
    from pallet_geometry import (
        AxisAssignment,
        CameraFacingDimensionsWHD,
        axis_assignment_candidates_from_camera_facing_dimensions,
        camera_facing_keypoints_3d,
        camera_facing_to_canonical_pose,
        canonical_keypoints_3d,
        canonical_to_camera_facing_keypoint_permutation,
        canonical_to_camera_facing_transform,
        make_pose_transform,
        validate_proper_rotation,
    )
    from object_geometry_registry import (  # type: ignore[no-redef]
        DEFAULT_REGISTRY_PATH,
        PLASTIC_OBJECT_TYPE,
        ObjectGeometrySpec,
        load_object_geometry_registry,
    )
    from real_gt_v2_schema import (
        KEYPOINT_COUNT,
        KEYPOINT_FRAME,
        SCHEMA_VERSION,
        validate_gt_v2,
    )
    from pallet_symmetry import (  # type: ignore[no-redef]
        SymmetryContractError,
        ValidatedSymmetryContract,
        load_symmetry_contract,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_CSV = (
    REPO_ROOT / "challenge/real_gt_v2/audit/LEGACY_GT_PER_FRAME.csv")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "challenge/real_gt_v2/migrated_gt"
DEFAULT_REPORT_ROOT = REPO_ROOT / "challenge/real_gt_v2"
DEFAULT_SYMMETRY_CONTRACT = (
    REPO_ROOT / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json")
SOURCE_DATA_ROOT = Path("challenge/data/01_real")

PROJECTION_PARITY_THRESHOLD_PX = 1e-4
ROTATION_THRESHOLD = 1e-6
MIGRATION_GATE_SCHEMA_VERSION = "real_pallet_gt_v2_migration_gate_v2"
MIGRATION_STATUS = "MANUAL_REVIEW_REQUIRED"
BLOCKED_REASON = "UNCONFIRMED_SIGNED_CANONICAL_AXIS"
EQUIVALENCE_RESOLUTION_MODE = "YAW_180_EQUIVALENCE_CLASS"
UNRESOLVED_RESOLUTION_MODE = "UNRESOLVED_SIGNED_AXIS"
EQUIVALENCE_REPORT_STATUS = "CANONICAL_POSE_EQUIVALENCE_RESOLVED"
EXISTING_OUTPUT_POLICIES = frozenset({"error", "skip-identical"})
DEFAULT_EXISTING_OUTPUT_POLICY = "skip-identical"


class MigrationError(RuntimeError):
    """A source frame cannot be migrated without violating the contract."""


class ExistingOutputProtectionError(MigrationError):
    """An existing v2 label cannot be proven safe to leave unchanged."""


@dataclass(frozen=True)
class FileState:
    sha256: str
    mtime_ns: int
    size_bytes: int


@dataclass(frozen=True)
class MigrationDiagnostics:
    assignments: tuple[str, ...]
    projection_parity_max_px: float
    rotation_orthogonality_max_error: float
    rotation_det_max_abs_error: float
    reflection_count: int
    manual_kps_preserved: bool
    legacy_fields_preserved: bool
    schema_valid: bool
    yaw180_equivalence_class_exact: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path) -> FileState:
    stat = path.stat()
    return FileState(
        sha256=sha256_file(path),
        mtime_ns=int(stat.st_mtime_ns),
        size_bytes=int(stat.st_size),
    )


def _portable_path(path: Path, repo_root: Path) -> str:
    """Use repository-relative artifact paths without hiding external inputs."""

    absolute = Path(path).resolve(strict=False)
    try:
        return absolute.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(absolute)


def _load_repo_symmetry_contract(
        path: Path | str,
        repo_root: Path,
) -> ValidatedSymmetryContract:
    """Load a frozen contract and require its exact file to live in the repo."""

    source = Path(path).expanduser().resolve()
    try:
        source.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise MigrationError(
            "symmetry contract must be a repository-contained file"
        ) from exc
    try:
        return load_symmetry_contract(source)
    except SymmetryContractError as exc:
        raise MigrationError(f"invalid frozen symmetry contract: {exc}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _baseline_state(row: Mapping[str, str]) -> FileState:
    try:
        return FileState(
            sha256=row["label_sha256"],
            mtime_ns=int(row["label_mtime_ns"]),
            size_bytes=int(row["label_size_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError("Phase-A CSV has an invalid source baseline") from exc


def _resolve_source(row: Mapping[str, str], repo_root: Path) -> Path:
    raw = Path(row["label_path"])
    source = raw if raw.is_absolute() else repo_root / raw
    return source.resolve()


def _destination_relative(source: Path, repo_root: Path) -> Path:
    source_root = (repo_root / SOURCE_DATA_ROOT).resolve()
    try:
        return source.relative_to(source_root)
    except ValueError as exc:
        raise MigrationError(
            f"source label is outside the audited real-data root: {source}") from exc


def _load_rows(audit_csv: Path) -> list[dict[str, str]]:
    with audit_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise MigrationError("Phase-A CSV contains no frames")
    required = {
        "frame_id", "source_set", "label_path", "label_sha256",
        "label_mtime_ns", "label_size_bytes",
    }
    missing = required - set(rows[0])
    if missing:
        raise MigrationError(
            f"Phase-A CSV is missing columns: {sorted(missing)}")
    frame_ids = [row["frame_id"] for row in rows]
    label_paths = [row["label_path"] for row in rows]
    if len(frame_ids) != len(set(frame_ids)):
        raise MigrationError("Phase-A CSV contains duplicate frame_id values")
    if len(label_paths) != len(set(label_paths)):
        raise MigrationError("Phase-A CSV contains duplicate label paths")
    return rows


def _parse_pose_transform(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MigrationError(
            "legacy pose_transform must be a numeric 4x4 matrix") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise MigrationError("legacy pose_transform must be a finite 4x4 matrix")
    if not np.allclose(
            transform[3], np.array([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0, atol=ROTATION_THRESHOLD):
        raise MigrationError("legacy pose_transform has an invalid last row")
    try:
        rotation = validate_proper_rotation(
            transform[:3, :3], name="legacy pose rotation",
            atol=ROTATION_THRESHOLD)
    except ValueError as exc:
        raise MigrationError(str(exc)) from exc
    return transform, rotation, transform[:3, 3].copy()


def _camera_facing_dimensions(value: Any) -> CameraFacingDimensionsWHD:
    if not isinstance(value, Mapping):
        raise MigrationError("legacy dimensions_m must be an object")
    try:
        return CameraFacingDimensionsWHD(
            width_m=value["width"],
            height_m=value["height"],
            depth_m=value["depth"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError(f"invalid legacy dimensions_m: {exc}") from exc


def _camera_matrix(document: Mapping[str, Any]) -> np.ndarray:
    try:
        intrinsics = document["camera_data"]["intrinsics"]
        matrix = np.array([
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError("camera_data.intrinsics is incomplete") from exc
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise MigrationError("camera intrinsics must be positive and finite")
    return matrix


def _image_size(document: Mapping[str, Any]) -> tuple[int, int]:
    try:
        width = int(document["camera_data"]["width"])
        height = int(document["camera_data"]["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError("camera_data width/height is incomplete") from exc
    if width <= 0 or height <= 0:
        raise MigrationError("camera image dimensions must be positive")
    return width, height


def _project(points: np.ndarray, rotation: np.ndarray,
             translation: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera_points = (rotation @ points.T).T + translation
    if not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 0.0).any():
        raise MigrationError("pose projection violates cheirality")
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = K[0, 0] * camera_points[:, 0] / camera_points[:, 2] + K[0, 2]
    pixels[:, 1] = K[1, 1] * camera_points[:, 1] / camera_points[:, 2] + K[1, 2]
    return pixels


def _candidate_record(
        assignment: AxisAssignment,
        R_cf: np.ndarray,
        t_cf: np.ndarray,
        K: np.ndarray,
        physical_dimensions: Any,
) -> tuple[dict[str, Any], float, float, float, int]:
    axis_rotation = canonical_to_camera_facing_transform(assignment)
    permutation = canonical_to_camera_facing_keypoint_permutation(
        assignment, physical_dimensions)
    R_canonical, t_canonical = camera_facing_to_canonical_pose(
        R_cf, t_cf, assignment)
    transform = make_pose_transform(R_canonical, t_canonical)

    legacy_points = camera_facing_keypoints_3d(
        assignment, physical_dimensions)
    canonical_ordered = canonical_keypoints_3d(
        physical_dimensions)[list(permutation)]
    legacy_pixels = _project(legacy_points, R_cf, t_cf, K)
    canonical_pixels = _project(
        canonical_ordered, R_canonical, t_canonical, K)
    parity = float(np.linalg.norm(
        legacy_pixels - canonical_pixels, axis=1).max(initial=0.0))
    orthogonality = float(np.max(np.abs(
        R_canonical.T @ R_canonical - np.eye(3))))
    det_error = float(abs(np.linalg.det(R_canonical) - 1.0))
    reflection_count = int(np.linalg.det(axis_rotation) < 0.0)

    return ({
        "axis_assignment": assignment.value,
        "pose_transform": transform.tolist(),
        "canonical_to_camera_facing_rotation": axis_rotation.tolist(),
        "canonical_to_camera_facing_keypoint_permutation": list(permutation),
        "projection_parity_max_px": parity,
        "status": "CANDIDATE_ONLY_UNCONFIRMED_SIGN",
    }, parity, orthogonality, det_error, reflection_count)


def _yaw180_equivalence_class_exact(
        records: list[Mapping[str, Any]],
) -> bool:
    """Prove that the deterministic signed pair is exactly one C2 class."""

    if len(records) != 2:
        return False
    try:
        first_assignment = AxisAssignment(records[0]["axis_assignment"])
        second_assignment = AxisAssignment(records[1]["axis_assignment"])
        first = np.asarray(records[0]["pose_transform"], dtype=np.float64)
        second = np.asarray(records[1]["pose_transform"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if first.shape != (4, 4) or second.shape != (4, 4):
        return False
    if (second_assignment.yaw_degrees - first_assignment.yaw_degrees) % 360 != 180:
        return False
    yaw180 = canonical_to_camera_facing_transform(AxisAssignment.YAW_180)
    return bool(
        np.array_equal(second[:3, :3], first[:3, :3] @ yaw180)
        and np.array_equal(second[:3, 3], first[:3, 3])
    )


def _effective_projected_keypoints(obj: Mapping[str, Any]) -> list[Any]:
    corners = obj.get("projected_cuboid")
    if not isinstance(corners, list) or len(corners) < 8:
        raise MigrationError("projected_cuboid must contain eight points")
    centroid = obj.get("projected_cuboid_centroid")
    return list(corners[:8]) + [centroid]


def _finite_xy(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x, y]).all() or (x == -1.0 and y == -1.0):
        return None
    return [x, y]


def _keypoint_annotations(
        obj: Mapping[str, Any], width: int, height: int,
) -> tuple[list[dict[str, Any]], list[int], bool | None]:
    projected = _effective_projected_keypoints(obj)
    manual = obj.get("manual_kps")
    extrapolated_mask = obj.get("extrapolated_mask")
    has_point_provenance = (
        isinstance(manual, (list, tuple))
        and len(manual) == KEYPOINT_COUNT
        and isinstance(extrapolated_mask, (list, tuple))
        and len(extrapolated_mask) == KEYPOINT_COUNT
        and all(type(value) is bool for value in extrapolated_mask)
    )
    annotations: list[dict[str, Any]] = []
    outside: list[int] = []
    unknown_coordinate = False
    for index, raw in enumerate(projected):
        source = "unknown"
        xy = None
        if has_point_provenance:
            manual_xy = _finite_xy(manual[index])
            if manual_xy is not None:
                xy = manual_xy
                source = (
                    "extrapolated" if extrapolated_mask[index]
                    else "manual_click")
        if xy is None:
            # Stored projection remains the non-destructive coordinate fallback,
            # but without a matching manual point its point-level source is not
            # promoted.  In particular, centroid fallback may be an average of
            # corners rather than a direct PnP projection.
            xy = _finite_xy(raw)
        if xy is None:
            in_frame = False
            unknown_coordinate = True
        else:
            in_frame = 0.0 <= xy[0] < width and 0.0 <= xy[1] < height
            if not in_frame:
                outside.append(index)
        annotations.append({
            "xy": xy,
            "visibility": 0,
            "in_frame": bool(in_frame),
            "source": source,
            "reason": "unknown",
        })
    is_truncated = None if unknown_coordinate else bool(outside)
    return annotations, outside, is_truncated


def _bbox_outside_fraction(
        projected: Iterable[Any], width: int, height: int,
) -> float | None:
    points = [_finite_xy(value) for value in projected]
    finite = np.asarray([point for point in points if point is not None], dtype=float)
    if len(finite) < 4:
        return None
    x0, y0 = finite.min(axis=0)
    x1, y1 = finite.max(axis=0)
    area = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
    if area <= 0.0:
        return None
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(width), x1), min(float(height), y1)
    intersection = float(max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0))
    return float(np.clip(1.0 - intersection / area, 0.0, 1.0))


def migrate_legacy_document(
        legacy_document: Mapping[str, Any],
        *,
        source_label: str,
        source_state: FileState,
        geometry_spec: ObjectGeometrySpec | None = None,
        population_role: str = "DEV",
        intrinsics_quality: str = "UNKNOWN",
        intrinsics_source: str | None = None,
) -> tuple[dict[str, Any], MigrationDiagnostics]:
    """Build and validate one GT-v2 document without touching its source."""

    if legacy_document.get("schema_version") == SCHEMA_VERSION:
        raise MigrationError("source already claims real_pallet_gt_v2")
    geometry_spec = geometry_spec or load_object_geometry_registry().resolve(
        PLASTIC_OBJECT_TYPE)
    if population_role not in {"DEV", "FINAL"}:
        raise MigrationError("population_role must be DEV or FINAL")
    try:
        legacy_obj = legacy_document["objects"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise MigrationError("legacy document has no objects[0]") from exc
    if not isinstance(legacy_obj, Mapping):
        raise MigrationError("legacy objects[0] must be an object")

    original_obj = copy.deepcopy(legacy_obj)
    migrated = copy.deepcopy(dict(legacy_document))
    migrated_obj = migrated["objects"][0]
    legacy_transform, R_cf, t_cf = _parse_pose_transform(
        original_obj.get("pose_transform"))
    cf_dimensions = _camera_facing_dimensions(original_obj.get("dimensions_m"))
    try:
        assignments = axis_assignment_candidates_from_camera_facing_dimensions(
            cf_dimensions,
            physical_dimensions=geometry_spec.physical_dimensions,
        )
    except (TypeError, ValueError) as exc:
        raise MigrationError(str(exc)) from exc
    K = _camera_matrix(legacy_document)
    width, height = _image_size(legacy_document)

    candidate_records: list[dict[str, Any]] = []
    parity_values: list[float] = []
    orthogonality_values: list[float] = []
    det_errors: list[float] = []
    reflection_count = 0
    for assignment in assignments:
        try:
            record, parity, orthogonality, det_error, reflections = (
                _candidate_record(
                    assignment,
                    R_cf,
                    t_cf,
                    K,
                    geometry_spec.physical_dimensions,
                ))
        except ValueError as exc:
            raise MigrationError(
                f"canonical candidate {assignment.value} is invalid: {exc}"
            ) from exc
        candidate_records.append(record)
        parity_values.append(parity)
        orthogonality_values.append(orthogonality)
        det_errors.append(det_error)
        reflection_count += reflections

    annotations, outside, is_truncated = _keypoint_annotations(
        original_obj, width, height)
    projected = _effective_projected_keypoints(original_obj)

    migrated["schema_version"] = SCHEMA_VERSION
    if geometry_spec.object_type != PLASTIC_OBJECT_TYPE:
        migrated["object_type"] = geometry_spec.object_type
        migrated["population_role"] = population_role
        migrated["intrinsics_quality"] = intrinsics_quality
        migrated["intrinsics_source"] = intrinsics_source
    migrated["real_gt_v2_migration"] = {
        "source_label": source_label,
        "source_sha256": source_state.sha256,
        "source_mtime_ns": source_state.mtime_ns,
        "source_size_bytes": source_state.size_bytes,
        "status": MIGRATION_STATUS,
        "blocked_reason": BLOCKED_REASON,
    }
    migrated_obj["keypoint_frame"] = KEYPOINT_FRAME
    if geometry_spec.object_type != PLASTIC_OBJECT_TYPE:
        migrated_obj["object_type"] = geometry_spec.object_type
    migrated_obj["physical_dimensions_m"] = (
        geometry_spec.physical_dimensions.as_dict())
    migrated_obj["camera_facing_pnp"] = {
        "axis_assignment": None,
        "axis_assignment_candidates": [value.value for value in assignments],
        "dimensions_m": cf_dimensions.as_dict(),
        "pose_transform": legacy_transform.tolist(),
        "selection_reason": "LEGACY_WD_RESOLVES_PARITY_ONLY",
    }
    migrated_obj["canonical_pose"] = None
    migrated_obj["canonical_pose_candidates"] = candidate_records
    migrated_obj["legacy"] = {
        "dimensions_m": copy.deepcopy(original_obj.get("dimensions_m")),
        "pose_transform": copy.deepcopy(original_obj.get("pose_transform")),
        "fix_swap": copy.deepcopy(original_obj.get("fix_swap")),
    }
    migrated_obj["keypoint_annotations"] = annotations
    migrated_obj["occlusion_level"] = "unknown"
    migrated_obj["truncation"] = {
        "is_truncated": is_truncated,
        "outside_keypoints": outside,
        "bbox_outside_fraction": _bbox_outside_fraction(
            projected[:8], width, height),
    }
    migrated_obj["migration_status"] = MIGRATION_STATUS
    migrated_obj["manual_review_reasons"] = (
        (["WOOD_SYMMETRY_UNREVIEWED"]
         if geometry_spec.symmetry_status == "UNREVIEWED" else [])
        + [
            BLOCKED_REASON,
            "LEGACY_KEYPOINT_PROVENANCE_UNKNOWN",
            "LEGACY_VISIBILITY_UNKNOWN",
        ]
    )

    manual_preserved = migrated_obj.get("manual_kps") == original_obj.get("manual_kps")
    legacy_preserved = all(
        migrated_obj.get(field) == original_obj.get(field)
        for field in original_obj
    )
    try:
        validate_gt_v2(migrated)
    except ValueError as exc:
        raise MigrationError(f"migrated v2 schema validation failed: {exc}") from exc

    diagnostics = MigrationDiagnostics(
        assignments=tuple(value.value for value in assignments),
        projection_parity_max_px=max(parity_values, default=float("inf")),
        rotation_orthogonality_max_error=max(
            orthogonality_values, default=float("inf")),
        rotation_det_max_abs_error=max(det_errors, default=float("inf")),
        reflection_count=reflection_count,
        manual_kps_preserved=manual_preserved,
        legacy_fields_preserved=legacy_preserved,
        schema_valid=True,
        yaw180_equivalence_class_exact=(
            _yaw180_equivalence_class_exact(candidate_records)),
    )
    return migrated, diagnostics


_REPORT_FIELDS = [
    "frame_id", "source_set", "source_label", "output_label", "status",
    "output_action", "reasons", "axis_assignment_candidates",
    "projection_parity_max_px",
    "rotation_orthogonality_max_error", "rotation_det_max_abs_error",
    "reflection_count", "manual_kps_preserved", "legacy_fields_preserved",
    "schema_valid", "yaw180_equivalence_class_exact",
    "source_sha_before", "source_sha_after",
    "source_mtime_ns_before", "source_mtime_ns_after",
    "source_size_bytes_before", "source_size_bytes_after", "source_untouched",
]

_QUEUE_FIELDS = [
    "frame_id", "source_set", "source_label", "output_label",
    "reason", "axis_assignment_candidates",
]

_VISIBILITY_QUEUE_FIELDS = [
    "frame_id", "source_set", "source_label", "output_label",
    "reason", "unknown_keypoint_indices", "unknown_keypoint_count",
]


def _empty_report_row(
        row: Mapping[str, str],
        source: Path,
        output: Path,
        repo_root: Path,
) -> dict[str, Any]:
    return {
        "frame_id": row["frame_id"],
        "source_set": row["source_set"],
        "source_label": _portable_path(source, repo_root),
        "output_label": _portable_path(output, repo_root),
        "status": "",
        "output_action": "",
        "reasons": "",
        "axis_assignment_candidates": "[]",
        "projection_parity_max_px": "",
        "rotation_orthogonality_max_error": "",
        "rotation_det_max_abs_error": "",
        "reflection_count": "",
        "manual_kps_preserved": "",
        "legacy_fields_preserved": "",
        "schema_valid": "",
        "yaw180_equivalence_class_exact": "",
        "source_sha_before": "",
        "source_sha_after": "",
        "source_mtime_ns_before": "",
        "source_mtime_ns_after": "",
        "source_size_bytes_before": "",
        "source_size_bytes_after": "",
        "source_untouched": "",
    }


def _write_artifacts(
        report_root: Path,
        report_rows: list[dict[str, Any]],
        queue_rows: list[dict[str, Any]],
        visibility_queue_rows: list[dict[str, Any]],
        gate: dict[str, Any],
) -> None:
    _atomic_csv(
        report_root / "MIGRATION_REPORT.csv", report_rows, _REPORT_FIELDS)
    _atomic_csv(
        report_root / "MANUAL_REVIEW_QUEUE.csv", queue_rows, _QUEUE_FIELDS)
    _atomic_csv(
        report_root / "VISIBILITY_REVIEW_QUEUE.csv",
        visibility_queue_rows,
        _VISIBILITY_QUEUE_FIELDS,
    )
    _atomic_json(report_root / "MIGRATION_GATE.json", gate)


def migrate_from_audit(
        *,
        audit_csv: Path = DEFAULT_AUDIT_CSV,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        report_root: Path = DEFAULT_REPORT_ROOT,
        repo_root: Path = REPO_ROOT,
        expected_count: int | None = 140,
        dry_run: bool = False,
        existing_output_policy: str = DEFAULT_EXISTING_OUTPUT_POLICY,
        symmetry_contract: Path | str | None = None,
        object_type: str = PLASTIC_OBJECT_TYPE,
        geometry_registry: Path | str = DEFAULT_REGISTRY_PATH,
        population_role: str = "DEV",
        intrinsics_quality: str = "UNKNOWN",
        intrinsics_source: str | None = None,
) -> dict[str, Any]:
    """Migrate the audited membership and return the aggregate gate document.

    Existing v2 JSON is never overwritten.  The default policy accepts an
    existing file only when its parsed document is exactly the deterministic
    mechanical migration result, and then skips the write.  ``error`` rejects
    every existing output.  A human-reviewed document therefore fails closed
    under either policy and remains byte-for-byte untouched.
    ``symmetry_contract=None`` preserves the unresolved signed-axis gate.  A
    supplied, valid frozen yaw-180 contract promotes the existing two-candidate
    representation as an equivalence class without filling ``canonical_pose``.
    """

    audit_csv = Path(audit_csv).resolve()
    output_root = Path(output_root).resolve()
    report_root = Path(report_root).resolve()
    repo_root = Path(repo_root).resolve()
    try:
        registry = load_object_geometry_registry(geometry_registry)
        geometry_spec = registry.resolve(object_type)
    except (OSError, ValueError) as exc:
        raise MigrationError(f"invalid object geometry registry/type: {exc}") from exc
    if population_role not in {"DEV", "FINAL"}:
        raise MigrationError("population_role must be DEV or FINAL")
    geometry_evidence = {
        "object_type": geometry_spec.object_type,
        "population_role": population_role,
        "physical_dimensions_m": geometry_spec.physical_dimensions.as_dict(),
        "geometry_registry_path": _portable_path(registry.source_path, repo_root),
        "geometry_registry_sha256": registry.sha256,
        "intrinsics_quality": intrinsics_quality,
        "intrinsics_source": intrinsics_source,
    }
    if existing_output_policy not in EXISTING_OUTPUT_POLICIES:
        raise MigrationError(
            "existing_output_policy must be one of "
            f"{sorted(EXISTING_OUTPUT_POLICIES)}")
    rows = _load_rows(audit_csv)

    report_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    visibility_queue_rows: list[dict[str, Any]] = []
    sources: list[tuple[dict[str, str], Path, Path, FileState]] = []
    preflight_errors: list[str] = []
    output_protection_errors: list[str] = []
    symmetry_contract_errors: list[str] = []
    identical_outputs: set[Path] = set()
    validated_symmetry: ValidatedSymmetryContract | None = None
    if symmetry_contract is not None:
        try:
            validated_symmetry = _load_repo_symmetry_contract(
                symmetry_contract, repo_root)
        except (OSError, MigrationError) as exc:
            reason = f"SYMMETRY_CONTRACT_INVALID:{exc}"
            preflight_errors.append(reason)
            symmetry_contract_errors.append(reason)
    if expected_count is not None and len(rows) != int(expected_count):
        preflight_errors.append(
            f"EXPECTED_COUNT_MISMATCH:{len(rows)}!={int(expected_count)}")
    symmetry_contract_path = (
        _portable_path(validated_symmetry.source_path, repo_root)
        if validated_symmetry is not None
        and validated_symmetry.source_path is not None
        else None
    )
    symmetry_contract_sha256 = (
        validated_symmetry.sha256 if validated_symmetry is not None else None)
    resolution_mode = (
        EQUIVALENCE_RESOLUTION_MODE
        if validated_symmetry is not None else UNRESOLVED_RESOLUTION_MODE)

    # Phase A established that all nine point-level visibility/provenance states
    # are unknown in every legacy frame.  Keep that human work separate from
    # signed-axis review even when a later source or geometry gate fails.
    for row in rows:
        raw_source = Path(row.get("label_path", "missing"))
        queue_source = (
            raw_source if raw_source.is_absolute() else repo_root / raw_source)
        try:
            queue_relative = _destination_relative(
                queue_source.resolve(), repo_root)
            queue_output = output_root / queue_relative
        except MigrationError:
            queue_output = (
                output_root / row.get("source_set", "unknown")
                / queue_source.name)
        visibility_queue_rows.append({
            "frame_id": row.get("frame_id", ""),
            "source_set": row.get("source_set", ""),
            "source_label": _portable_path(queue_source, repo_root),
            "output_label": _portable_path(queue_output, repo_root),
            "reason": "LEGACY_VISIBILITY_AND_PROVENANCE_UNKNOWN",
            "unknown_keypoint_indices": json.dumps(list(range(KEYPOINT_COUNT))),
            "unknown_keypoint_count": KEYPOINT_COUNT,
        })

    for row in rows:
        try:
            source = _resolve_source(row, repo_root)
            relative = _destination_relative(source, repo_root)
            output = output_root / relative
            if output.resolve() == source:
                raise MigrationError(
                    "output label resolves to its source label; refusing "
                    "an in-place migration")
            baseline = _baseline_state(row)
            if not source.is_file():
                raise MigrationError("source label does not exist")
            before = file_state(source)
            if before != baseline:
                raise MigrationError(
                    "source SHA/mtime/size differs from Phase-A baseline")

            # Protect annotation work transactionally: inspect every existing
            # destination before the write loop starts.  There is deliberately
            # no overwrite/force mode in the paper-GT migrator.
            if not dry_run and (output.exists() or output.is_symlink()):
                if output.is_symlink() or not output.is_file():
                    raise ExistingOutputProtectionError(
                        "existing output is not a regular file")
                if existing_output_policy == "error":
                    raise ExistingOutputProtectionError(
                        "output already exists (policy=error)")
                try:
                    with source.open("r", encoding="utf-8") as stream:
                        legacy = json.load(stream)
                    expected, _diagnostics = migrate_legacy_document(
                        legacy,
                        source_label=row["label_path"],
                        source_state=before,
                        geometry_spec=geometry_spec,
                        population_role=population_role,
                        intrinsics_quality=intrinsics_quality,
                        intrinsics_source=intrinsics_source,
                    )
                    with output.open("r", encoding="utf-8") as stream:
                        existing = json.load(stream)
                except (OSError, json.JSONDecodeError, MigrationError) as exc:
                    raise ExistingOutputProtectionError(
                        f"cannot verify existing output: {exc}") from exc
                if existing != expected:
                    raise ExistingOutputProtectionError(
                        "existing output differs from deterministic migration; "
                        "it may contain human review and will not be overwritten")
                identical_outputs.add(output)
            sources.append((row, source, output, before))
        except ExistingOutputProtectionError as exc:
            reason = f"{row.get('frame_id', '?')}:{exc}"
            preflight_errors.append(reason)
            output_protection_errors.append(reason)
        except (KeyError, OSError, json.JSONDecodeError, MigrationError) as exc:
            preflight_errors.append(f"{row.get('frame_id', '?')}:{exc}")

    if preflight_errors:
        if output_protection_errors:
            preflight_blocked_reason = "EXISTING_OUTPUT_PROTECTED"
        elif symmetry_contract_errors:
            preflight_blocked_reason = "SYMMETRY_CONTRACT_INVALID"
        else:
            preflight_blocked_reason = "SOURCE_BASELINE_MISMATCH"
        for row in rows:
            raw = Path(row.get("label_path", "missing"))
            source = raw if raw.is_absolute() else repo_root / raw
            output = output_root / row.get("source_set", "unknown") / source.name
            report = _empty_report_row(row, source, output, repo_root)
            report["status"] = preflight_blocked_reason
            report["output_action"] = "NOT_WRITTEN"
            report["reasons"] = json.dumps(preflight_errors, ensure_ascii=False)
            report_rows.append(report)
            queue_rows.append({
                "frame_id": row.get("frame_id", ""),
                "source_set": row.get("source_set", ""),
                "source_label": _portable_path(source, repo_root),
                "output_label": _portable_path(output, repo_root),
                "reason": preflight_blocked_reason,
                "axis_assignment_candidates": "[]",
            })
        gate = {
            "schema_version": MIGRATION_GATE_SCHEMA_VERSION,
            "status": "BLOCKED",
            "blocked_reason": preflight_blocked_reason,
            "dry_run": bool(dry_run),
            "source_audit_csv": _portable_path(audit_csv, repo_root),
            **geometry_evidence,
            "pose_resolution_mode": resolution_mode,
            "symmetry_contract_path": symmetry_contract_path,
            "symmetry_contract_sha256": symmetry_contract_sha256,
            "source_count": len(rows),
            "migrated_count": 0,
            "output_json_count": (
                0 if dry_run else sum(
                    1 for path in output_root.rglob("*.json")
                    if path.is_file())),
            "manual_review_required_count": len(queue_rows),
            "visibility_review_required_count": len(visibility_queue_rows),
            "canonical_pose_resolved_count": 0,
            "canonical_pose_equivalence_resolved_count": 0,
            "preflight_errors": preflight_errors,
            "existing_output_policy": existing_output_policy,
            "thresholds": {
                "rotation_max_error": ROTATION_THRESHOLD,
                "projection_parity_max_px": PROJECTION_PARITY_THRESHOLD_PX,
            },
        }
        _write_artifacts(
            report_root, report_rows, queue_rows, visibility_queue_rows, gate)
        return gate

    migrated_count = 0
    diagnostics_all: list[MigrationDiagnostics] = []
    failures: list[str] = []
    for row, source, output, before in sources:
        report = _empty_report_row(row, source, output, repo_root)
        report["source_sha_before"] = before.sha256
        report["source_mtime_ns_before"] = before.mtime_ns
        report["source_size_bytes_before"] = before.size_bytes
        try:
            with source.open("r", encoding="utf-8") as stream:
                legacy = json.load(stream)
            migrated, diagnostics = migrate_legacy_document(
                legacy,
                source_label=row["label_path"],
                source_state=before,
                geometry_spec=geometry_spec,
                population_role=population_role,
                intrinsics_quality=intrinsics_quality,
                intrinsics_source=intrinsics_source,
            )
            if not dry_run and output not in identical_outputs:
                _atomic_json(output, migrated)
                output_action = "CREATED"
            elif not dry_run:
                output_action = "SKIPPED_IDENTICAL"
            else:
                output_action = "DRY_RUN_NOT_WRITTEN"
            migrated_count += 1
            diagnostics_all.append(diagnostics)
            equivalence_resolved = bool(
                validated_symmetry is not None
                and diagnostics.yaw180_equivalence_class_exact)
            report.update({
                "status": (
                    EQUIVALENCE_REPORT_STATUS
                    if equivalence_resolved else MIGRATION_STATUS),
                "output_action": output_action,
                "reasons": json.dumps(
                    (["FROZEN_YAW_180_EQUIVALENCE_CLASS"]
                     if equivalence_resolved else [BLOCKED_REASON])
                    + [
                        "LEGACY_KEYPOINT_PROVENANCE_UNKNOWN",
                        "LEGACY_VISIBILITY_UNKNOWN",
                    ]
                ),
                "axis_assignment_candidates": json.dumps(
                    diagnostics.assignments),
                "projection_parity_max_px": diagnostics.projection_parity_max_px,
                "rotation_orthogonality_max_error": (
                    diagnostics.rotation_orthogonality_max_error),
                "rotation_det_max_abs_error": (
                    diagnostics.rotation_det_max_abs_error),
                "reflection_count": diagnostics.reflection_count,
                "manual_kps_preserved": diagnostics.manual_kps_preserved,
                "legacy_fields_preserved": diagnostics.legacy_fields_preserved,
                "schema_valid": diagnostics.schema_valid,
                "yaw180_equivalence_class_exact": (
                    diagnostics.yaw180_equivalence_class_exact),
            })
            if not equivalence_resolved:
                queue_rows.append({
                    "frame_id": row["frame_id"],
                    "source_set": row["source_set"],
                    "source_label": _portable_path(source, repo_root),
                    "output_label": _portable_path(output, repo_root),
                    "reason": (
                        "YAW180_EQUIVALENCE_CLASS_INVALID"
                        if validated_symmetry is not None else BLOCKED_REASON),
                    "axis_assignment_candidates": json.dumps(
                        diagnostics.assignments),
                })
        except (OSError, json.JSONDecodeError, MigrationError) as exc:
            reason = f"{row['frame_id']}:{type(exc).__name__}:{exc}"
            failures.append(reason)
            report["status"] = "MIGRATION_ERROR"
            report["output_action"] = "NOT_WRITTEN"
            report["reasons"] = json.dumps([reason], ensure_ascii=False)
            queue_rows.append({
                "frame_id": row["frame_id"],
                "source_set": row["source_set"],
                "source_label": _portable_path(source, repo_root),
                "output_label": _portable_path(output, repo_root),
                "reason": reason,
                "axis_assignment_candidates": "[]",
            })
        report_rows.append(report)

    source_untouched = True
    for report, (_, source, _output, before) in zip(report_rows, sources):
        try:
            after = file_state(source)
            unchanged = after == before
            report["source_sha_after"] = after.sha256
            report["source_mtime_ns_after"] = after.mtime_ns
            report["source_size_bytes_after"] = after.size_bytes
            report["source_untouched"] = unchanged
            source_untouched = source_untouched and unchanged
            if not unchanged:
                failures.append(
                    f"{_portable_path(source, repo_root)}:"
                    "SOURCE_CHANGED_DURING_MIGRATION")
        except OSError as exc:
            source_untouched = False
            failures.append(
                f"{_portable_path(source, repo_root)}:"
                f"SOURCE_RECHECK_FAILED:{exc}")

    projection_max = max(
        (item.projection_parity_max_px for item in diagnostics_all),
        default=None,
    )
    orthogonality_max = max(
        (item.rotation_orthogonality_max_error for item in diagnostics_all),
        default=None,
    )
    det_error_max = max(
        (item.rotation_det_max_abs_error for item in diagnostics_all),
        default=None,
    )
    reflection_count = sum(item.reflection_count for item in diagnostics_all)
    manual_preserved = all(
        item.manual_kps_preserved for item in diagnostics_all)
    legacy_preserved = all(
        item.legacy_fields_preserved for item in diagnostics_all)
    schema_valid = all(item.schema_valid for item in diagnostics_all)
    equivalence_exact = bool(diagnostics_all) and all(
        item.yaw180_equivalence_class_exact for item in diagnostics_all)
    equivalence_resolved_count = (
        sum(item.yaw180_equivalence_class_exact for item in diagnostics_all)
        if validated_symmetry is not None else 0
    )
    output_count = (0 if dry_run else sum(
        1 for _ in output_root.rglob("*.json") if _.is_file()))
    symmetry_contract_current = True
    if validated_symmetry is not None:
        try:
            symmetry_contract_current = bool(
                validated_symmetry.source_path is not None
                and sha256_file(validated_symmetry.source_path)
                == validated_symmetry.sha256)
        except OSError:
            symmetry_contract_current = False

    checks = {
        "source_sha_and_mtime_unchanged": source_untouched,
        "canonical_physical_dimensions_exact": bool(diagnostics_all),
        "rotation_orthogonality_within_threshold": bool(
            orthogonality_max is not None
            and orthogonality_max <= ROTATION_THRESHOLD),
        "rotation_determinant_within_threshold": bool(
            det_error_max is not None and det_error_max <= ROTATION_THRESHOLD),
        "projection_parity_within_threshold": bool(
            projection_max is not None
            and projection_max <= PROJECTION_PARITY_THRESHOLD_PX),
        "manual_kps_exact": manual_preserved,
        "legacy_fields_preserved": legacy_preserved,
        "reflection_transform_count_zero": reflection_count == 0,
        "schema_valid": schema_valid,
        "yaw180_equivalence_class_exact": equivalence_exact,
        "symmetry_contract_sha256_current": symmetry_contract_current,
        "migration_failures_zero": not failures,
        "membership_count_exact": (
            expected_count is None or len(rows) == int(expected_count)),
        "output_count_exact": dry_run or output_count == len(rows),
    }
    geometry_checks_pass = all(checks.values())
    equivalence_promotion_pass = bool(
        validated_symmetry is not None
        and not dry_run
        and geometry_checks_pass
        and equivalence_resolved_count == len(rows)
        and not queue_rows
    )
    if equivalence_promotion_pass:
        gate_status = "PASS"
        blocked_reason = None
    elif not geometry_checks_pass:
        gate_status = "BLOCKED"
        blocked_reason = "MIGRATION_GATE_FAILURE"
    elif validated_symmetry is None:
        gate_status = "BLOCKED"
        blocked_reason = BLOCKED_REASON
    elif dry_run:
        gate_status = "BLOCKED"
        blocked_reason = "DRY_RUN_CANNOT_PROMOTE_EQUIVALENCE_CLASS"
    else:
        gate_status = "BLOCKED"
        blocked_reason = "YAW180_EQUIVALENCE_CLASS_INVALID"
    gate = {
        "schema_version": MIGRATION_GATE_SCHEMA_VERSION,
        "status": gate_status,
        "blocked_reason": blocked_reason,
        "dry_run": bool(dry_run),
        "source_audit_csv": _portable_path(audit_csv, repo_root),
        **geometry_evidence,
        "pose_resolution_mode": resolution_mode,
        "symmetry_contract_path": symmetry_contract_path,
        "symmetry_contract_sha256": symmetry_contract_sha256,
        "existing_output_policy": existing_output_policy,
        "existing_output_identical_skip_count": len(identical_outputs),
        "source_count": len(rows),
        "migrated_count": migrated_count,
        "output_json_count": output_count,
        "manual_review_required_count": len(queue_rows),
        "visibility_review_required_count": len(visibility_queue_rows),
        "canonical_pose_resolved_count": 0,
        "canonical_pose_equivalence_resolved_count": (
            equivalence_resolved_count),
        "geometry_candidate_checks_pass": geometry_checks_pass,
        "checks": checks,
        "maxima": {
            "rotation_orthogonality_max_error": orthogonality_max,
            "rotation_det_max_abs_error": det_error_max,
            "projection_parity_max_px": projection_max,
        },
        "reflection_transform_count": reflection_count,
        "failures": failures,
        "thresholds": {
            "rotation_max_error": ROTATION_THRESHOLD,
            "projection_parity_max_px": PROJECTION_PARITY_THRESHOLD_PX,
        },
        "interpretation": (
            "All canonical targets are represented by the frozen yaw-180 "
            "equivalence class; no singular signed pose was fabricated."
            if equivalence_promotion_pass else
            "Candidate geometry may pass, but signed canonical pose remains "
            "unresolved without a valid, materialized yaw-180 equivalence gate."
        ),
    }
    _write_artifacts(
        report_root, report_rows, queue_rows, visibility_queue_rows, gate)
    return gate


def _path_argument(value: str) -> Path:
    return Path(value).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=_path_argument,
                        default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--output-root", type=_path_argument,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=_path_argument,
                        default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--repo-root", type=_path_argument, default=REPO_ROOT)
    parser.add_argument("--expected-count", type=int, default=140)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--symmetry-contract",
        type=_path_argument,
        default=None,
        help=(
            "object-specific frozen symmetry contract. Omit for UNREVIEWED; "
            "the plastic default is selected only for the plastic object type"
        ),
    )
    parser.add_argument("--object-type", default=PLASTIC_OBJECT_TYPE)
    parser.add_argument(
        "--geometry-registry", type=_path_argument, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--population-role", choices=["DEV", "FINAL"], default="DEV")
    parser.add_argument(
        "--intrinsics-quality",
        choices=["CALIBRATED", "SENSOR_PROFILE_SCALED", "ESTIMATED_HFOV", "UNKNOWN"],
        default="UNKNOWN",
    )
    parser.add_argument("--intrinsics-source", default=None)
    parser.add_argument(
        "--existing-output-policy",
        choices=sorted(EXISTING_OUTPUT_POLICIES),
        default=DEFAULT_EXISTING_OUTPUT_POLICY,
        help=(
            "existing v2 handling; skip-identical never rewrites a file and "
            "fails if it differs from mechanical migration"
        ),
    )
    args = parser.parse_args()
    registry = load_object_geometry_registry(args.geometry_registry)
    selected_object = registry.resolve(args.object_type)
    selected_symmetry = args.symmetry_contract
    if selected_symmetry is None and selected_object.object_type == PLASTIC_OBJECT_TYPE:
        selected_symmetry = DEFAULT_SYMMETRY_CONTRACT
    gate = migrate_from_audit(
        audit_csv=args.audit_csv,
        output_root=args.output_root,
        report_root=args.report_root,
        repo_root=args.repo_root,
        expected_count=args.expected_count,
        dry_run=args.dry_run,
        existing_output_policy=args.existing_output_policy,
        symmetry_contract=selected_symmetry,
        object_type=selected_object.object_type,
        geometry_registry=args.geometry_registry,
        population_role=args.population_role,
        intrinsics_quality=args.intrinsics_quality,
        intrinsics_source=args.intrinsics_source,
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    # An unresolved sign or a dry-run promotion is an expected blocked outcome.
    # Invalid/missing symmetry evidence and mechanical failures return failure.
    if gate.get("blocked_reason") == "SYMMETRY_CONTRACT_INVALID":
        return 2
    return 0 if gate.get("geometry_candidate_checks_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
