"""Validation helpers for the additive real-pallet GT v2 schema.

GT v2 deliberately leaves the historical NDDS-compatible fields in place.
Paper-facing consumers use the explicit v2 fields and must not infer a signed
physical pose from ``legacy.dimensions_m`` or ``legacy.pose_transform``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # Package import (tests/evaluation code).
    from .pallet_geometry import (
        AxisAssignment,
        CameraFacingDimensionsWHD,
        axis_assignment_candidates_from_camera_facing_dimensions,
        canonical_to_camera_facing_keypoint_permutation,
        canonical_to_camera_facing_transform,
        physical_dimensions_xyz,
        validate_proper_rotation,
    )
    from .object_geometry_registry import (
        PLASTIC_OBJECT_TYPE,
        WOOD_OBJECT_TYPE,
        ObjectGeometrySpec,
        load_object_geometry_registry,
    )
except ImportError:  # Direct ``python scripts/annotate/...`` execution.
    from pallet_geometry import (
        AxisAssignment,
        CameraFacingDimensionsWHD,
        axis_assignment_candidates_from_camera_facing_dimensions,
        canonical_to_camera_facing_keypoint_permutation,
        canonical_to_camera_facing_transform,
        physical_dimensions_xyz,
        validate_proper_rotation,
    )
    from object_geometry_registry import (  # type: ignore[no-redef]
        PLASTIC_OBJECT_TYPE,
        WOOD_OBJECT_TYPE,
        ObjectGeometrySpec,
        load_object_geometry_registry,
    )


SCHEMA_VERSION = "real_pallet_gt_v2"
KEYPOINT_FRAME = "camera_dynamic_0123_v4"
KEYPOINT_COUNT = 9

VISIBILITY_VALUES = frozenset({0, 1, 2})
KEYPOINT_SOURCES = frozenset({
    "manual_click",
    "extrapolated",
    "pnp_projected",
    "centroid_auto",
    "unknown",
})
KEYPOINT_REASONS = frozenset({
    "visible",
    "occluded",
    "truncated",
    "unknown",
})
OCCLUSION_LEVELS = frozenset({"none", "partial", "heavy", "unknown"})
INTRINSICS_QUALITY_VALUES = frozenset({
    "CALIBRATED",
    "SENSOR_PROFILE_SCALED",
    "ESTIMATED_HFOV",
    "UNKNOWN",
})


class SchemaValidationError(ValueError):
    """Raised when a document claims GT v2 but violates its contract."""


def _fail(path: str, message: str) -> None:
    raise SchemaValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail(path, "must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(path, "must be a finite number")
    if not np.isfinite(number):
        _fail(path, "must be finite")
    return number


def _validate_xy(value: Any, path: str) -> None:
    if value is None:
        return
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != 2):
        _fail(path, "must be null or [x, y]")
    _finite_number(value[0], f"{path}[0]")
    _finite_number(value[1], f"{path}[1]")


def _validate_pose_transform(value: Any, path: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail(path, "must be a numeric 4x4 matrix")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        _fail(path, "must be a finite 4x4 matrix")
    if not np.allclose(
            transform[3], np.array([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0, atol=1e-9):
        _fail(path, "last row must be [0, 0, 0, 1]")
    try:
        validate_proper_rotation(transform[:3, :3], name=path, atol=1e-6)
    except ValueError as exc:
        _fail(path, str(exc))
    return transform


def _validate_assignment(value: Any, path: str) -> AxisAssignment:
    if not isinstance(value, str):
        _fail(path, "must be a signed YAW_* string")
    try:
        return AxisAssignment(value)
    except ValueError:
        _fail(path, "must be YAW_0, YAW_90, YAW_180, or YAW_270")


def _validate_permutation(value: Any, path: str) -> tuple[int, ...]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or len(value) != KEYPOINT_COUNT):
        _fail(path, f"must contain {KEYPOINT_COUNT} indices")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        _fail(path, "must contain integer indices")
    result = tuple(int(item) for item in value)
    if set(result) != set(range(KEYPOINT_COUNT)):
        _fail(path, "must be a bijection of indices 0..8")
    return result


def validate_keypoint_annotation(entry: Any, index: int | None = None) -> None:
    """Validate one v2 per-keypoint annotation.

    Visibility 0 may retain an ``xy`` migrated from the legacy label.  It means
    the visibility/provenance state is unknown, not that the historical
    coordinate has been deleted.  Visibility 1/2 requires a coordinate.
    """

    path = ("keypoint_annotation" if index is None
            else f"keypoint_annotations[{index}]")
    item = _mapping(entry, path)
    for field in ("xy", "visibility", "in_frame", "source", "reason"):
        if field not in item:
            _fail(path, f"missing required field {field!r}")
    _validate_xy(item["xy"], f"{path}.xy")
    visibility = item["visibility"]
    if type(visibility) is not int or visibility not in VISIBILITY_VALUES:
        _fail(f"{path}.visibility", "must be integer 0, 1, or 2")
    if visibility in (1, 2) and item["xy"] is None:
        _fail(f"{path}.xy", "is required when visibility is 1 or 2")
    if not isinstance(item["in_frame"], bool):
        _fail(f"{path}.in_frame", "must be bool")
    if item["source"] not in KEYPOINT_SOURCES:
        _fail(
            f"{path}.source",
            f"must be one of {sorted(KEYPOINT_SOURCES)}",
        )
    if item["reason"] not in KEYPOINT_REASONS:
        _fail(
            f"{path}.reason",
            f"must be one of {sorted(KEYPOINT_REASONS)}",
        )


def keypoint_annotations_to_ultralytics(
        annotations: Sequence[Mapping[str, Any]],
) -> list[list[float]]:
    """Map GT-v2 points to the local Ultralytics ``[x, y, v]`` contract.

    The audited Ultralytics 8.4.60 loader/loss masks only ``v == 0`` and treats
    1 and 2 as equally supervised locations.  A migrated visibility-0 point may
    retain legacy ``xy`` for audit/bbox geometry, but the training target must
    still be ``[0, 0, 0]`` so unknown provenance is not silently supervised.
    Coordinates are returned in their input units; dataset-specific
    normalization belongs to the surrounding label converter.
    """

    if (not isinstance(annotations, Sequence)
            or isinstance(annotations, (str, bytes))
            or len(annotations) != KEYPOINT_COUNT):
        _fail(
            "keypoint_annotations",
            f"must contain exactly {KEYPOINT_COUNT} entries",
        )
    result: list[list[float]] = []
    for index, entry in enumerate(annotations):
        validate_keypoint_annotation(entry, index=index)
        visibility = int(entry["visibility"])
        xy = entry["xy"]
        if visibility == 0 or xy is None:
            result.append([0.0, 0.0, 0.0])
        else:
            result.append([
                float(xy[0]),
                float(xy[1]),
                float(visibility),
            ])
    return result


def _validate_physical_dimensions(
        value: Any,
        path: str,
        expected_dimensions: Any,
) -> None:
    item = _mapping(value, path)
    expected = physical_dimensions_xyz(expected_dimensions).as_dict()
    if set(item) != {"x", "y", "z"}:
        _fail(path, "must contain exactly x, y, z")
    for axis, expected_value in expected.items():
        actual = _finite_number(item[axis], f"{path}.{axis}")
        if not np.isclose(actual, expected_value, rtol=0.0, atol=1e-12):
            _fail(
                f"{path}.{axis}",
                f"must equal the canonical value {expected_value}",
            )


def _validate_camera_dimensions(
        value: Any, path: str) -> CameraFacingDimensionsWHD:
    item = _mapping(value, path)
    if set(item) != {"width", "height", "depth"}:
        _fail(path, "must contain exactly width, height, depth")
    for key in ("width", "height", "depth"):
        if _finite_number(item[key], f"{path}.{key}") <= 0.0:
            _fail(f"{path}.{key}", "must be positive")
    try:
        return CameraFacingDimensionsWHD(
            width_m=item["width"],
            height_m=item["height"],
            depth_m=item["depth"],
        )
    except (TypeError, ValueError) as exc:
        _fail(path, str(exc))


def _validate_rotation(value: Any, path: str) -> None:
    try:
        validate_proper_rotation(value, name=path, atol=1e-6)
    except (TypeError, ValueError) as exc:
        _fail(path, str(exc))


def _validate_pose_record(
        value: Any,
        path: str,
    *,
    camera_facing_transform: np.ndarray,
    physical_dimensions: Any,
) -> AxisAssignment:
    item = _mapping(value, path)
    for field in (
        "axis_assignment",
        "pose_transform",
        "canonical_to_camera_facing_rotation",
        "canonical_to_camera_facing_keypoint_permutation",
    ):
        if field not in item:
            _fail(path, f"missing required field {field!r}")
    assignment = _validate_assignment(
        item["axis_assignment"], f"{path}.axis_assignment")
    pose_transform = _validate_pose_transform(
        item["pose_transform"], f"{path}.pose_transform")
    _validate_rotation(
        item["canonical_to_camera_facing_rotation"],
        f"{path}.canonical_to_camera_facing_rotation",
    )
    permutation = _validate_permutation(
        item["canonical_to_camera_facing_keypoint_permutation"],
        f"{path}.canonical_to_camera_facing_keypoint_permutation",
    )
    expected_rotation = canonical_to_camera_facing_transform(assignment)
    actual_rotation = np.asarray(
        item["canonical_to_camera_facing_rotation"], dtype=np.float64)
    if not np.array_equal(actual_rotation, expected_rotation):
        _fail(
            f"{path}.canonical_to_camera_facing_rotation",
            "does not match the signed axis assignment",
        )
    expected_permutation = (
        canonical_to_camera_facing_keypoint_permutation(
            assignment, physical_dimensions))
    if permutation != expected_permutation:
        _fail(
            f"{path}.canonical_to_camera_facing_keypoint_permutation",
            "does not match exact coordinate-set matching for the assignment",
        )
    expected_pose = np.eye(4, dtype=np.float64)
    expected_pose[:3, :3] = (
        camera_facing_transform[:3, :3] @ expected_rotation)
    expected_pose[:3, 3] = camera_facing_transform[:3, 3]
    if not np.allclose(
            pose_transform, expected_pose, rtol=0.0, atol=1e-6):
        _fail(
            f"{path}.pose_transform",
            "is not the proper camera-facing to canonical pose conversion",
        )
    return assignment


def _object_geometry(data: Mapping[str, Any], obj: Mapping[str, Any]) -> ObjectGeometrySpec:
    """Resolve an explicit object type, retaining old plastic-v2 compatibility."""

    root_type = data.get("object_type")
    object_type = obj.get("object_type")
    if root_type is None and object_type is None:
        # Existing plastic GT v2 predates the multi-object field.  This is the
        # only compatibility default; manifests still declare plastic before
        # paper evaluation opens the label.
        canonical_type = PLASTIC_OBJECT_TYPE
    else:
        if not isinstance(root_type, str) or not isinstance(object_type, str):
            _fail("object_type", "must be present at root and objects[0]")
        if root_type != object_type:
            _fail("object_type", "root and objects[0] values must match")
        canonical_type = root_type
    try:
        spec = load_object_geometry_registry().resolve(canonical_type)
    except ValueError as exc:
        _fail("object_type", str(exc))
    if spec.object_type == WOOD_OBJECT_TYPE:
        if data.get("population_role") not in {"DEV", "FINAL"}:
            _fail("population_role", "wood GT must have role DEV or FINAL")
        quality = data.get("intrinsics_quality")
        if quality not in INTRINSICS_QUALITY_VALUES:
            _fail(
                "intrinsics_quality",
                f"must be one of {sorted(INTRINSICS_QUALITY_VALUES)}",
            )
    return spec


def validate_gt_v2(document: Any) -> None:
    """Validate the additive fields required on a GT v2 document."""

    data = _mapping(document, "document")
    if data.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version", f"must equal {SCHEMA_VERSION!r}")
    objects = data.get("objects")
    if (not isinstance(objects, Sequence) or isinstance(objects, (str, bytes))
            or len(objects) != 1):
        _fail("objects", "must contain exactly one object")
    obj = _mapping(objects[0], "objects[0]")
    geometry_spec = _object_geometry(data, obj)
    physical_dimensions = geometry_spec.physical_dimensions

    required = (
        "keypoint_frame",
        "physical_dimensions_m",
        "camera_facing_pnp",
        "canonical_pose",
        "canonical_pose_candidates",
        "legacy",
        "keypoint_annotations",
        "occlusion_level",
        "truncation",
        "migration_status",
    )
    for field in required:
        if field not in obj:
            _fail("objects[0]", f"missing required v2 field {field!r}")
    if obj["keypoint_frame"] != KEYPOINT_FRAME:
        _fail("objects[0].keypoint_frame", f"must equal {KEYPOINT_FRAME!r}")
    _validate_physical_dimensions(
        obj["physical_dimensions_m"],
        "objects[0].physical_dimensions_m",
        physical_dimensions,
    )

    cf = _mapping(obj["camera_facing_pnp"], "objects[0].camera_facing_pnp")
    for field in (
        "axis_assignment",
        "axis_assignment_candidates",
        "dimensions_m",
        "pose_transform",
    ):
        if field not in cf:
            _fail("objects[0].camera_facing_pnp", f"missing {field!r}")
    if cf["axis_assignment"] is not None:
        _validate_assignment(
            cf["axis_assignment"], "objects[0].camera_facing_pnp.axis_assignment")
    candidates = cf["axis_assignment_candidates"]
    if (not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes)) or len(candidates) < 1):
        _fail(
            "objects[0].camera_facing_pnp.axis_assignment_candidates",
            "must be a non-empty list",
        )
    candidate_assignments = tuple(
        _validate_assignment(value, (
            "objects[0].camera_facing_pnp.axis_assignment_candidates"
            f"[{index}]"
        )) for index, value in enumerate(candidates)
    )
    if len(set(candidate_assignments)) != len(candidate_assignments):
        _fail(
            "objects[0].camera_facing_pnp.axis_assignment_candidates",
            "must not contain duplicates",
        )
    cf_dimensions = _validate_camera_dimensions(
        cf["dimensions_m"], "objects[0].camera_facing_pnp.dimensions_m")
    try:
        expected_candidates = (
            axis_assignment_candidates_from_camera_facing_dimensions(
                cf_dimensions,
                physical_dimensions=physical_dimensions,
            ))
    except (TypeError, ValueError) as exc:
        _fail("objects[0].camera_facing_pnp.dimensions_m", str(exc))
    if candidate_assignments != expected_candidates:
        _fail(
            "objects[0].camera_facing_pnp.axis_assignment_candidates",
            "must be the deterministic signed pair allowed by W/D parity",
        )
    cf_transform = _validate_pose_transform(
        cf["pose_transform"], "objects[0].camera_facing_pnp.pose_transform")

    canonical_pose = obj["canonical_pose"]
    canonical_assignment = None
    if canonical_pose is not None:
        canonical_assignment = _validate_pose_record(
            canonical_pose,
            "objects[0].canonical_pose",
            camera_facing_transform=cf_transform,
            physical_dimensions=physical_dimensions,
        )

    pose_candidates = obj["canonical_pose_candidates"]
    if (not isinstance(pose_candidates, Sequence)
            or isinstance(pose_candidates, (str, bytes))):
        _fail("objects[0].canonical_pose_candidates", "must be a list")
    seen: set[AxisAssignment] = set()
    for index, record in enumerate(pose_candidates):
        path = f"objects[0].canonical_pose_candidates[{index}]"
        assignment = _validate_pose_record(
            record,
            path,
            camera_facing_transform=cf_transform,
            physical_dimensions=physical_dimensions,
        )
        if assignment in seen:
            _fail(path, "duplicates an axis assignment")
        seen.add(assignment)
        parity_error = record.get("projection_parity_max_px")
        if parity_error is not None and _finite_number(
                parity_error, f"{path}.projection_parity_max_px") < 0.0:
            _fail(f"{path}.projection_parity_max_px", "must be non-negative")
    if seen != set(candidate_assignments):
        _fail(
            "objects[0].canonical_pose_candidates",
            "assignments must match camera_facing_pnp candidates",
        )

    selected_assignment = cf["axis_assignment"]
    if selected_assignment is None:
        if canonical_pose is not None:
            _fail(
                "objects[0].canonical_pose",
                "must be null until one signed axis assignment is confirmed",
            )
    else:
        selected = AxisAssignment(selected_assignment)
        if selected not in candidate_assignments:
            _fail(
                "objects[0].camera_facing_pnp.axis_assignment",
                "must be one of axis_assignment_candidates",
            )
        if canonical_pose is None:
            _fail(
                "objects[0].canonical_pose",
                "is required for a confirmed signed axis assignment",
            )
        if canonical_assignment != selected:
            _fail(
                "objects[0].canonical_pose.axis_assignment",
                "must match camera_facing_pnp.axis_assignment",
            )

    legacy = _mapping(obj["legacy"], "objects[0].legacy")
    for field in ("dimensions_m", "pose_transform", "fix_swap"):
        if field not in legacy:
            _fail("objects[0].legacy", f"missing {field!r}")

    annotations = obj["keypoint_annotations"]
    if (not isinstance(annotations, Sequence)
            or isinstance(annotations, (str, bytes))
            or len(annotations) != KEYPOINT_COUNT):
        _fail(
            "objects[0].keypoint_annotations",
            f"must contain exactly {KEYPOINT_COUNT} entries",
        )
    for index, entry in enumerate(annotations):
        validate_keypoint_annotation(entry, index=index)

    if obj["occlusion_level"] not in OCCLUSION_LEVELS:
        _fail(
            "objects[0].occlusion_level",
            f"must be one of {sorted(OCCLUSION_LEVELS)}",
        )
    truncation = _mapping(obj["truncation"], "objects[0].truncation")
    for field in ("is_truncated", "outside_keypoints", "bbox_outside_fraction"):
        if field not in truncation:
            _fail("objects[0].truncation", f"missing {field!r}")
    if (truncation["is_truncated"] is not None
            and type(truncation["is_truncated"]) is not bool):
        _fail("objects[0].truncation.is_truncated", "must be bool or null")
    outside = truncation["outside_keypoints"]
    if (not isinstance(outside, Sequence) or isinstance(outside, (str, bytes))
            or any(isinstance(index, bool) or not isinstance(index, int)
                   or not 0 <= index < KEYPOINT_COUNT for index in outside)):
        _fail(
            "objects[0].truncation.outside_keypoints",
            "must contain keypoint indices 0..8",
        )
    if len(set(outside)) != len(outside):
        _fail(
            "objects[0].truncation.outside_keypoints",
            "must not contain duplicate indices",
        )
    fraction = truncation["bbox_outside_fraction"]
    if fraction is not None:
        fraction_value = _finite_number(
            fraction, "objects[0].truncation.bbox_outside_fraction")
        if not 0.0 <= fraction_value <= 1.0:
            _fail(
                "objects[0].truncation.bbox_outside_fraction",
                "must be in [0, 1] or null",
            )

    if canonical_pose is None and not pose_candidates:
        _fail(
            "objects[0].canonical_pose_candidates",
            "must preserve candidates while canonical_pose is unresolved",
        )


def validate_gt_document(document: Any, *, allow_legacy: bool = True) -> str:
    """Validate v2 or acknowledge an old readable schema.

    Returns ``"v2"`` or ``"legacy"``.  Legacy acceptance is intentionally
    shallow: it is for backward-compatible loading, not paper evaluation.
    """

    data = _mapping(document, "document")
    if data.get("schema_version") == SCHEMA_VERSION:
        validate_gt_v2(data)
        return "v2"
    if not allow_legacy:
        _fail("schema_version", f"must equal {SCHEMA_VERSION!r}")
    objects = data.get("objects")
    if (not isinstance(objects, Sequence) or isinstance(objects, (str, bytes))
            or len(objects) < 1 or not isinstance(objects[0], Mapping)):
        _fail("objects", "legacy document must contain objects[0]")
    return "legacy"


__all__ = [
    "KEYPOINT_COUNT",
    "KEYPOINT_FRAME",
    "KEYPOINT_REASONS",
    "KEYPOINT_SOURCES",
    "OCCLUSION_LEVELS",
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "VISIBILITY_VALUES",
    "keypoint_annotations_to_ultralytics",
    "validate_gt_document",
    "validate_gt_v2",
    "validate_keypoint_annotation",
]
