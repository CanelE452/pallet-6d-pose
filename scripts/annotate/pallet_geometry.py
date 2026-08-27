"""Typed pallet geometry and physical/camera-facing frame conversions.

The legacy annotation code represents dimensions as a positional ``(W, D, H)``
tuple.  JSON uses the named order ``width, height, depth``.  This module keeps
those two conventions out of the physical-frame API by using distinct named
dataclasses.

Frame contract
--------------

Canonical physical frame (centroid origin, right handed)::

    +X  physical 1.10 m axis
    +Y  physical top-to-bottom axis (``down``), 0.11 m
    +Z  physical 1.30 m axis

The camera-facing frame uses the same sign convention as
``annotate_pnp.make_pallet_keypoints_3d_diagram``: +X right, +Y down, +Z far;
indices 0..3 are the near face.  ``AxisAssignment`` describes the signed yaw
that maps canonical coordinates into that camera-facing coordinate system.

For ``A = canonical_to_camera_facing_transform(assignment)`` and
``p = canonical_to_camera_facing_keypoint_permutation(assignment)``::

    P_cf[i] = A @ P_canonical[p[i]]
    R_canonical = R_cf @ A
    t_canonical = t_cf

The translation equality relies on both frames sharing keypoint 8 as their
origin.  A parity-only W/D decision is deliberately insufficient for the pose
conversion: 0/180 and 90/270 remain distinct signed physical poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable

import numpy as np


CANONICAL_X_M = 1.10
CANONICAL_Y_M = 0.11
CANONICAL_Z_M = 1.30

# Migration/evaluation gates use the same numerical definition of a proper
# rotation.  Exact quarter-turn frame transforms still have zero error.
_ROTATION_ATOL = 1e-6


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


@dataclass(frozen=True, kw_only=True)
class PhysicalDimensionsXYZ:
    """Full physical extents in the fixed canonical X/Y/Z frame."""

    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _positive_finite(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _positive_finite(self.y_m, "y_m"))
        object.__setattr__(self, "z_m", _positive_finite(self.z_m, "z_m"))

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x_m, "y": self.y_m, "z": self.z_m}


@dataclass(frozen=True, kw_only=True)
class CameraFacingDimensionsWHD:
    """Full extents named in the dynamic camera-facing W/H/D frame."""

    width_m: float
    height_m: float
    depth_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "width_m", _positive_finite(self.width_m, "width_m"))
        object.__setattr__(
            self, "height_m", _positive_finite(self.height_m, "height_m"))
        object.__setattr__(
            self, "depth_m", _positive_finite(self.depth_m, "depth_m"))

    def as_dict(self) -> dict[str, float]:
        return {
            "width": self.width_m,
            "height": self.height_m,
            "depth": self.depth_m,
        }

    def as_legacy_wdh_tuple(self) -> tuple[float, float, float]:
        """Explicit adapter for legacy annotation functions expecting W,D,H."""

        return (self.width_m, self.depth_m, self.height_m)


class AxisAssignment(str, Enum):
    """Signed canonical yaw relative to the camera-facing frame."""

    YAW_0 = "YAW_0"
    YAW_90 = "YAW_90"
    YAW_180 = "YAW_180"
    YAW_270 = "YAW_270"

    @property
    def yaw_degrees(self) -> int:
        return {
            AxisAssignment.YAW_0: 0,
            AxisAssignment.YAW_90: 90,
            AxisAssignment.YAW_180: 180,
            AxisAssignment.YAW_270: 270,
        }[self]


def _require_signed_assignment(
        assignment: AxisAssignment | str) -> AxisAssignment:
    """Return a full signed assignment; reject parity-only descriptions."""

    if isinstance(assignment, AxisAssignment):
        return assignment
    if isinstance(assignment, str):
        try:
            return AxisAssignment(assignment)
        except ValueError as exc:
            raise ValueError(
                "axis_assignment must be one signed value: "
                "YAW_0, YAW_90, YAW_180, or YAW_270; a W/D parity-only "
                "assignment cannot define a canonical pose"
            ) from exc
    raise TypeError(
        "axis_assignment must be an AxisAssignment or its exact string value")


def canonical_dimensions() -> PhysicalDimensionsXYZ:
    """Return the one physical pallet dimension contract used by GT v2."""

    return PhysicalDimensionsXYZ(
        x_m=CANONICAL_X_M, y_m=CANONICAL_Y_M, z_m=CANONICAL_Z_M)


def _diagram_points(width: float, height: float, depth: float) -> np.ndarray:
    """Camera-facing index diagram for named W/H/D extents."""

    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-w, -h, -d],
        [+w, -h, -d],
        [+w, +h, -d],
        [-w, +h, -d],
        [-w, -h, +d],
        [+w, -h, +d],
        [+w, +h, +d],
        [-w, +h, +d],
    ], dtype=np.float64)
    return np.vstack([corners, np.zeros((1, 3), dtype=np.float64)])


def canonical_keypoints_3d() -> np.ndarray:
    """Return 8 fixed physical corners and centroid in canonical index order."""

    dims = canonical_dimensions()
    return _diagram_points(dims.x_m, dims.y_m, dims.z_m)


def canonical_to_camera_facing_transform(
        axis_assignment: AxisAssignment | str) -> np.ndarray:
    """Return the exact proper rotation mapping canonical coordinates to CF."""

    assignment = _require_signed_assignment(axis_assignment)
    # Exact integer quarter-turn, not sin/cos values close to zero.  Its sign is
    # part of the tested frame contract: +X -> -Z and +Z -> +X at YAW_90.
    quarter_turn = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    turns = assignment.yaw_degrees // 90
    rotation = np.linalg.matrix_power(quarter_turn, turns).astype(np.float64)
    validate_proper_rotation(rotation, name="canonical_to_camera_facing")
    return rotation


def camera_facing_dimensions(
        axis_assignment: AxisAssignment | str) -> CameraFacingDimensionsWHD:
    """Return named W/H/D extents for one fully signed yaw assignment."""

    assignment = _require_signed_assignment(axis_assignment)
    dims = canonical_dimensions()
    if assignment in (AxisAssignment.YAW_0, AxisAssignment.YAW_180):
        width, depth = dims.x_m, dims.z_m
    else:
        width, depth = dims.z_m, dims.x_m
    return CameraFacingDimensionsWHD(
        width_m=width, height_m=dims.y_m, depth_m=depth)


def camera_facing_keypoints_3d(
        axis_assignment: AxisAssignment | str) -> np.ndarray:
    """Return camera-facing model points for one signed physical assignment."""

    dims = camera_facing_dimensions(axis_assignment)
    return _diagram_points(dims.width_m, dims.height_m, dims.depth_m)


@lru_cache(maxsize=4)
def _generated_permutation(assignment: AxisAssignment) -> tuple[int, ...]:
    """Generate ``perm[cf_index] = canonical_index`` by 3-D exact matching."""

    canonical = canonical_keypoints_3d()
    rotation = canonical_to_camera_facing_transform(assignment)
    rotated = (rotation @ canonical.T).T
    target = camera_facing_keypoints_3d(assignment)

    permutation: list[int] = []
    used: set[int] = set()
    for cf_index, point in enumerate(target):
        # Every transform is an integer quarter-turn and both cuboids are
        # constructed from the same named extents.  Requiring literal equality
        # keeps the mapping a coordinate-set contract, not a nearest-neighbour
        # heuristic that could silently accept a changed model.
        matches = np.flatnonzero(np.all(rotated == point, axis=1))
        if len(matches) != 1:
            raise RuntimeError(
                f"axis {assignment.value}: camera-facing keypoint {cf_index} "
                f"has {len(matches)} canonical coordinate matches")
        canonical_index = int(matches[0])
        if canonical_index in used:
            raise RuntimeError(
                f"axis {assignment.value}: coordinate matching is not bijective")
        used.add(canonical_index)
        permutation.append(canonical_index)

    expected = set(range(len(canonical)))
    if used != expected:
        raise RuntimeError(
            f"axis {assignment.value}: coordinate matching omitted indices "
            f"{sorted(expected - used)}")
    return tuple(permutation)


def canonical_to_camera_facing_keypoint_permutation(
        axis_assignment: AxisAssignment | str) -> tuple[int, ...]:
    """Return the automatically matched ``perm[cf] = canonical`` bijection."""

    assignment = _require_signed_assignment(axis_assignment)
    return _generated_permutation(assignment)


def axis_assignment_candidates_from_camera_facing_dimensions(
        dimensions: CameraFacingDimensionsWHD,
        *,
        atol: float = 1e-9,
) -> tuple[AxisAssignment, AxisAssignment]:
    """Return the two signed yaw candidates allowed by a W/D parity choice.

    Dimensions distinguish the physical X and Z axes because 1.10 != 1.30,
    but cannot distinguish their signs.  The function therefore always returns
    a two-element tuple and never chooses a canonical pose.
    """

    if not isinstance(dimensions, CameraFacingDimensionsWHD):
        raise TypeError("dimensions must be CameraFacingDimensionsWHD")
    physical = canonical_dimensions()
    if not np.isclose(dimensions.height_m, physical.y_m, rtol=0.0, atol=atol):
        raise ValueError("camera-facing height does not match canonical Y")
    short_width = (
        np.isclose(dimensions.width_m, physical.x_m, rtol=0.0, atol=atol)
        and np.isclose(dimensions.depth_m, physical.z_m, rtol=0.0, atol=atol)
    )
    long_width = (
        np.isclose(dimensions.width_m, physical.z_m, rtol=0.0, atol=atol)
        and np.isclose(dimensions.depth_m, physical.x_m, rtol=0.0, atol=atol)
    )
    if short_width:
        return (AxisAssignment.YAW_0, AxisAssignment.YAW_180)
    if long_width:
        return (AxisAssignment.YAW_90, AxisAssignment.YAW_270)
    raise ValueError(
        "camera-facing W/H/D is not one of the two canonical axis parities")


def validate_proper_rotation(
        rotation: np.ndarray | Iterable[Iterable[float]],
        *,
        name: str = "rotation",
        atol: float = _ROTATION_ATOL,
) -> np.ndarray:
    """Validate and return a 3x3 finite orthonormal matrix with det +1."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    orthogonality_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > atol:
        raise ValueError(
            f"{name} is not orthonormal (max error {orthogonality_error})")
    if abs(determinant - 1.0) > atol:
        kind = "reflection" if determinant < 0.0 else "non-proper rotation"
        raise ValueError(f"{name} is a {kind} (det={determinant})")
    return matrix


def camera_facing_to_canonical_pose(
        R_cf: np.ndarray | Iterable[Iterable[float]],
        t_cf: np.ndarray | Iterable[float],
        axis_assignment: AxisAssignment | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a camera-facing object pose into one signed canonical pose."""

    assignment = _require_signed_assignment(axis_assignment)
    rotation_cf = validate_proper_rotation(R_cf, name="R_cf")
    translation_cf = np.asarray(t_cf, dtype=np.float64).reshape(-1)
    if translation_cf.shape != (3,) or not np.isfinite(translation_cf).all():
        raise ValueError("t_cf must contain three finite values")
    axis_rotation = canonical_to_camera_facing_transform(assignment)
    rotation_canonical = rotation_cf @ axis_rotation
    validate_proper_rotation(rotation_canonical, name="R_canonical")
    return rotation_canonical, translation_cf.copy()


def make_pose_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a validated 4x4 object-to-camera transform."""

    rotation = validate_proper_rotation(rotation)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


__all__ = [
    "AxisAssignment",
    "CameraFacingDimensionsWHD",
    "PhysicalDimensionsXYZ",
    "axis_assignment_candidates_from_camera_facing_dimensions",
    "camera_facing_dimensions",
    "camera_facing_keypoints_3d",
    "camera_facing_to_canonical_pose",
    "canonical_dimensions",
    "canonical_keypoints_3d",
    "canonical_to_camera_facing_keypoint_permutation",
    "canonical_to_camera_facing_transform",
    "make_pose_transform",
    "validate_proper_rotation",
]
