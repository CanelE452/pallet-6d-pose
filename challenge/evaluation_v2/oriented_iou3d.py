"""Exact oriented 3D box IoU.

`metric_split_lock.md` 2.3 requires the exact oriented-box IoU, explicitly not an
axis-aligned approximation.  The paper evaluator had no implementation at all; this
is it.

The intersection of two convex boxes is a convex polytope.  Each box contributes six
half-spaces, so the intersection is the feasible set of twelve linear inequalities.
We enumerate its vertices by solving every triple of planes and keeping the solutions
that satisfy all twelve constraints, then take the convex-hull volume.

    volume(A ∩ B) = ConvexHull(vertices).volume
    IoU           = inter / (volume(A) + volume(B) - inter)

No sampling, no voxelisation, no axis-aligned fallback.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

try:  # SciPy is available in the analysis environment
    from scipy.spatial import ConvexHull, QhullError
except ImportError:  # pragma: no cover - exercised only without SciPy
    ConvexHull = None  # type: ignore[assignment]
    QhullError = Exception  # type: ignore[assignment]

# 세 평면이 거의 평행하면 교점이 수치적으로 폭발한다.  그 삼중조합은 버린다.
_MIN_TRIPLE_DETERMINANT = 1e-9
# 부동소수 오차를 감안한 half-space 포함 판정 여유.
_FEASIBILITY_TOLERANCE = 1e-7


def _validate(rotation, translation, extents, name: str):
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} rotation must be a finite (3,3) matrix")
    if np.max(np.abs(matrix.T @ matrix - np.eye(3))) > 1e-6:
        raise ValueError(f"{name} rotation is not orthonormal")
    centre = np.asarray(translation, dtype=np.float64).reshape(-1)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError(f"{name} translation must be three finite values")
    size = np.asarray(extents, dtype=np.float64).reshape(-1)
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError(f"{name} extents must be three positive finite values")
    return matrix, centre, size


def _half_spaces(rotation: np.ndarray, centre: np.ndarray, extents: np.ndarray):
    """Six inequalities n.x <= d describing the box."""

    normals, offsets = [], []
    for axis in range(3):
        direction = rotation[:, axis]
        half = extents[axis] / 2.0
        normals.append(direction)
        offsets.append(float(direction @ centre) + half)
        normals.append(-direction)
        offsets.append(-float(direction @ centre) + half)
    return np.asarray(normals), np.asarray(offsets)


def box_volume(extents: Sequence[float]) -> float:
    size = np.asarray(extents, dtype=np.float64).reshape(-1)
    return float(size[0] * size[1] * size[2])


def intersection_volume(
    rotation_a, translation_a, extents_a,
    rotation_b, translation_b, extents_b,
) -> float:
    """Exact volume of the intersection of two oriented boxes."""

    rot_a, cen_a, ext_a = _validate(rotation_a, translation_a, extents_a, "a")
    rot_b, cen_b, ext_b = _validate(rotation_b, translation_b, extents_b, "b")

    normals_a, offsets_a = _half_spaces(rot_a, cen_a, ext_a)
    normals_b, offsets_b = _half_spaces(rot_b, cen_b, ext_b)
    normals = np.vstack([normals_a, normals_b])
    offsets = np.concatenate([offsets_a, offsets_b])

    # 빠른 기각 — 분리축이 하나라도 있으면 교집합이 없다.
    corners_b = _corners(rot_b, cen_b, ext_b)
    corners_a = _corners(rot_a, cen_a, ext_a)
    for axis in range(3):
        for rot, cen, ext, other in ((rot_a, cen_a, ext_a, corners_b),
                                     (rot_b, cen_b, ext_b, corners_a)):
            direction = rot[:, axis]
            half = ext[axis] / 2.0
            projected = other @ direction
            middle = float(direction @ cen)
            if projected.min() >= middle + half - _FEASIBILITY_TOLERANCE:
                return 0.0
            if projected.max() <= middle - half + _FEASIBILITY_TOLERANCE:
                return 0.0

    vertices = []
    for i, j, k in combinations(range(len(normals)), 3):
        matrix = normals[[i, j, k]]
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < _MIN_TRIPLE_DETERMINANT:
            continue
        try:
            point = np.linalg.solve(matrix, offsets[[i, j, k]])
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        scale = max(1.0, float(np.abs(point).max()))
        if np.all(normals @ point <= offsets + _FEASIBILITY_TOLERANCE * scale):
            vertices.append(point)

    if len(vertices) < 4:
        return 0.0
    points = np.unique(np.round(np.asarray(vertices), 9), axis=0)
    if len(points) < 4:
        return 0.0
    if ConvexHull is None:  # pragma: no cover
        raise RuntimeError("scipy is required for exact oriented 3D IoU")
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0  # degenerate (coplanar) intersection has zero volume


def _corners(rotation: np.ndarray, centre: np.ndarray, extents: np.ndarray):
    half = extents / 2.0
    local = np.array([[sx, sy, sz]
                      for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                      for sz in (-1.0, 1.0)]) * half
    return (rotation @ local.T).T + centre


def oriented_iou_3d(
    rotation_a, translation_a, extents_a,
    rotation_b, translation_b, extents_b,
) -> float:
    """Intersection over union of two oriented 3D boxes, in [0, 1]."""

    inter = intersection_volume(rotation_a, translation_a, extents_a,
                                rotation_b, translation_b, extents_b)
    if inter <= 0.0:
        return 0.0
    union = box_volume(extents_a) + box_volume(extents_b) - inter
    if union <= 0.0:
        return 0.0
    return float(min(max(inter / union, 0.0), 1.0))
