"""Pallet 3D graph, semantic edge classes, and 180-degree symmetry.

PalletGraph-6D treats the pallet as a *dimensioned graph* rather than nine
independent points: the known (W, D, H) fix the length of every edge, so an
edge observed in the image constrains the pose even when its endpoints are
off-screen or unlocalisable.

Everything here is derived from the existing camera-facing 0123 convention via
``annotate_pnp.make_pallet_keypoints_3d``.  The corner order is never
re-defined locally — the edge classes are *derived* from the actual 3D
coordinates so a convention change cannot silently invert this module.

Local frame (see annotate_pnp.make_pallet_keypoints_3d_diagram):
    X = right  -> spans W, "width" edges
    Y = down   -> spans H, "vertical" edges  (OpenCV +y is down, so up = -Y)
    Z = forward-> spans D, "depth" edges
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

try:  # normal in-repo use
    import annotate_pnp as APNP
except ImportError:  # pragma: no cover - path-dependent
    APNP = None

N_CORNERS = 8
N_KEYPOINTS = 9
LINE_CLASSES = ("width", "depth", "vertical")
_AXIS_TO_CLASS = {0: "width", 1: "vertical", 2: "depth"}


def make_corners(width: float, depth: float, height: float) -> np.ndarray:
    """Nine keypoints (8 corners + centroid) in the canonical local frame."""
    if APNP is None:  # pragma: no cover
        raise RuntimeError("annotate_pnp is required for the canonical corner order")
    points = np.asarray(
        APNP.make_pallet_keypoints_3d(width, depth, height), dtype=np.float64
    )
    if points.shape != (N_KEYPOINTS, 3):
        raise RuntimeError(f"unexpected canonical keypoint shape {points.shape}")
    return points


def derive_edges(points: np.ndarray) -> dict[str, tuple[tuple[int, int], ...]]:
    """Group the 12 cuboid edges by which local axis they span.

    An edge is a corner pair differing along exactly one axis.  Deriving this
    from coordinates (instead of hard-coding index pairs) keeps the module
    correct if the corner convention is ever revised.
    """
    corners = np.asarray(points, dtype=np.float64)[:N_CORNERS]
    grouped: dict[str, list[tuple[int, int]]] = {name: [] for name in LINE_CLASSES}
    for i in range(N_CORNERS):
        for j in range(i + 1, N_CORNERS):
            delta = np.abs(corners[i] - corners[j])
            moving = int(np.count_nonzero(delta > 1e-9))
            if moving != 1:
                continue
            grouped[_AXIS_TO_CLASS[int(np.argmax(delta))]].append((i, j))
    for name, pairs in grouped.items():
        if len(pairs) != 4:
            raise RuntimeError(f"{name} class has {len(pairs)} edges, expected 4")
    return {name: tuple(pairs) for name, pairs in grouped.items()}


def edge_sets(
    width: float = 1.1, depth: float = 1.3, height: float = 0.12
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Semantic edge sets.  Dimensions only need to be pairwise distinct."""
    if len({round(width, 9), round(depth, 9), round(height, 9)}) != 3:
        raise ValueError(
            "edge classes are derived from distinct extents; pass distinct W/D/H"
        )
    return derive_edges(make_corners(width, depth, height))


def symmetry_rotation() -> np.ndarray:
    """180 degrees about the pallet up-axis (local Y)."""
    return np.array(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )


def symmetry_permutation(
    width: float = 1.1, depth: float = 1.3, height: float = 0.12
) -> tuple[int, ...]:
    """Index mapping induced by the 180-degree rotation.

    Derived numerically so it cannot drift from the corner convention.  For the
    canonical order this is ``(5, 4, 7, 6, 1, 0, 3, 2, 8)``, matching the
    permutation already used by the frozen ADD-S metric.
    """
    points = make_corners(width, depth, height)
    rotated = (symmetry_rotation() @ points.T).T
    permutation = []
    for index in range(N_KEYPOINTS):
        distances = np.linalg.norm(points - rotated[index], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > 1e-9:
            raise RuntimeError("180-degree symmetry is not exact for these dimensions")
        permutation.append(nearest)
    if len(set(permutation)) != N_KEYPOINTS:
        raise RuntimeError("symmetry permutation is not a bijection")
    return tuple(permutation)


def apply_symmetry(pose_R: np.ndarray, pose_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The equivalent pose obtained by rotating the object 180 degrees.

    ``t`` is unchanged: the rotation is about the object's own centre, which is
    the origin of the local frame.
    """
    return np.asarray(pose_R, dtype=np.float64) @ symmetry_rotation(), np.asarray(
        pose_t, dtype=np.float64
    ).reshape(3).copy()


def wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def wrap_half_pi(angle_rad: float) -> float:
    """Fold an angle into (-pi/2, pi/2]: 0 and pi become identical."""
    return float((angle_rad + math.pi / 2.0) % math.pi - math.pi / 2.0)


def yaw_from_rotation(rotation: np.ndarray) -> float:
    """Camera yaw of the pallet local forward axis, in radians.

    Same definition as the frozen diagnostic's ``yaw_deg`` (atan2(R[0,2], R[2,2])),
    kept here in radians so the modulo-pi arithmetic stays exact.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    return math.atan2(float(matrix[0, 2]), float(matrix[2, 2]))


def yaw_error_mod_pi_deg(predicted_R: np.ndarray, reference_R: np.ndarray) -> float:
    """Yaw error folded modulo 180 degrees.

    A pallet rotated by 180 degrees is the same physical placement for this
    task, so treating a 180-degree flip as a failure would misreport the metric.
    """
    difference = yaw_from_rotation(predicted_R) - yaw_from_rotation(reference_R)
    return abs(math.degrees(wrap_half_pi(difference)))


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    trace = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(trace)))


def rotation_error_sym_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation error under the 180-degree ambiguity."""
    flipped, _ = apply_symmetry(b, np.zeros(3))
    return min(rotation_error_deg(a, b), rotation_error_deg(a, flipped))


def corner_error_sym(
    predicted: dict[str, Any], reference: dict[str, Any],
    dims: tuple[float, float, float]
) -> Optional[float]:
    """Symmetry-aware mean 3D corner distance (metres)."""
    if predicted is None or reference is None or dims is None:
        return None
    points = make_corners(*dims)
    first = (np.asarray(predicted["R"]) @ points.T).T + np.asarray(
        predicted["t"]
    ).reshape(3)
    second = (np.asarray(reference["R"]) @ points.T).T + np.asarray(
        reference["t"]
    ).reshape(3)
    permutation = np.asarray(symmetry_permutation(*dims), dtype=np.int64)
    direct = float(np.mean(np.linalg.norm(first - second, axis=1)))
    flipped = float(np.mean(np.linalg.norm(first - second[permutation], axis=1)))
    return min(direct, flipped)


def object_diagonal(dims: tuple[float, float, float]) -> float:
    width, depth, height = (float(v) for v in dims)
    return float(math.sqrt(width**2 + depth**2 + height**2))


def project_points(
    points_3d: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
    intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pixels (N,2), depth (N,)).  Depth is returned so callers can
    reject points behind the camera instead of silently projecting them."""
    camera = (np.asarray(rotation, dtype=np.float64) @ np.asarray(
        points_3d, dtype=np.float64
    ).T).T + np.asarray(translation, dtype=np.float64).reshape(3)
    depth = camera[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    pixels = np.stack(
        [
            matrix[0, 0] * camera[:, 0] / safe + matrix[0, 2],
            matrix[1, 1] * camera[:, 1] / safe + matrix[1, 2],
        ],
        axis=1,
    )
    return pixels, depth


def clip_segment_to_image(
    start: np.ndarray, end: np.ndarray, width: int, height: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky clip of a 2D segment to [0,width-1] x [0,height-1]."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0 - 0.0), (dx, (width - 1.0) - x0),
        (-dy, y0 - 0.0), (dy, (height - 1.0) - y0),
    ):
        if abs(p) < 1e-12:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t1 <= t0:
        return None
    return (
        np.array([x0 + t0 * dx, y0 + t0 * dy]),
        np.array([x0 + t1 * dx, y0 + t1 * dy]),
    )


# Faces as corner-index quadruples, derived once from the canonical corners so
# the normals below stay tied to the real convention.
def face_definitions(
    width: float = 1.1, depth: float = 1.3, height: float = 0.12
) -> list[dict[str, Any]]:
    """Six faces with outward normals in the local frame."""
    corners = make_corners(width, depth, height)[:N_CORNERS]
    faces: list[dict[str, Any]] = []
    for axis, sign in ((0, -1), (0, +1), (1, -1), (1, +1), (2, -1), (2, +1)):
        extreme = sign * np.max(sign * corners[:, axis])
        members = tuple(
            int(i) for i in range(N_CORNERS)
            if abs(corners[i, axis] - extreme) < 1e-9
        )
        if len(members) != 4:
            raise RuntimeError(f"face on axis {axis} sign {sign} has {len(members)}")
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = float(sign)
        faces.append({"axis": axis, "sign": sign, "corners": members, "normal": normal})
    return faces


def edge_adjacent_faces(
    dims: tuple[float, float, float] = (1.1, 1.3, 0.12)
) -> dict[tuple[int, int], list[np.ndarray]]:
    """For each cuboid edge, the outward normals of its two adjacent faces."""
    faces = face_definitions(*dims)
    edges = derive_edges(make_corners(*dims))
    mapping: dict[tuple[int, int], list[np.ndarray]] = {}
    for pairs in edges.values():
        for i, j in pairs:
            adjacent = [
                face["normal"] for face in faces
                if i in face["corners"] and j in face["corners"]
            ]
            if len(adjacent) != 2:
                raise RuntimeError(f"edge {(i, j)} has {len(adjacent)} adjacent faces")
            mapping[(i, j)] = adjacent
    return mapping


def visible_edges(
    rotation: np.ndarray, translation: np.ndarray,
    dims: tuple[float, float, float]
) -> dict[tuple[int, int], bool]:
    """Self-visibility: an edge counts as visible when at least one adjacent
    face is camera-facing.

    This is geometric self-occlusion only.  It says nothing about occlusion by
    other objects, which is why the loader additionally intersects with the
    real mask when one exists.
    """
    corners = make_corners(*dims)[:N_CORNERS]
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    result: dict[tuple[int, int], bool] = {}
    for edge, normals in edge_adjacent_faces(dims).items():
        midpoint = 0.5 * (corners[edge[0]] + corners[edge[1]])
        view = rotation @ midpoint + translation
        norm = float(np.linalg.norm(view))
        if norm < 1e-9:
            result[edge] = False
            continue
        view = view / norm
        result[edge] = any(
            float(np.dot(rotation @ normal, view)) < 0.0 for normal in normals
        )
    return result


def projected_edges(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], image_size: tuple[int, int],
    visibility_aware: bool = False, min_length_px: float = 4.0
) -> list[dict[str, Any]]:
    """Project every semantic edge, clip to the image, and label its class.

    Returns one record per usable edge.  ``visibility_aware=False`` is the
    amodal variant (an optimistic upper bound that can include self-occluded
    edges) and must never be reported as "visible".
    """
    width_px, height_px = int(image_size[0]), int(image_size[1])
    corners = make_corners(*dims)[:N_CORNERS]
    pixels, depth = project_points(corners, rotation, translation, intrinsics)
    visibility = (
        visible_edges(rotation, translation, dims) if visibility_aware else None
    )
    records: list[dict[str, Any]] = []
    for line_class, pairs in derive_edges(make_corners(*dims)).items():
        for i, j in pairs:
            if depth[i] <= 1e-6 or depth[j] <= 1e-6:
                continue  # behind the camera; projection is meaningless
            if visibility is not None and not visibility[(i, j)]:
                continue
            clipped = clip_segment_to_image(pixels[i], pixels[j], width_px, height_px)
            if clipped is None:
                continue
            start, end = clipped
            length = float(np.linalg.norm(end - start))
            if length < min_length_px:
                continue
            records.append(
                {
                    "edge": (int(i), int(j)),
                    "line_class": line_class,
                    "start": start,
                    "end": end,
                    "length_px": length,
                    "self_visible": None if visibility is None else visibility[(i, j)],
                }
            )
    return records


def sample_along(
    start: np.ndarray, end: np.ndarray, min_samples: int = 8,
    pixels_per_sample: float = 4.0
) -> np.ndarray:
    """Evenly spaced samples on a segment, density tied to its pixel length."""
    length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    count = max(int(min_samples), int(math.ceil(length / max(pixels_per_sample, 1e-6))))
    ratios = np.linspace(0.0, 1.0, count)
    return np.asarray(start)[None, :] + ratios[:, None] * (
        np.asarray(end) - np.asarray(start)
    )[None, :]
