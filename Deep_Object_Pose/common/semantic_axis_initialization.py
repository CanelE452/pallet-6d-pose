"""SAI — Semantic-Axis Initialization.

Given three semantic raster line maps (width / depth / vertical), the camera
intrinsics, and the known W,D,H, produce **full rotation candidates without any
GT pose and without any point-PnP initialisation**.

Why this exists
---------------
G0/G1 showed that oracle semantic lines carry enough information to recover a
pose *when GT roll/pitch/translation are supplied*.  That is an upper bound on
information, not a capability: the solver was handed most of the answer.  SAI
removes every GT pose component from the solver, so what remains is the
question that actually matters — can the geometry layer stand on the lines
alone?

Method
------
A pallet is a box, so its edges fall into three mutually orthogonal object
axes.  Every image line of a given semantic class is the projection of a 3D
line parallel to that object axis, therefore all lines of one class meet at a
single vanishing point, and the camera-frame direction of that axis is
``normalize(K^-1 v)``.  Two axes are enough (the third is their cross product),
which is why this is preferred over a 5-D brute-force search.

No representation from the failed tracks appears here: nothing regresses a
point, an offset, or an endpoint, and no line is ever converted into a
keypoint.  The only inputs are raster support masks.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Optional

import numpy as np

try:
    import pallet_graph_geometry as PG
except ImportError:  # pragma: no cover
    from . import pallet_graph_geometry as PG  # type: ignore

EPS = 1e-12
# --- fixed configuration (frozen before looking at any N87 metric) ----------
MORPH_CLOSE_PX = 2
MIN_COMPONENT_LENGTH_PX = 8.0
MIN_COMPONENT_LENGTH_DIAG_FRAC = 0.015
MIN_COMPONENT_PIXELS = 6
MAX_FIT_RMS_PX = 3.0
DEGENERATE_AXIS_DOT = 0.95
K_ROTATION = 8
AXIS_REFINE_ITERATIONS = 10
LAMBDA_ORTHOGONALITY = 1.0
LAMBDA_FIT = 0.10
SINGULAR_RATIO_MIN = 1.5   # VP is only trusted when the null space is distinct

# Object-frame axis of each semantic class, derived from the shared convention
# rather than assumed: width spans local X, vertical spans local Y, depth Z.
CLASS_AXIS = {
    "width": np.array([1.0, 0.0, 0.0]),
    "vertical": np.array([0.0, 1.0, 0.0]),
    "depth": np.array([0.0, 0.0, 1.0]),
}


def verify_class_axes(dims: tuple[float, float, float]) -> dict[str, np.ndarray]:
    """Confirm CLASS_AXIS against the live corner convention.

    If the corner order ever changes, this raises instead of silently
    initialising a rotation with permuted axes.
    """
    corners = PG.make_corners(*dims)[: PG.N_CORNERS]
    edges = PG.derive_edges(PG.make_corners(*dims))
    for name, pairs in edges.items():
        i, j = pairs[0]
        direction = corners[j] - corners[i]
        direction = direction / max(float(np.linalg.norm(direction)), EPS)
        expected = CLASS_AXIS[name]
        if abs(abs(float(np.dot(direction, expected))) - 1.0) > 1e-9:
            raise RuntimeError(
                f"class '{name}' axis {direction} does not match {expected}; "
                "the corner convention changed"
            )
    return dict(CLASS_AXIS)


# ============================================================================
# Phase C — semantic line component extraction
# ============================================================================
def fit_line_weighted_tls(
    points: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    """Weighted total-least-squares 2-D line fit.

    Returns ``(l, rms)`` with ``l = (a, b, c)`` normalised so ``a^2+b^2 = 1``.
    TLS (not y-on-x regression) is required because a pallet edge can be
    arbitrarily oriented, including exactly vertical.
    """
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 2:
        raise ValueError("need at least two support pixels to fit a line")
    w = (np.ones(xy.shape[0]) if weights is None
         else np.asarray(weights, dtype=np.float64).reshape(-1))
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= EPS:
        w = np.ones(xy.shape[0])
        total = float(w.sum())
    centroid = (w[:, None] * xy).sum(axis=0) / total
    centred = xy - centroid
    scatter = (centred * w[:, None]).T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    normal = normal / max(float(np.linalg.norm(normal)), EPS)
    offset = -float(normal @ centroid)
    residual = centred @ normal
    rms = float(np.sqrt(np.average(residual**2, weights=w)))
    return np.array([normal[0], normal[1], offset], dtype=np.float64), rms


def extract_line_components(
    support: dict[str, np.ndarray], image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """Connected components of each semantic support mask, fitted to lines."""
    import cv2

    diagonal = math.hypot(float(image_size[0]), float(image_size[1]))
    min_length = max(MIN_COMPONENT_LENGTH_PX,
                     MIN_COMPONENT_LENGTH_DIAG_FRAC * diagonal)
    kernel = np.ones((MORPH_CLOSE_PX + 1, MORPH_CLOSE_PX + 1), np.uint8)
    components: list[dict[str, Any]] = []
    for line_class in PG.LINE_CLASSES:
        mask = np.asarray(support.get(line_class), dtype=np.uint8)
        if mask.size == 0 or int(mask.sum()) == 0:
            continue
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels = cv2.connectedComponents(closed, connectivity=8)
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            if xs.size < MIN_COMPONENT_PIXELS:
                continue
            xy = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
            extent = float(
                np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
            )
            if extent < min_length:
                continue
            line, rms = fit_line_weighted_tls(xy)
            if not np.isfinite(line).all() or rms > MAX_FIT_RMS_PX:
                continue
            components.append(
                {
                    "semantic_class": line_class,
                    "line": line,
                    "support_length": extent,
                    "support_mass": float(xs.size),
                    "fit_rms": rms,
                    "bbox": (float(xs.min()), float(ys.min()),
                             float(xs.max()), float(ys.max())),
                    "centroid": (float(xs.mean()), float(ys.mean())),
                    "pixel_count": int(xs.size),
                }
            )
    return components


# ============================================================================
# Phase D — vanishing points and full rotation
# ============================================================================
def component_weight(component: dict[str, Any]) -> float:
    return float(
        component["support_length"]
        * (component["support_mass"] / max(component["support_length"], 1.0))
        / (component["fit_rms"] + 1e-3)
    )


def vanishing_point(
    components: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Least-squares intersection of >=2 lines, in homogeneous form.

    A finite and an infinite vanishing point are both representable; the vector
    is simply not normalised to w=1.
    """
    if len(components) < 2:
        return None
    rows = np.stack([
        math.sqrt(max(component_weight(c), 0.0)) * c["line"] for c in components
    ])
    _, singular, vt = np.linalg.svd(rows)
    vector = vt[-1]
    norm = float(np.linalg.norm(vector))
    if norm < EPS or not np.isfinite(vector).all():
        return None
    vector = vector / norm
    ratio = (float(singular[-2] / max(singular[-1], EPS))
             if singular.size >= 2 else float("inf"))
    residual = float(np.sqrt(np.mean((rows @ vector) ** 2)))
    return {
        "vanishing_point": vector,
        "singular_values": singular.tolist(),
        "singular_ratio": ratio,
        "residual": residual,
        "n_components": len(components),
        "well_conditioned": bool(ratio >= SINGULAR_RATIO_MIN),
    }


def axis_from_vanishing_point(
    vector: np.ndarray, intrinsics: np.ndarray
) -> Optional[np.ndarray]:
    direction = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)) @ np.asarray(
        vector, dtype=np.float64
    )
    norm = float(np.linalg.norm(direction))
    if norm < EPS or not np.isfinite(direction).all():
        return None
    return direction / norm


def line_plane_residual(
    rotation: np.ndarray, components: list[dict[str, Any]], intrinsics: np.ndarray
) -> float:
    """sum_j w_j (l_j^T K R a_c)^2 — independent of translation."""
    if not components:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    KR = np.asarray(intrinsics, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64)
    for component in components:
        axis = CLASS_AXIS[component["semantic_class"]]
        value = float(component["line"] @ (KR @ axis))
        weight = component_weight(component)
        total += weight * value * value
        weight_sum += weight
    return total / max(weight_sum, EPS)


def refine_rotation(
    rotation: np.ndarray, components: list[dict[str, Any]],
    intrinsics: np.ndarray, iterations: int = AXIS_REFINE_ITERATIONS
) -> np.ndarray:
    """Coordinate descent on SO(3) for the line-plane objective.

    Deliberately gradient-free and tiny: the objective is cheap, and a hand
    derived Jacobian that silently disagrees with the reported energy is
    exactly the failure mode this project already hit once.
    """
    from dimension_guided_graph_pose import so3_exp  # local import: same package

    current = np.asarray(rotation, dtype=np.float64).copy()
    best = line_plane_residual(current, components, intrinsics)
    step = 0.05  # radians
    for _ in range(int(iterations)):
        improved = False
        for axis in range(3):
            for sign in (+1.0, -1.0):
                omega = np.zeros(3)
                omega[axis] = sign * step
                candidate = so3_exp(omega) @ current
                value = line_plane_residual(candidate, components, intrinsics)
                if value < best - 1e-15:
                    current, best, improved = candidate, value, True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return current


def orthogonalize(matrix: np.ndarray) -> Optional[np.ndarray]:
    """Nearest proper rotation (det = +1) via SVD."""
    if not np.isfinite(matrix).all():
        return None
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ correction @ vt
    if not np.isfinite(rotation).all():
        return None
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        return None
    return rotation


def axis_observability(
    axes: dict[str, Optional[np.ndarray]]
) -> dict[str, Any]:
    """How many semantic axes are usable, and whether any pair is degenerate."""
    usable = {name: v for name, v in axes.items() if v is not None}
    degenerate = []
    for a, b in itertools.combinations(sorted(usable), 2):
        if abs(float(np.dot(usable[a], usable[b]))) > DEGENERATE_AXIS_DOT:
            degenerate.append((a, b))
    return {
        "observable_axes": sorted(usable),
        "n_observable": len(usable),
        "degenerate_pairs": degenerate,
        "usable": len(usable) >= 2 and not degenerate,
        "reason": (
            None if (len(usable) >= 2 and not degenerate)
            else ("fewer_than_2_axes" if len(usable) < 2 else "degenerate_axis_pair")
        ),
    }


def rotation_candidates(
    axes: dict[str, Optional[np.ndarray]],
    components_by_class: dict[str, list[dict[str, Any]]],
    intrinsics: np.ndarray,
    k: int = K_ROTATION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate sign/completion candidates and rank them GT-free.

    A fitted 2-D line has no direction, so each recovered axis is known only up
    to sign; the candidates below enumerate those signs.  Width and depth are
    never swapped — that would discard the semantic class, which is the whole
    point of the representation.
    """
    observability = axis_observability(axes)
    if not observability["usable"]:
        return [], observability

    everything = [c for group in components_by_class.values() for c in group]
    seen: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []
    names = ("width", "vertical", "depth")
    present = [n for n in names if axes.get(n) is not None]

    for signs in itertools.product((+1.0, -1.0), repeat=len(present)):
        directions: dict[str, np.ndarray] = {
            name: sign * axes[name] for name, sign in zip(present, signs)
        }
        if len(present) == 2:
            missing = [n for n in names if n not in directions][0]
            first, second = present
            cross = np.cross(directions[first], directions[second])
            norm = float(np.linalg.norm(cross))
            if norm < EPS:
                continue
            cross = cross / norm
            # orient the completed axis so the frame stays right-handed
            order = {"width": 0, "vertical": 1, "depth": 2}
            parity = 1.0
            triple = sorted(names, key=lambda n: order[n])
            if triple.index(missing) == 1:
                parity = -1.0
            directions[missing] = parity * cross
        matrix = np.stack(
            [directions["width"], directions["vertical"], directions["depth"]], axis=1
        )
        rotation = orthogonalize(matrix)
        if rotation is None:
            continue
        rotation = refine_rotation(rotation, everything, intrinsics)
        reorthogonalized = orthogonalize(rotation)
        if reorthogonalized is not None:
            rotation = reorthogonalized
        # collapse yaw+180 equivalents: same physical placement for a pallet
        duplicate = False
        for kept in seen:
            if PG.rotation_error_sym_deg(rotation, kept) < 1.0:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append(rotation)
        residual = line_plane_residual(rotation, everything, intrinsics)
        orthogonality = float(
            np.linalg.norm(rotation.T @ rotation - np.eye(3))
        )
        fit = float(np.mean([c["fit_rms"] for c in everything])) if everything else 0.0
        candidates.append(
            {
                "R": rotation,
                "signs": signs,
                "E_axis": residual,
                "score": residual + LAMBDA_ORTHOGONALITY * orthogonality
                + LAMBDA_FIT * fit,
                "orthogonality": orthogonality,
                "fit_rms_mean": fit,
                "completed_axis": (
                    None if len(present) == 3
                    else [n for n in names if n not in present][0]
                ),
            }
        )
    candidates.sort(key=lambda c: c["score"])
    return candidates[: int(k)], observability


def semantic_axis_initialization(
    support: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    dims: tuple[float, float, float],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Full entry point.  Note the signature: no GT rotation, no GT translation,
    no point-PnP pose.  Only raster support, K, known dimensions, image size."""
    verify_class_axes(dims)
    components = extract_line_components(support, image_size)
    by_class: dict[str, list[dict[str, Any]]] = {n: [] for n in PG.LINE_CLASSES}
    for component in components:
        by_class[component["semantic_class"]].append(component)

    vanishing: dict[str, Any] = {}
    axes: dict[str, Optional[np.ndarray]] = {}
    for line_class in PG.LINE_CLASSES:
        estimate = vanishing_point(by_class[line_class])
        vanishing[line_class] = estimate
        axes[line_class] = (
            None if estimate is None
            else axis_from_vanishing_point(estimate["vanishing_point"], intrinsics)
        )
    candidates, observability = rotation_candidates(axes, by_class, intrinsics)
    return {
        "components": components,
        "components_by_class": {k: len(v) for k, v in by_class.items()},
        "vanishing": vanishing,
        "axes": axes,
        "observability": observability,
        "candidates": candidates,
        "n_candidates": len(candidates),
    }
