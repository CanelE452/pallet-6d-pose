"""DGP — Dimension-Guided Graph Pose.

One full SE(3) optimisation that consumes two kinds of image evidence:

* point residuals: reprojection of the 9 keypoints against the decoded 2D
  points (exactly the evidence the current pipeline already uses), and
* line residuals: each 3D edge is projected with the *known* (W, D, H), clipped
  to the image, sampled, and scored against a semantic line probability map.

The line term is deliberately NOT converted back into keypoints.  The failed
vector/offset/voting tracks all tried to reconstruct points from another
representation; here the line map is read directly as a 2D residual, so a line
fragment whose endpoints are off-screen still constrains the pose.

Rotation is updated in the Lie algebra (never by adding to Euler angles), and
every iteration is guarded for depth, conditioning, finiteness, and step size.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

try:
    import pallet_graph_geometry as PG
except ImportError:  # pragma: no cover
    from . import pallet_graph_geometry as PG  # type: ignore

EPS = 1e-9
DEFAULT_HUBER_POINT_PX = 8.0
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_DAMPING = 1e-3
MAX_CONDITION_NUMBER = 1e10
MAX_TRANSLATION_STEP_M = 0.5
MAX_ROTATION_STEP_RAD = 0.5
# Per-axis trust radii: the largest raw step an undamped iteration may take.
TRUST_ROTATION_RAD = 0.05   # ~2.9 degrees
TRUST_TRANSLATION_M = 0.05


def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(v) for v in np.asarray(vector).reshape(3))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3) + hat(omega)
    axis = omega / angle
    skew = hat(axis)
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def huber_weight(residual_norm: float, delta: float) -> float:
    """IRLS weight for the Huber loss."""
    if residual_norm <= delta or residual_norm < EPS:
        return 1.0
    return float(delta / residual_norm)


def bilinear_sample(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample a single-channel map at float pixel coordinates.

    Returns ``(values, inside)``; out-of-bounds samples are reported rather
    than clamped, so callers can exclude them instead of inventing evidence at
    the border.
    """
    grid = np.asarray(image, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"expected a 2-D map, got {grid.shape}")
    height, width = grid.shape
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = coords[:, 0], coords[:, 1]
    inside = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    xc = np.clip(x, 0.0, width - 1.0)
    yc = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    ax, ay = xc - x0, yc - y0
    values = (
        grid[y0, x0] * (1 - ax) * (1 - ay)
        + grid[y0, x1] * ax * (1 - ay)
        + grid[y1, x0] * (1 - ax) * ay
        + grid[y1, x1] * ax * ay
    )
    return values, inside


class LineEvidence:
    """Semantic line probability maps in image pixel coordinates.

    ``maps`` is ``{"width": HxW, "depth": HxW, "vertical": HxW}`` with values in
    [0,1].  ``class_agnostic`` collapses all three to one map, which is how the
    generic-edge baseline is expressed without changing the solver.
    """

    def __init__(
        self, maps: dict[str, np.ndarray], image_size: tuple[int, int],
        class_agnostic: bool = False
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.class_agnostic = bool(class_agnostic)
        if class_agnostic:
            stacked = np.max(
                np.stack([np.asarray(m, dtype=np.float64) for m in maps.values()]),
                axis=0,
            )
            self.maps = {name: stacked for name in PG.LINE_CLASSES}
        else:
            missing = set(PG.LINE_CLASSES) - set(maps)
            if missing:
                raise ValueError(f"missing line classes: {sorted(missing)}")
            self.maps = {
                name: np.asarray(maps[name], dtype=np.float64)
                for name in PG.LINE_CLASSES
            }
        for name, grid in self.maps.items():
            if grid.ndim != 2:
                raise ValueError(f"line map '{name}' must be 2-D, got {grid.shape}")

    def scale_to_map(self, pixels: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Image pixels -> map coordinates (maps may be lower resolution)."""
        height, width = grid.shape
        sx = (width - 1.0) / max(self.image_size[0] - 1.0, 1.0)
        sy = (height - 1.0) / max(self.image_size[1] - 1.0, 1.0)
        scaled = np.asarray(pixels, dtype=np.float64).copy()
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy
        return scaled


def line_energy(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], evidence: LineEvidence,
    visibility_aware: bool = True, min_length_px: float = 6.0,
    epsilon: float = 1e-4
) -> tuple[float, dict[str, Any]]:
    """Mean negative log-probability of the projected edges under the maps.

    Lower is better; a pose whose projected edges land on high line probability
    scores near zero.
    """
    records = PG.projected_edges(
        rotation, translation, intrinsics, dims, evidence.image_size,
        visibility_aware=visibility_aware, min_length_px=min_length_px,
    )
    per_edge: list[float] = []
    n_samples = 0
    for record in records:
        grid = evidence.maps[record["line_class"]]
        samples = PG.sample_along(record["start"], record["end"])
        values, inside = bilinear_sample(grid, evidence.scale_to_map(samples, grid))
        if not bool(inside.any()):
            continue
        probability = np.clip(values[inside], 0.0, 1.0)
        per_edge.append(float(np.mean(-np.log(probability + epsilon))))
        n_samples += int(inside.sum())
    if not per_edge:
        return 0.0, {"n_edges": 0, "n_samples": 0, "available": False}
    return float(np.mean(per_edge)), {
        "n_edges": len(per_edge), "n_samples": n_samples, "available": True,
    }


def point_energy(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], observations: np.ndarray,
    valid: np.ndarray, huber_delta: float = DEFAULT_HUBER_POINT_PX
) -> tuple[float, dict[str, Any]]:
    points = PG.make_corners(*dims)
    projected, depth = PG.project_points(points, rotation, translation, intrinsics)
    usable = np.asarray(valid, dtype=bool) & (depth > 1e-6)
    if not bool(usable.any()):
        return 0.0, {"n_points": 0, "positive_depth_fraction": 0.0}
    residual = projected[usable] - np.asarray(observations, dtype=np.float64)[usable]
    norms = np.linalg.norm(residual, axis=1)
    losses = np.where(
        norms <= huber_delta,
        0.5 * norms**2,
        huber_delta * (norms - 0.5 * huber_delta),
    )
    return float(np.mean(losses)), {
        "n_points": int(usable.sum()),
        "positive_depth_fraction": float(np.mean(depth > 1e-6)),
        "residual_median_px": float(np.median(norms)),
    }


def ground_energy(
    rotation: np.ndarray, ground_normal: Optional[np.ndarray],
    pallet_up_local: np.ndarray = np.array([0.0, -1.0, 0.0]),
    huber_delta: float = 0.2
) -> float:
    """Zero unless a trustworthy ground normal is supplied for this frame."""
    if ground_normal is None:
        return 0.0
    predicted = np.asarray(rotation, dtype=np.float64) @ pallet_up_local
    difference = float(
        np.linalg.norm(predicted - np.asarray(ground_normal, dtype=np.float64).reshape(3))
    )
    if difference <= huber_delta:
        return 0.5 * difference**2
    return huber_delta * (difference - 0.5 * huber_delta)


def total_energy(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], observations: np.ndarray, valid: np.ndarray,
    evidence: Optional[LineEvidence], lambda_point: float, lambda_line: float,
    lambda_ground: float, ground_normal: Optional[np.ndarray],
    visibility_aware: bool, huber_delta: float
) -> tuple[float, dict[str, Any]]:
    e_point, point_info = point_energy(
        rotation, translation, intrinsics, dims, observations, valid, huber_delta
    )
    if evidence is None or lambda_line == 0.0:
        e_line, line_info = 0.0, {"n_edges": 0, "n_samples": 0, "available": False}
    else:
        e_line, line_info = line_energy(
            rotation, translation, intrinsics, dims, evidence,
            visibility_aware=visibility_aware,
        )
    e_ground = ground_energy(rotation, ground_normal)
    total = (
        lambda_point * e_point + lambda_line * e_line + lambda_ground * e_ground
    )
    return total, {
        "E_point": e_point, "E_line": e_line, "E_ground": e_ground,
        "E_total": total, **point_info, **{f"line_{k}": v for k, v in line_info.items()},
    }


def solve(
    initial_R: np.ndarray, initial_t: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], observations: np.ndarray, valid: np.ndarray,
    evidence: Optional[LineEvidence] = None, lambda_point: float = 1.0,
    lambda_line: float = 0.0, lambda_ground: float = 0.0,
    ground_normal: Optional[np.ndarray] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    huber_delta: float = DEFAULT_HUBER_POINT_PX,
    visibility_aware: bool = True, damping: float = DEFAULT_DAMPING,
) -> dict[str, Any]:
    """Damped Gauss-Newton on SE(3) with numeric Jacobians.

    The line term has no closed-form Jacobian (it reads a learned raster), so
    the whole objective is differentiated numerically in the 6-D tangent space.
    With <=6 iterations and 6 parameters this is 7 energy evaluations per
    iteration, which is affordable and avoids a hand-derived gradient that
    could silently disagree with the energy actually being reported.
    """
    rotation = np.asarray(initial_R, dtype=np.float64).copy()
    translation = np.asarray(initial_t, dtype=np.float64).reshape(3).copy()
    observations = np.asarray(observations, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    n_valid = int(valid.sum())
    diagnostics: dict[str, Any] = {
        "n_valid_points": n_valid,
        "iterations": 0,
        "fallback": False,
        "fallback_reason": None,
        "converged": False,
        "condition_number": None,
        "positive_depth_ok": True,
        "energy_history": [],
    }
    if n_valid < 4:
        diagnostics.update(
            {"fallback": True, "fallback_reason": "fewer_than_4_correspondences"}
        )
        return {"R": rotation, "t": translation, **diagnostics}

    def energy_at(delta: np.ndarray) -> tuple[float, dict[str, Any]]:
        new_R = so3_exp(delta[:3]) @ rotation
        new_t = translation + delta[3:]
        return total_energy(
            new_R, new_t, intrinsics, dims, observations, valid, evidence,
            lambda_point, lambda_line, lambda_ground, ground_normal,
            visibility_aware, huber_delta,
        )

    current, info = energy_at(np.zeros(6))
    diagnostics["energy_history"].append(current)
    diagnostics["initial_info"] = info
    if not math.isfinite(current):
        diagnostics.update({"fallback": True, "fallback_reason": "non_finite_initial"})
        return {"R": rotation, "t": translation, **diagnostics}

    # Probe sizes are in the same units as the trust radii below.  A too-small
    # probe makes the line term (a sampled raster) look like numerical noise.
    step_rot, step_trans = 5e-3, 5e-3
    trust = np.array(
        [TRUST_ROTATION_RAD] * 3 + [TRUST_TRANSLATION_M] * 3, dtype=np.float64
    )
    lam = float(damping)
    for iteration in range(int(max_iterations)):
        gradient = np.zeros(6)
        hessian_diag = np.zeros(6)
        ok = True
        for axis in range(6):
            size = step_rot if axis < 3 else step_trans
            probe = np.zeros(6)
            probe[axis] = size
            forward, _ = energy_at(probe)
            probe[axis] = -size
            backward, _ = energy_at(probe)
            if not (math.isfinite(forward) and math.isfinite(backward)):
                ok = False
                break
            gradient[axis] = (forward - backward) / (2.0 * size)
            hessian_diag[axis] = (forward - 2.0 * current + backward) / (size**2)
        if not ok:
            diagnostics["fallback_reason"] = "non_finite_derivative"
            break

        # Trust-region floor.  In the Huber linear regime (large residuals) the
        # finite-difference curvature collapses towards zero, which would send
        # the Gauss-Newton step to the clipping limit and get every candidate
        # rejected.  Flooring the curvature at |grad|/trust bounds the raw step
        # by the trust radius per axis, which is the Levenberg interpretation of
        # the damping term.
        floor = np.abs(gradient) / np.maximum(trust, 1e-12)
        hessian_diag = np.maximum(np.maximum(hessian_diag, floor), 1e-8)

        positive = hessian_diag[hessian_diag > 0]
        condition = float(np.max(positive) / max(np.min(positive), 1e-12))
        diagnostics["condition_number"] = condition
        if condition > MAX_CONDITION_NUMBER:
            diagnostics["fallback_reason"] = "ill_conditioned"
            break

        applied = False
        for _ in range(6):  # backtracking on the damping factor
            step = -gradient / (hessian_diag * (1.0 + lam))
            rotation_step = float(np.linalg.norm(step[:3]))
            translation_step = float(np.linalg.norm(step[3:]))
            if rotation_step > MAX_ROTATION_STEP_RAD:
                step[:3] *= MAX_ROTATION_STEP_RAD / rotation_step
            if translation_step > MAX_TRANSLATION_STEP_M:
                step[3:] *= MAX_TRANSLATION_STEP_M / translation_step
            candidate, candidate_info = energy_at(step)
            if math.isfinite(candidate) and candidate < current:
                rotation = so3_exp(step[:3]) @ rotation
                translation = translation + step[3:]
                current, info = candidate, candidate_info
                lam = max(lam * 0.5, 1e-6)
                applied = True
                break
            lam = lam * 4.0 + 1.0
        diagnostics["iterations"] = iteration + 1
        diagnostics["energy_history"].append(current)
        if not applied:
            diagnostics["converged"] = True
            break

    _, final_info = total_energy(
        rotation, translation, intrinsics, dims, observations, valid, evidence,
        lambda_point, lambda_line, lambda_ground, ground_normal,
        visibility_aware, huber_delta,
    )
    diagnostics["final_info"] = final_info
    diagnostics["positive_depth_ok"] = (
        float(final_info.get("positive_depth_fraction", 0.0)) > 0.5
    )
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
        diagnostics.update({"fallback": True, "fallback_reason": "non_finite_pose"})
        return {
            "R": np.asarray(initial_R, dtype=np.float64),
            "t": np.asarray(initial_t, dtype=np.float64).reshape(3),
            **diagnostics,
        }
    return {"R": rotation, "t": translation, **diagnostics}


def solve_with_symmetry(
    initial_R: np.ndarray, initial_t: np.ndarray, **kwargs: Any
) -> dict[str, Any]:
    """Refine both 180-degree hypotheses and keep the lower-energy one.

    The pallet is 180-degree ambiguous, so seeding a single hypothesis lets the
    optimiser sit in whichever basin PnP happened to pick.
    """
    candidates = []
    for label, (rotation, translation) in (
        ("as_given", (initial_R, initial_t)),
        ("rot180", PG.apply_symmetry(initial_R, initial_t)),
    ):
        result = solve(rotation, translation, **kwargs)
        result["hypothesis"] = label
        result["final_energy"] = float(
            result.get("final_info", {}).get("E_total", float("inf"))
        )
        candidates.append(result)
    usable = [c for c in candidates if not c["fallback"]]
    pool = usable or candidates
    best = min(pool, key=lambda c: c["final_energy"])
    best["hypothesis_energies"] = {
        c["hypothesis"]: c["final_energy"] for c in candidates
    }
    return best


# ============================================================================
# Continuous semantic-line energy (DGP v2)
# ============================================================================
# The v1 energy was discontinuous in pose for two reasons that a global search
# cannot tolerate:
#   (a) visibility was re-decided per candidate, so the *set* of contributing
#       edges changed as the pose moved, and
#   (b) the energy averaged per edge, so losing one edge rescaled the whole
#       objective.
# Together they made the yaw landscape a staircase (verified empirically), and
# a staircase has no usable gradient and no meaningful global minimum.
#
# v2 fixes the edge set once per frame, normalises by SAMPLE weight rather than
# by edge count, and reads a smooth distance field instead of a 1-pixel raster.
COARSE_SIGMA_FRACTION = 0.020
MID_SIGMA_FRACTION = 0.010
FINE_SIGMA_FRACTION = 0.005
MIN_EDGE_LENGTH_PX = 2.0


def image_diagonal(image_size) -> float:
    width, height = float(image_size[0]), float(image_size[1])
    return math.sqrt(width * width + height * height)


def sigma_schedule(image_size) -> dict[str, float]:
    diagonal = image_diagonal(image_size)
    return {
        "coarse": COARSE_SIGMA_FRACTION * diagonal,
        "mid": MID_SIGMA_FRACTION * diagonal,
        "fine": FINE_SIGMA_FRACTION * diagonal,
    }


class ContinuousLineField:
    """Distance-field semantic line evidence at three fixed scales.

    ``support`` holds, per class, the observed line sample coordinates in image
    pixels; it is what the REVERSE term matches against, and it is computed once
    per frame so no candidate pose can change it.
    """

    def __init__(
        self,
        distance: dict[str, np.ndarray],
        support: dict[str, np.ndarray],
        image_size,
        scale: int = 1,
        class_agnostic: bool = False,
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.scale = int(scale)
        self.class_agnostic = bool(class_agnostic)
        self.sigmas = sigma_schedule(self.image_size)
        missing = set(PG.LINE_CLASSES) - set(distance)
        if missing:
            raise ValueError(f"missing distance fields: {sorted(missing)}")
        self.distance = {
            name: np.asarray(distance[name], dtype=np.float32)
            for name in PG.LINE_CLASSES
        }
        self.support = {
            name: np.asarray(support.get(name, np.zeros((0, 2))), dtype=np.float64)
            for name in PG.LINE_CLASSES
        }

    def sample_distance(self, name: str, pixels: np.ndarray):
        """Distance (in IMAGE pixels) at the given image-pixel coordinates."""
        grid = self.distance[name]
        coords = np.asarray(pixels, dtype=np.float64).reshape(-1, 2) / self.scale
        values, inside = bilinear_sample(grid, coords)
        return values * self.scale, inside

    def total_support(self) -> int:
        return int(sum(v.shape[0] for v in self.support.values()))


def _rho(distance: np.ndarray, sigma: float) -> np.ndarray:
    """Bounded, smooth line residual in [0,1]; 0 exactly on the line."""
    return 1.0 - np.exp(-(distance**2) / (2.0 * sigma * sigma))


def fixed_edge_set(
    rotation, translation, dims, mode: str = "amodal"
) -> list[tuple[tuple[int, int], str]]:
    """Edges chosen ONCE per frame; never re-decided per candidate pose.

    ``mode='amodal'``  -> all 12 edges.
    ``mode='visible'`` -> edges self-visible at the given (reference) pose.
    """
    edges = PG.derive_edges(PG.make_corners(*dims))
    if mode == "amodal":
        return [(pair, name) for name, pairs in edges.items() for pair in pairs]
    if mode != "visible":
        raise ValueError(f"unknown edge-set mode: {mode}")
    visibility = PG.visible_edges(rotation, translation, dims)
    return [
        (pair, name)
        for name, pairs in edges.items()
        for pair in pairs
        if visibility[pair]
    ]


def continuous_line_energy(
    rotation,
    translation,
    intrinsics,
    dims,
    field: ContinuousLineField,
    edge_set: list[tuple[tuple[int, int], str]],
    sigma_name: str = "coarse",
    use_reverse: bool = True,
    pixels_per_sample: float = 4.0,
):
    """Sample-weight-normalised forward(+reverse) line energy.

    FORWARD  : projected model edge samples -> observed line distance field.
    REVERSE  : observed line support samples -> nearest projected model edge of
               the SAME semantic class.  Without it, a pose that puts one short
               edge on a strong line while the rest of the structure is wrong
               can score well.
    """
    sigma = field.sigmas[sigma_name]
    corners = PG.make_corners(*dims)[:PG.N_CORNERS]
    projected, depth = PG.project_points(corners, rotation, translation, intrinsics)
    width, height = field.image_size

    forward_num = 0.0
    forward_den = 0.0
    per_class_samples: dict[str, list[np.ndarray]] = {
        name: [] for name in PG.LINE_CLASSES
    }
    n_edges_used = 0
    for (i, j), line_class in edge_set:
        if depth[i] <= 1e-6 or depth[j] <= 1e-6:
            continue  # behind the camera: no image evidence exists
        clipped = PG.clip_segment_to_image(projected[i], projected[j], width, height)
        if clipped is None:
            continue
        start, end = clipped
        if float(np.linalg.norm(end - start)) < MIN_EDGE_LENGTH_PX:
            continue
        samples = PG.sample_along(start, end, pixels_per_sample=pixels_per_sample)
        distance, inside = field.sample_distance(line_class, samples)
        if not bool(inside.any()):
            continue
        usable = distance[inside]
        forward_num += float(np.sum(_rho(usable, sigma)))
        forward_den += float(usable.size)
        per_class_samples[line_class].append(samples[inside])
        n_edges_used += 1

    forward = forward_num / (forward_den + EPS) if forward_den > 0 else 1.0

    reverse = 1.0
    reverse_den = 0.0
    if use_reverse:
        reverse_num = 0.0
        for line_class in PG.LINE_CLASSES:
            observed = field.support[line_class]
            if observed.shape[0] == 0:
                continue
            model = per_class_samples[line_class]
            if not model:
                # the model predicts no edge of this class where one is observed
                reverse_num += float(observed.shape[0])
                reverse_den += float(observed.shape[0])
                continue
            model_points = np.concatenate(model, axis=0)
            deltas = observed[:, None, :] - model_points[None, :, :]
            nearest = np.sqrt(np.min(np.einsum("ijk,ijk->ij", deltas, deltas), axis=1))
            reverse_num += float(np.sum(_rho(nearest, sigma)))
            reverse_den += float(nearest.size)
        reverse = reverse_num / (reverse_den + EPS) if reverse_den > 0 else 1.0

    energy = 0.5 * forward + 0.5 * reverse if use_reverse else forward
    return float(energy), {
        "E_forward": float(forward),
        "E_reverse": float(reverse) if use_reverse else None,
        "n_edges_used": int(n_edges_used),
        "n_edges_fixed": int(len(edge_set)),
        "n_forward_samples": int(forward_den),
        "n_reverse_samples": int(reverse_den),
        "sigma_px": float(sigma),
    }
