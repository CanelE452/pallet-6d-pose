"""PPD — Pallet Polarity Disambiguation.

SAI produces *unsigned* semantic axes: a fitted 2-D line has no direction, so
``l^T K R a = 0`` holds equally for ``a`` and ``-a``.  Every sign combination
therefore has identical line-plane energy, and the solver cannot tell a pallet
from an upside-down pallet.  Measured on strict N87: 30/86 selected poses were
vertically inverted.

What actually went wrong in the previous gate (recorded precisely, because the
first explanation was wrong): ``rotation_error_sym_deg`` did NOT hide the
failure — it reported the forbidden width/depth-axis inversions as ~180 deg
errors, exactly as it should.  The gate hid them, by judging on a *subset
median*: with 30/86 inverted, the other 56 keep the median at 1.84 deg.  A
catastrophic tail is invisible to a median.

Two things live here:

1. **Signed metrics** — an error definition in which yaw+180 is still a success
   (a pallet rotated about its own up-axis is the same placement) while a
   top/base inversion is a failure, plus vertical polarity as its own explicit
   scalar so a tail can never again be averaged away.  Reprojection is
   index-wise over the allowed permutations only; Hungarian matching would
   re-pair an inverted pose's corners and report a small error.

2. **Polarity scorers** — ways to pick, among candidates that differ only in
   sign, the physically upright one.  None of them re-solves the pose; they
   only re-rank an existing candidate set.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

try:
    import pallet_graph_geometry as PG
except ImportError:  # pragma: no cover
    from . import pallet_graph_geometry as PG  # type: ignore

EPS = 1e-12
# Frozen before any N87 number was inspected.
HEATMAP_STAGES = (3, 4, 5)          # zero-based indices of belief stages 4,5,6
HEATMAP_TEMPERATURE = 0.1           # reuse of the DiffPnP local temperature
HEATMAP_WINDOW = 3                  # 3x3 neighbourhood
MIN_VALID_CORNERS = 4
POLARITY_CLASSES = (
    "top_width", "top_depth", "base_width", "base_depth", "vertical",
)


# ============================================================================
# Phase B — signed metrics
# ============================================================================
def object_up_axis(dims: tuple[float, float, float]) -> np.ndarray:
    """Object-frame up direction, derived from the vertical edge, not hardcoded.

    OpenCV's +Y points down in camera space and the pallet's local Y spans H,
    so 'up' is the negative vertical axis.  Deriving it keeps this correct if
    the corner convention is ever revised.
    """
    corners = PG.make_corners(*dims)[: PG.N_CORNERS]
    vertical = PG.derive_edges(PG.make_corners(*dims))["vertical"]
    i, j = vertical[0]
    delta = corners[j] - corners[i]
    axis = delta / max(float(np.linalg.norm(delta)), EPS)
    # point it "up": the top face is the one with the smaller Y (Y grows down)
    if corners[j][1] > corners[i][1]:
        axis = -axis
    return axis


def allowed_symmetries(dims: tuple[float, float, float]) -> list[np.ndarray]:
    """Identity and 180 degrees about the pallet up-axis — nothing else.

    A 180-degree rotation about width or depth turns the pallet upside down and
    must NOT be folded into the metric.
    """
    return [np.eye(3), PG.symmetry_rotation()]


def vertical_polarity_error_deg(
    predicted_R: np.ndarray, reference_R: np.ndarray,
    dims: tuple[float, float, float]
) -> float:
    """Angle between predicted and true camera-frame up.  >=90 deg = inverted."""
    axis = object_up_axis(dims)
    predicted = np.asarray(predicted_R, dtype=np.float64) @ axis
    reference = np.asarray(reference_R, dtype=np.float64) @ axis
    cosine = float(np.clip(np.dot(predicted, reference), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def polarity_correct(
    predicted_R: np.ndarray, reference_R: np.ndarray,
    dims: tuple[float, float, float]
) -> bool:
    return vertical_polarity_error_deg(predicted_R, reference_R, dims) < 90.0


def signed_rotation_error_deg(
    predicted_R: np.ndarray, reference_R: np.ndarray,
    dims: tuple[float, float, float]
) -> float:
    """Geodesic error minimised over the ALLOWED symmetries only."""
    reference = np.asarray(reference_R, dtype=np.float64)
    return min(
        PG.rotation_error_deg(predicted_R, reference @ S)
        for S in allowed_symmetries(dims)
    )


def yaw180_permutation(dims: tuple[float, float, float]) -> tuple[int, ...]:
    """Keypoint permutation induced by the allowed yaw+180 symmetry."""
    return PG.symmetry_permutation(*dims)


def fixed_indexed_reprojection(
    pose: dict[str, Any], observations: list[Any], intrinsics: np.ndarray,
    dims: tuple[float, float, float]
) -> Optional[float]:
    """Index-wise reprojection under the allowed permutations ONLY.

    Hungarian / order-free matching is forbidden here: it would silently
    re-pair an upside-down pose's corners with the GT corners and report a
    small error, which is exactly how the previous gate passed.
    """
    if pose is None or intrinsics is None or dims is None:
        return None
    points = PG.make_corners(*dims)
    projected, depth = PG.project_points(
        points, pose["R"], np.asarray(pose["t"]).reshape(3), intrinsics)
    permutation = np.asarray(yaw180_permutation(dims), dtype=np.int64)
    best: Optional[float] = None
    for order in (np.arange(PG.N_KEYPOINTS), permutation):
        errors = []
        for index in range(PG.N_KEYPOINTS):
            observed = observations[index]
            if observed is None or not np.isfinite(np.asarray(observed)).all():
                continue
            source = int(order[index])
            if depth[source] <= 1e-6:
                continue
            errors.append(float(np.linalg.norm(
                projected[source] - np.asarray(observed, dtype=np.float64))))
        if errors:
            value = float(np.mean(errors))
            best = value if best is None else min(best, value)
    return best


# ============================================================================
# Phase C — candidate polarity grouping (GT-free)
# ============================================================================
def candidate_polarity(
    rotation: np.ndarray, dims: tuple[float, float, float]
) -> str:
    """'upright' when the object's up-axis points up in the camera frame.

    Camera +Y is down, so an upward-pointing axis has a negative Y component.
    This uses no GT: it is a property of the candidate alone.
    """
    up_camera = np.asarray(rotation, dtype=np.float64) @ object_up_axis(dims)
    return "upright" if float(up_camera[1]) < 0.0 else "inverted"


def group_candidates(
    candidates: list[np.ndarray], dims: tuple[float, float, float],
    yaw_tolerance_deg: float = 5.0
) -> list[dict[str, Any]]:
    """Tag each candidate with a yaw-equivalence id and a polarity label."""
    rotations = [np.asarray(c, dtype=np.float64).reshape(3, 3) for c in candidates]
    groups: list[dict[str, Any]] = []
    yaw_ids: list[int] = []
    for index, rotation in enumerate(rotations):
        assigned = None
        for other in range(index):
            if PG.rotation_error_sym_deg(rotation, rotations[other]) < yaw_tolerance_deg:
                assigned = yaw_ids[other]
                break
        yaw_ids.append(assigned if assigned is not None else len(set(yaw_ids)))
    for index, rotation in enumerate(rotations):
        groups.append({
            "candidate_index": index,
            "yaw_equivalence_id": yaw_ids[index],
            "vertical_polarity": candidate_polarity(rotation, dims),
        })
    # pair each candidate with its upside-down partner, if present
    for entry in groups:
        entry["inverted_partner"] = None
        this = rotations[entry["candidate_index"]]
        for other in groups:
            if other is entry or other["vertical_polarity"] == entry["vertical_polarity"]:
                continue
            if signed_rotation_error_deg(this, rotations[other["candidate_index"]], dims) > 90.0:
                entry["inverted_partner"] = other["candidate_index"]
                break
    return groups


# ============================================================================
# Phase D — frozen heatmap polarity scorer (no training)
# ============================================================================
def spatial_softmax(heatmap: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = values / float(temperature)
    logits = logits - float(logits.max())
    probability = np.exp(logits)
    total = float(probability.sum())
    return probability / total if total > EPS else np.full_like(probability, 1.0 / probability.size)


def heatmap_polarity_score(
    rotation: np.ndarray, translation: np.ndarray, intrinsics: np.ndarray,
    dims: tuple[float, float, float], belief_stages: np.ndarray,
    image_size: tuple[int, int], window: int = HEATMAP_WINDOW,
    temperature: float = HEATMAP_TEMPERATURE
) -> tuple[Optional[float], dict[str, Any]]:
    """Negative log spatial likelihood of a candidate's projected corners.

    Uses the *spatial distribution* of the belief map, not its peak value, so
    the score is invariant to an additive offset and a flat "no response" map
    contributes near-uniform likelihood rather than a confident vote.
    """
    stages = np.asarray(belief_stages, dtype=np.float32)
    width, height = int(image_size[0]), int(image_size[1])
    grid_h, grid_w = stages.shape[-2], stages.shape[-1]
    corners = PG.make_corners(*dims)[: PG.N_CORNERS]
    projected, depth = PG.project_points(corners, rotation, translation, intrinsics)

    per_stage: dict[int, list[float]] = {}
    valid_corners = 0
    radius = window // 2
    for corner in range(PG.N_CORNERS):
        if depth[corner] <= 1e-6:
            continue
        x = projected[corner, 0] * grid_w / max(width, 1)
        y = projected[corner, 1] * grid_h / max(height, 1)
        if not (0 <= x < grid_w and 0 <= y < grid_h):
            continue
        valid_corners += 1
        cx, cy = int(round(float(x))), int(round(float(y)))
        for stage in HEATMAP_STAGES:
            probability = spatial_softmax(stages[stage, corner], temperature)
            x0, x1 = max(0, cx - radius), min(grid_w, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(grid_h, cy + radius + 1)
            mass = float(probability[y0:y1, x0:x1].sum())
            per_stage.setdefault(stage, []).append(-math.log(max(mass, 1e-12)))
    if valid_corners < MIN_VALID_CORNERS:
        return None, {"valid_corners": valid_corners, "undecidable": True}
    stage_means = {s: float(np.mean(v)) for s, v in per_stage.items()}
    return float(np.mean(list(stage_means.values()))), {
        "valid_corners": valid_corners,
        "undecidable": False,
        "per_stage": stage_means,
    }


def select_polarity(
    scores: dict[str, Optional[float]]
) -> tuple[Optional[str], Optional[float]]:
    """Lowest-energy polarity class; margin is recorded, never used to abstain."""
    usable = {k: v for k, v in scores.items() if v is not None and math.isfinite(v)}
    if not usable:
        return None, None
    ordered = sorted(usable.items(), key=lambda kv: kv[1])
    margin = (ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else None
    return ordered[0][0], margin


# ============================================================================
# Phase E — polarity-aware semantic edge classes
# ============================================================================
def polarity_edge_classes(
    dims: tuple[float, float, float]
) -> list[tuple[tuple[int, int], str]]:
    """Split width/depth edges into top/base by object-frame vertical level.

    Derived from the 3-D coordinates, never from hardcoded keypoint ids.
    """
    corners = PG.make_corners(*dims)[: PG.N_CORNERS]
    axis = object_up_axis(dims)
    level = corners @ axis                      # larger = closer to the top
    top_level, base_level = float(level.max()), float(level.min())
    edges = PG.derive_edges(PG.make_corners(*dims))
    out: list[tuple[tuple[int, int], str]] = []
    for name, pairs in edges.items():
        for i, j in pairs:
            if name == "vertical":
                out.append(((i, j), "vertical"))
                continue
            mean_level = 0.5 * (level[i] + level[j])
            near_top = abs(mean_level - top_level) < abs(mean_level - base_level)
            out.append(((i, j), f"{'top' if near_top else 'base'}_{name}"))
    counts = {c: sum(1 for _, k in out if k == c) for c in POLARITY_CLASSES}
    if counts["vertical"] != 4 or any(counts[c] != 2 for c in POLARITY_CLASSES[:4]):
        raise RuntimeError(f"unexpected polarity edge counts: {counts}")
    return out
