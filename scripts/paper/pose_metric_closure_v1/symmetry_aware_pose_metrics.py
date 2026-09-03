"""6D pose metrics under a declared 180-degree symmetry group.

Both pallets are rectangular, so a 180-degree turn about the vertical axis maps the
object onto itself and a 90-degree turn does not.  The evaluation therefore treats
orientation as a **180-degree equivalence class** and must leave a wrong long/short
axis choice showing as roughly 90 degrees of error.

    S = { I , Ry(180) }

    yaw error        delta = |wrap180(yaw_pred - yaw_gt)|
                     yaw   = min(delta, 180 - delta)          range 0..90

    rotation error   min over S of geodesic(R_pred, R_gt @ S)

    symmetry-aware   min over S of  mean_i || T_pred X_i - T_gt S X_i ||
    ADD

    pose AUC         area under accuracy(tau) for tau in [0, 0.1 x diameter],
                     1001 integration points, normalised to [0, 1]

90 degrees is never absorbed by the symmetry group.  For a forklift the difference
between entering the pockets and hitting the deck is exactly that rotation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# 정본 적분 해상도.  결과를 본 뒤에 바꾸지 않는다.
AUC_INTEGRATION_POINTS = 1001
AUC_MAX_FRACTION = 0.1

_YAW_180 = np.diag([-1.0, 1.0, -1.0])
SYMMETRY_GROUP: tuple[np.ndarray, ...] = (np.eye(3), _YAW_180)
SYMMETRY_GROUP_YAW_DEGREES: tuple[float, ...] = (0.0, 180.0)


def _rotation(matrix, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite (3,3) rotation")
    if np.max(np.abs(value.T @ value - np.eye(3))) > 1e-6:
        raise ValueError(f"{name} is not orthonormal")
    if abs(float(np.linalg.det(value)) - 1.0) > 1e-6:
        raise ValueError(f"{name} is not a proper rotation")
    return value


def _translation(vector, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be three finite values")
    return value


def _geodesic_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip((np.trace(b.T @ a) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def wrap180(degrees: float) -> float:
    """Fold an angle into (-180, 180].

    The half-open interval matters at the boundary: 180 folds to +180, not -180.
    """

    value = float((float(degrees) + 180.0) % 360.0 - 180.0)
    return 180.0 if value == -180.0 else value


def yaw_error_degrees(predicted_rotation, target_rotation) -> float:
    """Yaw about the vertical axis, folded into the 180-degree class.

    Returns 0..90.  A wrong long/short axis choice lands near 90.
    """

    predicted = _rotation(predicted_rotation, "predicted_rotation")
    target = _rotation(target_rotation, "target_rotation")
    relative = target.T @ predicted
    delta = abs(wrap180(np.degrees(np.arctan2(relative[0, 2], relative[2, 2]))))
    return float(min(delta, 180.0 - delta))


def rotation_error_degrees(predicted_rotation, target_rotation) -> float:
    """Geodesic angle, minimised over the declared symmetry group."""

    predicted = _rotation(predicted_rotation, "predicted_rotation")
    target = _rotation(target_rotation, "target_rotation")
    return float(min(_geodesic_degrees(predicted, target @ member)
                     for member in SYMMETRY_GROUP))


def translation_error_m(predicted_translation, target_translation) -> float:
    predicted = _translation(predicted_translation, "predicted_translation")
    target = _translation(target_translation, "target_translation")
    return float(np.linalg.norm(predicted - target))


def translation_components_m(predicted_translation, target_translation) -> dict:
    """Lateral (x,y) and depth (z) split, for the appendix."""

    predicted = _translation(predicted_translation, "predicted_translation")
    target = _translation(target_translation, "target_translation")
    difference = predicted - target
    return {
        "total_m": float(np.linalg.norm(difference)),
        "lateral_m": float(np.linalg.norm(difference[[0, 1]])),
        "depth_m": float(abs(difference[2])),
    }


def symmetry_aware_add_m(
    model_points: Sequence[Sequence[float]] | np.ndarray,
    predicted_rotation, predicted_translation,
    target_rotation, target_translation,
) -> float:
    """min over S of the mean corresponding-point distance.

    This is a **group-aware** ADD, not the unrestricted nearest-neighbour ADD-S.
    Unrestricted nearest-neighbour matching would let a 90-degree swap score zero on
    a square footprint, which is exactly the failure the evaluation must expose.
    """

    points = np.asarray(model_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("model_points must be a finite (N,3) array")
    predicted = _rotation(predicted_rotation, "predicted_rotation")
    target = _rotation(target_rotation, "target_rotation")
    offset_pred = _translation(predicted_translation, "predicted_translation")
    offset_gt = _translation(target_translation, "target_translation")

    transformed_pred = (predicted @ points.T).T + offset_pred
    best = float("inf")
    for member in SYMMETRY_GROUP:
        rotated = (member @ points.T).T
        transformed_gt = (target @ rotated.T).T + offset_gt
        distance = float(np.linalg.norm(transformed_pred - transformed_gt, axis=1).mean())
        best = min(best, distance)
    return best


def model_diameter_m(model_points: Sequence[Sequence[float]] | np.ndarray) -> float:
    points = np.asarray(model_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("model_points must have shape (N>=2,3)")
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    diameter = float(pairwise.max())
    if diameter <= 0.0:
        raise ValueError("model diameter must be positive")
    return diameter


def pose_auc(errors_m: Iterable[float], diameter_m: float, *,
             max_fraction: float = AUC_MAX_FRACTION,
             points: int = AUC_INTEGRATION_POINTS) -> float:
    """Area under accuracy(tau) for tau in [0, max_fraction x diameter].

    Threshold-free by construction, so it cannot be tuned after seeing results.
    The integration resolution is frozen at AUC_INTEGRATION_POINTS.
    """

    values = np.asarray(list(errors_m), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("errors_m must be a non-empty 1-D sequence")
    if np.isnan(values).any() or (values < 0.0).any():
        raise ValueError("errors_m must be non-negative and not NaN")
    if not np.isfinite(diameter_m) or diameter_m <= 0.0:
        raise ValueError("diameter_m must be positive and finite")
    limit = float(max_fraction) * float(diameter_m)
    thresholds = np.linspace(0.0, limit, points)
    accuracy = np.array([(values <= threshold).mean() for threshold in thresholds])
    # np.trapezoid 는 numpy 2.0+ 이름이다.  이 환경은 1.x 라 trapz 를 쓴다.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    area = float(integrate(accuracy, thresholds) / limit)
    return float(min(max(area, 0.0), 1.0))


def cuboid_model_points(extents: Sequence[float]) -> np.ndarray:
    """8 corners in the camera-facing 0123 order used throughout the repository."""

    across, height, along = (float(v) for v in extents)
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)
