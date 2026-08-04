"""Viewpoint variables and role-conditioned bias models.

The question this supports is whether the far-face bias is repeatable given the
camera viewpoint and the corner's physical role, or whether it is noise.  The
view is derived from the GT pose in the pallet's own frame, never invented, and
yaw is carried as a double angle so that a pallet rotated by 180 degrees -- which
the project already treats as the same pose -- maps to the same view target.  A
top-bottom inversion is a different pose and must not collapse onto it.

Nothing here trains: the bias models are least squares fits evaluated
leave-one-session-out.
"""
from __future__ import annotations

import numpy as np

RIDGE_LAMBDA = 1e-3          # fixed before any result is seen
EPS = 1e-6

# Fixed feature basis, frozen ahead of the gate.
FEATURE_NAMES_B2 = ("bias", "cos2psi", "sin2psi", "sin_elev", "cos_elev")
FEATURE_NAMES_B3 = FEATURE_NAMES_B2 + ("log_scale", "cos2psi_sin_elev",
                                       "sin2psi_sin_elev")


def viewing_direction(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unit vector from the object towards the camera, in object coordinates."""
    direction = -np.asarray(R, float).T @ np.asarray(t, float).reshape(3)
    norm = float(np.linalg.norm(direction))
    return direction / max(norm, EPS)


def view_angles(R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """(azimuth psi, elevation epsilon) in radians.

    Local X spans width, Y spans height and points down in the OpenCV
    convention, Z spans depth, so elevation uses -y.
    """
    v = viewing_direction(R, t)
    psi = float(np.arctan2(v[0], v[2]))
    epsilon = float(np.arcsin(np.clip(-v[1], -1.0, 1.0)))
    return psi, epsilon


def object_scale(projected: np.ndarray, image_size: tuple[int, int]) -> float:
    """GT 8-corner bbox diagonal over the image diagonal."""
    points = np.asarray(projected, float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return EPS
    span = points.max(axis=0) - points.min(axis=0)
    image_diagonal = float(np.hypot(*image_size))
    return float(np.hypot(*span) / max(image_diagonal, EPS))


def view_feature(psi: float, epsilon: float, scale: float,
                 full: bool = True) -> np.ndarray:
    """The frozen basis.  full=False gives the B2 control without scale terms."""
    cos2, sin2 = np.cos(2.0 * psi), np.sin(2.0 * psi)
    sin_e, cos_e = np.sin(epsilon), np.cos(epsilon)
    base = [1.0, cos2, sin2, sin_e, cos_e]
    if not full:
        return np.asarray(base, float)
    return np.asarray(base + [float(np.log(max(scale, EPS))),
                              cos2 * sin_e, sin2 * sin_e], float)


def yaw_double_angle(psi: float) -> np.ndarray:
    return np.asarray([np.cos(2.0 * psi), np.sin(2.0 * psi)], float)


def elevation_pair(epsilon: float) -> np.ndarray:
    return np.asarray([np.sin(epsilon), np.cos(epsilon)], float)


# ============================================================================
# corner physical role
# ============================================================================
def corner_roles(near: tuple[int, ...], far: tuple[int, ...],
                 top: tuple[int, ...], bottom: tuple[int, ...],
                 left: tuple[int, ...], right: tuple[int, ...]
                 ) -> dict[int, dict[str, str]]:
    """Role attributes per corner id, taken from the existing grouping."""
    roles = {}
    for corner in range(8):
        roles[corner] = {
            "depth": "near" if corner in near else "far",
            "height": "top" if corner in top else "bottom",
            "side": "left" if corner in left else "right",
        }
    assert set(near) | set(far) == set(range(8))
    assert set(top) | set(bottom) == set(range(8))
    assert set(left) | set(right) == set(range(8))
    return roles


# ============================================================================
# bias models, fitted per corner id
# ============================================================================
def fit_ridge(features: np.ndarray, targets: np.ndarray,
              lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """Closed-form ridge; the intercept column is not penalised."""
    X = np.asarray(features, float)
    Y = np.asarray(targets, float)
    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ Y)


class BiasModel:
    """B1 role-constant, or B2/B3 role x view ridge, fitted per corner."""

    def __init__(self, kind: str, full_basis: bool = True) -> None:
        assert kind in ("none", "constant", "linear")
        self.kind = kind
        self.full_basis = full_basis
        self.weights: dict[int, np.ndarray] = {}
        self.mean: dict[int, np.ndarray] = {}
        self.std: dict[int, np.ndarray] = {}

    def fit(self, corner_ids, features, deltas) -> "BiasModel":
        corner_ids = np.asarray(corner_ids)
        features = np.asarray(features, float)
        deltas = np.asarray(deltas, float)
        for corner in np.unique(corner_ids):
            mask = corner_ids == corner
            if self.kind == "constant":
                self.weights[int(corner)] = deltas[mask].mean(axis=0)
                continue
            block = features[mask]
            # standardise on the training fold only; column 0 is the intercept
            mean = block.mean(axis=0)
            std = block.std(axis=0)
            mean[0], std[0] = 0.0, 1.0
            std = np.where(std < 1e-8, 1.0, std)
            self.mean[int(corner)] = mean
            self.std[int(corner)] = std
            self.weights[int(corner)] = fit_ridge((block - mean) / std,
                                                  deltas[mask])
        return self

    def predict(self, corner_ids, features) -> np.ndarray:
        corner_ids = np.asarray(corner_ids)
        features = np.asarray(features, float)
        out = np.zeros((len(corner_ids), 2), float)
        if self.kind == "none":
            return out
        for index, corner in enumerate(corner_ids):
            key = int(corner)
            if key not in self.weights:
                continue
            if self.kind == "constant":
                out[index] = self.weights[key]
            else:
                normalised = (features[index] - self.mean[key]) / self.std[key]
                out[index] = normalised @ self.weights[key]
        return out
