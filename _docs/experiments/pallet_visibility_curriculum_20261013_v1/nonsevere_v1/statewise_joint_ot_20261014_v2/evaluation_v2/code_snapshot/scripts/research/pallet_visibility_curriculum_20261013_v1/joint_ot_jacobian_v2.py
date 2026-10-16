"""Exact derivatives for the existing joint O/T continuous pose problem.

This module is pure numerical code: no model, image, data or filesystem access.
Parameter ordering is rotation-vector radians then translation / fixed scale.
The native object coordinates and full camera matrix are never reindexed.
Constraint ordering matches joint_ot_pnp: all positive depths, then each B/I
seed shift and covariance residual, then every O/T domain inequality.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def so3_right_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """Derivative of Exp(phi) in its right tangent, stable near phi=0."""
    phi = np.asarray(rotvec, dtype=float)
    theta2 = float(phi @ phi)
    if theta2 < 1e-6:
        # Terms through theta^6 avoid cancellation in 1-cos(theta) and
        # theta-sin(theta), including exact zero and very small rotations.
        a = .5 - theta2 / 24 + theta2**2 / 720 - theta2**3 / 40320
        b = 1 / 6 - theta2 / 120 + theta2**2 / 5040 - theta2**3 / 362880
    else:
        theta = np.sqrt(theta2)
        a = (1 - np.cos(theta)) / theta2
        b = (theta - np.sin(theta)) / (theta2 * theta)
    omega = skew(phi)
    return np.eye(3) - a * omega + b * (omega @ omega)


def projection_and_jacobian(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
                            scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return xy[N,2], depth[N], dxy[N,2,6] and ddepth[N,6].

    The finite denominator extension exactly follows the existing projector:
    |homogeneous_z|<1e-10 uses a signed constant denominator. Its derivative
    is zero within each side of that clipping interval. At the clipping
    boundaries or sign jump the extension is not differentiable; valid poses
    have positive depths far above it.
    """
    p = np.asarray(p, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    K = np.asarray(K, dtype=float)
    if p.shape != (6,) or xyz.ndim != 2 or xyz.shape[1] != 3 or K.shape != (3, 3):
        raise ValueError("Expected pose[6], native xyz[N,3] and K[3,3]")
    if not np.isfinite(p).all() or not np.isfinite(xyz).all() or not np.isfinite(K).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pose, geometry, camera and positive translation scale must be finite")
    # Older SciPy's Cython wrapper requests a writable memoryview even though
    # it does not mutate phi; allow immutable caller-owned pose arrays.
    R = Rotation.from_rotvec(p[:3].copy()).as_matrix()
    camera = xyz @ R.T + p[3:] * scale
    # d(R*X)/dphi = -R*[X]_cross*J_right(phi).
    object_skew = np.zeros((len(xyz), 3, 3))
    object_skew[:, 0, 1] = -xyz[:, 2]; object_skew[:, 0, 2] = xyz[:, 1]
    object_skew[:, 1, 0] = xyz[:, 2]; object_skew[:, 1, 2] = -xyz[:, 0]
    object_skew[:, 2, 0] = -xyz[:, 1]; object_skew[:, 2, 1] = xyz[:, 0]
    camera_jacobian = np.empty((len(xyz), 3, 6))
    camera_jacobian[:, :, :3] = -np.einsum("ij,njk,kl->nil", R, object_skew, so3_right_jacobian(p[:3]))
    camera_jacobian[:, :, 3:] = np.eye(3) * scale
    homogeneous = camera @ K.T
    z = homogeneous[:, 2]
    clipped = np.abs(z) < 1e-10
    denominator = np.where(clipped, np.where(z < 0, -1e-10, 1e-10), z)
    xy = homogeneous[:, :2] / denominator[:, None]
    homogeneous_jacobian = np.einsum("ij,njk->nik", K, camera_jacobian)
    denominator_jacobian = homogeneous_jacobian[:, 2].copy()
    denominator_jacobian[clipped] = 0
    xy_jacobian = (homogeneous_jacobian[:, :2] / denominator[:, None, None]
                   - homogeneous[:, :2, None] * denominator_jacobian[:, None]
                   / denominator[:, None, None]**2)
    return xy, camera[:, 2], xy_jacobian, camera_jacobian[:, 2]


def _whitening(measurement: Mapping) -> np.ndarray:
    if "whitening" in measurement:
        return np.asarray(measurement["whitening"], dtype=float)
    return np.linalg.inv(np.linalg.cholesky(np.asarray(measurement["covariance_px2"], dtype=float)))


def visible_residual_and_jacobian(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
                                  measurements: Sequence[Mapping], scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Frozen covariance-whitened B/I residual and exact 2M-by-6 Jacobian."""
    xy, _, jacobian, _ = projection_and_jacobian(p, xyz, K, scale)
    residuals, rows = [], []
    for measurement in measurements:
        native_id = int(measurement["native_corner"])
        whitening = _whitening(measurement)
        residuals.append(whitening @ (xy[native_id] - measurement["xy"]))
        rows.append(whitening @ jacobian[native_id])
    if not measurements:
        return np.empty(0), np.empty((0, 6))
    return np.concatenate(residuals), np.concatenate(rows, axis=0)


def visible_objective_and_gradient(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
                                   measurements: Sequence[Mapping], scale: float,
                                   huber_scale: float) -> tuple[float, np.ndarray]:
    """The solver's radial pseudo-Huber objective and exact 6-vector gradient."""
    if not np.isfinite(huber_scale) or huber_scale <= 0:
        raise ValueError("huber_scale must be positive")
    residual, jacobian = visible_residual_and_jacobian(p, xyz, K, measurements, scale)
    radial = residual.reshape((-1, 2))
    squared = np.einsum("ij,ij->i", radial, radial)
    denominator = np.sqrt(1 + squared / huber_scale**2)
    objective = float(np.sum(2 * huber_scale**2 * (denominator - 1)))
    derivative = (2 * radial / denominator[:, None]).reshape(-1)
    return objective, derivative @ jacobian


def constraint_values_and_jacobian(p: np.ndarray, xyz: np.ndarray, K: np.ndarray,
                                   measurements: Sequence[Mapping], domains: Sequence[Mapping],
                                   sides: Sequence[str | None], reference_projection: np.ndarray,
                                   image_hw: Sequence[int], scale: float,
                                   config: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """All hard inequalities and exact Jacobian; nonnegative means feasible.

    Domains have the normalized representation from joint_ot_pnp._inputs:
    ellipse basis[2,2], half_axes_px, or bounds_xyxy, plus halfplanes[a,b,c]
    meaning a*x+b*y<=c. T additionally intersects the chosen exterior side.
    Neither values nor derivatives average away an individual violation.
    """
    xy, depth, xy_jacobian, depth_jacobian = projection_and_jacobian(p, xyz, K, scale)
    reference = np.asarray(reference_projection, dtype=float)
    positive_depth = float(config["positive_depth_m"])
    max_mahalanobis = float(config["maximum_visible_mahalanobis"])
    values = ((depth - positive_depth) / scale).tolist()
    rows = list(depth_jacobian / scale)
    for measurement in measurements:
        native_id = int(measurement["native_corner"])
        delta = xy[native_id] - reference[native_id]
        protection2 = float(measurement["protection_px"])**2
        values.append(float(1 - delta @ delta / protection2))
        rows.append((-2 * delta / protection2) @ xy_jacobian[native_id])
        whitening = _whitening(measurement)
        whitened = whitening @ (xy[native_id] - measurement["xy"])
        values.append(float(1 - whitened @ whitened / max_mahalanobis**2))
        rows.append((-2 * (whitening.T @ whitened) / max_mahalanobis**2) @ xy_jacobian[native_id])
    if len(domains) != len(sides):
        raise ValueError("Exactly one exterior side assignment is needed per domain")
    for domain, side in zip(domains, sides):
        native_id = int(domain["native_corner"])
        point = xy[native_id]
        shape = domain["shape"]
        if shape == "ellipse":
            basis = np.asarray(domain["basis"], dtype=float)
            delta = basis.T @ (point - domain["center_xy"])
            axes = np.asarray(domain["half_axes_px"], dtype=float)
            values.append(float(1 - np.sum((delta / axes)**2)))
            rows.append((-2 * (basis @ (delta / axes**2))) @ xy_jacobian[native_id])
        elif shape == "box":
            bounds = np.asarray(domain["bounds_xyxy"], dtype=float)
            span = bounds[2:] - bounds[:2]
            values.extend(((point - bounds[:2]) / span).tolist())
            values.extend(((bounds[2:] - point) / span).tolist())
            rows.extend(np.diag(1 / span) @ xy_jacobian[native_id])
            rows.extend(-np.diag(1 / span) @ xy_jacobian[native_id])
        elif shape != "halfplane":
            raise ValueError("Expected normalized ellipse, box or halfplane domain")
        for a, b, c in np.asarray(domain["halfplanes"], dtype=float).reshape((-1, 3)):
            norm = np.hypot(a, b)
            values.append(float((c - a * point[0] - b * point[1]) / norm))
            rows.append((np.array([-a, -b]) / norm) @ xy_jacobian[native_id])
        if domain["state"] == "T":
            height, width = image_hw
            margin = float(domain["outside_margin_px"])
            if side == "left":
                value, derivative = -margin - point[0], np.array([-1., 0.])
            elif side == "right":
                value, derivative = point[0] - width - margin, np.array([1., 0.])
            elif side == "top":
                value, derivative = -margin - point[1], np.array([0., -1.])
            elif side == "bottom":
                value, derivative = point[1] - height - margin, np.array([0., 1.])
            else:
                raise ValueError("A T exterior branch needs left/right/top/bottom")
            values.append(float(value))
            rows.append(derivative @ xy_jacobian[native_id])
    return np.asarray(values), np.asarray(rows).reshape((-1, 6))
