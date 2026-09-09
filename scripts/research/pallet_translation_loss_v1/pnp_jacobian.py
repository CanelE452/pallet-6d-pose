"""Local PnP Jacobian at a given pose, and its finite-difference check.

Convention, fixed here once so every consumer shares it:

    R(xi) = expm(skew(dr)) @ R0      dr in the *camera* frame
    t(xi) = t0 + dt                  dt in the *camera* frame
    xi    = [drx, dry, drz, dtx, dty, dtz]

J is d vec(u) / d xi with vec(u) = [u0x, u0y, u1x, u1y, ...] over the 8 cuboid
corners.  Only the 8 corners enter — index 8 (centroid) is not an independent
3D correspondence and is excluded from every translation-risk quantity.
"""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def expm_so3(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3) + skew(v)
    K = skew(v / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """X (N,3) object points -> (N,2) pixels.  No distortion (evaluator uses None)."""
    P = X @ R.T + t.reshape(1, 3)
    z = P[:, 2]
    u = K[0, 0] * (P[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (P[:, 1] / z) + K[1, 2]
    return np.stack([u, v], axis=1)


def pnp_jacobian(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(2N, 6) analytic Jacobian in the convention above."""
    P = X @ R.T + t.reshape(1, 3)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z

    # d u / d P  ->  (N, 2, 3)
    dudP = np.zeros((len(X), 2, 3))
    dudP[:, 0, 0] = fx * inv_z
    dudP[:, 0, 2] = -fx * x * inv_z2
    dudP[:, 1, 1] = fy * inv_z
    dudP[:, 1, 2] = -fy * y * inv_z2

    # P = expm(skew(dr)) @ (R X) + (t + dt), so at dr = 0
    #   d P / d dr = -skew(R X)   -- the *rotated* point, not the translated one
    #   d P / d dt = I
    RX = X @ R.T
    dPdxi = np.zeros((len(X), 3, 6))
    for i in range(len(X)):
        dPdxi[i, :, :3] = -skew(RX[i])
        dPdxi[i, :, 3:] = np.eye(3)

    J = np.einsum("nij,njk->nik", dudP, dPdxi)      # (N, 2, 6)
    return J.reshape(-1, 6)


def finite_difference_jacobian(K, R, t, X, eps=1e-6) -> np.ndarray:
    out = np.zeros((2 * len(X), 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = eps
        Rp, tp = expm_so3(d[:3]) @ R, t + d[3:]
        Rm, tm = expm_so3(-d[:3]) @ R, t - d[3:]
        up = project(K, Rp, tp, X).reshape(-1)
        um = project(K, Rm, tm, X).reshape(-1)
        out[:, j] = (up - um) / (2 * eps)
    return out
