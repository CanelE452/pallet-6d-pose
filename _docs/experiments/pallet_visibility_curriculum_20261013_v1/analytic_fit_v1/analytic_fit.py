"""Exact derivative of the existing initial-relative pose parameterization.

This module changes no observations, weights, robust policy, pose priors or
optimizer tolerances. The OpenCV Rodrigues derivative covers only the local
rotation; the camera projection uses the supplied complete 3x3 matrix.
"""
from __future__ import annotations

import cv2
import numpy as np


def projection_jacobian(params, xyz_initial_camera_rotation, K, initial_t, depth_scale):
    """Return the (2N,6) derivative of uv for delta-R and normalized delta-t.

    xyz_initial_camera_rotation is xyz @ initial_R.T. Output rows have the same
    point-major uv order as the existing residual's reshape(-1).
    """
    rotation, derivative = cv2.Rodrigues(np.asarray(params[:3], dtype=float))
    d_rotation = derivative.reshape(3, 3, 3)
    X0 = np.asarray(xyz_initial_camera_rotation, dtype=float)
    camera = X0 @ rotation.T + np.asarray(initial_t) + params[3:] * depth_scale
    homogeneous = camera @ np.asarray(K).T
    # The complete quotient rule also supports skew and a nontrivial K row 3.
    duv_dcamera = (
        K[None, :2, :] * homogeneous[:, 2, None, None]
        - homogeneous[:, :2, None] * K[None, 2:3, :]
    ) / homogeneous[:, 2, None, None] ** 2
    dcamera_drotation = np.einsum("pj,ikj->pki", X0, d_rotation)
    jacobian = np.empty((len(X0), 2, 6), dtype=float)
    jacobian[:, :, :3] = np.einsum("puk,pki->pui", duv_dcamera, dcamera_drotation)
    jacobian[:, :, 3:] = duv_dcamera * depth_scale
    return jacobian.reshape(-1, 6)


def weighted_residual_jacobian(params, X0, K, initial_t, depth_scale, weights, cfg):
    """Apply frozen IRLS sqrt weights and the exact existing prior derivative."""
    derivative = projection_jacobian(params, X0, K, initial_t, depth_scale)
    derivative *= np.repeat(np.sqrt(weights), 2)[:, None]
    prior = np.diag(np.r_[
        np.full(3, cfg.prior_strength_px / np.deg2rad(cfg.rotation_prior_deg)),
        np.full(3, cfg.prior_strength_px / cfg.translation_prior_fraction),
    ])
    return np.vstack([derivative, prior])
