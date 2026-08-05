"""Nine-point pose consistency for the far-decoupled adapter.

Every earlier attempt lowered corner error without moving the pose, so this term
does not ask the corners to sit closer to their own targets.  It asks the four
corners the adapter may touch to form, together with the four it may not and the
centroid, a correspondence set whose solved pose matches the GT pose.

The Gauss-Newton is seeded from the GT pose.  That is a training-time
convenience for stability -- inference never sees it -- and the objective is
measured against GT geometry, never against the predicted 2D observations,
because fitting those harder is exactly what failed before.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

GN_STEPS = 4
GN_DAMPING = 1e-3
GN_DELTA_CLIP = 0.5
GN_COND_MAX = 1e8
HUBER_DELTA = 1.0


def rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    batch = rvec.shape[0]
    theta = rvec.norm(dim=1, keepdim=True).clamp_min(1e-9)
    axis = rvec / theta
    cos, sin = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(axis[:, :1])
    cross = torch.cat([zero, -axis[:, 2:3], axis[:, 1:2],
                       axis[:, 2:3], zero, -axis[:, 0:1],
                       -axis[:, 1:2], axis[:, 0:1], zero], dim=1).reshape(batch, 3, 3)
    eye = torch.eye(3, device=rvec.device, dtype=rvec.dtype)[None]
    outer = torch.bmm(axis[:, :, None], axis[:, None, :])
    return (cos[..., None] * eye + sin[..., None] * cross
            + (1 - cos)[..., None] * outer)


def to_rodrigues(rotation: torch.Tensor) -> torch.Tensor:
    trace = ((rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2] - 1) / 2)
    angle = torch.arccos(trace.clamp(-1 + 1e-7, 1 - 1e-7))[:, None]
    axis = torch.stack([rotation[:, 2, 1] - rotation[:, 1, 2],
                        rotation[:, 0, 2] - rotation[:, 2, 0],
                        rotation[:, 1, 0] - rotation[:, 0, 1]], dim=1)
    return axis / (2 * torch.sin(angle).clamp_min(1e-7)) * angle


def project(rvec, tvec, points, K):
    rotation = rodrigues(rvec)
    camera = torch.bmm(points, rotation.transpose(1, 2)) + tvec[:, None, :]
    uvw = torch.bmm(camera, K.transpose(1, 2))
    return uvw[:, :, :2] / uvw[:, :, 2:3].clamp_min(1e-6), camera


def jacobian(rvec, tvec, points, K):
    batch, n, _ = points.shape
    rotation = rodrigues(rvec)
    camera = torch.bmm(points, rotation.transpose(1, 2)) + tvec[:, None, :]
    fx = K[:, 0, 0].view(batch, 1)
    fy = K[:, 1, 1].view(batch, 1)
    x, y = camera[:, :, 0], camera[:, :, 1]
    z = camera[:, :, 2].clamp_min(1e-6)
    dpdP = torch.zeros(batch, n, 2, 3, device=points.device, dtype=points.dtype)
    dpdP[:, :, 0, 0] = fx / z
    dpdP[:, :, 0, 2] = -fx * x / (z * z)
    dpdP[:, :, 1, 1] = fy / z
    dpdP[:, :, 1, 2] = -fy * y / (z * z)
    rotated = camera - tvec[:, None, :]
    dPdr = torch.zeros(batch, n, 3, 3, device=points.device, dtype=points.dtype)
    dPdr[:, :, 0, 1] = rotated[:, :, 2]
    dPdr[:, :, 0, 2] = -rotated[:, :, 1]
    dPdr[:, :, 1, 0] = -rotated[:, :, 2]
    dPdr[:, :, 1, 2] = rotated[:, :, 0]
    dPdr[:, :, 2, 0] = rotated[:, :, 1]
    dPdr[:, :, 2, 1] = -rotated[:, :, 0]
    dPdt = torch.eye(3, device=points.device,
                     dtype=points.dtype).view(1, 1, 3, 3).expand(batch, n, -1, -1)
    return torch.cat([torch.matmul(dpdP, dPdr), torch.matmul(dpdP, dPdt)],
                     dim=-1).reshape(batch, 2 * n, 6)


def gauss_newton(observed, points, K, rvec, tvec):
    """Differentiable refinement from the GT seed; returns pose and a validity."""
    valid = torch.ones(observed.shape[0], dtype=torch.bool, device=observed.device)
    for _ in range(GN_STEPS):
        uv, camera = project(rvec, tvec, points, K)
        residual = (uv - observed).reshape(observed.shape[0], -1, 1)
        J = jacobian(rvec, tvec, points, K)
        JtJ = torch.bmm(J.transpose(1, 2), J)
        eye = torch.eye(6, device=J.device, dtype=J.dtype)[None]
        A = JtJ + GN_DAMPING * eye
        cond = torch.linalg.cond(A)
        step_ok = torch.isfinite(cond) & (cond < GN_COND_MAX) \
            & (camera[..., 2].min(dim=1).values > 0)
        try:
            delta = torch.linalg.solve(A, -torch.bmm(J.transpose(1, 2), residual))
        except Exception:
            valid = valid & False
            break
        delta = delta.reshape(-1, 6)
        step_ok = step_ok & torch.isfinite(delta).all(dim=1)
        norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta = delta * (GN_DELTA_CLIP / norm).clamp(max=1.0)
        keep = step_ok[:, None].to(delta.dtype)
        rvec = rvec + delta[:, :3] * keep
        tvec = tvec + delta[:, 3:] * keep
        valid = valid & step_ok
    return rvec, tvec, valid


def pose_consistency(observed, points, K, gt_rvec, gt_tvec, diagonal_3d,
                     diagonal_2d, frame_valid) -> dict[str, torch.Tensor]:
    """L_3d and L_reproj against GT geometry, reported separately."""
    rvec, tvec, ok = gauss_newton(observed, points, K,
                                  gt_rvec.clone(), gt_tvec.clone())
    usable = (frame_valid & ok).to(observed.dtype)
    if float(usable.sum()) == 0:
        zero = torch.zeros((), device=observed.device, dtype=observed.dtype)
        return {"l3d": zero, "lreproj": zero, "usable": usable.sum()}
    rotation = rodrigues(rvec)
    gt_rotation = rodrigues(gt_rvec)
    predicted = torch.bmm(points, rotation.transpose(1, 2)) + tvec[:, None, :]
    target = torch.bmm(points, gt_rotation.transpose(1, 2)) + gt_tvec[:, None, :]
    distance = (predicted - target).norm(dim=-1) / diagonal_3d[:, None].clamp_min(1e-6)
    l3d = (F.huber_loss(distance, torch.zeros_like(distance), reduction="none",
                        delta=HUBER_DELTA).mean(dim=1) * usable).sum() / usable.sum()
    uv_pred, _ = project(rvec, tvec, points, K)
    uv_gt, _ = project(gt_rvec, gt_tvec, points, K)
    pixel = (uv_pred - uv_gt).norm(dim=-1) / diagonal_2d[:, None].clamp_min(1e-6)
    lreproj = (F.huber_loss(pixel, torch.zeros_like(pixel), reduction="none",
                            delta=HUBER_DELTA).mean(dim=1) * usable).sum() / usable.sum()
    return {"l3d": l3d, "lreproj": lreproj, "usable": usable.sum()}
