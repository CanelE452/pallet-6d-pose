"""Differentiable joint point+line pose refinement (unrolled Levenberg-Marquardt).

Point residuals use a Huber IRLS weight. Line residuals use a mixture over the
predicted modes of each structural edge, and BOTH endpoints of an edge share the
same mode responsibility -- an endpoint may not pick its own convenient line.
Everything is expressed in raw image pixels normalised by the image diagonal.
"""
from __future__ import annotations
import torch
from torch import Tensor
from .geometry import EDGES, corners, exp_so3, project, project_jacobian


def _line_distance(uv: Tensor, lines: Tensor) -> Tensor:
    """uv [B,E,2], lines [B,E,M,3] homogeneous -> [B,E,M] absolute distance."""
    n = lines[..., :2]
    norm = torch.linalg.vector_norm(n, dim=-1).clamp_min(1e-8)
    return ((uv[:, :, None, :] * n).sum(-1) + lines[..., 2]).abs() / norm


def energy(K, R, t, dims, uv_obs, point_mask, lines, line_mass, line_mask,
           diag, *, huber=.02, sigma=.02):
    """Robust objective actually being minimised.

    Returns ``(energy [B], valid [B])``. A candidate whose object falls behind
    the camera is reported invalid rather than being given a huge finite
    penalty: an arbitrary constant would dominate any softmax over candidates.
    """
    X = corners(dims)
    uv, z = project(K, R, t, X)
    scale = diag[:, None]
    r = torch.linalg.vector_norm(uv[:, :9] - uv_obs, dim=-1) / scale
    m = point_mask.to(r.dtype)
    pt = torch.where(r <= huber, .5 * r ** 2, huber * (r - .5 * huber))
    total = (pt * m).sum(-1)
    if lines is not None:
        u = torch.tensor([e[0] for e in EDGES], device=uv.device)
        v = torch.tensor([e[1] for e in EDGES], device=uv.device)
        d1 = _line_distance(uv[:, u], lines) / scale[..., None]
        d2 = _line_distance(uv[:, v], lines) / scale[..., None]
        sq = (d1 ** 2 + d2 ** 2) / (2 * sigma ** 2)
        # Mixture weights are a per-edge distribution. Without this, adding any
        # extra mode would lower the energy for free.
        logw = torch.log_softmax(torch.log(line_mass.clamp_min(1e-12)), dim=-1)
        mix = -torch.logsumexp(logw - sq, dim=-1)
        total = total + (mix * line_mask.to(mix.dtype)).sum(-1)
    valid = (z[:, :9] > 0).all(-1) & torch.isfinite(total)
    return total, valid


def refine(K, R0, t0, dims, uv_obs, point_mask, lines, line_mass, line_mask,
           diag, *, iters=5, huber=.02, sigma=.02, damping=1e-3):
    """Unrolled LM. Gradients flow to every input including the line tensors."""
    B = K.shape[0]
    R, t = R0, t0
    u_idx = torch.tensor([e[0] for e in EDGES], device=K.device)
    v_idx = torch.tensor([e[1] for e in EDGES], device=K.device)
    scale = diag[:, None]
    X = corners(dims)
    for _ in range(iters):
        uv, _ = project(K, R, t, X)
        J = project_jacobian(K, R, t, X)                       # [B,9+,2,6]
        res = (uv[:, :9] - uv_obs) / scale[..., None]
        Jp = J[:, :9] / scale[..., None, None]
        r = torch.linalg.vector_norm(res, dim=-1)
        w = torch.where(r <= huber, torch.ones_like(r), (huber / r.clamp_min(1e-9)))
        w = w * point_mask.to(w.dtype)
        A = (Jp.transpose(-1, -2) @ (w[..., None, None] * Jp)).sum(1)
        b = (Jp.transpose(-1, -2) @ (w[..., None, None] * res[..., None])).sum(1).squeeze(-1)
        if lines is not None:
            n = lines[..., :2]
            nn = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(1e-8)
            d1 = _line_distance(uv[:, u_idx], lines) / scale[..., None]
            d2 = _line_distance(uv[:, v_idx], lines) / scale[..., None]
            sq = (d1 ** 2 + d2 ** 2) / (2 * sigma ** 2)
            # One responsibility per (edge, mode); shared by both endpoints.
            logw = torch.log_softmax(torch.log(line_mass.clamp_min(1e-12)), dim=-1)
            resp = torch.softmax(logw - sq, dim=-1)
            resp = resp * line_mask[..., None].to(resp.dtype)
            sgn1 = torch.sign(((uv[:, u_idx][:, :, None, :] * n).sum(-1) + lines[..., 2]))
            sgn2 = torch.sign(((uv[:, v_idx][:, :, None, :] * n).sum(-1) + lines[..., 2]))
            for idx, dd, sg in ((u_idx, d1, sgn1), (v_idx, d2, sgn2)):
                Je = J[:, idx] / scale[..., None, None]        # [B,12,2,6]
                # d(distance)/d(pose) = sign * n_hat^T * d(uv)/d(pose)
                row = (sg[..., None, None] * nn[:, :, :, None, :] @ Je[:, :, None]).squeeze(-2)
                A = A + torch.einsum('bem,bemi,bemj->bij', resp / sigma ** 2, row, row)
                b = b + torch.einsum('bem,bem,bemi->bi', resp / sigma ** 2, dd, row)
        eye = torch.eye(6, dtype=A.dtype, device=A.device).expand(B, 6, 6)
        step = torch.linalg.solve(A + damping * eye * (1 + A.diagonal(dim1=-2, dim2=-1).mean(-1)
                                                       [:, None, None]), -b[..., None]).squeeze(-1)
        R = exp_so3(step[:, :3]) @ R
        t = t + step[:, 3:]
    return R, t


def select(K, R, t, dims, uv_obs, point_mask, lines, line_mass, line_mask, diag, **kw):
    """Pick one pose per item from candidate starts by observed energy alone.

    ``R``/``t`` are [B,S,...]. No ground truth is consulted. Candidates whose
    energy is non-finite or whose object falls behind the camera are rejected.
    """
    B, S = R.shape[:2]
    flat = lambda x: x.reshape(B * S, *x.shape[2:])
    rep = lambda x: x[:, None].expand(B, S, *x.shape[1:]).reshape(B * S, *x.shape[1:])
    e, ok = energy(rep(K), flat(R), flat(t), rep(dims), rep(uv_obs), rep(point_mask),
                   rep(lines) if lines is not None else None,
                   rep(line_mass) if lines is not None else None,
                   rep(line_mask) if lines is not None else None, rep(diag), **kw)
    e, ok = e.reshape(B, S), ok.reshape(B, S)
    masked = torch.where(ok, e, torch.full_like(e, float('inf')))
    best = masked.argmin(-1)
    rows = torch.arange(B, device=R.device)
    return R[rows, best], t[rows, best], e, best, ok[rows, best]
