"""Symmetry-aware training objective.

One symmetry element g is chosen per object by the total loss -- not per point,
not per term. Point, line, pose, seed and ranking all read the same permuted
ground truth, so an equivalent state never receives contradictory supervision.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F
from .geometry import EDGES, corners, symmetry_permutations, yaw


def line_targets(points: Tensor, box: Tensor, thetas: int, rhos: int) -> tuple[Tensor, Tensor]:
    """GT structural lines as (theta, rho) bins in ROI-normalised coordinates.

    An edge is supervised only where the GT line actually crosses the ROI. This
    is a support-and-intersection test, not a claim that the edge is visible.
    """
    centre = (box[:, :2] + box[:, 2:]) * .5
    half = (box[:, 2:] - box[:, :2]).clamp_min(1e-6) * .5
    p = (points[:, :8] - centre[:, None]) / half[:, None]
    u = torch.tensor([e[0] for e in EDGES], device=points.device)
    v = torch.tensor([e[1] for e in EDGES], device=points.device)
    a, b = p[:, u], p[:, v]
    d = b - a
    n = torch.stack((-d[..., 1], d[..., 0]), -1)
    ln = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    ok = ln[..., 0] > 1e-6
    n = n / ln.clamp_min(1e-6)
    theta = torch.atan2(n[..., 1], n[..., 0]) % math.pi
    rho = (n * a).sum(-1)
    flip = ((torch.atan2(n[..., 1], n[..., 0]) % (2 * math.pi)) >= math.pi)
    rho = torch.where(flip, -rho, rho)                 # theta seam flips the rho sign
    t_idx = (theta / (math.pi / thetas)).round().long() % thetas
    rho_max = math.sqrt(2.)
    r_idx = ((rho + rho_max) / (2 * rho_max) * (rhos - 1)).round().long()
    inside = (r_idx >= 0) & (r_idx < rhos) & ok & (rho.abs() <= rho_max)
    return (t_idx * rhos + r_idx).clamp(0, thetas * rhos - 1), inside


def _permute(points, valid, R, dims, g, order):
    """Apply symmetry element g: GT points reorder, GT rotation absorbs the yaw."""
    perms = symmetry_permutations(order)
    idx = torch.tensor(perms[g % order], device=points.device)
    Rg = yaw(g * (4 // order), dtype=R.dtype, device=R.device)
    return points[:, idx], valid[:, idx], R @ Rg.T


def evaluate_group(model_out, obs, sup, order_mask, g, solver, cfg):
    """Loss terms for one symmetry element, returned per item."""
    from .solver import refine, energy
    pts, valid, Rg = _permute(sup['points'], sup['valid'], sup['R'], obs['dims'], g, 2)
    Rg = torch.where(order_mask[:, None, None], Rg, sup['R'])
    pts = torch.where(order_mask[:, None, None], pts, sup['points'])
    valid = torch.where(order_mask[:, None], valid, sup['valid'])
    diag = torch.linalg.vector_norm(obs['image_hw'].flip(-1), dim=-1)
    b = pts.shape[0]

    point = (F.smooth_l1_loss(model_out['points'] / diag[:, None, None],
                              pts / diag[:, None, None], beta=.01, reduction='none')
             .sum(-1) * valid).sum(-1) / valid.sum(-1).clamp_min(1)

    line = torch.zeros(b, device=pts.device)
    if model_out['line_logits'] is not None:
        tgt, ok = line_targets(pts, obs['box'], cfg['thetas'], cfg['rhos'])
        logp = F.log_softmax(model_out['line_logits'], -1)
        picked = logp.gather(-1, tgt[..., None]).squeeze(-1)
        line = -(picked * ok).sum(-1) / ok.sum(-1).clamp_min(1)

    S = model_out['R0'].shape[1]
    X = corners(obs['dims'])
    gtc = X @ Rg.transpose(-1, -2) + sup['t'][:, None]
    obj = torch.linalg.vector_norm(obs['dims'], dim=-1)
    seed_err = (torch.linalg.vector_norm(
        X[:, None] @ model_out['R0'].transpose(-1, -2) + model_out['t0'][:, :, None] - gtc[:, None],
        dim=-1)[..., :8].mean(-1) / obj[:, None])
    seed = seed_err.min(-1).values

    rep = lambda x: x[:, None].expand(b, S, *x.shape[1:]).reshape(b * S, *x.shape[1:])
    flat = lambda x: x.reshape(b * S, *x.shape[2:])
    lines = model_out['lines']
    Rr, tr = refine(rep(obs['K']), flat(model_out['R0']), flat(model_out['t0']), rep(obs['dims']),
                    rep(model_out['points']), rep(obs['point_mask']),
                    rep(lines) if lines is not None else None,
                    rep(model_out['mass']) if lines is not None else None,
                    rep(obs['line_mask']) if lines is not None else None,
                    rep(diag), iters=cfg['iters'])
    e, ok = energy(rep(obs['K']), Rr, tr, rep(obs['dims']), rep(model_out['points']), rep(obs['point_mask']),
                   rep(lines) if lines is not None else None,
                   rep(model_out['mass']) if lines is not None else None,
                   rep(obs['line_mask']) if lines is not None else None, rep(diag))
    e, ok = e.reshape(b, S), ok.reshape(b, S)
    keep = torch.where(ok.any(-1, keepdim=True), ok, torch.ones_like(ok))
    err = (torch.linalg.vector_norm(
        rep(X) @ Rr.transpose(-1, -2) + tr[:, None] - rep(gtc), dim=-1)[..., :8].mean(-1)
        ).reshape(b, S) / obj[:, None]
    # The energy and the normalised corner error live on different scales, so
    # they get separate frozen temperatures. Invalid candidates are removed from
    # both distributions rather than being pushed down with a large constant.
    elog = (-e / cfg['energy_temperature']).masked_fill(~keep, -1e9)
    tlog = (-err.detach() / cfg['temperature']).masked_fill(~keep, -1e9)
    w = torch.softmax(elog, -1)
    pose = (w.detach() * err).sum(-1)
    rank = -(torch.softmax(tlog, -1) * torch.log_softmax(elog, -1)).sum(-1)
    total = (cfg['w_point'] * point + cfg['w_line'] * line + cfg['w_pose'] * pose
             + cfg['w_seed'] * seed + cfg['w_rank'] * rank)
    return total, dict(point=point, line=line, pose=pose, seed=seed, rank=rank,
                       best_corner_error=err.min(-1).values)


def symmetric_loss(model_out, obs, sup, orders, solver, cfg):
    """min over g of the joint objective; C4 is unreachable in this population."""
    order_mask = orders == 2
    cands = [evaluate_group(model_out, obs, sup, order_mask, g, solver, cfg) for g in (0, 1)]
    stacked = torch.stack([c[0] for c in cands], -1)
    best = stacked.argmin(-1)
    total = stacked.gather(-1, best[:, None]).squeeze(-1)
    parts = {k: torch.stack([c[1][k] for c in cands], -1).gather(-1, best[:, None]).squeeze(-1)
             for k in cands[0][1]}
    return total.mean(), {k: float(v.mean().detach()) for k, v in parts.items()}, best
