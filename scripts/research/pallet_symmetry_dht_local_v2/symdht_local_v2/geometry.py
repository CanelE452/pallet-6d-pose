"""Continuous targets and one-alternative-per-edge local fusion."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.geometry import (
    clip_vector_norm, lines_grid_to_raw,
)
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.objective import (
    raw_to_grid, segment_intersects_square,
)

UTILITY_GAIN_THRESHOLD_PX = 0.25


def continuous_line_targets(points: Tensor, valid: Tensor, box: Tensor, lattice,
                            candidate_valid: Tensor | None = None):
    """Return fixed-bandwidth soft targets plus continuous GT grid lines.

    Bandwidth is exactly one lattice interval on each axis.  The target is
    frozen before performance and does not change the 36x65 lattice.
    """
    safe = torch.where(valid[..., None], points, torch.zeros_like(points))
    grid = raw_to_grid(safe, box)
    endpoints = torch.as_tensor(EDGES, device=points.device)
    a, b = grid[:, endpoints[:, 0]], grid[:, endpoints[:, 1]]
    support = valid[:, endpoints].all(-1)
    segment = b - a
    support = support & (torch.linalg.vector_norm(segment, dim=-1) > 1e-6)
    support = support & segment_intersects_square(a, b)
    ah = torch.cat((a, torch.ones_like(a[..., :1])), -1)
    bh = torch.cat((b, torch.ones_like(b[..., :1])), -1)
    line = torch.linalg.cross(ah, bh)
    line = line / torch.linalg.vector_norm(line[..., :2], dim=-1, keepdim=True).clamp_min(1e-8)

    candidates = lattice.lines.to(device=points.device, dtype=points.dtype)
    dot = torch.einsum("beq,mq->bem", line[..., :2], candidates[:, :2])
    sign = torch.where(dot >= 0, 1., -1.)
    angle = torch.acos(dot.abs().clamp(0, 1))
    offset = candidates[None, None, :, 2] * sign - line[..., 2, None]
    angle_step = math.pi / lattice.theta_bins
    rho_step = 2 * math.sqrt(2) / (lattice.rho_bins - 1)
    scores = -.5 * ((angle / angle_step).square() + (offset / rho_step).square())
    if candidate_valid is not None:
        scores = scores.masked_fill(~candidate_valid, -1e4)
        support = support & candidate_valid.any(-1)
    distribution = torch.softmax(scores, -1)
    hard_index = scores.argmax(-1)
    return distribution, hard_index, support, line


def decode_single_mode(logits: Tensor, valid: Tensor, lattice, box: Tensor):
    """Decode one MAP-local continuous alternative for each semantic edge."""
    masked = logits.masked_fill(~valid, -1e4)
    probability = torch.softmax(masked, -1) * valid.to(logits.dtype)
    anchor_index = masked.argmax(-1)
    base = lattice.lines.to(device=logits.device, dtype=logits.dtype)
    anchor = base[anchor_index]
    dot = torch.einsum("beq,mq->bem", anchor[..., :2], base[:, :2])
    sign = torch.where(dot >= 0, 1., -1.)
    angle_bw = math.pi / lattice.theta_bins * 1.5
    rho_bw = 2 * math.sqrt(2) / (lattice.rho_bins - 1) * 1.5
    distance = ((1 - dot.abs().clamp_max(1)) / (1 - math.cos(angle_bw))
                + (base[None, None, :, 2] * sign - anchor[..., 2, None]).square() / rho_bw ** 2)
    window = (distance <= 1) & valid
    available = (masked.max(-1).values > -9000) & window.any(-1)
    local = torch.softmax((logits - .5 * distance).masked_fill(~window, -1e4), -1)
    local = local * window
    local = local / local.sum(-1, keepdim=True).clamp_min(1e-12)
    grid_line = torch.einsum("bem,bem,mq->beq", local, sign, base)
    grid_line = grid_line / torch.linalg.vector_norm(grid_line[..., :2], dim=-1, keepdim=True).clamp_min(1e-8)
    raw_line, normal_ok = lines_grid_to_raw(grid_line[:, :, None], box)
    raw_line = raw_line[:, :, 0]
    available = available & normal_ok[:, :, 0]
    mass = (probability * window).sum(-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    count = valid.sum(-1).clamp_min(2).to(logits.dtype)
    ambiguity = (entropy / count.log()).clamp(0, 1)
    return {
        "raw_line": raw_line, "available": available, "mode_mass": mass,
        "anchor_index": anchor_index, "ambiguity": ambiguity,
        "conditional_probability": probability,
    }


def single_mode_wls(base_points: Tensor, point_valid: Tensor, point_sigma: Tensor,
                    lines: Tensor, utility: Tensor, line_sigma: Tensor,
                    image_hw: Tensor, *, max_shift_fraction: float = .01):
    """Fuse at most one alternative per semantic edge; never sum edge modes."""
    bsz = base_points.shape[0]
    dtype, device = base_points.dtype, base_points.device
    eye = torch.eye(2, dtype=dtype, device=device).expand(bsz, 8, 2, 2)
    matrix = eye * point_sigma[:, :8].clamp_min(1e-8).square().reciprocal()[..., None, None]
    rhs = torch.zeros((bsz, 8, 2), dtype=dtype, device=device)
    inv_line_var = line_sigma.clamp_min(1e-8).square().reciprocal()
    for edge_index, (a, b) in enumerate(EDGES):
        normal, offset = lines[:, edge_index, :2], lines[:, edge_index, 2]
        weight = utility[:, edge_index] * inv_line_var
        outer = normal[..., :, None] * normal[..., None, :]
        contribution = weight[..., None, None] * outer
        for corner in (a, b):
            signed = (normal * base_points[:, corner]).sum(-1) + offset
            force = -(weight * signed)[..., None] * normal
            matrix[:, corner] = matrix[:, corner] + contribution
            rhs[:, corner] = rhs[:, corner] + force
    delta8 = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    diagonal = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1)
    delta8 = clip_vector_norm(delta8, (max_shift_fraction * diagonal)[:, None, None])
    delta8 = torch.where(point_valid[:, :8, None], delta8, torch.zeros_like(delta8))
    delta = torch.cat((delta8, torch.zeros_like(base_points[:, 8:9])), 1)
    corrected = torch.cat((base_points[:, :8] + delta8, base_points[:, 8:9]), 1)
    return corrected, delta


def candidate_utility(base_points: Tensor, point_valid: Tensor, point_sigma: Tensor,
                      lines: Tensor, target: Tensor, target_valid: Tensor,
                      line_sigma: Tensor, image_hw: Tensor,
                      threshold_px: float = UTILITY_GAIN_THRESHOLD_PX):
    """Detached one-edge counterfactual gain used only for synthetic labels/audit."""
    target = torch.where(target_valid[..., None], target, base_points)
    gains = torch.zeros(base_points.shape[0], len(EDGES), device=base_points.device,
                        dtype=base_points.dtype)
    usable = torch.zeros_like(gains, dtype=torch.bool)
    diagonal = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1)
    for edge_index, (a, b) in enumerate(EDGES):
        corners = [a, b]
        normal, offset = lines[:, edge_index, :2], lines[:, edge_index, 2]
        points = base_points[:, corners]
        signed = (points * normal[:, None]).sum(-1) + offset[:, None]
        point_var = point_sigma[:, corners].clamp_min(1e-8).square()
        line_var = line_sigma[:, None].clamp_min(1e-8).square()
        fraction = point_var / (point_var + line_var)
        delta = -signed[..., None] * fraction[..., None] * normal[:, None]
        delta = clip_vector_norm(delta, (.01 * diagonal)[:, None, None])
        corrected = points + delta
        mask = target_valid[:, corners] & point_valid[:, corners]
        before = torch.linalg.vector_norm(points - target[:, corners], dim=-1)
        after = torch.linalg.vector_norm(corrected - target[:, corners], dim=-1)
        count = mask.sum(-1)
        gain = ((before - after) * mask).sum(-1) / count.clamp_min(1)
        gains[:, edge_index] = gain
        usable[:, edge_index] = count > 0
    positive = usable & (gains > threshold_px)
    return positive.to(base_points.dtype), gains, usable
