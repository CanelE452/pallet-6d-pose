"""Coordinate transforms and the strictly local weighted least-squares fusion."""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import EDGES


def roi_grid_to_raw(box: Tensor) -> Tensor:
    """Return homogeneous affine mapping ROI coordinates [-1,1] to raw pixels."""
    box = box.reshape(-1, 4)
    x1, y1, x2, y2 = box.unbind(-1)
    z = torch.zeros_like(x1)
    o = torch.ones_like(x1)
    return torch.stack((
        torch.stack(((x2 - x1) / 2, z, (x2 + x1) / 2), -1),
        torch.stack((z, (y2 - y1) / 2, (y2 + y1) / 2), -1),
        torch.stack((z, z, o), -1),
    ), -2)


def lines_grid_to_raw(lines: Tensor, box: Tensor) -> tuple[Tensor, Tensor]:
    """Transform homogeneous grid lines to normalized raw-pixel lines."""
    affine = roi_grid_to_raw(box)
    # x_raw=A*x_grid, so l_raw=A^{-T}*l_grid.
    raw = torch.linalg.solve(affine.transpose(-1, -2)[:, None, None], lines.unsqueeze(-1)).squeeze(-1)
    norm = torch.linalg.vector_norm(raw[..., :2], dim=-1, keepdim=True)
    valid = torch.isfinite(raw).all(-1) & (norm.squeeze(-1) > 1e-10)
    raw = raw / norm.clamp_min(1e-10)
    return raw, valid


def clip_vector_norm(delta: Tensor, maximum: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return delta * torch.minimum(torch.ones_like(norm), maximum / norm.clamp_min(1e-12))


def local_wls_fusion(
    base_points: Tensor,
    point_valid: Tensor,
    point_sigma: Tensor,
    lines: Tensor,
    weights: Tensor,
    line_sigma: Tensor,
    image_hw: Tensor,
    *,
    max_shift_fraction: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Correct corners using incident infinite lines, keeping point 8 bit-exact.

    ``lines`` and ``weights`` have shape B,12,M,3 and B,12,M.  Invalid or
    unavailable modes must have zero weight.  No pose, camera or target enters.
    """
    if base_points.ndim != 3 or base_points.shape[1:] != (9, 2):
        raise ValueError("base_points must be [B,9,2]")
    bsz = base_points.shape[0]
    dtype, device = base_points.dtype, base_points.device
    eye = torch.eye(2, dtype=dtype, device=device).expand(bsz, 8, 2, 2)
    inv_point_var = point_sigma[:, :8].clamp_min(1e-8).square().reciprocal()
    matrix = eye * inv_point_var[..., None, None]
    rhs = torch.zeros((bsz, 8, 2), dtype=dtype, device=device)
    line_var = line_sigma.clamp_min(1e-8).square()
    for edge_index, (a, b) in enumerate(EDGES):
        normal = lines[:, edge_index, :, :2]
        offset = lines[:, edge_index, :, 2]
        w = weights[:, edge_index] / line_var[:, None]
        outer = normal[..., :, None] * normal[..., None, :]
        for corner in (a, b):
            signed = (normal * base_points[:, corner, None]).sum(-1) + offset
            matrix[:, corner] = matrix[:, corner] + (w[..., None, None] * outer).sum(1)
            rhs[:, corner] = rhs[:, corner] - (w * signed)[..., None].mul(normal).sum(1)
    delta8 = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    diagonal = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1)
    delta8 = clip_vector_norm(delta8, (max_shift_fraction * diagonal)[:, None, None])
    delta8 = torch.where(point_valid[:, :8, None], delta8, torch.zeros_like(delta8))
    corrected8 = base_points[:, :8] + delta8
    # Concatenation copies centre and makes the exact-preservation contract explicit.
    corrected = torch.cat((corrected8, base_points[:, 8:9]), 1)
    delta = torch.cat((delta8, torch.zeros_like(base_points[:, 8:9])), 1)
    return corrected, delta


def edge_graph_preserved(permutation: list[int] | tuple[int, ...]) -> bool:
    if sorted(permutation) != list(range(9)) or permutation[8] != 8:
        return False
    graph = {tuple(sorted(edge)) for edge in EDGES}
    mapped = {tuple(sorted((permutation[a], permutation[b]))) for a, b in EDGES}
    return mapped == graph
