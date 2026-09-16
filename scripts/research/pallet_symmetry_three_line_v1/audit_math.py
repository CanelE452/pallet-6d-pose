"""New diagnostic math only; no repository I/O, model loading or training.
Coordinate contract: all 2D coordinates/line coefficients/sigma use the SAME
original-image pixel frame. Row-vector object coordinates are explicit inputs.
"""
from __future__ import annotations

import math
from typing import Iterable, Literal
import numpy as np
import torch


def validate_edges(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int64)
    if e.shape != (12, 2) or (e < 0).any() or (e >= 8).any():
        raise ValueError('Expected twelve edges between eight cuboid corners')
    keys = [tuple(sorted(row)) for row in e.tolist()]
    if len(set(keys)) != 12 or any(a == b for a, b in keys):
        raise ValueError('Edges must be unique and nondegenerate')
    incidence = np.zeros((8, 12), dtype=bool)
    for j, (a, b) in enumerate(e):
        incidence[[a, b], j] = True
    if not np.all(incidence.sum(1) == 3):
        raise ValueError('Each cuboid corner must have exactly THREE incident edges')
    # Connectedness: degree-three alone does not certify the cuboid topology.
    reached = {0}
    for _ in range(8):
        for a, b in e.tolist():
            if a in reached or b in reached:
                reached.update((a, b))
    if len(reached) != 8:
        raise ValueError('Disconnected graph')
    return incidence


def validate_group(permutations: np.ndarray, edges: np.ndarray) -> None:
    p = np.asarray(permutations, dtype=np.int64)
    validate_edges(edges)
    if p.ndim != 2 or p.shape[1] != 9 or len(p) not in (1, 2, 4):
        raise ValueError('Expected a C1/C2/C4 group of nine-index permutations')
    if not np.array_equal(p[0], np.arange(9)):
        raise ValueError('Identity must be first')
    group = {tuple(v.tolist()) for v in p}
    if len(group) != len(p):
        raise ValueError('Duplicate group elements')
    edge_set = {tuple(sorted(e)) for e in np.asarray(edges).tolist()}
    for a in p:
        if not np.array_equal(np.sort(a), np.arange(9)) or a[8] != 8:
            raise ValueError('Permutation must be bijective and preserve centre 8')
        if {tuple(sorted((int(a[i]), int(a[j])))) for i, j in edges} != edge_set:
            raise ValueError('Edge topology is not preserved')
        if tuple(np.argsort(a).tolist()) not in group:
            raise ValueError('Inverse missing')
        for b in p:
            if tuple(a[b].tolist()) not in group:
                raise ValueError('Group is not closed')


def derive_permutations(x3d: np.ndarray, rotations: np.ndarray,
                        edges: np.ndarray, relative_tolerance: float = 1e-6) -> np.ndarray:
    """Derive only from supplied, externally authorized physical rotations.
    NOT a symmetry detector. A square bounding box does not authorize C4.
    convention: (R @ (X_j-centre)) + centre == X_{permutation[j]}.
    """
    x = np.asarray(x3d, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if x.shape != (9, 3) or not np.isfinite(x).all():
        raise ValueError('Explicit eight corners and centre are required')
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError('Rotation matrix shape')
    tolerance = relative_tolerance * max(float(np.ptp(x[:8], axis=0).max()), 1e-9)
    if np.linalg.norm(x[8] - x[:8].mean(0)) > tolerance:
        raise ValueError('Index 8 must be the cuboid centroid')
    out = []
    for r in rotations:
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(r), 1, atol=1e-8):
            raise ValueError('Only proper rotations, no reflection')
        rotated = (x - x[8]) @ r.T + x[8]
        distances = np.linalg.norm(rotated[:, None] - x[None], axis=-1)
        perm = distances.argmin(1)
        if (distances[np.arange(9), perm] > tolerance).any():
            raise ValueError('Requested rotation does not preserve this physical geometry')
        out.append(perm)
    result = np.stack(out)
    validate_group(result, edges)
    return result


def edge_channel_permutations(perms: np.ndarray, edges: np.ndarray) -> np.ndarray:
    validate_group(perms, edges)
    lookup = {tuple(sorted(e)): j for j, e in enumerate(np.asarray(edges).tolist())}
    return np.asarray([[lookup[tuple(sorted((int(p[a]), int(p[b]))))]
                        for a, b in edges] for p in perms], dtype=np.int64)


def symmetry_error(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray,
                   permutations: np.ndarray, penalty: float,
                   pred_valid: np.ndarray | None = None) -> dict:
    """One whole-object branch for ALL reported 2D metrics.
    Uses all finite annotated GT corners; missing predictions incur penalty.
    GT coordinates AND masks move together. No GT is passed to inference.
    """
    pr, y = np.asarray(pred, float), np.asarray(gt, float)
    perms = np.asarray(permutations, dtype=np.int64)
    if pr.shape != (9, 2) or y.shape != (9, 2) or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Coordinates/penalty invalid')
    if perms.ndim != 2 or perms.shape[1] != 9:
        raise ValueError('Permutation shape')
    gy = np.asarray(gt_valid, bool) & np.isfinite(y).all(1)
    pp = np.isfinite(pr).all(1)
    if pred_valid is not None:
        pp &= np.asarray(pred_valid, bool)
    if gy[:8].sum() == 0:
        return {'evaluable': False, 'reason': 'NO_ANNOTATED_CORNERS'}
    targets = y[perms]
    masks = gy[perms]
    if not np.all(masks[:, :8].sum(1) == gy[:8].sum()):
        raise ValueError('Permutation changed GT denominator')
    # Sanitize BEFORE arithmetic, not a NaN-producing norm masked afterwards.
    diff = np.where(pp[None, :, None], pr[None], 0) - np.where(masks[..., None], targets, 0)
    distances = np.linalg.norm(diff, axis=-1)
    distances = np.where(pp[None], distances, penalty)
    distances = np.where(masks, distances, 0)
    means = distances[:, :8].sum(1) / masks[:, :8].sum(1)
    branch = int(np.argmin(means))
    return {'evaluable': True, 'branch': branch, 'fixed_mean': float(means[0]),
            'equivalent_mean': float(means[branch]), 'all_branch_means': means,
            'errors': distances[branch], 'gt_mask': masks[branch],
            'target': targets[branch], 'n_corners': int(gy[:8].sum())}


def conditional_line_distribution(logits: torch.Tensor, valid: torch.Tensor) -> tuple:
    """Entropy is an uncalibrated concentration diagnostic, NOT visibility."""
    if logits.ndim != 2 or logits.shape != valid.shape:
        raise ValueError('Expected line logits and valid mask [E,M]')
    mask = valid.bool() & torch.isfinite(logits)
    count = mask.sum(-1)
    active = count > 0
    masked = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    maximum = masked.max(-1).values
    maximum = torch.where(active, maximum, torch.zeros_like(maximum))
    unnormalized = torch.where(mask, torch.exp(masked - maximum[:, None]), torch.zeros_like(logits))
    q = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(q * q.clamp_min(1e-30).log()).sum(-1)
    denom = count.clamp_min(2).to(logits.dtype).log()
    ambiguity = torch.where(count > 1, entropy / denom, torch.ones_like(entropy)).clamp(0, 1)
    low = torch.where(mask, logits, torch.full_like(logits, torch.inf)).min(-1).values
    uniform = active & ((maximum - low).abs() <= 1e-7)
    ambiguity = torch.where(uniform, torch.ones_like(ambiguity), ambiguity)
    concentration = torch.where(active, 1 - ambiguity, torch.zeros_like(ambiguity))
    return q, ambiguity, concentration, active, mask


@torch.no_grad()
def line_candidate_evidence(locations: torch.Tensor, line_logits: torch.Tensor,
                            line_valid: torch.Tensor, lines_raw: torch.Tensor,
                            sigma_raw: torch.Tensor, edges: np.ndarray,
                            mode: Literal['three_equal','three_ambiguity','one_ambiguity']) -> dict:
    """[8,C,2] candidate locations; 12 SEMANTIC lines, each with M modes.
    All lattice modes are marginalized. 'three' means incident cuboid edges,
    NOT three highest Hough peaks. No WLS/intersection/new coordinates.
    """
    incidence = validate_edges(edges)
    if locations.ndim != 3 or locations.shape[0] != 8 or locations.shape[-1] != 2:
        raise ValueError('locations must have shape [8,C,2]')
    if mode not in {'three_equal', 'three_ambiguity', 'one_ambiguity'}:
        raise ValueError('Unknown ablation mode')
    if line_logits.shape[0] != 12 or lines_raw.shape != (line_logits.shape[1], 3):
        raise ValueError('Twelve semantic channels and explicit raw lattice lines required')
    if sigma_raw.shape != (line_logits.shape[1],):
        raise ValueError('sigma_raw[M] is required')
    norms = lines_raw[:, :2].norm(dim=-1)
    geometry_valid = (torch.isfinite(lines_raw).all(-1) & (norms > 1e-12)
                      & torch.isfinite(sigma_raw) & (sigma_raw > 0))
    unit = torch.nan_to_num(lines_raw) / torch.where(geometry_valid, norms, torch.ones_like(norms))[:, None]
    sigma = torch.where(geometry_valid, sigma_raw, torch.ones_like(sigma_raw))
    q, ambiguity, concentration, active, mask = conditional_line_distribution(
        line_logits, line_valid.bool() & geometry_valid[None])
    # EQUAL is a control; AMBIGUITY is the requested primary method.
    reliability = active.to(q.dtype) if mode == 'three_equal' else concentration
    scores, edge_parts = [], []
    for k in range(8):
        ids = np.flatnonzero(incidence[k])
        if mode == 'one_ambiguity':
            best = int(torch.argmax(reliability[ids]).item())  # ID-order tie break, no GT
            ids = ids[best:best+1]
        x = locations[k]
        good_x = torch.isfinite(x).all(-1)
        safe = torch.where(good_x[:, None], x, torch.zeros_like(x))
        signed = safe @ unit[:, :2].T + unit[:, 2]
        kernel = torch.exp(-.5 * (signed / sigma[None]).square())
        row, pieces = torch.zeros(len(x), dtype=q.dtype, device=q.device), []
        for e in ids.tolist():
            posterior_support = kernel @ q[e]
            background = (kernel * mask[e][None]).sum(-1) / mask[e].sum().clamp_min(1)
            # Uniform-prior correction prevents Hough lattice density bias.
            log_ratio = ((posterior_support + 1e-12) / (background + 1e-12)).log().clamp(-1, 1)
            # Flat / unavailable evidence is exactly neutral, not a new point prior.
            same = (posterior_support - background).abs() <= 1e-7 * (background.abs() + 1e-12)
            log_ratio = torch.where(same | ~good_x, torch.zeros_like(log_ratio), log_ratio)
            part = reliability[e] * log_ratio
            row += part
            pieces.append(part)
        divisor = 1 if mode == 'one_ambiguity' else 3
        scores.append(row / divisor)  # Do NOT divide by sum(reliability)!
        edge_parts.append(torch.stack(pieces))
    return {'bias': torch.stack(scores), 'edge_parts': edge_parts,
            'ambiguity': ambiguity, 'concentration': concentration,
            'line_available': active, 'conditional_probability': q}


@torch.no_grad()
def rerank_logits(point_logits: torch.Tensor, bias: torch.Tensor,
                  temperature: float, eta: float) -> torch.Tensor:
    if point_logits.shape != bias.shape or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Point logits/temperature invalid')
    if not math.isfinite(eta) or eta < 0 or eta > 1:
        raise ValueError('eta must be a finite scalar in [0,1]')
    if eta == 0:
        return point_logits.clone()  # Exact baseline bypass, no new decode.
    if not torch.isfinite(bias).all() or bias.abs().max() > 1.00001:
        raise ValueError('Three-line bias must be bounded in [-1,1]')
    # Existing decode divides by T. Multiplication avoids accidental double scaling.
    return point_logits + temperature * eta * bias
