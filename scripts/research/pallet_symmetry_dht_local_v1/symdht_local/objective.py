"""Whole-object symmetry-aware training objective."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .constants import EDGES


def raw_to_grid(points: Tensor, box: Tensor) -> Tensor:
    scale = (box[:, None, 2:] - box[:, None, :2]).clamp_min(1e-6)
    return 2 * (points - box[:, None, :2]) / scale - 1


def segment_intersects_square(a: Tensor, b: Tensor) -> Tensor:
    """Liang-Barsky segment intersection with the closed [-1,1]^2 square."""
    direction = b - a
    low = torch.zeros(a.shape[:-1], device=a.device, dtype=a.dtype)
    high = torch.ones_like(low)
    okay = torch.ones_like(low, dtype=torch.bool)
    for p, q in ((-direction[..., 0], a[..., 0] + 1),
                 (direction[..., 0], 1 - a[..., 0]),
                 (-direction[..., 1], a[..., 1] + 1),
                 (direction[..., 1], 1 - a[..., 1])):
        parallel = p.abs() < 1e-12
        okay = okay & (~parallel | (q >= 0))
        ratio = q / torch.where(parallel, torch.ones_like(p), p)
        low = torch.where((p < 0) & ~parallel, torch.maximum(low, ratio), low)
        high = torch.where((p > 0) & ~parallel, torch.minimum(high, ratio), high)
    return okay & (low <= high)


def line_targets(points: Tensor, valid: Tensor, box: Tensor, lattice) -> tuple[Tensor, Tensor]:
    safe_points = torch.where(valid[..., None], points, torch.zeros_like(points))
    grid = raw_to_grid(safe_points, box)
    endpoints = torch.as_tensor(EDGES, device=points.device)
    a, b = grid[:, endpoints[:, 0]], grid[:, endpoints[:, 1]]
    edge_valid = valid[:, endpoints].all(-1)
    segment = b - a
    edge_valid = edge_valid & (torch.linalg.vector_norm(segment, dim=-1) > 1e-6)
    edge_valid = edge_valid & segment_intersects_square(a, b)
    homogeneous_a = torch.cat((a, torch.ones_like(a[..., :1])), -1)
    homogeneous_b = torch.cat((b, torch.ones_like(b[..., :1])), -1)
    line = torch.linalg.cross(homogeneous_a, homogeneous_b)
    line = line / torch.linalg.vector_norm(line[..., :2], dim=-1, keepdim=True).clamp_min(1e-8)
    candidates = lattice.lines.to(points.dtype)
    dot = torch.einsum("beq,mq->bem", line[..., :2], candidates[:, :2])
    sign = torch.where(dot >= 0, 1., -1.)
    angle_cost = 1 - dot.abs().clamp_max(1)
    offset_cost = (candidates[None, None, :, 2] * sign - line[..., 2, None]).square()
    index = (angle_cost + offset_cost).argmin(-1)
    return index, edge_valid


def _masked_mean(value: Tensor, mask: Tensor, dimension: int) -> Tensor:
    return (value * mask.to(value.dtype)).sum(dimension) / mask.sum(dimension).clamp_min(1)


def objective(output: dict[str, Tensor], batch: dict, model,
              weights: dict[str, float]) -> dict[str, Tensor]:
    """Choose exactly one explicit permutation per sample for every loss term."""
    target = batch["target_points"].to(output["points"].device)
    valid = batch["target_valid"].to(output["points"].device)
    box = batch["box"].to(output["points"].device)
    base = batch["base_points"].to(output["points"].device)
    image_hw = batch["image_hw"].to(output["points"].device)
    diagonal = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1)
    candidates = []
    components = []
    for batch_index, permutations in enumerate(batch["symmetry_permutations"]):
        sample = []
        sample_components = []
        for permutation in permutations.to(target.device):
            gt = target[batch_index:batch_index + 1, permutation]
            gt_valid = valid[batch_index:batch_index + 1, permutation]
            safe_gt = torch.where(gt_valid[..., None], gt,
                                  base[batch_index:batch_index + 1].detach())
            norm = diagonal[batch_index:batch_index + 1, None]
            corrected_distance = torch.linalg.vector_norm(output["points"][batch_index:batch_index + 1] - safe_gt, dim=-1)
            base_distance = torch.linalg.vector_norm(base[batch_index:batch_index + 1] - safe_gt, dim=-1)
            corner_valid = gt_valid[:, :8]
            point = _masked_mean(F.smooth_l1_loss(
                output["points"][batch_index:batch_index + 1, :8] / norm[..., None],
                safe_gt[:, :8] / norm[..., None], reduction="none").sum(-1), corner_valid, 1)
            no_harm = _masked_mean(F.relu(corrected_distance[:, :8] - base_distance[:, :8]) / norm,
                                   corner_valid, 1)
            target_index, support = line_targets(safe_gt, gt_valid, box[batch_index:batch_index + 1], model.lattice)
            logits = output["line_logits"][batch_index:batch_index + 1]
            lattice_valid = output["line_valid"][batch_index:batch_index + 1]
            supported_bin = lattice_valid.gather(-1, target_index[..., None]).squeeze(-1)
            support = support & supported_bin
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_index.reshape(-1), reduction="none")
            line = _masked_mean(ce.reshape(1, 12), support, 1)
            null_target = (~support).to(logits.dtype)
            null = F.binary_cross_entropy_with_logits(
                output["null_logit"][batch_index:batch_index + 1], null_target, reduction="none").mean(1)
            anchor = torch.linalg.vector_norm(output["correction"][batch_index:batch_index + 1, :8], dim=-1).mean(1) / norm.squeeze(1)
            values = {"point": point, "line": line, "null": null,
                      "no_harm": no_harm, "anchor": anchor}
            total = sum(float(weights[name]) * value for name, value in values.items())
            sample.append(total.squeeze(0))
            sample_components.append({name: value.squeeze(0) for name, value in values.items()})
        candidates.append(torch.stack(sample))
        components.append(sample_components)
    chosen = torch.stack([value.detach().argmin() for value in candidates])
    selected_total = torch.stack([candidates[i][chosen[i]] for i in range(len(candidates))])
    result = {"loss": selected_total.mean(), "symmetry_choice": chosen}
    for name in weights:
        selected = torch.stack([components[i][int(chosen[i])][name] for i in range(len(components))])
        result[name] = selected.mean()
    return result
