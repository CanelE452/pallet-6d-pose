"""Corrected v2 objective with soft lines and direct correction utility."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import candidate_utility, continuous_line_targets


def _masked_mean(value: Tensor, mask: Tensor, dimension: int) -> Tensor:
    return (value * mask.to(value.dtype)).sum(dimension) / mask.sum(dimension).clamp_min(1)


def objective(output: dict[str, Tensor], batch: dict, model,
              weights: dict[str, float]) -> dict[str, Tensor]:
    target = batch["target_points"].to(output["points"].device)
    valid = batch["target_valid"].to(output["points"].device)
    base = batch["base_points"].to(output["points"].device)
    box = batch["box"].to(output["points"].device)
    image_hw = batch["image_hw"].to(output["points"].device)
    point_sigma = batch["point_sigma"].to(output["points"].device)
    diagonal = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1)
    candidates, components = [], []
    for batch_index, permutations in enumerate(batch["symmetry_permutations"]):
        sample, sample_components = [], []
        for permutation in permutations.to(target.device):
            gt = target[batch_index:batch_index + 1, permutation]
            gt_valid = valid[batch_index:batch_index + 1, permutation]
            safe_gt = torch.where(gt_valid[..., None], gt, base[batch_index:batch_index + 1].detach())
            norm = diagonal[batch_index:batch_index + 1, None]
            corrected_distance = torch.linalg.vector_norm(
                output["points"][batch_index:batch_index + 1] - safe_gt, dim=-1)
            base_distance = torch.linalg.vector_norm(
                base[batch_index:batch_index + 1] - safe_gt, dim=-1)
            corner_valid = gt_valid[:, :8]
            point = _masked_mean(F.smooth_l1_loss(
                output["points"][batch_index:batch_index + 1, :8] / norm[..., None],
                safe_gt[:, :8] / norm[..., None], reduction="none").sum(-1), corner_valid, 1)
            no_harm = _masked_mean(
                F.relu(corrected_distance[:, :8] - base_distance[:, :8]) / norm,
                corner_valid, 1)
            soft, _, support, _ = continuous_line_targets(
                safe_gt, gt_valid, box[batch_index:batch_index + 1], model.lattice,
                output["line_valid"][batch_index:batch_index + 1])
            log_probability = F.log_softmax(output["line_logits"][batch_index:batch_index + 1], -1)
            line = _masked_mean(-(soft * log_probability).sum(-1), support, 1)
            with torch.no_grad():
                utility_target, _, utility_valid = candidate_utility(
                    base[batch_index:batch_index + 1],
                    batch["point_valid"][batch_index:batch_index + 1].to(target.device),
                    point_sigma[batch_index:batch_index + 1],
                    output["raw_line"][batch_index:batch_index + 1].detach(),
                    safe_gt, gt_valid, .005 * diagonal[batch_index:batch_index + 1],
                    image_hw[batch_index:batch_index + 1])
                utility_valid = utility_valid & support
            utility_raw = F.binary_cross_entropy_with_logits(
                output["utility_logit"][batch_index:batch_index + 1], utility_target,
                reduction="none")
            utility = _masked_mean(utility_raw, utility_valid, 1)
            values = {"point": point, "line": line, "utility": utility, "no_harm": no_harm}
            total = sum(float(weights[name]) * value for name, value in values.items())
            sample.append(total.squeeze(0)); sample_components.append({k: v.squeeze(0) for k, v in values.items()})
        candidates.append(torch.stack(sample)); components.append(sample_components)
    chosen = torch.stack([value.detach().argmin() for value in candidates])
    result = {"loss": torch.stack([candidates[i][chosen[i]] for i in range(len(candidates))]).mean(),
              "symmetry_choice": chosen}
    for name in weights:
        result[name] = torch.stack([components[i][int(chosen[i])][name] for i in range(len(components))]).mean()
    return result
