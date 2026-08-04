"""Symmetry-aware corner-role objective.

The label assignment is chosen once per frame -- either the identity or the
yaw+180 permutation -- and every term in the frame uses that same choice.  A
per-corner minimum would silently permit a mixture no rigid pose can produce.

None of these terms touches a belief map.  They act on the role score, so a
wrong belief peak is discouraged by making its feature stop looking like the
corner it claims to be, not by pushing the peak down.
"""
from __future__ import annotations

import sys
import pathlib

import torch
import torch.nn.functional as F

_COMMON = pathlib.Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
import corner_role_adapter as CRA  # noqa: E402

N_CORNERS = CRA.N_CORNERS
MARGIN_CROSS = 0.20
MARGIN_WRONG = 0.20
LOCAL_SLACK = 0.10
WRONG_MIN_CELLS = 4.0
CLOSE_PAIR_CELLS = 1.5
SUBTERMS = ("proto", "cross", "wrong", "teacher_wrong", "local")


def _gather_assigned(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """scores B x N x 8, labels B x N -> B x N score of the assigned corner."""
    return scores.gather(2, labels[..., None].clamp(min=0)).squeeze(-1)


def choose_assignment(score_maps: torch.Tensor, points: torch.Tensor,
                      valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per frame, pick identity or yaw+180 by whichever CE is lower.

    Returns (labels B x N, used_swap B).
    """
    batch, corners = valid.shape
    sampled = CRA.bilinear_sample(score_maps, points)          # B x N x 8
    identity = torch.arange(corners, device=valid.device)[None].expand(batch, -1)
    swapped = CRA.apply_permutation(identity)
    weight = valid.float()

    def frame_cost(labels):
        flat = F.cross_entropy(sampled.reshape(-1, N_CORNERS),
                               labels.reshape(-1).clamp(0, N_CORNERS - 1),
                               reduction="none").reshape(batch, corners)
        return (flat * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    use_swap = frame_cost(swapped) < frame_cost(identity)
    labels = torch.where(use_swap[:, None], swapped, identity)
    return labels, use_swap


def prototype_loss(score_maps, points, valid, labels) -> torch.Tensor:
    sampled = CRA.bilinear_sample(score_maps, points)
    weight = valid.float()
    flat = F.cross_entropy(sampled.reshape(-1, N_CORNERS),
                           labels.reshape(-1).clamp(0, N_CORNERS - 1),
                           reduction="none").reshape(valid.shape)
    return (flat * weight).sum() / weight.sum().clamp_min(1.0)


def cross_location_loss(score_maps, points, valid, labels) -> torch.Tensor:
    """Corner i's own score must beat its score at another corner's location."""
    sampled = CRA.bilinear_sample(score_maps, points)           # B x N x 8
    own = _gather_assigned(sampled, labels)                     # B x N
    batch, corners = valid.shape
    distance = torch.cdist(points, points)                      # B x N x N
    pair = (valid[:, :, None] & valid[:, None, :]
            & (distance > CLOSE_PAIR_CELLS)
            & ~torch.eye(corners, dtype=torch.bool,
                         device=valid.device)[None])
    # score of corner i evaluated at corner j's location.
    # sampled is [b, j, k]; index the role axis by labels[i] so the result is
    # [b, j, i] and then transpose.  Indexing the location axis instead would
    # silently reproduce `own` and make the loss a constant.
    index = labels[:, None, :].expand(batch, corners, corners)   # [b, j, i]
    at_other = sampled.gather(2, index).transpose(1, 2)          # [b, i, j]
    loss = F.softplus(at_other - own[:, :, None] + MARGIN_CROSS)
    return (loss * pair.float()).sum() / pair.float().sum().clamp_min(1.0)


def peak_coordinates(belief: torch.Tensor) -> torch.Tensor:
    """Detached argmax per corner channel, in belief-grid coordinates."""
    batch, corners, height, width = belief.shape
    flat = belief.reshape(batch, corners, -1).argmax(dim=-1)
    return torch.stack([(flat % width).float(), (flat // width).float()],
                       dim=-1).detach()


def wrong_peak_loss(score_maps, points, valid, labels, peaks,
                    exclude: torch.Tensor | None = None) -> torch.Tensor:
    """The assigned role must score higher at GT than at a far wrong peak."""
    sampled_gt = CRA.bilinear_sample(score_maps, points)
    sampled_wrong = CRA.bilinear_sample(score_maps, peaks)
    score_gt = _gather_assigned(sampled_gt, labels)
    score_wrong = _gather_assigned(sampled_wrong, labels)
    distance = torch.linalg.norm(peaks - points, dim=-1)
    usable = valid & (distance > WRONG_MIN_CELLS)
    if exclude is not None:
        usable = usable & ~exclude
    loss = F.softplus(score_wrong - score_gt + MARGIN_WRONG)
    return (loss * usable.float()).sum() / usable.float().sum().clamp_min(1.0)


def local_consistency_loss(score_maps, points, valid, labels) -> torch.Tensor:
    """Role identity must not collapse onto a single subpixel cell."""
    offsets = torch.tensor([[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                            if not (dx == 0 and dy == 0)],
                           dtype=points.dtype, device=points.device)
    centre = _gather_assigned(CRA.bilinear_sample(score_maps, points), labels)
    total = torch.zeros((), device=points.device, dtype=points.dtype)
    weight = valid.float()
    for offset in offsets:
        shifted = points + offset[None, None, :]
        inside = CRA.valid_corner_mask(shifted, weight)
        # a neighbour that lands on another GT corner is not evidence
        distance = torch.cdist(shifted, points)
        collides = (distance <= CLOSE_PAIR_CELLS).float()
        collides = collides - torch.eye(points.shape[1], device=points.device
                                        )[None] * collides
        keep = (inside & (collides.sum(dim=-1) == 0)).float() * weight
        neighbour = _gather_assigned(CRA.bilinear_sample(score_maps, shifted),
                                     labels)
        term = F.relu(centre.detach() - LOCAL_SLACK - neighbour)
        total = total + (term * keep).sum() / keep.sum().clamp_min(1.0)
    return total / len(offsets)


def teacher_anchor_loss(student_stages, teacher_stages, channel_mask,
                        hard_tail: torch.Tensor) -> torch.Tensor:
    """Channel-masked MSE to the frozen teacher, relaxed on its own failures."""
    weight = channel_mask.clone().float()
    weight[:, :N_CORNERS] = weight[:, :N_CORNERS] * torch.where(
        hard_tail, torch.full_like(hard_tail, 0.25, dtype=weight.dtype),
        torch.ones_like(hard_tail, dtype=weight.dtype))
    total = torch.zeros((), device=weight.device)
    for student, teacher in zip(student_stages, teacher_stages):
        error = (student - teacher.detach()) ** 2
        per_channel = error.mean(dim=(-2, -1))
        total = total + (per_channel * weight).sum() / weight.sum().clamp_min(1.0)
    return total / max(len(student_stages), 1)


class CornerRoleObjective:
    """All role subterms for one batch, reported separately for calibration."""

    def __call__(self, score_maps, points, valid, student_belief,
                 teacher_belief=None) -> dict[str, torch.Tensor]:
        labels, used_swap = choose_assignment(score_maps, points, valid)
        student_peaks = peak_coordinates(student_belief[:, :N_CORNERS])
        losses = {
            "proto": prototype_loss(score_maps, points, valid, labels),
            "cross": cross_location_loss(score_maps, points, valid, labels),
            "wrong": wrong_peak_loss(score_maps, points, valid, labels,
                                     student_peaks),
            "local": local_consistency_loss(score_maps, points, valid, labels),
        }
        if teacher_belief is None:
            losses["teacher_wrong"] = torch.zeros((), device=points.device)
        else:
            teacher_peaks = peak_coordinates(teacher_belief[:, :N_CORNERS])
            duplicate = (torch.linalg.norm(teacher_peaks - student_peaks,
                                           dim=-1) < 0.5)
            losses["teacher_wrong"] = wrong_peak_loss(
                score_maps, points, valid, labels, teacher_peaks,
                exclude=duplicate)
        losses["_labels"] = labels
        losses["_used_swap"] = used_swap
        return losses
