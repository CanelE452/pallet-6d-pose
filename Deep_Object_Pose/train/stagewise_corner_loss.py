"""Stage-wise corner losses that attack sharpening-without-correction directly.

On the mechanism set the far-face error does not improve across the refinement
stages -- 21.42 px at stage 4 against 22.08 px at stage 6 -- while the peak rises
from 0.695 to 0.804.  A Gaussian MSE lowers the average residual over the whole
target and never has to pay for one tall peak in the wrong place, so these four
terms add what it is missing:

    mass      probability inside the GT 3x3 must go up
    rank      the best belief outside a GT-exclusion radius must lose to the GT
    distance  the whole probability mass, not just the peak, must move to the GT
    progress  a later stage may not be further away or hold less GT mass

Everything reads raw belief through a full-map softmax at temperature 0.1: no
sigmoid, no absolute threshold, and no local-window-only probability, because a
corner whose peak sits 30 cells away from the GT has to receive gradient.

The centroid channel is excluded throughout -- it keeps its legacy MSE only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_CORNERS = 8
TEMPERATURE = 0.1
GT_WINDOW = 1          # radius -> 3x3
GT_EXCLUSION = 4       # belief cells kept away from the GT for the wrong peak
RANK_MARGIN = 0.10     # belief units
STAGE_WEIGHTS = (0.25, 0.50, 1.00)
GRID_DIAGONAL = float((50 ** 2 + 50 ** 2) ** 0.5)


def valid_corners(centres: torch.Tensor, grid: int = 50) -> torch.Tensor:
    """A corner counts when its transformed GT centre lands inside the grid.

    The raster target being all zero is not the test: a truncated corner can
    produce an empty Gaussian while its centre is still on the map.
    """
    inside = ((centres[..., 0] >= 0) & (centres[..., 0] < grid)
              & (centres[..., 1] >= 0) & (centres[..., 1] < grid))
    return inside & torch.isfinite(centres).all(dim=-1)


def spatial_probability(belief: torch.Tensor) -> torch.Tensor:
    """Full-map softmax over the raw belief at temperature 0.1."""
    batch, corners, height, width = belief.shape
    flat = belief.reshape(batch, corners, -1) / TEMPERATURE
    return torch.softmax(flat, dim=-1).reshape(batch, corners, height, width)


def _grid_coordinates(belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = belief.shape[-2:]
    ys = torch.arange(height, device=belief.device, dtype=belief.dtype)
    xs = torch.arange(width, device=belief.device, dtype=belief.dtype)
    return xs[None, None, None, :], ys[None, None, :, None]


def gt_window_mask(belief: torch.Tensor, centres: torch.Tensor,
                   radius: int = GT_WINDOW) -> torch.Tensor:
    """Boolean 3x3 (cropped at the border) around each GT centre."""
    xs, ys = _grid_coordinates(belief)
    cx = centres[..., 0][..., None, None].round()
    cy = centres[..., 1][..., None, None].round()
    return ((xs - cx).abs() <= radius) & ((ys - cy).abs() <= radius)


def gt_mass(belief: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    probability = spatial_probability(belief)
    return (probability * gt_window_mask(belief, centres)).sum(dim=(-2, -1))


def mass_loss(stages: tuple[torch.Tensor, ...], centres: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), device=centres.device, dtype=centres.dtype)
    weight_sum = 0.0
    for belief, weight in zip(stages, STAGE_WEIGHTS):
        mass = gt_mass(belief[:, :N_CORNERS], centres)
        term = -(mass + 1e-9).log()
        total = total + weight * (term * valid).sum() / valid.sum().clamp_min(1.0)
        weight_sum += weight
    return total / weight_sum


def rank_loss(stages: tuple[torch.Tensor, ...], centres: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
    """The tallest belief away from the GT must sit below the GT by a margin.

    score_gt is a soft maximum over the GT window so the whole window receives
    gradient; score_wrong is a hard maximum, whose index is not differentiable
    but whose selected cell is.
    """
    total = torch.zeros((), device=centres.device, dtype=centres.dtype)
    weight_sum = 0.0
    for belief, weight in zip(stages, STAGE_WEIGHTS):
        heat = belief[:, :N_CORNERS]
        window = gt_window_mask(heat, centres)
        masked = heat.masked_fill(~window, -1e4)
        score_gt = TEMPERATURE * torch.logsumexp(
            masked.flatten(2) / TEMPERATURE, dim=-1)

        xs, ys = _grid_coordinates(heat)
        cx = centres[..., 0][..., None, None]
        cy = centres[..., 1][..., None, None]
        far_enough = ((xs - cx) ** 2 + (ys - cy) ** 2) > GT_EXCLUSION ** 2
        outside = heat.masked_fill(~far_enough, -1e4)
        score_wrong = outside.flatten(2).max(dim=-1).values

        term = F.softplus((score_wrong - score_gt + RANK_MARGIN) / TEMPERATURE)
        total = total + weight * (term * valid).sum() / valid.sum().clamp_min(1.0)
        weight_sum += weight
    return total / weight_sum


def expected_distance(belief: torch.Tensor, centres: torch.Tensor,
                      diagonal: torch.Tensor) -> torch.Tensor:
    """Huber of the normalised distance under the full-map probability.

    The expectation is taken over the distance, not over the coordinate: a
    bimodal map's mean coordinate falls between the two peaks and would look
    good while sitting on neither.
    """
    probability = spatial_probability(belief)
    xs, ys = _grid_coordinates(belief)
    cx = centres[..., 0][..., None, None]
    cy = centres[..., 1][..., None, None]
    distance = ((xs - cx) ** 2 + (ys - cy) ** 2 + 1e-9).sqrt()
    normalised = distance / diagonal[:, None, None, None].clamp_min(1e-6)
    huber = F.huber_loss(normalised, torch.zeros_like(normalised),
                         reduction="none", delta=1.0)
    return (probability * huber).sum(dim=(-2, -1))


def distance_loss(stages: tuple[torch.Tensor, ...], centres: torch.Tensor,
                  valid: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), device=centres.device, dtype=centres.dtype)
    weight_sum = 0.0
    for belief, weight in zip(stages, STAGE_WEIGHTS):
        term = expected_distance(belief[:, :N_CORNERS], centres, diagonal)
        total = total + weight * (term * valid).sum() / valid.sum().clamp_min(1.0)
        weight_sum += weight
    return total / weight_sum


def progress_loss(stages: tuple[torch.Tensor, ...], centres: torch.Tensor,
                  valid: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    """A later stage may not drift away or shed GT mass.

    The earlier stage is detached so the constraint cannot be met by making
    stage 4 worse.
    """
    h4, h5, h6 = (stage[:, :N_CORNERS] for stage in stages)
    d4 = expected_distance(h4, centres, diagonal)
    d5 = expected_distance(h5, centres, diagonal)
    d6 = expected_distance(h6, centres, diagonal)
    m4 = gt_mass(h4, centres)
    m5 = gt_mass(h5, centres)
    m6 = gt_mass(h6, centres)
    term = (F.relu(d5 - d4.detach()) + F.relu(d6 - d5.detach())
            + F.relu(m4.detach() - m5) + F.relu(m5.detach() - m6))
    return (term * valid).sum() / valid.sum().clamp_min(1.0)


def bbox_diagonal(centres: torch.Tensor, valid: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample GT bbox diagonal in belief cells, with a recorded fallback."""
    masked = centres.masked_fill(~valid[..., None], float("nan"))
    lower = torch.nan_to_num(masked, nan=float("inf")).amin(dim=1)
    upper = torch.nan_to_num(masked, nan=float("-inf")).amax(dim=1)
    diagonal = torch.linalg.norm(upper - lower, dim=-1)
    usable = torch.isfinite(diagonal) & (diagonal > 1.0)
    fallback = ~usable
    return torch.where(usable, diagonal,
                       torch.full_like(diagonal, GRID_DIAGONAL)), fallback


class StagewiseCornerLoss:
    """The four terms, evaluated on belief stages 4, 5 and 6."""

    names = ("mass", "rank", "distance", "progress")

    def __call__(self, beliefs, centres: torch.Tensor, valid_input: torch.Tensor
                 ) -> dict[str, torch.Tensor]:
        stages = (beliefs[3], beliefs[4], beliefs[5])
        centres = centres[:, :N_CORNERS].to(stages[0].dtype)
        valid = (valid_input[:, :N_CORNERS] > 0) & valid_corners(centres)
        valid = valid.to(stages[0].dtype)
        diagonal, fallback = bbox_diagonal(centres, valid > 0)
        return {
            "mass": mass_loss(stages, centres, valid),
            "rank": rank_loss(stages, centres, valid),
            "distance": distance_loss(stages, centres, valid, diagonal),
            "progress": progress_loss(stages, centres, valid, diagonal),
            "_valid_corners": valid.sum().detach(),
            "_diagonal_fallback": fallback.sum().detach(),
        }
