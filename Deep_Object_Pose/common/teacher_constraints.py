"""Teacher-constrained hard-positive losses for fixed-grid DOPE beliefs.

The helpers in this module deliberately operate on only the final belief map.
Teacher tensors and all discrete selections are detached, so gradients update
the student while never leaking into the frozen teacher or through a top-k
decision.  Per-frame outputs make it possible to apply the constraints only to
the teacher's worst mask-extent tail.
"""

import math

import torch
import torch.nn.functional as F

try:  # Support both package and legacy ``sys.path`` imports used by train.py.
    from .heatmap_refinement import local_softargmax_2d
except ImportError:  # pragma: no cover - exercised by the legacy trainer path
    from heatmap_refinement import local_softargmax_2d


def _validate_belief_pair(student_belief, teacher_belief):
    if student_belief.ndim != 4:
        raise ValueError("belief maps must have shape (B,C,H,W)")
    if student_belief.shape != teacher_belief.shape:
        raise ValueError(
            "student and teacher beliefs must have the same shape, got "
            f"{tuple(student_belief.shape)} and {tuple(teacher_belief.shape)}")
    if not (student_belief.is_floating_point()
            and teacher_belief.is_floating_point()):
        raise TypeError("student and teacher beliefs must be floating point")


def _channel_weights(reference, channel_mask):
    """Return a clamped ``(B,C)`` mask on ``reference``'s device/dtype."""
    batch, channels = reference.shape[:2]
    if channel_mask is None:
        return reference.new_ones((batch, channels))
    weights = torch.as_tensor(
        channel_mask, device=reference.device, dtype=reference.dtype)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)
    if weights.ndim != 2 or weights.shape[1] != channels:
        raise ValueError(
            "channel mask must have shape (C,) or (B,C), got "
            f"{tuple(weights.shape)}")
    if weights.shape[0] == 1 and batch != 1:
        weights = weights.expand(batch, -1)
    elif weights.shape[0] != batch:
        raise ValueError(
            f"channel mask batch is {weights.shape[0]}, expected 1 or {batch}")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("channel mask must contain only finite values")
    return weights.clamp(0.0, 1.0)


def _reduce_valid_frames(per_frame, valid, reduction):
    if reduction == "none":
        return per_frame
    if reduction == "sum":
        return per_frame[valid].sum() if bool(valid.any()) \
            else per_frame.sum() * 0.0
    if reduction == "mean":
        return per_frame[valid].mean() if bool(valid.any()) \
            else per_frame.sum() * 0.0
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def final_belief_distillation_loss(
    student_belief,
    teacher_belief,
    channel_mask=None,
    reduction="mean",
):
    """MSE-distill a final student belief map from a detached teacher map.

    ``channel_mask`` may be ``(C,)`` or ``(B,C)`` and may contain fractional
    weights.  Spatial pixels are averaged within each frame.  With
    ``reduction="none"`` the result is a ``(B,)`` vector; frames with no valid
    channels are graph-connected zeros and are omitted from mean/sum
    reductions.
    """
    _validate_belief_pair(student_belief, teacher_belief)
    teacher = teacher_belief.detach().to(
        device=student_belief.device, dtype=student_belief.dtype)
    weights = _channel_weights(student_belief, channel_mask)
    expanded_valid = (weights > 0).unsqueeze(-1).unsqueeze(-1)
    # Invalid pseudo-label channels may deliberately contain placeholders;
    # choose the finite zero branch before squaring so 0 * NaN cannot leak.
    difference = torch.where(
        expanded_valid, student_belief - teacher,
        torch.zeros_like(student_belief))
    error_per_channel = difference.square().flatten(2).mean(2)
    denominator = weights.sum(dim=1)
    per_frame = (error_per_channel * weights).sum(dim=1) \
        / denominator.clamp_min(torch.finfo(student_belief.dtype).tiny)
    valid = denominator > 0
    per_frame = torch.where(valid, per_frame, per_frame * 0.0)
    return _reduce_valid_frames(per_frame, valid, reduction)


def teacher_peak_retention_per_frame(
    student_belief,
    teacher_belief,
    channel_validity=None,
    teacher_peak_threshold=0.0,
    margin=0.0,
):
    """Return channel-wise teacher-peak hinge losses averaged per frame.

    Each teacher channel's argmax cell is selected from a detached teacher.
    At that exact cell the student is penalized by

    ``relu((teacher_peak - margin) - student_at_teacher_peak)``.

    A channel participates only when its optional validity is positive, its
    teacher peak is finite, and the peak is at least
    ``teacher_peak_threshold``.  The return value is ``(per_frame, valid)``;
    ``valid`` is true when at least one channel participated in that frame.
    """
    _validate_belief_pair(student_belief, teacher_belief)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if not math.isfinite(float(teacher_peak_threshold)):
        raise ValueError("teacher_peak_threshold must be finite")

    teacher = teacher_belief.detach().to(
        device=student_belief.device, dtype=student_belief.dtype)
    batch, channels = student_belief.shape[:2]
    teacher_flat = teacher.reshape(batch, channels, -1)
    # NaNs must not become an arbitrary selected target.  +/-inf is also
    # excluded by the finite-peak validity below.
    selection_map = torch.nan_to_num(
        teacher_flat, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
    peak_index = selection_map.argmax(dim=-1).detach()
    teacher_peak = teacher_flat.gather(
        -1, peak_index.unsqueeze(-1)).squeeze(-1)
    student_peak = student_belief.reshape(batch, channels, -1).gather(
        -1, peak_index.unsqueeze(-1)).squeeze(-1)

    channel_weights = _channel_weights(student_belief, channel_validity)
    channel_valid = (
        (channel_weights > 0)
        & torch.isfinite(teacher_peak)
        & torch.isfinite(student_peak)
        & (teacher_peak >= float(teacher_peak_threshold))
    )
    # Choose finite values before the hinge so the excluded branch has finite
    # forward values *and* finite zero gradients.
    safe_teacher_peak = torch.where(
        channel_valid, teacher_peak, torch.zeros_like(teacher_peak))
    safe_student_peak = torch.where(
        channel_valid, student_peak, torch.zeros_like(student_peak))
    hinge = F.relu(safe_teacher_peak - float(margin) - safe_student_peak)
    effective_weights = channel_weights * channel_valid.to(channel_weights.dtype)
    denominator = effective_weights.sum(dim=1)
    per_frame = (hinge * effective_weights).sum(dim=1) \
        / denominator.clamp_min(torch.finfo(student_belief.dtype).tiny)
    valid = denominator > 0
    per_frame = torch.where(valid, per_frame, per_frame * 0.0)
    return per_frame, valid


def teacher_peak_retention_loss(
    student_belief,
    teacher_belief,
    channel_validity=None,
    teacher_peak_threshold=0.0,
    margin=0.0,
    reduction="mean",
):
    """Reduced form of :func:`teacher_peak_retention_per_frame`."""
    per_frame, valid = teacher_peak_retention_per_frame(
        student_belief,
        teacher_belief,
        channel_validity=channel_validity,
        teacher_peak_threshold=teacher_peak_threshold,
        margin=margin,
    )
    return _reduce_valid_frames(per_frame, valid, reduction)


def top_fraction_cvar(
    per_frame_loss,
    valid,
    top_fraction,
    rank_by=None,
):
    """Average the largest valid one-sided losses (empirical upper-tail CVaR).

    Selection uses detached values and a stable descending sort, with ties
    resolved by original batch order.  By default the optimized losses are
    also the ranking scores.  ``rank_by`` may instead supply detached teacher
    extent scores, allowing the student loss to be optimized on precisely the
    teacher-worst tail.  The selected count is exactly
    ``max(1, ceil(valid_count * top_fraction))`` whenever a valid finite frame
    exists.  Invalid and non-finite frames are excluded.

    Returns ``(loss, info)``.  ``info`` contains Python numbers suitable for
    logging: ``valid_count``, ``selected_count``, ``selected_fraction`` (among
    valid frames), and the requested ``top_fraction``.
    """
    if per_frame_loss.ndim != 1:
        raise ValueError("per_frame_loss must have shape (B,)")
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError("top_fraction must be in (0,1]")
    valid_mask = torch.as_tensor(valid, device=per_frame_loss.device).bool()
    if valid_mask.ndim != 1 or valid_mask.shape != per_frame_loss.shape:
        raise ValueError("valid must have the same (B,) shape as per_frame_loss")
    if rank_by is None:
        ranking_score = per_frame_loss.detach()
    else:
        ranking_score = torch.as_tensor(
            rank_by, device=per_frame_loss.device,
            dtype=per_frame_loss.dtype).detach()
        if ranking_score.ndim != 1 or ranking_score.shape != per_frame_loss.shape:
            raise ValueError("rank_by must have the same (B,) shape as loss")
    valid_mask = (
        valid_mask
        & torch.isfinite(per_frame_loss.detach())
        & torch.isfinite(ranking_score)
    )
    valid_index = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_count = int(valid_index.numel())
    if valid_count == 0:
        safe = torch.where(
            torch.isfinite(per_frame_loss), per_frame_loss,
            torch.zeros_like(per_frame_loss))
        zero = safe.sum() * 0.0
        return zero, {
            "valid_count": 0,
            "selected_count": 0,
            "selected_fraction": 0.0,
            "top_fraction": float(top_fraction),
        }

    selected_count = min(
        valid_count,
        max(1, int(math.ceil(valid_count * float(top_fraction)))))
    valid_ranking_score = ranking_score[valid_index]
    # Sorting detached values makes the ranking non-differentiable by design;
    # indexing the original tensor preserves gradients for selected frames.
    order = torch.argsort(
        valid_ranking_score, descending=True, stable=True)
    selected_index = valid_index[order[:selected_count]]
    loss = per_frame_loss[selected_index].clamp_min(0.0).mean()
    return loss, {
        "valid_count": valid_count,
        "selected_count": selected_count,
        "selected_fraction": selected_count / float(valid_count),
        "top_fraction": float(top_fraction),
    }


def mask_extent_per_frame(
    belief,
    mask,
    mask_valid,
    radius=5,
    temperature=0.10,
    tolerance=1.0,
):
    """Measure one-sided cuboid-under-mask extent for every frame.

    The geometry matches :class:`heatmap_refinement.MaskExtentLoss`, but keeps
    the per-frame tensors needed for teacher-tail ranking.  Returns
    ``(per_frame, valid, mean_gap_cells_per_frame)``.  ``per_frame`` is the
    mean four-side gap normalized by the grid's larger dimension; the last
    output is the same gap in belief-grid cells.
    """
    if belief.ndim != 4 or belief.shape[1] < 8:
        raise ValueError("belief must have shape (B,C>=8,H,W)")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape (B,1,H,W)")
    if belief.shape[0] != mask.shape[0] or belief.shape[-2:] != mask.shape[-2:]:
        raise ValueError("belief and mask batch/spatial dimensions must match")
    if radius < 0 or temperature <= 0:
        raise ValueError("radius must be >=0 and temperature >0")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    frame_mask_valid = torch.as_tensor(
        mask_valid, device=belief.device).reshape(-1).bool()
    if frame_mask_valid.numel() != belief.shape[0]:
        raise ValueError("mask_valid must contain one value per frame")

    aligned_mask = mask.to(device=belief.device)
    pred_xy = local_softargmax_2d(
        belief[:, :8], radius=int(radius), temperature=float(temperature))
    _batch, _channels, height, width = aligned_mask.shape
    active = aligned_mask[:, 0] > 0.5
    area = active.flatten(1).sum(dim=1)
    valid = frame_mask_valid & (area >= 4)

    ys = torch.arange(height, device=belief.device).view(1, height, 1)
    xs = torch.arange(width, device=belief.device).view(1, 1, width)
    x_min = torch.where(active, xs, width).flatten(1).amin(dim=1).to(belief.dtype)
    x_max = torch.where(active, xs, -1).flatten(1).amax(dim=1).to(belief.dtype)
    y_min = torch.where(active, ys, height).flatten(1).amin(dim=1).to(belief.dtype)
    y_max = torch.where(active, ys, -1).flatten(1).amax(dim=1).to(belief.dtype)

    pred_min = pred_xy.amin(dim=1)
    pred_max = pred_xy.amax(dim=1)
    tol = float(tolerance)
    gaps = torch.stack([
        F.relu(pred_min[:, 0] - (x_min + tol)),
        F.relu((x_max - tol) - pred_max[:, 0]),
        F.relu(pred_min[:, 1] - (y_min + tol)),
        F.relu((y_max - tol) - pred_max[:, 1]),
    ], dim=1)
    mean_gap_cells = gaps.mean(dim=1)
    per_frame = mean_gap_cells / float(max(height, width))
    return per_frame, valid, mean_gap_cells


__all__ = [
    "final_belief_distillation_loss",
    "mask_extent_per_frame",
    "teacher_peak_retention_loss",
    "teacher_peak_retention_per_frame",
    "top_fraction_cvar",
]
