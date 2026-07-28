"""Fixed-50 heatmap refinement losses and evaluation helpers.

These utilities deliberately do not change the 400x400 input or 50x50 belief
grid.  They cover opt-in additions:

* a mask-extent under-coverage loss for the final belief maps;
* a projected 8-corner span loss aligned to the deployed centroid decoder;
* a signed, GT-direction footprint loss that cannot hide edge reversal;
* per-corner localization sigma calibration for weighted/defensive PnP.

The public decoder returns the exact quantities consumed by the safe PnP path:
``sigma9`` in belief-grid cells. W/D candidates remain geometry-ranked and an
ambiguous pair is rejected; no learned prior is emitted because the audited
training labels collapse to a single class under the safe solver convention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_dope_output(output):
    """Normalize legacy and refinement DOPE return arities.

    Returns ``(beliefs, affinities, vec, seg, refinement)``.  A refinement model
    has the fixed 5-tuple schema introduced in ``common/models.py``; legacy
    2/3/4-tuples remain accepted for backward-compatible training/evaluation.
    """
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("DOPE output must be a tuple/list with at least two items")
    beliefs, affinities = output[0], output[1]
    if len(output) == 5 and isinstance(output[4], dict):
        return beliefs, affinities, output[2], output[3], output[4]
    if len(output) == 4:
        return beliefs, affinities, output[2], output[3], {}
    if len(output) == 3:
        return beliefs, affinities, output[2], None, {}
    if len(output) == 2:
        return beliefs, affinities, None, None, {}
    raise ValueError(f"unsupported DOPE output arity: {len(output)}")


def pseudo_label_channel_masks(keypoint_valid):
    """Map nine pseudo-keypoint validity flags to DOPE loss channels.

    Belief channel ``i`` is valid exactly when keypoint ``i`` is present.  Each
    corner affinity is a two-channel vector from corner ``i`` to the centroid,
    so both of its channels are valid only when *both* corner ``i`` and
    centroid 8 are present.  Leading dimensions are preserved, allowing either
    one ``(9,)`` annotation or a collated ``(B,9)`` batch.
    """
    valid = torch.as_tensor(keypoint_valid)
    if valid.ndim < 1 or valid.shape[-1] != 9:
        raise ValueError(
            "pseudo keypoint validity must have final dimension 9, "
            f"got {tuple(valid.shape)}")
    valid = valid.bool()
    belief_mask = valid.to(dtype=torch.float32)
    affinity_valid = valid[..., :8] & valid[..., 8:9]
    affinity_mask = affinity_valid.repeat_interleave(2, dim=-1).to(
        dtype=torch.float32)
    return belief_mask, affinity_mask


def channel_masked_mse(prediction, target, channel_mask):
    """MSE over valid channels while preserving the legacy all-ones result.

    ``channel_mask`` is ``(C,)`` or ``(B,C)`` and is broadcast over spatial
    dimensions.  Dividing the masked full-tensor mean by the valid-channel
    fraction is equivalent to averaging only valid elements.  With an all-ones
    mask the expression reduces to the historical ``square().mean()``.
    """
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError(
            "prediction and target must have the same (B,C,...) shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}")
    mask = torch.as_tensor(
        channel_mask, device=prediction.device, dtype=prediction.dtype)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2:
        raise ValueError(
            f"channel_mask must be (C,) or (B,C), got {tuple(mask.shape)}")
    if mask.shape[1] != prediction.shape[1]:
        raise ValueError(
            f"mask has {mask.shape[1]} channels, expected {prediction.shape[1]}")
    if mask.shape[0] not in (1, prediction.shape[0]):
        raise ValueError(
            f"mask batch is {mask.shape[0]}, expected 1 or {prediction.shape[0]}")
    mask = mask.clamp(0.0, 1.0)
    spatial_shape = (1,) * (prediction.ndim - 2)
    broadcast_mask = mask.reshape(mask.shape[0], mask.shape[1], *spatial_shape)
    squared_error = (prediction - target).square()
    masked_mean = (squared_error * broadcast_mask).mean()
    valid_fraction = mask.mean()
    return masked_mean / valid_fraction.clamp_min(
        torch.finfo(prediction.dtype).tiny)


def _smoothed_peak_cells(maps, sigma=2.0, threshold=None):
    """Torch equivalent of the deployed Gaussian-smooth + 4-neighbour NMS."""
    batch, channels, height, width = maps.shape
    radius = int(4.0 * sigma + 0.5)
    axis = torch.arange(
        -radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel_1d = torch.exp(-0.5 * (axis / float(sigma)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, -1)
    flat_maps = maps.reshape(batch * channels, 1, height, width)
    # scipy.ndimage.gaussian_filter uses reflected boundaries by default.
    pad_mode = "reflect" if min(height, width) > radius else "replicate"
    padded = F.pad(
        flat_maps, (radius, radius, radius, radius), mode=pad_mode)
    smooth = F.conv2d(padded, kernel).reshape(batch, channels, height, width)

    left = torch.zeros_like(smooth)
    right = torch.zeros_like(smooth)
    up = torch.zeros_like(smooth)
    down = torch.zeros_like(smooth)
    left[:, :, 1:, :] = smooth[:, :, :-1, :]
    right[:, :, :-1, :] = smooth[:, :, 1:, :]
    up[:, :, :, 1:] = smooth[:, :, :, :-1]
    down[:, :, :, :-1] = smooth[:, :, :, 1:]
    peaks = ((smooth >= left) & (smooth >= right)
             & (smooth >= up) & (smooth >= down))
    if threshold is not None:
        peaks = peaks & (smooth > float(threshold))
    score = maps.masked_fill(~peaks, -torch.inf).reshape(batch, channels, -1)
    index = score.argmax(dim=-1)
    peak_found = torch.isfinite(score.max(dim=-1).values)
    # Without a threshold a finite map always has a local maximum, but retain a
    # defensive fallback. With a threshold, the fallback coordinate is ignored
    # by the returned detection-valid mask.
    no_peak = ~peak_found
    if bool(no_peak.any()):
        raw_index = maps.reshape(batch, channels, -1).argmax(dim=-1)
        index = torch.where(no_peak, raw_index, index)
    return index, peak_found


def weighted_centroid_decode(
    belief,
    window=11,
    offset=0.4395,
    smooth_sigma=2.0,
    peak_threshold=None,
    return_peak_valid=False,
):
    """Decode the strongest peak with an 11x11 weighted centroid.

    ``belief`` is ``(B,K,H,W)`` or ``(K,H,W)``.  The function is NaN-safe and
    returns ``(xy, peak_value, peak_xy)``.  ``peak_xy`` is the integer map cell
    used to sample the uncertainty map.  This mirrors the deployed sub-pixel
    weighted-centroid policy while keeping the operation torch-only.
    """
    squeeze = belief.ndim == 3
    if squeeze:
        belief = belief.unsqueeze(0)
    if belief.ndim != 4:
        raise ValueError(f"belief must be (B,K,H,W) or (K,H,W), got {belief.shape}")
    if window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")

    maps = torch.nan_to_num(belief, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    batch, channels, height, width = maps.shape
    flat = maps.reshape(batch, channels, -1)
    confidence = flat.max(dim=-1).values
    index, peak_found = _smoothed_peak_cells(
        maps, sigma=smooth_sigma, threshold=peak_threshold)
    peak_y = torch.div(index, width, rounding_mode="floor")
    peak_x = index.remainder(width)
    radius = window // 2
    decoded = torch.empty(batch, channels, 2, device=maps.device, dtype=maps.dtype)

    # B*K is small (normally 12*9); explicit patches avoid a large unfold and
    # make the border behavior identical to the numpy evaluator.
    for b in range(batch):
        for k in range(channels):
            px, py = int(peak_x[b, k]), int(peak_y[b, k])
            x0, x1 = max(0, px - radius), min(width, px + radius + 1)
            y0, y1 = max(0, py - radius), min(height, py + radius + 1)
            patch = maps[b, k, y0:y1, x0:x1].clamp_min(0)
            mass = patch.sum()
            if bool(torch.isfinite(mass)) and float(mass) > 1.0e-12:
                ys = torch.arange(y0, y1, device=maps.device, dtype=maps.dtype)
                xs = torch.arange(x0, x1, device=maps.device, dtype=maps.dtype)
                wy = (patch.sum(dim=1) * ys).sum() / mass
                wx = (patch.sum(dim=0) * xs).sum() / mass
                decoded[b, k, 0] = wx + offset
                decoded[b, k, 1] = wy + offset
            else:
                decoded[b, k, 0] = peak_x[b, k].to(maps.dtype)
                decoded[b, k, 1] = peak_y[b, k].to(maps.dtype)

    peak_xy = torch.stack([peak_x, peak_y], dim=-1)
    if squeeze:
        result = (decoded[0], confidence[0], peak_xy[0])
        if return_peak_valid:
            result = result + (peak_found[0],)
        return result
    result = (decoded, confidence, peak_xy)
    if return_peak_valid:
        result = result + (peak_found,)
    return result


def _sample_at_peak(maps, peak_xy):
    """Sample ``(B,K,H,W)`` maps at integer ``(B,K,2)`` x/y cells."""
    if maps.ndim != 4 or peak_xy.ndim != 3:
        raise ValueError("maps/peak_xy shapes must be (B,K,H,W)/(B,K,2)")
    batch, channels, height, width = maps.shape
    if peak_xy.shape[:2] != (batch, channels):
        raise ValueError("map channel count and peak count differ")
    x = peak_xy[..., 0].long().clamp(0, width - 1)
    y = peak_xy[..., 1].long().clamp(0, height - 1)
    index = y * width + x
    return maps.reshape(batch, channels, -1).gather(-1, index.unsqueeze(-1)).squeeze(-1)


def decode_refinement_outputs(
    belief,
    refinement,
    threshold=None,
    window=11,
    offset=0.4395,
):
    """Decode keypoints plus localization uncertainty for evaluation.

    Returns a dict with:

    * ``keypoints``: ``(B,9,2)`` weighted-centroid coordinates in 50-grid cells;
    * ``confidence`` and ``valid`` per corner;
    * ``sigma9``: predicted 1-sigma localization error in grid cells, or None;

    To convert sigma for anisotropically resized original images, multiply its
    x/y copies by ``orig_width/50`` and ``orig_height/50`` respectively.  A
    scalar inverse-variance PnP weight can then use ``1/(sigma_px**2 + eps)``.
    """
    squeeze = belief.ndim == 3
    maps = belief.unsqueeze(0) if squeeze else belief
    xy, confidence, peak_xy, peak_found = weighted_centroid_decode(
        maps, window=window, offset=offset,
        peak_threshold=threshold, return_peak_valid=True)
    valid = torch.ones_like(confidence, dtype=torch.bool)
    if threshold is not None:
        valid = (confidence >= float(threshold)) & peak_found

    sigma = None
    log_sigma_map = (refinement or {}).get("corner_log_sigma")
    if log_sigma_map is not None:
        sampled = _sample_at_peak(log_sigma_map, peak_xy)
        sigma = torch.exp(sampled.clamp(math.log(0.25), math.log(20.0)))
        sigma = torch.nan_to_num(sigma, nan=20.0, posinf=20.0, neginf=0.25)

    result = {
        "keypoints": xy,
        "confidence": confidence,
        "valid": valid,
        "sigma9": sigma,
        "peak_xy": peak_xy,
    }
    if squeeze:
        for key in ("keypoints", "confidence", "valid", "sigma9", "peak_xy"):
            if result[key] is not None:
                result[key] = result[key][0]
    return result


def local_softargmax_2d(maps, radius=5, temperature=0.10):
    """Differentiable local soft-argmax around a detached strongest cell.

    Restricting support avoids the global softmax/background-centre bias of MSE
    belief maps.  The selected window is discrete, while coordinates within it
    remain differentiable.  Returns ``(B,K,2)`` x/y map coordinates.
    """
    if maps.ndim != 4:
        raise ValueError("maps must have shape (B,K,H,W)")
    if radius < 0 or temperature <= 0:
        raise ValueError("radius must be >=0 and temperature >0")
    clean = torch.nan_to_num(maps, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    batch, channels, height, width = clean.shape
    flat = clean.reshape(batch, channels, -1)
    peak = flat.argmax(dim=-1).detach()
    cy = torch.div(peak, width, rounding_mode="floor")
    cx = peak.remainder(width)

    ys = torch.arange(height, device=maps.device, dtype=maps.dtype)
    xs = torch.arange(width, device=maps.device, dtype=maps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx_flat, yy_flat = xx.reshape(1, 1, -1), yy.reshape(1, 1, -1)
    support = (
        (xx_flat - cx.unsqueeze(-1)).abs() <= radius
    ) & (
        (yy_flat - cy.unsqueeze(-1)).abs() <= radius
    )
    logits = flat / float(temperature)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits.masked_fill(~support, -1.0e4)
    prob = torch.softmax(logits, dim=-1)
    x = (prob * xx_flat).sum(dim=-1)
    y = (prob * yy_flat).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def differentiable_weighted_centroid_2d(
    maps,
    window=11,
    offset=0.4395,
    smooth_sigma=2.0,
):
    """Differentiable form of the deployed local weighted-centroid decoder.

    The Gaussian-smoothed NMS cell is a detached discrete selection. Inside the
    selected ``window``, the positive belief mass and its centroid remain
    differentiable. This avoids training a global-softmax coordinate that the
    evaluator never consumes.
    """
    if maps.ndim != 4:
        raise ValueError("maps must have shape (B,K,H,W)")
    if window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    if smooth_sigma <= 0:
        raise ValueError("smooth_sigma must be positive")

    clean = torch.nan_to_num(
        maps, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    batch, channels, height, width = clean.shape
    # Gradients flow through the centroid mass below, not the argmax index.
    with torch.no_grad():
        peak, _found = _smoothed_peak_cells(
            clean.detach(), sigma=float(smooth_sigma), threshold=None)
    peak_y = torch.div(peak, width, rounding_mode="floor")
    peak_x = peak.remainder(width)

    ys = torch.arange(height, device=maps.device, dtype=maps.dtype)
    xs = torch.arange(width, device=maps.device, dtype=maps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(1, 1, -1)
    yy = yy.reshape(1, 1, -1)
    radius = window // 2
    support = (
        (xx - peak_x.unsqueeze(-1)).abs() <= radius
    ) & (
        (yy - peak_y.unsqueeze(-1)).abs() <= radius
    )

    mass = clean.reshape(batch, channels, -1).clamp_min(0.0)
    mass = mass * support.to(mass.dtype)
    denominator = mass.sum(dim=-1)
    safe_denominator = denominator.clamp_min(1.0e-12)
    x = (mass * xx).sum(dim=-1) / safe_denominator
    y = (mass * yy).sum(dim=-1) / safe_denominator
    has_mass = denominator > 1.0e-12
    x = torch.where(has_mass, x + float(offset), peak_x.to(maps.dtype))
    y = torch.where(has_mass, y + float(offset), peak_y.to(maps.dtype))
    return torch.stack([x, y], dim=-1)


def _huber_nonnegative(value, delta):
    """Huber penalty for a non-negative residual with bounded gradient."""
    value = value.clamp_min(0.0)
    quadratic = 0.5 * value.square()
    linear = float(delta) * (value - 0.5 * float(delta))
    return torch.where(value <= float(delta), quadratic, linear)


class ProjectedSpanLoss(nn.Module):
    """Penalize an 8-corner projected footprint that is smaller than its GT.

    The target is the transformed ``projected_cuboid`` itself, not a pallet
    segmentation mask. GT PCA axes make the span comparison orientation-aware
    and translation-invariant. A small ordered-coordinate anchor and a soft
    overshoot guard prevent satisfying the span term with one remote outlier.

    All coordinates stay on the existing 50x50 grid. Frames are used only when
    all eight GT corners are far enough inside the grid that the legacy sigma-2
    belief generator supervised every channel.
    """

    def __init__(
        self,
        window=11,
        offset=0.4395,
        smooth_sigma=2.0,
        interior_margin=4.0,
        min_span=2.0,
        coord_weight=1.0,
        overshoot_weight=0.25,
        overshoot_ratio=1.10,
        huber_delta=0.05,
        footprint_edge_weight=0.0,
        hard_edge_threshold=0.85,
        hard_example_gain=0.0,
        min_edge_length=1.0,
    ):
        super().__init__()
        if interior_margin < 0 or min_span <= 0:
            raise ValueError("interior_margin must be >=0 and min_span >0")
        if coord_weight < 0 or overshoot_weight < 0:
            raise ValueError("loss component weights must be non-negative")
        if overshoot_ratio < 1.0:
            raise ValueError("overshoot_ratio must be >=1")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if footprint_edge_weight < 0 or hard_example_gain < 0:
            raise ValueError("edge weight and hard-example gain must be non-negative")
        if not 0 < hard_edge_threshold <= 1.0:
            raise ValueError("hard_edge_threshold must be in (0,1]")
        if min_edge_length <= 0:
            raise ValueError("min_edge_length must be positive")
        self.window = int(window)
        self.offset = float(offset)
        self.smooth_sigma = float(smooth_sigma)
        self.interior_margin = float(interior_margin)
        self.min_span = float(min_span)
        self.coord_weight = float(coord_weight)
        self.overshoot_weight = float(overshoot_weight)
        self.overshoot_ratio = float(overshoot_ratio)
        self.huber_delta = float(huber_delta)
        self.footprint_edge_weight = float(footprint_edge_weight)
        self.hard_edge_threshold = float(hard_edge_threshold)
        self.hard_example_gain = float(hard_example_gain)
        self.min_edge_length = float(min_edge_length)

    def forward(self, belief, target_xy, target_valid):
        if belief.ndim != 4 or belief.shape[1] < 8:
            raise ValueError("belief must have shape (B,K>=8,H,W)")
        if target_xy.ndim != 3 or target_xy.shape[1:] != (9, 2):
            raise ValueError("target_xy must have shape (B,9,2)")
        if target_valid.ndim != 2 or target_valid.shape[1] != 9:
            raise ValueError("target_valid must have shape (B,9)")
        if target_xy.shape[0] != belief.shape[0]:
            raise ValueError("belief and target batch sizes differ")

        pred = differentiable_weighted_centroid_2d(
            belief[:, :8], window=self.window, offset=self.offset,
            smooth_sigma=self.smooth_sigma)
        gt = target_xy[:, :8].to(device=belief.device, dtype=belief.dtype)
        valid_corner = target_valid[:, :8].to(device=belief.device).bool()
        finite = torch.isfinite(gt).all(dim=-1)
        gt_safe = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)

        height, width = belief.shape[-2:]
        margin = self.interior_margin
        interior = (
            (gt_safe[..., 0] >= margin)
            & (gt_safe[..., 0] < float(width) - margin)
            & (gt_safe[..., 1] >= margin)
            & (gt_safe[..., 1] < float(height) - margin)
        )
        frame_valid = (valid_corner & finite & interior).all(dim=1)

        gt_centered = gt_safe - gt_safe.mean(dim=1, keepdim=True)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        # Target axes are constants; no unstable eigendecomposition gradient is
        # needed when a footprint is close to square.
        covariance = torch.bmm(
            gt_centered.transpose(1, 2), gt_centered) / 8.0
        _eigenvalues, axes = torch.linalg.eigh(covariance.detach())
        gt_axis = torch.bmm(gt_centered, axes)
        pred_axis = torch.bmm(pred_centered, axes)
        gt_span = gt_axis.amax(dim=1) - gt_axis.amin(dim=1)
        pred_span = pred_axis.amax(dim=1) - pred_axis.amin(dim=1)
        frame_valid = frame_valid & (gt_span >= self.min_span).all(dim=1)

        ratio = pred_span.clamp_min(0.25) / gt_span.clamp_min(self.min_span)
        log_ratio = torch.log(ratio)
        under_residual = torch.relu(-log_ratio)
        over_residual = torch.relu(
            log_ratio - math.log(self.overshoot_ratio))
        under_loss = _huber_nonnegative(
            under_residual, self.huber_delta).mean(dim=1)
        over_loss = _huber_nonnegative(
            over_residual, self.huber_delta).mean(dim=1)

        gt_min = gt_safe.amin(dim=1)
        gt_max = gt_safe.amax(dim=1)
        gt_diagonal = torch.linalg.vector_norm(
            gt_max - gt_min, dim=1).clamp_min(self.min_span)
        coord_residual = torch.linalg.vector_norm(
            pred - gt_safe, dim=-1) / gt_diagonal.unsqueeze(1)
        coord_loss = _huber_nonnegative(
            coord_residual, self.huber_delta).mean(dim=1)

        # Image-plane width and front-to-rear/depth edges are the dimensions
        # that collapse in the audited UC failures; vertical height edges stay
        # near GT and are intentionally excluded. Rear-right edges touching
        # channels 5/6 get extra weight because their ordered errors dominate.
        edge_i = torch.tensor(
            [0, 3, 4, 7, 0, 1, 2, 3],
            device=belief.device, dtype=torch.long)
        edge_j = torch.tensor(
            [1, 2, 5, 6, 4, 5, 6, 7],
            device=belief.device, dtype=torch.long)
        edge_importance = belief.new_tensor(
            [1.0, 1.0, 1.5, 1.5, 1.0, 2.0, 2.0, 1.0])
        gt_edge = torch.linalg.vector_norm(
            gt_safe[:, edge_i] - gt_safe[:, edge_j], dim=-1)
        pred_edge = torch.linalg.vector_norm(
            pred[:, edge_i] - pred[:, edge_j], dim=-1)
        edge_valid = gt_edge >= self.min_edge_length
        edge_ratio = pred_edge.clamp_min(0.25) / gt_edge.clamp_min(
            self.min_edge_length)
        edge_log_ratio = torch.log(edge_ratio)
        edge_under = torch.relu(-edge_log_ratio)
        edge_over = torch.relu(
            edge_log_ratio - math.log(self.overshoot_ratio))
        edge_penalty = (
            _huber_nonnegative(edge_under, self.huber_delta)
            + self.overshoot_weight
            * _huber_nonnegative(edge_over, self.huber_delta)
        )
        edge_weights = edge_valid.to(belief.dtype) * edge_importance
        edge_loss = (edge_penalty * edge_weights).sum(dim=1) / (
            edge_weights.sum(dim=1).clamp_min(1.0))

        # Hard mining is based on mean width/depth scale, not one noisy edge.
        width_ratio = edge_ratio[:, :4].mean(dim=1)
        depth_ratio = edge_ratio[:, 4:].mean(dim=1)
        min_footprint_ratio = torch.minimum(width_ratio, depth_ratio)
        hard_example = (
            min_footprint_ratio.detach() < self.hard_edge_threshold)
        hard_weight = 1.0 + self.hard_example_gain * hard_example.to(
            belief.dtype)

        per_frame = hard_weight * (
            under_loss
            + self.overshoot_weight * over_loss
            + self.coord_weight * coord_loss
            + self.footprint_edge_weight * edge_loss
        )
        if not bool(frame_valid.any()):
            zero = belief.sum() * 0.0
            return zero, {
                "valid_frac": 0.0,
                "mean_min_span_ratio": 0.0,
                "mean_under_log": 0.0,
                "mean_coord_cells": 0.0,
                "mean_min_edge_ratio": 0.0,
                "hard_fraction": 0.0,
            }

        loss = per_frame[frame_valid].mean()
        coord_cells = torch.linalg.vector_norm(
            pred - gt_safe, dim=-1).mean(dim=1)
        return loss, {
            "valid_frac": float(frame_valid.float().mean().detach().item()),
            "mean_min_span_ratio": float(
                ratio[frame_valid].amin(dim=1).mean().detach().item()),
            "mean_under_log": float(
                under_residual[frame_valid].mean().detach().item()),
            "mean_coord_cells": float(
                coord_cells[frame_valid].mean().detach().item()),
            "mean_min_edge_ratio": float(
                min_footprint_ratio[frame_valid].mean().detach().item()),
            "hard_fraction": float(
                hard_example[frame_valid].float().mean().detach().item()),
        }


class SignedFootprintLoss(nn.Module):
    """One-sided 8-corner footprint loss in the target's signed directions.

    The deployed differentiable weighted-centroid decoder first converts the
    final belief maps to coordinates on the unchanged 50x50 grid. Width and
    front-to-rear/depth edges are then projected onto their *GT edge unit
    vectors*. This signed projection distinguishes a correct-length edge from
    an orthogonal or reversed edge, unlike a norm-only length comparison.

    A second term projects every corner relative to the mean of the eight
    corners onto its GT radial direction. It prevents a solution in which a
    subset of edges expands while individual corners remain contracted. Both
    terms penalize only under-coverage at ratios below one. A weaker overshoot
    guard starts above ``overshoot_ratio`` to prevent unbounded expansion.

    Frames are gated exactly at the footprint level: all eight labels must be
    valid, finite, sufficiently interior to the fixed grid, and have
    non-degenerate width/depth edges and radial vectors. ``forward_coordinates``
    exposes the coordinate-level calculation for audits; normal training uses
    ``forward`` so gradients pass through the deployed centroid decoder.
    """

    # Same ordered width/depth edge sets as ProjectedSpanLoss. Keeping the
    # order is important: signed projection uses GT[j] - GT[i].
    _EDGE_I = (0, 3, 4, 7, 0, 1, 2, 3)
    _EDGE_J = (1, 2, 5, 6, 4, 5, 6, 7)

    def __init__(
        self,
        window=11,
        offset=0.4395,
        smooth_sigma=2.0,
        interior_margin=4.0,
        min_edge_length=1.0,
        min_radial_length=1.0,
        width_weight=1.0,
        depth_weight=1.0,
        radial_weight=1.0,
        overshoot_weight=0.20,
        overshoot_ratio=1.10,
        huber_delta=0.05,
        coverage_epsilon=1.0e-6,
        grid_size=50,
    ):
        super().__init__()
        if window < 1 or window % 2 != 1:
            raise ValueError("window must be a positive odd integer")
        if smooth_sigma <= 0:
            raise ValueError("smooth_sigma must be positive")
        if interior_margin < 0:
            raise ValueError("interior_margin must be non-negative")
        if min_edge_length <= 0 or min_radial_length <= 0:
            raise ValueError("minimum edge/radial lengths must be positive")
        if min(width_weight, depth_weight, radial_weight) < 0:
            raise ValueError("footprint component weights must be non-negative")
        if width_weight + depth_weight + radial_weight <= 0:
            raise ValueError("at least one footprint component must be enabled")
        if overshoot_weight < 0:
            raise ValueError("overshoot_weight must be non-negative")
        if overshoot_ratio < 1.0:
            raise ValueError("overshoot_ratio must be >=1")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0 <= coverage_epsilon < 1.0e-2:
            raise ValueError("coverage_epsilon must be in [0,1e-2)")
        if int(grid_size) != 50:
            raise ValueError("SignedFootprintLoss is fixed to the deployed 50 grid")

        self.window = int(window)
        self.offset = float(offset)
        self.smooth_sigma = float(smooth_sigma)
        self.interior_margin = float(interior_margin)
        self.min_edge_length = float(min_edge_length)
        self.min_radial_length = float(min_radial_length)
        self.width_weight = float(width_weight)
        self.depth_weight = float(depth_weight)
        self.radial_weight = float(radial_weight)
        self.overshoot_weight = float(overshoot_weight)
        self.overshoot_ratio = float(overshoot_ratio)
        self.huber_delta = float(huber_delta)
        # Removes floating-point residue for an exactly matched footprint;
        # this is not a practical under-coverage deadband (1 ppm of an edge).
        self.coverage_epsilon = float(coverage_epsilon)
        self.grid_size = 50

    @staticmethod
    def _zero_info():
        return {
            "valid_frac": 0.0,
            "mean_width_signed_ratio": 0.0,
            "mean_depth_signed_ratio": 0.0,
            "mean_min_edge_signed_ratio": 0.0,
            "mean_radial_signed_ratio": 0.0,
            "mean_min_radial_signed_ratio": 0.0,
            "mean_edge_undercoverage": 0.0,
            "mean_radial_undercoverage": 0.0,
            "mean_undercoverage": 0.0,
            "mean_overshoot": 0.0,
            "undercovered_edge_fraction": 0.0,
            "undercovered_corner_fraction": 0.0,
        }

    def _coordinate_loss(
        self,
        pred,
        target_xy,
        target_valid,
        height,
        width,
    ):
        if pred.ndim != 3 or pred.shape[1:] != (8, 2):
            raise ValueError("pred must have shape (B,8,2)")
        if target_xy.ndim != 3 or target_xy.shape[1:] != (9, 2):
            raise ValueError("target_xy must have shape (B,9,2)")
        if target_valid.ndim != 2 or target_valid.shape[1] != 9:
            raise ValueError("target_valid must have shape (B,9)")
        if target_xy.shape[0] != pred.shape[0] \
                or target_valid.shape[0] != pred.shape[0]:
            raise ValueError("prediction and target batch sizes differ")
        if int(height) != self.grid_size or int(width) != self.grid_size:
            raise ValueError(
                "SignedFootprintLoss requires deployed 50x50 coordinates")

        gt = target_xy[:, :8].to(device=pred.device, dtype=pred.dtype)
        valid_corner = target_valid[:, :8].to(device=pred.device).bool()
        finite = torch.isfinite(gt).all(dim=-1)
        gt_safe = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)

        margin = self.interior_margin
        interior = (
            (gt_safe[..., 0] >= margin)
            & (gt_safe[..., 0] < float(width) - margin)
            & (gt_safe[..., 1] >= margin)
            & (gt_safe[..., 1] < float(height) - margin)
        )
        frame_valid = (valid_corner & finite & interior).all(dim=1)

        edge_i = torch.tensor(
            self._EDGE_I, device=pred.device, dtype=torch.long)
        edge_j = torch.tensor(
            self._EDGE_J, device=pred.device, dtype=torch.long)
        gt_edge_vector = gt_safe[:, edge_j] - gt_safe[:, edge_i]
        gt_edge_length = torch.linalg.vector_norm(gt_edge_vector, dim=-1)
        gt_edge_unit = gt_edge_vector / gt_edge_length.clamp_min(
            self.min_edge_length).unsqueeze(-1)
        pred_edge_vector = pred[:, edge_j] - pred[:, edge_i]
        pred_edge_signed = (pred_edge_vector * gt_edge_unit).sum(dim=-1)
        edge_ratio = pred_edge_signed / gt_edge_length.clamp_min(
            self.min_edge_length)
        frame_valid = frame_valid & (
            gt_edge_length >= self.min_edge_length).all(dim=1)

        gt_center = gt_safe.mean(dim=1, keepdim=True)
        pred_center = pred.mean(dim=1, keepdim=True)
        gt_radial_vector = gt_safe - gt_center
        gt_radial_length = torch.linalg.vector_norm(
            gt_radial_vector, dim=-1)
        gt_radial_unit = gt_radial_vector / gt_radial_length.clamp_min(
            self.min_radial_length).unsqueeze(-1)
        pred_radial_vector = pred - pred_center
        pred_radial_signed = (
            pred_radial_vector * gt_radial_unit).sum(dim=-1)
        radial_ratio = pred_radial_signed / gt_radial_length.clamp_min(
            self.min_radial_length)
        frame_valid = frame_valid & (
            gt_radial_length >= self.min_radial_length).all(dim=1)

        # Signed ratios are intentionally not clamped: an orthogonal edge has
        # ratio zero and a reversed edge has a negative ratio, both of which
        # must be penalized despite having the correct Euclidean norm.
        edge_under = torch.relu(
            1.0 - self.coverage_epsilon - edge_ratio)
        radial_under = torch.relu(
            1.0 - self.coverage_epsilon - radial_ratio)
        edge_over = torch.relu(edge_ratio - self.overshoot_ratio)
        radial_over = torch.relu(radial_ratio - self.overshoot_ratio)

        width_under_loss = _huber_nonnegative(
            edge_under[:, :4], self.huber_delta).mean(dim=1)
        depth_under_loss = _huber_nonnegative(
            edge_under[:, 4:], self.huber_delta).mean(dim=1)
        radial_under_loss = _huber_nonnegative(
            radial_under, self.huber_delta).mean(dim=1)
        width_over_loss = _huber_nonnegative(
            edge_over[:, :4], self.huber_delta).mean(dim=1)
        depth_over_loss = _huber_nonnegative(
            edge_over[:, 4:], self.huber_delta).mean(dim=1)
        radial_over_loss = _huber_nonnegative(
            radial_over, self.huber_delta).mean(dim=1)

        component_weight = (
            self.width_weight + self.depth_weight + self.radial_weight)
        under_loss = (
            self.width_weight * width_under_loss
            + self.depth_weight * depth_under_loss
            + self.radial_weight * radial_under_loss
        ) / component_weight
        over_loss = (
            self.width_weight * width_over_loss
            + self.depth_weight * depth_over_loss
            + self.radial_weight * radial_over_loss
        ) / component_weight
        per_frame = under_loss + self.overshoot_weight * over_loss

        if not bool(frame_valid.any()):
            return pred.sum() * 0.0, self._zero_info()

        loss = per_frame[frame_valid].mean()
        mean_edge_under = edge_under.mean(dim=1)
        mean_radial_under = radial_under.mean(dim=1)
        mean_under = (
            self.width_weight * edge_under[:, :4].mean(dim=1)
            + self.depth_weight * edge_under[:, 4:].mean(dim=1)
            + self.radial_weight * mean_radial_under
        ) / component_weight
        mean_over = (
            self.width_weight * edge_over[:, :4].mean(dim=1)
            + self.depth_weight * edge_over[:, 4:].mean(dim=1)
            + self.radial_weight * radial_over.mean(dim=1)
        ) / component_weight
        return loss, {
            "valid_frac": float(frame_valid.float().mean().detach().item()),
            "mean_width_signed_ratio": float(
                edge_ratio[frame_valid, :4].mean().detach().item()),
            "mean_depth_signed_ratio": float(
                edge_ratio[frame_valid, 4:].mean().detach().item()),
            "mean_min_edge_signed_ratio": float(
                edge_ratio[frame_valid].amin(dim=1).mean().detach().item()),
            "mean_radial_signed_ratio": float(
                radial_ratio[frame_valid].mean().detach().item()),
            "mean_min_radial_signed_ratio": float(
                radial_ratio[frame_valid].amin(dim=1).mean().detach().item()),
            "mean_edge_undercoverage": float(
                mean_edge_under[frame_valid].mean().detach().item()),
            "mean_radial_undercoverage": float(
                mean_radial_under[frame_valid].mean().detach().item()),
            "mean_undercoverage": float(
                mean_under[frame_valid].mean().detach().item()),
            "mean_overshoot": float(
                mean_over[frame_valid].mean().detach().item()),
            "undercovered_edge_fraction": float(
                (edge_under[frame_valid] > 0).float().mean().detach().item()),
            "undercovered_corner_fraction": float(
                (radial_under[frame_valid] > 0).float().mean().detach().item()),
        }

    def forward_coordinates(
        self,
        predicted_xy,
        target_xy,
        target_valid,
    ):
        """Evaluate already-decoded ``(B,8,2)`` coordinates on the 50 grid."""
        return self._coordinate_loss(
            predicted_xy, target_xy, target_valid,
            self.grid_size, self.grid_size)

    def forward(self, belief, target_xy, target_valid):
        if belief.ndim != 4 or belief.shape[1] < 8:
            raise ValueError("belief must have shape (B,K>=8,50,50)")
        if belief.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError("SignedFootprintLoss requires a 50x50 belief grid")
        pred = differentiable_weighted_centroid_2d(
            belief[:, :8], window=self.window, offset=self.offset,
            smooth_sigma=self.smooth_sigma)
        return self._coordinate_loss(
            pred, target_xy, target_valid,
            belief.shape[-2], belief.shape[-1])


class CornerUncertaintyLoss(nn.Module):
    """Calibrate per-corner log sigma to detached observed localization error."""

    def __init__(self, min_sigma=0.25, max_sigma=20.0, window=11):
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.window = int(window)

    def forward(self, belief, log_sigma_map, target_xy, target_valid):
        with torch.no_grad():
            decoded, _confidence, peak_xy = weighted_centroid_decode(
                belief.detach(), window=self.window)
            error = torch.linalg.vector_norm(decoded - target_xy, dim=-1)
            error = torch.nan_to_num(
                error, nan=self.max_sigma,
                posinf=self.max_sigma, neginf=self.max_sigma)
            target_log_sigma = error.clamp(
                self.min_sigma, self.max_sigma).log()
            valid = target_valid.bool() & torch.isfinite(target_xy).all(dim=-1)

        pred_log_sigma = _sample_at_peak(log_sigma_map, peak_xy)
        pred_log_sigma = torch.nan_to_num(
            pred_log_sigma,
            nan=math.log(self.max_sigma),
            posinf=math.log(self.max_sigma),
            neginf=math.log(self.min_sigma),
        ).clamp(math.log(self.min_sigma), math.log(self.max_sigma))
        if not bool(valid.any()):
            zero = log_sigma_map.sum() * 0.0
            return zero, {"valid_frac": 0.0, "mean_error": 0.0, "mean_sigma": 0.0}
        per_corner = F.smooth_l1_loss(
            pred_log_sigma, target_log_sigma, reduction="none", beta=0.25)
        loss = per_corner[valid].mean()
        sigma = pred_log_sigma.detach().exp()
        info = {
            "valid_frac": float(valid.float().mean().item()),
            "mean_error": float(error[valid].mean().item()),
            "mean_sigma": float(sigma[valid].mean().item()),
        }
        return loss, info


class MaskExtentLoss(nn.Module):
    """Penalize only a predicted cuboid bbox lying *inside* the real mask bbox."""

    def __init__(self, radius=5, temperature=0.10, tolerance=1.0):
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.tolerance = float(tolerance)

    def forward(self, belief, mask, mask_valid):
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape (B,1,H,W)")
        pred_xy = local_softargmax_2d(
            belief[:, :8], radius=self.radius, temperature=self.temperature)
        batch, _channels, height, width = mask.shape
        active = mask[:, 0] > 0.5
        area = active.flatten(1).sum(dim=1)
        valid = mask_valid.view(-1).bool() & (area >= 4)

        ys = torch.arange(height, device=mask.device).view(1, height, 1)
        xs = torch.arange(width, device=mask.device).view(1, 1, width)
        x_min = torch.where(active, xs, width).flatten(1).amin(dim=1).to(belief.dtype)
        x_max = torch.where(active, xs, -1).flatten(1).amax(dim=1).to(belief.dtype)
        y_min = torch.where(active, ys, height).flatten(1).amin(dim=1).to(belief.dtype)
        y_max = torch.where(active, ys, -1).flatten(1).amax(dim=1).to(belief.dtype)

        pred_min = pred_xy.amin(dim=1)
        pred_max = pred_xy.amax(dim=1)
        tol = self.tolerance
        gaps = torch.stack([
            F.relu(pred_min[:, 0] - (x_min + tol)),
            F.relu((x_max - tol) - pred_max[:, 0]),
            F.relu(pred_min[:, 1] - (y_min + tol)),
            F.relu((y_max - tol) - pred_max[:, 1]),
        ], dim=1)
        per_frame = gaps.mean(dim=1) / float(max(height, width))
        if not bool(valid.any()):
            zero = belief.sum() * 0.0
            return zero, {"valid_frac": 0.0, "mean_gap_cells": 0.0}
        loss = per_frame[valid].mean()
        return loss, {
            "valid_frac": float(valid.float().mean().item()),
            "mean_gap_cells": float(gaps[valid].mean().detach().item()),
        }
