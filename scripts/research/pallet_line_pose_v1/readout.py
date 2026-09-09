"""GT-free, temperature-calibrated readout of the frozen line-head logits.

This module changes no trained parameter. It reconstructs exactly the geometric
chart and sample-coverage validity of model.forward without image features.
"""
from __future__ import annotations

import math

import torch

from model import build_candidates, summarize_distribution, fuse_corners


def prepare_readout(logits, points, boxes, point_valid, input_shape, temperature=1.):
    """Return the geometric/posterior state, without any GT or feature input.

    All coordinates are letterboxed input pixels. Temperature is one positive
    scalar fixed for an arm/seed on synthetic calibration data, not per image.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if logits.shape != (len(points), 8, 222) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits[B,8,222]")
    if points.dtype != logits.dtype or boxes.dtype != points.dtype:
        raise ValueError("logits, points and boxes must use the same floating dtype")
    if input_shape.shape != (len(points), 2) or not torch.isfinite(input_shape).all() or (input_shape <= 0).any():
        raise ValueError("input_shape must be positive finite [B,2] H,W")
    out = build_candidates(points, boxes, point_valid)
    # Match model.forward's candidate positions, including its tangent order.
    h = out["candidate_h"]
    normal, rho = h[..., :2], -h[..., 2]
    tangent = torch.stack([-normal[..., 1], normal[..., 0]], -1)
    midpoint = out["midpoint"][:, :, None]
    base = midpoint + (rho - (normal * midpoint).sum(-1))[..., None] * normal
    along = torch.linspace(-.5, .5, 32, dtype=points.dtype, device=points.device)
    positions = (base[..., None, :] + tangent[..., None, :]
                 * out["length"][:, :, None, None, None] * along[None, None, None, :, None])
    in_frame = ((positions[..., 0] >= 0) & (positions[..., 1] >= 0)
                & (positions[..., 0] < input_shape[:, None, None, None, 1])
                & (positions[..., 1] < input_shape[:, None, None, None, 0]))
    out["line_valid"] = out["line_valid"] & in_frame.any(-1).any(-1)
    out["sample_coverage"] = in_frame.to(points.dtype).mean(-1)
    tempered = logits / temperature
    out.update(summarize_distribution(tempered, h))
    out.update(points_raw=points, points=points, boxes=boxes, input_shape=input_shape,
               logits=tempered, temperature=temperature)
    return out


def apply_readout(prepared, lam=1., cap=None):
    """Fuse one prepared state; cap is a scalar or [B] INPUT-pixel norm limit.

    To cap at fraction f of the original-image diagonal use
    cap=f*hypot(raw_height,raw_width)*letterbox_gain. No image dimensions are
    inferred from predictions or GT. Missing points and center8 stay unchanged.
    """
    points = prepared["points_raw"]
    if cap is not None:
        cap = torch.as_tensor(cap, device=points.device, dtype=points.dtype)
        if cap.ndim == 0:
            cap = cap.expand(len(points))
        if cap.shape != (len(points),) or not torch.isfinite(cap).all() or (cap < 0).any():
            raise ValueError("cap must be finite nonnegative scalar or [B] input pixels")
    refined, variance, raw_variance = fuse_corners(points, prepared["point_valid"],
        prepared["h_mean"], prepared["moment"], prepared["null_probability"],
        prepared["line_valid"], prepared["anchor_std"], lam)
    if float(lam) != 0 and cap is not None:
        valid = prepared["point_valid"][:, :8]
        delta = torch.where(valid[..., None], refined[:, :8] - points[:, :8], torch.zeros_like(points[:, :8]))
        length = delta.norm(dim=-1)
        factor = torch.minimum(torch.ones_like(length), cap[:, None] / length.clamp_min(1e-12))
        clipped = points[:, :8] + factor[..., None] * delta
        # Unclipped and unavailable entries copy their prior values exactly.
        use_clip = valid & (length > cap[:, None])
        corners = torch.where(use_clip[..., None], clipped, refined[:, :8])
        refined = torch.cat([corners, points[:, 8:]], 1)
    out = dict(prepared)
    out.update(points=refined, line_variance=variance, raw_line_variance=raw_variance,
               lam=float(lam), cap_input_px=cap)
    return out


def readout(logits, points, boxes, point_valid, input_shape,
            temperature=1., lam=1., cap=None):
    """Pure GT-free distribution decoding and optional displacement clipping."""
    state = prepare_readout(logits, points, boxes, point_valid, input_shape, temperature)
    return apply_readout(state, lam, cap)
