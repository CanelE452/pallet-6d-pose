"""Training-only targets for twelve camera-facing *structural* cuboid lines.

The head is global: this module never supplies a crop, point proposal, or GT
to its forward pass. It consumes the already transformed YOLO training batch.
``visibility > 0`` means a coordinate is supervised; it does not establish
that a physical edge is visible. No original-image padding mask is inferred.

Coordinates follow the Hough block's declared cell grid, not an assertion
about convolutional receptive-field centres. Normalized input (x, y) maps to
``(x*W-W/2, y*H-H/2)`` on the actual rectangular P4 map. Normal angle is in
[0, pi); crossing the angular seam reverses signed rho.

Each line deposits a bilinear target into at most four theta/rho bins. Targets
from different objects in the same role are combined by maximum, not summed
or forced into a single categorical distribution. Independent role channels
retain coincident lines. Only bins whose infinite line intersects the actual
input footprint are supervised. This is NOT the sparse operator's vote-mass
mask: at axis-aligned angles, subpixel rho bins can have zero centre votes
while still representing a valid image line, learned through Hough convolutions.

For a supervised image/role, BCE on nonzero target bins and BCE on background
bins are separately averaged and weighted equally. A background-only role
uses the background mean alone. Means are then taken over supervised roles
per image and over all batch images (an entirely ignored image contributes
zero). Thus a large background lattice does not drown the sparse positives.

If ANY annotated object has an unknown, degenerate, outside-only or otherwise
unsupported line in a role, that entire image/role is ignored. This conservative
policy avoids falsely labeling the object's unknown global line as background.
A truly empty annotated image remains valid background supervision.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))
ROLE_NAMES = ("front_top_width", "front_right_height", "front_bottom_width",
              "front_left_height", "rear_top_width", "rear_right_height",
              "rear_bottom_width", "rear_left_height", "left_top_depth",
              "right_top_depth", "right_bottom_depth", "left_bottom_depth")
TARGET_POLICY = "bilinear_max_union_balanced_region_bce_v1"


def normalized_to_feature(points_xy: Tensor, height: int, width: int) -> Tensor:
    """Convert normalized, already augmented input coordinates to P4 centres."""
    size = points_xy.new_tensor((width, height))
    return points_xy * size - size / 2


def canonical_lines(segments: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return theta, signed rho and nondegeneracy for (..., 2, 2) endpoints.

    Endpoint reversal has no effect on the represented line. Nonfinite or
    coincident inputs return finite placeholder parameters with valid=False.
    """
    finite = torch.isfinite(segments).flatten(-2).all(-1)
    safe = torch.nan_to_num(segments, nan=0., posinf=0., neginf=0.)
    delta = safe[..., 1, :] - safe[..., 0, :]
    valid = finite & (delta.square().sum(-1) > 1e-12)
    theta = torch.remainder(torch.atan2(delta[..., 0], -delta[..., 1]), math.pi)
    normal = torch.stack((theta.cos(), theta.sin()), -1)
    rho = (normal * safe[..., 0, :]).sum(-1)
    return theta, rho, valid


def segment_intersects_input(segments: Tensor, height: int, width: int) -> Tensor:
    """Positive-length segment intersection with the actual input footprint.

    The footprint includes the outer half cells: [-W/2, W/2] x [-H/2, H/2].
    Merely intersecting the image as an infinite extended line is insufficient.
    """
    finite = torch.isfinite(segments).flatten(-2).all(-1)
    safe = torch.nan_to_num(segments, nan=0., posinf=0., neginf=0.)
    start = safe[..., 0, :]
    delta = safe[..., 1, :] - start
    half = start.new_tensor((width, height)) / 2
    parallel = delta.abs() <= 1e-12
    denominator = torch.where(parallel, torch.ones_like(delta), delta)
    a, b = (-half - start) / denominator, (half - start) / denominator
    entry, leave = torch.minimum(a, b), torch.maximum(a, b)
    entry = torch.where(parallel, torch.full_like(entry, -torch.inf), entry)
    leave = torch.where(parallel, torch.full_like(leave, torch.inf), leave)
    inside_parallel = (~parallel | ((start >= -half) & (start <= half))).all(-1)
    low = entry.amax(-1).clamp_min(0)
    high = leave.amin(-1).clamp_max(1)
    visible_length = (high - low).clamp_min(0) * delta.norm(dim=-1)
    return finite & inside_parallel & (visible_length > 1e-6)


def _bilinear_indices(theta: Tensor, rho: Tensor, lattice: Mapping,
                      valid_bins: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Four sparse target entries per line, with antipodal angular wrapping."""
    t, r = int(lattice["theta_bins"]), int(lattice["rho_bins"])
    angle = theta * (t / math.pi)
    # Suppress float32 trigonometric leakage at exact lattice locations.
    angle = torch.where((angle - angle.round()).abs() < 1e-5, angle.round(), angle)
    lower_t = angle.floor().long()
    alpha_t = angle - lower_t
    raw_t = torch.stack((lower_t, lower_t + 1), -1)
    angular_weight = torch.stack((1 - alpha_t, alpha_t), -1)
    wrapped_t = raw_t.remainder(t)
    signed_rho = rho[..., None] * torch.where(raw_t >= t, -1., 1.)
    fractional_r = (signed_rho + float(lattice["rho_max"])) / float(lattice["rho_step"])
    fractional_r = torch.where((fractional_r - fractional_r.round()).abs() < 1e-5,
                               fractional_r.round(), fractional_r)
    lower_r = fractional_r.floor().long()
    alpha_r = fractional_r - lower_r
    indices_r = torch.stack((lower_r, lower_r + 1), -1)
    weights = angular_weight[..., None] * torch.stack((1 - alpha_r, alpha_r), -1)
    indices_t = wrapped_t[..., None].expand_as(indices_r)
    inside = (indices_r >= 0) & (indices_r < r)
    linear = indices_t * r + indices_r.clamp(0, r - 1)
    weights = weights * inside * valid_bins.flatten()[linear]
    total = weights.sum(dim=(-1, -2))
    # Boundary/invalid bins are discarded; accessible target mass is conserved.
    weights = weights / total.clamp_min(1e-12)[..., None, None]
    return linear.flatten(-2), weights.flatten(-2), total > 1e-8


@torch.no_grad()
def build_line_targets(keypoints: Tensor, batch_idx: Tensor, batch_size: int,
                       input_hw: tuple[int, int], lattice: Mapping, *,
                       device: torch.device | str | None = None,
                       dtype: torch.dtype = torch.float32) -> dict:
    """Build [B,12,T,R] targets/masks from transformed training labels only."""
    h, w = int(lattice["height"]), int(lattice["width"])
    t, r = int(lattice["theta_bins"]), int(lattice["rho_bins"])
    ih, iw = (int(v) for v in input_hw)
    if min(h, w, t, r, ih, iw, batch_size) <= 0:
        raise ValueError("positive image, lattice and batch dimensions required")
    if not math.isclose(ih / h, iw / w, rel_tol=0, abs_tol=1e-6):
        raise ValueError("actual input and feature map must have the same aspect ratio")
    if keypoints.ndim != 3 or tuple(keypoints.shape[1:]) != (9, 3):
        raise ValueError("keypoints must be [objects,9,3] normalized xy/visibility")
    if batch_idx.numel() != keypoints.shape[0]:
        raise ValueError("one batch index is required per object")
    if r != 2 * round(float(lattice["rho_max"]) / float(lattice["rho_step"])) + 1:
        raise ValueError("symmetric signed-rho lattice required")
    device = keypoints.device if device is None else device
    points = keypoints.detach().to(device=device, dtype=dtype)
    raw_batch_idx = batch_idx.detach().flatten().to(device=device)
    object_batch = raw_batch_idx.long()
    if object_batch.numel() and bool(((object_batch < 0) | (object_batch >= batch_size)
                                     | (raw_batch_idx != object_batch)).any()):
        raise ValueError("batch indices must be integers inside this batch")
    if "valid" in lattice:
        valid_bins = torch.as_tensor(lattice["valid"], device=device, dtype=torch.bool)
        if tuple(valid_bins.shape) != (t, r):
            raise ValueError("lattice valid mask must have shape [theta,rho]")
    else:
        raise ValueError("lattice valid physical-footprint mask is required")
    edges = torch.tensor(EDGES, device=device)
    corners = points[:, :8, :]
    endpoint_labels = corners[:, edges]
    known = (torch.isfinite(endpoint_labels).flatten(-2).all(-1)
             & (endpoint_labels[..., 2] > 0).all(-1))
    segments = normalized_to_feature(endpoint_labels[..., :2], h, w)
    theta, rho, nondegenerate = canonical_lines(segments)
    intersects = segment_intersects_input(segments, h, w)
    in_rho = rho.abs() <= float(lattice["rho_max"]) + 1e-6
    indices, weights, accessible = _bilinear_indices(theta, rho, lattice, valid_bins)
    usable = known & nondegenerate & intersects & in_rho & accessible
    image_role = object_batch[:, None] * 12 + torch.arange(12, device=device)[None]
    unsupported = torch.zeros(batch_size * 12, device=device, dtype=torch.long)
    unsupported.scatter_add_(0, image_role.flatten(), (~usable).long().flatten())
    role_valid = (unsupported == 0).reshape(batch_size, 12)
    # Duplicate annotations cannot increase target amplitude beyond one.
    target_flat = torch.zeros(batch_size * 12 * t * r, device=device, dtype=dtype)
    destination = image_role[..., None] * (t * r) + indices
    source = weights * usable[..., None]
    target_flat.scatter_reduce_(0, destination.flatten(), source.flatten(),
                                reduce="amax", include_self=True)
    target = target_flat.reshape(batch_size, 12, t, r)
    target = target * role_valid[..., None, None]
    mask = role_valid[..., None, None] & valid_bins[None, None]
    positive = target > 0
    diagnostics = {
        "line_instances_total": torch.tensor(known.numel(), device=device),
        "supported_line_instances": usable.sum(),
        "unknown_endpoint_line_instances": (~known).sum(),
        "degenerate_line_instances": (known & ~nondegenerate).sum(),
        "outside_segment_line_instances": (known & nondegenerate & ~intersects).sum(),
        "out_of_lattice_line_instances": (known & nondegenerate & intersects & ~in_rho).sum(),
        "no_footprint_bin_line_instances": (known & nondegenerate & intersects & in_rho & ~accessible).sum(),
        "valid_image_roles": role_valid.sum(),
        "ignored_image_roles": (~role_valid).sum(),
        "positive_bins": positive.sum(),
        "background_only_image_roles": (role_valid & ~positive.flatten(-2).any(-1)).sum(),
        "images_with_any_supervision": role_valid.any(-1).sum(),
    }
    return dict(target=target, mask=mask, role_valid=role_valid,
                instance_usable=usable, theta=theta, rho=rho,
                diagnostics=diagnostics)


def auxiliary_line_loss(logits: Tensor, keypoints: Tensor, batch_idx: Tensor,
                        input_hw: tuple[int, int], lattice: Mapping) -> tuple[Tensor, dict]:
    """Mean per-image balanced BCE; call once after the stock E2E loss.

    This function never consumes decoded predicted points, boxes, matching
    choices, confidence scores, original-image GT, or physical dimensions.
    """
    expected = (12, int(lattice["theta_bins"]), int(lattice["rho_bins"]))
    if logits.ndim != 4 or tuple(logits.shape[1:]) != expected:
        raise ValueError(f"line logits must have shape [B,{expected}]")
    work = logits if logits.dtype == torch.float64 else logits.float()
    data = build_line_targets(keypoints, batch_idx, len(work), input_hw, lattice,
                              device=work.device, dtype=work.dtype)
    target, mask = data["target"], data["mask"]
    positive = (target > 0) & mask
    background = (target == 0) & mask
    bce = F.binary_cross_entropy_with_logits(work, target, reduction="none")
    count_p = positive.sum(dim=(-1, -2))
    count_n = background.sum(dim=(-1, -2))
    loss_p = (bce * positive).sum(dim=(-1, -2)) / count_p.clamp_min(1)
    loss_n = (bce * background).sum(dim=(-1, -2)) / count_n.clamp_min(1)
    components = (count_p > 0).to(work.dtype) + (count_n > 0).to(work.dtype)
    role_loss = (loss_p + loss_n) / components.clamp_min(1)
    supervised_roles = data["role_valid"].sum(-1)
    image_loss = role_loss.sum(-1) / supervised_roles.clamp_min(1)
    loss = image_loss.mean()
    diagnostics = {**data["diagnostics"],
                   "line_loss": loss.detach(),
                   "positive_region_bce": (loss_p.sum() / (count_p > 0).sum().clamp_min(1)).detach(),
                   "background_region_bce": (loss_n.sum() / (count_n > 0).sum().clamp_min(1)).detach()}
    return loss, diagnostics
