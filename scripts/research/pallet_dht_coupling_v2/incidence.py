"""Training-only incidence of assigned predicted corners and image-derived lines.

The neural forward is unchanged. This loss uses stock TAL assignment, never a
GT crop or a line fitted to predicted/GT points as the predicted line. Both
endpoints must agree with the SAME member of the full Hough mixture. There is
no mean line, argmax, physical-visibility assertion, or inference-time solver.

The frozen 60,000-image source has exactly one annotated object per image.
This implementation deliberately supports zero or one and rejects multiple
objects rather than inventing instance assignments for the global line map.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from scripts.research.pallet_dht_joint_v1.line_targets import EDGES, build_line_targets

RESIDUAL_SCALE_DIAGONAL_FRACTION = .01
MINIMUM_RESIDUAL_SCALE_INPUT_PX = 1.
INCIDENCE_POLICY = "singleton_same_line_log_mixture_pseudohuber_v1"


def decode_points_px(raw_branch: Mapping, anchor_points: Tensor,
                     stride_tensor: Tensor) -> Tensor:
    """Decode the actual PoseLoss26 dense corner coordinates to input pixels.

    PoseLoss26 uses (raw_xy + anchor_xy) * stride, without the older *2/-0.5
    recipe. Assignment inputs come from the SAME stock criterion invocation.
    The returned [B,A,9,2] tensor retains its prediction graph.
    """
    raw = raw_branch["kpts"]
    if raw.ndim != 3 or raw.shape[1] != 27:
        raise ValueError("Expected stock nine-keypoint raw kpts [B,27,A]")
    b, _, a = raw.shape
    if tuple(anchor_points.shape) != (a, 2) or tuple(stride_tensor.shape) != (a, 1):
        raise ValueError("Anchor/stride shapes differ from the stock prediction")
    points = raw.permute(0, 2, 1).reshape(b, a, 9, 3)[..., :2]
    return (points + anchor_points[None, :, None, :]) * stride_tensor[None, :, None, :]


def pair_mixture_incidence(logits: Tensor, pairs: Tensor, normals: Tensor,
                           rhos: Tensor, scale: float | Tensor) -> Tensor:
    """Return [anchors,roles] losses for finite, already-supported hypotheses.

    logits [roles,K], pairs [anchors,roles,2,2], normals [K,2], rhos [K].
    p_k = sigmoid(logit_k) / sum_j sigmoid(logit_j), conditional on the valid
    footprint. These are relative nonnegative line scores, NOT calibrated
    categorical probabilities. Stable log-sigmoid/logsumexp keeps every mode.

    c_k = .5 * sum_{endpoint=1,2}(sqrt(1+(residual/scale)^2)-1).
    loss = -log sum_k p_k exp(-c_k). The joint pair cost is formed BEFORE
    marginalization, so endpoints cannot choose separate incompatible modes.
    """
    if logits.ndim != 2 or pairs.ndim != 4 or pairs.shape[-2:] != (2, 2):
        raise ValueError("Expected logits [roles,K] and pairs [anchors,roles,2,2]")
    if pairs.shape[1] != logits.shape[0] or tuple(normals.shape) != (logits.shape[1], 2):
        raise ValueError("Pair roles and line hypotheses disagree")
    if tuple(rhos.shape) != (logits.shape[1],) or logits.shape[1] == 0:
        raise ValueError("At least one finite candidate line is required")
    for name, value in (("line logits", logits), ("predicted endpoints", pairs),
                        ("line normals", normals), ("line offsets", rhos)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite {name} in incidence")
    if not bool(torch.isfinite(torch.as_tensor(scale)).all()) or float(scale) <= 0:
        raise ValueError("Positive finite residual scale required")
    # Signed residual changes sign under (normal,rho)->(-normal,-rho), while
    # pseudo-Huber is even. No angular unwrapping or antipodal averaging occurs.
    residual = (torch.einsum("arec,kc->arek", pairs, normals) - rhos) / scale
    costs = .5 * (torch.sqrt(1. + residual.square()) - 1.).sum(-2)
    log_mass = F.logsigmoid(logits)
    log_probability = log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)
    return -torch.logsumexp(log_probability[None] - costs, dim=-1)


def incidence_loss(line_logits: Tensor, lattice: Mapping, pred_points_px: Tensor,
                   fg_mask: Tensor, target_gt_idx: Tensor,
                   gt_keypoints_normalized: Tensor, batch_idx: Tensor,
                   input_hw: tuple[int, int]) -> tuple[Tensor, dict]:
    """Mean-per-image loss on the actual stock-assigned positive predictions.

    Inputs: image line logits [B,12,T,R], predicted points [B,A,9,2] in network
    input pixels; fg_mask/target_gt_idx [B,A] from the unchanged stock TAL call;
    normalized transformed GT [N,9,3] and its image index [N]. GT contributes
    only support and the existing object identity. No GT location contributes
    to a line score, mixture weight, residual, or prediction coordinate.

    Each valid role averages all foreground anchors for its object. Roles
    average within an image; all B images average, including ignored/empty
    images as zero. No confidence gate or predicted-length gate is introduced.
    Caller combines the two E2E branches with their stock weights, then applies
    its registered incidence coefficient and batch-size scaling ONCE each.
    """
    if line_logits.ndim != 4 or line_logits.shape[1] != 12:
        raise ValueError("Expected image-derived [B,12,T,R] line logits")
    b, _, t, r = line_logits.shape
    if pred_points_px.ndim != 4 or tuple(pred_points_px.shape[2:]) != (9, 2):
        raise ValueError("Expected predicted points [B,A,9,2] in input pixels")
    if pred_points_px.shape[0] != b or tuple(fg_mask.shape) != tuple(pred_points_px.shape[:2]):
        raise ValueError("Foreground mask differs from dense point predictions")
    if tuple(target_gt_idx.shape) != tuple(fg_mask.shape) or fg_mask.dtype != torch.bool:
        raise ValueError("Expected boolean foreground and matching target-index arrays")
    if tuple(gt_keypoints_normalized.shape[1:]) != (9, 3) or batch_idx.numel() != len(gt_keypoints_normalized):
        raise ValueError("Expected normalized GT [N,9,3] and N image indices")
    h, w = int(lattice["height"]), int(lattice["width"])
    ih, iw = map(int, input_hw)
    if min(b, h, w, ih, iw) <= 0 or not math.isclose(ih / h, iw / w, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Input and feature footprints must have the same aspect ratio")
    if (t, r) != (int(lattice["theta_bins"]), int(lattice["rho_bins"])):
        raise ValueError("Logits and line lattice differ")
    raw_indices = batch_idx.detach().flatten().to(line_logits.device)
    image_indices = raw_indices.long()
    if bool(((raw_indices != image_indices) | (image_indices < 0) | (image_indices >= b)).any()):
        raise ValueError("GT image indices must be valid integers")
    counts = torch.bincount(image_indices, minlength=b)
    if bool((counts > 1).any()):
        raise ValueError("Incidence scope is at most one annotated object per image; multi-object assignment is unsupported")
    # Same source convention and support policy as the unchanged auxiliary GT
    # target builder. The built target values are NOT used by this loss.
    work_dtype = torch.float64 if line_logits.dtype == torch.float64 or pred_points_px.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=line_logits.device.type, enabled=False):
        logits = line_logits.to(work_dtype)
        points = pred_points_px.to(work_dtype)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite image line logits")
        data = build_line_targets(gt_keypoints_normalized, batch_idx, b, (ih, iw), lattice,
                                  device=logits.device, dtype=work_dtype)
        physical = torch.as_tensor(lattice["valid"], device=logits.device, dtype=torch.bool)
        if tuple(physical.shape) != (t, r) or not bool(physical.any()):
            raise ValueError("Nonempty physical-footprint line mask required")
        theta = torch.as_tensor(lattice["theta_values"], device=logits.device, dtype=work_dtype)
        rho = torch.as_tensor(lattice["rho_values"], device=logits.device, dtype=work_dtype)
        if tuple(theta.shape) != (t,) or tuple(rho.shape) != (r,):
            raise ValueError("Malformed theta/rho lattice vectors")
        normals = torch.stack((theta.cos(), theta.sin()), -1)[:, None].expand(t, r, 2)[physical]
        offsets = rho[None].expand(t, r)[physical]
        stride = iw / w
        scale_px = max(MINIMUM_RESIDUAL_SCALE_INPUT_PX, RESIDUAL_SCALE_DIAGONAL_FRACTION * math.hypot(iw, ih))
        scale_feature = scale_px / stride
        centre = points.new_tensor((w / 2, h / 2))
        edges = torch.tensor(EDGES, device=logits.device)
        # Empty slices preserve a zero gradient without inf*0 or reading any
        # background point values. They also leave centroid gradient zero.
        zero = logits.reshape(-1)[:0].sum() + points[..., :0, :].sum()
        image_losses = []
        used_images = used_roles = used_anchors = used_anchor_roles = 0
        for image in range(b):
            active = fg_mask[image]
            if bool(active.any()):
                if int(counts[image]) != 1 or bool((target_gt_idx[image, active] != 0).any()):
                    raise ValueError("Foreground assignment does not identify this image's unique GT object")
            if int(counts[image]) == 0 or not bool(active.any()):
                image_losses.append(zero)
                continue
            object_index = int(torch.nonzero(image_indices == image, as_tuple=False)[0, 0])
            usable = data["instance_usable"][object_index] & data["role_valid"][image]
            if not bool(usable.any()):
                image_losses.append(zero)
                continue
            endpoints = (points[image, active, :8] / stride - centre)[:, edges[usable]]
            values = pair_mixture_incidence(logits[image, usable][:, physical], endpoints,
                                            normals, offsets, scale_feature)
            image_losses.append(values.mean())
            n_anchors, n_roles = values.shape
            used_images += 1; used_roles += n_roles; used_anchors += n_anchors
            used_anchor_roles += n_anchors * n_roles
        loss = torch.stack(image_losses).mean()
        diagnostics = dict(incidence_loss=loss.detach(), incidence_images=used_images,
            incidence_batch_images=b, incidence_supported_image_roles=used_roles,
            incidence_foreground_anchors=used_anchors, incidence_anchor_roles=used_anchor_roles,
            incidence_scale_input_px=scale_px,
            incidence_physical_hypotheses=int(physical.sum()), incidence_policy=INCIDENCE_POLICY)
        return loss, diagnostics
