"""Local D/P heads for frozen FULL ResNet features.

The visual evidence, 222-candidate P readout, D residual control, losses and
coordinate conventions are copied from the byte-identical DOPE/old-ResNet
refiner implementation. P5 adds the already-used zero-effect 5D metadata-logit
residual; P5_CONSTANT uses the identical module with an all-zero context.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import HEAD_ARMS, HEAD_CONFIG
from .dimensions import context_for_arm


def finite(points, valid):
    return valid.bool() & torch.isfinite(points).all(-1) & ~(points == -1).all(-1)


def sample(feature, positions, input_shape, stride):
    b, channels, height, width = feature.shape
    ys = (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5) * stride
    xs = (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5) * stride
    support = ((ys[None, :, None] < input_shape[:, 0, None, None])
               & (xs[None, None, :] < input_shape[:, 1, None, None]))
    feature = feature * support[:, None]
    grid = 2 * positions / positions.new_tensor([stride * width, stride * height]) - 1
    value = F.grid_sample(feature, grid.reshape(b, -1, positions.shape[-2], 2),
                          mode="bilinear", padding_mode="zeros", align_corners=False)
    return value.reshape(b, channels, *positions.shape[1:-1]).permute(0, 2, 3, 1, 4)


class GenericPointRefiner(nn.Module):
    """Visual P: a 222-way local displacement distribution for corners 0..7."""

    def __init__(self, c3=128, c4=256, hidden=16, encoded=24,
                 stencil_fraction=0.1310373991727829, stride3=8, stride4=16):
        super().__init__()
        if any(int(value) != value or value < 1 for value in (c3, c4, hidden, encoded)):
            raise ValueError("Channel counts must be positive integers")
        if any(not math.isfinite(float(value)) or value <= 0
               for value in (stride3, stride4, stencil_fraction)):
            raise ValueError("Strides and stencil_fraction must be positive finite")
        self.c3, self.c4 = int(c3), int(c4)
        self.stride3, self.stride4 = float(stride3), float(stride4)
        self.stencil_fraction = float(stencil_fraction)
        self.adapt3 = nn.Sequential(nn.Conv2d(c3, hidden, 1), nn.SiLU())
        self.adapt4 = nn.Sequential(nn.Conv2d(c4, hidden, 1), nn.SiLU())
        self.patch_body = nn.Sequential(
            nn.Conv2d(2 * hidden, encoded, 3, padding=1), nn.SiLU(),
            nn.Conv2d(encoded, encoded, 1), nn.SiLU())
        self.role_embedding = nn.Embedding(8, 8)
        self.scorer = nn.Sequential(nn.Linear(2 * encoded + 16, 64), nn.SiLU(), nn.Linear(64, 1))
        self.null_scorer = nn.Sequential(nn.Linear(2 * encoded + 13, 64), nn.SiLU(), nn.Linear(64, 1))
        angle = torch.arange(13) * 2 * math.pi / 13
        radius = torch.arange(1, 18) * 0.08 / 17
        displacement = torch.stack((angle.cos()[:, None] * radius,
                                    angle.sin()[:, None] * radius), -1).reshape(221, 2)
        self.register_buffer("displacements", torch.cat((displacement, torch.zeros(1, 2))))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 8), indexing="ij")
        self.register_buffer("stencil", torch.stack((xx, yy), -1).reshape(32, 2) / (2 * math.sqrt(2)))

    def _evidence(self, p3, p4, points, boxes, point_valid, input_shape):
        if points.ndim != 3 or points.shape[1:] != (9, 2):
            raise ValueError("points must be [B,9,2]")
        batch = len(points)
        if boxes.shape != (batch, 4) or point_valid.shape != (batch, 9) or input_shape.shape != (batch, 2):
            raise ValueError("Expected boxes[B,4], point_valid[B,9], input_shape[B,2]")
        if (p3.ndim != 4 or p4.ndim != 4 or p3.shape[:2] != (batch, self.c3)
                or p4.shape[:2] != (batch, self.c4)):
            raise ValueError("Feature batch/channel counts disagree with constructor")
        dtype = self.role_embedding.weight.dtype
        points, boxes, input_shape = points.to(dtype), boxes.to(dtype), input_shape.to(dtype)
        valid = finite(points, point_valid)
        box_valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
        safe = torch.where(valid[..., None], points, torch.zeros_like(points))
        box = torch.where(box_valid[:, None], boxes, boxes.new_tensor([0, 0, 1, 1]))
        size = box[:, 2:] - box[:, :2]
        diagonal = size.norm(dim=-1).clamp_min(1)
        center = (box[:, 2:] + box[:, :2]) * 0.5
        candidates = diagonal[:, None, None] * self.displacements[None]
        locations = safe[:, :8, None] + candidates[:, None, :-1]
        positions = (locations[:, :, :, None] + diagonal[:, None, None, None, None]
                     * self.stencil_fraction * self.stencil[None, None, None])
        inside = ((positions[..., 0] >= 0) & (positions[..., 1] >= 0)
                  & (positions[..., 0] < input_shape[:, None, None, None, 1])
                  & (positions[..., 1] < input_shape[:, None, None, None, 0]))
        evidence = torch.cat((
            sample(self.adapt3(p3.to(dtype)), positions, input_shape, self.stride3),
            sample(self.adapt4(p4.to(dtype)), positions, input_shape, self.stride4),
        ), -2) * inside[..., None, :]
        b, keypoints, candidates_count, channels, _ = evidence.shape
        patch = self.patch_body(evidence.reshape(b * keypoints * candidates_count, channels, 4, 8))
        pooled = torch.cat((patch.mean((-2, -1)), patch.amax((-2, -1))), -1)
        pooled = pooled.reshape(b, keypoints, candidates_count, -1)
        own = (safe[:, :8] - center[:, None]) / diagonal[:, None, None]
        geometry = torch.cat((
            own,
            (size / diagonal[:, None])[:, None].expand(-1, 8, -1),
            (size[:, 0] / size[:, 1].clamp_min(1e-6)).log()[:, None, None].expand(-1, 8, 1),
        ), -1)
        context = torch.cat((geometry, self.role_embedding.weight[None].expand(b, -1, -1)), -1)
        displacement = self.displacements[None, None, :-1].expand(b, 8, -1, -1) / 0.08
        descriptors = torch.cat((
            pooled,
            context[:, :, None].expand(-1, -1, candidates_count, -1),
            displacement,
            inside.to(dtype).mean(-1, keepdim=True),
        ), -1)
        supported = valid[:, :8] & box_valid[:, None] & inside.any(-1).any(-1)
        output = dict(points_raw=points, point_valid=valid, point_support=supported,
                      candidate_displacements=candidates, box_diagonal=diagonal,
                      coverage=inside.to(dtype).mean(-1), boxes=boxes)
        return output, descriptors, pooled, context

    def forward(self, p3, p4, points, boxes, point_valid, input_shape,
                lam=1.0, temperature=1.0, cap=None):
        output, descriptors, pooled, context = self._evidence(
            p3, p4, points, boxes, point_valid, input_shape)
        logits = self.scorer(descriptors).squeeze(-1)
        null = self.null_scorer(torch.cat((pooled.mean(-2), context), -1))
        output["logits"] = torch.cat((logits, null), -1)
        output["points"] = decode(output, temperature=temperature, lam=lam, cap=cap)
        return output


def _decode_options(lam, cap, points):
    if not math.isfinite(float(lam)) or float(lam) < 0:
        raise ValueError("lam must be finite and nonnegative")
    if cap is not None:
        cap = torch.as_tensor(cap, device=points.device, dtype=points.dtype)
        if cap.ndim == 0:
            cap = cap.expand(len(points))
        if cap.shape != (len(points),) or not torch.isfinite(cap).all() or (cap < 0).any():
            raise ValueError("cap must be a finite nonnegative scalar or [batch]")
    return cap


def _apply_delta(output, delta, cap):
    if cap is not None:
        delta = delta * torch.minimum(
            torch.ones_like(delta[..., 0]),
            cap[:, None] / delta.norm(dim=-1).clamp_min(1e-12))[..., None]
    points = output["points_raw"]
    corners = torch.where(output["point_support"][..., None], points[:, :8] + delta, points[:, :8])
    return torch.cat((corners, points[:, 8:]), 1)


def decode(output, temperature=1.0, lam=1.0, cap=None):
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    points = output["points_raw"]
    cap = _decode_options(lam, cap, points)
    if lam == 0:
        return points.clone()
    probability = (output["logits"] / float(temperature)).softmax(-1)
    delta = (probability[..., None] * output["candidate_displacements"][:, None]).sum(-2)
    return _apply_delta(output, delta * float(lam), cap)


def targets(output, gt_points, gt_valid):
    points = output["points_raw"]
    ground_truth = gt_points.to(points)
    valid = finite(ground_truth, gt_valid)[:, :8] & output["point_support"]
    residual = torch.where(valid[..., None], ground_truth[:, :8] - points[:, :8],
                           torch.zeros_like(points[:, :8]))
    distance = (output["candidate_displacements"][:, None] - residual[:, :, None]).square().sum(-1)
    sigma = output["box_diagonal"] * 0.08 / 17
    distribution = (-distance / (2 * sigma[:, None, None].square())).softmax(-1)
    return dict(distribution=distribution, support=valid, residual=residual, sigma=sigma)


def point_loss(output, gt_points, gt_valid):
    target = targets(output, gt_points, gt_valid)
    mask = target["support"]
    count = mask.sum(-1)
    cross_entropy = -(target["distribution"] * output["logits"].log_softmax(-1)).sum(-1)
    per_frame = (cross_entropy * mask).sum(-1) / count.clamp_min(1)
    return per_frame.sum() / (count > 0).sum().clamp_min(1)


class DirectResidualControl(GenericPointRefiner):
    def __init__(self, **config):
        super().__init__(**config)
        del self.scorer
        del self.null_scorer
        encoded = int(config.get("encoded", 24))
        self.descriptor_encoder = nn.Sequential(nn.Linear(2 * encoded + 16, 40), nn.SiLU())
        self.residual_head = nn.Sequential(nn.Linear(80 + 13, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, p3, p4, points, boxes, point_valid, input_shape,
                lam=1.0, temperature=1.0, cap=None):
        if float(temperature) != 1.0:
            raise ValueError("Direct regression has no temperature")
        output, descriptors, _, context = self._evidence(
            p3, p4, points, boxes, point_valid, input_shape)
        encoded = self.descriptor_encoder(descriptors)
        summary = torch.cat((encoded.mean(-2), encoded.amax(-2), context), -1)
        output["delta_normalized"] = 0.08 * self.residual_head(summary)
        output["points"] = decode_direct(output, lam=lam, cap=cap)
        return output


def decode_direct(output, lam=1.0, cap=None):
    points = output["points_raw"]
    cap = _decode_options(lam, cap, points)
    if lam == 0:
        return points.clone()
    delta = output["delta_normalized"] * output["box_diagonal"][:, None, None] * float(lam)
    return _apply_delta(output, delta, cap)


def direct_loss(output, gt_points, gt_valid):
    points = output["points_raw"]
    ground_truth = gt_points.to(points)
    mask = finite(ground_truth, gt_valid)[:, :8] & output["point_support"]
    residual = torch.where(mask[..., None], ground_truth[:, :8] - points[:, :8],
                           torch.zeros_like(points[:, :8]))
    target = residual / output["box_diagonal"][:, None, None]
    error = (output["delta_normalized"] - target).abs().mean(-1)
    count = mask.sum(-1)
    per_frame = (error * mask).sum(-1) / count.clamp_min(1)
    return per_frame.sum() / (count > 0).sum().clamp_min(1)


class DimensionConditionedPointRefiner(GenericPointRefiner):
    """P plus a zero-initialized candidate-logit residual from canonical 5D."""

    def __init__(self, **config):
        super().__init__(**config)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU())
        self.metadata_scorer = nn.Sequential(nn.Linear(27, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.metadata_scorer[-1].weight)
        nn.init.zeros_(self.metadata_scorer[-1].bias)

    def forward(self, p3, p4, points, boxes, point_valid, input_shape, *,
                dimension_context, lam=1.0, temperature=1.0, cap=None):
        if (dimension_context is None or dimension_context.shape != (len(points), 5)
                or not torch.isfinite(dimension_context).all()):
            raise ValueError("Finite dimension_context[B,5] is required")
        output = super().forward(p3, p4, points, boxes, point_valid, input_shape, lam=0)
        batch = len(points)
        embedding = self.metadata_encoder(dimension_context.to(self.role_embedding.weight))
        role = self.role_embedding.weight[None, :, None].expand(batch, -1, 222, -1)
        displacement = (self.displacements / 0.08)[None, None].expand(batch, 8, -1, -1)
        null = torch.zeros((batch, 8, 222, 1), device=points.device, dtype=embedding.dtype)
        null[:, :, -1] = 1
        features = torch.cat((embedding[:, None, None].expand(-1, 8, 222, -1),
                              role, displacement, null), -1)
        residual = self.metadata_scorer(features).squeeze(-1)
        output["base_logits"] = output["logits"]
        output["metadata_residual"] = residual
        output["logits"] = output["logits"] + residual
        output["points"] = decode(output, temperature=temperature, lam=lam, cap=cap)
        return output


def model_for_arm(arm: str, config: dict | None = None):
    if arm not in HEAD_ARMS:
        raise ValueError(f"Unknown head arm: {arm}")
    options = dict(HEAD_CONFIG if config is None else config)
    if arm == "D0":
        return DirectResidualControl(**options)
    if arm == "P0":
        return GenericPointRefiner(**options)
    return DimensionConditionedPointRefiner(**options)


def forward_arm(model, batch: dict, arm: str, **decode_options):
    keys = ("p3", "p4", "points", "boxes", "point_valid", "input_shape")
    arguments = [batch[key] for key in keys]
    if arm in ("P5", "P5_CONSTANT"):
        context = context_for_arm(batch.get("dimension_context"), arm)
        return model(*arguments, dimension_context=context, **decode_options)
    return model(*arguments, **decode_options)


def loss_for_arm(output, batch: dict, arm: str):
    return direct_loss(output, batch["gt_points"], batch["gt_valid"]) if arm == "D0" else point_loss(
        output, batch["gt_points"], batch["gt_valid"])

