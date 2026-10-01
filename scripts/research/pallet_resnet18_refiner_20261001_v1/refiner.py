"""Local displacement P and matched direct-regression D for frozen DOPE features.

The original P and D architectures, loss reductions and decoding are retained.
Only channel counts and the two sampling strides are configurable. Input points,
boxes, input_shape and cap use the SAME network-input coordinate system. Sampling
uses the original P cell-center convention: feature index j corresponds to
(j + .5) * stride; the external backbone adapter owns coordinate conversion.
There is no detector, checkpoint loading, GT access or instance selection here.

References (unchanged):
  pallet_final_ml_contribution_test_v1/generic_point_refiner.py
  pallet_sensors_refinement_closeout_v1/direct_residual_control.py
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def finite(points, valid):
    return valid.bool() & torch.isfinite(points).all(-1) & ~(points == -1).all(-1)


def sample(feature, positions, input_shape, stride):
    """Original align_corners=False bilinear sampler with padded cells masked."""
    b, c, h, w = feature.shape
    ys = (torch.arange(h, device=feature.device, dtype=feature.dtype) + .5) * stride
    xs = (torch.arange(w, device=feature.device, dtype=feature.dtype) + .5) * stride
    support = ((ys[None, :, None] < input_shape[:, 0, None, None])
               & (xs[None, None, :] < input_shape[:, 1, None, None]))
    feature = feature * support[:, None]
    grid = 2 * positions / positions.new_tensor([stride * w, stride * h]) - 1
    out = F.grid_sample(feature, grid.reshape(b, -1, positions.shape[-2], 2),
                        mode='bilinear', padding_mode='zeros', align_corners=False)
    return out.reshape(b, c, *positions.shape[1:-1]).permute(0, 2, 3, 1, 4)


class GenericPointRefiner(nn.Module):
    """222-way local displacement distribution; DOPE channel/stride defaults."""

    def __init__(self, c3=256, c4=128, hidden=16, encoded=24,
                 stencil_fraction=.25, stride3=4, stride4=8):
        super().__init__()
        if any(int(v) != v or v < 1 for v in (c3, c4, hidden, encoded)):
            raise ValueError('Channel counts must be positive integers')
        if any(not math.isfinite(float(v)) or v <= 0
               for v in (stride3, stride4, stencil_fraction)):
            raise ValueError('Strides and stencil_fraction must be positive finite')
        self.c3, self.c4 = int(c3), int(c4)
        self.stride3, self.stride4 = float(stride3), float(stride4)
        self.stencil_fraction = float(stencil_fraction)
        self.adapt3 = nn.Sequential(nn.Conv2d(c3, hidden, 1), nn.SiLU())
        self.adapt4 = nn.Sequential(nn.Conv2d(c4, hidden, 1), nn.SiLU())
        self.patch_body = nn.Sequential(nn.Conv2d(2 * hidden, encoded, 3, padding=1),
                                       nn.SiLU(), nn.Conv2d(encoded, encoded, 1), nn.SiLU())
        self.role_embedding = nn.Embedding(8, 8)
        self.scorer = nn.Sequential(nn.Linear(2 * encoded + 16, 64), nn.SiLU(), nn.Linear(64, 1))
        self.null_scorer = nn.Sequential(nn.Linear(2 * encoded + 13, 64), nn.SiLU(), nn.Linear(64, 1))
        angle = torch.arange(13) * 2 * math.pi / 13
        radius = torch.arange(1, 18) * .08 / 17
        displacement = torch.stack([angle.cos()[:, None] * radius,
                                    angle.sin()[:, None] * radius], -1).reshape(221, 2)
        self.register_buffer('displacements', torch.cat([displacement, torch.zeros(1, 2)]))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 8), indexing='ij')
        self.register_buffer('stencil', torch.stack([xx, yy], -1).reshape(32, 2) / (2 * math.sqrt(2)))

    def _evidence(self, p3, p4, points, boxes, point_valid, input_shape):
        if points.ndim != 3 or points.shape[1:] != (9, 2):
            raise ValueError('points must be [B,9,2]')
        b = len(points)
        if (boxes.shape != (b, 4) or point_valid.shape != (b, 9)
                or input_shape.shape != (b, 2)):
            raise ValueError('Expected boxes[B,4], point_valid[B,9], input_shape[B,2]')
        if (p3.ndim != 4 or p4.ndim != 4 or p3.shape[:2] != (b, self.c3)
                or p4.shape[:2] != (b, self.c4)):
            raise ValueError('Feature batch/channel counts disagree with constructor')
        dtype = self.role_embedding.weight.dtype
        points, boxes, input_shape = points.to(dtype), boxes.to(dtype), input_shape.to(dtype)
        valid = finite(points, point_valid)
        box_valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
        safe = torch.where(valid[..., None], points, torch.zeros_like(points))
        bb = torch.where(box_valid[:, None], boxes, boxes.new_tensor([0, 0, 1, 1]))
        size = bb[:, 2:] - bb[:, :2]
        diag = size.norm(dim=-1).clamp_min(1)
        center = (bb[:, 2:] + bb[:, :2]) * .5
        candidates = diag[:, None, None] * self.displacements[None]
        locations = safe[:, :8, None] + candidates[:, None, :-1]
        positions = (locations[:, :, :, None] + diag[:, None, None, None, None]
                     * self.stencil_fraction * self.stencil[None, None, None])
        inside = ((positions[..., 0] >= 0) & (positions[..., 1] >= 0)
                  & (positions[..., 0] < input_shape[:, None, None, None, 1])
                  & (positions[..., 1] < input_shape[:, None, None, None, 0]))
        evidence = torch.cat([
            sample(self.adapt3(p3.to(dtype)), positions, input_shape, self.stride3),
            sample(self.adapt4(p4.to(dtype)), positions, input_shape, self.stride4),
        ], -2) * inside[..., None, :]
        b, k, c, channels, _ = evidence.shape
        patch = self.patch_body(evidence.reshape(b * k * c, channels, 4, 8))
        pooled = torch.cat([patch.mean((-2, -1)), patch.amax((-2, -1))], -1).reshape(b, k, c, -1)
        own = (safe[:, :8] - center[:, None]) / diag[:, None, None]
        geom = torch.cat([
            own, (size / diag[:, None])[:, None].expand(-1, 8, -1),
            (size[:, 0] / size[:, 1].clamp_min(1e-6)).log()[:, None, None].expand(-1, 8, 1),
        ], -1)
        context = torch.cat([geom, self.role_embedding.weight[None].expand(b, -1, -1)], -1)
        displacement = self.displacements[None, None, :-1].expand(b, 8, -1, -1) / .08
        descriptors = torch.cat([
            pooled, context[:, :, None].expand(-1, -1, c, -1), displacement,
            inside.to(dtype).mean(-1, keepdim=True),
        ], -1)
        supported = valid[:, :8] & box_valid[:, None] & inside.any(-1).any(-1)
        out = dict(points_raw=points, point_valid=valid, point_support=supported,
                   candidate_displacements=candidates, box_diagonal=diag,
                   coverage=inside.to(dtype).mean(-1), boxes=boxes)
        return out, descriptors, pooled, context

    def forward(self, p3, p4, points, boxes, point_valid, input_shape,
                lam=1., temperature=1., cap=None):
        out, descriptors, pooled, context = self._evidence(
            p3, p4, points, boxes, point_valid, input_shape)
        logits = self.scorer(descriptors).squeeze(-1)
        null = self.null_scorer(torch.cat([pooled.mean(-2), context], -1))
        out['logits'] = torch.cat([logits, null], -1)
        out['points'] = decode(out, temperature=temperature, lam=lam, cap=cap)
        return out


def _decode_options(lam, cap, points):
    if not math.isfinite(float(lam)) or float(lam) < 0:
        raise ValueError('lam must be finite and nonnegative')
    if cap is not None:
        cap = torch.as_tensor(cap, device=points.device, dtype=points.dtype)
        if cap.ndim == 0:
            cap = cap.expand(len(points))
        if cap.shape != (len(points),) or not torch.isfinite(cap).all() or (cap < 0).any():
            raise ValueError('cap must be a finite nonnegative scalar or [batch]')
    return cap


def _apply_delta(output, delta, cap):
    if cap is not None:
        delta = delta * torch.minimum(torch.ones_like(delta[..., 0]),
                                      cap[:, None] / delta.norm(dim=-1).clamp_min(1e-12))[..., None]
    points = output['points_raw']
    corners = torch.where(output['point_support'][..., None], points[:, :8] + delta, points[:, :8])
    return torch.cat([corners, points[:, 8:]], 1)


def decode(output, temperature=1., lam=1., cap=None):
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError('temperature must be finite and positive')
    points = output['points_raw']
    cap = _decode_options(lam, cap, points)
    if lam == 0:
        return points.clone()
    probability = (output['logits'] / float(temperature)).softmax(-1)
    delta = (probability[..., None] * output['candidate_displacements'][:, None]).sum(-2) * float(lam)
    return _apply_delta(output, delta, cap)


def targets(output, gt_points, gt_valid):
    points = output['points_raw']
    gt = gt_points.to(points)
    valid = finite(gt, gt_valid)[:, :8] & output['point_support']
    residual = torch.where(valid[..., None], gt[:, :8] - points[:, :8], torch.zeros_like(points[:, :8]))
    distance = (output['candidate_displacements'][:, None] - residual[:, :, None]).square().sum(-1)
    sigma = output['box_diagonal'] * .08 / 17
    q = (-distance / (2 * sigma[:, None, None].square())).softmax(-1)
    return dict(distribution=q, support=valid, residual=residual, sigma=sigma)


def loss(output, gt_points, gt_valid):
    """P soft-target CE: supported corners/frame, then frames with support."""
    target = targets(output, gt_points, gt_valid)
    mask = target['support']
    count = mask.sum(-1)
    ce = -(target['distribution'] * output['logits'].log_softmax(-1)).sum(-1)
    per = (ce * mask).sum(-1) / count.clamp_min(1)
    return per.sum() / (count > 0).sum().clamp_min(1)


class DirectResidualControl(GenericPointRefiner):
    """Original D readout/L1 package on exactly the same P sampling evidence."""

    def __init__(self, c3=256, c4=128, hidden=16, encoded=24,
                 stencil_fraction=.25, stride3=4, stride4=8):
        super().__init__(c3=c3, c4=c4, hidden=hidden, encoded=encoded,
                         stencil_fraction=stencil_fraction, stride3=stride3, stride4=stride4)
        del self.scorer
        del self.null_scorer
        self.descriptor_encoder = nn.Sequential(nn.Linear(2 * encoded + 16, 40), nn.SiLU())
        self.residual_head = nn.Sequential(nn.Linear(80 + 13, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, p3, p4, points, boxes, point_valid, input_shape,
                lam=1., temperature=1., cap=None, return_descriptors=False):
        if float(temperature) != 1.:
            raise ValueError('Direct regression has no temperature; use temperature=1')
        out, descriptors, _, context = self._evidence(p3, p4, points, boxes, point_valid, input_shape)
        encoded = self.descriptor_encoder(descriptors)
        summary = torch.cat([encoded.mean(-2), encoded.amax(-2), context], -1)
        out['delta_normalized'] = .08 * self.residual_head(summary)
        out['points'] = decode_direct(out, lam=lam, cap=cap)
        if return_descriptors:
            out['candidate_descriptors'] = descriptors
        return out


def decode_direct(output, lam=1., cap=None):
    points = output['points_raw']
    cap = _decode_options(lam, cap, points)
    if lam == 0:
        return points.clone()
    delta = output['delta_normalized'] * output['box_diagonal'][:, None, None] * float(lam)
    return _apply_delta(output, delta, cap)


def direct_loss(output, gt_points, gt_valid):
    """Original D normalized L1; same per-frame validity weighting as P."""
    points = output['points_raw']
    gt = gt_points.to(points)
    mask = finite(gt, gt_valid)[:, :8] & output['point_support']
    residual = torch.where(mask[..., None], gt[:, :8] - points[:, :8], torch.zeros_like(points[:, :8]))
    target = residual / output['box_diagonal'][:, None, None]
    error = (output['delta_normalized'] - target).abs().mean(-1)
    count = mask.sum(-1)
    per = (error * mask).sum(-1) / count.clamp_min(1)
    return per.sum() / (count > 0).sum().clamp_min(1)
