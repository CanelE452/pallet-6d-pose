"""New ablation: shared P evidence, direct residual regression (not PoseFix)."""
from __future__ import annotations
import importlib.util
import math
import os
from pathlib import Path
import torch
from torch import nn


def _reference():
    # Place this file in scripts/research/pallet_sensors_refinement_closeout_v1/.
    root = Path(os.environ['PALLET_REPO_ROOT']) if 'PALLET_REPO_ROOT' in os.environ else Path(__file__).resolve().parents[3]
    path = root / 'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py'
    if not path.is_file():
        raise FileNotFoundError(f'Existing P implementation not found: {path}')
    spec = importlib.util.spec_from_file_location('_frozen_p_reference_for_direct', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _reference()


class DirectResidualControl(REF.GenericPointRefiner):
    """Fresh weights only. Use the locked P config; do not load trained P weights."""
    def __init__(self, c3=64, c4=128, hidden=16, encoded=24, stencil_fraction=.25):
        super().__init__(c3=c3, c4=c4, hidden=hidden, encoded=encoded,
                         stencil_fraction=stencil_fraction)
        del self.scorer
        del self.null_scorer
        # With the current locked P config: 19,450 params vs P's 18,962.
        self.descriptor_encoder = nn.Sequential(nn.Linear(2 * encoded + 16, 40), nn.SiLU())
        self.residual_head = nn.Sequential(nn.Linear(80 + 13, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, p3, p4, points, boxes, point_valid, input_shape,
                lam=1., cap=None, return_descriptors=False):
        dtype = self.role_embedding.weight.dtype
        points, boxes, input_shape = points.to(dtype), boxes.to(dtype), input_shape.to(dtype)
        valid = REF.finite(points, point_valid)
        box_valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
        safe = torch.where(valid[..., None], points, torch.zeros_like(points))
        bb = torch.where(box_valid[:, None], boxes, boxes.new_tensor([0, 0, 1, 1]))
        size = bb[:, 2:] - bb[:, :2]
        diag = size.norm(dim=-1).clamp_min(1)
        center = (bb[:, 2:] + bb[:, :2]) * .5
        candidates = diag[:, None, None] * self.displacements[None]
        locations = safe[:, :8, None] + candidates[:, None, :-1]
        positions = locations[:, :, :, None] + diag[:, None, None, None, None] * self.stencil_fraction * self.stencil[None, None, None]
        inside = ((positions[..., 0] >= 0) & (positions[..., 1] >= 0)
                  & (positions[..., 0] < input_shape[:, None, None, None, 1])
                  & (positions[..., 1] < input_shape[:, None, None, None, 0]))
        evidence = torch.cat([
            REF.sample(self.adapt3(p3.to(dtype)), positions, input_shape, 8),
            REF.sample(self.adapt4(p4.to(dtype)), positions, input_shape, 16),
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
            pooled, context[:, :, None].expand(-1, -1, c, -1),
            displacement, inside.to(dtype).mean(-1, keepdim=True),
        ], -1)
        encoded = self.descriptor_encoder(descriptors)
        summary = torch.cat([encoded.mean(-2), encoded.amax(-2), context], -1)
        # Unbounded normalized regression output; final inference cap is shared with P.
        delta_normalized = .08 * self.residual_head(summary)
        support = valid[:, :8] & box_valid[:, None] & inside.any(-1).any(-1)
        out = dict(points_raw=points, point_valid=valid, point_support=support,
                   box_diagonal=diag, delta_normalized=delta_normalized)
        out['points'] = decode_direct(out, lam=lam, cap=cap)
        if return_descriptors:
            out['candidate_descriptors'] = descriptors
        return out


def decode_direct(output, lam=1., cap=None):
    if not math.isfinite(float(lam)) or float(lam) < 0:
        raise ValueError('lam must be finite and nonnegative')
    points = output['points_raw']
    if lam == 0:
        return points.clone()
    delta = output['delta_normalized'] * output['box_diagonal'][:, None, None] * float(lam)
    if cap is not None:
        cap = torch.as_tensor(cap, device=points.device, dtype=points.dtype)
        if cap.ndim == 0:
            cap = cap.expand(len(points))
        if cap.shape != (len(points),) or not torch.isfinite(cap).all() or (cap < 0).any():
            raise ValueError('cap must be a finite nonnegative scalar or [batch]')
        factor = torch.minimum(torch.ones_like(delta[..., 0]), cap[:, None] / delta.norm(dim=-1).clamp_min(1e-12))
        delta = delta * factor[..., None]
    corners = torch.where(output['point_support'][..., None], points[:, :8] + delta, points[:, :8])
    return torch.cat([corners, points[:, 8:]], 1)


def direct_loss(output, gt_points, gt_valid):
    """Normalized L1 residual regression; same per-frame validity weighting as P."""
    points = output['points_raw']
    gt = gt_points.to(points)
    mask = REF.finite(gt, gt_valid)[:, :8] & output['point_support']
    safe_residual = torch.where(mask[..., None], gt[:, :8] - points[:, :8], torch.zeros_like(points[:, :8]))
    target = safe_residual / output['box_diagonal'][:, None, None]
    error = (output['delta_normalized'] - target).abs().mean(-1)
    count = mask.sum(-1)
    per_frame = (error * mask).sum(-1) / count.clamp_min(1)
    return per_frame.sum() / (count > 0).sum().clamp_min(1)
