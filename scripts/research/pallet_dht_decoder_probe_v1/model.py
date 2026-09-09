"""Small image/point/candidate decoder; the frozen detector is not part of it.

Both learned arms instantiate this exact class with the same seed. Candidate
construction differs outside the model: point-centred image grids versus
semantic-line intersections. A zero final projection gives exact identity at
initialization. Centroid, unavailable points and empty detections are preserved.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


INPUT_KEYS = ("points", "point_conf", "point_valid", "boxes", "diagonal", "detected",
              "point_visual", "global_visual", "candidate_xy", "candidate_valid",
              "candidate_geometry", "candidate_visual")


class CandidateRefiner(nn.Module):
    def __init__(self, *, residual_diagonal_fraction=.05, geometry_features=18):
        super().__init__()
        self.residual_diagonal_fraction = float(residual_diagonal_fraction)
        self.geometry_features = int(geometry_features)
        self.visual_norm = nn.LayerNorm(128)
        self.global_encoder = nn.Sequential(nn.Linear(166, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.point_encoder = nn.Sequential(nn.Linear(128, 32), nn.SiLU())
        self.corner_embedding = nn.Embedding(8, 8)
        self.candidate_encoder = nn.Sequential(nn.Linear(128 + geometry_features + 2, 64), nn.SiLU(),
                                               nn.Linear(64, 64), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(168, 64), nn.SiLU(), nn.Linear(64, 1))
        self.residual = nn.Sequential(nn.Linear(170, 64), nn.SiLU(), nn.Linear(64, 2))
        self.gate = nn.Linear(170, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, batch, *, return_diagnostics=False):
        points, mask = batch['points'], batch['candidate_valid'].bool()
        if points.shape[1:] != (9, 2) or mask.shape[1:] != (8, 49):
            raise ValueError('Expected9 points and8x49 candidates')
        if not bool(mask[..., 0].all()):
            raise ValueError('Identity candidate must be available')
        diagonal = batch['diagonal'].reshape(-1, 1, 1).clamp_min(1.)
        centre = (batch['boxes'][:, :2] + batch['boxes'][:, 2:]) * .5
        normalized = (points - centre[:, None]) / diagonal
        size = (batch['boxes'][:, 2:] - batch['boxes'][:, :2]) / diagonal[:, 0]
        point_summary = torch.cat((normalized.flatten(1), batch['point_conf'],
                                   batch['point_valid'].to(points.dtype), size), dim=1)
        global_context = self.global_encoder(torch.cat((point_summary,
            self.visual_norm(batch['global_visual'])), dim=1))
        local_context = self.point_encoder(self.visual_norm(batch['point_visual']))
        roles = self.corner_embedding(torch.arange(8, device=points.device))[None].expand(points.shape[0], -1, -1)
        context = torch.cat((global_context[:, None].expand(-1, 8, -1), local_context, roles), dim=-1)
        offsets = (batch['candidate_xy'] - points[:, :8, None]) / diagonal[:, None]
        encoded = self.candidate_encoder(torch.cat((self.visual_norm(batch['candidate_visual']),
            batch['candidate_geometry'], offsets), dim=-1))
        score = self.score(torch.cat((context[:, :, None].expand(-1, -1, 49, -1), encoded), dim=-1)).squeeze(-1)
        attention = score.masked_fill(~mask, -torch.inf).softmax(-1)
        pooled_features = (attention[..., None] * encoded).sum(2)
        pooled_offsets = (attention[..., None] * offsets).sum(2)
        combined = torch.cat((context, pooled_features, pooled_offsets), dim=-1)
        residual = self.residual_diagonal_fraction * diagonal * torch.tanh(self.residual(combined))
        gate = torch.tanh(self.gate(combined))
        # The candidate displacement is NOT globally capped: far ID-repair
        # candidates remain reachable. The signed gate is explicitly reported.
        delta = gate * pooled_offsets * diagonal + residual
        movable = batch['point_valid'][:, :8].bool() & batch['detected'].bool()[:, None]
        delta = torch.where(movable[..., None], delta, torch.zeros_like(delta))
        corners = points[:, :8] + delta
        output = torch.cat((corners, points[:, 8:]), dim=1)
        if return_diagnostics:
            return output, dict(attention=attention, delta=delta, signed_gate=gate,
                residual=residual,
                weighted_candidate_xy=points[:, :8] + pooled_offsets * diagonal,
                weighted_candidate_displacement_normalized=pooled_offsets)
        return output


def load_refiner(checkpoint: str | Path, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu')
    if saved.get('schema') != 'pallet_dht_decoder_checkpoint_v1' or not saved.get('complete'):
        raise ValueError('Expected completed decoder checkpoint')
    model = CandidateRefiner(**saved['model_config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to(device).eval()
    return model, saved


def to_device(arrays, indices, device):
    return {key: torch.as_tensor(arrays[key][indices]).to(device) for key in INPUT_KEYS}
