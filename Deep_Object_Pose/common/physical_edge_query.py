"""Twelve fixed cuboid-edge-role queries over the frozen A1 F50 feature.

The dense twelve-edge field failed because its activation diffused on real data
until the correct line was no longer in the top five candidates.  This replaces
the field with twelve queries, each permanently bound to one edge role, so a
role's geometry is read out directly instead of being searched for.

The roles are roles, not physical timber.  Under `camera_dynamic_0123_v4` the
corner indices are assigned per frame from the camera's viewpoint, so role 0 is
"the edge between whichever corners are currently indexed 0 and 1", not one
particular wooden edge tracked across viewpoints.

Query k is edge role k for the whole run.  There is no Hungarian matching
anywhere: an assignment step would let the head relabel its own outputs and make
a role-conditioned claim unfalsifiable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

GRID = 50
D_MODEL = 64
N_ROLES = 12


def sine_position_encoding(grid: int = GRID, channels: int = D_MODEL) -> torch.Tensor:
    """Fixed 2D sine/cosine encoding, (1, channels, grid, grid).  Not learned."""
    if channels % 4 != 0:
        raise ValueError("channels must be divisible by 4")
    quarter = channels // 4
    y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                          torch.arange(grid, dtype=torch.float32), indexing="ij")
    frequency = torch.exp(torch.arange(quarter, dtype=torch.float32)
                          * (-math.log(10000.0) / max(quarter - 1, 1)))
    parts = []
    for axis in (x, y):
        scaled = axis[None] * frequency[:, None, None]
        parts.extend((scaled.sin(), scaled.cos()))
    return torch.cat(parts, dim=0)[None]


class PhysicalEdgeQueryHead(nn.Module):
    """F50 -> twelve edge-role segments in belief-grid coordinates.

    Outputs per role: centre, an undirected unit direction, a positive
    half-length, and a support logit.  Support is geometric frame support, not
    visibility -- the paper dataset carries no trustworthy per-edge visibility.
    """

    def __init__(self, in_channels: int = 128, d_model: int = D_MODEL,
                 layers: int = 2, heads: int = 4, ffn: int = 128,
                 grid: int = GRID, roles: int = N_ROLES) -> None:
        super().__init__()
        self.grid = grid
        self.roles = roles
        self.project = nn.Conv2d(in_channels, d_model, 1)
        self.register_buffer("position", sine_position_encoding(grid, d_model),
                             persistent=False)
        self.queries = nn.Embedding(roles, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, ffn, dropout=0.0,
                                           batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.to_centre = nn.Linear(d_model, 2)
        self.to_direction = nn.Linear(d_model, 2)
        self.to_half_length = nn.Linear(d_model, 1)
        self.to_support = nn.Linear(d_model, 1)

    def forward(self, feature: torch.Tensor) -> dict:
        batch = feature.shape[0]
        memory = (self.project(feature) + self.position).flatten(2).transpose(1, 2)
        query = self.queries.weight[None].expand(batch, -1, -1)
        decoded = self.decoder(query, memory)

        # Centre is left unbounded: an edge role may legitimately sit outside
        # the grid, and squashing it would fabricate an in-frame answer.
        centre = self.to_centre(decoded) * self.grid
        direction = self.to_direction(decoded)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        half_length = torch.nn.functional.softplus(self.to_half_length(decoded)) + 1e-3
        support = self.to_support(decoded)
        return {"centre": centre, "direction": direction,
                "half_length": half_length, "support_logit": support,
                "p0": centre - half_length * direction,
                "p1": centre + half_length * direction,
                "attention_memory": memory}


def orientation_loss(direction: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """1 - |cos| : undirected, so a flipped segment is not penalised."""
    cosine = (direction * target).sum(dim=-1).abs().clamp(max=1.0)
    weight = mask.clamp_min(0.0)
    return ((1.0 - cosine) * weight).sum() / weight.sum().clamp_min(1.0)
