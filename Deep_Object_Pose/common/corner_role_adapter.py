"""Corner-role encoder, prototypes and role-conditioned FiLM for DOPE.

The frozen shared feature separates pallet from background (AUC 0.77-0.86 against
random locations) but not one corner from another on the pallet itself (0.59-0.64
against the wrong peak).  Everything that tried to repair the belief after the
fact failed, so this does not touch the belief at all: it learns a feature that
carries corner identity and uses it to modulate what stages 4-6 read.

The FiLM output convolution is exactly zero-initialised, so with the flag on and
no training the network reproduces ep57 bit for bit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CORNERS = 8
ROLE_DIM = 32
TEMPERATURE = 0.10
BELIEF_GRID = 50
HIGH_GRID = 100
SHARED_CHANNELS = 128

# yaw+180 about the pallet up axis, in the camera-facing 0123 convention.
# The whole frame takes this permutation or none of it -- never per corner.
YAW180_PERMUTATION = (5, 4, 7, 6, 1, 0, 3, 2)


def find_feature_layer(trunk: nn.Sequential, sample: torch.Tensor, grid: int
                       ) -> tuple[int, int]:
    """Last layer of the trunk whose output is grid x grid.  Never hardcoded."""
    found = None
    activation = sample
    with torch.no_grad():
        for index, layer in enumerate(trunk):
            activation = layer(activation)
            if activation.shape[-2:] == (grid, grid):
                found = (index, int(activation.shape[1]))
    if found is None:
        raise RuntimeError(f"no {grid}x{grid} feature in the trunk")
    return found


class CornerRoleEncoder(nn.Module):
    """F100 + F50 -> a 32-channel role embedding at the belief resolution."""

    def __init__(self, channels_high: int, channels_low: int = SHARED_CHANNELS
                 ) -> None:
        super().__init__()
        assert channels_low == SHARED_CHANNELS, channels_low
        self.high = nn.Sequential(
            nn.Conv2d(channels_high, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.low = nn.Sequential(
            nn.Conv2d(channels_low, 64, 1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, ROLE_DIM, 3, padding=1),
            nn.GroupNorm(8, ROLE_DIM), nn.ReLU(inplace=True))
        self.prototypes = nn.Parameter(torch.randn(N_CORNERS, ROLE_DIM) * 0.1)

    def forward(self, feature_high: torch.Tensor, feature_low: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        assert feature_high.shape[-2:] == (HIGH_GRID, HIGH_GRID), feature_high.shape
        assert feature_low.shape[-2:] == (BELIEF_GRID, BELIEF_GRID), feature_low.shape
        assert feature_low.shape[1] == SHARED_CHANNELS, feature_low.shape
        embedding = self.fuse(torch.cat([self.high(feature_high),
                                         self.low(feature_low)], dim=1))
        normalised = F.normalize(embedding, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        # cosine similarity as a classifier logit -- no sigmoid anywhere
        score = torch.einsum("bchw,kc->bkhw", normalised, prototypes) / TEMPERATURE
        return {"embedding": embedding, "normalised": normalised, "score": score}


class RoleConditionedFiLM(nn.Module):
    """Per-stage modulation of the shared feature, zero-initialised."""

    def __init__(self, channels: int = SHARED_CHANNELS) -> None:
        super().__init__()
        self.channels = channels
        self.trunk = nn.Sequential(
            nn.Conv2d(ROLE_DIM + N_CORNERS, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(64, 2 * channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, embedding: torch.Tensor, score: torch.Tensor,
                shared: torch.Tensor) -> torch.Tensor:
        conditioning = torch.cat([embedding, torch.softmax(score, dim=1)], dim=1)
        gamma, beta = self.out(self.trunk(conditioning)).chunk(2, dim=1)
        return shared * (1.0 + gamma) + beta


# ============================================================================
# sampling helpers
# ============================================================================
def bilinear_sample(maps: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample B x C x H x W at B x N x 2 belief-grid coordinates.  No rounding."""
    height, width = maps.shape[-2:]
    x = points[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = points[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1)[:, :, None, :]     # B x N x 1 x 2
    sampled = F.grid_sample(maps, grid, mode="bilinear", align_corners=True)
    return sampled[..., 0].transpose(1, 2)                # B x N x C


def valid_corner_mask(points: torch.Tensor, flags: torch.Tensor,
                      grid: int = BELIEF_GRID) -> torch.Tensor:
    """Validity follows the transformed GT centre, not an empty raster."""
    inside = ((points[..., 0] >= 0) & (points[..., 0] < grid)
              & (points[..., 1] >= 0) & (points[..., 1] < grid))
    return (flags > 0) & inside & torch.isfinite(points).all(dim=-1)


def apply_permutation(labels: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(YAW180_PERMUTATION, device=labels.device,
                            dtype=labels.dtype)
    return table[labels]
