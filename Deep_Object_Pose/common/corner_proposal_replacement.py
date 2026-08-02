"""Corner proposal replacement head for DOPE.

The frozen-feature screens showed that the shared 50x50 feature separates pallet
from background but not one corner from another, so a corrector reading it
cannot know where to move a confidently-wrong peak.  This module therefore does
not correct a frozen belief: it produces its own full-map proposal from a jointly
trained feature and blends it into the belief through a per-corner gate.

    H_ref,i = (1 - g_i) * H_base,i + g_i * sigmoid(Q_i)

Belief operating range was audited on the cached ep57 stage-6 output over the
mechanism set: min -0.030, max 1.004, i.e. a raw convolution output regressed
onto a Gaussian whose peak is 1.0.  sigmoid keeps the proposal inside that same
range so the existing decoder threshold and peak semantics are untouched.

The gate bias starts at -4.6 (g ~ 0.01) so training begins essentially at the
ep57 behaviour while gradients still flow everywhere.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CORNERS = 8
QUERY_DIM = 64
GATE_INIT_BIAS = -4.6
BELIEF_GRID = 50
HIGH_GRID = 100


def find_feature_layer(trunk: nn.Sequential, sample: torch.Tensor, grid: int
                       ) -> tuple[int, int]:
    """Last layer of `trunk` whose output is grid x grid.  Never hardcoded."""
    found: tuple[int, int] | None = None
    activation = sample
    with torch.no_grad():
        for index, layer in enumerate(trunk):
            activation = layer(activation)
            if activation.shape[-2:] == (grid, grid):
                found = (index, int(activation.shape[1]))
    if found is None:
        raise RuntimeError(f"no {grid}x{grid} feature in the trunk")
    return found


def canonical_corner_coordinates(width: float, depth: float, height: float
                                 ) -> np.ndarray:
    """The eight cuboid corners in the project's camera-facing 0123 order."""
    from pallet_graph_geometry import make_corners  # local import: shared source

    return np.asarray(make_corners(width, depth, height)[:8], dtype=np.float32)


class CornerProposalReplacement(nn.Module):
    """Fuse F100+F50, emit 8 full-map proposals, blend them per corner."""

    def __init__(self, channels_high: int, channels_low: int) -> None:
        super().__init__()
        self.channels_high = channels_high
        self.channels_low = channels_low

        self.project_high = nn.Conv2d(channels_high, 64, 3, stride=2, padding=1)
        self.project_low = nn.Conv2d(channels_low, 64, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.pixel = nn.Conv2d(128, QUERY_DIM, 1)

        self.corner_embedding = nn.Embedding(N_CORNERS, QUERY_DIM)
        # 3 canonical coordinates + 3 dimensions
        self.query_context = nn.Sequential(
            nn.Linear(6, QUERY_DIM), nn.ReLU(inplace=True),
            nn.Linear(QUERY_DIM, QUERY_DIM),
        )
        self.query_out = nn.Linear(QUERY_DIM, QUERY_DIM)

        # gate sees the query, a global feature summary and belief statistics
        self.gate = nn.Sequential(
            nn.Linear(QUERY_DIM + 128 + 8, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, GATE_INIT_BIAS)

    # -- belief statistics -------------------------------------------------
    @staticmethod
    def belief_statistics(h4: torch.Tensor, h5: torch.Tensor, h6: torch.Tensor
                          ) -> torch.Tensor:
        """Per-corner summary of stages 4-6: B x 8 x 8."""
        def summarise(heat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch, corners, height, width = heat.shape
            flat = heat.reshape(batch, corners, -1)
            weights = torch.softmax(flat / 0.1, dim=-1)
            device = heat.device
            xs = torch.arange(width, device=device, dtype=heat.dtype)
            ys = torch.arange(height, device=device, dtype=heat.dtype)
            grid_x = xs.repeat(height)
            grid_y = ys.repeat_interleave(width)
            centre = torch.stack([(weights * grid_x).sum(-1),
                                  (weights * grid_y).sum(-1)], dim=-1)
            entropy = -(weights * (weights + 1e-9).log()).sum(-1)
            return centre, entropy

        centre4, _ = summarise(h4)
        centre6, entropy6 = summarise(h6)
        peak4 = h4.amax(dim=(-2, -1))
        peak5 = h5.amax(dim=(-2, -1))
        peak6 = h6.amax(dim=(-2, -1))
        drift = centre6 - centre4
        return torch.stack([
            peak4, peak5, peak6, entropy6 / 10.0,
            centre6[..., 0] / BELIEF_GRID, centre6[..., 1] / BELIEF_GRID,
            drift[..., 0] / BELIEF_GRID, drift[..., 1] / BELIEF_GRID,
        ], dim=-1)

    # -- forward -----------------------------------------------------------
    def forward(self, feature_high: torch.Tensor, feature_low: torch.Tensor,
                h4: torch.Tensor, h5: torch.Tensor, h6: torch.Tensor,
                canonical: torch.Tensor, dimensions: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        assert feature_high.shape[-2:] == (HIGH_GRID, HIGH_GRID), feature_high.shape
        assert feature_low.shape[-2:] == (BELIEF_GRID, BELIEF_GRID), feature_low.shape
        batch = feature_low.shape[0]

        fused = self.fuse(torch.cat([self.project_high(feature_high),
                                     self.project_low(feature_low)], dim=1))
        pixels = self.pixel(fused)                       # B x 64 x 50 x 50
        pixels_flat = pixels.reshape(batch, QUERY_DIM, -1)

        ids = torch.arange(N_CORNERS, device=fused.device)
        context = torch.cat([canonical, dimensions[:, None, :].expand(
            batch, N_CORNERS, 3)], dim=-1)               # B x 8 x 6
        queries = self.query_out(
            self.corner_embedding(ids)[None] + self.query_context(context))

        logits = torch.einsum("bqc,bcp->bqp", queries, pixels_flat)
        logits = logits / (QUERY_DIM ** 0.5)
        proposal = logits.reshape(batch, N_CORNERS, BELIEF_GRID, BELIEF_GRID)

        statistics = self.belief_statistics(h4[:, :N_CORNERS], h5[:, :N_CORNERS],
                                            h6[:, :N_CORNERS])
        summary = fused.mean(dim=(-2, -1))[:, None, :].expand(batch, N_CORNERS, 128)
        gate = torch.sigmoid(self.gate(
            torch.cat([queries, summary, statistics], dim=-1))).squeeze(-1)

        base = h6[:, :N_CORNERS]
        transformed = torch.sigmoid(proposal)
        refined = (1.0 - gate[..., None, None]) * base \
            + gate[..., None, None] * transformed
        return {"base": base, "proposal": proposal,
                "proposal_transformed": transformed,
                "gate": gate, "refined": refined}


# ============================================================================
# losses
# ============================================================================
def spatial_probability(maps: torch.Tensor, temperature: float = 0.1
                        ) -> torch.Tensor:
    batch, corners, height, width = maps.shape
    return torch.softmax(maps.reshape(batch, corners, -1) / temperature,
                         dim=-1).reshape(batch, corners, height, width)


def neighbourhood_nll(probability: torch.Tensor, centres: torch.Tensor,
                      valid: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """-log of the probability mass inside the (2r+1)^2 window around GT."""
    batch, corners, height, width = probability.shape
    device = probability.device
    ys = torch.arange(height, device=device)[None, None, :, None]
    xs = torch.arange(width, device=device)[None, None, None, :]
    cx = centres[..., 0][..., None, None]
    cy = centres[..., 1][..., None, None]
    window = ((xs - cx.round()).abs() <= radius) & ((ys - cy.round()).abs() <= radius)
    mass = (probability * window).sum(dim=(-2, -1)).clamp_min(1e-9)
    loss = -mass.log()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def expected_coordinate(probability: torch.Tensor) -> torch.Tensor:
    batch, corners, height, width = probability.shape
    device = probability.device
    xs = torch.arange(width, device=device, dtype=probability.dtype)
    ys = torch.arange(height, device=device, dtype=probability.dtype)
    return torch.stack([(probability.sum(-2) * xs).sum(-1),
                        (probability.sum(-1) * ys).sum(-1)], dim=-1)


def coordinate_huber(predicted: torch.Tensor, centres: torch.Tensor,
                     valid: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    residual = (predicted - centres) / diagonal[:, None, None].clamp_min(1e-6)
    loss = F.huber_loss(residual, torch.zeros_like(residual), reduction="none",
                        delta=1.0).sum(-1)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def proposal_objective(maps: torch.Tensor, centres: torch.Tensor,
                       valid: torch.Tensor, diagonal: torch.Tensor
                       ) -> torch.Tensor:
    probability = spatial_probability(maps)
    return (neighbourhood_nll(probability, centres, valid)
            + 0.25 * coordinate_huber(expected_coordinate(probability),
                                      centres, valid, diagonal))
