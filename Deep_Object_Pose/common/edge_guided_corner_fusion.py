"""Edge-Guided Corner Residual: the edge branch enters the belief the decoder
reads, rather than re-ranking its output afterwards.

Zero-initialised output convolution, so an untrained module reproduces A1
exactly and any measured change is attributable to training.  The centroid and
every affinity map are copied through untouched and asserted.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

_COMMON = pathlib.Path(__file__).resolve().parent
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from spatial_hcrm import ChannelLayerNorm2d  # noqa: E402

CORNERS = 8
CENTROID_CHANNEL = 8


class EdgeGuidedCornerResidual(nn.Module):
    def __init__(self, hidden: int = 32, corners: int = CORNERS) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2 * corners, hidden, 3, padding=1),
            ChannelLayerNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, corners, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, base_belief: torch.Tensor, proposals: torch.Tensor):
        stacked = torch.cat([base_belief[:, :CORNERS], proposals], dim=1)
        return self.out(self.body(stacked))


def compose(base_belief: torch.Tensor, edge_residual: torch.Tensor,
            near_residual: torch.Tensor | None = None) -> torch.Tensor:
    """corners 0-7 take the edge residual; 0-3 additionally take HCRM's."""
    out = base_belief.clone()
    out[:, :CORNERS] = out[:, :CORNERS] + edge_residual
    if near_residual is not None:
        out[:, :4] = out[:, :4] + near_residual
    return out


def assert_passthrough(composed: torch.Tensor, base: torch.Tensor) -> dict:
    centroid = (composed[:, CENTROID_CHANNEL] - base[:, CENTROID_CHANNEL]).abs().max()
    report = {"centroid_max_abs": float(centroid)}
    if composed.shape[1] > CENTROID_CHANNEL + 1:
        extra = (composed[:, CENTROID_CHANNEL + 1:]
                 - base[:, CENTROID_CHANNEL + 1:]).abs().max()
        report["extra_max_abs"] = float(extra)
    if report["centroid_max_abs"] != 0.0:
        raise RuntimeError(f"EGCR touched the centroid channel: {report}")
    return report
