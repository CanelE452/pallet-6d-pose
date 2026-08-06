"""Spatial Heatmap Context Recovery Module and its pointwise control.

The canonical corner audit found near corners fail mostly by *missing response*
rather than wrong localization -- recall 71.6% overall, 64.9% and 67.0% for IDs 1
and 2, but a median error of 5.7px once detected.  A frozen-feature probe then
showed the near hard-pair signal survives in F50 only when its spatial
arrangement is kept: pooled 0.8718 against spatial 0.9425 cross-set AUC.  That
is separability in a probe, not belief generation, so this module exists to test
whether the arrangement can be converted into an actual near-corner response.

The residual is added to the final belief tensor the decoder reads, on the four
near channels only.  Far corners, the centroid and every affinity map are copied
through untouched, which is asserted rather than assumed.  Both heads end in a
zero-initialised convolution, so an untrained module reproduces A1 exactly and
any measured change is attributable to training rather than to insertion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NEAR_CHANNELS = (0, 1, 2, 3)     # matches paper_s2_mechanism_diagnostic.NEAR_KP
FAR_CHANNELS = (4, 5, 6, 7)
CENTROID_CHANNEL = 8
N_BELIEF = 9
F50_CHANNELS = 128


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over the channel axis at each location, independently.

    Group normalisation works over (C, H, W), so a single cell changes the
    statistic every other cell is divided by -- which silently gave the
    pointwise control access to a global spatial summary.  This normalises each
    location using only its own channel vector, so H1 really is 1x1 and the
    only difference between the arms is the 5x5 support.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def _zero_init(convolution: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(convolution.weight)
    if convolution.bias is not None:
        nn.init.zeros_(convolution.bias)
    return convolution


class PointwiseHCRM(nn.Module):
    """H1 control: the same F50 channels, no spatial neighbourhood at all.

    Every convolution is 1x1, so a cell's residual is a function of that cell's
    feature vector alone.  If this matches the spatial arm, the spatial claim
    has no support and the simpler module wins.
    """

    def __init__(self, in_channels: int = F50_CHANNELS, hidden: int = 64,
                 out_channels: int = len(NEAR_CHANNELS)) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            ChannelLayerNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            ChannelLayerNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = _zero_init(nn.Conv2d(hidden, out_channels, 1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(feature))


class SpatialHCRM(nn.Module):
    """H2: depthwise 3x3 stages keep the arrangement the probe needed.

    Depthwise separable rather than dense 3x3 so the parameter count stays
    within 1.5x of the pointwise control -- otherwise a win would be capacity,
    not spatial context.
    """

    def __init__(self, in_channels: int = F50_CHANNELS, hidden: int = 64,
                 out_channels: int = len(NEAR_CHANNELS)) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, hidden, 1),
            ChannelLayerNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            ChannelLayerNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = _zero_init(nn.Conv2d(hidden, out_channels, 1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(feature))


ARMS = {"H1": PointwiseHCRM, "H2": SpatialHCRM, "H3": SpatialHCRM}


def build(arm: str, seed: int = 1, in_channels: int = F50_CHANNELS) -> nn.Module:
    torch.manual_seed(seed)
    return ARMS[arm](in_channels=in_channels)


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def compose(base_belief: torch.Tensor, residual: torch.Tensor,
            permutation: tuple[int, ...] | None = None) -> torch.Tensor:
    """Add the residual to the near channels and copy everything else.

    ``permutation`` is the H2-SHUFFLE control: it reorders the residual
    channels only, leaving the base belief and the decoder convention alone, so
    a gain that survives shuffling was never about near-channel semantics.
    """
    if base_belief.shape[1] < N_BELIEF:
        raise RuntimeError(f"belief has {base_belief.shape[1]} channels")
    if residual.shape[1] != len(NEAR_CHANNELS):
        raise RuntimeError(f"residual has {residual.shape[1]} channels")
    if permutation is not None:
        residual = residual[:, list(permutation)]
    out = base_belief.clone()
    out[:, list(NEAR_CHANNELS)] = out[:, list(NEAR_CHANNELS)] + residual
    return out


def assert_untouched(composed: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    """Far, centroid and any extra channel must be bit-identical to the base."""
    report = {}
    far = (composed[:, list(FAR_CHANNELS)] - base[:, list(FAR_CHANNELS)]).abs().max()
    centroid = (composed[:, CENTROID_CHANNEL] - base[:, CENTROID_CHANNEL]).abs().max()
    report["far_max_abs"] = float(far)
    report["centroid_max_abs"] = float(centroid)
    if composed.shape[1] > N_BELIEF:
        extra = (composed[:, N_BELIEF:] - base[:, N_BELIEF:]).abs().max()
        report["extra_max_abs"] = float(extra)
    if report["far_max_abs"] != 0.0 or report["centroid_max_abs"] != 0.0:
        raise RuntimeError(f"HCRM modified a forbidden channel: {report}")
    return report
