"""Instance-aware edge heads on top of a frozen encoder.

The trunk is the one the five-class PPD head already uses -- a 1x1 stem over
``[feature, gate, gate*feature]`` followed by two residual blocks and a 1x1
output -- with the output width changed from five semantic classes to twelve
physical edges.  Keeping the trunk identical is what makes the F50 arm, the
multi-scale arm and the five-class control differ in one thing at a time.

No vector, offset, endpoint or voting output exists here.  A line is never
converted into a keypoint inside the head: that happens only in the fixed O12
decoder, from the predicted fields alone.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_COMMON = pathlib.Path(__file__).resolve().parent
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

import polarity_aware_line_head as PLH  # noqa: E402


class EdgeFieldHead(nn.Module):
    """PPD trunk with a configurable number of output maps."""

    def __init__(self, feature_channels: int, out_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2 * feature_channels + 1, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(PLH._ResidualBlock(hidden), PLH._ResidualBlock(hidden))
        self.out = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, feature: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if gate is None:
            gate = torch.ones_like(feature[:, :1])
        stacked = torch.cat([feature, gate, gate * feature], dim=1)
        return self.out(self.blocks(self.stem(stacked)))


class MultiScaleFusion(nn.Module):
    """Align the 100x100 tap to the 50x50 tap and concatenate.

    The high-resolution branch is reduced to 64 channels before a 2x2 average
    pool, so the fused tensor is 192 channels at the shared 50x50 resolution.
    """

    def __init__(self, high_channels: int = 256, low_channels: int = 128,
                 high_reduced: int = 64) -> None:
        super().__init__()
        self.high = nn.Sequential(nn.Conv2d(high_channels, high_reduced, 1),
                                  nn.ReLU(inplace=True))
        self.low = nn.Sequential(nn.Conv2d(low_channels, low_channels, 1),
                                 nn.ReLU(inplace=True))
        self.out_channels = high_reduced + low_channels

    def forward(self, feature_high: torch.Tensor, feature_low: torch.Tensor) -> torch.Tensor:
        reduced = F.avg_pool2d(self.high(feature_high), 2, 2)
        if reduced.shape[-2:] != feature_low.shape[-2:]:
            raise RuntimeError(
                f"multi-scale misalignment {tuple(reduced.shape[-2:])} vs "
                f"{tuple(feature_low.shape[-2:])}")
        return torch.cat([reduced, self.low(feature_low)], dim=1)


class InstanceEdgeModel(nn.Module):
    """Frozen encoder taps plus one trainable edge head."""

    def __init__(self, arm: str, out_channels: int, high_channels: int = 256,
                 low_channels: int = 128) -> None:
        super().__init__()
        assert arm in ("F50", "MS", "F100"), arm
        self.arm = arm
        if arm == "MS":
            self.fusion = MultiScaleFusion(high_channels, low_channels)
            self.head = EdgeFieldHead(self.fusion.out_channels, out_channels)
        elif arm == "F50":
            self.fusion = None
            self.head = EdgeFieldHead(low_channels, out_channels)
        else:
            self.fusion = None
            self.head = EdgeFieldHead(high_channels, out_channels)

    def forward(self, feature_high: torch.Tensor | None,
                feature_low: torch.Tensor | None) -> torch.Tensor:
        if self.arm == "MS":
            return self.head(self.fusion(feature_high, feature_low))
        return self.head(feature_low if self.arm == "F50" else feature_high)


def masked_field_loss(logits: torch.Tensor, targets: torch.Tensor,
                      positive_weight: torch.Tensor,
                      channel_mask: torch.Tensor) -> torch.Tensor:
    """Class-balanced weighted BCE, skipping channels with no in-frame segment.

    An edge that is entirely outside the frame carries no target to learn, so
    its channel is masked out for that sample.  An occluded but in-frame edge
    keeps weight 1.0: the decoder needs it, and dropping it would teach the
    head that hidden structure does not exist.
    """
    weight = positive_weight.view(1, -1, 1, 1).to(logits.device)
    element = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="none")
    mask = channel_mask.view(channel_mask.shape[0], channel_mask.shape[1], 1, 1)
    total = (element * mask).sum()
    count = mask.expand_as(element).sum().clamp_min(1.0)
    return total / count


def polarity_contrast_loss(logits: torch.Tensor, targets: torch.Tensor,
                           pairs: list[tuple[int, int]],
                           margin: float = PLH.CONTRAST_MARGIN) -> torch.Tensor:
    """PPD's top/base contrast, applied to instance channels instead of classes.

    The pairs are derived from the topology (an edge and the same edge
    translated along the height axis), not written by hand.  Pixels where both
    partners are high are excluded, exactly as in the five-class version.
    """
    losses = []
    for top_index, base_index in pairs:
        top_logit = logits[:, top_index]
        base_logit = logits[:, base_index]
        top_target = targets[:, top_index]
        base_target = targets[:, base_index]
        top_positive = ((top_target >= PLH.POSITIVE_THRESHOLD)
                        & (base_target <= PLH.OPPOSITE_THRESHOLD))
        base_positive = ((base_target >= PLH.POSITIVE_THRESHOLD)
                         & (top_target <= PLH.OPPOSITE_THRESHOLD))
        if bool(top_positive.any()):
            losses.append(F.softplus(margin - (top_logit - base_logit))[top_positive].mean())
        if bool(base_positive.any()):
            losses.append(F.softplus(margin - (base_logit - top_logit))[base_positive].mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
