"""Stage-1 heads: palletness, keypoint visibility, truncation, and assembly.

None of these touch the belief or affinity representation.  They sit beside it
so that the pose path is unchanged at step 0 and the only new information at
inference time is a per-keypoint visibility state, which Visibility-Aware Pose
Assembly uses to drop correspondences the network says are off-screen.

The visibility head reads a detached copy of the final belief stack.  Without the
detach, a classifier that wants an off-screen channel to look different would be
able to reach back and reshape the belief map itself, which is not what is being
asked of it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_KP = 9
VIS_CLASSES = 3
VIS_VISIBLE, VIS_OCCLUDED, VIS_OFF_SCREEN = 0, 1, 2
VAPA_OFF_SCREEN_THRESHOLD = 0.5        # fixed before any result


class PalletnessResponseHead(nn.Module):
    """PRH.  Object-level pallet extent at belief resolution."""

    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(feature))


class KeypointVisibilityHead(nn.Module):
    """KVH.  Three states per keypoint: visible, occluded, off-screen."""

    def __init__(self, in_channels: int = 128, n_kp: int = N_KP) -> None:
        super().__init__()
        self.n_kp = n_kp
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels + n_kp, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(64, n_kp * VIS_CLASSES, 1)

    def forward(self, feature: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([feature, belief.detach()], dim=1)
        hidden = self.trunk(merged)
        logits = self.out(hidden)
        pooled = (logits.mean(dim=(2, 3)) + logits.amax(dim=(2, 3))) * 0.5
        return pooled.reshape(-1, self.n_kp, VIS_CLASSES)


class TruncationHead(nn.Module):
    """One logit: does any corner leave the frame."""

    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, 32), nn.ReLU(inplace=True),
                                 nn.Linear(32, 1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature.mean(dim=(2, 3)))


def off_screen_probability(visibility_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(visibility_logits, dim=-1)[..., VIS_OFF_SCREEN]


def visibility_aware_assembly(points, off_screen_probability_values,
                              threshold: float = VAPA_OFF_SCREEN_THRESHOLD):
    """VAPA.  Drop the correspondences the network calls off-screen.

    Visible and occluded keypoints are both kept -- an occluded corner still has
    a real image position.  No GT is consulted; the decision comes from the
    predicted distribution only.  Below four survivors the caller's own solver
    will refuse, exactly as it does today.
    """
    probabilities = list(off_screen_probability_values)
    kept, dropped = [], []
    for index, point in enumerate(points):
        if point is None:
            kept.append(None)
            continue
        if index < len(probabilities) and probabilities[index] >= threshold:
            kept.append(None)
            dropped.append(index)
            continue
        kept.append(point)
    return kept, dropped
