"""Feature-conditioned bounded belief residual (PAPER_S2 micro-arch screen, B1).

Motivation (mechanism diagnosis, strict mechanism-val N=87):
    far/depth keypoints are *confidently wrong* — the belief peak grows from
    0.383 (stage 1) to 0.855 (stage 6) while the localisation error only moves
    31.9 -> 21.4 -> 22.1 px.  The refinement stages therefore raise confidence
    without correcting the position that was already wrong at stage 1.

Hypothesis under test:
    a small head that re-reads the *shared image feature* (which the belief
    stages only see indirectly, through their own previous output) can apply a
    bounded correction to the final belief without disturbing the frozen base.

Design constraints that make the screen interpretable:
    * zero-initialised output convolution -> the very first forward is exactly
      the frozen base (``max|H_final - H_base| == 0``), so any measured change
      is attributable to training rather than to re-parameterisation;
    * ``0.25 * tanh`` bound -> the head cannot overwrite the base belief, so the
      frozen base acts as the anchor and no separate anchor loss is needed;
    * detached inputs + frozen base -> gradients reach the head only.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoundedBeliefResidual(nn.Module):
    """``H_final = H_base + amplitude * tanh(head([F, H_base]))``.

    Parameters
    ----------
    feature_channels:
        Channels of the shared VGG feature map (``out1``); 128 for DopeNetwork.
    belief_channels:
        Number of belief channels (9 for the pallet cuboid + centroid).
    hidden_channels:
        Width of the single hidden 3x3 convolution.
    amplitude:
        Maximum absolute correction applied to any belief cell.
    """

    def __init__(
        self,
        feature_channels: int = 128,
        belief_channels: int = 9,
        hidden_channels: int = 16,
        amplitude: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or belief_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        if not (amplitude > 0.0):
            raise ValueError("amplitude must be positive")
        self.feature_channels = int(feature_channels)
        self.belief_channels = int(belief_channels)
        self.amplitude = float(amplitude)

        self.conv1 = nn.Conv2d(
            self.feature_channels + self.belief_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, self.belief_channels, kernel_size=1)
        self.reset_output_to_zero()

    def reset_output_to_zero(self) -> None:
        """Exact identity at initialisation (not merely small)."""
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def residual(
        self, feature: torch.Tensor, belief_base: torch.Tensor
    ) -> torch.Tensor:
        """Bounded correction ``Delta_H``; inputs are detached here, not outside."""
        if feature.ndim != 4 or belief_base.ndim != 4:
            raise ValueError("feature and belief_base must be (B,C,H,W)")
        if feature.shape[0] != belief_base.shape[0]:
            raise ValueError("batch size mismatch between feature and belief")
        if feature.shape[-2:] != belief_base.shape[-2:]:
            raise ValueError(
                "feature and belief must share the spatial grid, got "
                f"{tuple(feature.shape[-2:])} and {tuple(belief_base.shape[-2:])}"
            )
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"expected {self.feature_channels} feature channels, "
                f"got {feature.shape[1]}"
            )
        if belief_base.shape[1] != self.belief_channels:
            raise ValueError(
                f"expected {self.belief_channels} belief channels, "
                f"got {belief_base.shape[1]}"
            )
        stacked = torch.cat([feature.detach(), belief_base.detach()], dim=1)
        return self.amplitude * torch.tanh(self.conv2(self.act(self.conv1(stacked))))

    def forward(
        self, feature: torch.Tensor, belief_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(H_final, Delta_H)``.

        ``H_final`` is deliberately NOT clamped: the evaluation decoder consumes
        the raw belief convention, and clamping here would silently change the
        operating point relative to the frozen baseline.
        """
        delta = self.residual(feature, belief_base)
        return belief_base.detach() + delta, delta


@torch.no_grad()
def residual_diagnostics(
    delta: torch.Tensor, amplitude: float, saturation_ratio: float = 0.99
) -> dict[str, float]:
    """Collapse detectors for the B1 screen.

    ``saturation`` counts cells that sit at the tanh bound; a head that spends
    most of its output there is no longer applying a bounded correction and the
    run must stop.
    """
    values = delta.detach().float()
    magnitude = values.abs()
    bound = float(amplitude) * float(saturation_ratio)
    per_channel = magnitude.mean(dim=(0, 2, 3)) if values.ndim == 4 else magnitude
    return {
        "residual_abs_mean": float(magnitude.mean()),
        "residual_abs_max": float(magnitude.max()),
        "residual_saturation_fraction": float((magnitude >= bound).float().mean()),
        "residual_channel_abs_mean": [float(v) for v in per_channel.reshape(-1)],
    }
