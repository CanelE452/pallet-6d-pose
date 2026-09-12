"""Trainable line heads around an immutable stock-point anchor."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .constants import (GRID_SIZE, LINE_SIGMA_DIAGONAL_FRACTION,
                        MAX_SHIFT_DIAGONAL_FRACTION, N_MODES, RHO_BINS,
                        THETA_BINS)
from .geometry import local_wls_fusion
from .hough import DHTLineHead, DirectLineHead, Lattice, decode_modes


class LocalLineFusion(nn.Module):
    def __init__(self, *, arm: str, in_channels: int = 192, latent: int = 32,
                 grid_size: int = GRID_SIZE, theta_bins: int = THETA_BINS,
                 rho_bins: int = RHO_BINS, modes: int = N_MODES,
                 point_sigma_fraction: float = .005,
                 line_sigma_fraction: float = LINE_SIGMA_DIAGONAL_FRACTION,
                 max_shift_fraction: float = MAX_SHIFT_DIAGONAL_FRACTION):
        super().__init__()
        if arm not in {"direct", "hough"}:
            raise ValueError("arm must be direct or hough")
        if point_sigma_fraction != .005 or line_sigma_fraction != .005 or max_shift_fraction != .01:
            raise ValueError("v1 sigma/max-shift constants are frozen")
        self.arm = arm
        self.in_channels = in_channels
        self.modes = modes
        self.line_sigma_fraction = line_sigma_fraction
        self.max_shift_fraction = max_shift_fraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent, 1), nn.GroupNorm(4, latent), nn.SiLU(),
            nn.Conv2d(latent, latent, 3, padding=1), nn.GroupNorm(4, latent), nn.SiLU(),
        )
        self.lattice = Lattice(grid_size, grid_size, theta_bins, rho_bins)
        head = DirectLineHead if arm == "direct" else DHTLineHead
        self.line_head = head(latent, self.lattice)
        self.null_head = nn.Linear(latent, 12)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        features = observation["features"]
        content = observation["content"].to(features.dtype)
        if features.ndim != 4 or features.shape[1:] != (self.in_channels, 24, 24):
            raise ValueError("expected frozen ROI features [B,192,24,24]")
        latent = self.stem(features)
        masked = latent * content
        pooled = masked.sum((-2, -1)) / content.sum((-2, -1)).clamp_min(1)
        null_logit = self.null_head(pooled)
        line_logits, line_valid = self.line_head(latent, content)
        decoded = decode_modes(line_logits, line_valid, null_logit, self.lattice,
                               observation["box"], self.modes)
        image_hw = observation["image_hw"]
        diagonal = torch.linalg.vector_norm(image_hw.flip(-1), dim=-1)
        line_sigma = self.line_sigma_fraction * diagonal
        corrected, correction = local_wls_fusion(
            observation["base_points"], observation["point_valid"],
            observation["point_sigma"], decoded["raw_lines"],
            decoded["absolute_weight"], line_sigma, image_hw,
            max_shift_fraction=self.max_shift_fraction,
        )
        return {
            "points": corrected,
            "correction": correction,
            "line_logits": line_logits,
            "line_valid": line_valid,
            "null_logit": null_logit,
            **decoded,
        }

    def config(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "in_channels": self.in_channels,
            "latent": self.stem[0].out_channels,
            "grid_size": self.lattice.height,
            "theta_bins": self.lattice.theta_bins,
            "rho_bins": self.lattice.rho_bins,
            "modes": self.modes,
            "point_sigma_fraction": .005,
            "line_sigma_fraction": self.line_sigma_fraction,
            "max_shift_fraction": self.max_shift_fraction,
        }


def build_model(config: dict[str, Any]) -> LocalLineFusion:
    allowed = {"arm", "in_channels", "latent", "grid_size", "theta_bins", "rho_bins",
               "modes", "point_sigma_fraction", "line_sigma_fraction", "max_shift_fraction"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")
    return LocalLineFusion(**config)
