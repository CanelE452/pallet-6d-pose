"""v2 model: shared initialization, direct utility, soft-line MAP decoding."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import (
    DHTLineHead, DirectLineHead, Lattice,
)

from .geometry import decode_single_mode, single_mode_wls


class LocalLineFusionV2(nn.Module):
    def __init__(self, *, arm: str, common_seed: int, in_channels: int = 192,
                 latent: int = 32, grid_size: int = 24, theta_bins: int = 36,
                 rho_bins: int = 65, point_sigma_fraction: float = .005,
                 line_sigma_fraction: float = .005, max_shift_fraction: float = .01):
        super().__init__()
        if arm not in {"direct", "hough"}:
            raise ValueError("arm must be direct or hough")
        if (grid_size, theta_bins, rho_bins) != (24, 36, 65):
            raise ValueError("v2 lattice is frozen at 24 ROI and 36x65 DHT")
        if (point_sigma_fraction, line_sigma_fraction, max_shift_fraction) != (.005, .005, .01):
            raise ValueError("v2 sigma/max-shift constants are frozen")
        self.arm, self.common_seed = arm, int(common_seed)
        self.in_channels = in_channels
        self.line_sigma_fraction = line_sigma_fraction
        self.max_shift_fraction = max_shift_fraction
        # Common modules consume an isolated, arm-independent RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100_000 + self.common_seed)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, latent, 1), nn.GroupNorm(4, latent), nn.SiLU(),
                nn.Conv2d(latent, latent, 3, padding=1), nn.GroupNorm(4, latent), nn.SiLU(),
            )
            self.utility_head = nn.Linear(latent, 12)
        self.lattice = Lattice(grid_size, grid_size, theta_bins, rho_bins)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(200_000 + self.common_seed)
            head = DirectLineHead if arm == "direct" else DHTLineHead
            self.line_head = head(latent, self.lattice)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        features = observation["features"]
        content = observation["content"].to(features.dtype)
        latent = self.stem(features)
        pooled = (latent * content).sum((-2, -1)) / content.sum((-2, -1)).clamp_min(1)
        utility_logit = self.utility_head(pooled)
        line_logits, line_valid = self.line_head(latent, content)
        decoded = decode_single_mode(line_logits, line_valid, self.lattice, observation["box"])
        utility = torch.sigmoid(utility_logit) * decoded["available"].to(features.dtype)
        diagonal = torch.linalg.vector_norm(observation["image_hw"].flip(-1), dim=-1)
        corrected, correction = single_mode_wls(
            observation["base_points"], observation["point_valid"], observation["point_sigma"],
            decoded["raw_line"], utility, self.line_sigma_fraction * diagonal,
            observation["image_hw"], max_shift_fraction=self.max_shift_fraction,
        )
        return {
            "points": corrected, "correction": correction,
            "line_logits": line_logits, "line_valid": line_valid,
            "utility_logit": utility_logit, "utility": utility, **decoded,
        }

    def config(self) -> dict[str, Any]:
        return {"arm": self.arm, "common_seed": self.common_seed,
                "in_channels": self.in_channels, "latent": self.stem[0].out_channels,
                "grid_size": self.lattice.height, "theta_bins": self.lattice.theta_bins,
                "rho_bins": self.lattice.rho_bins, "point_sigma_fraction": .005,
                "line_sigma_fraction": self.line_sigma_fraction,
                "max_shift_fraction": self.max_shift_fraction}


def build_model(config: dict[str, Any], *, seed: int | None = None) -> LocalLineFusionV2:
    value = dict(config)
    if seed is not None:
        value["common_seed"] = int(seed)
    return LocalLineFusionV2(**value)
