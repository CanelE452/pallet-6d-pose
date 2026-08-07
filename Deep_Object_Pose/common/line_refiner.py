"""Local line refiner: read a strip of feature around a coarse line, correct it.

The budget is 1.0 degree and 0.5 belief cell.  Before writing a predictor that
has to find lines, this asks a narrower question -- given a coarse line already
near the right place, can any available feature be read precisely enough to
close that gap?  Same refiner on every feature source, so the comparison is the
feature and nothing else.

There is no half-length output.  CIGM consumes infinite lines, and a segment
extent the pipeline never uses would only add a loss term to trade against.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LONGITUDINAL = 64          # samples along the line
TRANSVERSE = 7             # strip offsets across it, in feature cells


def sample_strip(feature: torch.Tensor, normal: torch.Tensor, rho: torch.Tensor,
                 grid: int, longitudinal: int = LONGITUDINAL,
                 transverse: int = TRANSVERSE) -> torch.Tensor:
    """(B,R,C,transverse,longitudinal) feature strip around each coarse line.

    Sampling is differentiable, so the refiner is trained through the same
    operation it uses at inference.
    """
    batch, roles = normal.shape[0], normal.shape[1]
    device = feature.device
    direction = torch.stack([-normal[..., 1], normal[..., 0]], -1)
    centre = normal * rho[..., None]
    t = torch.linspace(-grid / 2, grid / 2, longitudinal, device=device)
    s = torch.linspace(-(transverse // 2), transverse // 2, transverse, device=device)
    points = (centre[:, :, None, None, :]
              + direction[:, :, None, None, :] * t[None, None, None, :, None]
              + normal[:, :, None, None, :] * s[None, None, :, None, None])
    # grid_sample wants [-1, 1] with x first
    norm = (points / (grid - 1)) * 2 - 1
    flat = norm.reshape(batch, roles * transverse, longitudinal, 2)
    sampled = F.grid_sample(feature, flat, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    return sampled.reshape(batch, -1, roles, transverse, longitudinal).permute(0, 2, 1, 3, 4)


class LocalLineRefiner(nn.Module):
    """Strip -> (delta angle, delta rho, log sigma angle, log sigma rho)."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 transverse: int = TRANSVERSE) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, (transverse, 5), padding=(0, 2)),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, (1, 5), padding=(0, 2)),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden * 2, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, strip: torch.Tensor) -> dict:
        batch, roles = strip.shape[0], strip.shape[1]
        x = strip.reshape(batch * roles, *strip.shape[2:])
        x = self.body(x).squeeze(2)
        pooled = torch.cat([x.mean(-1), x.max(-1).values], dim=-1)
        out = self.head(pooled).reshape(batch, roles, 4)
        return {"delta_theta": out[..., 0], "delta_rho": out[..., 1],
                "log_sigma_theta": out[..., 2].clamp(-6, 4),
                "log_sigma_rho": out[..., 3].clamp(-6, 4)}


def apply_delta(normal: torch.Tensor, rho: torch.Tensor,
                delta_theta: torch.Tensor, delta_rho: torch.Tensor):
    theta = torch.atan2(normal[..., 1], normal[..., 0]) + delta_theta
    return torch.stack([theta.cos(), theta.sin()], -1), rho + delta_rho


def line_losses(normal, rho, gt_normal, gt_rho):
    """Sign-aligned, so a flipped-but-identical line is not penalised."""
    sign = torch.sign((normal * gt_normal).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    aligned_normal = normal * sign
    aligned_rho = rho * sign.squeeze(-1)
    cos = (aligned_normal * gt_normal).sum(-1).abs().clamp(max=1.0)
    angle = torch.rad2deg(torch.arccos(cos))
    offset = (aligned_rho - gt_rho).abs()
    return {"angle_deg": angle, "offset_cell": offset,
            "L_angle": (1.0 - cos).mean(),
            "L_offset": F.smooth_l1_loss(aligned_rho, gt_rho)}
