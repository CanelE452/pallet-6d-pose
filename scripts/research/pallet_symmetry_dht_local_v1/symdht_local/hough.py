"""Dense reference DHT, direct lattice readout, and reliability decoding."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .geometry import lines_grid_to_raw


def hough_pad(value: Tensor, padding: int = 1) -> Tensor:
    """Pad theta with the required rho flip across the theta/pi seam."""
    if padding < 1:
        return value
    value = torch.cat((value[..., -padding:, :].flip(-1), value,
                       value[..., :padding, :].flip(-1)), -2)
    return torch.nn.functional.pad(value, (padding, padding))


class HoughConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, value: Tensor) -> Tensor:
        return torch.nn.functional.silu(self.norm(self.conv(hough_pad(value))))


class Lattice(nn.Module):
    def __init__(self, height: int = 24, width: int = 24,
                 theta_bins: int = 36, rho_bins: int = 65):
        super().__init__()
        if min(height, width) < 2 or theta_bins < 4 or rho_bins < 5 or rho_bins % 2 != 1:
            raise ValueError("invalid lattice; an odd symmetric rho axis is required")
        self.height, self.width = height, width
        self.theta_bins, self.rho_bins = theta_bins, rho_bins
        theta = torch.arange(theta_bins, dtype=torch.float64) * math.pi / theta_bins
        rho = torch.linspace(-math.sqrt(2), math.sqrt(2), rho_bins, dtype=torch.float64)
        yy, xx = torch.meshgrid(
            (torch.arange(height, dtype=torch.float64) + .5) * 2 / height - 1,
            (torch.arange(width, dtype=torch.float64) + .5) * 2 / width - 1,
            indexing="ij",
        )
        xy = torch.stack((xx.flatten(), yy.flatten()), -1)
        normals = torch.stack((theta.cos(), theta.sin()), -1)
        normals[normals.abs() < 1e-12] = 0
        projected = normals @ xy.T
        position = (projected - rho[0]) / (rho[1] - rho[0])
        nearest = position.round()
        position = torch.where((position - nearest).abs() < 1e-10, nearest, position)
        lower = position.floor().long()
        fraction = position - lower
        rows, columns, values = [], [], []
        angle_index = torch.arange(theta_bins)[:, None].expand_as(lower)
        pixel_index = torch.arange(height * width)[None].expand_as(lower)
        for index, weight in ((lower, 1 - fraction), (lower + 1, fraction)):
            keep = (index >= 0) & (index < rho_bins) & (weight > 0)
            rows.append((angle_index * rho_bins + index)[keep])
            columns.append(pixel_index[keep])
            values.append(weight[keep])
        if theta_bins * rho_bins * height * width > 20_000_000:
            raise ValueError("dense reference DHT matrix is too large")
        voting = torch.zeros(theta_bins * rho_bins, height * width, dtype=torch.float64)
        voting.index_put_((torch.cat(rows), torch.cat(columns)), torch.cat(values), accumulate=True)
        normal_grid = normals[:, None].expand(-1, rho_bins, -1)
        rho_grid = rho[None].expand(theta_bins, -1)
        # n.x-rho=0, represented as (nx,ny,c=-rho).
        lines = torch.cat((normal_grid, -rho_grid[..., None]), -1).reshape(-1, 3)
        self.register_buffer("voting", voting.float(), persistent=False)
        self.register_buffer("lines", lines.float(), persistent=False)
        self.register_buffer("xy", xy.float(), persistent=False)

    @property
    def bins(self) -> int:
        return self.theta_bins * self.rho_bins

    def forward(self, features: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, height, width = features.shape
        if (height, width) != (self.height, self.width) or content.shape != (batch, 1, height, width):
            raise ValueError("DHT feature/content shape mismatch")
        mask = content.to(features.dtype).flatten(2)
        matrix = self.voting.to(features.dtype)
        mass = mask @ matrix.T
        value = (features * content).flatten(2) @ matrix.T
        value = value / mass.clamp_min(1e-6)
        return value.reshape(batch, features.shape[1], self.theta_bins, self.rho_bins), \
            mass.reshape(batch, 1, self.theta_bins, self.rho_bins)


class DHTLineHead(nn.Module):
    def __init__(self, channels: int, lattice: Lattice, roles: int = 12):
        super().__init__()
        self.lattice = lattice
        self.readout = nn.Sequential(
            HoughConv(channels), HoughConv(channels), nn.Conv2d(channels, roles, 1),
        )

    def forward(self, features: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        votes, mass = self.lattice(features, content)
        logits = self.readout(votes).flatten(2)
        valid = (mass.flatten(2) > 1e-6).expand(-1, logits.shape[1], -1)
        return logits.masked_fill(~valid, -1e4), valid


class DirectLineHead(nn.Module):
    """Role queries score the same analytic lattice without line-aligned voting."""
    def __init__(self, channels: int, lattice: Lattice, roles: int = 12):
        super().__init__()
        self.lattice = lattice
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.queries = nn.Parameter(torch.randn(roles, channels) * .02)
        self.position = nn.Linear(2, channels)
        self.line_embedding = nn.Sequential(nn.Linear(6, channels), nn.SiLU(),
                                            nn.Linear(channels, channels))
        self.output = nn.Linear(channels, channels)

    def forward(self, features: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        keys = self.key(features).flatten(2).transpose(1, 2)
        keys = keys + self.position(self.lattice.xy)[None]
        values = self.value(features).flatten(2).transpose(1, 2)
        attention = torch.einsum("ec,bpc->bep", self.queries, keys) / math.sqrt(keys.shape[-1])
        attention = attention.masked_fill(~content.flatten(2).bool(), -1e4)
        descriptor = torch.softmax(attention, -1) @ values
        nx, ny, rho = self.lattice.lines.unbind(-1)
        analytic = torch.stack((nx * nx - ny * ny, 2 * nx * ny, rho * nx,
                                rho * ny, rho.square(), torch.ones_like(rho)), -1)
        logits = self.output(descriptor) @ self.line_embedding(analytic).T / math.sqrt(keys.shape[-1])
        mass = content.to(features.dtype).flatten(2) @ self.lattice.voting.to(features.dtype).T
        valid = (mass > 1e-6).expand(-1, logits.shape[1], -1)
        return logits.masked_fill(~valid, -1e4), valid


def reliability(logits: Tensor, valid: Tensor, null_logit: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute conditional distribution and uncalibrated availability proxies."""
    if logits.shape != valid.shape or null_logit.shape != logits.shape[:2]:
        raise ValueError("reliability shape mismatch")
    masked = logits.masked_fill(~valid, -1e4)
    probability = torch.softmax(masked, -1) * valid.to(logits.dtype)
    valid_count = valid.sum(-1).clamp_min(1).to(logits.dtype)
    log_evidence = torch.logsumexp(masked, -1) - valid_count.log()
    p_nonnull = torch.sigmoid(log_evidence - null_logit)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    denom = valid_count.log().clamp_min(1e-12)
    concentration = (1 - entropy / denom).clamp(0, 1)
    spread = masked.max(-1).values - masked.masked_fill(~valid, float("inf")).min(-1).values
    concentration = torch.where((valid_count <= 1) | (spread <= 1e-7),
                                torch.zeros_like(concentration), concentration)
    return probability, p_nonnull, concentration, p_nonnull * concentration


def decode_modes(logits: Tensor, valid: Tensor, null_logit: Tensor, lattice: Lattice,
                 box: Tensor, modes: int = 3) -> dict[str, Tensor]:
    """Hard mode locations with differentiable local averaging and absolute mass.

    Retained mode masses are sums from the original conditional distribution;
    they are deliberately not re-normalized across retained modes.
    """
    probability, p_nonnull, concentration, usable = reliability(logits, valid, null_logit)
    base = lattice.lines.to(logits.dtype)
    work = logits.masked_fill(~valid, -1e4)
    decoded, masses, available = [], [], []
    angle_bw = math.pi / lattice.theta_bins * 1.5
    rho_bw = 2 * math.sqrt(2) / (lattice.rho_bins - 1) * 1.5
    for _ in range(modes):
        anchor_index = work.argmax(-1)
        anchor = base[anchor_index]
        dot = torch.einsum("beq,mq->bem", anchor[..., :2], base[:, :2])
        sign = torch.where(dot >= 0, 1., -1.)
        angle_error = 1 - dot.abs().clamp_max(1)
        offset_error = base[None, None, :, 2] * sign - anchor[..., 2, None]
        distance = angle_error / (1 - math.cos(angle_bw)) + offset_error.square() / rho_bw ** 2
        window = (distance <= 1) & valid
        okay = (work.max(-1).values > -9000) & window.any(-1)
        local = torch.softmax((logits - .5 * distance).masked_fill(~window, -1e4), -1)
        local = local * window
        local = local / local.sum(-1, keepdim=True).clamp_min(1e-12)
        line = torch.einsum("bem,bem,mq->beq", local, sign, base)
        line = line / torch.linalg.vector_norm(line[..., :2], dim=-1, keepdim=True).clamp_min(1e-8)
        decoded.append(line)
        masses.append((probability * window).sum(-1))
        available.append(okay)
        work = work.masked_fill(distance <= 4, -1e4)
    grid_lines = torch.stack(decoded, -2)
    mode_mass = torch.stack(masses, -1)
    mode_valid = torch.stack(available, -1)
    raw_lines, normal_ok = lines_grid_to_raw(grid_lines, box)
    mode_valid = mode_valid & normal_ok
    mode_mass = torch.where(mode_valid, mode_mass, torch.zeros_like(mode_mass))
    absolute_weight = usable[..., None] * mode_mass
    return {
        "raw_lines": raw_lines,
        "mode_mass": mode_mass,
        "mode_valid": mode_valid,
        "absolute_weight": absolute_weight,
        "p_nonnull": p_nonnull,
        "concentration": concentration,
        "reliability": usable,
        "conditional_probability": probability,
    }
