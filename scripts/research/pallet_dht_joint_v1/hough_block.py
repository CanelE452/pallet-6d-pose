"""Global HT / normalized-transpose feedback before the original YOLO pose head.

Image coordinates are feature cell centres: (u+.5-W/2, v+.5-H/2).
The sparse operator is fixed geometry; all feature projections are trainable.
Backprojection is a normalized transpose, not a mathematical inverse of HT.
No point, detection, ground-truth or object-dimension inputs are used here.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.research.deep_hough_side_v1.dht import SparseDHT, hough_pad2d


class HoughConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm = nn.GroupNorm(4, channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(hough_pad2d(x, 1))))


def smooth_rho(value: Tensor) -> Tensor:
    """Symmetric [1/4,1/2,1/4] rho kernel, with zero padding at the boundary."""
    padded = F.pad(value, (1, 1))
    return .25 * padded[..., :-2] + .5 * padded[..., 1:-1] + .25 * padded[..., 2:]


def normalized_hough(features: Tensor, geometry: SparseDHT) -> Tensor:
    """A'=K_rho A, normalized by A'1; fills between-centre rho sampling gaps."""
    sums = geometry(features)
    if geometry.normalize:
        sums = sums * geometry.mass.to(sums.dtype)
    return smooth_rho(sums) / smooth_rho(geometry.mass.to(sums.dtype)).clamp_min(1e-7)


def normalized_backprojection(hough: Tensor, geometry: SparseDHT) -> Tensor:
    """A'^T H / A'^T 1 for the SAME A'=K_rho A used in normalized_hough."""
    b, c, t, r = hough.shape
    if (t, r) != (geometry.theta_bins, geometry.rho_bins):
        raise ValueError("Hough map and voting lattice disagree")
    with torch.autocast(device_type=hough.device.type, enabled=False):
        values = hough if hough.dtype == torch.float64 else hough.float()
        matrix = geometry.vote_matrix.to(dtype=values.dtype)
        # K is symmetric, so A'^T = A^T K. Subcell semantic peaks now reach
        # their adjacent image cells even at exactly vertical/horizontal lines.
        flat = smooth_rho(values).reshape(b * c, t * r).t()
        spatial = torch.sparse.mm(matrix.t(), flat).t()
        if not hasattr(geometry, "_backprojection_mass"):
            ones = torch.ones(1, t, r, device=hough.device, dtype=torch.float32)
            geometry._backprojection_mass = torch.sparse.mm(
                geometry.vote_matrix.t(), smooth_rho(ones).reshape(t * r, 1)
            ).t().clamp_min(1e-7)
        spatial = spatial / geometry._backprojection_mass.to(values.dtype)
        return spatial.reshape(b, c, geometry.height, geometry.width)


class HoughFeatureFusion(nn.Module):
    """P4 global line features feed back into P3/P4/P5 before point prediction.

    ``line_logits`` and ``lattice`` are transient outputs consumed by the joint
    criterion. Geometry is cached by input shape/device in FP32 and excluded
    from checkpoint/deepcopy state. No new trainable parameters are made during
    forward. Residual output projections start at zero for exact R0 parity.
    """

    def __init__(self, arm: str = "hough_joint", channels=(64, 128, 256), latent: int = 16):
        super().__init__()
        if arm not in {"hough_joint", "hough_features"}:
            raise ValueError(arm)
        self.arm = arm
        self.theta_bins, self.rho_step, self.rho_max = 90, 0.5, 28.0
        self.rho_bins = 113
        self.reduce = nn.Sequential(nn.Conv2d(channels[1], latent, 1, bias=False),
                                    nn.GroupNorm(4, latent), nn.SiLU())
        self.hough_layers = nn.Sequential(HoughConv(latent), HoughConv(latent))
        self.line_head = nn.Conv2d(latent, 12, 1)
        nn.init.constant_(self.line_head.bias, -4.0)
        self.spatial_mix = nn.Sequential(nn.Conv2d(latent + 12, latent, 1, bias=False),
                                         nn.GroupNorm(4, latent), nn.SiLU())
        self.outputs = nn.ModuleList(nn.Conv2d(latent, c, 1) for c in channels)
        for output in self.outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self._geometry_cache = {}
        self.line_logits = None
        self.lattice = None

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.update(_geometry_cache={}, line_logits=None, lattice=None)
        return state

    def geometry(self, height: int, width: int, device: torch.device) -> SparseDHT:
        key = (int(height), int(width), str(device))
        if key not in self._geometry_cache:
            if math.hypot((height - 1) / 2, (width - 1) / 2) > self.rho_max:
                raise ValueError("P4 exceeds the frozen imgsz640 Hough lattice")
            self._geometry_cache[key] = SparseDHT(height, width, self.theta_bins,
                                                  self.rho_step, False, self.rho_max).to(device)
        return self._geometry_cache[key]

    def clear_transient(self):
        self.line_logits = None
        self.lattice = None

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != 3:
            raise ValueError("expected P3, P4 and P5")
        p4 = features[1]
        h, w = p4.shape[-2:]
        geometry = self.geometry(h, w, p4.device)
        reduced = self.reduce(p4)
        voted = normalized_hough(reduced, geometry).to(reduced.dtype)
        latent = self.hough_layers(voted)
        self.line_logits = self.line_head(latent)
        # A geometrically valid line may fall between sampled feature centres
        # and have zero sparse voting mass (notably axis-aligned half-cell
        # gaps). Hough convolutions can infer that bin from its neighbours.
        # Do not erase its supervision merely because no centre voted there.
        extent = (geometry.theta_radians.cos().abs() * w / 2
                  + geometry.theta_radians.sin().abs() * h / 2)
        footprint_valid = geometry.rho_values[None].abs() <= extent[:, None] + 1e-6
        self.lattice = {
            "height": h, "width": w, "theta_bins": self.theta_bins,
            "rho_bins": self.rho_bins, "rho_step": self.rho_step,
            "rho_max": self.rho_max, "theta_values": geometry.theta_radians,
            "rho_values": geometry.rho_values, "valid": footprint_valid,
            "vote_valid": geometry.valid,
        }
        # Both semantic line evidence and latent line context can train from
        # the original point objective once residual output weights open.
        line_features = torch.cat((latent, self.line_logits.sigmoid()), dim=1)
        spatial = normalized_backprojection(line_features, geometry).to(p4.dtype)
        spatial = self.spatial_mix(spatial)
        return [x + projection(F.interpolate(spatial, size=x.shape[-2:],
                                             mode="bilinear", align_corners=False)).to(x.dtype)
                for x, projection in zip(features, self.outputs)]
