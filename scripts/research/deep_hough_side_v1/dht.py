"""Differentiable, line-aligned feature voting for the pallet DHT experiment.

This is a task-adapted implementation of the feature-space Hough aggregation
described by Han et al., *Deep Hough Transform for Semantic Line Detection*,
ECCV 2020, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540239.pdf.
It is not a reproduction of their full multiscale network or CUDA extension.

The lattice uses the angle of the line's NORMAL, measured in image coordinates
(x right, y down): ``(x-cx)*cos(theta) + (y-cy)*sin(theta) = rho``. A vertical
line therefore has theta=0, and a horizontal line has theta=90 degrees. Pixel
centers are 0..W-1 and 0..H-1; the origin of a 50x50 map is (24.5, 24.5).

Each feature pixel votes at every theta, linearly splitting its value between
the two neighboring rho bins. Equivalently each bin aggregates a triangular
strip around an infinite candidate line, clipped to the feature image. This
avoids nearest-bin discontinuities in the fixed discretization. ``normalize``
divides by the same interpolation weights summed over all image pixels, making
the result a weighted line average instead of favoring long central lines.
The published DHT uses sums; normalization is an explicit experimental choice.

Voting geometry is fixed and represented by a sparse matrix, while the input
features and their preceding convolutions receive gradients through sparse
matrix multiplication. No gradients through the lattice geometry are claimed.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseDHT(nn.Module):
    """Map ``[B,C,H,W]`` image features to ``[B,C,theta_bins,rho_bins]``.

    ``rho_step`` is measured in input feature pixels. By default rho extends to
    the smallest multiple of rho_step covering every pixel center, with a
    symmetric lattice. For 50x50, 180 angles and rho_step=0.5 this is 180x141.
    Flattening the last two dimensions gives theta-major, rho-minor order.

    Geometry buffers are nonpersistent because they are deterministic from the
    constructor arguments. Checkpoints must retain the experiment arguments.
    Half/bfloat16 input is accumulated in float32 and returned in float32;
    float32/float64 input retains its dtype. This operator has no parameters.
    """

    def __init__(
        self,
        height: int = 50,
        width: int = 50,
        theta_bins: int = 180,
        rho_step: float = 0.5,
        normalize: bool = True,
        rho_max: float | None = None,
    ) -> None:
        super().__init__()
        if height < 1 or width < 1 or theta_bins < 1:
            raise ValueError("height, width, and theta_bins must be positive")
        if not math.isfinite(rho_step) or rho_step <= 0:
            raise ValueError("rho_step must be finite and positive")
        self.height = int(height)
        self.width = int(width)
        self.theta_bins = int(theta_bins)
        self.rho_step = float(rho_step)
        self.normalize = bool(normalize)
        self.center_x = (self.width - 1) / 2.0
        self.center_y = (self.height - 1) / 2.0
        reach = math.hypot(self.center_x, self.center_y)
        if rho_max is None:
            rho_ticks = math.ceil(reach / self.rho_step)
        else:
            if not math.isfinite(rho_max) or rho_max < reach - 1e-9:
                raise ValueError("rho_max must cover the input pixel-center diagonal")
            rho_ticks = round(rho_max / self.rho_step)
            if not math.isclose(rho_ticks * self.rho_step, rho_max, abs_tol=1e-8):
                raise ValueError("rho_max must be a multiple of rho_step")
        self.rho_max = rho_ticks * self.rho_step
        self.rho_bins = 2 * rho_ticks + 1

        # Construct in float64 so axis-aligned lines do not acquire numerical
        # leakage into adjacent bins from approximate sin(pi) / cos(pi/2).
        theta = torch.arange(self.theta_bins, dtype=torch.float64) * (
            math.pi / self.theta_bins
        )
        cosine, sine = theta.cos(), theta.sin()
        cosine[cosine.abs() < 1e-12] = 0
        sine[sine.abs() < 1e-12] = 0
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64) - self.center_y,
            torch.arange(self.width, dtype=torch.float64) - self.center_x,
            indexing="ij",
        )
        projected = (
            cosine[:, None] * xx.flatten()[None]
            + sine[:, None] * yy.flatten()[None]
        )
        bins = (projected + self.rho_max) / self.rho_step
        # Near-integer roundoff is geometric noise, not a real small vote.
        nearest = bins.round()
        bins = torch.where((bins - nearest).abs() < 1e-10, nearest, bins)
        lo = bins.floor().long()
        alpha = bins - lo
        n_pixels = self.height * self.width
        pixels = torch.arange(n_pixels).expand(self.theta_bins, -1)
        angles = torch.arange(self.theta_bins)[:, None] * self.rho_bins
        row_chunks, col_chunks, weight_chunks = [], [], []
        for rho_index, weight in ((lo, 1 - alpha), (lo + 1, alpha)):
            keep = (weight > 0) & (rho_index >= 0) & (rho_index < self.rho_bins)
            row_chunks.append((angles + rho_index)[keep])
            col_chunks.append(pixels[keep])
            weight_chunks.append(weight[keep])
        rows = torch.cat(row_chunks)
        cols = torch.cat(col_chunks)
        weights = torch.cat(weight_chunks).float()
        matrix = torch.sparse_coo_tensor(
            torch.stack((rows, cols)),
            weights,
            (self.theta_bins * self.rho_bins, n_pixels),
        ).coalesce()
        mass = torch.zeros(self.theta_bins * self.rho_bins).scatter_add_(
            0, matrix.indices()[0], matrix.values()
        ).reshape(self.theta_bins, self.rho_bins)

        self.register_buffer("vote_matrix", matrix, persistent=False)
        self.register_buffer("mass", mass, persistent=False)
        self.register_buffer("valid", mass > 1e-7, persistent=False)
        self.register_buffer("theta_radians", theta.float(), persistent=False)
        self.register_buffer("theta_degrees", (theta * (180 / math.pi)).float(), persistent=False)
        self.register_buffer(
            "rho_values",
            torch.arange(-rho_ticks, rho_ticks + 1).float() * self.rho_step,
            persistent=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"expected [B,C,{self.height},{self.width}], got {tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise TypeError("DHT requires floating-point features")
        if features.device != self.vote_matrix.device:
            raise ValueError("move the DHT module to the same device as its features")
        batch, channels = features.shape[:2]
        # Sparse CUDA kernels and their gradients are checked in FP32; avoiding
        # autocast here also prevents accumulated line sums losing precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features if features.dtype == torch.float64 else features.float()
            matrix = self.vote_matrix.to(dtype=values.dtype)
            flat = values.reshape(batch * channels, self.height * self.width).t()
            votes = torch.sparse.mm(matrix, flat).t().reshape(
                batch, channels, self.theta_bins, self.rho_bins
            )
            if self.normalize:
                votes = votes / self.mass.to(dtype=values.dtype).clamp_min(1e-7)
        return votes

    def extra_repr(self) -> str:
        return (
            f"height={self.height}, width={self.width}, theta_bins={self.theta_bins}, "
            f"rho_bins={self.rho_bins}, rho_step={self.rho_step}, "
            f"normalize={self.normalize}"
        )


def hough_pad2d(features: Tensor, padding: int | tuple[int, int] = 1) -> Tensor:
    """Pad a ``[..., theta, rho]`` map with the correct undirected-line seam.

    Theta wraps by 180 degrees and reverses rho: ``(theta+pi,rho)`` is the same
    line as ``(theta,-rho)``. Rho is zero-padded. A tuple means (theta_pad,
    rho_pad). Use this before a convolution with padding=0; ordinary circular
    theta padding is geometrically incorrect. The rho lattice must be symmetric.
    """
    theta_pad, rho_pad = (padding, padding) if isinstance(padding, int) else padding
    if theta_pad < 0 or rho_pad < 0 or theta_pad > features.shape[-2]:
        raise ValueError("invalid Hough padding")
    if theta_pad:
        features = torch.cat(
            (
                features[..., -theta_pad:, :].flip(-1),
                features,
                features[..., :theta_pad, :].flip(-1),
            ),
            dim=-2,
        )
    return F.pad(features, (rho_pad, rho_pad, 0, 0))
