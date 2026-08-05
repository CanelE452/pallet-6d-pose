"""Far-decoupled refinement adapter for DOPE.

Across nine fixed near/far stage combinations, using stage 6 for the near face
and stage 5 for the far face ranked first on corner error on both the canonical
and the wood evaluation sets, cutting far error by about 16% while leaving the
near face untouched.  It did not reach the pose.  This adapter keeps that
anchor -- far starts from H5 -- and learns a residual for the far channels only,
so the near corners and the centroid stay bit-identical to ep57.

The output convolution is exactly zero-initialised, so before training the far
arm reproduces the static fusion and the near arm reproduces the base.
No bound, no clamp, no sigmoid: a residual that has to move a peak tens of
pixels must not be capped in advance.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NEAR = slice(0, 4)
FAR = slice(4, 8)
CENTROID = slice(8, 9)
IN_CHANNELS = 128 + 9 + 9 + 9 + 16      # F50, H4, H5, H6, A6
OUT_CHANNELS = 4


class PFDRAdapter(nn.Module):
    """Residual over four belief channels, from detached frozen features."""

    def __init__(self, in_channels: int = IN_CHANNELS) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, OUT_CHANNELS, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def build_input(feature: torch.Tensor, h4: torch.Tensor, h5: torch.Tensor,
                    h6: torch.Tensor, a6: torch.Tensor) -> torch.Tensor:
        """Everything the frozen base produced, detached."""
        return torch.cat([feature.detach(), h4.detach(), h5.detach(),
                          h6.detach(), a6.detach()], dim=1)

    def forward(self, adapter_input: torch.Tensor) -> torch.Tensor:
        assert adapter_input.shape[1] == self.in_channels, adapter_input.shape
        return self.out(self.trunk(adapter_input))


def fuse_far(h5: torch.Tensor, h6: torch.Tensor,
             delta: torch.Tensor) -> torch.Tensor:
    """near = H6, far = H5 + delta, centroid = H6.  Near and centroid are copied
    straight through so they stay exactly equal to the base."""
    out = h6.clone()
    out[:, FAR] = h5[:, FAR] + delta
    return out


def fuse_near(h6: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The negative control: same capacity, applied to the near face instead."""
    out = h6.clone()
    out[:, NEAR] = h6[:, NEAR] + delta
    return out


def local_soft_argmax(belief: torch.Tensor, temperature: float = 0.1,
                      window: int = 11, sigma: float = 2.0) -> torch.Tensor:
    """Differentiable coordinate read-out used only by the training losses.

    A fixed Gaussian blur picks the window centre -- that index is detached --
    and the expectation runs over the raw belief inside the window, so gradient
    reaches the values rather than the argmax.
    """
    batch, channels, height, width = belief.shape
    radius = window // 2
    size = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(size, dtype=belief.dtype, device=belief.device) - size // 2
    kernel1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel1d = kernel1d / kernel1d.sum()
    # replicate rather than zero padding: a belief whose background sits below
    # zero would otherwise have its border lifted above the true peak and the
    # argmax would pin to a corner of the map.
    half = size // 2
    smooth = belief.reshape(batch * channels, 1, height, width)
    smooth = torch.nn.functional.pad(smooth, (half, half, 0, 0), mode="replicate")
    smooth = torch.nn.functional.conv2d(smooth, kernel1d.view(1, 1, 1, -1))
    smooth = torch.nn.functional.pad(smooth, (0, 0, half, half), mode="replicate")
    smooth = torch.nn.functional.conv2d(smooth, kernel1d.view(1, 1, -1, 1))
    smooth = smooth.reshape(batch, channels, height, width)

    flat = smooth.reshape(batch, channels, -1).argmax(dim=-1).detach()
    cy = (flat // width).long()
    cx = (flat % width).long()

    # explicit advanced indexing.  gather() needs the index to match the input
    # on every other axis and quietly produced cross-channel windows here; the
    # padding value must also lose the softmax, hence -1e4 rather than zero.
    padded = torch.nn.functional.pad(belief, (radius,) * 4, value=-1e4)
    offsets = torch.arange(-radius, radius + 1, device=belief.device)
    bi = torch.arange(batch, device=belief.device)[:, None, None, None]
    ci = torch.arange(channels, device=belief.device)[None, :, None, None]
    ry = cy[..., None, None] + offsets[None, None, :, None] + radius
    rx = cx[..., None, None] + offsets[None, None, None, :] + radius
    patch = padded[bi, ci, ry, rx]                       # B x C x window x window

    weights = torch.softmax(patch.reshape(batch, channels, -1) / temperature,
                            dim=-1).reshape(batch, channels, window, window)
    ys = (cy[..., None, None] + offsets[None, None, :, None]).to(belief.dtype)
    xs = (cx[..., None, None] + offsets[None, None, None, :]).to(belief.dtype)
    ys = ys.expand(batch, channels, window, window)
    xs = xs.expand(batch, channels, window, window)
    return torch.stack([(weights * xs).sum(dim=(-2, -1)),
                        (weights * ys).sum(dim=(-2, -1))], dim=-1)
