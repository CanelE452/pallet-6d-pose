"""Small frozen-backbone query readers. No data or evaluation-file imports.

The scalar output is always the logit at spatial index (4, 4). The spatial
model retains a 5x5 receptive field on the 9x9 P3 ROI and never pools the ROI.
The optional RGB branch is likewise read at its exact spatial center; RGB
coordinates must be centered on the same native query as the P3 cache.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HeadConfig:
    architecture: str = "spatial"
    in_channels: int = 65
    hidden_channels: int = 32
    coordinate_channels: bool = True
    rgb_branch: bool = False
    rgb_hidden_channels: int = 16


class QueryHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HeadConfig) else HeadConfig(**(config or {}))
        c = self.config
        if c.architecture not in {"spatial", "gap"}:
            raise ValueError("architecture must be spatial or gap")
        channels = c.in_channels + (2 if c.coordinate_channels else 0)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, c.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.hidden_channels, c.hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        if c.rgb_branch:
            h = c.rgb_hidden_channels
            self.rgb_encoder = nn.Sequential(nn.Conv2d(3, h, 3, padding=1), nn.GELU(),
                                             nn.Conv2d(h, h, 3, padding=1), nn.GELU())
            channels = c.hidden_channels + h
        else:
            self.rgb_encoder = None
            channels = c.hidden_channels
        self.readout = nn.Conv2d(channels, 1, 1)

    def _features(self, x, rgb=None):
        if x.ndim != 4 or x.shape[1:] != (self.config.in_channels, 9, 9):
            raise ValueError(f"expected N,{self.config.in_channels},9,9, got {tuple(x.shape)}")
        if self.config.coordinate_channels:
            axis = torch.linspace(-1, 1, 9, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            coordinates = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(len(x), -1, -1, -1)
            x = torch.cat((x, coordinates), dim=1)
        h = self.encoder(x)
        if self.rgb_encoder is not None:
            if rgb is None or rgb.ndim != 4 or rgb.shape[:2] != (len(x), 3):
                raise ValueError("RGB branch requires N,3,H,W RGB tensor in [0,1]")
            if rgb.shape[-2] % 2 != 1 or rgb.shape[-1] % 2 != 1:
                raise ValueError("RGB dimensions must be odd to preserve an exact center pixel")
            rh = self.rgb_encoder(rgb)
            # align_corners=True on odd-sized grids preserves their exact center.
            rh = F.interpolate(rh, (9, 9), mode="bilinear", align_corners=True)
            h = torch.cat((h, rh), dim=1)
        return h

    def dense_logits(self, x, rgb=None):
        return self.readout(self._features(x, rgb))[:, 0]

    def forward(self, x, rgb=None, return_dense=False):
        dense = self.dense_logits(x, rgb)
        scalar = dense[:, 4, 4] if self.config.architecture == "spatial" else dense.mean(dim=(-2, -1))
        return (scalar, dense) if return_dense else scalar


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path, model, **metadata):
    payload = {"format_version": 1, "head_config": asdict(model.config),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               **metadata}
    torch.save(payload, Path(path))


def load_model(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = QueryHead(payload["head_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
