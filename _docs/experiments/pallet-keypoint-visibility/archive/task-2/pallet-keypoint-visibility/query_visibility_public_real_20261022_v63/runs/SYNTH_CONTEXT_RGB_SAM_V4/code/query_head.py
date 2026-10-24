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


def native9_rgb_features(features):
    """Sample a full-side odd RGB grid at native P3's nine cell centers.

    P3 ROI samples cover offsets (i-4)*L/9. The RGB crop spans [-L/2,L/2],
    so these map to normalized [-8/9,8/9]. Separable bilinear slices keep
    CUDA backward deterministic, unlike CUDA grid_sample backward.
    """
    height, width = features.shape[-2:]
    if features.ndim != 4 or height % 2 != 1 or width % 2 != 1:
        raise ValueError("native9 expects N,C,odd,odd RGB feature map")

    def indices(size, index):
        numerator = (size - 1) // 2 + (size - 1) * index
        return numerator // 9, (numerator % 9) / 9.0

    rows = []
    for index in range(9):
        lo, frac = indices(height, index)
        rows.append(features[..., lo, :] if frac == 0 else
                    (1 - frac) * features[..., lo, :] + frac * features[..., lo + 1, :])
    sampled = torch.stack(rows, dim=-2)
    columns = []
    for index in range(9):
        lo, frac = indices(width, index)
        columns.append(sampled[..., lo] if frac == 0 else
                       (1 - frac) * sampled[..., lo] + frac * sampled[..., lo + 1])
    return torch.stack(columns, dim=-1)


def target_context_from_rois(rois):
    """GT-free prototype from one target's query ROIs, (...,Q,65,9,9).

    Feature channel 64 is the saved native-grid validity bit. Average only the
    first 64 feature channels over valid cells of all Q initial-query ROIs.
    No target mask, visibility annotation, or refined coordinates are read.
    """
    tensor_input = isinstance(rois, torch.Tensor)
    tensor = rois.float() if tensor_input else torch.as_tensor(rois).float()
    if tensor.ndim < 4 or tuple(tensor.shape[-3:]) != (65, 9, 9):
        raise ValueError("prototype expects (...,Q,65,9,9) ROI array")
    features, valid = tensor[..., :64, :, :], tensor[..., 64:65, :, :].clamp(0, 1)
    result = (features * valid).sum(dim=(-4, -2, -1)) / valid.sum(dim=(-4, -2, -1)).clamp_min(1)
    return result if tensor_input else result.cpu().numpy()


@dataclass
class HeadConfig:
    architecture: str = "spatial"
    in_channels: int = 65
    hidden_channels: int = 32
    coordinate_channels: bool = True
    rgb_branch: bool = False
    rgb_hidden_channels: int = 16
    target_context: bool = False
    context_channels: int = 64
    # Missing in old checkpoints: reproduce the original V2 pilot behavior.
    rgb_sampling_version: str = "legacy_endpoints"
    sam_branch: bool = False
    sam_hidden_channels: int = 8
    sam_prior_mode: str = "none"
    sam_logit_scale: float = 8.0
    sam_zero_init_residual: bool = True


class QueryHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HeadConfig) else HeadConfig(**(config or {}))
        c = self.config
        if c.architecture not in {"spatial", "gap"}:
            raise ValueError("architecture must be spatial or gap")
        if c.rgb_sampling_version not in {"legacy_endpoints", "native9"}:
            raise ValueError("unknown RGB sampling version")
        if c.sam_prior_mode not in {"none", "learned_raw_residual"}:
            raise ValueError("unknown SAM prior mode")
        channels = c.in_channels + (2 if c.coordinate_channels else 0) + (c.context_channels if c.target_context else 0)
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
        if c.sam_branch:
            h = c.sam_hidden_channels
            self.sam_encoder = nn.Sequential(nn.Conv2d(1, h, 3, padding=1), nn.GELU(),
                                             nn.Conv2d(h, h, 3, padding=1), nn.GELU())
            channels += h
        else:
            self.sam_encoder = None
        self.readout = nn.Conv2d(channels, 1, 1)
        if c.sam_branch and c.sam_prior_mode == "learned_raw_residual":
            self.sam_prior_weight = nn.Parameter(torch.ones(()))
            if c.sam_zero_init_residual:
                nn.init.zeros_(self.readout.weight)
                nn.init.zeros_(self.readout.bias)
        else:
            self.register_parameter("sam_prior_weight", None)

    def _features(self, x, rgb=None, context=None, sam=None):
        if x.ndim != 4 or x.shape[1:] != (self.config.in_channels, 9, 9):
            raise ValueError(f"expected N,{self.config.in_channels},9,9, got {tuple(x.shape)}")
        if self.config.coordinate_channels:
            axis = torch.linspace(-1, 1, 9, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            coordinates = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(len(x), -1, -1, -1)
            x = torch.cat((x, coordinates), dim=1)
        if self.config.target_context:
            expected = (len(x), self.config.context_channels)
            if context is None or tuple(context.shape) != expected:
                raise ValueError(f"target-context head requires context of shape {expected}")
            x = torch.cat((x, context[:, :, None, None].expand(-1, -1, 9, 9)), dim=1)
        h = self.encoder(x)
        if self.rgb_encoder is not None:
            if rgb is None or rgb.ndim != 4 or rgb.shape[:2] != (len(x), 3):
                raise ValueError("RGB branch requires N,3,H,W RGB tensor in [0,1]")
            if rgb.shape[-2] % 2 != 1 or rgb.shape[-1] % 2 != 1:
                raise ValueError("RGB dimensions must be odd to preserve an exact center pixel")
            rh = self.rgb_encoder(rgb)
            if self.config.rgb_sampling_version == "native9":
                rh = native9_rgb_features(rh)
            else:
                rh = F.interpolate(rh, (9, 9), mode="bilinear", align_corners=True)
            h = torch.cat((h, rh), dim=1)
        if self.sam_encoder is not None:
            if sam is None or tuple(sam.shape) != (len(x), 1, 9, 9):
                raise ValueError("SAM branch requires N,1,9,9 normalized selected-logit grid")
            h = torch.cat((h, self.sam_encoder(sam)), dim=1)
        return h

    def dense_logits(self, x, rgb=None, context=None, sam=None):
        dense = self.readout(self._features(x, rgb, context, sam))[:, 0]
        if self.sam_prior_weight is not None:
            dense = dense + self.sam_prior_weight * self.config.sam_logit_scale * sam[:, 0]
        return dense

    def forward(self, x, rgb=None, context=None, sam=None, return_dense=False):
        dense = self.dense_logits(x, rgb, context, sam)
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
