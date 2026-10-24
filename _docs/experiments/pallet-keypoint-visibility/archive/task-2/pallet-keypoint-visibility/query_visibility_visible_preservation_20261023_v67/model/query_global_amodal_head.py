"""Global shape and native-query fusion for six valid exact/radius-2 state pairs.

Joint states: (V,V), (E,V), (E,E), (O,V), (O,E), (O,O).
The exact and nearby visible/full marginals obey their inclusion constraints
by construction. Semantic corner self-visibility is not one of these labels.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


JOINT_PAIRS = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
STATE_NAMES = ("VISIBLE_TARGET", "EXTERNAL_SUPPORT_PROXY", "OUTSIDE_FULL_SUPPORT")


@dataclass
class HeadConfig:
    p3_channels: int = 65
    p3_hidden: int = 32
    rgb_native_channels: int = 75
    rgb_hidden: int = 16
    sam_channels: int = 2
    sam_hidden: int = 8
    context_channels: int = 64
    context_hidden: int = 16
    fusion_hidden: int = 32
    box_grid: bool = True
    coordinate_channels: bool = True
    sam_role: str = "features_only"
    joint_order: str = "VV_EV_EE_OV_OE_OO"
    global_channels: int = 5
    global_bottleneck: int = 32
    global_sampling: str = "nearest_onehot_matmul"


def classes_to_joint(classes, valid):
    """NumPy conversion with explicit rejection of impossible valid labels."""
    classes, valid = np.asarray(classes), np.asarray(valid, bool)
    if classes.ndim != 4 or classes.shape[1:] != (2, 9, 9) or valid.shape != (len(classes), 9, 9):
        raise ValueError("classes N,2,9,9 and valid N,9,9 are required")
    output = np.full(valid.shape, -1, np.int64)
    for joint, (exact, nearby) in enumerate(JOINT_PAIRS):
        output[(classes[:, 0] == exact) & (classes[:, 1] == nearby) & valid] = joint
    if (output[valid] < 0).any():
        indices = np.argwhere(valid & (output < 0))[:5]
        raise ValueError(f"impossible exact/radius2 label pair at {indices.tolist()}")
    return output


def joint_to_classes(joint):
    pairs = np.asarray(JOINT_PAIRS, np.int64)
    joint = np.asarray(joint)
    if ((joint < 0) | (joint > 5)).any():
        raise ValueError("joint codes must be 0..5")
    return pairs[joint]


def marginals(joint_probability):
    """Class axis is 1, accepting N,6 or N,6,H,W probabilities."""
    p = joint_probability
    if p.ndim not in (2, 4) or p.shape[1] != 6:
        raise ValueError("joint probabilities must be N,6 or N,6,H,W")
    stack = torch.stack if isinstance(p, torch.Tensor) else np.stack
    exact = stack((p[:, 0], p[:, 1] + p[:, 2], p[:, 3] + p[:, 4] + p[:, 5]), 1)
    nearby = stack((p[:, 0] + p[:, 1] + p[:, 3], p[:, 2] + p[:, 4], p[:, 5]), 1)
    return {"joint_probability": p, "state_probability": stack((exact, nearby), 1),
            "visible_probability": stack((p[:, 0], p[:, 0] + p[:, 1] + p[:, 3]), 1),
            "full_probability": stack((p[:, :3].sum(1), p[:, :5].sum(1)), 1)}


def sample_nearest_onehot(features, target_inverse, indices, valid):
    """Deterministic differentiable nearest sampling; no CUDA grid_sample scatter."""
    if features.ndim != 4 or indices.shape != valid.shape or indices.shape[1:] != (9, 9):
        raise ValueError("features U,C,H,W and N,9,9 indices/valid required")
    n, cells = len(indices), features.shape[-2] * features.shape[-1]
    if target_inverse.shape != (n,):
        raise ValueError("target_inverse must be N")
    if indices.dtype != torch.long or target_inverse.dtype != torch.long:
        raise ValueError("indices and target_inverse must be int64")
    # Cache validators check bounds before GPU use. The one-hot matrix makes
    # backward a deterministic batched GEMM, rather than repeated gather scatter.
    onehot = torch.nn.functional.one_hot(indices.reshape(n, 81), cells).to(features.dtype)
    onehot = onehot * valid.reshape(n, 81, 1).to(features.dtype)
    assignment = torch.nn.functional.one_hot(target_inverse, len(features)).to(features.dtype)
    selected = torch.matmul(assignment, features.flatten(1)).reshape(n, features.shape[1], cells)
    return torch.bmm(selected, onehot.transpose(1, 2)).reshape(n, features.shape[1], 9, 9)


class GlobalShapeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(inp, out):
            return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(out, out, 3, padding=1), nn.GELU())
        self.encoder0 = block(5, 8)
        self.encoder1 = block(8, 16)
        self.encoder2 = block(16, 24)
        self.encoder3 = block(24, 32)
        self.decoder2 = nn.Sequential(nn.Conv2d(56, 24, 3, padding=1), nn.GELU())
        self.decoder1 = nn.Sequential(nn.Conv2d(40, 16, 3, padding=1), nn.GELU())
        self.decoder0 = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.GELU())
        self.readout = nn.Conv2d(8, 6, 1)

    def forward(self, value):
        if value.ndim != 4 or value.shape[1:] != (5, 64, 64):
            raise ValueError("global input must be U,5,64,64")
        pool = torch.nn.functional.avg_pool2d
        def up(x):
            return x.repeat_interleave(2, 2).repeat_interleave(2, 3)
        e0 = self.encoder0(value)
        e1 = self.encoder1(pool(e0, 2))
        e2 = self.encoder2(pool(e1, 2))
        latent = self.encoder3(pool(e2, 2))
        d2 = self.decoder2(torch.cat((up(latent), e2), 1))
        d1 = self.decoder1(torch.cat((up(d2), e1), 1))
        d0 = self.decoder0(torch.cat((up(d1), e0), 1))
        return self.readout(d0), latent


class QueryGlobalAmodalHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HeadConfig) else HeadConfig(**(config or {}))
        c = self.config
        if (c.sam_role != "features_only" or c.joint_order != "VV_EV_EE_OV_OE_OO"
                or c.global_sampling != "nearest_onehot_matmul" or c.global_channels != 5
                or c.global_bottleneck != 32):
            raise ValueError("only the registered features-only global six-joint model is supported")
        self.global_encoder = GlobalShapeUNet()
        self.p3_encoder = nn.Sequential(nn.Conv2d(c.p3_channels, c.p3_hidden, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(c.p3_hidden, c.p3_hidden, 3, padding=1), nn.GELU())
        self.rgb_encoder = nn.Sequential(nn.Conv2d(c.rgb_native_channels, c.rgb_hidden, 1), nn.GELU(),
                                          nn.Conv2d(c.rgb_hidden, c.rgb_hidden, 3, padding=1), nn.GELU())
        self.sam_encoder = nn.Sequential(nn.Conv2d(c.sam_channels, c.sam_hidden, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(c.sam_hidden, c.sam_hidden, 3, padding=1), nn.GELU())
        self.context_encoder = nn.Sequential(nn.Linear(c.context_channels, c.context_hidden), nn.GELU())
        channels = c.p3_hidden + c.rgb_hidden + c.sam_hidden + c.context_hidden + 32 + 6
        channels += (4 if c.box_grid else 0) + (2 if c.coordinate_channels else 0)
        self.fusion = nn.Sequential(nn.Conv2d(channels, c.fusion_hidden, 1), nn.GELU(),
                                     nn.Conv2d(c.fusion_hidden, c.fusion_hidden, 3, padding=1), nn.GELU())
        self.readout = nn.Conv2d(c.fusion_hidden, 6, 1)

    def dense_logits(self, x, rgb_native, sam, context, box_grid, global_logits,
                     global_bottleneck, target_inverse, global_index64,
                     global_index8, global_query_valid):
        n, c = len(x), self.config
        expected = ((x, (n, c.p3_channels, 9, 9), "x"),
                    (rgb_native, (n, c.rgb_native_channels, 9, 9), "rgb_native"),
                    (sam, (n, c.sam_channels, 9, 9), "sam"),
                    (context, (n, c.context_channels), "context"))
        for value, shape, name in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(value.shape)}")
        prototype = self.context_encoder(context)[:, :, None, None].expand(-1, -1, 9, 9)
        branches = [self.p3_encoder(x), self.rgb_encoder(rgb_native), self.sam_encoder(sam), prototype,
                    sample_nearest_onehot(global_bottleneck, target_inverse, global_index8, global_query_valid),
                    sample_nearest_onehot(global_logits, target_inverse, global_index64, global_query_valid)]
        if c.box_grid:
            if box_grid is None or tuple(box_grid.shape) != (n, 4, 9, 9):
                raise ValueError("input-derived box_grid N,4,9,9 is required")
            branches.append(box_grid)
        if c.coordinate_channels:
            axis = torch.linspace(-1, 1, 9, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            branches.append(torch.stack((xx, yy))[None].expand(n, -1, -1, -1))
        return self.readout(self.fusion(torch.cat(branches, dim=1)))

    def forward(self, x, rgb_native, sam, context, box_grid, global_input,
                target_inverse, global_index64, global_index8, global_query_valid,
                return_dense=False):
        global_logits, latent = self.global_encoder(global_input)
        dense = self.dense_logits(x, rgb_native, sam, context, box_grid, global_logits,
                                  latent, target_inverse, global_index64,
                                  global_index8, global_query_valid)
        center = dense[:, :, 4, 4]
        return (center, dense, global_logits) if return_dense else center


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path, model, **metadata):
    torch.save({"format_version": 1, "head_config": asdict(model.config),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **metadata}, Path(path))


def load_model(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = QueryGlobalAmodalHead(payload["head_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
