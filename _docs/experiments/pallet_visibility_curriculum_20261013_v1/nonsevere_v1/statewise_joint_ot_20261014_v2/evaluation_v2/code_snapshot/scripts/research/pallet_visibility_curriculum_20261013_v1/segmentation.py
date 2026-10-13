"""Tiny RGB-only source-supervised pallet material/shape segmentation.

Inference consumes an image and frozen weights only. The foreground sigmoid
is an uncalibrated learned score; an object mask cannot identify a particular
hidden 3-D keypoint or establish ownership of a same-coloured occluder.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CONFIG = {"channels": [8, 16, 32], "input_size": 256,
                  "input_order": "RGB", "normalization": "uint8/255", "threshold": .5}
_MODELS = {}


class Block(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=False),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=False))

    def forward(self, image):
        return self.layers(image)


class TinyUNet(nn.Module):
    def __init__(self, channels=(8, 16, 32)):
        super().__init__()
        a, b, c = channels
        self.enc1 = Block(3, a)
        self.enc2 = Block(a, b)
        self.enc3 = Block(b, c)
        self.dec2 = Block(b + c, b)
        self.dec1 = Block(a + b, a)
        self.out = nn.Conv2d(a, 1, 1)

    def forward(self, image):
        first = self.enc1(image)
        second = self.enc2(F.max_pool2d(first, 2))
        third = self.enc3(F.max_pool2d(second, 2))
        second_out = self.dec2(torch.cat((second, F.interpolate(third, size=second.shape[-2:], mode="nearest")), 1))
        first_out = self.dec1(torch.cat((first, F.interpolate(second_out, size=first.shape[-2:], mode="nearest")), 1))
        return self.out(first_out)


def image_tensor(image_bgr, size=256):
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image [H,W,3]")
    rgb = cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255).unsqueeze(0)


def load_segmenter(modelweights, device=None):
    """Cache a frozen tiny model. No datasets, labels, or other models are read."""
    path = Path(modelweights).resolve()
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(path), path.stat().st_mtime_ns, target)
    if key not in _MODELS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(DEFAULT_CONFIG, **checkpoint["config"])
        model = TinyUNet(config["channels"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(target).eval().requires_grad_(False)
        _MODELS[key] = (model, config, target)
    return _MODELS[key]


def segment(image_bgr, modelweights, device=None):
    """Return a float32 H×W learned foreground score, with no GT input."""
    model, config, target = load_segmenter(modelweights, device)
    image = np.asarray(image_bgr)
    with torch.inference_mode():
        probability = model(image_tensor(image, config["input_size"]).to(target)).sigmoid()[0, 0].cpu().numpy()
    return cv2.resize(probability.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
