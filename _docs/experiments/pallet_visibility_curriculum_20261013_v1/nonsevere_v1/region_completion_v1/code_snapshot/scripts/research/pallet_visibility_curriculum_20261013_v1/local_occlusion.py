"""Source-only local foreground-support CNN with synthetic opacity supervision.

This model predicts a local RGB support event, not verified physical3D corner
visibility. Inference uses RGB64 patches, a predicted-hull channel, and a
query-centred heatmap. Same-RGB alternative ownership remains unidentifiable.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

from . import appearance as A


SIZE = 64
_MODELS = {}
_yy, _xx = np.mgrid[:SIZE, :SIZE]
HEAT = np.exp(-((_xx - 31.5) ** 2 + (_yy - 31.5) ** 2) / (2 * 3. ** 2)).astype(np.float32)


class LocalSupportCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(5, 8, 5, stride=2, padding=2, bias=False), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(4, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, value):
        return self.head(self.body(value)).squeeze(1)


def patch_inputs(image_bgr, points):
    image = np.asarray(image_bgr)
    q = np.asarray(points, float)[:8]
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or q.shape != (8, 2):
        raise ValueError("Expected uint8 BGR[H,W,3] and predicted points[8/9,2]")
    hull = np.zeros(image.shape[:2], np.uint8)
    valid = np.isfinite(q).all(1) & ~(q == -1).all(1)
    if valid.sum() >= 3:
        A._fill_polygon(hull, q[valid])
    patches, expected = [], []
    for point, ok in zip(q, valid):
        if not ok:
            patches.append(np.zeros((SIZE, SIZE, 3), np.uint8))
            expected.append(np.zeros((SIZE, SIZE), np.uint8))
            continue
        center = tuple(float(x) for x in point)
        patches.append(cv2.getRectSubPix(image, (SIZE, SIZE), center))
        expected.append(cv2.getRectSubPix(hull, (SIZE, SIZE), center))
    return np.asarray(patches), np.asarray(expected), valid


def tensor_inputs(rgb_bgr, expected, device):
    values = np.ascontiguousarray(np.asarray(rgb_bgr)[..., ::-1].transpose(0, 3, 1, 2))
    rgb = torch.from_numpy(values).to(device, dtype=torch.float32).div_(255.)
    geometry = torch.from_numpy(np.ascontiguousarray(expected[:, None])).to(device, dtype=torch.float32).div_(255.)
    heat = torch.from_numpy(HEAT)[None, None].to(device).expand(len(rgb), -1, -1, -1)
    return torch.cat((rgb, geometry, heat), 1)


def load_model(weights, device=None):
    path = Path(weights).resolve()
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(path), path.stat().st_mtime_ns, target)
    if key not in _MODELS:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = LocalSupportCNN().to(target)
        model.load_state_dict(saved["state_dict"])
        _MODELS[key] = (model.eval().requires_grad_(False), target)
    return _MODELS[key]


def infer_local_support(image_bgr, points, self_hidden, weights, device=None):
    q = np.asarray(points, float)[:8]
    hidden = np.asarray(self_hidden, bool)
    if hidden.shape != (8,):
        raise ValueError("Expected fixed self_hidden[8]")
    patches, expected, valid = patch_inputs(image_bgr, q)
    model, target = load_model(weights, device)
    with torch.inference_mode():
        probability = model(tensor_inputs(patches, expected, target)).sigmoid().cpu().numpy()
    p = np.clip(probability.astype(float), 1e-8, 1 - 1e-8)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.)
    h, w = np.asarray(image_bgr).shape[:2]
    states = []
    for k in range(8):
        inside = valid[k] and 0 <= q[k, 0] < w and 0 <= q[k, 1] < h
        if not valid[k]:
            state = "UNKNOWN"
        elif not inside:
            state = "OUT_OF_FRAME"
        elif hidden[k]:
            state = "SELF_OCCLUDED"
        elif entropy[k] >= .8:
            state = "UNKNOWN_HIGH_ENTROPY"
        else:
            state = "LOCAL_FOREGROUND_SUPPORTED" if p[k] >= .5 else "LOCAL_FOREGROUND_ABSENCE_SUSPECTED"
        states.append(state)
    return dict(local_support_score=probability.astype(float).tolist(), entropy=entropy.tolist(), states=states,
        inference_inputs=["RGB64patch", "predicted_hull_channel", "center_heatmap", "fixed_self_hidden"],
        physical3D_visibility_certified=False, fit_points_changed=False,
        limits=["source_graphite_only", "local_support_is_not_semantic_corner_visibility",
                "same_RGB_alternative_ownership", "entropy_is_not_an_error_certificate"])
