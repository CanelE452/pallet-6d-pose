"""국소 semantic corner 전문가 — 작은 U-Net. teacher-time only.

R0 좌표는 **검색 중심만** 제공하고, 최종 위치는 국소 RGB 가 정한다.
그래서 학습 때 crop 중심을 정확한 GT 에 두지 않고, SOURCE_DEV 에서 실제로 측정한
R0 coarse residual 벡터를 복원추출해 흔든다.

출력 네 가지
    heatmap        crop 안 잔차 위치 분포 (spatial softmax)
    visibility     그 코너가 crop 안에 있고 보이는가
    uncertainty    자기 오차의 크기 예측 — 추론 때 gate 로 쓴다
    edge_dirs      그 코너에 붙는 투영 cuboid 변 2개의 방향 (sin/cos)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 64
HEATMAP_SIGMA = 2.0
N_CORNERS = 8
ID_DIM = 16


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LocalCornerSpecialist(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.embed = nn.Embedding(N_CORNERS, ID_DIM)
        w = width
        self.enc1 = _block(3 + ID_DIM, w)
        self.enc2 = _block(w, w * 2)
        self.enc3 = _block(w * 2, w * 4)
        self.bottleneck = _block(w * 4, w * 4)
        self.dec3 = _block(w * 4 + w * 4, w * 2)
        self.dec2 = _block(w * 2 + w * 2, w)
        self.dec1 = _block(w + w, w)
        self.head_heat = nn.Conv2d(w, 1, 1)
        self.head_scalar = nn.Sequential(
            nn.Linear(w * 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 6))
        self.pool = nn.MaxPool2d(2)

    def forward(self, crop: torch.Tensor, corner_id: torch.Tensor) -> dict:
        b, _, h, w = crop.shape
        ident = self.embed(corner_id).view(b, ID_DIM, 1, 1).expand(b, ID_DIM, h, w)
        e1 = self.enc1(torch.cat([crop, ident], 1))          # 64
        e2 = self.enc2(self.pool(e1))                        # 32
        e3 = self.enc3(self.pool(e2))                        # 16
        bt = self.bottleneck(self.pool(e3))                  # 8
        d3 = self.dec3(torch.cat([F.interpolate(bt, scale_factor=2), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        logits = self.head_heat(d1).flatten(1)               # (b, 64*64)
        scal = self.head_scalar(bt.mean(dim=(2, 3)))
        return {
            "heat_logits": logits,
            "visibility_logit": scal[:, 0],
            "log_uncertainty": scal[:, 1],
            "edge_dirs": scal[:, 2:6].view(b, 2, 2),
        }


# ------------------------------------------------------------------ helpers --
def gaussian_target(xy: np.ndarray, size: int = CROP, sigma: float = HEATMAP_SIGMA):
    """crop 좌표 (x, y) 에 놓인 정규화 가우시안."""
    ax = np.arange(size, dtype=np.float32)
    gx = np.exp(-((ax - xy[0]) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ax - xy[1]) ** 2) / (2 * sigma ** 2))
    g = np.outer(gy, gx)
    s = g.sum()
    return (g / s).astype(np.float32) if s > 1e-12 else g.astype(np.float32)


def mixture_target(points: np.ndarray, size: int = CROP, sigma: float = 3.0):
    """teacher 좌표들을 중심으로 한 등가중 가우시안 혼합 — hard coordinate 를 쓰지 않는다."""
    acc = np.zeros((size, size), dtype=np.float32)
    n = 0
    for p in points:
        if not np.isfinite(p).all() or not (-2 <= p[0] < size + 2 and -2 <= p[1] < size + 2):
            continue
        acc += gaussian_target(p, size, sigma)
        n += 1
    if n == 0:
        return None
    acc /= acc.sum()
    return acc


# crop 추출은 torch 없는 mtcd_common 에 산다 — 여기서는 재수출만 한다.
from mtcd_common import extract_crop  # noqa: E402,F401


def soft_argmax(heat_logits: torch.Tensor, size: int = CROP) -> torch.Tensor:
    p = F.softmax(heat_logits, dim=1).view(-1, size, size)
    ax = torch.arange(size, device=heat_logits.device, dtype=heat_logits.dtype)
    x = (p.sum(1) * ax).sum(1)
    y = (p.sum(2) * ax).sum(1)
    return torch.stack([x, y], 1)


def hard_argmax(heat_logits: torch.Tensor, size: int = CROP) -> torch.Tensor:
    idx = heat_logits.argmax(dim=1)
    return torch.stack([(idx % size).float(), (idx // size).float()], 1)
