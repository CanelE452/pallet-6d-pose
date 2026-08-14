"""run_live.py — I/O + 전처리 모듈.

NpDepthFrame  : numpy uint16(mm) depth → RealSense depth_frame 인터페이스 흉내
load_seq      : data/outside/capture* 시퀀스 (RGB + depth) 로드
sample_depth  : depth_frame 의 (x, y) 주변 픽셀 중앙값
scale_K       : intrinsic K 행렬 scale (resize 시 보정)
run_forward   : DOPE forward (belief + affinity tensor 반환)
extract_peaks : belief map 의 채널별 peak 위치 + 값
build_belief_grid : 9-belief 채널을 3×3 grid heatmap 으로
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from scipy.ndimage import gaussian_filter


_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class NpDepthFrame:
    """numpy uint16(mm) depth → RealSense depth_frame 인터페이스 흉내."""
    def __init__(self, depth_mm_u16):
        self.d = depth_mm_u16
        self.h, self.w = depth_mm_u16.shape

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_distance(self, x, y):
        xi, yi = int(x), int(y)
        if 0 <= yi < self.h and 0 <= xi < self.w:
            return float(self.d[yi, xi]) / 1000.0
        return 0.0


def load_seq(seq_dir):
    """data/outside/capture* 형식 시퀀스 로드.

    반환: (frames, K) — frames 는 [(rgb_path, depth_path), ...]
    """
    K = None
    K_path = os.path.join(seq_dir, "cam_K.txt")
    if os.path.isfile(K_path):
        K = np.loadtxt(K_path, dtype=np.float64).reshape(3, 3)

    rgb_paths = sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames in {seq_dir}/rgb/")
    depth_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in depth_paths}
    frames = []
    for r in rgb_paths:
        stem = os.path.splitext(os.path.basename(r))[0]
        frames.append((r, depth_by_stem.get(stem)))
    return frames, K


def sample_depth(depth_frame, x, y, radius=3):
    """(x, y) 주변 radius 픽셀의 depth 중앙값 (m). None 이면 sample 불가."""
    if depth_frame is None:
        return None
    fw, fh = depth_frame.get_width(), depth_frame.get_height()
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < fw and 0 <= ny < fh:
                d = depth_frame.get_distance(nx, ny)
                if d > 0.05:
                    vals.append(d)
    return float(np.median(vals)) if vals else None


def scale_K(K, s):
    """resize 비율 s 에 맞춰 intrinsic K 조정 (fx, fy, cx, cy 모두 ×s)."""
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] *= s; K2[1, 2] *= s
    return K2


def run_forward(net, img_rgb):
    """DOPE forward — RGB image → (belief, affinity) tensor (last stage)."""
    t = _transform(img_rgb)
    with torch.no_grad():
        out, seg = net(Variable(t).cuda().unsqueeze(0))
    return out[-1][0], seg[-1][0]


def extract_peaks(vertex2, sigma=3):
    """채널별 (Gaussian smoothed) peak 위치 + 값.
    각 entry: {'val': float, 'x': int (pixel), 'y': int (pixel)}.
    output_size 50 → input 400 라 ×8 곱해 image 좌표.
    """
    peaks = []
    for ch in range(vertex2.size(0)):
        raw = vertex2[ch].cpu().numpy()
        sm = gaussian_filter(raw, sigma=sigma)
        val = float(sm.max())
        idx = np.unravel_index(sm.argmax(), sm.shape)
        peaks.append({'val': val, 'x': idx[1] * 8, 'y': idx[0] * 8})
    return peaks


def build_belief_grid(vertex2, img_bgr):
    """9-channel belief map → 3×3 grid heatmap (debug 시각화)."""
    upsampling = nn.UpsamplingNearest2d(scale_factor=8)
    h, w = img_bgr.shape[:2]
    cells = []
    for ch in range(min(vertex2.size(0), 9)):
        b = vertex2[ch].clone()
        bmin, bmax = float(b.min()), float(b.max())
        if bmax > bmin:
            b = (b - bmin) / (bmax - bmin)
        b_up = upsampling(b.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
        b_np = cv2.resize(b_up.cpu().numpy(), (w, h))
        hm = cv2.applyColorMap((b_np * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        bg = (img_bgr.astype(np.float32) * 0.4).astype(np.uint8)
        ov = cv2.addWeighted(bg, 1, hm, 0.6, 0)
        lbl = f"c{ch}" if ch < 8 else "ctr"
        cv2.putText(ov, lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(ov)
    while len(cells) < 9:
        cells.append(np.zeros((h, w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[r*3:r*3+3]) for r in range(3)]
    return np.vstack(rows)
