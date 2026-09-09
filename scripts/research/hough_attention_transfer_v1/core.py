"""Shared, explicitly supervised attention experiment. No real-data training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
GRID = 50
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load_dh():
    p = ROOT / 'scripts/stage0/line/direct_hough_role_heatmap.py'
    spec = importlib.util.spec_from_file_location('ATTN_DH', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def load_backbone(path, device):
    """Load only the exact frozen DOPE VGG tap; no belief/affinity inference."""
    sys.path.insert(0, str(ROOT / 'Deep_Object_Pose/common'))
    from models import DopeNetwork
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model = DopeNetwork().vgg
    weights = {k[len('vgg.'):]: v for k, v in state.items() if k.startswith('vgg.')}
    model.load_state_dict(weights, strict=True)
    return model.requires_grad_(False).eval().to(device)


def forward_attention(model, f50, hypothesis):
    """Same descriptor operations as DirectHoughModel, return actual weights."""
    enc = model.encoder
    batch = len(f50)
    flat = f50.flatten(2).transpose(1, 2)
    xy = enc.coordinates[None].expand(batch, -1, -1)
    tokens = enc.to_token(torch.cat([flat, xy], -1))
    query = enc.norm_query(enc.queries.weight[None].expand(batch, -1, -1))
    attended, weights = enc.attention(query, tokens, tokens, need_weights=True,
                                     average_attn_weights=True)
    descriptor = enc.norm_out(query + attended)
    descriptor = descriptor + enc.ffn(descriptor)
    return model.head(descriptor, hypothesis), weights.reshape(batch, 12, GRID, GRID)


def foreground_loss(weights, mask, supported):
    """-log attention mass on visible pallet; does NOT demand uniform attention.

    Same foreground target for every supported role, including hidden cuboid
    edges: this tests coarse object localization, not per-edge localization.
    """
    mass = (weights * mask[:, None]).sum((-1, -2))
    valid = supported.float() * (mask.sum((-1, -2)) > 0)[:, None]
    loss = (-mass.clamp_min(1e-8).log() * valid).sum() / valid.sum().clamp_min(1)
    return loss, mass


def prepare(record, pad):
    bgr = cv2.imread(record['image'])
    if bgr is None:
        raise ValueError(f"Unreadable image: {record['image']}")
    height, width = bgr.shape[:2]
    if (width, height) != (record['width'], record['height']):
        raise ValueError(f"Image/GT size mismatch: {record['id']}")
    canvas = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    rgb = cv2.cvtColor(cv2.resize(canvas, (400, 400)), cv2.COLOR_BGR2RGB)
    image = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    points = np.asarray(record['gt_points'], np.float64)
    grid = (points + pad) * np.array([GRID / (width + 2*pad), GRID / (height + 2*pad)])
    mask = np.zeros((height, width), np.float32)
    if record.get('mask'):
        source = cv2.imread(record['mask'], cv2.IMREAD_UNCHANGED)
        if source is None or source.shape[:2] != (height, width):
            raise ValueError(f"Bad mask: {record['id']}")
        if source.ndim == 3:
            source = source.max(-1)
        mask = (source > 0).astype(np.float32)
        if mask.sum() == 0:
            raise ValueError(f"Empty mask: {record['id']}")
    mask = np.pad(mask, pad)
    mask = cv2.resize(mask, (GRID, GRID), interpolation=cv2.INTER_AREA)
    return image, grid, mask


def geometry(dh, records, grids):
    tc, rc, supported = dh.batch_rows({'grid': np.asarray(grids)}, EDGES)
    # Support means segment intersects ORIGINAL image; it is not visibility.
    for i, rec in enumerate(records):
        pts = np.asarray(rec['gt_points'], float)
        valid = np.asarray(rec.get('gt_valid', [True]*8), bool)
        for r, (a, b) in enumerate(EDGES):
            good = valid[a] and valid[b] and np.isfinite(pts[[a, b]]).all()
            if good:
                # Scale to unit square before canonical Liang-Barsky clipping.
                p = pts[[a, b]] / [rec['width'], rec['height']]
                _, _, hit = dh.V2.clip_segment(p[0], p[1], lo=0., hi=1.)
                good = bool(hit) and np.linalg.norm(pts[a]-pts[b]) >= 2.
            supported[i, r] &= bool(good)
    return tc, rc, supported


def line_pixels(dh, theta, rho, width, height, pad):
    rad, canonical_rho = dh.canonical_from_centred(theta, rho)
    normal = torch.stack([rad.cos(), rad.sin()], -1).cpu().numpy()
    base = canonical_rho.cpu().numpy()[:, None] * normal
    direction = np.stack([-normal[:, 1], normal[:, 0]], -1)
    points = np.stack([base - 100*direction, base + 100*direction], 1)
    return points * np.array([(width+2*pad)/GRID, (height+2*pad)/GRID]) - pad


def pixel_errors(lines, gt_points):
    """Angle in original image and mean endpoint-to-infinite-line distance."""
    truth = np.asarray(gt_points)[np.asarray(EDGES)]
    v = lines[:, 1] - lines[:, 0]
    v /= np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)
    g = truth[:, 1] - truth[:, 0]
    g /= np.linalg.norm(g, axis=-1, keepdims=True).clip(1e-12)
    angles = np.rad2deg(np.arccos(np.abs((v*g).sum(-1)).clip(0, 1)))
    normal = np.stack([-v[:, 1], v[:, 0]], -1)
    distance = np.abs(((truth-lines[:, :1]) * normal[:, None]).sum(-1)).mean(-1)
    return angles, distance


def original_attention(weights, width, height, pad):
    """Crop padding from the attention canvas for an honest original-image plot.

    Return original-region density WITHOUT renormalizing away pad attention.
    Integral may be <1; figures must not mistake this for raw token weights.
    """
    if pad == 0:
        return weights
    full_w, full_h = width + 2*pad, height + 2*pad
    # Coordinate centers corresponding to original-image bins, align_corners=False.
    xs = (torch.arange(GRID, device=weights.device) + .5) / GRID * width + pad
    ys = (torch.arange(GRID, device=weights.device) + .5) / GRID * height + pad
    yy, xx = torch.meshgrid(2*ys/full_h-1, 2*xs/full_w-1, indexing='ij')
    grid = torch.stack([xx, yy], -1)[None]
    cropped = F.grid_sample(weights[None], grid, mode='bilinear', align_corners=False)[0]
    return cropped * (width/full_w) * (height/full_h)
