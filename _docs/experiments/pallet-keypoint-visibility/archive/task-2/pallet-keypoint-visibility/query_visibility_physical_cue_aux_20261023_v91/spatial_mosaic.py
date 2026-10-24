"""GT-free native-pixel evidence mosaic from every fixed grid tile.

Construct each frame's map before reference/label scoring. Blending uses factual
input support and fixed tent overlap weights, never cue FULL/GT/score selection.
The tile samples and native output pixels use pixel-center coordinates 0..W-1.
"""
import numpy as np
import torch
from torch.nn import functional as F


def make_frame_mosaic(tile_logits, tile_valid, queries, sides, raw_hw, device='cpu'):
    H, W = map(int, raw_hw)
    q = np.asarray(queries, np.float64); Ls = np.asarray(sides)
    assert q.shape == (len(tile_logits), 2) and Ls.shape == (len(tile_logits),)
    assert tuple(tile_logits.shape[1:]) == (9, 9) and tile_valid.shape == tile_logits.shape
    numerator = torch.zeros((H, W), dtype=torch.float32, device=device)
    denominator = torch.zeros_like(numerator)
    # V89 windows have integer centers/even sides. Reject unreviewed affine
    # changes rather than silently round subpixel or rescale source inputs.
    assert np.all(q == np.floor(q)) and all(int(L) == L and int(L) % 2 == 0 for L in Ls)
    for index, (query, side) in enumerate(zip(q, Ls)):
        L = int(side); x0, y0 = (query - L / 2).astype(int)
        x1, x2 = max(0, x0), min(W, x0 + L)
        y1, y2 = max(0, y0), min(H, y0 + L)
        if x1 >= x2 or y1 >= y2:
            continue
        xs = torch.arange(x1, x2, dtype=torch.float32, device=device)
        ys = torch.arange(y1, y2, dtype=torch.float32, device=device)
        ux = 2 * (xs - float(query[0])) / L
        uy = 2 * (ys - float(query[1])) / L
        gy, gx = torch.meshgrid(uy, ux, indexing='ij')
        grid = torch.stack((gx, gy), -1)[None]
        logit = torch.as_tensor(tile_logits[index], dtype=torch.float32, device=device)[None, None]
        valid = torch.as_tensor(tile_valid[index], dtype=torch.float32, device=device)[None, None]
        support = F.grid_sample(valid, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, 0]
        value = F.grid_sample(logit * valid, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, 0]
        tent = (1 - ux.abs()).clamp_min(0)[None] * (1 - uy.abs()).clamp_min(0)[:, None]
        numerator[y1:y2, x1:x2] += tent * value
        denominator[y1:y2, x1:x2] += tent * support
    available = denominator > 0
    result = numerator / denominator.clamp_min(1e-12)
    # Unknown pixels are retained as a separate mask; zero is just numeric-safe.
    result = torch.where(available, result, torch.zeros_like(result))
    return result.cpu().numpy(), available.cpu().numpy(), denominator.cpu().numpy()


def sample_native_region(logit_map, valid_map, locations_xy):
    """Score already completed native mosaic at GT-only locations, final stage."""
    H, W = logit_map.shape
    points = np.asarray(locations_xy, np.float64)
    assert points.ndim == 2 and points.shape[1] == 2
    finite = np.isfinite(points).all(1)
    in_frame = finite & (points[:, 0] >= 0) & (points[:, 0] <= W-1) & (points[:, 1] >= 0) & (points[:, 1] <= H-1)
    safe = np.where(np.isfinite(points), points, 0)
    normalized = 2 * (safe + .5) / np.array([W, H]) - 1
    grid = torch.as_tensor(normalized, dtype=torch.float32)[None, None]
    valid = torch.as_tensor(valid_map, dtype=torch.float32)[None, None]
    value = torch.as_tensor(logit_map, dtype=torch.float32)[None, None] * valid
    den = F.grid_sample(valid, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, 0, 0]
    num = F.grid_sample(value, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, 0, 0]
    available = in_frame & (den.numpy() > 0)
    return (num / den.clamp_min(1e-12)).numpy(), available
