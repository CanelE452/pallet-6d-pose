"""V90 input-only geometry; no files, GT, detector, or learned head imports.

The 80x80 sampling canvas is the frozen V60 storage convention, NOT the actual
P3 shape. Full-frame archives preserve the unpadded stride-8 feature plane.
"""
import numpy as np
import torch
from torch.nn import functional as F


def expected_affine(raw_hw, added_border=100):
    h, w = map(int, raw_hw)
    ch, cw = h + 2 * added_border, w + 2 * added_border
    gain = min(640 / ch, 640 / cw)
    rw, rh = round(cw * gain), round(ch * gain)
    ih, iw = rh + (640 - rh) % 32, rw + (640 - rw) % 32
    offset = np.array([round((iw - rw) / 2 - .1), round((ih - rh) / 2 - .1)], np.float64)
    return dict(canvas_shape=[ch, cw], input_shape=[ih, iw],
                p3_actual_hw=[ih // 8, iw // 8], gain=float(gain), offset=offset)


def native_grid_explicit(query_xy, side, raw_hw):
    q = np.asarray(query_xy, np.float64)
    side = np.broadcast_to(np.asarray(side, np.float64), (len(q),))
    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
        raise ValueError('Finite unchanged tile coordinates required')
    if not np.isfinite(side).all() or np.any(side <= 0):
        raise ValueError('Positive unchanged explicit tile L required')
    h, w = raw_hw
    a = (np.arange(9, dtype=np.float64) + .5) / 9 - .5
    xx, yy = np.meshgrid(a, a)
    grid = q[:, None, None, :] + np.stack([xx, yy], -1)[None] * side[:, None, None, None]
    good = ~(q == -1).all(-1)
    valid = (grid[..., 0] >= 0) & (grid[..., 0] < w) & (grid[..., 1] >= 0) & (grid[..., 1] < h) & good[:, None, None]
    return grid, valid


def sample_p3(p3_actual, query_xy, side, raw_hw, gain, offset, added_border):
    """CPU F16->F32, V60 float64 coords->F32 grid, then F16 ROI65.

    Temporarily recreate V60's right/bottom zero-padded 80 canvas in RAM so the
    normalization and rounding match the historical exact sampler. Padding is
    never stored as a real feature plane. Valid native-image sample supports
    must lie strictly inside the actual stride-8 plane.
    """
    plane = np.asarray(p3_actual)
    if plane.dtype != np.float16 or plane.ndim != 3 or plane.shape[0] != 64:
        raise ValueError('Unpadded FP16 P3[64,h3,w3] required')
    h3, w3 = plane.shape[-2:]
    if not (0 < h3 <= 80 and 0 < w3 <= 80) or not np.isfinite(plane).all():
        raise ValueError('Invalid finite stride-8 P3 plane')
    grid, valid = native_grid_explicit(query_xy, side, raw_hw)
    network = (grid + added_border) * gain + np.asarray(offset, np.float64)
    index = network / 8.
    # bilinear support of any factual native-image point cannot touch canvas pad.
    lower = np.floor(index)
    support_inside = (lower[..., 0] >= 0) & (lower[..., 1] >= 0) & (lower[..., 0] + 1 < w3) & (lower[..., 1] + 1 < h3)
    if not support_inside[valid].all():
        raise ValueError('A native-valid ROI samples outside the actual feature plane')
    norm = 2 * (index + .5) / 80 - 1
    tensor = torch.from_numpy(np.array(plane, copy=True))[None].float()
    tensor = F.pad(tensor, (0, 80 - w3, 0, 80 - h3))
    g = torch.as_tensor(norm, dtype=torch.float32)
    v = torch.as_tensor(valid[:, None], dtype=torch.float32)
    with torch.inference_mode():
        roi = F.grid_sample(tensor.expand(len(grid), -1, -1, -1), g,
                            mode='bilinear', padding_mode='zeros', align_corners=False)
        out = torch.cat([roi * v, v], 1).half().numpy()
    if out.shape != (len(grid), 65, 9, 9) or not np.isfinite(out).all():
        raise ValueError('Invalid sampled ROI65')
    return out, valid.reshape(len(grid), -1).any(1), valid


def crop(rgb,q,L,n):
    h,w=rgb.shape[:2];u=(np.arange(n,dtype=np.float64)+.5)/n-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);dx=xx-ix;dy=yy-iy
    acc=np.zeros((n,n,3),np.float64)
    for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x,y=ix+ax,iy+ay;inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        acc[inside]+=rgb[y[inside],x[inside]].astype(np.float64)*weight[inside,None]
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    out=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);out[~valid]=0
    return out.transpose(2,0,1).copy(),valid[None].copy()


def query_validity(q, raw_hw):
    q = np.asarray(q, np.float64)
    h, w = raw_hw
    finite = np.isfinite(q).all(1) & ~(q == -1).all(1)
    pixel = np.floor(np.where(finite[:, None], q, -1e8) + .5)
    continuous = finite & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    nearest = finite & (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    return continuous & nearest, continuous, nearest
