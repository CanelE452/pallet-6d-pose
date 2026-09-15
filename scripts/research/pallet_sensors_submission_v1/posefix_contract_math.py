"""NumPy reference checks for a PoseFix-derived pallet9 adapter.

Not a network implementation, not training, and not a replacement for
TF/PyTorch operator and checkpoint parity. Layout: [B, H, W, K].
"""
from __future__ import annotations
import numpy as np


def gaussian_input_maps(points, valid, shape=(384, 288), sigma=9.0):
    """Match official input-map amplitude: exp(-distance²/(2*sigma²)) * 255."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    if p.ndim != 3 or p.shape[-1] != 2 or v.shape != p.shape[:2]:
        raise ValueError('Expected points [B,K,2] and valid [B,K]')
    h, w = map(int, shape)
    if h <= 0 or w <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid shape or sigma')
    if np.any(v & ~np.isfinite(p).all(-1)):
        raise ValueError('A valid input point must be finite')
    safe = np.where(v[..., None], p, 0.0)
    yy, xx = np.mgrid[:h, :w]
    dx = xx[None, :, :, None] - safe[:, None, None, :, 0]
    dy = yy[None, :, :, None] - safe[:, None, None, :, 1]
    maps = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * 255.0
    return np.where(v[:, None, None, :], maps, 0.0).astype(np.float32)


def coordinate_expectation(logits, input_shape=(384, 288)):
    """Zero-based expectation; official default input/output scale is 4 on both axes."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 4 or not np.isfinite(z).all():
        raise ValueError('Expected finite logits [B,H,W,K]')
    b, h, w, k = z.shape
    ih, iw = map(int, input_shape)
    if min(b, h, w, k, ih, iw) <= 0:
        raise ValueError('Empty dimensions')
    # The official extraction uses the height ratio for both coordinates.
    # For its locked 384x288 -> 96x72 geometry the ratios must match.
    if not np.isclose(ih / h, iw / w, atol=0.0, rtol=1e-12):
        raise ValueError('Non-isotropic scale needs an explicit adaptation, not silent parity')
    q = z.reshape(b, h * w, k)
    q = np.exp(q - q.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    yy, xx = np.mgrid[:h, :w]
    x = np.sum(q * xx.reshape(1, -1, 1), axis=1) * (iw / w)
    y = np.sum(q * yy.reshape(1, -1, 1), axis=1) * (ih / h)
    return np.stack((x, y), axis=-1)


def axis_aligned_crop_matrix(box, input_shape=(384, 288), expansion=1.25):
    """Source-adapter crop geometry; no GT box substitution."""
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('Expected a finite xyxy box')
    wh = box[2:] - box[:2]
    ih, iw = map(int, input_shape)
    if (wh <= 0).any() or min(ih, iw) <= 0 or not np.isfinite(expansion) or expansion <= 0:
        raise ValueError('Invalid box, shape, or expansion')
    center = (box[:2] + box[2:]) / 2.0
    w, h = wh
    ratio = iw / ih
    if w > h * ratio:
        h = w / ratio
    else:
        w = h * ratio
    w *= expansion
    h *= expansion
    return np.array([[iw/w, 0., iw*(.5-center[0]/w)],
                     [0., ih/h, ih*(.5-center[1]/h)],
                     [0., 0., 1.]], dtype=np.float64)


def transform_points(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if p.ndim < 2 or p.shape[-1] != 2 or m.shape != (3,3):
        raise ValueError('Invalid point/matrix shapes')
    if not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError('Nonfinite values require explicit masking')
    q = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1) @ m.T
    if np.any(np.abs(q[...,2]) <= 1e-12):
        raise ValueError('Degenerate homogeneous transformation')
    return q[...,:2] / q[...,2:3]


def self_test():
    p = np.array([[[12., 21.], [np.nan, np.nan]]])
    v = np.array([[True, False]])
    hm = gaussian_input_maps(p, v)
    assert hm.shape == (1, 384, 288, 2)
    assert hm[0, 21, 12, 0] == 255.0
    assert np.all(hm[...,1] == 0.) and np.isfinite(hm).all()
    assert np.isclose(hm[0,21,21,0], 255*np.exp(-.5), rtol=1e-6)
    flat = np.zeros((1,96,72,2))
    assert np.allclose(coordinate_expectation(flat), [[[142.,190.],[142.,190.]]])
    peaked = np.full((1,96,72,1), -1000.)
    peaked[0,20,10,0] = 0.
    assert np.array_equal(coordinate_expectation(peaked), np.array([[[40.,80.]]]))
    try:
        coordinate_expectation(np.zeros((1,96,70,1)))
    except ValueError:
        pass
    else:
        raise AssertionError('Unequal scale was accepted as official parity')
    rng = np.random.default_rng(20260915)
    worst = 0.
    for _ in range(100):
        xy = rng.uniform(-100, 500, size=2)
        wh = rng.uniform(10, 500, size=2)
        m = axis_aligned_crop_matrix(np.r_[xy, xy+wh])
        pts = rng.uniform(-200, 1000, size=(9,2))
        back = transform_points(transform_points(pts, m), np.linalg.inv(m))
        worst = max(worst, float(np.max(np.abs(back-pts))))
    assert worst < 1e-9
    return dict(scope='NumPy generated examples only; no network, no checkpoint, no image evaluation',
                input_map_peak=255.0, invalid_map_zero=True, expectation_zero_based=True,
                nonisotropic_scale_rejected=True, affine_cases=100, max_roundtrip_px=worst)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))

