"""V237 pure NumPy reference math; no files, labels, model or random choices.

Prospective common-native-scene construction, not a V235 byte-replay claim.
The caller owns SOURCE admission, roles, native IDs and availability sidecars.
"""
import numpy as np


def query_extent(query8):
    """Literal V107 rule: finite non-(-1,-1) q, >=2, positive coordinate spans.

    Availability is deliberately not added to the historical V107 definition.
    """
    q = np.asarray(query8, dtype=np.float64)
    if q.shape != (8, 2):
        return None
    good = np.isfinite(q).all(1) & ~(q == [-1, -1]).all(1)
    if good.sum() < 2:
        return None
    lo, hi = q[good].min(0), q[good].max(0)
    return np.r_[lo, hi].tolist() if (hi > lo).all() else None


def context_grid(box, out_hw=(224, 224)):
    b = np.asarray(box, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all() or not (b[2:] > b[:2]).all():
        raise ValueError('Finite positive original q8 box required')
    h, w = map(int, out_hw)
    if min(h, w) < 1:
        raise ValueError('Positive output dimensions required')
    x = b[0] + (np.arange(w) + .5) * (b[2] - b[0]) / w
    y = b[1] + (np.arange(h) + .5) * (b[3] - b[1]) / h
    return np.stack(np.meshgrid(x, y), -1)


def local_grid(q, L):
    """Inverse of RAW96 address=(p-q)*96/L+47.5; no recentering."""
    q = np.asarray(q)
    scale = np.asarray(L)
    if q.dtype != np.float64 or q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('Finite original F64 q[2] required')
    if scale.dtype != np.float64 or scale.shape != () or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive original F64 L required')
    x = q[0] + (np.arange(96, dtype=np.float64) - 47.5) * scale / 96.
    y = q[1] + (np.arange(96, dtype=np.float64) - 47.5) * scale / 96.
    return np.stack(np.meshgrid(x, y), -1)


def sample_native_field(field, grid):
    """V107 F64 NW/NE/SW/SE zero-neighbor sum -> F32, no renormalization.

    RGB and alpha coverage use the same scene grid. Validity means all
    positive-weight native neighbors exist; it is not a semantic label.
    """
    field = np.asarray(field)
    grid = np.asarray(grid, dtype=np.float64)
    if field.ndim not in (2, 3) or min(field.shape[:2]) < 1 or grid.shape[-1] != 2:
        raise ValueError('Native HW[channels] field and [...,2] grid required')
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    h, w = field.shape[:2]
    x, y = grid[..., 0], grid[..., 1]
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    dx, dy = x - ix, y - iy
    out = np.zeros((*x.shape, field.shape[2]), np.float64)
    valid = np.ones(x.shape, bool)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                            (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        xx, yy = ix + ox, iy + oy
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        out += field[np.clip(yy, 0, h-1), np.clip(xx, 0, w-1)].astype(np.float64) * (weight*inside)[..., None]
        valid &= inside | (weight == 0)
    result = out.astype(np.float32)
    return (result[..., 0] if scalar else result), valid


def lift_raw96_layer(layer_rgb96, alpha96, q, L, native_hw):
    """A *new*, explicit rasterization of an old local-layer proposal.

    RAW96 cells are half-open [i-.5,i+.5); native pixel p maps to
    u=(p-q)*96/L+47.5. Nearest cell selects both RGB and binary alpha.
    This cannot be assumed to replay the old post-crop paste after resampling.
    """
    rgb, alpha = np.asarray(layer_rgb96), np.asarray(alpha96)
    local_grid(q, L)  # Shape/type checks only; no input edits.
    if rgb.dtype != np.uint8 or rgb.shape != (96, 96, 3) or alpha.dtype != bool or alpha.shape != (96, 96):
        raise ValueError('Layer uint8[96,96,3] and bool[96,96] required')
    h, w = map(int, native_hw)
    if min(h, w) < 1:
        raise ValueError('Positive native image dimensions required')
    x = (np.arange(w, dtype=np.float64) - q[0]) * 96. / L + 47.5
    y = (np.arange(h, dtype=np.float64) - q[1]) * 96. / L + 47.5
    xx, yy = np.meshgrid(x, y)
    inside = (xx >= -.5) & (xx < 95.5) & (yy >= -.5) & (yy < 95.5)
    ix = np.clip(np.floor(xx + .5).astype(np.int64), 0, 95)
    iy = np.clip(np.floor(yy + .5).astype(np.int64), 0, 95)
    return rgb[iy, ix].copy(), inside & alpha[iy, ix]


def reference_opacity(alpha_native, reference):
    """Teacher-only support diagnostics, never D/E or a deployment channel.

    Report continuous r and original floor(r+.5) 3x3 native four-tap supports
    separately. Insufficient/outside support remains unsupported, never E.
    """
    a = np.asarray(alpha_native)
    r = np.asarray(reference)
    if a.ndim != 2 or a.dtype != bool or r.dtype != np.float64 or r.shape != (2,) or not np.isfinite(r).all():
        raise ValueError('Native bool alpha and finite F64 reference required')
    nearest = np.floor(r + .5).astype(np.int64)
    samples = np.array([(nearest[0]+x, nearest[1]+y)
                        for y in (-1, 0, 1) for x in (-1, 0, 1)], np.float64)
    def check(points):
        corners = np.floor(points).astype(np.int64)
        taps = corners[:, None, :] + np.array([[0, 0], [1, 0], [0, 1], [1, 1]])[None]
        inside = bool(((taps[..., 0] >= 0) & (taps[..., 0] < a.shape[1]) &
                       (taps[..., 1] >= 0) & (taps[..., 1] < a.shape[0])).all())
        if not inside:
            return {'supported': False, 'all_covered': None, 'all_clear': None}
        selected = a[taps[..., 1], taps[..., 0]]
        return {'supported': True, 'all_covered': bool(selected.all()), 'all_clear': bool((~selected).all())}
    return {'continuous_reference': check(r[None]),
            'nearest_half_up_3x3': check(samples), 'native_gold_or_new_label': False}


def compose_native_then_sample(base_rgb, query8, q, L, layer_rgb96, alpha96, reference=None):
    """Minimal prospective API: one native raster -> both unchanged input grids.

    Returns F32 HWC views before any future uint8/normalization policy. Existing
    RAW96 byte parity, finite-part label admission and physical rendering are
    not certified by this reference implementation.
    """
    base = np.asarray(base_rgb)
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError('Original native HWC uint8 RGB required')
    box = query_extent(query8)
    if box is None:
        return {'supported': False, 'reason': 'ORIGINAL_Q8_BOX_UNSUPPORTED', 'new_label': None}
    local = local_grid(q, L)
    context = context_grid(box)
    donor_native, alpha_native = lift_raw96_layer(layer_rgb96, alpha96, q, L, base.shape[:2])
    scene = base.copy()
    scene[alpha_native] = donor_native[alpha_native]  # Opaque, no clean base at covered native pixels.
    local_rgb, local_valid = sample_native_field(scene, local)
    context_rgb, context_valid = sample_native_field(scene, context)
    local_coverage, _ = sample_native_field(alpha_native, local)
    context_coverage, _ = sample_native_field(alpha_native, context)
    return {'supported': True, 'native_composite': scene, 'native_alpha': alpha_native,
            'local_rgb_F32': local_rgb, 'context_rgb_F32': context_rgb,
            'local_valid': local_valid, 'context_valid': context_valid,
            'local_layer_coverage_diagnostic_only': local_coverage,
            'context_layer_coverage_diagnostic_only': context_coverage,
            'context_box_original_q8': box, 'original_query_xy': q.copy(),
            'original_L': np.float64(L), 'original_query8': np.asarray(query8).copy(),
            'teacher_opacity_diagnostic_only': None if reference is None else reference_opacity(alpha_native, reference),
            'existing_V235_pair_byte_replay_claim': False, 'new_label': None}
