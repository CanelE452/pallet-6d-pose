"""Prospective geometry-only V2 helpers. V1 code/fixtures stay immutable.

Pure generated-array preparation; no source readers, annotation reader or model.
These proposals are not adopted for actual source generation by this module.
"""
from pathlib import Path
import hashlib
import importlib.util

import numpy as np
from PIL import Image

V1_PATH = Path(__file__).resolve().parent.parent / 'same_q_pair_builder.py'
V1_SHA = '7450f610bd3f7567c2f87e9fe18197e5a064b64ba02c61f1a3e58d85ed3731fb'
if hashlib.sha256(V1_PATH.read_bytes()).hexdigest() != V1_SHA:
    raise ValueError('Immutable V1 dependency changed')
spec = importlib.util.spec_from_file_location('same_q_v1_bound', V1_PATH)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

CORE_MARGIN_PER_SIDE = 2
MAX_WHOLE_LONGEST_OVER_L = .80
MAX_UPSCALE = 4.
M96_RELATIVE_TOLERANCE = .01
M96_ABSOLUTE_NUMERICAL_TOLERANCE = 1e-9


def largest_opaque_rectangle(alpha):
    """Exact all-True axis-aligned rectangle, max area then lex xyxy tie.

    Stack O(HW). A mask hole is never filled, and the rectangle is a certificate
    inside the donor, not a replacement rectangular synthetic RGB occluder.
    """
    alpha = np.asarray(alpha, bool)
    if alpha.ndim != 2:
        raise ValueError('2D binary alpha required')
    h, w = alpha.shape; heights = np.zeros(w, np.int64)
    best_area = 0; best = None
    for y in range(h):
        heights = np.where(alpha[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            height = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > height:
                left, hh = stack.pop()
                rect = (left, y - hh + 1, x, y + 1)
                area = hh * (x - left)
                if area > best_area or (area == best_area and area > 0 and (best is None or rect < best)):
                    best_area, best = area, rect
                start = left
            if height > 0 and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return best


def support_scaled_cutout(donor_rgb, donor_visible, protected, L):
    """One scale from opaque core and protected geometry, never a sweep.

    Apply uniform aspect-preserving transform to WHOLE eroded donor cutout RGB
    and alpha. Out-of-bound scale or failed actual resized-alpha certification
    returns U; do not try a different donor/core or rescale after that failure.
    """
    yy, xx = np.nonzero(protected)
    if not len(xx) or not np.isfinite(L) or L <= 0:
        return None, {'reason': 'EMPTY_PROTECTED_OR_INVALID_L_U'}
    # This calls V1 erode1+tight whole-cutout at native scale, not the .40L rule.
    vis = np.asarray(donor_visible, bool)
    padded = np.pad(vis, 1)
    eroded = np.logical_and.reduce([padded[y:y + vis.shape[0], x:x + vis.shape[1]]
                                    for y in range(3) for x in range(3)])
    ey, ex = np.nonzero(eroded)
    if not len(ex):
        return None, {'reason': 'NO_ERODED_DONOR_U'}
    native_longest = max(int(ex.max() - ex.min() + 1), int(ey.max() - ey.min() + 1))
    base = v1.prepare_cutout(donor_rgb, vis, native_longest)
    rect = largest_opaque_rectangle(base['alpha'])
    if rect is None:
        return None, {'reason': 'NO_OPAQUE_RECTANGLE_U'}
    x0, y0, x1, y1 = rect
    required = np.array([xx.max() - xx.min() + 1, yy.max() - yy.min() + 1], np.float64)
    desired = required + 2 * CORE_MARGIN_PER_SIDE
    s = max(desired[0] / (x1 - x0), desired[1] / (y1 - y0), 9 / native_longest)
    bh, bw = base['alpha'].shape
    new_w = int(np.ceil(bw * s)); new_h = int(np.ceil(bh * s))
    facts = {'source_opaque_core_xyxy': list(rect), 'protected_native_wh': required.astype(int).tolist(),
             'desired_opaque_core_wh': desired.astype(int).tolist(), 'scale': float(s),
             'whole_resized_hw': [new_h, new_w], 'whole_limit_native': float(MAX_WHOLE_LONGEST_OVER_L * L),
             'no_scale_sweep_or_donor_replacement': True}
    if s > MAX_UPSCALE or max(new_h, new_w) > MAX_WHOLE_LONGEST_OVER_L * L:
        return None, dict(facts, reason='SCALE_OR_WHOLE_CUTOUT_BOUND_EXCEEDED_U')
    rgb = np.asarray(Image.fromarray(base['rgb']).resize((new_w, new_h), Image.Resampling.BILINEAR)).copy()
    alpha = np.asarray(Image.fromarray(base['alpha']).resize((new_w, new_h), Image.Resampling.NEAREST)).astype(bool)
    actual_core = largest_opaque_rectangle(alpha)
    if actual_core is None or actual_core[2] - actual_core[0] < desired[0] or actual_core[3] - actual_core[1] < desired[1]:
        return None, dict(facts, reason='ACTUAL_RESIZED_OPAQUE_CORE_TOO_SMALL_U')
    facts['actual_resized_opaque_core_xyxy'] = list(actual_core)
    facts['reason'] = 'OPAQUE_CORE_SCALE_GEOMETRY_READY_NOT_PIXEL_ADMISSION'
    return dict(base, rgb=rgb, alpha=alpha, resized_hw=[new_h, new_w],
                scale=float(s), opaque_core_xyxy=list(actual_core)), facts


def alpha_mass96(alpha, q, L):
    """Exact RGB96 phase, bilinear alpha coverage WITHOUT uint8 quantization."""
    h, w = alpha.shape; u = (np.arange(96, dtype=np.float64) + .5) / 96 - .5
    xx, yy = np.meshgrid(q[0] + u * L, q[1] + u * L)
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    dx, dy = xx - ix, yy - iy; out = np.zeros((96, 96), np.float64)
    for ax, ay, weight in [(0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)),
                           (0, 1, (1 - dx) * dy), (1, 1, dx * dy)]:
        x, y = ix + ax, iy + ay
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        out[inside] += np.asarray(alpha, np.float64)[y[inside], x[inside]] * weight[inside]
    valid = (xx >= -.5) & (xx < w - .5) & (yy >= -.5) & (yy < h - .5)
    out[~valid] = 0
    return float(out.sum())


def mass_certificate(row, cover_alpha, sham_alpha):
    h, w = row['raw_hw']; q = np.asarray(row['query_xy']); extent = row['L'] * 47.5 / 96
    yy, xx = np.mgrid[:h, :w]
    inside_crop = ((xx >= q[0] - extent) & (xx <= q[0] + extent)
                   & (yy >= q[1] - extent) & (yy <= q[1] + extent))
    total = [int(a.sum()) for a in (cover_alpha, sham_alpha)]
    local_native = [int((a & inside_crop).sum()) for a in (cover_alpha, sham_alpha)]
    local96 = [alpha_mass96(a, q, row['L']) for a in (cover_alpha, sham_alpha)]
    difference = abs(local96[0] - local96[1])
    tolerance = M96_ABSOLUTE_NUMERICAL_TOLERANCE + M96_RELATIVE_TOLERANCE * np.mean(local96)
    passed = (total[0] > 0 and total[0] == total[1]
              and local_native == total and min(local96) > 0 and difference <= tolerance)
    return bool(passed), {'native_global_alpha_cells': total, 'native_in_crop_alpha_cells': local_native,
                          'bilinear_RGB96_alpha_pixel_equivalents': local96,
                          'absolute_mass96_difference': float(difference), 'allowed_difference': float(tolerance),
                          'PASS': bool(passed)}


def core_aligned_pair(row, cutout, protected):
    """Finite9x25 pairs, first SHA-ranked feasible+mass-matched pair; no q move."""
    yy, xx = np.nonzero(protected); cx, cy = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
    x0, y0, x1, y1 = cutout['opaque_core_xyxy']
    core_centre = np.array([(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2])
    cover_xy = [tuple(np.floor(np.array([cx, cy]) - core_centre + [dx, dy] + .5).astype(int))
                for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]]
    cover_xy = sorted(set(cover_xy), key=lambda xy: v1.digest(row['input_id'], row['native_slot_logical_only'], 'cover', *xy))
    _, sham_xy = v1.candidate_positions(row, cutout, protected)
    attempts = 0
    for cxy in cover_xy:
        if not v1.fixed_crop_contains_cutout(row, cutout, cxy):
            continue
        ca = v1.translated_alpha(row['raw_hw'], cutout, cxy)
        if ca is None or not ca[protected].all():
            continue
        for sxy in sham_xy:
            if not v1.fixed_crop_contains_cutout(row, cutout, sxy):
                continue
            sa = v1.translated_alpha(row['raw_hw'], cutout, sxy)
            if sa is None or sa[protected].any():
                continue
            attempts += 1; passed, facts = mass_certificate(row, ca, sa)
            if passed:
                return ((cxy, ca), (sxy, sa)), dict(facts, geometry_pair_checks=attempts)
    return None, {'PASS': False, 'reason': 'NO_FIXED_PLACEMENT_PAIR_WITH_MATCHED_IN_CROP_MASS_U',
                  'geometry_pair_checks': attempts}
