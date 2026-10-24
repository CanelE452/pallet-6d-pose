"""Pure NumPy reference-local support teacher. No file/model/solver I/O.

VISIBLE/COVERED describe expected target silhouette support near fixed r.
They are not human semantic D/E/S, point-on-mesh or query-pixel labels.
GT r, masks, self flags and returned diagnostics must never enter a network.
"""
import numpy as np

VISIBLE, COVERED, UNKNOWN = 1, 0, -1


def _mask(value, name):
    a = np.asarray(value)
    if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(name + ' must be a finite nonnegative 2D target mask')
    if not np.isin(a, [0, 1, 255]).all():
        raise ValueError(name + ' must be a binary mask encoded as bool, 0/1 or 0/255')
    return a > 0


def _point_valid(points, h, w):
    finite = np.isfinite(points).all(1)
    index = np.full(points.shape, -1, np.int64)
    index[finite] = np.floor(points[finite] + .5).astype(np.int64)
    valid = finite & (points[:, 0] >= 0) & (points[:, 0] < w)
    valid &= (points[:, 1] >= 0) & (points[:, 1] < h)
    nearest = finite & (index[:, 0] >= 0) & (index[:, 0] < w)
    nearest &= (index[:, 1] >= 0) & (index[:, 1] < h)
    return valid & nearest, index


def reference_local_support(M0, M4, r8, *, self_proxy8=None, q8=None, local_side8=None):
    """Return a typed GT-only teacher/diagnostic dict for one paired instance.

    Strong complete3x3 M0+M4 -> VISIBLE, complete3x3 M0+zeroM4 -> COVERED.
    Otherwise use disk rho=max(4,.05*max(ptp(r8))), sigma=rho/2, peak1
    Gaussian. Adequacy is >=4 M0 pixels and A>=1 pixel-equivalent; V/A>=.9
    is VISIBLE, <=.1 COVERED, otherwise U. Self never changes these labels.

    Pixel centers are integer OpenCV coordinates. r is never clipped/moved.
    Invalid r (continuous OR nearest bounds) has U even if disk intersects
    the image. A valid boundary r may use the clipped observed footprint;
    near_boundary and valid footprint fraction are explicit. They are not
    extra label/loss vetoes. Corpus/fit admission is a later frozen policy.
    """
    full, visible = _mask(M0, 'M0'), _mask(M4, 'M4')
    if full.shape != visible.shape or np.any(visible & ~full):
        raise ValueError('Paired same-camera masks require M4 subset of M0')
    r = np.asarray(r8, np.float64)
    if r.shape != (8, 2):
        raise ValueError('Exactly eight stored dynamic native references required')
    h, w = full.shape
    gt_valid, center_xy = _point_valid(r, h, w)
    layout_valid = bool(np.isfinite(r).all())
    rho = max(4., .05 * float(np.ptp(r, axis=0).max())) if layout_valid else np.nan
    sigma = rho / 2.
    label = np.full(8, UNKNOWN, np.int8)
    state = np.full(8, 'U', dtype='<U7')
    rule = np.full(8, 'U_INVALID_REFERENCE_LAYOUT', dtype='<U48')
    center_M0 = np.full(8, -1, np.int8); center_M4 = center_M0.copy()
    strong_window = np.zeros(8, bool)
    strong_full = np.zeros(8, bool)
    strong_visible = np.zeros(8, bool); strong_covered = np.zeros(8, bool)
    count_A = np.zeros(8, np.int32); count_V = count_A.copy()
    mass_A = np.zeros(8, np.float64); mass_V = mass_A.copy()
    ratio = np.full(8, np.nan, np.float64)
    valid_count = count_A.copy(); total_count = count_A.copy()
    coverage = np.zeros(8, np.float64); near_boundary = np.ones(8, bool)
    adequate = np.zeros(8, bool)
    for k in range(8):
        if not layout_valid:
            continue
        x, y = center_xy[k]
        if gt_valid[k]:
            center_M0[k] = full[y, x]; center_M4[k] = visible[y, x]
        strong_window[k] = bool(gt_valid[k] and 1 <= x < w - 1 and 1 <= y < h - 1)
        if strong_window[k]:
            f = full[y - 1:y + 2, x - 1:x + 2]
            v = visible[y - 1:y + 2, x - 1:x + 2]
            strong_full[k] = bool(f.all())
            strong_visible[k] = bool(f.all() and v.all())
            strong_covered[k] = bool(f.all() and not v.any())
        xs = np.arange(int(np.floor(r[k, 0] - rho)), int(np.ceil(r[k, 0] + rho)) + 1)
        ys = np.arange(int(np.floor(r[k, 1] - rho)), int(np.ceil(r[k, 1] + rho)) + 1)
        xx, yy = np.meshgrid(xs, ys)
        d2 = (xx - r[k, 0]) ** 2 + (yy - r[k, 1]) ** 2
        disk = d2 <= rho ** 2
        valid_disk = disk & (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        total_count[k] = int(disk.sum()); valid_count[k] = int(valid_disk.sum())
        coverage[k] = valid_count[k] / total_count[k] if total_count[k] else 0.
        near_boundary[k] = not strong_window[k] or valid_count[k] < total_count[k]
        px, py = xx[valid_disk], yy[valid_disk]
        weights = np.exp(-d2[valid_disk] / (2 * sigma ** 2))
        expected = full[py, px]; seen = visible[py, px]
        count_A[k] = int(expected.sum()); count_V[k] = int(seen.sum())
        mass_A[k] = float(weights[expected].sum()); mass_V[k] = float(weights[seen].sum())
        if mass_A[k] > 0:
            ratio[k] = mass_V[k] / mass_A[k]
        adequate[k] = bool(count_A[k] >= 4 and mass_A[k] >= 1.)
        if not gt_valid[k]:
            rule[k] = 'U_REFERENCE_INVALID'
        elif strong_visible[k]:
            label[k] = VISIBLE; rule[k] = 'STRONG_3X3_VISIBLE'
        elif strong_covered[k]:
            label[k] = COVERED; rule[k] = 'STRONG_3X3_COVERED'
        elif not adequate[k]:
            rule[k] = 'U_INSUFFICIENT_EXPECTED_SUPPORT'
        elif ratio[k] >= .9:
            label[k] = VISIBLE; rule[k] = 'FOOTPRINT_VISIBLE'
        elif ratio[k] <= .1:
            label[k] = COVERED; rule[k] = 'FOOTPRINT_COVERED'
        else:
            rule[k] = 'U_MIXED_FOOTPRINT_SUPPORT'
    state[label == VISIBLE] = 'VISIBLE'; state[label == COVERED] = 'COVERED'
    if self_proxy8 is not None:
        s = np.asarray(self_proxy8)
        if s.shape != (8,):
            raise ValueError('self_proxy8 must have eight flags')
        self_separate = s.copy()
    else:
        self_separate = None
    query_diagnostic = None
    if q8 is not None:
        q = np.asarray(q8, np.float64)
        if q.shape != (8, 2):
            raise ValueError('q8 must have eight unchanged input queries')
        q_valid, qi = _point_valid(q, h, w)
        qM0 = np.full(8, -1, np.int8); qM4 = qM0.copy()
        qM0[q_valid] = full[qi[q_valid, 1], qi[q_valid, 0]]
        qM4[q_valid] = visible[qi[q_valid, 1], qi[q_valid, 0]]
        query_diagnostic = {'valid': q_valid, 'center_M0': qM0, 'center_M4': qM4,
                            'query_to_reference_offset': r - q}
        if local_side8 is not None:
            side = np.broadcast_to(np.asarray(local_side8, float), (8,))
            if not np.isfinite(side).all() or not (side > 0).all():
                raise ValueError('local side must be finite positive scalar/eight values')
            offset = (r - q) / side[:, None]
            query_diagnostic['query_to_reference_offset_over_L'] = offset
            query_diagnostic['reference_in_local_square'] = (
                np.isfinite(offset).all(1) & (offset >= -.5).all(1) & (offset < .5).all(1))
    return {
        'label': label, 'state': state, 'rule': rule,
        'rho_px': rho, 'sigma_px': sigma, 'reference_layout_finite': layout_valid,
        'GTref_valid_xy': gt_valid, 'reference_nearest_xy': center_xy,
        'center_M0': center_M0, 'center_M4': center_M4,
        'strong_3x3_window_valid': strong_window, 'strong_3x3_full_M0': strong_full,
        'strong_3x3_visible': strong_visible, 'strong_3x3_covered': strong_covered,
        'expected_support_pixels_A': count_A, 'visible_support_pixels_V': count_V,
        'weighted_expected_mass_A': mass_A, 'weighted_visible_mass_V': mass_V,
        'visible_over_expected_ratio': ratio, 'footprint_support_adequate': adequate,
        'valid_footprint_pixels': valid_count, 'full_disk_lattice_pixels': total_count,
        'footprint_valid_fraction': coverage, 'near_boundary': near_boundary,
        'self_proxy8_separate': self_separate, 'query_diagnostics_separate': query_diagnostic,
    }
