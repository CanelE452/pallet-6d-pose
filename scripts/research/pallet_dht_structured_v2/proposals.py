"""Fixed GT-free whole-layout proposals in original-image pixels.

This module generates candidates, never chooses one using labels or scores.
The downstream learned verifier must compare complete same-ID layouts.
"""
from __future__ import annotations

import numpy as np


YAW90 = np.array([1, 5, 6, 2, 0, 4, 7, 3], dtype=np.int64)
C4 = np.stack([np.arange(8), YAW90, YAW90[YAW90], YAW90[YAW90[YAW90]]])
DEFAULT_CONFIG = dict(
    schema='pallet_dht_structured_proposals_v1',
    stored_layouts=64,
    n_generated_before_dedup=54,
    c4_seed_permutations=C4.tolist(),
    full_snap_alphas=[.25, .5, 1.],
    single_corner_snap_alpha=.5,
    translation_diagonal_fraction=.005,
    translation_order=[[1, 0], [-1, 0], [0, 1], [0, -1]],
    bbox_scale_factors=[.98, 1.02],
    scale_center='axis_aligned_bbox_of_eight_original_predicted_corners',
    snap_pool='same_semantic_corner_cached_intersection_slots_1_through_48',
    snap_metric='Euclidean_raw_pixel_distance_to_each_C4_seed_corner',
    snap_tie='lowest_cached_intersection_slot',
    no_intersection='keep_that_seed_corner',
    seed_policy='first_four_C4_seeds_always_retained_even_if_coordinates_duplicate',
    other_dedup='stable_exact_coordinate_equality_against_all_previous_retained_layouts',
    padding='repeat_original_coordinates_with_valid_false',
    centroid='copy_original_index8',
    missing_corner='only_original_slot0_valid',
    output_ids='camera_dynamic_0123_v4_same_ID_candidate_labels',
    hard_semantic_filter=False,
    DLT_or_PnP_used=False,
    GT_used=False,
    old_global_bank_used=False,
)


def build_proposals(points9, intersections_xy, intersection_valid, raw_diagonal,
                    source_tag='', point_valid=None):
    """Return 64 whole-layout slots from predicted coordinates alone.

    Args:
        points9: [9,2], original-image baseline prediction. Index8 is copied.
        intersections_xy: [8,49,2], cached same-role intersections; slot0 is
            the old original-point fallback and is deliberately excluded.
        intersection_valid: [8,49], original inference validity, never GT.
        raw_diagonal: hypot(original image width,height), positive pixels.
        source_tag: optional provenance text, never influences proposals.
        point_valid: optional [9] prediction mask, never annotation mask.

    Each C4 seed is snapped using role pool i for its *new* semantic corner i,
    not pool permutation[i]. Missing baseline corners retain only the exact
    original layout. A missing centroid does not suppress valid 8-corner work.
    Candidate selection and scoring are deliberately outside this function.
    """
    p = np.asarray(points9, dtype=np.float64)
    xy = np.asarray(intersections_xy, dtype=np.float64)
    iv = np.asarray(intersection_valid, dtype=bool)
    if p.shape != (9, 2) or xy.shape != (8, 49, 2) or iv.shape != (8, 49):
        raise ValueError('Expected points[9,2], intersections[8,49,2], valid[8,49]')
    if not np.isfinite(raw_diagonal) or raw_diagonal <= 0:
        raise ValueError('raw_diagonal must be positive finite original-image pixels')
    if not isinstance(source_tag, str):
        raise TypeError('source_tag must be provenance text')
    known = np.isfinite(p).all(axis=1)
    if point_valid is not None:
        pv = np.asarray(point_valid, dtype=bool)
        if pv.shape != (9,):
            raise ValueError('point_valid must have shape [9]')
        known &= pv
    layouts = np.broadcast_to(p, (64, 9, 2)).copy()
    valid = np.zeros(64, dtype=bool)
    kind = ['padding']*64
    candidate_index = np.full((64, 8), -1, dtype=np.int64)
    c4_quarters = np.full(64, -1, dtype=np.int64)
    snap_alpha = np.zeros((64, 8), dtype=np.float64)
    usable = iv & np.isfinite(xy).all(axis=-1)
    usable[:, 0] = False
    diagnostics = dict(
        uses_gt=False, source_tag_used_for_geometry=False,
        raw_diagonal=float(raw_diagonal),
        corner_prediction_valid=known[:8].tolist(),
        usable_intersections_per_corner=usable.sum(1).tolist(),
        excluded_nonfinite_intersections=int((iv & ~np.isfinite(xy).all(-1)).sum()),
        n_generated_before_dedup=0, n_deduplicated=0, n_valid=0,
        c4_seed_slots=[0, 1, 2, 3],
        fallback_reason=None,
    )
    if not known[:8].all():
        valid[0] = True
        kind[0] = 'baseline_missing_corner_fallback'
        c4_quarters[0] = 0
        diagnostics.update(n_generated_before_dedup=1, n_valid=1,
                           c4_seed_slots=[0], fallback_reason='missing_baseline_corner')
        return dict(layouts=layouts, valid=valid, kind=kind,
                    candidate_index=candidate_index, c4_quarters=c4_quarters,
                    snap_alpha=snap_alpha, source_tag=source_tag, diagnostics=diagnostics)

    used = 0
    def append(corners, label, quarter, indices=None, alphas=None, force=False):
        nonlocal used
        diagnostics['n_generated_before_dedup'] += 1
        if not np.isfinite(corners).all():
            raise ValueError('Proposal arithmetic produced nonfinite coordinates')
        if not force and any(np.array_equal(corners, layouts[j, :8]) for j in range(used)):
            diagnostics['n_deduplicated'] += 1
            return
        if used >= 64:
            raise RuntimeError('Fixed proposal bank exceeded 64 slots')
        layouts[used, :8] = corners
        valid[used] = True
        kind[used] = label
        c4_quarters[used] = quarter
        if indices is not None:
            candidate_index[used] = indices
        if alphas is not None:
            snap_alpha[used] = alphas
        used += 1

    seeds = p[:8][C4]
    nearest = seeds.copy()
    indices = np.full((4, 8), -1, dtype=np.int64)
    for quarter in range(4):
        append(seeds[quarter], f'baseline_c4_{quarter}', quarter, force=True)
        for corner in range(8):
            pool = np.flatnonzero(usable[corner])
            if len(pool):
                delta = xy[corner, pool]-seeds[quarter, corner]
                distance = np.hypot(delta[:, 0], delta[:, 1])
                k = int(pool[np.argmin(distance)])
                nearest[quarter, corner] = xy[corner, k]
                indices[quarter, corner] = k
    # Frozen order: all whole snaps, then all single-corner half snaps.
    for quarter in range(4):
        for alpha in (.25, .5, 1.):
            q = seeds[quarter] + alpha*(nearest[quarter]-seeds[quarter])
            append(q, f'snap_all_c4_{quarter}_alpha_{alpha:g}', quarter,
                   indices[quarter], np.where(indices[quarter] >= 0, alpha, 0.))
    for quarter in range(4):
        for corner in range(8):
            q = seeds[quarter].copy()
            q[corner] += .5*(nearest[quarter, corner]-q[corner])
            ci = np.full(8, -1, dtype=np.int64)
            ci[corner] = indices[quarter, corner]
            alpha = np.zeros(8)
            alpha[corner] = .5 if ci[corner] >= 0 else 0.
            append(q, f'snap_corner_c4_{quarter}_corner_{corner}_alpha_0.5', quarter, ci, alpha)
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        delta = np.zeros(2)
        delta[axis] = sign*.005*raw_diagonal
        append(p[:8]+delta, f'translate_{"xy"[axis]}_{sign:+d}', 0)
    center = (p[:8].min(0)+p[:8].max(0))*.5
    for scale in (.98, 1.02):
        append(center+scale*(p[:8]-center), f'scale_corner_bbox_{scale:g}', 0)
    if diagnostics['n_generated_before_dedup'] != 54:
        raise AssertionError('Proposal enumeration drifted from fixed 54-item design')
    diagnostics.update(n_valid=int(valid.sum()),
                       scale_center_xy=center.tolist(),
                       nearest_intersection_index_by_c4=indices.tolist())
    return dict(layouts=layouts, valid=valid, kind=kind,
                candidate_index=candidate_index, c4_quarters=c4_quarters,
                snap_alpha=snap_alpha, source_tag=source_tag, diagnostics=diagnostics)
