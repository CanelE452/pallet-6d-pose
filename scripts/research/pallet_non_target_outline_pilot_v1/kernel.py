"""Fixed-candidate, two-direction outline censoring; no observation fitting.

Only a separately supplied, frozen binary non-target mask may mark preclip
hull parents. The saved target hull, finite segments, samples, residuals,
robust losses and normalizers remain unchanged. No surviving polygon is made.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

ATOL = 1e-9
VARIANTS = ('BASELINE_E', 'SUPPORT_AWARE_RERANK', 'MASS_MATCHED_CONTROL')


def _array(value, shape=None):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError('Original finite array with matching shape required')
    return array


def _clip(a, b, bounds):
    """The exact original Liang--Barsky arithmetic, not a new line detector."""
    direction = b-a
    lo, hi = 0., 1.
    for axis in (0, 1):
        if abs(direction[axis]) < 1e-14:
            if not bounds[axis][0] <= a[axis] <= bounds[axis][1]:
                return None
        else:
            first = (bounds[axis][0]-a[axis])/direction[axis]
            second = (bounds[axis][1]-a[axis])/direction[axis]
            lo = max(lo, min(first, second))
            hi = min(hi, max(first, second))
            if lo >= hi:
                return None
    return a+lo*direction, a+hi*direction


def lineage_for_outline(outline):
    """Recover original parent membership, requiring bit-exact saved geometry."""
    if not outline.get('supported'):
        return dict(supported=False, used_segments=[], sample_parent_segment_ids=[],
                    checks=dict(no_original_supported_outline=True))
    polygon = _array(outline['polygon_xy'])
    h, w = map(int, outline['image_hw'])
    cfg = outline['configuration']
    margin = float(cfg['frame_margin_px'])
    bounds = ((margin, w-1-margin), (margin, h-1-margin))
    spacing = float(cfg['observed_sample_spacing_px'])
    parents, sample_ids, samples, excluded = [], [], [], []
    for original_index, (a, b) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
        clipped = _clip(a, b, bounds)
        if clipped is None:
            excluded.append(dict(original_hull_edge_index=original_index, full_parent_xy=[a.tolist(), b.tolist()]))
            continue
        first, last = clipped
        count = max(1, int(np.ceil(float(np.linalg.norm(last-first))/spacing)))
        positions = first[None, :]+((np.arange(count)+.5)/count)[:, None]*(last-first)
        sid = len(parents)
        indices = list(range(len(samples), len(samples)+count))
        parents.append(dict(segment_id=sid, original_hull_edge_index=original_index,
                            full_parent_xy=[a.tolist(), b.tolist()], xy=[first.tolist(), last.tolist()],
                            sample_indices=indices))
        samples.extend(positions.tolist())
        sample_ids.extend([sid]*count)
    if not np.array_equal([p['xy'] for p in parents], outline['observed_boundary_segments_xy']):
        raise ValueError('Original preclip-parent to finite-segment parity failed')
    if not np.array_equal(samples, outline['observed_boundary_samples_xy']):
        raise ValueError('Original parent to observed-sample exact ordering failed')
    return dict(supported=True, polygon_xy=deepcopy(outline['polygon_xy']), used_segments=parents,
                excluded_frame_parents=excluded, sample_parent_segment_ids=sample_ids,
                image_hw=[h, w], checks=dict(finite_segment_order_exact=True, sample_order_exact=True,
                                           original_preclip_endpoints_preserved=True))


def parent_support(lineage, binary_mask=None, *, input_status='NO_ADMISSIBLE_NON_TARGET_EVIDENCE'):
    """Nearest pixel is floor(x+0.5); clipped endpoints are never queried."""
    mask = None
    if binary_mask is not None:
        mask = np.asarray(binary_mask)
        if mask.shape != tuple(lineage['image_hw']) or mask.dtype.kind not in 'buif':
            raise ValueError('Frozen binary mask on the original canvas required')
        if not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
            raise ValueError('Mask must already be binary; thresholding is a supplier decision')
    rows = []
    for segment in lineage['used_segments']:
        record = deepcopy(segment)
        coordinates, membership = [], []
        valid = True
        for xy in segment['full_parent_xy']:
            # Original hull vertices are nonnegative integer image coordinates.
            # Explicit half-up nearest convention is retained for the API fixture.
            pixel = np.floor(np.asarray(xy, dtype=float)+.5).astype(int)
            coordinates.append(pixel.tolist())
            if mask is None:
                membership.append(None)
            elif not (0 <= pixel[0] < mask.shape[1] and 0 <= pixel[1] < mask.shape[0]):
                membership.append(None)
                valid = False
            else:
                membership.append(bool(mask[pixel[1], pixel[0]]))
        dependent = bool(mask is not None and valid and any(membership))
        record.update(input_status=input_status, endpoint_pixels_xy=coordinates,
            endpoint_mask_membership=membership, censor=dependent,
            support_status=('NON_TARGET_DEPENDENT_PARENT' if dependent else
                            'UNRESOLVED_OR_NO_INPUT_KEEP_ORIGINAL' if mask is None or not valid else
                            'NO_NON_TARGET_ENDPOINT_SUPPORT_KEEP_ORIGINAL'),
            parent_membership_admissible=bool(mask is not None and valid),
            pixel_rule='floor(x+0.5), raw full-canvas coordinates; preclip endpoints only',
            all_parent_pixels_non_target_claimed=False, depth_order_claimed=False)
        rows.append(record)
    return rows


def nearest_original_segment(points, segments):
    """Original finite-segment projection and original-order argmin tie rule."""
    points = _array(points)
    segments = _array(segments)
    a, b = segments[:, 0], segments[:, 1]
    delta = b-a
    squared = np.sum(delta*delta, axis=1)
    alpha = np.sum((points[:, None, :]-a[None, :, :])*delta[None, :, :], axis=2)/np.maximum(squared, 1e-20)
    foot = a[None, :, :]+np.clip(alpha, 0, 1)[..., None]*delta[None, :, :]
    residuals = points[:, None, :]-foot
    nearest = np.argmin(np.sum(residuals*residuals, axis=2), axis=1)
    return nearest, residuals[np.arange(len(points)), nearest]


def _direction(row, keep, parent_ids):
    weights = _array(row['arc_or_sample_weights'])
    terms = _array(row['weighted_sample_contributions'])
    count = len(weights)
    if terms.shape != (count,) or np.asarray(keep).shape != (count,):
        raise ValueError('Original direction sample cardinality mismatch')
    if np.any(weights <= 0):
        raise ValueError('Original positive sampling mass required')
    normalizer = float(row['normalizer'])
    if not math.isclose(math.fsum(weights), normalizer, abs_tol=ATOL, rel_tol=0):
        raise ValueError('Original mass/normalizer parity failed')
    old = math.fsum(terms)
    if not math.isclose(old, float(row['weighted_contribution']), abs_tol=ATOL, rel_tol=0):
        raise ValueError('Original weighted sample sum parity failed')
    kept = math.fsum(float(v) for v, good in zip(terms, keep) if good)
    removed = math.fsum(float(v) for v, good in zip(terms, keep) if not good)
    kept_mass = math.fsum(float(v) for v, good in zip(weights, keep) if good)
    # A direction with no removed samples is copied exactly, not re-arithmetized.
    ratio = 1. if all(keep) else 0. if not any(keep) else kept_mass/normalizer
    return dict(direction=row['direction'], original_count=count, retained_count=sum(map(bool, keep)),
                original_mass=normalizer, retained_mass=kept_mass, retained_mass_ratio=ratio,
                original_weighted_cost=float(row['weighted_contribution']),
                support_weighted_cost=float(row['weighted_contribution']) if all(keep) else kept,
                censored_weighted_cost=removed,
                mass_control_weighted_cost=float(row['weighted_contribution'])*ratio,
                sample_parent_segment_ids=list(map(int, parent_ids)), retained_sample_flags=list(map(bool, keep)),
                original_normalizer_preserved=True, factor_and_half_already_in_saved_contributions=True)


def evaluate_outline(candidate, dense, outline, lineage, support):
    """Use saved per-sample losses at one original candidate; no pose is solved."""
    objective = float(candidate['original_record']['objective'])
    components = deepcopy(dense['weighted_components'])
    if not math.isclose(objective, float(dense['objective']), abs_tol=ATOL, rel_tol=0):
        raise ValueError('Candidate and saved fixed-cost objective disagree')
    rows = [r for r in dense['per_observation'] if r['loss_family'] == 'outline']
    old_outline = float(components['outline'])
    parent_flags = np.asarray([p['censor'] for p in support], dtype=bool)
    if len(parent_flags) != len(lineage['used_segments']):
        raise ValueError('Exactly one validity record per original finite parent required')
    directions = []
    if dense['outline_active']:
        if {r['direction'] for r in rows} != {'model_to_observed', 'observed_to_model'} or len(rows) != 2:
            raise ValueError('Both original outline directions are required')
        for row in rows:
            if row['direction'] == 'observed_to_model':
                if not np.array_equal(row['sample_xy'], outline['observed_boundary_samples_xy']):
                    raise ValueError('Observed sample coordinates/order changed')
                parents = np.asarray(lineage['sample_parent_segment_ids'], dtype=int)
            else:
                parents, residual = nearest_original_segment(row['sample_xy'], outline['observed_boundary_segments_xy'])
                if not np.array_equal(residual, row['residual_xy_px']):
                    # Arithmetic uses the exact original formula. Do not accept
                    # approximate nearest-neighbor identity as a parent proof.
                    raise ValueError('Original model residual/nearest finite-segment parity failed')
            keep = ~parent_flags[parents]
            directions.append(_direction(row, keep, parents))
        if not math.isclose(math.fsum(d['original_weighted_cost'] for d in directions), old_outline,
                            abs_tol=ATOL, rel_tol=0):
            raise ValueError('Original two-direction outline component parity failed')
    elif rows or old_outline != 0:
        raise ValueError('Inactive original outline must have no nonzero contribution')
    identity = not any(p['censor'] for p in support) or not dense['outline_active']
    removed = math.fsum(d['censored_weighted_cost'] for d in directions)
    new_outline = old_outline if identity else math.fsum(d['support_weighted_cost'] for d in directions)
    mass_outline = old_outline if identity else math.fsum(d['mass_control_weighted_cost'] for d in directions)
    # Exact no-input/no-parent identity is necessary for stable original ties.
    support_score = objective if identity else math.fsum((objective, -removed))
    control_score = objective if identity else math.fsum((objective, -old_outline, mass_outline))
    if not math.isclose(support_score, math.fsum((objective, -old_outline, new_outline)),
                        abs_tol=ATOL, rel_tol=0):
        raise ValueError('Old-minus-censored and component replacement disagree')
    return dict(candidate_id=candidate['candidate_id'], stage=candidate['stage'], family=candidate['family'],
                old_objective=objective, original_weighted_components=components,
                scores=dict(BASELINE_E=objective, SUPPORT_AWARE_RERANK=support_score,
                            MASS_MATCHED_CONTROL=control_score),
                original_outline_cost=old_outline, support_outline_cost=new_outline,
                mass_control_outline_cost=mass_outline, censored_outline_cost=removed,
                identity_shortcut=identity, directions=directions,
                original_hard_constraints_preserved=True, new_observations_created=0,
                new_model_samples_created=0, new_pose_fits=0)
