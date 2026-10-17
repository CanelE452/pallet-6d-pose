"""One fixed RGB-visible support score at unchanged retained poses.

The observation remains a union of open polylines. Only the *model* is
convex-hulled, following the original whole-outline observer. This module
contains no detector, mask operation, correspondence fitting, or pose fit.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.spatial import ConvexHull

ATOL = 1e-9
VARIANTS = ('BASELINE_E', 'HULL_ONEWAY', 'HULL_ONEWAY_MASS', 'VISIBLE_SUPPORT')


def _array(value, shape=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError('Finite array of the declared shape required')
    return result


def validate_polylines(polylines, image_hw):
    """Technical validation only; the author, rather than this code, sees RGB.

    Exact repeated/reversed raw segments and collinear overlapping lengths
    are rejected before scoring. No new vertex, closure, or dedup radius is
    introduced. Distinct chains are never joined by their endpoints.
    """
    h, w = map(int, image_hw)
    segments, ids = [], set()
    for polyline in polylines:
        line_id = polyline['line_id']
        if not isinstance(line_id, str) or not line_id or line_id in ids:
            raise ValueError('Unique nonempty polyline IDs required')
        ids.add(line_id)
        points = _array(polyline['points_xy'])
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError('Each open polyline requires at least two XY vertices')
        if np.any(points < 0) or np.any(points[:, 0] > w-1) or np.any(points[:, 1] > h-1):
            raise ValueError('Raw full-image coordinates must be within the original canvas')
        if polyline.get('closed') is True or np.array_equal(points[0], points[-1]):
            raise ValueError('This pilot takes open chains; implicit polygon closure is forbidden')
        for index, (first, last) in enumerate(zip(points[:-1], points[1:])):
            delta = last-first
            squared = float(delta@delta)
            if squared == 0:
                raise ValueError('Zero-length visible segment')
            for old in segments:
                a, b = np.asarray(old['xy'])
                old_delta = b-a
                # Exact arithmetic test for duplicate raw support. No spatial
                # tolerance changes which distinct image feature is accepted.
                cross = lambda x, y: float(x[0]*y[1]-x[1]*y[0])
                if cross(delta, old_delta) == 0 and cross(delta, a-first) == 0:
                    endpoints = sorted((float((a-first)@delta/squared), float((b-first)@delta/squared)))
                    if max(0., endpoints[0]) < min(1., endpoints[1]):
                        raise ValueError('Repeated/overlapping collinear raw support must be resolved before freeze')
            segments.append(dict(line_id=line_id, segment_index=index,
                                 xy=[first.tolist(), last.tolist()]))
    return segments


def sample_visible(polylines, image_hw, spacing):
    """Length-weighted midpoint quadrature, with no samples in omitted gaps."""
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Original positive observed spacing required')
    segments = validate_polylines(polylines, image_hw)
    points, masses, provenance, lengths = [], [], [], []
    for segment in segments:
        first, last = np.asarray(segment['xy'])
        length = float(np.linalg.norm(last-first))
        count = max(1, int(np.ceil(length/spacing)))
        mass = (length/spacing)/count
        points.extend((first[None, :]+((np.arange(count)+.5)/count)[:, None]*(last-first)).tolist())
        masses.extend([mass]*count)
        provenance.extend(dict(line_id=segment['line_id'], segment_index=segment['segment_index'],
                               within_segment_sample_index=i) for i in range(count))
        lengths.append(length)
    total_length = math.fsum(lengths)
    total_mass = math.fsum(masses)
    if not math.isclose(total_mass, total_length/spacing, abs_tol=ATOL, rel_tol=0):
        raise ValueError('Midpoint mass does not preserve observed length')
    return dict(polyline_count=len(polylines), segment_count=len(segments), sample_count=len(points),
                total_length_px=total_length, original_sample_spacing_px=spacing,
                visible_mass_Z_E=total_mass, sample_xy=points, sample_masses=masses,
                sample_sources=provenance, segments=segments,
                endpoints_used_as_keypoints=False, gaps_joined=False,
                samples_are_independent_measurements=False)


def project_model(candidate, K):
    """Original project_native arithmetic using this candidate's native basis.

    R already transforms native XYZ. Q is preserved in the candidate and its
    physical pose, and must not be applied a second time during projection.
    """
    xyz = _array(candidate['native_xyz'], (8, 3))
    R = _array(candidate['R'], (3, 3))
    t = _array(candidate['t'], (3,))
    K = _array(K, (3, 3))
    camera = xyz@R.T+t
    homogeneous = camera@K.T
    depth = camera[:, 2]
    if np.min(depth) <= 1e-8:
        raise ValueError('Original retained candidate has invalid positive depth')
    denominator = homogeneous[:, 2]
    safe = np.where(np.abs(denominator) < 1e-10,
                    np.where(denominator < 0, -1e-10, 1e-10), denominator)
    projected = homogeneous[:, :2]/safe[:, None]
    polygon = projected[ConvexHull(projected).vertices]
    segments = np.stack((polygon, np.roll(polygon, -1, axis=0)), axis=1)
    return projected, polygon, segments


def finite_distances(points, segments):
    """Same original finite-segment projection; stable segment-order argmin."""
    points, segments = _array(points), _array(segments)
    if points.ndim != 2 or points.shape[1] != 2 or segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError('XY points and finite segment endpoints required')
    first, last = segments[:, 0], segments[:, 1]
    delta = last-first
    squared = np.sum(delta*delta, axis=1)
    alpha = np.sum((points[:, None, :]-first[None, :, :])*delta[None, :, :], axis=2)/np.maximum(squared, 1e-20)
    feet = first[None, :, :]+np.clip(alpha, 0, 1)[..., None]*delta[None, :, :]
    residuals = points[:, None, :]-feet
    nearest = np.argmin(np.sum(residuals*residuals, axis=2), axis=1)
    residual = residuals[np.arange(len(points)), nearest]
    return np.linalg.norm(residual, axis=1), residual, nearest


def robust_losses(distances, sigma, delta):
    radius = _array(distances)/float(sigma)
    delta = float(delta)
    if sigma <= 0 or delta <= 0 or not np.isfinite((sigma, delta)).all():
        raise ValueError('Original positive robust scales required')
    return radius, delta**2*(np.sqrt(1+(radius/delta)**2)-1)


def _saved_directions(dense, outline):
    rows = [r for r in dense['per_observation'] if r['loss_family'] == 'outline']
    if not dense['outline_active']:
        if rows or dense['weighted_components']['outline'] != 0:
            raise ValueError('Inactive outline contains nonzero costs')
        return {}
    directions = {r['direction']: r for r in rows}
    if len(rows) != 2 or set(directions) != {'observed_to_model', 'model_to_observed'}:
        raise ValueError('Original pair of weighted outline directions required')
    cfg = outline['configuration']
    for row in rows:
        weights = _array(row['arc_or_sample_weights'])
        terms = _array(row['weighted_sample_contributions'])
        if np.any(weights <= 0) or not math.isclose(math.fsum(weights), row['normalizer'], abs_tol=ATOL, rel_tol=0):
            raise ValueError('Original direction mass mismatch')
        if not math.isclose(math.fsum(terms), row['weighted_contribution'], abs_tol=ATOL, rel_tol=0):
            raise ValueError('Original weighted direction sum mismatch')
        if row['factor_weight'] != cfg['outline_factor_weight'] or row['direction_weight'] != .5:
            raise ValueError('Original lambda/2 configuration mismatch')
    observed = directions['observed_to_model']
    if not np.array_equal(observed['sample_xy'], outline['observed_boundary_samples_xy']):
        raise ValueError('Original observed sample order/canvas changed')
    if not math.isclose(math.fsum(row['weighted_contribution'] for row in rows),
                        dense['weighted_components']['outline'], abs_tol=ATOL, rel_tol=0):
        raise ValueError('Original outline component and directions differ')
    return directions


def evaluate_candidate(candidate, outline, K, visible=None, input_status='NO_INPUT_IDENTITY_TEST'):
    """Four costs, all replacing only the old outline and never adding to it."""
    dense = candidate['dense']
    objective = float(candidate['original_record']['objective'])
    components = deepcopy(dense['weighted_components'])
    if not math.isclose(objective, dense['objective'], abs_tol=ATOL, rel_tol=0):
        raise ValueError('Original candidate and sealed dense objective differ')
    non_outline = math.fsum(components[name] for name in ('full_BI', 'partial_normal', 'prior'))
    if not math.isclose(math.fsum((non_outline, components['outline'])), objective, abs_tol=ATOL, rel_tol=0):
        raise ValueError('Original full/partial/prior/outline decomposition mismatch')
    directions = _saved_directions(dense, outline)
    active = bool(visible is not None and visible['sample_count'] and directions)
    base = dict(candidate_id=candidate['candidate_id'], stage=candidate['stage'], family=candidate['family'],
                old_objective=objective, original_weighted_components=components,
                non_outline_cost=non_outline, input_status=input_status, treatment_active=active,
                identity_shortcut=not active, original_hard_constraints_preserved=True,
                new_pose_fits=0, new_candidate_calls=0, new_model_forward_calls=0,
                original_full_partial_prior_preserved=True,
                lambda_half_not_applied_twice=True, observed_hull_created=False,
                model_to_visible_reverse_term_created=False)
    old_observed = float(directions['observed_to_model']['weighted_contribution']) if directions else 0.
    old_model = float(directions['model_to_observed']['weighted_contribution']) if directions else 0.
    base.update(original_observed_weighted_cost=old_observed, original_model_weighted_cost=old_model,
                original_outline_cost=float(components['outline']))
    if not active:
        return dict(**base, scores={v: objective for v in VARIANTS},
                    replacement_outline_costs={v: float(components['outline']) for v in VARIANTS},
                    visible_evidence=None, old_observed_mass_Z_H=(directions['observed_to_model']['normalizer'] if directions else None),
                    visible_mass_Z_E=None, r=None)
    cfg = outline['configuration']
    points, polygon, model_segments = project_model(candidate, K)
    # Establish that using the saved R directly has the same fixed native
    # projection contract as the original dense observer (which converted
    # through a rotation vector internally).
    old = directions['observed_to_model']
    old_distance, old_residual, _ = finite_distances(old['sample_xy'], model_segments)
    if not np.allclose(old_residual, old['residual_xy_px'], atol=ATOL, rtol=0):
        raise ValueError('Native projected finite hull differs from original dense residuals')
    _, old_rho = robust_losses(old_distance, cfg['normal_sigma_px'], cfg['huber_delta'])
    if not np.allclose(old_rho, old['robust_losses'], atol=ATOL, rtol=0):
        raise ValueError('Original outline robust scale differs')
    if visible['original_sample_spacing_px'] != cfg['observed_sample_spacing_px']:
        raise ValueError('New sampling spacing must equal this original frame/stage/family configuration')
    Z_H = float(old['normalizer'])
    Z_E = float(visible['visible_mass_Z_E'])
    r = min(1., Z_E/Z_H)
    distances, residuals, nearest = finite_distances(visible['sample_xy'], model_segments)
    radius, rho = robust_losses(distances, cfg['normal_sigma_px'], cfg['huber_delta'])
    masses = np.asarray(visible['sample_masses'], dtype=float)
    raw_robust_sum = math.fsum(float(a*b) for a, b in zip(masses, rho))
    A_E = raw_robust_sum/Z_E
    half_lambda = .5*float(cfg['outline_factor_weight'])
    new_weighted = half_lambda*r*A_E
    one_way = old_observed
    one_way_mass = r*old_observed
    replacements = dict(BASELINE_E=float(components['outline']), HULL_ONEWAY=one_way,
                        HULL_ONEWAY_MASS=one_way_mass, VISIBLE_SUPPORT=new_weighted)
    scores = {'BASELINE_E': objective}
    scores.update({key: math.fsum((non_outline, value)) for key, value in replacements.items() if key != 'BASELINE_E'})
    contributions = masses*rho/Z_E*half_lambda*r
    if not math.isclose(math.fsum(contributions), new_weighted, abs_tol=ATOL, rel_tol=0):
        raise ValueError('Visible per-sample weighted sum differs from scalar')
    evidence = dict(**deepcopy(visible), raw_distance_px=distances.tolist(),
                    residual_xy_px=residuals.tolist(), nearest_model_segment_indices=nearest.tolist(),
                    standardized_radius=radius.tolist(), robust_losses=rho.tolist(),
                    weighted_sample_contributions=contributions.tolist(),
                    raw_distance_sum_px=math.fsum(distances), raw_distance_mean_px=float(distances.mean()),
                    raw_distance_max_px=float(distances.max()),
                    mass_weighted_distance_mean_px=math.fsum(float(a*b) for a, b in zip(masses, distances))/Z_E,
                    mass_weighted_robust_sum=raw_robust_sum, robust_mean_A_E=A_E,
                    model_projected_vertices_xy=points.tolist(), model_polygon_xy=polygon.tolist(),
                    model_finite_segments_xy=model_segments.tolist(), normal_sigma_px=cfg['normal_sigma_px'],
                    huber_delta=cfg['huber_delta'], outline_factor_weight=cfg['outline_factor_weight'],
                    direction_weight=.5, reverse_direction_present=False)
    return dict(**base, scores=scores, replacement_outline_costs=replacements,
                old_observed_mass_Z_H=Z_H, visible_mass_Z_E=Z_E, r=r,
                visible_evidence=evidence,
                original_native_projection_residual_max_abs=float(np.max(np.abs(old_residual-np.asarray(old['residual_xy_px'])))))
