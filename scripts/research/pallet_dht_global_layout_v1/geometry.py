"""GT-free, approximate joint selection of one eight-corner cuboid layout.

This module reads predicted points and image-predicted Hough evidence only.
All coordinates and distance scales are original-image pixels. The projective
camera is a consistency model, not a calibrated metric pose or a GT relabeling.
"""
from itertools import combinations
import math

import numpy as np

from scripts.research.pallet_dht_decoder_probe_v1.geometry import (
    EDGES, INCIDENT_ROLES, feature_line_to_raw, topk_lines,
)


ORDERS = ((0,1,2,3,4,5,6,7), (4,5,6,7,0,1,2,3),
          (0,4,5,1,3,7,6,2), (3,2,6,7,0,1,5,4))
YAW90 = np.array([1,5,6,2,0,4,7,3], dtype=np.int64)
YAW_PERMUTATIONS = np.stack([np.arange(8), YAW90, YAW90[YAW90], YAW90[YAW90[YAW90]]])
CUBOID = np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
                   [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1]], dtype=np.float64)
CUBOID_H = np.column_stack([CUBOID, np.ones(8)])
DEFAULT_CONFIG = dict(
    schema='pallet_global_layout_geometry_v1', candidates_per_corner=16,
    baseline_coordinate_candidates=8, intersection_candidates=8, beam_width=64,
    orders=[list(o) for o in ORDERS], top_k_lines=4,
    nms_angle_degrees=4., nms_rho_feature_cells=1.,
    line_sigma_raw_diagonal_fraction=.01, line_sigma_min_raw_px=1.,
    anchor_scale_raw_diagonal_fraction=.1,
    collapse_rms_raw_diagonal_fraction=1e-8, image_spread_rank_ratio_min=1e-8,
    dlt_rank_ratio_min=1e-10, camera_rank_ratio_min=1e-8,
    same_depth_relative_margin=1e-6,
    projective_penalty='mean8_pseudoHuber(raw_reprojection_distance/sigma)',
    pseudo_huber_delta=1.,
    candidate_pruning='mean_three_incident_single_endpoint_mixture_cost_only',
    line_probability='normalized_sigmoid_mass_among_four_retained_NMS_peaks',
    shared_cost='mean12[-log sum_k p_rk exp(-0.5*(H(d_u/sigma)+H(d_v/sigma)))]',
    independent_cost='mean8[mean3[-log sum_k p_rk exp(-H(d_i/sigma))]]',
    shared_replaces_independent=True,
    geometry_bank_depends_on='w_point_only; same_bank_for_all_w_geom',
    tie_policy='stable_generation_order; baseline_whole_layout_first',
    invalid_baseline_exception='eligible_with_geometry_cost_zero; not_a_valid_geometry_claim',
    missing_input_policy='baseline_exact_for_missing_detection_or_any_missing_corner_or_no_line_roles',
    missing_line_role_policy='zero_cost_with_fixed12_or3_denominator; record_available_roles',
    inference='finite_beam_search_and_explicit_hypotheses; no_global_optimality_guarantee',
    output='selected_candidate_coordinates; no_DLT_reprojection_of_output',
    centroid='copy_original', evaluation_identity='same_camera_dynamic_0123_v4',
)


def pseudo_huber(x):
    x = np.asarray(x, dtype=np.float64)
    return np.hypot(1., x) - 1.


def _logsumexp(x, axis=-1):
    top = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.exp(x-top).sum(axis=axis))


def endpoint_cost(points, lines, weights, sigma):
    """Single-point mixture over unit-Hessian raw-pixel line equations."""
    points = np.asarray(points, dtype=np.float64)
    lines = np.asarray(lines, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(weights):
        return np.zeros(points.shape[:-1], dtype=np.float64)
    residual = (points @ lines[:, :2].T + lines[:, 2]) / sigma
    return -_logsumexp(np.log(weights) - pseudo_huber(residual))


def pair_cost(first, second, lines, weights, sigma):
    """Both endpoints must be explained by the SAME latent line mode.

    Returns [len(first),len(second)]. It is not the sum of two independently
    chosen mixtures. For one mode it equals half the two endpoint costs.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if not len(weights):
        return np.zeros((len(first), len(second)), dtype=np.float64)
    a = pseudo_huber((first @ lines[:, :2].T + lines[:, 2]) / sigma)
    b = pseudo_huber((second @ lines[:, :2].T + lines[:, 2]) / sigma)
    terms = np.log(weights)[None, None] - .5*(a[:, None] + b[None])
    return -_logsumexp(terms)


def projective_penalty(points_xy, raw_diagonal, sigma, config=None):
    """Normalized DLT fit of 8 labeled cube vertices to an arbitrary 3x4 P.

    A projective camera absorbs unknown cuboid dimensions. Its homogeneous
    last coordinate must have one sign at all eight vertices. Vanishing
    points at infinity are allowed; image-space parallelism is not required.
    All thresholds are numerical guards, not a physical visibility decision.
    """
    cfg = DEFAULT_CONFIG if config is None else config
    q = np.asarray(points_xy, dtype=np.float64)
    def invalid(reason, **extra):
        return dict(valid=False, reason=reason, cost=None, **extra)
    if q.shape != (8, 2) or not np.isfinite(q).all():
        return invalid('nonfinite_or_malformed_points')
    center = q.mean(0)
    centered = q-center
    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if rms <= cfg['collapse_rms_raw_diagonal_fraction']*raw_diagonal:
        return invalid('collapsed_image_layout')
    spread = np.linalg.svd(centered, compute_uv=False)
    if spread[-1] <= cfg['image_spread_rank_ratio_min']*spread[0]:
        return invalid('collinear_image_layout')
    scale = math.sqrt(2.)/rms
    normalized = centered*scale
    a = np.zeros((16,12), dtype=np.float64)
    a[0::2,0:4] = CUBOID_H
    a[1::2,4:8] = CUBOID_H
    a[0::2,8:12] = -normalized[:,0,None]*CUBOID_H
    a[1::2,8:12] = -normalized[:,1,None]*CUBOID_H
    try:
        _, singular, vh = np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError:
        return invalid('dlt_svd_failure')
    # At least 11 independent constraints are needed for a unique camera.
    if singular[-2] <= cfg['dlt_rank_ratio_min']*singular[0]:
        return invalid('rank_deficient_dlt')
    camera = vh[-1].reshape(3,4)
    cs = np.linalg.svd(camera, compute_uv=False)
    if cs[-1] <= cfg['camera_rank_ratio_min']*cs[0]:
        return invalid('rank_deficient_camera')
    homogeneous = CUBOID_H @ camera.T
    depths = homogeneous[:,2]
    margin = cfg['same_depth_relative_margin']*np.max(np.abs(depths))
    if not (np.all(depths > margin) or np.all(depths < -margin)):
        return invalid('mixed_or_nearzero_projective_depth')
    fit = homogeneous[:,:2]/depths[:,None]/scale + center
    if not np.isfinite(fit).all():
        return invalid('nonfinite_reprojection')
    distances = np.linalg.norm(fit-q, axis=1)
    return dict(valid=True, reason='valid_projective_fit',
                cost=float(pseudo_huber(distances/sigma).mean()),
                residual_px=distances, mean_residual_px=float(distances.mean()),
                maximum_residual_px=float(distances.max()),
                fitted_points_xy=fit, normalized_camera=camera,
                depth_absolute_ratio=float(np.abs(depths).min()/np.abs(depths).max()))


def prepare_inputs(points_xy, point_valid, width, height, candidates_xy,
                   candidate_valid, line_h, line_weights, line_valid,
                   detected=True, config=None):
    """Pure prediction-only constructor, also used by generated geometry tests."""
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    if cfg['candidates_per_corner'] != 16 or cfg['intersection_candidates'] != 8:
        raise ValueError('Frozen pilot requires K16 = 8 predicted points + 8 intersections')
    p = np.asarray(points_xy, dtype=np.float64)
    pv = np.asarray(point_valid, dtype=bool)
    old = np.asarray(candidates_xy, dtype=np.float64)
    ov = np.asarray(candidate_valid, dtype=bool)
    h = np.asarray(line_h, dtype=np.float64)
    weights = np.asarray(line_weights, dtype=np.float64)
    lv = np.asarray(line_valid, dtype=bool)
    if p.shape != (9,2) or pv.shape != (9,) or old.shape != (8,49,2) or ov.shape != (8,49):
        raise ValueError('Malformed point candidate shapes')
    if h.shape != (12,4,3) or weights.shape != (12,4) or lv.shape != (12,4):
        raise ValueError('Malformed role/line shapes')
    if min(width,height) <= 0 or not np.isfinite([width,height]).all():
        raise ValueError('Invalid raw image dimensions')
    if not np.isfinite(p[pv]).all() or not np.isfinite(old[ov]).all():
        raise ValueError('Nonfinite valid predicted point')
    if not np.isfinite(h[lv]).all() or not np.isfinite(weights[lv]).all() or np.any(weights[lv] <= 0):
        raise ValueError('Invalid positive line masses')
    norms = np.linalg.norm(h[lv,:2], axis=-1)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero line normal')
    h = h.copy()
    h[lv] /= norms[:,None]
    diagonal = math.hypot(width,height)
    sigma = max(cfg['line_sigma_min_raw_px'], cfg['line_sigma_raw_diagonal_fraction']*diagonal)
    lines, masses = [], []
    for role in range(12):
        lines.append(h[role,lv[role]])
        w = weights[role,lv[role]]
        masses.append(w/w.sum() if len(w) else w)
    fallback = None
    if not detected: fallback = 'no_detection'
    elif not pv[:8].all(): fallback = 'missing_predicted_corner'
    elif not lv.any(): fallback = 'no_line_evidence'
    safe_points = np.where(np.isfinite(p),p,0.)
    pool = np.repeat(safe_points[None,:8], 8, axis=0)
    pool = np.concatenate([pool,np.repeat(safe_points[:8,None],8,axis=1)],axis=1)
    pool_valid = np.zeros((8,16),bool)
    pool_valid[:,:8] = pv[None,:8]
    sources = np.full((8,16),-1,dtype=np.int64)
    for corner in range(8):
        indices = np.flatnonzero(ov[corner,1:])+1
        if len(indices):
            q = old[corner,indices]
            cost = np.mean([endpoint_cost(q,lines[r],masses[r],sigma)
                            for r in INCIDENT_ROLES[corner]],axis=0)
            chosen = indices[np.lexsort((indices,cost))[:8]]
            pool[corner,8:8+len(chosen)] = old[corner,chosen]
            pool_valid[corner,8:8+len(chosen)] = True
            sources[corner,8:8+len(chosen)] = chosen
    unary = np.zeros((8,16))
    for i,roles in enumerate(INCIDENT_ROLES):
        unary[i] = np.mean([endpoint_cost(pool[i],lines[r],masses[r],sigma) for r in roles],axis=0)
    anchors = pseudo_huber(np.linalg.norm(pool-safe_points[:8,None],axis=-1)/
                          (cfg['anchor_scale_raw_diagonal_fraction']*diagonal))
    pairs = np.stack([pair_cost(pool[u],pool[v],lines[r],masses[r],sigma)
                      for r,(u,v) in enumerate(EDGES)])
    if not np.isfinite(unary).all() or not np.isfinite(anchors).all() or not np.isfinite(pairs).all():
        raise ValueError('Nonfinite derived cost')
    return dict(schema='pallet_global_layout_prepared_v1',config=cfg,
                points_xy=p.copy(),point_valid=pv.copy(),detected=bool(detected),
                raw_shape_hw=np.array([height,width]),diagonal=diagonal,sigma=sigma,
                candidate_xy=pool,candidate_valid=pool_valid,candidate_source_slot=sources,
                original_candidates_xy=old.copy(),original_candidate_valid=ov.copy(),
                unary_line_cost=unary,anchor_cost=anchors,pair_line_cost=pairs,
                lines=lines,line_weights=masses,available_line_roles=lv.any(1),
                fallback_reason=fallback,uses_gt=False)


def prepare(record, frame_arrays, evidence_arrays):
    """Read only prediction fields, raw dimensions and saved Hough evidence.

    frame_arrays and evidence_arrays are NPZ-like mappings. They may contain
    unrelated fields; no GT, matching, visibility label or pose is accessed.
    """
    b = record['baseline']
    theta = np.asarray(frame_arrays['theta'],dtype=np.float64)
    rho = np.asarray(frame_arrays['rho'],dtype=np.float64)
    logits = np.asarray(frame_arrays['logits'],dtype=np.float64)
    valid = np.asarray(frame_arrays['lattice_valid'],dtype=bool)
    feature_shape = frame_arrays['feature_shape_hw']
    input_shape = frame_arrays['input_shape_hw']
    affine = frame_arrays['raw_to_input_affine']
    if logits.shape != (12,len(theta),len(rho)) or valid.shape != (len(theta),len(rho)):
        raise ValueError('Malformed Hough lattice')
    line_h = np.zeros((12,4,3))
    weights = np.zeros((12,4))
    line_valid = np.zeros((12,4),bool)
    top_config = dict(top_k=4,nms_angle_degrees=4.,nms_rho_feature_cells=1.)
    for role in range(12):
        peaks = topk_lines(logits[role],theta,rho,valid,top_config)
        logmass = []
        for k,peak in enumerate(peaks):
            line_h[role,k] = feature_line_to_raw(peak['theta'],peak['rho'],
                                               feature_shape,input_shape,affine)
            line_valid[role,k] = True
            logmass.append(-np.logaddexp(0.,-peak['logit']))
        if len(logmass):
            logmass = np.asarray(logmass)
            weights[role,:len(logmass)] = np.exp(logmass-logmass.max())
    old_lines = np.asarray(evidence_arrays['line_fusion__line_peaks_h_raw'],dtype=float)
    old_valid = np.asarray(evidence_arrays['line_fusion__line_peak_valid'],dtype=bool)
    if not np.array_equal(line_valid,old_valid) or not np.allclose(line_h[line_valid],old_lines[line_valid],rtol=0,atol=1e-8):
        raise ValueError('Cached line/affine provenance mismatch')
    out = prepare_inputs(b['points'],b['point_valid'],record['width'],record['height'],
                         evidence_arrays['line_fusion__candidates_xy'],
                         evidence_arrays['line_fusion__candidate_valid'],
                         line_h,weights,line_valid,detected=b['detected'])
    out['id'] = record['id']
    return out


def _layout_terms(prepared, indices):
    indices = np.asarray(indices,dtype=np.int64)
    rows = np.arange(8)
    unary = float(prepared['unary_line_cost'][rows,indices].mean())
    anchor = float(prepared['anchor_cost'][rows,indices].mean())
    per_edge = np.array([prepared['pair_line_cost'][r,indices[u],indices[v]]
                         for r,(u,v) in enumerate(EDGES)])
    return unary,anchor,float(per_edge.mean()),per_edge


def score_layout(prepared, points_xy, w_point, w_geom, baseline_exception=False):
    """Public prediction-only scorer for any explicitly provided whole layout.

    A caller may use this after selection for a separately labelled GT-oracle
    diagnostic. This function has no GT input and does not choose the layout.
    Invalid-layout zero geometry cost is permitted ONLY for exact baseline.
    """
    q = np.asarray(points_xy,dtype=np.float64)
    if q.shape != (9,2) or not np.isfinite(q[:8]).all():
        raise ValueError('Finite eight corners and copied centroid required')
    if baseline_exception and not np.array_equal(q,prepared['points_xy'],equal_nan=True):
        raise ValueError('Only exact original layout may use the baseline exception')
    if not np.isfinite([w_point,w_geom]).all() or min(w_point,w_geom)<0:
        raise ValueError('Nonnegative finite weights required')
    per_edge = np.array([pair_cost(q[u:u+1],q[v:v+1],prepared['lines'][r],
                                   prepared['line_weights'][r],prepared['sigma'])[0,0]
                         for r,(u,v) in enumerate(EDGES)])
    unary = np.mean([np.mean([endpoint_cost(q[i:i+1],prepared['lines'][r],
                                          prepared['line_weights'][r],prepared['sigma'])[0]
                             for r in INCIDENT_ROLES[i]]) for i in range(8)])
    anchor = pseudo_huber(np.linalg.norm(q[:8]-prepared['points_xy'][:8],axis=-1)/
                          (prepared['config']['anchor_scale_raw_diagonal_fraction']*prepared['diagonal'])).mean()
    geometry = projective_penalty(q[:8],prepared['diagonal'],prepared['sigma'],prepared['config'])
    eligible = bool(geometry['valid'] or baseline_exception)
    gcost = float(geometry['cost']) if geometry['valid'] else (0. if baseline_exception else None)
    raw = dict(shared_line=float(per_edge.mean()),independent_line=float(unary),
               point_prior=float(anchor),geometry=gcost)
    weighted = dict(shared_line=raw['shared_line'],point_prior=float(w_point*anchor),
                    geometry=float(w_geom*gcost) if gcost is not None else None)
    return dict(points_xy=q.tolist(),raw_terms=raw,
                weights=dict(shared_line=1.,point_prior=float(w_point),geometry=float(w_geom)),
                weighted_terms=weighted,total_score=sum(weighted.values()) if eligible else None,
                eligible=eligible,geometry_valid=bool(geometry['valid']),geometry_reason=geometry['reason'],
                baseline_invalid_geometry_zero_cost_exception=bool(baseline_exception and not geometry['valid']),
                per_edge_shared_cost=per_edge.tolist(),uses_gt=False)


def independent_select(prepared, w_point):
    if not np.isfinite(w_point) or w_point < 0:
        raise ValueError('Nonnegative finite point weight required')
    scores = prepared['unary_line_cost']+w_point*prepared['anchor_cost']
    # Put the original semantic point first in an exact tie, not coordinate0.
    indices = np.empty(8,dtype=np.int64)
    for i in range(8):
        order = np.r_[i,np.arange(16)[np.arange(16)!=i]]
        indices[i] = order[np.argmin(np.where(prepared['candidate_valid'][i,order],scores[i,order],np.inf))]
    if prepared['fallback_reason']:
        indices = np.arange(8)
    q = prepared['points_xy'].copy()
    if not prepared['fallback_reason']:
        q[:8] = prepared['candidate_xy'][np.arange(8),indices]
    unary,anchor,shared,per_edge = _layout_terms(prepared,indices)
    return dict(points_xy=q,point_valid=prepared['point_valid'].copy(),indices=indices,
                independent_line_cost=unary,point_prior_cost=anchor,
                total_score=unary+w_point*anchor,shared_line_cost=shared,
                per_edge_shared_cost=per_edge,w_point=float(w_point),uses_gt=False,
                fallback_reason=prepared['fallback_reason'],centroid_preserved=True)


def _beam(prepared, order, w_point):
    assignments = np.full((1,8),-1,dtype=np.int64)
    scores = np.zeros(1)
    done = set()
    for corner in order:
        choices = np.flatnonzero(prepared['candidate_valid'][corner])
        extended = scores[:,None]+w_point*prepared['anchor_cost'][corner,choices][None]/8.
        for role,(u,v) in enumerate(EDGES):
            if corner==u and v in done:
                extended += prepared['pair_line_cost'][role][choices[None,:],assignments[:,v,None]]/12.
            elif corner==v and u in done:
                extended += prepared['pair_line_cost'][role][assignments[:,u,None],choices[None,:]]/12.
        take = np.argsort(extended.ravel(),kind='stable')[:prepared['config']['beam_width']]
        parent,choice = np.unravel_index(take,extended.shape)
        assignments = assignments[parent].copy()
        assignments[:,corner] = choices[choice]
        scores = extended.ravel()[take]
        done.add(corner)
    return assignments


def build_layouts(prepared, w_point):
    independent = independent_select(prepared,w_point)
    all_indices, kinds, seen = [],[],{}
    explicit_seed_indices = {}
    def add(a,kind):
        key = tuple(map(int,a))
        if key not in seen:
            seen[key]=len(all_indices);all_indices.append(key);kinds.append(kind)
        return seen[key]
    explicit_seed_indices['baseline'] = add(np.arange(8),'baseline')
    explicit_seed_indices['baseline_yaw0'] = 0
    if not prepared['fallback_reason']:
        for k,perm in enumerate(YAW_PERMUTATIONS[1:],1):
            explicit_seed_indices[f'baseline_yaw{k*90}'] = add(perm,f'baseline_yaw{k*90}')
        explicit_seed_indices['independent'] = add(independent['indices'],'independent')
        for k,order in enumerate(ORDERS):
            for indices in _beam(prepared,order,w_point):add(indices,f'beam_order{k}')
    ii = np.asarray(all_indices,dtype=np.int64)
    layouts = np.repeat(prepared['points_xy'][None],len(ii),axis=0)
    if not prepared['fallback_reason']:
        layouts[:,:8] = prepared['candidate_xy'][np.arange(8)[None],ii]
    unaries,anchors,shared,edge_costs,geom,valid,reasons,raw_geom = [],[],[],[],[],[],[],[]
    for k,(indices,layout) in enumerate(zip(ii,layouts)):
        u,a,s,e = _layout_terms(prepared,indices)
        g = projective_penalty(layout[:8],prepared['diagonal'],prepared['sigma'],prepared['config'])
        unaries.append(u);anchors.append(a);shared.append(s);edge_costs.append(e)
        geom.append(float(g['cost']) if g['valid'] else (0. if k==0 else np.inf))
        valid.append(g['valid']);reasons.append(g['reason']);raw_geom.append(g)
    return dict(schema='pallet_global_layout_bank_v1',hypothesis_xy=layouts,candidate_indices=ii,
                hypothesis_kind=kinds,independent=independent,w_point=float(w_point),
                independent_line_cost=np.asarray(unaries),point_prior_cost=np.asarray(anchors),
                shared_line_cost=np.asarray(shared),per_edge_shared_cost=np.asarray(edge_costs),
                geometry_cost=np.asarray(geom),geometry_valid=np.asarray(valid,bool),
                geometry_reason=reasons,geometry_details=raw_geom,baseline_hypothesis_index=0,
                explicit_seed_indices=explicit_seed_indices,
                baseline_geometry_exception=not valid[0],fallback_reason=prepared['fallback_reason'],
                bank_is_geometry_weight_independent=True,uses_gt=False)


def select_layout(prepared, bank, w_geom, top_n=3):
    if not np.isfinite(w_geom) or w_geom < 0:
        raise ValueError('Nonnegative finite geometry weight required')
    eligible = bank['geometry_valid'].copy()
    eligible[0] = True  # Explicit identity fallback, never a valid-geometry claim.
    scores = bank['shared_line_cost']+bank['w_point']*bank['point_prior_cost']
    if w_geom:
        scores = scores+w_geom*bank['geometry_cost']
    scores = np.where(eligible,scores,np.inf)
    order = np.argsort(scores,kind='stable')
    chosen = int(order[0])
    def details(k):
        gvalid = bool(bank['geometry_valid'][k])
        gcost = float(bank['geometry_cost'][k]) if gvalid or k==0 else None
        return dict(index=int(k),kind=bank['hypothesis_kind'][k],
                    candidate_indices=bank['candidate_indices'][k].tolist(),
                    points_xy=bank['hypothesis_xy'][k].tolist(),
                    raw_terms=dict(shared_line=float(bank['shared_line_cost'][k]),
                                   independent_line=float(bank['independent_line_cost'][k]),
                                   point_prior=float(bank['point_prior_cost'][k]),geometry=gcost),
                    weights=dict(shared_line=1.,point_prior=bank['w_point'],geometry=float(w_geom)),
                    weighted_terms=dict(shared_line=float(bank['shared_line_cost'][k]),
                                        point_prior=float(bank['w_point']*bank['point_prior_cost'][k]),
                                        geometry=float(w_geom*gcost) if gcost is not None else None),
                    total_score=float(scores[k]) if np.isfinite(scores[k]) else None,
                    score_direction='lower_is_better',score_unit='dimensionless_pseudoHuber_and_negative_log_mixture',
                    per_edge_shared_cost=bank['per_edge_shared_cost'][k].tolist(),
                    geometry_valid=gvalid,geometry_reason=bank['geometry_reason'][k],
                    baseline_invalid_geometry_zero_cost_exception=bool(k==0 and not gvalid))
    finite_order = [int(k) for k in order if np.isfinite(scores[k])]
    return dict(schema='pallet_global_layout_selection_v1',points_xy=bank['hypothesis_xy'][chosen].copy(),
                point_valid=prepared['point_valid'].copy(),selected_index=chosen,
                selected=details(chosen),top_hypotheses=[details(k) for k in finite_order[:top_n]],
                explicit_hypotheses={name:details(k) for name,k in bank['explicit_seed_indices'].items()},
                selected_score=float(scores[chosen]),
                score_gap=float(scores[finite_order[1]]-scores[chosen]) if len(finite_order)>1 else None,
                n_hypotheses=len(scores),n_geometry_valid=int(bank['geometry_valid'].sum()),
                baseline_geometry_exception=bool(bank['baseline_geometry_exception']),
                baseline_exception_selected=bool(chosen==0 and bank['baseline_geometry_exception']),
                baseline_selected=chosen==0,fallback_reason=prepared['fallback_reason'],
                identity_forced_by_input_failure=prepared['fallback_reason'] is not None,
                geometry_penalty_enabled=bool(w_geom>0),geometry_safety_guard_retained=True,
                centroid_preserved=True,uses_gt=False,global_optimality_claim=False)
