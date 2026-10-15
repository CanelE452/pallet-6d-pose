"""Opt-in face-aware partial B/I image observations, never complete corner snaps.

Initial registered face visibility guides the association only. A B/I anchor
can query an actual incident RGB line even when its non-S mate is O/T/U: that
mate supplies model direction only, never a coordinate or image feature.
Ambiguous/grazing complete observations can be explicitly revoked in favour
of a line normal and a broad raw-centred CAL-borrowed tangent interval. The
original complete q/noise/reliability remain archived and available for seeds.
No state, model, prior, O/T domain or U region is changed here.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from itertools import combinations
import cv2
import numpy as np

from . import joint_ot_pnp as V3
from . import visible_bi_global_edge_correction_v4_9c as G
from . import boundary_refine as F
from .interior_observation import NEIGHBORS
from .truncation_interval_v4_2 import load_truncation_interval_model

FACE_TOLERANCE_DEG = 2.
DEFAULT_CONFIGURATION = deepcopy(G.DEFAULT_CONFIGURATION)
DEFAULT_CONFIGURATION['normal_search_radius_px'] = (G.DEFAULT_CONFIGURATION['along_end_px']
                                                   + G.DEFAULT_CONFIGURATION['boundary_context_px'])


def _config(configuration):
    cfg = G._config({**DEFAULT_CONFIGURATION, **(configuration or {})})
    # The new reach is the existing 36px line length plus its 7px context.
    if cfg['normal_search_radius_px'] != cfg['along_end_px'] + cfg['boundary_context_px']:
        raise ValueError('Face-aware reach must equal existing along_end + context')
    return cfg


def _face_context(geometry):
    xyz = V3._finite_array(geometry['native_xyz'], (8, 3), 'registered native xyz')
    R = V3._finite_array(geometry['R_cf'], (3, 3), 'initial model rotation')
    t = V3._finite_array(geometry['centroid'], (3,), 'initial centroid')
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(R), 1., atol=1e-6):
        raise ValueError('Proper initial native camera rotation required')
    # geometry.centroid is the camera location of the native model centroid.
    camera = xyz.mean(axis=0) - R.T @ t
    normals = V3.registered_incident_normals(xyz)
    rays = camera - xyz
    cosine = [n @ rays[k] / max(float(np.linalg.norm(rays[k])), 1e-12)
              for k, n in enumerate(normals)]
    band = float(np.sin(np.deg2rad(FACE_TOLERANCE_DEG)))
    return dict(normals=normals, cosine=cosine, band=band, camera_in_native=camera)


def _edge_face_role(context, k, mate):
    normals = context['normals']; cosines = context['cosine']; band = context['band']
    shared = [i for i, n in enumerate(normals[k])
              if any(np.allclose(n, m, atol=1e-8, rtol=0) for m in normals[mate])]
    if len(shared) != 2:
        raise ValueError('Native incident edge must share exactly two registered faces')
    values = [float(cosines[k][i]) for i in shared]
    roles = [('STRONG_VISIBLE' if v > band else 'STRONG_HIDDEN' if v < -band else 'GRAZING')
             for v in values]
    if roles.count('STRONG_VISIBLE') == 2:
        role = 'INTERNAL_TWO_VISIBLE_FACES'
    elif sorted(roles) == ['STRONG_HIDDEN', 'STRONG_VISIBLE']:
        role = 'OUTLINE_VISIBLE_HIDDEN_FACES'
    elif 'GRAZING' in roles:
        role = 'GRAZING_FACE_ASSOCIATION_UNRESOLVED'
    else:
        role = 'NO_STRONGLY_VISIBLE_INCIDENT_FACE'
    return dict(role=role, face_normals_native=[normals[k][i].tolist() for i in shared],
                initial_signed_face_cosines=values, face_visibility_roles=roles,
                face_tolerance_deg=FACE_TOLERANCE_DEG,
                conditional_on_initial_model_pose=True, independent_image_visibility=False)


def _guarded_trace(lab, gradients, mask, xy, normal, offsets, internal, cfg, unknown_model_mate):
    """Exclude an unknown model-mate neighbourhood before sampling any pixels."""
    if unknown_model_mate is None:
        return G._measure_trace(lab, gradients, mask, xy, normal, offsets, internal, cfg), None
    # Every remap probe is at most context px from xy. Keeping xy >= 2*context
    # makes all actual probes >= context px from the unknown model corner.
    context = max(float(cfg['RGB_probe_px']), float(cfg['boundary_context_px']))
    safe = np.linalg.norm(xy - unknown_model_mate, axis=-1) >= 2. * context
    query = xy.copy()
    query[~safe] = [-1000000., -1000000.]
    trace = G._measure_trace(lab, gradients, mask, query, normal, offsets, internal, cfg)
    trace['keep'] &= safe[np.arange(len(xy)), trace['best']]
    return trace, dict(excluded_query_candidates=int((~safe).sum()),
                       protected_model_mate_radius_px=context,
                       exclusion_applied_before_RGB_and_mask_sampling=True)


def _local_edge(mask, lab, gradients, k, mate, point, projection, internal, cfg):
    a = point['xy']; guide = projection[mate] - projection[k]
    length = float(np.linalg.norm(guide))
    result = dict(edge=k*8+mate, endpoints=[k, mate], native_corner=k, native_neighbor=mate,
                  supported=False, whole_edge_search=False,
                  guide_source='nonS_model_mate_direction_only_no_neighbor_coordinate',
                  measurement_source='OBSERVED_LOCAL_ANCHORED_FACE_RGB_WITH_MASK_CONTEXT',
                  neighbor_coordinate_reads=0, neighbor_corner_queries=0, GT_inputs=False)
    if length < 10.:
        return dict(result, reason='short_model_only_incident_guide')
    direction = guide / length; seed_normal = np.array([-direction[1], direction[0]])
    end = min(cfg['along_end_px'], .8*length, length-cfg['boundary_context_px'])
    if end <= cfg['along_start_px']:
        return dict(result, reason='no_anchor_local_span_before_unknown_mate')
    along = np.linspace(cfg['along_start_px'], end, cfg['along_samples'])
    positions = .15 + .7*(along-along[0])/(along[-1]-along[0])
    local = np.arange(-cfg['global_hypothesis_probe_px'], cfg['global_hypothesis_probe_px']+.01, cfg['normal_step_px'])
    shifts = np.arange(-cfg['normal_search_radius_px'], cfg['normal_search_radius_px']+.01, cfg['normal_step_px'])
    line_cfg = dict(along_interval=[.15, .85], huber_delta_px=cfg['maximum_line_rms_px'],
                    maximum_angle_change_deg=cfg['maximum_direction_change_deg'],
                    minimum_inlier_count=cfg['minimum_inlier_count'],
                    minimum_inlier_fraction=cfg['minimum_inlier_fraction'],
                    minimum_inlier_span_fraction=cfg['minimum_inlier_span_fraction'],
                    maximum_line_rms_px=cfg['maximum_line_rms_px'])
    hypotheses = []
    for shift in shifts:
        xy = (a + shift*seed_normal + along[:, None]*direction)[:, None, :] + local[None, :, None]*seed_normal
        trace, avoidance = _guarded_trace(lab, gradients, mask, xy, seed_normal, local, internal, cfg, projection[mate])
        selected = np.flatnonzero(trace['keep'])
        if len(selected) < cfg['minimum_inlier_count']:
            continue
        observed = xy[selected, trace['best'][selected]]
        confidence = np.clip(trace['contrast'][selected, trace['best'][selected]]/40., .05, 1.)
        fitted, diag = F._fit_line(observed, confidence, positions[selected], seed_normal, len(along), line_cfg)
        if fitted is None:
            continue
        n = np.asarray(diag['normal']); inliers = np.asarray(diag['inlier_mask'], bool)
        actual = observed[inliers]; tangent = np.array([-n[1], n[0]])
        interval = [float((actual@tangent).min()), float((actual@tangent).max())]
        energy = trace['energy'][selected, trace['best'][selected]]
        hypotheses.append({**result, **diag, 'supported': True,
                           'observed_points': observed.tolist(), 'inlier_mask': inliers.tolist(),
                           'actual_inlier_tangent_interval_px': interval,
                           'RGB_supported_sample_count': len(selected), 'samples': len(along),
                           'local_along_interval_px': [float(along[0]), float(along[-1])],
                           'unknown_mate_query_avoidance': avoidance,
                           'mask_context_kind': 'TARGET_TARGET_INTERNAL' if internal else 'TARGET_BACKGROUND_OUTLINE',
                           'hypothesis_source_shift_px': float(shift),
                           'quality': [float(diag['inlier_fraction']), float(diag['inlier_span_fraction']),
                                       int(inliers.sum()), -float(diag['inlier_rms_px']),
                                       float(np.median(energy)), -abs(float(shift))]})
    if not hypotheses:
        return dict(result, reason='no_anchor_local_actual_RGB_face_line', hypotheses=0)
    chosen = max(hypotheses, key=lambda r: tuple(r['quality']))
    chosen['valid_hypotheses'] = len(hypotheses)
    return chosen


def _whole_edge(image,mask,lab,gradients,k,mate,points,projection,internal,cfg):
    # Both endpoints have already passed the B/I and initial-S gates.
    a=points[k]['xy'];b=points[mate]['xy'];vector=b-a;length=float(np.linalg.norm(vector))
    result=dict(edge=k*8+mate,endpoints=[k,mate],native_corner=k,native_neighbor=mate,
        supported=False,measurement_source='OBSERVED_FULL_EDGE_FACE_RGB_WITH_MASK_CONTEXT',
        guide_source='actual_BI_endpoints_and_fixed_native_model_direction',
        whole_edge_search=True,GT_inputs=False)
    if length<10:return dict(result,reason='short_visible_endpoint_segment')
    guide=projection[mate]-projection[k];guide_length=float(np.linalg.norm(guide))
    if guide_length<1e-9:return dict(result,reason='degenerate_model_direction')
    seed_normal=np.array([-guide[1],guide[0]])/guide_length
    direction=vector/length;search_normal=np.array([-direction[1],direction[0]])
    central_grid=np.linspace(cfg['central_start_fraction'],cfg['central_end_fraction'],cfg['along_samples'])
    fractions=np.r_[central_grid,np.asarray(cfg['endpoint_sample_offsets_px'])/length,
                    1-np.asarray(cfg['endpoint_sample_offsets_px'])/length]
    fractions=np.unique(fractions[(fractions>0)&(fractions<1)])
    # Endpoint validation samples never enter the fit merely because they
    # happen to fall within 10--90% on a short edge. Coincident grid points
    # are a single observation, never a second independent sample.
    central=np.isclose(fractions[:,None],central_grid[None,:],atol=1e-12,rtol=0).any(axis=1)
    shifts=np.arange(-cfg['normal_search_radius_px'],cfg['normal_search_radius_px']+.01,cfg['normal_step_px'])
    local=np.arange(-cfg['global_hypothesis_probe_px'],cfg['global_hypothesis_probe_px']+.01,cfg['normal_step_px'])
    hypotheses=[]
    for shift in shifts:
        start=a+shift*search_normal
        # A corrected opposite endpoint anchors only the line guide. The
        # final line and support still come from actual independent samples.
        end=b if points[mate]['corrected'] else b+shift*search_normal
        tangent=end-start;tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
        if normal@seed_normal<0:normal=-normal
        anchors=start[None,:]+fractions[:,None]*(end-start)
        xy=anchors[:,None,:]+local[None,:,None]*normal
        trace=_measure_trace(lab,gradients,mask,xy,normal,local,internal,cfg)
        selected=np.arange(len(fractions))[trace['keep']&central]
        if len(selected)<cfg['minimum_inlier_count']:continue
        obs=xy[selected,trace['best'][selected]]
        confidence=np.clip(trace['contrast'][selected,trace['best'][selected]]/40.,.05,1.)
        line_cfg=dict(along_interval=[cfg['central_start_fraction'],cfg['central_end_fraction']],huber_delta_px=cfg['maximum_line_rms_px'],
            maximum_angle_change_deg=cfg['maximum_direction_change_deg'],minimum_inlier_count=cfg['minimum_inlier_count'],
            minimum_inlier_fraction=cfg['minimum_inlier_fraction'],minimum_inlier_span_fraction=cfg['minimum_inlier_span_fraction'],maximum_line_rms_px=cfg['maximum_line_rms_px'])
        fitted,diag=F._fit_line(obs,confidence,fractions[selected],seed_normal,int(central.sum()),line_cfg)
        if fitted is None:continue
        n=np.asarray(diag['normal']);offset=diag['offset'];central_inlier=np.asarray(diag['inlier_mask'],bool)
        all_selected=np.arange(len(fractions))[trace['keep']]
        all_observed=xy[all_selected,trace['best'][all_selected]]
        all_inliers=np.abs(all_observed@n+offset)<=cfg['maximum_line_rms_px']
        if int(all_inliers.sum())<cfg['minimum_inlier_count']:continue
        endpoint_support=_endpoint_support(fractions,central,all_selected,all_inliers)
        if not endpoint_support['passed']:continue
        endpoint_residuals={}
        for endpoint in (k,mate):
            if points[endpoint]['corrected']:
                endpoint_residuals[str(endpoint)]=abs(float(n@points[endpoint]['xy']+offset))
        margin=cfg['along_start_px']+cfg['boundary_context_px']
        if any(v>margin for v in endpoint_residuals.values()):continue
        actual_points=all_observed[all_inliers];t=np.array([-n[1],n[0]])
        interval=(actual_points@t).min(),(actual_points@t).max()
        energy=trace['energy'][selected,trace['best'][selected]]
        item={**result,**diag,'supported':True,'observed_points':all_observed.tolist(),'inlier_mask':all_inliers.tolist(),
            'inlier_count':int(all_inliers.sum()),'observation_confidence':np.clip(trace['contrast'][all_selected,trace['best'][all_selected]]/40.,.05,1.).tolist(),
            'central_observed_points':obs.tolist(),'central_inlier_mask':central_inlier.tolist(),
            'central_sample_count':int(central.sum()),'central_fraction_grid':central_grid.tolist(),
            'endpoint_validation_fraction_grid':fractions[~central].tolist(),
            'endpoint_validation_inlier_count':int(np.count_nonzero(all_inliers&~central[all_selected])),
            'independent_both_endpoint_support':endpoint_support,
            'RGB_supported_sample_count':int(trace['keep'].sum()),
            'central_RGB_supported_sample_count':len(selected),'actual_inlier_tangent_interval_px':list(map(float,interval)),
            'corrected_endpoint_normal_residual_px':endpoint_residuals,'endpoint_agreement_limit_px':margin,
            'mask_context_kind':'TARGET_TARGET_INTERNAL' if internal else 'TARGET_BACKGROUND_OUTLINE',
            'hypothesis_source_shift_px':float(shift),'endpoint_support_measured_separately':True,
            'quality':[float(diag['inlier_fraction']),float(diag['inlier_span_fraction']),int(central_inlier.sum()),
                       -float(diag['inlier_rms_px']),float(np.median(energy)),-abs(float(shift))]}
        hypotheses.append(item)
    if not hypotheses:return dict(result,reason='no_full_edge_RGB_context_supported_line',hypotheses=0)
    chosen=max(hypotheses,key=lambda r:tuple(r['quality']))
    chosen['valid_hypotheses']=len(hypotheses)
    return chosen



def _junctions(rays, q, cfg, image_hw):
    candidates = []; margin = cfg['along_start_px']+cfg['boundary_context_px']
    h, w = image_hw
    for a, b in combinations(rays, 2):
        matrix = np.asarray([a['normal'], b['normal']]); sine = abs(float(np.linalg.det(matrix)))
        if sine < cfg['minimum_intersection_sine']:
            continue
        point = np.linalg.solve(matrix, -np.asarray([a['offset'], b['offset']]))
        support = G._support(point, [a, b], margin); shift = float(np.linalg.norm(point-q))
        if support['passed'] and shift <= cfg['maximum_corner_shift_px'] and 0 <= point[0] < w and 0 <= point[1] < h:
            candidates.append(dict(edges=[a['edge'], b['edge']], intersection_xy=point.tolist(),
                                   intersection_sine=sine, finite_support=support, shift_px=shift))
    disagreement = max((float(np.linalg.norm(np.asarray(a['intersection_xy'])-b['intersection_xy']))
                        for a in candidates for b in candidates), default=0.)
    return dict(admissible_pairs=candidates, candidate_disagreement_px=disagreement,
                unique_within_existing_3px_rule=bool(candidates) and disagreement <= cfg['maximum_candidate_disagreement_px'],
                diagnostic_intersections_are_not_new_full_corner_observations=True)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name+' requires SHA256')
    return value


def observe_face_aware_partial_bi(image_bgr, pallet_mask, geometry, states, fit, *,
                                raw_native_points, localization_model, nonself_scale_px,
                                image_binding, configuration=None):
    cfg = _config(configuration); image = np.asarray(image_bgr); mask = np.asarray(pallet_mask, float)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Actual uint8 BGR image required')
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('Bound full mask occupancy in[0,1] required')
    states = [V3._state(s) for s in states]
    if len(states) != 8 or len(geometry['initial_self_hidden']) != 8 or tuple(geometry['image_hw']) != image.shape[:2]:
        raise ValueError('Bound eight states and actual canvas required')
    if not isinstance(image_binding, Mapping) or tuple(image_binding.get('image_hw', ())) != image.shape[:2]:
        raise ValueError('Caller original image binding required')
    binding = deepcopy(dict(image_binding)); _sha(binding.get('sha256'), 'original image')
    pixels = G.OLD.image_sha256(image)
    if 'image_pixel_sha256' in binding and binding['image_pixel_sha256'] != pixels:
        raise ValueError('Caller pixel binding differs')
    binding['image_pixel_sha256'] = pixels
    if not isinstance(localization_model, Mapping):
        raise ValueError('Explicit existing CAL localization model mapping required; no runtime file lookup')
    model = load_truncation_interval_model(localization_model)
    scale = float(nonself_scale_px)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Frozen positive nonS prediction scale required')
    radius = float(model['radius_ratio'])*scale
    hidden = geometry['initial_self_hidden']
    points = G._eligible_points(fit, states, hidden, image.shape[:2])
    context = _face_context(geometry)
    # The geometry-only O/T/U mate slot is accessed only after the S gates.
    allowed = [k for k in range(8) if not hidden[k] and states[k] != 'S']
    projection = {k: V3._finite_array(geometry['initial_projection_xy'][k], (2,), 'nonS model projection') for k in allowed}
    lab = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32), (3, 3), .8)
    gradients = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)/8., cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)/8.
    records = []; edges = []; observations = []; queried = []
    for k, state in enumerate(states):
        audit = dict(native_corner=k, state=state, accepted_partial=False, coordinate_read=False, pixel_queried=False)
        records.append(audit)
        if hidden[k] or state not in ('B', 'I') or k not in points:
            audit['reason'] = 'excluded_before_coordinate_and_local_RGB_queries'; continue
        q = points[k]['xy']; m = points[k]['record']
        queried.append(k); audit.update(coordinate_read=True, pixel_queried=True, original_source=m.get('source'))
        rays = []
        for mate in NEIGHBORS[k]:
            if hidden[mate] or states[mate] == 'S':
                rays.append(dict(edge=k*8+mate, native_corner=k, native_neighbor=mate, supported=False,
                                 reason='S_endpoint_excluded_before_projection_and_RGB_query', pixel_queried=False)); continue
            face = _edge_face_role(context, k, mate)
            if face['role'] not in ('INTERNAL_TWO_VISIBLE_FACES', 'OUTLINE_VISIBLE_HIDDEN_FACES'):
                rays.append(dict(edge=k*8+mate, native_corner=k, native_neighbor=mate, supported=False,
                                 reason='incident_face_visibility_ambiguous_or_hidden', face_association=face, pixel_queried=False)); continue
            internal = face['role'] == 'INTERNAL_TWO_VISIBLE_FACES'
            if mate in points:
                ray = _whole_edge(image, mask, lab, gradients, k, mate, points, projection, internal, cfg)
            else:
                ray = _local_edge(mask, lab, gradients, k, mate, points[k], projection, internal, cfg)
            ray.update(face_association=face, pixel_queried=True, neighbor_state=states[mate])
            rays.append(ray); edges.append(ray)
        audit['rays'] = rays
        supported = [r for r in rays if r.get('supported')]
        junction = _junctions(supported, q, cfg, image.shape[:2]); audit['junction_audit'] = junction
        grazing = any(abs(float(c)) <= context['band'] for c in context['cosine'][k])
        audit['initial_incident_face_grazing'] = grazing
        if not supported:
            audit['reason'] = 'no_stable_face_associated_actual_RGB_line'; continue
        corrected = points[k]['corrected']
        if corrected and not grazing and junction['unique_within_existing_3px_rule']:
            audit['reason'] = 'existing_complete_BI_with_unique_stable_face_junction_preserved'; continue
        if corrected and not grazing and len(junction['admissible_pairs']) < 2:
            audit['reason'] = 'existing_corrected_BI_not_revoked_without_grazing_or_actual_pair_ambiguity'; continue
        if not corrected and not grazing and junction['unique_within_existing_3px_rule']:
            audit['reason'] = 'unique_full_junction_diagnostic_no_new_full_point_created'; continue
        reason = ('grazing_incident_face_junction' if grazing else
                  'only_local_visible_edge_no_neighbor_measurement' if any(not r['whole_edge_search'] for r in supported) else
                  'no_unique_face_supported_junction')
        # Choose one stable line, preferring a crease between two strong faces.
        # This is partial evidence even when other non-universal lines exist.
        chosen = max(supported, key=lambda r: (r['face_association']['role'] == 'INTERNAL_TWO_VISIBLE_FACES', tuple(r['quality'])))
        n = np.asarray(chosen['normal']); tangent = np.array([-n[1], n[0]])
        raw = V3._finite_array(raw_native_points[k], (2,), 'first B/I raw forecast')
        covariance = V3._finite_array(m['covariance_px2'], (2, 2), 'original observed covariance')
        effective = V3._finite_array(m['fit_covariance_px2'], (2, 2), 'original fixed CAL fit covariance')
        if not np.allclose(effective, effective.T, atol=1e-12, rtol=0):
            raise ValueError('Fixed effective covariance must be symmetric')
        np.linalg.cholesky(effective); reliability = float(m['fit_reliability_weight'])
        if not np.isfinite(reliability) or reliability <= 0:
            raise ValueError('Fixed positive fit reliability required')
        actual_interval = chosen['actual_inlier_tangent_interval_px']
        raw_interval = [float(tangent@raw)-radius, float(tangent@raw)+radius]
        interval = [min(actual_interval[0], raw_interval[0]), max(actual_interval[1], raw_interval[1])]
        margin = float(cfg['along_start_px']+cfg['boundary_context_px'])
        line = dict(normal_unit_xy=n.tolist(), offset_px=float(chosen['offset']),
                    normal_variance_px2=float(n@effective@n), support_tangent_interval_px=interval,
                    support_extrapolation_px=margin,
                    provenance=dict(native_neighbor=chosen['native_neighbor'], actual_RGB_line_id=chosen['edge'],
                        actual_observed_inlier_tangent_interval_px=actual_interval,
                        raw_tangent_CAL_borrowed_interval_px=raw_interval,
                        face_association=chosen['face_association'], line_record=deepcopy(chosen),
                        tangent_domain_rule='hull(actual RGB inliers, first raw forecast tangent +/- CAL nonS max)',
                        normal_variance_rule='one normal.T @ original fixed CAL fit covariance @ normal',
                        normal_variance_calibrated=False, shared_line_independence_claimed=False))
        display = q-n*(n@q+chosen['offset'])
        archive = dict(original_measurement_source=m.get('source'), original_observed_xy=q.tolist(),
                       original_observed_covariance_px2=covariance.tolist(),
                       original_effective_fit_covariance_px2=effective.tolist(),
                       original_fit_reliability_weight=reliability, revocation_reason=reason,
                       original_raw_xy=raw.tolist(), normal_projected_display_xy=display.tolist(),
                       display_coordinate_is_full_2D_measurement=False,
                       geometry_face_roles=[r.get('face_association') for r in rays if 'face_association' in r],
                       actual_RGB_line_ids=[chosen['edge']], CAL_localization_model_sha256=model['model_sha256'],
                       borrowed_tangent_radius_px=radius, tangent_range_is_probability=False,
                       no_precise_2D_corner_or_neighbor_unknown_measurement_created=True)
        observations.append(dict(native_corner=k, state=state, lines=[line], image_binding=deepcopy(binding),
            fit_reliability_weight=reliability, replaces_complete_BI_likelihood=True,
            explicit_complete_BI_likelihood_revocation=True, association_provenance=archive))
        audit.update(accepted_partial=True, reason=reason, selected_neighbor_ids=[chosen['native_neighbor']],
                     normal_projected_display_xy=display.tolist(), full_2D_corrected_point_created=False)
    return dict(schema='FACE_AWARE_PARTIAL_BI_OBSERVATIONS_V4_13', boundary_observations=observations,
                records=records, edge_records=edges, configuration=cfg, queried_native_ids=queried,
                visible_measurements=[deepcopy(points[k]['record']) for k in points], states_unchanged=True,
                image_binding=binding, image_pixel_sha256=pixels, S_coordinate_reads=0, S_local_queries=0,
                U_coordinate_reads=0, U_corner_queries=0, O_T_coordinate_reads=0, GT_inputs=False,
                localization_model_sha256=model['model_sha256'], borrowed_tangent_radius_px=radius,
                unchanged_nonself_prediction_scale_px=scale, normal_variance_calibrated=False,
                tangent_range_is_probability=False, full_corner_measurements_created=0,
                complete_q_used_only_as_old_seeds_or_archived_reference_when_revoked=True)


observe_face_aware_partial_BI = observe_face_aware_partial_bi
