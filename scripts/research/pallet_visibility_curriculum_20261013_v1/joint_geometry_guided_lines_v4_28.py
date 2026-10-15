"""Opt-in joint geometry-guided *image line proposals*, not KP/PnP evidence.

Actual lines still come from frozen17c image-lattice NMS and TLS. Physical-axis
vanishing points propose additional strips through eligible B/I observations;
they never replace a measured line. The image-endpoint guide and original17c
candidates remain. A deterministic bounded beam ranks an image-line graph with
one weak normalized guide term, never an independent likelihood per edge.

S has no local coordinate/image query. U/O/T can supply model direction only;
their image observations are not inputs. Rounded-junction diagnostics exist
only for B/I and are explicitly not adopted full 2D corner measurements.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import combinations
from types import FunctionType
import hashlib
import json

import cv2
import numpy as np

from . import image_supported_edge_chains_v4_17c as NMS
from .interior_observation import NEIGHBORS


SCHEMA = 'JOINT_GEOMETRY_GUIDED_IMAGE_LINE_PROPOSALS_V4_28'
DEFAULT_GRAPH_CONFIGURATION = dict(maximum_beam_states=64, reported_joint_alternatives=8,
    missing_edge_cost=6., image_quality_weight=1., incident_graph_weight=1.,
    shared_axis_concurrency_weight=1., coarse_guide_weight=.10,
    B_or_I_anchor_displacement_weight=.10)
RECIPE = dict(schema=SCHEMA, proposal_only=True, new_CAL_fit=False,
    measured_line_source='frozen17c actual image-gradient NMS/TLS; no pose line as measurement',
    guide_sources=['original17c', 'physical VP through measured BI anchor', 'measured BI endpoint direction'],
    allowed_pose_guides='original input coarse/fixed uniform proposal bank only; no e/GT/error-selected pose',
    guide_dependency='one normalized weak guide energy per graph; not independent image evidence',
    native_axis='dominant component of native edge delta @ that guide physical Q; verified one-axis edge',
    candidate_policy='all actual candidates archived; bounded beam is approximate engineering ranking',
    mask_role='actual inside/outside occupancy plus conditional face role, not parallelism alone',
    dimensions='projected named model-edge offset audits only; no exact ruler/corner hard lock',
    rounded_corner='exclude existing11px around candidate BI junction; refit actual support, extend straight lines',
    precise_BI_or_U_corner_adoption=False, U_intersections=False,
    states_or_old_points_changed=False, new_pose_NN_or_GT_calls=0,
    calibrated_probability_confidence_or_independent_likelihood=False,
    two_user_selected_posthoc_cases_not_population_performance=True)


def _finite(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite ' + name + ' required')
    return a


def _rotation(value, name):
    a = _finite(value, (3, 3), name)
    if not np.allclose(a.T@a, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(a), 1., atol=1e-7):
        raise ValueError('Proper ' + name + ' required')
    return a


def _config(configuration):
    options = {} if configuration is None else dict(configuration)
    if set(options) - {'line_configuration', 'graph_configuration'}:
        raise ValueError('Unknown joint geometry proposal configuration')
    line = NMS._config(options.get('line_configuration'))
    graph = {**DEFAULT_GRAPH_CONFIGURATION, **options.get('graph_configuration', {})}
    if set(graph) != set(DEFAULT_GRAPH_CONFIGURATION) or any(not np.isfinite(v) or v <= 0 for v in graph.values()):
        raise ValueError('Positive declared graph parameters required')
    for key in ('maximum_beam_states', 'reported_joint_alternatives'):
        if int(graph[key]) != graph[key]:
            raise ValueError('Integer graph budget required')
    return line, graph


def _digest(value):
    def encode(v):
        if isinstance(v, dict): return {k:encode(x) for k,x in v.items()}
        if isinstance(v, (tuple, list)): return [encode(x) for x in v]
        if isinstance(v, np.ndarray): return encode(v.tolist())
        if isinstance(v, np.generic): return v.item()
        return v
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _geometry(geometry):
    xyz = _finite(geometry['native_xyz'], (8, 3), 'native model')
    R = _rotation(geometry['R_cf'], 'native camera rotation')
    Q = _rotation(geometry['physical_registration_Q'], 'physical registration Q')
    K = _finite(geometry['K'], (3, 3), 'intrinsics')
    if abs(np.linalg.det(K)) <= 1e-12: raise ValueError('Nonsingular K required')
    center = _finite(geometry['centroid'], (3,), 'camera model center')
    t = center-R@xyz.mean(axis=0)
    camera = xyz@R.T+t
    h = camera@K.T
    projection = {}
    for k in range(8):
        if camera[k, 2] > 1e-9 and abs(h[k, 2]) > 1e-9:
            projection[k] = h[k, :2]/h[k, 2]
    vanishing = [K@R@Q[:, axis] for axis in range(3)]
    # Homogeneous vectors are unit-normalized; infinity is represented by w=0.
    vanishing = [v/np.linalg.norm(v) for v in vanishing]
    return dict(xyz=xyz, R=R, Q=Q, K=K, center=center, t=t,
                projection=projection, vanishing=vanishing,
                physical_R=R@Q, face_context=NMS.D._face_context(geometry))


def vanishing_direction_at(homogeneous_vp, image_xy, *, sign_reference=None):
    """Scale/sign invariant finite or infinite VP direction, at a pixel anchor."""
    v = _finite(homogeneous_vp, (3,), 'homogeneous vanishing point')
    if np.linalg.norm(v) < 1e-15: raise ValueError('Nonzero homogeneous vanishing point required')
    v = v/np.linalg.norm(v)
    a = _finite(image_xy, (2,), 'image anchor')
    direction = v[:2]-v[2]*a
    if np.linalg.norm(direction) < 1e-12: raise ValueError('Anchor at vanishing point has no line direction')
    direction /= np.linalg.norm(direction)
    if sign_reference is not None:
        ref = _finite(sign_reference, (2,), 'edge direction sign reference')
        if direction@ref < 0: direction = -direction
    elif direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return direction


def _edge_axis(g, k, mate):
    physical_delta = (g['xyz'][mate]-g['xyz'][k])@g['Q']
    axis = int(np.argmax(np.abs(physical_delta)))
    length = abs(float(physical_delta[axis]))
    remainder = np.delete(physical_delta, axis)
    if length <= 1e-10 or np.max(np.abs(remainder)) > max(1e-8, length*1e-6):
        raise ValueError('Registered native edge is not a single physical-axis edge')
    return axis


def prepare_joint_geometry_guides(geometry, states, visible_measurements, *, image_binding,
                                  configuration=None, pose_guides=None):
    cfg, graph_cfg = _config(configuration)
    states = [NMS.V3._state(s) for s in states]
    hidden = list(geometry['initial_self_hidden'])
    if len(states) != 8 or len(hidden) != 8: raise ValueError('Eight fixed states/initial-S flags required')
    hw = tuple(geometry['image_hw'])
    if tuple(image_binding.get('image_hw', ())) != hw: raise ValueError('Bound image canvas required')
    NMS.D._sha(image_binding.get('sha256'), 'image')
    points = NMS.G._eligible_points(visible_measurements, states, hidden, hw)
    initial = _geometry(geometry)
    bank = [('ORIGINAL_INITIAL', geometry, initial)]
    names = {'ORIGINAL_INITIAL'}
    for item in pose_guides or []:
        name = str(item['guide_id'])
        if name in names: raise ValueError('Unique fixed pose-guide IDs required')
        if item.get('source_kind') not in ('ORIGINAL_TASK_COARSE', 'UNIFORM_INITIAL_PROPOSAL_BANK'):
            raise ValueError('Only original task/fixed uniform input proposal guides allowed; no selected e/GT pose')
        trial = item['geometry']
        if tuple(trial['image_hw']) != hw or not np.allclose(trial['K'], geometry['K'], atol=0, rtol=0):
            raise ValueError('Pose guide camera binding differs')
        bank.append((name, trial, _geometry(trial))); names.add(name)
    proposals, skipped = [], []
    for k in range(8):
        # State/initial-S gates precede all per-observation coordinate fields.
        if hidden[k] or states[k] not in ('B', 'I') or k not in points: continue
        point = points[k]['record']; anchor = np.asarray(point['xy'], float)
        for mate in NEIGHBORS[k]:
            if hidden[mate] or states[mate] == 'S':
                skipped.append(dict(native_corner=k, native_neighbor=mate, reason='S_EDGE_GATED_BEFORE_POINT_OR_PIXEL_QUERY'))
                continue
            original_mate = initial['projection'].get(mate)
            for name, trial, g in bank:
                if k not in g['projection'] or mate not in g['projection'] or original_mate is None: continue
                guide_delta = g['projection'][mate]-g['projection'][k]
                length = float(np.linalg.norm(guide_delta))
                if length < 10: continue
                face = NMS.D._edge_face_role(g['face_context'], k, mate)
                if face['role'] not in ('INTERNAL_TWO_VISIBLE_FACES', 'OUTLINE_VISIBLE_HIDDEN_FACES'): continue
                axis = _edge_axis(g, k, mate)
                direction = vanishing_direction_at(g['vanishing'][axis], anchor, sign_reference=guide_delta)
                proposal = dict(guide_id=name, guide_kind='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR',
                    native_corner=k, native_neighbor=mate, native_edge=sorted([k, mate]),
                    physical_axis=axis, vanishing_point_h=g['vanishing'][axis].tolist(),
                    anchor_xy=anchor.tolist(), direction_xy=direction.tolist(), model_length_px=length,
                    search_anchor_kind='OBSERVED_BI_REFERENCE_BAND', observed_BI_reference_xy=anchor.tolist(),
                    original_observation_source=point['source'], model_anchor_xy=g['projection'][k].tolist(),
                    anchor_normal_band_px=[-cfg['normal_search_radius_px'], cfg['normal_search_radius_px']],
                    anchor_band_is_engineering_search_reach_not_probability=True,
                    original_unknown_mate_model_xy=(original_mate.tolist() if mate not in points else None),
                    face_association=face, geometry_pixel_projection_is_not_measurement=True)
                proposals.append(proposal)
                if np.linalg.norm(g['projection'][k]-anchor) > 1e-8:
                    model_anchor = g['projection'][k]
                    alternate = dict(proposal, search_anchor_kind='MODEL_PROJECTION_SEARCH_ONLY_BAND',
                        anchor_xy=model_anchor.tolist(),
                        direction_xy=vanishing_direction_at(g['vanishing'][axis],model_anchor,
                            sign_reference=guide_delta).tolist())
                    proposals.append(alternate)
            if mate in points and k in initial['projection'] and mate in initial['projection']:
                delta = np.asarray(points[mate]['record']['xy'])-anchor
                length = float(np.linalg.norm(delta))
                if length >= 10:
                    # Image endpoint direction provides an alternative to a wrong coarse R.
                    face = NMS.D._edge_face_role(initial['face_context'], k, mate)
                    if face['role'] in ('INTERNAL_TWO_VISIBLE_FACES', 'OUTLINE_VISIBLE_HIDDEN_FACES'):
                        proposals.append(dict(guide_id='MEASURED_BI_ENDPOINTS', guide_kind='MEASURED_BI_ENDPOINT_DIRECTION',
                            native_corner=k, native_neighbor=mate, native_edge=sorted([k,mate]),
                            physical_axis=_edge_axis(initial,k,mate), vanishing_point_h=None,
                            anchor_xy=anchor.tolist(), direction_xy=(delta/length).tolist(), model_length_px=length,
                            search_anchor_kind='OBSERVED_BI_REFERENCE_BAND', observed_BI_reference_xy=anchor.tolist(),
                            original_observation_source=point['source'],
                            original_unknown_mate_model_xy=None, face_association=face,
                            geometry_pixel_projection_is_not_measurement=True))
    # Same physical R/center after Q rebinding is an alias, not new direction evidence.
    aliases = []
    for (a, _, ga), (b, _, gb) in combinations(bank, 2):
        if (np.allclose(ga['physical_R'], gb['physical_R'], atol=1e-8, rtol=0)
                and np.allclose(ga['center'], gb['center'], atol=1e-8, rtol=0)):
            aliases.append([a, b])
    return dict(schema='JOINT_GEOMETRY_GUIDE_BANK_V4_28', guide_proposals=proposals, skipped_edges=skipped,
        line_configuration=cfg, graph_configuration=graph_cfg, image_binding=deepcopy(dict(image_binding)),
        original_geometry_digest=_digest({k:geometry[k] for k in
            ('native_xyz','R_cf','centroid','physical_registration_Q','K','image_hw','initial_self_hidden')}),
        same_physical_pose_aliases=aliases, pose_guide_count=len(bank),
        S_or_U_observation_coordinate_reads=0, pose_or_image_calls=0)


def _guided_edge(image, mask, lab, gradients, proposal, points, cfg, visibility_context=None):
    k, mate = proposal['native_corner'], proposal['native_neighbor']
    anchor = np.asarray(proposal['anchor_xy'])
    direction = np.asarray(proposal['direction_xy'])
    length = proposal['model_length_px']
    # Search-only coordinates set the frozen TLS seed, never the reported line.
    search_projection = {k:anchor, mate:anchor+direction*length}
    actual_unknown = proposal['original_unknown_mate_model_xy']
    original_lattice = NMS._lattice_strip
    query_audit=[]
    def lattice(a, ignored_direction, lo, hi, radius, hw, ignored_unknown, context):
        xy,along=original_lattice(anchor, direction, lo, hi, radius, hw,
            None if actual_unknown is None else np.asarray(actual_unknown), context)
        before=len(xy)
        if visibility_context is not None and len(xy):
            keep=visibility_context.query_allowed(k,mate,xy,context_radius_px=context)
            xy,along=xy[keep],along[keep]
        query_audit.append(dict(search_anchor_xy=anchor.tolist(),search_direction_xy=direction.tolist(),
            along_interval_px=[float(lo),float(hi)],normal_radius_px=float(radius),
            lattice_pixels_before_additional_exclusion=before,lattice_pixels_after_exclusion=len(xy)))
        return xy,along
    namespace = dict(NMS._edge_chains.__globals__)
    namespace['_lattice_strip'] = lattice
    call = FunctionType(NMS._edge_chains.__code__, namespace)
    out = call(image, mask, lab, gradients, k, mate, points[k]['record'],
        points[mate]['record'] if mate in points else None, search_projection,
        proposal['face_association'], cfg)
    out.update(guide_proposal=deepcopy(proposal), frozen17c_NMS_TLS_math=True,
        proposed_direction_is_not_measured_line=True, original_unknown_mate_exclusion_preserved=True,
        actual_query_strip_audit=query_audit)
    for chain in [*out.get('chains',[]),*out.get('rejected_chains',[])]:
        if 'observed_points' not in chain:continue
        normal,offset,actual=_actual_line(chain)
        values=(actual-anchor)@direction
        chain['legacy_observed_endpoint_reference_support']=chain.pop('endpoint_support',None)
        chain['query_strip_actual_support']=dict(along_min_px=float(values.min()),along_max_px=float(values.max()),
            query_anchor_xy=anchor.tolist(),query_direction_xy=direction.tolist(),
            measured_BI_reference_xy=proposal['observed_BI_reference_xy'],
            endpoint_reference_normal_shift_px=float(abs(normal@np.asarray(proposal['observed_BI_reference_xy'])+offset)))
    return out


def _external_context(pallet_mask,image_hw,cfg):
    """Global exterior contours distinguish mask holes from its exterior."""
    from scipy.spatial import cKDTree
    mask=np.asarray(pallet_mask,float)
    if mask.shape!=tuple(image_hw) or not np.isfinite(mask).all():raise ValueError('Same-canvas finite target mask required')
    binary=(mask>=.5).astype(np.uint8)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    points=np.concatenate([c.reshape(-1,2) for c in contours]) if contours else np.empty((0,2))
    h,w=image_hw;context=cfg['boundary_context_px']
    if len(points):
        points=points[(points[:,0]>=context)&(points[:,0]<w-1-context)
            &(points[:,1]>=context)&(points[:,1]<h-1-context)]
    _,labels=cv2.connectedComponents(binary,connectivity=8)
    return dict(tree=cKDTree(points) if len(points) else None,labels=labels,
                real_external_contour_points=len(points),canvas_closing_edges_excluded=True)


def _line_external_context(node,context,cfg):
    actual=np.asarray(node['actual_inliers']);normal=np.asarray(node['normal'])
    if context is None:return dict(available=False,external_support_fraction=None)
    distance=context['tree'].query(actual)[0] if context['tree'] is not None else np.full(len(actual),np.inf)
    labels=context['labels'];h,w=labels.shape
    side_labels=[]
    for sign in (-1,1):
        xy=np.rint(actual+sign*normal*cfg['boundary_context_px']).astype(int)
        xy=np.clip(xy,[0,0],[w-1,h-1]);side_labels.extend(labels[xy[:,1],xy[:,0]].tolist())
    foreground=np.asarray([x for x in side_labels if x>0],int)
    dominant=float(np.bincount(foreground).max()/len(foreground)) if len(foreground) else 0.
    proximity=float(np.mean(distance<=cfg['boundary_context_px']))
    return dict(available=True,external_support_fraction=proximity*dominant,
        actual_support_near_global_external_fraction=proximity,same_foreground_component_fraction=dominant,
        median_distance_to_real_external_contour_px=float(np.median(distance)) if np.isfinite(distance).any() else None,
        real_external_contour_points=context['real_external_contour_points'],
        mask_holes_not_external=True,canvas_closing_edges_excluded=True,
        independent_occluder_or_native_identity_certified=False)


def _actual_line(record):
    normal = _finite(record['normal'], (2,), 'measured line normal')
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12: raise ValueError('Nonzero measured line normal required')
    offset = float(record['offset'])/norm; normal = normal/norm
    observed = np.asarray(record['observed_points'], float)
    inliers = np.asarray(record['inlier_mask'], bool)
    if observed.ndim != 2 or observed.shape[1] != 2 or inliers.shape != (len(observed),) or not np.isfinite(observed).all():
        raise ValueError('Actual measured support/inlier mask required')
    actual = observed[inliers]
    if len(actual) < 2: raise ValueError('At least two actual inliers required')
    return normal, offset, actual


def _intersection(a, b, minimum_sine):
    normals = np.asarray([a[0], b[0]])
    sine = abs(float(np.linalg.det(normals)))
    if sine < minimum_sine: return None, sine
    return np.linalg.solve(normals, -np.asarray([a[1], b[1]])), sine


def rounded_BI_junction(lines, cfg, image_hw=None):
    """Pure actual-support exclusion/refit. Caller has already gated a B/I owner."""
    if len(lines) != 2: raise ValueError('Two actual incident line records required')
    initial = [_actual_line(line) for line in lines]
    point, sine = _intersection(initial[0], initial[1], cfg['minimum_intersection_sine'])
    out = dict(available=False, initial_line_sine=sine, full_2D_corner_measurement=False,
        adopted=False, actual_corner_pixel_or_RGB_patch_searched=False)
    if point is None: return dict(out, reason='near_parallel_lines_no_precise_junction')
    margin = float(cfg['along_start_px']+cfg['boundary_context_px'])
    fitted, details = [], []
    for normal, offset, actual in initial:
        keep = np.linalg.norm(actual-point, axis=1) >= margin
        selected = actual[keep]
        if len(selected) < cfg['minimum_inlier_count']:
            return dict(out, reason='insufficient_straight_support_after_rounding_exclusion',
                        candidate_extrapolated_intersection_xy=point.tolist())
        center = selected.mean(axis=0)
        _, vectors = np.linalg.eigh((selected-center).T@(selected-center))
        n = vectors[:, 0]
        if n@normal < 0: n = -n
        c = float(-n@center)
        rms = float(np.sqrt(np.mean((selected@n+c)**2)))
        if rms > cfg['maximum_line_rms_px']:
            return dict(out, reason='straight_support_TLS_guard_failed')
        tangent = np.array([-n[1], n[0]])
        fitted.append((n,c,selected))
        details.append(dict(normal=n.tolist(), offset=c, actual_points=selected.tolist(),
            excluded_actual_points=int((~keep).sum()), retained_actual_points=len(selected), RMS_px=rms,
            actual_tangent_interval_px=[float((selected@tangent).min()),float((selected@tangent).max())]))
    extended, final_sine = _intersection(fitted[0], fitted[1], cfg['minimum_intersection_sine'])
    if extended is None: return dict(out, reason='refitted_lines_near_parallel')
    gaps=[]
    for normal,offset,actual in fitted:
        tangent=np.array([-normal[1],normal[0]])
        v=actual@tangent;u=float(extended@tangent)
        gaps.append(float(max(v.min()-u,0.,u-v.max())))
    inside=None if image_hw is None else bool(0<=extended[0]<image_hw[1] and 0<=extended[1]<image_hw[0])
    return dict(out, available=True, candidate_extrapolated_intersection_xy=extended.tolist(),
        initial_extrapolated_intersection_xy=point.tolist(), final_line_sine=final_sine,
        rounding_exclusion_radius_px=margin, straight_actual_support=details,
        extrapolation_beyond_straight_support_px=gaps,finite_support_11px_guard_would_pass=max(gaps)<=margin,
        within_canvas=inside,available_means_computable_not_production_guard_pass=True,
        original_guide_angle_or_complete_KP_guards_not_certified=True,
        interpretation='intersection of extended measured straight edges; no actual sharp-corner pixel asserted')


def _node(record, chain, reference_index, cfg):
    n,c,actual = _actual_line(chain)
    k,mate = int(record['native_corner']), int(record['native_neighbor'])
    mask_plus = np.asarray(chain.get('mask_plus', []), float)
    mask_minus = np.asarray(chain.get('mask_minus', []), float)
    inliers = np.asarray(chain['inlier_mask'], bool)
    if mask_plus.shape == inliers.shape and mask_minus.shape == inliers.shape:
        outside = float(np.mean((np.minimum(mask_plus[inliers],mask_minus[inliers]) <= cfg['maximum_outside'])
            & (np.maximum(mask_plus[inliers],mask_minus[inliers]) >= cfg['minimum_inside'])))
    else: outside = .5
    contrast = np.asarray(chain.get('RGB_contrast', []), float)
    contrast = float(np.median(contrast[inliers])) if contrast.shape == inliers.shape else cfg['minimum_color_transition']
    image_energy = ((1-min(1.,float(chain['inlier_fraction'])))
        +(1-min(1.,float(chain['inlier_span_fraction'])))
        +(float(chain['inlier_rms_px'])/cfg['maximum_line_rms_px'])**2
        +1/(1+max(0.,contrast)/max(1e-9,cfg['minimum_color_transition'])))
    return dict(name=f'{min(k,mate)}-{max(k,mate)}/{reference_index}', native_edge=sorted([k,mate]),
        native_corner=k, native_neighbor=mate, normal=n.tolist(), offset=c,
        actual_inliers=actual.tolist(), line_record=chain,
        support_anchor_xy=actual.mean(axis=0).tolist(), actual_outline_fraction=outside,
        image_quality_energy=float(image_energy), provenance=[reference_index],
        old_corrected_endpoint_normal_guard_pass=bool(chain.get('corrected_endpoint_normal_guard_pass',True)),
        old_corrected_endpoint_normal_guard_limit_px=float(cfg['along_start_px']+cfg['boundary_context_px']),
        original_corner_shift_cap_px=float(cfg['maximum_corner_shift_px']),
        production_guard_bypassed_only_for_unadopted_line_candidate=not bool(chain.get('corrected_endpoint_normal_guard_pass',True)))


def _score_graph(selected, points, states, guide_bank, cfg, graph_cfg):
    image_energy = sum(n['image_quality_energy'] if n is not None else graph_cfg['missing_edge_cost'] for n in selected)
    nodes = [n for n in selected if n is not None]
    margin = float(cfg['normal_search_radius_px'])
    incident_terms = []
    junctions = []
    for k in sorted(points):
        if states[k] not in ('B','I'): continue
        incident = [n for n in nodes if k in n['native_edge']]
        if not incident: continue
        # A B anchor should have actual exterior context, not only parallel slats.
        role_cost=(1-max(n['actual_outline_fraction']*(n['global_external_context']['external_support_fraction']
            if n['global_external_context']['available'] else .5) for n in incident)) if states[k]=='B' else 0.
        intersections = []
        for a,b in combinations(incident,2):
            key=tuple(sorted([a['name'],b['name']]))
            cache=guide_bank['_cached_intersections']
            if key not in cache:
                cache[key]=_intersection((a['normal'],a['offset']), (b['normal'],b['offset']), cfg['minimum_intersection_sine'])
            p,sine=cache[key]
            if p is not None: intersections.append(p)
        scatter = 0.; anchor_cost = 0.
        if intersections:
            center = np.median(intersections,axis=0)
            scatter = float(np.mean(np.sum((np.asarray(intersections)-center)**2,axis=1))/margin**2)
            anchor_cost = float(np.sum((center-np.asarray(points[k]['record']['xy']))**2)/margin**2)
            junctions.append(dict(native_corner=k, state=states[k], line_pair_intersections_xy=np.asarray(intersections).tolist(),
                intersection_scatter_px2=scatter*margin**2,
                observed_BI_to_intersection_shift_px=float(np.sqrt(anchor_cost)*margin),
                old_12px_corner_cap_would_pass=bool(np.sqrt(anchor_cost)*margin<=cfg['maximum_corner_shift_px']),
                full_2D_corner_measurement=False))
        incident_terms.append(role_cost+scatter+graph_cfg['B_or_I_anchor_displacement_weight']*anchor_cost)
    # Each original guide contributes a NORMALIZED mean, not N independent priors.
    guide_costs = {}
    for node in nodes:
        for name,value in guide_bank['_cached_node_guide_costs'][node['name']].items():
            guide_costs.setdefault(name,{})[node['name']]=value
    means = {name:float(np.mean(list(values.values()))) for name,values in guide_costs.items() if values}
    weak = min(means.values()) if means else 0.
    # Concurrency among independently measured named parallel edges is evaluated
    # in normalized image homogeneous coordinates, including VP at infinity.
    concurrency = []
    measured_axis_rays={}
    axis_edges=guide_bank['_cached_edge_axes']
    for axis in range(3):
        group = [n for n in nodes if axis_edges.get(tuple(n['native_edge']))==axis]
        if len(group)<2:continue
        lines = np.array([n['normal']+[n['offset']] for n in group],float)
        h,w = guide_bank['image_binding']['image_hw']
        transform = np.array([[w,0,w/2],[0,h,h/2],[0,0,1.]])
        normalized = lines@transform
        normalized /= np.linalg.norm(normalized[:,:2],axis=1)[:,None]
        _,_,vectors = np.linalg.svd(normalized,full_matrices=True)
        vp = transform@vectors[-1]
        ray=np.linalg.solve(guide_bank['_cached_K'],vp)
        if np.linalg.norm(ray)>1e-12:measured_axis_rays[axis]=ray/np.linalg.norm(ray)
        residual = []
        for n in group:
            try: residual.append(float(np.asarray(n['normal'])@vanishing_direction_at(vp,n['support_anchor_xy']))**2)
            except ValueError:pass
        if residual and len(group)>=3:concurrency.append(float(np.mean(residual)))
    orthogonality=[float(a@b)**2 for a,b in combinations(measured_axis_rays.values(),2)]
    orthogonality_cost=float(np.mean(orthogonality)) if orthogonality else 0.
    cost = (graph_cfg['image_quality_weight']*image_energy
        +graph_cfg['incident_graph_weight']*sum(incident_terms)
        +graph_cfg['shared_axis_concurrency_weight']*(sum(concurrency)+orthogonality_cost)
        +graph_cfg['coarse_guide_weight']*weak)
    return dict(total_engineering_energy=float(cost), image_quality_energy=float(image_energy),
        incident_graph_energy=float(sum(incident_terms)), shared_axis_concurrency_energy=float(sum(concurrency)),
        image_measured_axis_orthogonality_energy=orthogonality_cost,
        independently_measured_axis_ray_count=len(measured_axis_rays),
        weak_normalized_coarse_guide_energy=weak, normalized_guide_means=means,
        guide_term_count=1 if means else 0, junctions=junctions)


def rank_joint_geometry_line_graph(chain_observation, geometry, states, visible_measurements, *,
                                   image_binding, configuration=None, pose_guides=None,pallet_mask=None,
                                   visibility_context=None):
    cfg, graph_cfg = _config(configuration)
    guides = prepare_joint_geometry_guides(geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides)
    states = [NMS.V3._state(s) for s in states]
    hidden = list(geometry['initial_self_hidden'])
    points = NMS.G._eligible_points(visible_measurements,states,hidden,tuple(geometry['image_hw']))
    context=None if pallet_mask is None else _external_context(pallet_mask,geometry['image_hw'],cfg)
    groups, dedup,visibility_rejected = {}, {},[]
    for ri, record in enumerate(chain_observation['edge_records']):
        k,mate = int(record['native_corner']),int(record['native_neighbor'])
        if hidden[k] or hidden[mate] or states[k] not in ('B','I') or states[mate]=='S' or k not in points:continue
        if mate not in NEIGHBORS[k]:raise ValueError('Only physical incident native edges allowed')
        accepted=list(record.get('chains',[]))
        # Actual TLS/RGB-passing lines rejected solely by an old corrected-q
        # guard remain diagnostic proposals. No production KP guard is changed.
        rejected=[c for c in record.get('rejected_chains',[])
                  if c.get('reason')=='existing_corrected_endpoint_normal_guard_failed']
        for ci,chain in enumerate(accepted+rejected):
            node = _node(record,chain,f'{ri}:{ci}',cfg)
            actual = np.asarray(node['actual_inliers'])
            if visibility_context is not None and not np.all(visibility_context.query_allowed(k,mate,actual,
                    context_radius_px=max(cfg['RGB_probe_px'],cfg['boundary_context_px']))):
                visibility_rejected.append(dict(line_name=node['name'],native_edge=node['native_edge'],
                    reason='historical_support_not_allowed_by_current_unknown_or_independent_occluder_context',
                    record_reference=f'{ri}:{ci}'))
                continue
            node['global_external_context']=_line_external_context(node,context,cfg)
            order = np.lexsort((actual[:,1],actual[:,0]))
            canonical=np.asarray(node['normal'])
            sign=1. if canonical[np.argmax(np.abs(canonical))]>=0 else -1.
            support_id=_digest(dict(edge=node['native_edge'],actual=np.round(actual[order],4)))
            key = _digest(dict(edge=node['native_edge'],actual=actual[order],
                normal=canonical*sign,offset=node['offset']*sign))
            node.update(support_id=support_id,exact_fit_id=key,
                shared_support_fit_variants_are_alternatives_not_independent=True)
            if key in dedup:
                dedup[key]['provenance'].append(f'{ri}:{ci}');continue
            dedup[key]=node;groups.setdefault(tuple(node['native_edge']),[]).append(node)
    guide_costs={}
    for node in dedup.values():
        costs={}
        for p in guides['guide_proposals']:
            if p['vanishing_point_h'] is None or p['native_edge']!=node['native_edge']:continue
            try:
                d=vanishing_direction_at(p['vanishing_point_h'],node['support_anchor_xy'])
                costs[p['guide_id']]=float(np.asarray(node['normal'])@d)**2
            except ValueError:pass
        guide_costs[node['name']]=costs
    guides['_cached_node_guide_costs']=guide_costs
    guides['_cached_edge_axes']={tuple(p['native_edge']):p['physical_axis'] for p in guides['guide_proposals']
        if p['guide_id']=='ORIGINAL_INITIAL'}
    guides['_cached_K']=np.asarray(geometry['K'],float)
    guides['_cached_intersections']={}
    beam = [([],_score_graph([],points,states,guides,cfg,graph_cfg))]
    explored, truncated = 0,False
    for edge in sorted(groups):
        expanded = []
        for selected,_ in beam:
            for node in [*groups[edge],None]:
                path=selected+[node]
                score=_score_graph(path,points,states,guides,cfg,graph_cfg)
                expanded.append((path,score));explored+=1
        expanded.sort(key=lambda pair:(pair[1]['total_engineering_energy'],
            tuple(n['name'] if n is not None else '~' for n in pair[0])))
        if len(expanded)>graph_cfg['maximum_beam_states']:truncated=True
        beam=expanded[:int(graph_cfg['maximum_beam_states'])]
    alternatives=[]
    for selected,score in beam[:int(graph_cfg['reported_joint_alternatives'])]:
        alternatives.append(dict(selected_line_names=[n['name'] for n in selected if n is not None],**score))
    selected_names=set(alternatives[0]['selected_line_names']) if alternatives else set()
    selected=[node for node in dedup.values() if node['name'] in selected_names]
    rounded=[]
    for k in sorted(points):
        if states[k] not in ('B','I'):continue
        incident=[n for n in selected if k in n['native_edge']]
        for a,b in combinations(incident,2):
            rounded.append(dict(native_corner=k,state=states[k],line_names=[a['name'],b['name']],
                **rounded_BI_junction([a['line_record'],b['line_record']],cfg,geometry['image_hw'])))
    for name in ('_cached_node_guide_costs','_cached_edge_axes','_cached_K','_cached_intersections'):guides.pop(name)
    return dict(schema='JOINT_NATIVE_IMAGE_LINE_GRAPH_RANKING_V4_28',
        selected_line_names=sorted(selected_names), line_candidates=list(dedup.values()),
        joint_alternatives=alternatives, rounded_BI_junctions=rounded,
        graph_configuration=graph_cfg, original_guide_bank=guides,
        unique_physical_edge_count=len(groups), actual_unique_line_candidates=len(dedup),
        joint_states_evaluated=explored, beam_truncated=truncated,
        historical_line_candidates_excluded_by_visibility=visibility_rejected,
        global_optimum_or_calibrated_confidence_claim=False, U_junctions_or_measurements_created=0,
        ranking_is_engineering_proposal_only=True, precise_corner_or_PnP_adoption=False)


def observe_joint_geometry_guided_lines(image_bgr,pallet_mask,geometry,states,visible_measurements,*,
                                        image_binding,configuration=None,base_observation=None,pose_guides=None,
                                        visibility_context=None):
    cfg,_ = _config(configuration)
    guides = prepare_joint_geometry_guides(geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides)
    if base_observation is None:
        observe=NMS.observe_image_supported_edge_chains
        if visibility_context is not None:
            def gated_edge(*args):
                k,mate=args[4],args[5]
                def lattice(*values):
                    xy,along=NMS._lattice_strip(*values)
                    keep=visibility_context.query_allowed(k,mate,xy,context_radius_px=values[-1])
                    return xy[keep],along[keep]
                namespace=dict(NMS._edge_chains.__globals__);namespace['_lattice_strip']=lattice
                return FunctionType(NMS._edge_chains.__code__,namespace)(*args)
            namespace=dict(observe.__globals__);namespace['_edge_chains']=gated_edge
            observe=FunctionType(observe.__code__,namespace)
        base=observe(image_bgr,pallet_mask,geometry,states,visible_measurements,
                     image_binding=image_binding,configuration=cfg)
    else:
        base=deepcopy(base_observation)
        if (base.get('schema')!='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C'
                or base.get('image_binding',{}).get('sha256')!=image_binding['sha256']
                or base['image_binding'].get('image_pixel_sha256')!=NMS.G.OLD.image_sha256(np.asarray(image_bgr))):
            raise ValueError('Base17c must be exact same-image observation')
    image=np.asarray(image_bgr);mask=np.asarray(pallet_mask,float)
    if (image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or mask.shape!=image.shape[:2]
            or tuple(geometry['image_hw'])!=image.shape[:2] or not np.isfinite(mask).all()):
        raise ValueError('Actual bound uint8 BGR image and finite occupancy mask required')
    if np.any((mask<0)|(mask>1)):raise ValueError('Occupancy mask must be within0..1')
    hidden=geometry['initial_self_hidden']
    points=NMS.G._eligible_points(visible_measurements,states,hidden,image.shape[:2])
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients=(cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.)
    extra=[_guided_edge(image,mask,lab,gradients,p,points,cfg,visibility_context) for p in guides['guide_proposals']]
    combined=dict(base,edge_records=base['edge_records']+extra)
    joint=rank_joint_geometry_line_graph(combined,geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides,pallet_mask=mask,visibility_context=visibility_context)
    return dict(schema=SCHEMA,recipe=deepcopy(RECIPE),image_binding=deepcopy(base['image_binding']),
        base_observation=base,additional_guided_edge_records=extra,joint_graph=joint,
        original17c_candidates_preserved=True,states_changed=False,old_points_changed=0,
        current_visibility_context_applied_before_base_and_additional_queries=visibility_context is not None and base_observation is None,
        S_or_U_local_observation_coordinate_reads=0,U_or_S_anchor_corner_queries=0,
        pose_or_NN_calls=0,GT_inputs=False,new_likelihood_or_default_adopted=False)
