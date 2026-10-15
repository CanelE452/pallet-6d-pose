"""Opt-in gap-linked actual-image NMS chains for conditionally associated native B/I edges.

The initial model proposes a search strip and native edge association only.
Pixels are sampled on the image lattice and localized along the measured image
gradient, rather than on parallel rows of a model-normal search window. Every
candidate chain and ambiguity remains explicit. This producer does not alter
states, old corner coordinates/covariances, or construct precise corner inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import cv2
import numpy as np

from . import boundary_refine as F
from . import face_aware_partial_bi_observations_v4_13d as D
from . import joint_ot_pnp as V3
from . import visible_bi_global_edge_correction_v4_9c as G
from .interior_observation import NEIGHBORS


DEFAULT_CONFIGURATION = deepcopy(D.DEFAULT_CONFIGURATION)
# Pixel quantization is an engineering floor, not a fitted total-error model.
PIXEL_LOCALIZATION_FLOOR_PX = 1.


def _config(configuration):
    return D._config(configuration)


def _sample(array, xy):
    # OpenCV remap requires each source/destination axis < SHRT_MAX. This
    # destination-only batching changes no coordinate, interpolation or data.
    xy = np.asarray(xy)
    if len(xy) == 0:
        return np.empty((0,)+np.asarray(array).shape[2:], dtype=float)
    maximum_destination_rows = np.iinfo(np.int16).max - 1
    pieces = [G._sample(array, xy[start:start+maximum_destination_rows, None, :])[:, 0]
              for start in range(0, len(xy), maximum_destination_rows)]
    return np.concatenate(pieces, axis=0)


def _image_gradient_normal(gx, gy, seed_normal):
    """Principal LAB structure-tensor direction measured on the image lattice."""
    a = np.sum(gx*gx, axis=-1)
    b = np.sum(gx*gy, axis=-1)
    c = np.sum(gy*gy, axis=-1)
    angle = .5*np.arctan2(2*b, a-c)
    normal = np.column_stack([np.cos(angle), np.sin(angle)])
    normal[(normal@seed_normal) < 0] *= -1
    strength = np.sqrt(np.maximum(.5*(a+c+np.sqrt((a-c)**2+4*b*b)), 0))
    return normal, strength


def _lattice_strip(anchor, direction, lo, hi, radius, image_hw, unknown_mate, context):
    normal = np.array([-direction[1], direction[0]])
    corners = np.array([anchor+u*direction+v*normal for u in (lo, hi) for v in (-radius, radius)])
    h, w = image_hw
    left, top = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    right, bottom = np.minimum(np.ceil(corners.max(axis=0)).astype(int), [w-1, h-1])
    if right < left or bottom < top: return np.empty((0, 2)), np.empty(0)
    yy, xx = np.mgrid[top:bottom+1, left:right+1]
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    u = (xy-anchor)@direction
    v = (xy-anchor)@normal
    keep = (u >= lo) & (u <= hi) & (np.abs(v) < radius)
    # This gate precedes every RGB/mask remap, including gradient NMS probes.
    if unknown_mate is not None:
        keep &= np.linalg.norm(xy-unknown_mate, axis=-1) >= 2*context
    keep &= ((xy[:, 0] >= context) & (xy[:, 0] < w-1-context) &
             (xy[:, 1] >= context) & (xy[:, 1] < h-1-context))
    return xy[keep], u[keep]


def _chain_uncertainty(actual, normal, rms, cfg):
    tangent = np.array([-normal[1], normal[0]])
    values = actual@tangent
    span = float(np.ptp(values))
    # One-pixel displacement at both ends cannot produce a zero angle budget.
    # No sqrt(N) reduction: NMS neighbours and RGB/mask conditioning correlate.
    sigma_n = max(PIXEL_LOCALIZATION_FLOOR_PX, float(rms))
    sigma_angle = float(np.arctan2(np.sqrt(2)*sigma_n, max(span, 1e-12)))
    return dict(parameterization='normal distance at observed inlier centroid and line angle',
                anchor_xy=actual.mean(axis=0).tolist(), normal_sigma_px=sigma_n,
                angle_sigma_rad=sigma_angle, angle_sigma_deg=float(np.rad2deg(sigma_angle)),
                pixel_localization_floor_px=PIXEL_LOCALIZATION_FLOOR_PX,
                rule='atan2(sqrt(2)*max(1px,RMS), actual inlier tangent span); no sample-count precision boost',
                calibrated=False, predictive_coverage=False,
                mask_model_guide_selection_conditioning_included_in_CAL_model=False,
                normal_angle_independence_claimed=False,
                intercept_must_not_be_double_counted_with_archived_partial_normal=True)


def _collinear_component_hypotheses(xy, observed, valid, component, direction, seed_normal, cfg):
    """Link actual peak components across <=11px gaps; never create pixels.

    Overlapping parallel components cannot be joined. Distinct admissible paths
    remain alternatives; a combinatorial overflow explicitly abstains instead
    of silently retaining a truncated set.
    """
    from scipy.spatial import cKDTree
    labels = [int(label) for label in np.unique(component[valid]) if label]
    groups = {label:np.flatnonzero(valid & (component==label)) for label in labels}
    raw = [dict(component_ids=[label], indices=indices.tolist(), max_gap_px=0.)
           for label,indices in groups.items() if len(indices)>=2]
    margin = float(cfg['along_start_px']+cfg['boundary_context_px'])
    audit = dict(rule='nonoverlapping approximately collinear actual NMS components; gap <= existing4+7px',
                 maximum_gap_px=margin, actual_components=len(labels), joined_hypotheses=0,
                 invented_bridge_points=0, all_parallel_alternatives_preserved=True,
                 combinatorial_overflow_abstained=False, links=[])
    if not raw: return [],audit
    def fitted(indices):
        p=observed[indices];center=p.mean(axis=0);_,vectors=np.linalg.eigh((p-center).T@(p-center))
        normal=vectors[:,0]
        if normal@seed_normal<0:normal=-normal
        angle=float(np.degrees(np.arccos(np.clip(normal@seed_normal,-1.,1.))))
        rms=float(np.sqrt(np.mean(((p-center)@normal)**2)))
        return angle,rms
    searchable=[]
    for record in raw:
        label=record['component_ids'][0];indices=np.asarray(record['indices']);angle,rms=fitted(indices)
        if angle>cfg['maximum_direction_change_deg'] or rms>cfg['maximum_line_rms_px']:continue
        along=observed[indices]@direction
        searchable.append(dict(label=label,indices=indices,lo=float(along.min()),hi=float(along.max()),
            first=observed[indices[np.argmin(along)]],last=observed[indices[np.argmax(along)]]))
    if not searchable:return raw,audit
    tree=cKDTree(np.asarray([r['first'] for r in searchable]));graph={r['label']:[] for r in searchable}
    bylabel={r['label']:r for r in searchable}
    gapmap={}
    for left in searchable:
        for j in tree.query_ball_point(left['last'],margin):
            right=searchable[j]
            if right['lo']<=left['hi']+1e-9:continue
            indices=np.r_[left['indices'],right['indices']]
            angle,rms=fitted(indices)
            if angle>cfg['maximum_direction_change_deg'] or rms>cfg['maximum_line_rms_px']:continue
            gap=float(np.linalg.norm(right['first']-left['last']))
            graph[left['label']].append(right['label']);gapmap[(left['label'],right['label'])]=gap
            audit['links'].append(dict(components=[left['label'],right['label']],actual_endpoint_gap_px=gap,
                along_gap_px=right['lo']-left['hi'],combined_RMS_px=rms))
    joined={};budget=int(cfg['line_resampling'])
    def walk(path):
        indices=np.concatenate([bylabel[label]['indices'] for label in path])
        extensions=[]
        for label in graph[path[-1]]:
            angle,rms=fitted(np.r_[indices,bylabel[label]['indices']])
            if angle<=cfg['maximum_direction_change_deg'] and rms<=cfg['maximum_line_rms_px']:
                extensions.append(label)
        if not extensions:
            if len(path)>1:
                joined[tuple(path)]=dict(component_ids=path,indices=indices.tolist(),
                    max_gap_px=max(gapmap[(a,b)] for a,b in zip(path[:-1],path[1:])))
                if len(joined)>budget:raise OverflowError
            return
        for label in extensions:walk(path+[label])
    try:
        for record in searchable:walk([record['label']])
    except (OverflowError,RecursionError):
        audit['combinatorial_overflow_abstained']=True
        audit['all_parallel_alternatives_preserved']=False
        return [],audit
    audit['joined_hypotheses']=len(joined)
    # Keep raw components too; an extended path is a hypothesis, not a forced association.
    return raw+list(joined.values()),audit


def _edge_chains(image, mask, lab, gradients, k, mate, point, mate_point, projection,
                 face, cfg):
    q = np.asarray(point['xy'], float)
    model_guide = projection[mate]-projection[k]
    model_length = float(np.linalg.norm(model_guide))
    output = dict(native_corner=k, native_neighbor=mate, native_edge=[k, mate],
                  supported=False, chains=[], selected_chain_index=None,
                  face_association=deepcopy(face),
                  native_association_is_conditional_on_initial_geometry=True,
                  actual_native_identity_certified=False,
                  neighbor_image_coordinate_used=mate_point is not None,
                  neighbor_unknown_corner_queries=0, S_local_queries=0,
                  guide_defines_measurement_direction=False,
                  measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT',
                  image_peak_localization='integer image lattice -> image-gradient NMS/subpixel peak -> connected chain TLS',
                  full_2D_corner_created=False, GT_inputs=False)
    if model_length < 10:
        return dict(output, reason='short_model_only_incident_guide')
    seed_normal = np.array([-model_guide[1], model_guide[0]])/model_length
    context = max(cfg['RGB_probe_px'], cfg['boundary_context_px'])
    unknown_mate = None
    if mate_point is None:
        direction = model_guide/model_length
        lo = cfg['along_start_px']
        hi = min(cfg['along_end_px'], .8*model_length, model_length-context)
        unknown_mate = projection[mate]
        whole = False
    else:
        vector = np.asarray(mate_point['xy'], float)-q
        length = float(np.linalg.norm(vector))
        if length < 10: return dict(output, reason='short_visible_endpoint_segment')
        direction = vector/length
        lo, hi = cfg['along_start_px'], length-cfg['along_start_px']
        whole = True
    if hi <= lo: return dict(output, reason='no_actual_search_span_before_mate')
    xy, along = _lattice_strip(q, direction, lo, hi, cfg['normal_search_radius_px'],
                               image.shape[:2], unknown_mate, context)
    output.update(whole_edge_search=whole, lattice_pixels_queried=len(xy),
                  normal_search_radius_px=cfg['normal_search_radius_px'],
                  along_interval_px=[float(lo), float(hi)],
                  mask_context_kind=('TARGET_TARGET_INTERNAL' if face['role']=='INTERNAL_TWO_VISIBLE_FACES'
                                     else 'TARGET_BACKGROUND_OUTLINE'))
    if len(xy) == 0: return dict(output, reason='empty_safe_image_lattice')
    gx, gy = (_sample(g, xy) for g in gradients)
    image_normal, strength = _image_gradient_normal(gx, gy, seed_normal)
    nms_step = PIXEL_LOCALIZATION_FLOOR_PX
    # Every probe remains inside the pre-gated context radius.
    def strength_at(points):
        xx, yy = (_sample(g, points) for g in gradients)
        return _image_gradient_normal(xx, yy, seed_normal)[1]
    minus = strength_at(xy-image_normal*nms_step)
    plus = strength_at(xy+image_normal*nms_step)
    nms = (strength >= minus) & (strength >= plus) & ((strength > minus) | (strength > plus))
    curvature = minus-2*strength+plus
    shift = np.zeros(len(xy))
    concave = curvature < -1e-12
    shift[concave] = .5*(minus[concave]-plus[concave])/curvature[concave]
    # NMS implies an interior quadratic maximum; clipping is recorded, not hidden.
    censored = np.abs(shift) >= .5
    shift = np.clip(shift, -.5, .5)
    observed = xy+shift[:, None]*image_normal*nms_step
    # Same old RGB and occupancy gates, evaluated at the actual image peak.
    xx, yy = (_sample(g, observed) for g in gradients)
    energy = np.linalg.norm(xx*seed_normal[0]+yy*seed_normal[1], axis=-1)
    magnitude = np.sqrt(np.sum(xx*xx+yy*yy, axis=-1))
    alignment = energy/np.maximum(magnitude, 1e-12)
    probe = cfg['RGB_probe_px']
    contrast = np.linalg.norm(_sample(lab, observed+image_normal*probe)-
                              _sample(lab, observed-image_normal*probe), axis=-1)
    mp = _sample(mask, observed+image_normal*probe)
    mm = _sample(mask, observed-image_normal*probe)
    internal = face['role']=='INTERNAL_TWO_VISIBLE_FACES'
    occupancy = ((mp>=cfg['minimum_inside']) & (mm>=cfg['minimum_inside']) if internal else
                 (np.maximum(mp,mm)>=cfg['minimum_inside']) &
                 (np.minimum(mp,mm)<=cfg['maximum_outside']) &
                 (np.abs(mp-mm)>=cfg['minimum_transition']))
    valid = (nms & concave & occupancy & (energy>=cfg['minimum_gradient']) &
             (alignment>=cfg['minimum_gradient_alignment']) &
             (contrast>=cfg['minimum_color_transition']))
    output.update(image_NMS_peak_count=int(nms.sum()), RGB_context_peak_count=int(valid.sum()),
                  subpixel_quadratic_boundary_hit_count=int((valid & censored).sum()),
                  response_is_not_window_clipped=True)
    lattice = np.zeros(mask.shape, np.uint8)
    integer = xy.astype(int)
    lattice[integer[valid, 1], integer[valid, 0]] = 1
    _, labels = cv2.connectedComponents(lattice, connectivity=8)
    component = labels[integer[:, 1], integer[:, 0]]
    line_cfg = dict(along_interval=[0., 1.], huber_delta_px=cfg['maximum_line_rms_px'],
                    maximum_angle_change_deg=cfg['maximum_direction_change_deg'],
                    minimum_inlier_count=cfg['minimum_inlier_count'],
                    minimum_inlier_fraction=cfg['minimum_inlier_fraction'],
                    minimum_inlier_span_fraction=cfg['minimum_inlier_span_fraction'],
                    maximum_line_rms_px=cfg['maximum_line_rms_px'])
    hypotheses, linkage = _collinear_component_hypotheses(xy, observed, valid, component, direction, seed_normal, cfg)
    output['component_linking_audit'] = linkage
    if linkage['combinatorial_overflow_abstained']:
        return dict(output, reason='too_many_actual_component_link_alternatives_explicit_abstention')
    failures = []
    for label, hypothesis in enumerate(hypotheses):
        selected = np.asarray(hypothesis['indices'], dtype=int)
        if len(selected)<cfg['minimum_inlier_count']: continue
        t = (along[selected]-lo)/(hi-lo)
        confidence = np.clip(contrast[selected]/40., .05, 1.)
        fitted, diag = F._fit_line(observed[selected], confidence, t, seed_normal,
                                    len(selected), line_cfg)
        if fitted is None:
            failures.append(dict(component=int(label), pixel_count=len(selected), **diag)); continue
        inliers = np.asarray(diag['inlier_mask'], bool)
        actual = observed[selected][inliers]
        normal = np.asarray(diag['normal'])
        tangent = np.array([-normal[1], normal[0]])
        values = actual@tangent
        along_actual = (actual-q)@direction
        # Actual measured support, never an extension or invented closing edge.
        endpoint_support = dict(first=bool(along_actual.min() <= cfg['along_start_px']+cfg['boundary_context_px']),
                                second=(bool(along_actual.max() >= length-(cfg['along_start_px']+cfg['boundary_context_px'])) if whole else None))
        endpoint_normal = {}
        for endpoint in [point, mate_point] if whole else [point]:
            if endpoint.get('source')!='VISIBLE_BACKBONE_FALLBACK':
                endpoint_normal[str(endpoint['native_corner'])] = float(abs(normal@np.asarray(endpoint['xy'])+diag['offset']))
        endpoint_pass = all(v<=cfg['along_start_px']+cfg['boundary_context_px'] for v in endpoint_normal.values())
        record = dict(**diag, component=int(label),
            component_ids=hypothesis['component_ids'], actual_max_gap_px=hypothesis['max_gap_px'],
            invented_bridge_points=0, measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT',
            native_corner=k, native_neighbor=mate, native_edge=[k,mate],
            actual_inlier_tangent_interval_px=[float(values.min()),float(values.max())],
            actual_inlier_image_lattice_xy=xy[selected][inliers].tolist(),
            image_gradient_normal_xy=image_normal[selected].tolist(),
            measured_subpixel_shift_px=shift[selected].tolist(),
            RGB_gradient_energy=energy[selected].tolist(),
            RGB_alignment=alignment[selected].tolist(), RGB_contrast=contrast[selected].tolist(),
            mask_plus=mp[selected].tolist(), mask_minus=mm[selected].tolist(),
            subpixel_boundary_hit_count=int(censored[selected].sum()),
            endpoint_support=endpoint_support, corrected_endpoint_normal_residual_px=endpoint_normal,
            corrected_endpoint_normal_guard_pass=endpoint_pass,
            conditional_uncertainty=_chain_uncertainty(actual,normal,diag['inlier_rms_px'],cfg),
            mask_context_kind=output['mask_context_kind'], native_association_is_conditional=True,
            quality=[float(diag['inlier_span_fraction']),int(inliers.sum()),
                     float(np.median(energy[selected][inliers])),-float(diag['inlier_rms_px'])],
            no_precise_2D_corner_or_unknown_mate_measurement_created=True)
        if endpoint_pass: output['chains'].append(record)
        else: failures.append({**record, 'reason':'existing_corrected_endpoint_normal_guard_failed'})
    output['rejected_chains'] = failures
    if not output['chains']: return dict(output, reason='no_connected_image_peak_chain_passed_existing_line_guards')
    # All alternatives stay archived; selection is descriptive, never KP adoption.
    output['selected_chain_index'] = max(range(len(output['chains'])), key=lambda i:tuple(output['chains'][i]['quality']))
    output.update(supported=True, reason='actual_image_NMS_chain_supported_conditional_native_association',
                  multiple_supported_chains=len(output['chains'])>1,
                  selected_chain_is_not_independently_certified=True)
    return output


def observe_image_supported_edge_chains(image_bgr, pallet_mask, geometry, states, visible_measurements,
                                         *, image_binding, configuration=None):
    cfg = _config(configuration)
    image = np.asarray(image_bgr)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Actual uint8 BGR image required')
    mask = np.asarray(pallet_mask, float)
    if mask.shape!=image.shape[:2] or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
        raise ValueError('Bound finite occupancy mask required')
    states = [V3._state(state) for state in states]
    hidden = geometry['initial_self_hidden']
    if len(states)!=8 or len(hidden)!=8 or tuple(geometry['image_hw'])!=image.shape[:2]:
        raise ValueError('Bound eight states and canvas required')
    if not isinstance(image_binding, Mapping) or tuple(image_binding.get('image_hw',()))!=image.shape[:2]:
        raise ValueError('Explicit original image binding required')
    D._sha(image_binding.get('sha256'),'image')
    pixel_sha = G.OLD.image_sha256(image)
    if image_binding.get('image_pixel_sha256',pixel_sha)!=pixel_sha: raise ValueError('Image pixel binding differs')
    points = G._eligible_points(visible_measurements,states,hidden,image.shape[:2])
    # Unknown mates contribute only model geometry after the permanent S gates.
    projection = {k:V3._finite_array(geometry['initial_projection_xy'][k],(2,),'nonS model projection')
                  for k in range(8) if not hidden[k] and states[k]!='S'}
    face_context = D._face_context(geometry)
    lab = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients = (cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.)
    records=[]
    for k,state in enumerate(states):
        if hidden[k] or state not in ('B','I') or k not in points: continue
        for mate in NEIGHBORS[k]:
            if hidden[mate] or states[mate]=='S':
                records.append(dict(native_corner=k,native_neighbor=mate,supported=False,
                    reason='S_endpoint_excluded_before_projection_and_image_query',pixel_queried=False));continue
            face = D._edge_face_role(face_context,k,mate)
            if face['role'] not in ('INTERNAL_TWO_VISIBLE_FACES','OUTLINE_VISIBLE_HIDDEN_FACES'):
                records.append(dict(native_corner=k,native_neighbor=mate,supported=False,
                    reason='initial_face_association_ambiguous_or_hidden',face_association=face,pixel_queried=False));continue
            record = _edge_chains(image,mask,lab,gradients,k,mate,points[k]['record'],
                points[mate]['record'] if mate in points else None,projection,face,cfg)
            records.append(dict(record,pixel_queried=True,anchor_state=state,neighbor_state=states[mate]))
    return dict(schema='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C',edge_records=records,
        configuration=cfg,image_binding={**deepcopy(dict(image_binding)),'image_pixel_sha256':pixel_sha},
        S_local_queries=0,S_coordinate_reads=0,U_coordinate_reads=0,U_corner_queries=0,
        O_T_coordinate_reads=0,old_corner_observations_changed=0,full_2D_corner_measurements_created=0,
        states_changed=False,pose_calls=0,GT_inputs=False,
        association_and_uncertainty_are_conditional=True,independent_native_association_certified=False,
        calibrated_confidence_or_future_coverage_claimed=False)
