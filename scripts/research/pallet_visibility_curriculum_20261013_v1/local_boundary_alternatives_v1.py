"""Same local stable-corner RGB line family for RGB, Tiny and SAM evidence.

Initial non-S registered geometry guides searches only. Two actual incident
lines are measured independently; their intersection is an unadopted image
candidate. This family has no unknown contour split or raw/old-corrected KP.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib

import cv2
import numpy as np

from . import common as C
from . import boundary_refine as F
from . import visible_bi_correction_v2 as B
from . import visible_bi_boundary_recovery_v4_3b as SUPPORT


DEFAULT_CONFIG=deepcopy(B.DEFAULT_CONFIG)


def _ray(image,mask,lab,gradients,seed,mate,guide,center,cfg):
    length=float(np.linalg.norm(guide))
    row=dict(native_neighbor=int(mate),supported=False,samples=int(cfg['along_samples']),
             measurement_source='actual_RGB_normal_gradient',observed_points=[],observation_confidence=[],
             guide_source='initial_nonS_registered_model_direction_only')
    if length<10.:return dict(row,reason='short_model_guide')
    direction=guide/length;n=np.array([-direction[1],direction[0]])
    if n@(center-seed)<0:n=-n
    end=min(cfg['along_end_px'],.8*length)
    if end<=cfg['along_start_px']:return dict(row,reason='short_search_span')
    along=np.linspace(cfg['along_start_px'],end,int(cfg['along_samples']))
    offsets=np.arange(-cfg['normal_search_radius_px'],cfg['normal_search_radius_px']+.01,cfg['normal_step_px'])
    xy=seed+along[:,None,None]*direction+offsets[None,:,None]*n
    x,y=(B._sample(g,xy) for g in gradients)
    energy=np.linalg.norm(x*n[0]+y*n[1],axis=-1)
    alignment=energy/np.maximum(np.sqrt(np.sum(x*x+y*y,axis=-1)),1e-12)
    color=np.linalg.norm(B._sample(lab,xy+n*1.5)-B._sample(lab,xy-n*1.5),axis=-1)
    valid=(energy>=cfg['minimum_gradient'])&(alignment>=cfg['minimum_gradient_alignment'])&(color>=cfg['minimum_color_transition'])
    margin=1.5
    if mask is not None:
        probe,context=cfg['boundary_probe_px'],cfg['boundary_context_px']
        inside=sum(B._sample(mask,xy+n*probe*s) for s in (.5,1.,1.5))/3.
        outside=sum(B._sample(mask,xy-n*probe*s) for s in (.5,1.,1.5))/3.
        inner,outer=B._sample(mask,xy+n*context),B._sample(mask,xy-n*context)
        valid&=(inside>=cfg['minimum_inside'])&(outside<=cfg['maximum_outside'])&(inside-outside>=cfg['minimum_transition'])&(inner>=.50)&(outer<=.45)
        margin=max(context,1.5*probe)
        row['measurement_source']='actual_RGB_and_target_background_mask_transition'
    h,w=image.shape[:2];a,b=xy+n*margin,xy-n*margin
    valid&=(a[...,0]>=0)&(a[...,0]<w-1)&(a[...,1]>=0)&(a[...,1]<h-1)&(b[...,0]>=0)&(b[...,0]<w-1)&(b[...,1]>=0)&(b[...,1]<h-1)
    # Identical original I score and actual RGB confidence in all three arms.
    score=np.where(valid,energy*alignment-.05*np.abs(offsets),-np.inf)
    best=score.argmax(1);keep=np.isfinite(score[np.arange(len(along)),best])
    observed=xy[np.arange(len(along))[keep],best[keep]]
    confidence=np.clip(color[np.arange(len(along))[keep],best[keep]]/40.,.05,1.)
    row.update(actual_transition_sample_count=int(keep.sum()),search_seed_xy=seed.tolist(),
        search_along_px=along.tolist(),observed_points=observed.tolist(),observation_confidence=confidence.tolist(),
        normal_search_radius_px=cfg['normal_search_radius_px'],mask_gate_applied=mask is not None,
        selection_score='original_I_RGB_energy*alignment-.05*abs(normal_offset)')
    if keep.sum()<cfg['minimum_inlier_count']:return dict(row,reason='insufficient_actual_RGB_transition_samples')
    positions=.15+.7*(along[keep]-along[0])/(along[-1]-along[0])
    line_cfg=dict(along_interval=[.15,.85],huber_delta_px=1.25,
        maximum_angle_change_deg=cfg['maximum_direction_change_deg'],minimum_inlier_count=cfg['minimum_inlier_count'],
        minimum_inlier_fraction=cfg['minimum_inlier_fraction'],minimum_inlier_span_fraction=cfg['minimum_inlier_span_fraction'],
        maximum_line_rms_px=cfg['maximum_line_rms_px'])
    fitted,diag=F._fit_line(observed,confidence,positions,n,len(along),line_cfg)
    row.update(diag,supported=fitted is not None)
    return row


def fit_local_boundary(image_bgr,geometry,native_corner,neighbor_ids,*,configuration=None,
                       pallet_mask=None,mask_provider='RGB_GRADIENT'):
    """Return ``{status,fit_payload,audit}`` for two local incident branches.

Requires image_hw, initial_roles, initial_self_hidden, initial_projection_xy.
No S projection coordinates are indexed. Caller owns frozen actual B/U gates.
Mask providers use the same RGB score/guards plus original mask occupancy gates.
All output remains conditional native association, with no KP/pose adoption.
"""
    audit=dict(fitting_family='STABLE_LOCAL_TWO_INCIDENT_LINES_V1',unknown_split_search=False,
        mask_provider=str(mask_provider),raw_or_old_corrected_keypoint_used=False,GT_inputs=False,
        user_marked_point_used=False,S_raw_coordinates_accessed=False,S_local_queries=0,
        projected_coordinates_used_as_measurements=False,neural_model_forwards=0,pose_fits=0,
        state_changed=False,selected_intersection_is_known_native_KP=False,
        native_association_is_conditional=True,actual_state_eligibility_is_caller_responsibility=True,
        sample_continuity='ordered samples plus original robust TLS angle/RMS/inlier-span guards',
        added_continuity_or_GT_thresholds=False)
    def finish(status,reason,payload=None):
        return C.finite(dict(status=status,fit_payload=payload,audit={**audit,'reason':reason}))
    image=np.asarray(image_bgr)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:return finish('invalid_input','actual_uint8_BGR_required')
    if not isinstance(geometry,Mapping) or tuple(geometry.get('image_hw',()))!=image.shape[:2]:
        return finish('invalid_input','geometry_image_hw_mismatch')
    if isinstance(native_corner,(bool,np.bool_)) or not isinstance(native_corner,(int,np.integer)) or not 0<=native_corner<8:
        return finish('invalid_input','native_integer_corner_required')
    if (not isinstance(neighbor_ids,(list,tuple,np.ndarray)) or len(neighbor_ids)!=2
            or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 0<=x<8 for x in neighbor_ids)
            or len(set(neighbor_ids))!=2 or native_corner in neighbor_ids):
        return finish('invalid_input','two_distinct_native_neighbors_required')
    hidden=geometry.get('initial_self_hidden');roles=geometry.get('initial_roles');source=geometry.get('initial_projection_xy')
    if (not isinstance(hidden,(list,tuple,np.ndarray)) or len(hidden)!=8 or any(not isinstance(x,(bool,np.bool_)) for x in hidden)
            or not isinstance(roles,(list,tuple,np.ndarray)) or len(roles)!=8
            or not isinstance(source,(list,tuple,np.ndarray)) or len(source)!=8):
        return finish('invalid_input','explicit_initial_geometry_gates_required')
    requested=[int(native_corner),*map(int,neighbor_ids)]
    if any(hidden[x] for x in requested):return finish('excluded','initial_S_excluded_before_image_query')
    if any(roles[x]!='B' for x in requested):return finish('excluded','initial_B_roles_required')
    P=np.full((8,2),np.nan);nonself=[x for x in range(8) if not hidden[x]]
    for x in nonself:
        p=np.asarray(source[x],float)
        if p.shape!=(2,) or not np.isfinite(p).all():return finish('invalid_input','finite_nonS_initial_geometry_required')
        P[x]=p
    if len(nonself)<3:return finish('excluded','insufficient_nonS_hull_geometry')
    hull=[nonself[int(x)] for x in cv2.convexHull(P[nonself].astype(np.float32),returnPoints=False).ravel()]
    k=int(native_corner)
    if k not in hull:return finish('excluded','corner_not_on_nonS_model_hull')
    position=hull.index(k);mates=[hull[(position-1)%len(hull)],hull[(position+1)%len(hull)]]
    if set(mates)!=set(neighbor_ids):return finish('excluded','neighbors_not_incident_nonS_hull_vertices')
    mates=sorted(mates,key=lambda x:(P[x,0],x));guides=np.asarray([P[x]-P[k] for x in mates]);lengths=np.linalg.norm(guides,axis=1)
    if np.any(lengths<=1e-12):return finish('excluded','degenerate_model_incident_directions')
    cfg=dict(DEFAULT_CONFIG)
    if not isinstance(configuration,(Mapping,type(None))) or set(configuration or {})-set(cfg):return finish('invalid_input','unknown_original_image_guard')
    cfg.update(configuration or {})
    if any(not np.isscalar(x) or not np.isfinite(x) for x in cfg.values()):return finish('invalid_input','finite_original_image_guards_required')
    positive=['normal_search_radius_px','normal_step_px','along_start_px','along_end_px','boundary_probe_px','boundary_context_px','maximum_corner_shift_px']
    if any(cfg[x]<=0 for x in positive) or cfg['along_end_px']<=cfg['along_start_px']:return finish('invalid_input','positive_original_search_spans_required')
    if int(cfg['along_samples'])!=cfg['along_samples'] or cfg['along_samples']<3:return finish('invalid_input','integer_along_samples_required')
    initial_sine=abs(float(np.linalg.det(guides/lengths[:,None])))
    audit.update(native_corner=k,neighbor_native_ids=mates,initial_direction_sine=initial_sine,
        nonS_model_center_xy=P[nonself].mean(0).tolist(),original_guard_configuration=cfg,
        image_pixel_sha256=B.image_sha256(image),image_hw=list(image.shape[:2]),local_image_queries_performed=False)
    if initial_sine<cfg['minimum_intersection_sine']:return finish('excluded','shallow_geometry_requires_global_branch_family')
    h,w=image.shape[:2]
    if not (0<=P[k,0]<w and 0<=P[k,1]<h):return finish('excluded','model_corner_out_of_canvas')
    mask=None
    if pallet_mask is not None:
        if str(mask_provider)=='RGB_GRADIENT':return finish('invalid_input','RGB_arm_must_not_receive_mask')
        mask=np.asarray(pallet_mask,float)
        if mask.shape!=image.shape[:2] or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
            return finish('invalid_input','finite_bound_target_mask_in_0_1_required')
        audit['mask_probability_sha256']=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    elif str(mask_provider)!='RGB_GRADIENT':return finish('invalid_input','mask_provider_requires_bound_mask')
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
    rays=[_ray(image,mask,lab,gradients,P[k],mate,guide,P[nonself].mean(0),cfg) for mate,guide in zip(mates,guides)]
    for ray in rays:ray['edge']=k*8+ray['native_neighbor']
    audit.update(local_image_queries_performed=True,queried_association_native_ids=[k,*mates],
        mask_gate_applied=mask is not None,whole_corner_seed_shift_guard_px=cfg['maximum_corner_shift_px'])
    segments=[np.asarray(ray['observed_points'],float).reshape(-1,2) for ray in rays]
    points=np.vstack(segments);counts=[len(x) for x in segments]
    confidence=np.concatenate([np.asarray(ray['observation_confidence'],float) for ray in rays])
    pair=dict(left=rays[0],right=rays[1],both_original_line_guards_pass=all(ray['supported'] for ray in rays),
        unknown_split_search=False,fit_is_independent_incident_line_pair=True)
    payload=dict(contour=points,contour_segments=segments,branch_sample_ranges=[[0,counts[0]],[counts[0],sum(counts)]],
        contour_is_segmentation_boundary=False,contour_is_two_unconnected_observed_branches=True,
        refined_points=points,RGB_eligible=np.ones(len(points),bool),observation_confidence=confidence,
        split_profile=[pair],selected_representative=None,bootstrap_records=[],native_corner=k,
        neighbor_ids=mates,state_changed=False,accepted_corner=False,GT_inputs=False)
    if not pair['both_original_line_guards_pass']:return finish('lines_unsupported','original_line_support_guards_failed',payload)
    matrix=np.asarray([ray['normal'] for ray in rays]);sine=abs(float(np.linalg.det(matrix)))
    pair.update(intersection_sine=sine,normal_matrix_condition_number=float(np.linalg.cond(matrix)))
    if sine<cfg['minimum_intersection_sine']:return finish('intersection_rejected','fitted_lines_near_parallel',payload)
    candidate=np.linalg.solve(matrix,-np.asarray([ray['offset'] for ray in rays]))
    shift=float(np.linalg.norm(candidate-P[k]));maximum=cfg['along_start_px']+cfg['boundary_context_px']
    guard=SUPPORT._support_interval_guard(candidate,rays,maximum)
    pair.update(intersection_xy=candidate.tolist(),finite_intersection=True,
        observed_support_audit=guard,finite_observed_support_pass=guard['passed'],
        corner_seed_shift_px=shift,corner_seed_shift_guard_px=cfg['maximum_corner_shift_px'],
        robust_Huber_cost=float(sum(np.sum(np.where(np.abs(np.asarray(ray['observed_points'])@np.asarray(ray['normal'])+ray['offset'])<=1.25,
            .5*(np.asarray(ray['observed_points'])@np.asarray(ray['normal'])+ray['offset'])**2,
            1.25*(np.abs(np.asarray(ray['observed_points'])@np.asarray(ray['normal'])+ray['offset'])-.625))) for ray in rays)))
    if not guard['passed']:return finish('intersection_rejected','beyond_original_11px_observed_support_margin',payload)
    if shift>cfg['maximum_corner_shift_px']:return finish('intersection_rejected','beyond_original_12px_model_corner_search_guard',payload)
    if not (0<=candidate[0]<w and 0<=candidate[1]<h):return finish('intersection_rejected','intersection_outside_canvas',payload)
    payload['selected_representative']=pair
    return finish('fit_supported','unadopted_actual_image_incident_line_intersection',payload)
