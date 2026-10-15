"""SAM-free lower-boundary proposal and unknown-split RGB line fitting.

Only initial registered model geometry guides a finite search band. Actual RGB
transition samples determine the trace and unknown split. No raw keypoint,
corrected B measurement, segmentation mask, annotation, or pose target enters.
The shared versioned fitter is pure image-line geometry; its offline main and
all neural forwards are never invoked. Output remains an association hypothesis.
"""
from copy import deepcopy
import importlib.util

import cv2
import numpy as np

from . import common as C
from . import visible_bi_correction_v2 as B


DEFAULT_CONFIG = deepcopy(B.DEFAULT_CONFIG)
FITTER_RELATIVE_PATH = ('_docs/experiments/pallet_visibility_curriculum_20261013_v1/'
    'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/'
    'kp3_bottom_piecewise_v1/diagnose_piecewise_lower_contour_v1.py')
FITTER_SHA256 = 'bc723279044288384f80e3cf3734e4fd53f40c4dd1f9786f553b8350ba54fcd4'
_FITTER = None


def _fitter():
    global _FITTER
    path=C.ROOT/FITTER_RELATIVE_PATH
    if C.sha(path)!=FITTER_SHA256:
        raise ValueError('Immutable pure unknown-split fitter binding changed')
    if _FITTER is None:
        spec=importlib.util.spec_from_file_location('_rgb_lower_boundary_pure_fitter_v1',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FITTER=module
    return _FITTER


def _config(configuration):
    cfg=dict(DEFAULT_CONFIG)
    if set(configuration or {})-set(cfg):raise ValueError('Unknown original RGB image-guard field')
    cfg.update(configuration or {})
    if not all(np.isfinite(v) for v in cfg.values()):raise ValueError('Finite original image guards required')
    for key in ('normal_search_radius_px','normal_step_px','along_start_px','along_end_px',
                'maximum_line_rms_px','boundary_context_px','minimum_gradient'):
        if cfg[key]<=0:raise ValueError(f'{key} must be positive')
    return cfg


def _inward_normal(direction,model_center,point):
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    n=np.array([-direction[1],direction[0]])
    return -n if n@(model_center-point)<0 else n


def _ordered_trace(score,eligible,offsets,maximum_step_px,huber_scale):
    """Finite offset-state DP. Missing rows remain missing, never measurements."""
    count,states=score.shape
    # Positive image evidence competes with a zero-evidence gap. A gap retains
    # a geometric search location only, and is excluded before any line fit.
    emission=np.where(eligible,score,0.)
    last=emission[0].copy();parents=np.empty((count,states),np.int32)
    jumps=int(np.floor(maximum_step_px/(offsets[1]-offsets[0])+1e-9))
    for i in range(1,count):
        candidates=np.full((2*jumps+1,states),-np.inf)
        predecessors=np.full((2*jumps+1,states),-1,np.int32)
        for row,jump in enumerate(range(-jumps,jumps+1)):
            current=np.arange(states);previous=current+jump
            keep=(previous>=0)&(previous<states)
            penalty=abs(jump*(offsets[1]-offsets[0]))/huber_scale
            candidates[row,keep]=last[previous[keep]]-penalty
            predecessors[row,keep]=previous[keep]
        chosen=candidates.argmax(0)
        parents[i]=predecessors[chosen,np.arange(states)]
        last=emission[i]+candidates[chosen,np.arange(states)]
    state=int(last.argmax());path=np.empty(count,np.int32);path[-1]=state
    for i in range(count-1,0,-1):
        state=int(parents[i,state]);path[i-1]=state
    return path,float(last.max())


def fit_lower_boundary_rgb(image_bgr,geometry,native_corner,neighbor_ids,*,configuration=None):
    """Return ``{status, fit_payload, audit}`` for an RGB-only lower apex.

``geometry`` requires native ``initial_projection_xy`` and
``initial_self_hidden``. Caller applies its frozen B/U eligibility policy.
Search is limited to the original normal12px band around model lower edges;
an actual boundary beyond that band can be missed. No raw junction is accepted.
"""
    cfg=_config(configuration);image=np.asarray(image_bgr)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('Actual uint8 BGR image required')
    P=np.asarray(geometry.get('initial_projection_xy'),float)
    hidden=np.asarray(geometry.get('initial_self_hidden'),bool)
    if P.shape!=(8,2) or not np.isfinite(P).all() or hidden.shape!=(8,):
        raise ValueError('Registered native geometry and explicit initial-S gate required')
    k=int(native_corner);neighbors=tuple(map(int,neighbor_ids))
    if k not in range(8) or len(neighbors)!=2 or any(x not in range(8) for x in neighbors) or len({k,*neighbors})!=3:
        raise ValueError('Native junction and two distinct neighbors required')
    audit=dict(method='RGB_NORMAL_GRADIENT_CONTINUOUS_TRACE_V1',native_corner=k,
        neighbor_ids=list(neighbors),SAM_mask_used=False,Tiny_mask_used=False,
        original_or_corrected_keypoint_used=False,GT_inputs=False,user_marked_point_used=False,
        initial_S_excluded_before_image_queries=True,initial_S_raw_accessed=False,
        neural_model_forwards=0,added_neural_model_weights=0,new_pose_fits=0,state_changed=False,
        projected_positions_used_as_image_measurements=False,
        helper_binding=dict(path=FITTER_RELATIVE_PATH,sha256=FITTER_SHA256),
        image_pixel_sha256=B.image_sha256(image),image_hw=list(image.shape[:2]),
        configuration=cfg,search_limitation='actual edge outside model normal12px band may be missed',
        target_background_semantics_not_certified_by_RGB=True,
        shadow_or_floor_edges_can_remain_ambiguous=True)
    def fail(status):return C.finite(dict(status=status,fit_payload=None,audit=audit))
    if hidden[k] or any(hidden[x] for x in neighbors):
        audit['local_image_queries_performed']=False
        return fail('INITIAL_SELF_EXCLUDED')
    h,w=image.shape[:2]
    if not (0<=P[k,0]<w and 0<=P[k,1]<h):
        audit['local_image_queries_performed']=False
        return fail('MODEL_APEX_OUTSIDE_CANVAS')
    left,right=sorted(neighbors,key=lambda x:P[x,0])
    if not P[left,0]<P[k,0]<P[right,0] or P[k,1]<max(P[left,1],P[right,1]):
        audit['local_image_queries_performed']=False
        return fail('NOT_MONOTONE_LOWER_MODEL_APEX')
    center=P[~hidden].mean(0)
    normals=np.asarray([_inward_normal(P[left]-P[k],center,P[k]),
                        _inward_normal(P[right]-P[k],center,P[k])])
    start=max(0.,float(P[left,0]));end=min(w-1.,float(P[right,0]))
    # Native one-pixel along-image cadence, matching raster exterior contours.
    xs=np.arange(np.ceil(start),np.floor(end)+1.,1.)
    if len(xs)<3:return fail('SHORT_MODEL_SEARCH_SPAN')
    ys=np.interp(xs,[P[left,0],P[k,0],P[right,0]],[P[left,1],P[k,1],P[right,1]])
    guide=np.column_stack([xs,ys])
    row_normals=np.where((xs<=P[k,0])[:,None],normals[0],normals[1])
    offsets=np.arange(-cfg['normal_search_radius_px'],cfg['normal_search_radius_px']+.01,cfg['normal_step_px'])
    xy=guide[:,None,:]+offsets[None,:,None]*row_normals[:,None,:]
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gx,gy=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
    x,y=B._sample(gx,xy),B._sample(gy,xy)
    normal_gradient=x*row_normals[:,None,0,None]+y*row_normals[:,None,1,None]
    energy=np.linalg.norm(normal_gradient,axis=-1)
    alignment=energy/np.maximum(np.sqrt(np.sum(x*x+y*y,axis=-1)),1e-12)
    color=np.linalg.norm(B._sample(lab,xy+row_normals[:,None,:]*1.5)
                         -B._sample(lab,xy-row_normals[:,None,:]*1.5),axis=-1)
    valid=(energy>=cfg['minimum_gradient'])&(alignment>=cfg['minimum_gradient_alignment'])&(color>=cfg['minimum_color_transition'])
    a,b=xy+row_normals[:,None,:]*1.5,xy-row_normals[:,None,:]*1.5
    valid&=(a[...,0]>=0)&(a[...,0]<w-1)&(a[...,1]>=0)&(a[...,1]<h-1)&(b[...,0]>=0)&(b[...,0]<w-1)&(b[...,1]>=0)&(b[...,1]<h-1)
    # Same frozen I appearance score, normalized by its existing gradient gate.
    score=(energy*alignment-.05*np.abs(offsets))/cfg['minimum_gradient']
    limit=cfg['normal_step_px']+np.tan(np.deg2rad(cfg['maximum_direction_change_deg']))
    traces=[]
    for polarity in (-1,1):
        admissible=valid&(polarity*normal_gradient[...,0]>=0)
        path,value=_ordered_trace(score,admissible,offsets,limit,1.25)
        eligible=admissible[np.arange(len(xs)),path]
        traces.append(dict(polarity=polarity,path=path,eligible=eligible,score=value))
    chosen=max(traces,key=lambda t:(t['score'],int(t['eligible'].sum()),-t['polarity']))
    points=xy[np.arange(len(xs)),chosen['path']]
    eligible=chosen['eligible'];confidence=np.clip(color[np.arange(len(xs)),chosen['path']]/40.,.05,1.)
    audit.update(local_image_queries_performed=True,queried_association_native_ids=[k,left,right],
        initial_model_center_uses_nonS_only=True,model_search_polyline=guide.tolist(),
        normal_search_radius_px=cfg['normal_search_radius_px'],native_pixel_trace_step_px=1.,
        maximum_offset_step_px=limit,continuity_rule='oldnormal_step +tan(oldangle12deg)*native1px',
        continuity_penalty_scale_px=1.25,continuity_penalty_source='existingHuber1.25',
        global_luminance_polarities_tried=[-1,1],selected_polarity=chosen['polarity'],
        trace_path_score=chosen['score'],proposed_trace_samples=len(points),
        actual_RGB_supported_samples=int(eligible.sum()),actual_RGB_supported_fraction=float(eligible.mean()),
        unsupported_search_locations_excluded_before_line_fit=True,
        unknown_split_has_no_model_or_raw_junction_position_penalty=True,
        SAM_Tiny_or_neural_forward_calls=0)
    fitter=_fitter()
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    profile,selected=fitter.search_unknown_split(points,arc,eligible,confidence,normals,cfg,
                                               native_corner=k,neighbor_ids=(left,right))
    common=normals.mean(0);common/=np.linalg.norm(common)
    one=fitter.fit_branch(points,arc,eligible,confidence,np.arange(len(points)),0,len(points)-1,
                         common,cfg,k,k)
    payload=dict(contour=points,arc=arc,refined_points=points,RGB_eligible=eligible,
        observation_confidence=confidence,one_line=one,split_profile=profile,
        selected_representative=selected,bootstrap_records=[],
        contour_is_segmentation_boundary=False,
        contour_represents_candidate_search_trace_with_explicit_missing_samples=True,
        native_corner=k,neighbor_ids=[left,right],state_changed=False,
        raw_junction_coordinate_used=False,accepted_corner=False,GT_inputs=False)
    audit['qualified_two_line_fit']=selected is not None
    audit['supported_fit_does_not_certify_native_vertex']=True
    return C.finite(dict(status='FITTED_CANDIDATE' if selected else 'NO_QUALIFIED_TWO_LINE_FIT',
                         fit_payload=payload,audit=audit))
