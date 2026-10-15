"""Same opt-in mask lower-boundary adapter for Tiny and SAM.

Initial NONSELF model projections select endpoints and directions. No corrected
image endpoint, raw junction, painted mark or GT point is accepted. Each actual
exterior contour is compared with both projected neighbor endpoints; the chosen
component must be within the ORIGINAL search radius plus boundary context at
both endpoints. Its lower image arc receives the immutable unknown-split RGB
fitter, with original line guards and bootstrap disabled.

Returned line/intersection evidence is conditional on contour/native association
and remains diagnostic; this adapter creates no measured KP or pose and fits no
total covariance. Cached or fresh masks use exactly the same callable.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from . import common as C
from . import visible_bi_correction_v2 as B


FITTER_SOURCE=C.ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/kp3_bottom_piecewise_v1/diagnose_piecewise_lower_contour_v1.py'
FITTER_SHA256='bc723279044288384f80e3cf3734e4fd53f40c4dd1f9786f553b8350ba54fcd4'
_FITTER=None


def _load_fitter():
    global _FITTER
    if hashlib.sha256(FITTER_SOURCE.read_bytes()).hexdigest()!=FITTER_SHA256:
        raise RuntimeError('Frozen piecewise fitter source binding mismatch')
    if _FITTER is None:
        spec=importlib.util.spec_from_file_location('mask_lower_frozen_piecewise_v1',FITTER_SOURCE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_FITTER=module
    return _FITTER


def _arc(contour,start,end,step):
    ids=[start];index=start
    while index!=end:
        index=(index+step)%len(contour);ids.append(index)
        if len(ids)>len(contour):raise ValueError('Malformed closed exterior contour')
    return contour[ids].copy()


def fit_lower_boundary_mask(image_bgr,pallet_mask,geometry,native_corner,neighbor_ids,*,
                            configuration=None,mask_provider='TINYUNET'):
    """Return ``{status, fit_payload, audit}`` from an actual mask and geometry.

``geometry`` requires initial_projection_xy[8,2], initial_self_hidden[8],
initial_roles[8], image_hw[2]. Provider is metadata only; both masks use .5,
the same component/arc/guard rules and the same frozen RGB unknown-split fit.
Actual B/U visibility and both neighbors' actual B eligibility belong to the
caller. A finite initial k projection outside canvas is audited rather than
used as a truncation certificate. No corrected/raw image KP enters this API.
"""
    audit=dict(mask_provider=str(mask_provider),mask_threshold=.5,
        contour_selection_source='initial_nonS_model_neighbor_projections_only',
        SAM_corrected_endpoints_used=False,raw_junction_coordinate_used=False,
        painted_point_used=False,GT_inputs=False,S_local_queries=0,bootstrap=False,
        native_association_is_conditional=True,selected_intersection_is_known_native_KP=False,
        pose_fits=0,model_forward_calls=0,runtime_scope='mask input plus RGB fit; no mask-model latency claim',
        fitter_sha256=FITTER_SHA256,actual_state_eligibility_is_caller_responsibility=True)
    def finish(status,reason,fit=None):
        return dict(status=status,fit_payload=fit,audit={**audit,'reason':reason})
    try:
        image=np.asarray(image_bgr);mask=np.asarray(pallet_mask,float)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            return finish('invalid_input','actual_uint8_BGR_image_required')
        if mask.shape!=image.shape[:2] or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
            return finish('invalid_input','finite_bound_mask_in_0_1_required')
        if not isinstance(geometry,Mapping) or tuple(geometry.get('image_hw',()))!=image.shape[:2]:
            return finish('invalid_input','geometry_image_hw_mismatch')
        if isinstance(native_corner,(bool,np.bool_)) or not isinstance(native_corner,(int,np.integer)) or not 0<=native_corner<8:
            return finish('invalid_input','native_corner_integer_0_7_required')
        if (not isinstance(neighbor_ids,(list,tuple,np.ndarray)) or len(neighbor_ids)!=2
                or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 0<=k<8 for k in neighbor_ids)
                or len(set(neighbor_ids))!=2 or native_corner in neighbor_ids):
            return finish('invalid_input','two_distinct_native_neighbor_ids_required')
        hidden=geometry.get('initial_self_hidden');roles=geometry.get('initial_roles');source=geometry.get('initial_projection_xy')
        if (not isinstance(hidden,(list,tuple,np.ndarray)) or len(hidden)!=8
                or any(not isinstance(v,(bool,np.bool_)) for v in hidden)
                or not isinstance(roles,(list,tuple,np.ndarray)) or len(roles)!=8
                or not isinstance(source,(list,tuple,np.ndarray)) or len(source)!=8):
            return finish('invalid_input','initial_geometry_schema_malformed')
        # Gate S before indexing any projected point or choosing a mask arc.
        requested=[int(native_corner),*[int(k) for k in neighbor_ids]]
        if any(hidden[k] for k in requested):
            return finish('excluded','initial_S_excluded_before_local_mask_query')
        if any(roles[k]!='B' for k in requested):
            return finish('excluded','initial_boundary_role_required_for_corner_and_neighbors')
        projected=np.full((8,2),np.nan)
        nonself=[k for k in range(8) if not hidden[k]]
        for k in nonself:
            q=np.asarray(source[k],float)
            if q.shape!=(2,) or not np.isfinite(q).all():
                return finish('invalid_input','finite_nonS_initial_projection_required')
            projected[k]=q
        if len(nonself)<3:
            return finish('excluded','insufficient_nonS_model_hull')
        hull_indices=cv2.convexHull(projected[nonself].astype(np.float32),returnPoints=False).ravel()
        hull=[nonself[int(k)] for k in hull_indices]
        k=int(native_corner)
        if k not in hull or len(hull)<3:
            return finish('excluded','corner_not_on_nonS_initial_model_hull')
        index=hull.index(k);mates=[hull[(index-1)%len(hull)],hull[(index+1)%len(hull)]]
        if set(mates)!=set(neighbor_ids):
            return finish('excluded','neighbor_ids_do_not_match_nonS_initial_hull')
        mates=sorted(mates,key=lambda j:(projected[j,0],j))
        guides=np.asarray([projected[j]-projected[k] for j in mates]);lengths=np.linalg.norm(guides,axis=1)
        if np.any(lengths<=1e-12):return finish('excluded','degenerate_initial_branch_direction')
        sine=abs(float(np.linalg.det(guides/lengths[:,None])))
        height,width=image.shape[:2]
        audit.update(native_corner=k,neighbor_native_ids=mates,
            initial_neighbor_projection_xy=projected[mates].tolist(),initial_corner_projection_xy=projected[k].tolist(),
            initial_corner_projection_in_canvas=bool(0<=projected[k,0]<width and 0<=projected[k,1]<height),
            initial_direction_sine=sine,shallow_sine_filter_active=False,nonS_model_center_xy=projected[nonself].mean(0).tolist(),
            image_pixel_sha256=B.image_sha256(image),mask_probability_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest())
        if projected[k,1]<=max(projected[j,1] for j in mates):
            return finish('excluded','corner_not_a_lower_initial_model_apex')
        cfg=dict(B.DEFAULT_CONFIG)
        if not isinstance(configuration,(Mapping,type(None))) or set(configuration or {})-set(cfg):
            return finish('invalid_input','unknown_original_guard_configuration')
        cfg.update(configuration or {})
        if any(not np.isscalar(v) or not np.isfinite(v) for v in cfg.values()):
            return finish('invalid_input','finite_original_guard_configuration_required')
        maximum_endpoint_distance=float(cfg['normal_search_radius_px']+cfg['boundary_context_px'])
        if maximum_endpoint_distance<=0:return finish('invalid_input','positive_original_endpoint_guard_required')
        audit.update(endpoint_distance_guard_px=maximum_endpoint_distance,
            endpoint_distance_guard_source='original_normal_search_radius_px + original_boundary_context_px',
            original_guard_configuration=cfg)
        contours,_=cv2.findContours((mask>=.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        audit['exterior_component_count']=len(contours)
        if not contours:return finish('contour_missing','no_actual_mask_exterior_component')
        candidates=[]
        for ci,raw in enumerate(contours):
            contour=np.asarray(raw[:,0,:],float)
            if contour.ndim!=2 or contour.shape[1]!=2 or len(contour)<3 or not np.isfinite(contour).all():continue
            differences=[np.linalg.norm(contour-projected[j],axis=1) for j in mates]
            indices=[int(d.argmin()) for d in differences];distances=[float(d[i]) for d,i in zip(differences,indices)]
            candidates.append(dict(component_index=ci,contour=contour,nearest_indices=indices,distances=distances,
                distance_sum=float(sum(distances)),contour_samples=len(contour)))
        audit['component_endpoint_distance_audit']=[{key:value for key,value in candidate.items() if key!='contour'} for candidate in candidates]
        if not candidates:return finish('contour_malformed','no_finite_exterior_contour_with_three_samples')
        selected=min(candidates,key=lambda value:(value['distance_sum'],value['component_index']))
        audit.update(selected_component_index=selected['component_index'],
            nearest_projected_endpoint_distance_px=selected['distances'],
            nearest_exterior_endpoint_xy=selected['contour'][selected['nearest_indices']].tolist(),
            component_selection_rule='minimum_sum_of_two_projected_endpoint_nearest_exterior_contour_sample_distances',
            endpoint_guard_pass=bool(max(selected['distances'])<=maximum_endpoint_distance))
        if not audit['endpoint_guard_pass']:return finish('endpoint_guard_rejected','projected_endpoint_too_far_from_selected_exterior_component')
        start,end=selected['nearest_indices']
        if start==end:return finish('contour_malformed','both_projected_endpoints_map_to_same_contour_sample')
        arcs=[_arc(selected['contour'],start,end,step) for step in (-1,1)]
        means=[float(arc[:,1].mean()) for arc in arcs]
        contour=arcs[int(np.argmax(means))]
        audit.update(alternative_arc_mean_image_y=means,
            contour_arc_selection_rule='greater_mean_image_y_actual_arc_between_initial_projected_neighbors',
            selected_arc_sample_count=len(contour),
            selected_arc_length_px=float(np.linalg.norm(np.diff(contour,axis=0),axis=1).sum()))
        if len(contour)<3:return finish('contour_malformed','selected_lower_arc_has_fewer_than_three_samples')
        fitter=_load_fitter()
        center=projected[nonself].mean(0)
        normals=[fitter.inward_normal(projected[j]-projected[k],center,contour.mean(0)) for j in mates]
        audit['initial_inward_seed_normals']=[n.tolist() for n in normals]
        fit=fitter.fit_piecewise_boundary(image,mask,contour,seed_normals=normals,
            configuration=cfg,do_bootstrap=False,native_corner=k,neighbor_ids=tuple(mates))
        profile=fit['split_profile'];qualified=[p for p in profile if p['both_original_line_guards_pass'] and p['finite_intersection'] and p['finite_observed_support_pass']]
        audit.update(profile_candidate_count=len(profile),qualified_profile_count=len(qualified),
            RGB_supported_sample_count=int(np.asarray(fit['RGB_eligible'],bool).sum()),
            total_arc_sample_count=len(contour),no_intersection_sine_threshold_added=True,
            normal_covariance_calibrated=False,mask_native_association_certified=False)
        if fit['selected_representative'] is None:
            return finish('no_qualified_fit','no_two_RGB_supported_finite_branch_fits',C.finite(fit))
        return finish('qualified_fit','conditional_unknown_split_actual_mask_RGB_lines',C.finite(fit))
    except (ValueError,TypeError,IndexError,KeyError,cv2.error) as error:
        audit['exception_type']=type(error).__name__
        return finish('invalid_input','malformed_mask_geometry_or_original_fitter_input: '+str(error))
