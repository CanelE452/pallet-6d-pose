"""Opt-in observed lower-contour factors, without a raw junction coordinate.

Known visible boundary endpoints select an actual image arc. Its unknown split
is fitted from RGB/SAM evidence. U remains U; no complete point is invented.
The normal error budget is conservatively borrowed from the existing CAL
fallback-B scale and is explicitly not a fitted boundary calibration.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as V
from . import visible_bi_correction_v2 as B
from .statewise_uncertainty_calibration_v4 import RECIPE


FITTER_SOURCE = V.FIX/'boundary_recovery_v1/kp3_bottom_piecewise_v1/diagnose_piecewise_lower_contour_v1.py'


def load_piecewise_fitter(expected_sha256=None):
    if expected_sha256 is not None and C.sha(FITTER_SOURCE)!=expected_sha256:
        raise ValueError('Frozen piecewise fitter changed')
    spec=importlib.util.spec_from_file_location('observed_piecewise_boundary_v1',FITTER_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def observe_lower_boundary(image_bgr, pallet_mask, geometry, states, state_records,
                           correction, uncertainty_model, nonself_scale_px, *,
                           image_sha256=None, fitter_sha256=None, do_bootstrap=False):
    """Extract soft normal evidence for unresolved shallow lower B/U corners.

    Both neighboring endpoints must already be actual B image observations.
    Neither raw B/U/S positions nor a user-drawn point enters junction fitting.
    The whole observed arc bounds tangent support, preserving split ambiguity.
    """
    image=np.asarray(image_bgr);mask=np.asarray(pallet_mask,float)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Actual uint8 BGR image required')
    if mask.shape!=image.shape[:2] or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
        raise ValueError('Bound actual SAM target mask required')
    if tuple(geometry['image_hw'])!=image.shape[:2] or len(states)!=8 or len(state_records)!=8:
        raise ValueError('Geometry, state records and current canvas must match')
    if uncertainty_model.get('fit_partition')!='CAL':
        raise ValueError('Existing CAL error-budget model required')
    P=np.asarray(geometry['initial_projection_xy'],float)
    if P.shape!=(8,2) or not np.isfinite(P).all():raise ValueError('Eight initial model directions required')
    hull=cv2.convexHull(P.astype(np.float32),returnPoints=False).ravel().tolist()
    neighbors={k:(hull[(i-1)%len(hull)],hull[(i+1)%len(hull)]) for i,k in enumerate(hull)}
    records={int(r['native_corner']):r for r in correction['records']}
    measured={int(m['native_corner']):m for m in correction['visible_measurements']}
    if correction['image_pixel_sha256']!=B.image_sha256(image):raise ValueError('Correction RGB binding mismatch')
    cfg=correction['configuration'];fitter=load_piecewise_fitter(fitter_sha256)
    ratio=float(np.clip(float(nonself_scale_px)/uncertainty_model['reference_scale_px'],*RECIPE['scene_scale_ratio_clip']))
    fallback_floor=float(uncertainty_model['groups']['B:fallback']['floor_px2'])*ratio**2
    borrowed_base_variance=float(uncertainty_model['covariance_scale'])*cfg['detector_sigma_px']**2+fallback_floor
    observations=[];audit=[];fits={}
    for k,state in enumerate(states):
        item=dict(native_corner=k,state=state,image_arc_queried=False,accepted_partial=False)
        audit.append(item)
        if state not in {'B','U'} or geometry['initial_self_hidden'][k]:
            item['reason']='excluded_before_boundary_observation_query';continue
        rec=state_records[k]
        if rec['initial_role']!='B' or not rec.get('rgb_evaluated') or rec.get('raw_query_out_of_frame') is not False:
            item['reason']='no_in_canvas_initial_boundary_role';continue
        if state=='B' and records[k].get('corrected'):
            item['reason']='existing_complete_B_observation_preserved';continue
        if k not in neighbors:
            item['reason']='not_on_initial_model_hull';continue
        mates=sorted(neighbors[k],key=lambda j:P[j,0]);left,right=mates
        guides=np.asarray([P[left]-P[k],P[right]-P[k]])
        lengths=np.linalg.norm(guides,axis=1)
        if np.any(lengths<=0):item['reason']='degenerate_model_direction';continue
        sine=abs(float(np.linalg.det(guides/lengths[:,None])))
        item['initial_direction_sine']=sine
        if sine>=cfg['minimum_intersection_sine'] or P[k,1]<=max(P[left,1],P[right,1]):
            item['reason']='not_a_shallow_lower_model_junction';continue
        if any(states[j]!='B' or j not in measured or not measured[j].get('image_measurement') for j in mates):
            item['reason']='two_actual_visible_boundary_endpoints_required';continue
        endpoint_xy=[np.asarray(measured[j]['xy'],float) for j in mates]
        contour,_,selection=fitter.lower_contour(mask,*endpoint_xy)
        selection=deepcopy(selection)
        selection['rule']='greater_mean_image_y_arc_between_two_observed_boundary_endpoints'
        selection['endpoint_native_ids']=mates
        normals=[fitter.inward_normal(P[j]-P[k],P.mean(0),contour.mean(0)) for j in mates]
        fit=fitter.fit_piecewise_boundary(image,mask,contour,seed_normals=normals,
            configuration=cfg,do_bootstrap=do_bootstrap,native_corner=k,neighbor_ids=tuple(mates))
        fits[k]=C.finite(fit)
        item.update(image_arc_queried=True,neighbor_native_ids=mates,contour_selection=selection,
            user_or_raw_junction_coordinate_used=False,fit_source_sha256=C.sha(FITTER_SOURCE))
        chosen=fit['selected_representative']
        if chosen is None:
            item['reason']='no_two_RGB_supported_finite_branch_fits';continue
        supported=np.asarray(fit['refined_points'])[np.asarray(fit['RGB_eligible'],bool)]
        lines=[]
        for ray in (chosen['left'],chosen['right']):
            normal=np.asarray(ray['normal'],float);tangent=np.array([-normal[1],normal[0]])
            positions=supported@tangent
            lines.append(dict(normal_unit_xy=normal.tolist(),offset_px=float(ray['offset']),
                normal_variance_px2=2.*borrowed_base_variance,
                support_tangent_interval_px=[float(positions.min()),float(positions.max())],
                support_extrapolation_px=float(cfg['along_start_px']+cfg['boundary_context_px']),
                provenance=dict(actual_RGB_mask_branch=True,normal_variance_budget='2 * existing CAL fallback-B effective variance',
                    boundary_total_error_calibration=False,paired_independence_or_new_sensor_information=False,
                    support_source='whole RGB-supported observed arc; no best-split/raw-x anchor')))
        observation=dict(native_corner=k,state=state,lines=lines,
            image_binding=dict(sha256=image_sha256 or B.image_sha256(image),image_hw=list(image.shape[:2])),
            association_provenance=dict(initial_hull_native_association_is_conditional=True,observed_neighbor_ids=mates,
                fitted_split_is_not_known_native_vertex=True,user_or_raw_junction_coordinate_used=False,
                state_changed=False,total_error_budget_is_borrowed_assumption=True,correlated_contour_reuse=True,
                variance_count_inflation=2.,source_sha256=C.sha(FITTER_SOURCE)))
        observations.append(observation)
        item.update(accepted_partial=True,reason='conditional_observed_branch_normal_factors',
            observed_intersection_xy=chosen['intersection_xy'],intersection_sine=chosen['intersection_sine'],
            full_2D_point_created=False,normal_variance_px2=2.*borrowed_base_variance)
    return C.finite(dict(boundary_observations=observations,records=audit,fits=fits,
        states_unchanged=True,S_boundary_queries=0,GT_inputs=False,user_junction_point_inputs=False,
        declared_scope='shallow lower silhouette junction with two observed B endpoints',
        variance_policy=dict(CAL_model_not_refitted=True,nonself_scale_ratio=ratio,
            borrowed_base_variance_px2=borrowed_base_variance,two_factor_count_inflation=2.,
            new_boundary_total_error_covariance_calibrated=False),fitter_sha256=C.sha(FITTER_SOURCE)))


def fixed_fit_measurements(original_calibrated, correction):
    """Use recovered image coordinates with the original fixed fit scales."""
    old={int(m['native_corner']):m for m in original_calibrated}
    result=[]
    for observed in correction['visible_measurements']:
        k=int(observed['native_corner']);record=deepcopy(observed)
        record['fit_covariance_px2']=deepcopy(old[k]['fit_covariance_px2'])
        record['fit_reliability_weight']=old[k]['fit_reliability_weight']
        record['fixed_original_fit_scale']=True
        result.append(record)
    if set(old)!={m['native_corner'] for m in result}:raise ValueError('Frozen visible native IDs changed')
    return result
