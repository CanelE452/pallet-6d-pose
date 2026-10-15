"""Opt-in full-edge RGB reevaluation for every eligible visible B/I observation.

Only actual B/I endpoint coordinates guide searches. Initial model directions
check native association; S/O/T/U coordinates are never read or queried. RGB
lines are fit on central10--90% samples; distinct actual endpoint validation
requires inlier support from both ends before accepting a full edge. Two supported nonparallel lines can yield a measured corner;
otherwise the frozen point stays unchanged. No pose, GT, calibration or state
change occurs here. Image ownership and resampling covariance are conditional.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import combinations
from collections.abc import Mapping
import cv2
import numpy as np

from . import common as C
from . import boundary_refine as F
from . import visible_bi_correction_v2 as OLD
from .interior_observation import NEIGHBORS
from scripts.research.pallet_foundation_mask_20261013_v1.uncertainty import intersection_uncertainty

DEFAULT_CONFIGURATION=dict(OLD.DEFAULT_CONFIG)
DEFAULT_CONFIGURATION.update(central_start_fraction=.1,central_end_fraction=.9,
    endpoint_sample_offsets_px=(4.,7.,11.),minimum_successful_pair_resamples=16,
    maximum_candidate_disagreement_px=3.,RGB_probe_px=1.5,
    global_hypothesis_probe_px=2.5,global_readout_alpha=1.)


def _config(configuration):
    cfg=deepcopy(DEFAULT_CONFIGURATION)
    if configuration:
        if set(configuration)-set(cfg):raise ValueError('Unknown global-edge configuration')
        cfg.update(configuration)
    if not all(np.isfinite(float(v)) for k,v in cfg.items() if k!='endpoint_sample_offsets_px'):
        raise ValueError('Finite global-edge constants required')
    if (cfg['normal_search_radius_px']<=0 or cfg['normal_step_px']<=0
        or not 0<cfg['central_start_fraction']<cfg['central_end_fraction']<1
        or cfg['global_readout_alpha']!=1. or cfg['RGB_probe_px']<=0
        or cfg['global_hypothesis_probe_px']<=0):raise ValueError('Invalid fixed global-edge recipe')
    for key in ('along_samples','minimum_inlier_count','line_resampling','minimum_successful_pair_resamples','seed'):
        if isinstance(cfg[key],bool) or int(cfg[key])!=cfg[key] or cfg[key]<0:raise ValueError('Integer sample counts required')
        cfg[key]=int(cfg[key])
    if cfg['along_samples']<6 or cfg['line_resampling']<16:raise ValueError('Adequate actual sample/resampling count required')
    offsets=np.asarray(cfg['endpoint_sample_offsets_px'],float)
    if offsets.ndim!=1 or len(offsets)<2 or not np.isfinite(offsets).all() or np.any(offsets<=0):raise ValueError('Positive endpoint sample offsets required')
    cfg['endpoint_sample_offsets_px']=offsets.tolist()
    return cfg


def _id(record):
    k=record.get('native_corner',record.get('native_id'))
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0<=k<8:raise ValueError('Valid native ID required')
    return int(k)


def _eligible_points(fit,states,hidden,image_hw):
    result={};seen=set();h,w=image_hw
    for record in fit:
        k=_id(record)
        if k in seen:raise ValueError('Duplicate visible native ID')
        seen.add(k)
        # The state and initial-S gates precede any coordinate/covariance access.
        if hidden[k] or states[k] not in ('B','I'):continue
        if record.get('state')!=states[k]:raise ValueError('Fixed visible state mismatch')
        xy=np.asarray(record['xy'],float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not(0<=xy[0]<w and 0<=xy[1]<h):raise ValueError('Actual in-frame B/I endpoint required')
        result[k]=dict(xy=xy.copy(),record=record,corrected=record.get('source')!='VISIBLE_BACKBONE_FALLBACK')
    return result


def _sample(array,xy):
    return cv2.remap(array,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(float)


def _measure_trace(lab,gradients,mask,xy,normal,offsets,internal,cfg):
    gx,gy=(_sample(v,xy) for v in gradients)
    energy=np.linalg.norm(gx*normal[0]+gy*normal[1],axis=-1)
    magnitude=np.sqrt(np.sum(gx*gx+gy*gy,axis=-1));alignment=energy/np.maximum(magnitude,1e-12)
    probe=cfg['RGB_probe_px'];plus=xy+normal*probe;minus=xy-normal*probe
    contrast=np.linalg.norm(_sample(lab,plus)-_sample(lab,minus),axis=-1)
    inside_plus=_sample(mask,plus);inside_minus=_sample(mask,minus)
    if internal:
        context=(inside_plus>=cfg['minimum_inside'])&(inside_minus>=cfg['minimum_inside'])
    else:
        context=(np.maximum(inside_plus,inside_minus)>=cfg['minimum_inside'])&(np.minimum(inside_plus,inside_minus)<=cfg['maximum_outside'])
        context &= np.abs(inside_plus-inside_minus)>=cfg['minimum_transition']
    h,w=mask.shape;margin=max(probe,cfg['boundary_context_px'])
    a=xy+normal*margin;b=xy-normal*margin
    inframe=(a[...,0]>=0)&(a[...,0]<w-1)&(a[...,1]>=0)&(a[...,1]<h-1)&(b[...,0]>=0)&(b[...,0]<w-1)&(b[...,1]>=0)&(b[...,1]<h-1)
    valid=(energy>=cfg['minimum_gradient'])&(alignment>=cfg['minimum_gradient_alignment'])&(contrast>=cfg['minimum_color_transition'])&context&inframe
    scores=np.where(valid,energy*alignment-.05*np.abs(offsets)[None,:],-np.inf)
    best=np.argmax(scores,axis=1);keep=np.isfinite(scores[np.arange(len(xy)),best])
    return dict(best=best,keep=keep,energy=energy,contrast=contrast,alignment=alignment,
                mask_plus=inside_plus,mask_minus=inside_minus)


def _endpoint_support(fractions,central,selected,inliers):
    """Each endpoint needs distinct actual RGB inlier validation evidence.

    Endpoint samples are separate from the central fit even on short edges.
    A location that coincides with a central sample cannot count twice.
    """
    fractions=np.asarray(fractions);central=np.asarray(central,bool)
    selected=np.asarray(selected,int);inliers=np.asarray(inliers,bool)
    eligible=inliers&~central[selected]
    first=int(np.count_nonzero(eligible&(fractions[selected]<.5)))
    second=int(np.count_nonzero(eligible&(fractions[selected]>.5)))
    return dict(passed=first>0 and second>0,first_endpoint_inliers=first,
                second_endpoint_inliers=second,
                requirement='at_least_one_distinct_actual_RGB_inlier_at_each_endpoint',
                implies_calibrated_ownership_or_exact_native_identity=False)


def _edge(image,mask,lab,gradients,k,mate,points,projection,states,cfg):
    # Both endpoints have already passed the B/I and initial-S gates.
    a=points[k]['xy'];b=points[mate]['xy'];vector=b-a;length=float(np.linalg.norm(vector))
    result=dict(edge=k*8+mate,endpoints=[k,mate],native_corner=k,native_neighbor=mate,
        supported=False,measurement_source='OBSERVED_FULL_EDGE_RGB_WITH_MASK_CONTEXT',
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
    internal=states[k]=='I' or states[mate]=='I'
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


def _support(point,lines,margin):
    audits=[]
    for line in lines:
        n=np.asarray(line['normal']);t=np.array([-n[1],n[0]]);s=float(t@point);lo,hi=line['actual_inlier_tangent_interval_px'];gap=max(lo-s,s-hi,0.)
        audits.append(dict(edge=line['edge'],distance_to_actual_inlier_interval_px=float(gap),passed=gap<=margin+1e-9))
    return dict(passed=all(a['passed'] for a in audits),maximum_extrapolation_px=margin,lines=audits)


def correct_visible_bi_global_edges(image_bgr,pallet_mask,geometry,states,frozen_fit,*,configuration=None):
    cfg=_config(configuration);image=np.asarray(image_bgr);mask=np.asarray(pallet_mask,float)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Actual uint8 BGR image required')
    if mask.shape!=image.shape[:2] or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):raise ValueError('Bound full mask in[0,1] required')
    if tuple(geometry['image_hw'])!=image.shape[:2] or len(states)!=8 or len(geometry['initial_self_hidden'])!=8:raise ValueError('Bound canvas/states required')
    if any(s not in ('B','I','S','O','T','U') for s in states):raise ValueError('Unknown state')
    hidden=geometry['initial_self_hidden'];points=_eligible_points(frozen_fit,states,hidden,image.shape[:2])
    # Only already eligible non-S coordinates enter this native direction map.
    projection={k:np.asarray(geometry['initial_projection_xy'][k],float) for k in points}
    if any(p.shape!=(2,) or not np.isfinite(p).all() for p in projection.values()):raise ValueError('Finite initial model directions required')
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
    edges={};records=[];updated=[deepcopy(points[k]['record']) for k in points];by_id={_id(m):m for m in updated};changed=[]
    for k in range(8):
        record=dict(native_corner=k,state=states[k],corrected=False,pixel_queried=False)
        records.append(record)
        if hidden[k] or states[k] not in ('B','I') or k not in points:
            record['reason']='excluded_before_coordinate_or_RGB_query';continue
        record.update(pixel_queried=True,input_xy=points[k]['xy'].tolist(),
                      original_source=points[k]['record'].get('source'),original_corrected=points[k]['corrected'])
        rays=[]
        for mate in NEIGHBORS[k]:
            if mate not in points:continue
            key=(k,mate)
            if key not in edges:edges[key]=_edge(image,mask,lab,gradients,k,mate,points,projection,states,cfg)
            rays.append(edges[key])
        record['rays']=rays;record['eligible_neighbor_ids']=[r['native_neighbor'] for r in rays]
        supported=[r for r in rays if r['supported']];pairs=[];margin=cfg['along_start_px']+cfg['boundary_context_px']
        for first,second in combinations(supported,2):
            matrix=np.asarray([first['normal'],second['normal']]);sine=abs(float(np.linalg.det(matrix)))
            if sine<cfg['minimum_intersection_sine']:continue
            point=np.linalg.solve(matrix,-np.asarray([first['offset'],second['offset']]))
            support=_support(point,[first,second],margin);shift=float(np.linalg.norm(point-points[k]['xy']));h,w=image.shape[:2]
            if support['passed'] and shift<=cfg['maximum_corner_shift_px'] and 0<=point[0]<w and 0<=point[1]<h:
                pairs.append(dict(first=first,second=second,point=point,sine=sine,support=support,shift=shift))
        record['admissible_pair_count']=len(pairs)
        if not pairs:
            record['reason']='no_two_independent_supported_finite_RGB_edges';continue
        disagreement=max(float(np.linalg.norm(a['point']-b['point'])) for a in pairs for b in pairs)
        record['candidate_disagreement_px']=disagreement
        if disagreement>cfg['maximum_candidate_disagreement_px']:
            record['reason']='ambiguous_global_incident_intersections';continue
        chosen=max(pairs,key=lambda r:r['sine']);first,second,point=chosen['first'],chosen['second'],chosen['point']
        payload=dict(edge_records=[first,second],corner_records=[dict(corner=k,refined=True,candidate=point.tolist(),incident_shared_edges=[first['edge'],second['edge']])])
        covariance,_=intersection_uncertainty(payload,repetitions=cfg['line_resampling'],block_length=3,normal_floor_px=1.,seed=cfg['seed'])
        cov=covariance.get(k)
        if cov is None or cov['successful_pair_resamples']<cfg['minimum_successful_pair_resamples']:
            record['reason']='insufficient_conditional_line_resamples';continue
        sigma=np.asarray(cov['covariance_px2'],float)
        if not np.isfinite(sigma).all() or np.min(np.linalg.eigvalsh(sigma))<=0:raise ValueError('Invalid conditional RGB intersection covariance')
        by_id[k].update(xy=point.tolist(),covariance_px2=sigma.tolist(),source=states[k]+'_ALL_VISIBLE_GLOBAL_RGB_EDGE_INTERSECTION',
            image_measurement=True,measurement_is_pose_projection=False)
        record.update(corrected=True,reason='supported_full_edge_RGB_intersection',measured_xy=point.tolist(),
            intersection_xy=point.tolist(),intersection_sine=chosen['sine'],applied_shift_px=chosen['shift'],
            selected_rays=[first,second],finite_support_guard=chosen['support'],conditional_covariance=cov,
            observed_covariance_is_calibrated_total_error=False,old_effective_fit_covariance_preserved=True)
        changed.append(k)
    return C.finite(dict(schema='VISIBLE_BI_GLOBAL_EDGE_REEVALUATION_V4_9C',configuration=cfg,
        visible_measurements=updated,records=records,edge_records=list(edges.values()),changed_native_ids=changed,
        image_pixel_sha256=OLD.image_sha256(image),image_hw=list(image.shape[:2]),states=list(states),
        queried_native_ids=[r['native_corner'] for r in records if r['pixel_queried']],
        S_coordinate_reads=0,S_RGB_queries=0,U_coordinate_reads=0,U_RGB_queries=0,
        O_T_coordinate_reads=0,O_T_RGB_queries=0,GT_inputs=False,pose_fits=0,calibration_fits=0,
        original_effective_fit_covariance_and_reliability_preserved=True,
        all_eligible_visible_BI_remeasured=True,
        unsupported_or_ambiguous_BI_preserved=True,
        mask_context_is_conditional_ownership_not_certificate=True,
        no_partial_I_solver_or_fake_full_KP=True))
