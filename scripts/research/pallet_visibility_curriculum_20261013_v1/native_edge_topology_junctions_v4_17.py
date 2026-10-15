"""Opt-in B silhouette junction proposals from actual NMS edge chains.

Two distinct named incident *external* chains can propose a junction only
when their observed finite support and the same actual external mask contour
agree locally. I internal creases remain separate partial evidence. Every
admissible alternative is retained; no original observation/state/noise is
changed, and these proposals are never automatically precise corner inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from itertools import combinations
import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import face_aware_partial_bi_observations_v4_13d as D
from . import image_supported_edge_chains_v4_17b as CHAINS
from . import joint_ot_pnp as V3
from . import visible_bi_global_edge_correction_v4_9c as G
from .interior_observation import NEIGHBORS


def mask_pixel_sha256(mask):
    value=np.ascontiguousarray(mask,dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _sample(mask,points):
    return G._sample(mask,np.asarray(points,float)[:,None,:])[:,0]


def _topology(mask):
    binary=(mask>=.5).astype(np.uint8)
    _,labels=cv2.connectedComponents(binary,connectivity=8)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    loops=[]
    for contour in contours:
        xy=contour[:,0,:].astype(float)
        if len(xy)<2:continue
        ids=labels[xy[:,1].astype(int),xy[:,0].astype(int)]
        ids=ids[ids>0]
        if not len(ids):continue
        component=int(np.bincount(ids).argmax())
        segment=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
        loops.append(dict(component=component,xy=xy,tree=cKDTree(xy),
                          cumulative=np.r_[0.,np.cumsum(segment)],perimeter=float(segment.sum())))
    return labels,loops


def _foreground_component_at_probe(labels,mask,probe):
    """Use actual bilinear foreground contributors, with no invented pixels."""
    h,w=mask.shape;result=[]
    for point in probe:
        x,y=np.floor(point).astype(int);candidates=[]
        for yy in (y,y+1):
            for xx in (x,x+1):
                if not(0<=xx<w and 0<=yy<h) or labels[yy,xx]<=0:continue
                weight=max(0.,1-abs(point[0]-xx))*max(0.,1-abs(point[1]-yy))*mask[yy,xx]
                if weight>0:candidates.append((weight,int(labels[yy,xx])))
        result.append(max(candidates)[1] if candidates else 0)
    return np.asarray(result,int)


def _validate_chain(chain,k,mate,cfg):
    if (chain.get('native_corner')!=k or chain.get('native_neighbor')!=mate
            or chain.get('measurement_source')!='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT'
            or chain.get('mask_context_kind') not in ('TARGET_BACKGROUND_OUTLINE','TARGET_TARGET_INTERNAL')):
        raise ValueError('Bound actual NMS chain and explicit native edge required')
    observed=np.asarray(chain['observed_points'],float);good=np.asarray(chain['inlier_mask'],bool)
    normal=np.asarray(chain['normal'],float);offset=float(chain['offset'])
    if (observed.ndim!=2 or observed.shape[1]!=2 or good.shape!=(len(observed),)
            or not np.isfinite(observed).all() or normal.shape!=(2,)
            or not np.isfinite(normal).all() or not np.isfinite(offset)
            or not np.isclose(np.linalg.norm(normal),1.,atol=1e-7,rtol=0)):
        raise ValueError('Finite actual inlier points and unit TLS normal required')
    summary=np.asarray([chain['inlier_fraction'],chain['inlier_span_fraction'],
                        chain['inlier_rms_px'],chain['angle_change_deg']],float)
    if (not np.isfinite(summary).all() or (summary<0).any()
            or not np.isclose(good.mean(),chain['inlier_fraction'],atol=1e-12,rtol=0)
            or good.sum()<cfg['minimum_inlier_count'] or good.sum()!=chain['inlier_count']
            or chain['inlier_fraction']<cfg['minimum_inlier_fraction']
            or chain['inlier_span_fraction']<cfg['minimum_inlier_span_fraction']
            or chain['inlier_rms_px']>cfg['maximum_line_rms_px']
            or chain['angle_change_deg']>cfg['maximum_direction_change_deg']
            or chain.get('corrected_endpoint_normal_guard_pass') is not True):
        raise ValueError('Existing actual line/support/endpoint guards required')
    for key,minimum in (('RGB_gradient_energy',cfg['minimum_gradient']),
                        ('RGB_alignment',cfg['minimum_gradient_alignment']),
                        ('RGB_contrast',cfg['minimum_color_transition'])):
        values=np.asarray(chain[key],float)
        if values.shape!=(len(observed),) or not np.isfinite(values).all() or (values[good]<minimum).any():
            raise ValueError('Actual positive RGB support required: '+key)
    actual=observed[good];tangent=np.array([-normal[1],normal[0]])
    interval=np.asarray(chain['actual_inlier_tangent_interval_px'],float)
    projected=actual@tangent
    if interval.shape!=(2,) or not np.allclose(interval,[projected.min(),projected.max()],atol=1e-7,rtol=0):
        raise ValueError('Actual finite support must match measured inlier pixels')
    rms=float(np.sqrt(np.mean((actual@normal+offset)**2)))
    if not np.isclose(rms,chain['inlier_rms_px'],atol=1e-7,rtol=0):
        raise ValueError('Actual RMS differs from archived chain')
    return observed,good,normal,offset,tangent,interval


def _chain_topology(chain,k,mate,mask,labels,loops,cfg):
    observed,good,normal,offset,tangent,interval=_validate_chain(chain,k,mate,cfg)
    gradient_normals=np.asarray(chain['image_gradient_normal_xy'],float)
    if gradient_normals.shape!=observed.shape or not np.isfinite(gradient_normals).all():
        raise ValueError('Actual image-gradient direction per observed sample required')
    if not np.allclose(np.linalg.norm(gradient_normals,axis=1),1.,atol=1e-6,rtol=0):
        raise ValueError('Unit actual image-gradient normals required')
    probe=cfg['RGB_probe_px']
    plus=observed+gradient_normals*probe;minus=observed-gradient_normals*probe
    mp,mm=_sample(mask,plus),_sample(mask,minus)
    if (not np.allclose(mp,chain['mask_plus'],atol=1e-6,rtol=0)
            or not np.allclose(mm,chain['mask_minus'],atol=1e-6,rtol=0)):
        raise ValueError('Actual chain occupancy evidence differs from supplied mask')
    internal=chain['mask_context_kind']=='TARGET_TARGET_INTERNAL'
    if internal:
        context=(mp>=cfg['minimum_inside'])&(mm>=cfg['minimum_inside'])
    else:
        context=((np.maximum(mp,mm)>=cfg['minimum_inside'])&(np.minimum(mp,mm)<=cfg['maximum_outside'])&
                 (np.abs(mp-mm)>=cfg['minimum_transition']))
    if not context[good].all():raise ValueError('Actual inlier target/background context is not supported')
    if internal:
        return dict(eligible_external=False,reason='I_internal_crease_retained_separately_not_B_silhouette')
    inward=np.where((mp>=mm)[:,None],gradient_normals,-gradient_normals)
    inside=np.where((mp>=mm)[:,None],plus,minus)
    components=_foreground_component_at_probe(labels,mask,inside[good])
    positive=components[components>0]
    if not len(positive):return dict(eligible_external=False,reason='no_actual_foreground_component')
    component=int(np.bincount(positive).argmax())
    fraction=float(np.mean(components==component))
    if fraction<cfg['minimum_inlier_fraction']:
        return dict(eligible_external=False,reason='actual_chain_spans_conflicting_mask_components',
                    component_support_fraction=fraction)
    possible=[]
    actual=observed[good]
    for index,loop in enumerate(loops):
        if loop['component']!=component:continue
        distance,nearest=loop['tree'].query(actual)
        exterior_fraction=float(np.mean(distance<=cfg['boundary_context_px']))
        if exterior_fraction>=cfg['minimum_inlier_fraction']:
            possible.append(dict(loop=index,exterior_support_fraction=exterior_fraction,
                median_distance_to_actual_exterior_px=float(np.median(distance)),
                actual_inlier_nearest_contour_indices=np.asarray(nearest,int).tolist()))
    if not possible:
        return dict(eligible_external=False,reason='target_background_edge_is_not_supported_by_actual_external_contour',
                    component=component,component_support_fraction=fraction)
    mean_inward=inward[good].mean(axis=0);norm=float(np.linalg.norm(mean_inward))
    if norm<=1e-12:return dict(eligible_external=False,reason='ambiguous_actual_inside_direction')
    return dict(eligible_external=True,component=component,component_support_fraction=fraction,
                external_loop_support=possible,measured_inward_unit_xy=(mean_inward/norm).tolist(),
                normal=normal,offset=offset,tangent=tangent,interval=interval,actual=actual)


def _junction_topology(point,first,second,mask,loops,cfg):
    context=float(cfg['boundary_context_px']);margin=float(cfg['along_start_px']+context)
    if first['component']!=second['component']:
        return dict(passed=False,reason='incident_chains_belong_to_different_foreground_components')
    audit=[]
    for evidence in (first,second):
        signed=(evidence['actual']-point)@evidence['tangent']
        nearest_endpoint=min(abs(float(signed.min())),abs(float(signed.max())))
        dominant_fraction=max(float(np.mean(signed>=0)),float(np.mean(signed<=0)))
        passed=nearest_endpoint<=margin and dominant_fraction>=cfg['minimum_inlier_fraction']
        audit.append(dict(nearest_actual_support_endpoint_tangent_distance_px=nearest_endpoint,
                          actual_support_one_sided_fraction=dominant_fraction,passed=passed))
    if not all(x['passed'] for x in audit):
        return dict(passed=False,reason='intersection_is_not_near_two_actual_chain_ends',chain_end_support=audit)
    shared=set(x['loop'] for x in first['external_loop_support'])&set(x['loop'] for x in second['external_loop_support'])
    local=[]
    for index in sorted(shared):
        loop=loops[index]
        distance,corner_index=loop['tree'].query(point)
        if distance>context:continue
        endpoints=[]
        for evidence in (first,second):
            nearest=evidence['actual'][np.argmin(np.linalg.norm(evidence['actual']-point,axis=1))]
            _,i=loop['tree'].query(nearest);endpoints.append(int(i))
        length=abs(float(loop['cumulative'][endpoints[0]]-loop['cumulative'][endpoints[1]]))
        arc=min(length,loop['perimeter']-length)
        if arc<=2*margin:
            local.append(dict(loop=index,component=first['component'],
                junction_distance_to_actual_external_contour_px=float(distance),
                nearest_junction_contour_index=int(corner_index),chain_end_contour_indices=endpoints,
                shortest_actual_external_contour_arc_px=arc,maximum_arc_from_two_11px_support_budgets=2*margin))
    if not local:
        return dict(passed=False,reason='no_short_same_external_contour_arc_connects_actual_chain_ends',
                    chain_end_support=audit)
    inward=np.asarray(first['measured_inward_unit_xy'])+np.asarray(second['measured_inward_unit_xy'])
    length=float(np.linalg.norm(inward))
    if length<=1e-12:return dict(passed=False,reason='opposed_inside_directions_do_not_define_B_corner')
    inward/=length
    probes=np.asarray([point+inward*context,point-inward*context])
    h,w=mask.shape
    if (probes[:,0]<0).any() or (probes[:,0]>=w-1).any() or (probes[:,1]<0).any() or (probes[:,1]>=h-1).any():
        return dict(passed=False,reason='local_B_wedge_context_outside_actual_canvas')
    values=_sample(mask,probes)
    passed=bool(values[0]>=cfg['minimum_inside'] and values[1]<=cfg['maximum_outside'])
    return dict(passed=passed,reason='same_actual_external_arc_and_foreground_wedge' if passed else 'actual_mask_wedge_not_B_corner',
                chain_end_support=audit,local_external_contours=local,
                actual_mask_inside_outside_probe_xy=probes.tolist(),actual_mask_inside_outside_values=values.tolist(),
                inside_direction_from_actual_chain_context=inward.tolist(),
                topology_is_conditional_mask_evidence_not_native_identity_certificate=True)


def propose_native_edge_junctions(chain_observation,pallet_mask,geometry,states,visible_measurements,*,
                                    image_binding,mask_binding,configuration=None):
    """Preserve all B two-outline candidates; I remains typed internal evidence."""
    if not isinstance(chain_observation,Mapping) or chain_observation.get('schema')!='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17B':
        raise ValueError('Bound opt-in17b actual-image NMS chain observation required')
    cfg=CHAINS._config(configuration)
    if chain_observation['configuration']!=cfg:raise ValueError('Same frozen chain/line recipe required')
    states=[V3._state(x) for x in states];hidden=geometry['initial_self_hidden']
    mask=np.asarray(pallet_mask,float)
    if (len(states)!=8 or len(hidden)!=8 or mask.shape!=tuple(geometry['image_hw'])
            or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1))):
        raise ValueError('Bound states, original canvas and occupancy mask required')
    D._sha(image_binding.get('sha256'),'image')
    if (image_binding!=chain_observation['image_binding'] or tuple(image_binding['image_hw'])!=mask.shape
            or not isinstance(mask_binding,Mapping) or mask_binding.get('image_sha256')!=image_binding['sha256']
            or mask_binding.get('mask_pixel_sha256')!=mask_pixel_sha256(mask)):
        raise ValueError('Current chain/image/mask binding differs')
    points=G._eligible_points(visible_measurements,states,hidden,mask.shape)
    face=D._face_context(geometry);labels,loops=_topology(mask)
    per_corner={k:[] for k in points};rejected=[]
    for ray_index,ray in enumerate(chain_observation['edge_records']):
        k=V3._id(ray)
        if hidden[k] or states[k] not in ('B','I') or k not in points:continue
        mate=ray.get('native_neighbor')
        if isinstance(mate,bool) or not isinstance(mate,(int,np.integer)) or not 0<=mate<8:
            raise ValueError('Valid native neighbor required')
        if hidden[mate] or states[mate]=='S':continue
        if mate not in NEIGHBORS[k]:raise ValueError('Actual incident native edge required')
        if not ray.get('supported'):continue
        role=D._edge_face_role(face,k,mate)['role']
        required='OUTLINE_VISIBLE_HIDDEN_FACES' if states[k]=='B' else 'INTERNAL_TWO_VISIBLE_FACES'
        if ray['face_association']['role']!=role:raise ValueError('Initial conditional face association differs')
        if role!=required:
            rejected.append(dict(native_corner=k,native_neighbor=int(mate),reason='B_outline_and_I_internal_crease_are_not_mixed'));continue
        for index,chain in enumerate(ray['chains']):
            expected='TARGET_BACKGROUND_OUTLINE' if states[k]=='B' else 'TARGET_TARGET_INTERNAL'
            if chain.get('mask_context_kind')!=expected:raise ValueError('Conditional face and actual mask contexts differ')
            evidence=_chain_topology(chain,k,int(mate),mask,labels,loops,cfg)
            ref=dict(ray_index=ray_index,chain_index=index,native_corner=k,native_neighbor=int(mate))
            per_corner[k].append(dict(ref=ref,chain=chain,evidence=evidence))
    records=[];h,w=mask.shape;margin=cfg['along_start_px']+cfg['boundary_context_px']
    for k,state in enumerate(states):
        record=dict(native_corner=k,state=state,candidates=[],full_KP_adopted=False)
        records.append(record)
        if hidden[k] or state not in ('B','I') or k not in points:
            record['reason']='excluded_before_observation_and_coordinate_fields';continue
        if state=='I':
            record.update(reason='I_internal_creases_remain_separate_partial_evidence',
                          internal_chain_refs=[x['ref'] for x in per_corner[k]])
            continue
        eligible=[x for x in per_corner[k] if x['evidence'].get('eligible_external')]
        record['external_chain_refs']=[x['ref'] for x in eligible]
        record['rejected_chain_topology']=[dict(ref=x['ref'],audit=x['evidence']) for x in per_corner[k] if not x['evidence'].get('eligible_external')]
        pair_rejections=[]
        for first,second in combinations(eligible,2):
            if first['ref']['native_neighbor']==second['ref']['native_neighbor']:continue
            a,b=first['evidence'],second['evidence'];matrix=np.asarray([a['normal'],b['normal']])
            sine=abs(float(np.linalg.det(matrix)))
            refs=[first['ref'],second['ref']]
            if sine<cfg['minimum_intersection_sine']:
                pair_rejections.append(dict(refs=refs,reason='nearly_parallel_external_lines',intersection_sine=sine));continue
            point=np.linalg.solve(matrix,-np.asarray([a['offset'],b['offset']]))
            support=[]
            for evidence in (a,b):
                t=float(evidence['tangent']@point);lo,hi=evidence['interval']
                gap=max(float(lo-t),float(t-hi),0.)
                support.append(dict(actual_tangent_interval_px=evidence['interval'].tolist(),extrapolation_px=gap,passed=gap<=margin))
            shift=float(np.linalg.norm(point-points[k]['xy']))
            if not all(x['passed'] for x in support) or shift>cfg['maximum_corner_shift_px'] or not(0<=point[0]<w and 0<=point[1]<h):
                pair_rejections.append(dict(refs=refs,reason='existing_finite_support_shift_or_canvas_guard_failed',
                                            actual_finite_support=support,shift_px=shift));continue
            topology=_junction_topology(point,a,b,mask,loops,cfg)
            if not topology['passed']:
                pair_rejections.append(dict(refs=refs,reason=topology['reason'],topology=topology));continue
            record['candidates'].append(dict(chain_refs=refs,intersection_xy=point.tolist(),intersection_sine=sine,
                actual_finite_support=support,shift_from_archived_B_observation_px=shift,local_topology=topology,
                candidate_is_not_adopted_precise_KP=True,requires_separate_conditional_bootstrap_before_any_full_point_adoption=True,
                total_error_calibration_or_native_identity_not_proved=True))
        record['rejected_pairs']=pair_rejections
        candidate=record['candidates']
        disagreement=max((float(np.linalg.norm(np.asarray(a['intersection_xy'])-b['intersection_xy']))
                          for a in candidate for b in candidate),default=0.)
        record.update(candidate_count=len(candidate),candidate_disagreement_px=disagreement,
            geometrically_unique_within_existing_3px_rule=bool(candidate) and disagreement<=cfg['maximum_candidate_disagreement_px'],
            precise_point_selected=False,
            reason=('topology_supported_B_junction_candidates_preserved' if candidate else
                    'no_two_distinct_supported_external_chains_with_consistent_local_topology'))
    return dict(schema='NATIVE_EDGE_TOPOLOGY_JUNCTION_PROPOSALS_V4_17',records=records,
        rejected_contexts=rejected,configuration=cfg,image_binding=deepcopy(dict(image_binding)),mask_binding=deepcopy(dict(mask_binding)),
        S_coordinate_reads=0,S_local_queries=0,U_coordinate_reads=0,U_corner_queries=0,O_T_coordinate_reads=0,
        old_q_covariance_state_changes=0,precise_KP_adoptions=0,pose_calls=0,NN_calls=0,GT_inputs=False,
        all_admissible_alternatives_preserved=True,B_outline_and_I_crease_mixed=False,
        actual_mask_topology_is_conditional_not_ownership_certificate=True,
        full_measurement_bootstrap_and_total_error_calibration_run=False)
