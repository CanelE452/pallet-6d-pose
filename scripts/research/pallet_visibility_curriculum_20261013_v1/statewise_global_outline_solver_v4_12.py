"""Version4.12: caller-bound CAL normal bias/noise, unchanged joint guards.
Zero mean and original1px scale recover the v4.11 objective.
"""
"""Version4.11: unchanged v4.10 objective with analytic outline gradient.
No observation, model, likelihood, hard guard or feasible start is dropped.
"""
"""Continuous whole-mask shape refinement with unchanged BI/OT/U constraints.

Global mask geometry is a shape cue, never a keypoint observation. Its outer
convex envelope deliberately fills holes/occlusion and is not a visibility or
physical-axis certificate. Both directions of continuous polygon-segment
distance enter one arc-normalized robust term. No S/U/O/T pixel query, raw
hidden coordinate, GT, image/model forward or newly fitted uncertainty occurs.
Saved full-search poses initialize additional local SE(3) refinement; global
optimality and uniqueness are not proved. All old hard constraints remain.
"""
from copy import deepcopy
from itertools import product
from collections.abc import Mapping
import hashlib

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation

from . import statewise_partial_boundary_u_solver_v4_6 as P
from . import joint_ot_pnp as V
from . import joint_ot_jacobian_v2 as J

DEFAULT_OUTLINE_CONFIGURATION=dict(normal_sigma_px=1.,huber_delta=1.25,
    frame_margin_px=7.,model_samples_per_segment=15,observed_sample_spacing_px=1.,
    outline_factor_weight=1.,numeric_derivative_step=1e-5)


def _outline_config(configuration=None):
    cfg=deepcopy(DEFAULT_OUTLINE_CONFIGURATION)
    if configuration:
        if set(configuration)-set(cfg):raise ValueError('Unknown whole-outline configuration')
        cfg.update(configuration)
    if any(isinstance(x,(bool,np.bool_)) or not np.isfinite(x) for x in cfg.values()):
        raise ValueError('Finite numeric outline configuration required')
    if any(cfg[k]<=0 for k in ('normal_sigma_px','huber_delta','model_samples_per_segment',
            'observed_sample_spacing_px','numeric_derivative_step')) or cfg['frame_margin_px']<0 or cfg['outline_factor_weight']<0:
        raise ValueError('Positive outline scales and nonnegative weight/margin required')
    if int(cfg['model_samples_per_segment'])!=cfg['model_samples_per_segment'] or cfg['model_samples_per_segment']<2:
        raise ValueError('At least two integral segment samples required')
    cfg['model_samples_per_segment']=int(cfg['model_samples_per_segment'])
    return cfg


def _clip(a,b,bounds):
    direction=b-a;lo,hi=0.,1.
    for axis in (0,1):
        if abs(direction[axis])<1e-14:
            if not bounds[axis][0]<=a[axis]<=bounds[axis][1]:return None
        else:
            first=(bounds[axis][0]-a[axis])/direction[axis];second=(bounds[axis][1]-a[axis])/direction[axis]
            lo=max(lo,min(first,second));hi=min(hi,max(first,second))
            if lo>=hi:return None
    return a+lo*direction,a+hi*direction


def _segments(polygon):
    return np.asarray(polygon),np.roll(np.asarray(polygon),-1,axis=0)


def _nearest_residual(points,polygon):
    return _nearest_segment_residual(points,np.stack(_segments(polygon),axis=1))


def _nearest_segment_residual(points,segments):
    a,b=np.asarray(segments)[:,0],np.asarray(segments)[:,1];d=b-a;squared=np.sum(d*d,axis=1)
    alpha=np.sum((points[:,None,:]-a[None,:,:])*d[None,:,:],axis=2)/np.maximum(squared,1e-20)
    foot=a[None,:,:]+np.clip(alpha,0,1)[...,None]*d[None,:,:]
    residual=points[:,None,:]-foot;index=np.argmin(np.sum(residual*residual,axis=2),axis=1)
    return residual[np.arange(len(points)),index]


def observe_global_outline(pallet_mask,image_hw,*,image_binding=None,configuration=None):
    """Read an existing global mask once; no detector/KP/raw coordinates."""
    cfg=_outline_config(configuration);h,w=map(int,image_hw);mask=np.asarray(pallet_mask,float)
    if mask.shape!=(h,w) or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
        raise ValueError('Bound finite global mask in [0,1] required')
    y,x=np.nonzero(mask>=.5)
    audit=dict(image_hw=[h,w],image_binding=deepcopy(image_binding),
        mask_array_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
        configuration=cfg,global_mask_only=True,S_local_queries=0,U_local_queries=0,
        convex_envelope_is_not_actual_complete_surface=True)
    if len(x)<3:return {**audit,'supported':False,'reason':'insufficient_mask_foreground'}
    polygon=cv2.convexHull(np.column_stack([x,y]).astype('float32')).reshape(-1,2).astype(float)
    if len(polygon)<3 or abs(cv2.contourArea(polygon.astype('float32')))<1:
        return {**audit,'supported':False,'reason':'degenerate_mask_envelope'}
    margin=cfg['frame_margin_px'];bounds=((margin,w-1-margin),(margin,h-1-margin));points=[];segments=[]
    for a,b in zip(*_segments(polygon)):
        clipped=_clip(a,b,bounds)
        if clipped is None:continue
        a,b=clipped;segments.append([a.tolist(),b.tolist()]);length=float(np.linalg.norm(b-a));n=max(1,int(np.ceil(length/cfg['observed_sample_spacing_px'])))
        points.extend(a[None,:]+((np.arange(n)+.5)/n)[:,None]*(b-a))
    if len(points)<3:return {**audit,'supported':False,'reason':'no_non_canvas_outline'}
    return {**audit,'supported':True,'polygon_xy':polygon.tolist(),
        'observed_boundary_segments_xy':segments,'canvas_closing_segments_not_observations':True,
        'observed_boundary_samples_xy':np.asarray(points).tolist(),
        'mask_area_px':int(len(x)),'foreground_hull_area_px':float(cv2.contourArea(polygon.astype('float32')))}


def _outline_value(p,xyz,K,scale,observation,cfg):
    R,t=V._pose(p,scale);q,depth=V.project_native(xyz,R,t,K)
    if not np.isfinite(q).all() or np.min(depth)<=1e-8:return 1e12,{'valid_projection':False}
    try:polygon=q[ConvexHull(q).vertices]
    except QhullError:return 1e12,{'valid_projection':False}
    observed=np.asarray(observation['observed_boundary_samples_xy'],float)
    actual=np.asarray(observation['observed_boundary_segments_xy'],float)
    reverse=_nearest_residual(observed,polygon);h,w=observation['image_hw'];margin=cfg['frame_margin_px']
    points=[];weights=[];bounds=((margin,w-1-margin),(margin,h-1-margin))
    for a,b in zip(*_segments(polygon)):
        clipped=_clip(a,b,bounds)
        if clipped is None:continue
        a,b=clipped;length=float(np.linalg.norm(b-a));n=cfg['model_samples_per_segment']
        points.extend(a[None,:]+((np.arange(n)+.5)/n)[:,None]*(b-a));weights.extend([length/n]*n)
    if not points:return 1e12,{'valid_projection':False}
    forward=_nearest_segment_residual(np.asarray(points),actual)
    def robust(residual,weight):
        radius=np.linalg.norm(residual,axis=1)/cfg['normal_sigma_px'];delta=cfg['huber_delta']
        loss=delta**2*(np.sqrt(1+(radius/delta)**2)-1)
        return float(np.average(loss,weights=weight))
    first=robust(forward,weights);second=robust(reverse,None)
    def stats(residual):
        distance=np.linalg.norm(residual,axis=1)
        return dict(mean_px=float(distance.mean()),median_px=float(np.median(distance)),P90_px=float(np.quantile(distance,.9)),count=len(distance))
    return .5*(first+second),dict(valid_projection=True,model_to_observed=stats(forward),
        observed_to_model=stats(reverse),model_to_observed_robust_mean=first,observed_to_model_robust_mean=second,
        normalized_by_arc_length_not_duplicated_pixel_count=True)


from .global_outline_calibrated_factor_v4_12 import outline_value_and_gradient


def _outline_value(p,xyz,K,scale,observation,cfg):
    """Scalar audit uses the same signed/calibrated factor as optimization."""
    value,_,audit=outline_value_and_gradient(p,xyz,K,scale,observation,cfg)
    return value,audit


def _prepare(xyz,K,states,visible,domains,image_hw,coarse_pose,config,boundary_observations,u_regions):
    cfg=P._configuration(config);requested=cfg['prior_weight'];cfg['prior_weight']=0.
    source=V._records(visible);original={V._id(r):r for r in source}
    xyz,K,states,measurements,domains,image_hw=V._inputs(xyz,K,states,source,domains,image_hw,P._helper_config(cfg))
    normals=V.registered_incident_normals(xyz)
    if any(n.shape!=(3,3) or not np.allclose(n@n.T,np.eye(3),atol=1e-6,rtol=0) for n in normals):
        raise ValueError('Registered original eight-vertex orthogonal cuboid required')
    boundary=P._boundary_inputs(boundary_observations,states,image_hw,xyz);regions=P.U._u_region_inputs(u_regions,states)
    if not {b['native_corner'] for b in boundary}.issubset(original):raise ValueError('Partial B requires immutable visible record')
    for b in boundary:
        if original[b['native_corner']].get('source','VISIBLE_BACKBONE_FALLBACK')!='VISIBLE_BACKBONE_FALLBACK':
            raise ValueError('Corrected complete B cannot be replaced by partial')
    for m in measurements:
        record=original[m['native_corner']];state=states[m['native_corner']]
        observed=m['covariance_px2'].copy();covariance=np.asarray(record.get('fit_covariance_px2',
            cfg['covariance_scale']*observed+np.asarray(cfg['state_BI_total_error_floor_px2'][state])),float)
        if covariance.shape!=(2,2) or not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0):
            raise ValueError('Symmetric finite effective covariance required')
        weight=float(record.get('fit_reliability_weight',cfg['state_loss_weights'][state]))
        if not np.isfinite(weight) or weight<=0:raise ValueError('Positive frozen reliability required')
        m.update(observed_covariance_px2=observed,observed_whitening=m['whitening'].copy(),
            covariance_px2=covariance.copy(),whitening=np.linalg.inv(np.linalg.cholesky(covariance)),
            fit_reliability_weight=weight,fit_covariance_source='UNCHANGED_CALLER_OR_STATE_TOTAL_COVARIANCE')
    for b in boundary:
        if b['fit_reliability_weight']!=next(m['fit_reliability_weight'] for m in measurements if m['native_corner']==b['native_corner']):
            raise ValueError('Partial reliability must equal frozen B reliability')
    Q=V._finite_array(coarse_pose.get('physical_registration_Q',np.eye(3)),(3,3),'registration Q')
    R=V._finite_array(coarse_pose['R'],(3,3),'coarse R');t=V._finite_array(coarse_pose['t'],(3,),'coarse t')
    if not V._valid_rotation(Q) or not V._valid_rotation(R):raise ValueError('Proper R/Q required')
    if 'R_physical' in coarse_pose and not np.allclose(R@Q,coarse_pose['R_physical'],atol=1e-7,rtol=0):
        raise ValueError('Coarse physical registration binding mismatch')
    prior=dict(R=R,center=R@xyz.mean(0)+t,whitening=np.eye(6))
    return xyz,K,states,measurements,domains,image_hw,cfg,boundary,regions,Q,prior,requested


def refine_global_outline_joint_pose(xyz,K,states,visible,domains,image_hw,coarse_pose,*,
        outline_observation,initial_solutions,initializations=None,u_regions=None,boundary_observations=None,
        config=None,outline_configuration=None):
    """Additional continuous six-DOF joint fit, initialized by saved solutions.

    ``initial_solutions`` must use this exact native basis and Q. Optional
    extra R/t initializations must already be transformed into this basis.
    No new P3P/global search is claimed; the caller binds the old full search.
    Empty/zero/unsupported outline returns the first saved solution exactly.
    """
    ocfg=_outline_config(outline_configuration)
    if not isinstance(initial_solutions,(list,tuple)) or not initial_solutions:raise ValueError('At least one saved solution required')
    if ocfg['outline_factor_weight']==0 or not outline_observation.get('supported'):
        return deepcopy(initial_solutions[0])
    xyz,K,states,measurements,domains,image_hw,cfg,boundary,regions,Q,prior,requested=_prepare(
        xyz,K,states,visible,domains,image_hw,coarse_pose,config,boundary_observations,u_regions)
    if len(measurements)<2 or (len(measurements)==2 and not (domains or regions or boundary)):
        return deepcopy(initial_solutions[0])
    if tuple(outline_observation['image_hw'])!=image_hw:raise ValueError('Current canvas/outline binding mismatch')
    if outline_observation['configuration']!={**ocfg}:raise ValueError('Observed outline construction/config must agree')
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3);starts=[]
    for j,solution in enumerate(initial_solutions):
        if solution.get('accepted'):
            if not np.allclose(np.asarray(solution['R'])@Q,solution['R_physical'],atol=1e-7,rtol=0):raise ValueError('Saved solution uses another native registration')
            starts.append(dict(R=solution['R'],t=solution['t'],name=f'SAVED_{j}_WINNER'))
        for i,sample in enumerate(solution.get('feasible_pose_samples',[])):
            starts.append(dict(R=sample['R'],t=sample['t'],name=f'SAVED_{j}_SAMPLE_{i}'))
    starts.extend(deepcopy(list(initializations or [])))
    unique=[]
    for start in starts:
        R=V._finite_array(start['R'],(3,3),'initial R');t=V._finite_array(start['t'],(3,),'initial t')
        if not V._valid_rotation(R):raise ValueError('Proper starting rotation required')
        if not any(V._same_pose((R,t),(s['R'],s['t']),scale) for s in unique):unique.append(dict(R=R,t=t,name=start.get('name','SUPPLIED')))
    choices=[(list(d['outside_canvas_side']) if isinstance(d['outside_canvas_side'],tuple)
        else [d['outside_canvas_side']] if d['outside_canvas_side'] else list(V._SIDES))
        if d['state']=='T' else [None] for d in domains]
    if int(np.prod([len(c) for c in choices],dtype=int))>cfg['maximum_union_branches']:
        raise ValueError('Full original T-union branch budget exceeded')
    def objective(p):
        value,gradient,parts=P._joint_objective_pair(p,xyz,K,measurements,scale,prior,cfg,boundary,1.)
        extra,eg,ea=outline_value_and_gradient(p,xyz,K,scale,outline_observation,ocfg)
        return value+ocfg['outline_factor_weight']*extra,gradient+ocfg['outline_factor_weight']*eg,{**parts,
            'global_outline_robust_mean_loss':extra,'global_outline_factor_weight':ocfg['outline_factor_weight'],
            'global_outline_audit':ea,'coarse_pose_prior_loss':0.,'S_keypoint_image_terms':0,'U_keypoint_image_terms':0}
    pair=V._cached_value_jacobian(objective);records=[];solutions=[]
    for branch,sides in enumerate(product(*choices)):
        constraint=V._cached_value_jacobian(lambda p:P._joint_constraint_pair(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg,boundary,regions))
        def consider(p,name,optimizer=None):
            R,t=V._pose(p,scale);q,depth=V.project_native(xyz,R,t,K);slack=constraint(p)[0]
            feasible=bool(np.isfinite(q).all() and np.isfinite(slack).all() and np.min(slack)>=-cfg['feasibility_tolerance']
                and np.min(depth)>=cfg['positive_depth_m']-1e-10 and V._valid_rotation(R))
            value,_,parts=pair(p);record=dict(initialization=name,branch=branch,joint_feasible=feasible,
                objective=float(value),objective_components=parts,minimum_constraint_slack=float(np.min(slack)),
                optimizer_success=None if optimizer is None else bool(optimizer.success),
                iterations=0 if optimizer is None else int(optimizer.nit),
                optimizer_message='saved feasible contender' if optimizer is None else str(optimizer.message))
            records.append(record)
            if feasible and parts['global_outline_audit'].get('valid_projection') and not any(V._same_pose((R,t),(s['R'],s['t']),scale) for s in solutions):
                solutions.append(dict(R=R,t=t,q=q,p=np.asarray(p).copy(),sides=sides,name=name,
                    objective=float(value),components=parts,record=record))
        for start in unique:
            p=V._pose_vector(start['R'],start['t'],scale);consider(p,start['name']+'_UNREFINED')
            try:
                answer=minimize(lambda p:pair(p)[0],p,jac=lambda p:pair(p)[1].copy(),method='SLSQP',
                    constraints=[dict(type='ineq',fun=lambda p:constraint(p)[0].copy(),jac=lambda p:constraint(p)[1].copy())],
                    options=dict(maxiter=cfg['max_joint_iterations'],ftol=cfg['optimizer_ftol'],disp=False))
                consider(answer.x,start['name'],answer)
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                records.append(dict(initialization=start['name'],branch=branch,joint_feasible=False,error=str(exc)))
    result=deepcopy(initial_solutions[0]);result['schema']='statewise_global_outline_solver_v4_12'
    result['parent_search_audit']=deepcopy(result.get('search'));result['candidates']=records
    result['global_outline_factor_audit']=dict(configuration=ocfg,requested_prior_weight=requested,effective_prior_weight=0.,
        engineering_normal_noise_is_CAL_learned=bool(outline_observation.get('outline_noise_model_sha256')),
        outline_noise_model_sha256=outline_observation.get('outline_noise_model_sha256'),
        expected_outward_mask_bias_px=float(outline_observation.get("normal_bias_px",0.)),
        conditional_line_repeatability_is_total_noise=False,
        outline_gradient="analytic piecewise nearest-segment, clipping and arc-weight derivative",global_convex_shape_not_visibility_certificate=True,
        mask_binding=deepcopy(outline_observation.get('image_binding')),mask_array_sha256=outline_observation['mask_array_sha256'],
        all_OT_U_partial_tangent_guards_preserved=True,BI_q_observed_and_fit_covariances_preserved=True,
        S_coordinate_reads=0,S_local_RGB_queries=0,U_local_RGB_queries=0,global_mask_geometry_uses_full_cuboid=True)
    result['search']=dict(saved_global_search_reused_as_initializations=True,new_P3P_or_global_starts=0,
        joint_initializations_per_branch=len(unique),T_union_branches_searched=len(list(product(*choices))),
        continuous_SE3_refinement=True,global_optimality_proved=False)
    if not solutions:
        result.update(accepted=False,status='no_joint_outline_refinement_found',fallback_reason='saved_and_refined_candidates_failed_independent_joint_guards',
            R=None,t=None,R_physical=None,projected_native=None,rigid_projection8=None,
            OT_constraints_satisfied=False,U_constraints_satisfied=False,OT_solved=False,
            feasible_pose_samples=[],pose_family_candidates=[])
        return result
    solutions.sort(key=lambda s:(s['objective'],s['name']));chosen=solutions[0];R,t,q,p=chosen['R'],chosen['t'],chosen['q'],chosen['p']
    va=P._visible_audit(q,measurements,-R.T@t,image_hw,cfg)
    da=V._domain_audit(q,domains,chosen['sides'],image_hw,P._helper_config(cfg));ua=P.U._u_audit(q,regions,cfg['feasibility_tolerance'])
    ba=P._boundary_audit(p,xyz,K,boundary,image_hw,scale,cfg,1.)
    if not all(a['visible_support_guards_pass'] for a in va) or not all(a['feasible'] for a in da+ua+ba):
        raise RuntimeError('Final independent scalar joint constraint audit failed')
    alternatives=[]
    for other in solutions[1:]:
        rot=float(np.rad2deg(Rotation.from_matrix(other['R']@R.T).magnitude()));trans=float(np.linalg.norm(other['t']-t)/scale)
        if (rot>cfg['ambiguity_rotation_deg'] or trans>cfg['ambiguity_translation_fraction']) and other['objective']-chosen['objective']<=cfg['ambiguity_objective_gap']:
            alternatives.append(dict(R=other['R'].tolist(),t=other['t'].tolist(),objective_gap=other['objective']-chosen['objective']))
    complete=[m for m in measurements if m['native_corner'] not in {b['native_corner'] for b in boundary}]
    visible_jacobian=J.visible_residual_and_jacobian(p,xyz,K,complete,scale)[1]
    boundary_jacobian=P._normal_residual_and_jacobian(p,xyz,K,boundary,scale)[1]
    boundary_jacobian*=np.repeat([np.sqrt(b['fit_reliability_weight']) for b in boundary],[len(b['lines']) for b in boundary])[:,None]
    actual=np.asarray(outline_observation['observed_boundary_samples_xy'],float)
    def global_residual(position):
        rot,trans=V._pose(position,scale);projection,_=V.project_native(xyz,rot,trans,K)
        hull=projection[ConvexHull(projection).vertices]
        return _nearest_residual(actual,hull).ravel()/(ocfg['normal_sigma_px']*np.sqrt(len(actual)))
    epsilon=ocfg['numeric_derivative_step'];global_jacobian=np.empty((len(actual)*2,6))
    for k in range(6):
        delta=np.zeros(6);delta[k]=epsilon
        global_jacobian[:,k]=(global_residual(p+delta)-global_residual(p-delta))/(2*epsilon)
    image_jacobian=np.vstack([visible_jacobian,boundary_jacobian,np.sqrt(ocfg['outline_factor_weight'])*global_jacobian])
    slack,guards=P._joint_constraint_pair(p,xyz,K,measurements,domains,chosen['sides'],image_hw,scale,cfg,boundary,regions)
    active=np.flatnonzero(slack<=1e-5);joint_rank=V._rank(np.vstack([image_jacobian,guards[active]]))
    family=len(measurements)==2;ambiguous=bool(family or alternatives or joint_rank['rank']<6)
    result.update(accepted=True,status='joint_feasible_outline_pose_family' if ambiguous else 'joint_feasible_outline',
        R=R.tolist(),t=t.tolist(),R_physical=(R@Q).tolist(),projected_native=q.tolist(),rigid_projection8=q.tolist(),
        objective=chosen['objective'],objective_components=chosen['components'],
        visible_protection_audit=va,joint_constraint_audit=da,U_region_constraint_audit=ua,boundary_observation_audit=ba,
        minimum_joint_slack=chosen['record']['minimum_constraint_slack'],OT_constraints_satisfied=True,U_constraints_satisfied=True,
        partial_B_constraints_satisfied=True,OT_solved=bool(domains) and not ambiguous,OT_unique_solution_claimed=False,
        rank=dict(visible=V._rank(visible_jacobian),BI_partial_and_global_outline=V._rank(image_jacobian),
            visible_and_active_joint=joint_rank,active_constraint_count=len(active),
            global_shape_rank_is_not_semantic_KP_uniqueness=True,prior_not_counted_as_image_information=True),
        ambiguity=dict(detected=ambiguous,alternatives=alternatives,pose_family_from_two_BI=family,
            local_underdetermined=joint_rank['rank']<6,global_uniqueness_proved=False))
    result['feasible_pose_samples']=[dict(R=s['R'].tolist(),t=s['t'].tolist(),projected_native=s['q'].tolist(),
        objective=s['objective'],initialization=s['name'],all_OT_individual_constraints_pass=True,
        all_U_individual_constraints_pass=True,all_partial_B_tangent_constraints_pass=True) for s in solutions]
    result['pose_family_candidates']=deepcopy(result['feasible_pose_samples']) if ambiguous else []
    result['reported_native']=[None]*8
    partial={b['native_corner'] for b in boundary}
    for m in measurements:
        if m['native_corner'] not in partial:result['reported_native'][m['native_corner']]=m['xy'].tolist()
    for k,s in enumerate(states):
        if s in ('S','O','T'):result['reported_native'][k]=q[k].tolist()
    for key,value in result.get('boundary_readout',{}).items():
        value['model_projection_xy']=q[int(key)].tolist()
    result['output_corner_sources']=['PARTIAL_BOUNDARY_NORMAL_AND_TANGENT_INTERVAL' if k in partial
        else 'IMMUTABLE_BI_OBSERVATION' if s in ('B','I') else 'S_MODEL_PROJECTION_ONLY' if s=='S'
        else 'UNRESOLVED' if s=='U' else 'AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION' if ambiguous
        else 'JOINT_FEASIBLE_REPRESENTATIVE_PROJECTION' for k,s in enumerate(states)]
    result['prior_audit'].update(prior_weight=0.,enabled=False,bias_active_in_loss=False)
    result.setdefault('provenance',{}).update(GT_inputs=False,S_image_terms=0,S_region_terms=0,
        global_outline_shape_factor=True,initial_pose_prior=False,
        outline_noise_is_engineering_assumption=not bool(outline_observation.get('outline_noise_model_sha256')),
        outline_noise_CAL_model_sha256=outline_observation.get('outline_noise_model_sha256'))
    normals=V.registered_incident_normals(xyz);camera=-R.T@t
    finalself=[i for i,n in enumerate(normals) if np.max(n@(camera-xyz[i])/np.linalg.norm(camera-xyz[i]))<-np.sin(np.deg2rad(cfg['geometry_tolerance_deg']))]
    initialself=[i for i,s in enumerate(states) if s=='S']
    result['final_model_visibility_audit']=dict(initial_S_ids=initialself,final_model_self_ids=finalself,
        S_set_changed=set(initialself)!=set(finalself),S_state_is_initial_geometry_conditional=True)
    return result
