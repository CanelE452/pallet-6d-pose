"""Two-DOF top-normal plus existing archived tangents, saved-start joint SE3.

New fork of the analytic v11 candidate/constraint body. All hard OT/U/BI/
partial guards are unchanged. Normal cut locus invalidates that objective
candidate, not the frame/pool. No new full P3P/global search or image query.
"""
from copy import deepcopy
from types import SimpleNamespace
from itertools import product
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from . import statewise_global_outline_solver_v4_11 as ANALYTIC
from . import statewise_partial_BI_outline_solver_v4_13 as G13
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from . import statewise_archived_BI_tangent_outline_solver_v4_15 as T_OUTLINE
from . import top_plane_normal_factor_v4_16 as NORMAL

V=ANALYTIC.V;J=ANALYTIC.J
_outline_config=ANALYTIC._outline_config
_nearest_residual=ANALYTIC._nearest_residual
outline_value_and_gradient=ANALYTIC.outline_value_and_gradient
observe_global_outline=ANALYTIC.observe_global_outline
DEFAULT_OUTLINE_CONFIGURATION=ANALYTIC.DEFAULT_OUTLINE_CONFIGURATION

def _data_objective(*args):
    return T.objective_with_tangent(*args,prediction_tangent_weight=1.,tangent_variance_inflation=4.)
P=SimpleNamespace(**{**vars(G13.P),'_joint_objective_pair':_data_objective,
    '_boundary_inputs':T._boundary_inputs,'validate_complete_BI_likelihood_revocations':T.validate_archived_predictions})
_prepare=T.T._clone(G13._prepare,{'P':P})

def _normal_parameters(parameters):
    if not isinstance(parameters,dict):raise ValueError('Explicit frozen CAL normal parameters required')
    sigma=float(parameters['sigma_angle_rad']);weight=float(parameters.get('weight',1.));delta=float(parameters.get('huber_scale',3.))
    if not np.isfinite(sigma)or sigma<=0 or not np.isfinite(weight)or weight<0 or not np.isfinite(delta)or delta<=0:
        raise ValueError('Positive normal sigma/delta, nonnegative weight required')
    return dict(sigma_angle_rad=sigma,weight=weight,huber_scale=delta)

def _normal_safe(p,R,parameters):
    try:
        value,gradient,audit=NORMAL.normal_value_and_gradient(p,R,**parameters)
        return value,gradient,{**audit,'valid_normal_math':True}
    except NORMAL.NormalCutLocusError as error:
        return 1e12,np.zeros(6),dict(valid_normal_math=False,reason=str(error),exact_antipode_is_not_zero_residual=True)

def _finalize(result,regions,boundary,xyz,K,scale,cfg):
    result.setdefault('boundary_factor_audit',{})
    result=G13._finalize_outline(result,regions,boundary,xyz,K,scale,cfg)
    result=T.augment(result,xyz,K,boundary,1.,4.,'statewise_top_normal_outline_solver_v4_16')
    active=bool(result['tangent_prediction_factor_audit']['active_native_ids'])
    result['boundary_factor_audit'].update(uncertain_archived_tangent_cost_active=active,
        raw_tangent_has_no_point_cost=not active,old_complete_2D_likelihood_restored=False,
        complete_BI_2D_likelihood_replaced=bool(boundary))
    return result

def _reject(result,regions,boundary,xyz,K,scale,cfg,parameters,reason):
    result.update(accepted=False,status='no_joint_top_normal_refinement_found',fallback_reason=reason,
        R=None,t=None,R_physical=None,projected_native=None,rigid_projection8=None,
        objective=None,objective_components={},reported_native=[None]*8,boundary_observation_audit=[],
        OT_constraints_satisfied=False,U_constraints_satisfied=False,OT_solved=False,
        feasible_pose_samples=[],pose_family_candidates=[],minimum_joint_slack=None)
    result.setdefault('fallback_pose',{})
    result.setdefault('top_plane_normal_prior_audit',dict(parameters=parameters,active=False,reason=reason))
    return _finalize(result,regions,boundary,xyz,K,scale,cfg)

def refine_global_outline_joint_pose(xyz,K,states,visible,domains,image_hw,coarse_pose,*,
        outline_observation,initial_solutions,initializations=None,u_regions=None,boundary_observations=None,
        config=None,outline_configuration=None,normal_parameters=None):
    """Additional continuous six-DOF joint fit, initialized by saved solutions.

    ``initial_solutions`` must use this exact native basis and Q. Optional
    extra R/t initializations must already be transformed into this basis.
    No new P3P/global search is claimed; the caller binds the old full search.
    None/zero normal delegates the archived-tangent parent exactly. Active
    normal uses saved-start continuous fitting even if the outline is absent;
    a normal prior cannot replace insufficient original image support.
    """
    ocfg=_outline_config(outline_configuration)
    if not isinstance(initial_solutions,(list,tuple)) or not initial_solutions:raise ValueError('At least one saved solution required')
    parameters=(_normal_parameters(normal_parameters) if normal_parameters is not None
                else dict(sigma_angle_rad=1.,weight=0.,huber_scale=3.))
    if parameters['weight']==0:
        return T_OUTLINE.refine_global_outline_joint_pose(xyz,K,states,visible,domains,image_hw,coarse_pose,
            outline_observation=outline_observation,initial_solutions=initial_solutions,initializations=initializations,
            u_regions=u_regions,boundary_observations=boundary_observations,config=config,outline_configuration=outline_configuration,
            prediction_tangent_weight=1.,tangent_variance_inflation=4.)
    use_outline=bool(ocfg['outline_factor_weight']>0 and outline_observation.get('supported'))
    xyz,K,states,measurements,domains,image_hw,cfg,boundary,regions,Q,prior,requested=_prepare(
        xyz,K,states,visible,domains,image_hw,coarse_pose,config,boundary_observations,u_regions)
    if len(measurements)<2 or (len(measurements)==2 and not (domains or regions or boundary)):
        return _reject(deepcopy(initial_solutions[0]),regions,boundary,xyz,K,
            max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3),cfg,parameters,
            'insufficient original data support; normal prior is not an image measurement')
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
        if use_outline:extra,eg,ea=outline_value_and_gradient(p,xyz,K,scale,outline_observation,ocfg)
        else:extra,eg,ea=0.,np.zeros(6),dict(valid_projection=True,supported=False,outline_factor_used=False)
        nv,ng,na=_normal_safe(p,prior['R'],parameters)
        value+=nv;gradient+=ng;parts={**parts,'top_plane_normal_prior_robust_loss':nv,'top_plane_normal_prior':na,
            'valid_objective_math':na['valid_normal_math']}
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
            if feasible and parts['valid_objective_math'] and parts['global_outline_audit'].get('valid_projection') and not any(V._same_pose((R,t),(s['R'],s['t']),scale) for s in solutions):
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
    result=deepcopy(initial_solutions[0]);result['schema']='statewise_top_normal_outline_solver_v4_16'
    result['parent_search_audit']=deepcopy(result.get('search'));result['candidates']=records
    result['top_plane_normal_prior_audit']=dict(parameters=parameters,coarse_normal_center=(prior['R']@NORMAL.TOP_NORMAL_CF).tolist(),
        oriented_normal_only=True,native_yaw_position_depth_direct_cost=False,correlated_backbone_regularizer=True,
        prior_not_counted_as_image_information=True,independent_probability_or_uniqueness_claim=False,
        invalid_cut_locus_evaluations=sum(not r.get('objective_components',{}).get('top_plane_normal_prior',{}).get('valid_normal_math',True)for r in records))
    result.setdefault('provenance',{}).update(initial_full_pose_prior=False,initial_top_plane_normal_prior=True,
        legacy_initial_pose_prior_flag_means_full_6DOF_prior_only=True)
    result['global_outline_factor_audit']=dict(configuration=ocfg,outline_factor_used=use_outline,requested_prior_weight=requested,effective_prior_weight=0.,
        engineering_normal_noise_not_CAL_calibrated=True,
        outline_gradient="analytic piecewise nearest-segment, clipping and arc-weight derivative",global_convex_shape_not_visibility_certificate=True,
        mask_binding=deepcopy(outline_observation.get('image_binding')),mask_array_sha256=outline_observation['mask_array_sha256'],
        all_OT_U_partial_tangent_guards_preserved=True,BI_q_observed_and_fit_covariances_preserved=True,
        S_coordinate_reads=0,S_local_RGB_queries=0,U_local_RGB_queries=0,global_mask_geometry_uses_full_cuboid=True)
    result['search']=dict(saved_global_search_reused_as_initializations=True,new_P3P_or_global_starts=0,
        joint_initializations_per_branch=len(unique),T_union_branches_searched=len(list(product(*choices))),
        continuous_SE3_refinement=True,global_optimality_proved=False)
    if not solutions:
        return _reject(result,regions,boundary,xyz,K,scale,cfg,parameters,
            'saved and refined candidates failed independent joint guards or valid normal mathematics')
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
    actual=np.asarray(outline_observation.get('observed_boundary_samples_xy',[]),float).reshape((-1,2))
    def global_residual(position):
        rot,trans=V._pose(position,scale);projection,_=V.project_native(xyz,rot,trans,K)
        hull=projection[ConvexHull(projection).vertices]
        return _nearest_residual(actual,hull).ravel()/(ocfg['normal_sigma_px']*np.sqrt(len(actual)))
    epsilon=ocfg['numeric_derivative_step'];global_jacobian=np.empty((len(actual)*2,6))
    for k in range(6):
        delta=np.zeros(6);delta[k]=epsilon
        if use_outline:global_jacobian[:,k]=(global_residual(p+delta)-global_residual(p-delta))/(2*epsilon)
        else:global_jacobian[:,k]=0.
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
        global_outline_shape_factor=True,initial_pose_prior=False,outline_noise_is_engineering_assumption=True)
    normals=V.registered_incident_normals(xyz);camera=-R.T@t
    finalself=[i for i,n in enumerate(normals) if np.max(n@(camera-xyz[i])/np.linalg.norm(camera-xyz[i]))<-np.sin(np.deg2rad(cfg['geometry_tolerance_deg']))]
    initialself=[i for i,s in enumerate(states) if s=='S']
    result['final_model_visibility_audit']=dict(initial_S_ids=initialself,final_model_self_ids=finalself,
        S_set_changed=set(initialself)!=set(finalself),S_state_is_initial_geometry_conditional=True)
    return _finalize(result,regions,boundary,xyz,K,scale,cfg)
