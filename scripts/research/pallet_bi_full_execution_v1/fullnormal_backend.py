"""New full finite-search hierarchy for explicitly revoked normal-only factors.

This is a matched new backend, not original-D1 parity. Numerical starts,
iteration caps, O/T branches, priors and outline primitives are inherited.
Revoked full points never enter PnP, seed objectives, rank or geometry guards;
their immutable typed archives are still authenticated by the original parser.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import product, permutations
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from scripts.research.pallet_bi_conditioned_observation_v1 import solver as LOCAL
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_u_solver_v4_13 as ORIGINAL
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_global_outline_solver_v4_10 as G10
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_outline_solver_v4_13 as OUTLINE
from scripts.research.pallet_selection_score_diagnosis_v1.fixed_cost import Context
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as FEATURES
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import owner_model as OWNER

BASE=ORIGINAL.BASE;N=ORIGINAL.N;U=ORIGINAL.U;V=ORIGINAL.V3;V3=V;J=ORIGINAL.J
P=sys.modules[__name__]
_configuration=BASE._configuration
_helper_config=BASE._helper_config
_boundary_inputs=ORIGINAL._boundary_inputs
validate_complete_BI_likelihood_revocations=ORIGINAL.validate_complete_BI_likelihood_revocations
_joint_objective_pair=ORIGINAL._joint_objective_pair
_normal_residual_and_jacobian=ORIGINAL._normal_residual_and_jacobian
_outline_config=OUTLINE._outline_config
outline_value_and_gradient=OUTLINE.outline_value_and_gradient
_outline_value=OUTLINE._outline_value


def configuration(original):
    result=deepcopy(original)
    result.update(covariance_scale=1.,state_BI_total_error_floor_px2={s:[[0.,0.],[0.,0.]]for s in ('B','I')},state_loss_weights={'B':1.,'I':1.})
    return _configuration(result)


def _complete(measurements,boundary):
    ids={b['native_corner']for b in boundary}
    return [m for m in measurements if m['native_corner']not in ids]


def _joint_constraint_pair(p,xyz,K,measurements,domains,sides,image_hw,scale,cfg,boundary,regions,analytic=True):
    # Normal support intervals remain archived. No partial tangent constraint
    # and no revoked point geometry/canvas row is passed to optimization.
    return U._joint_constraint_pair(p,xyz,K,_complete(measurements,boundary),domains,sides,image_hw,scale,cfg,regions,analytic)


def _prepare(xyz,K,states,visible,domains,image_hw,coarse_pose,config,boundary_observations,u_regions):
    cfg=_configuration(config);source=V._records(visible)
    xyz,K,measurements,hw,boundary=LOCAL._prepare(xyz,K,states,source,image_hw,boundary_observations or[],cfg)
    structural=[s if s in ('O','T','S')else'U'for s in states]
    _,_,_,_,parsed_domains,_=V._inputs(xyz,K,structural,[],domains,hw,_helper_config(cfg))
    regions=U._u_region_inputs(u_regions,states)
    R=V._finite_array(coarse_pose['R'],(3,3),'coarse R');t=V._finite_array(coarse_pose['t'],(3,),'coarse t')
    Q=V._finite_array(coarse_pose.get('physical_registration_Q',np.eye(3)),(3,3),'Q')
    if not V._valid_rotation(R)or not V._valid_rotation(Q):raise ValueError('Proper R/Q required')
    if 'R_physical'in coarse_pose and not np.allclose(R@Q,coarse_pose['R_physical'],atol=1e-7,rtol=0):raise ValueError('Physical registration mismatch')
    center=R@xyz.mean(0)+t
    if center[2]<=0:raise ValueError('Positive registered coarse center required')
    cov=cfg['prior_covariance6']
    if cov is None:
        deviations=np.r_[np.repeat(np.deg2rad(cfg['prior_rotation_sigma_deg']),3),np.repeat(cfg['prior_ray_sigma'],2),cfg['prior_log_depth_sigma']]
        cov=np.diag(deviations**2)
    prior=dict(R=R,center=center,whitening=np.linalg.inv(np.linalg.cholesky(np.asarray(cov,float))))
    return xyz,K,list(states),measurements,parsed_domains,hw,cfg,boundary,regions,Q,prior,cfg['prior_weight']


def _initial_poses(xyz,K,complete,coarse,initializations,cfg):
    supplied=[dict(R=coarse['R'],t=coarse['t'],name='REGISTERED_COARSE_PRIOR'),*deepcopy(initializations or[])]
    if complete:return V._initial_poses(xyz,K,complete,supplied,cfg)
    # The original span-based depth initializer is undefined with zero full
    # points. Keep the same finite rotation list/cap at the known N0 center.
    starts=deepcopy(supplied);center=np.asarray(coarse['R'])@xyz.mean(0)+coarse['t'];rotations=[]
    for order in permutations(range(3)):
        for signs in product([-1,1],repeat=3):
            matrix=np.eye(3)[:,order]*np.asarray(signs)
            if np.linalg.det(matrix)>0:rotations.append(matrix)
    for index,R in enumerate(rotations[:int(cfg['global_rotation_starts'])]):
        t=center-R@xyz.mean(0)
        if not any(V._same_pose((R,t),(np.asarray(s['R']),np.asarray(s['t'])),1.)for s in starts):
            starts.append(dict(name='NORMAL_ONLY_GLOBAL_ROTATION_'+str(index),R=R,t=t))
    return starts,dict(initialization_count=len(starts),p3p_triples_possible=0,p3p_triples_searched=0,
        P3P_or_SQPNP_called=False,global_rotation_start_cap=cfg['global_rotation_starts'],
        global_depth_initialization_rule='no full point span exists; fixed actual N0 physical center',
        uses_revoked_archive_for_initialization=False)


def _branches(domains,cfg):
    choices=[(list(d['outside_canvas_side'])if isinstance(d['outside_canvas_side'],tuple)else[d['outside_canvas_side']]if d['outside_canvas_side']else list(V._SIDES))if d['state']=='T'else[None]for d in domains]
    count=int(np.prod([len(c)for c in choices],dtype=int))
    if count>cfg['maximum_union_branches']:raise ValueError('Original T-union branch cap exceeded')
    return list(product(*choices))


def _empty(coarse,states,cfg):
    return dict(schema='BI_FULL_NORMAL_ONLY_FINITE_SEARCH_V1',accepted=False,status='unresolved',R=None,t=None,R_physical=None,
        objective=None,candidates=[],all_visited_endpoints=[],feasible_pose_samples=[],pose_family_candidates=[],
        fallback_pose=dict(R=deepcopy(coarse['R']),t=deepcopy(coarse['t']),source='INITIAL_REGISTERED_COARSE_PNP'),
        fallback_reason=None,states=list(states),optimizer_calls=0,search={},
        provenance=dict(new_matched_backend=True,original_D1_parity_claimed=False,GT_inputs=False,
            archived_partial_2D_seed_terms=0,archived_partial_2D_guard_terms=0,partial_tangent_constraints=0),
        prior_audit=dict(prior_weight=cfg['prior_weight'],bias_strength=cfg['bias_strength'],new_bias_fit=False))


def _endpoint(answer,start,branch,stage,pair,constraint,xyz,K,scale,cfg):
    R,t=V._pose(answer.x,scale);q,depth=V.project_native(xyz,R,t,K);slack=constraint(answer.x)[0]
    value,_,parts=pair(answer.x)
    finite=bool(np.isfinite(q).all()and np.isfinite(slack).all()and np.isfinite(answer.x).all()and np.isfinite(value))
    feasible=bool(finite and V._valid_rotation(R)and np.min(depth)>=cfg['positive_depth_m']-1e-10 and np.min(slack)>=-cfg['feasibility_tolerance'])
    return dict(R=R,t=t,q=q,p=answer.x.copy(),name=start['name'],objective=float(value)if np.isfinite(value)else None,
        feasible=feasible,record=dict(R=R.tolist(),t=t.tolist(),p=answer.x.tolist(),initialization=start['name'],branch=branch,
            phase=stage,optimizer_success=bool(answer.success),optimizer_status=int(answer.status),optimizer_message=str(answer.message),
            iterations=int(answer.nit),joint_feasible=feasible,objective=float(value)if np.isfinite(value)else None,
            objective_components=parts,minimum_constraint_slack=float(np.min(slack))if finite else None,
            actual_pose_saved_even_if_infeasible=True))


def _finish(result,solutions,xyz,K,Q,measurements,boundary,domains,regions,scale,cfg):
    if not solutions:
        result.update(status='no_joint_feasible_pose_found',fallback_reason='all_searched_joint_branches_infeasible_or_numerical')
        return result
    solutions.sort(key=lambda s:(s['objective'],s['name']));chosen=solutions[0]
    R,t,p=chosen['R'],chosen['t'],chosen['p'];complete=_complete(measurements,boundary)
    point=J.visible_residual_and_jacobian(p,xyz,K,complete,scale)[1]
    if complete:point*=np.repeat(np.sqrt([m['fit_reliability_weight']for m in complete]),2)[:,None]
    normal=N._normal_residual_and_jacobian(p,xyz,K,boundary,scale)[1]
    if boundary:normal*=np.repeat(np.sqrt([b['fit_reliability_weight']for b in boundary]),[len(b['lines'])for b in boundary])[:,None]
    rank=V._rank(np.vstack([point,normal]));ambiguous=rank['rank']<6
    result.update(accepted=True,status='joint_feasible_ambiguous'if ambiguous else'joint_feasible',
        R=R.tolist(),t=t.tolist(),R_physical=(R@Q).tolist(),projected_native=chosen['q'].tolist(),
        objective=chosen['objective'],objective_components=chosen['record']['objective_components'],
        rank=dict(visible=V._rank(point),BI_and_partial_boundary=rank),ambiguity=dict(detected=ambiguous,global_uniqueness_proved=False),
        minimum_joint_slack=chosen['record']['minimum_constraint_slack'],OT_constraints_satisfied=True,U_constraints_satisfied=True,
        OT_solved=bool(domains)and not ambiguous,OT_unique_solution_claimed=False,
        feasible_pose_samples=[dict(R=s['R'].tolist(),t=s['t'].tolist(),objective=s['objective'],initialization=s['name'])for s in solutions])
    result['pose_family_candidates']=deepcopy(result['feasible_pose_samples'])if ambiguous else[]
    return result


def solve_boundary_u_region_joint_ot_pnp(xyz,K,states,visible,domains,image_hw,coarse_pose,initializations=None,config=None,*,boundary_observations=None,u_regions=None,boundary_factor_weight=1.):
    if boundary_factor_weight!=1.:raise ValueError('Fixed normal-factor weight1 required')
    xyz,K,states,measurements,domains,hw,cfg,boundary,regions,Q,prior,_=_prepare(xyz,K,states,visible,domains,image_hw,coarse_pose,config,boundary_observations,u_regions)
    result=_empty(coarse_pose,states,cfg);complete=_complete(measurements,boundary);scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    if not complete and not boundary and not domains and not regions:
        result.update(status='empty_observation',fallback_reason='no_image_information');return result
    starts,search=_initial_poses(xyz,K,complete,coarse_pose,initializations,cfg);result['search']=search
    result['ordered_actual_initializations']=deepcopy(starts)
    result['search'].update(full_point_seed_ids=[m['native_corner']for m in complete],partial_archive_ids_excluded_from_seed=[b['native_corner']for b in boundary],
        numeric_caps={k:cfg[k]for k in ('max_p3p_triples','global_rotation_starts','max_seed_nfev','max_joint_iterations','maximum_union_branches')})
    seed_pair=V._cached_value_jacobian(lambda p:BASE._objective_pair(p,xyz,K,complete,scale,prior,cfg))
    joint_pair=V._cached_value_jacobian(lambda p:_joint_objective_pair(p,xyz,K,measurements,scale,prior,cfg,boundary,1.))
    def fit(start,sides,branch,stage):
        seed=stage=='FULL_POINT_PRIOR_SEED';pair=seed_pair if seed else joint_pair
        constraint=V._cached_value_jacobian((lambda p:BASE._constraints(p,xyz,K,complete,[],(),hw,scale,cfg))if seed else
            (lambda p:_joint_constraint_pair(p,xyz,K,measurements,domains,sides,hw,scale,cfg,boundary,regions)))
        result['optimizer_calls']+=1
        answer=minimize(lambda p:pair(p)[0],V._pose_vector(np.asarray(start['R']),np.asarray(start['t']),scale),method='SLSQP',
            jac=(lambda p:pair(p)[1].copy())if cfg['analytic_jacobian']else None,
            constraints=[dict(type='ineq',fun=lambda p:constraint(p)[0].copy(),**({'jac':lambda p:constraint(p)[1].copy()}if cfg['analytic_jacobian']else{}))],
            options=dict(maxiter=cfg['max_seed_nfev']if seed else cfg['max_joint_iterations'],ftol=cfg['optimizer_ftol'],disp=False))
        endpoint=_endpoint(answer,start,branch,stage,pair,constraint,xyz,K,scale,cfg)
        result['all_visited_endpoints'].append(endpoint['record']);return endpoint
    seeds=[]
    for start in starts:
        try:
            solved=fit(start,(),None,'FULL_POINT_PRIOR_SEED')
            if solved['feasible']and not any(V._same_pose((solved['R'],solved['t']),(s['R'],s['t']),scale)for s in seeds):seeds.append(solved)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError)as exc:
            result['candidates'].append(dict(phase='FULL_POINT_PRIOR_SEED',initialization=start['name'],error=repr(exc),actual_pose_unavailable=True))
    seeds.sort(key=lambda s:(s['objective'],s['name']));joint_starts=[dict(name='BI_PRIOR_SEED_'+s['name'],R=s['R'],t=s['t'])for s in seeds]
    for start in starts:
        if not any(V._same_pose((np.asarray(start['R']),np.asarray(start['t'])),(np.asarray(s['R']),np.asarray(s['t'])),scale)for s in joint_starts):joint_starts.append(start)
    result['search'].update(BI_prior_local_minima=len(seeds),joint_initializations_per_branch=len(joint_starts),T_union_branches_searched=len(_branches(domains,cfg)))
    result['ordered_actual_joint_initializations']=deepcopy(joint_starts)
    solutions=[]
    for branch,sides in enumerate(_branches(domains,cfg)):
        for start in joint_starts:
            try:
                solved=fit(start,sides,branch,'JOINT_FULL_AND_NORMAL_OT_U');result['candidates'].append(solved['record'])
                if solved['feasible']and not any(V._same_pose((solved['R'],solved['t']),(s['R'],s['t']),scale)for s in solutions):solutions.append(solved)
            except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError)as exc:
                result['candidates'].append(dict(phase='JOINT_FULL_AND_NORMAL_OT_U',initialization=start['name'],branch=branch,error=repr(exc),actual_pose_unavailable=True))
    return _finish(result,solutions,xyz,K,Q,measurements,boundary,domains,regions,scale,cfg)


def refine_global_outline_joint_pose(xyz,K,states,visible,domains,image_hw,coarse_pose,*,outline_observation,initial_solutions,initializations=None,u_regions=None,boundary_observations=None,config=None,outline_configuration=None):
    cfg=_configuration(config);cfg['prior_weight']=0.
    xyz,K,states,measurements,domains,hw,cfg,boundary,regions,Q,prior,_=_prepare(xyz,K,states,visible,domains,image_hw,coarse_pose,cfg,boundary_observations,u_regions)
    ocfg=_outline_config(outline_configuration or outline_observation.get('configuration'));scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    if not initial_solutions:raise ValueError('P1/WD0 initial solution list required')
    if not outline_observation.get('supported')or ocfg['outline_factor_weight']==0:
        result=deepcopy(initial_solutions[0]);result['outline_stage_exact_reused_without_outline']=True;result['optimizer_calls']=0;result['all_visited_endpoints']=[];return result
    starts=[]
    for j,solution in enumerate(initial_solutions):
        if solution.get('accepted'):starts.append(dict(R=solution['R'],t=solution['t'],name=f'SAVED_{j}_WINNER'))
        for i,sample in enumerate(solution.get('feasible_pose_samples',[])):starts.append(dict(R=sample['R'],t=sample['t'],name=f'SAVED_{j}_SAMPLE_{i}'))
    starts.extend(deepcopy(initializations or[]));unique=[]
    for s in starts:
        R,t=np.asarray(s['R']),np.asarray(s['t'])
        if not any(V._same_pose((R,t),(u['R'],u['t']),scale)for u in unique):unique.append(dict(R=R,t=t,name=s.get('name','SUPPLIED')))
    result=_empty(coarse_pose,states,cfg)
    result['ordered_actual_initializations']=deepcopy(unique)
    result['full_saved_initial_solutions']=deepcopy(initial_solutions)
    result['additional_actual_initializations']=deepcopy(initializations or[])
    if not unique:
        result.update(status='no_admissible_P1_WD0_start',fallback_reason='no_saved_fullsearch_feasible_start');return result
    def objective(p):
        v,g,parts=_joint_objective_pair(p,xyz,K,measurements,scale,prior,cfg,boundary,1.)
        extra,eg,audit=outline_value_and_gradient(p,xyz,K,scale,outline_observation,ocfg)
        return v+ocfg['outline_factor_weight']*extra,g+ocfg['outline_factor_weight']*eg,{**parts,
            'global_outline_robust_mean_loss':extra,'global_outline_factor_weight':ocfg['outline_factor_weight'],'global_outline_audit':audit}
    pair=V._cached_value_jacobian(objective);solutions=[]
    from types import SimpleNamespace
    for branch,sides in enumerate(_branches(domains,cfg)):
        constraint=V._cached_value_jacobian(lambda p:_joint_constraint_pair(p,xyz,K,measurements,domains,sides,hw,scale,cfg,boundary,regions))
        for start in unique:
            p0=V._pose_vector(start['R'],start['t'],scale)
            for kind in ('UNREFINED','REFINED'):
                try:
                    if kind=='UNREFINED':answer=SimpleNamespace(x=p0,success=True,status=0,message='saved unrefined contender',nit=0)
                    else:
                        result['optimizer_calls']+=1
                        answer=minimize(lambda p:pair(p)[0],p0,jac=lambda p:pair(p)[1].copy(),method='SLSQP',
                            constraints=[dict(type='ineq',fun=lambda p:constraint(p)[0].copy(),jac=lambda p:constraint(p)[1].copy())],
                            options=dict(maxiter=cfg['max_joint_iterations'],ftol=cfg['optimizer_ftol'],disp=False))
                    solved=_endpoint(answer,start,branch,'OUTLINE_'+kind,pair,constraint,xyz,K,scale,cfg)
                    result['all_visited_endpoints'].append(solved['record']);result['candidates'].append(solved['record'])
                    if (solved['feasible']and solved['record']['objective_components']['global_outline_audit'].get('valid_projection')
                            and not any(V._same_pose((solved['R'],solved['t']),(s['R'],s['t']),scale)for s in solutions)):solutions.append(solved)
                except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError)as exc:
                    result['candidates'].append(dict(initialization=start['name'],branch=branch,phase='OUTLINE_'+kind,error=repr(exc),actual_pose_unavailable=True))
    result['search']=dict(saved_global_search_reused_as_initializations=True,new_P3P_or_global_starts=0,
        joint_initializations_per_branch=len(unique),T_union_branches_searched=len(_branches(domains,cfg)),
        outline_gradient='original analytic v4.11 primitive',G10_G13_shared_typed_normal_objective=True)
    return _finish(result,solutions,xyz,K,Q,measurements,boundary,domains,regions,scale,cfg)


def prepare_context(packet,family):
    spec=packet['solver_input'];fam=packet['original_e_stage1_family_seed_bank'][family]['family'];cfg=deepcopy(spec['config']);cfg['prior_weight']=0.
    values=_prepare(fam['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],fam['coarse_pose'],cfg,spec['boundary_observations'],spec['u_regions'])
    xyz,K,states,measurements,domains,hw,cfg,boundary,regions,Q,prior,requested=values
    outline=spec['outline_observation'];ocfg=_outline_config(outline.get('configuration'));use=bool(outline.get('supported')and ocfg['outline_factor_weight']>0)
    for m in measurements:
        source=next(r for r in spec['visible']if r['native_corner']==m['native_corner'])
        m.update(source=source.get('source','UNKNOWN_IMAGE_SOURCE'),state=states[m['native_corner']],image_measurement=True,measurement_is_pose_projection=False)
    return Context(packet,'G13',family,P,xyz,K,states,measurements,domains,tuple(hw),cfg,boundary,regions,Q,prior,requested,
        max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3),outline,ocfg,use,'BI_FULL_NORMAL_ONLY_FINITE_SEARCH_V1',True,False)


def compile_models(packet,config):
    models={}
    for family in packet['original_e_stage1_family_seed_bank']:
        ctx=prepare_context(packet,family);owner=OWNER.compile_owner_groups(ctx,none_prior=config['none_prior']);constraints=[]
        for perm in owner['permutations']:
            ms=FEATURES._rebuild_geometry(OWNER.map_owner_records(ctx.measurements,perm,ctx.xyz),ctx)
            bs=FEATURES._rebuild_geometry(OWNER.map_owner_records(ctx.boundary,perm,ctx.xyz),ctx)
            ds=OWNER.map_owner_records(ctx.domains,perm,ctx.xyz);us=OWNER.map_owner_records(ctx.regions,perm,ctx.xyz)
            constraints.append((ms,bs,ds,us))
        models[family]=FEATURES.FamilyModel(ctx,owner,None,constraints,deepcopy(constraints))
    return models
