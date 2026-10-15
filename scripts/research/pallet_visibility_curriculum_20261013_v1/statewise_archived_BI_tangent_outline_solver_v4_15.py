"""Same frozen-family outline/normal joint solve with archived predictionT.

Dedicated legacyB archives and original typedBI archives authenticate old
q/noise/weights. No old normal cost is restored or duplicated. Empty/weight0
delegates immutable v13; prior stays0, all OT/U/physical guards stay intact.
"""
from types import SimpleNamespace
from copy import deepcopy
from . import statewise_partial_BI_outline_solver_v4_13 as OLD
from . import statewise_archived_BI_tangent_solver_v4_15 as T

DEFAULT_OUTLINE_CONFIGURATION=OLD.DEFAULT_OUTLINE_CONFIGURATION
observe_global_outline=OLD.observe_global_outline


def refine_global_outline_joint_pose(xyz,K,states,visible,domains,image_hw,coarse_pose,*,
        outline_observation,initial_solutions,initializations=None,u_regions=None,boundary_observations=None,
        config=None,outline_configuration=None,prediction_tangent_weight=1.,tangent_variance_inflation=4.):
    weight,inflation=T._parameters(prediction_tangent_weight,tangent_variance_inflation)
    args=(xyz,K,states,visible,domains,image_hw,coarse_pose)
    kw=dict(outline_observation=outline_observation,initial_solutions=initial_solutions,
        initializations=initializations,u_regions=u_regions,boundary_observations=boundary_observations,
        config=config,outline_configuration=outline_configuration)
    if weight==0 or not boundary_observations:return OLD.refine_global_outline_joint_pose(*args,**kw)
    boundary=T._boundary_inputs(boundary_observations,states,image_hw)
    if not T._tangent_records(boundary,inflation):return OLD.refine_global_outline_joint_pose(*args,**kw)
    source_records=OLD.V._records(visible);domain_records=OLD.V._records(domains)
    supplied_initializations=list(initializations or [])
    args=(xyz,K,states,source_records,domain_records,image_hw,coarse_pose)
    kw['initializations']=supplied_initializations
    def objective(*a):return T.objective_with_tangent(*a,prediction_tangent_weight=weight,tangent_variance_inflation=inflation)
    def solve(*a,**k):return T.solve_archived_BI_tangent_u_region_joint_ot_pnp(*a,**k,
        prediction_tangent_weight=weight,tangent_variance_inflation=inflation)
    facade=SimpleNamespace(**{**vars(OLD.P),'_joint_objective_pair':objective,
        '_boundary_inputs':T._boundary_inputs,'solve_partial_BI_u_region_joint_ot_pnp':solve,
        'validate_complete_BI_likelihood_revocations':T.validate_archived_predictions})
    prepare=T.T._clone(OLD._prepare,{'P':facade})
    # v13 delegates legacy-only records to v11. Reuse that actual body with
    # private objective/prepare helpers instead of pretending legacyB is typed.
    typed=any(b['explicit_complete_BI_likelihood_revocation'] for b in boundary)
    if not typed:
        T.validate_archived_predictions(boundary,source_records,config)
        count=len(source_records)
        ocfg=OLD._outline_config(outline_configuration)
        if not outline_observation.get('supported') or ocfg['outline_factor_weight']==0 or count<2:
            starts=deepcopy(supplied_initializations)
            for j,solution in enumerate(initial_solutions):
                if solution.get('accepted'):starts.append(dict(R=solution['R'],t=solution['t'],name=f'SAVED_{j}_WINNER'))
                for i,sample in enumerate(solution.get('feasible_pose_samples',[])):
                    starts.append(dict(R=sample['R'],t=sample['t'],name=f'SAVED_{j}_SAMPLE_{i}'))
            cfg=deepcopy(config or {});cfg['prior_weight']=0.
            return solve(*args,config=cfg,initializations=starts,
                         boundary_observations=boundary_observations,u_regions=u_regions)
        operation=T.T._clone(OLD.OLD.refine_global_outline_joint_pose,
            {'P':facade,'_prepare':prepare,'_finalize_outline':OLD._finalize_outline})
    else:
        operation=T.T._clone(OLD.refine_global_outline_joint_pose,{'P':facade,'_prepare':prepare})
    return T.augment(operation(*args,**kw),xyz,K,boundary,weight,inflation,
        'statewise_archived_BI_tangent_outline_solver_v4_15')
