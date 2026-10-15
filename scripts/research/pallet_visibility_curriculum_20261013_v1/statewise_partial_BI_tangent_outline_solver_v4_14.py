"""Same typed BI/outline SE(3), with optional uncertain prediction tangent.

Old RGB-normal covariance/guards, whole-outline factor, seeds and every O/T/U
remain unchanged. Tangent weight0 delegates immutable v13 exactly. Positive
weight reuses its body with a private pure helper facade; parent modules are
never mutated. Varianceinflation4/original1 are declared assumptions, not fits.
"""
from types import SimpleNamespace

from . import statewise_partial_BI_outline_solver_v4_13 as OLD
from . import statewise_partial_BI_tangent_solver_v4_14 as T

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
    if weight==0 or not boundary_observations:
        return OLD.refine_global_outline_joint_pose(*args,**kw)
    boundary=T._boundary_inputs(boundary_observations,states,image_hw)
    if not T._tangent_records(boundary,inflation):return OLD.refine_global_outline_joint_pose(*args,**kw)
    def objective(*a):return T.objective_with_tangent(*a,prediction_tangent_weight=weight,tangent_variance_inflation=inflation)
    def solve(*a,**k):return T.solve_partial_BI_tangent_u_region_joint_ot_pnp(*a,**k,
        prediction_tangent_weight=weight,tangent_variance_inflation=inflation)
    facade=SimpleNamespace(**{**vars(OLD.P),'_joint_objective_pair':objective,
        '_boundary_inputs':T._boundary_inputs,'solve_partial_BI_u_region_joint_ot_pnp':solve})
    prepare=T._clone(OLD._prepare,{'P':facade})
    operation=T._clone(OLD.refine_global_outline_joint_pose,{'P':facade,'_prepare':prepare})
    result=operation(*args,**kw)
    return T.augment_tangent_audit(result,xyz,K,boundary,weight,inflation,
        'statewise_partial_BI_tangent_outline_solver_v4_14')
