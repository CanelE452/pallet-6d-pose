from copy import deepcopy
import numpy as np
from . import statewise_top_normal_outline_solver_v4_16 as N
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from .test_statewise_partial_BI_outline_solver_v4_13 import sample
from .test_statewise_partial_boundary_u_solver_v4_6 import arguments,observation


def test_zero_normal_exact_t15_and_active_joint_typed_guards_preserved():
    a,b,regions,old,outline=sample()
    records=T.archive_legacy_B_tangent_predictions([b],a[3],a[2],a[7])
    kw=dict(config=a[7],boundary_observations=records,u_regions=regions,outline_observation=outline,initial_solutions=[old])
    parent=N.T_OUTLINE.refine_global_outline_joint_pose(*a[:7],**kw)
    result=N.refine_global_outline_joint_pose(*a[:7],**kw,normal_parameters=dict(sigma_angle_rad=.1,weight=0))
    assert result==parent
    assert N.refine_global_outline_joint_pose(*a[:7],**kw)==parent
    original=N.G13.P._joint_objective_pair;before=deepcopy(records)
    active=N.refine_global_outline_joint_pose(*a[:7],**kw,normal_parameters=dict(sigma_angle_rad=.1))
    assert active['accepted'] and active['OT_constraints_satisfied'] and active['U_constraints_satisfied']
    assert active['reported_native'][b['native_corner']] is None and records==before
    assert active['objective_components']['top_plane_normal_prior']['valid_normal_math']
    assert active['top_plane_normal_prior_audit']['prior_not_counted_as_image_information']
    assert active['schema']=='statewise_top_normal_outline_solver_v4_16'
    assert N.G13.P._joint_objective_pair is original


def test_legacy_and_no_boundary_normal_is_still_active():
    a,b,regions,old,outline=sample();a=arguments();legacy=observation(a)
    records=T.archive_legacy_B_tangent_predictions([legacy],a[3],a[2],a[7])
    for rr in (records,[]):
        result=N.refine_global_outline_joint_pose(*a[:7],config=a[7],boundary_observations=rr,u_regions=regions,
            outline_observation=outline,initial_solutions=[old],normal_parameters=dict(sigma_angle_rad=.1))
        assert result['accepted'] and result['objective_components']['top_plane_normal_prior']['valid_normal_math']
        assert result['top_plane_normal_prior_audit']['native_yaw_position_depth_direct_cost'] is False


def test_exact_antipode_candidate_is_flagged_not_zero_cost_or_pool_abort():
    value,gradient,audit=N._normal_safe(np.r_[[np.pi,0,0],[0,0,1]],np.eye(3),dict(sigma_angle_rad=.1,weight=1,huber_scale=3))
    assert value==1e12 and np.array_equal(gradient,np.zeros(6)) and not audit['valid_normal_math']


def test_unsupported_outline_gets_fresh_normal_data_refine_not_stale_objective():
    a,b,regions,old,outline=sample();unsupported={k:v for k,v in outline.items()if k not in ('observed_boundary_samples_xy','observed_boundary_segments_xy')}
    unsupported.update(supported=False,reason='synthetic no noncanvas boundary')
    records=T.archive_legacy_B_tangent_predictions([b],a[3],a[2],a[7])
    result=N.refine_global_outline_joint_pose(*a[:7],config=a[7],boundary_observations=records,u_regions=regions,
        outline_observation=unsupported,initial_solutions=[old],normal_parameters=dict(sigma_angle_rad=.1))
    assert result['accepted'] and not result['global_outline_factor_audit']['outline_factor_used']
    assert 'top_plane_normal_prior_robust_loss' in result['objective_components']


def test_rejected_no_starts_and_insufficient_data_have_fresh_current_audits():
    a,b,regions,old,outline=sample()
    old=deepcopy(old);old.update(accepted=False,feasible_pose_samples=[],pose_family_candidates=[],
        objective=123.,objective_components={'stale_seed_objective':True})
    records=T.archive_legacy_B_tangent_predictions([b],a[3],a[2],a[7])
    for insufficient in (False,True):
        args=list(deepcopy(a))
        if insufficient:
            keep=b['native_corner'];args[3]=[m for m in args[3]if m['native_corner']==keep]
            args[2]=['U'if k!=keep and s in ('B','I')else s for k,s in enumerate(args[2])]
        result=N.refine_global_outline_joint_pose(*args[:7],config=args[7],boundary_observations=records,
            u_regions=regions,outline_observation=outline,initial_solutions=[old],normal_parameters=dict(sigma_angle_rad=.1))
        assert not result['accepted'] and result['schema']=='statewise_top_normal_outline_solver_v4_16'
        assert result['objective'] is None and result['objective_components']=={}
        assert 'tangent_prediction_factor_audit' in result and 'top_plane_normal_prior_audit' in result
        assert not result['partial_BI_constraints_satisfied'] and not result['U_constraints_satisfied']
        assert result['reported_native'][b['native_corner']] is None
