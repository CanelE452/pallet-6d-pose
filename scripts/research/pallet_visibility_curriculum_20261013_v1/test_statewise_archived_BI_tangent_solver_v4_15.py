"""Focused legacy archive/directional factor contracts, no dataset fits."""
from copy import deepcopy
import numpy as np
import pytest
from . import statewise_archived_BI_tangent_solver_v4_15 as N
from . import statewise_archived_BI_tangent_outline_solver_v4_15 as OUT
from .test_statewise_partial_boundary_u_solver_v4_6 import arguments,observation,region,parameters
from .test_statewise_partial_BI_u_solver_v4_13 import typed_fixture
from .test_statewise_partial_BI_outline_solver_v4_13 import sample


def fixture():
    a=arguments();original=observation(a)
    records=N.archive_legacy_B_tangent_predictions([original],a[3],a[2],a[7])
    return a,original,records


def test_legacy_archive_exact_binding_no_fake_typed_revocation_or_input_changes():
    a,b,records=fixture();normalized=N._boundary_inputs(records,a[2],a[5],a[0])
    N.validate_archived_predictions(normalized,a[3],a[7])
    assert records[0]['replaces_raw_B_likelihood'] is True
    assert 'explicit_complete_BI_likelihood_revocation' not in records[0]
    assert normalized[0]['legacy_fallback_B_record'] and not normalized[0]['explicit_complete_BI_likelihood_revocation']
    assert {k:v for k,v in records[0].items() if k!=N.ARCHIVE_KEY}==b
    assert N.ARCHIVE_KEY not in b
    t=N._tangent_records(normalized,4.)[0]
    assert t['variance_px2']==4*t['original_variance_px2'] and t['reliability']==a[3][0]['fit_reliability_weight']


@pytest.mark.parametrize('field',['original_observed_xy','original_observed_covariance_px2','original_effective_fit_covariance_px2'])
def test_legacy_archive_q_or_covariance_mismatch_rejected(field):
    a,b,records=fixture();value=np.asarray(records[0][N.ARCHIVE_KEY][field]);value.flat[0]+=.01
    records[0][N.ARCHIVE_KEY][field]=value.tolist();normalized=N._boundary_inputs(records,a[2],a[5])
    with pytest.raises(ValueError,match='differs from immutable'):N.validate_archived_predictions(normalized,a[3],a[7])


@pytest.mark.parametrize('state',['S','U','O','T'])
def test_initial_state_gate_before_poison_archive_coordinates(state):
    a=arguments();k=a[3][0]['native_corner'];a[2][k]=state
    with pytest.raises(ValueError,match='gate rejects'):
        N.archive_legacy_B_tangent_predictions([{'native_corner':k,'lines':object(),N.ARCHIVE_KEY:object()}],a[3],a[2],a[7])


def test_legacy_directional_gradient_FD_and_normal_coordinate_not_duplicated():
    a,b,records=fixture();boundary=N._boundary_inputs(records,a[2],a[5],a[0]);scale,p=parameters(a)
    at=p+np.array([.01,-.02,.005,.008,-.011,.025])
    value,gradient=N._tangent_objective_pair(at,a[0],a[1],boundary,scale,1.,1.25,4.)
    numeric=N.P.V3._jacobian(lambda x:np.array([N._tangent_objective_pair(x,a[0],a[1],boundary,scale,1.,1.25,4.)[0]]),at)[0]
    assert value>0 and np.allclose(gradient,numeric,atol=2e-5,rtol=2e-6)
    changed=deepcopy(boundary);changed[0][N.ARCHIVE_KEY]['original_observed_xy'][1]+=1000.
    second=N._tangent_objective_pair(at,a[0],a[1],changed,scale,1.,1.25,4.)
    assert second[0]==value and np.array_equal(second[1],gradient)


def test_zero_exact_v13_active_guards_and_private_globals_immutable():
    a,b,records=fixture();rr=[region(a)]
    kw=dict(config=a[7],boundary_observations=records,u_regions=rr)
    parent=N.P.solve_partial_BI_u_region_joint_ot_pnp(*a[:7],**kw)
    assert N.solve_archived_BI_tangent_u_region_joint_ot_pnp(*a[:7],**kw,prediction_tangent_weight=0.)==parent
    old=N.P._joint_objective_pair;old_parser=N.P._boundary_inputs;before=deepcopy(records)
    result=N.solve_archived_BI_tangent_u_region_joint_ot_pnp(*a[:7],**kw)
    assert result['accepted'] and result['OT_constraints_satisfied'] and result['U_constraints_satisfied']
    assert result['reported_native'][b['native_corner']] is None
    assert result['tangent_prediction_factor_audit']['legacy_fallback_B_archive_ids']==[b['native_corner']]
    assert 'uncertain_archived_BI_tangent_robust_loss' in result['objective_components']
    assert result['BI_prior_seed']==parent['BI_prior_seed'] and records==before
    assert N.P._joint_objective_pair is old and N.P._boundary_inputs is old_parser


def test_legacy_only_outline_actually_uses_tangent_without_fake_typed_flags():
    a,b,rr,old,outline=sample();a=arguments();legacy=observation(a)
    records=N.archive_legacy_B_tangent_predictions([legacy],a[3],a[2],a[7])
    kw=dict(config=a[7],boundary_observations=records,u_regions=rr,outline_observation=outline,initial_solutions=[old])
    parent=OUT.OLD.refine_global_outline_joint_pose(*a[:7],**kw)
    assert OUT.refine_global_outline_joint_pose(*a[:7],**kw,prediction_tangent_weight=0.)==parent
    result=OUT.refine_global_outline_joint_pose(*a[:7],**kw)
    assert result['accepted'] and 'uncertain_archived_BI_tangent_robust_loss' in result['objective_components']
    assert result['tangent_prediction_factor_audit']['legacy_fallback_B_archive_ids']==[legacy['native_corner']]
    assert result['reported_native'][legacy['native_corner']] is None


def test_typed_only_algebra_exact_v14_and_fallback_source_gate():
    a,b=typed_fixture();boundary=N._boundary_inputs([b],a[2],a[5],a[0]);scale,p=parameters(a)
    x=N._tangent_objective_pair(p,a[0],a[1],boundary,scale,1.,1.25,4.)
    y=N.T._tangent_objective_pair(p,a[0],a[1],boundary,scale,1.,1.25,4.)
    assert x[0]==y[0] and np.array_equal(x[1],y[1])
    a=arguments();b=observation(a);a[3][0]['source']='B_ACTUAL_BOUNDARY_RECOVERY'
    with pytest.raises(ValueError,match='fallbackB source'):N.archive_legacy_B_tangent_predictions([b],a[3],a[2],a[7])


def test_legacy_archive_validator_handles_one_pass_visible_iterable():
    a,b,records=fixture();normalized=N._boundary_inputs(records,a[2],a[5],a[0])
    N.validate_archived_predictions(normalized,(r for r in a[3]),a[7])
