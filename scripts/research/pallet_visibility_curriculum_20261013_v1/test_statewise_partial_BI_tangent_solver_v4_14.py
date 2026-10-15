"""Focused uncertain-tangent algebra/delegation tests, no dataset or fitting."""
from copy import deepcopy

import numpy as np
import pytest

from . import statewise_partial_BI_tangent_solver_v4_14 as NEW
from . import statewise_partial_BI_tangent_outline_solver_v4_14 as OUT
from .test_statewise_partial_BI_u_solver_v4_13 import typed_fixture
from .test_statewise_partial_boundary_u_solver_v4_6 import region,parameters
from .test_statewise_partial_BI_outline_solver_v4_13 import sample


def test_tangent_gradient_FD_and_directional_variance_inflation():
    a,b=typed_fixture();scale,p=parameters(a);boundary=NEW._boundary_inputs([b],a[2],a[5],a[0])
    at=p+np.array([.01,-.02,.005,.008,-.011,.025])
    value,gradient=NEW._tangent_objective_pair(at,a[0],a[1],boundary,scale,1.,1.25,4.)
    numerical=NEW.V._jacobian(lambda pp:np.array([NEW._tangent_objective_pair(pp,a[0],a[1],boundary,scale,1.,1.25,4.)[0]]),at)[0]
    assert value>0 and np.allclose(gradient,numerical,atol=2e-5,rtol=2e-6)
    r1,j1=NEW._tangent_residual_and_jacobian(at,a[0],a[1],boundary,scale,1.)
    r4,j4=NEW._tangent_residual_and_jacobian(at,a[0],a[1],boundary,scale,4.)
    assert np.array_equal(r4,r1/2) and np.array_equal(j4,j1/2)
    assert NEW._tangent_records(boundary,4.)[0]['reliability']==b['fit_reliability_weight']


def test_uncertain_tangent_has_no_archived_normal_point_cost():
    a,b=typed_fixture();scale,p=parameters(a);boundary=NEW._boundary_inputs([b],a[2],a[5],a[0])
    first=NEW._tangent_objective_pair(p,a[0],a[1],boundary,scale,1.,1.25,4.)
    changed=deepcopy(boundary);changed[0]['association_provenance']['original_observed_xy'][1]+=1000.
    second=NEW._tangent_objective_pair(p,a[0],a[1],changed,scale,1.,1.25,4.)
    assert first[0]==second[0] and np.array_equal(first[1],second[1])
    changed[0]['association_provenance']['original_observed_xy'][0]+=10.
    assert NEW._tangent_objective_pair(p,a[0],a[1],changed,scale,1.,1.25,4.)[0]>first[0]


@pytest.mark.parametrize('state',['S','U','O','T'])
def test_state_gate_before_poison_archived_coordinate_or_line_fields(state):
    with pytest.raises(ValueError,match='B/I before'):
        NEW._tangent_records([{'state':state,'lines':object(),'association_provenance':object()}],4.)


def test_multi_line_tangent_basis_requires_explicit_index():
    a,b=typed_fixture();b['lines'].append({**b['lines'][0],'normal_unit_xy':[1.,0.]})
    boundary=NEW._boundary_inputs([b],a[2],a[5])
    with pytest.raises(ValueError,match='explicit tangent_prediction_line_index'):NEW._tangent_records(boundary,4.)
    b['tangent_prediction_line_index']=1;boundary=NEW._boundary_inputs([b],a[2],a[5])
    assert np.array_equal(NEW._tangent_records(boundary,4.)[0]['tangent'],[0.,1.])


def test_off_exact_parent_and_private_globals_unchanged():
    a,b=typed_fixture();rr=[region(a)]
    parent=NEW.OLD.solve_partial_BI_u_region_joint_ot_pnp(*a[:7],config=a[7],boundary_observations=[b],u_regions=rr)
    result=NEW.solve_partial_BI_tangent_u_region_joint_ot_pnp(*a[:7],config=a[7],boundary_observations=[b],u_regions=rr,prediction_tangent_weight=0.)
    assert result==parent
    old_objective=NEW.OLD._joint_objective_pair;old_parser=NEW.OLD._boundary_inputs
    saved=deepcopy(a[3]);bb=deepcopy(b)
    result=NEW.solve_partial_BI_tangent_u_region_joint_ot_pnp(*a[:7],config=a[7],boundary_observations=[b],u_regions=rr)
    assert result['accepted'] and result['OT_constraints_satisfied'] and result['U_constraints_satisfied']
    assert result['reported_native'][b['native_corner']] is None
    assert result['objective_components']['tangent_prediction_is_precise_RGB_coordinate'] is False
    assert NEW.OLD._joint_objective_pair is old_objective and NEW.OLD._boundary_inputs is old_parser
    assert a[3]==saved and b==bb
    assert result['BI_prior_seed']==parent['BI_prior_seed']


def test_outline_off_exact_and_active_uncertain_readout():
    a,b,rr,old,outline=sample();kwargs=dict(config=a[7],boundary_observations=[b],u_regions=rr,
        outline_observation=outline,initial_solutions=[old])
    parent=OUT.OLD.refine_global_outline_joint_pose(*a[:7],**kwargs)
    assert OUT.refine_global_outline_joint_pose(*a[:7],**kwargs,prediction_tangent_weight=0.)==parent
    old_p=OUT.OLD.P;old_prepare=OUT.OLD._prepare
    result=OUT.refine_global_outline_joint_pose(*a[:7],**kwargs)
    assert result['accepted'] and result['partial_BI_constraints_satisfied']
    assert result['reported_native'][b['native_corner']] is None
    assert result['tangent_prediction_factor_audit']['variance_inflation']==4.
    assert result['tangent_prediction_factor_audit']['weight']==1.
    assert OUT.OLD.P is old_p and OUT.OLD._prepare is old_prepare
