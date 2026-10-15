"""Focused typed partial B/I contracts; synthetic geometry, no images or GT."""
from copy import deepcopy

import numpy as np
import pytest

from . import joint_ot_pnp as V
from . import statewise_partial_BI_u_solver_v4_13 as NEW
from . import statewise_partial_boundary_u_solver_v4_6 as OLD
from . import statewise_uncertain_solver_v4_bias as BASE
from . import statewise_u_region_solver_v4_5 as U
from .test_statewise_partial_boundary_u_solver_v4_6 import arguments,observation,region,parameters


def typed_fixture(state='I'):
    a=arguments();m=a[3][0];k=m['native_corner'];a[2][k]=state
    m.update(state=state,source='I_ALL_VISIBLE_GLOBAL_RGB_EDGE_INTERSECTION' if state=='I' else 'B_ACTUAL_BOUNDARY_RECOVERY')
    b=observation(a);b.pop('replaces_raw_B_likelihood');b.update(state=state,
        replaces_complete_BI_likelihood=True,explicit_complete_BI_likelihood_revocation=True)
    b['association_provenance'].update(original_measurement_source=m['source'],
        original_observed_xy=deepcopy(m['xy']),original_observed_covariance_px2=deepcopy(m['covariance_px2']),
        original_effective_fit_covariance_px2=deepcopy(m['fit_covariance_px2']),
        original_fit_reliability_weight=m['fit_reliability_weight'],
        revocation_reason='no_unique_face_supported_junction',
        display_coordinate_is_full_2D_measurement=False,
        normal_projected_display_xy=deepcopy(m['xy']))
    return a,b


def solve(a,records=None,weight=1.,**kw):
    return NEW.solve_partial_BI_u_region_joint_ot_pnp(*a[:7],config=a[7],
        boundary_observations=records,boundary_factor_weight=weight,**kw)


@pytest.mark.parametrize('state',['B','I'])
def test_corrected_typed_source_archives_and_state_preserved(state):
    a,b=typed_fixture(state);saved=deepcopy(b)
    normalized=NEW._boundary_inputs([b],a[2],a[5],a[0])
    assert NEW.validate_complete_BI_likelihood_revocations(normalized,a[3],a[7])==[b['native_corner']]
    assert normalized[0]['state']==state and b==saved
    assert normalized[0]['association_provenance']['original_measurement_source']==a[3][0]['source']
    assert not normalized[0]['legacy_fallback_B_record']


@pytest.mark.parametrize('field',['original_observed_xy','original_observed_covariance_px2',
    'original_effective_fit_covariance_px2','original_measurement_source','original_fit_reliability_weight'])
def test_changed_archive_rejected_before_complete_cost_revocation(field):
    a,b=typed_fixture()
    if field=='original_measurement_source':b['association_provenance'][field]='VISIBLE_BACKBONE_FALLBACK'
    elif field=='original_fit_reliability_weight':b['association_provenance'][field]=.7
    else:
        value=np.asarray(b['association_provenance'][field]);value.flat[0]+=.01
        b['association_provenance'][field]=value.tolist()
    normalized=NEW._boundary_inputs([b],a[2],a[5],a[0])
    with pytest.raises(ValueError,match='archive|source exactly|reliability exactly'):
        NEW.validate_complete_BI_likelihood_revocations(normalized,a[3],a[7])


@pytest.mark.parametrize('state',['S','U','O','T'])
def test_forbidden_states_gate_before_poison_line_or_coordinate_fields(state):
    a,b=typed_fixture();k=b['native_corner'];a[2][k]=state
    with pytest.raises(ValueError,match='only B/I'):
        solve(a,[{'native_corner':k,'lines':object(),'association_provenance':object()}],weight=0.)


def test_explicit_reason_and_partial_readout_required_corrected_legacy_rejected():
    a,b=typed_fixture();bad=deepcopy(b);bad['association_provenance']['revocation_reason']='because_GT'
    with pytest.raises(ValueError,match='revocation reason'):NEW._boundary_inputs([bad],a[2],a[5])
    bad=deepcopy(b);bad['association_provenance']['display_coordinate_is_full_2D_measurement']=True
    with pytest.raises(ValueError,match='partial'):NEW._boundary_inputs([bad],a[2],a[5])
    a=arguments();b=observation(a);a[3][0]['source']='B_ACTUAL_BOUNDARY_RECOVERY'
    normalized=NEW._boundary_inputs([b],a[2],a[5],a[0])
    with pytest.raises(ValueError,match='fallback source only'):
        NEW.validate_complete_BI_likelihood_revocations(normalized,a[3],a[7])


def test_empty_zero_exact_immutable_parent_and_legacy_B_compatibility():
    a=arguments();regions=[region(a)];parent=OLD.solve_boundary_u_region_joint_ot_pnp(*a[:7],config=a[7],u_regions=regions)
    assert solve(a,u_regions=regions)==parent
    a,b=typed_fixture();parent=OLD.solve_boundary_u_region_joint_ot_pnp(*a[:7],config=a[7],u_regions=regions)
    assert solve(a,[b],weight=0.,u_regions=regions)==parent
    a=arguments();b=observation(a)
    normalized=NEW._boundary_inputs([b],a[2],a[5],a[0])
    assert normalized[0]['legacy_fallback_B_record']
    assert NEW.validate_complete_BI_likelihood_revocations(normalized,a[3],a[7])==[b['native_corner']]


def test_corrected_I_complete_cost_removed_normal_only_gradient_FD():
    a,b=typed_fixture();scale,p=parameters(a);cfg=BASE._configuration({**a[7],'prior_weight':0.})
    boundary=NEW._boundary_inputs([b],a[2],a[5],a[0]);k=b['native_corner']
    measure=dict(native_corner=k,xy=np.asarray(a[3][0]['xy']),whitening=np.eye(2),fit_reliability_weight=.8)
    prior=dict(R=np.asarray(a[6]['R']),center=np.asarray(a[6]['t']),whitening=np.eye(6))
    at=p+np.array([.01,-.02,.005,.008,-.011,.025])
    value,grad,meta=NEW._joint_objective_pair(at,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)
    numeric=V._jacobian(lambda pp:np.array([NEW._joint_objective_pair(pp,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)[0]]),at)[0]
    assert np.allclose(grad,numeric,atol=2e-5,rtol=2e-6)
    other={**measure,'xy':measure['xy']+1000.,'whitening':np.eye(2)*100.}
    altered=NEW._joint_objective_pair(at,a[0],a[1],[other],scale,prior,cfg,boundary,1.)
    assert altered[0]==value and np.array_equal(altered[1],grad)
    assert meta['removed_complete_BI_2D_likelihood_ids']==[k]
    shifted=p.copy();shifted[3]+=.04
    assert NEW._joint_objective_pair(p,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)[0]==NEW._joint_objective_pair(shifted,a[0],a[1],[measure],scale,prior,cfg,boundary,1.)[0]


def test_typed_I_joint_guard_Jacobian_keeps_old_OT_U_and_visible_guards():
    a,b=typed_fixture();scale,p=parameters(a);cfg=BASE._configuration(a[7])
    xyz,K,states,ms,ds,hw=V._inputs(*a[:6],BASE._helper_config(cfg))
    boundary=NEW._boundary_inputs([b],states,hw,xyz);regions=U._u_region_inputs([region(a)],states)
    values,jac=NEW._joint_constraint_pair(p,xyz,K,ms,ds,(None,),hw,scale,cfg,boundary,regions)
    numeric=V._jacobian(lambda pp:NEW._joint_constraint_pair(pp,xyz,K,ms,ds,(None,),hw,scale,cfg,boundary,regions)[0],p)
    assert np.allclose(jac,numeric,atol=2e-5,rtol=2e-6)
    old,_=BASE._constraints(p,xyz,K,ms,ds,(None,),hw,scale,cfg)
    assert np.array_equal(values[:len(old)],old) and len(values)==len(old)+3


def test_active_corrected_I_seed_Q_ranges_archives_and_generator_reuse():
    a,b=typed_fixture();saved=deepcopy(a[3]);saved_b=deepcopy(b);regions=[region(a)]
    parent=OLD.solve_boundary_u_region_joint_ot_pnp(*a[:7],config=a[7],u_regions=regions)
    result=solve(a,[b],u_regions=regions)
    assert result['accepted'] and result['states']==a[2]
    assert result['BI_prior_seed']==parent['BI_prior_seed']
    assert result['partial_BI_constraints_satisfied'] and result['U_constraints_satisfied']
    assert result['OT_constraints_satisfied'] and all(x['feasible'] for x in result['joint_constraint_audit'])
    k=b['native_corner'];assert result['reported_native'][k] is None
    readout=result['boundary_readout'][str(k)]
    assert readout['state']=='I' and readout['reported_full_2D_xy'] is None
    assert readout['archived_complete_BI_measurement_source']==a[3][0]['source']
    assert np.allclose(result['R_physical'],np.asarray(result['R'])@a[6]['physical_registration_Q'])
    assert a[3]==saved and b==saved_b
    generated=list(a);generated[3]=iter(a[3]);generated[4]=iter(a[4])
    assert solve(generated,[b],u_regions=regions,initializations=iter([]))==result
