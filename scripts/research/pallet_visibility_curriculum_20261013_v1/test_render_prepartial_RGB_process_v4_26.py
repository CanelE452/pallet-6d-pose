"""Pure report contracts; no actual rows, metrics, pixels or reference reads."""
from copy import deepcopy

import numpy as np
import pytest

from . import render_prepartial_RGB_process_v4_26 as R


def sample_arm(key):
    zero=key in (R.ZERO_NATIVE,R.ZERO_LOCK)
    containment=key in R.CONTAINMENT_METHODS
    p=0. if zero else 4.
    native,ov,sv=10.,2.,3.
    shape,weighted=(4.,6.)if containment else(0.,0.)
    total=native+ov+sv+p+weighted
    sol=dict(native_observation_objective=native,regularizer_costs=[
        dict(kind=kind,weighted_cost=value)for kind,value in zip(R.EXP.P.REGULARIZER_KINDS,(ov,sv,p))],
        objective=total,coarse_pose_prior_count=1,CAL_coarse_pose_prior_audit=dict(
            mode='FULL'if zero else'TRANSLATION',factor_weight=0. if zero else 1.,
            mean_bias_subtracted_once=True,translation_covariance_is_marginal_block=not zero))
    if containment:sol.update(global_containment_cost=shape,weighted_containment_objective=weighted,
        global_containment_weight=1.5,containment_count_in_optimizer=1)
    return dict(accepted=True,solution=sol,candidate_selection=dict(selected_name='one',
        old_full_outline_factor_or_tie_break_used=False,candidates=[dict(name='one',total_energy=total)]))


def test_three_fixed_cases_seven_uniform_methods_and_explicit_memory_gate(monkeypatch):
    assert R.CASES==('eval_noapril:1775201414728017664','plastic_day_01:005516','wood_night_01:031426')
    assert R.METHODS==tuple(R.EXP.ARMS) and len(R.METHODS)==7
    assert len(R.TITLES)==len(R.CAPTIONS)==14
    assert R.CONTAINMENT_METHODS==(R.TP_C,R.TP_LOCK)
    assert R.LOCKED_METHODS==(R.ZERO_LOCK,R.TP_NATIVE_LOCK,R.TP_LOCK)
    def forbidden(*args,**kwargs):raise AssertionError('Read before root GO')
    monkeypatch.setattr(R.C,'read',forbidden)
    monkeypatch.setattr(R.C,'rows',forbidden)
    with pytest.raises(ValueError,match='explicit root'):R.render()


@pytest.mark.parametrize('key',R.METHODS[1:])
def test_zero_and_TP_ledger_C_once_with_no_e_cost_or_original_mask(key):
    arm=sample_arm(key);before=deepcopy(arm)
    energy=R._energies(arm,key)
    assert arm==before
    assert energy['total']==sum(energy[v]for v in ('native','O','S','P','weighted_C'))
    assert energy['coarse_prior_weight']==(0. if key in(R.ZERO_NATIVE,R.ZERO_LOCK)else 1.)
    if key in(R.ZERO_NATIVE,R.ZERO_LOCK):assert energy['P']==0.
    if key not in R.CONTAINMENT_METHODS:assert energy['weighted_C']==energy['C']==0.
    assert not energy['old_full_outline_or_numeric_mask_tie_used']
    assert not R._energies({},R.BASELINE)['applied']


def test_noP_cannot_accidentally_become_a_full_rotation_prior():
    a=sample_arm(R.ZERO_NATIVE);a['solution']['CAL_coarse_pose_prior_audit']['factor_weight']=1.
    with pytest.raises(ValueError,match='zero-schema'):R._energies(a,R.ZERO_NATIVE)
    a=sample_arm(R.ZERO_NATIVE);a['solution']['regularizer_costs'][-1]['weighted_cost']=1.
    with pytest.raises(ValueError,match='exactly zero'):R._energies(a,R.ZERO_NATIVE)


def test_original_target_same_with_zero_full_schema_and_active_marginal_TP():
    xyz=R.C.cuboid(1.3,.11,1.1)+[.02,-.01,.03]
    original=dict(original_native_xyz=xyz.tolist(),original_R_cf=np.eye(3).tolist(),
        original_t=[.1,.3,2.4],original_first_R_physical=np.eye(3).tolist(),
        original_first_center_camera_xyz=(xyz.mean(0)+[.1,.3,2.4]).tolist(),
        prior_covariance6=np.diag(np.arange(1.,7.)).tolist(),prior_bias6=[.01]*6,bias_strength=1.)
    priors=dict(TRANSLATION={**deepcopy(original),'factor_weight':1.},FULL={**deepcopy(original),'factor_weight':0.})
    K=np.array([[600.,2.,320.],[0.,610.,240.],[0.,0.,1.]])
    before=deepcopy(priors)
    geometry=R._original_coarse_geometry(priors,K)
    assert priors==before
    assert geometry['noP_weight']==0. and geometry['TP_weight']==1.
    np.testing.assert_array_equal(geometry['TP_marginal_covariance3'],np.diag([4.,5.,6.]))
    np.testing.assert_array_equal(geometry['original_first_model_projection_xy'],R.C.project(xyz,np.eye(3),np.array([.1,.3,2.4]),K))
    assert not geometry['e_numeric_pose_used']
    priors['FULL']['factor_weight']=1.
    with pytest.raises(ValueError,match='placeholder weight0'):R._original_coarse_geometry(priors,K)


def test_restored_display_gates_hidden_payload_before_parent_renderer(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Bad state reached point fields')
    monkeypatch.setattr(R.PREVIOUS,'_replacement_panel',forbidden)
    task=dict(states=['B','U','B','U','B','O','S','S'],initial_self_hidden=[False]*6+[True,True])
    for k in(1,6,7):
        with pytest.raises(ValueError,match='BI state gate'):
            R._restoration_panel(None,task,dict(restored_prepartial_complete_ids=[k]),'')
