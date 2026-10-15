"""Synthetic geometry only; no dataset, reference, image or model access."""
from copy import deepcopy
import inspect

import cv2
import numpy as np
import pytest

from . import fresh_observed_BI_pnp_proposals_v4_32 as V


def fixture(ids=tuple(range(8))):
    xyz = np.array([[-.6,-.1,-.4],[.6,-.1,-.4],[.6,.1,-.4],[-.6,.1,-.4],
                    [-.6,-.1,.4],[.6,-.1,.4],[.6,.1,.4],[-.6,.1,.4]]) + [.17,-.08,.23]
    R = cv2.Rodrigues(np.array([.19,-.31,.12]))[0]
    t = np.array([.11,-.18,3.4])
    K = np.array([[793.,31.,421.],[0.,811.,287.],[0.,0.,1.]])
    Q = cv2.Rodrigues(np.array([0.,np.pi/2,0.]))[0]
    h = (xyz @ R.T + t) @ K.T
    xy = h[:,:2] / h[:,2,None]
    states = ['B' if k in ids else 'U' for k in range(8)]
    visible = [dict(native_corner=k,xy=xy[k].tolist(),source='VISIBLE_BACKBONE_FALLBACK',
        covariance_px2=[[2.,.1],[.1,3.]],fit_covariance_px2=[[4.,.2],[.2,5.]],
        fit_reliability_weight=.7) for k in ids]
    coarse = dict(R=cv2.Rodrigues(np.array([.21,-.28,.09]))[0].tolist(),
        t=[.14,-.2,3.6],source=V.COARSE_SOURCE,independent_of_cached_pose_bank=True,
        physical_registration_Q=Q.tolist())
    return dict(native_xyz=xyz,K=K,states=states,initial_self_hidden=[False]*8,
        visible_measurements=visible,initial_coarse_pose=coarse,physical_registration_Q=Q),R,t


def nearest(result,R,t):
    return min(np.linalg.norm(np.asarray(p['R'])-R)+np.linalg.norm(np.asarray(p['t'])-t)
        for p in result['proposals'])


def test_full_cube_skew_K_nonzero_mean_fresh_solution_and_fixed_budget():
    inputs,R,t=fixture();before=deepcopy(inputs)
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert result['subset_count']==127 and result['SQPNP_calls']<=127
    assert result['fresh_seed_count']>0 and nearest(result,R,t)<1e-6
    assert result['admitted_BI_ids']==list(range(8))
    for p in result['proposals']:
        Rp=np.asarray(p['R']);tp=np.asarray(p['t'])
        assert np.allclose(Rp.T@Rp,np.eye(3),atol=1e-7) and np.linalg.det(Rp)>0
        assert np.all((inputs['native_xyz']@Rp.T+tp)[:,2]>0)
        assert np.allclose(p['centroid'],Rp@inputs['native_xyz'].mean(axis=0)+tp)
        assert np.allclose(p['R_physical'],Rp@inputs['physical_registration_Q'])
        assert p['centroid_is_native_t'] is False and p['covariance_weighted_PnP'] is False
    for key in ('native_xyz','K','physical_registration_Q'):
        assert np.array_equal(inputs[key],before[key])
    assert inputs['visible_measurements']==before['visible_measurements']
    assert inputs['initial_coarse_pose']==before['initial_coarse_pose']
    assert result['admitted_BI_archive'][0]['fit_covariance_px2']==before['visible_measurements'][0]['fit_covariance_px2']


def test_exact_three_BI_uses_local_SQPNP_and_retains_all_valid_solutions():
    inputs,R,t=fixture((0,1,4))
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert result['subset_count']==1 and result['SQPNP_calls']==1
    assert nearest(result,R,t)<1e-5


class Poison(dict):
    def __getitem__(self,key):
        if key!='native_corner':raise AssertionError('Excluded payload read')
        return super().__getitem__(key)
    def __contains__(self,key):raise AssertionError('Excluded payload inspected')
    def __iter__(self):raise AssertionError('Excluded payload serialized')
    def __deepcopy__(self,memo):raise AssertionError('Excluded payload copied')


def test_initial_S_and_S_U_O_T_gate_before_any_measurement_payload():
    inputs,_,_=fixture((0,1,4))
    inputs['states']=['B','I','S','U','B','O','T','B']
    inputs['initial_self_hidden'][7]=True
    inputs['visible_measurements'] += [Poison(native_corner=k) for k in (2,3,5,6,7)]
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert result['admitted_BI_ids']==[0,1,4]
    assert result['new_S_U_O_T_measurement_payload_reads']==0


def test_fewer_than_three_returns_only_own_attested_coarse_not_a_cache_fallback():
    inputs,_,_=fixture((0,1))
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert len(result['proposals'])==1 and result['SQPNP_calls']==0
    assert result['proposals'][0]['source']==V.COARSE_SOURCE
    assert result['abstention_reason']=='fewer_than_three_BI'
    assert result['coarse_independence_is_caller_attestation_not_inferred'] is True


def test_degenerate_image_and_solver_failure_abstain_without_hidden_rescue(monkeypatch):
    inputs,_,_=fixture((0,1,4))
    for record in inputs['visible_measurements']:record['xy']=[12.,34.]
    def forbidden(*args,**kw):raise AssertionError('Degenerate subset must not call solver')
    monkeypatch.setattr(V.cv2,'solvePnPGeneric',forbidden)
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert result['fresh_seed_count']==0 and result['SQPNP_calls']==0
    inputs,_,_=fixture((0,1,4))
    def failure(*args,**kw):raise cv2.error('synthetic failure')
    monkeypatch.setattr(V.cv2,'solvePnPGeneric',failure)
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert len(result['proposals'])==1 and result['SQPNP_calls']==1
    assert result['rejected_subsets_or_solutions'][0]['reason']=='SQPNP_failed'


def test_bad_depth_nonfinite_and_duplicate_solutions_are_audited(monkeypatch):
    inputs,_,_=fixture((0,1,4))
    own=inputs['initial_coarse_pose'];rv=cv2.Rodrigues(np.asarray(own['R']))[0]
    def solutions(*args,**kw):
        return 4,[rv,rv,np.array([[np.nan],[0.],[0.]]),np.zeros((3,1))],\
            [np.asarray(own['t'])[:,None],np.asarray(own['t'])[:,None],np.ones((3,1)),np.array([[0.],[0.],[-5.]])]
    monkeypatch.setattr(V.cv2,'solvePnPGeneric',solutions)
    result=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    assert len(result['proposals'])==1 and result['valid_fresh_solver_outputs_before_numeric_dedup']==2
    assert len(result['proposals'][0]['duplicate_generation_provenance'])==2
    assert len(result['rejected_subsets_or_solutions'])==2


def test_consistent_physical_Q_and_native_basis_reexpression_preserve_pose():
    inputs,R,t=fixture();first=V.generate_fresh_observed_BI_pnp_proposals(**inputs)
    A=cv2.Rodrigues(np.array([.12,.41,-.07]))[0]
    changed=deepcopy(inputs);changed['native_xyz']=inputs['native_xyz']@A
    changed['physical_registration_Q']=A.T@inputs['physical_registration_Q']
    changed['initial_coarse_pose']['R']=(np.asarray(inputs['initial_coarse_pose']['R'])@A).tolist()
    changed['initial_coarse_pose']['physical_registration_Q']=changed['physical_registration_Q'].tolist()
    second=V.generate_fresh_observed_BI_pnp_proposals(**changed)
    assert nearest(second,R@A,t)<1e-6
    p1=min(first['proposals'],key=lambda p:np.linalg.norm(np.asarray(p['R'])-R)+np.linalg.norm(np.asarray(p['t'])-t))
    p2=min(second['proposals'],key=lambda p:np.linalg.norm(np.asarray(p['R'])-R@A)+np.linalg.norm(np.asarray(p['t'])-t))
    assert np.allclose(p1['R_physical'],p2['R_physical'],atol=1e-7)
    assert np.allclose(p1['centroid'],p2['centroid'],atol=1e-7)


@pytest.mark.parametrize('change',['rotation','Q_mismatch','covariance','source','K','native_id'])
def test_invalid_bindings_fail_closed(change):
    inputs,_,_=fixture((0,1,4))
    if change=='rotation':inputs['initial_coarse_pose']['R'][0][0]+=1
    elif change=='Q_mismatch':inputs['initial_coarse_pose']['physical_registration_Q']=np.eye(3).tolist()
    elif change=='covariance':inputs['visible_measurements'][0]['fit_covariance_px2']=[[1.,2.],[2.,1.]]
    elif change=='source':inputs['initial_coarse_pose']['source']='SAME_FROZEN_E'
    elif change=='K':inputs['K'][0,0]=0
    else:inputs['visible_measurements'][0]['native_corner']=True
    with pytest.raises(ValueError):V.generate_fresh_observed_BI_pnp_proposals(**inputs)


def test_no_cached_bank_or_baseline_parameters_can_enter_public_API():
    names=set(inspect.signature(V.generate_fresh_observed_BI_pnp_proposals).parameters)
    assert names=={'native_xyz','K','states','initial_self_hidden','visible_measurements',
        'initial_coarse_pose','physical_registration_Q'}
    inputs,_,_=fixture((0,1))
    with pytest.raises(TypeError):V.generate_fresh_observed_BI_pnp_proposals(**inputs,candidate_bank=[])
