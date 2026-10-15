"""State/readout and disabled-prior contracts for the uniform controller."""
from copy import deepcopy

import numpy as np
import pytest

from . import evaluate_prepartial_RGB_preservation_v4_26 as V


def task():
    xyz=V.C.cuboid(.8,.14,.59).tolist()
    return dict(states=['B','U','B','U','B','O','S','S'],
        initial_self_hidden=[False]*6+[True,True],boundary_merged=[dict(native_corner=k)for k in(0,4)],
        spec=dict(K=[[500.,0,320.],[0,500.,240.],[0,0,1.]],visible=[
            dict(native_corner=0,source='B_TARGET_BACKGROUND_LINES',xy=[310.,230.]),
            dict(native_corner=2,source='B_ACTUAL_BOUNDARY_RECOVERY',xy=[330.,250.]),
            dict(native_corner=4,source='VISIBLE_BACKBONE_FALLBACK',xy=[311.,231.])]),
        families=dict(ORIGINAL_WD=dict(native_xyz=xyz,physical_registration_Q=np.eye(3).tolist(),
            coarse_pose=dict(R=np.eye(3).tolist(),t=[0.,0.,3.]))))


def test_restore_rgb_partial_but_keep_exact_fallback_partial():
    t=task();before=deepcopy(t)
    assert V._restored_ids(t)==[0]
    assert t==before


def test_unknown_nonempty_source_keeps_the_same_point_policy():
    t=task();t['spec']['visible'][0]['source']='UNRECOGNIZED_NONEMPTY_SOURCE'
    assert V._restored_ids(t)==[0]


class StateOnlyMeasurement(dict):
    def __getitem__(self,key):
        if key!='native_corner':raise AssertionError('Hidden observation field accessed')
        return super().__getitem__(key)


@pytest.mark.parametrize('k',[1,6,7])
def test_gate_before_hidden_or_U_measurement_fields(k):
    t=task();t['spec']['visible'].insert(0,StateOnlyMeasurement(native_corner=k))
    with pytest.raises(ValueError,match='gate'):V._restored_ids(t)


def test_readout_restores_only_original_RGB_and_has_no_precise_U():
    t=task();family=t['families']['ORIGINAL_WD']
    selected=dict(family=family,accepted=True)
    pose=dict(R_cf=np.eye(3).tolist(),centroid=[0.,0.,3.])
    bundle=dict(runtime_complete_BI_replacement_plan=dict(corner_plans=[
        dict(native_corner=0,point_cost_revoked=False),dict(native_corner=2,point_cost_revoked=False)]))
    r=V._readout(t,selected,pose,bundle)
    assert r['reported_points'][0]==t['spec']['visible'][0]['xy']
    assert r['reported_points'][4] is None
    assert r['reported_points'][1] is None and r['reported_points'][3] is None
    for k in(6,7):assert r['reported_points'][k]==r['model_projection8'][k]
    assert r['new_full_2D_corner_measurements']==0


def test_revoked_fallback_readout_is_not_a_precise_corner():
    t=task();t['spec']['visible'][1]['source']='VISIBLE_BACKBONE_FALLBACK'
    selected=dict(family=t['families']['ORIGINAL_WD'],accepted=True)
    pose=dict(R_cf=np.eye(3).tolist(),centroid=[0.,0.,3.])
    bundle=dict(runtime_complete_BI_replacement_plan=dict(corner_plans=[
        dict(native_corner=2,point_cost_revoked=True,tangent_kind='ARCHIVED_COMPLETE_BI_TANGENT')]))
    r=V._readout(t,selected,pose,bundle)
    assert r['reported_points'][2] is None
    assert r['point_sources'][2]=='IMAGE_LINES_AND_UNCERTAIN_ARCHIVED_TANGENT'


def test_a_restored_RGB_point_cannot_be_revoked_again():
    t=task();selected=dict(family=t['families']['ORIGINAL_WD'],accepted=True)
    pose=dict(R_cf=np.eye(3).tolist(),centroid=[0.,0.,3.])
    bundle=dict(runtime_complete_BI_replacement_plan=dict(corner_plans=[dict(native_corner=0,point_cost_revoked=True)]))
    with pytest.raises(ValueError,match='non-restored'):V._readout(t,selected,pose,bundle)


def test_noP_control_has_exact_zero_value_and_gradient():
    t=task();models=dict(COARSE=dict(prior_covariance6=np.eye(6).tolist(),prior_fit=dict(median_bias6=[0.]*6)),
        COARSE_BINDING=dict(sha256='a'*64))
    zero=V._prepare_pose_prior(t,models,'FULL')
    translation=V._prepare_pose_prior(t,models,'TRANSLATION')
    f=t['families']['ORIGINAL_WD'];xyz=np.asarray(f['native_xyz']);scale=1.2
    p=V.BASE.NATIVE.V._pose_vector(np.eye(3),np.array([.2,-.1,4.]),scale)
    for prior,mode in((zero,'FULL'),(translation,'TRANSLATION')):
        bound=V.PRIOR.rebind_coarse_pose_prior(prior,xyz,f['physical_registration_Q'])
        value,gradient,audit=V.PRIOR.coarse_pose_prior_value_and_gradient(p,xyz,scale,bound)
        if mode=='FULL':
            assert value==0. and np.array_equal(gradient,np.zeros(6))
            assert audit['factor_weight']==0.
        else:assert value>0. and np.linalg.norm(gradient)>0


def test_exact_design_rejects_accidental_active_full_prior():
    d=V.C.read(V.PLAN);V._check_declaration(d)
    d['profiles'][0]['coarse_prior_weight']=1.
    with pytest.raises(ValueError,match='preregistered'):V._check_declaration(d)
