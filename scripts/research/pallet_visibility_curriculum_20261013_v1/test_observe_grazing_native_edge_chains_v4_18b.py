from copy import deepcopy

import numpy as np
import pytest

from . import observe_grazing_native_edge_chains_v4_18b as D


def _fixture():
    geometry=dict(native_xyz=np.arange(24,dtype=float).reshape(8,3).tolist(),K=np.eye(3).tolist(),
        physical_registration_Q=np.eye(3).tolist(),cf_extents=[1.,.2,.8],R_cf=np.eye(3).tolist(),
        centroid=[0.,0.,3.],initial_self_hidden=[False]*5+[True,False,False],image_hw=[480,640])
    states=['I','B','U','O','T','S','B','U']
    visible=[dict(native_corner=k,state=states[k],xy=[100.+k,200.+k],covariance=[[2.,0.],[0.,3.]],
        effective_fit_covariance=[[4.,0.],[0.,5.]],reliability=.5,source='IMAGE_ORIGINAL') for k in (0,1,6)]
    old_regions=[dict(native_corner=k,state='U',center_xy=[200.,300.],half_axes_px=[50.,50.]) for k in (2,7)]
    original=dict(id='s:1',partition='CAL',session='s',image='image.png',image_sha256='a'*64,
        mask_binding=dict(path='mask.npz',sha256='b'*64,bytes=10),states=states,initial_geometry=geometry,
        visible_measurements=visible,domains=[dict(native_corner=3,state='O')],u_regions=old_regions,
        old_boundary_all_sha256='c'*64,nonself_prediction_scale_px=100.,
        unchanged_observation_math_sha256='d'*64)
    new_states=deepcopy(states);new_states[1]='U'
    region=dict(native_corner=1,state='U',center_xy=[101.,201.],half_axes_px=[54.,54.])
    adapted=dict(id='s:1',partition='CAL',session='s',image='image.png',image_sha256='a'*64,
        states=new_states,initial_self_hidden=geometry['initial_self_hidden'],
        families={'ORIGINAL_WD':dict(native_xyz=geometry['native_xyz'],cf_extents=geometry['cf_extents'],
            physical_registration_Q=geometry['physical_registration_Q'],coarse_pose=dict(R=geometry['R_cf'],t=geometry['centroid']))},
        spec=dict(states=new_states,visible=[visible[0],visible[2]],domains=original['domains'],
            K=geometry['K'],image_hw=geometry['image_hw']),u_regions=old_regions+[region],boundary_merged=[],
        grazing_adaptation=dict(original_states=states,geometry_gate=dict(states=new_states,new_U_native_ids=[1]),
            new_U_regions=[region],missing_new_U_raw_native_ids=[],U_radius_px=54.,borrowed_CAL264_radius_ratio=.54))
    return deepcopy(original),deepcopy(adapted)


def test_join_keeps_whitelist_and_removes_new_U_before_query():
    original,adapted=_fixture();before=deepcopy((original,adapted))
    task=D._join_task(original,adapted)
    assert task['states'][1]=='U'
    assert [r['native_corner'] for r in task['visible_measurements']]==[0,6]
    assert task['visible_measurements']==[original['visible_measurements'][0],original['visible_measurements'][2]]
    assert task['initial_geometry']==original['initial_geometry']
    assert task['u_regions']==adapted['u_regions']
    assert 'proposals' not in task and 'original_raw_points' not in task
    assert (original,adapted)==before


@pytest.mark.parametrize('native_id,new_state',[(5,'U'),(2,'B'),(3,'U'),(4,'U')])
def test_old_S_U_O_T_cannot_change(native_id,new_state):
    original,adapted=_fixture()
    adapted['states'][native_id]=new_state
    with pytest.raises(ValueError):D._join_task(original,adapted)


def test_retained_noise_and_previous_regions_are_exact():
    original,adapted=_fixture();adapted['spec']['visible'][0]['xy'][0]+=1
    with pytest.raises(ValueError,match='Retained B/I'):D._join_task(original,adapted)
    original,adapted=_fixture();adapted['u_regions'][0]['center_xy'][0]+=1
    with pytest.raises(ValueError,match='Previous U'):D._join_task(original,adapted)


@pytest.mark.parametrize('field', ['K','physical_registration_Q'])
def test_initial_K_Q_must_match(field):
    original,adapted=_fixture()
    if field=='K':adapted['spec']['K'][0][0]+=1
    else:adapted['families']['ORIGINAL_WD'][field][0][0]+=.1
    with pytest.raises(ValueError,match='registered geometry'):D._join_task(original,adapted)


def test_worker_passes_new_gate_to_frozen_NMS_operation(monkeypatch):
    original,adapted=_fixture();task=D._join_task(original,adapted);seen=[]
    def saved_operation(received):
        seen.append(deepcopy(received))
        assert received['states'][1]=='U'
        assert all(r['native_corner']!=1 for r in received['visible_measurements'])
        return dict(contract={})
    monkeypatch.setattr(D.OLD,'observe_one',saved_operation)
    result=D.observe_one(task)
    assert len(seen)==1 and seen[0]==task
    assert result['new_grazing_U_native_ids']==[1]
    assert result['contract']['fresh_NMS_after_new_U_gate']
    assert result['contract']['new_CAL_noise_refit_required']
