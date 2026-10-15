from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from . import render_raw_yolo_N3_subpix_gain10_process_v1 as R


def fixture():
    K=[[500.,0,320.],[0,510.,240.],[0,0,1.]];dims=[1.2,.16,1.]
    pose=dict(available=True,R_cf=np.eye(3).tolist(),R_physical=np.eye(3).tolist(),
              centroid=[.1,.2,3.],cf_extents=dims,selected_hypothesis='WIDE')
    q=np.vstack((R.C.project(R.C.cuboid(*dims),np.eye(3),pose['centroid'],K),[320.,240.]))
    qN=q.copy();qN[:8]+=[1.,-1.];qS=qN.copy();qS[:8]+=[.2,.3];qF=qS.copy();qF[:8,0]-=.1
    frame=dict(id='fake:1',partition='TEST',image='fake.png',image_sha256='0'*64,raw_hw=[480,640],K=K,xyz=dims,
        points={'BASE':q.tolist(),'N3_SUBPIX':qF.tolist()},initial_poses={'BASE':deepcopy(pose),'N3_SUBPIX':deepcopy(pose)},prediction_support=[True]*9)
    stage=dict(id='fake:1',method='N3_SUBPIX',raw_hw=[480,640],q0=q.tolist(),qN=qN.tolist(),qS=qS.tolist(),qFinal=qF.tolist())
    methods={m:dict(id='fake:1',partition='TEST',method=m,native_points=deepcopy(frame['points'][k]),actual_pose=deepcopy(pose))
        for m,k in ((R.RAW,'BASE'),(R.AFTER,'N3_SUBPIX'))}
    return frame,stage,methods


def test_all_four_stages_are_exact_and_inputs_immutable():
    args=fixture();before=deepcopy(args);out=R.stage_case(*args,metric={})
    assert args==before
    for key in R.STAGES:assert out['stage_xy'][key]==args[1][key]
    assert set(out['poses'])=={R.RAW,R.AFTER}
    assert 'no stored standalone F pose' in out['qS']


def test_native_projection_uses_saved_cf_not_physical_rotation():
    K=np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]])
    Q=np.array([[0.,0.,1.],[0,1.,0.],[-1.,0.,0.]])
    pose=dict(available=True,R_cf=np.eye(3).tolist(),R_physical=Q.tolist(),centroid=[.1,0,3.],cf_extents=[1.,.16,1.2])
    actual,q=R.projection(pose,K,[1.2,.16,1.])
    expected=R.C.project(R.C.cuboid(1.,.16,1.2),np.eye(3),pose['centroid'],K)
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(q,Q)
    wrong=R.C.project(R.C.cuboid(1.,.16,1.2),Q,pose['centroid'],K)
    assert not np.allclose(actual,wrong)


def test_bad_physical_mapping_rejected():
    f,s,m=fixture();m[R.RAW]['actual_pose']['R_physical']=[[-1.,0,0],[0,1.,0],[0,0,-1.]]
    f['initial_poses']['BASE']=deepcopy(m[R.RAW]['actual_pose'])
    with pytest.raises(ValueError,match='physical Q'):R.stage_case(f,s,m,{})


@pytest.mark.parametrize('field',['q0','qFinal'])
def test_coordinate_identity_failure_is_not_silently_routed(field):
    f,s,m=fixture();s[field][0][0]+=.01
    with pytest.raises(ValueError,match='canonical coordinates'):R.stage_case(f,s,m,{})


def test_missing_subpix_stage_is_not_invented():
    f,s,m=fixture();del s['qS']
    with pytest.raises(KeyError):R.stage_case(f,s,m,{})


def test_whitelist_skips_even_malformed_reference_payload():
    line='{"id":"fake:1","corner":{"GT":NOT_VALID_JSON},"method":"N3_SUBPIX","qS":[[1,2]],"truth_pose":WRONG}'
    assert R.SELECT.select_prediction_fields(line,{'id','method','qS'})=={'id':'fake:1','method':'N3_SUBPIX','qS':[[1,2]]}


def test_viewport_only_uses_stored_q_not_pose_reference_or_metrics():
    f,s,m=fixture();case=R.stage_case(f,s,m,{})
    box=R.viewport(case);case['native_model_projection_xy']={R.RAW:[[1e9,-1e9]]*8}
    case['metrics']={'GT':[1e99,1e99]};assert R.viewport(case)==box


def test_go_before_any_io(monkeypatch):
    def poison(*args,**kwargs):raise AssertionError('IO before GO')
    monkeypatch.setattr(R.C,'read',poison);monkeypatch.setattr(R.C,'sha',poison)
    monkeypatch.setattr(Path,'exists',poison)
    with pytest.raises(PermissionError):R.main(root_saved_render_GO=False)


def test_pair_gain_loss_contract_is_exact():
    ids=list(R.FIXED_IDS)+[f'other:{i}' for i in range(39)]
    pair=dict(rows=49,ordered_ids=ids,groups={'gain':list(R.FIXED_IDS),'loss':[]},pairs=[])
    for i,identifier in enumerate(ids):
        pair['pairs'].append(dict(id=identifier,raw_yolo_native_base={'translation_cm':6 if i<10 else 2,'rotation_deg':1},N3_SUBPIX={'translation_cm':3,'rotation_deg':2}))
    assert R.paired_ids(pair)==R.FIXED_IDS
    pair['groups']['loss']=['unexpected']
    with pytest.raises(ValueError,match='zero losses'):R.paired_ids(pair)
