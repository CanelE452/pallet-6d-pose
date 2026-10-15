import ast
import json
import numpy as np
import pytest
from scripts.research.pallet_minimal_bi_ablation_v1 import inputs as I


def test_lexical_whitelist_never_decodes_reference_payload():
    parse=I.prediction_parser()
    # Deliberately invalid JSON inside the skipped field demonstrates that
    # its numeric/reference content is never given to json.raw_decode.
    row='{"id":"case", "reference":{"poison":INVALID},"method":"BASE","native_points":[[1,2]]}'
    assert parse(row,{'id','method','native_points'})==dict(id='case',method='BASE',native_points=[[1,2]])


def test_original_prediction_AST_excludes_metric_and_metadata():
    ns={'np':np,'cv2':object(),'select_pnp_hypotheses':None}
    I._defs(I.SOURCE_ROOT/I.POSE_SOURCE,{'rotations','infer'},ns)
    assert 'metric' not in ns and 'metadata' not in ns and 'main' not in ns
    expected=ast.parse((I.SOURCE_ROOT/I.POSE_SOURCE).read_text())
    assert [n.name for n in expected.body if isinstance(n,ast.FunctionDef) and n.name in ('rotations','infer')]==['rotations','infer']


def test_old_F_projected_cube_and_own_Q_center():
    import cv2
    infer=I.existing_F()
    ns={'np':np,'cv2':cv2}
    I._defs(I.SOURCE_ROOT/I.SOLVE_SOURCE,{'cuboid'},ns)
    xyz=np.array([1.1,.11,1.3]);K=np.array([[600.,0,320],[0,610,240],[0,0,1]])
    R=cv2.Rodrigues(np.array([.3,-.5,.1]))[0];t=np.array([.1,-.1,4.])
    q=cv2.projectPoints(ns['cuboid'](*xyz),cv2.Rodrigues(R)[0],t,K,None)[0].reshape(-1,2)
    pose=infer(np.vstack((q,q.mean(0))),K,xyz,False)
    assert pose['available']
    assert np.allclose(pose['centroid'],t,atol=1e-8)
    assert np.allclose(np.asarray(pose['R_cf']).T@pose['R_physical'],np.eye(3),atol=1e-8)
    assert infer(None,K,xyz,False)==dict(available=False)


def test_pose_comparison_has_absolute_only_tolerance():
    p=dict(available=True,selected_hypothesis='WD',R_cf=np.eye(3).tolist(),R_physical=np.eye(3).tolist(),centroid=[0,0,4],cf_extents=[1,.1,2],reprojection_px=1.)
    assert I.assert_pose_equal(p,json.loads(json.dumps(p)))==0
    other=json.loads(json.dumps(p));other['centroid'][0]=2e-9
    with pytest.raises(ValueError,match='centroid'):I.assert_pose_equal(p,other)


def test_recording_group_merges_aliases_not_collection():
    doc={'groups':[{'recording_id':'REC_1','is_collection':False,'sessions':[{'session_key':'data/x/sessions/daywood'},{'session_key':'data/x/sessions/dayplastic'}]}, {'recording_id':'COLL','is_collection':True,'sessions':[{'session_key':'data/x/sessions/daywood'}]}]}
    assert I.recording_map(doc)=={'daywood':'REC_1','dayplastic':'REC_1'}
