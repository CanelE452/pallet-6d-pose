"""Registration and observation provenance boundaries for frozen e inputs."""
from copy import deepcopy
import json

import numpy as np
import pytest

from . import e_inputs as E


def pose(extents=(1.1,.11,1.3), Q=None):
    Q = np.eye(3) if Q is None else np.asarray(Q, dtype=float)
    return dict(available=True, R_cf=Q.T.tolist(), R_physical=np.eye(3).tolist(),
        physical_registration_Q=Q.tolist(), centroid=[0.,0.,4.],
        cf_extents=list(extents), fallback=False)


def specimen():
    points = np.arange(18, dtype=float).reshape(9,2).tolist()
    frame = dict(id='session:frame', partition='CAL', session='session', image='rgb/frame.png',
        image_sha256='a'*64, xyz=[1.1,.11,1.3], K=[[500,0,320],[0,500,240],[0,0,1]],
        arms={'N3_SUBPIX':dict(native_points=points, keypoints_conf=[i/10 for i in range(9)],
            prediction_support=[True]*9, coarse_pose=pose())})
    Q = np.array([[0,0,1],[0,1,0],[-1,0,0.]])
    archived = {k:deepcopy(frame[k]) for k in ('id','partition','session','image','image_sha256')}
    archived['e'] = dict(actual_pose=pose((1.3,.11,1.1), Q), accepted=True,
        WD_family='SWAPPED_WD', method='FROZEN_WD_FACE_AWARE_PARTIAL_BI')
    return frame, archived


def test_unique_physical_registration_reindexes_only_observations():
    frame, archived = specimen()
    before, frozen = deepcopy(frame), deepcopy(archived)
    out = E.attach_e_branch(frame, archived)
    branch = out['arms']['E_FROZEN']
    ids = branch['native_vertex_mapping']['target_to_source_native_ids']
    assert sorted(ids) == list(range(8)) and ids != list(range(8))
    assert branch['native_points'][:8] == [before['arms']['N3_SUBPIX']['native_points'][k] for k in ids]
    assert branch['keypoints_conf'][:8] == [before['arms']['N3_SUBPIX']['keypoints_conf'][k] for k in ids]
    assert branch['native_points'][8] == before['arms']['N3_SUBPIX']['native_points'][8]
    assert branch['coarse_pose'] == archived['e']['actual_pose']
    assert frame == before and archived == frozen
    mapping = branch['native_vertex_mapping']
    np.testing.assert_allclose(np.array(mapping['source_physical_vertex_xyz'])[ids],
        mapping['target_physical_vertex_xyz'], atol=1e-12, rtol=0)
    assert not branch['contract']['model_projection_or_mixed_readout_used_as_observation']


def test_family_change_does_not_use_e_model_projection_or_readout():
    frame, archived = specimen()
    archived['e']['projected_native'] = [[999999,999999]]*8
    archived['e']['keypoint_readout'] = {'reported_points':[[999999,999999]]*8}
    out = E.attach_e_branch(frame, archived)
    assert max(max(p) for p in out['arms'][E.BRANCH]['native_points']) == 17


def test_nonbijective_or_unknown_dimensions_fail_closed():
    frame, archived = specimen()
    archived['e']['actual_pose']['cf_extents'] = [1.2,.11,1.1]
    with pytest.raises(ValueError, match='no unique fixed physical vertex bijection'):
        E.attach_e_branch(frame, archived)
    with pytest.raises(ValueError, match='no unique fixed physical vertex bijection'):
        E.physical_vertex_bijection(pose((1e-10,1e-10,1e-10)),
                                   pose((1e-10,1e-10,1e-10)), [1e-10]*3)


@pytest.mark.parametrize('field,value', [('image_sha256','b'*64), ('id','other:frame')])
def test_identity_mismatch_rejected(field, value):
    frame, archived = specimen()
    archived[field] = value
    with pytest.raises(ValueError, match='identity mismatch'):
        E.attach_e_branch(frame, archived)


def test_inconsistent_physical_Q_rejected():
    frame, archived = specimen()
    archived['e']['actual_pose']['physical_registration_Q'] = np.eye(3).tolist()
    with pytest.raises(ValueError, match='registration differ'):
        E.attach_e_branch(frame, archived)


def test_frozen_pose_behind_camera_rejected():
    frame, archived = specimen()
    archived['e']['actual_pose']['centroid'] = [0,0,-4]
    with pytest.raises(ValueError, match='positive camera depth'):
        E.attach_e_branch(frame, archived)


def test_lexical_whitelist_does_not_decode_unused_nested_records(monkeypatch):
    original, decoded = json.loads, []
    def tracked(text, *a, **kw):
        decoded.append(text)
        return original(text, *a, **kw)
    monkeypatch.setattr(E.json, 'loads', tracked)
    text = '{"id":"frame","unused":{"sensitive_metadata":[1,2,{"escaped":"}\\\""}]},"arms":{"e":{"available":true}}}'
    selected = E.select_fields(text, {'id','arms'}, raw_fields={'arms'})
    assert selected == {'id':'frame', 'arms':'{"e":{"available":true}}'}
    assert all('sensitive_metadata' not in value for value in decoded)


def test_duplicate_json_keys_rejected_even_when_unused():
    with pytest.raises(ValueError, match='Unique JSON object keys'):
        E.select_fields('{"id":"frame","unused":1,"unused":2}', {'id'})
