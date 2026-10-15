"""Whitelist/native basis checks; synthetic metadata, no dataset/GT access."""
from copy import deepcopy
import json

import numpy as np
import pytest

from . import remainder_native_inputs_v1 as NEW


def sample():
    reference=dict(id='new:1',image='unused.png',image_sha256='a'*64,image_hw=[480,640],K=[[600,0,320],[0,600,240],[0,0,1]],
        session='new',known_physical_dimensions_xyz_m=[1.1,.11,1.3],historical_prediction_metadata_object_type='known',
        upstream_prediction={'sha256':'b'*64},upstream_cache={'sha256':'c'*64})
    q=[[100.+k,200.+k] for k in range(9)]
    prediction=dict(id='new:1',method='N3_SUBPIX',raw_hw=[480,640],qFinal=q,native_points=deepcopy(q),
        actual_pose=dict(available=True,R_cf=np.eye(3).tolist(),R_physical=np.eye(3).tolist(),centroid=[0,0,3],cf_extents=[1.1,.11,1.3]))
    registry={'physical_dimensions_m':{'x':1.1,'y':.11,'z':1.3}}
    return reference,prediction,registry


def test_whitelist_skips_nested_scoring_without_decoding_values():
    # The forbidden object deliberately has invalid scalar JSON. Text skipping
    # succeeds; decoding it as scored/reference data would necessarily fail.
    line='{"id":"new:1","corner":{"reference": NOT_A_JSON_VALUE,"nested":[{"s":"}\\\"["}]},"method":"N3_SUBPIX","qFinal":[[1,2]],"truth_pose":"secret"}'
    assert NEW.select_prediction_fields(line,{'id','method','qFinal'})=={'id':'new:1','method':'N3_SUBPIX','qFinal':[[1,2]]}


def test_prediction_only_Q_native_center_and_state_gate_package():
    r,p,g=sample();out=NEW.predictor_record(r,p,g)
    assert out['frame']['raw_points']==p['qFinal'] and out['frame']['raw_points'][8]==p['qFinal'][8]
    assert out['frame']['initial_pose']==p['actual_pose'] and out['original_source']==out['frame']
    assert np.allclose(out['frame']['physical_registration_Q'],np.eye(3))
    for k,s in enumerate(out['initial_geometry']['initial_self_hidden']):
        if s:assert out['raw_points_initial_S_gated'][k]==[-1.,-1.]
    assert out['raw_points_initial_S_gated'][8]==p['qFinal'][8]
    assert not any(key in json.dumps(out['frame']) for key in ('human_state','truth','annotation','GT_order'))
    changed=deepcopy(p);changed['actual_pose']['R_physical']=np.diag([-1.,1.,-1.]).tolist()
    with pytest.raises(ValueError,match='Q differs'):NEW.predictor_record(r,changed,g)
