"""Source/cap/identity boundaries without model forwards or reference IO."""
from copy import deepcopy
import gzip
from pathlib import Path

import numpy as np
import pytest

from scripts.research.pallet_single_component_bi_v1 import inputs as I


# In the isolated sparse worktree, data is a read-only original asset alias.
# In a regular clone this resolves to the same repository root.
SOURCE = (Path(__file__).resolve().parents[3] / 'data').resolve().parent


def fixture():
    q0 = np.array([[20.+8*k,35.]for k in range(8)]+[[60.,45.]])
    qn = q0.copy(); qn[:8,0] += .5
    raw_sub = q0.copy();raw_sub[:8,1] += 100.
    ns_sub = qn.copy();ns_sub[:8,1] += .2
    cap = I.cap_function(SOURCE)
    support = np.ones(9,bool)
    qs = cap(q0,raw_sub,120,90,support)
    qns = cap(q0,ns_sub,120,90,support)
    candidate = dict(score=.9,box_xyxy=[0.,0.,100.,80.],keypoints_conf=[.8]*9)
    meta = dict(id='toy:0',session='toy',image_sha256='a'*64,raw_hw=[90,120],K=np.eye(3).tolist(),
                selected_index=0,candidate_metadata=candidate,dimensions_whd_m=[1.1,.11,1.3],
                q0=q0.tolist(),qN=qn.tolist(),prediction_support=support.tolist())
    common = dict(keypoints_conf=candidate['keypoints_conf'],prediction_support=support.tolist(),
                  coarse_pose={'available':False})
    frame = {k:deepcopy(v)for k,v in meta.items()if k not in ['q0','qN','dimensions_whd_m']}
    frame['xyz']=meta['dimensions_whd_m'];frame['arms']={
        'RAW_YOLO':dict(deepcopy(common),native_points=q0.tolist()),
        'N3_SUBPIX':dict(deepcopy(common),native_points=qns.tolist())}
    records = {}
    for method,q in [('BASE',q0),('N3',qn),('SUBPIX',qs),('N3_SUBPIX',qns)]:
        row = dict(id=frame['id'],method=method,raw_hw=[90,120],native_points=q.tolist(),qFinal=q.tolist(),
            actual_pose=None,prediction_support=support.tolist(),fixed_metadata=dict(
                selected_index=0,candidate_metadata=deepcopy(candidate),preserved=True))
        if method=='SUBPIX':row['qS']=raw_sub.tolist()
        if method=='N3_SUBPIX':row['qS']=ns_sub.tolist()
        records[(frame['id'],method)] = row
    return frame,meta,records,cap


def test_exact_component_arrays_preserve_existing_initializers_and_do_not_mutate_inputs():
    frame,meta,records,cap=fixture();before=deepcopy((frame,meta,records))
    selected=I.validate_frame(frame,meta,records,cap)
    assert selected['N3_ONLY']==meta['qN']
    assert selected['SUBPIX_ONLY']==records[('toy:0','SUBPIX')]['native_points']
    assert selected['N3_ONLY']!=frame['arms']['N3_SUBPIX']['native_points']
    assert before==(frame,meta,records)


def test_cap_is_around_original_RAW_and_preserves_center_and_missing_support():
    cap=I.cap_function(SOURCE)
    q0=np.array([[20.+8*k,35.]for k in range(8)]+[[60.,45.]])
    proposed=q0.copy();proposed[:8,0]+=100.
    support=np.ones(9,bool);support[1]=False;q0[2]=[-1.,-1.]
    final=cap(q0,proposed,120,90,support)
    assert final[0,0]==q0[0,0]+1.5
    assert np.array_equal(final[[1,2,8]],q0[[1,2,8]])


@pytest.mark.parametrize('mutation', ['raw','combined','n3','center','support','metadata','dims','cap_origin'])
def test_foreign_or_redefined_component_source_fails_closed(mutation):
    frame,meta,records,cap=fixture()
    if mutation=='raw':frame['arms']['RAW_YOLO']['native_points'][0][0]+=1
    elif mutation=='combined':frame['arms']['N3_SUBPIX']['native_points'][0][0]+=1
    elif mutation=='n3':records[('toy:0','N3')]['native_points'][0][0]+=1
    elif mutation=='center':
        for field in ('native_points','qFinal'):records[('toy:0','N3')][field][8][0]+=1
        meta['qN'][8][0]+=1
    elif mutation=='support':records[('toy:0','SUBPIX')]['prediction_support'][0]=False
    elif mutation=='metadata':records[('toy:0','N3')]['fixed_metadata']['selected_index']=1
    elif mutation=='dims':frame['xyz']=[1.1,1.3,.11]
    elif mutation=='cap_origin':
        q0=np.array(meta['q0']);qn=np.array(meta['qN']);native=np.array(records[('toy:0','N3_SUBPIX')]['qS']);native[:8,1]+=50
        wrong=cap(qn,native,120,90,np.array(meta['prediction_support']))
        records[('toy:0','N3_SUBPIX')].update(qS=native.tolist(),native_points=wrong.tolist(),qFinal=wrong.tolist())
        frame['arms']['N3_SUBPIX']['native_points']=wrong.tolist()
    with pytest.raises(ValueError):I.validate_frame(frame,meta,records,cap)


def test_lexical_projection_never_decodes_forbidden_metric_or_GT_objects(tmp_path):
    parser=I.P.prediction_parser(SOURCE)
    # The actual source lexical scanner skips the opaque token rather than
    # passing it to json.loads; all decoded fields remain valid JSON.
    text='{"id":"toy:0","method":"N3","native_points":[[1,2]],"actual_pose":null,"metric":FORBIDDEN_OPAQUE,"GT":{"x":[99,100]}}'
    selected=parser(text,I.ALLOWED_ROW)
    assert selected==dict(id='toy:0',method='N3',native_points=[[1,2]],actual_pose=None)
    assert not ({'GT','metric','pose','corner','canonical_observed'} & I.ALLOWED_ROW)
    assert not ({'evaluation','population','dependencies'} & I.ALLOWED_PROTOCOL)


def test_historical_predictor_stream_rejects_duplicate_and_missing_method(tmp_path):
    parser=I.P.prediction_parser(SOURCE)
    path=tmp_path/'rows.gz'
    rows=['{"id":"toy:0","method":"'+m+'","native_points":[]}'for m in ('BASE','N3','SUBPIX','N3_SUBPIX')]
    with gzip.open(path,'wt')as stream:stream.write('\n'.join(rows)+'\n')
    assert len(I.predictor_rows(path,{'toy:0'},parser))==4
    with gzip.open(path,'wt')as stream:stream.write('\n'.join(rows+[rows[0]])+'\n')
    with pytest.raises(ValueError,match='Duplicate'):I.predictor_rows(path,{'toy:0'},parser)
    with gzip.open(path,'wt')as stream:stream.write('\n'.join(rows[:-1])+'\n')
    with pytest.raises(ValueError,match='All four'):I.predictor_rows(path,{'toy:0'},parser)
