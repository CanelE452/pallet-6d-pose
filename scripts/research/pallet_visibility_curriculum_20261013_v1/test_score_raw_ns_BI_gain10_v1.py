"""Toy score gating/denominator/legacy metric tests; no real reference IO."""
import gzip
import json

import numpy as np
import pytest

from . import score_raw_ns_BI_gain10_v1 as S


def test_no_GO_and_invalid_seal_stop_before_any_IO(monkeypatch):
    def no(*args,**kwargs):raise AssertionError('No IO expected')
    monkeypatch.setattr(S.C,'sha',no)
    with pytest.raises(PermissionError):S.score()
    with pytest.raises(ValueError,match='before IO'):S.score(root_scoring_GO=True,expected_prediction_seal_sha256='bad')


def _fixture(tmp_path,monkeypatch):
    inp=tmp_path/'input';inp.mkdir();out=tmp_path/'output';out.mkdir()
    monkeypatch.setattr(S,'INPUT',inp)
    ids=['toy:'+str(i)for i in range(10)]
    (inp/'INPUTS.json').write_text('{}')
    (inp/'INPUT_SEAL.json').write_text(json.dumps(dict(ordered_ids=ids,inputs_sha256=S.C.sha(inp/'INPUTS.json'))))
    ih=S.C.sha(inp/'INPUTS.json');sh=S.C.sha(inp/'INPUT_SEAL.json')
    monkeypatch.setattr(S,'INPUT_SHA256',ih);monkeypatch.setattr(S,'INPUT_SEAL_SHA256',sh)
    protocol=dict(inputs_sha256=ih,input_seal_sha256=sh,source_sha256={},input_sha256={})
    (out/'PROTOCOL.json').write_text(json.dumps(protocol))
    records=[dict(id=fid,predictions={arm:dict(actual_pose=dict(available=False))for arm in S.ARMS})for fid in ids]
    def seal(rows=records):
        with gzip.open(out/'PREDICTIONS.jsonl.gz','wt')as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        doc=dict(rows=10,ordered_ids=ids,arms=list(S.ARMS),before_scoring=True,GT_inputs=False,
            protocol_sha256=S.C.sha(out/'PROTOCOL.json'),predictions_sha256=S.C.sha(out/'PREDICTIONS.jsonl.gz'))
        (out/'PREDICTION_SEAL.json').write_text(json.dumps(doc))
        return S.C.sha(out/'PREDICTION_SEAL.json')
    return out,records,seal


def test_all_ten_four_arm_predictions_gate_without_reference(tmp_path,monkeypatch):
    out,records,seal=_fixture(tmp_path,monkeypatch)
    digest=seal();_,_,actual=S.prediction_gate(out,digest)
    assert [r['id']for r in actual]==[r['id']for r in records]
    assert not(S.INPUT/'SCORING_REFERENCE_MANIFEST.json').exists()


@pytest.mark.parametrize('problem',['drop_row','drop_arm','duplicate_id','tamper_journal'])
def test_incomplete_or_changed_predictions_stop_before_GT(tmp_path,monkeypatch,problem):
    out,records,seal=_fixture(tmp_path,monkeypatch)
    if problem=='drop_row':records.pop()
    elif problem=='drop_arm':records[4]['predictions'].pop(S.ARMS[-1])
    elif problem=='duplicate_id':records[4]['id']=records[0]['id']
    digest=seal()
    if problem=='tamper_journal':(out/'PREDICTIONS.jsonl.gz').write_bytes(b'changed')
    with pytest.raises(ValueError):S.score(root_scoring_GO=True,folder=out,expected_prediction_seal_sha256=digest)
    assert not(out/'SCORE.json').exists()


def test_unavailable_poses_preserve_denominator_and_failed_success():
    rows=[dict(metrics={arm:dict(translation_cm=None,rotation_deg=None,success_5cm_5deg=False)for arm in S.ARMS})for _ in range(10)]
    for result in S.summary(rows).values():
        assert result['rows']==10 and result['missing_pose_count']==10 and result['joint_successes']==0
        assert result['mean_translation_cm']is None and result['max_translation_cm']is None


def test_unchanged_legacy_metric_has_strict_five_threshold_and_physical_symmetry():
    truth=dict(R=np.eye(3).tolist(),t=[0,0,2],xyz=[1.1,.1,1.3],order=2)
    pose=dict(R_physical=np.diag([-1,1,-1]).tolist(),centroid=[0,0,2])
    assert S.pose_metric(pose,truth)['success_5cm_5deg']
    pose['centroid']=[.05,0,2]
    metric=S.pose_metric(pose,truth)
    assert metric['translation_cm']==5 and not metric['success_5cm_5deg']
