from copy import deepcopy
import gzip
import hashlib
import io
import json

import pytest

from . import archive_compact_largest_component_global_outline_poses_v4_33 as A
from .test_archive_compact_clean_fresh_BI_poses_v4_32_r2_v1 import row


def test_original_compact_and_HashReader_are_exact_direct_aliases():
    assert A.compact is A.OLD.compact and A.HashReader is A.OLD.HashReader
    assert A.SCALARS is A.OLD.SCALARS and A.ARMS is A.OLD.ARMS
    assert A.C.sha(A.OLD.__file__)==A.OLD_SHA256
    assert A.SOURCE!=A.OLD.SOURCE and A.OUTPUT!=A.OLD.OUTPUT and A.PLAN!=A.OLD.PLAN


def test_poses_native_t_center_topWD_E_scalars_and_status_copied_without_transform():
    value=row();before=deepcopy(value)
    value['arms'][A.ARMS[0]]['WD_family']='ORIGINAL_WD'
    value['arms'][A.ARMS[1]].update(accepted=False,pose_available=False,actual_pose=None)
    before=deepcopy(value);result=A.compact(value)
    assert value==before and result['full_costplan_seed_start_or_trajectory_archive']is False
    for arm in A.ARMS:
        assert result['arms'][arm]['actual_pose']==value['arms'][arm]['actual_pose']
        assert result['arms'][arm]['WD_family']==value['arms'][arm]['WD_family']
        assert result['arms'][arm]['selected_scalar']=={k:value['arms'][arm]['solution'][k]for k in A.SCALARS if k in value['arms'][arm]['solution']}
        assert 'full_costplan'not in result['arms'][arm]
    result['arms'][A.ARMS[0]]['actual_pose']['native_t'][0]=123.
    assert value==before


def test_rootGO_and_all_explicit_closure_SHAs_before_IO(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No actual IO')
    monkeypatch.setattr(A.C,'read',forbidden);monkeypatch.setattr(A.C,'sha',forbidden)
    with pytest.raises(PermissionError):A.declare()
    with pytest.raises(PermissionError):A.run()
    for missing in ('prediction','score','audit'):
        kwargs={f'expected_{name}_sha256':'a'*64 for name in ('prediction','score','audit')}
        kwargs[f'expected_{missing}_sha256']=None
        with pytest.raises(ValueError,match='before IO'):A.declare(root_metadata_GO=True,**kwargs)


def test_same_hashreader_covers_two_gzip_members_and_unicode():
    blob=gzip.compress('한글\n'.encode())+gzip.compress(b'next\n')
    reader=A.HashReader(io.BytesIO(blob))
    with gzip.GzipFile(fileobj=reader,mode='rb')as stream:assert stream.read()=='한글\nnext\n'.encode()
    assert reader.digest.hexdigest()==hashlib.sha256(blob).hexdigest()and reader.byte_count==len(blob)


def test_postscore_closure_directly_joins_prediction_protocol_scoring_and_central(monkeypatch):
    ids=[f'toy:{i}'for i in range(238)]
    documents={A.SOURCE/'PREDICTION_SEAL.json':dict(rows=238,ordered_ids=ids,arms=list(A.ARMS),arm_count=5,
        all_declared_arms_complete=True,no_population_drop=True,outline_selection_profile=A.PROFILE,protocol_sha256='d'*64),
        A.SOURCE/'SCORE_SEAL.json':dict(prediction_seal=dict(sha256='a'*64),all_rows_kept=True,
            scoring_protocol_sha256='e'*64,outline_selection_profile=A.PROFILE,native_CAL32_models_relabelled=False),
        A.SOURCE/'PROTOCOL.json':dict(central_plan_sha256=A.CENTRAL_SHA256),
        A.SOURCE/'SCORING_PROTOCOL.json':dict(central_plan_sha256=A.CENTRAL_SHA256,prediction_seal_sha256='a'*64),
        A.SOURCE/'independent_root_audit_v1/AUDIT.json':dict(rows=238,failure_count=0,failures=[],prediction_seal=dict(sha256='a'*64)),
        A.SOURCE/'independent_root_audit_v1/AUDIT_SEAL.json':dict(receipt_sha256='c'*64)}
    hashes={A.SOURCE/'PREDICTION_SEAL.json':'a'*64,A.SOURCE/'SCORE_SEAL.json':'b'*64,
        A.SOURCE/'independent_root_audit_v1/AUDIT.json':'c'*64,A.CENTRAL:A.CENTRAL_SHA256,
        A.SOURCE/'PROTOCOL.json':'d'*64,A.SOURCE/'SCORING_PROTOCOL.json':'e'*64}
    monkeypatch.setattr(A.C,'sha',lambda path:hashes.get(path,A.OLD_SHA256))
    monkeypatch.setattr(A.C,'read',lambda path:deepcopy(documents[path]))
    assert A._closed('a'*64,'b'*64,'c'*64)['ordered_ids']==ids
    documents[A.SOURCE/'SCORING_PROTOCOL.json']['central_plan_sha256']='0'*64
    with pytest.raises(ValueError,match='closure'):A._closed('a'*64,'b'*64,'c'*64)


def prepare(tmp_path,monkeypatch,*,short=False):
    monkeypatch.setattr(A.C,'ROOT',tmp_path)
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(A,'SOURCE',source);monkeypatch.setattr(A,'OUTPUT',tmp_path/'archive')
    monkeypatch.setattr(A,'PLAN',tmp_path/'plan/PREDECLARATION.json');A.PLAN.parent.mkdir()
    ids=[f'SYNTHETIC:木{i}'for i in range(238)];values=[]
    for i,ident in enumerate(ids):
        value=row();value.update(id=ident,partition='CAL'if i<46 else'TEST'if i<95 else'REMAINDER');values.append(value)
    actual=values[:-1]if short else values
    split=80
    encoded=[(''.join(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'for value in group)).encode()for group in (actual[:split],actual[split:])]
    blob=b''.join(gzip.compress(data)for data in encoded)
    (source/'PREDICTIONS.jsonl.gz').write_bytes(blob)
    pred=dict(ordered_ids=ids,predictions_sha256=hashlib.sha256(blob).hexdigest())
    plan=dict(source_sha256={},original_prediction_seal_sha256='a'*64,score_seal_sha256='b'*64,
        independent_audit_sha256='c'*64,ordered_ids=ids,original_prediction_journal_sha256=pred['predictions_sha256'])
    A.C.write(A.PLAN,plan);A.C.write(A.PLAN.with_name('PLAN_SEAL.json'),dict(plan_sha256=A.C.sha(A.PLAN)))
    calls=[];monkeypatch.setattr(A,'_closed',lambda *args:calls.append(args)or pred)
    return values,blob,calls


def test_toy_all238x5_onepass_compact_no_input_E_or_observation_reads(tmp_path,monkeypatch):
    values,blob,calls=prepare(tmp_path,monkeypatch)
    seal=A.run(root_actual_archive_GO=True)
    assert seal==A.C.sha(A.OUTPUT/'SEAL.json')and len(calls)==2
    with gzip.open(A.OUTPUT/'POSES.jsonl.gz','rt',encoding='utf-8')as stream:actual=[json.loads(line)for line in stream]
    assert actual==[A.compact(value)for value in values]
    summary=A.C.read(A.OUTPUT/'SUMMARY.json')
    assert summary['rows']==238 and summary['cohorts']==dict(CAL=46,TEST=49,REMAINDER=143)
    assert summary['decode_passes']==1 and summary['original_compressed_bytes_read']==len(blob)
    assert all(info['rows']==238 for info in summary['arms'].values())
    assert A.C.read(A.OUTPUT/'SEAL.json')['arm_rows']==1190
    with pytest.raises(FileExistsError):A.run(root_actual_archive_GO=True)


def test_short_population_failure_preserves_partial_attempt_without_success_seal(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch,short=True)
    with pytest.raises(ValueError,match='count changed'):A.run(root_actual_archive_GO=True)
    assert A.C.read(A.OUTPUT/'FAILURE_V1.json')['rows_completed']==237
    assert not(A.OUTPUT/'SEAL.json').exists()
