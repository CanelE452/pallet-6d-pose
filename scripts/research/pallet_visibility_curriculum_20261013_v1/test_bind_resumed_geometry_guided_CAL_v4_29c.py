"""Synthetic byte/source closure only; groups/journals deliberately undecodable."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from . import bind_resumed_geometry_guided_CAL_v4_29c as W


def fixture(root,monkeypatch):
    obs=root/'obs';obs.mkdir();(obs/'row_spool').mkdir()
    source=root/'resume_source.py';source.write_text('synthetic source bytes\n')
    protected=root/'protected.txt';protected.write_text('unchanged')
    own=root/'binding_source.py';own.write_bytes(Path(W.__file__).read_bytes());monkeypatch.setattr(W,'__file__',str(own))
    sources={str(source.relative_to(root)):W._sha(source)}
    ids=[f'synthetic:{i}'for i in range(46)]
    entries={}
    for ident in ids:
        name=W.hashlib.sha256(ident.encode()).hexdigest()+'.jsonl.gz';path=obs/'row_spool'/name
        path.write_bytes(b'not decoded observation row '+ident.encode())
        entries[ident]=dict(id=ident,partition='CAL',sha256=W._sha(path),canonical_row_sha256='a'*64)
    old={i:entries[i]for i in ids[:32]};new={i:entries[i]for i in ids[32:]}
    protocol=dict(source_sha256=sources,input_sha256={},protected_user_sha256={'protected.txt':W._sha(protected)})
    W._write(obs/'PROTOCOL.json',protocol)
    W._write(obs/'PREPARATION_SEAL.json',dict(rows=238,ordered_ids=ids+[f'REST:{i}'for i in range(192)],protocol_sha256=W._sha(obs/'PROTOCOL.json')))
    W._write(obs/'CAL_EXECUTION_PROTOCOL.json',dict(preboot=True))
    (obs/'CAL_OBSERVATIONS.jsonl.gz').write_bytes(b'not decoded CAL46 payload')
    W._write(obs/'CAL_SUMMARY.json',dict(rows=46));W._write(obs/'CAL_ROW_INDEX.json',dict(entries=list(entries.values())))
    restart=root/'RESTART.json';W._write(restart,dict(verified=True))
    plan=root/'PLAN.json';W._write(plan,dict(source_sha256=sources,restart_receipt=W._binding(restart,root)))
    immutable={str((obs/'row_spool'/W.hashlib.sha256(ident.encode()).hexdigest()).relative_to(root))+'.jsonl.gz':entry['sha256']for ident,entry in old.items()}
    for name in ('PROTOCOL.json','PREPARATION_SEAL.json','CAL_EXECUTION_PROTOCOL.json'):
        immutable[str((obs/name).relative_to(root))]=W._sha(obs/name)
    pre=obs/'CAL_RESUME_PREFLIGHT_SEAL_V1.json'
    W._write(pre,dict(source_sha256=sources,immutable_file_sha256=immutable,exact_existing_entries=old,
        missing_CAL_ids=ids[32:],plan=W._binding(plan,root),restart_receipt=W._binding(restart,root)))
    execution=obs/'CAL_RESUME_EXECUTION_PROTOCOL_V1.json'
    W._write(execution,dict(source_sha256=sources,missing_CAL_ids=ids[32:],pixel_NMS_TLS_guides_graph_or_guard_changes=0,
        plan=W._binding(plan,root),preflight_seal=W._binding(pre,root)))
    sealed=obs/'CAL_RESUME_EXECUTION_SEAL_V1.json'
    W._write(sealed,dict(rows=14,original_rows_reused=32,final_CAL_rows=46,new_row_entries=new,
        original32_immutable=True,same_observe_one_NMS_TLS_guides_graph=True,source_sha256=sources,
        execution_protocol=W._binding(execution,root),preflight=W._binding(pre,root),plan=W._binding(plan,root)))
    cal=obs/'CAL_OBSERVATION_SEAL.json'
    W._write(cal,dict(rows=46,ordered_ids=ids,protocol_sha256=W._sha(obs/'PROTOCOL.json'),
        preparation_seal_sha256=W._sha(obs/'PREPARATION_SEAL.json'),execution_protocol_sha256=W._sha(obs/'CAL_EXECUTION_PROTOCOL.json'),
        observations_sha256=W._sha(obs/'CAL_OBSERVATIONS.jsonl.gz'),summary_sha256=W._sha(obs/'CAL_SUMMARY.json'),
        row_index_sha256=W._sha(obs/'CAL_ROW_INDEX.json'),resume_source_sha256=sources,
        resume_execution_seal=W._binding(sealed,root),resume_preflight_seal=W._binding(pre,root),
        complete_CAL46_includes_exact32_preboot_plus14_resumed=True))
    supp=obs/'CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json'
    W._write(supp,dict(rows=46,exact_preboot_rows=32,new_resumed_rows=14,source_sha256=sources,
        no_old_observation_remeasurement=True,producer_math_changes=0,resume_execution_seal=W._binding(sealed,root),
        CAL_observation_seal=W._binding(cal,root),original_protocol=W._binding(obs/'PROTOCOL.json',root),
        original_preparation=W._binding(obs/'PREPARATION_SEAL.json',root),original_execution=W._binding(obs/'CAL_EXECUTION_PROTOCOL.json',root)))
    canonical=root/'canonical';canonical.mkdir()
    (canonical/'CAL_PREPARED_GROUPS.jsonl.gz').write_bytes(b'not decoded selected groups or tangent axes')
    W._write(canonical/'CAL_PREFLIGHT_V1.json',dict(error_count=0,successful_metadata_rows=46))
    W._write(canonical/'PREPARATION_SEAL.json',dict(ids=ids,source_sha256=sources,
        input_sha256={str(cal.relative_to(root)):W._sha(cal)},protected_user_sha256=protocol['protected_user_sha256'],
        before_CAL_reference_decode=True))
    W._write(canonical/'CAL_PREFLIGHT_SEAL_V1.json',dict(preparation_sha256=W._sha(canonical/'PREPARATION_SEAL.json'),
        report_sha256=W._sha(canonical/'CAL_PREFLIGHT_V1.json'),prepared_groups_sha256=W._sha(canonical/'CAL_PREPARED_GROUPS.jsonl.gz')))
    return obs,canonical


def test_metadata_GO_gate_precedes_any_io(monkeypatch):
    monkeypatch.setattr(W,'_read',lambda path:pytest.fail('IO before GO'))
    with pytest.raises(PermissionError):W.prepare_bound_metadata()


def test_exact32_plus14_sources_and_opaque_payloads_propagate_to_new_prepare_union(tmp_path,monkeypatch):
    obs,canonical=fixture(tmp_path,monkeypatch);before={str(p):W._sha(p)for d in (obs,canonical)for p in d.rglob('*')if p.is_file()}
    output=tmp_path/'bound';result=W.prepare_bound_metadata(canonical,output,obs,root=tmp_path,root_metadata_GO=True)
    assert result['sha256']==W._sha(output/'CAL_PREFLIGHT_SEAL_V1.json')
    new=W._read(output/'PREPARATION_SEAL.json');receipt=W._read(output/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json')
    assert receipt['original_rows']==32 and receipt['new_rows']==14 and receipt['journal_rows_or_numeric_references_decoded']==0
    for name in ('CAL_PREPARED_GROUPS.jsonl.gz','CAL_PREFLIGHT_V1.json'):
        assert (output/name).read_bytes()==(canonical/name).read_bytes()
    assert 'resume_source.py'in new['source_sha256']and 'binding_source.py'in new['source_sha256']
    for name in ('CAL_RESUME_EXECUTION_PROTOCOL_V1.json','CAL_RESUME_EXECUTION_SEAL_V1.json','CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json'):
        assert str((obs/name).relative_to(tmp_path))in new['input_sha256']
    assert str((output/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json').relative_to(tmp_path))in new['input_sha256']
    for mapping in ('source_sha256','input_sha256','protected_user_sha256'):W._verify_map(new[mapping],tmp_path)
    assert before=={str(p):W._sha(p)for d in (obs,canonical)for p in d.rglob('*')if p.is_file()}
    with pytest.raises(FileExistsError):W.prepare_bound_metadata(canonical,output,obs,root=tmp_path,root_metadata_GO=True)


@pytest.mark.parametrize('which',['source','old_spool','new_spool','CAL_payload'])
def test_modified_source_old_new_spool_or_payload_rejected_before_any_copy(tmp_path,monkeypatch,which):
    obs,canonical=fixture(tmp_path,monkeypatch)
    paths={'source':tmp_path/'resume_source.py','old_spool':obs/'row_spool'/(W.hashlib.sha256(b'synthetic:0').hexdigest()+'.jsonl.gz'),
        'new_spool':obs/'row_spool'/(W.hashlib.sha256(b'synthetic:45').hexdigest()+'.jsonl.gz'),'CAL_payload':obs/'CAL_OBSERVATIONS.jsonl.gz'}
    paths[which].write_bytes(b'changed')
    with pytest.raises(ValueError):W.prepare_bound_metadata(canonical,tmp_path/'bound',obs,root=tmp_path,root_metadata_GO=True)
    assert not(tmp_path/'bound').exists()


def test_missing14_overlap_rejected_even_when_resume_seal_rebound(tmp_path,monkeypatch):
    obs,_=fixture(tmp_path,monkeypatch);p=obs/'CAL_RESUME_PREFLIGHT_SEAL_V1.json';pre=W._read(p)
    pre['missing_CAL_ids'][0]='synthetic:0';W._write(p,pre)
    with pytest.raises(ValueError,match='disjoint'):W.verify_resume_closure(obs,root=tmp_path)


def test_source_union_conflict_rejected_before_copy(tmp_path,monkeypatch):
    obs,canonical=fixture(tmp_path,monkeypatch);prep=W._read(canonical/'PREPARATION_SEAL.json')
    with pytest.raises(ValueError,match='ancestry conflict'):
        W._union(prep['source_sha256'],{'resume_source.py':'0'*64})
