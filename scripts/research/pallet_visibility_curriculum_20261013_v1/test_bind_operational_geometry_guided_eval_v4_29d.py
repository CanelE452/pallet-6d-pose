"""Synthetic, opaque input bytes prove provenance binding without data decode."""
from copy import deepcopy
from pathlib import Path

import pytest

from . import bind_operational_geometry_guided_eval_v4_29d as X


def fixture(root,monkeypatch):
    W=X.W;obs=root/'obs';obs.mkdir();unbound=root/'unbound';unbound.mkdir()
    own=root/'binding.py';own.write_bytes(Path(X.__file__).read_bytes());monkeypatch.setattr(X,'__file__',str(own))
    helper=root/'cal_binding.py';helper.write_bytes(Path(W.__file__).read_bytes());monkeypatch.setattr(W,'__file__',str(helper))
    (root/'test_bind_operational_geometry_guided_eval_v4_29d.py').write_text('synthetic test source\n')
    controller=root/X.CONTROLLER;controller.parent.mkdir(parents=True);controller.write_bytes(b'frozen controller fixture')
    monkeypatch.setattr(X,'CONTROLLER_SHA256',W._sha(controller))
    extra=root/'operational_source.py';extra.write_bytes(b'new private namespace only')
    protected=root/'protected.txt';protected.write_bytes(b'preserved user bytes')
    base_source={X.CONTROLLER:W._sha(controller)};sources={**base_source,'operational_source.py':W._sha(extra)}
    protected_map={'protected.txt':W._sha(protected)};ids=[f'CAL:{i}'for i in range(46)]+[f'REST:{i}'for i in range(192)]
    W._write(obs/'PROTOCOL.json',dict(source_sha256=base_source,input_sha256={},protected_user_sha256=protected_map))
    W._write(obs/'PREPARATION_SEAL.json',dict(ordered_ids=ids,protocol_sha256=W._sha(obs/'PROTOCOL.json')))
    (obs/'INPUTS.jsonl.gz').write_bytes(b'not decoded original task bytes')
    (obs/'CAL_OBSERVATIONS.jsonl.gz').write_bytes(b'not decoded CAL46 observation bytes')
    W._write(obs/'CAL_OBSERVATION_SEAL.json',dict(rows=46,ordered_ids=ids[:46]))
    (obs/'REST_OBSERVATIONS.jsonl.gz').write_bytes(b'not decoded REST192 observations')
    W._write(obs/'REST_OBSERVATION_SEAL.json',dict(rows=192,ordered_ids=ids[46:],
        observations_sha256=W._sha(obs/'REST_OBSERVATIONS.jsonl.gz')))
    W._write(obs/'REST_OPERATIONAL_PREPARATION_SEAL_V1.json',dict(schema='EXACT_GRAPH_OPERATIONAL_REST_PREPARATION_SEAL_V29D'))
    replay=root/'replay';replay.mkdir()
    methods=[dict(method=m,canonical_graph_sha256='a'*64,stored_graph_sha256='a'*64,
        exact_full_graph_vs_stored=True,source_bytes_unchanged=True,producer_globals_unchanged=True,
        new_RGB_NMS_TLS_NN_reference_posefit_or_startsearch_calls=0)for m in ('ORIGINAL','BOUNDED_HEAP_FRESH_MEMO')]
    W._write(replay/'RESULT.json',dict(failure_count=0,cases=[dict(exact_original_optimized_stored=True,
        methods=deepcopy(methods))for _ in range(2)]))
    W._write(replay/'SEAL.json',dict(result_sha256=W._sha(replay/'RESULT.json')))
    immutable={W._relative(p,root):W._sha(p) for p in (obs/'PROTOCOL.json',obs/'PREPARATION_SEAL.json',
        obs/'INPUTS.jsonl.gz',obs/'CAL_OBSERVATION_SEAL.json',obs/'CAL_OBSERVATIONS.jsonl.gz')}
    input_map={W._relative(p,root):W._sha(p)for p in (replay/'RESULT.json',replay/'SEAL.json')}
    common=dict(rows=192,source_sha256=sources,input_sha256=input_map,protected_user_sha256=protected_map,
        original_immutable_file_sha256=immutable,actual_replay_proof=W._binding(replay/'SEAL.json',root),
        preparation_seal=W._binding(obs/'REST_OPERATIONAL_PREPARATION_SEAL_V1.json',root),
        REST_observation_seal=W._binding(obs/'REST_OBSERVATION_SEAL.json',root))
    execution=obs/'REST_OPERATIONAL_EXECUTION_SEAL_V1.json'
    W._write(execution,dict(**common,schema=X.EXEC_SCHEMA))
    (obs/'OBSERVATIONS.jsonl.gz').write_bytes(b'not decoded final all238 observations')
    final=obs/'OBSERVATION_SEAL.json'
    W._write(final,dict(rows=238,ordered_ids=ids,source_sha256=sources,
        protocol_sha256=W._sha(obs/'PROTOCOL.json'),observations_sha256=W._sha(obs/'OBSERVATIONS.jsonl.gz'),
        preparation_seal_sha256=W._sha(obs/'PREPARATION_SEAL.json'),
        partition_seals={p:W._sha(obs/(p+'_OBSERVATION_SEAL.json'))for p in ('CAL','REST')},
        operational_execution_seal=W._binding(execution,root)))
    W._write(obs/'REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json',dict(**common,schema=X.COMPLETE_SCHEMA,
        final_rows=238,execution_seal=W._binding(execution,root),final_all238_observation_seal=W._binding(final,root)))
    protocol=dict(ids=ids,arms=deepcopy(X.ARMS),source_sha256=base_source,input_sha256={},
        protected_user_sha256=protected_map,full_pose_TP_topnormal_prior_weights=0.,
        old_e_numeric_pose_added_as_prior_or_fallback=False,
        invariant_inference_settings=dict(q_covariance='exact',S='model only',U='regions only',noise='unchanged'))
    W._write(unbound/'PROTOCOL.json',protocol)
    (unbound/'INPUTS.jsonl.gz').write_bytes(b'not decoded joined v29 input bytes')
    reseal(unbound)
    return obs,unbound


def reseal(folder):
    W=X.W;protocol=W._read(folder/'PROTOCOL.json')
    W._write(folder/'INPUT_SEAL.json',dict(rows=238,ordered_ids=protocol['ids'],
        protocol_sha256=W._sha(folder/'PROTOCOL.json'),input_sha256=W._sha(folder/'INPUTS.jsonl.gz')))


def test_actual_controller_source_pin_is_current():
    assert X.W._sha(X.ROOT/X.CONTROLLER)==X.CONTROLLER_SHA256


def test_GO_precedes_all_metadata_io(monkeypatch):
    monkeypatch.setattr(X.W,'_read',lambda p:pytest.fail('IO before GO'))
    with pytest.raises(PermissionError):X.prepare_bound_evaluation()


def test_exact_bytes_and_inference_settings_with_additive_complete_ancestry(tmp_path,monkeypatch):
    obs,unbound=fixture(tmp_path,monkeypatch);W=X.W
    before={str(p):W._sha(p)for folder in(obs,unbound)for p in folder.rglob('*')if p.is_file()}
    original=W._read(unbound/'PROTOCOL.json');output=tmp_path/'canonical'
    binding=X.prepare_bound_evaluation(unbound,output,obs,root=tmp_path,root_prepare_GO=True)
    assert binding['sha256']==W._sha(output/'INPUT_SEAL.json')
    protocol=W._read(output/'PROTOCOL.json');seal=W._read(output/'INPUT_SEAL.json')
    assert (output/'INPUTS.jsonl.gz').read_bytes()==(unbound/'INPUTS.jsonl.gz').read_bytes()
    for name in original:
        if name not in ('source_sha256','input_sha256'):assert protocol[name]==original[name]
    assert seal['ordered_ids']==original['ids'] and seal['rows']==238
    assert seal['protocol_sha256']==W._sha(output/'PROTOCOL.json')
    assert 'operational_source.py'in protocol['source_sha256']
    assert W._relative(obs/'REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json',tmp_path)in protocol['input_sha256']
    assert W._relative(unbound/'INPUT_SEAL.json',tmp_path)in protocol['input_sha256']
    for name in ('source_sha256','input_sha256','protected_user_sha256'):W._verify_map(protocol[name],tmp_path)
    assert before=={str(p):W._sha(p)for folder in(obs,unbound)for p in folder.rglob('*')if p.is_file()}
    with pytest.raises(FileExistsError):X.prepare_bound_evaluation(unbound,output,obs,root=tmp_path,root_prepare_GO=True)


@pytest.mark.parametrize('path',['operational_source.py','obs/REST_OBSERVATIONS.jsonl.gz',
    'obs/OBSERVATIONS.jsonl.gz','obs/CAL_OBSERVATIONS.jsonl.gz','replay/RESULT.json'])
def test_source_payload_immutable_CAL_or_replay_corruption_stops_before_output(tmp_path,monkeypatch,path):
    obs,unbound=fixture(tmp_path,monkeypatch);(tmp_path/path).write_bytes(b'changed')
    with pytest.raises((ValueError,KeyError)):
        X.prepare_bound_evaluation(unbound,tmp_path/'canonical',obs,root=tmp_path,root_prepare_GO=True)
    assert not(tmp_path/'canonical').exists()


def test_predictions_before_binding_rejected(tmp_path,monkeypatch):
    obs,unbound=fixture(tmp_path,monkeypatch);(unbound/'frames').mkdir()
    with pytest.raises(ValueError,match='before any predictions'):
        X.prepare_bound_evaluation(unbound,tmp_path/'canonical',obs,root=tmp_path,root_prepare_GO=True)


def test_changed_arms_rejected_even_with_consistent_unbound_seal(tmp_path,monkeypatch):
    obs,unbound=fixture(tmp_path,monkeypatch);p=unbound/'PROTOCOL.json';d=X.W._read(p)
    d['arms'].pop();X.W._write(p,d);reseal(unbound)
    with pytest.raises(ValueError,match='frozen five-arm'):
        X.prepare_bound_evaluation(unbound,tmp_path/'canonical',obs,root=tmp_path,root_prepare_GO=True)


def test_completion_execution_map_disagreement_rejected(tmp_path,monkeypatch):
    obs,_=fixture(tmp_path,monkeypatch);p=obs/'REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json';d=X.W._read(p)
    d['source_sha256']['operational_source.py']='0'*64;X.W._write(p,d)
    with pytest.raises(ValueError,match='map conflict'):X.verify_operational_closure(obs,root=tmp_path)


@pytest.mark.parametrize('kind',['summary','method_record'])
def test_failed_actual_replay_cannot_be_relabelled_as_success(tmp_path,monkeypatch,kind):
    obs,_=fixture(tmp_path,monkeypatch);W=X.W;p=tmp_path/'replay/RESULT.json';d=W._read(p)
    if kind=='summary':d['cases'][1]['exact_original_optimized_stored']=False
    else:d['cases'][1]['methods'][1]['canonical_graph_sha256']='b'*64
    W._write(p,d)
    # This tests the content gate as well as hashes: rebind all proof maps.
    W._write(tmp_path/'replay/SEAL.json',dict(result_sha256=W._sha(p)))
    for name in ('REST_OPERATIONAL_EXECUTION_SEAL_V1.json','REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json'):
        q=obs/name;record=W._read(q);record['actual_replay_proof']=W._binding(tmp_path/'replay/SEAL.json',tmp_path)
        record['input_sha256']={W._relative(f,tmp_path):W._sha(f)for f in(p,tmp_path/'replay/SEAL.json')};W._write(q,record)
    execution=obs/'REST_OPERATIONAL_EXECUTION_SEAL_V1.json';final=obs/'OBSERVATION_SEAL.json'
    record=W._read(final);record['operational_execution_seal']=W._binding(execution,tmp_path);W._write(final,record)
    q=obs/'REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json';record=W._read(q)
    record['execution_seal']=W._binding(execution,tmp_path);record['final_all238_observation_seal']=W._binding(final,tmp_path);W._write(q,record)
    with pytest.raises(ValueError,match='Both complete actual CAL'):X.verify_operational_closure(obs,root=tmp_path)
