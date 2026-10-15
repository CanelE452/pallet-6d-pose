"""Pure REST operational/source-closure wiring, no real RGB/models or labels."""
from copy import deepcopy
import pickle
import pytest
from . import observe_geometry_guided_exact_REST_v4_29d as R


def test_source_input_merge_exact_and_conflicts_fail_without_mutation():
    first={'old':'a'};second={'old':'a','new':'b'};before=deepcopy((first,second))
    assert R._merge(first,second)==second and(first,second)==before
    with pytest.raises(ValueError,match='conflict'):R._merge(first,{'old':'changed'})


def test_private_partition_loop_and_picklable_workers_do_not_patch_parent_globals():
    parent=R.B.observe;before={k:id(v)for k,v in parent.__globals__.items()}
    observe=R.make_partition_observer()
    assert observe.__code__ is parent.__code__
    assert observe.__globals__ is not parent.__globals__
    assert observe.__globals__['V'] is not R.V
    assert observe.__globals__['V']._init_worker is R._init_worker
    assert observe.__globals__['V'].observe_one is R._observe_one
    assert pickle.loads(pickle.dumps(R._observe_one)) is R._observe_one
    assert before=={k:id(v)for k,v in parent.__globals__.items()}


def test_original_initializer_precedes_private_model_capture_and_worker_is_exact(monkeypatch):
    seen=[];monkeypatch.setattr(R,'_WORKER',None)
    with pytest.raises(RuntimeError,match='initializer'):R._observe_one({'id':'fake'})
    monkeypatch.setattr(R.V,'_init_worker',lambda:seen.append('original_initializer'))
    def factory():
        assert seen==['original_initializer'];seen.append('private_capture')
        return lambda task:task
    monkeypatch.setattr(R.A,'make_observe_one',factory)
    R._init_worker();task={'id':'same','states':['S']*8}
    assert R._observe_one(task) is task and seen==['original_initializer','private_capture']


def test_actual_REST_gate_precedes_all_IO(tmp_path):
    with pytest.raises(ValueError,match='Separate root'):R.observe(tmp_path)
    assert not list(tmp_path.iterdir())


def test_both_actual_graph_cases_must_pass_before_operational_use(monkeypatch,tmp_path):
    documents={'SEAL.json':{'result_sha256':'fixed','plan_sha256':'fixed'},
        'RESULT.json':{'failure_count':1,'cases':[{'id':'a','exact_original_optimized_stored':True},
            {'id':'wood','exact_original_optimized_stored':False}]},'PLAN.json':{'cases':[{'id':'a'},{'id':'wood'}]}}
    monkeypatch.setattr(R,'PROOF',tmp_path);monkeypatch.setattr(R.C,'read',lambda path:documents[path.name])
    monkeypatch.setattr(R.C,'sha',lambda path:'fixed')
    with pytest.raises(ValueError,match='Both fixed CAL'):R._proof()


def test_final_closure_maps_identical_and_no_reverse_completion_cycle(monkeypatch,tmp_path):
    monkeypatch.setattr(R,'OUTPUT',tmp_path);monkeypatch.setattr(R,'_verify',lambda mapping:None)
    monkeypatch.setattr(R.C,'ROOT',tmp_path)
    maps=dict(source_sha256={'newsource.py':'hash'},input_sha256={'input':'hash'},
        protected_user_sha256={'protected':'hash'},original_immutable_file_sha256={'CAL46':'hash'})
    execution=dict(schema='EXACT_GRAPH_OPERATIONAL_REST_EXECUTION_SEAL_V29D',rows=192,**maps,
        preparation_seal={'path':'prep','sha256':'p','bytes':1},actual_replay_proof={'path':'proof','sha256':'q','bytes':1},
        REST_observation_seal={'path':'REST','sha256':'r','bytes':1})
    R.C.write(tmp_path/R.EXECUTION_SEAL,execution);before=(tmp_path/R.EXECUTION_SEAL).read_bytes()
    def parent_finalize(output):
        R.C.write(output/'OBSERVATION_SEAL.json',dict(rows=238,source_sha256={'old':'h'},
            protocol_sha256='unchanged_original_protocol',observations_sha256='unchanged_all238_journal'))
    monkeypatch.setattr(R.B,'finalize',parent_finalize)
    R.finalize(tmp_path)
    completed=R.C.read(tmp_path/R.COMPLETION);seal=R.C.read(tmp_path/'OBSERVATION_SEAL.json')
    for key,value in maps.items():assert completed[key]==value
    assert completed['final_rows']==238 and completed['rows']==192
    assert seal['source_sha256']==maps['source_sha256']
    assert seal['protocol_sha256']=='unchanged_original_protocol'
    assert seal['observations_sha256']=='unchanged_all238_journal'
    assert seal['operational_execution_seal']['sha256']==R.C.sha(tmp_path/R.EXECUTION_SEAL)
    assert completed['final_all238_observation_seal']['sha256']==R.C.sha(tmp_path/'OBSERVATION_SEAL.json')
    assert 'completion_supplement'not in seal and before==(tmp_path/R.EXECUTION_SEAL).read_bytes()
