"""Retry failure bookkeeping and additive closure; synthetic archives only."""
from concurrent.futures import Future
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest

from . import run_observed_bi_edge_REST_v4_30c as V
from .test_rerank_observed_bi_edge_REST_v4_30b import fixture


@pytest.mark.parametrize('function', [V.prepare,V.rerank,V.finalize])
def test_no_go_precedes_all_io(monkeypatch,function):
    monkeypatch.setattr(V.C,'read',lambda *a:pytest.fail('IO before root GO'))
    with pytest.raises(PermissionError):function()


def test_private_retry_edits_only_metadata_globals_and_names():
    before=dict(V.W.rerank.__globals__)
    private=V._resumed_rerank_function()
    assert private.__globals__['REST_EXEC']==V.EXEC
    assert private.__globals__['R'] is V.W.R
    assert all(V.W.rerank.__globals__[k] is value for k,value in before.items())
    assert V.W.REST_EXEC=='REST_EXECUTION_PROTOCOL_V30B.json'


def test_real_file_main_can_spawn_importable_pure_target(tmp_path):
    # Operational regression for the failed stdin main. This starts no dataset
    # worker: the already-existing common serializer is the pure spawn target.
    launcher=tmp_path/'spawn_fixture.py'
    launcher.write_text("from concurrent.futures import ProcessPoolExecutor\n"
        "from multiprocessing import get_context\n"
        "from scripts.research.pallet_visibility_curriculum_20261013_v1 import run_observed_bi_edge_REST_v4_30c as V\n"
        "if __name__ == '__main__':\n"
        "    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:\n"
        "        assert pool.submit(V.C.finite,{'id':'synthetic','value':3}).result(timeout=30)=={'id':'synthetic','value':3}\n"
        "    print('CLI_FILE_SPAWN_PASS')\n")
    result=subprocess.run([sys.executable,str(launcher)],cwd=V.C.ROOT,
        env={**os.environ,'PYTHONPATH':str(V.C.ROOT)},capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    assert 'CLI_FILE_SPAWN_PASS' in result.stdout


def test_failed_future_logs_identifiers_and_always_cleans_pool(tmp_path,monkeypatch):
    monkeypatch.setattr(V.C,'ROOT',tmp_path)
    cases=[dict(id='test:1'),dict(id='test:2')];clean=[]
    protocol=dict(parent_REST_entries=cases,configuration={})
    V.C.write(tmp_path/V.W.REST_PREP,dict(synthetic_preparation=True))
    monkeypatch.setattr(V,'_bound_load',lambda output:protocol)
    monkeypatch.setattr(V,'_closure_write',V.C.write)
    class Pool:
        def __init__(self,**kwargs):pass
        def submit(self,worker,*args):
            assert worker is V.R._worker
            f=Future()
            if args[0]['id']=='test:1':f.set_exception(RuntimeError('synthetic worker failure'))
            else:f.set_result(({},{}))
            return f
        def shutdown(self,**kwargs):clean.append(kwargs)
    def first(pending,**kwargs):return {next(iter(pending))},set()
    monkeypatch.setattr(V.W,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(V.W,'wait',first)
    # Frozen code globals point at the original namespace; replace those
    # symbols in the private copy, preserving actual frozen worker identity.
    function=V._resumed_rerank_function()
    function.__globals__.update(ProcessPoolExecutor=Pool,wait=first)
    with pytest.raises(RuntimeError,match='synthetic worker failure'):
        function(root_REST_rerank_GO=True,workers=2,output=tmp_path)
    record=V.C.read(tmp_path/V.FAILURE)
    assert record['pending_ids']==['test:2'] and record['completed_ids']==[]
    assert clean==[dict(wait=False,cancel_futures=True)]
    assert (tmp_path/V.EXEC).exists() and not(tmp_path/V.W.REST_EXEC).exists()


def test_finalize_binds_new_caller_without_editing_original_cal(tmp_path,monkeypatch):
    output,ids,rows,entries,immutable=fixture(tmp_path,monkeypatch)
    V.C.write(output/V.W.REST_EXEC,dict(original_failed_attempt=True))
    V.C.write(output/V.OWNER_FAILURE,dict(new_REST_rerank_rows=0))
    V.C.write(output/V.PROTOCOL,dict(source_sha256={'adapter.py':'synthetic'},input_sha256={}))
    V.C.write(output/V.PREP,dict(prepared=True))
    V.C.write(output/V.EXEC,dict(workers=3))
    V.C.write(output/'REST_ROW_INDEX.json',dict(entries=entries[46:]))
    V.R.save_rows_fast(output/'REST_OBSERVATIONS.jsonl.gz',rows[46:])
    V.C.write(output/'REST_OBSERVATION_SEAL.json',dict(rows=192,ordered_ids=ids[46:],
        observations_sha256=V.C.sha(output/'REST_OBSERVATIONS.jsonl.gz'),
        row_index_sha256=V.C.sha(output/'REST_ROW_INDEX.json'),protocol_sha256=V.C.sha(output/V.W.REST_PROTOCOL)))
    original_load=V.W._load
    def augmented(path):
        d=original_load(path)
        # A real source file proves the final source union remains verified.
        s=tmp_path/'adapter.py';s.write_text('pass\n')
        return {**d,'source_sha256':{'adapter.py':V.C.sha(s)},
            'input_sha256':{str((output/V.EXEC).relative_to(tmp_path)):V.C.sha(output/V.EXEC)}}
    monkeypatch.setattr(V,'_bound_load',augmented)
    V.finalize(root_finalize_GO=True,output=output)
    final=V.C.read(output/'FINAL_PROTOCOL.json');seal=V.C.read(output/'OBSERVATION_SEAL.json')
    completed=V.C.read(output/'FINAL_COMPLETION_RECEIPT_V30B.json')
    assert final['source_sha256']=={'adapter.py':V.C.sha(tmp_path/'adapter.py')}
    for document in (final,seal,completed):
        refs=document['operational_resumption']
        assert refs['resume_execution']['sha256']==V.C.sha(output/V.EXEC)
        assert refs['original_failed_execution']['sha256']==V.C.sha(output/V.W.REST_EXEC)
    assert seal['protocol_sha256']==V.C.sha(output/'FINAL_PROTOCOL.json')
    assert completed['observation_seal']['sha256']==V.C.sha(output/'OBSERVATION_SEAL.json')
    assert all(V.C.sha(tmp_path/p)==h for p,h in immutable.items())
    assert list(V.C.rows(output/'OBSERVATIONS.jsonl.gz'))==rows
