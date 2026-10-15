"""Synthetic append/finalization/scheduling contracts; no real archive IO."""
from concurrent.futures import Future
from copy import deepcopy

import pytest

from . import rerank_observed_bi_edge_REST_v4_30b as V


@pytest.mark.parametrize('function,kwargs',[(V.prepare,{}),(V.rerank,{}),(V.finalize,{})])
def test_no_go_precedes_all_io(monkeypatch,function,kwargs):
    monkeypatch.setattr(V.C,'read',lambda *a:pytest.fail('IO before root GO'))
    with pytest.raises(PermissionError):function(**kwargs)


def test_conflicting_ancestry_is_not_silently_overwritten():
    assert V._union({'a':'same'},{'a':'same','b':'new'})=={'a':'same','b':'new'}
    with pytest.raises(ValueError,match='conflict'):V._union({'a':'old'},{'a':'new'})


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(V.C,'ROOT',tmp_path)
    output=tmp_path/'new';output.mkdir();(output/'row_spool').mkdir()
    ids=[f'c:{i}' for i in range(46)]+[f't:{i}' for i in range(49)]+[f'r:{i}' for i in range(143)]
    rows=[dict(id=i,partition='CAL'if n<46 else'TEST'if n<95 else'REMAINDER',
        summary=dict(zero_selected_edges=True),selection_profile=V.PROFILE,
        observation=dict(selected_chain_refs=[]))for n,i in enumerate(ids)]
    entries=[]
    for row in rows:
        path=V.OLD._row_path(output,row['id']);V.R.save_rows_fast(path,[row])
        entries.append(dict(id=row['id'],partition=row['partition'],path=path.name,
            sha256=V.C.sha(path),canonical_row_sha256=V.OLD.OLD._digest(row)))
    original=dict(ids=ids,configuration={'fixed':43},selection_profile=V.PROFILE,
        raw_observation_generation=V.RAW,source_sha256={},input_sha256={},protected_user_sha256={})
    V.C.write(output/'PROTOCOL.json',original)
    V.C.write(output/'PREPARATION_SEAL.json',{'CAL':'frozen'})
    V.C.write(output/'CAL_EXECUTION_PROTOCOL.json',{'workers':3})
    V.C.write(output/'CAL_SUMMARY.json',{'rows':46})
    V.C.write(output/'CAL_ROW_INDEX.json',dict(entries=entries[:46]))
    V.R.save_rows_fast(output/'CAL_OBSERVATIONS.jsonl.gz',rows[:46])
    V.C.write(output/'CAL_OBSERVATION_SEAL.json',dict(rows=46,ordered_ids=ids[:46]))
    immutable=V._cal_files(output)
    protocol=dict(ids=ids,CAL_ids=ids[:46],REST_ids=ids[46:],configuration={'fixed':43},
        source_sha256={},input_sha256={},protected_user_sha256={},
        original_CAL_file_sha256=immutable,parent_REST_entries=entries[46:])
    V.C.write(output/V.REST_PROTOCOL,protocol)
    V.C.write(output/V.REST_PREP,dict(protocol_sha256=V.C.sha(output/V.REST_PROTOCOL),ordered_ids=ids[46:]))
    return output,ids,rows,entries,immutable


def test_full238_finalize_preserves_all_cal_bytes_and_zero_rows(tmp_path,monkeypatch):
    output,ids,rows,entries,immutable=fixture(tmp_path,monkeypatch)
    V.C.write(output/V.REST_EXEC,dict(workers=3))
    V.C.write(output/'REST_ROW_INDEX.json',dict(entries=entries[46:]))
    V.R.save_rows_fast(output/'REST_OBSERVATIONS.jsonl.gz',rows[46:])
    V.C.write(output/'REST_OBSERVATION_SEAL.json',dict(rows=192,ordered_ids=ids[46:],
        observations_sha256=V.C.sha(output/'REST_OBSERVATIONS.jsonl.gz'),
        row_index_sha256=V.C.sha(output/'REST_ROW_INDEX.json'),protocol_sha256=V.C.sha(output/V.REST_PROTOCOL)))
    result=V.finalize(root_finalize_GO=True,output=output)
    seal=V.C.read(output/'OBSERVATION_SEAL.json');final=V.C.read(output/'FINAL_PROTOCOL.json')
    assert result['sha256']==V.C.sha(output/'OBSERVATION_SEAL.json')
    assert seal['rows']==238 and seal['ordered_ids']==ids
    assert seal['protocol_sha256']==V.C.sha(output/'FINAL_PROTOCOL.json')
    assert seal['partition_seals']=={p:V.C.sha(output/(p+'_OBSERVATION_SEAL.json'))for p in('CAL','REST')}
    assert final['ids']==ids and final['selection_profile']==V.PROFILE
    assert final['selected_chain_refs_graph_and_row_summary_are_authoritative']
    assert list(V.C.rows(output/'OBSERVATIONS.jsonl.gz'))==rows
    assert all(V.C.sha(tmp_path/path)==digest for path,digest in immutable.items())
    with pytest.raises(FileExistsError):V.finalize(root_finalize_GO=True,output=output)


def test_cal_immutability_failure_blocks_before_finalize(tmp_path,monkeypatch):
    output,_,_,_,_=fixture(tmp_path,monkeypatch)
    V.C.write(output/'CAL_SUMMARY.json',dict(rows=0))
    with pytest.raises(ValueError):V.finalize(root_finalize_GO=True,output=output)
    assert not(output/'FINAL_PROTOCOL.json').exists()


def test_same_frozen_worker_all192_bounded_pending(tmp_path,monkeypatch):
    output,ids,rows,entries,immutable=fixture(tmp_path,monkeypatch)
    # Simulate uncomputed REST spools; CAL46 stays frozen.
    for ident in ids[46:]:V.OLD._row_path(output,ident).unlink()
    called=[];live=[];max_pending=0
    by_id={r['id']:r for r in rows}
    def frozen_worker(case,configuration,target):
        assert configuration=={'fixed':43} and target==output
        called.append(case['id']);row=by_id[case['id']]
        path=V.OLD._row_path(output,case['id']);V.R.save_rows_fast(path,[row])
        return {**deepcopy(case),'sha256':V.C.sha(path)},dict(id=case['id'],zero_selected=True)
    class Pool:
        def __init__(self,**kwargs):assert kwargs['max_workers']==3
        def submit(self,worker,*args):
            nonlocal max_pending
            assert worker is V.R._worker
            future=Future();future.set_result(worker(*args));live.append(future)
            max_pending=max(max_pending,len(live));return future
        def shutdown(self,**kwargs):pass
    def done_once(pending,**kwargs):
        future=next(iter(pending));live.remove(future);return {future},set(pending)-{future}
    monkeypatch.setattr(V.R,'_worker',frozen_worker)
    monkeypatch.setattr(V,'ProcessPoolExecutor',Pool);monkeypatch.setattr(V,'wait',done_once)
    result=V.rerank(root_REST_rerank_GO=True,workers=3,output=output)
    assert called==ids[46:] and max_pending<=3
    assert result['sha256']==V.C.sha(output/'REST_OBSERVATION_SEAL.json')
    assert [e['id']for e in V.C.read(output/'REST_ROW_INDEX.json')['entries']]==ids[46:]
    assert list(V.C.rows(output/'REST_OBSERVATIONS.jsonl.gz'))==rows[46:]
    assert all(V.C.sha(tmp_path/path)==digest for path,digest in immutable.items())
    with pytest.raises(FileExistsError):V.rerank(root_REST_rerank_GO=True,workers=3,output=output)


@pytest.mark.parametrize('workers',[True,0,4])
def test_worker_budget_rejects_before_archive_io(monkeypatch,workers):
    monkeypatch.setattr(V,'_load',lambda *a:pytest.fail('Invalid worker count reached archive IO'))
    with pytest.raises(ValueError):V.rerank(root_REST_rerank_GO=True,workers=workers)
