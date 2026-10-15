"""Operational spool/order/reuse tests; synthetic rows and no images/models."""
from concurrent.futures import Future
from pathlib import Path
import weakref
import pytest

from . import observe_geometry_guided_edges_v4_29b as V


def row(i):
    return dict(id='synthetic:'+str(i),partition='CAL',summary=dict(selected_physical_edges=i%3,
        selected_partial_edges=i%2,actual_chain_hypotheses=i,zero_selected_edges=i%3==0),
        unchanged_observation_math_sha256='a'*64,observation=dict(actual_points=[[i,1.]]))


def test_single_spool_exact_value_hash_and_no_silent_second_row(tmp_path):
    data=row(0);path=tmp_path/'row.jsonl.gz';V.C.save_rows(path,[data]);entry=V._index_row(data,path)
    assert V._read_single(path,entry)==data
    duplicate=tmp_path/'two_rows.jsonl.gz';V.C.save_rows(duplicate,[data,row(1)]);entry['sha256']=V.C.sha(duplicate)
    with pytest.raises(ValueError,match='single canonical'):V._read_single(duplicate,entry)


def test_metadata_summary_keeps_zero_edge_frames_and_no_pixel_payload():
    entries=[dict(id=r['id'],partition=r['partition'],summary=r['summary'])for r in map(row,range(46))]
    summary=V._summary(entries)
    assert summary['rows']==46 and summary['partition_counts']=={'CAL':46}
    assert summary['zero_selected_edge_frames']==16
    assert all('observation'not in e for e in entries)


def test_bounded_pending_releases_futures_and_reuses_exact_old_rows_without_worker(tmp_path,monkeypatch):
    (tmp_path/'row_spool').mkdir();reused={}
    for i in range(2):
        data=row(i);path=V._row_path(tmp_path,data['id']);V.C.save_rows(path,[data]);reused[data['id']]=V._index_row(data,path)
    V.C.write(tmp_path/'REUSE_SEAL.json',dict(entries=reused))
    protocol=dict(source_sha256={},input_sha256={},protected_user_sha256={})
    prep=dict(ordered_ids=[row(i)['id']for i in range(46)])
    V.C.write(tmp_path/'PREPARATION_SEAL.json',{})
    monkeypatch.setattr(V,'_load',lambda output:(protocol,prep))
    original_rows=V.C.rows
    monkeypatch.setattr(V.C,'rows',lambda path:iter([dict(id=row(i)['id'],partition='CAL')for i in range(46)])
        if Path(path).name=='INPUTS.jsonl.gz'else original_rows(path))
    calls=[];refs=[];peak=[]
    class Pool:
        def __init__(self,**kwargs):self.workers=kwargs['max_workers']
        def submit(self,operation,task):
            calls.append(task['id']);future=Future();future.set_result(row(int(task['id'].split(':')[1])))
            refs.append(weakref.ref(future));peak.append(sum(r()is not None for r in refs))
            return future
        def shutdown(self,**kwargs):pass
    monkeypatch.setattr(V,'ProcessPoolExecutor',Pool)
    captured={}
    def emit(output,partition,p,prepared,entries):captured.update(entries)
    monkeypatch.setattr(V,'_emit_partition',emit)
    V.observe(tmp_path,'CAL',workers=3)
    assert len(calls)==44 and 'synthetic:0'not in calls and 'synthetic:1'not in calls
    assert len(captured)==46 and max(peak)<=3
    assert sum(r()is not None for r in refs)==0
    assert all('observation'not in entry for entry in captured.values())
    assert V._read_single(V._row_path(tmp_path,'synthetic:1'),captured['synthetic:1'])==row(1)
