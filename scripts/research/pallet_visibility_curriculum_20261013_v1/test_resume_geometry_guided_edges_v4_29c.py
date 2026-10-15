"""Synthetic-only restart storage contract; no models/images/labels or fits."""
from concurrent.futures import Future
import weakref
import pytest

from . import resume_geometry_guided_edges_v4_29c as R


def row(i):
    return dict(id='synthetic:'+str(i),partition='CAL',schema=R.V.ROW_SCHEMA,
        observation_generation=R.B.P.OBSERVATION_GENERATION,unchanged_observation_math_sha256='a'*64,
        summary=dict(selected_physical_edges=i%3,selected_partial_edges=i%2,
            actual_chain_hypotheses=i,zero_selected_edges=i%3==0),observation=dict(actual_points=[[i,1.]]))


def spool(output,i):
    data=row(i);path=R.B._row_path(output,data['id']);R.C.save_rows(path,[data])
    return R.B._index_row(data,path)


def test_actual_GO_gate_precedes_all_IO(monkeypatch):
    def blocked(*args,**kwargs):raise AssertionError('No IO before root GO')
    monkeypatch.setattr(R.C,'read',blocked)
    with pytest.raises(ValueError,match='explicit root'):
        R.resume()


def test_exact_existing_spool_sha_single_row_canonical_math_are_required(tmp_path):
    (tmp_path/'row_spool').mkdir();entry=spool(tmp_path,0)
    metadata={entry['id']:dict(partition='CAL',math_sha256='a'*64)}
    result=R._verify_spool_set(tmp_path,metadata,{entry['id']:entry})
    assert result[entry['id']]['canonical_row_sha256']==entry['canonical_row_sha256']
    bad={**entry,'canonical_row_sha256':'0'*64}
    with pytest.raises(ValueError,match='single canonical'):
        R._verify_spool_set(tmp_path,metadata,{entry['id']:bad})
    with pytest.raises(ValueError,match='query math'):
        R._verify_spool_set(tmp_path,{entry['id']:dict(partition='CAL',math_sha256='b'*64)},{entry['id']:entry})


def test_extra_or_partial_spool_is_not_silently_reused(tmp_path):
    (tmp_path/'row_spool').mkdir();entry=spool(tmp_path,0);spool(tmp_path,1)
    with pytest.raises(ValueError,match='Exact complete spool set'):
        R._verify_spool_set(tmp_path,{entry['id']:dict(partition='CAL',math_sha256='a'*64)},{entry['id']:entry})


def fake_pool(monkeypatch):
    calls=[];refs=[];peaks=[]
    class Pool:
        def __init__(self,**kwargs):self.workers=kwargs['max_workers']
        def submit(self,operation,task):
            calls.append(task['id']);future=Future();future.set_result(row(int(task['id'].split(':')[1])))
            refs.append(weakref.ref(future));peaks.append(sum(ref()is not None for ref in refs))
            return future
        def shutdown(self,**kwargs):pass
    monkeypatch.setattr(R,'ProcessPoolExecutor',Pool)
    return calls,refs,peaks


def test_pending_future_bound_and_existing_row_never_resubmitted_or_rewritten(tmp_path,monkeypatch):
    (tmp_path/'row_spool').mkdir();old=spool(tmp_path,0);entries={old['id']:old};before=R.C.sha(R.B._row_path(tmp_path,old['id']))
    calls,refs,peaks=fake_pool(monkeypatch)
    tasks=[dict(id=row(i)['id'])for i in range(1,7)]
    new=R._collect_missing(tmp_path,tasks,entries,{row(i)['id']:'a'*64 for i in range(7)},3)
    assert len(new)==6 and len(entries)==7 and row(0)['id']not in calls
    assert max(peaks)<=3 and sum(ref()is not None for ref in refs)==0
    assert before==R.C.sha(R.B._row_path(tmp_path,old['id']))


def test_existing_row_accidentally_in_missing_inputs_fails_before_worker(tmp_path,monkeypatch):
    (tmp_path/'row_spool').mkdir();old=spool(tmp_path,0);calls,_,_=fake_pool(monkeypatch)
    with pytest.raises(ValueError,match='never reach image worker'):
        R._collect_missing(tmp_path,[dict(id=old['id'])],{old['id']:old},{old['id']:'a'*64},3)
    assert calls==[]and R.C.sha(R.B._row_path(tmp_path,old['id']))==old['sha256']


def test_exact32_plus14_first_CAL46_seal_and_resume_supplement_preserve_all_originals(tmp_path,monkeypatch):
    # Test artifacts live outside the repository; keep actual content hashes
    # while avoiding the production-only repository-relative path restriction.
    monkeypatch.setattr(R.OLD,'_binding',lambda path:dict(path=str(path),sha256=R.C.sha(path)))
    (tmp_path/'row_spool').mkdir();existing={row(i)['id']:spool(tmp_path,i)for i in range(32)}
    protocol=dict(source_sha256={},input_sha256={},protected_user_sha256={})
    for name,value in [('PROTOCOL.json',protocol),('PREPARATION_SEAL.json',{}),('CAL_EXECUTION_PROTOCOL.json',dict(preboot=True)),
            ('CAL_LIVE_PROGRESS.json',dict(completed=32)),('REUSE_SEAL.json',{})]:R.C.write(tmp_path/name,value)
    R.C.save_rows(tmp_path/'INPUTS.jsonl.gz',[
        dict(id=row(i)['id'],partition='CAL',unchanged_observation_math_sha256='a'*64)for i in range(46)])
    plan=tmp_path/'PLAN.json';restart=tmp_path/'RESTART.json';R.C.write(plan,{});R.C.write(restart,{})
    monkeypatch.setattr(R,'PLAN',plan);monkeypatch.setattr(R,'RESTART',restart)
    immutable={str(p):R.C.sha(p)for p in tmp_path.glob('*.json')if p not in(plan,restart)}
    immutable.update({str(R.B._row_path(tmp_path,ident)):e['sha256']for ident,e in existing.items()})
    immutable[str(tmp_path/'INPUTS.jsonl.gz')]=R.C.sha(tmp_path/'INPUTS.jsonl.gz')
    prep=dict(ordered_ids=[row(i)['id']for i in range(46)])
    monkeypatch.setattr(R.B,'_load',lambda output:(protocol,prep))
    R.C.write(tmp_path/R.PREFLIGHT,dict(source_sha256={},immutable_file_sha256=immutable,
        plan=dict(sha256=R.C.sha(plan)),restart_receipt=dict(sha256=R.C.sha(restart)),exact_existing_entries=existing,
        missing_CAL_ids=[row(i)['id']for i in range(32,46)]))
    calls,refs,peaks=fake_pool(monkeypatch)
    R.resume(tmp_path,3,root_actual_observation_GO=True)
    assert len(calls)==14 and set(calls)=={row(i)['id']for i in range(32,46)}
    for path,digest in immutable.items():assert R.C.sha(path)==digest
    seal=R.C.read(tmp_path/'CAL_OBSERVATION_SEAL.json');supp=R.C.read(tmp_path/R.SUPPLEMENT)
    assert seal['rows']==46 and seal['complete_CAL46_includes_exact32_preboot_plus14_resumed']
    assert seal['resume_execution_seal']['sha256']==R.C.sha(tmp_path/R.EXECUTION_SEAL)
    assert supp['CAL_observation_seal']['sha256']==R.C.sha(tmp_path/'CAL_OBSERVATION_SEAL.json')
    assert [r['id']for r in R.C.rows(tmp_path/'CAL_OBSERVATIONS.jsonl.gz')]==prep['ordered_ids']
    assert max(peaks)<=3 and sum(ref()is not None for ref in refs)==0
