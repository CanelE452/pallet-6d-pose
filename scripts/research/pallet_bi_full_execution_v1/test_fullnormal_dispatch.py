from multiprocessing import get_context
from pathlib import Path
import pytest

from scripts.research.pallet_bi_full_execution_v1 import fullnormal_dispatch as D,fullnormal_parallel as R,qcov as Q


def test_existing_row_primitives_are_exact_function_objects_and_frozen_source():
    assert D.row_job is R.row_job
    assert D.initialize_worker is R.initialize_worker
    assert D.aggregate is R.aggregate
    assert D.verify_packet_bindings is R.verify_packet_bindings
    binding=D.source_bindings()
    assert binding['row_helper_sha256']==Q.sha(R.__file__)==D.ROW_HELPER_EXPECTED_SHA256
    assert binding['orchestration_sha256']==Q.sha(D.__file__)


def test_three_spawned_constructed_workers_no_dataset_solves(tmp_path):
    jobs=[('constructed',i,dict(id=str(i)),None,None)for i in range(9)]
    ctx=get_context('spawn');barrier=ctx.Barrier(3)
    results,failures,receipt=D.execute_static(jobs,tmp_path,3,D.initialize_constructed_probe,(barrier,),D.constructed_probe)
    assert not failures
    assert len(results)==9 and len({r['worker_pid']for r in results})==3
    assert all(set(r['thread_limits'].values())=={'1'}for r in results)
    assert sum(r['actual_new_optimizer_calls_this_job']for r in results)==0
    assert receipt==dict(submitted_jobs=9,unsubmitted_jobs=0,checkpoint_stop_observed=False,active_rows_drained_before_exit=True)


def test_checkpoint_stop_submits_no_rows_and_preserves_denominator(tmp_path):
    stop=tmp_path/'STOP';stop.write_text('authorized technical checkpoint stop')
    jobs=[('constructed',0,dict(id='unstarted'),None,None)]
    results,failures,receipt=D.execute_static(jobs,tmp_path,3,None,(),D.constructed_probe,stop)
    assert not results and not failures
    assert receipt['unsubmitted_jobs']==1 and receipt['checkpoint_stop_observed']


def test_reuse_proof_checks_packet_source_payload_and_origin_bytes(tmp_path):
    source=tmp_path/'origin';source.mkdir();Q.write(source/'RESULT.json.gz',dict(value=3));Q.write(source/'FEATURE.json.gz',dict(value=4))
    binding={'frozen.py':'abc'};signature='desired'
    seal=dict(cache_payload_sha256=signature,source_binding=binding,artifacts={n:Q.sha(source/n)for n in ('RESULT.json.gz','FEATURE.json.gz')})
    Q.write(source/'FRAME_SEAL.json',seal);root=tmp_path/'reuse';target=root/'C0/real/0000';target.mkdir(parents=True)
    import shutil
    for p in source.iterdir():shutil.copy2(p,target/p.name)
    packet_seal=tmp_path/'PACKET_SEAL.json';Q.write(packet_seal,dict(complete=True))
    Q.write(root/'REUSE_MANIFEST.json',dict(packet_seal_sha256=Q.sha(packet_seal),eligible_n=1,rows=[dict(domain='real',ordinal=0,arm='C0',source=str(source),
        source_frame_seal_sha256=Q.sha(source/'FRAME_SEAL.json'),desired_cache_payload_sha256=signature)]))
    assert D.verify_reuse_root(root,packet_seal,binding)['authenticated_rows']==1
    Q.write(target/'RESULT.json.gz',dict(value=9))
    with pytest.raises(ValueError,match='artifact'):D.verify_reuse_root(root,packet_seal,binding)
