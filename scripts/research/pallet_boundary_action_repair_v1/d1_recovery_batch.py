"""Runtime-only restart of a terminated shared D1 dispatcher.

The frozen dispatcher and numerical producer are imported unchanged. A fresh
private runtime namespace preserves the interrupted lock/journal while the
same verified completed frame folders are reused. No new observation is run.
"""
from __future__ import annotations
import argparse,json,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path

from . import d1_memoized_batch as D
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,sha,canonical


def utc():return datetime.now(timezone.utc).isoformat()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare(private):
    start,cpu=time.monotonic(),time.process_time();root=Path(private)
    runtime=root/'d1_recovery_after_interruption_01'
    original_lock=root/'D1_SHARED_EXECUTION_BATCH_LOCK.json'
    lock=read(original_lock)
    source=D.source_bindings()
    if source!=lock['source_sha256']:raise ValueError('Original frozen scientific/runtime source bytes changed')
    # The first verified absent-process read was captured in the original
    # tool result. This conservatively bounds the terminated parent lifetime.
    absent_utc='2026-10-18T03:12:59.809020+00:00'
    expected=(2107373,2108523,2108524,2108525,2108526)
    for pid in expected:
        if Path('/proc',str(pid)).exists():raise ValueError('Old PID unexpectedly exists; review identity before restart')
    journal=root/'D1_SHARED_RUNTIME_JOURNAL.jsonl'
    completed_journal=[json.loads(x)for x in journal.read_text().splitlines()if x.strip()]
    if len(completed_journal)!=652 or any(r.get('status')=='FAILED_TECHNICAL'for r in completed_journal):
        raise ValueError('Interrupted journal differs from preserved observed state')
    completed={};unfinished=[]
    for domain in('real','synthetic'):
        base=root/('d1_'+domain+'_candidates_v1')/'frames';completed[domain]={}
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():continue
            if(folder/'FRAME_SEAL.json').exists():
                completed[domain][folder.name]=sha(folder/'FRAME_SEAL.json')
            else:
                stat=folder.stat();destination=root/'d1_interrupted_shared_attempt_01'/domain/folder.name
                files={str(f.relative_to(folder)):sha(f)for f in folder.rglob('*')if f.is_file()}
                if destination.exists():raise ValueError('Interrupted attempt archive already exists')
                unfinished.append(dict(domain=domain,index=int(folder.name),original_path=str(folder),
                    archived_path=str(destination),original_directory_mtime_epoch=stat.st_mtime,
                    file_sha256=files,FRAME_SEAL_present=False,wall_seconds=None,CPU_seconds=None,
                    optimizer_calls=None,PnP_calls=None,NN_calls=0,
                    unknown_scope='Producer had not persisted E_RUN/FRAME_SEAL. Partial solver/feature work and numeric counts are unknown, not zero. Observation was reused; no new NN forward is part of CPU jobs.'))
    if{d:len(x)for d,x in completed.items()}!={'real':238,'synthetic':454}or len(unfinished)!=4:
        raise ValueError('Interrupted sealed/unsealed cardinalities changed')
    interruption=dict(schema='D1_INTERRUPTED_SHARED_EXECUTION_V1',status='INTERRUPTED_EXTERNAL_CAUSE_UNKNOWN',
        execution_batch_id=lock['execution_batch_id'],started_utc=lock['started_utc'],
        last_completed_frame_utc=completed_journal[-1]['ended_utc'],first_verified_processes_absent_utc=absent_utc,
        original_tool_session_id=76213,observed_tool_session_exit_code=143,worker_exit_codes=None,
        termination_signal_sender='UNKNOWN',concrete_termination_cause='UNKNOWN',
        agent_issued_termination=False,raw_traceback_file=None,workers=lock['workers'],
        parent_wall_seconds=None,parent_process_lifetime_upper_bound_seconds=(datetime.fromisoformat(absent_utc)-datetime.fromisoformat(lock['started_utc'])).total_seconds(),
        parent_wall_scope='Original exact end timer was lost; batch start to first verified absent-process read is a conservative parent-wall upper bound.',
        journal_rows=len(completed_journal),domain_completed_frames={d:len(x)for d,x in completed.items()},
        completed_frame_seal_sha256=completed,unsealed_attempts=unfinished,
        original_execution_lock_sha256=sha(original_lock),original_journal_sha256=sha(journal),
        original_log_sha256=sha(root/'D1_SHARED_FIXED_E_LIVE.log'),
        interrupted_partial_operations=None,interrupted_partial_CPU_seconds=None,
        completed_worker_operations_scope='Available per-frame completed counts remain in original FRAME_SEAL files. They do not measure the four interrupted attempts.',
        last_verified_worker_resource_snapshot=dict(associated_forecast='D1_SHARED_COST_FORECAST_27.json',
            snapshot_utc_approx='2026-10-18T03:11:57.692817+00:00',
            CPU_seconds_display_precision_one_second={str(p):v for p,v in[(2108523,9767),(2108524,9766),(2108525,9767),(2108526,9767)]},
            RSS_KiB={str(p):v for p,v in[(2108523,1362104),(2108524,1356068),(2108525,1348116),(2108526,1347668)]},
            scope='Transcribed from read-only ps tool output, not a persisted process exit/rusage record; these are cumulative lower bounds, not row stage attribution.'),
        created_utc=utc(),reference_or_accuracy_reads=0)
    write_new(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json',interruption)
    runtime.mkdir()
    links=['d1_upstream_v1','d1_real_observations_v1','d1_synthetic_observations_v1',
        'd1_real_candidates_v1','d1_synthetic_candidates_v1','D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json',
        'D1_REAL_CONTROLLED_QUIESCE.json','D1_CACHE_PARITY_FIXED_SELECTION.json','d1_cache_parity_actual']
    for name in links:
        target=root/name
        if not target.exists():raise ValueError('Required frozen input missing: '+name)
        (runtime/name).symlink_to(target,target_is_directory=target.is_dir())
    for item in unfinished:
        dest=Path(item['archived_path']);dest.parent.mkdir(parents=True,exist_ok=True)
        Path(item['original_path']).rename(dest)
    amendment=dict(schema='D1_INTERRUPTION_RECOVERY_TECHNICAL_LOCK_V1',status='LOCKED',
        original_private_root=str(root),recovery_private_root=str(runtime),
        original_execution_batch_id=lock['execution_batch_id'],
        original_lock_sha256=sha(original_lock),original_journal_sha256=sha(journal),
        interrupted_execution_receipt_sha256=sha(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json'),
        frozen_original_source_sha256=source,recovery_launcher_source_sha256=sha(__file__),
        completed_frames_reused=completed,completed_frame_counts={d:len(x)for d,x in completed.items()},
        pending_expected_rows=549,pending_expected_domain_rows={'real':0,'synthetic':549},
        unfinished_attempts_preserved=unfinished,new_runtime_namespace_links=links,
        exact_same_input_observation_solver_feature_procedure=True,model_or_scientific_setting_changes=False,
        new_observation_or_NN_forwards=0,new_PRIMARY_reference_reads=0,
        original_paths_preserved=['D1_SHARED_EXECUTION_BATCH_LOCK.json','D1_SHARED_RUNTIME_JOURNAL.jsonl','D1_SHARED_FIXED_E_LIVE.log'],
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,created_utc=utc())
    write_new(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json',amendment)
    return amendment


def run(private,old_run,original_run,typed_run,*,workers=4,GO=False):
    if not GO or not 1<=workers<=4:raise ValueError('Existing root recovery GO and CPU budget required')
    root=Path(private);amendment=read(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json')
    if amendment['recovery_launcher_source_sha256']!=sha(__file__):raise ValueError('Recovery launcher source changed')
    if amendment['frozen_original_source_sha256']!=D.source_bindings():raise ValueError('Original frozen procedure changed')
    if sha(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json')!=amendment['original_lock_sha256']or sha(root/'D1_SHARED_RUNTIME_JOURNAL.jsonl')!=amendment['original_journal_sha256']:
        raise ValueError('Interrupted lock/journal must remain unchanged')
    runtime=Path(amendment['recovery_private_root'])
    result=D.run(runtime,old_run,original_run,typed_run,workers=workers,GO=GO)
    receipt=runtime/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json';destination=root/receipt.name
    if destination.exists():raise ValueError('Canonical final runtime receipt must not overwrite old output')
    shutil.copy2(receipt,destination)
    final=dict(schema='D1_INTERRUPTION_RECOVERY_FINALIZATION_V1',status='COMPLETE',
        recovery_execution_batch_id=read(receipt)['execution_batch_id'],
        recovery_runtime_lock_sha256=sha(runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json'),
        recovery_runtime_journal_sha256=sha(runtime/'D1_SHARED_RUNTIME_JOURNAL.jsonl'),
        recovery_runtime_receipt_sha256=sha(receipt),canonical_receipt_exact_copy=sha(destination)==sha(receipt),
        recovery_technical_amendment_lock_sha256=sha(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json'),
        interrupted_lock_and_journal_unchanged=sha(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json')==amendment['original_lock_sha256']and sha(root/'D1_SHARED_RUNTIME_JOURNAL.jsonl')==amendment['original_journal_sha256'],
        final_feature_seals={d:sha(root/('d1_'+d+'_candidates_v1')/'FEATURE_SEAL.json')for d in('real','synthetic')},
        all_original_and_restarted_rows_same_frozen_scientific_source=True,
        no_new_observation_or_model_forward=True,reference_or_accuracy_reads=0,source_sha256=sha(__file__))
    write_new(root/'D1_RECOVERY_FINALIZATION_RECEIPT.json',final)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--private',required=True)
    p.add_argument('--old-run');p.add_argument('--original-run');p.add_argument('--typed-run')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--root-GO',action='store_true');a=p.parse_args()
    print(canonical(prepare(a.private)if a.mode=='prepare'else run(a.private,a.old_run,a.original_run,a.typed_run,workers=a.workers,GO=a.root_GO)))
