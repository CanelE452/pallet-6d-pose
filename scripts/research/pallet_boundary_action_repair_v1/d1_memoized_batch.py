"""Runtime-only shared dispatcher for the unchanged fixed D1 procedure.

Completed V3 rows are verified and preserved. Original REMAINDER143 stages
are reused only through the exact mathematical-input memoization adapter.
Real and synthetic rows share one worker pool and one wall-clock ledger.
"""
from __future__ import annotations
import argparse,hashlib,os,resource,shutil,time,traceback,uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path

from . import d1_batch as B,d1_cpu_logging_v3 as V3,d1_stage_memoization as M
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,rows,sha,write,save_rows,canonical


def utc():return datetime.now(timezone.utc).isoformat()


def source_bindings():
    result=M.source_bindings();root=Path(__file__).parents[3]
    result[str(Path(__file__).relative_to(root))]=sha(__file__);return result


def validate_completed(frame,obs,folder):
    """Accept earlier reporting wrappers only when the frozen math is exact."""
    folder=Path(folder);obs=Path(obs);s=read(folder/'FRAME_SEAL.json')
    if s['id']!=frame['id']or not s['complete']or s['observation_seal_sha256']!=sha(obs/'OBSERVATION_SEAL.json'):
        raise ValueError('Completed frame ID/observation changed')
    expected=V3.source_bindings()
    if any(s['source_sha256'].get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed frame frozen mathematical source changed')
    for name,key in [('E_RUN.jsonl.gz','e_run_sha256'),('FEATURES.jsonl.gz','features_sha256'),('BOUNDARY_BUNDLE.jsonl.gz','boundary_bundle_sha256')]:
        if sha(folder/name)!=s[key]:raise ValueError('Completed frame output changed: '+name)
    if not s['e0_independently_executed']or s['new_B_or_A_candidates']or s['history_only_union']:
        raise ValueError('Completed frame procedure mismatch')
    return s


def worker(domain,index,frame,obs,folder,old_run,entry):
    started=utc()
    try:
        result=(M.solver_job(frame,obs,folder,old_run,entry)if entry is not None else V3.solver_job(frame,obs,folder,old_run))
        return dict(domain=domain,index=index,id=frame['id'],seal=result,
            execution_kind='EXACT_ORIGINAL_STAGE_MEMOIZATION'if entry is not None else'NEW_UNCHANGED_E_SOLVE',
            started_utc=started,ended_utc=utc())
    except BaseException as exc:
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        failure=dict(domain=domain,index=index,id=frame['id'],status='FAILED_TECHNICAL',started_utc=started,
            ended_utc=utc(),error=repr(exc),traceback=traceback.format_exc(),source_sha256=source_bindings())
        write(folder/('DISPATCH_FAILURE_'+uuid.uuid4().hex+'.json'),failure)
        return failure


def run(private,old_run,original_run,typed_run,*,workers=4,GO=False):
    if not GO or not 1<=workers<=4:raise ValueError('Root execution GO and at most four workers required')
    root=Path(private);started=time.monotonic();parentcpu=time.process_time();startutc=utc();batchid='D1_SHARED_FIXED_E_'+uuid.uuid4().hex
    source=source_bindings();amendment=root/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json'
    a=read(amendment)
    if a['additional_source_sha256']!={k:v for k,v in source.items()if k not in V3.source_bindings()}:
        raise ValueError('Additional memoization/runtime source lock changed')
    if a['frozen_V3_source_sha256']!=V3.source_bindings():raise ValueError('Frozen original D1 source changed')
    quiesce=read(root/'D1_REAL_CONTROLLED_QUIESCE.json')
    if quiesce['status']!='QUIESCED_WITH_ALL_STARTED_ROWS_SEALED':raise ValueError('Original real batch must be safely quiesced')
    entries=M.load_entries(original_run,typed_run)
    domains={};pending=[];completed={};modes=Counter()
    for domain,name in [('real','REAL238'),('synthetic','SYNTHETIC1003')]:
        inp=root/'d1_upstream_v1'/(name+'_UPSTREAM.jsonl.gz');obs=root/('d1_'+domain+'_observations_v1');out=root/('d1_'+domain+'_candidates_v1')
        frames=list(rows(inp));oseal=read(obs/'OBSERVATION_SEAL.json')
        if [f['id']for f in frames]!=oseal['ordered_ids']or not oseal['complete']or oseal['input_sha256']!=sha(inp):
            raise ValueError('Same original ordered inputs and sealed observations required')
        if oseal['source_sha256']!=B.source_bindings():raise ValueError('Original observation source closure changed')
        out.mkdir(parents=True,exist_ok=True)
        if not(out/'BATCH_LOCK.json').exists():
            write(out/'BATCH_LOCK.json',dict(schema='D1_SOLVER_LOCK_V1',inputs_sha256=sha(inp),
                observation_seal_sha256=sha(obs/'OBSERVATION_SEAL.json'),source_sha256=V3.source_bindings(),
                ordered_ids=[f['id']for f in frames],old_config_sha256=sha(Path(old_run)/'CONFIG_FROZEN.json'),workers=workers,
                shared_execution_batch_id=batchid,history_only_union=False))
        oldlock=read(out/'BATCH_LOCK.json')
        if oldlock['source_sha256']!=V3.source_bindings():raise ValueError('Original CPU source closure changed')
        for k,v in dict(inputs_sha256=sha(inp),observation_seal_sha256=sha(obs/'OBSERVATION_SEAL.json'),
            ordered_ids=[f['id']for f in frames],old_config_sha256=sha(Path(old_run)/'CONFIG_FROZEN.json')).items():
            if oldlock[k]!=v:raise ValueError('Preserved original batch lock changed: '+k)
        domains[domain]=dict(inputs=inp,obs=obs,out=out,frames=frames,done={})
        for i,f in enumerate(frames):
            folder=out/'frames'/f'{i:04d}';op=obs/'frames'/f'{i:04d}'
            # The actual fixed first-REMAINDER parity solve is reused verbatim.
            parity=root/'d1_cache_parity_actual'/'frames'/f'{i:04d}'
            if domain=='real'and i==read(root/'D1_CACHE_PARITY_FIXED_SELECTION.json')['index']and not(folder/'FRAME_SEAL.json').exists():
                validate_completed(f,op,parity)
                if folder.exists()and any(folder.iterdir()):raise ValueError('Unsealed parity destination has prior records')
                folder.mkdir(parents=True,exist_ok=True)
                for sourcefile in parity.iterdir():
                    if sourcefile.is_file():shutil.copy2(sourcefile,folder/sourcefile.name)
                write(out/'PARITY_ACTUAL_FRAME_IMPORT.json',dict(id=f['id'],index=i,source_frame_seal_sha256=sha(parity/'FRAME_SEAL.json'),
                    destination_frame_seal_sha256=sha(folder/'FRAME_SEAL.json'),extra_solver_calls=0,source_files_preserved=True))
            if(folder/'FRAME_SEAL.json').exists():
                result=validate_completed(f,op,folder);domains[domain]['done'][i]=result
                modes['PREVIOUS_COMPLETED_V3_ROWS']+=1
            else:
                if folder.exists()and any(folder.iterdir()):raise ValueError('Prior unsealed frame needs explicit technical review: '+str(folder))
                entry=entries.get(f['id'])if domain=='real'else None
                pending.append((domain,i,f,str(op),str(folder),str(old_run),entry))
    lock=dict(schema='D1_SHARED_FIXED_E_RUNTIME_LOCK_V1',execution_batch_id=batchid,started_utc=startutc,
        workers=workers,additional_source_sha256=a['additional_source_sha256'],source_sha256=source,
        technical_amendment_lock_sha256=sha(amendment),quiesce_receipt_sha256=sha(root/'D1_REAL_CONTROLLED_QUIESCE.json'),
        ordered_pending=[dict(domain=j[0],index=j[1],id=j[2]['id'],kind='EXACT_ORIGINAL_STAGE_MEMOIZATION'if j[-1]is not None else'NEW_UNCHANGED_E_SOLVE')for j in pending],
        domain_inputs={d:dict(inputs_sha256=sha(x['inputs']),observation_seal_sha256=sha(x['obs']/'OBSERVATION_SEAL.json'),
            preserved_batch_lock_sha256=sha(x['out']/'BATCH_LOCK.json'),preserved_completed_frames=len(x['done']))for d,x in domains.items()},
        method_changes=False,reference_or_accuracy_reads=0)
    write(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json',lock)
    failures=[];journals=[]
    journal=(root/'D1_SHARED_RUNTIME_JOURNAL.jsonl').open('x')
    with journal,ProcessPoolExecutor(max_workers=workers)as pool:
        futures={pool.submit(worker,*job):job[:3]for job in pending}
        for future in as_completed(futures):
            receipt=future.result();journals.append(receipt)
            if receipt.get('status')=='FAILED_TECHNICAL':failures.append(receipt)
            else:
                domains[receipt['domain']]['done'][receipt['index']]=receipt['seal'];modes[receipt['execution_kind']]+=1
            print(canonical(dict(stage='D1_SHARED_FIXED_E',domain=receipt['domain'],done=len(journals),total=len(pending),
                id=receipt['id'],status=receipt.get('status','SEALED'),wall_seconds=time.monotonic()-started)),flush=True)
            journal.write(canonical({k:v for k,v in receipt.items()if k!='seal'})+'\n');journal.flush()
    ended=utc();wall=time.monotonic()-started
    runtime=dict(schema='D1_SHARED_FIXED_E_RUNTIME_RECEIPT_V1',execution_batch_id=batchid,status='FAILED_TECHNICAL'if failures else'COMPLETE',
        started_utc=startutc,ended_utc=ended,wall_seconds=wall,workers=workers,rows=len(pending),execution_mode_counts=dict(modes),
        domain_frame_counts={d:len(x['done'])for d,x in domains.items()},source_sha256=source,
        execution_lock_sha256=sha(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json'),failures=failures,
        job_wall_seconds=sum(r['seal']['resource']['wall_seconds']for r in journals if'seal'in r),
        job_CPU_seconds=sum(r['seal']['resource']['CPU_seconds']for r in journals if'seal'in r),
        operations=dict(sum((Counter(r['seal']['operations'])for r in journals if'seal'in r),Counter())),
        parent_process_CPU_seconds=time.process_time()-parentcpu,
        parent_max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        exited_child_max_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        memory_scope='Linux rusage parent process high-water and maximum exited-child high-water; not simultaneous aggregate RAM',
        same_shared_parentwall_count_once=True,reference_or_accuracy_reads=0)
    write(root/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json',runtime)
    if failures:raise RuntimeError('All scientific rows retained; technical failures prevent completion seal')
    result={}
    for domain,x in domains.items():
        out=x['out'];frames=x['frames'];done=x['done']
        if len(done)!=len(frames):raise ValueError('Missing frame would change denominator')
        save_rows(out/'HEAD_FEATURES.jsonl.gz',[next(rows(out/'frames'/f'{i:04d}'/'FEATURES.jsonl.gz'))for i in range(len(frames))])
        seal=dict(schema='D1_FEATURE_SEAL_V1',complete=True,rows=len(frames),head_features_sha256=sha(out/'HEAD_FEATURES.jsonl.gz'),
            input_sha256=sha(x['inputs']),observation_seal_sha256=sha(x['obs']/'OBSERVATION_SEAL.json'),source_sha256=source,
            ordered_ids=[f['id']for f in frames],e0_independently_executed=True,e0_name='E0_REGENERATED',
            same_automatic_procedure_both_domains=True,history_only_union=False,new_B_or_A_candidates=False,
            feature_meaning_unchanged=True,feature_scores_use_original_unrepaired_A_B_energies=True,
            execution_batch_id=batchid,shared_parentwall_seconds=wall,wall_seconds=wall,shared_parentwall_count_once=True,
            shared_execution_receipt_sha256=sha(root/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json'),
            per_frame_execution_seal_sha256={f'{i:04d}':sha(out/'frames'/f'{i:04d}'/'FRAME_SEAL.json')for i in range(len(frames))},
            operations=dict(sum((Counter(s['operations'])for s in done.values()),Counter())),
            sum_frame_CPU_seconds=sum(s['resource']['CPU_seconds']for s in done.values()),
            failures=[],all_rows_kept=True,before_supervision_and_PRIMARY_scoring=True,
            execution_proof=dict(technical_amendment_lock_sha256=sha(amendment),
                shared_runtime_lock_sha256=sha(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json'),
                exact_stage_memoized_rows=sum('scripts/research/pallet_boundary_action_repair_v1/d1_stage_memoization.py'in s['source_sha256']for s in done.values())))
        write(out/'FEATURE_SEAL.json',seal);result[domain]=seal
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--old-run',required=True)
    p.add_argument('--original-run',required=True);p.add_argument('--typed-run',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--root-GO',action='store_true')
    a=p.parse_args();print(canonical(run(a.private,a.old_run,a.original_run,a.typed_run,workers=a.workers,GO=a.root_GO)))
