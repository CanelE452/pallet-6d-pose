"""Immutable plan, artifact-based resume state, budget checks and scoring gate.

Long-running branches use their own fixed-input resumable commands. A PID or
directory alone is never a completion signal. This controller binds their
actual output seals before the separate evaluator is permitted to read GT.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read, rows, sha, write
from .registry import ALL_ARMS, B_ARMS, L_GROUPS, SEEDS, evaluation_protocol


def freeze(run_dir):
    run_dir=Path(run_dir);draft=read(run_dir/'PLAN_DRAFT.json')
    receipts=['P0_BASELINE_INPUT_INVENTORY.json','B_SOURCE_LOCK_v2.json','B_LOCAL_REGULARIZATION.json',
        'D1_PREFLIGHT_V2.json','L_D0_TRAINING_GO_LOCK.json','LEARNING_PLAN.json']
    if not all((run_dir/name).exists()for name in receipts):
        raise ValueError('Method receipts not ready: '+str([n for n in receipts if not(run_dir/n).exists()]))
    here=Path(__file__).parent
    names=['finite_boundary.py','finite_solver.py','finite_batch.py','d1_batch.py','d1_producer.py',
        'pipeline_contract.py','action_data.py','action_models.py','train_actions.py','predict_actions.py','registry.py']
    sources={p.name:sha(p)for p in (here/n for n in names)}
    if (here/'d1_cpu_logging_v3.py').exists():sources['d1_cpu_logging_v3.py']=sha(here/'d1_cpu_logging_v3.py')
    protocol=evaluation_protocol();write(run_dir/'EVALUATION_PROTOCOL.json',protocol)
    plan=dict(draft,status='FROZEN_BEFORE_NEW_PRIMARY_SCORING',frozen_epoch=time.time(),
        input_contracts_pending=[],new_primary_scoring_allowed=False,
        method_source_sha256=sources,method_receipts_sha256={n:sha(run_dir/n)for n in receipts},
        evaluation_protocol_sha256=sha(run_dir/'EVALUATION_PROTOCOL.json'),arms=ALL_ARMS,
        technical_amendments='preserve original failed attempts; no scientific redesign after PRIMARY; reporting-only source additions bound separately',
        public_rgb_policy='publish only verified rights; otherwise numeric/geometry views with exact missing reason, private RGB gallery retained')
    write(run_dir/'PLAN_LOCK.json',plan)
    packages={}
    for name in ('numpy','scipy','torch','opencv-python','Pillow','matplotlib','ultralytics'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='not installed under that distribution name'
    write(run_dir/'ENVIRONMENT.json',dict(python=platform.python_version(),executable=os.sys.executable,
        platform=platform.platform(),packages=packages,threads={k:os.environ.get(k)for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        CUDA_measurement='actual per-network CUDA events in observation receipts; no inferred Jetson FPS',
        git_version=subprocess.check_output(['git','--version'],text=True).strip()))
    return plan


def progress(run_dir):
    root=Path(run_dir);b=root/'b_finite_v1';d={}
    for label,folder,seal in [('B',b,'BATCH_SEAL.json'),
        ('D1_REAL_OBSERVATIONS',root/'d1_real_observations_v1','OBSERVATION_SEAL.json'),
        ('D1_SYNTHETIC_OBSERVATIONS',root/'d1_synthetic_observations_v1','OBSERVATION_SEAL.json'),
        ('D1_REAL_CANDIDATES',root/'d1_real_candidates_v1','FEATURE_SEAL.json'),
        ('D1_SYNTHETIC_CANDIDATES',root/'d1_synthetic_candidates_v1','FEATURE_SEAL.json'),
        ('ALL_PREDICTIONS',root,'PREDICTION_SEAL.json')]:
        path=folder/seal
        count=len(list((folder/'frames').glob('*/FRAME_SEAL.json')))
        if not count:count=len(list((folder/'frames').glob('*/OBSERVATION_SEAL.json')))
        value=dict(status='RUNNING_OR_NOT_STARTED',sealed_frames=count)
        if path.exists():
            s=read(path);value.update(status='COMPLETED'if s.get('complete')else'INCOMPLETE',seal_sha256=sha(path),rows=s.get('rows'))
        failures=list((folder/'frames').glob('*/FAILURE.json'))+list((folder/'frames').glob('*/OBSERVATION_FAILURE.json'))
        value['failure_files']=len(failures);d[label]=value
    manifest=root/'action_training_v1/TRAINING_MANIFEST.json'
    if manifest.exists():
        runs=read(manifest).get('runs',[]);d['L']=dict(status='COMPLETED'if len(runs)==21 else'PARTIAL',
            sealed_runs=len(runs),expected_runs=21,new_completed=sum(not r.get('checkpoint_reused',False)for r in runs),
            reused=sum(r.get('checkpoint_reused',False)for r in runs))
    return dict(schema='ARTIFACT_BASED_RUN_STATE_V1',epoch=time.time(),branches=d,
        primary_reference_opened=(root/'evaluation_v1/REFERENCE_OPEN_RECEIPT.json').exists())


def budget(run_dir):
    root=Path(run_dir);plan=read(root/'PLAN_LOCK.json')if(root/'PLAN_LOCK.json').exists()else read(root/'PLAN_DRAFT.json')
    limit=plan['limits'];disk=sum(p.stat().st_size for p in root.rglob('*')if p.is_file()and not p.is_symlink())
    solver_seconds=0.;forward_seconds=0.;solve_calls=Counter();observed=0
    batch_times={};GPU_host_upper_bound=0.;seen_batches={}
    for name,folder,sealname,clockkey,lockname in [
        ('B',root/'b_finite_v1','BATCH_SEAL.json','elapsed_seconds','BATCH_LOCK.json'),
        ('D1_REAL_OBSERVATIONS',root/'d1_real_observations_v1','OBSERVATION_SEAL.json','wall_seconds','OBSERVATION_LOCK.json'),
        ('D1_SYNTHETIC_OBSERVATIONS',root/'d1_synthetic_observations_v1','OBSERVATION_SEAL.json','wall_seconds','OBSERVATION_LOCK.json'),
        ('D1_REAL_CANDIDATES',root/'d1_real_candidates_v1','FEATURE_SEAL.json','wall_seconds','BATCH_LOCK.json'),
        ('D1_SYNTHETIC_CANDIDATES',root/'d1_synthetic_candidates_v1','FEATURE_SEAL.json','wall_seconds','BATCH_LOCK.json')]:
        if(folder/sealname).exists():
            s=read(folder/sealname);seconds=s[clockkey];measured=True
            batch_id=s.get('execution_batch_id',name)
            if batch_id in seen_batches:
                previous=seen_batches[batch_id]
                if seconds!=previous['seconds']:raise ValueError('Shared batch has different wall values')
                previous.setdefault('shared_seals',[]).append(dict(name=name,sha256=sha(folder/sealname)))
                continue
        elif(folder/lockname).exists():seconds=time.time()-(folder/lockname).stat().st_mtime;measured=False
        else:continue
        batch_times[name]=dict(seconds=seconds,complete_measured_batch_wall=measured,
            pending_clock='elapsed since immutable batch-lock creation; completed receipt is authoritative'if not measured else None)
        if measured:seen_batches[batch_id]=batch_times[name]
        if 'OBSERVATIONS'in name:GPU_host_upper_bound+=seconds
    for directory,filename in [('evaluation_v1','EVALUATION_SEAL.json'),('independent_numeric_v1','INDEPENDENT_CHECK.json')]:
        p=root/directory/filename
        if p.exists():
            s=read(p);batch_times[directory]=dict(seconds=s['cost']['wall_seconds'],complete_measured_batch_wall=True)
    # Final producer ledger replaces candidate envelopes and includes drained/
    # separately profiled actual jobs; it must never sum per-worker frame walls.
    d1_receipt=root/'D1_COST_AND_EXECUTION_RECEIPT.json'
    if d1_receipt.exists():
        d1=read(d1_receipt)
        ledger=d1.get('distinct_batch_wall_ledger')
        if ledger is not None:
            for name in list(batch_times):
                if name.startswith('D1_'):del batch_times[name]
            ids=set()
            for item in ledger:
                key=item['batch_id']
                if key in ids:raise ValueError('D1 distinct batch ledger duplicate: '+key)
                ids.add(key)
                batch_times['D1_LEDGER:'+key]=dict(seconds=item['wall_seconds'],complete_measured_batch_wall=not item.get('wall_is_upper_bound',False),
                    wall_is_upper_bound=item.get('wall_is_upper_bound',False),receipt_sha256=sha(d1_receipt),scope=item.get('scope'))
    parent_batch_seconds=sum(v['seconds']for v in batch_times.values())
    final_cost_path=root/'COST_AND_EXECUTION.json'
    final_cost=None
    if final_cost_path.exists():
        final_cost=read(final_cost_path)
        parent_batch_seconds=final_cost['measured_batch_wall_sum_seconds']
        if final_cost.get('GPU_full_process_host_seconds_upper_bound')is not None:
            GPU_host_upper_bound=final_cost['GPU_full_process_host_seconds_upper_bound']
    for directory in ('b_finite_v1','d1_real_candidates_v1','d1_synthetic_candidates_v1'):
        for p in (root/directory/'frames').glob('*/FRAME_SEAL.json'):
            receipt=read(p);r=receipt['resource'];solver_seconds+=r.get('elapsed_seconds',r.get('wall_seconds',0.))
            solve_calls.update(receipt.get('operations',{}))
            solve_calls['B_SLSQP_attempts']+=r.get('new_optimizer_attempts',0)
    for directory in ('d1_real_observations_v1','d1_synthetic_observations_v1'):
        for p in(root/directory/'frames').glob('*/OBSERVATION_SEAL.json'):
            receipt=read(p);r=receipt['resource'];solver_seconds+=r.get('wall_seconds',0.)
            forward_seconds+=r.get('CUDA_forward_seconds',0.);observed+=1
    attempts=root/'action_training_v1/TRAINING_ATTEMPTS.jsonl'
    training_attempts=sum(1 for _ in rows(attempts))if attempts.exists()else 0
    elapsed=time.time()-plan['campaign_start_epoch']
    checks=dict(wall=elapsed<=limit['campaign_wall_hours']*3600,
        GPU_host_upper_bound=GPU_host_upper_bound<=limit['GPU_activity_hours']*3600,
        batch_wall_sum=parent_batch_seconds<=limit['solver_analysis_job_wall_sum_hours']*3600,
        training_attempts=training_attempts<=limit['new_training_runs'],disk=disk<=limit['new_artifact_GiB']*1024**3)
    return dict(schema='LIVE_BUDGET_V1',epoch=time.time(),status='PASS'if all(checks.values())else'BUDGET_EXHAUSTED',
        checks=checks,campaign_elapsed_seconds=elapsed,production_batch_wall_sum_seconds=parent_batch_seconds,
        batch_times=batch_times,worker_job_wall_sum_seconds=solver_seconds,
        final_cost_receipt_sha256=sha(final_cost_path)if final_cost is not None else None,
        final_cost_overrides_production_subtotal=final_cost is not None,
        budget_unit='sum of distinct batch parent wall times; concurrent batches add; worker-time sum is separately reported, not CPU core-hours limit',
        accounting_scope='primary production and completed numeric analysis; calibration, training, technical failures, other analysis and rendering are added in final cost report',
        GPU_forward_event_seconds_lower_bound=forward_seconds,GPU_observation_host_seconds_upper_bound=GPU_host_upper_bound,
        GPU_capture_gap='SAM persistent predictor hooks missing after first frame; full batch host wall bounds all GPU activity; separate exact cold/warm profile available',
        new_training_attempts=training_attempts,
        new_artifact_bytes=disk,actual_observation_frames=observed,solve_calls=dict(solve_calls),limits=limit)


def scoring_gate(run_dir):
    root=Path(run_dir);state=progress(root);b=budget(root)
    expected=['B','D1_REAL_OBSERVATIONS','D1_SYNTHETIC_OBSERVATIONS','D1_REAL_CANDIDATES','D1_SYNTHETIC_CANDIDATES','ALL_PREDICTIONS','L']
    if any(state['branches'].get(k,{}).get('status')!='COMPLETED'for k in expected):raise ValueError('Every registered output must be complete before scoring')
    if b['status']!='PASS':raise ValueError('Resource budget exhausted')
    plan=read(root/'PLAN_LOCK.json');here=Path(__file__).parent
    for name,digest in plan['method_source_sha256'].items():
        if sha(here/name)!=digest:raise ValueError('Frozen science source changed: '+name)
    for name,digest in plan['method_receipts_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen method receipt changed: '+name)
    if sha(root/'EVALUATION_PROTOCOL.json')!=plan['evaluation_protocol_sha256']:raise ValueError('Frozen evaluation protocol bytes changed')
    amendment=root/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json'
    if not amendment.exists():raise ValueError('Exact stage memoization amendment lock missing')
    change=read(amendment);repo=here.parents[2]
    for field in ('additional_source_sha256','frozen_V3_source_sha256'):
        for name,digest in change[field].items():
            if sha(repo/name)!=digest:raise ValueError('Memoization source binding changed: '+name)
    for name,digest in change['evidence_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Memoization evidence changed: '+name)
    from .recovery_gate import verify_recovery_for_scoring
    recovery_bindings=verify_recovery_for_scoring(root,repo)
    if not (root/'RESOURCE_BUDGET_UNIT_LOCK.json').exists():raise ValueError('Resource unit clarification missing')
    manifest=read(root/'action_training_v1/TRAINING_MANIFEST.json');runs=manifest['runs']
    if {(r['arm'],r['seed'])for r in runs}!={(g,s)for g in L_GROUPS for s in SEEDS}:raise ValueError('Full seven groups and same3seeds required')
    if sum(not r.get('checkpoint_reused',False)for r in runs)!=18:raise ValueError('Unexpected new training count')
    if b['new_training_attempts']!=18:raise ValueError('Failed training attempts must count toward cap')
    for r in runs:
        if sha(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    seal=read(root/'PREDICTION_SEAL.json')
    if sha(root/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:raise ValueError('Prediction bytes changed')
    if seal['plan_lock_sha256']!=sha(root/'PLAN_LOCK.json')or seal['evaluation_protocol_sha256']!=sha(root/'EVALUATION_PROTOCOL.json'):
        raise ValueError('Combined output seal bound to another plan/protocol')
    receipt=dict(schema='ALL_OUTPUTS_SCORING_GO_V1',all_registered_predictions_sealed=True,
        prediction_seal_sha256=sha(root/'PREDICTION_SEAL.json'),plan_sha256=sha(root/'PLAN_LOCK.json'),
        evaluation_protocol_sha256=sha(root/'EVALUATION_PROTOCOL.json'),new_primary_reference_reads_before_gate=0,
        actual_training_attempts=18,reused_checkpoints=3,epoch=time.time(),runtime_budget=b,
        cache_amendment_sha256=sha(amendment),resource_unit_clarification_sha256=sha(root/'RESOURCE_BUDGET_UNIT_LOCK.json'),
        recovery_bindings=recovery_bindings,
        source_sha256=sha(__file__),output_seals={k:v.get('seal_sha256')for k,v in state['branches'].items()})
    write(root/'ALL_OUTPUTS_SCORING_GO.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','status','budget','scoring-go']);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    f={'freeze':freeze,'status':progress,'budget':budget,'scoring-go':scoring_gate}[a.stage]
    print(json.dumps(f(a.run_dir),ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__':main()
