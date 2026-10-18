"""Read-only runtime forecast, with no target/reference or pose scoring."""
from __future__ import annotations
import argparse,json,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np


def forecast(private,output):
    root=Path(private);lock=json.loads((root/'D1_SHARED_EXECUTION_BATCH_LOCK.json').read_text())
    journal_paths=[root/'D1_SHARED_RUNTIME_JOURNAL.jsonl'];active_lock=lock;interrupted=None
    if(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json').exists():
        recovery=json.loads((root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json').read_text());runtime=Path(recovery['recovery_private_root'])
        active_lock=json.loads((runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json').read_text())
        journal_paths.append(runtime/'D1_SHARED_RUNTIME_JOURNAL.jsonl')
        interrupted=json.loads((root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json').read_text())
    finished=[json.loads(line)for journal in journal_paths for line in journal.read_text().splitlines()if line.strip()]
    if len({(r['domain'],r['index'])for r in finished})!=len(finished):raise ValueError('Recovery journal overlap')
    planned=Counter((j['domain'],j['kind'])for j in lock['ordered_pending']);done=Counter();walls=defaultdict(list);cpu=defaultdict(list)
    for j in finished:
        if j.get('status')=='FAILED_TECHNICAL':continue
        p=root/('d1_'+j['domain']+'_candidates_v1')/'frames'/f"{j['index']:04d}"/'FRAME_SEAL.json'
        r=json.loads(p.read_text());key=(j['domain'],j['execution_kind']);done[key]+=1
        walls[key].append(r['resource']['wall_seconds']);cpu[key].append(r['resource']['CPU_seconds'])
    groups={};point=0.;not_estimable=[]
    for key,n in planned.items():
        sample=walls[key];label='/'.join(key);remaining=n-done[key]
        if sample:point+=remaining*float(np.mean(sample))/lock['workers']
        elif remaining:not_estimable.append(label)
        groups[label]=dict(planned_rows=n,sealed_rows=done[key],remaining_including_active_rows=remaining,sample_count=len(sample),
            sealed_job_wall_sum_seconds=sum(sample),sealed_job_CPU_sum_seconds=sum(cpu[key]),
            mean_job_wall_seconds=float(np.mean(sample))if sample else None,
            P50_job_wall_seconds=float(np.quantile(sample,.5))if sample else None,
            P90_job_wall_seconds=float(np.quantile(sample,.9))if sample else None,
            maximum_job_wall_seconds=max(sample)if sample else None,
            estimate_status='NO_SAMPLE'if not sample else'EARLY_SMALL_SAMPLE'if len(sample)<10 else'DESCRIPTIVE_NO_CONFIDENCE_GUARANTEE')
    active=[]
    for domain in ('real','synthetic'):
        p=root/('d1_'+domain+'_candidates_v1')/'frames'
        if p.exists():
            active.extend(dict(domain=domain,index=int(f.name),directory_age_seconds=time.time()-f.stat().st_mtime)
                for f in p.iterdir()if f.is_dir()and not(f/'FRAME_SEAL.json').exists())
    plan_kind={(j['domain'],j['index']):j['kind']for j in lock['ordered_pending']}
    for label,group in groups.items():
        ages=[r['directory_age_seconds']for r in active if r['domain']+'/'+plan_kind.get((r['domain'],r['index']),'UNKNOWN')==label]
        group['active_directory_ages_seconds']=ages
        group['started_job_service_mean_lower_bound_seconds']=(group['sealed_job_wall_sum_seconds']+sum(ages))/(group['sealed_rows']+len(ages))if group['sealed_rows']+len(ages)else None
    result=dict(schema='D1_REFERENCE_FREE_RUNTIME_FORECAST_V1',generated_utc=datetime.now(timezone.utc).isoformat(),
        execution_batch_id=active_lock['execution_batch_id'],workers=active_lock['workers'],groups=groups,active_unsealed_frame_directories=active,
        sealed_rows=len(finished),technical_failures=sum(j.get('status')=='FAILED_TECHNICAL'for j in finished),
        batch_elapsed_host_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(active_lock['started_utc'])).total_seconds(),
        original_interrupted_batch_id=lock['execution_batch_id']if interrupted else None,
        interrupted_parentwall_upper_bound_seconds=interrupted['parent_process_lifetime_upper_bound_seconds']if interrupted else None,
        interrupted_unsealed_attempt_rows=len(interrupted['unsealed_attempts'])if interrupted else 0,
        runtime_interruptions=1 if interrupted else 0,
        sum_sealed_worker_job_wall_seconds=sum(map(sum,walls.values())),
        estimated_remaining_parentwall_seconds=None if not_estimable else point,partial_measured_group_remaining_wall_seconds=point,
        unestimated_groups=not_estimable,
        interpretation='Descriptive current-run throughput only. The completion-only point estimate can be biased downward while slow jobs remain active; directory ages provide only a service-time lower bound, not a confidence interval. Active work already done is not subtracted. Unobserved groups are not assigned zero or extrapolated from real data. Not a pose-accuracy evaluation.',
        reference_or_accuracy_reads=0)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(forecast(a.private,a.output)))
