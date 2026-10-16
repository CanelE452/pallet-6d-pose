"""Accounting from immutable saved slots; never interpret it as full FPS."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import json
import statistics
import time

from .common import ARMS,read,write,sha


def measure(folder):
    root=Path(folder)
    slots=[read(p) for p in sorted((root/'slots').glob('*.json'))]
    ids=read(root/'REPLAY_PROTOCOL.json')['ordered_ids']
    if len(slots)!=952 or Counter(x['arm'] for x in slots)!=dict.fromkeys(ARMS,238):
        raise ValueError('Complete fixed four by238 slots required')
    identities={(i,a,ident) for i,ident in enumerate(ids) for a in ARMS}
    if {(x['index'],x['arm'],x['id']) for x in slots}!=identities:
        raise ValueError('Filename/arm/ID manifest mismatch')
    per_arm={}
    for arm in ARMS:
        values=[r for r in slots if r['arm']==arm]
        seconds=[r['elapsed_seconds'] for r in values]
        endpoint_kinds=Counter()
        for r in values:endpoint_kinds.update(c['kind'] for c in r['result'].get('generated_candidates',[]))
        per_arm[arm]=dict(top_level_slots=238,new_slots=238,
            sum_slot_elapsed_seconds=sum(seconds),worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in values),
            median_slot_elapsed_seconds=statistics.median(seconds),mean_slot_elapsed_seconds=statistics.mean(seconds),
            max_slot_elapsed_seconds=max(seconds),retained_endpoint_counts_by_kind=dict(endpoint_kinds),
            max_slot_elapsed_id=max(values,key=lambda r:r['elapsed_seconds'])['id'],
            SLSQP_candidate_records=(sum(len(r['result']['solution'].get('candidates',[])) for r in values)
                                     if arm.startswith('REOBS') else None),
            seed_least_squares_attempts=(sum(r['result']['solution'].get('search',{}).get('initialization_count',0)
                                        for r in values) if arm=='REOBS_MULTI' else 0 if arm.endswith('SINGLE') else None),
            optimizer_function_evaluations='NOT_RECORDED; endpoint/seed counts are not function-evaluation counts')
    ledger=[json.loads(line) for line in (root/'RESOURCE_LEDGER.jsonl').read_text().splitlines()]
    cpu=sum(r['elapsed_seconds'] for r in ledger if r['kind']=='cpu_solver_analysis')
    gpu=sum(r['elapsed_seconds'] for r in ledger if r['kind']=='gpu_process')
    task_wall=time.time()-read(root/'START.json')['goal_created_epoch']
    write(root/'COST.json',dict(schema='ACTUAL_FIXED_FOUR_PATH_COST_V1',top_level_new_pose_slots=952,
        unique_slot_ID_arm_pairs=952,pilot_slots_reused_by_full_run=24,additional_original_e_logging_slots=0,
        original_e_logging_shared_with_TYPED_MULTI=238,per_arm=per_arm,
        cpu_solver_analysis_stage_elapsed_seconds=cpu,gpu_process_elapsed_seconds=gpu,
        whole_task_wall_seconds=task_wall,worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in slots),
        complete_worker_cpu_measured_for_new_pose_slots=True,
        historical_source_read_and_first_failed_extraction_cpu_not_precisely_captured=True,
        input_materialization_wall_in_whole_task=True,runner_stage_timer_excludes_job_materialization=True,
        inference_new_YOLO_N3_SubPix_state_mask_forward_calls=0,new_training_updates=0,new_seed=0,
        uses_e_initialization=dict(REOBS_MULTI=True,REOBS_SINGLE=True,TYPED_MULTI=False,TYPED_SINGLE=True),
        all_diagnostics_require_original_e_pipeline_artifacts=True,full_pipeline_FPS_claimed=False,
        cost_scope='cached-observation pose replay and saved-output analysis; E0 upstream cost is additional and unchanged',
        accounting_capture='before publication; final publication wall recorded separately in push receipt',
        budget=dict(cpu_stage_seconds=7200,gpu_process_seconds=3600,heavy_workers_max=3),
        within_measured_stage_budget=cpu<=7200 and gpu<=3600,
        ledger_sha256=sha(root/'RESOURCE_LEDGER.jsonl')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();measure(a.folder)
