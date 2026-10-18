"""Prediction-only D1 cost ledger; missing timings remain explicitly unknown."""
from __future__ import annotations
import argparse,json,os,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,rows,sha,write


def stamp(seconds):return datetime.fromtimestamp(seconds,timezone.utc).isoformat()
def rows_plain(path):
    with Path(path).open()as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def ledger(private,output):
    started,cpu=time.monotonic(),time.process_time()
    root=Path(private);jobs=[];unknown=[]
    def sealed_job(ident,relative,kind,workers=1):
        path=root/relative;s=read(path);wall=s.get('wall_seconds',s.get('resource',{}).get('wall_seconds'))
        jobs.append(dict(execution_batch_id=ident,kind=kind,status=s.get('status','COMPLETE'),workers=workers,
            parent_wall_seconds=wall,parent_wall_scope='original sealed function-call timer',
            started_utc=None,ended_utc=None,seal_filesystem_mtime_utc=stamp(path.stat().st_mtime),
            timestamp_limit='Original per-batch UTC start/end was not captured; seal mtime is not substituted as exact execution end',
            operations=s.get('operations'),
            source_receipt_sha256=sha(path),CPU_seconds=s.get('CPU_seconds',s.get('sum_frame_CPU_seconds',s.get('resource',{}).get('CPU_seconds')))))
        return s
    obs={d:sealed_job('D1_OBSERVE_'+d,'d1_'+d+'_observations_v1/OBSERVATION_SEAL.json','GPU_OBSERVATION')for d in('real','synthetic')}
    technical_obs=[sealed_job('D1_TECH_OBSERVE_CAL','d1_tech_real_obs_v2/OBSERVATION_SEAL.json','TECHNICAL_GPU_OBSERVATION'),
        sealed_job('D1_TECH_OBSERVE_SYNTH','d1_tech_synth_obs_v1/OBSERVATION_SEAL.json','TECHNICAL_GPU_OBSERVATION')]
    for ident,folder in [('D1_TECH_E_CAL','d1_tech_real_solve_v1'),('D1_TECH_E_SYNTH','d1_tech_synth_solve_v1'),('D1_TECH_E_V3_CAL','d1_tech_cpu_reporting_v3')]:
        sealed_job(ident,folder+'/FEATURE_SEAL.json','TECHNICAL_UNCHANGED_E_SOLVE')
    first=read(root/'D1_CACHE_PARITY_FIXED_SELECTION.json');slot=f"frames/{first['index']:04d}/FRAME_SEAL.json"
    actual=sealed_job('D1_TECH_ACTUAL_FIRST_REMAINDER','d1_cache_parity_actual/'+slot,'TECHNICAL_ACTUAL_E_REUSED_IN_FINAL_BANK')
    memo=sealed_job('D1_TECH_MEMO_FIRST_REMAINDER','d1_cache_parity_memoized/'+slot,'TECHNICAL_EXACT_CACHE_PARITY')
    if(root/'D1_FINAL_SOURCE_MEMO_PARITY.json').exists():
        current=sealed_job('D1_TECH_FINAL_FROZEN_SOURCE_MEMO','D1_FINAL_SOURCE_MEMO_PARITY.json','TECHNICAL_EXACT_CACHE_PARITY')
        if current['status']!='PASS':raise ValueError('Final frozen-source memo parity must pass')
    if(root/'D1_READ_ONLY_STACK_SNAPSHOT.json').exists():
        sealed_job('D1_READ_ONLY_STACK_SNAPSHOT','D1_READ_ONLY_STACK_SNAPSHOT.json','READ_ONLY_TECHNICAL_DIAGNOSTIC')
    profile=read(root/'D1_PERSISTENT_HOOK_PROFILE.json')
    jobs.append(dict(execution_batch_id='D1_TECH_PERSISTENT_PROFILE',kind='TECHNICAL_GPU_OBSERVATION',status=profile['status'],workers=1,
        parent_wall_seconds=profile['total_host_seconds'],CPU_seconds=profile['total_CPU_seconds'],started_utc=None,ended_utc=None,
        source_receipt_sha256=sha(root/'D1_PERSISTENT_HOOK_PROFILE.json')))
    quiesce=read(root/'D1_REAL_CONTROLLED_QUIESCE.json');startticks=quiesce['verified_processes'][0]['startticks'];hz=os.sysconf('SC_CLK_TCK')
    elapsed=quiesce['started_monotonic']-startticks/hz+quiesce['drain_wall_seconds']
    realout=root/'d1_real_candidates_v1';initial_indices=read(root/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json')['validated_completed_original_frame_indices']
    if first['index']in initial_indices:raise ValueError('Actual parity import must be disjoint from safely completed original rows')
    if sha(root/'d1_cache_parity_actual'/slot)!=sha(realout/slot):raise ValueError('Actual parity frame must be reused verbatim in the final bank')
    initial=[read(realout/'frames'/f'{i:04d}'/'FRAME_SEAL.json')for i in initial_indices]
    jobs.append(dict(execution_batch_id='D1_ORIGINAL_REAL_V3B_PARTIAL',kind='ACTUAL_UNCHANGED_E_BATCH',status='QUIESCED_ALL_STARTED_ROWS_SEALED',workers=2,
        parent_wall_seconds=None,parent_process_lifetime_upper_bound_seconds=elapsed,
        parent_wall_scope='Exact function-call end was not captured after controlled quiesce; process startticks to safe-drain end conservatively bounds it',
        started_utc=None,ended_utc=quiesce['ended_utc'],completed_rows=len(initial),started_rows_without_seal=0,active_solver_terminated=False,
        worker_job_wall_seconds=sum(s['resource']['wall_seconds']for s in initial),worker_CPU_seconds=sum(s['resource']['CPU_seconds']for s in initial),
        operations=dict(sum((Counter(s['operations'])for s in initial),Counter())),
        source_receipt_sha256=sha(root/'D1_REAL_CONTROLLED_QUIESCE.json')))
    shared=read(root/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')
    recovery_mode=(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json').exists()
    if recovery_mode:
        interrupted=read(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json')
        recovered=read(root/'D1_RECOVERY_FINALIZATION_RECEIPT.json');recovery_lock=read(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json')
        if interrupted['schema']!='D1_INTERRUPTED_SHARED_EXECUTION_V1'or interrupted['execution_batch_id']==shared['execution_batch_id']or recovered['status']!='COMPLETE':
            raise ValueError('Interrupted and resumed execution batches must be distinct and complete')
        if recovered['recovery_runtime_receipt_sha256']!=sha(root/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')or recovered['recovery_technical_amendment_lock_sha256']!=sha(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json')or recovery_lock['interrupted_execution_receipt_sha256']!=sha(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json'):
            raise ValueError('Standalone cost recovery receipt bindings failed')
        interrupted_rows=[read(root/('d1_'+r['domain']+'_candidates_v1')/'frames'/f"{r['index']:04d}"/'FRAME_SEAL.json')for r in rows_plain(root/'D1_SHARED_RUNTIME_JOURNAL.jsonl')]
        jobs.append(dict(execution_batch_id=interrupted['execution_batch_id'],kind='INTERRUPTED_UNCHANGED_E_SHARED_BATCH',
            status=interrupted['status'],workers=interrupted['workers'],parent_wall_seconds=None,
            parent_process_lifetime_upper_bound_seconds=interrupted['parent_process_lifetime_upper_bound_seconds'],
            parent_wall_scope=interrupted['parent_wall_scope'],started_utc=interrupted['started_utc'],ended_utc=None,
            first_verified_processes_absent_utc=interrupted['first_verified_processes_absent_utc'],
            observed_tool_session_exit_code=interrupted['observed_tool_session_exit_code'],worker_exit_codes=None,
            completed_rows=len(interrupted_rows),interrupted_unsealed_attempt_rows=len(interrupted['unsealed_attempts']),
            completed_worker_wall_seconds=sum(s['resource']['wall_seconds']for s in interrupted_rows),
            completed_worker_CPU_seconds=sum(s['resource']['CPU_seconds']for s in interrupted_rows),
            operations_completed_rows_only=dict(sum((Counter(s['operations'])for s in interrupted_rows),Counter())),
            interrupted_partial_operations=None,interrupted_partial_CPU_seconds=None,
            partial_count_scope='The four interrupted attempts were not sealed; their optimizer/PnP/CPU counts remain unknown, not zero. Their runtime is covered by the parent-wall upper bound.',
            last_verified_worker_resource_snapshot=interrupted['last_verified_worker_resource_snapshot'],
            source_receipt_sha256=sha(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json')))
        supervisor=read(root/'D1_RECOVERY_SUPERVISOR_RECEIPT.json')
        if supervisor['child_actual_exit_code']!=0:raise ValueError('Actual recovery process exit must pass')
        jobs.append(dict(shared,kind='RESUMED_UNCHANGED_E_SHARED_BATCH',parent_wall_seconds=supervisor['wall_seconds'],
            started_utc=supervisor['started_utc'],ended_utc=supervisor['ended_utc'],wall_seconds=supervisor['wall_seconds'],
            internal_D_run_started_utc=shared['started_utc'],internal_D_run_ended_utc=shared['ended_utc'],internal_D_run_wall_seconds=shared['wall_seconds'],
            internal_D_run_parent_process_CPU_seconds=shared['parent_process_CPU_seconds'],parent_process_CPU_seconds=None,
            supervisor_waited_child_CPU_seconds=supervisor['exited_child_CPU_seconds'],supervisor_waited_child_max_RSS_KiB=supervisor['exited_child_max_RSS_KiB'],
            CPU_scope='Internal D.run parent-only CPU and actual waited-child cumulative CPU are separate measurements; outer supervisor self-CPU was not captured.',
            parent_wall_scope=supervisor['scope'],actual_process_exit_code=supervisor['child_actual_exit_code'],
            supervisor_receipt_sha256=sha(root/'D1_RECOVERY_SUPERVISOR_RECEIPT.json')))
        sealed_job('D1_RECOVERY_PREPARATION','D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json','REFERENCE_FREE_TECHNICAL_RECOVERY')
        unknown.append('four interrupted unsealed CPU solve/feature attempts: per-attempt operations and CPU unknown; original parent-wall upper bound is retained')
    else:jobs.append(dict(shared,kind='SHARED_REAL_AND_SYNTHETIC_UNCHANGED_E_BATCH',parent_wall_seconds=shared['wall_seconds']))
    for name in ['D1_OBSERVATION_COST_AUDIT.json','D1_EXISTING_FIXED_E_STAGE_AUDIT.json','D1_CACHE_BATCH_LOADER_PARITY.json','D1_PREFLIGHT.json','D1_PREFLIGHT_V2.json','D1_REAL238_FRAME_COVERAGE_SNAPSHOT.json','D1_PROCESS_RESOURCE_SNAPSHOT_02.json','D1_BUDGET_SNAPSHOT_BEFORE_FINAL_FITS.json','DATA_PIPELINE_PARITY.json']:
        s=read(root/name);key='audit_wall_seconds'if'audit_wall_seconds'in s else'wall_seconds'
        jobs.append(dict(execution_batch_id=name.removesuffix('.json'),kind='REFERENCE_FREE_TECHNICAL_AUDIT',status=s.get('status','COMPLETE'),workers=1,
            parent_wall_seconds=s.get(key),CPU_seconds=s.get('audit_CPU_seconds',s.get('CPU_seconds')),started_utc=None,ended_utc=None,source_receipt_sha256=sha(root/name)))
    attempts=[dict(id='D1_TECH_OBSERVATION_DESCRIPTOR_PATH',status='FAILED_TECHNICAL',evidence='D1_TECHNICAL_ATTEMPTS_RETROSPECTIVE.json',raw_attempt_artifact='d1_tech_real_obs_v1/OBSERVATION_LOCK.json',raw_traceback_file=None,
            reason='Initial relative descriptor path failed before model forward',new_NN_calls=0,new_PnP_or_optimizer_calls=0),
        dict(id='D1_CPU_CLONE_KEYWORD_DEFAULTS',status='FAILED_TECHNICAL',evidence='D1_TECHNICAL_ATTEMPTS_RETROSPECTIVE.json',raw_attempt_artifact='D1_REAL_CPU_LIVE.log',raw_traceback_file=None,
            reason='Cloned Python function keyword-only defaults were absent; repaired in frozen V3B before any frame computation',new_NN_calls=0,new_PnP_or_optimizer_calls=0),
        dict(id='D1_MEMO_PARITY_ANALYSIS_ATTEMPT1',status='FAILED_TECHNICAL',evidence='D1_MEMO_PARITY_ANALYSIS_ATTEMPT1.json',new_NN_calls=0,new_PnP_or_optimizer_calls=0),
        dict(id='D1_CACHE_BATCH_LOADER_TEST_ATTEMPT1',status='FAILED_TECHNICAL_TEST_ASSERTION',evidence='D1_CACHE_BATCH_LOADER_TEST_ATTEMPT1.json',new_NN_calls=0,new_PnP_or_optimizer_calls=0)]
    for a in attempts:
        a.update(parent_wall_seconds=None,CPU_seconds=None,unknown_cost_reason='Attempt timer was not captured; not replaced with zero')
        p=root/a['evidence'];a['evidence_sha256']=sha(p)if p.exists()else None
    unknown.extend(a['id']for a in attempts)
    unknown+=['read-only source/cache equivalence audits without a timer','initial cache preparation and packaging commands without a timer','unit tests and failed/successful ad-hoc technical checks without a batch timer']
    cost=read(root/'D1_OBSERVATION_COST_AUDIT.json');direct=Counter();reconstructed=Counter()
    for d in cost['domains'].values():
        direct.update(d['direct_hook_counts']);reconstructed.update(d['SAM_actual_calls_reconstructed_from_successful_provider_receipts'])
    techdirect=Counter()
    for s in technical_obs:
        techdirect.update({k:v for k,v in s['actual_calls'].items()if isinstance(v,int)and not isinstance(v,bool)})
    for phase in profile['phases']:techdirect.update({k:v for k,v in phase['actual_calls'].items()if isinstance(v,int)and not isinstance(v,bool)})
    final={d:read(root/('d1_'+d+'_candidates_v1')/'FEATURE_SEAL.json')for d in('real','synthetic')}
    verified={d:read(root/('d1_'+d+'_candidates_v1')/'D1_EXECUTION_RECEIPT.json')for d in final}
    if read(root/'DATA_PIPELINE_PARITY.json')['status']!='PASS':raise ValueError('Overall independent D1 parity must pass')
    for d,v in verified.items():
        if not v['independent_execution_checks_passed']or not v['original_archive_bindings_checks_passed']or not all(v['frozen_cache_evidence_checks'].values())or not v['actual_model_files_SHA_verified']or not v['actual_model_bindings_identical_both_domains']or not v['actual_producer_source_files_SHA_verified']:
            raise ValueError('Final actual/memo and frozen archive bindings require independent PASS')
        if v['actual_new_vs_exact_memo_frames']['exact_original_memo']!=final[d]['execution_proof']['exact_stage_memoized_rows']:
            raise ValueError('Final memo frame count differs from independent E_RUN/journal proof')
    jobs.append(dict(execution_batch_id='D1_COST_RECEIPT_ASSEMBLY',kind='REFERENCE_FREE_TECHNICAL_AUDIT',status='COMPLETE',workers=1,
        parent_wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,started_utc=None,ended_utc=None,
        timer_scope='Receipt assembly through final source/input reads; final JSON serialization/write excluded',source_sha256=sha(__file__)))
    if len({j['execution_batch_id']for j in jobs})!=len(jobs):raise ValueError('Distinct batch-wall ledger cannot contain duplicate execution IDs')
    result=dict(schema='D1_COST_AND_EXECUTION_RECEIPT_V1',status='COMPLETE_WITH_DECLARED_MEASUREMENT_GAPS',
        unique_execution_batches=jobs,batch_id_unique=len({j['execution_batch_id']for j in jobs})==len(jobs),
        distinct_batch_wall_ledger=[dict(batch_id=j['execution_batch_id'],kind=j['kind'],
            wall_seconds=j.get('parent_wall_seconds')if j.get('parent_wall_seconds')is not None else j.get('parent_process_lifetime_upper_bound_seconds'),
            scope=j.get('parent_wall_scope',j.get('timer_scope','measured unique execution batch parent wall; simultaneous worker sums are separate')),
            wall_is_upper_bound=j.get('parent_wall_seconds')is None and j.get('parent_process_lifetime_upper_bound_seconds')is not None)for j in jobs
            if j.get('parent_wall_seconds')is not None or j.get('parent_process_lifetime_upper_bound_seconds')is not None],
        parentwall_accounting='Shared real/synthetic execution_batch_id is counted once. Concurrent worker-job sums are reported separately, not multiplied into batch wall.',
        known_parentwall_seconds=sum(j['parent_wall_seconds']for j in jobs if j.get('parent_wall_seconds')is not None),
        known_parentwall_plus_declared_process_lifetime_bounds_seconds=sum(j['parent_wall_seconds']for j in jobs if j.get('parent_wall_seconds')is not None)+sum(j.get('parent_process_lifetime_upper_bound_seconds',0.)for j in jobs if j.get('parent_wall_seconds')is None),
        exact_all_D1_CPU_and_analysis_wall_seconds=None,unmeasured_costs=unknown,technical_attempts=attempts,
        final_bank=dict(rows=sum(s['rows']for s in final.values()),domain_rows={d:s['rows']for d,s in final.items()},
            additional_interrupted_unsealed_attempt_rows=len(interrupted['unsealed_attempts'])if recovery_mode else 0,
            original_fixed_e_stage_memoized_rows=sum(v['actual_new_vs_exact_memo_frames']['exact_original_memo']for v in verified.values()),
            actual_new_unchanged_e_rows=sum(v['actual_new_vs_exact_memo_frames']['new_unchanged_e']for v in verified.values()),
            operations=dict(sum((Counter(s['operations'])for s in final.values()),Counter())),
            operations_scope='Final-bank provenance summary only; these operations are already represented in unique execution jobs and must not be added to those cost counts',
            producer_service_resource_by_domain_and_execution_mode={d:v['producer_service_resource_by_execution_mode']for d,v in verified.items()},
            producer_service_timer_scope='e + original 62-feature scoring + output serialization/provenance combined, after worker preload. Optimizer/feature stage split unmeasured; mixed original partial2-worker, actual parity1-worker and shared4-worker campaign contexts.',
            count_scope='Final bank execution receipt count; first-REMAINDER actual technical solve is imported and counted once in unique physical execution ledger, not run twice'),
        preexisting_stage_reuse=dict(cache_eligible_rows=143,
            original_execution_rows_reused=sum(s['execution_proof']['exact_stage_memoized_rows']for s in final.values()),
            one_row_executed_anew_for_exact_parity=True,
            original_execution_cost_seconds=None,original_call_counts=None,
            cost_scope='Original fixed P1/WD0/G10/G13 executions predate this campaign; current reuse costs are measured separately and original unknown cost is not zero'),
        preexisting_upstream_cache=dict(rows=1241,new_detector_or_N3_SubPix_forwards=0,
            original_cache_creation_wall_seconds=None,fresh_RGB_to_final_pose_latency_seconds=None,
            scope='Existing frozen N3/SubPix/F cache was reused. Measured observations and e/feature stages are additional costs; original upstream cache creation cost is unknown, not zero.'),
        independent_execution_validation=dict(data_pipeline_parity_sha256=sha(root/'DATA_PIPELINE_PARITY.json'),
            domain_execution_receipt_sha256={d:sha(root/('d1_'+d+'_candidates_v1')/'D1_EXECUTION_RECEIPT.json')for d in verified},
            actual_new_vs_exact_memo_frames={d:v['actual_new_vs_exact_memo_frames']for d,v in verified.items()},
            original_returned_optimizer_outcome_records_by_stage={d:v['original_returned_optimizer_outcome_records_by_stage']for d,v in verified.items()},
            diagnostic_count_scope='Returned original stage views, not unique optimizer executions or stage timings. Nested backend/family views can repeat records; later partial backend returns remain stage-unattributed. Actual newly hooked numeric calls are separate; memoization retains original diagnostic records.',
            actual_model_bindings_identical_both_domains=all(v['actual_model_bindings_identical_both_domains']for v in verified.values()),
            actual_model_files_SHA_verified=all(v['actual_model_files_SHA_verified']for v in verified.values()),
            actual_model_bindings_sha256={d:v['actual_model_bindings_sha256']for d,v in verified.items()},
            actual_producer_source_files_SHA_verified=all(v['actual_producer_source_files_SHA_verified']for v in verified.values()),
            frozen_cache_archive_and_packet_bindings_verified=all(v['original_archive_bindings_checks_passed']and all(v['frozen_cache_evidence_checks'].values())for v in verified.values())),
        observation_cost=dict(bulk_rows=1241,direct_hook_counts=dict(direct),SAM_actual_calls_reconstructed_from_provider_receipts=dict(reconstructed),
            bulk_top_level_neural_forwards=1241*6,technical_extra_observation_reexecutions=4,technical_extra_top_level_neural_forwards=24,
            technical_direct_hook_counts=dict(techdirect),state_classifier_head_is_separately_counted_CPU_stage=True,
            CUDA_events_lower_bound_seconds=sum(d['recorded_partial_CUDA_forward_seconds']for d in cost['domains'].values()),
            bulk_GPU_process_hostwall_upper_bound_seconds=cost['total_observation_GPU_process_wall_upper_bound_seconds'],
            actual_total_GPU_activity_seconds=None,missing_GPU_time_not_zero=True,
            actual_bulk_peak_allocated_GPU_bytes=max(d['peak_allocated_GPU_bytes']for d in cost['domains'].values()),
            extra_profile_receipt_sha256=sha(root/'D1_PERSISTENT_HOOK_PROFILE.json'),provider_audit_sha256=sha(root/'D1_OBSERVATION_COST_AUDIT.json')),
        memory_measurement=dict(parent_max_RSS_KiB=shared['parent_max_RSS_KiB'],exited_child_max_RSS_KiB=shared['exited_child_max_RSS_KiB'],
            scope=shared['memory_scope']+('; final resumed pool only; interrupted pool has last-verified RSS lower-bound snapshot, not an exit high-water measurement'if recovery_mode else''),simultaneous_total_RSS_bytes=None),
        reference_or_accuracy_reads=0,all_original_attempt_records_preserved=True,verifier_source_sha256=sha(__file__))
    write(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(ledger(a.private,a.output),ensure_ascii=False))
