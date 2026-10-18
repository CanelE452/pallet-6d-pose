"""Small public audit/cost summaries from preserved private receipts, postseal."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import read, rows, write, sha, save_rows


AUDITS = ('independent_p0_v1','independent_supervision_v2','independent_f1_v1',
          'independent_b_geometry_v1','independent_visual_v1',
          'independent_training_f1_v1','independent_evaluation_v1')


def summary(run_dir,public):
    run_dir,public=Path(run_dir),Path(public)
    pred=read(run_dir/'PREDICTION_MANIFEST.json')
    ev=read(run_dir/'evaluation_v1/EVALUATION_SEAL.json')
    if pred['status']!='ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING' or ev['status']!='COMPLETE':
        raise ValueError('Closure requires actual fixed outputs and completed evaluation')
    audits=[]
    required=list(AUDITS)
    if pred['visual_executed']:
        required.append('independent_training_visual_v1')
    for name in required:
        path=run_dir/name/'VERIFICATION.json'
        if not path.is_file():
            raise FileNotFoundError('Required independent audit missing: '+name)
        value=read(path)
        if value.get('failure_count',0)!=0 or not str(value['status']).startswith('PASS'):
            raise ValueError('Audit is not PASS: '+name)
        audits.append(dict(relative_private_source=name+'/VERIFICATION.json',original_sha256=sha(path),
            **{k:value[k] for k in ('schema','source_sha256','status','checks','checks_by_kind','failure_count','independent_scope','scope','reference_scope',
                'maximum_differences','max_numerical_differences','cost','counts',
                'original_1e_10_roundtrip_failures_retained','material_owner_status',
                'matched','matched_training_validation','original_context_helper_reused_for_tiny_residual_check',
                'maximum_logit_absolute_difference','all_F0_comparisons_bitwise','checkpoint_replayed_rows',
                'tiny_checkpoint_forward_count','bulk_V_cache_generated','source_contracts',
                'CSV_rows_checked','fixed_primary_bank','numerical_expression_differences','tolerances',
                'strict_delta_direction_source','production_metric_aggregate_paired_imports',
                'real_reference_decode_after_complete_prediction_and_evaluation_seals') if k in value}))
    preservation_path=run_dir/'ORIGINAL_PRESERVATION_NORMAL_MODE_VERIFICATION.json'
    preservation=read(preservation_path)
    if preservation['status']!='PASS' or preservation['failure_count']!=0:
        raise ValueError('Original preservation same-view verification is not PASS')
    prior_path=run_dir/'ORIGINAL_PRESERVATION_VERIFICATION.json'
    prior=read(prior_path)
    if preservation['prior_failed_comparison']['sha256']!=sha(prior_path):
        raise ValueError('Preserved all-files status-view comparison changed')
    preserved_checks=preservation['checks']
    if not all(c['PASS'] for c in preserved_checks):
        raise ValueError('Original preservation check failed')
    preservation_summary=dict(
        status=preservation['status'],source_receipt_sha256=sha(preservation_path),
        START_sha256=preservation['START_sha256'],original_status_sha256=preservation['actual_status_sha256'],
        original_HEAD=preservation['original_HEAD'],local_main=preservation['local_main'],
        protected_files_checked=len(preservation['protected_files']),
        original_input_artifacts_checked=len(preservation['original_input_artifacts']),
        original_index_unchanged=preservation['original_index_before_sha256']==preservation['original_index_after_sha256'],
        prior_all_files_view_comparison=dict(status=prior['status'],failure_count=prior['failure_count'],
            receipt_sha256=sha(prior_path),actual_status_sha256=prior['actual_status_sha256'],
            explanation='START used default normal untracked-directory summaries; first comparison used all-files enumeration. Original FAIL retained; same original command/view subsequently matched exact bytes.'),
        raw_status_paths_or_bytes_published=False)
    review=read(public/'PIXEL_REVIEW.json')
    write(public/'VERIFICATION.json',dict(schema='RISK_INFORMATION_LEVELLED_VERIFICATION_V1',
        status='EXECUTION_AND_NUMERICAL_AUDITS_COMPLETE_WITH_EXPLICIT_LIMITATIONS',independent_audits=audits,
        levels=dict(input_match=dict(status='PASS',evidence='independent_p0_v1 + independent_f1_v1'),
            execution=dict(status='COMPLETE',evidence='all registered outputs before reference receipt'),
            numeric=dict(status='PASS_INDEPENDENT_IMPLEMENTATION',evidence='independent_evaluation_v1'),
            supervision=dict(status='PASS_INDEPENDENT_IMPLEMENTATION',evidence='independent_supervision_v2'),
            geometry=dict(status='SEMANTIC_PASS_WITH_ORIGINAL_STRICT_FAILURES_RETAINED',evidence='independent_b_geometry_v1'),
            original_preservation=preservation_summary,
            pixel_model_review=dict(status=review['status'],evidence='PIXEL_REVIEW.json'),
            independent_physical_truth=dict(status='NOT_PERFORMED',independent_real_test_n=0),
            remote_publication=dict(status='REPORTED_SEPARATELY_BY_ACTUAL_REMOTE_RECEIPT',evidence='PUBLICATION_RECEIPT.json')),
        prediction_manifest_sha256=sha(run_dir/'PREDICTION_MANIFEST.json'),
        evaluation_seal_sha256=sha(run_dir/'evaluation_v1/EVALUATION_SEAL.json'),
        limitations=['Repeated-development real data; no new independent real test',
            'Python audit hook is not an OS sandbox; native I/O/preloaded-object scope limited',
            'B raw float32 vs parameterized roundtrip strict assertion660/3010 preserved',
            'Material variants/scene visibility exist; per-edge mesh-face/pixel ownership unverified',
            'Model pixel review is neither human review nor physical ground-truth certification']))
    receipts=[]
    for p in sorted((run_dir/'execution').glob('*/RECEIPT.json')):
        value=read(p)
        receipts.append(dict(name=p.parent.name,original_receipt_sha256=sha(p),
            **{k:v for k,v in value.items() if k in ('status','actual_exit_code','wall_seconds',
                'child_CPU_user_seconds','child_CPU_system_seconds','child_peak_rss_kib',
                'start_epoch','end_epoch','standalone_batch_wall','CUDA_activity_seconds')}))
    fits=[]
    for p in sorted((run_dir/'training').glob('*/RUN_RECEIPT.json')):
        value=read(p)
        fits.append(dict(arm=value['arm'],seed=value['seed'],epochs=value['epochs'],best_epoch=value['best_epoch'],
            parameters=value['parameters'],checkpoint_sha256=value['checkpoint_sha256'],history_sha256=value['history_sha256'],
            cost=value['cost'],source_receipt_sha256=sha(p)))
    f1=read(run_dir/'F1_FEATURE_EXECUTION_SUMMARY.json')
    # Entire measured structure is separate from fit inner timers; never add
    # parent/child nested scopes into one misleading campaign total.
    costs=dict(schema='RISK_INFORMATION_ACTUAL_COST_SCOPES_V1',
        nonoverlapping_sum_policy='Do not sum inner fit/feature timing with containing CLI receipts; jobs may overlap in elapsed wall.',
        complete_recorded_child_processes=receipts,per_fit_inner_measurements=fits,
        F1_feature_execution_summary_sha256=sha(run_dir/'F1_FEATURE_EXECUTION_SUMMARY.json'),
        F1_distinct_batch_wall_seconds=f1['formal_batch_wall_sum_seconds'],
        F1_distinct_batch_CPU_seconds=f1['formal_CPU_sum_seconds'],
        F1_note=f1['time_scope'],
        F1_normalizer_wall_seconds=None,F1_normalizer_note='Production normalizer CLI not instrumented; unknown, not zero.',
        independent_audit_inner_measurements=[dict(name=a['relative_private_source'].split('/')[0],
            original_receipt_sha256=a['original_sha256'],cost=a['cost'])for a in audits if 'cost'in a],
        independent_audit_timer_scope='Existing audit run-function timers may exclude interpreter/import setup. These measurements overlap a whole-process launch receipt when that receipt exists; never add both.',
        original_preservation_inner_measurements=[dict(status=v['status'],original_receipt_sha256=sha(p),cost=v['cost'])
            for p,v in((prior_path,prior),(preservation_path,preservation))],
        prediction=dict(**{k:pred[k] for k in ('wall_seconds','CPU_seconds','GPU_activity_seconds',
            'GPU_activity_note','peak_rss_kib','head_forward_host_timings','shared_normalized_feature_host_seconds',
            'archived_baseline_forward_status','timing_limitation') if k in pred}),
        evaluation=dict(wall_seconds=ev.get('wall_seconds')),
        original_D1_producer_cost='REUSED_EXISTING_CACHE; incremental rerun0, original production cost is not zero',
        source_editing_or_uninstrumented_time_seconds=None,
        deployment_FPS='NOT_MEASURED; cached RTX3080 campaign timing is not Jetson deployment latency',
        extra_GPU_activity_cap_hours=8,CPU_feature_diagnostic_batch_wall_cap_hours=10,
        extra_GPU_activity_exact_seconds=None)
    additional=[]
    for name,scope in(
            ('b_reference_support_v2/B_REFERENCE_SUPPORT_AUDIT.json','Synthetic B fixed-pose producer timer after backend preload; excludes startup/preload and final seal write.'),
            ('B_SAME_OBJECTIVE_UNIQUE_FRAME_SUMMARY.json','Saved synthetic diagnosis regrouping timer; no new cost evaluation or solve.'),
            ('B_FEASIBILITY_CONFIG_RECEIPT.json','Original synthetic E_RUN configuration read-only scan; no cost evaluation or solve.'),
            ('b_reference_metadata_v1/RECEIPT.json','Original synthetic renderer metadata read-only export; not a feature input.'),
            ('independent_training_f1_v1/LAUNCH_RECEIPT.json','Whole independent F1 matched-audit process; contains the corresponding inner audit measurement.'),
            ('independent_training_visual_v1/LAUNCH_RECEIPT.json','Whole independent visual matched-audit process; contains its inner audit timer.')):
        path=run_dir/name
        if path.exists():
            value=read(path)
            numeric={k:value[k]for k in('wall_seconds','CPU_seconds','peak_rss_kib',
                'actual_command_wall_seconds','inner_audit_wall_seconds','actual_exit_code')if k in value}
            if 'cost'in value:numeric['cost']=value['cost']
            additional.append(dict(relative_private_source=name,original_receipt_sha256=sha(path),scope=scope,measurements=numeric))
    costs['additional_recorded_measurements_not_additive_with_containing_launches']=additional
    integrated=run_dir/'TRAINING_COST_AND_EXECUTION.json'
    if integrated.exists():
        value=read(integrated)
        costs['integrated_training_whole_process_ledger']=dict(original_sha256=sha(integrated),
            public_evidence='TRAINING_COST_AND_EXECUTION.json',
            scope='Eight disjoint training/probe CLI batches; do not add again to per-fit, R_F0/R_F1/V or probe inner timers.',
            measured_values={k:v for k,v in value.items() if isinstance(v,(int,float,bool))or v is None})
    for name in ('R_F0_COST_AND_EXECUTION.json','R_F1_COST_AND_EXECUTION.json','V_TRAINING_EXECUTION_RECEIPT.json'):
        path=run_dir/name
        if path.exists():
            value=read(path)
            costs[name]=dict(original_sha256=sha(path),**{k:value[k] for k in (
                'whole_cli_wall_seconds','measured_whole_CLI_wall_subtotal_seconds','measured_whole_CLI_CPU_seconds',
                'wall_sum_seconds','actual_optimizer_attempts','successful_new_fits','successful_fits',
                'failed_optimizer_fits','non_optimizer_failed_CLI_batches','new_optimizer_attempts',
                'GPU_host_wall_upper_bound_seconds','CPU_seconds','exact_CUDA_activity_seconds',
                'GPU_time_scope','GPU_peak_allocated_bytes','peak_rss_kib') if k in value})
    write(public/'COST_AND_EXECUTION.json',costs)
    source_log=run_dir/'RUN_LOG.jsonl'
    events=[]
    if source_log.exists():
        for value in rows(source_log):
            events.append({k:value[k] for k in ('event','operation','epoch','wall_seconds','CPU_seconds',
                'exit_code','arm','seed','status') if k in value})
    save_rows(public/'RUN_LOG.jsonl',events)
    return costs


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();summary(a.run_dir,a.public)
