"""Fixed local C2 contrasts and support diagnostics from complete saved scores.

No models, geometry fitting or new bootstrap draws are executed. Numerical
LOCAL NEW, weak/strong saved inlier support, actual pose error and contribution
of unused lines are distinct fields and populations. V7 scalar/posthoc kernels
are reused unchanged with a method-label lookup adapter only.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
from pathlib import Path
import time
import json

import numpy as np

from . import common as C
from ..pallet_three_head_observation_20261010_v7 import statistics as V7

METRICS,SCOPES = V7.METRICS,V7.SCOPES
BOOTSTRAP = C.REPO/'_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz'
ALL_METHODS = ('BASE','N3_SUBPIX',C.PRIMARY,C.COMPARATOR)


def compact_rows(path):
    for row in C.rows(path):
        yield V7.compact_saved_witnesses(row)


def outcome(row,comparison):
    if not (row['pose']['available'] and comparison['pose']['available']):
        return dict(category='POSE_UNAVAILABLE',delta=None)
    delta = {key:V7.metric(row,key)-V7.metric(comparison,key) for key in METRICS}
    T,R = delta['translation_cm'],delta['rotation_deg']
    category = 'BOTH_BETTER' if T < 0 and R < 0 else 'BOTH_WORSE' if T > 0 and R > 0 else 'MIXED_OR_UNCHANGED'
    return dict(category=category,delta=delta)


def local_fields(row,point,n3):
    solver = row['solver']
    U,lines = list(solver['used']),list(solver['line_edges'])
    pool = solver.get('factor_pool')
    new = bool(row['new_pose_estimated'])
    whole = solver.get('point_line_geometry')
    inliers = solver.get('diagnostic_inlier_geometry')
    support_scalar = solver.get('diagnostic_inlier_scalar_residuals')
    support_rank6 = solver.get('diagnostic_inlier_support_rank6')
    if new:
        C.require(solver['local_refinement_estimated'] is True and solver['initial_pose_start_used'] is True and
            solver['global_uniqueness_proven'] is False and solver['initial_pose_residual_prior'] is False and
            solver['inlier_geometry_is_acceptance_gate'] is False and type(support_rank6) is bool and
            type(support_scalar) is int,'accepted local diagnostic support schema required')
        C.require(whole['observed_normal_joint']['numerical_rank'] == 6 and
            whole['modeled_normal_joint']['numerical_rank'] == 6,'accepted whole actual pool rank6')
        C.require(pool['point_ids'] == U and pool['line_edges'] == lines and
            pool['scalar_residuals'] == 2*(len(U)+len(lines)),'actual factor pool identity')
    support = ('NOT_NUMERIC_NEW' if not new else 'FULL_RANK_8PX_INLIER_SUPPORT' if
        support_scalar >= 6 and support_rank6 else 'WEAK_OR_UNAVAILABLE_8PX_INLIER_SUPPORT')
    return dict(actual_point_ids=U,actual_unused_line_edges=lines,point_count=len(U),line_count=len(lines),
        factor_pool=pool,full_actual_pool_observed_rank=whole['observed_normal_joint']['numerical_rank'] if whole else None,
        full_actual_pool_modeled_rank=whole['modeled_normal_joint']['numerical_rank'] if whole else None,
        numeric_LOCAL_NEW=new,point_comparator_NEW=point['new_pose_estimated'],
        numeric_rescue_over_point=bool(new and not point['new_pose_estimated']),
        fewer_than4_points_and_actual_lines=bool(len(U)<4 and lines),
        no_unused_lines_point_LOCAL_only=not lines,
        diagnostic_inlier_point_ids=list(solver['final_inliers']),
        diagnostic_inlier_line_edges=list(solver.get('final_line_inliers',[])),
        diagnostic_inlier_scalar_residuals=support_scalar,diagnostic_inlier_support_rank6=support_rank6,
        diagnostic_support_category=support,
        diagnostic_inlier_observed_rank=inliers['observed_normal_joint']['numerical_rank'] if inliers else None,
        diagnostic_inlier_modeled_rank=inliers['modeled_normal_joint']['numerical_rank'] if inliers else None,
        diagnostic_inlier_geometry_unavailable=solver.get('diagnostic_inlier_geometry_unavailable'),
        initial_pose_used_as_LOCAL_start=solver['initial_pose_start_used'],
        initial_pose_residual_prior=False,global_uniqueness_proven=False,
        numeric_NEW_is_not_pose_accuracy_success=True,
        actual_pose_vs_point=outcome(row,point),actual_pose_vs_N3=outcome(row,n3),
        line_effect_causally_isolated=False,
        causal_limit='line-present and line-zero populations differ; no same-pool line-removal ablation is added')


def posthoc(row,n3,point,observation):
    adapted = dict(row,method=C.PARENT_PRIMARY)
    result = V7.posthoc(adapted,n3,observation)
    result['method'] = row['method']
    result['posthoc_kernel_label_adapter_only'] = True
    result['local_point_line_diagnostics'] = local_fields(row,point,n3)
    return result


def local_summary(rows,diagnostics):
    lookup = {row['id']:row for row in rows}
    def group(predicate):
        selected = [r for r in diagnostics if predicate(r['local_point_line_diagnostics'])]
        ids = [r['id'] for r in selected]
        actual = [lookup[fid] for fid in ids]
        return dict(frames=len(ids),ids=ids,saved_actual_pose_and_status=V7.summaries(actual),
            actual_pose_outcomes_vs_N3=dict(Counter(r['local_point_line_diagnostics']['actual_pose_vs_N3']['category'] for r in selected)),
            actual_pose_outcomes_vs_point=dict(Counter(r['local_point_line_diagnostics']['actual_pose_vs_point']['category'] for r in selected)),
            actual_pose_outcome_IDs_vs_N3={name:[r['id'] for r in selected if
                r['local_point_line_diagnostics']['actual_pose_vs_N3']['category'] == name]
                for name in ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE')},
            actual_pose_outcome_IDs_vs_point={name:[r['id'] for r in selected if
                r['local_point_line_diagnostics']['actual_pose_vs_point']['category'] == name]
                for name in ('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE')})
    predicates = dict(ALL=lambda x:True,
        ACTUAL_UNUSED_LINES_POSITIVE=lambda x:x['line_count']>0,
        UNUSED_LINES_ZERO_POINT_LOCAL=lambda x:x['line_count']==0,
        POINTS_LT4_WITH_ACTUAL_LINES=lambda x:x['point_count']<4 and x['line_count']>0,
        POINTS_LT4_WITHOUT_LINES=lambda x:x['point_count']<4 and x['line_count']==0,
        NUMERIC_RESCUE_OVER_POINT=lambda x:x['numeric_rescue_over_point'],
        NUMERIC_RESCUE_POINTS_LT4_WITH_LINES=lambda x:x['numeric_rescue_over_point'] and x['point_count']<4 and x['line_count']>0,
        NUMERIC_NEW_LINES_POSITIVE=lambda x:x['numeric_LOCAL_NEW'] and x['line_count']>0,
        NUMERIC_NEW_LINES_ZERO_POINT_LOCAL=lambda x:x['numeric_LOCAL_NEW'] and x['line_count']==0,
        NUMERIC_NEW_WITH_FULL_RANK_8PX_SUPPORT=lambda x:x['diagnostic_support_category']=='FULL_RANK_8PX_INLIER_SUPPORT',
        NUMERIC_NEW_WITH_WEAK_8PX_SUPPORT=lambda x:x['diagnostic_support_category']=='WEAK_OR_UNAVAILABLE_8PX_INLIER_SUPPORT')
    return dict(groups={name:group(predicate) for name,predicate in predicates.items()},
        actual_point_count_histogram=dict(Counter(r['local_point_line_diagnostics']['point_count'] for r in diagnostics)),
        actual_unused_line_count_histogram=dict(Counter(r['local_point_line_diagnostics']['line_count'] for r in diagnostics)),
        support_weakness_is_not_automatic_frame_failure=True,numeric_NEW_is_not_goal_success=True,
        line_zero_point_LOCAL_is_not_line_effect=True,global_uniqueness_proven=False,
        no_new_support_rank_or_inlier_acceptance_gate=True,
        local_group_comparison_is_observational_not_new_causal_ablation=True)


def compute(new,point,fixed,observations,ledgers,cohort,bootstrap):
    ids = list(cohort['ids'])
    C.require(len(ids) == len(set(ids)) == 245 and len(new) == len(point) == len(ledgers) == 245 and
        len(fixed) == 490,'complete local/comparator/fixed scored population')
    groups = {method:{} for method in ALL_METHODS}
    for row in new+point+fixed:
        C.require(row['method'] in groups and row['id'] not in groups[row['method']],'unknown/duplicate score identity')
        groups[row['method']][row['id']] = row
    C.require(all(set(g) == set(ids) for g in groups.values()),'all four methods retain same245')
    labels = {r['id']:r['label'] for r in cohort['frames']}
    C.require(Counter(labels.values()) == {'clean':153,'moderate':92},'fixed difficulty strata')
    sessions = bootstrap['sessions'];draws = np.asarray(bootstrap['counts'],np.int64)
    C.require(draws.shape == (10000,13) and len(set(sessions)) == 13 and
        (draws >= 0).all() and (draws.sum(1) == 13).all(),'existing13-session10000draws')
    index = {name:i for i,name in enumerate(sessions)}
    C.require(all(r['session'] in index for r in new+point+fixed),'known bootstrap sessions')
    obs = {r['id']:r for r in observations if r['head_arm'] == 'IMAGE_ROLE'}
    C.require(len(observations) == 735 and len(obs) == 245 and set(obs) == set(ids) and
        all(r['GT_input'] is False for r in observations),'unchanged full ROLE observation population')
    C.require([r['id'] for r in ledgers] == ids and all(r['parent_unchanged'] for r in ledgers),'unchanged local ledgers')
    diagnostics = [posthoc(r,groups['N3_SUBPIX'][r['id']],groups[C.COMPARATOR][r['id']],obs[r['id']]) for r in new]
    strata = {}
    for name,selected in (('combined',ids),('easy',[fid for fid in ids if labels[fid]=='clean']),
        ('medium',[fid for fid in ids if labels[fid]=='moderate'])):
        selected_set = set(selected)
        rows = [groups[C.PRIMARY][fid] for fid in selected]
        diag = [r for r in diagnostics if r['id'] in selected_set]
        strata[name] = dict(frames=len(selected),ids=selected,
            methods={method:V7.summaries([groups[method][fid] for fid in selected]) for method in ALL_METHODS},
            contrasts={a+'_minus_'+b:{scope:V7.paired(groups[a],groups[b],selected,index,draws,scope)
                for scope in SCOPES} for a,b in C.CONTRASTS},
            diagnostics={C.PRIMARY:V7.diagnostic_summary(diag)},local_point_line_diagnostics=local_summary(rows,diag))
    flags = {}
    for comparator in ('N3_SUBPIX',C.COMPARATOR,'BASE'):
        pair = strata['combined']['contrasts'][C.PRIMARY+'_minus_'+comparator]['common_operational']
        complete = pair['common_frames'] == 245
        flags[comparator] = dict(all245_common_outputs=complete,
            mean_T_and_R_both_lower=bool(complete and all(pair['metrics'][key]['mean_delta']<0
                for key in ('translation_cm','rotation_deg'))),
            mean_pose_error_claim_requires_full_operational_population=True)
    return dict(schema='fixed_sparse_local_point_line_saved_row_statistics_v9',complete=True,
        population=dict(frames=245,clean=153,moderate=92,severe_excluded=74,sessions=13,all_frames_retained=True),
        primary_method=C.PRIMARY,comparator=C.COMPARATOR,new_accuracy_primary=False,
        methods=strata['combined']['methods'],contrasts=strata['combined']['contrasts'],strata=strata,
        verdict=dict(full_operational_comparisons=flags,numeric_LOCAL_NEW_is_not_goal_success=True,
            independent_generalization_established=False,global_uniqueness_proven=False,
            existing_geometric_proxy_mean_improvement_is_not_independent_physical_truth=True),
        reference='Known GEOMETRIC_PROXY DEV; independent physical truth and unseen generalization not validated',
        bootstrap=dict(resamples=10000,sessions=sessions,new_draws_generated=0,
            serialized_raw_sha256=bootstrap['serialized_raw_sha256']),
        actual_arithmetic=dict(saved_scored_rows=980,local_rows=245,point_control_rows=245,fixed_rows=490,
            posthoc_rows=245,source_observation_rows=735,control_ledger_rows=245,
            methods=4,strata=3,contrasts_per_stratum=3,paired_scopes=3,paired_groups=27,metric_CIs_requested=81,
            moment_distributions=108,moment_scalars=648),
        borrowed_unchanged_kernels=['V7.moments','V7.summaries','V7.paired','V7.posthoc with label lookup adapter','V7.diagnostic_summary'],
        local_full_pool_and_diagnostic_inlier_support_distinct=True,
        no_performance_based_threshold_or_support_gate_changes=True,
        new_detector_N3_head_PnP_local_optimizer_ray_training_RGB_scoring_calls=0),diagnostics


def run(args):
    from .evaluator import sealed
    C.verify_protocol(args);C.protect(args)
    folder = Path(args.input)
    C.require(folder.resolve() == Path(args.output).resolve(),'statistics input/output must identify same completed V9 evidence')
    seal,_,validation,_,_ = sealed(args)
    receipt = C.read(folder/'SCORING_RECEIPT.json')
    C.require(receipt['complete'] is True and receipt['scored_method_rows'] == receipt['comparator_rows'] == 245 and
        receipt['fixed_control_rows'] == 490 and receipt['methods'] == list(C.METHODS) and
        receipt['comparator'] == C.COMPARATOR and receipt['GT_access_only_after_complete_seal_cleanup_and_validation'] is True and
        receipt['independent_geometry_validation'] == validation and receipt['new_local_optimizers'] == 0,
        'complete unchanged local/control score receipt')
    for key,name in (('geometry_seal','GEOMETRY_SEAL.json'),('control_receipt','CONTROL_RECEIPT.json'),
        ('predictions','PREDICTIONS.jsonl.gz'),('comparator_predictions','COMPARATOR_PREDICTIONS.jsonl.gz'),
        ('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz'),('parity','SCORING_PARITY.json')):
        C.bound(folder/name,receipt[key],'scored local '+key)
    parity = C.read(folder/'SCORING_PARITY.json')
    C.require(parity['passed'] and parity['fixed_rows'] == 490 and parity['point_control_rows'] == 245 and
        parity['total_rows'] == 735,'all735 existing score parities')
    for name in ('STATISTICS_STARTED.json','METRICS.json','POSTHOC_ROWS.jsonl.gz','METRICS.csv','STATISTICS_FAILURE.json',
        'PENDING_POSTHOC_ROWS.jsonl.gz','INTERRUPTED_POSTHOC_ROWS.jsonl.gz'):
        C.output_path(args,name)
    start = time.monotonic()
    C.write_new(C.output_path(args,'STATISTICS_STARTED.json'),dict(protocol=C.binding(args.protocol),
        scoring_receipt=C.binding(folder/'SCORING_RECEIPT.json'),new_model_PnP_local_optimizer_score_draws=0))
    try:
        new,point,fixed = [list(compact_rows(folder/name)) for name in
            ('PREDICTIONS.jsonl.gz','COMPARATOR_PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz')]
        observations = list(compact_rows(Path(args.parent)/'OBSERVATIONS.jsonl.gz'))
        ledgers = list(C.rows(folder/'CONTROL_LEDGERS.jsonl.gz'))
        with gzip.open(BOOTSTRAP,'rt',encoding='utf-8') as stream:
            bootstrap = json.load(stream)
        result,diagnostics = compute(new,point,fixed,observations,ledgers,C.read(args.cohort),bootstrap)
        dependencies = dict(new_scored_rows=folder/'PREDICTIONS.jsonl.gz',comparator_scored_rows=folder/'COMPARATOR_PREDICTIONS.jsonl.gz',
            fixed_scored_rows=folder/'FIXED_PREDICTIONS.jsonl.gz',scoring_receipt=folder/'SCORING_RECEIPT.json',
            geometry_seal=folder/'GEOMETRY_SEAL.json',control_receipt=folder/'CONTROL_RECEIPT.json',
            control_ledgers=folder/'CONTROL_LEDGERS.jsonl.gz',scoring_parity=folder/'SCORING_PARITY.json',
            observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',validation_protocol=folder/'VALIDATION_PROTOCOL.json',
            validation_receipt=folder/'VALIDATION_CHECKS.json',statistics_code=Path(__file__),
            borrowed_statistics_code=Path(V7.__file__),protocol=Path(args.protocol),cohort=Path(args.cohort),bootstrap=BOOTSTRAP)
        result['bindings'] = {name:C.binding(path) for name,path in dependencies.items()}
        result['saved_row_memory_policy'] = dict(original_full_serialized_rows_preserved=True,
            complete_SHA_checked_before_loading=True,unused_heavy_candidate_arrays_removed_only_in_RAM=True)
        from ..pallet_three_head_observation_20261010_v7.run import RowWriter
        writer = RowWriter(C.output_path(args,'POSTHOC_ROWS.jsonl.gz'),
            interrupted_path=C.output_path(args,'INTERRUPTED_POSTHOC_ROWS.jsonl.gz'))
        try:
            for row in diagnostics:
                writer.write(row)
            writer.close();C.require(not writer.close_errors,'local posthoc close/fsync failed');writer.promote()
        except BaseException:
            writer.preserve_interrupted();raise
        result['posthoc_rows'] = C.binding(folder/'POSTHOC_ROWS.jsonl.gz')
        result['elapsed_seconds'] = time.monotonic()-start
        C.verify_protocol(args);C.protect(args)
        C.write_new(C.output_path(args,'METRICS.json'),result)
        with C.output_path(args,'METRICS.csv').open('x',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('stratum','method','scope','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'))
            for name,block in result['strata'].items():
                for method,summary in block['methods'].items():
                    for scope,values in summary['metrics'].items():
                        for metric,values in values.items():
                            writer.writerow([name,method,scope,metric]+[values[field] for field in
                                ('n','mean','sample_variance','sample_std','median','P90','max','unit')])
        print(json.dumps(dict(complete=True,verdict=result['verdict'],arithmetic=result['actual_arithmetic'])),flush=True)
    except BaseException as error:
        C.write_new(C.output_path(args,'STATISTICS_FAILURE.json'),dict(complete=False,
            error=dict(type=type(error).__name__,message=str(error)),automatic_retry=False,
            new_model_PnP_local_optimizer_score_draw_training_RGB_calls=0))
        raise


def main():
    parser = C.parser(__doc__,None)
    parser.add_argument('--input',default=str(C.PRIVATE/'geometry'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
