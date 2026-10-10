"""Fixed eight V8 contrasts from complete saved scores; no pose/model execution.

The published V7 scalar metric/posthoc kernels are reused unchanged. Only the
new control labels/population are adapted, without changing observations,
mask decisions, accepted NEW states or diagnostic inliers.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np

from . import common as C
from ..pallet_three_head_observation_20261010_v7 import statistics as V7


METRICS, SCOPES = V7.METRICS, V7.SCOPES
BOOTSTRAP = C.REPO/'_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz'


def compact_rows(path):
    for row in C.rows(path):
        yield V7.compact_saved_witnesses(row)


def posthoc(row, n3, observation):
    # V7's kernel looks up a fixed method spec. This copy changes its label
    # only; normalized ROLE sparse coordinates/fit/mask/status stay untouched.
    adapted = dict(row, method=C.PARENT_PRIMARY)
    result = V7.posthoc(adapted, n3, observation)
    result['method'] = row['method']
    result['posthoc_kernel_label_adapter_only'] = True
    result['standard_diagnostic_inlier_count_is_not_a_NEW_acceptance_gate'] = True
    return result


def compute(new, fixed, observations, ledgers, cohort, bootstrap):
    ids = list(cohort['ids'])
    C.require(len(ids) == len(set(ids)) == 245 and len(new) == 735 and len(fixed) == 490,
              'complete same245 three-control scored population required')
    methods = ('BASE','N3_SUBPIX') + tuple(C.METHODS)
    groups = {method:{} for method in methods}
    for row in new+fixed:
        C.require(row['method'] in groups and row['id'] not in groups[row['method']], 'unknown/duplicate scored row')
        groups[row['method']][row['id']] = row
    C.require(all(set(group) == set(ids) for group in groups.values()), 'scored populations differ')
    labels = {row['id']:row['label'] for row in cohort['frames']}
    C.require(Counter(labels.values()) == {'clean':153,'moderate':92}, 'difficulty strata differ')
    sessions = bootstrap['sessions']
    draws = np.asarray(bootstrap['counts'],np.int64)
    C.require(draws.shape == (10000,13) and len(set(sessions)) == 13 and
        (draws >= 0).all() and (draws.sum(1) == 13).all(), 'frozen13-session draws differ')
    session_index = {session:i for i,session in enumerate(sessions)}
    C.require(all(row['session'] in session_index for row in new+fixed), 'unknown draw session')
    contrasts = tuple(C.CONTRASTS)
    C.require(len(contrasts) == len(set(contrasts)) == 8, 'eight frozen contrasts required')
    obs = {row['id']:row for row in observations if row['head_arm'] == 'IMAGE_ROLE'}
    C.require(len(observations) == 735 and len(obs) == 245 and set(obs) == set(ids) and
        all(row['GT_input'] is False for row in observations), 'parent observation population differs')
    C.require(len(ledgers) == 245 and [row['id'] for row in ledgers] == ids and
        all(row['parent_replay_parity']['passed'] for row in ledgers), 'control ledger population/parity differs')
    diagnostic_rows = [posthoc(row,groups['N3_SUBPIX'][row['id']],obs[row['id']]) for row in new]
    strata = {}
    for name, selected in [('combined',ids),('easy',[fid for fid in ids if labels[fid] == 'clean']),
                           ('medium',[fid for fid in ids if labels[fid] == 'moderate'])]:
        chosen = set(selected)
        strata[name] = dict(frames=len(selected),ids=selected,
            methods={method:V7.summaries([groups[method][fid] for fid in selected]) for method in methods},
            contrasts={a+'_minus_'+b:{scope:V7.paired(groups[a],groups[b],selected,session_index,draws,scope)
                for scope in SCOPES} for a,b in contrasts},
            diagnostics={method:V7.diagnostic_summary([row for row in diagnostic_rows
                if row['method'] == method and row['id'] in chosen]) for method in C.METHODS})
    primary = strata['combined']['contrasts'][C.PRIMARY+'_minus_N3_SUBPIX']['common_operational']
    complete = primary['common_frames'] == 245
    improve = complete and all(primary['metrics'][metric]['mean_delta'] < 0
        for metric in ('translation_cm','rotation_deg'))
    same_U = [row for row in ledgers if row['H_and_no_H_used_U_equal']]
    removed = Counter(k for row in ledgers for k in row['actual_H_removed_sparse_ids'])
    return dict(schema='fixed_same_observation_saved_row_statistics_v8',complete=True,
        population=dict(frames=245,clean=153,moderate=92,severe_excluded=74,sessions=13,all_frames_retained=True),
        primary_method=C.PRIMARY,comparator='N3_SUBPIX',new_accuracy_primary=False,
        methods=strata['combined']['methods'],contrasts=strata['combined']['contrasts'],strata=strata,
        reference='Known GEOMETRIC_PROXY DEV; independent physical truth and unseen generalization not validated',
        verdict=dict(existing_parent_primary_full_operational_T_and_R_improved=bool(improve),
            all245_paired_outputs_available=complete,independent_generalization_established=False,
            diagnostic_result_does_not_change_frozen_primary=True),
        same_observation_diagnostics=dict(H_and_no_H_same_used_U_frames=len(same_U),
            same_U_ids=[row['id'] for row in same_U],actual_H_removed_sparse_ID_counts=dict(removed),
            fit_only_parity_scope='same decoded q and solver; does not remove H/role/proposal feature dependencies',
            full_head_or_occlusion_classifier_necessity_identified=False,
            diagnostic_inliers_do_not_gate_standard_NEW=True),
        bootstrap=dict(resamples=10000,sessions=sessions,new_draws_generated=0,
            serialized_raw_sha256=bootstrap['serialized_raw_sha256']),
        actual_arithmetic=dict(saved_scored_rows=1225,posthoc_rows=735,source_observation_rows=735,
            control_ledger_rows=245,strata=3,contrasts_per_stratum=8,paired_scopes=3,metric_CIs_requested=216),
        borrowed_unchanged_kernels=['V7.moments','V7.summaries','V7.paired','V7.posthoc with label adapter','V7.diagnostic_summary'],
        new_detector_N3_head_PnP_optimizer_ray_training_RGB_scoring_calls=0), diagnostic_rows


def run(args):
    from .evaluator import sealed
    C.verify_protocol(args)
    C.protect(args)
    folder = Path(args.input)
    C.require(folder.resolve() == Path(args.output).resolve(), 'statistics input/output must identify same new V8 evidence')
    seal, _, validation, _ = sealed(args)
    receipt = C.read(folder/'SCORING_RECEIPT.json')
    C.require(receipt['complete'] is True and receipt['scored_method_rows'] == 735 and
        receipt['fixed_control_rows'] == 490 and receipt['methods'] == list(C.METHODS) and
        receipt['GT_access_only_after_complete_seal_cleanup_and_validation'] is True and
        receipt['independent_geometry_validation'] == validation, 'complete validated score receipt required')
    for key, name in [('geometry_seal','GEOMETRY_SEAL.json'),('control_receipt','CONTROL_RECEIPT.json'),
            ('predictions','PREDICTIONS.jsonl.gz'),('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz'),
            ('parity','SCORING_PARITY.json')]:
        C.bound(folder/name,receipt[key],'scored '+key)
    parity = C.read(folder/'SCORING_PARITY.json')
    C.require(parity['passed'] is True and parity['fixed_rows'] == 490 and parity['parent_primary_rows'] == 245,
              'all parent score parities required')
    for name in ('STATISTICS_STARTED.json','METRICS.json','POSTHOC_ROWS.jsonl.gz','METRICS.csv','STATISTICS_FAILURE.json',
                 'PENDING_POSTHOC_ROWS.jsonl.gz','INTERRUPTED_POSTHOC_ROWS.jsonl.gz'):
        C.output_path(args,name)
    start = time.monotonic()
    C.write_new(C.output_path(args,'STATISTICS_STARTED.json'),dict(protocol=C.binding(args.protocol),
        scoring_receipt=C.binding(folder/'SCORING_RECEIPT.json'),new_model_PnP_score_draws=0))
    try:
        new,fixed = [list(compact_rows(folder/name)) for name in ('PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz')]
        observations = list(compact_rows(Path(args.parent)/'OBSERVATIONS.jsonl.gz'))
        ledgers = list(C.rows(folder/'CONTROL_LEDGERS.jsonl.gz'))
        with gzip.open(BOOTSTRAP,'rt',encoding='utf-8') as stream:
            bootstrap = json.load(stream)
        result, diagnostic_rows = compute(new,fixed,observations,ledgers,C.read(args.cohort),bootstrap)
        dependencies = dict(new_scored_rows=folder/'PREDICTIONS.jsonl.gz',fixed_scored_rows=folder/'FIXED_PREDICTIONS.jsonl.gz',
            scoring_receipt=folder/'SCORING_RECEIPT.json',geometry_seal=folder/'GEOMETRY_SEAL.json',
            control_receipt=folder/'CONTROL_RECEIPT.json',control_ledgers=folder/'CONTROL_LEDGERS.jsonl.gz',
            scoring_parity=folder/'SCORING_PARITY.json',observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',
            validation_protocol=folder/'VALIDATION_PROTOCOL.json',validation_receipt=folder/'VALIDATION_CHECKS.json',
            statistics_code=Path(__file__),borrowed_statistics_code=Path(V7.__file__),
            protocol=Path(args.protocol),cohort=Path(args.cohort),bootstrap=BOOTSTRAP)
        result['bindings'] = {name:C.binding(path) for name,path in dependencies.items()}
        result['saved_row_memory_policy'] = dict(original_full_serialized_rows_preserved=True,
            complete_SHA_checked_before_loading=True,unused_heavy_candidate_arrays_removed_only_in_RAM=True)
        from ..pallet_three_head_observation_20261010_v7.run import RowWriter
        writer = RowWriter(C.output_path(args,'POSTHOC_ROWS.jsonl.gz'),
            interrupted_path=C.output_path(args,'INTERRUPTED_POSTHOC_ROWS.jsonl.gz'))
        try:
            for row in diagnostic_rows:
                writer.write(row)
            writer.close()
            C.require(not writer.close_errors,'posthoc close/fsync failed')
            writer.promote()
        except BaseException:
            writer.preserve_interrupted()
            raise
        result['posthoc_rows'] = C.binding(Path(args.output)/'POSTHOC_ROWS.jsonl.gz')
        result['elapsed_seconds'] = time.monotonic()-start
        C.verify_protocol(args)
        C.protect(args)
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
        print(json.dumps(dict(complete=True,primary=result['verdict'],arithmetic=result['actual_arithmetic'])),flush=True)
    except BaseException as error:
        C.write_new(C.output_path(args,'STATISTICS_FAILURE.json'),dict(complete=False,
            error=dict(type=type(error).__name__,message=str(error)),automatic_retry=False,
            new_model_PnP_score_draw_training_RGB_calls=0))
        raise


def main():
    parser = C.parser(__doc__,None)
    parser.add_argument('--input',default=str(C.PRIVATE/'geometry'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
