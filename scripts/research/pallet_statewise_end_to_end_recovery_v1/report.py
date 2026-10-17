"""Render the frozen campaign's complete postseal report; never select a model.

This module consumes already-scored, independently checked outputs.  It does
not import a metric, solver, training routine or reference document.  Explicit
root GO and the prediction/evaluation/checker hash chain precede result reads.
The private report is subsequently redacted by ``publish.py`` for Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time

from .common import read, sha, write

BASELINES = ('R0', 'N0', 'E0', 'POSEFIX_RAW',
             'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
COMPARATORS = ('E0', 'N0', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
POPULATIONS = ('PRIMARY192', 'TEST49', 'REMAINDER143', 'CAL46')
DENOMINATORS = dict(PRIMARY192=192, TEST49=49, REMAINDER143=143, CAL46=46)
SOURCE_NAMESPACE = 'scripts/research/pallet_statewise_end_to_end_recovery_v1'
PUBLIC_NAMESPACE = '_docs/experiments/pallet_statewise_end_to_end_recovery_v1'


def _fmt(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(_cell, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(_cell, row)) + ' |' for row in rows])


def _optional(path):
    return read(path) if Path(path).is_file() else None


def _sealed_artifacts(folder, seal):
    """No path escape and no result decoding before all artifact bytes match."""
    folder = Path(folder).resolve()
    for name, digest in seal.get('artifacts', {}).items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha(path) != digest:
            raise ValueError('Changed/missing sealed artifact: ' + name)


def verify_report_inputs(run_dir, evaluation_dir, independent_receipt, *, report_GO=False):
    if report_GO is not True:
        raise PermissionError('Root explicit postseal report GO is required before any result read')
    run, evaluation = Path(run_dir), Path(evaluation_dir)
    plan_path, prediction_path = run/'PLAN_LOCK.json', run/'PREDICTION_SEAL.json'
    prediction = read(prediction_path)
    if (prediction.get('complete') is not True or prediction.get('before_scoring') is not True
            or prediction.get('reference_inputs') is not False
            or prediction.get('plan_lock_sha256') != sha(plan_path)):
        raise ValueError('Prediction seal must bind the unchanged plan and pre-reference outputs')
    # Hash the private journal, without decoding reference/pose documents.
    journal = (run/prediction.get('prediction_filename', 'PREDICTIONS.jsonl.gz')).resolve()
    if not journal.is_relative_to(run.resolve()):raise ValueError('Prediction journal must remain below the private run')
    if sha(journal) != prediction['predictions_sha256']:
        raise ValueError('Prediction journal changed')
    seal_path = evaluation/'EVALUATION_SEAL.json'
    seal = read(seal_path)
    bindings = seal.get('input_bindings', {})
    if (seal.get('explicit_root_scoring_GO') is not True
            or bindings.get('plan_lock') != sha(plan_path)
            or bindings.get('new_prediction_seal') != sha(prediction_path)
            or bindings.get('new_predictions') != sha(journal)):
        raise ValueError('Evaluation does not bind these frozen predictions')
    _sealed_artifacts(evaluation, seal)
    independent = read(independent_receipt)
    if (independent.get('status') != 'PASS' or independent.get('failure_count') != 0
            or sha(seal_path) not in independent.get('input_bindings', {}).values()
            or sha(evaluation/'RESULTS.json') not in independent.get('input_bindings', {}).values()):
        raise ValueError('A passing independent check bound to this evaluation is required')
    return prediction, seal, independent


def _inventory(registry):
    old = tuple(registry.get('original_arms', BASELINES))
    new = tuple(item.get('name', item.get('arm')) for item in registry['arms'])
    if old != BASELINES or len(new) != 16 or len(set(new)) != 16 or set(new) & set(old):
        raise ValueError('The six old and sixteen registered new arms must all remain visible')
    return old+new, new


def _execution_status(run, results, new_arms):
    actual = _optional(run/'ARM_EXECUTION_STATUS.json') or {}
    result = {}
    for arm in new_arms:
        present = all(arm in results['populations'][p] for p in POPULATIONS)
        entry = actual.get(arm, {})
        if isinstance(entry, str):
            entry = {'status': entry}
        if present:
            if entry.get('status') not in (None, 'EXECUTED', 'DONE'):
                raise ValueError('Execution inventory contradicts actual scored predictions: '+arm)
            result[arm] = dict(status='EXECUTED')
        else:
            why = entry or results.get('unexecuted_arms', {}).get(arm, {})
            result[arm] = dict(status=why.get('status', 'NOT_EXECUTED'),
                               reason=why.get('reason', 'No sealed output; numerical result N/A'))
    return result


def classify(arm, primary, *, independent_passed=True):
    """Conservative descriptive rules from instruction12.3, never deployment.

    No seed ranking, fitted threshold, p-value or PRIMARY-based gating is
    created. 'No new worsening' is conservatively operationalized as no worse
    mean T/R, three specified tail counts or no-pose count against all four
    predefined comparators. This descriptive implementation is explicit.
    """
    if arm not in primary:
        return dict(category='INCONCLUSIVE_DATA_OR_REFERENCE', reason='No actual sealed prediction/scoring output')
    if not independent_passed:
        return dict(category='INCONCLUSIVE_DATA_OR_REFERENCE', reason='Independent arithmetic check unavailable')
    value = primary[arm]
    compared = [primary[x] for x in COMPARATORS]
    higher = all(value['successes'] > other['successes'] for other in compared)
    fields = ('T_gt50cm', 'T_gt100cm', 'R_gt45deg', 'no_pose')
    mean_keys = ('translation_cm', 'rotation_deg')
    no_worse = all(value[k] <= other[k] for other in compared for k in fields)
    no_worse &= all(value[k]['mean'] is not None and other[k]['mean'] is not None
                    and value[k]['mean'] <= other[k]['mean'] for other in compared for k in mean_keys)
    if higher:
        return dict(category='PROMISING_FOR_INDEPENDENT_VALIDATION' if no_worse else 'SUCCESS_GAIN_WITH_TRADEOFF',
                    reason='Joint success exceeds all four predefined comparators; '+
                           ('specified mean/tails/coverage do not worsen' if no_worse else 'specified mean/tails/coverage include worsening'))
    e0 = primary['E0']
    improves = any(value[k]['mean'] is not None and e0[k]['mean'] is not None
                   and value[k]['mean'] < e0[k]['mean'] for k in mean_keys)
    improves |= any(value[k] < e0[k] for k in fields)
    return dict(category='CONTINUOUS_ERROR_GAIN_ONLY' if improves else 'NO_GAIN_OR_HARM',
                reason='No joint-success advantage over all four; '+
                       ('one or more continuous/tail/coverage measures improve against E0; inspect tradeoffs' if improves else 'no specified continuous/tail/coverage improvement against E0'))


def _metric_cells(summary):
    if summary is None:
        return ['N/A']*12
    t, r = summary['translation_cm'], summary['rotation_deg']
    return [f"{summary['successes']}/{summary['denominator']} ({100*summary['success_rate']:.2f}%)",
            f"{_fmt(t['mean'])} ± {_fmt(t['sample_std'])}",
            f"{_fmt(r['mean'])} ± {_fmt(r['sample_std'])}",
            f"{t['n']}/{r['n']}",
            f"{_fmt(t['P90'])}/{_fmt(t['P95'])}/{_fmt(t['max'])}",
            f"{_fmt(r['P90'])}/{_fmt(r['P95'])}/{_fmt(r['max'])}",
            str(summary['T_gt50cm']), str(summary['T_gt100cm']), str(summary['R_gt45deg']),
            str(summary['no_pose']), str(summary.get('metric_error_n', 0)),
            str(summary.get('status_counts', {}).get('fallback', {}).get('true_n', 'unknown'))]


def common_tables(results, arms, execution):
    lines = []
    for population in POPULATIONS:
        lines += [f'### {population} — 전체 분모 {DENOMINATORS[population]}', '']
        values = results['populations'][population]
        body = []
        for arm in arms:
            status = 'REUSED_SEALED_BASELINE' if arm in BASELINES else execution[arm]['status']
            body.append([arm, status]+_metric_cells(values.get(arm)))
        lines += [_table(['방법', '실행', '공동≤5cm·5°', '위치cm 평균±표본SD',
                         '회전° 평균±표본SD', '가용n T/R', 'T P90/P95/max', 'R P90/P95/max',
                         'T>50cm', 'T>100cm', 'R>45°', 'no-pose', 'metric error', 'fallback'], body), '']
    return lines


def paired_tables(results, new_arms):
    lines = []
    for population in POPULATIONS:
        body = []
        for after in new_arms:
            for before in COMPARATORS:
                item = results['paired'][population].get(after+'_minus_'+before)
                if item is None:
                    body.append([after, before]+['N/A']*11)
                    continue
                c = item['counts'];d = item['deltas']['success_rate']
                ci = d['recording_cluster_CI'].get('CI95')
                sensitivity = d['named_session_cluster_sensitivity_CI'].get('CI95')
                interval = lambda x:'N/A' if x is None else f'[{100*x[0]:.2f}, {100*x[1]:.2f}]'
                ct = item['both_failure_continuous']['translation_cm']
                cr = item['both_failure_continuous']['rotation_deg']
                body.append([after,before,c['BOTH_SUCCESS'],c['RECOVERY'],c['DAMAGE'],c['BOTH_FAILURE'],
                             f"{item['success_difference_count']:+d}",interval(ci),interval(sensitivity),
                             f"{ct['improved_n']}/{ct['worsened_n']}",
                             f"{_fmt(ct['improvement_sum'])}/{_fmt(ct['worsening_sum'])}",
                             f"{cr['improved_n']}/{cr['worsened_n']}",
                             f"{_fmt(cr['improvement_sum'])}/{_fmt(cr['worsening_sum'])}"])
        lines += [f'### {population}: 짝지은 전량 손익', '',
                  _table(['새 방법','비교','공동성공','회복','손상','공동실패','성공차 장',
                          '기록군집95%CI %p','이름session95%CI %p','공동실패 T 감소/증가 n',
                          'T 감소/증가 합cm','공동실패 R 감소/증가 n','R 감소/증가 합°'], body), '']
    return lines


def stage_table(stage, new_arms):
    result=[]
    for population in POPULATIONS:
        value=stage['populations'][population]
        result += [f'### {population}: 후보 단계 진단', '',
                   f"기존 고유 carried bank에 성공 후보가 있는 영상: {value['carried_oracle_successes']}/{value['denominator']}. "
                   '이는 평가 참조를 나중에 적용한 상한이며 실행 중 사용한 점수가 아니다.', '']
        body=[]
        for arm in new_arms:
            a=value['arms'].get(arm)
            if not a:
                body.append([arm]+['N/A']*6);continue
            field=lambda k:'N/A' if a.get(k) is None else a[k]
            body.append([arm,field('generated_before_constraints_oracle_successes'),
                         field('after_constraints_oracle_successes'),field('any_family_winner_successes'),
                         field('final_successes'),a['generated_oracle_known_frames'],
                         json.dumps(a['family_intermediate_status_counts'],ensure_ascii=False)])
        result += [_table(['방법','유한 시도+carried 상한','제약 후 상한','family 승자 상한',
                           '최종 실제 성공','생성 상한 가용영상','family 기록 상태'],body), '']
    return result


def _training_manifest(run):
    folder=run/'head_training_v1'
    if (folder/'TRAINING_MANIFEST.json').is_file():
        value=read(folder/'TRAINING_MANIFEST.json')
        if value.get('complete') is not True or value.get('total_training_runs') != 6:
            raise ValueError('Partial training cannot be represented as six completed runs')
        return value
    refusal=_optional(folder/'TRAINING_NOT_RUN.json')
    return dict(schema='EXPLICIT_TRAINING_EXECUTION_MANIFEST_V1',total_training_runs=0,runs=[],
                status='NOT_APPLICABLE' if refusal else 'NOT_EXECUTED',
                reason=refusal or 'No sealed training manifest; all learning results are N/A',
                predefined_seeds=[20261018,20261019,20261020],models=['L_LINEAR','L_HEAD'])


def verify_synthetic_head_evaluation(run):
    """Verify optional six-head validation without decoding target/pose banks.

    The fixed helper already performed postprediction target evaluation. Here
    only its two small outputs and input-seal hashes are connected to the
    report. A completed six-run training cannot silently lose this validation.
    """
    run=Path(run);folder=run/'synthetic_head_evaluation_v1';path=folder/'EVALUATION_SEAL.json'
    if not path.is_file():
        if (run/'head_training_v1/TRAINING_MANIFEST.json').is_file():
            raise ValueError('Six trained heads require their sealed synthetic joint/abstention validation')
        return None
    seal=read(path)
    if (seal.get('schema')!='SYNTHETIC_HEAD_EVALUATION_SEAL_V1' or seal.get('complete') is not True
            or seal.get('explicit_root_GO') is not True or seal.get('rows')!=1003):
        raise ValueError('Complete exact1003 synthetic head evaluation seal required')
    filenames=('SYNTHETIC_LEARNED_EVALUATION.json','SYNTHETIC_LEARNED_EVALUATION_KO.md')
    aliases={'result_sha256':filenames[0],'markdown_sha256':filenames[1]}
    for key,name in aliases.items():
        digest=sha(folder/name)
        if digest!=seal.get(key) or digest!=seal.get('artifacts',{}).get(name):
            raise ValueError('Synthetic head small result bytes changed: '+name)
    required=dict(distribution_seal='head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json',
                  pose_prediction_seal='learned_synthetic_v1/PREDICTION_SEAL.json',
                  training_manifest='head_training_v1/TRAINING_MANIFEST.json',
                  GT='synthetic_v1/SYNTH_GT_LABELS.jsonl.gz',
                  labels='head_labels_v1/SYNTHETIC_HEAD_TARGETS.jsonl.gz',
                  label_seal='head_labels_v1/LABEL_SEAL.json',split_lock='synthetic_v1/DATA_SPLIT_LOCK.json')
    for key,name in required.items():
        if seal.get('input_bindings',{}).get(key)!=sha(run/name):
            raise ValueError('Synthetic head evaluation input binding changed: '+key)
    distributions=read(run/required['distribution_seal'])
    if (distributions.get('complete') is not True or distributions.get('rows')!=1003
            or distributions.get('reference_reads')!=0 or distributions.get('before_reference_decode') is not True
            or distributions.get('tiny_head_forward_calls')!=6018 or distributions.get('new_training_updates')!=0):
        raise ValueError('Synthetic distributions must be exact frozen six-head inference, without new fitting')
    result=read(folder/filenames[0])
    if (result.get('complete') is not True or result.get('rows')!=1003
            or result.get('no_model_choice_or_fit') is not True
            or result.get('first_representative_seed')!=20261018
            or result.get('input_bindings')!=seal['input_bindings']):
        raise ValueError('Synthetic validation result/selection contract changed')
    return dict(folder=folder,seal=seal,seal_sha256=sha(path),files=filenames,
                result_sha256=sha(folder/filenames[0]),markdown_sha256=sha(folder/filenames[1]),
                validation_readout=(dict(denominator=result['populations']['validation']['denominator'],
                    representative_head_successes=result['populations']['validation']['arms']['L_HEAD_s20261018']['successes'],
                    N0_successes=result['populations']['validation']['N0_successes'],
                    eligible_candidate_oracle_successes=result['populations']['validation']['candidate_oracle_successes'],
                    no_good_frames=result['populations']['validation']['no_good_frames'],
                    abstentions=result['populations']['validation']['arms']['L_HEAD_s20261018']['abstentions'],
                    good_but_abstained=result['populations']['validation']['arms']['L_HEAD_s20261018']['abstain_with_good'],
                    no_good_abstained=result['populations']['validation']['arms']['L_HEAD_s20261018']['abstain_with_no_good'],
                    all_seed_successes={a:v['successes'] for a,v in result['populations']['validation']['arms'].items()})
                    if 'populations' in result else None))


def _without_pose_matrices(value):
    """A small forensic appendix retains numbers but not private pose payloads."""
    removed={'actual_pose','start_pose','R_physical','centroid','native_xyz','object_xyz','R_cf','native_t','physical_registration_Q'}
    if isinstance(value,dict):return {k:_without_pose_matrices(v) for k,v in value.items() if k not in removed}
    if isinstance(value,list):return [_without_pose_matrices(v) for v in value]
    return value


def optional_appendices(run,independent_path,independent):
    """Validate small postseal audits. Never generate or alter a pose/selector."""
    run=Path(run);files=[];bindings={};summary={}
    def add(path,name=None,transform=None):
        path=Path(path);files.append(dict(source=path,name=name or path.name,transform=transform))
        bindings[str(path)]=sha(path)
    rounding=independent.get('rounding_direction_audit')
    if rounding:
        path=Path(independent_path).parent/rounding['file'];audit=read(path)
        if (sha(path)!=rounding['sha256'] or audit.get('status')!='PASS'
                or audit.get('zero_tolerance_changed') is not False
                or audit.get('predictions_references_models_success_criteria_weights_or_reported_numbers_changed') is not False):
            raise ValueError('Strict direction audit must preserve original predictions and evaluator counts')
        for original,digest in audit.get('previous_failure_artifacts',{}).items():
            if sha(original)!=digest:raise ValueError('Original failed checker provenance changed')
            if Path(original).name=='INDEPENDENT_CHECK.json':add(original,'INDEPENDENT_CHECK_V1_FAILED.json')
        add(path);add(independent_path,'INDEPENDENT_CHECK_CURRENT.json')
        summary['rounding_direction']=dict(previous_failed_fields=audit['previous_failed_count_fields_exactly_explained'],
            sign_switch_rows=audit['sign_switch_pair_metric_rows'],frames=audit['unique_sign_switch_frame_count'],
            bitwise_original_expression_checks=audit['original_expression_value_checks'])
    score_check=independent.get('stored_score_audit')
    if score_check:
        path=Path(independent_path).parent/score_check['file']
        if sha(path)!=score_check['sha256'] or read(path).get('status')!='PASS':
            raise ValueError('Stored-score/selection audit must pass unchanged')
        add(path)
    folder=run/'positive_mass_v1';seal_path=folder/'POSITIVE_MASS_SEAL.json'
    if seal_path.is_file():
        seal=read(seal_path);check_path=folder/'INDEPENDENT_MASS_COUNT_CHECK.json';check=read(check_path)
        if (seal.get('complete') is not True or seal.get('synthetic_only') is not True
                or check.get('status')!='PASS' or check.get('PRIMARY_results_read') is not False
                or check.get('input_bindings',{}).get('diagnosis_seal')!=sha(seal_path)):
            raise ValueError('Positive-mass diagnosis needs its own synthetic-only independent PASS')
        _sealed_artifacts(folder,seal)
        for key,relative in (('distribution_seal','head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json'),
                             ('label_seal','head_labels_v1/LABEL_SEAL.json')):
            if seal['input_bindings'][key]!=sha(run/relative):raise ValueError('Positive-mass input seal changed')
        add(seal_path);add(check_path)
        for name in ('POSITIVE_MASS_DIAGNOSIS.json','POSITIVE_MASS_DIAGNOSIS_KO.md','POSITIVE_MASS_ROWS.csv'):add(folder/name)
        first=check.get('arms',{}).get('L_HEAD_s20261018')
        summary['positive_mass']=dict(synthetic_validation_only=True,not_runtime_oracle_rule=True,
            predeclared_head_good_but_abstained=first.get('good_abstained') if first else None,
            good_mass_sum_gt_none=first.get('good_sum_gt_none') if first else None,
            good_mass_sum_le_none=first.get('good_sum_le_none') if first else None)
    folder=run/'boundary_extreme_forensics_v1';seal_path=folder/'FORENSIC_SEAL.json'
    if seal_path.is_file():
        seal=read(seal_path)
        if (seal.get('complete') is not True or seal.get('fitting_guard',{}).get('forbidden_attempt_count')!=0
                or seal.get('input_bindings',{}).get('prediction_seal')!=sha(run/'PREDICTION_SEAL.json')):
            raise ValueError('Boundary forensics must bind original predictions and forbid fitting')
        _sealed_artifacts(folder,seal)
        add(seal_path,'BOUNDARY_EXTREME_FORENSIC_SEAL.json')
        add(folder/'BOUNDARY_EXTREME_FORENSICS_KO.md')
        add(folder/'BOUNDARY_EXTREME_FORENSICS.json','BOUNDARY_EXTREME_FORENSICS_PUBLIC.json',_without_pose_matrices)
        summary['boundary_extreme']=dict(fixed_pose_rechecks_only=True,private_pose_values_removed_from_public_derivative=True)
        note_path=folder/'CONTRACT_NOTE_SEAL.json'
        if note_path.is_file():
            note=read(note_path)
            if (note.get('complete') is not True
                    or note.get('input_bindings',{}).get('original_forensic')!=sha(folder/'BOUNDARY_EXTREME_FORENSICS.json')
                    or note.get('input_bindings',{}).get('original_forensic_seal')!=sha(seal_path)):
                raise ValueError('Physical support contract note must bind untouched original forensics')
            _sealed_artifacts(folder,note)
            add(note_path)
            add(folder/'CONTRACT_VALIDITY_NOTE.json');add(folder/'CONTRACT_VALIDITY_NOTE_KO.md')
            summary['boundary_extreme']['contract_note_bound']=True
    return dict(files=files,input_bindings=bindings,summary=summary)


def _bindings(run, source_root, prediction, evaluation_seal, independent_path):
    names=['PLAN_LOCK.json','CONFIG_FROZEN.json','ARM_REGISTRY.json','INPUT_CONTRACT_REPAIR_V1.json',
           'SOURCE_SYMMETRY_PROSE_CORRECTION.json','TECHNICAL_FEATURE_PACKAGING_AMENDMENT.json',
           'RUNTIME_WORKER_SCHEDULING_AMENDMENT.json',
           'ENVIRONMENT.json','FINAL_PRE_SCORING_TESTS.json','FINAL_PRE_SCORING_TESTS_REPAIRED.json',
           'POSTSEAL_TEST_ISOLATION_REPAIR.json','P6_POSTTRAIN_VALIDATION_AMENDMENT.json',
           'SYNTHETIC_REFERENCE_FIREWALL_SOURCE_AUDIT.json',
           'ROOT_RESULT_INTERPRETATION_KO.md','ROOT_RESULT_INTERPRETATION_KO_PRELIMINARY_V1.md',
           'ROOT_ACTUAL_VISUAL_REVIEW.json','CLOSEOUT_SCIENTIFIC_REVIEW.json','CLOSEOUT_SCIENTIFIC_REVIEW.md',
           'PUBLICATION_PROSE_REPAIR_V2.json',
           'L_PRELABEL_SYMMETRY_DESCRIPTION_AMENDMENT.json','synthetic_v1/SYNTH_GT_AUDIT_ATTEMPTS.json',
           'inputs_all_available_v1/INPUT_COVERAGE_INDEPENDENT.json',
           'synthetic_v1/PROVISION_SEAL.json','synthetic_v1/DATA_SPLIT_LOCK.json',
           'synthetic_v1/NONE_PRIOR_CALIBRATION.json','real_batch_all_available_v1/BATCH_SEAL.json',
           'real_features_v1/FEATURE_SEAL.json','learned_real_v1/PREDICTION_SEAL.json',
           'learned_synthetic_v1/PREDICTION_SEAL.json','head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json',
           'synthetic_head_evaluation_v1/EVALUATION_SEAL.json',
           'head_training_v1/TRAINING_MANIFEST.json','PREDICTION_SEAL.json']
    sources={str(p.relative_to(source_root)):sha(p)
             for p in sorted((source_root/SOURCE_NAMESPACE).glob('*.py'))}
    public_receipts={str(p):sha(p) for p in sorted((source_root/PUBLIC_NAMESPACE).glob('*.json'))
                     if p.name in ('INPUT_COVERAGE_INDEPENDENT.json','A_OWNER_EQUIVALENCE_AUDIT.json','SOLVER_REVIEW_V1.json')
                     or p.name.startswith('GALLERY_PRESENTATION_AMENDMENT') or p.name=='GALLERY_REPRESENTATIVE_LOCK.json'}
    return dict(schema='ACTUAL_SOURCE_INPUT_BINDINGS_V1',
                source_root=str(source_root),source_files=sources,
                private_artifacts={str(run/name):sha(run/name) for name in names if (run/name).is_file()},
                technical_public_receipts=public_receipts,
                evaluation_input_bindings=evaluation_seal['input_bindings'],
                independent_check=dict(path=str(independent_path),sha256=sha(independent_path)),
                prediction_arms=prediction['arms'],
                code_hashes_do_not_imply_public_raw_data_availability=True)


def _cost(run, training, stage=None, *, independent=None, results=None, gallery_receipt=None, cost_ledger=None):
    """Keep receipt scopes explicit; reused frames and RSS cannot be summed twice."""
    paths=['real_batch_v1/STOPPED_INPUT_CONTRACT_REPAIR.json',
           'real_batch_all_available_v1/BATCH_SEAL.json',
           'synthetic_pilot_batch_v1/BATCH_SEAL.json',
           'synthetic_expansion_batch_v1/BATCH_SEAL.json',
           'synthetic_expansion_batch_v1/EXTENSION_REUSE_RECEIPT.json',
           'synthetic_expanded_batch_v1/BATCH_SEAL.json',
           'synthetic_v1/PROVISION_SEAL.json',
           'synthetic_v1/PILOT512_ROOT_FRAMES_AUTO_MASK_SEAL.json',
           'synthetic_v1/EXPANSION1003_ROOT_FRAMES_AUTO_MASK_SEAL.json',
           'synthetic_v1/NONE_PRIOR_CALIBRATION.json',
           'head_training_v1/TRAINING_MANIFEST.json',
           'learned_real_v1/PREDICTION_SEAL.json',
           'learned_synthetic_v1/PREDICTION_SEAL.json',
           'head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json',
           'synthetic_head_evaluation_v1/EVALUATION_SEAL.json',
           'evaluation_v1/EVALUATION_SEAL.json']
    paths += ['INPUT_CONTRACT_REPAIR_V1.json','TECHNICAL_FEATURE_PACKAGING_AMENDMENT.json',
              'RUNTIME_WORKER_SCHEDULING_AMENDMENT.json',
              'SOURCE_SYMMETRY_PROSE_CORRECTION.json','L_PRELABEL_SYMMETRY_DESCRIPTION_AMENDMENT.json',
              'inputs_all_available_v1/INPUT_COVERAGE_INDEPENDENT.json',
              'synthetic_v1/NONE_PRIOR_TECHNICAL_ATTEMPT1.json',
              'synthetic_v1/NONE_PRIOR_TECHNICAL_ATTEMPT2.json',
              'synthetic_v1/ROOT_ADAPTER_FORMAT_REPAIR.json',
              'synthetic_v1/ROOT_ADAPTER_T_DOMAIN_REPAIR.json',
              'synthetic_v1/SYNTH_GT_AUDIT_ATTEMPTS.json','b_boundary_v1/CAL_PROBE_RECEIPT.json',
              'SOLVER_REVIEW_V1.json']
    paths += ['p0_baseline/BASELINE_REPRODUCTION.json','synthetic_v1/SELECTION_RECEIPT.json',
              'synthetic_v1/adapter_attempt2_missing_T_domain/PILOT512_ROOT_FRAMES_AUTO_MASK_SEAL.json',
              'head_labels_v1/LABEL_SEAL.json','real_features_v1/FEATURE_SEAL.json',
              'synthetic_features_v1/FEATURE_SEAL.json']
    paths += ['independent_v1/INDEPENDENT_CHECK.json','positive_mass_v1/POSITIVE_MASS_SEAL.json',
              'positive_mass_v1/INDEPENDENT_MASS_COUNT_CHECK.json','boundary_extreme_forensics_v1/FORENSIC_SEAL.json',
              'mechanism_v1/MECHANISM_SEAL.json']
    paths += [str(p.relative_to(run)) for p in run.glob('gallery_v*/GALLERY_RECEIPT.json')]
    # Include each executed input-production receipt, including technical
    # attempts. They contain scoped delta CPU/wall measurements, not GT poses.
    paths += [str(p.relative_to(run)) for p in (run/'synthetic_v1').glob('*_SEAL.json')]
    paths += [str(p.relative_to(run)) for p in (run/'synthetic_v1'/'adapter_attempt2_missing_T_domain').glob('*_SEAL.json')]
    paths=list(dict.fromkeys(paths))
    receipts={name:dict(sha256=sha(run/name),receipt=read(run/name)) for name in paths if (run/name).is_file()}
    resources=[]
    if (run/'RESOURCE_LEDGER.jsonl').is_file():
        for line in (run/'RESOURCE_LEDGER.jsonl').read_text().splitlines():
            if line.strip():resources.append(json.loads(line))
    partial=receipts.get('real_batch_v1/STOPPED_INPUT_CONTRACT_REPAIR.json',{}).get('receipt')
    if (run/'real_batch_v1').is_dir():
        stopped_frames=[]
        for p in sorted((run/'real_batch_v1').glob('frames/*/FRAME_SEAL.json')):
            value=read(p)
            stopped_frames.append(dict(id=value['id'],resource=value.get('resource'),seal_sha256=sha(p)))
        # Some batch versions use the frame index directly below out/.
        if not stopped_frames:
            for p in sorted((run/'real_batch_v1').glob('*/FRAME_SEAL.json')):
                value=read(p)
                stopped_frames.append(dict(id=value['id'],resource=value.get('resource'),seal_sha256=sha(p)))
    else:stopped_frames=[]
    technical_frames=[]
    for p in sorted((run/'technical_cal2_v1').glob('frames/*/FRAME_SEAL.json')):
        value=read(p)
        technical_frames.append(dict(id=value['id'],resource=value.get('resource'),seal_sha256=sha(p)))
    technical=[]
    probe=run/'TECHNICAL_SYNTH_ONE_FRAME_NO_RGB_MASK.json'
    if probe.is_file():
        value=read(probe);result=value.get('result',{})
        technical.append(dict(path=str(probe),sha256=sha(probe),candidate_count=value.get('candidate_count'),
            feature_length=value.get('feature_length'),actual_new_solve_attempts=result.get('actual_new_solve_attempts'),
            bank_sizes={k:len(v) for k,v in result.get('banks',{}).items()},
            reference_inputs=result.get('reference_inputs'),
            wall_CPU_memory='NOT_MEASURED unless explicitly present in original receipt',
            full_pose_bank_private_only=True))
    seen=set();disk=0
    for p in run.rglob('*'):
        if p.is_file():
            stat=p.stat();key=(stat.st_dev,stat.st_ino)
            if key not in seen:disk+=stat.st_size;seen.add(key)
    scoped_calls={}
    if stage is not None:
        for population,parts in (('PRIMARY192',('TEST','REMAINDER')),('TEST49',('TEST',)),
                                 ('REMAINDER143',('REMAINDER',)),('CAL46',('CAL',))):
            selected=[r for r in stage['rows'] if r['partition'] in parts]
            ransac=[a for r in selected for a in r.get('arms',{}).get('CTRL_RANSAC',{}).get('ransac_attempt_metadata',[])]
            scoped_calls[population]=dict(frames=len(selected),
                 actual_new_optimizer_attempts=sum(r['new_attempt_count'] for r in selected),
                 RANSAC_family_records=len(ransac),
                 RANSAC_calls=sum(not a['status'].startswith('UNSUPPORTED_') for a in ransac),
                 RANSAC_status_counts=dict(Counter(a['status'] for a in ransac)),
                 new_backbone_forwards=0,old_upstream_production_cost_not_zeroed=True)
    # Actual known scope totals, not a claim of complete accounting. In
    # particular, choose the complete1003 CPU ledger once: its reused512 costs
    # are the original pilot work, not491 additional repeated solves.
    known_cpu=[];known_gpu=[];seen_receipts=set()
    for name,entry in receipts.items():
        value=entry['receipt'];digest=entry['sha256']
        if digest in seen_receipts:continue
        seen_receipts.add(digest)
        if name in ('synthetic_pilot_batch_v1/BATCH_SEAL.json','synthetic_expanded_batch_v1/BATCH_SEAL.json'):
            continue
        if 'sum_frame_cpu_seconds' in value:
            known_cpu.append(dict(scope=name,CPU_seconds=value['sum_frame_cpu_seconds']))
        else:
            cost=value.get('cost',value)
            cpu=next((cost[k] for k in ('CPU_seconds','cpu_seconds','process_cpu_seconds')
                      if isinstance(cost.get(k),(int,float))),None)
            if cpu is not None:known_cpu.append(dict(scope=name,CPU_seconds=cpu))
        if isinstance(value.get('GPU_elapsed_seconds'),(int,float)):
            known_gpu.append(dict(scope=name,GPU_seconds=value['GPU_elapsed_seconds'],
                                  new_model_forward_calls=value.get('new_model_forward_calls')))
    for name,values in (('stopped52_frame_resources',stopped_frames),('technical_CAL2_frame_resources',technical_frames)):
        known_cpu.append(dict(scope=name,CPU_seconds=sum((v.get('resource') or {}).get('process_cpu_seconds',0.) for v in values),
                              recorded_frames=len(values)))
    for item in resources:
        if isinstance(item.get('process_cpu_seconds'),(int,float)):
            known_cpu.append(dict(scope='RESOURCE_LEDGER:'+item['stage'],CPU_seconds=item['process_cpu_seconds']))
    for name,value in (('independent_check',independent),('evaluation_results',results),('gallery',gallery_receipt)):
        if value and isinstance(value.get('process_cpu_seconds'),(int,float)):
            if not any(value==entry['receipt'] for entry in receipts.values()):
                known_cpu.append(dict(scope=name,CPU_seconds=value['process_cpu_seconds']))
    controller=_optional(run/'CONTROLLER_STATE.json') or {}
    runtime=_optional(controller.get('runtime_config_path','')) if controller.get('runtime_config_path') else None
    campaign_start=(runtime or {}).get('campaign_start_epoch',1792252965)
    now=time.time()
    total_known=dict(campaign_start_epoch=campaign_start,cost_snapshot_epoch=now,
                     campaign_elapsed_wall_seconds=now-campaign_start,
                     CPU_seconds_known_subtotal=sum(v['CPU_seconds'] for v in known_cpu),
                     CPU_scopes=known_cpu,GPU_seconds=sum(v['GPU_seconds'] for v in known_gpu),
                     GPU_scopes=known_gpu,
                     training_GPU_seconds=0. if all(v.get('cost',{}).get('device')=='CPU' for v in training.get('runs',[])) else None,
                     new_backbone_or_SAM_forward_calls_recorded=sum(v.get('new_model_forward_calls') or 0 for v in known_gpu),
                     CPU_subtotal_is_lower_bound_not_complete=True,
                     historical_upstream_production_cost='UNKNOWN_SEPARATE; cached reuse does not imply zero production cost')
    head_forward_counts={}
    for name in ('learned_real_v1/PREDICTION_SEAL.json','learned_synthetic_v1/PREDICTION_SEAL.json'):
        entry=receipts.get(name,{}).get('receipt')
        if entry:
            head_forward_counts[name]=dict(tiny_head_forward_calls=entry['rows']*len(entry['checkpoints']),
                operation='fixed checkpoint pose prediction; inference only',new_training_updates=0)
    distribution=receipts.get('head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json',{}).get('receipt')
    if distribution:
        head_forward_counts['head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json']=dict(
            tiny_head_forward_calls=distribution['tiny_head_forward_calls'],
            operation='additional frozen-head pass to archive full log-softmax; inference only',
            new_training_updates=distribution['new_training_updates'])
    total_known['tiny_head_inference_counts']=head_forward_counts
    total_known['tiny_head_forward_calls_recorded']=sum(v['tiny_head_forward_calls'] for v in head_forward_counts.values())
    total_known['new_neural_forward_calls_recorded']=(total_known['new_backbone_or_SAM_forward_calls_recorded']+
                                                    total_known['tiny_head_forward_calls_recorded'])
    full_real=receipts.get('real_batch_all_available_v1/BATCH_SEAL.json',{}).get('receipt',{})
    full_synth=receipts.get('synthetic_expansion_batch_v1/BATCH_SEAL.json',{}).get('receipt',{})
    total_known['actual_optimizer_attempt_counts']=dict(
        final_real238=full_real.get('new_optimizer_attempts'),
        final_synthetic1003=full_synth.get('new_optimizer_attempts'),
        stopped_input52=sum((v.get('resource') or {}).get('new_optimizer_attempts',0) for v in stopped_frames),
        technical_CAL2=sum((v.get('resource') or {}).get('new_optimizer_attempts',0) for v in technical_frames),
        technical_synthetic_examples=sum(v.get('actual_new_solve_attempts') or 0 for v in technical),
        note='Counts include failed/infeasible local attempts; reused pilot512 is included once in final1003')
    batch_wall=[dict(scope=name,wall_seconds=receipts[name]['receipt']['elapsed_seconds'])
                for name in ('synthetic_pilot_batch_v1/BATCH_SEAL.json','synthetic_expansion_batch_v1/BATCH_SEAL.json',
                             'real_batch_all_available_v1/BATCH_SEAL.json') if name in receipts]
    total_known['completed_solver_batch_cumulative_wall_seconds']=sum(v['wall_seconds'] for v in batch_wall)
    total_known['completed_solver_batch_wall_scopes']=batch_wall
    total_known['solver_analysis_full_cost_caveat']='Completed batch job walls include overlap as separate work scopes; stopped/technical/analysis work is listed separately and root ledger defines complete approved-budget accounting.'
    plan=_optional(run/'PLAN_LOCK.json') or {}
    return dict(schema='SCOPED_ACTUAL_COST_AND_EXECUTION_V1',receipts=receipts,resource_ledger=resources,
                training_total_runs=training['total_training_runs'],training_cost=training.get('cost'),
                stopped_input_contract_attempt=partial,stopped_sealed_frame_costs=stopped_frames,
                technical_CAL2_frame_costs=technical_frames,total_known_cost=total_known,
                root_cost_ledger=read(cost_ledger) if cost_ledger is not None else None,
                frozen_budget_limits=plan.get('runtime_budget',dict(wall_hours=24,gpu_hours=8,solver_batch_wall_hours=10,new_disk_gib=25,workers=4)),
                other_technical_executions=technical,
                real_actual_call_counts=scoped_calls,
                private_disk_unique_inode_bytes=disk,
                accounting_notes=[
                    '42 per-frame attempts are local SLSQP attempts across the registered models, not all PnP calls.',
                    'Up to two actual RANSAC calls/frame and synthetic producer SQPnP/LM operations are separate counters.',
                    'Batch elapsed times may overlap; summing them is not end-to-end wall time.',
                    'Expanded synthetic per-frame CPU includes reused pilot512 work; do not add pilot CPU again.',
                    'Stopped52-frame input-coverage attempt is preserved separately and is not the final arm.',
                    'Per-process peak RSS is a lifetime high-water mark and is not an additive allocation.',
                    'Cached historical upstream production cost is separate from newly executed solver/head/mask cost.',
                    'Absent resource measurements are unknown, never zero.'],
                exact_end_to_end_totals_available=False)


def ambiguity_posthoc(csv_path):
    """Fixed diagnostic bins only; no fitted gate, seed selection or pose change."""
    with Path(csv_path).open() as f:raw=list(csv.DictReader(f))
    definitions={'A_owner_entropy':[i*math.log(4)/4 for i in range(5)],
                 'A_none_mean':[0.,.25,.5,.75,1.]}
    output={}
    primary=[r for r in raw if r['partition'] in ('TEST','REMAINDER')]
    for arm in sorted({r['arm'] for r in primary}):
        selected=[r for r in primary if r['arm']==arm];features={}
        for name,edges in definitions.items():
            values=[(r,float(r[name])) for r in selected if r.get(name) not in (None,'','None')]
            bins=[]
            for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                subset=[r for r,x in values if x>=lo and (x<=hi if i==len(edges)-2 else x<hi)]
                successes=sum(r['success'] in ('True','true','1') for r in subset)
                means={}
                for metric in ('translation_cm','rotation_deg'):
                    valid=[float(r[metric]) for r in subset if r.get(metric) not in (None,'','None')]
                    means[metric]=dict(n=len(valid),mean=sum(valid)/len(valid) if valid else None)
                bins.append(dict(lower=lo,upper=hi,upper_inclusive=i==len(edges)-2,n=len(subset),
                                 successes=successes,failure_rate=1-successes/len(subset) if subset else None,errors=means))
            features[name]=dict(available_n=len(values),missing_n=len(selected)-len(values),bins=bins)
        output[arm]=features
    return dict(schema='FIXED_BIN_POSTHOC_AMBIGUITY_V1',population='PRIMARY192',arms=output,
                bins_fixed_to_theoretical_C4_entropy_and_unit_interval=True,
                no_gate_fit_or_threshold_retuning=True,association_is_not_calibration_or_safety=True,
                learned_softmax_relationship='N/A unless separately archived selected confidence; not inferred from A entropy')


def _selected_cases(results, new_arms):
    """Deterministic text witnesses, for every arm; figures remain a separate step."""
    cases=[]
    for arm in new_arms:
        pair=results['paired']['PRIMARY192'].get(arm+'_minus_E0')
        if pair is None:continue
        for kind in ('RECOVERY','DAMAGE'):
            ids=sorted(pair['ids'][kind])
            cases.append(dict(arm=arm,kind=kind,id=ids[0] if ids else None,
                              rule='lexicographically first immutable ID; postseal descriptive only'))
        for field in ('translation_cm','rotation_deg'):
            values=pair['both_failure_continuous'][field].get('largest_worsenings',[])
            cases.append(dict(arm=arm,kind='BOTH_FAILURE_WORSENING',metric=field,
                              case=values[0] if values else None,
                              rule='largest positive after-minus-before error; tie immutable ID'))
    return cases


def representative_head_readout(results,stage,csv_path):
    """Describe the predeclared first seed; never search for a best method."""
    arm='L_HEAD_s20261018'
    if arm not in results['populations']['PRIMARY192']:return None
    primary=[r for r in stage['rows'] if r['partition'] in ('TEST','REMAINDER')]
    selection={r['id']:r['arms'][arm].get('final_candidate_id') for r in primary}
    with Path(csv_path).open() as f:
        statuses={r['id']:r for r in csv.DictReader(f) if r['arm']==arm and r['partition'] in ('TEST','REMAINDER')}
    def source(cid):
        return 'N0' if cid=='N0' else 'E0' if cid=='E0' else 'NEW_LOCAL' if (cid or '').startswith('NEW/') else 'EXISTING_CARRIED'
    pair=results['paired']['PRIMARY192'][arm+'_minus_E0']
    recoveries=pair['ids']['RECOVERY']
    def compact(value):
        return {k:value[k] for k in ('denominator','successes','no_pose','translation_cm','rotation_deg',
                                     'T_gt50cm','T_gt100cm','R_gt45deg')}
    comparisons={}
    for before in COMPARATORS:
        value=results['paired']['PRIMARY192'][arm+'_minus_'+before]
        comparisons[before]=dict(success_difference=value['success_difference_count'],counts=value['counts'],
            recording_success_CI95=value['deltas']['success_rate']['recording_cluster_CI']['CI95'],
            named_session_success_CI95=value['deltas']['success_rate']['named_session_cluster_sensitivity_CI']['CI95'],
            comparator_summary=compact(results['populations']['PRIMARY192'][before]))
    return dict(method=arm,representative_seed=20261018,primary_best_seed_selected=False,
        successes=results['populations']['PRIMARY192'][arm]['successes'],
        primary_summary=compact(results['populations']['PRIMARY192'][arm]),comparisons=comparisons,
        partition_summaries={p:compact(results['populations'][p][arm]) for p in POPULATIONS},
        paired_vs_E0=pair['counts'],success_difference=pair['success_difference_count'],
        selected_source_counts=dict(Counter(source(cid) for cid in selection.values())),
        E0_recovery_selected_N0_n=sum(selection[fid]=='N0' for fid in recoveries),
        E0_recovery_fallback_N0_n=sum(selection[fid]=='N0' and statuses[fid].get('fallback')=='True' for fid in recoveries),
        actual_new_pose_recovery_attribution='Only exact selected candidate ID/status; no claim that every head gain came from new A/B solves')


def render_report(results, arms, new_arms, execution, stage, training, costs, decision, validation,
                  mechanism_available=False,synthetic_head_evaluation_available=False,appendix_summary=None,
                  synthetic_validation_readout=None):
    lines=['# 자동 상태별 end-to-end 회복 실험', '',
           '## 1. 전체 공통 비교표', '',
           '동일 영상·동일 physical C2 참조에서 같은 최종 자세의 중심 위치≤5cm AND 회전≤5°를 성공으로 판정했다. '
           '기존 6개와 새 등록 16개를 모두 표시한다. 미실행은 N/A이며 실패·fallback·no-pose를 분모에서 빼지 않았다. '
           '평균/표본SD(ddof=1)·분위수는 가용 자세에서 계산하며 가용 n을 함께 적었다. 평균을 낼 때 이상치를 제거하지 않았다.', '']
    lines += common_tables(results,arms,execution)
    lines += ['## 2. 성공 차이·회복·손상·큰 실패', '',
              '아래는 사전 지정 E0·N0·두 PoseFix와의 모든 짝지은 비교다. '
              '기록군집 재표집은 frame-weighted 비율을 유지하고, 이름상 session 결과는 민감도 분석이다. '
              '95% 구간에 0이 들어가는 것을 동등성으로 해석하지 않는다. 양쪽 실패의 연속오차 증가를 이진 손상0으로 숨기지 않는다.', '']
    lines += paired_tables(results,new_arms)
    policy=results['bootstrap_policy']
    lines += [f"Bootstrap: seed={policy['seed']}, 반복={policy['draws']}, 원 기록군집6개 및 이름상 session8개. "
              f"새 실행 arm {len(results['new_arms'])} × 비교4개 × 모집단4개 = {len(results['new_arms'])*16} contrasts. "
              '학습 seed3개는 같은 영상의 별도 방법이며 독립 영상576장으로 세지 않았다.', '',
              '## 3. 바뀐 계산과 표준 대조', '',
              '- A: globally consistent upright owner assignment와 none을 함께 갖는 정규화 component energy다. '
              'family2×C4의8 표기는 정확한 물리 owner 배열4개, 기존 C2까지 몫내면2개다. '
              '기존 W/D+heading 공간을 넓힌 새로운8 물리 가설이 아니다. 새 차이는 전역 owner mixture와 none, 공통 G13 재평가다. '
              'reliability로 temper한 합성 energy를 교정된 joint posterior로 부르지 않는다.',
              '- B: 동일 RGB와 자동 마스크에서 고정한 normal profile의 경계 offset 분포를 후보 전체에 공유한다. '
              '원 RGB query/분포를 고정하지만 후보 평가는 무한 지지선 연장을 허용한다(유한변 지원 한계는 후술). '
              'B가 소비한 full/partial 및 hull을 중복 가중하지 않는다. '
              '동일 지지의 edge 대조군과 구분해 경계 분포를 유지한 효과를 본다.',
              '- 후보 전달: 모든 실제 가용 이전 후보/endpoint를 보존하고 공통 G13 목적함수로 다시 계산한다. '
              '원 G10 값과 G13 값을 직접 빼거나 비교하지 않는다. 원 archive 수, 양의 깊이 진입 수, 고유 carried 수는 서로 다른 수다.',
              '- full solve: fixed/A/B/AB/edge는 같은 family별4개, 최대8개 제안 예산이다. '
              'CTRL_FIXED의 family별 N0 초기화1회(총2회), 표준 full-point RANSAC, fixed-bank 재선택은 별도 대조다.',
              '- L: 동일 유한 후보·같은62개 특징·같은 감독에서 선형과 작은2층 head를 각각3seed 학습한다. '
              'source별 원 대칭을 사용한 합성 감독은 C2 502장과 C1 501장이고, validation235장은 모두 C1이다. '
              'PRIMARY192의 기존 참조는 모두 C2다. C1의180° phase를 분별할 수 없는 특징/상태가 남을 수 있다. '
              '고정 config의 과도한 source-C1 문구는 숫자/label 변경 없는 prose amendment로 보존한다.', '',
              '합성 입력은 실제 renderer RGB와 기존 upstream의 오류 있는 출력 cache를 사용한다. '
              'B/I/S/T는 예측과 기하에서 만들지만 별도의 합성 external-O classifier는 없다. '
              'Tiny placeholder는 source-B 특징으로 사용하지 않는다. real O/T/U는 원 자동 관측을 계승한다. '
              'GT visibility나 GT mask를 feature로 쓰지 않았으며 합성 학습의 O 오류 coverage가 실사와 동등하다고 주장하지 않는다.', '',
              '합성에는 독립 E0 추론이 없고 initial/coarse/fallback은 같은 N3→SubPix→기존 F pose다. '
              '초기 bank는 N0와 두 F W/D 가설(중복 제거 전 최대3개)이며 원 partial·U 영역 관측 없이 fallback covariance를 사용한다. '
              '실사는 G10/G13의 가용 과거 bank와 별도 N0/E0, 원 full/partial/U/O 관측·보정 recipe를 보존한다. '
              '따라서 같은 solver·feature 계산 코드를 사용했다는 사실이 같은 관측·후보 분포에서 학습·평가했다는 뜻은 아니다. '
              '상세 지원 범위는 [DATA_SPLITS_AND_LABEL_QUALITY.md](DATA_SPLITS_AND_LABEL_QUALITY.md)의 비교표를 따른다.', '',
              '## 4. 고정 bank·생성 상한·실제 최종 선택', '',
              '오라클은 참조를 나중에 적용한 진단 상한이며 새 선택기의 검증 성능이 아니다. '
              '작은 위치오차 후보와 작은 회전오차 후보를 서로 합치지 않고 같은 후보의 공동 성공만 센다. '
              'CTRL_RANSAC의 전체 root가 저장되지 않은 부분과 CTRL_FIXED의 family 중간 선택은 ARCHIVE_GAP으로 남긴다. '
              'L은 직접 softmax 선택이므로 family 중간단계가 적용되지 않는다.', '']
    lines += stage_table(stage,new_arms)
    if mechanism_available:
        lines += ['이전 E0 성공114·family 선택10·같은-family 선택22·후보 부재46의 고정 strata별 '
                  '회복/손상 및 N0/두PF 비교, 합성 train/validation의 학습 대조는 '
                  '[MECHANISM_DIAGNOSIS_KO.md](MECHANISM_DIAGNOSIS_KO.md)와 '
                  '[MECHANISM_DIAGNOSIS.json](MECHANISM_DIAGNOSIS.json)에 전량 기록했다. '
                  '후보 오라클·constraint loss·선택 loss를 실제 새 성능과 구분한다.', '']
    lines += ['## 5. 회복·손상·공동실패 악화 사례', '',
              '모든 ID와 양쪽 오차는 [PAIRED_TRANSITIONS.csv](PAIRED_TRANSITIONS.csv), '
              '대표 선택 규칙과 수치는 [REPRESENTATIVE_CASES.json](REPRESENTATIVE_CASES.json)에 있다. '
              '원예측·실제영상관측·모델 투영·참조는 서로 다른 의미다. 본 Markdown 생성만으로 RGB를 직접 확인했다고 주장하지 않는다. '
              '실제 그림/단계 갤러리는 별도 renderer의 봉인/검수 기록으로 확인한다.', '']
    diagnostics=[]
    for arm in new_arms:
        summary=results['populations']['PRIMARY192'].get(arm)
        if not summary:continue
        a=summary.get('application_diagnostics',{})
        diagnostics.append([arm,a.get('B_active_n','N/A'),a.get('B_unsupported_n','N/A'),
                            a.get('B_application_unknown_n','N/A'),a.get('unchanged_from_N0_n','N/A'),
                            a.get('unchanged_from_E0_n','N/A'),a.get('success_with_B_unapplied_n','N/A')])
    lines += [_table(['방법','B 적용','B 비적용','B 불명','N0 동일','E0 동일','B 비적용 상태 성공'],diagnostics), '',
              'B 비적용 성공은 해당 영상에서 새 경계 보정의 회복을 입증하지 않는다. '
              'A-only/고정/RANSAC 경로의 B 비적용은 설계상 정상이며 곧바로 실패로 해석하지 않는다. '
              'A owner entropy와 none 평균의 실제 오류 관계는 [AMBIGUITY_POSTHOC.json](AMBIGUITY_POSTHOC.json)의 '
              '이론 범위 고정4구간으로만 사후 집계한다. 결측은 별도로 세며 gate를 새로 fit하지 않는다. '
              'L softmax confidence가 metric 장부에 없으면 해당 관계를 관측했다고 주장하지 않는다.', '',
              '## 6. 선행 원리·실증 차이·신규성의 한계', '',
              'A는 EPOS의 대안 대응을 하나의 관측에 묶는 원리와 mixture/null 원리를 참고했으며 EPOS 네트워크나 max-mixture SLAM 재현이 아니다. '
              'B는 SRT3D의 FG/BG normal-profile 관측 원리를 별도 NumPy 구현했다. '
              'L은 저장된 후보의 이산 감독이며 EPro-PnP 연속 자세 분포 학습 재현이 아니다. '
              '구현 출처와 license/제외 범위는 [LITERATURE_TO_IMPLEMENTATION.md](LITERATURE_TO_IMPLEMENTATION.md)에 정리한다. '
              '기존 같은-edge mixture와 A의 전역 owner 혼합은 계산 단위가 다르지만 새로운 물리 대칭을 발견한 것은 아니다.', '',
              '## 7. 실제 실행·모든 시도·비용·확인 공백', '',
              '처음 real_batch_v1은 실제 가용 과거 후보/endpoint의 입력 누락 때문에52개 sealed frame 뒤 중지했다. '
              '당시 새 PRIMARY 결과를 읽지 않았고, 원 시도/실패를 지우지 않은 채 입력 계약만 고쳐 real_batch_all_available_v1을 별도로 실행했다. '
              '최종 arm은 수정 입력 전체238개를 사용한다. source 축·T-domain·feature 포장·symmetry 설명의 기술 보정은 각 원 장부와 해시로 연결한다. '
              '이전 시도의 비용도 최종 방법의 비용과 별도로 보존한다.', '',
              '프레임당42회는 여러 경로의 local SLSQP 시도 합계다. 별도의 최대2회 full-point RANSAC 호출이나 '
              'synthetic 입력 생산의 SQPnP/LM을 포함한 전체PnP 횟수가 아니다. 실제 RANSAC 상태·호출 수는 '
              'COST_AND_EXECUTION의 real_actual_call_counts, synthetic 생산 F/adapter/재시도 횟수는 원 operation_counts와 '
              'root 비용 장부로 구분한다. 후보별 solver elapsed를 전체 end-to-end FPS로 바꾸지 않는다.', '',
              '참조를 읽지 않은 독립 입력 coverage 검산은238프레임의 원래 가용 R/t 역할29,564개를 모두 보존했음을 확인했다. '
              '양의 깊이 조건으로29,556개가 실제 진입했고8개는 archive에 남긴 채 projection에서 제외됐다. '
              '회전이 원 기록에 없는 outcome은 새 자세로 복원한 척하지 않는다. '
              '학습 전 합성 감독 audit의 read-only63초 중단 및 혼합 대칭을 모두C1로 가정한 assertion 실패도 보존한다. '
              '일시적인 보조 audit fifth-CPU 실행은 즉시 중지했으며 수치·GT·후보를 바꾸지 않았다. '
              '정확한 최종 CPU/RSS를 측정하지 않은 기술 시도는 unknown이다.', '',
              f"실제 학습 실행 수: **{training['total_training_runs']}회**. "
              '미실행은 실행한 것으로 세지 않으며 checkpoint·epoch·step·exposure·자원은 TRAINING_MANIFEST.json이 권위 기록이다.', '']
    trainrows=[]
    for kind in ('L_LINEAR','L_HEAD'):
        for seed in (20261018,20261019,20261020):
            item=next((v for v in training.get('runs',[]) if v['kind']==kind and v['seed']==seed),None)
            if item:
                c=item.get('cost',{})
                trainrows.append([kind,seed,item['epochs'],item['best_epoch'],item['optimizer_steps'],item['training_frame_exposures'],
                                  _fmt(c.get('wall_seconds')), _fmt(c.get('CPU_seconds',c.get('process_cpu_seconds'))),
                                  item.get('parameters','N/A'),item.get('checkpoint_sha256','N/A')])
            else:trainrows.append([kind,seed]+['N/A']*8)
    lines += [_table(['구조','seed','epoch','best epoch','steps','frame exposures','벽시계s','CPUs','params','best SHA'],trainrows), '']
    if not training['total_training_runs']:
        lines += ['학습 미실행 이유: '+json.dumps(training.get('reason'),ensure_ascii=False), '']
    if synthetic_head_evaluation_available:
        lines += ['고정6개 checkpoint의 실제 합성 train/validation 공동성공·후보 oracle·보류·N0 fallback·전체/good/no-good CE는 '
                  '[SYNTHETIC_LEARNED_EVALUATION_KO.md](SYNTHETIC_LEARNED_EVALUATION_KO.md)와 '
                  '[SYNTHETIC_LEARNED_EVALUATION.json](SYNTHETIC_LEARNED_EVALUATION.json)에 있다. '
                  'CE만으로 자세 성공을 대신하지 않는다. source별 원 C1/C2를 유지했으며 validation은 early stopping에 사용한 개발자료다. '
                  'log-softmax 보존을 위한 추가6,018 tiny-head forward는 새 학습·seed 선택이 아니다.', '']
    if synthetic_validation_readout:
        v=synthetic_validation_readout
        lines += [f"합성 validation에서 사전 대표 head는{v['representative_head_successes']}/{v['denominator']}로 "
                  f"원 N0의{v['N0_successes']}/{v['denominator']}보다 낮다. "
                  f"좋은 eligible 후보가 있는 oracle은{v['eligible_candidate_oracle_successes']}/{v['denominator']}다. "
                  'PRIMARY115/192만 앞세워 학습의 전이를 입증하지 않는다. validation은 early stopping에 사용한 C1 한 asset이고, '
                  '실사의 원 C2·독립N0/E0·과거 bank·partial/U/O 관측 패키지와 후보/불확실성 분포가 다르다.', '']
        lines += [f"좋은 후보가 없는{v['no_good_frames']}장도 유지했다. 대표 head의 보류{v['abstentions']}장은 "
                  f"good 존재{v['good_but_abstained']}장과 no-good{v['no_good_abstained']}장으로 나뉜다. "
                  f"전체6개 seed의 validation 성공은{json.dumps(v['all_seed_successes'],ensure_ascii=False)}다. "
                  'CE가 개선된 것과 N0보다 최종 pose가 나은 것은 별도 결과다.', '']
    lines += ['비용은 [COST_AND_EXECUTION.json](COST_AND_EXECUTION.json)의 실제 scope별 기록을 따른다. '
              '동시에 실행한 batch의 벽시계를 합쳐 end-to-end 시간이라고 하지 않는다. '
              '확장 synthetic의 재사용512 프레임을 pilot 비용에 다시 더하지 않으며 RSS high-water mark는 합산하지 않는다. '
              f"현재 private 고유 inode 점유: {costs['private_disk_unique_inode_bytes']/(1024**3):.3f} GiB. "
              '측정하지 않은 총비용은 unknown으로 남긴다.', '',
              f"캠페인 시작 epoch={costs['total_known_cost']['campaign_start_epoch']}부터 보고 시점까지 실제 경과 "
              f"{costs['total_known_cost']['campaign_elapsed_wall_seconds']/3600:.3f}시간. "
              f"기록된 추가 GPU {costs['total_known_cost']['GPU_seconds']:.6f}초 "
              f"({costs['total_known_cost']['GPU_seconds']/3600:.6f} GPUh), "
              f"고유 scope의 알려진 CPU 소계 {costs['total_known_cost']['CPU_seconds_known_subtotal']/3600:.3f} CPUh. "
              'CPU 소계는 미측정 audit/제어기 비용이 남은 하한이다. 기존 upstream의 역사적 생산비는 별도 unknown이다.', '',
              '동결 자원 상한은 캠페인24시간, 추가GPU8시간, solver/analysis batch 벽시계10시간, 새 디스크25GiB, '
              'CPU worker4개다. 위 캠페인 경과·CUDA event GPU활동·고유inode 점유는 서로 다른 측정 범위다. '
              f"완료한 pilot512/확장491추가/실사238 batch의 job 벽시계 소계는 "
              f"{costs['total_known_cost']['completed_solver_batch_cumulative_wall_seconds']/3600:.3f}시간이며 "
              '동시 실행한 job 시간을 더한 작업 scope 소계이지 end-to-end 경과시간이 아니다. '
              '중단52·기술검사·분석 및 미측정 비용을 포함한 전체 예산 검토는 함께 연결한 root 비용 장부를 따른다. '
              'pilot512와 확장1003 CPU를 이중 합산하지 않고, 최종 재사용 mask receipt의0신규호출로 최초512의18.59초를 지우지 않았다. '
              '보조 합성 audit의 일시적 fifth-CPU 예외는 위 중단 기록에 별도로 공개한다.', '',
              f"독립 산술검산: {validation['independent_check']['status']}; "
              f"failure_count={validation['independent_check']['failure_count']}. "
              '독립 metric 코드는 공통 evaluator를 import하지 않고 실제 저장 pose의 C2와 공동 성공, 분모·분포·paired 결과를 다시 계산한다. '
              '이는 물리 GT의 독립 확인이나 새 촬영 validation이 아니다.', '',
              'PRIMARY192는 과거 개발과 진단에 반복 노출됐다. CAL46은 별도 calibration/진단이며 주평가에 합치지 않았다. '
              '학습용 upstream이 과거 source assets를 본 이력, 검증 asset 하나·C1 phase 한계, 기존 geometric reference와 '
              '실제 물리 owner/pose의 독립 확인 공백을 유지한다. 새 유리한 정답·평가 대칭으로 과거 실패를 바꾸지 않았다.', '',
              '## 8. 최종 권고와 남은 최소 확인', '',decision['recommendation_ko'], '',
              '개발 판정은 사전 정의 범주의 보수적 기술 분류다. 모든 비교군보다 성공 수가 높을 때만 성공 우위를 인정하고, '
              '평균 T/R·큰오류3종·no-pose가 어느 비교군보다 나쁘면 tradeoff로 남긴다. '
              '연속오차 개선 분류는 일부 개선의 존재를 뜻하며 다른 열의 악화를 숨기지 않는다. '
              '통계적/실사용 우월성 또는 물리 안전을 보장하지 않는다.', '',
              _table(['방법','개발 판정','이유'],[[a,decision['arms'][a]['category'],decision['arms'][a]['reason']] for a in new_arms]), '',
              '대표 seed는 사전 고정20261018이다. validation epoch/구조 선택과 관측된 PRIMARY 최고 행은 다른 개념이다. '
              '여기서는 PRIMARY 최고 seed를 운영 대표로 교체하지 않는다. 현재 운영 기본값·기존 N3/SubPix/e·모든 과거 결과·논문은 유지한다.', '']
    additions=[];appendix_summary=appendix_summary or {}
    direction=appendix_summary.get('rounding_direction')
    if direction:
        additions += [f"독립검산 v1의{direction['previous_failed_fields']}개 방향 계수 차이는 "
            f"{direction['frames']}영상·{direction['sign_switch_rows']}개 pair/metric 행의 strict-zero 부동소수점 표현 차이로 전부 설명했다. "
            f"원 evaluator 표현의{direction['bitwise_original_expression_checks']}개 metric 값을 bitwise 재현했고, "
            'stable atan2/fsum의 성공·오차 크기·분포·CI와 원 공개 계수는 바꾸지 않았다. '
            '미세한 부호를 물리적으로 의미 있는 회복/악화라고 해석하지 않는다. '
            '[ROUNDING_DIRECTION_AUDIT.json](ROUNDING_DIRECTION_AUDIT.json), '
            '[원 실패 v1](INDEPENDENT_CHECK_V1_FAILED.json)을 함께 보존한다.', '']
    if appendix_summary.get('positive_mass'):
        mass=appendix_summary['positive_mass']
        additions += ['[POSITIVE_MASS_DIAGNOSIS_KO.md](POSITIVE_MASS_DIAGNOSIS_KO.md)는 합성 validation의 '
            'good 후보 확률합과 none의 관계를 원 분포/라벨로만 진단했다. GT로 정의한 good 집합의 합은 실행 시 알 수 없는 값이며, '
            '새 선택 규칙·회복 성능·보류 gate로 바꾸지 않았다. 독립 질량 계수 검산도 함께 공개한다.', '']
        if mass.get('predeclared_head_good_but_abstained') is not None:
            additions += [f"대표 head는 좋은 후보가 있는데 보류한{mass['predeclared_head_good_but_abstained']}장 중 "
                f"GT로 정의한 good 확률합>none인 경우{mass['good_mass_sum_gt_none']}장, "
                f"≤none인 경우{mass['good_mass_sum_le_none']}장이었다. "
                '일부 질량 분산과 일치하지만 보류 실패의 주원인으로 단정할 수 없다. 이 숫자는 새 runtime 규칙의 회복 성능이 아니다.', '']
    if appendix_summary.get('boundary_extreme'):
        additions += ['[BOUNDARY_EXTREME_FORENSICS_KO.md](BOUNDARY_EXTREME_FORENSICS_KO.md)의 고정 pose 재평가는 '
            'B862cm 선택의 교차 t≈−18.4(유한변 내부0/1), edge의 축소 투영0/2, CAL AB의 축소 투영0/6을 확인했다. '
            '원 구현은 후보의 무한 지지선과 normal 교차를 평가하며 후보 변의 t∈[0,1]을 검사하지 않는다. '
            '따라서 **산술검산PASS는 physical-support 계약PASS가 아니다.** finite projected-edge/physical-edge라는 문구가 '
            '실제 보존 범위를 충분히 설명하지 못했으며, B/AB가 물리 근거상 우월하다는 주장은 기각한다. '
            '이는 동결 config에 새 제약 위반을 소급 선언하는 것과 별개인 표현/지원 범위의 한계다. 원 출력은 보존하며 사후 B2를 만들지 않았다.', '']
    head=decision.get('predeclared_representative_head')
    if head:
        additions += [f"사전 대표 head는{head['successes']}/192, E0 대비 회복{head['paired_vs_E0']['RECOVERY']}·"
            f"손상{head['paired_vs_E0']['DAMAGE']}이다. E0 회복 중{head['E0_recovery_fallback_N0_n']}장은 N0 보류/fallback이며 "
            f"전체 선택 출처는{json.dumps(head['selected_source_counts'],ensure_ascii=False)}다. "
            '이 회복을 새 A/B pose 생성의 직접 성과로 바꾸지 않는다. 전체22방법·원49/추가143·평균/꼬리·CI 해석은 '
            '[ROOT_RESULT_INTERPRETATION_KO.md](ROOT_RESULT_INTERPRETATION_KO.md)에 연결한다.', '']
    text='\n'.join(lines)
    if additions:text=text.replace('## 8. 최종 권고와 남은 최소 확인','\n'.join(additions)+'\n## 8. 최종 권고와 남은 최소 확인')
    return text


def run(run_dir, *, output_dir=None, evaluation_dir=None, independent_receipt=None,
        source_root=None, gallery_receipt=None, cost_ledger=None, mechanism_dir=None, report_GO=False):
    if report_GO is not True:
        raise PermissionError('Root explicit report GO is required')
    start=time.monotonic();run=Path(run_dir).resolve()
    evaluation=Path(evaluation_dir or run/'evaluation_v1')
    independent_path=Path(independent_receipt or run/'independent_v2'/'INDEPENDENT_CHECK.json')
    source_root=Path(source_root or Path(__file__).resolve().parents[3])
    prediction, evaluation_seal, independent=verify_report_inputs(run,evaluation,independent_path,report_GO=True)
    # New numerical results are first decoded only after the entire gate above.
    results=read(evaluation/'RESULTS.json');stage=read(evaluation/'CANDIDATE_STAGE_DIAGNOSIS.json')
    registry=read(run/'ARM_REGISTRY.json');arms,new_arms=_inventory(registry)
    for population in POPULATIONS:
        if any(v['denominator']!=DENOMINATORS[population] for v in results['populations'][population].values()):
            raise ValueError('Full population denominator changed')
    if set(results['arms'])-set(arms):raise ValueError('Unregistered arm in final comparison')
    execution=_execution_status(run,results,new_arms)
    synthetic_evaluation=verify_synthetic_head_evaluation(run)
    appendices=optional_appendices(run,independent_path,independent)
    gallery=read(gallery_receipt) if gallery_receipt is not None else None
    training=_training_manifest(run);costs=_cost(run,training,stage,independent=independent,results=results,
                                               gallery_receipt=gallery,cost_ledger=cost_ledger)
    decisions={a:classify(a,results['populations']['PRIMARY192']) for a in new_arms}
    promising=[a for a,v in decisions.items() if v['category']=='PROMISING_FOR_INDEPENDENT_VALIDATION']
    recommendation=('운영 기본값은 유지한다. 사전 고정 대표와 전체 손익이 지지하는 경로만 독립 세션·물리 owner 확인으로 넘긴다. '
                    '현재 자료의 관측상 우위로 운영 채택하거나 PRIMARY 최고 seed를 고르지 않는다.' if promising else
                    '운영 기본값은 유지한다. 이번 결과만으로 새 자동 경로를 채택하지 않고, 실제 physical owner·phase를 독립 확인할 최소 새 자료를 먼저 확보한다. '
                    '성공 수·평균·꼬리의 손익은 모두 보존하며 PRIMARY 최고 seed를 대안으로 고르지 않는다.')
    decision=dict(schema='CONSERVATIVE_DEVELOPMENT_DECISION_V1',arms=decisions,
                  recommendation_ko=recommendation,operating_default_changed=False,paper_changed=False,
                  independent_validation_performed=False,representative_seed=20261018,
                  primary_best_seed_selected=False,decision_is_not_deployment_selection=True,
                  source_C1_C2_vs_primary_C2_limitation=True)
    head_readout=representative_head_readout(results,stage,evaluation/'ROW_METRICS.csv')
    decision['predeclared_representative_head']=head_readout
    decision['recommendation_ko']=('운영 기본값은 유지한다. 다음 한 가지는 원 N0↔기존E0 전환의 근거를 '
        '독립 실사 자료에서 확인하는 것이다. 사전 대표 head의 작은 성공 이득과 회복/손상·평균/꼬리 tradeoff를 보존하되, '
        '이번 PRIMARY 결과로 새 gate를 fit하거나 실행하지 않고 최고 seed로 교체하지 않는다.')
    decision['physical_support_contract']=dict(arithmetic_PASS_is_not_physical_support_PASS=True,
        finite_candidate_edge_range_guard_implemented=False if appendices['summary'].get('boundary_extreme') else None,
        B_AB_physical_superiority_claim='REJECTED' if appendices['summary'].get('boundary_extreme') else 'NOT_CERTIFIED',
        frozen_explicit_constraint_violation_claimed=False,posthoc_B2_constructed=False,
        physical_owner_external_occlusion_independently_certified=False)
    decision['synthetic_validation_readout']=synthetic_evaluation['validation_readout'] if synthetic_evaluation else None
    bindings=_bindings(run,source_root,prediction,evaluation_seal,independent_path)
    if synthetic_evaluation:
        bindings['synthetic_head_evaluation']=dict(seal_sha256=synthetic_evaluation['seal_sha256'],
            result_sha256=synthetic_evaluation['result_sha256'],markdown_sha256=synthetic_evaluation['markdown_sha256'])
    bindings['optional_appendix_inputs']=appendices['input_bindings']
    if cost_ledger:bindings['root_cost_ledger']=dict(path=str(cost_ledger),sha256=sha(cost_ledger))
    validation=dict(schema='REPORT_VALIDATION_BINDINGS_V1',independent_check=independent,
                    scientific_test_receipts={str(p):sha(p) for p in sorted((source_root/PUBLIC_NAMESPACE).glob('*CHECK*.json'))},
                    all_22_registered_rows_visible=True,all_4_populations_visible=True,
                    all_failures_and_unexecuted_preserved=True,raw_RGB_review_not_implied_by_report_generation=True,
                    prediction_evaluation_check_hash_chain_passed=True)
    validation['physical_support_contract']=decision['physical_support_contract']
    validation['optional_appendices']=appendices['summary']
    validation['gallery_receipt']=dict(path=str(gallery_receipt),sha256=sha(gallery_receipt),
                                      raw_RGB_review_claim='See separate human inspection receipt; renderer alone is insufficient') if gallery_receipt else None
    visual_review=run/'ROOT_ACTUAL_VISUAL_REVIEW.json'
    validation['root_actual_visual_review']=dict(path=str(visual_review),sha256=sha(visual_review),
        review_claim_owned_by_receipt_author=True,report_generation_alone_implies_image_review=False) if visual_review.is_file() else None
    mechanism=Path(mechanism_dir) if mechanism_dir else None
    if mechanism:
        required=('MECHANISM_DIAGNOSIS.json','MECHANISM_DIAGNOSIS_KO.md','MECHANISM_SEAL.json')
        if not all((mechanism/name).is_file() for name in required):raise ValueError('Both mechanism numeric/prose outputs required')
        mechanism_seal=read(mechanism/'MECHANISM_SEAL.json')
        if (mechanism_seal.get('complete') is not True or mechanism_seal.get('explicit_root_GO') is not True
                or sha(evaluation/'EVALUATION_SEAL.json') not in mechanism_seal.get('input_bindings',{}).values()
                or sha(run/'PREDICTION_SEAL.json') not in mechanism_seal.get('input_bindings',{}).values()):
            raise ValueError('Mechanism diagnosis must bind these unchanged predictions/evaluation')
        _sealed_artifacts(mechanism,mechanism_seal)
        bindings['mechanism_artifacts']={str(mechanism/name):sha(mechanism/name) for name in required}
    output=Path(output_dir or run/'report_v1');output.mkdir(parents=True,exist_ok=False)
    write(output/'SOURCE_BINDINGS.json',bindings)
    write(output/'TRAINING_MANIFEST.json',training)
    write(output/'COST_AND_EXECUTION.json',costs)
    write(output/'VALIDATION.json',validation)
    write(output/'FINAL_DECISION.json',decision)
    write(output/'REPRESENTATIVE_CASES.json',dict(cases=_selected_cases(results,new_arms),images_reviewed=False))
    write(output/'AMBIGUITY_POSTHOC.json',ambiguity_posthoc(evaluation/'ROW_METRICS.csv'))
    if mechanism:
        for name in ('MECHANISM_DIAGNOSIS.json','MECHANISM_DIAGNOSIS_KO.md','MECHANISM_SEAL.json'):
            (output/name).write_bytes((mechanism/name).read_bytes())
    if synthetic_evaluation:
        for name in synthetic_evaluation['files']:
            (output/name).write_bytes((synthetic_evaluation['folder']/name).read_bytes())
    for item in appendices['files']:
        if item['transform']:
            value=item['transform'](read(item['source']))
            value['public_derivation']=dict(original_private_sha256=sha(item['source']),full_pose_matrices_removed=True,
                remaining_scores_metrics_ids_and_support_diagnostics_unchanged=True)
            write(output/item['name'],value)
        else:(output/item['name']).write_bytes(item['source'].read_bytes())
    interpretation=run/'ROOT_RESULT_INTERPRETATION_KO.md'
    if interpretation.is_file():(output/interpretation.name).write_bytes(interpretation.read_bytes())
    write(output/'PREDICTION_MANIFEST.json',dict(schema='ACTUAL_PREDICTION_EXECUTION_MANIFEST_V1',
          seal=prediction,execution_status=execution,baseline_reuse=True,old_six_poses_unchanged=True,
          all238frames_kept=True,population_failures={p:{a:dict(failures=v['failures'],no_pose=v['no_pose'],
          metric_error_ids=v.get('metric_error_ids',[]),status_counts=v['status_counts'])
          for a,v in results['populations'][p].items()} for p in POPULATIONS}))
    write(output/'RUN_STATE.json',dict(schema='FINAL_SNAPSHOT_WITH_PROVENANCE_V1',
          historical_run_state=_optional(run/'RUN_STATE.json'),controller_state=_optional(run/'CONTROLLER_STATE.json'),
          execution_status=execution,report_completed=True,publication_completed=False,
          note='A report snapshot does not rewrite controller stage proofs or imply git publication'))
    text=render_report(results,arms,new_arms,execution,stage,training,costs,decision,validation,
                       mechanism is not None,synthetic_evaluation is not None,appendices['summary'],
                       synthetic_evaluation['validation_readout'] if synthetic_evaluation else None)
    if (run/'PUBLICATION_PROSE_REPAIR_V2.json').is_file():
        text=text.replace('## 8. 최종 권고와 남은 최소 확인',
            '최종 검토 후 B의 관측 생성과 후보 기하 연장을 구분하도록 문구만 수정했다. '
            '원 보고서·봉인은 보존했으며 수치는 변경하지 않았다. '
            '[PUBLICATION_PROSE_REPAIR_V2.json](PUBLICATION_PROSE_REPAIR_V2.json)에 재생성 경위를 연결한다. '
            '최종 V2 게시 재현 명령은 [REPRODUCE_REVIEWED_PUBLICATION_V2.md](REPRODUCE_REVIEWED_PUBLICATION_V2.md)를 따른다.\n\n'
            '## 8. 최종 권고와 남은 최소 확인')
    if gallery_receipt:
        private_index=Path(gallery_receipt).parent/'index.html'
        text+='\n단계별 실제 RGB 갤러리(private): '+str(private_index)+'. '
        numeric_folder=Path((gallery or {}).get('public_numeric_export',{}).get('directory','metrics_gallery_v3')).name
        text+=f'공개 숫자 산점도: [{numeric_folder}/index.html]({numeric_folder}/index.html). '
        text+='기본 log1p(T/cm)와 전체범위 linear 전환을 제공하고 tick·tooltip은 원 cm다. 꼬리를 제거하지 않았으며 private RGB는 공개하지 않는다.\n'
    (output/'FINAL_REPORT_KO.md').write_text(text)
    # Results and row-level CSVs are copied by publish.py only after redaction.
    write(output/'REPORT_SEAL.json',dict(schema='POSTSEAL_REPORT_SEAL_V1',complete=True,explicit_root_report_GO=True,
          evaluation_dir=str(evaluation),evaluation_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'),
          prediction_seal_sha256=sha(run/'PREDICTION_SEAL.json'),plan_sha256=sha(run/'PLAN_LOCK.json'),
          independent_receipt=str(independent_path),independent_receipt_sha256=sha(independent_path),
          synthetic_head_evaluation_seal_sha256=synthetic_evaluation['seal_sha256'] if synthetic_evaluation else None,
          artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()},elapsed_seconds=time.monotonic()-start))
    return dict(output_dir=str(output),registered_arms=len(arms),training_runs=training['total_training_runs'],
                independent_status=independent['status'],report_seal_sha256=sha(output/'REPORT_SEAL.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--output-dir');p.add_argument('--evaluation-dir');p.add_argument('--independent-receipt')
    p.add_argument('--source-root');p.add_argument('--gallery-receipt');p.add_argument('--cost-ledger');p.add_argument('--mechanism-dir')
    p.add_argument('--report-GO',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.run_dir,output_dir=a.output_dir,evaluation_dir=a.evaluation_dir,
          independent_receipt=a.independent_receipt,source_root=a.source_root,gallery_receipt=a.gallery_receipt,
          cost_ledger=a.cost_ledger,mechanism_dir=a.mechanism_dir,report_GO=a.report_GO),ensure_ascii=False))


if __name__=='__main__':main()
