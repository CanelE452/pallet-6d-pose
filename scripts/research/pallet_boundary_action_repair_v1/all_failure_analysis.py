"""Post-seal H1 all-action-failure continuous-error diagnosis (§7.2).

No references, model forward, pose metric, optimizer, bank or prediction journal
is opened. Only the already evaluated metrics and arm-specific eligible-action
diagnosis are decoded, after --analysis-GO and immutable input checks. This is
descriptive diagnosis, never a new selector's validation score. The separate
output tree preserves the original posthoc_v1 seal/tree.

python -m scripts.research.pallet_boundary_action_repair_v1.all_failure_analysis \
  --run-dir /registered/run --evaluation /registered/run/evaluation_v1 \
  --output /registered/run/h1_all_failure_v1 --analysis-GO
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from .registry import ALL_ARMS, EXPECTED_N, L_ARMS, POPULATIONS

H1_ARMS = tuple(arm for arm in L_ARMS if '_H1_' in arm)
REFERENCES = ('N0', 'E0', 'E0_REGENERATED')
JOURNALS = ('SCORED_ROWS.jsonl.gz', 'STAGE_DIAGNOSIS.jsonl.gz')
AXES = (('T', 'translation_cm'), ('R', 'rotation_deg'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def raw_scalar(value):
    """Keep nonfinite input identity without writing invalid JSON numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return '+Infinity' if value > 0 else '-Infinity'
        return value
    return repr(value)


def side(row, arm):
    """Read availability only; never decode or recompute R/t pose geometry."""
    poses, metrics = row.get('poses', {}), row.get('metrics', {})
    pose, metric = poses.get(arm), metrics.get(arm)
    metric = metric if isinstance(metric, dict) else {}
    return dict(actual_pose_present=isinstance(pose, dict),
        actual_pose_available=pose.get('available') if isinstance(pose, dict) else None,
        metric_present=isinstance(metrics.get(arm), dict),
        metric_pose_available=metric.get('pose_available'),
        success=metric.get('success'), metric_error=metric.get('metric_error'),
        **{key: metric.get(key) for _, key in AXES})


def axis_reasons(value, key):
    reasons = []
    if not value['actual_pose_present']:
        reasons.append('missing_actual_pose')
    elif value['actual_pose_available'] is not True:
        reasons.append('actual_pose_unavailable' if value['actual_pose_available'] is False else 'actual_pose_availability_unknown')
    if not value['metric_present']:
        reasons.append('missing_metric')
    elif value['metric_pose_available'] is not True:
        reasons.append('metric_pose_unavailable' if value['metric_pose_available'] is False else 'metric_pose_availability_unknown')
    number = value[key]
    if number is None:
        reasons.append('null_error')
    elif isinstance(number, bool) or not isinstance(number, (int, float)):
        reasons.append('invalid_error_type')
    elif not math.isfinite(number):
        reasons.append('nonfinite_error')
    return reasons


def comparison_row(row, arm, reference, stage):
    selected, baseline = side(row, arm), side(row, reference)
    source = row.get('action_sources', {}).get(arm, {})
    baseline_source = row.get('action_sources', {}).get(reference, {})
    status = row.get('status', {}).get(arm, {})
    ref_status = row.get('status', {}).get(reference, {})
    result = dict(id=row['id'], partition=row['partition'], session=row.get('session'),
        cluster_id=row.get('cluster_id'), arm=arm, comparator=reference,
        selected_candidate_id=stage.get('selected_candidate_id'),
        eligible_candidate_count=stage['after_constraints'].get('candidate_count'),
        eligible_available_n=stage['after_constraints'].get('available_n'),
        eligible_has_success=False, selected_source=source.get('role'),
        selected_aliases_json=json.dumps(source.get('aliases'), ensure_ascii=False, sort_keys=True),
        comparator_source=baseline_source.get('role'),
        selected_fallback=status.get('fallback'), selected_status=status.get('status'),
        selected_technical_failure=status.get('technical_failure'),
        comparator_fallback=ref_status.get('fallback'), comparator_status=ref_status.get('status'))
    for prefix, value in (('selected', selected), ('comparator', baseline)):
        for key in ('actual_pose_present', 'actual_pose_available', 'metric_present',
                    'metric_pose_available', 'success', 'metric_error'):
            result[prefix + '_' + key] = value[key]
    for label, key in AXES:
        first, second = axis_reasons(selected, key), axis_reasons(baseline, key)
        result['selected_' + key] = raw_scalar(selected[key])
        result['comparator_' + key] = raw_scalar(baseline[key])
        result[label + '_selected_exclusion_reasons'] = ';'.join(first)
        result[label + '_comparator_exclusion_reasons'] = ';'.join(second)
        comparable = not first and not second
        delta = selected[key] - baseline[key] if comparable else None
        # Overflow is not silently turned into a valid extreme or a zero.
        if delta is not None and not math.isfinite(delta):
            comparable, delta = False, None
            result[label + '_delta_exclusion_reason'] = 'nonfinite_subtraction'
        else:
            result[label + '_delta_exclusion_reason'] = ''
        result[label + '_comparable'] = comparable
        result[label + '_delta'] = delta
        for direction, predicate in (('worse', lambda n: n > 0),
                                     ('improved', lambda n: n < 0), ('equal', lambda n: n == 0)):
            result[label + '_' + direction] = predicate(delta) if comparable else None
    result['both_axes_comparable'] = result['T_comparable'] and result['R_comparable']
    for direction in ('worse', 'improved'):
        result[direction + '_either'] = (result['T_' + direction] or result['R_' + direction]
            if result['both_axes_comparable'] else None)
    result['equal_both'] = (result['T_equal'] and result['R_equal']
        if result['both_axes_comparable'] else None)
    return result


def moments(values):
    if not values:
        return dict(n=0, sum=None, mean=None, min=None, max=None,
                    aggregate_nonfinite=False)
    total = sum(values)
    finite = math.isfinite(total)
    return dict(n=len(values), sum=total if finite else None,
        mean=total / len(values) if finite else None, min=min(values), max=max(values),
        aggregate_nonfinite=not finite)


def reason_counts(values, key):
    return dict(sorted(Counter(reason for row in values
        for reason in row[key].split(';') if reason).items()))


def summarize(values, population, arm, comparator, denominator):
    result = dict(population=population, arm=arm, comparator=comparator,
        population_denominator=denominator, all_failure_n=len(values),
        not_all_failure_n=denominator-len(values),
        both_axes_comparable_n=sum(row['both_axes_comparable'] for row in values),
        both_axes_excluded_n=sum(not row['both_axes_comparable'] for row in values),
        worseEither_n=sum(row['worse_either'] is True for row in values),
        improvedEither_n=sum(row['improved_either'] is True for row in values),
        equalBoth_n=sum(row['equal_both'] is True for row in values),
        selected_source_counts=dict(sorted(Counter(row['selected_source'] or 'UNKNOWN' for row in values).items())),
        selected_fallback_counts=dict(sorted(Counter('true' if row['selected_fallback'] is True else
            'false' if row['selected_fallback'] is False else 'unknown' for row in values).items())),
        selected_status_counts=dict(sorted(Counter(str(row['selected_status']) if row['selected_status'] is not None else 'UNKNOWN'
            for row in values).items())))
    for prefix in ('selected', 'comparator'):
        for flag in ('actual_pose_present', 'actual_pose_available', 'metric_present', 'metric_pose_available'):
            key = prefix + '_' + flag
            result[key + '_true_n'] = sum(row[key] is True for row in values)
            result[key + '_false_n'] = sum(row[key] is False for row in values)
            result[key + '_unknown_n'] = sum(row[key] is not True and row[key] is not False for row in values)
    for label, _ in AXES:
        deltas = [row[label + '_delta'] for row in values if row[label + '_comparable']]
        result[label + '_comparable_n'] = len(deltas)
        result[label + '_excluded_n'] = len(values) - len(deltas)
        for direction in ('worse', 'improved', 'equal'):
            selected = [row[label + '_delta'] for row in values if row[label + '_' + direction] is True]
            result[label + '_' + direction + '_n'] = len(selected)
            result[label + '_' + direction + '_delta'] = moments(selected)
        result[label + '_delta'] = moments(deltas)
        for prefix in ('selected', 'comparator'):
            key = label + '_' + prefix + '_exclusion_reasons'
            result[key] = reason_counts(values, key)
        result[label + '_delta_exclusion_reasons'] = reason_counts(values, label + '_delta_exclusion_reason')
    return result


def analyze(scored, stages, *, arms=H1_ARMS, populations=POPULATIONS, expected_n=EXPECTED_N):
    """Pure constructed-data entry point; no files, reference math or fitting."""
    indexes = []
    for name, records in (('scored', scored), ('stage', stages)):
        index = {}
        for row in records:
            if not isinstance(row.get('id'), str) or row['id'] in index:
                raise ValueError('Invalid/duplicate ' + name + ' frame identity')
            index[row['id']] = row
        indexes.append(index)
    scored_by_id, stage_by_id = indexes
    if set(scored_by_id) != set(stage_by_id):
        raise ValueError('Scored and stage frame identity sets differ')
    actual_n = {name: sum(row['partition'] in parts for row in scored)
        for name, parts in populations.items()}
    if actual_n != expected_n:
        raise ValueError('Population denominators changed: ' + repr(actual_n))
    registered_partitions = set(part for parts in populations.values() for part in parts)
    details, subset_ids = [], {arm: [] for arm in arms}
    for row in scored:
        stage_row = stage_by_id[row['id']]
        if row['partition'] not in registered_partitions or stage_row['partition'] != row['partition']:
            raise ValueError('Stage/scored partition identity changed')
        for arm in arms:
            stage = stage_row['arms'][arm]
            eligible = stage['after_constraints']['has_success']
            if not isinstance(eligible, bool):
                raise ValueError('Missing exact eligible has_success boolean')
            if eligible:
                continue
            if stage.get('final_success') is not False or row.get('metrics', {}).get(arm, {}).get('success') is not False:
                raise ValueError('All-action failure contradicts final selected success')
            subset_ids[arm].append(row['id'])
            details.extend(comparison_row(row, arm, comparator, stage) for comparator in REFERENCES)
    buckets = {(arm, ref): [] for arm in arms for ref in REFERENCES}
    for row in details:
        buckets[(row['arm'], row['comparator'])].append(row)
    summaries = [summarize([row for row in buckets[(arm, ref)] if row['partition'] in parts],
        pop, arm, ref, actual_n[pop]) for pop, parts in populations.items()
        for arm in arms for ref in REFERENCES]
    return details, dict(schema='H1_ALL_ACTION_FAILURE_CONTINUOUS_ANALYSIS_V1',
        complete=True, arms=list(arms), comparators=list(REFERENCES),
        populations={name: list(parts) for name, parts in populations.items()},
        population_denominators=actual_n, full_frame_n=len(scored),
        all_failure_ids_by_arm=subset_ids, detail_rows=len(details), summary_rows=len(summaries),
        rows=summaries,
        definitions=dict(all_failure='This arm after_constraints.has_success is exactly false; pair/full eligibility are separate.',
            delta='saved selected error minus saved comparator error; original float subtraction, strict >0/<0/==0; no new tolerance',
            comparable='actual pose present and available, saved metric pose_available, finite numeric error on both sides',
            either='T and R both comparable; logical OR of strictly positive/negative deltas; tradeoffs may be both worseEither and improvedEither',
            moments='Signed deltas; empty comparison or sign subset has n=0 and sum/mean/min/max=null, never a manufactured zero.',
            nonfinite='Raw NaN/+Infinity/-Infinity preserved as text in CSV; exclusion reasons and null aggregates explicitly counted.',
            population='CSV contains one row per qualifying frame/arm/comparator, without duplicating PRIMARY/subpopulation views.',
            inference='Descriptive post-seal diagnosis only; no new policy, seed choice, calibrated safety guarantee or validation score.'),
        no_new_forward_PnP_metrics_optimization_or_selection=True, reference_payload_opened=False)


def verify_inputs(run_dir, evaluation):
    run_dir, evaluation = Path(run_dir), Path(evaluation)
    paths = {'evaluation_seal': evaluation/'EVALUATION_SEAL.json',
        'prediction_seal': run_dir/'PREDICTION_SEAL.json',
        'all_outputs_GO': run_dir/'ALL_OUTPUTS_SCORING_GO.json',
        'plan': run_dir/'PLAN_LOCK.json'}
    bindings = {name: dict(file=path.name, sha256=sha(path)) for name, path in paths.items()}
    ev, pred, gate = (read(paths[key]) for key in ('evaluation_seal', 'prediction_seal', 'all_outputs_GO'))
    pred_sha, plan_sha = bindings['prediction_seal']['sha256'], bindings['plan']['sha256']
    if ev.get('schema') != 'BOUNDARY_ACTION_EVALUATION_SEAL_V1' or ev.get('complete') is not True or ev.get('all_failures_retained') is not True:
        raise ValueError('Incomplete original evaluation seal')
    if pred.get('complete') is not True or pred.get('before_scoring') is not True or pred.get('reference_inputs') is not False:
        raise ValueError('Predictions were not reference-free sealed before scoring')
    if set(pred.get('arms', [])) != set(ALL_ARMS) or len(pred['arms']) != len(ALL_ARMS):
        raise ValueError('Predeclared37 arm inventory changed')
    if ev.get('prediction_seal_sha256') != pred_sha or gate.get('prediction_seal_sha256') != pred_sha:
        raise ValueError('Evaluation/all-output GO prediction binding changed')
    if gate.get('all_registered_predictions_sealed') is not True or gate.get('plan_sha256') != plan_sha or pred.get('plan_lock_sha256') != plan_sha:
        raise ValueError('All-output GO/plan binding changed')
    if ev.get('input_predictions_sha256') != pred.get('predictions_sha256'):
        raise ValueError('Evaluation prediction journal hash differs from saved seal')
    if len(pred.get('ordered_ids', [])) != 238 or len(set(pred['ordered_ids'])) != 238 or pred.get('rows') != 238:
        raise ValueError('Sealed original238 identity inventory changed')
    if ev.get('registry_sha256') != sha(Path(__file__).with_name('registry.py')):
        raise ValueError('Evaluation arm registry source changed')
    for name in JOURNALS:
        path = evaluation/name
        expected = ev.get('artifacts', {}).get(name)
        if not isinstance(expected, str) or sha(path) != expected:
            raise ValueError('Evaluated artifact changed: ' + name)
        paths[name], bindings[name] = path, dict(file=name, sha256=expected)
    return paths, bindings, pred


def run(run_dir, evaluation=None, output=None, *, analysis_GO=False):
    if not analysis_GO:
        raise PermissionError('Root post-seal analysis GO required; no result was read')
    started, cpu = time.monotonic(), time.process_time()
    run_dir = Path(run_dir)
    evaluation = Path(evaluation) if evaluation is not None else run_dir/'evaluation_v1'
    output = Path(output) if output is not None else run_dir/'h1_all_failure_v1'
    if output.exists():
        raise FileExistsError('Preserve existing output: ' + str(output))
    for protected in (evaluation.resolve(), (run_dir/'posthoc_v1').resolve()):
        if output.resolve() == protected or protected in output.resolve().parents:
            raise ValueError('A separate H1 analysis output directory is required')
    paths, bindings, pred = verify_inputs(run_dir, evaluation)
    scored, stages = (read_rows(paths[name]) for name in JOURNALS)
    if [row['id'] for row in scored] != pred['ordered_ids'] or [row['id'] for row in stages] != pred['ordered_ids']:
        raise ValueError('Evaluated journals differ from sealed238 ordered identities')
    details, summary = analyze(scored, stages)
    for name, path in paths.items():
        if sha(path) != bindings[name]['sha256']:
            raise ValueError('Input changed during analysis: ' + name)
    summary['input_bindings'] = bindings
    summary['source_sha256'] = sha(__file__)
    summary['registry_sha256'] = sha(Path(__file__).with_name('registry.py'))
    output.mkdir(parents=True, exist_ok=False)
    fields = list(comparison_row(dict(id='', partition='', poses={}, metrics={}), '', '',
        dict(after_constraints={})).keys())
    with (output/'H1_ALL_FAILURE_ROWS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(details)
    write(output/'H1_ALL_FAILURE_SUMMARY.json', summary)
    seal = dict(schema='H1_ALL_ACTION_FAILURE_ANALYSIS_SEAL_V1', complete=True,
        input_bindings=bindings, source_sha256=sha(__file__),
        registry_sha256=summary['registry_sha256'], frame_n=summary['full_frame_n'],
        H1_arm_n=len(H1_ARMS), comparator_n=len(REFERENCES),
        detail_rows=len(details), summary_rows=summary['summary_rows'],
        artifacts={path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()},
        cost=dict(wall_seconds=time.monotonic()-started, CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        reference_payload_opened=False, new_pose_metric_evaluations=0,
        model_forwards=0, optimizations=0, selections_changed=0,
        original_posthoc_tree_modified=False)
    write(output/'ANALYSIS_SEAL.json', seal)
    return seal


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--evaluation', type=Path, help='Default RUN_DIR/evaluation_v1')
    p.add_argument('--output', type=Path, help='New separate directory; default RUN_DIR/h1_all_failure_v1')
    p.add_argument('--analysis-GO', action='store_true', help='Root GO to read the sealed evaluated journals')
    return p


def main():
    args = parser().parse_args()
    print(json.dumps(run(args.run_dir, args.evaluation, args.output,
        analysis_GO=args.analysis_GO), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
