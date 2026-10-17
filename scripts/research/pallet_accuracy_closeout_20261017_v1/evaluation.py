"""Common saved-pose evaluator, run only after both prediction ledgers are sealed.

No selector, image, neural framework, PnP, or optimization is imported. The two
PoseFix policies remain independently named; neither is deployed as a winner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .baseline_audit import (ARMS, PARTITIONS, distribution, load_bound_original,
                             metric, read_json, read_journal, sha, write_json)

PF_ARMS = ('POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
BOOTSTRAP_SEED = 20260917  # existing descriptive-statistics seed, never a model seed
BOOTSTRAP_DRAWS = 10000
POPULATIONS = {'PRIMARY192': ('TEST', 'REMAINDER'), 'CAL46': ('CAL',),
               'TEST49': ('TEST',), 'REMAINDER143': ('REMAINDER',),
               'COMMON95': ('CAL', 'TEST')}


def protocol_value():
    return dict(schema='ACCURACY_CLOSEOUT_EVALUATOR_PROTOCOL_V1',
                populations=POPULATIONS, original_arms=ARMS, fixed_pf_arms=PF_ARMS,
                metrics=dict(success='one saved pose: translation_cm<=5 AND physical_C2_rotation_deg<=5',
                             symmetry='I and Y(pi) only; rectangular Y(pi/2) excluded',
                             continuous='available-only; missing stays binary failure; no imputation',
                             std='sample ddof=1', quantile='NumPy linear', tails='T>50cm,T>100cm,R>45deg'),
                bootstrap=dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS,
                               primary_unit='original cluster_id (whole recording)',
                               sensitivity_unit='named session',
                               estimand='paired frame-weighted mean difference, whole clusters sampled with replacement',
                               CI='descriptive percentile .025/.975, linear interpolation',
                               limitation='few clusters, repeated real development data; no independent validation or equivalence claim'),
                comparator_policy='TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER',
                selector_is_separate=True, source_sha256=sha(__file__),
                metric_source_sha256=sha(Path(__file__).with_name('baseline_audit.py')))


def write_protocol(path):
    write_json(path, protocol_value())


def _prediction_entry(value):
    if 'actual_pose' in value:
        return value
    if 'pose' in value:
        return {**value, 'actual_pose': value['pose']}
    raise ValueError('Explicit actual_pose or pose required for every prediction')


def decode_predictions(path):
    rows = read_journal(path)
    merged = {}
    for row in rows:
        identity = {k: row[k] for k in ('id', 'partition', 'session')}
        if 'cluster_id' in row:
            identity['cluster_id'] = row['cluster_id']
        fid = row['id']
        preds = row.get('predictions')
        if preds is None:
            preds = {row.get('method', row.get('arm')): row}
        if fid not in merged:
            merged[fid] = {**identity, 'predictions': {}}
        if any(merged[fid][k] != identity[k] for k in identity):
            raise ValueError('Conflicting identity in prediction journal')
        for arm, value in preds.items():
            if not isinstance(arm, str) or not arm or arm in merged[fid]['predictions']:
                raise ValueError('Missing or duplicated prediction arm')
            merged[fid]['predictions'][arm] = _prediction_entry(value)
    return list(merged.values())


def verify_prediction_seal(path, seal_path, expected_ids, expected_sha=None):
    """Supports explicit original-style or new closeout seal names, never guesses bytes."""
    if expected_sha is not None and sha(seal_path) != expected_sha:
        raise ValueError('Requested immutable prediction seal SHA changed')
    seal = read_json(seal_path)
    actual = sha(path)
    digests = [seal.get(k) for k in ('predictions_sha256', 'prediction_sha256', 'predictions_file_sha256')]
    digests += list((seal.get('artifacts') or {}).values())
    if actual not in digests:
        raise ValueError('Prediction bytes are not bound by seal')
    for field in ('GT_inputs', 'reference_inputs', 'reference_reads', 'GT_reads'):
        if field in seal and seal[field] not in (False, 0, []):
            raise ValueError('Selector/predictor used reference inputs: ' + field)
    if not any(seal.get(k) is True for k in ('before_scoring', 'before_new_scoring', 'before_reference_scoring')):
        raise ValueError('Predictions must be sealed before reference scoring')
    sealed_ids = seal.get('ordered_ids')
    if (not isinstance(sealed_ids, list) or len(sealed_ids) != len(expected_ids)
            or len(set(sealed_ids)) != len(expected_ids) or set(sealed_ids) != set(expected_ids)):
        raise ValueError('Prediction seal must bind all original238 identities exactly once')
    rows = decode_predictions(path)
    if [r['id'] for r in rows] != sealed_ids:
        raise ValueError('Prediction rows must preserve their own sealed identity order')
    by_id = {r['id']: r for r in rows}
    return [by_id[fid] for fid in expected_ids], seal


def normalize_status(entry):
    value = dict(entry.get('prediction_status') or {})
    for field in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout', 'status', 'reason'):
        value.setdefault(field, entry.get(field))
    for field in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout'):
        if value[field] is not None and type(value[field]) is not bool:
            raise ValueError('Boolean or explicit unknown required: ' + field)
    return value


def _counter_status(rows, arm):
    result = {}
    for key in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout'):
        values = [row['status'][arm].get(key) for row in rows]
        result[key] = dict(true_n=sum(v is True for v in values), false_n=sum(v is False for v in values),
                           unknown_n=sum(v is None for v in values))
    result['raw_status'] = dict(Counter(str(r['status'][arm].get('status')) for r in rows))
    return result


def aggregate(rows, arms):
    result = {}
    for arm in arms:
        m = [r['metrics'][arm] for r in rows]
        result[arm] = dict(denominator=len(rows), successes=sum(v['success'] for v in m),
                           success_rate=sum(v['success'] for v in m) / len(rows),
                           no_pose=sum(not v['pose_available'] for v in m),
                           translation_cm=distribution([v['translation_cm'] for v in m], len(m)),
                           rotation_deg=distribution([v['rotation_deg'] for v in m], len(m)),
                           T_gt50cm=sum(v['translation_cm'] is not None and v['translation_cm'] > 50. for v in m),
                           T_gt100cm=sum(v['translation_cm'] is not None and v['translation_cm'] > 100. for v in m),
                           R_gt45deg=sum(v['rotation_deg'] is not None and v['rotation_deg'] > 45. for v in m),
                           status_counts=_counter_status(rows, arm),
                           failures=[r['id'] for r in rows if not r['metrics'][arm]['success']])
    return result


def _cluster_ci(values, rows, unit):
    # Values may be None for unavailable continuous pairs; binary never missing.
    labels = sorted({r[unit] for r in rows})
    if len(labels) < 2:
        return dict(CI95=None, reason='fewer than two original clusters', cluster_count=len(labels))
    sums = np.array([sum(float(v) for r, v in zip(rows, values) if r[unit] == label and v is not None)
                     for label in labels])
    counts = np.array([sum(v is not None for r, v in zip(rows, values) if r[unit] == label)
                       for label in labels])
    # Local RNG means every declared contrast uses the same paired draw schedule.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(labels), np.repeat(1. / len(labels), len(labels)), BOOTSTRAP_DRAWS)
    n = weights @ counts
    numer = weights @ sums
    draws = numer[n > 0] / n[n > 0]
    return dict(CI95=[float(v) for v in np.quantile(draws, [.025, .975])] if len(draws) else None,
                cluster_count=len(labels), clusters=labels, cluster_sizes=dict(Counter(r[unit] for r in rows)),
                valid_resamples=len(draws), empty_resamples=int(np.sum(n == 0)),
                seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_DRAWS, unit=unit,
                method='whole-cluster percentile bootstrap, paired frame-weighted ratio; linear quantile')


def paired(rows, before, after):
    groups = dict(BOTH_SUCCESS=[], RECOVERY=[], DAMAGE=[], BOTH_FAILURE=[])
    success_delta, dt, da = [], [], []
    for row in rows:
        a, b = row['metrics'][before], row['metrics'][after]
        key = ('BOTH_SUCCESS' if a['success'] and b['success'] else 'RECOVERY' if b['success']
               else 'DAMAGE' if a['success'] else 'BOTH_FAILURE')
        groups[key].append(row['id'])
        success_delta.append(int(b['success']) - int(a['success']))
        dt.append(None if a['translation_cm'] is None or b['translation_cm'] is None
                  else b['translation_cm'] - a['translation_cm'])
        da.append(None if a['rotation_deg'] is None or b['rotation_deg'] is None
                  else b['rotation_deg'] - a['rotation_deg'])
    deltas = dict(success_rate=success_delta, translation_cm=dt, rotation_deg=da)
    return dict(before=before, after=after, denominator=len(rows), counts={k: len(v) for k, v in groups.items()},
                ids=groups, success_difference_count=sum(success_delta),
                success_difference_percentage_points=100. * sum(success_delta) / len(rows),
                deltas={key: dict(mean=float(np.mean([v for v in values if v is not None]))
                                 if any(v is not None for v in values) else None,
                                 common_available_n=sum(v is not None for v in values),
                                 improved=sum(v < 0 for v in values if v is not None) if key != 'success_rate'
                                 else sum(v > 0 for v in values),
                                 worsened=sum(v > 0 for v in values if v is not None) if key != 'success_rate'
                                 else sum(v < 0 for v in values),
                                 recording_cluster_CI=_cluster_ci(values, rows, 'cluster_id'),
                                 named_session_cluster_sensitivity_CI=_cluster_ci(values, rows, 'session'))
                        for key, values in deltas.items()},
                both_failure_tail={arm: aggregate([r for r in rows if r['id'] in groups['BOTH_FAILURE']], (arm,))[arm]
                                   for arm in (before, after)} if groups['BOTH_FAILURE'] else {})


def _csv(path, rows):
    if not rows:
        raise ValueError('Nonempty CSV expected')
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(*, baseline_folder, predictions, prediction_seal, protocol_lock, output,
        expected_prediction_seal_sha256=None, additional_predictions=None, additional_seal=None):
    start, cpu = time.perf_counter(), time.process_time()
    protocol_lock = Path(protocol_lock)
    if read_json(protocol_lock) != json.loads(json.dumps(protocol_value())):
        raise ValueError('Predeclared evaluator protocol changed')
    # Hash/decode prediction identities only. Numeric reference JSON is first
    # decoded below, after all new predictor-only seals have passed.
    old_seal = read_json(Path(baseline_folder) / 'SCORE_SEAL.json')
    original_bindings = {}
    for name in ('BASELINE_PREDICTIONS.jsonl.gz', 'FIXED_REFERENCES.json', 'ROW_METRICS.csv'):
        digest = sha(Path(baseline_folder) / name)
        if digest != old_seal['artifacts'][name]:
            raise ValueError('Original sealed artifact changed: ' + name)
        original_bindings[name] = digest
    original = read_journal(Path(baseline_folder) / 'BASELINE_PREDICTIONS.jsonl.gz')
    ids = [r['id'] for r in original]
    # Validate both predictor-only artifacts before decoding numeric reference.
    fresh, seal = verify_prediction_seal(predictions, prediction_seal, ids,
                                        expected_prediction_seal_sha256)
    inventory = set(fresh[0]['predictions'])
    if not set(PF_ARMS).issubset(inventory):
        raise ValueError('Both fixed PoseFix+SubPix comparators are required')
    extras = []
    if additional_predictions:
        additional, _ = verify_prediction_seal(additional_predictions, additional_seal, ids)
        extras = list(additional[0]['predictions'])
    else:
        additional = []
    arms = tuple(dict.fromkeys((*ARMS, *PF_ARMS, *extras)))
    original, references, _ = load_bound_original(baseline_folder)
    rows = []
    for i, (old, new) in enumerate(zip(original, fresh)):
        identity = {k: old[k] for k in ('id', 'partition', 'session', 'cluster_id')}
        if any(new[k] != identity[k] for k in identity if k in new):
            raise ValueError('New/original identity metadata differ')
        entries = {a: old['predictions'][a] for a in ARMS}
        entries.update({a: new['predictions'][a] for a in PF_ARMS})
        if additional:
            if any(additional[i][k] != identity[k] for k in identity if k in additional[i]):
                raise ValueError('Additional/original identity metadata differ')
            entries.update(additional[i]['predictions'])
        row = {**identity, 'metrics': {}, 'status': {}, 'poses': {}}
        for arm in arms:
            entry = entries[arm]
            row['metrics'][arm] = metric(entry['actual_pose'], references[old['partition']][old['id']])
            row['status'][arm] = normalize_status(entry)
            row['poses'][arm] = entry['actual_pose']
        rows.append(row)
    subsets = {name: [r for r in rows if r['partition'] in parts] for name, parts in POPULATIONS.items()}
    expected_n = {'PRIMARY192': 192, 'CAL46': 46, 'TEST49': 49, 'REMAINDER143': 143, 'COMMON95': 95}
    if {key: len(value) for key, value in subsets.items()} != expected_n:
        raise ValueError('Every population denominator must remain exact')
    summaries = {key: aggregate(value, arms) for key, value in subsets.items()}
    contrasts = [('N0', 'E0'), ('POSEFIX_RAW', 'E0'), *[(a, 'E0') for a in PF_ARMS]]
    for arm in extras:
        contrasts += [(a, arm) for a in ('E0', 'N0', *PF_ARMS)]
    contrasts = list(dict.fromkeys(contrasts))
    paired_all = {key: {after + '_minus_' + before: paired(value, before, after) for before, after in contrasts}
                  for key, value in subsets.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    row_values = []
    for row in rows:
        for arm in arms:
            row_values.append({**{k: row[k] for k in ('id', 'partition', 'session', 'cluster_id')}, 'arm': arm,
                               **row['metrics'][arm], **row['status'][arm]})
    allkeys = list(dict.fromkeys(k for row in row_values for k in row))
    row_values = [{k: row.get(k) for k in allkeys} for row in row_values]
    _csv(output / 'FAIR_ROW_METRICS.csv', row_values)
    comparison = []
    for population, methods in summaries.items():
        for arm, summary in methods.items():
            row = dict(population=population, arm=arm, denominator=summary['denominator'],
                       successes=summary['successes'], success_rate=summary['success_rate'],
                       no_pose=summary['no_pose'], T_gt50cm=summary['T_gt50cm'],
                       T_gt100cm=summary['T_gt100cm'], R_gt45deg=summary['R_gt45deg'])
            for field in ('translation_cm', 'rotation_deg'):
                row.update({field + '_' + k: summary[field][k] for k in
                            ('n', 'mean', 'sample_std', 'sample_variance', 'median', 'P90', 'P95', 'max')})
            for field in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout'):
                row.update({field + '_' + k: v for k, v in summary['status_counts'][field].items()})
            comparison.append(row)
    _csv(output / 'FAIR_192_COMPARISON.csv', comparison)
    transitions = []
    for row in rows:
        for before, after in contrasts:
            a, b = row['metrics'][before], row['metrics'][after]
            group = ('BOTH_SUCCESS' if a['success'] and b['success'] else 'RECOVERY' if b['success']
                     else 'DAMAGE' if a['success'] else 'BOTH_FAILURE')
            transitions.append({**{k: row[k] for k in ('id', 'partition', 'session', 'cluster_id')},
                                'before': before, 'after': after, 'transition': group,
                                'before_success': a['success'], 'after_success': b['success'],
                                'before_T_cm': a['translation_cm'], 'after_T_cm': b['translation_cm'],
                                'before_R_deg': a['rotation_deg'], 'after_R_deg': b['rotation_deg']})
    _csv(output / 'PAIRED_TRANSITIONS.csv', transitions)
    session_summaries = {name: {session: aggregate([r for r in value if r['session'] == session], arms)
                                for session in sorted({r['session'] for r in value})}
                         for name, value in subsets.items()}
    result = dict(schema='ACCURACY_CLOSEOUT_FAIR_FIXED_POSE_EVALUATION_V1', arms=arms,
                  populations=summaries, paired=paired_all, session_summaries=session_summaries,
                  comparator_policy='TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER',
                  input_bindings=dict(original=original_bindings, predictions=sha(predictions),
                                      prediction_seal=sha(prediction_seal), evaluator_protocol=sha(protocol_lock),
                                      additional_predictions=sha(additional_predictions) if additional_predictions else None,
                                      additional_seal=sha(additional_seal) if additional_seal else None),
                  identity_join=dict(new_source_order_preserved_in_prediction_seal=True,
                                     numeric_pose_values_unchanged=True,
                                     output_order='original BASELINE_PREDICTIONS order',
                                     source_order_differs=seal['ordered_ids'] != ids,
                                     operation='exact unique ID join, never nearest pose or success matching'),
                  evaluation_reference='existing geometric reconstruction, physical/C2; not independent physical ground truth',
                  all_rows_kept=True, no_reference_reads_inside_predictor_asserted_in_seal=True,
                  selector_imports=0, new_model_forwards=0, new_pose_solves=0,
                  elapsed_seconds=time.perf_counter() - start, process_cpu_seconds=time.process_time() - cpu,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()))
    write_json(output / 'FAIR_RESULTS.json', result)
    write_json(output / 'FAIR_SCORED_ROWS.json', {'rows': rows})
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'EVALUATION_SEAL.json', dict(schema='ACCURACY_CLOSEOUT_EVALUATION_SEAL_V1',
                                                   **{k: result[k] for k in ('input_bindings', 'elapsed_seconds', 'max_rss_kib')},
                                                   rows=238, arm_rows=238 * len(arms), artifacts=artifacts))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write-protocol')
    p.add_argument('--baseline-folder')
    p.add_argument('--predictions')
    p.add_argument('--prediction-seal')
    p.add_argument('--expected-prediction-seal-sha256')
    p.add_argument('--protocol-lock')
    p.add_argument('--additional-predictions')
    p.add_argument('--additional-seal')
    p.add_argument('--output')
    a = p.parse_args()
    if a.write_protocol:
        write_protocol(a.write_protocol)
        print(sha(a.write_protocol))
        return
    result = run(baseline_folder=a.baseline_folder, predictions=a.predictions, prediction_seal=a.prediction_seal,
                 expected_prediction_seal_sha256=a.expected_prediction_seal_sha256,
                 protocol_lock=a.protocol_lock, output=a.output,
                 additional_predictions=a.additional_predictions, additional_seal=a.additional_seal)
    print(json.dumps({arm: m['successes'] for arm, m in result['populations']['PRIMARY192'].items()}))


if __name__ == '__main__':
    main()
