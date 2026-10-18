"""Independent CSV-to-saved-measurement reporting audit; no metrics/models/GT import."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = (20261018, 20261019, 20261020)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, figures, original_manifest, destination, *, postseal_GO=False):
    if not postseal_GO:raise PermissionError('Root postseal reporting GO required before scalar-result access')
    root, figures, original_manifest, destination = map(Path, (root, figures, original_manifest, destination))
    prediction=read(root/'PREDICTION_MANIFEST.json');seal=read(root/'evaluation_v1/EVALUATION_SEAL.json')
    if prediction['status']!='ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING' or seal['status']!='COMPLETE' or seal['prediction_manifest_sha256']!=sha(root/'PREDICTION_MANIFEST.json'):
        raise ValueError('Actual prediction/evaluation seal chain required')
    for name in ('EVALUATION.json','SCORED_ROWS.jsonl.gz'):
        if sha(root/'evaluation_v1'/name)!=seal['artifacts'][name]:raise ValueError('Archived evaluation artifact changed')
    result = read(root / 'evaluation_v1/EVALUATION.json')
    with gzip.open(root / 'evaluation_v1/SCORED_ROWS.jsonl.gz', 'rt') as stream:
        scored = [json.loads(line) for line in stream if line.strip()]
    primary = [r for r in scored if r['partition'] != 'CAL']
    by_id = {r['id']: r for r in primary};arms = result['arms'];checks = 0;max_mean_abs_difference = 0.

    def check(value, why):
        nonlocal checks
        checks += 1
        if not value:raise AssertionError(why)

    def equal(actual, expected, why, *, mean=False):
        nonlocal max_mean_abs_difference
        if expected is None:check(actual == '', why);return
        if isinstance(expected, bool):check(actual == str(expected), why);return
        if isinstance(expected, str):check(actual == expected, why);return
        value = float(actual)
        if mean:
            max_mean_abs_difference = max(max_mean_abs_difference, abs(value - expected))
            check(math.isclose(value, expected, abs_tol=1e-12, rel_tol=1e-10), why)
        else:check(value == expected, why)

    def table(name):
        with (figures / name).open(newline='') as stream:return list(csv.DictReader(stream))

    def unique(records, keys, expected):
        tuples = [tuple(r[k] for k in keys) for r in records]
        check(len(tuples) == len(set(tuples)), 'duplicate CSV identity ' + str(keys))
        check(set(tuples) == set(expected), 'CSV identity population ' + str(keys))

    def group(arm):
        a = arm.rsplit('_s', 1)
        return a[0] if len(a) == 2 and a[1].isdigit() else arm

    def before(arm):
        a = arm.rsplit('_s', 1)
        return 'D1_H1_MLP_s' + (a[1] if len(a) == 2 and a[1].isdigit() else '20261018')

    check(len(primary) == len(by_id) == 192, 'whole primary identities')
    counts = {}
    data = table('success_and_large_errors.csv');counts['success'] = len(data)
    unique(data, ('arm',), [(a,) for a in arms])
    for r in data:
        target = result['populations']['PRIMARY192'][r['arm']]
        for key in ('successes', 'T_gt50cm', 'T_gt100cm', 'R_gt45deg', 'no_pose'):equal(r[key], target[key], 'success field ' + key)
        equal(r['denominator'], 192, 'success full denominator')

    data = table('whole_error_distributions.csv');counts['ECDF'] = len(data)
    unique(data, ('id', 'arm', 'axis'), [(fid, a, axis) for fid in by_id for a in arms for axis in ('translation_cm', 'rotation_deg')])
    rank = {}
    for arm in arms:
        for axis in ('translation_cm', 'rotation_deg'):
            valid = sorted((r['metrics'][arm][axis], r['id']) for r in primary if r['metrics'][arm][axis] is not None)
            rank.update({(fid, arm, axis): i + 1 for i, (_, fid) in enumerate(valid)})
    for r in data:
        value = by_id[r['id']]['metrics'][r['arm']][r['axis']]
        key = (r['id'], r['arm'], r['axis'])
        equal(r['error'], value, 'every saved ECDF value')
        equal(r['available'], value is not None, 'ECDF availability')
        equal(r['denominator'], 192, 'ECDF whole denominator')
        equal(r['order'], rank.get(key), 'ECDF rank')
        equal(r['cumulative_fraction_of_full_denominator'], rank[key] / 192 if key in rank else None, 'ECDF fraction')

    data = table('H1_recovery_damage_and_common_failure.csv');counts['H1_transitions'] = len(data)
    expected = [a for a in arms if group(a) != 'D1_H1_MLP']
    unique(data, ('arm',), [(a,) for a in expected])
    for r in data:
        arm = r['arm'];base = before(arm);equal(r['before'], base, 'same-seed H1 comparator')
        c = Counter();both = []
        for record in primary:
            b, a = record['metrics'][base]['success'], record['metrics'][arm]['success']
            c['both_success_n' if b and a else 'recovery' if a else 'damage' if b else 'both_failure_n'] += 1
            if not b and not a:both.append(record)
        for key in ('both_success_n', 'recovery', 'damage', 'both_failure_n'):equal(r[key], c[key], 'H1 transition ' + key)
        equal(r['denominator'], 192, 'transition whole denominator')
        for axis, prefix, suffix in (('translation_cm', 'T', 'cm'), ('rotation_deg', 'R', 'deg')):
            d = [t['metrics'][arm][axis] - t['metrics'][base][axis] for t in both if all(t['metrics'][a][axis] is not None for a in (arm, base))]
            equal(r['delta_' + prefix + '_n'], len(d), 'comparable both failure')
            equal(r['delta_' + prefix + '_mean_' + suffix], statistics.mean(d) if d else None, 'independent stable mean', mean=True)
            equal(r['delta_' + prefix + '_worsened_n'], sum(v > 0 for v in d), 'raw strict positive delta')

    data = table('fixed_bank_all_failure_continuous_scatter.csv');counts['all_failure_scatter'] = len(data)
    failed = {fid: r for fid, r in by_id.items() if not r['full_bank_has_success']}
    selected = [a for a in arms if group(a) in ('R_CONT_F0', 'H1_NO_B_FEATURE', 'H1_GUARDED_SWITCH', 'H1_GUARD_CALIBRATED_TRIAL', 'AUG_RANDOM_H1', 'AUG_STRUCTURED_H1')]
    unique(data, ('id', 'arm'), [(fid, a) for fid in failed for a in selected])
    for r in data:
        record = failed[r['id']];arm = r['arm'];base = before(arm)
        equal(r['denominator'], len(failed), 'dynamic actual allfailure denominator');equal(r['before'], base, 'allfailure comparator')
        comparable = all(record['metrics'][a][axis] is not None for a in (arm, base) for axis in ('translation_cm', 'rotation_deg'))
        equal(r['comparable'], comparable, 'allfailure comparable')
        for axis in ('translation_cm', 'rotation_deg'):
            equal(r['delta_' + axis], record['metrics'][arm][axis] - record['metrics'][base][axis] if comparable else None, 'every scatter raw delta')

    data = table('actual_sources_and_H1_recovery_sources.csv');counts['action_sources'] = len(data)
    unique(data, ('arm', 'scope', 'source'), [(a, scope, source) for a in arms for scope in ('ALL', 'RECOVERY_VS_H1') for source in ('N0', 'E0', 'OTHER', 'NO_POSE')])
    for r in data:
        arm = r['arm'];base = before(arm);chosen = [v for v in primary if r['scope'] == 'ALL' or (not v['metrics'][base]['success'] and v['metrics'][arm]['success'])]
        equal(r['n'], sum(v['action_sources'][arm] == r['source'] for v in chosen), 'actual physical source count')
        equal(r['scope_n'], len(chosen), 'source conditional denominator');equal(r['whole_denominator'], 192, 'source whole denominator')

    expected_history = {}
    for path, wanted in ((original_manifest, 'D1_H1_MLP'), (root / 'NO_B_TRAINING_MANIFEST.json', 'H1_NO_B_FEATURE')):
        for receipt in read(path)['runs']:
            if receipt['arm'] != wanted:continue
            history_path = Path(receipt.get('history', Path(receipt['checkpoint']).parent / 'HISTORY.json'))
            check(sha(history_path) == receipt['history_sha256'], 'original sealed history bytes')
            for h in read(history_path):expected_history[(wanted, str(receipt['seed']), str(h['epoch']))] = (receipt, h)
    data = table('learning_and_pose_curves.csv');counts['learning'] = len(data)
    unique(data, ('arm', 'seed', 'epoch'), expected_history)
    for r in data:
        receipt, h = expected_history[(r['arm'], r['seed'], r['epoch'])];v = h['validation']
        mapping = dict(best_epoch=receipt['best_epoch'], train_frames=h['train']['frames'], validation_frames=v['frames'], train_loss=h['train']['loss'], validation_loss=v['loss'], validation_success=v['success_5cm5deg'], validation_T_mean_cm=v['translation_cm']['mean'], validation_R_mean_deg=v['rotation_deg']['mean'], validation_Brier=v['frame_balanced_Brier'])
        mapping.update({'validation_' + k: val for k, val in v['tails'].items()})
        for k, val in mapping.items():equal(r[k], val, 'every learning value ' + k)

    policy = read(root / 'POLICY_CALIBRATION.json');expected_tau = {}
    for seed, v in policy['seeds'].items():
        def tau_key(t):return (t['kind'], t.get('value'))
        def metrics(m):return dict(denominator=m['denominator'], successes=m['successes'], T_mean_cm=m['translation_cm']['mean'], R_mean_deg=m['rotation_deg']['mean'], T_gt50cm=m['T_gt50cm'], T_gt100cm=m['T_gt100cm'], R_gt45deg=m['R_gt45deg'], changes=m['changes'])
        for i, t in enumerate(v['all_thresholds']):
            expected_tau[(seed, 'SYNTHETIC_DEVELOPMENT', str(i))] = dict(metrics(t['metrics']), tau_kind=t['tau']['kind'], tau_value=t['tau'].get('value'), is_active=tau_key(t['tau']) == tau_key(v['active_tau']), is_calibrated_trial=tau_key(t['tau']) == tau_key(v['calibrated_trial_tau']), admissible=t['admissible'], policy_status=v['status'], confirmation_passed=None)
        if v['confirmation'] is not None:
            ranks = [tau_key(t['tau']) for t in v['all_thresholds']]
            for mode in ('baseline', 'selected'):
                rank = 0 if mode == 'baseline' else ranks.index(tau_key(v['calibrated_trial_tau']))
                tau = {'kind': 'NEGATIVE_INFINITY'} if mode == 'baseline' else v['calibrated_trial_tau']
                expected_tau[(seed, 'SYNTHETIC_CONFIRMATION_' + mode.upper(), str(rank))] = dict(metrics(v['confirmation'][mode]), tau_kind=tau['kind'], tau_value=tau.get('value'), is_active=None, is_calibrated_trial=mode == 'selected', admissible=None, policy_status=v['status'], confirmation_passed=v['confirmation']['passed'])
    data = table('synthetic_tau_success_risk.csv');counts['synthetic_tau'] = len(data)
    unique(data, ('seed', 'scope', 'threshold_rank'), expected_tau)
    for r in data:
        for k, val in expected_tau[(r['seed'], r['scope'], r['threshold_rank'])].items():equal(r[k], val, 'every synthetic policy row ' + k)

    manifest = read(figures / 'FIGURES.json')
    for r in manifest['images'] + manifest['machine_readable_data']:
        check(sha(figures / r['file']) == r['sha256'], 'actual figure artifact bytes')
    receipt = dict(status='PASS', schema='INDEPENDENT_FIGURE_TABLE_REVIEW_V1', checks=checks, failures=0, CSV_rows=counts,
        population=192, arms=len(arms), full_bank_all_failure_n=len(failed), comparison_scope='Saved per-image scalar metrics and synthetic learning/policy records; no real poses, GT, models or production plotting/metric helpers imported',
        stable_mean_max_abs_difference=max_mean_abs_difference, strict_delta_signs_used_without_tolerance=True,
        source_sha256=sha(__file__), figure_manifest_sha256=sha(figures / 'FIGURES.json'), evaluation_sha256=sha(root / 'evaluation_v1/EVALUATION_SEAL.json'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:json.dump(receipt, stream, indent=2);stream.write('\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run-dir', 'figures', 'original-manifest', 'output'):p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--postseal-GO',action='store_true')
    a = p.parse_args();print(json.dumps(audit(a.run_dir, a.figures, a.original_manifest, a.output,postseal_GO=a.postseal_GO)))
