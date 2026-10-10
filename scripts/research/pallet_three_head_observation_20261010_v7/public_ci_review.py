"""Public, standard-library-only replay of fixed V7 moments and paired CIs.

This supplemental inspector reads already scored rows. It never imports the
production pipeline/statistics/verify modules or opens private weights/GT.
Freeze its own code and public input bytes before one arithmetic pass. Saved
pose metrics and the existing geometric proxy remain external truth limits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
FILES = ('PUBLIC_CI_PROTOCOL.json', 'PUBLIC_CI_STARTED.json', 'PUBLIC_CI_CHECKS.json')
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')
ABS_TOL = 1e-9
REL_TOL = 1e-12


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def reject_symlink(path):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink: ' + str(path))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path)
    reject_symlink(path)
    relative = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest(path))


def bound(path, expected, label):
    actual = binding(path)
    require(actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256'],
            'byte binding differs: ' + label)


def public_path(item):
    relative = Path(item['path'])
    require(item.get('origin') == 'public_repository' and not relative.is_absolute() and
            '..' not in relative.parts, 'nonpublic dependency')
    path = REPO / relative
    reject_symlink(path)
    return path


def write_new(path, value):
    reject_symlink(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()


def guard(args, stage):
    folder, output = Path(args.input).absolute(), Path(args.output).absolute()
    reject_symlink(folder)
    reject_symlink(output)
    require(folder.is_dir(), 'input evidence directory missing')
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES if stage == 'freeze' else FILES[1:]:
        require(not (output / name).exists(), 'preserve existing ' + name)
    return folder, output


def paths(folder):
    protocol = read(folder / 'PROTOCOL.json')
    result = {name: folder / name for name in ('PROTOCOL.json', 'SCORING_RECEIPT.json',
              'PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz', 'METRICS.json')}
    result['cohort'] = public_path(protocol['inputs']['cohort'])
    result['bootstrap'] = public_path(protocol['inputs']['bootstrap_draws'])
    result['checker_code'] = Path(__file__).resolve()
    return result


def validate_metadata(folder, files):
    protocol, scoring, metrics = [read(folder / n) for n in
        ('PROTOCOL.json', 'SCORING_RECEIPT.json', 'METRICS.json')]
    require(protocol['frames'] == scoring['frames'] == 245 and protocol['primary'] ==
            metrics['primary_method'] == 'IMAGE_ROLE_BOUNDARY_ONLY', 'population/primary drift')
    methods = ['BASE', 'N3_SUBPIX', *protocol['methods']]
    require(len(methods) == len(set(methods)) == 10 and len(protocol['methods']) == 8,
            'ten distinct fixed methods required')
    contrasts = [tuple(pair) for pair in protocol['contrasts']]
    require(len(contrasts) == len(set(contrasts)) == 16 and
            all(len(pair) == 2 and all(m in methods for m in pair) for pair in contrasts),
            'sixteen fixed contrasts required')
    require(scoring['complete'] is True and scoring['scored_method_rows'] == 1960 and
            scoring['fixed_control_rows'] == 490 and scoring['methods'] == protocol['methods'],
            'complete scored population required')
    require(scoring['GT_access_only_after_complete_seal'] is True and
            scoring['old_v4_control_parity_before_reference_reads'] is True and
            scoring['new_detector_forwards'] == scoring['new_head_forwards'] ==
            scoring['new_pose_fits'] == scoring['new_rays'] == 0, 'postseal scoring contract')
    require(metrics['complete'] is True, 'complete statistics required')
    for key, name in (('predictions', 'PREDICTIONS.jsonl.gz'),
                      ('fixed_predictions', 'FIXED_PREDICTIONS.jsonl.gz')):
        bound(files[name], scoring[key], 'scoring:' + key)
    for key, name in (('new_scored_rows', 'PREDICTIONS.jsonl.gz'),
                      ('fixed_scored_rows', 'FIXED_PREDICTIONS.jsonl.gz'),
                      ('scoring_receipt', 'SCORING_RECEIPT.json'),
                      ('protocol', 'PROTOCOL.json'), ('cohort', 'cohort'), ('bootstrap', 'bootstrap')):
        bound(files[name], metrics['bindings'][key], 'metrics:' + key)
    bound(files['cohort'], protocol['inputs']['cohort'], 'frozen cohort')
    bound(files['bootstrap'], protocol['inputs']['bootstrap_draws'], 'frozen draws')
    return protocol, scoring, metrics, methods, contrasts


def freeze(args):
    folder, output = guard(args, 'freeze')
    files = paths(folder)
    protocol, _, _, methods, contrasts = validate_metadata(folder, files)
    write_new(output / FILES[0], dict(schema='supplemental_public_v7_CI_protocol_v1',
        inputs={n: binding(p) for n, p in files.items()}, accuracy_protocol_sha256=binding(files['PROTOCOL.json'])['sha256'],
        methods=methods, contrasts=[list(pair) for pair in contrasts], primary=protocol['primary'],
        frames=245, strata={'combined': 245, 'easy': 153, 'medium': 92},
        paired_scopes=list(SCOPES), sessions=13, existing_resamples=10000,
        arithmetic='independent standard-library fsum and linear interpolation quantile; no production imports',
        scalar_absolute_tolerance=ABS_TOL, scalar_relative_tolerance=REL_TOL,
        frozen_after_existing_accuracy_statistics=True, frozen_before_own_arithmetic=True,
        existing_accuracy_core_policy_or_rows_modified=0, new_draw_seed_model_PnP_score_training_RGB_GT_calls=0,
        no_automatic_numeric_retry=True,
        limits=['Checks saved metric arithmetic, not private GT truth or GPU execution authenticity.',
                'Existing geometric proxy DEV and session bootstrap are not independent generalization evidence.']))
    print(json.dumps(dict(frozen=True, protocol=binding(output / FILES[0]))), flush=True)


class Audit:
    def __init__(self):
        self.exact_checks = 0
        self.numeric_checks = 0
        self.max_difference = 0.0

    def eq(self, actual, expected, label):
        self.exact_checks += 1
        require(actual == expected, 'exact check: ' + label)

    def close(self, actual, expected, label):
        self.numeric_checks += 1
        if expected is None:
            require(actual is None, 'expected None: ' + label)
            return
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected),
                'finite scalar: ' + label)
        self.max_difference = max(self.max_difference, abs(actual - expected))
        require(math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                'numeric check: ' + label)


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    offset = (len(values) - 1) * probability
    lower = math.floor(offset)
    if lower == len(values) - 1:
        return float(values[lower])
    fraction = offset - lower
    return float(values[lower] + (values[lower + 1] - values[lower]) * fraction)


def distribution(values):
    n = len(values)
    mean = math.fsum(values) / n if n else None
    variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else None
    return dict(n=n, mean=mean, sample_variance=variance,
        sample_std=math.sqrt(variance) if variance is not None else None,
        median=quantile(values, .5), P90=quantile(values, .9), max=max(values) if n else None)


def compact_rows(path):
    """Parse one original full line at a time; retain only scored scalar fields."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            pose = row['pose']
            require(type(pose['available']) is bool and type(row['new_pose_estimated']) is bool and
                    type(row['fallback_used']) is bool, 'recorded boolean status required')
            require(not (row['new_pose_estimated'] and row['fallback_used']), 'NEW/fallback overlap')
            require(row['pose_available'] == pose['available'], 'available status differs')
            values = ({'translation_cm': float(pose['translation_cm']),
                       'rotation_deg': float(pose['rotation_deg']),
                       'ADDsym_cm': float(pose['ADDsym_m']) * 100} if pose['available'] else {})
            require(all(math.isfinite(x) for x in values.values()), 'nonfinite saved pose metric')
            yield dict(id=row['id'], session=row['session'], method=row['method'],
                available=pose['available'], new=row['new_pose_estimated'], fallback=row['fallback_used'],
                status=row['output_status'], values=values)


def run(args):
    folder, output = guard(args, 'run')
    lock = read(output / FILES[0])
    files = paths(folder)
    current = {name: binding(path) for name, path in files.items()}
    require(current == lock['inputs'], 'own frozen code/public input drift')
    write_new(output / FILES[1], dict(protocol=binding(output / FILES[0]),
        actual_saved_arithmetic_run_attempt=1, new_model_PnP_GT_score_draw_training_RGB_calls=0))
    start = time.perf_counter()
    audit = Audit()
    result = dict(schema='independent_public_v7_moment_and_CI_checks_v1', complete=False, passed=False,
        protocol=binding(output / FILES[0]), inputs=current, actual_saved_arithmetic_runs=1,
        production_modules_imported=False, standard_library_only=True,
        new_model_PnP_optimizer_ray_private_GT_score_draw_seed_training_RGB_calls=0,
        raw_rows_streamed_and_unmodified=True,
        retained_row_fields=['id', 'session', 'method', 'pose available/scalars', 'new', 'fallback', 'status'],
        reference_limit='existing GEOMETRIC_PROXY DEV saved metrics; not independent physical truth',
        failure_count=0, failures=[])
    counts = Counter()
    try:
        protocol, _, metrics, methods, contrasts = validate_metadata(folder, files)
        audit.eq(methods, lock['methods'], 'own methods')
        audit.eq([list(x) for x in contrasts], lock['contrasts'], 'own contrasts')
        audit.eq(list(SCOPES), lock['paired_scopes'], 'own scopes')
        cohort = read(files['cohort'])
        ids = cohort['ids']
        audit.eq(len(ids), 245, 'cohort count')
        audit.eq(len(set(ids)), 245, 'unique cohort')
        frames = {r['id']: r for r in cohort['frames']}
        audit.eq(set(frames), set(ids), 'cohort ID join')
        audit.eq(len(cohort['frames']), 245, 'unique cohort rows')
        audit.eq(dict(Counter(r['label'] for r in frames.values())),
                 {'clean': 153, 'moderate': 92}, 'fixed labels')
        with gzip.open(files['bootstrap'], 'rt', encoding='utf-8') as stream:
            bootstrap = json.load(stream)
        sessions, draws = bootstrap['sessions'], bootstrap['counts']
        audit.eq(len(sessions), 13, 'draw sessions')
        audit.eq(len(set(sessions)), 13, 'unique sessions')
        audit.eq(len(draws), 10000, 'existing draw count')
        for index, draw in enumerate(draws):
            audit.eq(len(draw), 13, 'draw width.' + str(index))
            audit.eq(sum(draw), 13, 'session draw multiplicity.' + str(index))
            require(all(type(v) is int and v >= 0 for v in draw), 'integer draw count')
        session_index = {name: i for i, name in enumerate(sessions)}
        require(all(r['session'] in session_index for r in frames.values()), 'unknown cohort session')
        groups = {method: {} for method in methods}
        for filename, expected_count, allowed in (
                ('PREDICTIONS.jsonl.gz', 1960, set(protocol['methods'])),
                ('FIXED_PREDICTIONS.jsonl.gz', 490, {'BASE', 'N3_SUBPIX'})):
            loaded = 0
            for row in compact_rows(files[filename]):
                require(row['method'] in allowed and row['id'] in frames, 'method/ID outside cohort')
                group = groups[row['method']]
                require(row['id'] not in group, 'duplicate method/ID')
                audit.eq(row['session'], frames[row['id']]['session'], 'row session join')
                group[row['id']] = row
                loaded += 1
            audit.eq(loaded, expected_count, filename + '.row count')
            counts['saved_scored_rows_streamed'] += loaded
        for method, group in groups.items():
            audit.eq(set(group), set(ids), 'complete same245.' + method)
        strata = [('combined', ids), ('easy', [i for i in ids if frames[i]['label'] == 'clean']),
                  ('medium', [i for i in ids if frames[i]['label'] == 'moderate'])]
        contrast_results = {}
        for name, selected_ids in strata:
            block = metrics['strata'][name]
            audit.eq(block['frames'], len(selected_ids), name + '.denominator')
            audit.eq(block['ids'], selected_ids, name + '.ID order')
            audit.eq(set(block['methods']), set(methods), name + '.methods')
            audit.eq(set(block['contrasts']), {a + '_minus_' + b for a, b in contrasts}, name + '.contrasts')
            # Small saved-scalar moments; no full witness is retained in RAM.
            for method in methods:
                data = [groups[method][i] for i in selected_ids]
                actual = block['methods'][method]
                operational = [r for r in data if r['available']]
                new = [r for r in operational if r['new']]
                fallback = [r for r in operational if r['fallback']]
                for key, value in dict(denominator=len(data), operational=len(operational),
                        new_pose=len(new), fallback=len(fallback), no_pose=len(data)-len(operational),
                        statuses=dict(Counter(r['status'] for r in data)),
                        fixed_control=sum(r['status'] == 'FRESH_FIXED_CONTROL' for r in data)).items():
                    audit.eq(actual[key], value, name + '.' + method + '.' + key)
                for scope, data_scope in [('operational', operational), ('new_pose', new), ('fallback', fallback)]:
                    for key in METRICS:
                        expected = distribution([r['values'][key] for r in data_scope])
                        recorded = actual['metrics'][scope][key]
                        audit.eq(recorded['n'], expected['n'], 'moment n')
                        for field in ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'max'):
                            audit.close(recorded[field], expected[field], 'moment.' + field)
                            counts['moment_scalars_checked'] += 1
            contrast_results[name] = {}
            for candidate_name, comparator_name in contrasts:
                key = candidate_name + '_minus_' + comparator_name
                candidate, comparator = groups[candidate_name], groups[comparator_name]
                contrast_results[name][key] = {}
                for scope in SCOPES:
                    recorded = block['contrasts'][key][scope]
                    pair_ids = [i for i in selected_ids if candidate[i]['available'] and comparator[i]['available'] and
                        (scope == 'common_operational' or candidate[i]['new']) and
                        (scope != 'both_new_pose' or comparator[i]['new'])]
                    n = len(pair_ids)
                    multiplicities = [0] * 13
                    for fid in pair_ids:
                        multiplicities[session_index[candidate[fid]['session']]] += 1
                    denominators = [sum(draw[j] * multiplicities[j] for j in range(13)) for draw in draws]
                    nonempty = sum(denominator > 0 for denominator in denominators)
                    for field, value in dict(scope=scope, denominator=len(selected_ids), common_frames=n,
                            pair_ids=pair_ids, candidate_new_pose_in_pairs=sum(candidate[i]['new'] for i in pair_ids),
                            comparator_new_pose_in_pairs=sum(comparator[i]['new'] for i in pair_ids),
                            bootstrap_nonempty_resamples=nonempty, bootstrap_empty_resamples=10000-nonempty).items():
                        audit.eq(recorded[field], value, name + '.' + key + '.' + field)
                    checked_metrics = {}
                    for metric in METRICS:
                        deltas = [candidate[i]['values'][metric] - comparator[i]['values'][metric] for i in pair_ids]
                        sums = [math.fsum(delta for fid, delta in zip(pair_ids, deltas)
                                          if session_index[candidate[fid]['session']] == j) for j in range(13)]
                        samples = [math.fsum(draw[j] * sums[j] for j in range(13)) / denominator
                                   for draw, denominator in zip(draws, denominators) if denominator > 0]
                        expected = distribution(deltas)
                        interval = [quantile(samples, .025), quantile(samples, .975)] if samples else None
                        actual = recorded['metrics'][metric]
                        audit.eq(actual['n'], n, 'paired metric n')
                        audit.eq(actual['bootstrap_nonempty_resamples'], nonempty, 'paired nonempty draw')
                        audit.close(actual['mean_delta'], expected['mean'], 'paired mean delta')
                        if interval is None:
                            audit.eq(actual['CI95'], None, 'empty CI remains None')
                            counts['None_CI_slots'] += 1
                        else:
                            require(isinstance(actual['CI95'], list) and len(actual['CI95']) == 2, 'CI shape')
                            for actual_limit, expected_limit in zip(actual['CI95'], interval):
                                audit.close(actual_limit, expected_limit, 'CI endpoint')
                            counts['nonempty_CI_slots'] += 1
                        checked_metrics[metric] = dict(n=n, mean_delta=expected['mean'],
                            paired_delta_distribution=expected, CI95=interval,
                            bootstrap_nonempty_resamples=nonempty)
                        counts['metric_CI_slots_checked'] += 1
                        counts['paired_delta_scalars_computed'] += n
                        counts['nonempty_resampled_means_computed'] += len(samples)
                    contrast_results[name][key][scope] = dict(n=n, pair_ids=pair_ids,
                        bootstrap_empty_resamples=10000-nonempty, metrics=checked_metrics)
                    counts['pair_groups_checked'] += 1
                    counts['resample_denominators_computed'] += 10000
        audit.eq(counts['pair_groups_checked'], 144, 'all fixed paired groups')
        audit.eq(counts['metric_CI_slots_checked'], 432, 'all fixed CI slots')
        audit.eq(counts['moment_scalars_checked'], 1620, 'all fixed moment scalars')
        # Re-hash complete original inputs after arithmetic; do not mutate rows.
        audit.eq({name: binding(path) for name, path in files.items()}, current, 'inputs unchanged after arithmetic')
        result.update(complete=True, passed=True, checked_contrasts=contrast_results,
            new_draws_generated=0, actual_counts=dict(counts))
    except Exception as error:
        result.update(failure_count=1, failures=[dict(type=type(error).__name__, message=str(error))],
                      actual_counts=dict(counts))
    result.update(exact_checks=audit.exact_checks, numeric_checks=audit.numeric_checks,
        max_numeric_difference=audit.max_difference, absolute_tolerance=ABS_TOL,
        relative_tolerance=REL_TOL, elapsed_seconds=time.perf_counter()-start, Python=sys.version)
    write_new(output / FILES[2], result)
    print(json.dumps({key: result[key] for key in ('passed', 'complete', 'failure_count', 'actual_counts',
                                                 'exact_checks', 'numeric_checks', 'max_numeric_difference')},
                     ensure_ascii=False), flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, default=DOC,
                        help='new receipt names in DOC or a separate review directory')
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else run(args)


if __name__ == '__main__':
    main()
