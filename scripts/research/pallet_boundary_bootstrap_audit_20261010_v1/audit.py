"""Replay recorded v2 paired CIs independently, from public saved rows only.

No original statistics, model, GT, pose, image or ray module is imported. Freeze
once, then perform one exclusive arithmetic run; a failure is preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

REPO = Path(__file__).resolve().parents[3]
NAME = 'pallet_boundary_bootstrap_audit_20261010_v1'
DOC = REPO / '_docs/experiments' / NAME
V2 = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
PRIMARY = 'N3_VALIDATED_ROLE'
METHODS = ('VALIDATED_ROLE_ONLY', PRIMARY, 'N3_VALIDATED_ROLE_NO_MASK',
           'N3_BASIN_ROBUST', 'N3_BASIN_STANDARD')
CONTRASTS = [(m, 'N3_SUBPIX') for m in METHODS] + [
    (PRIMARY, 'N3_BASIN_ROBUST'), (PRIMARY, 'N3_VALIDATED_ROLE_NO_MASK'),
    ('N3_BASIN_ROBUST', 'N3_BASIN_STANDARD')]
SCOPES = ('common_operational', 'candidate_new_pose')
FIELDS = {'translation_cm': ('translation_cm', 1., 'cm'),
          'rotation_deg': ('rotation_deg', 1., 'degree'),
          'ADDsym_cm': ('ADDsym_m', 100., 'cm')}
ABS_TOL = 1e-10
REL_TOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(value)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve()
    assert p.is_relative_to(REPO), 'All authoritative inputs are public repository files'
    return dict(path=str(p.relative_to(REPO)), sha256=sha(p), bytes=p.stat().st_size)


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        return json.load(f)


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_new(path, values):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as stream:
            for value in values:
                stream.write((json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                          allow_nan=False) + '\n').encode())


def input_paths():
    return dict(code=Path(__file__),
        scored_rows=V2 / 'PREDICTIONS.jsonl.gz',
        fixed_scored_rows=V2 / 'FIXED_PREDICTIONS.jsonl.gz',
        recorded_metrics=V2 / 'METRICS.json',
        cohort=REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json',
        bootstrap=REPO / '_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz',
        previous_metadata_only_verification=V2 / 'REVIEW_CHECKS.json',
        recorded_statistics_protocol=V2 / 'STATISTICS_PROTOCOL.json',
        original_statistics_code=REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/statistics.py',
        original_statistics_helper_code=REPO / 'scripts/research/pallet_kp_corrected_supervision_20261010_v1/statistics.py')


def tracked_snapshot():
    listed = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0')
    # Sparse-checkout paths absent from this worktree are not claimed inspected.
    return [binding(REPO / name) for name in sorted(listed) if name and (REPO / name).is_file()]


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__,
                CPU_thread_environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
                original_statistics_imported=False, other_research_modules_imported=False)


def freeze(output):
    begin = time.monotonic()
    assert not (output / 'PROTOCOL.json').exists() and not (output / 'CHECKS.json').exists(), 'Preserve existing audit'
    output.mkdir(parents=True, exist_ok=True)
    p = input_paths()
    before = {k: binding(v) for k, v in p.items()}
    cohort = read(p['cohort'])
    assert cohort['count'] == len(cohort['ids']) == len(set(cohort['ids'])) == 245
    assert Counter(r['label'] for r in cohort['frames']) == {'clean': 153, 'moderate': 92}
    preserved = tracked_snapshot()
    assert before == {k: binding(v) for k, v in p.items()}
    protocol = dict(schema='independent_v2_saved_bootstrap_protocol_v1', frozen_before_CI_replay=True,
        inputs=before, protected_prior_materialized_tracked_files=preserved,
        protected_file_count=len(preserved),
        population=dict(real_evaluation_reexecuted=False, saved_new_rows=1225, saved_fixed_rows=490,
            combined=245, easy=153, medium=92, severe_excluded=74, sessions=13, resamples=10000),
        methods=list(METHODS), controls=['BASE', 'N3_SUBPIX'], contrasts=[list(x) for x in CONTRASTS],
        scopes=list(SCOPES), metrics={k:dict(raw_field=v[0], factor=v[1], unit=v[2]) for k, v in FIELDS.items()},
        calculation=dict(pair_selection='Both stored pose.available; candidate_new_pose additionally stored candidate.new_pose_estimated. Preserve cohort order and exact IDs.',
            delta='(candidate stored proxy error - comparator stored proxy error) * factor; ADDsym_m difference then times100',
            cluster_totals='Independently append raw delta by session and math.fsum each session; count frames per session explicitly, without original numpy.bincount implementation.',
            denominator='sum each existing13-column integer multiplicity row * paired-frame counts; zero denominator resample excluded and counted',
            numerator='sum elementwise float64 existing multiplicity * independently fsum session delta; divide by positive frame denominator',
            CI='Python sorted resampled mean list, linear order-statistic interpolation at p=.025 and p=.975; h=(n-1)*p. No original statistics imports or fresh draws.',
            means='math.fsum paired deltas / paired frame count; an empty pair requires mean and CI None',
            exact_checks=['cohort order and per-method population', 'common_ids', 'pair_ids', 'excluded_ids', 'common_frames', 'denominator',
                          'bootstrap_nonempty_resamples', 'recorded metric n/unit', 'combined aliases'],
            numeric_tolerance=dict(absolute=ABS_TOL, relative=REL_TOL,
                                   rule='abs(replayed-recorded)<=absolute+relative*abs(recorded)',
                                   purpose='fixed floating-point summation tolerance, not a performance threshold'),
            canonical_pair_groups=48, canonical_metric_CIs=144,
            combined_alias_metric_CIs=48, expected_total_recorded_metric_CI_comparisons=192),
        execution=dict(protocol_freeze_calls=1, allowed_arithmetic_runs=1, CI_replays_before_freeze=0,
            new_accuracy_scores=0, detector_forwards=0, neck_forwards=0, head_forwards=0,
            pose_fits=0, rays=0, training_updates=0, RGB_generated_or_decoded=0, fresh_bootstrap_draws=0),
        limits=['Stored GEOMETRIC_PROXY errors are inputs; no physical GT, scorer, model, ray or pose solver is independently executed.',
            'This checks numeric uncertainty computation; it does not establish real accuracy improvement, causal attribution or physical boundary ownership.',
            'Fixed old13session multiplicities are reused for245 and its strata; inference changes, threshold selection and new draws are absent.',
            'Previous CI metadata-only receipt remains byte-for-byte preserved and is not retroactively presented as numeric replay.',
            'Any failure remains a separate CHECKS receipt; no changed definition or retry is permitted within this frozen run.'],
        runtime=runtime(), freeze_elapsed_seconds=time.monotonic() - begin)
    write_new(output / 'PROTOCOL.json', protocol)
    print(json.dumps(dict(stage='frozen', protected_files=len(preserved), protocol=binding(output / 'PROTOCOL.json'),
                          actual_CI_replays=0)))


def linear_quantile(sorted_values, p):
    if not sorted_values:
        return None
    h = (len(sorted_values) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    fraction = h - lo
    return float(sorted_values[lo] + fraction * (sorted_values[hi] - sorted_values[lo]))


def run(output):
    begin = time.monotonic()
    assert not (output / 'CHECKS.json').exists() and not (output / 'ROWS.jsonl.gz').exists(), 'Preserve prior arithmetic run'
    protocol_path = output / 'PROTOCOL.json'
    protocol = read(protocol_path)
    protocol_binding = binding(protocol_path)
    paths = input_paths()
    checks = Counter()
    work = Counter(arithmetic_runs=1, new_accuracy_scores=0, detector_forwards=0, neck_forwards=0,
                   head_forwards=0, pose_fits=0, rays=0, training_updates=0,
                   RGB_generated_or_decoded=0, fresh_bootstrap_draws=0)
    failures = []
    result_rows = []
    largest_difference = 0.

    def exact(actual, expected, label):
        checks['exact_checks'] += 1
        if actual != expected:
            failures.append(dict(check=label, observed=plain(actual), expected=plain(expected)))

    def numeric(actual, expected, label):
        nonlocal largest_difference
        checks['numeric_checks'] += 1
        if actual is None or expected is None:
            exact(actual, expected, label + '/null')
            return
        if not isinstance(expected, (float, int)) or not math.isfinite(expected) or not math.isfinite(actual):
            failures.append(dict(check=label, observed=plain(actual), expected=plain(expected), reason='nonfinite or invalid numeric type'))
            return
        gap = abs(actual - expected)
        largest_difference = max(largest_difference, gap)
        if gap > ABS_TOL + REL_TOL * abs(expected):
            failures.append(dict(check=label, observed=actual, expected=expected, absolute_difference=gap))

    result = dict(schema='independent_v2_saved_bootstrap_checks_v1', complete=False, passed=False,
        protocol=protocol_binding, inputs=protocol['inputs'], runtime=runtime(), actual_execution=work,
        private_assets_opened=False, new_GT_read=False, GT_used_only_as_previously_recorded_proxy_errors=True,
        original_statistics_imported=False, no_other_research_module_imports=True,
        limits=protocol['limits'])
    try:
        before = {k: binding(v) for k, v in paths.items()}
        assert before == protocol['inputs'], 'Frozen authoritative input changed'
        assert runtime() == protocol['runtime']
        prior = protocol['protected_prior_materialized_tracked_files']
        assert all(binding(REPO / b['path']) == b for b in prior), 'Prior published bytes changed before audit'
        metrics = read(paths['recorded_metrics'])
        cohort = read(paths['cohort'])
        bootstrap = read(paths['bootstrap'])
        new = rows(paths['scored_rows'])
        fixed = rows(paths['fixed_scored_rows'])
        work['saved_scored_rows_loaded'] = len(new) + len(fixed)
        exact(len(new), 1225, 'new-row count')
        exact(len(fixed), 490, 'fixed-row count')
        ids = cohort['ids']
        assert len(ids) == len(set(ids)) == 245
        assert cohort['count'] == 245
        labels = {r['id']: r['label'] for r in cohort['frames']}
        frame_sessions = {r['id']: r['session'] for r in cohort['frames']}
        assert set(labels) == set(ids) and Counter(labels.values()) == {'clean': 153, 'moderate': 92}
        grouped = {m: {} for m in ('BASE', 'N3_SUBPIX') + METHODS}
        for row in new + fixed:
            m, fid = row['method'], row['id']
            assert m in grouped and fid in ids and fid not in grouped[m], 'Raw method/ID missing, duplicated or foreign'
            assert row['session'] == frame_sessions[fid], 'Stored session differs from frozen cohort'
            assert type(row['pose']['available']) is bool and type(row['new_pose_estimated']) is bool
            assert row['pose_available'] == row['pose']['available'] and row['no_pose'] != row['pose']['available']
            if row['pose']['available']:
                assert all(math.isfinite(row['pose'][v[0]]) for v in FIELDS.values())
            grouped[m][fid] = row
        assert all(len(g) == 245 and set(g) == set(ids) for g in grouped.values())
        sessions = bootstrap['sessions']
        assert len(sessions) == len(set(sessions)) == 13
        draw_lists = bootstrap['counts']
        assert len(draw_lists) == 10000 and all(len(x) == 13 and sum(x) == 13 and
            all(type(v) is int and 0 <= v <= 13 for v in x) for x in draw_lists)
        draw_uint16 = np.asarray(draw_lists, dtype='<u2')
        draw_hash = hashlib.sha256(draw_uint16.tobytes(order='C')).hexdigest()
        assert draw_hash == bootstrap['serialized_raw_sha256'] == metrics['bootstrap']['serialized_raw_sha256']
        exact(metrics['bootstrap']['sessions'], sessions, 'bootstrap session order')
        exact(metrics['bootstrap']['resamples'], 10000, 'recorded resamples')
        exact(metrics['bootstrap']['new_draws_generated'], 0, 'recorded fresh draws')
        assert {r['session'] for r in new + fixed} == set(sessions)
        draw_int = draw_uint16.astype(np.int64)
        draw_float = draw_uint16.astype(np.float64)
        session_index = {name: i for i, name in enumerate(sessions)}
        work['original_multiplicity_rows_loaded'] = len(draw_lists)
        work['original_multiplicity_entries_loaded'] = draw_uint16.size
        expected_contrasts = {a + '_minus_' + b for a, b in CONTRASTS}
        exact(set(metrics['strata']), {'combined', 'easy', 'medium'}, 'stratum names')
        assert metrics['bindings']['new_scored_rows']['sha256'] == before['scored_rows']['sha256']
        assert metrics['bindings']['fixed_scored_rows']['sha256'] == before['fixed_scored_rows']['sha256']
        assert metrics['bindings']['cohort']['sha256'] == before['cohort']['sha256']
        assert metrics['bindings']['bootstrap']['sha256'] == before['bootstrap']['sha256']
        strata = [('combined', ids), ('easy', [fid for fid in ids if labels[fid] == 'clean']),
                  ('medium', [fid for fid in ids if labels[fid] == 'moderate'])]
        for name, selected in strata:
            stored_stratum = metrics['strata'][name]
            exact(stored_stratum['frames'], len(selected), name + '/frames')
            exact(set(stored_stratum['contrasts']), expected_contrasts, name + '/contrasts')
            for a, b in CONTRASTS:
                key = a + '_minus_' + b
                exact(set(stored_stratum['contrasts'][key]), set(SCOPES), name + '/' + key + '/scopes')
                for scope in SCOPES:
                    pair = stored_stratum['contrasts'][key][scope]
                    common = [fid for fid in selected if grouped[a][fid]['pose']['available'] and
                        grouped[b][fid]['pose']['available'] and
                        (scope == 'common_operational' or grouped[a][fid]['new_pose_estimated'])]
                    excluded = [fid for fid in selected if fid not in set(common)]
                    prefix = name + '/' + key + '/' + scope
                    for field, expected in [('common_ids', common), ('pair_ids', common), ('excluded_ids', excluded),
                        ('common_frames', len(common)), ('denominator', len(selected)), ('scope', scope)]:
                        exact(pair[field], expected, prefix + '/' + field)
                    counts = [0] * 13
                    for fid in common:
                        counts[session_index[frame_sessions[fid]]] += 1
                    denominator = np.sum(draw_int * np.asarray(counts, np.int64)[None, :], axis=1, dtype=np.int64)
                    nonempty = denominator > 0
                    nonempty_count = int(nonempty.sum())
                    exact(pair['bootstrap_nonempty_resamples'], nonempty_count, prefix + '/nonempty')
                    exact(set(pair['metrics']), set(FIELDS), prefix + '/metrics')
                    record = dict(stratum=name, contrast=key, candidate=a, comparator=b, scope=scope,
                        denominator=len(selected), common_frames=len(common), pair_ids=common, excluded_ids=excluded,
                        sessions=sessions, paired_frame_counts_by_session=counts,
                        bootstrap_nonempty_resamples=nonempty_count,
                        bootstrap_empty_resamples=10000 - nonempty_count,
                        denominator_int64_SHA256=hashlib.sha256(denominator.astype('<i8').tobytes()).hexdigest(), metrics={})
                    work['pair_groups_computed'] += 1
                    work['bootstrap_denominators_computed'] += len(denominator)
                    for metric, (raw_field, factor, unit) in FIELDS.items():
                        reported = pair['metrics'][metric]
                        exact(reported['n'], len(common), prefix + '/' + metric + '/n')
                        exact(reported['unit'], unit, prefix + '/' + metric + '/unit')
                        delta = [(grouped[a][fid]['pose'][raw_field] - grouped[b][fid]['pose'][raw_field]) * factor for fid in common]
                        per_session = [[] for _ in sessions]
                        for fid, value in zip(common, delta):
                            per_session[session_index[frame_sessions[fid]]].append(value)
                        totals = [math.fsum(v) for v in per_session]
                        numerator = np.sum(draw_float * np.asarray(totals, np.float64)[None, :], axis=1, dtype=np.float64)
                        samples = numerator[nonempty] / denominator[nonempty]
                        assert np.isfinite(samples).all()
                        ordered = sorted(samples.tolist())
                        ci = [linear_quantile(ordered, .025), linear_quantile(ordered, .975)] if ordered else None
                        mean = math.fsum(delta) / len(delta) if delta else None
                        numeric(mean, reported['mean_delta'], prefix + '/' + metric + '/mean_delta')
                        if ci is None or reported['CI95'] is None:
                            exact(ci, reported['CI95'], prefix + '/' + metric + '/CI95/null')
                        else:
                            assert isinstance(reported['CI95'], list) and len(reported['CI95']) == 2
                            for j in range(2):
                                numeric(ci[j], reported['CI95'][j], prefix + '/' + metric + '/CI95/' + str(j))
                                work['canonical_CI_endpoints_compared'] += 1
                        record['metrics'][metric] = dict(raw_field=raw_field, factor=factor, unit=unit,
                            n=len(delta), mean_delta=mean, reported_mean_delta=reported['mean_delta'],
                            session_delta_totals_fsum=totals,
                            bootstrap_mean_float64_SHA256=hashlib.sha256(samples.astype('<f8').tobytes()).hexdigest(),
                            CI95=ci, reported_CI95=reported['CI95'],
                            CI95_absolute_difference=None if ci is None else [abs(ci[j] - reported['CI95'][j]) for j in range(2)])
                        work['distinct_metric_CIs_replayed'] += 1
                        work['bootstrap_numerators_computed'] += len(numerator)
                        work['nonempty_resampled_means_computed'] += len(samples)
                        work['paired_raw_deltas_computed'] += len(delta)
                        if name == 'combined':
                            alias = metrics['contrasts'][key][scope]
                            exact(alias, pair, 'combined_alias/' + key + '/' + scope)
                            alias_metric = alias['metrics'][metric]
                            if ci is None:
                                exact(alias_metric['CI95'], None, 'combined_alias/' + key + '/' + scope + '/' + metric)
                            else:
                                for j in range(2):
                                    numeric(ci[j], alias_metric['CI95'][j], 'combined_alias/' + key + '/' + scope + '/' + metric + '/' + str(j))
                                    work['alias_CI_endpoints_compared'] += 1
                            work['combined_alias_metric_CIs_checked'] += 1
                    result_rows.append(record)
        assert work['pair_groups_computed'] == 48 and work['distinct_metric_CIs_replayed'] == 144
        assert work['combined_alias_metric_CIs_checked'] == 48
        after = {k: binding(v) for k, v in paths.items()}
        old_after = [binding(REPO / b['path']) for b in prior]
        assert before == after and prior == old_after and binding(protocol_path) == protocol_binding
        result.update(complete=True, passed=not failures, input_hashes_before_after_equal=True,
            protected_prior_materialized_tracked_files_unchanged=True, protected_prior_file_count=len(prior),
            bootstrap_serialized_raw_SHA256=draw_hash, CI_numeric_replay=True,
            failure_count=len(failures), failures=failures, comparison_counts=dict(checks),
            max_numeric_difference=largest_difference, actual_execution=dict(work),
            source_population=dict(combined=245, easy=153, medium=92, sessions=13, resamples=10000),
            primary=next(r for r in result_rows if r['stratum'] == 'combined' and
                         r['contrast'] == PRIMARY + '_minus_N3_SUBPIX' and r['scope'] == 'common_operational'))
    except Exception as error:
        result.update(failure_count=len(failures) + 1, failures=failures,
            fatal_failure=dict(type=type(error).__name__, message=str(error)),
            comparison_counts=dict(checks), max_numeric_difference=largest_difference,
            actual_execution=dict(work), failure_preserved=True)
    save_new(output / 'ROWS.jsonl.gz', result_rows)
    result['computed_rows'] = binding(output / 'ROWS.jsonl.gz')
    result['elapsed_seconds'] = time.monotonic() - begin
    write_new(output / 'CHECKS.json', result)
    print(json.dumps(dict(passed=result['passed'], complete=result['complete'], failure_count=result.get('failure_count'),
        max_numeric_difference=largest_difference, actual_execution=dict(work), checks=binding(output / 'CHECKS.json'))))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    parser.add_argument('--output', type=Path, default=DOC)
    arguments = parser.parse_args()
    freeze(arguments.output.resolve()) if arguments.stage == 'freeze' else run(arguments.output.resolve())
