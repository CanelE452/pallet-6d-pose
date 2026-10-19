"""Public saved-metric tables from independently checked postseal bundles.

Only packaging and metadata validation occur here. No raw reference input,
pose metric calculation, calibration, inference, solve, fit or bootstrap is used.
Each saved seed is a distinct arm; repeated baselines are exported once.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from . import comparison_figures as C

M = C.M
POPS = {'PRIMARY192': ('TEST', 'REMAINDER'), 'TEST49': ('TEST',),
        'REMAINDER143': ('REMAINDER',), 'CAL46': ('CAL',)}
EXPECTED = {'PRIMARY192': 192, 'TEST49': 49, 'REMAINDER143': 143, 'CAL46': 46}
ROW_FIELDS = ('id', 'partition', 'session', 'cluster_id', 'arm',
              'translation_cm', 'rotation_deg', 'success', 'pose_available')
STATS = ('n', 'missing', 'mean', 'sample_variance', 'sample_std', 'median', 'P90', 'P95', 'max')
AXES = (('translation_cm', 'T', 'cm'), ('rotation_deg', 'R', 'deg'))
SUMMARY_FIELDS = ('population', 'arm', 'denominator', 'successes', 'success_rate',
                  'no_pose', 'T_gt50cm', 'T_gt100cm', 'R_gt45deg',
                  'fallback_known_n', 'fallback_true_n', 'fallback_false_n', 'fallback_unknown_n')


def stat_column(prefix, unit, name):
    if name in ('n', 'missing'):
        return prefix + '_' + name
    return prefix + '_' + name + '_' + unit + ('2' if name == 'sample_variance' else '')


ALL_FIELDS = SUMMARY_FIELDS + tuple(stat_column(prefix, unit, name)
    for _, prefix, unit in AXES for name in STATS)


def count(value, n, label):
    if type(value) is not int or not 0 <= value <= n:
        raise ValueError('Invalid full-population count: ' + label)
    return value


def public_summary(population, arm, summary):
    """Copy saved descriptive values; never calculate moments or pose errors."""
    n = EXPECTED[population]
    if summary['denominator'] != n:
        raise ValueError('Full population denominator changed')
    out = dict(population=population, arm=arm, denominator=n)
    for key in ('successes', 'no_pose', 'T_gt50cm', 'T_gt100cm', 'R_gt45deg'):
        out[key] = count(summary[key], n, key)
    rate = summary['success_rate']
    if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate != out['successes'] / n:
        raise ValueError('Saved success count/rate differs')
    if out['successes'] > n - out['no_pose']:
        raise ValueError('No-pose outcomes must remain binary failures')
    out['success_rate'] = rate
    fallback = summary['status_counts']['fallback']
    for key in ('true_n', 'false_n', 'unknown_n'):
        out['fallback_' + key] = count(fallback[key], n, 'fallback_' + key)
    if sum(fallback[key] for key in ('true_n', 'false_n', 'unknown_n')) != n:
        raise ValueError('Fallback true/false/unknown must cover the full population')
    out['fallback_known_n'] = fallback['true_n'] + fallback['false_n']
    for axis, prefix, unit in AXES:
        dist = summary[axis]
        available = count(dist['n'], n, axis + '/n')
        missing = count(dist['missing'], n, axis + '/missing')
        if (available + missing != n or dist['full_denominator'] != n or
            dist['ddof'] != 1 or dist['available_only'] is not True or dist['tail_trimming'] is not False):
            raise ValueError('Saved available-only distribution contract differs')
        if missing != out['no_pose']:
            raise ValueError('Saved no-pose and continuous-error missing counts differ')
        for name in STATS:
            value = dist[name]
            if name not in ('n', 'missing'):
                should_be_null = available < (2 if name in ('sample_variance', 'sample_std') else 1)
                if should_be_null:
                    if value is not None:
                        raise ValueError('Undefined saved statistic must remain null')
                elif (isinstance(value, bool) or not isinstance(value, (float, int)) or
                      not math.isfinite(value) or value < 0):
                    raise ValueError('Finite nonnegative saved statistic required')
            out[stat_column(prefix, unit, name)] = value
    if (out['T_gt100cm'] > out['T_gt50cm'] or out['T_gt50cm'] > out['T_n'] or
        out['R_gt45deg'] > out['R_n']):
        raise ValueError('Saved tail count exceeds available denominator')
    return out


def authenticate_tables(manifest):
    """Extend the existing figure authentication with summary/ancestry checks."""
    plan, rows, _ = C.authenticate(manifest)
    saved, independent_saved, arm_sources, bindings = {}, {}, {}, []
    for ordinal, source in enumerate(plan['sources']):
        ev, ind = Path(source['evaluation_dir']), Path(source['independent_dir'])
        es = M.read(ev / 'EVALUATION_SEAL.json')
        artifacts = {}
        for name in ('SCORED_ROWS.jsonl.gz', 'SUMMARY.json', 'PAIRED.json'):
            actual = M.sha(ev / name)
            if actual != es['artifacts'][name]:
                raise ValueError('Canonical artifact changed: ' + name)
            artifacts[name] = actual
        check = M.read(ind / 'INDEPENDENT_VERIFICATION.json')
        if (check.get('errors') != [] or check.get('rows') != 238 or
            check.get('arms') != es['arms'] or check.get('populations') != EXPECTED or
            es.get('populations') != EXPECTED or check.get('reference_sha256') != es['reference_sha256'] or
            check.get('prediction_bindings') != es['prediction_bindings'] or
            check.get('success_classifications_exact') is not True or not check.get('source_sha256')):
            raise ValueError('Independent verification ancestry or full-population check differs')
        canonical = M.read(ev / 'SUMMARY.json')
        independent = M.read(ind / 'INDEPENDENT_SUMMARY.json')
        if set(canonical) != set(POPS) or set(independent) != set(POPS):
            raise ValueError('Exactly the four original reporting populations required')
        arms = es['arms']
        if not arms or len(arms) != len(set(arms)):
            raise ValueError('Unique completed arm names required')
        for pop in POPS:
            if set(canonical[pop]) != set(arms) or set(independent[pop]) != set(arms):
                raise ValueError('Every completed arm must cover every population')
            for arm in arms:
                key = (pop, arm)
                cp = public_summary(pop, arm, canonical[pop][arm])
                ip = public_summary(pop, arm, independent[pop][arm])
                for name, value in cp.items():
                    other = ip[name]
                    if isinstance(value, float) and isinstance(other, (float, int)):
                        agrees = abs(value - other) <= 1e-6
                    else:
                        agrees = value == other
                    if not agrees:
                        raise ValueError('Canonical/independent saved summary differs: ' + arm + '/' + name)
                if key in saved and (saved[key] != canonical[pop][arm] or
                                     independent_saved[key] != independent[pop][arm]):
                    raise ValueError('Repeated baseline summaries differ across sources: ' + arm)
                saved[key], independent_saved[key] = canonical[pop][arm], independent[pop][arm]
        for arm in arms:
            arm_sources.setdefault(arm, []).append(ordinal)
        # Public provenance contains hashes and arm/source ordinals, no private
        # input paths, raw status strings, poses, reference values or weights.
        bindings.append(dict(source_ordinal=ordinal, arms=arms,
            evaluation_seal_sha256=M.sha(ev / 'EVALUATION_SEAL.json'), canonical_artifacts=artifacts,
            independent_artifacts=source['independent_artifacts'],
            independent_checker_source_sha256=check['source_sha256'],
            prediction_bindings=[{k: b[k] for k in ('sha256', 'seal_sha256')} for b in es['prediction_bindings']]))
    arms = sorted(arm_sources)
    if any(set(row['metrics']) != set(arms) for row in rows.values()):
        raise ValueError('All238 rows must have every unique completed arm')
    summary_rows = [public_summary(pop, arm, saved[(pop, arm)]) for pop in POPS for arm in arms]
    row_rows = []
    for row in sorted(rows.values(), key=lambda item: item['id']):
        for arm in arms:
            metric = C.public_metric(row['metrics'][arm])
            if (type(metric['success']) is not bool or type(metric['pose_available']) is not bool or
                (not metric['pose_available'] and metric['success']) or
                (metric['pose_available'] and any(metric[k] is None for k in ('translation_cm', 'rotation_deg')))):
                raise ValueError('Saved outcome/pose-availability contract differs')
            row_rows.append({k: row[k] for k in ('id', 'partition', 'session', 'cluster_id')} |
                            dict(arm=arm, **metric))
    return summary_rows, row_rows, dict(manifest_sha256=M.sha(manifest),
        reference_sha256=M.read(Path(plan['sources'][0]['evaluation_dir']) / 'EVALUATION_SEAL.json')['reference_sha256'],
        sources=bindings, arm_source_ordinals=arm_sources)


def write_csv(path, fields, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(manifest, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Export requires a new output folder; existing folders are never overwritten')
    summary, rows, bindings = authenticate_tables(manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'ALL_POPULATION_METRICS.csv', ALL_FIELDS, summary)
    write_csv(output / 'ROW_METRICS.csv', ROW_FIELDS, rows)
    schema = dict(schema='PUBLIC_SAVED_METRICS_TABLES_V1', populations=EXPECTED,
        population_partitions={k: list(v) for k, v in POPS.items()},
        summary_columns=list(ALL_FIELDS), row_columns=list(ROW_FIELDS),
        summary_values='copied from authenticated canonical SUMMARY.json; no moments or errors recalculated',
        units=dict(T='cm', T_sample_variance='cm^2', R='deg', R_sample_variance='deg^2', success_rate='fraction'),
        sample_variance_ddof=1, sample_std_ddof=1, quantiles='saved linear quantiles',
        continuous='available-only n; missing is blank, never zero imputed; no trimming or winsorization',
        CSV_null='empty field; an actual numeric zero remains 0 or 0.0', CSV_boolean='True or False',
        binary='successes, no_pose and fallback true/false/unknown use full population denominator',
        tails='strict T>50cm, T>100cm, R>45deg; copied saved counts',
        seeds='each seed is a separate named arm; seeds are not pooled as independent frame samples',
        repeated_baselines='one arm output, exact canonical and independent summary equality required across sources',
        populations_overlap='PRIMARY192 is TEST49 plus REMAINDER143; CAL46 is separate',
        privacy='only named identity fields, saved scalar errors and outcomes; no R/t, raw poses, references or weights',
        independent_real_test_n=0)
    M.write(output / 'METRICS_SCHEMA.json', schema)
    M.write(output / 'INPUT_BINDINGS.json', bindings)
    names = ('ALL_POPULATION_METRICS.csv', 'ROW_METRICS.csv', 'METRICS_SCHEMA.json', 'INPUT_BINDINGS.json')
    seal = dict(schema='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1', complete=True,
        arms=sorted(bindings['arm_source_ordinals']), populations=EXPECTED, summary_rows=len(summary),
        scalar_row_rows=len(rows), identities=238, all_outliers_and_failures_retained=True,
        metrics_recomputed=False, raw_references_decoded=0, solve_fit_forward_calls=0,
        source_sha256=M.sha(__file__), authentication_source_sha256=M.sha(C.__file__),
        io_utility_source_sha256=M.sha(M.__file__),
        bindings_sha256=M.sha(output / 'INPUT_BINDINGS.json'),
        artifacts={name: M.sha(output / name) for name in names}, independent_real_test_n=0)
    M.write(output / 'EXPORT_SEAL.json', seal)
    return seal


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(export(a.manifest, a.output), sort_keys=True))
