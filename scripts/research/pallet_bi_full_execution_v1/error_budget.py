"""Posthoc descriptive error budgets from authenticated public scalar tables.

No pose/reference/feature/RGB is read and no error metric is recomputed. Existing
5/50/100cm and5/45deg cutoffs describe all saved errors, with no tail removal.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import math
from pathlib import Path

from . import export_results as E

M = E.M
ABS_TOL = 1e-6
REL_TOL = 1e-10
CATEGORIES = ('JOINT_SUCCESS', 'TRANSLATION_ONLY_FAILURE', 'ROTATION_ONLY_FAILURE', 'BOTH_FAILURE', 'NO_POSE')
AXES = [('T', 'cm', 'translation_cm', ('<=5', '(5,50]', '(50,100]', '>100')),
        ('R', 'deg', 'rotation_deg', ('<=5', '(5,45]', '>45'))]
JOINT_FIELDS = ('population', 'arm', 'full_denominator', 'category', 'n', 'fraction_of_full_population')
BAND_FIELDS = ('population', 'arm', 'axis', 'unit', 'band', 'full_denominator', 'available_n', 'missing_n',
               'band_n', 'band_sum', 'contribution_to_available_mean', 'band_squared_deviation_sum',
               'contribution_to_sample_variance', 'full_available_mean', 'full_available_sample_variance')
CHECK_FIELDS = ('population', 'arm', 'axis', 'full_denominator', 'available_n', 'missing_n', 'sum_band_n',
                'saved_mean', 'reconstructed_mean', 'mean_absolute_residual',
                'saved_sample_variance', 'reconstructed_sample_variance', 'variance_absolute_residual')


def named_arms(count):
    prefixes = ['S1_UNION', 'S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE']
    if count == 40:prefixes += ['S2_C0', 'S2_CP', 'S2_CB', 'F_COMMON', 'F_BI']
    elif count != 25:raise ValueError('Only explicit initial25 or final40 saved-arm scopes are supported')
    seeds = (20261018, 20261019, 20261020)
    return sorted(['N0', 'E0', *['D1_H1_MLP_s'+str(s) for s in seeds],
        'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR',
        *[a+'_s'+str(s) for a in prefixes for s in seeds]])


def number(value):
    if value == '':return None
    result = float(value)
    if not math.isfinite(result) or result < 0:raise ValueError('Finite nonnegative saved error/statistic required')
    return result


def boolean(value):
    if value not in ('True', 'False'):raise ValueError('Explicit saved CSV boolean required')
    return value == 'True'


def csv_rows(path, fields):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):raise ValueError('Public saved-table whitelist/schema differs')
        result = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in result):
        raise ValueError('Malformed public scalar CSV row')
    return result


def authenticate(export_dir, expected_seal_sha256, expected_arm_count):
    root = Path(export_dir);sp = root/'EXPORT_SEAL.json'
    if M.sha(sp) != expected_seal_sha256:raise ValueError('Trusted public exporter seal changed')
    seal = M.read(sp);arms = named_arms(expected_arm_count)
    if (seal.get('schema') != 'PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1' or seal.get('complete') is not True or
            seal['arms'] != arms or seal['populations'] != E.EXPECTED or seal['summary_rows'] != len(arms)*4 or
            seal['scalar_row_rows'] != len(arms)*238 or seal['identities'] != 238 or
            not seal['all_outliers_and_failures_retained'] or seal['metrics_recomputed'] or
            seal['raw_references_decoded'] != 0 or seal['solve_fit_forward_calls'] != 0 or
            seal['source_sha256'] != M.sha(E.__file__) or
            seal['authentication_source_sha256'] != M.sha(E.C.__file__) or seal['io_utility_source_sha256'] != M.sha(M.__file__)):
        raise ValueError('Authenticated whole-population saved-metric exporter contract required')
    names = {'ALL_POPULATION_METRICS.csv', 'ROW_METRICS.csv', 'METRICS_SCHEMA.json', 'INPUT_BINDINGS.json'}
    if set(seal['artifacts']) != names:raise ValueError('Complete public export artifact bundle required')
    for name, digest in seal['artifacts'].items():
        if M.sha(root/name) != digest:raise ValueError('Public scalar artifact changed: '+name)
    schema = M.read(root/'METRICS_SCHEMA.json')
    if (schema['populations'] != E.EXPECTED or schema['sample_variance_ddof'] != 1 or
            schema['sample_std_ddof'] != 1 or schema['summary_columns'] != list(E.ALL_FIELDS) or
            schema['row_columns'] != list(E.ROW_FIELDS)):
        raise ValueError('Saved public metric schema/ddof differs')
    raw = csv_rows(root/'ROW_METRICS.csv', E.ROW_FIELDS);rows = [];identities = {};seen = set()
    for record in raw:
        arm, fid = record['arm'], record['id'];key = arm, fid
        if arm not in arms or key in seen or not fid:raise ValueError('Unknown/duplicate public frame-arm identity')
        seen.add(key)
        identity = {k: record[k] for k in ('id', 'partition', 'session', 'cluster_id')}
        if fid in identities and identities[fid] != identity:raise ValueError('Cross-arm population/recording identity changed')
        identities[fid] = identity
        t, r = number(record['translation_cm']), number(record['rotation_deg'])
        available, success = boolean(record['pose_available']), boolean(record['success'])
        if (available and (t is None or r is None)) or (not available and (t is not None or r is not None or success)):
            raise ValueError('NO_POSE must remain a binary failure with missing continuous errors')
        if success != bool(available and t <= 5 and r <= 5):raise ValueError('Saved success differs from fixed5cm/5deg bits')
        rows.append(dict(identity, arm=arm, translation_cm=t, rotation_deg=r, pose_available=available, success=success))
    if (len(identities) != 238 or len(rows) != 238*len(arms) or
            Counter(r['partition'] for r in identities.values()) != Counter(TEST=49, REMAINDER=143, CAL=46) or
            any({r['id'] for r in rows if r['arm'] == arm} != set(identities) for arm in arms)):
        raise ValueError('Every original238 identity required for every declared arm')
    summaries = {};raw_summary = csv_rows(root/'ALL_POPULATION_METRICS.csv', E.ALL_FIELDS)
    for record in raw_summary:
        key = record['population'], record['arm']
        if key in summaries or key[0] not in E.EXPECTED or key[1] not in arms:raise ValueError('Invalid public summary arm/population')
        summaries[key] = record
    if set(summaries) != {(p, a) for p in E.EXPECTED for a in arms}:raise ValueError('All4 full-population summaries required')
    return arms, rows, summaries, dict(export_seal_sha256=expected_seal_sha256, artifacts=seal['artifacts'],
        producer_source_sha256=seal['source_sha256'], authentication_source_sha256=seal['authentication_source_sha256'])


def joint_category(row):
    if not row['pose_available']:return 'NO_POSE'
    t, r = row['translation_cm'] > 5, row['rotation_deg'] > 5
    return 'BOTH_FAILURE' if t and r else 'TRANSLATION_ONLY_FAILURE' if t else 'ROTATION_ONLY_FAILURE' if r else 'JOINT_SUCCESS'


def band_index(axis, value):
    if value <= 5:return 0
    if axis == 'T':return 1 if value <= 50 else 2 if value <= 100 else 3
    return 1 if value <= 45 else 2


def close(actual, expected, label):
    if (actual is None) != (expected is None) or (actual is not None and not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)):
        raise ValueError('Saved-statistic conservation failed: '+label)
    return None if actual is None else abs(actual-expected)


def budget(values, axis, saved_mean, saved_variance):
    """Contributions use all available observations and the saved full mean."""
    n = len(values);labels = AXES[0 if axis == 'T' else 1][3]
    if any(v is None or not math.isfinite(v) or v < 0 for v in values):raise ValueError('Available saved errors expected')
    mean = math.fsum(values)/n if n else None
    close(mean, saved_mean, axis+'/full mean')
    if (saved_variance is None) != (n < 2):raise ValueError('Undefined sample variance must remain null')
    bands = []
    for i, label in enumerate(labels):
        selected = [v for v in values if band_index(axis, v) == i]
        total = math.fsum(selected)
        squared = math.fsum((v-saved_mean)**2 for v in selected) if n else 0.
        bands.append(dict(band=label, band_n=len(selected), band_sum=total,
            contribution_to_available_mean=total/n if n else None, band_squared_deviation_sum=squared,
            contribution_to_sample_variance=squared/(n-1) if n >= 2 else None))
    reconstructed_mean = math.fsum(b['contribution_to_available_mean'] for b in bands) if n else None
    reconstructed_variance = math.fsum(b['contribution_to_sample_variance'] for b in bands) if n >= 2 else None
    residuals = dict(mean_absolute_residual=close(reconstructed_mean, saved_mean, axis+'/band means'),
                     variance_absolute_residual=close(reconstructed_variance, saved_variance, axis+'/sample variance'))
    return bands, dict(reconstructed_mean=reconstructed_mean, reconstructed_sample_variance=reconstructed_variance, **residuals)


def derive(arms, rows, summaries):
    joint, bands, checks = [], [], []
    for pop, parts in E.POPS.items():
        for arm in arms:
            subset = [r for r in rows if r['arm'] == arm and r['partition'] in parts]
            summary = summaries[(pop, arm)];n = E.EXPECTED[pop]
            counts = Counter(joint_category(r) for r in subset)
            if (len(subset) != n or int(summary['denominator']) != n or
                    counts['JOINT_SUCCESS'] != int(summary['successes']) or counts['NO_POSE'] != int(summary['no_pose'])):
                raise ValueError('Joint saved outcomes/full denominator differ')
            for category in CATEGORIES:
                joint.append(dict(population=pop, arm=arm, full_denominator=n, category=category,
                                  n=counts[category], fraction_of_full_population=counts[category]/n))
            for axis, unit, field, labels in AXES:
                values = [r[field] for r in subset if r['pose_available']]
                available, missing = int(summary[axis+'_n']), int(summary[axis+'_missing'])
                if available != len(values) or missing != counts['NO_POSE'] or available+missing != n:
                    raise ValueError('Available-only continuous denominator differs')
                mean = number(summary[axis+'_mean_'+unit]);variance = number(summary[axis+'_sample_variance_'+unit+'2'])
                result, residuals = budget(values, axis, mean, variance)
                if axis == 'T':
                    if result[2]['band_n']+result[3]['band_n'] != int(summary['T_gt50cm']) or result[3]['band_n'] != int(summary['T_gt100cm']):
                        raise ValueError('Saved translation tails differ')
                elif result[2]['band_n'] != int(summary['R_gt45deg']):raise ValueError('Saved rotation tails differ')
                context = dict(population=pop, arm=arm, axis=axis, full_denominator=n, available_n=available, missing_n=missing)
                bands += [dict(context, unit=unit, **b, full_available_mean=mean, full_available_sample_variance=variance) for b in result]
                checks.append(dict(context, sum_band_n=sum(b['band_n'] for b in result), saved_mean=mean,
                    saved_sample_variance=variance, **residuals))
    return joint, bands, checks


def run(export_dir, expected_seal_sha256, output, *, expected_arm_count=40):
    output = Path(output)
    if output.exists():raise FileExistsError('Error budget requires a fresh output folder')
    arms, rows, summaries, bindings = authenticate(export_dir, expected_seal_sha256, expected_arm_count)
    joint, bands, checks = derive(arms, rows, summaries)
    output.mkdir(parents=True, exist_ok=False)
    E.write_csv(output/'JOINT_CATEGORIES.csv', JOINT_FIELDS, joint)
    E.write_csv(output/'ERROR_BANDS.csv', BAND_FIELDS, bands)
    E.write_csv(output/'CONSERVATION_CHECKS.csv', CHECK_FIELDS, checks)
    M.write(output/'ERROR_BUDGET_SCHEMA.json', dict(schema='POSTHOC_SAVED_SCALAR_ERROR_BUDGET_V1',
        populations=E.EXPECTED, categories=CATEGORIES, fixed_bands={a: labels for a, u, f, labels in AXES},
        boundary='5,50,100cm and5,45deg included in the lower band;success T<=5 AND R<=5',
        mean_contribution='sum(x in band)/N_available;sum over bands equals saved full available mean',
        variance_contribution='sum((x-saved_full_mean)^2 in band)/(N_available-1);sum equals saved sample variance,ddof1',
        units=dict(T=dict(errors_sums_and_mean_contributions='cm', squared_deviations_and_variance_contributions='cm^2'),
                   R=dict(errors_sums_and_mean_contributions='deg', squared_deviations_and_variance_contributions='deg^2')),
        missing='NO_POSE is a full-denominator binary failure;continuous errors excluded only because unavailable,never set to zero',
        undefined='mean contributions null when N=0;sample-variance contributions null when N<2;empty populated-band contributions are0',
        fixed_conservation_tolerance=dict(absolute=ABS_TOL, relative=REL_TOL),
        interpretation='posthoc descriptive accounting,not a new preregistered hypothesis test or selector-performance estimate;bands are not removed',
        populations_overlap='PRIMARY192=TEST49+REMAINDER143;CAL46 separate;seeds distinct arms and not independent frame samples',
        no_outlier_exclusion=True, no_metric_recalculation=True, public_fields='arm/population/category/band counts and scalar contribution statistics only',
        joint_columns=JOINT_FIELDS, band_columns=BAND_FIELDS, check_columns=CHECK_FIELDS))
    M.write(output/'INPUT_BINDINGS.json', bindings)
    names = ['JOINT_CATEGORIES.csv', 'ERROR_BANDS.csv', 'CONSERVATION_CHECKS.csv', 'ERROR_BUDGET_SCHEMA.json', 'INPUT_BINDINGS.json']
    seal = dict(schema='SAVED_SCALAR_ERROR_BUDGET_SEAL_V1', complete=True, arms=arms, populations=E.EXPECTED,
        campaign_scope='ALL40_COMPLETED_ARMS' if expected_arm_count == 40 else 'INITIAL25_COMPLETED_ARMS_ONLY',
        joint_rows=len(joint), band_rows=len(bands), conservation_rows=len(checks), source_sha256=M.sha(__file__),
        scalar_exporter_source_sha256=M.sha(E.__file__), input_export_seal_sha256=expected_seal_sha256,
        maximum_mean_absolute_residual=max((c['mean_absolute_residual'] or 0 for c in checks), default=0),
        maximum_variance_absolute_residual=max((c['variance_absolute_residual'] or 0 for c in checks), default=0),
        fixed_conservation_tolerance=dict(absolute=ABS_TOL, relative=REL_TOL), posthoc_descriptive=True,
        preregistered_test=False, selector_performance_claim=False, no_outlier_exclusion=True,
        raw_reference_reads=0, pose_or_RGB_reads=0, new_pose_metric_calls=0, new_solves=0, new_fits=0,
        independent_real_test_n=0, artifacts={n: M.sha(output/n) for n in names})
    M.write(output/'ERROR_BUDGET_SEAL.json', seal)
    return seal


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('export-dir', 'export-seal-sha256', 'output'):p.add_argument('--'+name, required=True)
    p.add_argument('--expected-arm-count', type=int, choices=(25, 40), default=40)
    a = p.parse_args();print(run(a.export_dir, a.export_seal_sha256, a.output, expected_arm_count=a.expected_arm_count))
