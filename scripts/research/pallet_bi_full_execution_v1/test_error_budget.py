"""Saved scalar fixtures only; no pose/reference/model/image or child calls."""
import csv
from copy import deepcopy
import math
from pathlib import Path
import statistics

import pytest

from scripts.research.pallet_bi_full_execution_v1 import error_budget as B

E, M = B.E, B.M


def row(t, r, available=True):
    return dict(translation_cm=t, rotation_deg=r, pose_available=available,
                success=bool(available and t <= 5 and r <= 5))


def test_inclusive5cm5deg_and_every_joint_failure_category():
    assert [B.joint_category(r) for r in [row(5, 5), row(5+1e-8, 5), row(5, 5+1e-8),
        row(5+1e-8, 5+1e-8), row(None, None, False)]] == list(B.CATEGORIES)
    assert B.band_index('T', 5) == 0 and B.band_index('T', 50) == 1 and B.band_index('T', 100) == 2
    assert B.band_index('T', 50+1e-8) == 2 and B.band_index('T', 100+1e-8) == 3
    assert B.band_index('R', 45) == 1 and B.band_index('R', 45+1e-8) == 2


@pytest.mark.parametrize('axis,values,counts', [
    ('T', [0., 5., 5.+1e-8, 50., 50.+1e-8, 100., 100.+1e-8, 10000.], [2, 2, 2, 2]),
    ('R', [0., 5., 5.+1e-8, 45., 45.+1e-8, 179.], [2, 2, 2])])
def test_available_mean_and_global_sample_variance_conserve_with_all_tails(axis, values, counts):
    mean, variance = statistics.mean(values), statistics.variance(values)
    bands, checks = B.budget(values, axis, mean, variance)
    assert [r['band_n'] for r in bands] == counts
    assert math.isclose(math.fsum(r['band_sum'] for r in bands), math.fsum(values))
    assert math.isclose(math.fsum(r['contribution_to_available_mean'] for r in bands), mean)
    assert math.isclose(math.fsum(r['contribution_to_sample_variance'] for r in bands), variance)
    assert checks['mean_absolute_residual'] <= B.ABS_TOL
    assert checks['variance_absolute_residual'] <= B.ABS_TOL
    if axis == 'T':assert bands[-1]['contribution_to_available_mean'] > .9*mean
    with pytest.raises(ValueError, match='conservation'):
        B.budget(values, axis, mean, variance+10)


def test_zero_and_one_available_never_invent_mean_or_sample_variance():
    bands, checks = B.budget([], 'T', None, None)
    assert all(b['band_n'] == 0 and b['contribution_to_available_mean'] is None and
               b['contribution_to_sample_variance'] is None for b in bands)
    assert checks['reconstructed_mean'] is checks['reconstructed_sample_variance'] is None
    bands, checks = B.budget([50.], 'T', 50., None)
    assert sum(b['contribution_to_available_mean'] for b in bands) == 50
    assert all(b['contribution_to_sample_variance'] is None for b in bands)
    with pytest.raises(ValueError, match='Undefined'):
        B.budget([50.], 'T', 50., 0.)


def fixture(root, count=40):
    arms = B.named_arms(count);rows = []
    samples = [(5., 5.), (5.+1e-8, 5.), (5., 5.+1e-8), (6., 6.),
               (50., 45.), (100., 45.+1e-8), (10000., 179.), (None, None)]
    for i in range(238):
        t, r = samples[i % len(samples)]
        for arm in arms:
            rows.append(dict(id='PRIVATE_ROW_ID_'+str(i), partition='TEST' if i < 49 else 'REMAINDER' if i < 192 else 'CAL',
                session='PRIVATE_SESSION', cluster_id='REC_'+str(i % 6), arm=arm,
                translation_cm=t, rotation_deg=r, pose_available=t is not None,
                success=t is not None and t <= 5 and r <= 5))
    summaries = []
    for pop, parts in E.POPS.items():
        for arm in arms:
            subset = [r for r in rows if r['arm'] == arm and r['partition'] in parts];n = len(subset)
            values = {key: [r[key] for r in subset if r['pose_available']] for key in ('translation_cm', 'rotation_deg')}
            record = {k: '' for k in E.ALL_FIELDS}
            record.update(population=pop, arm=arm, denominator=n, successes=sum(r['success'] for r in subset),
                success_rate=sum(r['success'] for r in subset)/n, no_pose=sum(not r['pose_available'] for r in subset),
                T_gt50cm=sum(v > 50 for v in values['translation_cm']), T_gt100cm=sum(v > 100 for v in values['translation_cm']),
                R_gt45deg=sum(v > 45 for v in values['rotation_deg']), fallback_known_n=0,
                fallback_true_n=0, fallback_false_n=0, fallback_unknown_n=n)
            for axis, unit, key, labels in B.AXES:
                vv = values[key]
                record.update({axis+'_n': len(vv), axis+'_missing': n-len(vv), axis+'_mean_'+unit: statistics.mean(vv),
                    axis+'_sample_variance_'+unit+'2': statistics.variance(vv), axis+'_sample_std_'+unit: statistics.stdev(vv),
                    axis+'_median_'+unit: statistics.median(vv), axis+'_P90_'+unit: max(vv),
                    axis+'_P95_'+unit: max(vv), axis+'_max_'+unit: max(vv)})
            summaries.append(record)
    root.mkdir(parents=True)
    E.write_csv(root/'ROW_METRICS.csv', E.ROW_FIELDS, rows)
    E.write_csv(root/'ALL_POPULATION_METRICS.csv', E.ALL_FIELDS, summaries)
    M.write(root/'METRICS_SCHEMA.json', dict(populations=E.EXPECTED, sample_variance_ddof=1, sample_std_ddof=1,
        summary_columns=list(E.ALL_FIELDS), row_columns=list(E.ROW_FIELDS)))
    M.write(root/'INPUT_BINDINGS.json', dict(constructed_only=True))
    seal = dict(schema='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1', complete=True, arms=arms, populations=E.EXPECTED,
        summary_rows=4*count, scalar_row_rows=238*count, identities=238,
        all_outliers_and_failures_retained=True, metrics_recomputed=False, raw_references_decoded=0, solve_fit_forward_calls=0,
        source_sha256=M.sha(E.__file__), authentication_source_sha256=M.sha(E.C.__file__), io_utility_source_sha256=M.sha(M.__file__),
        artifacts={n: M.sha(root/n) for n in ('ALL_POPULATION_METRICS.csv', 'ROW_METRICS.csv', 'METRICS_SCHEMA.json', 'INPUT_BINDINGS.json')})
    M.write(root/'EXPORT_SEAL.json', seal)
    return M.sha(root/'EXPORT_SEAL.json')


def repin(root):
    seal = M.read(root/'EXPORT_SEAL.json')
    for name in seal['artifacts']:seal['artifacts'][name] = M.sha(root/name)
    (root/'EXPORT_SEAL.json').write_text(__import__('json').dumps(seal, sort_keys=True)+'\n')
    return M.sha(root/'EXPORT_SEAL.json')


def rewrite_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(rows)


def test_full40_export_no_pose_counts_conservation_and_public_whitelist(tmp_path):
    root = tmp_path/'saved';digest = fixture(root);out = tmp_path/'budget'
    seal = B.run(root, digest, out)
    assert seal['complete'] and seal['campaign_scope'] == 'ALL40_COMPLETED_ARMS'
    assert (seal['joint_rows'], seal['band_rows'], seal['conservation_rows']) == (800, 1120, 320)
    assert seal['posthoc_descriptive'] and not seal['preregistered_test'] and not seal['selector_performance_claim']
    joint = B.csv_rows(out/'JOINT_CATEGORIES.csv', B.JOINT_FIELDS)
    for arm in B.named_arms(40):
        subset = [r for r in joint if r['arm'] == arm and r['population'] == 'PRIMARY192']
        assert sum(int(r['n']) for r in subset) == 192
        assert int(next(r['n'] for r in subset if r['category'] == 'NO_POSE')) == 24
    for path in out.iterdir():
        assert 'PRIVATE_ROW_ID' not in path.read_text() and 'PRIVATE_SESSION' not in path.read_text()
    assert seal['maximum_mean_absolute_residual'] < 1e-6
    assert seal['maximum_variance_absolute_residual'] < 1e-6
    with pytest.raises(FileExistsError):B.run('unreadable', 'digest', out)


def test_initial25_scope_is_explicit_and_cannot_impersonate_final40(tmp_path):
    root = tmp_path/'saved';digest = fixture(root, 25)
    with pytest.raises(ValueError, match='exporter contract'):B.run(root, digest, tmp_path/'bad')
    seal = B.run(root, digest, tmp_path/'initial', expected_arm_count=25)
    assert seal['campaign_scope'] == 'INITIAL25_COMPLETED_ARMS_ONLY' and len(seal['arms']) == 25


def test_tampered_csv_and_resealed_wrong_digest_are_rejected(tmp_path):
    root = tmp_path/'saved';digest = fixture(root)
    path = root/'ROW_METRICS.csv';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='artifact changed'):B.run(root, digest, tmp_path/'budget')
    repin(root)
    with pytest.raises(ValueError, match='Trusted'):B.run(root, digest, tmp_path/'budget')
    assert not (tmp_path/'budget').exists()


@pytest.mark.parametrize('change', ['success', 'no_pose', 'mean', 'variance', 'missing_identity'])
def test_saved_scalar_contract_disagreements_rejected_before_output(tmp_path, change):
    root = tmp_path/'saved';fixture(root)
    if change in ('success', 'no_pose', 'missing_identity'):
        path = root/'ROW_METRICS.csv';rows = B.csv_rows(path, E.ROW_FIELDS)
        if change == 'success':rows[0]['success'] = 'False'
        elif change == 'no_pose':
            target = next(r for r in rows if r['pose_available'] == 'False');target['translation_cm'] = '0'
        else:rows.pop()
        rewrite_csv(path, E.ROW_FIELDS, rows)
    else:
        path = root/'ALL_POPULATION_METRICS.csv';rows = B.csv_rows(path, E.ALL_FIELDS)
        key = 'T_mean_cm' if change == 'mean' else 'T_sample_variance_cm2'
        rows[0][key] = str(float(rows[0][key])+100)
        rewrite_csv(path, E.ALL_FIELDS, rows)
    digest = repin(root)
    with pytest.raises(ValueError):B.run(root, digest, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
