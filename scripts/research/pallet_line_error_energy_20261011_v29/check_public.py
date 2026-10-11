"""Independent stdlib reader of all saved line-energy rows and pinned Git inputs."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path

PARENT = '0c210572582057a8e614e899dbf41205f97ec897'
PREFIX = '_docs/experiments/'
INPUTS = {
    'ready': PREFIX+'pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz',
    'decoded': PREFIX+'pallet_source_corner_admission_20261011_v24/decoder/DECODED_FRAMES.jsonl.gz',
    'lines': PREFIX+'pallet_point_admission_mechanism_20261011_v25/source/LINE_ROWS.jsonl.gz',
    'corners': PREFIX+'pallet_point_admission_mechanism_20261011_v25/source/CORNER_ROWS.jsonl.gz',
    'observations': 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py',
    'calibration_code': 'scripts/research/pallet_three_head_observation_20261010_v7/calibration.py',
}
ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
for arm in ARMS:
    INPUTS[arm+'_CALIBRATION'] = PREFIX+'pallet_three_head_observation_20261010_v7/source_calibration/'+arm+'/CALIBRATION.json'


def binding(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read_rows(data):
    return [json.loads(s) for s in gzip.decompress(data).decode().splitlines() if s.strip()]


def near(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a)-set(b), set(b)-set(a))
        for k in a: compare(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y)
    elif isinstance(a, float): near(a, b)
    else: assert a == b, (a, b)


def distribution(values):
    z = sorted(values)
    if not z: return {'n': 0}
    n = len(z)
    m = sum(z)/n
    v = sum(x*x for x in z)/n-m*m
    v = max(0., v)
    def quantile(p):
        pos = (n-1)*p
        i = int(pos)
        return z[i] if i == n-1 else z[i]+(z[i+1]-z[i])*(pos-i)
    return dict(n=n, mean=m, variance=v, std=math.sqrt(v), median=quantile(.5), P90=quantile(.9), max=z[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--checkout', type=Path, required=True)
    a = ap.parse_args()
    d = a.root
    protocol = json.loads((d/'PROTOCOL.json').read_text())
    checks = json.loads((d/'CHECKS.json').read_text())
    summary = json.loads((d/'SUMMARY.json').read_text())
    for name, expected in checks['output_bindings'].items():
        assert binding((d/name).read_bytes()) == expected
    assert binding((d/'audit_saved_line_energy.py').read_bytes()) == {k: protocol['inputs']['code'][k] for k in ('bytes', 'sha256')}
    upstream = {}
    for name, relative in INPUTS.items():
        r = subprocess.run(['git', '-C', str(a.checkout), 'show', PARENT+':'+relative], check=True, capture_output=True)
        assert binding(r.stdout) == {k: protocol['inputs'][name][k] for k in ('bytes', 'sha256')}, name
        upstream[name] = r.stdout
    ready = {r['id']: r for r in read_rows(upstream['ready']) if r['partition'] == 'calibration'}
    decoded = read_rows(upstream['decoded'])
    old_lines = {(r['arm'], r['id'], r['edge']): r for r in read_rows(upstream['lines'])}
    raw_lines = {(f['arm'], f['id'], line['edge']): line for f in decoded for line in f['lines']}
    frame_rows = read_rows((d/'FRAME_ROWS.jsonl.gz').read_bytes())
    actual = read_rows((d/'LINE_ROWS.jsonl.gz').read_bytes())
    lookup = {(r['arm'], r['frame_id'], r['edge']): r for r in actual}
    assert len(ready) == 128 and len(decoded) == len(frame_rows) == 384
    assert len(lookup) == len(actual) == len(raw_lines) == len(old_lines) == 1302
    assert set(lookup) == set(raw_lines) == set(old_lines)
    assert frame_rows == [dict(arm=f['arm'], frame_id=f['id'], line_ids=[r['edge'] for r in f['lines']], new_line_fit=0) for f in decoded]
    pos_count = 0
    for key, row in lookup.items():
        line, previous = raw_lines[key], old_lines[key]
        queries = line['queries']
        points, radii = line['support_points'], line['query_radii_px']
        assert row['support_queries'] == queries == previous['support_queries']
        assert len(queries) == len(set(queries)) == len(points) == len(radii) >= 3
        for field, src in [('predicted_support_xy', 'support_points'), ('query_radius_px', 'query_radii_px'),
                           ('predicted_normal', 'normal'), ('predicted_tangent', 'tangent'),
                           ('predicted_center_xy', 'center'), ('predicted_offset_px', 'offset'),
                           ('recorded_parameter_covariance', 'parameter_covariance')]:
            assert row[field] == line[src]
        n, t, c, off = line['normal'], line['tangent'], line['center'], line['offset']
        near(n[0]*n[0]+n[1]*n[1], 1.)
        near(t[0]*t[0]+t[1]*t[1], 1.)
        near(n[0]*t[0]+n[1]*t[1], 0.)
        w = [max(.5, r)**-2 for r in radii]
        W = sum(w)
        for j in range(2): near(sum(q*p[j] for q, p in zip(w, points))/W, c[j])
        near(n[0]*c[0]+n[1]*c[1], off)
        compare(row['weights'], w)
        tls = [n[0]*p[0]+n[1]*p[1]-off for p in points]
        self2 = sum(q*x*x for q, x in zip(w, tls))/W
        self2u = sum(x*x for x in tls)/len(tls)
        standardized = sum(q*x*x for q, x in zip(w, tls))/len(tls)
        scale = max(1., standardized)
        compare(row['TLS_self_residual_px'], tls)
        near(row['TLS_self_RMS_weighted_px'], math.sqrt(self2))
        near(row['TLS_self_RMS_unweighted_px'], math.sqrt(self2u))
        near(row['TLS_self_RMS_unweighted_px'], previous['predicted_support_rms_px'])
        near(math.sqrt(self2u), line['residual_rms_px'])
        near(row['TLS_standardized_MSE_unweighted'], standardized)
        near(math.sqrt(standardized), line['standardized_residual_rms'])
        near(row['local_line_covariance_scale'], scale)
        along = [t[0]*(p[0]-c[0])+t[1]*(p[1]-c[1]) for p in points]
        A, B = sum(q*s*s for q, s in zip(w, along)), sum(q*s for q, s in zip(w, along))
        det = A*W-B*B
        assert det > 0
        compare(line['parameter_covariance'], [[scale*W/det, -scale*B/det], [-scale*B/det, scale*A/det]])
        consistent = all(abs(x) <= r+1e-9 for x, r in zip(tls, radii))
        assert consistent == row['final_line_consistent'] == previous['final_line_consistent']
        targets = [ready[key[1]]['queries'][q] for q in queries]
        states = dict(collections.Counter(q['proposed_target'] for q in targets))
        assert states == row['source_support_states'] == previous['source_support_states']
        positive = all(q['proposed_target'] == 'POSITIVE' for q in targets)
        assert positive == row['source_certified_all_POS'] == previous['source_certified_all_POS']
        assert row['source_query_anchor_not_native_N3'] and row['physical_line_ownership_on_real'] == 'UNKNOWN'
        assert row['new_line_fit'] is False
        if not positive:
            for field in ('actual_POS_target_xy', 'normal_query_error_px', 'weighted_mean_error_px', 'weighted_actual_normal_MSE_px2', 'mean_component_fraction'):
                assert row[field] is None
            continue
        pos_count += 1
        assert all(q['ownership'] == 'NONCOPLANAR_ACTUAL_SHARED_WIRE' for q in targets)
        gt = [q['actual_target_uv'] for q in targets]
        assert gt == row['actual_POS_target_xy']
        errors = [(p[0]-g[0])*n[0]+(p[1]-g[1])*n[1] for p, g in zip(points, gt)]
        compare(row['normal_query_error_px'], errors)
        mu = sum(q*e for q, e in zip(w, errors))/W
        mse = sum(q*e*e for q, e in zip(w, errors))/W
        # Independent pairwise form of the weighted within-line second moment.
        within = sum(w[i]*w[j]*(errors[i]-errors[j])**2 for i in range(len(w)) for j in range(i+1, len(w)))/(W*W)
        near(mse, mu*mu+within)
        near(row['weighted_mean_error_px'], mu)
        near(mu, -previous['certified_center_bias_px'])
        near(row['weighted_actual_normal_MSE_px2'], mse)
        near(row['weighted_actual_normal_RMS_px'], math.sqrt(mse))
        near(row['actual_RMS_unweighted_px'], math.sqrt(sum(e*e for e in errors)/len(errors)))
        near(row['weighted_within_support_variance_px2'], within)
        near(row['mean_component_energy_px2'], mu*mu)
        if mse == 0: assert row['mean_component_fraction'] is None
        else: near(row['mean_component_fraction'], mu*mu/mse)
        near(row['identity_residual_px2'], mse-mu*mu-within)
        sigma = math.sqrt(line['parameter_covariance'][1][1])
        near(row['recorded_local_offset_sigma_px'], sigma)
        near(sigma, previous['predicted_center_offset_sigma_px'])
        near(row['abs_mean_over_local_offset_sigma'], abs(mu)/sigma)
        assert row['actual_weighted_normal_MSE_exceeds_TLS_weighted_self_MSE'] == (mse > self2)
        assert row['error_energy_not_statistical_covariance']
    assert pos_count == summary['certified_POS_lines'] == 1194 and summary['all_lines'] == 1302
    fields = ('mean_component_fraction', 'weighted_actual_normal_RMS_px', 'TLS_self_RMS_weighted_px',
              'actual_RMS_unweighted_px', 'TLS_self_RMS_unweighted_px', 'weighted_within_support_variance_px2', 'abs_mean_over_local_offset_sigma')
    distribution_cells = 0
    for arm in ARMS:
        rr = [r for r in actual if r['arm'] == arm]
        pos = [r for r in rr if r['source_certified_all_POS']]
        g = summary['groups'][arm]
        assert g['all_lines'] == len(rr) and g['certified_POS_lines'] == len(pos)
        assert g['undefined_POS_error_lines'] == len(rr)-len(pos)
        assert g['local_scale_equal_one_all_lines'] == sum(r['local_line_covariance_scale'] == 1 for r in rr)
        assert g['actual_error_exceeds_self_error_POS_lines'] == sum(r['actual_weighted_normal_MSE_exceeds_TLS_weighted_self_MSE'] for r in pos)
        assert g['zero_actual_error_energy_lines'] == sum(r['weighted_actual_normal_MSE_px2'] == 0 for r in pos)
        for field in fields:
            expected = distribution([r[field] for r in pos if r[field] is not None])
            compare(g[field], expected)
            distribution_cells += len(expected)
        mse = sum(r['weighted_actual_normal_MSE_px2'] for r in pos)
        mean2 = sum(r['mean_component_energy_px2'] for r in pos)
        near(g['sum_equal_line_weighted_actual_MSE_px2'], mse)
        near(g['sum_equal_line_mean_component_energy_px2'], mean2)
        near(g['aggregate_equal_line_mean_energy_fraction'], mean2/mse)
        compare(summary['existing_global_CAL_uncertainty_scales'][arm], json.loads(upstream[arm+'_CALIBRATION'])['uncertainty'])
    old_corners = read_rows(upstream['corners'])
    large = [r for r in old_corners if r['arm'] == 'IMAGE_ROLE' and r['decoder_selected'] and r['saved_corner_error_px'] > 8]
    assert len(large) == len(summary['existing_four_large_corner_cases']) == 4
    for old, saved in zip(large, summary['existing_four_large_corner_cases']):
        assert saved['frame_id'] == old['id'] and saved['corner'] == old['corner']
        near(saved['existing_corner_error_px'], old['saved_corner_error_px'])
        assert saved['existing_point_admission_accepted'] == old['existing_point_admission_accepted']
        compare(saved['line_energy_rows'], [lookup[(old['arm'], old['id'], e)] for e in old['edges']])
    for flag in ('new_line_fit', 'new_PnP', 'new_projection', 'new_pose_accuracy', 'new_training', 'new_RGB'):
        assert summary[flag] == protocol[flag] == 0
    assert not summary['deployment_covariance_changed'] and not summary['accuracy_goal_achieved']
    assert summary['real_covariance_correction_not_validated']
    print(json.dumps(dict(passed=True, upstream_commit=PARENT, all_upstream_Git_blobs_SHA_verified=True,
        frame_rows=384, line_rows=1302, certified_POS_lines=1194, undefined_error_lines=108,
        all_query_joins_errors_and_recorded_covariances_recomputed=True,
        within_variance_recomputed_by_independent_pairwise_identity=True,
        distribution_cells_verified=distribution_cells, large_corner_cases_verified=4,
        new_line_fit=0, new_PnP=0, new_projection=0, new_training=0, new_pose_accuracy=0)))


if __name__ == '__main__': main()
