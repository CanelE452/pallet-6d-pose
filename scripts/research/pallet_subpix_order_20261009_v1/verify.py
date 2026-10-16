"""Independent NumPy verification of saved rows; no model, OpenCV, or F imports."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import subprocess
import numpy as np

WORKTREE = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PALLET_SOURCE_ROOT', str(WORKTREE))).resolve()
DOC = WORKTREE / '_docs/experiments/pallet_subpix_order_20261009_v1'
PRIOR = SOURCE / '_docs/experiments/pallet_n3_subpix_20261008_v1'
ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX', 'SUBPIX_N3')
FIELDS = ('corner_px', 'translation_cm', 'rotation_deg', 'ADDsym_cm')
GRADES = ('clean', 'moderate', 'severe')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def same(a, b, label, tolerance=1e-7):
    assert np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True), label


def distribution(values):
    a = np.asarray(values, np.float64)
    assert a.ndim == 1 and np.isfinite(a).all()
    n = len(a)
    return dict(n=n, mean=float(a.mean()) if n else None,
        sample_variance=float(a.var(ddof=1)) if n >= 2 else None,
        sample_std=float(a.std(ddof=1)) if n >= 2 else None,
        median=float(np.median(a)) if n else None,
        P90=float(np.quantile(a, .9)) if n else None)


def metric_values(records, field):
    if field == 'corner_px':
        return [e for r in records for e in r['corner']['observed_errors']]
    key = 'ADDsym_m' if field == 'ADDsym_cm' else field
    factor = 100 if field == 'ADDsym_cm' else 1
    return [r['pose'][key] * factor for r in records if r['pose']['available']]


def differences(after, before, field):
    if field == 'corner_px':
        mask = np.asarray(after['canonical_observed']) & np.asarray(before['canonical_observed'])
        return np.asarray(after['corner']['canonical_errors'], float)[mask] - np.asarray(before['corner']['canonical_errors'], float)[mask]
    key = 'ADDsym_m' if field == 'ADDsym_cm' else field
    factor = 100 if field == 'ADDsym_cm' else 1
    return np.asarray([(after['pose'][key]-before['pose'][key])*factor] if after['pose']['available'] and before['pose']['available'] else [])


def cap_from_base(base, proposed, cap):
    result = np.array(base, np.float64, copy=True)
    delta = np.asarray(proposed, np.float64)[:8] - result[:8]
    length = np.linalg.norm(delta, axis=-1)
    scale = np.ones(8)
    moving = length > 0
    scale[moving] = np.minimum(1, cap / length[moving])
    result[:8] += delta * scale[:, None]
    return result


def rotation(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def verify(check_user_state=False):
    protocol = read(DOC / 'PROTOCOL.json')
    checks = read(DOC / 'CHECKS.json')
    runtime = read(DOC / 'RUNTIME.json')
    summary = read(DOC / 'SUMMARY.json')
    pair = read(DOC / 'PAIRED_COMPARISON.json')
    prior_rows = rows(PRIOR / 'PREDICTIONS.jsonl.gz')
    reverse_rows = rows(DOC / 'PREDICTIONS.jsonl.gz')
    timed = rows(DOC / 'RUNTIME_ROWS.jsonl.gz')
    assert len(prior_rows) == 1276 and len(reverse_rows) == 319 and len(timed) == 750
    methods = {a: [r for r in prior_rows if r['method'] == a] for a in ARMS[:4]}
    ids = [r['id'] for r in methods['BASE']]
    reverse = {r['id']: r for r in reverse_rows}
    assert len(ids) == len(set(ids)) == len(reverse) == 319 and set(reverse) == set(ids)
    methods['SUBPIX_N3'] = [reverse[i] for i in ids]
    assert all([r['id'] for r in rr] == ids for rr in methods.values())
    sessions = [r['session'] for r in methods['BASE']]
    assert len(set(sessions)) == 13
    grade_path = SOURCE / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json'
    grade_audit = read(grade_path)
    grade_rows = [r for r in grade_audit['rows'] if r['population'] == 'DEV319']
    grades = {r['id']: r for r in grade_rows}
    assert len(grade_rows) == len(grades) == 319 and set(grades) == set(ids)
    grade_counts = dict(Counter(r['severity'] for r in grade_rows))
    assert grade_counts == dict(clean=153, moderate=92, severe=74)
    assert grade_audit['classification_semantics_status'] == 'NOT_CONFIRMED'
    prior_protocol = read(PRIOR / 'PROTOCOL.json')
    inputs = {r['id']: r for r in prior_protocol['input_manifest']}
    grade_bindings = read(DOC / 'GRADE_BINDINGS.json')
    binding_rows = {r['id']: r for r in grade_bindings['frame_bindings']}
    for fid, label in grades.items():
        assert label['session'] == methods['BASE'][ids.index(fid)]['session']
        assert label['image']['sha256'] == inputs[fid]['image_sha256']
        assert sha(SOURCE / label['image']['path']) == label['image']['sha256']
        assert sha(SOURCE / label['annotation']['path']) == label['annotation']['sha256']
        assert binding_rows[fid]['image_sha256'] == label['image']['sha256']
        assert binding_rows[fid]['annotation_sha256'] == label['annotation']['sha256']
        assert binding_rows[fid]['severity'] == label['severity']
    groups = {'ALL': ids, **{g: [i for i in ids if grades[i]['severity'] == g] for g in GRADES}}
    assert set().union(*(set(groups[g]) for g in GRADES)) == set(ids)
    assert sum(len(groups[g]) for g in GRADES) == 319
    assert all(not(set(groups[a]) & set(groups[b])) for a in GRADES for b in GRADES if a < b)
    targets = read(SOURCE / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json')
    pose_rows = Counter()
    max_pose_metric_difference = Counter()
    for arm, records in methods.items():
        for row in records:
            points = np.asarray(row['qFinal'], np.float64)
            target = targets[row['id']]
            gt = np.asarray(target['gt'], np.float64)
            valid = np.asarray(target['valid'], bool)
            valid &= np.isfinite(gt).all(-1) & ~np.all(gt == -1, axis=-1)
            observed_prediction = np.isfinite(points).all(-1) & ~np.all(points == -1, axis=-1) & bool(target['matched'])
            permutations = np.asarray(target['permutations'], int)
            diagonal = np.hypot(*row['raw_hw'])
            errors = []
            for perm in permutations:
                error = np.linalg.norm(points - gt[perm], axis=-1)
                error = np.where(observed_prediction, error, diagonal)
                errors.append(np.where(valid[perm], error, 0))
            errors = np.asarray(errors)
            means = errors[:, :8].sum(-1) / valid[permutations][:, :8].sum(-1)
            branch = int(np.argmin(means)); perm = permutations[branch]
            assert branch == row['corner']['branch']
            canonical = np.full(9, np.nan); mask = valid[perm]
            canonical[perm[mask]] = errors[branch][mask]
            observed = np.zeros(9, bool); observed[perm] = observed_prediction; observed &= valid
            same(canonical[:8], np.asarray(row['corner']['canonical_errors'], float), 'canonical error')
            assert np.array_equal(observed[:8], row['canonical_observed'])
            same(errors[branch, :8][valid[perm][:8]], row['corner']['errors'], 'full corner errors')
            same(errors[branch, :8][valid[perm][:8] & observed_prediction[:8]], row['corner']['observed_errors'], 'observed errors')
            same([means[branch], means[branch]/diagonal, means[0]/diagonal],
                 [row['corner']['frame_mean_px'], row['corner']['E_sym'], row['corner']['E_fixed']], 'corner aggregates')
            meta = inputs[row['id']]
            assert row['fixed_metadata']['selected_index'] == meta['selected_index']
            assert row['fixed_metadata']['candidate_metadata'] == meta['candidate_metadata']
            assert row['fixed_metadata']['preserved']
            assert np.array_equal(points[8], np.asarray(meta['q0'])[8])
            assert row['prediction_support'] == meta['prediction_support']
            actual = row.get('actual_pose')
            if actual is None:
                assert arm in ('BASE', 'SUBPIX') and not row['pose_parameters_saved']
                continue
            pose_rows[arm] += 1
            assert actual['available'] == row['pose']['available']
            if not actual['available']:
                continue
            truth = target['truth']
            R = np.asarray(actual['R_physical']); G = np.asarray(truth['R'])
            t = np.asarray(actual['centroid']); gt_t = np.asarray(truth['t'])
            dims = np.asarray(truth['xyz'])
            corners = np.asarray(list(itertools.product([-1., 1.], repeat=3))) * dims[None]/2
            rotation_errors, add_errors = [], []
            for angle in np.arange(truth['order']) * 2*np.pi/truth['order']:
                target_rotation = G @ rotation(angle)
                relative = target_rotation.T @ R
                rotation_errors.append(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
                add_errors.append(np.linalg.norm(corners@R.T + t - (corners@target_rotation.T + gt_t), axis=-1).mean())
            reconstructed = dict(translation_cm=np.linalg.norm(t-gt_t)*100,
                                 rotation_deg=min(rotation_errors), ADDsym_m=min(add_errors))
            for key, value in reconstructed.items():
                same(value, row['pose'][key], 'actual pose metric '+key)
                max_pose_metric_difference[key] = max(max_pose_metric_difference[key], abs(float(value)-row['pose'][key]))
            same(R.T@R, np.eye(3), 'proper rotation'); same(np.linalg.det(R), 1, 'rotation determinant')
            assert actual['selected_hypothesis'] == row['final_hypothesis'] and t[2] > 0
    assert pose_rows['SUBPIX_N3'] == 319

    distributions, damages = {}, {}
    for group, members in groups.items():
        assert summary['groups'][group]['ids'] == members
        member_set = set(members)
        rr = {a: [r for r in records if r['id'] in member_set] for a, records in methods.items()}
        population = summary['groups'][group]['population']
        assert population['frames'] == len(members)
        assert population['sessions'] == len({r['session'] for r in rr['BASE']})
        assert population['reference_corners'] == sum(len(r['corner']['errors']) for r in rr['BASE'])
        assert population['observed_corners'] == sum(len(r['corner']['observed_errors']) for r in rr['BASE'])
        assert population['matched_frames'] == sum(r['corner']['matched'] for r in rr['BASE'])
        distributions[group] = {}; damages[group] = {}
        for arm, records in rr.items():
            saved = summary['groups'][group]['summaries'][arm]
            distributions[group][arm] = {}
            for field in FIELDS:
                result = distribution(metric_values(records, field))
                for key, value in result.items():
                    same(value, saved['metrics'][field][key], group+'/'+arm+'/'+field+'/'+key, 1e-8)
                distributions[group][arm][field] = result
            meters = distribution([r['pose']['ADDsym_m'] for r in records if r['pose']['available']])
            for key, factor in [('mean', 100), ('sample_std', 100), ('sample_variance', 10000)]:
                same(distributions[group][arm]['ADDsym_cm'][key], meters[key]*factor, 'ADD unit factors', 1e-8)
            full = np.asarray([e for r in records for e in r['corner']['errors']])
            for threshold in (5, 10, 20):
                same((full <= threshold).mean(), saved['corner']['PCK'][str(threshold)], 'PCK')
            assert int((full > 20).sum()) == saved['corner']['gross20_count']
            same((full > 20).mean(), saved['corner']['gross20_rate'], 'gross20 rate')
            assert saved['pose']['available'] == sum(r['pose']['available'] for r in records)
            assert saved['pose']['failures'] == len(records)-saved['pose']['available']
            normalized = np.asarray([r['pose']['ADDsym_normalized'] if r['pose']['available'] else np.inf for r in records])
            thresholds = np.linspace(0, .1, 1001)
            curve = np.asarray([(normalized <= threshold).mean() for threshold in thresholds])
            same(np.trapz(curve, thresholds)/.1, saved['ADDsym_AUC']['full'], 'full ADD AUC')
            for before, damage_key in [('BASE', 'RAW_canonical_damage'), ('N3', 'N3_canonical_damage')]:
                old, new = [], []
                for a, b in zip(rr[before], records):
                    for k, valid in enumerate(a['corner']['canonical_valid']):
                        if valid:
                            old.append(a['corner']['canonical_errors'][k]); new.append(b['corner']['canonical_errors'][k])
                old, new = np.asarray(old), np.asarray(new)
                result = dict(good5_to_bad10=int(((old < 5) & (new > 10)).sum()),
                              bad20_to_good10=int(((old > 20) & (new < 10)).sum()))
                assert all(result[k] == summary['groups'][group][damage_key][arm][k] for k in result)
                damages[group][before+'->'+arm] = result
    units, inverse = np.unique(sessions, return_inverse=True)
    weights = np.random.default_rng(20260917).multinomial(13, np.full(13, 1/13), size=10000).astype(np.uint16)
    draw_sha = hashlib.sha256(weights.tobytes()).hexdigest()
    assert draw_sha == summary['bootstrap']['draw_sha256']
    comparisons = summary['comparisons']
    assert len(comparisons) >= 7
    paired = {}; interval_count = 0
    for name, grouped in comparisons.items():
        after, before = name.split('_minus_'); paired[name] = {}
        for group, members in groups.items():
            member_set = set(members); paired[name][group] = {}
            common = [(a, b) for a, b in zip(methods[after], methods[before]) if a['id'] in member_set]
            quadrants = Counter(); hypotheses = {state: [] for state in ('SWITCH', 'NO_SWITCH', 'UNAVAILABLE')}
            for a, b in common:
                state = 'UNAVAILABLE'
                if a['pose']['available'] and b['pose']['available']:
                    dt = a['pose']['translation_cm']-b['pose']['translation_cm']
                    dr = a['pose']['rotation_deg']-b['pose']['rotation_deg']
                    state = ('BOTH_DECREASE' if dt < 0 and dr < 0 else 'BOTH_INCREASE' if dt > 0 and dr > 0
                        else 'T_DECREASE_R_INCREASE' if dt < 0 and dr > 0 else 'T_INCREASE_R_DECREASE' if dt > 0 and dr < 0
                        else 'ONE_OR_BOTH_EXACTLY_UNCHANGED')
                quadrants[state] += 1
                state = 'UNAVAILABLE'
                if a['pose']['available'] and b['pose']['available'] and a.get('final_hypothesis') and b.get('final_hypothesis'):
                    state = 'SWITCH' if a['final_hypothesis'] != b['final_hypothesis'] else 'NO_SWITCH'
                hypotheses[state].append((a, b))
            assert all(quadrants[k] == v for k, v in grouped[group]['pose_quadrants']['counts'].items())
            for state, selected in hypotheses.items():
                saved_h = grouped[group]['hypothesis']['groups'][state]
                assert saved_h['frames'] == len(selected) and saved_h['ids'] == [a['id'] for a, b in selected]
                for field in FIELDS:
                    values = [v for a, b in selected for v in differences(a, b, field)]
                    for key, value in distribution(values).items():
                        if value is None:
                            assert saved_h['mean_error_change'][field][key] is None
                        else:
                            same(value, saved_h['mean_error_change'][field][key], 'hypothesis distribution', 1e-8)
            for field in FIELDS:
                delta = [[] for _ in units]; contributing_frames = 0
                for i, (a, b) in enumerate(zip(methods[after], methods[before])):
                    if a['id'] not in member_set:
                        continue
                    if field == 'corner_px':
                        mask = np.asarray(a['canonical_observed']) & np.asarray(b['canonical_observed'])
                        values = np.asarray(a['corner']['canonical_errors'], float)[mask] - np.asarray(b['corner']['canonical_errors'], float)[mask]
                    else:
                        key = 'ADDsym_m' if field == 'ADDsym_cm' else field
                        factor = 100 if field == 'ADDsym_cm' else 1
                        values = [(a['pose'][key]-b['pose'][key])*factor] if a['pose']['available'] and b['pose']['available'] else []
                    contributing_frames += bool(len(values)); delta[inverse[i]].extend(values)
                counts = np.asarray([len(v) for v in delta], float)
                totals = np.asarray([np.sum(v) for v in delta], float)
                denominator = weights@counts; keep = denominator > 0
                boot = (weights@totals)[keep]/denominator[keep]
                pooled = np.concatenate(delta)
                mean = float(pooled.mean()); interval = np.quantile(boot, [.025, .975]) if np.count_nonzero(counts) > 1 else None
                saved = grouped[group]['statistics'][field]
                same(mean, saved['mean_paired_difference'], 'paired mean', 1e-8)
                same(interval, saved['CI95'], 'paired interval', 1e-8)
                assert saved['common_eligible_observations'] == len(pooled)
                assert saved['common_eligible_frames'] == contributing_frames
                assert saved['eligible_sessions'] == int(np.count_nonzero(counts))
                assert saved['empty_resamples'] == int((~keep).sum())
                assert saved['valid_resamples'] == int(keep.sum()) and saved['draw_sha256'] == draw_sha
                assert saved == pair['comparisons'][name][group]['statistics'][field]
                marginal = grouped[group]['marginal_change'][field]
                for statistic in ('mean', 'median', 'P90'):
                    a = distributions[group][after][field][statistic]; b = distributions[group][before][field][statistic]
                    saved_change = marginal[statistic]
                    same(a-b, saved_change['after_minus_before'], 'marginal difference')
                    same(b-a, saved_change['absolute_reduction_before_minus_after'], 'absolute reduction')
                    same(100*(b-a)/b, saved_change['relative_reduction_percent'], 'relative reduction')
                paired[name][group][field] = dict(mean=mean, CI95=interval.tolist() if interval is not None else None,
                    observations=len(pooled), contributing_sessions=int(np.count_nonzero(counts)), empty_draws=int((~keep).sum()))
                interval_count += 1

    ranking_count = 0
    for name, metrics in summary['grade_effect_ranking'].items():
        after, before = name.split('_minus_')
        for field, statistics in metrics.items():
            for statistic, saved in statistics.items():
                values = {}
                for g in GRADES:
                    a = distributions[g][after][field][statistic]; b = distributions[g][before][field][statistic]
                    values[g] = dict(final_error=a, absolute_reduction=b-a, relative_reduction_percent=100*(b-a)/b)
                    for key, value in values[g].items():
                        same(value, saved['values'][g][key], 'grade ranking values', 1e-8)
                for key, winner_key, flag, minimum in [('final_error', 'smallest_final_error_grades', 'clean_final_error_smallest', True),
                    ('absolute_reduction', 'largest_absolute_reduction_grades', 'clean_absolute_reduction_largest', False),
                    ('relative_reduction_percent', 'largest_relative_reduction_grades', 'clean_relative_reduction_largest', False)]:
                    best = (min if minimum else max)(v[key] for v in values.values())
                    winners = [g for g in GRADES if values[g][key] == best]
                    assert saved[winner_key] == winners and saved[flag] == ('clean' in winners)
                ranking_count += 1
    with (DOC/'RESULTS.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len({(r['group'], r['method'], r['metric']) for r in csv_rows}) == 80
    for record in csv_rows:
        g, arm, field = record['group'], record['method'], record['metric']
        result = distributions[g][arm][field]
        for key, value in result.items():
            same(float(record[key]), value, 'CSV distribution', 1e-8)
        member_set = set(groups[g]); selected = [r for r in methods[arm] if r['id'] in member_set]
        values = metric_values(selected, field)
        same([float(record['min']), float(record['max'])], [min(values), max(values)], 'CSV bounds')
        assert record['ddof'] == '1' and int(record['group_frames']) == len(selected)
    assert b'\r' not in (DOC/'RESULTS.csv').read_bytes()
    first_stage = read(DOC/'EXISTING_GRADE_SUMMARY.json')
    for g in groups:
        assert first_stage['groups'][g]['ids'] == groups[g]
        for arm in ARMS[:4]:
            assert first_stage['groups'][g]['summaries'][arm] == summary['groups'][g]['summaries'][arm]

    cap_counts = Counter(); sub_calls = sample_calls = head_calls = 0
    changed_input_frames = changed_input_corners = 0; maximum_trace_error = 0.
    normalization = read(SOURCE / '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json')
    identity = []
    internal_rounding = []; final_rounding_only = []
    for row in methods['SUBPIX_N3']:
        q0, qs_native, qs, qsn, final = [np.asarray(row[key], float) for key in ('q0', 'qS_native', 'qS', 'qSN', 'qFinal')]
        fid = row['id']; h, w = row['raw_hw']; cap = .01*np.hypot(h, w)
        assert np.array_equal(q0, methods['BASE'][ids.index(fid)]['qFinal'])
        assert np.array_equal(qs, methods['SUBPIX'][ids.index(fid)]['qFinal'])
        same(cap_from_base(q0, qs_native, cap), qs, 'SUBPIX input cap', 1e-12)
        same(cap_from_base(q0, qsn, cap), final, 'final Base cap', 1e-12)
        assert np.linalg.norm(final[:8]-q0[:8], axis=-1).max() <= cap+1e-10
        internal = np.linalg.norm(qsn[:8]-qs[:8], axis=-1)
        assert internal.max() <= cap+1e-4
        for k in np.flatnonzero(internal > cap+1e-10):
            internal_rounding.append(dict(id=fid, corner=int(k), excess_px=float(internal[k]-cap)))
        before = np.linalg.norm(qsn[:8]-q0[:8], axis=-1)
        d = row['correction']; trace = d['input_trace']
        for key, actual in [('SUBPIX_cap_active8', np.linalg.norm(qs_native[:8]-q0[:8], axis=-1) > cap),
                            ('N3_internal_cap_active8', (np.asarray(d['N3_internal_unclamped_move_px8']) > cap) & np.asarray(trace['output_support'])),
                            ('final_cap_active8', before > cap)]:
            assert np.array_equal(actual, d[key])
            cap_counts[key+'_corners'] += int(actual.sum()); cap_counts[key+'_frames'] += int(actual.any())
        for k in np.flatnonzero((before > cap) & (before-cap <= 1e-4)):
            final_rounding_only.append(dict(id=fid, corner=int(k), excess_px=float(before[k]-cap)))
        for corner in d['SUBPIX']['corner_records']:
            if corner['status'] != 'refined':
                assert np.array_equal(qs_native[corner['corner']], q0[corner['corner']])
        head_calls += trace['head_calls']; sample_calls += trace['sample_calls']; sub_calls += d['algorithm_corner_calls']
        assert trace['head_calls'] == 1 and trace['sample_calls'] == 2
        assert trace['original_shared_features'] and not trace['stale_point_cache_used'] and not trace['GT_inputs']
        assert trace['shared_feature_dtype'] == 'torch.float16' and trace['raw_to_network_applied_once']
        gain = min(640/(h+200), 640/(w+200))
        shape = trace['input_shape']; offset = np.array([round((shape[1]-round((w+200)*gain))/2-.1), round((shape[0]-round((h+200)*gain))/2-.1)])
        assert gain == trace['gain'] and np.array_equal(offset, trace['affine_offset'])
        network = ((qs+100)*gain+offset).astype(np.float32)
        assert np.array_equal(network, trace['branch_input_points'])
        assert hashlib.sha256(network[None].tobytes()).hexdigest() == trace['head_input_points_hash']
        assert trace['input_support_mask'] == np.isfinite(network).all(-1).tolist()
        box = np.asarray(row['fixed_metadata']['candidate_metadata']['box_xyxy'])
        box_net = (((box+100).reshape(2, 2)*gain+offset).reshape(4)).astype(np.float32)
        size = box_net[2:]-box_net[:2]; diag = np.linalg.norm(size).astype(np.float32)
        center = ((box_net[2:]+box_net[:2])*.5).astype(np.float32)
        own = (network[:8]-center)/diag
        error = float(np.max(np.abs(own-np.asarray(trace['box_relative_own']))))
        same(own, trace['box_relative_own'], 'independent box context', 2e-7)
        maximum_trace_error = max(maximum_trace_error, error)
        first_offset = (diag*np.float32(.08/17)).astype(np.float32)
        first_centers = network[:8].copy(); first_centers[:, 0] += first_offset
        same(first_centers, trace['candidate_centers_first'], 'independent candidate center', 1e-4)
        assert trace['candidate_centers_shape'] == [1, 8, 221, 2]
        assert [r['stride'] for r in trace['sample_records']] == [8, 16]
        assert all(r['positions_shape'] == [1, 8, 221, 32, 2] and r['formula_max_abs_error'] == 0 for r in trace['sample_records'])
        assert trace['sample_records'][0]['positions_sha256'] == trace['sample_records'][1]['positions_sha256']
        assert trace['box_context_formula_error'] == 0 and len(trace['logits_sha256']) == 64
        dimensions = np.asarray(targets[fid]['dimensions_WDH'], float)
        logs = np.log(dimensions)
        dimension_features = np.array([logs[0], logs[1], logs[2], logs[0]-logs[1], logs[2]-.5*(logs[0]+logs[1])])
        context = ((dimension_features-np.asarray(normalization['mean']))/np.asarray(normalization['scale'])).astype(np.float32)
        assert np.array_equal(context, trace['context'])
        changed = np.any(qs[:8] != q0[:8], axis=-1)
        assert np.array_equal(changed, d['input_changed_corners8'])
        changed_input_frames += int(changed.any()); changed_input_corners += int(changed.sum())
        if np.array_equal(qs, q0):
            n3 = np.asarray(methods['N3'][ids.index(fid)]['qFinal'])
            assert np.array_equal(qsn, n3)
            same(final, cap_from_base(q0, n3, cap), 'identity final cap', 1e-12)
            identity.append(fid)
    assert len(identity) == 4 and head_calls == 319 and sample_calls == 638 and sub_calls == 2446
    assert set(identity) == set(protocol['parity']['identity_ids'])
    assert cap_counts['SUBPIX_cap_active8_corners'] == 0
    assert cap_counts['N3_internal_cap_active8_corners'] == 75
    assert cap_counts['final_cap_active8_corners'] == 47
    assert checks['execution']['reverse_accuracy_head_calls'] == checks['execution']['reverse_accuracy_F_calls'] == 319
    assert checks['execution']['backbone_accuracy_calls'] == checks['execution']['new_control_parity_F_calls'] == 0
    assert checks['execution']['accuracy_PnP_counts'] == dict(solvePnP=957, solvePnPGeneric=0, solvePnPRefineLM=957)
    repair = read(DOC/'PRECHECK_REPAIR.json')
    assert repair['status'] == 'RESOLVED' and not repair['outcome_based_rerun']
    assert all(repair[k] == 0 for k in ('head_calls', 'F_calls', 'backbone_calls', 'optimizer_updates'))

    timing_stats = {}; timing_counts = Counter(); panel_grades = Counter()
    lookup = {(r['id'], arm): r for arm, records in methods.items() for r in records}
    for arm in ARMS:
        all_arm = [r for r in timed if r['arm'] == arm]
        measured = [r for r in all_arm if r['phase'] == 'measured']
        assert len(all_arm) == 150 and len(measured) == 130
        assert sum(r['phase'] == 'warmup' for r in all_arm) == 20
        assert len({r['id'] for r in measured}) == 26
        timing_stats[arm] = {}
        for group in ('ALL', *GRADES):
            selected = measured if group == 'ALL' else [r for r in measured if r['grade'] == group]
            out = distribution([r['full_ms'] for r in selected]); timing_stats[arm][group] = out
            saved = runtime['summaries'][arm]['full_pipeline'] if group == 'ALL' else runtime['grade_summaries'][group][arm]['full_pipeline']
            for key, stored in [('n', 'n'), ('mean', 'mean_ms'), ('sample_variance', 'sample_variance_ms2'), ('sample_std', 'sample_std_ms'), ('median', 'median_ms'), ('P90', 'p90_ms')]:
                same(out[key], saved[stored], 'latency statistics', 1e-9)
        for r in all_arm:
            cached = lookup[(r['id'], arm)]
            assert np.array_equal(r['corrected_points'], cached['qFinal']) and r['actual_pose'] == cached['actual_pose']
            assert r['grade'] == grades[r['id']]['severity'] and r['RAW_cached_bitexact'] and r['parity_status'] == 'PASS'
            timing_counts.update(r['call_counts'])
    panel_grades.update(grades[r['frame_id']]['severity'] for r in runtime['panel'])
    assert dict(panel_grades) == runtime['grade_panel']['counts'] == dict(clean=12, moderate=8, severe=6)
    assert timing_counts == dict(solvePnP=2250, solvePnPGeneric=0, solvePnPRefineLM=2250, cornerSubPix=3510)
    assert runtime['complete'] and runtime['status'] == 'DONE'
    assert runtime['execution']['pipeline_calls_complete'] == runtime['execution']['final_F_calls_complete'] == 750
    assert runtime['execution']['active_job'] is None and runtime['execution']['intermediate_N3_F_calls'] == 0
    assert runtime['execution_model_forwards']['detector'] == 751 and runtime['execution_model_forwards']['N3'] == 450
    assert runtime['execution_model_forwards']['duplicate_N3_backbone_forwards'] == 0
    assert all(s['quiet'] and not s['foreign_cpu_workloads'] and not s['foreign_gpu_pids'] for s in runtime['environment']['interference_snapshots'])
    verified_bindings = 0
    for binding in runtime['input_bindings']:
        candidates = [WORKTREE/binding['path'], SOURCE/binding['path']]
        if binding['origin'] == 'immutable_baseline_dependency' and '/' not in binding['path']:
            candidates.extend((SOURCE/'scripts').rglob(binding['path']))
        candidates = [p for p in candidates if p.is_file() and sha(p) == binding['sha256']]
        assert candidates, 'Runtime input binding does not match either declared root'
        verified_bindings += 1

    protection = dict(user_state_requested=check_user_state)
    if check_user_state:
        original = protocol['user_changes']
        assert all(sha(SOURCE/path) == digest for path, digest in original['tracked_sha256'].items())
        status_sha = hashlib.sha256(subprocess.check_output(['git', 'status', '--short'], cwd=SOURCE)).hexdigest()
        diff_sha = hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'], cwd=SOURCE)).hexdigest()
        assert status_sha == original['status_sha256'] and diff_sha == original['tracked_diff_sha256']
        protection.update(six_original_user_file_hashes_preserved=True,
                          original_status_sha256=status_sha, original_tracked_diff_sha256=diff_sha)
    leaks = []
    for path in DOC.iterdir():
        if not path.is_file() or path.name == 'VERIFICATION.json':
            continue
        if path.suffix == '.gz':
            with gzip.open(path, 'rt') as stream: text = stream.read()
        else: text = path.read_text()
        if any(token in text for token in ('/home/', '/tmp/', '/dev/shm/')):
            leaks.append(path.name)
    assert not leaks
    public_files = list(DOC.iterdir()) + list(Path(__file__).parent.iterdir())
    assert not any(p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.pt', '.npz', '.pth', '.pdf', '.tex', '.bib') for p in public_files)
    protection.update(public_personal_absolute_path_leaks=leaks, new_RGB_private_annotation_weight_and_manuscript_files=0)
    assert sha(DOC/'PREDICTIONS.jsonl.gz') == checks['raw_rows_sha256']
    assert sha(DOC/'PROTOCOL.json') == checks['protocol_sha256']
    assert sha(DOC/'RUNTIME_ROWS.jsonl.gz') == runtime['raw_rows']['sha256']
    for binding in protocol['code']:
        root = WORKTREE if binding['origin'] == 'experiment' else SOURCE
        assert sha(root/binding['path']) == binding['sha256']
    result = dict(schema='independent_order_and_grade_saved_row_verification_v1', status='PASS',
        implementation='Independent NumPy formulas; no experiment statistics/evaluator, model, OpenCV or F imports',
        execution=dict(new_model_forwards=0, new_final_F_calls=0, new_optimizer_updates=0, new_annotations=0),
        rows=dict(prior_reused=1276, reverse=319, analytical_total=1595, frames_per_route=319, sessions=13,
                  actual_pose_parameters_verified_by_route=dict(pose_rows), runtime_total=750, runtime_measured=650),
        grade=dict(counts=grade_counts, contributing_sessions={g: len({grades[i]['session'] for i in groups[g]}) for g in GRADES},
                   ID_image_annotation_sha_join_verified=319, disjoint_grade_partition=True, source_semantics='NOT_CONFIRMED'),
        metrics=dict(all1595_2D_errors_recomputed=True, actual_pose_metric_max_abs_difference=dict(max_pose_metric_difference),
                     distributions_verified=80, independent_distributions=distributions, canonical_damage=damages,
                     CSV_rows_verified=80, first_stage_existing_four_route_summaries_unchanged=True,
                     grade_metric_statistic_rankings_verified=ranking_count, all_comparison_hypothesis_distributions_and_pose_quadrants_verified=True,
                     ADDsym_mean_std_variance_unit_factors=[100, 100, 10000]),
        paired=dict(comparisons=len(comparisons), metric_group_intervals=interval_count, draw_sha256=draw_sha,
                    resamples=10000, seed=20260917, original13_session_draws_before_grade_mask=True,
                    original_corner_frame_pooling=True, independent_results=paired),
        reverse=dict(head_calls=head_calls, actual_candidate_sample_calls=sample_calls, identity4_native_and_capped_exact=True,
                     changed_input_frames=changed_input_frames, changed_input_corners=changed_input_corners,
                     affine_candidate_box_context_recomputed=True, maximum_independent_box_context_error=maximum_trace_error,
                     preserved_shared_FP16_features=True, SUBPIX_corner_calls=sub_calls, cap_counts=dict(cap_counts),
                     original_N3_float32_input_cap_excesses=internal_rounding, final_cap_rounding_only_excesses=final_rounding_only,
                     pre_forward_checkpoint_loader_repair=repair),
        runtime=dict(independent_statistics=timing_stats, input_bindings_verified=verified_bindings,
                     final_coordinates_and_pose_all750_exact=True, measured_panel_grade_counts=dict(panel_grades),
                     call_counts=dict(timing_counts), interference_snapshots_all_quiet=True,
                     new_final_F_total=1069, new_N3_heads_total=769, detector_explicit=750, detector_initialization=1,
                     preserved_receipt_notes=['Nested execution.status STARTED is the final running journal snapshot; outer DONE, all750 completed and active_job=None prove completion.',
                         'Some inherited input origin labels say experiment, and six baseline code entries use basenames; bindings were resolved to actual source/worktree files and verified by SHA256.',
                         'A raw-only reporting pass stopped on a provenance dictionary alias assertion before writing; copying the dictionary resolved bookkeeping without changing rows, statistics, model calls or F calls.']),
        protection=protection,
        artifact_sha256={p.name: sha(p) for p in DOC.iterdir() if p.is_file() and p.name != 'VERIFICATION.json'},
        code_sha256={p.name: sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()})
    destination = DOC/'VERIFICATION.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', distributions=80, paired_metric_groups=interval_count,
        reverse_rows=319, runtime_rows=750, final_F_total=1069, verification_sha256=sha(destination)), ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-user-state', action='store_true', help='Check the source workspace against the private starting-state hashes sealed in PROTOCOL.')
    verify(parser.parse_args().check_user_state)
