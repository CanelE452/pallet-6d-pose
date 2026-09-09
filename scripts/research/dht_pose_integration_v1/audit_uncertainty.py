"""Independent CPU checks and paired-frame intervals for frozen uncertainty fusion.

This audit imports no experiment scorer or fusion implementation. Stored points
are rescored directly; selected weighted solutions are checked with a separate
least-squares formulation. Only the registered manifest is read, never sealed
final data. Bootstrap intervals condition on the reused real DEV52 images.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EDGES = ((1, 2), (3, 0), (5, 6), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
INCIDENT = tuple(tuple(i for i, edge in enumerate(EDGES) if k in edge) for k in range(8))
FAMILIES = ('point_sigma_only', 'line_uncertainty_only', 'point_line_uncertainty')
CONTROLS = ('previous_joint_gate', 'unconditional_lambda1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def equal(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def check_close(a, b, name, maxima, atol=1e-8, rtol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), name
    valid = np.isfinite(a)
    maximum = float(np.abs(a[valid]-b[valid]).max()) if valid.any() else 0.
    maxima[name] = max(maxima.get(name, 0.), maximum)
    assert np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), (name, maximum)


def segment_in_image(a, b, width, height):
    lower, upper = 0., 1.
    for origin, direction, bound in zip(a, b-a, (width, height)):
        if abs(direction) < 1e-12:
            if not 0 <= origin <= bound:
                return False
        else:
            ends = sorted((-origin/direction, (bound-origin)/direction))
            lower, upper = max(lower, ends[0]), min(upper, ends[1])
    return lower <= upper


def truth_support(record):
    gt = np.asarray(record['gt_points'], float)[:8]
    valid = np.asarray(record['gt_valid'], bool) & np.isfinite(gt).all(-1)
    support = np.array([valid[a] and valid[b] and np.linalg.norm(gt[b]-gt[a]) >= 2.
                        and segment_in_image(gt[a], gt[b], record['width'], record['height'])
                        for a, b in EDGES])
    return gt, valid, support


def metrics(record, points, valid, baseline):
    p = np.asarray(points, float)
    pv = np.asarray(valid, bool) & np.isfinite(p).all(-1)
    gt, gv, support = truth_support(record)
    observed = pv[:8] & gv
    diagonal = np.hypot(record['width'], record['height'])
    error, penalty = np.full(8, np.nan), np.full(8, np.nan)
    error[observed] = np.linalg.norm(p[:8][observed]-gt[observed], axis=1)
    penalty[gv] = diagonal
    penalty[observed] = error[observed]
    line_error, angle, line_penalty = [np.full(8, np.nan) for _ in range(3)]
    line_penalty[support] = diagonal
    line_observed = 0
    for role, (a, b) in enumerate(EDGES):
        vector = p[b]-p[a]
        length = np.linalg.norm(vector)
        if not (support[role] and pv[a] and pv[b] and np.isfinite(length)
                and length > max(1e-9, diagonal*1e-12)):
            continue
        direction = vector/length
        normal = np.array([-direction[1], direction[0]])
        actual_direction = (gt[b]-gt[a])/np.linalg.norm(gt[b]-gt[a])
        cross = actual_direction[0]*direction[1]-actual_direction[1]*direction[0]
        angle[role] = np.rad2deg(np.arctan2(abs(cross), abs(actual_direction@direction)))
        line_error[role] = np.mean(np.abs((gt[[a, b]]-p[a])@normal))
        line_penalty[role] = line_error[role]
        line_observed += 1
    movement = np.linalg.norm(p-np.asarray(baseline, float), axis=1)
    moved = pv[:8] & (movement[:8] > 1e-9)
    return dict(n_gt=int(gv.sum()), n_pred=int(observed.sum()), corner_errors=error,
                corner_penalized=penalty,
                corner_capped_normalized=float(np.minimum(penalty[gv]/diagonal, 1).mean()),
                frame_corner_mean_px=float(error[observed].mean()) if observed.any() else None,
                frame_corner_penalized_px=float(penalty[gv].mean()),
                all8_gt_valid=bool(gv.all()), all8_at_10px=bool((observed & (error <= 10)).all()) if gv.all() else None,
                line_errors=line_error, line_angles=angle, line_penalized=line_penalty,
                line_n_gt=int(support.sum()), line_n_pred=line_observed,
                n_adjusted_corners=int(moved.sum()), n_adjusted_frames=int(moved.any()),
                mean_move_px=float(movement[:8][pv[:8]].mean()) if pv[:8].any() else 0.)


def weighted_lstsq(points, valid, lines, vp, vl, lam, cap):
    result = points.copy()
    if lam == 0:
        return result
    for corner, roles in enumerate(INCIDENT):
        if not valid[corner]:
            continue
        design = [np.diag(1/np.sqrt(vp[corner]))]
        rhs = [np.zeros(2)]
        for j, role in enumerate(roles):
            direction = lines[role, 1]-lines[role, 0]
            normal = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
            scale = np.sqrt(lam/vl[corner, j])
            design.append((scale*normal)[None])
            rhs.append(np.array([-scale*normal@(points[corner]-lines[role, 0])]))
        delta = np.linalg.lstsq(np.vstack(design), np.concatenate(rhs), rcond=None)[0]
        if cap is not None and np.linalg.norm(delta) > cap:
            delta *= cap/np.linalg.norm(delta)
        result[corner] += delta
    return result


def primary_summary(rows):
    error = np.concatenate([r['corner_errors'] for r in rows])
    finite = error[np.isfinite(error)]
    n_gt, n_pred = sum(r['n_gt'] for r in rows), sum(r['n_pred'] for r in rows)
    full = [r['all8_at_10px'] for r in rows if r['all8_gt_valid']]
    return dict(n_frames=len(rows), n_gt_corners=n_gt, n_pred_corners=n_pred,
                corner_mean_px=float(finite.mean()), corner_median_px=float(np.median(finite)),
                corner_p90_px=float(np.percentile(finite, 90)), corner_coverage=n_pred/n_gt,
                **{f'pck_{t}px': float((finite <= t).sum()/n_gt) for t in (5, 10, 20)},
                frame_penalized_mean_px=float(np.mean([r['frame_corner_penalized_px'] for r in rows])),
                capped_normalized_mean=float(np.mean([r['corner_capped_normalized'] for r in rows])),
                all8_10px_success=float(np.mean(full)) if full else None,
                n_adjusted_corners=sum(r['n_adjusted_corners'] for r in rows),
                n_adjusted_frames=sum(r['n_adjusted_frames'] for r in rows),
                mean_move_px=float(np.mean([r['mean_move_px'] for r in rows])))


def auroc(scores, labels):
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return np.nan
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_ranks = (np.cumsum(counts)-(counts-1)/2)[inverse]
    return float((average_ranks[labels].sum()-positives*(positives+1)/2)/(positives*negatives))


def audit(output):
    live = output.parent
    required = ['PROTOCOL.json', 'PREDICTIONS.json', 'FRAME_METRICS.json', 'RESULTS.json',
                'CALIBRATION.json', 'SELECTION.json', 'EVALUATION_AUDIT.json',
                'YOLO_UNCERTAINTY.json', 'DHT_UNCERTAINTY.json']
    assert all((output/name).is_file() for name in required), 'Evaluation outputs not ready'
    paths = [output/name for name in required] + [output/'RELIABILITY.json', live/'manifest.json', live/'BASELINE_YOLO.json',
             live/'fusion_diagnosis_v1/GATED_PREDICTIONS.json', Path(__file__).resolve()]
    before = {str(path): sha(path) for path in paths}
    protocol, predictions, frame_file, results, calibration, selection, evaluation, point_file, line_file = [read(output/name) for name in required]
    assert evaluation['PASS'] and all(v['complete'] for v in
           (predictions, frame_file, results, calibration, selection, evaluation, point_file, line_file))
    for name, digest in protocol['source_sha256'].items():
        assert sha(live/name) == digest, name
    for name, digest in evaluation['output_sha256'].items():
        assert sha(output/name) == digest, name
    for path, digest in results['source_sha256'].items():
        assert sha(path) == digest, path
    records = read(live/'manifest.json')['records']
    assert len(records) == 692 and {r['population'] for r in records} == {'synth_val', 'synth_test', 'cross_v4', 'real_dev'}
    expected = [r['id'] for r in records]
    baseline = read(live/'BASELINE_YOLO.json')['records']
    previous = read(live/'fusion_diagnosis_v1/GATED_PREDICTIONS.json')['models']['YOLO']['records']
    for values in (predictions['records'], baseline, previous, point_file['records'], line_file['records']):
        assert [r['id'] for r in values] == expected
    ordered_val = sorted([r['id'] for r in records if r['population'] == 'synth_val'],
                         key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    assert len(ordered_val) == 256
    assert protocol['calibration_ids'] == ordered_val[:128]
    assert protocol['selection_ids'] == ordered_val[128:]
    assert calibration['ids'] == protocol['calibration_ids'] and selection['ids'] == protocol['selection_ids']
    assert not set(calibration['ids']) & set(selection['ids'])
    assert selection['calibration_sha256'] == sha(output/'CALIBRATION.json')
    for source, functions in [('evaluate_uncertainty.py', ('prepare', 'predict')), ('uncertainty_geometry.py', ('fuse', 'line_variance_at_points'))]:
        tree = ast.parse((HERE/source).read_text())
        for function in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]:
            assert not any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('gt_')
                           for n in ast.walk(function)), (source, function.name)
            assert not any(isinstance(n, ast.Name) and n.id.startswith('gt_') for n in ast.walk(function)), (source, function.name)
    maxima, independently_scored = {}, {}
    rows = {(r['arm'], r['seed'], r['id']): r for r in frame_file['records']}
    assert len(rows) == len(frame_file['records']) == 692*16
    family_index = np.array([0, 0, 0, 0, 1, 1, 1, 1])[np.asarray(INCIDENT)]
    point_ratios, point_errors = [], []
    line_ratios, line_errors = [[], []], [[], []]
    for record, point, line in zip(records, point_file['records'], line_file['records']):
        if record['id'] not in set(calibration['ids']):
            continue
        gt, gv, supported = truth_support(record)
        p = np.asarray(point['kps'], float)[:8]
        valid = np.asarray(point['kp_valid'], bool)[:8] & gv & np.isfinite(p).all(-1)
        sigma = np.full((9, 2), np.nan) if point['sigma_original_px'] is None else np.asarray(point['sigma_original_px'], float)
        error2 = (p[valid]-gt[valid])**2
        point_ratios.extend(error2/np.maximum(sigma[:8][valid]**2, 1.))
        point_errors.extend(error2)
        for seed in protocol['seeds']:
            item = line['seeds'][str(seed)]
            matrices, modes = np.asarray(item['moment_about_mode']), np.asarray(item['h_mode'])
            for role, (a, b) in enumerate(EDGES):
                if supported[role]:
                    z = np.column_stack((gt[[a, b]], np.ones(2)))
                    raw = np.maximum(np.diag(z@matrices[role]@z.T), 1.)
                    residual2 = (z@modes[role])**2
                    line_ratios[role//4].extend(residual2/raw)
                    line_errors[role//4].extend(residual2)
    for key, value in [('point_alpha_xy', np.mean(point_ratios, axis=0)),
                       ('constant_point_variance_xy', np.maximum(np.mean(point_errors, axis=0), 1.)),
                       ('line_beta_height_depth', [np.mean(v) for v in line_ratios]),
                       ('constant_line_variance_height_depth', [max(np.mean(v), 1.) for v in line_errors])]:
        check_close(value, calibration[key], 'calibration_'+key, maxima)
    weighted_checks = 0
    for record, stored, base, old, raw_point, dht in zip(records, predictions['records'], baseline, previous, point_file['records'], line_file['records']):
        p, valid = np.asarray(base['kps'], float), np.asarray(base['kp_valid'], bool)
        assert equal(stored['baseline']['kps'], p) and stored['baseline']['kp_valid'] == valid.tolist()
        assert equal(stored['baseline']['kp_conf'], base['kp_conf'])
        sigma = np.full((9, 2), np.nan) if raw_point['sigma_original_px'] is None else np.asarray(raw_point['sigma_original_px'], float)
        vp_dynamic = np.maximum(np.maximum(sigma**2, 1.)*np.asarray(calibration['point_alpha_xy']), 1.)
        check_close(stored['baseline']['point_variance_px2'], vp_dynamic, 'baseline_point_variance', maxima)
        variants = [('baseline', 0, stored['baseline'])]
        assert set(stored['seeds']) == {str(s) for s in protocol['seeds']}
        for seed, arms in stored['seeds'].items():
            assert set(arms) == set(CONTROLS+FAMILIES)
            variants += [(arm, int(seed), pred) for arm, pred in arms.items()]
        for arm, seed, pred in variants:
            q, qv = np.asarray(pred['kps'], float), np.asarray(pred['kp_valid'], bool)
            assert q.shape == (9, 2) and np.array_equal(qv, valid)
            assert equal(q[~valid], p[~valid]) and equal(q[8], p[8])
            assert np.isfinite(q[valid]).all()
            if arm in CONTROLS:
                old_arm = 'confidence_line_move' if arm == 'previous_joint_gate' else arm
                assert equal(q, old['seeds'][str(seed)][old_arm]['kps'])
            elif arm in FAMILIES:
                vp = np.asarray(pred['point_variance_px2'], float)
                vl = np.asarray(pred['incident_line_variance_px2'], float)
                item = dht['seeds'][str(seed)]
                expected_vp = np.broadcast_to(calibration['constant_point_variance_xy'], (9, 2)) if arm == 'line_uncertainty_only' else vp_dynamic
                if arm == 'point_sigma_only':
                    expected_vl = np.asarray(calibration['constant_line_variance_height_depth'])[family_index]
                else:
                    z = np.column_stack((p[:8], np.ones(8)))
                    matrices = np.asarray(item['moment_about_mode'])
                    raw = np.array([[z[k]@matrices[r]@z[k] for r in roles] for k, roles in enumerate(INCIDENT)])
                    expected_vl = np.maximum(np.maximum(raw, 1.)*np.asarray(calibration['line_beta_height_depth'])[family_index], 1.)
                check_close(vp, expected_vp, 'fusion_point_variance', maxima)
                check_close(vl, expected_vl, 'fusion_line_variance', maxima, atol=1e-7)
                rule = selection['families'][arm]['selected_rule']
                cap = rule['max_move_diagonal_fraction']
                cap = None if cap is None else cap*np.hypot(record['width'], record['height'])
                if cap is not None:
                    assert np.all(np.linalg.norm(q[:8][valid[:8]]-p[:8][valid[:8]], axis=1) <= cap+1e-8)
                if rule['lambda'] == 0:
                    assert equal(q, p)
                independently_fused = weighted_lstsq(p, valid, np.asarray(item['lines']), vp, vl, rule['lambda'], cap)
                check_close(q, independently_fused, 'weighted_lstsq_original_px', maxima)
                weighted_checks += 1
            calculated = metrics(record, q, qv, p)
            saved = rows[(arm, seed, record['id'])]
            assert saved['population'] == record['population'] and saved['group'] == record['group']
            for key, value in calculated.items():
                check_close(value, saved[key], 'metric_'+key, maxima)
            if arm != 'baseline':
                check_close(np.linalg.norm(q-p, axis=1), pred['move_px'], 'stored_move_px', maxima)
            independently_scored[(arm, seed, record['id'])] = calculated
    for summary in results['summaries']:
        chosen = [independently_scored[(summary['arm'], summary['seed'], r['id'])] for r in records
                  if r['population'] == summary['population'] and (summary['group'] == 'all' or r['group'] == summary['group'])]
        for key, value in primary_summary(chosen).items():
            check_close(value, summary[key], 'summary_'+key, maxima)
    for family in FAMILIES:
        item = selection['families'][family]
        best = min(item['grid'], key=lambda r: (r['score'], r['rule']['lambda'],
                   float('inf') if r['rule']['max_move_diagonal_fraction'] is None else r['rule']['max_move_diagonal_fraction']))
        assert item['selected_rule'] == best['rule'] and len(item['grid']) == 10
        score = np.mean([independently_scored[(family, s, i)]['corner_capped_normalized']
                         for s in protocol['seeds'] for i in selection['ids']])
        check_close(score, item['selection_score'], 'selected_score', maxima, atol=1e-12)
    real = [r for r in records if r['population'] == 'real_dev']
    assert len(real) == 52
    groups = sorted({r['group'] for r in real})
    rng = np.random.default_rng(7261)
    draws = np.concatenate([rng.choice([i for i, r in enumerate(real) if r['group'] == group],
                  size=(5000, sum(r['group'] == group for r in real)), replace=True) for group in groups], axis=1)
    intervals = []
    for arm in FAMILIES:
        for reference in ('baseline', 'previous_joint_gate'):
            values = []
            for record in real:
                seed_values = []
                for seed in protocol['seeds']:
                    a = independently_scored[(arm, seed, record['id'])]
                    b = independently_scored[(reference, 0 if reference == 'baseline' else seed, record['id'])]
                    ae, be = a['corner_errors'], b['corner_errors']
                    observed = np.isfinite(ae)
                    assert np.array_equal(observed, np.isfinite(be)) and a['n_gt'] == b['n_gt']
                    seed_values.append([float((ae[observed]-be[observed]).sum()), int(observed.sum()),
                                        int((ae[observed] <= 10).sum()-(be[observed] <= 10).sum()), a['n_gt']])
                values.append(np.mean(seed_values, axis=0))
            values = np.asarray(values)
            sampled, total = values[draws].sum(axis=1), values.sum(axis=0)
            intervals.append(dict(model='YOLO', arm=arm, reference=reference, population='real_dev', group='all',
                n_unique_frames=52, n_observed_corners=int(total[1]), n_gt_corners=int(total[3]),
                mean_observed_corner_error_delta_px=float(total[0]/total[1]),
                mean_error_delta_ci95=np.percentile(sampled[:, 0]/sampled[:, 1], [2.5, 97.5]),
                pck10_delta_percentage_points=float(100*total[2]/total[3]),
                pck10_delta_ci95=np.percentile(100*sampled[:, 2]/sampled[:, 3], [2.5, 97.5])))
    point_by_id = {r['id']: r for r in point_file['records']}
    frame_signals = []
    for record in real:
        point = point_by_id[record['id']]
        gt, gv, _ = truth_support(record)
        p = np.asarray(point['kps'], float)[:8]
        valid = np.asarray(point['kp_valid'], bool)[:8] & gv & np.isfinite(p).all(-1)
        if valid.any():
            error = np.linalg.norm(p[valid]-gt[valid], axis=1)
            low_conf = 1-np.asarray(point['kp_conf'], float)[:8][valid]
            sigma = np.linalg.norm(np.asarray(point['sigma_original_px'], float)[:8][valid], axis=1)
            frame_signals.append(np.column_stack((low_conf, sigma, error > 10)))
        else:
            frame_signals.append(np.empty((0, 3)))
    signals = np.concatenate(frame_signals)
    full_auc = np.array([auroc(signals[:, k], signals[:, 2].astype(bool)) for k in range(2)])
    bootstrap_auc = []
    for draw in draws:
        sample = np.concatenate([frame_signals[i] for i in draw])
        bootstrap_auc.append([auroc(sample[:, k], sample[:, 2].astype(bool)) for k in range(2)])
    bootstrap_auc = np.asarray(bootstrap_auc)
    assert np.isfinite(bootstrap_auc).all(), 'A bootstrap draw has only one error class'
    reliability_rows = read(output/'RELIABILITY.json')['rows']
    for k, signal in enumerate(('low_kp_conf', 'point_sigma_norm')):
        reported = next(r for r in reliability_rows if r['signal'] == signal and r['population'] == 'real_dev' and r['group'] == 'all')
        check_close(full_auc[k], reported['auroc'], 'reliability_auroc', maxima, atol=1e-12)
    reliability_bootstrap = dict(model='YOLO', population='real_dev', group='all', n_unique_frames=52,
        n_observed_corners=len(signals), error_threshold_px=10,
        signals=[dict(signal=signal, auroc=full_auc[k], auroc_ci95=np.percentile(bootstrap_auc[:, k], [2.5, 97.5]))
                 for k, signal in enumerate(('low_kp_conf', 'point_sigma_norm'))],
        difference=dict(minuend='point_sigma_norm', subtrahend='low_kp_conf', auroc_delta=full_auc[1]-full_auc[0],
                        auroc_delta_ci95=np.percentile(bootstrap_auc[:, 1]-bootstrap_auc[:, 0], [2.5, 97.5])),
        interpretation='Higher score predicts larger point error. Point observations are resampled only through paired frames, not independently. These descriptive proxy intervals do not alter calibration, selected rules, or fusion outputs.')
    assert {path: sha(path) for path in before} == before, 'Audit inputs changed'
    independent = dict(schema='dht_uncertainty_independent_audit_v1', complete=True, PASS=True,
        n_frames=692, verified_prediction_rows=len(independently_scored), verified_weighted_solutions=weighted_checks,
        n_summary_rows=len(results['summaries']), calibration_selection_disjoint=True,
        synthetic_sha_ordered_128_128=True, no_real_fit_or_selection=True,
        stored_predictions_rescored_without_production_metric_import=True,
        weighted_fusion_checked_by_independent_lstsq=True, missing_and_centroid_preserved=True,
        historical_controls_bitwise_preserved=True, source_artifact_hashes_unchanged=True,
        max_absolute_differences=maxima, input_sha256=before,
        scope='Numerical and protocol audit PASS, not a statement that accuracy improved. Static review confirms inference consumes predicted coordinates/uncertainties and image dimensions; GT enters calibration128 and separate scoring only.')
    write(output/'INDEPENDENT_AUDIT.json', independent)
    statistical = dict(schema='dht_uncertainty_statistical_audit_v1', complete=True, PASS=True,
        paired_bootstrap=intervals, independent_audit_sha256=sha(output/'INDEPENDENT_AUDIT.json'),
        point_uncertainty_reliability=reliability_bootstrap,
        input_sha256=before, bootstrap=dict(replicates=5000, random_seed=7261,
            groups={g: sum(r['group'] == g for r in real) for g in groups},
            sampling='Paired image resampling within each real DEV group; all three DHT seeds averaged within each original image before resampling. The same paired frame draws serve every comparison.',
            estimator='Difference in observed-corner mean error uses summed within-frame error differences divided by observed corners. PCK10 includes all valid GT corners; missing predictions fail. Three seeds do not create extra independent images.',
            limitations='Conditional descriptive intervals on these 52 reused DEV images. Correlated frames/scenes, model-training uncertainty and multiple comparisons are not fully represented. No independent real final test was accessed. Previous joint gate used all256 synthetic validation images and is a historical reference.'))
    write(output/'STATISTICAL_AUDIT.json', statistical)
    print(json.dumps(clean(dict(PASS=True, verified_prediction_rows=len(independently_scored),
                               verified_weighted_solutions=weighted_checks, paired_bootstrap=intervals)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    audit(parser.parse_args().output_dir.resolve())
