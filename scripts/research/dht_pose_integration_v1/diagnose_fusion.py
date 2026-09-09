"""Post-hoc geometric diagnosis of frozen corner/line fusion, without training.

Difficulty, perfect lines, and per-corner best-of predictions use GT exclusively
for diagnosis. They are not deployable gates or new selected evaluation arms.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

import evaluate as E
import fusion as F

HERE = Path(__file__).resolve().parent
TARGET_PATH = HERE.parent / 'deep_hough_side_v1/targets.py'
spec = importlib.util.spec_from_file_location('frozen_dht_targets', TARGET_PATH)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
CONDITIONS = ('fixed1', 'selected', 'perfect_lines_fixed1',
              'perfect_lines_selected', 'oracle_min_baseline_fixed1')
DIFFICULTIES = ('all', 'easy_le10', 'moderate_10_20', 'hard_gt20', 'missing', 'unknown_gt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = np.asarray(values, float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else None


def percentile(values, q):
    values = np.asarray(values, float)
    return float(np.percentile(values[np.isfinite(values)], q)) if np.isfinite(values).any() else None


def aggregate_seeds(per_seed):
    keys = [k for k in per_seed[0] if k != 'seed']
    result = {}
    for key in keys:
        values = [r[key] for r in per_seed if r[key] is not None]
        result[key] = {'mean': mean(values), 'min': min(values), 'max': max(values)} if values else None
    return result


def normal_geometry(lines):
    lines = np.asarray(lines, float)
    direction = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(direction, axis=-1)
    valid = np.isfinite(lines).all(axis=(1, 2)) & (lengths > 1e-9)
    tangent = np.full((8, 2), np.nan)
    tangent[valid] = direction[valid] / lengths[valid, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    return normal, tangent, valid


def error_for(row, condition):
    if condition in ('fixed1', 'selected'):
        return row[condition]['error_px']
    return row['oracle_gt_only'][condition + '_error_px']


def summarize_corners(rows, condition, seeds):
    per_seed = []
    for seed in seeds:
        subset = [r for r in rows if r['seed'] == seed]
        available = [r for r in subset if r['gt_valid'] and r['baseline_valid']]
        e = np.array([r['baseline_error_px'] for r in available], float)
        f = np.array([error_for(r, condition) for r in available], float)
        delta = f - e
        entry = dict(seed=seed, n_frames=len({r['id'] for r in subset}),
                     n_gt_corners=sum(r['gt_valid'] for r in subset),
                     n_observed_corners=len(available),
                     baseline_mean_px=mean(e), baseline_median_px=percentile(e, 50),
                     fused_mean_px=mean(f), fused_median_px=percentile(f, 50),
                     delta_mean_px=mean(delta), delta_median_px=percentile(delta, 50),
                     improved_fraction=mean(delta < -1e-9), worsened_fraction=mean(delta > 1e-9),
                     unchanged_fraction=mean(np.abs(delta) <= 1e-9),
                     mean_gain_when_improved_px=mean(-delta[delta < -1e-9]),
                     mean_loss_when_worsened_px=mean(delta[delta > 1e-9]),
                     loss_p90_when_worsened_px=percentile(delta[delta > 1e-9], 90),
                     delta_squared_mean_px2=mean(f * f - e * e))
        if condition in ('fixed1', 'selected'):
            movements = [r[condition] for r in available]
            moving = [r for r in movements if r['movement_px'] > 1e-8]
            worse = [r for r in movements if r['worsened']]
            entry.update(movement_mean_px=mean([r['movement_px'] for r in movements]),
                movement_p90_px=percentile([r['movement_px'] for r in movements], 90),
                linear_term_mean_px2=mean([r['linear_term_px2'] for r in movements]),
                movement_squared_mean_px2=mean([r['movement_squared_px2'] for r in movements]),
                moving_fraction=mean([r['movement_px'] > 1e-8 for r in movements]),
                wrong_direction_fraction_of_moving=mean([r['wrong_direction'] for r in moving]),
                wrong_direction_fraction_of_worsened=mean([r['wrong_direction'] for r in worse]),
                overshoot_fraction_of_worsened=mean([r['overshoot'] for r in worse]))
        per_seed.append(entry)
    return {'metrics': aggregate_seeds(per_seed), 'per_seed': per_seed}


def summarize_lines(rows, seeds):
    per_seed = []
    for seed in seeds:
        subset = [r for r in rows if r['seed'] == seed]
        observed = [r for r in subset if r['distance_px'] is not None]
        d = [r['distance_px'] for r in observed]
        a = [r['angle_deg'] for r in observed]
        per_seed.append(dict(seed=seed, n_gt_lines=len(subset), n_predicted_lines=len(observed),
            distance_mean_px=mean(d), distance_median_px=percentile(d, 50), distance_p90_px=percentile(d, 90),
            angle_mean_deg=mean(a), angle_median_deg=percentile(a, 50),
            success_fraction=(sum(r['distance_px'] <= 8 and r['angle_deg'] <= 5 for r in observed) / len(subset)) if subset else None))
    return {'metrics': aggregate_seeds(per_seed), 'per_seed': per_seed}


def frame_paired(rows, condition, seeds):
    by_frame = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_frame[r['id']][r['seed']].append(r)
    frame_values, per_seed_values = [], defaultdict(list)
    for frame_id, seed_rows in sorted(by_frame.items()):
        values = []
        for seed in seeds:
            subset = seed_rows[seed]
            n_gt = sum(r['gt_valid'] for r in subset)
            # Missing corners are unchanged, so their equal diagonal penalties cancel.
            delta = sum(error_for(r, condition) - r['baseline_error_px'] for r in subset
                        if r['gt_valid'] and r['baseline_valid']) / n_gt
            values.append(delta)
            per_seed_values[seed].append(delta)
        frame_values.append(float(np.mean(values)))
    values = np.asarray(frame_values)
    rng = np.random.default_rng(20260907)
    bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
    return dict(n_independent_frames=len(values), n_seeds=len(seeds),
                mean_delta_px=float(values.mean()), median_delta_px=float(np.median(values)),
                mean_delta_frame_bootstrap95_px=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                improved_frames=int((values < -1e-9).sum()), worsened_frames=int((values > 1e-9).sum()),
                unchanged_frames=int((np.abs(values) <= 1e-9).sum()),
                per_seed_mean_delta_px={str(k): mean(v) for k, v in per_seed_values.items()},
                frame_ids=sorted(by_frame), frame_seed_mean_delta_px=frame_values)


def quantization_reference(records, seeds, dht):
    grids = np.array([(np.asarray(r['gt_points'], float) + 100) *
                      [50 / (r['width'] + 200), 50 / (r['height'] + 200)] for r in records])
    theta, rho, support = T.make_targets(records, grids, pad=100)
    rounded_theta = np.floor(theta.astype(float) + .5).astype(int)
    seam = rounded_theta == 180
    rounded_rho = np.where(seam, -rho, rho)
    raw_rho_index = np.floor((rounded_rho + 35) / .5 + .5).astype(int)
    quant_rho = np.clip(raw_rho_index, 0, 140) * .5 - 35
    quant_theta = rounded_theta % 180
    groups = defaultdict(list)
    maximum_roundtrip_distance, maximum_roundtrip_angle = 0.0, 0.0
    for i, r in enumerate(records):
        exact = T.line_pixels(theta[i], rho[i], r['width'], r['height'])
        qa, qd = T.pixel_errors(exact, r['gt_points'])
        if support[i].any():
            maximum_roundtrip_distance = max(maximum_roundtrip_distance, float(qd[support[i]].max()))
            maximum_roundtrip_angle = max(maximum_roundtrip_angle, float(qa[support[i]].max()))
        quantized = T.line_pixels(quant_theta[i], quant_rho[i], r['width'], r['height'])
        angle, distance = T.pixel_errors(quantized, r['gt_points'])
        actual = {seed: T.pixel_errors(dht[i]['seeds'][str(seed)]['lines'], r['gt_points']) for seed in seeds}
        for group in ['all'] + ([r['group']] if r['population'] == 'real_dev' else []):
            for role in np.flatnonzero(support[i]):
                family = 'height' if role < 4 else 'depth'
                item = dict(id=r['id'], role=int(role), quant_distance_px=float(distance[role]),
                            quant_angle_deg=float(angle[role]), dht={str(seed):
                            dict(distance_px=float(actual[seed][1][role]), angle_deg=float(actual[seed][0][role])) for seed in seeds})
                groups[(r['population'], group, 'all')].append(item)
                groups[(r['population'], group, family)].append(item)
    summaries = []
    for (population, group, family), rows in sorted(groups.items()):
        d, a = [r['quant_distance_px'] for r in rows], [r['quant_angle_deg'] for r in rows]
        per_seed = []
        for seed in seeds:
            learned = [r['dht'][str(seed)]['distance_px'] for r in rows]
            per_seed.append(dict(seed=seed, distance_mean_px=mean(learned), distance_median_px=percentile(learned, 50),
                distance_p90_px=percentile(learned, 90), distance_above_quantized_fraction=mean(np.array(learned) > np.array(d))))
        summaries.append(dict(population=population, group=group, role_family=family,
            n_frames=len({r['id'] for r in rows}), n_supported_lines=len(rows),
            quantized_gt=dict(distance_mean_px=mean(d), distance_median_px=percentile(d, 50),
                distance_p90_px=percentile(d, 90), angle_mean_deg=mean(a), angle_p90_deg=percentile(a, 90)),
            dht_learned=dict(metrics=aggregate_seeds(per_seed), per_seed=per_seed)))
    audit = dict(theta_bin_deg=1, rho_bin_feature_pixels=.5, grid_size=50, pad=100,
        supported_seam_wrap_count=int((seam & support).sum()),
        supported_rho_clipped_count=int((((raw_rho_index < 0) | (raw_rho_index > 140)) & support).sum()),
        exact_target_roundtrip_max_distance_px=maximum_roundtrip_distance,
        exact_target_roundtrip_max_angle_deg=maximum_roundtrip_angle,
        rho_one_bin_original_perpendicular_px_range_640x480=[6.8, 8.4],
        definition='Nearest parameter bins, theta180 wrap flips rho; GT-only reference, not causal error attribution')
    assert maximum_roundtrip_distance < 1e-3 and maximum_roundtrip_angle < 1e-3
    # Explicit seam test: theta179.8,rho7 -> theta0,rho-7.
    assert (int(np.floor(179.8 + .5)) % 180, -7) == (0, -7)
    return summaries, audit


def diagnose(run_dir):
    run_dir = Path(run_dir).resolve()
    output = run_dir / 'fusion_diagnosis_v1'
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'PURPOSE.md').exists():
        raise ValueError('PURPOSE.md must exist before diagnosis')
    cfg, manifest, variants, dht = E.inputs(run_dir)
    source_names = ['CONFIG.json', 'manifest.json', 'DHT_LINES.json', 'BASELINE_DOPE.json',
                    'BASELINE_YOLO.json', 'SELECTION.json', 'FUSED_PREDICTIONS.json', 'RESULTS.json']
    source_hashes = {name: sha(run_dir / name) for name in source_names}
    selection = E.read(run_dir / 'SELECTION.json')
    saved_fused = E.read(run_dir / 'FUSED_PREDICTIONS.json')['models']
    records, seeds = manifest['records'], cfg['dht']['seeds']
    assert len(records) == 692 and len(manifest['populations']['real_dev']) == 52 and seeds == [1, 2, 3]
    assert tuple(F.SIDE_EDGES) == tuple(T.SIDE_EDGES)
    corners, corner_groups, line_groups = [], defaultdict(list), defaultdict(list)
    max_decomposition_error = 0.0
    max_perfect_increase = 0.0
    max_saved_prediction_error = 0.0
    unchanged_centers, missing_preserved = 0, 0
    for model, baselines in variants.items():
        selected_lambda = selection['models'][model]['lambda']
        for i, (r, base) in enumerate(zip(records, baselines)):
            gt = np.asarray(r['gt_points'], float)[:8]
            gt_valid = np.asarray(r['gt_valid'], bool)[:8] & np.isfinite(gt).all(-1)
            p = np.asarray(base['kps'], float)
            p_valid = np.asarray(base['kp_valid'], bool) & np.isfinite(p).all(-1)
            gt_lines = gt[np.asarray(F.SIDE_EDGES)].copy()
            gt_lines[~np.array([gt_valid[a] and gt_valid[b] for a, b in F.SIDE_EDGES])] = np.nan
            for seed in seeds:
                lines = np.asarray(dht[i]['seeds'][str(seed)]['lines'], float)
                normal, tangent, line_valid = normal_geometry(lines)
                outputs, oracle = {}, {}
                for condition, lam in [('fixed1', 1), ('selected', selected_lambda)]:
                    q, valid = F.fuse_corners(p, p_valid, lines, lam, r['width'], r['height'])
                    perfect, perfect_valid = F.fuse_corners(p, p_valid, gt_lines, lam, r['width'], r['height'])
                    outputs[condition], oracle[condition] = q, perfect
                    np.testing.assert_equal(valid, p_valid)
                    np.testing.assert_equal(perfect_valid, p_valid)
                    np.testing.assert_equal(q[8], p[8])
                    np.testing.assert_equal(q[~p_valid], p[~p_valid])
                    unchanged_centers += 1
                    missing_preserved += int((~p_valid[:8]).sum())
                    saved = np.asarray(saved_fused[model]['records'][i]['seeds'][str(seed)][condition]['kps'], float)
                    np.testing.assert_allclose(q, saved, atol=1e-9, rtol=0, equal_nan=True)
                    finite = np.isfinite(saved)
                    if finite.any():
                        max_saved_prediction_error = max(max_saved_prediction_error, float(np.abs(q[finite] - saved[finite]).max()))
                for k, roles in enumerate(F.INCIDENT_ROLES):
                    row = dict(id=r['id'], index=i, population=r['population'], group=r['group'], model=model,
                        seed=seed, corner=k, gt_valid=bool(gt_valid[k]), baseline_valid=bool(p_valid[k]),
                        baseline_xy=p[k].tolist(), gt_xy=gt[k].tolist(), baseline_error_px=None,
                        difficulty='unknown_gt' if not gt_valid[k] else 'missing', incident_roles=list(roles),
                        incident_line_gt_distance_px=None, incident_baseline_residual_px=None,
                        incident_normal_error_px=None, incident_tangent_error_px=None,
                        incident_sin_angle=None, fixed1=None, selected=None, oracle_gt_only={})
                    n, t, anchors = normal[list(roles)], tangent[list(roles)], lines[list(roles), 0]
                    if line_valid[list(roles)].all():
                        row['incident_sin_angle'] = float(abs(np.linalg.det(n)))
                    if gt_valid[k]:
                        row['incident_line_gt_distance_px'] = np.abs(np.sum(n * (gt[k] - anchors), -1)).tolist()
                    if gt_valid[k] and p_valid[k]:
                        e = p[k] - gt[k]
                        baseline_error = float(np.linalg.norm(e))
                        row['baseline_error_px'] = baseline_error
                        row['difficulty'] = 'easy_le10' if baseline_error <= 10 else 'moderate_10_20' if baseline_error <= 20 else 'hard_gt20'
                        row['incident_baseline_residual_px'] = np.abs(np.sum(n * (p[k] - anchors), -1)).tolist()
                        row['incident_normal_error_px'] = (n @ e).tolist()
                        row['incident_tangent_error_px'] = (t @ e).tolist()
                    for condition in ('fixed1', 'selected'):
                        q = outputs[condition][k]
                        detail = dict(xy=q.tolist(), error_px=None, delta_error_px=None, movement_px=None,
                            linear_term_px2=None, movement_squared_px2=None, delta_squared_px2=None,
                            dot_e_d_px2=None, wrong_direction=None, worsened=None, improved=None, overshoot=None,
                            incident_normal_movement_px=None, incident_tangent_movement_px=None)
                        if gt_valid[k] and p_valid[k]:
                            d = q - p[k]
                            error = float(np.linalg.norm(q - gt[k]))
                            dot, d2 = float(np.dot(e, d)), float(np.dot(d, d))
                            squared = error * error - baseline_error * baseline_error
                            max_decomposition_error = max(max_decomposition_error, abs(squared - (2 * dot + d2)))
                            detail.update(error_px=error, delta_error_px=error-baseline_error,
                                movement_px=float(np.linalg.norm(d)), linear_term_px2=2*dot,
                                movement_squared_px2=d2, delta_squared_px2=squared, dot_e_d_px2=dot,
                                wrong_direction=dot > 1e-9, worsened=error > baseline_error + 1e-9,
                                improved=error < baseline_error - 1e-9,
                                overshoot=dot < -1e-9 and squared > 1e-9,
                                incident_normal_movement_px=(n @ d).tolist(),
                                incident_tangent_movement_px=(t @ d).tolist())
                            perfect_error = float(np.linalg.norm(oracle[condition][k] - gt[k]))
                            max_perfect_increase = max(max_perfect_increase, perfect_error - baseline_error)
                            row['oracle_gt_only']['perfect_lines_' + condition + '_error_px'] = perfect_error
                        else:
                            row['oracle_gt_only']['perfect_lines_' + condition + '_error_px'] = None
                        row[condition] = detail
                    row['oracle_gt_only']['oracle_min_baseline_fixed1_error_px'] = (
                        min(row['baseline_error_px'], row['fixed1']['error_px']) if row['baseline_error_px'] is not None else None)
                    corners.append(row)
                    for group in ['all'] + ([r['group']] if r['population'] == 'real_dev' else []):
                        corner_groups[(model, r['population'], group, 'all')].append(row)
                        corner_groups[(model, r['population'], group, row['difficulty'])].append(row)
                for condition, point in [('baseline', p), ('fixed1', outputs['fixed1']), ('selected', outputs['selected']),
                                         ('dht_only', None), ('dht_matched_baseline', None)]:
                    derived, derived_valid = (lines, line_valid) if point is None else F.corners_to_lines(point, p_valid)
                    matched_support = F.corners_to_lines(p, p_valid)[1] if condition == 'dht_matched_baseline' else None
                    lm = F.line_metrics(derived, derived_valid, gt, gt_valid, r['width'], r['height'], gt_support=matched_support)
                    for role in np.flatnonzero(lm['gt_support']):
                        item = dict(seed=seed, id=r['id'], distance_px=float(lm['distance_px'][role]) if lm['predicted_on_gt'][role] else None,
                                    angle_deg=float(lm['angle_deg'][role]) if lm['predicted_on_gt'][role] else None)
                        for group in ['all'] + ([r['group']] if r['population'] == 'real_dev' else []):
                            for family in ('all', 'height' if role < 4 else 'depth'):
                                line_groups[(model, condition, r['population'], group, family)].append(item)
        print(f'Diagnosed original and fixed/selected/oracle coordinates: {model}', flush=True)
    assert len(corners) == 692 * 3 * 3 * 8
    assert max_decomposition_error < 1e-5 and max_perfect_increase < 1e-7
    corner_summary, frame_summary, constraints, parallel = [], [], [], []
    for (model, population, group, difficulty), rows in sorted(corner_groups.items()):
        for condition in CONDITIONS:
            corner_summary.append(dict(model=model, population=population, group=group, difficulty=difficulty,
                condition=condition, **summarize_corners(rows, condition, seeds)))
        if difficulty == 'all':
            for condition in CONDITIONS:
                frame_summary.append(dict(model=model, population=population, group=group, condition=condition,
                                          **frame_paired(rows, condition, seeds)))
        for family, j in [('height', 0), ('depth', 1)]:
            per_seed = []
            for seed in seeds:
                available = [r for r in rows if r['seed'] == seed and r['gt_valid'] and r['baseline_valid']]
                bias = np.array([r['incident_line_gt_distance_px'][j] for r in available])
                normal_error = np.abs([r['incident_normal_error_px'][j] for r in available])
                tangent_error = np.abs([r['incident_tangent_error_px'][j] for r in available])
                per_seed.append(dict(seed=seed, n_observed_corners=len(available),
                    line_gt_distance_mean_px=mean(bias), line_gt_distance_median_px=percentile(bias, 50),
                    baseline_abs_normal_error_mean_px=mean(normal_error), baseline_abs_tangent_error_mean_px=mean(tangent_error),
                    line_closer_than_baseline_in_normal_fraction=mean(bias < normal_error)))
            constraints.append(dict(model=model, population=population, group=group, difficulty=difficulty,
                                    role_family=family, metrics=aggregate_seeds(per_seed), per_seed=per_seed))
        if difficulty == 'all':
            for label, lo, hi in [('angle_lt5', 0, np.sin(np.deg2rad(5))),
                                  ('angle_5_15', np.sin(np.deg2rad(5)), np.sin(np.deg2rad(15))),
                                  ('angle_ge15', np.sin(np.deg2rad(15)), 1.00000001)]:
                subset = [r for r in rows if r['incident_sin_angle'] is not None and lo <= r['incident_sin_angle'] < hi]
                if subset:
                    parallel.append(dict(model=model, population=population, group=group, incident_angle_bin=label,
                                         condition='fixed1', **summarize_corners(subset, 'fixed1', seeds)))
    line_summary = [dict(model=k[0], condition=k[1], population=k[2], group=k[3], role_family=k[4],
                         **summarize_lines(rows, seeds)) for k, rows in sorted(line_groups.items())]
    quantized, quant_audit = quantization_reference(records, seeds, dht)
    report = dict(schema='dht_fusion_posthoc_diagnosis_v1', complete=True,
        scope='Frozen actual-image batch1 predictions; descriptive GT-based geometry diagnosis, no training or selection changes',
        source_sha256=source_hashes, models=list(variants), seeds=seeds,
        population_frames={k: len(v) for k, v in manifest['populations'].items()},
        selected_lambda={k: v['lambda'] for k, v in selection['models'].items()}, fixed_comparison_lambda=1,
        corner_summary=corner_summary, frame_paired_summary=frame_summary,
        constraint_summary=constraints, parallel_summary=parallel, line_summary=line_summary,
        quantization_reference=quantized,
        interpretation={
            'difficulty': 'Posthoc GT baseline error bins <=10, (10,20], >20px; not deployable gates.',
            'squared_error': 'With e=baseline-GT and d=fused-baseline: delta_squared=2*e.dot(d)+d.dot(d). Wrong direction means e.dot(d)>0. Overshoot means e.dot(d)<0 but final squared error increases.',
            'line_complementarity': 'A predicted line constrains normal position; tangent position is unobserved by that line alone. Compare its GT normal bias against baseline normal error, not full corner error.',
            'line_matching': 'dht_only uses all GT-supported lines; dht_matched_baseline further restricts to baseline-observed endpoint pairs for a corresponding-line comparison. This mask is diagnostic only and never enters fusion.',
            'parallel': 'Absolute sine of angle between incident predicted lines. Baseline regularization keeps the solve invertible even when sine approaches zero; tangent information remains weak.',
            'oracle_perfect': 'GT-perfect structural lines with the SAME fixed/selected strength are an oracle reference, not a globally optimal upper bound and not deployable.',
            'oracle_best_of': 'Per-corner min(GT error of baseline, GT error of fixed1) is a best-of-two oracle bound; GT selects every point and cannot be used at inference.',
            'quantization': 'GT line parameters rounded to nearest theta/rho bins with seam sign correction. This separates a quantization reference from learned-line errors, not a causal percentage decomposition.',
            'denominators': 'Corner and line summaries first compute each seed statistic, then average seeds. Frame bootstrap averages seeds within a frame before resampling frames; 52 real images remain 52 observations.',
            'frame_delta': 'Mean per-frame corner error with unchanged missing-diagonal penalties; missing-corner contributions cancel between paired arms.',
            'limits': 'Posthoc reused DEV52; structural/amodal cuboid labels do not certify visible physical edges. No6D pose claim, no new gate, no padding/view runs executed by this script.'})
    geometry = dict(schema='dht_fusion_geometry_audit_v1', status='PASS', source_sha256=source_hashes,
        code_sha256={str(p): sha(p) for p in (Path(__file__), HERE/'fusion.py', HERE/'resize_control.py', TARGET_PATH)},
        n_corner_records=len(corners), n_frames=692, real_dev_frames=52, seeds=seeds,
        maximum_squared_error_decomposition_error_px2=max_decomposition_error,
        maximum_perfect_line_oracle_error_increase_px=max_perfect_increase,
        maximum_difference_from_frozen_fused_predictions_px=max_saved_prediction_error,
        unchanged_centroid_checks=unchanged_centers, unchanged_missing_corner_checks=missing_preserved,
        incidence={'edges': F.SIDE_EDGES, 'roles_per_corner': F.INCIDENT_ROLES,
                   'height_roles': [0, 1, 2, 3], 'depth_roles': [4, 5, 6, 7]},
        quantization=quant_audit,
        inference_geometry='frozen fuse_corners receives no GT; separate explicit oracle calls receive GT lines only for diagnosis',
        independent_observation_note='Seed-corner records are not independent real frames')
    assert all(sha(run_dir/name) == digest for name, digest in source_hashes.items())
    E.write(output/'CORNER_DIAGNOSTICS.json', dict(schema='dht_fusion_corner_diagnostics_v1',
            n_records=len(corners), row_key=['model', 'id', 'seed', 'corner'],
            gt_only_fields=['difficulty', 'gt_xy', 'baseline_error_px', 'incident_line_gt_distance_px',
                            'incident_normal_error_px', 'incident_tangent_error_px', 'oracle_gt_only',
                            'all fixed1/selected error and direction diagnostics'], records=corners))
    E.write(output/'DIAGNOSIS.json', report)
    E.write(output/'GEOMETRY_AUDIT.json', geometry)
    print(json.dumps({'status': 'PASS', 'output': str(output), 'corner_records': len(corners),
                      'max_decomposition_error_px2': max_decomposition_error}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    diagnose(parser.parse_args().run_dir)
