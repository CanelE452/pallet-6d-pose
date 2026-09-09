"""Calibrate on synthetic128, select on synthetic128, then evaluate uncertainty fusion."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

import evaluate as E
import fusion as F
import uncertainty_geometry as U

HERE = Path(__file__).resolve().parent
FAMILIES = ('point_sigma_only', 'line_uncertainty_only', 'point_line_uncertainty')
ARMS = ('baseline', 'previous_joint_gate', 'unconditional_lambda1', *FAMILIES)
ROLE_FAMILY = np.array([0, 0, 0, 0, 1, 1, 1, 1])
INCIDENT_FAMILY = ROLE_FAMILY[np.asarray(F.INCIDENT_ROLES)]


def sigma_array(point):
    if point['sigma_original_px'] is None:
        assert not any(point['kp_valid']), 'Observed points require sigma outputs'
        return np.full((9, 2), np.nan)
    sigma = np.asarray(point['sigma_original_px'], float)
    assert sigma.shape == (9, 2)
    return sigma


def source_hashes(out):
    return {str(p): E.sha(p) for p in [out/'PROTOCOL.json', out/'YOLO_UNCERTAINTY.json',
            out/'DHT_UNCERTAINTY.json', HERE/'evaluate_uncertainty.py', HERE/'uncertainty_geometry.py',
            HERE/'evaluate.py', HERE/'fusion.py']}


def freeze(path, value):
    cleaned = E.clean(value)
    if path.exists() and E.read(path) != cleaned:
        raise ValueError(f'Frozen artifact differs: {path}')
    E.write(path, cleaned)
    assert E.read(path) == cleaned
    return E.read(path)


def calibrate(out, protocol, records, points, lines):
    calibration_ids = set(protocol['calibration_ids'])
    selection_ids = set(protocol['selection_ids'])
    assert len(calibration_ids) == len(selection_ids) == 128
    assert not calibration_ids & selection_ids
    ratios_point, squared_point = [], []
    ratios_line, squared_line = [[], []], [[], []]
    for r, point, line in zip(records, points, lines):
        if r['id'] not in calibration_ids:
            continue
        assert r['population'] == 'synth_val'
        p, gt = np.asarray(point['kps'], float)[:8], np.asarray(r['gt_points'], float)[:8]
        observed = np.asarray(point['kp_valid'], bool)[:8] & np.asarray(r['gt_valid'], bool)
        observed &= np.isfinite(p).all(-1) & np.isfinite(gt).all(-1)
        raw_variance = np.maximum(sigma_array(point)[:8]**2, 1.)
        assert np.isfinite(raw_variance[observed]).all()
        error2 = (p[observed]-gt[observed])**2
        squared_point.extend(error2)
        ratios_point.extend(error2/raw_variance[observed])
        support = F.gt_line_support(gt, r['gt_valid'], r['width'], r['height'])
        for seed in protocol['seeds']:
            dht = line['seeds'][str(seed)]
            h = np.asarray(dht['h_mode'], float)
            matrices = np.asarray(dht['moment_about_mode'], float)
            for role, (a, b) in enumerate(F.SIDE_EDGES):
                if not support[role]:
                    continue
                z = np.column_stack([gt[[a, b]], np.ones(2)])
                raw = np.maximum(np.einsum('ki,ij,kj->k', z, matrices[role], z), 1.)
                residual2 = (z@h[role])**2
                family = ROLE_FAMILY[role]
                ratios_line[family].extend(residual2/raw)
                squared_line[family].extend(residual2)
    result = dict(schema='dht_uncertainty_calibration_v1', complete=True,
                  ids=protocol['calibration_ids'], n_frames=128, population='synth_val_calibration',
                  point_alpha_xy=np.mean(ratios_point, axis=0),
                  constant_point_variance_xy=np.maximum(np.mean(squared_point, axis=0), 1.),
                  line_beta_height_depth=[np.mean(v) for v in ratios_line],
                  constant_line_variance_height_depth=[max(float(np.mean(v)), 1.) for v in squared_line],
                  n_point_coordinate_pairs=len(ratios_point),
                  n_line_endpoint_seed_observations=[len(v) for v in ratios_line],
                  source_sha256=source_hashes(out),
                  semantics='Global variance multipliers fitted only on calibration128. Working squared-error scales, not certified probabilistic coverage.')
    return freeze(out/'CALIBRATION.json', result)


def prepare(records, points, lines, calibration, seeds):
    prepared = []
    for r, point, line in zip(records, points, lines):
        p = np.asarray(point['kps'], float)
        vp = np.maximum(np.maximum(sigma_array(point)**2, 1.) *
                        np.asarray(calibration['point_alpha_xy']), 1.)
        constant_point = np.broadcast_to(calibration['constant_point_variance_xy'], p.shape).copy()
        constant_line = np.asarray(calibration['constant_line_variance_height_depth'])[INCIDENT_FAMILY]
        rows = {}
        for seed in seeds:
            dht = line['seeds'][str(seed)]
            raw = U.line_variance_at_points(p, dht['moment_about_mode'])
            vl = np.maximum(np.maximum(raw, 1.) *
                            np.asarray(calibration['line_beta_height_depth'])[INCIDENT_FAMILY], 1.)
            rows[str(seed)] = dict(lines=dht['lines'], raw_line_variance=raw,
                                  point_variance=vp, line_variance=vl,
                                  constant_point_variance=constant_point, constant_line_variance=constant_line)
        prepared.append(dict(points=p, valid=np.asarray(point['kp_valid'], bool),
                             width=r['width'], height=r['height'], seeds=rows))
    return prepared


def predict(prepared, seed, family, rule):
    values = prepared['seeds'][str(seed)]
    vp = values['constant_point_variance'] if family == 'line_uncertainty_only' else values['point_variance']
    vl = values['constant_line_variance'] if family == 'point_sigma_only' else values['line_variance']
    cap = rule['max_move_diagonal_fraction']
    max_move = None if cap is None else cap*np.hypot(prepared['width'], prepared['height'])
    p, valid = U.fuse(prepared['points'], prepared['valid'], values['lines'], vp, vl,
                      rule['lambda'], prepared['width'], prepared['height'], max_move=max_move)
    return p, valid, vp, vl


def select(out, protocol, records, prepared):
    chosen_ids = set(protocol['selection_ids'])
    indices = [i for i, r in enumerate(records) if r['id'] in chosen_ids]
    assert len(indices) == 128 and all(records[i]['population'] == 'synth_val' for i in indices)
    families = {}
    for family in FAMILIES:
        grid = []
        for lam in protocol['lambda_grid']:
            for cap in protocol['max_move_diagonal_fractions']:
                rule = {'lambda': lam, 'max_move_diagonal_fraction': cap}
                seed_scores = []
                for seed in protocol['seeds']:
                    scores = []
                    for i in indices:
                        p, valid, _, _ = predict(prepared[i], seed, family, rule)
                        scores.append(E.point_error(records[i], p, valid)['mean_capped_diagonal_normalized_error'])
                    seed_scores.append(float(np.mean(scores)))
                grid.append(dict(rule=rule, score=float(np.mean(seed_scores)), per_seed_scores=seed_scores))
        best = min(grid, key=lambda x: (x['score'], x['rule']['lambda'],
                                       x['rule']['max_move_diagonal_fraction'] if x['rule']['max_move_diagonal_fraction'] is not None else float('inf')))
        families[family] = dict(selected_rule=best['rule'], selection_score=best['score'], grid=grid)
        print(f'Synthetic selection: {family}: {best["rule"]}, score={best["score"]:.8f}', flush=True)
    return freeze(out/'SELECTION.json', dict(schema='dht_uncertainty_selection_v1', complete=True,
        population='synth_val_selection', n_frames=128, ids=protocol['selection_ids'],
        calibration_sha256=E.sha(out/'CALIBRATION.json'), source_sha256=source_hashes(out), families=families))


def aggregate(summaries):
    result = []
    keys = ['corner_mean_px', 'corner_median_px', 'corner_p90_px', 'pck_5px', 'pck_10px', 'pck_20px',
            'corner_coverage', 'frame_penalized_mean_px', 'capped_normalized_mean', 'all8_10px_success',
            'n_adjusted_corners', 'n_adjusted_frames', 'mean_move_px']
    for identity in sorted({(r['arm'], r['population'], r['group']) for r in summaries}):
        rows = [r for r in summaries if (r['arm'], r['population'], r['group']) == identity]
        metrics = {}
        for k in keys:
            values = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
            metrics[k] = dict(mean=float(np.mean(values)), min=min(values), max=max(values)) if values else dict(mean=None, min=None, max=None)
        result.append(dict(model='YOLO', arm=identity[0], population=identity[1], group=identity[2],
                           seeds=[r['seed'] for r in rows], metrics=metrics))
    return result


def auc(values, labels):
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        return None
    return float((rankdata(values)[labels].sum()-npos*(npos+1)/2)/(npos*nneg))


def reliability(out, protocol, records, points, lines, prepared):
    observations = []
    for r, point, line, prep in zip(records, points, lines, prepared):
        p, valid = prep['points'][:8], prep['valid'][:8]
        gt = np.asarray(r['gt_points'], float)
        for corner in range(8):
            if not valid[corner] or not r['gt_valid'][corner]:
                continue
            error = float(np.linalg.norm(p[corner]-gt[corner]))
            raw_sigma = sigma_array(point)[corner]
            scores = dict(low_kp_conf=1-float(point['kp_conf'][corner]),
                          point_sigma_norm=float(np.linalg.norm(raw_sigma)),
                          point_sigma_calibrated_norm=float(np.sqrt(prep['seeds']['1']['point_variance'][corner].sum())))
            for signal, value in scores.items():
                observations.append(dict(id=r['id'], population=r['population'], group=r['group'],
                                         signal=signal, value=value, error_px=error, threshold_px=10))
        support = F.gt_line_support(gt, r['gt_valid'], r['width'], r['height'])
        for seed in protocol['seeds']:
            dht = line['seeds'][str(seed)]
            h = np.asarray(dht['h_mode'], float)
            matrices = np.asarray(dht['moment_about_mode'], float)
            for role, (a, b) in enumerate(F.SIDE_EDGES):
                if not (valid[a] and valid[b] and support[role]):
                    continue
                zpred = np.column_stack([p[[a, b]], np.ones(2)])
                zgt = np.column_stack([gt[[a, b]], np.ones(2)])
                spread = float(np.sqrt(np.maximum(np.einsum('ki,ij,kj->k', zpred, matrices[role], zpred), 0).mean()))
                error = float(np.abs(zgt@h[role]).mean())
                scores = dict(line_posterior_rms=spread,
                              low_line_peak=1-dht['peak_probability'][role],
                              line_entropy=dht['entropy_normalized'][role])
                for signal, value in scores.items():
                    observations.append(dict(id=r['id'], population=r['population'], group=r['group'],
                                             signal=signal, value=float(value), error_px=error, threshold_px=8))
    calibration_ids = set(protocol['calibration_ids'])
    signals = sorted({r['signal'] for r in observations})
    thresholds = {s: np.quantile([r['value'] for r in observations if r['signal'] == s and r['id'] in calibration_ids],
                                 [.25, .5, .75]).tolist() for s in signals}
    rows = []
    for signal in signals:
        for population in ('synth_val', 'synth_test', 'cross_v4', 'real_dev'):
            groups = ['all'] + sorted({r['group'] for r in observations if r['population'] == population}) if population == 'real_dev' else ['all']
            for group in groups:
                subset = [r for r in observations if r['signal'] == signal and r['population'] == population and
                          (group == 'all' or r['group'] == group)]
                if not subset:
                    continue
                values = np.asarray([r['value'] for r in subset])
                errors = np.asarray([r['error_px'] for r in subset])
                indices = np.digitize(values, thresholds[signal])
                bins = []
                for b in range(4):
                    take = indices == b
                    bins.append(dict(bin=b, n_observations=int(take.sum()), n_corners=int(take.sum()) if signal.startswith('point') or signal == 'low_kp_conf' else None,
                                     n_frames=len({r['id'] for r, keep in zip(subset, take) if keep}),
                                     mean_signal=float(values[take].mean()) if take.any() else None,
                                     mean_error_px=float(errors[take].mean()) if take.any() else None,
                                     p90_error_px=float(np.percentile(errors[take], 90)) if take.any() else None))
                rho = float(spearmanr(values, errors).statistic) if len(np.unique(values)) > 1 and len(np.unique(errors)) > 1 else None
                rows.append(dict(signal=signal, population=population, group=group, n_observations=len(subset),
                                 n_frames=len({r['id'] for r in subset}), spearman=rho,
                                 error_threshold_px=subset[0]['threshold_px'], auroc=auc(values, errors > subset[0]['threshold_px']),
                                 mean_error_px=float(errors.mean()), p90_error_px=float(np.percentile(errors, 90)), bins=bins))
    result = dict(complete=True, thresholds_fit_ids=protocol['calibration_ids'], thresholds=thresholds, rows=rows,
                  scope='Prediction-only uncertainty scores versus GT error after selection. Point observations are corners; line observations include three dependent seeds, not independent frames. Quantile thresholds use calibration128 only. Sigma/variance ellipses are not coverage intervals.')
    E.write(out/'RELIABILITY.json', result)
    return result


def main(live):
    out = live/'uncertainty_fusion_v1'
    protocol = E.read(out/'PROTOCOL.json')
    for name, digest in protocol['source_sha256'].items():
        assert E.sha(live/name) == digest, name
    config, manifest = E.read(live/'CONFIG.json'), E.read(live/'manifest.json')
    records = manifest['records']
    point_file, line_file = E.read(out/'YOLO_UNCERTAINTY.json'), E.read(out/'DHT_UNCERTAINTY.json')
    assert point_file['complete'] and line_file['complete']
    point_audit, line_audit = E.read(out/'EXTRACTION_AUDIT_YOLO.json'), E.read(out/'EXTRACTION_AUDIT_DHT.json')
    assert point_audit['PASS'] and line_audit['PASS']
    assert point_audit['prediction_sha256'] == E.sha(out/'YOLO_UNCERTAINTY.json')
    assert line_audit['output_sha256'] == E.sha(out/'DHT_UNCERTAINTY.json')
    points, lines = point_file['records'], line_file['records']
    expected = [r['id'] for r in records]
    assert [r['id'] for r in points] == [r['id'] for r in lines] == expected
    source_before = source_hashes(out)
    baseline = E.read(live/'BASELINE_YOLO.json')['records']
    for p, b in zip(points, baseline):
        assert np.array_equal(np.asarray(p['kps'], float), np.asarray(b['kps'], float), equal_nan=True)
        assert p['kp_valid'] == b['kp_valid']
    calibration = calibrate(out, protocol, records, points, lines)
    prepared = prepare(records, points, lines, calibration, protocol['seeds'])
    selection = select(out, protocol, records, prepared)
    selection_sha = E.sha(out/'SELECTION.json')
    previous = E.read(live/'fusion_diagnosis_v1/GATED_PREDICTIONS.json')['models']['YOLO']['records']
    assert [r['id'] for r in previous] == expected
    predictions, frame_rows = [], []
    for i, (r, point, prep) in enumerate(zip(records, points, prepared)):
        p, valid = prep['points'], prep['valid']
        saved = dict(id=r['id'], baseline=dict(kps=p, kp_valid=valid, kp_conf=point['kp_conf'],
                    sigma_original_px=sigma_array(point), point_variance_px2=prep['seeds']['1']['point_variance']), seeds={})
        base_row = dict(model='YOLO', arm='baseline', seed=0, **E.score_frame(r, p, valid),
                        n_adjusted_corners=0, n_adjusted_frames=0, mean_move_px=0.)
        frame_rows.append(base_row)
        for seed in protocol['seeds']:
            arms = {}
            for arm in ARMS[1:]:
                if arm in FAMILIES:
                    q, v, vp, vl = predict(prep, seed, arm, selection['families'][arm]['selected_rule'])
                else:
                    old_arm = 'confidence_line_move' if arm == 'previous_joint_gate' else 'unconditional_lambda1'
                    old = previous[i]['seeds'][str(seed)][old_arm]
                    q, v = np.asarray(old['kps'], float), np.asarray(old['kp_valid'], bool)
                    vp, vl = prep['seeds'][str(seed)]['point_variance'], prep['seeds'][str(seed)]['line_variance']
                assert np.array_equal(v, valid)
                assert np.array_equal(q[~valid], p[~valid], equal_nan=True)
                assert np.array_equal(q[8], p[8], equal_nan=True)
                movement = np.linalg.norm(q-p, axis=-1)
                adjusted = valid[:8] & (movement[:8] > 1e-9)
                arms[arm] = dict(kps=q, kp_valid=v, point_variance_px2=vp, incident_line_variance_px2=vl,
                                 move_px=movement, variance_used_for_fusion=arm in FAMILIES)
                frame_rows.append(dict(model='YOLO', arm=arm, seed=seed, **E.score_frame(r, q, v),
                                       n_adjusted_corners=int(adjusted.sum()), n_adjusted_frames=int(adjusted.any()),
                                       mean_move_px=float(movement[:8][valid[:8]].mean()) if valid[:8].any() else 0.))
            saved['seeds'][str(seed)] = arms
        predictions.append(saved)
    summaries = []
    for arm, seed in [('baseline', 0)] + [(a, s) for a in ARMS[1:] for s in protocol['seeds']]:
        for population in ('synth_val', 'synth_test', 'cross_v4', 'real_dev'):
            groups = ['all'] + sorted({r['group'] for r in records if r['population'] == population}) if population == 'real_dev' else ['all']
            for group in groups:
                rows = [r for r in frame_rows if r['arm'] == arm and r['seed'] == seed and r['population'] == population and (group == 'all' or r['group'] == group)]
                summaries.append(dict(model='YOLO', arm=arm, seed=seed, population=population, group=group,
                                      **E.summarize(rows, config), n_adjusted_corners=sum(r['n_adjusted_corners'] for r in rows),
                                      n_adjusted_frames=sum(r['n_adjusted_frames'] for r in rows), mean_move_px=float(np.mean([r['mean_move_px'] for r in rows]))))
    for family in FAMILIES:
        scores = [r['corner_capped_normalized'] for r in frame_rows if r['arm'] == family and r['id'] in set(protocol['selection_ids'])]
        assert abs(np.mean(scores)-selection['families'][family]['selection_score']) < 1e-12
    E.write(out/'PREDICTIONS.json', dict(complete=True, records=predictions))
    E.write(out/'FRAME_METRICS.json', dict(complete=True, records=frame_rows))
    reliability(out, protocol, records, points, lines, prepared)
    result = dict(schema='dht_uncertainty_results_v1', complete=True, model='YOLO', n_frames=len(records),
                  n_frame_variant_rows=len(frame_rows), summaries=summaries, seed_summary=aggregate(summaries),
                  selection=selection, calibration=calibration, source_sha256=source_before,
                  scope='Separate synthetic128 calibration and synthetic128 selection, followed by reused realDEV52 evaluation; no new neural training. Explicit lambda0 fallback. No6D pose or independent real generalization claim.',
                  limitations=['RLE scale and Hough posterior spread are empirically rescaled working variances, not certified probabilities.',
                               'RLE raw scale evaluated in original pixels after matching the selected detection anchor.',
                               'Global synthetic variance multipliers may not transfer to real images.',
                               'Missing points remain missing. Side labels include amodal structure.',
                               'Old joint gate used all256 validation frames; its comparison here is a historical reference.'])
    E.write(out/'RESULTS.json', result)
    assert source_hashes(out) == source_before and E.sha(out/'SELECTION.json') == selection_sha
    E.write(out/'EVALUATION_AUDIT.json', dict(PASS=True, complete=True, source_sha256=source_before,
            calibration_selection_disjoint=True, no_real_fitting=True, selection_score_recomputed=True,
            missing_centroid_preserved=True, baseline_coordinates_bitwise_preserved=True,
            n_frame_variant_rows=len(frame_rows), output_sha256={name: E.sha(out/name) for name in
              ['CALIBRATION.json', 'SELECTION.json', 'PREDICTIONS.json', 'FRAME_METRICS.json', 'RELIABILITY.json', 'RESULTS.json']}))
    print('Uncertainty evaluation complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
