"""Select fusion on synthetic validation, then freeze and evaluate paired outputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import fusion as F
from resize_control import correct_legacy_resize_coordinates


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write(path, value):
    temporary = Path(path).with_suffix('.pending.json')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def observed_sources(root):
    files = [root/name for name in ['CONFIG.json', 'manifest.json', 'AUXILIARY_PROTOCOL.json',
                                    'BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'DHT_LINES.json']]
    files += [HERE/name for name in ['evaluate.py', 'fusion.py', 'resize_control.py']]
    return {str(p): sha(p) for p in files}


def inputs(root):
    cfg, manifest = read(root/'CONFIG.json'), read(root/'manifest.json')
    records = manifest['records']
    dope, yolo, dht = [read(root/name) for name in ['BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'DHT_LINES.json']]
    for artifact in [dope, yolo, dht]:
        assert artifact['complete'] and artifact['config_sha256'] == sha(root/'CONFIG.json')
        assert artifact['manifest_sha256'] == sha(root/'manifest.json')
        assert [r['id'] for r in artifact['records']] == [r['id'] for r in records]
        for path, digest in artifact['source_sha256'].items():
            source_path = Path(path)
            if not source_path.is_absolute():
                source_path = HERE.parents[2] / source_path
            assert sha(source_path) == digest, f'Source changed: {path}'
    variants = {'DOPE': dope['records'], 'DOPE_exact_resize': copy.deepcopy(dope['records']), 'YOLO': yolo['records']}
    for r, base in zip(records, variants['DOPE_exact_resize']):
        corrected, valid = correct_legacy_resize_coordinates(np.asarray(base['kps'], float), base['kp_valid'],
                         r['width'], r['height'], base['input_shape_chw'], pad=100, shortest_side=400)
        base['kps'], base['kp_valid'] = clean(corrected), valid.tolist()
    return cfg, manifest, variants, dht['records']


def point_error(r, points, valid):
    return F.corner_metrics(points, valid, r['gt_points'], r.get('gt_valid', [True]*8), r['width'], r['height'])


def fuse(r, base, dht, seed, lam):
    return F.fuse_corners(np.asarray(base['kps'], float), base['kp_valid'], dht['seeds'][str(seed)]['lines'],
                          lam, r['width'], r['height'])


def select(root, cfg, manifest, variants, dht):
    # Only these indices enter parameter selection. No real/test metric is read.
    indices = manifest['populations']['synth_val']
    assert len(indices) == 256 and all(manifest['records'][i]['population'] == 'synth_val' for i in indices)
    result = dict(schema='dht_pose_selection_v1', config_sha256=sha(root/'CONFIG.json'),
                  manifest_sha256=sha(root/'manifest.json'), source_sha256=observed_sources(root),
                  population='synth_val', n_frames=256, seeds=cfg['dht']['seeds'],
                  objective=cfg['fusion']['selection'], models={})
    for name, baselines in variants.items():
        grid = []
        for lam in cfg['fusion']['lambda_grid']:
            values = []
            for seed in cfg['dht']['seeds']:
                scores = []
                for i in indices:
                    r = manifest['records'][i]
                    points, valid = fuse(r, baselines[i], dht[i], seed, lam)
                    scores.append(point_error(r, points, valid)['mean_capped_diagonal_normalized_error'])
                values.append(float(np.mean(scores)))
            grid.append(dict(lambda_value=lam, mean_score=float(np.mean(values)), per_seed=values))
        best = min(grid, key=lambda x: (x['mean_score'], x['lambda_value']))
        result['models'][name] = dict(lambda_value=best['lambda_value'], **{'lambda': best['lambda_value']}, grid=grid)
        print(f'Synthetic validation selection {name}: lambda={best["lambda_value"]}', flush=True)
    path = root/'SELECTION.json'
    if path.exists() and read(path) != result:
        raise ValueError('Frozen synthetic selection differs; investigate before evaluating')
    write(path, result)
    # Explicit read-back freeze boundary before held-out and real scoring.
    assert read(path) == result
    return result


def score_frame(r, points, valid):
    corners = point_error(r, points, valid)
    lines, line_valid = F.corners_to_lines(points, valid)
    metrics = F.line_metrics(lines, line_valid, r['gt_points'], r.get('gt_valid', [True]*8), r['width'], r['height'])
    return dict(id=r['id'], population=r['population'], group=r['group'],
                n_gt=corners['n_gt'], n_pred=corners['n_pred_on_gt'],
                corner_errors=corners['error_px'], corner_penalized=corners['penalized_error_px'],
                corner_capped_normalized=corners['mean_capped_diagonal_normalized_error'],
                frame_corner_mean_px=corners['mean_error_px'],
                frame_corner_penalized_px=corners['mean_error_missing_diagonal_px'],
                all8_gt_valid=corners['all8_gt_valid'], all8_at_10px=corners['all8_at_10px'],
                line_errors=metrics['distance_px'], line_angles=metrics['angle_deg'],
                line_penalized=metrics['penalized_distance_px'], line_n_gt=metrics['n_gt'], line_n_pred=metrics['n_pred_on_gt'])


def summarize(rows, cfg):
    corners = np.concatenate([r['corner_errors'] for r in rows])
    finite = corners[np.isfinite(corners)]
    line_dist = np.concatenate([r['line_errors'] for r in rows])
    line_angle = np.concatenate([r['line_angles'] for r in rows])
    observed_line = np.isfinite(line_dist) & np.isfinite(line_angle)
    n_gt, n_pred = sum(r['n_gt'] for r in rows), sum(r['n_pred'] for r in rows)
    n_line_gt = sum(r['line_n_gt'] for r in rows)
    complete_gt = [r for r in rows if r['all8_gt_valid']]
    return dict(n_frames=len(rows), n_gt_corners=n_gt, n_pred_corners=n_pred,
                corner_coverage=n_pred/n_gt, frames_with_any_corner=sum(r['n_pred'] > 0 for r in rows),
                frames_with_all8=sum(r['n_pred'] == 8 for r in rows),
                corner_median_px=float(np.median(finite)) if len(finite) else None,
                corner_p90_px=float(np.percentile(finite, 90)) if len(finite) else None,
                corner_mean_px=float(np.mean(finite)) if len(finite) else None,
                frame_penalized_mean_px=float(np.mean([r['frame_corner_penalized_px'] for r in rows])),
                capped_normalized_mean=float(np.mean([r['corner_capped_normalized'] for r in rows])),
                **{f'pck_{t}px': float(np.sum(finite <= t)/n_gt) for t in cfg['evaluation']['corner_pck_px']},
                all8_10px_success=float(np.mean([r['all8_at_10px'] for r in complete_gt])) if complete_gt else None,
                all8_eligible_frames=len(complete_gt), line_n_gt=n_line_gt,
                line_coverage=sum(r['line_n_pred'] for r in rows)/n_line_gt if n_line_gt else None,
                line_distance_median_px=float(np.median(line_dist[observed_line])) if observed_line.any() else None,
                line_distance_p90_px=float(np.percentile(line_dist[observed_line], 90)) if observed_line.any() else None,
                line_angle_median_deg=float(np.median(line_angle[observed_line])) if observed_line.any() else None,
                line_success_fraction=float(np.sum(observed_line & (line_dist <= 8) & (line_angle <= 5))/n_line_gt) if n_line_gt else None)


def evaluate(root):
    cfg, manifest, variants, dht = inputs(root)
    before = observed_sources(root)
    selection = select(root, cfg, manifest, variants, dht)
    selected_sha = sha(root/'SELECTION.json')
    records = manifest['records']
    frame_rows, summaries, paired = [], [], []
    fused = dict(schema='dht_pose_fused_v1', config_sha256=sha(root/'CONFIG.json'),
                 manifest_sha256=sha(root/'manifest.json'), selection_sha256=selected_sha, models={})
    for name, baselines in variants.items():
        stored = []
        for r, base in zip(records, baselines):
            stored.append(dict(id=r['id'], baseline=dict(kps=base['kps'], kp_valid=base['kp_valid']), seeds={str(s): {} for s in cfg['dht']['seeds']}))
        fused['models'][name] = dict(records=stored)
        model_rows = []
        for r, base in zip(records, baselines):
            row = score_frame(r, np.asarray(base['kps'], float), base['kp_valid'])
            model_rows.append(dict(model=name, arm='baseline', seed=0, lambda_value=0, **row))
        for seed in cfg['dht']['seeds']:
            for arm, lam in [('selected', selection['models'][name]['lambda']), ('fixed1', cfg['fusion']['diagnostic_lambda'])]:
                for i, r in enumerate(records):
                    points, valid = fuse(r, baselines[i], dht[i], seed, lam)
                    if not np.array_equal(valid, np.asarray(baselines[i]['kp_valid'], bool)):
                        raise ValueError('Fusion unexpectedly changed baseline validity')
                    if not np.array_equal(points[8], np.asarray(baselines[i]['kps'], float)[8], equal_nan=True):
                        raise ValueError('Fusion changed centroid')
                    stored[i]['seeds'][str(seed)][arm] = dict(kps=clean(points), kp_valid=valid.tolist())
                    model_rows.append(dict(model=name, arm=arm, seed=seed, lambda_value=lam, **score_frame(r, points, valid)))
        for arm, seed in [('baseline', 0)] + [(a, s) for s in cfg['dht']['seeds'] for a in ['selected', 'fixed1']]:
            for population in cfg['evaluation']['populations']:
                pop_rows = [r for r in model_rows if r['arm'] == arm and r['seed'] == seed and r['population'] == population]
                groups = ['all'] + sorted({r['group'] for r in pop_rows}) if population == 'real_dev' else ['all']
                for group in groups:
                    subset = [r for r in pop_rows if group == 'all' or r['group'] == group]
                    summaries.append(dict(model=name, arm=arm, seed=seed, population=population, group=group, **summarize(subset, cfg)))
                    if arm != 'baseline':
                        base_by_id = {r['id']: r for r in model_rows if r['arm'] == 'baseline'}
                        delta = np.asarray([r['frame_corner_penalized_px']-base_by_id[r['id']]['frame_corner_penalized_px'] for r in subset])
                        paired.append(dict(model=name, arm=arm, seed=seed, population=population, group=group,
                                           n_frames=len(subset), mean_delta_px=float(delta.mean()), median_delta_px=float(np.median(delta)),
                                           improved_frames=int((delta < -1e-9).sum()), worsened_frames=int((delta > 1e-9).sum()),
                                           unchanged_frames=int((np.abs(delta) <= 1e-9).sum())))
        frame_rows.extend(model_rows)
        print(f'Evaluated baseline and all selected/fixed seed variants: {name}', flush=True)
    seed_summary = []
    aggregate_keys = ['corner_median_px', 'corner_p90_px', 'corner_mean_px', 'frame_penalized_mean_px',
                      'capped_normalized_mean', 'pck_5px', 'pck_10px', 'pck_20px', 'all8_10px_success',
                      'line_distance_median_px', 'line_distance_p90_px', 'line_angle_median_deg', 'line_success_fraction']
    for model, arm, population, group in sorted({(r['model'], r['arm'], r['population'], r['group']) for r in summaries}):
        subset = [r for r in summaries if (r['model'], r['arm'], r['population'], r['group']) == (model, arm, population, group)]
        metrics = {}
        for key in aggregate_keys:
            values = [r[key] for r in subset if r[key] is not None]
            metrics[key] = dict(mean=float(np.mean(values)), min=min(values), max=max(values)) if values else None
        seed_summary.append(dict(model=model, arm=arm, population=population, group=group, seeds=[r['seed'] for r in subset], metrics=metrics))
    if observed_sources(root) != before or sha(root/'SELECTION.json') != selected_sha:
        raise ValueError('Inputs/selection changed during evaluation')
    write(root/'FUSED_PREDICTIONS.json', fused)
    write(root/'FRAME_METRICS.json', dict(schema='dht_pose_frame_metrics_v1', records=frame_rows))
    result = dict(schema='dht_pose_results_v1', complete=True, config_sha256=sha(root/'CONFIG.json'),
                  manifest_sha256=sha(root/'manifest.json'), selection_sha256=selected_sha, source_sha256=before,
                  selection=selection['models'], summaries=summaries, seed_summary=seed_summary, paired=paired,
                  n_frames=len(records), n_frame_variant_rows=len(frame_rows),
                  scope='Inference-time corner/line fusion; not joint training, native YOLO DHT head, or validated6D pose improvement.',
                  metric_notes='Corner pixel median/P90 condition on available predictions; PCK includes missing as failures. Seed summaries average per-seed statistics, not pooled repeats. DEV52 is not an independent final test.',
                  limitations=['Amodal structural side lines, not physical visible-edge labels.',
                               'DOPE uses a single channelwise corner decoder without affinity instance grouping.',
                               'Strict indexed corner evaluation; no GT-chosen symmetry or instance association.',
                               'Backbone synthetic-pretraining overlap with synthetic evaluation is unverified.',
                               'Geometrically corrected DOPE is a preregistered analytic control, not real-data tuning.'])
    write(root/'RESULTS.json', result)
    write(root/'EVALUATION_COMPLETION.json', dict(PASS=True, complete=True, selection_frozen=True,
           no_real_selection=True, source_sha256=before, selection_sha256=selected_sha,
           output_sha256={name: sha(root/name) for name in ['FUSED_PREDICTIONS.json', 'FRAME_METRICS.json', 'RESULTS.json']},
           scope='Artifact and protocol completion; PASS does not mean accuracy improved.'))
    print('Paired evaluation complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run_dir.resolve())


if __name__ == '__main__':
    main()
