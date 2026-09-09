"""Read-only independent metric/protocol audit; writes only EVALUATION_AUDIT.json."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from fusion import fuse_corners


def audit(root, input_scope):
    root = Path(root).resolve()
    def read(name):
        return json.loads((root / name).read_text())
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    names = ['CONFIG.json', 'manifest.json', 'BASELINE_DOPE.json', 'BASELINE_YOLO.json',
             'DHT_LINES.json', 'SELECTION.json', 'FUSED_PREDICTIONS.json',
             'FRAME_METRICS.json', 'RESULTS.json', 'EVALUATION_COMPLETION.json']
    hashes = {name: sha(root / name) for name in names}
    cfg, manifest, selection = read('CONFIG.json'), read('manifest.json'), read('SELECTION.json')
    fused, metrics, results = read('FUSED_PREDICTIONS.json'), read('FRAME_METRICS.json'), read('RESULTS.json')
    dht = read('DHT_LINES.json')['records']
    records, pops = manifest['records'], manifest['populations']
    assert len(records) == 692 and len(pops['synth_val']) == 256 and len(pops['real_dev']) == 52
    assert len(set(pops['synth_val'])) == 256
    assert all(records[i]['population'] == 'synth_val' for i in pops['synth_val'])
    assert set(pops['synth_val']).isdisjoint(pops['real_dev'])
    assert cfg['fusion']['lambda_grid'] == [0, .25, .5, 1, 2, 4]
    assert cfg['dht']['seeds'] == [1, 2, 3] == selection['seeds']
    assert selection['population'] == 'synth_val' and selection['n_frames'] == 256
    assert list(inspect.signature(fuse_corners).parameters) == ['points', 'valid', 'lines', 'lam', 'width', 'height']

    def independent_metric(record, points, pred_valid):
        truth = np.asarray(record['gt_points'], float)[:8]
        valid = np.asarray(record['gt_valid'], bool)[:8] & np.isfinite(truth).all(-1)
        observed = np.asarray(pred_valid, bool)[:8] & np.isfinite(points[:8]).all(-1) & valid
        distance = np.full(8, np.nan)
        distance[observed] = np.linalg.norm(points[:8][observed] - truth[observed], axis=-1)
        diagonal = np.hypot(record['width'], record['height'])
        penalized = np.full(8, diagonal)
        penalized[observed] = distance[observed]
        return valid, observed, distance, float(np.minimum(penalized[valid] / diagonal, 1).mean()), float(penalized[valid].mean())

    # Recompute synthetic-only selection objective without using corner_metrics.
    max_score_error, grid_scores = 0.0, {}
    for model, entries in fused['models'].items():
        grid_scores[model] = []
        for saved in selection['models'][model]['grid']:
            lam, seed_scores = saved['lambda_value'], []
            for seed in cfg['dht']['seeds']:
                scores = []
                for i in pops['synth_val']:
                    r, b = records[i], entries['records'][i]['baseline']
                    points, valid = fuse_corners(np.asarray(b['kps'], float), b['kp_valid'],
                        dht[i]['seeds'][str(seed)]['lines'], lam, r['width'], r['height'])
                    scores.append(independent_metric(r, points, valid)[3])
                seed_scores.append(float(np.mean(scores)))
            mean = float(np.mean(seed_scores))
            max_score_error = max(max_score_error, abs(mean - saved['mean_score']),
                                  float(np.max(np.abs(np.asarray(seed_scores) - saved['per_seed']))))
            grid_scores[model].append({'lambda': lam, 'mean_score': mean})
        best = min(grid_scores[model], key=lambda v: (v['mean_score'], v['lambda']))['lambda']
        assert best == selection['models'][model]['lambda']
    assert max_score_error < 1e-12

    # Baseline coordinate correction is checked using the closed-form affine map.
    raw_dope, raw_yolo = read('BASELINE_DOPE.json')['records'], read('BASELINE_YOLO.json')['records']
    max_resize_error = 0.0
    for model in ('DOPE', 'DOPE_exact_resize', 'YOLO'):
        raw = raw_yolo if model == 'YOLO' else raw_dope
        for i, entry in enumerate(fused['models'][model]['records']):
            p = np.asarray(raw[i]['kps'], float)
            expected = p.copy()
            if model == 'DOPE_exact_resize':
                width, height = records[i]['width'], records[i]['height']
                _, nh, nw = raw[i]['input_shape_chw']
                sc = 400 / min(width, height)
                finite = np.isfinite(p).all(-1)
                expected[finite] = (p[finite] + 100) * [sc * width / nw, sc * height / nh] - 100
            b = entry['baseline']
            np.testing.assert_allclose(np.asarray(b['kps'], float), expected, atol=1e-11, rtol=0, equal_nan=True)
            assert b['kp_valid'] == raw[i]['kp_valid']
            finite = np.isfinite(expected)
            if finite.any():
                max_resize_error = max(max_resize_error, float(np.abs(np.asarray(b['kps'], float)[finite] - expected[finite]).max()))

    stored_rows = {(r['model'], r['arm'], r['seed'], r['id']): r for r in metrics['records']}
    center_checks, missing_checks, zero_checks, metric_checks = 0, 0, 0, 0
    max_metric_error = 0.0
    for model, entries in fused['models'].items():
        for i, entry in enumerate(entries['records']):
            b = entry['baseline']
            variants = [('baseline', 0, b)] + [(arm, int(seed), pred)
                for seed, arms in entry['seeds'].items() for arm, pred in arms.items()]
            for arm, seed, pred in variants:
                p = np.asarray(pred['kps'], float)
                if arm != 'baseline':
                    assert pred['kp_valid'] == b['kp_valid']
                    missing_checks += 1
                    np.testing.assert_equal(p[8], np.asarray(b['kps'], float)[8])
                    center_checks += 1
                    if arm == 'selected' and selection['models'][model]['lambda'] == 0:
                        np.testing.assert_equal(p, np.asarray(b['kps'], float))
                        zero_checks += 1
                gt, observed, distance, capped, penalized = independent_metric(records[i], p, pred['kp_valid'])
                saved = stored_rows[(model, arm, seed, entry['id'])]
                assert int(gt.sum()) == saved['n_gt'] and int(observed.sum()) == saved['n_pred']
                np.testing.assert_allclose(distance, np.asarray(saved['corner_errors'], float), atol=1e-11, rtol=0, equal_nan=True)
                max_metric_error = max(max_metric_error, abs(capped - saved['corner_capped_normalized']),
                                       abs(penalized - saved['frame_corner_penalized_px']))
                expected_all8 = bool((observed & (distance <= 10)).all()) if gt.all() else None
                assert saved['all8_at_10px'] == expected_all8
                metric_checks += 1
    assert max_metric_error < 1e-10

    summary_checks, denominator_examples = 0, []
    for s in results['summaries']:
        rows = [r for r in metrics['records'] if r['model'] == s['model'] and r['arm'] == s['arm']
                and r['seed'] == s['seed'] and r['population'] == s['population']
                and (s['group'] == 'all' or r['group'] == s['group'])]
        n_gt, n_pred = sum(r['n_gt'] for r in rows), sum(r['n_pred'] for r in rows)
        errors = np.asarray([e for r in rows for e in r['corner_errors']], float)
        assert n_gt == s['n_gt_corners'] and n_pred == s['n_pred_corners']
        assert abs(n_pred / n_gt - s['corner_coverage']) < 1e-12
        for threshold in (5, 10, 20):
            assert abs(np.sum(errors <= threshold) / n_gt - s[f'pck_{threshold}px']) < 1e-12
        eligible = [r['all8_at_10px'] for r in rows if r['all8_gt_valid']]
        assert len(eligible) == s['all8_eligible_frames']
        if eligible:
            assert abs(np.mean(eligible) - s['all8_10px_success']) < 1e-12
        summary_checks += 1
        if s['population'] == 'real_dev' and s['group'] == 'all' and s['arm'] == 'baseline':
            denominator_examples.append({k: s[k] for k in ('model', 'n_frames', 'n_gt_corners',
                'n_pred_corners', 'corner_coverage', 'pck_10px', 'all8_eligible_frames')})
    completion = read('EVALUATION_COMPLETION.json')
    assert completion['PASS'] and completion['complete']
    for name, digest in completion['output_sha256'].items():
        assert sha(root / name) == digest
    assert all(sha(root / name) == digest for name, digest in hashes.items())
    report = dict(schema='dht_pose_independent_evaluation_audit_v1', status='PASS',
        input_scope=input_scope, audited_root=str(root), audited_sha256=hashes,
        audit_source_sha256=sha(__file__),
        source_review=dict(selection='synth_val256 only; fixed six lambdas; three seeds equally averaged; smaller-lambda exact tie break; SELECTION read-back before scoring',
            inference='evaluate.fuse passes predicted kps/valid, predicted lines, lambda and image sizes only; no GT/support/facing inputs',
            ground_truth='used only by selection on synthetic val and evaluation metrics; no real/model/role/instance matching',
            resize_control='analytic per-axis inverse-size correction, no GT or error inputs',
            line_metrics='GT support restricted to >=2px annotated segments intersecting original frame, evaluator only'),
        selected_lambda={k: v['lambda'] for k, v in selection['models'].items()},
        independently_recomputed_grid=grid_scores, maximum_selection_score_error=max_score_error,
        maximum_baseline_affine_error_px=max_resize_error, maximum_frame_metric_error=max_metric_error,
        frame_variant_metric_checks=metric_checks, unchanged_validity_checks=missing_checks,
        unchanged_centroid_checks=center_checks, lambda_zero_exact_identity_checks=zero_checks,
        summary_denominator_checks=summary_checks, real_dev_denominator_examples=denominator_examples,
        limitations=['PASS establishes protocol/metric consistency, not accuracy improvement.',
                     'Cached-line and fresh batch1 DHT inputs are distinct experiments; inspect input_scope and exact DHT_LINES hash.',
                     'This audit does not establish real physical edge visibility or 6D pose correctness.'])
    (root / 'EVALUATION_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'input_scope', 'selected_lambda',
          'maximum_selection_score_error', 'frame_variant_metric_checks', 'unchanged_centroid_checks')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--input-scope', required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.input_scope)
