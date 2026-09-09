"""Independently verify stored gated corners and bootstrap paired real frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def audit(output):
    source = output.parent
    protocol = read(output/'PROTOCOL.json')
    for name, digest in protocol['input_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    manifest = read(source/'manifest.json')
    truth = {r['id']: r for r in manifest['records']}
    real_ids = {r['id'] for r in manifest['records'] if r['population'] == 'real_dev'}
    predictions = read(output/'GATED_PREDICTIONS.json')['models']
    rows = read(output/'GATE_FRAME_METRICS.json')['records']
    by_key = {(r['model'], r['arm'], r['seed'], r['id']): r for r in rows}
    assert len(by_key) == len(rows), 'Duplicate metric identity'
    checked = 0
    differences = {}
    for model, model_data in predictions.items():
        assert [r['id'] for r in model_data['records']] == [r['id'] for r in manifest['records']]
        for record in model_data['records']:
            gt_record = truth[record['id']]
            gt = np.asarray(gt_record['gt_points'], float)[:8]
            gv = np.asarray(gt_record.get('gt_valid', [True]*8), bool) & np.isfinite(gt).all(-1)
            base = record['baseline']
            bp = np.asarray(base['kps'], float)
            bv = np.asarray(base['kp_valid'], bool)[:8] & np.isfinite(bp[:8]).all(-1)
            observed = gv & bv
            be = np.linalg.norm(bp[:8][observed]-gt[observed], axis=-1)
            for seed, arms in record['seeds'].items():
                for arm, pred in arms.items():
                    p = np.asarray(pred['kps'], float)
                    valid = np.asarray(pred['kp_valid'], bool)
                    assert np.array_equal(valid, base['kp_valid']), 'Validity changed'
                    assert np.array_equal(p[8], bp[8], equal_nan=True), 'Center changed'
                    assert np.array_equal(p[:8][~bv], bp[:8][~bv], equal_nan=True), 'Missing corner changed'
                    error = np.full(8, np.nan)
                    error[observed] = np.linalg.norm(p[:8][observed]-gt[observed], axis=-1)
                    row = by_key[(model, arm, int(seed), record['id'])]
                    assert np.allclose(error, np.asarray(row['corner_errors'], float),
                                       atol=1e-8, rtol=0, equal_nan=True), 'Corner error mismatch'
                    penalized = np.full(8, np.nan)
                    diagonal = np.hypot(gt_record['width'], gt_record['height'])
                    penalized[gv] = diagonal
                    penalized[observed] = error[observed]
                    score = np.mean(np.minimum(penalized[gv]/diagonal, 1))
                    assert abs(score-row['corner_capped_normalized']) < 1e-12, 'Score mismatch'
                    if record['id'] in real_ids:
                        key = (model, arm, record['id'])
                        differences.setdefault(key, []).append({
                            'error_delta_sum': float(np.sum(error[observed]-be)),
                            'n_pred': int(observed.sum()), 'n_gt': int(gv.sum()),
                            'pck_delta_sum': int(np.sum(error[observed] <= 10)-np.sum(be <= 10)),
                            'group': gt_record['group'],
                        })
                    checked += 1
    real = [r for r in manifest['records'] if r['population'] == 'real_dev']
    assert len(real) == 52
    groups = sorted({r['group'] for r in real})
    rng = np.random.default_rng(7261)
    draws = np.concatenate([rng.choice([i for i, r in enumerate(real) if r['group'] == group],
                                      size=(5000, sum(r['group'] == group for r in real)), replace=True)
                            for group in groups], axis=1)
    intervals = []
    for model, arm in sorted({(k[0], k[1]) for k in differences}):
        values = []
        for record in real:
            seed_rows = differences[(model, arm, record['id'])]
            assert len(seed_rows) == 3, 'Expected all three seeds for paired frame'
            values.append([np.mean([r[k] for r in seed_rows])
                           for k in ['error_delta_sum', 'n_pred', 'pck_delta_sum', 'n_gt']])
        values = np.asarray(values)
        sampled = values[draws].sum(axis=1)
        totals = values.sum(axis=0)
        err_samples = sampled[:, 0]/sampled[:, 1]
        pck_samples = 100*sampled[:, 2]/sampled[:, 3]
        intervals.append(dict(model=model, arm=arm, n_unique_frames=52,
                              mean_observed_corner_error_delta_px=float(totals[0]/totals[1]),
                              mean_error_delta_ci95=np.percentile(err_samples, [2.5, 97.5]).tolist(),
                              pck10_delta_percentage_points=float(100*totals[2]/totals[3]),
                              pck10_delta_ci95=np.percentile(pck_samples, [2.5, 97.5]).tolist()))
    result = dict(PASS=True, verified_prediction_rows=checked,
                  source_artifact_hashes_unchanged=True, paired_bootstrap=intervals,
                  bootstrap=dict(replicates=5000, random_seed=7261, sampling='Paired image resampling within the three real DEV groups; three DHT seeds averaged inside each image.',
                                 limitations='Intervals describe resampling of these 52 reused DEV images. They do not establish independent generalization; correlated video frames/scenes and model uncertainty are not fully represented.'))
    (output/'STATISTICAL_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    audit(parser.parse_args().output_dir.resolve())
