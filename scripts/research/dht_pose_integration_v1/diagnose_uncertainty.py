"""Describe frozen uncertainty-fusion changes by GT difficulty; never select a rule."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import evaluate as E
from evaluate_uncertainty import FAMILIES


def main(live):
    out = live/'uncertainty_fusion_v1'
    paths = [live/'manifest.json', out/'RESULTS.json', out/'PREDICTIONS.json',
             out/'SELECTION.json', Path(__file__)]
    source = {str(p): E.sha(p) for p in paths}
    assert E.read(out/'RESULTS.json')['complete']
    manifest = E.read(live/'manifest.json')['records']
    predictions = E.read(out/'PREDICTIONS.json')['records']
    observations = []
    for r, p in zip(manifest, predictions):
        assert r['id'] == p['id']
        baseline = np.asarray(p['baseline']['kps'], float)[:8]
        gt = np.asarray(r['gt_points'], float)[:8]
        valid = np.asarray(p['baseline']['kp_valid'], bool)[:8] & r['gt_valid']
        base_error = np.linalg.norm(baseline-gt, axis=-1)
        for arm in FAMILIES:
            fused = np.asarray([p['seeds'][str(s)][arm]['kps'][:8] for s in (1, 2, 3)], float)
            errors = np.linalg.norm(fused-gt, axis=-1)
            movement = np.linalg.norm(fused-baseline, axis=-1)
            for k in np.flatnonzero(valid):
                observations.append(dict(id=r['id'], population=r['population'], group=r['group'],
                    arm=arm, corner=int(k), difficulty='easy_le10' if base_error[k] <= 10 else
                    ('moderate_10to20' if base_error[k] <= 20 else 'hard_gt20'),
                    baseline_error_px=base_error[k], fused_error_px=errors[:, k].mean(),
                    improved_fraction=(errors[:, k] < base_error[k]-1e-9).mean(),
                    worsened_fraction=(errors[:, k] > base_error[k]+1e-9).mean(),
                    mean_move_px=movement[:, k].mean()))
    rows = []
    for population in ('synth_test', 'cross_v4', 'real_dev'):
        groups = ['all'] + sorted({r['group'] for r in observations if r['population'] == population}) if population == 'real_dev' else ['all']
        for group in groups:
            for arm in FAMILIES:
                for difficulty in ('easy_le10', 'moderate_10to20', 'hard_gt20'):
                    subset = [r for r in observations if r['population'] == population and r['arm'] == arm
                              and r['difficulty'] == difficulty and (group == 'all' or r['group'] == group)]
                    if not subset:
                        continue
                    means = {key: float(np.mean([r[key] for r in subset])) for key in
                             ('baseline_error_px', 'fused_error_px', 'improved_fraction', 'worsened_fraction', 'mean_move_px')}
                    rows.append(dict(population=population, group=group, arm=arm, difficulty=difficulty,
                                     n_corners=len(subset), n_frames=len({r['id'] for r in subset}), **means,
                                     delta_mean_px=means['fused_error_px']-means['baseline_error_px']))
    assert {str(p): E.sha(p) for p in paths} == source
    E.write(out/'DIAGNOSIS.json', dict(complete=True, source_sha256=source, rows=rows,
        semantics='Post-selection GT error strata, not an inference gate. Each unique corner averages its three DHT-seed outcomes. No selection or calibration modification.',
        difficulty_thresholds_px=[10, 20]))
    print('Frozen selected-family GT-difficulty diagnosis complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
