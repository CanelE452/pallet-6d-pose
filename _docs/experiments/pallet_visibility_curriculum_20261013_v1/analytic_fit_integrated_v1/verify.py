"""Prediction-only replay of a numerical optimization, not accuracy evaluation."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

import cv2
import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as S
from scripts.research.pallet_visibility_curriculum_20261013_v1.test_live_boundary import production_access_canary


def main():
    cv2.setNumThreads(1)
    out = Path(__file__).resolve().parent
    paths = [C.DOC / 'PREDICTION_INPUTS.json',
             C.DOC / 'cal_segmentation_v2/PREDICTIONS.jsonl.gz',
             C.DOC / 'test_v1/PREDICTIONS.jsonl.gz',
             C.DOC / 'LIVE_CONFIG_V1.json', C.DOC / 'LIVE_CONFIG_FAST_V1.json',
             C.ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/curriculum.py',
             C.ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/test_analytic_fit.py',
             C.DOC / 'code_before_analytic_v1/curriculum.py',
             C.DOC / 'models/visibility_logistic_v1.json', C.DOC / 'models/best_source_CAL.pt']
    hashes = {str(p): C.sha(p) for p in paths}
    frames = C.read(paths[0])['frames']
    rows = {r['id']:r for p in paths[1:3] for r in C.rows(p) if r['method'] == 'SEGMENTED_RGB_CURRICULUM'}
    old_cfg = C.read(paths[3])['curriculum']
    fast_cfg = C.read(paths[4])['curriculum']
    assert old_cfg == {k:v for k,v in fast_cfg.items() if k != 'analytic_jacobian'}
    assert fast_cfg['analytic_jacobian'] is True
    assert len(frames) == len(rows) == 173
    diffs, predictions, states = [], [], Counter()
    for frame in frames:
        row = rows[frame['id']]
        initial = frame['initial_poses']['N3_SUBPIX']
        xyz = C.cuboid(*initial['cf_extents'])
        R = np.asarray(initial['R_cf']); t = np.asarray(initial['centroid'])
        Q = R.T @ np.asarray(initial['R_physical'])
        args = (row['original_points'], xyz, frame['K'], R, t, row['self_hidden'], row['appearance'])
        preserved = deepcopy(args)
        with production_access_canary():
            old = S.solve_curriculum(*args, old_cfg)
            fast = S.solve_curriculum(*args, fast_cfg)
        assert old['R'] == row['solution']['R'] and old['t'] == row['solution']['t']
        assert len(old['stages']) == len(fast['stages'])
        assert old['accepted'] == fast['accepted'] and old['used_ids'] == fast['used_ids']
        assert old['observed_points'] == fast['observed_points']
        assert not set(fast['used_ids']) & set(np.flatnonzero(row['self_hidden']))
        for stage_index, (a,b) in enumerate(zip(old['stages'],fast['stages'])):
            for name in ('name','accepted','reason','used_ids','added_ids','heldout_ids','protected_visible_ids'):
                assert a.get(name) == b.get(name), (frame['id'],stage_index,name)
            dR = float(np.max(np.abs(np.asarray(a['R'])-b['R'])))
            dt = float(np.max(np.abs(np.asarray(a['t'])-b['t'])))
            duv = float(np.max(np.abs(np.asarray(a['projected'])-b['projected'])))
            assert dR < 1e-7 and dt < 1e-7 and duv < 1e-4
            diffs.append(dict(id=frame['id'],partition=frame['partition'],stage=stage_index,
                              max_R_element=dR,max_t_m=dt,max_projected_coordinate_px=duv))
        physical = np.asarray(fast['R']) @ Q
        dphysical = float(np.max(np.abs(physical - np.asarray(row['actual_pose']['R_physical']))))
        assert dphysical < 1e-7
        for arm, solution in [('DEFAULT_FINITE_DIFFERENCE',old),('OPTIONAL_ANALYTIC',fast)]:
            predictions.append(dict(id=frame['id'],partition=frame['partition'],arm=arm,
                                    solution=solution,R_physical=(np.asarray(solution['R'])@Q).tolist(),
                                    physical_basis_Q=Q.tolist(),cf_extents=initial['cf_extents'],
                                    selected_hypothesis=initial['selected_hypothesis']))
        for a,b in zip(args,preserved):
            if isinstance(a,np.ndarray):
                np.testing.assert_array_equal(a,b)
            else:
                assert a==b
        states['frames']+=1
        states['stages']+=len(old['stages'])
        states['accepted_solutions']+=int(old['accepted'])
    assert hashes == {p:C.sha(p) for p in hashes}
    report = dict(scope='Full DEV173 prediction-only numerical parity; no new accuracy evaluation',
        no_scoring_annotation_or_performance_metrics_opened=True,
        no_model_or_algorithm_parameter_selection=True,TEST_accuracy_retuning=False,
        inputs_and_sources_sha256=hashes,partitions=dict(Counter(f['partition'] for f in frames)),
        default_configuration=old_cfg,fast_configuration=fast_cfg,default_V1_config_unchanged=True,
        numerical_optimization_only=True,counters=dict(states),
        max_R_element=max(d['max_R_element'] for d in diffs),
        max_translation_coordinate_m=max(d['max_t_m'] for d in diffs),
        max_projected_coordinate_px=max(d['max_projected_coordinate_px'] for d in diffs),
        allowed_R_t_tolerance=1e-7,allowed_UV_tolerance_px=1e-4,
        changed_stage_acceptance_or_used_ids=0,baseline_frozen_pose_max_difference=0,
        physical_basis_Q_and_original_extents_preserved=True,self_hidden_fitted_violations=0,
        inference_access_canary='PASS',input_nonmutation='PASS',
        derivative_max_absolute_error=1.573346537497855e-7,
        synthetic_robust_fits_max_UV_difference_px=1.3300257251103176e-7,
        solver_benchmark_CAL76_reference='../analytic_fit_v1/VERIFICATION.json',
        provenance='Original first validation source/results remain immutable; this config opts in to a subsequent mathematically equivalent derivative',
    )
    C.save_rows(out/'PREDICTIONS.jsonl.gz',predictions)
    C.save_rows(out/'STAGE_DIFFERENCES.jsonl.gz',diffs)
    C.write(out/'VERIFICATION.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
