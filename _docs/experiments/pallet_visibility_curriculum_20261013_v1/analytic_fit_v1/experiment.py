"""CAL-only derivative/solver optimization audit, without scoring annotations."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import cv2
import numpy as np

import curriculum_analytic as experimental
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as baseline
from scripts.research.pallet_visibility_curriculum_20261013_v1.test_live_boundary import production_access_canary

OUT = Path(__file__).resolve().parent
CONFIG = dict(visible_threshold=.35)


def unchanged_outside_fit():
    old_path = C.ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/curriculum.py'
    old_tree = ast.parse(old_path.read_text())
    new_tree = ast.parse((OUT / 'curriculum_analytic.py').read_text())
    for tree in (old_tree, new_tree):
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != '_fit']
    return ast.dump(old_tree) == ast.dump(new_tree)


def initial_arguments(frame, row):
    pose = frame['initial_poses']['N3_SUBPIX']
    return (np.asarray(row['original_points'], float), C.cuboid(*pose['cf_extents']),
            np.asarray(frame['K'], float), np.asarray(pose['R_cf'], float),
            np.asarray(pose['centroid'], float), np.asarray(row['self_hidden'], bool),
            deepcopy(row['appearance']), CONFIG)


def pose_difference(old, new):
    return dict(max_R_element=float(np.max(np.abs(np.asarray(old['R']) - new['R']))),
                max_t_element=float(np.max(np.abs(np.asarray(old['t']) - new['t']))),
                max_projected_coordinate_px=float(np.max(np.abs(np.asarray(old['projected']) - new['projected']))))


def numerical_diff(old, new, key):
    if key not in old or key not in new or old[key] is None or new[key] is None:
        return None
    difference = np.abs(np.asarray(old[key], float) - np.asarray(new[key], float))
    return float(np.max(difference)) if difference.size else 0.


def milliseconds(values):
    values = np.asarray(values, float) * 1000
    return dict(mean=float(values.mean()), median=float(np.median(values)), p90=float(np.percentile(values, 90)))


class Counters:
    def __init__(self, module):
        self.module = module
        self.counts = Counter()

    def __enter__(self):
        self.original_ls = self.module.least_squares
        self.original_fit = self.module._fit

        def least_squares(fun, *args, **kwargs):
            self.counts['least_squares_calls'] += 1

            def counted_fun(params):
                self.counts['residual_calls'] += 1
                return fun(params)

            if callable(kwargs.get('jac')):
                original_jac = kwargs['jac']

                def counted_jac(params):
                    self.counts['analytic_jacobian_calls'] += 1
                    return original_jac(params)
                kwargs['jac'] = counted_jac
            return self.original_ls(counted_fun, *args, **kwargs)

        def fit(*args):
            self.counts['fit_calls'] += 1
            return self.original_fit(*args)

        self.module.least_squares = least_squares
        self.module._fit = fit
        return self.counts

    def __exit__(self, *unused):
        self.module.least_squares = self.original_ls
        self.module._fit = self.original_fit


def main():
    cv2.setNumThreads(1)
    source = C.ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/curriculum.py'
    inputs = C.SCRATCH / 'CAL_INPUTS.json'
    cache = C.SCRATCH / 'cal_segmentation_v2/PREDICTIONS.jsonl.gz'
    hashes = {str(p): C.sha(p) for p in (source, inputs, cache)}
    frames = C.read(inputs)['frames']
    cached = {r['id']: r for r in C.rows(cache) if r['method'] == 'SEGMENTED_RGB_CURRICULUM'}
    assert len(frames) == len(cached) == 76
    assert all(f['partition'] == 'CAL' for f in frames)
    assert unchanged_outside_fit()
    arguments = [initial_arguments(frame, cached[frame['id']]) for frame in frames]
    preserved = deepcopy(arguments)
    rows, stage_diffs, changes, cache_diffs = [], [], [], []
    with production_access_canary():
        for frame, args in zip(frames, arguments):
            old = baseline.solve_curriculum(*args)
            new = experimental.solve_curriculum(*args)
            expected = cached[frame['id']]['solution']
            cache_diffs.append(pose_difference(expected, old))
            assert len(old['stages']) == len(new['stages'])
            signatures = ('name', 'accepted', 'reason', 'used_ids', 'added_ids',
                          'heldout_ids', 'protected_visible_ids')
            for index, (previous, current) in enumerate(zip(old['stages'], new['stages'])):
                state_changes = {k: [previous.get(k), current.get(k)] for k in signatures if previous.get(k) != current.get(k)}
                if state_changes:
                    changes.append(dict(id=frame['id'], stage=index, changes=state_changes))
                differences = pose_difference(previous, current)
                for key in ('fit_score_after', 'visible_score_after', 'protected_score_after',
                            'residual_fit_after', 'residual_visible_after', 'residual_heldout_after',
                            'candidate_R', 'candidate_t'):
                    differences[key] = numerical_diff(previous, current, key)
                stage_diffs.append(dict(id=frame['id'], stage=index, **differences))
            for arm, solution in [('FINITE_DIFFERENCE', old), ('ANALYTIC_JACOBIAN', new)]:
                rows.append(dict(id=frame['id'], partition='CAL', arm=arm, solution=solution))
            assert old['observed_points'] == new['observed_points']
            assert old['accepted'] == new['accepted']
            assert old['used_ids'] == new['used_ids']
            assert not set(new['used_ids']) & set(np.flatnonzero(args[5]))

        # Every input receives one complete warmup per arm. The same 76-frame
        # population is then evaluated three times; arm order alternates by
        # repeat/frame. Timings include geometry, validation and stages.
        for args in arguments:
            baseline.solve_curriculum(*args)
            experimental.solve_curriculum(*args)
        elapsed = {arm: [] for arm in ('FINITE_DIFFERENCE', 'ANALYTIC_JACOBIAN')}
        measured = []
        with Counters(baseline) as old_counts, Counters(experimental) as new_counts:
            for repeat in range(3):
                for frame_index, (frame, args) in enumerate(zip(frames, arguments)):
                    arms = [('FINITE_DIFFERENCE', baseline), ('ANALYTIC_JACOBIAN', experimental)]
                    if (repeat + frame_index) % 2:
                        arms.reverse()
                    for arm, module in arms:
                        started = time.perf_counter()
                        module.solve_curriculum(*args)
                        duration = time.perf_counter() - started
                        elapsed[arm].append(duration)
                        measured.append(dict(id=frame['id'], repeat=repeat, arm=arm, seconds=duration))

    for current, original in zip(arguments, preserved):
        for a, b in zip(current[:6], original[:6]):
            np.testing.assert_array_equal(a, b)
        assert current[6:] == original[6:]
    assert hashes == {path: C.sha(path) for path in hashes}
    maxima = {key: max((r[key] for r in stage_diffs if r.get(key) is not None), default=0.)
              for key in stage_diffs[0] if key not in ('id', 'stage')}
    timing = {arm: milliseconds(values) for arm, values in elapsed.items()}
    result = dict(
        scope='CAL76 optimization parity; no pose/visibility annotations or TEST inputs',
        source_main_unchanged=True, source_AST_unchanged_except_fit=True,
        input_sha256=hashes, configuration=CONFIG, frame_count=76,
        acceptance_count={arm: sum(r['solution']['accepted'] for r in rows if r['arm'] == arm) for arm in elapsed},
        stage_count=len(stage_diffs), stage_or_used_id_changes=changes,
        max_stage_numerical_differences=maxima,
        baseline_vs_frozen_cache_max={key: max(d[key] for d in cache_diffs) for key in cache_diffs[0]},
        timings_ms=timing, repetitions=3, calls_per_arm=228, warmups_per_input_per_arm=1,
        median_speedup=timing['FINITE_DIFFERENCE']['median']/timing['ANALYTIC_JACOBIAN']['median'],
        mean_speedup=timing['FINITE_DIFFERENCE']['mean']/timing['ANALYTIC_JACOBIAN']['mean'],
        optimizer_call_counts={'FINITE_DIFFERENCE':dict(old_counts),'ANALYTIC_JACOBIAN':dict(new_counts)},
        inference_annotation_and_cache_access_canary='PASS', input_nonmutation='PASS',
        acceptance_policy='Any changed stage decision prevents default integration pending review',
        integration_eligible_no_acceptance_changes=not changes,
    )
    C.save_rows(OUT/'PREDICTIONS.jsonl.gz',rows)
    C.save_rows(OUT/'STAGE_DIFFERENCES.jsonl.gz',stage_diffs)
    C.save_rows(OUT/'TIMINGS.jsonl.gz',measured)
    C.write(OUT/'VERIFICATION.json',result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
