"""Only two corrected CPU expectations, preserving the first 14-group attempt.

The original test source/solver/pipeline remain byte-frozen. No other twelve
groups are rerun. Three-point local fitting may correctly return ambiguity;
the packet group continues through NEW, native fallback and complete failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from . import test_solver as T
from . import pipeline as driver
from ..pallet_same_observation_controls_20261010_v8 import test_controls as T8
from ..pallet_same_observation_controls_20261010_v8.controls import semantic_sha256

DOC, PRIVATE, CODE, REPO = T.DOC, T.PRIVATE, T.CODE, T.REPO
TARGETS = ('test_no_unused_lines_three_actual_points_remains_LOCAL',
           'test_packet_wrapper_NEW_H_projection_and_full_fallback')


class ReviewContracts(unittest.TestCase):
    assert_local_contract = T.LocalContracts.assert_local_contract

    def test_no_unused_lines_three_actual_points_remains_LOCAL(self):
        U = (0, 1, 4)
        q, K, xyz, _, seed, _, _, _ = T.fixture(U)
        result = T.solve(q, K, xyz, dict(corners=[], lines=[], partial_lines=[]), seed)
        self.assertIn(result['state'], ('NEW_POSE', 'AMBIGUOUS_LOCAL_POINT_LINE'), result['optimizer_attempts'])
        self.assert_local_contract(result, U)
        self.assertEqual(result['line_edges'], [])
        self.assertEqual(result['factor_pool']['scalar_residuals'], 6)
        self.assertEqual(result['operation_counts']['local_optimizer_calls'], 2)
        self.assertEqual({e['dimension_index'] for e in result['optimizer_attempts']}, {0, 1})
        accepted = [c for c in result['all_candidate_solutions'] if c['accepted']]
        self.assertTrue(accepted)
        for c in accepted:
            self.assertEqual(c['fit_input_ids'], list(U))
            self.assertEqual(c['fit_line_edges'], [])
            self.assertEqual(c['point_line_geometry']['observed_normal_joint']['numerical_rank'], 6)
            self.assertEqual(c['point_line_geometry']['modeled_normal_joint']['numerical_rank'], 6)
        if result['state'] == 'AMBIGUOUS_LOCAL_POINT_LINE':
            self.assertFalse(result['available'])
            self.assertTrue(result['unresolved_ambiguity'])
            self.assertTrue(any(not a['numerically_equivalent_to_selected'] and a['numerical_objective_tie']
                                for a in result['alternatives']))
            self.assertFalse(result['hidden_reprojected'])
        else:
            self.assertTrue(result['available'])
            self.assertTrue(result['fit_has_fewer_than_four_actual_points'])
        T.DETAILS['review_three_point_LOCAL'] = dict(state=result['state'],
            valid_local_branches=len(accepted), guaranteed_NEW_claim=False,
            independent_P3P_or_PnP_claim=False, global_uniqueness_claim=False,
            both_dimensions_tried=True, operations=result['operation_counts'])

    def test_packet_wrapper_NEW_H_projection_and_full_fallback(self):
        U, H = (0, 1, 4), (7,)
        T.COUNTS['synthetic_parent_pose_paths'] += 1
        packet = T8.fixture(ids=U, H=H)
        parent = packet['parent_row']
        q = np.asarray(parent['input_points'], float)
        full = np.asarray(packet['native_N3_points'], float)
        raw = T.observation(full[:8], U)
        queries = [dict(query=i, edge=i//7, calibrated_model_coverage=i//7 in T.CAL_EDGES,
                        selected_xy=None) for i in range(84)]
        for line in raw['lines']:
            for query_id, xy in zip(line['queries'], line['support_points']):
                queries[query_id]['selected_xy'] = list(xy)
        parent['observation_contract']['corner_admission'] = [
            dict(id=c['id'], accepted=True, edges=c['edges']) for c in raw['corners']]
        packet['role_observation'] = dict(id=parent['id'], session=parent['session'], head_arm='IMAGE_ROLE',
            raw_logits_sha256=parent['observation_raw_logits_sha256'],
            initial_N3_pose=copy.deepcopy(parent['initial_pose']), predicted_N3_hidden=list(H),
            native_N3_points=full.tolist(), GT_input=False, corners=raw['corners'], lines=raw['lines'],
            partial_lines=[], queries=queries)
        missing_CAL = copy.deepcopy(packet)
        first_query = missing_CAL['role_observation']['lines'][0]['queries'][0]
        missing_CAL['role_observation']['queries'][first_query]['calibrated_model_coverage'] = False
        with self.assertRaises(ValueError):
            driver.validate_local_packet(missing_CAL)
        saved = semantic_sha256(packet)
        T.COUNTS['fixture_local_packet_paths'] += 1
        result, ledger = driver.solve_local_packet(packet)
        T.add_local_operations(result['solver'])
        self.assertEqual(semantic_sha256(packet), saved)
        np.testing.assert_array_equal(np.asarray(result['input_points'], float), q)
        np.testing.assert_array_equal(np.asarray(result['native_points'])[8], full[8])
        self.assertTrue(result['new_pose_estimated'], result['solver']['state'])
        self.assertFalse(result['reprojections_reused_as_observations'])
        self.assertEqual(result['reprojected_ids'], list(H))
        np.testing.assert_array_equal(np.asarray(result['native_points'])[7], result['solver']['projected'][7])
        self.assertFalse(set(result['solver']['fit_input_ids']) & set(H))
        for field in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
            np.testing.assert_allclose(result['actual_pose'][field], result['solver'][field], atol=0, rtol=0)
        def wrapper_failure(*args, **kwargs):
            T.COUNTS['injected_optimizer_entries'] += 1
            raise ValueError('synthetic wrapper failure after repaired NEW equality')
        with patch.object(T.S, 'least_squares', wrapper_failure):
            T.COUNTS['fixture_local_packet_paths'] += 1
            fallback, fallback_ledger = driver.solve_local_packet(packet)
        T.add_local_operations(fallback['solver'])
        self.assertFalse(fallback['new_pose_estimated'])
        self.assertTrue(fallback['fallback_used'])
        self.assertEqual(semantic_sha256(fallback['actual_pose']), semantic_sha256(parent['initial_pose']))
        np.testing.assert_array_equal(np.asarray(fallback['native_points'], float), full)
        self.assertEqual(fallback['reprojected_ids'], [])
        unavailable = copy.deepcopy(packet)
        unavailable['parent_row']['initial_pose']['available'] = False
        unavailable['parent_row'].update(actual_pose=copy.deepcopy(unavailable['parent_row']['initial_pose']),
            pose_available=False, no_pose=True, fallback_used=False, output_status='POSE_FAILURE')
        unavailable['role_observation']['initial_N3_pose'] = copy.deepcopy(unavailable['parent_row']['initial_pose'])
        T.COUNTS['fixture_local_packet_paths'] += 1
        failed, failed_ledger = driver.solve_local_packet(unavailable)
        T.add_local_operations(failed['solver'])
        self.assertFalse(failed['new_pose_estimated'])
        self.assertFalse(failed['fallback_used'])
        self.assertTrue(failed['no_pose'])
        self.assertEqual(failed['output_status'], 'POSE_FAILURE')
        self.assertEqual(semantic_sha256(packet), saved)
        T.DETAILS['review_packet'] = dict(real_assets=False, final_RT_solver_join=True,
            center_preserved=True, three_paths_completed=True,
            outcomes=[result['output_status'], fallback['output_status'], failed['output_status']],
            NEW_ledger=ledger, fallback_ledger=fallback_ledger, failure_ledger=failed_ledger)


def code_paths():
    paths = dict(solver=CODE/'solver.py', pipeline=CODE/'pipeline.py', tests=CODE/'test_solver.py',
        v5_solver=REPO/'scripts/research/pallet_partial_line_independent_20261010_v5/solver.py',
        v4_pose=REPO/'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py',
        v8_fixture=REPO/'scripts/research/pallet_same_observation_controls_20261010_v8/test_controls.py',
        v8_controls=REPO/'scripts/research/pallet_same_observation_controls_20261010_v8/controls.py',
        v1_solver=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
        v1_inference=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/inference.py',
        v1_common=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/common.py',
        v2_pipeline=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',
        v2_common=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/common.py')
    return dict(paths, review=Path(__file__))


def guard_output(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink output ancestry')
    path = path.resolve()
    if not (path.parent == DOC.resolve() or path.parent.is_relative_to(PRIVATE.resolve())):
        raise ValueError('only new V9 DOC/private outputs')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('preserve existing review/joined attempt')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DOC/'CPU_REVIEW_PROTOCOL.json')
    parser.add_argument('--original', type=Path, default=DOC/'LOCAL_LINE_CHECKS.json')
    parser.add_argument('--output', type=Path, default=DOC/'LOCAL_LINE_REVIEW_CHECKS.json')
    parser.add_argument('--joined', type=Path, default=DOC/'JOINED_LOCAL_CONTRACT_CHECKS.json')
    args = parser.parse_args()
    output, joined = guard_output(args.output), guard_output(args.joined)
    started = guard_output(output.with_name(output.stem+'_STARTED.json'))
    if len({output, joined, started}) != 3:
        raise ValueError('distinct exclusive output paths required')
    paths = code_paths(); code = {key: T.binding(path) for key, path in paths.items()}
    protocol = json.loads(args.protocol.read_text()); original = json.loads(args.original.read_text())
    protocol_binding, original_binding = T.binding(args.protocol), T.binding(args.original)
    original_started = args.original.with_name(args.original.stem+'_STARTED.json')
    old_started = json.loads(original_started.read_text())
    old_started_binding = T.binding(original_started)
    original_code = {k: v for k, v in code.items() if k != 'review'}
    original_protocol_path = REPO/original['protocol']['path']
    old_protocol = json.loads(original_protocol_path.read_text())
    original_protocol_binding = T.binding(original_protocol_path)
    unsuccessful = original['failures'] + original['errors']
    names = [item['test'].split(' ')[0] for item in unsuccessful]
    old_valid = (original['schema'] == 'sparse_local_line_CPU_checks_v9' and original['complete'] is True and
        original['passed'] is False and original['tests_run'] == 14 and len(original['failures']) == 1 and
        len(original['errors']) == 1 and sorted(names) == sorted(TARGETS) and original['code'] == original_code and
        old_started['code'] == original_code and original['code_unchanged_after_tests'] is True and
        original['protocol_unchanged_after_tests'] is True and original['production_local_optimizer_dispatch_count_join'] is True and
        original['cleanup_errors'] == [] and original['exception'] is None and
        old_protocol['code'] == original_code and original_protocol_binding == original['protocol'])
    if (protocol['schema'] != 'pre_execution_synthetic_sparse_local_line_CPU_review_protocol_v9' or
        protocol['code'] != code or protocol['original_receipt'] != original_binding or
        protocol['synthetic_only'] is not True or protocol['test_groups_expected'] != 2 or
        protocol['targeted_test_names'] != list(TARGETS) or not old_valid):
        raise ValueError('review prefreeze/original12PASS evidence differs before fixture arithmetic')
    T.write_new(started, dict(schema='sparse_local_line_CPU_review_started_v9', code=code,
        protocol=protocol_binding, original_receipt=original_binding, original_started=old_started_binding,
        original_protocol=original_protocol_binding, targeted_groups=2, other_twelve_rerun=False,
        synthetic_only=True, automatic_retry=False))
    T.COUNTS.clear(); T.DETAILS.clear()
    old_threads = cv2.getNumThreads(); cv2.setNumThreads(1)
    primitives = Counter({name: 0 for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM',
        'projectPoints', 'Rodrigues', 'cornerSubPix')})
    originals = {name: getattr(cv2, name) for name in primitives}
    scipy_original = T.S.least_squares
    def real_optimizer(*a, **kw):
        T.COUNTS['actual_real_SciPy_least_squares_entries'] += 1
        return scipy_original(*a, **kw)
    T.S.least_squares = real_optimizer
    for name, fn in originals.items():
        def counted(*a, _name=name, _fn=fn, **kw):
            primitives[_name] += 1
            return _fn(*a, **kw)
        setattr(cv2, name, counted)
    result, failure, cleanup_errors, reads = None, None, [], []
    began = time.monotonic()
    try:
        with T.no_external_assets() as reads:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReviewContracts)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        try:
            T.S.least_squares = scipy_original
            for name, fn in originals.items():
                setattr(cv2, name, fn)
            cv2.setNumThreads(old_threads)
        except BaseException as error:
            cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
    unchanged = {k: T.binding(p) for k, p in paths.items()} == code
    preserved_old = (T.binding(args.original) == original_binding and T.binding(original_started) == old_started_binding and
                     T.binding(original_protocol_path) == original_protocol_binding)
    protocol_unchanged = T.binding(args.protocol) == protocol_binding
    dispatch_join = T.COUNTS['local:local_optimizer_calls'] == (
        T.COUNTS['actual_real_SciPy_least_squares_entries']+T.COUNTS['injected_optimizer_entries'])
    complete = result is not None and failure is None and not cleanup_errors
    passed = (complete and result.wasSuccessful() and result.testsRun == 2 and not reads and
              unchanged and protocol_unchanged and preserved_old and dispatch_join)
    receipt = dict(schema='sparse_local_line_CPU_review_checks_v9', complete=complete, passed=passed,
        tests_run=result.testsRun if result else 0, targeted_test_names=list(TARGETS), other_twelve_rerun=False,
        failures=[dict(test=str(test), traceback=trace) for test, trace in result.failures] if result else [],
        errors=[dict(test=str(test), traceback=trace) for test, trace in result.errors] if result else [],
        exception=failure, cleanup_errors=cleanup_errors, code=code, protocol=protocol_binding,
        original_receipt=original_binding, original_started=old_started_binding, original_protocol=original_protocol_binding,
        original_attempt_preserved_unchanged=preserved_old, code_unchanged_after_tests=unchanged,
        protocol_unchanged_after_tests=protocol_unchanged, production_local_optimizer_dispatch_count_join=dispatch_join,
        actual_OpenCV_entry_calls=dict(primitives), actual_logical_calls=dict(T.COUNTS), details=T.DETAILS,
        forbidden_asset_reads=reads, wall_seconds=time.monotonic()-began,
        new_model_real_GT_image_training_RGB_calls=0, latency_benchmark=False, automatic_retry=False)
    T.write_new(output, receipt)
    T.write_new(joined, dict(schema='sparse_local_line_joined_CPU_contract_checks_v9', complete=complete,
        passed=passed and old_valid, code=code, original_protocol=original_protocol_binding,
        original_receipt=original_binding, original_started=old_started_binding,
        review_protocol=protocol_binding, review_receipt=T.binding(output), review_started=T.binding(started),
        original_attempt_tests=14, original_attempt_passed_groups=12, original_attempt_failed_groups=2,
        original_attempt_reported_passed=False, review_targeted_tests=2, contract_groups=14,
        accepted_original_unchanged_groups=12, accepted_review_corrected_groups=2 if passed else 0,
        other_twelve_rerun=False, solver_or_threshold_changed=False, review_is_new_performance_tuning=False,
        original_attempt_preserved_unchanged=preserved_old, cleanup_errors=cleanup_errors,
        new_model_real_GT_image_training_RGB_calls=0, automatic_retry=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
