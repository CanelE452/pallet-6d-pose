"""Meaningful synthetic CPU contracts for fixed same-observation controls.

Code construction is not execution. Invoke only after root review and freezing:
python -B -m scripts.research.pallet_same_observation_controls_20261010_v8.test_controls
    --output /new/private/path/CONTROL_CHECKS.json

Fixtures are explicitly mathematical: no real images, learned model, stored
observations, source arrays, GT file, ray, training or timing is accessed.
Actual frozen V4 numeric solvers run when this suite is deliberately invoked.
The injected SSE-tie fixture is marked separately from natural IPPE returns.
Every OpenCV entry and fixture/control logical solve is recorded in the new
receipt; failures and the started receipt are preserved without overwrite.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode = True
import cv2
import numpy as np

from . import controls as C
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble
from ..pallet_observation_refiner_20261009_v1.solver import Candidate, cuboid, project

COUNTS = Counter()
DETAILS = {}
SIZE = (1280, 960)


def mathematical_geometry():
    K = np.array([[600., 0., 640.], [0., 600., 480.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.3])
    R = cv2.Rodrigues(np.array([.3, .25, .1]))[0]
    t = np.array([0., .1, 4.])
    native = np.vstack([project(cuboid(*xyz), R, t, K), [641.25, 481.75]])
    initial = dict(available=True, R_cf=R.tolist(), R_physical=R.tolist(),
                   centroid=t.tolist(), cf_extents=xyz.tolist(), selected_hypothesis='REGISTRY_WD')
    return K, xyz, R, t, native, initial


def direct_solve(bank, **kwargs):
    COUNTS['fixture_direct_pose_paths'] += 1
    return bank.solve(**kwargs)


def control_solve(packet):
    results, ledger = C.solve_controls(packet)
    COUNTS['control_logical_pose_paths'] += ledger['logical_pose_paths']
    COUNTS['control_replay_pose_paths'] += ledger['existing_control_replay_pose_paths']
    COUNTS['control_new_diagnostic_pose_paths'] += ledger['new_diagnostic_pose_paths']
    return results, ledger


def fixture(ids=tuple(range(8)), H=(6,), corrupt=(), native_hidden_shift=False,
            alternate_initial=False):
    """Build a complete synthetic packet, not a declaration about real assets."""
    K, xyz, R, t, native, initial = mathematical_geometry()
    if native_hidden_shift:
        native[list(H)] += np.array([13., -9.])
    if alternate_initial:
        initial['R_cf'] = cv2.Rodrigues(np.array([-.3, .6, .2]))[0].tolist()
        initial['R_physical'] = initial['R_cf']
        initial['centroid'] = [.2, -.1, 5.]
    sparse = np.full((9, 2), np.nan)
    truth = project(cuboid(*xyz), R, t, K)
    sparse[list(ids)] = truth[list(ids)]
    sparse[8] = native[8]
    for k, offset in corrupt:
        sparse[k] += np.asarray(offset)
    contract = dict(validated_boundary_corner_ids=sorted(ids), hybrid_boundary_corner_ids=[],
        missing_ROLE_ONLY_corner_is_NaN=True, boundary_only_native_distance_used_for_admission=False,
        boundary_only_LOO_used_for_admission=False, native_N3_used_for_numeric_final_fit=False,
        heldout_pose_calls=0)
    solved = direct_solve(PoseBank(sparse, K, xyz, image_size=SIZE), hidden=list(H), robust=True)
    parent = assemble(native, sparse, initial, list(H), solved, 'VALIDATED_ROLE_ONLY', contract)
    parent.update(id='synthetic:fixed', session='synthetic', method=C.PARENT_METHOD,
        K=K.tolist(), xyz=xyz.tolist(), raw_hw=[SIZE[1], SIZE[0]], selected_index=0,
        fixed_metadata=dict(candidate_metadata=dict(score=.9, cls=0), preserved=True),
        head_arm='IMAGE_ROLE', head_mode='IMAGE_ROLE', observation_supply='BOUNDARY_ONLY',
        head_checkpoint_sha256=C.ROLE_CHECKPOINT_SHA256,
        calibration_binding=dict(path='synthetic/CALIBRATION.json', sha256='1'*64, bytes=0),
        selected_corner_ids=sorted(ids), predicted_initial_N3_hidden=list(H),
        partial_lines_not_pose_inputs=True, final_numeric_pose_has_initial_prior=False,
        observation_raw_logits_sha256='2'*64, selected_queries=0, source_lines=[], oracle=False)
    completion = dict(complete=True, frames=245, geometry_rows=1960, observation_rows=735,
        fixed_rows=490, cleanup_error=None, protocol_sha256='3'*64, geometry_sha256='4'*64,
        inference_receipt_sha256='5'*64, synthetic_fixture_declaration=True)
    return dict(parent_row=parent, native_N3_points=native.tolist(), parent_completion=completion)


@contextmanager
def no_external_assets():
    saved = [(builtins, 'open', builtins.open), (io, 'open', io.open), (Path, 'open', Path.open)]
    attempts = []
    tokens = ('TARGETS', 'GROUND_TRUTH', 'STATIC_VISIBILITY', 'READY_SOURCE', 'FEATURES.NPY',
              'PREDICTIONS.JSON', 'OBSERVATIONS.JSON', 'INPUTS.JSON', 'BEST.PT', 'LAST.PT',
              'IMAGE_ROLE.PT', 'GEOMETRY_ONLY.PT', 'IMAGE_NO_ROLE.PT', '.PNG', '.JPG', '.JPEG')
    def guarded(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and any(t in str(path).upper() for t in tokens):
                attempts.append(str(path))
                raise AssertionError('synthetic suite attempted external asset read')
            return fn(path, *args, **kwargs)
        return call
    for obj, name, fn in saved:
        setattr(obj, name, guarded(fn))
    try:
        yield attempts
    finally:
        for obj, name, fn in saved:
            setattr(obj, name, fn)


class ControlContracts(unittest.TestCase):
    def test_fixed_three_routes_without_model_or_decoder(self):
        self.assertEqual(C.METHODS, ('ROLE_BOUNDARY_H_ROBUST', 'ROLE_BOUNDARY_H_STANDARD',
                                     'ROLE_BOUNDARY_NO_MASK_ROBUST'))
        self.assertTrue(C.POLICY['primary_is_unchanged_parent_diagnostic_control'])
        self.assertFalse(C.POLICY['controls_change_observation_admission'])
        self.assertFalse(C.POLICY['latency_claim'])
        packet = fixture()
        before = C.semantic_sha256(packet)
        rows, ledger = control_solve(packet)
        self.assertEqual([r['method'] for r in rows], list(C.METHODS))
        self.assertEqual(ledger['logical_pose_paths'], 3)
        self.assertEqual(ledger['existing_control_replay_pose_paths'], 1)
        self.assertEqual(ledger['new_diagnostic_pose_paths'], 2)
        self.assertEqual(before, C.semantic_sha256(packet))
        self.assertTrue(ledger['parent_replay_parity']['passed'])
        for row in rows:
            self.assertEqual(row['head_forward_calls'], 0)
            self.assertEqual(row['calibration_or_decode_calls'], 0)
            self.assertFalse(row['solver']['known_dimension_constraint_used'])
            self.assertFalse(row['solver']['prior_used'])
            self.assertEqual(row['fixed_metadata'], packet['parent_row']['fixed_metadata'])

    def test_actual_four_five_standard_all_dimensions_and_IPPE_branches(self):
        details = []
        for ids in ((0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 4, 5)):
            packet = fixture(ids=ids, H=())
            rows, _ = control_solve(packet)
            parent = packet['parent_row']
            expected = direct_solve(PoseBank(parent['input_points'], parent['K'], parent['xyz'], image_size=SIZE),
                                    hidden=[], robust=False)
            C.solver_parity(rows[1]['solver'], expected)
            actual = rows[1]['solver']
            self.assertEqual(actual['used'], list(ids))
            self.assertFalse(actual['known_dimension_constraint_used'])
            self.assertFalse(actual['prior_used'])
            self.assertEqual({c['dimension_index'] for c in actual['all_candidate_solutions']}, {0, 1})
            if ids == (0, 1, 2, 3):
                roots = [c for c in actual['all_candidate_solutions']
                         if c['dimension_index'] == 0 and c['generator'] == 'IPPE_PLANE']
                self.assertGreaterEqual(len({c['solution_index'] for c in roots}), 2)
            if actual['available']:
                self.assertGreaterEqual(actual['geometry']['jacobian']['numerical_rank'], 6)
            details.append(dict(ids=list(ids), state=actual['state'],
                candidate_count=len(actual['all_candidate_solutions']),
                multiple_solutions=actual['multiple_solutions'], unresolved_ambiguity=actual['unresolved_ambiguity']))
        # A mathematical five-point plane is not five native cuboid face IDs.
        K, _, R, t, _, _ = mathematical_geometry()
        plane = np.array([[-.5, -.4, 0.], [.5, -.4, 0.], [.5, .4, 0.], [-.5, .4, 0.], [.15, .05, 0.]])
        for n in (4, 5):
            measured = project(plane[:n], R, t, K)
            ret = cv2.solvePnPGeneric(plane[:n], measured, K, None, flags=cv2.SOLVEPNP_IPPE)
            self.assertTrue(ret[0])
            self.assertGreaterEqual(len(ret[1]), 2)
            errors = [np.max(np.abs(project(plane[:n], cv2.Rodrigues(rv)[0], tv, K)-measured))
                      for rv, tv in zip(ret[1], ret[2])]
            self.assertLess(min(errors), 1e-6)
        DETAILS['four_five_standard'] = dict(native_cases=details, natural_IPPE4_5_returned_branches=True,
                                             mathematical_fifth_plane_point_not_native_corner=True)

    def test_fewer_than_four_and_nonfinite_sparse_not_filled(self):
        packet = fixture(ids=(0, 1, 2), H=(6,))
        rows, ledger = control_solve(packet)
        for row in rows:
            self.assertFalse(row['new_pose_estimated'])
            self.assertTrue(row['fallback_used'])
            self.assertFalse(row['hidden_reprojected'])
            self.assertEqual(row['solver']['state'], 'INSUFFICIENT_OBSERVATIONS')
            self.assertTrue(np.isnan(np.asarray(row['input_points'])[3:8]).all())
            self.assertTrue(np.array_equal(row['native_points'], packet['native_N3_points'], equal_nan=True))
            self.assertFalse(row['missing_sparse_filled_for_numeric_fit'])
        self.assertEqual(ledger['coordinate_banks'][0]['generic_calls'], 0)
        bad = copy.deepcopy(packet)
        bad['parent_row']['input_points'] = np.asarray(bad['parent_row']['input_points']).tolist()
        bad['parent_row']['input_points'][0][1] = None
        with self.assertRaisesRegex(ValueError, 'missing sparse coordinates'):
            C.validate_parent(bad)
        bad = copy.deepcopy(packet)
        bad['parent_row']['input_points'] = np.asarray(bad['parent_row']['input_points']).tolist()
        bad['parent_row']['input_points'][3] = packet['native_N3_points'][3]
        with self.assertRaisesRegex(ValueError, 'missing sparse coordinates'):
            C.validate_parent(bad)

    def test_shared_bank_mask_exclusion_and_NEW_H_reprojection_without_refit(self):
        packet = fixture(H=(6,), corrupt=((6, (30., 20.)),), native_hidden_shift=True)
        rows, ledger = control_solve(packet)
        self.assertEqual(ledger['coordinate_bank_count'], 1)
        self.assertLessEqual(ledger['coordinate_banks'][0]['subsets_considered'], 140)
        self.assertGreater(ledger['coordinate_banks'][0]['generic_cache_hits'], 0)
        self.assertFalse(ledger['H_and_no_H_used_U_equal'])
        self.assertEqual(ledger['actual_H_removed_sparse_ids'], [6])
        for row in rows[:2]:
            self.assertTrue(row['new_pose_estimated'])
            for field in ('used', 'fit_input_ids', 'final_inliers', 'generator_ids'):
                self.assertNotIn(6, row['solver'].get(field, []))
            for candidate in row['solver']['all_candidate_solutions']:
                self.assertNotIn(6, candidate['generator_ids'])
                self.assertNotIn(6, candidate['actual_fit_input_ids'])
            self.assertEqual(row['reprojected_ids'], [6])
            self.assertTrue(np.allclose(np.asarray(row['native_points'])[6],
                                       np.asarray(row['actual_pose']['projected'])[6], rtol=0, atol=1e-9))
            self.assertFalse(row['reprojections_reused_as_observations'])
            self.assertTrue(np.array_equal(np.asarray(row['native_points'])[8], packet['native_N3_points'][8]))
        self.assertEqual(rows[2]['hidden_initial'], [])
        self.assertEqual(rows[2]['predicted_initial_N3_hidden'], [6])
        self.assertEqual(rows[2]['reprojected_ids'], [])
        self.assertIn(6, rows[2]['solver']['used'])
        DETAILS['shared_mask_bank'] = ledger['coordinate_banks'][0]

    def test_equal_U_records_same_pose_but_distinct_hidden_display(self):
        packet = fixture(ids=(0, 1, 2, 3, 4, 5, 7), H=(6,), native_hidden_shift=True)
        rows, ledger = control_solve(packet)
        self.assertTrue(ledger['H_and_no_H_used_U_equal'])
        self.assertTrue(ledger['H_and_no_H_equal_U_numeric_parity']['passed'])
        self.assertEqual(ledger['actual_H_removed_sparse_ids'], [])
        self.assertTrue(rows[0]['new_pose_estimated'])
        self.assertTrue(rows[2]['new_pose_estimated'])
        self.assertEqual(rows[0]['reprojected_ids'], [6])
        self.assertEqual(rows[2]['reprojected_ids'], [])
        self.assertFalse(np.allclose(np.asarray(rows[0]['native_points'])[6], np.asarray(rows[2]['native_points'])[6]))
        self.assertTrue(np.array_equal(np.asarray(rows[2]['native_points'])[6], packet['native_N3_points'][6]))
        for row in rows:
            self.assertTrue(np.isnan(np.asarray(row['input_points'])[6]).all())

    def test_native_hidden_and_initial_numeric_pose_do_not_enter_final_solve(self):
        packets = [fixture(ids=(0, 1, 2, 3, 4, 5, 7), H=(6,), native_hidden_shift=shift,
                           alternate_initial=shift) for shift in (False, True)]
        first, _ = control_solve(packets[0])
        second, _ = control_solve(packets[1])
        for a, b in zip(first, second):
            self.assertTrue(a['new_pose_estimated'])
            C.solver_parity(a['solver'], b['solver'])
            self.assertFalse(a['native_N3_used_as_independent_RGB_observations'])
            self.assertFalse(b['native_N3_used_as_independent_RGB_observations'])
        DETAILS['conditional_numeric_invariance'] = dict(fixed_q_H=True, changed_native_H_and_initial_R_t=True,
            full_feature_or_mask_statistical_independence_claim=False)

    def test_one_two_wrong_points_robust_diagnostics_and_ordinary_whole_U(self):
        cases = []
        for corrupt in (((0, (24., -16.)),), ((0, (24., -16.)), (1, (-20., 28.)))):
            packet = fixture(H=(), corrupt=corrupt)
            rows, _ = control_solve(packet)
            robust, ordinary = rows[0]['solver'], rows[1]['solver']
            self.assertEqual(ordinary['used'], list(range(8)))
            if ordinary['available']:
                self.assertEqual(ordinary['fit_input_ids'], list(range(8)))
                self.assertAlmostEqual(ordinary['sse_px2'], sum(r*r for r in ordinary['residuals_used_px']), places=7)
            if robust['available']:
                self.assertGreaterEqual(len(robust['final_inliers']), 4)
            if len(corrupt) == 1:
                self.assertTrue(robust['available'])
                self.assertNotIn(0, robust['final_inliers'])
                K, xyz, R, t, _, _ = mathematical_geometry()
                truth = project(cuboid(*xyz), R, t, K)
                good = list(range(1, 8))
                robust_error = float(np.linalg.norm(np.asarray(robust['projected'])[good] - truth[good], axis=1).mean())
                self.assertLess(robust_error, 1e-5)
                if ordinary['available']:
                    ordinary_error = float(np.linalg.norm(np.asarray(ordinary['projected'])[good] - truth[good], axis=1).mean())
                    self.assertLess(robust_error, ordinary_error)
            cases.append(dict(bad_ids=[k for k, _ in corrupt], robust_state=robust['state'],
                              ordinary_state=ordinary['state'], robust_final_inliers=robust['final_inliers'],
                              ordinary_diagnostic_inliers=ordinary['final_inliers']))
        DETAILS['controlled_wrong_points'] = dict(cases=cases, universal_two_error_recovery_asserted=False)

    def test_ordinary_tied_SSE_ignores_different_inlier_counts_and_marks_ambiguity(self):
        K, xyz, R, t, _, _ = mathematical_geometry()
        ids = (0, 1, 2, 4, 5)
        model = cuboid(*xyz)
        projected_a = project(model, R, t, K)
        projected_b = project(model, R, t + [.1, 0., 0.], K)
        d2 = ((projected_b[list(ids)] - projected_a[list(ids)])**2).sum(axis=1)
        alpha = np.full(len(ids), .5 + .5*d2[0]/d2[1:].sum())
        alpha[0] = 0.
        q = np.full((9, 2), np.nan)
        q[list(ids)] = projected_a[list(ids)] + alpha[:, None]*(projected_b[list(ids)]-projected_a[list(ids)])
        q[8] = [641.25, 481.75]
        bank = PoseBank(q, K, xyz, image_size=SIZE)
        a = Candidate(0, ids, 0, 'INJECTED_SSE_TIE_A', cv2.Rodrigues(R)[0], t.reshape(3, 1), projected_a)
        b = Candidate(0, ids, 1, 'INJECTED_SSE_TIE_B', cv2.Rodrigues(R)[0], (t+[.1, 0., 0.]).reshape(3, 1), projected_b)
        # This labelled branch injection exercises score/ambiguity semantics;
        # natural Generic/IPPE returns are tested in the separate test above.
        bank.standard_cache[ids] = [a, b]
        bank._lm = lambda candidate, fit_ids: None
        sa, ia, _ = bank._score(a, ids, False)
        sb, ib, _ = bank._score(b, ids, False)
        self.assertNotEqual(len(ia), len(ib))
        self.assertLessEqual(abs(sa[0]-sb[0]), 1e-8)
        solved = direct_solve(bank, hidden=[], robust=False)
        self.assertFalse(solved['available'])
        self.assertEqual(solved['state'], 'AMBIGUOUS_PNP')
        self.assertTrue(solved['unresolved_ambiguity'])
        self.assertEqual(solved['used'], list(ids))
        single = PoseBank(q, K, xyz, image_size=SIZE)
        single.standard_cache[ids] = [a]
        single._lm = lambda candidate, fit_ids: None
        low_inlier_ordinary = direct_solve(single, hidden=[], robust=False)
        self.assertTrue(low_inlier_ordinary['available'])
        self.assertLess(len(low_inlier_ordinary['final_inliers']), 4)
        self.assertEqual(low_inlier_ordinary['fit_input_ids'], list(ids))
        DETAILS['injected_ordinary_SSE_tie'] = dict(SSE_a=sa[0], SSE_b=sb[0],
            diagnostic_inliers_a=len(ia), diagnostic_inliers_b=len(ib),
            branch_source='explicit test injection, not a natural PnP solution return', state=solved['state'],
            single_branch_ordinary_NEW_with_fewer_than4_diagnostic_inliers=True)

    def test_incomplete_scored_wrong_role_packets_rejected_before_bank(self):
        packet = fixture()
        variants = []
        for field, value in (('complete', False), ('frames', 244), ('geometry_rows', 1959),
                             ('observation_rows', 734), ('fixed_rows', 489), ('cleanup_error', 'failure')):
            bad = copy.deepcopy(packet)
            bad['parent_completion'][field] = value
            variants.append(bad)
        bad = copy.deepcopy(packet); bad['parent_row']['pose'] = dict(translation_cm=0.); variants.append(bad)
        bad = copy.deepcopy(packet); bad['parent_row']['head_arm'] = 'IMAGE_NO_ROLE'; variants.append(bad)
        bad = copy.deepcopy(packet); bad['parent_row']['observation_supply'] = 'CORNERWISE_HYBRID'; variants.append(bad)
        calls = []
        original = C.PoseBank
        def forbidden_bank(*args, **kwargs):
            calls.append(True)
            raise AssertionError('invalid parent constructed a numeric bank')
        C.PoseBank = forbidden_bank
        try:
            for bad in variants:
                with self.assertRaises(ValueError):
                    C.solve_controls(bad)
        finally:
            C.PoseBank = original
        self.assertEqual(calls, [])
        DETAILS['incomplete_parent_rejection'] = dict(invalid_packets=len(variants), numeric_bank_constructions=0)

    def test_control_parity_mismatch_stops_before_other_two_solves(self):
        packet = fixture()
        packet['parent_row']['native_points'] = np.asarray(packet['parent_row']['native_points']).tolist()
        packet['parent_row']['native_points'][0][0] += 1.
        calls = []
        original = C.PoseBank
        class TracedBank(PoseBank):
            def solve(self, *args, **kwargs):
                calls.append(dict(hidden=kwargs.get('hidden'), robust=kwargs.get('robust')))
                COUNTS['aborted_control_replay_pose_paths'] += 1
                return super().solve(*args, **kwargs)
        C.PoseBank = TracedBank
        try:
            with self.assertRaisesRegex(ValueError, 'native_points numeric parity'):
                C.solve_controls(packet)
        finally:
            C.PoseBank = original
        self.assertEqual(calls, [dict(hidden=[6], robust=True)])
        DETAILS['replay_parity_failure'] = dict(actual_pose_paths=1, new_diagnostic_pose_paths=0)


def binding(path):
    p = Path(path)
    data = p.read_bytes()
    return dict(path=p.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(C._canonical(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    started = output.with_name(output.stem + '_STARTED.json')
    if output.exists() or started.exists() or output.is_symlink() or started.is_symlink():
        raise FileExistsError('preserve existing CPU check receipts')
    output.parent.mkdir(parents=True, exist_ok=True)
    code = dict(controls=binding(C.__file__), tests=binding(__file__))
    write_new(started, dict(schema='same_observation_CPU_checks_started_v8', code=code,
        synthetic_only=True, model_calls=0, learning_updates=0, real_GT_reads=0, new_RGB=0))
    cv2.setNumThreads(1)
    primitive = Counter()
    names = ('solvePnPGeneric', 'solvePnPRefineLM', 'solvePnP', 'projectPoints')
    saved = {name: getattr(cv2, name) for name in names}
    for name, fn in saved.items():
        def counted(*a, _name=name, _fn=fn, **kw):
            primitive[_name] += 1
            return _fn(*a, **kw)
        setattr(cv2, name, counted)
    start = time.monotonic()
    result = None
    try:
        with no_external_assets() as reads:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControlContracts)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        receipt = dict(schema='same_observation_CPU_checks_v8', complete=True,
            passed=result.wasSuccessful() and not reads, tests_run=result.testsRun,
            failures=[dict(test=str(t), traceback=s) for t, s in result.failures],
            errors=[dict(test=str(t), traceback=s) for t, s in result.errors],
            code=code, actual_OpenCV_entry_calls=dict(primitive), actual_logical_calls=dict(COUNTS),
            details=DETAILS, forbidden_asset_reads=reads, wall_seconds=time.monotonic()-start,
            model_calls=0, real_images=0, real_GT_reads=0, source_cache_reads=0,
            new_training_updates=0, new_RGB=0, latency_benchmark=False,
            natural_IPPE_tests_and_injected_SSE_tie_are_separate=True)
        write_new(output, receipt)
    finally:
        for name, fn in saved.items():
            setattr(cv2, name, fn)
    if result is None or not result.wasSuccessful() or reads:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
