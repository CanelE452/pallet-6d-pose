"""Frozen synthetic CPU contracts for the local sparse ROLE point/line path.

Writing this file is not execution. The root freezes its source and invokes
this CLI once. No real images, GT, model, source arrays or saved observations
are opened. Natural SciPy solves and controlled optimizer-return injections
are reported separately; an injected ambiguity is not a naturally found root.
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
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import cv2
import numpy as np

from . import solver as S
from ..pallet_partial_line_independent_20261010_v5 import solver as V5
from ..pallet_observation_refiner_20261009_v1.solver import cuboid, project

REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_sparse_local_line_20261010_v9'
PRIVATE = Path('/tmp/pallet-sparse-local-line-private-20261010-v9')
CAL_EDGES = (0, 2, 4, 8, 9, 10, 11)
SIZE = (1280, 960)
COUNTS, DETAILS = Counter(), {}


def add_local_operations(result):
    COUNTS.update({'local:' + k: v for k, v in result['operation_counts'].items()})


def geometry():
    """Deterministic mathematical geometry, never a rendered/image fixture."""
    K = np.array([[600., 0., 640.], [0., 600., 480.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.3])
    R = cv2.Rodrigues(np.array([.3, .25, .1]))[0]
    t = np.array([0., .1, 4.])
    uv = project(cuboid(*xyz), R, t, K)
    return K, xyz, R, t, uv


def pose(R, t, xyz):
    return dict(available=True, R_cf=R.tolist(), R_physical=R.tolist(),
        centroid=np.asarray(t).tolist(), cf_extents=xyz.tolist(),
        selected_hypothesis='REGISTRY_WD')


def observation(uv, corners=(), edges=CAL_EDGES):
    lines = []
    for edge in edges:
        a, b = V5.EDGES[edge]
        tangent = uv[b] - uv[a]
        length = float(np.linalg.norm(tangent))
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        fractions = np.array([1., 3., 5.]) / 8.
        support = np.array([(1-u)*uv[a] + u*uv[b] for u in fractions])
        lines.append(dict(edge=edge, endpoints=[a, b], normal=normal.tolist(),
            offset=float(normal @ uv[a]), support_points=support.tolist(),
            queries=[edge*7+i for i in (0, 2, 4)], query_radii_px=[1., 1., 1.],
            support_length_px=float(np.linalg.norm(support[-1]-support[0])),
            synthetic_registry_and_CAL_line=True))
    corner_rows = []
    for corner in corners:
        incident = [edge for edge in edges if corner in V5.EDGES[edge]]
        if len(incident) < 2:
            raise ValueError('fixture corner lacks two supported incident lines')
        corner_rows.append(dict(id=corner, xy=uv[corner].tolist(), edges=incident[:2],
            radius_px=1., source='synthetic two physical/CAL line intersection'))
    # Deliberately empty: final unused factors must be rebuilt from .lines.
    return dict(corners=corner_rows, lines=lines, partial_lines=[])


def fixture(corners=(), H=(), edges=CAL_EDGES, perturbed=True):
    K, xyz, R, t, uv = geometry()
    q = np.full((9, 2), np.nan)
    q[list(corners)] = uv[list(corners)]
    q[8] = [641.25, 481.75]
    seed_R = cv2.Rodrigues(np.array([.31, .24, .105]))[0] if perturbed else R
    seed_t = t + [.015, -.01, .025] if perturbed else t
    return q, K, xyz, observation(uv, corners, edges), pose(seed_R, seed_t, xyz), uv, R, t


def solve(q, K, xyz, obs, initial, adopted=(), H=(), point_result=None):
    COUNTS['fixture_local_solve_paths'] += 1
    bank = S.LocalPointLineBank(q, K, xyz, SIZE)
    result = bank.solve(obs, adopted_boundary_corner_ids=adopted, hidden=H,
                        initial_pose=initial, point_result=point_result)
    add_local_operations(result)
    return result


def fitted_result(z, residual, success=True, status=1):
    return SimpleNamespace(x=np.asarray(z), success=success, status=status, nfev=1, njev=1,
                           cost=S.soft_l1_cost(residual))


class LocalContracts(unittest.TestCase):
    def assert_local_contract(self, result, U, H=()):
        self.assertEqual(result['used'], list(U))
        self.assertFalse(result['prior_used'])
        self.assertFalse(result['initial_pose_residual_prior'])
        self.assertFalse(result['initial_dimension_prior_used'])
        self.assertFalse(result['independent_PnP_or_PnL'])
        self.assertFalse(result['global_uniqueness_proven'])
        self.assertTrue(result['rank_is_local_only'])
        self.assertTrue(result['no_minimum_point_inlier_gate'])
        self.assertFalse(result['reprojected_points_reused_as_observations'])
        self.assertFalse(result['same_edge_point_line_double_count'])
        self.assertEqual(result['derived_line_points'], 0)
        for entry in result['optimizer_attempts']:
            self.assertEqual(entry['fit_point_ids'], list(U))
            self.assertFalse(set(entry['fit_point_ids']) & set(H))
            self.assertFalse(set(entry['fit_line_edges']) & set(result['consumed_edges']))
            self.assertEqual(entry['factor_pool_binding'], result['factor_pool_binding'])
            self.assertFalse(entry['initial_pose_residual_prior'])
        for record in result['all_candidate_solutions']:
            self.assertEqual(record['fit_input_ids'], list(U))
            self.assertEqual(record['fit_line_edges'], result['line_edges'])
            self.assertEqual(record['factor_pool_binding'], result['factor_pool_binding'])
            rr = np.asarray(record['scalar_residuals_px'])
            self.assertAlmostEqual(record['soft_l1_cost'], S.soft_l1_cost(rr), places=8)
            self.assertAlmostEqual(record['sse_px2'], float(rr @ rr), places=7)
            self.assertTrue(np.isfinite(record['model_camera_depths_m']).all())
            self.assertGreater(min(record['model_camera_depths_m']), 0.)
        if result['available']:
            self.assertEqual(result['state'], 'NEW_POSE')
            self.assertEqual(result['fit_input_ids'], list(U))
            self.assertEqual(result['point_line_geometry']['observed_normal_joint']['numerical_rank'], 6)
            self.assertEqual(result['point_line_geometry']['modeled_normal_joint']['numerical_rank'], 6)
            self.assertFalse(result['inlier_geometry_is_acceptance_gate'])
            self.assertEqual(result['diagnostic_inlier_scalar_residuals'],
                             2*(len(result['final_inliers'])+len(result['final_line_inliers'])))

    def test_fixed_local_policy_without_new_initializers(self):
        self.assertEqual(S.POLICY['loss'], 'soft_l1')
        self.assertEqual(S.POLICY['f_scale_px'], 8.)
        self.assertEqual(S.POLICY['max_nfev'], 50)
        self.assertAlmostEqual(S.POLICY['line_weight'], 1/np.sqrt(2))
        self.assertEqual(S.POLICY['minimum_actual_points'], 0)
        self.assertFalse(S.POLICY['random_starts'])
        self.assertFalse(S.POLICY['independent_PnP_or_PnL'])
        self.assertTrue(S.POLICY['source_CAL_support_required_by_packet'])
        self.assertEqual(CAL_EDGES, (0, 2, 4, 8, 9, 10, 11))

    def test_natural_zero_to_three_points_with_supported_lines(self):
        details = []
        for U in ((), (0,), (0, 1), (0, 1, 4)):
            q, K, xyz, obs, seed, uv, _, _ = fixture(U)
            result = solve(q, K, xyz, obs, seed, adopted=U)
            self.assertTrue(result['available'], (U, result['state'], result['optimizer_attempts']))
            self.assert_local_contract(result, U)
            self.assertTrue(result['fit_has_fewer_than_four_actual_points'])
            self.assertEqual(result['factor_pool']['actual_point_factors'], len(U))
            self.assertLess(np.max(np.abs(np.asarray(result['projected'])-uv)), 1e-3)
            self.assertTrue(np.isnan(np.asarray(result['points_final'])[:8][[k for k in range(8) if k not in U]]).all())
            self.assertEqual(result['points_final'][8], q[8].tolist())
            details.append(dict(actual_points=len(U), point_ids=list(U),
                unused_line_edges=result['line_edges'], state=result['state'],
                point_inliers=len(result['final_inliers']), local_only=True,
                operations=result['operation_counts']))
        DETAILS['natural_zero_to_three'] = details

    def test_insufficient_factors_do_not_start_optimizer(self):
        for U in ((), (0,)):
            # No adopted intersection: the point fixture is an independent
            # mathematical point; no claim that one line made a corner.
            q, K, xyz, _, seed, uv, _, _ = fixture(U)
            obs = observation(uv, (), (0,))
            result = solve(q, K, xyz, obs, seed)
            self.assertEqual(result['state'], 'INSUFFICIENT_LOCAL_FACTORS')
            self.assertFalse(result['available'])
            self.assertEqual(result['operation_counts']['local_optimizer_calls'], 0)
            self.assertLess(result['factor_pool']['scalar_residuals'], 6)

    def test_parallel_noisy_observed_rank_is_not_sufficient(self):
        q, K, xyz, obs, seed, _, R, t = fixture((), edges=(0, 2, 4), perturbed=False)
        for line, angle in zip(obs['lines'], (.01, -.02, .03)):
            c, s = np.cos(angle), np.sin(angle)
            normal = np.array([[c, -s], [s, c]]) @ np.asarray(line['normal'])
            support = np.asarray(line['support_points'])
            center = support.mean(0)
            tangent = np.array([normal[1], -normal[0]])
            span = np.linalg.norm(support[-1]-support[0]) / 2
            line.update(normal=normal.tolist(), offset=float(normal @ center),
                        support_points=(center + np.array([-span, 0., span])[:, None]*tangent).tolist())
        bank = S.LocalPointLineBank(q, K, xyz, SIZE)
        lines, _, _ = bank.factor_validator._lines(obs, (), set())
        z = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
        joint = V5.geometry_info(cuboid(*xyz), z, K, q, (), lines)
        self.assertEqual(joint['observed_normal_joint']['numerical_rank'], 6)
        self.assertEqual(joint['modeled_normal_joint']['numerical_rank'], 5)
        def returned(fun, z0, **kwargs):
            COUNTS['injected_optimizer_entries'] += 1
            return fitted_result(z0, fun(z0))
        with patch.object(S, 'least_squares', returned):
            result = solve(q, K, xyz, obs, seed)
        self.assertFalse(result['available'])
        self.assertEqual(result['operation_counts']['local_optimizer_calls'], 2)
        self.assertEqual(result['state'], 'NUMERICAL_RANK_DEFICIENT')
        self.assertTrue(all(not record['accepted'] for record in result['all_candidate_solutions']))
        DETAILS['parallel_phantom'] = dict(observed_rank=6, modeled_rank=5,
            optimizer_calls=2, controlled_return_branches=True, physical_CAL_edges=[0, 2, 4])

    def test_consumed_lines_duplicates_and_raw_partial_not_authority(self):
        q, K, xyz, obs, seed, _, _, _ = fixture((0,))
        original = copy.deepcopy(obs)
        obs['lines'].append(copy.deepcopy(obs['lines'][-1]))
        result = solve(q, K, xyz, obs, seed, adopted=(0,))
        self.assertTrue(result['available'], result['state'])
        self.assertEqual(result['consumed_edges'], [0, 8])
        self.assertEqual(result['line_edges'], [2, 4, 9, 10, 11])
        self.assertEqual(len(result['line_edges']), len(set(result['line_edges'])))
        self.assertEqual(obs['partial_lines'], [])
        self.assert_local_contract(result, (0,))
        bad = copy.deepcopy(original)
        bad['lines'].append(copy.deepcopy(bad['lines'][-1]))
        bad['lines'][-1]['offset'] += 1.
        rejected = solve(q, K, xyz, bad, seed, adopted=(0,))
        self.assertEqual(rejected['state'], 'INVALID_OBSERVATION_CONTRACT')
        self.assertEqual(rejected['operation_counts']['local_optimizer_calls'], 0)
        nonphysical = copy.deepcopy(original)
        unsupported = copy.deepcopy(nonphysical['lines'][0])
        unsupported.update(edge=1, endpoints=[1, 2], queries=[7, 9, 11])
        nonphysical['lines'].append(unsupported)
        filtered = solve(q, K, xyz, nonphysical, seed, adopted=(0,))
        self.assertTrue(filtered['available'], filtered['state'])
        self.assertNotIn(1, filtered['line_edges'])
        self.assertTrue(any(check['edge'] == 1 and check['accepted'] is False and
                            check['reason'] == 'SOURCE_REGISTRY_UNSUPPORTED_EDGE'
                            for check in filtered['line_contract_checks']))

    def test_H_coordinate_invariance_and_projection_never_refit(self):
        U, H = (0, 1, 4), (7,)
        q, K, xyz, obs, seed, _, _, _ = fixture(U)
        # H receives artificial initial image coordinates only. No corner7
        # boundary observation is invented to make two supported incident edges.
        answers = []
        for hidden_xy in ([900., -800.], [-1., -1.], [np.nan, np.nan]):
            modified = q.copy(); modified[7] = hidden_xy
            answer = solve(modified, K, xyz, obs, seed, adopted=U, H=H)
            self.assertTrue(answer['available'], answer['state'])
            self.assert_local_contract(answer, U, H)
            self.assertIn(11, answer['line_edges'])
            self.assertEqual(answer['points_final'][7], answer['projected'][7])
            self.assertEqual(answer['points_final'][8], q[8].tolist())
            self.assertEqual(answer['operation_counts']['hidden_reprojection_assignments'], 1)
            self.assertEqual(answer['operation_counts']['local_optimizer_calls'],
                             sum(e['actual_optimizer_called'] for e in answer['optimizer_attempts']))
            answers.append(answer)
        for answer in answers[1:]:
            np.testing.assert_allclose(answer['R_cf'], answers[0]['R_cf'], atol=1e-7, rtol=0)
            np.testing.assert_allclose(answer['centroid'], answers[0]['centroid'], atol=1e-7, rtol=0)
            self.assertEqual(answer['factor_pool_binding'], answers[0]['factor_pool_binding'])
        DETAILS['H_invariance'] = dict(H=list(H), actual_fit_ids=list(U), retained_H_endpoint_edge=11,
            hidden_variants=3, projected_reused_as_fit=False)

    def test_both_dimensions_multiple_sources_and_exact_start_cache(self):
        q, K, xyz, obs, seed, _, _, _ = fixture((0, 1, 4))
        result = solve(q, K, xyz, obs, seed, adopted=(0, 1, 4), point_result=copy.deepcopy(seed))
        self.assertTrue(result['available'], result['state'])
        self.assertEqual({e['dimension_index'] for e in result['optimizer_attempts']}, {0, 1})
        self.assertEqual({e['source'] for e in result['optimizer_attempts']},
                         {'SEALED_INITIAL_N3', 'SEALED_AVAILABLE_POINT_ONLY'})
        self.assertEqual(result['operation_counts']['local_start_entries'], 4)
        self.assertEqual(result['operation_counts']['local_start_cache_hits'], 2)
        self.assertEqual(result['operation_counts']['local_optimizer_calls'], 2)
        self.assertEqual(result['local_start_cache']['unique_numeric_starts'], 2)
        for witness in result['start_source_witnesses']:
            self.assertTrue(witness['accepted'])
            self.assertFalse(witness['residual_prior'])
            self.assertTrue(witness['all_registry_dimensions_will_be_tried'])
        self.assert_local_contract(result, (0, 1, 4))
        DETAILS['duplicate_source_fixture'] = dict(second_source='synthetic declared available point-only pose',
            not_a_new_actual_PnP_result=True, entries=4, actual_optimizers=2, exact_cache_hits=2)

    def test_missing_initialization_and_invalid_seed_are_explicit(self):
        q, K, xyz, obs, seed, _, _, _ = fixture(())
        for initial in (None, dict(available=False), dict(seed, centroid=[float('nan'), 0., 4.])):
            result = solve(q, K, xyz, obs, initial)
            self.assertFalse(result['available'])
            self.assertEqual(result['state'], 'NO_LOCAL_INITIALIZATION')
            self.assertEqual(result['operation_counts']['local_optimizer_calls'], 0)

    def test_no_unused_lines_three_actual_points_remains_LOCAL(self):
        U = (0, 1, 4)
        q, K, xyz, _, seed, _, _, _ = fixture(U)
        # These are explicitly independent mathematical point factors, not
        # fake line endpoints. The original local C2 also permits line0+3points.
        result = solve(q, K, xyz, dict(corners=[], lines=[], partial_lines=[]), seed)
        self.assertTrue(result['available'], result['state'])
        self.assert_local_contract(result, U)
        self.assertEqual(result['line_edges'], [])
        self.assertEqual(result['fit_line_edges'], [])
        self.assertEqual(result['factor_pool']['scalar_residuals'], 6)
        self.assertTrue(result['fit_has_fewer_than_four_actual_points'])
        DETAILS['no_unused_line_point_local'] = dict(actual_points=3, unused_lines=0,
            independent_P3P_or_PnP_claim=False, global_uniqueness_claim=False)

    def test_initial_modeled_edge_collapse_is_diagnostic_only(self):
        q, K, xyz, obs, _, _, R, t = fixture((), perturbed=False)
        initial = pose(np.eye(3), [xyz[0]/2, xyz[1]/2, 4.], xyz)
        z = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
        base_rvec = np.zeros(3)
        def returned(fun, z0, **kwargs):
            COUNTS['injected_optimizer_entries'] += 1
            if np.linalg.norm(np.asarray(z0)[:3]-base_rvec) > 1e-6:
                return fitted_result(z0, fun(z0), success=False, status=0)
            # Explicitly inject a correct returned local branch. This tests
            # dispatch/acceptance after unavailable initial diagnostic only.
            return fitted_result(z, fun(z))
        with patch.object(S, 'least_squares', returned):
            result = solve(q, K, xyz, obs, initial)
        self.assertTrue(result['available'], result['state'])
        self.assert_local_contract(result, ())
        first = result['optimizer_attempts'][0]
        self.assertTrue(first['actual_optimizer_called'])
        self.assertIsNotNone(first['initial_geometry_unavailable'])
        self.assertEqual(result['point_line_geometry']['modeled_normal_joint']['numerical_rank'], 6)
        DETAILS['initial_collapse'] = dict(edge=8, unavailable_initial_diagnostic_is_not_veto=True,
            controlled_final_branch=True, natural_convergence_claim=False)

    def test_optimizer_failure_and_negative_depth_are_not_NEW(self):
        q, K, xyz, obs, seed, _, _, _ = fixture(())
        def fail(*args, **kwargs):
            COUNTS['injected_optimizer_entries'] += 1
            raise ValueError('synthetic explicit optimizer failure')
        with patch.object(S, 'least_squares', fail):
            result = solve(q, K, xyz, obs, seed)
        self.assertFalse(result['available'])
        self.assertEqual(result['state'], 'LOCAL_NUMERICAL_FAILURE')
        self.assertEqual(result['operation_counts']['local_optimizer_calls'], 2)
        self.assertEqual(result['operation_counts']['local_optimizer_completions'], 0)
        def behind(fun, z0, **kwargs):
            COUNTS['injected_optimizer_entries'] += 1
            z = np.asarray(z0).copy(); z[5] = -4.
            return fitted_result(z, fun(z))
        with patch.object(S, 'least_squares', behind):
            result = solve(q, K, xyz, obs, seed)
        self.assertFalse(result['available'])
        self.assertEqual(result['state'], 'LOCAL_NUMERICAL_FAILURE')
        self.assertEqual(result['candidate_count'], 0)

    def test_distinct_equal_objective_returned_branches_are_ambiguous(self):
        # Deliberate returned-branch injection. Midpoint image observations and
        # the center pose's line normals give opposite residuals for t_x=+/-d.
        # Production independently recomputes projection/objective/J/rank; we
        # do not claim these forced branches are natural optimizer minima.
        U = (0, 1, 4)
        q, K, xyz, obs, _, _, R, t = fixture(U, perturbed=False)
        delta = .04
        first, second = t + [delta, 0., 0.], t - [delta, 0., 0.]
        a, b = pose(R, first, xyz), pose(R, second, xyz)
        base_rvec = cv2.Rodrigues(R)[0].ravel()
        def returned(fun, z0, **kwargs):
            COUNTS['injected_optimizer_entries'] += 1
            z0 = np.asarray(z0)
            if np.linalg.norm(z0[:3]-base_rvec) > 1e-6:
                return fitted_result(z0, fun(z0), success=False, status=0)
            return fitted_result(z0, fun(z0))
        with patch.object(S, 'least_squares', returned):
            result = solve(q, K, xyz, obs, a, adopted=U, point_result=b)
        self.assertFalse(result['available'])
        self.assertEqual(result['state'], 'AMBIGUOUS_LOCAL_POINT_LINE')
        self.assertTrue(result['unresolved_ambiguity'])
        accepted = [r for r in result['all_candidate_solutions'] if r['accepted']]
        self.assertEqual(len(accepted), 2)
        self.assertLess(abs(accepted[0]['soft_l1_cost']-accepted[1]['soft_l1_cost']), 1e-8)
        self.assertGreater(np.max(np.abs(np.asarray(accepted[0]['projected'])-
                                          np.asarray(accepted[1]['projected']))), 1e-5)
        self.assert_local_contract(result, U)
        DETAILS['controlled_ambiguity'] = dict(injected_return_branches=True,
            natural_optimizer_root_claim=False, independently_recomputed_rank_objective=True,
            distinct_physical_translation=True, state=result['state'])

    def test_exact_forward_J_and_factor_normalization(self):
        q, K, xyz, obs, _, _, R, t = fixture((0,))
        bank = S.LocalPointLineBank(q, K, xyz, SIZE)
        lines, consumed, _ = bank.factor_validator._lines(obs, (0,), {0})
        z = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
        rr, J, uv = V5.residual_jacobian(cuboid(*xyz), z, K, q, (0,), lines)
        numeric = np.empty_like(J)
        for column in range(6):
            step = np.zeros(6); step[column] = 1e-6
            plus = V5.residual_jacobian(cuboid(*xyz), z+step, K, q, (0,), lines)[0]
            minus = V5.residual_jacobian(cuboid(*xyz), z-step, K, q, (0,), lines)[0]
            numeric[:, column] = (plus-minus)/(2e-6)
        self.assertLess(np.max(np.abs(J-numeric)), 1e-4)
        self.assertEqual(consumed, [0, 8])
        for index, line in enumerate(lines):
            endpoint = uv[list(V5.EDGES[line['edge']])] @ line['normal'] - line['offset']
            np.testing.assert_allclose(rr[2+2*index:4+2*index], endpoint/np.sqrt(2), atol=1e-10, rtol=0)
        DETAILS['forward_J'] = dict(max_abs_difference=float(np.max(np.abs(J-numeric))),
            tested_scalar_rows=len(rr), finite_difference_step=1e-6,
            one_edge_one_factor_despite_two_components=True)

    def test_packet_wrapper_NEW_H_projection_and_full_fallback(self):
        # Reuse only the published synthetic packet builder; no saved raw data.
        from . import pipeline as driver
        from ..pallet_same_observation_controls_20261010_v8 import test_controls as T8
        from ..pallet_same_observation_controls_20261010_v8.controls import semantic_sha256
        U, H = (0, 1, 4), (7,)
        COUNTS['synthetic_parent_pose_paths'] += 1
        packet = T8.fixture(ids=U, H=H)
        parent = packet['parent_row']
        q = np.asarray(parent['input_points'], float)
        K, xyz = np.asarray(parent['K']), np.asarray(parent['xyz'])
        full = np.asarray(packet['native_N3_points'], float)
        raw = observation(full[:8], U)
        queries = [dict(query=i, edge=i//7, calibrated_model_coverage=i//7 in CAL_EDGES,
                        selected_xy=None) for i in range(84)]
        for line in raw['lines']:
            for query_id, xy in zip(line['queries'], line['support_points']):
                queries[query_id]['selected_xy'] = list(xy)
        parent['observation_contract']['corner_admission'] = [
            dict(id=corner['id'], accepted=True, edges=corner['edges']) for corner in raw['corners']]
        role = dict(id=parent['id'], session=parent['session'], head_arm='IMAGE_ROLE',
            raw_logits_sha256=parent['observation_raw_logits_sha256'],
            initial_N3_pose=copy.deepcopy(parent['initial_pose']), predicted_N3_hidden=list(H),
            native_N3_points=full.tolist(), GT_input=False,
            corners=raw['corners'], lines=raw['lines'], partial_lines=[], queries=queries)
        packet['role_observation'] = role
        missing_CAL = copy.deepcopy(packet)
        first_query = missing_CAL['role_observation']['lines'][0]['queries'][0]
        missing_CAL['role_observation']['queries'][first_query]['calibrated_model_coverage'] = False
        with self.assertRaises(ValueError):
            driver.validate_local_packet(missing_CAL)
        saved = semantic_sha256(packet)
        COUNTS['fixture_local_packet_paths'] += 1
        result, ledger = driver.solve_local_packet(packet)
        add_local_operations(result['solver'])
        self.assertEqual(semantic_sha256(packet), saved)
        np.testing.assert_array_equal(np.asarray(result['input_points'], float), q)
        self.assertEqual(result['native_points'][8], full[8].tolist())
        self.assertTrue(result['new_pose_estimated'], result['solver']['state'])
        self.assertFalse(result['reprojections_reused_as_observations'])
        self.assertEqual(result['reprojected_ids'], list(H))
        self.assertEqual(result['native_points'][7], result['solver']['projected'][7])
        self.assertFalse(set(result['solver']['fit_input_ids']) & set(H))
        for field in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
            np.testing.assert_allclose(result['actual_pose'][field], result['solver'][field], atol=0, rtol=0)
        def wrapper_failure(*args, **kwargs):
            COUNTS['injected_optimizer_entries'] += 1
            raise ValueError('synthetic wrapper failure')
        with patch.object(S, 'least_squares', wrapper_failure):
            COUNTS['fixture_local_packet_paths'] += 1
            fallback, _ = driver.solve_local_packet(packet)
        add_local_operations(fallback['solver'])
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
        COUNTS['fixture_local_packet_paths'] += 1
        failed, _ = driver.solve_local_packet(unavailable)
        add_local_operations(failed['solver'])
        self.assertFalse(failed['new_pose_estimated'])
        self.assertFalse(failed['fallback_used'])
        self.assertTrue(failed['no_pose'])
        self.assertEqual(failed['output_status'], 'POSE_FAILURE')
        DETAILS['packet'] = dict(real_assets=False, final_RT_solver_join=True,
            center_preserved=True, fallback_preserves_full_native_N3=True,
            ledger=ledger)


@contextmanager
def no_external_assets():
    saved = [(builtins, 'open', builtins.open), (io, 'open', io.open), (Path, 'open', Path.open)]
    attempts = []
    tokens = ('TARGETS', 'GROUND_TRUTH', 'STATIC_VISIBILITY', 'READY_SOURCE', 'FEATURES.NPY',
        'PREDICTIONS.JSON', 'OBSERVATIONS.JSON', 'INPUTS.JSON', 'BEST.PT', 'LAST.PT',
        'IMAGE_ROLE.PT', 'GEOMETRY_ONLY.PT', 'IMAGE_NO_ROLE.PT', '.PNG', '.JPG', '.JPEG')
    def guarded(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and any(token in str(path).upper() for token in tokens):
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


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(finite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DOC/'LOCAL_LINE_CHECKS.json')
    parser.add_argument('--protocol', type=Path, default=DOC/'CPU_TEST_PROTOCOL.json')
    args = parser.parse_args()
    output = args.output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('symlink output ancestry')
    output = output.resolve()
    if not (output.parent == DOC.resolve() or output.parent.is_relative_to(PRIVATE.resolve())):
        raise ValueError('only new V9 DOC/private outputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = output.with_name(output.stem+'_STARTED.json')
    if output.exists() or started.exists():
        raise ValueError('preserve existing CPU attempt')
    paths = dict(solver=CODE/'solver.py', pipeline=CODE/'pipeline.py', tests=Path(__file__),
        v5_solver=REPO/'scripts/research/pallet_partial_line_independent_20261010_v5/solver.py',
        v4_pose=REPO/'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py',
        v8_fixture=REPO/'scripts/research/pallet_same_observation_controls_20261010_v8/test_controls.py',
        v8_controls=REPO/'scripts/research/pallet_same_observation_controls_20261010_v8/controls.py',
        v1_solver=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
        v1_inference=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/inference.py',
        v1_common=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/common.py',
        v2_pipeline=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py',
        v2_common=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/common.py')
    code = {key: binding(path) for key, path in paths.items()}
    protocol = json.loads(args.protocol.read_text())
    protocol_binding = binding(args.protocol)
    if (protocol['schema'] != 'pre_execution_synthetic_sparse_local_line_CPU_contract_v9' or
            protocol['code'] != code or protocol['synthetic_only'] is not True or
            protocol['test_groups_expected'] != 14):
        raise ValueError('CPU protocol/code/population differs before CPU fixture arithmetic')
    write_new(started, dict(schema='sparse_local_line_CPU_checks_started_v9', code=code,
        protocol=protocol_binding, synthetic_only=True, model_calls=0,
        real_GT_reads=0, new_RGB=0, automatic_retry=False))
    COUNTS.clear(); DETAILS.clear()
    old_threads = cv2.getNumThreads(); cv2.setNumThreads(1)
    primitive = Counter({name: 0 for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM',
                                            'projectPoints', 'Rodrigues', 'cornerSubPix')})
    originals = {name: getattr(cv2, name) for name in primitive}
    scipy_original = S.least_squares
    def real_optimizer(*args, **kwargs):
        COUNTS['actual_real_SciPy_least_squares_entries'] += 1
        return scipy_original(*args, **kwargs)
    S.least_squares = real_optimizer
    for name, fn in originals.items():
        def counted(*a, _name=name, _fn=fn, **kwargs):
            primitive[_name] += 1
            return _fn(*a, **kwargs)
        setattr(cv2, name, counted)
    result = None
    reads = []
    failure = None
    cleanup_errors = []
    started_time = time.monotonic()
    try:
        with no_external_assets() as reads:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(LocalContracts)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        try:
            S.least_squares = scipy_original
        except BaseException as error:
            cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
        for name, fn in originals.items():
            try:
                setattr(cv2, name, fn)
            except BaseException as error:
                cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
        try:
            cv2.setNumThreads(old_threads)
        except BaseException as error:
            cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
    code_unchanged = {key: binding(path) for key, path in paths.items()} == code
    protocol_unchanged = binding(args.protocol) == protocol_binding
    dispatch_count_join = (COUNTS['local:local_optimizer_calls'] ==
        COUNTS['actual_real_SciPy_least_squares_entries'] + COUNTS['injected_optimizer_entries'])
    complete = result is not None and failure is None and not cleanup_errors
    passed = (complete and result.wasSuccessful() and result.testsRun == 14 and not reads and
              code_unchanged and protocol_unchanged and dispatch_count_join)
    write_new(output, dict(schema='sparse_local_line_CPU_checks_v9', complete=complete, passed=passed,
        tests_run=result.testsRun if result else 0,
        failures=[dict(test=str(test), traceback=trace) for test, trace in result.failures] if result else [],
        errors=[dict(test=str(test), traceback=trace) for test, trace in result.errors] if result else [],
        exception=failure, cleanup_errors=cleanup_errors, code=code, protocol=protocol_binding,
        code_unchanged_after_tests=code_unchanged, protocol_unchanged_after_tests=protocol_unchanged,
        production_local_optimizer_dispatch_count_join=dispatch_count_join,
        local_optimizer_dispatches_include_controlled_returns=True,
        actual_OpenCV_entry_calls=dict(primitive), actual_logical_calls=dict(COUNTS), details=DETAILS,
        forbidden_asset_reads=reads, wall_seconds=time.monotonic()-started_time,
        model_calls=0, real_images=0, real_GT_reads=0, source_cache_reads=0,
        new_training_updates=0, new_RGB=0, latency_benchmark=False,
        controlled_return_branches_and_natural_solvers_separate=True, automatic_retry=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
