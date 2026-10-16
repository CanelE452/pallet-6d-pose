"""Isolated observed-frame finite-edge objective over frozen V15 proposals.

Only finite-line containment changes. Point-only numerical callbacks remain
the frozen V13 callbacks. The final count/full-pool truncated-SSE decision
uses the V18 restoration rule: retain a true non-equivalent numerical tie,
without the historical different-cost same-support veto. No parent module
attribute is assigned here. Importing or solving is reserved for ROOT after
the new source/test protocol has been frozen.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from itertools import combinations
from math import comb
from pathlib import Path

import cv2
import numpy as np

from scripts.research.pallet_factor_subset_consensus_20261011_v15 import solver as S15
from scripts.research.pallet_finite_edge_consensus_repair_20261011_v13 import solver as S13
from scripts.research.pallet_consensus_decision_repair_20261011_v18 import solver as D18

P, V5, V9 = S15.P, S13.V5, S13.V9
Candidate, EDGES = S15.Candidate, S13.EDGES
weak, _copy = S13.weak, S15._copy
METHOD = 'OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS'
DECISION_POLICY = dict(
    schema='same_full_pool_count_SSE_decision_policy_v19',
    restored_rule_source='frozen V18 true numerical-tie rule',
    operating_scope='accepted V19 point-only or actual point+unused-edge candidates',
    ranking='descending full-pool point+one-edge inlier count, same-full-pool truncated factor SSE, dimension, logical start',
    numerical_score_tie_px2=D18.TIE_PX2,
    ambiguity='physically non-equivalent accepted alternative with equal vote count and absolute same-full-pool truncated-SSE difference <= inherited1e-8',
    same_support_different_cost_veto_removed=True,
    different_fit_pool_soft_l1_costs_compared=False,
    original_candidate_rank_weak_depth_actual_inlier_and_H_gates_preserved=True,
    alternatives_multiple_solutions_and_gaps_retained=True,
    physical_ownership_certified=False, global_uniqueness_proven=False,
)
POLICY = dict(
    schema='observed_frame_factor_subset_consensus_policy_v19',
    method=METHOD, inherited_candidate_generation_policy=copy.deepcopy(S15.POLICY),
    inherited_point_only_decision_restoration_policy=copy.deepcopy(D18.POLICY),
    operative_output_decision_policy=copy.deepcopy(DECISION_POLICY),
    finite_line_objective='original two endpoint-normal rows/sqrt(2); N actual-support outside hinges in fixed observed tangent/sqrt(N)',
    containment_interval='sorted projections of the same two model endpoints onto the fixed observed tangent',
    line_vote_unit='one actual unused semantic edge',
    tangent_recomputed_from_projected_model_for_containment=False,
    modeled_diagnostic_changes='endpoint normal rows only; observed containment tangent remains fixed',
    point_only_numerical_callbacks='exact frozen V13 dispatch',
    final_ambiguity='non-equivalent accepted alternative with equal full-pool vote count and truncated-SSE difference <= inherited1e-8',
    different_cost_same_support_veto_removed=True,
    count_full_pool_truncated_SSE_dimension_start_order_unchanged=True,
    optimizer_loss_weights_thresholds_starts_and_work_bounds_unchanged=True,
    stable_normal_incidence_guard_unchanged=True,
    parent_modules_mutated=False, synthetic_query_3D_correspondences_added=0,
    H_and_center_excluded_from_fit_and_score=True, NEW_H_projection_once_no_refit=True,
    interval_cusp_derivatives_are_not_observability_certificates=True,
    source_radius_is_not_used_as_a_new_gate=True,
    old_564_scalar_failures_repaired_or_erased=False,
    physical_ownership_certified=False, global_uniqueness_proven=False,
    numerical_NEW_is_not_accuracy_success=True, known_DEV=True,
    actual_integration_or_accuracy_verified=False,
)

EXPECTED_SOURCE_SHA256 = {
    'V15': '4f447f8fa21ae11d90be19f7423b6c42b8dbde4cd8bacb32450b54dfe4c495a9',
    'V13': '3b8cdeb5bc189cb196be4066d095ead081f8a5c3de1b6ee62c28e9402621e4d7',
    'V18': 'db948468b0fffe62460f919940b9445ba97d1384b0d3d3d85730f8cc2321b9ed',
    'V14': '17828a9f0ff7f895898cab761d6aee8e20533a7564c49353ecfe3d03ee7bd6ec',
    'V5': '2de94d3911569500128bdd005a26fb9b7959e97923f0e1afb86b9628ad3c5d30',
    'V4': 'ef827022bef6807c5f7e4016d7ebefa1838c5f013f6d48be6f4932e0c9ad4a40',
    'geometry': '78e2abd2ed187b652a26a6940a190a2f63c524bbcb9a3804c2de60264666e2a2',
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(node):
    return ast.dump(node, include_attributes=False)


def _bound_tree(path, key):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == EXPECTED_SOURCE_SHA256[key], 'immutable source changed: ' + key)
    return ast.parse(raw, filename=str(path)), dict(path=str(path), sha256=digest, bytes=len(raw))


def _method(tree, class_name, method_name):
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    _require(len(classes) == 1, 'unique source class required')
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == method_name]
    _require(len(methods) == 1, 'unique source method required: ' + method_name)
    return copy.deepcopy(methods[0])


def least_squares(*args, **kwargs):
    """Forward to the live original dispatch, including ROOT's counted hook."""
    return S13.least_squares(*args, **kwargs)


def _line_block(endpoints, endpoint_J, line, length, *, modeled_normals=False):
    """Containment uses the bound stdlib helper; normal rows stay NumPy-exact."""
    normal = np.asarray(line['normal'], float)
    support = np.asarray(line['source_line']['support_points'], float)
    N = len(support)
    if N < 3 or support.shape != (N, 2) or not np.isfinite(support).all():
        raise ValueError('FINITE_ACTUAL_SUPPORT_CONTRACT')
    residual, matrix, witness = _observed_frame_block(
        np.asarray(endpoints, float).tolist(), np.asarray(endpoint_J, float).tolist(),
        normal.tolist(), float(line['offset']), support.tolist())
    residual, matrix = np.asarray(residual, float), np.asarray(matrix, float)
    if modeled_normals:
        tangent = (endpoints[1] - endpoints[0]) / length
        normal = np.array([-tangent[1], tangent[0]])
    # Preserve exactly the original normal/J arithmetic required by G14's
    # independently recomputed normal-row scale identity. Only hinge rows
    # come from the new fixed observed-frame helper, even for modeled J.
    signed = endpoints @ normal - line['offset']
    residual[:2] = signed / np.sqrt(2.)
    matrix[:2] = [normal @ endpoint_J[0] / np.sqrt(2.), normal @ endpoint_J[1] / np.sqrt(2.)]
    detail = dict(witness)
    detail.update(normal_signed_px=signed.tolist(),
        projected_segment_length_px=length,
        observed_interval_length_px=abs(witness['endpoint_u'][1] - witness['endpoint_u'][0]),
        longitudinal_support_px=[u - min(witness['endpoint_u']) for u in witness['support_u']],
        longitudinal_coordinate_frame='FIXED_OBSERVED_TANGENT_RELATIVE_TO_LOWER_PROJECTED_ENDPOINT',
        modeled_normals_held_fixed=modeled_normals,
        containment_tangent_uses_observed_normal_even_when_modeled=True)
    return residual, matrix, detail


def finite_residual_jacobian(model, z, K, points, point_ids, lines, *, modeled_normals=False):
    """Same forward point/normal math; observed-frame finite containment only."""
    if not lines:
        return S13.finite_residual_jacobian(model, z, K, points, point_ids, lines,
                                          modeled_normals=modeled_normals)
    return _line_residual_kernel(model, z, K, points, point_ids, lines,
                                 modeled_normals=modeled_normals)


def _score_line(endpoints, line, support, length):
    residual, _, detail = _line_block(endpoints, np.zeros((2, 2, 6)), line, length)
    detail.update(score_state='FINITE_EDGE_BLOCK', actual_support_xy=support.tolist())
    return residual, detail


def _prepare_isolated_kernels():
    """Prove the exact source region changed before compiling isolated copies."""
    tree13, bound13 = _bound_tree(S13.__file__, 'V13')
    tree15, bound15 = _bound_tree(S15.__file__, 'V15')
    _, bound18 = _bound_tree(D18.__file__, 'V18')
    _, bound14 = _bound_tree(S15.G14.__file__, 'V14')
    _, bound5 = _bound_tree(V5.__file__, 'V5')
    _, bound4 = _bound_tree(P.__file__, 'V4')
    geometry_path = Path(__file__).with_name('geometry.py')
    geometry_tree, bound_geometry = _bound_tree(geometry_path, 'geometry')
    geometry_ns = {'__name__': 'v19_isolated_observed_geometry', '__file__': str(geometry_path)}
    exec(compile(geometry_tree, str(geometry_path), 'exec'), geometry_ns)

    old_residual = [n for n in tree13.body if isinstance(n, ast.FunctionDef) and n.name == 'finite_residual_jacobian']
    _require(len(old_residual) == 1, 'unique frozen finite residual required')
    residual = copy.deepcopy(old_residual[0])
    loops = [n for n in residual.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'line']
    _require(len(loops) == 1, 'unique finite-line residual loop required')
    loop = loops[0]
    _require(len(loop.body) == 27 and isinstance(loop.body[5], ast.If)
             and isinstance(loop.body[-1], ast.Expr), 'frozen finite-line scaffold changed')
    old_line_body = copy.deepcopy(loop.body)
    replacement = ast.parse("""
block, block_J, detail = _line_block(q[[a,b]], J[[a,b]], line, length, modeled_normals=modeled_normals)
start = len(residual)
residual.extend(block)
matrices.extend(block_J)
blocks.append(dict(edge=line['edge'], endpoints=[a,b], scalar_start=start,
    scalar_end=len(residual), support_query_ids=list(line['support_query_ids']),
    actual_support_xy=np.asarray(line['source_line']['support_points'],float).tolist(),
    query_radii_px=list(line['source_line']['query_radii_px']), radius_gate_applied=False,
    physical_ownership_truth='UNKNOWN', **detail))
""").body
    loop.body = copy.deepcopy(old_line_body[:6]) + replacement
    undone = copy.deepcopy(residual)
    [n for n in undone.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'line'][0].body = old_line_body
    _require(_canonical(undone) == _canonical(old_residual[0]), 'finite callback skeleton inverse failed')

    geometry = _method(tree13, 'FiniteEdgeConsensusBank', '_geometry')
    attempt = _method(tree13, 'FiniteEdgeConsensusBank', '_attempt')
    for name, fn in (('_geometry', geometry), ('_attempt', attempt)):
        _require(_canonical(fn) == _canonical(_method(tree13, 'FiniteEdgeConsensusBank', name)),
                 'unchanged callback whole AST differs: ' + name)

    old_score = _method(tree13, 'FiniteEdgeConsensusBank', '_score')
    score = copy.deepcopy(old_score)
    score_loops = [n for n in score.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'line']
    _require(len(score_loops) == 1, 'unique score edge loop required')
    cases = [n for n in score_loops[0].body if isinstance(n, ast.If)]
    _require(len(cases) == 1 and len(cases[0].orelse) == 7, 'frozen score branch changed')
    old_else = copy.deepcopy(cases[0].orelse)
    cases[0].orelse = ast.parse("""
block, detail = _score_line(q[[a,b]], line, support, length)
edge.append(float(np.linalg.norm(block)))
scalar.extend(block.tolist())
common.update(detail)
""").body
    undone_score = copy.deepcopy(score)
    score_loop = [n for n in undone_score.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'line'][0]
    [n for n in score_loop.body if isinstance(n, ast.If)][0].orelse = old_else
    _require(_canonical(undone_score) == _canonical(old_score), 'score skeleton inverse failed')

    old_solve = _method(tree15, 'FactorSubsetConsensusBank', 'solve')
    solve = copy.deepcopy(old_solve)
    vetoes = [n for n in ast.walk(solve) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == 'same_support']
    _require(len(vetoes) == 1 and len(vetoes[0].orelse) == 1
             and isinstance(vetoes[0].orelse[0], ast.If)
             and _canonical(vetoes[0].orelse[0].test) == _canonical(ast.Name(id='tied', ctx=ast.Load())),
             'unique historical any-cost support veto required')
    old_veto = copy.deepcopy(vetoes[0])
    vetoes[0].test = copy.deepcopy(old_veto.orelse[0].test)
    vetoes[0].body = copy.deepcopy(old_veto.orelse[0].body)
    vetoes[0].orelse = []
    restored = copy.deepcopy(solve)
    ties = [n for n in ast.walk(restored) if isinstance(n, ast.If)
            and _canonical(n) == _canonical(vetoes[0])]
    _require(len(ties) == 1, 'unique repaired numerical-tie veto required')
    ties[0].test, ties[0].body, ties[0].orelse = old_veto.test, old_veto.body, old_veto.orelse
    _require(_canonical(restored) == _canonical(old_solve), 'decision whole-AST inverse failed')
    _require(D18.TIE_PX2 == S13.POLICY['numerical_score_tie_px2'] == 1e-8,
             'current restored numerical-tie policy changed')

    # Each function receives this NEW module namespace. The originals keep
    # their original globals; no mutable-global context or monkeypatch occurs.
    residual.name = '_line_residual_kernel'
    geometry.name, attempt.name = '_isolated_geometry', '_isolated_attempt'
    score.name, solve.name = '_isolated_score', '_isolated_solve'
    module = ast.Module(body=[residual, geometry, attempt, score, solve], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(Path(__file__)), 'exec'), globals())
    proof = dict(schema='observed_frame_isolation_source_proof_v19',
        source_bindings=dict(V13=bound13,V15=bound15,V18=bound18,V14=bound14,V5=bound5,V4=bound4,geometry=bound_geometry),
        finite_callback_changed_region='only per-line body after unchanged endpoint/length/degeneracy prefix',
        finite_callback_inverse_whole_AST_equal=True,
        geometry_and_optimizer_method_bodies_whole_AST_unchanged=True,
        callback_globals='new isolated namespace; finite_residual_jacobian is V19; least_squares forwards to live S13 dispatch',
        score_changed_region='only valid finite-line else body; degenerate scoring outlier branch retained',
        score_inverse_whole_AST_equal=True,
        solve_changed_region='one historical same-support/elif-tied veto replaced by tied-only veto',
        solve_inverse_whole_AST_equal=True, count_SSE_ranking_gates_H_and_work_bounds_changed=False,
        point_only_callbacks_delegate_frozen_parent=True, parent_globals_mutated=False,
        source_AST_checks_are_not_numerical_integration_tests=True)
    return geometry_ns['observed_frame_block'], proof


_observed_frame_block, SOURCE_PROOF = _prepare_isolated_kernels()


class ObservedFrameFactorSubsetConsensusBank(S15.FactorSubsetConsensusBank):
    """Original exhaustive finite proposals with one corrected line objective."""
    def _geometry(self, model, z, point_ids, lines):
        if not lines:
            return S13.FiniteEdgeConsensusBank._geometry(self, model, z, point_ids, lines)
        return _isolated_geometry(self, model, z, point_ids, lines)

    def _score(self, model, z, U, lines):
        if not lines:
            return S13.FiniteEdgeConsensusBank._score(self, model, z, U, lines)
        return _isolated_score(self, model, z, U, lines)

    def _attempt(self, model, z0, point_ids, lines, phase, entry):
        if not lines:
            return S13.FiniteEdgeConsensusBank._attempt(self, model, z0, point_ids, lines, phase, entry)
        return _isolated_attempt(self, model, z0, point_ids, lines, phase, entry)

    def solve(self, observation, adopted_boundary_corner_ids=(), hidden=(), *, initial_pose=None, point_result=None):
        result = _isolated_solve(self, observation, adopted_boundary_corner_ids, hidden,
                                 initial_pose=initial_pose, point_result=point_result)
        result.update(solver=METHOD, inherited_solver_policy=result['policy'], policy=copy.deepcopy(POLICY),
            operative_output_decision_policy=copy.deepcopy(DECISION_POLICY),
            inherited_solver_policy_is_candidate_generation_history=True,
            observed_frame_source_proof=copy.deepcopy(SOURCE_PROOF),
            new_finite_objective_is_not_historical_scalar_failure_repair=True,
            numerical_NEW_is_not_accuracy_success=True, global_uniqueness_proven=False)
        return result
