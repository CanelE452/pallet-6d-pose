"""Two actual observed-frame solves on unchanged sealed V7 hybrid coordinates.

The frozen V16 packet validator and full paired solve body are reused. The
frozen V15 endpoint-filter/repair body is reused in its own namespace. Only
the solver class and method/policy metadata dispatch differ. No parent module
global is assigned, no stored pose is replayed, and no point/line observation
is generated here. ROOT owns the freeze, actual execution and evaluation.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from scripts.research.pallet_hybrid_factor_subset_consensus_20261011_v16 import pipeline_ready as P16
from scripts.research.pallet_factor_subset_consensus_20261011_v15 import pipeline as P15
from . import solver as S19

PARENT_METHOD = P16.PARENT_METHOD
METHODS = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS',
           'HYBRID_OBSERVED_FRAME_POINT_ONLY')
PRIMARY = METHODS[0]
POLICY = dict(
    schema='sealed_hybrid_observed_frame_factor_subset_ablation_v19',
    methods=list(METHODS), primary=PRIMARY, parent_method=PARENT_METHOD,
    inherited_hybrid_supply_policy=copy.deepcopy(P16.POLICY),
    solver=copy.deepcopy(S19.POLICY),
    observation_supply='exact sealed V7 IMAGE_ROLE_CORNERWISE_HYBRID',
    corner_selection_reexecuted=False,
    actual_numeric_solve_calls_per_frame=2,
    shared_same_coordinate_four_ID_hypotheses=True,
    point_only_has_no_line_factors=True,
    point_only_callbacks_are_frozen_V13=True,
    decision_policy_both_views=copy.deepcopy(S19.DECISION_POLICY),
    endpoint_decisions='fresh unchanged native endpoint LOO on actual unused line pool',
    endpoint_unknown_retained=True, endpoint_contradicted_only_removed=True,
    source_edge_consumption='actually adopted boundary q only',
    H_and_center_never_fit_or_scored=True,
    NEW_H_projection_once_without_refit=True,
    new_head_decode_selection_or_training_calls=0,
    threshold_or_weight_search=False, parent_globals_mutated=False,
    physical_ownership_certified=False, global_uniqueness_proven=False,
    old_full_V16_564_FAIL_remains_FAIL=True,
    known_DEV=True, unseen_evaluation=False, latency_claim=False,
)
EXPECTED_SOURCE_SHA256 = {
    'V16_pipeline_ready': 'b940cde6eee5023ddd1b1a7c18ff3e5cb77a48567aa58a44f0067fe66481a7d3',
    'V15_pipeline': 'c5d2472720524783d9f4703ffc9416730fdf6dd6f478566b6a5c2e7d93012586',
    'V19_solver': '990be7c054917259f4323bb0c26efb68dbb6f2c580ec863f0a5e8f62afdb1d58',
}


def require(value, message):
    P16.require(value, message)


def clone(value):
    return copy.deepcopy(value)


def semantic(value):
    return P16.semantic(value)


def validate_hybrid_packet(packet):
    """Exact original source/role/q/H validator, with no policy adaptation."""
    return P16.validate_hybrid_packet(packet)


def _bound_function(module, key, name):
    path = Path(module.__file__)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == EXPECTED_SOURCE_SHA256[key], 'frozen pipeline source changed: ' + key)
    tree = ast.parse(raw, filename=str(path))
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'unique complete source function required: ' + name)
    return copy.deepcopy(found[0]), dict(path=str(path), sha256=digest, bytes=len(raw))


def _compile_whole(function, namespace, source_path):
    """Copy definitions, never call or modify the original module namespace."""
    original = ast.dump(function, include_attributes=False)
    copied = copy.deepcopy(function)
    require(ast.dump(copied, include_attributes=False) == original,
            'entire source function AST must remain unchanged')
    module = ast.Module(body=[copied], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), 'exec'), namespace)
    return namespace[copied.name]


def _prepare_isolated_pipeline():
    repair, repair_binding = _bound_function(P15, 'V15_pipeline', 'repair_observations')
    pair, pair_binding = _bound_function(P16, 'V16_pipeline_ready', 'solve_hybrid_packet')
    solver_path = Path(S19.__file__)
    solver_raw = solver_path.read_bytes()
    require(hashlib.sha256(solver_raw).hexdigest() == EXPECTED_SOURCE_SHA256['V19_solver'],
            'frozen observed-frame solver source changed')
    repair_ns = dict(vars(P15))
    repair_ns.update(FactorSubsetConsensusBank=S19.ObservedFrameFactorSubsetConsensusBank,
                     METHODS=(PRIMARY,), PRIMARY=PRIMARY,
                     POLICY=clone(POLICY), __name__=__name__ + '.isolated_repair')
    repair_function = _compile_whole(repair, repair_ns, repair_binding['path'])
    # P16's paired function only references these two attributes on P15.
    # Its packet validation remains the original P16 function and namespace.
    facade = SimpleNamespace(repair_observations=repair_function, PRIMARY=PRIMARY)
    pair_ns = dict(vars(P16))
    pair_ns.update(P15=facade,
                   FactorSubsetConsensusBank=S19.ObservedFrameFactorSubsetConsensusBank,
                   METHODS=METHODS, PRIMARY=PRIMARY, POLICY=clone(POLICY),
                   validate_hybrid_packet=validate_hybrid_packet,
                   __name__=__name__ + '.isolated_pair')
    pair_function = _compile_whole(pair, pair_ns, pair_binding['path'])
    proof = dict(schema='hybrid_observed_frame_pipeline_isolation_v19',
        source_bindings=dict(V15_pipeline=repair_binding,V16_pipeline_ready=pair_binding,
            V19_solver=dict(path=str(solver_path),sha256=EXPECTED_SOURCE_SHA256['V19_solver'],bytes=len(solver_raw))),
        V15_repair_whole_function_AST_unchanged=True,
        V16_paired_solve_whole_function_AST_unchanged=True,
        V16_hybrid_packet_validation_direct_delegate=True,
        repair_namespace_substitutions=['FactorSubsetConsensusBank','METHODS','PRIMARY','POLICY','__name__'],
        paired_namespace_substitutions=['P15 local facade','FactorSubsetConsensusBank','METHODS','PRIMARY','POLICY','validate_hybrid_packet direct delegate','__name__'],
        parent_globals_mutated=False, source_AST_proof_is_not_full_geometry_validation=True,
        fresh_native_endpoint_body_unchanged=True, same_q_H_and_shared_raw4_bank_body_unchanged=True)
    return repair_function, pair_function, proof


_isolated_repair, _isolated_pair, SOURCE_PROOF = _prepare_isolated_pipeline()


def repair_observations(points, native_points, K, xyz, observation, hidden,
                        adopted_boundary_corner_ids, *, initial_pose, point_result,
                        image_size=(640,480), endpoint_audit=None, observer=None, point_bank=None):
    """One real filtered observed-frame solve, unchanged endpoint LOO body."""
    views, ledger = _isolated_repair(points,native_points,K,xyz,observation,hidden,
        adopted_boundary_corner_ids,initial_pose=initial_pose,point_result=point_result,
        image_size=image_size,endpoint_audit=endpoint_audit,observer=observer,point_bank=point_bank)
    ledger.update(inherited_ledger_schema=ledger['schema'],
        schema='observed_frame_endpoint_filtered_solver_ledger_v19',
        observed_frame_pipeline_source_proof=clone(SOURCE_PROOF),
        finite_observation_policy=clone(POLICY))
    return views, ledger


def solve_hybrid_packet(packet, *, observer=None):
    """Two real solves, one fresh endpoint audit, shared same-q four-ID roots."""
    rows, ledger = _isolated_pair(packet, observer=observer)
    require([r['method'] for r in rows] == list(METHODS), 'two fixed V19 view labels required')
    for row in rows:
        row.update(observed_frame_pipeline_source_proof=clone(SOURCE_PROOF),
            finite_observation_policy=clone(POLICY),
            pose_estimation_kind='HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS',
            new_finite_objective_is_not_historical_scalar_failure_repair=True)
    line = ledger['point_line_ledger']
    line.update(inherited_ledger_schema=line['schema'],
        schema='observed_frame_endpoint_filtered_solver_ledger_v19',
        observed_frame_pipeline_source_proof=clone(SOURCE_PROOF),finite_observation_policy=clone(POLICY))
    ledger.update(inherited_ledger_schema=ledger['schema'],
        schema='observed_frame_hybrid_factor_subset_paired_ledger_v19',
        observed_frame_pipeline_source_proof=clone(SOURCE_PROOF),finite_observation_policy=clone(POLICY),
        historical_V16_poses_or_geometry_used_as_observations=False,
        old_full_V16_564_FAIL_remains_FAIL=True)
    return rows, ledger
