"""One live point-only closure on unchanged sealed V7 hybrid coordinates.

The original packet validator preserves the corrected head, selected q and
H-union-heldout corner decisions. No line or endpoint LOO is needed here.
The ROOT driver owns file/population authority, primitive counts and cleanup.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

from scripts.research.pallet_hybrid_factor_subset_consensus_20261011_v16 import pipeline_ready as P16

SOLVER_SHA = '1603401ed1f7b1cde301b272bfc1f2f6d3c3935f371ba9c7bafd56e2a893d099'
PARENT_SHA = 'b940cde6eee5023ddd1b1a7c18ff3e5cb77a48567aa58a44f0067fe66481a7d3'


def binding(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def require(value,message):
    P16.require(value,message)


def clone(value):
    return copy.deepcopy(value)


def semantic(value):
    return P16.semantic(value)


parent_binding = binding(Path(P16.__file__))
require(parent_binding['sha256']==PARENT_SHA,'frozen hybrid packet validator source changed')
solver_path = Path(__file__).with_name('solver.py')
solver_binding = binding(solver_path)
require(solver_binding['sha256']==SOLVER_SHA,'exact private V21 closure solver source required')
_spec = importlib.util.spec_from_file_location(__name__+'.isolated_point_closure_solver',solver_path)
require(_spec is not None and _spec.loader is not None,'explicit sibling solver specification')
S21 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = S21
try:
    _spec.loader.exec_module(S21)
except BaseException:
    sys.modules.pop(_spec.name,None)
    raise

PARENT_METHOD = P16.PARENT_METHOD
METHODS = ('HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE',)
PRIMARY = METHODS[0]
POLICY = dict(schema='sealed_hybrid_point_whole_U_closure_policy_v21',methods=list(METHODS),
    primary=PRIMARY,parent_method=PARENT_METHOD,solver=clone(S21.POLICY),
    observation_supply='exact sealed V7 IMAGE_ROLE_CORNERWISE_HYBRID',
    actual_numeric_solve_calls_per_frame=1,point_only=True,
    new_corner_selection_calls=0,new_endpoint_line_LOO_calls=0,new_line_solve_calls=0,
    same_q_H_K_xyz_and_original_corner_LOO=True,H_and_center_excluded=True,
    original_initial_pose_is_fallback_only_not_a_start=True,
    source_edge_double_counting=False,partial_lines_not_pose_inputs=True,
    all_raw4_roots_and_exact_cache_preserved=True,
    NEW_H_projected_once_without_refit=True,full_native_N3_fallback=True,
    training_RGB_head_decode_CAL_or_GT_calls=0,threshold_or_setting_search=False,
    parent_globals_mutated=False,numerical_NEW_is_not_physical_accuracy_success=True,
    known_DEV=True,unseen_evaluation=False,latency_claim=False,
    old_full_V16_564_FAIL_remains_FAIL=True)
SOURCE_PROOF = dict(schema='point_whole_U_closure_pipeline_source_proof_v21',
    source_bindings=dict(V16_packet_validator=parent_binding,V21_solver=solver_binding),
    hybrid_packet_validator_is_exact_frozen_direct_delegate=True,
    mathematical_solver_callbacks_not_copied_or_changed=True,
    solver_whole_AST_coverage_proof=clone(S21.SOURCE_PROOF),parent_globals_mutated=False,
    source_proof_is_not_full_geometry_or_accuracy_validation=True)


def validate_hybrid_packet(packet):
    return P16.validate_hybrid_packet(packet)


def solve_hybrid_packet(packet,*,observer=None):
    before = semantic(packet);p = validate_hybrid_packet(packet);parent = p['row']
    bank = P16.PoseBank(p['q'],p['K'],p['xyz'],image_size=p['size'])
    bank_before = clone(bank.ledger)
    U = tuple(k for k in bank.eligible if k not in p['H'])
    observation = clone(p['observation']);observation['lines'] = []
    subject = S21.PointClosureSolver(bank)
    solved = subject.solve(U,hidden=p['H'],line_blocks=(),observation=observation,
        initial_pose=None,point_result=None)
    require(solved['line_edges']==solved['fit_line_edges']==solved['consumed_edges']==[],
        'line factor entered the point-only closure')
    require(set(p['H']).isdisjoint(solved['fit_input_ids']) and 8 not in solved['fit_input_ids']
        and not solved['prior_used'] and not solved['known_dimension_constraint_used'],
        'H, center, initial or dimension prior entered final fit')
    row = {key:clone(parent[key]) for key in ('id','session','K','xyz','raw_hw','selected_index',
        'fixed_metadata','head_arm','head_mode','head_checkpoint_sha256','calibration_binding')}
    row.update(P16.assemble(p['native'],p['q'],clone(parent['initial_pose']),p['H'],solved,
        'N3_VALIDATED_ROLE',p['contract']))
    row.update(method=PRIMARY,parent_method=PARENT_METHOD,observation_supply='CORNERWISE_HYBRID',
        selected_corner_ids=list(p['adopted']),original_native_N3_points=p['native'].tolist(),
        predicted_initial_N3_hidden=list(p['H']),original_self_hidden_initial=list(p['H']),
        initial_H_applied=True,cornerwise_selection=clone(p['contract']['cornerwise_selection']),
        observation_raw_logits_sha256=parent.get('observation_raw_logits_sha256'),
        hybrid_consensus_policy=clone(POLICY),whole_U_closure_pipeline_source_proof=clone(SOURCE_PROOF),
        native_N3_is_numeric_final_pose_observation=True,missing_sparse_filled_for_numeric_fit=False,
        partial_lines_not_pose_inputs=True,local_point_line_refinement=False,
        pose_estimation_kind=PRIMARY,initial_pose_may_supply_local_optimizer_start=False,
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        parent_geometry_row_semantic_sha256=semantic(parent),
        parent_observation_row_semantic_sha256=semantic(p['observation']),
        parent_native_N3_semantic_sha256=semantic(p['native']),parent_completion=clone(p['completion']),
        point_only_observation_semantic_sha256=semantic(observation),
        head_forward_calls=0,calibration_or_decode_calls=0,new_corner_selection_calls=0,
        new_endpoint_validation_calls=0,cached_accuracy_row_used=False,
        solver_replay_latency_claim=False,oracle=False)
    require(P16.np.array_equal(P16.np.asarray(row['input_points']),p['q'],equal_nan=True)
        and P16.np.array_equal(P16.np.asarray(row['native_points'])[8],p['native'][8],equal_nan=True),
        'selected observations or detector center changed')
    require(not row['reprojections_reused_as_observations'] and before==semantic(packet),
        'hidden projection re-fit or original packet mutation')
    if observer is not None: observer('point_closure_return',row)
    return [row],dict(schema='same_hybrid_point_whole_U_closure_ledger_v21',id=parent['id'],
        session=parent['session'],actual_solver_calls=1,reported_output_views=1,methods=list(METHODS),
        input_packet_semantic_sha256=before,parent_geometry_row_semantic_sha256=semantic(parent),
        parent_observation_row_semantic_sha256=semantic(p['observation']),
        parent_native_N3_semantic_sha256=semantic(p['native']),selected_input_semantic_sha256=semantic(p['q']),
        actual_adopted_boundary_corner_ids=list(p['adopted']),initial_H=list(p['H']),used=list(U),
        point_bank_input_hash=bank.digest,point_bank_before=bank_before,point_bank_after=clone(bank.ledger),
        point_bank_operation_counts={key:bank.ledger[key]-bank_before[key] for key in bank.ledger},
        point_solver=clone(solved),point_solver_counts=clone(subject.counts),
        closure_ledger=clone(solved['closure_ledger']),policy=clone(POLICY),source_proof=clone(SOURCE_PROOF),
        source_geometry_ownership_truth='UNKNOWN',copied_V7_corner_selection_not_new_LOO=True,
        new_endpoint_validation_calls=0,new_GT_scores_models_training_RGB_calls=0,latency_claim=False)
