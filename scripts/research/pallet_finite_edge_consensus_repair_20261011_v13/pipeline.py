"""Finite physical edge consensus on unchanged corrected IMAGE_ROLE observations.

Native coordinates are used only for endpoint LOO and full-output fallback.
Final fit uses the selected actual sparse points and unconsumed finite edges.
"""
from __future__ import annotations
import copy
import numpy as np

from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
from ..pallet_endpoint_observability_repair_20261011_v11 import pipeline as V11
from ..pallet_sparse_local_line_20261010_v9 import pipeline as V9
from ..pallet_sparse_local_line_20261010_v9 import solver as S9
from ..pallet_same_observation_controls_20261010_v8 import controls as V8
from .solver import FiniteEdgeConsensusBank, POLICY as SOLVER_POLICY

METHODS=('FINITE_EDGE_CONSENSUS',)
PRIMARY=METHODS[0]
POLICY=dict(schema='finite_edge_consensus_execution_v13', methods=list(METHODS),
    primary=PRIMARY, solver=copy.deepcopy(SOLVER_POLICY),
    endpoint_filter=copy.deepcopy(V11.A10.POLICY),
    original_sparse_ROLE_queries_corners_and_H_unchanged=True,
    retained_UNKNOWN_is_not_frame_failure=True,
    native_coordinates_are_final_fit_observations=False,
    initial_pose_is_start_only=True, H_initial_coordinates_excluded=True,
    accepted_NEW_H_reprojected_once_without_refit=True,
    training_updates=0,new_RGB=0,known_DEV=True,unseen_evaluation=False,
    performance_threshold_search=False)

def require(v,s): V8.require(v,s)
def _copy(v): return copy.deepcopy(v)

def repair_observations(points,native_points,K,xyz,observation,hidden,
                        adopted_boundary_corner_ids,*,initial_pose,point_result,
                        image_size=(640,480),endpoint_audit=None,observer=None):
    inputs=dict(points=points,native=native_points,K=K,xyz=xyz,observation=observation,
        hidden=hidden,adopted=adopted_boundary_corner_ids,initial_pose=initial_pose,
        point_result=point_result)
    before=V8.semantic_sha256(inputs)
    require(not V11.A10.TRUTH_FIELDS.intersection(observation),'GT-free actual observation required')
    q,native_q,K,xyz=(np.asarray(v,float) for v in (points,native_points,K,xyz))
    H=tuple(V8._ids(list(hidden),'hidden'))
    adopted=tuple(V8._ids(list(adopted_boundary_corner_ids),'adopted'))
    bank=FiniteEdgeConsensusBank(q,K,xyz,image_size=image_size)
    U=tuple(k for k in bank.bank.eligible if k not in H)
    lines,consumed,checks,contract_error=[],[],[],None
    try: lines,consumed,checks=bank.factor_validator._lines(observation,adopted,set(U))
    except (KeyError,TypeError,ValueError,IndexError) as e:
        contract_error=dict(type=type(e).__name__,message=str(e))
    native=PoseBank(native_q,K,xyz,image_size=image_size)
    native_before=native.ledger.copy()
    if endpoint_audit is None:
        records=V11._endpoint_records(native,lines,H,observer)
        source='FRESH_NATIVE_ENDPOINT_LOO'
    else:
        records=V11._validate_saved_records(endpoint_audit,native,lines,H,q,adopted,K,xyz,
            image_size,consumed,checks,contract_error)
        require(endpoint_audit['actual_sparse_used_ids']==list(U),'same-source sparse IDs required')
        source='SEALED_V10_ENDPOINT_DECISION_REPLAY'
    rejected=[r['edge'] for r in records if r['decision']=='CONTRADICTED_REJECT']
    retained=[l['edge'] for l in lines if l['edge'] not in rejected]
    filtered=_copy(observation)
    if contract_error is None:
        filtered['lines']=[_copy(l) for l in observation['lines'] if l['edge'] not in rejected]
    require(V8.semantic_sha256(filtered['queries'])==V8.semantic_sha256(observation['queries'])
        and V8.semantic_sha256(filtered['corners'])==V8.semantic_sha256(observation['corners']),
        'endpoint filter changed actual point/query observations')
    solved=bank.solve(filtered,adopted_boundary_corner_ids=adopted,hidden=H,
        initial_pose=_copy(initial_pose),point_result=_copy(point_result))
    require(solved['used']==list(U) and solved['line_edges']==retained
        and solved['consumed_edges']==consumed,'finite solver source pool differs')
    require(set(H).isdisjoint(solved['fit_input_ids']) and solved['prior_used'] is False
        and solved['initial_pose_residual_prior'] is False,'hidden/prior entered final fit')
    filter_witness=dict(schema='actual_unused_line_endpoint_filter_v13',
        endpoint_decision_source=source,
        original_observation_semantic_sha256=V8.semantic_sha256(observation),
        filtered_observation_semantic_sha256=V8.semantic_sha256(filtered),
        original_factor_pool_binding=None if contract_error else S9._binding(dict(
            points=q[list(U)],point_ids=U,lines=[dict(edge=l['edge'],normal=l['normal'],
            offset=l['offset'],support_query_ids=l['support_query_ids']) for l in lines])),
        original_actual_unused_edge_ids=[l['edge'] for l in lines],
        retained_unused_edge_ids=retained,rejected_edge_ids=rejected,
        unknown_retained_edge_ids=[r['edge'] for r in records if r['decision']=='UNVERIFIED_RETAIN'],
        consumed_edges=consumed,original_line_contract_checks=checks,contract_error=contract_error,
        original_actual_source_lines=[V11._source_line(l) for l in lines],
        endpoint_records=_copy(records),physical_ownership_truth='UNKNOWN',
        no_endpoint_projection_as_observation=True,corners_and_queries_unchanged=True)
    ledger=dict(schema='finite_edge_consensus_ledger_v13',actual_filtered_local_solve_calls=1,
        reported_output_views=1,endpoint_decision_source=source,
        endpoint_validation_requests=len(records) if endpoint_audit is None else 0,
        endpoint_decision_records=len(records),endpoint_native_bank_before=native_before,
        endpoint_native_bank_after=native.ledger.copy(),endpoint_native_input_hash=native.digest,
        local_operation_counts=_copy(solved['operation_counts']),local_bank_counts=_copy(bank.counts),
        local_coordinate_bank=dict(input_hash=bank.bank.digest,**dict(bank.bank.ledger)),
        observation_filter=filter_witness,full_solver=_copy(solved),
        original_H_preserved=True,new_GT_scores_training_RGB_calls=0,latency_claim=False)
    require(before==V8.semantic_sha256(inputs),'input observations mutated')
    return {PRIMARY:solved},ledger

def solve_repair_packet(local_packet,audit_row,*,observer=None):
    before=V8.semantic_sha256(dict(local_packet=local_packet,audit_row=audit_row))
    p=V9.validate_local_packet(local_packet);parent=p['row']
    require(audit_row['id']==parent['id'] and audit_row['session']==parent['session']
        and audit_row['parent_geometry_row_semantic_sha256']==V8.semantic_sha256(parent)
        and audit_row['original_ROLE_observation_semantic_sha256']==V8.semantic_sha256(p['observation_row'])
        and audit_row['original_native_N3_semantic_sha256']==V8.semantic_sha256(p['native']),
        'immutable V10 same-frame lineage differs')
    views,ledger=repair_observations(p['q'],p['native'],p['K'],p['xyz'],p['observation'],p['H'],
        p['contract']['validated_boundary_corner_ids'],initial_pose=parent['initial_pose'],
        point_result=parent['solver'],image_size=p['size'],endpoint_audit=audit_row,observer=observer)
    require(ledger['observation_filter']['original_factor_pool_binding']==audit_row['factor_pool_binding'],
        'original point/line pool differs from V10')
    solved=views[PRIMARY]
    row={k:_copy(parent[k]) for k in ('id','session','K','xyz','raw_hw','selected_index',
        'fixed_metadata','head_arm','head_mode','head_checkpoint_sha256','calibration_binding') if k in parent}
    row.update(assemble(p['native'],p['q'],parent['initial_pose'],p['H'],solved,
        'VALIDATED_ROLE_ONLY',p['contract']))
    row.update(method=PRIMARY,comparator=V11.PRIMARY,parent_method=V11.PARENT_PRIMARY,
        observation_supply='BOUNDARY_ONLY',selected_corner_ids=list(p['contract']['validated_boundary_corner_ids']),
        original_native_N3_points=np.asarray(p['native']).tolist(),
        predicted_initial_N3_hidden=list(p['H']),original_self_hidden_initial=list(p['H']),
        initial_H_applied=True,finite_edge_consensus_policy=_copy(POLICY),
        observation_filter=_copy(ledger['observation_filter']),
        pose_estimation_kind='FINITE_EDGE_CONSENSUS_LOCAL_REFINEMENT',
        local_point_line_refinement=True,missing_sparse_filled_for_numeric_fit=False,
        native_N3_is_numeric_final_pose_observation=False,
        final_numeric_pose_has_initial_prior=False,initial_pose_may_supply_local_optimizer_start=True,
        original_lines_used_only_if_not_consumed_by_actual_adopted_points=True,
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        parent_geometry_row_semantic_sha256=V8.semantic_sha256(parent),
        parent_observation_row_semantic_sha256=V8.semantic_sha256(p['observation_row']),
        parent_native_N3_semantic_sha256=V8.semantic_sha256(p['native']),
        parent_completion=_copy(p['completion']),
        endpoint_audit_semantic_sha256=V8.semantic_sha256(audit_row),
        cached_accuracy_row_used=False,solver_replay_latency_claim=False,oracle=False)
    ledger.update(id=parent['id'],session=parent['session'],input_packet_semantic_sha256=before,
        parent_geometry_row_semantic_sha256=row['parent_geometry_row_semantic_sha256'],
        parent_observation_row_semantic_sha256=row['parent_observation_row_semantic_sha256'],
        endpoint_audit_semantic_sha256=row['endpoint_audit_semantic_sha256'])
    require(before==V8.semantic_sha256(dict(local_packet=local_packet,audit_row=audit_row)),
        'sealed packet mutated')
    return [row],ledger
