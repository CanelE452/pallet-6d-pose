"""Compare original native and already selected hybrid points with one standard backend.

The selection, H and all coordinates are sealed V7 inputs. No detector, head,
LOO, line factor, initial-pose prior or truth is executed here. An ordinary
whole-U fit is a numerical pose output, not a correspondence accuracy certificate.
"""
from __future__ import annotations
import copy
import numpy as np

from ..pallet_hybrid_factor_subset_consensus_20261011_v16 import pipeline_ready as H16
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank, POLICY as POSE_POLICY
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble, observation_points

PARENT_METHOD=H16.PARENT_METHOD
METHODS=('HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP')
PRIMARY=METHODS[0]
POLICY=dict(schema='fixed_native_hybrid_standard_policy_v17',methods=list(METHODS),
    parent_method=PARENT_METHOD,hybrid_supply='exact sealed V7 IMAGE_ROLE_CORNERWISE_HYBRID selected q',
    native_supply='exact original OBS native_N3_points; not parent corrected display',
    solver=copy.deepcopy(POSE_POLICY),robust=False,ordinary_whole_active_U=True,
    final8px_inliers_are_diagnostic_only=True,stored_J_fit_ids='whole solver.used for ordinary J',
    original_H_unchanged=True,initial_pose_used_only_for_H_and_fallback=True,
    initial_pose_or_dimension_prior=False,all_registered_dimensions=True,
    corner_selection_reexecuted=False,head_or_decoder_calls=0,new_training_updates=0,
    new_RGB=0,new_threshold_weight_seed_search=False,partial_line_factors=0,
    numerical_solve_calls_per_frame=2,exact_same_coordinate_banks_shared=True,
    H_initial_XY_and_center_excluded_from_fit_and_score=True,
    NEW_H_projection_without_refit=True,full_original_N3_fallback=True,
    numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
    known_DEV=True,unseen_evaluation=False,replay_wall_is_latency=False)


def require(v,s):H16.require(v,s)
def clone(v):return copy.deepcopy(v)
def semantic(v):return H16.semantic(v)


def solve_standard_packet(packet):
    """Return HYBRID then NATIVE rows and one full coordinate-bank ledger."""
    p=H16.validate_hybrid_packet(packet);before=semantic(packet);parent=p['row']
    banks={};records=[];result=[]
    empty=dict(corners=[],lines=[],queries=[],selected_queries=0,raw_logits_sha256=None)
    native_contract=observation_points(p['native'],empty,p['H'])[2]
    for method,points,contract,adopted in (
        (METHODS[0],p['q'],p['contract'],p['adopted']),
        (METHODS[1],p['native'],native_contract,[])):
        # PoseBank's digest also contains K, xyz and image size. Array equality
        # confirms exact coordinates; no rounding or H-dependent cache key.
        key=points.tobytes()
        if key not in banks:banks[key]=PoseBank(points,p['K'],p['xyz'],image_size=p['size'])
        bank=banks[key]
        require(np.array_equal(bank.points,points,equal_nan=True) and np.array_equal(bank.K,p['K'])
            and np.array_equal(bank.xyz,p['xyz']) and bank.image_size==p['size']
            and bank.known_dimension_index is None,'same exact ordinary coordinate bank required')
        bank_before=clone(bank.ledger)
        solved=bank.solve(hidden=p['H'],robust=False)
        bank_after=clone(bank.ledger)
        require(solved['prior_used'] is False and solved['known_dimension_constraint_used'] is False
            and solved['solver']=='OBSERVATION_ONLY_STANDARD'
            and solved['hidden']==p['H'] and set(p['H']).isdisjoint(solved['fit_input_ids']),
            'ordinary backend or H fit contract differs')
        if solved['available']:
            require(solved['fit_input_ids']==solved['used'],'ordinary accepted fit is whole active U')
        stored_J_ids=list(solved['used']) if solved.get('geometry',{}).get('jacobian') else []
        row={k:clone(parent[k]) for k in ('id','session','K','xyz','raw_hw','selected_index','fixed_metadata')}
        row.update(assemble(p['native'],points,clone(parent['initial_pose']),p['H'],solved,
            'N3_VALIDATED_ROLE' if method==PRIMARY else 'NATIVE_STANDARD_PNP',contract))
        hybrid=method==PRIMARY
        row.update(method=method,parent_method=PARENT_METHOD,
            head_arm='IMAGE_ROLE' if hybrid else None,head_mode='IMAGE_ROLE' if hybrid else None,
            head_checkpoint_sha256=parent['head_checkpoint_sha256'] if hybrid else None,
            calibration_binding=clone(parent['calibration_binding']) if hybrid else None,
            observation_supply='CORNERWISE_HYBRID' if hybrid else 'NATIVE_N3',
            selected_corner_ids=list(adopted),original_native_N3_points=p['native'].tolist(),
            parent_observation_contract=clone(p['contract']),
            predicted_initial_N3_hidden=list(p['H']),original_self_hidden_initial=list(p['H']),
            initial_H_applied=True,standard_control_policy=clone(POLICY),
            stored_jacobian_fit_ids=stored_J_ids,ordinary_inlier_count_is_diagnostic_only=True,
            native_reference_id=parent['id'],native_N3_is_numeric_final_pose_observation=True,
            missing_sparse_filled_for_numeric_fit=False,partial_lines_not_pose_inputs=True,
            local_point_line_refinement=False,pose_estimation_kind='ORDINARY_WHOLE_U_PNP',
            numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
            final_numeric_pose_has_initial_prior=False,initial_pose_used_only_for_H_and_fallback=True,
            conditional_H_from_initial_N3=True,Base_proposal_feature_dependence_retained=hybrid,
            selection_validation_fit_excludes_candidate_corner=hybrid,
            selection_validation_is_fully_statistically_independent=False,
            selection_validation_has_shared_initial_prior=False,
            parent_geometry_row_semantic_sha256=semantic(parent),
            parent_observation_row_semantic_sha256=semantic(p['observation']),
            parent_native_N3_semantic_sha256=semantic(p['native']),parent_completion=clone(p['completion']),
            head_forward_calls=0,calibration_or_decode_calls=0,new_corner_selection_calls=0,
            cached_accuracy_row_used=False,solver_replay_latency_claim=False,oracle=False)
        if hybrid:
            row.update(cornerwise_selection=clone(contract['cornerwise_selection']),
                observation_raw_logits_sha256=parent.get('observation_raw_logits_sha256'))
        require(np.array_equal(np.asarray(row['input_points']),points,equal_nan=True)
            and np.array_equal(np.asarray(row['native_points'])[8],p['native'][8],equal_nan=True),
            'ordinary input or unchanged center differs')
        require(row['reprojections_reused_as_observations'] is False,'H projection reused as observation')
        records.append(dict(method=method,input_hash=bank.digest,bank_before=bank_before,
            bank_after=bank_after,operation_counts=clone(solved['operation_counts']),
            actual_active_ids=list(solved['used']),stored_jacobian_fit_ids=stored_J_ids,
            input_points_semantic_sha256=semantic(points),line_factor_count=0))
        result.append(row)
    require(semantic(packet)==before,'immutable original hybrid packet changed')
    ledger=dict(schema='native_hybrid_standard_paired_ledger_v17',id=parent['id'],session=parent['session'],
        methods=list(METHODS),actual_standard_solver_calls=2,reported_output_views=2,
        input_packet_semantic_sha256=before,parent_geometry_row_semantic_sha256=semantic(parent),
        parent_observation_row_semantic_sha256=semantic(p['observation']),
        parent_native_N3_semantic_sha256=semantic(p['native']),initial_H=list(p['H']),
        hybrid_adopted_boundary_ids=list(p['adopted']),ordinary_routes=records,
        coordinate_bank_count=len(banks),same_coordinate_bank_shared=len(banks)==1,
        same_input_q=np.array_equal(p['q'],p['native'],equal_nan=True),
        same_coordinate_bytes=p['q'].tobytes()==p['native'].tobytes(),
        coordinate_banks=[dict(input_hash=b.digest,counts=clone(b.ledger)) for b in banks.values()],
        bank_operation_counts={name:sum(b.ledger[name] for b in banks.values()) for name in next(iter(banks.values())).ledger},
        corner_selection_executed=False,endpoint_LOO_executed=False,line_factor_count=0,
        original_V4_corner_selection_copied=True,physical_ownership_truth='UNKNOWN',
        new_GT_scores_models_training_RGB_calls=0,latency_claim=False)
    return result,ledger
