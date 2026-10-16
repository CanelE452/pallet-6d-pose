"""Fresh ROLE hybrid selection and one restored point-only consensus solve.

The actual V7 selected-coordinate bank is shared with the unchanged numerical
S15 point-only solver. No V16 saved coordinate, pose or line audit is an input.
"""
from collections import Counter
import copy
import numpy as np

from ..pallet_hybrid_standard_controls_20261011_v17.deployment import Pipeline as Frontend,RESERVED
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import preserve_prediction,assemble
from ..pallet_cornerwise_independent_20261010_v4.selection import POLICY as SELECTION_POLICY
from ..pallet_sparse_local_line_20261010_v9 import common as C
from . import pipeline as P

ROLE_PARENT='IMAGE_ROLE_CORNERWISE_HYBRID'
STAGES=('detector','N3_correction','initial_pose','boundary_observation',
    'hybrid_selection_and_point_consensus','final_pose_reprojection_and_metadata')

def assemble_result(point,native,selected,initial,H,contract,solved,ledger):
    row={k:copy.deepcopy(point[k]) for k in (*RESERVED,'head_arm','head_mode',
        'head_checkpoint_sha256','calibration_binding') if k in point}
    row.update(assemble(native,selected,initial,H,solved,'N3_VALIDATED_ROLE',contract))
    row.update(method=P.PRIMARY,parent_method=ROLE_PARENT,observation_supply='CORNERWISE_HYBRID',
        selected_corner_ids=list(contract['hybrid_boundary_corner_ids']),original_native_N3_points=np.asarray(native).tolist(),
        predicted_initial_N3_hidden=list(H),original_self_hidden_initial=list(H),initial_H_applied=True,
        parent_observation_contract=copy.deepcopy(contract),cornerwise_selection=copy.deepcopy(contract['cornerwise_selection']),
        observation_raw_logits_sha256=point.get('observation_raw_logits_sha256'),decision_restoration_policy=copy.deepcopy(P.POLICY),
        operative_output_decision_policy=copy.deepcopy(P.POLICY),decision_restoration=copy.deepcopy(solved['decision_restoration']),
        decision_execution_kind='FRESH_ORIGINAL_POINT_ONLY_SOLVE',
        pose_estimation_kind='FINITE_FOUR_POINT_CONSENSUS_COUNT_SSE_DECISION',
        local_point_line_refinement=False,partial_lines_not_pose_inputs=True,
        native_N3_is_numeric_final_pose_observation=True,missing_sparse_filled_for_numeric_fit=False,
        final_numeric_pose_has_initial_prior=False,initial_pose_used_only_for_H_and_fallback=True,
        selected_output_pose_was_actually_fit_in_this_fresh_call=bool(solved['available']),
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        fresh_complete_capture=True,fresh_hybrid_corner_selection_recomputed_inside_call=True,
        fresh_point_solver_recomputed_inside_call=True,fresh_original_point_consensus_solver_recomputed_inside_call=True,
        fresh_actual_selected_coordinate_bank_reused_inside_call=True,
        fresh_corner_selection_policy=copy.deepcopy(SELECTION_POLICY),
        fresh_endpoint_LOO_recomputed_inside_call=False,line_factor_count=0,
        numerical_candidate_generator_refit_and_thresholds_unchanged=True,
        saved_decision_replay=False,cached_accuracy_row_used=False,parent_completion_receipt_used_as_deployment_input=False,
        saved_accuracy_coordinates_or_pose_as_input=False,runtime_only_fresh_metadata=True,
        inherited_accuracy_zero_solver_counts_are_not_fresh_execution_counts=True,
        original_paired_accuracy_arm_executed_inside_primary_call=False,oracle=False,
        fresh_consensus_ledger=copy.deepcopy(ledger))
    return row

class Pipeline(Frontend):
    """Reuse audited capture/constructor helpers; dispatch a new single solve."""
    def __init__(self,args):
        super().__init__(args);self.consensus_operation_counts=Counter();self.direct_point_operation_counts=Counter()
        self.guard_operation_counts=Counter();self.decision_counts=Counter()
        self.last_consensus_ledger=None;self.fresh_bank_type=P.fresh_bank_class()

    def packet(self,meta,capture,result):
        original=self.base.packet;expected=original(meta,capture,P.PRIMARY,{})
        payload=dict(result);removed=[]
        for field in RESERVED:
            if field in payload:
                P.require(C.semantic_sha256(payload[field])==C.semantic_sha256(expected[field]),
                    'identical original packet metadata required: '+field)
                del payload[field];removed.append(field)
        value=original(meta,capture,P.PRIMARY,payload)
        self.packet_compatibility_counts['calls']+=1;self.packet_compatibility_counts['identical_fields_removed']+=len(removed)
        value['packet_compatibility_review']=dict(removed_identical_fields=removed,numerical_result_changed=False,
            original_frame_metadata_preserved=True,original_packet_function_used=True)
        return value

    def predict(self,image,K,xyz,metadata=None,method=P.PRIMARY,stage_callback=None):
        P.require(method==P.PRIMARY,'single fresh V18 restored point-consensus primary required')
        self.last_capture=self.last_point_result=self.last_point_bank_ledger=self.last_consensus_ledger=None
        self.last_point_bank_reuse=None;self.last_execution_events=[]
        self.counts['fresh_predict_calls_started']+=1;self.counts[method+':started']+=1
        with self.old.no_truth_reads(),self.torch.no_grad():
            capture=self.base.capture(image,K,xyz,metadata,need_arms=('IMAGE_ROLE',),need_n3=True,
                need_base_pose=False,stage_callback=stage_callback);self.last_capture=capture
            self.counts['fresh_hybrid_point_paths_started']+=1
            with self.capture_point_banks() as banks:rows,point_ledger=self.base.outcomes(capture,(ROLE_PARENT,))
            P.require(len(rows)==1 and 1<=len(banks)<=2,'one original hybrid point route and actual bank registry required')
            point=rows[0];self.last_point_result,self.last_point_bank_ledger=point,point_ledger
            selected=np.asarray(point['input_points'],float);native=np.asarray(capture['native_N3_points'],float)
            camera=np.asarray(K,float);dimensions=np.asarray(xyz,float);size=(image.shape[1],image.shape[0])
            selected_banks=[b for b in banks if np.array_equal(b.points,selected,equal_nan=True)
                and np.array_equal(b.K,camera) and np.array_equal(b.xyz,dimensions)
                and b.image_size==size and b.known_dimension_index is None]
            P.require(len(selected_banks)==1,'unique actual selected q/K/dimensions/size bank required')
            shared=selected_banks[0];before=copy.deepcopy(shared.ledger);cache_before=set(shared.numeric_cache)
            self.counts['fresh_hybrid_point_paths_complete']+=1;H=list(capture['hidden']);contract=point['observation_contract']
            observation=copy.deepcopy(capture['observations']['IMAGE_ROLE']);observation['lines']=[]
            before_observation=C.semantic_sha256(observation)
            bank=self.fresh_bank_type(selected,camera,dimensions,image_size=size,point_bank=shared)
            self.counts['fresh_consensus_paths_started']+=1;self.last_execution_events.append(dict(phase='point_consensus_solve_started'))
            solved=bank.solve(observation,adopted_boundary_corner_ids=(),hidden=H,
                initial_pose=copy.deepcopy(capture['initial_pose']),point_result=copy.deepcopy(point['solver']))
            after=copy.deepcopy(shared.ledger);direct={k:after[k]-before[k] for k in after};cache_after=set(shared.numeric_cache)
            P.require(before_observation==C.semantic_sha256(observation) and solved['line_edges']==solved['fit_line_edges']==solved['consumed_edges']==[],
                'fresh point-only observation changed or line entered fit')
            P.require(solved['prior_used'] is False and solved['known_dimension_constraint_used'] is False
                and solved['hidden']==H and set(H).isdisjoint(solved['fit_input_ids'])
                and (not solved['available'] or len(solved['fit_input_ids'])>=4),'fixed point-only/H/no-prior contract differs')
            P.require(direct==solved['hypothesis_bank_operation_counts'] and cache_before==cache_after
                and direct['generic_calls']==direct['generic_cache_misses']==0,
                'identical selected four-ID hypotheses numerically recomputed')
            operations=copy.deepcopy(solved['operation_counts']);self.consensus_operation_counts.update(operations)
            self.direct_point_operation_counts.update(direct)
            guards={k:v for k,v in operations.items() if k.startswith('stable_incidence_')};self.guard_operation_counts.update(guards)
            decision=solved['decision_restoration'];self.decision_counts['fresh_original_solver_calls']+=1
            self.decision_counts['pure_output_decisions']+=1;self.decision_counts['restored']+=int(decision['restored'])
            self.decision_counts['true_tie_retained']+=int(decision['true_non_equivalent_numeric_tie_retained'])
            self.decision_counts['new_display_H_assignments']+=decision['new_display_H_assignments']
            reuse=dict(schema='fresh_point_consensus_selected_bank_reuse_v18',actual_constructor_count=len(banks),
                constructor_restored=True,same_coordinate_bank_reused=True,selected_input_hash=shared.digest,
                selected_bank_before=before,selected_bank_after=after,direct_bank_delta=direct,
                numeric_cache_before=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(cache_before)],
                numeric_cache_after=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(cache_after)],
                additional_Generic_calls=direct['generic_calls'],additional_Generic_cache_misses=direct['generic_cache_misses'],
                existing_numeric_keys_preserved=True,cache_reconstructed_from_serialized_Rt=False)
            self.last_point_bank_reuse=reuse
            self.last_consensus_ledger=dict(schema='fresh_restored_point_consensus_ledger_v18',original_point_only_solver_calls=1,
                pure_output_decisions=1,reported_primary_views=1,second_solver_calls=0,input_points=selected.tolist(),
                original_native_N3_points=native.tolist(),K=camera.tolist(),xyz=dimensions.tolist(),
                raw_hw=[image.shape[0],image.shape[1]],hidden_initial=H,original_hybrid_contract=copy.deepcopy(contract),
                point_only_observation_semantic_sha256=before_observation,solver=copy.deepcopy(solved),
                consensus_operation_counts=operations,direct_point_bank_operation_counts=direct,
                stable_incidence_operation_counts=guards,decision=copy.deepcopy(decision),
                actual_selected_point_bank_reuse=copy.deepcopy(reuse),line_factors=0,endpoint_LOO_calls=0,
                native_paired_control_calls=0,saved_accuracy_row_as_input=False)
            self.last_execution_events.append(dict(phase='point_consensus_solve_returned'))
            self.counts['fresh_consensus_paths_complete']+=1
            result=self.packet(capture['metadata'],capture,assemble_result(point,native,selected,capture['initial_pose'],H,contract,solved,self.last_consensus_ledger))
            result.update(fresh_point_solver=copy.deepcopy(point['solver']),fresh_point_coordinate_bank_ledger=copy.deepcopy(point_ledger),
                fresh_actual_selected_bank_reuse=copy.deepcopy(reuse))
            self.counts['reported_primary_views_from_single_consensus_solve']+=1
            if stage_callback:stage_callback('hybrid_selection_and_point_consensus')
            prediction=copy.deepcopy(capture['prediction']);index=prediction['selected_index']
            if index is not None:prediction['candidates'][index]['keypoints_xy']=result['native_points']
            preserve_prediction(capture['raw'],prediction)
            P.require(np.array_equal(np.asarray(result['native_points'])[8],native[8],equal_nan=True),'fresh detector center changed')
            if stage_callback:stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction,original_base_points=capture['original_base_points'],native_N3_points=capture['native_N3_points'],
                observations=capture['observations'],observation=capture['observation'],fresh_capture_input_only=True)
        self.counts['fresh_predict_calls_complete']+=1;self.counts[method+':complete']+=1;return result
