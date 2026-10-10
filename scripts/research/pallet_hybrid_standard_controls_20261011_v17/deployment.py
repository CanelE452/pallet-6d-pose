"""Fresh fixed ROLE capture/LOO followed by whole-U ordinary hybrid PnP.

The actual selected-coordinate bank from unchanged V7 outcomes is reused.
Endpoint, line, LOCAL and paired native-control paths are not executed here.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import numpy as np

from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline as RolePipeline
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import preserve_prediction,assemble
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
from ..pallet_cornerwise_independent_20261010_v4.selection import POLICY as SELECTION_POLICY
from ..pallet_sparse_local_line_20261010_v9 import common as C
from . import pipeline as P

STAGES=('detector','N3_correction','initial_pose','boundary_observation',
    'hybrid_selection_and_standard_PnP','final_pose_reprojection_and_metadata')
RESERVED=('id','session','method','K','xyz','raw_hw','selected_index','fixed_metadata')

def assemble_result(point,native,selected,initial,H,contract,solved):
    row={k:copy.deepcopy(point[k]) for k in (*RESERVED,'head_arm','head_mode',
        'head_checkpoint_sha256','calibration_binding') if k in point}
    row.update(assemble(native,selected,initial,H,solved,'N3_VALIDATED_ROLE',contract))
    row.update(method=P.PRIMARY,parent_method=P.PARENT_METHOD,observation_supply='CORNERWISE_HYBRID',
        selected_corner_ids=list(contract['hybrid_boundary_corner_ids']),original_native_N3_points=np.asarray(native).tolist(),
        predicted_initial_N3_hidden=list(H),original_self_hidden_initial=list(H),initial_H_applied=True,
        parent_observation_contract=copy.deepcopy(contract),cornerwise_selection=copy.deepcopy(contract['cornerwise_selection']),
        observation_raw_logits_sha256=point.get('observation_raw_logits_sha256'),standard_control_policy=copy.deepcopy(P.POLICY),
        stored_jacobian_fit_ids=list(solved['used']) if solved.get('geometry',{}).get('jacobian') else [],
        ordinary_inlier_count_is_diagnostic_only=True,pose_estimation_kind='ORDINARY_WHOLE_U_PNP',
        local_point_line_refinement=False,partial_lines_not_pose_inputs=True,
        native_N3_is_numeric_final_pose_observation=True,missing_sparse_filled_for_numeric_fit=False,
        final_numeric_pose_has_initial_prior=False,initial_pose_used_only_for_H_and_fallback=True,
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        fresh_complete_capture=True,fresh_hybrid_corner_selection_recomputed_inside_call=True,
        fresh_point_solver_recomputed_inside_call=True,fresh_standard_solver_recomputed_inside_call=True,
        fresh_actual_selected_coordinate_bank_reused_inside_call=True,
        native_paired_accuracy_arm_executed_inside_primary_call=False,
        fresh_corner_selection_policy=copy.deepcopy(SELECTION_POLICY),
        accuracy_contract_zero_model_and_selection_counts_are_not_fresh_execution_counts=True,
        accuracy_contract_two_control_paths_are_not_fresh_primary_path_count=True,
        fresh_endpoint_LOO_recomputed_inside_call=False,line_factor_count=0,
        cached_accuracy_row_used=False,parent_completion_receipt_used_as_deployment_input=False,
        saved_accuracy_coordinates_or_pose_as_input=False,runtime_only_fresh_metadata=True,oracle=False)
    return row

class Pipeline:
    def __init__(self,args):
        self.base=RolePipeline(args,head_arms=('IMAGE_ROLE',));self.counts=Counter()
        self.ordinary_operation_counts=Counter();self.point_bank_capture_counts=Counter()
        self.packet_compatibility_counts=Counter();self.closed=False
        self.last_capture=self.last_point_result=self.last_point_bank_ledger=None
        self.last_standard_ledger=self.last_point_bank_reuse=None;self.last_execution_events=[]
    @property
    def torch(self):return self.base.torch
    @property
    def runtime(self):return self.base.runtime
    @property
    def old(self):return self.base.old
    @property
    def heads(self):return self.base.heads
    @property
    def models(self):return self.base.models
    @property
    def bindings(self):return self.base.bindings

    @contextmanager
    def capture_point_banks(self):
        namespace=self.base.outcomes.__func__.__globals__;original=namespace['PoseBank'];banks=[]
        P.require(original is PoseBank,'unchanged V7 point-bank class identity differs')
        def observed(*args,**kwargs):
            self.point_bank_capture_counts['constructor_entries']+=1
            try:bank=original(*args,**kwargs)
            except BaseException:self.point_bank_capture_counts['constructor_exceptions']+=1;raise
            self.point_bank_capture_counts['constructor_returns']+=1;banks.append(bank);return bank
        self.point_bank_capture_counts['observer_scopes_started']+=1;namespace['PoseBank']=observed
        try:yield banks
        finally:
            namespace['PoseBank']=original
            P.require(namespace['PoseBank'] is original,'V7 point constructor identity not restored')
            self.point_bank_capture_counts['observer_scopes_restored']+=1

    def packet(self,meta,capture,result):
        original=self.base.packet;expected=original(meta,capture,P.PRIMARY,{})
        payload=dict(result);removed=[]
        for field in RESERVED:
            if field in payload:
                C.require(C.semantic_sha256(payload[field])==C.semantic_sha256(expected[field]),
                    'identical original packet metadata required: '+field)
                del payload[field];removed.append(field)
        value=original(meta,capture,P.PRIMARY,payload)
        self.packet_compatibility_counts['calls']+=1;self.packet_compatibility_counts['identical_fields_removed']+=len(removed)
        value['packet_compatibility_review']=dict(removed_identical_fields=removed,numerical_result_changed=False,
            original_frame_metadata_preserved=True,original_packet_function_used=True)
        return value

    def predict(self,image,K,xyz,metadata=None,method=P.PRIMARY,stage_callback=None):
        P.require(method==P.PRIMARY,'single fresh V17 HYBRID ordinary primary required')
        self.last_capture=self.last_point_result=self.last_point_bank_ledger=None
        self.last_standard_ledger=self.last_point_bank_reuse=None;self.last_execution_events=[]
        self.counts['fresh_predict_calls_started']+=1;self.counts[method+':started']+=1
        with self.old.no_truth_reads(),self.torch.no_grad():
            capture=self.base.capture(image,K,xyz,metadata,need_arms=('IMAGE_ROLE',),need_n3=True,
                need_base_pose=False,stage_callback=stage_callback);self.last_capture=capture
            self.counts['fresh_hybrid_point_paths_started']+=1
            with self.capture_point_banks() as banks:rows,point_ledger=self.base.outcomes(capture,(P.PARENT_METHOD,))
            P.require(len(rows)==1 and 1<=len(banks)<=2,'one original hybrid point route and actual bank registry required')
            point=rows[0];self.last_point_result,self.last_point_bank_ledger=point,point_ledger
            selected=np.asarray(point['input_points'],float);native=np.asarray(capture['native_N3_points'],float)
            size=(image.shape[1],image.shape[0]);camera=np.asarray(K,float);dimensions=np.asarray(xyz,float)
            selected_banks=[b for b in banks if np.array_equal(b.points,selected,equal_nan=True)
                and np.array_equal(b.K,camera) and np.array_equal(b.xyz,dimensions)
                and b.image_size==size and b.known_dimension_index is None]
            P.require(len(selected_banks)==1,'unique actual selected q/K/dimensions/size bank required')
            bank=selected_banks[0];before=copy.deepcopy(bank.ledger);cache_before=set(bank.numeric_cache)
            self.counts['fresh_hybrid_point_paths_complete']+=1
            H=list(capture['hidden']);contract=point['observation_contract']
            self.counts['fresh_standard_paths_started']+=1;self.last_execution_events.append(dict(phase='ordinary_solve_started'))
            solved=bank.solve(hidden=H,robust=False)
            after=copy.deepcopy(bank.ledger);operations={k:after[k]-before[k] for k in after}
            cache_after=set(bank.numeric_cache);new_keys=sorted(cache_after-cache_before)
            P.require(solved['prior_used'] is False and solved['known_dimension_constraint_used'] is False
                and solved['solver']=='OBSERVATION_ONLY_STANDARD' and solved['hidden']==H
                and set(H).isdisjoint(solved['fit_input_ids'])
                and (not solved['available'] or solved['fit_input_ids']==solved['used']),
                'ordinary whole-U/H/prior contract differs')
            P.require(operations==solved['operation_counts'] and cache_before<=cache_after
                and operations['generic_cache_misses']==len(new_keys),'ordinary shared-bank ledger/cache differs')
            P.require(all(tuple(ids)==tuple(solved['used']) for _,ids in new_keys),
                'fresh ordinary generated an unrelated numeric problem')
            reuse=dict(schema='fresh_hybrid_standard_selected_bank_reuse_v17',actual_constructor_count=len(banks),
                constructor_restored=True,same_coordinate_bank_reused=True,selected_input_hash=bank.digest,
                selected_bank_before=before,selected_bank_after=after,ordinary_bank_delta=operations,
                numeric_cache_before=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(cache_before)],
                numeric_cache_after=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(cache_after)],
                newly_requested_numeric_problems=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in new_keys],
                additional_whole_U_Generic_calls=operations['generic_calls'],
                existing_numeric_keys_preserved=True,cache_reconstructed_from_serialized_Rt=False,
                new_whole_U_numeric_problem_may_be_required=True,raw_four_ID_candidates_reused_if_previously_requested=True)
            self.last_point_bank_reuse=reuse;self.ordinary_operation_counts.update(operations)
            self.last_standard_ledger=dict(schema='fresh_hybrid_ordinary_ledger_v17',ordinary_solver_calls=1,
                reported_primary_views=1,input_points=selected.tolist(),original_native_N3_points=native.tolist(),
                K=camera.tolist(),xyz=dimensions.tolist(),raw_hw=[image.shape[0],image.shape[1]],hidden_initial=H,
                original_hybrid_contract=copy.deepcopy(contract),solver=copy.deepcopy(solved),
                ordinary_operation_counts=copy.deepcopy(operations),actual_selected_point_bank_reuse=copy.deepcopy(reuse),
                line_factors=0,endpoint_LOO_calls=0,LOCAL_optimizer_calls=0,native_paired_control_calls=0)
            self.last_execution_events.append(dict(phase='ordinary_solve_returned'))
            self.counts['fresh_standard_paths_complete']+=1
            result=self.packet(capture['metadata'],capture,assemble_result(point,native,selected,capture['initial_pose'],H,contract,solved))
            result.update(fresh_point_solver=copy.deepcopy(point['solver']),fresh_point_coordinate_bank_ledger=copy.deepcopy(point_ledger),
                fresh_standard_ledger=copy.deepcopy(self.last_standard_ledger),fresh_actual_selected_bank_reuse=copy.deepcopy(reuse))
            self.counts['reported_primary_views_from_single_standard_solve']+=1
            if stage_callback:stage_callback('hybrid_selection_and_standard_PnP')
            prediction=copy.deepcopy(capture['prediction']);index=prediction['selected_index']
            if index is not None:prediction['candidates'][index]['keypoints_xy']=result['native_points']
            preserve_prediction(capture['raw'],prediction)
            P.require(np.array_equal(np.asarray(result['native_points'])[8],native[8],equal_nan=True),'fresh detector center changed')
            if stage_callback:stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction,original_base_points=capture['original_base_points'],native_N3_points=capture['native_N3_points'],
                observations=capture['observations'],observation=capture['observation'],fresh_capture_input_only=True)
        self.counts['fresh_predict_calls_complete']+=1;self.counts[method+':complete']+=1;return result

    def close(self):
        if not self.closed:self.closed=True;self.base.close()
