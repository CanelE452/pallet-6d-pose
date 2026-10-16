"""Fresh factor-subset consensus with actual in-memory four-ID cache reuse.

The detector, fixed N3, ROLE head, sparse point control, native endpoint LOO,
mixed proposals, actual-inlier refits and final H projection run in one call.
No sealed observations, audit decisions, reference poses or accuracy rows are
prediction inputs. A scoped constructor observer retains the actual point bank.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import numpy as np

from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline as RolePipeline
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import preserve_prediction,assemble
from ..pallet_sparse_local_line_20261010_v9 import common as C
from . import pipeline as P

STAGES=('detector','N3_correction','initial_pose','boundary_observation',
    'endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
RESERVED=('id','session','method','K','xyz','raw_hw','selected_index','fixed_metadata')

def assemble_result(point,native,sparse,initial,H,contract,solved,ledger):
    row={k:copy.deepcopy(point[k]) for k in (*RESERVED,'head_arm','head_mode',
        'head_checkpoint_sha256','calibration_binding') if k in point}
    row.update(assemble(native,sparse,initial,H,solved,'VALIDATED_ROLE_ONLY',contract))
    row.update(method=P.PRIMARY,comparator='FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE',parent_method=P.V11.PARENT_PRIMARY,
        observation_supply='BOUNDARY_ONLY',selected_corner_ids=list(contract['validated_boundary_corner_ids']),
        original_native_N3_points=np.asarray(native).tolist(),predicted_initial_N3_hidden=list(H),
        original_self_hidden_initial=list(H),initial_H_applied=True,factor_subset_consensus_policy=copy.deepcopy(P.POLICY),
        observation_filter=copy.deepcopy(ledger['observation_filter']),
        pose_estimation_kind='FINITE_POINT_AND_MIXED_FACTOR_SUBSET_LOCAL_CONSENSUS',local_point_line_refinement=True,
        missing_sparse_filled_for_numeric_fit=False,native_N3_is_numeric_final_pose_observation=False,
        final_numeric_pose_has_initial_prior=False,initial_pose_may_supply_local_optimizer_start=True,
        original_lines_used_only_if_not_consumed_by_actual_adopted_points=True,
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        fresh_complete_capture=True,fresh_point_solver_recomputed_inside_call=True,
        fresh_endpoint_LOO_recomputed_inside_call=True,fresh_raw_point_roots_reused_inside_call=True,
        fresh_point_result_available_verdict_is_not_a_root_filter=True,
        actual_fit_incidence_guard_applied_per_candidate=True,three_point_P3P_main_path_added=False,
        cached_accuracy_row_used=False,cached_V10_endpoint_decisions_used=False,
        parent_completion_receipt_used_as_deployment_input=False,saved_accuracy_coordinates_or_pose_as_input=False,
        runtime_only_fresh_metadata=True,kernel_start_source_labels_are_identifiers_not_receipts=True,
        kernel_start_source_provenance=dict(ORIGINAL_INITIAL_N3_ONLY='fresh initial N3 from this call; mixed LOCAL starts only'),
        point_only_and_partial_line_paths_are_separate_ablation_components=True,oracle=False)
    return row

class Pipeline:
    def __init__(self,args):
        self.base=RolePipeline(args,head_arms=('IMAGE_ROLE',));self.counts=Counter()
        self.local_operation_counts=Counter();self.endpoint_operation_counts=Counter()
        self.direct_point_operation_counts=Counter();self.point_bank_capture_counts=Counter()
        self.packet_compatibility_counts=Counter();self.closed=False
        self.last_capture=self.last_point_result=self.last_point_bank_ledger=self.last_repair_ledger=None
        self.last_point_bank_reuse=None;self.last_execution_events=[]

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
        # Observe the exact constructor used by the unchanged outcomes body;
        # no old source edit, class replacement or numeric cache reconstruction.
        namespace=self.base.outcomes.__func__.__globals__;original=namespace['PoseBank'];banks=[]
        P.require(original is P.PoseBank,'actual original point-bank class identity differs')
        def observed(*args,**kwargs):
            self.point_bank_capture_counts['constructor_entries']+=1
            try:bank=original(*args,**kwargs)
            except BaseException:self.point_bank_capture_counts['constructor_exceptions']+=1;raise
            self.point_bank_capture_counts['constructor_returns']+=1;banks.append(bank);return bank
        self.point_bank_capture_counts['observer_scopes_started']+=1;namespace['PoseBank']=observed
        try:yield banks
        finally:
            namespace['PoseBank']=original
            P.require(namespace['PoseBank'] is original,'scoped actual point constructor not restored')
            self.point_bank_capture_counts['observer_scopes_restored']+=1

    def packet(self,meta,capture,result):
        original=self.base.packet;expected=original(meta,capture,P.PRIMARY,{})
        payload=dict(result);removed=[]
        for field in RESERVED:
            if field in payload:
                C.require(C.semantic_sha256(payload[field])==C.semantic_sha256(expected[field]),
                    'identical packet metadata required: '+field)
                del payload[field];removed.append(field)
        value=original(meta,capture,P.PRIMARY,payload)
        self.packet_compatibility_counts['calls']+=1;self.packet_compatibility_counts['identical_fields_removed']+=len(removed)
        value['packet_compatibility_review']=dict(removed_identical_fields=removed,numerical_result_changed=False,
            original_frame_metadata_preserved=True,original_packet_function_used=True)
        return value

    def predict(self,image,K,xyz,metadata=None,method=P.PRIMARY,stage_callback=None):
        P.require(method==P.PRIMARY,'single fresh V15 method required')
        self.last_capture=self.last_point_result=self.last_point_bank_ledger=self.last_repair_ledger=None
        self.last_point_bank_reuse=None;self.last_execution_events=[]
        self.counts['fresh_predict_calls_started']+=1;self.counts[method+':started']+=1
        with self.old.no_truth_reads(),self.torch.no_grad():
            capture=self.base.capture(image,K,xyz,metadata,need_arms=('IMAGE_ROLE',),need_n3=True,
                need_base_pose=False,stage_callback=stage_callback);self.last_capture=capture
            self.counts['fresh_sparse_point_paths_started']+=1
            with self.capture_point_banks() as banks:rows,point_ledger=self.base.outcomes(capture,(P.V11.PARENT_PRIMARY,))
            P.require(len(rows)==len(banks)==1,'one actual sameq sparse point bank required')
            point,shared=rows[0],banks[0];self.last_point_result,self.last_point_bank_ledger=point,point_ledger
            self.counts['fresh_sparse_point_paths_complete']+=1
            sparse=np.asarray(point['input_points'],float);native=np.asarray(capture['native_N3_points'],float)
            observation=capture['observations']['IMAGE_ROLE'];contract=point['observation_contract'];H=list(capture['hidden'])
            P.require(np.array_equal(shared.points,sparse,equal_nan=True) and np.array_equal(shared.K,np.asarray(K,float))
                and np.array_equal(shared.xyz,np.asarray(xyz,float)) and shared.known_dimension_index is None,
                'actual captured point bank/q/K/xyz or independent dimension prior differs')
            bank_before=shared.ledger.copy();self.counts['fresh_repair_paths_started']+=1
            self.counts['fresh_filtered_local_paths_started']+=1
            self.last_execution_events.append(dict(phase='factor_subset_repair_call_started'))
            def observed(event):
                self.last_execution_events.append(copy.deepcopy(event));phase=event['phase'];self.counts[phase+':events']+=1
                if phase=='endpoint_request_started':self.counts['fresh_native_endpoint_requests_started']+=1
                elif phase=='endpoint_request_returned':self.counts['fresh_native_endpoint_requests_complete']+=1
            solvers,ledger=P.repair_observations(sparse,native,np.asarray(K),np.asarray(xyz),observation,H,
                contract['validated_boundary_corner_ids'],initial_pose=copy.deepcopy(capture['initial_pose']),
                point_result=copy.deepcopy(point['solver']),image_size=(image.shape[1],image.shape[0]),observer=observed,
                point_bank=shared)
            bank_after=shared.ledger.copy();delta={k:bank_after[k]-bank_before[k] for k in bank_after}
            P.require(ledger['actual_direct_point_bank_reused'] is True and delta==ledger['direct_point_bank_operation_counts']
                and delta['generic_calls']==0 and delta['generic_cache_misses']==0,
                'same q/K/dim/four-ID Generic problem was numerically recomputed')
            reuse=dict(schema='fresh_actual_point_bank_reuse_v15',input_hash=shared.digest,
                actual_constructor_count=len(banks),constructor_restored=True,same_coordinate_bank_reused=True,
                bank_before=bank_before,bank_after=bank_after,bank_delta=delta,
                additional_direct_Generic_calls=delta['generic_calls'],numeric_cache_entries=len(shared.numeric_cache),
                raw_four_ID_root_cache_hits=delta['generic_cache_hits'],point_available_verdict_not_a_root_filter=True,
                cache_reconstructed_from_serialized_Rt=False)
            self.last_point_bank_reuse=reuse;ledger['fresh_actual_point_bank_reuse']=copy.deepcopy(reuse)
            self.last_repair_ledger=ledger;self.local_operation_counts.update(ledger['local_operation_counts'])
            self.direct_point_operation_counts.update(delta)
            before,after=ledger['endpoint_native_bank_before'],ledger['endpoint_native_bank_after']
            native_delta={k:after[k]-before[k] for k in after}
            P.require(all(v>=0 for v in native_delta.values()),'negative fresh native operation delta')
            self.endpoint_operation_counts.update(native_delta)
            P.require(set(solvers)=={P.PRIMARY} and ledger['actual_filtered_local_solve_calls']==ledger['reported_output_views']==1
                and ledger['endpoint_decision_source']=='FRESH_NATIVE_ENDPOINT_LOO'
                and ledger['endpoint_validation_requests']==ledger['endpoint_decision_records'],
                'single fresh subset solve or endpoint validation skipped')
            self.last_execution_events.append(dict(phase='factor_subset_repair_call_returned'))
            self.counts['fresh_filtered_local_paths_complete']+=1
            result=self.packet(capture['metadata'],capture,assemble_result(point,native,sparse,capture['initial_pose'],H,
                contract,solvers[P.PRIMARY],ledger))
            result.update(fresh_point_solver=copy.deepcopy(point['solver']),fresh_point_coordinate_bank_ledger=copy.deepcopy(point_ledger),
                fresh_repair_ledger=copy.deepcopy(ledger),fresh_actual_point_bank_reuse=copy.deepcopy(reuse))
            self.counts['fresh_repair_paths_complete']+=1;self.counts['reported_views_from_single_filtered_local']+=1
            if stage_callback:stage_callback('endpoint_validation_and_filtered_local')
            prediction=copy.deepcopy(capture['prediction']);index=prediction['selected_index']
            if index is not None:prediction['candidates'][index]['keypoints_xy']=result['native_points']
            preserve_prediction(capture['raw'],prediction)
            P.require(np.array_equal(np.asarray(result['native_points'])[8],native[8],equal_nan=True),'fresh detector center changed')
            if stage_callback:stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction,original_base_points=capture['original_base_points'],native_N3_points=capture['native_N3_points'],
                observations=capture['observations'],observation=capture['observation'],fresh_capture_input_only=True)
        self.counts['fresh_predict_calls_complete']+=1;self.counts[method+':complete']+=1
        return result

    def close(self):
        if not self.closed:self.closed=True;self.base.close()
