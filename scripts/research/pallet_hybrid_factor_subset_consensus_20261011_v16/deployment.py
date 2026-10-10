"""Fresh V4 hybrid selection, shared raw point roots and S15 point-line consensus.

The paired point-only accuracy ablation is not run in the primary deployment
call. Native and selected point banks are the actual banks used by unchanged
V7 outcomes. Both constructor scopes restore their original function identity.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import numpy as np

from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline as RolePipeline
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import preserve_prediction,assemble
from ..pallet_sparse_local_line_20261010_v9 import common as C
from . import pipeline_ready as P

STAGES=('detector','N3_correction','initial_pose','boundary_observation',
    'endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
RESERVED=('id','session','method','K','xyz','raw_hw','selected_index','fixed_metadata')

def assemble_result(point,native,selected,initial,H,contract,solved,ledger):
    row={k:copy.deepcopy(point[k]) for k in (*RESERVED,'head_arm','head_mode',
        'head_checkpoint_sha256','calibration_binding') if k in point}
    row.update(assemble(native,selected,initial,H,solved,'N3_VALIDATED_ROLE',contract))
    row.update(method=P.PRIMARY,parent_method=P.PARENT_METHOD,observation_supply='CORNERWISE_HYBRID',
        selected_corner_ids=list(contract['hybrid_boundary_corner_ids']),original_native_N3_points=np.asarray(native).tolist(),
        predicted_initial_N3_hidden=list(H),original_self_hidden_initial=list(H),initial_H_applied=True,
        hybrid_consensus_policy=copy.deepcopy(P.POLICY),cornerwise_selection=copy.deepcopy(contract['cornerwise_selection']),
        observation_filter=copy.deepcopy(ledger['observation_filter']),
        observation_raw_logits_sha256=point.get('observation_raw_logits_sha256'),
        pose_estimation_kind='FIXED_HYBRID_FACTOR_SUBSET_CONSENSUS',local_point_line_refinement=True,
        missing_sparse_filled_for_numeric_fit=False,native_N3_is_numeric_final_pose_observation=True,
        final_numeric_pose_has_initial_prior=False,initial_pose_may_supply_local_optimizer_start=True,
        original_lines_used_only_if_not_consumed_by_actual_adopted_points=True,partial_lines_not_pose_inputs=False,
        numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
        fresh_complete_capture=True,fresh_hybrid_corner_selection_recomputed_inside_call=True,
        hybrid_consensus_policy_is_frozen_accuracy_contract_copy=True,
        accuracy_contract_zero_model_and_selection_counts_are_not_fresh_execution_counts=True,
        fresh_corner_selection_policy=copy.deepcopy(P.S4.POLICY),
        fresh_point_solver_recomputed_inside_call=True,fresh_endpoint_LOO_recomputed_inside_call=True,
        fresh_raw_selected_and_native_point_roots_reused_inside_call=True,
        point_only_paired_accuracy_arm_executed_inside_primary_call=False,
        actual_fit_incidence_guard_applied_per_candidate=True,three_point_P3P_main_path_added=False,
        cached_accuracy_row_used=False,cached_endpoint_decisions_used=False,
        parent_completion_receipt_used_as_deployment_input=False,saved_accuracy_coordinates_or_pose_as_input=False,
        runtime_only_fresh_metadata=True,oracle=False)
    return row

class Pipeline:
    def __init__(self,args):
        self.base=RolePipeline(args,head_arms=('IMAGE_ROLE',));self.counts=Counter()
        self.local_operation_counts=Counter();self.endpoint_operation_counts=Counter()
        self.direct_point_operation_counts=Counter();self.point_bank_capture_counts=Counter()
        self.native_bank_reuse_counts=Counter();self.packet_compatibility_counts=Counter();self.closed=False
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
        namespace=self.base.outcomes.__func__.__globals__;original=namespace['PoseBank'];banks=[]
        P.require(original is P.PoseBank,'unchanged V7 point-bank class identity differs')
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

    @contextmanager
    def reuse_endpoint_native_bank(self,bank):
        # P15 normally constructs this same native-coordinate bank. Reuse the
        # actual bank already populated by V4 H-union-k selection; no serialized
        # pose reconstruction and no solver/class/source-file replacement.
        namespace=P.P15.repair_observations.__globals__;original=namespace['PoseBank']
        P.require(original is P.PoseBank,'unchanged endpoint point-bank class identity differs')
        def reused(points,K,xyz,*,image_size):
            self.native_bank_reuse_counts['constructor_requests']+=1
            P.require(np.array_equal(bank.points,np.asarray(points,float),equal_nan=True)
                and np.array_equal(bank.K,np.asarray(K,float)) and np.array_equal(bank.xyz,np.asarray(xyz,float))
                and bank.image_size==tuple(image_size) and bank.known_dimension_index is None,
                'fresh native endpoint bank coordinates/K/dimensions/size or prior differ')
            self.native_bank_reuse_counts['actual_native_bank_reuses']+=1;return bank
        self.native_bank_reuse_counts['observer_scopes_started']+=1;namespace['PoseBank']=reused
        try:yield
        finally:
            namespace['PoseBank']=original
            P.require(namespace['PoseBank'] is original,'endpoint point constructor identity not restored')
            self.native_bank_reuse_counts['observer_scopes_restored']+=1

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
        P.require(method==P.PRIMARY,'single fresh V16 primary required')
        self.last_capture=self.last_point_result=self.last_point_bank_ledger=self.last_repair_ledger=None
        self.last_point_bank_reuse=None;self.last_execution_events=[]
        self.counts['fresh_predict_calls_started']+=1;self.counts[method+':started']+=1
        with self.old.no_truth_reads(),self.torch.no_grad():
            capture=self.base.capture(image,K,xyz,metadata,need_arms=('IMAGE_ROLE',),need_n3=True,
                need_base_pose=False,stage_callback=stage_callback);self.last_capture=capture
            self.counts['fresh_hybrid_point_paths_started']+=1
            with self.capture_point_banks() as banks:rows,point_ledger=self.base.outcomes(capture,(P.PARENT_METHOD,))
            P.require(len(rows)==1 and 1<=len(banks)<=2,'one original hybrid point route and native/selected bank registry required')
            point=rows[0];self.last_point_result,self.last_point_bank_ledger=point,point_ledger
            selected=np.asarray(point['input_points'],float);native=np.asarray(capture['native_N3_points'],float)
            size=(image.shape[1],image.shape[0]);camera=np.asarray(K,float);dimensions=np.asarray(xyz,float)
            def matches(bank,points):
                return (np.array_equal(bank.points,points,equal_nan=True) and np.array_equal(bank.K,camera)
                    and np.array_equal(bank.xyz,dimensions) and bank.image_size==size and bank.known_dimension_index is None)
            selected_banks=[b for b in banks if matches(b,selected)];native_banks=[b for b in banks if matches(b,native)]
            P.require(len(selected_banks)==len(native_banks)==1,'unique actual selected/native q-bank identity required')
            shared,native_bank=selected_banks[0],native_banks[0];same_bank=shared is native_bank
            bank_before=copy.deepcopy(shared.ledger);native_before=copy.deepcopy(native_bank.ledger)
            native_cache_before=set(native_bank.numeric_cache);native_built_before=native_bank.hypotheses is not None
            self.counts['fresh_hybrid_point_paths_complete']+=1
            observation=capture['observations']['IMAGE_ROLE'];contract=point['observation_contract'];H=list(capture['hidden'])
            self.counts['fresh_repair_paths_started']+=1;self.counts['fresh_filtered_local_paths_started']+=1
            self.last_execution_events.append(dict(phase='factor_subset_repair_call_started'))
            def observed(event):
                self.last_execution_events.append(copy.deepcopy(event));phase=event['phase'];self.counts[phase+':events']+=1
                if phase=='endpoint_request_started':self.counts['fresh_native_endpoint_requests_started']+=1
                elif phase=='endpoint_request_returned':self.counts['fresh_native_endpoint_requests_complete']+=1
            with self.reuse_endpoint_native_bank(native_bank):
                solvers,ledger=P.P15.repair_observations(selected,native,camera,dimensions,observation,H,
                    contract['hybrid_boundary_corner_ids'],initial_pose=copy.deepcopy(capture['initial_pose']),
                    point_result=copy.deepcopy(point['solver']),image_size=size,observer=observed,point_bank=shared)
            bank_after=copy.deepcopy(shared.ledger);native_after=copy.deepcopy(native_bank.ledger)
            direct=copy.deepcopy(ledger['direct_point_bank_operation_counts'])
            endpoint_records=ledger['observation_filter']['endpoint_records']
            endpoint_phase_after=(endpoint_records[-1]['bank_after'] if endpoint_records else ledger['endpoint_native_bank_before'])
            endpoint={k:endpoint_phase_after[k]-ledger['endpoint_native_bank_before'][k] for k in native_after}
            selected_delta={k:bank_after[k]-bank_before[k] for k in bank_after}
            native_delta={k:native_after[k]-native_before[k] for k in native_after}
            P.require(all(v>=0 for v in (*direct.values(),*endpoint.values()))
                and selected_delta=={k:direct[k]+(endpoint[k] if same_bank else 0) for k in direct}
                and native_delta=={k:endpoint[k]+(direct[k] if same_bank else 0) for k in endpoint},
                'selected/native shared-bank phase deltas differ')
            P.require(direct['generic_calls']==direct['generic_cache_misses']==0,
                'identical fresh selected four-ID hypotheses numerically recomputed')
            native_cache_after=set(native_bank.numeric_cache);new_native_keys=sorted(native_cache_after-native_cache_before)
            P.require(native_cache_before<=native_cache_after and endpoint['generic_cache_misses']==len(new_native_keys),
                'native endpoint numeric cache does not preserve existing problems')
            if native_built_before:
                P.require(not any(len(ids)==4 for _,ids in new_native_keys),
                    'built native four-ID subset bank missed an identical hypothesis')
            reuse=dict(schema='fresh_hybrid_selected_native_point_bank_reuse_v16',actual_constructor_count=len(banks),
                constructor_restored=True,endpoint_constructor_restored=True,same_coordinate_bank_reused=True,
                selected_native_bank_is_same_object=same_bank,selected_input_hash=shared.digest,native_input_hash=native_bank.digest,
                selected_bank_before=bank_before,selected_bank_after=bank_after,selected_bank_delta=selected_delta,
                native_bank_before=native_before,native_bank_after=native_after,native_bank_delta=native_delta,
                direct_bank_delta=direct,endpoint_bank_delta=endpoint,
                additional_selected_Generic_calls=direct['generic_calls'],additional_native_Generic_calls=endpoint['generic_calls'],
                native_four_ID_bank_built_before_endpoint=native_built_before,
                native_numeric_cache_before=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(native_cache_before)],
                native_numeric_cache_after=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in sorted(native_cache_after)],
                newly_requested_native_numeric_problems=[dict(dimension_index=d,point_ids=list(ids)) for d,ids in new_native_keys],
                additional_native_four_ID_numeric_problems=sum(len(ids)==4 for _,ids in new_native_keys),
                distinct_native_refit_ID_problems_may_be_new=True,
                point_available_verdict_not_a_root_filter=True,cache_reconstructed_from_serialized_Rt=False,
                original_cornerwise_policy=copy.deepcopy(P.S4.POLICY),fresh_cornerwise_selection=copy.deepcopy(contract['cornerwise_selection']))
            self.last_point_bank_reuse=reuse;ledger['fresh_actual_hybrid_bank_reuse']=copy.deepcopy(reuse)
            ledger['fresh_native_endpoint_phase_after']=copy.deepcopy(endpoint_phase_after)
            ledger['fresh_native_endpoint_phase_operation_counts']=copy.deepcopy(endpoint)
            ledger['native_end_of_repair_ledger_can_include_selected_direct_hits']=same_bank
            self.last_repair_ledger=ledger;self.local_operation_counts.update(ledger['local_operation_counts'])
            self.direct_point_operation_counts.update(direct);self.endpoint_operation_counts.update(endpoint)
            P.require(set(solvers)=={P.P15.PRIMARY} and ledger['actual_filtered_local_solve_calls']==ledger['reported_output_views']==1
                and ledger['endpoint_decision_source']=='FRESH_NATIVE_ENDPOINT_LOO', 'single primary filtered path required')
            self.last_execution_events.append(dict(phase='factor_subset_repair_call_returned'))
            self.counts['fresh_filtered_local_paths_complete']+=1
            result=self.packet(capture['metadata'],capture,assemble_result(point,native,selected,capture['initial_pose'],H,
                contract,solvers[P.P15.PRIMARY],ledger))
            result.update(fresh_point_solver=copy.deepcopy(point['solver']),fresh_point_coordinate_bank_ledger=copy.deepcopy(point_ledger),
                fresh_repair_ledger=copy.deepcopy(ledger),fresh_actual_hybrid_bank_reuse=copy.deepcopy(reuse))
            self.counts['fresh_repair_paths_complete']+=1;self.counts['reported_primary_views_from_single_filtered_local']+=1
            if stage_callback:stage_callback('endpoint_validation_and_filtered_local')
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
