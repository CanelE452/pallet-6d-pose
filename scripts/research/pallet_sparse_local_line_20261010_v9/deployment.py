"""Fresh complete V9 deployment, composing the unchanged single ROLE capture.

This adapter never receives sealed parent rows, population receipts, reference
poses or saved coordinates. Each predict call captures its own detector/N3
observations. The fresh point-only solve supplies the available local start.
"""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np

from . import common as C
from .solver import LocalPointLineBank, POLICY
from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline as RolePipeline
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble, preserve_prediction

ARMS = ('BASE', 'N3_SUBPIX', C.COMPARATOR, C.PRIMARY)
STAGES = ('detector', 'N3_correction', 'initial_pose', 'boundary_observation',
          'final_pose_reprojection_and_metadata')


class Pipeline:
    """One model lifecycle with four fresh requested paths and one loaded head."""

    def __init__(self, args):
        self.base = RolePipeline(args, head_arms=('IMAGE_ROLE',))
        self.counts = Counter()
        self.local_operation_counts = Counter()
        self.last_capture = None
        self.last_point_result = None
        self.last_point_bank_ledger = None
        self.last_local_bank = None
        self.closed = False

    @property
    def torch(self):
        return self.base.torch

    @property
    def runtime(self):
        return self.base.runtime

    @property
    def old(self):
        return self.base.old

    @property
    def heads(self):
        return self.base.heads

    @property
    def models(self):
        return self.base.models

    @property
    def bindings(self):
        return self.base.bindings

    def predict(self, image, K, xyz, metadata=None, method=C.PRIMARY, stage_callback=None):
        C.require(method in ARMS, 'unknown fresh V9 deployment route')
        # Local starts and controls are recomputed within this call. Nothing
        # from a prior frame/call or the accuracy reference enters capture.
        self.last_capture = self.last_point_result = self.last_point_bank_ledger = self.last_local_bank = None
        learned = method in (C.COMPARATOR, C.PRIMARY)
        fixed_base = method == 'BASE'
        self.counts['fresh_predict_calls_started'] += 1
        self.counts[method + ':started'] += 1
        with self.old.no_truth_reads(), self.torch.no_grad():
            capture = self.base.capture(image, K, xyz, metadata,
                need_arms=('IMAGE_ROLE',) if learned else (), need_n3=not fixed_base,
                need_base_pose=False, stage_callback=stage_callback)
            self.last_capture = capture
            if fixed_base:
                capture['initial_base_pose'] = capture['initial_pose']
            if not learned:
                result = self.base.fixed_result(capture, method)
            else:
                self.counts['fresh_sparse_point_paths_started'] += 1
                point_rows, point_bank_ledger = self.base.outcomes(capture, (C.PARENT_PRIMARY,))
                C.require(len(point_rows) == 1, 'fresh point result population differs')
                point = point_rows[0]
                self.last_point_result, self.last_point_bank_ledger = point, point_bank_ledger
                self.counts['fresh_sparse_point_paths_complete'] += 1
                if method == C.COMPARATOR:
                    result = copy.deepcopy(point)
                    result.update(method=C.COMPARATOR, fresh_point_control_equivalent_parent_method=C.PARENT_PRIMARY,
                        existing_control_replay=False, cached_accuracy_row_used=False,
                        fresh_complete_capture=True, runtime_only_fresh_metadata=True)
                else:
                    sparse = np.asarray(point['input_points'], float)
                    native = np.asarray(capture['native_N3_points'], float)
                    contract = point['observation_contract']
                    H = list(capture['hidden'])
                    size = (image.shape[1], image.shape[0])
                    self.counts['fresh_local_banks_started'] += 1
                    bank = LocalPointLineBank(sparse, np.asarray(K), np.asarray(xyz), image_size=size)
                    self.last_local_bank = bank
                    self.counts['fresh_local_banks_complete'] += 1
                    self.counts['fresh_local_paths_started'] += 1
                    solved = bank.solve(capture['observations']['IMAGE_ROLE'],
                        adopted_boundary_corner_ids=contract['validated_boundary_corner_ids'], hidden=H,
                        initial_pose=copy.deepcopy(capture['initial_pose']), point_result=copy.deepcopy(point['solver']))
                    self.counts['fresh_local_paths_complete'] += 1
                    self.local_operation_counts.update(solved['operation_counts'])
                    assembled = assemble(native, sparse, capture['initial_pose'], H, solved,
                        'VALIDATED_ROLE_ONLY', contract)
                    # Preserve fresh head/source metadata while replacing every
                    # numerical fit/display/status field with the local output.
                    local = {key: copy.deepcopy(point[key]) for key in (
                        'head_arm', 'head_mode', 'head_checkpoint_sha256', 'calibration_binding',
                        'mask_diagnostic', 'observation_raw_logits_sha256', 'selected_queries',
                        'source_lines', 'model_query_anchor', 'Base_proposal_feature_dependence_retained')}
                    local.update(assembled)
                    local.update(comparator=C.COMPARATOR, observation_supply='BOUNDARY_ONLY',
                        selected_corner_ids=list(contract['validated_boundary_corner_ids']),
                        predicted_initial_N3_hidden=H, original_self_hidden_initial=H,
                        original_native_N3_points=native.tolist(), initial_H_applied=True,
                        local_point_line_policy=copy.deepcopy(POLICY), local_point_line_refinement=True,
                        pose_estimation_kind='LOCAL_POINT_LINE_REFINEMENT', independent_PnP_or_PnL=False,
                        partial_lines_not_pose_inputs=False, final_numeric_pose_has_initial_prior=False,
                        initial_pose_used_only_for_H_and_fallback=False,
                        initial_pose_may_supply_local_optimizer_start=True,
                        initial_pose_source='fresh fixed N3_SUBPIX and available fresh sparse point pose; LOCAL starts only',
                        fresh_point_start_recomputed_inside_call=True, fresh_point_method=C.PARENT_PRIMARY,
                        conditional_H_from_initial_N3=True, missing_sparse_filled_for_numeric_fit=False,
                        native_N3_is_numeric_final_pose_observation=False, display_missing_uses_native_N3=True,
                        stored_jacobian_fit_ids=list(solved['fit_input_ids']) if solved['available'] else [],
                        stored_jacobian_fit_line_edges=list(solved['fit_line_edges']) if solved['available'] else [],
                        stored_jacobian_scope='whole actual pool;8px diagnostic inliers are not an acceptance gate',
                        original_lines_used_only_if_not_consumed_by_actual_adopted_points=True,
                        full_candidate_and_operation_witnesses_preserved=True,
                        fresh_complete_capture=True, cached_accuracy_row_used=False,
                        parent_completion_receipt_used_as_deployment_input=False,
                        sealed_parent_packet_adapter_called=False, runtime_only_fresh_metadata=True,
                        kernel_start_source_labels_are_identifiers_not_receipts=True,
                        kernel_start_source_provenance=dict(SEALED_INITIAL_N3='fresh initial N3 from this call',
                            SEALED_AVAILABLE_POINT_ONLY='available fresh sparse point solve from this call'),
                        global_uniqueness_proven=False, numeric_NEW_is_not_pose_accuracy_success=True,
                        ensemble_observations=False, oracle=False)
                    result = self.base.packet(capture['metadata'], capture, C.PRIMARY, local)
                    result['fresh_point_solver'] = copy.deepcopy(point['solver'])
                    result['fresh_point_coordinate_bank_ledger'] = copy.deepcopy(point_bank_ledger)
                    result['fresh_local_coordinate_bank_ledger'] = dict(input_hash=bank.bank.digest, **dict(bank.bank.ledger))
                    C.require(np.array_equal(np.asarray(result['input_points']), sparse, equal_nan=True), 'fresh sparse fit was filled')
                    C.require(set(H).isdisjoint(solved['fit_input_ids']), 'fresh self-hidden initial coordinate entered local fit')
            prediction = copy.deepcopy(capture['prediction'])
            index = prediction['selected_index']
            if index is not None:
                prediction['candidates'][index]['keypoints_xy'] = result['native_points']
            preserve_prediction(capture['raw'], prediction)
            C.require(np.array_equal(np.asarray(result['native_points'])[8],
                np.asarray(capture['native_N3_points'])[8], equal_nan=True), 'fresh center changed')
            if stage_callback:
                stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction, original_base_points=capture['original_base_points'],
                native_N3_points=capture['native_N3_points'], observations=capture['observations'],
                observation=capture['observation'], fresh_capture_input_only=True,
                saved_accuracy_coordinates_or_pose_as_input=False)
        self.counts['fresh_predict_calls_complete'] += 1
        self.counts[method + ':complete'] += 1
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.base.close()
