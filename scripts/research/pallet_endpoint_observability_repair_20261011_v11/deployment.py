"""Fresh complete deployment for the two V11 views of one local proposal.

Each prediction captures detector, fixed N3 and the frozen IMAGE_ROLE head,
then computes its point-only start, native endpoint LOO and filtered local
fit within the call. It never receives sealed accuracy coordinates or audit
receipts. Requesting a different view starts a new complete prediction; both
views within one call share exactly one local solve.
"""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np

from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline as RolePipeline
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import preserve_prediction
from . import pipeline as P

ARMS = ('BASE', 'N3_SUBPIX', P.POINT_COMPARATOR, *P.METHODS)
STAGES = ('detector', 'N3_correction', 'initial_pose', 'boundary_observation',
    'endpoint_validation_and_filtered_local', 'final_pose_reprojection_and_metadata')


class Pipeline:
    """One fixed model lifecycle; no cache survives between predict calls."""

    def __init__(self, args):
        self.base = RolePipeline(args, head_arms=('IMAGE_ROLE',))
        self.counts = Counter()
        self.local_operation_counts = Counter()
        self.endpoint_operation_counts = Counter()
        self.last_capture = self.last_point_result = self.last_point_bank_ledger = None
        self.last_repair_ledger = self.last_views = None
        self.last_execution_events = []
        self.closed = False

    @property
    def torch(self): return self.base.torch

    @property
    def runtime(self): return self.base.runtime

    @property
    def old(self): return self.base.old

    @property
    def heads(self): return self.base.heads

    @property
    def models(self): return self.base.models

    @property
    def bindings(self): return self.base.bindings

    def predict(self, image, K, xyz, metadata=None, method=P.PRIMARY, stage_callback=None):
        P.require(method in ARMS, 'unknown fresh V11 deployment route')
        self.last_capture = self.last_point_result = self.last_point_bank_ledger = None
        self.last_repair_ledger = self.last_views = None
        self.last_execution_events = []
        self.counts['fresh_predict_calls_started'] += 1
        self.counts[method + ':started'] += 1
        learned = method in (P.POINT_COMPARATOR, *P.METHODS)
        with self.old.no_truth_reads(), self.torch.no_grad():
            capture = self.base.capture(image, K, xyz, metadata,
                need_arms=('IMAGE_ROLE',) if learned else (), need_n3=method != 'BASE',
                need_base_pose=False, stage_callback=stage_callback)
            self.last_capture = capture
            if method == 'BASE': capture['initial_base_pose'] = capture['initial_pose']
            if not learned:
                result = self.base.fixed_result(capture, method)
            else:
                self.counts['fresh_sparse_point_paths_started'] += 1
                point_rows, point_bank_ledger = self.base.outcomes(capture, (P.PARENT_PRIMARY,))
                P.require(len(point_rows) == 1, 'fresh sparse point control population differs')
                point = point_rows[0]
                self.last_point_result, self.last_point_bank_ledger = point, point_bank_ledger
                self.counts['fresh_sparse_point_paths_complete'] += 1
                if method == P.POINT_COMPARATOR:
                    result = copy.deepcopy(point)
                    result.update(method=P.POINT_COMPARATOR,
                        fresh_point_control_equivalent_parent_method=P.PARENT_PRIMARY,
                        existing_control_replay=False, fresh_complete_capture=True,
                        cached_accuracy_row_used=False, runtime_only_fresh_metadata=True)
                else:
                    sparse = np.asarray(point['input_points'], float)
                    native = np.asarray(capture['native_N3_points'], float)
                    observation = capture['observations']['IMAGE_ROLE']
                    contract, H = point['observation_contract'], list(capture['hidden'])
                    self.counts['fresh_repair_paths_started'] += 1

                    def observed(event):
                        self.last_execution_events.append(copy.deepcopy(event))
                        phase = event['phase']
                        self.counts[phase + ':events'] += 1
                        if phase == 'endpoint_request_started':
                            self.counts['fresh_native_endpoint_requests_started'] += 1
                        elif phase == 'endpoint_request_returned':
                            self.counts['fresh_native_endpoint_requests_complete'] += 1
                        elif phase == 'filtered_local_request_started':
                            self.counts['fresh_filtered_local_paths_started'] += 1
                        elif phase == 'filtered_local_request_returned':
                            self.counts['fresh_filtered_local_paths_complete'] += 1

                    solvers, ledger = P.repair_observations(sparse, native, np.asarray(K), np.asarray(xyz),
                        observation, H, contract['validated_boundary_corner_ids'],
                        initial_pose=copy.deepcopy(capture['initial_pose']),
                        point_result=copy.deepcopy(point['solver']),
                        image_size=(image.shape[1], image.shape[0]), observer=observed)
                    self.last_repair_ledger = ledger
                    self.local_operation_counts.update(ledger['local_operation_counts'])
                    self.endpoint_operation_counts.update(ledger['endpoint_native_bank_delta'])
                    P.require(ledger['endpoint_decision_source'] == 'FRESH_NATIVE_ENDPOINT_LOO' and
                        ledger['actual_filtered_local_solve_calls'] == 1 and
                        ledger['endpoint_validation_requests'] == ledger['endpoint_decision_records'],
                        'fresh endpoint validation or single local solve was skipped')
                    views = {}
                    for output_method in P.METHODS:
                        local = P.assemble_view(point, native, sparse, capture['initial_pose'], H,
                            contract, solvers[output_method], output_method, ledger, fresh=True)
                        local.update(fresh_point_start_recomputed_inside_call=True,
                            fresh_point_method=P.PARENT_PRIMARY,
                            fresh_endpoint_LOO_recomputed_inside_call=True,
                            parent_completion_receipt_used_as_deployment_input=False,
                            sealed_parent_packet_adapter_called=False,
                            cached_V10_endpoint_decisions_used=False,
                            runtime_only_fresh_metadata=True,
                            kernel_start_source_labels_are_identifiers_not_receipts=True,
                            kernel_start_source_provenance=dict(SEALED_INITIAL_N3='fresh initial N3 from this call',
                                SEALED_AVAILABLE_POINT_ONLY='available fresh sparse point solve from this call'),
                            ensemble_observations=False)
                        views[output_method] = self.base.packet(capture['metadata'], capture, output_method, local)
                    self.last_views = views
                    result = copy.deepcopy(views[method])
                    result['fresh_point_solver'] = copy.deepcopy(point['solver'])
                    result['fresh_point_coordinate_bank_ledger'] = copy.deepcopy(point_bank_ledger)
                    result['fresh_repair_ledger'] = copy.deepcopy(ledger)
                    self.counts['fresh_repair_paths_complete'] += 1
                    self.counts['reported_views_from_single_filtered_local'] += 2
                    if stage_callback: stage_callback('endpoint_validation_and_filtered_local')
            prediction = copy.deepcopy(capture['prediction'])
            index = prediction['selected_index']
            if index is not None:
                prediction['candidates'][index]['keypoints_xy'] = result['native_points']
            preserve_prediction(capture['raw'], prediction)
            P.require(np.array_equal(np.asarray(result['native_points'])[8],
                np.asarray(capture['native_N3_points'])[8], equal_nan=True), 'fresh detector center changed')
            if stage_callback: stage_callback('final_pose_reprojection_and_metadata')
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
