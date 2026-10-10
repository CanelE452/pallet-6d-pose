"""Separate C2: unused partial lines with unchanged point-only LOO selection."""
from __future__ import annotations
import copy
import numpy as np
from . import common as C
from .selection import select_corners
from .solver import PointLineBank
from ..pallet_cornerwise_independent_20261010_v4 import pose
from ..pallet_boundary_corner_refiner_20261010_v2.deployment import Pipeline as Parent
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import observation_points, assemble, preserve_prediction


class Pipeline(Parent):
    def outcomes(self, capture, methods=C.METHODS):
        native, initial, hidden = capture['native_N3_points'], capture['initial_pose'], capture['hidden']
        meta = capture['metadata']
        K, xyz = np.asarray(meta['K']), np.asarray(meta['xyz'])
        size = (meta['raw_hw'][1], meta['raw_hw'][0])
        _, _, native_contract = observation_points(native, capture['observation'], hidden)
        banks, results = {}, []

        def bank(points):
            key = np.asarray(points, dtype='<f8').tobytes()
            if key not in banks:
                banks[key] = pose.PoseBank(points, K, xyz, image_size=size)
            return banks[key]

        selected = None
        with self.old.no_truth_reads():
            for method in methods:
                C.require(method in C.METHODS, 'unknown v5 route')
                H = [] if method == 'N3_INDEPENDENT_ROBUST_NO_MASK' else hidden
                boundary_route = method in (C.PRIMARY, 'N3_INDEPENDENT_CORNERWISE_ROLE')
                points, contract = native, native_contract
                if boundary_route:
                    if selected is None:
                        selected = select_corners(native, capture['observation'], hidden, K, xyz, size, bank=bank(native))
                        count = selected[1]['cornerwise_selection']['LOO_solve_calls']
                        self.counts['LOO_pose_paths'] += count
                        self.counts['independent_LOO_pose_paths'] += count
                    points, contract = selected
                if method == C.PRIMARY:
                    solved = PointLineBank(points, K, xyz, image_size=size, bank=bank(points)).solve(
                        capture['observation'], adopted_boundary_corner_ids=contract['hybrid_boundary_corner_ids'], hidden=H)
                    self.counts['partial_line_pose_paths'] += 1
                    for counter, value in solved.get('point_line_operation_counts', {}).items():
                        self.counts['C2_' + counter] += value
                else:
                    solved = bank(points).solve(hidden=H, robust=True)
                C.require(solved['prior_used'] is False, 'initial numerical prior entered C2/control')
                self.counts['final_pose_paths'] += 1
                self.counts['independent_final_pose_paths'] += 1
                if method == C.PRIMARY:
                    contract = copy.deepcopy(contract)
                    contract.update(partial_line_path_added=True,
                        lines_are_not_extra_pose_factors=not bool(solved.get('line_edges')),
                        C2_unused_line_edges=solved.get('line_edges', []),
                        C2_consumed_edges=solved.get('consumed_edges', []),
                        C2_final_line_inliers=solved.get('final_line_inliers', []),
                        C2_same_edge_point_line_double_count=False)
                result = assemble(native, points, initial, H, solved,
                                  'N3_VALIDATED_ROLE' if boundary_route else method, contract)
                if boundary_route:
                    result['cornerwise_selection'] = copy.deepcopy(contract['cornerwise_selection'])
                result.update(
                    selection_validation_fit_excludes_candidate_corner=boundary_route,
                    selection_validation_is_fully_statistically_independent=False,
                    selection_validation_has_shared_initial_prior=False,
                    final_numeric_pose_has_initial_prior=False,
                    initial_pose_used_only_for_H_and_fallback=True,
                    conditional_H_from_initial_N3=True,
                    Base_proposal_feature_dependence_retained=True,
                    mask_diagnostic=capture['mask_diagnostic'],
                    predicted_initial_N3_hidden=list(hidden),
                    observation_raw_logits_sha256=capture['observation'].get('raw_logits_sha256'),
                    selected_corner_ids=contract['hybrid_boundary_corner_ids'] if boundary_route else [],
                    selected_queries=capture['observation'].get('selected_queries', 0),
                    source_lines=[int(line['edge']) for line in capture['observation']['lines']],
                    model_query_anchor='unchanged original Base predictions',
                    initial_pose_source='fresh fixed N3_SUBPIX; mask/fallback only for numeric solves',
                    partial_lines_not_pose_inputs=method != C.PRIMARY,
                    C2_same_IMAGE_ROLE_observations=method == C.PRIMARY,
                    C2_selection_LOO_uses_lines=False,
                    C2_initial_numeric_pose_prior=False,
                    C2_line_consensus_scoring_used=bool(solved.get('line_edges')) if method == C.PRIMARY else False,
                    local_point_line_refinement=bool(solved.get('available') and solved.get('selected_pose_refined_with_lines')) if method == C.PRIMARY else False)
                results.append(self.packet(meta, capture, method, result))
        return results, dict(coordinate_banks=[dict(input_hash=b.digest, **dict(b.ledger)) for b in banks.values()])

    def predict(self, image, K, xyz, metadata=None, method=C.PRIMARY, stage_callback=None):
        C.require(method in C.METHODS + ('BASE', 'N3_SUBPIX'), 'unknown route')
        base = method == 'BASE'
        role = method in (C.PRIMARY, 'N3_INDEPENDENT_CORNERWISE_ROLE')
        with C.inference_canary():
            capture = self.capture(image, K, xyz, metadata, need_role=role,
                                   need_n3=not base, need_base_pose=False, stage_callback=stage_callback)
            if base:
                capture['initial_base_pose'] = capture['initial_pose']
            result = self.fixed_result(capture, method) if method in ('BASE', 'N3_SUBPIX') else self.outcomes(capture, (method,))[0][0]
            prediction = copy.deepcopy(capture['prediction'])
            index = prediction['selected_index']
            if index is not None:
                prediction['candidates'][index]['keypoints_xy'] = result['native_points']
            preserve_prediction(capture['raw'], prediction)
            if stage_callback:
                stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction, original_base_points=capture['original_base_points'],
                          native_N3_points=capture['native_N3_points'], observation=capture['observation'])
            return result
