"""Three fixed heads, two fixed point-only observation supplies, no pose prior.

The detector, N3 estimator, corrected last checkpoints, source calibration
procedure, decoder, v4 corner selector and v4 numerical solver are unchanged.
Only the predeclared observation supply and the training-time head mode differ.
Neither truth nor saved evaluation predictions are deployment inputs.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import sys

import numpy as np

from . import common as C
from ..pallet_boundary_corner_refiner_20261010_v2 import observations as decoder
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import (
    Pipeline as CaptureHelpers, assemble, observation_points, preserve_prediction,
)
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
from ..pallet_cornerwise_independent_20261010_v4.selection import select_corners


CHECKPOINT_SHAS = {
    'GEOMETRY_ONLY': 'd188dcc68bd8795c88232d5bf1b85259d684695723b96809017abd47d6ac009b',
    'IMAGE_NO_ROLE': '9c52a2e2036ee8f65ba2e191bdcbebe93ac3ecde60a6aa31e71a4ffbadc0f835',
    'IMAGE_ROLE': '882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d',
}
POLICY = dict(
    schema='fixed_three_head_two_point_supplies_policy_v7',
    heads=list(C.ARMS),
    checkpoint_selection='corrected step3000 last checkpoint of each original head; no outcome selection',
    head_modes=dict(GEOMETRY_ONLY='zero image/neck channels0:19; retain geometry and role25:28',
                    IMAGE_NO_ROLE='zero role25:28; retain image/neck and geometry',
                    IMAGE_ROLE='unchanged full28 channels'),
    source_calibration='same frozen CAL128 procedure per head; exact existing ROLE calibration reused',
    shared_capture='one fixed detector, N3 correction, N3 initial pose, Base query/feature pose per frame',
    boundary_only='v2 sparse admitted actual boundary intersections; missing coordinates remain NaN',
    boundary_only_native_distance_or_LOO_gate=False,
    boundary_only_native_coordinates_used_for_numeric_fit=False,
    hybrid='unchanged v4 cornerwise native-N3 heldout validation and actual-boundary replacement',
    final_solver='unchanged prior-free v4 finite four-ID consensus across both registry dimension hypotheses',
    initial_rotation_translation_projection_dimension_prior=False,
    conditional_mask='unchanged initial N3 H; excluded from generation/scoring/refit, never a frame veto',
    reprojection='replace H only after a new final pose; never reuse projections as observations',
    missing_boundary_filled_for_boundary_only=False,
    minimum_actual_correspondences=4,
    hypothesis_cache='exact coordinate/K/xyz/image-size numeric bank shared across heads, masks and heldouts',
    partial_line_pose_factors=False,
    new_training_updates=0,
    new_RGB=0,
    real_GT_for_decisions=False,
    selection_is_fully_statistically_independent=False,
    physical_corner_ownership_independently_certified=False,
)


def requested_arms(values):
    arms = tuple(values)
    C.require(len(arms) == len(set(arms)) and set(arms) <= set(C.ARMS),
              'requested heads must be distinct original training arms')
    return arms


def calibration_paths(args):
    root = Path(getattr(args, 'calibration_root', C.CAL_ROOT))
    return {arm: root / arm / 'CALIBRATION.json' for arm in C.ARMS}


def checkpoint_preflight(args):
    """Read metadata and hash immutable weights before importing Torch/models."""
    C.require(args.source_root and args.baseline_root,
              'explicit source-root/baseline-root required')
    completion_path = Path(args.fits) / 'TRAINING_COMPLETION.json'
    completion = C.read(completion_path)
    C.require(completion['complete'] and completion['formal_updates'] == completion['total_updates'] == 9000
              and completion['seed'] == 1 and completion['batch'] == 16
              and completion['throwaway_updates'] == 0 and completion['same_initial_tensor_sha']
              and completion['same_batch_order'] and completion['same_update_budget']
              and completion['input_hashes_unchanged'], 'corrected three-head training completion invalid')
    records = {row['arm']: row for row in completion['checkpoints']}
    C.require(set(records) == set(C.ARMS) and len(records) == len(completion['checkpoints']),
              'completed checkpoint population differs')
    bindings = {}
    for arm in C.ARMS:
        path = Path(args.fits) / (arm + '.pt')
        C.bound(path, records[arm]['checkpoint'], arm + ' completed checkpoint')
        C.require(C.sha(path) == CHECKPOINT_SHAS[arm], arm + ' corrected last checkpoint differs')
        bindings[arm] = C.binding(path)
    return completion, dict(training_completion=C.binding(completion_path), checkpoints=bindings)


class Pipeline(CaptureHelpers):
    """Reusable deployment with one active lifecycle and explicit head/supply.

    ``predict(..., arm='IMAGE_NO_ROLE', supply='BOUNDARY_ONLY')`` performs one
    head forward. ``capture(..., need_arms=C.ARMS)`` shares one feature tensor
    among the three accuracy paths. Calibration/checkpoint dependencies are
    source-only; the cohort and reference pose are not constructor inputs.
    """
    _active_lifecycle = False

    def __init__(self, args=None, calibrations=None, head_arms=C.ARMS):
        if args is None:
            args = C.parser(__doc__).parse_args([])
        self.args = args
        self.head_arms = requested_arms(head_arms)
        self.completion, dependency = checkpoint_preflight(args)
        paths = calibration_paths(args)
        self.calibrations = {arm: decoder.load_calibration(paths[arm]) for arm in self.head_arms}
        if calibrations is not None:
            C.require(set(calibrations) == set(self.head_arms), 'supplied calibration head population differs')
            for arm in self.head_arms:
                C.require(C.finite(calibrations[arm]) == C.finite(self.calibrations[arm]),
                          'supplied calibration differs from frozen source-only ' + arm)
        # Common validates source-calibration execution/reuse and arm/weight
        # bindings without changing the exact reused ROLE calibration bytes.
        C.validate_calibrations(args, self.head_arms)
        base_path = Path(args.base_weights) if args.base_weights else Path(args.source_root) / C.V.BASE_WEIGHT_REL
        n3_path = Path(args.N3_weights) if args.N3_weights else Path(args.source_root) / C.V.N3_WEIGHT_REL
        C.require(C.sha(base_path) == C.V.BASE_WEIGHT_SHA and C.sha(n3_path) == C.V.N3_WEIGHT_SHA,
                  'fixed Base/N3 checkpoint identities differ')
        C.require(not Pipeline._active_lifecycle, 'close active Pipeline before constructing another')
        import scripts.research as research
        self._before_namespace = list(research.__path__)
        self.context = C.V.legacy_context(args)
        self._context_active = False
        self.closed = False
        self.models = None
        self.heads = {}
        self.counts = Counter()
        Pipeline._active_lifecycle = True
        try:
            self.old, self.learned, self.runtime = self.context.__enter__()
            self._context_active = True
            baseline_research = Path(args.baseline_root).resolve() / 'scripts/research'
            C.require(baseline_research.is_dir(), 'validated baseline research package missing')
            if str(baseline_research) not in research.__path__:
                research.__path__ = list(research.__path__) + [str(baseline_research)]
            import torch
            import cv2
            torch.set_num_threads(4)
            cv2.setNumThreads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            self.torch = torch
            self.decoder = decoder
            from scripts.research.pallet_n3_completion_v3 import square_yolo as Y
            original_selection = Y.selection_contract
            original_base = self.runtime.BASE_CHECKPOINT

            def located_selection(*a, **kw):
                selected = copy.deepcopy(original_selection(*a, **kw))
                spec = selected['methods']['N3_DIM_SYM_seed1']
                C.require(spec['checkpoint_sha256'] == C.V.N3_WEIGHT_SHA,
                          'original fixed N3 selection differs')
                spec['checkpoint_path'] = str(n3_path)
                return selected

            self.runtime.BASE_CHECKPOINT = base_path
            Y.selection_contract = located_selection
            try:
                self.models = self.runtime.Pipeline()
            finally:
                self.runtime.BASE_CHECKPOINT = original_base
                Y.selection_contract = original_selection
            self.heads = {arm: self.learned.load_head(arm) for arm in self.head_arms}
            self.head = self.heads.get('IMAGE_ROLE')  # compatible accounting alias; not an extra forward
            self.calibration = self.calibrations.get('IMAGE_ROLE')
            self.bindings = self.models.bindings + [dependency['training_completion'], C.binding(__file__),
                C.binding(Path(decoder.__file__))] + [dependency['checkpoints'][arm] for arm in self.head_arms]
            self.bindings += [C.binding(paths[arm]) for arm in self.head_arms]
        except BaseException:
            self._cleanup(*sys.exc_info())
            raise

    def _cleanup(self, *exception):
        if self.closed:
            return
        self.closed = True
        import scripts.research as research
        try:
            if self.models is not None:
                self.models.close()
        finally:
            try:
                if self._context_active:
                    self._context_active = False
                    self.context.__exit__(*(exception or (None, None, None)))
            finally:
                research.__path__ = self._before_namespace
                Pipeline._active_lifecycle = False

    def close(self):
        self._cleanup()

    def capture(self, image, K, xyz, metadata=None, need_arms=C.ARMS,
                need_n3=True, need_base_pose=False, stage_callback=None):
        """One shared frozen detector/query/features capture, one call per requested head."""
        from scripts.research.pallet_observation_refiner_20261009_v1.inference import hidden_mask
        arms = requested_arms(need_arms)
        C.require(set(arms) <= set(self.heads), 'requested head was not loaded for this lifecycle')
        meta = self.registry_metadata(image, K, xyz, metadata)
        mark = stage_callback or (lambda _: None)
        with self.old.no_truth_reads(), self.torch.no_grad():
            captured = self.models.extractor.predict(image)
            self.counts['detector_calls'] += 1
            mark('detector')
            raw = dict(candidates=self.models.inf.serial(captured['candidates']), selected_index=captured['selected_index'])
            index = raw['selected_index']
            original = np.full((9, 2), np.nan) if index is None else np.asarray(raw['candidates'][index]['keypoints_xy'], float)
            if need_n3:
                prediction, diagnostic, intermediate = self.models.correct('N3_SUBPIX', image, captured, meta)
                self.counts['N3_route_calls'] += 1
                native = np.full((9, 2), np.nan) if prediction['selected_index'] is None else np.asarray(prediction['candidates'][prediction['selected_index']]['keypoints_xy'], float)
            else:
                prediction, diagnostic, intermediate, native = raw, None, None, original.copy()
            mark('N3_correction')
            initial = self.models.pose.infer(native, np.asarray(K), np.asarray(xyz), source=False)
            self.counts['initial_pose_calls'] += 1
            base_initial = None
            if need_base_pose:
                base_initial = self.models.pose.infer(original, np.asarray(K), np.asarray(xyz), source=False)
                self.counts['base_control_pose_calls'] += 1
            hidden, hd = hidden_mask(initial)
            mark('initial_pose')
            decoded = {}
            if arms and index is not None:
                features, query = self.learned.M.inputs(image, captured, original, np.asarray(K), np.asarray(xyz))
                self.counts['feature_initial_pose_calls'] += 1
                self.counts['shared_feature_tensors'] += 1
                query['raw_hw'] = list(image.shape[:2])
                for arm in arms:
                    logits = self.heads[arm](features[None], arm)[0]
                    self.counts[arm + '_head_calls'] += 1
                    self.counts['head_calls'] += 1
                    decoded[arm] = self.decoder.decode(query, logits, self.calibrations[arm])
            else:
                for arm in arms:
                    decoded[arm] = dict(corners=[], lines=[], queries=[], selected_queries=0,
                        raw_logits_sha256=None, all_no_match=True, no_detection=index is None,
                        head_not_requested=False)
            mark('boundary_observation')
        alias = decoded.get('IMAGE_ROLE', next(iter(decoded.values()), dict(
            corners=[], lines=[], queries=[], selected_queries=0, raw_logits_sha256=None,
            all_no_match=True, no_detection=index is None, head_not_requested=True)))
        return dict(raw=raw, prediction=prediction, original_base_points=original,
            native_N3_points=native, initial_pose=initial, initial_base_pose=base_initial,
            hidden=hidden, mask_diagnostic=hd, observations=decoded, observation=alias,
            requested_head_arms=list(arms), N3_diagnostic=diagnostic,
            N3_intermediate_points=intermediate, metadata=meta)

    def outcomes(self, capture, methods=C.METHODS):
        """Six fixed head/supply routes plus native masked/unmasked controls."""
        native = np.asarray(capture['native_N3_points'], float)
        initial, hidden, meta = capture['initial_pose'], capture['hidden'], capture['metadata']
        K, xyz = np.asarray(meta['K']), np.asarray(meta['xyz'])
        size = (meta['raw_hw'][1], meta['raw_hw'][0])
        C.require(len(methods) == len(set(methods)) and set(methods) <= set(C.METHODS),
                  'unknown or duplicate three-head route')
        banks, supplies, results = {}, {}, []

        def bank(points):
            values = np.asarray(points, dtype=np.float64)
            key = values.astype('<f8', copy=False).tobytes()
            if key not in banks:
                banks[key] = PoseBank(values, K, xyz, image_size=size)
            return banks[key]

        with self.old.no_truth_reads():
            for method in methods:
                spec = C.METHOD_SPECS.get(method)
                arm, supply = (spec['arm'], spec['supply']) if spec is not None else (None, 'NATIVE_N3')
                H = [] if method == 'N3_INDEPENDENT_ROBUST_NO_MASK' else hidden
                if arm is not None:
                    C.require(arm in capture['observations'], 'capture lacks requested head ' + arm)
                    observation = capture['observations'][arm]
                    key = (arm, supply)
                    if key not in supplies:
                        if supply == 'BOUNDARY_ONLY':
                            sparse, _, contract = observation_points(native, observation, hidden)
                            contract = copy.deepcopy(contract)
                            contract.update(boundary_only_native_distance_used_for_admission=False,
                                boundary_only_LOO_used_for_admission=False,
                                native_N3_used_for_numeric_final_fit=False,
                                heldout_pose_calls=0)
                            supplies[key] = (sparse, contract)
                        elif supply == 'CORNERWISE_HYBRID':
                            selected, contract = select_corners(native, observation, hidden, K, xyz, size, bank=bank(native))
                            self.counts['LOO_pose_paths'] += contract['cornerwise_selection']['LOO_solve_calls']
                            self.counts[arm + '_LOO_pose_paths'] += contract['cornerwise_selection']['LOO_solve_calls']
                            supplies[key] = (selected, contract)
                        else:
                            C.require(False, 'unknown observation supply')
                    points, contract = supplies[key]
                else:
                    observation = dict(corners=[], lines=[], selected_queries=0, raw_logits_sha256=None)
                    points = native
                    contract = observation_points(native, observation, H)[2]
                solved = bank(points).solve(hidden=H, robust=True)
                C.require(solved['prior_used'] is False and solved['known_dimension_constraint_used'] is False,
                          'initial/external dimension prior entered final pose solve')
                self.counts['final_pose_paths'] += 1
                if arm is not None:
                    self.counts[arm + '_' + supply + '_final_pose_paths'] += 1
                result = assemble(native, points, initial, H, solved,
                    'VALIDATED_ROLE_ONLY' if supply == 'BOUNDARY_ONLY' else
                    'N3_VALIDATED_ROLE' if supply == 'CORNERWISE_HYBRID' else method, contract)
                if supply == 'CORNERWISE_HYBRID':
                    result['cornerwise_selection'] = copy.deepcopy(contract['cornerwise_selection'])
                selected_ids = (contract['validated_boundary_corner_ids'] if supply == 'BOUNDARY_ONLY' else
                                contract['hybrid_boundary_corner_ids'] if supply == 'CORNERWISE_HYBRID' else [])
                result.update(head_arm=arm, observation_supply=supply,
                    head_mode=arm, head_checkpoint_sha256=CHECKPOINT_SHAS.get(arm),
                    calibration_binding=C.binding(calibration_paths(self.args)[arm]) if arm else None,
                    selection_validation_fit_excludes_candidate_corner=supply == 'CORNERWISE_HYBRID',
                    selection_validation_is_fully_statistically_independent=False,
                    selection_validation_has_shared_initial_prior=False,
                    final_numeric_pose_has_initial_prior=False,
                    initial_pose_used_only_for_H_and_fallback=True,
                    conditional_H_from_initial_N3=True,
                    Base_proposal_feature_dependence_retained=True,
                    mask_diagnostic=capture['mask_diagnostic'], predicted_initial_N3_hidden=list(hidden),
                    observation_raw_logits_sha256=observation.get('raw_logits_sha256'),
                    selected_corner_ids=list(selected_ids), selected_queries=observation.get('selected_queries', 0),
                    source_lines=[int(line['edge']) for line in observation['lines']],
                    model_query_anchor='unchanged original Base predictions',
                    initial_pose_source='fresh fixed N3_SUBPIX; mask/fallback only for numeric solves',
                    partial_lines_not_pose_inputs=True,
                    all_heads_are_separate_ablations=True, ensemble_observations=False)
                results.append(self.packet(meta, capture, method, result))
        return results, dict(coordinate_banks=[dict(prior_backend=False, input_hash=b.digest, **dict(b.ledger))
            for b in banks.values()], coordinate_bank_count=len(banks),
            same_native_bank_shared_across_heads_masks_and_heldouts=True,
            four_ID_subset_bound_per_dimension=70,
            refit_and_LM_calls_separate_from_four_ID_generations=True)

    def predict(self, image, K, xyz, metadata=None, method=None, *, arm=None,
                supply=None, stage_callback=None):
        if arm is not None or supply is not None:
            C.require(method is None and arm in C.ARMS and supply in ('BOUNDARY_ONLY', 'CORNERWISE_HYBRID'),
                      'supply one valid arm/supply pair or a method, not both')
            matches = [m for m, spec in C.METHOD_SPECS.items()
                       if spec['arm'] == arm and spec['supply'] == supply]
            C.require(len(matches) == 1, 'head/supply route is not uniquely registered')
            method = matches[0]
        method = C.PRIMARY if method is None else method
        C.require(method in C.METHODS + ('BASE', 'N3_SUBPIX'), 'unknown deployment route')
        spec = C.METHOD_SPECS.get(method)
        arms = (spec['arm'],) if spec is not None else ()
        base = method == 'BASE'
        with C.inference_canary():
            captured = self.capture(image, K, xyz, metadata, need_arms=arms,
                need_n3=not base, need_base_pose=False, stage_callback=stage_callback)
            if base:
                captured['initial_base_pose'] = captured['initial_pose']
            result = self.fixed_result(captured, method) if method in ('BASE', 'N3_SUBPIX') else self.outcomes(captured, (method,))[0][0]
            prediction = copy.deepcopy(captured['prediction'])
            index = prediction['selected_index']
            if index is not None:
                prediction['candidates'][index]['keypoints_xy'] = result['native_points']
            preserve_prediction(captured['raw'], prediction)
            if stage_callback:
                stage_callback('final_pose_reprojection_and_metadata')
            result.update(prediction=prediction, original_base_points=captured['original_base_points'],
                native_N3_points=captured['native_N3_points'], observations=captured['observations'],
                observation=captured['observation'])
            return result
