"""Deployable validated boundary/N3 corner refinement, and a sealed fresh run.

Pipeline.predict(image, K, xyz, metadata=None) uses no cohort, stored keypoint
rows, reference pose, human visibility or cached truth. xyz is physical (W,H,D)
in meters and K acts on native full-image pixel coordinates.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path
import time

from . import common as C


def preserve_prediction(before, after):
    """Verify selection, all confidence/class metadata and nonselected points."""
    import numpy as np
    C.require(before['selected_index'] == after['selected_index'] and
              len(before['candidates']) == len(after['candidates']), 'detector selection/candidates changed')
    for i, (a, b) in enumerate(zip(before['candidates'], after['candidates'])):
        C.require(set(a) == set(b), 'candidate fields changed')
        C.require(C.finite({k:v for k,v in a.items() if k != 'keypoints_xy'}) ==
                  C.finite({k:v for k,v in b.items() if k != 'keypoints_xy'}), 'confidence/class metadata changed')
        p, q = np.asarray(a['keypoints_xy'], float), np.asarray(b['keypoints_xy'], float)
        if i != before['selected_index']:
            C.require(np.array_equal(p, q, equal_nan=True), 'nonselected candidate coordinates changed')
        else:
            C.require(np.array_equal(p[8], q[8], equal_nan=True), 'selected detector center changed')
    return True


def observation_points(native_n3, observation, hidden):
    """Keep the provenance of native RGB and validated boundary points explicit."""
    import numpy as np
    n3 = np.asarray(native_n3, float)
    C.require(n3.shape == (9, 2), 'native N3 point shape differs')
    raw_selected = {int(c['id']): c for c in observation['corners']}
    C.require(len(raw_selected) == len(observation['corners']) and set(raw_selected) <= set(range(8)),
              'validated corners have duplicate/invalid IDs')
    line_checks = []
    invalid_edges = set()
    for line in observation['lines']:
        edge = int(line['edge'])
        support = np.asarray(line['support_points'], float)
        normal = np.asarray(line['normal'], float)
        radii = np.asarray(line['query_radii_px'], float)
        C.require(support.ndim == 2 and support.shape[1] == 2 and normal.shape == (2,) and
                  radii.shape == (len(support),), 'final line support/radius shape differs')
        residual = np.abs(support @ normal - float(line['offset']))
        valid = bool(np.isfinite(residual).all() and np.isfinite(radii).all() and
                     (radii >= 0).all() and np.all(residual <= radii + 1e-9))
        if not valid:
            invalid_edges.add(edge)
        line_checks.append(dict(edge=edge, support_queries=line['queries'], absolute_residuals_px=residual.tolist(),
            query_radii_px=radii.tolist(), accepted=valid, tolerance_px=1e-9,
            reason='accepted' if valid else 'FINAL_LINE_CONSENSUS_INCONSISTENT'))
    selected, admission = {}, []
    for k, corner in sorted(raw_selected.items()):
        C.require(len(corner['edges']) == 2 and len(set(corner['edges'])) == 2 and
                  set(corner['edges']) <= {int(l['edge']) for l in observation['lines']},
                  'corner has no two distinct supporting decoded lines')
        radius = float(corner['radius_px'])
        reason = ('FINAL_LINE_CONSENSUS_INCONSISTENT' if set(corner['edges']) & invalid_edges else
                  'UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS' if not math.isfinite(radius) or
                  radius < 0 or radius > C.CORNER_OBSERVATION_UNCERTAINTY_CAP_PX else 'accepted')
        admission.append(dict(id=k, edges=corner['edges'], radius_px=radius,
                              accepted=reason == 'accepted', reason=reason))
        if reason == 'accepted':
            selected[k] = np.asarray(corner['xy'], float)
    C.require(all(p.shape == (2,) and np.isfinite(p).all() for p in selected.values()),
              'validated boundary corner is not finite')
    sparse = np.full((9, 2), np.nan)
    sparse[8] = n3[8]
    hybrid = n3.copy()
    replaced, rejected = [], []
    diagnostic = []
    for k, point in sorted(selected.items()):
        sparse[k] = point
        valid_n3 = bool(np.isfinite(n3[k]).all() and not np.all(n3[k] == -1))
        distance = float(np.linalg.norm(point - n3[k])) if valid_n3 else None
        close = bool(valid_n3 and distance <= C.NATIVE_REPLACEMENT_CAP_PX)
        used = bool(k not in hidden and close)
        if used:
            hybrid[k] = point
            replaced.append(k)
        else:
            rejected.append(k)
        diagnostic.append(dict(id=k, validated_xy=point.tolist(), native_N3_xy=n3[k].tolist(),
            distance_to_native_N3_px=distance, selected_for_hybrid=used,
            reason='accepted' if used else 'initial_self_hidden' if k in hidden else
                   'native_N3_unavailable' if not valid_n3 else 'outside_fixed8px_N3_basin'))
    C.require(np.array_equal(sparse[8], n3[8], equal_nan=True) and
              np.array_equal(hybrid[8], n3[8], equal_nan=True), 'center changed')
    return sparse, hybrid, dict(validated_boundary_corner_ids=sorted(selected),
        hybrid_boundary_corner_ids=replaced, hybrid_rejected_boundary_corner_ids=rejected,
        native_N3_RGB_corner_ids=[k for k in range(8) if k not in replaced],
        replacement_cap_px=C.NATIVE_REPLACEMENT_CAP_PX, per_validated_corner=diagnostic,
        computed_decoder_corner_ids=sorted(raw_selected), corner_admission=admission,
        corner_uncertainty_cap_px=C.CORNER_OBSERVATION_UNCERTAINTY_CAP_PX,
        final_line_consensus_checks=line_checks, invalid_final_line_edges=sorted(invalid_edges),
        model_query_anchor='original Base predictions; unchanged training feature distribution',
        N3_is_independent_RGB_observation=True, lines_are_not_extra_pose_factors=True,
        partial_line_path_added=False, missing_ROLE_ONLY_corner_is_NaN=True)


def assemble(native_n3, input_points, initial, hidden, solved, method, contract):
    """Display selected observations and final hidden reprojections; no new fit."""
    import numpy as np
    from scripts.research.pallet_observation_refiner_20261009_v1.inference import hidden_mask
    native = np.asarray(native_n3, float)
    inputs = np.asarray(input_points, float)
    new = bool(solved.get('available'))
    fallback = bool(not new and initial.get('available', False))
    final = solved if new else initial
    out = native.copy()
    if new:
        for k in range(8):
            if k not in hidden and np.isfinite(inputs[k]).all() and not np.all(inputs[k] == -1):
                out[k] = inputs[k]
        if hidden:
            out[list(hidden)] = np.asarray(solved['projected'])[list(hidden)]
        C.require(set(hidden).isdisjoint(solved.get('fit_input_ids', [])), 'hidden initial coordinate entered fit')
    C.require(np.array_equal(out[8], native[8], equal_nan=True), 'selected center changed')
    if not new:
        C.require(np.array_equal(out, native, equal_nan=True), 'fallback is not full native N3')
    after, _ = hidden_mask(final) if new else ([], {})
    boundary = set(contract['validated_boundary_corner_ids'] if method == 'VALIDATED_ROLE_ONLY'
                   else contract['hybrid_boundary_corner_ids'] if method.startswith('N3_VALIDATED_ROLE') else [])
    source = ['FINAL_POSE_REPROJECTION' if new and k in hidden else
              'VALIDATED_BOUNDARY_INTERSECTION' if new and k in boundary else
              'NATIVE_N3_RGB' for k in range(8)] + ['UNCHANGED_DETECTOR_CENTER']
    return dict(solver=solved, actual_pose=final, native_points=out,
        pose_available=bool(final.get('available', False)), new_pose_estimated=new,
        fallback_used=fallback, no_pose=not bool(final.get('available', False)),
        output_status='NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',
        hidden_initial=list(hidden), hidden_after=after,
        hidden_set_changed=bool(new and set(after) != set(hidden)), excluded=list(hidden),
        hidden_reprojected=bool(new and hidden), reprojected_ids=list(hidden) if new else [],
        reprojections_reused_as_observations=False, input_points=inputs, initial_pose=initial,
        observation_contract=copy.deepcopy(contract), output_coordinate_sources=source,
        native_N3_used_as_independent_RGB_observations=method != 'VALIDATED_ROLE_ONLY',
        no_match_filled_as_boundary_observation=False, local_point_line_refinement=False,
        oracle=False)


class Pipeline:
    """One fixed detector capture shared by fixed N3 and the corrected ROLE head.

The source/baseline/checkpoint/calibration dependencies are required. Frozen
evaluation populations and GT caches are not required for this API.
"""
    def __init__(self, args=None, calibration=None):
        if args is None:
            args = C.parser(__doc__).parse_args([])
        self.args = args
        self.context = C.legacy_context(args)
        self.closed = False
        self.models = None
        self.counts = Counter()
        try:
            self.old, self.learned, self.runtime = self.context.__enter__()
            import torch
            import cv2
            torch.set_num_threads(4)
            cv2.setNumThreads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            self.torch = torch
            from . import observations
            self.decoder = observations
            self.calibration = calibration if calibration is not None else observations.load_calibration(args.calibration)
            completion = C.read(Path(args.fits) / 'TRAINING_COMPLETION.json')
            C.require(completion['complete'] and completion['formal_updates'] == completion['total_updates'] == 9000 and
                      completion['seed']==1 and completion['batch']==16 and completion['throwaway_updates']==0 and
                      completion['same_initial_tensor_sha'] and completion['same_batch_order'] and
                      completion['same_update_budget'] and completion['input_hashes_unchanged'], 'corrected ROLE completion absent')
            role = next(r for r in completion['checkpoints'] if r['arm'] == 'IMAGE_ROLE')
            C.bound(Path(args.fits) / 'IMAGE_ROLE.pt', role['checkpoint'], 'fixed last ROLE checkpoint')
            C.require(C.sha(Path(args.fits)/'IMAGE_ROLE.pt')==C.ROLE_WEIGHT_SHA,'fixed corrected last ROLE identity differs')
            base_path = Path(args.base_weights) if args.base_weights else Path(args.source_root)/C.BASE_WEIGHT_REL
            n3_path = Path(args.N3_weights) if args.N3_weights else Path(args.source_root)/C.N3_WEIGHT_REL
            C.require(C.sha(base_path)==C.BASE_WEIGHT_SHA and C.sha(n3_path)==C.N3_WEIGHT_SHA,
                      'fixed Base/N3 checkpoint identities differ')
            # Only checkpoint locations may differ. The original selection contract,
            # state dictionaries and every numerical correction setting stay fixed.
            from scripts.research.pallet_n3_completion_v3 import square_yolo as Y
            selection = Y.selection_contract
            original_base = self.runtime.BASE_CHECKPOINT
            def located_selection(*a, **kw):
                selected = copy.deepcopy(selection(*a, **kw))
                spec = selected['methods']['N3_DIM_SYM_seed1']
                C.require(spec['checkpoint_sha256']==C.N3_WEIGHT_SHA,'original fixed N3 selection differs')
                spec['checkpoint_path'] = str(n3_path)
                return selected
            self.runtime.BASE_CHECKPOINT = base_path
            Y.selection_contract = located_selection
            try:
                self.models = self.runtime.Pipeline()
            finally:
                self.runtime.BASE_CHECKPOINT = original_base
                Y.selection_contract = selection
            self.head = self.learned.load_head('IMAGE_ROLE')
            self.bindings = self.models.bindings + [C.binding(args.calibration),
                C.binding(Path(args.fits) / 'IMAGE_ROLE.pt'), C.binding(__file__),
                C.binding(Path(observations.__file__))]
        except BaseException:
            if self.models is not None:
                self.models.close()
            self.context.__exit__(*__import__('sys').exc_info())
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            if self.models is not None:
                self.models.close()
            self.context.__exit__(None, None, None)

    def registry_metadata(self, image, K, xyz, metadata=None):
        import numpy as np
        meta = dict(metadata or {})
        K = np.asarray(K, float)
        xyz = np.asarray(xyz, float)
        C.require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0,
                  'invalid native-image camera matrix')
        C.require(xyz.shape == (3,) and np.isfinite(xyz).all() and (xyz > 0).all(), 'invalid physical W,H,D')
        C.require(image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8,
                  'native image must be uint8 BGR')
        dims = xyz[[0, 2, 1]]
        if 'object_type' not in meta:
            registry = self.models.env.read(self.models.env.ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')['objects']
            matches = [r['object_type'] for r in registry if
                       np.allclose([r['physical_dimensions_m'][k] for k in ('x', 'y', 'z')], xyz, rtol=0, atol=1e-12)]
            C.require(len(matches) == 1, 'dimensions must identify one registered object or metadata must supply object_type')
            meta['object_type'] = matches[0]
        known, _ = self.models.inf.registry_input(meta['object_type'])
        C.require(np.array_equal(known, dims), 'registered dimensions disagree with supplied physical W,H,D')
        return dict(meta, K=K.tolist(), xyz=xyz.tolist(), raw_hw=list(image.shape[:2]),
                    camera_intrinsics=K.tolist(), dimensions_wdh_m=dims.tolist())

    def capture(self, image, K, xyz, metadata=None, need_role=True, need_n3=True,
                need_base_pose=False, stage_callback=None):
        import numpy as np
        from scripts.research.pallet_observation_refiner_20261009_v1.inference import hidden_mask
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
                native_n3 = np.full((9, 2), np.nan) if prediction['selected_index'] is None else np.asarray(prediction['candidates'][prediction['selected_index']]['keypoints_xy'], float)
            else:
                prediction, diagnostic, intermediate, native_n3 = raw, None, None, original.copy()
            mark('N3_correction')
            initial = self.models.pose.infer(native_n3, np.asarray(K), np.asarray(xyz), source=False)
            self.counts['initial_pose_calls'] += 1
            base_initial = None
            if need_base_pose:
                base_initial = self.models.pose.infer(original, np.asarray(K), np.asarray(xyz), source=False)
                self.counts['base_control_pose_calls'] += 1
            hidden, hd = hidden_mask(initial)
            mark('initial_pose')
            if need_role and index is not None:
                x, query = self.learned.M.inputs(image, captured, original, np.asarray(K), np.asarray(xyz))
                query['raw_hw'] = list(image.shape[:2])
                logits = self.head(x[None], 'IMAGE_ROLE')[0]
                observation = self.decoder.decode(query, logits, self.calibration)
                self.counts['ROLE_head_calls'] += 1
                self.counts['feature_initial_pose_calls'] += 1
            else:
                observation = dict(corners=[], lines=[], queries=[], selected_queries=0,
                                   raw_logits_sha256=None, all_no_match=True,
                                   no_detection=index is None, ROLE_not_requested=not need_role)
            mark('boundary_observation')
        return dict(raw=raw, prediction=prediction, original_base_points=original,
            native_N3_points=native_n3, initial_pose=initial, initial_base_pose=base_initial,
            hidden=hidden, mask_diagnostic=hd, observation=observation,
            N3_diagnostic=diagnostic, N3_intermediate_points=intermediate, metadata=meta)

    def outcomes(self, capture, methods=C.METHODS):
        import numpy as np
        from . import pose
        n3 = capture['native_N3_points']
        initial = capture['initial_pose']
        hidden = capture['hidden']
        meta = capture['metadata']
        sparse, hybrid, contract = observation_points(n3, capture['observation'], hidden)
        coordinates = {'ROLE_ONLY': sparse, 'HYBRID': hybrid, 'N3': n3}
        banks = {}
        results = []
        with self.old.no_truth_reads():
            for method in methods:
                C.require(method in C.METHODS, 'unknown deployed method')
                key = 'ROLE_ONLY' if method == 'VALIDATED_ROLE_ONLY' else 'HYBRID' if method.startswith('N3_VALIDATED_ROLE') else 'N3'
                points = coordinates[key]
                if key not in banks:
                    banks[key] = pose.PoseBank(points, np.asarray(meta['K']), np.asarray(meta['xyz']),
                        image_size=(meta['raw_hw'][1], meta['raw_hw'][0]))
                H = [] if method == 'N3_VALIDATED_ROLE_NO_MASK' else hidden
                solved = pose.refine(points, np.asarray(meta['K']), np.asarray(meta['xyz']), initial, H,
                    (meta['raw_hw'][1], meta['raw_hw'][0]), robust=method != 'N3_BASIN_STANDARD', bank=banks[key])
                self.counts['final_pose_paths'] += 1
                result = assemble(n3, points, initial, H, solved, method, contract)
                result.update(mask_diagnostic=capture['mask_diagnostic'],
                    predicted_initial_N3_hidden=list(hidden),
                    observation_raw_logits_sha256=capture['observation'].get('raw_logits_sha256'),
                    selected_corner_ids=contract['validated_boundary_corner_ids'],
                    selected_queries=capture['observation'].get('selected_queries', 0),
                    source_lines=[int(l['edge']) for l in capture['observation']['lines']],
                    model_query_anchor='original Base predictions', initial_pose_source='fresh fixed N3_SUBPIX',
                    dimensions_semantics='xyz physical W,H,D; N3 registry W,D,H',
                    partial_lines_not_pose_inputs=True)
                results.append(self.packet(meta, capture, method, result))
        return results, {key: dict(bank.ledger) for key, bank in banks.items()}

    @staticmethod
    def packet(meta, capture, method, result):
        selected = capture['raw']['selected_index']
        candidate = None if selected is None else {
            k: v for k, v in capture['raw']['candidates'][selected].items() if k != 'keypoints_xy'}
        return dict(id=meta.get('id', meta.get('frame_id', 'deployment')), session=meta.get('session', 'deployment'),
            method=method, K=meta['K'], xyz=meta['xyz'], raw_hw=meta['raw_hw'], selected_index=selected,
            fixed_metadata=dict(candidate_metadata=candidate, preserved=True), **result)

    def fixed_result(self, capture, arm):
        import numpy as np
        base = arm == 'BASE'
        points = capture['original_base_points'] if base else capture['native_N3_points']
        initial = capture['initial_base_pose'] if base else capture['initial_pose']
        C.require(initial is not None, 'fresh Base control pose was not requested')
        result = dict(actual_pose=initial, native_points=np.asarray(points).copy(), input_points=np.asarray(points).copy(),
            initial_pose=initial, solver=None, new_pose_estimated=False, fallback_used=False,
            pose_available=bool(initial.get('available')), no_pose=not bool(initial.get('available')),
            output_status='FRESH_FIXED_CONTROL', hidden_initial=[], hidden_after=[], hidden_set_changed=False,
            excluded=[], hidden_reprojected=False, reprojected_ids=[], reprojections_reused_as_observations=False,
            oracle=False, input_coordinate_source='original Base RGB' if base else 'fixed N3_SUBPIX RGB')
        return self.packet(capture['metadata'], capture, arm, result)

    def predict(self, image, K, xyz, metadata=None, method=C.PRIMARY, stage_callback=None):
        """Fresh complete deployment path. metadata and evaluation caches are optional."""
        C.require(method in C.METHODS + ('BASE', 'N3_SUBPIX'), 'unknown deployment method')
        base = method == 'BASE'
        need_role = method.startswith('N3_VALIDATED_ROLE') or method == 'VALIDATED_ROLE_ONLY'
        capture = self.capture(image, K, xyz, metadata, need_role=need_role, need_n3=not base,
                               need_base_pose=False, stage_callback=stage_callback)
        if base:
            capture['initial_base_pose'] = capture['initial_pose']
        if method in ('BASE', 'N3_SUBPIX'):
            result = self.fixed_result(capture, method)
        else:
            result = self.outcomes(capture, (method,))[0][0]
        prediction = copy.deepcopy(capture['prediction'])
        index = prediction['selected_index']
        if index is not None:
            prediction['candidates'][index]['keypoints_xy'] = result['native_points']
        preserve_prediction(capture['raw'], prediction)
        if stage_callback:
            stage_callback('final_pose_reprojection_and_metadata')
        return dict(result, prediction=prediction, original_base_points=capture['original_base_points'],
                    native_N3_points=capture['native_N3_points'], observation=capture['observation'])


def freeze(args):
    C.cohort_frames(args)
    prior = C.prior_snapshot(args.prior_bindings)
    C.write_new(args.protocol, dict(schema='validated_boundary_corner_refinement_protocol_v2',
        methods=list(C.METHODS), primary=C.PRIMARY, frames=245, output_rows=1225,
        fixed_control_rows=490, native_replacement_cap_px=8.0,
        corner_observation_uncertainty_cap_px=8.0,
        final_line_consensus_veto='all stored final support residuals <= query radius + 1e-9; one pass; no extra refit',
        GT_tuning=False,
        new_training_updates=0, RGB_regenerated=0, checkpoint_selection='corrected last IMAGE_ROLE step3000',
        initial_pose='fresh fixed N3_SUBPIX old prediction-only pose',
        mask='fixed initial N3 convex-cuboid visibility margin2deg; differs recorded, never frame veto',
        missing_boundary='ROLE_ONLY NaN; hybrid uses separately tagged independent native N3 RGB observations',
        self_reprojection='new valid final R,t projects excluded H; no projection is re-fit',
        same_coordinate_hypotheses_shared=True, fixed_inputs=C.protocol_inputs(args),
        protected_materialized_files=len(prior), runtime='fresh complete deployment path, separate quiet window'))
    print('BOUNDARY_REFINER_FROZEN', C.sha(args.protocol), flush=True)


def infer(args):
    import cv2
    import numpy as np
    C.verify_protocol(args)
    prior = C.prior_snapshot(args.prior_bindings)
    frames = C.cohort_frames(args)
    started = C.output_path(args, 'INFERENCE_STARTED.json')
    for name in ('OBSERVATIONS.jsonl.gz', 'GEOMETRY_SEALED.jsonl.gz', 'FIXED_GEOMETRY_SEALED.jsonl.gz',
                 'GEOMETRY_SEAL.json', 'BASE_N3_PARITY.json'):
        C.output_path(args, name)
    C.write_new(started, dict(protocol=C.binding(args.protocol), configured_frames=245,
        configured_geometric_rows=1225, configured_fixed_rows=490, actual_detector_calls=0))
    geometry, controls, observations, parity, banks = [], [], [], [], []
    start = time.monotonic()
    pose_references = {(r['id'], r['method'].split('_NO_MASK_STANDARD')[0]): r['initial_pose']
                       for r in C.rows(C.OLD / 'POSE_DIAGNOSTICS.jsonl.gz')
                       if r['method'] in ('BASE_NO_MASK_STANDARD', 'N3_SUBPIX_NO_MASK_STANDARD')}
    with Pipeline(args) as pipeline:
        with pipeline.old.no_truth_reads(), pipeline.torch.no_grad(), C.primitive_counter() as cv_calls:
            for frame in frames:
                image_path = Path(args.source_root) / frame['image']
                C.require(C.sha(image_path) == frame['image_sha256'], 'original RGB hash differs')
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                C.require(image is not None and list(image.shape[:2]) == frame['raw_hw'], 'original image shape/decode differs')
                captured = pipeline.capture(image, frame['K'], frame['xyz'], frame, need_base_pose=True)
                index = captured['raw']['selected_index']
                metadata = None if index is None else {k:v for k,v in captured['raw']['candidates'][index].items() if k!='keypoints_xy'}
                C.require(index==frame['selected_index'] and C.finite(metadata)==C.finite(frame['candidate_metadata']),
                          'fresh detector selection/confidence/box metadata differs')
                preserve_prediction(captured['raw'],captured['prediction'])
                base_error = pipeline.runtime._point_parity(captured['original_base_points'], frame['points']['BASE'], 1e-7, ('BASE', frame['id']))
                n3_error = pipeline.runtime._point_parity(captured['native_N3_points'], frame['points']['N3_SUBPIX'], 1e-7, ('N3_SUBPIX', frame['id']))
                base_pose = pipeline.runtime._pose_parity(captured['initial_base_pose'], pose_references[(frame['id'], 'BASE')])
                n3_pose = pipeline.runtime._pose_parity(captured['initial_pose'], pose_references[(frame['id'], 'N3_SUBPIX')])
                parity.append(dict(id=frame['id'], BASE=base_error, N3_SUBPIX=n3_error,
                    BASE_pose=base_pose, N3_SUBPIX_pose=n3_pose,
                    candidate_metadata_preserved=True, center_unchanged=bool(np.array_equal(captured['native_N3_points'][8], captured['original_base_points'][8], equal_nan=True))))
                controls.extend(pipeline.fixed_result(captured, arm) for arm in ('BASE', 'N3_SUBPIX'))
                result, ledger = pipeline.outcomes(captured)
                geometry.extend(result)
                banks.append(dict(id=frame['id'], coordinate_banks=ledger))
                observations.append(dict(id=frame['id'], session=frame['session'], method='VALIDATED_IMAGE_ROLE',
                    original_base_points=captured['original_base_points'], native_N3_points=captured['native_N3_points'],
                    initial_N3_pose=captured['initial_pose'], predicted_N3_hidden=captured['hidden'],
                    GT_input=False, **captured['observation']))
                if len(observations) % 32 == 0:
                    print('BOUNDARY_REFINER_GEOMETRY', len(observations), 245, flush=True)
        calls = dict(pipeline.counts)
        primitive_calls = dict(cv_calls)
        model_calls = dict(detector=pipeline.models.detector_forwards, N3=pipeline.models.n3_forwards,
                           ROLE=calls.get('ROLE_head_calls', 0))
        model_bindings = pipeline.bindings
    C.require(len(observations) == 245 and len(geometry) == 1225 and len(controls) == 490, 'complete output population differs')
    C.save_rows(C.output_path(args, 'OBSERVATIONS.jsonl.gz'), observations)
    C.save_rows(C.output_path(args, 'GEOMETRY_SEALED.jsonl.gz'), geometry)
    C.save_rows(C.output_path(args, 'FIXED_GEOMETRY_SEALED.jsonl.gz'), controls)
    C.write_new(C.output_path(args, 'BASE_N3_PARITY.json'), dict(passed=True, rows=parity, rtol=0, atol=1e-7))
    C.require(C.prior_snapshot(args.prior_bindings) == prior, 'protected previous files changed')
    C.write_new(C.output_path(args, 'GEOMETRY_SEAL.json'), dict(schema='validated_boundary_geometry_seal_v2',
        complete=True, frames=245, methods=list(C.METHODS), rows=1225, fixed_rows=490,
        GT_read_allowed=False, protocol=C.binding(args.protocol), observations=C.binding(Path(args.output) / 'OBSERVATIONS.jsonl.gz'),
        geometry=C.binding(Path(args.output) / 'GEOMETRY_SEALED.jsonl.gz'),
        fixed_geometry=C.binding(Path(args.output) / 'FIXED_GEOMETRY_SEALED.jsonl.gz'),
        parity=C.binding(Path(args.output) / 'BASE_N3_PARITY.json'), actual_calls=calls,
        actual_OpenCV_entry_calls=primitive_calls,
        actual_model_forwards=model_calls, banks=banks, model_bindings=model_bindings,
        wall_seconds=time.monotonic() - start, runtime_benchmark=False,
        protected_materialized_files=len(prior)))
    print('BOUNDARY_REFINER_SEALED', len(geometry), 'before_scoring', flush=True)


def predict_cli(args):
    import cv2
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    C.require(image is not None, 'deployment image decode failed')
    destination = Path(args.output)
    C.require(destination.suffix == '.json' and not destination.is_symlink(), 'predict --output must be a new JSON file')
    from argparse import Namespace
    directory_args = Namespace(**vars(args))
    directory_args.output = str(destination.parent)
    destination = C.output_path(directory_args, destination.name)
    args.output = str(destination.parent)
    with Pipeline(args) as pipeline:
        result = pipeline.predict(image, C.read(args.K_json), C.read(args.dimensions_json),
                                  method=args.method)
    C.write_new(destination, result)
    print('BOUNDARY_REFINER_PREDICTED', destination, flush=True)


def main():
    parser = C.parser(__doc__, ('freeze', 'preflight', 'infer', 'predict'))
    parser.add_argument('--image')
    parser.add_argument('--K-json')
    parser.add_argument('--dimensions-json')
    parser.add_argument('--method', choices=C.METHODS, default=C.PRIMARY)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args)
    elif args.stage == 'preflight':
        C.verify_protocol(args); C.cohort_frames(args); C.prior_snapshot(args.prior_bindings)
        print('BOUNDARY_REFINER_PREFLIGHT_PASS', flush=True)
    elif args.stage == 'predict':
        C.require(args.image and args.K_json and args.dimensions_json, 'predict needs image/K-json/dimensions-json')
        predict_cli(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
