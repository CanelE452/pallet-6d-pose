"""One-shot synthetic CPU contracts for three heads and two point supplies.

No learned model, detector, N3 correction, source cache, real RGB, private GT,
ray or training is executed. Capture dispatch uses explicitly labelled stubs;
point-only solver and corner-heldout selector checks use actual frozen v4 math.
Run only after parent review/GO. Existing receipts are never overwritten.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
import io
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
import cv2
import numpy as np

from . import common as C
from . import pipeline as P
from ..pallet_cornerwise_independent_20261010_v4 import pose as V4
from ..pallet_cornerwise_independent_20261010_v4.selection import select_corners as frozen_select
from ..pallet_observation_refiner_20261009_v1.solver import cuboid, project

SIZE = (1280, 960)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def fixture():
    K = np.array([[600., 0., 640.], [0., 600., 480.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.3])
    R = cv2.Rodrigues(np.array([.3, .25, .1]))[0]
    t = np.array([0., .1, 4.])
    q = np.vstack([project(cuboid(*xyz), R, t, K), [641.25, 481.75]])
    initial = dict(available=True, R_cf=R.tolist(), R_physical=R.tolist(),
                   centroid=t.tolist(), cf_extents=xyz.tolist())
    return K, xyz, q, initial


def observation(q, ids):
    """Ideal mathematical supports; not a real physical-ownership assertion."""
    lines, corners = {}, []
    for k in ids:
        incident = [e for e, endpoints in enumerate(EDGES) if k in endpoints][:2]
        corners.append(dict(id=k, xy=np.asarray(q[k]).tolist(), edges=incident, radius_px=.5))
        for edge in incident:
            if edge in lines:
                continue
            a, b = EDGES[edge]
            tangent = (q[b] - q[a]) / np.linalg.norm(q[b] - q[a])
            normal = np.array([-tangent[1], tangent[0]])
            support = np.array([(1-u)*q[a] + u*q[b] for u in (.25, .5, .75)])
            lines[edge] = dict(edge=edge, normal=normal.tolist(), offset=float(normal @ q[a]),
                support_points=support.tolist(), query_radii_px=[1., 1., 1.],
                queries=[edge*7+j for j in range(3)])
    return dict(corners=corners, lines=[lines[e] for e in sorted(lines)], queries=[],
                selected_queries=3*len(lines), raw_logits_sha256='synthetic_no_head_logits',
                all_no_match=not bool(lines))


@contextmanager
def forbidden_reads():
    saved = [(builtins, 'open', builtins.open), (io, 'open', io.open), (Path, 'open', Path.open)]
    tokens = ('TARGETS', 'GROUND_TRUTH', 'READY_SOURCE', 'FEATURES.NPY', 'BEST.PT', 'LAST.PT',
              'IMAGE_ROLE.PT', 'GEOMETRY_ONLY.PT', 'IMAGE_NO_ROLE.PT', 'PREDICTIONS.JSON',
              'OBSERVATIONS.JSON', 'INPUTS.JSON', 'STATIC_VISIBILITY')
    attempts = []

    def wrap(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and any(token in str(path).upper() for token in tokens):
                attempts.append(str(path))
                raise AssertionError('forbidden private/cache/model read')
            return fn(path, *args, **kwargs)
        return call

    for obj, name, fn in saved:
        setattr(obj, name, wrap(fn))
    try:
        yield attempts
    finally:
        for obj, name, fn in saved:
            setattr(obj, name, fn)


def equal_numeric(a, b):
    for key in ('available', 'state', 'used', 'fit_input_ids', 'final_inliers', 'selected_hypothesis'):
        assert a.get(key) == b.get(key), key
    for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected', 'residuals_used_px'):
        if a.get(key) is not None:
            assert np.allclose(a[key], b[key], rtol=0, atol=1e-6), key
    for key in ('truncated_sse_px2', 'sse_px2'):
        if a.get(key) is not None:
            assert abs(a[key] - b[key]) <= 1e-8, key
    assert a['prior_used'] is b['prior_used'] is False


def audit_solver(result, hidden):
    blocked = set(hidden)
    for field in ('used', 'fit_input_ids', 'final_inliers', 'generator_ids'):
        assert blocked.isdisjoint(result.get(field, [])), field
    for candidate in result.get('all_candidate_solutions', []):
        for field in ('generator_ids', 'actual_fit_input_ids', 'inlier_ids'):
            assert blocked.isdisjoint(candidate[field]), field
    for field in ('prior_used', 'initial_pose_used', 'initial_projection_used',
                  'initial_dimension_prior_used', 'known_dimension_constraint_used',
                  'reprojected_points_reused_as_observations'):
        assert result[field] is False, field


def dummy_pipeline(root):
    pipe = P.Pipeline.__new__(P.Pipeline)
    pipe.args = SimpleNamespace(calibration_root=str(root))
    pipe.counts = Counter()
    pipe.old = SimpleNamespace(no_truth_reads=nullcontext)
    return pipe


def capture_fixture(q, K, xyz, initial, observations, hidden=()):
    raw = dict(selected_index=0, candidates=[dict(score=.9, cls=0, keypoints_xy=q.tolist()),
                                              dict(score=.1, cls=1, keypoints_xy=(q+50).tolist())])
    return dict(raw=raw, prediction=copy.deepcopy(raw), original_base_points=q.copy(),
        native_N3_points=q.copy(), initial_pose=copy.deepcopy(initial), initial_base_pose=copy.deepcopy(initial),
        hidden=list(hidden), mask_diagnostic=dict(synthetic_fixture=True), observations=observations,
        observation=observations.get('IMAGE_ROLE', next(iter(observations.values()), {})),
        metadata=dict(id='synthetic', session='synthetic', K=K.tolist(), xyz=xyz.tolist(), raw_hw=[SIZE[1], SIZE[0]]))


def capture_stub(q, K, xyz):
    """A call-routing witness, not an actual detector or learned head."""
    pipe = P.Pipeline.__new__(P.Pipeline)
    pipe.counts = Counter()
    pipe.old = SimpleNamespace(no_truth_reads=nullcontext)
    pipe.torch = SimpleNamespace(no_grad=nullcontext)
    trace = Counter()
    forwarded = []
    features = np.arange(84*28*65, dtype=np.float32).reshape(84, 28, 65)
    predicted = dict(selected_index=0, candidates=[dict(score=.9, cls=0, keypoints_xy=q.tolist())])

    def extract(_):
        trace['detector_stub_calls'] += 1
        return copy.deepcopy(predicted)

    def correct(method, image, captured, meta):
        assert method == 'N3_SUBPIX'
        trace['N3_stub_calls'] += 1
        return copy.deepcopy(predicted), {}, q.copy()

    def infer(points, camera, dims, source):
        assert source is False
        trace['initial_pose_stub_calls'] += 1
        return dict(available=False)

    def inputs(image, captured, original, camera, dims):
        trace['feature_inputs_stub_calls'] += 1
        assert np.array_equal(original, q)
        return features, dict(synthetic_feature_fixture=True)

    class Head:
        def __init__(self, arm):
            self.arm = arm

        def __call__(self, x, mode):
            assert mode == self.arm and np.shares_memory(x, features)
            forwarded.append(dict(arm=self.arm, mode=mode, shared_tensor=True))
            trace['head_stub_calls'] += 1
            return np.zeros((1, 84, 66), np.float32)

    pipe.models = SimpleNamespace(extractor=SimpleNamespace(predict=extract),
        correct=correct, pose=SimpleNamespace(infer=infer),
        inf=SimpleNamespace(serial=lambda x: copy.deepcopy(x)))
    pipe.learned = SimpleNamespace(M=SimpleNamespace(inputs=inputs))
    pipe.heads = {arm: Head(arm) for arm in C.ARMS}
    pipe.calibrations = {arm: {} for arm in C.ARMS}
    pipe.decoder = SimpleNamespace(decode=lambda query, logits, calibration: observation(q, (0, 1, 2, 4, 5)))
    pipe.registry_metadata = lambda image, camera, dims, metadata: dict(metadata or {},
        K=np.asarray(camera).tolist(), xyz=np.asarray(dims).tolist(), raw_hw=list(image.shape[:2]))
    return pipe, trace, forwarded


def run(args):
    C.require(args.source_root and args.baseline_root, 'explicit source and baseline roots required')
    output = C.output_path(args, 'PIPELINE_CHECKS.json')
    started = C.output_path(args, 'PIPELINE_CHECKS_STARTED.json')
    C.write_new(started, dict(schema='three_head_pipeline_checks_started_v7',
        code=C.binding(P.__file__), tests=C.binding(__file__), real_model_calls=0,
        declared_scope='synthetic point math and capture dispatch stubs; no learning/real evaluation'))
    cv2.setNumThreads(1)
    counts, checks = Counter(), []
    names = ('solvePnPGeneric', 'solvePnPRefineLM', 'solvePnP', 'projectPoints')
    original = {name: getattr(cv2, name) for name in names}
    for name, fn in original.items():
        def counted(*a, _name=name, _fn=fn, **kw):
            counts[_name] += 1
            return _fn(*a, **kw)
        setattr(cv2, name, counted)

    def check(name, fn):
        before = counts.copy()
        try:
            with forbidden_reads() as reads:
                details = fn() or {}
            assert not reads
            checks.append(dict(name=name, status='PASS', details=C.finite(details),
                               actual_calls=dict(counts-before), forbidden_read_attempts=reads))
        except Exception as error:
            checks.append(dict(name=name, status='FAIL', error=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc(), actual_calls=dict(counts-before)))

    try:
        with tempfile.TemporaryDirectory(prefix='three-head-pipeline-fixture-', dir='/tmp') as temporary:
            root = Path(temporary)
            for arm in C.ARMS:
                (root/arm).mkdir()
                C.write_new(root/arm/'CALIBRATION.json', dict(synthetic_fixture=True))

            def routes():
                assert len(C.METHODS) == len(set(C.METHODS)) == 8
                assert len(C.METHOD_SPECS) == 6
                assert C.PRIMARY == 'IMAGE_ROLE_BOUNDARY_ONLY'
                assert {(s['arm'], s['supply']) for s in C.METHOD_SPECS.values()} == {
                    (a, s) for a in C.ARMS for s in ('BOUNDARY_ONLY', 'CORNERWISE_HYBRID')}
                assert not P.POLICY['initial_rotation_translation_projection_dimension_prior']
                assert not P.POLICY['partial_line_pose_factors']
                assert not P.POLICY['boundary_only_native_distance_or_LOO_gate']
                return dict(heads=3, supplies=2, native_controls=2, fixed_primary=C.PRIMARY)
            check('fixed_three_by_two_and_native_controls_no_prior_or_C2', routes)

            def shared_capture():
                K, xyz, q, _ = fixture()
                pipe, trace, forwarded = capture_stub(q, K, xyz)
                captured = pipe.capture(np.zeros((SIZE[1], SIZE[0], 3), np.uint8), K, xyz,
                    dict(id='synthetic'), need_arms=C.ARMS, need_base_pose=True)
                assert trace == Counter(detector_stub_calls=1, N3_stub_calls=1,
                    initial_pose_stub_calls=2, feature_inputs_stub_calls=1, head_stub_calls=3)
                assert [r['mode'] for r in forwarded] == list(C.ARMS)
                assert set(captured['observations']) == set(C.ARMS)
                assert captured['observation'] is captured['observations']['IMAGE_ROLE']
                assert pipe.counts['feature_initial_pose_calls'] == pipe.counts['shared_feature_tensors'] == 1
                assert pipe.counts['head_calls'] == 3
                assert all(pipe.counts[a+'_head_calls'] == 1 for a in C.ARMS)
                return dict(dispatch_stub_counts=dict(trace), actual_learned_head_forwards=0,
                            head_mode_arguments=forwarded, feature_tensor_shared=True)
            check('capture_shares_one_tensor_and_dispatches_exact_training_modes', shared_capture)

            def single_head_capture():
                K, xyz, q, _ = fixture()
                cases = []
                for arm in C.ARMS:
                    pipe, trace, forwarded = capture_stub(q, K, xyz)
                    result = pipe.capture(np.zeros((SIZE[1], SIZE[0], 3), np.uint8), K, xyz, need_arms=(arm,))
                    assert trace['head_stub_calls'] == trace['feature_inputs_stub_calls'] == 1
                    assert list(result['observations']) == [arm] and forwarded[0]['mode'] == arm
                    cases.append(dict(arm=arm, calls=dict(trace)))
                pipe, trace, _ = capture_stub(q, K, xyz)
                result = pipe.capture(np.zeros((SIZE[1], SIZE[0], 3), np.uint8), K, xyz, need_arms=())
                assert not result['observations'] and not trace['head_stub_calls'] and not trace['feature_inputs_stub_calls']
                return dict(single_head_capture_stubs=cases, native_control_has_zero_feature_or_head_stub_calls=True)
            check('chosen_head_only_and_native_control_zero_head_dispatch', single_head_capture)

            def sparse_no_native_gate():
                K, xyz, q, initial = fixture()
                native = q.copy(); native[:8] += [35., -27.]
                obs = observation(q, (0, 1, 2, 3, 4, 5))
                capture = capture_fixture(native, K, xyz, initial, {a: copy.deepcopy(obs) for a in C.ARMS}, hidden=(6, 7))
                pipe = dummy_pipeline(root)
                saved = P.select_corners
                P.select_corners = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('boundary-only called LOO'))
                try:
                    rows, banks = pipe.outcomes(capture, tuple(a+'_BOUNDARY_ONLY' for a in C.ARMS))
                finally:
                    P.select_corners = saved
                assert banks['coordinate_bank_count'] == 1
                for row in rows:
                    assert row['solver']['available'], row['solver']['state']
                    assert np.array_equal(np.asarray(row['input_points'])[:6], q[:6])
                    assert np.isnan(np.asarray(row['input_points'])[6:8]).all()
                    assert not row['native_N3_used_as_independent_RGB_observations']
                    assert not row['observation_contract']['boundary_only_native_distance_used_for_admission']
                    audit_solver(row['solver'], (6, 7))
                assert pipe.counts['LOO_pose_paths'] == 0
                return dict(outside_native8px_boundary_still_admitted=True, actual_final_paths=3,
                            LOO_calls=0, unique_sparse_bank=1)
            check('boundary_only_admits_valid_sparse_points_without_native_basin_or_LOO', sparse_no_native_gate)

            def sparse_not_filled():
                K, xyz, q, initial = fixture()
                obs = observation(q, (0, 1, 4))
                capture = capture_fixture(q, K, xyz, initial, {'IMAGE_ROLE': obs})
                rows, _ = dummy_pipeline(root).outcomes(capture, (C.PRIMARY,))
                row = rows[0]
                assert row['solver']['state'] == 'INSUFFICIENT_OBSERVATIONS'
                assert row['solver']['used'] == [0, 1, 4]
                assert row['fallback_used'] and row['output_status'] == 'N3_BASELINE_FALLBACK'
                assert np.array_equal(row['native_points'], q) and not row['new_pose_estimated']
                assert np.isnan(np.asarray(row['input_points'])[[2, 3, 5, 6, 7]]).all()
                return dict(sparse_available_points=3, default_return_is_declared_N3_fallback=True,
                            displayed_N3_coordinates_not_final_fit_inputs=True)
            check('less_than_four_actual_boundary_points_is_unavailable_not_filled', sparse_not_filled)

            def actual_four_five():
                K, xyz, q, initial = fixture(); details = []
                for ids in ((0, 1, 2, 4), (0, 1, 2, 4, 6)):
                    obs = observation(q, ids)
                    capture = capture_fixture(q, K, xyz, initial, {'IMAGE_ROLE': obs})
                    row = dummy_pipeline(root).outcomes(capture, (C.PRIMARY,))[0][0]
                    sparse = P.observation_points(q, obs, ())[0]
                    direct = V4.PoseBank(sparse, K, xyz, image_size=SIZE).solve(robust=True)
                    equal_numeric(row['solver'], direct)
                    assert row['solver']['used'] == list(ids)
                    assert row['solver']['state'] not in ('INSUFFICIENT_OBSERVATIONS', 'DEGENERATE_OBSERVATIONS')
                    assert row['solver']['eligible_candidate_count'] > 0
                    audit_solver(row['solver'], ())
                    details.append(dict(actual_points=len(ids), state=row['solver']['state'],
                        unresolved_ambiguity=row['solver']['unresolved_ambiguity']))
                return dict(actual_numeric_results=details, natural_ambiguity_not_forced_to_success=True)
            check('actual_four_and_five_point_sparse_paths_match_frozen_solver', actual_four_five)

            def hybrid_parity():
                K, xyz, q, initial = fixture(); native = q.copy(); native[0] += [2., 1.]
                obs = observation(q, (0, 1, 2, 4, 5))
                capture = capture_fixture(native, K, xyz, initial, {'IMAGE_ROLE': obs}, hidden=(7,))
                pipe = dummy_pipeline(root)
                row = pipe.outcomes(capture, ('IMAGE_ROLE_CORNERWISE_HYBRID',))[0][0]
                bank = V4.PoseBank(native, K, xyz, image_size=SIZE)
                expected, contract = frozen_select(native, obs, (7,), K, xyz, SIZE, bank=bank)
                direct_bank = bank if np.array_equal(expected, native, equal_nan=True) else V4.PoseBank(expected, K, xyz, image_size=SIZE)
                solved = direct_bank.solve(hidden=(7,), robust=True)
                assert np.array_equal(row['input_points'], expected, equal_nan=True)
                assert row['observation_contract']['hybrid_boundary_corner_ids'] == contract['hybrid_boundary_corner_ids']
                equal_numeric(row['solver'], solved)
                for record in row['observation_contract']['cornerwise_records']:
                    if record['heldout_pose_calls']:
                        audit_solver(record['loo_solver'], (record['id'], 7))
                return dict(exact_same_selected_coordinates=True, same_v4_numeric_status_and_pose=True,
                            actual_LOO_calls=pipe.counts['LOO_pose_paths'])
            check('ROLE_hybrid_is_unchanged_v4_point_control_with_actual_heldout_exclusion', hybrid_parity)

            def shared_banks():
                K, xyz, q, initial = fixture()
                obs = observation(q, (0, 1, 2, 3, 4, 5))
                captured = capture_fixture(q, K, xyz, initial, {a: copy.deepcopy(obs) for a in C.ARMS})
                pipe = dummy_pipeline(root)
                rows, banks = pipe.outcomes(captured)
                assert len(rows) == 8 and banks['coordinate_bank_count'] == 2
                assert pipe.counts['head_calls'] == 0  # outcomes do not call any head
                assert pipe.counts['LOO_pose_paths'] == 18
                ledgers = banks['coordinate_banks']
                for ledger in ledgers:
                    assert ledger['subsets_considered'] <= 140
                native_rows = [r for r in rows if r['head_arm'] is None]
                assert len(native_rows) == 2
                assert native_rows[0]['solver']['input_hash'] == native_rows[1]['solver']['input_hash']
                return dict(unique_coordinate_banks=2, actual_LOO_requests=18,
                            shared_native_bank_across_three_heads_and_both_native_masks=True,
                            ledgers=ledgers)
            check('same_coordinates_share_finite_bank_across_three_heads_and_native_masks', shared_banks)

            def hidden_and_metadata():
                K, xyz, q, initial = fixture(); obs = observation(q, tuple(range(8)))
                capture = capture_fixture(q, K, xyz, initial, {'IMAGE_ROLE': obs}, hidden=(6,))
                row = dummy_pipeline(root).outcomes(capture, (C.PRIMARY,))[0][0]
                assert row['new_pose_estimated'] and row['hidden_reprojected']
                audit_solver(row['solver'], (6,))
                assert np.array_equal(row['native_points'][6], np.asarray(row['solver']['projected'])[6])
                assert np.array_equal(row['native_points'][8], q[8])
                assert not row['reprojections_reused_as_observations']
                after = copy.deepcopy(capture['prediction'])
                after['candidates'][0]['keypoints_xy'] = row['native_points']
                assert P.preserve_prediction(capture['raw'], after)
                try:
                    altered = copy.deepcopy(after); altered['candidates'][0]['score'] = .91
                    P.preserve_prediction(capture['raw'], altered)
                except RuntimeError as error:
                    assert 'confidence/class' in str(error)
                else:
                    raise AssertionError('candidate confidence mutation was accepted')
                return dict(final_H_projection_no_refit=True, center_and_detector_metadata_preserved=True,
                            remaining_point_pose_used_despite_mask_exclusion=True)
            check('H_fit_exclusion_final_projection_and_detector_metadata_center', hidden_and_metadata)

            def final_support_veto():
                K, xyz, q, initial = fixture(); obs = observation(q, (0, 1, 2, 4, 5))
                obs['lines'][0]['support_points'][0] = (
                    np.asarray(obs['lines'][0]['support_points'][0]) + 30*np.asarray(obs['lines'][0]['normal'])).tolist()
                _, _, contract = P.observation_points(q, obs, ())
                assert contract['invalid_final_line_edges'] == [obs['lines'][0]['edge']]
                rejected = [r for r in contract['corner_admission'] if not r['accepted']]
                assert rejected and all(r['reason'] == 'FINAL_LINE_CONSENSUS_INCONSISTENT' for r in rejected)
                obs = observation(q, (0,)); obs['corners'][0]['radius_px'] = 8.000001
                sparse, _, contract = P.observation_points(q, obs, ())
                assert np.isnan(sparse[:8]).all() and not contract['validated_boundary_corner_ids']
                return dict(final_support_residual_veto_and_above8px_corner_uncertainty_abstain=True)
            check('frozen_support_and_uncertainty_admission_remain_effective', final_support_veto)

            def boundary_no_initial_numeric_influence():
                K, xyz, q, initial = fixture(); obs = observation(q, (0, 1, 2, 3, 4, 5))
                results = []
                for shift in (0., 100.):
                    native = q.copy(); native[:8] += shift
                    changed = copy.deepcopy(initial); changed['centroid'] = [10.+shift, -3., 8.]
                    changed['R_cf'] = np.eye(3).tolist(); changed['cf_extents'] = xyz[[2, 1, 0]].tolist()
                    captured = capture_fixture(native, K, xyz, changed, {'IMAGE_ROLE': obs}, hidden=(6, 7))
                    row = dummy_pipeline(root).outcomes(captured, (C.PRIMARY,))[0][0]
                    assert row['solver']['available']
                    results.append(row['solver'])
                equal_numeric(*results)
                return dict(frozen_H_conditioned_sparse_numeric_pose_invariant_to_initial_R_t_dimension_and_native_coordinates=True,
                            proposal_generation_not_claimed_independent=True)
            check('boundary_only_final_numeric_pose_never_uses_initial_or_native_coordinate_prior', boundary_no_initial_numeric_influence)

            def failed_preflight():
                with tempfile.TemporaryDirectory(prefix='three-head-preflight-', dir='/tmp') as temp:
                    bad = Path(temp)
                    C.write_new(bad/'TRAINING_COMPLETION.json', dict(complete=True, formal_updates=8999, total_updates=9000))
                    args_bad = SimpleNamespace(source_root='/synthetic_source', baseline_root='/synthetic_baseline', fits=str(bad))
                    try:
                        P.checkpoint_preflight(args_bad)
                    except RuntimeError as error:
                        assert 'training completion invalid' in str(error)
                    else:
                        raise AssertionError('incomplete training was accepted')
                assert 'torch' not in sys.modules
                return dict(incomplete_completion_rejected_before_Torch_or_model_import=True,
                            fake_fixture_only=True)
            check('invalid_completion_fails_before_heavy_import_or_weights', failed_preflight)

            def lifecycle():
                import scripts.research as research
                before = list(research.__path__); events = Counter()
                pipe = P.Pipeline.__new__(P.Pipeline)
                pipe.closed = False; pipe._context_active = True; pipe._before_namespace = before
                research.__path__ = before + ['/synthetic_extra_namespace']
                P.Pipeline._active_lifecycle = True

                def close_models():
                    events['models_close_stub_calls'] += 1
                    raise RuntimeError('synthetic model-close failure')

                def exit_context(*_):
                    events['context_exit_stub_calls'] += 1
                pipe.models = SimpleNamespace(close=close_models)
                pipe.context = SimpleNamespace(__exit__=exit_context)
                try:
                    pipe.close()
                except RuntimeError as error:
                    assert str(error) == 'synthetic model-close failure'
                else:
                    raise AssertionError('expected synthetic close failure')
                assert list(research.__path__) == before and not P.Pipeline._active_lifecycle
                pipe.close()
                assert events == Counter(models_close_stub_calls=1, context_exit_stub_calls=1)
                return dict(cleanup_restores_namespace_even_if_models_close_fails=True,
                            repeated_close_is_idempotent=True, counts=dict(events))
            check('single_outer_lifecycle_cleanup_is_exception_safe_and_idempotent', lifecycle)

    finally:
        for name, fn in original.items():
            setattr(cv2, name, fn)
    passed = all(row['status'] == 'PASS' for row in checks)
    payload = dict(schema='three_head_two_supply_synthetic_pipeline_checks_v7', complete=True,
        passed=passed, check_count=len(checks), checks=checks, actual_OpenCV_calls=dict(counts),
        actual_detector_forwards=0, actual_N3_forwards=0, actual_head_forwards=0,
        actual_real_images=0, actual_private_GT_reads=0, actual_rays=0, training_updates=0,
        source='synthetic analytic projections, actual frozen v4 PnP/selector, explicitly labelled capture dispatch stubs',
        limits='dispatch stubs do not certify actual GPU inputs or head/checkpoint correctness; formal frozen execution checks those bindings',
        code={name: C.binding(path) for name, path in dict(pipeline=P.__file__, tests=__file__,
            borrowed_pose=V4.__file__, borrowed_selection=frozen_select.__code__.co_filename).items()})
    C.write_new(output, payload)
    print('THREE_HEAD_PIPELINE_CHECKS', len(checks), 'PASS' if passed else 'FAIL', flush=True)
    return payload


if __name__ == '__main__':
    parser = C.parser(__doc__)
    parser.set_defaults(output=str(C.DOC))
    result = run(parser.parse_args())
    raise SystemExit(0 if result['passed'] else 1)
