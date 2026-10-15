"""Pure contracts and frozen numerical-path comparisons; no actual RGB/model IO."""
from copy import deepcopy
import ast
import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as R
from scripts.research.pallet_minimal_bi_ablation_v1 import cli


def fixture():
    K = np.array([[180., 0, 80], [0, 180., 60], [0, 0, 1]])
    rotation = Rotation.from_euler('xyz', [8, -22, 4], degrees=True).as_matrix()
    Q = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    pose = dict(available=True, R_cf=rotation.tolist(), R_physical=(rotation @ Q).tolist(),
                centroid=[.04, -.02, 3.], cf_extents=[1.2, .2, .8])
    geometry = R.registered_geometry((120, 160), K, R.C.cuboid(*pose['cf_extents']), pose)
    q = np.vstack([geometry['initial_projection_xy'] + [1.5, -.5], [80., 60.]])
    image = np.zeros((120, 160, 3), np.uint8)
    return image, q, K, pose, geometry


class Provider:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return dict(pallet_mask=np.zeros(image.shape[:2], np.float32),
                    state_pallet_mask=np.ones(image.shape[:2], np.float32),
                    provenance=dict(schema='TOY_MASKS', prompt_native_ids=kwargs['prompt_native_ids']))


def state_producer(image, raw, hidden, roles, projected, model, **kwargs):
    records, eligible = R.geometry_state_gates_v4(image, raw, hidden, roles, projected)
    for k in eligible:
        records[k].update(state=roles[k], rgb_evaluated=True, p_visible=1., abstained=False)
    return dict(states=[r['state'] for r in records], records=records,
                rgb_queried_ids=eligible.tolist())


def components(producer=state_producer):
    return R.Components(None, Provider(), state_producer=producer)


def empty_observation(states=None):
    _, _, K, pose, geometry = fixture()
    return dict(states=states or ['U'] * 8,
                initial_geometry=R._json(geometry), frozen_visible_measurements=[]), pose, K


def measurement(k, xy):
    return dict(native_corner=k, xy=list(xy), covariance_px2=np.eye(2).tolist(),
                protection_px=8., provenance={'source': 'TOY_BI'})


def test_parity_tolerances_locked_and_same_entire_solver_configuration():
    assert R.PARITY_POLICY['numeric_atol'] == 1e-9
    assert R.PARITY_POLICY['numeric_rtol'] == 0
    assert R.PARITY_POLICY['tolerance_may_be_relaxed_after_results'] is False
    assert R.SOLVER_CONFIG == {**R.JOINT._DEFAULTS, 'analytic_jacobian': True}


def test_registered_geometry_matches_frozen_function_without_importing_old_stack():
    source = Path(R.JOINT.__file__).with_name('statewise_joint_api_v2.py').read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in ('_array', '_rotation', 'registered_geometry')]
    namespace = dict(np=np, cv2=R.cv2, ConvexHull=R.ConvexHull, C=R.C,
                     project_native=R.JOINT.project_native, Any=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-pure-geometry>', 'exec'), namespace)
    image, _, K, pose, geometry = fixture()
    old = namespace['registered_geometry'](image.shape[:2], K, geometry['native_xyz'], pose)
    assert R._json(old) == R._json(geometry)


def test_S_erased_before_mask_and_state_and_no_nonBI_local_queries():
    image, q, K, pose, geometry = fixture()
    hidden = geometry['initial_self_hidden']
    assert hidden.any()
    q[:8][hidden] = [1e8, -1e8]
    original = q.copy()
    seen = {}
    def producer(image, raw, hidden, roles, projected, model, **kwargs):
        assert np.array_equal(raw[:8][hidden], np.full((hidden.sum(), 2), -1.))
        result = state_producer(image, raw, hidden, roles, projected, model, **kwargs)
        eligible = result['rgb_queried_ids']
        for k, state in zip(eligible, ['O', 'U', 'T']):
            result['states'][k] = state
            result['records'][k]['state'] = state
        seen['states'] = result['states']
        return result
    context = components(producer)
    result = R.observe(image, q, K, pose, components=context)
    assert np.array_equal(q, original)
    ids = context.mask_provider.calls[0]['prompt_native_ids']
    assert not set(np.flatnonzero(hidden)).intersection(ids)
    assert 8 not in ids
    records = result['observation']['visible_correction']['records']
    for k, state in enumerate(seen['states']):
        if state in ('S', 'O', 'T', 'U'):
            assert not records[k].get('pixel_queried')
    assert {m['native_corner'] for m in result['observation']['frozen_visible_measurements']} == {
        k for k, state in enumerate(seen['states']) if state in ('B', 'I')}


def test_observe_does_not_mutate_image_q_K_pose_or_provider_masks():
    image, q, K, pose, _ = fixture()
    before = (image.copy(), q.copy(), K.copy(), deepcopy(pose))
    context = components()
    a = R.observe(image, q, K, pose, components=context)
    a['masks']['pallet_mask'][:] = 1
    b = R.observe(image, q, K, pose, components=context)
    assert not b['masks']['pallet_mask'].any()
    assert np.array_equal(image, before[0]) and np.array_equal(q, before[1])
    assert np.array_equal(K, before[2]) and pose == before[3]
    assert a['observation'] == b['observation']
    assert 'ot_domains' not in a['observation'] and 'u_region_display' not in a['observation']


def test_empty_invalid_queries_no_mask_forward_and_empty_BI():
    image, q, K, pose, _ = fixture()
    q[:] = [-1., -1.]
    context = components()
    result = R.observe(image, q, K, pose, components=context)
    assert context.mask_provider.calls == []
    assert result['observation']['frozen_visible_measurements'] == []
    assert not result['masks']['pallet_mask'].any()
    solved = R.solve(result['observation'], pose, K, image.shape[:2])
    assert not solved['accepted'] and solved['fallback']
    assert solved['actual_pose']['source'] == 'ORIGINAL_PNP_FALLBACK'


@pytest.mark.parametrize('count', [0, 1, 2])
def test_underconstrained_keeps_row_and_own_coarse(count):
    observation, pose, K = empty_observation()
    geometry = observation['initial_geometry']
    for k in range(count):
        observation['states'][k] = 'B'
        observation['frozen_visible_measurements'].append(measurement(k, geometry['initial_projection_xy'][k]))
    before = deepcopy(observation)
    result = R.solve(observation, pose, K, [120, 160])
    assert not result['accepted'] and result['fallback']
    assert result['actual_pose']['R_cf'] == pose['R_cf']
    assert result['actual_pose']['centroid'] == pose['centroid']
    assert result['input_native_ids'] == list(range(count))
    assert before == observation


@pytest.mark.parametrize('covariance', [[[1, 2], [0, 1]], [[1, 0], [0, 0]], [[1, 0], [0, -1]], [[float('nan'), 0], [0, 1]]])
def test_bad_covariance_is_rejected_not_repaired(covariance):
    observation, pose, K = empty_observation()
    observation['states'][0] = 'B'
    observation['frozen_visible_measurements'] = [measurement(0, [60, 60])]
    observation['frozen_visible_measurements'][0]['covariance_px2'] = covariance
    with pytest.raises(ValueError, match='covariance'):
        R.solve(observation, pose, K, [120, 160])


@pytest.mark.parametrize('state', ['S', 'O', 'T', 'U'])
def test_nonBI_correspondence_rejected(state):
    observation, pose, K = empty_observation()
    observation['states'][0] = state
    observation['frozen_visible_measurements'] = [measurement(0, [60, 60])]
    with pytest.raises(ValueError, match='Only unique B/I'):
        R.solve(observation, pose, K, [120, 160])


def test_duplicate_BI_rejected():
    observation, pose, K = empty_observation()
    observation['states'][0] = 'B'
    observation['frozen_visible_measurements'] = [measurement(0, [60, 60])] * 2
    with pytest.raises(ValueError, match='Only unique B/I'):
        R.solve(observation, pose, K, [120, 160])


@pytest.mark.parametrize('branch', ['accepted', 'seed', 'coarse'])
def test_exact_frozen_solver_path_config_basis_fallback_and_measurement_projection_separation(monkeypatch, branch):
    observation, pose, K = empty_observation(['B', 'I', 'O', 'T', 'U', 'S', 'U', 'U'])
    observation['frozen_visible_measurements'] = [measurement(0, [40., 40.]), measurement(1, [80., 40.])]
    before = deepcopy(observation)
    captured = {}
    Rnext = Rotation.from_euler('xyz', [9, -20, 3], degrees=True).as_matrix()
    solved = dict(accepted=branch == 'accepted', R=Rnext.tolist(), t=[.1, .2, 3.1],
        corrected_BI_seed=dict(R=Rnext.tolist(), t=[.2, .1, 3.2], visible_support_guards_pass=branch == 'seed'))
    def fake(xyz, K, states, visible, domains, image_hw, **kwargs):
        captured.update(states=states, visible=visible, domains=domains, **kwargs)
        return deepcopy(solved)
    monkeypatch.setattr(R.JOINT, 'solve_joint_ot_pnp', fake)
    source = Path(R.JOINT.__file__).with_name('statewise_joint_experiment_v2.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_bi_only')
    namespace = dict(C=R.C, np=np, deepcopy=deepcopy, solve_joint_ot_pnp=fake)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen-bi-only>', 'exec'), namespace)
    old = namespace['_bi_only'](deepcopy(observation), dict(initial_pose=pose, K=K, raw_hw=[120, 160]), deepcopy(R.SOLVER_CONFIG))
    actual = R.solve(observation, pose, K, [120, 160])
    for key, value in R.C.finite(old).items():
        assert actual[key] == value
    assert actual['accepted'] == (branch == 'accepted')
    assert actual['fallback'] == (branch != 'accepted')
    assert captured['states'] == ['B', 'I', 'U', 'U', 'U', 'S', 'U', 'U']
    assert captured['domains'] == []
    assert captured['config'] == R.SOLVER_CONFIG
    assert captured['initializations'] == [dict(R=pose['R_cf'], t=pose['centroid'])]
    assert actual['frozen_visible_measurements'] == before['frozen_visible_measurements']
    assert not np.array_equal(np.asarray(actual['model_projection_xy'])[:2], [m['xy'] for m in before['frozen_visible_measurements']])
    assert observation == before


def test_observe_GT_reference_file_blocked_and_no_N3_needed(monkeypatch):
    image, q, K, pose, _ = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('No file access is permitted while in-memory observation/solve runs')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    result = R.observe(image, q, K, pose, components=components())
    assert result['operations']['new_N3_or_YOLO_forwards'] == 0
    assert result['operations']['state_ResNet_features_retained']
    # No file paths, score or labels exist in the solve interface.
    observation, pose, K = empty_observation()
    assert R.solve(observation, pose, K, [120, 160])['fallback']


def test_cli_GT_field_rejected_before_any_image_model_reference_access(tmp_path, monkeypatch):
    _, q, K, pose, _ = fixture()
    path = tmp_path / 'predictor.json'
    path.write_text(json.dumps(dict(K=K.tolist(), native_points=q.tolist(), coarse_pose=pose,
                                   GT='reference-never-open.json')))
    monkeypatch.setattr(R, 'load_components', lambda **kwargs: pytest.fail('model loaded before input validation'))
    monkeypatch.setattr(cli.cv2, 'imread', lambda *args: pytest.fail('RGB accessed before input validation'))
    with pytest.raises(ValueError, match='predictor input'):
        cli.main(['run', '--input', str(path), '--output', str(tmp_path / 'output')])
    assert not (tmp_path / 'output').exists()


def test_loader_keeps_frozen_ResNet_and_loads_no_N3_U_OT_assets(tmp_path, monkeypatch):
    array_path = tmp_path / 'state_arrays.npz'
    np.savez(array_path, coefficients=np.arange(3.))
    for name in ('resnet.weights', 'tiny.weights', 'sam.weights'):
        (tmp_path / name).write_bytes(name.encode())
    state = dict(schema=R.STATE_SCHEMA, arrays_file=array_path.name,
        arrays_sha256=R.sha256(array_path), feature_family='NON_S_RGB26_GEOMETRY_RAW_SPATIAL',
        backbone=dict(weights_repo_relative='resnet.weights',
                      weights_sha256=R.sha256(tmp_path / 'resnet.weights')),
        visible_threshold=.97, abstention_margin=.01)
    masks = dict(pallet_segmenter=dict(weights_repo_relative='tiny.weights',
                                      weights_sha256=R.sha256(tmp_path / 'tiny.weights')))
    boundary = dict(mobile_sam_vendor='vendor', mobile_sam_checkpoint='sam.weights',
                    model_sha256={'mobile_sam_checkpoint': R.sha256(tmp_path / 'sam.weights')})
    paths = {}
    for name, value in [('state.json', state), ('masks.json', masks), ('boundary.json', boundary)]:
        paths[name] = tmp_path / name
        paths[name].write_text(json.dumps(value))
    seen = {}
    def provider(**kwargs):
        seen.update(kwargs)
        return Provider()
    monkeypatch.setattr(R.MASK, 'StatewiseMaskProvider', provider)
    loaded = R.load_components(asset_root=tmp_path, state_descriptor=paths['state.json'],
        mask_descriptor=paths['masks.json'], boundary_config=paths['boundary.json'], device='cpu')
    assert loaded.state_model.descriptor == state
    assert loaded.state_model.frozen_backbone.weights_path == tmp_path / 'resnet.weights'
    assert loaded.state_model.frozen_backbone.backbone is None  # lazy; no neural forward
    assert not loaded.state_model.arrays['coefficients'].flags.writeable
    assert seen['tiny_weights'] == tmp_path / 'tiny.weights'
    assert seen['sam_checkpoint'] == tmp_path / 'sam.weights'
    assert set(loaded.bindings) == {'state_descriptor', 'mask_descriptor', 'boundary_config',
        'state_arrays', 'state_backbone', 'tiny_segmenter', 'mobile_sam'}
    assert not any('n3' in str(p).lower() for p in tmp_path.iterdir())


def test_single_CLI_solve_roundtrip_without_model_RGB_N3_or_reference(tmp_path, monkeypatch):
    observation, pose, K = empty_observation()
    source = dict(K=K.tolist(), native_points=[[-1., -1.]] * 9, coarse_pose=pose, raw_hw=[120, 160])
    (tmp_path / 'input.json').write_text(json.dumps(source))
    (tmp_path / 'observation.json').write_text(json.dumps(observation))
    monkeypatch.setattr(R, 'load_components', lambda **kwargs: pytest.fail('solve loaded neural components'))
    monkeypatch.setattr(cli.cv2, 'imread', lambda *args: pytest.fail('solve opened an image'))
    out = cli.main(['solve', '--input', str(tmp_path / 'input.json'),
        '--observation', str(tmp_path / 'observation.json'), '--output', str(tmp_path / 'out')])
    result = json.loads((out / 'POSE.json').read_text())
    assert result['fallback'] and not result['accepted']
    assert json.loads((out / 'RUN_RECEIPT.json').read_text())['reference_or_GT_inputs'] is False
    with pytest.raises(FileExistsError):
        cli.main(['solve', '--input', str(tmp_path / 'input.json'),
            '--observation', str(tmp_path / 'observation.json'), '--output', str(out)])


def test_minimal_import_graph_excludes_experiment_OT_U_scoring_and_N3():
    program = '''
import sys
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime
bad = [k for k in sys.modules if k.endswith(('statewise_joint_api_v2','statewise_joint_api_v4',
'statewise_joint_experiment_v2','statewise_evaluation_v4','ot_domains_v2',
'uncertain_u_regions_v4_5','evaluate_u_regions_v4_5')) or 'n3_subpix' in k.lower()]
assert not bad, bad
assert runtime.SOLVER_CONFIG['analytic_jacobian'] is True
'''
    run = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def test_actual_call_instrumentation_hooks_lazy_models_and_restores(monkeypatch):
    import torch
    from types import SimpleNamespace
    from scripts.research.pallet_minimal_bi_ablation_v1.instrumentation import instrument, SEG, STATE
    backbone = R.JointObservationModel({}, {}, Path('unused'), 'cpu', backbone=torch.nn.Identity())
    model = R.StateVisibilityModel({}, {}, backbone)
    tiny = torch.nn.Identity()
    monkeypatch.setattr(SEG, 'load_segmenter', lambda *args: (tiny, {'input_size': 8}, 'cpu'))
    monkeypatch.setattr(STATE, 'predict_actual_visibility', lambda *args: np.ones(1))
    monkeypatch.setattr(STATE, 'spatial_features', lambda *args: np.zeros((1, 1)))
    class Predictor:
        def __init__(self):
            self.model = SimpleNamespace(image_encoder=torch.nn.Identity(), mask_decoder=torch.nn.Identity())
        def set_image(self, value):
            return self.model.image_encoder(value)
        def predict(self, value):
            return self.model.mask_decoder(value)
    provider = Provider()
    provider.predictor = Predictor()
    context = R.Components(model, provider, state_producer=state_producer)
    original_solve = R.solve
    original_predict = Predictor.predict
    image, q, K, pose, _ = fixture()
    with instrument(context) as meter:
        tensor = torch.zeros(1, 3, 8, 8)
        backbone.network()(tensor)
        R.MASK.segment(image, 'unused', 'cpu')
        provider.predictor.set_image(tensor)
        provider.predictor.predict(tensor)
        STATE.predict_actual_visibility(None)
        STATE.spatial_features(None)
        R.observe(image, q, K, pose, components=context)
        observation, pose, K = empty_observation()
        R.solve(observation, pose, K, [120, 160])
        counts = meter.snapshot()
        for key in ('observe_calls', 'solve_calls', 'state_classifier_head_calls',
                    'state_feature_extractor_calls', 'state_backbone_forward_calls',
                    'tiny_forward_calls', 'tiny_segment_calls', 'sam_set_image_calls',
                    'sam_predict_calls', 'sam_image_encoder_forward_calls',
                    'sam_mask_decoder_forward_calls', 'bi_correction_calls', 'boundary_recovery_calls'):
            assert counts[key] == 1, (key, counts)
        for key in ('n3_forward_calls', 'subpix_calls', 'OT_builder_calls', 'U_display_builder_calls'):
            assert counts[key] == 0
        meter.reset()
        assert not any(meter.snapshot()[key] for key in counts if key.endswith('_calls'))
        provider.predictor.predict(tensor)
        assert meter.snapshot()['sam_mask_decoder_forward_calls'] == 1
    assert R.solve is original_solve and Predictor.predict is original_predict
    assert not tiny._forward_pre_hooks and not backbone.backbone._forward_pre_hooks


def test_instrumentation_blocks_N3_SubPix_and_explicit_reference_roots(tmp_path):
    from scripts.research.pallet_minimal_bi_ablation_v1.instrumentation import instrument
    reference = tmp_path / 'isolated_reference'
    before = builtins.open
    with instrument(forbidden_asset_roots=[reference]) as meter:
        with pytest.raises(RuntimeError, match='import blocked'):
            __import__('scripts.research.pallet_n3_forbidden')
        with pytest.raises(RuntimeError, match='asset access blocked'):
            (tmp_path / 'N3_SUBPIX_OUTPUT.json').open()
        with pytest.raises(RuntimeError, match='asset access blocked'):
            (reference / 'labels.json').open()
        with pytest.raises(RuntimeError, match='SubPix'):
            R.cv2.cornerSubPix(None, None, None, None, None)
        assert meter.snapshot()['blocked_imports'] == 1
        assert meter.snapshot()['blocked_asset_accesses'] == 2
        assert meter.snapshot()['subpix_calls'] == 1
        with pytest.raises(RuntimeError, match='cannot overlap'):
            with instrument():
                pass
    assert builtins.open is before
