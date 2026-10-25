"""Invented SOURCE-shaped metadata/tensors; no actual source/model assets."""
import ast
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
R = HERE.parent


def read_code(path):
    path = str(path)
    if any(x in path.casefold() for x in ('v112', '__pycache__', '.pyc')) or '..' in Path(path).parts:
        raise ValueError('Forbidden before FS')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(path).parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        child = os.open(Path(path).name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(child, 'rb') as stream:
            return stream.read()
    finally:
        os.close(fd)


def module_code(path):
    raw = read_code(path)
    ns = {'__name__': 'invented_checks_only', '__file__': str(path)}
    exec(compile(raw, str(path), 'exec'), ns)
    return ns, raw


def expect_failure(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected rejected metadata')


def run():
    torch.set_num_threads(1)
    a, code = module_code(HERE/'train_context_residual.py')
    c, _ = module_code(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py')
    b, _ = module_code(R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py')
    raw = read_code(R/'V233_DETERMINISTIC_CROP_FIX_A/train_crop_adaptation.py')
    selected = [x for x in ast.parse(raw).body if isinstance(x, ast.FunctionDef) and x.name == 'fixed_draws']
    old = {'np': np}; exec(compile(ast.Module(body=selected, type_ignores=[]), '<original-fixed-draws-only>', 'exec'), old)

    # Exact old initial base, new residual zero output, known86054 scope.
    torch.manual_seed(17); base = c['CropFamilyHead']()
    torch.manual_seed(17); head = b['RelationalQueryHead']()
    assert sum(p.numel() for p in head.parameters()) == 86054
    assert len(list(head.parameters())) == 10
    assert all(torch.equal(v, head.base.state_dict()[k]) for k, v in base.state_dict().items())

    g = 4; n = 32
    geometry = {'q8': np.stack([np.array([[10.+k, 20.+2*k] for k in range(8)], np.float64) for _ in range(g)]),
                'L8': np.full((g, 8), 50., np.float64),
                'original_available8': np.ones((g, 8), bool), 'node_available': np.ones((g, 8), bool),
                'node_fraction': np.ones((g, 8), np.float32), 'context_fraction': np.ones(g, np.float32),
                'box_xyxy': np.tile(np.array([10., 20., 17., 34.], np.float64), (g, 1))}
    a['validate_geometry'](geometry, g)
    group = np.repeat(np.arange(g, dtype=np.int64), 8); slot = np.tile(np.arange(8, dtype=np.int64), g)
    data = {'target': np.tile(np.r_[np.zeros(4, np.int8), np.ones(4, np.int8)], g),
            'family': (group//2).astype(np.int8), 'fit': group % 2 == 0,
            'available': np.ones(n, bool), 'row_id': np.array(['toy:'+str(i) for i in range(n)])}
    rows = []
    for i in range(n):
        rows.append({'row_id': str(data['row_id'][i]), 'sample_index': i, 'group_index': int(group[i]),
                     'active_slot': int(slot[i]), 'q': geometry['q8'][group[i], slot[i]].tolist(),
                     'L': 50., 'original_query_available': True, 'crop_available': True,
                     'fit': bool(data['fit'][i]), 'family_code': int(data['family'][i]), 'target': int(data['target'][i])})
    masked = dict(rows[0], row_id='toy:S_U_CONTEXT_ONLY', sample_index=None, target=-1)
    gg, ss = a['map_samples'](rows+[masked], data, geometry)
    assert np.array_equal(gg, group) and np.array_equal(ss, slot)
    bad = [dict(x) for x in rows]; bad[0]['q'] = [10.01, 20.]
    expect_failure(lambda: a['map_samples'](bad, data, geometry))
    bad = [dict(x) for x in rows]; bad[0]['fit'] = False
    expect_failure(lambda: a['map_samples'](bad, data, geometry))

    rel = a['relative_xy'](geometry, group, slot)
    assert rel.dtype == np.float32 and not rel[np.arange(n), slot].any()
    expected = np.clip((geometry['q8'][group] - geometry['q8'][group, slot][:, None])/50., -8, 8).astype(np.float32)
    expected[np.arange(n), slot] = 0
    assert np.array_equal(rel, expected)
    unavailable = {k: v.copy() for k, v in geometry.items()}
    unavailable['original_available8'][:, 7] = False
    assert not a['relative_xy'](unavailable, group, slot)[:, 7].any()

    draws, eligible = old['fixed_draws'](data)
    repeat, _ = old['fixed_draws'](data)
    assert draws.shape == (1000, 32) and np.array_equal(draws, repeat)
    assert eligible[draws].all()
    for f, offset in ((0, 0), (1, 16)):
        assert (data['family'][draws[:, offset:offset+16]] == f).all()
        assert (data['target'][draws[:, offset:offset+8]] == 0).all()
        assert (data['target'][draws[:, offset+8:offset+16]] == 1).all()

    # Full supported branch is exercised with no DINO/model weights: a parameter-free mock.
    rng = np.random.Generator(np.random.PCG64(238))
    rgb = rng.integers(0, 256, (g, 8, 3, 96, 96), dtype=np.uint8)
    valid = np.ones((g, 8, 96, 96), bool); valid[0, 3, 47, 47] = False; valid[1, 5] = False
    ctx = rng.integers(0, 256, (g, 3, 224, 224), dtype=np.uint8).astype(np.float32)
    cv = np.ones((g, 224, 224), bool); cv[1] = False
    geom = {k: v.copy() for k, v in geometry.items()}
    geom['node_fraction'] = valid.mean((2, 3)).astype(np.float32)
    geom['node_available'] = valid[:, :, 47:49, 47:49].all((2, 3)) & valid.any((2, 3))
    geom['context_fraction'] = cv.mean((1, 2)).astype(np.float32)
    class MockFrozenEncoder:
        def forward_features(self, x):
            mean = x.mean((1, 2, 3))
            descriptor = torch.arange(1, 385, dtype=torch.float32)[None] + mean[:, None]
            return {'x_norm_clstoken': descriptor}
    def preprocess(rgb, validity):
        fractions = torch.stack((validity.float().mean((1, 2, 3)), validity[:, :, 32:64, 32:64].float().mean((1, 2, 3))), 1)
        views = torch.cat((F.interpolate(rgb.float(), size=224, mode='bilinear', align_corners=False),
                           F.interpolate(rgb[:, :, 32:64, 32:64].float(), size=224, mode='bilinear', align_corners=False)), 0)
        return views, fractions
    raw_context = read_code(a['PINS']['context_preprocess_CODE'][0])
    pure = [x for x in ast.parse(raw_context).body if isinstance(x, ast.FunctionDef) and x.name == 'preprocess_target']
    ns = {'torch': torch, 'MEAN': (.485, .456, .406), 'STD': (.229, .224, .225)}
    exec(compile(ast.Module(body=pure, type_ignores=[]), '<pinned-V108-pure>', 'exec'), ns)
    assert not ns['preprocess_target'](torch.from_numpy(ctx[1:2]), torch.from_numpy(cv[1:2, None])).any()
    node, context, count = a['extract_context_features'](MockFrozenEncoder(), rgb, valid, ctx, cv, geom, preprocess, ns['preprocess_target'], 'cpu')
    assert count == int(geom['node_available'].sum())
    assert not node[~geom['node_available']].any() and not context[1].any()
    assert np.isfinite(node).all() and np.isfinite(context).all()

    # Pure random head one backward only, no optimizer step or actual training.
    x = torch.randn(32, 1186)
    nodes = F.normalize(torch.randn(32, 8, 384), dim=2)
    nodes[torch.arange(32), torch.from_numpy(slot)] = x[:, :384]
    args = (nodes, torch.randn(32, 384), torch.from_numpy(slot), torch.ones(32, 8, dtype=torch.bool),
            torch.ones(32, 8, dtype=torch.bool), torch.ones(32, dtype=torch.bool), torch.from_numpy(rel))
    qa = a['alignment'](type('Module', (), b), x, args)
    assert qa['checked'] == 32 and qa['max_abs'] == 0
    z, _ = head(x, *args)
    assert torch.equal(z, base(x))
    target = torch.from_numpy(data['target']); family = torch.from_numpy(data['family'])
    loss, parts = c['family_loss'](z, target, family)
    assert torch.equal(loss, .5*(parts[0]+parts[1]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())

    def summary(d, e):
        return {f'family{f}_DEV': {'D': {'recall': d}, 'E': {'recall': e}} for f in (0, 1)}
    assert a['source_gate'](summary(.5, .2), summary(.9, .8))['PASS']
    assert not a['source_gate'](summary(.5, .2), summary(.9, .49))['PASS']
    for p in ('/never/V112/data', '/never/__pycache__/x.py', '/never/x.pyc'):
        expect_failure(lambda p=p: a['safe'](p))

    result = {'completed': True, 'PASS': True, 'fixture_groups': 7,
              'code_sha256': hashlib.sha256(code).hexdigest(),
              'base_initial_exact_and_zero_residual': True, 'parameter_count': 86054,
              'all10_random_head_gradients_finite': True, 'random_head_backward_calls': 1,
              'optimizer_steps': 0, 'actual_training': 0,
              'original_PCG17_1000draws_shape_and_quotas': True,
              'sample_map_identity_and_role_failure_checks': True,
              'F64_relative_self_U_zero_and_context_mask_branches': True,
              'parameter_free_mock_encoder_only': True,
              'new_SOURCE_feature_RGB_NPZ_PT_GT_or_model_weight_reads': 0,
              'real_DINO_instantiations_or_forwards': 0}
    with open(HERE/'TOY_RESULT.json', 'x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
