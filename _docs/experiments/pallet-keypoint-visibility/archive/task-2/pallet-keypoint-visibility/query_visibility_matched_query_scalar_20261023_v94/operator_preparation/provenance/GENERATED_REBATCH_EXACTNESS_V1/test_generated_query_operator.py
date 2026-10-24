"""Generated tensors only; no source artifact/checkpoint/RGB/GT reader."""
from pathlib import Path
import ast
import hashlib
import inspect
import json

import torch
from torch import nn

from query_center_scalar import QueryCenterScalar, known_binary_loss, decode_binary

ROOT = Path(__file__).resolve().parent


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def inputs(n):
    p3 = torch.randn(n, 65, 9, 9)
    p3[:, 64] = 1
    rgb = torch.randint(0, 256, (n, 3, 96, 96), dtype=torch.uint8)
    return p3, rgb, torch.ones(n, 1, 96, 96, dtype=torch.bool)


def test():
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    torch.use_deterministic_algorithms(True)
    m = QueryCenterScalar()
    x = inputs(16)
    out = m(*x)
    assert tuple(out['query_logit'].shape) == (16,)
    assert out['query_available'].all() and torch.equal(out['query_logit'], torch.zeros(16))
    assert tuple(inspect.signature(m.forward).parameters) == ('p3_roi', 'local_rgb96', 'local_valid96')
    assert not any(isinstance(t, (nn.Dropout, nn.modules.batchnorm._BatchNorm)) for t in m.modules())

    # Nonzero generated weights prevent output/gradient identity from being vacuous.
    with torch.no_grad():
        m.readout[-1].weight.normal_(0, .2)
        m.readout[-1].bias.fill_(.13)
    m.train(); train = m(*x)
    m.eval(); evaluation = m(*x)
    deployment = m(*x)
    for key in train:
        assert torch.equal(train[key], evaluation[key]) and torch.equal(train[key], deployment[key])
    assert train['query_logit'].abs().sum() > 0
    # A frame with eight queries is only batch grouping, never an identity input.
    split = torch.cat([m(*(v[i:i+8] for v in x))['query_logit'] for i in [0, 8]])
    assert torch.equal(split, train['query_logit'])

    rejects = 0
    for field in ['reference_xy', 'uncertainty', 'GT_region', 'mosaic', 'native_slot',
                  'target_id', 'source_id', 'depth', 'pose', 'semantic_label']:
        try:
            m(*x, **{field: torch.zeros(16, 2)})
        except TypeError:
            rejects += 1
        else:
            raise AssertionError('GT/metadata accepted: ' + field)
    try:
        m({'p3_roi': x[0]}, x[1], x[2])
    except TypeError:
        rejects += 1
    else:
        raise AssertionError('Dictionary accepted')

    # Masked RGB/P3 contents cannot affect output; support flags remain unchanged.
    p, r, v = inputs(2)
    p[:, 64, :2, :2] = 0
    v[:, :, :16, :16] = False
    p2 = p.clone(); p2[:, :64, :2, :2] = 12345
    r2 = r.clone(); r2[:, :, :16, :16] = 255 - r2[:, :, :16, :16]
    a = m(p, r, v); b = m(p2, r2, v)
    for key in a:
        assert torch.equal(a[key], b[key])
    assert torch.all(a['attention_diagnostic_only'][:, :2, :2] == 0)
    assert torch.allclose(a['attention_diagnostic_only'].flatten(1).sum(1), torch.ones(2), atol=1e-7, rtol=0)

    # Unknown rows and known unavailable rows contribute zero gradient, no U=>0.
    z = torch.tensor([-.4, .7, 1.2, -.8], requires_grad=True)
    available = torch.tensor([True, True, False, True])
    labels = torch.tensor([0., 1., 1., float('nan')])
    known = torch.tensor([True, True, True, False])
    lo = known_binary_loss({'query_logit': z, 'query_available': available}, labels, known)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z[:2], labels[:2])
    assert torch.equal(lo['loss'], expected) and int(lo['known_unavailable_count']) == 1
    lo['loss'].backward()
    assert torch.equal(z.grad[2:], torch.zeros(2)) and z.grad[:2].abs().min() > 0
    empty = torch.randn(3, requires_grad=True)
    zero = known_binary_loss({'query_logit': empty, 'query_available': torch.zeros(3, dtype=torch.bool)},
                             torch.tensor([-1., -1., -1.]), torch.zeros(3, dtype=torch.bool))['loss']
    assert zero == 0; zero.backward(); assert torch.equal(empty.grad, torch.zeros(3))

    # All-invalid and centre-invalid rows return factual U even with valid neighbours.
    p, r, v = inputs(2)
    p[0, 64] = 0; v[0] = False
    v[1, 0, 47, 47] = False
    unavailable = m(p, r, v)
    assert not unavailable['query_available'].any()
    assert torch.equal(unavailable['query_logit'], torch.zeros(2))
    assert torch.equal(decode_binary(unavailable), torch.tensor([-1, -1], dtype=torch.int8))
    assert unavailable['attention_diagnostic_only'][0].abs().sum() == 0

    # Attention gives a generated far-corner input a path to the SAME centre scalar.
    p, r, v = inputs(1); p.requires_grad_()
    global_out = m(p, r, v)
    global_out['query_logit'].sum().backward()
    far_grad = float(p.grad[0, :64, 0, 0].abs().sum())
    assert far_grad > 0
    grad_names = [name for name, param in m.named_parameters()
                  if param.grad is not None and param.grad.abs().sum() > 0]
    assert any(s.startswith('query.') for s in grad_names) and any(s.startswith('key.') for s in grad_names)
    assert any(s.startswith('local_encoder.') for s in grad_names)
    assert any(s.startswith('p3_encoder.') for s in grad_names)
    # Masked P3 input gradients are exactly zero.
    p, r, v = inputs(1); p[:, 64, 0, 0] = 0; p.requires_grad_()
    m(p, r, v)['query_logit'].sum().backward()
    assert torch.equal(p.grad[0, :64, 0, 0], torch.zeros(64))

    # Sign is authoritative even when FP32 sigmoid rounds tiny logits to .5.
    tiny = torch.tensor([-1e-12, 0., 1e-12, -20.])
    decoded = decode_binary({'query_logit': tiny, 'query_available': torch.tensor([True, True, True, False])})
    assert torch.equal(decoded, torch.tensor([0, 1, 1, -1], dtype=torch.int8))
    assert torch.sigmoid(tiny[0]) == .5
    model_source = ROOT / 'query_center_scalar.py'
    tree = ast.parse(model_source.read_text())
    prohibited_calls = {'grid_sample', 'region_logit', 'load', 'load_state_dict'}
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls.intersection(prohibited_calls)
    return {
        'completed': True, 'PASS': True,
        'model_code': binding(model_source), 'generated_code': binding(Path(__file__)),
        'parameter_count': sum(t.numel() for t in m.parameters()),
        'train_VAL_deployment_nonzero_scalar_and_diagnostic_identity_exact': True,
        'flat_vs_two_generated_eight_query_groups_exact': True,
        'GT_and_metadata_rejections': rejects,
        'masked_content_invariance_and_masked_P3_gradient_zero': True,
        'available_attention_mass_one_unavailable_mass_zero': True,
        'known_available_only_loss_no_unknown_negative_and_empty_graph_zero': True,
        'far_corner_P3_to_query_scalar_gradient_abs_sum': far_grad,
        'query_key_RGB_P3_gradient_paths_nonzero': True,
        'authoritative_sign_vs_rounded_sigmoid_checked': True,
        'centre_invalid_U_despite_other_support': True,
        'runtime': {'torch': torch.__version__, 'device': 'CPU', 'threads': 1, 'seed': 20261023},
        'actual_source_arrays_pixels_GT_checkpoint_parameter_reads': 0,
        'actual_forward_fit_current_PnP': 0,
        'note': 'Generated forward only. Gradient reach does not prove learnt localization or semantic identifiability.'}


if __name__ == '__main__':
    result = test()
    out = ROOT / 'GENERATED_OPERATOR_TESTS.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
