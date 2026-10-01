"""CPU-only synthetic contract tests; no images, datasets or trained weights.

Run directly with python -B test_refiner.py, or through pytest. Original P/D
modules are loaded only to construct fresh toy weights for exact parity tests.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import torch

torch.set_num_threads(1)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = module('_dope_refiner_test_subject', HERE / 'refiner.py')
P = module('_original_point_reference', ROOT / 'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py')
D = module('_original_direct_reference', ROOT / 'scripts/research/pallet_sensors_refinement_closeout_v1/direct_residual_control.py')


def exact(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().contiguous().reshape(-1).view(torch.uint8))


def fixture(c3=2, c4=3, stride3=4, stride4=8, batch=1):
    gen = torch.Generator().manual_seed(203)
    p3 = torch.randn(batch, c3, int(64 / stride3), int(64 / stride3), generator=gen)
    p4 = torch.randn(batch, c4, int(64 / stride4), int(64 / stride4), generator=gen)
    points = torch.tensor([[[18., 20.], [32., 20.], [39., 24.], [19., 27.],
                            [21., 36.], [33., 37.], [41., 40.], [17., 39.], [28., 29.]]]).repeat(batch, 1, 1)
    boxes = torch.tensor([[10., 12., 49., 47.]]).repeat(batch, 1)
    valid = torch.ones(batch, 9, dtype=torch.bool)
    shape = torch.tensor([[64., 64.]]).repeat(batch, 1)
    return [p3, p4, points, boxes, valid, shape]


def compare_outputs(old, new):
    for key in old:
        exact(old[key], new[key])


def test_original_P_exact_parity():
    # Full original channel counts/default stencil and original strides.
    torch.manual_seed(17)
    old = P.GenericPointRefiner()
    new = R.GenericPointRefiner(c3=64, c4=128, stride3=8, stride4=16)
    new.load_state_dict(old.state_dict(), strict=True)
    args = fixture(64, 128, 8, 16)
    a = old(*args, temperature=.7, lam=1.2, cap=.4)
    b = new(*args, temperature=.7, lam=1.2, cap=.4)
    compare_outputs(a, b)
    gt = args[2] + .8
    compare_outputs(P.targets(a, gt, args[4]), R.targets(b, gt, args[4]))
    la, lb = P.loss(a, gt, args[4]), R.loss(b, gt, args[4])
    exact(la, lb)
    la.backward()
    lb.backward()
    for (ka, va), (kb, vb) in zip(old.named_parameters(), new.named_parameters()):
        assert ka == kb
        exact(va.grad, vb.grad)


def test_original_D_exact_parity():
    torch.manual_seed(19)
    old = D.DirectResidualControl()
    new = R.DirectResidualControl(c3=64, c4=128, stride3=8, stride4=16)
    new.load_state_dict(old.state_dict(), strict=True)
    args = fixture(64, 128, 8, 16)
    a = old(*args, lam=1.2, cap=.4, return_descriptors=True)
    b = new(*args, lam=1.2, cap=.4, return_descriptors=True)
    compare_outputs(a, b)
    gt = args[2] + .8
    la, lb = D.direct_loss(a, gt, args[4]), R.direct_loss(b, gt, args[4])
    exact(la, lb)
    la.backward()
    lb.backward()
    for (ka, va), (kb, vb) in zip(old.named_parameters(), new.named_parameters()):
        assert ka == kb
        exact(va.grad, vb.grad)


def test_stride_bilinear_affine_and_padding():
    for stride in (4, 8, 16):
        feature = torch.arange(16.).reshape(1, 1, 4, 4)
        positions = torch.tensor([[[[[1.5 * stride, 1.5 * stride],
                                      [2. * stride, 2. * stride]]]]])
        actual = R.sample(feature, positions, torch.tensor([[4 * stride, 4 * stride]]), stride)
        torch.testing.assert_close(actual.flatten(), torch.tensor([5., 7.5]), rtol=0, atol=0)
        # Translation by one full cell in x and y for this affine ramp.
        moved = R.sample(feature, positions + stride,
                         torch.tensor([[4 * stride, 4 * stride]]), stride)
        torch.testing.assert_close(moved.flatten(), actual.flatten() + 5, rtol=0, atol=0)
        # Nonzero cached padding (including possible adapter bias) is masked.
        padded = torch.ones(1, 1, 4, 4) * 99
        point = torch.tensor([[[[[3.5 * stride, 3.5 * stride]]]]])
        zero = R.sample(padded, point, torch.tensor([[2 * stride, 2 * stride]]), stride)
        assert zero.item() == 0


def test_uniform_coordinate_scaling():
    args = fixture()
    for cls in (R.GenericPointRefiner, R.DirectResidualControl):
        torch.manual_seed(25)
        first = cls(c3=2, c4=3)
        scaled = cls(c3=2, c4=3, stride3=8, stride4=16)
        scaled.load_state_dict(first.state_dict())
        doubled = [args[0], args[1], args[2] * 2, args[3] * 2, args[4], args[5] * 2]
        a, b = first(*args, cap=.3), scaled(*doubled, cap=.6)
        key = 'logits' if cls is R.GenericPointRefiner else 'delta_normalized'
        exact(a[key], b[key])
        exact(a['points'] * 2, b['points'])
        # Original-pixel displacement is input displacement divided by gain.
        exact(a['points'] - args[2], (b['points'] - doubled[2]) / 2)


def test_same_P_D_evidence():
    torch.manual_seed(44)
    p = R.GenericPointRefiner(c3=2, c4=3)
    d = R.DirectResidualControl(c3=2, c4=3)
    state = d.state_dict()
    for key, value in p.state_dict().items():
        if key in state:
            state[key] = value
    d.load_state_dict(state)
    args = fixture()
    p_base, p_descriptors, _, _ = p._evidence(*args)
    d_out = d(*args, return_descriptors=True)
    exact(p_descriptors, d_out['candidate_descriptors'])
    for key, value in p_base.items():
        exact(value, d_out[key])


def test_missing_center_noop_and_cap():
    for cls, loss_fn in ((R.GenericPointRefiner, R.loss), (R.DirectResidualControl, R.direct_loss)):
        model = cls(c3=2, c4=3)
        args = fixture(batch=2)
        args[2][0, 0] = -1
        args[2][0, 1] = float('nan')
        args[4][0, 2] = False
        args[3][1] = float('nan')
        out = model(*args, lam=3, cap=.2)
        exact(out['points'][0, :3], args[2][0, :3])
        exact(out['points'][1], args[2][1])
        exact(out['points'][:, 8], args[2][:, 8])
        exact(model(*args, lam=0)['points'], args[2])
        exact(model(*args, cap=0)['points'], args[2])
        moved = (out['points'] - args[2])[:, 3:8].norm(dim=-1)
        assert moved.max() <= .20001
        assert torch.isfinite(loss_fn(out, args[2] + .1, args[4]))
        empty = loss_fn(out, args[2], torch.zeros_like(args[4]))
        assert empty.item() == 0
        model.zero_grad(set_to_none=True)
        empty.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all()
                   and v.grad.count_nonzero() == 0 for v in model.parameters())


def test_lattice_and_target_loss_reduction():
    model = R.GenericPointRefiner(c3=2, c4=3)
    args = fixture(batch=2)
    out = model(*args)
    assert out['logits'].shape == (2, 8, 222)
    assert model.displacements.shape == (222, 2)
    assert (model.displacements.norm(dim=-1) == 0).sum() == 1
    assert len(torch.unique(model.displacements[:-1], dim=0)) == 221
    torch.testing.assert_close(model.displacements.norm(dim=-1).max(), torch.tensor(.08))
    gt = args[2].clone()
    gt[:, :8] += out['candidate_displacements'][:, 14, None]
    valid = args[4].clone()
    valid[1, :8] = False
    target = R.targets(out, gt, valid)
    assert (target['distribution'][0].argmax(-1) == 14).all()
    torch.testing.assert_close(target['distribution'].sum(-1), torch.ones(2, 8))
    manual = -(target['distribution'][0] * out['logits'][0].log_softmax(-1)).sum(-1).mean()
    exact(R.loss(out, gt, valid), manual)
    d = R.DirectResidualControl(c3=2, c4=3)
    dout = d(*args)
    dmanual = (dout['delta_normalized'][0]
               - (gt[0, :8] - args[2][0, :8]) / dout['box_diagonal'][0]).abs().mean()
    torch.testing.assert_close(R.direct_loss(dout, gt, valid), dmanual, rtol=0, atol=1e-9)


def test_DOPE_channels_gradients_and_no_reference_inputs():
    args = fixture(256, 128)
    for cls, loss_fn in ((R.GenericPointRefiner, R.loss), (R.DirectResidualControl, R.direct_loss)):
        torch.manual_seed(71)
        model = cls()
        assert model.c3 == 256 and model.c4 == 128 and model.stride3 == 4 and model.stride4 == 8
        args_before = [x.clone() for x in args]
        out = model(*args)
        value = loss_fn(out, args[2] + .8, args[4])
        value.backward()
        assert torch.isfinite(value)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.adapt3[0].weight.grad.norm() > 0
        assert model.adapt4[0].weight.grad.norm() > 0
        assert model.patch_body[0].weight.grad.norm() > 0
        for before, after in zip(args_before, args):
            exact(before, after)
        assert all('gt' not in n.lower() and 'target' not in n.lower()
                   for n in inspect.signature(cls.forward).parameters)


def test_invalid_options():
    args = fixture()
    for cls in (R.GenericPointRefiner, R.DirectResidualControl):
        model = cls(c3=2, c4=3)
        for kwargs in ({'lam': -1}, {'cap': float('nan')}, {'cap': -1}, {'cap': [1, 2]}):
            try:
                model(*args, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(kwargs)
    direct = R.DirectResidualControl(c3=2, c4=3)
    try:
        direct(*args, temperature=2)
    except ValueError:
        pass
    else:
        raise AssertionError('D must not invent a temperature operation')


def run():
    names = [name for name in globals() if name.startswith('test_')]
    for name in names:
        globals()[name]()
        print(name, 'PASS', flush=True)
    print(json.dumps(dict(PASS=True, tests=len(names), device='cpu', threads=torch.get_num_threads(),
                          synthetic_only=True, actual_trained_weights_read=0,
                          datasets_read=0, optimizer_steps=0, GPU_used=False)), flush=True)


if __name__ == '__main__':
    run()
