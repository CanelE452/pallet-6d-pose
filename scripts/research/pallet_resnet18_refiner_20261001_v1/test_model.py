"""CPU-only architecture/mapping tests; no dataset, GPU or optimizer execution.

The independent local reference is the MIT-licensed Integral Human Pose
ResNetBackbone + DeconvHead, configured to the same Microsoft SimpleBaseline
ResNet18/three-deconvolution specification. This checks architecture/numerics,
not reproduction of the original top-down human training or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXTERNAL = ROOT / 'data/pallet/results/pallet_sensors_refinement_closeout_v1/external/integral-human-pose/pytorch_projects/common_pytorch/base_modules'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


M = module('_resnet18_pallet_model_under_test', HERE / 'model.py')


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().contiguous().reshape(-1).view(torch.uint8))


def expect_error(function, types=(ValueError, TypeError, RuntimeError)):
    try:
        function()
    except types:
        return
    raise AssertionError('Expected contract rejection')


def test_reference_architecture_and_output():
    B = module('_independent_integral_backbone', EXTERNAL / 'resnet.py')
    H = module('_independent_integral_heatmap_head', EXTERNAL / 'deconv_head.py')
    torch.manual_seed(19)
    tested = M.SimpleBaselineResNet18().eval()
    backbone = B.ResNetBackbone(BasicBlock, [2, 2, 2, 2]).eval()
    head = H.DeconvHead(512, 3, 256, 4, 1, 9, 1).eval()
    state = tested.state_dict()
    body_state = {k: v for k, v in state.items() if k.startswith(M.BODY_PREFIXES)}
    backbone.load_state_dict(body_state, strict=True)
    head_state = {}
    for key, value in state.items():
        if key.startswith('deconv_layers.'):
            head_state[key.replace('deconv_layers.', 'features.', 1)] = value
        elif key.startswith('final_layer.'):
            head_state[key.replace('final_layer.', 'features.9.', 1)] = value
    head.load_state_dict(head_state, strict=True)
    assert [len(getattr(tested, f'layer{i}')) for i in range(1, 5)] == [2, 2, 2, 2]
    assert all(type(block) is BasicBlock for i in range(1, 5) for block in getattr(tested, f'layer{i}'))
    convs = [layer for layer in tested.deconv_layers if isinstance(layer, nn.ConvTranspose2d)]
    assert len(convs) == 3
    for layer in convs:
        assert layer.out_channels == 256 and layer.kernel_size == (4, 4)
        assert layer.stride == (2, 2) and layer.padding == (1, 1)
        assert layer.output_padding == (0, 0) and layer.bias is None
    assert tested.final_layer.kernel_size == (1, 1) and tested.final_layer.bias is not None
    for layer in tested.modules():
        if isinstance(layer, nn.BatchNorm2d):
            assert layer.eps == 1e-5 and layer.momentum == .1
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        actual = tested(x)
        expected = head(backbone(x))
    exact(actual, expected)
    assert actual.shape == (1, 9, 16, 24)
    assert sum(p.numel() for p in tested.parameters()) == (
        sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters()))


def test_mapping_failures_are_atomic():
    torch.manual_seed(23)
    tested = M.SimpleBaselineResNet18()
    source = resnet18(weights=None).state_dict()
    before = {k: v.clone() for k, v in tested.state_dict().items()}
    variants = []
    missing = dict(source)
    missing.pop('layer4.1.bn2.running_var')
    variants.append(missing)
    extra = dict(source, unexpected=torch.zeros(1))
    variants.append(extra)
    partial_classifier = dict(source)
    partial_classifier.pop('fc.bias')
    variants.append(partial_classifier)
    partial_legacy = dict(source)
    partial_legacy.pop('bn1.num_batches_tracked')
    variants.append(partial_legacy)
    wrong_shape = dict(source, **{'conv1.weight': torch.zeros(64, 3, 3, 3)})
    variants.append(wrong_shape)
    wrong_dtype = dict(source, **{'conv1.weight': source['conv1.weight'].double()})
    variants.append(wrong_dtype)
    nonfinite = dict(source, **{'conv1.weight': torch.full_like(source['conv1.weight'], float('nan'))})
    variants.append(nonfinite)
    for state in variants:
        expect_error(lambda: tested.load_imagenet_state(state))
        assert tested.pretrained_loaded is False
        for key, value in tested.state_dict().items():
            exact(value, before[key])
    expect_error(lambda: tested.load_imagenet_state({'state_dict': source}))


def test_mapping_and_features(state=None):
    torch.manual_seed(31)
    source = resnet18(weights=None)
    supplied = source.state_dict() if state is None else state
    source.load_state_dict(supplied, strict=True)
    source.eval()
    tested = M.SimpleBaselineResNet18().eval()
    head_before = {k: v.clone() for k, v in tested.state_dict().items()
                   if not k.startswith(M.BODY_PREFIXES)}
    receipt = tested.load_imagenet_state(supplied)
    assert receipt['excluded_classifier_keys'] == ['fc.bias', 'fc.weight']
    assert tested.pretrained_loaded and receipt['missing_keys'] == receipt['unexpected_keys'] == []
    for key, value in tested.state_dict().items():
        exact(value, source.state_dict()[key] if key.startswith(M.BODY_PREFIXES) else head_before[key])
    expected_legacy = sorted(k for k in tested.state_dict()
                             if k.startswith(M.BODY_PREFIXES) and k not in supplied)
    assert receipt['legacy_bn_counters_initialized_to_zero'] == expected_legacy
    assert len(expected_legacy) in (0, 20)
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        x1 = source.layer1(source.maxpool(source.relu(source.bn1(source.conv1(x)))))
        p3 = source.layer2(x1)
        p4 = source.layer3(p3)
        actual = tested(x, return_features=True)
        direct = tested.features_only(x)
    assert set(actual) == {'heatmaps', 'features'}
    assert actual['features'][0].shape == (1, 128, 8, 12)
    assert actual['features'][1].shape == (1, 256, 4, 6)
    for a, b, c in zip(actual['features'], direct, (p3, p4)):
        exact(a, b)
        exact(a, c)
        assert not a.requires_grad and not b.requires_grad
    # Exact body-only mapping is also permitted; half a classifier is not.
    body = {k: v for k, v in supplied.items() if k.startswith(M.BODY_PREFIXES)}
    assert tested.load_imagenet_state(body)['excluded_classifier_keys'] == []


def test_baseline_gradients_and_frozen_prefix():
    torch.manual_seed(47)
    tested = M.SimpleBaselineResNet18().train()
    x = torch.randn(2, 3, 64, 64)
    output = tested(x, return_features=True)
    assert all(v.requires_grad for v in output['features'])
    # Synthetic differentiability fixture, not a fit or the production loss.
    target = torch.randn_like(output['heatmaps'])
    value = (output['heatmaps'] - target).square().mean()
    value.backward()
    for name, parameter in tested.named_parameters():
        assert parameter.requires_grad and parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in ('conv1.weight', 'layer2.0.conv1.weight', 'layer3.0.conv1.weight',
                 'layer4.0.conv1.weight', 'deconv_layers.0.weight', 'final_layer.weight'):
        assert dict(tested.named_parameters())[name].grad.norm() > 0, name
    expect_error(lambda: tested.features_only(x))
    tested.eval()
    before = {k: v.clone() for k, v in tested.named_buffers()}
    features = tested.features_only(x)
    assert all(not v.requires_grad for v in features)
    for key, value in tested.named_buffers():
        exact(value, before[key])


def test_full_resolution_shapes_without_real_forward():
    # Meta tensors check the actual operator shapes without a 384x512 CPU/GPU run.
    tested = M.SimpleBaselineResNet18().to('meta').eval()
    x = torch.empty(1, 3, *M.INPUT_SHAPE_HW, device='meta')
    with torch.no_grad():
        output = tested(x, return_features=True)
    assert output['heatmaps'].shape == (1, 9, *M.OUTPUT_SHAPE_HW)
    assert output['features'][0].shape == (1, 128, 48, 64)
    assert output['features'][1].shape == (1, 256, 24, 32)


def test_input_contract():
    tested = M.SimpleBaselineResNet18().eval()
    for shape in ((1, 1, 64, 64), (1, 3, 63, 64), (0, 3, 64, 64), (1, 3, 16, 32)):
        expect_error(lambda: tested(torch.empty(shape)))
    expect_error(lambda: tested(torch.zeros(1, 3, 64, 64, dtype=torch.uint8)))
    assert tuple(inspect.signature(M.SimpleBaselineResNet18.forward).parameters) == (
        'self', 'images', 'return_features')
    expect_error(lambda: M.SimpleBaselineResNet18(num_keypoints=0))


def run(pretrained=None):
    state = None
    binding = None
    if pretrained is not None:
        path = Path(pretrained)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
        state = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
        binding = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
    tests = [test_reference_architecture_and_output, test_mapping_failures_are_atomic,
             test_mapping_and_features, test_baseline_gradients_and_frozen_prefix,
             test_full_resolution_shapes_without_real_forward, test_input_contract]
    for test in tests:
        test(state) if test is test_mapping_and_features else test()
        print(test.__name__, 'PASS', flush=True)
    print(json.dumps(dict(PASS=True, tests=len(tests), threads=torch.get_num_threads(),
        device='cpu', tiny_actual_input_max_hw=[64, 96], requested_full_shape_checked_on='meta',
        pretrained=binding, datasets_read=0, optimizer_steps=0, GPU_used=False,
        reference='local MIT Integral ResNet18 body/DeconvHead, same official SimpleBaseline architecture')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained')
    run(parser.parse_args().pretrained)
