"""Strict trained ResNet18 adapter for frozen P/D feature extraction.

Only the completed, fixed epoch60 checkpoint is accepted. There is no ImageNet,
random-weight or missing-file fallback and no model download. The CPU selfcheck
uses an explicitly untrained in-memory fixture solely to check shapes, feature
equivalence and normal tensor/autograd behavior; it writes no checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
RAW = ROOT / 'data/pallet/results' / HERE.name
CHECKPOINT = RAW / 'baseline_final.pt'
FEATURE_CHANNELS = (128, 256)
FEATURE_STRIDES = (8, 16)
FEATURE_TAPS = ('layer2', 'layer3')


def local_module(name, filename):
    """Avoid collisions with the independent DOPE namespace's module names."""
    key = 'pallet_resnet18_adapter_' + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


I = local_module('input_data', 'input_data.py')
prepare = I.prepare


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_binding(binding):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = ROOT / path
    assert file_sha(path) == binding['sha256'], ('Changed bound file', path)
    if 'bytes' in binding:
        assert path.stat().st_size == binding['bytes'], path
    return path


def recipe():
    return dict(schema='resnet18_trained_refiner_adapter_v1',
        baseline='Fixed final epoch60 source-trained SimpleBaselineResNet18 pallet9',
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), input=I.contract(),
        feature_taps=list(FEATURE_TAPS), feature_channels=list(FEATURE_CHANNELS),
        feature_strides=list(FEATURE_STRIDES), feature_dtype='float32',
        input_shape=[3,384,512], native_shape_feature_padding=False,
        feature_sampling='Existing P cell-center grid_sample convention; no exact receptive-field-center claim',
        checkpoint_fallback=False, model_download=False, GT_inputs=False,
        frozen=True, TF32=False, cudnn_benchmark=False, AMP=False,
        points_net_dtype='float32', points_original_dtype='float64',
        refinement_inverse='Preserve FP64 baseline original points and add net delta divided by exact per-axis affine scale; center8/box/mask unchanged')


class FrozenResnetAdapter:
    def __init__(self, device='cpu', checkpoint=CHECKPOINT):
        checkpoint = Path(checkpoint)
        receipt_path = DOC / 'BASELINE_TRAINING_COMPLETE.json'
        protocol_path = DOC / 'BASELINE_PROTOCOL.json'
        receipt = read(receipt_path)
        assert receipt['complete'] is True and receipt['final_checkpoint_only'] is True
        assert receipt['epochs'] == 60 and receipt['seed'] == 42 and receipt['real_training'] == 0
        assert receipt['train_images'] == 55980 and receipt['calibration_images'] == 1004
        expected_steps = math.ceil(55980 / 16) * 60
        assert receipt['updates'] == expected_steps and receipt['source_exposures'] == 55980*60
        assert verify_binding(receipt['protocol']).resolve() == protocol_path.resolve()
        bound_checkpoint = verify_binding(receipt['final_checkpoint'])
        assert checkpoint.resolve() == bound_checkpoint.resolve() == CHECKPOINT.resolve()
        protocol = read(protocol_path)
        assert protocol['schema'] == 'resnet18_full_image_pallet9_baseline_protocol_v1'
        assert protocol['epochs'] == 60 and protocol['seed'] == 42 and protocol['effective_batch'] == 16
        assert protocol['new_real_supervision'] == 0 and not protocol['YOLO_dependency'] and not protocol['DOPE_dependency']
        assert protocol['input']['height'] == 384 and protocol['input']['width'] == 512
        assert protocol['targets']['channels'] == 9 and protocol['targets']['stride'] == 4
        assert protocol['targets']['sigma'] == I.SIGMA and protocol['decoding']['peak_threshold'] == I.CONFIDENCE_THRESHOLD
        assert protocol['precision'] == dict(model='FP32', AMP=False, TF32=False, cudnn_benchmark=False)
        for binding in protocol['bindings']:
            verify_binding(binding)
        # Current input/model bytes must be present in the original training lock.
        for filename in ('input_data.py', 'model.py'):
            found = [binding for binding in protocol['bindings']
                     if (ROOT / binding['path']).resolve() == (HERE / filename).resolve()]
            assert len(found) == 1 and found[0]['sha256'] == file_sha(HERE / filename)
        assert receipt['source'] == protocol['source']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert payload['architecture'] == 'SimpleBaselineResNet18' and payload['num_keypoints'] == 9
        assert payload['epoch'] == 60 and payload['step'] == expected_steps
        assert payload['protocol_sha256'] == file_sha(protocol_path) == receipt['protocol']['sha256']
        state = payload['model_state_dict']
        M = local_module('model', 'model.py')
        self.network = M.SimpleBaselineResNet18(num_keypoints=9, pretrained_state=None)
        expected = self.network.state_dict()
        assert set(state) == set(expected), 'Full body and trained heatmap head are required'
        for key, value in state.items():
            assert isinstance(value, torch.Tensor) and value.shape == expected[key].shape and value.dtype == expected[key].dtype, key
            assert not value.is_meta and (not value.is_floating_point() or torch.isfinite(value).all()), key
        self.network.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.network.requires_grad_(False).eval().to(self.device)
        self.checkpoint_sha256 = receipt['final_checkpoint']['sha256']
        self.baseline_protocol_sha256 = receipt['protocol']['sha256']
        self.training_receipt_sha256 = file_sha(receipt_path)
        self.trained_baseline_loaded = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    prepare = staticmethod(prepare)

    def _check_runtime(self, tensor):
        assert tensor.ndim == 4 and tuple(tensor.shape[1:]) == (3,384,512)
        assert tensor.shape[0] > 0 and tensor.dtype == torch.float32
        assert not any(module.training for module in self.network.modules())
        assert not any(parameter.requires_grad for parameter in self.network.parameters())
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark

    def _check_features(self, features, n):
        assert len(features) == 2
        for value, channels, stride in zip(features, FEATURE_CHANNELS, FEATURE_STRIDES):
            assert value.shape == (n, channels, 384//stride, 512//stride)
            assert value.dtype == torch.float32 and not value.requires_grad
            assert not torch.is_inference(value), 'Trainable heads require normal frozen-feature tensors'

    @torch.no_grad()
    def features_only(self, batch_tensor):
        """Frozen layer2/layer3 prefix only, returning normal detached tensors."""
        self._check_runtime(batch_tensor)
        # Explicitly leave a surrounding inference_mode, if a caller used one.
        # This permits a trainable P/D head to save these tensors for backward.
        with torch.inference_mode(False), torch.no_grad():
            value = batch_tensor.to(self.device)
            if torch.is_inference(value):
                value = value.clone()
            features = tuple(feature.detach() for feature in self.network.features_only(value))
        self._check_features(features, len(batch_tensor))
        return features

    @torch.no_grad()
    def infer_batch(self, images, source_pre_padded=False, return_features=True):
        """Mixed native image sizes are legal: each is independently letterboxed."""
        assert len(images) > 0
        with torch.inference_mode(False), torch.no_grad():
            prepared = [prepare(image, source_pre_padded) for image in images]
            tensor = torch.stack([item['tensor'] for item in prepared]).to(self.device)
            self._check_runtime(tensor)
            result = self.network(tensor, return_features=return_features)
            if return_features:
                assert set(result) == {'heatmaps', 'features'}
                heatmaps = result['heatmaps']; features = tuple(value.detach() for value in result['features'])
                self._check_features(features, len(images))
            else:
                heatmaps = result; features = ()
            assert heatmaps.shape == (len(images),9,96,128) and heatmaps.dtype == torch.float32
            maps = heatmaps.detach().cpu().numpy()
        outputs = []
        for j, meta in enumerate(prepared):
            decoded = I.decode(maps[j], meta)
            decoded['points_net'] = decoded['points_net'].astype(np.float32)
            assert decoded['points_original'].dtype == np.float64
            decoded.update({key:value for key,value in meta.items() if key != 'tensor'})
            decoded.update(checkpoint_sha256=self.checkpoint_sha256,
                baseline_protocol_sha256=self.baseline_protocol_sha256,
                training_receipt_sha256=self.training_receipt_sha256,
                n_detected_corners=int(decoded['valid'][:8].sum()),
                status='OK' if decoded['bbox_original'] is not None else 'NO_BOX',
                belief_shape=[9,96,128])
            if return_features:
                decoded['features'] = tuple(feature[j] for feature in features)
            outputs.append(decoded)
        return outputs

    def infer(self, image, source_pre_padded=False, return_features=True):
        return self.infer_batch([image], source_pre_padded, return_features)[0]


def selfcheck():
    """No trained artifacts, actual images or GPU: explicit CPU shape fixture."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20261001)
    M = local_module('model', 'model.py')
    fixture = FrozenResnetAdapter.__new__(FrozenResnetAdapter)
    fixture.network = M.SimpleBaselineResNet18(pretrained_state=None).requires_grad_(False).eval()
    fixture.device = torch.device('cpu')
    fixture.checkpoint_sha256 = 'UNTRAINED_CPU_FIXTURE_NOT_A_CHECKPOINT'
    fixture.baseline_protocol_sha256 = 'NO_PROTOCOL_READ'
    fixture.training_receipt_sha256 = 'NO_TRAINING_RECEIPT_READ'
    fixture.trained_baseline_loaded = False
    raw = np.zeros((61,83,3), np.uint8)
    prepared = prepare(raw)
    x = prepared['tensor'][None]
    with torch.no_grad():
        full = fixture.network(x, return_features=True)
    prefix = fixture.features_only(x)
    assert full['heatmaps'].shape == (1,9,96,128)
    for expected, actual in zip(full['features'], prefix):
        assert torch.equal(expected, actual)
    # A normal feature tensor can be retained for a learned head's backward.
    head = torch.nn.Conv2d(128,1,1)
    head(prefix[0]).mean().backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert all(parameter.grad is None for parameter in fixture.network.parameters())
    with torch.inference_mode():
        inherited = fixture.features_only(x)
    assert all(not torch.is_inference(value) for value in inherited)
    # Check wrapper shapes/metadata only, never treat random outputs as accuracy.
    result = fixture.infer(raw, return_features=True)
    assert result['points_net'].shape == (9,2) and result['points_net'].dtype == np.float32
    assert result['points_original'].shape == (9,2) and result['points_original'].dtype == np.float64
    assert result['valid'].shape == result['confidence'].shape == (9,)
    assert result['input_shape'] == [3,384,512]
    assert np.array_equal(np.isfinite(result['points_original']).all(-1), result['valid'])
    result_no_features = fixture.infer(raw, return_features=False)
    assert 'features' not in result_no_features
    assert np.array_equal(result_no_features['points_original'], result['points_original'], equal_nan=True)
    print(json.dumps(dict(PASS=True, synthetic_untrained_CPU_fixture=True,
        actual_images_read=0, trained_checkpoint_reads=0, GPU_calls=0,
        heatmap_shape=[1,9,96,128], feature_shapes=[list(value.shape) for value in prefix],
        full_prefix_exact=True, normal_tensor_head_backward=True,
        constructor_bypassed_only_for_explicit_selfcheck=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['selfcheck'])
    parser.parse_args()
    selfcheck()
