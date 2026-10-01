"""Frozen DOPE input/feature adapter; no labels, PnP, model downloads or imputation.

Input coordinates mean the supplied image: prepared padded canvas for synthetic
source, original unpadded pixels for real RGB. The size-only affine deliberately
preserves the historical decoder's +0.4395 convention; it does not introduce a
half-pixel correction or claim exact receptive-field alignment.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / 'weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth'
CHECKPOINT_SHA = '0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f'
MODEL_SOURCE = ROOT / 'Deep_Object_Pose/common/models.py'
DECODER_SOURCE = ROOT / 'scripts/data_prep/filters/filter_pr_camfacing.py'
PREPROCESS_SOURCE = ROOT / 'scripts/stage0/eval_harness/eval_pvnet_heads.py'
CANVAS_SOURCE = ROOT / 'data/pallet/eval_results/stage16_truncation_addon/capturecad_b2_eval/eval_capturecad_b2.py'
PAD = 100
SHORTEST_SIDE = 400
THRESHOLD = .3
FEATURE_CHANNELS = (256, 128)
FEATURE_STRIDES = (4, 8)
FEATURE_TAPS = (17, 26)
# Preserve canonical FP32 pixels -> FP64 normalize -> final FP32 cast exactly.
MEAN = np.array([.485, .456, .406])
STD = np.array([.229, .224, .225])
PREFLIGHT_CORRECTIONS = [
    'Before any model forward, CPU strict load succeeded but prototype tap16 ReLU assertion failed: VGG19 index16 is Conv2d, index17 is post-ReLU. Root approved17/26 before protocol sealing.',
    'Before any model forward, static review restored canonical FP64 mean/std followed by final FP32 cast; no data-dependent tolerance change.'
]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def recipe():
    return dict(schema='pallet_dope_refiner_adapter_v1', checkpoint_sha256=CHECKPOINT_SHA,
        source_input='Already reflect100-padded image; no additional border; coordinates remain prepared pixels.',
        real_input='Native image; BORDER_REFLECT_101100; returned original coordinates exclude this padding.',
        preprocess='Prepared canvas -> resize native WxH -> short-side400 actual dimensions floored to8 -> ImageNet FP32.',
        coordinate_inverse='Actual per-axis canvas/network size inverse; no nominal-scale or extra half-pixel correction.',
        feature_sampling='Network coordinates divide by stride4/8 for existing P grid_sample conventions; no exact physical receptive-field-center claim.',
        belief_threshold=THRESHOLD, decoder='Canonical Gaussian sigma2/local peak/radius5/offset0.4395 per semantic channel.',
        feature_taps=['vgg.17:ReLU_conv3_4','vgg.26:ReLU_final128'],
        feature_channels=list(FEATURE_CHANNELS), feature_strides=list(FEATURE_STRIDES),
        feature_dtype='float32', corner_mapping='Unchanged channels0..7 and centroid8; no permutation.',
        box_source='Bounding box of finite predicted corners only; minimum3 corners and both extents>1 original-input pixel.',
        score_source='Maximum belief peak over8corners; not calibrated YOLO box confidence.',
        missing_policy='NaN points and false valid preserved; absent box is None; no missing-point completion.',
        TF32=False, half=False, cudnn_benchmark=False, frozen=True,
        affinity_instance_grouping=False, new_PnP_solves=0, GT_inputs=False)


def prepare(image, source_pre_padded=False):
    """Return CPU FP32 tensor and exact size-only input->network affine."""
    image = np.asarray(image)
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    h, w = image.shape[:2]
    if source_pre_padded:
        assert min(h, w) > 2 * PAD, 'Prepared source must include its100px border.'
        native_h, native_w = h - 2 * PAD, w - 2 * PAD
        canvas = image
        shift = 0
    else:
        native_h, native_w = h, w
        canvas = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        shift = PAD
    ch, cw = canvas.shape[:2]
    # Keep both interpolation stages, including the intermediate uint8 rounding.
    resized_back = cv2.resize(canvas, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized_back, cv2.COLOR_BGR2RGB)
    sc = SHORTEST_SIDE / min(native_h, native_w)
    nw = max(8, int(round(native_w * sc)) & ~7)
    nh = max(8, int(round(native_h * sc)) & ~7)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255. - MEAN) / STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()
    affine = np.array([[nw / cw, 0., shift * nw / cw],
                       [0., nh / ch, shift * nh / ch], [0., 0., 1.]], np.float64)
    return dict(tensor=tensor, affine_input_to_net=affine,
        affine_net_to_input=np.linalg.inv(affine), input_shape=list(tensor.shape),
        input_hw=[h, w], native_hw=[native_h, native_w], canvas_hw=[ch, cw],
        source_pre_padded=bool(source_pre_padded), additional_padding=shift,
        existing_source_padding=PAD if source_pre_padded else 0)


def transform_points(points, affine):
    points = np.asarray(points, np.float64)
    affine = np.asarray(affine, np.float64)
    assert points.ndim == 2 and points.shape[1] == 2 and affine.shape == (3, 3)
    out = np.full(points.shape, np.nan, np.float64)
    finite = np.isfinite(points).all(1)
    hp = np.c_[points[finite], np.ones(int(finite.sum()))] @ affine.T
    assert np.all(hp[:, 2] != 0)
    out[finite] = hp[:, :2] / hp[:, 2, None]
    return out


def predicted_box(points, valid):
    points = np.asarray(points, np.float64)
    valid = np.asarray(valid, bool)
    assert points.shape == (9, 2) and valid.shape == (9,)
    assert np.isfinite(points).all(1).tolist() == valid.tolist()
    corners = points[:8][valid[:8]]
    if len(corners) < 3:
        return None
    box = np.r_[corners.min(0), corners.max(0)]
    return box if np.all(box[2:] - box[:2] > 1.) else None


def extract_keypoints_from_belief(belief_maps, threshold=0.3):
    # Function body copied verbatim from DECODER_SOURCE; selfcheck compares AST.
    OFFSET, RAN = 0.4395, 5
    keypoints = []
    for i in range(belief_maps.shape[0]):
        bmap = belief_maps[i]
        if bmap.max() < threshold:
            keypoints.append((-1, -1, float(bmap.max())))
            continue
        sm = gaussian_filter(bmap, sigma=2)
        p = 1
        pl = np.zeros_like(sm); pl[p:, :] = sm[:-p, :]
        pr = np.zeros_like(sm); pr[:-p, :] = sm[p:, :]
        pu = np.zeros_like(sm); pu[:, p:] = sm[:, :-p]
        pd = np.zeros_like(sm); pd[:, :-p] = sm[:, p:]
        peaks = ((sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) &
                 (sm > threshold))
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            keypoints.append((-1, -1, float(bmap.max())))
            continue
        best = int(np.argmax([bmap[y, x] for y, x in zip(ys, xs)]))
        px, py = int(xs[best]), int(ys[best])
        y0, y1 = max(0, py - RAN), min(bmap.shape[0], py + RAN + 1)
        x0, x1 = max(0, px - RAN), min(bmap.shape[1], px + RAN + 1)
        patch = bmap[y0:y1, x0:x1]
        if patch.sum() > 0:
            xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            wx = float(np.average(xg, weights=patch)) + OFFSET
            wy = float(np.average(yg, weights=patch)) + OFFSET
        else:
            wx, wy = float(px), float(py)
        keypoints.append((wx, wy, float(bmap.max())))
    return keypoints


def decode(belief, prepared):
    belief = np.asarray(belief, np.float32)
    assert belief.ndim == 3 and belief.shape[0] == 9 and np.isfinite(belief).all()
    _, nh, nw = prepared['input_shape']
    bh, bw = belief.shape[1:]
    assert (bh * 8, bw * 8) == (nh, nw)
    peaks = extract_keypoints_from_belief(belief, THRESHOLD)
    net64 = np.full((9, 2), np.nan, np.float64)
    valid = np.zeros(9, bool)
    for i, (x, y, _) in enumerate(peaks):
        if x >= 0 and np.isfinite([x, y]).all():
            net64[i] = [x * nw / bw, y * nh / bh]
            valid[i] = True
    original = transform_points(net64, prepared['affine_net_to_input'])
    box = predicted_box(original, valid)
    box_net = None if box is None else transform_points(box.reshape(2, 2), prepared['affine_input_to_net']).reshape(4)
    confidence = np.asarray([p[2] for p in peaks], np.float32)
    return dict(points_net=net64.astype(np.float32), points_original=original,
        valid=valid, bbox_net=box_net, bbox_original=box, confidence=confidence,
        score=float(confidence[:8].max()), n_detected_corners=int(valid[:8].sum()),
        status='OK' if box is not None else 'NO_BOX', belief_shape=list(belief.shape))


class FrozenDopeAdapter:
    """Construct once; infer all stages, or run only frozen VGG during P training."""
    def __init__(self, device='cpu', checkpoint=CHECKPOINT):
        checkpoint = Path(checkpoint)
        assert file_sha(checkpoint) == CHECKPOINT_SHA, 'Frozen DOPE checkpoint mismatch'
        spec = importlib.util.spec_from_file_location('pallet_dope_refiner_frozen_model', MODEL_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network = module.DopeNetwork(pretrained=False, stop_at_stage=6)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        assert set(k.split('.')[0] for k in state) == {'vgg'} | {f'm{i}_{j}' for i in range(1, 7) for j in (1, 2)}
        self.network.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.network.requires_grad_(False).eval().to(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        assert isinstance(self.network.vgg[17], torch.nn.ReLU)
        assert isinstance(self.network.vgg[26], torch.nn.ReLU)

    prepare = staticmethod(prepare)

    def _check_runtime(self, tensor):
        assert tensor.ndim == 4 and tensor.shape[1] == 3 and tensor.dtype == torch.float32
        assert tensor.shape[2] % 8 == tensor.shape[3] % 8 == 0
        assert not self.network.training and not any(p.requires_grad for p in self.network.parameters())
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark

    @torch.no_grad()
    def features_only(self, batch_tensor):
        """Return normal detached tensors, usable by trainable head autograd."""
        self._check_runtime(batch_tensor)
        value = batch_tensor.to(self.device)
        output = []
        for i, layer in enumerate(self.network.vgg):
            value = layer(value)
            if i in FEATURE_TAPS:
                output.append(value)
        for feature, channels, stride in zip(output, FEATURE_CHANNELS, FEATURE_STRIDES):
            assert feature.shape == (len(batch_tensor), channels, batch_tensor.shape[2] // stride, batch_tensor.shape[3] // stride)
            assert feature.dtype == torch.float32 and not feature.requires_grad
        return tuple(output)

    @torch.no_grad()
    def infer_batch(self, images, source_pre_padded=False, return_features=True):
        """Identical-shape batches only; no image zero-padding or hidden buckets."""
        assert len(images) > 0
        prepared_rows = [prepare(image, source_pre_padded) for image in images]
        assert len({tuple(p['input_shape']) for p in prepared_rows}) == 1, 'Bucket identical network shapes before infer_batch.'
        tensor = torch.stack([p['tensor'] for p in prepared_rows]).to(self.device)
        self._check_runtime(tensor)
        captured = {}
        handles = [self.network.vgg[index].register_forward_hook(
            lambda _module, _inputs, output, key=index: captured.__setitem__(key, output))
            for index in FEATURE_TAPS] if return_features else []
        try:
            beliefs, affinities = self.network(tensor)
        finally:
            for handle in handles:
                handle.remove()
        assert len(beliefs) == len(affinities) == 6
        features = tuple(captured[index] for index in FEATURE_TAPS) if return_features else ()
        for feature, channels, stride in zip(features, FEATURE_CHANNELS, FEATURE_STRIDES):
            assert feature.shape == (len(images), channels, tensor.shape[2] // stride, tensor.shape[3] // stride)
        belief_cpu = beliefs[-1].detach().cpu().numpy()
        outputs = []
        for i, prepared in enumerate(prepared_rows):
            result = decode(belief_cpu[i], prepared)
            result.update({k: v for k, v in prepared.items() if k != 'tensor'})
            if return_features:
                result['features'] = tuple(feature[i] for feature in features)
            result['checkpoint_sha256'] = CHECKPOINT_SHA
            outputs.append(result)
        return outputs

    def infer(self, image, source_pre_padded=False, return_features=True):
        return self.infer_batch([image], source_pre_padded, return_features)[0]


def selfcheck():
    # No checkpoint load, network construction, GPU/model forward or real image.
    def body(path):
        tree = ast.parse(Path(path).read_text())
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_keypoints_from_belief').body
    assert [ast.dump(n) for n in body(__file__)] == [ast.dump(n) for n in body(DECODER_SOURCE)]
    image = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    real = prepare(image)
    canvas = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    source = prepare(canvas, True)
    torch.testing.assert_close(real['tensor'], source['tensor'], rtol=0, atol=0)
    assert real['input_shape'] == [3, 400, 528] and source['additional_padding'] == 0
    p = np.array([[0., 0.], [639., 479.], [317.25, 121.75], [np.nan, np.nan]])
    net = transform_points(p, real['affine_input_to_net'])
    np.testing.assert_allclose(transform_points(net, real['affine_net_to_input']), p, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(transform_points(p + PAD, source['affine_input_to_net']), net, rtol=0, atol=1e-12, equal_nan=True)
    # Two resizes and normalization are independently rebuilt.
    manual = cv2.resize(canvas, (640, 480), interpolation=cv2.INTER_LINEAR)
    manual = cv2.resize(cv2.cvtColor(manual, cv2.COLOR_BGR2RGB), (528, 400), interpolation=cv2.INTER_LINEAR)
    manual = ((manual.astype(np.float32)/255.-MEAN)/STD).transpose(2, 0, 1)
    np.testing.assert_array_equal(real['tensor'].numpy(), manual.astype(np.float32))
    # Execute only the actual historical pure functions/constants, not their
    # module import trees (which contain unrelated evaluation utilities).
    prep_ast = ast.parse(PREPROCESS_SOURCE.read_text())
    nodes = [n for n in prep_ast.body if
        (isinstance(n, ast.FunctionDef) and n.name == 'preprocess') or
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('MEAN', 'STD') for t in n.targets))]
    canvas_ast = ast.parse(CANVAS_SOURCE.read_text())
    nodes += [n for n in canvas_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'pad_frame']
    historical = {'np': np, 'torch': torch, 'cv2': cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical_pure_preprocessing>', 'exec'), historical)
    original_tensor, nw, nh, _ = historical['preprocess'](historical['pad_frame'](image, PAD))
    torch.testing.assert_close(real['tensor'], original_tensor[0], rtol=0, atol=0)
    assert (nw, nh) == (528, 400)
    for h, w in ((680,840),(740,1160),(680,920),(760,760)):
        source_image = np.zeros((h, w, 3), np.uint8)
        p = prepare(source_image, True)
        reference = historical['preprocess'](cv2.resize(source_image, (w-200,h-200), interpolation=cv2.INTER_LINEAR))[0]
        torch.testing.assert_close(p['tensor'], reference[0], rtol=0, atol=0)
        assert p['tensor'].dtype == torch.float32
    belief = np.zeros((9, 50, 66), np.float32)
    empty = decode(belief, real)
    assert not empty['valid'].any() and empty['bbox_original'] is None and np.isnan(empty['points_net']).all()
    yy, xx = np.mgrid[:50, :66]
    for j, i in enumerate((0, 1, 2, 8)):
        belief[i] = np.exp(-((xx - (15+7*j))**2 + (yy-(12+3*j))**2)/18.)
    result = decode(belief, real)
    assert result['valid'].tolist() == [True, True, True, False, False, False, False, False, True]
    assert result['bbox_original'] is not None and result['n_detected_corners'] == 3
    assert result['points_net'].dtype == np.float32 and np.isnan(result['points_original'][3:8]).all()
    print('DOPE_ADAPTER_SELFCHECK_PASS CPU synthetic coordinates/decoder/historical_preprocess_byte_exact; model_forwards=0', flush=True)


if __name__ == '__main__':
    selfcheck()
