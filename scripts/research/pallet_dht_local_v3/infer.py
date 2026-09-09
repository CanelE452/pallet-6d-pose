"""Raw BGR -> original frozen joint backbone -> calibrated local refinement.

No GT, cached predictions, or per-image scale selection. This bounded local
offset module preserves semantic IDs and cannot repair C4 correspondence.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from . import cache as C
from . import train as T
from .real_evaluation import apply_scale, load_selection, plain
from scripts.research.pallet_line_pose_v1 import source_data as SD
from scripts.research.pallet_dht_decoder_probe_v1 import cache as OLD
from scripts.research.pallet_dht_structured_v2 import infer as COMMON


def image_inputs(original_bgr, candidates, captured_input_hw, device='cpu'):
    """Canonical gray/input affine without image files, labels or proposals."""
    import cv2
    if original_bgr is None or original_bgr.ndim != 3 or original_bgr.shape[2] != 3 or original_bgr.dtype != np.uint8:
        raise ValueError('Original uint8 HxWx3 BGR required')
    h, w = original_bgr.shape[:2]
    reflected = cv2.copyMakeBorder(original_bgr, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
    chw, tf = SD.prepare_image(reflected, imgsz=640, auto=True, stride=32)
    if tuple(tf.input_shape_hw) != tuple(map(int, captured_input_hw)):
        raise ValueError('Preprocessor shape differs from actual frozen backbone hook')
    gray = cv2.cvtColor(np.rint(chw.transpose(1, 2, 0)*255.).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    canvas = np.zeros((1, 640, 640), np.uint8)
    canvas[0, :gray.shape[0], :gray.shape[1]] = gray
    selected = int(np.argmax([c['score'] for c in candidates])) if candidates else None
    item = candidates[selected] if selected is not None else None
    points = np.array(item['keypoints_xy'], np.float32) if item and item['keypoints_xy'] is not None else np.zeros((9, 2), np.float32)
    valid = np.isfinite(points).all(-1) if item and item['keypoints_xy'] is not None else np.zeros(9, bool)
    conf = np.array(item['keypoints_conf'], np.float32) if item and item['keypoints_conf'] is not None else np.zeros(9, np.float32)
    if points.shape != (9, 2) or conf.shape != (9,) or not np.isfinite(conf).all():
        raise ValueError('Canonical nine-point output required')
    points[~valid] = 0
    baseline = dict(points=points.tolist(), point_valid=valid.tolist(), point_conf=conf.tolist(),
        box_xyxy=item['box_xyxy'] if item else [0., 0., 0., 0.],
        score=item['score'] if item else 0., detected=item is not None,
        selected_instance=selected, all_candidates=copy.deepcopy(candidates))
    affine = np.array([[tf.gain, 0., 100*tf.gain+tf.pad_ltrb[0]],
                       [0., tf.gain, 100*tf.gain+tf.pad_ltrb[1]]], np.float32)
    values = dict(image_gray=canvas, baseline_points=points, point_valid=valid,
        point_conf=conf, detected=np.array(item is not None, bool),
        diagonal=np.array(np.hypot(h, w), np.float32), raw_to_input_affine=affine,
        input_shape_hw=np.array(tf.input_shape_hw, np.int64), raw_shape_hw=np.array([h, w], np.int64))
    inputs = {key: torch.from_numpy(np.array(value[None], copy=True)).to(device) for key, value in values.items()}
    inputs['image_gray'] = inputs['image_gray'].float().div_(255.)
    inputs['input_content_mask'] = C.content_mask(inputs['raw_to_input_affine'], inputs['raw_shape_hw'], inputs['input_shape_hw'])
    return inputs, baseline, dict(raw_shape_hw=[h, w], input_shape_hw=list(tf.input_shape_hw),
        raw_to_input_affine=affine.tolist(), transform=tf.to_dict())


class RawLocalPredictor:
    """Loads completed weights/scale; construction itself performs no forward."""
    def __init__(self, checkpoint, device='cuda:0'):
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.name != 'checkpoint_final.pth' or checkpoint.parent.parent.name != 'runs':
            raise ValueError('Use run/runs/<arm>_seedN/checkpoint_final.pth')
        self.run = checkpoint.parents[2]
        self.arm, text = checkpoint.parent.name.rsplit('_seed', 1)
        self.seed, self.device = int(text), COMMON.torch_device(device)
        self.selection, self.bindings = load_selection(self.run, self.arm, self.seed)
        self.scale = self.selection['scale']
        self.refiner = T.load_trained(self.run, self.arm, self.seed, self.device).float().eval()
        for parameter in self.refiner.parameters():
            parameter.requires_grad_(False)
        protocol = C.read(self.run/'PROTOCOL.json')
        base = Path(protocol['input_run']).resolve()
        C.verify_hashes({str(base/'PROTOCOL.json'): protocol['input_protocol_sha256'],
                        str(base/'CACHE_COMPLETION.json'): protocol['input_cache_completion_sha256']})
        lineage = C.read(base/'PROTOCOL.json')
        snapshot_ref = lineage['input_snapshot']
        C.verify_hashes({snapshot_ref['path']: snapshot_ref['sha256']})
        snapshot = C.read(snapshot_ref['path'])['sha256']
        old_run = Path(lineage['input_run'])
        def frozen_json(path):
            path = Path(path).resolve()
            C.verify_hashes({str(path): snapshot[str(path)]})
            self.bindings[str(path)] = snapshot[str(path)]
            return C.read(path)
        original = frozen_json(old_run/'TRAIN_PROTOCOL.json')
        extraction = frozen_json(old_run/'EXTRACTION_COMPLETE.json')
        if not (extraction['complete'] and extraction['PASS']):
            raise ValueError('Completed original frozen extraction required')
        if (extraction['precision'], extraction['cudnn_benchmark'], extraction['cudnn_allow_tf32'], extraction['matmul_allow_tf32']) != ('FP32', False, True, False):
            raise ValueError('Original backbone numerical recipe differs')
        C.verify_hashes(extraction['bindings']['source_sha256'])
        backbone = original['backbone']
        if backbone['sha256'] != COMMON.BACKBONE_SHA:
            raise ValueError('Original frozen joint backbone identity differs')
        self.bindings.update({str(self.run/'PROTOCOL.json'): C.sha(self.run/'PROTOCOL.json'),
            str(base/'PROTOCOL.json'): protocol['input_protocol_sha256'],
            snapshot_ref['path']: snapshot_ref['sha256'], backbone['checkpoint']: backbone['sha256']})
        C.verify_hashes(self.bindings)
        self.capture = OLD.FrozenCapture(backbone['checkpoint'], COMMON.predictor_device(device))
        self.warmup_forwards, self.warmed = extraction['warmup_forwards'], False
        paths = [__file__, C.__file__, T.__file__, SD.__file__, COMMON.__file__,
                 Path(__file__).with_name('model.py'), Path(__file__).with_name('real_evaluation.py')]
        self.source_sha256 = {str(Path(p).resolve()): C.sha(p) for p in paths}
        self.source_sha256.update(OLD.source_hashes())

    def predict(self, original_bgr):
        started = time.perf_counter()
        warmed = 0
        with COMMON.precision(True), torch.inference_mode():
            if not self.warmed:
                for _ in range(self.warmup_forwards):
                    self.capture.predict(original_bgr)
                    warmed += 1
                self.warmed = True
            candidates, captured, backbone_ms = self.capture.predict(original_bgr)
        inputs, baseline, geometry = image_inputs(original_bgr, candidates, captured['input_shape_hw'], self.device)
        use = baseline['detected'] and any(baseline['point_valid'][:8]) and self.scale != 0
        diagnostics = None
        unscaled = np.array(baseline['points'], np.float64)
        if use:
            with COMMON.precision(False), torch.inference_mode():
                predicted, diag = self.refiner(inputs)
            unscaled = predicted[0].detach().cpu().numpy()
            diagnostics = {key: plain(value[0]) if torch.is_tensor(value) else plain(value) for key, value in diag.items()}
        selected = apply_scale(baseline['points'], unscaled, baseline['point_valid'], self.scale)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return dict(schema='pallet_dht_local_raw_inference_v3', complete=True, arm=self.arm, seed=self.seed,
            baseline=baseline, baseline_points=baseline['points'], selected_points=selected.tolist(),
            point_valid=baseline['point_valid'], scale=self.scale, diagnostics=diagnostics,
            image_geometry=geometry, refiner_used=bool(use),
            fallback_reason=None if use else 'zero_scale_or_missing_detection_or_corners',
            timings=dict(total_after_BGR_input_ms=(time.perf_counter()-started)*1000,
                canonical_backbone_ms=backbone_ms, warmup_forwards=warmed, accuracy_forwards=1),
            input_sha256=self.bindings, source_sha256=self.source_sha256,
            GT_read=False, cached_predictions_used=False, baseline_centroid_preserved=True,
            boxes_scores_other_candidates_preserved=True, raw_cache_parity_previously_verified=False,
            scope='Calibrated local point offsets with original semantic IDs. No C4 repair, saliency or correctness-calibrated line probabilities.')

    def close(self):
        self.capture.handle.remove()
        for handle in self.capture.predictor._evidence_handles:
            handle.remove()


def infer_image(image_path, checkpoint, device='cuda:0'):
    import cv2
    path = Path(image_path).resolve()
    digest = C.sha(path)
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError('Original BGR image cannot be decoded')
    predictor = RawLocalPredictor(checkpoint, device)
    try:
        result = predictor.predict(raw)
    finally:
        predictor.close()
    if C.sha(path) != digest:
        raise ValueError('Original image changed during inference')
    result.update(image=str(path), image_sha256=digest)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    C.write(args.output, infer_image(args.image, args.checkpoint, args.device))
