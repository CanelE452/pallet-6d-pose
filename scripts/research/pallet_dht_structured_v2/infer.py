"""Original BGR -> frozen point/DHT backbone -> trained whole-layout verifier.

Uses the original reflected/rectangular FP32 predictor and its actual P4 hook.
No annotation, GT mask, cached prediction, or image-dependent tuning is read.
The checkpoint must be a completed main run; supplied margin is explicit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from pathlib import Path
import time

import numpy as np
import torch

from . import cache as C
from . import data_ops as D
from . import evaluation as E
from . import proposals as P
from .model import LayoutVerifier
from scripts.research.pallet_dht_decoder_probe_v1 import cache as OLD
from scripts.research.pallet_dht_decoder_probe_v1 import geometry as G
from scripts.research.pallet_line_pose_v1 import source_data as SD


SCHEMA = 'pallet_dht_structured_raw_inference_v2'
BACKBONE_SHA = '0960fb32fd99fd0837a588792e07727298fc6f88f1e4ec3464efd69bd5574d37'


@contextmanager
def precision(cudnn_allow_tf32):
    """Original backbone and new verifier intentionally use different settings."""
    before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
              torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = bool(cudnn_allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_tf32) = before


def torch_device(value):
    text = str(value)
    return torch.device('cuda:' + text if text.isdigit() else text)


def predictor_device(value):
    device = torch_device(value)
    return str(device.index or 0) if device.type == 'cuda' else str(device)


def source_sha256():
    paths = {str(Path(p).resolve()) for p in (__file__, C.__file__, D.__file__, E.__file__,
             P.__file__, G.__file__, SD.__file__)}
    paths |= set(OLD.source_hashes())
    paths.add(str(Path(__file__).with_name('model.py').resolve()))
    paths.add(str(Path(__file__).with_name('train.py').resolve()))
    return {path: C.sha(path) for path in sorted(paths)}


def observation_from_capture(candidates, captured, raw_shape_hw, geometry_config):
    """Reproduce decoder.cache.extract's prediction-only fields exactly.

    The content affine comes from the existing source_data transform. Returned
    predictor points already use the canonical inverse and are not transformed
    again or clipped to the raw image. No synthetic target branch is invoked.
    """
    h, w = map(int, raw_shape_hw)
    if min(h, w) < 1:
        raise ValueError('Positive original-image dimensions required')
    selected = int(np.argmax([c['score'] for c in candidates])) if candidates else None
    item = candidates[selected] if selected is not None else None
    points = np.array(item['keypoints_xy'], np.float32) if item and item['keypoints_xy'] is not None else np.zeros((9, 2), np.float32)
    valid = np.isfinite(points).all(-1) if item and item['keypoints_xy'] is not None else np.zeros(9, bool)
    confidence = np.array(item['keypoints_conf'], np.float32) if item and item['keypoints_conf'] is not None else np.zeros(9, np.float32)
    if points.shape != (9, 2) or confidence.shape != (9,) or not np.isfinite(confidence).all():
        raise ValueError('Expected canonical nine-point/confidence output')
    points[~valid] = 0
    baseline = dict(points=points.tolist(), point_valid=valid.tolist(), point_conf=confidence.tolist(),
        box_xyxy=item['box_xyxy'] if item else [0., 0., 0., 0.], score=item['score'] if item else 0.,
        detected=item is not None, selected_instance=selected, all_candidates=copy.deepcopy(candidates))
    transform = SD.transform((h + 200, w + 200), captured['input_shape_hw'])
    affine = np.array([[transform.gain, 0, 100 * transform.gain + transform.pad_ltrb[0]],
                       [0, transform.gain, 100 * transform.gain + transform.pad_ltrb[1]]], np.float64)
    frame = {**captured, 'raw_to_input_affine': affine}
    record = dict(width=w, height=h, baseline=baseline, raw_to_input_affine=affine.tolist(),
        input_shape_hw=np.asarray(captured['input_shape_hw']).tolist())
    bank = G.build_candidates(captured['logits'], captured['theta'], captured['rho'],
        captured['lattice_valid'], captured['feature_shape_hw'], captured['input_shape_hw'],
        affine, points, confidence, config=geometry_config)
    evidence = {'line_fusion__' + key: bank[key] for key in
        ('line_peaks_h_raw', 'line_peak_valid', 'candidates_xy', 'candidate_valid')}
    packed = C.pack_observation(record, frame, evidence)
    return packed, dict(baseline=baseline, raw_to_input_affine=affine.tolist(),
        input_shape_hw=record['input_shape_hw'], feature_shape_hw=list(np.asarray(captured['p4']).shape[-2:]),
        transform=transform.to_dict(), raw_shape_hw=[h, w])


def score_observation(packed, verifier, margin, device='cpu'):
    """Same clean proposals, FP32 layout coordinates and strict margin as eval."""
    if margin != 'identity_only' and (not np.isfinite(margin) or margin < 0):
        raise ValueError('Nonnegative margin or identity_only required')
    baseline = np.asarray(packed['baseline_points'], np.float32)
    point_valid = np.asarray(packed['point_valid'], bool)
    if not bool(packed['detected']) or not point_valid[:8].all():
        return dict(selected_points=baseline.tolist(), selected_index=0, best_index=0,
            costs=None, candidate_valid=None, cost_gap=None, proposal=None, diagnostics=None,
            verifier_used=False, fallback_reason='missing_detection_or_baseline_corner')
    inputs = {key: torch.from_numpy(np.asarray(value)[None].copy()).to(device) for key, value in packed.items()}
    # The empty target object passes through untouched; no target is consulted.
    batch, valid, _ = D.prepare_batch(inputs, {}, ['clean'], np.zeros((1, 4)))
    proposal = P.build_proposals(packed['baseline_points'], packed['intersections'],
        packed['intersection_valid'], float(packed['diagonal']), point_valid=point_valid)
    expected_layouts = np.asarray(proposal['layouts'], np.float32)
    if not np.array_equal(expected_layouts, batch['layouts'][0].detach().cpu().numpy()):
        raise ValueError('Raw and cached clean proposal conventions differ')
    with precision(False), torch.inference_mode():
        cost, diagnostics = verifier(batch, return_diagnostics=True)
    costs = cost.detach().float().cpu().numpy()
    validity = valid.detach().cpu().numpy()
    selected = int(E.select_indices(costs, validity, margin)[0])
    best = int(np.argmin(np.where(validity[0], costs[0], np.inf)))
    selected_points = baseline.copy() if selected == 0 else expected_layouts[selected].copy()
    selected_points[8] = baseline[8]
    if not np.array_equal(selected_points[8], baseline[8]):
        raise ValueError('Centroid changed')
    def plain(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return value
    return dict(selected_points=selected_points.tolist(), selected_index=selected, best_index=best,
        costs=costs[0].tolist(), candidate_valid=validity[0].tolist(),
        cost_gap=float(costs[0, 0] - costs[0, best]),
        proposal=plain(proposal), diagnostics=plain(diagnostics), verifier_used=True,
        fallback_reason='identity_only_policy' if margin == 'identity_only' else
            'margin_not_exceeded_or_identity_best' if selected == 0 else None)


class RawLayoutPredictor:
    """Reusable raw-image predictor; model construction performs no forwards."""
    def __init__(self, verifier_checkpoint, margin, device='cuda:0'):
        from .train import load_trained
        self.checkpoint = Path(verifier_checkpoint).resolve()
        if self.checkpoint.name != 'checkpoint_final.pth' or self.checkpoint.parent.parent.name != 'runs':
            raise ValueError('Use the completed run/runs/<arm>_seedN/checkpoint_final.pth')
        self.run = self.checkpoint.parents[2]
        name = self.checkpoint.parent.name
        arm, seed_text = name.rsplit('_seed', 1)
        self.arm, self.seed = arm, int(seed_text)
        self.protocol = C.read(self.run / 'PROTOCOL.json')
        self.device, self.margin = torch_device(device), margin
        if margin != 'identity_only' and (not np.isfinite(margin) or margin < 0):
            raise ValueError('Margin must be nonnegative or identity_only')
        self.verifier = load_trained(self.run, arm, self.seed, self.device)
        if not isinstance(self.verifier, LayoutVerifier):
            raise ValueError('Expected the trained LayoutVerifier architecture')
        self.verifier.float().eval()
        for parameter in self.verifier.parameters():
            parameter.requires_grad_(False)
        old_root = Path(self.protocol['input_run']).resolve()
        snapshot_ref = self.protocol['input_snapshot']
        if C.sha(snapshot_ref['path']) != snapshot_ref['sha256']:
            raise ValueError('Source snapshot changed')
        snapshot = C.read(snapshot_ref['path'])
        expected = snapshot['sha256']
        self.bindings = {str(self.checkpoint): C.sha(self.checkpoint),
            str(self.run / 'PROTOCOL.json'): C.sha(self.run / 'PROTOCOL.json'),
            snapshot_ref['path']: snapshot_ref['sha256']}
        def frozen_json(path):
            path = Path(path).resolve()
            if C.sha(path) != expected[str(path)]:
                raise ValueError(f'Original extraction metadata changed: {path}')
            self.bindings[str(path)] = C.sha(path)
            return C.read(path)
        old_protocol = frozen_json(old_root / 'TRAIN_PROTOCOL.json')
        extraction = frozen_json(old_root / 'EXTRACTION_COMPLETE.json')
        if not extraction['complete'] or not extraction['PASS']:
            raise ValueError('Completed original extraction metadata required')
        if (extraction['precision'], extraction['cudnn_benchmark'], extraction['cudnn_allow_tf32'],
                extraction['matmul_allow_tf32']) != ('FP32', False, True, False):
            raise ValueError('Original extraction precision recipe differs')
        OLD.verify_hashes(extraction['bindings']['source_sha256'])
        backbone = old_protocol['backbone']
        if backbone['sha256'] != BACKBONE_SHA or C.sha(backbone['checkpoint']) != BACKBONE_SHA:
            raise ValueError('Original frozen joint backbone changed')
        self.bindings[backbone['checkpoint']] = BACKBONE_SHA
        self.geometry_config = old_protocol['geometry']
        self.warmup_forwards = extraction['warmup_forwards']
        self.warmed = False
        self.capture = OLD.FrozenCapture(backbone['checkpoint'], predictor_device(device))
        self.source_sha256 = source_sha256()

    def predict(self, original_bgr):
        if original_bgr is None or original_bgr.dtype != np.uint8 or original_bgr.ndim != 3 or original_bgr.shape[2] != 3:
            raise ValueError('Expected original uint8 HxWx3 BGR pixels')
        started = time.perf_counter()
        warmed = 0
        with precision(True), torch.inference_mode():
            if not self.warmed:
                for _ in range(self.warmup_forwards):
                    self.capture.predict(original_bgr)
                    warmed += 1
                self.warmed = True
            candidates, captured, backbone_ms = self.capture.predict(original_bgr)
        packed, record = observation_from_capture(candidates, captured, original_bgr.shape[:2], self.geometry_config)
        prediction = score_observation(packed, self.verifier, self.margin, self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        baseline = record['baseline']
        result = dict(schema=SCHEMA, complete=True, arm=self.arm, seed=self.seed,
            baseline=baseline, baseline_points=baseline['points'], selected_points=prediction.pop('selected_points'),
            point_valid=baseline['point_valid'], margin=self.margin, **prediction,
            image_geometry={key: value for key, value in record.items() if key != 'baseline'},
            line_evidence=dict(line_h=packed['line_h'].tolist(), peak_logits=packed['peak_logits'].tolist(),
                peak_valid=packed['peak_valid'].tolist(), edges=list(map(list, G.EDGES)),
                semantics='Actual raw-image structural-role logits/modes; not observed edge visibility, calibrated correctness or attention.'),
            timings=dict(total_after_BGR_input_ms=(time.perf_counter() - started) * 1000,
                canonical_backbone_ms=backbone_ms, warmup_forwards=warmed,
                accuracy_forwards=1, warmup_in_total_time=bool(warmed)),
            bindings=self.bindings, source_sha256=self.source_sha256,
            GT_read=False, baseline_centroid_preserved=True, boxes_scores_other_candidates_preserved=True,
            cached_prediction_used=False, raw_cache_parity_previously_verified=False,
            margin_semantics='Caller-supplied frozen rule; this call does not calibrate or tune it.')
        return result

    def close(self):
        self.capture.handle.remove()
        for handle in self.capture.predictor._evidence_handles:
            handle.remove()


def infer_image(image_path, verifier_checkpoint, margin, device='cuda:0'):
    import cv2
    image_path = Path(image_path).resolve()
    image_digest = C.sha(image_path)
    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f'Cannot decode original BGR image: {image_path}')
    model = RawLayoutPredictor(verifier_checkpoint, margin, device)
    try:
        result = model.predict(raw)
    finally:
        model.close()
    if C.sha(image_path) != image_digest:
        raise ValueError('Image changed during raw inference')
    result.update(image=str(image_path), image_sha256=image_digest)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--margin', required=True, help='Nonnegative frozen margin or identity_only')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    margin = args.margin if args.margin == 'identity_only' else float(args.margin)
    C.write(args.output, infer_image(args.image, args.checkpoint, margin, args.device))
    print(args.output.resolve())
