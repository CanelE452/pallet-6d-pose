"""Build the NEW pointline_v4 export from a fresh native P3/P4 capture.

Frozen provenance (baseline points, role lines, intersections, synthetic GT)
is REUSED from the completed structured-v2 pack; only the feature planes are
newly captured, and the captured P4 is compared bit-for-bit with the cached P4.
No GT is read by the observation path and no repository artifact is modified.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from cli.common import (ROOT, RESULTS, STRUCTURED_V2, DECODER_PROBE, PADDED_FEATURE_HW,
                        backbone_binding, read, write, sha256)  # noqa: E402
sys.path.insert(0, str(ROOT))
from pointline_v4.adapter_helpers import (find_unique_hough_module, capture_native_features,
                                          explicit_feature_affine, content_mask_from_raw_rectangle)  # noqa: E402
from pointline_v4.model import Observation  # noqa: E402
from pointline_v4.cache_io import SCHEMA  # noqa: E402

# [확인] Verified against the repository sampler contract (decoder_probe
# cache.sample_visual: grid = 2*input/size-1, align_corners=False), which places
# feature centre index j at input coordinate stride*(j+0.5). Measured slope in
# AFFINE_PROBE.json confirms 1/stride; the probe cannot resolve the remaining
# 0.03-cell offset choice, which is constant across all four arms.
INDEX_OFFSET = (-.5, -.5)
SPLITS = dict(train='train', calibration='calibration', synth_val='synth_val')


def load_raw(record, cv2):
    if sha256(record['image']) != record['image_sha256']:
        raise ValueError('Source image changed: ' + record['image'])
    image = cv2.imread(record['image'])
    if image is None:
        raise ValueError('Image decode failed: ' + record['image'])
    raw = image[100:-100, 100:-100].copy() if record['prepared_image'] else image
    if raw.shape[:2] != (record['height'], record['width']):
        raise ValueError('Raw image shape differs')
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default=str(RESULTS / 'export'))
    ap.add_argument('--limit', type=int, default=0, help='Smoke only; never used for the registered export')
    args = ap.parse_args()
    import cv2
    from scripts.research.pallet_dht_structured_v2.cache import CachedData
    from scripts.research.pallet_dht_structured_v2.proposals import build_proposals
    from scripts.research.pallet_dht_joint_v1.evaluate import CanonicalPredictor

    torch.set_num_threads(2); cv2.setNumThreads(1)
    # Reproduce the numerical environment recorded by the frozen extraction
    # (EXTRACTION_COMPLETE.json), not a stricter one of our own choosing.
    recorded = read(DECODER_PROBE / 'EXTRACTION_COMPLETE.json')
    torch.backends.cudnn.benchmark = bool(recorded['cudnn_benchmark'])
    torch.backends.cudnn.allow_tf32 = bool(recorded['cudnn_allow_tf32'])
    torch.backends.cuda.matmul.allow_tf32 = bool(recorded['matmul_allow_tf32'])

    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Export exists; preserve it rather than overwriting')
    data = CachedData(STRUCTURED_V2, verify=True)
    manifest = read(DECODER_PROBE / 'MANIFEST.json')
    if len(manifest['records']) != len(data.records):
        raise ValueError('Manifest/cache length mismatch')
    binding = backbone_binding()
    predictor = CanonicalPredictor(binding['checkpoint'], '0')
    module = find_unique_hough_module(predictor.model.model)

    def state_sha():
        import hashlib
        h = hashlib.sha256()
        for name, value in predictor.model.model.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    warm = load_raw(manifest['records'][0], cv2)
    for _ in range(5):
        predictor.predict(warm)
    before = state_sha()

    wanted = {}
    for split, key in SPLITS.items():
        wanted[split] = list(data.populations[key])
    if args.limit:
        wanted = {s: v[:args.limit] for s, v in wanted.items()}
    order = sorted({i for v in wanted.values() for i in v})

    parity = []
    records_by_index = {}
    started = time.perf_counter()
    for count, index in enumerate(order):
        record = manifest['records'][index]
        cached = data.records[index]
        if record['id'] != cached['id'] or record['image_sha256'] != cached['image_sha256']:
            raise ValueError('Manifest/cache record binding mismatch')
        raw = load_raw(record, cv2)
        with capture_native_features(module, detach=True, cpu=True) as capture:
            predictor.predict(raw)
        if capture.calls != 1:
            raise ValueError('Expected exactly one captured forward')
        p3, p4 = capture.features
        input_hw = np.asarray(data.inputs['input_shape_hw'][index], np.int64)
        for stride, feature in ((8, p3), (16, p4)):
            expect = tuple(int(v) // stride for v in input_hw)
            if tuple(feature.shape[-2:]) != expect or feature.shape[1] != (64 if stride == 8 else 128):
                raise ValueError(f'Captured stride{stride} plane {tuple(feature.shape)} != expected {expect}')
        # Strict parity against the already-frozen cached P4 (zero-padded to 40x40).
        h4, w4 = p4.shape[-2:]
        cached_p4 = torch.from_numpy(np.asarray(data.inputs['p4'][index])[:, :h4, :w4].copy())
        difference = float((cached_p4 - p4[0]).abs().max())
        parity.append(dict(frame_id=record['id'], max_abs_p4_difference=difference,
                           bit_exact=bool(torch.equal(cached_p4, p4[0]))))

        planes, affines, masks = [], [], []
        raw_to_input = torch.from_numpy(np.asarray(data.inputs['raw_to_input_affine'][index], np.float64))[None]
        raw_hw = torch.tensor([[record['height'], record['width']]], dtype=torch.long)
        for stride, feature in ((8, p3), (16, p4)):
            ph, pw = PADDED_FEATURE_HW[stride]
            fh, fw = feature.shape[-2:]
            if fh > ph or fw > pw:
                raise ValueError('Feature plane exceeds the fixed padded footprint')
            padded = torch.zeros((1, feature.shape[1], ph, pw), dtype=torch.float32)
            padded[:, :, :fh, :fw] = feature
            affine = explicit_feature_affine(raw_to_input, stride_xy=(stride, stride), index_offset_xy=INDEX_OFFSET)
            mask = content_mask_from_raw_rectangle(affine, raw_hw, (ph, pw))
            extent = torch.zeros((1, 1, ph, pw), dtype=torch.bool)
            extent[:, :, :fh, :fw] = True
            planes.append(padded); affines.append(affine); masks.append(mask & extent)

        baseline = np.asarray(data.inputs['baseline_points'][index], np.float64)
        point_valid = np.asarray(data.inputs['point_valid'][index], bool)
        diagonal = float(data.inputs['diagonal'][index])
        proposal = build_proposals(baseline, np.asarray(data.inputs['intersections'][index], np.float64),
                                   np.asarray(data.inputs['intersection_valid'][index], bool),
                                   diagonal, point_valid=point_valid)
        observation = Observation(
            features=tuple(planes), raw_to_feature=tuple(affines), content_valid=tuple(masks),
            baseline=torch.from_numpy(baseline.astype(np.float32))[None],
            point_valid=torch.from_numpy(point_valid)[None],
            point_conf=torch.from_numpy(np.asarray(data.inputs['point_conf'][index], np.float32))[None],
            layouts=torch.from_numpy(proposal['layouts'].astype(np.float32))[None],
            candidate_valid=torch.from_numpy(proposal['valid'])[None],
            line_h=torch.from_numpy(np.asarray(data.inputs['line_h'][index], np.float32))[None],
            line_logits=torch.from_numpy(np.asarray(data.inputs['peak_logits'][index], np.float32))[None],
            line_valid=torch.from_numpy(np.asarray(data.inputs['peak_valid'][index], bool))[None],
            diagonal=torch.tensor([diagonal], dtype=torch.float32))
        observation.validate((64, 128))
        supervision = dict(points=torch.from_numpy(np.asarray(data.targets['points'][index], np.float32))[None],
                           supervised=torch.from_numpy(np.asarray(data.targets['valid'][index], bool))[None],
                           matched=torch.from_numpy(np.asarray(data.targets['matched'][index], bool)).reshape(1))
        stem = f'{index:06d}'
        obs_path = out / 'observations' / (stem + '.pt')
        sup_path = out / 'supervision' / (stem + '.pt')
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        sup_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({f: getattr(observation, f) for f in
                    ('features', 'raw_to_feature', 'content_valid', 'baseline', 'point_valid', 'point_conf',
                     'layouts', 'candidate_valid', 'line_h', 'line_logits', 'line_valid', 'diagonal')}, obs_path)
        torch.save(supervision, sup_path)
        records_by_index[index] = dict(frame_id=record['id'], session_id=record['session_id'],
                                       origin='source_synthetic', state='native',
                                       observation=str(obs_path.relative_to(out)), observation_sha256=sha256(obs_path),
                                       supervision=str(sup_path.relative_to(out)), supervision_sha256=sha256(sup_path))
        if (count + 1) % 200 == 0:
            print(f'exported {count + 1}/{len(order)} ({time.perf_counter() - started:.0f}s)', flush=True)

    after = state_sha()
    if before != after:
        raise ValueError('Frozen backbone/BN state changed during capture')
    for split, indices in wanted.items():
        write(out / f'{split}.json', dict(schema=SCHEMA, role=split, channels=[64, 128],
                                          records=[records_by_index[i] for i in indices]))
    exact = sum(r['bit_exact'] for r in parity)
    write(RESULTS / 'EXPORT_COMPLETION.json', dict(
        schema='pointline_v4_export_completion_1', complete=True,
        frames=len(order), splits={s: len(v) for s, v in wanted.items()},
        backbone=binding, frozen_state_sha_before=before, frozen_state_sha_after=after,
        p4_parity_bit_exact_frames=exact, p4_parity_frames=len(parity),
        p4_parity_max_abs_difference=max(r['max_abs_p4_difference'] for r in parity),
        strict_p4_parity_PASS=bool(exact == len(parity)),
        index_offset_xy=list(INDEX_OFFSET), padded_feature_hw={str(k): list(v) for k, v in PADDED_FEATURE_HW.items()},
        real_GT_read=False, elapsed_seconds=time.perf_counter() - started,
        manifest_sha256={s: sha256(out / f'{s}.json') for s in wanted},
        note='Baseline points, role lines, intersections and synthetic GT are reused from the frozen '
             'structured-v2 pack; only P3/P4 are newly captured from the same verified predictor.',
        parity_rows=parity))
    print(f'strict P4 parity: {exact}/{len(parity)} bit exact; '
          f'max abs diff {max(r["max_abs_p4_difference"] for r in parity):.3e}')


if __name__ == '__main__':
    main()
