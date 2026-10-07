"""Bind existing TRAIN217 and EVAL128 without inference or reference coordinates.

The old approved, actually used TRAIN population is reused in full: no new
selection by pseudo quality, teacher outcomes, severity or evaluation error.
Original recording identities are recovered by existing provenance plus fresh
RGB SHA equality, never by interpreting a renamed REC identifier as a capture.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_depth_signal_pilot_20261007_v1'
PRIVATE = Path('/dev/shm/pallet-depth-signal-pilot-20261007/INPUTS_PRIVATE.json')
CLEAN = ROOT / 'data/pallet/results/pallet_clean_to_pose_transfer_v1'
EVAL = CLEAN / 'evaluation/S42'
SENSOR = ROOT / 'data/pallet/results/paper_depth_selftrain_v1/sensor_validation_v1'
PARENT = ROOT / '_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only/PROTOCOL.json'
MEMBERS = CLEAN / 'CLEAN_MEMBERSHIP_PRIVATE.json'
ACCEPTED = ROOT / 'data/pallet/results/pallet_type_selftrain_v1/PSEUDO_ACCEPTED.json'
ADAPT = ROOT / 'data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv'
FRAMES = ROOT / 'data/evaluation/pallet_eval_v1/manifests/frames.csv'
RAW = ROOT / 'data/pallet/raw_data'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                sha256=sha(path), bytes=path.stat().st_size)


def verified_image(binding):
    path = ROOT / binding['path']
    actual = bind(path)
    assert actual['sha256'] == binding['sha256'], ('RGB SHA changed', path)
    with Image.open(path) as image:
        hw = [image.height, image.width]
    return actual, hw


def original_index():
    """Reuse historical SHA provenance; do not rerun the 506-frame audit."""
    frames = {r['frame_id']: r for r in csv.DictReader(FRAMES.open())}
    index = {}
    for row in csv.DictReader((SENSOR / 'SENSOR_VALIDATION_POPULATION.csv').open()):
        assert row['frame_id'] in frames
        digest = frames[row['frame_id']]['source_image_sha256']
        assert digest
        original = ROOT / row['rgb_path']
        assert original.parent.parent.name == row['source_recording']
        index.setdefault(digest, []).append(original)
    return index


def resolve_original(binding, index):
    """Candidate stem lookup is accepted only after full original RGB SHA match."""
    candidates = list(index.get(binding['sha256'], []))
    if not candidates:
        stem = Path(binding['path']).stem
        roots = [RAW, RAW / 'outside', RAW / 'night']
        for root in roots:
            if root.is_dir():
                for session in sorted(root.iterdir()):
                    rgb = session / 'rgb'
                    if rgb.is_dir():
                        candidates.extend(rgb.glob(stem + '.*'))
    matched = sorted({p.resolve() for p in candidates if p.is_file() and sha(p) == binding['sha256']})
    assert len(matched) == 1, ('Original RGB provenance unresolved/ambiguous', binding['path'], matched)
    return matched[0]


def selected(packet):
    index = packet['selected_index']
    assert index is not None and 0 <= index < len(packet['candidates'])
    candidate = packet['candidates'][index]
    q = np.asarray(candidate['keypoints_xy'], dtype=np.float64)
    assert q.shape == (9, 2) and np.isfinite(q).all() and not (q == -1).all(axis=1).any()
    return candidate, q


def sensor_fields(original, image_hw, K, audited_recordings):
    session = original.parent.parent
    camera = session / 'cam_K.txt'
    assert camera.is_file(), ('Missing original K', original)
    actual_K = np.loadtxt(camera).reshape(3, 3)
    assert np.isfinite(actual_K).all() and actual_K[0, 0] > 0 and actual_K[1, 1] > 0
    assert np.allclose(actual_K, np.asarray(K), rtol=0, atol=1e-9), ('K mismatch', session)
    depth = session / 'depth' / (original.stem + '.png')
    depth_binding = None
    if depth.is_file():
        with Image.open(depth) as im:
            assert [im.height, im.width] == image_hw, ('Native RGB/depth shape mismatch', depth)
        depth_binding = bind(depth)
    audited = session.name in audited_recordings
    return dict(original_recording=session.name, original_rgb=bind(original),
                camera=bind(camera), depth=depth_binding, depth_unit_scale=.001,
                invalid_raw_codes=[0, 65535], invalid_codes_m=[65.535],
                invalid_code_source='inherited conservative sensor-audit uint16 dtype-max exclusion; acquisition meaning of saturation code unknown; zero already excluded by teacher',
                depth_available=depth_binding is not None,
                pairing_evidence='original RGB SHA verified; matching stored depth stem; same-time/FOV acquisition unproven',
                sensor_validation_recording_audited=audited,
                sensor_validation_applicability=('SAME_ORIGINAL_RECORDING_EMPIRICAL_EVIDENCE'
                    if audited else 'UNTESTED_RECORDING_EXPLORATORY_DAY_NIGHT_TRANSFER'),
                registration_status=('EMPIRICALLY_COLOR_COMPATIBLE_ACQUISITION_UNPROVEN'
                    if audited else 'RECORDING_SPECIFIC_COMPATIBILITY_NOT_ESTABLISHED'),
                depth_semantics='repository consumer optical-Z interpretation; acquisition optical-Z/range declaration absent')


def public_row(row):
    keys = ('id', 'recording', 'original_recording', 'image', 'original_rgb', 'camera', 'depth',
            'depth_available', 'depth_unit_scale', 'invalid_raw_codes', 'invalid_codes_m', 'invalid_code_source',
            'pairing_evidence', 'registration_status',
            'sensor_validation_recording_audited', 'sensor_validation_applicability',
            'depth_semantics', 'support_count', 'pose0_cache_available')
    return {k: row[k] for k in keys if k in row}


def sample_sensor(row):
    rgb = cv2.imread(str(ROOT / row['original_rgb']['path']))
    depth = cv2.imread(str(ROOT / row['depth']['path']), cv2.IMREAD_UNCHANGED)
    assert rgb is not None and depth is not None and depth.dtype == np.uint16
    assert depth.shape == rgb.shape[:2] == tuple(row['hw'])
    valid = (depth > 0) & (depth < 65535)
    return dict(id=row['id'], original_recording=row['original_recording'],
                rgb_hw=list(rgb.shape[:2]), depth_hw=list(depth.shape), depth_dtype=str(depth.dtype),
                zero_fraction=float((depth == 0).mean()), saturated_65535_fraction=float((depth == 65535).mean()),
                valid_raw_min=int(depth[valid].min()), valid_raw_max=int(depth[valid].max()),
                acquisition_metadata_files=sorted(p.name for p in (ROOT / row['original_rgb']['path']).parent.parent.iterdir() if p.is_file()),
                confirms='actual decoding/native grid and storage type only; no scale or registration fit')


def build():
    """Return (private runtime inputs, public provenance receipt); write nothing."""
    members = [r for r in read(MEMBERS)['rows'] if r['used']]
    assert len(members) == len({r['train_id'] for r in members}) == 217
    accepted = {r['id']: r for r in read(ACCEPTED) if r['kind'] == 'PLASTIC'}
    parent = read(PARENT)
    train_list = ROOT / parent['datasets']['RAW']['train_list']['path']
    assert sha(train_list) == parent['datasets']['RAW']['train_list']['sha256']
    old_images = {Path(p).stem: Path(p) for p in train_list.read_text().splitlines() if Path(p).name.startswith('PLASTIC__')}
    assert set(old_images) == {r['train_id'] for r in members}
    index = original_index()
    audited_recordings = {r['source_recording'] for r in
        csv.DictReader((SENSOR / 'SENSOR_VALIDATION_POPULATION.csv').open())}
    train = []
    labels = []
    for member in sorted(members, key=lambda r: r['train_id']):
        row = accepted[member['train_id']]
        assert row['object_type'] == 'plastic_standard_110x130x11'
        assert member['image'] == row['image']
        image, hw = verified_image(row['image'])
        original = (ROOT / image['path']).resolve()
        assert original.parent.name == 'rgb' and original.parent.parent.name == member['session']
        candidate, q = selected(row['raw'])
        label = Path(str(old_images[row['id']]).replace('/images/', '/labels/')).with_suffix('.txt')
        values = np.asarray(label.read_text().split(), dtype=float)
        assert values.shape == (32,)
        points = values[5:].reshape(9, 3)
        visibility = points[:, 2].astype(int)
        assert np.isin(visibility, [0, 1, 2]).all()
        support = visibility == 2
        expected = np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) & (q[:, 0] < hw[1]) & (q[:, 1] >= 0) & (q[:, 1] < hw[0]) & (np.asarray(candidate['keypoints_conf']) >= .5)
        # The previous RAW/REF export used their common trusted mask.  It can
        # exclude a RAW point that alone passes confidence/native bounds.
        # Preserve that exact old mask instead of expanding it here.
        assert np.all(~support | expected), ('Existing pseudo support unsupported by RAW', row['id'])
        assert np.allclose(points[support, :2], (q[support] + 100) / [hw[1] + 200, hw[0] + 200], rtol=0, atol=1e-8)
        labels.append(bind(label))
        fields = sensor_fields(original, hw, row['K'], audited_recordings)
        assert fields['depth_available'], ('TRAIN depth missing', row['id'])
        train.append(dict(id=row['id'], recording=member['recording'], image=image,
                          K=row['K'], xyz=[1.1, .11, 1.3], hw=hw, q0=q.tolist(),
                          support=support.tolist(), oldvisibility=visibility.tolist(), support_count=int(support.sum()),
                          prediction0=row['raw'], pose0=None, pose0_cache_available=False,
                          pose0_status='F_PENDING_NO_NEW_F_IN_INPUT_PREPARATION', old_label=labels[-1],
                          **fields))

    metadata = read(EVAL / 'METADATA.json')
    predictions = read(EVAL / 'PREDICTIONS.json')['R0']
    poses = read(EVAL / 'POSES.json')['R0']
    assert len(metadata) == len({r['id'] for r in metadata}) == 128
    assert set(predictions) == set(poses) == {r['id'] for r in metadata}
    evaluation = []
    for row in metadata:
        image, hw = verified_image(row['image'])
        assert hw == row['hw'] and row['xyz'] == [1.1, .11, 1.3]
        original = resolve_original(image, index)
        candidate, q = selected(predictions[row['id']])
        assert poses[row['id']]['available']
        # Production D9 inference support is not a human visibility label.
        evaluation.append(dict(**row, q0=q.tolist(), support=[True] * 9, support_count=9,
                               prediction0=predictions[row['id']], pose0=poses[row['id']],
                               pose0_cache_available=True, **sensor_fields(original, hw, row['K'], audited_recordings)))
    severity = Counter(r['severity'] for r in evaluation)
    assert severity == dict(CLEAN=29, MODERATE_OCCLUSION=21, SEVERE_OCCLUSION=78)
    sha_overlap = sorted({r['image']['sha256'] for r in train} & {r['image']['sha256'] for r in evaluation})
    recording_overlap = sorted({r['original_recording'] for r in train} & {r['original_recording'] for r in evaluation})
    named_overlap = sorted({r['recording'] for r in train} & {r['recording'] for r in evaluation})
    assert not sha_overlap and not recording_overlap and not named_overlap
    assert len(train) <= 256 and len(train) >= 32 and len({r['original_recording'] for r in train}) >= 2
    adapt = list(csv.DictReader(ADAPT.open()))
    assert len(adapt) == len({r['image_sha256'] for r in adapt}) == 1000
    assert {r['image']['sha256'] for r in train} <= {r['image_sha256'] for r in adapt}
    adapt_depth = sum((ROOT / r['image_path']).parent.parent.joinpath('depth', Path(r['image_path']).name).is_file()
                      and (ROOT / r['image_path']).parent.parent.joinpath('cam_K.txt').is_file() for r in adapt)
    samples = [next(r for r in train if r['original_recording'] == name) for name in ('capturepallet10', 'capturenight03')]
    samples.append(next(r for r in evaluation if r['original_recording'] == 'capturepalletcad'))
    sensor = read(SENSOR / 'SENSOR_VALIDATION_RESULT.json')
    summary = read(SENSOR / 'SENSOR_VALIDATION_SUMMARY.json')
    sources = [MEMBERS, ACCEPTED, PARENT, train_list, ADAPT, FRAMES,
               EVAL / 'METADATA.json', EVAL / 'PREDICTIONS.json', EVAL / 'POSES.json',
               SENSOR / 'SENSOR_VALIDATION_POPULATION.csv', SENSOR / 'SENSOR_VALIDATION_RESULT.json',
               SENSOR / 'SENSOR_VALIDATION_SUMMARY.json', SENSOR / 'SENSOR_VALIDATION_REPORT.md',
               ROOT / 'challenge/scripts/live/run_live_io.py',
               ROOT / 'scripts/research/pallet_type_selftrain_v1/pseudo.py',
               ROOT / 'scripts/research/pallet_type_selftrain_v1/train.py', Path(__file__)]
    public = dict(schema='depth_signal_pilot_input_provenance_v1', status='READY_INPUTS_EXPLORATORY',
                  train_pool='previously approved and actually used217; all217, no new selection or sampling',
                  TRAIN_support='exact previous RAW export visibility/common RAW-REF trusted mask; ignored1 remains ignored; not human visibility',
                  train_count=len(train), train_recordings=dict(Counter(r['original_recording'] for r in train)),
                  train_depth_camera_intersection=len(train), adaptation_pool1000=dict(count=len(adapt),
                      depth_camera_intersection=adapt_depth, recordings=dict(Counter(r['capture_session'] for r in adapt)),
                      audit_only_not_selected=True),
                  EVAL_count=len(evaluation), EVAL_severity=dict(severity), EVAL_clean=29, EVAL_occluded=99,
                  EVAL_depth_camera_intersection=sum(r['depth_available'] for r in evaluation),
                  EVAL_missing_depth_ids=[r['id'] for r in evaluation if not r['depth_available']],
                  EVAL_recordings=dict(Counter(r['original_recording'] for r in evaluation)),
                  sensor_recording_applicability=dict(audited_original_recordings=sorted(audited_recordings),
                      TRAIN_same_recording_evidence=sum(r['sensor_validation_recording_audited'] for r in train),
                      TRAIN_untested_recording_count=sum(not r['sensor_validation_recording_audited'] for r in train),
                      TRAIN_untested_recordings=dict(Counter(r['original_recording'] for r in train if not r['sensor_validation_recording_audited'])),
                      EVAL_depth_same_recording_evidence=sum(r['depth_available'] and r['sensor_validation_recording_audited'] for r in evaluation),
                      transfer_status='day/night aggregate evidence may motivate exploratory checks; unaudited recording-specific compatibility not established'),
                  TRAIN_EVAL_original_RGB_SHA_overlap=sha_overlap, TRAIN_EVAL_original_recording_overlap=recording_overlap,
                  TRAIN_EVAL_recording_alias_overlap=named_overlap,
                  INPUTS=dict(TRAIN=[public_row(r) for r in train], EVAL=[public_row(r) for r in evaluation]),
                  sources=[bind(p) for p in sources], existing_RAW_labels=labels, sample_sensor_decode=[sample_sensor(r) for r in samples],
                  existing_sensor_validation=dict(FINAL=sensor['FINAL'], population=sensor['population'],
                      ready_conditions=sensor['ready_conditions'], scale_evaluated=.001, scale_fitted=False,
                      all506_surface_residual_cm=summary['summary']['ALL']['surface_residual_median_cm_median'],
                      all506_ray_depth_residual_cm=summary['summary']['ALL']['depth_residual_median_cm_median'],
                      rerun=False, not_independent_confirmation=True),
                  dependencies=['RGB-depth same-time/FOV, distortion/extrinsics and optical-Z acquisition declaration absent; exploratory compatibility only',
                      'TRAIN45 from capturepallet10/capturenight10 have no same-recording evidence in the old506 audit; remain provisional exploratory inputs',
                      'Construct TRAIN217 P0 with the frozen original production F; no cached TRAIN pose0 consumed',
                      'EVAL noapril12 missing depth must remain original P0 fallback in full128 denominator',
                      'Fixed prediction-only teacher checks and EVAL>=16 depth/reference common gate remain pending before any fit'],
                  no_reference_coordinates_read=True, no_new_teacher_outcomes_used_for_selection=True,
                  execution=dict(new_NN_forwards=0, new_final_F=0, new_PnP=0, optimizer_updates=0,
                                 sensor_measurement_rerun_frames=0, new_registration_or_scale_fits=0,
                                 sample_RGB_depth_decodes=3))
    private = dict(schema='depth_signal_pilot_private_inputs_v1', TRAIN=train, EVAL=evaluation,
                   public_inputs_path=str(DOC / 'INPUTS.json'), training_recording_key='original_recording',
                   evaluation_recording_key='recording', no_GT_coordinates_read=True)
    return private, public


def _write(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, ('Completed inputs differ; preserve existing receipt', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    private, public = build()
    _write(PRIVATE, private)
    public['private_inputs'] = bind(PRIVATE)
    _write(DOC / 'INPUTS.json', public)
    print(json.dumps({k: public[k] for k in ('status', 'train_count', 'train_depth_camera_intersection', 'EVAL_count', 'EVAL_depth_camera_intersection')}, ensure_ascii=False))
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
