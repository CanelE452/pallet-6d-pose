"""TRAIN-only signed residual directions from immutable predictions and poses.

No label, image, model, solver, policy or new pose estimate is needed. Import and
``selfcheck`` are artifact-free; ``write`` requires the sealed AUDIT_PROTOCOL.
"""
from . import common as C
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

MODELS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
HYP = ('long-face-front', 'short-face-front')
RULE = 'BBOX_DIAGONAL_NORMALIZED_PROJECTED_MINUS_OBSERVED_XY9'
ROWS, DIM = 2598, 18
ATOL_PX, RTOL = 1e-4, 1e-6
READS = []
ALLOWED_PREDICTIONS = set()
GUARD_INSTALLED = False


def array_sha(value):
    """Same dtype/shape/bytes convention as original OLD.array_sha."""
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def install_guard():
    global GUARD_INSTALLED
    if GUARD_INSTALLED:
        return
    GUARD_INSTALLED = True

    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(C.ROOT) or p.suffix in ('.py', '.pyc'):
            return
        s = str(p)
        forbidden = ('SOURCE_TRAIN_LABELS', 'SYNTH_LABELS', 'SYNTH_RECORDS',
                     'GEOMETRY_SIDETABLE', 'GEOMETRY_RESOLVED_POSE_GT',
                     'DIMENSION_SIDECAR', 'SOURCE_MANIFEST', 'AXIS_REVIEW',
                     'TRUTH_FOR_DISPLAY', '/data/evaluation/', 'SOURCE_VAL_',
                     'TRAIN_CONVERGENCE', 'REAL_', 'POSE_METRICS',
                     '/fits/', '/model_parameters/', 'rejected_optimizer_state')
        assert not any(token in s for token in forbidden), ('INPUT_ONLY_READ_DENIED', s)
        assert p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp', '.pt', '.pth'), ('NO_IMAGE_OR_WEIGHT', s)
        if '/source_predictions/' in s:
            assert s in ALLOWED_PREDICTIONS, ('NOT_ELIGIBLE_TRAIN_PREDICTION', s)
        if p.suffix == '.npz':
            assert p in {C.PARENT_RAW / 'SOURCE_FEATURES.npz', C.RAW / 'TRAIN_DIRECTIONS.npz'}, ('UNAPPROVED_ARRAY', s)
        READS.append(s)

    sys.addaudithook(hook)


def project_direction(pose, points, box, K):
    """Pure final-pose projection; preserve corner order and add center index8."""
    assert pose['available']
    ext = np.asarray(pose['cf_extents'], np.float64)
    rot = np.asarray(pose['R_cf'], np.float64)
    trans = np.asarray(pose['centroid'], np.float64)
    points, box, K = (np.asarray(x, np.float64) for x in (points, box, K))
    assert ext.shape == (3,) and rot.shape == (3, 3) and trans.shape == (3,)
    assert points.shape == (9, 2) and box.shape == (4,) and K.shape == (3, 3)
    assert all(np.isfinite(x).all() for x in (ext, rot, trans, points, box, K))
    assert (ext > 0).all() and K[0, 0] > 0 and K[1, 1] > 0
    np.testing.assert_array_equal(K[2], [0., 0., 1.])
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-6, rtol=0)
    assert abs(np.linalg.det(rot) - 1.) <= 1e-6
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]], np.float64)
    corners = signs * (ext / 2.)
    camera = corners @ rot.T + trans
    assert np.isfinite(camera).all() and (camera[:, 2] != 0).all()
    homogeneous = camera @ K.T
    uv = homogeneous[:, :2] / homogeneous[:, 2, None]
    delta = uv - points
    # Exactly the old feature extractor's bounding-box diagonal contract.
    wh = np.maximum(box[2:] - box[:2], 1e-6)
    diagonal = float(np.linalg.norm(wh))
    norm_px = np.linalg.norm(delta, axis=1)
    direction64 = (delta / diagonal).reshape(DIM)
    direction18 = direction64.astype(np.float32)
    assert np.isfinite(direction18).all()
    return dict(direction18=direction18, direction64=direction64,
                norm_px=norm_px, diagonal=diagonal, projected=uv)


def parity(new, original, *, diagonal=1.):
    """Fixed tolerance: pixel tolerance scales with the bbox denominator."""
    new, original = np.asarray(new, np.float64), np.asarray(original, np.float64)
    assert new.shape == original.shape and np.isfinite(new).all() and np.isfinite(original).all()
    difference = np.abs(new - original)
    tolerance = ATOL_PX / diagonal + RTOL * np.abs(original)
    assert np.all(difference <= tolerance), ('RESIDUAL_PARITY_FAILED', float(difference.max()), float(tolerance.min()))
    return dict(max_absolute=float(difference.max()),
                max_fraction_of_tolerance=float(np.max(difference / tolerance)))


def normalization(values, valid):
    assert values.dtype == np.float32 and valid.dtype == bool
    selected = values[valid]
    assert selected.ndim == 2 and selected.shape[1] == DIM and len(selected) > 1
    mean = selected.mean(axis=0)
    std = np.maximum(selected.std(axis=0), np.float32(1e-6))
    assert mean.dtype == std.dtype == np.float32
    return mean, std


def load_inputs():
    """Authenticated input-only loader; returns only eligible TRAIN slices."""
    install_guard()
    protocol = C.protocol('AUDIT_PROTOCOL')
    assert protocol['frames'] == ROWS and protocol['models'] == list(MODELS)
    assert protocol['direction_dim'] == DIM and protocol['direction_rule'] == RULE
    assert protocol['source_TRAIN_only'] is True
    assert protocol['parity'] == dict(atol_px=ATOL_PX, rtol=RTOL)
    assert protocol['normalization'] == dict(source='R0_eligible_train_valid_candidates', std_floor=1e-6)
    paths = dict(source_contract=C.PARENT_DOC / 'SOURCE_CONTRACT.json',
                 feature_lock=C.PARENT_DOC / 'SOURCE_FEATURE_LOCK.json',
                 features=C.PARENT_RAW / 'SOURCE_FEATURES.npz', poses=C.PARENT_RAW / 'SOURCE_POSES.json',
                 metadata=C.PARENT_RAW / 'SOURCE_INPUTS.json',
                 source_predictions_lock=C.PARENT_DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert set(protocol['inputs']) == set(paths)
    for key, p in paths.items():
        assert protocol['inputs'][key] == C.bind(p), key
    for p in (Path(__file__), Path(C.__file__)):
        assert C.bind(p) in protocol['codes']
    contract = C.read(paths['source_contract'])
    assert contract['complete'] and contract['status'] == 'PASS'
    eligible = contract['fit_eligibility']['eligible_ids']['TRAIN']
    assert len(eligible) == len(set(eligible)) == ROWS
    eligible = set(eligible)
    lock = C.read(paths['feature_lock'])
    assert lock['complete'] and not lock['source_targets_read'] and not lock['real_targets_read']
    assert lock['models'] == list(MODELS) and lock['hypothesis_names'] == list(HYP)
    assert lock['features'] == protocol['inputs']['features']
    assert lock['poses'] == protocol['inputs']['poses']
    assert lock['metadata'] == protocol['inputs']['metadata']
    assert lock['predictions'] == protocol['inputs']['source_predictions_lock']
    for b in lock['operators']:
        C.verify(b)
    source = C.read(paths['metadata'])
    assert len(source) == 5120 and len({r['id'] for r in source}) == 5120
    expected_keys = {'ids', 'split', 'hypothesis_names'} | {m + s for m in MODELS for s in ('_geo', '_valid', '_GEO_index')}
    with np.load(paths['features'], allow_pickle=False) as f:
        assert set(f.files) == expected_keys
        ids = f['ids'].copy()
        assert ids.tolist() == [r['id'] for r in source]
        assert f['hypothesis_names'].tolist() == list(HYP)
        idx = np.array([i for i, fid in enumerate(ids) if str(fid) in eligible], np.int64)
        assert len(idx) == ROWS and set(ids[idx].tolist()) == eligible
        assert np.all(f['split'][idx] == 'TRAIN')
        features = {m: f[m + '_geo'][idx].copy() for m in MODELS}
        valid = {m: f[m + '_valid'][idx].copy() for m in MODELS}
        anchor = f['R0_GEO_index'][idx].copy()
    rows = [source[i] for i in idx]
    assert all(r['split'] == 'TRAIN' for r in rows)
    for m in MODELS:
        assert features[m].dtype == np.float32 and features[m].shape == (ROWS, 2, 94)
        assert valid[m].dtype == bool and valid[m].shape == (ROWS, 2)
        assert np.isfinite(features[m][valid[m]]).all()
        assert int(np.sum(~valid[m].any(1))) == 1
    assert anchor.dtype == np.int64 and anchor.shape == (ROWS,)
    present = valid['R0'].any(1)
    assert np.array_equal(anchor >= 0, present) and set(anchor.tolist()) <= {-1, 0, 1}
    assert valid['R0'][np.flatnonzero(present), anchor[present]].all()
    assert all(np.array_equal(v.any(1), present) for v in valid.values())
    assert int(valid['R0'].sum()) == 5194
    prediction_lock = C.read(paths['source_predictions_lock'])
    assert prediction_lock['complete'] and prediction_lock['frames'] == 5120
    assert prediction_lock['models'] == list(MODELS)
    assert prediction_lock['metadata'] == lock['metadata']
    assert prediction_lock['protocol'] == lock['protocol']
    C.verify(prediction_lock['protocol'])
    if prediction_lock.get('runtime_amendment'):
        C.verify(prediction_lock['runtime_amendment'])
    source_protocol = C.read(C.ROOT / prediction_lock['protocol']['path'])
    assert source_protocol['input'] == lock['metadata']
    assert source_protocol['source_contract'] == protocol['inputs']['source_contract']
    receipt_bindings = prediction_lock['receipts']
    assert set(receipt_bindings) == set(MODELS)
    predictions = {}
    for m, b in receipt_bindings.items():
        C.verify(b)
        receipt = C.read(C.ROOT / b['path'])
        assert receipt['complete'] and receipt['model'] == m and receipt['frames'] == 5120
        assert receipt['protocol'] == prediction_lock['protocol']
        assert receipt['checkpoint'] == source_protocol['checkpoints'][m]
        assert len(receipt['files']) == 5120
        selected = [receipt['files'][i] for i in idx]
        for i, binding in zip(idx, selected):
            path = C.PARENT_RAW / 'source_predictions' / m / f'{i:05d}.json'
            assert binding['path'] == str(path.relative_to(C.ROOT))
            ALLOWED_PREDICTIONS.add(str(path.resolve()))
        predictions[m] = selected
    poses_all = C.read(paths['poses'])
    assert poses_all['ids'] == ids.tolist() and poses_all['models'] == list(MODELS)
    assert poses_all['hypothesis_names'] == list(HYP)
    assert poses_all['source_targets_read'] is False and poses_all['real_targets_read'] is False
    poses = {m: {r['id']: poses_all['records'][m][r['id']] for r in rows} for m in MODELS}
    return dict(protocol=protocol, protocol_binding=C.bind(C.DOC / 'AUDIT_PROTOCOL.json'),
                feature_lock=lock, prediction_lock=prediction_lock, source_protocol=source_protocol,
                ids=ids[idx], source_index=idx, anchor_index=anchor, rows=rows,
                features=features, valid=valid, poses=poses, predictions=predictions,
                receipt_bindings=receipt_bindings)


def distribution(values):
    values = np.asarray(values, np.float64)
    assert values.size and np.isfinite(values).all()
    return dict(minimum=float(values.min()), maximum=float(values.max()),
                mean=float(values.mean()), median=float(np.median(values)),
                abs_P90=float(np.quantile(np.abs(values), .9)),
                exact_zero_count=int(np.sum(values == 0)), scalar_count=int(values.size))


def extract(data):
    arrays = dict(ids=data['ids'], source_index=data['source_index'], anchor_index=data['anchor_index'])
    audit = {}
    names = data['feature_lock']['feature_names']
    px_cols = [names.index(f'residual{k}_px') for k in range(9)]
    norm_cols = [names.index(f'residual{k}_bboxnorm') for k in range(9)]
    for model in MODELS:
        direction = np.zeros((ROWS, 2, DIM), np.float32)
        v = data['valid'][model]
        max_difference = dict(raw94_px=0., raw94_bboxnorm=0., frozen_corner8_px=0.)
        max_ratio = dict.fromkeys(max_difference, 0.)
        checked = 0
        for n, (row, binding) in enumerate(zip(data['rows'], data['predictions'][model])):
            C.verify(binding)
            saved = C.read(C.ROOT / binding['path'])
            assert saved['id'] == row['id'] and saved['model'] == model
            assert saved['protocol_sha'] == data['prediction_lock']['protocol']['sha256']
            assert saved['checkpoint_sha'] == data['source_protocol']['checkpoints'][model]['sha256']
            pred = saved['prediction']
            record = data['poses'][model][row['id']]
            hh = {h['name']: h for h in record['hypotheses']}
            assert len(hh) == len(record['hypotheses']) and set(hh) <= set(HYP)
            selected = pred['selected_index']
            if v[n].any():
                assert selected is not None and 0 <= selected < len(pred['candidates'])
                candidate = pred['candidates'][selected]
                q = np.asarray(candidate['keypoints_xy'], np.float64)
                assert q.shape == (9, 2) and np.isfinite(q).all() and not (q == -1).all(1).any()
            for j, name in enumerate(HYP):
                if not v[n, j]:
                    continue
                assert name in hh and hh[name]['pose']['available']
                out = project_direction(hh[name]['pose'], q, candidate['box_xyxy'], row['K'])
                old = data['features'][model][n, j]
                comparisons = dict(raw94_px=parity(out['norm_px'], old[px_cols]),
                    raw94_bboxnorm=parity(out['norm_px'] / out['diagonal'], old[norm_cols], diagonal=out['diagonal']),
                    frozen_corner8_px=parity(out['norm_px'][:8], hh[name]['inference_cues']['corner8_residual_px']))
                for key, values in comparisons.items():
                    max_difference[key] = max(max_difference[key], values['max_absolute'])
                    max_ratio[key] = max(max_ratio[key], values['max_fraction_of_tolerance'])
                direction[n, j] = out['direction18']
                checked += 1
        assert checked == int(v.sum()) and np.isfinite(direction).all()
        assert not direction[~v].any()
        arrays[model + '_direction18'] = direction
        arrays[model + '_valid'] = v.copy()
        audit[model] = dict(frames=ROWS, valid_candidates=checked,
            allinvalid_rows=int((~v.any(1)).sum()), allinvalid_ids=data['ids'][~v.any(1)].tolist(),
            direction_sha=array_sha(direction), valid_sha=array_sha(v), raw94_sha=array_sha(data['features'][model]),
            parity_max_absolute=max_difference, parity_max_fraction_of_tolerance=max_ratio,
            valid_direction_distribution=distribution(direction[v]),
            direction_order=[f'point{k}_{axis}' for k in range(9) for axis in ('x', 'y')])
    return arrays, audit


def write():
    install_guard()
    result_path = C.DOC / 'FEATURE_AUDIT.json'
    failed_path = C.DOC / 'FEATURE_AUDIT_FAILED.json'
    assert not result_path.exists() and not failed_path.exists(), 'Immutable completed/failed audit exists; no repeat.'
    start = time.monotonic()
    try:
        data = load_inputs()
        arrays, models = extract(data)
        mean, std = normalization(arrays['R0_direction18'], arrays['R0_valid'])
        norm = dict(source='R0_eligible_train_valid_candidates', valid_candidate_count=5194,
            dtype='float32', std_floor=1e-6, mean18=mean.tolist(), std18=std.tolist(),
            array_sha=array_sha(np.stack([mean, std])), mean_sha=array_sha(mean), std_sha=array_sha(std),
            std_floor_columns=np.flatnonzero(std == np.float32(1e-6)).tolist())
        path = C.RAW / 'TRAIN_DIRECTIONS.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            np.savez_compressed(f, **arrays)
        out = dict(complete=True, PASS=True, status='TRAIN_INPUT_DIRECTIONS_PARITY_PASS',
            created_at=C.now(), protocol=data['protocol_binding'], direction_rule=RULE,
            direction_dim=DIM, frames=ROWS, models=models, normalization=norm,
            directions=C.bind(path), ids_sha=array_sha(arrays['ids']), source_index_sha=array_sha(arrays['source_index']),
            anchor_index_sha=array_sha(arrays['anchor_index']),
            parity=dict(atol_px=ATOL_PX, rtol=RTOL, bboxnorm_atol='atol_px / bbox_diagonal'),
            source_TRAIN_only=True, source_labels_read=False, VAL_quality_read=False,
            real_targets_read=False, fits=0, image_forwards=0, PnP_calls=0,
            new_pose_estimates=0, argmin_calls=0, selected_policy_changes=0,
            invalid_rows_retained=True, container_disclosure=dict(
                SOURCE_FEATURES='Container includes TRAIN4096/VAL1024; only eligible TRAIN2598 arrays sliced.',
                SOURCE_POSES='Container includes TRAIN4096/VAL1024; only eligible TRAIN2598 model records accessed.',
                SOURCE_INPUTS='Input-only membership/K/dimensions container includes all5120; eligibleTRAIN only used.',
                prediction_jsons='Only source_index entries of eligible TRAIN opened; no VAL prediction files.'),
            bindings=dict(inputs=data['protocol']['inputs'], codes=data['protocol']['codes'],
                receipts=data['receipt_bindings'], original_operators=data['feature_lock']['operators'],
                source_protocol=data['prediction_lock']['protocol'],
                runtime_amendment=data['prediction_lock'].get('runtime_amendment')),
            read_paths=sorted(set(READS)), wall_seconds=time.monotonic() - start,
            method_success=False, goal_complete=False)
        C.save(result_path, out)
        print('TRAIN_DIRECTIONS_PARITY_PASS', json.dumps({m: x['parity_max_absolute'] for m, x in models.items()}), flush=True)
    except Exception as error:
        if not failed_path.exists():
            C.save(failed_path, dict(complete=False, PASS=False, status='TRAIN_INPUT_AUDIT_FAILED_STOP',
                created_at=C.now(), error_type=type(error).__name__, error=str(error),
                tolerance=dict(atol_px=ATOL_PX, rtol=RTOL), no_tolerance_relaxation=True,
                source_labels_read=False, VAL_quality_read=False, real_targets_read=False,
                fits=0, PnP_calls=0, read_paths=sorted(set(READS)), method_success=False, goal_complete=False))
        raise


def selfcheck():
    # Camera-facing projection, q0x/q0y ordering and center8 are independent fixtures.
    K = np.array([[600., 0., 420.], [0., 650., 340.], [0., 0., 1.]])
    a = .31
    rotation = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])
    pose = dict(available=True, cf_extents=[1.2, .14, .8], R_cf=rotation.tolist(),
                R_physical=np.eye(3).tolist(), centroid=[.2, -.1, 3.4])
    box = np.array([20., 30., 500., 430.])
    coordinates = []
    for sx, sy, sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),(0,0,0)]:
        x, y, z = sx * .6, sy * .07, sz * .4
        xc = np.cos(a) * x + np.sin(a) * z + .2
        yc = y - .1
        zc = -np.sin(a) * x + np.cos(a) * z + 3.4
        coordinates.append([600. * xc / zc + 420., 650. * yc / zc + 340.])
    coordinates = np.array(coordinates)
    signed = np.arange(18, dtype=np.float64).reshape(9, 2) / 7. - 1.
    observed = coordinates - signed
    result = project_direction(pose, observed, box, K)
    np.testing.assert_allclose(result['projected'], coordinates, atol=1e-12, rtol=0)
    expected = (signed / np.hypot(480., 400.)).reshape(18)
    np.testing.assert_allclose(result['direction18'], expected.astype(np.float32), atol=1e-10, rtol=1e-6)
    # Reflect padding only translates K/q/box; adding padding a second time is forbidden.
    padded = K.copy(); padded[0, 2] += 100.; padded[1, 2] += 100.
    again = project_direction(pose, observed + 100., box + 100., padded)
    np.testing.assert_allclose(again['direction18'], result['direction18'], atol=1e-10, rtol=1e-6)
    assert parity([1.00005], [1.])['max_fraction_of_tolerance'] < 1.
    try:
        parity([1.01], [1.])
    except AssertionError:
        pass
    else:
        raise AssertionError('Parity mismatch did not stop')
    v = np.array([[True, True], [False, False]])
    out = np.zeros((2, 2, 18), np.float32)
    out[0, 0] = result['direction18']; out[0, 1] = -result['direction18']
    mean, std = normalization(out, v)
    np.testing.assert_array_equal(mean, np.zeros(18, np.float32))
    np.testing.assert_array_equal(std, np.maximum(abs(result['direction18']), np.float32(1e-6)))
    assert not out[~v].any()
    assert not GUARD_INSTALLED and not READS, 'Toy must not access actual artifacts'
    print('SELF_CHECK_PASS artifact_reads=0 fits=0 PnP=0 TRAIN_extraction=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'write'])
    args = parser.parse_args()
    selfcheck() if args.stage == 'selfcheck' else write()
