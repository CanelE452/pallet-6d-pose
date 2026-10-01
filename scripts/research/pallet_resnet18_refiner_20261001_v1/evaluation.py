"""CPU-only Sensors DEV319 replay for frozen RESNET18 and its D/P refiners.

This adapter reuses the canonical GT validator, 2-D metric implementation,
prediction-only pose selector/SQPnP/RefineLM, and pose metrics.  It performs no
image/model forward or training.  Missing RESNET18 points are never imputed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments/pallet_resnet18_refiner_20261001_v1'
RAW = ROOT / 'data/pallet/results/pallet_resnet18_refiner_20261001_v1'
POSE = ROOT / 'data/pallet/results/paper_pose_metric_closure_v1'
POSE_CODE = ROOT / 'scripts/paper/pose_metric_closure_v1'
POS = ROOT / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
NEG = POS.with_name('DEV_NEG2689.json')
HISTORICAL = ROOT / '_docs/experiments/pallet_sensors_submission_v1/UNIFIED_DEV_RESULTS.json'
BOOT_CODE = ROOT / 'scripts/research/pallet_sensors_refinement_closeout_v1/paired_stats_helper.py'
ARMS = ('RESNET18', 'D1', 'D2', 'D3', 'P1', 'P2', 'P3')
REFINERS = ARMS[1:]
RESAMPLES = 10000
BOOT_SEED = 20260914
SCHEMA = 'resnet18_refiner_dev_predictions_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).absolute()
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)


def verify_binding(value):
    assert isinstance(value, dict) and 'path' in value and 'sha256' in value
    p = Path(value['path'])
    if not p.is_absolute():
        p = ROOT / p
    assert p.resolve().is_relative_to(ROOT.resolve()), p
    assert sha(p) == value['sha256'], ('Changed binding', p)
    if 'bytes' in value:
        assert p.stat().st_size == value['bytes'], p
    return p


def write(path, value):
    p = Path(path).absolute()
    assert p.is_relative_to(RAW) or p.is_relative_to(DOC), p
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == text, ('Refuse changed existing result', p)
    else:
        with p.open('x') as f:
            f.write(text)


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_modules():
    for p in (ROOT, POSE_CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from challenge.evaluation_v2 import paper_real_eval as E
    M = importlib.import_module('run_pose_evaluation')
    S = importlib.import_module('evaluate_pose_by_session')
    G = importlib.import_module('pose_evaluation_paths')
    B = imported('resnet18_refiner_existing_paired_stats', BOOT_CODE)
    return E, M, S, G, B


def canonical_key(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p.resolve().relative_to(ROOT.resolve()))


def validate_points(value, expected_valid=None):
    p = np.asarray(value, dtype=np.float64)
    assert p.shape == (9, 2), p.shape
    assert not np.isinf(p).any(), 'Inf coordinates are not a missing marker'
    finite = np.isfinite(p)
    assert np.array_equal(finite[:, 0], finite[:, 1]), 'Partly missing xy pair'
    valid = finite.all(-1)
    if expected_valid is not None:
        assert np.array_equal(valid, np.asarray(expected_valid, bool)), 'Missing mask drift'
    return p, valid


def validate_rows(rows, expected_keys=None):
    assert isinstance(rows, list) and rows
    keys = [r['image_key'] for r in rows]
    assert len(set(keys)) == len(keys), 'Duplicate image'
    if expected_keys is not None:
        assert keys == list(expected_keys), 'DEV membership/order drift'
    for r in rows:
        assert r['image_key'] == canonical_key(r['image_key']), 'Noncanonical image key'
        base, valid = validate_points(r['base_points'], r['point_valid'])
        assert set(r['refined']) == set(REFINERS), 'Exact six final refiner arms required'
        assert len(r['original_hw']) == 2 and min(r['original_hw']) > 0
        box = r['box_original']
        if box is not None:
            box = np.asarray(box, np.float64)
            assert box.shape == (4,) and np.isfinite(box).all()
            assert (box[2:] > box[:2]).all(), 'Invalid fixed RESNET18 bbox'
            assert valid[:8].sum() >= 3
            expected = np.r_[base[:8][valid[:8]].min(0), base[:8][valid[:8]].max(0)]
            assert np.allclose(box, expected, atol=1e-6, rtol=0), 'Base corner-hull bbox differs'
            assert np.isfinite(r['score']), 'Missing/nonfinite base score'
        for arm in REFINERS:
            points, _ = validate_points(r['refined'][arm], valid)
            assert np.array_equal(points[8], base[8], equal_nan=True), 'Centroid was refined'
            if box is None:
                assert np.array_equal(points, base, equal_nan=True), 'No-box frame changed'
    return rows


def load_inputs(path):
    payload = read(path)
    assert payload['schema'] == SCHEMA and payload['complete'] is True
    assert payload['coordinate_system'] == 'original_unpadded_pixels'
    for key in ('protocol', 'cache', 'selection'):
        verify_binding(payload[key])
    protocol = read(verify_binding(payload['protocol']))
    assert protocol['schema'] == 'resnet18_local_refiner_protocol_v1'
    for key in ('baseline', 'baseline_protocol', 'baseline_training_complete'):
        assert payload[key] == protocol[key]
        verify_binding(payload[key])
    baseline_receipt = read(verify_binding(payload['baseline_training_complete']))
    assert baseline_receipt['complete'] is True and baseline_receipt['epochs'] == 60
    assert baseline_receipt['real_training'] == 0 and baseline_receipt['final_checkpoint_only'] is True
    assert baseline_receipt['final_checkpoint']['sha256'] == payload['baseline']['sha256']
    assert baseline_receipt['protocol']['sha256'] == payload['baseline_protocol']['sha256']
    assert protocol['evaluation']['n'] == 319
    assert protocol['evaluation']['bootstrap_draws'] == RESAMPLES
    assert protocol['evaluation']['bootstrap_seed'] == BOOT_SEED
    assert protocol['evaluation']['real_based_selection'] is False
    for value in protocol['bindings']:
        verify_binding(value)
    assert set(payload['checkpoints']) == set(REFINERS)
    for value in payload['checkpoints'].values():
        verify_binding(value)
    code = payload['code']
    code = [code] if isinstance(code, dict) and 'path' in code else list(code.values()) if isinstance(code, dict) else code
    assert code
    for value in code:
        verify_binding(value)
    selected = read(verify_binding(payload['selection']))
    assert selected.get('complete') is True
    assert selected.get('no_real_selection') is True or selected.get('real_selection') is False
    assert verify_binding(selected['protocol']) == verify_binding(payload['protocol'])
    assert selected['protocol']['sha256'] == payload['protocol']['sha256']
    E, M, S, G, B = canonical_modules()
    pair = E.validate_evaluation_request(positive_manifest=POS, negative_manifest=NEG,
        population_role=E.PopulationRole.DEV, allow_unavailable_final=False)
    items = pair.positive.items
    assert len(items) == 319
    positive = read(POS)['items']
    ids = {r['frame_id']: r for r in positive}
    assert len(ids) == 319 and len({r['session_id'] for r in positive}) == 13
    keys = [canonical_key(i.image) for i in items]
    rows = validate_rows(payload['frames'], keys)
    manifest = read(POSE / 'AXIS_REVIEW_MANIFEST.json')
    pose_by_key = {canonical_key(r['image']): r for r in manifest['frames_list']}
    assert len(pose_by_key) == 319 and set(pose_by_key) == set(keys)
    gt = read(POSE / 'GEOMETRY_RESOLVED_POSE_GT.json')
    assert set(gt['frames']) == {r['frame_id'] for r in pose_by_key.values()}
    contract = G.load_pose_object_contract(str(POSE / 'POSE_EVAL_OBJECT_CONTRACT.json'))
    assert gt['pose_object_contract_sha256'] == sha(POSE / 'POSE_EVAL_OBJECT_CONTRACT.json')
    targets, pose_ids, source_paths = {}, {}, [POS, NEG, HISTORICAL, BOOT_CODE,
        POSE / 'AXIS_REVIEW_MANIFEST.json', POSE / 'POSE_EVAL_OBJECT_CONTRACT.json',
        POSE / 'GEOMETRY_RESOLVED_POSE_GT.json']
    mapping = []
    for item, row in zip(items, rows):
        meta = ids[item.frame_id]
        pm = pose_by_key[row['image_key']]
        assert row['frame_id'] == item.frame_id and row['session_id'] == meta['session_id']
        assert pm['session_id'] == row['session_id'] and pm['object_type'] == meta['object_type']
        assert canonical_key(item.label) == canonical_key(pm['annotation']), '2D/pose GT path mismatch'
        annotation = read(ROOT / pm['annotation'])
        camera = annotation['camera_data']
        assert list(row['original_hw']) == [camera['height'], camera['width']], 'Padding/image-size drift'
        target = E._legacy_forbidden_target(item)
        raw = camera['intrinsics']
        expected_K = np.array([[raw['fx'], 0., raw['cx']], [0., raw['fy'], raw['cy']], [0., 0., 1.]])
        assert np.array_equal(target.camera_intrinsics, expected_K), 'Canonical K drift'
        spec = G.object_spec(contract, pm['object_type'])
        dims = annotation['objects'][0]['physical_dimensions_m']
        assert sorted([float(dims['x']), float(dims['z'])]) == sorted([spec['long_m'], spec['short_m']])
        assert float(dims['y']) == spec['height_m']
        assert gt['frames'][pm['frame_id']]['object_type'] == pm['object_type']
        targets[item.frame_id] = target
        pose_ids[item.frame_id] = pm['frame_id']
        source_paths.append(ROOT / pm['annotation'])
        mapping.append(dict(frame_id=item.frame_id, pose_frame_id=pm['frame_id'],
            image_key=row['image_key'], session_id=row['session_id'], original_hw=row['original_hw'],
            object_type=pm['object_type'], camera_intrinsics=expected_K.tolist(),
            dimensions_m=dict(long=spec['long_m'], short=spec['short_m'], height=spec['height_m']),
            annotation=binding(ROOT / pm['annotation'])))
    assert sum(int(t.keypoint_supervision_mask.sum()) for t in targets.values()) == 2818
    for module in (E, M, S, G):
        source_paths.append(Path(module.__file__))
    source_paths += [ROOT / 'challenge/evaluation_v2/pnp_selector.py',
        ROOT / 'challenge/evaluation_v2/oriented_iou3d.py',
        POSE_CODE / 'symmetry_aware_pose_metrics.py',
        ROOT / 'challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json']
    sources = [binding(p) for p in dict.fromkeys(source_paths)]
    return payload, rows, items, targets, pose_ids, mapping, sources, (E, M, S, G, B)


def summarize_2d(E, rows, targets, arm):
    """Canonical whole-frame finite support plus explicit full-GT failures."""
    candidates, top, detailed = [], {}, []
    for row in rows:
        fid = row['frame_id']
        target = targets[fid]
        points, valid = validate_points(row['base_points'] if arm == 'RESNET18' else row['refined'][arm])
        iou = None
        if row['box_original'] is not None:
            box = np.asarray(row['box_original'], np.float64)
            iou = float(E._box_iou(box, target.box_xyxy))
            c = E.DetectionCandidate(fid, True, float(row['score']), box, points, iou)
            candidates.append(c)
            top[fid] = c
        matched = iou is not None and iou >= .5 and valid.all()
        mask = target.keypoint_supervision_mask
        distances = np.full(9, np.inf)
        distances[valid & mask] = np.linalg.norm(points[valid & mask] - target.keypoints_xy[valid & mask], axis=-1)
        errors = distances[mask] if matched else np.empty(0)
        detailed.append(dict(frame_id=fid, session_id=row['session_id'], image_key=row['image_key'],
            box_iou=iou, canonical_matched=bool(matched), gt_count=int(mask.sum()),
            finite_predicted_keypoints=int(valid.sum()), missing_supervised_keypoints=int((mask & ~valid).sum()),
            errors=errors, errors8=distances[:8][mask[:8]] if matched else np.empty(0),
            canonical_hits={str(t): int((errors <= t).sum()) for t in (5, 10, 20)},
            # Kept separate: unlike Sensors canonical all9-finite support,
            # this additional endpoint allows finite partial points, with the
            # same fixed IoU requirement and full GT denominator.
            partial_point_hits={str(t): int(((distances <= t) & mask).sum()) if iou is not None and iou >= .5 else 0 for t in (5, 10, 20)},
            point_errors=[float(v) if np.isfinite(v) else None for v in distances]))
    pair = SimpleNamespace(positive=SimpleNamespace(items=rows), negative=SimpleNamespace(items=[]))
    canonical = E._evaluate_2d_collected(pair, targets, candidates, top)
    canonical = {k: v for k, v in canonical.items() if not k.startswith('box_ap') and k != 'negative_count'}
    full = np.concatenate([r['errors'] for r in detailed])
    den = sum(r['gt_count'] for r in detailed)
    summary = dict(median_px=canonical['keypoint_location_median_px'],
        p90_px=canonical['keypoint_location_p90_px'], matched_frames=sum(r['canonical_matched'] for r in detailed),
        full_frame_denominator=len(rows), failed_or_excluded_frames=sum(not r['canonical_matched'] for r in detailed),
        supervised_points=len(full), gt_denominator=den,
        missing_supervised_keypoints=sum(r['missing_supervised_keypoints'] for r in detailed),
        ALL_GT_PCK={str(t): float(sum(r['canonical_hits'][str(t)] for r in detailed) / den) for t in (5, 10, 20)},
        partial_point_ALL_GT_PCK_SECONDARY={str(t): float(sum(r['partial_point_hits'][str(t)] for r in detailed) / den) for t in (5, 10, 20)},
        canonical=canonical,
        policy='Sensors conditional median/P90 require all9 finite and fixed base bbox IoU>=.5; ALL_GT_PCK counts every excluded supervised point as failure; visibility>0; no new prediction-confidence threshold.')
    assert summary['matched_frames'] == canonical['keypoint_matched_frame_count_iou50']
    assert summary['supervised_points'] == canonical['keypoint_all_labeled']['count']
    return summary, detailed


def solve_without_gt(points, camera, spec, M, G):
    """Same canonical MAIN choice and both SQPnP/RefineLM fits; no GT input."""
    import cv2
    points, valid = validate_points(points)
    usable = valid[:8]
    if usable.sum() < 6:
        return dict(status='FEWER_THAN_SIX_FINITE_CORNERS')
    chosen = None
    try:
        result = G.predict_pose_without_gt(points, camera, spec['long_m'],
            spec['short_m'], spec['height_m'])['selector_result']
        for hypothesis in result.hypotheses:
            if hypothesis.name == result.selected_hypothesis and hypothesis.success:
                dims = hypothesis.camera_facing_dimensions.as_dict()
                chosen = M.CF_WIDTH if abs(float(dims['width']) - spec['long_m']) < 1e-6 else M.CF_DEPTH
    except Exception as exc:
        # The canonical MAIN scorer similarly declines after selector errors.
        # Retain the reason instead of silently omitting its population row.
        return dict(status='CANONICAL_SELECTOR_EXCEPTION', reason=type(exc).__name__ + ': ' + str(exc))
    models = {M.CF_WIDTH: M.cuboid(spec['long_m'], spec['height_m'], spec['short_m']),
              M.CF_DEPTH: M.cuboid(spec['short_m'], spec['height_m'], spec['long_m'])}
    try:
        fits = {k: M.solve(model, points[:8], camera, usable) for k, model in models.items()}
    except cv2.error as exc:
        return dict(status='CANONICAL_PNP_FAILED', reason=str(exc))
    if chosen is None or any(value is None for value in fits.values()):
        return dict(status='CANONICAL_MAIN_UNAVAILABLE')
    rotation, translation, residual = fits[chosen]
    assert np.isfinite(rotation).all() and np.isfinite(translation).all() and np.isfinite(residual)
    return dict(status='OK', chosen=chosen, rotation=rotation, translation=translation,
                reprojection_mean_px=residual)


def pose_summary(values, denominator):
    from symmetry_aware_pose_metrics import pose_auc
    names = dict(rotation_median_deg='rotation_error_deg', translation_median_cm='translation_error_cm',
        yaw_median_deg='yaw_error_deg', iou3d_median='iou3d')
    out = dict(n=len(values), denominator=denominator, failure_count=denominator-len(values),
        coverage=len(values)/denominator if denominator else 0.,
        **{key:float(np.median([r[field] for r in values])) if values else None for key,field in names.items()})
    for key, field in (('translation_p90_cm', 'translation_error_cm'),
                       ('rotation_p90_deg', 'rotation_error_deg'),
                       ('yaw_p90_deg', 'yaw_error_deg')):
        out[key] = float(np.quantile([r[field] for r in values], .9)) if values else None
    out['add_sym_auc'] = pose_auc(np.array([r['add_sym_m'] for r in values]),
        float(np.median([r['diameter_m'] for r in values]))) if values else None
    out['axis_accuracy'] = float(np.mean([r['axis_correct'] for r in values])) if values else None
    return out


def canonical_pose(rows, pose_ids, modules, source_binding, mapping):
    """Reuse exact canonical pose primitives and retain every failed row.

    The historical script main functions assume an arm named R0 and nonempty
    support.  This adapter names the actual baseline RESNET18 and permits zero
    coverage, while leaving selector/solve/metric functions unchanged.
    """
    _, M, _, G, _ = modules
    from symmetry_aware_pose_metrics import cuboid_model_points
    truth = read(POSE / 'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
    meta = {r['frame_id']:r for r in mapping}
    output = {}
    for arm in ARMS:
        values, all_rows = [], []
        for row in rows:
            fid = row['frame_id']; item = meta[fid]
            dims = item['dimensions_m']
            spec = dict(long_m=dims['long'], short_m=dims['short'], height_m=dims['height'])
            points = row['base_points'] if arm == 'RESNET18' else row['refined'][arm]
            solved = (solve_without_gt(points, np.asarray(item['camera_intrinsics']), spec, M, G)
                      if row['box_original'] is not None else dict(status='NO_DETECTION'))
            record = dict(frame_id=fid, original_pose_frame_id=pose_ids[fid], session_id=row['session_id'],
                object_type=item['object_type'], image_key=row['image_key'], status=solved['status'])
            if solved['status'] != 'OK':
                if 'reason' in solved:
                    record['reason'] = solved['reason']
                all_rows.append(record)
                continue
            target = truth[pose_ids[fid]]
            td = target['physical_dimensions_m']
            extents = (td['across'], td['height'], td['along'])
            metrics = G.score_pose_against_gt(cuboid_model_points(extents),
                solved['rotation'], solved['translation'], np.asarray(target['R_gt_representative']),
                np.asarray(target['t_gt']), extents)
            assert all(np.isfinite(v) for v in metrics.values()), ('Nonfinite pose metric', arm, fid)
            record.update(translation_error_cm=metrics['translation_error_cm'],
                rotation_error_deg=metrics['rotation_error_deg'], yaw_error_deg=metrics['yaw_error_deg'],
                iou3d=metrics['iou3d'], add_sym_m=metrics['symmetry_aware_add_m'],
                diameter_m=metrics['model_diameter_m'],
                axis_correct=solved['chosen'] == target['physical_long_axis'],
                selected_axis=solved['chosen'], reprojection_mean_px=solved['reprojection_mean_px'])
            values.append(record); all_rows.append(record)
        summary = pose_summary(values, len(rows))
        summary['excluded_frame_ids'] = [r['frame_id'] for r in all_rows if r['status'] != 'OK']
        sessions = sorted({r['session_id'] for r in rows})
        by_session = {s:pose_summary([r for r in values if r['session_id'] == s],
            sum(r['session_id'] == s for r in rows)) for s in sessions}
        artifact = RAW / 'evaluation' / f'POSE_EVALUATION_{arm}.json'
        write(artifact, dict(complete=True, arm=arm, primary_path='MAIN', source=source_binding,
            canonical_functions=['pose_evaluation_paths.predict_pose_without_gt',
                'run_pose_evaluation.cuboid', 'run_pose_evaluation.solve',
                'pose_evaluation_paths.score_pose_against_gt', 'symmetry_aware_pose_metrics.pose_auc'],
            gt=binding(POSE / 'GEOMETRY_RESOLVED_POSE_GT.json'),
            object_contract=binding(POSE / 'POSE_EVAL_OBJECT_CONTRACT.json'),
            summary=summary, by_session=by_session, all_frame_rows=all_rows))
        output[arm] = dict(summary=summary, rows=values, all_frame_rows=all_rows, artifact=binding(artifact))
        print('RESNET18_REFINER_CANONICAL_MAIN', arm, len(values), '/', len(rows), flush=True)
    return output


def interval(values, observed):
    assert np.isfinite(values).all(), 'Empty/nonfinite bootstrap draws must not be discarded'
    return dict(delta=float(observed), low=float(np.quantile(values, .025)),
                high=float(np.quantile(values, .975)))


def paired_medians(left, right, sessions, B, *, level='session', resamples=RESAMPLES):
    """Mean of same-seed pooled-median differences, including one baseline."""
    if len(right) == 1:
        right = right * len(left)
    assert len(left) == len(right)
    n = len(sessions)
    for l, r in zip(left, right):
        assert len(l) == len(r) == n
        if any(len(a) != len(b) for a, b in zip(l, r)):
            return dict(status='SUPPORT_MISMATCH_NO_PAIRED_CONDITIONAL_CI', frames=n,
                policy='No favorable intersection; consult all-frame failure/coverage and full-GT PCK.')
    if not all(sum(map(len, x)) for x in left + right):
        return dict(status='NO_FINITE_CONDITIONAL_SUPPORT', frames=n)
    names = sorted(set(sessions)) if level == 'session' else list(range(n))
    lookup = {s: i for i, s in enumerate(names)}
    group = np.array([lookup[s] for s in sessions]) if level == 'session' else np.arange(n)
    ls = [B._prepare(x, group) for x in left]
    rs = [B._prepare(x, group) for x in right]
    units = len(names)
    def stat(c):
        return np.mean([B._median(a, c) - B._median(b, c) for a, b in zip(ls, rs)])
    observed = stat(np.ones(units, dtype=np.int64))
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.array([stat(rng.multinomial(units, np.full(units, 1/units))) for _ in range(resamples)])
    if not np.isfinite(draws).all():
        return dict(status='EMPTY_RESAMPLED_SUPPORT_NO_CI', frames=n,
            empty_draws=int((~np.isfinite(draws)).sum()), observed_delta=float(observed),
            policy='All draws retained as failure evidence; none dropped/replaced.')
    return dict(status='COMPLETE', **interval(draws, observed), frames=n, units=units,
        seeds=len(left), level=level, resamples=resamples, random_seed=BOOT_SEED,
        estimand='mean_seed(pooled median candidate - pooled median reference)',
        scope='Reused DEV; 13 sessions; unadjusted exploratory intervals; not independent confirmation.')


def paired_full_rates(left, right, sessions, numerator, denominator, resamples=RESAMPLES):
    if len(right) == 1:
        right = right * len(left)
    names = sorted(set(sessions))
    index = np.array([names.index(s) for s in sessions])
    def grouped(rows, fn):
        return np.bincount(index, weights=[fn(r) for r in rows], minlength=len(names))
    ln = np.array([grouped(r, numerator) for r in left])
    rn = np.array([grouped(r, numerator) for r in right])
    ld = np.array([grouped(r, denominator) for r in left])
    rd = np.array([grouped(r, denominator) for r in right])
    assert np.array_equal(ld, rd), 'Full denominators differ'
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(len(names), np.full(len(names), 1/len(names)), size=resamples)
    draws = ((counts @ ln.T) / (counts @ ld.T) - (counts @ rn.T) / (counts @ rd.T)).mean(1)
    observed = (ln.sum(1)/ld.sum(1) - rn.sum(1)/rd.sum(1)).mean()
    return dict(status='COMPLETE', **interval(draws, observed), level='session',
        frames=len(sessions), units=len(names), resamples=resamples, random_seed=BOOT_SEED,
        estimand='mean_seed(full-population rate candidate - rate reference)')


def paired_results(stores, poses, rows, B):
    sessions = [r['session_id'] for r in rows]
    ids = [r['frame_id'] for r in rows]
    results = {}
    for left_family, right_family in [('P', 'RESNET18'), ('D', 'RESNET18'), ('P', 'D')]:
        la = [left_family+str(s) for s in (1, 2, 3)]
        ra = ['RESNET18'] if right_family == 'RESNET18' else [right_family+str(s) for s in (1, 2, 3)]
        left = [stores[a] for a in la]
        right = [stores[a] for a in ra]
        entry = dict(conditional_keypoint_median={level: paired_medians(
            [[r['errors'] for r in rr] for rr in left],
            [[r['errors'] for r in rr] for rr in right], sessions, B, level=level)
            for level in ('session', 'frame')})
        entry['ALL_GT_PCK'] = {str(t): paired_full_rates(left, right, sessions,
            lambda r, key=str(t): r['canonical_hits'][key], lambda r: r['gt_count']) for t in (5, 10, 20)}
        entry['pose'] = {}
        for metric in ('translation_error_cm', 'rotation_error_deg', 'yaw_error_deg', 'iou3d'):
            pl, pr = [], []
            for aa, result in ((la, pl), (ra, pr)):
                for arm in aa:
                    lookup = {r['frame_id']: r[metric] for r in poses[arm]['rows']}
                    result.append([[lookup[fid]] if fid in lookup else [] for fid in ids])
            entry['pose'][metric] = paired_medians(pl, pr, sessions, B)
        def coverage_rows(aa):
            out = []
            for arm in aa:
                available = {r['frame_id'] for r in poses[arm]['rows']}
                out.append([dict(available=int(fid in available)) for fid in ids])
            return out
        entry['pose']['coverage'] = paired_full_rates(coverage_rows(la), coverage_rows(ra), sessions,
            lambda r: r['available'], lambda r: 1)
        results[left_family + '_minus_' + right_family] = entry
    return results


def mean_seed_results(methods):
    result = {}
    for family in ('D', 'P'):
        values = [methods[family+str(s)] for s in (1, 2, 3)]
        mean = lambda vs: float(np.mean(vs)) if all(v is not None for v in vs) else None
        result[family] = dict(seed_policy='mean of per-seed statistics; not an ensemble',
            **{key: mean([v[key] for v in values]) for key in ('median_px', 'p90_px', 'matched_frames',
                'failed_or_excluded_frames', 'supervised_points', 'missing_supervised_keypoints')},
            ALL_GT_PCK={str(t): mean([v['ALL_GT_PCK'][str(t)] for v in values]) for t in (5, 10, 20)},
            pose={key: mean([v['pose'].get(key) for v in values]) for key in ('rotation_median_deg',
                'translation_median_cm', 'yaw_median_deg', 'translation_p90_cm', 'rotation_p90_deg',
                'yaw_p90_deg', 'iou3d_median', 'add_sym_auc', 'coverage', 'failure_count')})
    return result


def write_frame_csv(rows, stores, poses):
    destination = RAW / 'DEV_FRAME_RESULTS.csv'
    assert not destination.exists()
    fields = ['arm', 'frame_id', 'session_id', 'image_key', 'canonical_matched', 'box_iou',
        'gt_count', 'finite_predicted_keypoints', 'missing_supervised_keypoints',
        'canonical_hits5', 'canonical_hits10', 'canonical_hits20',
        'pose_available', 'translation_error_cm', 'rotation_error_deg', 'yaw_error_deg', 'iou3d']
    fields += [f'point{k}_error_px' for k in range(9)]
    with destination.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            lookup = {r['frame_id']: r for r in poses[arm]['rows']}
            for record in stores[arm]:
                out = {k: record[k] for k in fields if k in record}
                out['arm'] = arm
                out.update({f'canonical_hits{t}': record['canonical_hits'][str(t)] for t in (5, 10, 20)})
                p = lookup.get(record['frame_id'])
                out['pose_available'] = p is not None
                for k in ('translation_error_cm', 'rotation_error_deg', 'yaw_error_deg', 'iou3d'):
                    out[k] = p[k] if p is not None else None
                out.update({f'point{k}_error_px': v for k, v in enumerate(record['point_errors'])})
                writer.writerow(out)


def score(path):
    payload, rows, _, targets, pose_ids, mapping, sources, modules = load_inputs(path)
    E, _, _, _, B = modules
    input_binding = binding(path)
    write(DOC / 'DEV_EVALUATION_LOCK.json', dict(complete=True, code=binding(__file__),
        predictions=input_binding, sources=sources, frame_mapping=mapping,
        positive_frames=319, sessions=13, supervised_keypoints=2818,
        new_image_forwards=0, negative_images_evaluated=0, RESNET18_AP_claim=False,
        bootstrap=dict(resamples=RESAMPLES, seed=BOOT_SEED, secondary_unadjusted=True),
        pose_policy='Unchanged canonical MAIN selector and SQPnP/RefineLM; explicit object contract; coverage denominator319; no silent intersection.'))
    methods, stores = {}, {}
    for arm in ARMS:
        methods[arm], stores[arm] = summarize_2d(E, rows, targets, arm)
    support = [[r['canonical_matched'] for r in stores[a]] for a in ARMS]
    assert all(x == support[0] for x in support), 'Fixed box/missing-mask contract did not preserve conditional support'
    poses = canonical_pose(rows, pose_ids, modules, input_binding, mapping)
    for arm in ARMS:
        methods[arm]['pose'] = poses[arm]['summary']
    paired = paired_results(stores, poses, rows, B)
    historical = read(HISTORICAL)
    assert historical['complete'] and historical['role'] == 'REUSED_DEV'
    historical_names = ['R0'] + [a+str(s) for a in ('P', 'D', 'L', 'PRIOR') for s in (1, 2, 3)]
    historical_rows = {name: dict(metrics=historical['methods'][name], sources=historical['sources'][name],
        status='HISTORICAL_YOLO_ESTIMATOR_RESULT_NOT_CURRENT_RESNET18_OUTPUT') for name in historical_names}
    for item in historical_rows.values():
        for b in item['sources']:
            verify_binding(b)
    write_frame_csv(rows, stores, poses)
    full_precision = {a: [dict(r, errors=r['errors'].tolist(), errors8=r['errors8'].tolist())
        for r in stores[a]] for a in ARMS}
    write(RAW / 'DEV_FULL_PRECISION_ERRORS.json', full_precision)
    result = dict(complete=True, schema='resnet18_refiner_dev_results_v1', role='REUSED_DEV',
        code=binding(__file__), evaluation_lock=binding(DOC / 'DEV_EVALUATION_LOCK.json'),
        baseline=payload['baseline'], baseline_protocol=payload['baseline_protocol'],
        baseline_training_complete=payload['baseline_training_complete'],
        predictions=input_binding, methods=methods, seed_mean=mean_seed_results(methods),
        positive_frames=319, sessions=13, gt_denominator=2818, same_conditional_support=True,
        negative_images_evaluated=0, RESNET18_AP_claim=False, new_image_forwards=0,
        geometry_derived_reference_not_independent_physical_metrology=True,
        primary_pose_path='MAIN', diagnostic_oracle_evaluated=False,
        canonical_pose_output_metadata_note='Unchanged canonical prediction-only selector/solve/metric primitives, new wrapper with explicit seven-arm naming and all319 success/failure rows. Zero forwards/training inside evaluation.',
        pose_artifacts={a:poses[a]['artifact'] for a in ARMS},
        historical_yolo_results=dict(source=binding(HISTORICAL), methods=historical_rows),
        frame_results=binding(RAW / 'DEV_FRAME_RESULTS.csv'),
        full_precision_errors=binding(RAW / 'DEV_FULL_PRECISION_ERRORS.json'),
        source_bindings=sources)
    for value in sources:
        verify_binding(value)
    assert binding(path) == input_binding
    write(DOC / 'DEV_PAIRED_RESULTS.json', dict(complete=True, results=paired,
        predictions=input_binding, code=binding(__file__), multiplicity='Unadjusted exploratory intervals',
        strict_support_no_silent_intersection=True))
    result['paired_results'] = binding(DOC / 'DEV_PAIRED_RESULTS.json')
    write(DOC / 'DEV_RESULTS.json', result)
    print(json.dumps(dict(PASS=True, frames=319, arms=7, csv_rows=319*7,
        outputs=[binding(DOC / n) for n in ('DEV_RESULTS.json', 'DEV_PAIRED_RESULTS.json')])), flush=True)


def selfcheck():
    E, M, _, G, B = canonical_modules()
    points = np.array([[1.,1.],[9.,1.],[9.,9.],[1.,9.],[2.,2.],[8.,2.],[8.,8.],[2.,8.],[5.,5.]])
    rows, targets = [], {}
    for index in range(4):
        p = points.copy()
        if index == 1:
            p[2] = np.nan
        if index == 2:
            p[:] = np.nan
        if index == 3:
            p[0] = [-4., 1.]  # finite out-of-frame coordinates remain valid
        valid = np.isfinite(p).all(-1)
        box = np.r_[p[:8][valid[:8]].min(0),p[:8][valid[:8]].max(0)].tolist() if valid[:8].sum() >= 3 else None
        q = p.copy(); q[:8][valid[:8]] += .1
        row = dict(image_key=f'synthetic/row{index}.png', frame_id=str(index), session_id=f's{index//2}',
            original_hw=[20,20], base_points=p.tolist(), point_valid=valid.tolist(),
            box_original=box, score=1., refined={a:q.tolist() for a in REFINERS})
        rows.append(row)
        targets[str(index)] = E.PositiveTarget(str(index),np.array([1.,1.,9.,9.]),points,
            np.ones(9,bool),np.ones(9,bool),np.full(9,2),np.eye(3),None,())
    validate_rows(rows)
    base, detail = summarize_2d(E,rows,targets,'RESNET18')
    assert base['gt_denominator'] == 36 and base['matched_frames'] == 2
    assert base['supervised_points'] == 18 and base['missing_supervised_keypoints'] == 10
    assert detail[1]['canonical_hits']['10'] == 0 and detail[1]['partial_point_hits']['10'] == 8
    assert detail[2]['partial_point_hits']['10'] == 0
    assert detail[3]['finite_predicted_keypoints'] == 9
    import copy
    broken = copy.deepcopy(rows); broken[0]['refined']['P1'][8][0] += 1
    try:
        validate_rows(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed center must fail')
    left = [[[2.],[3.]], [[3.],[4.]], [[4.],[5.]]]
    right = [[[1.],[2.]]]
    result = paired_medians(left,right,['a','b'],B,resamples=100)
    assert result['delta'] == result['low'] == result['high'] == 2.
    bad = paired_medians([[[2.],[]]],right,['a','b'],B,resamples=100)
    assert bad['status'].startswith('SUPPORT_MISMATCH')
    rates = paired_full_rates([[dict(hit=1,den=1),dict(hit=0,den=1)]],
        [[dict(hit=0,den=1),dict(hit=0,den=1)]], ['a','b'],
        lambda r:r['hit'], lambda r:r['den'], resamples=100)
    assert rates['delta'] == .5
    import cv2
    camera = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    spec = dict(long_m=1.3,short_m=1.1,height_m=.11)
    xyz = M.cuboid(spec['long_m'],spec['height_m'],spec['short_m'])
    projected, _ = cv2.projectPoints(np.r_[xyz,np.zeros((1,3))],
        np.array([.35,.2,.05]),np.array([.1,.15,3.]),camera,None)
    projected = projected.reshape(9,2)
    solved = solve_without_gt(projected,camera,spec,M,G)
    assert solved['status'] == 'OK', solved
    model = (M.cuboid(1.3,.11,1.1) if solved['chosen'] == M.CF_WIDTH else M.cuboid(1.1,.11,1.3))
    direct = M.solve(model,projected[:8],camera,np.ones(8,bool))
    assert np.array_equal(solved['rotation'],direct[0])
    assert np.array_equal(solved['translation'],direct[1])
    unavailable = solve_without_gt(np.full((9,2),np.nan),camera,spec,M,G)
    assert unavailable['status'] == 'FEWER_THAN_SIX_FINITE_CORNERS'
    empty = pose_summary([],319)
    assert empty['coverage'] == 0 and empty['failure_count'] == 319 and empty['translation_median_cm'] is None
    assert all(empty[k] is None for k in ('translation_p90_cm','rotation_p90_deg','yaw_p90_deg'))
    toy_pose = [dict(translation_error_cm=t,rotation_error_deg=2*t,yaw_error_deg=3*t,
        iou3d=.5,add_sym_m=.01,diameter_m=1.,axis_correct=True) for t in (1.,10.)]
    tail = pose_summary(toy_pose,319)
    np.testing.assert_allclose([tail['translation_p90_cm'],tail['rotation_p90_deg'],tail['yaw_p90_deg']],
        [9.1,18.2,27.3],rtol=0,atol=1e-14)
    assert tail['denominator'] == 319 and tail['failure_count'] == 317
    assert 'torch' not in sys.modules, 'CPU evaluator must not import Torch'
    print(json.dumps(dict(PASS=True, test_scope='invented inputs only', new_images=0,
        model_construction=0, forwards=0, fits=0, missing_and_full_denominator=True,
        center_drift_rejected=True, finite_out_of_frame_retained=True,
        paired_constant_shift_exact=True, support_mismatch_refused=True,
        canonical_pose_primitive_exact=True, zero_pose_coverage_retained=True,
        translation_rotation_yaw_P90=True)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['selfcheck', 'score'])
    p.add_argument('--predictions', type=Path, default=RAW / 'DEV_PREDICTIONS.json')
    args = p.parse_args()
    selfcheck() if args.command == 'selfcheck' else score(args.predictions)


if __name__ == '__main__':
    main()
