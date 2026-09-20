"""Bounded, two-stage filtered real adaptation of a COPY of frozen N2.

R0 cache -> confidence filter -> N2 normal/flip -> geometry filter -> training.
All artifacts go to new roots. No paper locks, labels, or source weights change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments/pallet_real_refiner_twostage_v1'
RAW = ROOT / 'data/pallet/results/pallet_real_refiner_twostage_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/research/pallet_dim_conditioned_p_v1'))
sys.path.insert(1, str(ROOT / 'scripts/self_training_yolo'))
import dcp_env as E
from refiner import model, forward, decode, context, train_loss
from inference import load_head, predict_captured, registry_input, serial
from scripts.evaluation import final_dimension_release as F

CACHE = ROOT / 'data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json'
FILTER = ROOT / 'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json'
GROUPS = ROOT / 'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
GREEN = F.DOC / 'green150_saved_labels_v1/DATASET_SNAPSHOT.json'
ARMS = ('SYN_ONLY', 'REAL_RAW', 'REAL_REFINED')
KEYS = ('p3', 'p4', 'points', 'boxes', 'point_valid', 'input_shape', 'context')
STEPS = 1500


def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def freeze(path, value):
    if path.exists():
        assert E.read(path) == value, ('Immutable artifact differs', str(path))
    else:
        write(path, value)


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending.pt')
    torch.save(value, temp)
    temp.replace(path)


def status(stage, **fields):
    row = dict(time=E.now(), stage=stage, **fields)
    write(DOC / 'STATUS.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def stage1(entry, lock):
    """Only raw R0 values are legal inputs to the first filter."""
    top = entry.get('top1')
    if top is None:
        return False
    p = np.asarray(top['keypoints_xy'])
    v = np.asarray(top['keypoints_conf']) >= lock['keypoint_validity']['kp_conf_threshold']
    v &= np.isfinite(p).all(-1)
    return bool(np.isfinite(top['box_conf']) and top['box_conf'] >= lock['TAU_BOX']
                and v[:8].sum() >= lock['keypoint_validity']['min_valid_corners'])


def stage2(scores, lock):
    thresholds = lock['geometry_thresholds']
    return all(scores.get(k) is not None and np.isfinite(scores[k])
               and scores[k] <= thresholds[t]
               for k, t in [('s_remove', 'tau_remove'), ('s_flip', 'tau_flip')])


def filtered_call(entry, lock, refine):
    """Regression-testable ordering: a rejected raw image never calls N2."""
    return refine(entry) if stage1(entry, lock) else None


def unflip(points, confidences, width, permutation):
    p = np.array(points, copy=True)
    p[:, 0] = width - 1 - p[:, 0]
    return p[permutation], np.asarray(confidences)[permutation]


def jitter(batch, generator):
    """Input-only perturbation; pseudo/GT targets and centroid never change."""
    out = dict(batch)
    points = batch['points'].clone()
    boxes = batch['boxes'].to(points)
    diag = (boxes[:, 2:] - boxes[:, :2]).norm(dim=-1)
    sigma = (.003 * diag).clamp(max=2.)
    noise = torch.randn(points[:, :8].shape, generator=generator,
                        device=points.device, dtype=points.dtype).clamp(-2, 2)
    mask = batch['point_valid'][:, :8, None]
    points[:, :8] = torch.where(mask, points[:, :8] + noise * sigma[:, None, None], points[:, :8])
    out['points'] = points
    return out


def prepare():
    lock = F.checked_lock()
    teacher = next(r['checkpoint'] for r in lock['heads'] if r['arm'] == 'N2_DIM_ONLY' and r['seed'] == 1)
    filters = E.read(FILTER)
    cache = E.read(CACHE)
    assert cache['teacher_sha256'] == E.R0_SHA
    assert E.sha(ROOT / cache['pool_manifest']) == cache['pool_manifest_sha256']
    assert E.sha(CACHE) == filters['teacher_cache_sha256']
    # Metadata only, not annotation coordinates or evaluation errors.
    dev = E.read(E.DOC / 'DEV_CACHE_COMPLETE.json')
    green = E.read(GREEN)
    baseline = E.read(E.LINE / 'baseline/FULL_CANDIDATES.json')
    eval_hashes = {r['image_sha256'] for r in baseline['frame_metadata'].values()}
    eval_hashes |= {r['image']['sha256'] for r in green['records']}
    eval_paths = [r['key'] for r in dev['records']] + [r['image']['path'] for r in green['records']]
    groups = E.read(GROUPS)
    key_group = {s['session_key']: g['recording_id'] for g in groups['groups']
                 if not g['is_collection'] for s in g['sessions']}
    eval_group_ids = {gid for key, gid in key_group.items()
                      if any(Path(p).is_relative_to(key) for p in eval_paths)}
    rejected, eligible, accepted = [], [], []
    for entry in cache['entries']:
        session_root = str(Path(entry['image_path']).parent.parent)
        gid = key_group.get(session_root)
        assert gid is not None, ('Unknown recording group', session_root)
        if entry['image_sha256'] in eval_hashes or gid in eval_group_ids:
            rejected.append(dict(image=entry['image_path'], reason='evaluation identity/recording'))
            continue
        # Partial-overlap recording aliases are also prohibited.
        aliases = {k for k, v in key_group.items() if v in eval_group_ids}
        overlap = any((r['session_a'] == session_root and r['session_b'] in aliases)
                      or (r['session_b'] == session_root and r['session_a'] in aliases)
                      for r in groups['partial_overlap_pairs'])
        if overlap:
            rejected.append(dict(image=entry['image_path'], reason='partial recording overlap'))
            continue
        eligible.append(entry)
        if stage1(entry, filters):
            assert E.sha(ROOT / entry['image_path']) == entry['image_sha256']
            accepted.append(entry)
    assert accepted
    protocol = dict(
        experiment='pallet_real_refiner_twostage_v1', status='SINGLE_SEED_DEVELOPMENT_SCREEN',
        user_order='raw R0 -> first filter -> N2 only on survivors -> second filter -> refiner training',
        teacher=teacher, model_lock=E.bound(F.DOC / 'MODEL_LOCK.json'),
        bindings=[E.bound(p) for p in [Path(__file__), HERE / 'test_run.py', CACHE, FILTER, GROUPS, GREEN,
            E.DOC / 'DEV_CACHE_COMPLETE.json', ROOT / 'scripts/self_training_yolo/pseudo_label_filters.py',
            ROOT / 'scripts/self_training_yolo/build_pseudo_manifests.py',
            ROOT / 'scripts/annotate/annotate_pnp.py',
            ROOT / 'scripts/evaluation/green_saved_labels_v1.py']],
        stage1='raw highest-box-confidence >= 0.85 and >= 6/8 finite corners with kp confidence >= 0.5',
        stage2='refined single-keypoint-removal <= 0.05 AND refined unflipped consistency <= 0.05; projected-diagonal normalization; existing F4 thresholds',
        reprojection_score='diagnostic only; no additional threshold',
        filters_reselected=False, pseudo_refresh=False, pseudo_gt_used=False,
        pool_object='plastic_standard_110x130x11', canonical_WDH_m=[1.1, 1.3, .11],
        dimensions_provenance='existing build_pseudo_manifests.POOL_OBJECT_TYPE -> registry; no per-image GT dimensions',
        camera='raw session cam_K, verified against old R0 cache; only filter uses intrinsics',
        baseline='existing N2_DIM_ONLY seed1; immutable', train_arms=list(ARMS),
        ablation='REAL_RAW and REAL_REFINED have identical final-survivor image membership, valid mask, views, synthetic replay, update order; only target coordinates differ. Does not isolate second-filter selection effect.',
        init='all train arms start from existing N2 seed1, not random init', seed=1,
        steps=STEPS, batch=16, source_slots=8, replacement_slots=8,
        source_replay=1024, real_pool='all stage2 survivors; no outcome-based resampling or minimum-quantity gate',
        optimizer=dict(name='AdamW', lr=1e-4, final_lr=1e-5, betas=[.9, .999], weight_decay=1e-4, gradient_clip=5.),
        loss='unchanged Gaussian candidate cross entropy, mean of source and replacement losses, weights 0.5 each',
        jitter='input corners only: Gaussian sigma=min(0.003*bbox_diagonal,2 input pixels), clipped at 2 sigma; same for all arms',
        real_views='two deterministic geometry-preserving RGB channel gain/offset views per accepted image; teacher uses unaugmented image; student uses one cached augmented view per exposure',
        photometric=dict(channel_gain=[.8, 1.2], additive_offset=[-12., 12.]),
        synthetic_views='existing original source features plus same coordinate jitter in both synthetic halves',
        checkpoint='last1500 only; no DEV-based early stopping or hyperparameter search',
        inference='R0 remains frozen; N2 dimensions/temperature/lambda/cap unchanged; no jitter/augmentation at inference',
        metrics='DEV319 and GREEN150 manual-only primary; green all-known proxy secondary; E_sym, median/P90, PCK10, good5_to_bad10',
        signal_rule='REAL_REFINED lower E_sym and non-worse P90/PCK10 vs both original N2 and SYN_ONLY in BOTH populations, zero good5_to_bad10 vs original; otherwise mixed/no-positive-screen. REAL_RAW is separate correction ablation.',
        green_excluded_from_training=True, independent_test=False, paper_model_auto_promotion=False,
        cuda_only=True, thermal_limit_C=80, system_changes=False,
    )
    freeze(DOC / 'PROTOCOL.json', protocol)
    freeze(DOC / 'STAGE1.json', dict(raw_images=len(cache['entries']), eligible_images=len(eligible),
        excluded=rejected, first_filter_pass=len(accepted), counts=dict(Counter(r['paper_condition'] for r in accepted)),
        sessions=dict(Counter(r['capture_session'] for r in accepted)), eval_recording_groups=sorted(eval_group_ids),
        no_eval_hash_overlap=not ({r['image_sha256'] for r in accepted} & eval_hashes), records=accepted))
    status('FIRST_FILTER_COMPLETE', raw=len(cache['entries']), accepted=len(accepted), counts=Counter(r['paper_condition'] for r in accepted))


def verify_protocol():
    protocol = E.read(DOC / 'PROTOCOL.json')
    F.checked_lock()
    for b in protocol['bindings'] + [protocol['teacher'], protocol['model_lock']]:
        F.verify(b)
    return protocol


def camera_check(entry):
    folder = (ROOT / entry['image_path']).parent.parent
    paths = [p for p in [folder / 'cam_K.txt', folder / 'camera_intrinsics.txt'] if p.exists()]
    assert paths, ('No raw camera file', str(folder))
    k = np.loadtxt(paths[0]).reshape(3, 3)
    assert np.allclose(k, entry['camera_matrix'], atol=1e-5, rtol=0)
    return k, E.bound(paths[0])


def input_batch(captured, dimensions, order, norm):
    inputs = E.old('features').branch_inputs(captured)
    assert inputs is not None
    batch = {k: torch.as_tensor(inputs[k], device='cuda')[None] for k in KEYS if k not in ('p3', 'p4', 'context')}
    batch.update(p3=captured['p3'], p4=captured['p4'],
                 context=torch.as_tensor(context(np.array(dimensions)[None], [order], norm, False), device='cuda'))
    return batch, inputs


@torch.no_grad()
def pseudo_cache():
    protocol = verify_protocol()
    final_path = DOC / 'STAGE2.json'
    if final_path.exists():
        for b in E.read(final_path)['accepted']:
            F.verify(b['cache'])
        return
    E.gpu()
    filters = E.read(FILTER)
    first = E.read(DOC / 'STAGE1.json')
    teacher, _ = load_head('N2_DIM_ONLY', 1)
    teacher_state = E.state_sha(teacher.state_dict())
    extractor = E.old('features').FrozenYoloFeatures(E.R0)
    norm = E.read(E.DOC / 'DIM_NORMALIZATION_LOCK.json')
    settings = F.checked_lock()
    dims, order = registry_input(protocol['pool_object'])
    from pseudo_label_filters import geometry_scores
    from build_pseudo_manifests import POOL_OBJECT_TYPE
    assert POOL_OBJECT_TYPE == protocol['pool_object'] and dims.tolist() == protocol['canonical_WDH_m']
    dimensions_xyz = dict(x=dims[0], y=dims[2], z=dims[1])
    perm = E.read(CACHE)['flip_contract']['flip_idx']
    records, accepted = [], []
    try:
        for i, entry in enumerate(first['records']):
            assert stage1(entry, filters)
            record_path = RAW / 'filter_records' / (entry['image_sha256'] + '.json')
            if record_path.exists():
                record = E.read(record_path)
                if record['accepted']:
                    F.verify(record['cache']); accepted.append(record)
                records.append(record)
                continue
            assert E.sha(ROOT / entry['image_path']) == entry['image_sha256']
            image = cv2.imread(str(ROOT / entry['image_path']))
            assert image is not None
            K, kb = camera_check(entry)
            cap = extractor.predict(image)
            idx = cap['selected_index']; assert idx is not None
            top = cap['candidates'][idx]
            assert abs(top['score'] - entry['top1']['box_conf']) < 1e-6
            assert np.allclose(top['keypoints_xy'], entry['top1']['keypoints_xy'], atol=1e-5, rtol=0)
            batch, inputs = input_batch(cap, dims, order, norm)
            refined, _ = predict_captured(teacher, 'N2_DIM_ONLY', cap, dims, order,
                settings['temperatures']['N2_DIM_ONLY_seed1'], settings['decode_rule'], image.shape[:2], norm)
            q = np.array(refined['candidates'][idx]['keypoints_xy'])
            flip_cap = extractor.predict(cv2.flip(image, 1))
            flip_pred, _ = predict_captured(teacher, 'N2_DIM_ONLY', flip_cap, dims, order,
                settings['temperatures']['N2_DIM_ONLY_seed1'], settings['decode_rule'], image.shape[:2], norm)
            flip_q = flip_valid = None
            if flip_pred['selected_index'] is not None:
                ft = flip_pred['candidates'][flip_pred['selected_index']]
                flip_q, fc = unflip(ft['keypoints_xy'], ft['keypoints_conf'], image.shape[1], perm)
                flip_valid = (fc >= filters['keypoint_validity']['kp_conf_threshold']) & np.isfinite(flip_q).all(-1)
            valid = (np.asarray(top['keypoints_conf']) >= filters['keypoint_validity']['kp_conf_threshold']) & np.isfinite(q).all(-1)
            try:
                scores = geometry_scores(q, valid, K, dimensions_xyz, flip_q, flip_valid)
                scores = {k: (float(scores[k]) if scores[k] is not None and np.isfinite(scores[k]) else None)
                          for k in ('s_reproj', 's_remove', 's_flip')}
                reason = None
            except cv2.error as exc:
                scores = dict(s_reproj=None, s_remove=None, s_flip=None)
                reason = str(exc)
            passed = stage2(scores, filters)
            record = {k: entry[k] for k in ('image_path', 'image_sha256', 'capture_session', 'paper_condition')}
            record.update(accepted=passed, scores=scores, camera=kb, error=reason,
                          raw_points=np.asarray(top['keypoints_xy']).tolist(), refined_points=q.tolist(),
                          valid=valid.tolist(), correction_px=np.linalg.norm(q[:8] - top['keypoints_xy'][:8], axis=-1).tolist())
            if passed:
                # Store target in the SAME input-pixel frame as original R0 points.
                target = inputs['points'] + (q - top['keypoints_xy']) * inputs['gain']
                assert np.array_equal(q[8], top['keypoints_xy'][8])
                original = {k: v[0].detach().cpu().clone() for k, v in batch.items()}
                views = []
                rng = np.random.default_rng(int(entry['image_sha256'][:8], 16))
                for _ in range(2):
                    gain = rng.uniform(.8, 1.2, size=(1, 1, 3))
                    offset = float(rng.uniform(-12, 12))
                    aug = np.clip(image.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
                    ac = extractor.predict(aug)
                    assert ac['input_shape'] == cap['input_shape']
                    views.append(dict(p3=ac['p3'][0].cpu(), p4=ac['p4'][0].cpu(),
                                      gain=gain.reshape(3).tolist(), offset=offset))
                dst = RAW / 'real_cache' / (entry['image_sha256'] + '.pt')
                save_tensor(dst, dict(original=original, views=views, refined_target=torch.tensor(target, dtype=torch.float32),
                    raw_target=original['points'].clone(), target_valid=torch.tensor(valid),
                    image_sha256=entry['image_sha256'], teacher=protocol['teacher'], GT_input=False))
                record['cache'] = E.bound(dst)
                accepted.append(record)
            freeze(record_path, record)
            records.append(record)
            if (i + 1) % 20 == 0:
                status('SECOND_FILTER', processed=i+1, total=len(first['records']), accepted=len(accepted), gpu=E.gpu())
    finally:
        extractor.close()
    assert E.state_sha(teacher.state_dict()) == teacher_state
    assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    freeze(final_path, dict(first_filter_pass=len(first['records']), second_filter_pass=len(accepted),
        counts=dict(Counter(r['paper_condition'] for r in accepted)),
        sessions=dict(Counter(r['capture_session'] for r in accepted)), accepted=accepted,
        records=records, teacher_unchanged=True, GT_input=False, stage_order_verified=True,
        raw_failed_images_refined=0, note='confidence scores are preserved, not recalibrated by the refiner'))
    assert accepted, 'No stage2 survivors: preserve diagnostics; do not relax thresholds'
    status('PSEUDO_CACHE_COMPLETE', accepted=len(accepted), counts=Counter(r['paper_condition'] for r in accepted))


def make_orders(data, nreal):
    rng = np.random.default_rng(1)
    pool = np.sort(rng.choice(data.train_rows, size=1024, replace=False))
    assert np.all(data.partitions[pool] == 'train')
    return dict(pool=pool, source=rng.choice(pool, size=(STEPS, 8)),
                replacement=rng.choice(pool, size=(STEPS, 8)),
                real=rng.integers(nreal, size=(STEPS, 8)), view=rng.integers(2, size=(STEPS, 8)))


def real_batch(caches, rows, views, target):
    chosen = [caches[int(i)] for i in rows]
    batch = {k: torch.stack([r['views'][int(v)][k] if k in ('p3', 'p4') else r['original'][k]
                             for r, v in zip(chosen, views)]).cuda() for k in KEYS}
    batch['gt_points'] = torch.stack([r[target] for r in chosen]).cuda()
    batch['gt_valid'] = torch.stack([r['target_valid'] for r in chosen]).cuda()
    return batch


def train():
    from data import PaperData
    protocol = verify_protocol()
    stage = E.read(DOC / 'STAGE2.json'); assert stage['second_filter_pass'] > 0
    data = PaperData()
    caches = []
    for r in stage['accepted']:
        F.verify(r['cache'])
        caches.append(torch.load(ROOT / r['cache']['path'], map_location='cpu', weights_only=False))
    orders = make_orders(data, len(caches))
    order_path = RAW / 'TRAIN_ORDER.npz'
    if not order_path.exists():
        np.savez(order_path, **orders)
    else:
        with np.load(order_path) as previous:
            assert all(np.array_equal(previous[k], v) for k, v in orders.items())
    freeze(DOC / 'TRAIN_INPUT_LOCK.json', dict(orders=E.bound(order_path), protocol=E.bound(DOC / 'PROTOCOL.json'),
        stage2=E.bound(DOC / 'STAGE2.json'), real_images=len(caches), source_unique=len(orders['pool']),
        train_only=True, synthetic_calibration_used=False,
        cache_manifest=E.bound(E.LINE / 'cache/CACHE_MANIFEST.json'),
        source_manifest=E.bound(E.LINE / 'SOURCE_MANIFEST.json'),
        dimension_sidecar=E.bound(E.RAW / 'DIMENSION_SIDECAR.npz')))
    original = torch.load(ROOT / protocol['teacher']['path'], map_location='cpu', weights_only=False)
    opt = protocol['optimizer']
    for arm in ARMS:
        destination = RAW / 'runs' / arm / 'last.pt'
        if destination.exists():
            ck = torch.load(destination, map_location='cpu', weights_only=False)
            assert ck['complete'] and ck['step'] == STEPS and ck['protocol_sha'] == E.sha(DOC / 'PROTOCOL.json')
            continue
        E.gpu(); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
        head = model('N2_DIM_ONLY', original['config']).cuda().train()
        head.load_state_dict(original['model_state_dict'])
        initial_sha = E.state_sha(head.state_dict())
        optimizer = torch.optim.AdamW(head.parameters(), lr=opt['lr'], betas=tuple(opt['betas']), weight_decay=opt['weight_decay'])
        gs = torch.Generator(device='cuda').manual_seed(1001)
        gr = torch.Generator(device='cuda').manual_seed(2001)
        begin = time.monotonic(); history = []; start_step = 0; prior = 0.
        resume = destination.parent / 'resume.pt'
        if resume.exists():
            ck = torch.load(resume, map_location='cpu', weights_only=False)
            assert ck['protocol_sha'] == E.sha(DOC / 'PROTOCOL.json')
            assert ck['input_sha'] == E.sha(DOC / 'TRAIN_INPUT_LOCK.json')
            head.load_state_dict(ck['model_state_dict']); optimizer.load_state_dict(ck['optimizer'])
            gs.set_state(ck['source_rng']); gr.set_state(ck['replacement_rng'])
            start_step = ck['step']; history = ck['history']; prior = ck['elapsed_seconds']
        for i in range(start_step, STEPS):
            lr = opt['final_lr'] + .5 * (opt['lr'] - opt['final_lr']) * (1 + math.cos(math.pi * i / STEPS))
            for g in optimizer.param_groups: g['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            syn = jitter(data.batch(orders['source'][i], 'N2_DIM_ONLY'), gs)
            sl = train_loss(forward(head, syn), syn, False)
            assert torch.isfinite(sl); (.5 * sl).backward()
            if arm == 'SYN_ONLY':
                other = data.batch(orders['replacement'][i], 'N2_DIM_ONLY')
            else:
                target = 'raw_target' if arm == 'REAL_RAW' else 'refined_target'
                other = real_batch(caches, orders['real'][i], orders['view'][i], target)
            other = jitter(other, gr)
            rl = train_loss(forward(head, other), other, False)
            assert torch.isfinite(rl); (.5 * rl).backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), opt['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if i == start_step or (i+1) % 100 == 0:
                row = dict(step=i+1, source_loss=float(sl.detach()), replacement_loss=float(rl.detach()),
                    lr=lr, gradient_norm=float(norm), elapsed_seconds=prior+time.monotonic()-begin)
                history.append(row); status('TRAINING', arm=arm, total=STEPS, **row, gpu=E.gpu())
            if (i+1) % 250 == 0 or i+1 == STEPS:
                ck = dict(arm=arm, config=original['config'], complete=i+1 == STEPS, step=i+1,
                    model_state_dict=head.state_dict(), optimizer=optimizer.state_dict(),
                    source_rng=gs.get_state(), replacement_rng=gr.get_state(),
                    protocol_sha=E.sha(DOC / 'PROTOCOL.json'), input_sha=E.sha(DOC / 'TRAIN_INPUT_LOCK.json'),
                    teacher=protocol['teacher'], history=history, elapsed_seconds=prior+time.monotonic()-begin)
                save_tensor(destination if ck['complete'] else resume, ck)
        assert E.state_sha(head.state_dict()) != initial_sha
        freeze(DOC / f'FIT_{arm}.json', dict(complete=True, steps=STEPS, seed=1,
            checkpoint=E.bound(destination), elapsed_seconds=ck['elapsed_seconds'], history=history,
            source_exposures=STEPS*8, replacement_exposures=STEPS*8,
            only_refiner_trainable=True, R0_not_instantiated_in_training=True))
        del head, optimizer, syn, other, sl, rl, ck
        torch.cuda.empty_cache()
    F.checked_lock()
    status('TRAINING_COMPLETE', fits=3, updates=STEPS*3)


def heads_for_eval():
    heads = {'N2_ORIGINAL': load_head('N2_DIM_ONLY', 1)[0]}
    for arm in ARMS:
        ck = torch.load(RAW / 'runs' / arm / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['complete'] and ck['step'] == STEPS
        head = model('N2_DIM_ONLY', ck['config']).cuda().eval()
        head.load_state_dict(ck['model_state_dict']); head.requires_grad_(False)
        heads[arm] = head
    return heads


@torch.no_grad()
def evaluate():
    verify_protocol(); E.gpu()
    from dev_evaluate import population_metadata, iou
    from eval_math import measure, summary, damage
    from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
    heads = heads_for_eval(); settings = F.checked_lock()
    norm = E.read(E.DOC / 'DIM_NORMALIZATION_LOCK.json')
    groups = {r['object_type']: r for r in E.read(E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    all_scores = {}; all_predictions = {}; preserves = 0

    def run_cap(captured, dimensions, order, hw):
        nonlocal preserves
        out = {'R0': dict(candidates=serial(captured['candidates']), selected_index=captured['selected_index'])}
        for name, head in heads.items():
            p, _ = predict_captured(head, 'N2_DIM_ONLY', captured, dimensions, order,
                settings['temperatures']['N2_DIM_ONLY_seed1'], settings['decode_rule'], hw, norm)
            out[name] = p; preserves += 1
        return out

    pe, pop = population_metadata()
    targets = {}
    for item, meta in pop:
        t = pe.E._legacy_forbidden_target(item)
        targets[item.frame_id] = dict(gt=np.array(t.keypoints_xy), valid=np.array(t.keypoint_supervision_mask),
                                      box=np.array(t.box_xyxy), meta=meta)
    scores = {k: [] for k in ['R0', *heads]}; predictions = {k: [] for k in scores}
    old = {r['id']: r for r in E.read(E.RAW / 'predictions/REAL_DEV/N2_DIM_ONLY_seed1.json')['records']}
    for i, record in enumerate(E.read(E.DOC / 'DEV_CACHE_COMPLETE.json')['records']):
        F.verify(record)
        row = torch.load(ROOT / record['path'], map_location='cpu', weights_only=False)
        cap = row['captured']
        for k in ('p3', 'p4'): cap[k] = cap[k].cuda()
        preds = run_cap(cap, row['dimensions'], row['order'], row['raw_hw'])
        target = targets[row['id']]; meta = target['meta']
        for name, p in preds.items():
            idx = p['selected_index']; candidate = None if idx is None else p['candidates'][idx]
            if name == 'N2_ORIGINAL' and idx is not None:
                assert np.allclose(candidate['keypoints_xy'], old[row['id']]['candidates'][idx]['keypoints_xy'], atol=1e-5, rtol=0)
            matched = candidate is not None and iou(candidate['box_xyxy'], target['box']) >= .5
            points = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
            metric = measure(points, target['gt'], target['valid'], groups[meta['object_type']]['permutations'], row['raw_hw'], matched, idx is not None)
            scores[name].append(dict(id=row['id'], session=meta['session_id'], **metric))
            predictions[name].append(dict(id=row['id'], **p))
        if (i+1) % 80 == 0: status('DEV_EVALUATION', done=i+1, total=319, gpu=E.gpu())
    all_scores['DEV319'] = scores; all_predictions['DEV319'] = predictions

    green = E.read(GREEN)
    scores = {mode: {k: [] for k in ['R0', *heads]} for mode in ['GREEN150_MANUAL', 'GREEN150_ALL_KNOWN_PROXY']}
    predictions = {k: [] for k in ['R0', *heads]}
    extractor = E.old('features').FrozenYoloFeatures(E.R0)
    oldgreen = E.read(F.RAW / 'green150_saved_labels_v1/PREDICTIONS.json')['predictions']['N2_DIM_ONLY_seed1']
    oldgreen = {r['id']: r for r in oldgreen}
    try:
        for i, row in enumerate(green['records']):
            F.verify(row['image']); F.verify(row['annotation'])
            im = cv2.imread(str(ROOT / row['image']['path'])); assert im is not None
            cap = extractor.predict(im)
            preds = run_cap(cap, row['canonical_WDH_m'], 4, im.shape[:2])
            doc = E.read(ROOT / row['annotation']['path'])
            gt, all_valid = annotation_arrays(doc)
            _, manual = annotation_arrays(doc, True)
            h, w = im.shape[:2]
            inside = all_valid & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
            box = np.r_[gt[inside].min(0), gt[inside].max(0)]
            for name, p in preds.items():
                idx = p['selected_index']; candidate = None if idx is None else p['candidates'][idx]
                if name == 'N2_ORIGINAL' and idx is not None:
                    assert np.allclose(candidate['keypoints_xy'], oldgreen[row['id']]['candidates'][idx]['keypoints_xy'], atol=1e-5, rtol=0)
                matched = candidate is not None and iou(candidate['box_xyxy'], box) >= .5
                points = np.full((9, 2), np.nan) if candidate is None else candidate['keypoints_xy']
                for mode, valid in [('GREEN150_MANUAL', manual), ('GREEN150_ALL_KNOWN_PROXY', all_valid)]:
                    metric = measure(points, gt, valid, groups['plastic_standard_110x110x15']['permutations'], im.shape[:2], matched, idx is not None)
                    scores[mode][name].append(dict(id=row['id'], session=row['session'], **metric))
                predictions[name].append(dict(id=row['id'], **p))
            if (i+1) % 30 == 0: status('GREEN_EVALUATION', done=i+1, total=150, gpu=E.gpu())
    finally:
        extractor.close()
    all_scores.update(scores); all_predictions['GREEN150'] = predictions
    results = {}
    for dataset, rows in all_scores.items():
        summaries = {k: summary(v) for k, v in rows.items()}
        contrasts = {}
        for name in ARMS:
            contrasts[name] = {}
            for reference in ('N2_ORIGINAL', 'SYN_ONLY', 'REAL_RAW'):
                contrasts[name][reference] = dict(delta_E_sym=summaries[name]['E_sym']-summaries[reference]['E_sym'],
                                                  **damage(rows[reference], rows[name]))
        results[dataset] = dict(summary=summaries, contrasts=contrasts)
    freeze(RAW / 'PREDICTIONS.json', all_predictions)
    freeze(RAW / 'PER_FRAME_METRICS.json', all_scores)
    positive = True
    for dataset in ('DEV319', 'GREEN150_MANUAL'):
        r = results[dataset]; a = r['summary']['REAL_REFINED']
        for ref in ('N2_ORIGINAL', 'SYN_ONLY'):
            b = r['summary'][ref]
            positive &= (a['E_sym'] < b['E_sym'] and a['matched_pooled_corner8_P90_px'] <= b['matched_pooled_corner8_P90_px']
                         and a['PCK']['10'] >= b['PCK']['10'])
        positive &= r['contrasts']['REAL_REFINED']['N2_ORIGINAL']['good5_to_bad10'] == 0
    freeze(DOC / 'RESULTS.json', dict(complete=True, results=results, seeds=1,
        verdict='POSITIVE_SCREEN_REQUIRES_REPLICATION' if positive else 'MIXED_OR_NO_POSITIVE_SCREEN',
        detector_preservation_checks=preserves, inference_only_coordinates_changed=True,
        negative_FP='Not rerun; R0/boxes/scores are unchanged by construction, so box-threshold FP cannot change',
        original_N2_prediction_parity=True, independent_confirmation=False, auto_promoted=False))
    F.checked_lock()
    report()
    status('COMPLETE', verdict=E.read(DOC / 'RESULTS.json')['verdict'], gpu=E.gpu())


def report():
    result = E.read(DOC / 'RESULTS.json'); first = E.read(DOC / 'STAGE1.json'); second = E.read(DOC / 'STAGE2.json')
    lines = ['# N2 실사 2단계 필터 적응 — 단일 seed 예비 실험', '',
        f"상태: {result['verdict']}. 기존 최종 모델/논문 표는 변경하지 않음.", '',
        f"R0 {first['raw_images']}장 → 원본 confidence 필터 {first['first_filter_pass']}장 → N2 보정 후 geometry/flip 필터 {second['second_filter_pass']}장.", '',
        f"최종 학습 풀: {second['counts']}; 세션: {second['sessions']}.", '',
        'R0와 pseudo teacher는 고정. 기존 N2 seed1에서 시작한 복사본 3개, 각 1,500 step / batch16. 실사 두 arm은 같은 이미지/마스크/증강/순서이며 target 좌표만 다름.', '',
        '실사는 기존 직사각 플라스틱 팔레트(1.1×1.3×0.11 m)이며, 초록 정사각 실사 학습은 아님. 초록 150장은 평가에만 사용.', '',
        '메트릭: 매칭된 코너 median/P90(px), 전체 eligible 코너 PCK10, 누락 패널티 포함 평균 E_sym. 작을수록 좋음(PCK10은 클수록 좋음).', '']
    for dataset, r in result['results'].items():
        lines += [f'## {dataset}', '', '| 모델 | median px | P90 px | PCK10 % | E_sym |', '|---|---:|---:|---:|---:|']
        for name, s in r['summary'].items():
            lines.append(f"| {name} | {s['matched_pooled_corner8_median_px']:.4f} | {s['matched_pooled_corner8_P90_px']:.4f} | {100*s['PCK']['10']:.3f} | {s['E_sym']:.7f} |")
        lines += ['', 'REAL_REFINED 비교:', '']
        for ref, c in r['contrasts']['REAL_REFINED'].items():
            lines.append(f"- vs {ref}: ΔE_sym={c['delta_E_sym']:+.8f}, 개선/악화 프레임 {c['improved_frames']}/{c['harmed_frames']}, good<5→bad>10 코너 {c['good5_to_bad10']}개.")
        lines.append('')
    lines += ['## 해석 범위', '',
        '- 단일 seed, 재사용 DEV에 대한 예비 결과. 독립 검증·통계적 유의성 주장 없음.',
        '- GREEN150 manual-only가 초록 주 지표, all-known은 PnP 생성점 포함 proxy. 평가 라벨의 기존 QA/카메라 불일치 한계 유지.',
        '- Teacher 예측은 정답이 아님. consistency와 학습 loss 개선만으로 좌표 정확도 개선을 주장하지 않음.',
        '- 기존 candidate CE를 보정 좌표로 학습하므로 soft distribution을 좌표로 압축한 데 따른 self-sharpening 효과도 포함됨.',
        '- 실사 arm은 photometric 증강+좌표 jitter, 합성 대조군은 기존 특징+동일 좌표 jitter. 합성 대비 차이는 이 적응 패키지 전체이며, 보정 좌표 고유 효과는 REAL_RAW 대비로 해석.',
        '- R0/박스/score/centroid는 불변. negative 추가 추론은 하지 않았으며 박스 confidence 기준 오검출 개선 주장은 불가.',
        '- 1차 필터 탈락 이미지에는 N2를 호출하지 않았음. 2차 필터를 완화하거나 결과를 보고 샘플을 교체하지 않았음.', '']
    (DOC / 'RESULTS_KO.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'pseudo', 'train', 'evaluate', 'all'))
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    try:
        if args.stage == 'prepare': prepare(); return
        assert torch.cuda.is_available(), 'CUDA unavailable; do not fall back to CPU'
        E.gpu()
        if args.stage == 'all':
            prepare(); pseudo_cache(); train(); evaluate()
        else:
            {'pseudo': pseudo_cache, 'train': train, 'evaluate': evaluate}[args.stage]()
    except Exception as exc:
        status('FAILED', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
