"""Bounded single-seed C/D/E screen, with source and recording locks."""
from __future__ import annotations
import argparse
from collections import Counter
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
DOC = ROOT / '_docs/experiments/pallet_large_error_refiner_v1'
RAW = ROOT / 'data/pallet/results/pallet_large_error_refiner_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/research/pallet_dim_conditioned_p_v1'))
sys.path.insert(1, str(ROOT / 'scripts/self_training_yolo'))
import dcp_env as E
from refiner import context
from scripts.evaluation import final_dimension_release as F
from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
from scripts.research.pallet_large_error_refiner_v1.model import WideRefiner, KEYS, corrupt, objective

ARMS = ('C_SYN_NORMAL', 'D_SYN_LARGE', 'E_SYN_REAL_LARGE')
STEPS = 2000
AUDIT = ROOT / '_docs/experiments/pallet_transfer_replay_control_v1/DATA_AUDIT.json'
GROUPS = ROOT / 'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
GREEN = F.DOC / 'green150_saved_labels_v1/DATASET_SNAPSHOT.json'
FILTER = ROOT / 'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json'


def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze(path, value):
    if path.exists(): assert E.read(path) == value, ('Immutable artifact differs', str(path))
    else: write(path, value)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.pt')
    torch.save(value, tmp); tmp.replace(path)


def status(stage, **fields):
    row = dict(time=E.now(), stage=stage, **fields)
    write(DOC / 'STATUS.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def session(path): return str(Path(path).parent.parent)


def manual_target(doc):
    gt, valid = annotation_arrays(doc, True)
    entries = doc['objects'][0]['keypoint_annotations']
    cam = doc['camera_data']; h, w = cam['height'], cam['width']
    valid &= np.array([r.get('in_frame') is True for r in entries])
    valid &= (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
    valid[8] = False
    return gt, valid


def prepare():
    from dev_evaluate import population_metadata
    lock = F.checked_lock()
    _, pop = population_metadata()
    items = {item.frame_id: (item, meta) for item, meta in pop}
    selected = [i for i in E.read(AUDIT)['selected_ids'] if not i.startswith('eval_pallet07:')]
    assert len(selected) == 24
    group_doc = E.read(GROUPS)
    key_group = {s['session_key']: g['recording_id'] for g in group_doc['groups']
                 if not g['is_collection'] for s in g['sessions']}
    train_groups = {key_group[session(items[i][0].image)] for i in selected}
    # Reserve whole candidate recordings, including images with unknown point provenance.
    train_aliases = {k for k, v in key_group.items() if v in train_groups}
    forbidden = set(train_aliases)
    for p in group_doc['partial_overlap_pairs']:
        if p['session_a'] in train_aliases: forbidden.add(p['session_b'])
        if p['session_b'] in train_aliases: forbidden.add(p['session_a'])
    reserved_hashes = {E.sha(ROOT / items[i][0].image) for i in selected}
    dev_cache = E.read(E.DOC / 'DEV_CACHE_COMPLETE.json')
    cache_by_id = {r['id']: r for r in dev_cache['records']}
    train, provenance, eval_records = [], [], []
    for ident in selected:
        item, meta = items[ident]; doc = E.read(ROOT / item.label)
        gt, valid = manual_target(doc)
        provenance.append(dict(id=ident, verified_corners=int(valid.sum()),
            sources=dict(Counter(k.get('source', 'unknown') for k in doc['objects'][0]['keypoint_annotations'][:8]))))
        if valid.any():
            train.append(dict(id=ident, image=E.bound(ROOT / item.image), annotation=E.bound(ROOT / item.label),
                cache=cache_by_id[ident], manual_mask=valid.tolist(), manual_corners=int(valid.sum()),
                object_type=meta['object_type'], session=meta['session_id']))
    for ident, (item, meta) in items.items():
        s = session(item.image)
        assert s in key_group
        if key_group[s] in train_groups or s in forbidden: continue
        assert E.sha(ROOT / item.image) not in reserved_hashes
        # Intrinsics only: no GT pose, axis assignment, or dimension hypothesis is passed to inference/gates.
        candidates = [ROOT / k / name for k, gid in key_group.items() if gid == key_group[s]
                      for name in ('cam_K.txt', 'camera_intrinsics.txt') if (ROOT / k / name).exists()]
        if candidates:
            K = np.loadtxt(candidates[0]).reshape(3, 3)
            camera = E.bound(candidates[0]); camera_source = 'raw recording camera file'
        else:
            intr = E.read(ROOT / item.label)['camera_data']['intrinsics']
            K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
            camera = E.bound(ROOT / item.label); camera_source = 'annotation calibration only; unverified acquisition'
        eval_records.append(dict(id=ident, image=E.bound(ROOT / item.image), annotation=E.bound(ROOT / item.label),
            cache=cache_by_id[ident], object_type=meta['object_type'], session=meta['session_id'],
            camera=camera, camera_source=camera_source, K=K.tolist(), recording=key_group[s]))
    green = E.read(GREEN)
    for r in green['records']:
        assert r['image']['sha256'] not in reserved_hashes
        assert session(r['image']['path']) not in forbidden
        assert key_group.get(session(r['image']['path'])) not in train_groups
    assert len(train) == 9 and len(eval_records) == 72
    assert all(not t['id'].startswith('eval_pallet07:') for t in train)
    split = dict(candidate_ids=selected, reserved_recording_groups=sorted(train_groups), provenance=provenance,
        train=train, train_frames=len(train), manual_corners=sum(t['manual_corners'] for t in train),
        evaluation=eval_records, evaluation_frames=len(eval_records),
        evaluation_sessions=dict(Counter(t['session'] for t in eval_records)), green=E.bound(GREEN),
        green_train=False, image_hash_overlap=False, recording_overlap=False,
        candidate_labels_not_exported_or_modified=True, independent_test=False)
    freeze(DOC / 'SPLIT.json', split)
    protocol = dict(experiment='pallet_large_error_refiner_v1', status='SINGLE_SEED_DEVELOPMENT_SCREEN',
        seed=1, steps=STEPS, batch=16, microbatch=8, checkpoint='last2000 only; no eval-based checkpoint selection',
        arms=dict(A='frozen N2_DIM_ONLY seed1, cap=0.01 image diagonal',
                  B='same A, cap=0.04 image diagonal; candidate radius unchanged',
                  C='new joint full-ROI model; source synthetic ordinary R0 inputs',
                  D='same C architecture/init; synthetic R0 plus large input perturbations',
                  E='same D; replace half batch with verified human-click real targets'),
        architecture='64/128 P3/P4 -> 16/16 projected channels -> 32x32 box+20% ROI; 8 Gaussian input-corner maps sigma .04 ROI; 2 XY maps; 2 conv32 SiLU; pool4x4; 535->128->16 MLP; zero-init final; radial tanh cap .20 bbox diagonal; centroid unchanged',
        corruption='sample modes 0.50 unchanged, 0.25 one random corner, 0.25 vertical edge [(0,3),(1,2),(4,7),(5,6)] translated together; uniform direction and 0.05-0.15 bbox-diagonal length; input only, no GT changes',
        loss='masked corner SmoothL1 on residual/bbox diagonal, beta=.01; sum XY, mean valid corners per image then mean images; fixed camera-facing indices; identical CDE',
        augmentation='only coordinate corruption; cached original image features for all arms; no real photometric advantage',
        optimizer=dict(name='AdamW', lr=3e-4, final_lr=3e-5, weight_decay=1e-4, clip=5.),
        synthetic_train_pool=2048, source_half_paired=True, synthetic_stress_holdout=256,
        real_supervision='only explicit manual_click, visibility != 0, in_frame=True, finite inside-image 8 corners; no PnP/unknown/centroid/pseudo targets',
        real_candidate_frames=24, verified_real_frames=len(train), verified_corners=split['manual_corners'],
        inference_metadata='canonical dimensions from object registry, known object type; existing TRAIN normalization; no pose/GT selection',
        filter='same confidence >=.85, >=6 corners conf>=.5; remove <=.05 AND flip<=.05; fail -> original A coordinates, no frame removal; all arms reported ungated and gated',
        metrics='whole-object approved C1/C2/C4 E_sym, fixed-index E_fixed, pooled P90, PCK10; recovery R0>20 to <=10 and damage R0<5 to >10 on same canonical GT corner identities',
        pass_rule='separately ungated and gated: recovery improvement vs A >=10pp, PCK10 delta >=-0.5pp and damage rate delta <=+0.5pp on BOTH DEV72 and GREEN150_MANUAL; empty denominator not estimable',
        evaluation_reused=True, independent_confirmation=False, physical_pose_claim=False, role_permutation_repair=False,
        original_paper_release_untouched=True, auto_promote=False, no_other_training_runs=True,
        synthetic_stress='256 matched non-train rows (existing heldout/selection/calibration partitions), fixed input-corruption RNG 219; all-corner input-pixel errors R0/C/D/E, clean and stress; mechanism diagnostic, no model selection',
        gpu_thermal_limit_C=80, system_changes=False,
        model_lock=E.bound(F.DOC / 'MODEL_LOCK.json'), split=E.bound(DOC / 'SPLIT.json'),
        original_head=next(r['checkpoint'] for r in lock['heads'] if r['arm']=='N2_DIM_ONLY' and r['seed']==1),
        bindings=[E.bound(p) for p in [HERE/'run.py', HERE/'model.py', HERE/'evaluate.py', HERE/'test_model.py', HERE/'test_contract.py',
            AUDIT, GROUPS, GREEN, FILTER, E.DOC/'DEV_CACHE_COMPLETE.json',
            ROOT/'scripts/self_training_yolo/pseudo_label_filters.py', ROOT/'scripts/evaluation/green_saved_labels_v1.py']])
    freeze(DOC / 'PROTOCOL.json', protocol)
    status('PREPARED', manual_frames=len(train), manual_corners=split['manual_corners'], eval_frames=72, green=150)


def verify():
    p = E.read(DOC / 'PROTOCOL.json'); F.checked_lock()
    for b in [*p['bindings'], p['split'], p['model_lock'], p['original_head']]: F.verify(b)
    return p


def input_batch(cap, dims, order, device='cuda'):
    inputs = E.old('features').branch_inputs(cap)
    if inputs is None: return None, None
    b = {k: torch.as_tensor(inputs[k], device=device)[None] for k in ('points','boxes','point_valid','input_shape')}
    for k in ('p3', 'p4'):
        tensor = cap[k]
        b[k] = (tensor[None] if tensor.ndim == 3 else tensor).to(device)
    norm = E.read(E.DOC / 'DIM_NORMALIZATION_LOCK.json')
    b['context'] = torch.as_tensor(context(np.asarray(dims)[None], [order], norm, False), device=device)
    return b, inputs


def real_caches():
    result = []
    for r in E.read(DOC / 'SPLIT.json')['train']:
        for k in ('image', 'annotation', 'cache'): F.verify(r[k])
        row = torch.load(ROOT/r['cache']['path'], map_location='cpu', weights_only=False)
        b, inputs = input_batch(row['captured'], row['dimensions'], row['order'], 'cpu')
        assert b is not None
        gt, valid = manual_target(E.read(ROOT / r['annotation']['path']))
        assert valid.tolist() == r['manual_mask']
        gain, offset = E.old('features').canvas_affine(row['captured']['canvas_shape'], row['captured']['input_shape'])
        mapped = (gt + row['captured']['added_border']) * gain + offset
        back = (mapped-offset)/gain-row['captured']['added_border']
        assert np.allclose(back[valid], gt[valid], atol=1e-8, rtol=0)
        b['gt_points'] = torch.as_tensor(mapped, dtype=torch.float32)[None]
        b['gt_valid'] = torch.as_tensor(valid)[None]
        result.append(b)
    return result


def real_batch(caches, rows):
    return {k: torch.cat([caches[int(i)][k] for i in rows], 0).cuda()
            for k in (*KEYS, 'gt_points', 'gt_valid')}


def inputs_lock(data):
    dst = RAW / 'TRAIN_ORDERS.npz'
    if not dst.exists():
        rng = np.random.default_rng(1)
        pool = np.sort(rng.choice(data.train_rows, 2048, replace=False))
        eligible = data.validation_rows[np.asarray(data.arrays['matched'][data.validation_rows], bool)]
        held = np.sort(rng.choice(eligible, 256, replace=False))
        RAW.mkdir(parents=True, exist_ok=True)
        np.savez(dst, pool=pool, held=held, source=rng.choice(pool,(STEPS,8)),
            replacement=rng.choice(pool,(STEPS,8)), real=rng.integers(0,9,(STEPS,8)))
    orders = np.load(dst)
    assert set(orders['pool']) <= set(data.train_rows)
    assert set(orders['held']) <= set(data.validation_rows)
    assert not set(orders['held']) & set(orders['pool'])
    freeze(DOC / 'TRAIN_INPUT_LOCK.json', dict(protocol=E.bound(DOC/'PROTOCOL.json'), orders=E.bound(dst),
        train_rows=orders['pool'].tolist(), heldout_rows=orders['held'].tolist(),
        real_ids=[r['id'] for r in E.read(DOC/'SPLIT.json')['train']],
        real_manual_corners=sum(r['manual_corners'] for r in E.read(DOC/'SPLIT.json')['train']),
        train_partitions=dict(Counter(str(data.partitions[int(i)]) for i in orders['pool'])),
        heldout_partitions=dict(Counter(str(data.partitions[int(i)]) for i in orders['held'])),
        train_eval_disjoint=True))
    return orders


def train():
    protocol = verify(); E.gpu()
    from data import PaperData
    data = PaperData(); orders = inputs_lock(data); caches = real_caches()
    torch.manual_seed(1)
    initial = WideRefiner().state_dict(); init_sha = E.state_sha(initial)
    for arm in ARMS:
        destination = RAW / 'runs' / arm / 'last.pt'
        if destination.exists():
            ck = torch.load(destination, map_location='cpu', weights_only=False)
            assert ck['complete'] and ck['protocol_sha'] == E.sha(DOC/'PROTOCOL.json')
            assert ck['input_sha'] == E.sha(DOC/'TRAIN_INPUT_LOCK.json')
            continue
        head = WideRefiner().cuda().train(); head.load_state_dict(initial)
        assert E.state_sha(head.state_dict()) == init_sha
        opt = protocol['optimizer']; optimizer = torch.optim.AdamW(head.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
        gs = torch.Generator(device='cuda').manual_seed(1001)
        gr = torch.Generator(device='cuda').manual_seed(2001)
        history = []; start = 0; prior = 0.; resume = destination.with_name('resume.pt')
        if resume.exists():
            ck = torch.load(resume, map_location='cpu', weights_only=False)
            assert ck['protocol_sha'] == E.sha(DOC/'PROTOCOL.json') and ck['input_sha'] == E.sha(DOC/'TRAIN_INPUT_LOCK.json')
            head.load_state_dict(ck['state']); optimizer.load_state_dict(ck['optimizer'])
            gs.set_state(ck['source_rng']); gr.set_state(ck['other_rng'])
            history = ck['history']; start = ck['step']; prior = ck['elapsed_seconds']
        begin = time.monotonic()
        for i in range(start, STEPS):
            lr = opt['final_lr'] + .5*(opt['lr']-opt['final_lr'])*(1+math.cos(math.pi*i/STEPS))
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            b = data.batch(orders['source'][i], 'N2_DIM_ONLY')
            if arm != 'C_SYN_NORMAL': b = corrupt(b, gs)
            left = objective(head(b), b); assert torch.isfinite(left); (.5*left).backward()
            b = real_batch(caches, orders['real'][i]) if arm == 'E_SYN_REAL_LARGE' else data.batch(orders['replacement'][i], 'N2_DIM_ONLY')
            if arm != 'C_SYN_NORMAL': b = corrupt(b, gr)
            right = objective(head(b), b); assert torch.isfinite(right); (.5*right).backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), opt['clip'], error_if_nonfinite=True)
            optimizer.step()
            if i == start or (i+1)%100 == 0:
                row = dict(step=i+1, source_loss=float(left.detach()), other_loss=float(right.detach()),
                    gradient_norm=float(gradient), elapsed_seconds=prior+time.monotonic()-begin, lr=lr)
                history.append(row); status('TRAINING', arm=arm, **row, gpu=E.gpu())
            if (i+1)%250 == 0:
                ck = dict(arm=arm, complete=i+1==STEPS, step=i+1, seed=1, initial_sha=init_sha,
                    state=head.state_dict(), optimizer=optimizer.state_dict(), source_rng=gs.get_state(), other_rng=gr.get_state(),
                    protocol_sha=E.sha(DOC/'PROTOCOL.json'), input_sha=E.sha(DOC/'TRAIN_INPUT_LOCK.json'),
                    history=history, elapsed_seconds=prior+time.monotonic()-begin)
                save(destination if ck['complete'] else resume, ck)
        freeze(DOC / f'FIT_{arm}.json', dict(checkpoint=E.bound(destination), steps=STEPS, initial_sha=init_sha,
            history=history, elapsed_seconds=ck['elapsed_seconds'], only_refiner_trainable=True, R0_not_loaded_during_training=True))
        del head, optimizer, b, left, right, ck
        torch.cuda.empty_cache()
    F.checked_lock(); status('TRAINING_COMPLETE', updates=3*STEPS)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare','train','evaluate','all'))
    args = parser.parse_args(); torch.set_num_threads(4); cv2.setNumThreads(1)
    try:
        if args.stage == 'prepare': prepare(); return
        assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback'
        E.gpu()
        if args.stage in ('all',): prepare()
        if args.stage in ('all','train'): train()
        if args.stage in ('all','evaluate'):
            from scripts.research.pallet_large_error_refiner_v1.evaluate import evaluate
            evaluate()
    except Exception as exc:
        status('FAILED', error=repr(exc)); raise


if __name__ == '__main__': main()
