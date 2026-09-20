"""Two fresh COCO-pose initializations, matched 1,000-update development screen.

No R0 continuation, real-GT training, checkpoint selection, or upstream writes.
Run prepare -> probe -> lock -> train n -> train m -> evaluate -> report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import PoseModel

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/pallet/results/pallet_capacity_screen_v1'
DOC = ROOT / '_docs/experiments/pallet_capacity_screen_v1'
SOURCE = ROOT / 'challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k'
WEIGHTS = ROOT / 'challenge/weights/pretrained_yolo'
R0_ARGS = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/args.yaml'
BATCH, STEPS, SEED = 4, 1000, 1
MILESTONES = (250, 500, 1000)
DATA = yaml.safe_load((SOURCE / 'data.yaml').read_text())
_original = yaml.safe_load(R0_ARGS.read_text())
_keys = ('box cls dfl pose kobj rle hsv_h hsv_s hsv_v degrees translate scale shear perspective '
         'flipud fliplr bgr mosaic mixup cutmix copy_paste erasing momentum weight_decay').split()
HYP = {k: _original[k] for k in _keys}
HYP.update(task='pose', imgsz=640, batch=BATCH, epochs=10, optimizer='SGD', lr0=.01,
           lrf=.01, deterministic=True, close_mosaic=2)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == payload, f'Immutable artifact differs: {path}'
    else:
        with path.open('x') as f:
            f.write(payload)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gpu(exclusive=True):
    query = lambda kind, fields: subprocess.check_output(
        ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True).strip()
    values = query('gpu', 'name,temperature.gpu,memory.used,utilization.gpu,power.draw')
    processes = query('compute-apps', 'pid,process_name,used_memory')
    foreign = [line for line in processes.splitlines() if line.split(',')[0].strip() != str(os.getpid())]
    if exclusive:
        assert not foreign, f'Foreign compute process; do not terminate it: {foreign}'
    assert float(values.split(',')[1]) < 80, f'GPU thermal guard: {values}'
    return dict(gpu=values, processes=processes, timestamp=time.time(), system_changes=False)


def evaluation_module():
    path = ROOT / 'scripts/research/pallet_active_learning_v1/simulation_evaluate.py'
    spec = importlib.util.spec_from_file_location('capacity_previous_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RAW, module.DOC = RAW, DOC
    return module


def prepare():
    assert not (DOC / 'PROTOCOL_LOCK.json').exists()
    files = sorted((SOURCE / 'images/train').glob('*'))
    files = [p for p in files if p.suffix.lower() in ('.jpg', '.png', '.jpeg')]
    assert len(files) == 55980, len(files)
    selected = np.random.default_rng(SEED).permutation(len(files))[:BATCH * STEPS]
    rows = []
    for j in selected:
        image = files[int(j)]
        label = SOURCE / 'labels/train' / image.with_suffix('.txt').name
        assert image.is_file() and label.is_file()
        # Original images and labels stay untouched; local dataset contains links only.
        for kind, source in [('images', image), ('labels', label)]:
            dest = RAW / 'dataset' / kind / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(source.resolve())
            assert dest.resolve() == source.resolve()
        rows.append(dict(image=str(image.relative_to(ROOT)), resolved_image=str(image.resolve()),
                         image_sha256=sha(image), label=str(label.relative_to(ROOT)), label_sha256=sha(label)))
    write(RAW / 'SOURCE_BINDINGS.json', rows)
    write(DOC / 'DATA_SUMMARY.json', dict(source_train=55980, screen_subset=len(rows),
        selection='Uniform permutation seed1; first4000; selected without real evaluation outcomes',
        source_data_yaml_sha256=sha(SOURCE / 'data.yaml'), bindings_sha256=sha(RAW / 'SOURCE_BINDINGS.json'),
        train_real_images=0, all_mosaic_donors_restricted_to_same4000=True))
    old_split = read(ROOT / '_docs/experiments/pallet_active_learning_v1/retrospective_v1/SPLIT.json')
    evaluation = old_split['pool'] + old_split['evaluation']
    assert len(evaluation) == 319
    assert not {r['image_sha256'] for r in rows} & {r['image_sha256'] for r in evaluation}
    write(DOC / 'SPLIT.json', dict(train_real=[], evaluation=evaluation))
    write(DOC / 'PREPARE.json', dict(status='PASS', actual_positive_evaluation=319,
          no_selected_train_positive_image_hash_overlap=True, train_domain='synthetic only',
          source_init_sha256={a: sha(WEIGHTS / f'yolo26{a}-pose.pt') for a in ('n', 'm')}))
    print('PREPARE_PASS', len(rows), flush=True)


def seed_all(seed=SEED):
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def model(arm):
    seed_all()
    source = YOLO(str(WEIGHTS / f'yolo26{arm}-pose.pt'), task='pose').model.float()
    m = PoseModel(copy.deepcopy(source.yaml), ch=3, nc=1, data_kpt_shape=(9, 3), verbose=False)
    m.load(source, verbose=False)
    compatible = {k: v for k, v in source.state_dict().items()
                  if k in m.state_dict() and v.shape == m.state_dict()[k].shape}
    assert compatible and all(torch.equal(v, m.state_dict()[k]) for k, v in compatible.items())
    m.args = get_cfg(overrides=HYP)
    m.names = {0: 'pallet'}
    m.model[-1].dfl.requires_grad_(False)
    assert m.model[-1].nc == 1 and m.model[-1].nk == 27 and m.model[-1].end2end
    return m, dict(parameters=sum(p.numel() for p in m.parameters()), pretrained_tensors=len(compatible),
                   total_state_tensors=len(m.state_dict()), initial_state_sha256=tensor_sha(m.state_dict()),
                   source_sha256=sha(WEIGHTS / f'yolo26{arm}-pose.pt'))


class Samples(Dataset):
    def __init__(self, late=False):
        hyp = get_cfg(overrides=HYP)
        if late:
            hyp.mosaic = 0.
        self.data = YOLODataset(img_path=str(RAW / 'dataset/images'), imgsz=640, batch_size=BATCH,
            augment=True, hyp=hyp, rect=False, cache=False, stride=32, pad=0., data=DATA,
            task='pose', prefix='capacity: ')
        assert len(self.data) == BATCH * STEPS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        index, position = key
        seed = 10_000_000 + position
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self.data[index]


def loader(start, stop):
    data = Samples(late=start >= 800)
    order = np.random.default_rng(101).permutation(len(data))
    batches = [[(int(order[p]), p) for p in range(step * BATCH, (step + 1) * BATCH)]
               for step in range(start, stop)]
    return DataLoader(data, batch_sampler=batches, num_workers=2, collate_fn=YOLODataset.collate_fn,
                      pin_memory=True, generator=torch.Generator().manual_seed(101))


def optimizer(m):
    groups = [[], [], []]
    for mod in m.modules():
        for name, p in mod.named_parameters(recurse=False):
            if p.requires_grad:
                i = 0 if name == 'bias' else 2 if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) else 1
                groups[i].append(p)
    return torch.optim.SGD([dict(params=g, weight_decay=.0005 if i == 1 else 0., is_bias=i == 0)
                            for i, g in enumerate(groups)], lr=.01, momentum=.937, nesterov=True)


def learning_rate(step, is_bias):
    target = .01 * (.01 + .99 * (1 + math.cos(math.pi * step / (STEPS - 1))) / 2)
    return float(np.interp(step, [0, 100], [.1 if is_bias else 0., target])) if step < 100 else target


def update(m, criterion, opt, batch, step):
    digest = tensor_sha({k: batch[k] for k in ('img', 'batch_idx', 'cls', 'bboxes', 'keypoints')})
    for g in opt.param_groups:
        g['lr'] = learning_rate(step, g['is_bias'])
        g['momentum'] = float(np.interp(step, [0, 100], [.8, .937]))
    batch = {k: (v.cuda(non_blocking=True).float() / 255 if k == 'img' else v.cuda(non_blocking=True))
             if torch.is_tensor(v) else v for k, v in batch.items()}
    opt.zero_grad(set_to_none=True)
    # Explicit FP32: each iteration performs one optimizer step; no AMP overflow skips.
    loss = criterion(m(batch['img']), batch)[0].sum()
    assert torch.isfinite(loss), f'Nonfinite loss at update {step}'
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 10., error_if_nonfinite=True)
    opt.step()
    return dict(step=step + 1, input_sha256=digest, loss=float(loss.detach()),
                gradient_norm=float(norm), image_names=[Path(p).name for p in batch['im_file']],
                lrs=[g['lr'] for g in opt.param_groups])


def probe():
    assert not (DOC / 'PROTOCOL_LOCK.json').exists()
    pre = gpu()
    receipts = {}
    for arm in ('n', 'm'):
        m, init = model(arm)
        m = m.cuda().train()
        criterion, opt = m.init_criterion(), optimizer(m)
        torch.cuda.reset_peak_memory_stats()
        records, durations = [], []
        for i, batch in enumerate(loader(0, 5)):
            torch.cuda.synchronize()
            t = time.monotonic()
            records.append(update(m, criterion, opt, batch, i))
            torch.cuda.synchronize()
            durations.append(time.monotonic() - t)
        receipts[arm] = dict(init=init, peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
            seconds_per_step_median=float(np.median(durations[1:])), records=records, discarded=True)
        print('PROBE', arm, receipts[arm]['peak_allocated_MiB'], receipts[arm]['seconds_per_step_median'], flush=True)
        del m, criterion, opt
        torch.cuda.empty_cache()
    assert [r['input_sha256'] for r in receipts['n']['records']] == [r['input_sha256'] for r in receipts['m']['records']]
    write(DOC / 'PROBE.json', dict(status='PASS', batch=BATCH, fp32=True, telemetry=pre,
        formal_updates=0, discarded_probe_updates=10, actual_augmented_input_parity=True, arms=receipts))


def lock():
    assert read(DOC / 'PROBE.json')['status'] == 'PASS'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'main'
    E = evaluation_module()
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), R0_ARGS,
        WEIGHTS / 'yolo26n-pose.pt', WEIGHTS / 'yolo26m-pose.pt', RAW / 'SOURCE_BINDINGS.json',
        DOC / 'SPLIT.json', Path(E.__file__), Path(E.P.E.__file__), E.P.POS, E.P.NEG,
        E.P.POSE / 'GEOMETRY_RESOLVED_POSE_GT.json', E.P.POSE / 'AXIS_REVIEW_MANIFEST.json',
        E.P.POSE / 'POSE_EVAL_OBJECT_CONTRACT.json',
        ROOT / 'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py']}
    write(DOC / 'PROTOCOL_LOCK.json', dict(status='LOCKED_BEFORE_FORMAL_TRAINING', source_bindings=bindings,
        start_main=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        arms=['n', 'm'], seed=SEED, optimizer_updates_per_arm=STEPS, milestones=list(MILESTONES),
        physical_batch=BATCH, gradient_accumulation=1, precision='FP32, no skipped optimizer steps',
        train_data='Uniform fixed4000 synthetic subset from R0 source55980. No real GT in training.',
        initialization='Official COCO pose n/m; resized 1-class9-keypoint head; no trained pallet checkpoint',
        intervention='Whole YOLO n/m model scale, NOT backbone-only; pretraining compute not matched',
        training=HYP, BN='normal running-stat training, same physical batch',
        schedule='100step warmup, cosine1000 updates; mosaic off at800; stock E2ELoss.update each100 steps',
        recipe_scope='Matched short screen, not full R0 reproduction. FP32/no EMA/no accumulation; compressed schedule.',
        environment_note='Optional Albumentations disabled by existing ImageCompression API mismatch in both arms; CuDNN nvrtc fallback warning observed in probe. Actual tensor parity checked, not bit-exact cross-device training claimed.',
        evaluation='All historical DEV319 positives +2689 negatives; fixed canonical inference and MAIN PnP',
        selection='1000step final only;250/500 trajectory descriptive; no best checkpoint/extra seed/retry/tuning',
        gate=dict(final_common_corner_median_ratio_max=.95, final_common_corner_p90_ratio_max=.95,
            final_common_gross20_nonworse=True, final_translation_ratio_max=.95,
            detection_nonworse=True, AP50_95_nonworse=True, pose_coverage_nonworse=True,
            final_yaw_ratio_max=1.10, earlier_500_common_p90_and_translation_nonworse=True,
            minimum_common_frames=100),
        verdicts=['SIGNAL_FOR_CONFIRMATION', 'NO_SIGNAL_WITHIN_1000_STEPS', 'INCONCLUSIVE_INSUFFICIENT_COVERAGE'],
        limitations='One seed, small synthetic subset, short convergence, repeated DEV and reconstructed pose GT. No capacity rejection, novelty or independent confirmation.',
        gpu_guard='No foreign compute; sample every50updates; halt at80C. No reboot, driver or power changes.'))
    write(DOC / 'GPU_PREFLIGHT.json', gpu())
    print('PROTOCOL_LOCKED', flush=True)


def verify_lock():
    for path, expected in read(DOC / 'PROTOCOL_LOCK.json')['source_bindings'].items():
        assert sha(ROOT / path) == expected, path


def save_model(m, out, step):
    dest = out / f'step{step:04}.pt'
    assert not dest.exists()
    saved = copy.deepcopy(m).cpu().eval()
    saved.args = vars(saved.args)
    if hasattr(saved, 'criterion'):
        delattr(saved, 'criterion')
    torch.save(dict(model=saved, ema=None, train_args=HYP, epoch=step // 100 - 1,
                    optimizer=None, actual_optimizer_updates=step), dest)
    return sha(dest)


def train(arm):
    verify_lock()
    pre = gpu()
    out = RAW / 'runs' / arm
    assert not (out / 'START.json').exists(), 'Never silently repeat a started formal fit'
    m, init = model(arm)
    assert init == read(DOC / 'PROBE.json')['arms'][arm]['init']
    write(out / 'START.json', dict(arm=arm, init=init, gpu=pre, lock_sha256=sha(DOC / 'PROTOCOL_LOCK.json')))
    m = m.cuda().train()
    criterion, opt = m.init_criterion(), optimizer(m)
    assert type(criterion).__name__ == 'E2ELoss' and type(criterion.one2one).__name__ == 'PoseLoss26'
    trace, snapshots, checkpoints = [], [], {}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for start, stop in ((0, 800), (800, STEPS)):
        for step, batch in enumerate(loader(start, stop), start):
            trace.append(update(m, criterion, opt, batch, step))
            done = step + 1
            if done % 100 == 0:
                criterion.update()
            if done % 50 == 0:
                snapshots.append(dict(step=done, **gpu()))
                print(json.dumps(dict(arm=arm, updates=done, loss=trace[-1]['loss'],
                    elapsed_seconds=round(time.monotonic() - started), gpu=snapshots[-1]['gpu'])), flush=True)
                write(out / f'trace_{done:04}.json', trace[-50:])
            if done in MILESTONES:
                checkpoints[str(done)] = save_model(m, out, done)
    assert len(trace) == STEPS and criterion.updates == 10
    assert all(torch.isfinite(v).all() for v in m.state_dict().values())
    write(out / 'EXPOSURE.json', trace)
    write(out / 'TRAINING_AUDIT.json', dict(status='PASS', arm=arm, optimizer_updates=len(trace),
        init=init, checkpoints=checkpoints, fp32=True, stock_criterion=True, criterion_updates=criterion.updates,
        elapsed_seconds=time.monotonic() - started, peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
        telemetry=snapshots, source_code_sha256=sha(Path(__file__))))
    print('TRAIN_COMPLETE', arm, flush=True)


def audit():
    verify_lock()
    receipts, traces = {}, {}
    for arm in ('n', 'm'):
        folder = RAW / 'runs' / arm
        a = read(folder / 'TRAINING_AUDIT.json')
        assert a['optimizer_updates'] == STEPS
        assert a['source_code_sha256'] == sha(Path(__file__))
        traces[arm] = read(folder / 'EXPOSURE.json')
        assert len(traces[arm]) == STEPS
        for step in MILESTONES:
            path = folder / f'step{step:04}.pt'
            assert sha(path) == a['checkpoints'][str(step)]
            ck = torch.load(path, map_location='cpu', weights_only=False)
            assert ck['actual_optimizer_updates'] == step
            assert sum(p.numel() for p in ck['model'].parameters()) == a['init']['parameters']
            assert all(torch.isfinite(v).all() for v in ck['model'].state_dict().values())
        receipts[arm] = a
    for n, m in zip(traces['n'], traces['m']):
        assert n['step'] == m['step'] and n['input_sha256'] == m['input_sha256'] and n['lrs'] == m['lrs']
    for r in read(RAW / 'SOURCE_BINDINGS.json'):
        assert sha(ROOT / r['label']) == r['label_sha256']
        assert sha(ROOT / r['image']) == r['image_sha256']
    write(DOC / 'TRAINING_AUDIT.json', dict(status='PASS', formal_fits=2, optimizer_updates=2000,
        actual_all1000_augmented_batches_identical=True, source_images_labels_unchanged=True, arms=receipts))


def evaluate():
    audit()
    E = evaluation_module()
    P = E.P
    pair = P.population()
    targets = {i.frame_id: P.E._legacy_forbidden_target(i) for i in pair.positive.items}
    for step in MILESTONES:
        for arm in ('n', 'm'):
            name = f'{arm}_{step}'
            d = RAW / 'evaluation' / name
            cache = d / 'PREDICTIONS.json'
            ck = RAW / 'runs' / arm / f'step{step:04}.pt'
            if not cache.exists():
                gpu()
                predictor = P.E._UltralyticsPredictor(ck, '0')
                frames = {}
                for i, item in enumerate([*pair.positive.items, *pair.negative.items]):
                    values = predictor.predict(ROOT / item.image)
                    frames[P.canonical_key(item.image)] = [dict(score=float(s), box_xyxy=b.tolist(),
                        keypoints_xy=k.tolist() if k is not None else None) for s, b, k in values]
                    if (i + 1) % 500 == 0:
                        gpu()
                        print('INFERENCE', name, i + 1, flush=True)
                del predictor
                torch.cuda.empty_cache()
                write(cache, dict(schema_version='paper_cached_predictions_v1', complete=True,
                    model=name, weights_sha256=sha(ck), frames=frames))
            frames = read(cache)['frames']
            collected = P.E._collect_predictions(pair, P.E._CachedPredictor(cache), validated_targets=targets)
            metrics = P.E._evaluate_2d_collected(pair, *collected)
            top = collected[2]
            rows = {}
            for item in pair.positive.items:
                pred, target = top.get(item.frame_id), targets[item.frame_id]
                errors = []
                if pred is not None and pred.keypoints_xy is not None:
                    errors = np.linalg.norm(pred.keypoints_xy - target.keypoints_xy, axis=1)[target.keypoint_supervision_mask].tolist()
                rows[item.frame_id] = dict(matched=bool(pred is not None and pred.target_iou >= .5), errors_px=errors)
            write(d / 'PER_FRAME.json', rows)
            if any(frames[P.canonical_key(item.image)] for item in pair.positive.items):
                pose = E.pose_evaluation(d, frames, name)
            else:
                pose = dict(ALL=dict(n=0), coverage=0.)
            write(d / 'RESULT.json', dict(name=name, step=step, metrics=metrics, pose=pose,
                 Det=sum(r['matched'] for r in rows.values()) / 319))
            print('EVALUATED', name, flush=True)


def paired_geometry(n, m):
    ids = sorted(k for k in n if n[k]['matched'] and m[k]['matched'] and n[k]['errors_px'] and m[k]['errors_px'])
    def stats(rows):
        a = np.concatenate([rows[k]['errors_px'] for k in ids]) if ids else np.array([])
        return dict(median_px=float(np.median(a)) if len(a) else None,
                    p90_px=float(np.quantile(a, .9)) if len(a) else None,
                    gross20=float(np.mean(a > 20)) if len(a) else None)
    return dict(frames=len(ids), frame_ids=ids, n=stats(n), m=stats(m))


def report():
    verify_lock()
    results, pairs = {}, {}
    for step in MILESTONES:
        results[str(step)] = {a: read(RAW / 'evaluation' / f'{a}_{step}' / 'RESULT.json') for a in ('n', 'm')}
        pairs[str(step)] = paired_geometry(*[read(RAW / 'evaluation' / f'{a}_{step}' / 'PER_FRAME.json') for a in ('n', 'm')])
    # Undefined metrics fail safely; no changing gates after seeing results.
    def le(a, b, ratio=1.):
        return a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and a <= ratio * b
    last, pair = results['1000'], pairs['1000']
    earlier, ep = results['500'], pairs['500']
    pose = {a: last[a]['pose']['ALL'] for a in ('n', 'm')}
    checks = dict(common_frames=pair['frames'] >= 100,
        corner_median=le(pair['m']['median_px'], pair['n']['median_px'], .95),
        corner_p90=le(pair['m']['p90_px'], pair['n']['p90_px'], .95),
        gross20=le(pair['m']['gross20'], pair['n']['gross20']),
        translation=le(pose['m'].get('translation_median_cm'), pose['n'].get('translation_median_cm'), .95),
        detection=le(last['n']['Det'], last['m']['Det']),
        AP=le(last['n']['metrics']['box_ap50_95'], last['m']['metrics']['box_ap50_95']),
        pose_coverage=le(pose['n']['n'], pose['m']['n']),
        yaw=le(pose['m'].get('yaw_median_deg'), pose['n'].get('yaw_median_deg'), 1.1),
        earlier_p90=le(ep['m']['p90_px'], ep['n']['p90_px']),
        earlier_translation=le(earlier['m']['pose']['ALL'].get('translation_median_cm'), earlier['n']['pose']['ALL'].get('translation_median_cm')))
    verdict = ('INCONCLUSIVE_INSUFFICIENT_COVERAGE' if not checks['common_frames'] else
               'SIGNAL_FOR_CONFIRMATION' if all(checks.values()) else 'NO_SIGNAL_WITHIN_1000_STEPS')
    write(DOC / 'RESULTS.json', dict(verdict=verdict, checks=checks, common_geometry=pairs, results=results))
    write(DOC / 'CLOSEOUT.json', dict(status='COMPLETE', formal_updates=2000, discarded_probe_updates=10,
        evaluations=6, gpu=gpu(), new_retraining_after_results=0, pushed=False, system_changes=False))
    print('VERDICT', verdict, checks, flush=True)


def driver():
    lock()
    commands = [('train_n', ['train', '--arm', 'n']), ('train_m', ['train', '--arm', 'm']),
                ('evaluate', ['evaluate']), ('report', ['report'])]
    for name, args in commands:
        dest = RAW / f'{name}.log'
        with dest.open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        print('PHASE_FINISHED', name, result.returncode, flush=True)
        if result.returncode:
            write(DOC / f'EXECUTION_ERROR_{name}.json', dict(phase=name, exit_code=result.returncode,
                  logs=str(dest.relative_to(ROOT)), no_automatic_refit=True))
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'probe', 'lock', 'train', 'audit', 'evaluate', 'report', 'driver'])
    parser.add_argument('--arm', choices=['n', 'm'])
    args = parser.parse_args()
    seed_all()
    if args.phase == 'train':
        assert args.arm
        train(args.arm)
    else:
        globals()[args.phase]()
