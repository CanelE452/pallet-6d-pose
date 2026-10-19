"""Six fixed paired position-quality fits, with counted failures and resume."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write, append_log
from . import models as M


def protocol():
    return dict(schema='MATCHED_LOCAL_SCORER_LEARNING_PROTOCOL_V1', arms=list(M.ARMS), seeds=list(M.SEEDS),
        representative_seed=M.SEEDS[0], architecture='concat(normalized numeric values,validity,c2)->64ReLU->64ReLU->1Softplus',
        optimizer='AdamW', learning_rate=.001, weight_decay=.0001, batch_frames=32, max_epochs=50, patience=5,
        target='log1p(norm(actual_applied_xy-aligned_virtual_corner)/2px)', objective='SmoothL1(beta1), candidate->corner->frame mean',
        partial_supervision='Fixed partial path excluded from point scorer loss and selection',
        normalizer='TRAIN valid numeric values only; raw validity appended',
        checkpoint='first strict minimum own validation position-quality loss',
        initialization='torch.manual_seed(seed) immediately before identical Scorer construction',
        order='np.random.default_rng(seed).permutation(training_frame_count) once each epoch',
        parameters_expected=27585, inference='one exact argmin candidate; q0-first fixed numeric ties',
        CPU_threads=1, GPU_jobs=0, new_H1_fits=0, new_real_supervision=False)


def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def cost(start, cpu):
    return dict(wall_seconds=time.monotonic() - start, CPU_seconds=time.process_time() - cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, device='CPU', threads=1)


def _log(root, event, **fields): append_log(root / 'RUN_LOG.jsonl', dict(event=event, **fields))


def reserve(root, arm, seed, resume):
    if arm not in M.ARMS or seed not in M.SEEDS: raise ValueError('No additional model or seed authorized')
    path = root / 'GLOBAL_FIT_ATTEMPTS.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.seek(0); old = [json.loads(s) for s in f if s.strip()]
        prior = [r for r in old if r['arm'] == arm and r['seed'] == seed]
        if prior and not resume: raise RuntimeError('Existing attempts preserved; explicit same-setting technical resume required')
        row = dict(attempt=len(old) + 1, arm=arm, seed=seed, status='OPTIMIZER_CONSTRUCTION_RESERVED',
                   timestamp=time.time(), resume=bool(resume), previous_attempts=[r['attempt'] for r in prior])
        f.write(json.dumps(row) + '\n'); f.flush(); fcntl.flock(f, fcntl.LOCK_UN)
    return row


def fit(items, norm, root, arm, seed, bindings, *, resume=False, technical_amendment=None):
    folder = root / 'training' / f'OBS_{arm}_s{seed}'
    if (folder / 'RUN_RECEIPT.json').exists():
        complete = read(folder / 'RUN_RECEIPT.json')
        if complete['bindings'] != bindings or not complete['complete']: raise ValueError('Completed fit binding differs')
        return complete
    if folder.exists() and not resume: raise FileExistsError('Do not overwrite unfinished fit')
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = M.Scorer(arm)
    initial = M.state_hash(model); assert sum(p.numel() for p in model.parameters()) == 27585
    old = None
    if resume:
        source = folder / 'last.pt'
        if not source.exists(): source = folder / 'initial.pt'
        old = torch.load(source, map_location='cpu', weights_only=False)
        if old['bindings']['scientific'] != bindings['scientific']: raise ValueError('Scientific resume contract changed')
        if old['bindings']['execution_source'] != bindings['execution_source']:
            if not technical_amendment or not read(technical_amendment).get('scientific_settings_unchanged'):
                raise ValueError('Changed execution source requires preserved technical amendment')
        if old['initial_state_sha256'] != initial: raise ValueError('Paired initialization changed on resume')
    attempt = reserve(root, arm, seed, resume); attempt_dir = folder / 'attempts' / f"{attempt['attempt']:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False); started, cpu = time.monotonic(), time.process_time()
    steps = 0; optimizer_created = False; history = []; best = float('inf'); best_epoch = None; wait = 0; next_epoch = 0
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001); optimizer_created = True
        initial_optimizer_signature = digest(optimizer.state_dict())
        if old is not None:
            model.load_state_dict(old['model_state_dict']); optimizer.load_state_dict(old['optimizer_state_dict'])
            rng.bit_generator.state = old['numpy_rng_state']; torch.set_rng_state(old['torch_rng_state'])
            state = old['runtime']; history = state['history']; best = state['best']; best_epoch = state['best_epoch']; wait = state['wait']; next_epoch = state['next_epoch']
        def checkpoint(epoch_next):
            return dict(arm=arm, seed=seed, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                numpy_rng_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state(), normalizer=norm,
                bindings=bindings, initial_state_sha256=initial, initial_optimizer_signature=initial_optimizer_signature,
                runtime=dict(next_epoch=epoch_next, best=best, best_epoch=best_epoch, wait=wait,
                    history=history, attempt=attempt['attempt']), protocol=protocol())
        if old is None: torch.save(checkpoint(0), folder / 'initial.pt')
        _log(root, 'OPTIMIZER_CREATED', arm=arm, seed=seed, attempt=attempt['attempt'], initial_state_sha256=initial,
             resume_epoch_index=next_epoch, initial_optimizer_signature=initial_optimizer_signature)
        for epoch_index in range(next_epoch, 50):
            if wait >= 5: break
            ts, tc = time.monotonic(), time.process_time(); order = rng.permutation(len(items['train']))
            model.train(); total = 0.; frames = 0
            for offset in range(0, len(order), 32):
                batch = M.collate([items['train'][int(i)] for i in order[offset:offset + 32]])
                if batch is None: continue
                optimizer.zero_grad(set_to_none=True); loss = M.loss_value(model, batch)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite position-quality loss')
                loss.backward(); optimizer.step(); steps += 1
                total += float(loss.detach()) * batch['frame_count']; frames += batch['frame_count']
            train = M.evaluate_loss(model, items['train']); validation = M.evaluate_loss(model, items['validation'])
            value = validation['loss']
            if value is None or not np.isfinite(value): raise FloatingPointError('No finite synthetic validation loss')
            history.append(dict(epoch_index=epoch_index + 1, training_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                training_optimizer_loss=total / frames if frames else None, train=train, validation=validation, cost=cost(ts, tc)))
            improved = value < best
            if improved: best, best_epoch, wait = value, epoch_index + 1, 0
            else: wait += 1
            saved = checkpoint(epoch_index + 1)
            if improved: torch.save(saved, folder / 'best.pt')
            torch.save(saved, folder / 'last.pt'); write(folder / 'HISTORY.json', history, overwrite=True)
            _log(root, 'TRAIN_EPOCH', arm=arm, seed=seed, attempt=attempt['attempt'], epoch_index=epoch_index + 1,
                 training_loss=train['loss'], validation_loss=value, optimizer_steps_this_attempt=steps)
        if not history or best_epoch is None: raise RuntimeError('No completed supervised epoch')
        result = dict(schema='MATCHED_LOCAL_SCORER_FIT_RECEIPT_V1', complete=True, status='COMPLETE', arm=arm, seed=seed,
            parameters=sum(p.numel() for p in model.parameters()), epochs=len(history), best_epoch=best_epoch,
            best_validation_loss=best, optimizer_created=True, optimizer_steps_this_attempt=steps,
            initial_state_sha256=initial, initial_optimizer_signature=initial_optimizer_signature,
            checkpoint_sha256=sha(folder / 'best.pt'), last_checkpoint_sha256=sha(folder / 'last.pt'),
            history_sha256=sha(folder / 'HISTORY.json'), attempt=attempt['attempt'], resumed=resume,
            bindings=bindings, cost=cost(started, cpu))
        write(folder / 'RUN_RECEIPT.json', result); write(attempt_dir / 'COMPLETE.json', result)
        _log(root, 'FIT_COMPLETE', arm=arm, seed=seed, attempt=attempt['attempt'])
        return result
    except BaseException as exc:
        for name in ('best.pt', 'last.pt', 'initial.pt', 'HISTORY.json'):
            p = folder / name
            if p.exists(): (attempt_dir / name).write_bytes(p.read_bytes())
        failure = dict(status='FAILED_TECHNICAL', arm=arm, seed=seed, attempt=attempt['attempt'], optimizer_created=optimizer_created,
            optimizer_steps_this_attempt=steps, finished_history_epochs=len(history), error=repr(exc), traceback=traceback.format_exc(),
            bindings=bindings, cost=cost(started, cpu))
        write(attempt_dir / 'FAILED_TECHNICAL.json', failure); _log(root, 'FIT_FAILED', arm=arm, seed=seed, attempt=attempt['attempt'], error=repr(exc)); raise


def prepare(features, targets, root, plan):
    records = list(rows(features)); labels = {r['id']: r for r in rows(targets)}
    if len(records) != 1003 or len({r['id'] for r in records}) != 1003 or {r['id'] for r in records} != set(labels):
        raise ValueError('Exact original 1003 synthetic feature/target joins required')
    split = {name: [r for r in records if r['split'] == name] for name in ('train', 'validation')}
    if [len(split[s]) for s in ('train', 'validation')] != [768, 235]: raise ValueError('Original 768/235 split required')
    if not any(len(c['candidate_ids']) > 1 and not c['partial_replaces_full_likelihood'] for r in split['train'] for c in r['corner_rows']):
        raise RuntimeError('No actual observation alternatives; do not run meaningless six fits')
    norm = M.normalizer(split['train']); normalizer_path = root / 'TRAIN_ONLY_NORMALIZER.json'
    if normalizer_path.exists():
        if read(normalizer_path) != norm: raise ValueError('Frozen train-only normalizer changed')
    else: write(normalizer_path, norm)
    items = {s: [M.prepare_frame(r, labels[r['id']], norm) for r in split[s]] for s in split}
    if any(not any(c['valid'].any() for i in items[s] for c in i['corners']) for s in split):
        raise RuntimeError('Missing actual synthetic point supervision')
    source = Path(__file__)
    bindings = dict(scientific=dict(features_sha256=sha(features), targets_sha256=sha(targets), normalizer_sha256=sha(normalizer_path),
        experiment_plan_sha256=sha(plan), protocol_sha256=digest(protocol()), model_source_sha256=sha(source.with_name('models.py')),
        feature_source_sha256=sha(source.with_name('features.py'))), execution_source=dict(training_source_sha256=sha(source)))
    return items, norm, bindings


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--features', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--arm', choices=M.ARMS); p.add_argument('--seed', type=int, choices=M.SEEDS)
    p.add_argument('--resume-technical', action='store_true'); p.add_argument('--technical-amendment', type=Path)
    a = p.parse_args(); torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not read(a.plan)['status'].startswith('FROZEN'): raise ValueError('Training requires frozen scientific plan')
    items, norm, binding = prepare(a.features, a.targets, a.run_dir, a.plan)
    results = []
    for seed in ((a.seed,) if a.seed else M.SEEDS):
        for arm in ((a.arm,) if a.arm else M.ARMS):
            results.append(fit(items, norm, a.run_dir, arm, seed, binding, resume=a.resume_technical,
                               technical_amendment=a.technical_amendment))
    completed = []
    for path in sorted((a.run_dir / 'training').glob('*/RUN_RECEIPT.json')): completed.append(read(path))
    write(a.run_dir / 'TRAINING_MANIFEST.json', dict(schema='MATCHED_LOCAL_OBSERVATION_TRAINING_MANIFEST_V1',
        protocol=protocol(), completed_fits=len(completed), runs=completed,
        actual_optimizer_attempts=sum(1 for _ in rows(a.run_dir / 'GLOBAL_FIT_ATTEMPTS.jsonl'))), overwrite=True)
    print(json.dumps(dict(completed_fits=len(completed), newly_completed_or_reused=len(results))))


if __name__ == '__main__': main()
