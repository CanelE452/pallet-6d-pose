"""Bounded six-group/three-seed action training, with explicit run accounting."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from .action_data import SEEDS, bound, read, rows, sha, write, validate_frame
from .action_models import ARMS, collate, loss_value, make_model, parameter_count, prepared, choose


def source_hashes():
    base = Path(__file__).parent
    return {name: sha(base / name) for name in ('action_data.py', 'action_models.py', 'train_actions.py')}


def protocol():
    return dict(schema='FROZEN_ACTION_LEARNING_PROTOCOL_V1', seeds=list(SEEDS), representative_seed=SEEDS[0],
        optimizer='AdamW', learning_rate=1e-3, weight_decay=1e-4, batch_frames=32, max_epochs=50, patience=5,
        checkpoint_selection='strictly lower validation loss of corresponding objective', device='CPU', threads=1,
        architectures={'MLP': '188->64ReLU->64ReLU->1;16321parameters', 'ROLE_AFFINE': 'two187->1 affine routed by shared is_N0 bit;376parameters'},
        H0='historical uniform-positive softmaxCE plus none; none returns N0;188legacyinputs',
        H1='unique actual poses including explicitN0;multi-positive/all-zero frame-balancedBCE;188inputs with is_N0 finalbit',
        pair='independently trained; actual N0/E0 actions only; E0 independent execution required',
        tie_rule='exactscore tie prefersN0, then numerical physical pose order; IDs not model inputs',
        normalization='sameD train eligible candidate fit; exactly reused forH0/H1; no val/real fitting',
        new_training_limit=18, feature_inputs_include_reference=False)


def costs(start, cpu):
    return dict(wall_seconds=time.monotonic() - start, CPU_seconds=time.process_time() - cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, GPU_seconds=0, device='CPU', threads=1)


def reserve(output, arm, seed, binding):
    """Reserve every new optimizer run, including technical failures, before fit."""
    output.mkdir(parents=True, exist_ok=True); ledger = output / 'TRAINING_ATTEMPTS.jsonl'
    with open(ledger, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.seek(0); entries = [json.loads(x) for x in f if x.strip()]
        if len(entries) >= 18: raise RuntimeError('Approved18 new training attempts exhausted; preserve all attempts')
        if any(x['arm'] == arm and x['seed'] == seed for x in entries): raise RuntimeError('A training attempt exists; explicit reviewed technical retry required, not silent overwrite')
        entry = dict(attempt=len(entries) + 1, arm=arm, seed=seed, state='RESERVED', unix_time=time.time(), **binding)
        f.write(json.dumps(entry) + '\n'); f.flush(); fcntl.flock(f, fcntl.LOCK_UN)
    return entry


def evaluate(model, items, objective):
    model.eval(); totals = Counter(); translations = []; rotations = []; sources = Counter(); action_by_role = {}; loss_numer = 0.; loss_denom = 0
    with torch.inference_mode():
        for offset in range(0, len(items), 32):
            group = items[offset:offset + 32]; batch = collate(group); logits = model(batch[0]).squeeze(-1)
            valid_frames = int(batch[1].any(-1).sum()); lv = float(loss_value(model, batch, objective))
            loss_numer += lv * valid_frames; loss_denom += valid_frames
            for index, item in enumerate(group):
                n = len(item['entries']); totals['frames'] += 1
                if n == 0: totals['zero_action_frames'] += 1; sources['NO_POSE'] += 1; continue
                local = logits[index, :n]; probabilities = local.softmax(0) if objective == 'H0' else local.sigmoid()
                targets = np.array([float(m['success_5cm5deg']) for m in item['metrics']]); ps = probabilities.numpy()
                totals['action_accuracy_sum'] += float(((ps >= .5) == targets).mean())
                totals['action_Brier_sum'] += float(np.square(ps - targets).mean()); totals['action_metric_frames'] += 1
                for j, entry in enumerate(item['entries']):
                    aliases = [] if entry is None else entry.get('source_baseline_aliases', [])
                    role = 'H0_NONE_TO_N0' if entry is None else ('N0' if 'N0' in aliases else 'E0' if 'E0' in aliases else 'OTHER')
                    rr = action_by_role.setdefault(role, dict(actions=0, correct=0, squared_error=0., positive_labels=0))
                    rr['actions'] += 1; rr['correct'] += int((ps[j] >= .5) == targets[j]); rr['squared_error'] += float((ps[j] - targets[j]) ** 2); rr['positive_labels'] += int(targets[j])
                totals['good_action_absent'] += int(not targets.any()); selected = choose(local.numpy(), item['entries'], objective)
                chosen = item['entries'][selected]; metric = item['metrics'][selected]
                aliases = [] if chosen is None else chosen.get('source_baseline_aliases', [])
                role = 'N0' if chosen is None or 'N0' in aliases else ('E0' if 'E0' in aliases else 'OTHER')
                sources[role] += 1; totals['success_5cm5deg'] += int(metric['success_5cm5deg'])
                if metric['translation_cm'] is not None:
                    translations.append(metric['translation_cm']); rotations.append(metric['rotation_deg'])
                    totals['T_gt50cm'] += int(metric['translation_cm'] > 50); totals['T_gt100cm'] += int(metric['translation_cm'] > 100); totals['R_gt45deg'] += int(metric['rotation_deg'] > 45)
                else: totals['no_pose'] += 1
    def dist(values):
        return dict(n=len(values), mean=float(np.mean(values)) if values else None,
            sample_SD=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            median=float(np.median(values)) if values else None, P90=float(np.percentile(values, 90)) if values else None,
            P95=float(np.percentile(values, 95)) if values else None, max=float(max(values)) if values else None)
    mf = totals['action_metric_frames']
    return dict(loss=loss_numer / loss_denom if loss_denom else None, frames=totals['frames'], loss_frames=loss_denom,
        zero_action_frames=totals['zero_action_frames'], success_5cm5deg=totals['success_5cm5deg'], good_action_absent=totals['good_action_absent'],
        translation_cm=dist(translations), rotation_deg=dist(rotations), tails={k: totals[k] for k in ('T_gt50cm', 'T_gt100cm', 'R_gt45deg')},
        selected_action_source=dict(sources), frame_balanced_action_accuracy=totals['action_accuracy_sum'] / mf if mf else None,
        frame_balanced_Brier=totals['action_Brier_sum'] / mf if mf else None,
        per_role={role: dict(**r, accuracy=r['correct'] / r['actions'], Brier=r['squared_error'] / r['actions']) for role, r in action_by_role.items()},
        H0_Brier_caveat='H0 softmax scores vs actual output binarysuccess are descriptive, not its CEtarget or calibrated probabilities' if objective == 'H0' else None)


def refresh_manifest(output):
    records = []
    for path in sorted(output.glob('*/RUN_RECEIPT.json')):
        row = read(path)
        if row.get('complete'): records.append(row)
    attempts_path = output / 'TRAINING_ATTEMPTS.jsonl'; attempts = list(rows(attempts_path)) if attempts_path.exists() else []
    done = Counter(r['arm'] for r in records)
    result = dict(schema='ACTION_TRAINING_MANIFEST_V1', complete=all(done[arm] == 3 for arm in ARMS), runs=records,
        planned_checkpoint_count=21, pending_arms={arm: 3 - done[arm] for arm in ARMS if done[arm] < 3}, actual_new_training_attempts=len(attempts),
        successful_new_training_runs=sum(not r.get('checkpoint_reused', False) for r in records), reused_checkpoints=sum(r.get('checkpoint_reused', False) for r in records),
        representative_seed=SEEDS[0], protocol=protocol(), source_hashes=source_hashes())
    write(output / 'TRAINING_MANIFEST.json', result); return result


def reuse_legacy(legacy_dir, output):
    manifest = read(legacy_dir / 'TRAINING_MANIFEST.json'); output.mkdir(parents=True, exist_ok=True)
    for r in manifest['runs']:
        if r['kind'] != 'L_HEAD': continue
        if r['seed'] not in SEEDS: raise ValueError('Unexpected legacy seed')
        cp = Path(r['checkpoint'])
        if sha(cp) != r['checkpoint_sha256']: raise ValueError('Legacy checkpoint changed')
        ck = torch.load(cp, map_location='cpu', weights_only=False); model = make_model('MLP'); model.load_state_dict(ck['model_state_dict'])
        if ck['input_dim'] != 188 or ck['kind'] != 'L_HEAD': raise ValueError('Historical H0 checkpoint incompatible')
        folder = output / f'D0_H0_MLP_s{r["seed"]}'; folder.mkdir(exist_ok=True)
        receipt = dict(complete=True, arm='D0_H0_MLP', seed=r['seed'], representative=r['seed'] == SEEDS[0], checkpoint_reused=True,
            checkpoint=str(cp), checkpoint_sha256=sha(cp), parameters=parameter_count('MLP'), architecture='MLP', objective='H0', domain='D0', pair=False,
            source_training_manifest_sha256=sha(legacy_dir / 'TRAINING_MANIFEST.json'), new_training_runs=0,
            old_training_cost_reused_only=r['cost'], original_epochs=r['epochs'], normalizer=ck['normalizer'])
        existing = folder / 'RUN_RECEIPT.json'
        if existing.exists() and read(existing) != receipt: raise ValueError('Immutable reused checkpoint receipt changed')
        write(existing, receipt)
    return refresh_manifest(output)


def train(features, feature_seal, labels_dir, output, arm, GO):
    if not GO: raise PermissionError('Actual small-head training requires explicit root GO')
    if arm == 'D0_H0_MLP': raise ValueError('Reuse the three historical checkpoints; do not retrain D0/H0')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    domain, objective, architecture, pair = ARMS[arm]; digest = bound(features, feature_seal); ls = read(labels_dir / 'LABEL_SEAL.json')
    if ls['domain'] != domain or ls['features_sha256'] != digest: raise ValueError('Wrong D or feature-label binding')
    if sha(labels_dir / 'ACTION_TARGETS.jsonl.gz') != ls['labels_sha256'] or sha(labels_dir / 'TRAIN_ONLY_NORMALIZER.json') != ls['normalizer_sha256']: raise ValueError('Label/normalizer bytes changed')
    if not ls['label_support_gate']['passed']:
        write(output / f'{arm}_NOT_RUN.json', dict(status='NOT_APPLICABLE_LABEL_SUPPORT_GATE', gate=ls['label_support_gate'], new_training_runs=0)); return
    if pair:
        fs = read(feature_seal)
        if not fs.get('e0_independently_executed', False): raise ValueError('Pair requires sealed proof of independently executed E0, not N0-copy')
    frames = list(rows(features)); labels = {r['id']: r for r in rows(labels_dir / 'ACTION_TARGETS.jsonl.gz')}; normalizer = read(labels_dir / 'TRAIN_ONLY_NORMALIZER.json')
    if {r['id'] for r in frames} != set(labels): raise ValueError('Frame/label coverage mismatch')
    for r in frames: validate_frame(r)
    items = {split: [prepared(r, labels[r['id']], normalizer, objective, pair) for r in frames if r['split'] == split] for split in ('train', 'validation')}
    if (len(items['train']), len(items['validation'])) != (768, 235): raise ValueError('Original768/235 split required')
    if not any(len(r['entries']) for r in items['validation']): raise ValueError('No validation action loss is available')
    output.mkdir(parents=True, exist_ok=True); write(output / 'TRAINING_PROTOCOL_LOCK.json', protocol())
    binding = dict(features_sha256=digest, feature_seal_sha256=sha(feature_seal), label_seal_sha256=sha(labels_dir / 'LABEL_SEAL.json'), normalizer_sha256=ls['normalizer_sha256'], source_hashes=source_hashes())
    for seed in SEEDS:
        folder = output / f'{arm}_s{seed}'; receipt_path = folder / 'RUN_RECEIPT.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get('complete') and all(receipt.get(k) == v for k, v in binding.items()) and sha(receipt['checkpoint']) == receipt['checkpoint_sha256']: continue
            raise RuntimeError('Existing incomplete or differently bound training attempt; preserve and review')
        attempt = reserve(output, arm, seed, binding); folder.mkdir(); start, cpu = time.monotonic(), time.process_time(); history = []
        try:
            torch.manual_seed(seed); rng = np.random.default_rng(seed); model = make_model(architecture)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001); best, wait, best_epoch = float('inf'), 0, None
            for epoch in range(50):
                model.train(); order = rng.permutation(len(items['train'])); total, denominator = 0., 0
                for offset in range(0, len(order), 32):
                    group = [items['train'][int(i)] for i in order[offset:offset + 32]]; batch = collate(group)
                    valid = int(batch[1].any(-1).sum())
                    if valid == 0: continue
                    optimizer.zero_grad(set_to_none=True); loss = loss_value(model, batch, objective)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    loss.backward(); optimizer.step(); total += float(loss.detach()) * valid; denominator += valid
                train_metrics = evaluate(model, items['train'], objective); val_metrics = evaluate(model, items['validation'], objective)
                vl = val_metrics['loss']
                if vl is None or not math.isfinite(vl): raise FloatingPointError('No finite validation loss')
                history.append(dict(epoch=epoch + 1, training_optimizer_loss=total / denominator if denominator else None, train=train_metrics, validation=val_metrics))
                checkpoint = dict(arm=arm, architecture=architecture, objective=objective, domain=domain, pair=pair, seed=seed, epoch=epoch + 1,
                    input_dim=188, model_state_dict=model.state_dict(), normalizer=normalizer, config=protocol(), bindings=binding)
                if vl < best:
                    best, best_epoch, wait = vl, epoch + 1, 0; torch.save(checkpoint, folder / 'best.pt')
                else: wait += 1
                torch.save(checkpoint, folder / 'last.pt'); write(folder / 'HISTORY.json', history)
                if wait >= 5: break
                if time.monotonic() - start > 3600: raise RuntimeError('Approved60minute per-run cap reached')
            receipt = dict(complete=True, arm=arm, seed=seed, representative=seed == SEEDS[0], checkpoint_reused=False,
                checkpoint=str(folder / 'best.pt'), checkpoint_sha256=sha(folder / 'best.pt'), last_checkpoint_sha256=sha(folder / 'last.pt'),
                history_sha256=sha(folder / 'HISTORY.json'), parameters=parameter_count(architecture), architecture=architecture, objective=objective, domain=domain, pair=pair,
                epochs=len(history), best_epoch=best_epoch, best_validation_loss=best, best_metrics=history[best_epoch - 1], actual_new_training_runs=1,
                optimizer_steps=math.ceil(768 / 32) * len(history), training_frame_exposures=768 * len(history), attempt=attempt['attempt'], cost=costs(start, cpu), **binding)
            write(receipt_path, receipt); refresh_manifest(output)
            print(json.dumps(dict(stage='TRAINED_ACTION_HEAD', arm=arm, seed=seed, epochs=len(history), best_epoch=best_epoch, validation_loss=best, wall_seconds=receipt['cost']['wall_seconds'])), flush=True)
        except BaseException as error:
            write(folder / 'FAILED_TECHNICAL.json', dict(complete=False, status='FAILED_TECHNICAL', arm=arm, seed=seed, error=repr(error), traceback=traceback.format_exc(),
                history_epochs=len(history), cost=costs(start, cpu), attempt=attempt, bindings=binding)); refresh_manifest(output); raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['reuse-legacy', 'train', 'manifest'])
    p.add_argument('--features', type=Path); p.add_argument('--feature-seal', type=Path); p.add_argument('--labels-dir', type=Path)
    p.add_argument('--legacy-dir', type=Path); p.add_argument('--output', type=Path, required=True); p.add_argument('--arm', choices=list(ARMS)); p.add_argument('--training-GO', action='store_true'); a = p.parse_args()
    if a.stage == 'reuse-legacy': reuse_legacy(a.legacy_dir, a.output)
    elif a.stage == 'train': train(a.features, a.feature_seal, a.labels_dir, a.output, a.arm, a.training_GO)
    else: refresh_manifest(a.output)


if __name__ == '__main__': main()
