"""Train the frozen-protocol line branches and cache synthetic-only logits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from features import BASELINE, BASELINE_SHA, sha
from model import PalletLinePoseHead, compute_loss

HERE = Path(__file__).resolve().parent
TENSOR_KEYS = ('p3', 'p4', 'points', 'boxes', 'point_valid', 'input_shape',
               'gt_points', 'gt_valid', 'gt_support')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class FeatureDataset:
    def __init__(self, run_dir, cache_dir):
        self.run_dir, self.directory = Path(run_dir), Path(cache_dir)
        self.source = read(self.run_dir/'SOURCE_MANIFEST.json')
        self.manifest = read(self.directory/'CACHE_MANIFEST.json')
        completion = read(self.directory/'CACHE_COMPLETE.json')
        if not completion['PASS'] or not completion['complete']:
            raise ValueError('Feature extraction is not complete')
        if completion['manifest_sha256'] != sha(self.directory/'CACHE_MANIFEST.json'):
            raise ValueError('Feature cache manifest changed')
        if self.manifest['source_manifest_sha256'] != sha(self.run_dir/'SOURCE_MANIFEST.json'):
            raise ValueError('Feature cache and source dataset disagree')
        if not np.load(self.directory/'done.npy', mmap_mode='r').all():
            raise ValueError('An uncommitted feature cache row remains')
        self.arrays = {name: np.load(self.directory/spec['file'], mmap_mode='r')
                       for name, spec in self.manifest['arrays'].items()}
        for name, value in self.arrays.items():
            spec = self.manifest['arrays'][name]
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
                raise ValueError(f'Cache array shape/dtype differs: {name}')
        self.indices = np.asarray(self.manifest['record_indices'], dtype=np.int64)
        if not np.array_equal(self.arrays['record_index'], self.indices):
            raise ValueError('Cache row identities differ')
        partitions = np.asarray([self.source['records'][i]['partition'] for i in self.indices])
        usable = self.arrays['matched'] & self.arrays['gt_valid'][:, :8].any(-1)
        self.train_rows = np.flatnonzero((partitions == 'train') & usable)
        self.validation_rows = np.flatnonzero(partitions != 'train')
        self.partitions = partitions
        if not len(self.train_rows):
            raise ValueError('No matched synthetic training instances')

    def batch(self, rows, device='cuda'):
        return {name: torch.from_numpy(np.array(self.arrays[name][rows], copy=True)).to(device)
                for name in TENSOR_KEYS}


def forward(model, batch, arm_spec, lam=1.):
    return model(*(batch[name] for name in TENSOR_KEYS[:6]),
                 lam=lam, geometry_only=arm_spec['geometry_only'])


class ShuffledRows:
    def __init__(self, rows, seed):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.rows)
        self.position, self.epochs_started = 0, 1

    def take(self, count):
        parts = []
        while count:
            available = min(count, len(self.order)-self.position)
            parts.append(self.order[self.position:self.position+available])
            count -= available
            self.position += available
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.rows)
                self.position = 0
                self.epochs_started += 1
        return np.concatenate(parts)

    def state(self):
        return dict(order=self.order, position=self.position,
                    epochs_started=self.epochs_started, rng=self.rng.bit_generator.state)

    def restore(self, state):
        if not np.array_equal(np.sort(state['order']), np.sort(self.rows)):
            raise ValueError('Resume training rows differ')
        self.order = np.asarray(state['order'], np.int64)
        self.position = state['position']
        self.epochs_started = state['epochs_started']
        self.rng.bit_generator.state = state['rng']


def learning_rate(step, protocol):
    settings = protocol['optimizer']
    warmup = settings['warmup_steps']
    if step <= warmup:
        return settings['lr']*step/warmup
    progress = (step-warmup)/(protocol['steps']-warmup)
    last = settings['cosine_final_lr_fraction']
    return settings['lr']*(last+(1-last)*.5*(1+math.cos(math.pi*progress)))


def identity(run_dir, dataset):
    return dict(train_protocol_sha256=sha(run_dir/'TRAIN_PROTOCOL.json'),
                source_manifest_sha256=sha(run_dir/'SOURCE_MANIFEST.json'),
                cache_manifest_sha256=sha(dataset.directory/'CACHE_MANIFEST.json'),
                cache_completion_sha256=sha(dataset.directory/'CACHE_COMPLETE.json'),
                model_source_sha256=sha(HERE/'model.py'), train_source_sha256=sha(Path(__file__)))


def train_one(run_dir, dataset, protocol, arm, seed, *, smoke_steps=0):
    smoke = smoke_steps > 0
    steps = smoke_steps if smoke else protocol['steps']
    destination = run_dir/'smoke'/'training' if smoke else run_dir/'runs'/f'{arm}_seed{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    bindings = identity(run_dir, dataset)
    spec = protocol['arms'][arm]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PalletLinePoseHead(**protocol['model_config']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['optimizer']['lr'],
                                  betas=tuple(protocol['optimizer']['betas']),
                                  weight_decay=protocol['optimizer']['weight_decay'])
    rows = dataset.train_rows[:32] if smoke else dataset.train_rows
    sampler = ShuffledRows(rows, seed)
    checkpoint = destination/'last.pt'
    start_step = 0
    initial_probe = None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu')
        if any(saved.get(key) != value for key, value in bindings.items()):
            raise ValueError(f'Resume source/protocol changed: {destination}')
        if (saved['arm'], saved['seed'], saved['smoke'], saved['expected_steps']) != (arm, seed, smoke, steps):
            raise ValueError('Resume arm/seed/budget differs')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        sampler.restore(saved['sampler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        start_step = saved['step']
        initial_probe = saved.get('initial_probe')
    fixed_batch = dataset.batch(rows[:min(16, len(rows))])
    if initial_probe is None:
        model.eval()
        with torch.no_grad():
            losses = compute_loss(forward(model, fixed_batch, spec), fixed_batch['gt_points'],
                                  fixed_batch['gt_valid'], spec['corner_weight'], fixed_batch['gt_support'])
        initial_probe = {key: float(losses[key]) for key in ('loss', 'line_loss', 'corner_loss')}
    if start_step == 0:
        model.train()
        outputs = forward(model, fixed_batch, spec)
        losses = compute_loss(outputs, fixed_batch['gt_points'], fixed_batch['gt_valid'],
                              spec['corner_weight'], fixed_batch['gt_support'])
        selected = [(name, value) for name, value in model.named_parameters()
                    if name in ('adapt3.0.weight', 'adapt4.0.weight', 'scorer.2.weight', 'null_scorer.2.weight')]
        gradients = torch.autograd.grad(losses['corner_loss'], [p for _, p in selected], allow_unused=True)
        gradient_audit = {name: float(g.norm()) if g is not None else None for (name, _), g in zip(selected, gradients)}
        if not all(value is None or math.isfinite(value) for value in gradient_audit.values()):
            raise ValueError('Nonfinite actual-data corner gradient')
        write(destination/'GRADIENT_AUDIT.json', dict(complete=True, arm=arm, seed=seed,
              actual_synthetic_record_indices=dataset.indices[rows[:len(fixed_batch['points'])]].tolist(),
              corner_loss=float(losses['corner_loss'].detach()), gradients=gradient_audit,
              semantics='Image adapter gradients are expected to be zero in geometry-only control.', **bindings))
    model.train()
    total_start = time.perf_counter()
    last_record = None
    for step in range(start_step+1, steps+1):
        begin = time.perf_counter()
        batch_rows = sampler.take(protocol['batch'])
        batch = dataset.batch(batch_rows)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, protocol)
        for group in optimizer.param_groups:
            group['lr'] = lr
        output = forward(model, batch, spec, protocol['training_lambda'])
        losses = compute_loss(output, batch['gt_points'], batch['gt_valid'],
                              spec['corner_weight'], batch['gt_support'])
        if not torch.isfinite(losses['loss']):
            raise ValueError(f'Nonfinite training loss at {arm}/{seed}/{step}')
        losses['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['optimizer']['gradient_clip_norm'],
                                             error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % protocol['log_every'] == 0 or step == steps:
            torch.cuda.synchronize()
            last_record = dict(step=step, arm=arm, seed=seed, lr=lr, gradient_norm=float(norm),
                **{key: float(value.detach()) for key, value in losses.items()},
                seconds_per_step=time.perf_counter()-begin,
                elapsed_seconds=time.perf_counter()-total_start,
                sampler_epochs_started=sampler.epochs_started,
                observed_training_frames=len(rows), exposed_instances=step*protocol['batch'])
            with (destination/'HISTORY.jsonl').open('a') as handle:
                handle.write(json.dumps(last_record, allow_nan=False)+'\n')
            write(destination/'PROGRESS.json', dict(complete=False, expected_steps=steps, **last_record))
            print(f'{arm} seed{seed} step{step}/{steps} loss={last_record["loss"]:.4f} '
                  f'line={last_record["line_loss"]:.4f} corner={last_record["corner_loss"]:.4f}', flush=True)
        if step % protocol['checkpoint_every'] == 0 or step == steps:
            payload = dict(schema='pallet_line_pose_checkpoint_v1', complete=not smoke and step == protocol['steps'],
                smoke=smoke, expected_steps=steps, step=step, arm=arm, seed=seed,
                model_config=protocol['model_config'], model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), sampler_state=sampler.state(),
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all(),
                baseline_checkpoint=str(BASELINE), baseline_checkpoint_sha256=BASELINE_SHA,
                initial_probe=initial_probe, updated_at_utc=datetime.now(timezone.utc).isoformat(), **bindings)
            temporary = destination/'last.pending.pt'
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu')
    if saved['step'] != steps or not all(torch.isfinite(v).all() for v in saved['model_state_dict'].values()):
        raise ValueError('Final checkpoint does not prove finite completed training')
    model.eval()
    with torch.no_grad():
        losses = compute_loss(forward(model, fixed_batch, spec), fixed_batch['gt_points'],
                              fixed_batch['gt_valid'], spec['corner_weight'], fixed_batch['gt_support'])
    final_probe = {key: float(losses[key]) for key in ('loss', 'line_loss', 'corner_loss')}
    completion = dict(complete=True, PASS=True, smoke=smoke, arm=arm, seed=seed, step=steps,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), initial_probe=initial_probe,
        final_probe=final_probe, training_frames=len(rows), source_train_frames=int((dataset.partitions == 'train').sum()),
        head_parameters=sum(p.numel() for p in model.parameters()), **bindings)
    if smoke:
        completion['PASS'] = final_probe['loss'] < initial_probe['loss']
        completion['scope'] = 'Actual synthetic overfit/gradient plumbing only; not paper performance or a full model.'
    write(destination/'COMPLETION.json', completion)
    if not completion['PASS']:
        raise ValueError('The actual-data smoke loss did not decrease')
    return model, checkpoint


@torch.no_grad()
def validation_logits(run_dir, dataset, protocol, arm, seed, model, checkpoint):
    destination = run_dir/'validation'
    destination.mkdir(exist_ok=True)
    rows = dataset.validation_rows
    indices = dataset.indices[rows]
    index_path = destination/'validation_indices.npy'
    if index_path.exists():
        if not np.array_equal(np.load(index_path), indices):
            raise ValueError('Validation row order differs')
    else:
        np.save(index_path, indices)
    target = destination/f'{arm}_seed{seed}_logits.npy'
    receipt_path = target.with_suffix('.json')
    if target.exists() and receipt_path.exists():
        saved = read(receipt_path)
        if (saved['complete'] and saved['checkpoint_sha256'] == sha(checkpoint)
                and saved['logits_sha256'] == sha(target)):
            return saved
        raise ValueError('Existing validation logits differ from the final checkpoint')
    temporary = target.with_suffix('.pending.npy')
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(rows), 8, 222))
    model.eval()
    for start in range(0, len(rows), protocol['batch']):
        batch_rows = rows[start:start+protocol['batch']]
        prediction = forward(model, dataset.batch(batch_rows), protocol['arms'][arm])
        logits = prediction['logits'].cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite synthetic inference logits')
        values[start:start+len(batch_rows)] = logits
    values.flush()
    del values
    temporary.replace(target)
    receipt = dict(complete=True, arm=arm, seed=seed, logits=str(target), logits_sha256=sha(target),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), n_frames=len(rows),
        validation_indices_sha256=sha(index_path), **identity(run_dir, dataset))
    write(receipt_path, receipt)
    print(f'Synthetic logits completed: {arm} seed{seed}, {len(rows)} frames', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cache_dir = (args.cache_dir or run_dir/'cache').resolve()
    protocol = read(run_dir/'TRAIN_PROTOCOL.json')
    for path, digest in protocol['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Frozen training source changed: {path}')
    dataset = FeatureDataset(run_dir, cache_dir)
    torch.set_num_threads(4)
    cv_seed = 1
    torch.backends.cudnn.benchmark = False
    if args.smoke_steps:
        if dataset.manifest['stage'] != 'smoke':
            raise ValueError('Smoke training requires the isolated smoke cache')
        train_one(run_dir, dataset, protocol, 'image_joint', cv_seed, smoke_steps=args.smoke_steps)
        return
    if dataset.manifest['stage'] != 'main' or len(dataset.indices) != 60000:
        raise ValueError('Full protocol requires all60000 source records')
    runs = []
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            model, checkpoint = train_one(run_dir, dataset, protocol, arm, seed)
            runs.append(validation_logits(run_dir, dataset, protocol, arm, seed, model, checkpoint))
            del model
            torch.cuda.empty_cache()
    write(run_dir/'LOGITS_MANIFEST.json', dict(schema='pallet_line_pose_logits_v1', complete=True,
        validation_indices=str(run_dir/'validation/validation_indices.npy'), runs=runs,
        expected_runs=len(protocol['seeds'])*len(protocol['arms']), **identity(run_dir, dataset)))
    print('All frozen-budget runs and synthetic logits are complete.', flush=True)


if __name__ == '__main__':
    main()
