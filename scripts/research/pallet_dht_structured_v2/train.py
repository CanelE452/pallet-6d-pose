"""Matched synthetic-only training of whole-layout verifiers on frozen P4.

Main runs always use the registered budget. Explicit smoke output lives under
smoke_training and cannot be loaded as a completed main checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import cache as C
from . import data_ops as D
from . import model as M
from . import proposals as P

ROOT = Path(__file__).resolve().parents[3]
PROBE_STEPS = (1, 2, 4, 16, 100, 500, 1000, 2000)


def tensor_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_torch(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending.pth')
    torch.save(value, pending)
    pending.replace(path)


def required_sources():
    paths = {str(Path(p).resolve()) for p in (__file__, C.__file__, D.__file__, M.__file__, P.__file__)}
    paths |= set(C.source_sha256())
    return {p: C.sha(p) for p in sorted(paths)}


def verify_bindings(run):
    run = Path(run).resolve()
    if not (run/'PURPOSE.md').is_file():
        raise ValueError('Experiment PURPOSE.md required')
    protocol = C.read(run/'PROTOCOL.json')
    freeze = C.read(run/'SOURCE_FREEZE.json')
    cache = C.read(run/'CACHE_COMPLETION.json')
    spec_path = run/'IMPLEMENTATION_SPEC.json'
    if not (freeze.get('complete') and cache.get('complete') and cache.get('PASS')):
        raise ValueError('Completed source freeze/cache required')
    actual = dict(protocol_sha256=C.sha(run/'PROTOCOL.json'),
        implementation_spec_sha256=C.sha(spec_path), cache_completion_sha256=C.sha(run/'CACHE_COMPLETION.json'))
    for key, digest in actual.items():
        if freeze[key] != digest:
            raise ValueError(f'Frozen binding changed: {key}')
    sources = required_sources()
    for path, digest in sources.items():
        if freeze['source_sha256'].get(path) != digest:
            raise ValueError(f'Unfrozen/changed training dependency: {path}')
    if cache['bindings']['protocol_sha256'] != actual['protocol_sha256']:
        raise ValueError('Cache/protocol binding differs')
    cfg = protocol['training']
    if cfg['steps'] != 2000 or cfg['batch'] != 16 or cfg['precision'] != 'FP32' or cfg['optimizer'] != 'AdamW':
        raise ValueError('Registered2000-update B16 FP32 AdamW recipe required')
    if tuple(cfg['proposal_state_probabilities']) != D.STATES or tuple(cfg['proposal_state_probabilities'].values()) != D.STATE_PROBABILITIES:
        raise ValueError('Data-op state order/probabilities differ from protocol')
    bindings = {**actual, 'source_freeze_sha256': C.sha(run/'SOURCE_FREEZE.json'),
        'source_sha256': sources, 'cache_records_sha256': cache['bindings']['cache_records_sha256'],
        'cache_arrays_sha256': cache['bindings']['cache_arrays_sha256']}
    return protocol, bindings


def plans(indices, steps, batch, seed):
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    sequence = []
    while len(sequence) < steps*batch:
        sequence.extend(rng.permutation(indices).tolist())
    order = np.asarray(sequence[:steps*batch], np.int64).reshape(steps, batch)
    state_rng = np.random.default_rng(seed+12345)
    states = state_rng.choice(np.asarray(D.STATES), size=steps*batch, p=D.STATE_PROBABILITIES).reshape(steps, batch)
    # Keep the specified RNG consumption order: all states, then all variates.
    variants = state_rng.random((steps*batch, 4)).reshape(steps, batch, 4)
    return order, states, variants


def plan_hashes(order, states, variants):
    return dict(image_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        state_order_sha256=hashlib.sha256(json.dumps(states.tolist(), separators=(',', ':')).encode()).hexdigest(),
        variants_sha256=hashlib.sha256(variants.tobytes()).hexdigest())


def learning_rate(step, expected_steps, initial):
    progress = (step-1)/max(1, expected_steps-1)
    return initial*(.1+.9*.5*(1.+math.cos(math.pi*progress)))


def move(values, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in values.items()}


def group_gradient_norms(model):
    groups = ('visual_projection', 'corner_encoder', 'edge_encoder', 'line_encoder', 'interaction',
              'quality', 'corner_error', 'context_encoder', 'semantic_embedding')
    result = {}
    for group in groups:
        terms = [p.grad.detach().double().square().sum() for name, p in model.named_parameters()
                 if name.startswith(group+'.') and p.grad is not None]
        result[group] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
    return result


def load_trained(run, arm, seed=1, device='cpu'):
    run = Path(run).resolve()
    protocol, bindings = verify_bindings(run)
    cell = run/'runs'/f'{arm}_seed{seed}'
    completion = C.read(cell/'COMPLETION.json')
    if not (completion.get('complete') and completion.get('PASS') and completion.get('stage') == 'main'
            and completion['bindings'] == bindings and completion['arm'] == arm and completion['seed'] == seed
            and completion['optimizer_steps'] == completion['expected_optimizer_steps'] == protocol['training']['steps']):
        raise ValueError('Completed, bound main verifier checkpoint required')
    path = cell/'checkpoint_final.pth'
    if C.sha(path) != completion['checkpoint_sha256']:
        raise ValueError('Final checkpoint hash changed')
    for file, key in [('history.json', 'history_sha256'), ('TRACE.jsonl', 'trace_sha256'),
                      ('GRADIENT_PROBES.json', 'gradient_probe_sha256'), ('SETUP.json', 'setup_sha256')]:
        if C.sha(cell/file) != completion[key]:
            raise ValueError(f'Final training evidence changed: {file}')
    saved = torch.load(path, map_location='cpu')
    if not (saved['schema'] == 'pallet_dht_structured_checkpoint_v2' and saved['complete'] and saved['stage'] == 'main'
            and saved['arm'] == arm and saved['seed'] == seed and saved['bindings'] == bindings
            and saved['optimizer_steps'] == saved['expected_optimizer_steps'] == protocol['training']['steps']
            and saved['final_state_tensor_sha256'] == tensor_sha(saved['state_dict'])):
        raise ValueError('Checkpoint contract/weights differ from completed training')
    model = M.LayoutVerifier(**saved['model_config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to(device).eval()
    return model


def _train(run, arm, seed, device, *, smoke_steps=None):
    run = Path(run).resolve()
    protocol, bindings = verify_bindings(run)
    cfg = protocol['training']
    if arm not in protocol['architecture']['arms'] or seed not in cfg['seeds']:
        raise ValueError('Arm/seed not registered')
    if smoke_steps is not None and not 1 <= smoke_steps <= 16:
        raise ValueError('Explicit smoke supports1..16 updates only')
    stage = 'smoke' if smoke_steps is not None else 'main'
    steps = smoke_steps if smoke_steps is not None else cfg['steps']
    cell = run/('smoke_training' if stage == 'smoke' else 'runs')/f'{arm}_seed{seed}'
    if (cell/'COMPLETION.json').exists():
        old = C.read(cell/'COMPLETION.json')
        if not (old['complete'] and old['PASS'] and old['bindings'] == bindings and old['stage'] == stage and old['optimizer_steps'] == steps):
            raise ValueError('Existing completed cell has a different contract')
        if stage == 'main':
            model = load_trained(run, arm, seed, 'cpu')
            del model
        elif C.sha(cell/'checkpoint_final.pth') != old['checkpoint_sha256']:
            raise ValueError('Changed smoke checkpoint')
        return old
    if cell.exists() and any(cell.iterdir()):
        raise ValueError('Incomplete cell exists; preserve and explicitly restart rather than silently overwrite/resume')
    cell.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = C.CachedData(run)
    training = np.asarray(data.populations['train'], np.int64)
    if len(training) != 1792 or len(set(training.tolist())) != 1792:
        raise ValueError('All registered1792 training records required, without eligibility filtering')
    if set(training.tolist()) & set(sum([data.populations[p] for p in ('calibration', 'synth_val', 'real_dev')], [])):
        raise ValueError('Training has evaluation/calibration overlap')
    # Smoke consumes the same initial plan as main, including state-RNG ordering.
    order, states, variants = plans(training, cfg['steps'], cfg['batch'], seed)
    plan_binding = plan_hashes(order, states, variants)
    architecture = {k: protocol['architecture'][k] for k in ('visual_channels', 'width', 'n_layers')}
    model = M.LayoutVerifier(arm=arm, **architecture).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 131458:
        raise ValueError('Matched131458-parameter verifier required')
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_sha = tensor_sha(initial)
    save_torch(cell/'INITIAL_STATE.pth', initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    C.write(cell/'SETUP.json', dict(schema='pallet_dht_structured_training_setup_v2', complete=True, stage=stage,
        arm=arm, seed=seed, parameters=parameters, bindings=bindings, expected_optimizer_steps=steps,
        registered_main_steps=cfg['steps'], batch=cfg['batch'], train_frames=len(training),
        calibration_frames=len(data.populations['calibration']), synthetic_validation_frames=len(data.populations['synth_val']),
        real_frames_used=0, initial_state_tensor_sha256=initial_sha,
        initial_state_file_sha256=C.sha(cell/'INITIAL_STATE.pth'), plan_hashes=plan_binding,
        state_generation='np.default_rng(seed+12345): choice(all32000states) then random((32000,4)); smoke takes the same main prefix.',
        precision='FP32', AMP=False, cudnn_benchmark=False, cudnn_allow_tf32=False, matmul_allow_tf32=False,
        optimizer='AdamW', optimizer_lr_initial=cfg['lr'], optimizer_weight_decay=cfg['weight_decay'],
        torch_version=torch.__version__, numpy_version=np.__version__, model_config=model.model_config))
    model.train()
    history, probes, window = [], [], []
    corner_exposures = 0
    successful_steps = 0
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for step in range(1, steps+1):
            indices, state, variation = order[step-1], states[step-1], variants[step-1]
            cpu_inputs, cpu_targets = data.batch(indices, device='cpu')
            inputs, candidate_valid, targets = D.prepare_batch(cpu_inputs, cpu_targets, state.tolist(), variation)
            inputs, targets = move(inputs, device), move(targets, device)
            candidate_valid = candidate_valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, cfg['steps'], cfg['lr'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            cost, diagnostics = model(inputs, return_diagnostics=True)
            if not bool(torch.isfinite(cost).all()):
                raise ValueError(f'Nonfinite model cost at step{step}')
            loss, components = D.training_loss(cost, diagnostics, inputs, candidate_valid, targets, cfg)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f'Nonfinite training loss at step{step}')
            loss.backward()
            if step in PROBE_STEPS or step == steps:
                probes.append(dict(step=step, loss=float(loss.detach()), components=components,
                    gradient_norm_by_group=group_gradient_norms(model), before_global_clip=True,
                    line_cues_max_abs=float(diagnostics['line_cues'].detach().abs().max()),
                    line_cues_are_zero=bool(torch.count_nonzero(diagnostics['line_cues']) == 0),
                    corner_coverage_mean=float(diagnostics['corner_input_coverage'].mean()),
                    edge_coverage_mean=float(diagnostics['edge_input_coverage'].mean())))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            successful_steps += 1
            corner_exposures += components['supervised_corners']
            row = dict(step=step, indices=indices.tolist(), states=state.tolist(), variants=variation.tolist(),
                supervised_corners=components['supervised_corners'], valid_candidate_count=int(candidate_valid.sum()))
            with (cell/'TRACE.jsonl').open('a') as f:
                f.write(json.dumps(row, separators=(',', ':'))+'\n')
            window.append(dict(loss=float(loss.detach()), gradient_norm=float(norm), **components))
            if step % cfg['log_every'] == 0 or step == steps:
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError('Nonfinite updated model parameter')
                summary = {k: float(np.mean([v[k] for v in window])) for k in ('loss', 'ranking', 'regression', 'corner', 'gradient_norm')}
                entry = dict(step=step, optimizer_steps=successful_steps, mean_since_previous_log=summary,
                    window_steps=len(window), lr=lr, elapsed_seconds=time.perf_counter()-started)
                history.append(entry)
                window = []
                C.write(cell/'history.json', history)
                C.write(cell/'GRADIENT_PROBES.json', dict(complete=False, records=probes, before_global_clip=True,
                    backbone_forward_calls=0, interpretation='Group gradients show trainable dependence, not calibrated evidence reliability.'))
                print(f'{arm} seed{seed} {stage} step{step}/{steps} loss={summary["loss"]:.6f} lr={lr:.8g}', flush=True)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        current_protocol, final_bindings = verify_bindings(run)
        if final_bindings != bindings or successful_steps != steps:
            raise ValueError('Training inputs/source/budget changed during execution')
        final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        named_params = set(dict(model.named_parameters()))
        changed = sum(int(torch.count_nonzero(final[k] != initial[k])) for k in named_params)
        if changed == 0:
            raise ValueError('No learned parameter changed')
        if any(not torch.equal(final[k], initial[k]) for k in final if k not in named_params):
            raise ValueError('Fixed sampler/geometry buffer changed')
        optimizer_steps = Counter(int(s['step'].item()) for s in optimizer.state.values() if 'step' in s)
        if set(optimizer_steps) != {steps}:
            raise ValueError('Optimizer parameter update counters differ from completed steps')
        C.write(cell/'GRADIENT_PROBES.json', dict(complete=True, records=probes, before_global_clip=True,
            backbone_forward_calls=0, interpretation='Group gradients show trainable dependence, not calibrated evidence reliability.'))
        checkpoint = cell/'checkpoint_final.pth'
        saved = dict(schema='pallet_dht_structured_checkpoint_v2', complete=True, stage=stage, arm=arm, seed=seed,
            parameters=parameters, optimizer_steps=successful_steps, expected_optimizer_steps=steps,
            registered_main_steps=cfg['steps'], model_config=model.model_config, state_dict=final,
            optimizer=optimizer.state_dict(), bindings=bindings, plan_hashes=plan_binding,
            initial_state_tensor_sha256=initial_sha, final_state_tensor_sha256=tensor_sha(final),
            trace_sha256=C.sha(cell/'TRACE.jsonl'), history_sha256=C.sha(cell/'history.json'))
        save_torch(checkpoint, saved)
        result = dict(schema='pallet_dht_structured_training_cell_v2', complete=True, PASS=True, stage=stage,
            arm=arm, seed=seed, parameters=parameters, optimizer_steps=successful_steps, expected_optimizer_steps=steps,
            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=C.sha(checkpoint), bindings=bindings,
            initial_state_tensor_sha256=initial_sha, final_state_tensor_sha256=tensor_sha(final),
            changed_parameter_values=changed, optimizer_parameter_step_histogram=dict(optimizer_steps),
            plan_hashes=plan_binding, trace_sha256=C.sha(cell/'TRACE.jsonl'), history_sha256=C.sha(cell/'history.json'),
            gradient_probe_sha256=C.sha(cell/'GRADIENT_PROBES.json'), setup_sha256=C.sha(cell/'SETUP.json'),
            initial_state_file_sha256=C.sha(cell/'INITIAL_STATE.pth'), total_batch_exposures=steps*cfg['batch'],
            supervised_corner_exposures=corner_exposures, state_counts=dict(Counter(states[:steps].reshape(-1).tolist())),
            train_frames=len(training), real_samples_used=0, calibration_samples_used=0,
            all_zero_loss_batches=0, no_checkpoint_selection=True, new_CNN_forwards=0,
            elapsed_seconds=elapsed, mean_step_seconds=elapsed/steps,
            peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        C.write(cell/'COMPLETION.json', result)
        return result
    except Exception as exc:
        C.write(cell/'FAILURE.json', dict(complete=False, stage=stage, arm=arm, seed=seed,
            successful_optimizer_steps=successful_steps, exception_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), bindings=bindings, source_or_old_cache_modified=False))
        raise
    finally:
        del model, optimizer
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def train_arm(run, arm, seed=1, device='cuda:0'):
    return _train(run, arm, seed, device)


def train_smoke(run, arm, seed=1, device='cuda:0', steps=4):
    return _train(run, arm, seed, device, smoke_steps=steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--arm', choices=M.ARMS, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-steps', type=int, default=4)
    args = parser.parse_args()
    if not args.smoke and args.smoke_steps != 4:
        parser.error('--smoke-steps cannot override the registered main budget')
    result = train_smoke(args.run_dir, args.arm, args.seed, args.device, args.smoke_steps) if args.smoke else train_arm(args.run_dir, args.arm, args.seed, args.device)
    print(json.dumps(result, ensure_ascii=False))
