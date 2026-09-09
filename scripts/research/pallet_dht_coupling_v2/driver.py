"""Owned train→actual evaluation→statistics→report→verified delivery chain."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.research.pallet_dht_coupling_v2.prepare import ROOT, OLD, read, write, sha, check
from scripts.research.pallet_dht_joint_v1.driver import now

HERE = Path(__file__).resolve().parent


def verify_bindings(value, base):
    for category in ('source_sha256', 'input_sha256', 'output_sha256', 'source_results'):
        for filename, digest in value.get(category, {}).items():
            path = Path(filename)
            path = path if path.is_absolute() else base / path
            check(sha(path) == digest, f'Changed bound artifact: {path}')


def audit_training(root, protocol):
    import importlib
    import torch
    import scripts.research.pallet_dht_coupling_v2.integration  # checkpoint classes
    from scripts.research.pallet_dht_joint_v1.driver import HERE as original_here
    if str(original_here) not in sys.path:
        sys.path.insert(0, str(original_here))
    importlib.import_module('integration')  # historical checkpoint pickle class
    runs, traces, final_models = [], {}, []
    for seed in protocol['training']['seeds']:
        for arm in protocol['arms']:
            folder = root / 'runs' / f'{arm}_seed{seed}'
            completed = read(folder / 'COMPLETION.json')
            check(completed['complete'] and completed['PASS'] and completed['stage'] == 'main'
                  and completed['epochs_completed'] == protocol['training']['epochs']
                  and completed['optimizer_steps'] == protocol['expected_optimizer_steps_per_cell']
                  and completed['train_frames'] == 55980 and completed['val_frames'] == 4020
                  and completed['checkpoint_sha256'] == sha(completed['checkpoint'])
                  and completed['bindings']['protocol_sha256'] == sha(root / 'TRAIN_PROTOCOL.json'),
                  f'Unfinished or changed training cell: {arm}/{seed}')
            trace = sha(folder / 'BATCH_TRACE.jsonl')
            check(trace == completed['batch_trace_sha256'], 'Training trace changed')
            traces[arm, seed] = trace
            for group in ('backbone_neck', 'pose_head', 'hough'):
                check(completed['final_parameter_BN_changes'][group]['changed_values'] > 0,
                      f'No actual learning in {arm}/{seed}/{group}')
            for group in ('backbone_neck', 'pose_head'):
                check(completed['final_parameter_BN_changes'][group]['bn_changed'] > 0,
                      f'Frozen BN in {arm}/{seed}/{group}')
            checkpoint = torch.load(completed['checkpoint'], map_location='cpu')
            state = checkpoint['ema'].state_dict()
            digest = hashlib.sha256()
            for key, tensor in sorted(state.items()):
                digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            ema_sha = digest.hexdigest()
            final_models.append(ema_sha)
            del checkpoint, state
            runs.append(dict(arm=arm, seed=seed, checkpoint=completed['checkpoint'],
                checkpoint_sha256=completed['checkpoint_sha256'], ema_tensor_sha256=ema_sha,
                completion_sha256=sha(folder / 'COMPLETION.json'), trace_sha256=trace))
        check(len({traces[arm, seed] for arm in protocol['arms']}) == 1,
              f'Actual augmented inputs differ across arms: seed{seed}')
    for arm in protocol['arms']:
        check(len({traces[arm, seed] for seed in protocol['training']['seeds']}) == len(protocol['training']['seeds']),
              f'Seeds do not have distinct data schedules: {arm}')
    destination = root / 'TRAINING_AUDIT.json'
    existing = read(destination) if destination.exists() else None
    original = dict(complete=True, PASS=True, created_at_utc=existing['created_at_utc'] if existing else now(), runs=runs,
        same_seed_augmented_batches_exact=True, seeds_have_distinct_actual_batches=True,
        all_network_groups_and_BN_changed=True, protocol_sha256=sha(root / 'TRAIN_PROTOCOL.json'),
        distinct_actual_EMA_states=len(set(final_models)),
        identical_final_states_policy='Allowed for a null intervention; actual training and independent inference still required.',
        evaluation_weights='Final epoch EMA; no real or best-epoch selection')
    if existing is None:
        write(destination, original)
    else:
        check(existing == original, 'Completed training audit changed')
    references = read(root / 'REUSED_CONTROLS.json')
    verify_bindings(references, root)
    controls = {(row['arm'], row['seed']): row for row in references['controls']}
    rows = []
    for row in original['runs']:
        arm, seed = row['arm'], row['seed']
        reference = controls[('hough_joint', seed)]
        expected_trace = reference['artifacts']['batch_trace']['sha256']
        check(row['trace_sha256'] == expected_trace, f'Augmented data differs from original control: {arm}/{seed}')
        new_initial = root / 'runs' / f'{arm}_seed{seed}' / 'INITIAL_STATE.pt'
        old_initial = Path(reference['artifacts']['initial_state']['path'])
        a, b = torch.load(new_initial, map_location='cpu'), torch.load(old_initial, map_location='cpu')
        check(a.keys() == b.keys(), 'Initial model state keys differ from common joint control')
        check(all(torch.equal(a[key], b[key]) for key in a), 'Initial model tensor values differ from common joint control')
        rows.append(dict(arm=arm, seed=seed, same_initial_tensors=True,
                         actual_full_augmented_trace_exact=True, trace_sha256=expected_trace,
                         checkpoint_sha256=row['checkpoint_sha256'], initial_state_sha256=sha(new_initial)))
        del a, b
    result = dict(complete=True, PASS=True, n_new_cells=12, n_reused_control_cells=9, runs=rows,
                  reuse_manifest_sha256=sha(root / 'REUSED_CONTROLS.json'),
                  training_audit_sha256=sha(root / 'TRAINING_AUDIT.json'),
                  source_sha256={str(Path(__file__)): sha(__file__)})
    destination = root / 'MATCHED_CONTROL_AUDIT.json'
    if destination.exists():
        check(read(destination) == result, 'Completed matching audit changed')
    else:
        write(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = read(root / 'TRAIN_PROTOCOL.json')
    training = protocol['training']
    handle = (root / '.driver.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    preflight = {}
    for name in protocol['preflight_required']:
        value = read(root / name)
        check(value.get('complete') is True and value.get('PASS') is True, f'Preflight incomplete: {name}')
        verify_bindings(value, root)
        preflight[name] = sha(root / name)
    binding = dict(protocol_sha256=sha(root / 'TRAIN_PROTOCOL.json'), driver_sha256=sha(__file__),
                   training_source_sha256=protocol['source_code_sha256'], preflight_sha256=preflight)
    if (root / 'DRIVER_BINDING.json').exists():
        check(read(root / 'DRIVER_BINDING.json') == binding, 'Frozen main driver binding changed')
    else:
        write(root / 'DRIVER_BINDING.json', binding)

    def stage(label, module, extra=()):
        check(sha(root / 'TRAIN_PROTOCOL.json') == binding['protocol_sha256'], 'Training protocol changed')
        for file, digest in binding['training_source_sha256'].items():
            check(sha(file) == digest, f'Training source changed: {file}')
        command = [sys.executable, '-m', f'scripts.research.pallet_dht_coupling_v2.{module}', '--run-dir', str(root), *extra]
        logfile = root / 'logs' / f'{label}.log'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        print(f'{now()} START {label}', flush=True)
        with logfile.open('a') as stream:
            child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
            write(root / 'STATUS.json', dict(complete=False, stage=label, started_at_utc=now(),
                  command=command, driver_pid=os.getpid(), child_pid=child.pid, log=str(logfile)))
            try:
                code = child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        check(code == 0, f'{label} exited {code}; inspect {logfile}')
        print(f'{now()} DONE {label} ({time.time()-start:.1f}s)', flush=True)

    try:
        for seed in training['seeds']:
            for arm in protocol['arms']:
                extra = ['--arm', arm, '--seed', str(seed), '--device', '0']
                for name in ('epochs', 'batch', 'lr', 'optimizer', 'line_weight', 'incidence_weight', 'workers',
                             'lrf', 'warmup_epochs', 'warmup_bias_lr', 'amp_init_scale'):
                    extra.extend(['--' + name.replace('_', '-'), str(training[name])])
                folder = root / 'runs' / f'{arm}_seed{seed}'
                if (folder / 'CELL_CONFIG.json').exists() and not (folder / 'COMPLETION.json').exists():
                    if (folder / 'weights/last.pt').exists():
                        extra.append('--resume')
                    else:
                        archive = root / 'provenance/interrupted_before_first_epoch' / f'{folder.name}_{time.time_ns()}'
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        folder.rename(archive)
                stage(f'train_{arm}_seed{seed}', 'train', extra)
        audit_training(root, protocol)
        # Completion/reporting sources are frozen before the first new accuracy
        # prediction; they can be implemented during the long training phase.
        chain = read(root / 'COMPLETION_CHAIN_BINDING.json')
        check(chain.get('complete') is True and chain.get('PASS') is True, 'Completion chain not frozen')
        verify_bindings(chain, root)
        for seed in training['seeds']:
            for arm in protocol['arms']:
                checkpoint = root / 'runs' / f'{arm}_seed{seed}' / 'weights/final.pt'
                stage(f'evaluate_{arm}_seed{seed}', 'evaluate', ['--arm', arm, '--seed', str(seed),
                      '--checkpoint', str(checkpoint), '--device', '0', '--phase', 'all'])
        for module in ('runtime', 'aggregate', 'report', 'audit_outputs', 'visual_qa'):
            stage(module, module)
        write(root / 'STATUS.json', dict(complete=False, stage='awaiting_actual_screenshot_review',
              driver_pid=os.getpid(), started_at_utc=now(),
              action='Root agent inspect actual screenshots and bind VISUAL_REVIEW.json; no user permission requested'))
        while not (root / 'VISUAL_REVIEW.json').exists():
            time.sleep(5)
        review = read(root / 'VISUAL_REVIEW.json')
        check(review.get('complete') is True and review.get('PASS') is True, 'Actual screenshot review incomplete')
        verify_bindings(review, root)
        stage('finalize', 'finalize')
        done = read(root / 'COMPLETION.json')
        check(done['complete'] is True and done['PASS'] is True, 'Incomplete final artifact')
        write(root / 'STATUS.json', dict(complete=True, stage='complete', finished_at_utc=now(),
              completion_sha256=sha(root / 'COMPLETION.json')))
    except BaseException as exc:
        failure = dict(complete=False, PASS=False, time_utc=now(), type=type(exc).__name__, message=str(exc),
                       last_status=read(root / 'STATUS.json') if (root / 'STATUS.json').exists() else None)
        write(root / 'DRIVER_FAILURE.json', failure)
        raise


if __name__ == '__main__':
    main()
