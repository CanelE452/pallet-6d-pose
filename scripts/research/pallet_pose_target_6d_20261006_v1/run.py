"""Run the fixed 6D-target experiment and its synthetic-only seed gate.

All numerical outputs remain in this namespace or the external cost cache.
Publication to main is a separate reviewed Git operation after verification.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from .preflight import ROOT, DOC, sha, read, write

PREFIX = 'scripts.research.pallet_pose_target_6d_20261006_v1.'


class FinishLockedError(RuntimeError):
    """Another finish owns the cache; preserve its status and ledger."""


def producer_identity(pid):
    """Read Linux start ticks so a reused PID cannot impersonate the producer."""
    if pid is None:
        return None
    try:
        fields = Path('/proc').joinpath(str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])


def producer_alive(expected):
    current = producer_identity(expected['pid']) if expected else None
    return bool(current and current['start_ticks'] == expected['start_ticks']
                and current['state'] not in ('Z', 'X', 'x'))


def no_unresolved_attempts(ledger):
    unresolved = [index + 1 for index, entry in enumerate(ledger['stages']) if entry['state'] == 'ATTEMPT']
    if unresolved:
        raise RuntimeError('Unresolved stage ATTEMPT(s); ledger preserved, no duplicate stage: ' + str(unresolved))


def completed_cost(manifest):
    if manifest['status'] != 'PASS':
        return False
    assert manifest['completed_rows'] == 55915 and manifest['candidate_F_completed'] == 11238915
    assert manifest['incomplete_attempted_F'] == 0
    return True


def run_stage(module, arguments, cache):
    ledger_path = DOC / 'PIPELINE_EXECUTIONS.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(schema='pose_target_pipeline_execution_v1', stages=[])
    no_unresolved_attempts(ledger)
    entry = dict(module=module, command=[sys.executable, '-B', '-u', '-m', PREFIX + module, *map(str, arguments)],
                 state='ATTEMPT', start_unix=time.time(), code_sha256=sha(Path(__file__).parent / (module + '.py')))
    ledger['stages'].append(entry)
    write(ledger_path, ledger)
    log = cache / ('pipeline_' + str(len(ledger['stages'])).zfill(2) + '_' + module + '.log')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with log.open('w') as stream:
        result = subprocess.run(entry['command'], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    entry.update(state='COMPLETE' if result.returncode == 0 else 'FAILED', exit_code=result.returncode,
                 end_unix=time.time(), seconds=time.time() - entry['start_unix'], log=str(log), log_sha256=sha(log))
    write(ledger_path, ledger)
    print('PIPELINE_STAGE', module, result.returncode, round(entry['seconds'], 3), flush=True)
    if result.returncode:
        raise RuntimeError('Stage failed; preserve completed attempts: ' + str(log))


def execution_counts(cache, seeds):
    cost = read(DOC / 'POSE_COST_CACHE_MANIFEST.json')
    parity = read(DOC / 'POSE_TARGET_PARITY.json')
    fits = [read(DOC / f'fits/POSE_TARGET_GEO_seed{s}.json') for s in seeds]
    evaluations = [read(DOC / f'results/{split}_POSE_TARGET_GEO_J_seed{s}.json')
                   for s in seeds for split in ('SYNTH_HELDOUT', 'REAL_DEV')]
    receipts = dict(schema='pose_target_formal_training_receipts_v1', actual_fits=len(fits),
                    optimizer_updates=sum(f['updates'] for f in fits), exposures=sum(f['exposures'] for f in fits),
                    excluded_target_exposures=sum(f['excluded_target_exposures'] for f in fits),
                    seeds=seeds, fits=fits, checkpoint_selection='last6000 only', hyperparameter_search=0,
                    standalone_smoke_fits=0, standalone_smoke_optimizer_updates=0)
    write(DOC / 'TRAIN_RECEIPTS.json', receipts)
    aux = read(DOC / 'TRAIN_2D_6D_TARGET_COMPARISON.json')
    first_timing = read(DOC / 'COST_TIMING.json')
    result = dict(schema='pose_target_actual_execution_counts_v1', pose_cost_F_attempted=cost['candidate_F_attempted'],
                  pose_cost_F_completed=cost['candidate_F_completed'], pose_cost_F_failures=cost['candidate_F_failures'],
                  parity_F_attempted=parity['actual_F_calls'], parity_F_completed=parity['completed_F_calls'],
                  evaluation_F_attempted=sum(e['execution']['new_final_F_attempts'] for e in evaluations),
                  evaluation_F_completed=sum(e['execution']['new_final_F_completed'] for e in evaluations),
                  actual_fits=receipts['actual_fits'], optimizer_updates=receipts['optimizer_updates'],
                  training_forward_batches=receipts['optimizer_updates'], training_forward_examples=receipts['exposures'],
                  excluded_target_exposures=receipts['excluded_target_exposures'],
                  evaluation_forward_batches=sum(e['execution']['new_refiner_batches'] for e in evaluations),
                  evaluation_forward_examples=sum(e['execution']['new_refiner_examples'] for e in evaluations),
                  PnP_counts={k: cost['PnP_counts'][k] + parity['PnP_counts'][k] +
                              sum(e['execution']['PnP_counts'][k] for e in evaluations)
                              for k in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')},
                  CPU_pose_cost_wall_seconds=first_timing['seconds_wall'] + cost['seconds_wall'],
                  CPU_pose_cost_worker_seconds_sum=cost['chunk_seconds_sum'],
                  CPU_parity_wall_seconds=parity['seconds_wall'],
                  TRAIN_auxiliary_wall_seconds=aux['seconds'],
                  fit_wall_seconds=sum(f['seconds'] for f in fits),
                  evaluation_wall_seconds=sum(e['execution']['seconds_wall'] for e in evaluations),
                  timing_scope='Stage wall times; fit/evaluation include GPU work plus CPU/I/O, not isolated CUDA kernel time',
                  bank_generation=0, backbone_detector_forwards=0, real_training=0, PERM_fits=0,
                  auxiliary_refiner_forwards=0, auxiliary_F=0, hyperparameter_search=0,
                  cache_reuse='first16 TRAIN3216 completed F reused inside full cost; existing bank/features/orders/checkpoints/old metrics read-only',
                  pipeline_logs=str(cache), manuscript_writes=0)
    result['all_final_F_attempted'] = result['pose_cost_F_attempted'] + result['parity_F_attempted'] + result['evaluation_F_attempted']
    write(DOC / 'EXECUTION_COUNTS.json', result)
    return result


def finish(args):
    cache = args.cost_cache.resolve()
    # Keep this advisory lock for the complete orchestration, including failure
    # recording. A second finish never changes the active run's global files.
    with (cache / 'finish.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FinishLockedError('Another finish holds ' + str(cache / 'finish.lock')) from error
        try:
            ledger_path = DOC / 'PIPELINE_EXECUTIONS.json'
            if ledger_path.exists():
                no_unresolved_attempts(read(ledger_path))
            finish_locked(args, cache)
        except Exception as error:
            previous = read(DOC / 'STATUS.json') if (DOC / 'STATUS.json').exists() else {}
            write(DOC / 'PIPELINE_FAILURE.json', dict(status='FAILED_STAGE', exception=type(error).__name__,
                reason=str(error), actual_F_or_optimizer_attempts_preserved=True,
                unresolved_attempt_ledger_preserved=True, quiet_retry=0,
                previous_status=previous, cost_producer=getattr(args, '_cost_producer_identity', None),
                time_unix=time.time()))
            failed = dict(previous)
            failed.update(schema='pose_target_status_v1', status='FAILED_STAGE', stage=previous.get('stage', 'FINISH_GUARD'),
                reason=str(error), manuscript_changes=0, quiet_retry=0, end_unix=time.time())
            write(DOC / 'STATUS.json', failed)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def finish_locked(args, cache):
    started = time.time()
    producer = producer_identity(args.cost_producer_pid)
    args._cost_producer_identity = producer
    common = ['--source-root', args.source_root, '--bank-cache', args.bank_cache, '--cost-cache', cache]
    write(DOC / 'STATUS.json', dict(schema='pose_target_status_v1', status='RUNNING', stage='COST', publication_branch='main',
        baseline_commit=read(DOC / 'PROTOCOL.json')['baseline_commit'], manuscript_changes=0, start_unix=started,
        cost_producer=producer, cost_producer_pid=args.cost_producer_pid))
    last_report = 0
    while True:
        manifest = read(DOC / 'POSE_COST_CACHE_MANIFEST.json')
        if completed_cost(manifest):
            break
        if args.cost_producer_pid is not None and not producer_alive(producer):
            # The producer may have published its final manifest just before
            # exit. Accept that final PASS once; never restart or retry it.
            if completed_cost(read(DOC / 'POSE_COST_CACHE_MANIFEST.json')):
                break
            raise RuntimeError('Cost producer exited or changed start ticks before complete PASS; no retry: '
                               + str(args.cost_producer_pid))
        if time.time() - last_report >= 45:
            import numpy as np
            rows = np.load(cache / 'train_rows.npy', mmap_mode='r')
            done = np.load(cache / 'done.npy', mmap_mode='r')
            print('PIPELINE_WAIT_COST', int(done[rows].sum()), 55915, flush=True)
            last_report = time.time()
        time.sleep(5)
    assert read(DOC / 'POSE_TARGET_PARITY.json')['status'] == 'PASS'
    assert read(DOC / 'TRAIN_INITIALIZATION_PARITY.json')['status'] == 'PASS'
    run_stage('training', ['--stage', 'aux', *common], cache)
    assert read(DOC / 'TRAIN_2D_6D_TARGET_COMPARISON.json')['status'] == 'PASS'
    run_stage('training', ['--stage', 'fit', '--seed', '1', *common], cache)
    run_stage('evaluation', ['--seed', '1', '--split', 'all', *common], cache)
    run_stage('reporting', ['--stage', 'screen', '--repo-root', ROOT], cache)
    screen = read(DOC / 'SEED1_SCREENING.json')
    seeds = [1]
    if screen['continue_seeds']:
        assert screen['decision'] in ('CONTINUE', 'MIXED_CONTINUE')
        for seed in (2, 3):
            run_stage('training', ['--stage', 'fit', '--seed', str(seed), *common], cache)
            run_stage('evaluation', ['--seed', str(seed), '--split', 'all', *common], cache)
            seeds.append(seed)
    execution_counts(cache, seeds)
    run_stage('contract_tests', ['--artifacts'], cache)
    run_stage('reporting', ['--stage', 'final', '--seeds', *map(str, seeds), '--repo-root', ROOT], cache)
    write(DOC / 'STATUS.json', dict(schema='pose_target_status_v1', status='COMPUTE_AND_REPORT_DONE',
        stage='WAITING_INDEPENDENT_REVIEW_AND_MAIN_PUBLICATION', seeds=seeds, publication_branch='main',
        baseline_commit=read(DOC / 'PROTOCOL.json')['baseline_commit'],
        verdict=read(DOC / 'REPORT_EXECUTION.json')['verdict'], manuscript_changes=0,
        seed1_screening=screen['decision'], independent_physical_reference='not acquired; REAL_DEV same2D reference',
        start_unix=started, end_unix=time.time(), push='PENDING'))
    print('PIPELINE_FINISHED', seeds, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('finish',))
    parser.add_argument('--source-root', type=Path, default=Path('/home/minjae/Documents/github/pallet-pose'))
    parser.add_argument('--bank-cache', type=Path, default=Path('/tmp/pallet-joint-action-cache'))
    parser.add_argument('--cost-cache', type=Path, default=Path('/tmp/pallet-pose-target-6d-cache'))
    parser.add_argument('--cost-producer-pid', type=int,
                        help='Optional live cost producer PID; bind start ticks and fail if it exits before final PASS')
    args = parser.parse_args(argv)
    if args.cost_producer_pid is not None and args.cost_producer_pid <= 0:
        parser.error('--cost-producer-pid must be positive')
    finish(args)


if __name__ == '__main__':
    main()
