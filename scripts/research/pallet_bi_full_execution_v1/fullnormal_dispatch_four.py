"""Four-worker orchestration of the unchanged reviewed full-normal rows.

The new schema/SHA records scheduling truthfully. D3, its validators and every
numerical/feature primitive remain immutable. Production requires a fresh root
GO and output namespace; this module does not quiesce any other controller.
"""
from __future__ import annotations
import os
for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path
import time
import traceback

from . import fullnormal_dispatch as D3

ROW = D3.ROW
K, CAL, Q = D3.K, D3.CAL, D3.Q
D3_EXPECTED_SHA256 = '0691ab08e935537d5f0a47aa110c7d0a76a220c0634ca7f5ef233bc646d38c54'
SCHEMA = 'FULL_NORMAL_FOUR_CPU_DISPATCH_V1'
WORKERS = 4
initialize_worker = ROW.initialize_worker
row_job = ROW.row_job
aggregate = ROW.aggregate
verify_packet_bindings = ROW.verify_packet_bindings
verify_reuse_root = D3.verify_reuse_root


def source_bindings():
    """Retain the exact D3 validator and numerical closure, add only D4 SHA."""
    previous = D3.source_bindings()
    if previous['orchestration_sha256'] != D3_EXPECTED_SHA256:
        raise ValueError('Reviewed immutable D3 validation/orchestration source changed')
    return dict(orchestration_sha256=Q.sha(__file__),
        preserved_D3_validation_sha256=previous['orchestration_sha256'],
        row_helper_sha256=previous['row_helper_sha256'],
        numerical_kernel_source_binding=previous['numerical_kernel_source_binding'])


def execute_static(jobs, output, workers, initializer, initargs, callable_row, stop_file=None):
    """Same bounded FIFO/drain contract as D3; exactly four active rows max."""
    if workers != WORKERS:
        raise ValueError('This explicit four-CPU orchestration requires workers=4')
    output = Path(output)
    stop_file = Path(stop_file) if stop_file else None
    results, failures, next_index, started = [], [], 0, time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=initializer, initargs=initargs) as pool:
        pending = {}
        def enqueue():
            nonlocal next_index
            while next_index < len(jobs) and len(pending) < workers and not (stop_file and stop_file.exists()):
                job = jobs[next_index]
                next_index += 1
                pending[pool.submit(callable_row, job)] = job
        enqueue()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                domain, ordinal, frame, *_ = pending.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    Q.write(output / 'row_jobs' / domain / f'{ordinal:04d}.json', result)
                except Exception as error:
                    record = dict(domain=domain, ordinal=ordinal, id=frame['id'],
                        status='INCOMPLETE_TECHNICAL', error=repr(error), traceback=traceback.format_exc())
                    failures.append(record)
                    Q.write(output / 'row_jobs' / domain / f'{ordinal:04d}_FAILURE.json', record)
                print(dict(stage='FULL_NORMAL_FOUR_CPU_DISPATCH', completed_jobs=len(results),
                    technical_failures=len(failures), total_jobs=len(jobs), submitted_jobs=next_index,
                    workers=workers, wall_seconds=time.monotonic()-started), flush=True)
            enqueue()
    return results, failures, dict(submitted_jobs=next_index, unsubmitted_jobs=len(jobs)-next_index,
        checkpoint_stop_observed=bool(stop_file and stop_file.exists()), active_rows_drained_before_exit=True)


def run(prior_run, packet_root, packet_seal, environment, output, *, arms=('C0', 'CP', 'CB'),
        workers=WORKERS, reuse_root=None, root_GO=False, stop_file=None):
    if root_GO is not True:
        raise PermissionError('Fresh root four-CPU measured-headroom GO required')
    if workers != WORKERS:
        raise ValueError('This explicit four-CPU orchestration requires workers=4')
    prior_run, packet_root, packet_seal, environment, output = map(
        Path, (prior_run, packet_root, packet_seal, environment, output))
    if output.exists():
        raise FileExistsError('Four-CPU execution requires a new output namespace; preserve all previous roots')
    output.mkdir(parents=True, exist_ok=False)
    started, binding = time.monotonic(), source_bindings()
    frames = {domain: CAL.load_bound_frames(prior_run, domain)[0] for domain in ('real', 'synthetic')}
    if {d: len(ff) for d, ff in frames.items()} != {'real': 238, 'synthetic': 1003}:
        raise ValueError('Full original1241 required')
    expected, validation = verify_packet_bindings(packet_root, packet_seal, frames)
    validation.update(orchestration_sha256=binding['orchestration_sha256'],
        preserved_D3_validation_sha256=binding['preserved_D3_validation_sha256'],
        row_helper_sha256=binding['row_helper_sha256'])
    Q.write(output / 'PACKET_VALIDATION.json', validation)
    reuse = verify_reuse_root(reuse_root, packet_seal, binding['numerical_kernel_source_binding'])
    p0 = Q.read(prior_run / 'P0_INPUT_CONTRACT.json')
    jobs = [(domain, i, f, p0[domain]['inventory'][i]['observation_folder'], expected[(domain, i)])
            for domain, ff in frames.items() for i, f in enumerate(ff)]
    lock = dict(schema=SCHEMA, workers=workers, arms=list(arms),
        driver_sha256=binding['row_helper_sha256'], row_helper_sha256=binding['row_helper_sha256'],
        orchestration_sha256=binding['orchestration_sha256'],
        preserved_D3_validation_sha256=binding['preserved_D3_validation_sha256'],
        driver_field_scope='immutable row helper implementation; distinct four-CPU scheduling separately bound',
        numerical_kernel_source_binding=binding['numerical_kernel_source_binding'],
        numerical_kernel_unchanged_from_pilot=True, packet_root=str(packet_root), packet_seal=str(packet_seal),
        packet_seal_sha256=Q.sha(packet_seal),
        packet_validation_receipt_sha256=Q.sha(output / 'PACKET_VALIDATION.json'),
        environment_path=str(environment), environment_sha256=Q.sha(environment),
        prior_P0_sha256=Q.sha(prior_run / 'P0_INPUT_CONTRACT.json'),
        driver_validation_source_sha256={str(Path(ROW.action_data.__file__).resolve()): Q.sha(ROW.action_data.__file__)},
        exact_reuse_binding=reuse, static_ordered_row_ids={d: [f['id'] for f in ff] for d, ff in frames.items()},
        stop_file=str(stop_file) if stop_file else None, dispatch_window=workers,
        resource_policy=dict(CPU_workers=workers, BLAS_threads_per_worker=1, observation_or_backbone_training=False),
        new_driver_only_numerical_and_feature_definitions_unchanged=True)
    Q.write(output / 'PRODUCER_LOCK.json', lock)
    results, failures, dispatch = execute_static(jobs, output, workers, initialize_worker,
        (str(prior_run), str(packet_root), str(output), list(arms), str(reuse_root) if reuse_root else None),
        row_job, stop_file)
    summary = aggregate(output, frames, arms, lock, results, failures, started)
    Q.write(output / 'DISPATCH_COMPLETION.json', dict(dispatch,
        orchestration_sha256=binding['orchestration_sha256'], row_helper_sha256=binding['row_helper_sha256'],
        preserved_D3_validation_sha256=binding['preserved_D3_validation_sha256'],
        summary_sha256=Q.sha(output / 'SUMMARY.json')))
    return summary


def main():
    parser = argparse.ArgumentParser()
    for key in ('prior-run', 'packet-root', 'packet-seal', 'environment', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--arms', nargs='+', default=['C0', 'CP', 'CB'])
    parser.add_argument('--reuse-root')
    parser.add_argument('--stop-file')
    parser.add_argument('--root-go', action='store_true')
    a = parser.parse_args()
    summary = run(a.prior_run, a.packet_root, a.packet_seal, a.environment, a.output,
        arms=a.arms, workers=a.workers, reuse_root=a.reuse_root, stop_file=a.stop_file, root_GO=a.root_go)
    print({k: v for k, v in summary.items() if k not in ('missing_rows', 'technical_failures', 'full_feature_manifests')})


if __name__ == '__main__':
    main()
