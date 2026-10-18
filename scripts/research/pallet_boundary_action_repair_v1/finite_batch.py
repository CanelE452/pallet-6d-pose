"""Frozen-input resumable B finite-support batch; no reference decoding."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import canonical, read, rows, save_rows, sha, write, PredictionFirewall
from . import finite_solver as S
from . import finite_boundary as F


def original_frame(original_batch, index, frame):
    folder = Path(original_batch)/'frames'/f'{index:04d}'
    seal = read(folder/'FRAME_SEAL.json')
    if seal['id'] != frame['id'] or seal['result_sha256'] != sha(folder/'RESULT.jsonl.gz'):
        raise ValueError('Original per-frame immutable binding mismatch')
    if seal['input_row_sha256'] != hashlib.sha256(canonical(frame).encode()).hexdigest():
        raise ValueError('Original exact predictor input row mismatch')
    result = next(rows(folder/'RESULT.jsonl.gz'))
    bundle_path = folder/'BOUNDARY_BUNDLE.jsonl.gz'
    bundle = next(rows(bundle_path)) if bundle_path.exists() else None
    if bundle is not None and result['boundary']['bundle_sha256'] != sha(bundle_path):
        raise ValueError('Original fixed RGB profile bundle bytes mismatch')
    return result, bundle, seal


def job(index, frame, config, original_batch, regularization, output):
    import cv2
    cv2.setNumThreads(1)
    before, cpu = time.monotonic(), time.process_time(); folder = Path(output)
    original, bundle, original_seal = original_frame(original_batch, index, frame)
    result = S.generate(frame, config, original, bundle, regularization)
    result['resource'] = dict(elapsed_seconds=time.monotonic()-before, process_cpu_seconds=time.process_time()-cpu,
        new_optimizer_attempts=result['actual_new_solve_attempts'], new_backbone_forwards=0,
        restoration_iterations=sum(a['solve'].get('restoration_iterations', 0) for a in result['attempts']),
        slsqp_iterations=sum(a['solve'].get('iterations', 0) for a in result['attempts']))
    save_rows(folder/'RESULT.jsonl.gz', [result])
    write(folder/'FRAME_SEAL.json', dict(id=frame['id'], complete=True, before_scoring=True, reference_inputs=False,
        result_sha256=sha(folder/'RESULT.jsonl.gz'), input_row_sha256=hashlib.sha256(canonical(frame).encode()).hexdigest(),
        original_frame_seal_sha256=sha(Path(original_batch)/'frames'/f'{index:04d}'/'FRAME_SEAL.json'),
        fixed_universe_sha256=hashlib.sha256(canonical(F.universe(bundle)).encode()).hexdigest(),
        resource=result['resource']))
    return dict(id=frame['id'], path=str(folder/'RESULT.jsonl.gz'), **result['resource'])


def run(inputs, old_config, original_batch, regularization_path, output, *, workers=1, limit=None, cal_only=False):
    started = time.monotonic(); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    if not 1 <= workers <= 4:
        raise ValueError('Original campaign maximum four CPU workers')
    config = read(old_config); reg = read(regularization_path)
    if not reg.get('complete') or reg.get('real_fitted') or reg.get('validation_fitted'):
        raise ValueError('Frozen train-only local information receipt required')
    guard = PredictionFirewall(config.get('forbidden_prediction_paths', [])).install()
    original_batch = Path(original_batch)
    base_seal = read(original_batch/'BATCH_SEAL.json')
    if not base_seal['complete'] or base_seal['input_sha256'] != sha(inputs):
        raise ValueError('Exact original B238 inputs/seal required')
    all_frames = list(rows(inputs))
    frames = [(i, f) for i, f in enumerate(all_frames) if not cal_only or f['partition'] == 'CAL']
    if limit is not None:
        frames = frames[:limit]
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('finite_*.py')}
    lock = dict(schema='FINITE_B_BATCH_LOCK_V1', original_inputs_sha256=sha(inputs),
        original_config_sha256=sha(old_config), original_batch_seal_sha256=sha(original_batch/'BATCH_SEAL.json'),
        local_regularization_sha256=sha(regularization_path), source_sha256=code,
        ordered_ids=[f['id'] for _, f in frames], workers=workers, technical_limit=limit, cal_only=cal_only,
        max_new_solves_per_frame=18, restoration_inside_80_iteration_budget=True, reference_inputs=False)
    if (out/'BATCH_LOCK.json').exists():
        if read(out/'BATCH_LOCK.json') != lock:
            raise ValueError('Frozen batch changed; preserve and assign new technical attempt')
    else:
        write(out/'BATCH_LOCK.json', lock)
    if (out/'BATCH_SEAL.json').exists():
        seal = read(out/'BATCH_SEAL.json')
        if seal['predictions_sha256'] != sha(out/'PREDICTIONS.jsonl.gz') or seal['lock_sha256'] != sha(out/'BATCH_LOCK.json'):
            raise ValueError('Completed finite batch changed')
        return seal
    complete = {}; pending = []; failures = []
    for index, frame in frames:
        folder = out/'frames'/f'{index:04d}'; folder.mkdir(parents=True, exist_ok=True)
        if (folder/'FRAME_SEAL.json').exists():
            seal = read(folder/'FRAME_SEAL.json')
            if seal['id'] != frame['id'] or seal['result_sha256'] != sha(folder/'RESULT.jsonl.gz'):
                raise ValueError('Completed frame changed')
            complete[frame['id']] = dict(id=frame['id'], path=str(folder/'RESULT.jsonl.gz'), reused=True, **seal['resource'])
        else:
            pending.append((index, frame, folder))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, i, f, config, str(original_batch), reg, str(folder)): (i, f, folder) for i, f, folder in pending}
        for future in as_completed(futures):
            index, frame, folder = futures[future]
            try:
                complete[frame['id']] = future.result()
            except BaseException as exc:
                fail = dict(id=frame['id'], status='FAILED_TECHNICAL', error=repr(exc), epoch=time.time())
                failures.append(fail); write(folder/'FAILURE.json', fail, replace=True)
            write(out/'BATCH_PROGRESS.json', dict(complete_ids=list(complete), failures=failures,
                total=len(frames), elapsed_seconds=time.monotonic()-started), replace=True)
            print(canonical(dict(complete=len(complete), failures=len(failures), total=len(frames), elapsed_seconds=time.monotonic()-started)), flush=True)
    if failures:
        raise RuntimeError('Frame contract failures preserved; not disguised as N0 fallback: '+canonical(failures))
    values = [next(rows(complete[f['id']]['path'])) for _, f in frames]
    save_rows(out/'PREDICTIONS.jsonl.gz', [dict(id=v['id'], partition=v['partition'], session=v['session'], cluster_id=v['cluster_id'], predictions=v['predictions']) for v in values])
    seal = dict(schema='FINITE_B_BATCH_SEAL_V1', complete=True, rows=len(values), arms=list(S.ARMS),
        ordered_ids=[v['id'] for v in values], lock_sha256=sha(out/'BATCH_LOCK.json'),
        predictions_sha256=sha(out/'PREDICTIONS.jsonl.gz'), all_frames_kept=True, failures=[],
        actual_new_optimizer_attempts=sum(v['actual_new_solve_attempts'] for v in values),
        restoration_iterations=sum(v['resource']['restoration_iterations'] for v in values),
        slsqp_iterations=sum(v['resource']['slsqp_iterations'] for v in values),
        elapsed_seconds=time.monotonic()-started, sum_frame_cpu_seconds=sum(v['resource']['process_cpu_seconds'] for v in values),
        workers=workers, new_RGB_queries=0, new_model_forwards=0, before_scoring=True, reference_inputs=False,
        firewall=guard.receipt())
    write(out/'BATCH_SEAL.json', seal); return seal


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('calibrate'); c.add_argument('--inputs', required=True); c.add_argument('--old-config', required=True)
    c.add_argument('--original-batch', required=True); c.add_argument('--output', required=True)
    r = sub.add_parser('run'); r.add_argument('--inputs', required=True); r.add_argument('--old-config', required=True)
    r.add_argument('--original-batch', required=True); r.add_argument('--regularization', required=True)
    r.add_argument('--output', required=True); r.add_argument('--workers', type=int, default=1); r.add_argument('--limit', type=int); r.add_argument('--cal-only', action='store_true')
    a = p.parse_args()
    if a.command == 'calibrate':
        answer = S.calibrate_regularization(a.inputs, a.old_config, a.original_batch, a.output)
    else:
        answer = run(a.inputs, a.old_config, a.original_batch, a.regularization, a.output, workers=a.workers, limit=a.limit, cal_only=a.cal_only)
    print(canonical(answer))

if __name__ == '__main__':
    main()
