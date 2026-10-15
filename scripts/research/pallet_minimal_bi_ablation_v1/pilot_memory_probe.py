"""Observe the same fixed CAL6 only, measuring CUDA peak memory and identity."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from . import runtime as RT
from .batch import Budget, BRANCHES, components, _input, check_binding, read, write
from .instrumentation import instrument


def run(root, input_dir, folder, *, device='cuda'):
    root, folder = Path(root), Path(folder)
    if not device.startswith('cuda'):
        raise ValueError('Actual CUDA memory probe required')
    target = folder.parent / 'PILOT_MEMORY_PROBE.json'
    plan = folder.parent / 'PILOT_MEMORY_PROBE_PLAN.json'
    failure = folder.parent / 'PILOT_MEMORY_PROBE_FAILURE.txt'
    if any(p.exists() for p in (target, plan, failure)):
        raise FileExistsError('Preserve every pilot memory probe attempt')
    protocol = check_binding(root, folder)
    data, _ = _input(input_dir)
    panel = [(i, f) for i, f in enumerate(data['frames']) if f['id'] in protocol['pilot_ids']]
    if len(panel) != 6 or {f['id'] for _, f in panel} != set(protocol['pilot_ids']):
        raise ValueError('Same six fixed CAL IDs required')
    budget = Budget(folder.parent)
    budget.check('gpu_process')
    write(plan, dict(schema='MINIMAL_BI_PILOT_MEMORY_PROBE_PLAN_V1',
        protocol_sha256=RT.sha256(folder / 'PROTOCOL.json'),
        source_sha256=RT.sha256(Path(__file__)), pilot_ids=protocol['pilot_ids'],
        branches=list(BRANCHES), observations=12, pose_solves=0,
        reference_or_GT_inputs=False, parameter_or_seed_changes=False,
        branch_states_and_measurements_required_exact=True, masks_required_bitwise=True,
        CUDA_sync_per_observation=True, warmup_or_timing_claim=False,
        memory_scope='per-process PyTorch allocated/reserved, includes resident models and lazy first-call loading'))
    start = time.monotonic()
    status = 'technical_incomplete'
    records = []
    try:
        model = components(root, device)
        for index, frame in panel:
            image = cv2.imread(str(root / frame['image']))
            if image is None or list(image.shape[:2]) != frame['raw_hw']:
                raise ValueError('Same actual RGB canvas required')
            if RT.sha256(root / frame['image']) != frame['image_sha256']:
                raise ValueError('Source RGB changed')
            for branch in BRANCHES:
                expected = read(folder / 'observations' / f'{index:03d}_{branch}.json')['result']
                source = frame['arms'][branch]
                if expected['status'] != 'observed':
                    raise ValueError('Same pilot observation must already be available')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated_before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                tick = time.perf_counter()
                with instrument(model) as meter:
                    result = RT.observe(image, source['native_points'], frame['K'], source['coarse_pose'],
                        components=model, prediction_source=branch, record_id=frame['id'])
                    actual_calls = meter.snapshot()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                observation = result['observation']
                if observation['states'] != expected['observation']['states']:
                    raise AssertionError('Pilot state mismatch')
                if observation['frozen_visible_measurements'] != expected['observation']['frozen_visible_measurements']:
                    raise AssertionError('Pilot B/I measurement/covariance identity mismatch')
                masks_path = folder / expected['mask_binding']['path']
                if RT.sha256(masks_path) != expected['mask_binding']['sha256']:
                    raise ValueError('Original pilot mask binding changed')
                with np.load(masks_path, allow_pickle=False) as archive:
                    mask_identity = {key: bool(np.array_equal(archive[key], result['masks'][key]))
                                     for key in ('pallet_mask', 'state_pallet_mask')}
                if not all(mask_identity.values()):
                    raise AssertionError('Pilot masks are not bitwise identical')
                records.append(dict(id=frame['id'], branch=branch, elapsed_seconds=elapsed,
                    states_exact=True, measurements_and_covariances_exact=True,
                    mask_bitwise_identity=mask_identity, actual_calls=actual_calls,
                    allocated_before_bytes=allocated_before, reserved_before_bytes=reserved_before,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    allocated_after_bytes=torch.cuda.memory_allocated(),
                    reserved_after_bytes=torch.cuda.memory_reserved()))
                print('PILOT_MEMORY', len(records), '/12', branch, frame['id'],
                    records[-1]['peak_allocated_bytes'], records[-1]['peak_reserved_bytes'], flush=True)
                budget.check('gpu_process', time.monotonic() - start)
        status = 'complete'
        write(target, dict(schema='MINIMAL_BI_PILOT_MEMORY_PROBE_V1', status=status,
            plan_sha256=RT.sha256(plan), protocol_sha256=RT.sha256(folder / 'PROTOCOL.json'),
            source_sha256=RT.sha256(Path(__file__)), pilot_ids=protocol['pilot_ids'],
            observations=12, pose_solves=0, records=records,
            GPU_process_elapsed_seconds=time.monotonic() - start,
            max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
            max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records),
            reference_or_GT_inputs=False, new_NN_models_or_weights=False,
            observation_model_forwards_are_actual=True, accuracy_claim=False))
        return target
    except Exception:
        failure.write_text(traceback.format_exc())
        raise
    finally:
        budget.record('CAL_pilot_memory_probe', 'gpu_process', time.monotonic() - start,
                      status=status, new_observations=len(records), pose_solves=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.root, args.input_dir, args.output, device=args.device)
