"""Actual-worker CPU gate for the four paired training inputs; optimizer0.

All actual RGB/label paths and occurrence details stay in private RAW traces.
The existing full-model true-ignore verifier receives a TRAIN tensor injection;
its historical evaluation-fixture fallback is explicitly disabled.
"""
from __future__ import annotations

from collections import Counter
import inspect
from pathlib import Path
import random
import time
import traceback

import cv2
import numpy as np
import torch

from . import common as C
from .augmentation import digest, load_paired_labels
from .trainer import make_dataset, make_loader, batch_trace, loader_seed

K_BATCHES = 8


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def collect(protocol, paired, recordings, training_seed, target, condition, repeat=False):
    seed(training_seed)
    listing = C.ROOT/protocol['datasets'][target]['train_list']['path']
    dataset = make_dataset(listing, protocol['args'], paired, target, condition, recordings)
    assert len(dataset) == 1024 and len(set(dataset.im_files)) == 512+protocol['train_unique_images']
    loader = make_loader(dataset, batch=16, workers=2, training_seed=training_seed, pin_memory=False)
    assert loader.num_workers == 2 and len(loader) == 64
    rows = []; loss_sample = None
    try:
        for index, batch in enumerate(loader):
            info = batch.pop('transfer_info')
            rows.append(batch_trace(batch, info, epoch=0, index=index))
            # First applied REAL occurrence, chosen before any model call/error.
            if loss_sample is None and target == 'REF' and condition == 'OCC' and training_seed == 42:
                for image_index, item in enumerate(info):
                    if item['role'] == 'REAL' and item['applied']:
                        selected = batch['batch_idx'].long() == image_index
                        assert int(selected.sum()) == 1
                        loss_sample = dict(image=batch['img'][image_index:image_index+1].float()/255.,
                            keypoints=batch['keypoints'][selected].clone(), box=batch['bboxes'][selected].clone(),
                            selection=dict(batch=index, name=item['name'], recording=item['recording'],
                                reason='First applied TRAIN REF OCC occurrence in fixed K8; no model/error ranking',
                                before_image=item['before_image'], after_image=item['after_image'],
                                coordinates=digest(batch['keypoints'][selected]),
                                box=digest(batch['bboxes'][selected]), covered=item['actual_covered']))
                        break
            if index+1 == K_BATCHES:
                break
    finally:
        loader.iterator._shutdown_workers()
    assert len(rows) == K_BATCHES
    suffix = '_REPEAT' if repeat else ''
    path = C.RAW/f'PREFLIGHT_TRACE_S{training_seed}_{target}_{condition}{suffix}_PRIVATE.json'
    C.save(path, rows, True)
    print('CPU_STREAM_COMPLETE', training_seed, target, condition, suffix, flush=True)
    return rows, C.bind(path), loss_sample


def paired_checks(streams):
    counts = Counter()
    for training_seed in (42, 43):
        for condition in ('CLEAR', 'OCC'):
            raw, ref = [streams[training_seed, target, condition] for target in ('RAW', 'REF')]
            for a, b in zip(raw, ref):
                for key in ('names', 'images', 'before_images', 'after_images', 'boxes', 'support', 'batch_idx', 'roles'):
                    assert a[key] == b[key], ('RAW_REF', training_seed, condition, a['batch'], key)
                for ai, bi in zip(a['transfer'], b['transfer']):
                    for key in ('name', 'role', 'recording', 'plan', 'scheduled', 'applied', 'geometric_demotions'):
                        assert ai[key] == bi[key], ('RAW_REF_PLAN', key)
                counts['RAW_REF_exact_batches'] += 1
                counts['RAW_REF_coordinate_difference_batches'] += a['coordinates'] != b['coordinates']
        for target in ('RAW', 'REF'):
            clear, occ = [streams[training_seed, target, condition] for condition in ('CLEAR', 'OCC')]
            for a, b in zip(clear, occ):
                for key in ('names', 'before_images', 'boxes', 'support', 'coordinates', 'batch_idx', 'roles'):
                    assert a[key] == b[key], ('CLEAR_OCC', training_seed, target, a['batch'], key)
                for ai, bi in zip(a['transfer'], b['transfer']):
                    assert ai['plan'] == bi['plan'] and ai['before_image'] == bi['before_image']
                    assert not ai['applied'] and ai['before_image'] == ai['after_image']
                    if ai['role'] == 'SOURCE':
                        assert ai['after_image'] == bi['after_image'] and not bi['applied']
                        counts['source_exact_occurrences'] += 1
                    elif bi['applied']:
                        assert ai['after_image'] != bi['after_image'], 'Applied occlusion changed no RGB bytes'
                        counts['real_changed_occurrences'] += 1
                        assert bi['plan']['remaining'] >= 2 and len(bi['plan']['covered']) >= 1
                    else:
                        assert ai['after_image'] == bi['after_image']
                counts['CLEAR_OCC_exact_base_and_target_batches'] += 1
    assert counts['real_changed_occurrences'] > 0
    assert counts['RAW_REF_coordinate_difference_batches'] > 0
    return dict(counts)


def seed_checks(streams, repeat):
    reference = streams[43, 'REF', 'OCC']
    assert reference == repeat, 'Seed43 worker stream not reproducible'
    a, b = streams[42, 'REF', 'OCC'], reference
    counts = Counter()
    for left, right in zip(a, b):
        counts['order_changed_batches'] += left['names'] != right['names']
        counts['base_RGB_stream_changed_batches'] += left['before_images'] != right['before_images']
        counts['plan_stream_changed_batches'] += ([row['plan'] for row in left['transfer']] !=
                                                 [row['plan'] for row in right['transfer']])
    # Match repeated image identities too, so reordered RGB alone is not called
    # independent augmentation variation.
    grouped = {}
    for s in (42, 43):
        by_name = {}
        for row in streams[s, 'REF', 'OCC']:
            for item in row['transfer']:
                if item['role'] == 'REAL':
                    by_name.setdefault(item['name'], []).append(item)
        grouped[s] = by_name
    common = sorted(set(grouped[42]) & set(grouped[43]))
    assert common, 'No common real identities to test augmentation independently of order'
    changed_rgb = sum(grouped[42][name][0]['before_image'] != grouped[43][name][0]['before_image'] for name in common)
    changed_seed = sum(grouped[42][name][0]['plan']['seed'] != grouped[43][name][0]['plan']['seed'] for name in common)
    assert all(counts[key] > 0 for key in counts) and changed_rgb > 0 and changed_seed > 0
    return dict(counts, seed43_repeat_exact_batches=len(repeat), matched_real_identities=len(common),
                matched_identity_base_RGB_changed=changed_rgb, matched_identity_plan_seed_changed=changed_seed,
                note='Compare first occurrence of matched identity in each seed; not a claim of matched occurrence slots.')


def true_ignore_check(sample, protocol):
    from scripts.self_training_yolo.v3 import verify_true_ignore_contract as V
    assert sample is not None
    C.save(C.RAW/'TRUE_IGNORE_TRAIN_FIXTURE_PRIVATE.json', sample['selection'], True)
    output = C.RAW/'TRUE_IGNORE_CPU.json'
    if output.exists():
        result = C.read(output)
        assert result['status'] == 'PASS'
        return C.bind(output)
    previous = {key: getattr(V, key) for key in ('_SAMPLE', 'OUT', 'R0', 'IMGSZ', 'real_sample', 'build')}
    original_build = V.build
    def no_evaluation_fixture():
        raise AssertionError('Evaluation-fixture fallback is forbidden')
    def cpu_build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        assert all(p.device.type == 'cpu' for p in model.parameters())
        return model
    try:
        V._SAMPLE = (sample['image'], sample['keypoints'][0, :, :2], sample['box'])
        V.OUT, V.R0 = output, C.ROOT/protocol['initialization']['path']
        V.IMGSZ = int(sample['image'].shape[-1])
        V.real_sample, V.build = no_evaluation_fixture, cpu_build
        assert V.main() == 0 and C.read(output)['status'] == 'PASS'
    finally:
        for key, value in previous.items():
            setattr(V, key, value)
    return C.bind(output)


def run():
    destination = C.DOC/'PREFLIGHT.json'
    if destination.exists():
        result = C.read(destination)
        assert result['passed']
        for binding in result['bindings']:
            C.verify(binding)
        print('CPU_PREFLIGHT_ALREADY_PASSED', flush=True)
        return result
    attempt = C.now().replace(':', '').replace('.', '')
    start = time.monotonic()
    C.save(C.RAW/'preflight_attempts'/f'{attempt}_START.json',
           dict(at=C.now(), kind='CPU_ONLY_K8', fits=0, optimizer_updates=0), True)
    try:
        torch.set_num_threads(4); cv2.setNumThreads(1)
        protocol = C.read(C.DOC/'PRIMARY_PROTOCOL.json')
        assert protocol['locked_before_fit'] and protocol['args']['workers'] == 2
        assert protocol['args']['lr0'] == 1e-5
        for binding in protocol['inputs']+protocol['sources']+[protocol['initialization']]:
            C.verify(binding)
        paired = load_paired_labels(*[C.ROOT/protocol['datasets'][target]['train_list']['path'] for target in ('RAW', 'REF')])
        rows = C.read(C.RAW/'CLEAN_LOCKED_PRIVATE.json')['rows']
        recordings = {row['train_id']+'.png': row['recording'] for row in rows}
        assert len(paired) == protocol['train_unique_images']
        streams = {}; bindings = []; sample = None
        for training_seed in (42, 43):
            for target in ('RAW', 'REF'):
                for condition in ('CLEAR', 'OCC'):
                    trace, binding, fixture = collect(protocol, paired, recordings, training_seed, target, condition)
                    streams[training_seed, target, condition] = trace
                    bindings.append(binding)
                    if fixture is not None:
                        sample = fixture
        repeat, binding, _ = collect(protocol, paired, recordings, 43, 'REF', 'OCC', repeat=True)
        bindings.append(binding)
        parity = paired_checks(streams)
        seeds = seed_checks(streams, repeat)
        loss_binding = true_ignore_check(sample, protocol)
        bindings += [loss_binding, C.bind(C.RAW/'TRUE_IGNORE_TRAIN_FIXTURE_PRIVATE.json'),
                     C.bind(C.DOC/'PRIMARY_PROTOCOL.json'), C.bind(C.DOC/'CLEAN_LOCK.json')]
        import ultralytics.data.build as installed_build
        import ultralytics.data.augment as installed_augment
        from scripts.self_training_yolo.v3 import verify_true_ignore_contract, true_ignore_pose_loss
        bindings += [C.bind(Path(__file__).with_name(name)) for name in ('preflight.py', 'trainer.py', 'augmentation.py')]
        bindings += [C.bind(Path(inspect.getfile(module))) for module in
                     (installed_build, installed_augment, verify_true_ignore_contract, true_ignore_pose_loss)]
        exposures = {}
        for (s, target, condition), trace in streams.items():
            total = Counter(); by_recording = {}
            for batch in trace:
                for item in batch['transfer']:
                    if item['role'] == 'REAL':
                        total.update(images=1, scheduled=int(item['scheduled']), applied=int(item['applied']),
                                     supervised=item['actual_supervised'], covered=item['actual_covered'],
                                     remaining=item['actual_remaining'], geometric_demotions=item['geometric_demotions'])
                        if item['plan']['scheduled'] and not item['plan']['applied']:
                            total['failed_placements'] += 1
                        if item['applied']:
                            total['bbox_fraction_sum'] += item['plan']['bbox_fraction']
                for name, value in batch['recording_exposure'].items():
                    by_recording.setdefault(name, Counter()).update(value)
            exposures[f'S{s}_{target}_{condition}'] = dict(total=dict(total), recording={k: dict(v) for k, v in by_recording.items()})
        result = dict(passed=True, created_at=C.now(), K_batches=K_BATCHES, batch_size=16, workers=2,
            streams=9, paired_checks=parity, seed_checks=seeds, exposures=exposures,
            loader_seeds={str(s): loader_seed(s) for s in (42, 43)},
            source_policy='Base stock transform; no extra occlusion or support rewriting',
            support_policy='REAL post-affine RAW/REF v2 intersection; all other REAL points true-ignore1',
            loss_gradient_contract=loss_binding, true_ignore_fixture='First applied TRAIN REF OCC occurrence, fixed before model call; evaluation fallback disabled',
            bindings=bindings, CPU_seconds=time.monotonic()-start, GPU_seconds=0.,
            fits=0, optimizer_updates=0, optimizer_constructed=False,
            evaluation_reference_coordinates_read=False,
            scope='K8 input stream preflight, not proof of every future320 training batch; full fit trace parity still required')
        C.save(destination, result, True)
        C.save(C.RAW/'preflight_attempts'/f'{attempt}_PASS.json',
               dict(at=C.now(), result=C.bind(destination), CPU_seconds=result['CPU_seconds']), True)
        print('CPU_PREFLIGHT_PASS', parity, seeds, 'seconds', result['CPU_seconds'], flush=True)
        return result
    except BaseException as exc:
        C.save(C.RAW/'preflight_attempts'/f'{attempt}_FAILURE.json',
            dict(at=C.now(), failure_kind='CONTRACT_ASSERTION' if isinstance(exc, AssertionError) else 'TECHNICAL_EXECUTION_ERROR',
                 error=repr(exc), traceback=traceback.format_exc(), CPU_seconds=time.monotonic()-start,
                 fits=0, optimizer_updates=0, GPU_seconds=0.), True)
        raise


if __name__ == '__main__':
    run()
