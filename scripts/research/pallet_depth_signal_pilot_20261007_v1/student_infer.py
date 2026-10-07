"""Single-pass RGB-only student inference, with durable fail-closed attempts.

No reference, label coordinates, F, depth teacher, optimizer or scoring function
is called.  Source RGB is unpadded to its registered native canvas before the
unchanged historical standalone predictor adds its one reflected border.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C

STUDENTS = ('RAW_TARGET', 'GLOBAL_TARGET', 'DEPTH_TARGET')
SPLITS = ('EVAL', 'SOURCE', 'TRAIN_PROBE')
OLD = C.ROOT / 'data/pallet/results/pallet_clean_to_pose_transfer_v1/evaluation/S42'


def durable(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(C.PRIVATE) or path.is_relative_to(C.DOC)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    text = json.dumps(C.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with temporary.open('w') as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare():
    """Fixed IDs only, before student fits; never choose by prediction errors."""
    destination = C.DOC / 'INFERENCE_MANIFEST.json'
    if destination.exists():
        lock = C.read(destination)
        for binding in lock['sources'] + [lock['metadata']]:
            C.verify(binding)
        return lock
    protocol = C.read(C.DOC / 'PROTOCOL.json')
    teacher_lock = C.read(C.DOC / 'TEACHER_LOCK.json')
    C.verify(teacher_lock['outputs']); C.verify(teacher_lock['protocol'])
    C.verify(protocol['private_inputs'])
    inputs = C.read(C.PRIVATE / 'INPUTS_PRIVATE.json')
    teacher = C.read(C.PRIVATE / 'TEACHER_PRIVATE.json')
    source = C.read(C.DOC / 'SOURCE_EVAL_MANIFEST.json')
    assert source['status'] == 'PASS' and source['locked_before_student_fit']
    accepted_ids = sorted(r['id'] for r in teacher['TRAIN'] if r['result'].get('accepted'))
    assert len(accepted_ids) >= 32
    selected = set(accepted_ids[:32])
    by_id = {r['id']: r for r in inputs['TRAIN']}
    safe = lambda r: {k:r[k] for k in ('id', 'image', 'hw', 'K', 'xyz', 'recording', 'original_recording') if k in r}
    rows = dict(EVAL=[safe(r) for r in inputs['EVAL']],
                SOURCE=[dict(id=r['id'], image=r['image'], hw=r['raw_shape_hw'],
                             prepared_shape_hw=r['prepared_shape_hw'], reflect_pad_px=r['reflect_pad_px'],
                             scenario_id=r['scenario_id']) for r in source['rows']],
                TRAIN_PROBE=[safe(by_id[fid]) for fid in accepted_ids[:32]])
    assert {r['id'] for r in rows['TRAIN_PROBE']} == selected
    assert [len(rows[s]) for s in SPLITS] == [128, 256, 32]
    assert all(len({r['id'] for r in rows[s]}) == len(rows[s]) for s in SPLITS)
    assert not ({r['id'] for r in rows['EVAL']} & selected)
    assert not ({r['image']['sha256'] for r in rows['EVAL']} & {r['image']['sha256'] for r in rows['TRAIN_PROBE']})
    for population in rows.values():
        for row in population:
            C.verify(row['image'])
    private = C.PRIVATE / 'INFERENCE_METADATA_PRIVATE.json'
    C.save(private, rows)
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    sources = [C.DOC / 'PROTOCOL.json', C.DOC / 'TEACHER_LOCK.json',
               C.DOC / 'SOURCE_EVAL_MANIFEST.json', OLD / 'PREDICTIONS.json',
               Path(__file__), Path(inspect.getsourcefile(inspect.unwrap(predict)))]
    manifest = dict(schema='depth_pilot_student_inference_population_v1', status='PASS',
                    metadata=C.bind(private), counts={s:len(rows[s]) for s in SPLITS},
                    IDs={s:[r['id'] for r in rows[s]] for s in SPLITS},
                    TRAIN_probe_selection='first32 lexicographically sorted IDs among frozen accepted TRAIN; same qD target, selection before student fit',
                    selection_reference_access=False, inference_reference_access=False,
                    source_coordinate_contract=source['coordinate_contract'],
                    student_neural_examples_max=1248, additional_R0_neural_examples=288,
                    cached_R0_EVAL_reused=128, F_calls=0,
                    sources=[C.bind(path) for path in sources])
    C.save(destination, manifest)
    return manifest


def image_for(row):
    image = cv2.imread(str(C.ROOT / row['image']['path']))
    assert image is not None
    if 'reflect_pad_px' in row:
        assert list(image.shape[:2]) == row['prepared_shape_hw']
        padding = int(row['reflect_pad_px']); assert padding == 100
        native = image[padding:-padding, padding:-padding].copy()
        rebuilt = cv2.copyMakeBorder(native, padding, padding, padding, padding, cv2.BORDER_REFLECT_101)
        assert np.array_equal(rebuilt, image), ('Registered source reflect-border contract mismatch', row['id'])
        image = native
    assert list(image.shape[:2]) == row['hw'], ('Native image shape mismatch', row['id'])
    return image


def infer(arm):
    assert arm in ('R0', *STUDENTS)
    lock = prepare()
    for binding in lock['sources'] + [lock['metadata']]:
        C.verify(binding)
    rows = C.read(C.ROOT / lock['metadata']['path'])
    protocol = C.read(C.DOC / 'PROTOCOL.json')
    if arm == 'R0':
        checkpoint = protocol['initialization']
        jobs = [(split, row) for split in ('SOURCE', 'TRAIN_PROBE') for row in rows[split]]
    else:
        fit = C.read(C.DOC / f'FIT_{arm}_S42.json')
        assert fit['complete'] and fit['optimizer_steps'] == 320 and fit['initialization'] == protocol['initialization']
        assert fit['protected_state_exact'] and fit['checkpoint_selection'] == 'last only'
        checkpoint = fit['checkpoint']
        jobs = [(split, row) for split in SPLITS for row in rows[split]]
    C.verify(checkpoint)
    destination = C.PRIVATE / f'INFERENCE_{arm}.json'
    receipt_path = C.DOC / f'INFERENCE_RECEIPT_{arm}.json'
    attempt_path = C.PRIVATE / f'INFERENCE_ATTEMPT_{arm}.json'
    partial_path = C.PRIVATE / f'INFERENCE_PARTIAL_{arm}.jsonl'
    if receipt_path.exists():
        receipt = C.read(receipt_path)
        assert receipt['complete'] and receipt['checkpoint'] == checkpoint and receipt['manifest'] == C.bind(C.DOC / 'INFERENCE_MANIFEST.json')
        C.verify(receipt['predictions'])
        return receipt
    assert not any(p.exists() for p in (destination, attempt_path, partial_path)), 'Incomplete inference preserved; no implicit retry or overwrite'
    if arm != 'R0':
        baseline_receipt = C.read(C.DOC / 'INFERENCE_RECEIPT_R0.json')
        C.verify(baseline_receipt['predictions'])
        baseline = C.read(C.ROOT / baseline_receipt['predictions']['path'])['predictions']
    else:
        baseline = None
    stamp = dict(arm=arm, checkpoint=checkpoint, manifest=C.bind(C.DOC / 'INFERENCE_MANIFEST.json'),
                 code=C.bind(Path(__file__)), started=0, completed=0,
                 actual_model_forward_calls=0, actual_model_forward_examples=0,
                 dataset_predict_calls=0, dataset_neural_examples=0, F_calls=0, optimizer_updates=0)
    durable(attempt_path, dict(stamp, status='RESERVED_SINGLE_PASS', expected_examples=len(jobs)))
    import torch
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    assert torch.cuda.is_available(), 'Host CUDA required; no CPU fallback'
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    start = time.monotonic()
    predictions = {split:{} for split in SPLITS}
    if arm == 'R0':
        predictions['EVAL'] = C.read(OLD / 'PREDICTIONS.json')['R0']
        assert list(predictions['EVAL']) == [r['id'] for r in rows['EVAL']]
    parity = dict(detector_exact_frames=0, keypoint_confidence_changed_values=0)
    model = None
    hook = None
    try:
        model = YOLO(str(C.ROOT / checkpoint['path']), task='pose')
        # Hooks observe this one model instance only; legacy functions untouched.
        def count_forward(module, args):
            if args and isinstance(args[0], torch.Tensor):
                stamp['actual_model_forward_calls'] += 1
                stamp['actual_model_forward_examples'] += int(args[0].shape[0])
        hook = model.model.register_forward_pre_hook(count_forward)
        with partial_path.open('x') as partial:
            for index, (split, row) in enumerate(jobs):
                C.verify(row['image'])
                image = image_for(row)
                stamp['started'] += 1
                stamp['dataset_predict_calls'] += 1
                durable(attempt_path, dict(stamp, status='PREDICT_STARTED', current_split=split, current_id=row['id']))
                packet = predict(model, image)
                stamp['completed'] += 1
                stamp['dataset_neural_examples'] += 1
                predictions[split][row['id']] = packet
                partial.write(json.dumps(dict(split=split, id=row['id'], prediction=packet), allow_nan=False) + '\n')
                partial.flush(); os.fsync(partial.fileno())
                if baseline is not None:
                    previous = baseline[split][row['id']]
                    assert_detector_parity(previous, packet, teacher=False)
                    parity['detector_exact_frames'] += 1
                    for a, b in zip(previous['candidates'], packet['candidates']):
                        parity['keypoint_confidence_changed_values'] += int(np.count_nonzero(np.asarray(a['keypoints_conf']) != np.asarray(b['keypoints_conf'])))
                durable(attempt_path, dict(stamp, status='PREDICT_COMPLETE', current_split=split, current_id=row['id']))
                if (index + 1) % 32 == 0:
                    print('STUDENT_RGB_INFER', arm, index+1, len(jobs), flush=True)
        assert stamp['completed'] == stamp['started'] == len(jobs)
        assert stamp['dataset_neural_examples'] == len(jobs)
        assert all(list(predictions[s]) == [r['id'] for r in rows[s]] for s in SPLITS)
        C.save(destination, dict(checkpoint=checkpoint, manifest=C.bind(C.DOC / 'INFERENCE_MANIFEST.json'),
                                 predictions=predictions, native_unaugmented=True, highest_confidence_only=True,
                                 GT_or_target_matching=False, F_calls=0))
        receipt = dict(schema='depth_pilot_student_rgb_inference_v1', status='DONE', complete=True,
                       checkpoint=checkpoint, manifest=C.bind(C.DOC / 'INFERENCE_MANIFEST.json'),
                       predictions=C.bind(destination), partial_rows=C.bind(partial_path),
                       execution=stamp, parity=parity, seconds=time.monotonic()-start,
                       split_counts={s:len(predictions[s]) for s in SPLITS},
                       cached_R0_EVAL_reused=128 if arm == 'R0' else 0,
                       precision=dict(matmul_TF32=False, cudnn_TF32=True, cudnn_benchmark=False, half=False),
                       numerical_contract='unchanged reflect100 native standalone predictor; detector boxes/scores/index exact; trainable keypoint confidence may change',
                       references_read=False, teacher_target_input=False,
                       internal_model_forward_overhead=stamp['actual_model_forward_examples']-stamp['dataset_neural_examples'],
                       device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                       CUDA_version=torch.version.cuda, cuDNN_version=torch.backends.cudnn.version())
        durable(attempt_path, dict(stamp, status='DONE', receipt_path=str(receipt_path)))
        C.save(receipt_path, receipt)
        return receipt
    except BaseException as error:
        durable(attempt_path, dict(stamp, status='FAILED_SINGLE_PASS', error=repr(error), seconds=time.monotonic()-start))
        raise
    finally:
        if hook is not None:
            hook.remove()
        if model is not None:
            del model
        torch.cuda.empty_cache()


def finalize():
    receipts = {arm:C.read(C.DOC / f'INFERENCE_RECEIPT_{arm}.json') for arm in ('R0', *STUDENTS)}
    for receipt in receipts.values():
        assert receipt['complete'] and receipt['status'] == 'DONE'
        C.verify(receipt['predictions'])
    assert receipts['R0']['execution']['dataset_neural_examples'] == 288
    assert sum(receipts[arm]['execution']['dataset_neural_examples'] for arm in STUDENTS) == 1248
    lock = dict(schema='depth_pilot_all_student_predictions_v1', status='DONE',
                manifest=C.bind(C.DOC / 'INFERENCE_MANIFEST.json'),
                files=[receipts[arm]['predictions'] for arm in receipts],
                receipts={arm:C.bind(C.DOC / f'INFERENCE_RECEIPT_{arm}.json') for arm in receipts},
                checkpoints={arm:receipts[arm]['checkpoint'] for arm in receipts},
                student_neural_examples=1248, additional_R0_neural_examples=288,
                reused_R0_EVAL_predictions=128, F_calls=0, references_read=False,
                internal_model_forward_overhead=sum(r['internal_model_forward_overhead'] for r in receipts.values()))
    C.save(C.DOC / 'STUDENT_PREDICTIONS_LOCK.json', lock)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'infer', 'finalize'])
    parser.add_argument('--arm', choices=['R0', *STUDENTS])
    args = parser.parse_args()
    if args.stage == 'infer':
        assert args.arm
        return infer(args.arm)
    return globals()[args.stage]()


if __name__ == '__main__':
    main()
