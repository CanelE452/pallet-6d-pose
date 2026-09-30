"""Freeze pixel predictions for the fixed 128 plastic + 45 wood DEV images.

No evaluation reference, annotated coordinate, or model outcome metric is read.
The historical detector outputs are immutable inputs to every refiner. This
module deliberately does not score poses; evaluate.py may start only after the
complete nine-model PREDICTIONS_LOCK exists.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
import torch

from . import common as C
from scripts.research.pallet_posefix_large_error_v1 import core as CORE
from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved


META = C.RAW / 'EVAL_METADATA.json'
INPUT = C.DOC / 'INFERENCE_INPUT_LOCK.json'
BASE = C.RAW / 'BASELINE_CACHE.json'
PRED = C.RAW / 'predictions'
FIXED = ('R0', 'PRIOR1', 'FULL125')


def models():
    return [*FIXED, *(f'{arm}_s{seed}' for arm in C.ARMS for seed in C.SEEDS)]


def protocol_binding():
    path = C.DOC / 'PROTOCOL_EFFECTIVE.json'
    return C.bind(path if path.exists() else C.DOC / 'GOAL_PROTOCOL.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def lookup_binding(rows, path):
    rel = str(Path(path).resolve().relative_to(C.ROOT))
    found = [b for b in rows if b['path'] == rel]
    assert len(found) == 1, ('Missing or duplicated input binding', rel)
    C.verify(found[0])
    return found[0]


def preserved(base, pred):
    """Keep the existing full-candidate check, plus explicit invalid semantics."""
    assert_preserved(base, pred)
    assert len(pred['candidates']) == len(base['candidates'])
    index = base['selected_index']
    for j, (before, after) in enumerate(zip(base['candidates'], pred['candidates'])):
        for key in ('candidate_index', 'box_xyxy', 'score', 'keypoints_conf'):
            assert before.get(key) == after.get(key), (key, j)
        if j != index:
            assert before == after, ('Non-selected candidate changed', j)
    if index is not None:
        p = np.asarray(base['candidates'][index]['keypoints_xy'], float)
        q = np.asarray(pred['candidates'][index]['keypoints_xy'], float)
        invalid = ~np.isfinite(p).all(1) | (p == -1).all(1)
        invalid[8] = True
        np.testing.assert_allclose(q[invalid], p[invalid], rtol=0, atol=0,
                                   equal_nan=True)


def prepare():
    """Read and verify only caches, inference metadata, and RGB identities."""
    C.protocol()
    if INPUT.exists():
        lock = C.read(INPUT)
        C.verify(lock['protocol'])
        assert lock['protocol'] == protocol_binding()
        for b in lock['files']:
            C.verify(b)
        return lock
    final_manifest = C.D.DOC / 'RUN_MANIFEST_FINAL.json'
    diag = C.read(final_manifest)
    cache_path = C.D.RAW / 'INPUT_PREDICTIONS.json'
    meta_path = C.D.RAW / 'METADATA.json'
    bindings = [C.bind(final_manifest)]
    bindings += [lookup_binding(diag['private_artifacts'], p)
                 for p in (cache_path, meta_path)]
    baseline = C.read(cache_path)
    plastic = C.read(meta_path)
    assert len(plastic) == 128
    wood_root = C.ROOT / 'data/pallet/results/pallet_material_selftrain_closure_v1'
    wood_lock_path = C.ROOT / '_docs/experiments/pallet_material_selftrain_closure_v1/WOOD_PREDICTIONS_LOCK.json'
    wood_lock = C.read(wood_lock_path)
    bindings.append(C.bind(wood_lock_path))
    for p in (wood_root / 'EVAL_METADATA.json', wood_root / 'PREDICTIONS.json'):
        bindings.append(lookup_binding(wood_lock['files'], p))
    wood = C.read(wood_root / 'EVAL_METADATA.json')
    wood_predictions = C.read(wood_root / 'PREDICTIONS.json')['R0']
    lineage_path = C.D.DOC / 'MODEL_LINEAGE.json'
    lineage = C.read(lineage_path)['models']
    checkpoints = {name: lineage[name]['checkpoint'] for name in FIXED}
    assert wood_lock['checkpoints']['R0'] == checkpoints['R0']
    assert checkpoints['R0']['sha256'] == '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    bindings += [C.bind(lineage_path), *checkpoints.values()]
    assert len(wood) == 45 and {r['id'] for r in wood} == set(wood_predictions)
    allowed = {'id', 'K', 'xyz', 'hw', 'recording', 'recording_group',
               'severity', 'session', 'image', 'object_type'}
    records = plastic + wood
    assert all(set(r) <= allowed for r in records), 'Non-inference metadata field'
    ids = [r['id'] for r in records]
    assert len(ids) == len(set(ids)) == 173
    plastic_ids = [r['id'] for r in plastic]
    for name in ('identity', 'PRIOR1', 'FULL125'):
        assert set(plastic_ids) == set(baseline[name])
    cache = {'R0': {**baseline['identity'], **wood_predictions},
             'PRIOR1': baseline['PRIOR1'], 'FULL125': baseline['FULL125']}
    for name in ('PRIOR1', 'FULL125'):
        for fid in plastic_ids:
            preserved(cache['R0'][fid], cache[name][fid])
    for r in records:
        C.verify(r['image'])
    C.save(META, records)
    C.save(BASE, cache)
    bindings += [C.bind(META), C.bind(BASE)]
    for b in bindings:
        C.verify(b)
    lock = dict(protocol=protocol_binding(), original_protocol=C.bind(C.DOC / 'GOAL_PROTOCOL.json'),
                files=bindings, metadata=C.bind(META), baseline_cache=C.bind(BASE),
                checkpoints=checkpoints, image_bindings=[r['image'] for r in records],
                IDs=ids, plastic_IDs=plastic_ids, wood_IDs=[r['id'] for r in wood],
                population=dict(plastic128=128, wood45=45, total=173),
                models=models(), detector_new_forwards=0,
                allowed_refiner_inputs='RGB, frozen selected R0 box/coordinates/confidence only',
                geometry_metadata='K/xyz retained for later scoring, never passed to refiner',
                references_read=False, baseline_history='Reused existing DEV predictions; historical scoring exists')
    C.save(INPUT, lock)
    print('INFERENCE_INPUTS_LOCKED', len(ids), flush=True)
    return lock


def checkpoint_for(name, input_lock):
    if name in FIXED:
        return input_lock['checkpoints'][name]
    arm, seed_text = name.rsplit('_s', 1)
    assert arm in C.ARMS and int(seed_text) in C.SEEDS
    path = C.RAW / 'fits' / name / 'final.pt'
    assert path.is_file(), ('Required completed fit missing', name)
    return C.bind(path)


def load_refiner(name, binding, device):
    """New final.pt uses canonical, unwrapped PoseFixPallet9 state keys."""
    C.verify(binding)
    ck = torch.load(C.ROOT / binding['path'], map_location='cpu', weights_only=False)
    model = CORE.PoseFixPallet9()
    if name == 'PRIOR1':
        assert ck['complete'] and ck['step'] == 6000
        state = ck['model_state_dict']
    elif name == 'FULL125':
        fit = C.read(C.L.DOC / 'FIT_FULL.json')
        assert fit['complete'] and fit['checkpoint'] == binding and ck['step'] == 300
        assert C.L.state_hash(ck['model_state_dict']) == fit['final_state_sha']
        assert all(k.startswith('model.') for k in ck['model_state_dict'])
        state = {k.removeprefix('model.'): v for k, v in ck['model_state_dict'].items()}
    else:
        arm, seed = name.rsplit('_s', 1)
        assert ck['complete'] and ck['step'] == C.protocol()['updates']
        assert ck['arm'] == arm and ck['seed'] == int(seed)
        assert ck['protocol_sha256'] == protocol_binding()['sha256']
        state = ck['state']
        assert not any(k.startswith('model.') for k in state)
        assert C.L.state_hash(state) == ck['final_state_sha']
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    return model.to(device).eval().requires_grad_(False)


def infer_model(name, device='cuda'):
    lock = prepare()
    assert name in models()
    destination = PRED / f'{name}.json'
    receipt = PRED / f'{name}_RECEIPT.json'
    if destination.exists() and receipt.exists():
        result = C.read(receipt)
        C.verify(result['predictions'])
        C.verify(result['checkpoint'])
        C.verify(result['protocol'])
        assert result['protocol'] == protocol_binding()
        return result
    start = time.monotonic()
    cpu_start = time.process_time()
    cache = C.read(BASE)
    rows = C.read(META)
    raw = cache['R0']
    reused = copy.deepcopy(cache.get(name, {}))
    need = [r for r in rows if r['id'] not in reused]
    assert len(need) == (0 if name == 'R0' else 45 if name in FIXED else 173)
    checkpoint = checkpoint_for(name, lock)
    context = dict(model=name, checkpoint=checkpoint, protocol=protocol_binding(),
                   input_lock=C.bind(INPUT), code=C.bind(__file__), device=device,
                   torch_version=torch.__version__, precision='FP32', TF32=False,
                   cudnn_benchmark=False, cudnn_deterministic=True,
                   cap_fraction=None, crop_expansion=1.25,
                   expectation_decoder='unchanged CORE.predict')
    context_hash = digest(context)
    C.save(PRED / '_frames' / name / 'CONTEXT.json', context)
    prepared = []
    for index, row in enumerate(need):
        frame_path = PRED / '_frames' / name / f'{index:03d}.json'
        if frame_path.exists():
            frame = C.read(frame_path)
            assert frame['context_sha256'] == context_hash and frame['id'] == row['id']
            preserved(raw[row['id']], frame['prediction'])
        else:
            prepared.append((row, frame_path))
    model = None
    state_before = None
    if prepared:
        C.L.L.setup(device)
        cv2.setNumThreads(1)
        model = load_refiner(name, checkpoint, device)
        state_before = C.L.state_hash(model.state_dict())
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
    for index, (row, frame_path) in enumerate(prepared):
        C.verify(row['image'])
        image = cv2.imread(str(C.ROOT / row['image']['path']))
        assert image is not None and list(image.shape[:2]) == row['hw']
        has_forward = CORE.prepare_input(image, raw[row['id']]) is not None
        if device == 'cuda':
            torch.cuda.synchronize()
        frame_start = time.monotonic()
        prediction = CORE.predict(model, image, raw[row['id']], cap_fraction=None)
        if device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - frame_start
        preserved(raw[row['id']], prediction)
        C.save(frame_path, dict(id=row['id'], context_sha256=context_hash,
                               prediction=prediction, image_forward=int(has_forward),
                               inference_seconds=seconds, GT_read=False))
        if (index + 1) % 20 == 0 or index + 1 == len(prepared):
            if device == 'cuda':
                C.L.L.gpu_guard()
            print('PREDICT', name, index + 1, '/', len(prepared), flush=True)
    if model is not None:
        assert C.L.state_hash(model.state_dict()) == state_before, 'Inference mutated model/BN'
    total_forwards = 0
    measured_seconds = 0.0
    for index, row in enumerate(need):
        frame = C.read(PRED / '_frames' / name / f'{index:03d}.json')
        assert frame['context_sha256'] == context_hash and frame['id'] == row['id']
        reused[row['id']] = frame['prediction']
        total_forwards += frame['image_forward']
        measured_seconds += frame['inference_seconds']
    output = {row['id']: reused[row['id']] for row in rows}
    for fid, pred in output.items():
        preserved(raw[fid], pred)
    C.save(destination, output)
    result = dict(complete=True, model=name, predictions=C.bind(destination),
                  checkpoint=checkpoint, protocol=protocol_binding(),
                  context=C.bind(PRED / '_frames' / name / 'CONTEXT.json'),
                  rows=len(output), cached_rows=len(rows) - len(need),
                  new_prediction_rows=len(need), image_forwards=total_forwards,
                  device=device, dtype='float32', TF32=False,
                  frame_inference_seconds=measured_seconds,
                  current_process_wall_seconds=time.monotonic() - start,
                  current_process_CPU_seconds=time.process_time() - cpu_start,
                  current_process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_GPU_memory_bytes=torch.cuda.max_memory_allocated() if prepared and device == 'cuda' else 0,
                  model_state_unchanged=True, frozen_detector_preserved=True,
                  nonselected_candidates_unchanged=True, invalid_and_center_preserved=True,
                  reference_coordinates_read=False, fits=0, optimizer_updates=0)
    C.save(receipt, result)
    del model
    gc.collect()
    if device == 'cuda' and prepared:
        torch.cuda.empty_cache()
    print('PREDICTIONS_COMPLETE', name, len(output), total_forwards, flush=True)
    return result


def finalize():
    lock = prepare()
    rows = C.read(META)
    expected = {r['id'] for r in rows}
    raw = C.read(BASE)['R0']
    predictions, checkpoints, receipts = {}, {}, {}
    forwards = 0
    for name in models():
        path = PRED / f'{name}.json'
        receipt_path = PRED / f'{name}_RECEIPT.json'
        assert path.is_file() and receipt_path.is_file(), ('Incomplete model set', name)
        receipt = C.read(receipt_path)
        for key in ('predictions', 'checkpoint', 'protocol', 'context'):
            C.verify(receipt[key])
        assert receipt['protocol'] == protocol_binding() and receipt['complete']
        pred = C.read(path)
        assert set(pred) == expected
        for fid in expected:
            preserved(raw[fid], pred[fid])
        predictions[name] = C.bind(path)
        checkpoints[name] = receipt['checkpoint']
        receipts[name] = C.bind(receipt_path)
        forwards += receipt['image_forwards']
    budget = C.protocol()['budget']
    assert forwards <= budget['new_model_inference_max'] + budget['frozen_PRIOR1_FULL125_wood_baselines_max']
    output = dict(complete=True, frozen_at=C.now(), models=models(), IDs=[r['id'] for r in rows],
                  metadata=C.bind(META), predictions=predictions, checkpoints=checkpoints,
                  receipts=receipts, protocol=protocol_binding(), original_protocol=lock['original_protocol'],
                  input_lock=C.bind(INPUT), code=C.bind(__file__), image_forwards=forwards,
                  all_new_predictions_frozen_before_new_GT_scoring=True,
                  inference_reference_coordinates_read=False,
                  detector_box_score_confidence_center_invalid_preserved=True,
                  historical_baseline_scoring_exists=True, independent_confirmation=False,
                  next_stage='Separate same-GEO pose scoring; no inference model/seed selection')
    target = C.DOC / 'PREDICTIONS_LOCK.json'
    if target.exists():
        previous = C.read(target)
        output['frozen_at'] = previous['frozen_at']
    C.save(target, output)
    print('ALL_PREDICTIONS_FROZEN', len(models()), len(rows), forwards, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'baselines', 'new', 'all', 'finalize', 'model'])
    parser.add_argument('--model', choices=models())
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'finalize':
        finalize()
    elif args.stage == 'model':
        assert args.model is not None, '--model is required'
        infer_model(args.model, args.device)
    else:
        names = list(FIXED) if args.stage == 'baselines' else models()[3:] if args.stage == 'new' else models()
        for name in names:
            infer_model(name, args.device)
        if args.stage in ('new', 'all'):
            finalize()


if __name__ == '__main__':
    main()
