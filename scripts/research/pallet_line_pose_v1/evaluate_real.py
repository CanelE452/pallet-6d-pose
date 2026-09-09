"""Evaluate synthetic-selected heads with frozen paper candidates and evaluators.

--phase check is an offline contract audit: it performs no image loading/model
inference. All other phases require the complete synthetic selection and nine
completed step-6000 checkpoints. New real-image predictions cannot select rules.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.modules.head import Pose26

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as FX
import inference as I
import paper_evaluation as PE

ARMS = ('image_joint', 'geometry_joint', 'image_line_only')
SEEDS = (1, 2, 3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources():
    paths = [HERE/name for name in ('evaluate_real.py', 'inference.py', 'features.py',
                                   'model.py', 'readout.py', 'paper_evaluation.py')]
    paths += [Path(inspect.getfile(x)) for x in (YOLO, BasePredictor, Pose26)]
    return {str(p): PE.sha(p) for p in dict.fromkeys(paths)}


def baseline_inputs(run_dir):
    binding = PE.read(run_dir/'BASELINE_PROTOCOL.json')
    cache_path = run_dir/'baseline/FULL_CANDIDATES.json'
    cache = PE.read(cache_path)
    audit = PE.read(run_dir/'baseline/CACHE_AUDIT.json')
    require(cache.get('complete') and audit.get('PASS') and audit.get('complete'), 'Incomplete baseline replay')
    require(audit['output_sha256'] == PE.sha(cache_path), 'Baseline full-candidate hash changed')
    require(cache['protocol_sha256'] == PE.sha(run_dir/'BASELINE_PROTOCOL.json'), 'Baseline protocol mismatch')
    require(binding['weights_sha256'] == cache['weights_sha256'] == FX.BASELINE_SHA, 'Wrong baseline checkpoint')
    PE.assert_sources(binding)
    keys = [key for key in cache['frames'] if cache['frame_metadata'][key]['kind'] == 'positive']
    require(len(keys) == 319 and len(cache['frames']) == 3008, 'Frozen paper population changed')
    require(all(PE.canonical_key(key) == key for key in keys), 'Noncanonical positive image locator')
    return binding, cache, keys


def timing_keys(cache, keys):
    grouped = defaultdict(list)
    for key in keys:
        grouped[cache['frame_metadata'][key]['session_id']].append(key)
    require(len(grouped) == 13 and None not in grouped, 'Expected thirteen paper sessions')
    return [key for session in sorted(grouped) for key in
            sorted(grouped[session], key=lambda k: (hashlib.sha256(k.encode()).hexdigest(), k))[:2]]


def plan(run_dir, cache, keys):
    return PE.freeze(run_dir/'REAL_EVALUATION_PLAN.json', dict(
        schema='pallet_line_pose_real_evaluation_plan_v1', expected_runs=9,
        arms=list(ARMS), seeds=list(SEEDS), positive_frames=319, negative_frames=2689,
        baseline_cache_sha256=PE.sha(run_dir/'baseline/FULL_CANDIDATES.json'),
        inference='Original BGR only; no GT input. Canonical reflected YOLO wrapper. All candidates checked against frozen baseline before point replacement. Only highest-score instance corners0..7 may move.',
        selection='Complete synthetic-only SELECTION.json and all final step6000 checkpoints required before any new real forward pass.',
        timing=dict(keys=timing_keys(cache, keys), sampling='SHA256 of canonical image locator, first two per session; independent of predictions and GT error.',
            warmup=5, repeats=3, batch=1, order='Baseline/integrated order alternates by repeat and frame index; each pair uses the same already-loaded image.',
            definition='Synchronized perf_counter wall time from original uint8 BGR through reflect padding, YOLO preprocessing/forward/postprocessing, and integrated head/readout/original-pixel CPU output when used. Image file reading/decoding, model loading, parity audit, JSON writing and PnP excluded.',
            pnp_timing='Not measured separately: frozen paper evaluators expose evaluation results, not a synchronized deployment PnP-only latency. No pose-pipeline speed claim.',
            statistic='Median/P90 plus paired integrated-minus-baseline overhead; no inference-speed superiority criterion is selected from these data.'),
        role='reused DEV', held_out_final=False))


def selected_runs(run_dir):
    path = run_dir/'SELECTION.json'
    selected = PE.read(path)
    require(selected.get('schema') == 'pallet_line_pose_synthetic_selection_v1'
            and selected.get('complete') is True and selected.get('no_real_selection') is True
            and selected.get('selection_population') == 'synth_val', 'Frozen synthetic selection is required')
    rows = selected['runs']
    require(len(rows) == 9 and {(r['arm'], int(r['seed'])) for r in rows} ==
            {(a, s) for a in ARMS for s in SEEDS}, 'Selection must bind all nine unique arm/seed runs')
    runs = []
    for arm in ARMS:
        for seed in SEEDS:
            row = next(r for r in rows if (r['arm'], int(r['seed'])) == (arm, seed))
            checkpoint = I._resolve(row['checkpoint'], path.parent)
            digest = PE.sha(checkpoint)
            require(digest == row['checkpoint_sha256'], 'Selected checkpoint changed')
            data = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(data.get('schema') == 'pallet_line_pose_checkpoint_v1' and data.get('complete') is True
                    and data.get('smoke') is False and data.get('step') == data.get('expected_steps') == 6000
                    and (data.get('arm'), data.get('seed')) == (arm, seed), 'Only fixed completed main checkpoints are evaluable')
            completion_path = checkpoint.parent/'COMPLETION.json'
            completion = PE.read(completion_path)
            require(completion.get('complete') is True and completion.get('PASS') is True
                    and completion.get('smoke') is False and completion.get('step') == 6000
                    and completion['checkpoint_sha256'] == digest, 'Training completion proof failed')
            require(all(bool(torch.isfinite(v).all()) for v in data['model_state_dict'].values()), 'Nonfinite model checkpoint')
            rule = I.load_selection(path, data, digest)
            runs.append(dict(arm=arm, seed=seed, evaluation_arm=f'{arm}_seed{seed}', checkpoint=str(checkpoint),
                checkpoint_sha256=digest, training_completion_sha256=PE.sha(completion_path), selected_rule=rule))
    return selected, runs


def exact(a, b, label):
    if a is None or b is None:
        require(a is None and b is None, f'{label}: missing prediction changed')
        return
    x, y = np.asarray(a), np.asarray(b)
    if (x.shape != y.shape or not np.array_equal(x, y, equal_nan=True)
            or not np.array_equal(np.signbit(x), np.signbit(y))):
        delta = None if x.shape != y.shape else float(np.max(np.abs(x-y)))
        raise ValueError(f'{label}: frozen baseline parity failed; shapes={x.shape}/{y.shape}; max_abs_delta={delta}')


def check_prediction(prediction, references, *, frame_key):
    """No GT: reconstruct the wrapper's baseline stream and compare every item."""
    candidates = prediction['candidates']
    selected = prediction['selected_index']
    require(len(candidates) == len(references), f'{frame_key}: candidate count changed')
    expected = int(np.argmax([r['score'] for r in references])) if references else None
    require(selected == expected, f'{frame_key}: highest-score instance changed')
    changed = 0
    for index, (candidate, old) in enumerate(zip(candidates, references)):
        label = f'{frame_key}/candidate{index}'
        require(candidate.get('candidate_index') == index, f'{label}: candidate order changed')
        exact(candidate['score'], old['score'], label+'/score')
        exact(candidate['box_xyxy'], old['box_xyxy'], label+'/box')
        baseline = prediction['baseline_selected'] if index == selected else candidate
        require(baseline is not None, label+': missing baseline copy')
        exact(baseline['score'], old['score'], label+'/baseline_score')
        exact(baseline['box_xyxy'], old['box_xyxy'], label+'/baseline_box')
        exact(baseline['keypoints_xy'], old['keypoints_xy'], label+'/baseline_points')
        if index != selected or old['keypoints_xy'] is None or float(prediction['lam']) == 0:
            exact(candidate['keypoints_xy'], old['keypoints_xy'], label+'/preserved_points')
        else:
            points, previous = np.asarray(candidate['keypoints_xy']), np.asarray(old['keypoints_xy'])
            require(points.shape == (9, 2) and np.isfinite(points).all(), label+': invalid refined points')
            exact(points[8], previous[8], label+'/centroid')
            exact(candidate['keypoints_conf'], baseline['keypoints_conf'], label+'/keypoint_confidence')
            changed += int(np.any(points != previous))
    return changed


def load_bgr(key, metadata):
    path = (PE.REPO/key).resolve()
    require(PE.sha(path) == metadata['image_sha256'], f'Frozen positive image changed: {key}')
    image = cv2.imread(str(path))
    require(image is not None, f'Cannot decode image: {key}')
    return image


def assert_binding(value, binding):
    require(value.get('complete') is True and value.get('bindings') == binding, 'Existing real artifact binding differs')


def infer_one(run_dir, cache, keys, row, binding):
    directory = run_dir/'evaluation'/row['evaluation_arm']
    directory.mkdir(parents=True, exist_ok=True)
    output, audit_path = directory/'IMAGE_PREDICTIONS.json', directory/'INFERENCE_AUDIT.json'
    replacement_path = directory/'POINT_REPLACEMENTS.json'
    if output.exists():
        saved = PE.read(output); assert_binding(saved, binding)
        audit = PE.read(audit_path); assert_binding(audit, binding)
        require(audit['PASS'] and audit['image_predictions_sha256'] == PE.sha(output)
                and audit['point_replacements_sha256'] == PE.sha(replacement_path), 'Saved inference audit failed')
        require([r['image_key'] for r in saved['records']] == keys, 'Saved positive order changed')
        for record in saved['records']:
            check_prediction(record['prediction'], cache['frames'][record['image_key']], frame_key=record['image_key'])
        return replacement_path
    records, replacements, changed = [], {}, 0
    with I.PalletLinePoseInference(row['checkpoint'], run_dir/'SELECTION.json', device='cuda') as predictor:
        for key in keys:
            metadata = cache['frame_metadata'][key]
            image = load_bgr(key, metadata)
            prediction = predictor.predict(image, include_logits=False)
            try:
                changed += check_prediction(prediction, cache['frames'][key], frame_key=key)
            except Exception as error:
                PE.write(directory/'INFERENCE_FAILURE.json', dict(complete=False, PASS=False, bindings=binding,
                    image_key=key, error=str(error), prediction=I.jsonable(prediction), frozen_candidates=cache['frames'][key]))
                raise
            selected = prediction['selected_index']
            replacements[key] = [] if selected is None or prediction['candidates'][selected]['keypoints_xy'] is None else [dict(candidate_index=selected,
                keypoints_xy=I.jsonable(prediction['candidates'][selected]['keypoints_xy']))]
            records.append(dict(image_key=key, frame_id=metadata['frame_id'], image_sha256=metadata['image_sha256'],
                                prediction=I.jsonable(prediction)))
            if len(records) % 50 == 0:
                PE.write(directory/'INFERENCE_PROGRESS.json', dict(complete=False, frames=len(records), expected_frames=319, bindings=binding))
                print(f"Real inference {row['evaluation_arm']} {len(records)}/319", flush=True)
    require(sources() == binding['source_sha256'], 'Inference code changed during real extraction')
    PE.freeze(replacement_path, dict(schema='pallet_line_pose_point_replacements_v1', complete=True,
        baseline_cache_sha256=binding['baseline_cache_sha256'], arm=row['evaluation_arm'],
        selection_artifact=dict(path=str(run_dir/'SELECTION.json'), sha256=binding['selection_sha256']),
        bindings=binding, frames=replacements))
    PE.freeze(output, dict(schema='pallet_line_pose_real_image_predictions_v1', complete=True,
        arm=row['arm'], seed=row['seed'], bindings=binding, records=records))
    PE.write(audit_path, dict(complete=True, PASS=True, bindings=binding, frames=319,
        image_predictions_sha256=PE.sha(output), point_replacements_sha256=PE.sha(replacement_path),
        changed_selected_instances=changed, all_candidate_box_score_order_exact=True,
        baseline_keypoints_all_candidates_max_abs_delta_px=0., centroid_max_abs_delta_px=0.,
        nonselected_instances_unchanged=True, inference_gt_input=False,
        missing_detections_preserved=True, negative_forward_performed=False))
    return replacement_path


class PlainBaseline:
    """Canonical frozen paper predictor, accepting already-loaded original BGR."""
    def __init__(self, weights):
        self.model = YOLO(str(weights), task='pose')

    def predict(self, image):
        padded = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        result = self.model.predict(padded, conf=.001, imgsz=640, device='cuda', verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        scores = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()-100
        points = None if result.keypoints is None else result.keypoints.xy.detach().cpu().numpy()-100
        return [dict(score=float(scores[j]), box_xyxy=boxes[j].astype(np.float64),
                     keypoints_xy=None if points is None else points[j].astype(np.float64)) for j in range(len(scores))]


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter()-start)*1000


def stats(values):
    values = np.asarray(values, float)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)), p90=float(np.percentile(values, 90)))


def benchmark(run_dir, cache, rows, binding, timing):
    target = run_dir/'RUNTIME.json'
    if target.exists():
        value = PE.read(target); assert_binding(value, binding)
        require(value['PASS'] and value['parity_PASS'], 'Existing runtime parity failed')
        return target
    images = {key: load_bgr(key, cache['frame_metadata'][key]) for key in timing['keys']}
    baseline = PlainBaseline(FX.BASELINE)
    records, by_run = [], {}
    context, prediction, reference = None, None, None
    try:
        for row in rows:
            saved = PE.read(run_dir/'evaluation'/row['evaluation_arm']/'IMAGE_PREDICTIONS.json')
            assert_binding(saved, dict(**binding, run=row))
            prior = {r['image_key']:r['prediction'] for r in saved['records']}
            current = []
            with I.PalletLinePoseInference(row['checkpoint'], run_dir/'SELECTION.json', 'cuda') as predictor:
                for i in range(timing['warmup']):
                    image = images[timing['keys'][i % len(images)]]
                    baseline.predict(image); predictor.predict(image)
                for repeat in range(timing['repeats']):
                    for index, key in enumerate(timing['keys']):
                        image = images[key]
                        context = dict(evaluation_arm=row['evaluation_arm'], image_key=key, repeat=repeat)
                        prediction, reference = None, None
                        if (repeat+index) % 2:
                            prediction, integrated_ms = timed(lambda: predictor.predict(image))
                            reference, baseline_ms = timed(lambda: baseline.predict(image))
                        else:
                            reference, baseline_ms = timed(lambda: baseline.predict(image))
                            prediction, integrated_ms = timed(lambda: predictor.predict(image))
                        require(len(reference) == len(cache['frames'][key]), 'Runtime baseline candidate count mismatch')
                        for j, (actual, old) in enumerate(zip(reference, cache['frames'][key])):
                            for name in ('score', 'box_xyxy', 'keypoints_xy'):
                                exact(actual[name], old[name], f'timing/{key}/{j}/{name}')
                        check_prediction(prediction, cache['frames'][key], frame_key=key)
                        for j, (actual, old) in enumerate(zip(prediction['candidates'], prior[key]['candidates'])):
                            exact(actual['keypoints_xy'], old['keypoints_xy'], f'timing-versus-accuracy/{key}/{j}')
                        current.append(dict(arm=row['arm'], seed=row['seed'], evaluation_arm=row['evaluation_arm'],
                            image_key=key, session_id=cache['frame_metadata'][key]['session_id'], repeat=repeat,
                            baseline_first=(repeat+index) % 2 == 0, baseline_ms=baseline_ms, integrated_ms=integrated_ms,
                            added_ms=integrated_ms-baseline_ms, head_used=prediction['head_used']))
            records += current
            by_run[row['evaluation_arm']] = dict(arm=row['arm'], seed=row['seed'], lam=row['selected_rule']['lam'],
                baseline_ms=stats([r['baseline_ms'] for r in current]), integrated_ms=stats([r['integrated_ms'] for r in current]),
                paired_added_ms=stats([r['added_ms'] for r in current]), head_used_fraction=float(np.mean([r['head_used'] for r in current])))
            print(f"Real timing complete {row['evaluation_arm']}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    except Exception as error:
        PE.write(run_dir/'RUNTIME_FAILURE.json', dict(complete=False, PASS=False, bindings=binding,
            error=str(error), context=context, prediction=I.jsonable(prediction), baseline_prediction=I.jsonable(reference),
            records=records, scope='No replacement of accuracy outputs after a runtime parity failure.'))
        raise
    require(sources() == binding['source_sha256'], 'Runtime source code changed')
    PE.write(target, dict(schema='pallet_line_pose_runtime_v1', complete=True, PASS=True, parity_PASS=True,
        bindings=binding, protocol=timing, by_run=by_run, records=records,
        baseline_and_integrated_accuracy_prediction_max_abs_delta_px=0.,
        runtime=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), batch=1, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        scope='Actual repeated image-to-keypoint latency on26 fixed development frames. PnP and image decoding are excluded; lambda0 skips the trainable branch.'))
    return target


def verify_cell(directory, expected_binding):
    inference_audit = PE.read(directory/'INFERENCE_AUDIT.json')
    assert_binding(inference_audit, expected_binding)
    require(inference_audit['PASS']
            and inference_audit['image_predictions_sha256'] == PE.sha(directory/'IMAGE_PREDICTIONS.json')
            and inference_audit['point_replacements_sha256'] == PE.sha(directory/'POINT_REPLACEMENTS.json'),
            'Real inference artifact binding failed')
    completion = PE.read(directory/'COMPLETION.json')
    require(completion['complete'] and completion['PASS'], 'Incomplete paper evaluation')
    for name, digest in completion['output_sha256'].items():
        require(PE.sha(directory/name) == digest, f'Paper output changed: {directory/name}')
    identity = PE.read(directory/'IDENTITY_AUDIT.json')
    require(identity['PASS'] and identity['all_boxes_scores_order_unchanged'] and identity['negative_frames'] == 2689
            and not identity['negative_reinference_performed'], 'Paper negative identity audit failed')
    require(PE.read(directory/'RESULTS.json')['complete'], 'Missing paper results')


def offline_check(run_dir, cache, keys):
    """Toy outputs exercise failure guards, without loading a real image/model."""
    old = [dict(score=.9, box_xyxy=[1.,2.,3.,4.], keypoints_xy=np.arange(18).reshape(9,2).tolist()),
           dict(score=.5, box_xyxy=[2.,3.,4.,5.], keypoints_xy=np.ones((9,2)).tolist())]
    candidates = [dict(**copy.deepcopy(r), candidate_index=j, keypoints_conf=np.ones(9).tolist()) for j,r in enumerate(old)]
    good = dict(candidates=candidates, selected_index=0, baseline_selected=copy.deepcopy(candidates[0]), lam=1.)
    good['candidates'][0]['keypoints_xy'][0][0] += 1
    require(check_prediction(good, old, frame_key='toy') == 1, 'Valid refinement toy failed')
    cases = {}
    for name in ('box', 'score', 'order', 'baseline_point', 'centroid', 'nonselected', 'lambda_zero', 'selection'):
        bad = copy.deepcopy(good)
        if name == 'box': bad['candidates'][0]['box_xyxy'][0] += 1
        elif name == 'score': bad['candidates'][0]['score'] += .01
        elif name == 'order': bad['candidates'][1]['candidate_index'] = 0
        elif name == 'baseline_point': bad['baseline_selected']['keypoints_xy'][0][0] += 1
        elif name == 'centroid': bad['candidates'][0]['keypoints_xy'][8][0] += 1
        elif name == 'nonselected': bad['candidates'][1]['keypoints_xy'][0][0] += 1
        elif name == 'lambda_zero': bad['lam'] = 0.
        else: bad['selected_index'] = 1
        try: check_prediction(bad, old, frame_key='toy')
        except ValueError: cases[name] = True
        else: raise AssertionError(f'Mutation guard failed: {name}')
    check_prediction(dict(candidates=[], selected_index=None, baseline_selected=None, lam=1), [], frame_key='missing')
    PE.write(run_dir/'REAL_SETUP_AUDIT.json', dict(PASS=True, complete=True, schema='pallet_line_pose_real_setup_audit_v1',
        real_model_inference_performed=False, real_images_loaded=False, baseline_positive_metadata_count=len(keys),
        timing_frames=len(timing_keys(cache,keys)), timing_sessions=13, mutation_guards=cases,
        missing_detection_preserved=True, source_sha256=sources(),
        scope='Offline preparation only; not new-model accuracy, runtime, or experiment completion.'))


def run(run_dir, phase='all'):
    require((run_dir/'PURPOSE.md').is_file(), 'Missing experiment purpose')
    original, cache, keys = baseline_inputs(run_dir)
    planned = plan(run_dir, cache, keys)
    if phase == 'check':
        offline_check(run_dir, cache, keys)
        print('Offline real-evaluation contract PASS; no new real image loaded or inferred', flush=True)
        return
    selected, rows = selected_runs(run_dir)
    binding = dict(selection_sha256=PE.sha(run_dir/'SELECTION.json'),
        baseline_cache_sha256=PE.sha(run_dir/'baseline/FULL_CANDIDATES.json'),
        baseline_protocol_sha256=PE.sha(run_dir/'BASELINE_PROTOCOL.json'),
        evaluation_plan_sha256=PE.sha(run_dir/'REAL_EVALUATION_PLAN.json'), source_sha256=sources(), runs=rows)
    PE.freeze(run_dir/'REAL_EVALUATION_PROTOCOL.json', dict(complete=True, **binding, purpose='Bindings frozen before opening images for selected-model real inference.'))
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False
    for row in rows:
        directory = run_dir/'evaluation'/row['evaluation_arm']
        cell_binding = dict(**binding, run=row)
        if phase in ('all', 'infer'):
            infer_one(run_dir, cache, keys, row, cell_binding)
            gc.collect(); torch.cuda.empty_cache()
        if phase in ('all', 'evaluate'):
            audit = PE.read(directory/'INFERENCE_AUDIT.json'); assert_binding(audit, cell_binding)
            require(audit['PASS'] and audit['point_replacements_sha256'] == PE.sha(directory/'POINT_REPLACEMENTS.json'), 'Point-replacement audit mismatch')
            if not (directory/'COMPLETION.json').exists():
                PE.evaluate(run_dir, directory/'POINT_REPLACEMENTS.json', row['evaluation_arm'], original)
            verify_cell(directory, cell_binding)
    if phase in ('all', 'timing'):
        runtime_path = benchmark(run_dir, cache, rows, binding, planned['timing'])
    else:
        return
    summaries = []
    for row in rows:
        directory = run_dir/'evaluation'/row['evaluation_arm']; verify_cell(directory, dict(**binding, run=row))
        summaries.append(dict(arm=row['arm'], seed=row['seed'], evaluation_arm=row['evaluation_arm'], output_dir=str(directory),
            completion=str(directory/'COMPLETION.json'), completion_sha256=PE.sha(directory/'COMPLETION.json'),
            result_sha256=PE.sha(directory/'RESULTS.json'), identity_audit_sha256=PE.sha(directory/'IDENTITY_AUDIT.json'),
            inference_audit_sha256=PE.sha(directory/'INFERENCE_AUDIT.json'), checkpoint_sha256=row['checkpoint_sha256']))
    PE.assert_sources(original)
    require(sources() == binding['source_sha256'] and PE.sha(run_dir/'SELECTION.json') == binding['selection_sha256'], 'Final source/selection binding changed')
    PE.write(run_dir/'REAL_EVALUATION_COMPLETE.json', dict(complete=True, PASS=True, expected_runs=9, runs=summaries,
        selection_sha256=binding['selection_sha256'], baseline_cache_sha256=binding['baseline_cache_sha256'],
        source_sha256=binding['source_sha256'], runtime=dict(path=str(runtime_path), sha256=PE.sha(runtime_path)),
        positive_frames=319, negative_frames=2689, negative_outputs_preserved=True,
        real_selection_performed=False, held_out_final=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'check', 'infer', 'evaluate', 'timing'), default='all')
    args = parser.parse_args()
    try:
        run(args.run_dir.resolve(), args.phase)
    except Exception as error:
        PE.write(args.run_dir.resolve()/'REAL_EVALUATION_FAILURE.json', dict(complete=False, PASS=False,
            phase=args.phase, error=str(error), traceback=traceback.format_exc(), real_selection_performed=False))
        raise
