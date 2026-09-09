"""Keep frozen paper boxes/scores and evaluate replacement points in isolation.

--phase cache reconstructs the frozen R0 full-candidate cache once. It performs
baseline inference, never inference with the new model. Other phases are CPU
replays. Original evaluators are imported unchanged; only their input/output
path globals are bound to this isolated experiment directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
POSE = REPO/'data/pallet/results/paper_pose_metric_closure_v1'
OLD2D = REPO/'data/pallet/results/paper_eval_v1/arms/R0.json'
OLDCSV = OLD2D.with_name('R0_per_frame.csv')
POS = REPO/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
NEG = POS.with_name('DEV_NEG2689.json')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts/paper/pose_metric_closure_v1'))
from challenge.evaluation_v2 import paper_real_eval as E


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def freeze(path, value):
    if path.exists():
        assert read(path) == value, f'Existing frozen artifact differs: {path}'
    else:
        write(path, value)
    return value


def canonical_key(path):
    return E._display_path((REPO/path).resolve())


def population():
    pair = E.validate_evaluation_request(positive_manifest=POS, negative_manifest=NEG,
        population_role=E.PopulationRole.DEV, allow_unavailable_final=False)
    assert len(pair.positive.items) == 319 and len(pair.negative.items) == 2689
    return pair


def fixed_inputs():
    paths = [POS, NEG, OLD2D, OLDCSV, POSE/'INFERENCE_REPLAY_LOCK.json',
             POSE/'POSE_ARM_CHECKPOINT_LOCK.json', POSE/'AXIS_REVIEW_MANIFEST.json',
             POSE/'POSE_EVAL_OBJECT_CONTRACT.json', POSE/'GEOMETRY_RESOLVED_POSE_GT.json',
             POSE/'POSE_EVALUATION_R0.json', POSE/'predictions/R0.json',
             POSE/'POSE_PER_FRAME_BY_ARM.json', REPO/'_docs/paper/final/PAPER_CANONICAL_NUMBER_SOURCES.json']
    paths += [REPO/'scripts/paper/pose_metric_closure_v1'/name for name in
              ('run_pose_evaluation.py', 'evaluate_pose_by_session.py', 'paired_bootstrap_pose.py',
               'symmetry_aware_pose_metrics.py', 'pose_evaluation_paths.py')]
    paths += [Path(inspect.getfile(E)), REPO/'challenge/evaluation_v2/pnp_selector.py',
              REPO/'challenge/evaluation_v2/oriented_iou3d.py',
              REPO/'scripts/paper/paired_uncertainty_and_tails.py']
    for name in ('OBJECT_GEOMETRY_REGISTRY.json', 'MIGRATION_GATE.json', 'SYMMETRY_CONTRACT.json'):
        paths.append(REPO/'challenge/real_gt_v2'/name)
    paths.append(REPO/'challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json')
    for item in population().positive.items:
        paths.append((REPO/item.label).resolve())
    for frame in read(POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']:
        paths.append((REPO/frame['annotation']).resolve())
    return {str(path): sha(path) for path in dict.fromkeys(paths)}


def protocol(run_dir):
    checkpoint = read(POSE/'POSE_ARM_CHECKPOINT_LOCK.json')['arms']['R0']
    weights = REPO/checkpoint['checkpoint']
    assert sha(weights) == checkpoint['sha256'] == read(OLD2D)['weights']['sha256']
    value = dict(schema='pallet_line_pose_paper_baseline_protocol_v1', weights=str(weights),
        weights_sha256=checkpoint['sha256'], recipe=read(POSE/'INFERENCE_REPLAY_LOCK.json')['recipe'],
        source_sha256=fixed_inputs(),
        extraction_function_sha256=hashlib.sha256(inspect.getsource(extract_baseline).encode()).hexdigest(),
        evaluator_recipe_class_sha256=hashlib.sha256(inspect.getsource(E._UltralyticsPredictor).encode()).hexdigest(),
        population=dict(positive=319, negative=2689, role='DEV', held_out_final=False),
        inference_policy='Baseline R0 inference only, all319positive+2689negative once; no new-model inference. Exact frozen paper predictor class, reflect100/imgsz640/conf0.001 and original unpadding.',
        keypoint_metric='Original-pixel supervised visibility>0 keypoints0..8 pooled on top-score IoU>=0.5 matched frames. Prediction keypoint-confidence threshold is not applied.',
        pose_metric='Unchanged MAIN prediction-only axis selector plus SQPnP/RefineLM; geometry-reconstructed reference; same object contract and coverage denominator.',
        negative_policy='Once baseline full candidates are captured, new arms copy every negative candidate including order, boxes, scores and points. Hash identity proves unchanged evaluated classifier output; no new negative forward pass or speed claim.',
        replacement_schema='pallet_line_pose_point_replacements_v1: baseline_cache_sha256, arm, selection_artifact{path,sha256}, frames{positive_cache_image_key:[{candidate_index,keypoints_xy[9,2]}]}. All319positive keys required; empty replacement list is allowed.',
        bootstrap='Unchanged paper pooled-median bootstrap10000 seed20260902 and MAIN pose bootstrap10000 seed20260903, paired complete-session resampling. Reused DEV, no independent generalization claim.')
    return freeze(run_dir/'BASELINE_PROTOCOL.json', value)


def assert_sources(protocol_value):
    for path, digest in protocol_value['source_sha256'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    assert sha(protocol_value['weights']) == protocol_value['weights_sha256']


def extract_baseline(run_dir, binding):
    target = run_dir/'baseline/FULL_CANDIDATES.json'
    if target.exists():
        cached = read(target)
        assert cached['complete'] and cached['protocol_sha256'] == sha(run_dir/'BASELINE_PROTOCOL.json')
        return target
    pair = population()
    predictor = E._UltralyticsPredictor(Path(binding['weights']), '0')
    frozen_top = read(POSE/'predictions/R0.json')['frames']
    frozen_top_by_image = {canonical_key(f['image']): frozen_top[f['frame_id']]
                          for f in read(POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']}
    frozen_csv = {row['frame_id']: row for row in csv.DictReader(OLDCSV.open())}
    frames, metadata, parity = {}, {}, []
    maximum = 0.
    started = time.perf_counter()
    for kind, items in (('positive', pair.positive.items), ('negative', pair.negative.items)):
        for item in items:
            path = (REPO/item.image).resolve()
            candidates = predictor.predict(path)
            key = E._display_path(path)
            assert key not in frames
            frames[key] = [dict(score=float(score), box_xyxy=box.tolist(),
                               keypoints_xy=points.tolist() if points is not None else None)
                           for score, box, points in candidates]
            metadata[key] = dict(frame_id=item.frame_id, kind=kind, image_sha256=sha(path),
                                 session_id=getattr(item, 'session_id', None))
            row = frozen_csv[item.frame_id]
            assert int(row['candidate_count']) == len(candidates), f'Candidate count parity: {item.frame_id}'
            if candidates:
                top = max(frames[key], key=lambda x: x['score'])
                expected_box = [float(row[f'top_box_{axis}{j}']) for j in (1, 2) for axis in ('x', 'y')]
                delta = max(abs(top['score']-float(row['top_score'])),
                            float(np.max(np.abs(np.asarray(top['box_xyxy'])-expected_box))))
                if kind == 'positive':
                    old = frozen_top_by_image[key]
                    delta = max(delta, float(np.max(np.abs(np.asarray(top['keypoints_xy'])-old['keypoints_xy']))))
                maximum = max(maximum, delta)
                assert delta == 0., f'Frozen baseline numerical parity failed: {item.frame_id}, delta={delta}'
            parity.append(dict(frame_id=item.frame_id, kind=kind, candidate_count=len(candidates)))
            if len(frames) % 250 == 0:
                print(f'Frozen paper R0 cache {len(frames)}/3008', flush=True)
    assert len(frames) == 3008 and sum(map(len, frames.values())) == 4961
    assert_sources(binding)
    payload = dict(schema_version='paper_cached_predictions_v1', complete=True, model='R0',
        weights_sha256=binding['weights_sha256'], recipe=binding['recipe'],
        protocol_sha256=sha(run_dir/'BASELINE_PROTOCOL.json'), frames=frames, frame_metadata=metadata,
        candidate_count=4961, source_sha256=binding['source_sha256'],
        all_candidate_scope='All candidates newly replayed under frozen predictor. Historical artifacts retain only top1/count; full historical non-top candidate identity cannot be proven.',
        inference_seconds=time.perf_counter()-started)
    write(target, payload)
    write(run_dir/'baseline/CACHE_AUDIT.json', dict(PASS=True, complete=True, n_frames=3008,
        positive=319, negative=2689, candidates=4961, top_score_box_max_abs_delta=maximum,
        positive_top_keypoints_max_abs_delta=maximum, candidate_counts_all_match=True,
        historical_full_candidate_identity_available=False, raw_candidate_sha256=value_sha(frames),
        output_sha256=sha(target), per_frame=parity, original_sources_unchanged=True))
    print(f'Baseline cache complete: {target}', flush=True)
    return target


def replace_points(run_dir, cache_path, supplied, arm):
    cache = read(cache_path)
    frames = copy.deepcopy(cache['frames'])
    metadata = cache['frame_metadata']
    positive = {k for k, v in metadata.items() if v['kind'] == 'positive'}
    selection = None
    supplied_identity = None
    if supplied is not None:
        incoming = read(supplied)
        supplied_identity = dict(path=str(supplied), sha256=sha(supplied),
            metadata={k: v for k, v in incoming.items() if k != 'frames'})
        if incoming.get('schema_version') == 'paper_cached_predictions_v1':
            assert incoming.get('baseline_cache_sha256') == sha(cache_path)
            frames = incoming['frames']
        else:
            assert incoming['schema'] == 'pallet_line_pose_point_replacements_v1'
            assert incoming['baseline_cache_sha256'] == sha(cache_path) and incoming['arm'] == arm
            assert set(incoming['frames']) == positive
            for key, replacements in incoming['frames'].items():
                seen = set()
                for replacement in replacements:
                    index = replacement['candidate_index']
                    assert isinstance(index, int) and index not in seen and 0 <= index < len(frames[key])
                    seen.add(index)
                    value = np.asarray(replacement['keypoints_xy'], float)
                    assert value.shape == (9, 2) and np.isfinite(value).all()
                    frames[key][index]['keypoints_xy'] = value.tolist()
        selection = incoming['selection_artifact']
        assert sha(selection['path']) == selection['sha256'], 'Frozen selection artifact changed'
        selected = read(selection['path'])
        assert selected.get('complete') is True and selected.get('no_real_selection') is True
        assert selected.get('selection_population') == 'synth_val', 'Only frozen synthetic selection may precede paper DEV evaluation'
    assert list(frames) == list(cache['frames']), 'Candidate frame order changed'
    altered = 0
    for key, candidates in frames.items():
        old = cache['frames'][key]
        assert len(candidates) == len(old)
        for candidate, reference in zip(candidates, old):
            assert candidate['score'] == reference['score'] and candidate['box_xyxy'] == reference['box_xyxy']
            p, q = candidate['keypoints_xy'], reference['keypoints_xy']
            if q is None:
                assert p is None, 'Missing point prediction was reconstructed'
            elif p != q:
                assert key in positive
                assert np.asarray(p).shape == (9, 2) and np.isfinite(p).all()
                altered += 1
        if key not in positive:
            assert candidates == old, 'Negative raw candidate altered'
    negatives = {k: v for k, v in frames.items() if k not in positive}
    original_negatives = {k: v for k, v in cache['frames'].items() if k not in positive}
    box_scores = lambda fs: {k: [dict(score=v['score'], box_xyxy=v['box_xyxy']) for v in vs] for k, vs in fs.items()}
    assert value_sha(negatives) == value_sha(original_negatives)
    assert value_sha(box_scores(frames)) == value_sha(box_scores(cache['frames']))
    destination = run_dir/'evaluation'/arm
    freeze(destination/'PREDICTIONS.json', dict(schema_version='paper_cached_predictions_v1', complete=True,
        model=arm, weights_sha256=cache['weights_sha256'], baseline_cache_sha256=sha(cache_path),
        selection_artifact=selection, supplied_prediction_artifact=supplied_identity,
        recipe=cache['recipe'], frames=frames))
    write(destination/'IDENTITY_AUDIT.json', dict(PASS=True, complete=True, frames=3008,
        n_altered_positive_candidates=altered, all_boxes_scores_order_unchanged=True,
        negative_frames=2689, negative_candidate_count=sum(map(len, negatives.values())),
        negative_raw_candidates_sha256=value_sha(negatives), negative_reinference_performed=False,
        all_box_score_sha256=value_sha(box_scores(frames)), baseline_cache_sha256=sha(cache_path),
        explanation='New arms reuse baseline negative candidates byte-value-identically; classification outputs are preserved by construction. This does not measure a fresh negative forward path or model speed.'))
    return destination, frames, altered


def paper_2d(destination, predictions, weights):
    output = destination/'PAPER_2D.json'
    if not output.exists():
        argv = ['--positive-manifest', str(POS), '--negative-manifest', str(NEG), '--population-role', 'DEV',
            '--weights', weights, '--predictions', str(predictions), '--out', str(output),
            '--per-frame-out', str(destination/'PAPER_2D_per_frame.csv'),
            '--report-out', str(destination/'PAPER_2D.md'),
            '--migration-gate', str(REPO/'challenge/real_gt_v2/MIGRATION_GATE.json'),
            '--symmetry-contract', str(REPO/'challenge/real_gt_v2/SYMMETRY_CONTRACT.json'),
            '--object-migration-gate', f'wood_small_80x59x14={REPO}/challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json']
        E.run(E.build_parser().parse_args(argv))
    return read(output)


def paper_pose(destination, frames, arm):
    frozen = read(POSE/'predictions/R0.json')
    new = copy.deepcopy(frozen)
    new['arm'] = arm
    for item in read(POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']:
        candidates = frames[canonical_key(item['image'])]
        if not candidates:
            new['frames'][item['frame_id']] = dict(status='NO_DETECTION')
            continue
        top = max(candidates, key=lambda c: c['score'])
        new['frames'][item['frame_id']].update(box_xyxy=top['box_xyxy'], box_conf=top['score'],
                                              keypoints_xy=top['keypoints_xy'], detections=len(candidates))
    new['actual_source_predictions'] = str(destination/'PREDICTIONS.json')
    new['actual_source_sha256'] = sha(destination/'PREDICTIONS.json')
    freeze(destination/f'predictions/{arm}.json', new)
    freeze(destination/'predictions/R0.json', frozen)
    if not (destination/f'POSE_EVALUATION_{arm}.json').exists():
        module = importlib.import_module('run_pose_evaluation')
        previous = module.OUT_DIR, module.PREDICTIONS, sys.argv
        module.OUT_DIR, module.PREDICTIONS = destination, destination/'predictions'
        sys.argv = [str(Path(module.__file__)), '--pose-object-contract', str(POSE/'POSE_EVAL_OBJECT_CONTRACT.json'), '--arm', arm]
        try:
            assert module.main() == 0
        finally:
            module.OUT_DIR, module.PREDICTIONS, sys.argv = previous
    freeze(destination/'POSE_EVALUATION_R0.json', read(POSE/'POSE_EVALUATION_R0.json'))
    if not (destination/'POSE_PER_FRAME_BY_ARM.json').exists():
        module = importlib.import_module('evaluate_pose_by_session')
        names = ('OUT_DIR', 'DOC_DIR', 'PREDICTIONS', 'ARMS', 'LABELS')
        old = {name: getattr(module, name) for name in names}
        module.OUT_DIR, module.DOC_DIR, module.PREDICTIONS = destination, destination/'reports', destination/'predictions'
        module.ARMS, module.LABELS = ['R0', arm], {'R0': 'Frozen R0', arm: arm}
        try:
            assert module.main() == 0
        finally:
            for name, value in old.items():
                setattr(module, name, value)
    return read(destination/f'POSE_EVALUATION_{arm}.json')


def paired_statistics(destination, arm, passthrough):
    pose = importlib.import_module('paired_bootstrap_pose')
    output = destination/'POSE_PAIRED_BOOTSTRAP.json'
    if not output.exists():
        names = ('OUT_DIR', 'DOC_DIR', 'PER_FRAME', 'PAIRS')
        old = {name: getattr(pose, name) for name in names}
        pose.OUT_DIR, pose.DOC_DIR = destination, destination/'reports'
        pose.PER_FRAME, pose.PAIRS = destination/'POSE_PER_FRAME_BY_ARM.json', [(arm, 'R0')]
        try:
            assert pose.main() == 0
        finally:
            for name, value in old.items():
                setattr(pose, name, value)
    sys.path.insert(0, str(REPO/'scripts/paper'))
    two_d = importlib.import_module('paired_uncertainty_and_tails')
    def frame_rows(path):
        return {r['image']: dict(strict=[float(v) for v in r['top_keypoint_supervised_errors_px'].split(';')]
                  if r['top_keypoint_supervised_errors_px'] else [], session=r['session_id'])
                for r in csv.DictReader(path.open()) if r['kind'] == 'POSITIVE'}
    left, right = frame_rows(OLDCSV), frame_rows(destination/'PAPER_2D_per_frame.csv')
    keys = sorted(left)
    assert set(keys) == set(right)
    sessions = np.array([left[k]['session'] for k in keys])
    old_rng = two_d.RNG
    two_d.RNG = np.random.default_rng(20260902)
    try:
        frame = two_d.pooled_median_bootstrap(left, right, keys)
        cluster = two_d.pooled_median_bootstrap(left, right, keys, sessions)
    finally:
        two_d.RNG = old_rng
    write(destination/'KEYPOINT_PAIRED_BOOTSTRAP.json', dict(complete=True, arm=arm, reference='R0',
        statistic='Pooled supervised9-keypoint median difference; paired matched frames', resamples=10000,
        seed=20260902, frame_level=frame, session_cluster=cluster, n_sessions=len(set(sessions)),
        passthrough=passthrough, limitations='Reused DEV; sessions are few and no independent real holdout was accessed.'))


def evaluate(run_dir, supplied, arm, binding):
    cache_path = run_dir/'baseline/FULL_CANDIDATES.json'
    assert cache_path.exists(), 'Run --phase cache before CPU evaluation'
    cache_audit = read(run_dir/'baseline/CACHE_AUDIT.json')
    assert cache_audit['PASS'] and cache_audit['output_sha256'] == sha(cache_path)
    assert arm != 'R0' and re.fullmatch(r'[A-Za-z0-9_]+', arm)
    destination, frames, changed = replace_points(run_dir, cache_path, supplied, arm)
    two_d = paper_2d(destination, destination/'PREDICTIONS.json', binding['weights'])
    pose = paper_pose(destination, frames, arm)
    expected2d = read(OLD2D)['metrics']['box_and_keypoint_2d']
    actual2d = two_d['metrics']['box_and_keypoint_2d']
    expectedpose = read(POSE/'POSE_EVALUATION_R0.json')['paths']['MAIN']['ALL']
    actualpose = pose['paths']['MAIN']['ALL']
    keys2d = ['box_ap50', 'box_ap50_95', 'candidate_count', 'keypoint_matched_frame_count_iou50',
              'keypoint_location_median_px', 'keypoint_location_p90_px', 'keypoint_supervision_count']
    errors = {'2d': {k: float(actual2d[k])-float(expected2d[k]) for k in keys2d},
              'pose': {k: float(actualpose[k])-float(expectedpose[k]) for k in expectedpose}}
    assert all(abs(errors['2d'][k]) < 1e-12 for k in ('box_ap50', 'box_ap50_95', 'candidate_count'))
    if changed == 0:
        assert all(abs(v) < 1e-9 for block in errors.values() for v in block.values()), errors
    paired_statistics(destination, arm, passthrough=changed == 0)
    assert_sources(binding)
    write(destination/'RESULTS.json', dict(schema='pallet_line_pose_paper_evaluation_v1', complete=True,
        arm=arm, reference='R0', population=dict(positive=319, negative=2689, role='DEV', held_out_final=False),
        two_d={k: actual2d[k] for k in keys2d}, two_d_labeled_points=actual2d['keypoint_all_labeled']['count'],
        main_6d=actualpose, main_6d_coverage=pose['paths']['MAIN']['coverage'],
        reference_values=dict(two_d={k: expected2d[k] for k in keys2d}, main_6d=expectedpose),
        reference_sources=dict(two_d=dict(path=str(OLD2D), sha256=sha(OLD2D), key='metrics.box_and_keypoint_2d'),
            main_6d=dict(path=str(POSE/'POSE_EVALUATION_R0.json'), sha256=sha(POSE/'POSE_EVALUATION_R0.json'), key='paths.MAIN.ALL')),
        deltas=errors, negative_outputs_preserved=True,
        primary_limits='Paper2D pools supervised9-keypoints on IoU50 matched top detections; MAIN6D uses geometry-reconstructed reference and reports available-pose coverage. Reused DEV, not independent final evaluation.'))
    write(destination/'COMPLETION.json', dict(complete=True, PASS=True, arm=arm,
        baseline_passthrough=changed == 0, changed_positive_candidates=changed,
        point_predictions_sha256=sha(destination/'PREDICTIONS.json'), baseline_protocol_sha256=sha(run_dir/'BASELINE_PROTOCOL.json'),
        authoritative_baseline_deltas=errors, source_sha256=binding['source_sha256'],
        adapter_source_sha256=sha(Path(__file__)),
        actual_bindings=dict(predictions=str(destination/'PREDICTIONS.json'), pose_predictions=str(destination/'predictions'),
            gt=str(POSE/'GEOMETRY_RESOLVED_POSE_GT.json'), object_contract=str(POSE/'POSE_EVAL_OBJECT_CONTRACT.json')),
        original_evaluator_files_modified=False, negative_reinference_performed=False,
        legacy_metadata_note='Frozen pose evaluators retain historical prose/source labels in their raw reports. Actual new-arm input paths and provenance are recorded here; evaluator computations are unchanged.',
        output_sha256={p.name: sha(p) for p in destination.glob('*.json') if p.name != 'COMPLETION.json'}))
    print(json.dumps(dict(PASS=True, arm=arm, baseline_deltas=errors), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--arm', default='R0_PASSTHROUGH')
    parser.add_argument('--phase', choices=('cache', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    assert (run_dir/'PURPOSE.md').is_file()
    binding = protocol(run_dir)
    if args.phase in ('cache', 'all'):
        extract_baseline(run_dir, binding)
    if args.phase in ('evaluate', 'all'):
        evaluate(run_dir, args.predictions.resolve() if args.predictions else None, args.arm, binding)


if __name__ == '__main__':
    main()
