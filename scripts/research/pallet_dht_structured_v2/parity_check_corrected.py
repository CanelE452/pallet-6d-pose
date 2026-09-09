"""Preregistered five-image raw/cache parity; no annotation or target reads.

Preparation opens metadata/image bytes only. Execution additionally reads frozen
prediction arrays and models, and is blocked unless the full verifier advanced
on synthetic validation. Numerical failures remain failures at fixed tolerances.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import torch

from . import cache as C
from . import infer as I
from . import train as T

ARM = 'point_segment_hough'
PROBLEM = 'eval_pallet07:1778652166837872128'
TOLERANCES = dict(raw_coordinate_px=1e-3, affine=1e-6,
    unit_line_normal=1e-6, line_offset_px=1e-3,
    features_logits_costs_confidence=1e-4, rtol=0.)


def sources():
    return {**I.source_sha256(), str(Path(__file__).resolve()): C.sha(__file__)}


def state_metadata(capture):
    return {name: dict(shape=list(value.shape), dtype=str(value.dtype),
        sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        for name, value in capture.predictor.model.model.state_dict().items()}


def state_changes(before, after):
    removed, added = sorted(set(before)-set(after)), sorted(set(after)-set(before))
    changed = sorted(k for k in set(before)&set(after) if before[k] != after[k])
    return dict(removed_keys=removed, added_keys=added, changed_keys=changed,
        removed_count=len(removed), added_count=len(added), changed_count=len(changed),
        before_key_count=len(before), after_key_count=len(after))


def verify(mapping):
    for path, expected in mapping.items():
        if C.sha(path) != expected:
            raise ValueError(f'Frozen parity input changed: {path}')


def frozen_write(path, value):
    if Path(path).exists():
        if C.read(path) != value:
            raise ValueError(f'Existing parity artifact differs: {path}')
    else:
        C.write(path, value)


def selected_cases(records):
    chosen = [next(r for r in records if r['id'] == PROBLEM)]
    for population, height in [('real_dev', 26), ('real_dev', 34), ('synth_val', 30), ('synth_val', 40)]:
        pool = [r for r in records if r['population'] == population
                and r['feature_shape_hw'][0] == height and r['id'] not in {x['id'] for x in chosen}]
        if not pool:
            raise ValueError(f'No unused parity case for {population}/P4height{height}')
        chosen.append(min(pool, key=lambda r: (hashlib.sha256(r['id'].encode()).hexdigest(), r['id'])))
    if len({r['id'] for r in chosen}) != 5:
        raise ValueError('Five distinct preregistered cases required')
    return chosen


def prepare(run, seed=1):
    run = Path(run).resolve()
    protocol, bindings = T.verify_bindings(run)
    if seed not in protocol['training']['seeds']:
        raise ValueError('Unregistered trained seed')
    completion = C.read(run/'CACHE_COMPLETION.json')
    records_path, arrays_path = run/'CACHE_RECORDS.json', run/'CACHE_ARRAYS.json'
    verify({str(records_path): bindings['cache_records_sha256'], str(arrays_path): bindings['cache_arrays_sha256']})
    records = C.read(records_path)['records']
    selected = selected_cases(records)
    previous = Path(protocol['input_run'])
    snapshot_path = Path(protocol['input_snapshot']['path'])
    verify({str(snapshot_path): protocol['input_snapshot']['sha256']})
    snapshot = C.read(snapshot_path)['sha256']
    extraction_path = previous/'EXTRACTION_COMPLETE.json'
    verify({str(extraction_path): snapshot[str(extraction_path)]})
    extraction = C.read(extraction_path)
    if not (extraction['complete'] and extraction['PASS'] and extraction['warmup_forwards'] == 5):
        raise ValueError('Original five-warmup extraction required')
    paths = {str(run/name): C.sha(run/name) for name in
             ('PROTOCOL.json', 'SOURCE_FREEZE.json', 'IMPLEMENTATION_SPEC.json', 'CACHE_COMPLETION.json',
              'CACHE_RECORDS.json', 'CACHE_ARRAYS.json')}
    paths.update({str(snapshot_path): C.sha(snapshot_path), str(extraction_path): C.sha(extraction_path)})
    cases = []
    for record in selected:
        frame = previous/'frames'/f"{record['index']:06d}.npz"
        image = Path(record['image'])
        verify({str(image): record['image_sha256'], str(frame): extraction['frame_sha256'][str(frame)]})
        paths[str(image)] = record['image_sha256']
        paths[str(frame)] = extraction['frame_sha256'][str(frame)]
        cases.append({k: record[k] for k in ('index', 'id', 'population', 'image', 'image_sha256',
            'prepared_image', 'height', 'width', 'feature_shape_hw', 'input_shape_hw')})
        cases[-1].update(frame_npz=str(frame), frame_sha256=paths[str(frame)])
    result = dict(schema='pallet_dht_structured_raw_cache_parity_protocol_v1', complete=True,
        arm=ARM, seed=seed, cases=cases, tolerances=TOLERANCES,
        case_selection='Problem ID first, then SHA256(UTF8 id) ranked unused real heights26/34 and synth_val heights30/40; no GT or errors used.',
        gate_required='Completed bound full-arm synthetic advancement and its calibration-only selection',
        input_sha256=paths, source_sha256=sources(), input_array_sha256={spec['path']: completion['array_sha256'][spec['path']]
            for spec in C.read(arrays_path)['arrays']['inputs'].values()},
        matching='B1 raw and B1 cached clean scorer with the identical loaded verifier and frozen margin.',
        exact_fields=['availability masks', 'candidate order/index', 'selected index', 'centroid', 'geometry grid shape'],
        warmup_forwards=5, accuracy_forwards=5, expected_total_backbone_forwards=10,
        warmup_image=cases[0]['id'], warmup_scope='The original canonical count/settings; the warmup image is the first fixed parity case.',
        state_invariance_window='Immediately after fifth canonical warmup through tenth capture call (five actual images). Pre-warmup state recorded separately because first predict lazily fuses the backend.',
        precision_backbone=dict(FP32=True, cudnn_benchmark=False, cudnn_allow_tf32=True, matmul_allow_tf32=False),
        precision_verifier=dict(FP32=True, cudnn_benchmark=False, cudnn_allow_tf32=False, matmul_allow_tf32=False),
        no_GT_or_target_files_read=True, no_threshold_adaptation=True, failed_parity_is_not_PASS=True,
        scope='Operational five-case parity, not population accuracy or a bound on all future inputs.')
    frozen_write(run/'RAW_CACHE_PARITY_CORRECTED_PROTOCOL.json', result)
    return result


def compare(actual, reference, *, atol=None):
    """Preserve structural and numerical failures, including their exact maxima."""
    a, b = np.asarray(actual), np.asarray(reference)
    result = dict(actual_shape=list(a.shape), reference_shape=list(b.shape), exact=atol is None, atol=atol, rtol=0.)
    if a.shape != b.shape:
        return {**result, 'PASS': False, 'reason': 'shape_mismatch'}
    if a.dtype.kind in 'bOUS' or b.dtype.kind in 'bOUS':
        return {**result, 'PASS': bool(np.array_equal(a, b)), 'mismatch_count': int(np.count_nonzero(a != b))}
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {**result, 'PASS': False, 'reason': 'nonfinite'}
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    maximum = float(delta.max()) if delta.size else 0.
    position = list(np.unravel_index(int(delta.argmax()), delta.shape)) if delta.size and delta.ndim else []
    limit = 0. if atol is None else atol
    return {**result, 'PASS': maximum <= limit, 'max_abs_delta': maximum,
        'mean_abs_delta': float(delta.mean()) if delta.size else 0.,
        'mismatch_count': int((delta > limit).sum()), 'largest_index': [int(x) for x in position],
        'actual_at_largest': float(a[tuple(position)]) if delta.size else None,
        'reference_at_largest': float(b[tuple(position)]) if delta.size else None}


def execute(run, seed=1, device='cuda:0'):
    run = Path(run).resolve()
    pre = C.read(run/'RAW_CACHE_PARITY_CORRECTED_PROTOCOL.json')
    if pre != prepare(run, seed):
        raise ValueError('Parity protocol changed')
    target = run/'RAW_CACHE_PARITY_CORRECTED.json'
    if target.exists():
        previous = C.read(target)
        verify(previous['input_sha256']); verify(previous['source_sha256']); verify(previous['output_sha256'])
        if not previous['complete']:
            raise ValueError('Existing incomplete parity result must be preserved and diagnosed')
        return previous
    gate_path = run/'evaluation'/f'{ARM}_seed{seed}'/'SYNTHETIC_RESULTS.json'
    gate = C.read(gate_path)
    if not (gate['complete'] and gate['PASS'] and gate['arm'] == ARM and gate['seed'] == seed and gate['advancement']['advance']):
        raise ValueError('Full verifier has not passed the registered synthetic advancement gate')
    verify(gate['input_sha256']); verify(gate['source_sha256'])
    selection_path = run/f'SELECTION_{ARM}_seed{seed}.json'
    selection = C.read(selection_path)
    checkpoint = run/'runs'/f'{ARM}_seed{seed}'/'checkpoint_final.pth'
    if not (selection['complete'] and selection['PASS'] and selection['arm'] == ARM and selection['seed'] == seed
            and selection['no_real_selection'] and not selection['synthetic_validation_used_for_selection']
            and selection['checkpoint_sha256'] == C.sha(checkpoint)):
        raise ValueError('Completed checkpoint-bound synthetic selection required')
    verify(selection['input_sha256']); verify(selection['source_sha256'])
    input_sha = {**pre['input_sha256'], **pre['input_array_sha256'],
        str(run/'RAW_CACHE_PARITY_CORRECTED_PROTOCOL.json'): C.sha(run/'RAW_CACHE_PARITY_CORRECTED_PROTOCOL.json'),
        str(gate_path): C.sha(gate_path), str(selection_path): C.sha(selection_path), str(checkpoint): C.sha(checkpoint)}
    verify(input_sha)
    # Do not instantiate CachedData: that would open targets.npy unnecessarily.
    descriptions = C.read(run/'CACHE_ARRAYS.json')['arrays']['inputs']
    arrays = {k: np.load(v['path'], mmap_mode='r', allow_pickle=False) for k, v in descriptions.items()}
    records = C.read(run/'CACHE_RECORDS.json')['records']
    torch.set_num_threads(2)
    predictor = I.RawLayoutPredictor(checkpoint, selection['margin'], device)
    if predictor.warmup_forwards != 5:
        raise ValueError('Canonical warmup count changed')
    original_predict = predictor.capture.predict
    captured = dict(calls=0, last=None, warm_state_sha=None, warm_state_metadata=None)
    def capture(raw):
        value = original_predict(raw)
        captured['calls'] += 1
        captured['last'] = value
        if captured['calls'] == 5:
            captured['warm_state_sha'] = predictor.capture.state_sha()
            captured['warm_state_metadata'] = state_metadata(predictor.capture)
        return value
    predictor.capture.predict = capture
    observations, output_sha = [], {}
    out_dir = run/'provenance/raw_cache_parity_corrected'
    out_dir.mkdir(parents=True, exist_ok=True)
    prewarmup_state_sha = predictor.capture.state_sha()
    prewarmup_metadata = state_metadata(predictor.capture)
    try:
        for case in pre['cases']:
            record = records[case['index']]
            raw = I.OLD.load_raw(record)
            before_calls = captured['calls']
            actual = predictor.predict(raw)
            candidates, frame, _ = captured['last']
            packed, geometry = I.observation_from_capture(candidates, frame, raw.shape[:2], predictor.geometry_config)
            cached = {k: np.array(v[case['index']], copy=True) for k, v in arrays.items()}
            cached_score = I.score_observation(cached, predictor.verifier, selection['margin'], predictor.device)
            checks = {}
            for key in C.INPUT_SPECS:
                if key == 'line_h':
                    checks['packed.line_h.normal'] = compare(packed[key][..., :2], cached[key][..., :2], atol=1e-6)
                    checks['packed.line_h.offset_px'] = compare(packed[key][..., 2], cached[key][..., 2], atol=1e-3)
                    continue
                tolerance = None if np.asarray(packed[key]).dtype.kind in 'biu' else (
                    1e-6 if key == 'raw_to_input_affine' else 1e-3 if key in ('baseline_points', 'intersections', 'diagonal') else 1e-4)
                checks['packed.'+key] = compare(packed[key], cached[key], atol=tolerance)
            with np.load(case['frame_npz'], allow_pickle=False) as original:
                for key in ('p4', 'logits', 'theta', 'rho', 'lattice_valid', 'feature_shape_hw', 'input_shape_hw'):
                    tolerance = 1e-4 if key in ('p4', 'logits') else None
                    checks['capture.'+key] = compare(frame[key], original[key], atol=tolerance)
                checks['raw_affine_float64'] = compare(geometry['raw_to_input_affine'], original['raw_to_input_affine'], atol=1e-6)
            expected_baseline = record['baseline']
            for key in ('detected', 'selected_instance', 'point_valid'):
                checks['baseline.'+key] = compare(actual['baseline'][key], expected_baseline[key])
            checks['baseline.candidate_count'] = compare(len(candidates), len(expected_baseline['all_candidates']))
            if len(candidates) == len(expected_baseline['all_candidates']):
                for n, (a, b) in enumerate(zip(candidates, expected_baseline['all_candidates'])):
                    checks[f'baseline.candidate{n}.keys'] = compare(sorted(a), sorted(b))
                    for key in sorted(set(a) & set(b)):
                        tolerance = 1e-3 if key in ('box_xyxy', 'keypoints_xy') else 1e-4
                        checks[f'baseline.candidate{n}.{key}'] = compare(a[key], b[key], atol=tolerance)
            checks['verifier_used'] = compare(actual['verifier_used'], cached_score['verifier_used'])
            checks['selected_index'] = compare(actual['selected_index'], cached_score['selected_index'])
            checks['selected_points'] = compare(actual['selected_points'], cached_score['selected_points'], atol=1e-3)
            checks['centroid_between_paths'] = compare(np.asarray(actual['selected_points'])[8], np.asarray(cached_score['selected_points'])[8])
            checks['raw_centroid_preserved'] = compare(np.asarray(actual['selected_points'])[8], np.asarray(actual['baseline_points'])[8])
            checks['cached_centroid_preserved'] = compare(np.asarray(cached_score['selected_points'])[8], cached['baseline_points'][8])
            if actual['verifier_used'] and cached_score['verifier_used']:
                checks['learned_costs'] = compare(actual['costs'], cached_score['costs'], atol=1e-4)
                checks['candidate_valid'] = compare(actual['candidate_valid'], cached_score['candidate_valid'])
                checks['proposal.layouts'] = compare(actual['proposal']['layouts'], cached_score['proposal']['layouts'], atol=1e-3)
                for key in ('candidate_index', 'c4_quarters', 'kind', 'valid'):
                    checks['proposal.'+key] = compare(actual['proposal'][key], cached_score['proposal'][key])
            calls = captured['calls']-before_calls
            expected_calls = 6 if not observations else 1
            checks['actual_capture_call_count'] = compare(calls, expected_calls)
            checks['reported_warmups'] = compare(actual['timings']['warmup_forwards'], expected_calls-1)
            archive = out_dir/f"{case['index']:06d}_observations.npz"
            np.savez_compressed(archive, **{'raw_packed__'+k: np.asarray(v) for k, v in packed.items()},
                **{'cached_packed__'+k: np.asarray(v) for k, v in cached.items()},
                **{'raw_capture__'+k: np.asarray(v) for k, v in frame.items()})
            output_sha[str(archive)] = C.sha(archive)
            observations.append(dict(id=case['id'], index=case['index'], population=case['population'],
                PASS=all(x['PASS'] for x in checks.values()), checks=checks,
                failed_checks=[k for k, v in checks.items() if not v['PASS']], raw_prediction=actual,
                cached_prediction=cached_score, evidence_archive=str(archive), capture_calls=calls))
        state_after = predictor.capture.state_sha()
        final_metadata = state_metadata(predictor.capture)
    finally:
        predictor.close()
    verify(input_sha); verify(pre['source_sha256'])
    state_before = captured['warm_state_sha']
    unchanged = state_before == state_after
    passed = all(r['PASS'] for r in observations) and captured['calls'] == 10 and unchanged
    result = dict(schema='pallet_dht_structured_raw_cache_parity_result_v1', complete=True,
        PASS=passed, parity_PASS=passed, status='PASS' if passed else 'FAILED_FIXED_TOLERANCE_PARITY',
        arm=ARM, seed=seed, margin=selection['margin'], records=observations,
        n_cases=len(observations), warmup_forwards=5, accuracy_forwards=5,
        actual_capture_calls=captured['calls'], cached_verifier_batch_size=1, raw_verifier_batch_size=1,
        frozen_backbone_state_unchanged=unchanged, frozen_backbone_state_sha_before=state_before,
        frozen_backbone_state_sha_after=state_after, tolerances=TOLERANCES,
        prewarmup_state_sha256=prewarmup_state_sha,
        state_invariance_window='post-warmup5 to post-capture10; no parameter/BN update is expected after backend initialization.',
        lazy_initialization_state_changes=state_changes(prewarmup_metadata, captured['warm_state_metadata']),
        post_warmup_state_changes=state_changes(captured['warm_state_metadata'], final_metadata),
        state_metadata_before_warmup=prewarmup_metadata,
        state_metadata_after_fifth_warmup=captured['warm_state_metadata'],
        state_metadata_after_tenth_capture=final_metadata,
        tolerances_changed_after_observation=False, annotation_files_opened=0, target_arrays_opened=0,
        calibration_or_validation_error_arrays_decoded=0,
        input_sha256=input_sha, source_sha256=pre['source_sha256'], output_sha256=output_sha,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        limitations='Five fixed cases only. Numerical or selection failures are reported without relaxing limits; this does not measure population accuracy.')
    C.write(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'all'), default='prepare')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.phase in ('prepare', 'all'):
        prepare(args.run_dir, args.seed)
    if args.phase in ('run', 'all'):
        result = execute(args.run_dir, args.seed, args.device)
        print({key: result[key] for key in ('status', 'PASS', 'n_cases', 'actual_capture_calls')})
    else:
        print(args.run_dir.resolve()/'RAW_CACHE_PARITY_CORRECTED_PROTOCOL.json')
