"""Fixed GREEN0918_119 saved-prediction audit; CPU image correction and 2D only.

The protocol is written before correction.  Stored R0/N3 scores are reused, while
the four new arms use the unchanged whole-object evaluator. No model or pose
entry point is imported or called here.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from . import methods as A
from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M

SNAPSHOT = C.ROOT / '_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json'
SYMMETRY = C.ROOT / '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'
PREDICTIONS = C.ROOT / 'data/pallet/results/pallet_n3_completion_v3/SQUARE_YOLO_PREDICTIONS.json'
METRICS = C.ROOT / 'data/pallet/results/pallet_n3_completion_v3/SQUARE_YOLO_METRICS.json'
PROTOCOL = C.DOC / 'SQUARE_PROTOCOL.json'
ROW_PATH = C.DOC / 'SQUARE_PREDICTIONS.jsonl.gz'
RECEIPT = C.DOC / 'SQUARE_PREDICTIONS.json'
MODES = ('manual_declared', 'manual_in_frame')
BASELINES = {'BASE': 'R0', **{f'N3_seed{s}': f'N3_DIM_SYM_seed{s}' for s in (1, 2, 3)}}


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else str(path),
            'sha256': C.sha(path), 'bytes': path.stat().st_size}


def verify(entry):
    path = Path(entry['path'])
    path = path if path.is_absolute() else C.ROOT / path
    assert path.is_file() and path.stat().st_size == entry['bytes'], str(path)
    assert C.sha(path) == entry['sha256'], str(path)
    return path


def _truth(records, symmetry):
    groups = [r for r in symmetry['objects']
              if r['object_type'] == 'plastic_standard_110x110x15']
    assert len(groups) == 1 and groups[0]['group_order'] == 4
    permutations = np.asarray(groups[0]['permutations'], dtype=np.int64)
    assert permutations.shape == (4, 9) and np.all(permutations[:, 8] == 8)
    assert all(np.array_equal(np.sort(p), np.arange(9)) for p in permutations)
    result = {}
    for record in records:
        annotation = C.read(C.ROOT / record['annotation']['path'])
        gt, known = annotation_arrays(annotation)
        _, manual = annotation_arrays(annotation, True)
        h, w = record['original_hw']
        inside = np.isfinite(gt).all(-1) & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
        match_points = known & inside
        assert match_points.any(), record['id']
        result[record['id']] = dict(gt=gt, hw=[h, w], permutations=permutations,
                                   box=np.r_[gt[match_points].min(0), gt[match_points].max(0)],
                                   valid={'manual_declared': manual, 'manual_in_frame': manual & inside})
    assert sum(r['valid']['manual_declared'][:8].sum() for r in result.values()) == 602
    assert sum(r['valid']['manual_in_frame'][:8].sum() for r in result.values()) == 600
    return result


def _selected(row):
    selected = row['selected_index']
    candidates = row['candidates']
    assert selected == (int(np.argmax([r['score'] for r in candidates])) if candidates else None)
    return None if selected is None else candidates[selected]


def _points(candidate):
    points = np.full((9, 2), np.nan) if candidate is None else np.asarray(candidate['keypoints_xy'], dtype=np.float64)
    assert points.shape == (9, 2)
    return points


def _support(points):
    return np.isfinite(points).all(-1) & ~np.all(points == -1, axis=-1)


def _canonical_observed(points, target, mode, metric):
    observed = np.zeros(8, dtype=bool)
    if metric['evaluable']:
        perm = target['permutations'][metric['branch']]
        valid = np.asarray(target['valid'][mode], bool) & np.isfinite(target['gt']).all(-1) & ~np.all(target['gt'] == -1, axis=-1)
        native_observed = valid[perm[:8]] & _support(points)[:8] & bool(metric['matched'])
        observed[perm[:8][native_observed]] = True
        assert observed.sum() == len(metric['observed_errors'])
    return observed.tolist()


def _load():
    snapshot, payload, metrics = C.read(SNAPSHOT), C.read(PREDICTIONS), C.read(METRICS)
    records = snapshot['records']
    ids = [r['id'] for r in records]
    assert len(ids) == len(set(ids)) == 119 and len({r['session'] for r in records}) == 1
    assert payload['complete'] and payload['frames'] == 119 and metrics['complete']
    assert metrics['predictions']['sha256'] == C.sha(PREDICTIONS)
    assert metrics['snapshot']['sha256'] == C.sha(SNAPSHOT)
    assert metrics['symmetry_contract']['sha256'] == C.sha(SYMMETRY)
    truth = _truth(records, C.read(SYMMETRY))
    predictions, saved = {}, {}
    for name, old_name in BASELINES.items():
        rows = payload['predictions'][old_name]
        assert len(rows) == 119 and {r['id'] for r in rows} == set(ids)
        predictions[name] = {r['id']: r for r in rows}
        saved[name] = {}
        for mode in MODES:
            assert metrics['modes'][mode]['manual_corner_denominator'] == (602 if mode == 'manual_declared' else 600)
            mm = metrics['modes'][mode]['rows'][old_name]
            assert len(mm) == 119 and {r['id'] for r in mm} == set(ids)
            saved[name][mode] = {r['id']: r for r in mm}
        for record in records:
            row = predictions[name][record['id']]
            assert row['raw_hw'] == record['original_hw']
            chosen = _selected(row)
            points = _points(chosen)
            base = predictions['BASE'][record['id']]
            assert row['selected_index'] == base['selected_index'] and len(row['candidates']) == len(base['candidates'])
            for idx, (candidate, original) in enumerate(zip(row['candidates'], base['candidates'])):
                assert candidate.keys() == original.keys()
                assert all(candidate[key] == original[key] for key in candidate if key != 'keypoints_xy')
                assert np.array_equal(_points(candidate)[8], _points(original)[8], equal_nan=True)
                if idx != row['selected_index']:
                    assert np.array_equal(_points(candidate), _points(original), equal_nan=True)
            for mode in MODES:
                metric = saved[name][mode][record['id']]
                assert metric['session'] == record['session']
                assert metric['detected'] == (chosen is not None)
                assert metric['canonical_valid'] == (np.asarray(truth[record['id']]['valid'][mode], bool)[:8]).tolist()
                _canonical_observed(points, truth[record['id']], mode, metric)
    return records, truth, predictions, saved


def build_protocol():
    """Validate and join frozen inputs without image correction or scoring."""
    records, truth, predictions, saved = _load()
    sources = [binding(p) for p in (SNAPSHOT, SYMMETRY, PREDICTIONS, METRICS,
                                    Path(annotation_arrays.__code__.co_filename), Path(M.__file__),
                                    C.ROOT / 'scripts/research/pallet_n3_completion_v3/square.py',
                                    C.ROOT / 'scripts/research/pallet_n3_completion_v3/square_yolo.py',
                                    C.DOC / 'SQUARE_PROOF.json')]
    for record in records:
        assert record['canonical_WDH_m'] == [1.1, 1.1, .15]
        for key in ('image', 'annotation'):
            actual = binding(C.ROOT / record[key]['path'])
            assert actual['sha256'] == record[key]['sha256'] and actual['bytes'] == record[key]['bytes']
            sources.append(actual)
    native = []
    for record in records:
        row = predictions['BASE'][record['id']]
        chosen = _selected(row)
        points = _points(chosen)
        native.append({'id': record['id'], 'session': record['session'], 'raw_hw': row['raw_hw'],
                       'selected_index': row['selected_index'], 'candidate_count': len(row['candidates']),
                       'native_points': C.finite(points), 'prediction_support': _support(points).tolist(),
                       'fixed_detector_metadata': chosen})
    return C.finite(dict(schema='pallet_training_free_square119_protocol_v1', status='SEALED',
                        full_ids=[r['id'] for r in records], frames=119, methods=list(C.ARMS),
                        original_image_native_algorithm_calls=238, new_2d_rows=476,
                        reference={'manual_declared': 602, 'manual_in_frame': 600},
                        whole_object_C4_permutations=next(iter(truth.values()))['permutations'],
                        source_bindings=sources, square_code=binding(__file__),
                        configuration=A.method_configuration(), parent_protocol=binding(C.DOC / 'PROTOCOL.json'),
                        raw_contract=native, raw_contract_sha256=C.digest(native),
                        prediction_support='finite native coordinate pair and not [-1,-1]; no GT/confidence threshold',
                        baseline_metric_calls=0, baselines=BASELINES,
                        square_reselection_or_tuning=False, correlated_sessions=1,
                        confidence_intervals=0, new_detector_calls=0, new_refiner_calls=0, new_F_calls=0,
                        posefix_status='BLOCKED_2D_PoseFix', physical_pose_status='BLOCKED_NO_INDEPENDENT_CANONICAL_POSE_REFERENCE'))


def execute(protocol_path=PROTOCOL, expected_protocol_sha256=None):
    """Run exactly once after the supplied protocol has been sealed."""
    protocol_path = Path(protocol_path)
    assert expected_protocol_sha256 and C.sha(protocol_path) == expected_protocol_sha256
    protocol = C.read(protocol_path)
    assert protocol['status'] == 'SEALED' and protocol['square_code']['sha256'] == C.sha(__file__)
    assert protocol['configuration'] == A.method_configuration()
    assert protocol['new_2d_rows'] == 476 and protocol['original_image_native_algorithm_calls'] == 238
    verify(protocol['parent_protocol'])
    for entry in protocol['source_bindings']:
        verify(entry)
    if RECEIPT.exists():
        previous = C.read(RECEIPT)
        assert previous['complete'] and previous['square_protocol_sha256'] == expected_protocol_sha256
        assert C.sha(ROW_PATH) == previous['raw_rows_sha256']
        return previous
    assert not ROW_PATH.exists() and not ROW_PATH.with_name(ROW_PATH.name + '.pending').exists(), 'Incomplete square execution must not be silently retried'
    records, truth, predictions, saved = _load()
    assert [r['id'] for r in records] == protocol['full_ids']
    baseline_rows = {name: [] for name in BASELINES}
    for name in BASELINES:
        for record in records:
            fid = record['id']; points = _points(_selected(predictions[name][fid]))
            baseline_rows[name].append(dict(id=fid, session=record['session'],
                corners={mode: saved[name][mode][fid] for mode in MODES},
                canonical_observed={mode: _canonical_observed(points, truth[fid], mode, saved[name][mode][fid]) for mode in MODES}))
    started = time.perf_counter()
    counts = dict(image_decodes=0, native_algorithm_calls=0, native_SUBPIX_calls=0, native_CVRANK_calls=0,
                  cap_pure_calls=0, new_2d_scored_rows=0, new_2d_mode_measure_calls=0,
                  baseline_metric_calls=0, detector_calls=0, refiner_calls=0, final_F_calls=0,
                  solvePnP_calls=0, optimizer_updates=0)
    temporary = ROW_PATH.with_name(ROW_PATH.name + '.pending')
    ROW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=6) as output:
        for record in records:
            fid = record['id']; raw = predictions['BASE'][fid]; chosen = _selected(raw)
            initial = _points(chosen); support = _support(initial)
            image = cv2.imread(str(C.ROOT / record['image']['path']), cv2.IMREAD_COLOR)
            assert image is not None and list(image.shape[:2]) == record['original_hw'], fid
            counts['image_decodes'] += 1
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            target = truth[fid]
            for algorithm in A.METHODS:
                native, diagnostics = A.correct(gray, initial, support, algorithm)
                counts['native_algorithm_calls'] += 1
                counts[f'native_{algorithm}_calls'] += 1
                capped = A.cap_points(initial, native, image.shape[1], image.shape[0], support)
                counts['cap_pure_calls'] += 1
                for suffix, corrected in (('NATIVE', native), ('CAP1', capped)):
                    arm = algorithm + '_' + suffix
                    assert np.array_equal(corrected[8], initial[8], equal_nan=True)
                    assert np.array_equal(corrected[~support], initial[~support], equal_nan=True)
                    corners = {}
                    for mode in MODES:
                        base_metric = saved['BASE'][mode][fid]
                        metric = M.measure(corrected, target['gt'], target['valid'][mode],
                                           target['permutations'], target['hw'],
                                           matched=base_metric['matched'], detected=base_metric['detected'])
                        metric.update(id=fid, session=record['session'])
                        corners[mode] = metric
                        counts['new_2d_mode_measure_calls'] += 1
                    displacement = np.linalg.norm(corrected[:8] - initial[:8], axis=-1)
                    row = dict(id=fid, session=record['session'], method=arm, dataset='GREEN0918_119',
                               image_path=record['image']['path'], image_sha256=record['image']['sha256'],
                               raw_hw=raw['raw_hw'], initial_points=initial, corrected_points=corrected,
                               native_points=native, prediction_support=support, selected_index=raw['selected_index'],
                               candidate_count=len(raw['candidates']), fixed_detector_metadata=chosen,
                               corners=corners,
                               canonical_observed={mode: _canonical_observed(corrected, target, mode, corners[mode]) for mode in MODES},
                               correction=dict(native_algorithm=algorithm, cap_applied=suffix == 'CAP1',
                                               cap_px=.01 * np.hypot(*record['original_hw']),
                                               diagnostics=diagnostics, displacement_px8=displacement,
                                               corner_unchanged8=np.all((corrected[:8] == initial[:8]) |
                                                                       (np.isnan(corrected[:8]) & np.isnan(initial[:8])), axis=-1)),
                               new_F_calls=0, pose=None)
                    output.write(json.dumps(C.finite(row), ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                    counts['new_2d_scored_rows'] += 1
    assert counts['native_algorithm_calls'] == 238 and counts['new_2d_scored_rows'] == 476
    assert counts['new_2d_mode_measure_calls'] == 952 and counts['cap_pure_calls'] == 238
    temporary.replace(ROW_PATH)
    # Recheck all source bytes after CPU correction; these are small image/reference files only.
    for entry in protocol['source_bindings']:
        verify(entry)
    result = dict(schema='pallet_training_free_square119_predictions_v1', complete=True, status='DONE_2D_ONLY',
                  square_protocol_file=str(protocol_path.relative_to(C.ROOT)), square_protocol_sha256=expected_protocol_sha256,
                  raw_rows_file=str(ROW_PATH.relative_to(C.ROOT)), raw_rows_sha256=C.sha(ROW_PATH),
                  raw_rows_bytes=ROW_PATH.stat().st_size, full_denominator_per_arm=119, rows=476,
                  methods=list(C.ARMS), baselines=baseline_rows, reference=protocol['reference'],
                  direct_input_verified=True, source_bindings=protocol['source_bindings'],
                  input_preservation_after_actual_sha='PASS', counts=counts, new_F_calls=0,
                  baseline_metric_calls=0, square_reselection_or_tuning=False,
                  dataset_status='reused development 2D audit; one correlated capture session',
                  confidence_intervals=0, pose_3d=None, posefix_status='BLOCKED_2D_PoseFix',
                  seconds_accuracy_cpu=time.perf_counter() - started,
                  timing_scope='cached-RAW CPU accuracy stage; not end-to-end runtime')
    C.write(RECEIPT, result)
    return C.finite(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'execute'), required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--expected-protocol-sha256')
    args = parser.parse_args()
    if args.stage == 'prepare':
        assert not args.protocol.exists(), 'Preserve an already sealed square protocol'
        value = build_protocol(); C.write(args.protocol, value)
        print(json.dumps({'status': value['status'], 'frames': value['frames'],
                          'protocol_sha256': C.sha(args.protocol), 'new_2d_rows': value['new_2d_rows']}))
    else:
        value = execute(args.protocol, args.expected_protocol_sha256)
        print(json.dumps({k: value[k] for k in ('status', 'rows', 'counts', 'seconds_accuracy_cpu', 'raw_rows_sha256')}))


if __name__ == '__main__':
    main()
