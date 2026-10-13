"""Preregistered CAL-only independent pixel localization; infer, seal, score."""
import argparse
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from code_snapshot import appearance as A
from code_snapshot import boundary_refine as B
from code_snapshot import common as C
from code_snapshot.curriculum import solve_curriculum
from code_snapshot.run import no_annotations, simple_pose
from code_snapshot.scoring import pose_metric
from code_snapshot.segmentation import segment

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(C.finite(value), indent=2, allow_nan=False) + '\n')


def iter_rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def predicted_pose(initial, solution):
    Q = np.asarray(initial['R_cf']).T @ np.asarray(initial['R_physical'])
    return dict(R_cf=solution['R'], R_physical=(np.asarray(solution['R']) @ Q).tolist(),
                centroid=solution['t'], cf_extents=initial['cf_extents'])


def infer():
    protocol = read(HERE / 'PROTOCOL.json')
    assert not (HERE / 'PREDICTIONS.jsonl.gz').exists()
    assert sha(protocol['prediction_input']) == protocol['prediction_input_sha256']
    assert sha(protocol['baseline_predictions']) == protocol['baseline_predictions_sha256']
    assert sha(protocol['source_CNN_weights']) == protocol['source_CNN_sha256']
    for name, digest in protocol['code_sha256'].items():
        assert sha(HERE / 'code_snapshot' / name) == digest
    frames = read(protocol['prediction_input'])
    source = Path(frames['source_root'])
    frames = frames['frames']
    assert len(frames) == 76 and all(f['partition'] == 'CAL' for f in frames)
    ids = {f['id'] for f in frames}
    baseline = {(r['id'], r['method']): r for r in iter_rows(protocol['baseline_predictions'])}
    assert {key[0] for key in baseline} == ids
    write(HERE / 'INFERENCE_CODE_BINDING.json', dict(runner_sha256=sha(__file__),
          protocol_sha256=sha(HERE / 'PROTOCOL.json'), before_inference=True,
          before_scoring=True, exact_CAL_ids=sorted(ids)))
    cv2.setNumThreads(1)
    cache = HERE / 'mask_cache'
    cache.mkdir()
    masks, records, counts = [], [], collections.Counter()
    start = time.monotonic()
    with no_annotations():
        for index, frame in enumerate(frames):
            ident = frame['id']
            frozen = baseline[ident, 'INITIAL']
            q = np.asarray(frozen['original_points'], float)
            app = frozen['appearance']
            hidden = np.asarray(frozen['self_hidden'], bool)
            initial = frame['initial_poses']['N3_SUBPIX']
            assert np.array_equal(q, np.asarray(frame['points']['N3_SUBPIX'])[:8])
            assert np.allclose(initial['R_cf'], frozen['actual_pose']['R_cf'])
            assert np.allclose(initial['centroid'], frozen['actual_pose']['centroid'])
            xyz, K = C.cuboid(*initial['cf_extents']), np.asarray(frame['K'], float)
            R, t = np.asarray(initial['R_cf']), np.asarray(initial['centroid'])
            assert np.array_equal(hidden, C.self_hidden(xyz, R, t))
            image_path = source / frame['image']
            assert sha(image_path) == frame['image_sha256']
            image = cv2.imread(str(image_path))
            probability = segment(image, protocol['source_CNN_weights'])
            soft = np.rint(probability * 255).astype(np.uint8)
            scene = A._scene(image, q, hidden, np.isfinite(q).all(1) & ~(q == -1).all(1), A._config(None))
            mask_map = dict(RGB_BINARY=scene['material'], CNN_BINARY=(probability >= .5).astype(np.uint8) * 255, CNN_SOFT=soft)
            mask_binding = {}
            for name, mask in mask_map.items():
                path = cache / f'{index:03d}_{name}.png'
                cv2.imwrite(str(path), mask)
                mask_binding[name] = dict(path=path.name, sha256=sha(path), pixels=int(np.count_nonzero(mask)), dtype='uint8')
            masks.append(dict(id=ident, mask_binding=mask_binding))
            pose_cache = {q.tobytes(): (baseline[ident, 'RGB_CURRICULUM']['solution'], baseline[ident, 'ALL_STANDARD']['solution'])}
            for arm in protocol['grid_arms']:
                correction = B.infer_rgb_boundary_points(image, q, hidden, app, arm['config'],
                    material_mask=mask_map[arm['mask']], mask_source=arm['mask'])
                adjusted = np.asarray(correction['points'], float)
                assert np.array_equal(adjusted[hidden], q[hidden])
                key = adjusted.tobytes()
                if key not in pose_cache:
                    pose_cache[key] = (solve_curriculum(adjusted, xyz, K, R, t, hidden, app, protocol['fixed_curriculum']),
                                       simple_pose(adjusted, xyz, K, initial, np.ones(8, bool)))
                curriculum, standard = pose_cache[key]
                records.append(dict(id=ident, session=frame['session'], arm=arm['name'], mask=arm['mask'],
                    original_points=q.tolist(), corrected_points=adjusted.tolist(),
                    initial_self_hidden=hidden.tolist(), refined_ids=correction['observed_refined_ids'],
                    correction=correction, curriculum=curriculum, standard=standard,
                    curriculum_pose=predicted_pose(initial, curriculum), standard_pose=predicted_pose(initial, standard)))
                counts[arm['name'] + ':frames'] += bool(correction['observed_refined_ids'])
                counts[arm['name'] + ':corners'] += len(correction['observed_refined_ids'])
            if (index + 1) % 10 == 0:
                print('GRID_INFER', index + 1, '/', len(frames), round(time.monotonic() - start, 1), 'seconds', flush=True)
    with gzip.open(HERE / 'PREDICTIONS.jsonl.gz', 'wt', compresslevel=5) as stream:
        for row in records:
            stream.write(json.dumps(C.finite(row), separators=(',', ':'), allow_nan=False) + '\n')
    write(HERE / 'MASK_BINDINGS.json', masks)
    write(HERE / 'PREDICTION_SEAL.json', dict(complete=True, before_scoring=True,
        GT_read_canary='PASS', TEST_accessed=False, rows=len(records), frames=len(frames), arms=len(protocol['grid_arms']),
        sha256=sha(HERE / 'PREDICTIONS.jsonl.gz'), protocol_sha256=sha(HERE / 'PROTOCOL.json'),
        runner_sha256=sha(__file__), seconds=time.monotonic() - start, corrections=counts))
    print('SEALED', len(records), 'rows', round(time.monotonic() - start, 1), 'seconds', flush=True)


def distribution(values):
    values = np.asarray(values, float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                P90=float(np.quantile(values, .9)) if len(values) else None)


def score():
    protocol, seal = read(HERE / 'PROTOCOL.json'), read(HERE / 'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['GT_read_canary'] == 'PASS'
    assert seal['sha256'] == sha(HERE / 'PREDICTIONS.jsonl.gz')
    assert seal['protocol_sha256'] == sha(HERE / 'PROTOCOL.json')
    # This is the first coordinate/pose annotation access in this experiment.
    oracle = read(SCRATCH / 'ORACLE_CAL_INPUTS.json')['frames']
    assert len(oracle) == 76 and all(f['partition'] == 'CAL' for f in oracle)
    oracle = {f['id']: f for f in oracle}
    baseline = {(r['id'], r['method']): r for r in iter_rows(protocol['baseline_predictions'])}
    rows = list(iter_rows(HERE / 'PREDICTIONS.jsonl.gz'))
    assert {r['id'] for r in rows} == set(oracle)
    groups, baselines, corner_rows = collections.defaultdict(list), {}, []
    for method in ('INITIAL', 'RGB_CURRICULUM', 'SEGMENTED_RGB_CURRICULUM', 'ALL_STANDARD'):
        values = [pose_metric(baseline[ident, method]['actual_pose'], f['target']['truth']) for ident, f in oracle.items()]
        baselines[method] = dict(frames=76, metrics={key: distribution([m[key] for m in values])
            for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')},
            success=float(np.mean([m['success_5cm_5deg'] for m in values])))
    for row in rows:
        frame = oracle[row['id']]
        mapping = np.asarray(frame['scoring_metadata']['native_to_reference_corner_ids'], int)
        truth = np.asarray(frame['target']['gt'], float)[:8][mapping]
        states = frame['scoring_metadata']['human_states_native']
        q, p = np.asarray(row['original_points']), np.asarray(row['corrected_points'])
        visible = np.array([state == 'DIRECT_VISIBLE' for state in states]) & np.isfinite(truth).all(1) & ~(truth == -1).all(1)
        original_error = np.linalg.norm(q - truth, axis=1)
        corrected_error = np.linalg.norm(p - truth, axis=1)
        row['visible_original_errors'] = original_error[visible].tolist()
        row['visible_corrected_errors'] = corrected_error[visible].tolist()
        row['changed_visible_original_errors'], row['changed_visible_corrected_errors'] = [], []
        row['changed_non_direct'] = 0
        for k in row['refined_ids']:
            corner_rows.append(dict(id=row['id'], arm=row['arm'], corner=k, actual_state=states[k],
                initial_self_hidden=row['initial_self_hidden'][k], original_error_px=float(original_error[k]),
                corrected_error_px=float(corrected_error[k]), shift_px=float(np.linalg.norm(p[k] - q[k]))))
            if visible[k]:
                row['changed_visible_original_errors'].append(float(original_error[k]))
                row['changed_visible_corrected_errors'].append(float(corrected_error[k]))
            else:
                row['changed_non_direct'] += 1
        row['curriculum_metric'] = pose_metric(row['curriculum_pose'], frame['target']['truth'])
        row['standard_metric'] = pose_metric(row['standard_pose'], frame['target']['truth'])
        groups[row['arm']].append(row)
    summaries = {}
    for arm, values in groups.items():
        def concat(name):
            return [item for r in values for item in r[name]]
        before, after = concat('visible_original_errors'), concat('visible_corrected_errors')
        cb, ca = concat('changed_visible_original_errors'), concat('changed_visible_corrected_errors')
        summary = dict(frames=len(values), corrected_frames=sum(bool(r['refined_ids']) for r in values),
            corrected_corners=sum(len(r['refined_ids']) for r in values),
            changed_non_direct=sum(r['changed_non_direct'] for r in values),
            visible_original_error_px=distribution(before), visible_corrected_error_px=distribution(after),
            changed_visible_original_error_px=distribution(cb), changed_visible_corrected_error_px=distribution(ca),
            visible_mean_difference_px=float(np.mean(after) - np.mean(before)),
            changed_visible_improved=int(np.sum(np.asarray(ca) < np.asarray(cb))),
            changed_visible_worsened=int(np.sum(np.asarray(ca) > np.asarray(cb))),
            corner_reasons=dict(collections.Counter(c['reason'] for r in values for c in r['correction']['corner_records'])),
            edge_reasons=dict(collections.Counter(e['reason'] for r in values for e in r['correction']['edge_records'])))
        for name in ('curriculum', 'standard'):
            metric = name + '_metric'
            summary[name] = dict(accepted=sum(r[name]['accepted'] for r in values),
                metrics={key: distribution([r[metric][key] for r in values]) for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')},
                success=float(np.mean([r[metric]['success_5cm_5deg'] for r in values])))
        summaries[arm] = summary
    write(HERE / 'RESULTS.json', dict(schema='CAL_BOUNDARY_GRID_V1', CAL_only=True, TEST_accessed=False,
        frames=76, arms=len(groups), predictions_sha256=seal['sha256'],
        baselines=baselines, summary=summaries,
        limits=protocol['risks'], coordinate_GT='historical completed bbox annotations; DIRECT_VISIBLE only for primary coordinate metric'))
    with gzip.open(HERE / 'CHANGED_CORNER_SCORES.jsonl.gz', 'wt') as stream:
        for row in corner_rows:
            stream.write(json.dumps(row) + '\n')
    print('SCORED', len(rows), 'rows', len(corner_rows), 'changed corners', flush=True)
    for arm, summary in sorted(summaries.items(), key=lambda item: item[1]['visible_mean_difference_px']):
        print(arm, 'corners', summary['corrected_corners'], 'visible_delta', round(summary['visible_mean_difference_px'], 4),
              'T', round(summary['curriculum']['metrics']['translation_cm']['mean'], 4),
              'R', round(summary['curriculum']['metrics']['rotation_deg']['mean'], 4), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['infer', 'score'])
    args = parser.parse_args()
    infer() if args.command == 'infer' else score()
