"""Infer and seal new RGB decisions before opening evaluation annotations."""
import argparse
from collections import Counter
import contextlib
import time
import cv2
import numpy as np

from . import common as C


@contextlib.contextmanager
def no_annotations():
    import builtins
    import io
    from pathlib import Path
    originals = (builtins.open, io.open, Path.open)
    blocked = ('TARGETS.json', 'STATIC_VISIBILITY_MERGE_AUDIT', 'GEOMETRY_RESOLVED_POSE_GT')
    def wrap(function):
        def guarded(path, *args, **kwargs):
            if any(name in str(path) for name in blocked):
                raise AssertionError('Inference attempted evaluation annotation read: ' + str(path))
            return function(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, Path.open = map(wrap, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


def simple_pose(points, xyz, K, initial, mask):
    ids = np.flatnonzero(mask & np.isfinite(points).all(1) & ~(points == -1).all(1))
    if len(ids) < 4:
        return dict(R=initial['R_cf'], t=initial['centroid'], accepted=False, used_ids=ids.tolist(), reason='insufficient_observations')
    try:
        ok, rvec, tvec = cv2.solvePnP(xyz[ids], points[ids], K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise ValueError('solvePnP returned false')
        rvec, tvec = cv2.solvePnPRefineLM(xyz[ids], points[ids], K, None, rvec, tvec)
        R, t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        if not np.isfinite(t).all() or np.any((xyz @ R.T + t)[:, 2] <= 0):
            raise ValueError('invalid_camera_depth')
        return dict(R=R.tolist(), t=t.tolist(), accepted=True, used_ids=ids.tolist(), reason='standard_PnP')
    except (cv2.error, ValueError) as error:
        return dict(R=initial['R_cf'], t=initial['centroid'], accepted=False, used_ids=[], reason=str(error))


def infer(args):
    from .appearance import infer_appearance
    from .curriculum import solve_curriculum
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    contract = C.read(C.DOC / 'SPLIT.json')
    settings = C.read(args.config) if args.config else {}
    # The session list, not a metric, defines this run's population.
    partition = args.partition
    split_key = {'calibration': 'CAL', 'test': 'TEST'}
    ids = set(contract['real_split']['frame_ids'][split_key[partition]]) if partition != 'development' else set(sum(contract['real_split']['frame_ids'].values(), []))
    frames = [frame for frame in C.prediction_inputs() if frame['id'] in ids]
    assert len(frames) == len(ids)
    files = [C.ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1' / name
             for name in ('appearance.py', 'curriculum.py', 'common.py', 'run.py')]
    C.write(output / 'PROTOCOL.json', dict(partition=partition, ids=sorted(ids), settings=settings,
        code_sha256={path.name: C.sha(path) for path in files},
        contract_sha256=C.sha(C.DOC / 'DATA_CONTRACT.json'),
        prediction_inputs_sha256=C.sha(C.DOC / 'PREDICTION_INPUTS.json'),
        inference_inputs='current RGB, cached predicted points, camera K, registered dimensions, predicted initial pose only',
        external_visibility_source='RGB appearance, independent of annotation',
        reprojected_points_used_for_fit=False, new_detector_forward=0,
        coordinate_cache_reused=True, current_final_population_accessed=False))
    cv2.setNumThreads(1)
    results = []
    start = time.monotonic()
    with no_annotations():
        for index, frame in enumerate(frames):
            image_path = C.ROOT / frame['image']
            if C.sha(image_path) != frame['image_sha256']:
                raise ValueError('Image binding changed: ' + frame['id'])
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
            for arm in args.arms:
                q = np.asarray(frame['points'][arm], float)[:8]
                initial = frame['initial_poses'][arm]
                if not initial or not initial['available']:
                    raise ValueError('Initial pose unavailable: ' + frame['id'])
                xyz = C.cuboid(*initial['cf_extents'])
                K = np.asarray(frame['K'], float)
                R, t = np.asarray(initial['R_cf']), np.asarray(initial['centroid'])
                hidden = C.self_hidden(xyz, R, t)
                tick = time.monotonic()
                appearance = infer_appearance(image, q, hidden, settings.get('appearance'))
                appearance_seconds = time.monotonic() - tick
                h, w = image.shape[:2]
                inside = (q[:,0] >= 0) & (q[:,0] < w) & (q[:,1] >= 0) & (q[:,1] < h)
                methods = {
                    'INITIAL': dict(R=R.tolist(), t=t.tolist(), accepted=False, used_ids=list(range(8)), reason='frozen_initial'),
                    'ALL_STANDARD': simple_pose(q, xyz, K, initial, np.ones(8, bool)),
                    'SELF_STANDARD': simple_pose(q, xyz, K, initial, ~hidden & inside),
                }
                tick = time.monotonic()
                methods['RGB_CURRICULUM'] = solve_curriculum(q, xyz, K, R, t, hidden, appearance, settings.get('curriculum'))
                curriculum_seconds = time.monotonic() - tick
                # Same RGB scores and optimizer, with easy and unknown stages disabled.
                config = dict(settings.get('curriculum') or {})
                config.update(easy_visibility=1.01, enable_unknown=False)
                methods['RGB_ONE_STAGE'] = solve_curriculum(q, xyz, K, R, t, hidden, appearance, config)
                Q = R.T @ np.asarray(initial['R_physical'])
                for name, solution in methods.items():
                    predicted = dict(available=True, R_cf=solution['R'],
                        R_physical=(np.asarray(solution['R']) @ Q).tolist(),
                        centroid=solution['t'], cf_extents=initial['cf_extents'],
                        selected_hypothesis=initial['selected_hypothesis'])
                    results.append(dict(id=frame['id'], session=frame['session'], arm=arm, method=name,
                        actual_pose=predicted, solution=solution, self_hidden=hidden.tolist(),
                        appearance=appearance, original_points=q.tolist(),
                        projected_points=C.project(xyz, solution['R'], solution['t'], K).tolist(),
                        appearance_seconds=appearance_seconds, curriculum_seconds=curriculum_seconds))
            if (index + 1) % 10 == 0:
                print('INFER', index + 1, '/', len(frames), round(time.monotonic() - start, 1), 'seconds', flush=True)
    C.save_rows(output / 'PREDICTIONS.jsonl.gz', results)
    C.write(output / 'PREDICTION_SEAL.json', dict(complete=True, before_scoring=True, rows=len(results),
        sha256=C.sha(output / 'PREDICTIONS.jsonl.gz'), seconds=time.monotonic()-start,
        methods=dict(Counter(row['method'] for row in results)), GT_read_canary='PASS'))
    print('SEALED', output, len(results), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['infer', 'score'])
    parser.add_argument('--output', type=__import__('pathlib').Path, required=True)
    parser.add_argument('--partition', choices=['calibration', 'test', 'development'], default='calibration')
    parser.add_argument('--config', type=__import__('pathlib').Path)
    parser.add_argument('--arms', nargs='+', choices=['BASE', 'N3_SUBPIX'], default=['N3_SUBPIX'])
    args = parser.parse_args()
    if args.command == 'infer':
        infer(args)
    else:
        from .scoring import score
        score(args.output)


if __name__ == '__main__':
    main()
