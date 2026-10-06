"""Check deployed J outputs against locked A evaluation, outside the timer."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import cv2
from .runtime import DOC, binding, sha, write


def verify(source, receipt_path):
    from .geometry import build_bank, permute_bank
    receipt = json.loads(receipt_path.read_text())
    for item in receipt['bindings']:
        assert sha(item['path']) == item['sha256'], item['path']
    raw_path = receipt_path.parent / receipt['raw']['path']
    assert sha(raw_path) == receipt['raw']['sha256']
    raw = json.loads(raw_path.read_text())
    files = {'Frozen_J_seed1': DOC / 'results/A_REAL_DEV_FROZEN_N3_seed1.json',
             'GEO_seed1': DOC / 'results/A_REAL_DEV_FIT_GEO_seed1.json',
             'PERM_seed1': DOC / 'results/A_REAL_DEV_FIT_PERM_seed1.json'}
    cached = {}
    for name, path in files.items():
        packet = json.loads(path.read_text())
        assert packet['checkpoint_sha256'] == receipt['checkpoint_bindings'][name]['sha256'], (name, 'checkpoint identity')
        key = 'PERM_J' if name == 'PERM_seed1' else 'GEO_J'
        cached[name] = {r['id']: r for r in packet['rows'][key]}
    n3_path = source / 'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed1.json'
    n3 = {r['id']: r for r in json.loads(n3_path.read_text())['records']}
    axis_path = source / 'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json'
    # The axis/cache filename uses session__stamp; prediction/runtime IDs use
    # session:stamp. Bind through the reviewed axis entry, then assert capture ID.
    axis = {r['session_id'] + ':' + Path(r['image']).stem: r
            for r in json.loads(axis_path.read_text())['frames_list']}
    contexts = {}
    input_paths = [*files.values(), n3_path, axis_path, Path(__file__)]
    for fid in sorted({r['frame_id'] for r in raw['measurements']}):
        path = source / f"data/pallet/results/pallet_dim_conditioned_p_v1/DEV_cache/{axis[fid]['frame_id']}.pt"
        capture = torch.load(path, map_location='cpu', weights_only=False)
        assert capture['id'] == fid, ('cache identity', fid)
        pack = capture['captured']; ix = pack['selected_index']
        q = None if ix is None else np.asarray(pack['candidates'][ix]['keypoints_xy'], float)
        ann_path = source / axis[fid]['annotation']
        intr = json.loads(ann_path.read_text())['camera_data']['intrinsics']
        K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
        valid = np.zeros(9, bool) if q is None else np.isfinite(q).all(-1) & ~(q == -1).all(-1)
        if q is None:
            bank = None
        else:
            try:
                bank = build_bank(q, K, np.asarray(capture['dimensions'])[[0, 2, 1]], capture['raw_hw'], valid)
            except (cv2.error, ValueError, FloatingPointError):
                bank = {'points': q[None], 'hypotheses': ['NoOp'], 'details': [], 'reason': 'INITIALIZATION_FAILED'}
        contexts[fid] = (q, bank, valid)
        input_paths += [path, ann_path]
    errors = {name: [] for name in receipt['summary']}
    for row in raw['warmup'] + raw['measurements']:
        name, fid = row['path'], row['frame_id']
        q, bank, valid = contexts[fid]
        if name in cached:
            ref = cached[name][fid]
            action = ref['selected_index']
            assert row['pose_available'] == ref['pose']['available'], (name, fid, 'F coverage')
            assert row['final_W_D'] == ref['final_hypothesis'], (name, fid, 'final W/D')
            if q is None:
                expected = None
            else:
                deployed = permute_bank(bank, fid) if name == 'PERM_seed1' else bank
                expected = deployed['points'][action]
                assert row['diagnostic']['actions'] == len(deployed['points'])
                assert row['diagnostic'].get('selected_index', 0) == action, (name, fid, 'selected action')
        elif name == 'Base':
            expected = q
        else:
            frame = n3[fid]; ix = frame['selected_index']
            expected = None if ix is None else frame['candidates'][ix]['keypoints_xy']
        actual = row['points']
        assert (actual is None) == (expected is None), (name, fid, 'no detection')
        if actual is not None:
            actual, expected = np.asarray(actual, float), np.asarray(expected, float)
            assert actual.shape == expected.shape == (9, 2)
            assert np.array_equal(np.isnan(actual), np.isnan(expected))
            error = float(np.nanmax(np.abs(actual - expected)))
            errors[name].append(error)
            assert error <= 1e-3, (name, fid, error)
            if name in cached:
                assert np.array_equal(actual[8:], q[8:], equal_nan=True), 'centre changed'
                assert np.array_equal(actual[~valid], q[~valid], equal_nan=True), 'missing point changed'
        assert row['wall_ms'] > 0
    for name in errors:
        warm = [r for r in raw['warmup'] if r['path'] == name]
        rows = [r for r in raw['measurements'] if r['path'] == name]
        assert len(warm) == 20 and len(rows) == 130
        assert len({r['frame_id'] for r in rows}) == 26 and len({r['session'] for r in rows}) == 13
        assert all(len([r for r in rows if r['repeat'] == rep]) == 26 for rep in range(5))
        assert receipt['summary'][name]['median_ms'] == float(np.median([r['wall_ms'] for r in rows]))
        assert receipt['summary'][name]['p90_ms'] == float(np.quantile([r['wall_ms'] for r in rows], .9))
        assert receipt['summary'][name]['pose_success'] == sum(r['pose_available'] for r in rows)
    result = {'schema': 'joint_action_A_runtime_verification_v1', 'status': 'PASS',
              'calls_verified': len(raw['warmup']) + len(raw['measurements']),
              'point_tolerance_px': 1e-3, 'max_abs_output_difference_px': {k: max(v) if v else None for k, v in errors.items()},
              'reference_bindings': [binding(p) for p in dict.fromkeys(input_paths)],
              'scope': 'Every timed J action/final W-D/coverage matches locked A evaluation; Base/N3 points match actual cached predictions; comparison outside timer; no extra model inference or optimizer update'}
    write(receipt_path.parent / 'RUNTIME_A_VERIFICATION.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--receipt', type=Path, default=DOC / 'RUNTIME_A.json')
    args = parser.parse_args()
    print(json.dumps(verify(args.source_root, args.receipt), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
