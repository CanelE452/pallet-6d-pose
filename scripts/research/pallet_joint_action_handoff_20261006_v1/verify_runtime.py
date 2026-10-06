"""Offline runtime output parity and fixed-population/summary verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import DOC, binding, sha, write


def verify(source, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    raw_path = receipt_path.parent / receipt['raw']['path']
    assert sha(raw_path) == receipt['raw']['sha256']
    for entry in receipt['bindings']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    raw = json.loads(raw_path.read_text())
    n3_path = source / 'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed1.json'
    baseline_path = source / 'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'
    posefix_path = source / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed1_REAL_DEV.npz'
    selected_path = source / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
    n3 = {r['id']: r for r in json.loads(n3_path.read_text())['records']}
    baseline = json.loads(baseline_path.read_text())['frames']
    selected = {r['frame_id']: r for r in json.loads(selected_path.read_text())['selected']}
    with np.load(posefix_path, allow_pickle=False) as archive:
        prior = {str(fid): q for fid, q in zip(archive['ids'], archive['points'][:, 0, 1])}
    errors = {'Base': [], 'N3_seed1': [], 'PoseFix_seed1': []}
    for row in raw['warmup'] + raw['measurements']:
        fid, name = row['frame_id'], row['path']
        if name == 'PoseFix_seed1':
            expected = prior[fid]
        else:
            if name == 'N3_seed1':
                frame = n3[fid]
                ix = frame['selected_index']
                candidates = frame['candidates']
            else:
                candidates = baseline[selected[fid]['image_key']]
                ix = int(np.argmax([c['score'] for c in candidates])) if candidates else None
            expected = None if ix is None else candidates[ix]['keypoints_xy']
        actual = row['output']['points']
        assert (actual is None) == (expected is None), (fid, name, 'no detection')
        if actual is not None:
            actual, expected = np.asarray(actual, float), np.asarray(expected, float)
            assert actual.shape == expected.shape == (9, 2)
            assert np.array_equal(np.isnan(actual), np.isnan(expected))
            error = float(np.nanmax(np.abs(actual - expected)))
            errors[name].append(error)
            assert error <= 1e-3, (fid, name, error)
        assert row['wall_ms'] > 0
    summary_checks = 0
    for name in errors:
        warm = [r for r in raw['warmup'] if r['path'] == name]
        measured = [r for r in raw['measurements'] if r['path'] == name]
        assert len(warm) == 20 and len(measured) == 130
        assert len({r['frame_id'] for r in measured}) == 26
        assert len({r['session'] for r in measured}) == 13
        assert all(len([r for r in measured if r['repeat'] == rep]) == 26 for rep in range(5))
        values = [r['wall_ms'] for r in measured]
        assert receipt['summary'][name]['median_ms'] == float(np.median(values))
        assert receipt['summary'][name]['p90_ms'] == float(np.quantile(values, .9))
        assert receipt['summary'][name]['pose_success'] == sum(r['output']['pose_available'] for r in measured)
        summary_checks += 8
    result = {'schema': 'joint_action_runtime_verification_v1', 'status': 'PASS',
              'calls_verified': 450, 'summary_checks': summary_checks,
              'point_tolerance_px': 1e-3, 'max_abs_output_difference_px': {k: max(v) for k, v in errors.items()},
              'reference_bindings': [binding(p) for p in (n3_path, baseline_path, posefix_path, selected_path)],
              'scope': 'model identity and existing output parity outside timer, not new accuracy evaluation'}
    write(receipt_path.parent / 'RUNTIME_VERIFICATION.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--receipt', type=Path, default=DOC / 'RUNTIME_MATCHED.json')
    args = parser.parse_args()
    print(json.dumps(verify(args.source_root, args.receipt), indent=2))


if __name__ == '__main__':
    main()
