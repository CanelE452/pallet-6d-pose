"""CAL exploratory combination of two sealed UV arms and fixed CNN states."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = HERE.parent / 'cal_boundary_grid_v1'
sys.path.insert(0, str(GRID))
from grid_runner import sha, read, write, iter_rows, predicted_pose, distribution
from code_snapshot import common as C
from code_snapshot.curriculum import solve_curriculum
from code_snapshot.run import no_annotations
from code_snapshot.scoring import pose_metric


def infer():
    p = read(HERE / 'PROTOCOL.json')
    assert not (HERE / 'PREDICTIONS.jsonl.gz').exists()
    assert sha(p['coordinate_predictions']) == p['coordinate_predictions_sha256']
    assert sha(p['appearance_predictions']) == p['appearance_predictions_sha256']
    assert sha(p['prediction_input']) == p['prediction_input_sha256']
    for name, digest in p['code_sha256'].items():
        assert sha(Path(p['code_snapshot']) / name) == digest
    frames = read(p['prediction_input'])['frames']
    assert len(frames) == 76 and all(f['partition'] == 'CAL' for f in frames)
    original = {(r['id'], r['method']): r for r in iter_rows(p['appearance_predictions'])}
    selected = {a['coordinate_arm'] for a in p['arms']}
    coords = {(r['id'], r['arm']): r for r in iter_rows(p['coordinate_predictions']) if r['arm'] in selected}
    write(HERE / 'INFERENCE_CODE_BINDING.json', dict(runner_sha256=sha(__file__),
        protocol_sha256=sha(HERE / 'PROTOCOL.json'), before_inference_and_scoring=True))
    rows = []
    with no_annotations():
        for frame in frames:
            ident = frame['id']; initial = frame['initial_poses']['N3_SUBPIX']
            baseline = original[ident, 'SEGMENTED_RGB_CURRICULUM']
            appearance = baseline['appearance']
            hidden = np.asarray(baseline['self_hidden'], bool)
            q = np.asarray(baseline['original_points'], float)
            xyz, K = C.cuboid(*initial['cf_extents']), np.asarray(frame['K'])
            R, t = np.asarray(initial['R_cf']), np.asarray(initial['centroid'])
            assert np.array_equal(hidden, C.self_hidden(xyz, R, t))
            for arm in p['arms']:
                source = coords[ident, arm['coordinate_arm']]
                corrected = np.asarray(source['corrected_points'], float)
                assert np.array_equal(source['initial_self_hidden'], hidden)
                assert np.array_equal(corrected[hidden], q[hidden])
                assert np.array_equal(q, np.asarray(source['original_points']))
                corrected_ids = source['refined_ids']
                solution = solve_curriculum(corrected, xyz, K, R, t, hidden, appearance, p['fixed_curriculum']) if corrected_ids else baseline['solution']
                rows.append(dict(id=ident, session=frame['session'], arm=arm['name'],
                    coordinate_arm=arm['coordinate_arm'], coordinate_seal=p['coordinate_prediction_seal_sha256'],
                    appearance_method='SEGMENTED_RGB_CURRICULUM', initial_self_hidden=hidden.tolist(),
                    refined_ids=corrected_ids, corrected_points=corrected.tolist(),
                    shared_edge_correlations=source['correction']['diagnostics']['shared_edge_correlations'],
                    refined_corner_incident_edges={str(k):source['correction']['corner_records'][k]['incident_shared_edges'] for k in corrected_ids},
                    solution=solution, actual_pose=predicted_pose(initial, solution)))
    with gzip.open(HERE / 'PREDICTIONS.jsonl.gz', 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(C.finite(row), separators=(',', ':'), allow_nan=False) + '\n')
    write(HERE / 'PREDICTION_SEAL.json', dict(complete=True, before_scoring=True, GT_read_canary='PASS',
        TEST_accessed=False, rows=len(rows), sha256=sha(HERE / 'PREDICTIONS.jsonl.gz'),
        protocol_sha256=sha(HERE / 'PROTOCOL.json'), runner_sha256=sha(__file__)))
    print('HYBRID_SEALED', len(rows), flush=True)


def score():
    p, seal = read(HERE / 'PROTOCOL.json'), read(HERE / 'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['GT_read_canary'] == 'PASS'
    assert seal['sha256'] == sha(HERE / 'PREDICTIONS.jsonl.gz')
    assert seal['protocol_sha256'] == sha(HERE / 'PROTOCOL.json')
    oracle = {f['id']: f for f in read(HERE.parent / 'ORACLE_CAL_INPUTS.json')['frames']}
    assert len(oracle) == 76 and all(f['partition'] == 'CAL' for f in oracle.values())
    baseline = {(r['id'], r['method']): r for r in iter_rows(p['appearance_predictions'])}
    values = list(iter_rows(HERE / 'PREDICTIONS.jsonl.gz'))
    groups = collections.defaultdict(list)
    sessions = sorted({r['session'] for r in values}); rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(sessions), (10000, len(sessions)))
    for row in values:
        truth = oracle[row['id']]['target']['truth']
        row['metric'] = pose_metric(row['actual_pose'], truth)
        row['baseline_metric'] = pose_metric(baseline[row['id'], 'SEGMENTED_RGB_CURRICULUM']['actual_pose'], truth)
        row['frozen_initial_metric'] = pose_metric(baseline[row['id'], 'INITIAL']['actual_pose'], truth)
        groups[row['arm']].append(row)
    summaries = {}
    for arm, rows in groups.items():
        summary = dict(frames=len(rows), accepted=sum(r['solution']['accepted'] for r in rows),
            corrected_frames=sum(bool(r['refined_ids']) for r in rows), corrected_corners=sum(len(r['refined_ids']) for r in rows),
            metrics={k:distribution([r['metric'][k] for r in rows]) for k in ('translation_cm','rotation_deg','ADDsym_cm')},
            success_5cm_5deg=float(np.mean([r['metric']['success_5cm_5deg'] for r in rows])), paired={})
        for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm', 'success_5cm_5deg'):
            delta = np.asarray([float(r['metric'][key])-float(r['baseline_metric'][key]) for r in rows])
            sums = [sum(d for d,r in zip(delta,rows) if r['session']==s) for s in sessions]
            counts = [sum(r['session']==s for r in rows) for s in sessions]
            boot = np.asarray(sums)[draws].sum(1)/np.asarray(counts)[draws].sum(1)
            summary['paired'][key] = dict(mean_difference=float(delta.mean()), CI95=np.quantile(boot,[.025,.975]).tolist(),
                improved=int((delta>0).sum()) if key.startswith('success') else int((delta<0).sum()),
                worsened=int((delta<0).sum()) if key.startswith('success') else int((delta>0).sum()))
        summaries[arm] = summary
    with gzip.open(HERE / 'SCORED_ROWS.jsonl.gz', 'wt') as stream:
        for row in values:
            stream.write(json.dumps(C.finite(row), separators=(',', ':')) + '\n')
    write(HERE / 'RESULTS.json', dict(schema='CAL_BOUNDARY_HYBRID_V1', CAL_only=True, TEST_accessed=False,
        exploratory_post_grid_selection=True, predictions_sha256=seal['sha256'], summary=summaries,
        baseline='cached SEGMENTED_RGB_CURRICULUM', sessions=sessions, bootstrap_repeats=10000,
        limitations=p['limitations']))
    for arm, summary in summaries.items():
        print(arm, 'accepted',summary['accepted'],'T',summary['metrics']['translation_cm'],
              'R',summary['metrics']['rotation_deg'],'success',summary['success_5cm_5deg'],
              'PAIRED',summary['paired'], flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['infer','score']);args=parser.parse_args()
    infer() if args.command=='infer' else score()
