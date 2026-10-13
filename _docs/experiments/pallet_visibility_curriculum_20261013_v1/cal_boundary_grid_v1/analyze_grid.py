"""Post-seal CAL uncertainty and visual QA; never changes predictions."""
import collections
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from grid_runner import read, iter_rows, pose_metric, write

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent


def run():
    result = read(HERE / 'RESULTS.json')
    protocol = read(HERE / 'PROTOCOL.json')
    frames = read(protocol['prediction_input'])
    source = Path(frames['source_root'])
    frames = {f['id']: f for f in frames['frames']}
    oracle = {f['id']: f for f in read(SCRATCH / 'ORACLE_CAL_INPUTS.json')['frames']}
    predictions = list(iter_rows(HERE / 'PREDICTIONS.jsonl.gz'))
    by_key = {(r['id'], r['arm']): r for r in predictions}
    corners = list(iter_rows(HERE / 'CHANGED_CORNER_SCORES.jsonl.gz'))
    baseline = {(r['id'], r['method']): r for r in iter_rows(protocol['baseline_predictions'])}
    sessions = sorted({f['session'] for f in frames.values()})
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(sessions), (10000, len(sessions)))
    comparisons = {}
    for arm in result['summary']:
        coordinate_sums, coordinate_counts = [], []
        pose_sums, pose_counts = [], []
        for session in sessions:
            delta = []; pose_delta = []
            for ident, f in frames.items():
                if f['session'] != session:
                    continue
                row = by_key[ident, arm]; meta = oracle[ident]['scoring_metadata']
                truth = np.asarray(oracle[ident]['target']['gt'], float)[:8][meta['native_to_reference_corner_ids']]
                selected = np.array([s == 'DIRECT_VISIBLE' for s in meta['human_states_native']])
                q, p = np.asarray(row['original_points']), np.asarray(row['corrected_points'])
                delta.extend((np.linalg.norm(p - truth, axis=1) - np.linalg.norm(q - truth, axis=1))[selected])
                new_metric = pose_metric(row['curriculum_pose'], oracle[ident]['target']['truth'])
                old_metric = pose_metric(baseline[ident, 'RGB_CURRICULUM']['actual_pose'], oracle[ident]['target']['truth'])
                pose_delta.append([new_metric[key] - old_metric[key] for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')])
            coordinate_sums.append(float(np.sum(delta))); coordinate_counts.append(len(delta))
            pose_sums.append(np.sum(pose_delta, axis=0)); pose_counts.append(len(pose_delta))
        boot = np.asarray(coordinate_sums)[draws].sum(1) / np.asarray(coordinate_counts)[draws].sum(1)
        poseboot = np.asarray(pose_sums)[draws].sum(1) / np.asarray(pose_counts)[draws].sum(1)[:, None]
        comparisons[arm] = dict(pooled_visible_mean_delta_px=result['summary'][arm]['visible_mean_difference_px'],
            clustered_session_CI95_px=np.quantile(boot, [.025, .975]).tolist(),
            pose_vs_RGB_CURRICULUM_CI95=dict(zip(('translation_cm', 'rotation_deg', 'ADDsym_cm'),
                                                np.quantile(poseboot, [.025, .975], axis=0).T.tolist())))
    write(HERE / 'SESSION_BOOTSTRAP.json', dict(repeats=10000, sessions=sessions,
        cluster_unit='session', CAL_selection_bias=True, multiple_grid_selection_not_corrected=True,
        comparisons=comparisons))
    strict = [r for r in corners if r['arm'] == 'RGB_BINARY_strict_r12_s12']
    worst = sorted([r for r in corners if r['arm'] == 'RGB_BINARY_tolerant_r20_s12'],
                   key=lambda r: r['corrected_error_px'] - r['original_error_px'], reverse=True)[:2]
    soft = [r for r in corners if r['arm'] == 'CNN_SOFT_tolerant_r12_s12']
    chosen = strict + worst + soft
    canvas = np.full((len(chosen) * 390, 860, 3), 245, np.uint8)
    for index, corner in enumerate(chosen):
        ident, k, arm = corner['id'], corner['corner'], corner['arm']
        row = by_key[ident, arm]; frame = frames[ident]; meta = oracle[ident]['scoring_metadata']
        image = cv2.imread(str(source / frame['image']))
        q, p = np.asarray(row['original_points']), np.asarray(row['corrected_points'])
        truth = np.asarray(oracle[ident]['target']['gt'], float)[:8][meta['native_to_reference_corner_ids']]
        record = row['correction']['corner_records'][k]
        for edge_index in record['incident_shared_edges']:
            edge = row['correction']['edge_records'][edge_index]
            for x, y in edge['observed_points']:
                cv2.circle(image, tuple(np.rint([x, y]).astype(int)), 1, (0, 255, 255), -1)
        cv2.circle(image, tuple(np.rint(q[k]).astype(int)), 3, (255, 255, 0), 1)
        cv2.circle(image, tuple(np.rint(p[k]).astype(int)), 3, (0, 255, 0), 1)
        cv2.drawMarker(image, tuple(np.rint(truth[k]).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
        tile = np.full((390, 860, 3), 245, np.uint8)
        tile[45:380, :520] = cv2.resize(image, (520, 335))
        x, y = np.rint(q[k]).astype(int); h, w = image.shape[:2]; r = 45
        crop = image[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
        tile[50:370, 530:850] = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)
        title = f'{ident} KP{k} {corner["original_error_px"]:.2f}->{corner["corrected_error_px"]:.2f}px'
        cv2.putText(tile, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, arm + ' cyan=seed green=refined red=annotation', (4, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[index*390:(index+1)*390] = tile
    cv2.imwrite(str(HERE / 'CAL_VISUAL_QA.png'), canvas)
    write(HERE / 'CAL_VISUAL_QA.json', chosen)
    for arm in ('RGB_BINARY_strict_r12_s12', 'RGB_BINARY_tolerant_r20_s12', 'CNN_BINARY_tolerant_r12_s6'):
        print(arm, comparisons[arm])


if __name__ == '__main__':
    run()
