"""Separate renderer-correspondence sanity from production sign ambiguity."""
from collections import Counter
from pathlib import Path
import time

import numpy as np
from . import pose_oracle as O


def main():
    start = time.perf_counter(), time.process_time()
    output_path = O.RAW / 'SYNTHETIC_SANITY_DETAIL.json'
    if output_path.exists():
        print('SYNTHETIC_SANITY_DETAIL_ALREADY_FROZEN'); return
    prior = O.read(O.RAW / 'GEOMETRY_SANITY_PRIVATE.json')
    metrics = {r['id']: r for r in prior['metrics']['SYNTHETIC_EXACT_PROJECTION']}
    table_path = O.ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    table = dict(np.load(table_path)); index = {str(s): i for i, s in enumerate(table['stems'])}
    _, truth = O.D.Pose.metadata('SYNTH_HELDOUT')
    rows = []
    for fid in prior['synthetic_ids']:
        i = index[fid]; fx, fy, cx, cy = table['K'][i]; pad = table['pad'][i]
        K = np.array([[fx, 0., cx-pad], [0., fy, cy-pad], [0., 0., 1.]])
        X, R, t = table['Xcf'][i], table['R'][i], table['t'][i]
        q = O.project(X, R, t, K)
        solved = O.D.Pose.solve(X, q, K, np.ones(8, dtype=bool))
        assert solved is not None
        fitted_R, fitted_t, residual = solved
        projected = O.project(X, fitted_R, fitted_t, K)
        known_corner_error = float(np.mean(np.linalg.norm(X@fitted_R.T+fitted_t-(X@R.T+t), axis=1))/np.linalg.norm(table['dims'][i]))
        rows.append(dict(id=fid, symmetry_order=truth[fid]['order'], production=metrics[fid],
                         known_renderer_correspondence=dict(normalized_camera_corner_error=known_corner_error,
                                                             reprojection_max_px=float(np.linalg.norm(projected-q, axis=1).max()),
                                                             reprojection_mean_px=float(residual))))
    production_errors = [r['production']['ADDsym_normalized'] for r in rows]
    reference_errors = [r['known_renderer_correspondence']['normalized_camera_corner_error'] for r in rows]
    # Fixed numerical diagnostic boundary, never a success criterion for DEV methods.
    numerical_nonzero = [r for r in rows if r['production']['ADDsym_normalized'] > 1e-5]
    summary = dict(frames=len(rows), symmetry_order_counts=dict(Counter(r['symmetry_order'] for r in rows)),
                   production_ADDsym_AUC=O.D.Pose.pose_auc(production_errors, 1.),
                   renderer_correspondence_AUC=O.D.Pose.pose_auc(reference_errors, 1.),
                   renderer_correspondence_max_normalized_camera_corner_error=max(reference_errors),
                   renderer_correspondence_max_reprojection_px=max(r['known_renderer_correspondence']['reprojection_max_px'] for r in rows),
                   production_nonzero_over_1e_5=len(numerical_nonzero),
                   production_nonzero_symmetry_orders=dict(Counter(r['symmetry_order'] for r in numerical_nonzero)),
                   production_nonzero_rotation_deg=[r['production']['rotation_deg'] for r in numerical_nonzero],
                   production_nonzero_yaw_deg=[r['production']['yaw_deg'] for r in numerical_nonzero],
                   production_nonzero_translation_cm=[r['production']['translation_cm'] for r in numerical_nonzero])
    O.save(output_path, dict(is_oracle=True, GT_DEPENDENT=True, DIAGNOSTIC_ONLY=True, rows=rows, summary=summary,
                            timing=O.elapsed(start), source=O.bind(Path(__file__)), geometry=O.bind(table_path)))
    O.save(O.DOC / 'SYNTHETIC_SANITY_DETAIL.json', dict(is_oracle=True, GT_DEPENDENT=True, DIAGNOSTIC_ONLY=True,
           summary=summary, timing=O.elapsed(start), source=O.bind(Path(__file__)), geometry=O.bind(table_path),
           interpretation='Known renderer Xcf directly gives corner-to-object correspondence beyond production camera-facing parity. This privileged solver check is separate from production exact-point sanity. No deployment performance or general upper bound.',
           axis_metric_caveat='Do not interpret inherited synthetic axis_accuracy: original reference body_xyz uses physical dimensions, whereas production cf_extents can validly swap W/D.',
           residual9_metadata_erratum='The frozen D9_RESIDUAL9_CUES.json phrase 9-point solve means corner0..7 SQPnP/LM followed by 9-point projection residual; code and numeric residuals use the unchanged correct implementation.'))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
