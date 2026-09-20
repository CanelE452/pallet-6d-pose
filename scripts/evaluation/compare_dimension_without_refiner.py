"""Compare frozen N2 with R0, retaining registered dimensions and the same PnP."""
import json
import numpy as np
import torch
import cv2
from scripts.evaluation import final_dimension_release as F


def main():
    E = F.setup()
    import pose
    from dev_evaluate import population_metadata
    from eval_math import summary
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    meta, gt = pose.metadata('REAL_DEV')
    pe, population = population_metadata()
    baseline_path = E.LINE / 'baseline/FULL_CANDIDATES.json'
    baseline = F.read(baseline_path)['frames']
    inputs = {'R0': {}}
    for item, _ in population:
        candidates = baseline[pe.canonical_key(item.image)]
        selected = int(np.argmax([c['score'] for c in candidates])) if candidates else None
        inputs['R0'][item.frame_id] = None if selected is None else candidates[selected]['keypoints_xy']
    sources = [baseline_path, E.DOC / 'PAPER_POSE_RESULTS.json']
    for seed in (1, 2, 3):
        name = f'N2_DIM_ONLY_seed{seed}'
        path = E.RAW / f'predictions/REAL_DEV/{name}.json'
        sources.append(path)
        inputs[name] = {r['id']: None if r['selected_index'] is None else
                       r['candidates'][r['selected_index']]['keypoints_xy']
                       for r in F.read(path)['records']}
    rows, tables = {}, {}
    previous = F.read(E.DOC / 'PAPER_POSE_RESULTS.json')['summary']['REAL_DEV']
    for name, points in inputs.items():
        assert set(points) == set(meta) == set(gt)
        rr = [pose.metric((fid, pose.infer(p, *meta[fid]), gt[fid])) for fid, p in points.items()]
        rows[name] = rr
        available = [r for r in rr if r['available']]
        s = dict(frames=len(rr), available=len(available), coverage=len(available) / len(rr),
                 ADDsym_AUC_full=pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rr], 1.0))
        for key in ('translation_cm', 'rotation_deg', 'yaw_deg', 'IoU3D'):
            s[key] = dict(median=float(np.median([r[key] for r in available])),
                          P90=float(np.quantile([r[key] for r in available], .9)))
        if name != 'R0':
            assert s['available'] == previous[name]['available']
            for key in ('translation_cm', 'rotation_deg', 'yaw_deg', 'IoU3D'):
                for stat in ('median', 'P90'):
                    assert np.isclose(s[key][stat], previous[name][key][stat], atol=1e-7)
        tables[name] = s
        print(name, json.dumps(s), flush=True)
    green_path = F.RAW / 'green150_saved_labels_v1/METRICS.json'
    green = F.read(green_path)['modes']['manual_only']['rows']
    green_table = {name: summary(green[name]) for name in inputs}
    dev_path = F.DOC / 'DEV_COMPARISON.json'
    result = dict(comparison='R0 + registered dimensions + unchanged PnP versus N2 + same dimensions + same PnP',
                  new_training=0, new_neural_inference=0, pose_recomputed=True,
                  N2_pose_parity_with_existing_results=True, pose_summary=tables, pose_rows=rows,
                  green_manual_2d=green_table, dev_2d=F.read(dev_path)['table'],
                  limitations=['Reused DEV319; reconstructed pose GT, not independent physical measurement.',
                               'Green150 is saved-label 2D only; no valid new 6D comparison.',
                               'Removing N2 removes learned dimension conditioning; dimensions remain in PnP.'],
                  sources=[F.binding(p) for p in sources + [green_path, dev_path, E.ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
                                                           E.C.POSE / 'GEOMETRY_RESOLVED_POSE_GT.json', E.C.POSE / 'AXIS_REVIEW_MANIFEST.json',
                                                           E.HERE / 'pose.py', E.HERE / 'inference.py']])
    F.freeze(F.DOC / 'WITHOUT_REFINER_COMPARISON.json', result)
    print('GREEN_SEED1', json.dumps(green_table['N2_DIM_ONLY_seed1']), flush=True)


if __name__ == '__main__':
    main()
