"""Independent public-only pose-vector arithmetic; no F/model/evaluator imports."""
from collections import Counter
import math

import numpy as np

from . import common as C


def rotations(order):
    assert order in (1, 2, 4)
    return [np.asarray([[math.cos(t), 0, math.sin(t)], [0, 1, 0],
                        [-math.sin(t), 0, math.cos(t)]])
            for t in (2 * math.pi * j / order for j in range(order))]


def main():
    reference = C.read(C.DOC / 'EVALUATION_POSE_REFERENCE.json')
    assert reference['used_in_inference'] is False
    truth = {r['id']: r for r in reference['records']}
    assert len(truth) == 319
    maxima = dict(translation_cm=0., rotation_deg=0., ADDsym_m=0., ADDsym_normalized=0.)
    counts = Counter()
    failures = []
    new_F_calls = 0
    raw = list(C.rows(C.DOC / 'PREDICTIONS.jsonl.gz'))
    assert len(raw) == 5742
    for row in raw:
        p = row['actual_pose']
        assert row['evaluation_reference_used_in_inference'] is False
        assert p['available'] == row['pose']['available']
        new_F_calls += int(row.get('current_experiment_F_attempt', False))
        if not p['available']:
            failures.append(dict(id=row['id'], method=row['method'], seed=row['seed']))
            continue
        g = truth[row['id']]
        R = np.asarray(p['R_physical'], float)
        t = np.asarray(p['centroid'], float)
        G = np.asarray(g['R'], float)
        gt = np.asarray(g['t'], float)
        xyz = np.asarray(g['xyz'], float)
        assert np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-10)
        assert abs(np.linalg.det(R) - 1.) < 1e-10
        assert row['final_hypothesis'] == p['selected_hypothesis']
        assert row['fixed_metadata']['dimensions_pnp_WH_D_m'] == g['xyz']
        signs = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        cuboid = signs * xyz / 2.
        angles, distances = [], []
        for symmetry in rotations(g['order']):
            target = G @ symmetry
            angles.append(math.degrees(math.acos(float(np.clip((np.trace(target.T @ R) - 1.) / 2., -1., 1.)))))
            distances.append(float(np.mean(np.linalg.norm(cuboid @ R.T + t - (cuboid @ target.T + gt), axis=1))))
        add = min(distances)
        rebuilt = dict(translation_cm=float(np.linalg.norm(t - gt) * 100.),
                       rotation_deg=min(angles), ADDsym_m=add,
                       ADDsym_normalized=add / float(np.linalg.norm(xyz)))
        for key, value in rebuilt.items():
            difference = abs(value - row['pose'][key])
            maxima[key] = max(maxima[key], difference)
            assert difference < 1e-7, (row['id'], row['method'], key, difference)
        counts[(str(row['seed']), row['method'])] += 1
    assert new_F_calls == 1914
    payload = dict(status='PASS', rows=5742, new_F_calls_recorded=1914,
        independent_F_calls=0, independent_model_calls=0,
        available_pose_vectors=sum(counts.values()), failed=failures,
        by_seed_method={seed + '/' + method: n for (seed, method), n in counts.items()},
        max_abs_difference=maxima, absolute_tolerance=1e-7,
        reference_sha256=C.sha(C.DOC / 'EVALUATION_POSE_REFERENCE.json'),
        predictions_sha256=C.sha(C.DOC / 'PREDICTIONS.jsonl.gz'),
        reference_limit='geometric proxy, repeated REAL_DEV; no physical metrology',
        source_sha256=C.sha(__file__))
    C.write(C.DOC / 'POSE_NUMERIC_VERIFICATION.json', payload)
    print('POSE_NUMERIC_VERIFICATION_PASS', sum(counts.values()), maxima, flush=True)


if __name__ == '__main__':
    main()
