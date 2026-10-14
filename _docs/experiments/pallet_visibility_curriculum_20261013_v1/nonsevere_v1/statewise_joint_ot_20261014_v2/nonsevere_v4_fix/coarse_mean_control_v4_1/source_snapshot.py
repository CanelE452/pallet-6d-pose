"""Post-choice CAL mean-only ablation; no BI fitting, RGB or OT constraints."""
import argparse
from pathlib import Path
from copy import deepcopy
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from . import common as C
from .evaluation_bias_v4_1 import load_bias_selection
from .statewise_evaluation_v4 import _verify_prediction_seal, _summary, _verify_sources
from .scoring import pose_metric


def apply_mean(geometry, mean):
    mean = np.asarray(mean, dtype=float)
    R0, t0 = np.asarray(geometry['R_cf']), np.asarray(geometry['centroid'])
    center_xyz = np.asarray(geometry['native_xyz']).mean(axis=0)
    center0 = R0 @ center_xyz + t0
    if mean.shape != (6,) or not np.isfinite(mean).all() or center0[2] <= 0:
        raise ValueError('Finite mean and positive initial model-center depth required')
    R = Rotation.from_rotvec(mean[:3]).as_matrix() @ R0
    z = center0[2] * np.exp(mean[5])
    center = np.r_[(center0[:2] / center0[2] + mean[3:5]) * z, z]
    t = center - R @ center_xyz
    Q = np.asarray(geometry['physical_registration_Q'])
    # This identity checks the intended prior coordinates, including models
    # whose centroid differs from their coordinate origin.
    reconstructed = np.r_[Rotation.from_matrix(R @ R0.T).as_rotvec(),
        center[:2] / center[2] - center0[:2] / center0[2], np.log(center[2] / center0[2])]
    if not np.allclose(reconstructed, mean, atol=1e-12, rtol=0.):
        raise RuntimeError('Closed-form prior mean coordinate identity failed')
    return {'R_cf': R.tolist(), 'centroid': t.tolist(), 'R_physical': (R @ Q).tolist(),
            'physical_registration_Q': Q.tolist(), 'cf_extents': deepcopy(geometry['cf_extents']),
            'source': 'CAL_MEDIAN_COARSE_POSE_ONLY_NO_BI_NO_OT',
            'joint_OT_feasibility_checked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    run, out = args.evaluation.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve completed ablation')
    selected, model, cfg, choice, choice_seal = (load_bias_selection(args.selection)
        if args.selection else load_bias_selection())
    rows, protocol, seal = _verify_prediction_seal(run)
    mean = np.asarray(model['prior_fit']['median_bias6']) * cfg['bias_strength']
    plan = run.parent / 'COARSE_MEAN_ONLY_CONTROL_PLAN_V1.json'
    if C.read(plan)['bias_choice_sha256'] != choice['sha256']:
        raise ValueError('The additive control plan must bind this CAL choice')
    sources = {str((Path(__file__).parent / name).relative_to(C.ROOT)):
               C.sha(Path(__file__).parent / name) for name in (
                   'coarse_mean_control_v4_1.py', 'evaluation_bias_v4_1.py',
                   'statewise_evaluation_v4.py', 'scoring.py', 'common.py')}
    plan_sha = C.sha(plan)
    predicted = [{k: deepcopy(r[k]) for k in ('id', 'source_record_id', 'partition', 'session', 'cohort')}
                 | {'actual_pose': apply_mean(r['runtime']['initial_geometry'], mean)} for r in rows]
    out.mkdir(parents=True)
    C.write(out / 'PROTOCOL.json', {'schema': 'COARSE_CAL_MEAN_ONLY_POST_CHOICE_PROTOCOL_V1',
        'selected_config': choice, 'selection_seal': choice_seal,
        'source_prediction_seal_sha256': C.sha(run / 'PREDICTION_SEAL.json'),
        'source_sha256': sources, 'control_plan_sha256': plan_sha,
        'effective_mean6': mean.tolist(),
        'BI_fitting_or_OT_constraints_or_RGB_reads': False,
        'parameters_fitted_or_selected_by_this_control': False,
        'all98_retained': True, 'primary_severe': 0})
    shutil.copyfile(__file__, out / 'source_snapshot.py')
    C.save_rows(out / 'PREDICTIONS.jsonl.gz', predicted)
    C.write(out / 'PREDICTION_SEAL.json', {'before_own_GT_scoring': True,
        'predictions_sha256': C.sha(out / 'PREDICTIONS.jsonl.gz'),
        'protocol_sha256': C.sha(out / 'PROTOCOL.json'), 'rows': 98,
        'selected_parameters_unchanged': True})
    def verify_local():
        _verify_sources(sources)
        if C.sha(plan) != plan_sha:
            raise ValueError('Additive control plan changed')
        own = C.read(out / 'PREDICTION_SEAL.json')
        if (C.sha(out / 'PREDICTIONS.jsonl.gz') != own['predictions_sha256']
                or C.sha(out / 'PROTOCOL.json') != own['protocol_sha256']
                or C.sha(out / 'source_snapshot.py') != sources[str(Path(__file__).relative_to(C.ROOT))]):
            raise ValueError('Own control prediction/source seal changed')
    verify_local()
    # Own label-coordinate access starts only after the complete control seal.
    cohort = run.parents[2]
    label_paths = {'CAL': cohort / 'OOF_pose_v1/ISOLATED_CAL_SCORING.json',
                   'TEST': cohort / 'test_reused_v1/ISOLATED_TEST_SCORING.json'}
    labels = {p: C.read(path)['rows'] for p, path in label_paths.items()}
    for part, n in (('CAL', 46), ('TEST', 49)):
        ids = {r['id'] for r in predicted if r['partition'] == part and r['cohort'] == 'PRIMARY_NONSEVERE'}
        if len(ids) != n or ids != set(labels[part]):
            raise ValueError('Control/scorer primary population mismatch')
    scores = [r | {'metrics': pose_metric(r['actual_pose'],
                         labels[r['partition']][r['source_record_id']]['target']['truth'])}
              for r in predicted]
    summary = {p: _summary([r['metrics'] for r in scores
        if r['partition'] == p and r['cohort'] == 'PRIMARY_NONSEVERE']) for p in ('CAL', 'TEST')}
    load_bias_selection(args.selection) if args.selection else load_bias_selection()
    _verify_prediction_seal(run)
    verify_local()
    C.write(out / 'POSTSEAL_SCORING.json', {'schema': 'COARSE_MEAN_ONLY_POSTSEAL_SCORE_V1',
        'summary': summary, 'rows': scores,
        'label_sha256': {p: C.sha(path) for p, path in label_paths.items()},
        'prediction_seal_sha256': C.sha(out / 'PREDICTION_SEAL.json'),
        'control_did_not_change_final_method_selection': True,
        'independent_physical_metrology_or_generalization_claim': False})
    print(summary)


if __name__ == '__main__':
    main()
