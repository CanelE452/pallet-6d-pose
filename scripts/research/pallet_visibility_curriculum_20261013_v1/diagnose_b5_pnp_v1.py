"""Single-case, posthoc B5 weight/prior mechanism controls; no runtime changes.

Conditional image covariance is used only as a diagnostic intervention, not
as a calibrated total-error budget or a selected new method. All predictions
are sealed before the controls are scored against the existing reference.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import statewise_partial_boundary_u_solver_v4_6 as SOLVER
from . import evaluate_u_regions_v4_5 as PARENT
from .scoring import pose_metric


BASE = PARENT.BOUNDARY / 'shared_edge_partial_b_eval_v4_7'
OUTPUT = PARENT.BOUNDARY / 'b5_pnp_diagnosis_v1'
TARGET = 'eval_pallet07:1778652166837872128'
ARMS = ('EXACT_CURRENT_REPLAY', 'NO_PRIOR', 'B5_IMAGE_COV_DIAGNOSTIC',
        'B5_IMAGE_COV_NO_PRIOR_DIAGNOSTIC')


def _summary(solution):
    return dict(accepted=solution['accepted'], status=solution['status'],
        R=solution['R'], t=solution['t'], objective=solution.get('objective'),
        objective_components=solution.get('objective_components'),
        B5_residual_px=next((m['measurement_residual_px'] for m in
            solution.get('visible_protection_audit', []) if m['native_corner'] == 5), None),
        B6_normal_residual_px=next((l['normal_residual_px'] for m in
            solution.get('boundary_observation_audit', []) if m['native_corner'] == 6
            for l in m['lines']), None),
        rank=solution.get('rank'), ambiguity=solution.get('ambiguity'))


def main():
    if OUTPUT.exists(): raise FileExistsError('Preserve previous diagnosis')
    protocol = C.read(BASE / 'PROTOCOL.json')
    E._verify_sources(protocol['source_sha256']); E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    frame = BASE / 'TEST_frames' / (TARGET.replace(':', '__') + '.json')
    row = C.read(frame)
    upstream = PARENT.OLD_RUN.OUTPUT / 'PREDICTIONS.jsonl.gz'
    original = next(r for r in C.rows(upstream) if r['id'] == TARGET)
    rt = original['runtime']; g = row['runtime']['initial_geometry']
    assert g == rt['initial_geometry'] and row['states'] == rt['states']
    fit = deepcopy(row['frozen_BI_fit'])
    base_spec = E._physics_spec(rt, fit, protocol['solver_configuration'])
    boundary = row['partial_boundary_observation']['boundary_observations']
    regions = row['u_region_observation']['u_regions']
    old = row['arms']['SHARED_EDGE_PARTIAL_B']['solution']
    sources = deepcopy(protocol['source_sha256'])
    sources[str(Path(__file__).relative_to(C.ROOT))] = C.sha(__file__)
    inputs = {str(p.relative_to(C.ROOT)): C.sha(p) for p in
        (frame, upstream, BASE / 'PROTOCOL.json', BASE / 'TEST_PREDICTION_SEAL.json')}
    OUTPUT.mkdir(parents=True)
    control_protocol = dict(schema='B5_PNP_SINGLE_CASE_MECHANISM_DIAGNOSIS_V1',
        id=TARGET, arms=list(ARMS), hypothesis='old B fallback covariance and coarse prior compromise with new B5',
        intervention='2x2 prior_weight 1/0 and original B5 CAL fit covariance/image conditional covariance',
        only_B5_covariance_modified=True, other_points_and_weights_fixed=True,
        all_OT_U_partial_B_geometry_Q_fixed=True, S_queries=0, U_queries=0,
        conditional_covariance_is_not_new_total_error_calibration=True,
        production_method_or_CAL_models_changed=False, new_GT_inputs_to_solver=False,
        TEST_is_reused_posthoc_case_diagnosis=True,
        source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'])
    C.write(OUTPUT / 'PROTOCOL.json', control_protocol)
    for relative, digest in sources.items():
        dest = OUTPUT / 'code_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(C.ROOT / relative, dest)
        assert C.sha(dest) == digest
    started = time.monotonic(); results = {}
    immutable = E._digest(dict(base_spec=base_spec, boundary=boundary, regions=regions))
    for arm in ARMS:
        spec = deepcopy(base_spec)
        if arm in ('NO_PRIOR', 'B5_IMAGE_COV_NO_PRIOR_DIAGNOSTIC'):
            spec['config']['prior_weight'] = 0.
        if arm in ('B5_IMAGE_COV_DIAGNOSTIC', 'B5_IMAGE_COV_NO_PRIOR_DIAGNOSTIC'):
            point = next(m for m in spec['visible'] if m['native_corner'] == 5)
            point['fit_covariance_px2'] = deepcopy(point['covariance_px2'])
        solution = SOLVER.solve_boundary_u_region_joint_ot_pnp(
            **spec, boundary_observations=boundary, u_regions=regions, boundary_factor_weight=1.)
        assert immutable == E._digest(dict(base_spec=base_spec, boundary=boundary, regions=regions))
        if arm == 'EXACT_CURRENT_REPLAY':
            assert solution['accepted'] == old['accepted']
            assert np.allclose(solution['R'], old['R'], rtol=0, atol=1e-8)
            assert np.allclose(solution['t'], old['t'], rtol=0, atol=1e-8)
        results[arm] = dict(spec=spec, solution=solution, summary=_summary(solution))
        C.write(OUTPUT / (arm + '.json'), results[arm])
        print(arm, C.finite(results[arm]['summary']), flush=True)
    E._verify_sources(sources); E._verify_sources(inputs)
    E._verify_sources(protocol['protected_user_sha256'])
    C.write(OUTPUT / 'PREDICTION_SEAL.json', dict(
        protocol_sha256=C.sha(OUTPUT / 'PROTOCOL.json'), before_new_target_annotation_score=True,
        prediction_sha256={arm: C.sha(OUTPUT / (arm + '.json')) for arm in ARMS},
        exact_current_replay_R_t=True, new_fit_calls=4,
        source_sha256=sources, input_sha256=inputs,
        elapsed_seconds=time.monotonic()-started, no_production_config_or_model_change=True))
    label_path = E.COHORT / 'test_reused_v1/ISOLATED_TEST_SCORING.json'
    truth = C.read(label_path)['rows'][TARGET]['target']['truth']
    scores = {}
    for arm, result in results.items():
        solution = result['solution']
        if not solution['accepted']:
            scores[arm] = dict(unaccepted=True, no_joint_pose_metric_reported=True)
            continue
        pose = dict(R_physical=solution['R_physical'], centroid=solution['t'])
        scores[arm] = dict(pose_metric(pose, truth), B5_residual_px=result['summary']['B5_residual_px'],
            B6_normal_residual_px=result['summary']['B6_normal_residual_px'])
    C.write(OUTPUT / 'POSTSEAL_SCORE.json', dict(id=TARGET, arms=scores,
        labels_used_for_inference_or_parameter_selection=False, target_binding=E._binding(label_path),
        prior_or_weight_experiment_is_not_validation_for_deployment=True,
        source_inputs_protected_user_unchanged=True))
    print('POSTSEAL_METRICS', C.finite(scores), flush=True)


if __name__ == '__main__': main()
