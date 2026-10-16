"""Seal a fixed current-RGB joint/branch candidate before reused TEST scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from .api import VisibilityCurriculum
from .branch_curriculum import solve_branch_curriculum
from .joint_api import JointSpatialBranch
from .run import no_annotations


def original_family_control(result, initial, points, K):
    """Reuse exactly the candidate's image evidence, retaining the initial family."""
    original = next(x for x in result['branch_result']['candidates'] if x['original'])
    if not original['available']:
        raise ValueError('Supplied original family unexpectedly unavailable')
    pose = deepcopy(initial)
    pose.update(R_cf=original['R_cf'], R_physical=original['R_physical'],
                centroid=original['centroid'], cf_extents=original['dimensions'])
    solution = deepcopy(original['curriculum'])
    projected = C.project(C.cuboid(*pose['cf_extents']), pose['R_cf'], pose['centroid'], K)
    completed = np.array(points, float, copy=True)
    hidden = np.asarray(result['self_hidden'], bool)
    if solution['accepted']:
        completed[:8] = result['corrected_measurements'][:8]
        completed[np.flatnonzero(hidden)] = projected[hidden]
    return dict(actual_pose=pose, solution=solution, self_hidden=result['self_hidden'],
                appearance=result['appearance'], completed_points=completed.tolist(),
                projected_points=projected.tolist(), corrected_measurements=result['corrected_measurements'],
                provenance=dict(result['provenance'], branch_comparison_used=False), timing=result['timing'])


def visible_together_control(result, initial, points, K, dimensions):
    """Change only visible staging; retain the same evidence and unknown rule."""
    config = deepcopy(result['branch_result']['configuration'])
    config['curriculum']['staged_visible'] = False
    branch = solve_branch_curriculum(np.asarray(result['corrected_measurements'])[:8], K, dimensions,
                                    initial, result['self_hidden'], result['appearance'], config)
    actual = branch['actual_pose']
    projected = C.project(C.cuboid(*actual['cf_extents']), actual['R_cf'], actual['centroid'], K)
    completed = np.array(points, float, copy=True)
    hidden = np.asarray(result['self_hidden'], bool)
    if branch['accepted']:
        completed[:8] = result['corrected_measurements'][:8]
        completed[np.flatnonzero(hidden)] = projected[hidden]
    return dict(actual_pose=actual, solution=branch['solution'], branch_result=branch,
                self_hidden=result['self_hidden'], appearance=result['appearance'],
                completed_points=completed.tolist(), projected_points=projected.tolist(),
                corrected_measurements=result['corrected_measurements'],
                provenance=dict(result['provenance'], visible_staged=False, unknown_rule_unchanged=True))


def _row(frame, method, result):
    solution = deepcopy(result['solution'])
    if 'branch_result' in result:
        solution['selected_curriculum_accepted'] = solution['accepted']
        solution['accepted'] = result['branch_result']['accepted']
        solution['reason'] = result['branch_result']['selection_reason']
    return dict(id=frame['id'], session=frame['session'], arm='N3_SUBPIX', method=method,
                actual_pose=result['actual_pose'], solution=solution, self_hidden=result['self_hidden'],
                appearance=result['appearance'], original_points=frame['points']['N3_SUBPIX'][:8],
                native_points=result['completed_points'], projected_points=result['projected_points'],
                corrected_measurements=result.get('corrected_measurements'),
                joint_observation=result.get('joint_observation'), branch_result=result.get('branch_result'),
                provenance=result.get('provenance'), timing=result.get('timing'))


def _model_files(config_path, keys):
    config = C.read(config_path)
    paths = [(config_path.parent / config[k]).resolve() for k in keys]
    if 'joint_model' in keys:
        model_path = (config_path.parent / config['joint_model']).resolve()
        model = C.read(model_path)
        paths.extend([(model_path.parent / model['arrays_file']).resolve(),
                      C.ROOT / model['backbone']['weights_repo_relative']])
    return paths


def population_ids(input_path, split_path, partition, cohort_path=None):
    """Enforce the declared population before any image or model inference."""
    ids = set(C.read(split_path)['real_split']['frame_ids'][partition])
    frames = C.read(input_path)['frames']
    frame_ids = [f['id'] for f in frames]
    if len(frame_ids) != len(set(frame_ids)) or not ids.issubset(frame_ids):
        raise ValueError('Prediction inputs and unique split ids disagree')
    if cohort_path is not None:
        cohort = C.read(cohort_path)
        if C.sha(input_path) != cohort['hashes']['PREDICTION_INPUTS.json'] or C.sha(split_path) != cohort['hashes']['SPLIT.json']:
            raise ValueError('Declared cohort input/split hash changed')
        if set(frame_ids) & set(cohort['excluded_severe_ids']):
            raise ValueError('Severe frame entered a nonsevere inference population')
        if len(frames) != cohort['eligible_frames'] or len(ids) != cohort[partition+'_frames']:
            raise ValueError('Declared nonsevere cohort size changed')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', choices=('calibration', 'test'), required=True)
    parser.add_argument('--config-v1', type=Path, required=True)
    parser.add_argument('--config-joint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prediction-inputs', type=Path, default=C.DOC/'PREDICTION_INPUTS.json')
    parser.add_argument('--split', type=Path, default=C.DOC/'SPLIT.json')
    parser.add_argument('--cohort-protocol', type=Path)
    parser.add_argument('--matched-one-stage', action='store_true',
                        help='Add a same-evidence branch control with visible staging disabled')
    args = parser.parse_args()
    configs = [args.config_v1.resolve(), args.config_joint.resolve()]
    partition = {'calibration': 'CAL', 'test': 'TEST'}[args.partition]
    if (args.prediction_inputs.resolve() != (C.DOC/'PREDICTION_INPUTS.json').resolve()
            or args.split.resolve() != (C.DOC/'SPLIT.json').resolve()) and args.cohort_protocol is None:
        raise ValueError('A custom population requires an explicit bound cohort protocol')
    ids = population_ids(args.prediction_inputs, args.split, partition, args.cohort_protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_'))
    code_hashes = {p.name: C.sha(p) for p in files}
    bindings = _model_files(configs[0], ('visibility_model', 'segmentation_weights'))
    bindings += _model_files(configs[1], ('joint_model', 'segmentation_weights', 'legacy_visibility_model'))
    model_hashes = {str(p): C.sha(p) for p in bindings}
    config_hashes = {str(p): C.sha(p) for p in configs}
    input_hashes = {str(p.resolve()): C.sha(p) for p in (args.prediction_inputs, args.split)}
    if args.cohort_protocol:
        input_hashes[str(args.cohort_protocol.resolve())] = C.sha(args.cohort_protocol)
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    for path in files:
        shutil.copy2(path, snapshot / path.name)
    C.write(args.output / 'PROTOCOL.json', dict(
        partition=args.partition, ids=sorted(ids), code_sha256=code_hashes, model_sha256=model_hashes,
        config_sha256=config_hashes, population_files_sha256=input_hashes,
        prediction_inputs_sha256=C.sha(args.prediction_inputs),
        contract_sha256=C.sha(C.DOC / 'DATA_CONTRACT.json'),
        candidate_selection='CAL OOF RGB26_F4F8_PCA32, C=.01, usable .45, ridge100, correction16px, WD margin.5px2',
        deployment_fit_scope='Full declared CAL only; bound joint MODEL.json records all feature/head fitting bases',
        cohort_protocol=str(args.cohort_protocol.resolve()) if args.cohort_protocol else None,
        methods=['INITIAL', 'SEGMENTED_RGB_CURRICULUM', 'JOINT_SINGLE_FAMILY', 'JOINT_WD_BRANCH']
                + (['JOINT_VISIBLE_TOGETHER'] if args.matched_one_stage else []),
        matched_visible_staging_control=args.matched_one_stage,
        prior_TEST_metrics_already_observed=args.partition == 'test',
        evaluation_status='reused historical held-session exploratory evaluation' if args.partition == 'test'
                          else 'in-sample FULLCAL deployment replay, not generalization estimate',
        inference_inputs='current RGB, cached native N3 points, camera K, registered dimensions, original initial pose',
        new_detector_forward=0, current_RGB_joint_forward=True, self_hidden_fixed_from_original_pose=True,
        projected_points_used_for_fit=False, FINAL_rows_used=False))
    cv2.setNumThreads(1)
    started = time.perf_counter()
    rows = []
    with no_annotations():
        frames = [f for f in C.read(args.prediction_inputs)['frames'] if f['id'] in ids]
        assert len(frames) == len(ids)
        old = VisibilityCurriculum.from_config(configs[0])
        joint = JointSpatialBranch.from_config(configs[1])
        for index, frame in enumerate(frames):
            image_path = C.ROOT / frame['image']
            if C.sha(image_path) != frame['image_sha256']:
                raise ValueError('Image changed: ' + frame['id'])
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
            initial = frame['initial_poses']['N3_SUBPIX']
            q = np.asarray(frame['points']['N3_SUBPIX'], float)
            v1 = old.refine(image, q, frame['K'], initial)
            result = joint.refine(image, q, frame['K'], initial, frame['xyz'])
            single = original_family_control(result, initial, q, frame['K'])
            rows.extend([_row(frame, 'SEGMENTED_RGB_CURRICULUM', v1),
                         _row(frame, 'JOINT_SINGLE_FAMILY', single),
                         _row(frame, 'JOINT_WD_BRANCH', result)])
            if args.matched_one_stage:
                together = visible_together_control(result, initial, q, frame['K'], frame['xyz'])
                rows.append(_row(frame, 'JOINT_VISIBLE_TOGETHER', together))
            rows.append(dict(id=frame['id'], session=frame['session'], arm='N3_SUBPIX', method='INITIAL',
                actual_pose=initial, solution=dict(accepted=False, R=initial['R_cf'], t=initial['centroid'],
                    used_ids=list(range(8)), reason='frozen_initial'), self_hidden=v1['self_hidden'],
                appearance=v1['appearance'], original_points=q[:8].tolist(), native_points=q.tolist(),
                projected_points=C.project(C.cuboid(*initial['cf_extents']), initial['R_cf'],
                                           initial['centroid'], frame['K']).tolist()))
            if (index + 1) % 10 == 0:
                print('JOINT_LIVE_INFER', index + 1, len(frames), round(time.perf_counter() - started, 1), flush=True)
    assert code_hashes == {p.name: C.sha(p) for p in files}, 'Code changed during frozen inference'
    assert model_hashes == {str(p): C.sha(p) for p in bindings}, 'Model changed during inference'
    assert config_hashes == {str(p): C.sha(p) for p in configs}, 'Configuration changed during inference'
    assert input_hashes == {path: C.sha(path) for path in input_hashes}, 'Population changed during inference'
    C.save_rows(args.output / 'PREDICTIONS.jsonl.gz', rows)
    C.write(args.output / 'PREDICTION_SEAL.json', dict(before_scoring=True, complete=True,
        sha256=C.sha(args.output / 'PREDICTIONS.jsonl.gz'), rows=len(rows),
        methods=dict(Counter(r['method'] for r in rows)), GT_read_canary='PASS',
        seconds=time.perf_counter() - started))
    print('SEALED', len(rows), args.output, flush=True)


if __name__ == '__main__':
    main()
