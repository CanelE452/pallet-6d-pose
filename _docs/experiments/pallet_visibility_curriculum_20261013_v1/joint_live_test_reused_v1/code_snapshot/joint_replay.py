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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', choices=('calibration', 'test'), required=True)
    parser.add_argument('--config-v1', type=Path, required=True)
    parser.add_argument('--config-joint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configs = [args.config_v1.resolve(), args.config_joint.resolve()]
    partition = {'calibration': 'CAL', 'test': 'TEST'}[args.partition]
    ids = set(C.read(C.DOC / 'SPLIT.json')['real_split']['frame_ids'][partition])
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_'))
    code_hashes = {p.name: C.sha(p) for p in files}
    bindings = _model_files(configs[0], ('visibility_model', 'segmentation_weights'))
    bindings += _model_files(configs[1], ('joint_model', 'segmentation_weights', 'legacy_visibility_model'))
    model_hashes = {str(p): C.sha(p) for p in bindings}
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    for path in files:
        shutil.copy2(path, snapshot / path.name)
    C.write(args.output / 'PROTOCOL.json', dict(
        partition=args.partition, ids=sorted(ids), code_sha256=code_hashes, model_sha256=model_hashes,
        config_sha256={str(p): C.sha(p) for p in configs},
        prediction_inputs_sha256=C.sha(C.DOC / 'PREDICTION_INPUTS.json'),
        contract_sha256=C.sha(C.DOC / 'DATA_CONTRACT.json'),
        candidate_selection='CAL OOF RGB26_F4F8_PCA32, C=.01, usable .45, ridge100, correction16px, WD margin.5px2',
        deployment_fit_scope='FULLCAL frozen export_v2; distinct from CAL OOF predictions',
        methods=['INITIAL', 'SEGMENTED_RGB_CURRICULUM', 'JOINT_SINGLE_FAMILY', 'JOINT_WD_BRANCH'],
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
        frames = [f for f in C.prediction_inputs() if f['id'] in ids]
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
    C.save_rows(args.output / 'PREDICTIONS.jsonl.gz', rows)
    C.write(args.output / 'PREDICTION_SEAL.json', dict(before_scoring=True, complete=True,
        sha256=C.sha(args.output / 'PREDICTIONS.jsonl.gz'), rows=len(rows),
        methods=dict(Counter(r['method'] for r in rows)), GT_read_canary='PASS',
        seconds=time.perf_counter() - started))
    print('SEALED', len(rows), args.output, flush=True)


if __name__ == '__main__':
    main()
