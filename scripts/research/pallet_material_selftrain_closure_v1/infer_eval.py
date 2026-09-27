"""Frozen Wood RGB/metadata-only inference. Reference access is in score_eval only."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import numpy as np
from . import common as C

ARMS = (*C.ARMS, 'TEACHER')


def precision_contract(arm):
    """Match each historical inference path; teacher must not inherit student TF32."""
    assert arm in ARMS
    return dict(cudnn_allow_tf32=arm != 'TEACHER', matmul_allow_tf32=False,
                half=False, cudnn_benchmark=False)


def set_precision(arm, torch_module):
    contract = precision_contract(arm)
    torch_module.backends.cudnn.allow_tf32 = contract['cudnn_allow_tf32']
    torch_module.backends.cuda.matmul.allow_tf32 = contract['matmul_allow_tf32']
    torch_module.backends.cudnn.benchmark = contract['cudnn_benchmark']
    return contract


def validate_metadata(rows):
    allowed = {'id', 'recording', 'recording_group', 'severity', 'image', 'K', 'xyz', 'hw', 'session', 'object_type'}
    assert rows and len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        assert set(r) <= allowed, ('Unexpected reference-bearing metadata field', set(r) - allowed)
        assert {'id', 'recording', 'severity', 'image', 'K', 'xyz', 'hw'} <= set(r)
        assert np.asarray(r['K']).shape == (3, 3) and np.isfinite(r['K']).all()
        assert np.asarray(r['xyz']).shape == (3,) and np.all(np.asarray(r['xyz']) > 0)
        assert len(r['hw']) == 2 and min(r['hw']) > 0
        assert r.get('recording_group', r['recording']) == r['recording']
    return [r['id'] for r in rows]


def image_order_hash(rows):
    value = [(r['id'], r['image']['sha256']) for r in rows]
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def assert_detector_parity(base, pred, teacher=False):
    assert base['selected_index'] == pred['selected_index']
    assert len(base['candidates']) == len(pred['candidates'])
    for i, (a, b) in enumerate(zip(base['candidates'], pred['candidates'])):
        assert a['candidate_index'] == b['candidate_index']
        assert a['score'] == b['score'] and a['box_xyxy'] == b['box_xyxy']
        if teacher:
            assert a['keypoints_conf'] == b['keypoints_conf']
            if i != base['selected_index']:
                assert a['keypoints_xy'] == b['keypoints_xy']
            else:
                assert a['keypoints_xy'][8] == b['keypoints_xy'][8]


def pose_job(row):
    fid, pred, meta = row
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    # No metadata()/load() call: only supplied intrinsics and registered dimensions.
    return fid, D.Pose.infer(D.points(pred), np.array(meta['K']), np.array(meta['xyz']), False)


def thermal_guard():
    temp = int(subprocess.check_output([
        'nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'
    ], text=True).splitlines()[0])
    assert temp < 80, ('Thermal safety stop; no other processes modified', temp)
    return temp


def verify_final_lock():
    lock = C.read(C.DOC / 'WOOD_PREDICTIONS_LOCK.json')
    for b in lock['files']:
        C.verify(b)
    for arm, checkpoint in lock['checkpoints'].items():
        assert checkpoint == C.checkpoint(arm)
        C.verify(checkpoint)
    assert lock['GT/reference_not_read'] is True
    return lock


def cache_read(path, stamp):
    seal = path.with_name(path.stem + '_LOCK.json')
    if not path.exists():
        assert not seal.exists(), ('Incomplete cache seal', seal)
        return None
    assert seal.exists(), ('Unsealed cache requires explicit failure audit, not reuse', path)
    C.verify(C.read(seal)['file'])
    obj = C.read(path)
    assert obj['input_stamp'] == stamp, ('Cache contract changed', path)
    return obj['predictions']


def cache_write(path, predictions, stamp):
    C.save(path, dict(input_stamp=stamp, predictions=predictions), True)
    C.save(path.with_name(path.stem + '_LOCK.json'), dict(file=C.bind(path)), True)


def main():
    if (C.DOC / 'WOOD_PREDICTIONS_LOCK.json').exists():
        verify_final_lock()
        print('WOOD_PREDICTIONS_ALREADY_LOCKED', flush=True)
        return
    import cv2
    import torch
    from ultralytics import YOLO
    from scripts.research.pallet_clean19_structured_easyhard_v1.common import predict
    from scripts.research.pallet_posefix_replay_v1 import core as N
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), 'Host CUDA required; no silent CPU fallback'
    metadata_path = C.RAW / 'EVAL_METADATA.json'
    population_path = C.DOC / 'EVAL_POPULATION_LOCK.json'
    assert population_path.exists(), 'Freeze Wood population before inference'
    population = C.read(population_path)
    # The lock is hashed, not used as a source of any coordinates or targets.
    assert isinstance(population, dict)
    C.verify(population['metadata'])
    assert population['metadata'] == C.bind(metadata_path)
    rows = C.read(metadata_path)
    ids = validate_metadata(rows)
    assert len(ids) == population['n'] and ids == [r['id'] for r in population['records']]
    assert all(r['image'] == p['image'] for r, p in zip(rows, population['records']))
    for r in rows:
        C.verify(r['image'])
    order_hash = image_order_hash(rows)
    common_stamp = dict(metadata=C.bind(metadata_path), population_lock=C.bind(population_path),
                        image_order_sha256=order_hash, inference_code=C.bind(Path(__file__)),
                        RGB_predictor=C.bind(Path(inspect.getsourcefile(inspect.unwrap(predict)))), padding=100,
                        selection='highest confidence; no GT/target matching')
    C.save(C.DOC / 'WOOD_INFERENCE_GPU.json', dict(utc=C.now(), cuda=True,
           device=torch.cuda.get_device_name(0), temperature_C=thermal_guard()))
    predictions, poses, checkpoints, files = {}, {}, {}, [metadata_path, population_path, Path(__file__)]
    for arm in ARMS:
        precision = set_precision(arm, torch)
        checkpoint = C.checkpoint(arm)
        C.verify(checkpoint)
        checkpoints[arm] = checkpoint
        stamp = dict(common_stamp, arm=arm, checkpoint=checkpoint, precision=precision)
        if arm == 'TEACHER':
            stamp['R0_predictions'] = C.bind(C.RAW / 'EVAL_R0.json')
            stamp['teacher_predictor'] = C.bind(Path(N.C.__file__))
        dst = C.RAW / f'EVAL_{arm}.json'
        values = cache_read(dst, stamp)
        if values is None:
            if arm == 'TEACHER':
                ck = torch.load(C.ROOT / checkpoint['path'], map_location='cpu', weights_only=False)
                assert ck['step'] == 300 and ck['protocol_sha256'] == C.sha(N.DOC / 'PROTOCOL.json')
                model = N.C.PoseFixPallet9()
                model.load_state_dict(ck['model_state_dict'])
                model.cuda().eval().requires_grad_(False)
            else:
                model = YOLO(str(C.ROOT / checkpoint['path']), task='pose')
            values = {}
            for j, r in enumerate(rows):
                if j % 16 == 0:
                    print('WOOD_FROZEN_INFERENCE', arm, j, len(rows), thermal_guard(), flush=True)
                image = cv2.imread(str(C.ROOT / r['image']['path']))
                assert image is not None and list(image.shape[:2]) == r['hw']
                values[r['id']] = (N.C.predict(model, image, predictions['R0'][r['id']])
                                   if arm == 'TEACHER' else predict(model, image, padding=100))
            del model
            torch.cuda.empty_cache()
            cache_write(dst, values, stamp)
        assert list(values) == ids, ('Evaluation membership/order drift', arm)
        predictions[arm] = values
        if arm != 'R0':
            for fid in ids:
                assert_detector_parity(predictions['R0'][fid], values[fid], teacher=arm == 'TEACHER')
        pose_path = C.RAW / f'POSE_{arm}.json'
        pose_stamp = dict(predictions=C.bind(dst), metadata=C.bind(metadata_path),
                          solver=C.bind(Path(D.Pose.__file__)), selector=C.bind(Path(D.Selector.__file__)),
                          method='common original D9; no oracle')
        pose_values = cache_read(pose_path, pose_stamp)
        if pose_values is None:
            with ProcessPoolExecutor(max_workers=4) as pool:
                pose_values = dict(pool.map(pose_job, [(r['id'], values[r['id']], r) for r in rows], chunksize=8))
            cache_write(pose_path, pose_values, pose_stamp)
        assert list(pose_values) == ids
        poses[arm] = pose_values
        files.extend([dst, dst.with_name(dst.stem + '_LOCK.json'),
                      pose_path, pose_path.with_name(pose_path.stem + '_LOCK.json')])
        print('WOOD_ARM_FROZEN', arm, len(values), flush=True)
    C.save(C.RAW / 'PREDICTIONS.json', predictions, True)
    C.save(C.RAW / 'POSE_PREDICTIONS.json', poses, True)
    files.extend([C.RAW / 'PREDICTIONS.json', C.RAW / 'POSE_PREDICTIONS.json',
                  Path(D.Pose.__file__), Path(D.Selector.__file__), Path(N.C.__file__),
                  Path(inspect.getsourcefile(inspect.unwrap(predict)))])
    C.save(C.DOC / 'WOOD_PREDICTIONS_LOCK.json', dict(
        created_at=C.now(), files=[C.bind(p) for p in dict.fromkeys(files)], checkpoints=checkpoints,
        image_ids=ids, image_order_sha256=order_hash, frames=len(ids), arms=list(ARMS),
        detector_order_boxes_scores_exact=True, selected_detection_index_saved=True,
        precision_by_arm={arm: precision_contract(arm) for arm in ARMS},
        teacher_confidence_preserved=True, teacher_center_preserved=True,
        GT_reference_not_read=True, **{'GT/reference_not_read': True},
        reference_access_scope='This inference process reads only RGB, checkpoints, registered K/dimensions, IDs and group metadata; no annotation/reference coordinates. Prior provenance audits are separate.',
        selection='same highest-score detection and same production D9 for every arm; no severity routing or oracle',
        D9_contract='corner0..7 pose solve; original 9-point selector residual; registered Wood dimensions; SQPnP/LM unchanged',
        evaluation_filter='none', independent_confirmation=False), True)
    verify_final_lock()
    print('WOOD_ALL_PREDICTIONS_AND_D9_LOCKED', flush=True)


if __name__ == '__main__':
    main()
