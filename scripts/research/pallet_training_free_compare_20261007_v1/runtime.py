"""Fixed 26-image timing; cached accuracy predictions are never regenerated.

Only ``run_runtime`` starts models. Imports and the schedule command are CPU-only.
The same synchronized intervals are used for all seven full-pipeline routes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from . import common as C
from . import methods as M

ARMS = ('BASE', 'N3_seed1', 'PoseFix_seed1', *C.ARMS)
FRAMES, WARMUP, REPEATS = 26, 20, 5
MAX_PIPELINES = len(ARMS) * (WARMUP + FRAMES * REPEATS)
PANEL = C.ROOT / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
PANEL_LOCK = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/RUNTIME_PROTOCOL.json'
PANEL_LOCK_SHA = 'f411f8654cc8cb4578f7a3fb676e7cf444d340d2b1cddc1353a60135d9977e85'
BASE_CACHE = C.ROOT / 'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'
N3_SELECTION = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json'
NORMALIZATION = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json'
PF_SELECTION = C.ROOT / '_docs/experiments/pallet_sensors_submission_v1/PRIOR_SELECTION.json'
PF_CHECKPOINT = C.ROOT / 'data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR1/last.pt'
BASE_CHECKPOINT = C.ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
N3_POINTS = C.ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed1.json'
PF_POINTS = C.ROOT / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed1_REAL_DEV.npz'
HEAD_POINT_ATOL = 1e-4
PF_CROP_ATOL = 3e-4
YOLO_BACKEND_SOURCE = C.ROOT / 'scripts/research/pallet_three_backbone_runtime_20261002_v1/adapters.py'
PF_NUMERIC_SOURCE = C.ROOT / '_docs/experiments/pallet_sensors_submission_v1/RUNTIME_NUMERIC_AMENDMENT.json'


def schedules(arms=ARMS):
    """Fixed cyclic rotation; odd blocks reverse the route order."""
    arms = tuple(arms)
    assert arms and len(arms) == len(set(arms))
    warm, measured = [], []
    for index in range(WARMUP):
        order = list(arms[index % len(arms):] + arms[:index % len(arms)])
        if index % 2:
            order.reverse()
        for position, arm in enumerate(order):
            warm.append(dict(phase='warmup', arm=arm, warmup_index=index,
                             arm_position=position, image_index=index % FRAMES))
    for repeat in range(REPEATS):
        order = list(arms[repeat % len(arms):] + arms[:repeat % len(arms)])
        if repeat % 2:
            order.reverse()
        for image_index in range(FRAMES):
            for position, arm in enumerate(order):
                measured.append(dict(phase='measured', arm=arm, repeat=repeat,
                                     arm_position=position, image_index=image_index))
    assert Counter(r['arm'] for r in warm) == {a:20 for a in arms}
    assert Counter(r['arm'] for r in measured) == {a:130 for a in arms}
    return warm, measured


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=C.sha(path), bytes=path.stat().st_size)


def describe(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(n=0, median_ms=None, p90_ms=None, mean_ms=None)
    assert np.isfinite(values).all() and (values >= 0).all()
    return dict(n=len(values), median_ms=float(np.median(values)),
                p90_ms=float(np.quantile(values,.9)), mean_ms=float(values.mean()),
                quantile_method='numpy linear')


def _panel():
    assert C.sha(PANEL_LOCK) == PANEL_LOCK_SHA
    saved, lock, base = C.read(PANEL), C.read(PANEL_LOCK), C.read(BASE_CACHE)
    selected = saved['selected']
    assert saved['complete'] and base['complete'] and len(selected) == FRAMES
    assert [r['image_key'] for r in selected] == lock['keys']
    assert lock['warmup'] == WARMUP and lock['repeats'] == REPEATS and lock['seed'] == 1
    assert len({r['frame_id'] for r in selected}) == FRAMES
    sessions = Counter(r['session_id'] for r in selected)
    assert len(sessions) == 13 and set(sessions.values()) == {2}
    images, predictions = [], []
    for row in selected:
        path = C.ROOT / row['image_key']
        assert C.sha(path) == row['image']['sha256']
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None and list(image.shape[:2]) == row['original_hw']
        assert image.dtype == np.uint8 and image.ndim == 3
        assert np.isfinite(row['camera_intrinsics']).all()
        assert np.isfinite(row['dimensions_wdh_m']).all() and min(row['dimensions_wdh_m']) > 0
        candidates = base['frames'][row['image_key']]
        index = int(np.argmax([c['score'] for c in candidates])) if candidates else None
        predictions.append(dict(candidates=copy.deepcopy(candidates), selected_index=index))
        images.append(image)
    return selected, images, predictions, [binding(p) for p in (PANEL,PANEL_LOCK,BASE_CACHE)]


def _gate():
    protocol_path, tests_path = C.DOC/'PROTOCOL.json', C.DOC/'CHECKS_METHODS.json'
    protocol, tests = C.read(protocol_path), C.read(tests_path)
    assert tests['status'] == 'PASS'
    assert tests['code_sha256'] == {n:C.sha(Path(__file__).with_name(n)) for n in ('methods.py','tests.py')}
    assert tests['method_lock'] == M.method_configuration()['method_lock']
    assert protocol['methods'] == M.method_configuration()
    spec = protocol['runtime']
    assert tuple(spec['routes']) == ARMS and spec['warmup_each'] == WARMUP
    assert spec['measured_panel_repeats'] == REPEATS and spec['full_pipeline_ceiling'] == MAX_PIPELINES
    assert spec['panel_sha256'] == C.sha(PANEL)
    assert spec['panel_ids'] == [r['frame_id'] for r in C.read(PANEL)['selected']]
    for code in protocol['code']:
        assert C.sha(C.ROOT/code['path']) == code['sha256'],code['path']
    return [binding(protocol_path),binding(tests_path),binding(__file__),binding(M.__file__)]


def _head_references(selected):
    saved = C.read(N3_POINTS)
    assert saved['complete'] and saved['GT_input'] is False
    assert saved['checkpoint']['sha256'] == C.read(N3_SELECTION)['temperatures']['N3_DIM_SYM_seed1']['checkpoint']['sha256']
    n3 = {r['id']:_raw(r)[0] for r in saved['records']}
    assert len(n3) == len(saved['records']) == 319
    with np.load(PF_POINTS,allow_pickle=False) as arrays:
        assert arrays['points'].shape == (319,2,4,9,2)
        assert str(arrays['checkpoint_sha256'].item()) == C.read(PF_SELECTION)['checkpoints']['1']
        pf = {str(fid):p.copy() for fid,p in zip(arrays['ids'],arrays['points'][:,0,1])}
    assert len(pf) == 319 and all(r['frame_id'] in n3 and r['frame_id'] in pf for r in selected)
    return {'N3_seed1':n3,'PoseFix_seed1':pf},[binding(N3_POINTS),binding(PF_POINTS)]


def _head_parity(arm,points,frame_id,references,raw,models):
    if arm not in references:
        return None
    target = references[arm][frame_id]
    assert (points is None) == (target is None), ('Runtime saved-head detection parity',arm,frame_id)
    if points is None:
        return dict(max_abs_native_px=0.,atol_native_px=HEAD_POINT_ATOL,finite_support_equal=True)
    points,target = np.asarray(points,np.float64),np.asarray(target,np.float64)
    assert points.shape == target.shape == (9,2)
    support = np.isfinite(target)
    assert np.array_equal(np.isfinite(points),support), ('Runtime saved-head finite-mask parity',arm,frame_id)
    error = float(np.max(np.abs(points[support]-target[support]))) if support.any() else 0.
    if arm == 'PoseFix_seed1':
        box = raw['candidates'][raw['selected_index']]['box_xyxy']
        matrix = models.crop_matrix(np.asarray(box,np.float64))
        left,right = models.transform(points,matrix),models.transform(target,matrix)
        crop_error = float(np.max(np.abs(left[support]-right[support]))) if support.any() else 0.
        assert crop_error <= PF_CROP_ATOL, ('BLOCKED_RUNTIME: original PoseFix crop-coordinate guard failed',frame_id,crop_error,PF_CROP_ATOL)
        assert np.array_equal(points[8],target[8],equal_nan=True)
        return dict(max_abs_native_px=error,max_abs_crop_px=crop_error,atol_crop_px=PF_CROP_ATOL,
                    atol_native_px=None,rtol=0,finite_support_equal=True)
    assert error <= HEAD_POINT_ATOL, ('BLOCKED_RUNTIME: saved accuracy-head coordinates differ; no tolerance tuning',arm,frame_id,error,HEAD_POINT_ATOL)
    return dict(max_abs_native_px=error,atol_native_px=HEAD_POINT_ATOL,finite_support_equal=True)


def _consumed_warmup():
    """The archived first BASE warmup used one F; resume never repeats it."""
    path = C.DOC/'RUNTIME_WARMUP_ATTEMPT.json'
    if not path.exists():
        return None,[]
    prior = C.read(path);ex = prior['execution']
    assert prior['status'] == 'FAILED_RUNTIME' and prior['complete'] is False
    assert ex['pipeline_calls_started'] == ex['detector_calls'] == ex['final_F_calls_started'] == ex['final_F_calls_complete'] == 1
    assert ex['pipeline_calls_complete'] == 0 and ex['full_measured_rows'] == 0
    assert ex['solvePnP'] == ex['solvePnPRefineLM'] == 3 and ex['solvePnPGeneric'] == 0
    assert ex['active_job'] == schedules()[0][0]
    assert prior['reason']['type'] == 'AssertionError' and 'score' in prior['reason']['message']
    code = C.DOC/'RUNTIME_WARMUP_ATTEMPT_CODE.py.gz'
    import hashlib
    with gzip.open(code,'rb') as stream:
        code_sha = hashlib.sha256(stream.read()).hexdigest()
    expected = next(x['sha256'] for x in prior['input_bindings'] if Path(x['path']).name=='runtime.py')
    assert code_sha == expected
    with gzip.open(C.DOC/'RUNTIME_WARMUP_ROWS.jsonl.gz','rt') as stream:
        assert not stream.read().strip(), 'Discarded warmup timing was not retained; do not invent it'
    setup = C.read(C.DOC/'RUNTIME_SETUP_ATTEMPT.json')
    assert setup['status']=='BLOCKED_RUNTIME' and setup['execution']['final_F_calls_started']==0
    paths = [path,code,C.DOC/'RUNTIME_WARMUP_ROWS.jsonl.gz',
             C.DOC/'RUNTIME_SETUP_ATTEMPT.json',C.DOC/'RUNTIME_SETUP_CODE.py.gz',C.DOC/'RUNTIME_SETUP_ROWS.jsonl.gz']
    return prior,[binding(p) for p in paths]


def _gpu():
    fields = 'name,uuid,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit,clocks.sm,clocks.mem'
    status = subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader'],text=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    foreign = [r for r in apps.splitlines() if r.split(',')[0].strip() != str(os.getpid())
               and '/usr/share/rustdesk/rustdesk' not in r]
    assert not foreign, ('Other GPU compute preserved; full timing blocked',foreign)
    assert float(status.split(',')[5].strip()) < 80, ('GPU temperature gate',status)
    return dict(gpu=status, fields=fields, compute_apps=apps, foreign_compute=foreign,
                desktop_process_exception='/usr/share/rustdesk/rustdesk', system_changes=False)


def _raw(prediction):
    index = prediction['selected_index']
    if index is None:
        return None, np.zeros(9,bool)
    p = np.asarray(prediction['candidates'][index]['keypoints_xy'],np.float64)
    assert p.shape == (9,2) and not np.isinf(p).any()
    return p, np.isfinite(p).all(-1) & ~(p == -1).all(-1)


def _tf(image, prediction, arm):
    result = copy.deepcopy(prediction)
    p, support = _raw(prediction)
    if p is None:
        return result, dict(no_detection=True,algorithm_corner_calls=0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    native, diagnostic = M.correct(gray,p,support,arm.split('_')[0])
    q = M.cap_points(p,native,image.shape[1],image.shape[0],support) if arm.endswith('CAP1') else native
    result['candidates'][result['selected_index']]['keypoints_xy'] = q
    return result, diagnostic


class Pipeline:
    def __init__(self):
        import torch
        E, _ = C.legacy()
        from scripts.research.pallet_n3_completion_v3 import square_yolo as Y
        from scripts.research.pallet_sensors_submission_v1.prior_model import PoseFixPallet9, expectation
        from scripts.research.pallet_sensors_submission_v1.posefix_contract_math import axis_aligned_crop_matrix, transform_points
        self.pose = E.POSE
        self.env = self.pose.E
        self.inf = importlib.import_module('inference')
        assert Path(self.inf.__file__).resolve() == (self.env.HERE/'inference.py').resolve()
        self.spec = Y.selection_contract(root=C.ROOT,verify_checkpoints=True)['methods']['N3_DIM_SYM_seed1']
        self.norm = C.read(NORMALIZATION)
        # The immutable a22 code has no copy of the original source checkpoint.
        # Match load_head's arithmetic, with the explicitly bound MAIN file.
        n3_path = Path(self.spec['checkpoint_path'])
        assert C.sha(n3_path) == self.spec['checkpoint_sha256']
        ck = torch.load(n3_path,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step'] == 6000 and ck['baseline_checkpoint_sha256'] == self.env.R0_SHA
        self.n3 = importlib.import_module('refiner').model('N3_DIM_SYM',ck['config']).cuda().eval()
        self.n3.load_state_dict(ck['model_state_dict']);self.n3.requires_grad_(False);del ck
        assert C.sha(PF_CHECKPOINT) == C.read(PF_SELECTION)['checkpoints']['1']
        ck = torch.load(PF_CHECKPOINT,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step'] == 6000
        self.pf = PoseFixPallet9().cuda().eval().requires_grad_(False)
        self.pf.load_state_dict(ck['model_state_dict']); del ck
        self.expectation, self.crop_matrix, self.transform = expectation, axis_aligned_crop_matrix, transform_points
        self.extractor = self.env.old('features').FrozenYoloFeatures(BASE_CHECKPOINT,device='cuda')
        self.detector_forwards = self.n3_forwards = self.pf_forwards = 0
        self.hook = self.extractor.yolo.model.model[0].register_forward_pre_hook(self._detector_forward)
        self.bindings = [binding(p) for p in (BASE_CHECKPOINT,n3_path,PF_CHECKPOINT,N3_SELECTION,NORMALIZATION,PF_SELECTION)]
        self.bindings += [binding(p) for p in (Path(Y.__file__),Path(self.inf.__file__),
            Path(self.env.old('features').__file__),Path(self.pose.__file__),
            Path(importlib.import_module(PoseFixPallet9.__module__).__file__),
            Path(importlib.import_module(transform_points.__module__).__file__),
            Path(importlib.import_module('refiner').__file__),
            Path(importlib.import_module('generic_point_refiner').__file__))]
        self.contract = dict(N3=dict(checkpoint=self.spec['checkpoint_sha256'],temperature=self.spec['temperature'],rule=self.spec['rule']),
            PoseFix=dict(checkpoint=C.sha(PF_CHECKPOINT),step=6000,selected_tensor='points[:,0,1]',
                         path='synthetic-only PRIOR1, one uncapped RGB pass; no later real adaptation',cap=None))

    def _detector_forward(self,*_):
        self.detector_forwards += 1

    def correct(self,arm,image,captured,row):
        import torch
        raw = dict(candidates=self.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
        if arm == 'BASE':
            return raw, None
        if arm in C.ARMS:
            return _tf(image,raw,arm)
        if arm == 'N3_seed1':
            dims, order = self.inf.registry_input(row['object_type'])
            np.testing.assert_array_equal(dims,row['dimensions_wdh_m'])
            result, _ = self.inf.predict_captured(self.n3,'N3_DIM_SYM',captured,dims,order,
                self.spec['temperature'],self.spec['rule'],image.shape[:2],self.norm)
            self.n3_forwards += int(result['head_used'])
            return result, None
        assert arm == 'PoseFix_seed1'
        p, valid = _raw(raw)
        if p is None:
            return raw,None
        box = np.asarray(raw['candidates'][raw['selected_index']]['box_xyxy'],np.float64)
        if not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
            return raw,None
        matrix = self.crop_matrix(box)
        rgb = cv2.warpAffine(image,matrix[:2],(288,384),flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)
        points = self.transform(np.where(valid[:,None],p,0),matrix).astype(np.float32)
        args = [torch.as_tensor(x,device='cuda')[None] for x in (rgb.transpose(2,0,1),points,valid)]
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
            q = self.expectation(self.pf(*args))[0].cpu().numpy()
        self.pf_forwards += 1
        q = self.transform(q,np.linalg.inv(matrix));q[~valid] = p[~valid];q[8] = p[8]
        result = copy.deepcopy(raw)
        result['candidates'][result['selected_index']]['keypoints_xy'] = q
        return result,None

    def close(self):
        self.hook.remove();self.extractor.close()


def _raw_difference(actual,expected):
    values = dict(score_max_abs=0.,box_max_abs_native_px=0.,points_max_abs_native_px=0.,
                  candidate_count_actual=len(actual['candidates']),candidate_count_expected=len(expected['candidates']),
                  selected_index_actual=actual['selected_index'],selected_index_expected=expected['selected_index'])
    for a,b in zip(actual['candidates'],expected['candidates']):
        for key,name in [('score','score_max_abs'),('box_xyxy','box_max_abs_native_px'),('keypoints_xy','points_max_abs_native_px')]:
            left,right = np.asarray(a[key],np.float64),np.asarray(b[key],np.float64)
            if left.shape != right.shape:
                values[name] = None
            elif values[name] is not None:
                mask = np.isfinite(left)&np.isfinite(right)
                values[name] = max(values[name],float(np.max(np.abs(left[mask]-right[mask]))) if mask.any() else 0.)
    return values


def _check_raw(actual,expected):
    assert actual['selected_index'] == expected['selected_index']
    assert len(actual['candidates']) == len(expected['candidates'])
    for a,b in zip(actual['candidates'],expected['candidates']):
        for key in ('score','box_xyxy','keypoints_xy'):
            assert np.array_equal(np.asarray(a[key]),np.asarray(b[key]),equal_nan=True), ('Runtime RAW parity',key)


def _preserved(before,after):
    assert before['selected_index'] == after['selected_index'] and len(before['candidates']) == len(after['candidates'])
    selected = before['selected_index']
    for index,(a,b) in enumerate(zip(before['candidates'],after['candidates'])):
        for key in a:
            if key != 'keypoints_xy':
                assert np.array_equal(np.asarray(a[key]),np.asarray(b[key]),equal_nan=True),key
        p,q = np.asarray(a['keypoints_xy']),np.asarray(b['keypoints_xy'])
        mask = np.isfinite(p).all(-1) & ~(p == -1).all(-1)
        if index != selected:
            assert np.array_equal(p,q,equal_nan=True)
        else:
            assert np.array_equal(p[8],q[8],equal_nan=True)
            assert np.array_equal(p[~mask],q[~mask],equal_nan=True)
            assert np.isfinite(q[mask]).all()


def _durable(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    with pending.open('w') as stream:
        json.dump(C.finite(value),stream,allow_nan=False,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def _save_rows(rows):
    path = C.DOC/'RUNTIME_ROWS.jsonl.gz'
    temporary = path.with_name(path.name+'.pending')
    with gzip.open(temporary,'wt',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(C.finite(row),allow_nan=False,separators=(',',':'))+'\n')
    temporary.replace(path)
    return binding(path)


def run_runtime():
    """Root calls this once after READY. Interrupted attempts never retry F/NN."""
    import torch
    sources = _gate()
    path, journal = C.DOC/'RUNTIME.json', C.OUTPUT/'RUNTIME_ATTEMPT.json'
    if path.exists():
        prior = C.read(path)
        assert prior['input_bindings'][:len(sources)] == sources, 'Runtime code/protocol binding drift'
        for item in prior['input_bindings']:
            assert C.sha(item['path']) == item['sha256'],item['path']
        if prior.get('raw_rows'):
            assert C.sha(prior['raw_rows']['path']) == prior['raw_rows']['sha256']
        return prior
    if journal.exists():
        raise RuntimeError('BLOCKED_RUNTIME: prior attempt incomplete; no hidden replay or additional F')
    selected,images,predictions,panel_bindings = _panel()
    references,reference_bindings = _head_references(selected)
    prior,prior_bindings = _consumed_warmup()
    sources += panel_bindings+reference_bindings+prior_bindings
    sources += [binding(YOLO_BACKEND_SOURCE),binding(PF_NUMERIC_SOURCE)]
    assert C.read(PF_NUMERIC_SOURCE)['absolute_crop_tolerance'] == PF_CROP_ATOL
    warm,measured = schedules()
    consumed = int(prior is not None)
    state = dict(status='STARTED',input_bindings=sources,pipeline_calls_started=consumed,pipeline_calls_complete=0,
                 detector_calls=consumed,final_F_calls_started=consumed,final_F_calls_complete=consumed,
                 solvePnP=3*consumed,solvePnPGeneric=0,solvePnPRefineLM=3*consumed,max_pipeline_calls=MAX_PIPELINES,
                 previous_failed_warmup_consumed=consumed,new_pipeline_calls_started=0,new_final_F_calls_complete=0)
    _durable(journal,state)
    rows,models,originals = [],None,{}
    started = time.monotonic()
    environment = dict(packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics','numpy','opencv-python','threadpoolctl')},
                       opencv=cv2.__version__,thread_contract=dict(torch=4,opencv=1),batch=1)
    result = dict(schema='training_free_runtime_v1',status='FAILED_RUNTIME',complete=False,
        arms=list(ARMS),panel=selected,panel_sessions=13,frames=26,warmup_per_arm=20,repeats=5,
        schedule='cyclic route rotation, odd blocks reversed; every frame/arm exactly five measured repeats',
        boundaries=dict(full_pipeline='native uint8 BGR already in RAM -> YOLO preprocess/detector + FP16-rounded neck capture -> correction -> original prediction-only F; synchronized wall, all seven routes include identical intermediate synchronization instrumentation',
            stage_only='same full-call detector output already available -> correction including grayscale/crop, cap, head and transfers; contiguous synchronized subinterval, no separate inference/F replay',
            excluded=['file read/decode','model initialization/checkpoint load','GT metrics','parity checks','receipt writes']),
        GT_inference_access=False,GT_keypoints_read=False,GT_pose_read=False,no_training=True,
        BASE_parity_F_by_this_module=0,environment=environment,input_bindings=sources)
    result['prior_attempts']=dict(archived_setup_and_warmup=[b['path'] for b in prior_bindings],
        previous_warmup_consumed=consumed,first_warmup_time_unavailable_and_not_invented=bool(consumed),
        excluded_from_all_measured_statistics=True,reason='Restore original YOLO cuDNN TF32=True numerical contract; no model/method or result criterion tuning')
    try:
        environment['gpu_before'] = _gpu()
        assert torch.cuda.is_available()
        torch.set_num_threads(4);cv2.setNumThreads(1)
        torch.manual_seed(1);np.random.seed(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        environment['numeric']=dict(matmul_tf32=False,cudnn_tf32=True,cudnn_benchmark=False,cudnn_deterministic=False,
            detector='FP32 inference with original cuDNN TF32=True; unchanged FP16-rounded neck features',
            N3='original cuDNN TF32=True as recorded original seed1 fit',
            PoseFix='FP32 frozen ResNet152; cuDNN TF32=False inside this head only; uncapped raw firstpass',
            CUDA_version=torch.version.cuda,cuDNN_version=torch.backends.cudnn.version())
        models = Pipeline()
        sources += models.bindings;result['method_contracts'] = models.contract
        environment['threadpools'] = __import__('threadpoolctl').threadpool_info()
        environment['actual_torch_threads'] = torch.get_num_threads();environment['actual_opencv_threads'] = cv2.getNumThreads()
        environment['device']=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
        torch.cuda.reset_peak_memory_stats()
        for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM'):
            originals[name] = getattr(cv2,name)
            def wrapped(*args,_name=name,**kwargs):
                state[_name] += 1
                return originals[_name](*args,**kwargs)
            setattr(cv2,name,wrapped)
        with torch.no_grad():
            for job in warm[consumed:]+measured:
                if time.monotonic()-started > 3600:
                    raise RuntimeError('Fixed runtime operating limit reached; no additional calls')
                assert state['pipeline_calls_started'] < MAX_PIPELINES
                state['pipeline_calls_started'] += 1;state['active_job']=job
                state['new_pipeline_calls_started'] += 1
                _durable(journal,state)
                i,arm=job['image_index'],job['arm'];image=images[i];frame=selected[i]
                before_counts={k:state[k] for k in originals}
                torch.cuda.synchronize();t0=time.perf_counter_ns()
                state['detector_calls'] += 1
                captured=models.extractor.predict(image)
                torch.cuda.synchronize();t1=time.perf_counter_ns()
                prediction,diagnostic=models.correct(arm,image,captured,frame)
                torch.cuda.synchronize();t2=time.perf_counter_ns()
                p,_=_raw(prediction)
                state['final_F_calls_started'] += 1
                pose=models.pose.infer(p,np.asarray(frame['camera_intrinsics']),np.asarray(frame['dimensions_wdh_m'])[[0,2,1]],source=False)
                state['final_F_calls_complete'] += 1
                state['new_final_F_calls_complete'] += 1
                torch.cuda.synchronize();t3=time.perf_counter_ns()
                raw=dict(candidates=models.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
                row = dict(**job,id=frame['frame_id'],session=frame['session_id'],
                    full_ms=(t3-t0)/1e6,detector_ms=(t1-t0)/1e6,
                    stage_only_ms=None if arm=='BASE' else (t2-t1)/1e6,F_ms=(t3-t2)/1e6,
                    raw_points=_raw(raw)[0],corrected_points=p,selected_index=prediction['selected_index'],
                    pose_available=pose['available'],final_hypothesis=pose.get('selected_hypothesis'),
                    RAW_cached_bitexact=False,detection_and_center_missing_preserved=False,
                    PnP_counts={k:state[k]-before_counts[k] for k in originals},correction_diagnostic=diagnostic,
                    RAW_replay_difference=_raw_difference(raw,predictions[i]),parity_status='PENDING')
                rows.append(row)
                _check_raw(raw,predictions[i]);_preserved(raw,prediction)
                head_parity = _head_parity(arm,p,frame['frame_id'],references,raw,models)
                row.update(RAW_cached_bitexact=True,detection_and_center_missing_preserved=True,
                           saved_accuracy_head_parity=head_parity,parity_status='PASS')
                state['pipeline_calls_complete'] += 1;state['active_job']=None
                _durable(journal,state)
        assert state['pipeline_calls_complete'] == len(rows) == MAX_PIPELINES-consumed
        assert state['detector_calls'] == state['final_F_calls_complete'] == state['pipeline_calls_started'] == MAX_PIPELINES
        assert len([r for r in rows if r['phase']=='measured' and r['parity_status']=='PASS']) == 910
        result.update(status='DONE',complete=True)
        environment['gpu_after']=_gpu()
        environment['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        result.update(status='FAILED_RUNTIME',complete=False)
        result['reason']=dict(type=type(error).__name__,message=str(error))
        if state['pipeline_calls_started'] == 0:
            result['status']='BLOCKED_RUNTIME'
            # Available CPU correction-stage work is still useful; no detector/F.
            for job in sum(schedules(C.ARMS),[]):
                i=job['image_index'];t0=time.perf_counter_ns()
                prediction,diagnostic=_tf(images[i],predictions[i],job['arm'])
                t1=time.perf_counter_ns();_preserved(predictions[i],prediction)
                rows.append(dict(**job,id=selected[i]['frame_id'],session=selected[i]['session_id'],
                    full_ms=None,stage_only_ms=(t1-t0)/1e6,correction_diagnostic=diagnostic,
                    stage_only_cached_CPU=True,final_F_calls=0,detector_calls=0))
            result['fallback_boundary']='cached immutable RAW prediction + RAM raw BGR -> CPU grayscale/correction/cap only; F0/detector0'
    finally:
        for name,original in originals.items():
            setattr(cv2,name,original)
        if models is not None:
            result['execution_model_forwards']=dict(detector= models.detector_forwards,N3=models.n3_forwards,PoseFix=models.pf_forwards,
                detector_internal_initialization_warmup=max(0,models.detector_forwards-state['new_pipeline_calls_started']),
                previous_attempt_forwards=prior.get('execution_model_forwards') if prior else None,
                note='Detector module hook includes Ultralytics internal initialization warmup; image pipeline budget counts explicit predict calls separately')
            models.close()
        result['execution']=dict(state,stage_only_additional_F_calls=0,BASE_parity_F=0,
            full_measured_rows=sum(r['phase']=='measured' and r.get('full_ms') is not None and r.get('parity_status')=='PASS' for r in rows),
            stage_only_measured_rows=sum(r['phase']=='measured' and r['stage_only_ms'] is not None and
                (r.get('parity_status')=='PASS' or r.get('stage_only_cached_CPU')) for r in rows),
            evaluation_F_calls_by_this_module=0,optimizer_updates=0,elapsed_seconds=time.monotonic()-started)
        result['summaries']={arm:dict(
            full_pipeline=describe([r['full_ms'] for r in rows if r['arm']==arm and r['phase']=='measured' and r.get('full_ms') is not None and r.get('parity_status')=='PASS']),
            stage_only=describe([r['stage_only_ms'] for r in rows if r['arm']==arm and r['phase']=='measured' and r['stage_only_ms'] is not None and
                (r.get('parity_status')=='PASS' or r.get('stage_only_cached_CPU'))])) for arm in ARMS}
        result['raw_rows']=_save_rows(rows);result['input_bindings']=sources
        result['numeric_parity']=dict(RAW_native_bitexact_calls=sum(r.get('RAW_cached_bitexact',False) for r in rows),
            N3_atol_native_px=HEAD_POINT_ATOL,PoseFix_atol_crop_px=PF_CROP_ATOL,
            PoseFix_atol_native_px=None,checked_after_timing=True,new_detector_or_head_or_F_for_parity=0,
            heads={arm:dict(checked_calls=sum(r['arm']==arm and r.get('saved_accuracy_head_parity') is not None for r in rows),
                checked_unique_frames=len({r['id'] for r in rows if r['arm']==arm and r.get('saved_accuracy_head_parity') is not None}),
                max_abs_native_px=max((r['saved_accuracy_head_parity']['max_abs_native_px'] for r in rows if r['arm']==arm and r.get('saved_accuracy_head_parity') is not None),default=None))
                for arm in ('N3_seed1','PoseFix_seed1')})
        result['warmup_accounting']={arm:dict(valid=sum(r['arm']==arm and r['phase']=='warmup' and r.get('parity_status')=='PASS' for r in rows),
            failed_consumed=consumed if arm=='BASE' else 0,configured=20) for arm in ARMS}
        C.write(path,result);state['status']=result['status'];_durable(journal,state)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('schedule','measure'))
    args=parser.parse_args()
    if args.stage=='schedule':
        a,b=schedules();print(json.dumps(dict(warmup=len(a),measured=len(b),arms=ARMS,total=len(a+b))))
    else:
        result=run_runtime();print(json.dumps(dict(status=result['status'],execution=result['execution'])))
