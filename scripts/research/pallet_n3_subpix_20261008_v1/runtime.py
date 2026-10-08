"""One fixed 600-call benchmark of the four complete deployment routes.

Only ``run_runtime`` loads models. Accuracy coordinates are never regenerated
here; saved final coordinates and final poses are checked after each interval.
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
import re
import subprocess
import time

import cv2
import numpy as np

from . import common as C
from scripts.research.pallet_training_free_compare_20261007_v1 import methods as M

ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX')
FRAMES, WARMUP, REPEATS = 26, 20, 5
MAX_PIPELINES = 600
POINT_ATOL = 1e-7
HEAD_POINT_ATOL = 1e-4  # inherited deployment-head guard; final output is stricter
PANEL = C.ROOT / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
PANEL_LOCK = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/RUNTIME_PROTOCOL.json'
PANEL_LOCK_SHA = 'f411f8654cc8cb4578f7a3fb676e7cf444d340d2b1cddc1353a60135d9977e85'
BASE_CACHE = C.ROOT / 'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'
N3_SELECTION = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json'
NORMALIZATION = C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json'
BASE_CHECKPOINT = C.ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
N3_POINTS = C.ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed1.json'


def schedules(arms=ARMS):
    """Predetermined cyclic rotation; odd blocks reverse the route order."""
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
    assert len(warm + measured) == MAX_PIPELINES
    return warm, measured


def binding(path):
    """Public binding omits machine-specific absolute filesystem paths."""
    path = Path(path).resolve()
    for root, origin in ((C.DOC.parent.parent.parent.resolve(), 'experiment'),
                         (C.ROOT.resolve(), 'source')):
        if path.is_relative_to(root):
            name = str(path.relative_to(root))
            break
    else:
        name, origin = path.name, 'immutable_baseline_dependency'
    return dict(path=name, origin=origin, sha256=C.sha(path), bytes=path.stat().st_size)


def describe(values):
    values = np.asarray(values, np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    n = len(values)
    return dict(n=n, mean_ms=float(values.mean()) if n else None,
                sample_variance_ms2=float(values.var(ddof=1)) if n >= 2 else None,
                sample_std_ms=float(values.std(ddof=1)) if n >= 2 else None,
                median_ms=float(np.median(values)) if n else None,
                p90_ms=float(np.quantile(values,.9)) if n else None,
                ddof=1, quantile_method='numpy linear',
                std_definition='dispersion of measured per-call latency; not confidence interval')


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
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        assert np.isfinite(row['camera_intrinsics']).all()
        assert np.isfinite(row['dimensions_wdh_m']).all() and min(row['dimensions_wdh_m']) > 0
        candidates = base['frames'][row['image_key']]
        index = int(np.argmax([c['score'] for c in candidates])) if candidates else None
        predictions.append(dict(candidates=copy.deepcopy(candidates), selected_index=index))
        images.append(image)
    return selected, images, predictions, [binding(p) for p in (PANEL,PANEL_LOCK,BASE_CACHE)]


def _accuracy_references(selected):
    """Read fixed predictions only; never targets, visibility or reference poses."""
    path = C.DOC / 'PREDICTIONS.jsonl.gz'
    refs = {a:{} for a in ARMS}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            arm = row.get('method', row.get('arm'))
            assert arm in refs and row['id'] not in refs[arm]
            refs[arm][row['id']] = dict(points=row['native_points'], pose=row.get('actual_pose'))
    assert all(len(values) == 319 for values in refs.values())
    assert all(r['frame_id'] in refs[a] and refs[a][r['frame_id']]['pose'] is not None
               for r in selected for a in ARMS)
    n3 = C.read(N3_POINTS)
    assert n3['complete'] and n3['GT_input'] is False
    assert n3['checkpoint']['sha256'] == C.read(N3_SELECTION)['temperatures']['N3_DIM_SYM_seed1']['checkpoint']['sha256']
    heads = {r['id']:_raw(r)[0] for r in n3['records']}
    assert len(heads) == len(n3['records']) == 319
    return refs, heads, [binding(path), binding(N3_POINTS)]


def _raw(prediction):
    index = prediction['selected_index']
    if index is None:
        return None, np.zeros(9,bool)
    p = np.asarray(prediction['candidates'][index]['keypoints_xy'],np.float64)
    assert p.shape == (9,2) and not np.isinf(p).any()
    return p, np.isfinite(p).all(-1) & ~(p == -1).all(-1)


class Pipeline:
    """One YOLO forward per call; N3 shares that forward's neck tensors."""
    def __init__(self):
        import torch
        E, _ = C.legacy()
        from scripts.research.pallet_n3_completion_v3 import square_yolo as Y
        self.pose = E.POSE
        self.env = self.pose.E
        self.inf = importlib.import_module('inference')
        assert Path(self.inf.__file__).resolve() == (self.env.HERE/'inference.py').resolve()
        self.spec = Y.selection_contract(root=C.ROOT,verify_checkpoints=True)['methods']['N3_DIM_SYM_seed1']
        self.norm = C.read(NORMALIZATION)
        n3_path = Path(self.spec['checkpoint_path'])
        assert C.sha(n3_path) == self.spec['checkpoint_sha256']
        ck = torch.load(n3_path,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step'] == 6000 and ck['baseline_checkpoint_sha256'] == self.env.R0_SHA
        self.n3 = importlib.import_module('refiner').model('N3_DIM_SYM',ck['config']).cuda().eval()
        self.n3.load_state_dict(ck['model_state_dict']); self.n3.requires_grad_(False); del ck
        self.extractor = self.env.old('features').FrozenYoloFeatures(BASE_CHECKPOINT,device='cuda')
        self.detector_forwards = self.n3_forwards = 0
        self.hook = self.extractor.yolo.model.model[0].register_forward_pre_hook(self._detector_forward)
        self.bindings = [binding(p) for p in (BASE_CHECKPOINT,n3_path,N3_SELECTION,NORMALIZATION)]
        self.bindings += [binding(p) for p in (Path(Y.__file__),Path(self.inf.__file__),
            Path(self.env.old('features').__file__),Path(self.pose.__file__),
            Path(importlib.import_module('refiner').__file__),
            Path(importlib.import_module('generic_point_refiner').__file__))]
        self.contract = dict(checkpoint=self.spec['checkpoint_sha256'],temperature=self.spec['temperature'],
            rule=self.spec['rule'],head='unchanged N3_DIM_SYM seed1 step6000',
            input_output='original predict_captured; original FP16-rounded shared neck',
            final_cap_anchor='BASE q0; raw image diagonal 1%',SUBPIX=M.method_configuration()['SUBPIX'])

    def _detector_forward(self,*_):
        self.detector_forwards += 1

    def correct(self,arm,image,captured,row):
        raw = dict(candidates=self.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
        p0, support = _raw(raw)
        if arm == 'BASE':
            return raw, None, None
        result, qn = raw, None
        if arm in ('N3','N3_SUBPIX'):
            dims, order = self.inf.registry_input(row['object_type'])
            np.testing.assert_array_equal(dims,row['dimensions_wdh_m'])
            result, _ = self.inf.predict_captured(self.n3,'N3_DIM_SYM',captured,dims,order,
                self.spec['temperature'],self.spec['rule'],image.shape[:2],self.norm)
            self.n3_forwards += int(result['head_used'])
            qn = _raw(result)[0]
            if arm == 'N3':
                return result, None, qn
        if p0 is None:
            return result, dict(no_detection=True,algorithm_corner_calls=0), qn
        start = p0 if arm == 'SUBPIX' else qn
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        qs, diagnostic = M.correct(gray,start.copy(),support.copy(),'SUBPIX')
        final = M.cap_points(p0.copy(),qs,image.shape[1],image.shape[0],support.copy())
        result = copy.deepcopy(result)
        result['candidates'][result['selected_index']]['keypoints_xy'] = final
        diagnostic['cap_anchor'] = 'BASE'
        diagnostic['total_cap_px'] = .01*float(np.hypot(*image.shape[:2]))
        diagnostic['native_subpix_points'] = qs
        return result, diagnostic, qn

    def close(self):
        self.hook.remove(); self.extractor.close()


def _check_raw(actual,expected):
    assert actual['selected_index'] == expected['selected_index']
    assert len(actual['candidates']) == len(expected['candidates'])
    for a,b in zip(actual['candidates'],expected['candidates']):
        # FULL_CANDIDATES stores these three fields; runtime also carries the
        # inherited candidate_index and confidence vector and preserves them.
        for key in ('score','box_xyxy','keypoints_xy'):
            assert np.array_equal(np.asarray(a[key]),np.asarray(b[key]),equal_nan=True), ('Runtime RAW parity',key)


def _preserved(before,after,image,arm):
    assert before['selected_index'] == after['selected_index'] and len(before['candidates']) == len(after['candidates'])
    selected = before['selected_index']
    for index,(a,b) in enumerate(zip(before['candidates'],after['candidates'])):
        assert set(a) == set(b)
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
            moving = mask[:8]
            # Original N3 clips in float32 network coordinates and its archived
            # output has tiny native-pixel overshoots. Preserve its established
            # numerical contract; both new float64 cap routes are strict.
            tolerance = HEAD_POINT_ATOL if arm == 'N3' else 1e-10
            assert (np.linalg.norm(q[:8][moving]-p[:8][moving],axis=-1) <= .01*np.hypot(*image.shape[:2])+tolerance).all()


def _point_parity(actual,expected,atol,label):
    if actual is None:
        # E.scored stores NaN slots for no detection; absence is equivalent here.
        assert expected is None or not np.isfinite(np.asarray(expected,np.float64)).any(),label
        return dict(max_abs_px=0.,atol_px=atol,rtol=0,finite_support_equal=True)
    actual, expected = np.asarray(actual,np.float64), np.asarray(expected,np.float64)
    assert actual.shape == expected.shape == (9,2),label
    finite = np.isfinite(expected)
    assert np.array_equal(np.isfinite(actual),finite),label
    error = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
    assert error <= atol,('Saved accuracy-coordinate parity failed; fixed tolerance',label,error,atol)
    return dict(max_abs_px=error,atol_px=atol,rtol=0,finite_support_equal=True)


def _pose_parity(actual,expected):
    assert actual['available'] == expected['available'],'Saved accuracy-pose availability parity'
    assert actual.get('selected_hypothesis') == expected.get('selected_hypothesis'),'Saved accuracy-pose hypothesis parity'
    differences = {}
    if actual['available']:
        for key in ('R_cf','R_physical','centroid','cf_extents','reprojection_px'):
            a,b = np.asarray(actual[key],np.float64),np.asarray(expected[key],np.float64)
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            differences[key] = float(np.max(np.abs(a-b)))
            assert differences[key] <= POINT_ATOL,('Saved accuracy-pose numeric parity failed',key,differences[key],POINT_ATOL)
    return dict(atol=POINT_ATOL,rtol=0,max_abs_by_field=differences)


def _interference():
    """Read-only checks; preserve competing jobs and mark timing pending."""
    fields = 'name,uuid,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit,clocks.sm,clocks.mem'
    status = subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader'],text=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    gpu_foreign = [r for r in apps.splitlines() if r.split(',')[0].strip() != str(os.getpid())
                   and '/usr/share/rustdesk/rustdesk' not in r]
    table = subprocess.check_output(['ps','-eo','pid,ppid,pcpu,comm,args','--no-headers'],text=True)
    ancestors, cursor = {os.getpid()}, os.getppid()
    parsed = []
    for line in table.splitlines():
        parts = line.strip().split(None,4)
        if len(parts) == 5:
            parsed.append((int(parts[0]),int(parts[1]),float(parts[2]),parts[3],parts[4]))
    parent = {p:pp for p,pp,_,_,_ in parsed}
    while cursor and cursor not in ancestors:
        ancestors.add(cursor); cursor = parent.get(cursor,0)
    cpu_foreign=[]
    for pid,ppid,cpu,comm,args in parsed:
        if pid in ancestors or ppid == os.getpid():
            continue
        if re.search(r'python|torchrun|jupyter',comm,re.I) and re.search(r'pallet|benchmark|train|inference|evaluate|runtime',args,re.I):
            # Save process identity without private command arguments or paths.
            cpu_foreign.append(dict(pid=pid,ppid=ppid,pcpu_lifetime=cpu,command=comm,
                                    reason='other Python research/training/inference/benchmark process'))
    return dict(gpu=status,fields=fields,compute_apps=[dict(pid=int(r.split(',')[0].strip()),
                process=Path(r.split(',')[1].strip()).name,used_memory=r.split(',')[2].strip()) for r in apps.splitlines() if r],
                foreign_gpu_pids=[int(r.split(',')[0].strip()) for r in gpu_foreign],
                foreign_cpu_workloads=cpu_foreign,quiet=not gpu_foreign and not cpu_foreign,
                gpu_temperature_under_80=float(status.split(',')[5].strip()) < 80,
                desktop_process_exception='rustdesk',system_changes=False,
                limits='process snapshots detect declared research jobs; transient interference between snapshots may escape detection')


def _durable(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    with pending.open('w') as stream:
        json.dump(C.finite(value),stream,allow_nan=False,sort_keys=True)
        stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def _save_rows(rows):
    path = C.DOC/'RUNTIME_ROWS.jsonl.gz'
    temporary = path.with_name(path.name+'.pending')
    with gzip.open(temporary,'wt',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(C.finite(row),allow_nan=False,separators=(',',':'))+'\n')
    temporary.replace(path)
    return binding(path)


def run_runtime(remaining_seconds=1800.):
    """Run once after accuracy PASS. A partial model/F attempt is never replayed."""
    import torch
    path, journal = C.DOC/'RUNTIME.json', C.OUTPUT/'RUNTIME_ATTEMPT.json'
    if path.exists():
        previous=C.read(path)
        assert previous.get('runtime_code_sha256') == C.sha(Path(__file__)),'Runtime code drift; do not overwrite prior run'
        return previous
    if journal.exists():
        raise RuntimeError('RUNTIME_PENDING: prior attempt incomplete; no hidden replay or additional F')
    assert remaining_seconds > 0,'RUNTIME_PENDING: no active execution budget remains'
    checks, protocol = C.DOC/'CHECKS.json', C.DOC/'PROTOCOL.json'
    assert checks.exists() and protocol.exists()
    assert C.read(checks).get('status') == 'PASS','Accuracy and minimum checks must complete first'
    selected,images,predictions,panel_bindings = _panel()
    references,head_references,reference_bindings = _accuracy_references(selected)
    sources=[binding(p) for p in (checks,protocol,Path(__file__),Path(M.__file__))]+panel_bindings+reference_bindings
    warm, measured = schedules()
    state = dict(status='STARTED',pipeline_calls_started=0,pipeline_calls_complete=0,
                 detector_calls=0,final_F_calls_started=0,final_F_calls_complete=0,
                 solvePnP=0,solvePnPGeneric=0,solvePnPRefineLM=0,cornerSubPix=0,max_pipeline_calls=MAX_PIPELINES)
    _durable(journal,state)
    rows,models,originals = [],None,{}
    started=time.monotonic()
    environment=dict(packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics','numpy','opencv-python','threadpoolctl')},
                     opencv=cv2.__version__,thread_contract=dict(torch=4,opencv=1),batch=1,interference_snapshots=[])
    result=dict(schema='n3_subpix_runtime_v1',status='FAILED_RUNTIME',complete=False,
        runtime_code_sha256=C.sha(Path(__file__)),arms=list(ARMS),panel=selected,panel_sessions=13,
        frames=FRAMES,warmup_per_arm=WARMUP,repeats=REPEATS,max_pipeline_calls=MAX_PIPELINES,
        schedule='cyclic route rotation, odd blocks reversed; fixed26 every route five measured repeats',
        boundaries=dict(full_pipeline='native uint8 BGR already in RAM -> YOLO preprocess/detector + FP16-rounded neck capture -> correction (including gray, transfers, and BASE-anchor total cap) -> original prediction-only F exactly once; synchronized wall',
            stage_only='detector output already available -> full route correction; synchronized contiguous subinterval of the full call',
            excluded=['file read/decode','model initialization/checkpoint load','GT metrics','parity checks','receipt writes','interference snapshots']),
        GT_inference_access=False,GT_keypoints_read=False,GT_pose_read=False,no_training=True,
        numeric_tolerance=dict(final_points_and_pose_absolute=POINT_ATOL,N3_intermediate_native_absolute=HEAD_POINT_ATOL,
            total_cap_native_px=dict(N3=HEAD_POINT_ATOL,SUBPIX=1e-10,N3_SUBPIX=1e-10),
            N3_cap_note='Original float32 N3 cap arithmetic preserved; historical native-pixel overshoot is recorded in accuracy checks',rtol=0),
        operating_budget_remaining_seconds=float(remaining_seconds),input_bindings=sources)
    try:
        probe=_interference();environment['interference_snapshots'].append(dict(phase='before_models',**probe))
        if not probe['quiet'] or not probe['gpu_temperature_under_80']:
            raise RuntimeError('RUNTIME_PENDING: competing workload or thermal condition; jobs preserved')
        assert torch.cuda.is_available()
        torch.set_num_threads(4);cv2.setNumThreads(1)
        torch.manual_seed(1);np.random.seed(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        environment['numeric']=dict(matmul_tf32=False,cudnn_tf32=True,cudnn_benchmark=False,cudnn_deterministic=False,
            detector='FP32, original cuDNN TF32=True; unchanged FP16-rounded neck',
            N3='original seed1 numeric contract, cuDNN TF32=True',
            CUDA_version=torch.version.cuda,cuDNN_version=torch.backends.cudnn.version())
        models=Pipeline();sources+=models.bindings;result['method_contracts']=models.contract
        environment['threadpools']=[{k:v for k,v in pool.items() if k!='filepath'} for pool in __import__('threadpoolctl').threadpool_info()]
        environment['actual_torch_threads']=torch.get_num_threads();environment['actual_opencv_threads']=cv2.getNumThreads()
        environment['device']=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
        torch.cuda.reset_peak_memory_stats()
        for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','cornerSubPix'):
            originals[name]=getattr(cv2,name)
            def wrapped(*args,_name=name,**kwargs):
                state[_name]+=1
                return originals[_name](*args,**kwargs)
            setattr(cv2,name,wrapped)
        last_probe=time.monotonic()
        with torch.no_grad():
            for job in warm+measured:
                if time.monotonic()-started > remaining_seconds:
                    raise RuntimeError('RUNTIME_PENDING: fixed active execution operating budget reached')
                if time.monotonic()-last_probe >= 5 or (job['phase']=='measured' and job['image_index']==0 and job['arm_position']==0):
                    probe=_interference();environment['interference_snapshots'].append(dict(job=job,**probe));last_probe=time.monotonic()
                    if not probe['quiet'] or not probe['gpu_temperature_under_80']:
                        raise RuntimeError('RUNTIME_PENDING: competing workload appeared; jobs preserved, contaminated timing not official')
                assert state['pipeline_calls_started'] < MAX_PIPELINES
                state['pipeline_calls_started']+=1;state['active_job']=job
                _durable(journal,state)
                i,arm=job['image_index'],job['arm'];image=images[i];frame=selected[i]
                before_counts={k:state[k] for k in originals}
                torch.cuda.synchronize();t0=time.perf_counter_ns()
                state['detector_calls']+=1
                captured=models.extractor.predict(image)
                torch.cuda.synchronize();t1=time.perf_counter_ns()
                prediction,diagnostic,qn=models.correct(arm,image,captured,frame)
                torch.cuda.synchronize();t2=time.perf_counter_ns()
                p,_=_raw(prediction)
                state['final_F_calls_started']+=1
                pose=models.pose.infer(p,np.asarray(frame['camera_intrinsics']),np.asarray(frame['dimensions_wdh_m'])[[0,2,1]],source=False)
                state['final_F_calls_complete']+=1
                torch.cuda.synchronize();t3=time.perf_counter_ns()
                raw=dict(candidates=models.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
                row=dict(**job,id=frame['frame_id'],session=frame['session_id'],full_ms=(t3-t0)/1e6,
                    detector_ms=(t1-t0)/1e6,stage_only_ms=None if arm=='BASE' else (t2-t1)/1e6,F_ms=(t3-t2)/1e6,
                    raw_points=_raw(raw)[0],N3_intermediate_points=qn,corrected_points=p,
                    selected_index=prediction['selected_index'],actual_pose=pose,
                    call_counts={k:state[k]-before_counts[k] for k in originals},correction_diagnostic=diagnostic,parity_status='PENDING')
                rows.append(row)
                _check_raw(raw,predictions[i]);_preserved(raw,prediction,image,arm)
                if arm in ('N3','N3_SUBPIX'):
                    row['N3_intermediate_parity']=_point_parity(qn,head_references[frame['frame_id']],HEAD_POINT_ATOL,(arm,frame['frame_id'],'N3'))
                ref=references[arm][frame['frame_id']]
                row['final_points_parity']=_point_parity(p,ref['points'],POINT_ATOL,(arm,frame['frame_id'],'final'))
                row['final_pose_parity']=_pose_parity(pose,ref['pose'])
                if diagnostic is not None:
                    assert diagnostic['algorithm_corner_calls']==row['call_counts']['cornerSubPix']
                else:
                    assert row['call_counts']['cornerSubPix']==0
                row.update(RAW_cached_bitexact=True,detection_and_center_missing_preserved=True,parity_status='PASS')
                state['pipeline_calls_complete']+=1;state['active_job']=None
                _durable(journal,state)
        assert state['pipeline_calls_complete']==len(rows)==MAX_PIPELINES
        assert state['detector_calls']==state['final_F_calls_complete']==state['pipeline_calls_started']==MAX_PIPELINES
        assert len([r for r in rows if r['phase']=='measured' and r['parity_status']=='PASS'])==520
        probe=_interference();environment['interference_snapshots'].append(dict(phase='after_complete',**probe))
        if not probe['quiet'] or not probe['gpu_temperature_under_80']:
            raise RuntimeError('RUNTIME_PENDING: endpoint interference; timing retained but not official')
        result.update(status='DONE',complete=True)
        environment['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        result.update(status='RUNTIME_PENDING' if 'RUNTIME_PENDING' in str(error) else 'FAILED_RUNTIME',complete=False,
                      reason=dict(type=type(error).__name__,message=str(error)))
    finally:
        for name,original in originals.items():
            setattr(cv2,name,original)
        if models is not None:
            result['execution_model_forwards']=dict(detector=models.detector_forwards,N3=models.n3_forwards,
                detector_internal_initialization_warmup=max(0,models.detector_forwards-state['detector_calls']),
                duplicate_N3_backbone_forwards=0,
                note='Hook includes Ultralytics internal model initialization; 600-budget covers explicit image pipeline calls')
            models.close()
        result['environment']=environment
        result['execution']=dict(state,full_measured_rows=sum(r['phase']=='measured' and r['parity_status']=='PASS' for r in rows),
            evaluation_F_calls_by_this_module=0,extra_parity_F_calls=0,intermediate_N3_F_calls=0,
            optimizer_updates=0,elapsed_seconds=time.monotonic()-started)
        result['summaries']={arm:dict(full_pipeline=describe([r['full_ms'] for r in rows if r['arm']==arm and r['phase']=='measured' and r['parity_status']=='PASS']),
            stage_only=describe([r['stage_only_ms'] for r in rows if r['arm']==arm and r['phase']=='measured' and r['stage_only_ms'] is not None and r['parity_status']=='PASS'])) for arm in ARMS}
        result['statistics_official']=bool(result['complete'])
        result['latency_deltas_ms']={f'N3_SUBPIX_minus_{arm}':dict(
            mean_ms=result['summaries']['N3_SUBPIX']['full_pipeline']['mean_ms']-result['summaries'][arm]['full_pipeline']['mean_ms'],
            median_ms=result['summaries']['N3_SUBPIX']['full_pipeline']['median_ms']-result['summaries'][arm]['full_pipeline']['median_ms'])
            for arm in ('BASE','N3','SUBPIX')} if result['complete'] else None
        result['raw_rows']=_save_rows(rows);result['input_bindings']=sources
        result['numeric_parity']=dict(checked_after_timing=True,new_detector_head_or_F_for_parity=0,
            RAW_bitexact_calls=sum(r.get('RAW_cached_bitexact',False) for r in rows),
            arms={arm:dict(checked_calls=sum(r['arm']==arm and r['parity_status']=='PASS' for r in rows),
                unique_frames=len({r['id'] for r in rows if r['arm']==arm and r['parity_status']=='PASS'}),
                max_abs_final_px=max((r['final_points_parity']['max_abs_px'] for r in rows if r['arm']==arm and 'final_points_parity' in r),default=None)) for arm in ARMS})
        result['warmup_accounting']={arm:dict(valid=sum(r['arm']==arm and r['phase']=='warmup' and r['parity_status']=='PASS' for r in rows),configured=20) for arm in ARMS}
        C.write(path,result);state['status']=result['status'];_durable(journal,state)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('schedule','measure'))
    parser.add_argument('--remaining-seconds',type=float,default=1800.)
    args=parser.parse_args()
    if args.stage=='schedule':
        a,b=schedules();print(json.dumps(dict(warmup=len(a),measured=len(b),arms=ARMS,total=len(a+b))))
    else:
        result=run_runtime(args.remaining_seconds)
        print(json.dumps(dict(status=result['status'],execution=result['execution'])))
