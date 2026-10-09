"""Sequential measurement of twelve actual complete paths, with saved-output parity."""
from __future__ import annotations

from collections import Counter
import copy
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from . import methods as M

C.source_modules()
from scripts.research.pallet_n3_subpix_20261008_v1 import runtime as R

WARMUP, REPEATS, FRAMES = 20, 5, 26
MAX_PIPELINES = len(C.ARMS) * (WARMUP + REPEATS * FRAMES)


class Pipeline(R.Pipeline):
    def correct(self, arm, image, captured, frame):
        if arm in C.CONTROLS:
            return super().correct(arm, image, captured, frame)
        raw = dict(candidates=self.inf.serial(captured['candidates']), selected_index=captured['selected_index'])
        q0, support = R._raw(raw)
        start, label, limit = arm.split('_')
        result, qn = raw, None
        if start == 'N3':
            result, _, qn = super().correct('N3', image, captured, frame)
        if q0 is None:
            return result, dict(no_detection=True, algorithm_corner_calls=0), qn
        seed = q0 if start == 'BASE' else qn
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        native, diagnostic = M.correct(gray, seed.copy(), support.copy(),
                                       'WIDE_SUBPIX' if label == 'WIDE' else 'BOUNDARY')
        final = native if limit == 'NATIVE' else C.cap_points(q0, native, support, image.shape[:2])
        result = copy.deepcopy(result)
        result['candidates'][result['selected_index']]['keypoints_xy'] = final
        diagnostic.update(cap_anchor='original_BASE', capped=limit == 'CAP1',
                          total_cap_px=.01*np.hypot(*image.shape[:2]))
        return result, diagnostic, qn


def schedules():
    warm, measured = [], []
    arms = C.ARMS
    for index in range(WARMUP):
        order = list(arms[index % len(arms):] + arms[:index % len(arms)])
        if index % 2: order.reverse()
        for position, arm in enumerate(order):
            warm.append(dict(phase='warmup', arm=arm, warmup_index=index,
                             arm_position=position, image_index=index % FRAMES))
    for repeat in range(REPEATS):
        order = list(arms[repeat % len(arms):] + arms[:repeat % len(arms)])
        if repeat % 2: order.reverse()
        for image_index in range(FRAMES):
            for position, arm in enumerate(order):
                measured.append(dict(phase='measured', arm=arm, repeat=repeat,
                                     arm_position=position, image_index=image_index))
    assert Counter(r['arm'] for r in warm) == {a: WARMUP for a in arms}
    assert Counter(r['arm'] for r in measured) == {a: 130 for a in arms}
    assert len(warm + measured) == MAX_PIPELINES == 1800
    return warm, measured


def preserved(before, after, image, arm):
    if arm in C.CONTROLS or arm.endswith('CAP1'):
        return R._preserved(before, after, image, arm)
    assert before['selected_index'] == after['selected_index']
    assert len(before['candidates']) == len(after['candidates'])
    for index, (a, b) in enumerate(zip(before['candidates'], after['candidates'])):
        assert set(a) == set(b)
        for key in a:
            if key != 'keypoints_xy':
                assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]), equal_nan=True), key
        p, q = np.asarray(a['keypoints_xy']), np.asarray(b['keypoints_xy'])
        valid = np.isfinite(p).all(1) & ~(p == -1).all(1)
        if index != before['selected_index']:
            assert np.array_equal(p, q, equal_nan=True)
        else:
            assert np.array_equal(p[8], q[8], equal_nan=True)
            assert np.array_equal(p[~valid], q[~valid], equal_nan=True)
            assert np.isfinite(q[valid]).all()


def main():
    import torch
    assert C.read(C.DOC/'CHECKS.json')['status'] == 'PASS'
    path, journal = C.DOC/'RUNTIME.json', C.SCRATCH/'RUNTIME_ATTEMPT.json'
    assert not path.exists() and not journal.exists(), 'Preserve completed/partial timing attempts'
    selected, images, cached_detections, panel_bindings = R._panel()
    refs = C.read(C.DOC/'RUNTIME_REFERENCES.json')
    assert refs['GT_coordinates'] is False and refs['GT_pose'] is False
    assert [r['frame_id'] for r in selected] == refs['panel_ids']
    assert all(len(refs['references'][a])==26 for a in C.ARMS)
    assert C.digest(M.method_configuration())==C.digest(C.read(C.DOC/'PROTOCOL.json')['methods'])
    probe = R._interference()
    if not probe['quiet'] or not probe['gpu_temperature_under_80']:
        C.write(C.SCRATCH/'RUNTIME_WAITING.json',dict(status='WAITING_FOR_QUIET',interference=probe,
                                                  model_calls=0,F_calls=0))
        raise RuntimeError('Competing workload detected; no timing/model attempts started')
    assert torch.cuda.is_available()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.manual_seed(1); np.random.seed(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
    state=dict(status='STARTED',pipeline_calls_started=0,pipeline_calls_complete=0,
               final_F_calls_started=0,final_F_calls_complete=0,
               solvePnP=0,solvePnPGeneric=0,solvePnPRefineLM=0,cornerSubPix=0)
    R._durable(journal,state)
    models,originals,rows=None,{},[]
    snapshots=[dict(phase='before_models',**probe)]
    started=time.monotonic()
    result=dict(schema='visible_boundary_actual_runtime',complete=False,status='FAILED',
        arms=list(C.ARMS),panel_frames=26,panel_sessions=13,warmup_each=20,measured_each=130,
        full_pipeline_definition='Original RAM BGR -> actual detector -> optional fixed N3 -> actual correction and cap -> original final F, synchronized wall time',
        excluded=['image decode/read','model load','GT scoring','parity checks','journaling'],
        numerical_contract='unchanged original seed1 detector/N3 TF32/FP16-rounded shared neck',
        configuration=C.binding(Path(M.__file__)),code=C.binding(Path(__file__)),
        references=C.binding(C.DOC/'RUNTIME_REFERENCES.json'),panel_bindings=panel_bindings,
        cached_coordinate_replay_used_for_timing=False,other_benchmarks_parallel=False)
    try:
        models=Pipeline()
        for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','cornerSubPix'):
            originals[name]=getattr(cv2,name)
            def wrapped(*args,_name=name,**kwargs):
                state[_name]+=1
                return originals[_name](*args,**kwargs)
            setattr(cv2,name,wrapped)
        last_probe=time.monotonic()
        warm, measured=schedules()
        with torch.no_grad():
            for job in warm+measured:
                if time.monotonic()-last_probe>5:
                    probe=R._interference();snapshots.append(dict(job=job,**probe));last_probe=time.monotonic()
                    if not probe['quiet'] or not probe['gpu_temperature_under_80']:
                        raise RuntimeError('Competing workload/thermal condition appeared; timing is incomplete')
                state['pipeline_calls_started']+=1;state['active_job']=job
                R._durable(journal,state)
                i,arm=job['image_index'],job['arm'];image=images[i];frame=selected[i]
                before={k:state[k] for k in originals}
                torch.cuda.synchronize();t0=time.perf_counter_ns()
                captured=models.extractor.predict(image)
                torch.cuda.synchronize();t1=time.perf_counter_ns()
                prediction,diagnostic,qn=models.correct(arm,image,captured,frame)
                torch.cuda.synchronize();t2=time.perf_counter_ns()
                q,_=R._raw(prediction)
                state['final_F_calls_started']+=1
                pose=models.pose.infer(q,np.asarray(frame['camera_intrinsics']),
                    np.asarray(frame['dimensions_wdh_m'])[[0,2,1]],source=False)
                state['final_F_calls_complete']+=1
                torch.cuda.synchronize();t3=time.perf_counter_ns()
                raw=dict(candidates=models.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
                row=dict(**job,id=frame['frame_id'],session=frame['session_id'],
                    full_ms=(t3-t0)/1e6,detector_ms=(t1-t0)/1e6,
                    correction_ms=(t2-t1)/1e6,F_ms=(t3-t2)/1e6,
                    corrected_points=q,actual_pose=pose,
                    call_counts={k:state[k]-before[k] for k in originals},parity_status='PENDING')
                rows.append(row)
                R._check_raw(raw,cached_detections[i]);preserved(raw,prediction,image,arm)
                if arm.startswith('N3'):
                    row['N3_intermediate_parity']=R._point_parity(qn,refs['head_references'][frame['frame_id']],1e-4,(arm,frame['frame_id'],'N3'))
                reference=refs['references'][arm][frame['frame_id']]
                row['final_points_parity']=R._point_parity(q,reference['points'],1e-7,(arm,frame['frame_id'],'final'))
                row['final_pose_parity']=R._pose_parity(pose,reference['pose'])
                assert row['call_counts']['cornerSubPix']==(diagnostic or {}).get('algorithm_corner_calls',0)
                row['parity_status']='PASS';state['pipeline_calls_complete']+=1;state['active_job']=None
                R._durable(journal,state)
                if state['pipeline_calls_complete']%180==0:
                    print('RUNTIME',state['pipeline_calls_complete'],MAX_PIPELINES,'seconds',round(time.monotonic()-started,2),flush=True)
        assert state['pipeline_calls_complete']==state['final_F_calls_complete']==len(rows)==1800
        probe=R._interference();snapshots.append(dict(phase='after_complete',**probe))
        assert probe['quiet'] and probe['gpu_temperature_under_80']
        result.update(complete=True,status='DONE')
    except Exception as exc:
        result.update(reason=dict(type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        for name,original in originals.items():setattr(cv2,name,original)
        if models is not None:
            result['model_forwards']=dict(detector=models.detector_forwards,N3=models.n3_forwards,
                initialization_detector_calls=max(0,models.detector_forwards-state['pipeline_calls_started']),
                duplicate_backbone_forwards=0)
            result['model_bindings']=models.bindings;models.close()
        result['environment']=dict(opencv=cv2.__version__,torch=torch.__version__,
            device=torch.cuda.get_device_name(),torch_threads=torch.get_num_threads(),
            opencv_threads=cv2.getNumThreads(),interference_snapshots=snapshots,
            matmul_tf32=False,cudnn_tf32=True,batch=1)
        state['status']=result['status']
        result['execution']=dict(state,elapsed_seconds=time.monotonic()-started,training_updates=0)
        result['statistics_official']=result['complete']
        result['summaries']={arm:dict(
            full_pipeline=R.describe([r['full_ms'] for r in rows if r['arm']==arm and r['phase']=='measured' and r['parity_status']=='PASS']),
            correction=R.describe([r['correction_ms'] for r in rows if r['arm']==arm and r['phase']=='measured' and r['parity_status']=='PASS'])) for arm in C.ARMS}
        rawpath=C.DOC/'RUNTIME_ROWS.jsonl.gz'
        with gzip.open(rawpath,'wt',encoding='utf-8') as stream:
            for row in rows:stream.write(json.dumps(C.finite(row),allow_nan=False,separators=(',',':'))+'\n')
        result['raw_rows']=C.binding(rawpath)
        C.write(path,result);state['status']=result['status'];R._durable(journal,state)
    print('RUNTIME_COMPLETE',json.dumps(result['execution']),flush=True)


if __name__=='__main__':
    main()
