"""Five-route fixed26 timing wrapper around the unchanged 20261008 runtime.

The old model wrapper, resource checks, parity arithmetic and latency statistics
are reused read-only. This module never invokes the old runtime entrypoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from scripts.research.pallet_n3_subpix_20261008_v1 import runtime as R

ARMS = ('BASE','N3','SUBPIX','N3_SUBPIX','SUBPIX_N3')
FRAMES, WARMUP, REPEATS, MAX_PIPELINES = 26, 20, 5, 750
GRADES = ('clean','moderate','severe')
PRIOR = C.ROOT / '_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz'
LABELS = C.ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json'


def binding(path):
    path=Path(path).resolve()
    for root,origin in ((C.DOC.parents[2].resolve(),'experiment'),(C.ROOT.resolve(),'source')):
        if path.is_relative_to(root):
            return dict(path=str(path.relative_to(root)),origin=origin,sha256=C.sha(path),bytes=path.stat().st_size)
    return dict(path=path.name,origin='immutable_baseline_dependency',sha256=C.sha(path),bytes=path.stat().st_size)


def schedules():
    """Same predetermined cyclic/reversed blocks, extended to five routes."""
    jobs=[]
    for phase,blocks in (('warmup',WARMUP),('measured',REPEATS)):
        for block in range(blocks):
            order=list(ARMS[block % 5:]+ARMS[:block % 5])
            if block % 2: order.reverse()
            for index in ([block % FRAMES] if phase=='warmup' else range(FRAMES)):
                for position,arm in enumerate(order):
                    key='warmup_index' if phase=='warmup' else 'repeat'
                    jobs.append(dict(phase=phase,arm=arm,image_index=index,arm_position=position,**{key:block}))
    warm=[r for r in jobs if r['phase']=='warmup'];measured=[r for r in jobs if r['phase']=='measured']
    assert Counter(r['arm'] for r in warm)=={a:20 for a in ARMS}
    assert Counter(r['arm'] for r in measured)=={a:130 for a in ARMS}
    assert len(jobs)==MAX_PIPELINES
    return warm,measured


class Pipeline(R.Pipeline):
    """Four unchanged routes plus one actual SUBPIX-input N3 route."""
    def correct(self,arm,image,captured,row):
        if arm != 'SUBPIX_N3': return super().correct(arm,image,captured,row)
        from .reverse import reverse_captured
        dims,order=self.inf.registry_input(row['object_type'])
        np.testing.assert_array_equal(dims,row['dimensions_wdh_m'])
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        result,diagnostic,qsn=reverse_captured(self.inf,self.n3,captured,dims,order,
            self.spec['temperature'],self.spec['rule'],image.shape[:2],self.norm,gray,diagnostics=False)
        self.n3_forwards+=int(result['head_used'])
        return result,diagnostic,qsn


def _references(selected):
    refs={a:{} for a in ARMS}
    reverse=C.DOC/'PREDICTIONS.jsonl.gz'
    for path in (PRIOR,reverse):
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);arm=row.get('method',row.get('arm'))
                if arm in refs:
                    assert row['id'] not in refs[arm],('Duplicate cached reference',arm,row['id'])
                    refs[arm][row['id']]=dict(points=row['native_points'],pose=row.get('actual_pose'))
    assert all(len(v)==319 for v in refs.values())
    assert all(r['frame_id'] in refs[a] and refs[a][r['frame_id']]['pose'] is not None for r in selected for a in ARMS)
    n3=C.read(R.N3_POINTS)
    assert n3['complete'] and n3['GT_input'] is False
    assert n3['checkpoint']['sha256']==C.read(R.N3_SELECTION)['temperatures']['N3_DIM_SYM_seed1']['checkpoint']['sha256']
    heads={r['id']:R._raw(r)[0] for r in n3['records']}
    assert len(heads)==319
    labels=[r for r in C.read(LABELS)['rows'] if r['population']=='DEV319']
    grades={r['id']:r['severity'] for r in labels}
    assert len(grades)==len(labels)==319 and set(grades)==set(refs['BASE'])
    assert Counter(grades.values())=={'clean':153,'moderate':92,'severe':74}
    return refs,heads,grades,[binding(p) for p in (PRIOR,reverse,R.N3_POINTS,LABELS)]


def _save_rows(rows):
    path=C.DOC/'RUNTIME_ROWS.jsonl.gz';pending=path.with_name(path.name+'.pending')
    with gzip.open(pending,'wt',encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(C.finite(row),allow_nan=False,separators=(',',':'))+'\n')
    pending.replace(path)
    return binding(path)


def _summary(rows,arm,grade=None):
    selected=[r for r in rows if r['arm']==arm and r['phase']=='measured' and r['parity_status']=='PASS' and (grade is None or r['grade']==grade)]
    return dict(full_pipeline=R.describe([r['full_ms'] for r in selected]),
        stage_only=R.describe([r['stage_only_ms'] for r in selected if r['stage_only_ms'] is not None]),
        unique_frames=len({r['id'] for r in selected}),sessions=len({r['session'] for r in selected}))


def run_runtime(remaining_seconds=1800.):
    """One attempt, at most750 image pipelines; no intermediate or parity F."""
    import torch
    path,journal=C.DOC/'RUNTIME.json',C.OUTPUT/'RUNTIME_ATTEMPT.json'
    if path.exists():
        previous=C.read(path)
        assert previous.get('runtime_code_sha256')==C.sha(Path(__file__)),'Runtime code drift; preserve prior attempt'
        return previous
    assert not journal.exists(),'RUNTIME_PENDING: prior partial attempt must not be silently replayed'
    assert remaining_seconds>0,'RUNTIME_PENDING: no active execution budget remains'
    checks,protocol=C.DOC/'CHECKS.json',C.DOC/'PROTOCOL.json'
    assert C.read(checks).get('status')=='PASS','Accuracy minimum checks must pass first'
    selected,images,predictions,panel_bindings=R._panel()
    refs,heads,grades,ref_bindings=_references(selected)
    sources=[binding(p) for p in (checks,protocol,__file__,R.__file__,C.__file__,Path(__file__).with_name('reverse.py'))]+panel_bindings+ref_bindings
    warm,measured=schedules();rows=[];models=None;originals={};started=time.monotonic()
    state=dict(status='STARTED',pipeline_calls_started=0,pipeline_calls_complete=0,detector_calls=0,
        final_F_calls_started=0,final_F_calls_complete=0,solvePnP=0,solvePnPGeneric=0,solvePnPRefineLM=0,cornerSubPix=0,max_pipeline_calls=MAX_PIPELINES)
    R._durable(journal,state)
    environment=dict(packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics','numpy','opencv-python','threadpoolctl')},
        opencv=cv2.__version__,thread_contract=dict(torch=4,opencv=1),batch=1,interference_snapshots=[])
    panel_grades=Counter(grades[r['frame_id']] for r in selected)
    result=dict(schema='subpix_order_runtime_v1',status='FAILED_RUNTIME',complete=False,runtime_code_sha256=C.sha(Path(__file__)),
        arms=list(ARMS),panel=selected,panel_sessions=13,frames=26,warmup_per_arm=20,repeats=5,max_pipeline_calls=750,
        schedule='fixed cyclic rotation; odd blocks reversed; every fixed26 frame/route five measured repeats',
        boundaries=dict(full_pipeline='RAM native BGR -> one original YOLO forward/selection and FP16-rounded shared neck -> route correction, grayscale/transfers/caps -> final original F exactly once; synchronized wall',
            stage_only='contiguous synchronized correction subinterval of the full call',
            excluded=['image read/decode','checkpoint/model load','GT scoring','parity checks','receipt writes','interference snapshots']),
        prior_runtime_numbers_reused=False,GT_inference_access=False,GT_keypoints_read=False,GT_pose_read=False,grade_inference_input=False,no_training=True,
        numeric_tolerance=dict(final_points_and_pose_absolute=R.POINT_ATOL,N3_intermediate_native_absolute=R.HEAD_POINT_ATOL,
            total_cap_native_px=dict(N3=R.HEAD_POINT_ATOL,SUBPIX=1e-10,N3_SUBPIX=1e-10,SUBPIX_N3=1e-10),rtol=0),
        operating_budget_remaining_seconds=float(remaining_seconds),input_bindings=sources,
        grade_panel=dict(source=binding(LABELS),counts=dict(panel_grades),missing_grades=[g for g in GRADES if not panel_grades[g]],
            no_additional_sampling=True,interpretation='latency on fixed26 grade subsets; no grade-driven correction or route selection'))
    try:
        probe=R._interference();environment['interference_snapshots'].append(dict(phase='before_models',**probe))
        if not probe['quiet'] or not probe['gpu_temperature_under_80']: raise RuntimeError('RUNTIME_PENDING: competing workload or thermal condition; jobs preserved')
        assert torch.cuda.is_available()
        torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(1);np.random.seed(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        environment['numeric']=dict(matmul_tf32=False,cudnn_tf32=True,cudnn_benchmark=False,cudnn_deterministic=False,
            CUDA_version=torch.version.cuda,cuDNN_version=torch.backends.cudnn.version(),N3='unchanged seed1 and shared FP16-rounded neck')
        models=Pipeline();sources+=models.bindings;result['method_contracts']=models.contract
        result['method_contracts']['SUBPIX_N3']='SUBPIX CAP1 q0 -> actual predict_captured with selected coordinates replaced, shared original features -> q0 total CAP1'
        environment['threadpools']=[{k:v for k,v in p.items() if k!='filepath'} for p in __import__('threadpoolctl').threadpool_info()]
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
                if time.monotonic()-started>remaining_seconds: raise RuntimeError('RUNTIME_PENDING: fixed active operating budget reached')
                if time.monotonic()-last_probe>=5 or (job['phase']=='measured' and job['image_index']==0 and job['arm_position']==0):
                    probe=R._interference();environment['interference_snapshots'].append(dict(job=job,**probe));last_probe=time.monotonic()
                    if not probe['quiet'] or not probe['gpu_temperature_under_80']: raise RuntimeError('RUNTIME_PENDING: competing workload appeared; timings retained but not official')
                assert state['pipeline_calls_started']<MAX_PIPELINES
                state['pipeline_calls_started']+=1;state['active_job']=job;R._durable(journal,state)
                i,arm=job['image_index'],job['arm'];image=images[i];frame=selected[i];before={k:state[k] for k in originals}
                torch.cuda.synchronize();t0=time.perf_counter_ns();state['detector_calls']+=1
                captured=models.extractor.predict(image)
                torch.cuda.synchronize();t1=time.perf_counter_ns()
                prediction,diagnostic,qn=models.correct(arm,image,captured,frame)
                torch.cuda.synchronize();t2=time.perf_counter_ns();p,_=R._raw(prediction)
                state['final_F_calls_started']+=1
                pose=models.pose.infer(p,np.asarray(frame['camera_intrinsics']),np.asarray(frame['dimensions_wdh_m'])[[0,2,1]],source=False)
                state['final_F_calls_complete']+=1;torch.cuda.synchronize();t3=time.perf_counter_ns()
                raw=dict(candidates=models.inf.serial(captured['candidates']),selected_index=captured['selected_index'])
                row=dict(**job,id=frame['frame_id'],session=frame['session_id'],grade=grades[frame['frame_id']],full_ms=(t3-t0)/1e6,
                    detector_ms=(t1-t0)/1e6,stage_only_ms=None if arm=='BASE' else (t2-t1)/1e6,F_ms=(t3-t2)/1e6,
                    raw_points=R._raw(raw)[0],N3_intermediate_points=qn,corrected_points=p,actual_pose=pose,
                    selected_index=prediction['selected_index'],call_counts={k:state[k]-before[k] for k in originals},correction_diagnostic=diagnostic,parity_status='PENDING')
                rows.append(row);R._check_raw(raw,predictions[i]);R._preserved(raw,prediction,image,arm)
                if arm in ('N3','N3_SUBPIX'): row['N3_intermediate_parity']=R._point_parity(qn,heads[frame['frame_id']],R.HEAD_POINT_ATOL,(arm,frame['frame_id'],'N3'))
                ref=refs[arm][frame['frame_id']]
                row['final_points_parity']=R._point_parity(p,ref['points'],R.POINT_ATOL,(arm,frame['frame_id'],'final'))
                row['final_pose_parity']=R._pose_parity(pose,ref['pose'])
                assert (diagnostic['algorithm_corner_calls'] if diagnostic else 0)==row['call_counts']['cornerSubPix']
                row.update(RAW_cached_bitexact=True,detection_and_center_missing_preserved=True,parity_status='PASS')
                state['pipeline_calls_complete']+=1;state['active_job']=None;R._durable(journal,state)
        assert state['pipeline_calls_complete']==len(rows)==state['detector_calls']==state['final_F_calls_complete']==MAX_PIPELINES
        assert sum(r['phase']=='measured' for r in rows)==650
        probe=R._interference();environment['interference_snapshots'].append(dict(phase='after_complete',**probe))
        if not probe['quiet'] or not probe['gpu_temperature_under_80']: raise RuntimeError('RUNTIME_PENDING: endpoint interference; timings retained but not official')
        result.update(status='DONE',complete=True);environment['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    except Exception as error:
        result.update(status='RUNTIME_PENDING' if 'RUNTIME_PENDING' in str(error) else 'FAILED_RUNTIME',complete=False,reason=dict(type=type(error).__name__,message=str(error)))
    finally:
        for name,original in originals.items(): setattr(cv2,name,original)
        if models is not None:
            result['execution_model_forwards']=dict(detector=models.detector_forwards,N3=models.n3_forwards,
                detector_internal_initialization_warmup=max(0,models.detector_forwards-state['detector_calls']),duplicate_N3_backbone_forwards=0,
                note='Module hook includes internal Ultralytics initialization; 750 ceiling counts explicit image pipelines')
            models.close()
        result['environment']=environment
        result['execution']=dict(state,full_measured_rows=sum(r['phase']=='measured' and r['parity_status']=='PASS' for r in rows),
            evaluation_F_calls_by_this_module=0,extra_parity_F_calls=0,intermediate_N3_F_calls=0,optimizer_updates=0,elapsed_seconds=time.monotonic()-started)
        result['summaries']={a:_summary(rows,a) for a in ARMS}
        result['grade_summaries']={g:{a:_summary(rows,a,g) for a in ARMS} for g in GRADES}
        result['statistics_official']=bool(result['complete'])
        result['latency_deltas_ms']={f'{after}_minus_{before}':{f:result['summaries'][after]['full_pipeline'][f]-result['summaries'][before]['full_pipeline'][f] for f in ('mean_ms','median_ms')}
            for after,before in (('SUBPIX_N3','N3'),('SUBPIX_N3','SUBPIX'),('SUBPIX_N3','N3_SUBPIX'),('N3_SUBPIX','N3'),('N3_SUBPIX','SUBPIX'))} if result['complete'] else None
        result['numeric_parity']=dict(checked_after_timing=True,new_detector_head_or_F_for_parity=0,
            RAW_bitexact_calls=sum(r.get('RAW_cached_bitexact',False) for r in rows),
            arms={a:dict(checked_calls=sum(r['arm']==a and r['parity_status']=='PASS' for r in rows),
                unique_frames=len({r['id'] for r in rows if r['arm']==a and r['parity_status']=='PASS'}),
                max_abs_final_px=max((r['final_points_parity']['max_abs_px'] for r in rows if r['arm']==a and 'final_points_parity' in r),default=None)) for a in ARMS})
        result['warmup_accounting']={a:dict(valid=sum(r['arm']==a and r['phase']=='warmup' and r['parity_status']=='PASS' for r in rows),configured=20) for a in ARMS}
        result['raw_rows']=_save_rows(rows);result['input_bindings']=sources
        C.write(path,result);state['status']=result['status'];R._durable(journal,state)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('schedule','measure'))
    parser.add_argument('--remaining-seconds',type=float,default=1800.);args=parser.parse_args()
    if args.stage=='schedule':
        warm,measured=schedules();print(json.dumps(dict(warmup=len(warm),measured=len(measured),arms=ARMS,total=len(warm+measured))))
    else:
        result=run_runtime(args.remaining_seconds);print(json.dumps(dict(status=result['status'],execution=result['execution'])))
