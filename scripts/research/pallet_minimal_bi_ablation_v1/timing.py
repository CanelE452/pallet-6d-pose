"""Actual fixed-panel correction-block timing; upstream full timing is NA."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import platform
import time
import traceback

import cv2
import numpy as np
import torch

from . import runtime as RT
from .batch import (ARMS, Budget, components, read, write, _input,
                    check_binding, compare)
from .instrumentation import instrument


def _synchronize(device):
    if device.startswith('cuda'):torch.cuda.synchronize()


def measure(root,input_dir,folder,*,device='cuda'):
    folder=Path(folder);config=check_binding(root,folder);inputs,_=_input(input_dir,config)
    if not (folder/'PREDICTION_SEAL.json').exists():raise ValueError('Complete accuracy prediction seal required')
    prediction_seal=read(folder/'PREDICTION_SEAL.json')
    if (prediction_seal['predictions_sha256']!=RT.sha256(folder/'PREDICTIONS.jsonl.gz') or
            prediction_seal['protocol_sha256']!=RT.sha256(folder/'PROTOCOL.json') or
            prediction_seal['rows']!=238 or prediction_seal['arm_rows']!=952):
        raise ValueError('Complete accuracy seal byte binding changed')
    out=folder/'timing';out.mkdir(exist_ok=False)
    panel=[(i,f) for i,f in enumerate(inputs['frames']) if f['id'] in config['pilot_ids']]
    if len(panel)!=6:raise ValueError('Exact fixed six-image panel required')
    budget=Budget(folder.parent);budget.check('gpu_process');start=time.monotonic();status='technical_incomplete'
    try:
        model=components(root,device)
        # Decode/read once outside all timed calls, including accuracy comparators.
        images={f['id']:cv2.imread(str(Path(root)/f['image'])) for _,f in panel}
        accuracy={}
        for i,f in panel:
            if images[f['id']] is None:raise ValueError('Panel decode failed')
            for b in ('RAW_YOLO','N3_SUBPIX'):
                accuracy[(f['id'],b)]=dict(
                    observation=read(folder/'observations'/f'{i:03d}_{b}.json')['result'],
                    pose=read(folder/'poses'/f'{i:03d}_{b}.json')['result'])
        def call(f,arm):
            b='RAW_YOLO' if arm.startswith('R') else 'N3_SUBPIX'
            source=f['arms'][b]
            if arm.endswith('0'):
                return None,dict(actual_pose=deepcopy(source['coarse_pose']),accepted=None,fallback=None)
            observation=RT.observe(images[f['id']],source['native_points'],f['K'],source['coarse_pose'],
                components=model,prediction_source=b,record_id=f['id'])
            pose=RT.solve(observation['observation'],source['coarse_pose'],f['K'],f['raw_hw'])
            return observation,pose
        samples=[];warmups=[]
        with instrument(model) as meter:
            for arm in ARMS:
                for repetition in range(2):
                    f=panel[repetition%len(panel)][1]
                    meter.reset();_synchronize(device);tick=time.perf_counter()
                    observation,pose=call(f,arm);_synchronize(device)
                    warmups.append(dict(id=f['id'],arm=arm,repetition=repetition,
                        seconds=time.perf_counter()-tick,counts=meter.snapshot()))
            rng=np.random.default_rng(20260917)
            # Each image/repetition has exactly all four paths, with alternating
            # randomized pair order. No best-repeat selection or frame trimming.
            schedules=[]
            for repetition in range(5):
                for position,(_,f) in enumerate(panel):
                    base=list(ARMS)
                    if int(rng.integers(0,2)):base.reverse()
                    shift=(repetition+position)%4
                    base=base[shift:]+base[:shift]
                    schedules.append((repetition,f,base))
            for repetition,f,order in schedules:
                for arm in order:
                    meter.reset()
                    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                    _synchronize(device);tick=time.perf_counter()
                    observation,pose=call(f,arm);_synchronize(device)
                    elapsed=time.perf_counter()-tick
                    counts=meter.snapshot()
                    b='RAW_YOLO' if arm.startswith('R') else 'N3_SUBPIX'
                    expected=accuracy[(f['id'],b)]
                    if arm.endswith('1'):
                        compare(expected['pose'],pose,'timed_final_pose',atol=1e-9,rtol=0)
                        for key in ('states','initial_geometry','state_observation','mask_audit','frozen_visible_measurements'):
                            compare(expected['observation']['observation'][key],observation['observation'][key],
                                'timed_observation/'+key,atol=1e-8,rtol=1e-8)
                    else:compare(f['arms'][b]['coarse_pose'],pose['actual_pose'],'baseline_pass_through')
                    samples.append(dict(id=f['id'],arm=arm,repetition=repetition,path_order=order,
                        elapsed_seconds=elapsed,counts=counts,
                        final_pose_matches_accuracy=True,selected_hypothesis_and_acceptance_match=True,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else None))
                    print('TIME',len(samples),'/120',arm,f['id'],f'{elapsed:.3f}s',flush=True)
                    budget.check('gpu_process',time.monotonic()-start)
        summary={}
        for arm in ARMS:
            selected=[s for s in samples if s['arm']==arm]
            values=np.array([s['elapsed_seconds'] for s in selected])
            counts=Counter()
            for s in selected:
                counts.update({k:v for k,v in s['counts'].items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
            summary[arm]=dict(n=len(selected),mean_ms=float(values.mean()*1000),
                sample_sd_ms=float(values.std(ddof=1)*1000),median_ms=float(np.median(values)*1000),
                p95_ms=float(np.percentile(values,95)*1000),min_ms=float(values.min()*1000),max_ms=float(values.max()*1000),
                actual_call_counts=dict(counts),
                peak_allocated_bytes=max(s['gpu_peak_allocated_bytes'] or 0 for s in selected),
                peak_reserved_bytes=max(s['gpu_peak_reserved_bytes'] or 0 for s in selected),
                boundary='own F pose pass-through API overhead only' if arm.endswith('0') else 'actual complete correction block',
                full_RGB_inference_seconds=None)
        result=dict(schema='MINIMAL_BI_ACTUAL_FIXED_PANEL_COST_V1',status='complete',
            panel_ids=config['pilot_ids'],batch=1,warmups_per_path=2,repeats_per_image=5,
            random_order_seed=20260917,summary=summary,samples=samples,warmups=warmups,
            GPU_process_elapsed_seconds=time.monotonic()-start,
            environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,
                torch=torch.__version__,CUDA=torch.version.cuda,
                GPU=torch.cuda.get_device_name() if device.startswith('cuda') else None,
                precision='float32 neural inference; float64 geometry/CPU solver; no autocast/TF32',
                torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),BLAS_OpenMP_threads=1),
            correction_block_boundary='in-memory BGR + own q + own F -> own state/masks/BI -> final pose',
            decoded_files_in_timed_calls=0,model_loading_in_measured_calls=False,
            CUDA_synchronized=True,model_loading_may_be_in_first_excluded_warmup=True,
            full_RGB_inference_time='NA',missing_full_scope='YOLO, N3/SubPix and initial F not executed inside timed call',
            no_real_time_or_Jetson_claim=True,all_final_outputs_compared_to_accuracy=True)
        write(out/'TIMING.json',result);status='complete'
    except Exception:
        (out/'TECHNICAL_FAILURE.txt').write_text(traceback.format_exc());raise
    finally:
        budget.record('fixed6_actual_timing','gpu_process',time.monotonic()-start,status=status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    measure(a.root,a.input_dir,a.output,device=a.device)
