"""Actual E-conditioned correction-block cost on the existing fixed CAL6.

The E upstream is cached and excluded. This is not RGB-to-pose latency/FPS.
Shared observation+solve runs once per sample; each fixed guard is then timed.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import cv2
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1.instrumentation import instrument
from .guard import apply_guard
from .run import verify

def run(folder,source_root,prior):
    folder,source_root,prior=Path(folder),Path(source_root),Path(prior);verify(folder)
    frames=B.read(folder/'e_inputs/INPUTS.json')['frames'];panel=B.pilot_ids(frames)
    index={f['id']:(i,f) for i,f in enumerate(frames)}
    predictions={r['id']:r for r in B.rows(folder/'PREDICTIONS.jsonl.gz')}
    model=B.components(source_root,'cuda')
    torch.cuda.synchronize();begin=time.monotonic();samples=[];warmups=[]
    def execute(fid,warmup=False):
        i,f=index[fid];src=f['arms']['E_FROZEN'];im=cv2.imread(str(source_root/f['image']))
        if im is None or RT.sha256(source_root/f['image'])!=f['image_sha256']:raise ValueError('Same fixed RGB required')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with instrument(model) as meter:
            obs=RT.observe(im,src['native_points'],f['K'],src['coarse_pose'],components=model,
                prediction_source='E_FROZEN',record_id=fid)
            solved=RT.solve(obs['observation'],src['coarse_pose'],f['K'],f['raw_hw'])
            calls=meter.snapshot()
        torch.cuda.synchronize();common=time.perf_counter()-start
        outputs={};guard_seconds={}
        for mode in ('A','B'):
            tick=time.perf_counter();outputs[mode]=apply_guard(obs['observation'],solved,src['coarse_pose'],f['K'],mode)
            guard_seconds[mode]=time.perf_counter()-tick
        expected=B.read(folder/'e_run/poses'/f'{i:03d}_E_FROZEN.json')['result']
        B.compare(expected['actual_pose'],solved['actual_pose'],atol=1e-9,rtol=0)
        if expected['accepted']!=solved['accepted']:raise ValueError('Actual timed pose branch changed')
        saved=B.read(folder/'e_run/observations'/f'{i:03d}_E_FROZEN.json')['result']
        for key in ('states','frozen_visible_measurements','mask_audit','initial_geometry'):
            B.compare(saved['observation'][key],obs['observation'][key],atol=1e-8,rtol=1e-8)
        for mode in ('A','B'):
            expected_guard=predictions[fid]['predictions']['E'+mode]
            for key in ('accepted','fallback','status','reason','actual_pose'):
                B.compare(expected_guard[key],outputs[mode][key],atol=1e-9,rtol=0)
        record=dict(id=fid,warmup=warmup,observation_and_solve_seconds=common,guard_seconds=guard_seconds,
            total_A_seconds=common+guard_seconds['A'],total_B_seconds=common+guard_seconds['B'],
            actual_calls=calls,output_matches_accuracy=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (warmups if warmup else samples).append(record)
    for fid in panel[:2]:execute(fid,True)
    for repetition in range(2):
        for fid in panel:
            execute(fid);print('TIMED_E',len(samples),12,flush=True)
    def dist(values):
        a=np.asarray(values)*1000
        return dict(n=len(a),mean_ms=float(a.mean()),sample_std_ms=float(a.std(ddof=1)),
            median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)))
    output=dict(panel_ids=panel,warmups=warmups,samples=samples,repeats_per_image=2,warmups_count=2,batch=1,
        summary={m:dist([s['total_'+m+'_seconds'] for s in samples]) for m in ('A','B')},
        boundary='in-memory current RGB + own E-remapped image keypoints + cached E -> fresh state/masks/BI -> pose -> guard',
        file_decode_in_timing=False,model_loading_in_timing=False,CUDA_synchronized=True,
        full_RGB_to_pose_seconds=None,upstream_E_and_YOLO_N3_SubPix_not_executed=True,
        FPS_or_Jetson_claim=False,all_outputs_checked=True,environment=dict(GPU=torch.cuda.get_device_name(),
        torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,threads=1))
    B.write(folder/'timing/TIMING.json',output)
    B.Budget(folder).record('actual_E_correction_block_CAL6','gpu_process',time.monotonic()-begin,
        warmups=2,measured_calls=12,new_learning_or_seeds=False,status='complete')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();run(a.folder,a.source_root,a.prior)
