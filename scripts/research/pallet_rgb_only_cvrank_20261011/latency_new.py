"""New-route benchmark: unchanged prior 18-image/20-warmup/5-repeat protocol."""
import argparse, gzip, hashlib, importlib, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
import rgb_model
from latency import (clean, sha, write, moments, sync, install_pose_counter, counted_run, parity)
METHODS=("RGB_ONLY","RGB_ONLY_SUBPIX","CVRANK_NATIVE","CVRANK_CAP1","CVRANK_THEN_SUBPIX")

def route(args):
    doc=Path(args.public_dir);private=Path(args.private_dir);out=doc/f"LATENCY_{args.method}.json"
    assert not out.exists(),"Existing benchmark receipt preserved"
    runtime_path=private/"COMMON_INPUTS.npz";runtime=np.load(runtime_path,allow_pickle=False)
    reference=json.loads((Path(args.prior_public_dir)/'LATENCY_BASE.json').read_text())
    lookup={fid:k for k,fid in enumerate(runtime['ids'].tolist())}
    selected=[(fid,lookup[fid]) for fid in reference['panel_ids']]
    assert len(selected)==18 and reference['sessions']==9
    images={};decode_start=time.perf_counter()
    for fid,k in selected:
        image_path=Path(str(runtime["image_paths"][k]))
        if not image_path.is_absolute():image_path=Path(args.source_root)/image_path
        assert sha(image_path)==str(runtime["image_sha256"][k])
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        assert image is not None and tuple(image.shape[:2])==tuple(runtime["raw_hw"][k])
        images[fid]=image
    decode_seconds=time.perf_counter()-decode_start
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root));sys.path.insert(0,str(Path(__file__).resolve().parent))
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(),"GPU benchmark required"
    module=importlib.import_module("inference_new")
    active_start=time.perf_counter();engine=module.InferenceEngine(args,seed=1);sync()
    load_seconds=time.perf_counter()-active_start
    torch.cuda.reset_peak_memory_stats()
    memory_after_load=torch.cuda.memory_allocated()
    call_receipts=[];captures=[];latencies=[];warm_start=time.perf_counter()
    pose_counts,original_cv=install_pose_counter()
    for repetition in range(20):
        fid,k=selected[repetition%len(selected)]
        result=counted_run(engine,images[fid],args.method,np.array(runtime["K"][k],dtype=np.float64),np.array(runtime["xyz"][k],dtype=np.float64),pose_counts)
        result=clean(result)
        sync();call_receipts.append(dict(phase="WARMUP",id=fid,repetition=repetition+1,counters=result.get("counters"),available=bool(result.get("actual_pose",{}).get("available",False))))
    warm_seconds=time.perf_counter()-warm_start
    for fid,k in selected:
        for repetition in range(5):
            sync();start=time.perf_counter()
            result=counted_run(engine,images[fid],args.method,np.array(runtime["K"][k],dtype=np.float64),np.array(runtime["xyz"][k],dtype=np.float64),pose_counts)
            sync();elapsed=(time.perf_counter()-start)*1000;latencies.append(elapsed)
            result=clean(result)
            record=dict(phase="TIMED",id=fid,repetition=repetition+1,latency_ms=elapsed,counters=result.get("counters"),
                        available=bool(result.get("actual_pose",{}).get("available",False)))
            call_receipts.append(record)
            captures.append(dict(method=args.method,seed=1,id=fid,repetition=repetition+1,points=np.asarray(result["points"]).tolist(),
                                 actual_pose=result.get("actual_pose"),counters=result.get("counters")))
    sync();active_seconds=time.perf_counter()-active_start
    peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
    for name,fn in original_cv.items():setattr(cv2,name,fn)
    metadata=engine.metadata(args.method) if hasattr(engine,"metadata") else {}
    if not metadata and hasattr(engine,"parameter_counts"):
        metadata=engine.parameter_counts.get(args.method,{})
    capture_path=private/f"LATENCY_{args.method}_CAPTURES.jsonl.gz"
    assert not capture_path.exists()
    with gzip.open(capture_path,"xt",encoding="utf-8") as f:
        for capture in captures:f.write(json.dumps(capture,allow_nan=False)+"\n")
    accuracy=parity(captures,args.accuracy_predictions,args.method)
    panel_sha=hashlib.sha256(json.dumps([fid for fid,_ in selected],separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    receipt=dict(status="COMPLETE",method=args.method,seed=1,frames=18,sessions=9,batch=1,warmups=20,repetitions_per_image=5,timed_calls=90,
                 panel_ids=[fid for fid,_ in selected],panel_id_sha256=panel_sha,latency_ms=moments(latencies),
                 per_call=call_receipts,missing_pose_timed_calls=sum(not r["available"] for r in call_receipts if r["phase"]=="TIMED"),
                 peak_memory_MiB=peak_allocated/(1024**2),peak_reserved_MiB=peak_reserved/(1024**2),allocated_after_load_MiB=memory_after_load/(1024**2),
                 total_params=metadata.get("total_params"),additional_params=metadata.get("additional_params"),parameter_metadata=metadata,
                 precision=dict(dtype="FP32",autocast=False,matmul_TF32=False,cudnn_TF32=False,cudnn_benchmark=False),
                 inherited_neck_capture="Common feature-based routes: frozen p3/p4 capture FP16 rounding, then features.float() in FP32 corrector; PoseFix native RGB FP32",
                 boundary="Native image ndarray -> preprocessing / YOLO / native RGB tiles / corrector / SubPix / final PnP; excludes file decode and model load",
                 model_load_seconds_excluded=load_seconds,image_decode_seconds_excluded=decode_seconds,warmup_seconds=warm_seconds,
                 GPU_device_job_wall_seconds_conservative=active_seconds,GPU_device_minutes=active_seconds/60,
                 CUDA_sync_before_and_after_every_timed_call=True,independent_process_peak_reset_after_load=True,
                 full_model_memory_included=True,accuracy_parity=accuracy,source_sha256=sha(Path(__file__)),inference_source_sha256=sha(Path(module.__file__)),
                 runtime_inputs_sha256=sha(runtime_path),private_capture=dict(path=capture_path.name,sha256=sha(capture_path)),
                 actual_OpenCV_calls=dict(pose_counts),actual_OpenCV_total=sum(pose_counts.values()),
                 training_or_gradient_updates=0,GT_inputs=False,raw_RGB_published=False)
    write(out,receipt);print(json.dumps(dict(status="COMPLETE",method=args.method,latency_ms=receipt["latency_ms"],peak_memory_MiB=receipt["peak_memory_MiB"],GPU_seconds=active_seconds,parity=accuracy),ensure_ascii=False))


def aggregate(args):
    doc=Path(args.public_dir);old=Path(args.prior_public_dir)
    previous=json.loads((old/'LATENCY.json').read_text());methods={}
    oldmethods=('BASE','SUBPIX','OLD_P','OLD_P_SUBPIX','M1','M1_SUBPIX','M3','M3_SUBPIX','POSEFIX','POSEFIX_SUBPIX')
    reference=json.loads((old/'LATENCY_BASE.json').read_text())
    for method in oldmethods:
        methods[method]={**previous['methods'][method], 'source':'REUSED_PRIOR_18_ROUTE_BENCHMARK', 'original_receipt':'../pallet_nonsevere_precision_20261011/'+previous['methods'][method]['receipt']}
    for method in METHODS:
        path=doc/f'LATENCY_{method}.json';r=json.loads(path.read_text())
        assert r['panel_id_sha256']==reference['panel_id_sha256'] and r['panel_ids']==reference['panel_ids']
        assert r['precision']==reference['precision'] and r['timed_calls']==90 and r['warmups']==20
        methods[method]={key:r[key] for key in ('status','latency_ms','peak_memory_MiB','peak_reserved_MiB','total_params','additional_params','accuracy_parity','missing_pose_timed_calls','GPU_device_minutes','parameter_metadata')}
        methods[method].update(source='NEW_INDEPENDENT_PROCESS',receipt=path.name,receipt_sha256=sha(path))
    write(doc/'LATENCY.json',dict(status='COMPLETE',methods=methods,reused_old_routes=10,new_measured_routes=5,
        source_old_all18_latency_sha256=sha(old/'LATENCY.json'),same_panel_ID_and_precision_exact=True,
        old_PoseFix_SubPix_max_accuracy_parity_px=previous['methods']['POSEFIX_SUBPIX']['accuracy_parity']['max_native_coordinate_delta_px'],
        GPU_device_minutes=sum(methods[m]['GPU_device_minutes'] for m in METHODS),cost_is_auxiliary=True,
        no_equal_accuracy_or_efficiency_claim=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source-root','private-dir','public-dir','prior-public-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--method',choices=METHODS);p.add_argument('--aggregate',action='store_true')
    p.add_argument('--accuracy-predictions',nargs='*',default=[])
    args=p.parse_args();assert bool(args.method)!=args.aggregate
    aggregate(args) if args.aggregate else route(args)
