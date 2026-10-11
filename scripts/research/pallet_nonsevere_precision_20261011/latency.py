"""Independent-process, batch-one full pipeline latency and memory benchmark.

Each invocation measures one route, with 20 warmups then 5 repetitions for
each of the same 18 native images. File decode and model loading are excluded
from latency; native tile copy, detector, corrector, SubPix and PnP are inside
InferenceEngine.run. CUDA synchronization brackets every timed call.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

METHODS=("BASE","SUBPIX","OLD_P","OLD_P_SUBPIX","N3_DIM_SYM","N3_THEN_SUBPIX",
         "POSEFIX","POSEFIX_SUBPIX")+tuple(m+s for m in ("M0","M1","M2","M3","M4") for s in ("","_SUBPIX"))


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def moments(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),SD=float(a.std(ddof=1)),variance=float(a.var(ddof=1)),
                median=float(np.median(a)),P90=float(np.quantile(a,.9)),P95=float(np.quantile(a,.95)),
                minimum=float(a.min()),maximum=float(a.max()))
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()


def install_pose_counter():
    counts=Counter();original={}
    for name in ("solvePnP","solvePnPGeneric","solvePnPRansac","solvePnPRefineLM","solvePnPRefineVVS"):
        if not hasattr(cv2,name):continue
        original[name]=getattr(cv2,name)
        def wrapper(*args,_name=name,_fn=original[name],**kwargs):
            counts[_name]+=1
            return _fn(*args,**kwargs)
        setattr(cv2,name,wrapper)
    return counts,original


def counted_run(engine,image,method,K,xyz,counts):
    before=counts.copy();result=engine.run(image,method,K,xyz)
    result.setdefault("counters",{})["opencv_calls"]={key:counts[key]-before[key] for key in counts if counts[key]!=before[key]}
    return result


def panel(runtime):
    bysession=defaultdict(list)
    for k,(fid,session) in enumerate(zip(runtime["ids"].tolist(),runtime["sessions"].tolist())):bysession[session].append((fid,k))
    selected=[pair for session in sorted(bysession) for pair in sorted(bysession[session])[:2]]
    assert len(bysession)==9 and len(selected)==18
    return selected


def parity(captures,paths,method):
    if not paths:return dict(status="PENDING_SAME_PRECISION_ACCURACY_CAPTURE",accuracy_parity_checked=False)
    lookup={}
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line)
                if r["method"]==method and int(r["seed"])==1:
                    q=r.get("qFinal",r.get("q",r.get("points")))
                    if q is not None:lookup[r["id"]]=np.asarray(q,float)
    errors=[];missing=[]
    for r in captures:
        if r["id"] not in lookup:missing.append(r["id"]);continue
        q=np.asarray(r["points"],float);target=lookup[r["id"]]
        if q.shape!=target.shape:return dict(status="SHAPE_MISMATCH",accuracy_parity_checked=False)
        errors.append(float(np.max(np.abs(q-target))))
    return dict(status="CALCULATED" if not missing else "PARTIAL_CAPTURE_MISSING",accuracy_parity_checked=not missing,
                compared_calls=len(errors),missing_ids=sorted(set(missing)),max_native_coordinate_delta_px=max(errors) if errors else None,
                mean_call_max_delta_px=float(np.mean(errors)) if errors else None,
                interpretation="Measured native coordinate difference; old precision caches require explicit accuracy re-evaluation, never silently equivalent")


def route(args):
    doc=Path(args.public_dir);private=Path(args.private_dir);out=doc/f"LATENCY_{args.method}.json"
    assert not out.exists(),"Existing benchmark receipt preserved"
    runtime_path=private/"BASELINE_INPUTS.npz";runtime=np.load(runtime_path,allow_pickle=False)
    selected=panel(runtime);images={};decode_start=time.perf_counter()
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
    module=importlib.import_module("scripts.research.pallet_nonsevere_precision_20261011.inference")
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
    doc=Path(args.public_dir);output=doc/"LATENCY.json";assert not output.exists()
    methods={};panel_hashes=set();sources=set()
    for method in METHODS:
        path=doc/f"LATENCY_{method}.json"
        if not path.exists():
            methods[method]=dict(status="NOT_MEASURED",reason="No independent-process receipt; may be blocked or budget limited");continue
        r=json.loads(path.read_text());assert r["method"]==method and r["timed_calls"]==90 and r["warmups"]==20
        assert r["precision"]==dict(dtype="FP32",autocast=False,matmul_TF32=False,cudnn_TF32=False,cudnn_benchmark=False)
        panel_hashes.add(r["panel_id_sha256"]);sources.add(r["inference_source_sha256"])
        methods[method]=dict(status=r["status"],latency_ms=r["latency_ms"],peak_memory_MiB=r["peak_memory_MiB"],peak_reserved_MiB=r["peak_reserved_MiB"],
                             total_params=r["total_params"],additional_params=r["additional_params"],accuracy_parity=r["accuracy_parity"],receipt=path.name,
                             receipt_sha256=sha(path),missing_pose_timed_calls=r["missing_pose_timed_calls"],GPU_device_minutes=r["GPU_device_minutes"])
    assert len(panel_hashes)<=1 and len(sources)<=1
    result=dict(status="COMPLETE_WITH_EXPLICIT_MISSING" if any(r["status"]=="NOT_MEASURED" for r in methods.values()) else "COMPLETE",
                methods=methods,all_methods_same_panel_and_precision=True,latency_includes_native_tile_read_and_copy_inside_run=True,
                GPU_device_minutes=sum(r.get("GPU_device_minutes",0) for r in methods.values()),
                no_global_accuracy_speed_optimality_claim=True,source_sha256=sha(Path(__file__)))
    write(output,result);print(json.dumps(dict(status=result["status"],measured=sum(r["status"]!="NOT_MEASURED" for r in methods.values()),GPU_minutes=result["GPU_device_minutes"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--public-dir",required=True);parser.add_argument("--private-dir")
    parser.add_argument("--source-root");parser.add_argument("--method",choices=METHODS);parser.add_argument("--aggregate",action="store_true")
    parser.add_argument("--accuracy-predictions",nargs="*",default=[])
    args=parser.parse_args();assert bool(args.method)!=args.aggregate
    aggregate(args) if args.aggregate else route(args)
