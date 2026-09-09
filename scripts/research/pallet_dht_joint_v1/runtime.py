"""Paired whole-predictor timing on fixed26 DEV images, after accuracy is frozen."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
import cv2
from scripts.research.pallet_dht_joint_v1.evaluate import CanonicalPredictor, population, E, ROOT
from scripts.research.pallet_dht_joint_v1.driver import read, write, sha


def compare(actual, reference):
    if len(actual)!=len(reference):raise ValueError('Runtime prediction candidate count changed')
    worst=0.
    for a,b in zip(actual,reference):
        for k in ['score','box_xyxy','keypoints_xy','keypoints_conf']:
            if a[k] is None or b[k] is None:
                if a[k] is not b[k]:raise ValueError('Runtime prediction availability changed')
                continue
            x,y=np.asarray(a[k]),np.asarray(b[k])
            if x.shape!=y.shape or not np.allclose(x,y,atol=1e-4,rtol=0):
                raise ValueError(f'Runtime changed prediction {k}')
            worst=max(worst,float(np.max(np.abs(x-y))))
    return worst


def comparison_details(actual, reference):
    """Keep structural failures fatal; record the unchanged numeric comparison."""
    if len(actual)!=len(reference):raise ValueError('Runtime prediction candidate count changed')
    field_max={name:0. for name in ['score','box_xyxy','keypoints_xy','keypoints_conf']}
    for a,b in zip(actual,reference):
        for name in field_max:
            if a[name] is None or b[name] is None:
                if a[name] is not b[name]:raise ValueError('Runtime prediction availability changed')
                continue
            x,y=np.asarray(a[name]),np.asarray(b[name])
            if x.shape!=y.shape:raise ValueError(f'Runtime prediction shape changed: {name}')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Runtime prediction is nonfinite: {name}')
            field_max[name]=max(field_max[name],float(np.max(np.abs(x-y))))
    error=None
    try:
        compare(actual,reference)
    except ValueError as exc:
        # Structural/availability/finite checks already ran over every field.
        # Only the original numeric mismatch is recoverable for collection.
        if str(exc) not in {f'Runtime changed prediction {name}' for name in field_max}:
            raise
        error=dict(type=type(exc).__name__,message=str(exc))
    return dict(strict_parity_pass=error is None,error=error,
                max_abs_delta_by_field=field_max,max_abs_delta=max(field_max.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    root=args.run_dir.resolve();protocol=read(root/'TRAIN_PROTOCOL.json')
    if (root/'RUNTIME.json').exists():
        existing=read(root/'RUNTIME.json')
        strict_ok=not existing.get('strict_failures',[])
        status='COMPLETE' if strict_ok else 'COMPLETE_WITH_STRICT_PARITY_FAILURE'
        if (existing.get('complete') is True and existing.get('timing_collection_complete') is True
                and existing.get('status')==status
                and existing.get('PASS') is strict_ok and existing.get('parity_PASS') is strict_ok
                and existing.get('parity_policy')==dict(atol=1e-4,rtol=0,criterion_changed=False)
                and existing['protocol_sha256']==sha(root/'TRAIN_PROTOCOL.json')):
            for path,digest in existing['source_sha256'].items():
                if sha(path)!=digest:raise ValueError('Runtime source changed')
            return
        raise ValueError('Incomplete or incompatible existing runtime')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    pair=population();sessions={}
    for item in pair.positive.items:
        session=item.session_id
        sessions.setdefault(session,[]).append(item)
    items=[x for session in sorted(sessions) for x in sorted(sessions[session],key=lambda y:y.frame_id)[:2]]
    if len(items)!=26:raise ValueError('Expected2 images from each of13 sessions')
    paths=[(ROOT/item.image).resolve() for item in items]
    image_sha={str(path):sha(path) for path in paths}
    images=[cv2.imread(str(path)) for path in paths]
    if any(image is None for image in images):raise ValueError('Runtime image decode failed')
    if any(sha(path)!=image_sha[str(path)] for path in paths):raise ValueError('Runtime image changed during decode')
    keys=[E._display_path((ROOT/item.image).resolve()) for item in items]
    rows=[];strict_failures=[]
    sources={str(Path(__file__).resolve()):sha(__file__),str(root/'TRAIN_PROTOCOL.json'):sha(root/'TRAIN_PROTOCOL.json')}
    sources.update(image_sha)
    for name in ['evaluate.py','hough_block.py','integration.py']:
        path=Path(__file__).with_name(name);sources[str(path)]=sha(path)
    for name in ['FINDINGS.json','OPERATOR_ISOLATION.json','SUMMARY.json']:
        path=root/'provenance/runtime_parity_diagnostic'/name;sources[str(path)]=sha(path)
    for seed in protocol['training']['seeds']:
        predictors={};references={};values={}
        for arm in protocol['arms']:
            folder=root/'evaluation'/f'{arm}_seed{seed}'
            c=read(folder/'COMPLETION.json')
            if not(c['complete'] and c['PASS']):raise ValueError('Accuracy evaluation unfinished')
            cp=root/'runs'/f'{arm}_seed{seed}'/'weights/final.pt'
            if sha(cp)!=c['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            sources[str(cp)]=sha(cp);sources[str(folder/'PREDICTIONS.json')]=sha(folder/'PREDICTIONS.json')
            payload=read(folder/'PREDICTIONS.json')
            for key,path in zip(keys,paths):
                if payload['frame_metadata'][key]['image_sha256']!=image_sha[str(path)]:
                    raise ValueError('Runtime image differs from accuracy evaluation')
            predictors[arm]=CanonicalPredictor(cp);references[arm]=payload['frames'];values[arm]=[]
            # Every rectangular shape and image is warmed before measurement.
            for image in images:predictors[arm].predict(image)
        worst={arm:0. for arm in predictors}
        for repeat in range(3):
            for i,(image,key) in enumerate(zip(images,keys)):
                order=protocol['arms'] if (i+repeat)%2==0 else list(reversed(protocol['arms']))
                for arm in order:
                    candidates,duration=predictors[arm].predict(image)
                    detail=comparison_details(candidates,references[arm][key])
                    worst[arm]=max(worst[arm],detail['max_abs_delta'])
                    values[arm].append(dict(frame_id=items[i].frame_id,repeat=repeat,milliseconds=duration,
                        strict_parity_pass=detail['strict_parity_pass'],
                        max_abs_delta_by_field=detail['max_abs_delta_by_field']))
                    if not detail['strict_parity_pass']:
                        strict_failures.append(dict(arm=arm,seed=seed,frame_id=items[i].frame_id,
                            key=key,repeat=repeat,error=detail['error'],
                            actual_candidates=candidates,reference_candidates=references[arm][key],
                            max_abs_delta_by_field=detail['max_abs_delta_by_field'],
                            max_abs_delta=detail['max_abs_delta']))
        for arm,v in values.items():
            times=[x['milliseconds'] for x in v]
            rows.append(dict(arm=arm,seed=seed,n=len(v),median_ms=float(np.median(times)),
                             p90_ms=float(np.percentile(times,90)),mean_ms=float(np.mean(times)),
                             max_coordinate_or_score_delta=worst[arm],observations=v))
        del predictors,references;gc.collect();torch.cuda.empty_cache()
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('Runtime source/input changed during collection')
    strict_ok=not strict_failures
    result=dict(complete=True,timing_collection_complete=True,PASS=strict_ok,parity_PASS=strict_ok,
                status='COMPLETE' if strict_ok else 'COMPLETE_WITH_STRICT_PARITY_FAILURE',
                parity_policy=dict(atol=1e-4,rtol=0,criterion_changed=False),strict_failures=strict_failures,
                protocol_sha256=sha(root/'TRAIN_PROTOCOL.json'),
                source_sha256=sources,runs=rows,frames=[x.frame_id for x in items],repeats=3,
                includes='BGR image padding, letterbox, FP32 full YOLO+DHT+IHT+head, decode, CPU points/boxes/confidence extraction and GPU synchronization',
                excludes='File decoding and PnP',comparison='Same images and matched seeds, interleaved arms; warmed shapes; single RTX3080')
    write(root/'RUNTIME.json',result)
    print({f"{r['arm']}/{r['seed']}":r['median_ms'] for r in rows})


if __name__=='__main__':main()
