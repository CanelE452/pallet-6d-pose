"""Resumable frame execution with sealed predictor-only artifacts."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np

from .common import canonical,read,rows,save_rows,sha,write,PredictionFirewall
from . import boundary_model as B
from . import solver


def boundary_bundle(frame,config):
    binding=frame['packet'].get('provenance',{}).get('original_mask_binding')
    if binding is None:
        return None,dict(status='UNSUPPORTED_NO_AUTOMATIC_MASK',reference_inputs=False)
    image,mask=B.load_bound_frame(config['source_root'],frame['packet'])
    bundle=B.build_boundary_bundle(image,mask,frame['packet'],config.get('boundary_config'))
    summary=dict(status='SUPPORTED' if bundle['consumed_native_ids'] else 'UNSUPPORTED_FIXED_RGB_PROFILE',
                 consumed_native_ids=bundle['consumed_native_ids'],consumed_group_ids=bundle['consumed_group_ids'],
                 profile_count=sum(g['supported_query_count']for g in bundle['groups']),
                 source=bundle['source'],reference_inputs=False)
    return bundle,summary


def frame_job(frame,config,folder):
    import cv2
    cv2.setNumThreads(1)
    started,cpu=time.monotonic(),time.process_time()
    folder=Path(folder)
    bundle,boundary=boundary_bundle(frame,config)
    result=solver.generate_candidates(frame,config,bundle=bundle)
    models=solver.compile_frame(frame,config,bundle=bundle)
    scale=next(iter(models.values())).ctx.scale
    bank=solver.unique_poses(result['carried']+[c for v in result['banks'].values()for c in v]
                            +[c for c in result['attempts']if c['solve']['feasible']],scale)
    # Reuse exact already evaluated rows wherever possible. A physical pose's
    # score is deterministic in these fixed observations, independent of arm.
    available=[]
    for mode,candidates in result['banks'].items():
        available.extend(zip(candidates,result['bank_scores'][mode]))
    available.extend(zip(result['carried'],result['carried_scores']))
    scores=[]
    for c in bank:
        old=next((s for p,s in available if p['candidate_id']==c['candidate_id']),None)
        scores.append(old if old is not None else solver.candidate_scores(c,models))
    features=solver.features_for_bank(frame,models,bank,scores)
    for arm,pred in result['predictions'].items():
        pred['boundary_applied']=bool(boundary['consumed_native_ids']) if 'consumed_native_ids'in boundary and arm in(
            'B_FIXED_BANK','B_FULL','AB_FULL','CTRL_EDGE_BOUNDARY') else False
    result.update(boundary=boundary,learning_bank=bank,learning_scores=scores,
        features=dict(id=frame['id'],split=frame.get('split'),candidates=features,
                      fallback=frame['fallback'],feature_names=list(solver.FEATURE_NAMES),
                      transformation='signed_log1p31_plus31_explicit_validity_bits'),
        resource=dict(elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
                      new_model_forwards=0,new_optimizer_attempts=len(result['attempts'])))
    # Explicitly bind the candidate/features universe for postseal L diagnosis.
    result['selection_banks']={}
    if bundle is not None:
        save_rows(folder/'BOUNDARY_BUNDLE.jsonl.gz',[bundle])
        result['boundary']['bundle_sha256']=sha(folder/'BOUNDARY_BUNDLE.jsonl.gz')
    save_rows(folder/'RESULT.jsonl.gz',[result])
    write(folder/'FRAME_SEAL.json',dict(id=frame['id'],result_sha256=sha(folder/'RESULT.jsonl.gz'),
        config_sha256=config['config_sha256'],input_row_sha256=__import__('hashlib').sha256(canonical(frame).encode()).hexdigest(),
        before_scoring=True,reference_inputs=False,complete=True,resource=result['resource']))
    return dict(id=frame['id'],path=str(folder/'RESULT.jsonl.gz'),**result['resource'])


def run(inputs,config_path,output,*,workers=2,limit=None):
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=read(config_path);config['config_sha256']=sha(config_path)
    if workers>config['budget']['workers']:
        raise ValueError('CPU worker cap exceeded')
    guard=PredictionFirewall(config.get('forbidden_prediction_paths',[])).install()
    frames=list(rows(inputs))
    if limit is not None:
        frames=frames[:limit]
    input_digest=sha(inputs)
    code_hashes={p.name:sha(p)for p in Path(__file__).parent.glob('*.py')
                 if p.name in ('batch.py','solver.py','owner_model.py','boundary_model.py','common.py')}
    lock=dict(inputs_sha256=input_digest,config_sha256=sha(config_path),source_sha256=code_hashes,
              ordered_ids=[f['id']for f in frames],workers=workers,
              technical_limit=limit,reference_inputs=False,before_new_PRIMARY_scoring=True)
    lockfile=out/'BATCH_LOCK.json'
    if lockfile.exists():
        if read(lockfile)!=lock:
            raise ValueError('Batch lock changed; preserve run and assign a new technical attempt')
    else:
        write(lockfile,lock)
    if (out/'BATCH_SEAL.json').exists():
        sealed=read(out/'BATCH_SEAL.json')
        if (sealed['input_sha256']!=input_digest or sealed['config_sha256']!=sha(config_path)
            or sealed['source_sha256']!=code_hashes
            or sealed['predictions_sha256']!=sha(out/'PREDICTIONS.jsonl.gz')
            or sealed['head_features_sha256']!=sha(out/'HEAD_FEATURES.jsonl.gz')):
            raise ValueError('Completed batch seal or output changed')
        return sealed
    completed={};pending=[]
    for i,frame in enumerate(frames):
        folder=out/'frames'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True)
        sealfile=folder/'FRAME_SEAL.json'
        if sealfile.exists():
            seal=read(sealfile)
            if (seal['id']!=frame['id']or seal['config_sha256']!=sha(config_path)
                or sha(folder/'RESULT.jsonl.gz')!=seal['result_sha256']):
                raise ValueError('Completed frame binding changed')
            completed[frame['id']]=dict(id=frame['id'],reused=True,path=str(folder/'RESULT.jsonl.gz'),**seal['resource'])
        else:
            pending.append((frame,folder))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(frame_job,frame,config,folder):(frame,folder)for frame,folder in pending}
        failures=[]
        for future in as_completed(futures):
            frame,folder=futures[future]
            try:
                completed[frame['id']]=future.result()
            except BaseException as exc:
                failure=dict(id=frame['id'],error=repr(exc),status='FAILED_TECHNICAL',epoch=time.time())
                failures.append(failure)
                write(folder/'FAILURE.json',failure)
            n=len(completed)+len(failures)
            if n%8==0 or n==len(frames):
                print(canonical(dict(frames_processed=n,total=len(frames),failed=len(failures),
                      elapsed_seconds=time.monotonic()-started)),flush=True)
            write(out/'BATCH_PROGRESS.json',dict(complete_ids=list(completed),failures=failures,
                total=len(frames),elapsed_seconds=time.monotonic()-started),replace=True)
    if failures:
        # Individual optimization failures are already per-arm fallbacks. A
        # frame-level contract/IO failure is never disguised as a new baseline.
        raise RuntimeError(f'{len(failures)} frame contract failures; preserved all attempts; repair only their schemas/IO')
    values=[next(rows(completed[f['id']]['path']))for f in frames]
    save_rows(out/'PREDICTIONS.jsonl.gz',values)
    save_rows(out/'HEAD_FEATURES.jsonl.gz',[v['features']for v in values])
    receipt=dict(schema='COMPLETE_AUTOMATIC_BATCH_V1',complete=True,rows=len(values),
        ordered_ids=[v['id']for v in values],predictions_sha256=sha(out/'PREDICTIONS.jsonl.gz'),
        head_features_sha256=sha(out/'HEAD_FEATURES.jsonl.gz'),input_sha256=input_digest,
        config_sha256=sha(config_path),source_sha256=code_hashes,before_scoring=True,reference_inputs=False,
        new_optimizer_attempts=sum(v['actual_new_solve_attempts']for v in values),
        new_neural_forwards=0,workers=workers,elapsed_seconds=time.monotonic()-started,
        sum_frame_cpu_seconds=sum(v['resource']['process_cpu_seconds']for v in values),
        firewall=guard.receipt(),failures=[],all_frames_kept=True)
    write(out/'BATCH_SEAL.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int)
    a=p.parse_args();print(canonical(run(a.inputs,a.config,a.output,workers=a.workers,limit=a.limit)))
