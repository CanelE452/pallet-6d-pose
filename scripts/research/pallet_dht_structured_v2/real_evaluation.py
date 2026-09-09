"""Unchanged-ID real DEV evaluation downstream of synthetic-only selection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .cache import CachedData,read,write,sha
from .data_ops import fixed_states,prepare_batch
from .evaluation import select_indices
from .train import load_trained
from scripts.research.pallet_dht_decoder_probe_v1 import evaluate as METRIC
from scripts.research.pallet_dht_global_layout_v1.evaluation import iou

ROOT=Path(__file__).resolve().parents[3]


def verify_hashes(mapping):
    for path,expected in mapping.items():
        if sha(path)!=expected:raise ValueError(f'Frozen real evaluation binding changed: {path}')


def predict_real(run,seed=1,device='cuda:0',batch_size=8):
    run=Path(run).resolve();protocol=read(run/'PROTOCOL.json')
    gatepath=run/'evaluation'/f'point_segment_hough_seed{seed}'/'SYNTHETIC_RESULTS.json';gate=read(gatepath)
    if not (gate['complete'] and gate['PASS'] and gate['arm']=='point_segment_hough' and gate['seed']==seed and gate['advancement']['advance']):
        raise ValueError('Full verifier has not passed the registered synthetic advancement gate')
    verify_hashes(gate['input_sha256']);verify_hashes(gate['source_sha256'])
    cache=CachedData(run);indices=cache.populations['real_dev']
    if len(indices)!=319:raise ValueError('Canonical319 DEV frames required')
    records=[dict(index=i,id=cache.records[i]['id'],image=cache.records[i]['image'],
        image_sha256=cache.records[i]['image_sha256'],population='real_dev',session_id=cache.records[i]['session_id'],
        baseline=cache.records[i]['baseline'],arms={}) for i in indices]
    bindings={str(run/'PROTOCOL.json'):sha(run/'PROTOCOL.json'),str(run/'CACHE_COMPLETION.json'):sha(run/'CACHE_COMPLETION.json'),str(gatepath):sha(gatepath),**gate['input_sha256']}
    for arm in protocol['architecture']['arms']:
        path=run/f'SELECTION_{arm}_seed{seed}.json';selection=read(path)
        if not (selection['complete'] and selection['PASS'] and selection['arm']==arm and selection['seed']==seed and selection['no_real_selection'] and not selection['synthetic_validation_used_for_selection']):
            raise ValueError('Synthetic-calibration-only selection required')
        verify_hashes(selection['input_sha256']);verify_hashes(selection['source_sha256'])
        bindings[str(path)]=sha(path)
        checkpoint=run/'runs'/f'{arm}_seed{seed}'/'checkpoint_final.pth';bindings[str(checkpoint)]=sha(checkpoint)
        if bindings[str(checkpoint)]!=selection['checkpoint_sha256']:raise ValueError('Selection belongs to different checkpoint')
        model=load_trained(run,arm,seed,device)
        with torch.inference_mode():
            for start in range(0,len(indices),batch_size):
                ix=indices[start:start+batch_size];inputs,targets=cache.batch(ix,'cpu')
                if any(bool(x.any()) for x in targets.values()):raise ValueError('Real cache labels must be zero')
                states,variants=fixed_states(ix,'clean');batch,valid,_=prepare_batch(inputs,targets,states,variants)
                batch={k:v.to(device) for k,v in batch.items()};cost=model(batch).cpu().numpy();mask=valid.numpy()
                chosen=select_indices(cost,mask,selection['margin']);q=batch['layouts'].cpu().numpy().astype(float)
                for j,index in enumerate(chosen):
                    row=records[start+j];base=np.asarray(row['baseline']['points'],float)
                    output=base.copy() if index==0 else q[j,index].copy();output[8]=base[8]
                    row['arms'][arm]=dict(points=output.tolist(),point_valid=row['baseline']['point_valid'],
                        selected_index=int(index),margin=selection['margin'],
                        identity_cost=float(cost[j,0]),selected_cost=float(cost[j,index]),
                        costs=[float(v) if ok else None for v,ok in zip(cost[j],mask[j])],candidate_valid=mask[j].tolist())
        del model
    for path,expected in bindings.items():
        if sha(path)!=expected:raise ValueError('Prediction binding changed')
    result=dict(complete=True,PASS=True,seed=seed,selection_uses_GT=False,real_annotation_files_opened=0,
        input_sha256=bindings,source_sha256={str(Path(__file__).resolve()):sha(__file__)},records=records)
    path=run/f'REAL_PREDICTIONS_seed{seed}.json'
    if path.exists() and read(path)!=result:raise ValueError('Different existing frozen real predictions')
    write(path,result)
    return path


def evaluate_saved(run,seed=1):
    run=Path(run).resolve();protocol=read(run/'PROTOCOL.json');predpath=run/f'REAL_PREDICTIONS_seed{seed}.json'
    predictions=read(predpath)
    if not (predictions['complete'] and predictions['PASS'] and predictions['seed']==seed) or predictions['selection_uses_GT']:raise ValueError('GT-free saved predictions required')
    verify_hashes(predictions['input_sha256']);verify_hashes(predictions['source_sha256'])
    snapshot_path=protocol['input_snapshot']['path']
    if sha(snapshot_path)!=protocol['input_snapshot']['sha256']:raise ValueError('Original source snapshot changed')
    snapshot=read(snapshot_path)['sha256'];inputs={str(predpath):sha(predpath),str(snapshot_path):sha(snapshot_path)}
    def frozen_read(path):
        path=Path(path).resolve();expected=snapshot[str(path)]
        if sha(path)!=expected:raise ValueError(f'Original evaluation source changed: {path}')
        inputs[str(path)]=expected;return read(path)
    previous=frozen_read(Path(protocol['input_run'])/'FRAME_METRICS.json')
    old={r['id']:r for r in previous['records'] if r['population']=='real_dev'}
    manifest=frozen_read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    items={r['frame_id']:r for r in manifest['items']}
    if manifest['role']!='DEV' or len(items)!=319:raise ValueError('Reused DEV319 manifest required')
    if METRIC.BOOTSTRAP_SEED!=protocol['real_evaluation']['bootstrap_seed']:raise ValueError('Bootstrap seed changed')
    arms=('baseline',*protocol['architecture']['arms']);rows=[];annotations={}
    for record in predictions['records']:
        fid=record['id'];item=items[fid];prior=old[fid]
        if Path(record['image']).resolve()!=(ROOT/item['image_path']).resolve() or record['session_id']!=prior['session_id']:
            raise ValueError('Canonical image/session identity changed')
        obj=frozen_read(ROOT/item['gt_v2_path'])['objects'][0];aa=obj['keypoint_annotations'];annotations[fid]=aa
        xy=np.asarray([p['xy'] for p in aa],float);gtvalid=np.array([p['visibility']>0 for p in aa])
        base,bvalid=METRIC.finite_points(record['baseline']['points'],record['baseline']['point_valid'])
        targetbox=np.r_[xy[:8].min(0),xy[:8].max(0)];box=record['baseline']['box_xyxy']
        matched=box is not None and iou(np.asarray(box),targetbox)>=.5
        if not (np.array_equal(xy,prior['gt_points']) and np.array_equal(gtvalid,prior['gt_supervised']) and matched==prior['baseline_match_iou50']):
            raise ValueError('GT or original match changed')
        if not np.array_equal(base,prior['arms']['baseline']['points']):raise ValueError('Original baseline changed')
        row=dict(id=fid,population='real_dev',session_id=record['session_id'],gt_points=xy.tolist(),
            gt_supervised=gtvalid.tolist(),baseline_match_iou50=matched,arms={})
        for arm in arms:
            payload=record['baseline'] if arm=='baseline' else record['arms'][arm]
            p,valid=METRIC.finite_points(payload['points'],payload['point_valid'])
            if not np.array_equal(valid,bvalid) or not np.array_equal(p[8],base[8]):raise ValueError('Centroid/coverage changed')
            observed=gtvalid&valid&matched;distance=np.linalg.norm(p-xy,axis=-1);movement=np.linalg.norm(p-base,axis=-1)
            errors=[float(e) if ok else None for e,ok in zip(distance,observed)]
            if arm=='baseline' and errors!=prior['arms']['baseline']['errors_px']:raise ValueError('Baseline numerical replay differs')
            row['arms'][arm]=dict(points=payload['points'],errors_px=errors,move_px=[float(e) if ok else None for e,ok in zip(movement,valid)])
        rows.append(row)
    if len(rows)!=319 or len({r['id'] for r in rows})!=319 or len({r['session_id'] for r in rows})!=13:
        raise ValueError('Frame/session denominator changed')
    summaries=[dict(arm=arm,**METRIC.summarize(rows,arm)) for arm in arms]
    if summaries[0]['n_observed_points']!=2738 or abs(summaries[0]['p90_px']-41.48732863482036)>1e-12:
        raise ValueError('Canonical2738point41.4873px baseline failed')
    corners=[dict(arm=arm,**METRIC.summarize(rows,arm,tuple(range(8)))) for arm in arms]
    difficulty=[row for arm in arms for row in METRIC.point_difficulty(rows,arm)]
    pairs=[METRIC.paired(rows,'point_segment_hough',right,resamples=protocol['real_evaluation']['bootstrap_resamples']) for right in arms[:-1]]
    result=dict(complete=True,PASS=True,seed=seed,summaries=summaries,corner_only_summaries=corners,difficulty=difficulty,paired=pairs,
        input_sha256=inputs,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(METRIC.__file__).resolve()):sha(METRIC.__file__)},
        GT_unchanged=True,same_ID=True,real_training_or_margin_tuning=False,independent_final=False,
        metric_scope='Reused DEV319/13 sessions; original nine-point semantic IDs and baseline IoU matches, centroid copied, missing points retained in coverage denominator',
        stable_gain_claim=False,active_goal_complete=False)
    write(run/f'REAL_FRAME_METRICS_seed{seed}.json',dict(complete=True,records=rows))
    write(run/f'REAL_RESULTS_seed{seed}.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    predict_real(a.run_dir,a.seed,a.device);result=evaluate_saved(a.run_dir,a.seed)
    print([{k:r[k] for k in ('arm','median_px','p90_px','mean_px')} for r in result['summaries']])
