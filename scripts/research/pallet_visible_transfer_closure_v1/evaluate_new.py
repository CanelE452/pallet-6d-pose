"""Freeze both new students and unchanged D9 outputs BEFORE opening evaluation references."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import common as C
from .infer_train import predict
from .diagnose import stats,paired,aggregate_train
from scripts.research.pallet_selftraining_paper_closure_v1.freeze_predictions import pose_job
from scripts.research.pallet_selftraining_paper_closure_v1.evaluate import posejob,paired as fullpaired
from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
from scripts.research.pallet_verified_anchor_v1.evaluate import point

NEW=('RAW_NEW','REF_NEW')

def freeze():
    if (C.DOC/'PREDICTIONS_LOCK.json').exists():
        for b in C.read(C.DOC/'PREDICTIONS_LOCK.json')['files']:C.verify(b)
        print('PREDICTIONS_ALREADY_LOCKED');return
    assert not (C.DOC/'EXPERIMENT_BLOCKED.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    rr=C.P.records();trainrows=C.read(C.RAW/'TRAIN_TARGETS_PRIVATE.json');preds={};poses={};checks=[]
    meta={r['id']:r for r in C.read(C.P.META)};old=C.read(C.P.RAW/'PREDICTIONS.json')['R0']
    for a in NEW:
        fit=C.read(C.DOC/f'FIT_{a}.json');C.verify(fit['checkpoint']);assert fit['optimizer_steps']==640
        p=C.RAW/f'EVAL_{a}.json';tp=C.RAW/f'TRAIN_{a}.json'
        if not p.exists() or not tp.exists():
            model=YOLO(str(C.ROOT/fit['checkpoint']['path']),task='pose')
            for dst,records in [(p,rr),(tp,trainrows)]:
                if dst.exists():continue
                out={}
                for r in records:
                    C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
                    out[r['id']]=predict(model,im)
                C.save(dst,out,True)
            del model;torch.cuda.empty_cache()
        preds[a]=C.read(p)
        for fid,pred in preds[a].items():
            base=old[fid];assert pred['selected_index']==base['selected_index'] and len(pred['candidates'])==len(base['candidates'])
            for x,y in zip(pred['candidates'],base['candidates']):assert x['box_xyxy']==y['box_xyxy'] and x['score']==y['score']
        pp=C.RAW/f'POSE_{a}.json'
        if not pp.exists():
            with ProcessPoolExecutor(max_workers=4) as pool:
                out=dict(pool.map(pose_job,[(i,p,meta[i]) for i,p in preds[a].items()],chunksize=8))
            C.save(pp,out,True)
        poses[a]=C.read(pp);checks.extend([p,tp,pp,C.DOC/f'FIT_{a}.json'])
        print('PREDICTIONS_FROZEN',a,flush=True)
    C.save(C.DOC/'PREDICTIONS_LOCK.json',dict(utc=C.now(),files=[C.bind(p) for p in checks],reference_coordinates_opened=False,
        same_D9=True,detector_outputs_exact=True,selection='original highest-score candidate; original D9 geometry only',
        independent_test=False,known_DEV_does_not_become_TEST=True),True)

def score():
    if (C.DOC/'RESULTS.json').exists():print('RESULTS_ALREADY_COMPLETE');return
    for b in C.read(C.DOC/'PREDICTIONS_LOCK.json')['files']:C.verify(b)
    C.save(C.DOC/'SCORING_START.json',dict(utc=C.now(),lock=C.bind(C.DOC/'PREDICTIONS_LOCK.json')),True)
    records=C.P.records();ids=[r['id'] for r in records];groups=V.groups(records);truth=C.read(C.P.TRUTH);_,gt=D.Pose.metadata('REAL_DEV')
    pred={a:C.read(C.RAW/f'EVAL_{a}.json') for a in NEW};poses={a:C.read(C.RAW/f'POSE_{a}.json') for a in NEW}
    frames=C.read(C.P.RAW/'FRAME_METRICS.json');pm=C.read(C.P.RAW/'POSE_METRICS.json');fixed={}
    for a in NEW:
        frames[a]={};fixed[a]={}
        for fid in ids:
            p=C.P.selected(pred[a][fid]);g=truth[fid];matched=p is not None and V.E.C.H.E.O.iou(p['box_xyxy'],g['box'])>=.5
            q=np.full((9,2),np.nan) if p is None else p['keypoints_xy']
            frames[a][fid]=dict(id=fid,**V.E.P.M.measure(q,g['gt'],g['valid'],g['permutations'],g['hw'],matched,p is not None))
            fixed[a][fid]=dict(id=fid,**V.E.P.M.measure(q,g['gt'],g['valid'],[list(range(9))],g['hw'],matched,p is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:pm[a]=dict(pool.map(posejob,[(i,poses[a][i],gt[i]) for i in ids],chunksize=8))
    arms=(*C.ARMS,*NEW);out={g:{a:dict(twoD=summary([frames[a][i] for i in ii]),sixD=D.aggregate([pm[a][i] for i in ii])) for a in arms} for g,ii in groups.items()}
    ar=C.read(C.P.RAW/'ANCHOR_POINTS.json')
    for r in ar:
        for a in NEW:
            q=point(pred[a][r['frame_id']],r['corner_id']);r['missing'][a]=q is None;r['model_xy'][a]=None if q is None else q.tolist()
            r['errors'][a]=float(np.linalg.norm(q-np.array(r['verified_xy']))) if q is not None else float(np.hypot(*truth[r['frame_id']]['hw']))
    ag={'ALL':ar}
    for key in ('severity','recording','corner_id'):
        ag.update({f'{key}:{g}':[r for r in ar if r[key]==g] for g in sorted({r[key] for r in ar})})
    contrasts=(('RAW_LR5','REF_LR5'),('RAW_NEW','REF_NEW'),('REF_LR5','REF_NEW'),('RAW_LR5','RAW_NEW'))
    visible={g:{a:stats([r['errors'][a] for r in rr]) for a in (*arms,'TEACHER')} for g,rr in ag.items()}
    pairs={g:{f'{b}-minus-{a}':paired(rr,a,b) for a,b in contrasts} for g,rr in ag.items()}
    trainrows=C.read(C.RAW/'TRAIN_TARGETS_PRIVATE.json');train={}
    for a in NEW:
        p=C.read(C.RAW/f'TRAIN_{a}.json');train[a]={}
        for key in ('raw_target','ref_target'):
            errors=[];weights=[];perimage=[];occ=[]
            for r in trainrows:
                q=C.P.selected(p[r['id']]);assert q is not None
                mask=np.array(r['common_support'][:8]);d=np.linalg.norm(np.array(q['keypoints_xy'][:8])-np.array(r[key][:8]),axis=1)[mask]
                errors.extend(d);perimage.append(d.mean());occ.append(r['occurrences_per_epoch'])
            train[a][key]=dict(stats(errors),unique_image_mean=float(np.mean(perimage)),occurrence_mean=float(np.average(perimage,weights=occ)))
    result=dict(full128=out,visible66=visible,visible_pairs=pairs,full128_pairs={f'{b}-minus-{a}':fullpaired(frames[a],frames[b],pm[a],pm[b],records) for a,b in contrasts},
        visible_LORO={rec:{f'{b}-minus-{a}':paired([r for r in ar if r['recording']!=rec],a,b) for a,b in contrasts} for rec in sorted({r['recording'] for r in ar})},
        train_native_following=train,updates_each=640,independent_confirmation=False)
    C.save(C.RAW/'ANCHOR_POINTS_NEW_PRIVATE.json',ar);C.save(C.RAW/'FRAME_METRICS.json',{a:frames[a] for a in arms});C.save(C.RAW/'POSE_METRICS.json',{a:pm[a] for a in arms});C.save(C.RAW/'FIXED_ID_METRICS_NEW.json',fixed)
    C.save(C.DOC/'RESULTS.json',result,True)
    for a in arms:print(a,'visible',visible['ALL'][a]['PCK']['10']['correct'],'full',out['ALL'][a]['twoD']['PCK']['10'],'AUC',out['ALL'][a]['sixD']['ADDsym_AUC'],flush=True)

def source():
    if (C.DOC/'SOURCE_HOLDOUT.json').exists():return
    # Retain the original32-image synthetic framework validation role; not reliability calibration or real evaluation.
    proto=C.read(C.P.REC/'pose_only/PROTOCOL.json');data=str(C.ROOT/proto['datasets']['RAW']['data']['path']);out={}
    torch.set_num_threads(4);cv2.setNumThreads(1)
    for a in (*C.ARMS,*NEW):
        ck=C.checkpoint(a) if a in C.ARMS else C.read(C.DOC/f'FIT_{a}.json')['checkpoint']
        m=YOLO(str(C.ROOT/ck['path']),task='pose')
        r=m.val(data=data,split='val',imgsz=640,batch=16,device='0',workers=2,plots=False,save_json=False,project=str(C.RAW/'source_val'),name=a,verbose=False)
        out[a]=dict(r.results_dict);del m;torch.cuda.empty_cache()
    C.save(C.DOC/'SOURCE_HOLDOUT.json',dict(results=out,n=32,role='Original framework synthetic validation, not independent confirmation/calibration. Same data+settings for five models.',data=C.bind(Path(data))),True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','score','source']);a=p.parse_args();globals()[a.action]()
