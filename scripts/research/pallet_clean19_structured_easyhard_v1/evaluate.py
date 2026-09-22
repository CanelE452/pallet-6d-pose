import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from . import common as C
from . import diagnostics as D


def infer(material,arm):
    C.setup();C.guard();C.immutable()
    for mat in C.MATERIALS:
        for a in C.ARMS:assert C.read(C.DOC/f'FIT_{mat}_{a}.json')['complete']
    fit=C.read(C.DOC/f'FIT_{material}_{arm}.json');C.verify(fit['checkpoint'])
    model=YOLO(str(C.ROOT/fit['checkpoint']['path']),task='pose')
    records=C.read(C.H.P.DOC/'SPLIT.json')['evaluation'];predictions={}
    for r in records:
        if r['object_type'].upper()!=material:continue
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']))
        predictions[r['id']]=C.predict(model,im,100)
    C.save(C.RAW/f'EVAL_{material}_{arm}.json',dict(predictions=predictions,checkpoint=fit['checkpoint'],GT_input=False))
    # These diagnostics use frozen plans, never influence training or checkpoints.
    train=D.probe_model(model,material);source=D.source_model(model)
    C.save(C.RAW/f'DIAGNOSTICS_{material}_{arm}.json',dict(train=train,source=source,checkpoint=fit['checkpoint']))
    print('INFER_COMPLETE',material,arm,flush=True)


def transitions(left,right):
    loss=gain=0
    for b,n in zip(left,right):
        assert b['id']==n['id'] and b['canonical_valid']==n['canonical_valid']
        for i,v in enumerate(b['canonical_valid']):
            if v:
                a=b['canonical_errors'][i];z=n['canonical_errors'][i]
                loss+=a<=10 and z>10;gain+=a>10 and z<=10
    return dict(lost_correct10=int(loss),gained_correct10=int(gain),net_correct10=int(gain-loss),
        new_match_failures=sum(b['matched'] and not n['matched'] for b,n in zip(left,right)),match_recoveries=sum(not b['matched'] and n['matched'] for b,n in zip(left,right)),
        **C.H.P.M.damage(left,right))


def score():
    C.setup();C.immutable()
    records=C.read(C.H.P.DOC/'SPLIT.json')['evaluation'];ids=[r['id'] for r in records]
    preds={arm:{} for arm in C.ARMS};files=[]
    for mat in C.MATERIALS:
        for arm in C.ARMS:
            path=C.RAW/f'EVAL_{mat}_{arm}.json';files.append(C.bind(path));preds[arm].update(C.read(path)['predictions'])
    assert all(set(p)==set(ids) for p in preds.values())
    C.save(C.DOC/'EVALUATION_PREDICTIONS_LOCK.json',dict(files=files,all_six_fits_complete=True,locked_before_main_scoring=True))
    truth=C.read(C.H.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    metrics={'R0':C.read(C.H.P.RAW/'FRAME_METRICS.json')['R0'],
        'TEACHER':C.read(C.H.V.RAW/'FRAME_METRICS.json')['TYPE_REPLAY_PIPELINE'],
        'OLD_STUDENT':C.read(C.H.RAW/'FRAME_METRICS.json')}
    for arm,pp in preds.items():
        metrics[arm]={}
        for fid,p in pp.items():
            t=truth[fid];c=C.H.P.C.selected(p);matched=c is not None and C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
            q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
            metrics[arm][fid]=dict(id=fid,**C.H.P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
    # Pose predictions fixed before loading any6D reference.
    meta={r['id']:r for r in C.read(C.H.V.RAW/'INFERENCE_METADATA.json')};poses={}
    for arm,pp in preds.items():
        poses[arm]={}
        for fid,p in pp.items():
            m=meta[fid];c=C.H.P.C.selected(p);q=None if c is None else np.array(c['keypoints_xy'],float)
            if q is not None:q[(q==-1).all(1)]=np.nan
            poses[arm][fid]=C.H.V.Pose.infer(q,np.array(m['K']),np.array(m['xyz']),False)
    C.save(C.RAW/'POSE_PREDICTIONS.json',poses);C.save(C.DOC/'POSE_PREDICTIONS_LOCK.json',C.bind(C.RAW/'POSE_PREDICTIONS.json'))
    _,gt=C.H.V.Pose.metadata('REAL_DEV')
    historic=C.read(C.H.RAW/'pose/FRAME_METRICS.json')
    pm={'R0':historic['R0'],'TEACHER':historic['CORRECTED_TEACHER'],'OLD_STUDENT':historic['STUDENT']}
    for arm,pp in poses.items():
        with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(C.H.V.Pose.metric,[(fid,p,gt[fid]) for fid,p in pp.items()],chunksize=16))
        for r in rows:
            if r['available']:r['axis_correct']=bool(abs(pp[r['id']]['cf_extents'][0]-gt[r['id']]['body_xyz'][0])<1e-6)
        pm[arm]={r['id']:r for r in rows}
    groups={'ALL300':ids}
    groups.update({s:[r['id'] for r in records if r['severity']==s] for s in C.H.P.SEVERITIES})
    for mat in C.MATERIALS:
        groups[mat]=[r['id'] for r in records if r['object_type'].upper()==mat]
        for s in C.H.P.SEVERITIES:groups[mat+'_'+s]=[r['id'] for r in records if r['object_type'].upper()==mat and r['severity']==s]
    for session in sorted({r['session'] for r in records}):groups['SESSION_'+session]=[r['id'] for r in records if r['session']==session]
    results={};contrasts={};common={};axis={}
    for group,ii in groups.items():
        if not ii:continue
        results[group]={};contrasts[group]={};common[group]={};axis[group]={}
        matched=[fid for fid in ii if all(m[fid]['matched'] for m in metrics.values())]
        for arm,mm in metrics.items():
            rr=[mm[i] for i in ii];s=C.H.P.M.summary(rr)
            s['correct']={str(t):sum(e<=t for r in rr for e in r['errors']) for t in (5,10,20)}
            pose=[pm[arm][i] for i in ii];ok=[r for r in pose if r['available']]
            six=dict(frames=len(ii),available=len(ok),coverage=len(ok)/len(ii),axis_accuracy=float(np.mean([r['axis_correct'] for r in ok])) if ok else None,
                ADDsym_AUC=C.H.V.Pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in pose],1.))
            for k in ('translation_cm','rotation_deg','yaw_deg','IoU3D'):six[k]=dict(median=float(np.median([r[k] for r in ok])) if ok else None,P90=float(np.quantile([r[k] for r in ok],.9)) if ok else None)
            results[group][arm]=dict(twoD=s,sixD=six,vs_R0=transitions([metrics['R0'][i] for i in ii],rr))
            common[group][arm]=C.H.P.M.summary([mm[i] for i in matched])
            axis[group][arm]={str(v):dict(n=sum(r['axis_correct']==v for r in ok),rotation_median=float(np.median([r['rotation_deg'] for r in ok if r['axis_correct']==v])) if any(r['axis_correct']==v for r in ok) else None) for v in (True,False)}
        for a,b in [('S1','S0'),('S2','S1'),('S2','S0')]:
            contrasts[group][a+'-'+b]=dict(PCK10_delta_pp=100*(results[group][a]['twoD']['PCK']['10']-results[group][b]['twoD']['PCK']['10']),
                ADDsym_delta=results[group][a]['sixD']['ADDsym_AUC']-results[group][b]['sixD']['ADDsym_AUC'],
                axis_delta_pp=100*(results[group][a]['sixD']['axis_accuracy']-results[group][b]['sixD']['axis_accuracy']),
                **transitions([metrics[b][i] for i in ii],[metrics[a][i] for i in ii]))
    traceaudit={};reference_inventory=None
    for mat in C.MATERIALS:
        traces={a:C.read(C.RAW/f'TRACE_{mat}_{a}.json') for a in C.ARMS}
        for a in C.ARMS:
            fit=C.read(C.DOC/f'FIT_{mat}_{a}.json')
            if reference_inventory is None:reference_inventory=fit['trainable_inventory'];reference_initial=fit['initial_state']
            assert fit['trainable_inventory']==reference_inventory and fit['initial_state']==reference_initial
        for i in range(5120):
            rr=[traces[a][i] for a in C.ARMS]
            for key in ('occ','base_RGB_sha256','target_sha256','box_sha256','real'):assert len({r[key] for r in rr})==1
            if not rr[0]['real'] or not rr[1]['applied']:assert len({r['input_sha256'] for r in rr})==1
            assert rr[1]['applied']==rr[2]['applied']
        traceaudit[mat]=dict(occurrences=5120,parity=True,initialization_identical=True,inventory_identical=True)
    C.save(C.RAW/'FRAME_METRICS.json',metrics);C.save(C.RAW/'POSE_METRICS.json',pm)
    C.save(C.DOC/'RESULTS.json',dict(groups=results,contrasts=contrasts,common_matched_supplement=common,axis_conditioned_posthoc=axis))
    C.save(C.DOC/'TRAINING_PARITY.json',traceaudit)
    print('PRIMARY',json.dumps({g:contrasts[g] for g in C.H.P.SEVERITIES}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','score']);p.add_argument('--material',choices=C.MATERIALS);p.add_argument('--arm',choices=C.ARMS);a=p.parse_args()
    if a.stage=='infer':infer(a.material,a.arm)
    else:score()
