import argparse
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scripts.research.pallet_clean19_structured_easyhard_v1.diagnostics import source_model
from scripts.research.pallet_clean19_structured_easyhard_v1.evaluate import transitions
from scripts.research.pallet_clean19_pose_sensitive_diag_v1.evaluate import pose_row
from . import common as E

def infer():
    E.immutable();E.G.deterministic();E.C.guard();records=E.read(E.DOC/'SPLIT_LOCK.json')['heldout'];ids={r['id'] for r in records}
    assert all(E.read(E.DOC/f'FIT_{a}.json')['complete'] for a in E.ARMS)
    for a in E.ARMS:
        fit=E.read(E.DOC/f'FIT_{a}.json');E.verify(fit['checkpoint']);model=YOLO(str(E.ROOT/fit['checkpoint']['path']),task='pose');preds={}
        for r in records:preds[r['id']]=E.C.predict(model,cv2.imread(str(E.ROOT/r['image']['path'])),100)
        E.save(E.RAW/f'HELDOUT_{a}.json',dict(predictions=preds,checkpoint=fit['checkpoint'],GT_input=False));del model;torch.cuda.empty_cache();print('HELDOUT_FROZEN',a,len(preds),flush=True)
    oldR0=E.read(E.P.RAW/'BASELINE_PREDICTIONS.json')['R0'];oldS1=E.read(E.C.RAW/'EVAL_PLASTIC_S1.json')['predictions']
    E.save(E.RAW/'REFERENCE_PREDICTIONS.json',{a:{i:pp[i] for i in ids} for a,pp in [('R0',oldR0),('OLD_S1',oldS1)]})
    E.save(E.DOC/'PREDICTION_LOCK.json',dict(files=[E.bind(E.RAW/f'HELDOUT_{a}.json') for a in E.ARMS]+[E.bind(E.RAW/'REFERENCE_PREDICTIONS.json')],all_before_scoring=True,only_new_heldout=True,frames=len(records),student_alone=True))

def score():
    E.immutable();records=E.read(E.DOC/'SPLIT_LOCK.json')['heldout'];ids=[r['id'] for r in records]
    for b in E.read(E.DOC/'PREDICTION_LOCK.json')['files']:E.verify(b)
    preds=E.read(E.RAW/'REFERENCE_PREDICTIONS.json');preds.update({a:E.read(E.RAW/f'HELDOUT_{a}.json')['predictions'] for a in E.ARMS})
    meta={r['id']:r for r in E.read(E.V.RAW/'INFERENCE_METADATA.json')};poses={}
    for a,pp in preds.items():
        poses[a]={}
        for fid,p in pp.items():
            q=E.D.points(p);m=meta[fid];K=np.array(m['K']);xyz=np.array(m['xyz']);sel=E.D.select(q,K,xyz)
            poses[a][fid]=dict(current=E.D.Pose.infer(q,K,xyz,False),hypotheses=[dict(name=h['name'],pose=E.D.hyp_pose(h,q,K,xyz),score=h['score']) for h in sel['hypotheses']])
    E.save(E.RAW/'POSE_PREDICTIONS.json',poses);E.save(E.DOC/'POSE_PREDICTION_LOCK.json',dict(file=E.bind(E.RAW/'POSE_PREDICTIONS.json'),before_pose_GT=True,production_D9_unchanged=True))
    truth=E.read(E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');_,gt=E.D.Pose.metadata('REAL_DEV');fm={};manual={};pm={};by={r['id']:r for r in records}
    for a,pp in preds.items():
        fm[a]={};manual[a]={}
        for fid,p in pp.items():
            t=truth[fid];c=E.P.C.selected(p);matched=c is not None and E.C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5;q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
            fm[a][fid]=dict(id=fid,**E.P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
            mq,mask=E.manual(by[fid]);manual[a][fid]=dict(id=fid,**E.P.M.measure(q,mq,mask,t['permutations'],t['hw'],matched,c is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:pm[a]=dict(pool.map(pose_row,[(fid,poses[a][fid],gt[fid]) for fid in ids],chunksize=8))
        print('SCORED',a,flush=True)
    groups={'HELDOUT_ALL_PLASTIC':ids}
    for s in E.P.SEVERITIES:groups['HELDOUT_'+s.split('_')[0]]=[r['id'] for r in records if r['severity']==s]
    for rec in sorted({r['recording_group'] for r in records}):groups[rec]=[r['id'] for r in records if r['recording_group']==rec]
    results={};tt={};common={}
    for g,ii in groups.items():
        results[g]={};tt[g]={};common[g]={}
        if not ii:results[g]=dict(status='N/A',frames=0);continue
        matched=[i for i in ii if all(fm[a][i]['matched'] for a in E.ARMS)]
        for a in preds:
            rows=[fm[a][i] for i in ii];two=E.P.M.summary(rows);two['correct']={str(k):sum(e<=k for r in rows for e in r['errors']) for k in (5,10,20)}
            current=E.D.aggregate([pm[a][i]['current'] for i in ii]);current['coverage']=current['available']/len(ii);oracle=E.D.aggregate([pm[a][i]['oracle'] for i in ii])
            man=E.P.M.summary([manual[a][i] for i in ii])
            results[g][a]=dict(twoD=two,manual_only=man,current=current,oracle=oracle,selection_loss=oracle['ADDsym_AUC']-current['ADDsym_AUC'])
            common[g][a]=E.P.M.summary([fm[a][i] for i in matched])
        for left,right in zip(E.ARMS,E.ARMS[1:]):tt[g][right+'-minus-'+left]=transitions([fm[left][i] for i in ii],[fm[right][i] for i in ii])
    E.save(E.RAW/'FRAME_METRICS.json',fm);E.save(E.RAW/'MANUAL_METRICS.json',manual);E.save(E.RAW/'POSE_METRICS.json',pm)
    E.save(E.DOC/'RESULTS.json',dict(groups=results,common_matched_supplement=common,oracle_label='POSTHOC GT ORACLE_WD — NONDEPLOYABLE',manual_provenance_not_visibility=True,old300_direct_comparison_prohibited=True,recordings=len({r['recording_group'] for r in records}),uncertainty='Small fixed number of independent recording groups; per-recording deltas reported, no corner-independent significance or precise CI'))
    E.save(E.DOC/'TRANSITIONS.json',tt)

def diagnostics():
    E.G.deterministic();E.immutable();E.read(E.DOC/'PREDICTION_LOCK.json');targets=E.read(E.RAW/'TARGETS.json');allsource={};train={};raw={}
    for a in E.ARMS:
        fit=E.read(E.DOC/f'FIT_{a}.json');model=YOLO(str(E.ROOT/fit['checkpoint']['path']),task='pose');source=source_model(model);allsource[a]=E.C.summary(source);rows=[]
        for r in targets:
            if r['role']=='E' and a!=E.ARMS[0]:continue
            if r['role']=='H' and a==E.ARMS[0]:continue
            p=E.C.predict(model,cv2.imread(str(E.ROOT/r['image']['path'])),100)
            rows.append(dict(id=r['id'],role=r['role'],prediction=p,pseudo=E.C.native(p,r['target'],r['mask'],r['hw']),manual=E.C.native(p,r['manual'],r['mask'],r['hw'])))
        raw[a]=dict(source=source,train=rows);train[a]={role:{target:E.C.summary([r[target] for r in rows if r['role']==role]) for target in ('pseudo','manual')} for role in ('C0','E','H')}
        del model;torch.cuda.empty_cache();print('DIAGNOSTICS',a,flush=True)
    allsource['R0']=E.C.summary(E.read(E.C.RAW/'PROBE_BEFORE_PREDICTIONS.json')['PLASTIC']['R0']['source']);allsource['OLD_S1']=E.C.summary(E.read(E.C.RAW/'DIAGNOSTICS_PLASTIC_S1.json')['source'])
    teacher=[]
    for r in targets:
        if r['role']=='H':teacher.append(dict(id=r['id'],errors=np.linalg.norm(np.array(r['target'])-np.array(r['manual']),axis=1)[np.array(r['mask'])].tolist()))
    E.save(E.RAW/'DIAGNOSTICS.json',raw);E.save(E.DOC/'TRAIN_DIAGNOSTICS.json',dict(student=train,teacher_manual_on_fixed_H=E.C.summary(teacher),teacher_manual_rows=teacher,not_generalization=True,hidden_independent_GT=False))
    # Reuse previously verified exact source256 pose metadata and evaluator, no new algorithm.
    binding=E.read(E.G.DOC/'SOURCE_GEOMETRY_BINDING.json');assert binding['bound']==256
    for r in binding['records']:
        for key in ('image','label','renderer'):E.verify(r[key])
    E.verify(binding['table']);table=np.load(E.ROOT/binding['table']['path']);ix={str(s):i for i,s in enumerate(table['stems'])};_,gt=E.D.Pose.metadata('SYNTH_HELDOUT');sp={}
    for a in E.ARMS:
        poses={}
        for r in raw[a]['source']:
            i=ix[r['id']];fx,fy,cx,cy=table['K'][i];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            poses[r['id']]=E.D.Pose.infer(E.D.points(r['prediction']),K,table['dims'][i],True)
        with ProcessPoolExecutor(max_workers=4) as pool:metrics=list(pool.map(E.D.Pose.metric,[(fid,p,gt[fid]) for fid,p in poses.items()],chunksize=8))
        for m in metrics:
            if m['available']:m['axis_correct']=bool(abs(poses[m['id']]['cf_extents'][0]-gt[m['id']]['body_xyz'][0])<1e-6)
        sp[a]=E.D.aggregate(metrics)
    E.save(E.DOC/'SOURCE_PRESERVATION.json',dict(twoD=allsource,pose=sp,exact_heldout256=True,geometry_binding=E.bind(E.G.DOC/'SOURCE_GEOMETRY_BINDING.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','score','diagnostics']);a=p.parse_args();globals()[a.stage]()
