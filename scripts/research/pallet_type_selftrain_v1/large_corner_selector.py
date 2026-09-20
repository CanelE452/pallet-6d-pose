"""Source-only learned view/identity utility with frozen real-evaluation decisions."""
import argparse
import copy
import json
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor
from ultralytics import YOLO
from . import large_corner_views as L
from . import large_corner_selector_features as F
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM
from scripts.research.pallet_posefix_large_error_v1.evaluate import recovery_damage

C=L.C
PHASE='source_selector'
DOC=L.DOC/PHASE
RAW=L.RAW/PHASE
ARM='SOURCE_SELECTOR'
SPLIT=C.ROOT/'_docs/experiments/pallet_posefix_utility_selector_v1/SPLIT.json'
SIDECAR=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz'


def prepare():
    old=C.read(SPLIT);rng=np.random.default_rng(20260921);records=[];source=SourceData()
    forbidden={r['image']['sha256'] for r in C.read(L.DOC/'views/EVAL_PROTOCOL.json')['records']}
    for split,count in [('train',512),('val',128)]:
        pool=[r for r in old['records'] if r['split']==split]
        for index in sorted(rng.choice(len(pool),count,replace=False).tolist()):
            r=copy.deepcopy(pool[index]);meta=source.data.source['records'][int(source.data.indices[r['row']])]
            assert meta['source_kind']=='synthetic' and meta['reflect_pad_px']==100 and len(meta['targets'])==1
            assert r['image']['sha256'] not in forbidden;C.verify(r['image'])
            r['label']=C.bound(meta['label']);records.append(r)
    assert not ({r['scenario'] for r in records if r['split']=='train'} & {r['scenario'] for r in records if r['split']=='val'})
    assert not ({r['image']['sha256'] for r in records if r['split']=='train'} & {r['image']['sha256'] for r in records if r['split']=='val'})
    sources=[C.bound(p) for p in [__file__,F.__file__,Path(__file__).with_name('test_large_corner_selector.py'),
        L.__file__,SPLIT,SIDECAR,L.DOC/'NEXT_SOURCE_SELECTOR.md',L.DOC/'views/OUTPUTS_LOCK.json',C.N.E.R0]]
    protocol=dict(phase=PHASE,records=records,sources=sources,
        training='512 source train rows,128 scenario-disjoint validation rows; clean plus fixed RGB stress; no real GT training.',
        hypotheses='6 R0 views x native/reindexed90 output channels. Official per-source-row symmetry only in target scoring; fixedC2 disagreement feature for all rows. No additional real scoring symmetry.',
        features='183 RGB/prediction-only features; no synthetic/real GT, GT-derived crop, camera, CAD, or depth inputs.',
        stress='RGB*.65 + Gaussian sigma12 + Gaussian blur3/.8 + predicted-box-centered random4percent-area occluder. SeedSequence([20260921,row]). No coordinate perturbation.',
        target='100*(R0 mean8 error - candidate mean8 error)/predicted box diagonal, clipped[-100,100]; same per-row approved GT symmetry. All output boxes/center/confidences fixed to R0.',
        learner=dict(n_estimators=300,min_samples_leaf=15,max_depth=10,random_state=20260921,n_jobs=4),
        calibration_grid=[0,.25,.5,1,2,4,8],calibration='Source validation only: maximize matched >20to<=10 recoveries subject to <=1percent <5to>10 damage, also <=1percent clean-only damage. Ties fewer damaged then higher threshold. No passing threshold -> R0.',
        coverage_gate='At least10 distinct TRAIN rows and3 distinct VAL rows with a GT-best reindexed90 candidate improving mean by>1percent box diagonal; otherwise stop before fit, do not fabricate identity-corruption coverage.',
        exposure='R0 original validation/checkpoint selection and historical source-development exposure exist; selector train/val scenario separation is not globally untouched evaluation.',
        real_policy='Freeze fit+threshold before real features/choices. Freeze all194 outputs before scoring. No real threshold/model sweep; not an independent test.',
        dtype='R0 float32,cuDNN TF32 True,matmul TF32 False',sklearn_version=sklearn.__version__,
        no_new_real_GT=True,auto_promote=False,goal_complete=False)
    C.freeze(DOC/'PROTOCOL.json',protocol)
    with L.roots():L.R.evaluation_protocol(PHASE,[ARM],sources+[C.bound(DOC/'PROTOCOL.json')])
    print('SELECTOR_PROTOCOL_LOCKED',len(records),flush=True)


@torch.no_grad()
def predict_candidates(model,image):
    views={}
    for name in L.VARIANTS:
        if name=='R0':view=image;meta=dict(offset=[0,0],flip=False,original_width=image.shape[1]);size=640
        else:
            if views['R0'] is None:views[name]=None;continue
            try:view,meta,size=L.transform(image,views['R0']['box_xyxy'],name)
            except AssertionError:views[name]=None;continue
        padded=cv2.copyMakeBorder(view,100,100,100,100,cv2.BORDER_REFLECT_101)
        result=model.predict(padded,conf=.001,imgsz=size,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False,stream=False)[0]
        candidates=[]
        if result.boxes is not None and len(result.boxes):
            boxes=result.boxes.xyxy.cpu().numpy()-100;scores=result.boxes.conf.cpu().numpy()
            points=result.keypoints.xy.cpu().numpy()-100;conf=result.keypoints.conf.cpu().numpy()
            for j in range(len(scores)):
                candidates.append(L.restore(dict(candidate_index=j,score=float(scores[j]),box_xyxy=boxes[j].tolist(),keypoints_xy=points[j].tolist(),keypoints_conf=conf[j].tolist()),meta))
        if name=='R0':p=max(candidates,key=lambda x:x['score']) if candidates else None
        else:
            valid=[p for p in candidates if p['score']>=.25 and np.isfinite(p['keypoints_xy']).all()]
            p=max(valid,key=lambda p:L.R.E.O.iou(p['box_xyxy'],views['R0']['box_xyxy'])) if valid else None
            if p is not None and L.R.E.O.iou(p['box_xyxy'],views['R0']['box_xyxy'])<.5:p=None
        views[name]=p
    return views


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    for r in p['records']:C.verify(r['image']);C.verify(r['label'])
    return p


@torch.no_grad()
def generate():
    protocol=verify();C.N.setup();torch.set_num_interop_threads(1);assert torch.cuda.is_available();C.N.E.gpu()
    torch.backends.cudnn.allow_tf32=True;model=YOLO(str(C.N.E.R0),task='pose');source=SourceData();side=np.load(SIDECAR)
    cases=[];started=time.monotonic()
    for i,record in enumerate(protocol['records']):
        prepared=cv2.imread(str(C.ROOT/record['image']['path']));assert prepared is not None
        image=prepared[100:-100,100:-100].copy()
        assert np.array_equal(cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101),prepared),'Prepared source padding mismatch'
        clean=None
        for variant in ['clean','stress']:
            dest=RAW/'cases'/f"{record['row']:05d}_{variant}.json"
            if dest.exists():
                case=C.read(dest);assert case['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
                if variant=='clean':clean=case['views']
            else:
                im=image if variant=='clean' else (F.stress_image(image,clean['R0']['box_xyxy'],record['row']) if clean['R0'] else image)
                views=predict_candidates(model,im)
                if variant=='clean':clean=views
                case=dict(row=record['row'],id=f"{record['id']}:{variant}",split=record['split'],scenario=record['scenario'],variant=variant,views=views,protocol_sha256=C.sha(DOC/'PROTOCOL.json'))
                if views['R0'] is None:case.update(available=False,hypotheses=[])
                else:
                    hypotheses=F.hypotheses(views);evidence=F.image_evidence(im)
                    x=[F.features(im,views,h,evidence).tolist() for h in hypotheses]
                    # Synthetic supervision is accessed ONLY after candidate inputs/features exist.
                    item=source.item(record['row']);gt=item['original_gt']-100;valid=item['original_gt_valid']
                    meta=source.data.source['records'][int(source.data.indices[record['row']])];label=meta['targets'][0]
                    expected=np.array(label['keypoints_normalized']);expected[:,:2]*=np.array(prepared.shape[:2][::-1]);expected[:,:2]-=100
                    np.testing.assert_allclose(gt[valid],expected[valid,:2],atol=.003,rtol=0)
                    box=np.asarray(label['box_xywh_normalized']);center=box[:2]*prepared.shape[:2][::-1]-100;size=box[2:]*prepared.shape[:2][::-1]
                    matched=L.R.E.O.iou(views['R0']['box_xyxy'],np.r_[center-size/2,center+size/2])>=.5
                    perms=side['permutations'][record['row'],:side['order'][record['row']]].tolist()
                    scores=[EM.measure(h['candidate']['keypoints_xy'],gt,valid,perms,im.shape[:2],matched,True) for h in hypotheses]
                    diag=max(float(np.linalg.norm(np.array(views['R0']['box_xyxy'])[2:]-np.array(views['R0']['box_xyxy'])[:2])),1.)
                    for h,feature,s in zip(hypotheses,x,scores):
                        h.update(features=feature,metrics=dict(id=case['id'],**s),utility=float(np.clip(100*(scores[0]['frame_mean_px']-s['frame_mean_px'])/diag,-100,100)))
                    case.update(available=True,hypotheses=hypotheses,synthetic_gt_used_only_for_targets=True)
                C.freeze(dest,case)
            cases.append(dict(path=str(dest.relative_to(C.ROOT)),row=record['row'],variant=variant,split=record['split']))
        if (i+1)%40==0 or i+1==len(protocol['records']):print('SELECTOR_SOURCE',i+1,'/640',round(time.monotonic()-started,1),'sec',C.N.E.gpu(),flush=True)
    coverage={}
    for split in ['train','val']:
        selected=[C.read(C.ROOT/r['path']) for r in cases if r['split']==split];available=[r for r in selected if r['available']]
        rows=set()
        for r in available:
            best=max(r['hypotheses'],key=lambda h:h['utility'])
            if best['reindex'] and best['utility']>1:rows.add(r['row'])
        coverage[split]=dict(cases=len(selected),available=len(available),identity_positive_rows=len(rows))
    passed=coverage['train']['identity_positive_rows']>=10 and coverage['val']['identity_positive_rows']>=3
    C.freeze(DOC/'SOURCE_COMPLETE.json',dict(complete=True,cases=cases,coverage=coverage,identity_coverage_gate=passed,protocol=C.bound(DOC/'PROTOCOL.json')))
    print('SELECTOR_SOURCE_COMPLETE',coverage,'coverage_gate',passed,flush=True)


def select_index(case,prediction,threshold):
    if threshold is None:return 0
    scores=np.asarray(prediction).copy();scores[0]=0.;i=int(np.argmax(scores))
    return i if i>0 and scores[i]>threshold else 0


def calibration_stats(cases,predictions,threshold):
    before=[];after=[];clean_before=[];clean_after=[];chosen=0
    for c,p in zip(cases,predictions):
        i=select_index(c,p,threshold);a=c['hypotheses'][0]['metrics'];b=c['hypotheses'][i]['metrics'];before.append(a);after.append(b);chosen+=i!=0
        if c['variant']=='clean':clean_before.append(a);clean_after.append(b)
    return dict(recovery=recovery_damage(before,after,True),clean=recovery_damage(clean_before,clean_after,True),selected=chosen)


def fit():
    protocol=verify();done=C.read(DOC/'SOURCE_COMPLETE.json')
    if not done['identity_coverage_gate']:
        C.freeze(DOC/'FIT_STOP.json',dict(reason='INSUFFICIENT_SYNTHETIC_IDENTITY_ERROR_COVERAGE',coverage=done['coverage'],source=C.bound(DOC/'SOURCE_COMPLETE.json')))
        print('SELECTOR_FIT_STOP',done['coverage'],flush=True);return
    cases=[C.read(C.ROOT/r['path']) for r in done['cases']];train=[r for r in cases if r['available'] and r['split']=='train'];val=[r for r in cases if r['available'] and r['split']=='val']
    x=np.array([h['features'] for c in train for h in c['hypotheses'][1:]],np.float32);y=np.array([h['utility'] for c in train for h in c['hypotheses'][1:]],np.float32)
    weights=np.array([1/(len(c['hypotheses'])-1) for c in train for h in c['hypotheses'][1:]])
    model=RandomForestRegressor(**protocol['learner']);model.fit(x,y,sample_weight=weights)
    predictions=[model.predict(np.array([h['features'] for h in c['hypotheses']],np.float32)).tolist() for c in val]
    candidates=[]
    for threshold in protocol['calibration_grid']:
        s=calibration_stats(val,predictions,threshold);s['threshold']=threshold
        s['pass']=s['recovery']['damage_rate'] is not None and s['clean']['damage_rate'] is not None and s['recovery']['damage_rate']<=.01 and s['clean']['damage_rate']<=.01
        candidates.append(s)
    valid=[s for s in candidates if s['pass']]
    best=max(valid,key=lambda s:(s['recovery']['recovered'],-s['recovery']['damaged'],s['threshold'])) if valid else None
    path=RAW/'selector.joblib';assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path)
    C.freeze(DOC/'FIT.json',dict(complete=True,checkpoint=C.bound(path),source=C.bound(DOC/'SOURCE_COMPLETE.json'),protocol=C.bound(DOC/'PROTOCOL.json'),
        train_cases=len(train),train_candidates=len(x),validation_cases=len(val),threshold=None if best is None else best['threshold'],
        calibration=candidates,selected=best,real_GT_used=False,real_features_used=False))
    print('SELECTOR_FIT_COMPLETE',json.dumps(best),flush=True)


def predict():
    verify();fit=C.read(DOC/'FIT.json');C.verify(fit['checkpoint']);model=joblib.load(C.ROOT/fit['checkpoint']['path'])
    frames=C.read(L.RAW/'views/INFERENCE.json');base={r['id']:r for r in C.read(L.RAW/'views/EVAL_PREDICTIONS_R0.json')['records']};rows=[];decisions=[]
    for f in frames:
        C.verify(f['image']);im=cv2.imread(str(C.ROOT/f['image']['path']));h=F.hypotheses(f['views']);e=F.image_evidence(im)
        p=model.predict(np.array([F.features(im,f['views'],q,e) for q in h]));i=select_index(None,p,fit['threshold'])
        original=base[f['id']];out=L.replace_corners(original['prediction'],h[i]['candidate'])
        rows.append(dict(id=f['id'],kind='PLASTIC',raw_hw=f['raw_hw'],prediction=out))
        decisions.append(dict(id=f['id'],choice=h[i]['name'],predicted_utilities=p.tolist(),threshold=fit['threshold'],names=[q['name'] for q in h]))
    C.freeze(RAW/f'EVAL_PREDICTIONS_{ARM}.json',dict(complete=True,arm=ARM,checkpoint=fit['checkpoint'],records=rows,GT_free_decisions=True))
    C.freeze(RAW/'DECISIONS.json',decisions)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(complete=True,fit=C.bound(DOC/'FIT.json'),artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{ARM}.json'),C.bound(RAW/'DECISIONS.json')],decisions_before_real_GT=True))
    print('SELECTOR_REAL_OUTPUTS_LOCKED',len(rows),'changed',sum(r['choice']!='R0:native' for r in decisions),flush=True)


def score():
    for b in C.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    with L.roots():r=L.R.score(PHASE,ARM,False)
    base=C.read(L.RAW/'views/SCREEN_R0.json');damage=recovery_damage(base['metrics'],r['metrics'],True)
    frames=sum(any(a is not None and a>20 and b<=10 for a,b in zip(old['canonical_errors'],new['canonical_errors'])) for old,new in zip(base['metrics'],r['metrics']) if old['matched'])
    flags=dict(recovered_at_least5=damage['recovered']>=5,recovered_frames_at_least3=frames>=3,damage_at_most1percent=damage['damage_rate']<=.01,PCK20_preserved=r['symmetry']['PCK']['20']>=base['symmetry']['PCK']['20'])
    result=dict(complete=True,hard_recovery=damage,recovered_frames=frames,flags=flags,followup_permitted=all(flags.values()),
        symmetry=r['symmetry'],pose=r['pose'],fit=C.bound(DOC/'FIT.json'),outputs_lock=C.bound(DOC/'OUTPUTS_LOCK.json'),
        goal_complete=False,selftraining_done=False,independent_confirmation=False)
    C.freeze(RAW/'SUMMARY.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','generate','fit','predict','score']);a=p.parse_args();globals()[a.action]()
