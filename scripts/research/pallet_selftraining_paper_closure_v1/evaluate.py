"""Reference access occurs only after immutable prediction/pose locks."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import math
import numpy as np
from . import common as C
from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary,transitions
from scripts.research.pallet_verified_anchor_v1.evaluate import point,metrics,difference
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D

def posejob(x):
    fid,p,g=x
    return fid,D.metric(fid,p,g)

def paired(a,b,pa,pb,rr):
    groups=V.groups(rr);order=groups['ALL'];wins=Counter()
    for i in order:
        d=np.mean(b[i]['errors'])-np.mean(a[i]['errors'])
        wins['improved' if d < -1e-8 else 'worsened' if d>1e-8 else 'unchanged']+=1
    clusters=[]
    for g,ids in groups.items():
        if not g.startswith('REC_'):continue
        count=sum(len(a[i]['errors']) for i in ids)
        delta=sum(sum(e<=10 for e in b[i]['errors'])-sum(e<=10 for e in a[i]['errors']) for i in ids)
        clusters.append((delta,count))
    z=np.array(clusters);rng=np.random.default_rng(20260927)
    w=rng.multinomial(len(z),np.ones(len(z))/len(z),10000)
    values=100*(w@z[:,0])/(w@z[:,1])
    return dict(frame_mean_error=dict(wins),PCK10_delta_pp=float(100*z[:,0].sum()/z[:,1].sum()),
        recording_bootstrap_CI95_pp=np.quantile(values,[.025,.975]).tolist(),clusters=len(z),
        exploratory_CI_not_search_adjusted=True,independent_corner_test=False,
        transitions=transitions([a[i] for i in order],[b[i] for i in order]),
        ADDsym_AUC_delta=D.aggregate([pb[i] for i in order])['ADDsym_AUC']-D.aggregate([pa[i] for i in order])['ADDsym_AUC'])

def main():
    if (C.DOC/'CORE_RESULTS.json').exists(): print('RESULTS_ALREADY_FROZEN');return
    lock=C.read(C.DOC/'PREDICTIONS_LOCK.json')
    for b in lock['files']: C.verify(b)
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    C.save(C.DOC/'SCORING_START.json',dict(utc=C.now(),prediction_lock=C.bind(C.DOC/'PREDICTIONS_LOCK.json')),True)
    rr=C.records();ids=[r['id'] for r in rr];by={r['id']:r for r in rr};groups=V.groups(rr)
    preds=C.read(C.RAW/'PREDICTIONS.json');poses=C.read(C.RAW/'POSE_PREDICTIONS.json')
    truth=C.read(C.TRUTH);_,gt=D.Pose.metadata('REAL_DEV')
    fm={};fixed={};pm={}
    for arm in C.ARMS:
        fm[arm]={};fixed[arm]={}
        for i in ids:
            cand=C.selected(preds[arm][i]);t=truth[i]
            matched=cand is not None and V.E.C.H.E.O.iou(cand['box_xyxy'],t['box'])>=.5
            q=np.full((9,2),np.nan) if cand is None else cand['keypoints_xy']
            fm[arm][i]=dict(id=i,**V.E.P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,cand is not None))
            fixed[arm][i]=dict(id=i,**V.E.P.M.measure(q,t['gt'],t['valid'],[list(range(9))],t['hw'],matched,cand is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:
            pm[arm]=dict(pool.map(posejob,[(i,poses[arm][i],gt[i]) for i in ids],chunksize=8))
        print('SCORED',arm,flush=True)
    scores={g:{a:dict(twoD=summary([fm[a][i] for i in ii]),fixed_ID=summary([fixed[a][i] for i in ii]),
        sixD=D.aggregate([pm[a][i] for i in ii])) for a in C.ARMS} for g,ii in groups.items()}
    # Cross-check unchanged baseline against an already published identical population/contract.
    previous=C.read(V.DOC/'RESULTS.json')['groups']
    for g in groups:
        D.close(scores[g]['R0']['twoD'],previous[g]['R0']['twoD'])
        for k,v in scores[g]['R0']['sixD'].items():D.close(v,previous[g]['R0']['current'][k])
    rows=[];final=C.read(C.FINAL)
    assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    for fi,ci in final['review_queue']:
        frame=final['frames'][fi];corner=frame['corners'][ci];fid=frame['frame_id']
        if fid not in by or corner['status']!='DIRECT_VISIBLE':continue
        assert ci<8 and corner['coordinate_source']=='manual_click' and frame['image_sha256']==by[fid]['image']['sha256']
        errors={};missing={};xy={}
        for a in preds:
            q=point(preds[a].get(fid,{}),ci);missing[a]=q is None;xy[a]=None if q is None else q.tolist()
            errors[a]=float(np.linalg.norm(q-corner['xy'])) if q is not None else math.hypot(*truth[fid]['hw'])
        rows.append(dict(frame_id=fid,corner_id=ci,severity=by[fid]['severity'],recording=by[fid]['recording_group'],
            verified_xy=corner['xy'],errors=errors,missing=missing,model_xy=xy))
    assert len(rows)==66 and len({r['frame_id'] for r in rows})==16
    ag={'ALL':rows,**{g:[r for r in rows if r['severity']==s] for g,s in V.SEVS.items()}}
    quality={g:{a:metrics([r['errors'][a] for r in rs]) for a in preds} for g,rs in ag.items()}
    C.save(C.RAW/'ANCHOR_POINTS.json',rows,True)
    C.save(C.RAW/'FRAME_METRICS.json',fm,True);C.save(C.RAW/'FIXED_ID_METRICS.json',fixed,True);C.save(C.RAW/'POSE_METRICS.json',pm,True)
    C.save(C.DOC/'PSEUDO_LABEL_QUALITY.json',dict(groups=quality,frames=16,points=66,reference=C.bind(C.FINAL),
        teacher_checkpoint=C.read(C.ROOT/'_docs/experiments/pallet_posefix_replay_v1/FIT.json')['checkpoint'],
        quality_scope='Held-out labeled DEV proxy, not measured accuracy of unlabeled adaptation217; no filter at eval; fixed native identity',
        coverage={a:sum(not r['missing'][a] for r in rows) for a in preds},
        difference=difference(rows,'R0','TEACHER'),
        transitions=dict(recovery20_to10=sum(r['errors']['R0']>20 and r['errors']['TEACHER']<=10 for r in rows),
            damage5_to10=sum(r['errors']['R0']<5 and r['errors']['TEACHER']>10 for r in rows)),independent_test=False),True)
    contrasts={}
    for s in C.SUFFIXES:
        for a in ('RAW_'+s,'SYN_'+s,'R0'):
            b='REF_'+s;contrasts[b+'-minus-'+a]=paired(fm[a],fm[b],pm[a],pm[b],rr)
    C.save(C.DOC/'PAIRED_ANALYSIS.json',contrasts,True)
    C.save(C.DOC/'CORE_RESULTS.json',dict(groups=scores,frames=128,recordings=7,reference='geometry-resolved annotations, NOT independent physical6D',
        fixed_common_selector='Original D9; keypoint8 enters selection residual, corner0..7 solve; unchanged for all arms',
        independent_confirmation=False,new_fits=0,R0_parity=True,
        artifact_sources=[C.bind(p) for p in (C.RAW/'PREDICTIONS.json',C.RAW/'POSE_PREDICTIONS.json',C.TRUTH,C.META,C.RAW/'FRAME_METRICS.json',C.RAW/'POSE_METRICS.json')]),True)
    for a in C.ARMS:
        v=scores['ALL'][a];print(a,'PCK10',v['twoD']['PCK']['10'],'AUC',v['sixD']['ADDsym_AUC'],flush=True)
    print('Q1',quality['ALL']['R0'],quality['ALL']['TEACHER'],flush=True)

if __name__=='__main__':main()
