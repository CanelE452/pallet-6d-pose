"""Score frozen predictions; no model or selection changes are made here."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import common as C
from .infer import ARMS
from scripts.research.pallet_selector_recovery_v1 import common as F


def save(p,x):C.save(p,F.clean(x),immutable=True)


def main():
    pl=C.read(C.DOC/'POSE_DECISIONS_LOCK.json');C.verify(pl['poses']);C.verify(pl['raw_lock'])
    rl=C.read(C.DOC/'RAW_PREDICTIONS_LOCK.json');C.verify(rl['predictions'])
    assert rl['created_at']<=pl['created_at']<C.now()
    if (C.DOC/'RESULTS.json').exists():
        finish_decision(C.read(C.DOC/'RESULTS.json')['groups'],C.read(C.DOC/'TRAIN_FIT.json')['groups']);return
    if not (C.DOC/'SCORING_START.json').exists():
        save(C.DOC/'SCORING_START.json',dict(created_at=C.now(),pose_lock=C.bind(C.DOC/'POSE_DECISIONS_LOCK.json')))
    else:C.verify(C.read(C.DOC/'SCORING_START.json')['pose_lock'])
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary,transitions
    from scripts.research.pallet_clean19_pose_sensitive_diag_v1.evaluate import pose_row
    from scripts.research.pallet_verified_anchor_v1.evaluate import point,metrics
    rows=V.records();groups=V.groups(rows);pred=C.read(C.RAW/'RAW_PREDICTIONS.json');poses=C.read(C.RAW/'POSE_DECISIONS.json')
    truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');_,gt=V.E.D.Pose.metadata('REAL_DEV')
    selected=C.read(C.DOC/'HARD_SELECTION_PUBLIC.json')['rows'];recs={r['recording'] for r in selected if r['assignment']=='INITIAL'}
    assert not recs&{r['recording'] for r in rows}
    fm={};pm={};result={};tt={}
    baseline=C.read(C.ROOT/'_docs/experiments/pallet_single_model_preserve_v1/BASELINE_LOCK.json')
    for arm in ARMS:
        fm[arm]={}
        for r in rows:
            fid=r['id'];t=truth[fid];p=pred['real'][arm][fid];c=F.selected(p)
            matched=c is not None and V.E.C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
            fm[arm][fid]=dict(id=fid,**V.E.P.M.measure(np.full((9,2),np.nan) if c is None else c['keypoints_xy'],t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:
            pm[arm]=dict(pool.map(pose_row,[(r['id'],poses['real'][arm][r['id']],gt[r['id']]) for r in rows],chunksize=8))
        print('HARD_SCORE',arm,flush=True)
    for group,ids in groups.items():
        result[group]={}
        for arm in ARMS:
            cur=V.E.D.aggregate([pm[arm][i]['current'] for i in ids]);oracle=V.E.D.aggregate([pm[arm][i]['oracle'] for i in ids])
            result[group][arm]=dict(twoD=summary([fm[arm][i] for i in ids]),current=cur,oracle=oracle,selection_loss=oracle['ADDsym_AUC']-cur['ADDsym_AUC'])
        V.E.D.close(result[group]['BASE'],baseline['groups'][group])
        for arm in ARMS:
            result[group][arm]['D9']=V.E.D.aggregate([V.E.D.metric(i,poses['real'][arm][i]['D9'],gt[i]) for i in ids])
        tt[group]={a:transitions([fm['BASE'][i] for i in ids],[fm[a][i] for i in ids]) for a in ARMS[1:]}
    final=C.read(V.FINAL);assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    assert C.sha(V.FINAL)==C.read(V.ANCHOR/'METADATA_QA_FINAL.json')['final_reference_sha256']
    by={r['id']:r for r in rows};anchor=[]
    for fi,ci in final['review_queue']:
        f=final['frames'][fi];fid=f['frame_id'];k=f['corners'][ci]
        if fid not in by or k['status']!='DIRECT_VISIBLE':continue
        assert ci<8 and k['coordinate_source']=='manual_click'
        errors={}
        for a in ARMS:
            p=point(pred['real'][a][fid],ci);errors[a]=float(np.linalg.norm(p-np.array(k['xy']))) if p is not None else float(np.hypot(*truth[fid]['hw']))
        anchor.append(dict(frame_id=fid,corner_id=ci,severity=by[fid]['severity'],errors=errors))
    assert len(anchor)==66;aa={}
    for group,sev in [('ALL',None),('HARD','HARD'),('CLEAN','CLEAN'),('MODERATE','MODERATE_OCCLUSION'),('SEVERE','SEVERE_OCCLUSION')]:
        subset=[r for r in anchor if sev is None or (r['severity']!='CLEAN' if sev=='HARD' else r['severity']==sev)]
        aa[group]={a:metrics([r['errors'][a] for r in subset]) for a in ARMS};V.E.D.close(aa[group]['BASE'],baseline['anchors'][group])
    save(C.RAW/'FRAME_METRICS.json',fm);save(C.RAW/'POSE_METRICS.json',pm);save(C.RAW/'ANCHOR_METRICS.json',anchor)
    save(C.DOC/'VERIFIED_VISIBLE.json',dict(groups=aa,reference=final['reference_version'],binding=C.bind(V.FINAL)))
    save(C.DOC/'TRANSITIONS.json',tt)
    source(pred['source'],poses['source'],V,metrics)
    ll=C.read(C.DOC/'HARD_LABEL_LOCK.json');frames=C.read(C.ROOT/ll['labels']['path'])['frames'];fit={}
    for arm in ARMS:
        errors=[];perframe={}
        for fid,f in frames.items():
            p=F.selected(pred['train'][arm][fid]);e=[]
            for k,c in enumerate(f['corners']):
                if c['status']=='DIRECT_VISIBLE':e.append(float(np.linalg.norm(np.array(p['keypoints_xy'][k])-c['xy'])) if p else float(np.hypot(*f['size'])))
            perframe[fid]=metrics(e);errors+=e
        fit[arm]=dict(total=metrics(errors),per_frame=perframe)
    save(C.DOC/'TRAIN_FIT.json',dict(groups=fit,training_not_generalization=True))
    save(C.DOC/'RESULTS.json',dict(groups=result,already_viewed_DEV=True,independent_test=False,base_reproduced=True,
         selected_train_recording_intersection=[],same_frozen_GEO_LINEAR=True,oracle='GT POSTHOC NONDEPLOYABLE'))
    finish_decision(result,fit)


def finish_decision(result,fit):
    delta={g:{k:result[g]['H_MANUAL'][k]['ADDsym_AUC']-result[g]['BASE'][k]['ADDsym_AUC'] for k in ('current','oracle')} for g in ('CLEAN','MODERATE','SEVERE')}
    for g in delta:delta[g]['PCK10']=result[g]['H_MANUAL']['twoD']['PCK']['10']-result[g]['BASE']['twoD']['PCK']['10']
    hard=['MODERATE','SEVERE'];improved=[g for g in hard if delta[g]['current']>0]
    signal=any(delta[g]['PCK10']>0 or delta[g]['oracle']>0 for g in hard)
    deploy=bool(improved) and all(delta[g]['current']>=0 for g in hard+['CLEAN']) and all(delta[g]['PCK10']>0 or delta[g]['oracle']>0 for g in improved)
    if deploy:decision='HARD_SUPERVISION_DEPLOYABLE_GAIN'
    elif (improved or signal) and delta['CLEAN']['current']<0:decision='HARD_GAIN_CLEAN_TRADEOFF'
    elif signal and not improved:decision='HARD_SUPERVISION_LOCALIZATION_SIGNAL_SELECTOR_LIMIT'
    else:decision='NO_CONSISTENT_HARD_SUPERVISION_GAIN'
    save(C.DOC/'DECISION.json',dict(primary=decision,deltas=delta,stop_more_hard_labeling=True,auto_promote=False,
         rule='predeclared section13; mixed hard-severity current gains/losses without clean tradeoff are not deployable gain',
         manual_fit_PCK10=fit['H_MANUAL']['total']['PCK']['10'],single_seed=True))
    C.set_state('EVALUATED_REPORT_PENDING',training='COMPLETE',decision=decision)
    print('HARD_DECISION',decision,delta,flush=True)


def source(pred,poses,V,metrics):
    from scripts.research.pallet_selector_recovery_v1.synth_labels import exact
    from scripts.research.pallet_selector_recovery_v1.features import cuboid
    from challenge.evaluation_v2.pose_metrics import pose_auc
    inputs=C.read(C.RAW/'INFERENCE_INPUTS.json')['source'];gt=exact(inputs)
    records=C.read(F.STRUCT/'SOURCE_PROBE_PLAN.json')['records'];out={};maxerr=0
    for arm in ARMS:
        errors=[];adds=[]
        for r in records:
            fid=r['id'];h,w=r['hw'];t=r['targets'][0];kp=np.array(t['keypoints_normalized']);q=kp[:,:2]*[w,h];valid=kp[:,2]>0;valid[8]=False
            maxerr=max(maxerr,float(np.linalg.norm(gt[fid]['keypoints']-q,axis=1)[valid].max()))
            b=np.array(t['box_xywh_normalized'])*[w,h,w,h];box=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2];p=F.selected(pred[arm][fid]);matched=p is not None and V.E.C.H.E.O.iou(p['box_xyxy'],box)>=.5
            err=np.linalg.norm(np.array(p['keypoints_xy'])-q,axis=1) if matched else np.full(9,np.hypot(h,w));err[~np.isfinite(err)]=np.hypot(h,w);errors+=err[valid].tolist()
            pose=poses[arm][fid]['current'];g=gt[fid];value=float('inf')
            if pose['available']:
                x=cuboid(*g['dims'])[:8];pr=x@np.array(pose['R_physical']).T+pose['centroid']
                value=min(float(np.linalg.norm(pr-(x@(g['R']@sym).T+g['t']),axis=1).mean()/np.linalg.norm(g['dims'])) for sym in (np.eye(3),np.diag([-1.,1.,-1.])))
            adds.append(value)
        out[arm]=dict(twoD=metrics(errors),pose=dict(ADDsym_AUC=pose_auc(adds),median_ADDnorm=np.median(adds),available=int(np.isfinite(adds).sum()),frames=256))
    assert maxerr<.05
    old=C.read(C.ROOT/'_docs/experiments/pallet_single_model_preserve_v1/SOURCE_PRESERVATION.json')['groups']['BASE'];V.E.D.close(out['BASE'],old)
    save(C.DOC/'SOURCE_PRESERVATION.json',dict(groups=out,frames=256,exact_projection_max_px=maxerr,base_reproduced=True))


if __name__=='__main__':main()
