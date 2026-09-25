"""Reference scoring only after raw predictions and selector decisions freeze."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import common as C

def main():
    lock=C.read(C.DOC/'POSE_DECISIONS_LOCK.json');C.verify(lock['poses']);C.verify(lock['raw_lock']);rawlock=C.read(C.DOC/'RAW_PREDICTIONS_LOCK.json');C.verify(rawlock['predictions']);now=C.now();assert now>lock['created_at']>rawlock['created_at']
    C.freeze(C.DOC/'REFERENCE_READ_LOCK.json',dict(created_at=now,pose_lock=C.bind(C.DOC/'POSE_DECISIONS_LOCK.json')))
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary,transitions
    from scripts.research.pallet_clean19_pose_sensitive_diag_v1.evaluate import pose_row
    from scripts.research.pallet_verified_anchor_v1.evaluate import point,metrics
    rr=V.records();groups=V.groups(rr);pr=C.read(C.RAW/'RAW_PREDICTIONS.json');ps=C.read(C.RAW/'POSE_DECISIONS.json');truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');_,gt=V.E.D.Pose.metadata('REAL_DEV');fm={};pm={}
    for a in ('BASE','PRES1'):
        fm[a]={}
        for r in rr:
            fid=r['id'];p=pr['real'][a][fid];t=truth[fid];c=C.selected(p);matched=c is not None and V.E.C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
            fm[a][fid]=dict(id=fid,**V.E.P.M.measure(np.full((9,2),np.nan) if c is None else c['keypoints_xy'],t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:pm[a]=dict(pool.map(pose_row,[(r['id'],ps[a][r['id']],gt[r['id']]) for r in rr],chunksize=8))
    oldfm=C.read(C.PREV_RAW/'FRAME_METRICS.json');oldpm=C.read(C.PREV_RAW/'POSE_METRICS.json');fm['PRES1_D9']=fm['PRES1'];pm['PRES1_D9']={}
    for fid,p in ps['PRES1'].items():
        current=next((h['metric'] for h in pm['PRES1'][fid]['hypotheses'] if h['name']==p['D9'].get('selected_hypothesis')),dict(id=fid,available=False));pm['PRES1_D9'][fid]=dict(pm['PRES1'][fid],current=current)
    for a in ('S0','R0'):fm[a]=oldfm[a];pm[a]=oldpm[a]
    res={};tt={};baseline=C.read(C.DOC/'BASELINE_LOCK.json')
    for g,ids in groups.items():
        res[g]={}
        for a in fm:
            cur=V.E.D.aggregate([pm[a][i]['current'] for i in ids]);oracle=V.E.D.aggregate([pm[a][i]['oracle'] for i in ids]);res[g][a]=dict(twoD=summary([fm[a][i] for i in ids]),current=cur,oracle=oracle,selection_loss=oracle['ADDsym_AUC']-cur['ADDsym_AUC'])
        V.E.D.close(res[g]['BASE'],baseline['groups'][g]);tt[g]=transitions([fm['BASE'][i] for i in ids],[fm['PRES1'][i] for i in ids])
    final=C.read(V.FINAL);assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2';qa=C.read(V.ANCHOR/'METADATA_QA_FINAL.json');assert C.sha(V.FINAL)==qa['final_reference_sha256'];by={r['id']:r for r in rr};anchor=[]
    for fi,ci in final['review_queue']:
        f=final['frames'][fi];fid=f['frame_id'];k=f['corners'][ci]
        if fid not in by or k['status']!='DIRECT_VISIBLE':continue
        assert ci<8 and k['coordinate_source']=='manual_click';errs={}
        for a in ('BASE','PRES1'):
            q=point(pr['real'][a].get(fid,{}),ci);errs[a]=float(np.linalg.norm(q-np.array(k['xy']))) if q is not None else float(np.hypot(*truth[fid]['hw']))
        anchor.append(dict(frame_id=fid,corner_id=ci,severity=by[fid]['severity'],recording=by[fid]['recording_group'],errors=errs))
    aa={}
    for g,sev in [('ALL',None),('HARD','HARD'),('CLEAN','CLEAN'),('MODERATE','MODERATE_OCCLUSION'),('SEVERE','SEVERE_OCCLUSION')]:
        rows=[r for r in anchor if sev is None or (r['severity']!='CLEAN' if sev=='HARD' else r['severity']==sev)];aa[g]={a:metrics([r['errors'][a] for r in rows]) for a in ('BASE','PRES1')};V.E.D.close(aa[g]['BASE'],baseline['anchors'][g])
    assert len(anchor)==66
    C.freeze(C.RAW/'FRAME_METRICS.json',fm);C.freeze(C.RAW/'POSE_METRICS.json',pm);C.freeze(C.RAW/'ANCHOR_POINT_METRICS.json',anchor)
    C.freeze(C.DOC/'RESULTS.json',dict(groups=res,transitions=tt,reference='same already-viewed HELDOUT128',same_GEO_LINEAR=True,oracle='POSTHOC GT only, nondeployable',base_reproduced=True))
    C.freeze(C.DOC/'VERIFIED_VISIBLE.json',dict(groups=aa,reference=final['reference_version'],binding=C.bind(V.FINAL),fixed_identity=True,no_training=True,counts=dict(ALL=66,HARD=36,CLEAN=30,MODERATE=22,SEVERE=14)))
    source(pr['source'],V)
    delta={g:res[g]['PRES1']['current']['ADDsym_AUC']-res[g]['BASE']['current']['ADDsym_AUC'] for g in ('CLEAN','MODERATE','SEVERE','ALL')};clean=delta['CLEAN']>0;hard=delta['MODERATE']>=0 and delta['SEVERE']>=0
    decision='PRESERVATION_SUPPORTED' if clean and hard else 'CLEAN_RECOVERY_HARD_TRADEOFF' if clean else 'NO_CLEAN_RECOVERY'
    warning=aa['HARD']['PRES1']['PCK']['10']['correct']<aa['HARD']['BASE']['PCK']['10']['correct']
    C.freeze(C.DOC/'DECISION.json',dict(created_at=C.now(),primary=decision,delta_AUC=delta,verified_HARD_PCK10_warning=warning,hard_correct_delta=aa['HARD']['PRES1']['PCK']['10']['correct']-aa['HARD']['BASE']['PCK']['10']['correct'],
        hard_labeling='MIN_HARD_LABELING_NOT_NEEDED_NOW' if decision=='PRESERVATION_SUPPORTED' else 'PENDING_FIXED_SUPERVISION_GAP_DIAGNOSTIC',final_candidate=decision=='PRESERVATION_SUPPORTED',final_model_claim=False))
    print('PRESERVATION_DECISION',decision,delta,'anchor_warning',warning,flush=True)

def source(pred,V):
    from scripts.research.pallet_selector_recovery_v1.synth_labels import exact,addnorm
    from scripts.research.pallet_selector_recovery_v1 import features as F,models as M
    from challenge.evaluation_v2.pose_metrics import pose_auc
    import torch
    rec=C.read(C.STRUCT/'SOURCE_PROBE_PLAN.json')['records'];inputs=C.read(C.RAW/'INFERENCE_INPUTS.json')['source'];gt=exact(inputs);md={r['id']:r for r in inputs};ck=torch.load(C.ROOT/C.read(C.RAW/'INFERENCE_BINDINGS.json')['scorer']['path'],map_location='cpu',weights_only=False);out={};detail={};max_projection_error=0.
    for a in ('BASE','PRES1'):
        rows=[];adds=[];detail[a]=[]
        for r in rec:
            fid=r['id'];h,w=r['hw'];tar=r['targets'][0];kp=np.array(tar['keypoints_normalized']);qgt=kp[:,:2]*[w,h];valid=kp[:,2]>0;valid[8]=False
            max_projection_error=max(max_projection_error,float(np.linalg.norm(gt[fid]['keypoints']-qgt,axis=1)[valid].max()));c=C.selected(pred[a][fid]);b=np.array(tar['box_xywh_normalized'])*[w,h,w,h];box=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2];matched=c is not None and V.E.C.H.E.O.iou(c['box_xyxy'],box)>=.5
            err=np.full(9,np.hypot(h,w)) if c is None else np.linalg.norm(np.array(c['keypoints_xy'])-qgt,axis=1);err[~np.isfinite(err)]=np.hypot(h,w)
            if not matched:err[:]=np.hypot(h,w)
            rows.extend(err[valid].tolist());g=F.extract(pred[a][fid],md[fid]['K'],md[fid]['dims'],r['hw']);name=g['selection']
            if g['valid']:name=C.P.HYP[int(M.selection(M.scores(ck,np.array(g['features'],np.float32)[None]),C.P.HYP)[0])]
            hyp=next((h for h in g['hypotheses'] if h['name']==name),None);add=addnorm(hyp,gt[fid]) if hyp else float('inf');adds.append(add);detail[a].append(dict(id=fid,errors=err[valid],ADDnorm=add))
        from scripts.research.pallet_verified_anchor_v1.evaluate import metrics
        out[a]=dict(twoD=metrics(rows),pose=dict(ADDsym_AUC=pose_auc(adds),median_ADDnorm=np.median(adds),available=int(np.isfinite(adds).sum()),frames=256))
    assert max_projection_error<.05
    C.freeze(C.RAW/'SOURCE_FRAME_METRICS.json',detail);C.freeze(C.DOC/'SOURCE_PRESERVATION.json',dict(groups=out,frames=256,exact_projection_max_px=max_projection_error,source_binding=C.bind(C.STRUCT/'SOURCE_PROBE_PLAN.json'),GT_input_after_prediction_lock=True))

if __name__=='__main__':main()
