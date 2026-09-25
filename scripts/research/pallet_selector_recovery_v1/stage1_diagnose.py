"""Reference-dependent diagnosis, isolated from synthetic training."""
import numpy as np
from . import common as C
from . import features as F

def main():
    contract=C.read(C.DOC/'SELECTOR_FEATURE_CONTRACT.json')
    C.freeze(C.sdoc(1)/'REFERENCE_READ_LOCK.json',dict(created_at=C.now(),feature_contract=C.bind(C.DOC/'SELECTOR_FEATURE_CONTRACT.json')))
    assert contract['created_at']<C.read(C.sdoc(1)/'REFERENCE_READ_LOCK.json')['created_at']
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    rr=V.records();pred=C.read(C.PREV_RAW/'PREDICTIONS.json')['S1'];pm=C.read(C.PREV_RAW/'POSE_METRICS.json')['S1'];poses=C.read(C.PREV_RAW/'POSE_PREDICTIONS.json')['S1']
    meta={r['id']:r for r in C.read(V.E.V.RAW/'INFERENCE_METADATA.json')};truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');_,gt=V.E.D.Pose.metadata('REAL_DEV')
    rows=[];newpm={};fm={};role=C.read(C.PREV_DOC/'H10_ROLE_PREVALENCE.json')['strong_frames']
    for r in rr:
        fid=r['id'];m=meta[fid];p=pred[fid];t=truth[fid];out=F.extract(p,m['K'],m['xyz'],t['hw']);c=C.selected(p)
        matched=c is not None and V.E.C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
        fm[fid]=dict(id=fid,**V.E.P.M.measure(np.full((9,2),np.nan) if c is None else c['keypoints_xy'],t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
        current=V.E.D.Pose.infer(V.E.D.points(p),np.array(m['K']),np.array(m['xyz']),False)
        metric=V.E.D.metric(fid,current,gt[fid]);V.E.D.close(metric,pm[fid]['current']);newpm[fid]=metric
        assert out['selection']==current.get('selected_hypothesis')
        hh=[]
        for h in out['hypotheses']:
            old=next(x for x in pm[fid]['hypotheses'] if x['name']==h['name'])
            pose=F.production_pose(h,np.array(m['xyz']))
            if pose['available']:
                for k in ('R_cf','centroid','cf_extents'):assert np.allclose(pose[k],old['pose'][k],atol=1e-7,rtol=1e-7)
            hh.append(dict(name=h['name'],score=h['score'],components=h['score_components'],pose=old['pose'],metric=old['metric'],feature=F.vector(h,c,t['hw'])))
        selected=next((h for h in hh if h['name']==out['selection']),None);alt=next((h for h in hh if h['name']!=out['selection']),None)
        ok=[h for h in hh if h['metric']['available']];best=min(ok,key=lambda h:(h['metric']['ADDsym_normalized'],h['name'])) if ok else None
        margin=abs(hh[0]['score']-hh[1]['score']) if len(hh)==2 and all(h['score'] is not None for h in hh) else None
        better=bool(alt and alt['metric']['available'] and metric['available'] and alt['metric']['ADDsym_normalized']<metric['ADDsym_normalized'])
        rows.append(dict(id=fid,recording=r['recording_group'],severity=r['severity'],current=out['selection'],hypotheses=hh,
            category='POSE_UNAVAILABLE' if not metric['available'] else 'ALTERNATE_BETTER' if better else 'CURRENT_BEST',
            best=best['name'] if best else None,current_is_best=best is not None and best['name']==out['selection'],
            best_ADD_actual=best['metric']['ADDsym_normalized'] if best else None,
            oracle_current_ADD_gap=metric['ADDsym_normalized']-best['metric']['ADDsym_normalized'] if metric['available'] and best else None,
            score_margin=margin,axis_wrong=metric.get('axis_correct') is False,
            alternate_axis_correct=bool(alt and alt['metric'].get('axis_correct')),
            alternate_ADD_better=better,wrong_current_lower_reprojection=bool(selected and alt and metric.get('axis_correct') is False and selected['components']['reprojection_rmse_px']<alt['components']['reprojection_rmse_px']),
            both_invariant_zero=all(h['components'].get('invariant_violations')==0 for h in hh),
            keypoint_confidence=c['keypoints_conf'] if c else None,role_mismatch_tag=fid in role))
    groups=V.groups(rr);stats={};old=C.read(C.PREV_DOC/'RESULTS.json')['groups']
    for g,ii in groups.items():
        two=V.E.P.M.summary([fm[i] for i in ii]);cur=V.E.D.aggregate([newpm[i] for i in ii]);oracle=V.E.D.aggregate([pm[i]['oracle'] for i in ii])
        for k,a,b in [('PCK10',two['PCK']['10'],old[g]['S1']['twoD']['PCK']['10']),('AUC',cur['ADDsym_AUC'],old[g]['S1']['current']['ADDsym_AUC']),('oracle',oracle['ADDsym_AUC'],old[g]['S1']['oracle']['ADDsym_AUC'])]:assert abs(a-b)<=1e-7,(g,k,a,b)
        x=[r for r in rows if r['id'] in ii]
        stats[g]=dict(frames=len(ii),PCK10=two['PCK']['10'],CURRENT=cur,ORACLE=oracle,selection_loss=oracle['ADDsym_AUC']-cur['ADDsym_AUC'],
            axis_wrong=sum(r['axis_wrong'] for r in x),alternate_axis_correct=sum(r['alternate_axis_correct'] for r in x),alternate_ADD_better=sum(r['alternate_ADD_better'] for r in x),
            current_wrong_alternate_better=sum(r['axis_wrong'] and r['alternate_ADD_better'] for r in x),wrong_lower_reprojection=sum(r['wrong_current_lower_reprojection'] for r in x),both_invariant_zero=sum(r['both_invariant_zero'] for r in x))
    assert stats['MODERATE']['frames']==21
    C.freeze(C.sraw(1)/'FRAME_DIAGNOSTICS.json',rows)
    C.freeze(C.sdoc(1)/'MODERATE_SELECTOR_DIAGNOSTIC.json',dict(status='BASELINE_REPRODUCED',groups=stats,moderate=[r for r in rows if r['severity']=='MODERATE_OCCLUSION'],feature_contract=C.bind(C.DOC/'SELECTOR_FEATURE_CONTRACT.json'),no_GT_feature_selection=True))
    dist={}
    for cat in ('CURRENT_BEST','ALTERNATE_BETTER'):
        rr=[r for r in rows if r['severity']=='MODERATE_OCCLUSION' and r['category']==cat];vals={k:[] for k in ('margin','confidence','upright','spread')}
        for r in rr:
            h=next(h for h in r['hypotheses'] if h['name']==r['current']);vals['margin'].append(r['score_margin']);vals['confidence'].append(np.mean(r['keypoint_confidence']));vals['upright'].append(h['components']['upright_alignment']);vals['spread'].append(h['components']['spread_ratio'])
        dist[cat]={k:dict(n=len(v),values=v,quantiles=np.quantile(v,[0,.1,.5,.9,1]) if v else []) for k,v in vals.items()}
    C.freeze(C.sdoc(1)/'FEATURE_DISTRIBUTIONS.json',dist)
    print('STAGE1',stats['MODERATE'],flush=True)

if __name__=='__main__':main()
