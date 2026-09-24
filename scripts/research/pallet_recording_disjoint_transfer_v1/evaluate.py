"""Reference-only scoring, after both raw and pose freeze locks exist."""
from concurrent.futures import ProcessPoolExecutor
import math
import numpy as np
from . import common as C
from scripts.research.pallet_clean19_pose_sensitive_diag_v1.evaluate import pose_row
from scripts.research.pallet_verified_anchor_v1.evaluate import metrics as visible_metrics, point

def transitions(left, right):
    pairs=[]
    for a,b in zip(left,right):
        assert a['id']==b['id'] and a['canonical_valid']==b['canonical_valid']
        for j,valid in enumerate(a['canonical_valid']):
            if valid: pairs.append((a['canonical_errors'][j],b['canonical_errors'][j]))
    x=np.array(pairs,float)
    return dict(points=len(x),lost_correct10=int(((x[:,0]<=10)&(x[:,1]>10)).sum()),
        gained_correct10=int(((x[:,0]>10)&(x[:,1]<=10)).sum()),
        recovery20_to10=int(((x[:,0]>20)&(x[:,1]<=10)).sum()),
        damage5_to10=int(((x[:,0]<5)&(x[:,1]>10)).sum()),
        new_match_failures=sum(a['matched'] and not b['matched'] for a,b in zip(left,right)),
        match_recoveries=sum(not a['matched'] and b['matched'] for a,b in zip(left,right)),
        branch_changed=sum(a['branch']!=b['branch'] for a,b in zip(left,right)))

def summary(rows):
    out=C.E.P.M.summary(rows)
    out['correct']={str(t):sum(e<=t for r in rows for e in r['errors']) for t in (5,10,20)}
    return out

def main():
    assert not (C.DOC/'RESULTS.json').exists(), 'Preserve completed evaluation'
    raw_lock=C.read(C.DOC/'PREDICTIONS_LOCK.json');pose_lock=C.read(C.DOC/'POSE_PREDICTIONS_LOCK.json')
    C.verify(raw_lock['file']);C.verify(pose_lock['file']);C.verify(pose_lock['predictions_lock'])
    for k in ('selector','solver','metadata'): C.verify(pose_lock[k])
    start=C.now();assert raw_lock['created_at']<=pose_lock['created_at']<start
    C.save(C.RAW/'SCORING_START.json',dict(created_at=start,raw_lock=C.bind(C.DOC/'PREDICTIONS_LOCK.json'),pose_lock=C.bind(C.DOC/'POSE_PREDICTIONS_LOCK.json')))
    rr=C.records();ids=[r['id'] for r in rr];groups=C.groups(rr)
    preds=C.read(C.RAW/'PREDICTIONS.json');poses=C.read(C.RAW/'POSE_PREDICTIONS.json')
    # First reference read in this scoring workflow occurs only after locks above.
    truth=C.read(C.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');_,gt=C.E.D.Pose.metadata('REAL_DEV')
    fm={};fixed={};pm={}
    for arm in C.ARMS:
        fm[arm]={};fixed[arm]={}
        for fid in ids:
            p=preds[arm][fid];t=truth[fid];cand=C.E.P.C.selected(p)
            matched=cand is not None and C.E.C.H.E.O.iou(cand['box_xyxy'],t['box'])>=.5
            q=np.full((9,2),np.nan) if cand is None else cand['keypoints_xy']
            fm[arm][fid]=dict(id=fid,**C.E.P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,cand is not None))
            fixed[arm][fid]=dict(id=fid,**C.E.P.M.measure(q,t['gt'],t['valid'],[list(range(9))],t['hw'],matched,cand is not None))
        with ProcessPoolExecutor(max_workers=4) as pool:
            pm[arm]=dict(pool.map(pose_row,[(fid,poses[arm][fid],gt[fid]) for fid in ids],chunksize=8))
        print('SCORED',arm,len(pm[arm]),flush=True)
    results={};tt={}
    for name,ii in groups.items():
        results[name]={}
        for arm in C.ARMS:
            cur=C.E.D.aggregate([pm[arm][i]['current'] for i in ii]);cur['coverage']=cur['available']/len(ii)
            oracle=C.E.D.aggregate([pm[arm][i]['oracle'] for i in ii]);oracle['coverage']=oracle['available']/len(ii)
            results[name][arm]=dict(twoD=summary([fm[arm][i] for i in ii]),fixed_ID_supplement=summary([fixed[arm][i] for i in ii]),
                current=cur,oracle=oracle,selection_loss=oracle['ADDsym_AUC']-cur['ADDsym_AUC'])
        tt[name]=dict(canonical_GT_identity_aligned=transitions([fm['S0'][i] for i in ii],[fm['S1'][i] for i in ii]),
                      native_fixed_ID_no_symmetry=transitions([fixed['S0'][i] for i in ii],[fixed['S1'][i] for i in ii]))
    C.save(C.RAW/'FRAME_METRICS.json',fm);C.save(C.RAW/'FIXED_ID_METRICS.json',fixed);C.save(C.RAW/'POSE_METRICS.json',pm)
    C.save(C.DOC/'RESULTS.json',dict(groups=results,primary='S1-S0',frames=len(ids),independent_test=False,
        raw_and_pose_frozen_before_reference_scoring=True,oracle_label='POSTHOC GT ORACLE_WD — NONDEPLOYABLE',
        twoD_contract='Unchanged HELDOUT reference: whole-object best allowed symmetry; detection IoU>=.5; absent/mismatched native diagonal penalty; corners0..7.',
        transitions_contract='Canonical GT identity aligned after original symmetry branch; separate strict native fixed-ID supplement (no symmetry) also provided.',
        pose_contract='Unchanged production D9 selector (uses9points); corner0..7 pose solve; geometry-derived reference, not independent measured6D.',
        uncertainty='Single seed and 7 reused recording groups; no corner-independent statistical significance claim.'))
    C.save(C.DOC/'TRANSITIONS.json',tt)
    # Cross-check previously published supplementary arms without updating them.
    prior=C.read(C.E.DOC/'RESULTS.json')['groups']
    parity=[]
    for new,old in [('ALL','HELDOUT_ALL_PLASTIC'),('CLEAN','HELDOUT_CLEAN'),('MODERATE','HELDOUT_MODERATE'),('SEVERE','HELDOUT_SEVERE')]:
        for arm,oldarm in [('R0','R0'),('S1','OLD_S1')]:
            for field in ('twoD','current','oracle'):
                previous=prior[old][oldarm][field];current=results[new][arm][field]
                for k,v in previous.items():
                    if k in current: C.E.D.close(v,current[k])
            parity.append(dict(group=new,arm=arm,passed=True))
    C.save(C.DOC/'REFERENCE_PARITY.json',dict(previous_heldout_R0_S1=parity))
    anchors(preds,rr)
    historical(results)
    source={}
    sourceids=None
    for arm in ('S0','S1'):
        d=C.read(C.E.C.RAW/f'DIAGNOSTICS_PLASTIC_{arm}.json')
        assert d['checkpoint']==C.read(C.E.C.DOC/f'FIT_PLASTIC_{arm}.json')['checkpoint']
        if sourceids is None: sourceids=[r['id'] for r in d['source']]
        assert [r['id'] for r in d['source']]==sourceids and len(sourceids)==256
        source[arm]=C.E.C.summary(d['source'])
    C.save(C.DOC/'SOURCE_PRESERVATION.json',dict(source_frames=256,arms=source,existing_frozen_source_only=True,
        source_pose_not_recomputed=True,source_new_inference=0))
    print('RESULTS_COMPLETE',flush=True)

def anchors(preds,records):
    final=C.read(C.FINAL);qa=C.read(C.ANCHOR/'METADATA_QA_FINAL.json')
    assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2' and C.sha(C.FINAL)==qa['final_reference_sha256']
    teacher=C.read(C.E.V.RAW/'PREDICTIONS.json')['predictions']['TYPE_REPLAY_PIPELINE']
    models={**preds,'TEACHER':teacher};by={r['id']:r for r in records};rows=[]
    for fi,ci in final['review_queue']:
        frame=final['frames'][fi];fid=frame['frame_id'];corner=frame['corners'][ci]
        if fid not in by or corner['status']!='DIRECT_VISIBLE': continue
        assert 0<=ci<8 and corner['coordinate_source']=='manual_click' and corner['id']==ci
        assert frame['image_sha256']==by[fid]['image']['sha256']
        errors={};missing={}
        for arm,cache in models.items():
            q=point(cache.get(fid,{}),ci);missing[arm]=q is None
            errors[arm]=float(np.linalg.norm(q-np.array(corner['xy']))) if q is not None else math.hypot(*C.read(C.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')[fid]['hw'])
        rows.append(dict(frame_id=fid,corner_id=ci,severity=by[fid]['severity'],recording=by[fid]['recording_group'],errors=errors,missing=missing))
    grouped={'ALL':rows,**{k:[r for r in rows if r['severity']==s] for k,s in C.SEVS.items()},
             'HARD':[r for r in rows if r['severity']!='CLEAN']}
    result={g:{a:visible_metrics([r['errors'][a] for r in rs]) for a in models} for g,rs in grouped.items()}
    old=C.read(C.ANCHOR/'VERIFIED_RESULTS_FINAL.json')['groups'];oldt=C.read(C.ANCHOR/'TEACHER_SUPPLEMENT_FINAL.json')['groups']
    assert len(rows)==66 and len({r['frame_id'] for r in rows})==16
    for g,oldg in [('ALL','ALL'),*C.SEVS.items()]:
        C.E.D.close(result[g]['R0'],old[oldg]['R0']);C.E.D.close(result[g]['S1'],old[oldg]['OLD_S1']);C.E.D.close(result[g]['TEACHER'],oldt[oldg])
    C.save(C.RAW/'ANCHOR_POINT_METRICS.json',rows)
    C.save(C.DOC/'VERIFIED_ANCHOR_TRANSFER.json',dict(reference=final['reference_version'],reference_binding=C.bind(C.FINAL),
        groups=result,frames=len({r['frame_id'] for r in rows}),points=len(rows),fixed_identity=True,symmetry_remapping=False,
        coverage={a:dict(points=sum(not r['missing'][a] for r in rows),total=len(rows)) for a in models},
        annotation_written=False,not_population_performance=True,not_independent_6D_GT=True,
        correct10_delta={g:result[g]['S1']['PCK']['10']['correct']-result[g]['S0']['PCK']['10']['correct'] for g in result}))

def historical(results):
    old=C.read(C.E.C.DOC/'RESULTS.json');out={}
    for g,oldg in [('ALL','ALL300'),*C.SEVS.items()]:
        olddelta=old['contrasts'][oldg]['S1-S0']
        a,b=results[g]['S0'],results[g]['S1']
        new=dict(PCK10_delta_pp=100*(b['twoD']['PCK']['10']-a['twoD']['PCK']['10']),
                 ADDsym_delta=b['current']['ADDsym_AUC']-a['current']['ADDsym_AUC'],
                 oracle_delta=b['oracle']['ADDsym_AUC']-a['oracle']['ADDsym_AUC'])
        flags={k:bool(np.sign(olddelta[k])==np.sign(new[k])) for k in ('PCK10_delta_pp','ADDsym_delta')}
        direction='REPLICATED_ACROSS_RECORDINGS' if all(flags.values()) else 'SAME_SESSION_ONLY_OR_UNSTABLE' if not any(flags.values()) else 'MIXED'
        plastic_key='PLASTIC'+('' if g=='ALL' else '_'+oldg)
        out[g]=dict(historical_group=oldg,historical_frames=old['groups'][oldg]['S0']['sixD']['frames'],
            historical_delta=olddelta,historical_plastic_frames=old['groups'][plastic_key]['S0']['sixD']['frames'],
            historical_plastic_delta=old['contrasts'][plastic_key]['S1-S0'],heldout_frames=a['current']['frames'],
            recording_disjoint_delta=new,direction_replicated=flags,status=direction,
            note='Replicated direction can mean replicated deterioration; not an independent replication. Do not combine denominators.')
    C.save(C.DOC/'HISTORICAL_DIRECTION.json',dict(groups=out,historical_only=True,independent_test=False))

if __name__=='__main__':main()
