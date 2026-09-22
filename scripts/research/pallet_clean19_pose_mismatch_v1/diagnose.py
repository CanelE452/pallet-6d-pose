"""Reuse the production selector and solver; references enter metrics only."""
from collections import Counter
from pathlib import Path
import json
import subprocess
import numpy as np
import cv2
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as C
from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import EDGES
from challenge.evaluation_v2 import pnp_selector as Selector

ROOT=C.ROOT
NAME='pallet_clean19_pose_mismatch_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
Pose=C.H.V.Pose
ARMS=['R0','TEACHER','OLD_STUDENT','S0','S1','S2']
STUDENTS=['S0','S1','S2']
SEVS=list(C.H.P.SEVERITIES)
read=C.read

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def save(path,value):
    C.save(path,clean(value))

def load():
    records=read(C.H.P.DOC/'SPLIT.json')['evaluation']
    old=read(C.H.V.RAW/'PREDICTIONS.json')['predictions']
    preds={'R0':old['R0_DIRECT'],'TEACHER':old['TYPE_REPLAY_PIPELINE'],'OLD_STUDENT':{}}
    paths=[C.H.P.DOC/'SPLIT.json',C.H.V.RAW/'PREDICTIONS.json',C.H.V.RAW/'INFERENCE_METADATA.json',C.H.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json']
    for mat in C.MATERIALS:
        p=C.H.RAW/f'EVAL_PREDICTIONS_{mat}.json';paths.append(p)
        preds['OLD_STUDENT'].update({r['id']:r['prediction'] for r in read(p)['records']})
    for arm in STUDENTS:
        preds[arm]={}
        for mat in C.MATERIALS:
            p=C.RAW/f'EVAL_{mat}_{arm}.json';paths.append(p);preds[arm].update(read(p)['predictions'])
    poses=read(C.RAW/'POSE_PREDICTIONS.json')
    historic=read(C.H.RAW/'pose/PREDICTIONS.json')
    poses.update(R0=historic['R0'],TEACHER=historic['CORRECTED_TEACHER'],OLD_STUDENT=historic['STUDENT'])
    meta={r['id']:r for r in read(C.H.V.RAW/'INFERENCE_METADATA.json')}
    truth=read(C.H.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    _,gt=Pose.metadata('REAL_DEV')
    frame=read(C.RAW/'FRAME_METRICS.json');pm=read(C.RAW/'POSE_METRICS.json');results=read(C.DOC/'RESULTS.json')
    for root in (C.DOC,C.RAW):paths.extend(p for p in root.glob('*.json'))
    paths.extend([C.DOC/'REPORT_KO.md',C.DOC/'AUGMENTATION_PLAN.jsonl',C.H.RAW/'pose/PREDICTIONS.json',Path(Pose.__file__),Path(Selector.__file__),Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json',ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py'])
    for r in records:
        C.verify(r['annotation']);C.verify(r['image']);paths.extend([ROOT/r['annotation']['path'],ROOT/r['image']['path']])
    for p in read(C.DOC/'EVALUATION_PREDICTIONS_LOCK.json')['files']:C.verify(p)
    C.verify(read(C.DOC/'POSE_PREDICTIONS_LOCK.json'))
    assert len(records)==300 and len({r['id'] for r in records})==300
    assert all(set(preds[a])=={r['id'] for r in records} for a in ARMS)
    return dict(records=records,preds=preds,poses=poses,meta=meta,truth=truth,gt=gt,frame=frame,pm=pm,results=results,paths=sorted(set(paths)))

def points(pred):
    c=C.H.P.C.selected(pred)
    if c is None:return None
    q=np.array(c['keypoints_xy'],float);q[(q==-1).all(1)]=np.nan
    return q

def select(q,K,xyz):
    # This function accepts NO reference or label argument.
    if q is None:return dict(status='NO_DETECTION',selected_hypothesis=None,hypotheses=[],ambiguity='NO_DETECTION')
    try:return Selector.select_pnp_hypotheses(q,K,dict(x=max(xyz[0],xyz[2]),y=xyz[1],z=min(xyz[0],xyz[2])),None).to_dict()
    except (ValueError,cv2.error) as e:return dict(status='INPUT_REJECTED',selected_hypothesis=None,hypotheses=[],ambiguity=str(e))

def hyp_pose(h,q,K,xyz):
    if not h['success']:return dict(available=False)
    d=h['camera_facing_dimensions_m'];dims=np.array([d['width'],d['height'],d['depth']])
    valid=np.isfinite(q[:8]).all(1)
    if valid.sum()<6:return dict(available=False)
    try:solved=Pose.solve(Pose.cuboid(*dims),q[:8],K,valid)
    except cv2.error:return dict(available=False)
    if solved is None:return dict(available=False)
    R,t,res=solved
    if t[2]<=0 or not np.isfinite(R).all() or not np.isfinite(t).all():return dict(available=False)
    Q=np.eye(3) if abs(dims[0]-xyz[0])<1e-6 else Pose.rotations(4)[1]
    return dict(available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t.tolist(),cf_extents=dims.tolist(),selected_hypothesis=h['name'],reprojection_px=float(res))

def metric(fid,p,g):
    m=Pose.metric((fid,p,g))
    if m['available']:m['axis_correct']=bool(abs(p['cf_extents'][0]-g['body_xyz'][0])<1e-6)
    return m

def close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:close(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(float,np.floating)):assert np.isclose(a,b,atol=1e-7,rtol=1e-7,equal_nan=True),(a,b)
    else:assert a==b,(a,b)

def aggregate(rows):
    ok=[r for r in rows if r['available']]
    out=dict(frames=len(rows),available=len(ok),axis_accuracy=float(np.mean([r['axis_correct'] for r in ok])) if ok else None,
        axis_correct_count=sum(r['axis_correct'] for r in ok),ADDsym_AUC=Pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows],1.))
    for k in ('rotation_deg','yaw_deg','translation_cm','IoU3D','ADDsym_normalized'):
        out[k]=dict(median=float(np.median([r[k] for r in ok])) if ok else None,P90=float(np.quantile([r[k] for r in ok],.9)) if ok else None)
    return out

def provenance(record):
    a=read(ROOT/record['annotation']['path']);obj=a['objects'][0]
    entries=obj.get('keypoint_annotations',[])
    src=[entries[j].get('source','unknown') if j<len(entries) else 'unknown' for j in range(8)]
    manual=[s in ('manual','manual_click') for s in src]
    return dict(corner_sources=src,manual_only=all(manual),manual_count=sum(manual),unknown_count=src.count('unknown'),
        projected_count=sum('pnp' in s.lower() or 'project' in s.lower() for s in src),
        subset='manual_only' if all(manual) else 'has_unknown' if 'unknown' in src else 'mixed_documented',
        pose_reference='geometry-resolved annotation + known dimensions; NOT independent physical pose',axis_reference='W/D extents parity; NOT full rotation correctness')

def parity_and_frames(data):
    out={a:{} for a in ARMS};checks=Counter()
    for arm in ARMS:
        for r in data['records']:
            fid=r['id'];q=points(data['preds'][arm][fid]);m=data['meta'][fid];K=np.array(m['K']);xyz=np.array(m['xyz'])
            current=Pose.infer(q,K,xyz,False);close(current,data['poses'][arm][fid]);checks['pose']+=1
            mm=metric(fid,current,data['gt'][fid]);close(mm,data['pm'][arm][fid]);checks['pose_metrics']+=1
            # Recompute the same 2D metric and detection match gate.
            t=data['truth'][fid];c=C.H.P.C.selected(data['preds'][arm][fid]);matched=c is not None and C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
            fm=dict(id=fid,**C.H.P.M.measure(np.full((9,2),np.nan) if c is None else c['keypoints_xy'],t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
            close(fm,data['frame'][arm][fid]);checks['2d']+=1
            sr=select(q,K,xyz)
            assert sr['selected_hypothesis']==current.get('selected_hypothesis')
            hypotheses=[]
            for h in sr['hypotheses']:
                hp=hyp_pose(h,q,K,xyz);hm=metric(fid,hp,data['gt'][fid])
                if h['name']==sr['selected_hypothesis']:close(hp,current);close(hm,mm)
                hypotheses.append(dict(name=h['name'],success=h['success'],selector_score=h['score'],**h['score_components'],pose=hp,metric=hm))
            good=[h for h in hypotheses if h['metric'].get('axis_correct')]
            selected=next((h for h in hypotheses if h['name']==sr['selected_hypothesis']),None)
            alt=next((h for h in hypotheses if h['name']!=sr['selected_hypothesis']),None)
            scores=[h['selector_score'] for h in hypotheses if h['selector_score'] is not None]
            delta=abs(scores[0]-scores[1]) if len(scores)==2 else None
            category='POSE_UNAVAILABLE' if not mm['available'] else 'SELECTED_AXIS_CORRECT_DISTRIBUTION_REQUIRED' if mm['axis_correct'] else 'ALTERNATE_GOOD_SELECTOR_WRONG' if good else 'BOTH_HYPOTHESES_BAD_AXIS'
            out[arm][fid]=dict(frame_id=fid,severity=r['severity'],material=r['object_type'],session=r['session'],
                matched=fm['matched'],detected=fm['detected'],valid_corner_count=fm['corners'],
                PCK_counts={str(v):sum(e<=v for e in fm['errors']) for v in (5,10,20)},
                frame_mean_2d_error=fm['frame_mean_px'],frame_median_2d_error=float(np.median(fm['errors'])),canonical_errors=fm['canonical_errors'],
                current_pose=dict(**current,metric=mm),selector={k:v for k,v in sr.items() if k!='hypotheses' and k!='canonical_candidates'},hypotheses=hypotheses,
                category=category,score_delta_abs=delta,score_delta_normalized=delta/max(*map(abs,scores),1e-6) if delta is not None else None,
                alternate_ADD_better=bool(alt and alt['metric']['available'] and mm['available'] and alt['metric']['ADDsym_normalized']<mm['ADDsym_normalized']),
                wrong_lower_reprojection=bool(mm.get('axis_correct') is False and selected and alt and selected.get('reprojection_rmse_px') is not None and alt.get('reprojection_rmse_px') is not None and selected['reprojection_rmse_px']<alt['reprojection_rmse_px']),
                provenance=provenance(r))
        print('PARITY',arm,len(out[arm]),flush=True)
    # Reproduce every historical group and common-matched supplement.
    for group,arms in data['results']['groups'].items():
        ii=[r['id'] for r in data['records'] if group=='ALL300' or r['severity']==group or r['object_type'].upper()==group or r['object_type'].upper()+'_'+r['severity']==group or 'SESSION_'+r['session']==group]
        assert len(ii)==next(iter(arms.values()))['sixD']['frames']
        for arm in ARMS:
            agg=aggregate([out[arm][i]['current_pose']['metric'] for i in ii]);old=arms[arm]['sixD']
            for key in ('axis_accuracy','ADDsym_AUC','rotation_deg','yaw_deg','translation_cm','IoU3D'):close(agg[key],old[key])
            common=[data['frame'][arm][i] for i in ii if all(data['frame'][a][i]['matched'] for a in ARMS)]
            close(C.H.P.M.summary(common),data['results']['common_matched_supplement'][group][arm])
            checks['group_metrics']+=1
    return out,dict(checks)

def transitions(frames,data):
    result={}
    for sev in SEVS:
        result[sev]={}
        for arm in ('S1','S2'):
            rows=[]
            for r in data['records']:
                if r['severity']!=sev:continue
                fid=r['id'];a=frames['S0'][fid];b=frames[arm][fid];am=a['current_pose']['metric'];bm=b['current_pose']['metric']
                p=b['PCK_counts']['10']-a['PCK_counts']['10'];axis=None;add=iou=None
                if am['available'] and bm['available']:
                    axis=int(bm['axis_correct'])-int(am['axis_correct']);add=bm['ADDsym_normalized']-am['ADDsym_normalized'];iou=bm['IoU3D']-am['IoU3D']
                corners=[dict(corner=j,before=x,after=y,delta=y-x) for j,(x,y) in enumerate(zip(a['canonical_errors'],b['canonical_errors'])) if x is not None and y is not None]
                joint='PCK_'+('UP' if p>0 else 'DOWN' if p<0 else 'SAME')+'+AXIS_'+('UNAVAILABLE' if axis is None else 'UP' if axis>0 else 'DOWN' if axis<0 else 'SAME')
                rows.append(dict(frame_id=fid,model=arm,PCK10_count_delta=p,mean_error_delta=b['frame_mean_2d_error']-a['frame_mean_2d_error'],
                    axis_transition=(str(am.get('axis_correct'))+'->'+str(bm.get('axis_correct'))),axis_delta=axis,ADD_delta=add,IoU_delta=iou,joint=joint,
                    PCK_UP_AND_POSE_DOWN=bool(p>0 and add is not None and add>1e-12),PCK_UP_IoU_DOWN=bool(p>0 and iou is not None and iou< -1e-12),
                    before_selector=a['selector'],after_selector=b['selector'],before_scores=[h['selector_score'] for h in a['hypotheses']],after_scores=[h['selector_score'] for h in b['hypotheses']],corners=corners))
            result[sev][arm]=dict(rows=rows,joint_counts=dict(Counter(r['joint'] for r in rows)),axis_counts=dict(Counter(r['axis_transition'] for r in rows)),
                PCK_UP_AND_POSE_DOWN=sum(r['PCK_UP_AND_POSE_DOWN'] for r in rows),
                ADD_improves=sum(r['ADD_delta'] is not None and r['ADD_delta']< -1e-12 for r in rows),ADD_worsens=sum(r['ADD_delta'] is not None and r['ADD_delta']>1e-12 for r in rows))
    return result

def oracles(frames):
    out={}
    for sev in ['ALL300',*SEVS]:
        out[sev]={}
        for arm in STUDENTS:
            rr=[r for r in frames[arm].values() if sev=='ALL300' or r['severity']==sev];current=[];wd=[];axis=[]
            for r in rr:
                current.append(r['current_pose']['metric']);hh=[h for h in r['hypotheses'] if h['metric']['available']]
                key=lambda h:(h['metric']['ADDsym_normalized'],h['name'])
                best=min(hh,key=key)['metric'] if hh else dict(available=False)
                correct=[h for h in hh if h['metric']['axis_correct']]
                wd.append(best);axis.append(min(correct,key=key)['metric'] if correct else best)
            out[sev][arm]=dict(CURRENT=aggregate(current),ORACLE_WD=aggregate(wd),ORACLE_AXIS=aggregate(axis),
                category_counts=dict(Counter(r['category'] for r in rr)),alternate_ADD_better=sum(r['category']=='ALTERNATE_GOOD_SELECTOR_WRONG' and r['alternate_ADD_better'] for r in rr),
                axis_wrong=sum(m.get('axis_correct') is False for m in current))
    return dict(label='POSTHOC GT ORACLE — NONDEPLOYABLE',tie_rule='ADD minimum, then hypothesis name; axis oracle restricts to correct extents if any, otherwise same ADD rule',groups=out)

def perturbations(frames,data,tt):
    loo=[];replace=[];moderate=[r for r in data['records'] if r['severity']=='MODERATE_OCCLUSION']
    rng=np.random.default_rng(20260922);controls=sorted(r['id'] for r in rng.choice(moderate,6,replace=False))
    selection={}
    for arm in STUDENTS:
        mismatch={r['frame_id'] for r in tt['MODERATE_OCCLUSION'].get(arm,{}).get('rows',[]) if r['PCK_UP_AND_POSE_DOWN']}
        chosen={r['id'] for r in moderate if frames[arm][r['id']]['current_pose']['metric'].get('axis_correct') is False}|mismatch|set(controls)
        selection[arm]=sorted(chosen)
        for r in moderate:
            fid=r['id'];q=points(data['preds'][arm][fid]);m=data['meta'][fid];K=np.array(m['K']);xyz=np.array(m['xyz']);base=frames[arm][fid]['current_pose']['metric']
            for j in range(8):
                qq=None if q is None else q.copy()
                if qq is not None:qq[j]=np.nan
                n=0 if qq is None else int(np.isfinite(qq[:8]).all(1).sum())
                if n<6:loo.append(dict(frame_id=fid,model=arm,corner=j,remaining=n,status='SKIP_MIN6'));continue
                sr=select(qq,K,xyz);p=Pose.infer(qq,K,xyz,False)
                assert sr['status']=='INPUT_REJECTED' and not p['available']
                loo.append(dict(frame_id=fid,model=arm,corner=j,remaining=n,status='BLOCKED_CURRENT_SELECTOR_REQUIRES_ALL9_FINITE',reason=sr['ambiguity']))
            if fid not in chosen or q is None:continue
            target=np.array(data['truth'][fid]['gt']);valid=data['truth'][fid]['valid']
            for j in range(8):
                if not valid[j] or not np.isfinite(target[j]).all():continue
                qq=q.copy();qq[j]=target[j] # native camera-facing reference, no GT branch permutation
                sr=select(qq,K,xyz);p=Pose.infer(qq,K,xyz,False);mm=metric(fid,p,data['gt'][fid])
                replace.append(dict(frame_id=fid,model=arm,corner=j,label='GT_REPLACEMENT_ORACLE_NONDEPLOYABLE',reference_source=frames[arm][fid]['provenance']['corner_sources'][j],
                    base=base,after=mm,selected_hypothesis=sr['selected_hypothesis'],hypothesis_changed=sr['selected_hypothesis']!=frames[arm][fid]['selector']['selected_hypothesis'],
                    axis_recovered=base.get('axis_correct') is False and mm.get('axis_correct') is True,
                    axis_harmed=base.get('axis_correct') is True and mm.get('axis_correct') is False,
                    ADD_delta=mm['ADDsym_normalized']-base['ADDsym_normalized'] if base['available'] and mm['available'] else None,
                    rotation_delta=mm['rotation_deg']-base['rotation_deg'] if base['available'] and mm['available'] else None))
        print('PERTURBATIONS',arm,'LOO attempts',len(loo),'GT substitutions',len(replace),flush=True)
    stats=[]
    for arm in STUDENTS:
        for j in range(8):
            rr=[r for r in replace if r['model']==arm and r['corner']==j]
            stats.append(dict(model=arm,corner=j,trials=len(rr),axis_recovered=sum(r['axis_recovered'] for r in rr),axis_harmed=sum(r['axis_harmed'] for r in rr),
                ADD_improves=sum(r['ADD_delta'] is not None and r['ADD_delta']< -1e-12 for r in rr),ADD_harms=sum(r['ADD_delta'] is not None and r['ADD_delta']>1e-12 for r in rr)))
    return dict(status='BLOCKED_BY_UNCHANGED_SELECTOR_CONTRACT',no_algorithm_change=True,rows=loo,influence_counts=None),dict(label='GT_REPLACEMENT_ORACLE_NONDEPLOYABLE',random_controls=controls,selected_ids=selection,rows=replace,corner_summary=stats)

def pair_geometry(frames,data):
    out=[]
    xyz=Pose.cuboid(1.,.1,1.3)
    for arm in STUDENTS:
        for r in data['records']:
            fid=r['id'];q=points(data['preds'][arm][fid]);t=data['truth'][fid];g=np.array(t['gt']);valid=t['valid']
            if q is None:continue
            for a,b in EDGES:
                if not(valid[a] and valid[b]) or not np.isfinite(q[[a,b]]).all():continue
                v=q[b]-q[a];u=g[b]-g[a];lv=np.linalg.norm(v);lu=np.linalg.norm(u)
                angle=float(np.degrees(np.arccos(np.clip(np.dot(v,u)/(lv*lu),-1,1)))) if lv*lu>1e-9 else None
                family=['LR','TB_vertical','FR'][int(np.flatnonzero(xyz[a]!=xyz[b])[0])]
                out.append(dict(frame_id=fid,model=arm,severity=r['severity'],edge=f'{a}-{b}',family=family,
                    face='front' if a<4 and b<4 else 'rear' if a>=4 and b>=4 else 'front_rear',predicted_vector=v.tolist(),reference_vector=u.tolist(),
                    length_error=float(lv-lu),absolute_length_error=float(abs(lv-lu)),direction_error_deg=angle,axis_correct=frames[arm][fid]['current_pose']['metric'].get('axis_correct')))
    summary=[]
    for sev in SEVS:
        for arm in STUDENTS:
            for family in ['LR','TB_vertical','FR']:
                for axis in [None,True,False]:
                    rr=[r for r in out if r['severity']==sev and r['model']==arm and r['family']==family and (axis is None or r['axis_correct']==axis)]
                    aa=[r['direction_error_deg'] for r in rr if r['direction_error_deg'] is not None]
                    summary.append(dict(severity=sev,model=arm,family=family,axis_subset=axis,n=len(rr),angle_median=float(np.median(aa)) if aa else None,length_abs_median=float(np.median([r['absolute_length_error'] for r in rr])) if rr else None))
    return dict(rows=out,summary=summary,identity='native camera-facing channels, NOT symmetry-min aligned; directional reversals are retained',causal_claim=False)

def distributions(frames):
    output={}
    for sev in SEVS:
        output[sev]={}
        for arm in STUDENTS:
            result={}
            for axis in [True,False]:
                rr=[r for r in frames[arm].values() if r['severity']==sev and r['current_pose']['metric'].get('axis_correct')==axis]
                fields=['score_delta_abs','score_delta_normalized','reprojection_rmse_px','cheirality_fraction','invariant_violations','lr_violations','tb_violations','front_rear_violations','upright_alignment','spread_ratio']
                ds={}
                for k in fields:
                    values=[]
                    for r in rr:
                        selected=next((h for h in r['hypotheses'] if h['name']==r['selector']['selected_hypothesis']),{})
                        v=r.get(k) if k.startswith('score_delta') else selected.get(k)
                        if v is not None:values.append(v)
                    ds[k]=dict(n=len(values),q10=float(np.quantile(values,.1)) if values else None,median=float(np.median(values)) if values else None,q90=float(np.quantile(values,.9)) if values else None)
                result[str(axis)]=dict(n=len(rr),components=ds,wrong_lower_reprojection=sum(r['wrong_lower_reprojection'] for r in rr),both_zero_invariants=sum(len(r['hypotheses'])==2 and all(h.get('invariant_violations')==0 for h in r['hypotheses']) for r in rr),pose_distribution=aggregate([r['current_pose']['metric'] for r in rr]))
            output[sev][arm]=result
    return output
