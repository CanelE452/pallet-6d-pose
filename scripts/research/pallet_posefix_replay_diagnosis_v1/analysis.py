"""GT-side scoring only. Never imported by the neural inference stage."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy, math, time
import cv2
import numpy as np
from scripts.research.pallet_posefix_replay_diagnosis_v1 import run as R

DOC,RAW,ROOT=R.DOC,R.RAW,R.ROOT
read,write,freeze,bound=R.read,R.write,R.freeze,R.bound
BINS=['<=5','(5,10]','(10,20]','>20']

def stats(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                P90=float(np.quantile(a,.9)) if len(a) else None)

def fraction(a,mask):
    return float(np.asarray(a)[mask].mean()) if np.any(mask) else None

def cosines(delta):
    # frame, pass3, native corner8, xy
    norm=np.linalg.norm(delta,axis=-1)
    valid=(norm[:,:-1]>.001)&(norm[:,1:]>.001)
    cosine=np.full(valid.shape,np.nan)
    np.divide((delta[:,:-1]*delta[:,1:]).sum(-1),norm[:,:-1]*norm[:,1:],out=cosine,where=valid)
    return np.clip(cosine,-1,1),norm

def types(errors,norm,cos,hit):
    de=np.diff(errors,axis=1)
    return dict(monotonic_improve=(de < -1e-9).all(1),
        one_pass_best_then_regress=(errors[:,1]<errors[:,0]-1e-9)&((errors[:,2]>errors[:,1]+1e-9)|(errors[:,3]>errors[:,1]+1e-9)),
        oscillation=(((de[:,:-1]*de[:,1:])< -1e-18)&(cos<0)).any(1),
        no_response=(norm<=.001).all(1),repeated_cap=hit.sum(1)>=2)

def pose_job(task):
    fid,points,K,xyz,source,truth=task
    p=R.pose.infer(points,np.asarray(K),np.asarray(xyz),source)
    return dict(prediction=p,metric=R.pose.metric((fid,p,truth)))

def pose_summary(rows):
    available=[r for r in rows if r['available']]
    result=dict(frames=len(rows),available=len(available),coverage=len(available)/len(rows),
        ADDsym_AUC_full=R.pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows],1.0))
    for k in ['rotation_deg','yaw_deg','translation_cm','IoU3D']:
        result[k]=stats([r[k] for r in available])
    return result

def scalar_mean(values):
    if isinstance(values[0],dict): return {k:scalar_mean([v[k] for v in values]) for k in values[0]}
    if isinstance(values[0],list): return values[0] # non-evaluable IDs; not a performance quantity
    if values[0] is None: return None
    if isinstance(values[0],(int,float)): return float(np.mean(values))
    return values[0]

def verify_metric_parity(rows):
    expected={r['id']:r for r in read(ROOT/'_docs/experiments/pallet_final_paper_tables_v1/RESCORED_2D.json')['rows']['R0']}
    diffs=[]
    for r in rows:
        before=expected[r['id']]
        for k in ('errors','observed_errors','canonical_errors','E_sym','E_fixed','branch','matched','detected'):
            np.testing.assert_allclose(np.array(r[k],float),np.array(before[k],float),rtol=0,atol=1e-9,equal_nan=True)
        diffs.append(abs(r['E_sym']-before['E_sym']))
    return dict(frames=len(rows),maximum_Esym_difference=max(diffs),contract='Current 8corner whole-object proper group; no legacy9point mixing')

def preservation_checks(inputs,points):
    from inference import preservation
    n=0
    for i,r in enumerate(inputs):
        p0=np.array(r['points'],float)-r['pad']
        np.testing.assert_array_equal(points[i,:,0],np.stack([p0,p0]))
        for c in range(2):
            for p in range(4):
                np.testing.assert_array_equal(points[i,c,p,8],p0[8])
                if r['prediction'] is not None:
                    pred=r['prediction']; new=copy.deepcopy(pred['candidates']); idx=pred['selected_index']
                    if idx is not None: new[idx]['keypoints_xy']=points[i,c,p].tolist()
                    preservation(pred['candidates'],new,idx); n+=1
    return n

def score_split(seed,split,inputs,targets,pool):
    path=RAW/f'scores/seed{seed}_{split}.json'
    if path.exists(): return read(path)
    rows=[r for r in inputs if r['split']==split]
    z=np.load(RAW/f'predictions/seed{seed}_{split}.npz'); points=z['points']; raw=z['raw']; capped=z['capped']
    assert list(z['ids'])==[r['id'] for r in rows]
    checks=preservation_checks(rows,points)
    gt=np.array([targets[r['id']]['gt'] for r in rows],float)
    gt_valid=np.array([targets[r['id']]['valid'] for r in rows],bool)
    pvalid=np.array([r['valid'] for r in rows],bool)
    matched=np.array([targets[r['id']]['matched'] for r in rows],bool)
    dimensions=np.array([targets[r['id']]['dimensions_WDH'] for r in rows])
    cap=np.array([math.hypot(*r['raw_hw'])*.01 for r in rows])
    out=dict(seed=seed,split=split,frames=len(rows),preservation_checks=checks,chains={})
    saved={}; cached_pose0=None; metric_rows={}
    for chain,mode in enumerate(R.CHAINS):
        allmetrics=[]; poses=[]; matrices=[]
        for step in range(4):
            ms=[]; tasks=[]
            for i,r in enumerate(rows):
                t=targets[r['id']]; pts=points[i,chain,step]
                m=R.measure(pts,t['gt'],t['valid'],t['permutations'],r['raw_hw'],t['matched'],r['detected'])
                m.update(id=r['id'],object=t['object']); ms.append(m)
                tasks.append((r['id'],pts if r['detected'] else None,t['K'],t['xyz'],t['source'],t['truth']))
            if step==0 and cached_pose0 is not None: pp=cached_pose0
            else:
                pp=list(pool.map(pose_job,tasks,chunksize=16))
                if step==0: cached_pose0=pp
            allmetrics.append(ms); poses.append(pose_summary([r['metric'] for r in pp])); matrices.append(pp)
        metric_rows[mode]=allmetrics
        branch=np.array([[m.get('branch',0) for m in ms] for ms in allmetrics]).T
        canonical=np.array([[m.get('canonical_errors',[None]*8) for m in ms] for ms in allmetrics],float).transpose(1,0,2)
        # Fixed PASS0 correspondence only for causal movement vectors / error strata.
        perms=[targets[r['id']]['permutations'][int(branch[i,0])] for i,r in enumerate(rows)]
        locked_gt=np.array([gt[i,pr] for i,pr in enumerate(perms)])
        locked_valid=np.array([gt_valid[i,pr] for i,pr in enumerate(perms)])
        support=locked_valid[:,:8]&pvalid[:,:8]&matched[:,None]&np.isfinite(points[:,chain,0,:8]).all(-1)
        errors=np.linalg.norm(points[:,chain,:,:8]-locked_gt[:,None,:8],axis=-1)
        raw_errors=np.linalg.norm(raw[:,chain,:,:8]-locked_gt[:,None,:8],axis=-1)
        capped_errors=np.linalg.norm(capped[:,chain,:,:8]-locked_gt[:,None,:8],axis=-1)
        delta=np.diff(points[:,chain,:,:8],axis=1); cos,norm=cosines(delta)
        raw_norm=np.linalg.norm(raw[:,chain,:,:8]-points[:,chain,:3,:8],axis=-1)
        capped_norm=np.linalg.norm(capped[:,chain,:,:8]-points[:,chain,:3,:8],axis=-1)
        hit=raw_norm>cap[:,None,None]+1e-9
        cats=types(errors,norm,cos,hit)
        frame_error=np.array([[m.get('frame_mean_px',np.nan) for m in ms] for ms in allmetrics]).T
        frame_types=dict(monotonic_improve=(np.diff(frame_error,axis=1)< -1e-9).all(1),
            one_pass_best_then_regress=(frame_error[:,1]<frame_error[:,0]-1e-9)&((frame_error[:,2]>frame_error[:,1]+1e-9)|(frame_error[:,3]>frame_error[:,1]+1e-9)),
            oscillation=(((np.diff(frame_error,axis=1)[:,:-1]*np.diff(frame_error,axis=1)[:,1:])< -1e-18)&((cos<0)&support[:,None]).any(-1)).any(1),
            no_response=((norm<=.001)|~support[:,None]).all((1,2)),
            repeated_cap=(cats['repeated_cap']&support).any(1))
        frames_support=support.any(-1)
        movements={}; strata={}; symmetry={}
        for step in range(3):
            eligible=frames_support
            movements[str(step+1)]=dict(raw=stats(raw_norm[:,step][support]),capped=stats(capped_norm[:,step][support]),
                actual=stats(norm[:,step][support]),cap_hit_corner_fraction=fraction(hit[:,step],support),
                cap_hit_frame_fraction=fraction((hit[:,step]&support).any(-1),eligible),eligible_frames=int(eligible.sum()),
                eligible_corners=int(support.sum()),raw_improve_rate=fraction(raw_errors[:,step]<errors[:,step]-1e-9,support),
                cap_improve_rate=fraction(capped_errors[:,step]<errors[:,step]-1e-9,support),
                raw_worse_rate=fraction(raw_errors[:,step]>errors[:,step]+1e-9,support),
                same_input_cap_blocked_recovery=int((support&(errors[:,step]>20)&(raw_errors[:,step]<=10)&(capped_errors[:,step]>10)).sum()),
                same_input_cap_prevented_damage=int((support&(errors[:,step]<5)&(raw_errors[:,step]>10)&(capped_errors[:,step]<=10)).sum()))
        e0=errors[:,0]
        for label,mask in zip(BINS,[e0<=5,(e0>5)&(e0<=10),(e0>10)&(e0<=20),e0>20]):
            mask=mask&support; bystep={}
            for step in range(3):
                bystep[str(step+1)]=dict(delta_from_R0=stats((errors[:,step+1]-e0)[mask]),
                    delta_from_input=stats((errors[:,step+1]-errors[:,step])[mask]),
                    improvement_fraction=fraction(errors[:,step+1]<e0-1e-9,mask),
                    regression_fraction=fraction(errors[:,step+1]>e0+1e-9,mask),
                    cap_hit_fraction=fraction(hit[:,step],mask),
                    raw_better_than_capped_fraction=fraction(raw_errors[:,step]<capped_errors[:,step]-1e-9,mask),
                    raw_delta_from_input=stats((raw_errors[:,step]-errors[:,step])[mask]),
                    same_input_cap_lost_gain_px=stats((capped_errors[:,step]-raw_errors[:,step])[mask]),
                    raw_recovered_to10=int((mask&(e0>20)&(raw_errors[:,step]<=10)).sum()),
                    capped_recovered_to10=int((mask&(e0>20)&(capped_errors[:,step]<=10)).sum()))
            strata[label]=dict(n_corners=int(mask.sum()),passes=bystep)
        for step,ms in enumerate(allmetrics):
            eligible=[m for m in ms if m['evaluable'] and m['matched']]
            phase=[m['id'] for m in eligible if m['branch']!=0 and m['E_fixed']*math.hypot(*next(r['raw_hw'] for r in rows if r['id']==m['id']))>20 and m['frame_mean_px']<=10]
            symmetry[str(step)]=dict(branch_counts=dict(Counter(m['branch'] for m in eligible)),matched_frames=len(eligible),
                phase_only_frames=len(phase),phase_only_fraction=len(phase)/len(eligible) if eligible else None,phase_only_ids=phase,
                branch_changed_from_R0=sum(m.get('branch')!=b.get('branch') for m,b in zip(ms,allmetrics[0])),
                fixed_worse_sym_better_than_R0=sum(m.get('E_fixed',0)>b.get('E_fixed',0)+1e-12 and m.get('E_sym',0)<b.get('E_sym',0)-1e-12 for m,b in zip(ms,allmetrics[0])))
        oscillation=dict(corner_eligible=int(support.sum()),frame_eligible=int(frames_support.sum()),
            corner_types={k:int((v&support).sum()) for k,v in cats.items()},
            frame_types={k:int((v&frames_support).sum()) for k,v in frame_types.items()},pairs={})
        for j,label in enumerate(['d1_d2','d2_d3']):
            eligible=np.isfinite(cos[:,j])&support; values=cos[:,j]
            oscillation['pairs'][label]=dict(cosine=stats(values[eligible]),reversal=fraction(values<0,eligible),
                orthogonal_ish=fraction((values>=0)&(values<=.1),eligible),same_direction=fraction(values>.1,eligible),
                undefined_corners=int((support&~np.isfinite(values)).sum()))
        obj={}
        for typ in sorted({targets[r['id']]['object'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if targets[r['id']]['object']==typ]; mask=np.zeros_like(support); mask[ix]=support[ix]
            obj[typ]=dict(frames=len(ix),dimensions_WDH_unique=np.unique(dimensions[ix],axis=0).tolist(),
                passes={str(p):R.summary([ms[i] for i in ix]) for p,ms in enumerate(allmetrics)},
                cap_hit_PASS1=fraction(hit[:,0],mask),PASS3_vs_PASS1_corner_regression=fraction(errors[:,3]>errors[:,1]+1e-9,mask))
        out['chains'][mode]=dict(passes={str(p):R.summary(ms) for p,ms in enumerate(allmetrics)},pose={str(p):v for p,v in enumerate(poses)},
            movement=movements,error_strata=strata,oscillation=oscillation,symmetry=symmetry,objects=obj)
        saved.update({mode+'_'+k:v for k,v in dict(errors_native_PASS0_branch=errors,errors_canonical_current_branch=canonical,
            raw_errors=raw_errors,capped_errors=capped_errors,GT_support=support,current_branch=branch,
            raw_displacement=raw_norm,capped_displacement=capped_norm,actual_displacement=norm,cap_hit=hit,
            cosines=cos,locked_gt=locked_gt,**{'corner_'+k:v for k,v in cats.items()},**{'frame_'+k:v for k,v in frame_types.items()}).items()})
        write(RAW/f'pose/seed{seed}_{split}_{mode}.json',matrices)
    if split=='REAL_DEV': out['current_evaluator_parity']=verify_metric_parity(metric_rows['RAW'][0])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),ids=[r['id'] for r in rows],**saved)
    write(RAW/f'metrics/seed{seed}_{split}.json',metric_rows)
    freeze(path,out)
    print('SCORED',seed,split,'RAW med',[round(out['chains']['RAW']['passes'][str(p)]['matched_pooled_corner8_median_px'],3) for p in range(4)],flush=True)
    return out

def score():
    R.setup(); R.verify_sources()
    freeze(DOC/'ANALYSIS_CODE_LOCK.json',dict(code=bound(__file__),protocol=bound(DOC/'REPLAY_PROTOCOL.json'),training_updates=0))
    inputs=read(RAW/'INPUTS.json'); targets=read(RAW/'TARGETS.json'); results={}; start=time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for split in R.SPLITS:
            results[split]={}
            for s in (1,2,3):
                wait_start=time.monotonic()
                while not (RAW/f'predictions/seed{s}_{split}.npz').exists():
                    assert time.monotonic()-wait_start<7200, 'Inference output not available within2h; inspect inference log'
                    time.sleep(5)
                results[split][str(s)]=score_split(s,split,inputs,targets,pool)
    assert read(DOC/'INFERENCE_COMPLETE.json')['complete']
    panel={}
    for split,seeds in results.items():
        panel[split]=dict(per_seed={s:r['chains'] for s,r in seeds.items()},mean_seed={})
        for mode in R.CHAINS:
            panel[split]['mean_seed'][mode]={key:scalar_mean([r['chains'][mode][key] for r in seeds.values()]) for key in ('passes','pose','movement','error_strata','oscillation')}
    freeze(DOC/'PASS_METRICS_SYNTH.json',dict(complete=True,mean_definition='Arithmetic mean of three seed summary statistics, not pooled-seed median',splits={s:panel[s] for s in R.SPLITS[:3]}))
    freeze(DOC/'PASS_METRICS_REAL_DEV.json',dict(complete=True,role='REUSED_DEVELOPMENT_DIAGNOSIS_ONLY',splits={s:panel[s] for s in R.SPLITS[3:]}))
    for name,key in [('MOVEMENT_DISTRIBUTION','movement'),('OSCILLATION_ANALYSIS','oscillation'),('ERROR_STRATA','error_strata'),('SYMMETRY_DIAGNOSTIC','symmetry'),('POSE_RESULTS','pose')]:
        freeze(DOC/(name+'.json'),dict(complete=True,results={sp:{s:{mode:r['chains'][mode][key] for mode in R.CHAINS} for s,r in seeds.items()} for sp,seeds in results.items()}))
    freeze(DOC/'CAP_SATURATION.json',dict(complete=True,comparison='RAW and CAPPED candidates from identical input; feedback trajectories explicitly separate',
        results={sp:{s:{mode:dict(passes=r['chains'][mode]['movement'],initial_error_strata=r['chains'][mode]['error_strata']) for mode in R.CHAINS} for s,r in seeds.items()} for sp,seeds in results.items()}))
    freeze(DOC/'SCORING_COMPLETE.json',dict(complete=True,seconds=time.monotonic()-start,training_updates=0,
        current_evaluator_parity=[results['REAL_DEV'][str(s)]['current_evaluator_parity'] for s in (1,2,3)],
        prediction_preservation_checks=sum(r['preservation_checks'] for seeds in results.values() for r in seeds.values()),
        scoring_code=bound(__file__)))

def report():
    from presentation import build
    build()

def verify():
    R.verify_sources()
    lock=read(DOC/'ANALYSIS_CODE_LOCK.json'); assert R.E.sha(ROOT/lock['code']['path'])==lock['code']['sha256']
    for b in read(DOC/'INFERENCE_COMPLETE.json')['outputs']: assert R.E.sha(ROOT/b['path'])==b['sha256']
    assert read(DOC/'SCORING_COMPLETE.json')['training_updates']==0
    print('VERIFIED',flush=True)
