"""Full-precision read-only historical adapters; exact paired pooled statistics."""
import csv
import numpy as np
from env import *
from paired_stats_helper import paired_pooled_median,_prepare,_median
def load(include_d=False):
    pe=old('paper_evaluation');ag=old('aggregate_results');pop=pe.population()
    metadata={r['frame_id']:r for r in read(pe.POS)['items']};base=read(LINE/'baseline/FULL_CANDIDATES.json')
    names=['R0']+[f'{a}{s}' for a in ('P','L')+ (('D',) if include_d else ()) for s in (1,2,3)]
    image_to_id={pe.canonical_key(i.image):i.frame_id for i in pop.positive.items}
    pose_to_dev={r['frame_id']:image_to_id[pe.canonical_key(r['image'])] for r in read(C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']}
    assert len(pose_to_dev)==len(set(pose_to_dev.values()))==len(pop.positive.items)
    stores={};summaries={};pose={};sources={}
    for name in names:
        path=(BRAW/f'evaluation/{name}' if name.startswith('P') else RAW/f'evaluation/{name}' if name.startswith('D') else LINE/f'evaluation/image_line_only_seed{name[1:]}' if name!='R0' else LINE/'evaluation/R0')
        frames=base['frames'] if name=='R0' else read(path/'PREDICTIONS.json')['frames']
        csvpath=pe.OLDCSV if name=='R0' else path/'PAPER_2D_per_frame.csv'
        csvrows=ag.load_keypoint_rows(csvpath);rows=[];denom=0;max_rounding=0
        for item in pop.positive.items:
            key=item.frame_id;t=pe.E._legacy_forbidden_target(item);cand=frames[pe.canonical_key(item.image)]
            reference=base['frames'][pe.canonical_key(item.image)]
            assert len(cand)==len(reference)
            for c,r in zip(cand,reference):
                assert c['score']==r['score'] and c['box_xyxy']==r['box_xyxy']
                if c['keypoints_xy'] is not None:assert c['keypoints_xy'][8]==r['keypoints_xy'][8]
            top=max(cand,key=lambda c:c['score']) if cand else None
            matched=top is not None and pe.E._box_iou(np.asarray(top['box_xyxy']),t.box_xyxy)>=.5 and top['keypoints_xy'] is not None
            mask=t.keypoint_supervision_mask;denom+=int(mask.sum())
            errors=np.linalg.norm(np.asarray(top['keypoints_xy'],float)-t.keypoints_xy,axis=-1)[mask] if matched else np.empty(0)
            assert np.isfinite(errors).all(),'Nonfinite predictions: do not silently exclude'
            assert np.allclose(errors,csvrows[key]['errors'],atol=5.01e-7,rtol=0)
            if len(errors):max_rounding=max(max_rounding,float(np.max(np.abs(errors-csvrows[key]['errors']))))
            e8=np.linalg.norm(np.asarray(top['keypoints_xy'],float)[:8]-t.keypoints_xy[:8],axis=-1)[mask[:8]] if matched else np.empty(0)
            rows.append(dict(frame_id=key,session_id=metadata[key]['session_id'],errors=errors,errors8=e8,gt=int(mask.sum())))
        pp=C.POSE/'POSE_PER_FRAME_BY_ARM.json' if name=='R0' else path/'POSE_PER_FRAME_BY_ARM.json'
        arm=f'image_line_only_seed{name[1:]}' if name.startswith('L') else name
        original_prows=read(pp)['per_frame'][arm]
        prows=[dict(r,frame_id=pose_to_dev[r['frame_id']],original_pose_frame_id=r['frame_id'],session_id=metadata[pose_to_dev[r['frame_id']]]['session_id']) for r in original_prows]
        pose[name]={r['frame_id']:r for r in prows};assert len(pose[name])==len(prows)
        stores[name]=rows;e=np.concatenate([r['errors'] for r in rows]);e8=np.concatenate([r['errors8'] for r in rows])
        ps=ag.pose_summary(prows);ps.update(yaw_median_deg=float(np.median([r['yaw_error_deg'] for r in prows])),coverage=len(prows)/len(rows),excluded_frame_ids=sorted(set(metadata)-set(pose[name])))
        summaries[name]=dict(median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),frame_mean_px=float(np.mean([r['errors'].mean() for r in rows if len(r['errors'])])),
            gross20=float((e>20).mean()),max_error_px=float(e.max()),corner8_median_px=float(np.median(e8)),matched_frames=sum(bool(len(r['errors'])) for r in rows),supervised_points=len(e),gt_denominator=denom,
            ALL_GT_PCK={str(t):float((e<=t).sum()/denom) for t in (5,10,20)},pose=ps,csv_rounding_max_px=max_rounding)
        metric=read(pe.OLD2D if name=='R0' else path/'PAPER_2D.json')['metrics']['box_and_keypoint_2d']
        assert abs(metric['keypoint_location_median_px']-summaries[name]['median_px'])<1e-10
        sources[name]=[bound(p) for p in [csvpath,pp,pe.OLD2D if name=='R0' else path/'PAPER_2D.json']]
    support=[[len(r['errors']) for r in rows] for rows in stores.values()];assert all(v==support[0] for v in support),'Common support mismatch; report separately before paired precision'
    return stores,pose,summaries,sources
def contrast(stores,left='P',right='R0',level='session'):
    base=stores['R0'];sessions=[r['session_id'] for r in base]
    if right=='R0':return paired_pooled_median([r['errors'] for r in base],[[r['errors'] for r in stores[f'{left}{s}']] for s in (1,2,3)],sessions,level=level)
    names=sorted(set(sessions)) if level=='session' else list(range(len(base)))
    group=np.array([names.index(s) for s in sessions]) if level=='session' else np.arange(len(base));n=len(names)
    ls=[_prepare([r['errors'] for r in stores[f'{left}{s}']],group) for s in (1,2,3)]
    rs=[_prepare([r['errors'] for r in stores[f'{right}{s}']],group) for s in (1,2,3)]
    stat=lambda c:float(np.mean([_median(l,c)-_median(r,c) for l,r in zip(ls,rs)]))
    rng=np.random.default_rng(20260914);draws=np.array([stat(rng.multinomial(n,np.full(n,1/n))) for _ in range(10000)])
    return dict(estimand=f'mean_seed(pooled median {left}-pooled median same-seed {right})',delta=stat(np.ones(n,dtype=int)),low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),level=level,units=n,frames=len(base),resamples=10000,seed=20260914)
def run(include_d=False):
    verify();stores,poses,summaries,sources=load(include_d)
    write(DOC/('UNIFIED_DEV_RESULTS.json' if include_d else 'FIXED_DEV_RESULTS.json'),dict(complete=True,methods=summaries,sources=sources,role='REUSED_DEV',common_support_exact=True))
    pairs=[('P','R0')]+([('P','D')] if include_d else [])
    for l,r in pairs:
        result={level:contrast(stores,l,r,level) for level in ('session','frame')}
        sessions=sorted({row['session_id'] for row in stores['R0']});effects=[]
        for session in sessions:
            for mode in ('only_session','leave_one_session_out'):
                ids=[i for i,row in enumerate(stores['R0']) if (row['session_id']==session)==(mode=='only_session')]
                med=lambda name:float(np.median(np.concatenate([stores[name][i]['errors'] for i in ids])))
                ref=med('R0') if r=='R0' else float(np.mean([med(f'{r}{s}') for s in (1,2,3)]))
                effects.append(dict(comparison=f'{l}-{r}',session=session,mode=mode,frames=len(ids),reference_px=ref,candidate_px=float(np.mean([med(f'{l}{s}') for s in (1,2,3)])),delta_px=float(np.mean([med(f'{l}{s}') for s in (1,2,3)]))-ref))
        result['per_session_and_LOSO']=effects
        write(DOC/f'{l}_VS_{r}_PAIRED.json',result)
        print(l,r,result['session'],flush=True)
        path=DOC/('PER_SESSION_EFFECTS.csv' if r=='R0' else 'PER_SESSION_P_D.csv')
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
    write(RAW/'DEV_FULL_PRECISION_ERRORS.json',{name:[{**r,'errors':r['errors'].tolist(),'errors8':r['errors8'].tolist()} for r in rows] for name,rows in stores.items()})
if __name__=='__main__':run('--include-d' in sys.argv)
