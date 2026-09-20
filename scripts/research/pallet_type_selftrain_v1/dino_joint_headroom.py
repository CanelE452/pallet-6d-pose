"""Posthoc source frontier and real candidate headroom; never inference outputs."""
import numpy as np
import torch
from . import dino_joint as X
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=X.C;J=X.J


@torch.no_grad()
def main():
    p=X.verify();X.N.setup();print('GPU',X.N.E.gpu(),flush=True)
    binding=C.read(X.DOC/'SOURCE_CACHE.json')['artifact'];C.verify(binding)
    with np.load(C.ROOT/binding['path']) as z:data={k:np.array(z[k]) for k in z.files}
    model,mean,std,_=X.load();cal=np.isin(data['row'],p['calibration_rows'])
    prob=model(X.E.normalized(torch.as_tensor(data['features'][cal],device='cuda'),mean,std)).sigmoid().cpu().numpy()
    e=data['errors'][cal];clean=data['view'][cal]==0;benefit=J.beneficial(e)
    maxima=prob[:,1:].max(-1);thresholds=np.r_[np.unique(maxima),np.nextafter(maxima.max(),np.inf)]
    frontier=[]
    for t in thresholds:
        choice=J.choose(prob,t);use=choice!=0;s=J.stats(e[clean],choice[clean]);allstats=J.stats(e,choice)
        precision=float(benefit[np.arange(len(e)),choice][use].mean()) if use.any() else 1.
        feasible=s['PCK10']>=s['input_PCK10']-.01 and s['PCK20']>=s['input_PCK20'] and s['P90']<=1.1*s['input_P90'] and s['damaged']<=.01*s['good'] and precision>=.95
        frontier.append(dict(threshold=float(t),selected=int(use.sum()),beneficial_precision=precision,clean=s,combined=allstats,feasible=bool(feasible)))
    usable=[r for r in frontier if r['feasible'] and r['selected']]
    best=max(usable,key=lambda r:(r['clean']['recovered'],r['combined']['recovered'],r['threshold'])) if usable else None
    original={r['id']:r for r in C.read(X.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    # Fixed candidates are reconstructed from immutable prediction outputs, not GT.
    visual={a:{r['id']:r for r in C.read(X.V.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in X.E.ARMS}
    pe,pop=X.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    baseline={r['id']:r for r in C.read(X.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    rows=[];headroom=0;all20_before=0;all20_possible=0;spatial_count=0;spatial_possible=0
    for key,old in original.items():
        b=baseline[key]
        if not b['matched']:continue
        sets=[np.asarray(X.P.top(r['prediction'])['keypoints_xy']) for r in [old,visual['SYN'][key],visual['MIX'][key]]]
        candidates=[q[perm].copy() for q in sets for perm in J.PERMS]
        for q in candidates:q[8]=sets[0][8]
        target=targets[key];gt=np.asarray(target.keypoints_xy)
        measures=[EM.measure(q,gt,target.keypoint_supervision_mask,perms,old['raw_hw']) for q in candidates]
        ee=np.array([r['canonical_errors'] for r in measures],float);valid=np.isfinite(ee[0])
        oracle=int(np.nanmean(ee,axis=-1).argmin())
        oldmean=b['frame_mean_px'];bestmean=measures[oracle]['frame_mean_px']
        all20_before+=int((ee[0,valid]<=20).all());all20_possible+=int(any((r[valid]<=20).all() for r in ee))
        better=bestmean+5<oldmean;headroom+=int(better)
        nearest=np.linalg.norm(sets[0][:8,None]-gt[None,:8],axis=-1).min(0)
        far=valid&(nearest>40);spatial_count+=int(far.sum())
        # Union is a per-corner availability bound, not one coherent attainable output.
        possible=(ee<=10).any(0);spatial_possible+=int((far&possible).sum())
        rows.append(dict(id=key,R0_mean_px=oldmean,GT_ORACLE_NOT_METHOD_mean_px=bestmean,
            GT_ORACLE_NOT_METHOD_candidate=J.NAMES[oracle],candidate_mean_errors=[r['frame_mean_px'] for r in measures],
            all_corners20_possible=any((r[valid]<=20).all() for r in ee)))
    result=dict(status='POSTHOC_DIAGNOSTIC_NOT_DEPLOYABLE_RESULT',threshold_changed=False,predictions_changed=False,
        source_exact_frontier=frontier,source_feasible_nonempty=len(usable),source_best_nonempty=best,
        real_GT_headroom=dict(matched_frames=len(rows),mean_improvement_over5_possible_frames=headroom,
            all_supervised_corners20_R0_frames=all20_before,all_supervised_corners20_any_whole_candidate_frames=all20_possible,
            far_spatial_corners=spatial_count,far_spatial_any_candidate_any_branch_available=spatial_possible),rows=rows,
        warning='GT oracle candidates and source finer thresholds are diagnostic only; no new predictions/pseudo labels selected. Per-corner union bound cannot imply a coherent whole-frame solution.',
        evidence=[C.bound(__file__),C.bound(X.DOC/'PROTOCOL.json'),C.bound(X.DOC/'SOURCE_CACHE.json'),C.bound(X.DOC/'FIT_JOINT.json'),
            C.bound(X.RAW/'EVAL_PREDICTIONS_JOINT.json')]+[C.bound(X.V.RAW/f'EVAL_PREDICTIONS_{a}.json') for a in X.E.ARMS])
    C.freeze(X.DOC/'HEADROOM_DIAGNOSTIC.json',result)
    print('JOINT_HEADROOM',dict(source_best=best,source_nonempty=len(usable),real=result['real_GT_headroom'],
        requested=next(r for r in rows if r['id']=='eval_pallet07:1778652166837872128')),flush=True)


if __name__=='__main__':main()
