"""Posthoc ORACLE coverage of fixed GT-free modes; never a deployed correction."""
import numpy as np
import torch
from . import heatmap_mode_recovery as H

C=H.C


def top_five(logits):
    if logits.ndim!=4 or logits.shape[-2:]!=(96,72) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite B,K,96,72 logits')
    # Five candidates, 11x11 anchor suppression, 5x5 local average fixed here.
    # No target, initial points, geometry or GT correctness inputs.
    work=logits.clone();p=logits.flatten(2).softmax(-1).reshape_as(logits)
    yy=torch.arange(96,device=logits.device)[None,None,:,None]
    xx=torch.arange(72,device=logits.device)[None,None,None,:]
    locations=[];masses=[];anchors=[]
    for _ in range(5):
        anchor=work.flatten(2).argmax(-1);x=anchor%72;y=anchor//72
        dx=(xx-x[...,None,None]).abs();dy=(yy-y[...,None,None]).abs()
        window=(dx<=2)&(dy<=2)
        locations.append(H.N.C.expectation(logits.masked_fill(~window,-torch.inf)))
        masses.append((p*window).sum((-2,-1)))
        anchors.append(torch.stack([x,y],-1))
        work=work.masked_fill((dx<=5)&(dy<=5),-torch.inf)
    return dict(points=torch.stack(locations,2),mass=torch.stack(masses,2),anchors=torch.stack(anchors,2))


def main():
    H.verify();H.N.setup()
    for b in C.read(H.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    protocol=dict(status='POSTHOC_ORACLE_DIAGNOSTIC_NOT_METHOD',count=5,anchor_suppression_radius_cells=5,
        local_average_radius_cells=2,candidate_stage='GT-free pure logits;all coordinates frozen before diagnostic GT access',
        scoring='R0 selected whole-C2 branch fixed for all candidates;matched supervised first8corners only. GT nearest candidate is ORACLE coverage,not actual choice or performance. No relabeling,training targets,threshold tuning or promotion.',
        purpose='Determine whether a better candidate is present in frozen maps before deciding between proposal generation and selection. Reused DEV diagnostic,not independent confirmation.',
        inputs=[C.bound(H.RAW/f'REAL_LOGITS_{m}.npz') for m in H.MODELS],code=C.bound(__file__))
    C.freeze(H.DOC/'CANDIDATE_DIAGNOSTIC_PROTOCOL.json',protocol)
    arrays={};bindings=[]
    for name in H.MODELS:
        z=np.load(H.RAW/f'REAL_LOGITS_{name}.npz');a=top_five(torch.from_numpy(z['logits']))
        xy=a['points'].numpy();raw=np.stack([H.N.C.transform_points(q.reshape(-1,2),np.linalg.inv(m)).reshape(9,5,2) for q,m in zip(xy,z['matrix'])])
        path=H.RAW/f'FIVE_CANDIDATES_{name}.npz'
        H.I.save_npz(path,points=raw,probability_mass=a['mass'].numpy(),anchors=a['anchors'].numpy(),ids=z['ids'])
        arrays[name]=dict(np.load(path));bindings.append(C.bound(path))
    C.freeze(H.DOC/'CANDIDATES_LOCK.json',dict(artifacts=bindings,before_diagnostic_GT_read=True))
    # No GT access until ALL GT-free proposals above are fixed.
    baseline={r['id']:r for r in C.read(H.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    pe,pop=H.R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in baseline}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    results={};rows=[]
    for name,z in arrays.items():
        local=[]
        for idx,key in enumerate(z['ids']):
            key=str(key);b=baseline[key]
            if not b['matched']:continue
            gt=np.asarray(targets[key].keypoints_xy);valid=np.asarray(targets[key].keypoint_supervision_mask,bool)
            perm=perms[b['branch']]
            for j,g in enumerate(perm[:8]):
                if not valid[g]:continue
                e=np.linalg.norm(z['points'][idx,j]-gt[g],axis=-1)
                best=int(e.argmin());old=b['canonical_errors'][g]
                assert old is not None
                row=dict(id=key,model=name,native=j,GT_corner=g,R0_branch=b['branch'],R0_error=old,
                    candidate_errors=e.tolist(),ORACLE_rank=best+1,ORACLE_error=float(e[best]),
                    candidate_probability_mass=z['probability_mass'][idx,j].tolist())
                local.append(row);rows.append(row)
        table={}
        for threshold in [20,40,100]:
            hard=[r for r in local if r['R0_error']>threshold]
            possible=[r for r in hard if r['ORACLE_error']<=10]
            mass=[r['candidate_probability_mass'][r['ORACLE_rank']-1] for r in possible]
            table[str(threshold)]=dict(hard=len(hard),ORACLE_coverage_within10=len(possible),
                actual_top1_within10=sum(r['candidate_errors'][0]<=10 for r in hard),
                ORACLE_rank_counts={str(k):sum(r['ORACLE_rank']==k for r in possible) for k in range(1,6)},
                ORACLE_candidate_mass_quantiles=np.quantile(mass,[0,.5,1]).tolist() if mass else None)
        results[name]=table
    C.freeze(H.DOC/'CANDIDATE_HEADROOM.json',dict(status='ORACLE_NOT_ACHIEVED_PERFORMANCE',results=results,
        rows=rows,new_training=0,used_for_pseudo_labels=False,used_to_modify_predictions=False,
        candidate_lock=C.bound(H.DOC/'CANDIDATES_LOCK.json'),protocol=C.bound(H.DOC/'CANDIDATE_DIAGNOSTIC_PROTOCOL.json')))
    print('HEATMAP_CANDIDATE_ORACLE_ONLY',results,flush=True)


if __name__=='__main__':main()
