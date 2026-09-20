"""Unlabeled point-pool availability, distinct from best-whole-C2 scoring."""
import numpy as np
from . import dino_joint as X

C=X.C


def main():
    X.verify();roots=[X.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json']+[X.V.RAW/f'EVAL_PREDICTIONS_{a}.json' for a in X.E.ARMS]
    predictions=[{r['id']:r for r in C.read(path)['records'] if r['kind']=='PLASTIC'} for path in roots]
    base={r['id']:r for r in C.read(X.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    pe,pop=X.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in base}
    rows=[]
    for key,b in base.items():
        if not b['matched']:continue
        target=targets[key];gt=np.asarray(target.keypoints_xy);valid=np.asarray(target.keypoint_supervision_mask)[:8]
        pools=[np.asarray(X.P.top(pred[key]['prediction'])['keypoints_xy'])[:8] for pred in predictions]
        pools=[q[np.isfinite(q).all(-1)&~(q==-1).all(-1)] for q in pools]
        for j in np.flatnonzero(valid):
            d0=float(np.linalg.norm(pools[0]-gt[j],axis=-1).min())
            if d0<=40:continue
            d=[float(np.linalg.norm(q-gt[j],axis=-1).min()) for q in pools]
            rows.append(dict(id=key,GT_corner=int(j),nearest_R0_px=d0,nearest_any_SYN_px=d[1],nearest_any_MIX_px=d[2],nearest_any_pool_px=min(d),available_within10=min(d)<=10))
    assert len(rows)==64
    summary=dict(far_spatial_corners=64,any_native_point_in_three_sets_within10=sum(r['available_within10'] for r in rows),
        no_native_point_in_three_sets_within10=sum(not r['available_within10'] for r in rows))
    C.freeze(X.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',dict(status='POSTHOC_GT_DIAGNOSTIC_NOT_OUTPUT',summary=summary,rows=rows,
        clarification='HEADROOM_DIAGNOSTIC far_spatial_any_candidate_any_branch_available uses each whole candidate evaluated with its minimum-mean official C2 branch. It is not arbitrary-branch or unlabeled point-pool availability. This file ignores all channel/role correspondence and measures nearest ANY native point instead.',
        warning='Availability is not learned recovery or simultaneous pose feasibility. No candidate selected for output/training.',
        evidence=[C.bound(__file__),C.bound(X.DOC/'HEADROOM_DIAGNOSTIC.json')]+[C.bound(p) for p in roots]))
    print('JOINT_SPATIAL_POOL',summary,flush=True)


if __name__=='__main__':main()
