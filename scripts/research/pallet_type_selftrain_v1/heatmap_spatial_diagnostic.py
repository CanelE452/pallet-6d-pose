"""Separate existing-point-set proximity from genuinely missing spatial support.

GT-only posthoc diagnostic, never a model feature, gate or pseudo label.
"""
import numpy as np
from . import heatmap_mode_recovery as H

C=H.C


def main():
    H.verify()
    diagnostic=C.read(H.DOC/'CANDIDATE_HEADROOM.json')
    for b in C.read(H.DOC/'CANDIDATES_LOCK.json')['artifacts']:C.verify(b)
    preds={r['id']:np.asarray(H.P.top(r['prediction'])['keypoints_xy'])[:8]
           for r in C.read(H.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    pe,pop=H.R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds}
    rows=[]
    for row in diagnostic['rows']:
        gt=np.asarray(targets[row['id']].keypoints_xy)[row['GT_corner']]
        distance=np.linalg.norm(preds[row['id']]-gt,axis=-1)
        rows.append(dict(row,nearest_existing_R0_native=int(distance.argmin()),
            nearest_existing_R0_distance=float(distance.min()),existing_R0_point_within10=bool(distance.min()<=10)))
    result={}
    for name in H.MODELS:
        result[name]={}
        for threshold in [20,40,100]:
            table={}
            for present,label in [(True,'existing_R0_point_near_reference'),(False,'no_R0_point_near_reference')]:
                subset=[r for r in rows if r['model']==name and r['R0_error']>threshold and r['existing_R0_point_within10']==present]
                table[label]=dict(hard=len(subset),actual_top1_within10=sum(r['candidate_errors'][0]<=10 for r in subset),
                    ORACLE_five_candidate_coverage_within10=sum(r['ORACLE_error']<=10 for r in subset))
            result[name][str(threshold)]=table
    successes=[r for r in rows if r['R0_error']>40 and r['candidate_errors'][0]<=10]
    output=dict(status='POSTHOC_DIAGNOSTIC_ONLY',results=result,actual_top1_over40_cases=successes,
        interpretation='Near another R0 point suggests correspondence ambiguity,not proof of its cause;different projected corners can be close. No R0 point within10 is a stricter new-spatial-support bucket. Do not count coordinate readout movement alone as discovery of a previously missing physical corner.',
        oracle_warning='GT nearest of5 is an upper-bound diagnostic,not a learned selection,model accuracy,pseudo target or deployable correction.',
        model_outputs_modified=False,new_labels=0,new_training=0,
        evidence=[C.bound(__file__),C.bound(H.DOC/'CANDIDATE_HEADROOM.json'),C.bound(H.DOC/'COMPLETION_AUDIT.json')])
    C.freeze(H.DOC/'SPATIAL_DECOMPOSITION.json',output)
    print('SPATIAL_DECOMPOSITION',result,flush=True)


if __name__=='__main__':main()
