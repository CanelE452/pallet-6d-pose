"""GT-only reachable-domain diagnosis; never selects inference crops."""
import numpy as np
from . import dino_wide as W

D=W.D;C=W.C


def main():
    W.verify()
    receipts={r['id']:r for r in C.read(D.RAW/'INFERENCE_RECEIPTS.json')}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in base}
    corners=[r for r in C.read(D.RAW/'CORNER_DIAGNOSTICS.json')['rows'] if r['arm']=='SYN'];rows=[]
    for r in corners:
        point=np.asarray(targets[r['id']].keypoints_xy)[r['GT_corner']]
        matrix=np.asarray(receipts[r['id']]['matrix']);wide=W.W.widen_matrix(matrix)
        distances={}
        for name,matrix_,bound in [('parent',matrix,[284,380]),('wide',wide,[572,764])]:
            q=(matrix_@np.r_[point,1])[:2]
            distances[name]=float(np.linalg.norm(q-np.clip(q,[0,0],bound))/matrix_[0,0])
        h,w=base[r['id']]['raw_hw']
        rows.append(dict(id=r['id'],GT_corner=r['GT_corner'],R0_error=r['before'],nearest_R0_point_distance=r['nearest_R0_point_distance'],
            raw_image_contains_reference=bool(((point>=0)&(point<[w,h])).all()),minimum_possible_error_px=distances))
    groups={}
    for name,rr in [('all_matched_supervised',rows),('no_old_point_within40',[r for r in rows if r['nearest_R0_point_distance']>40]),
                    ('R0_over100',[r for r in rows if r['R0_error']>100])]:
        groups[name]=dict(corners=len(rr),in_raw_frame=sum(r['raw_image_contains_reference'] for r in rr),
            modes={m:dict(inside_grid=sum(r['minimum_possible_error_px'][m]<1e-9 for r in rr),
                reachable_within10=sum(r['minimum_possible_error_px'][m]<=10 for r in rr),
                max_minimum_error_px=max(r['minimum_possible_error_px'][m] for r in rr)) for m in ['parent','wide']})
    result=dict(status='GT_ONLY_OUTPUT_DOMAIN_DIAGNOSTIC_NOT_MODEL_ACCURACY',groups=groups,rows=rows,
        limitation='A reachable GT is not a predicted/recovered corner. Expanded FOV is fixed from R0 box,not selected using this GT. No frame/corner exclusions.',
        new_labels=0,predictions_modified=False,goal_complete=False,
        evidence=[C.bound(__file__),C.bound(W.DOC/'PROTOCOL.json'),C.bound(D.RAW/'INFERENCE_RECEIPTS.json'),C.bound(D.RAW/'CORNER_DIAGNOSTICS.json')])
    C.freeze(W.DOC/'CROP_SUPPORT_DIAGNOSTIC.json',result)
    print('CROP_SUPPORT_DIAGNOSTIC',groups,flush=True)


if __name__=='__main__':main()
