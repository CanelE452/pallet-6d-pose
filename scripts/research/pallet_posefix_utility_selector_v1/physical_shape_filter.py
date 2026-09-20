"""Two extra whole-image vetoes, never coordinate repair or a new PnP solver.

Only frozen all8-LOO survivors are eligible. For one common allowed registry
hypothesis require the existing LOO threshold, all-positive camera depths,
and six valid convex face cycles with winding consistent with its projection.
Global edge overlaps between DIFFERENT faces are explicitly allowed.
"""
import argparse
import copy
import json
from collections import Counter

import cv2
import numpy as np

from . import all8_replay_frame_filter as A
from . import core as P
from scripts.self_training_yolo import pseudo_label_filters as F

DOC=P.DOC/'physical_shape_filter'
RAW=P.RAW/'physical_shape_filter'
FACES=((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
FACE_NAMES=('near','far','top','right','bottom','left')
# Numerical, not calibrated failure-probability thresholds. Coordinates are
# normalised by the observed eight-point diagonal before these area tests.
AREA_EPS=1e-8
DEPTH_REL_EPS=1e-6


def cross(a,b):
    return float(a[0]*b[1]-a[1]*b[0])


def polygon_details(points):
    q=np.asarray(points,float)
    assert q.shape==(4,2) and np.isfinite(q).all()
    turns=[cross(q[(i+1)%4]-q[i],q[(i+2)%4]-q[(i+1)%4]) for i in range(4)]
    area=.5*sum(cross(q[i],q[(i+1)%4]) for i in range(4))
    def opposite(a,b):
        return (a>AREA_EPS and b < -AREA_EPS) or (b>AREA_EPS and a < -AREA_EPS)
    def proper_intersection(a,b,c,d):
        return opposite(cross(b-a,c-a),cross(b-a,d-a)) and opposite(cross(d-c,a-c),cross(d-c,b-c))
    crossing=proper_intersection(q[0],q[1],q[2],q[3]) or proper_intersection(q[1],q[2],q[3],q[0])
    concave=any(t>AREA_EPS for t in turns) and any(t < -AREA_EPS for t in turns)
    # Any collinear/duplicate vertex is a collapsed corner of the quadrilateral.
    degenerate=abs(area)<=AREA_EPS or any(abs(t)<=AREA_EPS for t in turns)
    return dict(signed_area=area,turns=turns,self_crossing=bool(crossing),
                concave=bool(concave),degenerate=bool(degenerate))


def face_checks(predicted,projected):
    q,r=np.asarray(predicted,float)[:8],np.asarray(projected,float)[:8]
    if q.shape!=(8,2) or r.shape!=(8,2) or not np.isfinite(q).all() or not np.isfinite(r).all():
        return dict(passed=False,reason='nonfinite_projection',faces=[])
    diagonal=float(np.linalg.norm(q[:,None,:]-q[None,:,:],axis=-1).max())
    if diagonal<=1e-9:return dict(passed=False,reason='collapsed_whole_prediction',faces=[])
    origin=q.mean(axis=0);q=(q-origin)/diagonal;r=(r-origin)/diagonal
    faces=[]
    for name,indices in zip(FACE_NAMES,FACES):
        a,b=polygon_details(q[list(indices)]),polygon_details(r[list(indices)])
        reasons=[]
        if a['self_crossing']:reasons.append('self_crossing')
        if a['concave']:reasons.append('concave_or_folded')
        if a['degenerate'] and not b['degenerate']:reasons.append('collapsed_face_or_corner')
        # A physically edge-on projected face has no reliable winding. Do not
        # falsely label its zero area as impossible just because it is edge-on.
        winding_tested=not a['degenerate'] and not b['degenerate']
        if winding_tested and a['signed_area']*b['signed_area']<0:
            reasons.append('winding_mismatch_with_PnP')
        faces.append(dict(name=name,corners=list(indices),predicted=a,projected=b,
            winding_tested=winding_tested,reasons=reasons,passed=not reasons))
    return dict(passed=all(f['passed'] for f in faces),faces=faces,
                observed_diagonal_px=diagonal,area_epsilon_normalized=AREA_EPS)


def depths_check(points_3d,rvec,tvec):
    xyz=np.asarray(points_3d,float)[:8]
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return dict(passed=False,reason='nonfinite_pose')
    rotation,_=cv2.Rodrigues(np.asarray(rvec,float))
    camera=xyz@rotation.T+np.asarray(tvec,float).reshape(1,3)
    scale=float(np.linalg.norm(xyz.max(0)-xyz.min(0)))
    threshold=max(1e-9,DEPTH_REL_EPS*scale)
    z=camera[:,2]
    return dict(passed=bool(np.isfinite(camera).all() and (z>threshold).all()),
        camera_depths_m=z.tolist(),minimum_depth_m=float(z.min()),epsilon_m=threshold)


def inspect(q,K,dimensions):
    hypotheses=[]
    for name,xyz in F.registry_hypotheses(dimensions):
        h=dict(name=name,loo_pass=False,depth_pass=False,face_pass=False,passed=False)
        try:
            score=F._hypothesis_scores(name,xyz,q,np.ones(9,bool),K,None,None)
            h['loo_score']=float(score.keypoint_removal) if np.isfinite(score.keypoint_removal) else None
            h['loo_pass']=h['loo_score'] is not None and h['loo_score']<=.05
            if score.rvec is not None and score.tvec is not None:
                depth=depths_check(xyz,score.rvec,score.tvec)
                h['depth']=depth;h['depth_pass']=depth['passed']
                projected=F._project(xyz,score.rvec,score.tvec,K)
                faces=face_checks(q,projected);h['shape']=faces;h['face_pass']=faces['passed']
                h['projected_xy']=projected.tolist()
                h['rvec']=score.rvec.reshape(-1).tolist();h['tvec']=score.tvec.reshape(-1).tolist()
                h['passed']=h['loo_pass'] and h['depth_pass'] and h['face_pass']
        except cv2.error as exc:h['solver_error']=str(exc)
        hypotheses.append(h)
    depth_pass=any(h['loo_pass'] and h['depth_pass'] for h in hypotheses)
    passed=any(h['passed'] for h in hypotheses)
    return dict(keep=passed,depth_pass=depth_pass,hypotheses=hypotheses,
        reason='accepted' if passed else ('nonphysical_or_unsolved_PnP' if not depth_pass else 'face_shape'),
        passing_hypotheses=[h['name'] for h in hypotheses if h['passed']])


def apply():
    P.verify();cv2.setNumThreads(1)
    previous=P.read(A.DOC/'OUTPUTS_LOCK.json')
    for b in previous['artifacts']+[previous['protocol']]:P.verify_binding(b)
    parent=P.read(A.DOC/'PROTOCOL.json')
    for b in [parent['source'],*parent['metadata'],*parent['code']]:P.verify_binding(b)
    protocol=dict(purpose='All8-LOO frozen Replay survivors -> positive camera depth and six-face topology/winding -> whole-image keep/drop',
        faces=[list(f) for f in FACES],face_names=FACE_NAMES,
        depth_epsilon='max(1e-9 m,1e-6 * registered 3D cuboid diagonal)',
        face_area_epsilon_normalized=AREA_EPS,
        normalization='Raw predicted 8-point diagonal; no GT',
        collapsed_definition='Numerically zero face area OR any collinear/duplicate face vertex, unless projected reference face is also degenerate',
        edge_on_policy='Skip winding when predicted or projected face is degenerate; proper within-face crossings and concavity still fail',
        winding='Compare with SAME indexed face of the candidate PnP projection, not a fixed screen orientation',
        hypothesis='Existence of ONE allowed W/D hypothesis passing original all8 LOO<=.05 AND all8positiveZ AND all six faces',
        solver='Unchanged existing EPNP for eight input points; no new optimizer or alternative-pose search',
        does_not_prove_no_physical_solution=True,global_edge_overlap_is_not_a_veto=True,
        no_new_reprojection_or_LOO_max_threshold=True,coordinates_changed=False,
        learned_selector=False,flip=False,self_training=False,evaluation_reused=True,
        parent_output=P.bound(A.DOC/'OUTPUTS_LOCK.json'),parent_results=P.bound(A.DOC/'RESULTS.json'),
        source=parent['source'],metadata=parent['metadata'],
        code=[P.bound(__file__),P.bound(P.ROOT/'scripts/self_training_yolo/pseudo_label_filters.py')],
        test_code=P.bound(P.HERE/'test_physical_shape_filter.py'))
    P.freeze(DOC/'PROTOCOL.json',protocol)
    if (DOC/'OUTPUTS_LOCK.json').exists():
        for b in P.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:P.verify_binding(b)
        print('PHYSICAL_SHAPE_OUTPUTS_ALREADY_FROZEN',flush=True);return
    eligible=P.read(A.RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json')
    olddec=P.read(A.RAW/'DECISIONS.json')
    split=P.read(P.ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json')['evaluation']
    green=P.read(P.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json')['records']
    meta={'DEV72':{r['id']:r for r in split},'GREEN150':{r['id']:r for r in green}}
    registry={r['object_type']:r['physical_dimensions_m'] for r in P.read(P.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')['objects']}
    accepted,decisions={},{}
    for ds,rows in eligible.items():
        accepted[ds],decisions[ds]=[],[]
        old_by_id={r['id']:r for r in olddec[ds]}
        for row in rows:
            original=json.dumps(row,sort_keys=True,allow_nan=False)
            q=np.asarray(A.N.top(row['prediction'])['keypoints_xy'],float)
            assert np.isfinite(q[:8]).all() and not (q[:8]==-1).all(1).any()
            m=meta[ds][row['id']];K=np.asarray(m['K'] if ds=='DEV72' else m['source_K'],float)
            kind=m['object_type'] if ds=='DEV72' else 'plastic_standard_110x110x15'
            result=inspect(q,K,registry[kind])
            loo=min(h['loo_score'] for h in result['hypotheses'] if h.get('loo_score') is not None)
            assert np.isclose(loo,old_by_id[row['id']]['s_remove_all8'],rtol=0,atol=1e-12)
            assert json.dumps(row,sort_keys=True,allow_nan=False)==original
            if result['keep']:accepted[ds].append(copy.deepcopy(row))
            decisions[ds].append(dict(id=row['id'],**result))
        print('PHYSICAL_SHAPE',ds,len(accepted[ds]),'/',len(rows),flush=True)
    P.freeze(RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json',accepted)
    P.freeze(RAW/'DECISIONS.json',decisions)
    P.freeze(DOC/'OUTPUTS_LOCK.json',dict(complete=True,all179_decisions_before_GT_scoring=True,
        protocol=P.bound(DOC/'PROTOCOL.json'),artifacts=[P.bound(RAW/n) for n in ('ACCEPTED_UNCHANGED_PREDICTIONS.json','DECISIONS.json')]))


def stats(rows):
    result=A.N.subset_summary(rows)
    errors=[e for r in rows for e in r['errors']]
    result.update(correct20=sum(e<=20 for e in errors),PCK20=sum(e<=20 for e in errors)/len(errors) if errors else None,
        bad20_images=sum(any(e>20 for e in r['errors']) for r in rows))
    return result


def score():
    lock=P.read(DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']+[lock['protocol']]:P.verify_binding(b)
    protocol=P.read(DOC/'PROTOCOL.json')
    for b in [protocol['source'],*protocol['metadata'],*protocol['code']]:P.verify_binding(b)
    old=P.read(A.DOC/'RESULTS.json');P.verify_binding(old['metrics_source'])
    metrics=P.read(P.N.RAW/'PER_FRAME_METRICS.json');decisions=P.read(RAW/'DECISIONS.json')
    summaries={}
    for ds,mode in [('DEV72','DEV72'),('GREEN150','GREEN150_MANUAL')]:
        d=decisions[ds];before={r['id'] for r in d};kept={r['id'] for r in d if r['keep']}
        depth={r['id'] for r in d if r['depth_pass']}
        m=metrics[mode]
        subset=lambda arm,ids:[r for r in m[arm] if r['id'] in ids]
        bad_before={r['id'] for r in subset('POSEFIX_RAW',before) if any(e>20 for e in r['errors'])}
        rejected=before-kept
        summaries[mode]=dict(before=stats(subset('POSEFIX_RAW',before)),
            after_depth_only=stats(subset('POSEFIX_RAW',depth)),after_both=stats(subset('POSEFIX_RAW',kept)),
            rejected=stats(subset('POSEFIX_RAW',rejected)),same_kept_N2=stats(subset('A_N2',kept)),
            removed_bad20_images=len(rejected&bad_before),removed_all_within20_images=len(rejected-bad_before),
            reason_counts=dict(Counter(r['reason'] for r in d)),
            retained_ids=sorted(kept),rejected_ids=sorted(rejected))
    result=dict(complete=True,summary=summaries,output_lock=P.bound(DOC/'OUTPUTS_LOCK.json'),
        source_metrics=old['metrics_source'],coordinates_changed=0,conditional_subset_metrics=True,
        evaluation_reused=True,model_modified=False,self_training=False)
    P.freeze(DOC/'RESULTS.json',result)
    print(json.dumps({k:{n:v for n,v in r.items() if not n.endswith('_ids')} for k,r in summaries.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['apply','score'])
    args=parser.parse_args();apply() if args.stage=='apply' else score()
