"""Replace the physical-shape postfilter with within-face X crossings ONLY."""
import argparse
import copy
import json

import numpy as np

from . import all8_replay_frame_filter as A
from . import core as P

DOC=P.DOC/'x_crossing_filter'
RAW=P.RAW/'x_crossing_filter'
FACES=((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
NAMES=('near','far','top','right','bottom','left')
EPS=1e-8


def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])


def proper_crossing(a,b,c,d):
    """Strict intersection inside BOTH segments; touch/collinearity is not X."""
    def opposite(x,y):return (x>EPS and y < -EPS) or (y>EPS and x < -EPS)
    return opposite(cross(b-a,c-a),cross(b-a,d-a)) and opposite(cross(d-c,a-c),cross(d-c,b-c))


def quadrilateral_x(q):
    q=np.asarray(q,float)
    assert q.shape==(4,2) and np.isfinite(q).all()
    return bool(proper_crossing(q[0],q[1],q[2],q[3]) or proper_crossing(q[1],q[2],q[3],q[0]))


def inspect(points):
    q=np.asarray(points,float)[:8]
    # Finite/missing checks belong to the already frozen all8 parent filter.
    assert q.shape==(8,2) and np.isfinite(q).all() and not (q==-1).all(1).any()
    d=float(np.linalg.norm(q[:,None,:]-q[None,:,:],axis=-1).max())
    q=(q-q.mean(0))/max(d,1e-9)
    faces=[dict(name=name,corners=list(face),x_crossing=quadrilateral_x(q[list(face)]))
           for name,face in zip(NAMES,FACES)]
    rejected=[r['name'] for r in faces if r['x_crossing']]
    return dict(keep=not rejected,reason='within_face_X' if rejected else 'accepted',
                crossing_faces=rejected,faces=faces,normalization_px=d)


def apply():
    P.verify()
    parent=P.read(A.DOC/'OUTPUTS_LOCK.json')
    for b in parent['artifacts']+[parent['protocol']]:P.verify_binding(b)
    protocol=dict(purpose='Use X-only postfilter INSTEAD OF previous physical-shape postfilter',
        input='All8 Replay median LOO survivors, NOT physical-shape survivors',
        rule='One of six faces has a proper interior crossing of its opposite boundary edges -> drop whole image',
        keep='Original complete Replay prediction, no coordinate changes or N2 fallback',
        faces=[list(f) for f in FACES],face_names=list(NAMES),numeric_orientation_epsilon=EPS,
        normalization='Observed eight-point diagonal, same numerical epsilon as previous crossing diagnostic',
        disabled=['positive-camera-depth veto','concavity veto','winding-vs-PnP veto','collapsed-face veto'],
        allowed=['crossing between different faces','endpoint touching','collinear overlap','concavity without X'],
        parent_output=P.bound(A.DOC/'OUTPUTS_LOCK.json'),parent_results=P.bound(A.DOC/'RESULTS.json'),
        code=P.bound(__file__),tests=P.bound(P.HERE/'test_x_crossing_filter.py'),
        GT_used_for_selection=False,model_modified=False,self_training=False,evaluation_reused=True)
    P.freeze(DOC/'PROTOCOL.json',protocol)
    if (DOC/'OUTPUTS_LOCK.json').exists():
        for b in P.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:P.verify_binding(b)
        print('X_ONLY_OUTPUTS_ALREADY_FROZEN',flush=True);return
    accepted,decisions={},{}
    for ds,rows in P.read(A.RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json').items():
        accepted[ds],decisions[ds]=[],[]
        for row in rows:
            before=json.dumps(row,sort_keys=True)
            result=inspect(A.N.top(row['prediction'])['keypoints_xy'])
            assert json.dumps(row,sort_keys=True)==before
            if result['keep']:accepted[ds].append(copy.deepcopy(row))
            decisions[ds].append(dict(id=row['id'],**result))
        print('X_ONLY',ds,len(accepted[ds]),'/',len(rows),flush=True)
    P.freeze(RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json',accepted)
    P.freeze(RAW/'DECISIONS.json',decisions)
    P.freeze(DOC/'OUTPUTS_LOCK.json',dict(complete=True,all179_decisions_before_GT_scoring=True,
        protocol=P.bound(DOC/'PROTOCOL.json'),artifacts=[P.bound(RAW/n) for n in ('ACCEPTED_UNCHANGED_PREDICTIONS.json','DECISIONS.json')]))


def stats(rows):
    result=A.N.subset_summary(rows);errors=[e for r in rows for e in r['errors']]
    result.update(correct20=sum(e<=20 for e in errors),PCK20=sum(e<=20 for e in errors)/len(errors) if errors else None,
        bad20_images=sum(any(e>20 for e in r['errors']) for r in rows))
    return result


def score():
    lock=P.read(DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']+[lock['protocol']]:P.verify_binding(b)
    for b in ('code','tests','parent_output','parent_results'):P.verify_binding(P.read(DOC/'PROTOCOL.json')[b])
    old=P.read(A.DOC/'RESULTS.json');P.verify_binding(old['metrics_source'])
    metrics=P.read(P.N.RAW/'PER_FRAME_METRICS.json');decisions=P.read(RAW/'DECISIONS.json')
    summaries={}
    for ds,mode in [('DEV72','DEV72'),('GREEN150','GREEN150_MANUAL')]:
        before={r['id'] for r in decisions[ds]};kept={r['id'] for r in decisions[ds] if r['keep']}
        rejected=before-kept;rows=metrics[mode]['POSEFIX_RAW']
        summaries[mode]=dict(before=stats([r for r in rows if r['id'] in before]),
            after=stats([r for r in rows if r['id'] in kept]),
            rejected=stats([r for r in rows if r['id'] in rejected]),
            rejected_ids=sorted(rejected),same_kept_N2=stats([r for r in metrics[mode]['A_N2'] if r['id'] in kept]))
    result=dict(complete=True,summary=summaries,output_lock=P.bound(DOC/'OUTPUTS_LOCK.json'),
        source_metrics=old['metrics_source'],coordinates_changed=0,self_training=False,
        conditional_subset_metrics=True,previous_physical_shape_filter_not_used=True)
    P.freeze(DOC/'RESULTS.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['apply','score'])
    args=parser.parse_args();apply() if args.stage=='apply' else score()
