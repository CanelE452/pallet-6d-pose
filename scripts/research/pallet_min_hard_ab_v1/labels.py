"""Human-only label validation; no model/PnP completion."""
from collections import Counter
import math
from . import common as C

STATUSES=('DIRECT_VISIBLE','OCCLUDED','OUT_OF_FRAME','UNCERTAIN')

def validate_frame(frame):
    errors=[];warnings=[];role=frame.get('role')
    if role not in ('ROLE_CONFIDENT','ROLE_UNCERTAIN'):
        return ['role missing'],[]
    if role=='ROLE_UNCERTAIN':
        if any(p.get('xy') is not None for p in frame['corners']):errors.append('uncertain role must have no primary xy')
        return errors,['ROLE_UNCERTAIN']
    w,h=frame['size'];box=frame.get('bbox')
    if not box or len(box)!=4 or not all(math.isfinite(x) for x in box) or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
        errors.append('invalid manual bbox')
    points={}
    for k,p in enumerate(frame['corners']):
        if p.get('status') not in STATUSES:errors.append(f'P{k} status missing');continue
        xy=p.get('xy')
        if p['status']=='DIRECT_VISIBLE':
            if xy is None or len(xy)!=2 or not all(math.isfinite(x) for x in xy) or not(0<=xy[0]<w and 0<=xy[1]<h):
                errors.append(f'P{k} invalid visible xy')
            else:points[k]=xy
        elif xy is not None:errors.append(f'P{k} nonvisible xy must be None')
    for a,p in points.items():
        for b,q in points.items():
            if a<b and math.dist(p,q)<=2:warnings.append(f'P{a}/P{b} near duplicate <=2px')
    for left,right in [(0,1),(3,2),(4,5),(7,6)]:
        if left in points and right in points and points[left][0]>=points[right][0]:warnings.append(f'LR role ordering P{left}/P{right}')
    for top,bottom in [(0,3),(1,2),(4,7),(5,6)]:
        if top in points and bottom in points and points[top][1]>=points[bottom][1]:warnings.append(f'TB role ordering P{top}/P{bottom}')
    if frame.get('role_note','').strip():warnings.append('role note requires own-annotation QA')
    return errors,warnings

def coverage(frames,selection):
    lookup={r['frame_id']:r for r in selection};counts=Counter();recs=set();usable=[]
    for fid,f in frames.items():
        if not f.get('complete') or f['role']!='ROLE_CONFIDENT':continue
        visible=[i for i,p in enumerate(f['corners']) if p['status']=='DIRECT_VISIBLE']
        if len(visible)<2:continue
        usable.append(fid);counts.update(visible);recs.add(lookup[fid]['recording'])
    passed=len(usable)>=6 and sum(counts.values())>=24 and sum(n>=3 for n in counts.values())>=3 and len(recs)>=3
    return dict(usable_frames=len(usable),direct_visible_clicks=sum(counts.values()),
                corner_counts={str(i):counts[i] for i in range(8)},recordings=len(recs),passed=passed)

def validate_resume():
    with C.exclusive('annotating'):
        state=C.state();lock=C.read(C.DOC/'HARD_SELECTION_LOCK.json');C.verify(lock['private_selection'])
        selected=C.read(C.RAW/'HARD_SELECTION_PRIVATE.json')['rows'];active=selected[:state['active_frames']]
        path=C.annotation_labels_path()
        if not path.exists():print('WAITING_FOR_HUMAN_HARD_ANNOTATION: no labels');return
        data=C.read(path);assert data['selection_sha256']==C.sha(C.DOC/'HARD_SELECTION_LOCK.json')
        assert set(data['frames'])<={r['frame_id'] for r in active}
        frames=data['frames'];missing=[];qa=[]
        for r in active:
            C.verify(r['image']);f=frames.get(r['frame_id'])
            if f is None or not f.get('complete'):missing.append(r['frame_id']);continue
            assert f['image_sha256']==r['image']['sha256']
            errors,warnings=validate_frame(f)
            assert not errors,(r['frame_id'],errors)
            if warnings and not f.get('qa_confirmed'):qa.append(dict(frame_id=r['frame_id'],warnings=warnings))
        if missing:print('WAITING_FOR_HUMAN_HARD_ANNOTATION:',len(missing),'unfinished');return
        if qa:
            C.save(C.RAW/'QA_TODO_PRIVATE.json',qa)
            C.set_state('WAITING_FOR_HUMAN_HARD_QA',active_frames=len(active),reserve=len(selected)-len(active),
                        command='python -m scripts.research.pallet_min_hard_ab_v1.annotate_hard',training='NOT_RUN')
            print('QA requires raw RGB + own clicks only:',len(qa));return
        cov=coverage(frames,selected)
        C.save(C.DOC/'HARD_COVERAGE_PUBLIC.json',cov)
        if not cov['passed'] and len(active)<len(selected):
            C.set_state('WAITING_FOR_HUMAN_HARD_ANNOTATION',active_frames=len(active)+1,reserve=len(selected)-len(active)-1,
                        command='python -m scripts.research.pallet_min_hard_ab_v1.annotate_hard',training='NOT_RUN')
            print('Coverage insufficient; activate one fixed-order reserve.');return
        if not cov['passed']:
            C.set_state('MIN_HARD_LABEL_COVERAGE_LIMITED',active_frames=len(active),training='NOT_RUN');return
        if not (C.DOC/'HARD_LABEL_LOCK.json').exists():
            C.save(C.RAW/'QA_HISTORY_PRIVATE.json',dict(events=[r for r in data['history'] if r['action']=='SECOND_PASS_QA']),immutable=True)
            C.save(C.DOC/'HARD_PROVENANCE_PUBLIC.json',dict(coverage=cov,source='human raw RGB only',
                   bbox='tight observed pallet envelope; association only',hidden_completion=False,
                   P8_supervised=False,role_uncertain_excluded=True,model_outputs_opened_before_lock=0),immutable=True)
            C.save(C.DOC/'HARD_LABEL_LOCK.json',dict(labels=C.bind(path),selection=C.bind(C.DOC/'HARD_SELECTION_LOCK.json'),
                   coverage=cov,created_at=C.now(),model_outputs_opened=0),immutable=True)
        C.set_state('HARD_LABELS_LOCKED_TRAINING_PENDING',training='NOT_RUN',
                    next='Matched-budget teacher/train/evaluate stage must pass pair-integrity checks before execution.')
