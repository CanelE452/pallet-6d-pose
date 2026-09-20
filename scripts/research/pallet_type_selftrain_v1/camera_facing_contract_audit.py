"""Read-only empirical audit of the legacy front-face rule; no relabeling."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from . import recovery_common as R
from . import identity_rank as I
from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4,polyarea

C=R.C
PHASE='camera_facing_contract_audit'
DOC=R.DOC/PHASE;RAW=R.RAW/PHASE
# Topology only, not estimated pose, measured dimensions or renderer GT origin.
ORIGIN=np.array([[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,-1],
                 [-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]],float)
FACES=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]]
HALF=[5,4,7,6,1,0,3,2,8]


def observe(points,valid):
    q=np.asarray(points,float)[:8];v=np.asarray(valid,bool)[:8]
    if q.shape!=(8,2) or v.shape!=(8,):raise ValueError('Expected eight corners')
    complete=bool(v.all() and np.isfinite(q).all() and not (q==-1).all(-1).any())
    if not complete:return dict(complete=False)
    p=compute_perm_v4(ORIGIN,q)
    if p is None:return dict(complete=True,rule_available=False)
    a=np.array([polyarea(q[f]) for f in FACES]);scale=max(float(np.linalg.norm(np.ptp(q,axis=0)))**2,1e-12)
    lr=[float(q[b,0]-q[a,0]) for a,b in [(0,1),(3,2),(4,5),(7,6)]]
    diff=abs(a[0]-a[1])-abs(a[2]-a[3])
    return dict(complete=True,rule_available=True,legacy_perm=p,legacy_identity=p==list(range(9)),
        legacy_in_approved_C2=p in [list(range(9)),HALF],
        legacy_same_axis=set(p[:4]) in [{0,1,2,3},{4,5,6,7}],
        front_vs_rear_area=float((a[0]-a[1])/scale),front_axis_advantage=float(diff/scale),
        min_LR_margin=float(min(lr)/np.sqrt(scale)),nonpositive_LR=sum(x<=0 for x in lr))


def summarize(rows):
    available=[r for r in rows if r['observation']['complete'] and r['observation'].get('rule_available')]
    n=len(available)
    result=dict(total=len(rows),complete=n,incomplete_or_unavailable=len(rows)-n)
    for key in ['legacy_identity','legacy_in_approved_C2','legacy_same_axis']:
        result[key]=dict(agree=sum(r['observation'][key] for r in available),denominator=n,
            fraction=float(sum(r['observation'][key] for r in available)/n) if n else None)
    result['nonpositive_LR_frames']=sum(r['observation']['nonpositive_LR']>0 for r in available)
    result['permutation_counts']={str(k):v for k,v in Counter(tuple(r['observation']['legacy_perm']) for r in available).most_common()}
    return result


def main():
    data=I.SourceData().data
    paths=[__file__,Path(__file__).with_name('test_camera_facing_contract_audit.py'),
        C.ROOT/'scripts/annotate/convert_to_camera_facing_v4.py',
        C.ROOT/'scripts/research/pallet_dht_structured_v2/SEMANTICS.md',
        C.ROOT/'scripts/research/pallet_dht_structured_v2/geometry_semantics.py',
        data.directory/'CACHE_MANIFEST.json',data.directory/'CACHE_COMPLETE.json',data.run_dir/'SOURCE_MANIFEST.json',
        R.BASE_DOC/'EVAL_PROTOCOL.json',R.BASE_RAW/'EVAL_PREDICTIONS_R0.json',R.BASE_RAW/'EVAL_METRICS.json',
        R.DOC/'hint_dropout/RESULTS.json']
    protocol=dict(status='READ_ONLY_TARGET_CONVENTION_AUDIT_NOT_CORRECTION',
        question='Does the local legacy area-based camera-facing converter agree with actual source/evaluation reference indexing? Its existence alone is not proof of current renderer provenance.',
        scope='All60000source label records plus all194ordinaryPLASTICeval references. Only fully supervised finite8-corner layouts support the area rule;report incomplete denominator explicitly. No unknown/PnP corner filled,labels changed,frames dropped or GT reindexed.',
        rule='Unmodified historical compute_perm_v4 with fixed cuboid TOPOLOGY (top/bottom/vertical pairing),actual2Dpoints. Compare exactidentity,approvedC2,andfront-axis agreement;do not treat C4 as equivalent.',
        geometry_limit='No new inference geometry filter,CAD,depth,pose fitting,model update or threshold selection. This is a diagnostic,not a deployable canonicalizer or a declaration that one label set is wrong.',
        labels_for_audit_only=True,new_annotations=0,new_tags=0,model_changes=0,
        sources=[C.bound(p) for p in paths])
    C.freeze(DOC/'PROTOCOL.json',protocol)
    source=[];buckets=defaultdict(list)
    for idx,record in enumerate(data.source['records']):
        target=record['targets'];assert len(target)==1
        q=np.asarray(target[0]['keypoints_normalized'],float)
        xy=q[:,:2]*np.asarray(record['prepared_shape_hw'][::-1])
        row=dict(id=record['id'],source=record['source'],partition=record['partition'],
                 observation=observe(xy,q[:,2]!=0))
        source.append(row);buckets[f'{record["source"]}/{record["partition"]}'].append(row)
        if (idx+1)%10000==0:print('CF_LABEL_AUDIT',idx+1,'/60000',flush=True)
    assert len(source)==60000
    C.freeze(RAW/'SOURCE_ROWS.json',source)
    source_summary={key:summarize(rows) for key,rows in buckets.items()}
    C.freeze(DOC/'SOURCE_SUMMARY.json',dict(all=summarize(source),by_source_partition=source_summary,
        source_rows=C.bound(RAW/'SOURCE_ROWS.json')))
    # Prediction observations are fixed without any real GT coordinates.
    pred={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    predictions=[]
    for key,r in pred.items():
        q=np.asarray(I.top(r['prediction'])['keypoints_xy'])
        predictions.append(dict(id=key,observation=observe(q,np.isfinite(q).all(-1)&~(q==-1).all(-1))))
    C.freeze(RAW/'REAL_PREDICTION_OBSERVATIONS.json',dict(rows=predictions,GT_free=True))
    metadata={r['id']:r for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records'] if r['kind']=='PLASTIC'}
    pe,pop=R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in metadata}
    baseline={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    references=[]
    for key in pred:
        C.verify(metadata[key]['annotation']);t=targets[key];b=baseline[key]
        references.append(dict(id=key,observation=observe(t.keypoints_xy,t.keypoint_supervision_mask),
            R0_frame_mean_px=b['frame_mean_px'],R0_matched=b['matched'],
            R0_hard100_corners=sum(e is not None and e>100 for e in b['canonical_errors']) if b['matched'] else None))
    pmap={r['id']:r['observation'] for r in predictions}
    cross=Counter()
    for r in references:
        a=r['observation'];b=pmap[r['id']]
        if a.get('rule_available') and b.get('rule_available'):
            cross[(a['legacy_identity'],b['legacy_identity'])]+=1
    result=dict(status='DIAGNOSTIC_ONLY_NO_TARGET_OR_PREDICTION_CHANGE',source=summarize(source),
        source_by_partition=source_summary,real_reference=summarize(references),real_R0=summarize(predictions),
        real_cross_counts={f'reference_identity={a},prediction_identity={b}':n for (a,b),n in cross.items()},
        real_reference_rows=references,new_labels=0,new_model_outputs=0,goal_complete=False,
        warning='Disagreement identifies unsupported assumptions about the old rule,not which target is physically correct. Do not canonicalize current labels/predictions or enlarge symmetry using this audit.',
        evidence=[C.bound(DOC/'PROTOCOL.json'),C.bound(DOC/'SOURCE_SUMMARY.json'),C.bound(RAW/'REAL_PREDICTION_OBSERVATIONS.json')])
    C.freeze(DOC/'RESULTS.json',result)
    for name in ['source','real_reference','real_R0','real_cross_counts']:print('CF_AUDIT',name,result[name],flush=True)


if __name__=='__main__':main()
