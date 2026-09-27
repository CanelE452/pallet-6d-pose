"""Native TRAIN fit and same-point reference decomposition; no optimization."""
from collections import Counter
import csv
import numpy as np
from scripts.research.pallet_verified_anchor_v1.evaluate import metrics, point
from . import common as C

def stats(x):
    x=np.asarray(x,float)
    return dict(metrics(x),mean_px=float(x.mean()) if len(x) else None)

def paired(rows,left='RAW_LR5',right='REF_LR5',key='errors'):
    a=np.array([r[key][left] for r in rows]);b=np.array([r[key][right] for r in rows])
    return dict(n=len(a),gains10=int(((a>10)&(b<=10)).sum()),losses10=int(((a<=10)&(b>10)).sum()),
        correct_delta10=int((b<=10).sum()-(a<=10).sum()),mean_error_delta=float((b-a).mean()) if len(a) else None,
        median_paired_delta=float(np.median(b-a)) if len(a) else None,
        continuous_improve=int((b<a-1e-8).sum()),continuous_worsen=int((b>a+1e-8).sum()),
        teacher_student={s:dict(teacher_only=sum(r['errors']['TEACHER']<=10<r['errors'][s] for r in rows),
            student_only=sum(r['errors'][s]<=10<r['errors']['TEACHER'] for r in rows),
            both_correct=sum(r['errors'][s]<=10 and r['errors']['TEACHER']<=10 for r in rows),
            both_wrong=sum(r['errors'][s]>10 and r['errors']['TEACHER']>10 for r in rows)) for s in (left,right)})

def evaluate():
    P=C.P;rows=C.read(P.RAW/'ANCHOR_POINTS.json');pred=C.read(P.RAW/'PREDICTIONS.json');truth=C.read(P.TRUTH)
    arms=list(rows[0]['errors']);groups={'ALL':rows}
    for key in ('severity','recording','corner_id'):
        groups.update({f'{key}:{g}':[r for r in rows if r[key]==g] for g in sorted({r[key] for r in rows})})
    for r in rows:
        old=truth[r['frame_id']];ci=r['corner_id'];g=np.array(old['gt'][ci]);assert old['valid'][ci] and np.isfinite(g).all()
        r['legacy_errors']={a:float(np.linalg.norm(point(pred[a][r['frame_id']],ci)-g)) if point(pred[a][r['frame_id']],ci) is not None else float(np.hypot(*old['hw'])) for a in arms}
        r['reference_distance']=float(np.linalg.norm(g-np.array(r['verified_xy'])))
    results=dict(groups={g:{a:stats([r['errors'][a] for r in rr]) for a in arms} for g,rr in groups.items()},
        paired={g:paired(rr) for g,rr in groups.items()},legacy_same66={a:stats([r['legacy_errors'][a] for r in rows]) for a in arms},
        legacy_paired_same66=paired(rows,key='legacy_errors'),reference_displacement=stats([r['reference_distance'] for r in rows]),
        reference_per_recording={g:stats([r['reference_distance'] for r in rr]) for g,rr in groups.items() if g.startswith('recording:')},
        sensitivity={s:paired(rows,f'RAW_{s}',f'REF_{s}') for s in P.SUFFIXES},
        leave_one_recording_out={g:paired([r for r in rows if r['recording']!=g]) for g in sorted({r['recording'] for r in rows})},
        protocol='All66 fixed IDs/native coordinates, no IoU gate/no symmetry; same prediction and image-diagonal missing penalty for both references. Points clustered in16 images; no independent-point test.',
        full128_original=C.read(P.DOC/'CORE_RESULTS.json')['groups']['ALL'],
        full128_harmonized_fixedID_no_matching={},sample_composition_note='Full128 original uses support/matching/symmetry contract; compare harmonized full128 legacy vs same66 legacy for sample composition, then same66 legacy vs verified for reference. No causal attribution from aggregate gap.')
    for a in C.ARMS:
        errors=[]
        for rr in P.records():
            fid=rr['id'];g=truth[fid]
            for ci in range(8):
                if not g['valid'][ci]:continue
                q=point(pred[a][fid],ci);errors.append(float(np.linalg.norm(q-np.array(g['gt'][ci]))) if q is not None else float(np.hypot(*g['hw'])))
        results['full128_harmonized_fixedID_no_matching'][a]=stats(errors)
    assert len(rows)==66 and len({r['frame_id'] for r in rows})==16
    assert results['paired']['severity:CLEAN']['correct_delta10']==2
    assert results['paired']['severity:MODERATE_OCCLUSION']['correct_delta10']==-2
    assert results['paired']['severity:SEVERE_OCCLUSION']['correct_delta10']==0
    C.save(C.RAW/'REFERENCE_POINTS_PRIVATE.json',rows)
    C.save(C.DOC/'REFERENCE_AND_THRESHOLD_SENSITIVITY.json',results)
    public=[{k:r[k] for k in ('frame_id','corner_id','severity','recording','errors','legacy_errors','missing','reference_distance')} for r in rows]
    C.save(C.DOC/'PAIRED_66_ROWS.json',public)
    return results

def aggregate_train(rows):
    out={'corners':{},'center':{}}
    for subset,dst in out.items():
        rr=[r for r in rows if (r['corner']<8)==(subset=='corners')]
        if not rr:continue
        nc=Counter(r['id'] for r in rr)
        weights={'unique_image_uniform':np.array([1/nc[r['id']] for r in rr]),
                 'actual_occurrence_image_weighted':np.array([r['occurrences']/nc[r['id']] for r in rr]),
                 'point_uniform':np.ones(len(rr))}
        dst['points']=len(rr);dst['unique_images']=len(nc)
        dst['residuals']={a:{t:dict(stats([r['residuals'][a][t] for r in rr]),
            weighted_mean={w:float(np.average([r['residuals'][a][t] for r in rr],weights=v)) for w,v in weights.items()}) for t in ('raw','ref')} for a in C.ARMS}
        dst['correction']=stats([r['correction'] for r in rr]);dst['student_change']=stats([r['student_change'] for r in rr])
        directional=[r for r in rr if r['cosine'] is not None]
        dst['direction']={'u_min_px':1,'v_min_px':1e-8,'n':len(directional),
            'aligned_fraction':float(np.mean([r['cosine']>0 for r in directional])) if directional else None,
            'cosine_median':float(np.median([r['cosine'] for r in directional])) if directional else None,
            'projection_fraction_median':float(np.median([r['projection'] for r in directional])) if directional else None,
            'not_a_success_score':True}
    return out

def train():
    for b in C.read(C.DOC/'TRAIN_PREDICTIONS_LOCK.json')['files']:C.verify(b)
    rows=C.read(C.RAW/'TRAIN_TARGETS_PRIVATE.json');ps={a:C.read(C.RAW/f'TRAIN_PREDICTIONS_{a}.json')['predictions'] for a in C.ARMS};points=[]
    areas=[]
    for r in rows:
        b=r['predicted_box'];areas.append(max(0,b[2]-b[0])*max(0,b[3]-b[1]))
    boundaries=np.quantile(areas,[1/3,2/3]);missing={a:0 for a in C.ARMS}
    for r,area in zip(rows,areas):
        pred={a:C.P.selected(ps[a][r['id']]) for a in C.ARMS}
        for a in C.ARMS:missing[a]+=pred[a] is None
        assert all(pred.values()),'Missing TRAIN prediction: report coverage rather than silently drop'
        for ci,valid in enumerate(r['common_support']):
            if not valid:continue
            raw=np.array(r['raw_target'][ci]);ref=np.array(r['ref_target'][ci]);p={a:np.array(pred[a]['keypoints_xy'][ci]) for a in C.ARMS}
            u=ref-raw;v=p['REF_LR5']-p['RAW_LR5'];un=float(np.linalg.norm(u));vn=float(np.linalg.norm(v));stable=un>=1 and vn>1e-8
            points.append(dict(id=r['id'],corner=ci,recording=r['recording'],occurrences=r['occurrences_per_epoch'],support_count=sum(r['common_support']),
                bbox_area=area,bbox_tertile=int(np.searchsorted(boundaries,area)),correction=un,student_change=vn,
                cosine=float(np.dot(u,v)/(un*vn)) if stable else None,projection=float(np.dot(u,v)/un**2) if stable else None,
                residuals={a:{t:float(np.linalg.norm(p[a]-g)) for t,g in [('raw',raw),('ref',ref)]} for a in C.ARMS}))
    result=dict(all=aggregate_train(points),unique_images=len(rows),real_slots_per_epoch=sum(r['occurrences_per_epoch'] for r in rows),
        multiplicity_histogram=dict(Counter(r['occurrences_per_epoch'] for r in rows)),missing=missing,
        point_support_per_image=dict(Counter(sum(r['common_support']) for r in rows)),
        total_corner_exposures_5epochs=sum(r['occurrences']*5 for r in points if r['corner']<8),
        grouped={},bbox_tertile_boundaries_px2=boundaries,
        definitions={'unique_image_uniform':'average per-image mean over common valid support, each unique image once',
            'actual_occurrence_image_weighted':'same per-image mean weighted by occurrence in1024-slot list;5epoch factor cancels',
            'point_uniform':'each supported native point once; per-point pooled median/P90',
            'accuracy':'PSEUDO_TARGET_FOLLOWING_NOT_GT_ACCURACY','inputs':'Native/unaugmented, highest confidence; no target matching; no augmented tensor cache'},curves={})
    for k in ('corner','recording','bbox_tertile','occurrences','support_count'):
        result['grouped'][k]={str(g):aggregate_train([r for r in points if r[k]==g]) for g in sorted({r[k] for r in points})}
    for a in C.ARMS[1:]:
        f=C.read(C.P.REC/'pose_only'/f'FIT_{a}.json');rr=list(csv.DictReader((C.ROOT/f['results_csv']['path']).open()))
        rr=[{k.strip():float(v) for k,v in r.items()} for r in rr]
        result['curves'][a]=rr
    result['loss_interpretation']=['CSV losses mix real/source and are not pixel residuals; location is area/support-normalized, RLE and visibility have different units/reduction.',
        'Real/source use identical implementation; equal RGB slots do not establish equal effective loss weights.',
        'Coordinate pose loss falls overall and in last interval, but rises at epoch3: NOT monotonic.',
        'Native residual does not prove insufficient updates: augmentation, contradictory pseudo labels, frozen representation and mixed replay compete.',
        'v1 true-ignore applies after augmentation; an out-of-frame transform can relabel visibility v0. No claim of full augmented-tensor parity.']
    C.save(C.DOC/'TRAIN_TARGET_TRANSFER.json',result);C.save(C.DOC/'TRAIN_POINT_RESIDUALS.json',points)
    return result

def main():
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    e=evaluate();t=train()
    print('VERIFIED_PAIRED',e['paired']['ALL']);print('TRAIN',t['all']['corners'])

if __name__=='__main__':main()
