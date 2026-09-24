"""Frozen-cache, fixed-identity visible 2D evaluation. No training/inference."""
from collections import Counter
import math
import numpy as np
from . import common as C
from .review_saved_keypoints import REVIEW
from .audit_completed_status import same_or_new

ARMS=('R0','OLD_S1','T0','T1','T2')
PAIRS=(('R0','OLD_S1'),('R0','T0'),('T0','T1'),('T1','T2'),('T0','T2'))

def metrics(values):
    e=np.array(values,dtype=float)
    return dict(n=len(e),median_px=float(np.median(e)) if len(e) else None,
        p90_px=float(np.percentile(e,90)) if len(e) else None,
        gt20=int(sum(e>20)),PCK={str(k):dict(correct=int(sum(e<=k)),total=len(e),
            fraction=float(np.mean(e<=k)) if len(e) else None) for k in (5,10,20)})

def point(pred, corner):
    idx=pred.get('selected_index')
    if idx is None:return None
    q=np.asarray(pred['candidates'][idx]['keypoints_xy'][corner],float)
    return q if q.shape==(2,) and np.isfinite(q).all() and not np.all(q==-1) else None

def difference(rows,left,right):
    a=metrics([r['errors'][left] for r in rows]);b=metrics([r['errors'][right] for r in rows])
    counts=Counter()
    for fid in {r['frame_id'] for r in rows}:
        rr=[r for r in rows if r['frame_id']==fid]
        delta=float(np.median([r['errors'][right] for r in rr])-np.median([r['errors'][left] for r in rr]))
        counts['win' if delta < -1e-9 else 'loss' if delta>1e-9 else 'tie']+=1
    return dict(right_minus_left=True,PCK10_correct_delta=b['PCK']['10']['correct']-a['PCK']['10']['correct'],
        median_error_delta_px=b['median_px']-a['median_px'] if rows else None,
        frame_median_win_loss_tie={k:counts[k] for k in ('win','loss','tie')})

def main():
    assert not (C.DOC/'VERIFIED_RESULTS.json').exists(), 'Preserve completed evaluation'
    lock=C.read(REVIEW/'FIRST_PASS_LOCK.json');qa=C.read(REVIEW/'QA_QUEUE.json')
    assert not qa['points'], 'Human second-pass QA is required before evaluating'
    assert C.sha(REVIEW/'LABELS.json')==lock['labels_sha256']
    labels=C.read(REVIEW/'FIRST_PASS_SNAPSHOT.json');selection=C.read(REVIEW/'ANCHOR_SELECTION.json')
    # Autosave and immutable snapshot differ only in JSON formatting/trailing newline.
    assert labels==C.read(REVIEW/'LABELS.json')
    verified=dict(reference='VERIFIED_VISIBLE_ANCHOR_V1',labels=labels,
        protocol='PNP_ASSISTED_KEYPOINTS_FIRST_THEN_STATUS',QA_required=0,QA_changed=0,
        unreviewed_points_excluded=True,not_independent_6D_GT=True,never_train=True)
    same_or_new(REVIEW/'VERIFIED_LABELS.json',verified)
    base=C.ROOT/'data/pallet/results/pallet_existing_data_transfer_v1'
    oldlock=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/PREDICTION_LOCK.json')
    for b in oldlock['files']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    same_or_new(C.DOC/'PREDICTIONS_LOCK.json',dict(files=oldlock['files'],
        verified_labels_sha256=C.sha(REVIEW/'VERIFIED_LABELS.json'),before_scoring=True,
        inference_runs=0,no_threshold_change=True,selected_index_unchanged=True))
    refs=C.read(base/'REFERENCE_PREDICTIONS.json');preds={a:refs[a] for a in ARMS[:2]}
    for a,name in zip(ARMS[2:],('T0_EASY_PSEUDO','T1_HARD_PSEUDO','T2_HARD_MANUAL')):
        preds[a]=C.read(base/f'HELDOUT_{name}.json')['predictions']
    legacy_path=C.ROOT/'data/pallet/results/pallet_replay_clean19_v1/TRUTH_FOR_DISPLAY_ONLY.json'
    assert C.sha(legacy_path)==qa['legacy_sha256'];legacy=C.read(legacy_path)
    rows=[]
    for fi,ci in labels['review_queue']:
        f=labels['frames'][fi];corner=f['corners'][ci];s=selection['frames'][fi]
        if corner['status']!='DIRECT_VISIBLE':continue
        assert corner['coordinate_source']=='manual_click' and C.valid_corner(corner,s['hw'])
        fid=f['frame_id'];g=np.array(corner['xy']);old=legacy[fid];oldq=np.array(old['gt'][ci]);oldvalid=bool(old['valid'][ci]) and np.isfinite(oldq).all()
        ld=float(np.linalg.norm(g-oldq)) if oldvalid else None
        errors={};old_errors={};coords={};missing={};category={}
        for a in ARMS:
            q=point(preds[a][fid],ci);missing[a]=q is None;coords[a]=q.tolist() if q is not None else None
            penalty=math.hypot(*s['hw'])
            errors[a]=float(np.linalg.norm(q-g)) if q is not None else penalty
            old_errors[a]=float(np.linalg.norm(q-oldq)) if q is not None and oldvalid else penalty if oldvalid else None
            e=errors[a]
            if ld is None:category[a]='LEGACY_UNAVAILABLE'
            elif ld<=10 and e<=10:category[a]='LEGACY_AND_MODEL_AGREE'
            elif ld<=10 and e>20:category[a]='MODEL_ERROR_CONFIRMED'
            elif ld>20 and e>20:category[a]='BOTH_DISAGREE'
            elif ld>20 and e<ld:category[a]='LEGACY_REFERENCE_DISAGREEMENT'
            else:category[a]='SMALL_DIFFERENCE'
        rows.append(dict(frame_id=fid,corner_id=ci,severity=s['severity'],recording=s['recording'],
            verified_xy=g.tolist(),legacy_xy=oldq.tolist() if oldvalid else None,legacy_distance=ld,
            errors=errors,legacy_errors=old_errors,model_xy=coords,missing=missing,categories=category))
    assert len(rows)==C.read(C.DOC/'STATUS_COVERAGE.json')['direct_visible']==66
    groups={'ALL':rows,**{s:[r for r in rows if r['severity']==s] for s in C.SEVERITIES}}
    groups.update({g:[r for r in rows if r['recording']==g] for g in sorted({r['recording'] for r in rows})})
    scores={g:{a:metrics([r['errors'][a] for r in rr]) for a in ARMS} for g,rr in groups.items()}
    baseline={g:{a:metrics([r['legacy_errors'][a] for r in rr if r['legacy_errors'][a] is not None]) for a in ARMS} for g,rr in groups.items()}
    pairs={g:{f'{b}-minus-{a}':difference(rr,a,b) for a,b in PAIRS} for g,rr in groups.items()}
    sensitivity={g:{f'{b}-minus-{a}':difference([r for r in rows if r['recording']!=g],a,b) for a,b in PAIRS} for g in sorted({r['recording'] for r in rows})}
    perframe={fid:{a:dict(visible_count=sum(r['frame_id']==fid for r in rows),
        mean_error=float(np.mean([r['errors'][a] for r in rows if r['frame_id']==fid])),
        median_error=float(np.median([r['errors'][a] for r in rows if r['frame_id']==fid]))) for a in ARMS} for fid in sorted({r['frame_id'] for r in rows})}
    percorner={str(i):{a:metrics([r['errors'][a] for r in rows if r['corner_id']==i]) for a in ARMS} for i in range(8)}
    agreement=metrics([r['legacy_distance'] for r in rows if r['legacy_distance'] is not None])
    categories={a:dict(Counter(r['categories'][a] for r in rows)) for a in ARMS}
    same_or_new(REVIEW/'SCORED_POINTS_PRIVATE.json',rows)
    same_or_new(REVIEW/'PER_FRAME_PRIVATE.json',perframe)
    same_or_new(C.DOC/'VERIFIED_RESULTS.json',dict(groups=scores,per_corner=percorner,
        fixed_identity=True,symmetry_remapping=False,missing_policy='native image diagonal penalty; no dropped points',
        legacy_reference_same_66_points=baseline,independent_test=False,operational_distribution=False))
    same_or_new(C.DOC/'MODEL_COMPARISON.json',dict(pairwise=pairs,leave_one_recording_out=sensitivity))
    same_or_new(C.DOC/'REFERENCE_DISAGREEMENT.json',dict(agreement=agreement,per_model_categories=categories,
        gt_error_automatically_declared=False,corner_identity_mismatch_candidates=0 if agreement['gt20']==0 else None,
        mismatch_rule='Only flags >20px disagreements; not an identity proof'))
    same_or_new(C.DOC/'VERIFIED_LABELS_PUBLIC_SUMMARY.json',dict(coverage=C.read(C.DOC/'STATUS_COVERAGE.json'),
        reference='VERIFIED_VISIBLE_ANCHOR_V1',QA_rechecks=0,QA_changed=0,exact_coordinates_private=True,
        extra_images=0,training_steps=0,protocol=verified['protocol']))
    print('EVALUATED',scores['ALL']);print('LEGACY_AGREEMENT',agreement);print('CATEGORIES',categories)

if __name__=='__main__':main()
