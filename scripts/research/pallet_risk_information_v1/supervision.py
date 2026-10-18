"""Synthetic-only continuous and partial-order supervision, no real I/O.

This module is imported by training/preparation only. Models and prediction
providers never import it. Original action metrics/labels stay unchanged.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import math
import numpy as np


def stable_error(pose, target):
    """Independent fsum/atan2 counterpart under each row's original C1/C2."""
    if not isinstance(pose,dict) or pose.get('available') is not True: return [math.nan,math.nan]
    try:
        R=np.asarray(pose['R_physical'],float);t=np.asarray(pose['centroid'],float)
        G=np.asarray(target['R_fixed'],float);g=np.asarray(target['t'],float)
        if R.shape!=(3,3) or G.shape!=(3,3) or t.shape!=(3,) or g.shape!=(3,): return [math.nan,math.nan]
        if not all(np.isfinite(a).all()for a in(R,t,G,g)): return [math.nan,math.nan]
        order=int(target['symmetry_order'])
        if order not in(1,2): raise ValueError('Original synthetic C1/C2 required')
        d=t-g;tc=100*math.sqrt(math.fsum(float(x)*float(x)for x in d))
        angles=[]
        for S in ([np.eye(3)]if order==1 else[np.eye(3),np.diag([-1.,1.,-1.])]):
            Q=(G@S).T@R;cosine=max(-1.,min(1.,(float(np.trace(Q))-1)/2))
            skew=[float(Q[2,1]-Q[1,2]),float(Q[0,2]-Q[2,0]),float(Q[1,0]-Q[0,1])]
            sine=.5*math.sqrt(math.fsum(x*x for x in skew))
            angles.append(math.degrees(math.atan2(sine,cosine)))
        return [tc,min(angles)]
    except (KeyError,TypeError,ArithmeticError): return [math.nan,math.nan]


def relation(errors, success, i, j):
    """-1=i better, +1=j better,0=no licensed preference."""
    if success[i]!=success[j]: return -1 if success[i] else 1
    if success[i]: return 0
    first,second=errors[i],errors[j]
    if np.all(first<=second) and np.any(first<second): return -1
    if np.all(second<=first) and np.any(second<first): return 1
    return 0


def prepare(entries, label, synthetic_target):
    """Original canonical H1 entries and exact archived per-action labels."""
    by_id={m['candidate_id']:m for m in label['action_metrics']}
    if len(by_id)!=len(label['action_metrics']): raise ValueError('Duplicate original action metric ID')
    if set(by_id)!={c['candidate_id']for c in entries}: raise ValueError('Original action supervision coverage changed')
    metrics=[by_id[c['candidate_id']]for c in entries]
    raw=np.asarray([[m['translation_cm']if m['translation_cm']is not None else math.nan,
        m['rotation_deg']if m['rotation_deg']is not None else math.nan]for m in metrics],dtype=np.float64).reshape(-1,2)
    stable=np.asarray([stable_error(c['actual_pose'],synthetic_target)for c in entries],dtype=np.float64).reshape(-1,2)
    success=np.asarray([m['success_5cm5deg']for m in metrics],dtype=bool)
    return from_errors(raw,stable,success,metrics=metrics)


def from_errors(raw, stable, success, *, metrics=None):
    raw=np.asarray(raw,float).reshape(-1,2);stable=np.asarray(stable,float).reshape(-1,2);success=np.asarray(success,bool)
    if stable.shape!=raw.shape or success.shape!=(len(raw),): raise ValueError('Supervision shapes disagree')
    validity=np.isfinite(raw)&(raw>=0);stable_valid=np.isfinite(stable)&(stable>=0)
    stable_success=stable_valid.all(-1)&(stable<=5).all(-1)
    target=np.zeros_like(raw)
    target[validity]=np.log1p(raw[validity]/5)
    pairs=[[],[]];counts=Counter();ambiguous=[]
    for i in range(len(raw)):
        for j in range(i+1,len(raw)):
            counts['total_pairs']+=1
            if success[i]and success[j]: counts['both_success_no_order']+=1;continue
            if not validity[[i,j]].all()or not stable_valid[[i,j]].all(): counts['unavailable_or_nonfinite_pair']+=1;continue
            original=relation(raw,success,i,j);alternative=relation(stable,stable_success,i,j)
            if original!=alternative:
                counts['NUMERICAL_ORDER_AMBIGUOUS']+=1;ambiguous.append((i,j,original,alternative));continue
            if not original:
                counts['incomparable_or_equal_failure_pair']+=1;continue
            category=0 if success[i]!=success[j] else 1
            pairs[category].append((i,j)if original<0 else(j,i))
    counts['success_failure_pairs']=len(pairs[0]);counts['failure_Pareto_pairs']=len(pairs[1])
    counts['usable_pairs']=sum(map(len,pairs));counts['actions']=len(raw)
    counts['positive_action_labels']=int(success.sum());counts['negative_action_labels']=len(success)-counts['positive_action_labels']
    counts['all_zero_frame']=int(len(raw)>0 and not success.any());counts['zero_action_frame']=int(len(raw)==0)
    counts['no_order_supervision_frame']=int(not counts['usable_pairs'])
    counts['all_zero_with_Pareto_supervision']=int(counts['all_zero_frame']and len(pairs[1])>0)
    counts['all_zero_without_order_supervision']=int(counts['all_zero_frame']and not counts['usable_pairs'])
    counts['finite_T_axes']=int(validity[:,0].sum());counts['finite_R_axes']=int(validity[:,1].sum())
    counts['missing_or_nonfinite_T_axes']=len(raw)-counts['finite_T_axes'];counts['missing_or_nonfinite_R_axes']=len(raw)-counts['finite_R_axes']
    return dict(raw_errors=raw,stable_errors=stable,original_success=success,continuous_target=target,
        continuous_validity=validity,pair_groups=tuple(np.asarray(p,dtype=np.int64).reshape(-1,2)for p in pairs),
        numerical_ambiguous_pairs=ambiguous,counts=dict(counts),metrics=metrics)


def sampled_pairs(supervision,seed,epoch,frame_number,*,training=True,limit=256):
    """Uniform fixed-order samples, independent of arm, scores and ID strings."""
    selected=[]
    for category,all_pairs in enumerate(supervision['pair_groups']):
        if training and len(all_pairs)>limit:
            rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(epoch),int(frame_number),category]))
            indices=np.sort(rng.choice(len(all_pairs),size=limit,replace=False));selected.append(all_pairs[indices])
        else:selected.append(all_pairs)
    return tuple(selected)


def pair_hash(groups):
    digest=hashlib.sha256()
    for values in groups:
        a=np.asarray(values,dtype='<i8').reshape(-1,2);digest.update(len(a).to_bytes(8,'little'));digest.update(a.tobytes())
    return digest.hexdigest()
