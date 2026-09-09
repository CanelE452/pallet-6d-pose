"""Paired descriptive errors and session-cluster uncertainty.

Canonical repository evaluation remains mandatory. These independent helpers
are useful cross-checks, not an automatic replacement of its GT/match contract.
"""
from __future__ import annotations
import numpy as np


def error_summary(pred,gt,supervised,pred_valid,matched,diagonal,*,corners_only=True):
    pred,gt=np.asarray(pred,float),np.asarray(gt,float)
    sup,valid=np.asarray(supervised,bool),np.asarray(pred_valid,bool)
    matched=np.asarray(matched,bool)
    diagonal=np.asarray(diagonal,float)
    if pred.shape!=gt.shape or pred.ndim!=3 or pred.shape[1:]!=(9,2):
        raise ValueError('Expected paired [N,9,2] coordinates')
    n=8 if corners_only else 9
    if np.any(sup[:,:n]&~np.isfinite(gt[:,:n]).all(-1)):
        raise ValueError('Supervised GT is nonfinite')
    observed=sup[:,:n]&valid[:,:n]&matched[:,None]&np.isfinite(pred[:,:n]).all(-1)
    err=np.linalg.norm(np.where(observed[...,None],pred[:,:n],0)-np.where(observed[...,None],gt[:,:n],0),axis=-1)
    pool=err[observed]
    count=observed.sum(1); frame_mask=count>0
    frame_mean=np.full(len(pred),np.nan)
    frame_mean[frame_mask]=err.sum(1)[frame_mask]/count[frame_mask]
    return dict(gt_points=int(sup[:,:n].sum()),observed_points=int(observed.sum()),missing_points=int((sup[:,:n]&~observed).sum()),
                observed_frames=int(frame_mask.sum()),mean_px=float(pool.mean()) if len(pool) else None,
                median_px=float(np.median(pool)) if len(pool) else None,p90_px=float(np.quantile(pool,.9)) if len(pool) else None,
                pck10_all_gt=float(((err<=10)&observed).sum()/max(1,sup[:,:n].sum())),
                errors=err,observed=observed,frame_mean_px=frame_mean,frame_normalized=frame_mean/diagonal)


def safety_counts(base_error,new_error,observed):
    b,n,m=np.asarray(base_error),np.asarray(new_error),np.asarray(observed,bool)
    good=m&(b<=10); hard=m&(b>20)
    return dict(good_count=int(good.sum()),good_damaged=int((good&(n>10)).sum()),
                good_damage_rate=float((good&(n>10)).sum()/max(1,good.sum())),hard_count=int(hard.sum()),
                hard_base_mean=float(b[hard].mean()) if hard.any() else None,
                hard_new_mean=float(n[hard].mean()) if hard.any() else None)


def paired_session_bootstrap(difference,session_ids,*,draws=20000,seed=20260909,alpha=.05):
    """Resample sessions, retaining ALL their paired frames and multiplicities.

    difference is already seed-averaged at matched frames. Result is conditional
    on those training seeds; no training-seed uncertainty is claimed.
    """
    values=np.asarray(difference,float); groups=np.asarray(session_ids,str)
    if values.ndim!=1 or groups.shape!=values.shape or not np.isfinite(values).all():
        raise ValueError('Supply finite, identically paired frame differences')
    names=np.unique(groups)
    if len(names)<2 or draws<100 or not 0<alpha<1:
        raise ValueError('At least two sessions and >=100 draws are required')
    sums=np.array([values[groups==g].sum() for g in names])
    counts=np.array([(groups==g).sum() for g in names])
    rng=np.random.default_rng(seed)
    # Chunking bounds memory, without changing RNG order.
    boot=[]
    for start in range(0,draws,1000):
        idx=rng.integers(0,len(names),(min(1000,draws-start),len(names)))
        boot.extend((sums[idx].sum(1)/counts[idx].sum(1)).tolist())
    lo,hi=np.quantile(boot,[alpha/2,1-alpha/2])
    return dict(mean=float(values.mean()),lower=float(lo),upper=float(hi),sessions=len(names),frames=len(values),
                draws=draws,seed=seed,alpha=alpha,unit='same as supplied difference',
                scope='paired-frame weighted, session resampling; fixed training seeds, reused DEV if applicable')
