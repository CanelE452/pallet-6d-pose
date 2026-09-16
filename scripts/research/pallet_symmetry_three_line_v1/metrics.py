"""Single global correspondence, full denominator, paired seed-mean bootstrap."""
import math
import numpy as np
from audit_math import symmetry_error

def measure(pred,gt,valid,perms,hw,pred_valid=None):
    diagonal=math.hypot(*hw);s=symmetry_error(pred,gt,valid,perms,diagonal,pred_valid)
    if not s['evaluable']:return dict(evaluable=False)
    errors=s['errors'][:8][s['gt_mask'][:8]]
    center=float(s['errors'][8]) if s['gt_mask'][8] else None
    return dict(evaluable=True,E_sym=s['equivalent_mean']/diagonal,E_fixed=s['fixed_mean']/diagonal,
      frame_mean_px=s['equivalent_mean'],fixed_frame_mean_px=s['fixed_mean'],branch=s['branch'],
      errors=errors.tolist(),n_corners=s['n_corners'],center_error_px=center,
      rescue=s['fixed_mean']>20 and s['equivalent_mean']<5,true_collapse=s['equivalent_mean']>20,
      gt_mask=s['gt_mask'].tolist(),target=np.where(np.isfinite(s['target']),s['target'],None).tolist())
def summarize(rows):
    rows=[r for r in rows if r['evaluable']];errors=np.concatenate([r['errors'] for r in rows])
    means=np.array([r['frame_mean_px'] for r in rows]);center=[r['center_error_px'] for r in rows if r['center_error_px'] is not None]
    return dict(frames=len(rows),annotated_corners=len(errors),primary_E_sym=float(np.mean([r['E_sym'] for r in rows])),
      E_fixed=float(np.mean([r['E_fixed'] for r in rows])),pooled_point_median_px=float(np.median(errors)),
      pooled_point_P90_px=float(np.quantile(errors,.9)),frame_mean_median_px=float(np.median(means)),
      frame_mean_P90_px=float(np.quantile(means,.9)),PCK={str(t):float((errors<=t).mean()) for t in [5,10,20]},
      gross20_point_rate=float((errors>20).mean()),true_collapse_frames=int((means>20).sum()),
      rescue_frames=sum(r['rescue'] for r in rows),center_median_px=float(np.median(center)) if center else None)
def contrast(left,right,clusters=None):
    # Arrays [training_seed, frame]; same frame/cluster draw shared by all seeds/arms.
    left=np.asarray(left,float);right=np.asarray(right,float);assert left.shape==right.shape
    delta=(left-right).mean(0);n=len(delta);rng=np.random.default_rng(20260916)
    if clusters is None:group=np.arange(n);level='frame';limitation='No verified independent synthetic scenario grouping; frame CI may understate dependence.'
    else:_,group=np.unique(clusters,return_inverse=True);level='session';limitation='Reused DEVELOPMENT, not confirmatory.'
    size=int(group.max())+1;tot=np.bincount(group,weights=delta,minlength=size);count=np.bincount(group,minlength=size)
    draws=[]
    for _ in range(100):
        weights=rng.multinomial(size,np.full(size,1/size),size=100)
        draws.extend(((weights@tot)/(weights@count)).tolist())
    return dict(delta=float(delta.mean()),CI95=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
      per_seed_delta=(left-right).mean(1).tolist(),improved_seeds=int(((left-right).mean(1)<0).sum()),
      seed=20260916,resamples=10000,level=level,units=size,frames=n,limitation=limitation,
      estimator='mean_seed(mean_frame E_sym difference)')
