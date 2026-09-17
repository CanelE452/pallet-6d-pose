"""GT-only evaluation: one branch and canonical-identity aligned damage."""
import math
import numpy as np

def measure(pred,gt,valid,perms,hw,matched=True,detected=True):
    p=np.array(pred,float);g=np.array(gt,float);v=np.array(valid,bool)&np.isfinite(g).all(-1)&~(g==-1).all(-1)
    if not v[:8].any():return dict(evaluable=False,matched=bool(matched),detected=bool(detected))
    perms=np.array(perms,int);diag=math.hypot(*hw);pv=np.isfinite(p).all(-1)&~(p==-1).all(-1)&matched
    tg=g[perms];mask=v[perms];diff=np.where(pv[None,:,None],p[None],0)-np.where(mask[...,None],tg,0)
    e=np.where(pv[None],np.linalg.norm(diff,axis=-1),diag);e=np.where(mask,e,0)
    mean=e[:,:8].sum(-1)/mask[:,:8].sum(-1);branch=int(np.argmin(mean));idx=perms[branch]
    canonical=np.full(9,np.nan);canonical[idx[mask[branch]]]=e[branch][mask[branch]]
    observed=mask[branch,:8]&pv[:8]
    return dict(evaluable=True,E_sym=float(mean[branch]/diag),E_fixed=float(mean[0]/diag),frame_mean_px=float(mean[branch]),branch=branch,
      errors=e[branch,:8][mask[branch,:8]].tolist(),observed_errors=e[branch,:8][observed].tolist(),
      canonical_errors=[None if not np.isfinite(x) else float(x) for x in canonical[:8]],
      canonical_valid=v[:8].tolist(),matched=bool(matched),detected=bool(detected),corners=int(v[:8].sum()),
      center_error=float(e[branch,8]) if v[8] else None)

def summary(rows):
    rr=[r for r in rows if r['evaluable']];e=np.array([x for r in rr for x in r['errors']]);obs=np.array([x for r in rr for x in r['observed_errors']])
    stat=lambda x,q:float(np.quantile(x,q)) if len(x) else None
    return dict(total_frames=len(rows),evaluable_frames=len(rr),non_evaluable_ids=[r['id'] for r in rows if not r['evaluable']],
      E_sym=float(np.mean([r['E_sym'] for r in rr])) if rr else None,E_fixed=float(np.mean([r['E_fixed'] for r in rr])) if rr else None,
      matched_pooled_corner8_median_px=stat(obs,.5),matched_pooled_corner8_P90_px=stat(obs,.9),
      full_penalty_median_px=stat(e,.5),full_penalty_P90_px=stat(e,.9),PCK={str(t):float((e<=t).mean()) if len(e) else None for t in [5,10,20]},
      gross20=float((e>20).mean()) if len(e) else None,detected=sum(r['detected'] for r in rows),matched=sum(r['matched'] for r in rows),
      missing=len(rows)-sum(r['detected'] for r in rows),coverage=sum(r['matched'] for r in rows)/len(rows) if rows else None,
      corners=len(e),observed_corners=len(obs))

def damage(base,new):
    assert [r['id'] for r in base]==[r['id'] for r in new]
    pairs=[(b,n) for b,n in zip(base,new) if b['evaluable'] and n['evaluable']]
    a=[];z=[]
    for b,n in pairs:
        assert b['canonical_valid']==n['canonical_valid']
        for i,valid in enumerate(b['canonical_valid']):
            if valid:a.append(b['canonical_errors'][i]);z.append(n['canonical_errors'][i])
    a=np.array(a);z=np.array(z);delta=np.array([n['frame_mean_px']-b['frame_mean_px'] for b,n in pairs])
    return dict(improved_frames=int((delta< -1e-9).sum()),harmed_frames=int((delta>1e-9).sum()),unchanged_frames=int((abs(delta)<=1e-9).sum()),
      good5_to_bad10=int(((a<5)&(z>10)).sum()),bad20_to_good10=int(((a>20)&(z<10)).sum()),
      reverse_good5_to_bad10=int(((z<5)&(a>10)).sum()),evaluation_branch_changed=sum(b['branch']!=n['branch'] for b,n in pairs),canonical_GT_identity_aligned=True)

def contrast(left,right,clusters=None):
    a=np.array(left,float);b=np.array(right,float);assert a.shape==b.shape and a.shape[0]==3
    d=(a-b).mean(0);rng=np.random.default_rng(20260917)
    _,g=np.unique(np.arange(len(d)) if clusters is None else clusters,return_inverse=True);n=int(g.max()+1)
    tot=np.bincount(g,weights=d,minlength=n);counts=np.bincount(g,minlength=n);samples=[]
    for _ in range(100):
        w=rng.multinomial(n,np.full(n,1/n),size=100);samples.extend(((w@tot)/(w@counts)).tolist())
    loso=[float((tot.sum()-tot[i])/(counts.sum()-counts[i])) for i in range(n)] if clusters is not None and n>1 else []
    return dict(delta=float(d.mean()),CI95=np.quantile(samples,[.025,.975]).tolist(),per_seed_delta=(a-b).mean(1).tolist(),
      improved_seeds=int(((a-b).mean(1)<0).sum()),units=n,resamples=10000,seed=20260917,
      LOSO=loso,level='frame' if clusters is None else 'cluster',multiplicity_adjusted=False)
