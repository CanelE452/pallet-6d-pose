"""Fixed prediction-only S0/S1/S2 rules; no S3 reference access here."""
import numpy as np
from . import solver
from scripts.research.pallet_vispnp_square6d_20261011.visibility import visibility

def select_rule(rule,q,K,xyz,source=False,support=None):
    if rule=='S0':return solver.production(q,K,xyz,source,support)
    q=np.asarray(q,float)
    usable=np.isfinite(q[:8]).all(-1)&~np.all(q[:8]==-1,axis=1)
    if support is not None:usable&=np.asarray(support,bool)[:8]
    diagnostic={}
    if rule=='S1':
        vis=visibility(q,usable)
        visible=np.asarray(vis['visible_mask'],bool)&usable
        diagnostic['visibility']=vis
        if int(visible.sum())<6:
            result=solver.production(q,K,xyz,source,support)
            result.update(rule='S1',fallback=True,fallback_reason='VISIBLE_SUPPORTED_CORNERS_LT6',**diagnostic)
            return result
        subsets=[np.flatnonzero(visible).tolist()]
    elif rule=='S2':subsets=[list(range(8))]+[[i for i in range(8) if i!=j] for j in range(8)]
    else:raise ValueError('Only fixed S0/S1/S2 are prediction-only')
    candidates=[]
    for si,subset in enumerate(subsets):
        for name in solver.NAMES:
            c=solver.fit(q,K,xyz,name,subset,source)
            c.update(hypothesis=name,subset_number=si)
            candidates.append(c)
    valid=[(c['subset_center_rmse_px'],i,c) for i,c in enumerate(candidates)
           if c['success'] and c['subset_center_rmse_px'] is not None]
    chosen=min(valid,key=lambda t:(t[0],t[1]))[2] if valid else None
    return dict(rule=rule,actual_pose=dict(available=False) if chosen is None else chosen['actual_pose'],
                hyp=None if chosen is None else chosen['hypothesis'],
                selected_hypothesis=None if chosen is None else chosen['hypothesis'],
                candidates=candidates,candidate_scores=[c['subset_center_rmse_px'] for c in candidates],
                subset_indices=[] if chosen is None else chosen['subset_indices'],fallback=False,
                status='NO_POSE' if chosen is None else 'AVAILABLE',reference_inputs=False,**diagnostic)
