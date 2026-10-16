"""Prediction-only mask and pose/reprojection; scoring has a separate module."""
import numpy as np
from . import common as C

def hidden_mask(initial):
    if not initial.get('available'):return [],dict(available=False,reason='initial_pose_unavailable')
    from .solver import cuboid
    x=cuboid(*initial['cf_extents']);R=np.asarray(initial['R_cf']);t=np.asarray(initial['centroid'])
    rays=(-R.T@t)[None,:]-x
    cosine=(np.sign(x)*rays/np.linalg.norm(rays,axis=1)[:,None]).max(1)
    hidden=np.flatnonzero(cosine < -np.sin(np.deg2rad(2.))).tolist()
    return hidden,dict(available=True,cosine=cosine,margin_deg=2.,geometry='convex cuboid proxy; not actual mesh visibility')

def finish(bank,original,initial,excluded=(),hidden=(),robust=True):
    solved=bank.solve(excluded=excluded,hidden=hidden,robust=robust)
    new=bool(solved['available']);fallback=(not new and initial.get('available',False))
    final=solved if new else initial
    out=np.asarray(original,float).copy()
    if new and hidden:out[list(hidden)]=np.asarray(solved['projected'])[list(hidden)]
    assert np.array_equal(out[8],np.asarray(original)[8],equal_nan=True)
    after,info=hidden_mask(final) if new else ([],{})
    if new:
        assert not set(hidden)&set(solved.get('used',solved.get('inliers',[])))
        assert not set(excluded)&set(solved.get('used',solved.get('inliers',[])))
    return dict(solver=solved,actual_pose=final,native_points=out,
        pose_available=bool(final.get('available',False)),new_pose_estimated=new,
        fallback_used=fallback,no_pose=not bool(final.get('available',False)),
        hidden_reprojected=new and bool(hidden),hidden_initial=list(hidden),
        hidden_after=after,hidden_set_changed=new and set(after)!=set(hidden),
        output_status='NEW_POSE' if new else 'BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',
        excluded=list(excluded),reprojected_ids=list(hidden) if new else [],
        reprojections_reused_as_observations=False)
