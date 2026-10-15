"""Exact piecewise gradient of the unchanged v4.10 whole-outline objective.

Convex-hull membership and nearest-segment association are fixed locally.
The envelope theorem handles optimized closest-point coordinates; model
segment clipping and arc-length weights are differentiated explicitly. Hull,
association and clipping topology transitions remain nonsmooth as in v4.10.
No keypoint coordinate, observation, configuration or guard is changed here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import statewise_global_outline_solver_v4_10 as BASE
from . import joint_ot_jacobian_v2 as J


def _nearest(points, segments):
    a,b=np.asarray(segments)[:,0],np.asarray(segments)[:,1]
    direction=b-a;squared=np.sum(direction*direction,axis=1)
    alpha=np.sum((points[:,None,:]-a[None,:,:])*direction[None,:,:],axis=2)/np.maximum(squared,1e-20)
    alpha=np.clip(alpha,0.,1.)
    foot=a[None,:,:]+alpha[...,None]*direction[None,:,:]
    residual=points[:,None,:]-foot
    index=np.argmin(np.sum(residual*residual,axis=2),axis=1)
    row=np.arange(len(points))
    return residual[row,index],index,alpha[row,index]


def _clip_with_jacobian(a,b,ja,jb,bounds):
    direction=b-a;derivative=jb-ja;lo,hi=0.,1.;lo_j=np.zeros(6);hi_j=np.zeros(6)
    for axis in (0,1):
        d=direction[axis]
        if abs(d)<1e-14:
            if not bounds[axis][0]<=a[axis]<=bounds[axis][1]:return None
        else:
            candidates=[]
            for limit in bounds[axis]:
                value=(limit-a[axis])/d
                gradient=(-ja[axis]*d-(limit-a[axis])*derivative[axis])/d**2
                candidates.append((value,gradient))
            first,second=sorted(candidates,key=lambda v:v[0])
            if first[0]>lo:lo,lo_j=first
            if second[0]<hi:hi,hi_j=second
            if lo>=hi:return None
    first=a+lo*direction;second=a+hi*direction
    first_j=ja+lo*derivative+direction[:,None]*lo_j
    second_j=ja+hi*derivative+direction[:,None]*hi_j
    return first,second,first_j,second_j


def _loss_and_force(residual,sigma,delta):
    squared=np.sum(residual*residual,axis=1)/sigma**2
    denominator=np.sqrt(1.+squared/delta**2)
    loss=delta**2*(denominator-1.)
    force=residual/(sigma**2*denominator[:,None])
    return loss,force


def outline_value_and_gradient(p,xyz,K,scale,observation,configuration=None):
    """Same value/audit as v4.10, analytic six-parameter gradient where smooth."""
    cfg=BASE._outline_config(configuration)
    q,depth,jq,_=J.projection_and_jacobian(p,xyz,K,scale)
    if not np.isfinite(q).all() or np.min(depth)<=1e-8:
        return 1e12,np.zeros(6),{'valid_projection':False}
    try:indices=ConvexHull(q).vertices
    except QhullError:return 1e12,np.zeros(6),{'valid_projection':False}
    polygon=q[indices];jp=jq[indices]
    next_polygon=np.roll(polygon,-1,axis=0);next_jp=np.roll(jp,-1,axis=0)
    model_segments=np.stack([polygon,next_polygon],axis=1)
    actual=np.asarray(observation['observed_boundary_segments_xy'],float)
    observed=np.asarray(observation['observed_boundary_samples_xy'],float)
    reverse,which,alpha=_nearest(observed,model_segments)
    reverse_loss,reverse_force=_loss_and_force(reverse,cfg['normal_sigma_px'],cfg['huber_delta'])
    foot_j=(1-alpha[:,None,None])*jp[which]+alpha[:,None,None]*next_jp[which]
    reverse_gradient=-np.einsum('ni,nij->nj',reverse_force,foot_j).mean(axis=0)
    second=float(reverse_loss.mean())
    h,w=observation['image_hw'];margin=cfg['frame_margin_px']
    bounds=((margin,w-1-margin),(margin,h-1-margin))
    points=[];point_j=[];weights=[];weight_j=[]
    n=cfg['model_samples_per_segment'];sample_alpha=(np.arange(n)+.5)/n
    for a,b,ja,jb in zip(polygon,next_polygon,jp,next_jp):
        clipped=_clip_with_jacobian(a,b,ja,jb,bounds)
        if clipped is None:continue
        a,b,ja,jb=clipped;d=b-a;dj=jb-ja;length=float(np.linalg.norm(d))
        if length<=1e-14:continue
        length_j=d@dj/length
        points.extend(a[None,:]+sample_alpha[:,None]*d)
        point_j.extend(ja[None,:,:]+sample_alpha[:,None,None]*dj)
        weights.extend([length/n]*n)
        weight_j.extend([length_j/n]*n)
    if not points:return 1e12,np.zeros(6),{'valid_projection':False}
    points=np.asarray(points);point_j=np.asarray(point_j);weights=np.asarray(weights);weight_j=np.asarray(weight_j)
    forward,_,_=_nearest(points,actual)
    forward_loss,forward_force=_loss_and_force(forward,cfg['normal_sigma_px'],cfg['huber_delta'])
    first=float(np.average(forward_loss,weights=weights))
    loss_gradient=np.einsum('ni,nij->nj',forward_force,point_j)
    forward_gradient=(np.sum(weights[:,None]*loss_gradient,axis=0)
        +np.sum((forward_loss-first)[:,None]*weight_j,axis=0))/weights.sum()
    def stats(residual):
        distance=np.linalg.norm(residual,axis=1)
        return dict(mean_px=float(distance.mean()),median_px=float(np.median(distance)),
            P90_px=float(np.quantile(distance,.9)),count=len(distance))
    audit=dict(valid_projection=True,model_to_observed=stats(forward),observed_to_model=stats(reverse),
        model_to_observed_robust_mean=first,observed_to_model_robust_mean=second,
        normalized_by_arc_length_not_duplicated_pixel_count=True)
    return .5*(first+second),.5*(forward_gradient+reverse_gradient),audit
