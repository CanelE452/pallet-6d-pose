"""Piecewise analytic signed-distance outline factor with fixed CAL bias.

Convex-hull membership and nearest-segment association are fixed locally.
The envelope theorem handles optimized closest-point coordinates; model
segment clipping and arc-length weights are differentiated explicitly. Hull,
association and clipping topology transitions remain nonsmooth as in v4.10.
No keypoint coordinate, observation, configuration or guard is changed here.
"""
from __future__ import annotations

# v4.12b: stable translated polygon orientation; collapsed projections are invalid.

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


def _polygon_orientation(polygon):
    relative=np.asarray(polygon,float)-np.asarray(polygon,float)[0]
    following=np.roll(relative,-1,axis=0)
    twice_area=float(np.sum(relative[:,0]*following[:,1]-relative[:,1]*following[:,0]))
    if not np.isfinite(twice_area) or twice_area==0.:
        return None
    return 1. if twice_area>0 else -1.


def _signed_loss_and_force(residual,points,segments,indices,polygon,bias,sigma,delta,*,reverse):
    """Signed finite-segment distance; orientation supplies the zero-distance derivative."""
    polygon=np.asarray(polygon,float);a=polygon;b=np.roll(polygon,-1,axis=0)
    orientation=_polygon_orientation(polygon)
    if orientation is None:raise ValueError('Nondegenerate polygon orientation required')
    edge=b-a;relative=points[:,None,:]-a
    cross=(edge[None,:,0]*relative[:,:,1]-edge[None,:,1]*relative[:,:,0])*orientation
    signed_halfplane_distance=cross/np.maximum(np.linalg.norm(edge,axis=1),np.finfo(float).tiny)[None,:]
    inside=np.min(signed_halfplane_distance,axis=1)>=-1e-8
    sign=np.where(inside,-1.,1.)
    if not reverse:sign=-sign
    distance=np.linalg.norm(residual,axis=1)
    signed=sign*distance
    centered=signed-bias
    denominator=np.sqrt(1.+(centered/(sigma*delta))**2)
    loss=delta**2*(denominator-1.)
    direction=np.asarray(segments)[indices,1]-np.asarray(segments)[indices,0]
    lengths=np.linalg.norm(direction,axis=1)
    if np.any(lengths<=1e-14):raise ValueError('Positive nearest segment length required')
    normal=orientation*np.column_stack([direction[:,1],-direction[:,0]])/lengths[:,None]
    if not reverse:normal=-normal
    gradient=normal.copy()
    nonzero=distance>1e-12
    gradient[nonzero]=sign[nonzero,None]*residual[nonzero]/distance[nonzero,None]
    force=centered[:,None]*gradient/(sigma**2*denominator[:,None])
    return loss,force


def outline_value_and_gradient(p,xyz,K,scale,observation,configuration=None):
    """Signed observed-minus-model residual about CAL mean; zero mean recovers v4.11."""
    cfg=BASE._outline_config(configuration)
    bias=float(observation.get('normal_bias_px',0.))
    if not np.isfinite(bias):raise ValueError('Finite caller-bound CAL normal bias required')
    q,depth,jq,_=J.projection_and_jacobian(p,xyz,K,scale)
    if not np.isfinite(q).all() or np.min(depth)<=1e-8:
        return 1e12,np.zeros(6),{'valid_projection':False}
    try:indices=ConvexHull(q).vertices
    except QhullError:return 1e12,np.zeros(6),{'valid_projection':False}
    polygon=q[indices];jp=jq[indices]
    if _polygon_orientation(polygon) is None:
        return 1e12,np.zeros(6),dict(valid_projection=False,reason='collapsed_projected_polygon')
    next_polygon=np.roll(polygon,-1,axis=0);next_jp=np.roll(jp,-1,axis=0)
    model_segments=np.stack([polygon,next_polygon],axis=1)
    actual=np.asarray(observation['observed_boundary_segments_xy'],float)
    observed=np.asarray(observation['observed_boundary_samples_xy'],float)
    reverse,which,alpha=_nearest(observed,model_segments)
    reverse_loss,reverse_force=_signed_loss_and_force(reverse,observed,model_segments,which,polygon,bias,
        cfg['normal_sigma_px'],cfg['huber_delta'],reverse=True)
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
    forward,forward_which,_=_nearest(points,actual)
    forward_loss,forward_force=_signed_loss_and_force(forward,points,actual,forward_which,
        np.asarray(observation['polygon_xy'],float),bias,cfg['normal_sigma_px'],cfg['huber_delta'],reverse=False)
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
        normalized_by_arc_length_not_duplicated_pixel_count=True,
        expected_observed_minus_model_normal_offset_px=bias,total_normal_sigma_px=cfg['normal_sigma_px'])
    return .5*(first+second),.5*(forward_gradient+reverse_gradient),audit
