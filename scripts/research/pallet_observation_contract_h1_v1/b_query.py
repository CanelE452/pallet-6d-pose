"""Reference-only B-query geometry. No RGB distribution is transported.

Q00 preserves stored query coordinates; other variants change only geometry.
The original finite tolerances are machine-precision rules, not fitted gates.
"""
from __future__ import annotations
import numpy as np

QUERY_VARIANTS=('Q00_ORIGINAL','Q10_GT_ANCHOR','Q01_GT_DIRECTION','Q11_GT_BOTH')
EPS=np.finfo(np.float64).eps
LAMBDA_EPS=8*np.sqrt(EPS)
PARALLEL_EPS=np.sqrt(EPS)


def finite_intersection(a,b,center,normal,*,image_hw,depth=None):
    a,b,c,n=map(lambda x:np.asarray(x,float),(a,b,center,normal))
    if any(x.shape!=(2,)for x in (a,b,c,n)):raise ValueError('Four finite image2-vectors required')
    v=b-a;w=a-c
    cross=lambda x,y:float(x[0]*y[1]-x[1]*y[0])
    D,N,M=cross(n,v),cross(w,n),cross(w,v)
    length=float(np.linalg.norm(v));pxeps=np.sqrt(EPS)*max(1.,float(np.hypot(*image_hw)))
    status='SUPPORTED_FINITE_SEGMENT'
    if not np.isfinite(np.r_[a,b,c,n,D,N,M,length]).all():status='NONFINITE_PROJECTION'
    elif depth is not None and(not np.isfinite(depth).all()or np.min(depth)<=0):status='NONPOSITIVE_DEPTH'
    elif length<=pxeps:status='DEGENERATE_PROJECTED_EDGE'
    elif abs(D)<=max(pxeps,PARALLEL_EPS*length):status='PARALLEL_NORMAL_AND_EDGE'
    lam=offset=None;point=None
    if status not in ('NONFINITE_PROJECTION','DEGENERATE_PROJECTED_EDGE','PARALLEL_NORMAL_AND_EDGE')and D!=0:
        lam=float(N/D);offset=float(M/D);point=(a+lam*v).tolist()
        if status=='SUPPORTED_FINITE_SEGMENT'and lam < -LAMBDA_EPS:status='BEFORE_FINITE_ENDPOINT'
        elif status=='SUPPORTED_FINITE_SEGMENT'and lam>1+LAMBDA_EPS:status='AFTER_FINITE_ENDPOINT'
    H,W=image_hw
    return dict(lambda_segment=lam,normal_offset_px=offset,intersection_xy=point,
        geometry_status=status,finite_support=status=='SUPPORTED_FINITE_SEGMENT',
        projected_length_px=length,denominator_px=D,
        intersection_in_canvas=point is not None and 0<=point[0]<W and 0<=point[1]<H,
        lambda_epsilon=float(LAMBDA_EPS),projected_length_epsilon_px=float(pxeps),parallel_angle_epsilon=float(PARALLEL_EPS))


def outward_normal(projected,k,m):
    p=np.asarray(projected,float);v=p[m]-p[k];length=float(np.linalg.norm(v))
    if not np.isfinite(p).all()or not np.isfinite(length)or length<=0:return None,None
    t=v/length;n=np.array([-t[1],t[0]])
    dot=float(n@((p[k]+p[m])/2-p.mean(0)))
    if dot<0:n=-n
    return t,n


def factorial_query(query,initial_projection,reference_projection,observed_anchor,*,pixel_scale,image_hw,depth=None):
    """Same native IDs/along/state/profile universe; no old RGB as a new cost."""
    k,m=int(query['native_corner']),int(query['native_neighbor'])
    init=np.asarray(initial_projection,float);gt=np.asarray(reference_projection,float);anchor=np.asarray(observed_anchor,float)
    t0,n0=outward_normal(init,k,m);tg,ng=outward_normal(gt,k,m)
    along=float(query['along_base_px'])*pixel_scale;gtlength=float(np.linalg.norm(gt[m]-gt[k]))
    out=[]
    for variant in QUERY_VARIANTS:
        replace_anchor=variant[1]=='1';replace_direction=variant[2]=='1'
        t=tg if replace_direction else t0
        n=ng if replace_direction else np.asarray(query['outward_normal_xy'],float)
        base=gt[k]if replace_anchor else anchor
        center=np.asarray(query['center_xy'],float)if variant=='Q00_ORIGINAL'else base+along*t if t is not None else np.full(2,np.nan)
        if n is None:n=np.full(2,np.nan)
        result=finite_intersection(gt[k],gt[m],center,n,image_hw=image_hw,depth=depth)
        H,W=image_hw;offsets=np.asarray(query['sample_offsets_base_px'],float)*pixel_scale
        samples=center+offsets[:,None]*n
        sample_canvas=bool(np.isfinite(samples).all()and((samples[:,0]>=0)&(samples[:,0]<W)&(samples[:,1]>=0)&(samples[:,1]<H)).all())
        center_canvas=bool(np.isfinite(center).all()and 0<=center[0]<W and 0<=center[1]<H)
        overrun=bool(np.isfinite(gtlength)and along>gtlength)
        out.append(dict(query_variant=variant,support_id=query['support_id'],native_corner=k,native_neighbor=m,
            along_base_px=query['along_base_px'],along_raw_px=along,reference_edge_length_px=gtlength,
            query_domain_overrun=overrun,query_domain_status='QUERY_DOMAIN_OVERRUN'if overrun else'WITHIN_ALONG_EDGE_LENGTH',
            center_xy=center.tolist(),tangent_xy=None if t is None else t.tolist(),outward_normal_xy=n.tolist(),
            original_observed_anchor_xy=anchor.tolist(),reference_anchor_xy=gt[k].tolist(),
            center_in_canvas=center_canvas,sample_line_in_canvas=sample_canvas,
            geometry_domain_joint_support=bool(result['finite_support']and not overrun and sample_canvas),
            original_profile_supported=bool(query['supported']),
            optical_distribution_bound_to_this_query=variant=='Q00_ORIGINAL',
            new_optical_energy_evaluated=False,original_RGB_distribution_reused_as_new_cost=False,
            diagnostic_reference_used=True,production_candidate=False,**result))
    return out
