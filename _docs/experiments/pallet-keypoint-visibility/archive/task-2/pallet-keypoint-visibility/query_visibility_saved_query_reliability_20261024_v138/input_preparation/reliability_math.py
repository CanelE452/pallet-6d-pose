"""Pure input-only algebra. No source files, GT, images, models or poses."""
import numpy as np

RISK_NAMES=('risk_low_detector_confidence','risk_low_native_keypoint_confidence',
 'risk_local_invalid_fraction','risk_central32_invalid_fraction','risk_context_invalid_fraction',
 'risk_input_unavailable','risk_native_bounds_invalid','risk_query_outside_pred_bbox_normalized',
 'risk_global_edge_concurrence_rms','risk_native_heldout_edge_rms')
# Original CF channel topology, not image-specific association or image sorting.
EDGE_GROUPS=(((0,1),(3,2),(4,5),(7,6)),((0,3),(1,2),(4,7),(5,6)),((0,4),(1,5),(2,6),(3,7)))
EDGE_GROUP_NAMES=('width','vertical','depth')
EPS=1e-10 # Fixed numerical rank/zero guard, never fitted to data or labels.
G42_NAMES=tuple([f'G42_native_slot_onehot_{k}'for k in range(8)]+
 [f'G42_relative_q{k}_{axis}'for k in range(8)for axis in('x','y')]+
 [f'G42_continuous_valid_{k}'for k in range(8)]+
 ['G42_context_x0_relative','G42_context_y0_relative','G42_context_x1_relative','G42_context_y1_relative',
  'G42_qx_over_W','G42_qy_over_H','G42_spanx_over_scale','G42_spany_over_scale','G42_H_over_scale','G42_W_over_scale'])
DESCRIPTOR_NAMES=G42_NAMES+tuple(f'q{k}_{axis}_over_image'for k in range(8)for axis in('x','y'))
assert len(G42_NAMES)==42 and len(DESCRIPTOR_NAMES)==58

def finite_native(q):
    if q is None:return False
    a=np.asarray(q,np.float64)
    return bool(a.shape==(2,)and np.isfinite(a).all()and not(a==-1).all())

def local_valid96(q,L,hw):
    """Exactly V90 sample-centre validity, with no RGB sampling."""
    h,w=hw;u=(np.arange(96,dtype=np.float64)+.5)/96-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L)
    return(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)

def context_valid224(box,hw,target_grid):
    """Exactly existing target sampler's positive-bilinear-neighbor mask."""
    h,w=hw;grid=target_grid(box);x,y=grid[...,0],grid[...,1]
    ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    valid=np.ones(x.shape,bool)
    for ox,oy,weight in((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        valid&=inside|(weight==0)
    return valid

def outside_bbox(q,box):
    if not finite_native(q)or box is None:return np.nan
    b=np.asarray(box,np.float64)
    if b.shape!=(4,)or not np.isfinite(b).all()or not(b[2:]>b[:2]).all():return np.nan
    delta=np.maximum(np.maximum(b[:2]-q,np.asarray(q)-b[2:]),0.)
    return float(np.linalg.norm(delta)/max(float((b[2:]-b[:2]).max()),1.))

def line_through(a,b):
    line=np.cross(np.r_[a,1.],np.r_[b,1.]);length=np.linalg.norm(line[:2])
    return None if length<=EPS else line/length

def null_of_lines(lines):
    """Small homogeneous incidence SVD; never estimate camera, pose or q.

    A VP at infinity is valid: no division by its last coordinate occurs.
    Rank1/coincident-line configurations are undefined, never perfect zero.
    """
    if any(line is None for line in lines):return None,np.nan,np.nan
    matrix=np.asarray(lines,np.float64);_,s,vh=np.linalg.svd(matrix,full_matrices=False)
    condition=float(s[1]/s[0])if s[0]>EPS else np.nan
    if not np.isfinite(condition)or condition<=EPS:return None,np.nan,condition
    v=vh[-1];residual=float(np.sqrt(np.mean((matrix@v)**2)))
    return v,residual,condition

def edge_signals(q8,k):
    out=dict(global_group_residual=np.full(3,np.nan),global_group_condition=np.full(3,np.nan),
     global_group_valid=np.zeros(3,bool),heldout_group_residual=np.full(3,np.nan),heldout_group_condition=np.full(3,np.nan),
     heldout_group_valid=np.zeros(3,bool),global_rms=np.nan,heldout_rms=np.nan)
    if q8 is None:return out
    q=np.asarray(q8,np.float64)
    if q.shape!=(8,2)or not all(finite_native(x)for x in q):return out
    span=float(np.ptp(q,axis=0).max())
    if span<=EPS:return out
    q=(q-q.mean(0))/span
    for gi,edges in enumerate(EDGE_GROUPS):
        lines=[line_through(q[a],q[b])for a,b in edges]
        _,res,condition=null_of_lines(lines);out['global_group_condition'][gi]=condition
        if np.isfinite(res):out['global_group_residual'][gi]=res;out['global_group_valid'][gi]=True
        incident=[i for i,(a,b)in enumerate(edges)if k in(a,b)];assert len(incident)==1
        j=incident[0]
        if lines[j]is None:continue # A collapsed own edge is undefined, not perfect incidence.
        vp,_,condition=null_of_lines([line for i,line in enumerate(lines)if i!=j])
        out['heldout_group_condition'][gi]=condition
        if vp is None:continue
        a,b=edges[j];neighbor=b if a==k else a
        predicted_line=np.cross(np.r_[q[neighbor],1.],vp);norm=np.linalg.norm(predicted_line[:2])
        if norm<=EPS:continue
        residual=float(abs(predicted_line@np.r_[q[k],1.])/norm)
        out['heldout_group_residual'][gi]=residual;out['heldout_group_valid'][gi]=True
    if out['global_group_valid'].all():out['global_rms']=float(np.sqrt(np.mean(out['global_group_residual']**2)))
    if out['heldout_group_valid'].all():out['heldout_rms']=float(np.sqrt(np.mean(out['heldout_group_residual']**2)))
    return out

def risk_values(score,kconf,local_fraction,central_fraction,context_fraction,available,continuous,nearest,outside,geometry):
    return np.asarray([1-score,1-kconf,1-local_fraction,1-central_fraction,1-context_fraction,
     float(not available),float(not(continuous and nearest)),outside,geometry['global_rms'],geometry['heldout_rms']],np.float64)
