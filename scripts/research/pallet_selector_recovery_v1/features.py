"""No GT/reference imports. Both hypotheses use exactly the same function."""
import cv2
import numpy as np
from challenge.evaluation_v2 import pnp_selector as P
from . import common as C
from .feature_contract import names

def cuboid(w,h,d):
    return np.array([[-w/2,-h/2,-d/2],[w/2,-h/2,-d/2],[w/2,h/2,-d/2],[-w/2,h/2,-d/2],[-w/2,-h/2,d/2],[w/2,-h/2,d/2],[w/2,h/2,d/2],[-w/2,h/2,d/2],[0,0,0.]])

def vector(h,c,hw):
    if not h['success']:return None
    q=np.array(c['keypoints_xy']);conf=np.array(c['keypoints_conf']);box=np.array(c['box_xyxy']);wh=np.maximum(box[2:]-box[:2],1e-6)
    diag=np.linalg.norm(wh);area=np.prod(wh);sc=h['score_components'];dims=h['camera_facing_dimensions_m'];extent=np.array([dims['width'],dims['height'],dims['depth']]);scale=np.linalg.norm(extent)
    R=np.array(h['rotation_camera_facing']);t=np.array(h['translation_camera_facing']);proj=np.array(h['projected_keypoints']);cam=cuboid(*extent)@R.T+t;res=np.linalg.norm(proj-q,axis=1)
    v=[sc['reprojection_rmse_px'],sc['reprojection_rmse_px']/diag,*[sc[k] for k in ('cheirality_fraction','lr_violations','tb_violations','front_rear_violations','invariant_violations','upright_alignment','spread_ratio')],*extent,*t,*(t/scale),*R.ravel()]
    for ix in ([0,1,2,3],[4,5,6,7]):v.append(abs(cv2.contourArea(proj[ix].astype(np.float32)))/area)
    for pairs in (P.LR_PAIRS,P.TB_PAIRS,P.FR_PAIRS):
        e=np.array([np.linalg.norm(proj[a]-proj[b])/diag for a,b in pairs]);v.extend([e.mean(),e.min(),e.max()])
    depth=np.array([cam[b,2]-cam[a,2] for a,b in P.FR_PAIRS])
    for d in (depth,depth/scale):v.extend([d.mean(),d.min(),d.max()])
    v.extend(res);v.extend(res/diag)
    for rr in (res,res/diag):v.extend([rr.mean(),np.median(rr),np.quantile(rr,.9),rr.max(),rr[:4].mean(),rr[4:8].mean(),rr[8],np.sum(rr*conf)/max(conf.sum(),1e-9)])
    v.extend([c['score'],*conf,conf.mean(),conf.min(),np.quantile(conf,.1),np.median(conf),wh[0]/wh[1],area/np.prod(hw)])
    a=np.asarray(v,np.float32);assert len(a)==len(names())
    return a if np.isfinite(a).all() else None

def extract(pred,K,dims,hw):
    c=C.selected(pred)
    if c is None:return dict(valid=False,features=None,selection=None,hypotheses=[])
    q=np.array(c['keypoints_xy'],float)
    try:
        s=P.select_pnp_hypotheses(q,np.array(K),dict(x=float(max(dims[0],dims[2])),y=float(dims[1]),z=float(min(dims[0],dims[2]))),None).to_dict()
    except (ValueError,cv2.error):return dict(valid=False,features=None,selection=None,hypotheses=[])
    hh=sorted(s['hypotheses'],key=lambda h:h['name']);assert tuple(h['name'] for h in hh)==C.HYP
    vv=[vector(h,c,hw) for h in hh];valid=all(v is not None for v in vv)
    return dict(valid=valid,features=np.stack(vv).tolist() if valid else None,selection=s['selected_hypothesis'],hypotheses=hh)

def production_pose(h,dims,source=False):
    if not h['success']:return dict(available=False)
    R=np.array(h['rotation_camera_facing']);t=np.array(h['translation_camera_facing']);d=h['camera_facing_dimensions_m'];ex=np.array([d['width'],d['height'],d['depth']])
    if t[2]<=0:return dict(available=False)
    Q=np.eye(3) if abs(ex[0]-dims[0])<1e-6 else np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    physical=R@Q
    if source:physical=physical@np.diag([1.,-1.,-1.])
    return dict(available=True,R_cf=R.tolist(),R_physical=physical.tolist(),centroid=t.tolist(),cf_extents=ex.tolist(),selected_hypothesis=h['name'])
