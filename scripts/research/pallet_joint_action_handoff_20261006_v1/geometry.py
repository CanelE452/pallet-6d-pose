"""Prediction-only, centred camera-axis pose-coupled corner displacement bank.

Units: rotation radians, translation metres; K and q original pixels. Existing
PnP uses zero distortion. No reference input is accepted by this module.
"""
import hashlib,itertools
import cv2,numpy as np
from .a_common import POSE
NUMERIC=dict(axis_difference_rotation_rad=1e-5,axis_difference_translation_m=1e-5,
             sensitivity_min_px_per_unit=1e-8,depth_epsilon_m=1e-6,
             root_iterations=24,bracket_iterations=12,cap_tolerance_px=1e-6,
             coordinate_duplicate_tolerance_px=1e-7,headroom_ADDsym_m=1e-7)

def directions():
    dirs=[];radii=[]
    for axis in range(6):
        for sign in [-1,1]:
            for fraction in [.25,.5,1.]:
                v=np.zeros(6);v[axis]=sign;dirs.append(v);radii.append(fraction)
    for sign in itertools.product([-1,1],repeat=6):dirs.append(sign);radii.append(.5)
    return np.array(dirs,float),np.array(radii,float)

def rotations(v):
    angle=np.linalg.norm(v,axis=-1);axis=v/np.maximum(angle[:,None],1e-30)
    x,y,z=axis.T;K=np.zeros((len(v),3,3))
    K[:,0,1]=-z;K[:,0,2]=y;K[:,1,0]=z;K[:,1,2]=-x;K[:,2,0]=-y;K[:,2,1]=x
    return np.eye(3)[None]+np.sin(angle)[:,None,None]*K+(1-np.cos(angle))[:,None,None]*(K@K)

def project(R,t,X,K):
    camera=np.einsum('bij,nj->bni',R,X)+t[:,None]
    valid=np.isfinite(camera).all((1,2))&(camera[:,:,2]>NUMERIC['depth_epsilon_m']).all(-1)
    homogeneous=camera@K.T
    uv=homogeneous[:,:,:2]/homogeneous[:,:,2:]
    return uv,valid

def build_bank(q,K,xyz,raw_hw,point_valid=None):
    """Return <=201 actions, with exact-copy NoOp at index 0 and centre8 fixed."""
    q=np.array(q,copy=True);valid=np.isfinite(q[:8]).all(-1)&~(q[:8]==-1).all(-1)
    if point_valid is not None:valid &= np.asarray(point_valid,bool)[:8]
    rows=[q.copy()];hypotheses=['NoOp'];details=[]
    if valid.sum()<6:return dict(points=np.stack(rows),hypotheses=hypotheses,details=details,reason='PNP_SUPPORT_LT6')
    if abs(xyz[0]-xyz[2])<1e-9:hs=[('SQUARE_IDENTICAL_WD',np.array(xyz,float))]
    else:
        selected=POSE.select_pnp_hypotheses(q,K,dict(x=max(xyz[0],xyz[2]),y=xyz[1],z=min(xyz[0],xyz[2])),None)
        hs=[(h.name,np.array([h.camera_facing_dimensions.as_dict()[k] for k in ['width','height','depth']])) for h in selected.hypotheses if h.success]
    cap=.01*np.hypot(*raw_hw);dirs,fractions=directions()
    for name,cf in hs:
        X=POSE.cuboid(*cf)
        try:solved=POSE.solve(X,q[:8],K,valid)
        except (cv2.error,ValueError):continue
        if solved is None:continue
        R0,t0,_=solved;base,ok=project(R0[None],t0[None],X,K)
        if not ok[0]:continue
        axis_size=np.zeros(6)
        for axis in range(6):
            delta=np.zeros(6);eps=1e-5;delta[axis]=eps
            uv,good=project(rotations(delta[None,:3])@R0,t0[None]+delta[None,3:],X,K)
            s=np.max(np.linalg.norm(uv[0,valid]-base[0,valid],axis=-1))/eps if good[0] else 0
            if s>NUMERIC['sensitivity_min_px_per_unit']:axis_size[axis]=1/s
        enabled=(np.abs(dirs)*(axis_size==0)).sum(-1)==0
        v=dirs*axis_size;target=cap*fractions
        def at(a):
            uv,good=project(rotations(v[:,:3]*a[:,None])@R0,t0[None]+v[:,3:]*a[:,None],X,K)
            displacement=uv-base
            distance=np.max(np.linalg.norm(displacement[:,valid],axis=-1),axis=-1)
            return displacement,distance,good
        lo=np.zeros(100);hi=target.copy()
        for _ in range(NUMERIC['bracket_iterations']):
            _,dist,good=at(hi);grow=enabled&good&(dist<target);hi[grow]*=2
        _,dist,good=at(hi);bracket=enabled&((dist>=target)|~good)
        for _ in range(NUMERIC['root_iterations']):
            mid=(lo+hi)/2;_,dist,good=at(mid);lower=good&(dist<=target)
            lo=np.where(lower,mid,lo);hi=np.where(lower,hi,mid)
        delta,dist,good=at(lo)
        for j in range(100):
            if not(bracket[j] and good[j] and dist[j]>1e-9 and dist[j]<=cap+NUMERIC['cap_tolerance_px']):continue
            action=q.copy();action[:8][valid]+=delta[j,valid]
            # Joint coordinates and hypothesis only, no reference-based filtering.
            if any(n==name and np.max(np.abs(a[:8]-action[:8]))<1e-7 for a,n in zip(rows,hypotheses)):continue
            rows.append(action);hypotheses.append(name)
            details.append(dict(hypothesis=name,direction_index=j,radius_fraction=float(fractions[j]),max_move_px=float(dist[j]),rotation_rad=(v[j,:3]*lo[j]).tolist(),translation_m=(v[j,3:]*lo[j]).tolist()))
    return dict(points=np.stack(rows),hypotheses=hypotheses,details=details,reason='OK' if len(rows)>1 else 'NO_VALID_MOTION')

def permutation_indices(frame_id,count,seed=20261006):
    """Frame-specific, reference-blind fixed permutations; NoOp is never moved."""
    digest=hashlib.sha256(f'{seed}:{frame_id}'.encode()).digest();rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
    return np.stack([np.r_[0,rng.permutation(np.arange(1,count))] for _ in range(8)])

def permute_bank(bank,frame_id):
    q=bank['points'];indices=permutation_indices(frame_id,len(q));out=q.copy()
    for i in range(8):out[:,i]=q[indices[i],i]
    return dict(points=out,hypotheses=['NoOp']+['PERM']*(len(q)-1),details=bank['details'],reason=bank['reason'],permutation=indices)
