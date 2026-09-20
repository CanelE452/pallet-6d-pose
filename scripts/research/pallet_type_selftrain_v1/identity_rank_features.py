"""Image-conditioned candidate descriptors, with no R0-relative identity flag."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

QUARTER=np.array([4,0,3,7,5,1,2,6,8])
HALF=QUARTER[QUARTER]
PERMS=np.stack([np.arange(9),HALF,QUARTER,QUARTER[HALF]])
SIDES=np.array([[0,1,2,3],[1,5,6,2],[5,4,7,6],[4,0,3,7]])


def positions(q):
    """8 corners and4 interior samples on each side; no shape rejection."""
    values=[q[:8]]
    for a,b,c,d in SIDES:
        values.append(np.stack([(1-v)*((1-u)*q[a]+u*q[b])+v*((1-u)*q[d]+u*q[c])
            for v in [.25,.75] for u in [.25,.75]]))
    return np.concatenate(values)


def describe(p3,p4,points,box,input_shape):
    """One feature vector per C2 member of each of the two axis assignments."""
    q=np.asarray(points,float);box=np.asarray(box,float)
    assert q.shape==(9,2) and box.shape==(4,) and np.isfinite(q).all()
    scale=max(float(np.linalg.norm(box[2:]-box[:2])),1.);center=(box[:2]+box[2:])/2
    feature=[]
    for perm in PERMS:
        pose=q[perm];xy=positions(pose)
        inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<input_shape[1])&(xy[:,1]<input_shape[0])
        sampled=[]
        for f in [p3,p4]:
            t=torch.as_tensor(np.array(f,copy=True),dtype=torch.float32)[None]
            # Cache padded to640x640; same cell-center convention as line branch.
            grid=torch.as_tensor(2*xy/640-1,dtype=torch.float32)[None,None]
            value=F.grid_sample(t,grid,align_corners=False,padding_mode='zeros')[0,:,0].T.numpy()
            value[~inside]=0
            sampled.append(value)
        feature.append(np.r_[np.concatenate(sampled,axis=1).ravel(),
            ((pose[:8]-center)/scale).ravel(),inside.astype(float)])
    out=np.asarray(feature,np.float32).reshape(2,2,-1)
    assert out.shape==(2,2,4648) and np.isfinite(out).all()
    return out


class Ranker(nn.Module):
    def __init__(self,features=4648):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(features,128),nn.ReLU(),nn.Linear(128,1))

    def forward(self,x):
        # Average official C2 members; q and q180 must have exactly equal scores.
        return self.net(x).squeeze(-1).mean(-1)


def choose(probabilities,threshold):
    if threshold is None:return 0
    p=np.asarray(probabilities,float)
    assert p.shape==(2,) and np.isfinite(p).all()
    return int(p[1]>p[0] and p[1]>=threshold)
