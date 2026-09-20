"""Training-only affine augmentation of frozen features and existing coordinates.

This warps REPRESENTATIONS, not images followed by DINO. It does not assume that
the frozen backbone is exactly equivariant, and adds no real/pseudo labels.
"""
import numpy as np
import torch
from torch.nn import functional as F

HEIGHT=768
WIDTH=576


def matrices(count,step,device='cpu'):
    rng=np.random.default_rng(np.random.SeedSequence([20261011,int(step)]))
    out=[]
    center=np.array([(WIDTH-1)/2,(HEIGHT-1)/2])
    for _ in range(count):
        a=np.eye(3)
        if rng.random()>=.5:
            angle=np.deg2rad(rng.uniform(-10,10));scale=rng.uniform(.85,1.15)
            r=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            shift=rng.uniform(-.1,.1,2)*[WIDTH,HEIGHT]
            a[:2,:2]=r;a[:2,2]=center+shift-r@center
        out.append(a)
    return torch.as_tensor(np.stack(out),device=device,dtype=torch.float32)


def coordinates(points,a):
    return torch.bmm(points,a[:,:2,:2].transpose(1,2))+a[:,:2,2][:,None]


def supported(points,valid):
    return valid & torch.isfinite(points).all(-1) & (points>=0).all(-1) & (points[...,0]<WIDTH) & (points[...,1]<HEIGHT)


def warp(feature,a):
    # align_corners=False: a crop pixel x maps to (2*x+1)/WIDTH-1.
    n=a.new_tensor([[2/WIDTH,0,1/WIDTH-1],[0,2/HEIGHT,1/HEIGHT-1],[0,0,1]])
    theta=(n@torch.linalg.inv(a)@torch.linalg.inv(n))[:,:2]
    grid=F.affine_grid(theta,feature.shape,align_corners=False)
    z=F.grid_sample(feature,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    identity=(a==torch.eye(3,device=a.device)).all(-1).all(-1)
    # Identity samples remain bit-exact, not reinterpolated.
    return torch.where(identity[:,None,None,None],feature,z)


def augment(batch,a):
    out=dict(batch)
    out['feature']=tuple(warp(f,a) for f in batch['feature'])
    for name,mask in [('points','valid'),('target','target_valid')]:
        out[name]=coordinates(batch[name],a)
        out[mask]=supported(out[name],batch[mask])
    return out
