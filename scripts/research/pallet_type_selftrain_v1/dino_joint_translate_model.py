"""Whole-pattern registration against learned image likelihoods, no GT input."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from . import dino_joint_model as J

PERMS=J.PERMS


def logmass(syn,mix):
    assert syn.shape==mix.shape==(1,9,192,144)
    a=syn[:,:8].flatten(2).log_softmax(-1).reshape(1,8,192,144)
    b=mix[:,:8].flatten(2).log_softmax(-1).reshape_as(a)
    p=(torch.logaddexp(a,b)-math.log(2)).exp()
    return (F.avg_pool2d(p,5,stride=1,padding=2)*25).clamp_min(1e-12).log()[0]


def propose(likelihood,points,valid):
    """Search4 whole cube-graph permutations and one shared2D translation."""
    assert likelihood.shape==(8,192,144) and points.shape==(9,2)
    if not bool(valid[:8].all()) or not torch.isfinite(points).all():
        return dict(permutation=0,delta_crop=[0.,0.],gain=0.,baseline_score=None,score=None,available=False)
    yy,xx=torch.meshgrid(torch.arange(-96,97,device=points.device),torch.arange(-72,73,device=points.device),indexing='ij')
    delta=torch.stack([xx,yy],-1).reshape(-1,2).to(points.dtype);n=len(delta)
    pp=torch.as_tensor(np.asarray(PERMS)[:,:8],device=points.device)
    q=points[pp]/4
    locations=q[:,:,None,:]+delta[None,None,:,:]
    inside=((locations>=0)&(locations<=points.new_tensor([143,191]))).all(-1)
    grid=locations/points.new_tensor([143,191])*2-1
    maps=likelihood[None].expand(4,-1,-1,-1).reshape(32,1,192,144)
    sampled=F.grid_sample(maps,grid.reshape(32,1,n,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(4,8,n)
    sampled=torch.where(inside,sampled,torch.full_like(sampled,math.log(1e-12)))
    scores=sampled.mean(1);zero=int(((delta==0).all(-1)).nonzero()[0]);baseline=float(scores[0,zero])
    scores=scores.masked_fill(~inside.all(1),-torch.inf)
    value,index=scores.flatten().max(0)
    if not torch.isfinite(value):return dict(permutation=0,delta_crop=[0.,0.],gain=0.,baseline_score=baseline,score=baseline,available=False)
    perm=int(index)//n;ix=int(index)%n
    # Tie preference: retain exact original when no evidence gain exists.
    gain=float(value)-baseline
    if gain<=1e-6:perm=0;ix=zero;gain=0.;value=points.new_tensor(baseline)
    return dict(permutation=perm,delta_crop=(delta[ix]*4).tolist(),gain=max(0.,gain),
        baseline_score=baseline,score=float(value),available=True)


def restore(original,proposal,matrix,apply=True):
    original=np.asarray(original);out=original.copy()
    if not apply or not proposal['available']:return out
    perm=np.asarray(PERMS[proposal['permutation']],int)
    delta=np.asarray(proposal['delta_crop'])@np.linalg.inv(np.asarray(matrix))[:2,:2].T
    out[:8]=original[perm[:8]]+delta;out[8]=original[8]
    return out


def changed(proposal):
    return proposal['available'] and (proposal['permutation']!=0 or any(v!=0 for v in proposal['delta_crop']))
