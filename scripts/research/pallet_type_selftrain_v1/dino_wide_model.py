"""Same-parameter DINO head on a 2x wider field at unchanged pixel scale."""
import numpy as np
import torch
from torch.nn import functional as F
from .dino_localization_model import Head as ParentHead

HEIGHT=768; WIDTH=576; GRID_H=192; GRID_W=144


def widen_matrix(matrix):
    out=np.asarray(matrix,float).copy()
    if out.shape!=(3,3) or not np.isfinite(out).all():raise ValueError('Finite affine matrix required')
    out[:2,2]+=[144,192]
    return out


def prior_logits(points,valid):
    safe=torch.where(valid[...,None],points,torch.zeros_like(points))
    y=torch.arange(GRID_H,device=points.device,dtype=points.dtype)[None,None,:,None]*4
    x=torch.arange(GRID_W,device=points.device,dtype=points.dtype)[None,None,None,:]*4
    d=(x-safe[:,:,0,None,None]).square()+(y-safe[:,:,1,None,None]).square()
    return (-d/(2*12**2)).clamp_min(-6)*valid[:,:,None,None]


class Head(ParentHead):
    def forward(self,features,points,valid):
        assert features.shape[1:]==(384,56,42)
        prior=prior_logits(points,valid)
        hints=F.interpolate(prior.exp(),size=(56,42),mode='bilinear',align_corners=False)
        y,x=torch.meshgrid(torch.linspace(-1,1,56,device=features.device),
                           torch.linspace(-1,1,42,device=features.device),indexing='ij')
        xy=torch.stack([x,y])[None].expand(len(features),-1,-1,-1)
        q=self.mix(torch.cat([self.project(features),hints,xy],dim=1))
        q=self.up(F.interpolate(q,size=(112,84),mode='bilinear',align_corners=False))
        residual=F.interpolate(self.out(q),size=(GRID_H,GRID_W),mode='bilinear',align_corners=False)
        return prior+residual


def target_distribution(target,valid):
    valid=valid & torch.isfinite(target).all(-1) & (target>=0).all(-1) & (target[:,:,0]<WIDTH) & (target[:,:,1]<HEIGHT)
    safe=torch.where(valid[...,None],target/4,torch.zeros_like(target));floor=safe.floor();frac=safe-floor
    b,k,_=target.shape;q=target.new_zeros(b,k,GRID_H*GRID_W)
    for dx,dy in [(0,0),(0,1),(1,0),(1,1)]:
        x=floor[:,:,0].long()+dx;y=floor[:,:,1].long()+dy
        mask=valid&(x>=0)&(x<GRID_W)&(y>=0)&(y<GRID_H)
        mass=(frac[:,:,0] if dx else 1-frac[:,:,0])*(frac[:,:,1] if dy else 1-frac[:,:,1])*mask
        q.scatter_add_(2,(y.clamp(0,GRID_H-1)*GRID_W+x.clamp(0,GRID_W-1))[:,:,None],mass[:,:,None])
    q=q/q.sum(-1,keepdim=True).clamp_min(1e-12)
    return q,valid


def loss(logits,target,valid):
    mask=valid.clone();mask[:,8]=False;q,mask=target_distribution(target,mask)
    ce=-(q*logits.flatten(2).log_softmax(-1)).sum(-1)
    return (ce*mask).sum()/mask.sum().clamp_min(1)


def decode(logits):
    if logits.shape[-2:]!=(GRID_H,GRID_W) or not torch.isfinite(logits).all():raise ValueError('Finite192x144 logits required')
    flat=logits.flatten(2);anchor=flat.argmax(-1);ax=anchor%GRID_W;ay=anchor//GRID_W
    yy=torch.arange(GRID_H,device=logits.device)[None,None,:,None]
    xx=torch.arange(GRID_W,device=logits.device)[None,None,None,:]
    window=((xx-ax[...,None,None]).abs()<=2)&((yy-ay[...,None,None]).abs()<=2)
    prob=logits.masked_fill(~window,-torch.inf).flatten(2).softmax(-1).reshape_as(logits)
    return torch.stack([(prob*xx).sum((-2,-1)),(prob*yy).sum((-2,-1))],-1)*4
