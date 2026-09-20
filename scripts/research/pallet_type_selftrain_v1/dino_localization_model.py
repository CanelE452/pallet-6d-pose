"""Frozen-feature heatmap refiner with an uncapped global search domain."""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.research.pallet_sensors_submission_v1.prior_model import target_distribution
from .heatmap_mode_decoder import local_mode


def prior_logits(points,valid):
    safe=torch.where(valid[...,None],points,torch.zeros_like(points))
    y=torch.arange(96,device=points.device,dtype=points.dtype)[None,None,:,None]*4
    x=torch.arange(72,device=points.device,dtype=points.dtype)[None,None,None,:]*4
    distance=(x-safe[:,:,0,None,None]).square()+(y-safe[:,:,1,None,None]).square()
    return (-distance/(2*12**2)).clamp_min(-6)*valid[:,:,None,None]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(384,64,1),nn.GroupNorm(8,64),nn.GELU())
        self.mix=nn.Sequential(nn.Conv2d(75,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.up=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU())
        self.out=nn.Conv2d(32,9,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,features,points,valid):
        assert features.shape[1:]==(384,28,21)
        prior=prior_logits(points,valid)
        hints=F.interpolate(prior.exp(),size=(28,21),mode='bilinear',align_corners=False)
        y,x=torch.meshgrid(torch.linspace(-1,1,28,device=features.device),
                           torch.linspace(-1,1,21,device=features.device),indexing='ij')
        xy=torch.stack([x,y])[None].expand(len(features),-1,-1,-1)
        q=self.mix(torch.cat([self.project(features),hints,xy],dim=1))
        q=self.up(F.interpolate(q,size=(56,42),mode='bilinear',align_corners=False))
        residual=F.interpolate(self.out(q),size=(96,72),mode='bilinear',align_corners=False)
        return prior+residual


def loss(logits,target,valid):
    mask=valid.clone();mask[:,8]=False
    distribution,mask=target_distribution(target,mask)
    ce=-(distribution*logits.flatten(2).log_softmax(-1)).sum(-1)
    return (ce*mask).sum()/mask.sum().clamp_min(1)


def decode(logits):
    return local_mode(logits)
