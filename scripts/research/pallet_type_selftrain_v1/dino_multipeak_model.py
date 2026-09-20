"""GT-free separated heatmap modes; rank1 is the immutable original decoder."""
import torch
from torch.nn import functional as F
from . import dino_wide_model as W


def modes(logits,count=5,separation=8):
    if logits.shape[-2:]!=(W.GRID_H,W.GRID_W) or not torch.isfinite(logits).all():
        raise ValueError('Finite192x144 logits required')
    if not isinstance(count,int) or not 1<=count<=5 or separation<5:
        raise ValueError('At most5 modes; separated nonoverlapping5x5 windows')
    # Local maxima prevent suppressed peak shoulders being called new modes.
    maxima=logits==F.max_pool2d(logits,5,stride=1,padding=2)
    search=logits.masked_fill(~maxima,-torch.inf)
    yy=torch.arange(W.GRID_H,device=logits.device)[None,None,:,None]
    xx=torch.arange(W.GRID_W,device=logits.device)[None,None,None,:]
    logprob=logits.flatten(2).log_softmax(-1).reshape_as(logits)
    points=[];masses=[];values=[];valids=[];anchors=[]
    for _ in range(count):
        value,anchor=search.flatten(2).max(-1);valid=torch.isfinite(value)
        ax=anchor%W.GRID_W;ay=anchor//W.GRID_W
        window=((xx-ax[...,None,None]).abs()<=2)&((yy-ay[...,None,None]).abs()<=2)
        prob=logits.masked_fill(~window,-torch.inf).flatten(2).softmax(-1).reshape_as(logits)
        q=torch.stack([(prob*xx).sum((-2,-1)),(prob*yy).sum((-2,-1))],-1)*4
        mass=logprob.masked_fill(~window,-torch.inf).flatten(2).logsumexp(-1).exp()
        points.append(q);masses.append(torch.where(valid,mass,torch.zeros_like(mass)))
        values.append(torch.where(valid,value,torch.zeros_like(value)));valids.append(valid)
        anchors.append(torch.stack([ax,ay],-1))
        suppress=((xx-ax[...,None,None]).abs()<=separation)&((yy-ay[...,None,None]).abs()<=separation)
        search=search.masked_fill(suppress,-torch.inf)
    return dict(points=torch.stack(points,2),mass=torch.stack(masses,2),peak_logit=torch.stack(values,2),
        valid=torch.stack(valids,2),anchors=torch.stack(anchors,2))
