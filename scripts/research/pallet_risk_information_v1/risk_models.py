"""Prediction-only fixed architectures and losses; no target/reference I/O."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (20261018, 20261019, 20261020)
F0_DIM = 188
ARMS = ('R_CONT_F0', 'R_ORDER_F0', 'R_ORDER_F1_ZERO', 'R_ORDER_F1', 'V_ORDER_NULL', 'V_ORDER_RGB')


class RiskModel(nn.Module):
    def __init__(self, arm, extra_dim=0):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm = arm
        self.extra_dim = int(extra_dim) if arm not in ('R_CONT_F0','R_ORDER_F0') else 0
        if self.extra_dim < 0: raise ValueError('Negative extra block width')
        self.visual = arm.startswith('V_')
        if self.visual:
            self.local_cnn = nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.ReLU(),
                nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        width = F0_DIM + self.extra_dim + (340 if self.visual else 0)
        self.head = nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),
            nn.Linear(64,2 if arm=='R_CONT_F0' else 1))

    def forward(self, f0, extra=None, patches=None, region_validity=None):
        if f0.ndim != 2 or f0.shape[1] != F0_DIM: raise ValueError('Expected actual actions x188')
        blocks = [f0]
        if self.extra_dim:
            if extra is None or extra.shape != (len(f0),self.extra_dim): raise ValueError('F1 block alignment/width mismatch')
            blocks.append(torch.zeros_like(extra) if self.arm=='R_ORDER_F1_ZERO' else extra)
        if self.visual:
            if patches is None or tuple(patches.shape)!=(len(f0),20,4,16,16): raise ValueError('Expected candidate chunk x20x4x16x16')
            if region_validity is None or tuple(region_validity.shape)!=(len(f0),20): raise ValueError('Expected20 region-validity values per action')
            if self.arm=='V_ORDER_NULL': patches=torch.cat((torch.zeros_like(patches[:,:,:3]),patches[:,:,3:]),dim=2)
            encoded=self.local_cnn(patches.reshape(-1,4,16,16)).reshape(len(f0),320)
            blocks.extend((encoded,region_validity))
        value=self.head(torch.cat(blocks,dim=-1))
        return F.softplus(value) if self.arm=='R_CONT_F0' else value.squeeze(-1)


def parameters(arm, extra_dim=0):
    return sum(p.numel() for p in RiskModel(arm,extra_dim).parameters())


def choose(output, arm):
    """Input actions already have original H1 N0-first/physical-number order."""
    values=np.asarray(output,dtype=float)
    if not len(values): return None
    if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite model output')
    if arm=='R_CONT_F0':
        if values.ndim!=2 or values.shape[1]!=2: raise ValueError('Continuous head requires two axes')
        return int(np.argmin(values.max(axis=1)))
    if values.ndim!=1: raise ValueError('Rank head requires one score per action')
    return int(np.argmax(values))


def continuous_frame_loss(output, target, validity):
    """Finite axes -> valid action means -> one frame, never padded actions."""
    if output.shape!=target.shape or target.shape!=validity.shape: raise ValueError('Continuous target/axis mask shape mismatch')
    counts=validity.sum(-1); included=counts>0
    if not bool(included.any()): return None
    safe=torch.where(validity,target,torch.zeros_like(target))
    primitive=F.smooth_l1_loss(output,safe,reduction='none',beta=1.)
    means=(primitive*validity).sum(-1)/counts.clamp_min(1)
    return means[included].mean()


def order_frame_loss(scores, pair_groups, chunk_size=4096):
    """Mean within each relation type, then mean over present relation types."""
    groups=[]
    for pairs in pair_groups:
        if len(pairs)==0: continue
        index=torch.as_tensor(pairs,dtype=torch.long,device=scores.device)
        total=scores.sum()*0.
        for first in range(0,len(index),chunk_size):
            part=index[first:first+chunk_size]
            total=total+F.softplus(scores[part[:,1]]-scores[part[:,0]]).sum()
        groups.append(total/len(index))
    return torch.stack(groups).mean() if groups else None


def mean_frame_losses(losses):
    valid=[value for value in losses if value is not None]
    return (torch.stack(valid).mean(),len(valid)) if valid else (None,0)
