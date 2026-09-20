"""Score complete eight-corner hypotheses, never a mix of their corners."""
import numpy as np
import torch
from torch import nn
from . import dino_evidence_gate_model as G
from . import dino_wide_model as W

Q=np.array([4,0,3,7,5,1,2,6,8]);PERMS=np.stack([np.arange(9),Q[Q],Q,Q[Q[Q]]])
NAMES=[f'{family}_{rotation}' for family in ['R0','SYN','MIX'] for rotation in ['native','half','quarter','three_quarter']]
DIM=98


def describe(logits_syn,logits_mix,old,diag):
    """All descriptors depend only on image predictions and input coordinates."""
    syn=W.decode(logits_syn);mix=W.decode(logits_mix)
    candidates=torch.stack([q[:,p] for q in [old,syn,mix] for p in PERMS],1)
    sm,se=G.mass_map(logits_syn);mm,me=G.mass_map(logits_mix)
    sl0=G.sample(sm,old)[:,:8];ml0=G.sample(mm,old)[:,:8]
    scale=diag[:,None,None].clamp_min(1e-6);center=old[:,:8].mean(1,keepdim=True)
    values=[]
    for q in candidates.unbind(1):
        sl=G.sample(sm,q)[:,:8];ml=G.sample(mm,q)[:,:8]
        inside=((q[:,:8]>=0).all(-1)&(q[:,:8,0]<=572)&(q[:,:8,1]<=764)).float()
        distance=torch.cdist(q[:,:8],q[:,:8]);distance=distance.masked_fill(torch.eye(8,device=q.device,dtype=torch.bool)[None],torch.inf)
        values.append(torch.cat([sl,ml,sl-sl0,ml-ml0,((q[:,:8]-old[:,:8])/scale).flatten(1),
            ((q[:,:8]-center)/scale).flatten(1),inside,(q[:,:8]-syn[:,:8]).norm(dim=-1)/diag[:,None],
            (q[:,:8]-mix[:,:8]).norm(dim=-1)/diag[:,None],se[:,:8].mean(1,keepdim=True),me[:,:8].mean(1,keepdim=True),
            distance.amin(-1)/diag[:,None]],1))
    x=torch.stack(values,1);assert x.shape[-1]==DIM and torch.isfinite(x).all()
    return x,candidates


class Ranker(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def choose(probability,threshold):
    s=np.asarray(probability).copy();s[...,0]=-1
    j=s.argmax(-1);best=np.take_along_axis(s,j[...,None],axis=-1)[...,0]
    return np.where(best>=threshold,j,0)


def canonical_errors(candidates,gt,valid,perms,gain):
    """Supervision/scoring only: one approved whole symmetry per hypothesis."""
    result=[]
    for q in np.asarray(candidates):
        variants=[]
        for perm in np.asarray(perms,int):
            m=np.asarray(valid,bool)[perm].copy();m[8]=False
            e=np.linalg.norm(q-np.asarray(gt)[perm],axis=-1)/gain
            canonical=np.full(9,np.nan);canonical[perm[m]]=e[m]
            variants.append(canonical[:8])
        result.append(min(variants,key=lambda a:np.nanmean(a)))
    return np.asarray(result)


def beneficial(errors):
    e=np.asarray(errors);before=e[:,0:1]
    damage=((before<5)&(e>10)).sum(-1)
    return (np.nanmean(e,axis=-1)+5<np.nanmean(before,axis=-1))&(damage==0)


def stats(errors,choice):
    e=np.asarray(errors);before=e[:,0];after=e[np.arange(len(e)),choice];mask=np.isfinite(before)
    s=G.stats(before[mask],after[mask]);s['input_PCK20']=float((before[mask]<=20).mean());s['PCK20']=float((after[mask]<=20).mean())
    return s


def calibrate(probability,errors,clean):
    good=beneficial(errors);table=[];best=None
    for t in [*np.linspace(0,1,101),1.01]:
        choice=choose(probability,t);use=choice!=0
        precision=float(good[np.arange(len(good)),choice][use].mean()) if use.any() else 1.
        s=stats(errors[clean],choice[clean]);allstats=stats(errors,choice)
        feasible=s['PCK10']>=s['input_PCK10']-.01 and s['PCK20']>=s['input_PCK20'] and s['P90']<=1.1*s['input_P90'] and s['damaged']<=.01*s['good'] and precision>=.95
        row=dict(threshold=float(t),selected=int(use.sum()),beneficial_precision=precision,clean=s,combined=allstats,feasible=bool(feasible));table.append(row)
        key=(s['recovered'],allstats['recovered'],float(t))
        if feasible and (best is None or key>(best['clean']['recovered'],best['combined']['recovered'],best['threshold'])):best=row
    assert best is not None
    return best,table
