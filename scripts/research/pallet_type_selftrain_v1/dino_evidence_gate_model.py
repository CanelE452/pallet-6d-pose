"""GT-free image-evidence features and a small learned per-corner acceptor."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import dino_wide_model as W

NAMES=['own_log_old','own_log_new','own_log_ratio','other_log_old','other_log_new','other_log_ratio',
       'own_entropy','other_entropy','move_over_boxdiag','models_distance_over_boxdiag',
       'own_peak_logmass','other_peak_logmass','other_relative_new_logmass','old_inside','new_border_over_boxdiag','input_valid']


def mass_map(logits):
    b,k,h,w=logits.shape
    prob=logits.flatten(2).softmax(-1).reshape_as(logits)
    mass=F.avg_pool2d(prob.reshape(b*k,1,h,w),5,stride=1,padding=2)*25
    entropy=-(prob*prob.clamp_min(1e-12).log()).sum((-1,-2))/np.log(h*w)
    return mass.reshape(b,k,h,w),entropy


def sample(mass,points):
    b,k,h,w=mass.shape
    grid=points/points.new_tensor([4*(w-1),4*(h-1)])*2-1
    return F.grid_sample(mass.reshape(b*k,1,h,w),grid.reshape(b*k,1,1,2),align_corners=True,padding_mode='zeros').reshape(b,k).clamp_min(1e-12).log()


def features(logits,other_logits,old,valid,boxdiag_crop):
    """No targets, visibility GT, labels or metrics enter this function."""
    new=W.decode(logits);other=W.decode(other_logits)
    ownmass,entropy=mass_map(logits);othermass,otherentropy=mass_map(other_logits)
    safe=torch.where(torch.isfinite(old),old,torch.zeros_like(old))
    a=sample(ownmass,safe);b=sample(ownmass,new);c=sample(othermass,safe);d=sample(othermass,new)
    diag=boxdiag_crop[:,None].clamp_min(1e-6)
    move=(new-safe).norm(dim=-1)/diag;agreement=(new-other).norm(dim=-1)/diag
    peak=ownmass.flatten(2).amax(-1).clamp_min(1e-12).log()
    otherpeak=othermass.flatten(2).amax(-1).clamp_min(1e-12).log()
    inside=torch.isfinite(old).all(-1)&(old>=0).all(-1)&(old[:,:,0]<=4*(W.GRID_W-1))&(old[:,:,1]<=4*(W.GRID_H-1))
    border=torch.stack([new[:,:,0],new[:,:,1],4*(W.GRID_W-1)-new[:,:,0],4*(W.GRID_H-1)-new[:,:,1]],-1).amin(-1)/diag
    x=torch.stack([a,b,b-a,c,d,d-c,entropy,otherentropy,move,agreement,peak,otherpeak,d-otherpeak,inside.float(),border,valid.float()],-1)
    assert x.shape[-1]==len(NAMES) and torch.isfinite(x).all()
    return x,new


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(len(NAMES),32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,1))

    def forward(self,x):return self.net(x).squeeze(-1)


def choose(old,new,score,threshold,valid):
    mask=np.asarray(valid,bool).copy()&np.isfinite(score)&(np.asarray(score)>=threshold)
    mask[...,8]=False
    out=np.array(old,copy=True);out[mask]=np.asarray(new)[mask]
    return out,mask


def stats(before,after):
    a=np.asarray(before);b=np.asarray(after)
    return dict(n=len(a),input_PCK10=float((a<=10).mean()),PCK10=float((b<=10).mean()),
        input_P90=float(np.quantile(a,.9)),P90=float(np.quantile(b,.9)),
        hard=int((a>20).sum()),recovered=int(((a>20)&(b<=10)).sum()),
        good=int((a<5).sum()),damaged=int(((a<5)&(b>10)).sum()))


def calibrate(scores,before,after,clean):
    """Fixed grid, source-only. Null action is an explicit feasible fallback."""
    scores=np.asarray(scores);before=np.asarray(before);after=np.asarray(after);clean=np.asarray(clean,bool)
    rows=[];best=None
    for t in [*np.linspace(0,1,101),1.01]:
        use=scores>=t;out=np.where(use,after,before);s=stats(before[clean],out[clean])
        correct=(after<=10)&(after+5<before)
        precision=float(correct[use].mean()) if use.any() else 1.
        feasible=(s['PCK10']>=s['input_PCK10']-.01 and s['P90']<=s['input_P90']*1.1 and
                  s['damaged']<=.01*s['good'] and precision>=.95)
        recovered=int(((before>20)&(out<=10)).sum())
        r=dict(threshold=float(t),accepted=int(use.sum()),beneficial_precision=precision,feasible=bool(feasible),
               combined_clean_and_artificial_recovered=recovered,clean=s)
        rows.append(r)
        if feasible and (best is None or (s['recovered'],recovered,float(t))>(best['clean']['recovered'],best['combined_clean_and_artificial_recovered'],best['threshold'])):best=r
    assert best is not None
    return best,rows
