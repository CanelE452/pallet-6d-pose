"""Conditional lower-quantile gain; optional larger geometric line candidates."""
import torch
from torch import nn
import torch.nn.functional as F
from context15 import candidates

class StableEvaluator(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(15000+seed)
        self.project=nn.Conv2d(192,16,1);self.point_id=nn.Embedding(8,8);self.candidate_id=nn.Embedding(8,8)
        self.score=nn.Sequential(nn.Linear(384,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,O,rep,arm='balanced_rich',line_off=False):
        cand=candidates(O,rep);b,p,k,_=cand.shape;bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
        roi=2*(cand-bb[:,None,None,:2])/wh[:,None,None]-1;xy=cand/O['image_hw'].flip(-1)[:,None,None]
        delta=(cand-rep['base'][:,:,None])/diag[:,None,None,None]
        geom=torch.cat((roi,xy,delta,delta.norm(dim=-1,keepdim=True),rep['probabilities'][:,:,None].expand(-1,-1,k,-1)),-1)
        z=self.project(O['features'])*O['content'];y,x=torch.meshgrid(cand.new_tensor([-1,0,1]),cand.new_tensor([-1,0,1]),indexing='ij');offset=torch.stack((x.flatten(),y.flatten()),-1)
        grid=roi[:,:,:,None]+offset[None,None,None]*(.06*diag[:,None,None,None,None]/wh[:,None,None,None])
        patch=F.grid_sample(z,grid.reshape(b,p,k*9,2),align_corners=True,padding_mode='zeros').permute(0,2,3,1).reshape(b,p,k,144)
        pid=self.point_id(torch.arange(p,device=cand.device))[None,:,None].expand(b,-1,k,-1)
        cid=self.candidate_id(torch.arange(k,device=cand.device))[None,None].expand(b,p,-1,-1)
        ld=rep['line_descriptor'][:,:,None].expand(-1,-1,k,-1)
        if line_off:ld=torch.zeros_like(ld)
        desc=torch.cat((patch,patch-patch[:,:,:1],rep['context'][:,:,None].expand(-1,-1,k,-1),geom,pid,cid,ld),-1)
        raw=self.score(desc)[...,0];gain=raw-raw[...,:1];eligible=torch.ones_like(gain,dtype=torch.bool)
        if arm!='balanced_rich':eligible[...,5:]=False
        if line_off:eligible[...,3:]=False
        choice=gain.masked_fill(~eligible,-torch.inf).argmax(-1)
        points=cand.gather(2,choice[...,None,None].expand(-1,-1,1,2))[:,:,0]
        return dict(points=points,candidates=cand,gain=gain,choice=choice,eligible=eligible)

def weights(truth,valid,kind):
    # All valid observations retained; rare benefit/harm/near-neutral groups reweighted.
    bucket=torch.where(truth<-.25,0,torch.where(truth>.25,2,1));w=torch.ones_like(truth);counts=[]
    for typ in range(3):
        for candidate in range(1,truth.shape[-1]):
            mask=valid&(kind==typ);n=mask.sum();present=[int(((bucket[...,candidate]==j)&mask).sum()) for j in range(3)]
            nonempty=sum(c>0 for c in present)
            if nonempty:
                for j,count in enumerate(present):
                    if count:w[...,candidate][mask&(bucket[...,candidate]==j)]=min(10.,float(n)/(nonempty*count))
            counts.append(dict(type=typ,candidate=candidate,harm_neutral_benefit=present))
    return w,counts

def objective(out,truth,valid,weight,arm):
    tau=.5 if arm=='median_old' else .2
    error=truth-out['gain'];pinball=torch.maximum(tau*error,(tau-1)*error)
    mask=valid[:,:,None]&out['eligible'];mask[...,0]=False
    w=weight if arm.startswith('balanced') else torch.ones_like(weight)
    return (pinball*w)[mask].sum()/w[mask].sum().clamp_min(1)
