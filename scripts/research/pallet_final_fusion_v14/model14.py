"""Candidate-relative image evidence with point/candidate identities and optional lines."""
import torch
from torch import nn
import torch.nn.functional as F

class FusionEvaluator(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(14000+seed)
        self.project=nn.Conv2d(192,16,1);self.point_id=nn.Embedding(8,8);self.candidate_id=nn.Embedding(5,8)
        self.score=nn.Sequential(nn.Linear(384,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,O,rep,arm='heldout_fusion',line_off=False):
        from context14 import candidates
        cand=candidates(rep);b,p,k,_=cand.shape;bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
        roi=2*(cand-bb[:,None,None,:2])/wh[:,None,None]-1
        xy=cand/O['image_hw'].flip(-1)[:,None,None];delta=(cand-rep['base'][:,:,None])/diag[:,None,None,None]
        geom=torch.cat((roi,xy,delta,delta.norm(dim=-1,keepdim=True),rep['probabilities'][:,:,None].expand(-1,-1,k,-1)),-1)
        z=self.project(O['features'])*O['content']
        y,x=torch.meshgrid(cand.new_tensor([-1,0,1]),cand.new_tensor([-1,0,1]),indexing='ij');offset=torch.stack((x.flatten(),y.flatten()),-1)
        grid=roi[:,:,:,None]+offset[None,None,None]*(.06*diag[:,None,None,None,None]/wh[:,None,None,None])
        patch=F.grid_sample(z,grid.reshape(b,p,k*9,2),align_corners=True,padding_mode='zeros').permute(0,2,3,1).reshape(b,p,k,144)
        pid=self.point_id(torch.arange(p,device=cand.device))[None,:,None].expand(b,-1,k,-1)
        cid=self.candidate_id(torch.arange(k,device=cand.device))[None,None].expand(b,p,-1,-1)
        ld=rep['line_descriptor'][:,:,None].expand(-1,-1,k,-1)
        disabled=arm=='heldout_points' or line_off
        if disabled:ld=torch.zeros_like(ld)
        desc=torch.cat((patch,patch-patch[:,:,:1],rep['context'][:,:,None].expand(-1,-1,k,-1),geom,pid,cid,ld),-1)
        raw=self.score(desc)[...,0];gain=raw-raw[...,:1]
        eligible=torch.ones_like(gain,dtype=torch.bool)
        if disabled:eligible[...,3:]=False
        choice=gain.masked_fill(~eligible,-torch.inf).argmax(-1)
        points=cand.gather(2,choice[...,None,None].expand(-1,-1,1,2))[:,:,0]
        return dict(points=points,candidates=cand,gain=gain,choice=choice,eligible=eligible)

def objective(out,gt,valid):
    err=(out['candidates']-gt[:,:,None]).norm(dim=-1);truth=(err[...,:1]-err).detach()
    mask=valid[:,:,None]&out['eligible'];mask[...,0]=False
    regression=F.smooth_l1_loss(out['gain'][mask],truth[mask],beta=1)
    predicted=out['gain'].masked_fill(~out['eligible'],-1e4)
    preference=truth.masked_fill(~out['eligible'],-1e4).softmax(-1)
    ranking=-(preference*predicted.log_softmax(-1)).sum(-1)[valid].mean()
    return regression+.25*ranking,regression,ranking
