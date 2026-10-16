"""Shared candidate gain scoring head; frozen candidate0 is an exact zero reference."""
import torch
from torch import nn
import torch.nn.functional as F

class GainScorer(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(13000+seed)
        self.project=nn.Conv2d(192,32,1)
        self.score=nn.Sequential(nn.Linear(362,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
        self.constant=nn.Parameter(torch.zeros(3))

    def forward(self,O,rep,points,arm):
        b,p,k,_=points.shape;box=O['box'];wh=(box[:,2:]-box[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
        roi=2*(points-box[:,None,None,:2])/wh[:,None,None]-1
        image_xy=points/O['image_hw'].flip(-1)[:,None,None]
        displacement=(points-rep['base'][:,:,None])/diag[:,None,None,None]
        size=displacement.norm(dim=-1,keepdim=True)
        probability=rep['probabilities'][:,:,None].expand(-1,-1,k,-1)
        if arm=='image':
            z=self.project(O['features'])*O['content']
            y,x=torch.meshgrid(points.new_tensor([-1,0,1]),points.new_tensor([-1,0,1]),indexing='ij')
            offsets=torch.stack((x.flatten(),y.flatten()),-1)
            grid=roi[:,:,:,None]+offsets[None,None,None]*(.06*diag[:,None,None,None,None]/wh[:,None,None,None])
            local=F.grid_sample(z,grid.reshape(b,p,k*9,2),align_corners=True,padding_mode='zeros').permute(0,2,3,1).reshape(b,p,k,9*32)
            context=rep['context'][:,:,None].expand(-1,-1,k,-1)
        else:
            local=points.new_zeros(b,p,k,288);context=points.new_zeros(b,p,k,64)
        desc=torch.cat((local,roi,image_xy,displacement,size,probability,context),-1)
        raw=self.constant[None,None].expand(b,p,-1) if arm=='constant' else self.score(desc)[...,0]
        gain=raw-raw[...,:1]
        choice=gain.argmax(-1)
        output=points.gather(2,choice[:,:,None,None].expand(-1,-1,1,2))[:,:,0]
        return dict(points=output,gain=gain,choice=choice)

def objective(predicted,truth,valid):
    regression=F.smooth_l1_loss(predicted[...,1:][valid],truth[...,1:][valid],beta=1)
    # Candidate qualities have a pixel scale; soft target avoids arbitrary tie labels.
    preference=truth.detach().softmax(-1)
    ranking=-(preference*predicted.log_softmax(-1)).sum(-1)[valid].mean()
    return regression+.25*ranking,regression,ranking
