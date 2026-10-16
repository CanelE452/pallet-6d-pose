"""Predict a bounded 2D vector from local point + optional nearby line evidence."""
from context18 import *
from model16 import geometry
from torch import nn
import torch.nn.functional as F

class VectorRefiner(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(18000+seed)
        self.project=nn.Conv2d(192,16,1);self.point_id=nn.Embedding(8,8)
        self.point_encode=nn.Sequential(nn.Linear(225,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.line_project=nn.Conv2d(192,8,1);self.line_encode=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.vector=nn.Sequential(nn.Linear(160,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.vector[-1].weight);nn.init.zeros_(self.vector[-1].bias)

    def forward(self,O,rep,arm='line_vector',line_off=False,line_features=None):
        base=rep['base'];b,p,_=base.shape;bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
        roi=2*(base-bb[:,None,:2])/wh[:,None]-1;xy=base/O['image_hw'].flip(-1)[:,None];amodal=(rep['amodal']-base)/diag[:,None,None]
        yy,xx=torch.meshgrid(base.new_tensor([-1,0,1]),base.new_tensor([-1,0,1]),indexing='ij');offset=torch.stack((xx.flatten(),yy.flatten()),-1)
        grid=roi[:,:,None]+offset[None,None]*(.06*diag[:,None,None,None]/wh[:,None,None])
        patch=F.grid_sample(self.project(O['features'])*O['content'],grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1).flatten(-2)
        pid=self.point_id(torch.arange(p,device=base.device))[None].expand(b,-1,-1)
        token=self.point_encode(torch.cat((patch,rep['context'],pid,rep['probabilities'],roi,xy,amodal),-1))
        g=geometry(O,rep,base[:,:,None]);sample=g['sample'][:,:,0];sg=2*(sample-bb[:,None,None,None,None,:2])/wh[:,None,None,None,None]-1
        f=O['features'] if line_features is None else line_features
        lp=F.grid_sample(self.line_project(f)*O['content'],sg.reshape(b,1,-1,2),align_corners=True,padding_mode='zeros')[:,:,0].transpose(1,2).reshape(b,p,3,7,3,8)
        hw=O['image_hw'].flip(-1)[:,None,None,None,None];inside=((sample>=0)&(sample<hw)).all(-1)&((sg>=-1)&(sg<=1)).all(-1)
        dist=(g['distance'][:,:,0]/diag[:,None,None])[:,:,:,None,None].expand(-1,-1,-1,7,-1)
        n=g['normal'][:,:,:,None].expand(-1,-1,-1,7,-1);along=g['along'][None,None,None,:,None].expand(b,p,3,-1,-1);assoc=g['association'][:,:,:,None,None].expand(-1,-1,-1,7,-1)
        lt=self.line_encode(torch.cat(((lp*inside[...,None]).flatten(-2),inside.float(),dist,n,along,assoc),-1))
        pooled=torch.cat((lt.mean(-2),lt.amax(-2)),-1).flatten(-2)
        if arm.startswith('point_') or line_off:pooled=torch.zeros_like(pooled)
        raw=self.vector(torch.cat((token,pooled),-1));radius=RADIUS*diag
        delta=raw/(1+raw.square().sum(-1,keepdim=True)).sqrt()*radius[:,None,None]
        return dict(points=base+delta,base=base,delta=delta,raw_vector=raw,radius=radius,sample_center=g['center'][:,:,0])

def objective(out,gt,valid,arm):
    error=(out['points']-gt).norm(dim=-1);before=(out['base']-gt).norm(dim=-1).detach()
    primary=error[valid].mean();harm=torch.relu(error-before)[valid].mean()
    return primary+(2*harm if arm.endswith('_risk') else 0),primary,harm
