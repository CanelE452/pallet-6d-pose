"""Candidate-conditioned finite strips; forward never receives targets/masks."""
from context17 import *
from model15 import StableEvaluator,weights
from torch import nn
import torch.nn.functional as F

# Camera-facing 0123 contract, independently checked against build_eval_manifest.
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
HEIGHT={1,3,5,7}

def geometry(O,rep,cand):
    ix=rep['line_weights'][...,:-1].argsort(dim=-1,descending=True,stable=True)[...,:3]
    l=rep['lines'][:,None].expand(-1,8,-1,-1).gather(2,ix[...,None].expand(-1,-1,-1,3))
    l=l/l[...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8)
    sign=torch.where((l[...,0]<-1e-7)|((l[...,0].abs()<=1e-7)&(l[...,1]<0)),-1.,1.)
    l=l*sign[...,None];n=l[...,:2];t=torch.stack((-n[...,1],n[...,0]),-1)
    dist=(cand[:,:,:,None]*n[:,:,None]).sum(-1)+l[:,:,None,:,2]
    foot=cand[:,:,:,None]-dist[...,None]*n[:,:,None]
    diag=(O['box'][:,2:]-O['box'][:,:2]).norm(dim=-1).clamp_min(1)
    along=torch.linspace(-.35,.35,7,device=cand.device)
    center=foot[:,:,:,:,None]+along[None,None,None,None,:,None]*diag[:,None,None,None,None,None]*t[:,:,None,:,None]
    widths=cand.new_tensor([-.015,0,.015])
    sample=center[...,None,:]+widths[None,None,None,None,None,:,None]*diag[:,None,None,None,None,None,None]*n[:,:,None,:,None,None]
    association=rep['line_weights'][...,:-1].gather(-1,ix)
    return dict(center=center,sample=sample,normal=n,distance=dist,diag=diag,along=along,association=association,indices=ix)

class DetailEvaluator(StableEvaluator):
    def __init__(self,seed):
        super().__init__(seed)
        self.line_project=nn.Conv2d(192,8,1)
        self.rgb_project=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,8,3,padding=1),nn.SiLU())
        self.encode=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.roles=nn.Linear(16,4)
        self.line_score=nn.Sequential(nn.Linear(108,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.line_score[-1].weight);nn.init.zeros_(self.line_score[-1].bias)

    def forward(self,O,rep,arm='high_none',evidence_off=False,evidence_features=None):
        out=super().forward(O,rep,'balanced_rich');cand=out['candidates'];b,p,k,_=cand.shape
        g=geometry(O,rep,cand);sample=g['sample'];wh=(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1)
        grid=2*(sample-O['box'][:,None,None,None,None,None,:2])/wh[:,None,None,None,None,None]-1
        features=O['features'] if evidence_features is None else evidence_features
        z=self.line_project(features)*O['content']
        patch=F.grid_sample(z,grid.reshape(b,1,-1,2),align_corners=True,padding_mode='zeros')[:, :,0].transpose(1,2).reshape(b,p,k,3,7,3,8)
        rgb=O['rgb']
        if arm.startswith('low_'):rgb=F.interpolate(F.interpolate(rgb,size=24,mode='area'),size=96,mode='bilinear',align_corners=True)
        rz=self.rgb_project(rgb)
        rp=F.grid_sample(rz,grid.reshape(b,1,-1,2),align_corners=True,padding_mode='zeros')[:,:,0].transpose(1,2).reshape(b,p,k,3,7,3,8)
        patch=patch+rp
        hw=O['image_hw'].flip(-1)[:,None,None,None,None,None]
        inside=((sample>=0)&(sample<hw)).all(-1)&((grid>=-1)&(grid<=1)).all(-1)
        patch=patch*inside[...,None]
        dist=(g['distance']/g['diag'][:,None,None,None])[:,:,:,:,None,None].expand(-1,-1,-1,-1,7,1)
        n=g['normal'][:,:,None,:,None].expand(-1,-1,k,-1,7,-1)
        along=g['along'][None,None,None,None,:,None].expand(b,p,k,3,-1,-1)
        assoc=g['association'][:,:,None,:,None,None].expand(-1,-1,k,-1,7,-1)
        desc=torch.cat((patch.flatten(-2),inside.float(),dist,n,along,assoc),-1)
        token=self.encode(desc);logits=self.roles(token)
        pooled=torch.cat((token.mean(-2),token.amax(-2),logits.softmax(-1).mean(-2)),-1).flatten(-2)
        if evidence_off:pooled=torch.zeros_like(pooled)
        raw=self.line_score(pooled)[...,0];gain=out['gain']+raw-raw[...,:1]
        choice=gain.argmax(-1);points=cand.gather(2,choice[...,None,None].expand(-1,-1,1,2))[:,:,0]
        out.update(points=points,gain=gain,choice=choice,role_logits=logits,role_prob=logits.softmax(-1).mean(-2),sample_center=g['center'])
        return out

@torch.no_grad()
def role_targets(O,rep,gt,valid,masks):
    """Finite annotation cuboid-edge proxy, not a true visible mesh boundary GT."""
    x=geometry(O,rep,candidates(O,rep))['center'];edges=torch.tensor(EDGES,device=gt.device)
    a=gt[:,edges[:,0]];d=gt[:,edges[:,1]]-a
    v=x[...,None,:]-a[:,None,None,None,None]
    u=(v*d[:,None,None,None,None]).sum(-1)/d.square().sum(-1).clamp_min(1e-8)[:,None,None,None,None]
    err=(v-u.clamp(0,1)[...,None]*d[:,None,None,None,None]).norm(dim=-1)
    distance,nearest=err.min(-1)
    incident=((edges[nearest]==torch.arange(8,device=gt.device)[None,:,None,None,None,None]).any(-1))
    vertical=torch.zeros(12,dtype=torch.bool,device=gt.device);vertical[list(HEIGHT)]=True
    label=torch.where(incident,torch.where(vertical[nearest],2,1),3)
    hw=O['image_hw'].flip(-1)[:,None,None,None,None]
    inside=((x>=0)&(x<hw)).all(-1)
    covered=((x[...,0]>=masks[:,None,None,None,None,0])&(x[...,0]<masks[:,None,None,None,None,2])&(x[...,1]>=masks[:,None,None,None,None,1])&(x[...,1]<masks[:,None,None,None,None,3]))
    label=torch.where((distance<=2.5)&inside&~covered,label,0)
    label[~valid.all(-1)]=-100
    return label

def objective(out,truth,valid,w,arm,labels,role_weight):
    error=truth-out['gain'];pinball=torch.maximum(.2*error,-.8*error)
    mask=valid[:,:,None].expand_as(pinball).clone();mask[...,0]=False
    gain=pinball[mask].mean();aux=out['gain'].new_zeros(())
    if not arm.endswith('_none') and (labels!=-100).any():
        aux=F.cross_entropy(out['role_logits'].reshape(-1,4),labels.flatten(),weight=role_weight,ignore_index=-100)
    return gain+.25*aux,gain,aux

@torch.no_grad()
def revised_targets(O,rep,gt,valid,masks,boundary,has_mask):
    proxy=role_targets(O,rep,gt,valid,masks);proxy[~has_mask]=-100
    center=geometry(O,rep,candidates(O,rep))['center'];b=center.shape[0];bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1)
    grid=2*(center-bb[:,None,None,None,None,:2])/wh[:,None,None,None,None]-1
    distance=F.grid_sample(boundary[:,None],grid.reshape(b,1,-1,2),align_corners=True,padding_mode='border')[:,0,0].reshape(proxy.shape)
    inside=((grid>=-1)&(grid<=1)).all(-1)
    # Outside the ROI: unknown instead of falsely asserting no mask boundary.
    proxy[~inside]=-100
    mask=proxy.clone();mask[(mask>0)&(distance>2.5)]=0
    return proxy,mask
