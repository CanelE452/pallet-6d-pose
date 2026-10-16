"""Equal 36-sample observed area/line context, query-specific soft retention."""
from context19 import *
from torch import nn
import torch.nn.functional as F

def observed_samples(O,rep,mode):
    base=rep['base'];b,p,_=base.shape;bb=O['box'];hw=O['image_hw'].flip(-1);low=bb[:,:2].clamp_min(0);high=torch.minimum(bb[:,2:],hw);rect_ok=(high>low).all(-1);high=torch.maximum(high,low);span=high-low
    if mode=='area':
        z=(torch.arange(6,device=base.device,dtype=base.dtype)+.5)/6;y,x=torch.meshgrid(z,z,indexing='ij');uv=torch.stack((x.flatten(),y.flatten()),-1)
        pos=(low[:,None]+uv[None]*span[:,None])[:,None].expand(-1,p,-1,-1);valid=rect_ok[:,None,None].expand(-1,p,36);normal=pos.new_zeros((b,p,36,2));assoc=pos.new_zeros((b,p,36,1))
    else:
        idx=rep['line_weights'][...,:-1].argsort(dim=-1,descending=True,stable=True)[...,:3];line=rep['lines'][:,None].expand(-1,p,-1,-1).gather(2,idx[...,None].expand(-1,-1,-1,3));norm=line[...,:2].norm(dim=-1);line=line/norm.clamp_min(1e-8)[...,None]
        sign=torch.where((line[...,0]<-1e-7)|((line[...,0].abs()<=1e-7)&(line[...,1]<0)),-1.,1.);line=line*sign[...,None];n=line[...,:2];t=torch.stack((-n[...,1],n[...,0]),-1);center=((low+high)/2)[:,None,None];foot=center-((center*n).sum(-1)+line[...,2])[...,None]*n
        parallel=t.abs()<1e-7;safe=torch.where(parallel,torch.ones_like(t),t);a=(low[:,None,None]-foot)/safe;c=(high[:,None,None]-foot)/safe;amin=torch.minimum(a,c);amax=torch.maximum(a,c);amin=torch.where(parallel,torch.full_like(a,-1e8),amin);amax=torch.where(parallel,torch.full_like(a,1e8),amax)
        enter=amin.amax(-1);leave=amax.amin(-1);parallel_ok=(~parallel|((foot>=low[:,None,None]-1e-5)&(foot<=high[:,None,None]+1e-5))).all(-1);ok=(leave>=enter)&parallel_ok&rect_ok[:,None,None]&(norm>1e-8)
        enter=torch.where(ok,enter,torch.zeros_like(enter));leave=torch.where(ok,leave,torch.zeros_like(leave));u=(torch.arange(12,device=base.device,dtype=base.dtype)+.5)/12;pos=foot[...,None,:]+(enter[...,None]+u*(leave-enter)[...,None])[...,None]*t[...,None,:];pos=pos.flatten(2,3)
        valid=ok[...,None].expand(-1,-1,-1,12).flatten(2,3);normal=n[...,None,:].expand(-1,-1,-1,12,-1).flatten(2,3);assoc=rep['line_weights'][...,:-1].gather(-1,idx)[...,None,None].expand(-1,-1,-1,12,-1).flatten(2,3)
    valid=valid&((pos>=low[:,None,None]-1e-4)&(pos<=high[:,None,None]+1e-4)).all(-1)
    return dict(pos=pos,valid=valid,normal=normal,assoc=assoc)

class RetentionGate(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(19000+seed)
        self.project=nn.Conv2d(192,16,1);self.point_id=nn.Embedding(8,8);self.point_encode=nn.Sequential(nn.Linear(228,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.evidence_project=nn.Conv2d(192,24,1);self.evidence_encode=nn.Sequential(nn.Linear(30,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU());self.query=nn.Linear(64,32);self.gate=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1));nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
    def forward(self,O,rep,arm='line_gate',evidence_off=False,evidence_features=None):
        base=rep['base'];delta=rep['proposal_delta'];b,p,_=base.shape;bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1);roi=2*(base-bb[:,None,:2])/wh[:,None]-1
        yy,xx=torch.meshgrid(base.new_tensor([-1,0,1]),base.new_tensor([-1,0,1]),indexing='ij');offset=torch.stack((xx.flatten(),yy.flatten()),-1);grid=roi[:,:,None]+offset[None,None]*(.06*diag[:,None,None,None]/wh[:,None,None]);patch=F.grid_sample(self.project(O['features'])*O['content'],grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1).flatten(-2);dn=delta/diag[:,None,None]
        desc=torch.cat((patch,rep['context'],self.point_id(torch.arange(p,device=base.device))[None].expand(b,-1,-1),rep['probabilities'],roi,base/O['image_hw'].flip(-1)[:,None],(rep['amodal']-base)/diag[:,None,None],dn,dn.norm(dim=-1,keepdim=True)),-1);q=self.point_encode(desc)
        sample=observed_samples(O,rep,'area' if arm=='area_gate' else 'line');pos=sample['pos'];valid=sample['valid'];sg=2*(pos-bb[:,None,None,:2])/wh[:,None,None]-1;features=O['features'] if evidence_features is None else evidence_features
        sampled=F.grid_sample(self.evidence_project(features)*O['content'],sg,align_corners=True,padding_mode='zeros').permute(0,2,3,1);relative=(pos-base[:,:,None])/diag[:,None,None,None]
        tokens=self.evidence_encode(torch.cat((sampled*valid[...,None],relative,sample['normal'],sample['assoc'],valid[...,None].float()),-1));logits=(self.query(q)[:,:,None]*tokens).sum(-1)/(32**.5);weights=logits.masked_fill(~valid,-1e4).softmax(-1)*valid;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9);evidence=(weights[...,None]*tokens).sum(-2)
        if arm=='local_gate' or evidence_off:evidence=torch.zeros_like(evidence)
        alpha=self.gate(torch.cat((q,evidence),-1)).sigmoid()[...,0];points=base+alpha[...,None]*delta
        return dict(points=points,base=base,proposal_delta=delta,alpha=alpha,coverage=valid.float().mean(-1),sample_pos=pos,sample_valid=valid,attention=weights)

def oracle_alpha(base,delta,gt):
    return (((gt-base)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)).clamp(0,1)

def objective(out,gt,valid):
    target=oracle_alpha(out['base'],out['proposal_delta'],gt).detach();movement=out['proposal_delta'].norm(dim=-1).detach();loss=((out['alpha']-target).abs()*movement)[valid].mean();error=(out['points']-gt).norm(dim=-1)[valid].mean()
    return loss,error,target
