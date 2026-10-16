"""Point expert, two independently trained DHT experts, and point-wise risk routing.

The line decoder never receives predicted point coordinates. Its positions are
convex combinations of incident-role line intersections; it cannot regress an
arbitrary residual from the original point. Structural roles are cuboid proxies.
"""
import torch
from torch import nn
import torch.nn.functional as F
from plpose_local.geometry import EDGES
from plpose_local.heads import decode_modes,roi_lines_to_raw
from plpose_local.losses import line_targets
from model import GuidedDHT,point_maps

INCIDENT=[[i for i,e in enumerate(EDGES) if p in e] for p in range(8)]
ROLE_PAIRS=[(a,b) for a in range(3) for b in range(a+1,3)]

def diagonal(O):return (O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1)

def geometry(O):
    bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1)
    return (bb[:,:2]+bb[:,2:])/2,wh/2

class LineExpert(nn.Module):
    def __init__(self,checkpoint,seed,mode):
        super().__init__();self.dht=GuidedDHT(checkpoint,seed);self.mode=mode

    def forward(self,O,decode=False):
        # Independent arm does not even read O['points']; same gate parameters.
        maps=point_maps(O['points'],O['box']) if self.mode=='guided' else O['features'].new_zeros(len(O['box']),3,24,24)
        out=self.dht(O,maps,decode=False)
        if decode:
            ll,mass=decode_modes(out['logits'],self.dht.lattice,self.dht.thetas,self.dht.rhos,3)
            out.update(lines=ll,mass=mass,raw_lines=roi_lines_to_raw(ll,O['box']))
        return out

def line_loss(out,O,gt,valid):
    t,ok=line_targets(gt,O['box'],36,65)
    u,v=torch.tensor(EDGES,device=gt.device).T;ok=ok&valid[:,u]&valid[:,v]
    z=-out['logits'].log_softmax(-1).gather(-1,t[...,None])[...,0]
    return z[ok].mean() if ok.any() else out['logits'].sum()*0

class PointContext(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Conv2d(192,16,1);self.ident=nn.Embedding(8,8)
        self.encode=nn.Sequential(nn.Linear(9*16+16+8+2+2+3,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())

    def forward(self,O):
        z=self.project(O['features'])*O['content'];center,half=geometry(O);p=(O['points']-center[:,None])/half[:,None]
        axis=p.new_tensor([-.06,0,.06]);yy,xx=torch.meshgrid(axis,axis,indexing='ij');off=torch.stack((xx.flatten(),yy.flatten()),-1)
        grid=p[:,:,None]+off;raw=center[:,None,None]+grid*half[:,None,None]
        inside=((raw>=0)&(raw<O['image_hw'].flip(-1)[:,None,None])).all(-1)&(grid.abs()<=1).all(-1)
        patch=F.grid_sample(z,grid,padding_mode='zeros',align_corners=True).permute(0,2,3,1)*inside[...,None]
        gl=z.mean((-2,-1))[:,None].expand(-1,8,-1);ident=self.ident(torch.arange(8,device=z.device))[None].expand(len(z),-1,-1)
        return self.encode(torch.cat((patch.flatten(-2),gl,ident,p,O['points']/O['image_hw'].flip(-1)[:,None],O['dims'][:,None].expand(-1,8,-1)),-1))

class PointExpert(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(29100+seed);self.context=PointContext();self.head=nn.Linear(64,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,O):return O['points']+.15*diagonal(O)[:,None,None]*self.head(self.context(O)).tanh()

def intersection_candidates(lines,mass):
    """27 nonparallel incident-role pairs plus a regularized 3-role LS fallback."""
    ll=lines[:,INCIDENT]  # B,8,3,3,3
    mm=mass[:,INCIDENT];positions=[];quality=[];allowed=[];descriptors=[]
    for a,b in ROLE_PAIRS:
        for i in range(3):
            for j in range(3):
                l1,l2=ll[:,:,a,i],ll[:,:,b,j];n1,n2=l1[...,:2],l2[...,:2]
                det=n1[...,0]*n2[...,1]-n1[...,1]*n2[...,0];ok=det.abs()>=.1;safe=torch.where(ok,det,torch.ones_like(det))
                x=(-l1[...,2]*n2[...,1]+n1[...,1]*l2[...,2])/safe
                y=(-n1[...,0]*l2[...,2]+l1[...,2]*n2[...,0])/safe
                p=torch.stack((x,y),-1);ok=ok&(p.abs()<=4).all(-1)&torch.isfinite(p).all(-1)
                p=p.clamp(-4,4);q=torch.sqrt(mm[:,:,a,i]*mm[:,:,b,j]);positions.append(p);quality.append(q);allowed.append(ok)
                descriptors.append(torch.cat((p,l1,l2,det.abs()[...,None],q[...,None]),-1))
    # If all pairs are degenerate this is a finite minimum-norm estimate, not
    # evidence of a fully observable point. Router receives support statistics.
    l=ll[:,:,:,0];n=l[...,:2];w=mm[:,:,:,0].clamp_min(1e-6)
    A=torch.einsum('bpki,bpk,bpkj->bpij',n,w,n)
    rhs=-torch.einsum('bpki,bpk,bpk->bpi',n,w,l[...,2])
    eye=torch.eye(2,device=lines.device,dtype=lines.dtype)
    fallback=torch.linalg.solve(A+1e-4*eye,rhs[...,None])[...,0].clamp(-4,4)
    good=torch.stack(allowed,-1);positions.append(fallback);allowed.append(~good.any(-1));quality.append(w.mean(-1))
    descriptors.append(torch.cat((fallback,l[:,:,0],l[:,:,1],torch.zeros_like(w[:,:,:1]),w.mean(-1,keepdim=True)),-1))
    return dict(candidates=torch.stack(positions,2),mask=torch.stack(allowed,2),desc=torch.stack(descriptors,2),support=good.float().mean(-1))

class LineDecoder(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(29200+seed);self.ident=nn.Embedding(8,8)
        self.global_encode=nn.Sequential(nn.Linear(12*3*4,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(32+8+10,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,lines,mass,box):
        c=intersection_candidates(lines,mass);B=len(lines)
        gl=self.global_encode(torch.cat((lines,mass[...,None]),-1).flatten(1))
        ident=self.ident(torch.arange(8,device=lines.device))[None].expand(B,-1,-1)
        token=torch.cat((gl[:,None,None].expand(-1,8,28,-1),ident[:,:,None].expand(-1,-1,28,-1),c['desc']),-1)
        logits=self.head(token)[...,0].masked_fill(~c['mask'],-1e4);weights=logits.softmax(-1)
        uv=(weights[...,None]*c['candidates']).sum(2);center=(box[:,:2]+box[:,2:])/2;half=(box[:,2:]-box[:,:2]).clamp_min(1)/2
        points=center[:,None]+half[:,None]*uv
        entropy=-(weights*weights.clamp_min(1e-9).log()).sum(-1)/3.3323
        return dict(points=points,logits=logits,weights=weights,candidates=c['candidates'],mask=c['mask'],support=c['support'],entropy=entropy)

def decoder_loss(out,O,gt,valid):
    center,half=geometry(O);target=(gt-center[:,None])/half[:,None]
    distances=(out['candidates']-target[:,:,None]).norm(dim=-1).masked_fill(~out['mask'],1e6)
    label=distances.argmin(-1)
    p=((out['points']-gt)/diagonal(O)[:,None,None]).norm(dim=-1)[valid].mean()
    ce=F.cross_entropy(out['logits'][valid],label[valid]);return p+.05*ce

class Router(nn.Module):
    def __init__(self,seed,arm):
        super().__init__();torch.manual_seed(29300+seed);self.arm=arm;self.context=PointContext()
        self.state=nn.Linear(64,3);self.head=nn.Sequential(nn.Linear(64+3+8+4,64),nn.SiLU(),nn.Linear(64,4))
    def forward(self,O,candidates,quality):
        q=self.context(O);state=self.state(q);prob=state.softmax(-1)
        rel=(candidates-O['points'][:,:,None])/diagonal(O)[:,None,None,None]
        if self.arm=='point_router':
            rel=rel.clone();rel[:,:,2:]=0;quality=torch.zeros_like(quality)
        raw=self.head(torch.cat((q,prob if self.arm!='no_state' else torch.zeros_like(prob),rel.flatten(-2),quality),-1))
        risk=F.softplus(raw)
        if self.arm=='point_router':risk=risk.clone();risk[:,:,2:]=1e6
        chosen=risk.argmin(-1);points=candidates.gather(2,chosen[...,None,None].expand(-1,-1,1,2))[:,:,0]
        return dict(points=points,risk=risk,state=state,chosen=chosen)

def router_loss(out,candidates,O,gt,valid,kind,arm):
    target=((candidates-gt[:,:,None])/diagonal(O)[:,None,None,None]).norm(dim=-1).detach()
    n=2 if arm=='point_router' else 4
    regression=F.smooth_l1_loss(out['risk'][...,:n][valid],target[...,:n][valid],beta=.03)
    choice=F.cross_entropy((-out['risk'][...,:n]/.02)[valid],target[...,:n].argmin(-1)[valid])
    state=F.cross_entropy(out['state'][valid],kind[valid]) if arm!='no_state' else regression*0
    return regression+.05*choice+.1*state
