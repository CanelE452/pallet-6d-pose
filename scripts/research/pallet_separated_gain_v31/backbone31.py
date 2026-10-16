"""Predict signed directional amount and counterfactual gain, without state classes.

Every point may keep either existing parent or use a partial correction. A normal
candidate preserves its anchor's tangent component exactly. All data-dependent
choices at inference use learned gains; synthetic GT appears only in objective.
"""
import torch
from torch import nn
import torch.nn.functional as F
from model29 import INCIDENT,diagonal

ANCHORS=[0,1,0,0,0,1,1,1]+[0]*18+[1]*18
GROUPS=((1,),(2,3,4,5,6,7),tuple(range(8,44)))

def bounded(v,rad):return v*(rad/v.norm(dim=-1).clamp_min(1e-8)).clamp_max(1)[...,None]

def paths(O,S,arm):
    b,s=S['base'],S['strong'];diag=diagonal(O);B,P,_=b.shape;anchors=torch.stack((b,s),2)[:,:,ANCHORS];rad=.15*diag[:,None]
    delta=bounded(s-b,rad);x=delta*delta.new_tensor([1,0]);y=delta*delta.new_tensor([0,1])
    raw=torch.stack((torch.zeros_like(b),torch.zeros_like(b),delta,x,y,-delta,-x,-y),2)
    # 2 DHT sources x3 incident roles x3 modes for each point.
    ll=S['lines'][:, :, INCIDENT].permute(0,2,1,3,4,5).flatten(2,4)
    mass=S['mass'][:,:,INCIDENT].permute(0,2,1,3,4).flatten(2,4)
    norm=ll[...,:2].norm(dim=-1,keepdim=True);ll=ll/norm.clamp_min(1e-8);n=ll[...,:2]
    line_delta=[]
    for anchor in (b,s):line_delta.append(-((anchor[:,:,None]*n).sum(-1)+ll[...,2])[...,None]*n)
    ld=torch.cat(line_delta,2);ld=bounded(ld,rad[:,:,None]);mask=torch.ones(B,P,44,device=b.device,dtype=torch.bool)
    if arm=='generic':
        angle=torch.arange(18,device=b.device)*torch.pi/18
        direct=torch.stack((angle.cos(),angle.sin()),-1)
        ld=(.15*diag[:,None,None,None]*direct[None,None]).expand(-1,8,-1,-1).repeat(1,1,2,1)
    elif arm=='point_gain':mask[:,:,8:]=False;ld=torch.zeros_like(ld)
    elif arm=='whole_line':
        # Four whole-displacement alternatives, with the same learned signed
        # amount/risk head and the same preserved point parents.
        ld=torch.zeros_like(ld);mask[:,:,8:]=False
        for a,anchor in enumerate((b,s)):
            for expert in range(2):
                k=a*18+expert*9;ld[:,:,k]=bounded(S['expert_points'][:,:,expert]-anchor,rad);mask[:,:,8+k]=True
    return dict(anchors=anchors,directions=torch.cat((raw,ld),2),mask=mask,lines=ll,mass=mass,normals=n,diag=diag)

class FeatureBackbone(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(30000+seed)
        self.project=nn.Conv2d(192,8,1);self.ident=nn.Embedding(8,8);self.anchor_id=nn.Embedding(2,4);self.family_id=nn.Embedding(3,4)
        self.point_encode=nn.Sequential(nn.Linear(9*11+8+8+4+2+2+2,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.line_encode=nn.Sequential(nn.Linear(11*12+4,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(64+32+10+4,96),nn.SiLU(),nn.Linear(96,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[0]=.5493061443  # initial signed amount +0.5

    def forward(self,O,S,arm='normal_gain',evidence_mode='normal'):
        c=paths(O,S,arm);B,P,_=S['base'].shape;bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=c['diag'];hw=O['image_hw'].flip(-1)
        z=self.project(O['features'])*O['content'];pix=F.interpolate(O['edge'],size=z.shape[-2:],mode='bilinear',align_corners=True);feat=torch.cat((z,pix),1)
        def sample(pos,evidence=False):
            grid=2*(pos-bb[:,None,None,:2])/wh[:,None,None]-1;valid=((pos>=0)&(pos<hw[:,None,None])).all(-1)&(grid.abs()<=1).all(-1)
            source=feat.roll(1,0) if evidence and evidence_mode=='donor' else feat
            zz=F.grid_sample(source,grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1)*valid[...,None]
            return zz,valid
        anchors=torch.stack((S['base'],S['strong']),2);flat=anchors.flatten(1,2)
        axis=flat.new_tensor([-1,0,1]);yy,xx=torch.meshgrid(axis,axis,indexing='ij');off=torch.stack((xx.flatten(),yy.flatten()),-1)
        local,_=sample(flat[:,:,None]+.03*diag[:,None,None,None]*off);globalf=z.mean((-2,-1))[:,None].expand(-1,16,-1)
        ident=self.ident(torch.arange(8,device=z.device))[:,None].expand(-1,2,-1).reshape(16,8)[None].expand(B,-1,-1);aid=self.anchor_id(torch.arange(2,device=z.device))[None].expand(8,-1,-1).reshape(16,4)[None].expand(B,-1,-1)
        point=self.point_encode(torch.cat((local.flatten(-2),globalf,ident,aid,2*(flat-bb[:,None,:2])/wh[:,None]-1,flat/hw[:,None],(S['strong']-S['base'])[:,:,None].expand(-1,-1,2,-1).reshape(B,16,2)/diag[:,None,None]),-1)).reshape(B,8,2,64)
        point=point[:,:,ANCHORS];token=z.new_zeros(B,8,18,32);coverage=z.new_zeros(B,8,18)
        if arm not in ('point_gain','normal_no_evidence','generic') and evidence_mode!='off':
            n=c['normals'];center=(bb[:,:2]+bb[:,2:])/2;foot=center[:,None,None]-((center[:,None,None]*n).sum(-1)+c['lines'][...,2])[...,None]*n
            tangent=torch.stack((-n[...,1],n[...,0]),-1);u=torch.linspace(-.75,.75,12,device=z.device);pos=foot[...,None,:]+tangent[...,None,:]*u[None,None,None,:,None]*diag[:,None,None,None,None]
            zz,v=sample(pos.flatten(2,3),True);zz=zz.reshape(B,8,18,12,11);v=v.reshape(B,8,18,12);coverage=v.float().mean(-1)
            grad=zz[...,9:11];normal=(grad*n[...,None,:]).sum(-1);tang=(grad*tangent[...,None,:]).sum(-1)
            stats=torch.stack((coverage,normal.abs().mean(-1),tang.abs().mean(-1),normal.std(-1,unbiased=False)),-1)
            token=self.line_encode(torch.cat((zz.flatten(-2),stats),-1))
        if arm=='whole_line':
            token=token.reshape(B,8,2,9,32).mean(3,keepdim=True).expand(-1,-1,-1,9,-1).reshape(B,8,18,32)
            coverage=coverage.reshape(B,8,2,9).mean(3,keepdim=True).expand(-1,-1,-1,9).reshape(B,8,18)
        tokens=torch.cat((z.new_zeros(B,8,8,32),token,token),2)
        dest=c['anchors']+c['directions'];inside=((dest>=0)&(dest<hw[:,None,None])).all(-1).float()
        mass=torch.cat((z.new_zeros(B,8,8),c['mass'],c['mass']),2);cov=torch.cat((z.new_zeros(B,8,8),coverage,coverage),2)
        if arm in ('point_gain','generic'):mass=torch.zeros_like(mass)
        dd=c['directions']/diag[:,None,None,None];rel=(c['anchors']-S['base'][:,:,None])/diag[:,None,None,None]
        geom=torch.cat((dd,dd.norm(dim=-1,keepdim=True),rel,inside[...,None],mass[...,None],cov[...,None],2*(dest-bb[:,None,None,:2])/wh[:,None,None]-1),-1)
        fid=self.family_id(torch.tensor([0,0]+[1]*6+[2]*36,device=z.device))[None,None].expand(B,8,-1,-1)
        raw=self.head(torch.cat((point,tokens,geom,fid),-1));alpha=raw[...,0].tanh();gain=raw[...,1];gain=gain.clone();gain[:,:,0]=0
        prediction=c['anchors']+alpha[...,None]*c['directions'];chosen=gain.masked_fill(~c['mask'],-1e9).argmax(-1)
        points=prediction.gather(2,chosen[...,None,None].expand(-1,-1,1,2))[:,:,0]
        return dict(point_embedding=point,embedding=torch.cat((point,tokens,geom,fid),-1),points=points,alpha=alpha,gain=gain,chosen=chosen,candidate_points=prediction,anchors=c['anchors'],directions=c['directions'],candidate_mask=c['mask'],diag=diag,base=S['base'],strong=S['strong'])
