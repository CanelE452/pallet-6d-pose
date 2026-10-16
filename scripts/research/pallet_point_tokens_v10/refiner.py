"""Per-point local tokens + learned line association + directional residuals."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def direction_matrix(weights,normals):
    """PSD normal-direction precision proxy. Null attention contributes zero."""
    return torch.einsum('bpm,bmi,bmj->bpij',weights,normals,normals)


class Refiner(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(10000+seed)
        self.project=nn.Conv2d(192,32,1)
        self.identity=nn.Embedding(8,8)
        self.token=nn.Sequential(nn.Linear(586,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.query=nn.Linear(64,32)
        self.key=nn.Sequential(nn.Linear(39,32),nn.SiLU(),nn.Linear(32,32))
        self.null=nn.Linear(64,1);nn.init.constant_(self.null.bias,2.)
        self.offset=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.offset[-1].weight);nn.init.zeros_(self.offset[-1].bias)

    def forward(self,O,lines,mass,arm='directional',projection=None):
        z=self.project(O['features'])*O['content'].to(O['features'].dtype)
        points=O['points'];bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1.)
        center=(bb[:,:2]+bb[:,2:])*.5;diag=torch.linalg.vector_norm(wh,dim=-1)
        pgrid=2*(points-bb[:,None,:2])/wh[:,None]-1
        axis=points.new_tensor([-1.,0.,1.]);y,x=torch.meshgrid(axis,axis,indexing='ij')
        offsets=torch.stack((x.flatten(),y.flatten()),-1)
        # Keep each point separate; floating-point sampling at two local scales.
        grids=[]
        for radius in (.01,.05):
            grids.append(pgrid[:,:,None]+offsets[None,None]*(2*radius*diag[:,None,None,None]/wh[:,None,None]))
        grid=torch.cat(grids,2)
        local=F.grid_sample(z,grid,align_corners=True,padding_mode='zeros').permute(0,2,1,3).flatten(2)
        ident=self.identity(torch.arange(8,device=z.device))[None].expand(len(z),-1,-1)
        token=self.token(torch.cat((local,pgrid,ident),-1))
        # Gather image evidence along each independent DHT line, never between points.
        n=lines[...,:2]/torch.linalg.vector_norm(lines[...,:2],dim=-1,keepdim=True).clamp_min(1e-8)
        c=lines[...,2]/torch.linalg.vector_norm(lines[...,:2],dim=-1).clamp_min(1e-8)
        foot=center[:,None]-((center[:,None]*n).sum(-1)+c)[...,None]*n
        tangent=torch.stack((-n[...,1],n[...,0]),-1)
        along=torch.linspace(-.75,.75,9,device=z.device,dtype=z.dtype)
        raw=foot[:,:,None]+tangent[:,:,None]*along[None,None,:,None]*diag[:,None,None,None]
        lg=2*(raw-bb[:,None,None,:2])/wh[:,None,None]-1
        inside=(lg.abs()<=1).all(-1)
        lf=F.grid_sample(z,lg,align_corners=True,padding_mode='zeros').permute(0,2,3,1)
        lf=(lf*inside[...,None]).sum(2)/inside.sum(-1).clamp_min(1)[...,None]
        residual=(points[:,:,None]*n[:,None]).sum(-1)+c[:,None]
        rn=residual/diag[:,None,None]
        orientation=torch.stack((n[...,0].square()-n[...,1].square(),2*n[...,0]*n[...,1]),-1)
        descriptors=torch.cat((lf[:,None].expand(-1,8,-1,-1),orientation[:,None].expand(-1,8,-1,-1),
            rn[...,None]*n[:,None],rn.abs()[...,None],mass.clamp_min(1e-8).log()[:,None,:,None].expand(-1,8,-1,-1),
            inside.float().mean(-1)[:,None,:,None].expand(-1,8,-1,-1)),-1)
        keys=self.key(descriptors)
        scores=(self.query(token)[:,:,None]*keys).sum(-1)/math.sqrt(32)-rn.abs()/.05
        weights=torch.softmax(torch.cat((scores,self.null(token)),-1),-1)
        association=weights[...,:-1]
        context=(association[...,None]*keys).sum(2)
        A=direction_matrix(association,n)
        mode=arm if projection is None else projection
        if mode=='local':
            context=torch.zeros_like(context)
            A=torch.eye(2,device=z.device,dtype=z.dtype)[None,None].expand(len(z),8,2,2)
        elif mode=='isotropic':
            # Same total directional strength, spread uniformly over x/y.
            trace=A.diagonal(dim1=-2,dim2=-1).sum(-1)*.5
            A=trace[...,None,None]*torch.eye(2,device=z.device,dtype=z.dtype)
        elif mode=='off':A=torch.zeros_like(A)
        proposal=.02*diag[:,None,None]*torch.tanh(self.offset(torch.cat((token,context),-1)))
        delta=(A@proposal[...,None])[...,0]
        return dict(points=points+delta,delta=delta,proposal=proposal,A=A,weights=weights,
                    normals=n,raw_residual=residual,token=token)


def loss(out,original,gt,valid):
    element=F.smooth_l1_loss(out['points'],gt,beta=1.,reduction='none').sum(-1)
    before=torch.linalg.vector_norm(original-gt,dim=-1)
    after=torch.linalg.vector_norm(out['points']-gt,dim=-1)
    # Training loss, not a test-time filter or GT-dependent decision.
    safe=(after-before).clamp_min(0)
    return ((element+.25*safe)*valid).sum()/valid.sum().clamp_min(1)
