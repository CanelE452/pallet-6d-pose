"""Feature-space DHT and a non-DHT global-query baseline, written independently.

Geometry is fixed. Voting uses linear rho bin interpolation and content-mass
normalization. This is NOT a reproduction of an ECCV paper or optimized CUDA.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from .geometry import lines_to_raw


def hough_pad(x:Tensor,p:int=1):
    if p<1:return x
    # theta -> theta+pi implies rho -> -rho. Not ordinary circular padding.
    x=torch.cat((x[...,-p:,:].flip(-1),x,x[...,:p,:].flip(-1)),-2)
    return torch.nn.functional.pad(x,(p,p))

class HoughConv(nn.Module):
    def __init__(self,c):
        super().__init__();self.conv=nn.Conv2d(c,c,3);self.norm=nn.GroupNorm(4,c)
    def forward(self,x):return torch.nn.functional.silu(self.norm(self.conv(hough_pad(x))))

class Lattice(nn.Module):
    def __init__(self,height=24,width=24,theta_bins=36,rho_bins=65):
        super().__init__()
        if min(height,width)<2 or theta_bins<4 or rho_bins<5 or rho_bins%2!=1:
            raise ValueError('Invalid lattice; odd symmetric rho axis required')
        self.height,self.width,self.T,self.R=height,width,theta_bins,rho_bins
        theta=torch.arange(theta_bins,dtype=torch.float64)*math.pi/theta_bins
        rho=torch.linspace(-math.sqrt(2),math.sqrt(2),rho_bins,dtype=torch.float64)
        yy,xx=torch.meshgrid((torch.arange(height,dtype=torch.float64)+.5)*2/height-1,
                             (torch.arange(width,dtype=torch.float64)+.5)*2/width-1,indexing='ij')
        points=torch.stack((xx.flatten(),yy.flatten()),-1)
        normals=torch.stack((theta.cos(),theta.sin()),-1)
        normals[normals.abs()<1e-12]=0
        projected=normals@points.T
        pos=(projected-rho[0])/(rho[1]-rho[0])
        nearest=pos.round()
        pos=torch.where((pos-nearest).abs()<1e-10,nearest,pos)
        lo=pos.floor().long();frac=pos-lo
        rows=[];cols=[];values=[]
        angle_idx=torch.arange(theta_bins)[:,None].expand_as(lo)
        pixel_idx=torch.arange(height*width)[None].expand_as(lo)
        for idx,weight in ((lo,1-frac),(lo+1,frac)):
            keep=(idx>=0)&(idx<rho_bins)&(weight>0)
            rows.append((angle_idx*rho_bins+idx)[keep]);cols.append(pixel_idx[keep]);values.append(weight[keep])
        if theta_bins*rho_bins*height*width>20_000_000:
            raise ValueError('Dense reference DHT matrix too large; add independently validated sparse adapter')
        A=torch.zeros(theta_bins*rho_bins,height*width,dtype=torch.float64)
        A.index_put_((torch.cat(rows),torch.cat(cols)),torch.cat(values),accumulate=True)
        nt=normals[:,None].expand(-1,rho_bins,-1)
        rr=rho[None].expand(theta_bins,-1)
        self.register_buffer('A',A.float(),persistent=False)
        self.register_buffer('lines',torch.cat((nt,-rr[...,None]),-1).reshape(-1,3).float())
        self.register_buffer('xy',points.float())

    def forward(self,x:Tensor,content:Tensor):
        b,c,h,w=x.shape
        if (h,w)!=(self.height,self.width) or content.shape!=(b,1,h,w):raise ValueError('DHT input shape mismatch')
        mask=content.to(x.dtype).flatten(2)
        mass=mask@self.A.to(x.dtype).T
        value=(x*content).flatten(2)@self.A.to(x.dtype).T
        result=value/mass.clamp_min(1e-6)
        return result.reshape(b,c,self.T,self.R),mass.reshape(b,1,self.T,self.R)

class DHTLineHead(nn.Module):
    def __init__(self,channels:int,lattice:Lattice,latent=16):
        super().__init__();self.lattice=lattice
        self.reduce=nn.Sequential(nn.Conv2d(channels,latent,1),nn.SiLU(),nn.Conv2d(latent,latent,3,padding=1),nn.SiLU())
        self.readout=nn.Sequential(HoughConv(latent),HoughConv(latent),nn.Conv2d(latent,12,1))
    def forward(self,x,content):
        z=self.reduce(x);v,mass=self.lattice(z,content)
        logits=self.readout(v).flatten(2)
        valid=mass.flatten(2)>1e-6
        return logits.masked_fill(~valid,-1e4),valid.expand(-1,12,-1)

class DirectLineHead(nn.Module):
    """Global role queries scoring analytic line embeddings; NO line-aligned voting."""
    def __init__(self,channels:int,lattice:Lattice,latent=16):
        super().__init__();self.lattice=lattice
        self.key=nn.Conv2d(channels,latent,1);self.value=nn.Conv2d(channels,latent,1)
        self.queries=nn.Parameter(torch.randn(12,latent)*.02)
        self.position=nn.Linear(2,latent)
        self.embed=nn.Sequential(nn.Linear(6,latent),nn.SiLU(),nn.Linear(latent,latent))
        self.output=nn.Linear(latent,latent)
    def forward(self,x,content):
        b,c,h,w=x.shape
        k=self.key(x).flatten(2).transpose(1,2)+self.position(self.lattice.xy)[None]
        v=self.value(x).flatten(2).transpose(1,2)
        a=torch.einsum('ec,bpc->bep',self.queries,k)/math.sqrt(k.shape[-1])
        a=a.masked_fill(~content.flatten(2).bool(),-1e4)
        desc=torch.softmax(a,-1)@v
        nx,ny,rho=self.lattice.lines.unbind(-1)
        p=torch.stack((nx*nx-ny*ny,2*nx*ny,rho*nx,rho*ny,rho.square(),torch.ones_like(rho)),-1)
        logits=self.output(desc)@self.embed(p).T/math.sqrt(k.shape[-1])
        # Geometry coverage is shared with DHT; computing valid mass is not feature voting.
        mass=content.to(x.dtype).flatten(2)@self.lattice.A.to(x.dtype).T
        valid=(mass>1e-6).expand(-1,12,-1)
        return logits.masked_fill(~valid,-1e4),valid


def decode_modes(logits:Tensor,valid:Tensor,lattice:Lattice,raw_to_grid:Tensor,modes=3):
    """Hard mode location + differentiable LOCAL averaging (not global soft-argmax).
    Mode/window selection is piecewise, not differentiable through argmax/NMS.
    Return conditional retained-mode masses, not calibrated correctness scores.
    """
    if modes<1:raise ValueError('At least one mode')
    base=lattice.lines.to(logits.dtype)
    prob=torch.softmax(logits.masked_fill(~valid,-1e4),-1)*valid
    work=logits.masked_fill(~valid,-1e4)
    out=[];masses=[];available=[]
    angle_bw=math.pi/lattice.T*1.5;rho_bw=2*math.sqrt(2)/(lattice.R-1)*1.5
    for k in range(modes):
        anchor_idx=work.argmax(-1)
        anchor=base[anchor_idx]
        dot=torch.einsum('beq,mq->bem',anchor[...,:2],base[:,:2])
        sign=torch.where(dot>=0,1.,-1.)
        # absolute-normal agreement handles theta/rho antipodal seam.
        angle_error=1-dot.abs().clamp_max(1)
        offset_error=base[None,None,:,2]*sign-anchor[...,2,None]
        distance=angle_error/(1-math.cos(angle_bw))+offset_error.square()/rho_bw**2
        window=(distance<=1)&valid
        this_valid=(work.max(-1).values>-9000)&window.any(-1)
        weights=torch.softmax((logits-.5*distance).masked_fill(~window,-1e4),-1)*window
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        h=torch.einsum('bem,bem,mq->beq',weights,sign,base)
        h=h/torch.linalg.vector_norm(h[...,:2],dim=-1,keepdim=True).clamp_min(1e-8)
        out.append(h);masses.append((prob*window).sum(-1));available.append(this_valid)
        work=work.masked_fill(distance<=4,-1e4)
    lines=torch.stack(out,-2)
    mass=torch.stack(masses,-1);okay=torch.stack(available,-1)
    mass=torch.where(okay,mass,torch.zeros_like(mass))
    lp=(mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
    raw,norm_ok=lines_to_raw(lines,raw_to_grid)
    return raw,lp,okay&norm_ok
