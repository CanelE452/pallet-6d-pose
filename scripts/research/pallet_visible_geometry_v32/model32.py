"""Visible-role supervision -> differentiable Hough + endpoints -> joint point repair."""
import torch
from torch import nn
import torch.nn.functional as F
import math

class Hough(nn.Module):
    def __init__(self,size=24,theta=24,rho=48):
        super().__init__();self.theta=theta;self.rho=rho
        y,x=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij');a=torch.arange(theta)*math.pi/theta
        r=(x.flatten()[:,None]*a.cos()+y.flatten()[:,None]*a.sin()+math.sqrt(2))*(rho-1)/(2*math.sqrt(2));lo=r.floor().long().clamp(0,rho-1);hi=(lo+1).clamp_max(rho-1);w=(r-lo).clamp(0,1)
        basis=torch.zeros(size*size,theta*rho);rows=torch.arange(size*size)[:,None].expand(-1,theta);offset=torch.arange(theta)[None]*rho
        basis.index_put_((rows,lo+offset),1-w,accumulate=True);basis.index_put_((rows,hi+offset),w,accumulate=True)
        self.register_buffer('basis',basis);self.register_buffer('count',basis.sum(0).clamp_min(1))
    def forward(self,x):
        z=F.interpolate(x,size=(24,24),mode='bilinear',align_corners=True)
        return ((z.flatten(-2)@self.basis)/self.count).reshape(len(x),12,self.theta,self.rho)

class VisibleGeometry(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(32000+seed)
        self.project=nn.Conv2d(192,32,1);self.pixel=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU())
        self.trunk=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.roles=nn.Conv2d(32,24,1);self.hough=Hough();self.line_encode=nn.Sequential(nn.Linear(12*6*12,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU());self.end_encode=nn.Sequential(nn.Linear(12*8*8,64),nn.SiLU());self.ident=nn.Embedding(8,8)
        self.head=nn.Sequential(nn.Linear(9*32*2+32*4*4+128+8+6,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,O,S,arm='visible_hough',intervention=None):
        B=len(O['box']);bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
        z=F.interpolate(self.project(O['features'])*O['content'],size=(48,48),mode='bilinear',align_corners=True)
        pix=self.pixel(F.interpolate(O['edge'],size=(48,48),mode='bilinear',align_corners=True));z=self.trunk(torch.cat((z,pix),1))
        logits=self.roles(z);maps=logits.sigmoid()
        if intervention=='donor':maps=maps.roll(1,0)
        if intervention=='off':maps=torch.zeros_like(maps)
        hough=self.hough(maps[:,:12]);line=self.line_encode(F.adaptive_avg_pool2d(hough,(6,12)).flatten(1));end=self.end_encode(F.adaptive_avg_pool2d(maps[:,12:],(8,8)).flatten(1))
        if arm=='point' or intervention=='tokens_off':line=torch.zeros_like(line);end=torch.zeros_like(end)
        if intervention=='hough_off':line=torch.zeros_like(line)
        if intervention=='endpoints_off':end=torch.zeros_like(end)
        def patch(points):
            axis=points.new_tensor([-.03,0,.03]);y,x=torch.meshgrid(axis,axis,indexing='ij');off=torch.stack((x.flatten(),y.flatten()),-1)
            positions=points[:,:,None]+diag[:,None,None,None]*off;grid=2*(positions-bb[:,None,None,:2])/wh[:,None,None]-1;valid=(grid.abs()<=1).all(-1)&((positions>=0)&(positions<O['image_hw'].flip(-1)[:,None,None])).all(-1)
            return (F.grid_sample(z,grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1)*valid[...,None]).flatten(-2)
        base,strong=S['base'],S['strong'];gl=F.adaptive_avg_pool2d(z,(4,4)).flatten(1);ident=self.ident(torch.arange(8,device=z.device))[None].expand(B,-1,-1)
        coords=torch.cat((2*(base-bb[:,None,:2])/wh[:,None]-1,2*(strong-bb[:,None,:2])/wh[:,None]-1,(strong-base)/diag[:,None,None]),-1)
        context=torch.cat((patch(base),patch(strong),gl[:,None].expand(-1,8,-1),torch.cat((line,end),-1)[:,None].expand(-1,8,-1),ident,coords),-1)
        raw=self.head(context);mix=raw[...,0].tanh();delta=raw[...,1:].tanh();delta=delta/(delta.norm(dim=-1,keepdim=True).clamp_min(1));residual=.15*diag[:,None,None]*delta
        points=base+mix[...,None]*(strong-base)+residual
        return dict(points=points,mix=mix,residual=residual,role_logits=logits,role_maps=maps,hough=hough,base=base,strong=strong,diag=diag)

def objective(out,gt,valid,target,supervised,arm):
    point=((out['points']-gt).norm(dim=-1)/out['diag'][:,None])[valid].mean();aux=point*0
    if arm in ('visible_hough','virtual_hough') and supervised.any():
        z=out['role_logits'][supervised];y=target[supervised];loss=F.binary_cross_entropy_with_logits(z,y,reduction='none');pos=y>0;neg=~pos
        aux=(loss[pos].mean() if pos.any() else loss.sum()*0)+(loss[neg].mean() if neg.any() else loss.sum()*0)
    return point+.05*aux,dict(point=point,aux=aux,point_error=(out['points']-gt).norm(dim=-1)[valid].mean())
