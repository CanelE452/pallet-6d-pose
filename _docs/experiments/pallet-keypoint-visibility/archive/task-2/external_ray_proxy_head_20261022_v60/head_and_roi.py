import torch,numpy as np
from torch import nn
from torch.nn import functional as F

class ProxyHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(65,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def native_grid(points,hw):
 q=np.asarray(points,dtype=float)[:8];H,W=hw;good=np.isfinite(q).all(-1)&~(q==-1).all(-1);finite=q[good];L=float(np.clip(.25*np.max(np.ptp(finite,axis=0)),16,max(hw)))if len(finite)>=2 else 16.
 offset=((np.arange(9)+.5)/9-.5)*L;xx,yy=np.meshgrid(offset,offset);grid=q[:,None,None,:]+np.stack([xx,yy],-1)[None];valid=(grid[...,0]>=0)&(grid[...,0]<W)&(grid[...,1]>=0)&(grid[...,1]<H)&good[:,None,None];return grid,valid,L

def patches(captured,points,hw,gain,offset):
 grid,valid,L=native_grid(points,hw);pad=captured['added_border'];network=(grid+pad)*gain+offset;index=network/8.;norm=2*(index+.5)/80-1
 tensor=captured['p3'].float();g=torch.as_tensor(norm,dtype=torch.float32,device=tensor.device);v=torch.as_tensor(valid[:,None],dtype=torch.float32,device=tensor.device)
 with torch.no_grad():roi=F.grid_sample(tensor.expand(8,-1,-1,-1),g,mode='bilinear',padding_mode='zeros',align_corners=False);roi=roi*v;out=torch.cat([roi,v],1)
 assert out.shape==(8,65,9,9)and torch.isfinite(out).all();return out.half().cpu(),valid.reshape(8,-1).any(1),L

def roi_preflight():
 points=np.array([[200.+k,160.+k]for k in range(8)]+[[0.,0.]]);hw=[480,640];gain=.7619047619047619;offset=np.array([0.,13.]);grid,valid,L=native_grid(points,hw);net=(grid+100)*gain+offset;index=net/8;norm=2*(index+.5)/80-1;yy,xx=torch.meshgrid(torch.arange(80,dtype=torch.float64),torch.arange(80,dtype=torch.float64),indexing='ij');ramp=torch.stack([xx,yy])[None].expand(8,-1,-1,-1);sample=F.grid_sample(ramp,torch.tensor(norm,dtype=torch.float64),align_corners=False);err=float((sample.permute(0,2,3,1)-torch.tensor(index)).abs().max());back=(net-offset)/gain-100;roundtrip=float(np.abs(back-grid).max());assert err<1e-10 and roundtrip<1e-8
 return dict(coordinate_ramp_max_error=err,native_network_roundtrip_px=roundtrip,PASS=True,exact_RF_center_claim=False)
