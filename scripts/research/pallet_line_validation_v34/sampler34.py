"""Fixed-grid bilinear zero-padding sampler via a deterministic linear operator."""
import torch
import torch.nn.functional as F
ORIGINAL_GRID_SAMPLE=F.grid_sample

def dense_grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
    if mode!='bilinear' or padding_mode!='zeros' or align_corners is not True or grid.requires_grad:
        raise ValueError('This verified sampler covers fixed bilinear zero-padding grids with align_corners=True only.')
    B,C,H,W=x.shape;oh,ow=grid.shape[1:3];g=grid.reshape(B,-1,2);gx=(g[...,0]+1)*(W-1)/2;gy=(g[...,1]+1)*(H-1)/2;x0=gx.floor();y0=gy.floor();wx=gx-x0;wy=gy-y0;matrix=x.new_zeros(B,oh*ow,H*W)
    # One write per output row in each term; ordered four-term addition.
    for dx,dy,weight in ((0,0,(1-wx)*(1-wy)),(1,0,wx*(1-wy)),(0,1,(1-wx)*wy),(1,1,wx*wy)):
        xx=x0.long()+dx;yy=y0.long()+dy;valid=(xx>=0)&(xx<W)&(yy>=0)&(yy<H);index=(yy.clamp(0,H-1)*W+xx.clamp(0,W-1));term=torch.zeros_like(matrix);term.scatter_(2,index[...,None],(weight*valid)[...,None]);matrix=matrix+term
    return torch.bmm(x.flatten(-2),matrix.transpose(1,2)).reshape(B,C,oh,ow)

def install():F.grid_sample=dense_grid_sample

def precheck():
    from context34 import deterministic,RUN,write
    deterministic();rows=[]
    for shape in ((2,3,9,11),(2,32,48,48)):
        x=torch.randn(*shape,dtype=torch.float64,device='cuda',requires_grad=True);grid=torch.rand(shape[0],8,9,2,dtype=torch.float64,device='cuda')*3-1.5;grid[:,0,:5]=torch.tensor([[-1.,-1.],[1.,1.],[0.,0.],[-1.1,.4],[1.1,-.2]],device='cuda',dtype=torch.float64);w=torch.randn(shape[0],shape[1],8,9,device='cuda',dtype=torch.float64);a=ORIGINAL_GRID_SAMPLE(x,grid,align_corners=True);ga=torch.autograd.grad((a*w).sum(),x)[0];b=dense_grid_sample(x,grid,align_corners=True);gb=torch.autograd.grad((b*w).sum(),x)[0];fd=float((a-b).abs().max());gd=float((ga-gb).abs().max());assert fd<1e-12 and gd<1e-12;rows.append(dict(shape=shape,forward_max=fd,gradient_max=gd))
    x=torch.randn(16,32,48,48,device='cuda',requires_grad=True);grid=torch.rand(16,8,9,2,device='cuda')*3-1.5;w=torch.randn(16,32,8,9,device='cuda');gr=[]
    for _ in range(3):gr.append(torch.autograd.grad((dense_grid_sample(x,grid,align_corners=True)*w).sum(),x)[0])
    assert all(torch.equal(gr[0],g) for g in gr[1:]);write(RUN/'SAMPLER_PREFLIGHT.json',dict(status='PASS',float64_equivalence=rows,repeat_float32_gradients_exact=True,fixed_grid_only=True,optimizer_updates=0));print('SAMPLER34 PASS',flush=True)

if __name__=='__main__':precheck()
