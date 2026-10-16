"""Equivalent pooling for exactly divisible grids, with deterministic backward."""
import torch
import torch.nn.functional as F
ORIGINAL_ADAPTIVE_POOL=F.adaptive_avg_pool2d

def divisible_pool(x,size):
    if isinstance(size,int):size=(size,size)
    h,w=x.shape[-2:];oh,ow=size
    if h%oh or w%ow:raise ValueError(f'Non-divisible adaptive pool is not covered: {(h,w)} -> {size}')
    return F.avg_pool2d(x,kernel_size=(h//oh,w//ow),stride=(h//oh,w//ow))

def install():
    F.adaptive_avg_pool2d=divisible_pool

def precheck():
    from context34 import RUN,write,deterministic
    from model33 import Correspondence
    deterministic();rows=[]
    for shape,size in [((2,32,48,48),(4,4)),((2,12,24,48),(6,12)),((2,12,48,48),(8,8))]:
        x=torch.randn(*shape,dtype=torch.float64,device='cuda',requires_grad=True);weight=torch.randn(*shape[:-2],*size,dtype=torch.float64,device='cuda')
        torch.use_deterministic_algorithms(False);a=ORIGINAL_ADAPTIVE_POOL(x,size);ga=torch.autograd.grad((a*weight).sum(),x)[0];torch.use_deterministic_algorithms(True);b=divisible_pool(x,size);gb=torch.autograd.grad((b*weight).sum(),x)[0];d=float((a-b).abs().max());g=float((ga-gb).abs().max());assert d<1e-12 and g<1e-12;rows.append(dict(shape=shape,size=size,max_forward_difference=d,max_gradient_difference=g))
    install();B=2;O=dict(features=torch.randn(B,192,24,24,device='cuda'),content=torch.ones(B,1,24,24,dtype=torch.bool,device='cuda'),edge=torch.randn(B,3,96,96,device='cuda'),box=torch.tensor([[0.,0.,100.,70.]]*B,device='cuda'),image_hw=torch.tensor([[100.,100.]]*B,device='cuda'));base=torch.rand(B,8,2,device='cuda')*80;S=dict(base=base,strong=base+4)
    for arm in ('line_only','pooled'):
        m=Correspondence(1).cuda()
        with torch.no_grad():m.head[-1].weight.normal_(0,.01)
        out=m(O,S,arm);loss=out['points'].square().mean()+out['role_logits'].square().mean();loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    write(RUN/'DETERMINISTIC_PREFLIGHT.json',dict(status='PASS',pooling_checks=rows,full_model_backward_strict=True,optimizer_updates=0));print('DETERMINISTIC34 PREFLIGHT PASS',flush=True)

if __name__=='__main__':precheck()
