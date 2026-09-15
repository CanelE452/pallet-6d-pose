"""PoseFix-derived pallet9 torch port. TF1 reference/weights required before training.

Network arrangement follows Moon et al., MIT PoseFix_RELEASE 5556364.
Bottleneck operations follow TensorFlow Authors' Apache-2.0 ResNet v1.
Not torchvision ResNet and not a human-benchmark reproduction.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvBN(nn.Module):
    def __init__(self,cin,cout,k=1,stride=1,act=True,transpose=False):
        super().__init__()
        self.conv=(nn.ConvTranspose2d(cin,cout,k,stride=stride,padding=1,bias=False) if transpose else nn.Conv2d(cin,cout,k,stride=stride,padding=k//2,bias=False))
        # TF1 FusedBatchNorm enforces minimum epsilon 1.001e-5 even when the
        # official graph requests 1e-9. Match the executed operator, not text.
        self.bn=nn.BatchNorm2d(cout,eps=1.001e-5,momentum=.01)
        self.act=act
    def forward(self,x):
        x=self.bn(self.conv(x)); return F.relu(x) if self.act else x

class Bottleneck(nn.Module):
    def __init__(self,cin,width,stride):
        super().__init__();cout=4*width
        self.conv1=ConvBN(cin,width)
        self.conv2=ConvBN(width,width,3,stride)
        self.conv3=ConvBN(width,cout,act=False)
        self.shortcut=ConvBN(cin,cout,1,stride,False) if cin!=cout else None
        self.stride=stride
    def forward(self,x):
        skip=self.shortcut(x) if self.shortcut is not None else x[:,:,::self.stride,::self.stride]
        return F.relu(skip+self.conv3(self.conv2(self.conv1(x))))

def gaussian(points,valid):
    safe=torch.where(valid[...,None],points,torch.zeros_like(points))
    yy=torch.arange(384,device=points.device,dtype=points.dtype)[None,None,:,None]
    xx=torch.arange(288,device=points.device,dtype=points.dtype)[None,None,None,:]
    return torch.exp(-((xx-safe[:,:,0,None,None]).square()+(yy-safe[:,:,1,None,None]).square())/162)*255*valid[:,:,None,None]

def expectation(logits):
    b,k,h,w=logits.shape
    q=logits.flatten(2).softmax(-1).reshape(b,k,h,w)
    xx=torch.arange(w,device=q.device,dtype=q.dtype)
    yy=torch.arange(h,device=q.device,dtype=q.dtype)
    return torch.stack(((q.sum(2)*xx).sum(-1),(q.sum(3)*yy).sum(-1)),-1)*4

def target_distribution(target,valid):
    """Safe equivalent of GPU scatter: retain in-range neighbors, normalize mass.

    Nonfinite/outside crop GT is masked, never clamped into a new edge label.
    At final grid cell partial in-bounds bilinear mass is renormalized.
    """
    valid=valid & torch.isfinite(target).all(-1) & (target>=0).all(-1) & (target[:,:,0]<288) & (target[:,:,1]<384)
    safe=torch.where(valid[...,None],target/4,torch.zeros_like(target))
    floor=safe.floor();frac=safe-floor
    b,k,_=target.shape; q=target.new_zeros(b,k,96*72)
    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
        x=floor[:,:,0].long()+dx;y=floor[:,:,1].long()+dy
        mask=valid & (x>=0)&(x<72)&(y>=0)&(y<96)
        mass=(frac[:,:,0] if dx else 1-frac[:,:,0])*(frac[:,:,1] if dy else 1-frac[:,:,1])*mask
        q.scatter_add_(2,(y.clamp(0,95)*72+x.clamp(0,71))[:,:,None],mass[:,:,None])
    q=q/q.sum(-1,keepdim=True).clamp_min(1e-12)
    return q,valid

class PoseFixPallet9(nn.Module):
    def __init__(self):
        super().__init__();self.stem=ConvBN(12,64,7,2);cin=64;self.blocks=nn.ModuleList()
        for block,(width,count) in enumerate(zip((64,128,256,512),(3,8,36,3))):
            units=nn.ModuleList()
            for unit in range(count):
                stride=2 if block>0 and unit==0 else 1
                units.append(Bottleneck(cin,width,stride));cin=4*width
            self.blocks.append(units)
        self.up=nn.ModuleList([ConvBN(2048,256,4,2,transpose=True),ConvBN(256,256,4,2,transpose=True),ConvBN(256,256,4,2,transpose=True)])
        self.out=nn.Conv2d(256,9,1)
    def forward(self,rgb,points,valid,return_layers=False):
        x=self.stem(torch.cat((rgb,gaussian(points,valid)),1));layers={'stem':x}
        # Official zero pad then VALID max pool; not implicit -inf padding.
        x=F.max_pool2d(F.pad(x,(1,1,1,1),value=0),3,2)
        for i,units in enumerate(self.blocks):
            for unit in units:x=unit(x)
            layers['block'+str(i+1)]=x
        for i,up in enumerate(self.up):x=up(x);layers['up'+str(i+1)]=x
        x=self.out(x);layers['logits']=x
        return (x,layers) if return_layers else x
    def losses(self,logits,target,valid):
        valid=valid.clone();valid[:,8]=False
        q,mask=target_distribution(target,valid)
        safe=torch.where(mask[...,None],target,torch.zeros_like(target))
        ce=(-(q*logits.flatten(2).log_softmax(-1)).sum(-1)*mask).mean()
        coord=((expectation(logits)/4-safe/4).abs()*mask[...,None]).mean()
        reg=sum(m.weight.square().sum() for m in self.modules() if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)))*.5e-5
        return ce+coord+reg,dict(heatmap=ce,coordinate=coord,L2=reg)
    def load_tf(self,path):
        z=np.load(path);mapping={};used=set()
        def take(tensor,key,perm=None):
            a=z[key];a=a.transpose(perm) if perm else a
            assert tuple(tensor.shape)==a.shape,(key,tensor.shape,a.shape)
            with torch.no_grad():tensor.copy_(torch.from_numpy(np.array(a,copy=True)))
            used.add(key);mapping[key]=list(a.shape)
        def cb(module,prefix):
            take(module.conv.weight,prefix+'/weights',(3,2,0,1))
            for field,key in [('weight','gamma'),('bias','beta'),('running_mean','moving_mean'),('running_var','moving_variance')]:take(getattr(module.bn,field),prefix+'/BatchNorm/'+key)
            module.bn.num_batches_tracked.zero_()
        cb(self.stem,'resnet_v1_152/conv1')
        for i,units in enumerate(self.blocks):
            for j,u in enumerate(units):
                p=f'resnet_v1_152/block{i+1}/unit_{j+1}/bottleneck_v1'
                for n in ('conv1','conv2','conv3'):cb(getattr(u,n),p+'/'+n)
                if u.shortcut is not None:cb(u.shortcut,p+'/shortcut')
        for i,u in enumerate(self.up):cb(u,'up'+str(i+1))
        take(self.out.weight,'out/weights',(3,2,0,1));take(self.out.bias,'out/biases')
        assert set(z.files)==used,sorted(set(z.files)-used)
        return dict(loaded=mapping,missing=[],unexpected=[],all_parameters_loaded=True)

class TFAdam(torch.optim.Optimizer):
    """TF1 Adam: epsilon outside uncorrected sqrt(v); L2 is in graph loss."""
    def __init__(self,params,lr=.0005):super().__init__(params,dict(lr=lr,betas=(.9,.999),eps=1e-8))
    @torch.no_grad()
    def step(self):
        change=[]
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                s=self.state[p]
                if not s:s.update(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                s['step']+=1;s['m'].mul_(b1).add_(p.grad,alpha=1-b1);s['v'].mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
                alpha=group['lr']*math.sqrt(1-b2**s['step'])/(1-b1**s['step'])
                before=p.clone()
                p.addcdiv_(s['m'],s['v'].sqrt().add_(group['eps']),value=-alpha)
                change.append((p-before).square().sum())
        return torch.stack(change).sum().sqrt()
