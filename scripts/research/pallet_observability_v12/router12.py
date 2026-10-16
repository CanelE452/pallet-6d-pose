"""Learn original-correction suppression and separate amodal residual experts."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(12000+seed)
        self.project=nn.Conv2d(192,32,1);self.identity=nn.Embedding(8,8)
        self.token=nn.Sequential(nn.Linear(306,64),nn.SiLU(),nn.Linear(64,64))
        self.grid_feature=nn.Linear(32,64);self.grid_position=nn.Linear(2,64)
        self.cross=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.self_attention=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.norm1=nn.LayerNorm(64);self.norm2=nn.LayerNorm(64)
        self.feed=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64))
        self.route=nn.Linear(64,3);self.recovery=nn.Linear(64,4)
        nn.init.zeros_(self.route.weight)
        with torch.no_grad():self.route.bias.copy_(torch.tensor([math.log(.9),math.log(.05),math.log(.05)]))
        nn.init.zeros_(self.recovery.weight);nn.init.zeros_(self.recovery.bias)
        self.scalar=nn.Parameter(torch.tensor(math.log(9.)))

    def forward(self,O,base,arm):
        old=O['points'];box=O['box'];wh=(box[:,2:]-box[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
        z=self.project(O['features'])*O['content'];pgrid=2*(old-box[:,None,:2])/wh[:,None]-1
        y,x=torch.meshgrid(old.new_tensor([-1,0,1]),old.new_tensor([-1,0,1]),indexing='ij');offset=torch.stack((x.flatten(),y.flatten()),-1)
        grid=pgrid[:,:,None]+offset[None,None]*(.06*diag[:,None,None,None]/wh[:,None,None])
        patches=F.grid_sample(z,grid,padding_mode='zeros',align_corners=True).permute(0,2,1,3).flatten(2)
        hw=O['image_hw'];xy=old/hw.flip(-1)[:,None];boundary=torch.cat((xy,1-xy),-1)
        identity=self.identity(torch.arange(8,device=old.device))[None].expand(len(old),-1,-1)
        local_delta=base-old
        token=self.token(torch.cat((patches,pgrid,xy,boundary,local_delta/diag[:,None,None],identity),-1))
        global_image=F.adaptive_avg_pool2d(z,(8,8)).flatten(2).transpose(1,2)
        gy,gx=torch.meshgrid(torch.linspace(-1,1,8,device=z.device),torch.linspace(-1,1,8,device=z.device),indexing='ij')
        position=torch.stack((gx.flatten(),gy.flatten()),-1)
        memory=self.grid_feature(global_image)+self.grid_position(position)[None]
        context,attention=self.cross(token,memory,memory,need_weights=True)
        token=self.norm1(token+context);context,_=self.self_attention(token,token,token,need_weights=False)
        token=self.norm2(token+context);token=token+self.feed(token)
        logits=self.route(token);q=logits.softmax(-1)
        experts=.25*diag[:,None,None,None]*torch.tanh(self.recovery(token).reshape(len(old),8,2,2))
        if arm=='scalar_gate':
            keep=torch.sigmoid(self.scalar).expand(len(old),8,1);recovery=torch.zeros_like(old)
        elif arm=='learned_gate':keep=q[...,:1];recovery=torch.zeros_like(old)
        elif arm=='gated_amodal':
            keep=q[...,:1];recovery=(q[...,1:,None]*experts).sum(-2)
        elif arm=='always_amodal':
            keep=torch.ones_like(q[...,:1]);recovery=(logits[...,1:].softmax(-1)[...,None]*experts).sum(-2)
        else:raise ValueError(arm)
        points=old+keep*local_delta+recovery
        return dict(points=points,keep=keep,recovery=recovery,experts=experts,probabilities=q,logits=logits,base_points=base,attention=attention)

def objective(out,O,gt,valid,kind,arm,class_weights):
    e=F.smooth_l1_loss(out['points'],gt,beta=1,reduction='none').sum(-1)
    before=(O['points']-gt).norm(dim=-1);after=(out['points']-gt).norm(dim=-1)
    point=((e+.25*(after-before).clamp_min(0))*valid).sum()/valid.sum().clamp_min(1)
    visibility=F.cross_entropy(out['logits'][valid],kind[valid],weight=class_weights) if arm!='scalar_gate' else point.new_zeros(())
    return point+.5*visibility,point,visibility
