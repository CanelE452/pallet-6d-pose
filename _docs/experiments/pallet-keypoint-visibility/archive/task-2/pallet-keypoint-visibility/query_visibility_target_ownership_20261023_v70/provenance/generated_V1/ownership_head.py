"""Conditional V-vs-E query ownership, with an explicit other-ROI prototype.

No six-state/amodal output, no SAM sign gate/prior, no reference/slot identity.
The prototype is input-only and computed from OTHER initial ROIs of the same
current target/image. A negative output is conditional nonvisibility, not a
universal external/outside/self/physical-absence prediction.
"""
from dataclasses import asdict,dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class OwnershipConfig:
    p3_channels:int=65
    p3_hidden:int=24
    appearance_channels:int=64
    relation_hidden:int=24
    rgb_channels:int=75
    rgb_hidden:int=16
    sam_channels:int=2
    sam_hidden:int=8
    context_channels:int=64
    context_hidden:int=16
    prototype_info_channels:int=4
    fusion_hidden:int=32
    prototype_rule:str='other_ROIs_only_soft_MobileSAM_box_valid_weighted_P3; context_fallback_flag'
    SAM_role:str='features_only_no_prior_no_veto'
    output_scope:str='visible_given_known_exact_V_or_E'


class TargetOwnershipHead(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config if isinstance(config,OwnershipConfig) else OwnershipConfig(**(config or {}));c=self.config
        if c.SAM_role!='features_only_no_prior_no_veto' or c.output_scope!='visible_given_known_exact_V_or_E':
            raise ValueError('Only the registered conditional ownership task is supported')
        self.appearance_projection=nn.Conv2d(c.appearance_channels,c.relation_hidden,1,bias=False)
        self.p3_encoder=nn.Sequential(nn.Conv2d(c.p3_channels,c.p3_hidden,3,padding=1),nn.GELU(),
                                     nn.Conv2d(c.p3_hidden,c.p3_hidden,3,padding=1),nn.GELU())
        self.rgb_encoder=nn.Sequential(nn.Conv2d(c.rgb_channels,c.rgb_hidden,1),nn.GELU(),
                                      nn.Conv2d(c.rgb_hidden,c.rgb_hidden,3,padding=1),nn.GELU())
        self.sam_encoder=nn.Sequential(nn.Conv2d(c.sam_channels,c.sam_hidden,3,padding=1),nn.GELU(),
                                      nn.Conv2d(c.sam_hidden,c.sam_hidden,3,padding=1),nn.GELU())
        self.context_encoder=nn.Sequential(nn.Linear(c.context_channels,c.context_hidden),nn.GELU())
        channels=c.p3_hidden+c.rgb_hidden+c.sam_hidden+c.context_hidden+3*c.relation_hidden+1+4+2+c.prototype_info_channels
        self.fusion=nn.Sequential(nn.Conv2d(channels,c.fusion_hidden,1),nn.GELU(),
                                  nn.Conv2d(c.fusion_hidden,c.fusion_hidden,3,padding=1),nn.GELU())
        self.readout=nn.Conv2d(c.fusion_hidden,1,1)

    def forward(self,x,rgb_native,sam,context,box_grid,prototype,prototype_info,return_dense=False):
        n=len(x);c=self.config
        for value,shape,name in ((x,(n,65,9,9),'x'),(rgb_native,(n,75,9,9),'rgb_native'),
            (sam,(n,2,9,9),'sam'),(context,(n,64),'context'),(box_grid,(n,4,9,9),'box_grid'),
            (prototype,(n,64),'prototype'),(prototype_info,(n,4),'prototype_info')):
            if tuple(value.shape)!=shape:raise ValueError(name+': expected '+str(shape))
        query=F.normalize(self.appearance_projection(x[:,:64]),dim=1,eps=1e-6)
        target=F.normalize(self.appearance_projection(prototype[:,:,None,None]),dim=1,eps=1e-6).expand(-1,-1,9,9)
        product=query*target;absolute_difference=(query-target).abs();cosine=product.sum(1,keepdim=True)
        ctx=self.context_encoder(context)[:,:,None,None].expand(-1,-1,9,9)
        info=prototype_info[:,:,None,None].expand(-1,-1,9,9)
        axis=torch.linspace(-1,1,9,device=x.device,dtype=x.dtype);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        position=torch.stack((xx,yy))[None].expand(n,-1,-1,-1)
        value=torch.cat((self.p3_encoder(x),self.rgb_encoder(rgb_native),self.sam_encoder(sam),ctx,
                         target,absolute_difference,product,cosine,box_grid,position,info),1)
        dense=self.readout(self.fusion(value))[:,0]
        center=dense[:,4,4]
        return (center,dense) if return_dense else center


def parameter_count(model):return sum(p.numel() for p in model.parameters())


def save_checkpoint(path,model,**metadata):
    torch.save(dict(format='V70_CONDITIONAL_OWNERSHIP_V1',config=asdict(model.config),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},**metadata),Path(path))


def load_model(path,device='cpu'):
    payload=torch.load(Path(path),map_location='cpu',weights_only=False)
    if payload['format']!='V70_CONDITIONAL_OWNERSHIP_V1':raise ValueError('Ownership checkpoint format')
    model=TargetOwnershipHead(payload['config']);model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device).eval(),payload
