"""V237 pure Torch relational read proposal, selected by Root for V238 preparation.

No file/model/label access. Input token/image provenance is caller-owned. Base
keys match the existing CropFamilyHead exactly; no existing parameter is renamed.
"""
import math
import torch
from torch import nn

class CropFamilyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(1186,64),nn.ReLU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,features):
        if features.dtype!=torch.float32 or features.ndim!=2 or features.shape[1]!=1186:
            raise ValueError('Original FP32 active1186 required')
        return self.head(features)

class RelationalQueryResidual(nn.Module):
    """9956 parameters. No target IDs, states, masks, GT refs or model backbone.

    Eight nodes are full-CLS of original qj/Lj RAW96 from the SAME ACTUAL image.
    `local_supported8` means positive raw pixel support only, never semantic S/U.
    `relative_xy8` is existing input (qj-qi)/Li, clipped +/-8, FP32; no q change.
    Original availability and evidence support are separate from any confidence.
    """
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(384,16,bias=False)
        self.geometry_bias=nn.Linear(2,1,bias=False)
        self.readout=nn.Sequential(nn.Linear(116,32),nn.ReLU(),nn.Linear(32,2))
        nn.init.zeros_(self.geometry_bias.weight)
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def forward(self,active_features,query_local_cls8,target_context_cls,active_index,
                original_available8,local_supported8,context_supported,relative_xy8):
        n=len(active_features);device=active_features.device
        floats=((active_features,(n,1186)),(query_local_cls8,(n,8,384)),
                (target_context_cls,(n,384)),(relative_xy8,(n,8,2)))
        for x,shape in floats:
            if x.dtype!=torch.float32 or tuple(x.shape)!=shape or x.device!=device or not torch.isfinite(x).all():
                raise ValueError('Finite FP32 descriptors/geometry on one device required')
        for m,shape in ((original_available8,(n,8)),(local_supported8,(n,8)),(context_supported,(n,))):
            if m.dtype!=torch.bool or tuple(m.shape)!=shape or m.device!=device:raise ValueError('Separate bool original input/support masks required')
        if active_index.dtype!=torch.int64 or tuple(active_index.shape)!=(n,) or active_index.device!=device or not ((active_index>=0)&(active_index<8)).all():
            raise ValueError('Original logical active slot0..7, not GT instance index')
        if not (relative_xy8.abs()<=8).all():raise ValueError('Declared existing clipped relative geometry required')
        rows=torch.arange(n,device=device)
        active_available=original_available8[rows,active_index]
        active_pixel_supported=local_supported8[rows,active_index]
        # Active evidence is the original full96 CLS, not a replacement crop/read.
        active=active_features[:,:384]
        projected_active=self.project(active)
        projected_nodes=self.project(query_local_cls8)
        other=original_available8 & local_supported8
        other=other.clone();other[rows,active_index]=False
        score=(projected_nodes*projected_active[:,None,:]).sum(-1)/math.sqrt(16)
        score=score+self.geometry_bias(relative_xy8).squeeze(-1)
        weights=torch.softmax(score.masked_fill(~other,-1e4),dim=1)*other.to(torch.float32)
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        message=(projected_nodes*weights[:,:,None]).sum(1)
        context=torch.where(context_supported[:,None],target_context_cls,torch.zeros_like(target_context_cls))
        projected_context=self.project(context)
        flags=torch.stack((active_available.float(),active_pixel_supported.float(),other.sum(1).float()/7,context_supported.float()),1)
        condition32=active_features[:,-32:]
        read=torch.cat((projected_active,message,projected_active-message,
                        projected_active*message,projected_context,condition32,flags),1)
        if read.shape!=(n,116):raise AssertionError('Fixed relational feature width')
        return self.readout(read),{'attention_other7':weights,
            'original_active_available':active_available,'active_pixel_supported':active_pixel_supported,
            'other_count':other.sum(1),'context_supported':context_supported}

class RelationalQueryHead(nn.Module):
    """Unchanged76098 base +9956 zero-init residual =86054 parameters."""
    def __init__(self):
        super().__init__();self.base=CropFamilyHead();self.relational=RelationalQueryResidual()
    def load_original_initial_base(self,state_dict):
        # Caller supplies exact original initial head state; no checkpoint/file loader.
        self.base.load_state_dict(state_dict,strict=True)
    def forward(self,active_features,query_local_cls8,target_context_cls,active_index,
                original_available8,local_supported8,context_supported,relative_xy8):
        extra,details=self.relational(active_features,query_local_cls8,target_context_cls,
            active_index,original_available8,local_supported8,context_supported,relative_xy8)
        return self.base(active_features)+extra,details


def verify_active_node_alignment(active_features,query_local_cls8,active_index,
                                 original_available8,local_supported8,atol=1e-6):
    """Feature-build QA only. Never silently overwrite a mismatching active node.

    Uses FP32 unit-L2 fullCLS on both paths; no per-image fitted statistic.
    Root feature protocol must freeze tolerance before actual extraction.
    """
    rows=torch.arange(len(active_features),device=active_features.device)
    take=original_available8[rows,active_index]&local_supported8[rows,active_index]
    a=active_features[:,:384][take];b=query_local_cls8[rows,active_index][take]
    if not torch.allclose(a,b,rtol=0,atol=atol):raise ValueError('Original active fullCLS vs q8 node mismatch')
    return {'checked':int(take.sum()),'max_abs':float((a-b).abs().max()) if len(a) else 0.,'atol':float(atol)}


def fixed_family_decode(logits,family,original_active_available):
    """Zero logit threshold, U only from original input availability. No confidence."""
    n=len(logits)
    if logits.shape!=(n,2) or family.shape!=(n,) or family.dtype!=torch.int64 or original_active_available.dtype!=torch.bool or original_active_available.shape!=(n,):
        raise ValueError('Family0 MAIN/1 AUX and original availability required')
    if not ((family>=0)&(family<=1)).all():raise ValueError('Family route0/1 only')
    z=logits[torch.arange(n,device=logits.device),family]
    return torch.where(original_active_available,(z>=0).to(torch.int64),torch.full_like(family,-1))
