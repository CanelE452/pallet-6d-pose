"""Four matched SHAPES, not a claim of equal active parameter counts.

P: endpoint evidence; S: + finite-segment interior evidence;
H: + role-specific Hough evidence; HA: H + same-ID original-point references.
P/S share the fixed joint backbone and candidate bank with H/HA. They are NOT
fully Hough-free models. The core predicts candidate quality, never GT masks.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from torch import Tensor, nn
from .geometry import EDGES, affine_points, sample_plane, finite_segment_queries, line_endpoint_cues

ARMS = ('P','S','H','HA')

@dataclass
class Observation:
    features: tuple[Tensor,...]
    raw_to_feature: tuple[Tensor,...]
    content_valid: tuple[Tensor,...]
    baseline: Tensor                 # [B,9,2], finite transport coordinates
    point_valid: Tensor              # [B,9], prediction validity, NEVER GT v
    point_conf: Tensor               # [B,9], predicted confidence
    layouts: Tensor                  # [B,K,9,2]; candidate0 exact identity
    candidate_valid: Tensor          # [B,K], prediction-only geometric checks
    line_h: Tensor                   # [B,12,4,3], raw homogeneous equations
    line_logits: Tensor              # [B,12,4], original logits, not softmax
    line_valid: Tensor               # [B,12,4], prediction-only mask
    diagonal: Tensor                 # [B], raw image diagonal

    @classmethod
    def from_mapping(cls, data: dict) -> 'Observation':
        allowed = {f.name for f in fields(cls)}
        if set(data) != allowed:
            raise ValueError(f'Observation schema mismatch: extra={set(data)-allowed}, missing={allowed-set(data)}')
        return cls(**data)

    def to(self, device) -> 'Observation':
        def move(x):
            return tuple(t.to(device) for t in x) if isinstance(x,(tuple,list)) else x.to(device)
        return Observation(**{f.name:move(getattr(self,f.name)) for f in fields(self)})

    def validate(self, channels: tuple[int,...]) -> None:
        if self.baseline.ndim != 3 or self.baseline.shape[1:] != (9,2):
            raise ValueError('baseline must be [B,9,2]')
        b = self.baseline.shape[0]
        q = self.layouts
        if q.ndim!=4 or q.shape[0]!=b or q.shape[2:]!=(9,2) or q.shape[1]<1:
            raise ValueError('layouts must be [B,K,9,2]')
        shapes = dict(point_valid=(b,9),point_conf=(b,9),candidate_valid=(b,q.shape[1]),
                      line_h=(b,12,4,3),line_logits=(b,12,4),line_valid=(b,12,4),diagonal=(b,))
        for name,shape in shapes.items():
            if getattr(self,name).shape != shape:
                raise ValueError(f'{name} expected {shape}')
        for name in ('point_valid','candidate_valid','line_valid'):
            if getattr(self,name).dtype != torch.bool:
                raise TypeError(f'{name} must be bool')
        for name in ('baseline','layouts','point_conf','diagonal','line_h','line_logits'):
            if getattr(self,name).dtype != torch.float32:
                raise TypeError(f'{name} must use the FP32 export contract')
        for name in ('baseline','layouts','point_conf','diagonal'):
            if not torch.isfinite(getattr(self,name)).all():
                raise ValueError(f'{name} must be finite; retain original missing values separately')
        if (self.diagonal <= 0).any():
            raise ValueError('Positive raw diagonals required')
        if not torch.equal(q[:,0],self.baseline):
            raise ValueError('Candidate0 must exactly preserve baseline')
        if not torch.equal(q[:,:,8],self.baseline[:,None,8].expand(-1,q.shape[1],-1)):
            raise ValueError('Every candidate must preserve centroid8 exactly')
        if not self.candidate_valid[:,0].all():
            raise ValueError('Identity fallback must remain valid')
        incomplete = ~self.point_valid.all(-1)
        if self.candidate_valid[incomplete,1:].any():
            raise ValueError('Incomplete prediction: only identity is allowed')
        if not(len(self.features)==len(self.raw_to_feature)==len(self.content_valid)==len(channels)):
            raise ValueError('Feature plane count mismatch')
        for feat,affine,mask,c in zip(self.features,self.raw_to_feature,self.content_valid,channels):
            if feat.ndim!=4 or feat.shape[:2]!=(b,c) or affine.shape!=(b,2,3) or mask.shape!=(b,1,*feat.shape[-2:]):
                raise ValueError('Feature/affine/content-mask mismatch')
            if feat.dtype!=torch.float32 or affine.dtype!=torch.float32:
                raise TypeError('Feature planes and inference affines must be FP32; keep FP64 provenance separately')
            if mask.dtype!=torch.bool or not torch.isfinite(affine).all() or not torch.isfinite(feat).all():
                raise ValueError('Invalid feature-plane payload')


class EvidenceVerifier(nn.Module):
    def __init__(self, arm='H', channels=(64,128), visual=16, width=64, layers=2):
        super().__init__()
        if arm not in ARMS or not channels or visual<1 or width%4 or layers<1:
            raise ValueError('Invalid model configuration')
        self.arm,self.channels,self.visual,self.width = arm,tuple(channels),visual,width
        self.config = dict(arm=arm,channels=list(channels),visual=visual,width=width,layers=layers)
        self.projections = nn.ModuleList(nn.Sequential(nn.Conv2d(c,visual,1),nn.GELU()) for c in channels)
        n = len(channels)
        # Equal architecture/state tensors across arms. Some input columns are
        # inactive when a cue is withheld; the training receipt reports this.
        self.corner = nn.Sequential(nn.Linear(n*(9*visual+9)+5,width),nn.LayerNorm(width),nn.GELU(),nn.Linear(width,width))
        self.edge = nn.Sequential(nn.Linear(n*(24*visual+24)+9+16,width),nn.LayerNorm(width),nn.GELU(),nn.Linear(width,width))
        self.roles = nn.Embedding(20,width)
        layer = nn.TransformerEncoderLayer(width,4,2*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.interaction = nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.quality = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.corner_quality = nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,1))
        self.register_buffer('edges',torch.tensor(EDGES,dtype=torch.long))
        self.register_buffer('patch_offsets',torch.tensor([(x,y) for y in (-1.,0.,1.) for x in (-1.,0.,1.)]))
        self.register_buffer('interior_t',(torch.arange(8)+.5)/8.)

    def forward(self, o: Observation) -> tuple[Tensor,Tensor]:
        if not isinstance(o,Observation):
            raise TypeError('Pass a validated Observation; targets must be separate')
        o.validate(self.channels)
        q=o.layouts[:,:,:8]; b,k=q.shape[:2]
        u,v=self.edges[:,0],self.edges[:,1]
        patches,segments=[],[]
        for feat,affine,mask,project in zip(o.features,o.raw_to_feature,o.content_valid,self.projections):
            z=project(torch.where(mask,feat,torch.zeros_like(feat)))
            z=torch.where(mask,z,torch.zeros_like(z))
            centre=affine_points(q,affine)
            patch,pvalid=sample_plane(z,centre[...,None,:]+self.patch_offsets,mask)
            patches.extend((patch.flatten(-2),pvalid.to(z.dtype)))
            if self.arm=='P':
                # Same-sized edge encoder; no extra interior queries. Interpolate
                # observed endpoint features rather than zero an entire branch.
                left,right=patch[:,:,u,4,:],patch[:,:,v,4,:]
                t=self.interior_t[None,None,None,:,None]
                ev=(left[...,None,:]*(1-t)+right[...,None,:]*t)[...,None,:].expand(-1,-1,-1,-1,3,-1)
                em=(pvalid[:,:,u,4]&pvalid[:,:,v,4])[...,None,None].expand(-1,-1,-1,8,3)
            else:
                query=finite_segment_queries(centre[:,:,u],centre[:,:,v])
                ev,em=sample_plane(z,query,mask)
            segments.extend((ev.flatten(-3),em.to(z.dtype).flatten(-2)))
        scale=o.diagonal[:,None,None,None]
        position=(q-o.baseline[:,None,8:9])/scale
        if self.arm=='HA':
            anchor=o.baseline[:,None,:8].expand(-1,k,-1,-1)
            conf=o.point_conf[:,None,:8,None].expand(-1,k,-1,-1)
        else:
            # SOFT distance-weighted set reference. Symmetric in original point
            # order, including exact-distance ties (unlike argmin/gather).
            base=o.baseline[:,None,None,:8]
            distance=((q[...,None,:]-base)/scale[...,None]).square().sum(-1)
            valid=o.point_valid[:,None,None,:8]
            weight=torch.softmax((-distance/.0025).masked_fill(~valid,-1e9),-1)
            weight=weight*valid.to(weight.dtype)
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
            anchor=(weight[...,None]*base).sum(-2)
            conf=(weight*o.point_conf[:,None,None,:8]).sum(-1,keepdim=True)
        displacement=(q-anchor)/scale
        corner=self.corner(torch.cat((*patches,position,displacement,conf),-1))
        direction=position[:,:,v]-position[:,:,u]
        egeom=torch.cat((position[:,:,u],position[:,:,v],direction,
                         torch.linalg.vector_norm(direction,dim=-1,keepdim=True),conf[:,:,u],conf[:,:,v]),-1)
        if self.arm in ('H','HA'):
            cues=line_endpoint_cues(o.layouts,o.line_h,o.line_logits,o.line_valid,o.diagonal).flatten(-2)
        else:
            cues=q.new_zeros(b,k,12,16)
        edge=self.edge(torch.cat((*segments,egeom,cues),-1))
        tokens=torch.cat((corner,edge),2)+self.roles.weight[None,None]
        tokens=self.norm(self.interaction(tokens.reshape(b*k,20,self.width))).reshape(b,k,20,self.width)
        return self.quality(tokens.mean(2)).squeeze(-1),self.corner_quality(tokens[:,:,:8]).squeeze(-1)

    def parameter_receipt(self) -> dict:
        params=list(self.named_parameters())
        return dict(registered=sum(p.numel() for _,p in params),trainable=sum(p.numel() for _,p in params if p.requires_grad),
                    nonzero_gradient_elements=sum(int(torch.count_nonzero(p.grad)) for _,p in params if p.grad is not None),
                    note='Nonzero gradients describe this batch, not structural active capacity. P/S withhold sixteen Hough input columns.')
