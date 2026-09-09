"""Explicit feature-pixel coordinates, masked sampling and finite segments.

No guessed YOLO half-pixel offset lives here. The adapter must supply an audited
raw-pixel -> feature-pixel-centre affine for EVERY captured feature plane.
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F

EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def affine_points(xy: Tensor, affine: Tensor) -> Tensor:
    if xy.ndim < 3 or xy.shape[-1] != 2 or affine.shape != (xy.shape[0],2,3):
        raise ValueError('Expected [B,...,2] queries and [B,2,3] raw-to-feature affine')
    flat = xy.reshape(xy.shape[0],-1,2)
    out = torch.bmm(flat, affine[...,:2].transpose(1,2)) + affine[:,None,:,2]
    return out.reshape_as(xy)


def sample_plane(feature: Tensor, xy: Tensor, content_valid: Tensor) -> tuple[Tensor,Tensor]:
    """Sample feature CENTRE coordinates (0..W-1, 0..H-1).

    A query is usable only if ALL bilinear support is inside the content mask.
    Reflection/padding cannot be queried as positive evidence. The backbone's
    receptive fields can still contain padding context; this mask cannot undo it.
    Invalid queries are zeroed BEFORE grid_sample (which treats NaN specially).
    """
    if feature.ndim != 4:
        raise ValueError('Feature must be [B,C,H,W]')
    b,c,h,w = feature.shape
    if xy.shape[0] != b or xy.shape[-1] != 2 or content_valid.shape != (b,1,h,w):
        raise ValueError('Feature, query and mask shapes disagree')
    if content_valid.dtype != torch.bool:
        raise TypeError('content_valid must be bool')
    shape = xy.shape[1:-1]
    q = xy.reshape(b,-1,2)
    finite = torch.isfinite(q).all(-1)
    q = torch.where(finite[...,None], q, torch.zeros_like(q))
    extent = q.new_tensor([w-1,h-1])
    inside = finite & ((q >= 0) & (q <= extent)).all(-1)
    grid = (2*(q+.5)/q.new_tensor([w,h])-1)[:,None]
    masked = torch.where(content_valid, feature, torch.zeros_like(feature))
    val = F.grid_sample(masked, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    support = F.grid_sample(content_valid.to(feature.dtype), grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    usable = inside & (support[:,0,0] >= 1-1e-6)
    val = val[:,:,0].transpose(1,2)
    val = torch.where(usable[...,None],val,torch.zeros_like(val))
    return val.reshape(b,*shape,c), usable.reshape(b,*shape)


def finite_segment_queries(start: Tensor, end: Tensor, samples: int = 8,
                           offsets: tuple[float,...] = (-1.,0.,1.)) -> Tensor:
    """Ordered INTERIOR samples and normal offsets in feature-pixel units."""
    if start.shape != end.shape or start.shape[-1] != 2 or samples < 1:
        raise ValueError('Invalid segment endpoints or sample count')
    t = (torch.arange(samples,device=start.device,dtype=start.dtype)+.5)/samples
    delta = end-start
    normal = torch.stack((-delta[...,1],delta[...,0]),-1)
    normal = normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True).clamp_min(1e-8)
    centres = start[...,None,:]+delta[...,None,:]*t[:,None]
    cross = start.new_tensor(offsets)
    return centres[...,None,:] + normal[...,None,None,:]*cross[:,None]


def line_endpoint_cues(layouts: Tensor, line_h: Tensor, logits: Tensor,
                       valid: Tensor, diagonal: Tensor) -> Tensor:
    """[B,K,12,M,4]: absolute endpoint residuals, sigmoid logit, validity.

    Each pair of endpoints is tested against the SAME line mode. Absolute
    distances are invariant to homogeneous line scale/sign. Sigmoid values are
    scores, not calibrated probabilities that a structural edge is correct.
    """
    b,k = layouts.shape[:2]
    m = line_h.shape[2]
    if line_h.shape != (b,12,m,3) or logits.shape != (b,12,m) or valid.shape != (b,12,m):
        raise ValueError('Expected 12 roles and consistent line modes')
    h = torch.where(valid[...,None],line_h,torch.zeros_like(line_h))
    norm = torch.linalg.vector_norm(h[...,:2],dim=-1)
    valid = valid & torch.isfinite(h).all(-1) & torch.isfinite(logits) & (norm>1e-8)
    h = torch.where(valid[...,None],h/norm.clamp_min(1e-8)[...,None],torch.zeros_like(h))
    uv = torch.tensor(EDGES,device=layouts.device)
    first,second = layouts[:,:,uv[:,0]], layouts[:,:,uv[:,1]]
    sigma = (.01*diagonal[:,None,None,None]).clamp_min(1.)
    d1 = ((first[...,None,:]*h[:,None,...,:2]).sum(-1)+h[:,None,...,2]).abs()/sigma
    d2 = ((second[...,None,:]*h[:,None,...,:2]).sum(-1)+h[:,None,...,2]).abs()/sigma
    mass = torch.where(valid,logits,torch.zeros_like(logits)).sigmoid()[:,None].expand(-1,k,-1,-1)
    mask = valid[:,None].expand(-1,k,-1,-1)
    cue = torch.stack((d1,d2,mass,mask.to(layouts.dtype)),-1)
    return torch.where(mask[...,None],cue,torch.zeros_like(cue))
