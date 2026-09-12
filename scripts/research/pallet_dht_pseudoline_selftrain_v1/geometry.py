"""Training-only semantic incidence constraints; no WLS or pseudo corners."""
import torch
from torch.nn import functional as F

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES


def transform_points(points, affine):
    homogeneous=torch.cat((points,torch.ones_like(points[..., :1])),-1)
    result=homogeneous@affine.transpose(-1,-2)
    return result[..., :2]/result[..., 2:]


def transform_lines(lines, affine):
    result=torch.linalg.solve(affine.transpose(-1,-2),lines.transpose(-1,-2)).transpose(-1,-2)
    norm=result[..., :2].norm(dim=-1,keepdim=True)
    if (norm<1e-12).any():raise ValueError('degenerate line transform')
    return result/norm


def line_weights(lines, ambiguity, available):
    finite=torch.isfinite(lines).all(-1)&torch.isfinite(ambiguity)&(lines[..., :2].norm(dim=-1)>1e-8)
    safe_ambiguity=torch.nan_to_num(ambiguity,nan=1.,posinf=1.,neginf=1.)
    return (1-safe_ambiguity).clamp(0,1)*available.to(lines.dtype)*finite.to(lines.dtype)


def incidence_loss(points, lines, weights, diagonal, edges=EDGES):
    """Per-frame loss, normal only. No center8, GT, matching, or teacher update."""
    safe=torch.where((weights>0)[...,None],lines,torch.zeros_like(lines))
    safe=safe/safe[..., :2].norm(dim=-1,keepdim=True).clamp_min(1e-8)
    endpoints=torch.stack([points[:,[a,b]] for a,b in edges],1)
    signed=(endpoints*safe[...,None,:2]).sum(-1)+safe[...,None,2]
    residual=signed/diagonal[:,None,None]
    penalty=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.01,reduction='none').sum(-1)
    return (penalty*weights).sum(-1)/weights.sum(-1).clamp_min(1e-8)


def baseline_assignment(points, target, valid, permutations, diagonal, point_valid=None):
    choices=[]
    for i,perm in enumerate(permutations):
        gt=target[perm];mask=valid[perm][:8]
        error=(points[:8]-gt[:8]).norm(dim=-1)
        if point_valid is not None:
            error=torch.where(point_valid[:8],error,torch.full_like(error,diagonal))
        choices.append((float(error[mask].mean()) if mask.any() else float(diagonal),i))
    index=min(choices)[1];perm=permutations[index]
    return target[perm],valid[perm],index


def gt_point_loss(points, target, target_valid, point_valid, diagonal):
    """Diagnostic GT loss: sanitize ignored NaNs BEFORE differentiable subtraction."""
    mask=target_valid[:8]
    safe=torch.where(mask[:,None],target[:8],points[:8].detach())
    error=(points[:8]-safe).norm(dim=-1)
    full=torch.where(point_valid[:8],error,torch.ones_like(error)*diagonal)
    return full[mask].mean()/diagonal if mask.any() else points.sum()*0
