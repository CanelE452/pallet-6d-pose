"""An explicit identity option; no signed correction gate, GT or residual path."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def select_layout(cost: Tensor, layouts: Tensor, valid: Tensor, margin: float|None) -> tuple[Tensor,Tensor]:
    """Strict cost-gap threshold. None means always keep candidate0.

    Ties preserve identity. Scores are ranking costs, NOT calibrated utility.
    Selection is discrete; this function is not advertised as differentiable.
    """
    if cost.ndim!=2 or valid.shape!=cost.shape or layouts.shape[:2]!=cost.shape or layouts.shape[2:]!=(9,2):
        raise ValueError('Invalid selection shapes')
    if valid.dtype!=torch.bool or not valid[:,0].all():
        raise ValueError('Identity must be valid for every frame')
    if not torch.isfinite(cost[valid]).all():
        raise ValueError('Nonfinite cost on a valid candidate')
    if margin is not None and (not math.isfinite(margin) or margin<0):
        raise ValueError('Margin must be nonnegative finite, or None for identity')
    base=layouts[:,0]
    if not torch.equal(layouts[:,:,8],base[:,None,8].expand(-1,layouts.shape[1],-1)):
        raise ValueError('Candidates may not modify centroid8')
    if margin is None:
        return base.clone(),torch.zeros(cost.shape[0],dtype=torch.long,device=cost.device)
    safe=cost.masked_fill(~valid,float('inf'))
    best=safe.argmin(-1)
    rows=torch.arange(len(best),device=cost.device)
    gap=cost[:,0]-safe[rows,best]
    chosen=torch.where(gap>margin,best,torch.zeros_like(best))
    output=layouts[rows,chosen].clone()
    if not torch.equal(output[:,8],base[:,8]):
        raise AssertionError('Centroid changed')
    return output,chosen
