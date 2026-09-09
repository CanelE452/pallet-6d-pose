"""Direct candidate supervision, inherited in spirit from structured-v2.

This is NOT a new 'utility learning' invention. No discrete candidate geometry
or target is optimized by this loss. Targets are detached from the model graph.
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def candidate_targets(layouts: Tensor, gt: Tensor, supervised: Tensor, diagonal: Tensor) -> dict[str,Tensor]:
    b,k = layouts.shape[:2]
    if gt.shape!=(b,9,2) or supervised.shape!=(b,9) or diagonal.shape!=(b,):
        raise ValueError('Target schema mismatch')
    if supervised.dtype!=torch.bool or (diagonal<=0).any():
        raise ValueError('Boolean supervision and positive diagonals required')
    # Sanitize BOTH operands before norm; 0*NaN is NOT safe masking.
    with torch.no_grad():
        valid=supervised[:,:8] & torch.isfinite(gt[:,:8]).all(-1)
        bad=supervised[:,:8] & ~torch.isfinite(gt[:,:8]).all(-1)
        if bad.any():
            raise ValueError('A supervised GT coordinate is nonfinite')
        target=torch.where(valid[...,None],gt[:,:8],torch.zeros_like(gt[:,:8]))
        q=torch.where(valid[:,None,:,None],layouts[:,:,:8].detach(),torch.zeros_like(layouts[:,:,:8]))
        error=torch.linalg.vector_norm(q-target[:,None],dim=-1)
        error=torch.where(valid[:,None],error,torch.zeros_like(error))
        count=valid.sum(-1)
        mean=error.sum(-1)/count[:,None].clamp_min(1)
        scale=.01*diagonal[:,None]
        return dict(error_px=error,mean_px=mean,frame_valid=count>0,mask=valid,
                    cost=torch.log1p(mean/scale),corner_cost=torch.log1p(error/scale[:,:,None]))


def quality_loss(cost: Tensor, corner_cost: Tensor, target: dict[str,Tensor],
                 candidate_valid: Tensor, *, temperature=.25, beta=.1,
                 regression_weight=1., corner_weight=.25) -> tuple[Tensor,dict]:
    if temperature<=0 or beta<=0 or cost.shape!=candidate_valid.shape:
        raise ValueError('Invalid loss configuration or candidate mask')
    active_frames=target['frame_valid'] & candidate_valid.any(-1)
    if not active_frames.any():
        raise ValueError('No supervised frame in this optimizer batch; do not silently count a fake update')
    pred=cost[active_frames]; truth=target['cost'][active_frames].detach()
    valid=candidate_valid[active_frames]
    logp=F.log_softmax((-pred/temperature).masked_fill(~valid,-1e9),-1)
    probability=F.softmax((-truth/temperature).masked_fill(~valid,-1e9),-1)
    ranking=-(probability*logp).sum(-1).mean()
    regression=F.smooth_l1_loss(pred[valid],truth[valid],beta=beta)
    cmask=candidate_valid[:,:,None]&target['mask'][:,None]&active_frames[:,None,None]
    corner=F.smooth_l1_loss(corner_cost[cmask],target['corner_cost'][cmask].detach(),beta=beta)
    loss=ranking+regression_weight*regression+corner_weight*corner
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    return loss,dict(ranking=float(ranking.detach()),regression=float(regression.detach()),corner=float(corner.detach()),
                     supervised_frames=int(active_frames.sum()),supervised_corners=int(target['mask'][active_frames].sum()))
