"""ONE per-object symmetry choice shared by point, line, pose and seed losses.

The hypotheses are not averaged as rotations/coordinates. Their errors can be
weighted during training; inference chooses using predicted evidence only.
All weights are engineering pilot defaults, NOT validated optimum values.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import torch
from torch import Tensor
import torch.nn.functional as F
from .contracts import Observation,Supervision,symmetry_permutations,validate_rotation
from .geometry import cuboid,project,transform_points,EDGES

@dataclass
class LossConfig:
    point:float=1.
    line:float=.05
    pose:float=.2
    seed:float=.1
    ranking:float=.02
    line_target_sigma:float=.05     # normalized ROI distance; no edge visibility meaning
    score_temperature:float=1.
    gt_ranking_temperature:float=.1


def masked_mean(value,mask):
    return (torch.where(mask,value,torch.zeros_like(value))).sum()/mask.sum().clamp_min(1)


def segment_in_square(a:Tensor,b:Tensor):
    """Slab intersection for [-1,1]^2; includes segments with both endpoints outside."""
    delta=b-a
    parallel=delta.abs()<1e-10
    safe=torch.where(parallel,torch.ones_like(delta),delta)
    t0=(-1-a)/safe;t1=(1-a)/safe
    lower=torch.minimum(t0,t1);upper=torch.maximum(t0,t1)
    lower=torch.where(parallel,torch.full_like(lower,-1e8),lower)
    upper=torch.where(parallel,torch.full_like(upper,1e8),upper)
    enter=lower.amax(-1).clamp_min(0);exit=upper.amin(-1).clamp_max(1)
    return (enter<=exit)&(~(parallel&((a < -1)|(a > 1))).any(-1))


def line_supervision(logits,bin_valid,line_grid,target_points,target_valid,A,sigma):
    q=transform_points(target_points[None],A[None])[0]
    edges=torch.tensor(EDGES,device=q.device)
    a,b=q[edges[:,0]],q[edges[:,1]]
    supported=target_valid[edges[:,0]]&target_valid[edges[:,1]]
    supported &= segment_in_square(a,b)&((a-b).square().sum(-1)>1e-8)&bin_valid.any(-1)
    # A physical visibility target is intentionally NOT made from v>0.
    da=a@line_grid[:,:2].T+line_grid[:,2]
    db=b@line_grid[:,:2].T+line_grid[:,2]
    dist=.5*(da.square()+db.square())
    target_logits=(-dist/(2*sigma*sigma)).masked_fill(~bin_valid,-1e4)
    target_prob=torch.softmax(target_logits,-1).detach()
    pred_logprob=torch.log_softmax(logits.masked_fill(~bin_valid,-1e4),-1)
    ce=-(target_prob*pred_logprob).sum(-1)
    return masked_mean(ce,supported),supported


def compute_loss(output:dict,obs:Observation,gt:Supervision,line_grid:Tensor,cfg:LossConfig|None=None):
    cfg=cfg or LossConfig();b=len(obs.dims)
    if gt.points.shape!=(b,9,2) or gt.valid.shape!=(b,9) or len(gt.specs)!=b:
        raise ValueError('Supervision shape mismatch')
    validate_rotation(gt.R)
    if not torch.isfinite(gt.t).all() or not torch.isfinite(gt.points[gt.valid]).all():raise ValueError('Invalid supported GT')
    if not gt.valid.any():raise ValueError('No valid supervised points in training batch')
    # Ground truth cannot acquire gradients or change observation inputs.
    clean=torch.where(gt.valid[...,None],gt.points.detach(),torch.zeros_like(gt.points)).detach()
    totals=[];chosen=[];components=[];candidate_costs=[]
    for i in range(b):
        G,perms=symmetry_permutations(obs.dims[i].detach(),gt.specs[i])
        X=cuboid(obs.dims[i])
        world_pred=torch.einsum('mij,nj->mni',output['all_R'][i],X)+output['all_t'][i,:,None]
        world_seed=torch.einsum('mij,nj->mni',output['seed_R'][i],X)+output['seed_t'][i,:,None]
        diag=torch.linalg.vector_norm(obs.image_hw[i].to(clean.dtype))
        object_diag=torch.linalg.vector_norm(obs.dims[i])
        symmetry_losses=[];symmetry_parts=[]
        for g,perm in zip(G,perms):
            target=clean[i,perm];mask=gt.valid[i,perm]
            diff=(output['measurements'].points[i]-target)/diag
            pl=masked_mean(F.smooth_l1_loss(diff,torch.zeros_like(diff),beta=.01,reduction='none').sum(-1),mask)
            ll,supported=line_supervision(output['line_logits'][i],output['line_bin_valid'][i],line_grid,
                                           target,mask,output['raw_to_grid'][i],cfg.line_target_sigma)
            Rtarget=gt.R[i].detach()@g
            target_world=torch.einsum('ij,nj->ni',Rtarget,X)+gt.t[i].detach()[None]
            # Eight corners only; center is not an extra independent 3D observation.
            errors=torch.linalg.vector_norm(world_pred[:,:8]-target_world[None,:8],dim=-1).mean(-1)/object_diag
            seed_errors=torch.linalg.vector_norm(world_seed[:,:8]-target_world[None,:8],dim=-1).mean(-1)/object_diag
            scores=-output['energy'][i]/cfg.score_temperature
            probabilities=torch.softmax(scores,-1)
            pose_loss=(probabilities*errors).sum()
            # Supervised best-of-starts is TRAINING ONLY, never inference selection.
            seed_loss=seed_errors.min()
            target_rank=torch.softmax(-errors.detach()/cfg.gt_ranking_temperature,-1)
            ranking=-(target_rank*torch.log_softmax(scores,-1)).sum()
            total=cfg.point*pl+cfg.line*ll+cfg.pose*pose_loss+cfg.seed*seed_loss+cfg.ranking*ranking
            symmetry_losses.append(total)
            symmetry_parts.append(torch.stack((pl,ll,pose_loss,seed_loss,ranking)))
        values=torch.stack(symmetry_losses)
        index=values.detach().argmin()
        totals.append(values[index]);chosen.append(int(index));components.append(torch.stack(symmetry_parts)[index])
        candidate_costs.append(values.detach().cpu().tolist())
    loss=torch.stack(totals).mean()
    parts=torch.stack(components).mean(0)
    names=('point','line','pose','seed','ranking')
    return loss,dict(components={n:float(v.detach()) for n,v in zip(names,parts)},
                     symmetry_index=chosen,symmetry_total_costs=candidate_costs,
                     supervised_points=int(gt.valid.sum()),
                     finite=bool(torch.isfinite(loss)),shared_symmetry_for_all_terms=True,
                     warning='Coordinates supervised by v>0 are not edge-visibility labels; loss decrease is not accuracy evidence.')
