"""GT-only loss boundary. This module never selects model inputs or queries."""
import torch
from torch.nn import functional as F

def _present_class_mean(logits,positive,negative):
    target=positive.to(logits.dtype)
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    pos=positive.to(logits.dtype);neg=negative.to(logits.dtype)
    pc=pos.sum((-2,-1));nc=neg.sum((-2,-1))
    pm=(bce*pos).sum((-2,-1))/pc.clamp_min(1)
    nm=(bce*neg).sum((-2,-1))/nc.clamp_min(1)
    present=(pc>0).to(logits.dtype)+(nc>0).to(logits.dtype)
    # Empty components retain a differentiable graph with exactly zero gradient.
    return (pm+nm)/present.clamp_min(1),present

def dense_support_losses(aux_mask_logits,valid9,expected_support_M0,visible_target_M4):
    """Per-query present-positive/negative class means, then fixed batch mean.

    Channel0 M0: both target support and outside-support negatives are supervised.
    Channel1 V|M0: V is positive and Cmask is negative; outside-support is ignored.
    Invalid/unknown native cells are ignored in BOTH channels, never C/O labels.
    Supports flat N or F,8 leading dimensions without changing denominators.
    """
    if aux_mask_logits.shape[-3:]!=(2,9,9):
        raise ValueError('Auxiliary logits require [...,2,9,9]')
    expected=aux_mask_logits.shape[:-3]+(9,9)
    masks=(valid9,expected_support_M0,visible_target_M4)
    if not all(isinstance(x,torch.Tensor)and x.dtype==torch.bool and x.shape==expected for x in masks):
        raise TypeError('GT validity/M0/M4 must be separate bool tensors with matching leading shape')
    if not all(x.device==aux_mask_logits.device for x in masks):
        raise ValueError('Loss GT tensors and logits must share the device')
    if not torch.isfinite(aux_mask_logits).all():raise ValueError('Nonfinite auxiliary logits')
    if (valid9&visible_target_M4&~expected_support_M0).any():
        raise ValueError('Visible M4 must be a subset of M0 on known valid cells')
    z=aux_mask_logits.reshape(-1,2,9,9)
    v=valid9.reshape(-1,9,9);a=expected_support_M0.reshape(-1,9,9);t=visible_target_M4.reshape(-1,9,9)
    m0,count0=_present_class_mean(z[:,0],v&a,v&~a)
    cond,count1=_present_class_mean(z[:,1],v&a&t,v&a&~t)
    return dict(M0_loss=m0.mean(),VgivenM0_loss=cond.mean(),auxiliary_mean=(m0.mean()+cond.mean())/2,
                M0_per_query=m0,VgivenM0_per_query=cond,M0_present_classes=count0,VgivenM0_present_classes=count1)

def combined_support_loss(support_logit,primary_target,aux_mask_logits,valid9,expected_support_M0,visible_target_M4,aux_weight=.25):
    """Fixed primary mean BCE + lambda*(M0 loss + V|M0 loss)/2.

    Caller supplies only the unchanged admitted sampled query batch. This helper
    does not expand the sampler to support-U/missing/ref-MISS examples.
    """
    if support_logit.shape!=primary_target.shape:
        raise ValueError('Primary target shape must match the unchanged support logits')
    if aux_mask_logits.shape[:-3]!=support_logit.shape:
        raise ValueError('Auxiliary and primary query batches must be identical')
    if not ((primary_target==0)|(primary_target==1)).all():
        raise ValueError('Sampled primary labels must remain admitted known0/1; no U relabel')
    if aux_weight not in (0.,.25):
        raise ValueError('Only frozen lambda.25 and generated equivalence control lambda0 are supported')
    primary=F.binary_cross_entropy_with_logits(support_logit,primary_target.to(support_logit.dtype),reduction='mean')
    out=dense_support_losses(aux_mask_logits,valid9,expected_support_M0,visible_target_M4)
    return dict(loss=primary+aux_weight*out['auxiliary_mean'],primary_BCE=primary,**out)
