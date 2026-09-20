"""Fixed peak-local readout, no GT, coordinate prior, filter or learned gate."""
import torch
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation


def local_mode(logits):
    if logits.ndim!=4 or logits.shape[-2:]!=(96,72):
        raise ValueError('Expected B,K,96,72 PoseFix heatmap')
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
    # Fixed before any source/eval scoring; no radius/temperature sweep.
    flat=logits.flatten(2)
    anchor=flat.argmax(-1)
    ax=anchor%72;ay=anchor//72
    yy=torch.arange(96,device=logits.device)[None,None,:,None]
    xx=torch.arange(72,device=logits.device)[None,None,None,:]
    window=((xx-ax[...,None,None]).abs()<=2)&((yy-ay[...,None,None]).abs()<=2)
    selected=logits.masked_fill(~window,-torch.inf)
    return expectation(selected)


def ambiguity(logits):
    """GT-free diagnostics only, never used to gate or alter predictions."""
    mean=expectation(logits);mode=local_mode(logits)
    flat=logits.flatten(2);p=flat.softmax(-1)
    anchor=flat.argmax(-1);ax=anchor%72;ay=anchor//72
    yy=torch.arange(96,device=logits.device)[None,None,:,None]
    xx=torch.arange(72,device=logits.device)[None,None,None,:]
    window=((xx-ax[...,None,None]).abs()<=2)&((yy-ay[...,None,None]).abs()<=2)
    mass=(p.reshape_as(logits)*window).sum((-2,-1))
    return dict(mean=mean,mode=mode,local_mass=mass,entropy=-(p*p.clamp_min(1e-30).log()).sum(-1))
