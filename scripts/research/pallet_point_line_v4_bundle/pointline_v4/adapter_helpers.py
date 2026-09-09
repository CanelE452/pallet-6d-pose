"""Reusable repository hookup helpers; actual checkpoint/GT contracts remain local.

These helpers make no forward call and never open annotations. They intentionally
require explicit module, affine convention, raw dimensions and prediction masks.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from .model import Observation


@dataclass
class Capture:
    calls: int=0
    features: tuple[Tensor,...]|None=None


def find_unique_hough_module(model: nn.Module) -> nn.Module:
    matches=[m for m in model.modules() if type(m).__name__=='HoughFeatureFusion']
    if len(matches)!=1:raise ValueError(f'Expected one verified HoughFeatureFusion, found {len(matches)}')
    return matches[0]


@contextmanager
def capture_native_features(module: nn.Module, *, detach: bool=True, cpu: bool=True):
    """Tap the audited HoughFeatureFusion INPUT P3/P4, before residual feedback.

    Warmups belong outside this context. More than one captured call is an
    error, preventing accidental mixture of augmentations or images. A live
    training adapter can request detach=False,cpu=False to retain gradients.
    """
    if cpu and not detach:raise ValueError('Live gradient capture must stay on device')
    state=Capture()
    def hook(_module,args):
        if len(args)!=1 or not isinstance(args[0],(tuple,list)) or len(args[0])!=3:
            raise ValueError('Expected a single P3/P4/P5 sequence argument')
        if state.calls:raise ValueError('Capture context must bind exactly one actual forward')
        chosen=[]
        for feature in args[0][:2]:
            if not torch.is_tensor(feature) or feature.ndim!=4:raise ValueError('Invalid feature tensor')
            value=feature.detach().clone() if detach else feature
            chosen.append(value.cpu() if cpu else value)
        state.features=tuple(chosen);state.calls+=1
    handle=module.register_forward_pre_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def explicit_feature_affine(raw_to_input: Tensor, *, stride_xy: tuple[float,float],
                            index_offset_xy: tuple[float,float]) -> Tensor:
    """index = input / stride + explicit offset; no default half-pixel guess."""
    if raw_to_input.ndim!=3 or raw_to_input.shape[1:]!=(2,3) or min(stride_xy)<=0:
        raise ValueError('Expected [B,2,3] affine and positive strides')
    out=raw_to_input.double().clone()
    out=out/out.new_tensor(stride_xy)[None,:,None]
    out[:,:,2]+=out.new_tensor(index_offset_xy)[None]
    if not torch.isfinite(out).all():raise ValueError('Nonfinite transform')
    return out.float()


def content_mask_from_raw_rectangle(raw_to_feature: Tensor,raw_hw: Tensor,feature_hw: tuple[int,int]) -> Tensor:
    """Mask FEATURE CENTRES mapped inside raw centre-domain [0,W-1]x[0,H-1].

    This does not certify feature receptive fields or object/edge visibility.
    Raw image shape must be supplied from actual image metadata, never a GT box.
    """
    b=raw_to_feature.shape[0];h,w=feature_hw
    if raw_to_feature.shape!=(b,2,3) or raw_hw.shape!=(b,2) or min(h,w)<1 or (raw_hw<=0).any():
        raise ValueError('Invalid rectangle/affine')
    a=torch.eye(3,dtype=torch.float64,device=raw_to_feature.device)[None].repeat(b,1,1)
    a[:,:2]=raw_to_feature.double()
    if (torch.linalg.det(a).abs()<1e-12).any():raise ValueError('Singular affine')
    inverse=torch.linalg.inv(a)
    y,x=torch.meshgrid(torch.arange(h,device=a.device,dtype=a.dtype),torch.arange(w,device=a.device,dtype=a.dtype),indexing='ij')
    grid=torch.stack((x,y,torch.ones_like(x)),-1).reshape(1,-1,3).expand(b,-1,-1)
    raw=torch.bmm(grid,inverse.transpose(1,2))[...,:2]
    wh=raw_hw[:,[1,0]].to(a.dtype)
    inside=((raw>=0)&(raw<=wh[:,None]-1)).all(-1)
    return inside.reshape(b,1,h,w)


V2_INPUT_KEYS={'p4','baseline_points','layouts','point_conf','diagonal','raw_to_input_affine',
               'input_shape_hw','line_h','peak_logits','peak_valid'}


def from_verified_v2_batch(batch: dict, *, native_features: tuple[Tensor,Tensor],
                           affines: tuple[Tensor,Tensor], content_masks: tuple[Tensor,Tensor],
                           prediction_valid: Tensor,candidate_valid: Tensor) -> Observation:
    """Map the READ structured-v2 model-input schema into the NEW contract.

    Caller must bind native_features to this exact prediction snapshot. p4 in
    the old batch is not used as a replacement for newly captured native P3/P4.
    Missing raw coordinates must be transported as finite placeholders with
    prediction_valid preserved, never silently converted into valid detections.
    """
    if set(batch)!=V2_INPUT_KEYS:
        raise ValueError('Not the verified v2 model-input contract; do not guess or include targets')
    def fp(x):return x.float()
    return Observation(features=tuple(fp(x) for x in native_features),raw_to_feature=tuple(fp(x) for x in affines),
        content_valid=tuple(x.bool() for x in content_masks),baseline=fp(batch['baseline_points']),
        point_valid=prediction_valid.bool(),point_conf=fp(batch['point_conf']),layouts=fp(batch['layouts']),
        candidate_valid=candidate_valid.bool(),line_h=fp(batch['line_h']),line_logits=fp(batch['peak_logits']),
        line_valid=batch['peak_valid'].bool(),diagonal=fp(batch['diagonal']).reshape(-1))
