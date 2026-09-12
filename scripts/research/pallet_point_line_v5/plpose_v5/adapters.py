"""Small explicit adapters; no hidden repo imports, guessed strides or source GT fields."""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
from .geometry import sample_features
from .contracts import Observation

@contextmanager
def capture_module_inputs(module:nn.Module,detach:bool=False):
    """Attach to the verified detector/head module and run exactly ONE real forward.
    detach=False preserves the online gradient graph. Never call no_grad here.
    The caller must retain and compare the SAME forward's detector return value.
    """
    record={'calls':0,'features':None}
    def hook(mod,args):
        record['calls']+=1
        x=args[0]
        if not isinstance(x,(tuple,list)) or not all(torch.is_tensor(t) for t in x):
            raise ValueError('Expected verified feature-list input')
        record['features']=tuple(t.detach().clone() if detach else t.clone() for t in x)
    handle=module.register_forward_pre_hook(hook)
    try:
        yield record
        if record['calls']!=1:raise ValueError(f'Expected one forward, saw {record["calls"]}')
    finally:handle.remove()

class MultiScaleROI(nn.Module):
    """Native feature planes -> one per-PREDICTED-instance feature grid.
    Features/affines supplied per ROI (gather image index in repo adapter).
    Explicit transforms support non-square images and offset conventions.
    """
    def __init__(self,channels=(64,128),per_level=16,grid=24):
        super().__init__();self.grid=grid
        self.projections=nn.ModuleList(nn.Conv2d(c,per_level,1) for c in channels)
    def forward(self,features,affines,contents,box,K,dims,image_hw):
        if not (len(features)==len(affines)==len(contents)==len(self.projections)):
            raise ValueError('Feature plane contract mismatch')
        samples=[];masks=[]
        for f,A,m,projector in zip(features,affines,contents,self.projections):
            z=projector(f*m)*m
            sample,mask=sample_features(z,A,box,(self.grid,self.grid),m)
            samples.append(sample);masks.append(mask)
        valid=torch.stack(masks).all(0)
        return Observation(torch.cat(samples,1)*valid,valid,box,K,dims,image_hw)


def feature_affine(raw_to_input:torch.Tensor,stride_xy:tuple[float,float],center_offset_xy:tuple[float,float]):
    """Explicit input pixel CENTER of feature index0. Do not guess (.5s vs0)."""
    if raw_to_input.ndim!=3 or raw_to_input.shape[-2:]!=(3,3):raise ValueError('B,3,3 affine required')
    sx,sy=stride_xy;ox,oy=center_offset_xy
    if min(sx,sy)<=0:raise ValueError('Positive strides')
    local=torch.tensor([[1/sx,0,-ox/sx],[0,1/sy,-oy/sy],[0,0,1]],dtype=raw_to_input.dtype,device=raw_to_input.device)
    return local[None]@raw_to_input


def reframe_pose(R_old,t_old,old_from_new):
    """X_old = A X_new (same origin). Only an independently verified proper rotation."""
    from .contracts import validate_rotation
    validate_rotation(old_from_new)
    return R_old@old_from_new,t_old


def permutation_to_new(points_old,valid_old,new_to_old):
    """Explicit per-instance bijection. Cannot infer this from nearest GT at inference."""
    if new_to_old.shape!=valid_old.shape or new_to_old.shape[-1]!=9:raise ValueError('B,9 index map required')
    if not torch.equal(torch.sort(new_to_old,-1).values,torch.arange(9,device=new_to_old.device).expand_as(new_to_old)):
        raise ValueError('Expected a bijection')
    return torch.gather(points_old,1,new_to_old[...,None].expand(-1,-1,2)),torch.gather(valid_old,1,new_to_old)
