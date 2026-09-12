"""Fail-closed inference/supervision and symmetry contracts. No asset inference from shape."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .geometry import cuboid, rotation_y, EDGES

@dataclass
class Observation:
    features: Tensor           # B,C,H,W; real adapter supplies predicted-instance ROI features
    content: Tensor            # B,1,H,W; preprocessing support, never object GT mask
    box: Tensor                # B,4; predicted ROI in raw undistorted image pixels
    K: Tensor                  # B,3,3, same raw frame
    dims: Tensor               # B,3=(W,D,H) metres, deployable model metadata
    image_hw: Tensor           # B,2

    def validate(self):
        b,c,h,w=self.features.shape
        expected={'content':(b,1,h,w),'box':(b,4),'K':(b,3,3),'dims':(b,3),'image_hw':(b,2)}
        for name,shape in expected.items():
            x=getattr(self,name)
            if tuple(x.shape)!=shape or not torch.isfinite(x).all():
                raise ValueError(f'Invalid observation {name}, expected {shape}')
        if not torch.isfinite(self.features).all(): raise ValueError('Nonfinite features')
        if not ((self.content==0)|(self.content==1)).all(): raise ValueError('Binary content support required')
        if not self.content.flatten(1).any(1).all(): raise ValueError('No image content in a selected ROI')
        if not (self.dims>0).all() or not (self.image_hw>0).all(): raise ValueError('Positive dimensions/image size')
        if not (self.box[:,2:]>self.box[:,:2]).all(): raise ValueError('Invalid predicted ROI')
        if not torch.allclose(self.K[:,2],self.K.new_tensor([0.,0.,1.]).expand(b,-1),atol=1e-6):
            raise ValueError('K must be pinhole intrinsics; undistort beforehand')
        if not ((self.K[:,0,0]>0)&(self.K[:,1,1]>0)).all(): raise ValueError('Positive focal lengths required')
        return self

    def to(self,device):
        return Observation(**{k:v.to(device) for k,v in vars(self).items()})

@dataclass(frozen=True)
class SymmetrySpec:
    order: int
    verified: bool
    basis: str

    def matrices(self,dims:Tensor):
        if self.order not in (1,2,4): raise ValueError('Only upright C1/C2/C4 supported')
        if self.order>1 and (not self.verified or not self.basis.strip()):
            raise ValueError('Nontrivial symmetry requires a verified external contract')
        if dims.shape!=(3,): raise ValueError('One object dimension vector required')
        if self.order==4 and not torch.isclose(dims[0],dims[1],rtol=0,atol=1e-6):
            raise ValueError('C4 invalid for a rectangular fixed-axis cuboid')
        return rotation_y(torch.arange(self.order,device=dims.device,dtype=dims.dtype)*(2*math.pi/self.order))


def symmetry_permutations(dims:Tensor,spec:SymmetrySpec):
    X=cuboid(dims)
    G=spec.matrices(dims)
    rotated=torch.einsum('sij,nj->sni',G,X)
    distances=torch.cdist(rotated,X.expand(len(G),-1,-1))
    distance,permutation=distances.min(-1)
    if float(distance.max())>1e-5: raise ValueError('Transform is not a cuboid symmetry')
    if not all(torch.unique(p).numel()==9 and int(p[8])==8 for p in permutation):
        raise ValueError('Not a bijective center-preserving permutation')
    return G,permutation


def edge_permutations(point_permutations:Tensor):
    lookup={tuple(sorted(e)):i for i,e in enumerate(EDGES)}
    return torch.tensor([[lookup[tuple(sorted((int(p[a]),int(p[b]))))] for a,b in EDGES]
                         for p in point_permutations],device=point_permutations.device)

@dataclass
class Supervision:
    points: Tensor     # B,9,2; in NEW object-order; unknown may be NaN
    valid: Tensor      # B,9; coordinate supervision, NOT physical edge visibility
    R: Tensor          # B,3,3, NEW object-frame -> camera
    t: Tensor          # B,3 metres
    specs: list[SymmetrySpec]

    def to(self,device):
        return Supervision(self.points.to(device),self.valid.to(device),self.R.to(device),self.t.to(device),self.specs)


def validate_rotation(R:Tensor):
    eye=torch.eye(3,dtype=R.dtype,device=R.device)
    if not torch.isfinite(R).all() or not torch.allclose(R.transpose(-1,-2)@R,eye.expand_as(R),atol=1e-4) or not torch.allclose(torch.linalg.det(R),torch.ones_like(torch.linalg.det(R)),atol=1e-4):
        raise ValueError('Proper rotation required')
