"""Metres, undistorted image pixels; camera right/down/forward.

NEW object frame: +x width, +y up (height), +z depth. This is NOT the
repo camera_dynamic ordering: use an explicit audited frame transform.
R maps object to camera. Rotation update Exp(dr) @ R, translation t + dt.
"""
from __future__ import annotations
import torch
from torch import Tensor

SIGNS = ((-1,1,-1),(1,1,-1),(1,-1,-1),(-1,-1,-1),
         (-1,1,1),(1,1,1),(1,-1,1),(-1,-1,1))
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7))


def cuboid(dims: Tensor, center: bool = True) -> Tensor:
    """dims[...,3]=(width,depth,height) metres -> [...,9 or 8,3]."""
    if dims.shape[-1] != 3 or not torch.isfinite(dims).all() or not (dims > 0).all():
        raise ValueError('Positive finite (width,depth,height) in metres required')
    signs = dims.new_tensor(SIGNS)
    xyz = dims[..., [0,2,1]].unsqueeze(-2) * signs / 2
    return torch.cat((xyz, torch.zeros_like(xyz[..., :1, :])), -2) if center else xyz


def skew(v: Tensor) -> Tensor:
    x,y,z = v.unbind(-1); zero = torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)


def so3_exp(v: Tensor) -> Tensor:
    # torch.matrix_exp avoids sqrt-at-zero backward singularities.
    return torch.matrix_exp(skew(v))


def rotation_y(angle: Tensor) -> Tensor:
    vec = torch.stack((torch.zeros_like(angle), angle, torch.zeros_like(angle)), -1)
    return so3_exp(vec)


def rotation_6d(x: Tensor) -> Tensor:
    """Gram-Schmidt; caller uses nondegenerate seed initialization."""
    a,b=x[...,:3],x[...,3:]
    u=torch.nn.functional.normalize(a,dim=-1,eps=1e-8)
    v=torch.nn.functional.normalize(b-(u*b).sum(-1,keepdim=True)*u,dim=-1,eps=1e-8)
    return torch.stack((u,v,torch.linalg.cross(u,v,dim=-1)),-1)


def project(X: Tensor, R: Tensor, t: Tensor, K: Tensor, z_min: float=1e-4):
    """Compatible leading dims, X[...,N,3]. Returns pixels and true camera Z."""
    RX=torch.einsum('...ij,...nj->...ni',R,X)
    Y=RX+t.unsqueeze(-2)
    h=torch.einsum('...ij,...nj->...ni',K,Y)
    z=Y[...,2]
    q=h[...,:2]/h[...,2:].clamp_min(z_min)
    return q,z


def project_jacobian(X: Tensor,R: Tensor,t: Tensor,K: Tensor,z_min:float=1e-4):
    """dq/d[dr,dt] for Exp(dr)R,t+dt; rotation uses RX, NOT RX+t."""
    RX=torch.einsum('...ij,...nj->...ni',R,X)
    Y=RX+t.unsqueeze(-2)
    h=torch.einsum('...ij,...nj->...ni',K,Y)
    den=h[...,2].clamp_min(z_min)
    Jproj=(K[...,:2,:].unsqueeze(-3)*den[...,None,None]
           -h[...,:2,None]*K[...,2:3,:].unsqueeze(-3))/den[...,None,None].square()
    eye=torch.eye(3,device=X.device,dtype=X.dtype).expand(*RX.shape[:-1],3,3)
    J=Jproj@torch.cat((-skew(RX),eye),-1)
    return h[...,:2]/den[...,None],Y[...,2],J


def update_pose(R:Tensor,t:Tensor,delta:Tensor):
    return so3_exp(delta[...,:3])@R, t+delta[...,3:]


def transform_points(points:Tensor, affine:Tensor):
    # affine[...,3,3], point[...,N,2]; projective allowed.
    homogeneous=torch.cat((points,torch.ones_like(points[...,:1])),-1)
    q=torch.einsum('...ij,...nj->...ni',affine,homogeneous)
    return q[...,:2]/q[...,2:].clamp_min(1e-8)


def lines_from_points(points:Tensor):
    edges=torch.as_tensor(EDGES,device=points.device)
    a,b=points[...,edges[:,0],:],points[...,edges[:,1],:]
    h1=torch.cat((a,torch.ones_like(a[...,:1])),-1)
    h2=torch.cat((b,torch.ones_like(b[...,:1])),-1)
    raw=torch.linalg.cross(h1,h2,dim=-1)
    n=torch.linalg.vector_norm(raw[...,:2],dim=-1,keepdim=True)
    return raw/n.clamp_min(1e-8),n.squeeze(-1)>1e-6


def normalize_line(h:Tensor):
    norm=torch.linalg.vector_norm(h[...,:2],dim=-1,keepdim=True)
    return h/norm.clamp_min(1e-8), norm.squeeze(-1)>1e-8


def lines_to_raw(lines_grid:Tensor, raw_to_grid:Tensor):
    # l_grid^T A x_raw = 0 -> l_raw=A^T l_grid.
    raw=torch.einsum('bij,beqj->beqi',raw_to_grid.transpose(-1,-2),lines_grid)
    return normalize_line(raw)


def roi_affine(box:Tensor):
    """Raw pixel box endpoints -> normalized [-1,1] coordinates (not a GT crop)."""
    if box.ndim!=2 or box.shape[-1]!=4 or not torch.isfinite(box).all():
        raise ValueError('Expected B,4 finite predicted ROI boxes')
    size=box[:,2:]-box[:,:2]
    if not (size>0).all(): raise ValueError('ROI must have positive width and height')
    c=(box[:,:2]+box[:,2:])/2
    A=torch.eye(3,dtype=box.dtype,device=box.device).repeat(len(box),1,1)
    A[:,0,0]=2/size[:,0];A[:,1,1]=2/size[:,1]
    A[:,0,2]=-2*c[:,0]/size[:,0];A[:,1,2]=-2*c[:,1]/size[:,1]
    return A


def sample_features(feature:Tensor,raw_to_feature:Tensor,box:Tensor,out_hw:tuple[int,int],content:Tensor):
    """Explicit raw->feature pixel-center map; full bilinear support must be valid.
    Reflection's receptive-field influence is NOT removed by this mask.
    """
    b,c,h,w=feature.shape; oh,ow=out_hw
    if raw_to_feature.shape!=(b,3,3) or content.shape!=(b,1,h,w):
        raise ValueError('Feature affine/content shape mismatch')
    if not torch.allclose(raw_to_feature[:,2], feature.new_tensor([0.,0.,1.]).expand(b,-1)):
        raise ValueError('Feature transform must be affine')
    yy,xx=torch.meshgrid((torch.arange(oh,device=feature.device,dtype=feature.dtype)+.5)/oh,
                        (torch.arange(ow,device=feature.device,dtype=feature.dtype)+.5)/ow,indexing='ij')
    uv=torch.stack((xx,yy),-1).reshape(1,-1,2)
    raw=box[:,:2,None].transpose(1,2)+uv*(box[:,2:]-box[:,:2])[:,None]
    q=transform_points(raw,raw_to_feature)
    grid=(2*(q+.5)/q.new_tensor([w,h])-1).reshape(b,oh,ow,2)
    vals=torch.nn.functional.grid_sample(feature*content,grid,align_corners=False,padding_mode='zeros')
    support=torch.nn.functional.grid_sample(content.to(feature.dtype),grid,align_corners=False,padding_mode='zeros')
    valid=support>=1-1e-5
    return vals*valid,valid
