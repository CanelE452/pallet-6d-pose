"""Generated mathematical fixtures, NEVER real pallet evidence or benchmark data."""
from __future__ import annotations
import torch
from .geometry import cuboid,project,rotation_y,so3_exp,lines_from_points
from .contracts import Observation,Supervision,SymmetrySpec
from .solver import Measurements


def geometry_fixture(batch=1,dtype=torch.float64,noise=0.,seed=1):
    gen=torch.Generator().manual_seed(seed)
    dims=torch.tensor([1.1,1.3,.15],dtype=dtype).repeat(batch,1)
    K=torch.tensor([[500.,0,320.],[0,505.,240.],[0,0,1.]],dtype=dtype).repeat(batch,1,1)
    R=so3_exp(torch.tensor([.25,.55,.08],dtype=dtype).repeat(batch,1))@torch.diag(torch.tensor([1.,-1.,-1.],dtype=dtype))
    t=torch.tensor([.1,.06,3.5],dtype=dtype).repeat(batch,1)
    points,_=project(cuboid(dims),R,t,K)
    lines,valid=lines_from_points(points)
    observed=points+noise*torch.randn(points.shape,generator=gen,dtype=dtype)
    m=Measurements(observed,torch.ones(batch,9,dtype=torch.bool),torch.ones(batch,9,dtype=dtype)*2,
                   lines[:,:,None],valid[:,:,None],torch.zeros(batch,12,1,dtype=dtype),torch.ones(batch,12,1,dtype=dtype)*2)
    return dims,K,R,t,points,m


def generated_images(count=16,size=48,seed=7):
    """Draw projected cuboids only to exercise RGB/gradient/training I/O.
    These are not synthetic domain-generalization data; GT-derived boxes are
    EXPLICITLY diagnostic fixture ROI and forbidden as real inference ROI.
    """
    import numpy as np
    import cv2
    from .geometry import EDGES
    rng=np.random.default_rng(seed)
    images=[];dims=[];Ks=[];Rs=[];ts=[];points=[];boxes=[];specs=[]
    for i in range(count):
        order=(1,2,4)[i%3]
        d=torch.tensor([1.+rng.uniform(-.1,.1),1.3+rng.uniform(-.1,.1),.15],dtype=torch.float32)
        if order==4:d[1]=d[0]
        K=torch.tensor([[size*1.2,0,size/2],[0,size*1.2,size/2],[0,0,1]],dtype=torch.float32)
        R=so3_exp(torch.tensor([rng.uniform(.25,.5),rng.uniform(-.6,.6),rng.uniform(-.08,.08)],dtype=torch.float32))@torch.diag(torch.tensor([1.,-1.,-1.]))
        t=torch.tensor([rng.uniform(-.1,.1),rng.uniform(-.05,.05),rng.uniform(3.,4.)],dtype=torch.float32)
        uv,z=project(cuboid(d),R,t,K)
        canvas=np.zeros((size,size,3),np.float32)
        for e,(a,b) in enumerate(EDGES):
            cv2.line(canvas,tuple(np.round(uv[a].numpy()).astype(int)),tuple(np.round(uv[b].numpy()).astype(int)),
                     tuple(np.eye(3)[e%3].tolist()),1,lineType=cv2.LINE_AA)
        box=torch.cat((uv[:8].amin(0)-4,uv[:8].amax(0)+4))
        images.append(torch.from_numpy(canvas).permute(2,0,1));dims.append(d);Ks.append(K);Rs.append(R);ts.append(t);points.append(uv);boxes.append(box)
        specs.append(SymmetrySpec(order,True,'generated cuboid fixture geometry ONLY; not a real asset claim'))
    return dict(rgb=torch.stack(images),dims=torch.stack(dims),K=torch.stack(Ks),R=torch.stack(Rs),t=torch.stack(ts),
                points=torch.stack(points),box=torch.stack(boxes),specs=specs,image_hw=torch.tensor([size,size]).repeat(count,1).float(),
                scope='GENERATED_WIRING_ONLY',roi_origin='GT_DERIVED_DIAGNOSTIC_ONLY')
