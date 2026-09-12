import math
import pytest
import torch
from plpose_v5.geometry import *
from plpose_v5.contracts import *
from plpose_v5.objective import segment_in_square

torch.set_num_threads(2)

@pytest.mark.parametrize('dims',[[1.1,1.3,.11],[1.,1.,.15],[.8,.6,.14]])
def test_dimensions(dims):
    d=torch.tensor(dims,dtype=torch.float64);X=cuboid(d)
    assert X.shape==(9,3)
    torch.testing.assert_close(X[:8].amax(0)-X[:8].amin(0),d[[0,2,1]])
    assert not X[8].any()

@pytest.mark.parametrize('order',[1,2,4])
def test_symmetry_topology(order):
    d=torch.tensor([1.,1.,.15],dtype=torch.float64);spec=SymmetrySpec(order,True,'analytic fixture')
    G,p=symmetry_permutations(d,spec);ep=edge_permutations(p)
    assert ep.shape==(order,12)
    assert all(torch.unique(x).numel()==12 for x in ep)
    assert (p[:,8]==8).all()

@pytest.mark.parametrize('d',[[1.,1.1,.15],[1.,1.0001,.15]])
def test_reject_rectangular_c4(d):
    with pytest.raises(ValueError):SymmetrySpec(4,True,'fixture').matrices(torch.tensor(d))

def test_shape_not_enough_for_symmetry():
    with pytest.raises(ValueError):SymmetrySpec(4,False,'').matrices(torch.tensor([1.,1.,.1]))

@pytest.mark.parametrize('axis',range(6))
def test_projection_jacobian(axis):
    dtype=torch.float64
    X=cuboid(torch.tensor([[1.1,1.3,.15]],dtype=dtype))
    R=so3_exp(torch.tensor([[.25,-.4,.13]],dtype=dtype));t=torch.tensor([[.1,.2,3.]],dtype=dtype)
    K=torch.tensor([[[500.,0,320],[0,510.,240],[0,0,1]]],dtype=dtype)
    uv,z,J=project_jacobian(X,R,t,K)
    eps=1e-6;d=torch.zeros(1,6,dtype=dtype);d[0,axis]=eps
    rp,tp=update_pose(R,t,d);rm,tm=update_pose(R,t,-d)
    fd=(project(X,rp,tp,K)[0]-project(X,rm,tm,K)[0])/(2*eps)
    torch.testing.assert_close(fd,J[...,axis],rtol=1e-5,atol=1e-5)

def test_so3_zero_gradient():
    v=torch.zeros(1,3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(so3_exp,(v,))

def test_intrinsic_image_affine():
    X=cuboid(torch.tensor([[1.,1.3,.1]],dtype=torch.float64))
    R=torch.eye(3,dtype=torch.float64)[None];t=torch.tensor([[0.,0.,3.]],dtype=torch.float64)
    K=torch.tensor([[[500.,0,320],[0,500.,240],[0,0,1]]],dtype=torch.float64)
    A=torch.tensor([[[.7,0,13.],[0,.8,17.],[0,0,1]]],dtype=torch.float64)
    q,_=project(X,R,t,K);q2,_=project(X,R,t,A@K)
    torch.testing.assert_close(q2,transform_points(q,A))

def test_sampler_ramp_not_just_inverse_roundtrip():
    h,w=20,30
    yy,xx=torch.meshgrid(torch.arange(h,dtype=torch.float64),torch.arange(w,dtype=torch.float64),indexing='ij')
    f=torch.stack((xx,yy),0)[None]
    A=torch.tensor([[[.5,0,2.],[0,.25,3.],[0,0,1.]]],dtype=torch.float64)
    box=torch.tensor([[4.,8.,20.,40.]],dtype=torch.float64)
    out,valid=sample_features(f,A,box,(4,4),torch.ones(1,1,h,w,dtype=torch.float64))
    rx=4+(torch.arange(4,dtype=torch.float64)+.5)*4
    ry=8+(torch.arange(4,dtype=torch.float64)+.5)*8
    torch.testing.assert_close(out[0,0],(.5*rx+2)[None].expand(4,-1))
    torch.testing.assert_close(out[0,1],(.25*ry+3)[:,None].expand(-1,4))
    assert valid.all()

def test_content_mask_rejects_partial_bilinear_support():
    f=torch.ones(1,1,4,4);m=torch.ones_like(f);m[:,:,:,0]=0
    A=torch.eye(3)[None];box=torch.tensor([[0.,0.,2.,2.]])
    values,valid=sample_features(f,A,box,(2,2),m)
    assert not valid[0,0,:,0].any() and not values[0,0,:,0].any()

@pytest.mark.parametrize('a,b,result',[
    ([-2.,0],[2.,0],True),([-2.,2],[2.,2],False),([.5,.5],[.5,.7],True),
    ([-2.,1.5],[-1.5,2.],False),([-2.,0],[0.,2.],True)])
def test_segment_intersection(a,b,result):
    assert bool(segment_in_square(torch.tensor([a]),torch.tensor([b])))==result
