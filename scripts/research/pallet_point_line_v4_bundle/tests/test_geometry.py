import pytest,torch
from pointline_v4.geometry import affine_points,sample_plane,finite_segment_queries,line_endpoint_cues
from conftest import make_observation


def test_affine_uses_explicit_offset():
    xy=torch.tensor([[[4.,6.],[8.,2.]]]);a=torch.tensor([[[.5,0,.3],[0,.5,-.2]]])
    assert torch.allclose(affine_points(xy,a),torch.tensor([[[2.3,2.8],[4.3,.8]]]))


def test_bilinear_ramp():
    y,x=torch.meshgrid(torch.arange(8.),torch.arange(9.),indexing='ij');f=(2*x+3*y)[None,None]
    q=torch.tensor([[[2.25,3.5],[6.,1.]]]);m=torch.ones_like(f,dtype=torch.bool)
    value,valid=sample_plane(f,q,m)
    assert valid.all();assert torch.allclose(value[...,0],2*q[...,0]+3*q[...,1],atol=1e-5)


def test_feature_corners_have_no_half_pixel_shift():
    f=torch.arange(16.).reshape(1,1,4,4);m=torch.ones_like(f,dtype=torch.bool)
    out,v=sample_plane(f,torch.tensor([[[0.,0.],[3.,3.]]]),m)
    assert v.all();assert torch.equal(out[...,0],torch.tensor([[0.,15.]]))


def test_invalid_nan_query_zero_and_finite():
    f=torch.ones(1,1,4,4);m=torch.ones_like(f,dtype=torch.bool)
    out,v=sample_plane(f,torch.tensor([[[float('nan'),1.],[-1.,2.],[6.,6.]]]),m)
    assert not v.any();assert torch.isfinite(out).all();assert not out.any()


def test_partial_padding_support_rejected():
    f=torch.ones(1,1,5,5);m=torch.ones_like(f,dtype=torch.bool);m[:,:,:,0]=False
    out,v=sample_plane(f,torch.tensor([[[.5,2.],[1.,2.]]]),m)
    assert v.tolist()==[[False,True]];assert out[0,0,0]==0


def test_padding_values_do_not_leak_into_sampling():
    f=torch.ones(1,1,5,5);m=torch.ones_like(f,dtype=torch.bool);m[:,:,0]=False
    q=torch.tensor([[[2.,2.],[2.,.5]]]);a,_=sample_plane(f,q,m)
    f[:,:,0]=1e9;b,_=sample_plane(f,q,m);assert torch.equal(a,b)


def test_vertical_horizontal_and_zero_length_segments():
    start=torch.tensor([[[1.,1.],[2.,2.],[3.,3.]]]);end=torch.tensor([[[1.,9.],[10.,2.],[3.,3.]]])
    q=finite_segment_queries(start,end)
    assert q.shape==(1,3,8,3,2);assert torch.isfinite(q).all()
    assert (q[0,0,:,1,1]>1).all() and (q[0,0,:,1,1]<9).all()
    assert torch.equal(q[0,2],torch.full((8,3,2),3.))


def test_line_sign_and_scale_invariance():
    o=make_observation();args=(o.layouts,o.line_h,o.line_logits,o.line_valid,o.diagonal)
    a=line_endpoint_cues(*args);b=line_endpoint_cues(o.layouts,-7*o.line_h,*args[2:])
    assert torch.allclose(a,b,atol=1e-5)


def test_invalid_line_mode_zero():
    o=make_observation();o.line_valid[:,:,1]=False;o.line_h[:,:,1]=float('nan')
    out=line_endpoint_cues(o.layouts,o.line_h,o.line_logits,o.line_valid,o.diagonal)
    assert torch.isfinite(out).all();assert not out[:,:,:,1].any()


def test_feature_gradients_exist():
    f=torch.randn(1,2,5,5,requires_grad=True);m=torch.ones(1,1,5,5,dtype=torch.bool)
    out,_=sample_plane(f,torch.tensor([[[2.2,3.1]]]),m);out.sum().backward()
    assert f.grad.abs().sum()>0
