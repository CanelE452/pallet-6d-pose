import numpy as np
import torch
from . import dino_tail_affine_model as A


def test_identity_feature_and_coordinate_exact():
    torch.manual_seed(4)
    feature=torch.randn(2,3,56,42);points=torch.randn(2,9,2)*20+100
    a=torch.eye(3).repeat(2,1,1)
    assert torch.equal(A.warp(feature,a),feature)
    assert torch.equal(A.coordinates(points,a),points)


def test_integer_token_translation_matches_label_translation():
    f=torch.zeros(1,1,56,42);f[0,0,20,15]=1
    a=torch.eye(3)[None];a[0,0,2]=576/42*3;a[0,1,2]=768/56*2
    z=A.warp(f,a)
    assert z.reshape(-1).argmax().item()==22*42+18
    assert abs(z[0,0,22,18].item()-1)<1e-5
    point=torch.tensor([[[(15+.5)*576/42-.5,(20+.5)*768/56-.5]]])
    expected=torch.tensor([[[(18+.5)*576/42-.5,(22+.5)*768/56-.5]]])
    # FP32 at 300px has an ULP of about3e-5px.
    torch.testing.assert_close(A.coordinates(point,a),expected,atol=4e-5,rtol=0)


def test_sampling_reproducible_and_no_reflections():
    a=A.matrices(256,100);assert torch.equal(a,A.matrices(256,100))
    assert not torch.equal(a,A.matrices(256,101))
    identity=(a==torch.eye(3)).all(-1).all(-1).sum().item()
    assert 80<identity<180
    determinant=torch.linalg.det(a[:,:2,:2])
    assert (determinant>=.85**2-1e-6).all() and (determinant<=1.15**2+1e-6).all()
    points=torch.randn(256,9,2)*200
    torch.testing.assert_close(A.coordinates(A.coordinates(points,a),torch.linalg.inv(a)),points,atol=1e-3,rtol=0)
    # Independent high-precision inverse verifies the convention, not just a
    # relaxed FP32 round-trip threshold.
    a=a.double();points=points.double()
    torch.testing.assert_close(A.coordinates(A.coordinates(points,a),torch.linalg.inv(a)),points,atol=1e-9,rtol=0)


def test_transformed_out_of_support_is_masked_and_original_not_mutated():
    points=torch.tensor([[[575.,100.],[100.,200.],[-1.,-1.]]])
    b=dict(feature=(torch.ones(1,1,56,42),torch.ones(1,1,56,42)),points=points,target=points,
           valid=torch.tensor([[True,True,False]]),target_valid=torch.tensor([[True,False,False]]))
    a=torch.eye(3)[None];a[:,0,2]=20
    out=A.augment(b,a)
    assert out['valid'].tolist()==[[False,True,False]]
    assert not out['target_valid'].any()
    assert b['target_valid'].tolist()==[[True,False,False]]
    assert b['points'][0,0,0]==575


def test_nonidentity_warp_keeps_gradients_finite():
    f=torch.randn(2,2,56,42,requires_grad=True)
    a=torch.eye(3).repeat(2,1,1);a[:,0,2]=7
    A.warp(f,a).square().mean().backward()
    assert torch.isfinite(f.grad).all() and (f.grad.abs().sum()>0)
