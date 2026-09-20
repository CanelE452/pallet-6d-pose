import numpy as np
import torch
from . import dino_wide_model as W
from . import dino_localization_model as O


def test_matrix_doubles_fov_without_changing_resolution():
    a=np.array([[2.,0,-40],[0,2.,-60],[0,0,1.]])
    b=W.widen_matrix(a);np.testing.assert_array_equal(a[:2,:2],b[:2,:2])
    p=np.array([120.,160.,1.]);np.testing.assert_allclose((b@p)[:2]-(a@p)[:2],[144,192])
    np.testing.assert_array_equal(a,np.array([[2.,0,-40],[0,2.,-60],[0,0,1.]]))


def test_identical_parameters_and_initial_physical_prediction():
    torch.set_num_threads(2);torch.manual_seed(1);a=O.Head();torch.manual_seed(1);b=W.Head()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    p=torch.tensor([[[80.,100.]]*9]);valid=torch.ones(1,9,dtype=torch.bool)
    old=O.decode(a(torch.randn(1,384,28,21),p,valid))
    new=W.decode(b(torch.randn(1,384,56,42),p+torch.tensor([144.,192.]),valid))
    torch.testing.assert_close(old,new-torch.tensor([144.,192.]),atol=1e-4,rtol=0)


def test_outer_region_has_target_gradient_and_is_decodable():
    p=torch.tensor([[[224.,292.]]*9]);valid=torch.ones(1,9,dtype=torch.bool)
    logits=W.prior_logits(p,valid).detach().requires_grad_(True)
    target=torch.tensor([[[520.,700.]]*9]);W.loss(logits,target,valid).backward()
    assert logits.grad[0,0,175,130]<0 and logits.grad[:,8].abs().sum()==0
    with torch.no_grad():logits[:,0,175,130]=20
    torch.testing.assert_close(W.decode(logits)[0,0],target[0,0],atol=1e-3,rtol=0)


def test_unknown_and_outside_targets_not_filled():
    logits=torch.zeros(1,9,192,144,requires_grad=True);target=torch.full((1,9,2),-100.)
    value=W.loss(logits,target,torch.ones(1,9,dtype=torch.bool));assert value==0
    value.backward();assert logits.grad.abs().sum()==0
