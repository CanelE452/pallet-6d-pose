import torch
from . import dino_wide_visual_model as V
from . import dino_wide_model as W


def test_same_parameters_and_initialization():
    torch.set_num_threads(2);torch.manual_seed(3);a=W.Head();torch.manual_seed(3);b=V.Head()
    assert a.state_dict().keys()==b.state_dict().keys()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)


def test_output_independent_of_points_and_validity_after_learning():
    m=V.Head();torch.nn.init.normal_(m.out.weight);torch.nn.init.normal_(m.out.bias)
    f=torch.randn(1,384,56,42);p=torch.randn(1,9,2)*100
    a=m(f,p,torch.ones(1,9,dtype=torch.bool))
    b=m(f,p+1000,torch.zeros(1,9,dtype=torch.bool))
    c=m(f,torch.full_like(p,float('nan')),torch.ones(1,9,dtype=torch.bool))
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(a,c,atol=0,rtol=0)


def test_image_features_receive_gradient():
    m=V.Head();torch.nn.init.normal_(m.out.weight,std=.01)
    f=torch.randn(1,384,56,42,requires_grad=True);p=torch.full((1,9,2),200.)
    v=torch.ones(1,9,dtype=torch.bool);loss=V.loss(m(f,p,v),p,v);loss.backward()
    assert torch.isfinite(loss) and f.grad.abs().sum()>0 and m.project[0].weight.grad.abs().sum()>0


def test_unsupported_targets_and_center_not_supervised():
    z=torch.zeros(1,9,192,144,requires_grad=True);p=torch.full((1,9,2),-100.)
    p[:,8]=100.;v=torch.ones(1,9,dtype=torch.bool)
    loss=V.loss(z,p,v);loss.backward();assert loss==0 and z.grad.abs().sum()==0
