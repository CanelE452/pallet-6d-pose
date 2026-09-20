import torch
from .dino_localization_model import Head,prior_logits,loss,decode


def test_initialization_preserves_prior_and_inputs():
    torch.set_num_threads(2);torch.manual_seed(1)
    m=Head();q=torch.tensor([[[80.,100.]]*9]);v=torch.ones(1,9,dtype=torch.bool)
    before=q.clone();z=m(torch.randn(1,384,28,21),q,v)
    torch.testing.assert_close(z,prior_logits(q,v),atol=0,rtol=0)
    torch.testing.assert_close(decode(z),q,atol=1e-4,rtol=0)
    torch.testing.assert_close(q,before,atol=0,rtol=0)


def test_far_target_has_gradient_and_global_decoder_can_move_over100px():
    q=torch.tensor([[[20.,20.]]*9]);v=torch.ones(1,9,dtype=torch.bool)
    z=prior_logits(q,v).detach().requires_grad_(True)
    target=torch.tensor([[[220.,300.]]*9])
    loss(z,target,v).backward()
    assert z.grad[0,0,75,55]<0
    assert z.grad[:,8].abs().sum()==0
    with torch.no_grad():z[:,0,75,55]=20
    assert torch.linalg.norm(decode(z)[0,0]-q[0,0])>100


def test_invalid_targets_ignored_not_clamped():
    z=torch.zeros(1,9,96,72,requires_grad=True)
    target=torch.full((1,9,2),-100.);v=torch.ones(1,9,dtype=torch.bool)
    value=loss(z,target,v);assert value==0
    value.backward();assert z.grad.abs().sum()==0


def test_head_receives_finite_gradient():
    m=Head();q=torch.full((2,9,2),80.);v=torch.ones(2,9,dtype=torch.bool)
    z=m(torch.randn(2,384,28,21),q,v);l=loss(z,q+40,v);l.backward()
    assert torch.isfinite(l) and m.out.weight.grad.abs().sum()>0
