"""Independent geometry/correspondence checks for the train-only coupling loss."""
import math

import pytest
import torch

from scripts.research.pallet_dht_coupling_v2.incidence import (
    decode_points_px, incidence_loss, pair_mixture_incidence,
)


def setup_case(batch_size=1, anchors=2, dtype=torch.float64):
    h, w, t, r = 4, 6, 18, 17
    theta = torch.arange(t, dtype=dtype) * (math.pi / t)
    rho = torch.arange(-8, 9, dtype=dtype) * .5
    extent = theta.cos().abs() * w / 2 + theta.sin().abs() * h / 2
    lattice = dict(height=h, width=w, theta_bins=t, rho_bins=r, rho_step=.5,
        rho_max=4., theta_values=theta, rho_values=rho,
        valid=rho[None].abs() <= extent[:, None] + 1e-6)
    corners = torch.tensor([[.2,.3],[.7,.3],[.7,.6],[.2,.6],
                            [.3,.4],[.8,.4],[.8,.7],[.3,.7],[.5,.5]], dtype=dtype)
    gt = torch.cat((corners, torch.full((9, 1), 2., dtype=dtype)), -1)[None].repeat(batch_size, 1, 1)
    points = (corners * corners.new_tensor([60., 40.]))[None, None].repeat(batch_size, anchors, 1, 1).requires_grad_()
    logits = torch.linspace(-2., 1., batch_size * 12 * t * r, dtype=dtype).reshape(batch_size, 12, t, r).requires_grad_()
    foreground = torch.ones(batch_size, anchors, dtype=torch.bool)
    indices = torch.zeros(batch_size, anchors, dtype=torch.long)
    image = torch.arange(batch_size)
    return logits, lattice, points, foreground, indices, gt, image, (40, 60)


def test_actual_stock_decode_and_gradient():
    from ultralytics.utils.loss import PoseLoss26
    raw = torch.randn(2, 27, 7, dtype=torch.float64, requires_grad=True)
    anchors = torch.randn(7, 2, dtype=torch.float64)
    stride = torch.tensor([8,8,8,16,16,32,32], dtype=torch.float64)[:, None]
    actual = decode_points_px({'kpts': raw}, anchors, stride)
    stock = PoseLoss26.kpts_decode(anchors, raw.permute(0,2,1).reshape(2,7,9,3))
    expected = stock[..., :2] * stride[None, :, None]
    assert torch.equal(actual, expected)
    actual.sum().backward()
    gradient = raw.grad.permute(0,2,1).reshape(2,7,9,3)
    assert torch.equal(gradient[..., 0], stride[:, 0][None, :, None].expand(2,7,9))
    assert not gradient[..., 2].any()


def test_exact_line_zero_and_one_endpoint_wrong():
    logits = torch.zeros(1,1,dtype=torch.float64)
    normal = torch.tensor([[1.,0.]],dtype=torch.float64)
    rho = torch.zeros(1,dtype=torch.float64)
    good = torch.tensor([[[[0.,-1.],[0.,1.]]]],dtype=torch.float64)
    bad = good.clone(); bad[0,0,1,0] = 2.
    assert pair_mixture_incidence(logits,good,normal,rho,1.).item() == 0.
    loss = pair_mixture_incidence(logits,bad,normal,rho,1.)
    assert torch.allclose(loss, loss.new_tensor(.5 * (math.sqrt(5)-1)))


def test_swapped_endpoints_and_antipodal_seam_invariant():
    logits = torch.tensor([[.2,-1.,2.]],dtype=torch.float64)
    pairs = torch.tensor([[[[.2,.8],[1.2,-.3]]]],dtype=torch.float64)
    theta = torch.tensor([0.,math.pi-1e-7,.7],dtype=torch.float64)
    normal = torch.stack((theta.cos(),theta.sin()),-1)
    rho = torch.tensor([.3,-.3,1.],dtype=torch.float64)
    a = pair_mixture_incidence(logits,pairs,normal,rho,.7)
    b = pair_mixture_incidence(logits,pairs.flip(-2),normal,rho,.7)
    c = pair_mixture_incidence(logits,pairs,-normal,-rho,.7)
    assert torch.allclose(a,b,atol=1e-14,rtol=0)
    assert torch.equal(a,c)


def test_multimodal_pair_cannot_choose_different_lines_or_mean_line():
    logits = torch.zeros(1,2,dtype=torch.float64)
    normal = torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    rho = torch.tensor([-1.,1.],dtype=torch.float64)
    split_pair = torch.tensor([[[[-1.,-1.],[1.,1.]]]],dtype=torch.float64)
    joint = pair_mixture_incidence(logits,split_pair,normal,rho,.1).item()
    independent = .5 * sum(pair_mixture_incidence(logits,
        split_pair[..., endpoint:endpoint+1,:].repeat(1,1,2,1),normal,rho,.1).item() for endpoint in (0,1))
    assert joint > 9 and independent < .7
    actual_line = torch.tensor([[[[-1.,-1.],[-1.,1.]]]],dtype=torch.float64)
    phantom_mean = actual_line.clone(); phantom_mean[...,0] = 0.
    assert pair_mixture_incidence(logits,actual_line,normal,rho,.1).item() < .7
    assert pair_mixture_incidence(logits,phantom_mean,normal,rho,.1).item() > 9


def test_translation_invariance_and_gradients_to_both_predictions():
    logits = torch.tensor([[1.,-.7]],dtype=torch.float64,requires_grad=True)
    pair = torch.tensor([[[[.4,-1.],[.6,1.]]]],dtype=torch.float64,requires_grad=True)
    normal = torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    rho = torch.tensor([0.,2.],dtype=torch.float64)
    a = pair_mixture_incidence(logits,pair,normal,rho,.5)
    shift = torch.tensor([1.2,-.8],dtype=torch.float64)
    b = pair_mixture_incidence(logits,pair+shift,normal,rho+normal@shift,.5)
    assert torch.allclose(a,b,atol=1e-14,rtol=0)
    a.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert torch.isfinite(pair.grad).all() and pair.grad[...,0].sum() > 0
    assert not pair.grad[...,1].any()  # One vertical line cannot locate tangent y.


def test_double_autograd_gradcheck():
    logits = torch.tensor([[.1,-.4]],dtype=torch.float64,requires_grad=True)
    pair = torch.tensor([[[[.2,.1],[.5,.8]]]],dtype=torch.float64,requires_grad=True)
    normal = torch.tensor([[1.,0.],[.6,.8]],dtype=torch.float64)
    rho = torch.tensor([0.,1.],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x,y:pair_mixture_incidence(x,y,normal,rho,.8),
                                    (logits,pair),eps=1e-6,atol=1e-5,rtol=1e-4)


def test_scale_invariance_of_dimensionless_cost():
    x = torch.tensor([[[[.2,-1.],[.4,1.]]]],dtype=torch.float64)
    logits = torch.tensor([[1.,-.5]],dtype=torch.float64)
    n = torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
    rho = torch.tensor([.2,.3],dtype=torch.float64)
    a = pair_mixture_incidence(logits,x,n,rho,.6)
    b = pair_mixture_incidence(logits,x*7,n,rho*7,.6*7)
    assert torch.allclose(a,b,atol=1e-14,rtol=0)


def test_full_loss_backprop_and_centroid_ignored():
    args = setup_case()
    loss,info = incidence_loss(*args)
    assert loss > 0 and info['incidence_anchor_roles'] == 24
    loss.backward()
    assert args[0].grad.abs().sum() > 0 and args[2].grad[:,:,:8].abs().sum() > 0
    assert not args[2].grad[:,:,8].any()


def test_gt_coordinates_are_not_loss_residual_or_candidate_selection():
    args = list(setup_case())
    before,_ = incidence_loss(*args)
    args[5] = args[5].clone().requires_grad_()
    shifted = args[5].detach().clone();shifted[...,0] += .05;shifted[...,1] -= .03
    shifted.requires_grad_();args[5] = shifted
    after,_ = incidence_loss(*args)
    assert torch.equal(before,after)  # Same support, predictions, and assignment.
    after.backward()
    assert shifted.grad is None


def test_visibility_zero_only_masks_incident_supported_roles():
    args = list(setup_case())
    args[5] = args[5].clone();args[5][0,0,2] = 0.
    loss,info = incidence_loss(*args)
    assert loss > 0 and info['incidence_supported_image_roles'] == 9
    # Corner0 belongs to exactly three cuboid edges, not all twelve.
    loss.backward();assert not args[2].grad[:,:,0].any()


def test_all_unknown_empty_and_outside_only_images_are_zero():
    for kind in ['unknown','empty','outside']:
        args = list(setup_case())
        if kind == 'unknown':args[5][...,2] = 0
        elif kind == 'outside':args[5][...,:2] += 3.
        else:
            args[5] = args[5][:0];args[6] = args[6][:0];args[3][:] = False
        loss,info = incidence_loss(*args)
        assert loss.item() == 0 and info['incidence_images'] == 0
        loss.backward();assert not args[0].grad.any() and not args[2].grad.any()


def test_no_foreground_zero_and_batch_image_normalization():
    one = setup_case(batch_size=1)
    expected,_ = incidence_loss(*one)
    args = list(setup_case(batch_size=2))
    args[0] = one[0].detach().repeat(2,1,1,1).requires_grad_()
    args[3][1] = False
    actual,info = incidence_loss(*args)
    assert torch.allclose(actual,expected/2,atol=1e-14,rtol=0)
    assert info['incidence_images'] == 1 and info['incidence_batch_images'] == 2


def test_collapsed_predicted_endpoints_do_not_disable_loss():
    args = list(setup_case())
    args[2] = torch.zeros_like(args[2],requires_grad=True)
    loss,info = incidence_loss(*args)
    assert loss > 0 and info['incidence_anchor_roles'] == 24
    loss.backward();assert torch.isfinite(args[2].grad).all()


def test_multiobject_and_wrong_stock_assignment_fail_explicitly():
    args = list(setup_case())
    args[5] = args[5].repeat(2,1,1);args[6] = torch.tensor([0,0])
    with pytest.raises(ValueError,match='multi-object'):
        incidence_loss(*args)
    args = list(setup_case());args[4][0,0] = 1
    with pytest.raises(ValueError,match='unique GT'):
        incidence_loss(*args)


@pytest.mark.parametrize('which', ['line','point'])
def test_nonfinite_supported_prediction_fails(which):
    args = list(setup_case())
    index = 0 if which == 'line' else 2
    args[index] = args[index].detach().clone();args[index].reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        incidence_loss(*args)


def test_missing_corner_nonfinite_coordinate_is_ignored_consistently():
    args = list(setup_case())
    args[5][0,0,2] = 0.;args[5][0,0,:2] = float('nan')
    loss,info = incidence_loss(*args)
    assert torch.isfinite(loss) and info['incidence_supported_image_roles'] == 9
