"""Geometry, initialization, serialization and genuine feedback-gradient checks."""
import copy
import torch

from scripts.research.pallet_dht_joint_v1.hough_block import HoughFeatureFusion, normalized_backprojection, normalized_hough, smooth_rho
from scripts.research.deep_hough_side_v1.dht import SparseDHT, hough_pad2d


def test_constant_backprojection_and_vote():
    geometry = SparseDHT(8, 12, 90, .5, True, 28)
    ones = torch.ones(2, 3, 8, 12)
    votes = geometry(ones)
    assert torch.allclose(votes[..., geometry.valid], torch.ones_like(votes[..., geometry.valid]), atol=1e-6)
    back = normalized_backprojection(torch.ones(2, 3, 90, 113), geometry)
    assert torch.allclose(back, ones, atol=1e-6)
    smoothed = normalized_hough(ones, geometry)
    valid = smooth_rho(geometry.mass) > 1e-7
    assert torch.allclose(smoothed[..., valid], torch.ones_like(smoothed[..., valid]), atol=1e-6)


def test_subcell_semantic_peak_survives_and_effective_operator_is_adjoint():
    g = SparseDHT(8, 12, 90, .5, True, 28)
    hough = torch.zeros(1, 1, 90, 113)
    hough[..., 0, 56] = 1  # theta0,rho0 lies BETWEEN the two centre columns.
    assert not g.valid[0, 56]
    back = normalized_backprojection(hough, g)
    assert back[..., 5].sum() > 0 and back[..., 6].sum() > 0
    assert torch.allclose(back[..., 5], back[..., 6], atol=1e-8, rtol=1e-6)
    torch.manual_seed(8)
    x, y = torch.rand(1, 1, 8, 12), torch.rand(1, 1, 90, 113)
    # Compare unnormalized adjoints; each normalized operator has its own mass.
    forward_sums = smooth_rho(g(x) * g.mass)
    backward_sums = normalized_backprojection(y, g) * g._backprojection_mass.reshape(1,1,8,12)
    assert torch.allclose((forward_sums*y).sum(), (x*backward_sums).sum(), atol=2e-3, rtol=1e-6)


def test_zero_parity_then_point_gradient_and_serialization():
    torch.manual_seed(3)
    block = HoughFeatureFusion()
    features = [torch.randn(2, c, h, w, requires_grad=True)
                for c, h, w in [(64, 16, 24), (128, 8, 12), (256, 4, 6)]]
    out = block(features)
    assert all(torch.equal(a, b) for a, b in zip(out, features))
    # Zero output starts exact; test trainable feedback after the output opens.
    with torch.no_grad():
        for layer in block.outputs:
            layer.weight.normal_(0, .01)
    out = block(features)
    loss = sum((y * torch.randn_like(y)).mean() for y in out)
    loss.backward()
    for name in ["reduce.0.weight", "hough_layers.0.conv.weight", "line_head.weight", "spatial_mix.0.weight"]:
        grad = dict(block.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert features[1].grad.abs().sum() > 0
    cloned = copy.deepcopy(block)
    assert not cloned._geometry_cache and cloned.line_logits is None
    assert not any("vote_matrix" in k for k in block.state_dict())


def test_seam_flip_and_rectangular_cell_coordinate():
    x = torch.arange(12).reshape(1, 1, 3, 4)
    padded = hough_pad2d(x, (1, 0))
    assert torch.equal(padded[..., 0, :], x[..., -1, :].flip(-1))
    g = SparseDHT(8, 12, 90, .5, True, 28)
    phantom = torch.zeros(1, 1, 8, 12)
    phantom[..., 2] = 1
    votes = g(phantom)[0, 0]
    # x=2+.5-12/2=-3.5 at normal theta0; rho index=(-3.5+28)/.5=49.
    assert votes[0].argmax().item() == 49
    assert votes[0, 49].item() == 1
