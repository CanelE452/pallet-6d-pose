"""Geometry and gradient checks for the task-adapted DHT operator."""
import math

import pytest
import torch

from scripts.research.deep_hough_side_v1.dht import SparseDHT, hough_pad2d


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_axis_lines_and_rho_sign(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    op = SparseDHT(height=9, width=9, theta_bins=180, rho_step=1).to(device)
    features = torch.zeros(1, 2, 9, 9, device=device)
    features[0, 0, :, 6] = 1  # x=6 -> vertical, centered rho=+2
    features[0, 1, 1, :] = 1  # y=1 -> horizontal, centered rho=-3
    votes = op(features)
    for channel, angle, rho in ((0, 0, 2), (1, 90, -3)):
        ri = int((op.rho_values == rho).nonzero().item())
        assert float(votes[0, channel, angle, ri]) == pytest.approx(1)
        assert votes[0, channel, angle].argmax().item() == ri
        # Neighboring parallel lines do not see the impulse line.
        assert float(votes[0, channel, angle, ri - 1]) == pytest.approx(0)
        assert float(votes[0, channel, angle, ri + 1]) == pytest.approx(0)


def test_interpolated_vote_conservation_and_analytic_gradient():
    op = SparseDHT(height=7, width=8, theta_bins=12, rho_step=0.5, normalize=False)
    features = torch.randn(2, 3, 7, 8, requires_grad=True)
    votes = op(features)
    # Every pixel casts unit interpolation mass at each orientation.
    expected = features.sum((-1, -2))[:, :, None].expand(-1, -1, op.theta_bins)
    torch.testing.assert_close(votes.sum(-1), expected, rtol=2e-6, atol=2e-6)
    votes.sum().backward()
    torch.testing.assert_close(features.grad, torch.full_like(features, op.theta_bins))


def test_constant_input_is_one_on_every_supported_line():
    op = SparseDHT(height=13, width=11, theta_bins=18, rho_step=0.5)
    votes = op(torch.ones(1, 1, 13, 11))[0, 0]
    torch.testing.assert_close(votes[op.valid], torch.ones_like(votes[op.valid]))
    assert torch.count_nonzero(votes[~op.valid]).item() == 0


def test_single_pixel_votes_at_expected_rho():
    op = SparseDHT(height=5, width=5, theta_bins=4, rho_step=1, normalize=False)
    features = torch.zeros(1, 1, 5, 5)
    features[0, 0, 2, 3] = 1  # centered (x,y)=(1,0)
    votes = op(features)[0, 0]
    for ti, theta in enumerate(op.theta_radians):
        # Weighted rho average recovers the continuous analytical projection.
        projected = (votes[ti] * op.rho_values).sum()
        assert float(projected) == pytest.approx(math.cos(float(theta)), abs=1e-6)
        assert float(votes[ti].sum()) == pytest.approx(1)


def test_finite_difference_gradient():
    op = SparseDHT(height=3, width=4, theta_bins=4, rho_step=1).double()
    features = torch.randn(1, 1, 3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(op, (features,), eps=1e-6, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_forward_backward_agree():
    cpu = SparseDHT(height=11, width=12, theta_bins=18, rho_step=0.5)
    gpu = SparseDHT(height=11, width=12, theta_bins=18, rho_step=0.5).cuda()
    x = torch.randn(2, 3, 11, 12, requires_grad=True)
    y = x.detach().cuda().requires_grad_(True)
    a, b = cpu(x), gpu(y)
    torch.testing.assert_close(a, b.cpu(), rtol=1e-5, atol=1e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad.cpu(), rtol=2e-5, atol=1e-7)


def test_theta_seam_reverses_rho_and_rho_padding_is_zero():
    values = torch.arange(12).reshape(1, 1, 3, 4).float()
    padded = hough_pad2d(values, 1)
    torch.testing.assert_close(padded[..., 0, 1:-1], values[..., -1, :].flip(-1))
    torch.testing.assert_close(padded[..., -1, 1:-1], values[..., 0, :].flip(-1))
    torch.testing.assert_close(padded[..., 1:-1, 1:-1], values)
    assert torch.count_nonzero(padded[..., 0]).item() == 0
    assert torch.count_nonzero(padded[..., -1]).item() == 0
