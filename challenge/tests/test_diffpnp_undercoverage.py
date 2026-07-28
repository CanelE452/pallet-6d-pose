import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "train"))

from diffpnp3d_loss import DiffPnP3DLoss, _project_batch  # noqa: E402


def _fixture(dtype=torch.float64):
    points = torch.tensor([[[-0.5, -0.05, -0.65],
                            [0.5, -0.05, -0.65],
                            [0.5, 0.05, -0.65],
                            [-0.5, 0.05, -0.65],
                            [-0.5, -0.05, 0.65],
                            [0.5, -0.05, 0.65],
                            [0.5, 0.05, 0.65],
                            [-0.5, 0.05, 0.65]]], dtype=dtype)
    camera = torch.tensor([[[600.0, 0.0, 320.0],
                            [0.0, 600.0, 240.0],
                            [0.0, 0.0, 1.0]]], dtype=dtype)
    rotation = torch.eye(3, dtype=dtype).unsqueeze(0)
    translation = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype)
    diagonal = torch.tensor([1.65], dtype=dtype)
    valid = torch.tensor([True])
    rvec = torch.zeros(1, 3, dtype=dtype)
    projected, _ = _project_batch(rvec, translation, points, camera)
    return projected, points, camera, rotation, translation, diagonal, valid


def _loss(predicted, fixture):
    _gt, points, camera, rotation, translation, diagonal, valid = fixture
    module = DiffPnP3DLoss(
        n_gn=4,
        geometry_weight=0.0,
        undercoverage_weight=1.0,
        span_under_weight=0.0,
        depth_under_weight=1.0,
        depth_margin=1.15,
        hard_span_threshold=0.90,
        hard_depth_threshold=1.20,
        hard_example_gain=0.0,
    ).to(dtype=predicted.dtype)
    return module(
        predicted, points, camera, rotation, translation, diagonal, valid)


def test_pnp_undercoverage_is_one_sided_and_finite():
    fixture = _fixture()
    gt = fixture[0]
    center = gt.mean(dim=1, keepdim=True)
    shrunk = (center + 0.75 * (gt - center)).detach().requires_grad_(True)
    expanded = center + 1.10 * (gt - center)

    shrink_loss, info = _loss(shrunk, fixture)
    expanded_loss, _ = _loss(expanded, fixture)

    assert shrink_loss > 0.001
    assert float(expanded_loss) == 0.0
    assert info["mean_min_span_ratio"] < 0.80
    assert info["mean_tz_ratio"] > 1.20
    assert info["hard_fraction"] == 1.0
    shrink_loss.backward()
    assert shrunk.grad is not None
    assert torch.isfinite(shrunk.grad).all()
    assert float(shrunk.grad.norm()) > 0.0


def test_gradient_step_reduces_pnp_undercoverage_loss():
    fixture = _fixture()
    gt = fixture[0]
    center = gt.mean(dim=1, keepdim=True)
    predicted = (center + 0.80 * (gt - center)).detach().requires_grad_(True)

    before, _ = _loss(predicted, fixture)
    gradient, = torch.autograd.grad(before, predicted)
    candidate = (predicted.detach() - 5.0 * gradient).requires_grad_(True)
    after, _ = _loss(candidate, fixture)

    assert torch.isfinite(after)
    assert after < before
