import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "common"))

from heatmap_refinement import (  # noqa: E402
    ProjectedSpanLoss,
    differentiable_weighted_centroid_2d,
    weighted_centroid_decode,
)


def _maps_at(points, size=50, sigma=1.4):
    points = torch.as_tensor(points, dtype=torch.float32)
    ys = torch.arange(size, dtype=torch.float32)
    xs = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    distance2 = (
        (xx[None] - points[:, 0, None, None]).square()
        + (yy[None] - points[:, 1, None, None]).square()
    )
    return torch.exp(-0.5 * distance2 / sigma**2).unsqueeze(0)


def _fixture():
    # Channel order follows the camera-facing cuboid convention. Coordinates
    # are deliberately non-square so PCA axes are deterministic.
    centers = torch.tensor([
        [10.0, 12.0], [40.0, 12.0], [39.0, 24.0], [11.0, 24.0],
        [15.0, 28.0], [35.0, 28.0], [34.0, 38.0], [16.0, 38.0],
    ])
    gt8 = centers + 0.4395
    target = torch.zeros(1, 9, 2)
    target[0, :8] = gt8
    target[0, 8] = gt8.mean(dim=0)
    valid = torch.ones(1, 9)
    return centers, target, valid


def test_differentiable_decoder_matches_deployed_decoder():
    centers, _target, _valid = _fixture()
    maps = _maps_at(centers).requires_grad_(True)
    expected, _confidence, _peak = weighted_centroid_decode(maps)
    actual = differentiable_weighted_centroid_2d(maps)
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)
    actual.sum().backward()
    assert maps.grad is not None
    assert torch.isfinite(maps.grad).all()
    assert float(maps.grad.abs().sum()) > 0.0


def test_projected_span_penalizes_shrink_and_has_finite_gradient():
    centers, target, valid = _fixture()
    center = centers.mean(dim=0, keepdim=True)
    shrunk = center + 0.72 * (centers - center)
    matched_maps = _maps_at(centers)
    shrunk_maps = _maps_at(shrunk).requires_grad_(True)
    loss_fn = ProjectedSpanLoss()

    matched_loss, matched_info = loss_fn(matched_maps, target, valid)
    shrunk_loss, shrunk_info = loss_fn(shrunk_maps, target, valid)

    assert matched_info["valid_frac"] == 1.0
    assert shrunk_info["valid_frac"] == 1.0
    assert shrunk_info["mean_min_span_ratio"] < 0.80
    assert shrunk_loss > matched_loss + 1.0e-3
    shrunk_loss.backward()
    assert shrunk_maps.grad is not None
    assert torch.isfinite(shrunk_maps.grad).all()
    assert float(shrunk_maps.grad.abs().sum()) > 0.0


def test_gradient_step_reduces_shrink_loss():
    centers, target, valid = _fixture()
    center = centers.mean(dim=0, keepdim=True)
    shrunk = center + 0.78 * (centers - center)
    maps = _maps_at(shrunk).requires_grad_(True)
    loss_fn = ProjectedSpanLoss()
    before, _info = loss_fn(maps, target, valid)
    gradient, = torch.autograd.grad(before, maps)
    candidate = (maps.detach() - 0.5 * gradient).requires_grad_(True)
    after, _info = loss_fn(candidate, target, valid)
    assert torch.isfinite(after)
    assert after < before


def test_footprint_edge_hard_mining_focuses_shrunken_example():
    centers, target, valid = _fixture()
    center = centers.mean(dim=0, keepdim=True)
    shrunk = center + 0.70 * (centers - center)
    maps = _maps_at(shrunk)
    plain = ProjectedSpanLoss(
        footprint_edge_weight=1.0, hard_example_gain=0.0)
    mined = ProjectedSpanLoss(
        footprint_edge_weight=1.0,
        hard_edge_threshold=0.85,
        hard_example_gain=4.0,
    )
    plain_loss, _plain_info = plain(maps, target, valid)
    mined_loss, info = mined(maps, target, valid)
    assert info["hard_fraction"] == 1.0
    assert info["mean_min_edge_ratio"] < 0.85
    torch.testing.assert_close(mined_loss, plain_loss * 5.0)


def test_invalid_or_border_target_is_gated_to_connected_zero():
    centers, target, valid = _fixture()
    target[:, 0, 0] = 1.0
    maps = _maps_at(centers).requires_grad_(True)
    loss, info = ProjectedSpanLoss(interior_margin=4.0)(maps, target, valid)
    assert info["valid_frac"] == 0.0
    assert float(loss) == 0.0
    loss.backward()
    assert maps.grad is not None
    assert float(maps.grad.abs().sum()) == 0.0
