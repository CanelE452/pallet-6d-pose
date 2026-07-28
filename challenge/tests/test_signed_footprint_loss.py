import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "common"))

from heatmap_refinement import SignedFootprintLoss  # noqa: E402


def _fixture():
    """A centred, rotated/skew projected cuboid safely inside the 50 grid."""
    center = torch.tensor([24.0, 24.0])
    width = torch.tensor([18.0, 6.0])
    vertical = torch.tensor([-3.0, 10.0])
    depth = torch.tensor([5.0, 7.0])
    p0 = center - 0.5 * (width + vertical + depth)
    gt8 = torch.stack([
        p0,
        p0 + width,
        p0 + width + vertical,
        p0 + vertical,
        p0 + depth,
        p0 + width + depth,
        p0 + width + vertical + depth,
        p0 + vertical + depth,
    ])
    target = torch.zeros(1, 9, 2)
    target[0, :8] = gt8
    target[0, 8] = gt8.mean(dim=0)
    valid = torch.ones(1, 9)
    return gt8, target, valid


def _belief_for_decoded(points, sigma=1.25):
    """Gaussian beliefs whose deployed decoded locations are near ``points``."""
    points = torch.as_tensor(points)
    centers = points - 0.4395
    ys = torch.arange(50, dtype=points.dtype, device=points.device)
    xs = torch.arange(50, dtype=points.dtype, device=points.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    distance2 = (
        (xx[None] - centers[:, 0, None, None]).square()
        + (yy[None] - centers[:, 1, None, None]).square()
    )
    return torch.exp(-0.5 * distance2 / float(sigma) ** 2).unsqueeze(0)


@pytest.mark.parametrize(
    "edge,weights",
    [
        ((0, 1), {"width_weight": 1.0, "depth_weight": 0.0}),
        ((2, 6), {"width_weight": 0.0, "depth_weight": 1.0}),
    ],
)
def test_rotated_skew_edge_gradient_moves_endpoints_along_gt_direction(
    edge, weights,
):
    gt8, target, valid = _fixture()
    i, j = edge
    direction = gt8[j] - gt8[i]
    unit = direction / torch.linalg.vector_norm(direction)
    pred = gt8.clone()
    pred[i] += 2.0 * unit
    pred[j] -= 2.0 * unit
    pred = pred.unsqueeze(0).requires_grad_(True)
    loss_fn = SignedFootprintLoss(
        radial_weight=0.0, overshoot_weight=0.0, **weights)

    loss, info = loss_fn.forward_coordinates(pred, target, valid)
    gradient, = torch.autograd.grad(loss, pred)
    update = -gradient[0]

    assert info["valid_frac"] == 1.0
    assert float(loss) > 0.0
    # Gradient descent moves the start/end away from one another *along the
    # oblique GT direction*, rather than along image x/y or the predicted edge.
    assert torch.dot(update[i], -unit) > 0
    assert torch.dot(update[j], unit) > 0
    normal = torch.stack([-unit[1], unit[0]])
    assert abs(float(torch.dot(update[i], normal))) < 1.0e-7
    assert abs(float(torch.dot(update[j], normal))) < 1.0e-7


def test_correct_norm_orthogonal_and_reversed_edges_are_penalized():
    gt8, target, valid = _fixture()
    i, j = 0, 1
    edge = gt8[j] - gt8[i]
    length = torch.linalg.vector_norm(edge)
    unit = edge / length
    normal = torch.stack([-unit[1], unit[0]])
    midpoint = 0.5 * (gt8[i] + gt8[j])
    loss_fn = SignedFootprintLoss(
        width_weight=1.0,
        depth_weight=0.0,
        radial_weight=0.0,
        overshoot_weight=0.0,
    )

    matched, matched_info = loss_fn.forward_coordinates(
        gt8.unsqueeze(0), target, valid)
    orthogonal = gt8.clone()
    orthogonal[i] = midpoint - 0.5 * length * normal
    orthogonal[j] = midpoint + 0.5 * length * normal
    orthogonal_loss, orthogonal_info = loss_fn.forward_coordinates(
        orthogonal.unsqueeze(0), target, valid)
    reversed_edge = gt8.clone()
    reversed_edge[i], reversed_edge[j] = gt8[j].clone(), gt8[i].clone()
    reversed_loss, reversed_info = loss_fn.forward_coordinates(
        reversed_edge.unsqueeze(0), target, valid)

    assert float(matched) == pytest.approx(0.0, abs=1.0e-8)
    assert matched_info["mean_width_signed_ratio"] == pytest.approx(1.0)
    assert orthogonal_info["mean_min_edge_signed_ratio"] == pytest.approx(
        0.0, abs=1.0e-6)
    assert reversed_info["mean_min_edge_signed_ratio"] == pytest.approx(-1.0)
    assert orthogonal_loss > matched + 1.0e-3
    assert reversed_loss > orthogonal_loss


def test_uniform_contraction_moves_all_corners_outward_without_translation():
    gt8, target, valid = _fixture()
    center = gt8.mean(dim=0, keepdim=True)
    pred = (center + 0.72 * (gt8 - center)).unsqueeze(0)
    pred.requires_grad_(True)
    loss_fn = SignedFootprintLoss(
        width_weight=0.0,
        depth_weight=0.0,
        radial_weight=1.0,
        overshoot_weight=0.0,
    )

    loss, info = loss_fn.forward_coordinates(pred, target, valid)
    gradient, = torch.autograd.grad(loss, pred)
    update = -gradient[0]
    radial = gt8 - center
    radial_unit = radial / torch.linalg.vector_norm(
        radial, dim=-1, keepdim=True)
    outward_progress = (update * radial_unit).sum(dim=-1)

    assert info["undercovered_corner_fraction"] == 1.0
    assert info["mean_radial_signed_ratio"] == pytest.approx(0.72, abs=1.0e-6)
    assert bool((outward_progress > 0).all())
    assert float(update.mean(dim=0).norm()) < 1.0e-8


def test_matched_and_expanded_have_no_undercoverage_and_guard_overshoot():
    gt8, target, valid = _fixture()
    center = gt8.mean(dim=0, keepdim=True)
    loss_fn = SignedFootprintLoss(overshoot_ratio=1.10, overshoot_weight=0.20)

    matched_loss, matched_info = loss_fn.forward_coordinates(
        gt8.unsqueeze(0), target, valid)
    expanded = center + 1.20 * (gt8 - center)
    expanded_loss, expanded_info = loss_fn.forward_coordinates(
        expanded.unsqueeze(0), target, valid)

    assert matched_info["mean_undercoverage"] == pytest.approx(0.0, abs=1.0e-7)
    assert matched_info["undercovered_edge_fraction"] == 0.0
    assert matched_info["undercovered_corner_fraction"] == 0.0
    assert expanded_info["mean_undercoverage"] == pytest.approx(0.0, abs=1.0e-7)
    assert expanded_info["undercovered_edge_fraction"] == 0.0
    assert expanded_info["undercovered_corner_fraction"] == 0.0
    assert float(matched_loss) == pytest.approx(0.0, abs=1.0e-8)
    assert expanded_info["mean_overshoot"] == pytest.approx(0.10, abs=1.0e-6)
    assert float(expanded_loss) > 0.0


def test_end_to_end_fixed50_heatmap_gradient_is_finite_and_nonzero():
    gt8, target, valid = _fixture()
    center = gt8.mean(dim=0, keepdim=True)
    contracted = center + 0.76 * (gt8 - center)
    maps = _belief_for_decoded(contracted).requires_grad_(True)

    loss, info = SignedFootprintLoss()(maps, target, valid)
    loss.backward()

    assert info["valid_frac"] == 1.0
    assert info["mean_undercoverage"] > 0.15
    assert maps.grad is not None
    assert torch.isfinite(maps.grad).all()
    assert float(maps.grad.abs().sum()) > 0.0


def test_invalid_or_border_frame_returns_connected_zero():
    gt8, target, valid = _fixture()
    target[:, 0, 0] = 2.0
    maps = _belief_for_decoded(gt8).requires_grad_(True)

    loss, info = SignedFootprintLoss(interior_margin=4.0)(
        maps, target, valid)
    loss.backward()

    assert info["valid_frac"] == 0.0
    assert float(loss) == 0.0
    assert maps.grad is not None
    assert float(maps.grad.abs().sum()) == 0.0


def test_non_deployed_heatmap_grid_is_rejected():
    _gt8, target, valid = _fixture()
    maps = torch.zeros(1, 8, 40, 40)
    with pytest.raises(ValueError, match="50x50"):
        SignedFootprintLoss()(maps, target, valid)
