import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "train"))

from diffpnp3d_loss import (  # noqa: E402
    DiffPnP3DLoss,
    _pnp_fit_coverage,
    _project_batch,
)


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
    projected, _ = _project_batch(
        rvec, translation, points, camera)
    return projected, points, camera, rotation, translation, diagonal, valid


def _inconsistent_observations(projected, amount=50.0):
    """Make two corners disagree with every single rigid-pose projection."""
    observed = projected.clone()
    observed[:, 0, 0] -= amount
    observed[:, 1, 0] += amount
    return observed


def _fit_module(dtype=torch.float64, **overrides):
    options = {
        "n_gn": 4,
        "geometry_weight": 0.0,
        "undercoverage_weight": 0.0,
        "fit_coverage_weight": 1.0,
        "fit_span_margin": 1.0,
        "fit_hard_span_threshold": 0.90,
    }
    options.update(overrides)
    return DiffPnP3DLoss(**options).to(dtype=dtype)


def _loss(module, predicted, fixture, mask=None):
    (_projected, points, camera, rotation, translation,
     diagonal, valid) = fixture
    return module(
        predicted, points, camera, rotation, translation, diagonal,
        valid if mask is None else mask)


def test_fit_coverage_is_zero_for_consistent_projection_and_translation_invariant():
    fixture = _fixture()
    projected = fixture[0].detach().requires_grad_(True)
    loss, info = _loss(_fit_module(), projected, fixture)

    assert float(loss) < 1.0e-12
    assert info["mean_fit_coverage_L"] < 1.0e-12
    assert abs(info["mean_fit_min_span_ratio"] - 1.0) < 1.0e-12
    assert info["fit_hard_fraction"] == 0.0

    # Coverage centres the projection and target independently.  Absolute
    # image position therefore cannot change the footprint-size result.
    base, base_ratio, base_ok = _pnp_fit_coverage(
        projected.detach(), projected.detach(), 1.0, 0.05)
    shifted, shifted_ratio, shifted_ok = _pnp_fit_coverage(
        projected.detach() + torch.tensor([[[37.0, -19.0]]]),
        projected.detach() + torch.tensor([[[-81.0, 43.0]]]),
        1.0, 0.05)
    torch.testing.assert_close(shifted, base, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        shifted_ratio, base_ratio, atol=1.0e-12, rtol=0.0)
    assert torch.equal(shifted_ok, base_ok)


def test_observed_footprint_is_a_detached_one_sided_target():
    observed = _fixture()[0].detach().requires_grad_(True)
    center = observed.detach().mean(dim=1, keepdim=True)
    undercovered = (
        center + 0.80 * (observed.detach() - center)).requires_grad_(True)
    expanded = center + 1.10 * (observed.detach() - center)

    loss, ratio, frame_ok = _pnp_fit_coverage(
        undercovered, observed, 1.0, 0.05)
    expanded_loss, expanded_ratio, _ = _pnp_fit_coverage(
        expanded, observed, 1.0, 0.05)
    projected_gradient, observed_gradient = torch.autograd.grad(
        loss.sum(), (undercovered, observed), allow_unused=True)

    assert frame_ok.item()
    assert ratio.item() < 0.81
    assert float(loss) > 0.0
    assert projected_gradient is not None
    assert torch.isfinite(projected_gradient).all()
    assert float(projected_gradient.norm()) > 0.0
    assert observed_gradient is None
    assert expanded_ratio.item() > 1.0
    assert float(expanded_loss) == 0.0


def test_pose_inconsistent_observations_have_positive_finite_gradient():
    fixture = _fixture()
    observed = _inconsistent_observations(
        fixture[0]).detach().requires_grad_(True)
    loss, info = _loss(_fit_module(), observed, fixture)

    assert torch.isfinite(loss)
    assert float(loss) > 0.001
    assert info["mean_fit_min_span_ratio"] < 0.85
    assert info["fit_hard_fraction"] == 1.0
    gradient, = torch.autograd.grad(loss, observed)
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0

    # The feature is opt-in: its default zero weight cannot change the legacy
    # loss even when the PnP fit clearly undercovers its observations.
    disabled, disabled_info = _loss(
        _fit_module(fit_coverage_weight=0.0), observed, fixture)
    assert float(disabled) == 0.0
    assert disabled_info["mean_fit_min_span_ratio"] < 0.85


def test_gradient_step_improves_fit_coverage_without_nans():
    fixture = _fixture()
    module = _fit_module()
    observed = _inconsistent_observations(
        fixture[0]).detach().requires_grad_(True)

    before, before_info = _loss(module, observed, fixture)
    gradient, = torch.autograd.grad(before, observed)
    # Use a bounded 10-pixel vector step along the descent direction.
    step = 10.0 * gradient / gradient.norm().clamp_min(1.0e-12)
    candidate = (observed.detach() - step).requires_grad_(True)
    after, after_info = _loss(module, candidate, fixture)

    assert torch.isfinite(candidate).all()
    assert torch.isfinite(after)
    assert after < before
    assert (after_info["mean_fit_min_span_ratio"]
            > before_info["mean_fit_min_span_ratio"])
    after_gradient, = torch.autograd.grad(after, candidate)
    assert torch.isfinite(after_gradient).all()


def test_fit_coverage_respects_mask_and_conditioning_guards():
    fixture = _fixture()
    observed = _inconsistent_observations(fixture[0])

    masked_loss, masked_info = _loss(
        _fit_module(), observed, fixture, mask=torch.tensor([False]))
    conditioned_loss, conditioned_info = _loss(
        _fit_module(cond_max=1.0), observed, fixture)

    assert float(masked_loss) == 0.0
    assert masked_info["n_valid"] == 0
    assert masked_info["gated_out"] == 1
    assert float(conditioned_loss) == 0.0
    assert conditioned_info["n_valid"] == 0
    assert conditioned_info["skip_cond"] == 1
