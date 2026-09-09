"""CPU contract tests for the pre-registered spatial-fusion probe.

These tests deliberately exercise only small synthetic tensors.  They must not
open the feature cache, start extraction, train a probe, or inspect real GT.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_spatial_fusion.py"
CONTRACT = json.loads((HERE / "SPATIAL_FUSION_CONTRACT.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def spatial_fusion():
    assert RUNNER.is_file(), f"spatial-fusion runner is missing: {RUNNER}"
    spec = importlib.util.spec_from_file_location(
        "dimension_conditioning_spatial_fusion_under_test", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build(module, arm: str, seed: int = 20260829):
    """Construct an arm exactly as the same-seed contract specifies."""

    torch.manual_seed(seed)
    return module.SpatialFusionProbe(arm).cpu().eval()


def _batch(module, batch_size: int = 4):
    generator = torch.Generator(device="cpu").manual_seed(731)
    patches = torch.randn(
        batch_size,
        module.PATCH_TOKENS,
        module.TOKEN_DIM,
        generator=generator,
    )
    mask = torch.ones(batch_size, module.PATCH_TOKENS, dtype=torch.bool)
    mask[0, :5] = False
    mask[1, -7:] = False
    mask[2, 1::3] = False
    levels = torch.tensor([0, 1, 2, 1], dtype=torch.long)[:batch_size]
    kp = torch.randn(batch_size, 26, generator=generator)
    dims = torch.randn(batch_size, 4, generator=generator)
    return patches, mask, levels, kp, dims


def test_frozen_arm_and_patch_contract(spatial_fusion):
    expected_arms = {
        "S0_SPATIAL_NO_DIMS",
        "S1_SPATIAL_CONCAT",
        "S2_SPATIAL_FILM",
        "S3_SPATIAL_CROSS_ATTENTION",
    }
    assert set(spatial_fusion.ARMS) == expected_arms
    assert spatial_fusion.PATCH_SIZE == 7
    assert spatial_fusion.PATCH_TOKENS == 49
    assert spatial_fusion.TOKEN_DIM == 64
    assert CONTRACT["feature_recipe"]["tokens"] == spatial_fusion.PATCH_TOKENS
    assert CONTRACT["feature_recipe"]["channels"] == spatial_fusion.TOKEN_DIM


def test_masked_mean_max_ignores_padding_and_has_locked_shape(spatial_fusion):
    tokens = torch.arange(2 * 49 * 64, dtype=torch.float32).reshape(2, 49, 64)
    mask = torch.zeros(2, 49, dtype=torch.bool)
    mask[0, [0, 3, 17]] = True
    mask[1, [2, 48]] = True

    pooled = spatial_fusion._masked_mean_max(tokens, mask)
    expected_mean = torch.stack(
        (tokens[0, [0, 3, 17]].mean(0), tokens[1, [2, 48]].mean(0))
    )
    expected_max = torch.stack(
        (tokens[0, [0, 3, 17]].max(0).values, tokens[1, [2, 48]].max(0).values)
    )
    expected = torch.cat((expected_mean, expected_max), dim=1)

    assert pooled.shape == (2, 128)
    torch.testing.assert_close(pooled, expected, rtol=0.0, atol=0.0)

    poisoned = tokens.clone()
    poisoned[~mask] = 1.0e20
    torch.testing.assert_close(
        spatial_fusion._masked_mean_max(poisoned, mask),
        pooled,
        rtol=0.0,
        atol=0.0,
    )


def test_masked_mean_max_rejects_shape_contract_violations(spatial_fusion):
    tokens = torch.zeros(2, 49, 64)
    mask = torch.ones(2, 49, dtype=torch.bool)
    errors = (ValueError, RuntimeError, AssertionError)
    with pytest.raises(errors):
        spatial_fusion._masked_mean_max(tokens[:, :-1], mask)
    with pytest.raises(errors):
        spatial_fusion._masked_mean_max(tokens, mask[:, :-1])
    with pytest.raises(errors):
        spatial_fusion._masked_mean_max(tokens, torch.zeros_like(mask))

    # The runner deliberately normalizes a serialized 0/1 mask to bool.  That
    # permissive input path must remain numerically identical to a bool mask.
    torch.testing.assert_close(
        spatial_fusion._masked_mean_max(tokens, mask.float()),
        spatial_fusion._masked_mean_max(tokens, mask),
        rtol=0.0,
        atol=0.0,
    )


def test_all_arms_are_finite_cpu_binary_classifiers(spatial_fusion):
    batch = _batch(spatial_fusion)
    with torch.no_grad():
        for arm in spatial_fusion.ARMS:
            logits = _build(spatial_fusion, arm)(*batch)
            assert logits.shape == (batch[0].shape[0], 2)
            assert logits.device.type == "cpu"
            assert torch.isfinite(logits).all()


def test_no_dims_arm_is_dimension_invariant(spatial_fusion):
    model = _build(spatial_fusion, "S0_SPATIAL_NO_DIMS")
    patches, mask, levels, kp, dims = _batch(spatial_fusion)
    with torch.no_grad():
        first = model(patches, mask, levels, kp, dims)
        second = model(patches, mask, levels, kp, dims * -37.0 + 11.0)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_film_and_attention_adapter_parameter_counts_are_within_five_percent(
    spatial_fusion,
):
    film = _build(spatial_fusion, "S2_SPATIAL_FILM")
    attention = _build(spatial_fusion, "S3_SPATIAL_CROSS_ATTENTION")
    film_count = spatial_fusion._adapter_parameter_count(film)
    attention_count = spatial_fusion._adapter_parameter_count(attention)

    assert isinstance(film_count, int) and film_count > 0
    assert isinstance(attention_count, int) and attention_count > 0
    relative_difference = abs(film_count - attention_count) / max(
        film_count, attention_count
    )
    assert relative_difference < 0.05


def test_same_seed_conditioned_arms_have_identical_step_zero_logits(spatial_fusion):
    seed = 19
    models = {
        arm: _build(spatial_fusion, arm, seed)
        for arm in (
            "S1_SPATIAL_CONCAT",
            "S2_SPATIAL_FILM",
            "S3_SPATIAL_CROSS_ATTENTION",
        )
    }
    batch = _batch(spatial_fusion)
    with torch.no_grad():
        logits = {name: model(*batch) for name, model in models.items()}

    maximum = 0.0
    names = tuple(logits)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            delta = float((logits[first_name] - logits[second_name]).abs().max())
            maximum = max(maximum, delta)
    assert maximum <= 1.0e-7

    audit = spatial_fusion._step_zero_audit(models, batch, tolerance=1.0e-7)
    assert isinstance(audit, dict) and audit["passed"] is True
    assert all(audit["common_state_exact"].values())
    assert max(audit["logit_max_abs_diff"].values()) <= 1.0e-7
    assert audit["adapter_parameter_relative_difference"] < 0.05


def test_dimension_shuffle_is_a_deterministic_whole_triplet_derangement(
    spatial_fusion,
):
    index = np.arange(17, dtype=np.float32)
    dimensions = np.stack((index + 0.125, index + 100.25, index + 1000.5), axis=1)
    original = dimensions.copy()

    shuffled, permutation = spatial_fusion._shuffled_dimensions(
        dimensions, seed=42, require_all_triplets_changed=True
    )
    repeated, repeated_permutation = spatial_fusion._shuffled_dimensions(
        dimensions, seed=42, require_all_triplets_changed=True
    )

    assert np.array_equal(dimensions, original), "shuffle mutated its input"
    assert shuffled.shape == dimensions.shape
    assert shuffled.dtype == dimensions.dtype
    assert permutation == repeated_permutation
    assert np.array_equal(shuffled, repeated)
    assert sorted(permutation) == list(range(len(dimensions)))
    assert np.array_equal(shuffled, dimensions[np.asarray(permutation)])
    assert np.all(np.any(shuffled != dimensions, axis=1))

    # This additionally prevents independent per-axis column shuffles: every
    # output row must be one complete triplet from the input table.
    input_triplets = {tuple(row) for row in dimensions.tolist()}
    assert all(tuple(row) in input_triplets for row in shuffled.tolist())


def test_required_all_changed_shuffle_rejects_impossible_duplicate_triplets(
    spatial_fusion,
):
    dimensions = np.repeat(
        np.asarray([[1.1, 0.11, 1.3]], dtype=np.float32), repeats=8, axis=0
    )
    with pytest.raises((ValueError, RuntimeError)):
        spatial_fusion._shuffled_dimensions(
            dimensions, seed=42, require_all_triplets_changed=True
        )
