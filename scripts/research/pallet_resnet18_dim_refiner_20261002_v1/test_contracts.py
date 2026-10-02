"""CPU-only foundation tests; no image forward, GPU, or production training."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import random

import numpy as np
import torch

from . import common as C
from .data import order_digest, paired_order
from .dimensions import context_for_arm, dimension_features, normalized_context
from .refiner import (DimensionConditionedPointRefiner, forward_arm, loss_for_arm,
                      model_for_arm)
from .resume import (capture_rng, checkpoint_payload, restore_rng,
                     restore_training_state, validate_checkpoint, atomic_torch_save)
from .protocol import build_protocol, validate_full_completion


torch.set_num_threads(1)
TINY = dict(c3=2, c4=3, stride3=8, stride4=16, hidden=4, encoded=4,
            stencil_fraction=0.13)
HEX = "a" * 64


def fixture(batch=2):
    torch.manual_seed(91)
    points = torch.tensor([[[20.0 + k, 22.0 + k] for k in range(9)] for _ in range(batch)])
    valid = torch.ones(batch, 9, dtype=torch.bool)
    valid[0, 3] = False
    points[0, 3] = float("nan")
    return dict(
        p3=torch.randn(batch, 2, 8, 8),
        p4=torch.randn(batch, 3, 4, 4),
        points=points,
        boxes=torch.tensor([[5.0, 5.0, 58.0, 58.0]]).repeat(batch, 1),
        point_valid=valid,
        input_shape=torch.tensor([[64, 64]]).repeat(batch, 1),
        gt_points=torch.nan_to_num(points, nan=0.0) + 0.75,
        gt_valid=valid.clone(),
        dimension_context=torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5]]).repeat(batch, 1),
    )


def _reference_refiner_module():
    path = C.ROOT / "scripts/research/pallet_resnet18_refiner_20261001_v1/refiner.py"
    spec = importlib.util.spec_from_file_location("_immutable_resnet_refiner_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dimension_order_validation_and_constant_mask():
    dimensions = np.array([[1.1, 1.3, 0.11], [0.8, 0.59, 0.14]])
    features = dimension_features(dimensions)
    assert np.isclose(features[0, 0], np.log(1.1))
    assert np.isclose(features[0, 1], np.log(1.3))
    assert not np.array_equal(features, dimension_features(dimensions[:, [1, 0, 2]]))
    norm = dict(mean=[0] * 5, scale=[1] * 5)
    context = normalized_context(dimensions, norm)
    assert np.array_equal(context_for_arm(context, "P5"), context)
    assert np.array_equal(context_for_arm(context, "P5_CONSTANT"), np.zeros_like(context))
    assert context_for_arm(context, "P0") is None
    for bad in ([1, 2], [1, 2, 0], [1, float("nan"), 2]):
        try:
            dimension_features(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid dimensions: {bad}")


def test_production_parameter_counts():
    counts = {arm: sum(parameter.numel() for parameter in model_for_arm(arm).parameters())
              for arm in C.HEAD_ARMS}
    assert counts == {"D0": 22522, "P0": 22034, "P5": 23331, "P5_CONSTANT": 23331}


def test_visual_p_and_d_match_immutable_reference():
    reference = _reference_refiner_module()
    batch = fixture(batch=1)
    arguments = [batch[key] for key in ("p3", "p4", "points", "boxes", "point_valid", "input_shape")]
    pairs = (("P0", reference.GenericPointRefiner), ("D0", reference.DirectResidualControl))
    for arm, reference_class in pairs:
        torch.manual_seed(29)
        actual_model = model_for_arm(arm, TINY)
        torch.manual_seed(29)
        expected_model = reference_class(**TINY)
        for key, value in actual_model.state_dict().items():
            assert torch.equal(value, expected_model.state_dict()[key]), (arm, key)
        actual = forward_arm(actual_model, batch, arm, lam=0)
        expected = expected_model(*arguments, lam=0)
        keys = ("points", "point_support", "logits") if arm == "P0" else (
            "points", "point_support", "delta_normalized")
        for key in keys:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)


def test_p5_step0_exact_visual_logits_and_shared_initialization():
    batch = fixture()
    torch.manual_seed(7)
    visual = model_for_arm("P0", TINY)
    torch.manual_seed(7)
    dimension = model_for_arm("P5", TINY)
    for name, value in visual.state_dict().items():
        assert torch.equal(value, dimension.state_dict()[name]), name
    baseline = forward_arm(visual, batch, "P0", lam=0)
    for context in (torch.zeros(2, 5), torch.ones(2, 5), torch.arange(10).reshape(2, 5).float()):
        copy_batch = dict(batch, dimension_context=context)
        output = forward_arm(dimension, copy_batch, "P5", lam=0)
        assert torch.equal(baseline["logits"], output["logits"])
        assert output["metadata_residual"].abs().max() == 0


def test_p5_context_path_and_constant_arm():
    batch = fixture()
    torch.manual_seed(11)
    full = model_for_arm("P5", TINY)
    torch.manual_seed(11)
    constant = model_for_arm("P5_CONSTANT", TINY)
    assert isinstance(full, DimensionConditionedPointRefiner)
    full.metadata_scorer[-1].weight.data.normal_()
    constant.load_state_dict(full.state_dict())
    changed = dict(batch, dimension_context=batch["dimension_context"] + 2)
    a = forward_arm(full, batch, "P5", lam=0)
    b = forward_arm(full, changed, "P5", lam=0)
    assert torch.equal(a["base_logits"], b["base_logits"])
    assert not torch.equal(a["logits"], b["logits"])
    c = forward_arm(constant, batch, "P5_CONSTANT", lam=0)
    d = forward_arm(constant, changed, "P5_CONSTANT", lam=0)
    assert torch.equal(c["logits"], d["logits"])


def test_all_heads_finite_gradient_and_preservation():
    batch = fixture()
    for arm in C.HEAD_ARMS:
        torch.manual_seed(31)
        model = model_for_arm(arm, TINY)
        output = forward_arm(model, batch, arm, lam=0)
        torch.testing.assert_close(output["points"], batch["points"], rtol=0, atol=0,
                                   equal_nan=True)
        value = loss_for_arm(output, batch, arm)
        assert torch.isfinite(value)
        value.backward()
        assert model.adapt3[0].weight.grad.norm() > 0
        assert model.adapt4[0].weight.grad.norm() > 0
        if arm in ("P5", "P5_CONSTANT"):
            assert model.metadata_scorer[-1].weight.grad.norm() > 0
            assert model.metadata_encoder[0].weight.grad.norm() == 0
        moved = forward_arm(model, batch, arm, lam=1)
        assert torch.equal(moved["points"][:, 8], batch["points"][:, 8])
        torch.testing.assert_close(moved["points"][0, 3], batch["points"][0, 3],
                                   rtol=0, atol=0, equal_nan=True)


def test_paired_orders_are_shared_reproducible_and_complete():
    rows = np.arange(23)
    first = paired_order(rows, 1, steps=17, batch=4)
    second = paired_order(rows, 1, steps=17, batch=4)
    other = paired_order(rows, 2, steps=17, batch=4)
    assert first.shape == (17, 4)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert order_digest(first) == order_digest(second)


def _tiny_models_and_optimizers():
    models = {}
    optimizers = {}
    for index, arm in enumerate(C.HEAD_ARMS):
        torch.manual_seed(100 + index)
        models[arm] = model_for_arm(arm, TINY)
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=1e-3)
    return models, optimizers


def test_cpu_exact_resume_restores_models_optimizers_and_rng():
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    models, optimizers = _tiny_models_and_optimizers()
    batch = fixture(batch=1)
    for arm in C.HEAD_ARMS:
        optimizers[arm].zero_grad(set_to_none=True)
        value = loss_for_arm(forward_arm(models[arm], batch, arm, lam=0), batch, arm)
        value.backward(); optimizers[arm].step()
    payload = checkpoint_payload(step=1, protocol_sha256=HEX, baseline_sha256=HEX,
                                 order_sha256=HEX, models=models, optimizers=optimizers,
                                 history=[{"step": 1}], elapsed_seconds=1.25,
                                 include_cuda=False)
    saved = copy.deepcopy(payload)
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restored_models, restored_optimizers = _tiny_models_and_optimizers()
    restore_training_state(saved, models=restored_models, optimizers=restored_optimizers,
                           protocol_sha256=HEX, baseline_sha256=HEX,
                           order_sha256=HEX, include_cuda=False)
    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual_rng == expected_rng
    for arm in C.HEAD_ARMS:
        for key, value in models[arm].state_dict().items():
            assert torch.equal(value, restored_models[arm].state_dict()[key]), (arm, key)
    validate_checkpoint(saved, protocol_sha256=HEX, baseline_sha256=HEX, order_sha256=HEX)
    broken = copy.deepcopy(saved); broken["order_sha256"] = "b" * 64
    try:
        validate_checkpoint(broken, protocol_sha256=HEX, baseline_sha256=HEX, order_sha256=HEX)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted resume with a different batch order")


def test_rng_helpers_do_not_require_cuda():
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    state = capture_rng(include_cuda=False)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng(state, include_cuda=False)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected and state["cuda"] is None


def test_protocol_encodes_fixed_budget_and_missing_completion_guard(tmp_path, monkeypatch):
    source_manifest = tmp_path / "SOURCE_MANIFEST.json"
    dimension_sidecar = tmp_path / "DIMENSION_SIDECAR.npz"
    source_manifest.write_text("{}\n")
    dimension_sidecar.write_bytes(b"fixture")
    monkeypatch.setattr(C, "SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(C, "DIMENSION_SIDECAR", dimension_sidecar)
    fake = {"path": "fixture", "sha256": HEX, "bytes": 1}
    protocol = build_protocol(
        dict(checkpoint=fake, protocol=fake, completion=fake),
        [fake],
    )
    assert protocol["arms"] == list(C.HEAD_ARMS)
    assert protocol["seeds"] == [1, 2, 3]
    assert (protocol["steps"], protocol["batch"], protocol["fits"]) == (6000, 16, 12)
    assert protocol["training"]["real_training_images"] == 0
    assert protocol["calibration"]["real_selection"] is False
    assert protocol["selection"]["real_selection"] is False
    assert protocol["inputs"]["P5_CONSTANT"].startswith("Identical P5 architecture")
    absent = tmp_path / "FULL_COMPLETE.json"
    try:
        validate_full_completion(absent)
    except RuntimeError as error:
        assert "has not completed" in str(error)
    else:
        raise AssertionError("Protocol accepted a missing FULL completion receipt")


def test_atomic_checkpoint_roundtrip(tmp_path):
    models, optimizers = _tiny_models_and_optimizers()
    payload = checkpoint_payload(step=0, protocol_sha256=HEX, baseline_sha256=HEX,
                                 order_sha256=HEX, models=models, optimizers=optimizers,
                                 history=[], elapsed_seconds=0, include_cuda=False)
    destination = tmp_path / "resume.pt"
    atomic_torch_save(destination, payload)
    assert destination.is_file() and not Path(str(destination) + ".pending").exists()
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    validate_checkpoint(loaded, protocol_sha256=HEX, baseline_sha256=HEX,
                        order_sha256=HEX)
