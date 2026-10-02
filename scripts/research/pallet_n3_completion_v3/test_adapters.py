from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch
from torch import nn

from scripts.research.pallet_dope_refiner_20261001_v1 import dope_adapter as DOPE
from scripts.research.pallet_n3_completion_v3.resnet_constant_adapter import (
    CHECKPOINT_SHA256,
    FEATURE_CHANNELS,
    FEATURE_STRIDES,
    FILM_KEYS,
    FrozenConstantResnetAdapter,
    conditioned_model_module,
    fold_constant_state_dict,
    folded_rgb_network,
    input_module,
    load_constant_contract,
    network_delta_to_original,
)


class FixtureRgbNetwork(nn.Module):
    """Small image-only decoder fixture with production output shapes."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0
        logits = torch.full((9, 96, 128), -12., dtype=torch.float32)
        locations = (
            (20, 25), (100, 25), (100, 70), (20, 70),
            (30, 18), (90, 18), (90, 78), (30, 78), (60, 48),
        )
        for channel, (x, y) in enumerate(locations):
            logits[channel, y, x] = 12.
        self.register_buffer("fixture_logits", logits)

    def _features(self, images):
        batch = len(images)
        return (
            torch.zeros((batch, 128, 48, 64), dtype=images.dtype, device=images.device),
            torch.zeros((batch, 256, 24, 32), dtype=images.dtype, device=images.device),
        )

    def forward(self, images, return_features=False):
        self.calls += 1
        logits = self.fixture_logits[None].expand(len(images), -1, -1, -1).clone()
        if return_features:
            return {"heatmaps": logits, "features": self._features(images)}
        return logits

    def features_only(self, images):
        if self.training:
            raise RuntimeError("Fixture must be frozen")
        return self._features(images)


class ConstantCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit, cls.payload = load_constant_contract(load_checkpoint=True)

    @classmethod
    def tearDownClass(cls):
        del cls.payload

    def test_constant_checkpoint_identity_and_zero_mask(self):
        self.assertEqual(self.audit["checkpoint"]["sha256"], CHECKPOINT_SHA256)
        self.assertEqual(self.audit["constant_context_mask"], [0, 0, 0, 0, 0])
        self.assertFalse(self.audit["runtime_dimension_context"])
        self.assertEqual(
            {key: self.payload[key] for key in ("arm", "epoch", "step")},
            {"arm": "CONSTANT", "epoch": 10, "step": 34990})
        self.assertTrue(self.payload["model_state_dict"])

    def test_fold_removes_film_and_strict_loads_rgb_state(self):
        original = self.payload["model_state_dict"]
        folded = fold_constant_state_dict(original)
        self.assertTrue(FILM_KEYS.issubset(original))
        self.assertFalse(any(key.startswith("dimension_film.") for key in folded))
        network = folded_rgb_network(original)
        self.assertEqual(set(network.state_dict()), set(folded))
        parameters = inspect.signature(network.forward).parameters
        self.assertNotIn("dimensions", parameters)
        self.assertNotIn("dimension_context", parameters)

    def test_actual_constant_zero_logits_match_folded_rgb_logits(self):
        state = self.payload["model_state_dict"]
        conditioned = conditioned_model_module().SimpleBaselineResNet18(
            num_keypoints=9, pretrained_state=None).eval()
        conditioned.load_state_dict(state, strict=True)
        rgb = folded_rgb_network(state).eval()
        generator = torch.Generator().manual_seed(7301)
        images = torch.randn((1, 3, 32, 32), generator=generator)
        zeros = torch.zeros((1, 5), dtype=images.dtype)
        with torch.no_grad():
            expected = conditioned(images, zeros, return_features=True)
            actual = rgb(images, return_features=True)
        torch.testing.assert_close(actual["heatmaps"], expected["heatmaps"],
                                   rtol=2e-5, atol=2e-6)
        for got, want in zip(actual["features"], expected["features"]):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_rgb_public_interface_has_no_dimension_argument(self):
        for method in (FrozenConstantResnetAdapter.infer,
                       FrozenConstantResnetAdapter.infer_batch):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("dimensions", parameters)
            self.assertNotIn("dimension_context", parameters)


class AdapterFixtureTests(unittest.TestCase):
    def setUp(self):
        self.network = FixtureRgbNetwork()
        self.adapter = FrozenConstantResnetAdapter._cpu_fixture(self.network)

    def test_image_only_forward_and_normal_detached_taps(self):
        image = np.zeros((301, 777, 3), np.uint8)
        output = self.adapter.infer(image, return_features=True)
        self.assertEqual(self.network.calls, 1)
        self.assertTrue(output["base_rgb_only"])
        self.assertTrue(output["dimension_film_folded_at_zero"])
        self.assertFalse(output["runtime_dimension_context"])
        self.assertEqual(tuple(value.shape for value in output["features"]),
                         (torch.Size([128, 48, 64]), torch.Size([256, 24, 32])))
        self.assertEqual(FEATURE_CHANNELS, (128, 256))
        self.assertEqual(FEATURE_STRIDES, (8, 16))
        for feature in output["features"]:
            self.assertFalse(feature.requires_grad)
            self.assertFalse(torch.is_inference(feature))

    def test_bound_preprocess_affine_roundtrip_is_exact(self):
        image = np.zeros((301, 777, 3), np.uint8)
        prepared = self.adapter.prepare(image)
        affine = prepared["affine_input_to_net"]
        # Independent rounded x/y resize scales are intentionally anisotropic.
        self.assertNotEqual(float(affine[0, 0]), float(affine[1, 1]))
        points = np.array([[0., 0.], [776., 300.], [317.25, 121.75], [np.nan, np.nan]])
        network = input_module().transform_points(points, affine)
        replay = input_module().transform_points(network, prepared["affine_net_to_input"])
        np.testing.assert_allclose(replay, points, rtol=0, atol=1e-12, equal_nan=True)

    def test_anisotropic_original_space_circular_cap(self):
        affine = np.array([[.7, 0., 13.], [0., .4, 27.], [0., 0., 1.]])
        delta_net = np.array([[[7., 8.], [0., 0.], [.7, .4]]])
        original_hw = np.array([301, 777])
        actual = network_delta_to_original(
            delta_net, affine, original_hw, cap_fraction=.01)
        uncapped = delta_net / np.array([.7, .4])
        cap = .01 * np.hypot(*original_hw)
        norm = np.linalg.norm(uncapped, axis=-1)
        expected = uncapped * np.minimum(1., cap / np.maximum(norm, 1e-12))[..., None]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertLessEqual(float(np.linalg.norm(actual, axis=-1).max()), cap + 1e-12)
        # Translation belongs to points, never displacement inversion.
        changed_translation = affine.copy()
        changed_translation[:2, 2] = [-999., 543.]
        np.testing.assert_allclose(
            network_delta_to_original(delta_net, changed_translation, original_hw),
            actual, rtol=0, atol=0)


class DopeBindingTests(unittest.TestCase):
    def test_strict_checkpoint_hash_and_feature_taps(self):
        self.assertEqual(DOPE.file_sha(DOPE.CHECKPOINT), DOPE.CHECKPOINT_SHA)
        self.assertEqual(DOPE.FEATURE_CHANNELS, (256, 128))
        self.assertEqual(DOPE.FEATURE_STRIDES, (4, 8))
        self.assertEqual(DOPE.FEATURE_TAPS, (17, 26))
        recipe = DOPE.recipe()
        self.assertEqual(recipe["checkpoint_sha256"], DOPE.CHECKPOINT_SHA)
        self.assertEqual(recipe["feature_channels"], [256, 128])
        self.assertEqual(recipe["feature_strides"], [4, 8])
        self.assertTrue(recipe["frozen"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
