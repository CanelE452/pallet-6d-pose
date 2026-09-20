"""Synthetic, CPU-only regression tests; no dataset or model inputs."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from scripts.research.pallet_posefix_corner_gate_v1 import gates as G


def stub_geometry(residuals=None):
    return {
        "chosen_hypothesis": "square_footprint",
        "projected_diagonal_px": 100.0,
        "effective_valid": [True] * 9,
        "per_corner_remove": [0.01] * 8 if residuals is None else residuals,
    }


class CornerGateTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        self.dimensions = {"x": 1.1, "y": 0.15, "z": 1.1}
        self.valid = np.ones(9, dtype=bool)
        self.conf = np.ones(9, dtype=np.float64)

    def test_exact_fallback_and_centroid(self):
        base = np.arange(18, dtype=np.float32).reshape(9, 2)
        base[0, 0] = -0.0
        candidate = base.astype(np.float64) + 31.75
        rejected = G.apply_pairs(base, candidate, [False] * 4)
        self.assertEqual(rejected.dtype, base.dtype)
        self.assertEqual(rejected.tobytes(), base.tobytes())
        accepted = G.apply_pairs(base, candidate, [True] * 4)
        self.assertEqual(accepted[8].tobytes(), base[8].tobytes())
        np.testing.assert_array_equal(accepted[:8], candidate[:8].astype(np.float32))

    def test_pair_atomicity_and_median_failure(self):
        residuals = [0.105, 0.011, 0.011, 0.056, 0.011, 0.011, 0.011, 0.011]
        self.assertLess(np.median(residuals), G.GEOMETRY_THRESHOLD)
        decision = G.decide_pairs("geometry_only", stub_geometry(residuals), self.valid, 0.99, self.conf)
        self.assertEqual(decision["pair_accept"], [False, True, True, True])
        self.assertFalse(decision["corner_accept"][0])
        self.assertFalse(decision["corner_accept"][3])
        residuals[3] = 0.001
        decision = G.decide_pairs("geometry_only", stub_geometry(residuals), self.valid, 0.99, self.conf)
        self.assertTrue(decision["corner_evidence_pass"][3])
        self.assertFalse(decision["corner_accept"][3])

    def test_full_requires_flip_and_mode(self):
        flip, mode = np.zeros(8), np.zeros(8)
        flip[0] = np.inf
        mode[5] = 0.051
        decision = G.decide_pairs("full", stub_geometry(), self.valid, 0.85, self.conf,
                                  flip_residuals=flip, mode_separation=mode)
        self.assertEqual(decision["pair_accept"], [False, True, True, False])
        missing = G.decide_pairs("full", stub_geometry(), self.valid, 0.99, self.conf)
        self.assertFalse(any(missing["pair_accept"]))
        exact = G.decide_pairs("full", stub_geometry([.05] * 8), self.valid, .85, self.conf,
                               flip_residuals=[.05] * 8, mode_separation=[.05] * 8)
        self.assertTrue(all(exact["pair_accept"]))

    def test_confidence_and_minimum_validity(self):
        for box_conf in [0.849, np.nan, -1]:
            result = G.decide_pairs("geometry_only", stub_geometry(), self.valid, box_conf, self.conf)
            self.assertFalse(any(result["pair_accept"]))
        conf = self.conf.copy()
        conf[:3] = .499
        result = G.decide_pairs("geometry_only", stub_geometry(), self.valid, .99, conf)
        self.assertFalse(any(result["pair_accept"]))
        conf = self.conf.copy()
        conf[0] = .499
        result = G.decide_pairs("geometry_only", stub_geometry(), self.valid, .99, conf)
        self.assertEqual(result["pair_accept"], [False, True, True, True])

    def test_unflip_roundtrip_and_correspondence(self):
        q = np.arange(18, dtype=np.float64).reshape(9, 2) * 11
        mirrored = G.unflip_points(q, 640)
        self.assertEqual(mirrored[0, 0], 639 - q[1, 0])
        np.testing.assert_array_equal(G.unflip_points(mirrored, 640), q)
        residuals = G.normalized_distances(q, G.unflip_points(mirrored, 640),
                                           self.valid, self.valid, 100)
        np.testing.assert_array_equal(residuals, np.zeros(8))
        invalid = self.valid.copy()
        invalid[0] = False
        residuals = G.normalized_distances(q, q, self.valid, invalid, 100)
        self.assertTrue(np.isinf(residuals[0]))

    def test_geometry_parity_and_pin(self):
        xyz = G.F.cuboid_keypoints_3d(1.1, 1.1, .15)
        q = G.F._project(xyz, np.array([.4, -.3, .05]), np.array([.1, .2, 4.]), self.K)
        q[0] += [7., -3.]
        detail = G.geometry_details(q, self.valid, self.K, self.dimensions)
        old = G.F.geometry_scores(q, self.valid, self.K, self.dimensions)
        self.assertEqual(detail["legacy_geometry_scores"], old)
        self.assertEqual(detail["chosen_hypothesis"], "square_footprint")
        self.assertAlmostEqual(np.median(detail["per_corner_remove"]), detail["s_remove"], places=12)
        denominator = detail["projected_diagonal_px"] * 2
        pinned = G.geometry_details(q, self.valid, self.K, self.dimensions,
                                    hypothesis_name="square_footprint", normalization_px=denominator)
        np.testing.assert_allclose(pinned["per_corner_remove"], np.array(detail["per_corner_remove"]) / 2)

    def test_selects_one_hypothesis_not_per_corner_minima(self):
        details = [
            {"name": "first", "s_remove": .01, "s_reproj": .02, "projected_diagonal_px": 100.,
             "normalization_px": 100., "per_corner_remove": [.08, .01] * 4,
             "per_corner_reprojection": [.02] * 8},
            {"name": "second", "s_remove": .02, "s_reproj": .01, "projected_diagonal_px": 120.,
             "normalization_px": 120., "per_corner_remove": [.01, .08] * 4,
             "per_corner_reprojection": [.01] * 8},
        ]
        with patch.object(G.F, "registry_hypotheses", return_value=[("first", None), ("second", None)]), \
             patch.object(G.F, "geometry_scores", return_value={"s_remove": .01}), \
             patch.object(G, "_hypothesis_detail", side_effect=details):
            result = G.geometry_details(np.ones((9, 2)), self.valid, self.K, self.dimensions)
        self.assertEqual(result["chosen_hypothesis"], "first")
        self.assertEqual(result["per_corner_remove"], details[0]["per_corner_remove"])
        self.assertNotEqual(result["per_corner_remove"], [.01] * 8)

    def test_recheck_pins_scale_and_only_rejects_monotonically(self):
        base = np.arange(18, dtype=float).reshape(9, 2)
        candidate = base + 10
        seen = []

        def recheck(q, valid, K, dimensions, *, hypothesis_name, normalization_px):
            seen.append((hypothesis_name, normalization_px))
            active = [np.array_equal(q[list(pair)], candidate[list(pair)]) for pair in G.PAIRS]
            residuals = np.zeros(8)
            first_active = next(i for i, enabled in enumerate(active) if enabled)
            residuals[G.PAIRS[first_active][0]] = .051
            # Unchanged fallback corners may still be bad; they must not veto
            # unrelated retained candidate pairs during structural recheck.
            for enabled, pair in zip(active, G.PAIRS):
                if not enabled:
                    residuals[list(pair)] = 100
            return {"per_corner_remove": residuals.tolist()}

        with patch.object(G, "geometry_details", side_effect=recheck):
            result = G.gate_with_recheck(base, candidate, self.valid, self.K, self.dimensions,
                                         .99, self.conf, policy="geometry_only", geometry=stub_geometry())
        self.assertEqual(len(result["recheck_history"]), 4)
        self.assertEqual(result["points"].tobytes(), base.tobytes())
        self.assertEqual(seen, [("square_footprint", 100.)] * 4)
        self.assertEqual(result["pair_accept"], [False] * 4)

    def test_unchanged_bad_corner_does_not_veto_other_pair(self):
        base = np.arange(18, dtype=float).reshape(9, 2)
        candidate = base + 10
        initial = stub_geometry([.08, .01, .01, .08, .01, .01, .01, .01])
        with patch.object(G, "geometry_details", return_value={"per_corner_remove": [.9, .01, .01, .9, .01, .01, .01, .01]}):
            result = G.gate_with_recheck(base, candidate, self.valid, self.K, self.dimensions,
                                         .99, self.conf, policy="geometry_only", geometry=initial)
        self.assertEqual(result["pair_accept"], [False, True, True, True])
        self.assertEqual(len(result["recheck_history"]), 1)
        self.assertEqual(result["points"][[0, 3, 8]].tobytes(), base[[0, 3, 8]].tobytes())

    def test_shape_finiteness_and_degenerate_input(self):
        with self.assertRaises(ValueError):
            G.geometry_details(np.ones((8, 2)), self.valid, self.K, self.dimensions)
        with self.assertRaises(ValueError):
            G.geometry_details(np.ones((9, 2)), self.valid, self.K * np.nan, self.dimensions)
        with self.assertRaises(ValueError):
            G.geometry_details(np.ones((9, 2)), self.valid, self.K, self.dimensions, hypothesis_name="unknown")
        q = np.zeros((9, 2))
        q[0] = np.nan
        valid = np.zeros(9, bool)
        valid[:3] = True
        result = G.geometry_details(q, valid, self.K, self.dimensions)
        self.assertIsNone(result["chosen_hypothesis"])
        self.assertTrue(np.isinf(result["per_corner_remove"]).all())
        self.assertFalse(result["effective_valid"][0])
        with self.assertRaises(ValueError):
            G.apply_pairs(np.zeros((9, 2)), np.full((9, 2), np.nan), [True] * 4)

    def test_five_corner_legacy_solver_failure_falls_back(self):
        xyz = G.F.cuboid_keypoints_3d(1.1, 1.1, .15)
        q = G.F._project(xyz, np.array([.4, -.3, .05]), np.array([.1, .2, 4.]), self.K)
        valid = np.zeros(9, bool)
        valid[:5] = True
        detail = G.geometry_details(q, valid, self.K, self.dimensions)
        result = G.gate_with_recheck(q, q + 1, valid, self.K, self.dimensions, .99, self.conf,
                                     policy="geometry_only", geometry=detail)
        self.assertEqual(result["points"].tobytes(), q.tobytes())
        self.assertFalse(any(result["pair_accept"]))


if __name__ == "__main__":
    unittest.main()
