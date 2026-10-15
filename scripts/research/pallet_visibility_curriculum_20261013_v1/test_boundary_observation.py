"""Image-only observations, bounded assumptions and no fictitious confidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np

from . import boundary_observation as B
from .boundary_visibility import BoundaryVisibility


class BoundaryObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        weights = Path(self.tmp.name) / "unit.pt"
        weights.write_bytes(b"injected unit checkpoint")
        model = dict(schema="unit", mean=[0.] * 26, scale=[1.] * 26, coefficients=[0.] * 26,
                     intercept=np.log(.9 / .1), visible_threshold=.2, boundary_threshold=.35)
        self.visibility = BoundaryVisibility(model, weights)
        self.image = np.zeros((100, 100, 3), np.uint8)
        self.image[30:70, 30:70] = [50, 180, 50]
        self.mask = np.zeros((100, 100), bool)
        self.mask[30:70, 30:70] = True
        self.q = np.array([[27., 27.], [73., 27.], [73., 73.], [27., 73.],
                           [40., 40.], [60., 40.], [60., 60.], [40., 60.], [50., 50.]])
        self.K = np.array([[50., 0., 50.], [0., 50., 50.], [0., 0., 1.]])
        self.initial = dict(available=True, R_cf=np.eye(3).tolist(), centroid=[0., 0., 4.], cf_extents=[1., 1., 1.])
        self.profile = dict(minimum_visibility=.65, minimum_edge_length_px=8., minimum_inlier_count=5,
            minimum_inlier_fraction=.4, minimum_inlier_span_fraction=.5, maximum_line_rms_px=1.5,
            maximum_angle_change_deg=15., minimum_material_inside=.6, maximum_material_outside=.4,
            minimum_transition=.3, maximum_context_outside=.5, minimum_context_inside=.45,
            transition_probe_px=2., context_probe_px=5., maximum_corner_shift_px=12.)
        self.calls = []

    def evidence(self, image, q, H, weights, config, device):
        return dict(states=["SELF_OCCLUDED" if x else "VISIBLE_INTERIOR" for x in H],
            visibility_score=[0. if x else .9 for x in H], boundary_score=[.1] * 8,
            local_accuracy_score=[.3] * 8, diagnostics=dict(patch_radius_px=4,
                corners=[dict(reason="frozen_source_CNN_mask_and_raw_RGB_features") for _ in range(8)]))

    def provider(self, image, box):
        self.calls.append(box.copy())
        return dict(masks=np.stack([self.mask] * 3), quality=[.9, .9, .8])

    def assess(self):
        api = B.BoundaryObservation(self.visibility, boundary_profile=self.profile,
            uncertainty=dict(repetitions=64, block_length=3, normal_floor_px=1., seed=20261013),
            fusion=dict(detector_sigma_px=2., maximum_alpha=.5), mask_provider=self.provider)
        with patch.object(B, "_infer_evidence", side_effect=self.evidence) as tiny:
            result = api.assess(self.image, self.q, self.K, self.initial)
            self.assertEqual(tiny.call_count, 1)
        return result

    def test_two_observed_lines_and_bounded_blend_not_projected_measurements(self):
        result = self.assess()
        self.assertTrue(result["correction"]["observed_refined_ids"])
        raw = np.asarray(result["raw_points"])
        observed = np.asarray(result["observed_points"])
        fused = np.asarray(result["fused_points"])
        self.assertTrue(np.any(np.linalg.norm(observed[:8] - raw[:8], axis=1) > 0))
        self.assertTrue(all(row["alpha"] <= .5 for row in result["fusion"]))
        np.testing.assert_allclose(fused[:8] - raw[:8], .5 * (observed[:8] - raw[:8]), atol=1e-8)
        self.assertFalse(result["provenance"]["projected_coordinates_used_as_measurements"])
        self.assertTrue(all(len(row["incident_shared_edges"]) == 2 for row in result["intersection_covariance"].values()))

    def test_mask_box_uses_raw_queries_not_projection_or_ninth_center(self):
        self.q[8] = [999., 999.]
        self.K[0, 2] = 400.
        result = self.assess()
        np.testing.assert_array_equal(self.calls[-1], [27., 27., 73., 73.])
        self.assertEqual(result["mask"]["policy"], "BOX_ONLY")
        self.assertEqual(result["mask"]["points"], [])
        self.assertEqual(result["mask"]["selected_index"], 0)
        self.assertTrue(result["correction"]["observed_refined_ids"])
        self.assertFalse(any(result["boundary_visibility"]["geometric_boundary_candidates"]))

    def test_uniform_mask_cannot_create_coordinates_or_confidence(self):
        self.mask[:] = True
        result = self.assess()
        np.testing.assert_array_equal(result["observed_points"], self.q)
        np.testing.assert_array_equal(result["fused_points"], self.q)
        self.assertEqual(result["intersection_covariance"], {})
        self.assertFalse(result["provenance"]["covariance_is_calibrated_keypoint_confidence"])

    def test_deterministic_resampling_and_floor_do_not_certify_semantics(self):
        a, b = self.assess(), self.assess()
        self.assertEqual(a["intersection_covariance"], b["intersection_covariance"])
        self.assertEqual(a["fusion"], b["fusion"])
        self.assertTrue(a["intersection_covariance"])
        for row in a["intersection_covariance"].values():
            self.assertGreaterEqual(min(row["eigenvalues_px2"]), 1. - 1e-8)
        self.assertFalse(a["provenance"]["fusion_is_calibrated_Bayes_estimator"])
        self.assertEqual(a["visibility_appearance"], b["visibility_appearance"])

    def test_hidden_center_readonly_inputs_and_no_pose_fit(self):
        snapshots = [a.copy() for a in [self.image, self.q, self.K]]
        for a in [self.image, self.q, self.K]:
            a.flags.writeable = False
        before = deepcopy(self.initial)
        with patch("cv2.solvePnP", side_effect=AssertionError("No pose fit")):
            result = self.assess()
        for a, beforearray in zip([self.image, self.q, self.K], snapshots):
            np.testing.assert_array_equal(a, beforearray)
            self.assertFalse(a.flags.writeable)
        H = np.asarray(result["fixed_H"], bool)
        self.assertTrue(H.any())
        np.testing.assert_array_equal(np.asarray(result["observed_points"])[:8][H], self.q[:8][H])
        np.testing.assert_array_equal(np.asarray(result["fused_points"])[:8][H], self.q[:8][H])
        self.assertEqual(result["fused_points"][8], [50., 50.])
        self.assertEqual(self.initial, before)

    def test_invalid_initial_geometry_fails_before_any_model_forward(self):
        original = deepcopy(self.initial)
        invalid = []
        negative = deepcopy(original)
        negative["cf_extents"][0] = -1.
        invalid.append(negative)
        reflected = deepcopy(original)
        reflected["R_cf"][0][0] = -1.
        invalid.append(reflected)
        nonfinite = deepcopy(original)
        nonfinite["centroid"][0] = np.nan
        invalid.append(nonfinite)
        behind = deepcopy(original)
        behind["centroid"][2] = .1
        invalid.append(behind)
        for initial in invalid:
            self.initial = initial
            with self.assertRaises(ValueError):
                self.assess()
        self.assertEqual(self.calls, [])
        self.initial = original

    def test_missing_query_is_retained_json_safe_and_self_alias_exact(self):
        self.q[0] = [np.nan, np.nan]
        result = self.assess()
        self.assertEqual(result["raw_points"][0], [None, None])
        self.assertEqual(result["observed_points"][0], [None, None])
        self.assertEqual(result["fused_points"][0], [None, None])
        self.assertEqual(result["self_hidden"], result["fixed_H"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
