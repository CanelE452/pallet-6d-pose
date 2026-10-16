"""Visibility semantics stay separate from point reliability and projection."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import unittest
from unittest.mock import patch

import numpy as np

from .keypoint_status import assess_keypoint_status, status_records_to_json


class KeypointStatusTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array([[30., 30.], [80., 30.], [80., 60.], [30., 60.],
                           [40., 20.], [70., 20.], [70., 50.], [40., 50.], [55., 40.]])
        self.K = np.array([[50., 0., 60.], [0., 50., 50.], [0., 0., 1.]])
        self.initial = dict(R_cf=np.eye(3).tolist(), centroid=[0., 0., 4.], cf_extents=[1., .2, 1.])
        self.H = np.zeros(8, dtype=bool)
        self.appearance = dict(states=["VISIBLE_INTERIOR"] * 8, visibility_score=[.9] * 8,
                               boundary_score=[.2] * 8, diagnostics={})
        self.joint = dict(P_usable=[.9] * 8, eligible=[True] * 8, threshold=.45)

    def assess(self, **kwargs):
        return assess_keypoint_status(self.q, self.K, self.initial, self.H,
                                      self.appearance, self.joint, image_hw=(100, 120), **kwargs)

    def test_visible_with_low_usability_is_not_occluded(self):
        self.joint["P_usable"][0] = .05
        record = self.assess()[0]
        self.assertEqual(record.rgb_visibility_state, "VISIBLE_INTERIOR")
        self.assertEqual(record.rgb_boundary_role, "INTERIOR")
        self.assertEqual(record.rgb_visibility_score, .9)
        self.assertEqual(record.observation_usability_state, "LOW_USABILITY")
        self.assertFalse(record.joint_admitted)
        self.assertIsNone(record.low_RGB_support_hypothesis)
        self.assertFalse(record.external_occlusion_confirmed)

    def test_low_RGB_support_is_only_a_hypothesis_even_if_joint_admits(self):
        self.appearance["states"][0] = "UNKNOWN"
        self.appearance["visibility_score"][0] = .1
        self.appearance["boundary_score"][0] = .9
        record = self.assess()[0]
        self.assertEqual(record.rgb_visibility_state, "UNKNOWN")
        self.assertEqual(record.rgb_boundary_role, "UNKNOWN")
        self.assertTrue(record.joint_admitted)
        self.assertEqual(record.low_RGB_support_hypothesis,
                         "POSSIBLE_EXTERNAL_OCCLUSION_OR_LOCALIZATION_ERROR")
        self.assertFalse(record.external_occlusion_confirmed)

    def test_initial_projection_outside_does_not_change_query_admission(self):
        # The principal point is far from the actual image centre.
        self.K[0, 2] = 400.
        record = self.assess()[0]
        self.assertTrue(record.conditional_truncation)
        self.assertFalse(record.prediction_out_of_frame)
        self.assertTrue(record.joint_admitted)
        self.assertEqual(record.rgb_visibility_state, "VISIBLE_INTERIOR")
        self.assertFalse(record.projected_coordinates_used_as_measurements)

    def test_bad_raw_outside_does_not_become_geometric_truncation(self):
        self.q[0] = [130., 30.]
        self.joint["eligible"][0] = False
        self.appearance["states"][0] = "OUT_OF_FRAME"
        record = self.assess()[0]
        self.assertTrue(record.prediction_out_of_frame)
        self.assertFalse(record.conditional_truncation)
        self.assertEqual(record.rgb_visibility_state, "NOT_ASSESSED")
        self.assertFalse(record.joint_admitted)

    def test_simultaneous_self_and_outside_flags_are_preserved(self):
        self.H[0] = True
        self.q[0] = [130., 30.]
        self.K[0, 2] = 400.
        self.appearance["states"][0] = "OUT_OF_FRAME"
        self.joint["eligible"][0] = False
        record = self.assess()[0]
        self.assertTrue(record.geometric_self_occluded)
        self.assertTrue(record.prediction_out_of_frame)
        self.assertTrue(record.conditional_truncation)
        self.assertEqual(record.rgb_visibility_state, "NOT_ASSESSED")
        self.assertIsNone(record.rgb_visibility_score)

    def test_missing_points_native_order_center_and_inputs_are_preserved(self):
        self.q[0] = [-1., -1.]
        self.q[1] = [np.nan, 30.]
        self.joint["eligible"][:2] = [False, False]
        self.appearance["states"][:2] = ["UNKNOWN", "UNKNOWN"]
        arrays = [self.q, self.K, self.H]
        snapshots = [a.copy() for a in arrays]
        for array in arrays:
            array.flags.writeable = False
        initial, appearance, joint = deepcopy(self.initial), deepcopy(self.appearance), deepcopy(self.joint)
        with patch("builtins.open", side_effect=AssertionError("No file reads")):
            result = self.assess()
        self.assertEqual(len(result), 8)
        self.assertEqual([r.corner for r in result], list(range(8)))
        self.assertTrue(result[0].prediction_missing)
        self.assertTrue(result[1].prediction_missing)
        self.assertIsNone(result[0].prediction_out_of_frame)
        self.assertEqual(result[1].raw_xy, (None, 30.))
        self.assertEqual(result[2].raw_xy, tuple(self.q[2]))
        for before, after in zip(snapshots, arrays):
            np.testing.assert_array_equal(before, after)
            self.assertFalse(after.flags.writeable)
        np.testing.assert_array_equal(self.q[8], [55., 40.])
        self.assertEqual(self.initial, initial)
        self.assertEqual(self.appearance, appearance)
        self.assertEqual(self.joint, joint)
        with self.assertRaises(FrozenInstanceError):
            result[2].corner = 7
        encoded = status_records_to_json(result)
        json.dumps(encoded, allow_nan=False)
        encoded[2]["raw_xy"][0] = 999.
        self.assertEqual(result[2].raw_xy, tuple(self.q[2]))

    def test_invalid_initial_projection_is_unknown_not_truncation(self):
        self.initial["centroid"][2] = .1
        record = self.assess()[0]
        self.assertFalse(record.initial_projection_valid)
        self.assertIsNone(record.conditional_truncation)
        self.assertEqual(record.initial_projection_xy, (None, None))
        self.assertTrue(record.joint_admitted)

    def test_reject_postoverwrite_RGB_and_changed_eligibility(self):
        self.appearance["diagnostics"]["joint_observation"] = {"threshold": .45}
        with self.assertRaisesRegex(ValueError, "before joint"):
            self.assess()
        del self.appearance["diagnostics"]["joint_observation"]
        self.joint["eligible"][0] = False
        with self.assertRaisesRegex(ValueError, "retain original"):
            self.assess()

    def test_boundary_role_is_from_independent_visibility_assessment(self):
        self.appearance["states"][0] = "VISIBLE_BOUNDARY"
        self.appearance["boundary_score"][0] = .7
        self.joint["P_usable"][0] = .1
        record = self.assess()[0]
        self.assertEqual(record.rgb_boundary_role, "BOUNDARY")
        self.assertEqual(record.rgb_boundary_score, .7)
        self.assertFalse(record.joint_admitted)


if __name__ == "__main__":
    unittest.main()
