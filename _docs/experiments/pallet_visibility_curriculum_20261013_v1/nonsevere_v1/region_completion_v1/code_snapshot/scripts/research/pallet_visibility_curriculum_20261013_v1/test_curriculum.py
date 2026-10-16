"""Analytic geometry checks of information selection and initial-pose safety."""
import json
import unittest

import cv2
import numpy as np

from .curriculum import project, solve_curriculum


def fixture():
    a, b, c = .55, .075, .55
    xyz = np.array([[-a,-b,-c], [a,-b,-c], [a,b,-c], [-a,b,-c],
                    [-a,-b,c], [a,-b,c], [a,b,c], [-a,b,c]])
    K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
    R = cv2.Rodrigues(np.array([.38, -.45, .04]))[0]
    t = np.array([.06, -.03, 3.5])
    initial_R = cv2.Rodrigues(np.array([.02, -.025, .015]))[0] @ R
    initial_t = t + np.array([.025, -.018, .03])
    points = project(xyz, R, t, K)
    return xyz, K, R, t, initial_R, initial_t, points


def appearance(states):
    return {"states": states, "visibility_score": [.95] * 8,
            "boundary_score": [.95 if s == "VISIBLE_BOUNDARY" else .1 for s in states],
            "local_accuracy_score": [.95] * 8}


class CurriculumTests(unittest.TestCase):
    def test_easy_fit_improves_true_pose_and_never_uses_hidden_or_external_outlier(self):
        xyz, K, R, t, initial_R, initial_t, truth = fixture()
        points = truth.copy()
        points[3] += [120., -90.]  # Actual external occluder corrupts detector corner.
        points[7] += [-160., 80.]  # Hidden corner hallucination must not enter fitting.
        hidden = np.zeros(8, dtype=bool)
        hidden[7] = True
        states = ["VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY",
                  "OCCLUSION_SUSPECTED", "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR",
                  "VISIBLE_INTERIOR", "SELF_OCCLUDED"]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, hidden, appearance(states))
        self.assertTrue(result["accepted"])
        self.assertTrue(result["stages"][1]["accepted"])
        predicted = np.asarray(result["projected"])
        initial_error = np.mean(np.linalg.norm(project(xyz, initial_R, initial_t, K) - truth, axis=1))
        final_error = np.mean(np.linalg.norm(predicted - truth, axis=1))
        self.assertLess(final_error, initial_error * .15)
        self.assertLess(np.linalg.norm(np.asarray(result["t"]) - t), .003)
        self.assertEqual(result["observed_points"], points.tolist())
        for stage in result["stages"]:
            self.assertFalse({3, 7} & set(stage["used_ids"]))
        np.testing.assert_allclose(result["hidden_projected"], predicted[[7]])
        json.dumps(result, allow_nan=False)

    def test_thin_easy_set_waits_until_interior_addition_supplies_geometry(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        states = ["VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "UNKNOWN",
                  "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "VISIBLE_INTERIOR", "SELF_OCCLUDED"]
        hidden = np.arange(8) == 7
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, hidden, appearance(states))
        self.assertFalse(result["stages"][1]["accepted"])
        self.assertEqual(result["stages"][1]["reason"], "fewer_than_four_independent_observations")
        self.assertTrue(result["stages"][2]["accepted"])
        self.assertLess(result["residual_after"], .5)
        self.assertTrue(any(s["name"].startswith("add_consistent_unknown") for s in result["stages"]))
        accepted = [s for s in result["stages"][2:] if s["accepted"]]
        self.assertGreaterEqual(len(accepted), 3)
        for previous, current in zip(accepted, accepted[1:]):
            self.assertEqual(len(set(current["used_ids"]) - set(previous["used_ids"])), 1)

    def test_uniform_interior_is_added_with_less_weight_than_boundary(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        states = ["VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "UNKNOWN",
                  "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "VISIBLE_INTERIOR", "SELF_OCCLUDED"]
        evidence = appearance(states)
        evidence["local_accuracy_score"] = [.85 if s == "VISIBLE_BOUNDARY" else .18 for s in states]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.arange(8) == 7, evidence)
        self.assertTrue(result["stages"][2]["accepted"])
        added = result["stages"][2]
        weights = dict(zip(added["used_ids"], added["weights"]))
        self.assertLess(weights[2], weights[0] / 4)
        self.assertLess(result["residual_after"], .5)

    def test_unknown_can_add_evidence_after_visible_training_residual_reaches_plateau(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        states = ["VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "SELF_OCCLUDED",
                  "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "UNKNOWN", "SELF_OCCLUDED"]
        points[6] += [2, -1]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t,
                                  np.isin(np.arange(8), [3, 7]), appearance(states))
        self.assertEqual(result["unknown_added_ids"], [6])
        unknown_stage = next(s for s in result["stages"] if s["name"].startswith("add_consistent_unknown"))
        self.assertTrue(unknown_stage["accepted"])
        self.assertGreater(unknown_stage["visible_score_after"], unknown_stage["visible_score_before"])
        self.assertLess(unknown_stage["visible_score_after"] - unknown_stage["visible_score_before"], .10)
        self.assertLess(unknown_stage["fit_score_after"], unknown_stage["fit_score_before"])

    def test_one_stage_control_uses_all_credible_visible_observations_in_one_fit(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        states = ["VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "UNKNOWN",
                  "VISIBLE_BOUNDARY", "VISIBLE_INTERIOR", "VISIBLE_INTERIOR", "SELF_OCCLUDED"]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.arange(8) == 7,
                                  appearance(states), dict(staged_visible=False, enable_unknown=False))
        self.assertEqual(len(result["stages"]), 2)
        self.assertTrue(result["stages"][1]["accepted"])
        self.assertEqual(result["used_ids"], [0, 1, 2, 4, 5, 6])

    def test_three_visible_points_and_unknowns_preserve_initial_exactly(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        states = ["VISIBLE_BOUNDARY"] * 3 + ["UNKNOWN"] * 5
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.zeros(8, bool), appearance(states))
        self.assertFalse(result["accepted"])
        np.testing.assert_array_equal(result["R"], initial_R)
        np.testing.assert_array_equal(result["t"], initial_t)
        self.assertEqual(result["used_ids"], [])

    def test_collinear_image_observations_cannot_claim_independent_pose_support(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        points[:, 0] = np.arange(8) * 10 + 280
        points[:, 1] = 210.
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.zeros(8, bool),
                                  appearance(["VISIBLE_BOUNDARY"] * 8))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["fallback_reason"], "insufficient_image_spread")
        np.testing.assert_array_equal(result["R"], initial_R)

    def test_incoherent_unknown_outlier_is_not_added(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        points[6] += [75, -100]
        states = ["VISIBLE_BOUNDARY"] * 4 + ["VISIBLE_INTERIOR"] * 2 + ["UNKNOWN", "SELF_OCCLUDED"]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.arange(8) == 7, appearance(states))
        self.assertTrue(result["accepted"])
        self.assertFalse(any(6 in s["used_ids"] for s in result["stages"]))

    def test_thin_planar_front_face_does_not_create_confident_hidden_predictions(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        evidence = appearance(["VISIBLE_BOUNDARY"] * 4 + ["UNKNOWN"] * 4)
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.zeros(8, bool), evidence)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["fallback_reason"], "unstable_pose_geometry")
        self.assertEqual(result["stages"][1]["geometry"]["jacobian_rank"], 6)
        np.testing.assert_array_equal(result["R"], initial_R)

    def test_broad_planar_top_face_is_valid_independent_support(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        states = ["VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "SELF_OCCLUDED", "SELF_OCCLUDED",
                  "VISIBLE_BOUNDARY", "VISIBLE_BOUNDARY", "SELF_OCCLUDED", "SELF_OCCLUDED"]
        hidden = np.isin(np.arange(8), [2, 3, 6, 7])
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, hidden, appearance(states))
        self.assertTrue(result["accepted"])
        self.assertLess(result["residual_after"], .1)

    def test_large_coherent_wrong_corner_shift_is_rejected_by_initial_prior_limits(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        points += [180., 120.]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.zeros(8, bool),
                                  appearance(["VISIBLE_BOUNDARY"] * 8))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["fallback_reason"], "exceeds_initial_pose_motion_limit")
        np.testing.assert_array_equal(result["t"], initial_t)

    def test_fitting_does_not_promote_geometric_reprojection_to_observation(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        evidence = appearance(["VISIBLE_BOUNDARY"] * 8)
        evidence["independent"] = [True, True, True, False, False, False, False, False]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.zeros(8, bool), evidence)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["visible_ids"], [0, 1, 2])
        self.assertFalse(result["reprojected_points_reused_as_observations"])

    def test_missing_coordinate_remains_missing_in_observation_output(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        points[7] = [np.nan, np.nan]
        result = solve_curriculum(points, xyz, K, initial_R, initial_t, np.arange(8) == 7,
                                  appearance(["VISIBLE_BOUNDARY"] * 7 + ["SELF_OCCLUDED"]))
        self.assertEqual(result["observed_points"][7], [None, None])
        self.assertTrue(result["accepted"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
