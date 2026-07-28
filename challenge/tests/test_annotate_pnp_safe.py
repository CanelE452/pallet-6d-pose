"""Focused regression tests for the single-image safe PnP API."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import annotate_pnp as ap  # noqa: E402


class SafePnPTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        self.dims = (0.80, 0.59, 0.14)
        rvec = np.array([0.15, -0.25, 0.08])
        self.R, _ = cv2.Rodrigues(rvec)
        self.t = np.array([0.10, 0.05, 4.0])
        self.kps = ap.project_3d(
            ap.make_pallet_keypoints_3d(*self.dims),
            self.R,
            self.t,
            self.K,
        )

    def test_explicit_dims_are_honored_and_omitted_dims_track_global(self):
        explicit = ap.solve_pose(
            self.kps,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            auto_swap_dims=False,
        )
        self.assertIsNotNone(explicit)
        self.assertEqual(explicit["dims"], self.dims)

        previous = ap.PALLET_DIMS
        runtime_dims = (0.70, 0.60, 0.13)
        try:
            ap.PALLET_DIMS = runtime_dims
            inherited = ap.solve_pose(
                self.kps,
                self.K,
                img_shape=(480, 640),
                auto_swap_dims=False,
            )
        finally:
            ap.PALLET_DIMS = previous
        self.assertIsNotNone(inherited)
        self.assertEqual(inherited["dims"], runtime_dims)

    def test_topology_reports_corner_edge_and_axis_failures(self):
        missing_vertical_edge = list(self.kps)
        missing_vertical_edge[0] = None
        missing_vertical_edge[3] = None
        diagnostic = ap.assess_keypoint_topology(
            missing_vertical_edge, min_corners=7)
        self.assertFalse(diagnostic["accepted"])
        self.assertIn("insufficient_corners", diagnostic["reasons"])
        self.assertIn("missing_structural_edge", diagnostic["reasons"])
        self.assertIn(
            {"axis": "height", "edge": [0, 3]},
            diagnostic["missing_structural_edges"],
        )

        # A configurable 6-corner experiment may retain two complete edges
        # per axis when the missing corners are not adjacent.
        opposite_missing = list(self.kps)
        opposite_missing[0] = None
        opposite_missing[6] = None
        six_corner = ap.assess_keypoint_topology(
            opposite_missing, min_corners=6)
        self.assertTrue(six_corner["accepted"])
        self.assertEqual(
            six_corner["complete_edges_per_axis"],
            {"width": 2, "height": 2, "depth": 2},
        )

        top_face_only = [None] * 9
        for i in (0, 1, 4, 5):
            top_face_only[i] = self.kps[i]
        axis_failure = ap.assess_keypoint_topology(
            top_face_only,
            min_corners=4,
            reject_missing_structural_edge=False,
            min_complete_edges_per_axis=1,
        )
        self.assertEqual(axis_failure["reason"], "insufficient_axis_coverage")
        self.assertEqual(axis_failure["insufficient_axes"], ["height"])

    def test_safe_wrapper_rejects_six_corners_before_pnp(self):
        seven = list(self.kps)
        seven[0] = None
        accepted = ap.solve_pose_safe(
            seven, self.K, dims=self.dims, img_shape=(480, 640))
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["topology"]["n_corners"], 7)

        six = list(self.kps)
        six[0] = None
        six[6] = None
        result = ap.solve_pose_safe(
            six, self.K, dims=self.dims, img_shape=(480, 640))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "insufficient_corners")
        self.assertIsNone(result["pose"])
        self.assertEqual(result["candidates"], [])

        edge_missing = list(self.kps)
        edge_missing[0] = None
        edge_missing[3] = None
        edge_result = ap.solve_pose_safe(
            edge_missing,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            min_corners=6,
        )
        self.assertFalse(edge_result["accepted"])
        self.assertEqual(edge_result["reason"], "missing_structural_edge")

    def test_wd_candidates_gap_and_guarded_prior_tie_break(self):
        clear = ap.solve_pose(
            self.kps,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            wd_as_given_prob=0.1,
        )
        self.assertEqual(len(clear["_wd_candidates"]), 2)
        self.assertGreater(clear["_wd_score_gap_px"], 0.5)
        self.assertFalse(clear["_wd_ambiguous"])
        self.assertFalse(clear["_wd_prior_used"])
        self.assertEqual(clear["_wd_hypothesis"], "as_given")

        # Force a deliberately wide ambiguity band to exercise the learned
        # prior.  Both real PnP candidates remain fully exposed.
        tied = ap.solve_pose_safe(
            self.kps,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            wd_ambiguity_abs_px=10.0,
            wd_as_given_prob=0.1,
            wd_prior_min_confidence=0.6,
        )
        self.assertTrue(tied["accepted"])
        self.assertEqual(len(tied["candidates"]), 2)
        self.assertTrue(tied["pose"]["_wd_ambiguous"])
        self.assertTrue(tied["pose"]["_wd_prior_used"])
        self.assertTrue(tied["pose"]["_wd_prior_resolved_ambiguity"])
        self.assertEqual(tied["pose"]["_wd_hypothesis"], "swapped")
        self.assertEqual(tied["pose"]["_wd_legacy_hypothesis"], "as_given")

        unresolved = ap.solve_pose_safe(
            self.kps,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            wd_ambiguity_abs_px=10.0,
        )
        self.assertFalse(unresolved["accepted"])
        self.assertEqual(unresolved["reason"], "wd_ambiguous")
        self.assertEqual(len(unresolved["candidates"]), 2)

        low_confidence_prior = ap.solve_pose_safe(
            self.kps,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            wd_ambiguity_abs_px=10.0,
            wd_as_given_prob=0.60,
        )
        self.assertFalse(low_confidence_prior["accepted"])
        self.assertEqual(low_confidence_prior["reason"], "wd_ambiguous")
        self.assertFalse(low_confidence_prior["pose"]["_wd_prior_used"])
        self.assertEqual(
            low_confidence_prior["pose"]["_wd_prior_min_confidence"], 0.65)

    def test_uncertainty_changes_refinement_without_changing_api(self):
        noisy = [list(point) for point in self.kps]
        noisy[0][0] += 20.0
        noisy[0][1] -= 10.0
        plain = ap.solve_pose(
            noisy,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            auto_swap_dims=False,
        )
        weighted = ap.solve_pose(
            noisy,
            self.K,
            dims=self.dims,
            img_shape=(480, 640),
            auto_swap_dims=False,
            keypoint_uncertainties=[20.0] + [1.0] * 8,
        )
        self.assertFalse(plain["_weighted_pnp"])
        self.assertTrue(weighted["_weighted_pnp"])
        self.assertLess(weighted["reproj_error_px"], plain["reproj_error_px"])

    def test_safe_wrapper_rejects_projected_hull_contraction(self):
        raw = np.asarray(self.kps[:8], dtype=np.float64)
        center = raw.mean(axis=0)
        contracted = center + 0.5 * (raw - center)
        projected_all = contracted.tolist() + [center.tolist()]
        selected = {
            "projected_all": projected_all,
            "reproj_error_px": 0.0,
            "_v6_strict_passed": True,
            "_wd_ambiguous": False,
        }

        with mock.patch.object(
                ap, "solve_pose_candidates", return_value=[selected]), \
                mock.patch.object(
                    ap, "_select_pose_candidate", return_value=dict(selected)):
            result = ap.solve_pose_safe(
                self.kps,
                self.K,
                dims=self.dims,
                img_shape=(480, 640),
            )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "projection_contraction")
        self.assertEqual(result["reasons"], ["projection_contraction"])
        self.assertAlmostEqual(
            result["_projection_to_raw_area_ratio"], 0.25, places=5)
        self.assertEqual(result["_projection_contraction_threshold"], 0.75)
        self.assertEqual(
            result["pose"]["_projection_to_raw_area_ratio"],
            result["_projection_to_raw_area_ratio"],
        )
        self.assertEqual(
            result["pose"]["_projection_contraction_threshold"],
            result["_projection_contraction_threshold"],
        )

    def test_projection_guard_skips_when_fewer_than_six_raw_corners_are_finite(self):
        raw_with_nan = [list(point) for point in self.kps]
        for index in (0, 2, 5):
            raw_with_nan[index] = [float("nan"), float("nan")]

        projected = np.asarray(self.kps[:8], dtype=np.float64)
        center = projected.mean(axis=0)
        projected = center + 0.1 * (projected - center)
        selected = {
            "projected_all": projected.tolist() + [center.tolist()],
            "reproj_error_px": 0.0,
            "_v6_strict_passed": True,
            "_wd_ambiguous": False,
        }
        with mock.patch.object(
                ap, "solve_pose_candidates", return_value=[selected]), \
                mock.patch.object(
                    ap, "_select_pose_candidate", return_value=dict(selected)):
            result = ap.solve_pose_safe(
                raw_with_nan,
                self.K,
                dims=self.dims,
                img_shape=(480, 640),
            )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "ok")
        self.assertIsNone(result["_projection_to_raw_area_ratio"])
        self.assertIsNone(
            result["pose"]["_projection_to_raw_area_ratio"])


if __name__ == "__main__":
    unittest.main()
