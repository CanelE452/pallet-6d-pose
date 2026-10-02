from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.pallet_n3_completion_v3 import reuse as R


def _row(base: np.ndarray, candidate: np.ndarray, *, permutations=None,
         hw=(100, 100)) -> dict:
    gt = np.stack([np.arange(9, dtype=float), np.zeros(9)], axis=1)
    if permutations is None:
        permutations = [list(range(9))]
    return {
        "id": "frame0", "session": "session0", "hw": list(hw),
        "gt": gt.tolist(), "valid": [True] * 9,
        "permutations": permutations,
        "material": "plastic", "occlusion": "clean",
        "predictions": {"base": base.tolist(), "candidate": candidate.tolist()},
        "detected": {"base": True, "candidate": True},
        "matched": {"base": True, "candidate": True},
    }


class ReusePureTests(unittest.TestCase):
    def test_selected_uses_highest_score_and_empty_is_missing(self):
        self.assertIsNone(R._selected([]))
        rows = [{"score": .1, "id": 1}, {"score": .9, "id": 2}]
        self.assertEqual(R._selected(rows)["id"], 2)

    def test_explicit_source_root_must_contain_raw_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                R.resolve_source_root(directory)

    def test_mean_dict_keeps_invariant_counts_and_means_metrics(self):
        result = R._mean_dict([
            {"frames": 319, "metric": 1., "nested": {"x": 2.}},
            {"frames": 319, "metric": 3., "nested": {"x": 4.}},
            {"frames": 319, "metric": 5., "nested": {"x": 6.}},
        ])
        self.assertEqual(result["frames"], 319)
        self.assertEqual(result["metric"], 3.)
        self.assertEqual(result["nested"]["x"], 4.)

    def test_damage_fixes_base_whole_object_branch(self):
        gt = np.stack([np.arange(9, dtype=float), np.zeros(9)], axis=1)
        swap = [1, 0, 2, 3, 4, 5, 6, 7, 8]
        base = gt[swap].copy()
        candidate = gt.copy()  # would choose identity if re-optimized after correction
        row = _row(base, candidate, permutations=[list(range(9)), swap])
        result, frames = R.fixed_branch_damage([row], "base", "candidate", None)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(result["base_branch_fixed_for_candidate"])
        self.assertEqual(frames[0]["base_branch"], 1)
        self.assertGreater(frames[0]["candidate_mean_px"], 0.)

    def test_cap_violation_is_not_hidden(self):
        gt = np.stack([np.arange(9, dtype=float), np.zeros(9)], axis=1)
        candidate = gt.copy()
        candidate[:8, 1] += 2.
        row = _row(gt, candidate)
        result, _ = R.fixed_branch_damage([row], "base", "candidate", .01)
        self.assertEqual(result["status"], "BLOCKED_INTEGRITY")
        self.assertEqual(result["movement"]["cap_violations"], 1)
        self.assertEqual(result["corner_change"]["worsened"], 8)

    def test_paired_pose_keeps_failures_and_recoveries_separate(self):
        frames = [
            {"id": "a", "delta_px": -1.},
            {"id": "b", "delta_px": 2.},
        ]
        base = [
            {"id": "a", "available": True, "translation_cm": 3.,
             "rotation_deg": 2., "yaw_deg": 1.},
            {"id": "b", "available": False},
            {"id": "c", "available": True, "translation_cm": 1.,
             "rotation_deg": 1., "yaw_deg": 1.},
        ]
        candidate = [
            {"id": "a", "available": True, "translation_cm": 2.,
             "rotation_deg": 3., "yaw_deg": 1.},
            {"id": "b", "available": True, "translation_cm": 4.,
             "rotation_deg": 4., "yaw_deg": 4.},
            {"id": "c", "available": False},
        ]
        result = R.paired_2d_pose(frames, base, candidate)
        self.assertEqual(result["new_pose_failures"], 1)
        self.assertEqual(result["pose_recoveries"], 1)
        self.assertEqual(result["paired_with_fixed_branch_2d_frames"], 1)
        self.assertEqual(result["translation_cm"]["improved"], 1)


class ReuseRawRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.source_root = R.resolve_source_root()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))
        cls.core, cls.state = R.build_core(cls.source_root, include_pose=True)

    def test_A_fixed_numbers_and_new_pose_p90(self):
        section = self.core["A"]
        self.assertEqual(section["status"], "COMPLETE")
        self.assertEqual(section["fixed_number_regression"]["R0"]["status"], "PASS")
        self.assertEqual(
            section["fixed_number_regression"]["N3_DIM_SYM"]["status"], "PASS")
        self.assertAlmostEqual(
            section["new_pose_P90"]["R0"]["pose_translation_cm_P90"],
            40.53016590882621, places=10)
        self.assertIsNotNone(
            section["new_pose_P90"]["N3_DIM_SYM"]["pose_rotation_deg_P90"])

    def test_B_uses_all_three_seeds_and_same_denominator(self):
        section = self.core["B"]
        self.assertEqual(set(section["methods"]), set(R.DCP_ARMS))
        for panel in section["methods"].values():
            self.assertEqual(set(panel["per_seed"]), {"1", "2", "3"})
            self.assertEqual(panel["mean"]["headline"]["frames"], 319)
            self.assertEqual(panel["mean"]["headline"]["full_supervised_corners"], 2499)
            self.assertEqual(panel["mean"]["headline"]["observed_corners"], 2445)

    def test_C_and_D_locked_counts_and_unknown_visibility(self):
        self.assertEqual(self.core["C"]["frame_counts"], {"plastic": 194, "wood": 125})
        self.assertEqual(self.core["D"]["frame_counts"], {
            "clean": 29, "moderate": 20, "severe": 79, "unclassified": 191})
        visibility = self.core["D"]["corner_visibility"]
        self.assertEqual(visibility["status"], "BLOCKED_LABEL")
        self.assertIsNone(visibility["known_visible_corners"])
        self.assertEqual(visibility["unknown_supervised_corners"], 2499)

    def test_I_main319_remains_null_but_locked_safe128_is_scored(self):
        section, _ = R.build_student_audit(self.source_root, self.state)
        self.assertEqual(section["DEV319_table"]["status"], "BLOCKED_CONTRACT")
        self.assertIsNone(section["DEV319_table"]["result"])
        safe = section["safe_common_cohort"]
        self.assertEqual(safe["status"], "COMPLETE_REUSED_DEV")
        self.assertEqual(safe["methods"]["R0"]["mean"]["headline"]["frames"], 128)
        self.assertEqual(safe["exposure"]["student_train_RGB_overlap"], 0)
        self.assertFalse(safe["exposure"]["independent_test"])


if __name__ == "__main__":
    unittest.main()
