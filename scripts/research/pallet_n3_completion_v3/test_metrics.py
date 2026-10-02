from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from scripts.research.pallet_n3_completion_v3 import metrics


GT = np.array([
    [10., 10.], [20., 10.], [30., 10.], [40., 10.],
    [10., 20.], [20., 20.], [30., 20.], [40., 20.], [25., 15.],
])
IDENTITY = np.arange(9, dtype=np.int64)[None]


def shifted(amount):
    value = GT.copy()
    value[:8, 0] += amount
    return value.tolist()


def frame(fid, session, base, n3, *, base_matched=True, n3_matched=True,
          base_detected=True, n3_detected=True, material=None, occlusion=None):
    row = {
        "id": fid,
        "session": session,
        "hw": [60, 80],  # diagonal = 100 px
        "gt": GT.tolist(),
        "valid": [True] * 9,
        "permutations": IDENTITY.tolist(),
        "predictions": {"base": base, "n3": n3},
        "matched": {"base": base_matched, "n3": n3_matched},
        "detected": {"base": base_detected, "n3": n3_detected},
    }
    if material is not None:
        row["material"] = material
    if occlusion is not None:
        row["occlusion"] = occlusion
    return row


class DenominatorAndFailureTests(unittest.TestCase):
    def test_failure_stays_in_full_denominator_and_penalty(self):
        rows = [
            frame("a", "s1", shifted(0), shifted(0), material="plastic"),
            frame("b", "s2", None, None, base_matched=False, n3_matched=False,
                  base_detected=False, n3_detected=False),
        ]
        result = metrics.evaluate_method(rows, "base")
        den = result["corner"]["denominators"]
        values = result["corner"]["metrics"]
        self.assertEqual(den["frames"], 2)
        self.assertEqual(den["matched_frames"], 1)
        self.assertEqual(den["full_supervised_corners"], 16)
        self.assertEqual(den["observed_corners"], 8)
        self.assertEqual(values["matched_pooled_corner8_median_px"], 0.)
        self.assertEqual(values["full_PCK10_fraction"], .5)
        self.assertEqual(values["E_sym"], .5)
        self.assertEqual(values["full_penalty_median_px"], 50.)
        self.assertEqual(values["full_penalty_P90_px"], 100.)
        self.assertEqual(set(result["subgroups"]["material"]),
                         {"plastic", "unclassified"})
        self.assertEqual(set(result["subgroups"]["occlusion"]),
                         {"unclassified"})

    def test_matched_undetected_is_rejected(self):
        row = frame("a", "s1", shifted(0), shifted(0),
                    base_matched=True, base_detected=False)
        with self.assertRaisesRegex(ValueError, "matched frame cannot be undetected"):
            metrics.evaluate_method([row], "base")

    def test_whole_object_symmetry_branch_is_reused(self):
        permutation = np.array([
            np.arange(9),
            [1, 0, 3, 2, 5, 4, 7, 6, 8],
        ])
        pred = GT[permutation[1]].tolist()
        row = frame("a", "s1", pred, pred)
        row["permutations"] = permutation.tolist()
        result = metrics.evaluate_method([row], "base")
        self.assertEqual(result["corner_rows"][0]["branch"], 1)
        self.assertEqual(result["corner"]["metrics"]["E_sym"], 0.)


class PairedTruthfulnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [
            frame("improves", "s1", shifted(5), shifted(1),
                  material="plastic", occlusion="none"),
            frame("worsens", "s1", shifted(1), shifted(12),
                  material="plastic", occlusion="severe"),
            frame("same", "s2", shifted(2), shifted(2), material="wood"),
        ]
        cls.result = metrics.evaluate_comparison(cls.rows, "base", "n3")

    def test_improvement_no_change_and_worsening_are_all_retained(self):
        change = self.result["paired"]["frame_change"]
        self.assertEqual(change["comparable_frames"], 3)
        self.assertEqual(change["improved_frames"], 1)
        self.assertEqual(change["no_change_frames"], 1)
        self.assertEqual(change["worsened_frames"], 1)
        self.assertEqual(
            change["improved_frames"] + change["no_change_frames"] +
            change["worsened_frames"], 3)

    def test_conditional_summary_does_not_hide_full_population_harm(self):
        base = self.result["base"]["corner"]
        n3 = self.result["candidate"]["corner"]
        self.assertLessEqual(
            n3["metrics"]["matched_pooled_corner8_median_px"],
            base["metrics"]["matched_pooled_corner8_median_px"])
        self.assertLess(n3["metrics"]["full_PCK10_fraction"],
                        base["metrics"]["full_PCK10_fraction"])
        self.assertEqual(n3["denominators"]["full_supervised_corners"], 24)
        self.assertEqual(n3["denominators"]["observed_corners"], 24)

    def test_paired_comparison_rejects_detection_state_drift(self):
        rows = [frame("a", "s1", shifted(0), None, n3_matched=False,
                      n3_detected=False)]
        base = metrics.evaluate_method(rows, "base")
        n3 = metrics.evaluate_method(rows, "n3")
        with self.assertRaisesRegex(ValueError, "preserve base detection"):
            metrics.paired_corner_analysis(base["corner_rows"], n3["corner_rows"])

    def test_bootstrap_is_locked_paired_session_10k(self):
        boot = self.result["paired"]["bootstrap"]
        self.assertEqual(boot["unit"], "session")
        self.assertEqual(boot["sessions"], 2)
        self.assertEqual(boot["resamples"], 10_000)
        self.assertEqual(boot["seed"], 20260917)
        self.assertTrue(boot["paired_sampling"])
        self.assertTrue(boot["recalculates_each_statistic"])
        self.assertEqual(
            boot["metrics"]["full_PCK10_fraction"]["status"], "COMPLETE")
        self.assertEqual(
            boot["metrics"]["full_PCK10_fraction"]["invalid_draws"], 0)
        with self.assertRaisesRegex(ValueError, "locks 10,000"):
            metrics.session_bootstrap(
                self.result["base"]["corner_rows"],
                self.result["candidate"]["corner_rows"], resamples=100)

    def test_weighted_bootstrap_quantiles_match_explicit_session_repetition(self):
        scored = self.result["base"]["corner_rows"]
        sessions = ["s1", "s2"]
        counts = np.array([2, 1], dtype=np.int64)
        explicit = metrics._bootstrap_stats(scored, dict(zip(sessions, counts)))
        prepared = metrics._bootstrap_arrays(scored, {"s1": 0, "s2": 1})
        weighted = metrics._prepared_bootstrap_stats(prepared, counts)
        self.assertEqual(set(explicit), set(weighted))
        for key in explicit:
            self.assertAlmostEqual(explicit[key], weighted[key], places=12)

    def test_missing_labels_remain_unclassified_in_paired_subgroups(self):
        groups = self.result["paired"]["subgroups"]
        self.assertIn("unclassified", groups["occlusion"])
        self.assertEqual(groups["occlusion"]["unclassified"]["no_change_frames"], 1)
        self.assertEqual(set(groups["material"]), {"plastic", "wood"})


class PoseContractTests(unittest.TestCase):
    def test_pose_is_prediction_only_and_keeps_full_coverage_denominator(self):
        calls = []

        class FakePose:
            @staticmethod
            def infer(points, K, xyz, source=False):
                calls.append(points)
                return {"available": points is not None}

            @staticmethod
            def metric(task):
                fid, prediction, _ = task
                if not prediction["available"]:
                    return {"id": fid, "available": False}
                return {
                    "id": fid, "available": True, "translation_cm": 2.,
                    "rotation_deg": 3., "yaw_deg": 1., "IoU3D": .5,
                    "ADDsym_normalized": .02,
                }

            @staticmethod
            def pose_auc(values, _maximum):
                return float(np.isfinite(np.asarray(values)).mean())

        row = frame("a", "s1", shifted(0), shifted(0),
                    base_matched=False, base_detected=True)
        row["pose"] = {
            "K": np.eye(3).tolist(), "xyz": [1., .2, .8],
            "truth": {"unused_by_fake": True},
        }
        with mock.patch.object(metrics, "_pose_contract", return_value=FakePose):
            result = metrics.evaluate_method([row], "base", include_pose=True)
        self.assertIsNotNone(calls[0])  # unmatched did not gate prediction-only PnP
        self.assertEqual(result["pose"]["denominators"]["frames"], 1)
        self.assertEqual(result["pose"]["denominators"]["available_frames"], 1)
        self.assertEqual(result["pose"]["translation_cm"]["median"], 2.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
