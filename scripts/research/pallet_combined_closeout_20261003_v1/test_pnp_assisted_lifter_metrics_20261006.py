"""Metric fixtures are memory-only and never create real human sidecars."""
from __future__ import annotations

import copy
import unittest

from evaluate_pnp_assisted_lifter_20261006 import compute, paired_summary, prediction_points


def fixture(error=0.0):
    record = dict(frame_id="fixture:0", session_id="fixture",
                  points=[dict(id=i, x=0.0, y=0.0, source="pnp_projected") for i in range(8)])
    selected = dict(selected_index=0, selected_box_xyxy=[0, 0, 10, 10])
    method = dict(selected_index=0, selected_object=selected,
                  keypoints=[[float(error), 0.0] for _ in range(8)], keypoints_mask=[True] * 8)
    predictions = {"fixture:0": dict(methods=dict(Base=copy.deepcopy(method), N3=copy.deepcopy(method)))}
    return [record], predictions, {"fixture:0": "same"}


class AssistedMetricTest(unittest.TestCase):
    def test_pck_includes_exact_10_euclidean_boundary(self):
        refs, preds, choices = fixture()
        for method in preds["fixture:0"]["methods"].values():
            method["keypoints"] = [[6.0, 8.0]] * 7 + [[6.0, 8.000000001]]
        metrics, points, _ = compute(refs, preds, choices)
        self.assertEqual(metrics["methods"]["Base"]["pck10_hit_count"], 7)
        self.assertEqual(points[0]["Base_error_px"], 10.0)

    def test_different_object_keeps_all_eight_failures(self):
        refs, preds, choices = fixture()
        choices["fixture:0"] = "different"
        metrics, points, _ = compute(refs, preds, choices)
        for method in ("Base", "N3"):
            result = metrics["methods"][method]
            self.assertEqual(result["full_reference_point_count"], 8)
            self.assertEqual(result["pck10_full_reference_percent"], 0.0)
            self.assertEqual(result["conditional_error"]["count"], 0)
            self.assertIsNone(result["conditional_error"]["median_px"])
            self.assertEqual(result["failure_reasons"], {"wrong_selected_object": 8})
        self.assertEqual(len(points), 8)

    def test_null_nonfinite_explicit_missing_keep_full_denominator(self):
        refs, preds, choices = fixture()
        for method in preds["fixture:0"]["methods"].values():
            method["keypoints"][0] = [None, None]
            method["keypoints"][1] = [float("nan"), 0]
            method["keypoints"][2] = [float("inf"), 0]
            method["keypoints_mask"][3] = False
        metrics, _, _ = compute(refs, preds, choices)
        result = metrics["methods"]["Base"]
        self.assertEqual(result["full_reference_point_count"], 8)
        self.assertEqual(result["valid_matching_prediction_points"], 4)
        self.assertEqual(result["failed_reference_points"], 4)
        self.assertEqual(result["pck10_full_reference_percent"], 50.0)
        self.assertEqual(result["pck10_conditional_percent"], 100.0)

    def test_explicit_missing_does_not_become_valid_finite_placeholder(self):
        method = dict(keypoints=[[0.0, 0.0]] * 8, keypoints_mask=[False] + [True] * 7)
        self.assertEqual(prediction_points(method)[1], [False] + [True] * 7)

    def test_finite_outside_image_is_valid_prediction(self):
        method = dict(keypoints=[[-40, 600]] * 8, keypoints_mask=[True] * 8)
        self.assertEqual(prediction_points(method)[1], [True] * 8)

    def test_unresolved_or_incomplete_match_does_not_drop_frames(self):
        refs, preds, choices = fixture()
        with self.assertRaisesRegex(ValueError, "pending"):
            compute(refs, preds, {"fixture:0": "undetermined"})
        with self.assertRaisesRegex(ValueError, "every fixed"):
            compute(refs, preds, {})

    def test_median_difference_and_median_paired_delta_are_distinct(self):
        points = [dict(Base_error_px=a, N3_error_px=b) for a, b in zip([0, 100, 101], [50, 51, 102])]
        result = paired_summary(points)
        self.assertEqual(result["median_after_minus_median_before_px"], -49)
        self.assertEqual(result["median_of_paired_after_minus_before_px"], 1)

    def test_changed_mask_or_selection_fails_contract(self):
        refs, preds, choices = fixture()
        preds["fixture:0"]["methods"]["N3"]["keypoints_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "missingness"):
            compute(refs, preds, choices)
        refs, preds, choices = fixture()
        preds["fixture:0"]["methods"]["N3"]["selected_index"] = 1
        with self.assertRaisesRegex(ValueError, "selected object"):
            compute(refs, preds, choices)

    def test_reference_coordinate_and_ids_are_not_synthesized(self):
        refs, preds, choices = fixture()
        refs[0]["points"][0]["x"] = None
        with self.assertRaisesRegex(ValueError, "reference"):
            compute(refs, preds, choices)
        refs, preds, choices = fixture()
        refs[0]["points"].reverse()
        with self.assertRaisesRegex(ValueError, "canonical"):
            compute(refs, preds, choices)


if __name__ == "__main__":
    unittest.main()
