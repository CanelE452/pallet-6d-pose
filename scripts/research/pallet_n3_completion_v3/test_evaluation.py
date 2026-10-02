from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from scripts.research.pallet_n3_completion_v3 import evaluation


GT = np.array([
    [10., 10.], [30., 10.], [30., 30.], [10., 30.],
    [12., 12.], [28., 12.], [28., 28.], [12., 28.], [20., 20.],
])


def truth(count=2):
    rows = []
    for index in range(count):
        rows.append({
            "id": f"f{index}", "session": f"s{index}", "hw": [100, 100],
            "gt": GT.tolist(), "valid": [True] * 9,
            "box": [10., 10., 30., 30.],
            "permutations": [list(range(9))],
            "material": "plastic" if index == 0 else "wood",
            "occlusion": "none" if index == 0 else "unclassified",
        })
    return rows


def points(offset):
    value = GT.copy()
    value[:8] += offset
    return value.tolist()


def block(offset, *, bbox=None, detected=True):
    return {
        "detected": detected,
        "status": "OK" if detected else "NO_BOX",
        "points": points(offset) if detected else None,
        "valid": [detected] * 8 + [True],
        "bbox": [10., 10., 30., 30.] if bbox is None else bbox,
        "score": .9,
    }


def combined_payload(count=2):
    frames = []
    for index in range(count):
        frames.append({
            "id": f"f{index}", "session_id": f"s{index}",
            "raw_shape_hw": [100, 100],
            # This is deliberately wrong.  Evaluation must join caller truth.
            "gt": np.zeros((9, 2)).tolist(),
            "predictions": {
                "base": block(4.),
                "N3_seed1": block(2.),
                "N3_seed2": block(4.),
                "N3_seed3": block(6.),
            },
        })
    return {
        "schema": "pallet_n3_completion_v3_prediction_v1",
        "complete": True, "backbone": "dope",
        "coordinate_system": "original_unpadded_pixels",
        "methods": ["base", "N3_seed1", "N3_seed2", "N3_seed3"],
        "frames": frames,
    }


class CombinedSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluation.evaluate_payloads(
            [(None, combined_payload())], truth(), backbone="dope",
            include_pose=False, expected_frames=None)

    def test_all_seeds_and_exact_corner_contract_are_evaluated(self):
        result = self.result
        self.assertTrue(result["complete"])
        self.assertEqual(set(result["methods"]), set(evaluation.METHODS))
        base = result["methods"]["base"]["headline"]
        seed1 = result["methods"]["n3_seed1"]["headline"]
        self.assertAlmostEqual(base["matched_pooled_corner8_median_px"],
                               np.sqrt(32.))
        self.assertAlmostEqual(seed1["matched_pooled_corner8_median_px"],
                               np.sqrt(8.))
        self.assertEqual(base["full_supervised_corners"], 16)
        self.assertEqual(base["observed_corners"], 16)
        self.assertEqual(result["population"]["full_supervised_corners"], 16)
        self.assertEqual(result["population"]["sessions"], 2)
        self.assertEqual(
            result["population"]["subgroups"]["material"]["plastic"],
            {"frames": 1, "sessions": 1, "full_supervised_corners": 8})
        self.assertEqual(result["pose_status"], "DISABLED")

    def test_improvement_no_change_and_worsening_remain_visible(self):
        expected = {
            "n3_seed1": (2, 0, 0),
            "n3_seed2": (0, 2, 0),
            "n3_seed3": (0, 0, 2),
        }
        for method, counts in expected.items():
            change = self.result["comparisons"][f"base_to_{method}"][
                "result"]["frame_change"]
            self.assertEqual(
                (change["improved_frames"], change["no_change_frames"],
                 change["worsened_frames"]), counts)
            self.assertEqual(change["comparable_frames"], 2)

    def test_bootstrap_and_seed_aggregate_are_locked(self):
        paired = self.result["comparisons"]["base_to_n3_seed1"]["result"]
        bootstrap = paired["bootstrap"]
        self.assertEqual(bootstrap["resamples"], 10_000)
        self.assertEqual(bootstrap["seed"], 20260917)
        self.assertEqual(bootstrap["unit"], "session")
        self.assertTrue(bootstrap["paired_sampling"])
        self.assertEqual(self.result["seed_aggregate"]["status"], "COMPLETE")

    def test_raw_gt_is_ignored_and_subgroups_are_explicit(self):
        method = self.result["methods"]["base"]["result"]
        self.assertGreater(method["corner"]["metrics"]["E_sym"], 0.)
        self.assertEqual(set(method["subgroups"]["material"]),
                         {"plastic", "wood"})
        self.assertEqual(set(method["subgroups"]["occlusion"]),
                         {"none", "unclassified"})


class MissingAndIntegrityTests(unittest.TestCase):
    def test_legacy_p_and_d_are_not_renamed_to_n3(self):
        payload = {
            "schema": "dope_refiner_dev_predictions_v1", "complete": True,
            "coordinate_system": "original_unpadded_pixels",
            "frames": [{
                "frame_id": "f0", "session_id": "s0",
                "original_hw": [100, 100], "base_points": points(4.),
                "point_valid": [True] * 9,
                "box_original": [10., 10., 30., 30.], "score": .9,
                "refined": {"P1": points(1.), "D1": points(2.)},
            }],
        }
        result = evaluation.evaluate_payloads(
            [(None, payload)], truth(1), include_pose=False,
            expected_frames=None)
        self.assertEqual(result["methods"]["base"]["status"], "COMPLETE")
        for method in evaluation.METHODS[1:]:
            self.assertEqual(result["methods"][method]["status"], "MISSING")
            self.assertIsNone(result["methods"][method]["result"])
            self.assertEqual(result["methods"][method]["display"], "x")
        self.assertEqual(result["seed_aggregate"]["display"], "x")

    def test_changed_box_blocks_n3_before_scoring(self):
        payload = combined_payload(1)
        payload["frames"][0]["predictions"]["N3_seed1"]["bbox"] = [9., 10., 30., 30.]
        result = evaluation.evaluate_payloads(
            [(None, payload)], truth(1), include_pose=False,
            expected_frames=None)
        entry = result["methods"]["n3_seed1"]
        self.assertEqual(entry["status"], "BLOCKED_PRESERVATION")
        self.assertIsNone(entry["result"])
        self.assertIn("bbox changed", entry["reason"]["examples"][0]["error"])

    def test_incomplete_identity_and_wrong_coordinates_stay_x(self):
        payload = combined_payload(2)
        del payload["frames"][1]["predictions"]["N3_seed2"]
        result = evaluation.evaluate_payloads(
            [(None, payload)], truth(2), include_pose=False,
            expected_frames=None)
        self.assertEqual(result["methods"]["n3_seed2"]["status"],
                         "BLOCKED_IDENTITY")
        self.assertEqual(result["methods"]["n3_seed2"]["display"], "x")

        wrong = combined_payload(1)
        wrong["coordinate_system"] = "network_pixels"
        blocked = evaluation.evaluate_payloads(
            [(None, wrong)], truth(1), include_pose=False,
            expected_frames=None)
        self.assertEqual(blocked["methods"]["base"]["status"],
                         "BLOCKED_COORDINATE_SYSTEM")

        bad_hw = combined_payload(1)
        bad_hw["frames"][0]["raw_shape_hw"] = [99, 100]
        blocked = evaluation.evaluate_payloads(
            [(None, bad_hw)], truth(1), include_pose=False,
            expected_frames=None)
        self.assertEqual(blocked["methods"]["base"]["status"],
                         "BLOCKED_METADATA")

    def test_method_specific_candidate_schema_is_supported(self):
        payload = {
            "complete": True,
            "records": [{
                "id": "f0", "selected_index": 0,
                "candidates": [{
                    "keypoints_xy": points(2.),
                    "box_xyxy": [10., 10., 30., 30.], "score": .9,
                }],
            }],
        }
        normalized = evaluation.normalize_prediction_payload(
            payload, method_hint="n3_seed1")
        self.assertEqual(set(normalized["methods"]), {"n3_seed1"})
        self.assertEqual(set(normalized["methods"]["n3_seed1"]), {"f0"})


class FrozenTruthCompatibilityTests(unittest.TestCase):
    def test_dev319_loads_without_images_and_keeps_denominators(self):
        rows = evaluation.load_dev319_truth(include_pose=True)
        self.assertEqual(len(rows), 319)
        self.assertEqual(len({row["id"] for row in rows}), 319)
        self.assertEqual(sum(sum(row["valid"][:8]) for row in rows), 2499)
        self.assertEqual(Counter(row["material"] for row in rows),
                         Counter(plastic=194, wood=125))
        self.assertEqual(Counter(row["occlusion"] for row in rows),
                         Counter(unclassified=319))
        self.assertEqual(sum("pose" in row for row in rows), 319)


if __name__ == "__main__":
    unittest.main(verbosity=2)
