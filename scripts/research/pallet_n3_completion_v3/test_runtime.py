from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest

import numpy as np

from scripts.research.pallet_n3_completion_v3 import runtime


def fixture_selected():
    return [{
        "frame_id": f"s{index // 2}:f{index}",
        "session_id": f"s{index // 2}",
        "dimensions_wdh_m": [1.1, 1.3, .11],
        "camera_intrinsics": [[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]],
        "image": {"path": f"invented/image{index}.png", "sha256": "0", "bytes": 1},
        "camera_container": {
            "path": f"invented/annotation{index}.json", "sha256": "0", "bytes": 1},
    } for index in range(runtime.FRAMES)]


def fixture_row(job, selected, phase):
    index = job["image_index"]
    output = {"point_sha256": "0", "finite_corners": 8,
              "has_box": True, "pose": None}
    if job["path"].startswith("n3_"):
        output["preservation"] = {
            "center8_preserved": True,
            "missing_point_mask_preserved": True,
            "point_order_preserved": True,
            "selected_instance_box_score_preserved": True,
            "temperature_fixed_before_runtime": True,
            "seed1_final_checkpoint_fixed_before_runtime": True,
        }
    return {
        "phase": phase,
        **job,
        "frame_id": selected[index]["frame_id"],
        "session_id": selected[index]["session_id"],
        "cuda_event_ms": 2. + index / 100.,
        "wall_ms": 2.5 + index / 100.,
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "incremental_peak_allocated_bytes": 100,
        "output": output,
    }


def fixture_raw(backbone="dope"):
    selected = fixture_selected()
    return {
        "schema": runtime.RAW_SCHEMA,
        "complete": True,
        "backbone": backbone,
        "seed": runtime.SEED,
        "config": runtime._timing_config(),
        "hardware": {"name": "Invented RTX CPU-test fixture"},
        "parameters": {
            "base_total": 100,
            "n3_total": 20,
            "base_plus_n3_total": 120,
            "runtime_requires_grad": 0,
        },
        "selected": selected,
        "warmup": [fixture_row(job, selected, "warmup")
                   for job in runtime.warmup_schedule()],
        "measurements": [fixture_row(job, selected, "measured")
                         for job in runtime.schedule()],
        "bindings": [],
        "accuracy_values_read": False,
        "runtime_used_for_selection": False,
        "GT_keypoints_read": False,
        "decoded_images_in_RAM": True,
        "no_fastest_trial_selection": True,
    }


class QuantileTests(unittest.TestCase):
    def test_linear_quantiles_and_fps(self):
        values = [1., 2., 3., 4., 100.]
        result = runtime.describe_ms(values)
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["median_ms"], 3.)
        self.assertAlmostEqual(result["p90_ms"], 61.6)
        self.assertAlmostEqual(result["fps_from_median"], 1000 / 3.)
        self.assertEqual(result["quantile_method"], "numpy linear")

    def test_invalid_latency_is_rejected(self):
        for values in ([], [0.], [-1.], [float("nan")], [[1., 2.]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    runtime.describe_ms(values)


class PreservationTests(unittest.TestCase):
    def test_only_supported_valid_corners_move(self):
        points = np.arange(18, dtype=float).reshape(9, 2)
        points[2] = np.nan
        original = points.copy()
        valid = np.isfinite(points).all(-1)
        delta = np.ones((8, 2))
        support = np.array([True, False, True, True, False, False, True, True])
        refined, audit = runtime.apply_preserving_delta(
            points, valid, delta, support)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(refined[0], original[0] + 1)
        np.testing.assert_array_equal(refined[1], original[1])
        self.assertTrue(np.isnan(refined[2]).all())
        np.testing.assert_array_equal(refined[8], original[8])
        self.assertTrue(all(audit.values()))

    def test_partial_missing_pair_and_nonfinite_supported_delta_are_rejected(self):
        points = np.arange(18, dtype=float).reshape(9, 2)
        valid = np.ones(9, dtype=bool)
        broken = points.copy(); broken[1, 0] = np.nan
        with self.assertRaises(ValueError):
            runtime.apply_preserving_delta(
                broken, valid, np.zeros((8, 2)), np.ones(8, bool))
        delta = np.zeros((8, 2)); delta[0, 0] = np.inf
        with self.assertRaises(ValueError):
            runtime.apply_preserving_delta(points, valid, delta, np.ones(8, bool))


class SchemaTests(unittest.TestCase):
    def test_fixed_schedule_is_balanced(self):
        jobs = runtime.schedule()
        self.assertEqual(len(jobs), 26 * 5 * 3)
        self.assertEqual(Counter(row["path"] for row in jobs),
                         Counter({path: 130 for path in runtime.PATHS}))
        for path in runtime.PATHS:
            positions = Counter(row["path_position"] for row in jobs
                                if row["path"] == path)
            self.assertLessEqual(max(positions.values()) - min(positions.values()), 1)
        warmup = runtime.warmup_schedule()
        self.assertEqual(Counter(row["path"] for row in warmup),
                         Counter({path: 20 for path in runtime.PATHS}))

    def test_raw_and_receipt_schemas_recompute_summary(self):
        raw = fixture_raw()
        validated = runtime.validate_raw_payload(raw, "dope", verify_files=False)
        summary = runtime.summarize(validated["measurements"])
        receipt = {
            "schema": runtime.RECEIPT_SCHEMA,
            "complete": True,
            "backbone": "dope",
            "seed": 1,
            "config": runtime._timing_config(),
            "raw": {"path": "invented", "sha256": "0", "bytes": 1},
            "summary": summary,
            "parameters": raw["parameters"],
            "hardware": raw["hardware"],
            "measured_calls": 390,
            "warmup_calls": 60,
            "bindings": [],
            "same_RTX_required": True,
            "runtime_used_for_selection": False,
            "no_fastest_trial_selection": True,
        }
        runtime.validate_receipt_payload(
            receipt, "dope", raw_payload=raw, verify_files=False)
        damaged = deepcopy(receipt)
        damaged["summary"]["base_e2e"]["cuda_event_ms"]["median_ms"] += .01
        with self.assertRaises(ValueError):
            runtime.validate_receipt_payload(
                damaged, "dope", raw_payload=raw, verify_files=False)

    def test_schema_rejects_selection_and_preservation_drift(self):
        raw = fixture_raw()
        raw["runtime_used_for_selection"] = True
        with self.assertRaises(ValueError):
            runtime.validate_raw_payload(raw, "dope", verify_files=False)
        raw = fixture_raw()
        n3 = next(row for row in raw["measurements"]
                  if row["path"] == "n3_seed1_e2e")
        n3["output"]["preservation"]["center8_preserved"] = False
        with self.assertRaises(ValueError):
            runtime.validate_raw_payload(raw, "dope", verify_files=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
