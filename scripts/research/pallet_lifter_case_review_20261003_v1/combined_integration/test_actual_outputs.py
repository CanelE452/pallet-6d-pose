from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import build_actual_outputs as subject


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "output"


def method(state, x=None, z=None, yaw=None):
    return {
        "pose_state": state,
        "pose": {"x_m": x, "z_m": z, "yaw_deg": yaw},
        "detection_present": state != "no_pose",
        "pnp_failed": False,
        "keypoints_mask": [True] * 8 if state != "no_pose" else [],
        "object_match": None,
    }


def row(index, timestamp, b, n):
    return {
        "frame_id": f"S:{index}", "session_id": "S", "stored_index": index,
        "sensor_timestamp_ms": timestamp, "methods": {"Base": b, "N3": n},
    }


class SummaryUnitTests(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self):
        self.assertEqual(subject.percentile([0, 10], .9), 9.0)
        self.assertEqual(subject.percentile([1, 2, 3], .5), 2.0)
        self.assertIsNone(subject.percentile([], .5))

    def test_gap_censoring_and_paired_denominators(self):
        rows = [
            row(0, 0, method("fresh", 0, 1, 0), method("fresh", 0, 1, 0)),
            row(1, 100, method("no_pose"), method("fresh", 1, 1, 1)),
            row(2, 200, method("no_pose"), method("no_pose")),
            row(3, 400, method("fresh", 3, 2, 3), method("fresh", 3, 2, 3)),
            row(4, 500, method("fresh", 4, 3, 4), method("no_pose")),
        ]
        value, changes = subject.summarize_scope(rows, "S")
        bgap = value["methods"]["Base"]["no_pose_gaps"]
        ngap = value["methods"]["N3"]["no_pose_gaps"]
        self.assertEqual(bgap["gap_count"], 1)
        self.assertAlmostEqual(bgap["maximum_completed_duration_s"], .3)
        self.assertEqual(bgap["maximum_missing_stored_frame_count"], 2)
        self.assertEqual(ngap["gap_count"], 2)
        self.assertEqual(ngap["right_censored_gap_count"], 1)
        self.assertEqual(value["pairing"]["same_frame_both_finite_pose_count"], 2)
        self.assertEqual(value["pairing"]["same_frame_base_only_finite_pose_count"], 1)
        self.assertEqual(value["pairing"]["same_frame_n3_only_finite_pose_count"], 1)
        self.assertEqual(len(changes["Base"]), 1)
        # N3 has one valid adjacent pair at frames 0->1; its later finite
        # outputs are separated by no_pose states.
        self.assertEqual(len(changes["N3"]), 1)


@unittest.skipUnless((OUTPUT / "LIFTER_CONTINUITY_SUMMARY.json").is_file(), "actual derived outputs not built")
class ActualArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((OUTPUT / "LIFTER_CONTINUITY_SUMMARY.json").read_text())
        cls.receipt = json.loads((OUTPUT / "EXECUTION_FILE_RECEIPT.json").read_text())

    def test_actual_denominators(self):
        self.assertEqual(self.summary["overall"]["stored_frame_count"], 8910)
        for method in subject.METHODS:
            value = self.summary["overall"]["methods"][method]
            self.assertEqual((value["fresh_count"], value["held_count"], value["no_pose_count"]), (8772, 0, 138))
        paired = self.summary["overall"]["pairing"]
        self.assertEqual(paired["same_frame_both_finite_pose_count"], 8772)
        self.assertEqual(paired["both_methods_valid_same_adjacent_pair_count"], 8737)

    def test_regression_and_unresolved_references(self):
        check = self.summary["evaluator_regression_check"]
        self.assertEqual(check["status"], "VERIFIED_COMPLETE")
        self.assertTrue(check["all_passed"])
        self.assertGreater(check["check_count"], 100)
        self.assertTrue(all(value is None for value in self.summary["unresolved_reference_fields"].values()))

    def test_all_120_overlay_and_figures(self):
        self.assertEqual(self.summary["review120_prediction_overlay"]["frame_count"], 120)
        for name in ("lifter_prediction_timeseries.png", "lifter_output_coverage.png", "review120_prediction_overlay_contact_sheet.png"):
            path = OUTPUT / "figures" / name
            self.assertGreater(path.stat().st_size, 10_000)
            with Image.open(path) as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 500)

    def test_receipt_hashes_and_source_immutability(self):
        self.assertTrue(self.receipt["frozen_source_inputs_unchanged_during_build"])
        for section in ("source_inputs",):
            for record in self.receipt[section].values():
                path = ROOT / record["path"]
                self.assertEqual(path.stat().st_size, record["bytes"])
                self.assertEqual(subject.sha256(path), record["sha256"])
        for record in self.receipt["derived_outputs"]:
            path = ROOT / record["path"]
            self.assertEqual(subject.sha256(path), record["sha256"])
        receipt_path = OUTPUT / "EXECUTION_FILE_RECEIPT.json"
        expected = (OUTPUT / "EXECUTION_FILE_RECEIPT.sha256").read_text().split()[0]
        self.assertEqual(subject.sha256(receipt_path), expected)

    def test_report_preserves_claim_limits(self):
        report = (OUTPUT / "ACTUAL_INFERENCE_REPORT_KO.md").read_text()
        for phrase in ("정확도나 실제 정지 상태의 잡음을 뜻하지 않는다", "사람 visible-corner 정확도", "독립 물리 위치·방향 정확도"):
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
