"""Safety and evaluation-policy regressions for publication verification."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from verify_published_evidence import Evidence, recompute, slots, summarize


def fixture():
    record = dict(frame_id="sample:0", session_id="sample",
                  points=[dict(id=i, x=0., y=0., source="manual_click") for i in range(8)])
    prediction = dict(methods={method: dict(selected_index=0,
                                            keypoints=[[3., 4.]] * 8,
                                            keypoints_mask=[True] * 8) for method in ("Base", "N3")})
    return [record], {"sample:0": prediction}, {"sample:0": "same"}


class ContractTests(unittest.TestCase):
    def test_wrong_selected_object_fails_all_slots_without_shrinking_denominator(self):
        records, predictions, decisions = fixture()
        decisions["sample:0"] = "different"
        metrics, _, _ = recompute(records, predictions, decisions)
        for method in ("Base", "N3"):
            result = metrics["methods"][method]
            self.assertEqual(result["full_reference_point_count"], 8)
            self.assertEqual(result["failed_reference_points"], 8)
            self.assertEqual(result["pck10_full_reference_percent"], 0)
            self.assertIsNone(result["conditional_error"]["median_px"])

    def test_unknown_correspondence_cannot_be_discarded_or_called_wrong(self):
        records, predictions, decisions = fixture()
        decisions["sample:0"] = "undetermined"
        with self.assertRaisesRegex(ValueError, "pending"):
            recompute(records, predictions, decisions)

    def test_explicit_missingness_and_inclusive_threshold_preserve_full_population(self):
        records, predictions, decisions = fixture()
        for method in ("Base", "N3"):
            predictions["sample:0"]["methods"][method]["keypoints"][0] = [6., 8.]
            predictions["sample:0"]["methods"][method]["keypoints_mask"][1] = False
        metrics, _, _ = recompute(records, predictions, decisions)
        for method in ("Base", "N3"):
            result = metrics["methods"][method]
            self.assertEqual(result["pck10_hit_count"], 7)
            self.assertEqual(result["full_reference_point_count"], 8)
            self.assertEqual(result["failed_reference_points"], 1)
            self.assertEqual(result["pck10_full_reference_percent"], 87.5)

    def test_finite_off_image_points_remain_valid_and_nonboolean_mask_rejected(self):
        method = dict(keypoints=[[-100., 800.]] * 8, keypoints_mask=[True] * 8)
        self.assertTrue(all(slots(method)[1]))
        method["keypoints_mask"][0] = 1
        with self.assertRaisesRegex(ValueError, "mask"):
            slots(method)

    def test_median_difference_does_not_replace_median_paired_difference(self):
        rows = [dict(Base_error_px=a, N3_error_px=b,
                     Base_pck10_hit=a <= 10, N3_pck10_hit=b <= 10,
                     Base_failure_reason=None, N3_failure_reason=None)
                for a, b in zip([0., 100., 101.], [99., 100., 200.])]
        result = summarize(rows)["paired"]
        self.assertEqual(result["median_after_minus_median_before_px"], 0.)
        self.assertEqual(result["median_of_paired_after_minus_before_px"], 99.)

    def test_absolute_source_is_resolved_to_lossless_copy_in_fresh_clone(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "evidence.json.gz"
            with gzip.open(artifact, "wb") as handle:
                handle.write(b'{"value": 7}')
            manifest = dict(source_repository_prefix="/unavailable/original/repo",
                            compressed_sources=[dict(original_path="data/evidence.json",
                                                     artifact_path="evidence.json.gz", encoding="gzip")],
                            reference_images=[])
            evidence = Evidence(folder, manifest)
            self.assertEqual(evidence.json("/unavailable/original/repo/data/evidence.json"), {"value": 7})
            with self.assertRaises(ValueError):
                evidence.relative("/unrelated/secret.json")
            with self.assertRaises(ValueError):
                evidence.relative("../outside.json")


if __name__ == "__main__":
    unittest.main()
