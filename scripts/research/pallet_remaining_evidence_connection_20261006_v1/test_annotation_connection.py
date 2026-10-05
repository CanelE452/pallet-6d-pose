import copy
import unittest

from annotation_connection import join_tasks, source_points


class AnnotationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.frame = {"frame_id": "a:7", "session_id": "a", "saved_frame_index": 7,
                      "image_sha256": "encoded", "decoded_bgr_sha256": "pixels",
                      "raw_video_sha256": "video", "camera_sensor_timestamp_ms": 100.0,
                      "width": 640, "height": 480, "repeat_review": True}
        self.reference = copy.deepcopy(self.frame)

    def test_same_filename_without_exact_image_is_not_reusable(self):
        self.reference["image_sha256"] = "different_image"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            join_tasks([self.frame], {"a:7": self.reference}, set(), set())

    def test_same_pixels_from_different_frame_number_is_not_reusable(self):
        self.reference["saved_frame_index"] = 8
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            join_tasks([self.frame], {"a:7": self.reference}, set(), set())

    def test_existing_primary_does_not_fabricate_repeat(self):
        rows = join_tasks([self.frame], {"a:7": self.reference}, set(), set())
        self.assertEqual(rows[0]["task_status"], "REUSE_COMPLETED_EXPLORATORY_GEOMETRY")
        self.assertEqual(rows[1]["task_status"], "PENDING_REAL_REPEAT_RECORD")
        self.assertFalse(rows[1]["actual_repeat_record_exists"])
        self.assertTrue(rows[1]["completed_primary_image_reference_exists"])
        self.assertFalse(rows[0]["approved_official_direct_visible_reference"])

    def test_exclusion_remains_in_both_fixed_task_denominators(self):
        rows = join_tasks([self.frame], {}, {"a:7"}, set())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["task_status"] == "EXCLUDED_BY_USER" for row in rows))

    def test_excluded_frame_reintroduction_rejected(self):
        with self.assertRaisesRegex(ValueError, "reintroduced"):
            join_tasks([self.frame], {"a:7": self.reference}, {"a:7"}, set())

    def test_logical_candidate_is_not_approved_reference(self):
        rows = join_tasks([self.frame], {}, set(), {"a:7"})
        self.assertTrue(rows[0]["legacy_logical_candidate"])
        self.assertEqual(rows[0]["legacy_pixel_mapping_status"], "UNKNOWN_PIXEL_UNVERIFIED")
        self.assertFalse(rows[0]["approved_official_direct_visible_reference"])
        self.assertEqual(rows[0]["task_status"], "PENDING_REFERENCE")

    def test_manual_kps_does_not_turn_legacy_pnp_into_click_history(self):
        payload = {"objects": [{"manual_kps": [[1, 2], [3, 4]], "gt_source": "manual",
                                "keypoint_annotations": [{"source": "pnp_projected"}, {}]}]}
        counts = source_points(payload)
        self.assertEqual(counts["manual_click"], 0)
        self.assertEqual(counts["pnp_projected"], 1)
        self.assertEqual(counts["UNKNOWN"], 1)


if __name__ == "__main__":
    unittest.main()
