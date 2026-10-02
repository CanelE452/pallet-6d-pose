from __future__ import annotations

import inspect
import unittest

from scripts.research.pallet_n3_completion_v3 import lifter


class RecordedArtifactInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = lifter.discover_capture_root()
        cls.inventory = lifter.inventory_captures(cls.root)
        cls.by_id = {row["session_id"]: row
                     for row in cls.inventory["sessions"]}

    def test_exact_frozen_partition_and_all_25_hash_bindings(self):
        self.assertEqual(
            self.inventory["usable_session_ids"],
            ["173507", "174126", "174342", "174925"])
        self.assertEqual(self.inventory["corrupt_session_ids"], ["175419"])
        self.assertEqual(self.inventory["usable_video_frames"], 8910)
        bindings = [artifact for session in self.inventory["sessions"]
                    for artifact in session["files"].values()]
        self.assertEqual(len(bindings), 25)
        self.assertTrue(all(item["exists"] for item in bindings))
        self.assertTrue(all(item["binding_matches"] for item in bindings))

    def test_csv_and_jsonl_schemas_were_read_from_each_bound_file(self):
        for session_id, row in self.by_id.items():
            with self.subTest(session=session_id):
                self.assertEqual(
                    row["schemas"]["state"]["fields"], list(lifter.STATE_FIELDS))
                self.assertEqual(
                    row["schemas"]["timing"]["fields"], list(lifter.TIMING_FIELDS))
                control = row["schemas"]["control"]
                self.assertTrue(
                    lifter.CONTROL_REQUIRED_FIELDS.issubset(control["union_fields"]))
                self.assertIn("can_tx", control["phase_counts"])
                self.assertEqual(control["rows"],
                                 lifter.EXPECTED_FACTS[session_id]["control_rows"])

    def test_four_raw_videos_open_and_175419_is_locked_corrupt(self):
        for session_id in lifter.USABLE_SESSION_IDS:
            video = self.by_id[session_id]["video"]
            self.assertTrue(video["opened"])
            self.assertTrue(video["first_frame_decodes"])
            self.assertTrue(video["last_frame_decodes"])
            self.assertEqual((video["width"], video["height"]), (640, 480))
        corrupt = self.by_id["175419"]
        self.assertEqual(corrupt["status"], "CORRUPT_RAW_MP4")
        self.assertFalse(corrupt["video"]["opened"])
        self.assertEqual(corrupt["video"]["reported_frame_count"], 0)

    def test_sensor_timestamp_contract_and_35_row_startup_alignment(self):
        contract = self.inventory["timebase_contract"]
        self.assertEqual(contract["field"], "camera_sensor_timestamp_ms")
        self.assertFalse(contract["nominal_fps_used"])
        for session_id in lifter.USABLE_SESSION_IDS:
            timebase = self.by_id[session_id]["timebase"]
            self.assertEqual(timebase["field"], lifter.SENSOR_TIME_FIELD)
            self.assertFalse(timebase["nominal_fps_used_for_sampling_or_metrics"])
            self.assertEqual(timebase["nominal_stream_fps"], 30)
            self.assertEqual(timebase["video_timing_row_offset"], 35)
            self.assertEqual(timebase["mapped_first_frame_i"], 36)
            self.assertEqual(
                timebase["mapped_last_frame_i"],
                lifter.EXPECTED_FACTS[session_id]["timing_rows"])

    def test_inventory_marks_accuracy_x_because_no_independent_gt(self):
        reference = self.inventory["reference_contract"]
        self.assertFalse(reference["independent_ground_truth"])
        self.assertEqual(
            reference["accuracy_metrics"], "X_NO_INDEPENDENT_GROUND_TRUTH")
        self.assertEqual(reference["allowed_metrics"],
                         ["coverage", "jitter", "missing"])


class OfflineIteratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = lifter.discover_capture_root()

    def test_first_video_frame_uses_timing_row_36_not_nominal_fps(self):
        stream = lifter.iter_offline_frames(
            "173507", self.root, limit=1, verify=False)
        sample = next(stream)
        stream.close()
        self.assertEqual(sample.video_index, 0)
        self.assertEqual(sample.frame_i, 36)
        self.assertEqual(sample.camera_frame_number, 136)
        self.assertAlmostEqual(
            sample.camera_sensor_timestamp_ms, 1788251713263.987, places=3)
        self.assertEqual(sample.image_bgr.shape, (480, 640, 3))

    def test_time_sampler_uses_irregular_recorded_milliseconds(self):
        due = None
        selected = []
        timestamps = [0.0, 40.0, 120.0, 150.0, 260.0]
        for index, timestamp in enumerate(timestamps):
            use, due = lifter._is_time_sample(timestamp, due, 100.0)
            if use:
                selected.append(index)
        self.assertEqual(selected, [0, 2, 4])

    def test_predictor_callback_receives_read_only_offline_metadata(self):
        calls = []

        def predictor(image, metadata):
            calls.append((image.shape, dict(metadata)))
            return {"available": True, "fresh": True, "pos_z_m": 1.5}

        rows = list(lifter.iter_offline_predictions(
            "174126", predictor, self.root, limit=1, verify=False))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(calls[0][0], (480, 640, 3))
        self.assertEqual(rows[0]["frame_i"], 36)
        self.assertEqual(rows[0]["prediction"]["pos_z_m"], 1.5)

    def test_corrupt_session_cannot_enter_iterator(self):
        stream = lifter.iter_offline_frames("175419", self.root, verify=False)
        with self.assertRaisesRegex(ValueError, "no usable raw video"):
            next(stream)

    def test_module_has_no_vehicle_control_call_surface(self):
        source = inspect.getsource(lifter)
        for forbidden in ("import canlib", "writeSync(", "send_command(",
                          "set_joystick(", "move_fork("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class StabilityStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {lifter.SENSOR_TIME_FIELD: 0.0, "in_view": True,
             "prediction": {"available": True, "fresh": True,
                            "pos_x_m": 0.0, "yaw_deg": 179.0}},
            {lifter.SENSOR_TIME_FIELD: 100.0, "in_view": True,
             "prediction": {"available": True, "fresh": False,
                            "pos_x_m": 0.0, "yaw_deg": 179.0}},
            {lifter.SENSOR_TIME_FIELD: 200.0, "in_view": True,
             "prediction": {"available": True, "fresh": True,
                            "pos_x_m": 1.0, "yaw_deg": -179.0}},
            {lifter.SENSOR_TIME_FIELD: 300.0, "in_view": False,
             "prediction": None},
            {lifter.SENSOR_TIME_FIELD: 400.0, "in_view": True,
             "prediction": {"available": False}},
            {lifter.SENSOR_TIME_FIELD: 500.0, "in_view": True,
             "prediction": {"available": True, "fresh": True,
                            "pos_x_m": 2.0, "yaw_deg": -178.0}},
        ]

    def test_coverage_held_outputs_and_missing_runs_remain_distinct(self):
        result = lifter.stability_statistics(self.records)
        coverage = result["coverage"]
        missing = result["missing"]
        self.assertEqual(coverage["frames"], 6)
        self.assertEqual(coverage["available_outputs"], 4)
        self.assertEqual(coverage["fresh_predictions"], 3)
        self.assertEqual(coverage["held_previous_outputs"], 1)
        self.assertEqual(coverage["visible_missing_outputs"], 1)
        self.assertEqual(coverage["out_of_view_frames"], 1)
        self.assertEqual(missing["frames"], 2)
        self.assertEqual(missing["run_count"], 1)
        self.assertEqual(missing["longest_run_frames"], 2)
        self.assertAlmostEqual(missing["longest_observed_span_s"], .1)
        self.assertAlmostEqual(missing["longest_until_next_sample_s"], .2)

    def test_jitter_uses_sensor_dt_and_wraps_yaw(self):
        result = lifter.stability_statistics(self.records)
        displayed_x = result["jitter"]["pos_x_m"][
            "available_outputs_including_held"]
        fresh_yaw = result["jitter"]["yaw_deg"]["fresh_predictions_only"]
        self.assertEqual(displayed_x["adjacent_positive_dt_pairs"], 2)
        self.assertEqual(displayed_x["median_abs_step"], .5)
        # 179 -> -179 is a two-degree wrapped update over 0.2 seconds.
        self.assertEqual(fresh_yaw["adjacent_positive_dt_pairs"], 1)
        self.assertAlmostEqual(fresh_yaw["median_abs_step"], 2.0)
        self.assertAlmostEqual(fresh_yaw["median_abs_rate_per_s"], 10.0)
        self.assertTrue(fresh_yaw["angular_wrap_applied"])
        self.assertFalse(result["timebase"]["nominal_fps_used"])

    def test_equal_sensor_timestamps_are_reported_and_not_divided(self):
        records = [
            {lifter.SENSOR_TIME_FIELD: 10.0,
             "prediction": {"available": True, "pos_x_m": 0.0}},
            {lifter.SENSOR_TIME_FIELD: 10.0,
             "prediction": {"available": True, "pos_x_m": 2.0}},
        ]
        result = lifter.stability_statistics(records, value_fields=("pos_x_m",))
        jitter = result["jitter"]["pos_x_m"][
            "available_outputs_including_held"]
        self.assertEqual(result["timebase"]["duplicate_adjacent_timestamps"], 1)
        self.assertEqual(jitter["adjacent_zero_dt_pairs_skipped"], 1)
        self.assertIsNone(jitter["median_abs_rate_per_s"])

    def test_accuracy_and_pose_error_requests_are_rejected(self):
        for metric in ("accuracy", "translation_error", "rotation_error", "pck"):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "lack independent ground truth"):
                    lifter.validate_requested_metrics(("coverage", metric))
        result = lifter.stability_statistics(self.records)
        self.assertFalse(result["reference"]["independent_ground_truth"])
        self.assertEqual(
            result["reference"]["accuracy_metrics"],
            "X_NO_INDEPENDENT_GROUND_TRUTH")

    def test_stationary_filter_is_explicit_and_does_not_use_fps(self):
        records = [dict(row, stationary=(index % 2 == 0))
                   for index, row in enumerate(self.records)]
        result = lifter.stability_statistics(records, stationary_key="stationary")
        self.assertEqual(result["scope"], "stationary_only")
        self.assertEqual(result["coverage"]["frames"], 3)
        self.assertFalse(result["timebase"]["nominal_fps_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
