import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("neutral_panel", Path(__file__).with_name("neutral_interval_analysis.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fresh(x=0, z=1, yaw=0):
    return {"pose_state": "fresh", "pose": {"x_m": x, "z_m": z, "yaw_deg": yaw}}


class NeutralPanelContracts(unittest.TestCase):
    def test_360_wrap_removed_but_180_branch_retained(self):
        ordinary = module.pose_stats([fresh(yaw=179), fresh(yaw=-179)])
        branch = module.pose_stats([fresh(yaw=0), fresh(yaw=180)])
        self.assertEqual(ordinary["yaw_population_std_360_unwrapped_deg"], 1)
        self.assertEqual(branch["yaw_population_std_360_unwrapped_deg"], 90)
        self.assertEqual(branch["wrapped_yaw_jump_ge90_count"], 1)

    def test_population_spread_does_not_hide_pose_failures(self):
        stats = module.pose_stats([fresh(x=0), {"pose_state": "no_pose", "pose": None}, fresh(x=2)])
        self.assertEqual(stats["x_population_std_m"], 1)
        self.assertEqual(stats["stored_frame_count"], 3)
        self.assertEqual(stats["valid_output_count"], 2)
        self.assertEqual(stats["no_pose"], 1)

    def test_empty_spread_and_measured_zero_are_distinct(self):
        self.assertIsNone(module.pose_stats([{"pose_state": "no_pose", "pose": None}])["x_population_std_m"])
        self.assertEqual(module.pose_stats([fresh(), fresh()])["x_population_std_m"], 0)

    def test_false_numeric_pose_and_finite_no_pose_rejected(self):
        for method in [fresh(x=True), fresh(yaw=float("nan")), {"pose_state": "no_pose", "pose": fresh()["pose"]}]:
            with self.assertRaises(ValueError): module.pose_stats([method])

    def test_duplicate_sensor_frames_and_excluded_clock_segment(self):
        timeline = [{"frame_id": f"a:{i}", "session_id": "a", "saved_frame_index": str(i),
                     "sensor_clock_segment": str(segment), "camera_sensor_timestamp_ms": "100", "camera_frame_number": "7"}
                    for i, segment in enumerate([0, 0, 1])]
        candidate = {"interval_id": "a_stop", "session_id": "a", "sensor_start_ms": 100, "sensor_end_ms": 100,
                     "sensor_clock_segment": 0, "stored_frame_count": 2, "stored_start_frame_id": "a:0", "stored_end_frame_id": "a:1",
                     "right_censored_command_end": True, "reviewer_id": None, "reviewed_at": None, "fixed_before_predictions": "UNKNOWN"}
        predictions = {row["frame_id"]: {"methods": {name: fresh() for name in ("Base", "N3")}} for row in timeline}
        result = module.select_interval(candidate, timeline, predictions)
        self.assertEqual(result["frame_ids"], ["a:0", "a:1"])
        self.assertEqual(result["duplicate_sensor_timestamp_rows"], 1)
        self.assertEqual(result["relative_stationarity"], "UNKNOWN")
        self.assertFalse(result["legacy_stop_evaluator_usable"])


if __name__ == "__main__":
    unittest.main()
