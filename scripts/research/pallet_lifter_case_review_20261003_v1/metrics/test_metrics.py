"""Synthetic unit fixtures only: these never become human reference artifacts."""
import copy
import unittest

from evaluate import (ContractError, IDENTITY, evaluate_continuity,
                      evaluate_repeat_quality, evaluate_stop_intervals,
                      evaluate_visible_corners, index_predictions, wrap_delta)
from evaluate import validate_run_identity, frozen_legacy_bindings


BINDINGS = {"manifest_sha256": "manifest_test_only", "plan_sha256": "plan_test_only",
            "corner_contract_sha256": "contract_test_only", "corner_definition_version": "test_only_v1"}


def frame(session="s1", index=0, timestamp=None, camera_number=None):
    return {"frame_id": f"{session}:{index}", "session_id": session, "saved_frame_index": index,
            "camera_sensor_timestamp_ms": index*1000 if timestamp is None else timestamp,
            "camera_frame_number": index if camera_number is None else camera_number,
            "camera_timestamp_domain": "sensor_test", "width": 640, "height": 480,
            "image_sha256": "image_test_only", "decoded_bgr_sha256": "pixels_test_only", "repeat_review": True}


def method(state="fresh", yaw=0, points=None, mask=None):
    return {"pose_state": state, "pose": {"x_m": 0, "z_m": 3, "yaw_deg": yaw} if state != "no_pose" else None,
            "keypoints": points if points is not None else [[10+i*15, 20+i*5] for i in range(8)],
            "keypoints_mask": mask if mask is not None else [True]*8,
            "object_id": "test_target", "detection_present": True, "pnp_failed": state == "no_pose"}


def prediction(f, states=("fresh", "fresh"), yaw=0, execution_id=None):
    return {"frame_id": f["frame_id"], "session_id": f["session_id"], "stored_index": f["saved_frame_index"],
            "sensor_timestamp_ms": f["camera_sensor_timestamp_ms"], "camera_frame_number": f["camera_frame_number"],
            "decoded_bgr_sha256": f["decoded_bgr_sha256"], "inference_execution_id": execution_id or f["frame_id"],
            "methods": {"Base": method(states[0], yaw), "N3": method(states[1], yaw)}}


def reference(f, direct=(0, 1), review_pass="primary", reviewer="test_reviewer"):
    corners = []
    for i in range(8):
        visible = i in direct
        corners.append({"id": i, "visibility": "direct_visible" if visible else "not_direct_visible",
                        "external_occlusion": False, "self_occlusion": not visible, "out_of_frame": False,
                        "definition_uncertain": False, "definition_confirmed": True,
                        "x": 10+i*15 if visible else None, "y": 20+i*5 if visible else None,
                        "reason": "test_fixture"})
    return {"frame_id": f["frame_id"], "session_id": f["session_id"], "review_pass": review_pass,
            "status": "reviewed", "source_kind": "human_reviewed", "image_sha256": f["image_sha256"],
            "width": 640, "height": 480, **{k: BINDINGS[k] for k in ("plan_sha256", "corner_contract_sha256", "corner_definition_version")},
            "object": {"presence": "present", "target_object_id": "test_target", "target_identity_confirmed": True},
            "corners": corners, "reviewer": {"id": reviewer, "entered_by": "human", "confirmation": True,
                "machine_assistance": False, "previous_prediction_exposure": False, "previous_annotation_exposure": False},
            "review_time": {"started_at": "2000-01-01T00:00:00Z", "finished_at": "2000-01-01T00:00:01Z",
                            "duration_seconds": 1, "clock_source": "server_wall_and_monotonic"}}


def bundle(*records):
    return {"schema_version": "lifter_reference_review_v1", "source_kind": "human_reviewed",
            "bindings": BINDINGS.copy(), "records": list(records)}


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        f = frame()
        result = evaluate_visible_corners([f], bundle(reference(f)), {f["frame_id"]: prediction(f)}, BINDINGS)
        self.assertEqual(result["methods"]["Base"]["pck10_pct"], 100)
        self.assertEqual(result["methods"]["Base"]["conditional_error_median_px"], 0)
        self.assertEqual(result["paired_difference"]["median_n3_minus_base_same_corner_px"], 0)

    def test_missing_prediction_keeps_visible_denominator(self):
        f = frame()
        pred = prediction(f)
        pred["methods"]["Base"]["keypoints"] = None
        result = evaluate_visible_corners([f], bundle(reference(f)), {f["frame_id"]: pred}, BINDINGS)
        stats = result["methods"]["Base"]
        self.assertEqual((stats["visible_reference_corner_count"], stats["valid_predicted_corner_count"], stats["pck10_pct"]), (2, 0, 0))
        self.assertIsNone(stats["conditional_error_median_px"])

    def test_wrong_object_is_failure_and_unknown_is_blocked(self):
        f = frame()
        pred = prediction(f)
        pred["methods"]["Base"]["object_id"] = "other_target"
        result = evaluate_visible_corners([f], bundle(reference(f)), {f["frame_id"]: pred}, BINDINGS)
        self.assertEqual(result["methods"]["Base"]["failure_counts"], {"wrong_object": 2})
        pred["methods"]["Base"]["object_id"] = None
        with self.assertRaises(ContractError):
            evaluate_visible_corners([f], bundle(reference(f)), {f["frame_id"]: pred}, BINDINGS)

    def test_no_visible_corner_needs_no_correspondence(self):
        f = frame()
        pred = prediction(f)
        for p in pred["methods"].values():
            p["object_id"] = None
        result = evaluate_visible_corners([f], bundle(reference(f, direct=())), {f["frame_id"]: pred}, BINDINGS)
        self.assertEqual(result["methods"]["Base"]["visible_reference_corner_count"], 0)
        self.assertIsNone(result["methods"]["Base"]["pck10_pct"])
        self.assertEqual(result["status"], "BLOCKED_REFERENCE")

    def test_reference_integrity_blocks_unreviewed_hash_duplicate_range(self):
        f = frame()
        ref = reference(f)
        pred = {f["frame_id"]: prediction(f)}
        for mutate in (lambda r: r.update(source_kind="machine_proposed"),
                       lambda r: r.update(image_sha256="other"),
                       lambda r: r["corners"][0].update(x=None),
                       lambda r: r["corners"][0].update(x=640),
                       lambda r: r["corners"][1].update(id=0)):
            bad = copy.deepcopy(ref)
            mutate(bad)
            with self.assertRaises(ContractError):
                evaluate_visible_corners([f], bundle(bad), pred, BINDINGS)

    def test_whole_permutation_moves_coordinates_mask_and_ids_together(self):
        f = frame()
        pred = prediction(f)
        swap = (1, 0, 2, 3, 4, 5, 6, 7)
        for p in pred["methods"].values():
            p["keypoints"][0], p["keypoints"][1] = p["keypoints"][1], p["keypoints"][0]
            p["keypoints_mask"][0] = False
        result = evaluate_visible_corners([f], bundle(reference(f)), {f["frame_id"]: pred}, BINDINGS,
            mode="frozen_whole_permutation", allowed_permutations=[IDENTITY, swap])
        self.assertEqual(result["methods"]["Base"]["pck10_pct"], 50)
        rows = [r for r in result["corner_rows"] if r["method"] == "Base"]
        self.assertEqual([r["prediction_source_corner_id"] for r in rows], [1, 0])
        self.assertEqual([r["error_px"] for r in rows], [0, None])
        with self.assertRaises(ContractError):
            evaluate_visible_corners([f], bundle(reference(f)), {f["frame_id"]: pred}, BINDINGS,
                mode="canonical", allowed_permutations=[swap])

    def test_session_boundary_and_ending_gap_censoring(self):
        frames = [frame("s1", i) for i in range(3)] + [frame("s2", i) for i in range(2)]
        rows = [prediction(f, states=("no_pose", "no_pose") if f["saved_frame_index"] else ("fresh", "fresh")) for f in frames]
        result = evaluate_continuity(frames, index_predictions(rows, frames))
        first = result["sessions"]["s1"]["methods"]["Base"]["nonfresh_gaps"][0]
        self.assertTrue(first["right_censored"])
        self.assertIsNone(first["duration_s"])
        self.assertEqual(first["observed_missing_span_s"], 1)
        self.assertEqual(result["sessions"]["s2"]["methods"]["Base"]["nonfresh_gaps"][0]["missing_stored_frame_count"], 1)

    def test_duplicates_distinguish_camera_observation_and_execution(self):
        frames = [frame(index=0, timestamp=10, camera_number=20), frame(index=1, timestamp=10, camera_number=20), frame(index=2, timestamp=40, camera_number=23)]
        result = evaluate_continuity(frames, index_predictions([prediction(f) for f in frames], frames))["sessions"]["s1"]
        self.assertEqual(result["stored_frame_count"], 3)
        self.assertEqual(result["distinct_camera_observation_count"], 2)
        self.assertEqual(result["distinct_inference_execution_count"], 3)
        self.assertEqual(result["duplicate_sensor_timestamp_stored_count"], 1)
        self.assertEqual(result["camera_recording_gaps"][0]["unrecorded_camera_number_count"], 2)

    def test_wrap_preserves_branch_changes_and_held_is_not_fresh(self):
        frames = [frame(index=i) for i in range(4)]
        rows = [prediction(f, states=("held", "held") if i == 1 else ("fresh", "fresh"), yaw=y) for i, (f, y) in enumerate(zip(frames, (359, 1, 91, 271)))]
        stats = evaluate_continuity(frames, index_predictions(rows, frames))["sessions"]["s1"]["methods"]["Base"]
        self.assertEqual(stats["fresh_output_rate_pct"], 75)
        self.assertEqual([c["yaw_delta_360_wrapped_deg"] for c in stats["adjacent_changes"]], [2, 90, -180])
        self.assertEqual(stats["adjacent_output_change"]["wrapped_abs_ge90_deg_count"], 2)
        self.assertEqual(wrap_delta(-179, 179), 2)

    def test_all_missing_and_internal_failure_gap(self):
        frames = [frame(index=i) for i in range(3)]
        missing = [prediction(f, states=("no_pose", "no_pose")) for f in frames]
        stats = evaluate_continuity(frames, index_predictions(missing, frames))["sessions"]["s1"]["methods"]["Base"]
        self.assertEqual(stats["fresh_output_rate_pct"], 0)
        self.assertTrue(stats["nonfresh_gaps"][0]["left_censored"])
        self.assertTrue(stats["nonfresh_gaps"][0]["right_censored"])
        rows = [prediction(f, states=("no_pose", "no_pose") if i == 1 else ("fresh", "fresh")) for i, f in enumerate(frames)]
        gap = evaluate_continuity(frames, index_predictions(rows, frames))["sessions"]["s1"]["methods"]["Base"]["nonfresh_gaps"][0]
        self.assertEqual(gap["duration_s"], 1)
        self.assertFalse(gap["right_censored"])

    def test_invalid_state_and_clock_regression_are_not_fresh(self):
        f = frame()
        pred = prediction(f)
        pred["methods"]["Base"].pop("pose_state")
        with self.assertRaises(ContractError):
            evaluate_continuity([f], index_predictions([pred], [f]))
        frames = [frame(index=0, timestamp=5), frame(index=1, timestamp=1)]
        with self.assertRaises(ContractError):
            evaluate_continuity(frames, index_predictions([prediction(f) for f in frames], frames))

    def test_repeat_distinguishes_same_person_and_status_mismatch(self):
        f = frame()
        first, second = reference(f), reference(f, review_pass="repeat")
        second["corners"][0]["x"] += 3
        result = evaluate_repeat_quality([f], bundle(first, second), BINDINGS)
        self.assertEqual(result["reviewer_relation_counts"], {"same_person_repeat": 1})
        self.assertEqual(result["repeated_direct_visible_corner_click_difference_px"]["median"], 1.5)

    def test_stop_variation_requires_review_and_does_not_merge_intervals(self):
        frames = [frame(index=i) for i in range(2)]
        predictions = index_predictions([prediction(frames[0], yaw=179), prediction(frames[1], yaw=-179)], frames)
        self.assertIsNone(evaluate_stop_intervals(frames, predictions, None)["intervals"])
        with self.assertRaises(ContractError):
            evaluate_stop_intervals(frames, predictions, {"source_kind": "machine_proposed"})
        reviewed = {"source_kind": "human_reviewed", "fixed_before_predictions": True, "plan_sha256": BINDINGS["plan_sha256"], "intervals": [
            {"interval_id": "test_stop", "session_id": "s1", "sensor_start_ms": 0, "sensor_end_ms": 1000,
             "source_kind": "human_reviewed", "reviewer_id": "test_reviewer", "reviewed_at": "2000-01-01T00:00:01Z", "evidence": "synthetic_fixture_only"}]}
        result = evaluate_stop_intervals(frames, predictions, reviewed, BINDINGS)
        self.assertEqual(result["intervals"][0]["methods"]["Base"]["yaw_std_360_unwrapped_deg"], 1)

    def test_duplicate_number_with_changed_time_is_not_new_camera_observation(self):
        frames = [frame(index=0, timestamp=10, camera_number=20), frame(index=1, timestamp=11, camera_number=20)]
        result = evaluate_continuity(frames, index_predictions([prediction(f) for f in frames], frames))["sessions"]["s1"]
        self.assertEqual(result["distinct_camera_observation_count"], 1)
        self.assertEqual(result["distinct_inference_execution_count"], 2)
        self.assertEqual(result["camera_frame_number_with_conflicting_timestamp_count"], 1)

    def test_paired_delta_differs_from_difference_of_medians(self):
        f = frame()
        pred = prediction(f)
        for method_name, errors in (("Base", (0, 1, 100)), ("N3", (2, 4, 101))):
            for i, error in enumerate(errors):
                pred["methods"][method_name]["keypoints"][i][0] += error
        result = evaluate_visible_corners([f], bundle(reference(f, direct=(0, 1, 2))), {f["frame_id"]: pred}, BINDINGS)
        self.assertEqual(result["paired_difference"]["median_n3_minus_base_same_corner_px"], 2)
        self.assertEqual(result["paired_difference"]["median_n3_minus_median_base_px"], 3)

    def test_actual_cli_identity_rejects_other_weight_plan_and_guard(self):
        expected = [{"name": "test_weight", "path": "test/weight.pt", "sha256": "test_only", "bytes": 5}]
        identity = {"plan_sha256": "plan_test", "source_commit": "commit_test", "fixed_bindings": expected,
                    "pass_budget_s": 3600, "independent_reference": False}
        validate_run_identity(identity, "plan_test", "commit_test", expected)
        for change in ({"plan_sha256": "other_plan"}, {"fixed_bindings": []}, {"independent_reference": True}):
            with self.assertRaises(ContractError):
                validate_run_identity({**identity, **change}, "plan_test", "commit_test", expected)
        # Real immutable declaration is read as constants, without importing model/control code.
        bindings = frozen_legacy_bindings()
        self.assertEqual(len(bindings), 15)
        self.assertEqual({b["name"] for b in bindings if b["name"] in ("yolo_r0", "n3_checkpoint")}, {"yolo_r0", "n3_checkpoint"})


if __name__ == "__main__":
    unittest.main()
