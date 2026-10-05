"""Synthetic L4 fixtures.  They are never experiment or human evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
import urllib.request
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "metrics"))

from bridge import (GateError, adapt_row_masks, apply_sidecar,
                    build_object_match_queue, point_valid_mask, read_json, read_jsonl,
                    separated_status, sha256, write_json)
from object_match_review import Context, Server
from evaluate import ContractError, evaluate_visible_corners


BINDINGS = {"manifest_sha256": "m", "plan_sha256": "p",
            "corner_contract_sha256": "c", "corner_definition_version": "v"}


def png(width=20, height=10):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    raw = b"".join(b"\0" + b"\0\0\0" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(
        ">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def method(points=None):
    points = points if points is not None else [[2 + i, 3] for i in range(8)]
    return {"pose_state": "fresh", "pose": {"x_m": 0, "z_m": 3, "yaw_deg": 0},
            "detection_present": True, "pnp_failed": False,
            "keypoints": copy.deepcopy(points), "keypoints_mask": [True] * 8,
            "object_id": None, "object_match": None, "selected_index": 0,
            "selected_object": {"candidate_count": 1, "selected_index": 0,
                "selected_box_xyxy": [1, 1, 18, 9], "selected_score": .8,
                "selection_rule": "frozen confidence", "target_match_status": "not_human_reviewed"}}


def prediction(points=None):
    base, n3 = method(points), method(points)
    return {"frame_id": "s:0", "session_id": "s", "stored_index": 0,
            "sensor_timestamp_ms": 1000, "camera_frame_number": 7,
            "decoded_bgr_sha256": "pixels", "inference_execution_id": "test:0",
            "methods": {"Base": base, "N3": n3},
            "raw_shared_prediction": {"methods": {
                "R0": {"points_xy": copy.deepcopy(base["keypoints"])},
                "N3_seed1": {"points_xy": copy.deepcopy(n3["keypoints"])}}}}


def reference_record():
    corners = []
    for index in range(8):
        visible = index < 2
        corners.append({"id": index,
            "visibility": "direct_visible" if visible else "not_direct_visible",
            "external_occlusion": False, "self_occlusion": not visible,
            "out_of_frame": False, "definition_uncertain": False,
            "definition_confirmed": True, "x": 2 + index if visible else None,
            "y": 3 if visible else None, "reason": "synthetic fixture"})
    return {"frame_id": "s:0", "session_id": "s", "review_pass": "primary",
            "status": "reviewed", "source_kind": "human_reviewed",
            "image_sha256": "filled_in_test", "width": 20, "height": 10,
            "plan_sha256": "p", "corner_contract_sha256": "c",
            "corner_definition_version": "v",
            "object": {"presence": "present", "target_identity_confirmed": True,
                       "target_object_id": "fixture_target"},
            "corners": corners,
            "reviewer": {"id": "SYNTHETIC_CORNER_FIXTURE", "entered_by": "human",
                "confirmation": True, "machine_assistance": False,
                "previous_prediction_exposure": False,
                "previous_annotation_exposure": False,
                "exposure_notes": "synthetic fixture only"},
            "review_time": {"started_at": "2000-01-01T00:00:00Z",
                "finished_at": "2000-01-01T00:00:01Z", "duration_seconds": 1,
                "clock_source": "server_wall_and_monotonic"}}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        output_root = self.root / "data/pallet/results/pallet_lifter_case_review_20261003_v1"
        review_root = output_root / "review"
        review_root.mkdir(parents=True)
        self.plan = output_root / "LIFTER_EVALUATION_PLAN.json"
        write_json(self.plan, {"synthetic_fixture": True})
        self.contract = review_root / "CORNER_CONTRACT.json"
        write_json(self.contract, {"schema_version": "lifter_corner_contract_v1",
            "version": "v", "object_definition": "synthetic fixture",
            "source": "unit test only", "direct_click_policy": "human_confirmation_required",
            "corners": [{"id": index, "name": str(index), "definition": "synthetic"}
                        for index in range(8)]})
        image = review_root / "raw.png"
        image.write_bytes(png())
        self.manifest = review_root / "MANIFEST.json"
        write_json(self.manifest, {"schema_version": "lifter_review_manifest_v1",
            "plan_sha256": sha256(self.plan), "corner_contract_sha256": sha256(self.contract),
            "frames": [{"frame_id": "s:0", "session_id": "s", "saved_frame_index": 0,
                "camera_sensor_timestamp_ms": 1000, "repeat_review": False,
                "width": 20, "height": 10, "image_path": "raw.png",
                "image_sha256": sha256(image)}]})
        self.bindings = {"manifest_sha256": sha256(self.manifest),
            "plan_sha256": sha256(self.plan), "corner_contract_sha256": sha256(self.contract),
            "corner_definition_version": "v"}
        record = reference_record()
        record["image_sha256"] = sha256(image)
        for key in ("plan_sha256", "corner_contract_sha256", "corner_definition_version"):
            record[key] = self.bindings[key]
        self.reference = self.root / "reviewed.json"
        write_json(self.reference, {"schema_version": "lifter_reference_review_v1",
            "source_kind": "human_reviewed", "bindings": self.bindings,
            "records": [record]})
        self.predictions = self.root / "ALL_STORED_FRAMES.jsonl"
        self.predictions.write_text(json.dumps(prediction(), allow_nan=False) + "\n")
        write_json(self.root / "RUN_IDENTITY.json", {"synthetic_fixture": True})
        self.queue = self.root / "queue.json"
        build_object_match_queue(self.predictions, self.reference,
                                 self.manifest, self.queue)
        self.sidecar_counter = 0

    def reviewer(self):
        return {"id": "SYNTHETIC_OBJECT_FIXTURE", "entered_by": "human",
                "machine_assistance": True, "previous_prediction_exposure": True,
                "previous_annotation_exposure": False,
                "exposure_notes": "synthetic fixture only", "confirmation": True}

    def test_mask_none_nan_sentinel_offscreen_and_partial(self):
        nan = float("nan")
        points = [[-1, -1], [-12, 5], [0, 0], [2, 3], [None, 3],
                  [nan, 1], [30, 40], [1, 2]]
        self.assertEqual(point_valid_mask(points, missing_sentinels=[[-1, -1]]),
                         [False, True, True, True, False, False, True, True])
        self.assertEqual(point_valid_mask(points, explicit_valid=[True, False, True,
            True, True, True, True, True], missing_sentinels=[[-1, -1]]),
            [False, False, True, True, False, False, True, True])
        self.assertEqual(point_valid_mask(None), [])
        with self.assertRaises(GateError):
            point_valid_mask(None, explicit_valid=[True] * 8)

    def test_base_n3_missingness_must_match(self):
        row = prediction()
        row["methods"]["N3"]["keypoints"][1] = [None, None]
        with self.assertRaises(GateError):
            adapt_row_masks(row)

    def _human_sidecar_through_http(self, decision="same"):
        self.sidecar_counter += 1
        store = self.root / ("store_" + str(self.sidecar_counter) + ".json")
        context = Context(self.queue, self.manifest, self.predictions,
                          self.reference, store)
        server = Server(("127.0.0.1", 0), context)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:" + str(server.server_address[1])
        def post(route, body):
            request = urllib.request.Request(base + route,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Origin": base})
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        with urllib.request.urlopen(base + "/") as response:
            self.assertIn("리프터 대상 대응 검수", response.read().decode("utf-8"))
        with urllib.request.urlopen(base + "/api/catalog") as response:
            catalog = json.load(response)
        self.assertFalse(catalog["policy"]["show_model_corners"])
        with urllib.request.urlopen(base + "/api/image?frame_id=s%3A0") as response:
            self.assertTrue(response.read().startswith(b"\x89PNG"))
        started = post("/api/start", {"frame_id": "s:0", "reviewer": self.reviewer()})
        saved = post("/api/save", {"token": started["token"], "record": {
            "frame_id": "s:0", "decision": decision,
            "decision_reason": "synthetic fixture", "edit_reason": ""}})
        self.assertTrue(saved["saved"])
        with urllib.request.urlopen(base + "/api/export") as response:
            sidecar = json.load(response)
        output = self.root / ("sidecar_" + decision + "_" + str(self.sidecar_counter) + ".json")
        write_json(output, sidecar)
        return output

    def test_ui_save_sidecar_apply_then_existing_metrics(self):
        sidecar = self._human_sidecar_through_http("same")
        derived, receipt = self.root / "derived.jsonl", self.root / "receipt.json"
        apply_sidecar(self.predictions, self.queue, sidecar, derived, receipt)
        row = read_jsonl(derived)[0]
        frame = {"frame_id": "s:0", "session_id": "s", "saved_frame_index": 0,
            "camera_sensor_timestamp_ms": 1000, "camera_frame_number": 7,
            "camera_timestamp_domain": "synthetic", "width": 20, "height": 10,
            "image_sha256": read_json(self.reference)["records"][0]["image_sha256"],
            "decoded_bgr_sha256": "pixels", "repeat_review": False}
        bundle = read_json(self.reference)
        result = evaluate_visible_corners([frame], bundle, {"s:0": row}, self.bindings)
        self.assertEqual(result["methods"]["Base"]["pck10_pct"], 100)
        self.assertEqual(result["methods"]["N3"]["pck10_pct"], 100)
        self.assertTrue(row["methods"]["Base"]["object_match"])
        self.assertEqual(row["methods"]["Base"]["keypoints_mask"], [True] * 8)

    def test_different_is_kept_as_failure_and_undetermined_blocks(self):
        for decision in ("different", "undetermined"):
            with self.subTest(decision=decision):
                sidecar = self._human_sidecar_through_http(decision)
                derived = self.root / ("derived_" + decision + ".jsonl")
                apply_sidecar(self.predictions, self.queue, sidecar, derived,
                              self.root / ("receipt_" + decision + ".json"))
                row = read_jsonl(derived)[0]
                frame = {"frame_id": "s:0", "session_id": "s", "saved_frame_index": 0,
                    "camera_sensor_timestamp_ms": 1000, "camera_frame_number": 7,
                    "camera_timestamp_domain": "synthetic", "width": 20, "height": 10,
                    "image_sha256": read_json(self.reference)["records"][0]["image_sha256"],
                    "decoded_bgr_sha256": "pixels", "repeat_review": False}
                if decision == "different":
                    result = evaluate_visible_corners([frame], read_json(self.reference),
                                                      {"s:0": row}, self.bindings)
                    self.assertEqual(result["methods"]["Base"]["failure_counts"],
                                     {"wrong_object": 2})
                else:
                    with self.assertRaises(ContractError):
                        evaluate_visible_corners([frame], read_json(self.reference),
                                                 {"s:0": row}, self.bindings)

    def test_machine_decision_and_changed_selection_are_rejected(self):
        sidecar = self._human_sidecar_through_http("same")
        value = json.loads(sidecar.read_text())
        value["records"][0]["source_kind"] = "machine_proposed"
        write_json(sidecar, value)
        with self.assertRaises(GateError):
            apply_sidecar(self.predictions, self.queue, sidecar,
                          self.root / "bad.jsonl", self.root / "bad_receipt.json")
        sidecar = self._human_sidecar_through_http("same")
        value = json.loads(sidecar.read_text())
        value["records"][0]["selected_index"] = 2
        write_json(sidecar, value)
        with self.assertRaises(GateError):
            apply_sidecar(self.predictions, self.queue, sidecar,
                          self.root / "bad2.jsonl", self.root / "bad2_receipt.json")

    def test_partial_status_does_not_erase_completed_inference(self):
        preflight, inference, metrics = (self.root / "preflight.json",
                                         self.root / "inference.json",
                                         self.root / "metrics.json")
        write_json(preflight, {"status": "READY_TO_REVIEW", "missing_bindings": []})
        write_json(inference, {"status": "VERIFIED_COMPLETE", "inferred_frames": 8910})
        write_json(metrics, {"statuses": {"all_stored_predictions": "VERIFIED_COMPLETE",
            "human_reference": "WAITING_HUMAN"},
            "reviewed_stop_variation": {"status": "WAITING_HUMAN"},
            "independent_physical_accuracy": {"status": "BLOCKED_REFERENCE"}})
        status = separated_status(preflight=preflight, inference=inference, metrics=metrics)
        self.assertEqual(status["full_inference"]["status"], "VERIFIED_COMPLETE")
        self.assertEqual(status["corner_reference"]["status"], "WAITING_HUMAN")
        self.assertEqual(status["independent_physical_reference"]["status"], "BLOCKED_REFERENCE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
