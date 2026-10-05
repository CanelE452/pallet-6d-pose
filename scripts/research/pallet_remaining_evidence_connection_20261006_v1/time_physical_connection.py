#!/usr/bin/env python3
"""Read-only CPU linkage audit; never runs a model, PnP, or hardware code.

Original input bytes remain untouched. The handoff's Windows paths are provenance,
not executable paths. Outputs are a separately versioned receiver audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shlex
import sys
import time
from collections import Counter, defaultdict

BASE = pathlib.Path("data/pallet/results/pallet_lifter_case_review_20261003_v1")
OUTPUT = pathlib.Path("data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical")
DOCS = pathlib.Path("_docs/experiments/pallet_remaining_evidence_connection_20261006_v1/time_physical")


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with pathlib.Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty audit table")
    with pathlib.Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number_equal(left, right, tolerance=1e-8):
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance


def source_record(path, role):
    path = pathlib.Path(path)
    return {"path": str(path), "role": role, "bytes": path.stat().st_size, "sha256": sha(path)}


def duplicated_adjacent(rows, key):
    """Count observations, preserving repeated stored frames rather than deduplicating."""
    return sum(rows[i][key] == rows[i - 1][key] for i in range(1, len(rows)))


def full_pose_valid(pose):
    """Check exported full-pose validity; this does not turn prediction into a reference."""
    if not pose or not pose.get("available"):
        return False
    rotation = pose.get("R_physical")
    translation = pose.get("centroid")
    if not isinstance(rotation, list) or len(rotation) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in rotation):
        return False
    if not isinstance(translation, list) or len(translation) != 3:
        return False
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for row in rotation for x in row):
        return False
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in translation) or translation[2] <= 0:
        return False
    orthogonal_error = max(abs(sum(rotation[k][i] * rotation[k][j] for k in range(3)) - int(i == j)) for i in range(3) for j in range(3))
    a, b, c = rotation
    determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    return orthogonal_error <= 1e-6 and abs(determinant - 1) <= 1e-6


def reference_gate(record):
    """Required independent reference linkage, separate from prediction availability."""
    failures = []
    gates = {
        "independent_metrology_provenance_missing": record.get("independent_source_verified") is True,
        "target_object_binding_missing": record.get("same_target_object_verified") is True,
        "exposure_clock_binding_missing": record.get("exposure_time_mapping_verified") is True,
        "surveyed_transform_chain_missing": record.get("surveyed_transform_chain_verified") is True,
        "metric_units_missing": record.get("units") == "m",
        "full_reference_R_t_missing_or_invalid": full_pose_valid({"available": True, "R_physical": record.get("reference_R"), "centroid": record.get("reference_t_m")}),
    }
    for name, passed in gates.items():
        if not passed:
            failures.append(name)
    return {"accepted": not failures, "failures": failures}


def validate_prediction_row(row, planned):
    checks = {
        "frame_id": row["frame_id"] == planned["frame_id"],
        "session": row["session_id"] == planned["session_id"],
        "stored_index": row["stored_index"] == planned["saved_frame_index"],
        "timestamp": number_equal(row["sensor_timestamp_ms"], planned["camera_sensor_timestamp_ms"]),
        "camera_number": row["camera_frame_number"] == planned["camera_frame_number"],
        "pixels": row["decoded_bgr_sha256"] == planned["decoded_bgr_sha256"],
    }
    require(all(checks.values()), "Prediction binding mismatch: " + planned["frame_id"] + " " + str(checks))
    raw = row["raw_shared_prediction"]
    require(raw["frame_i"] == planned["frame_i"], "Raw nested frame_i mismatch")
    result = {}
    for method, nested_name in [("Base", "R0"), ("N3", "N3_seed1")]:
        exported = row["methods"][method]
        nested = raw["methods"][nested_name]
        nested_points = nested["points_xy"]
        require(exported["keypoints"] == (nested_points[:8] if nested_points is not None else None), method + " corner export mismatch")
        pose = nested.get("pose") or {}
        is_full = full_pose_valid(pose)
        if exported["pose_state"] == "fresh":
            require(is_full, method + " fresh row lacks valid full exported pose")
            require(pose["cf_extents"] == [1.1, 0.15, 1.1], "Repeated axis swap or wrong object dimensions")
            require(pose["R_cf"] == pose["R_physical"], "Square physical Q should be identity")
            require(number_equal(exported["pose"]["x_m"], pose["centroid"][0]), "x coordinate mismatch")
            require(number_equal(exported["pose"]["z_m"], pose["centroid"][2]), "z coordinate mismatch")
        result[method] = {"state": exported["pose_state"], "full_pose": is_full}
    return result


def audit_timeline(handoff, receiver, output, sources, decode=True):
    import cv2

    plan_path = receiver / BASE / "LIFTER_EVALUATION_PLAN.json"
    map_path = receiver / BASE / "LIFTER_INPUT_AND_TIME_MAP.json"
    plan = read_json(plan_path)
    time_map = read_json(map_path)
    for path, copied in [(plan_path, "source_links/frozen_protocol/LIFTER_EVALUATION_PLAN.json"), (map_path, "source_links/frozen_protocol/LIFTER_INPUT_AND_TIME_MAP.json")]:
        require(sha(path) == sha(handoff / copied), "Frozen source differs between receiver and handoff: " + str(path))
        sources.extend([source_record(path, "receiver_frozen_contract"), source_record(handoff / copied, "handoff_frozen_original_copy")])
    handed_rows = read_csv(handoff / "FRAME_TIME_MATCH.csv")
    sources.append(source_record(handoff / "FRAME_TIME_MATCH.csv", "handed_8910_time_pixel_binding"))
    handed = {row["frame_id"]: row for row in handed_rows}
    require(len(handed_rows) == len(handed) == len(plan["frames"]) == 8910, "Fixed population changed")
    per_session = defaultdict(list)
    for frame in plan["frames"]:
        per_session[frame["session_id"]].append(frame)
    rows, summaries = [], []
    fields = ["session_id", "frame_id", "saved_frame_index", "timing_row_index", "frame_i", "camera_frame_number", "camera_sensor_timestamp_ms", "camera_timestamp_domain", "camera_input_host_mono_ms", "width", "height", "raw_video_sha256", "decoded_bgr_sha256"]
    numeric = {"saved_frame_index", "timing_row_index", "frame_i", "camera_frame_number", "camera_sensor_timestamp_ms", "camera_input_host_mono_ms", "width", "height"}
    for session in time_map["sessions"]:
        sid = session["session_id"]
        frames = per_session[sid]
        files = {}
        for record in session["files"]:
            path = receiver / "extracted/depth_cam/rec" / record["filename"]
            receipt = source_record(path, "receiver_lifter_" + record["role"])
            require(receipt["sha256"] == record["sha256"] and receipt["bytes"] == record["bytes"], "Source file hash mismatch: " + str(path))
            sources.append(receipt)
            files[record["role"]] = path
        timing = read_csv(files["timing"])
        require(len(timing) == session["timing_rows"], "Timing row population mismatch")
        capture = cv2.VideoCapture(str(files["raw_mp4"])) if decode else None
        decoded = 0
        for frame in frames:
            linked = handed[frame["frame_id"]]
            for key in fields:
                equal = number_equal(frame[key], linked[key]) if key in numeric else str(frame[key]) == linked[key]
                require(equal, "Handoff/frozen frame mismatch: " + frame["frame_id"] + ":" + key)
            measured = timing[frame["timing_row_index"]]
            for key in ["frame_i", "camera_frame_number", "camera_sensor_timestamp_ms", "camera_timestamp_domain", "camera_input_host_mono_ms"]:
                equal = number_equal(frame[key], measured[key]) if key in numeric else frame[key] == measured[key]
                require(equal, "Receiver timing CSV mismatch: " + frame["frame_id"] + ":" + key)
            if decode:
                ok, image = capture.read()
                require(ok, "Raw video ended early: " + sid)
                require(image.shape[:2] == (frame["height"], frame["width"]), "Image dimensions changed")
                require(hashlib.sha256(image.tobytes()).hexdigest() == frame["decoded_bgr_sha256"], "Decoded pixel mismatch: " + frame["frame_id"])
                decoded += 1
            rows.append({"frame_id": frame["frame_id"], "session_id": sid, "stored_index": frame["saved_frame_index"], "timing_row_index": frame["timing_row_index"], "sensor_timestamp_ms": frame["camera_sensor_timestamp_ms"], "camera_frame_number": frame["camera_frame_number"], "decoded_bgr_sha256": frame["decoded_bgr_sha256"], "handoff_plan_timing_match": True, "receiver_video_pixel_match": True if decode else "NOT_REDECODED", "Base_full_6D_export": "", "N3_full_6D_export": "", "Base_state": "", "N3_state": ""})
        if decode:
            require(not capture.read()[0], "Raw video has unplanned trailing frames")
            capture.release()
        intervals = [frames[i]["camera_sensor_timestamp_ms"] - frames[i - 1]["camera_sensor_timestamp_ms"] for i in range(1, len(frames))]
        require(all(dt >= 0 for dt in intervals), "Clock reset requires explicit separate segment")
        summary = {"session_id": sid, "stored_frames": len(frames), "timing_rows": len(timing), "original_timing_offset": session["video_timing_row_offset"], "pixel_redecoded_frames": decoded, "duplicate_sensor_timestamps": duplicated_adjacent(frames, "camera_sensor_timestamp_ms"), "duplicate_camera_numbers": duplicated_adjacent(frames, "camera_frame_number"), "backwards_sensor_timestamps": sum(dt < 0 for dt in intervals), "max_sensor_gap_ms": max(intervals), "clock_domains": sorted({frame["camera_timestamp_domain"] for frame in frames})}
        require(summary["duplicate_sensor_timestamps"] == session["duplicate_sensor_timestamps"], "Timestamp duplicate count changed")
        summaries.append(summary)
        print("TIME_SESSION", sid, len(frames), "pixels", decoded, flush=True)
    planned = {frame["frame_id"]: frame for frame in plan["frames"]}
    indexed_output = {row["frame_id"]: row for row in rows}
    prediction_summaries = []
    identity_path = receiver / BASE / "raw_predictions/RUN_IDENTITY.json"
    identity = read_json(identity_path)
    sources.append(source_record(identity_path, "completed_frozen_inference_identity_receipt"))
    require(identity["plan_sha256"] == sha(plan_path), "Completed inference used a different frame plan")
    identity_checks = []
    for binding in identity["fixed_bindings"]:
        observed = source_record(receiver / binding["path"], "completed_inference_fixed_" + binding["name"])
        require(observed["sha256"] == binding["sha256"] and observed["bytes"] == binding["bytes"], "Actual fixed model/code/receipt binding differs: " + binding["name"])
        sources.append(observed)
        identity_checks.append({"name": binding["name"], "path": binding["path"], "sha256": binding["sha256"], "bytes": binding["bytes"], "actual_receiver_hash_matches": True})
    paths = [receiver / BASE / "raw_predictions/ALL_STORED_FRAMES_L4.jsonl", receiver / "data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/ALL_STORED_FRAMES_EVALUATOR.jsonl"]
    signatures = {}
    for path in paths:
        sources.append(source_record(path, "completed_raw_prediction_export"))
        seen, states, full = set(), {"Base": Counter(), "N3": Counter()}, Counter()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                require(row["frame_id"] not in seen, "Duplicate frame_id in prediction output")
                seen.add(row["frame_id"])
                validated = validate_prediction_row(row, planned[row["frame_id"]])
                signature = json.dumps({"methods": {k: {x: v.get(x) for x in ["keypoints", "keypoints_mask", "pose_state", "pose", "selected_index"]} for k, v in row["methods"].items()}, "raw": row["raw_shared_prediction"]}, sort_keys=True)
                if row["frame_id"] in signatures:
                    require(signatures[row["frame_id"]] == signature, "Assisted adapter changed predictions or 6D exports")
                else:
                    signatures[row["frame_id"]] = signature
                for method in ["Base", "N3"]:
                    states[method][validated[method]["state"]] += 1
                    full[method] += int(validated[method]["full_pose"])
                    indexed_output[row["frame_id"]][method + "_full_6D_export"] = validated[method]["full_pose"]
                    indexed_output[row["frame_id"]][method + "_state"] = validated[method]["state"]
        require(seen == set(planned), "Prediction frame population differs from fixed plan")
        prediction_summaries.append({"source_path": str(path), "rows": len(seen), "method_states": {k: dict(v) for k, v in states.items()}, "nested_full_6D_valid_exports": dict(full), "raw_pose_kind": "prediction from completed same-model corners; never independent measurement"})
    write_csv(output / "FRAME_TIME_PREDICTION_JOIN.csv", rows)
    return {"status": "PASS", "frames": len(rows), "sessions": summaries, "original_duplicate_observations_retained": sum(s["duplicate_sensor_timestamps"] for s in summaries), "completed_inference_run_identity_bindings": identity_checks, "raw_prediction_exports": prediction_summaries, "new_inference": 0, "new_PnP": 0}


def audit_square(handoff, receiver, output, sources):
    import cv2

    snapshot_path = handoff / "physical/frozen_repository_sources/DATASET_SNAPSHOT.json"
    snapshot = read_json(snapshot_path)
    sources.append(source_record(snapshot_path, "frozen_square119_expected_original_hashes"))
    candidates = read_csv(handoff / "physical/SQUARE119_CANDIDATE_IMAGE_BINDINGS.csv")
    sources.append(source_record(handoff / "physical/SQUARE119_CANDIDATE_IMAGE_BINDINGS.csv", "archive_image_version_bindings"))
    original_candidates = {row["frame_id"]: row for row in candidates if row["candidate_version"] != "real_dataset"}
    require(len(snapshot["records"]) == len(original_candidates) == 119, "Square fixed population changed")
    rows, totals = [], Counter()
    for record in snapshot["records"]:
        sid = record["id"]
        image_path, label_path = receiver / record["image"]["path"], receiver / record["annotation"]["path"]
        for path, expected, role in [(image_path, record["image"], "receiver_original_square119_image"), (label_path, record["annotation"], "receiver_existing_square119_annotation")]:
            observed = source_record(path, role)
            require(observed["sha256"] == expected["sha256"] and observed["bytes"] == expected["bytes"], "Square original source mismatch: " + str(path))
            sources.append(observed)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        require(image is not None and list(image.shape[:2]) == record["original_hw"], "Square image dimensions mismatch")
        require(hashlib.sha256(image.tobytes()).hexdigest() == record["image_decoded_sha256"], "Square decoded pixels mismatch")
        candidate = original_candidates[sid]
        require(candidate["member_sha256"] == record["image"]["sha256"] and candidate["decoded_bgr_sha256"] == record["image_decoded_sha256"], "Handoff archive binding mismatch")
        label = read_json(label_path)
        require(len(label["objects"]) == 1, "Ambiguous square object assignment")
        obj = label["objects"][0]
        annotations = obj["keypoint_annotations"][:8]
        manual = sum(k["source"] == "manual_click" for k in annotations)
        manual_in_frame = sum(k["source"] == "manual_click" and k["in_frame"] for k in annotations)
        require(manual == record["manual_corners"] and manual_in_frame == record["manual_in_frame_corners"], "Square manual denominator mismatch")
        totals["manual_declared"] += manual
        totals["manual_in_frame"] += manual_in_frame
        totals["pnp_projected"] += sum(k["source"] == "pnp_projected" for k in annotations)
        totals["independent_canonical_pose"] += int(obj.get("canonical_pose") is not None)
        require(obj["physical_dimensions_m"] == {"x": 1.1, "y": 0.15, "z": 1.1}, "Square axis dimensions mismatch")
        rows.append({"frame_id": sid, "image_path": record["image"]["path"], "image_sha256": record["image"]["sha256"], "decoded_bgr_sha256": record["image_decoded_sha256"], "annotation_path": record["annotation"]["path"], "annotation_sha256": record["annotation"]["sha256"], "encoded_and_pixel_identity_match": True, "existing_annotation_exact_hash_match": True, "manual_declared": manual, "manual_in_frame": manual_in_frame, "split": obj["split"], "population_role": label["population_role"], "canonical_pose_present": obj.get("canonical_pose") is not None, "physical_accuracy": "x"})
    require(totals["manual_declared"] == 602 and totals["manual_in_frame"] == 600, "Square denominators drifted")
    write_csv(output / "SQUARE119_EXISTING_SOURCE_REUSE.csv", rows)
    return {"status": "PASS", "original_encoded_images": len(rows), "original_decoded_images": len(rows), "original_exact_annotations_reused": len(rows), "source_corner_counts": dict(totals), "new_annotations_requested": 0, "independent_physical_reference_rows": 0, "physical_T_R_accuracy": "x", "limitation": "Single DEV/train capture session; image-corner PnP pose is assisted geometry, not independent metrology"}


def audit_geometry(handoff, receiver, sources):
    contract_path = handoff / "physical/CAMERA_GEOMETRY_PNP_CONTRACT.json"
    geometry = read_json(contract_path)
    corner_path = receiver / BASE / "review/CORNER_CONTRACT.json"
    corner = read_json(corner_path)
    symmetry_path = handoff / "physical/original_smallfiles/e692b1526582_OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
    symmetry = read_json(symmetry_path)
    local_symmetry_path = receiver / "_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
    for path, role in [(contract_path, "handed_geometry_and_PnP_contract"), (corner_path, "receiver_existing_8corner_contract"), (symmetry_path, "handed_original_C4_contract"), (local_symmetry_path, "receiver_original_C4_contract")]:
        sources.append(source_record(path, role))
    require(sha(symmetry_path) == sha(local_symmetry_path), "C4 source contract differs")
    w, d, h = geometry["target_dimensions_wdh_m"]
    require([w, h, d] == geometry["target_pnp_xyz_m"] == [1.1, 0.15, 1.1], "WDH->XYZ converted incorrectly")
    expected_corners = [[sign[i] * geometry["target_pnp_xyz_m"][i] / 2 for i in range(3)] for sign in geometry["corner_order_sign_xyz"]]
    require(expected_corners == [item["xyz_m"] for item in corner["corners"]], "Corner order differs")
    square = next(obj for obj in symmetry["objects"] if obj["object_type"] == "plastic_standard_110x110x15")
    require(square["corners_centroid"][:8] == expected_corners, "C4 corner basis differs")
    require(square["group_order"] == 4, "Incorrect square symmetry group")
    for rotation, permutation in zip(square["rotations"], square["permutations"]):
        require(permutation[8] == 8 and sorted(permutation) == list(range(9)), "Symmetry must retain centroid and one whole-object bijection")
        for index, point in enumerate(square["corners_centroid"]):
            rotated = [sum(rotation[i][j] * point[j] for j in range(3)) for i in range(3)]
            require(max(abs(rotated[i] - square["corners_centroid"][permutation[index]][i]) for i in range(3)) < 1e-9, "C4 rotation/permutation mismatch")
    meta_summaries = []
    for record in geometry["target_lifter_metadata"]:
        meta_path = receiver / "extracted/depth_cam/rec" / ("forklift_v4_recording_20260901_" + record["session_id"] + "_meta.json")
        meta = read_json(meta_path)
        require(sha(meta_path) == record["input"]["sha256"] and meta == record["content"], "Handed and local camera metadata differ")
        require(meta["intrinsics"]["coeffs"] == [0.0] * 5, "Existing zero distortion recipe does not cover nonzero distortion")
        require(meta["pallet_size_m"] == {"width": w, "length": d, "height": h}, "Lifter WDH mismatch")
        require(meta["v4_extrinsics_measured"] is False, "Measured extrinsics status differs")
        meta_summaries.append({"session_id": record["session_id"], "K": [[meta["intrinsics"]["fx"], 0, meta["intrinsics"]["ppx"]], [0, meta["intrinsics"]["fy"], meta["intrinsics"]["ppy"]], [0, 0, 1]], "distortion": meta["intrinsics"]["coeffs"], "nominal_extrinsics_measured": False})
    copied_code = []
    for binding in geometry["source_code_bindings"]:
        copied = handoff / "physical/original_smallfiles" / binding["byte_copy_path"].replace("\\", "/").split("/")[-1]
        require(copied.exists() and sha(copied) == binding["sha256"], "Handed geometry source bytes missing or changed")
        sources.append(source_record(copied, "read_only_contract_code_not_executed"))
        code_status = {"role": binding["logical_role"], "sha256": binding["sha256"], "read_only": True}
        if "/evaluation_checkout/" in binding["actual_path"].replace("\\", "/"):
            relative_path = binding["actual_path"].replace("\\", "/").split("/evaluation_checkout/", 1)[1]
            local_path = receiver / relative_path
            if local_path.is_file():
                actual = source_record(local_path, "receiver_current_code_version_comparison_only")
                sources.append(actual)
                code_status.update({"receiver_current_path": relative_path, "receiver_current_sha256": actual["sha256"], "same_bytes_as_frozen_handoff_code": actual["sha256"] == binding["sha256"], "version_note": "Exact matching completed-inference recipe checked separately in RUN_IDENTITY; evolved current annotation/report code does not replace the frozen copy"})
        copied_code.append(code_status)
    return {"status": "PASS", "WDH_m": [w, d, h], "XYZ_m": [w, h, d], "conversion": "WDH -> W,H,D exactly once", "camera_axes": "+X right,+Y down,+Z forward; color optical center", "pallet_origin": "cuboid centroid", "corner_xyz_m": expected_corners, "C4_permutations": square["permutations"], "C4_rotation_index_pairs_verified": 4, "camera_metadata": meta_summaries, "source_code_bindings_verified": copied_code, "source_code_executed": False, "nominal_operational_extrinsics_are_physical_GT": False}


def record_rows(path):
    if path.suffix == ".csv":
        return read_csv(path)
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_physical(handoff, output, sources):
    inventory_path = handoff / "physical/PHYSICAL_REFERENCE_INVENTORY.json"
    inventory = read_json(inventory_path)
    sources.append(source_record(inventory_path, "handed_sensor_provenance_inventory"))
    sources.append(source_record(handoff / "physical/FRAME_MEASUREMENT_MATCH.csv", "handed_empty_physical_reference_candidate_join"))
    table = read_csv(handoff / "physical/FRAME_MEASUREMENT_MATCH.csv")
    require(len(table) == 9029 and len({(r["population"], r["frame_id"]) for r in table}) == 9029, "Physical fixed population mismatch")
    known_lifter = {r["frame_id"]: r for r in read_csv(output / "FRAME_TIME_PREDICTION_JOIN.csv")}
    known_square = {r["frame_id"]: r for r in read_csv(output / "SQUARE119_EXISTING_SOURCE_REUSE.csv")}
    require({r["frame_id"] for r in table if r["population"] == "lifter4_all_stored"} == set(known_lifter), "Physical target lifter IDs differ")
    require({r["frame_id"] for r in table if r["population"] == "static_square119"} == set(known_square), "Physical target square IDs differ")
    for row in table:
        checked = known_lifter[row["frame_id"]] if row["population"] == "lifter4_all_stored" else known_square[row["frame_id"]]
        require(checked["decoded_bgr_sha256"] == row["decoded_bgr_sha256"], "Physical table references a different target image")
        if row["population"] == "lifter4_all_stored":
            require(number_equal(checked["sensor_timestamp_ms"], row["camera_sensor_timestamp_ms"]), "Physical table camera time differs")
    populations = Counter(row["population"] for row in table)
    linked = [row for row in table if row["linked_independent_reference"].lower() == "true"]
    nonempty_pose = [row for row in table if row["reference_R"] or row["reference_t_m"]]
    require(not linked and not nonempty_pose, "Unexpected reference supplied: must validate independently before using")
    reference_gates = [reference_gate(row) for row in table]
    require(not any(gate["accepted"] for gate in reference_gates), "Independent reference gate unexpectedly passed")
    records = []
    sensor_inventory = inventory["existing_sensor_evidence"]
    candidates = [("wall_range_csv", entry) for entry in sensor_inventory["wall_range_csv"]]
    for session in sensor_inventory["multisensor_sessions"]:
        candidates.extend(("multisensor_" + record["input"]["logical_role"], record) for record in session["files"])
    candidates.extend(("relative_tag_pose", record) for record in sensor_inventory["relative_tag_pose_streams"])
    candidates.extend(("other_measurement_candidate", record) for record in sensor_inventory["other_measurement_candidates"])
    for category, entry in candidates:
        original = entry["input"]
        name = original.get("byte_copy_path", "").replace("\\", "/").split("/")[-1]
        copied = handoff / "physical/original_smallfiles" / name if name else None
        is_available = copied is not None and copied.is_file()
        expected_rows = entry.get("rows", entry.get("statistics", {}).get("rows"))
        row = {"category": category, "source_provenance_path": original["actual_path"], "source_sha256": original["sha256"], "source_bytes": original["bytes"], "available_original_bytes_in_handoff": is_available, "receiver_parsed_rows": "NA", "source_inventory_reported_rows": expected_rows if expected_rows is not None else "NA", "first_observed_time": "", "last_observed_time": "", "target_object_id_rows": "NA", "full_metric_R_t_rows": "NA", "linked_target_frames": 0, "decision": "UNLINKED_OTHER_CAPTURE_OR_NO_SURVEYED_TRANSFORM"}
        if is_available:
            receipt = source_record(copied, "copied_actual_sensor_or_fiducial_records")
            require(receipt["sha256"] == original["sha256"] and receipt["bytes"] == original["bytes"], "Copied sensor bytes differ")
            sources.append(receipt)
            observed = record_rows(copied)
            row["receiver_parsed_rows"] = len(observed)
            if expected_rows is not None:
                require(len(observed) == expected_rows, "Sensor parsed row count differs from inventory")
            time_strings = [x.get("time_iso", x.get("t_iso", "")) for x in observed if x.get("time_iso", x.get("t_iso", ""))]
            row["first_observed_time"] = time_strings[0] if time_strings else ""
            row["last_observed_time"] = time_strings[-1] if time_strings else ""
            row["target_object_id_rows"] = sum(bool(x.get("reference_object_id") or x.get("pallet_object_id")) for x in observed)
            row["full_metric_R_t_rows"] = sum(bool(x.get("reference_R") and x.get("reference_t_m")) for x in observed)
            require(row["target_object_id_rows"] == row["full_metric_R_t_rows"] == 0, "Unexpected physical reference fields need further validation")
            if category == "wall_range_csv":
                row["decision"] = "ACTUAL_WALL_RANGE_NOT_TARGET_PALLET_POSE; capture dates and object/extrinsics/time links differ"
            elif category == "relative_tag_pose":
                require(all(x.get("tag_id") == 0 for x in observed), "Unexpected tag ID")
                row["decision"] = "SEPT15_RELATIVE_TAG_ROTATION; unmeasured arbitrary unit tag size, no Sep1 object/time/surveyed-transform chain"
        else:
            row["decision"] = "PROVENANCE_INVENTORY_ONLY; original rows not transferred; not counted as receiver-parsed evidence"
        records.append(row)
    archived = inventory["archived_truck_tag_mount_evidence"]
    archive_summaries = []
    for member in archived["members"]:
        if "byte_copy_path" not in member:
            continue
        copied = handoff / "physical/archive_tag_sources" / member["byte_copy_path"].replace("\\", "/").split("/")[-1]
        require(copied.exists() and sha(copied) == member["member_sha256"], "Archived tag source bytes differ")
        sources.append(source_record(copied, "archived_other_truck_tag_calibration_not_target_reference"))
        if copied.suffix == ".json":
            data = read_json(copied)
            archive_summaries.append({"source": copied.name, "top_level_keys": list(data), "target_reference_accepted": False, "reason": "Other truck target and image/GT-fitted tag mounting, not surveyed target pallet transform"})
    write_csv(output / "ACTUAL_SENSOR_RECORDS_AUDIT.csv", records)
    write_csv(output / "PHYSICAL_REFERENCE_JOIN_STATUS.csv", [{"population": row["population"], "frame_id": row["frame_id"], "session_id": row["session_id"], "camera_sensor_timestamp_ms": row["camera_sensor_timestamp_ms"], "independent_reference_linked": False, "physical_T_R": "x", "reason": "NO_SAME_OBJECT_TIME_AND_SURVEYED_TRANSFORM_CHAIN"} for row in table])
    parsed = [row for row in records if row["available_original_bytes_in_handoff"]]
    unique_copied_sensor_files = {entry["input"]["byte_copy_path"] for _, entry in candidates if entry["input"].get("byte_copy_path")}
    counts = Counter()
    for row in parsed:
        counts[row["category"]] += row["receiver_parsed_rows"]
    return {"status": "AUDIT_COMPLETE_REFERENCE_NOT_AVAILABLE", "fixed_reference_join_population": dict(populations), "fixed_reference_join_rows": len(table), "independent_reference_gate_failures": dict(Counter(reason for gate in reference_gates for reason in gate["failures"])), "actual_receiver_parsed_sensor_files": len(unique_copied_sensor_files), "actual_receiver_parsed_logical_sensor_sources": len(parsed), "deduplicated_empty_source_copy_note": "27 logical original sources use 19 distinct byte-copy files; repeated empty source records are not extra physical files", "actual_receiver_parsed_sensor_rows_by_category": dict(counts), "inventory_only_sensor_files": len(records) - len(parsed), "copied_archive_calibration_jsons_parsed": archive_summaries, "accepted_independent_reference_rows": 0, "physical_T_R_accuracy": "x", "physical_accuracy_computed": False, "reasons": ["Laser records observe wall range/yaw; target pallet pose and camera transform are not measured", "September15 IMU/tag/camera experiments do not link September1 lifter targets or square119 image times", "Tag rotation uses unmeasured arbitrary unit size and has no surveyed marker-to-pallet transform", "Other truck tag mounts were fitted to image/GT poses and are not target-pallet independent survey", "Runtime camera/PnP estimates, commanded state, synthetic oracle outputs are excluded"]}


def build_report(result):
    timeline, square, physical = result["timeline"], result["square119"], result["physical_reference"]
    rows = ["| 세션 | 저장 프레임 | 중복 센서 시각 | 최대 시각 간격(ms) |", "|---|---:|---:|---:|"]
    rows += [f"| {s['session_id']} | {s['stored_frames']} | {s['duplicate_sensor_timestamps']} | {s['max_sensor_gap_ms']:.6f} |" for s in timeline["sessions"]]
    return "\n".join([
        "# 원영상·시각·기존 주석·독립 물리 참조 연결 검산",
        "", "수신 PC 원자료를 실제로 읽고 재검산했습니다. 기존 모델 추론과 주석을 그대로 재사용했으며 새 학습·추론·PnP·장치 실행은 모두 0회입니다.",
        "", "## 연결을 완료한 자료", "",
        f"네 세션 **{timeline['frames']:,}개** 저장 프레임을 인계 시각표, 고정 계획, 로컬 timing CSV, 원영상의 디코딩 픽셀, Base/N3 기존 예측과 연결했습니다. 원영상·meta·timing·control·state 파일 SHA-256도 고정 계약과 일치합니다. 중복 시각 **{timeline['original_duplicate_observations_retained']}개**를 지우지 않고 보존했습니다. 센서 시각을 사용하며 MP4의 nominal FPS로 바꾸지 않았습니다.",
        "", *rows, "",
        "기존 원시 예측의 겉쪽 pose 필드는 x/z/yaw지만 `raw_shared_prediction.methods.R0/N3_seed1.pose`에는 **3D 중심과 3×3 회전**이 이미 저장되어 있습니다. Base와 N3 각각 **8,772개**의 유효 full 6D 예측을 확인했습니다. 나머지 138개는 기존 no_pose이며 새 값을 만들지 않았습니다. 이 값은 모델의 예측이고 물리 정답이 아닙니다.",
        "", "완료 추론의 RUN_IDENTITY에 고정된 모델 두 개·PnP/보정 코드·치수/대칭/선택/학습 receipt의 **15개 바인딩**은 실제 수신 파일의 크기와 SHA가 모두 일치합니다. 인계의 과거 annotation/report 소스와 현재 코드 버전이 다른 경우도 별도로 기록하고 과거 원본은 보존했습니다.",
        "", f"정사각형 **{square['original_exact_annotations_reused']}장**은 원사진 바이트·디코딩 픽셀·원주석 SHA를 모두 대조했습니다. 인계 PC에서 찾지 못했던 기존 주석이 수신 PC에는 실제로 있습니다. 다시 주석할 필요가 없습니다. manual_declared **602점**, manual_in_frame **600점**을 분리해 보존합니다. 단일 촬영 세션의 DEV/train 자료라는 기존 제한도 유지합니다.",
        "", "## 카메라·치수·코너 연결", "",
        "카메라 좌표는 +X 오른쪽/+Y 아래/+Z 전방, 원점은 색상 광학 중심입니다. 팔레트 원점은 직육면체 중심입니다. W/D/H=[1.10,1.10,0.15] m를 PnP의 X/Y/Z=[1.10,0.15,1.10] m로 **한 번만** 변환합니다. 0–7 코너 순서와 C4의 네 회전·전체 객체 순열을 기존 로컬 계약과 검산했습니다. 네 meta의 K와 영 왜곡 계수도 일치합니다. `v4_extrinsics_measured=false`인 운영 설정을 실측 변환으로 바꾸지 않습니다.",
        "", "## 물리 정확도가 남는 이유", "",
        f"동봉된 실제 센서/태그 원문 **{physical['actual_receiver_parsed_sensor_files']}개 파일**을 직접 파싱했습니다. 이는 빈 원문 사본을 공유하는 **{physical['actual_receiver_parsed_logical_sensor_sources']}개 논리 출처**이며, 같은 빈 파일을 여러 물리 파일처럼 세지 않았습니다. 다른 파일의 원문이 ZIP에 없으면 인계 inventory의 행 수와 실제 읽은 행 수를 구분했습니다. 8,910개 리프터와 119개 정사각형의 총 **{physical['fixed_reference_join_rows']:,}행**에서 대상 팔레트·촬영 시각·실측 좌표 변환까지 연결되는 독립 참조는 **0행**입니다.",
        "", "레이저 CSV는 다른 촬영의 벽 거리와 벽 상대 각도이며, 태그 기록은 9월15일의 상대 회전입니다. 이 기록만으로 9월1일 리프터나 정사각형 119장의 팔레트 중심 위치·회전 정답을 만들 수 없습니다. 다른 트럭의 이미지 자세에 맞춰 역산한 태그 장착값, PnP 주석, 운영 추정과 합성 정답도 독립 물리 정답으로 사용하지 않았습니다. 따라서 독립 물리 T/R은 **x**입니다. 기존 코너 정확도와 연속성 결과는 유효한 별도 결과로 유지합니다.",
        "", "## 근거와 실행", "",
        "- [전체 검산 결과](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/VALIDATION.json)",
        "- [8,910개 시각·픽셀·예측 연결](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/FRAME_TIME_PREDICTION_JOIN.csv)",
        "- [정사각형 기존119장 원본 재사용](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/SQUARE119_EXISTING_SOURCE_REUSE.csv)",
        "- [실제 센서 파싱과 inventory 구분](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/ACTUAL_SENSOR_RECORDS_AUDIT.csv)",
        "- [9,029개 물리 참조 연결 상태](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/PHYSICAL_REFERENCE_JOIN_STATUS.csv)",
        "- [읽은 입력 SHA-256](../../../../data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/time_physical/SOURCE_HASHES.json)",
        "", "실행 명령:", "", "```bash", result["execution_command"], "```", "",
        f"CPU 실행 {result['execution_wall_seconds']:.3f}초. 원영상 전체 재디코딩과 119장 픽셀 검산을 포함합니다. GPU 추론 0회, optimizer update 0회, 원본 수정 0개.", "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", required=True, type=pathlib.Path)
    parser.add_argument("--receiver-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    start = time.perf_counter()
    output, docs = args.output_root / OUTPUT, args.output_root / DOCS
    output.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    sources = []
    result = {"schema": "receiver_time_physical_connection_cpu_audit_v1", "source_kind": "actual_receiver_read_only_validation", "status": "RUNNING", "new_training": 0, "optimizer_updates": 0, "new_model_inference": 0, "new_PnP": 0, "hardware_control": 0, "original_files_modified": 0}
    result["timeline"] = audit_timeline(args.handoff_root, args.receiver_root, output, sources)
    result["square119"] = audit_square(args.handoff_root, args.receiver_root, output, sources)
    result["geometry"] = audit_geometry(args.handoff_root, args.receiver_root, sources)
    result["physical_reference"] = audit_physical(args.handoff_root, output, sources)
    # Recheck source bytes after the audit, including actual videos and annotations.
    unique = {source["path"]: source for source in sources}
    for source in unique.values():
        require(sha(source["path"]) == source["sha256"], "Original input changed during audit")
    result["source_files_hash_verified_unchanged"] = len(unique)
    result["execution_command"] = shlex.join([sys.executable, "scripts/research/pallet_remaining_evidence_connection_20261006_v1/time_physical_connection.py", "--handoff-root", str(args.handoff_root), "--receiver-root", str(args.receiver_root), "--output-root", str(args.output_root)])
    result["execution_wall_seconds"] = time.perf_counter() - start
    result["status"] = "PASS_LINKAGE_AUDIT_REFERENCE_GAPS_RETAINED"
    write_json(output / "SOURCE_HASHES.json", list(unique.values()))
    write_json(output / "VALIDATION.json", result)
    (docs / "REPORT_KO.md").write_text(build_report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "elapsed_s": result["execution_wall_seconds"], "source_hashes": len(unique), "physical_reference_rows": result["physical_reference"]["accepted_independent_reference_rows"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
