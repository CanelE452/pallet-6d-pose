"""Read frozen predictions during every handed CAN-neutral interval.

This descriptive command-conditioned panel does not approve stationary intervals
or call the frozen stationary-noise evaluator. No models or control code load.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

PREFIX = Path("_docs/experiments/pallet_remaining_evidence_connection_20261006_v1/neutral_intervals")
RESULTS = Path("data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/neutral_intervals")
PORTABLE = Path("_docs/experiments/pallet_paper_review_20261006_v1/portable_evidence/PORTABLE_EVIDENCE_MANIFEST.json")
RAW_PATH = "data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES_L4.jsonl"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def wrapped_delta(after, before):
    return (after - before + 180) % 360 - 180


def pose_stats(methods):
    states = {name: 0 for name in ("fresh", "held", "no_pose")}
    poses = []
    for method in methods:
        state = method.get("pose_state")
        if state not in states:
            raise ValueError("Unrecognized explicit pose state")
        states[state] += 1
        pose = method.get("pose")
        valid = isinstance(pose, dict) and all(finite(pose.get(k)) for k in ("x_m", "z_m", "yaw_deg"))
        if (state in ("fresh", "held")) != valid:
            raise ValueError("Pose state and finite output disagree")
        if valid:
            poses.append(pose)
    angles = []
    for pose in poses:
        angles.append(pose["yaw_deg"] if not angles else angles[-1] + wrapped_delta(pose["yaw_deg"], angles[-1]))
    jumps = [wrapped_delta(b["yaw_deg"], a["yaw_deg"]) for a, b in zip(poses, poses[1:])]
    return {
        "stored_frame_count": len(methods), "valid_output_count": len(poses), **states,
        "x_population_std_m": statistics.pstdev([p["x_m"] for p in poses]) if poses else None,
        "z_population_std_m": statistics.pstdev([p["z_m"] for p in poses]) if poses else None,
        "yaw_population_std_360_unwrapped_deg": statistics.pstdev(angles) if angles else None,
        "wrapped_yaw_jump_ge90_count": sum(abs(delta) >= 90 for delta in jumps),
        "spread_defined": bool(poses),
    }


def select_interval(candidate, timeline, predictions):
    session = candidate["session_id"]
    start, end = candidate["sensor_start_ms"], candidate["sensor_end_ms"]
    if not finite(start) or not finite(end) or end < start:
        raise ValueError("Invalid sensor-time boundaries")
    rows = [row for row in timeline if row["session_id"] == session
            and int(row["sensor_clock_segment"]) == candidate["sensor_clock_segment"]
            and start <= float(row["camera_sensor_timestamp_ms"]) <= end]
    rows.sort(key=lambda row: int(row["saved_frame_index"]))
    ids = [row["frame_id"] for row in rows]
    if len(ids) != candidate["stored_frame_count"]:
        raise ValueError("Handed interval frame count changed")
    if ids and (ids[0] != candidate["stored_start_frame_id"] or ids[-1] != candidate["stored_end_frame_id"]):
        raise ValueError("Interval endpoint identity changed")
    timestamps = [float(row["camera_sensor_timestamp_ms"]) for row in rows]
    methods = {}
    for method in ("Base", "N3"):
        methods[method] = pose_stats([predictions[frame_id]["methods"][method] for frame_id in ids])
    return {
        "interval_id": candidate["interval_id"], "session_id": session,
        "sensor_clock_segment": candidate["sensor_clock_segment"],
        "sensor_start_ms": start, "sensor_end_ms": end,
        "observed_sensor_span_s": (end - start) / 1000,
        "stored_start_frame_id": ids[0] if ids else None,
        "stored_end_frame_id": ids[-1] if ids else None,
        "stored_frame_count": len(ids),
        "duplicate_sensor_timestamp_rows": len(timestamps) - len(set(timestamps)),
        "duplicate_camera_number_rows": len(rows) - len({row["camera_frame_number"] for row in rows}),
        "maximum_adjacent_sensor_gap_ms": max((b-a for a, b in zip(timestamps, timestamps[1:])), default=None),
        "command_end_right_censored": candidate["right_censored_command_end"],
        "relative_stationarity": "UNKNOWN",
        "reviewer_id": candidate["reviewer_id"], "reviewed_at": candidate["reviewed_at"],
        "fixed_before_predictions": candidate["fixed_before_predictions"],
        "legacy_stop_evaluator_usable": False,
        "methods": methods, "frame_ids": ids,
    }


def load_inputs(root, handoff=None):
    input_dir = root / PREFIX / "inputs"
    manifest_path = input_dir / "INPUT_MANIFEST.json"
    if handoff:
        input_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for name in ("STOP_INTERVALS_INVENTORY.json", "STORED_SENSOR_TIMELINE.csv"):
            content = (handoff / "stops" / name).read_bytes()
            artifact = input_dir / (name + ".gz")
            artifact.write_bytes(gzip.compress(content, mtime=0))
            entries.append({"source_path": "stops/" + name, "source_bytes": len(content), "source_sha256": sha(content),
                            "published_path": str(artifact.relative_to(root)), "compressed_bytes": artifact.stat().st_size,
                            "compressed_sha256": sha(artifact.read_bytes())})
        manifest_path.write_text(json.dumps({"files": entries, "originals_preserved": True}, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text())
    contents = {}
    for entry in manifest["files"]:
        compressed = (root / entry["published_path"]).read_bytes()
        if sha(compressed) != entry["compressed_sha256"]:
            raise ValueError("Compressed input hash mismatch")
        data = gzip.decompress(compressed)
        if len(data) != entry["source_bytes"] or sha(data) != entry["source_sha256"]:
            raise ValueError("Original input hash mismatch")
        contents[Path(entry["source_path"]).name] = data.decode("utf-8-sig")
    portable = json.loads((root / PORTABLE).read_text())
    raw = next(entry for entry in portable["compressed_sources"] if entry["original_path"] == RAW_PATH)
    compressed = (root / raw["artifact_path"]).read_bytes()
    if sha(compressed) != raw["artifact_sha256"]:
        raise ValueError("Raw prediction compressed hash mismatch")
    data = gzip.decompress(compressed)
    if sha(data) != raw["original_sha256"] or len(data) != raw["original_bytes"]:
        raise ValueError("Raw prediction source hash mismatch")
    predictions = {}
    for line in data.splitlines():
        row = json.loads(line)
        if row["frame_id"] in predictions:
            raise ValueError("Duplicate frame ID cannot be collapsed")
        predictions[row["frame_id"]] = row
    timeline = list(csv.DictReader(io.StringIO(contents["STORED_SENSOR_TIMELINE.csv"])))
    inventory = json.loads(contents["STOP_INTERVALS_INVENTORY.json"])
    return inventory, timeline, predictions, {"inputs": manifest["files"], "prediction_original_path": RAW_PATH,
                                            "prediction_original_sha256": raw["original_sha256"],
                                            "prediction_published_path": raw["artifact_path"]}


def calculate(root, handoff=None):
    inventory, timeline, predictions, sources = load_inputs(root, handoff)
    if len(timeline) != 8910 or len(predictions) != 8910 or len({row["frame_id"] for row in timeline}) != 8910:
        raise ValueError("Frozen all-frame population changed")
    for row in timeline:
        pred = predictions[row["frame_id"]]
        if pred["session_id"] != row["session_id"] or pred["stored_index"] != int(row["saved_frame_index"]):
            raise ValueError("Frame identity mismatch")
        if pred["sensor_timestamp_ms"] != float(row["camera_sensor_timestamp_ms"]):
            raise ValueError("Sensor timestamp mismatch")
        if pred["camera_frame_number"] != int(row["camera_frame_number"]):
            raise ValueError("Camera frame number mismatch")
    candidates = inventory["machine_proposed_intervals"]
    if len(candidates) != 32:
        raise ValueError("The frozen handoff contains exactly 32 command candidates")
    intervals = [select_interval(c, timeline, predictions) for c in candidates]
    ids = [fid for item in intervals for fid in item["frame_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping command intervals must not be pooled")
    totals = {method: {key: sum(item["methods"][method][key] for item in intervals)
                       for key in ("stored_frame_count", "valid_output_count", "fresh", "held", "no_pose")}
              for method in ("Base", "N3")}
    return {
        "status": "COMPLETE_POSTHOC_NEUTRAL_COMMAND_DESCRIPTION",
        "scope": "Every supplied neutral-command interval; relative camera/pallet stationarity unconfirmed. This is output spread during a command condition, not stationary noise or accuracy.",
        "selection": "All 32 handed command-derived intervals; no filtering on model output or inferred stability",
        "frozen_stationary_noise": {"status": "WAITING_HUMAN", "value": None, "display": "x"},
        "independent_physical_accuracy": {"status": "BLOCKED_REFERENCE", "value": None, "display": "x"},
        "candidate_count": len(intervals),
        "handed_confirmed_stationary_intervals": len(inventory["existing_verified_human_stationary_intervals"]),
        "all_frame_identity_checks": 8910,
        "neutral_command_unique_stored_frames": len(ids),
        "outside_neutral_command_frames": len(timeline) - len(ids),
        "observed_sensor_spans_sum_s": sum(item["observed_sensor_span_s"] for item in intervals),
        "duplicate_sensor_timestamp_rows_in_intervals": sum(item["duplicate_sensor_timestamp_rows"] for item in intervals),
        "methods": totals, "sources": sources, "intervals": intervals,
        "formulas": {"x_z_spread": "population standard deviation (ddof=0) of finite x_m/z_m outputs inside each separate sensor-clock interval",
                     "yaw_spread": "population SD after sequential 360-degree unwrap; physical/canonical 90/180-degree changes retained; no symmetry realignment",
                     "weighting": "one stored frame per observation, retaining duplicate timestamps/camera frames; no pooling across intervals",
                     "missing": "Every stored frame retained in availability denominator; finite pose outputs only for conditional spread; empty valid outputs = NA, measured zero = 0"},
        "base_training_seed": 42, "n3_seed": 1,
        "new_training": 0, "optimizer_updates": 0, "new_model_inference": 0, "hardware_control": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    result = calculate(args.root, args.handoff)
    output = args.root / RESULTS
    if args.verify_only:
        expected = json.loads((output / "NEUTRAL_COMMAND_RESULT.json").read_text())
        if result != expected:
            raise ValueError("Published command-panel result differs from independently recomputed inputs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "NEUTRAL_COMMAND_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        columns = ["interval_id", "session_id", "observed_sensor_span_s", "stored_frame_count", "duplicate_sensor_timestamp_rows",
                   "method", "valid_output_count", "fresh", "held", "no_pose", "x_population_std_m", "z_population_std_m", "yaw_population_std_360_unwrapped_deg", "relative_stationarity"]
        with (output / "ALL_32_INTERVALS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for interval in result["intervals"]:
                for method, stats in interval["methods"].items():
                    row = {key: interval.get(key) for key in columns if key not in stats}
                    row.update({key: value for key, value in stats.items() if key in columns}); row["method"] = method
                    writer.writerow(row)
        (output / "EXECUTION_COST.json").write_text(json.dumps({"status": "PASS", "cpu_wall_seconds": time.perf_counter()-start,
                       "all_frame_identity_checks": 8910, "calculated_interval_methods": 64,
                       "new_model_inference": 0, "new_training": 0, "optimizer_updates": 0, "hardware_control": 0}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "verified_existing_result": args.verify_only,
                      "candidates": result["candidate_count"], "stored_frames": result["neutral_command_unique_stored_frames"],
                      "methods": result["methods"], "stationary_noise": "x", "cpu_wall_seconds": time.perf_counter()-start}))


if __name__ == "__main__":
    main()
