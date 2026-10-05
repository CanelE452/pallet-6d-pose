"""Separate stored-output continuity from human visible-corner accuracy.

Only offline JSON and CSV are read. No hardware module or training code is imported.
The prediction schema is emitted by the isolated run_pipeline.normalize function.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

METHODS = ("Base", "N3")
IDENTITY = tuple(range(8))
FROZEN_LIFTER_RUN_SHA256 = "6c659332e6b02381d2a6c358daf69d305a38c8d1ffb0b3964563b50d19f67932"


class ContractError(ValueError):
    """An identity, state, or reference cannot be verified."""


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def describe(values):
    return {"count": len(values), "median": percentile(values, .5),
            "p90": percentile(values, .9), "maximum": max(values) if values else None}


def wrap_delta(after, before):
    """Remove only 360-degree representation wrapping, retaining 90/180 changes."""
    return (after - before + 180.0) % 360.0 - 180.0


def _index_frames(frames):
    result = {}
    stored = set()
    for frame in frames:
        key = frame["frame_id"]
        pair = (frame["session_id"], frame["saved_frame_index"])
        if key in result or pair in stored:
            raise ContractError("Duplicate frame ID or session/stored index")
        if not finite(frame["camera_sensor_timestamp_ms"]):
            raise ContractError("Nonfinite sensor timestamp")
        result[key] = frame
        stored.add(pair)
    return result


def index_predictions(rows, frames):
    expected = _index_frames(frames)
    predictions = {}
    for row in rows:
        key = row["frame_id"]
        if key not in expected or key in predictions:
            raise ContractError("Unknown or duplicate prediction frame ID")
        frame = expected[key]
        for actual_key, planned_key in (("session_id", "session_id"),
                                        ("stored_index", "saved_frame_index"),
                                        ("sensor_timestamp_ms", "camera_sensor_timestamp_ms"),
                                        ("camera_frame_number", "camera_frame_number"),
                                        ("decoded_bgr_sha256", "decoded_bgr_sha256")):
            if row.get(actual_key) != frame.get(planned_key):
                raise ContractError("Prediction frame binding mismatch: " + actual_key)
        if not isinstance(row.get("inference_execution_id"), str) or not row["inference_execution_id"]:
            raise ContractError("Explicit inference_execution_id required")
        for method in METHODS:
            if method not in row.get("methods", {}):
                raise ContractError("Both frozen methods must be explicitly present")
        base, n3 = (row["methods"][m] for m in METHODS)
        for field in ("selected_index", "keypoints_mask"):
            if field in base and field in n3 and base[field] != n3[field]:
                raise ContractError("Frozen N3 changed original selection/missingness: " + field)
        predictions[key] = row
    return predictions


def _groups(frames):
    sessions = defaultdict(list)
    for frame in frames:
        sessions[frame["session_id"]].append(frame)
    for session, seq in sessions.items():
        seq.sort(key=lambda f: f["saved_frame_index"])
        for before, after in zip(seq, seq[1:]):
            if before.get("sensor_clock_segment", 0) == after.get("sensor_clock_segment", 0):
                if after["camera_sensor_timestamp_ms"] < before["camera_sensor_timestamp_ms"]:
                    raise ContractError("Sensor timestamp regression without declared clock segment: " + session)
                if after.get("camera_timestamp_domain") != before.get("camera_timestamp_domain"):
                    raise ContractError("Sensor timestamp domain changed without a declared clock segment")
    return sessions


def _pose(pred):
    pose = pred.get("pose")
    if not isinstance(pose, dict) or not all(finite(pose.get(k)) for k in ("x_m", "z_m", "yaw_deg")):
        return None
    return pose


def _state(pred):
    state = pred.get("pose_state")
    if state not in ("fresh", "held", "no_pose"):
        raise ContractError("Missing or invalid explicit pose_state")
    if state in ("fresh", "held") and _pose(pred) is None:
        raise ContractError("Fresh/held state requires a finite explicit output pose")
    if state == "no_pose" and _pose(pred) is not None:
        raise ContractError("no_pose cannot silently carry a finite output pose")
    return state


def _missing_runs(sequence, states, predicate):
    """Sampled sensor-time approximations, split on declared sensor clock resets."""
    runs, start = [], None
    for index in range(len(sequence) + 1):
        at_end = index == len(sequence)
        reset = (not at_end and index > 0 and
                 sequence[index].get("sensor_clock_segment", 0) != sequence[index-1].get("sensor_clock_segment", 0))
        missing = not at_end and predicate(states[index])
        if start is not None and (at_end or reset or not missing):
            last = index - 1
            right_censored = at_end or reset
            start_time = sequence[start]["camera_sensor_timestamp_ms"]
            last_time = sequence[last]["camera_sensor_timestamp_ms"]
            end_time = None if right_censored else sequence[index]["camera_sensor_timestamp_ms"]
            runs.append({"first_missing_frame_id": sequence[start]["frame_id"],
                         "last_missing_frame_id": sequence[last]["frame_id"],
                         "next_success_frame_id": None if right_censored else sequence[index]["frame_id"],
                         "first_missing_sensor_ms": start_time,
                         "last_missing_sensor_ms": last_time, "next_success_sensor_ms": end_time,
                         "missing_stored_frame_count": index-start,
                         "duration_s": None if right_censored else (end_time-start_time)/1000,
                         "observed_missing_span_s": (last_time-start_time)/1000,
                         "left_censored": start == 0 or (start > 0 and sequence[start].get("sensor_clock_segment", 0) != sequence[start-1].get("sensor_clock_segment", 0)),
                         "right_censored": right_censored, "sampled_approximation": True})
            start = None
        if missing and start is None:
            start = index
    return runs


def evaluate_continuity(frames, predictions):
    _index_frames(frames)
    groups = _groups(frames)
    if set(predictions) != {f["frame_id"] for f in frames}:
        raise ContractError("Whole stored-frame denominator requires a prediction record for every frame")
    result = {"status": "VERIFIED_COMPLETE", "evidence_kind": "inference",
              "denominator_kind": "all_verified_stored_frames", "stored_frame_count": len(frames),
              "human_verified_in_view_rate": None, "sessions": {},
              "note": "Camera observation gaps and failed stored outputs are separate; no unsaved frame was generated."}
    for session, seq in groups.items():
        camera_numbers = Counter((f.get("sensor_clock_segment", 0), f.get("camera_frame_number")) for f in seq)
        timestamps = Counter((f.get("sensor_clock_segment", 0), f["camera_sensor_timestamp_ms"]) for f in seq)
        observations = set(camera_numbers)
        segment_groups = defaultdict(list)
        camera_timestamp_sets = defaultdict(set)
        for f in seq:
            segment_groups[f.get("sensor_clock_segment", 0)].append(f)
            camera_timestamp_sets[(f.get("sensor_clock_segment", 0), f.get("camera_frame_number"))].add(f["camera_sensor_timestamp_ms"])
        deltas, gaps, regressions = [], [], []
        for before, after in zip(seq, seq[1:]):
            if before.get("sensor_clock_segment", 0) != after.get("sensor_clock_segment", 0):
                continue
            delta_s = (after["camera_sensor_timestamp_ms"] - before["camera_sensor_timestamp_ms"]) / 1000
            deltas.append(delta_s)
            bn, an = before.get("camera_frame_number"), after.get("camera_frame_number")
            if isinstance(bn, int) and isinstance(an, int) and an - bn > 1:
                gaps.append({"before_frame_id": before["frame_id"], "after_frame_id": after["frame_id"],
                             "before_sensor_ms": before["camera_sensor_timestamp_ms"], "after_sensor_ms": after["camera_sensor_timestamp_ms"],
                             "sensor_time_separation_s": delta_s, "unrecorded_camera_number_count": an-bn-1})
            elif isinstance(bn, int) and isinstance(an, int) and an < bn:
                regressions.append({"before_frame_id": before["frame_id"], "after_frame_id": after["frame_id"]})
        session_result = {"stored_frame_count": len(seq), "sensor_span_s": (seq[-1]["camera_sensor_timestamp_ms"]-seq[0]["camera_sensor_timestamp_ms"])/1000 if len(segment_groups) == 1 else None,
                          "sensor_clock_segment_count": len(segment_groups),
                          "sensor_clock_segments": [{"segment_id": segment, "stored_frame_count": len(sf), "sensor_span_s": (sf[-1]["camera_sensor_timestamp_ms"]-sf[0]["camera_sensor_timestamp_ms"])/1000} for segment, sf in segment_groups.items()],
                          "distinct_camera_observation_count": len(observations),
                          "camera_frame_number_with_conflicting_timestamp_count": sum(len(ts) > 1 for ts in camera_timestamp_sets.values()),
                          "duplicate_camera_frame_number_stored_count": sum(n-1 for n in camera_numbers.values()),
                          "duplicate_sensor_timestamp_stored_count": sum(n-1 for n in timestamps.values()),
                          "distinct_inference_execution_count": len({predictions[f["frame_id"]]["inference_execution_id"] for f in seq}),
                          "distinct_new_inference_execution_count": len({predictions[f["frame_id"]]["inference_execution_id"] for f in seq if predictions[f["frame_id"]].get("new_inference_executed") is True}) if all(type(predictions[f["frame_id"]].get("new_inference_executed")) is bool for f in seq) else None,
                          "cached_prediction_record_count": sum(predictions[f["frame_id"]].get("compute_source") == "verified_legacy_cache" for f in seq),
                          "compute_source_stored_frame_counts": dict(Counter(predictions[f["frame_id"]].get("compute_source", "unspecified") for f in seq)),
                          "camera_recording_gaps": gaps, "camera_frame_number_regressions": regressions,
                          "sensor_adjacent_delta_s": describe(deltas), "methods": {}}
        for method in METHODS:
            ps = [predictions[f["frame_id"]]["methods"][method] for f in seq]
            states = [_state(p) for p in ps]
            counts = Counter(states)
            changes = []
            for before, after, pbefore, pafter in zip(seq, seq[1:], ps, ps[1:]):
                if before.get("sensor_clock_segment", 0) != after.get("sensor_clock_segment", 0):
                    continue
                first, second = _pose(pbefore), _pose(pafter)
                if first is None or second is None:
                    continue
                angular = wrap_delta(second["yaw_deg"], first["yaw_deg"])
                changes.append({"before_frame_id": before["frame_id"], "after_frame_id": after["frame_id"],
                                "sensor_delta_s": (after["camera_sensor_timestamp_ms"]-before["camera_sensor_timestamp_ms"])/1000,
                                "x_delta_m": second["x_m"]-first["x_m"], "z_delta_m": second["z_m"]-first["z_m"],
                                "raw_yaw_delta_deg": second["yaw_deg"]-first["yaw_deg"], "yaw_delta_360_wrapped_deg": angular,
                                "includes_held_output": pbefore["pose_state"] == "held" or pafter["pose_state"] == "held"})
            detail_counts = {}
            for name in ("detection_present", "pnp_failed"):
                detail_counts[name+"_true_count"] = sum(p.get(name) is True for p in ps) if all(isinstance(p.get(name), bool) for p in ps) else None
            masks = [p.get("keypoints_mask") for p in ps]
            detail_counts["partial_keypoint_missing_frame_count"] = sum(0 < sum(mask) < 8 for mask in masks) if all(isinstance(mask, list) and len(mask) in (0, 8) and all(isinstance(v, bool) for v in mask) for mask in masks) else None
            session_result["methods"][method] = {"fresh_count": counts["fresh"], "held_count": counts["held"], "no_pose_count": counts["no_pose"],
                "fresh_output_rate_pct": 100*counts["fresh"]/len(seq), **detail_counts,
                "nonfresh_gaps": _missing_runs(seq, states, lambda s: s != "fresh"),
                "no_pose_gaps": _missing_runs(seq, states, lambda s: s == "no_pose"),
                "adjacent_output_change": {"pair_count": len(changes),
                    "abs_x_delta_m": describe([abs(c["x_delta_m"]) for c in changes]),
                    "abs_z_delta_m": describe([abs(c["z_delta_m"]) for c in changes]),
                    "abs_yaw_delta_360_wrapped_deg": describe([abs(c["yaw_delta_360_wrapped_deg"]) for c in changes]),
                    "wrapped_abs_ge90_deg_count": sum(abs(c["yaw_delta_360_wrapped_deg"]) >= 90 for c in changes),
                    "wrapped_abs_ge180_deg_count": sum(abs(c["yaw_delta_360_wrapped_deg"]) >= 180 for c in changes)},
                "adjacent_changes": changes}
        result["sessions"][session] = session_result
    result["methods"] = {method: {"fresh_count": sum(s["methods"][method]["fresh_count"] for s in result["sessions"].values()),
                         "held_count": sum(s["methods"][method]["held_count"] for s in result["sessions"].values()),
                         "no_pose_count": sum(s["methods"][method]["no_pose_count"] for s in result["sessions"].values())} for method in METHODS}
    for value in result["methods"].values():
        value["fresh_output_rate_pct"] = 100*value["fresh_count"]/len(frames) if frames else None
    return result


def _validate_reference(bundle, frames, bindings):
    if bundle.get("schema_version") != "lifter_reference_review_v1" or bundle.get("source_kind") != "human_reviewed":
        raise ContractError("Only the explicitly human-reviewed bundle schema is accepted")
    if bundle.get("bindings") != bindings:
        raise ContractError("Human bundle bindings changed")
    expected = _index_frames(frames)
    seen = set()
    for record in bundle.get("records", []):
        key = (record.get("frame_id"), record.get("review_pass"))
        if key in seen or key[0] not in expected or key[1] not in ("primary", "repeat"):
            raise ContractError("Unknown/duplicate reviewed frame or review pass")
        seen.add(key)
        frame = expected[key[0]]
        if key[1] == "repeat" and not frame.get("repeat_review", False):
            raise ContractError("Repeat frame was not frozen in the review plan")
        for field in ("session_id", "image_sha256", "width", "height"):
            if record.get(field) != frame.get(field):
                raise ContractError("Reference frame binding mismatch: " + field)
        for field in ("plan_sha256", "corner_contract_sha256", "corner_definition_version"):
            if record.get(field) != bindings.get(field):
                raise ContractError("Reference contract binding mismatch: " + field)
        if record.get("source_kind") != "human_reviewed":
            raise ContractError("Unreviewed reference blocked")
        reviewer = record.get("reviewer", {})
        if not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip() or reviewer.get("entered_by") != "human" or reviewer.get("confirmation") is not True:
            raise ContractError("Explicit human reviewer confirmation required")
        for field in ("machine_assistance", "previous_prediction_exposure", "previous_annotation_exposure"):
            if not isinstance(reviewer.get(field), bool):
                raise ContractError("Reviewer exposure answers required")
        times = record.get("review_time", {})
        if times.get("clock_source") != "server_wall_and_monotonic" or not finite(times.get("duration_seconds")) or times["duration_seconds"] < 0:
            raise ContractError("Actual review timing required")
        parsed_times = []
        for field in ("started_at", "finished_at"):
            try:
                stamp = datetime.fromisoformat(times[field].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    raise ValueError("Timezone missing")
                parsed_times.append(stamp)
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError("Review timestamps must be actual timezone-aware ISO times") from exc
        if parsed_times[1] < parsed_times[0]:
            raise ContractError("Review end precedes actual start time")
        if record.get("status") == "skipped":
            if not record.get("skip_reason") or record.get("corners") != []:
                raise ContractError("Skipped review requires reason and no claimed corners")
            continue
        if record.get("status") != "reviewed":
            raise ContractError("Draft record cannot enter evaluation")
        corners = record.get("corners", [])
        ids = [c.get("id") for c in corners]
        if len(ids) != 8 or set(ids) != set(range(8)) or any(type(i) is not int for i in ids):
            raise ContractError("Exactly eight unique canonical corner IDs required")
        obj = record.get("object", {})
        if obj.get("presence") not in ("present", "absent", "uncertain") or not isinstance(obj.get("target_identity_confirmed"), bool):
            raise ContractError("Object presence and identity assessment required")
        for corner in corners:
            vis = corner.get("visibility")
            if vis not in ("direct_visible", "not_direct_visible", "uncertain"):
                raise ContractError("Explicit corner visibility required")
            for field in ("external_occlusion", "self_occlusion", "out_of_frame", "definition_uncertain", "definition_confirmed"):
                if not isinstance(corner.get(field), bool):
                    raise ContractError("Corner attributes must remain separate explicit booleans")
            if vis == "direct_visible":
                if any(corner[k] for k in ("external_occlusion", "self_occlusion", "out_of_frame", "definition_uncertain")) or not corner["definition_confirmed"]:
                    raise ContractError("Direct visible reference has contradictory corner attributes")
                if obj.get("presence") != "present" or obj.get("target_identity_confirmed") is not True or not obj.get("target_object_id"):
                    raise ContractError("Visible corner target identity must be human-confirmed")
                if not finite(corner.get("x")) or not finite(corner.get("y")) or not (0 <= corner["x"] < frame["width"] and 0 <= corner["y"] < frame["height"]):
                    raise ContractError("Visible corner requires finite original-image in-range coordinates")
            elif corner.get("x") is not None or corner.get("y") is not None:
                raise ContractError("Non-visible corner cannot contain an inferred click")
    return bundle["records"]


def _method_errors(record, prediction, permutation):
    visible = [c for c in record["corners"] if c["visibility"] == "direct_visible"]
    if not visible:
        return []
    points = prediction.get("keypoints") if prediction else None
    mask = prediction.get("keypoints_mask") if prediction else None
    if points is None:
        return [(c["id"], None, "missing_prediction", permutation[c["id"]]) for c in visible]
    if not isinstance(points, list) or len(points) != 8 or not isinstance(mask, list) or len(mask) != 8 or any(type(v) is not bool for v in mask):
        raise ContractError("Predicted corners and masks must both have eight canonical slots")
    target = record["object"]["target_object_id"]
    if prediction.get("object_id") is not None:
        matches = prediction["object_id"] == target
    elif isinstance(prediction.get("object_match"), bool) and prediction.get("object_match_source_kind") == "human_reviewed":
        matches = prediction["object_match"]
    else:
        raise ContractError("Predicted object correspondence has not been independently confirmed")
    result = []
    for corner in visible:
        reference_id = corner["id"]
        source_id = permutation[reference_id]
        point = points[source_id]
        valid = mask[source_id] and isinstance(point, (list, tuple)) and len(point) == 2 and all(finite(v) for v in point)
        if not matches:
            error, reason = None, "wrong_object"
        elif not valid:
            error, reason = None, "missing_corner"
        else:
            error, reason = math.hypot(point[0]-corner["x"], point[1]-corner["y"]), None
        result.append((reference_id, error, reason, source_id))
    return result


def _corner_stats(rows, frame_count):
    errors = [r["error_px"] for r in rows if r["error_px"] is not None]
    count = len(rows)
    return {"reviewed_frame_count": frame_count, "visible_reference_corner_count": count,
            "valid_predicted_corner_count": len(errors), "pck10_success_corner_count": sum(e <= 10 for e in errors),
            "pck10_pct": 100*sum(e <= 10 for e in errors)/count if count else None,
            "conditional_error_median_px": percentile(errors, .5), "conditional_error_p90_px": percentile(errors, .9),
            "failure_counts": dict(Counter(r["failure_reason"] for r in rows if r["failure_reason"]))}


def evaluate_visible_corners(frames, bundle, predictions, bindings, *, mode="canonical", allowed_permutations=None):
    records = _validate_reference(bundle, frames, bindings)
    if mode == "canonical":
        if allowed_permutations not in (None, [list(IDENTITY)], [IDENTITY]):
            raise ContractError("Canonical ID evaluation cannot mix symmetry permutations")
        permutations = [IDENTITY]
    elif mode == "frozen_whole_permutation":
        if not allowed_permutations:
            raise ContractError("Whole-object permutation set must be frozen before evaluation")
        permutations = [tuple(p) for p in allowed_permutations]
        if any(len(p) != 8 or set(p) != set(IDENTITY) or any(type(i) is not int for i in p) for p in permutations):
            raise ContractError("Only whole eight-ID bijections are allowed")
    else:
        raise ContractError("Unknown corner evaluation mode")
    primary = [r for r in records if r["review_pass"] == "primary" and r["status"] == "reviewed"]
    rows = []
    for record in primary:
        row = predictions.get(record["frame_id"])
        if row is None:
            raise ContractError("Approved sample lacks a prediction execution record")
        for method in METHODS:
            pred = row["methods"][method]
            candidates = [_method_errors(record, pred, p) for p in permutations]
            # One complete permutation is chosen for the entire object. Masks and
            # coordinate source IDs travel together; no per-point nearest match.
            def score(candidate):
                errors = [c[1] for c in candidate if c[1] is not None]
                return (-len(errors), sum(errors)/len(errors) if errors else math.inf)
            chosen_index = min(range(len(candidates)), key=lambda i: (score(candidates[i]), i))
            for corner_id, error, reason, source_id in candidates[chosen_index]:
                rows.append({"frame_id": record["frame_id"], "session_id": record["session_id"], "method": method,
                             "reference_corner_id": corner_id, "prediction_source_corner_id": source_id,
                             "whole_permutation_index": chosen_index, "error_px": error,
                             "pck10_success": error is not None and error <= 10, "failure_reason": reason})
    result = {"status": "VERIFIED_COMPLETE" if len(primary) == len(frames) else "WAITING_HUMAN",
              "evidence_kind": "human_reviewed", "description": "Fixed sensor-time sample visible-corner accuracy",
              "mode": mode, "allowed_whole_permutations": [list(p) for p in permutations],
              "planned_frame_count": len(frames), "reviewed_frame_count": len(primary),
              "skipped_primary_frame_count": sum(r["status"] == "skipped" and r["review_pass"] == "primary" for r in records),
              "pending_primary_frame_count": len(frames)-len(primary), "sessions": {}, "methods": {}, "corner_rows": rows}
    for method in METHODS:
        result["methods"][method] = _corner_stats([r for r in rows if r["method"] == method], len(primary))
    if not rows:
        result["status"] = "BLOCKED_REFERENCE"
        result["reason"] = "No approved directly visible reference corners; accuracy remains null."
    sessions = sorted({f["session_id"] for f in frames})
    for session in sessions:
        selected = [r for r in primary if r["session_id"] == session]
        sr = {m: _corner_stats([r for r in rows if r["method"] == m and r["session_id"] == session], len(selected)) for m in METHODS}
        result["sessions"][session] = {"methods": sr}
    indexed = {(r["frame_id"], r["reference_corner_id"], r["method"]): r for r in rows}
    differences = []
    for record in primary:
        for corner in record["corners"]:
            key = (record["frame_id"], corner["id"])
            base, n3 = indexed.get((*key, "Base")), indexed.get((*key, "N3"))
            if base and n3 and base["error_px"] is not None and n3["error_px"] is not None:
                differences.append({"frame_id": key[0], "session_id": record["session_id"], "corner_id": key[1],
                                    "n3_minus_base_error_px": n3["error_px"]-base["error_px"]})
    def paired(session=None):
        dr = differences if session is None else [d for d in differences if d["session_id"] == session]
        stats = result["methods"] if session is None else result["sessions"][session]["methods"]
        before, after = stats["Base"]["conditional_error_median_px"], stats["N3"]["conditional_error_median_px"]
        return {"paired_valid_corner_count": len(dr), "median_n3_minus_base_same_corner_px": percentile([d["n3_minus_base_error_px"] for d in dr], .5),
                "median_n3_minus_median_base_px": after-before if before is not None and after is not None else None}
    result["paired_difference"] = paired()
    for session in sessions:
        result["sessions"][session]["paired_difference"] = paired(session)
    result["paired_corner_rows"] = differences
    return result


def evaluate_repeat_quality(frames, bundle, bindings):
    records = _validate_reference(bundle, frames, bindings)
    primary = {r["frame_id"]: r for r in records if r["review_pass"] == "primary" and r["status"] == "reviewed"}
    differences, status_mismatch, identity_mismatch = [], 0, 0
    relation = Counter()
    for repeat in records:
        if repeat["review_pass"] != "repeat" or repeat["status"] != "reviewed" or repeat["frame_id"] not in primary:
            continue
        before = primary[repeat["frame_id"]]
        relation["same_person_repeat" if before["reviewer"]["id"] == repeat["reviewer"]["id"] else "different_people"] += 1
        identity_mismatch += before["object"] != repeat["object"]
        corners = {c["id"]: c for c in before["corners"]}
        for after in repeat["corners"]:
            first = corners[after["id"]]
            fields = ("visibility", "external_occlusion", "self_occlusion", "out_of_frame", "definition_uncertain", "definition_confirmed")
            status_mismatch += any(first[k] != after[k] for k in fields)
            if first["visibility"] == after["visibility"] == "direct_visible":
                differences.append(math.hypot(after["x"]-first["x"], after["y"]-first["y"]))
    return {"status": "VERIFIED_COMPLETE" if sum(relation.values()) == sum(f.get("repeat_review", False) for f in frames) else "WAITING_HUMAN",
            "paired_review_frame_count": sum(relation.values()), "reviewer_relation_counts": dict(relation),
            "repeated_direct_visible_corner_click_difference_px": describe(differences),
            "corner_status_mismatch_count": status_mismatch, "object_identity_mismatch_count": identity_mismatch,
            "note": "No model errors or arbitrary pixel acceptance threshold were used to select repeated clicks."}


def evaluate_stop_intervals(frames, predictions, reviewed_intervals, bindings=None):
    if reviewed_intervals is None:
        return {"status": "WAITING_HUMAN", "intervals": None, "std_x_m": None, "std_z_m": None, "std_yaw_deg": None}
    if reviewed_intervals.get("source_kind") != "human_reviewed" or reviewed_intervals.get("fixed_before_predictions") is not True:
        raise ContractError("Stop candidates must be human-reviewed and frozen before viewing model predictions")
    if bindings is None or reviewed_intervals.get("plan_sha256") != bindings.get("plan_sha256"):
        raise ContractError("Reviewed stop intervals must bind to the frozen frame plan")
    if reviewed_intervals.get("confirmed_no_stop_intervals") is True:
        if reviewed_intervals.get("intervals"):
            raise ContractError("No-stop confirmation cannot contain stop intervals")
        return {"status": "VERIFIED_COMPLETE", "intervals": "NA", "std_x_m": "NA", "std_z_m": "NA", "std_yaw_deg": "NA"}
    result = []
    for interval in reviewed_intervals.get("intervals", []):
        if interval.get("source_kind") != "human_reviewed" or not interval.get("reviewer_id") or not interval.get("reviewed_at") or not interval.get("evidence"):
            raise ContractError("Actual stop review identity, time, and visual/physical evidence required")
        start, end = interval.get("sensor_start_ms"), interval.get("sensor_end_ms")
        if not finite(start) or not finite(end) or end < start:
            raise ContractError("Invalid reviewed stop sensor interval")
        selected = [f for f in frames if f["session_id"] == interval["session_id"] and start <= f["camera_sensor_timestamp_ms"] <= end]
        methods = {}
        for method in METHODS:
            output_states = [_state(predictions[f["frame_id"]]["methods"][method]) for f in selected]
            poses = [_pose(predictions[f["frame_id"]]["methods"][method]) for f in selected]
            poses = [p for p in poses if p is not None]
            angles = []
            for p in poses:
                angles.append(p["yaw_deg"] if not angles else angles[-1]+wrap_delta(p["yaw_deg"], angles[-1]))
            methods[method] = {"valid_output_frame_count": len(poses),
                "fresh_output_frame_count": output_states.count("fresh"), "held_output_frame_count": output_states.count("held"),
                "x_std_m": statistics.pstdev([p["x_m"] for p in poses]) if poses else None,
                "z_std_m": statistics.pstdev([p["z_m"] for p in poses]) if poses else None,
                "yaw_std_360_unwrapped_deg": statistics.pstdev(angles) if angles else None,
                "note": "Only 360 wrapping is removed; raw 90/180 branches remain in the variation."}
        result.append({"interval_id": interval["interval_id"], "session_id": interval["session_id"],
                       "duration_s": (end-start)/1000, "stored_frame_count": len(selected), "methods": methods})
    return {"status": "VERIFIED_COMPLETE", "intervals": result}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_predictions(path):
    if str(path).lower().endswith(".jsonl"):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    value = _read_json(path)
    return value if isinstance(value, list) else value["records"]


def _load_review_validator(path):
    """Optional authoritative review-tool validation, with no control imports."""
    spec = importlib.util.spec_from_file_location("lifter_review_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_legacy_bindings():
    """Read constants via AST only; never execute/import legacy inference code."""
    path = Path(__file__).resolve().parent.parent / "evaluation_checkout/scripts/research/pallet_n3_completion_v3/lifter_run.py"
    if sha256(path) != FROZEN_LIFTER_RUN_SHA256:
        raise ContractError("Frozen source binding declaration changed")
    values = {}
    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(value(n) for n in node.elts)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1 and not node.keywords:
            return str(Path(value(node.args[0])))
        raise ContractError("Unexpected expression in frozen binding declaration")
    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in ("R0_RELATIVE", "LEGACY_BINDINGS"):
                values[name] = value(statement.value)
    return [{"name": name, "path": path, "sha256": digest, "bytes": size} for name, path, digest, size in values["LEGACY_BINDINGS"]]


def validate_run_identity(identity, plan_hash, source_commit, expected_bindings):
    if identity.get("plan_sha256") != plan_hash or identity.get("source_commit") != source_commit:
        raise ContractError("Prediction run identity differs from frozen source/plan")
    if identity.get("fixed_bindings") != expected_bindings:
        raise ContractError("Prediction run weights or frozen settings differ from original bindings")
    if identity.get("pass_budget_s") != 3600 or identity.get("independent_reference") is not False:
        raise ContractError("Prediction run resource/reference guard changed")


def run_evaluation(manifest_path, plan_path, contract_path, output_dir, *, reviewed_path=None, predictions_path=None, stop_intervals_path=None):
    manifest, plan, contract = _read_json(manifest_path), _read_json(plan_path), _read_json(contract_path)
    if manifest["plan_sha256"] != sha256(plan_path) or manifest["corner_contract_sha256"] != sha256(contract_path):
        raise ContractError("Manifest plan/corner hashes changed")
    bindings = {"manifest_sha256": sha256(manifest_path), "plan_sha256": sha256(plan_path),
                "corner_contract_sha256": sha256(contract_path), "corner_definition_version": contract["version"]}
    planned = _index_frames(plan["frames"])
    for sample in manifest["frames"]:
        full = planned.get(sample["frame_id"])
        if full is None or any(sample.get(k) != full.get(k) for k in ("session_id", "saved_frame_index", "camera_sensor_timestamp_ms", "decoded_bgr_sha256", "width", "height")):
            raise ContractError("Review frame identity differs from the frozen all-stored-frame plan")
    validator_path = Path(__file__).resolve().parent.parent / "review" / "serve.py"
    validator = _load_review_validator(validator_path)
    context = validator.Context(manifest_path, plan_path, contract_path,
        reviewed_path or Path(output_dir) / "metrics_validation_unwritten_reference.json")
    report = {"schema_version": "lifter_case_metrics_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "bindings": bindings, "statuses": {}, "blockers": [], "fixed_sample_visible_corner_accuracy": None,
              "whole_stored_frame_continuity": None, "repeat_annotation_quality": None,
              "reviewed_stop_variation": {"status": "WAITING_HUMAN", "intervals": None},
              "independent_physical_accuracy": {"status": "BLOCKED_REFERENCE", "x_error_m": None, "z_error_m": None, "yaw_error_deg": None,
                   "reason": "Operational model CSV and click-derived PnP are not independent physical reference."}}
    report["input_file_sha256"] = {"manifest": sha256(manifest_path), "plan": sha256(plan_path), "corner_contract": sha256(contract_path)}
    bundle = None
    if reviewed_path and Path(reviewed_path).is_file():
        bundle = _read_json(reviewed_path)
        context.validate_bundle(bundle, allow_drafts=False)
        report["input_file_sha256"]["reviewed_reference"] = sha256(reviewed_path)
        _validate_reference(bundle, manifest["frames"], bindings)
        report["repeat_annotation_quality"] = evaluate_repeat_quality(manifest["frames"], bundle, bindings)
        reviewed_count = sum(r["review_pass"] == "primary" and r["status"] == "reviewed" for r in bundle["records"])
        report["statuses"]["human_reference"] = "VERIFIED_COMPLETE" if reviewed_count == len(manifest["frames"]) else "WAITING_HUMAN"
    else:
        report["statuses"]["human_reference"] = "WAITING_HUMAN"
        report["blockers"].append("Approved human reference file not supplied; visible-corner accuracy remains null.")
    predictions = None
    if predictions_path and Path(predictions_path).is_file():
        identity_path = Path(predictions_path).parent / "RUN_IDENTITY.json"
        if not identity_path.is_file():
            raise ContractError("Prediction file lacks its frozen RUN_IDENTITY.json")
        run_identity = _read_json(identity_path)
        validate_run_identity(run_identity, bindings["plan_sha256"], plan["source_commit"], frozen_legacy_bindings())
        report["input_file_sha256"]["prediction_run_identity"] = sha256(identity_path)
        report["input_file_sha256"]["predictions"] = sha256(predictions_path)
        predictions = index_predictions(_read_predictions(predictions_path), plan["frames"])
        report["prediction_record_count"] = len(predictions)
        try:
            report["whole_stored_frame_continuity"] = evaluate_continuity(plan["frames"], predictions)
            report["statuses"]["all_stored_predictions"] = "VERIFIED_COMPLETE"
        except ContractError as exc:
            report["statuses"]["all_stored_predictions"] = "BLOCKED_CONTRACT"
            report["blockers"].append(str(exc))
        if bundle:
            try:
                report["fixed_sample_visible_corner_accuracy"] = evaluate_visible_corners(manifest["frames"], bundle, predictions, bindings)
                report["statuses"]["visible_corner_accuracy"] = report["fixed_sample_visible_corner_accuracy"]["status"]
            except ContractError as exc:
                report["statuses"]["visible_corner_accuracy"] = "BLOCKED_CONTRACT"
                report["blockers"].append(str(exc))
        if stop_intervals_path:
            report["reviewed_stop_variation"] = evaluate_stop_intervals(plan["frames"], predictions, _read_json(stop_intervals_path), bindings)
    else:
        report["statuses"]["all_stored_predictions"] = "BLOCKED_CONTRACT"
        report["statuses"]["visible_corner_accuracy"] = "WAITING_HUMAN" if bundle is None else "BLOCKED_CONTRACT"
        report["prediction_record_count"] = 0
        report["blockers"].append("Frozen Base/N3 prediction file not supplied; no output or accuracy values invented.")
    report["status"] = next((status for status in ("BLOCKED_CONTRACT", "WAITING_HUMAN", "BLOCKED_REFERENCE") if status in report["statuses"].values()), "VERIFIED_COMPLETE")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "LIFTER_METRICS.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    corner_rows = report["fixed_sample_visible_corner_accuracy"]["corner_rows"] if report["fixed_sample_visible_corner_accuracy"] else []
    fields = ("frame_id", "session_id", "method", "reference_corner_id", "prediction_source_corner_id", "whole_permutation_index", "error_px", "pck10_success", "failure_reason")
    with (target / "LIFTER_CORNER_ERRORS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(corner_rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--corner-contract", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--stop-intervals", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(args.manifest, args.plan, args.corner_contract, args.output_dir,
            reviewed_path=args.reviewed, predictions_path=args.predictions, stop_intervals_path=args.stop_intervals)
    except (ContractError, KeyError, ValueError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED_CONTRACT", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": report["status"], "statuses": report["statuses"], "output": str(args.output_dir / "LIFTER_METRICS.json")}, ensure_ascii=False))
    return 0 if report["status"] == "VERIFIED_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
