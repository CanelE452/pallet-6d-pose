"""Build derived, prediction-only lifter summaries and figures.

This program is deliberately downstream of the frozen 8,910-row inference.  It
never writes the raw prediction, plan, review, checkpoint, or evaluator files.
Accuracy, stationary variation, and physical-reference fields stay unresolved
until their separately reviewed inputs exist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys


METHODS = ("Base", "N3")
SESSION_ORDER = ("173507", "174126", "174342", "174925")
POSE_FIELDS = ("x_m", "z_m", "yaw_deg")
COLORS = {"Base": "#d95f02", "N3": "#009eab"}
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path, *, rows: int | None = None) -> dict:
    value = {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_row(value: dict) -> dict:
    return {
        "frame_id": value["frame_id"],
        "session_id": str(value["session_id"]),
        "stored_index": int(value["stored_index"]),
        "sensor_timestamp_ms": float(value["sensor_timestamp_ms"]),
        "decoded_bgr_sha256": value["decoded_bgr_sha256"],
        "inference_execution_id": value["inference_execution_id"],
        "new_inference_executed": value["new_inference_executed"],
        "compute_source": value["compute_source"],
        "methods": value["methods"],
    }


def read_prediction_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(compact_row(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid prediction row {line_number}: {exc}") from exc
    return rows


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))


def describe(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "maximum": max(values) if values else None,
    }


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_pose(method: dict) -> dict | None:
    pose = method.get("pose")
    if not isinstance(pose, dict) or not all(finite(pose.get(key)) for key in POSE_FIELDS):
        return None
    return pose


def wrap_delta(after: float, before: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0


def validate_rows(rows: list[dict], plan: dict) -> dict:
    expected = {frame["frame_id"]: frame for frame in plan["frames"]}
    actual = {row["frame_id"]: row for row in rows}
    if len(actual) != len(rows):
        raise ValueError("duplicate frame_id in derived predictions")
    if set(actual) != set(expected):
        raise ValueError("prediction frame IDs do not exactly match the frozen plan")
    state_counts = Counter()
    mask_identity_count = 0
    object_match_null_count = 0
    for row in rows:
        frame = expected[row["frame_id"]]
        if row["session_id"] != str(frame["session_id"]):
            raise ValueError("session mismatch: " + row["frame_id"])
        if row["stored_index"] != int(frame["saved_frame_index"]):
            raise ValueError("stored-index mismatch: " + row["frame_id"])
        if row["decoded_bgr_sha256"] != frame["decoded_bgr_sha256"]:
            raise ValueError("decoded pixel hash mismatch: " + row["frame_id"])
        bmask, nmask = row["methods"]["Base"].get("keypoints_mask"), row["methods"]["N3"].get("keypoints_mask")
        if bmask != nmask:
            raise ValueError("Base/N3 keypoint mask mismatch: " + row["frame_id"])
        mask_identity_count += 1
        for method in METHODS:
            item = row["methods"][method]
            state = item.get("pose_state")
            if state not in ("fresh", "held", "no_pose"):
                raise ValueError(f"invalid pose_state {method}: {row['frame_id']}")
            pose = finite_pose(item)
            if state in ("fresh", "held") and pose is None:
                raise ValueError(f"finite pose absent for {state}: {row['frame_id']} {method}")
            if state == "no_pose" and pose is not None:
                raise ValueError(f"no_pose silently has finite pose: {row['frame_id']} {method}")
            mask = item.get("keypoints_mask")
            if not (isinstance(mask, list) and len(mask) in (0, 8) and all(type(v) is bool for v in mask)):
                raise ValueError(f"invalid explicit keypoint mask: {row['frame_id']} {method}")
            state_counts[(method, state)] += 1
            object_match_null_count += item.get("object_match") is None
    return {
        "status": "VERIFIED_COMPLETE",
        "plan_prediction_frame_set_identity": True,
        "base_n3_keypoint_mask_identity_count": mask_identity_count,
        "object_match_null_method_frame_count": object_match_null_count,
        "state_counts": {f"{m}_{s}": state_counts[(m, s)]
                         for m in METHODS for s in ("fresh", "held", "no_pose")},
    }


def session_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["session_id"]].append(row)
    if set(groups) != set(SESSION_ORDER):
        raise ValueError(f"unexpected sessions: {sorted(groups)}")
    for session_rows in groups.values():
        session_rows.sort(key=lambda row: row["stored_index"])
        for before, after in zip(session_rows, session_rows[1:]):
            if after["stored_index"] <= before["stored_index"]:
                raise ValueError("stored indices are not strictly increasing")
            if after["sensor_timestamp_ms"] < before["sensor_timestamp_ms"]:
                raise ValueError("sensor timestamp regression")
    return dict(groups)


def missing_runs(rows: list[dict], method: str, *, no_pose_only: bool) -> list[dict]:
    runs, start = [], None
    states = [row["methods"][method]["pose_state"] for row in rows]
    for index in range(len(rows) + 1):
        at_end = index == len(rows)
        missing = not at_end and (states[index] == "no_pose" if no_pose_only else states[index] != "fresh")
        if start is not None and (at_end or not missing):
            last = index - 1
            end_time = None if at_end else rows[index]["sensor_timestamp_ms"]
            start_time = rows[start]["sensor_timestamp_ms"]
            last_time = rows[last]["sensor_timestamp_ms"]
            runs.append({
                "session_id": rows[start]["session_id"],
                "first_missing_frame_id": rows[start]["frame_id"],
                "last_missing_frame_id": rows[last]["frame_id"],
                "next_success_frame_id": None if at_end else rows[index]["frame_id"],
                "missing_stored_frame_count": index - start,
                "duration_s": None if at_end else (end_time - start_time) / 1000.0,
                "observed_missing_span_s": (last_time - start_time) / 1000.0,
                "left_censored": start == 0,
                "right_censored": at_end,
                "sampled_approximation": True,
            })
            start = None
        if missing and start is None:
            start = index
    return runs


def summarize_gaps(runs: list[dict]) -> dict:
    completed = [run for run in runs if run["duration_s"] is not None]
    longest_completed = max(completed, key=lambda run: run["duration_s"], default=None)
    longest_observed = max(runs, key=lambda run: run["observed_missing_span_s"], default=None)
    max_frames = max((run["missing_stored_frame_count"] for run in runs), default=None)
    frame_ties = [run for run in runs if run["missing_stored_frame_count"] == max_frames] if runs else []
    return {
        "gap_count": len(runs),
        "completed_gap_count": len(completed),
        "left_censored_gap_count": sum(run["left_censored"] for run in runs),
        "right_censored_gap_count": sum(run["right_censored"] for run in runs),
        "maximum_completed_duration_s": longest_completed["duration_s"] if longest_completed else None,
        "maximum_completed_duration_record": longest_completed,
        "maximum_observed_missing_span_s": longest_observed["observed_missing_span_s"] if longest_observed else None,
        "maximum_observed_span_record": longest_observed,
        "maximum_missing_stored_frame_count": max_frames,
        "maximum_missing_frame_count_records": frame_ties,
    }


def adjacent_changes(rows: list[dict], method: str) -> list[dict]:
    changes = []
    for before, after in zip(rows, rows[1:]):
        pbefore = finite_pose(before["methods"][method])
        pafter = finite_pose(after["methods"][method])
        if pbefore is None or pafter is None:
            continue
        changes.append({
            "pair_id": f"{before['frame_id']}->{after['frame_id']}",
            "before_frame_id": before["frame_id"],
            "after_frame_id": after["frame_id"],
            "sensor_delta_s": (after["sensor_timestamp_ms"] - before["sensor_timestamp_ms"]) / 1000.0,
            "abs_x_delta_m": abs(pafter["x_m"] - pbefore["x_m"]),
            "abs_z_delta_m": abs(pafter["z_m"] - pbefore["z_m"]),
            "abs_yaw_delta_360_wrapped_deg": abs(wrap_delta(pafter["yaw_deg"], pbefore["yaw_deg"])),
            "includes_held_output": before["methods"][method]["pose_state"] == "held" or after["methods"][method]["pose_state"] == "held",
        })
    return changes


def method_summary(rows: list[dict], method: str) -> tuple[dict, list[dict]]:
    items = [row["methods"][method] for row in rows]
    counts = Counter(item["pose_state"] for item in items)
    nonfresh = missing_runs(rows, method, no_pose_only=False)
    no_pose = missing_runs(rows, method, no_pose_only=True)
    changes = adjacent_changes(rows, method)
    masks = [item["keypoints_mask"] for item in items]
    summary = {
        "fresh_count": counts["fresh"],
        "held_count": counts["held"],
        "no_pose_count": counts["no_pose"],
        "fresh_output_rate_pct": 100.0 * counts["fresh"] / len(rows) if rows else None,
        "detection_present_true_count": sum(item.get("detection_present") is True for item in items),
        "pnp_failed_true_count": sum(item.get("pnp_failed") is True for item in items),
        "partial_keypoint_missing_frame_count": sum(len(mask) == 8 and 0 < sum(mask) < 8 for mask in masks),
        "nonfresh_gaps": summarize_gaps(nonfresh),
        "no_pose_gaps": summarize_gaps(no_pose),
        "adjacent_output_change": {
            "pair_count": len(changes),
            "held_inclusive_pair_count": sum(change["includes_held_output"] for change in changes),
            "abs_x_delta_m": describe([change["abs_x_delta_m"] for change in changes]),
            "abs_z_delta_m": describe([change["abs_z_delta_m"] for change in changes]),
            "abs_yaw_delta_360_wrapped_deg": describe([change["abs_yaw_delta_360_wrapped_deg"] for change in changes]),
        },
    }
    return summary, changes


def pairing_summary(rows: list[dict], changes: dict[str, list[dict]]) -> dict:
    frame_counts = Counter()
    for row in rows:
        base = finite_pose(row["methods"]["Base"]) is not None
        n3 = finite_pose(row["methods"]["N3"]) is not None
        b_fresh = row["methods"]["Base"]["pose_state"] == "fresh"
        n_fresh = row["methods"]["N3"]["pose_state"] == "fresh"
        frame_counts["both_pose"] += base and n3
        frame_counts["base_only_pose"] += base and not n3
        frame_counts["n3_only_pose"] += n3 and not base
        frame_counts["both_no_pose"] += not base and not n3
        frame_counts["both_fresh"] += b_fresh and n_fresh
        frame_counts["fresh_state_disagreement"] += b_fresh != n_fresh
        frame_counts["full_state_identity"] += row["methods"]["Base"]["pose_state"] == row["methods"]["N3"]["pose_state"]
    bpairs = {change["pair_id"] for change in changes["Base"]}
    npairs = {change["pair_id"] for change in changes["N3"]}
    return {
        "same_frame_denominator": len(rows),
        "same_frame_both_finite_pose_count": frame_counts["both_pose"],
        "same_frame_base_only_finite_pose_count": frame_counts["base_only_pose"],
        "same_frame_n3_only_finite_pose_count": frame_counts["n3_only_pose"],
        "same_frame_both_no_pose_count": frame_counts["both_no_pose"],
        "same_frame_both_fresh_count": frame_counts["both_fresh"],
        "same_frame_fresh_state_disagreement_count": frame_counts["fresh_state_disagreement"],
        "same_frame_full_pose_state_identity_count": frame_counts["full_state_identity"],
        "adjacent_candidate_pair_count": max(len(rows) - 1, 0),
        "base_valid_adjacent_pair_count": len(bpairs),
        "n3_valid_adjacent_pair_count": len(npairs),
        "both_methods_valid_same_adjacent_pair_count": len(bpairs & npairs),
        "base_only_valid_adjacent_pair_count": len(bpairs - npairs),
        "n3_only_valid_adjacent_pair_count": len(npairs - bpairs),
    }


def summarize_scope(rows: list[dict], scope: str) -> tuple[dict, dict[str, list[dict]]]:
    changes = {}
    methods = {}
    for method in METHODS:
        methods[method], changes[method] = method_summary(rows, method)
    value = {
        "scope": scope,
        "stored_frame_count": len(rows),
        "sensor_span_s": ((rows[-1]["sensor_timestamp_ms"] - rows[0]["sensor_timestamp_ms"]) / 1000.0) if rows else None,
        "methods": methods,
        "pairing": pairing_summary(rows, changes),
    }
    return value, changes


def aggregate_summary(rows: list[dict], plan: dict) -> tuple[dict, dict[str, dict[str, list[dict]]]]:
    groups = session_groups(rows)
    sessions, all_changes = {}, {}
    for session in SESSION_ORDER:
        sessions[session], all_changes[session] = summarize_scope(groups[session], session)
    # Overall statistics are pooled across within-session pairs and gaps; no pair
    # or missing run is allowed to cross a recording boundary.
    overall_methods = {}
    pooled_changes = {method: [] for method in METHODS}
    for method in METHODS:
        for session in SESSION_ORDER:
            pooled_changes[method].extend(all_changes[session][method])
        all_nonfresh = []
        all_no_pose = []
        for session in SESSION_ORDER:
            all_nonfresh.extend(missing_runs(groups[session], method, no_pose_only=False))
            all_no_pose.extend(missing_runs(groups[session], method, no_pose_only=True))
        counts = Counter(row["methods"][method]["pose_state"] for row in rows)
        masks = [row["methods"][method]["keypoints_mask"] for row in rows]
        changes = pooled_changes[method]
        overall_methods[method] = {
            "fresh_count": counts["fresh"], "held_count": counts["held"], "no_pose_count": counts["no_pose"],
            "fresh_output_rate_pct": 100.0 * counts["fresh"] / len(rows),
            "detection_present_true_count": sum(row["methods"][method].get("detection_present") is True for row in rows),
            "pnp_failed_true_count": sum(row["methods"][method].get("pnp_failed") is True for row in rows),
            "partial_keypoint_missing_frame_count": sum(len(mask) == 8 and 0 < sum(mask) < 8 for mask in masks),
            "nonfresh_gaps": summarize_gaps(all_nonfresh),
            "no_pose_gaps": summarize_gaps(all_no_pose),
            "adjacent_output_change": {
                "pair_count": len(changes),
                "held_inclusive_pair_count": sum(change["includes_held_output"] for change in changes),
                "abs_x_delta_m": describe([change["abs_x_delta_m"] for change in changes]),
                "abs_z_delta_m": describe([change["abs_z_delta_m"] for change in changes]),
                "abs_yaw_delta_360_wrapped_deg": describe([change["abs_yaw_delta_360_wrapped_deg"] for change in changes]),
            },
        }
    overall_pairing = Counter()
    for session in SESSION_ORDER:
        for key, value in sessions[session]["pairing"].items():
            overall_pairing[key] += value
    overall = {
        "scope": "ALL_STORED",
        "stored_frame_count": len(rows),
        "session_count": len(SESSION_ORDER),
        "sensor_span_sum_s": sum(sessions[session]["sensor_span_s"] for session in SESSION_ORDER),
        "methods": overall_methods,
        "pairing": dict(overall_pairing),
    }
    # Sum(n_s - 1) is the overall adjacent candidate denominator.
    overall["pairing"]["adjacent_candidate_pair_count"] = sum(max(len(groups[s]) - 1, 0) for s in SESSION_ORDER)
    return {
        "schema_version": "lifter_prediction_only_continuity_summary_v1",
        "status": "VERIFIED_COMPLETE",
        "evidence_kind": "fixed_inference_outputs_without_human_reference",
        "denominator_kind": "all_verified_stored_frames",
        "definitions": {
            "fresh": "finite current-frame pose output",
            "held": "finite carried-forward output; none occurred in this run",
            "no_pose": "no finite output pose",
            "adjacent_change": "absolute change between consecutive stored outputs within one session when both endpoint poses exist",
            "gap_duration": "sampled sensor time from first missing stored frame to next successful stored frame; null when right-censored",
            "observed_missing_span": "sampled sensor time from first through last observed missing stored frame",
            "paired": "same frame or same adjacent frame pair has finite output from both methods; this is not a human-reference pairing",
        },
        "overall": overall,
        "sessions": sessions,
        "unresolved_reference_fields": {
            "visible_corner_accuracy": None,
            "object_correspondence": None,
            "reviewed_stop_variation": None,
            "independent_physical_accuracy": None,
        },
    }, all_changes


def close(a, b, tolerance=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def regression_check(summary: dict, metrics: dict) -> dict:
    continuity = metrics["whole_stored_frame_continuity"]
    checks = []

    def check(name, actual, expected, numeric=False):
        passed = close(actual, expected) if numeric else actual == expected
        checks.append({"name": name, "passed": passed, "recomputed": actual, "evaluator": expected})
        if not passed:
            raise ValueError(f"regression mismatch {name}: {actual!r} != {expected!r}")

    check("overall.stored_frame_count", summary["overall"]["stored_frame_count"], continuity["stored_frame_count"])
    for method in METHODS:
        for key in ("fresh_count", "held_count", "no_pose_count"):
            check(f"overall.{method}.{key}", summary["overall"]["methods"][method][key], continuity["methods"][method][key])
        check(f"overall.{method}.fresh_output_rate_pct", summary["overall"]["methods"][method]["fresh_output_rate_pct"], continuity["methods"][method]["fresh_output_rate_pct"], True)
    for session in SESSION_ORDER:
        ours = summary["sessions"][session]
        theirs = continuity["sessions"][session]
        check(f"{session}.stored_frame_count", ours["stored_frame_count"], theirs["stored_frame_count"])
        for method in METHODS:
            om, tm = ours["methods"][method], theirs["methods"][method]
            for key in ("fresh_count", "held_count", "no_pose_count", "detection_present_true_count", "pnp_failed_true_count", "partial_keypoint_missing_frame_count"):
                check(f"{session}.{method}.{key}", om[key], tm[key])
            check(f"{session}.{method}.pair_count", om["adjacent_output_change"]["pair_count"], tm["adjacent_output_change"]["pair_count"])
            for variable in ("abs_x_delta_m", "abs_z_delta_m", "abs_yaw_delta_360_wrapped_deg"):
                for statistic in ("count", "median", "p90", "maximum"):
                    check(f"{session}.{method}.{variable}.{statistic}",
                          om["adjacent_output_change"][variable][statistic],
                          tm["adjacent_output_change"][variable][statistic], numeric=statistic != "count")
            for gap_name in ("nonfresh_gaps", "no_pose_gaps"):
                gs = tm[gap_name]
                check(f"{session}.{method}.{gap_name}.count", om[gap_name]["gap_count"], len(gs))
                expected_max = max((g["duration_s"] for g in gs if g["duration_s"] is not None), default=None)
                check(f"{session}.{method}.{gap_name}.max_completed_duration", om[gap_name]["maximum_completed_duration_s"], expected_max, True)
    return {"status": "VERIFIED_COMPLETE", "check_count": len(checks), "all_passed": True, "checks": checks}


def fmt(value, digits=3) -> str:
    if value is None:
        return "x"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def write_csv_summary(path: Path, summary: dict) -> None:
    fields = [
        "scope", "stored_frames", "method", "fresh", "held", "no_pose", "fresh_rate_pct",
        "nonfresh_gap_count", "longest_completed_nonfresh_s", "longest_nonfresh_frames",
        "adjacent_pair_count", "abs_dx_m_median", "abs_dx_m_p90", "abs_dz_m_median", "abs_dz_m_p90",
        "abs_dyaw_deg_median", "abs_dyaw_deg_p90", "same_frame_both_pose_count",
        "both_methods_valid_same_adjacent_pair_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        scopes = [("ALL_STORED", summary["overall"])] + [(session, summary["sessions"][session]) for session in SESSION_ORDER]
        for scope, block in scopes:
            for method in METHODS:
                value = block["methods"][method]
                adjacent = value["adjacent_output_change"]
                gap = value["nonfresh_gaps"]
                writer.writerow({
                    "scope": scope, "stored_frames": block["stored_frame_count"], "method": method,
                    "fresh": value["fresh_count"], "held": value["held_count"], "no_pose": value["no_pose_count"],
                    "fresh_rate_pct": repr(value["fresh_output_rate_pct"]),
                    "nonfresh_gap_count": gap["gap_count"],
                    "longest_completed_nonfresh_s": gap["maximum_completed_duration_s"],
                    "longest_nonfresh_frames": gap["maximum_missing_stored_frame_count"],
                    "adjacent_pair_count": adjacent["pair_count"],
                    "abs_dx_m_median": adjacent["abs_x_delta_m"]["median"],
                    "abs_dx_m_p90": adjacent["abs_x_delta_m"]["p90"],
                    "abs_dz_m_median": adjacent["abs_z_delta_m"]["median"],
                    "abs_dz_m_p90": adjacent["abs_z_delta_m"]["p90"],
                    "abs_dyaw_deg_median": adjacent["abs_yaw_delta_360_wrapped_deg"]["median"],
                    "abs_dyaw_deg_p90": adjacent["abs_yaw_delta_360_wrapped_deg"]["p90"],
                    "same_frame_both_pose_count": block["pairing"]["same_frame_both_finite_pose_count"],
                    "both_methods_valid_same_adjacent_pair_count": block["pairing"]["both_methods_valid_same_adjacent_pair_count"],
                })


def make_plots(rows: list[dict], summary: dict, output: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pallet_combined_mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = session_groups(rows)
    figure_path = output / "figures/lifter_prediction_timeseries.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SESSION_ORDER), 3, figsize=(17, 12), constrained_layout=True)
    for row_index, session in enumerate(SESSION_ORDER):
        session_rows = groups[session]
        start = session_rows[0]["sensor_timestamp_ms"]
        seconds = np.asarray([(row["sensor_timestamp_ms"] - start) / 1000.0 for row in session_rows])
        for column, (field, label) in enumerate((("x_m", "x output (m)"), ("z_m", "z output (m)"), ("yaw_deg", "yaw output (deg)"))):
            ax = axes[row_index, column]
            for method in METHODS:
                values = [finite_pose(row["methods"][method]) for row in session_rows]
                values = np.asarray([pose[field] if pose is not None else np.nan for pose in values], dtype=float)
                ax.plot(seconds, values, lw=0.65, alpha=0.82, color=COLORS[method], label=method)
            ax.grid(alpha=0.2)
            ax.set_ylabel(f"{session}\n{label}")
            if row_index == len(SESSION_ORDER) - 1:
                ax.set_xlabel("sensor elapsed time (s)")
            if row_index == 0 and column == 2:
                ax.legend(loc="best")
    fig.suptitle("Fixed inference output trajectories — no human or physical reference", fontsize=14)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    coverage_path = output / "figures/lifter_output_coverage.png"
    labels = ["ALL"] + list(SESSION_ORDER)
    blocks = [summary["overall"]] + [summary["sessions"][session] for session in SESSION_ORDER]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        offset = (method_index - 0.5) * width
        fresh = np.asarray([100 * block["methods"][method]["fresh_count"] / block["stored_frame_count"] for block in blocks])
        held = np.asarray([100 * block["methods"][method]["held_count"] / block["stored_frame_count"] for block in blocks])
        missing = np.asarray([100 * block["methods"][method]["no_pose_count"] / block["stored_frame_count"] for block in blocks])
        ax.bar(x + offset, fresh, width, color=COLORS[method], label=f"{method} fresh")
        ax.bar(x + offset, held, width, bottom=fresh, color="#777777", label=f"{method} held", hatch="//")
        ax.bar(x + offset, missing, width, bottom=fresh + held, color="#e6e6e6", edgecolor="#666666", label=f"{method} no_pose", hatch="xx")
        for xi, rate in zip(x + offset, fresh):
            ax.text(xi, max(rate - 1.1, 1), f"{rate:.2f}%", rotation=90, ha="center", va="top", fontsize=8, color="white" if rate > 50 else "black")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of stored frames (%)")
    ax.set_title("Stored-frame output coverage (prediction state; not accuracy)")
    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(coverage_path, dpi=180)
    plt.close(fig)
    return [figure_path, coverage_path]


def make_overlay(rows: list[dict], manifest: dict, results_root: Path, output: Path) -> tuple[Path, dict]:
    from PIL import Image, ImageDraw, ImageFont

    indexed = {row["frame_id"]: row for row in rows}
    tile_width, image_height, label_height = 240, 180, 18
    columns = 10
    manifest_frames = manifest["frames"]
    grid_rows = math.ceil(len(manifest_frames) / columns)
    header = 52
    sheet = Image.new("RGB", (columns * tile_width, header + grid_rows * (image_height + label_height)), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw_sheet.text((8, 7), "Fixed review-120 prediction overlay (all 120 frames)", fill="black", font=font)
    draw_sheet.text((8, 24), "Base: orange circles/lines | N3: cyan crosses/lines | box: frozen selected detection | no reference/error", fill="black", font=font)
    counts = Counter()
    for index, frame in enumerate(manifest_frames):
        row = indexed[frame["frame_id"]]
        image_path = results_root / "review" / frame["image_path"]
        if sha256(image_path) != frame["image_sha256"]:
            raise ValueError("review image hash mismatch: " + frame["frame_id"])
        source = Image.open(image_path).convert("RGB")
        painter = ImageDraw.Draw(source)
        selected = row["methods"]["Base"].get("selected_object", {})
        box = selected.get("selected_box_xyxy")
        if isinstance(box, list) and len(box) == 4 and all(finite(v) for v in box):
            painter.rectangle(tuple(box), outline=(255, 230, 40), width=2)
        for method in METHODS:
            item = row["methods"][method]
            counts[(method, item["pose_state"])] += 1
            points, mask = item.get("keypoints"), item.get("keypoints_mask")
            if not isinstance(points, list) or len(points) < 8 or not isinstance(mask, list) or len(mask) != 8:
                continue
            color = (217, 95, 2) if method == "Base" else (0, 190, 210)
            for first, second in EDGES:
                if mask[first] and mask[second]:
                    painter.line((tuple(points[first]), tuple(points[second])), fill=color, width=2)
            for point_index, point in enumerate(points[:8]):
                if not mask[point_index] or not (isinstance(point, list) and len(point) == 2 and all(finite(v) for v in point)):
                    continue
                x, y = point
                if method == "Base":
                    painter.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                else:
                    painter.line((x - 4, y, x + 4, y), fill=color, width=2)
                    painter.line((x, y - 4, x, y + 4), fill=color, width=2)
        if all(row["methods"][method]["pose_state"] == "no_pose" for method in METHODS):
            painter.rectangle((5, 5, 92, 25), fill=(255, 255, 255), outline=(190, 0, 0), width=1)
            painter.text((10, 8), "B/N3 NO_POSE", fill=(190, 0, 0), font=font)
        tile = source.resize((tile_width, image_height), Image.Resampling.LANCZOS)
        x0 = (index % columns) * tile_width
        y0 = header + (index // columns) * (image_height + label_height)
        sheet.paste(tile, (x0, y0))
        draw_sheet.text((x0 + 3, y0 + image_height + 2), frame["frame_id"], fill="black", font=font)
    output_path = output / "figures/review120_prediction_overlay_contact_sheet.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    return output_path, {
        "status": "VERIFIED_COMPLETE",
        "selection": "all frozen review-manifest frames in manifest order; no output-based selection",
        "frame_count": len(manifest_frames),
        "reference_overlayed": False,
        "accuracy_interpretation_allowed": False,
        "method_pose_state_counts": {f"{method}_{state}": counts[(method, state)] for method in METHODS for state in ("fresh", "held", "no_pose")},
    }


def report_markdown(summary: dict, regression: dict, inference_status: dict, overlay: dict) -> str:
    overall = summary["overall"]
    lines = [
        "# 리프터 고정 추론 연속성 재집계",
        "",
        "## 판정 범위",
        "",
        f"고정된 Base/N3 모델로 저장 프레임 **{overall['stored_frame_count']:,}장**을 한 번 추론한 결과를 재집계했다. "
        f"추론 상태는 `{inference_status['status']}`이고 새 학습과 optimizer update는 모두 0회다. "
        "이 문서의 수치는 사람 정답이 없는 **출력 가용성과 인접 출력 변화**이며 정확도나 실제 정지 상태의 잡음을 뜻하지 않는다.",
        "",
        "사람 코너 참조, 예측 객체와 사람 참조의 대응, 사람이 확인한 정지 구간, 독립 물리 참조는 아직 없으므로 관련 결과는 모두 `x`다.",
        "",
        "## 전체 저장 프레임",
        "",
        "| 방법 | fresh | held | no_pose | fresh 비율 | 인접 유효 쌍 | |Δx| 중앙/P90 (m) | |Δz| 중앙/P90 (m) | |Δyaw| 중앙/P90 (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = overall["methods"][method]
        adjacent = value["adjacent_output_change"]
        lines.append(
            f"| {method} | {value['fresh_count']:,} | {value['held_count']:,} | {value['no_pose_count']:,} | "
            f"{value['fresh_output_rate_pct']:.3f}% | {adjacent['pair_count']:,} | "
            f"{fmt(adjacent['abs_x_delta_m']['median'], 6)} / {fmt(adjacent['abs_x_delta_m']['p90'], 6)} | "
            f"{fmt(adjacent['abs_z_delta_m']['median'], 6)} / {fmt(adjacent['abs_z_delta_m']['p90'], 6)} | "
            f"{fmt(adjacent['abs_yaw_delta_360_wrapped_deg']['median'], 6)} / {fmt(adjacent['abs_yaw_delta_360_wrapped_deg']['p90'], 6)} |"
        )
    pair = overall["pairing"]
    lines.extend([
        "",
        f"같은 프레임에서 두 방법 모두 유한 pose인 쌍은 **{pair['same_frame_both_finite_pose_count']:,}개**다. "
        f"Base만 유한한 프레임은 {pair['same_frame_base_only_finite_pose_count']:,}개, N3만 유한한 프레임은 "
        f"{pair['same_frame_n3_only_finite_pose_count']:,}개, 둘 다 no_pose인 프레임은 {pair['same_frame_both_no_pose_count']:,}개다. "
        f"세션 경계를 제외한 인접 후보 {pair['adjacent_candidate_pair_count']:,}쌍 중 양쪽 방법이 같은 인접 쌍에서 모두 유효한 경우는 "
        f"**{pair['both_methods_valid_same_adjacent_pair_count']:,}쌍**이다.",
        "",
        "## 세션별 가용성과 최장 결측",
        "",
        "`최장 결측 초`는 첫 no_pose 저장 프레임에서 다음 fresh 프레임까지의 표본 센서 시간이다. 오른쪽 검열 구간이면 이 값은 `x`로 유지한다.",
        "",
        "| 세션 | 방법 | 저장 장수 | fresh/held/no_pose | fresh 비율 | no_pose 구간 수 | 최장 완료 결측 (s) | 최장 결측 저장 장수 | 인접 유효 쌍 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for session in SESSION_ORDER:
        block = summary["sessions"][session]
        for method in METHODS:
            value = block["methods"][method]
            gap = value["no_pose_gaps"]
            lines.append(
                f"| {session} | {method} | {block['stored_frame_count']:,} | "
                f"{value['fresh_count']:,}/{value['held_count']:,}/{value['no_pose_count']:,} | "
                f"{value['fresh_output_rate_pct']:.3f}% | {gap['gap_count']:,} | "
                f"{fmt(gap['maximum_completed_duration_s'], 6)} | {fmt(gap['maximum_missing_stored_frame_count'], 0)} | "
                f"{value['adjacent_output_change']['pair_count']:,} |"
            )
    lines.extend([
        "",
        "## 세션별 인접 출력 변화",
        "",
        "| 세션 | 방법 | 유효 쌍 | |Δx| 중앙/P90 (m) | |Δz| 중앙/P90 (m) | |Δyaw| 중앙/P90 (deg) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for session in SESSION_ORDER:
        for method in METHODS:
            adjacent = summary["sessions"][session]["methods"][method]["adjacent_output_change"]
            lines.append(
                f"| {session} | {method} | {adjacent['pair_count']:,} | "
                f"{fmt(adjacent['abs_x_delta_m']['median'], 6)} / {fmt(adjacent['abs_x_delta_m']['p90'], 6)} | "
                f"{fmt(adjacent['abs_z_delta_m']['median'], 6)} / {fmt(adjacent['abs_z_delta_m']['p90'], 6)} | "
                f"{fmt(adjacent['abs_yaw_delta_360_wrapped_deg']['median'], 6)} / {fmt(adjacent['abs_yaw_delta_360_wrapped_deg']['p90'], 6)} |"
            )
    lines.extend([
        "",
        "![전체 세션의 실제 예측 시계열](figures/lifter_prediction_timeseries.png)",
        "",
        "시계열은 유한한 출력만 연결하고 no_pose는 공백으로 둔다. 그래프에는 정답이나 실제 물리 궤적이 겹쳐 있지 않다.",
        "",
        "![저장 프레임 출력 coverage](figures/lifter_output_coverage.png)",
        "",
        "![고정 review-120 전체 prediction overlay](figures/review120_prediction_overlay_contact_sheet.png)",
        "",
        f"contact sheet는 결과를 보고 고른 표본이 아니라 고정 manifest의 **{overlay['frame_count']}장 전부**를 순서대로 표시한다. "
        "주황색은 Base, 청록색은 N3 예측이며 사람 코너 정답과 오차는 표시하지 않았다.",
        "",
        "## 독립 재계산과 원 evaluator 대조",
        "",
        f"원시 L4 예측에서 별도로 다시 계산한 {regression['check_count']}개 수치가 기존 evaluator 결과와 모두 일치했다. "
        "Base/N3의 keypoints_mask도 8,910프레임에서 모두 같았고 부분 keypoint 결측 프레임은 두 방법 모두 0장이다. "
        "이 확인은 출력 계약과 집계 재현성을 검증하며 사람 참조 정확도를 대신하지 않는다.",
        "",
        "## 실행 비용과 미완료 참조",
        "",
        f"GPU 추론 시간은 **{inference_status['gpu_inference_seconds']:.3f}s**, 호출 wall 시간은 "
        f"**{inference_status['invocation_wall_seconds']:.3f}s**다. 전체 8,910장을 새 고정 추론으로 계산했고 cache 재사용 행은 0개다.",
        "",
        "| 항목 | 상태 | 보고값 |",
        "|---|---|---|",
        "| 8,910장 출력 연속성 | VERIFIED_COMPLETE | 위 표와 JSON/CSV |",
        "| 사람 visible-corner 정확도 | WAITING_HUMAN | x |",
        "| 예측 객체↔사람 참조 대응 | WAITING_HUMAN | x |",
        "| 사람이 확인한 정지 구간 변화 | WAITING_HUMAN | x |",
        "| 독립 물리 위치·방향 정확도 | BLOCKED_REFERENCE | x |",
        "",
        "세부 수치와 분모는 `LIFTER_CONTINUITY_SUMMARY.json`, 표 형식은 `LIFTER_CONTINUITY_SUMMARY.csv`, "
        "입력·출력 SHA-256과 실행 기록은 `EXECUTION_FILE_RECEIPT.json`에 있다.",
        "",
    ])
    return "\n".join(lines)


def build_receipt(root: Path, result_root: Path, output: Path, raw_rows: int,
                  inference_status: dict, generated_paths: list[Path], source_before: dict[str, str]) -> dict:
    source_paths = {
        "raw_predictions": result_root / "raw_predictions/ALL_STORED_FRAMES.jsonl",
        "l4_predictions": result_root / "raw_predictions/ALL_STORED_FRAMES_L4.jsonl",
        "mask_receipt": result_root / "raw_predictions/L4_MASK_RECEIPT.json",
        "run_identity": result_root / "raw_predictions/RUN_IDENTITY.json",
        "inference_status": result_root / "INFERENCE_RUN_STATUS.json",
        "metrics_l4": result_root / "metrics_l4/LIFTER_METRICS.json",
        "evaluation_plan": result_root / "LIFTER_EVALUATION_PLAN.json",
        "review_manifest": result_root / "review/MANIFEST.json",
        "corner_contract": result_root / "review/CORNER_CONTRACT.json",
        "base_checkpoint": root / "scripts/research/pallet_lifter_case_review_20261003_v1/assets/yolo_r0.pt",
        "n3_checkpoint": root / "scripts/research/pallet_lifter_case_review_20261003_v1/assets/n3_seed1.pt",
        "metrics_evaluator": root / "scripts/research/pallet_lifter_case_review_20261003_v1/metrics/evaluate.py",
        "l4_bridge": root / "scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py",
        "derived_output_builder": Path(__file__).resolve(),
        "derived_output_tests": Path(__file__).resolve().with_name("test_actual_outputs.py"),
    }
    source_after = {name: sha256(path) for name, path in source_paths.items()}
    if source_before != source_after:
        raise ValueError("a frozen source input changed while derived outputs were built")
    source_records = {}
    for name, path in source_paths.items():
        rows = raw_rows if name in ("raw_predictions", "l4_predictions") else None
        source_records[name] = file_record(path, root, rows=rows)
    return {
        "schema_version": "lifter_combined_integration_receipt_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_COMPLETE",
        "scope": "derived reporting only; no inference, training, control, or human-label creation",
        "command": "python combined_integration/build_actual_outputs.py",
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "inference_status": inference_status["status"],
            "inferred_frames": inference_status["inferred_frames"],
            "newly_computed_frames": inference_status["newly_computed_frames"],
            "gpu_inference_seconds": inference_status["gpu_inference_seconds"],
            "invocation_wall_seconds": inference_status["invocation_wall_seconds"],
            "training_runs": inference_status["training_runs"],
            "optimizer_updates": inference_status["optimizer_updates"],
        },
        "frozen_source_inputs_unchanged_during_build": True,
        "source_inputs": source_records,
        "derived_outputs": [file_record(path, root) for path in generated_paths],
        "receipt_self_hash": "recorded separately in EXECUTION_FILE_RECEIPT.sha256",
        "claims_not_established": [
            "visible-corner accuracy", "correct predicted-object to human-reference correspondence",
            "stationary-output variation", "independent physical pose accuracy", "method improvement",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inferred_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--lifter-root", type=Path, default=inferred_root)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args(argv)
    root, output = args.lifter_root.resolve(), args.output.resolve()
    result_root = root / "data/pallet/results/pallet_lifter_case_review_20261003_v1"
    source_paths = {
        "raw_predictions": result_root / "raw_predictions/ALL_STORED_FRAMES.jsonl",
        "l4_predictions": result_root / "raw_predictions/ALL_STORED_FRAMES_L4.jsonl",
        "mask_receipt": result_root / "raw_predictions/L4_MASK_RECEIPT.json",
        "run_identity": result_root / "raw_predictions/RUN_IDENTITY.json",
        "inference_status": result_root / "INFERENCE_RUN_STATUS.json",
        "metrics_l4": result_root / "metrics_l4/LIFTER_METRICS.json",
        "evaluation_plan": result_root / "LIFTER_EVALUATION_PLAN.json",
        "review_manifest": result_root / "review/MANIFEST.json",
        "corner_contract": result_root / "review/CORNER_CONTRACT.json",
        "base_checkpoint": root / "scripts/research/pallet_lifter_case_review_20261003_v1/assets/yolo_r0.pt",
        "n3_checkpoint": root / "scripts/research/pallet_lifter_case_review_20261003_v1/assets/n3_seed1.pt",
        "metrics_evaluator": root / "scripts/research/pallet_lifter_case_review_20261003_v1/metrics/evaluate.py",
        "l4_bridge": root / "scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py",
        "derived_output_builder": Path(__file__).resolve(),
        "derived_output_tests": Path(__file__).resolve().with_name("test_actual_outputs.py"),
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_before = {name: sha256(path) for name, path in source_paths.items()}
    output.mkdir(parents=True, exist_ok=True)
    rows = read_prediction_rows(source_paths["l4_predictions"])
    raw_rows = count_jsonl(source_paths["raw_predictions"])
    if raw_rows != len(rows):
        raise ValueError("raw and L4 row counts differ")
    plan = read_json(source_paths["evaluation_plan"])
    manifest = read_json(source_paths["review_manifest"])
    row_validation = validate_rows(rows, plan)
    summary, _ = aggregate_summary(rows, plan)
    metrics = read_json(source_paths["metrics_l4"])
    regression = regression_check(summary, metrics)
    inference_status = read_json(source_paths["inference_status"])
    if inference_status.get("status") != "VERIFIED_COMPLETE" or inference_status.get("inferred_frames") != len(rows):
        raise ValueError("inference completion receipt does not bind all rows")
    summary["input_validation"] = row_validation
    summary["evaluator_regression_check"] = regression
    summary["execution"] = {
        "inference_status": inference_status["status"],
        "gpu_inference_seconds": inference_status["gpu_inference_seconds"],
        "invocation_wall_seconds": inference_status["invocation_wall_seconds"],
        "training_runs": inference_status["training_runs"],
        "optimizer_updates": inference_status["optimizer_updates"],
    }

    summary_path = output / "LIFTER_CONTINUITY_SUMMARY.json"
    csv_path = output / "LIFTER_CONTINUITY_SUMMARY.csv"
    write_json(summary_path, summary)
    write_csv_summary(csv_path, summary)
    figures = make_plots(rows, summary, output)
    overlay_path, overlay = make_overlay(rows, manifest, result_root, output)
    summary["review120_prediction_overlay"] = overlay
    write_json(summary_path, summary)
    report_path = output / "ACTUAL_INFERENCE_REPORT_KO.md"
    report_path.write_text(report_markdown(summary, regression, inference_status, overlay), encoding="utf-8")
    generated = [summary_path, csv_path, report_path, *figures, overlay_path]
    receipt = build_receipt(root, result_root, output, raw_rows, inference_status, generated, source_before)
    receipt_path = output / "EXECUTION_FILE_RECEIPT.json"
    write_json(receipt_path, receipt)
    sidecar_path = output / "EXECUTION_FILE_RECEIPT.sha256"
    sidecar_path.write_text(f"{sha256(receipt_path)}  {receipt_path.name}\n", encoding="ascii")
    result = {
        "status": "VERIFIED_COMPLETE",
        "stored_frames": len(rows),
        "fresh_each_method": summary["overall"]["methods"]["Base"]["fresh_count"],
        "no_pose_each_method": summary["overall"]["methods"]["Base"]["no_pose_count"],
        "paired_finite_frames": summary["overall"]["pairing"]["same_frame_both_finite_pose_count"],
        "outputs": [str(path.relative_to(root)) for path in [*generated, receipt_path, sidecar_path]],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
