"""Verify completed published evidence offline with only the Python stdlib.

Default use prints a report without changing files. No model/training imports,
PnP recomputation, network, or hardware calls occur. Original absolute paths in
frozen result JSON are translated in memory using the publication manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "_docs/experiments/pallet_paper_review_20261006_v1/portable_evidence/PORTABLE_EVIDENCE_MANIFEST.json"
METHODS = ("Base", "N3")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Evidence:
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self.mapping = {row["original_path"]: row for row in
                        manifest["compressed_sources"] + manifest["reference_images"]}
        self.check_count, self.failures = 0, []

    def check(self, condition, label):
        self.check_count += 1
        if not condition:
            self.failures.append(label)

    def relative(self, path):
        source = str(path)
        prefix = self.manifest["source_repository_prefix"].rstrip("/") + "/"
        if source.startswith(prefix):
            source = source[len(prefix):]
        elif Path(source).is_absolute():
            raise ValueError(f"Unknown absolute evidence path: {source}")
        rel = PurePosixPath(source)
        if ".." in rel.parts or rel.is_absolute():
            raise ValueError(f"Evidence path escapes repository: {path}")
        return rel.as_posix()

    def open(self, source):
        rel = self.relative(source)
        entry = self.mapping.get(rel)
        path = self.root / (entry["artifact_path"] if entry else rel)
        return gzip.open(path, "rb") if entry and entry["encoding"] == "gzip" else path.open("rb")

    def content(self, source):
        with self.open(source) as handle:
            return handle.read()

    def json(self, source):
        return json.loads(self.content(source))

    def digest(self, source):
        digest, count = hashlib.sha256(), 0
        with self.open(source) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                count += len(block)
        return digest.hexdigest(), count

    def bind(self, binding, label):
        digest, count = self.digest(binding["path"])
        self.check(digest == binding["sha256"], label + ":sha256")
        if "bytes" in binding:
            self.check(count == binding["bytes"], label + ":bytes")

    def compare(self, actual, expected, label):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), label + ":keys")
            if isinstance(actual, dict):
                for key, value in expected.items():
                    if key in actual:
                        self.compare(actual[key], value, label + "." + str(key))
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(actual) == len(expected), label + ":length")
            if isinstance(actual, list):
                for index, (a, b) in enumerate(zip(actual, expected)):
                    self.compare(a, b, label + f"[{index}]")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            self.check(finite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), label)
        else:
            self.check(actual == expected, label)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def slots(method):
    points, mask = method.get("keypoints"), method.get("keypoints_mask")
    if points is None:
        if mask not in (None, []):
            raise ValueError("Missing keypoints have a nonempty mask")
        return [None] * 8, [False] * 8
    if not isinstance(points, list) or len(points) != 8:
        raise ValueError("Expected eight canonical prediction slots")
    if mask is not None and (not isinstance(mask, list) or len(mask) != 8
                             or any(type(v) is not bool for v in mask)):
        raise ValueError("Invalid frozen point mask")
    return points, [isinstance(p, (list, tuple)) and len(p) == 2 and all(finite(v) for v in p)
                    and (mask is None or mask[i]) for i, p in enumerate(points)]


def summarize(rows):
    denominator, methods = len(rows), {}
    for method in METHODS:
        values = [r[f"{method}_error_px"] for r in rows if r[f"{method}_error_px"] is not None]
        hits = sum(r[f"{method}_pck10_hit"] for r in rows)
        methods[method] = dict(
            conditional_error=dict(count=len(values), median_px=percentile(values, .5),
                                   p90_px=percentile(values, .9), mean_px=sum(values) / len(values) if values else None),
            full_reference_point_count=denominator, valid_matching_prediction_points=len(values),
            failed_reference_points=denominator - len(values),
            failure_reasons=dict(Counter(r[f"{method}_failure_reason"] for r in rows if r[f"{method}_failure_reason"])),
            pck10_hit_count=hits, pck10_full_reference_percent=100 * hits / denominator if denominator else None,
            pck10_conditional_percent=100 * hits / len(values) if values else None)
    common = [r for r in rows if r["Base_error_px"] is not None and r["N3_error_px"] is not None]
    before, after = ([r[f"{method}_error_px"] for r in common] for method in METHODS)
    delta = [b - a for a, b in zip(before, after)]
    paired = dict(
        common_valid_point_count=len(common),
        median_after_minus_median_before_px=percentile(after, .5) - percentile(before, .5) if common else None,
        median_of_paired_after_minus_before_px=percentile(delta, .5),
        p90_after_minus_p90_before_px=percentile(after, .9) - percentile(before, .9) if common else None,
        improved_points=sum(v < 0 for v in delta), worsened_points=sum(v > 0 for v in delta),
        tied_points=sum(v == 0 for v in delta),
        Base_only_valid_points=sum(r["Base_error_px"] is not None and r["N3_error_px"] is None for r in rows),
        N3_only_valid_points=sum(r["N3_error_px"] is not None and r["Base_error_px"] is None for r in rows))
    return dict(reference_point_count=denominator, methods=methods, paired=paired)


def recompute(records, predictions, decisions):
    if any(d not in ("same", "different") for d in decisions.values()):
        raise ValueError("Unresolved object correspondence remains pending")
    points, frames = [], []
    for record in records:
        fid, per_frame = record["frame_id"], []
        for point in record["points"]:
            row = dict(frame_id=fid, session_id=record["session_id"], point_id=point["id"],
                       reference_x=point["x"], reference_y=point["y"],
                       reference_source=point.get("source", "UNKNOWN"), object_match=decisions[fid])
            for method in METHODS:
                prediction = predictions[fid]["methods"][method]
                coords, mask = slots(prediction)
                reason = ("wrong_selected_object" if decisions[fid] == "different" else
                          "no_selected_object" if prediction.get("selected_index") is None else
                          "missing_or_invalid_prediction" if not mask[point["id"]] else None)
                error = None if reason else math.hypot(coords[point["id"]][0] - point["x"],
                                                       coords[point["id"]][1] - point["y"])
                row.update({f"{method}_error_px": error, f"{method}_failure_reason": reason,
                            f"{method}_pck10_hit": error is not None and error <= 10.0})
            row["paired_N3_minus_Base_px"] = row["N3_error_px"] - row["Base_error_px"] if row["Base_error_px"] is not None and row["N3_error_px"] is not None else None
            points.append(row)
            per_frame.append(row)
        frames.append(dict(frame_id=fid, session_id=record["session_id"], object_match=decisions[fid], **summarize(per_frame)))
    metrics = summarize(points)
    metrics.update(frame_count=len(records), decision_counts=dict(Counter(decisions.values())),
                   reference_source_counts=dict(Counter(p["reference_source"] for p in points)), sessions={})
    for session in dict.fromkeys(r["session_id"] for r in records):
        metrics["sessions"][session] = dict(frame_count=sum(r["session_id"] == session for r in records),
                                            **summarize([p for p in points if p["session_id"] == session]))
    delta = [f["methods"]["N3"]["conditional_error"]["median_px"] - f["methods"]["Base"]["conditional_error"]["median_px"]
             for f in frames if f["methods"]["Base"]["conditional_error"]["median_px"] is not None
             and f["methods"]["N3"]["conditional_error"]["median_px"] is not None]
    metrics["paired_frame_medians"] = dict(common_valid_frames=len(delta),
        median_N3_minus_Base_frame_median_px=percentile(delta, .5), improved_frames=sum(v < 0 for v in delta),
        worsened_frames=sum(v > 0 for v in delta), tied_frames=sum(v == 0 for v in delta))
    return metrics, points, frames


def verify(root):
    started = time.perf_counter()
    manifest = json.loads((Path(root) / MANIFEST).read_text())
    e = Evidence(root, manifest)
    e.check(manifest["original_paths_not_rewritten"] is True, "original paths preserved")
    e.check(len(manifest["compressed_sources"]) == 5, "five complete large files")
    e.check(len(manifest["reference_images"]) == 12, "twelve exact source images")
    for i, entry in enumerate(manifest["compressed_sources"] + manifest["reference_images"]):
        artifact = Path(root) / entry["artifact_path"]
        e.check(hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["artifact_sha256"], f"copy {i}:artifact SHA")
        e.check(artifact.stat().st_size == entry["artifact_bytes"], f"copy {i}:artifact bytes")
        digest, size = e.digest(entry["original_path"])
        e.check(digest == entry["original_sha256"] and size == entry["original_bytes"], f"copy {i}:lossless original")
        if entry["encoding"] == "gzip":
            with artifact.open("rb") as handle:
                header = handle.read(10)
            e.check(header[:2] == b"\x1f\x8b" and header[3] == 0 and header[4:8] == b"\x00\x00\x00\x00", f"copy {i}:deterministic gzip")
            e.check(digest == entry["decompressed_sha256"], f"copy {i}:decompressed SHA")
    result = e.json(manifest["result_path"])
    e.bind(dict(path=manifest["result_path"], sha256=manifest["result_sha256"]), "latest result")
    for name, binding in result["sources"].items():
        e.bind(binding, "latest source " + name)
    ref, proto, queue, sidecar, review, identity = (
        e.json(result["sources"][name]["path"]) for name in
        ("assisted_reference", "evaluation_protocol", "object_match_queue", "human_match_sidecar", "manifest", "run_identity"))
    review_rel = PurePosixPath(e.relative(result["sources"]["manifest"]["path"]))
    plan_path = str(review_rel.parent.parent / "LIFTER_EVALUATION_PLAN.json")
    corner_path = str(review_rel.parent / "CORNER_CONTRACT.json")
    for path, key in ((plan_path, "plan_sha256"), (corner_path, "corner_contract_sha256")):
        e.bind(dict(path=path, sha256=ref["bindings"][key]), key)
        e.check(proto["bindings"][key] == ref["bindings"][key], "protocol " + key)
    e.bind(ref["original_batch_binding"], "original fixed twelve batch")
    e.check(proto["bindings"]["batch_sha256"] == ref["original_batch_binding"]["sha256"], "protocol batch binding")
    if ref.get("existing_human_history_confirmation"):
        e.bind(ref["existing_human_history_confirmation"]["binding"], "actual earlier history record")
    if ref.get("later_current_manual_snapshot"):
        e.bind(ref["later_current_manual_snapshot"]["binding"], "separate later manual snapshot")
        e.check(ref["later_current_manual_snapshot"]["used_for_this_geometry_reference"] is False, "later clicks not substituted")
    for key, name in {"predictions_sha256": "predictions", "prediction_run_identity_sha256": "run_identity",
                      "reviewed_reference_sha256": "assisted_reference", "manifest_sha256": "manifest",
                      "queue_sha256": "object_match_queue"}.items():
        e.check(proto["bindings"][key] == result["sources"][name]["sha256"], "protocol source " + key)
        if key in queue["bindings"]:
            e.check(queue["bindings"][key] == result["sources"][name]["sha256"], "queue source " + key)
    e.check(sidecar["queue_sha256"] == result["sources"]["object_match_queue"]["sha256"], "human decisions bind exact queue")
    e.check(sidecar["source_kind"] == "human_reviewed", "actual human object-match source")
    records, ids = ref["records"], [r["frame_id"] for r in ref["records"]]
    e.check(len(ids) == len(set(ids)) == 12, "twelve unique references")
    e.check(proto["ordered_frame_ids"] == ids and proto["ordered_corner_ids"] == list(range(8)), "locked frame/corner order")
    e.check(proto["fixed_reference_point_denominator"] == 96 and proto["pck_threshold_px"] == 10.0
            and proto["include_all_reference_corners"] is True, "96 original slots and ten-pixel threshold")
    decisions = {r["frame_id"]: r["decision"] for r in sidecar["records"]}
    e.check(len(sidecar["records"]) == 12 and set(decisions) == set(ids), "all actual human decisions exactly once")
    queued, documented = ({r["frame_id"]: r for r in source["records" if source is queue else "frames"]} for source in (queue, review))
    for decision in sidecar["records"]:
        fid = decision["frame_id"]
        for key in ("target_object_id", "selected_index", "selected_box_xyxy"):
            e.check(decision[key] == queued[fid][key], f"{fid}:human {key}")
        e.check(decision["reviewer"]["entered_by"] == "human" and decision["reviewer"]["confirmation"] is True,
                f"{fid}:actual human selection provenance")
        e.check(decision["review_time"]["duration_seconds"] >= 0, f"{fid}:actual review time")
    selected, raw_ids, session_counts = {}, set(), Counter()
    pose_states = {method: Counter() for method in METHODS}
    with e.open(result["sources"]["predictions"]["path"]) as handle:
        for line in handle:
            prediction = json.loads(line)
            fid = prediction["frame_id"]
            e.check(fid not in raw_ids, f"raw {fid}:unique")
            raw_ids.add(fid)
            session_counts[prediction["session_id"]] += 1
            base, n3 = [prediction["methods"][method] for method in METHODS]
            e.check(base.get("selected_index") == n3.get("selected_index")
                    and base.get("selected_object") == n3.get("selected_object"), f"raw {fid}:shared object")
            e.check(slots(base)[1] == slots(n3)[1], f"raw {fid}:unchanged mask")
            for method in METHODS:
                pose_states[method][prediction["methods"][method]["pose_state"]] += 1
            if fid in decisions:
                selected[fid] = prediction
    plan = e.json(plan_path)
    e.check(len(raw_ids) == plan["stored_frame_count"] == 8910, "complete original 8910-frame run")
    e.check(set(selected) == set(ids), "every reference has original prediction")
    e.check(identity["plan_sha256"] == proto["bindings"]["plan_sha256"], "run identity plan binding")
    e.check(identity["source_commit"] == manifest["source_raw_prediction_commit"], "raw source commit retained")
    for record in records:
        fid = record["frame_id"]
        e.check([p["id"] for p in record["points"]] == list(range(8)), f"{fid}:canonical order")
        e.bind(record["source_pnp_binding"], fid + ":original G source")
        original = e.json(record["source_pnp_binding"]["path"])
        e.check([[p["x"], p["y"]] for p in record["points"]] == original["editor_kps_2d"][:8], f"{fid}:exact G coordinates")
        e.check([p["source"] for p in record["points"]] == [p["source"] for p in original["editor_keypoint_annotations"][:8]], f"{fid}:point provenance")
        e.check(original["evaluation_use"] is False and original["independent_reference"] is False, f"{fid}:original flags")
        image_path = str(review_rel.parent / record["image_path"])
        e.bind(dict(path=image_path, sha256=record["image_sha256"]), fid + ":source image")
        for key in ("session_id", "saved_frame_index", "camera_frame_number", "camera_sensor_timestamp_ms",
                    "decoded_bgr_sha256", "image_sha256", "image_path"):
            e.check(record[key] == documented[fid][key], f"{fid}:manifest {key}")
        for raw_key, ref_key in (("stored_index", "saved_frame_index"), ("sensor_timestamp_ms", "camera_sensor_timestamp_ms"),
                                 ("camera_frame_number", "camera_frame_number"), ("session_id", "session_id"),
                                 ("decoded_bgr_sha256", "decoded_bgr_sha256")):
            e.check(selected[fid][raw_key] == record[ref_key], f"{fid}:raw {raw_key}")
        for method in METHODS:
            model = selected[fid]["methods"][method]
            e.check(model["selected_index"] == queued[fid]["selected_index"]
                    and model["selected_object"]["selected_box_xyxy"] == queued[fid]["selected_box_xyxy"], f"{fid}:{method} selected box")
    metrics, points, frames = recompute(records, selected, decisions)
    e.compare(metrics, result["metrics"], "recomputed overall/session/paired")
    e.compare(frames, result["frame_results"], "recomputed twelve frames")
    output = PurePosixPath(e.relative(result["output_dir"]))
    e.compare(frames, e.json(str(output / "FRAME_RESULTS.json")), "saved frame artifact")
    csv_rows = list(csv.DictReader(io.StringIO(e.content(str(output / "POINT_ERRORS.csv")).decode())))
    e.check(len(csv_rows) == 96, "all ninety-six CSV point errors")
    for index, (actual, saved) in enumerate(zip(points, csv_rows)):
        for key, value in actual.items():
            matched = (saved[key] == "" if value is None else saved[key] == str(value) if isinstance(value, bool)
                       else math.isclose(float(saved[key]), value, rel_tol=1e-12, abs_tol=1e-10)
                       if isinstance(value, (int, float)) else saved[key] == value)
            e.check(matched, f"point CSV {index}:{key}")
    e.check(len(result["overlays"]) == 12 and [r["frame_id"] for r in result["overlays"]] == ids, "all twelve overlays")
    for overlay in result["overlays"]:
        e.bind(overlay, overlay["frame_id"] + ":comparison PNG")
        record = next(r for r in records if r["frame_id"] == overlay["frame_id"])
        e.check(overlay["source_image_sha256"] == record["image_sha256"], overlay["frame_id"] + ":overlay source")
    for key, value in (("training_runs", 0), ("optimizer_updates", 0), ("new_model_forward_frames", 0),
                       ("hardware_control_calls", 0), ("full_120_24_completed", False),
                       ("official_visible_corner_accuracy", "x"), ("stationary_noise", "x"),
                       ("independent_physical_TR", "x"), ("directly_visible_claim", False)):
        e.check(result[key] == value, "scope: " + key)
    return dict(status="PASS" if not e.failures else "FAIL", check_count=e.check_count,
                failed_check_count=len(e.failures), failed_checks=e.failures,
                verified_latest_result_path=manifest["result_path"], bound_latest_source_count=len(result["sources"]),
                raw_prediction_rows_reused=len(raw_ids), raw_session_rows=dict(session_counts),
                all_frame_pose_states={method: dict(counts) for method, counts in pose_states.items()},
                recomputed_reference_frames=len(records), recomputed_reference_points=len(points),
                exact_original_pnp_source_count=len(records), verified_comparison_png_count=len(result["overlays"]),
                source_point_counts=metrics["reference_source_counts"], object_match_counts=metrics["decision_counts"],
                metrics=metrics, compressed_artifact_count=len(manifest["compressed_sources"]),
                original_large_evidence_bytes=manifest["total_original_compressed_source_bytes"],
                compressed_evidence_bytes=manifest["total_compressed_artifact_bytes"],
                source_reference_image_bytes=manifest["reference_image_bytes"],
                portable_without_original_absolute_paths=True, stdlib_only=True, training_runs=0,
                optimizer_updates=0, new_model_forward_frames=0, hardware_control_calls=0, network_calls=0,
                cpu_wall_seconds=time.perf_counter() - started,
                verification_scope="Five lossless source copies, all 8910 shared selections/masks and completed exploratory 12-frame/96-point metrics; not physical T/R accuracy or 120/24 completion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional new verification report copy")
    args = parser.parse_args()
    try:
        report = verify(args.root)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        report = dict(status="FAIL", error=str(exc), training_runs=0, new_model_forward_frames=0, network_calls=0)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
