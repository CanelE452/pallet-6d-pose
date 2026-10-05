#!/usr/bin/env python3
"""Join immutable handoff annotations to existing receiver evidence; no inference.

Uses only the standard library. A matching frame filename is a candidate, not an
image identity proof. A primary observation is never a repeat observation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import time
import zipfile

NAME = "pallet_remaining_evidence_connection_20261006_v1"
LIFTER = "data/pallet/results/pallet_lifter_case_review_20261003_v1"
ASSISTED = "data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1"
RESULT = "_docs/experiments/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/RESULT.json"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def read_json(path):
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    return json.loads(read_bytes(path).decode("utf-8-sig"))


def checked_bytes(path, expected):
    data = read_bytes(path)
    if sha_bytes(data) != expected["sha256"]:
        raise ValueError(f"Source SHA-256 mismatch: {path}")
    if "bytes" in expected and len(data) != expected["bytes"]:
        raise ValueError(f"Source byte count mismatch: {path}")
    return data


def checked_binding(root, binding):
    name = binding["path"].replace("\\", "/")
    marker = "/pallet-pose/"
    if Path(name).is_absolute():
        if marker not in name:
            raise ValueError(f"Unresolvable original provenance path: {name}")
        name = name.split(marker, 1)[1]
    path = root / name
    if not path.exists():
        portable = root / "_docs/experiments/pallet_paper_review_20261006_v1/portable_evidence"
        manifest_path = portable / "PORTABLE_EVIDENCE_MANIFEST.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            for item in manifest.get("compressed_sources", []):
                if item.get("original_path") == name:
                    path = root / item["artifact_path"]
                    break
    return path, checked_bytes(path, binding)


def join_tasks(frames, references, excluded_ids, candidate_ids):
    """Keep image identity, analysis eligibility, and task pass as separate facts."""
    ids = [frame["frame_id"] for frame in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("Frozen manifest has duplicate primary frame IDs")
    if not set(excluded_ids).issubset(ids):
        raise ValueError("User exclusion not in frozen plan")
    if set(references) & set(excluded_ids):
        raise ValueError("Excluded frame was reintroduced in receiver panel")
    for fid, reference in references.items():
        frame = next((f for f in frames if f["frame_id"] == fid), None)
        if frame is None:
            raise ValueError(f"Receiver frame is outside frozen plan: {fid}")
        for field in ("image_sha256", "decoded_bgr_sha256", "saved_frame_index", "width", "height"):
            if reference[field] != frame[field]:
                raise ValueError(f"Receiver image/frame binding mismatch: {fid} {field}")
        if reference["session_id"] != frame["session_id"]:
            raise ValueError(f"Receiver recording session mismatch: {fid}")
    rows = []
    for frame in frames:
        fid = frame["frame_id"]
        for review_pass in (["primary", "repeat"] if frame.get("repeat_review") else ["primary"]):
            if fid in excluded_ids:
                status = "EXCLUDED_BY_USER"
            elif review_pass == "primary" and fid in references:
                status = "REUSE_COMPLETED_EXPLORATORY_GEOMETRY"
            elif review_pass == "repeat":
                status = "PENDING_REAL_REPEAT_RECORD"
            else:
                status = "PENDING_REFERENCE"
            rows.append({
                "task_id": f"{fid}|{review_pass}", "frame_id": fid,
                "session_id": frame["session_id"], "saved_frame_index": frame["saved_frame_index"],
                "review_pass": review_pass, "image_sha256": frame["image_sha256"],
                "decoded_bgr_sha256": frame["decoded_bgr_sha256"],
                "raw_video_sha256": frame["raw_video_sha256"],
                "camera_sensor_timestamp_ms": frame["camera_sensor_timestamp_ms"],
                "task_status": status,
                "completed_primary_image_reference_exists": fid in references,
                "actual_repeat_record_exists": False,
                "approved_official_direct_visible_reference": False,
                "legacy_logical_candidate": fid in candidate_ids,
                "legacy_pixel_mapping_status": "UNKNOWN_PIXEL_UNVERIFIED" if fid in candidate_ids else "NOT_APPLICABLE",
            })
    return rows


def walk_records(value):
    if isinstance(value, dict):
        if "frame_id" in value and "review_pass" in value:
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def load_readonly_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_reference_gate(receiver, recovery, frame_ids, manifest):
    """Execute the actual existing CPU gate; never turn saved drafts into approvals."""
    gate_path = receiver / "scripts/research/pallet_combined_closeout_20261003_v1/evaluate_lifter_connection_20261006.py"
    gate = load_readonly_module(gate_path, "existing_lifter_reference_gate")
    expected_bundle = receiver / LIFTER / "review/LIFTER_REFERENCE_REVIEWED.json"
    gate_status = gate.reference_gate(read_json(expected_bundle) if expected_bundle.exists() else recovery, frame_ids)
    evaluator_path = receiver / "scripts/research/pallet_lifter_case_review_20261003_v1/metrics/evaluate.py"
    evaluator = load_readonly_module(evaluator_path, "existing_lifter_reference_validator")
    try:
        evaluator._validate_reference(recovery, manifest["frames"], recovery["bindings"])
    except evaluator.ContractError as error:
        recovery_rejection = str(error)
    else:
        recovery_rejection = None
    audit = {
        "actual_existing_reference_gate_status": gate_status,
        "required_reviewed_bundle_relative_path": str(expected_bundle.relative_to(receiver)),
        "actual_reviewed_bundle_exists": expected_bundle.exists(),
        "actual_existing_validator_rejection_for_current_snapshot": recovery_rejection,
        "gate_source_sha256": sha_bytes(gate_path.read_bytes()),
        "validator_source_sha256": sha_bytes(evaluator_path.read_bytes()),
        "new_blindness_requirement_added": False,
        "record_gate_findings": [],
        "actual_bulk_approval_receipt_paths": [str(p.relative_to(receiver)) for p in sorted((receiver / LIFTER / "review").rglob("APPROVAL_RECEIPT.json"))],
        "actual_native_visibility_saved_files": [],
    }
    for path in sorted((receiver / LIFTER / "review/native_annotations").rglob("*.VISIBILITY_SAVED.json")):
        saved = read_json(path)
        audit["actual_native_visibility_saved_files"].append({
            "path": str(path.relative_to(receiver)), "sha256": sha_bytes(path.read_bytes()),
            "frame_id": saved["frame_id"], "source_kind": saved["source_kind"],
            "evaluation_use": saved["evaluation_use"], "provenance_status": saved["provenance_status"],
            "actual_human_save_recognized_without_promotion": True,
        })
    for fid in frame_ids:
        record = recovery["drafts"][fid + "|primary"]["record"]
        missing = []
        if record.get("source_kind") != "human_reviewed":
            missing.append("record.source_kind=human_reviewed")
        if record.get("status") not in ("reviewed", "skipped"):
            missing.append("record.status=reviewed_or_skipped")
        actor = record.get("reviewer")
        if not isinstance(actor, dict) or actor.get("entered_by") != "human" or actor.get("confirmation") is not True:
            missing.append("record.reviewer.entered_by=human_and_confirmation=true")
        if not record.get("review_time"):
            missing.append("record.actual_review_time")
        audit["record_gate_findings"].append({"frame_id": fid, "source_pointer": f"drafts.{fid}|primary.record",
                                               "actual_status": record.get("status"), "actual_reviewer": actor,
                                               "missing_existing_contract_fields": missing})
    if gate_status != "WAITING_HUMAN_REFERENCE_SUBMISSION" or recovery_rejection is None:
        raise ValueError("Receiving formal reference state differs; evaluate actual newly submitted contract")
    return audit


def sensitivity_panel(receiver, reference, recovery, matches, result):
    """Fixed later-coordinate sensitivity, deliberately not a submitted GT panel."""
    path, raw = checked_binding(receiver, result["sources"]["predictions"])
    needed = {item["frame_id"] for item in reference["records"]}
    predictions = {}
    for line in raw.splitlines():
        item = json.loads(line)
        if item["frame_id"] in needed:
            if item["frame_id"] in predictions:
                raise ValueError("Duplicated frozen prediction frame")
            predictions[item["frame_id"]] = item
    helper_path, _ = checked_binding(receiver, result["sources"]["script"])
    helper = load_readonly_module(helper_path, "existing_assisted_metric_functions")
    decisions = {item["frame_id"]: item["decision"] for item in matches["records"]}
    metrics96, points96, _ = helper.compute(reference["records"], predictions, decisions)
    if metrics96 != result["metrics"]:
        raise ValueError("Original 96-point metric regression failed")
    lookup96 = {(row["frame_id"], row["point_id"]): row for row in points96}
    points72, frame_rows = [], []
    unchanged_manual, changed_saved = 0, 0
    for ref in reference["records"]:
        fid = ref["frame_id"]
        record = recovery["drafts"][fid + "|primary"]["record"]
        if [c["id"] for c in record["corners"]] != list(range(8)):
            raise ValueError("Later snapshot corner IDs changed")
        predicted = predictions[fid]
        if predicted["decoded_bgr_sha256"] != ref["decoded_bgr_sha256"]:
            raise ValueError("Later sensitivity prediction image identity mismatch")
        human = next(item for item in matches["records"] if item["frame_id"] == fid)
        for method in ("Base", "N3"):
            selection = predicted["methods"][method]
            if human["selected_index"] != selection["selected_index"] or human["selected_box_xyxy"] != selection["selected_object"]["selected_box_xyxy"]:
                raise ValueError("Sensitivity reused human matching for a different selected prediction")
        slots = {m: helper.prediction_points(predicted["methods"][m]) for m in ("Base", "N3")}
        frame_points = []
        for corner in record["corners"]:
            if corner["x"] is None and corner["y"] is None:
                if corner.get("self_occlusion") is not True:
                    raise ValueError("Later no-coordinate slots are not the preserved 24 self states")
                continue
            index = corner["id"]
            if corner["visibility"] != "direct_visible" or corner["definition_confirmed"] is not True:
                raise ValueError("Later saved coordinate has uncertain visibility/corner definition")
            if any(corner.get(key) for key in ("external_occlusion", "self_occlusion", "out_of_frame", "definition_uncertain")):
                raise ValueError("Later saved coordinate has contradictory visibility")
            xy = [corner["x"], corner["y"]]
            if not all(helper.finite(v) for v in xy) or not (0 <= xy[0] < ref["width"] and 0 <= xy[1] < ref["height"]):
                raise ValueError("Later coordinate is invalid or out of image")
            original = ref["points"][index]
            same_xy = xy == [original["x"], original["y"]]
            unchanged_manual += same_xy and original["source"] == "manual_click"
            changed_saved += not same_xy
            point = {"frame_id": fid, "session_id": ref["session_id"], "point_id": index,
                     "reference_x": xy[0], "reference_y": xy[1],
                     "reference_source": "later_saved_direct_coordinate_unsubmitted_draft",
                     "object_match": decisions[fid], "draft_status": record["status"],
                     "formal_evaluation_use": False, "original_96_reference_source": original["source"],
                     "coordinate_equal_original_96_point": same_xy}
            for method in ("Base", "N3"):
                coords, mask = slots[method]
                reason = None
                if decisions[fid] == "different":
                    reason = "wrong_selected_object"
                elif predicted["methods"][method]["selected_index"] is None:
                    reason = "no_selected_object"
                elif not mask[index]:
                    reason = "missing_or_invalid_prediction"
                error = None if reason else math.hypot(coords[index][0] - xy[0], coords[index][1] - xy[1])
                point[f"{method}_error_px"] = error
                point[f"{method}_failure_reason"] = reason
                point[f"{method}_pck10_hit"] = error is not None and error <= 10.0
                point[f"original_96_{method}_error_px"] = lookup96[(fid, index)][f"{method}_error_px"]
            point["paired_N3_minus_Base_px"] = point["N3_error_px"] - point["Base_error_px"] if all(point[f"{m}_error_px"] is not None for m in ("Base", "N3")) else None
            points72.append(point)
            frame_points.append(point)
        frame_rows.append({"frame_id": fid, "session_id": ref["session_id"], **helper.summary(frame_points)})
    if len(points72) != 72 or unchanged_manual != 66 or changed_saved != 6:
        raise ValueError("Later fixed 72 slots changed; never choose a version by error")
    metrics72 = helper.summary(points72)
    metrics72["frame_count"] = len(frame_rows)
    metrics72["sessions"] = {session: helper.summary([row for row in points72 if row["session_id"] == session])
                             for session in dict.fromkeys(row["session_id"] for row in points72)}
    # Keep denominator change and six coordinate changes visible separately.
    fixed72_original = [lookup96[(p["frame_id"], p["point_id"])] for p in points72]
    intermediate = helper.summary(fixed72_original)
    return {
        "schema_version": "unsubmitted_later_reference_version_sensitivity_v1",
        "analysis_scope": "separate_exploratory_reference_version_sensitivity_only",
        "source_kind": "CPU_comparison_against_existing_unsubmitted_unverified_human_draft_snapshot",
        "formal_reference_evaluation_use": False, "formal_reference_approval_created": False,
        "official_direct_visible_accuracy": "x", "original_96_result_replaced": False,
        "fixed_frames": [item["frame_id"] for item in reference["records"]], "later_fixed_point_denominator": 72,
        "known_no_coordinate_self_occlusion_slots": 24, "source_snapshot_original_evaluation_use": False,
        "original_manual_points_preserved_exactly": unchanged_manual,
        "later_saved_coordinates_replacing_original_pnp_coordinates": changed_saved,
        "canonical_association": "same_original_frame_id_and_fixed_0_to_7_slot_no_permutation_no_symmetry_search",
        "reference_version_selected_by_error": False, "threshold": "Euclidean error <= 10 px",
        "PCK_failures_remain_in_full_fixed_72_denominator": True,
        "source_bindings": {"later_snapshot": reference["later_current_manual_snapshot"]["binding"],
                            "raw_predictions": result["sources"]["predictions"],
                            "original_metric_script": result["sources"]["script"],
                            "actual_human_object_match": result["sources"]["human_match_sidecar"]},
        "original_96_metric_regression": "PASS", "original_all96_metrics": metrics96,
        "fixed72_slot_original_reference_coordinates_metrics": intermediate,
        "later_fixed72_saved_draft_coordinates_metrics": metrics72,
        "frame_results72": frame_rows, "new_model_forward": 0, "training_runs": 0,
    }, points72


def source_points(payload):
    sources = Counter()
    for obj in payload.get("objects", []):
        # Only explicit saved per-point attribution. manual_kps is not proof.
        points = obj.get("keypoint_annotations", [])
        for point in points[:8]:
            sources[point.get("source", "UNKNOWN")] += 1
    return sources


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish an empty evidence table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def immutable_snapshot(handoff, output, inventory):
    """Make portable byte-identical small evidence copies; originals remain intact."""
    verified = handoff / "annotations/verified"
    destination = output / "inputs/annotations/verified"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in inventory["copied_sources"]:
        relative = Path(item["handoff_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe handoff copy path")
        data = checked_bytes(verified / relative, item)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Existing immutable snapshot differs: {target}")
        target.write_bytes(data)
        copied.append({"source_id": item["source_id"], "path": str(target.relative_to(output)),
                       "bytes": len(data), "sha256": sha_bytes(data)})
    for relative in ["ANNOTATION_INVENTORY.json", "SOURCE_CONTAINERS.json", "FRAME_REFERENCE_MATCH.csv",
                     "INPUTS_FOR_ROOT.json", "HANDOFF_INTEGRITY_CHECK.json", "LEGACY67_GIT_RECOVERY.json",
                     "OVERLAP_SOURCE_IMAGE_CANDIDATE_CHECK.json", "OTHER_RECORDING_PIXEL_MATCH.json"]:
        source = verified / relative
        data = read_bytes(source if source.exists() else source.with_suffix(source.suffix + ".gz"))
        target = destination / (relative + ".gz" if relative == "ANNOTATION_INVENTORY.json" else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = gzip.compress(data, mtime=0) if target.suffix == ".gz" else data
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Existing immutable metadata snapshot differs: {target}")
        target.write_bytes(content)
        copied.append({"path": str(target.relative_to(output)), "bytes": len(content),
                       "sha256": sha_bytes(content), "source_uncompressed_bytes": len(data),
                       "source_uncompressed_sha256": sha_bytes(data)})
    write_json(output / "SNAPSHOT_MANIFEST.json", {"source_kind": "immutable_input_snapshot", "files": copied})
    return copied


def run(handoff, receiver, output_root, archive_roots=()):
    start = time.perf_counter()
    cpu_start = time.process_time()
    output = output_root / f"data/pallet/results/{NAME}/annotations"
    docs = output_root / f"_docs/experiments/{NAME}/annotations"
    verified = handoff / "annotations/verified"
    inventory = read_json(verified / "ANNOTATION_INVENTORY.json")
    plan_path = receiver / LIFTER / "LIFTER_EVALUATION_PLAN.json"
    manifest_path = receiver / LIFTER / "review/MANIFEST.json"
    contract_path = receiver / LIFTER / "review/CORNER_CONTRACT.json"
    for path, key in [(plan_path, "original_plan_sha256"), (manifest_path, "manifest_sha256"),
                      (contract_path, "corner_contract_sha256")]:
        checked_bytes(path, {"sha256": inventory[key]})
    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    if len(plan["frames"]) != plan["stored_frame_count"]:
        raise ValueError("Stored frame plan count mismatch")
    if [f["frame_id"] for f in manifest["frames"]] != plan["review_ids"]:
        raise ValueError("Frozen primary frame sequence mismatch")
    for frame in manifest["frames"]:
        original = next(f for f in plan["frames"] if f["frame_id"] == frame["frame_id"])
        for field in ("decoded_bgr_sha256", "raw_video_sha256", "camera_sensor_timestamp_ms"):
            if original[field] != frame[field]:
                raise ValueError(f"Stored plan identity mismatch: {frame['frame_id']} {field}")
    reference = read_json(receiver / ASSISTED / "ASSISTED_REFERENCE.json")
    result = read_json(receiver / RESULT)
    source_checks = []
    for key, binding in result["sources"].items():
        path, data = checked_binding(receiver, binding)
        source_checks.append({"role": key, "original_binding_path": binding["path"],
                              "resolved_path": str(path), "bytes": len(data), "sha256": sha_bytes(data)})
    matches = read_json(receiver / ASSISTED / "LIFTER_OBJECT_MATCH_REVIEWED.json")
    match_by_id = {item["frame_id"]: item for item in matches["records"]}
    if len(match_by_id) != len(matches["records"]):
        raise ValueError("Duplicate human object matching records")
    refs = {item["frame_id"]: item for item in reference["records"]}
    if len(refs) != len(reference["records"]):
        raise ValueError("Duplicate receiver reference frame")
    exclusions = read_json(receiver / LIFTER / "review/USER_EXCLUSIONS.json")
    excluded = set(exclusions["excluded_frame_ids"])
    candidates = set(inventory["legacy67_candidate_original_primary_frame_ids"])
    rows = join_tasks(manifest["frames"], refs, excluded, candidates)
    frame_checks = []
    sources_count = Counter()
    actual_image_hash_count = 0
    for fid, item in refs.items():
        path, raw = checked_binding(receiver, item["source_pnp_binding"])
        payload = json.loads(raw)
        if payload["frame_id"] != fid or payload["review_pass"] != "primary":
            raise ValueError("PnP source frame/pass mismatch")
        if payload["image_sha256"] != item["image_sha256"]:
            raise ValueError("PnP source image SHA mismatch")
        if payload["evaluation_use"] is not False or payload["human_provenance_confirmed"] is not False:
            raise ValueError("Source approval flags changed")
        if len(item["points"]) != 8 or [p["id"] for p in item["points"]] != list(range(8)):
            raise ValueError("Receiver corner population/order changed")
        for point, xy, annotation in zip(item["points"], payload["editor_kps_2d"][:8], payload["editor_keypoint_annotations"][:8]):
            if [point["x"], point["y"]] != xy or point["source"] != annotation["source"]:
                raise ValueError("Receiver source corner was changed or renumbered")
            if not all(math.isfinite(x) for x in xy):
                raise ValueError("Receiver finite reference coordinate missing")
            sources_count[point["source"]] += 1
        human = match_by_id.get(fid)
        if not human or human["source_kind"] != "human_reviewed" or human["decision"] != "same":
            raise ValueError("Actual human object correspondence unavailable")
        if human["target_object_id"] != item["target_object_id"]:
            raise ValueError("Human matching target changed")
        image = receiver / LIFTER / "review" / item["image_path"]
        if not image.exists():
            image = receiver / "_docs/experiments/pallet_paper_review_20261006_v1/portable_evidence/reference_images" / Path(item["image_path"]).name
        image_data = checked_bytes(image, {"sha256": item["image_sha256"]})
        actual_image_hash_count += 1
        frame_checks.append({"frame_id": fid, "image_sha256": sha_bytes(image_data),
                             "source_pnp_sha256": sha_bytes(raw), "source_keypoint_frame": item["keypoint_frame"],
                             "canonical_ids_reused_without_remapping": list(range(8)),
                             "formal_reference_approval_promoted": False, "human_object_decision": "same"})
    if dict(sources_count) != reference["source_counts"]:
        raise ValueError("Receiver original direct/PnP source count changed")
    recovery_path = receiver / LIFTER / "review/VIEWER_DRAFT_RECOVERY.json"
    checked_bytes(recovery_path, reference["later_current_manual_snapshot"]["binding"])
    recovery = read_json(recovery_path)
    later = Counter()
    for fid in refs:
        record = recovery["drafts"][f"{fid}|primary"]["record"]
        for corner in record["corners"]:
            if corner.get("x") is not None and corner.get("y") is not None:
                later["saved_coordinate_count"] += 1
            if corner.get("self_occlusion") is True:
                later["self_occluded_state_count"] += 1
    if later != Counter({"saved_coordinate_count": 72, "self_occluded_state_count": 24}):
        raise ValueError("Later separate snapshot population changed")
    reference_gate = current_reference_gate(receiver, recovery, list(refs), manifest)
    sensitivity, sensitivity_points = sensitivity_panel(receiver, reference, recovery, matches, result)
    for row in rows:
        fid = row["frame_id"]
        if fid in refs:
            original_sources = Counter(point["source"] for point in refs[fid]["points"])
            corners = recovery["drafts"][fid + "|primary"]["record"]["corners"]
            row.update(original_primary_manual_click_points=original_sources["manual_click"],
                       original_primary_pnp_projected_points=original_sources["pnp_projected"],
                       later_primary_saved_coordinate_count=sum(c["x"] is not None and c["y"] is not None for c in corners),
                       later_primary_self_occlusion_state_count=sum(c["self_occlusion"] for c in corners),
                       later_primary_actual_submission_status="draft_not_human_reviewed_submission")
        else:
            row.update(original_primary_manual_click_points="x", original_primary_pnp_projected_points="x",
                       later_primary_saved_coordinate_count="x", later_primary_self_occlusion_state_count="x",
                       later_primary_actual_submission_status="NO_COMPLETED_RECEIVER_REFERENCE")
    repeat_hits = []
    for search in [receiver / LIFTER / "review", receiver / "data/pallet/results/pallet_combined_closeout_20261003_v1/human_review"]:
        if not search.exists():
            continue
        for path in sorted(search.rglob("*.json")):
            for record in walk_records(read_json(path)):
                if record["review_pass"] == "repeat":
                    repeat_hits.append({"path": str(path.relative_to(receiver)), "frame_id": record["frame_id"],
                                        "evaluation_use": record.get("evaluation_use", "UNKNOWN"),
                                        "status": record.get("status", "UNKNOWN")})
    # This audit intentionally does not manufacture an approval policy for a new schema.
    if repeat_hits:
        raise ValueError("Real repeat candidates exist: inspect their actual protocol before reporting counts")
    local_index = {}
    for search in [receiver / "outputs/annotations", receiver / "challenge/data/01_real/live_capture_gt"]:
        if search.exists():
            for path in sorted(search.rglob("*.json")):
                data = path.read_bytes()
                local_index.setdefault(sha_bytes(data), []).append(str(path.relative_to(receiver)))
    old_sources = {item["source_id"]: item for item in inventory["actual_annotation_sources"]}
    containers = read_json(verified / "SOURCE_CONTAINERS.json")["containers"]
    container_checks, actual_containers = [], {}
    for container in containers:
        basename = PureWindowsPath(container["absolute_path"]).name
        found = [root / basename for root in archive_roots if (root / basename).is_file()]
        for path in found:
            checked_bytes(path, container)
        if found:
            actual_containers[container["absolute_path"]] = zipfile.ZipFile(found[0])
        container_checks.append({**container, "receiver_container_status": "EXACT_EXISTING_CONTAINER" if found else "NOT_PRESENT_IN_RECEIVER_SEARCH_ROOTS",
                                 "receiver_search_roots": [str(root) for root in archive_roots],
                                 "receiver_exact_paths": [str(path) for path in found],
                                 "member_hash_is_not_container_hash": True})
    source_rows, explicit_sources = [], Counter()
    for item in inventory["copied_sources"]:
        source_path = verified / item["handoff_relative_path"]
        data = checked_bytes(source_path, item)
        source = old_sources[item["source_id"]]
        explicit_sources.update(source_points(json.loads(data)))
        declared_sources = Counter(corner["source"] for obj in source["records"] for corner in obj["corners"])
        if declared_sources != source_points(json.loads(data)):
            raise ValueError(f"Saved per-point attribution mismatch: {item['source_id']}")
        original = item["original_source"]
        blob_status = "NOT_APPLICABLE"
        if original.get("git_blob_sha1"):
            environment = dict(os.environ, GIT_NO_LAZY_FETCH="1")
            proc = subprocess.run(["git", "cat-file", "blob", original["git_blob_sha1"]], cwd=receiver,
                                  env=environment, capture_output=True, check=False)
            blob_status = "EXACT_EXISTING_GIT_BLOB" if proc.returncode == 0 and proc.stdout == data else "UNAVAILABLE_OR_DIFFERENT"
            if blob_status != "EXACT_EXISTING_GIT_BLOB":
                raise ValueError(f"Existing receiver Git blob mismatch: {item['source_id']}")
        pixel = source.get("pixel_evidence") or {}
        archive_status = "SOURCE_CONTAINER_NOT_PRESENT"
        source_pixel_hash_status = "SOURCE_IMAGE_NOT_PRESENT"
        container = actual_containers.get(original["absolute_path"])
        if container is not None and original.get("archive_member"):
            if container.read(original["archive_member"]) != data:
                raise ValueError(f"Receiver archive member differs: {item['source_id']}")
            archive_status = "EXACT_EXISTING_ARCHIVE_MEMBER"
            if pixel.get("image_locator"):
                image_data = container.read(pixel["image_locator"])
                if sha_bytes(image_data) != pixel["image_sha256"]:
                    raise ValueError(f"Receiver original annotation image differs: {item['source_id']}")
                source_pixel_hash_status = "EXACT_ENCODED_IMAGE_SHA_NOT_NEW_DECODED_PIXEL_AUDIT"
        source_rows.append({"source_id": item["source_id"], "handoff_relative_path": item["handoff_relative_path"],
                            "bytes": len(data), "sha256": sha_bytes(data),
                            "original_source_path": original["absolute_path"],
                            "archive_member": original.get("archive_member", ""),
                            "git_blob_sha1": original.get("git_blob_sha1", ""), "receiver_git_blob_status": blob_status,
                            "receiver_identical_file_paths": "|".join(local_index.get(sha_bytes(data), [])),
                            "receiver_source_archive_member_status": archive_status,
                            "receiver_original_source_image_status": source_pixel_hash_status,
                            "reported_frame_id_only": source.get("reported_frame_id_from_filename") or "",
                            "exact_same_plan_image_ids": "|".join(source.get("exact_pixel_frame_ids", [])),
                            "source_image_sha256": pixel.get("image_sha256", "UNKNOWN"),
                            "source_decoded_bgr_sha256": pixel.get("decoded_bgr_sha256", "UNKNOWN"),
                            "canonical_mapping_status": source.get("canonical_mapping_status", "UNKNOWN"),
                            "historical_review_pass": source.get("review_pass", "UNKNOWN"),
                            "manual_click_points": declared_sources["manual_click"],
                            "pnp_projected_points": declared_sources["pnp_projected"],
                            "unknown_point_sources": declared_sources["UNKNOWN"]})
    if len(source_rows) != 312 or explicit_sources != Counter(inventory["preserved_point_source_counts"]):
        raise ValueError("Handoff 312 original source population or point sources changed")
    for container in actual_containers.values():
        container.close()
    task_counts = Counter(row["task_status"] for row in rows)
    by_pass = {review_pass: dict(Counter(row["task_status"] for row in rows if row["review_pass"] == review_pass))
               for review_pass in ["primary", "repeat"]}
    unique = Counter()
    for frame in manifest["frames"]:
        fid = frame["frame_id"]
        unique["excluded" if fid in excluded else "existing_exploratory_primary" if fid in refs else "pending_reference"] += 1
    snapshot = immutable_snapshot(handoff, output, inventory)
    checks = {
        "frozen_plan_and_manifest_hashes_match": True, "stored_frames": len(plan["frames"]),
        "primary_image_identity_checks": len(manifest["frames"]), "receiver_actual_original_image_sha_checks": actual_image_hash_count,
        "original_pnp_file_checks": len(frame_checks), "unchanged_reference_coordinate_slots": sum(sources_count.values()),
        "actual_human_same_object_records": len(match_by_id), "immutable_original_json_copies_verified": len(source_rows),
        "receiver_legacy_git_blobs_exact": sum(row["receiver_git_blob_status"] == "EXACT_EXISTING_GIT_BLOB" for row in source_rows),
        "receiver_identical_existing_annotation_files": sum(bool(row["receiver_identical_file_paths"]) for row in source_rows),
        "receiver_existing_source_archive_members_exact": sum(row["receiver_source_archive_member_status"] == "EXACT_EXISTING_ARCHIVE_MEMBER" for row in source_rows),
        "receiver_existing_source_annotation_images_encoded_sha_exact": sum(row["receiver_original_source_image_status"].startswith("EXACT_ENCODED_IMAGE_SHA") for row in source_rows),
        "all_same_plan_legacy_pixel_matches": sum(bool(row["exact_same_plan_image_ids"]) for row in source_rows),
        "original_96_metrics_regression": "PASS", "later_draft72_sensitivity_computed": True,
        "formal_reference_bundle_actual_existing_gate": reference_gate["actual_existing_reference_gate_status"],
        "existing_repeat_record_search_hits": len(repeat_hits), "primary_records_used_as_repeat": 0,
        "original_approval_flags_promoted": 0, "original_annotation_files_rewritten": 0,
        "new_click_requests": 0, "new_annotations": 0, "new_training": 0, "new_model_inference": 0,
        "optimizer_updates": 0, "hardware_control": 0, "PDF_generation": 0,
    }
    connection = {
        "schema_version": "receiver_annotation_connection_v1", "source_kind": "CPU_identity_and_saved_source_verification",
        "source_bindings": {"plan_sha256": inventory["original_plan_sha256"], "manifest_sha256": inventory["manifest_sha256"],
                            "corner_contract_sha256": inventory["corner_contract_sha256"],
                            "handoff_inventory_sha256": sha_bytes(read_bytes(verified / ("ANNOTATION_INVENTORY.json" if (verified / "ANNOTATION_INVENTORY.json").exists() else "ANNOTATION_INVENTORY.json.gz"))),
                            "receiver_assisted_reference_sha256": sha_bytes((receiver / ASSISTED / "ASSISTED_REFERENCE.json").read_bytes()),
                            "receiver_original_result_sha256": sha_bytes((receiver / RESULT).read_bytes()),
                            "receiver_user_exclusions_sha256": sha_bytes((receiver / LIFTER / "review/USER_EXCLUSIONS.json").read_bytes())},
        "denominators": {"frozen_primary": 120, "frozen_repeat_tasks": 24, "frozen_tasks": len(rows), "unique_images": len(manifest["frames"])},
        "disjoint_task_status_counts": dict(task_counts), "task_status_by_pass": by_pass,
        "disjoint_unique_image_status_counts": dict(unique), "existing_exploratory_geometry": {
            "frames": len(refs), "points": sum(sources_count.values()), "source_counts": dict(sources_count),
            "actual_human_object_same_decisions": len(match_by_id), "metrics_reused_without_reestimation": result["metrics"],
            "official_direct_visible_reference_approval_created": False, "existing_work_needs_reannotation": False},
        "later_separate_current_snapshot": {**dict(later), "used_for_original_geometry_result": False,
                                             "is_independent_repeat": False, "unrelated_extra_draft_count": len(recovery["drafts"]) - len(refs)},
        "actual_receiving_reference_gate": reference_gate,
        "later_draft72_sensitivity_result": "LATER_DRAFT72_SENSITIVITY.json",
        "repeat_images_with_existing_primary_geometry": sum(row["review_pass"] == "repeat" and row["completed_primary_image_reference_exists"] for row in rows),
        "approved_original_120_24_reference_tasks": 0, "excluded_unique_frame_ids": sorted(excluded),
        "legacy_logical_candidate_ids": sorted(candidates), "legacy_logical_candidates_not_upgraded": True,
        "old_annotation_source_counts": {"legacy_four_session": 67, "other_recording_sessions": 245,
                                         "all": len(source_rows), "explicit_point_sources": dict(explicit_sources)},
        "source_container_checks": container_checks, "frame_reference_checks": frame_checks,
        "original_result_source_checks": source_checks, "validation": checks,
        "remaining_values": {"official_120_24_visible_accuracy": "x", "repeat_reliability": "x", "independent_physical_TR": "x"},
    }
    write_csv(output / "FRAME_TASK_CONNECTION.csv", rows)
    write_csv(output / "HANDOFF_SOURCE_CONNECTION.csv", source_rows)
    write_json(output / "CONNECTION.json", connection)
    write_json(output / "LATER_DRAFT72_SENSITIVITY.json", sensitivity)
    write_csv(output / "LATER_DRAFT72_POINT_ERRORS.csv", sensitivity_points)
    validation = {"status": "PASS", "checks": checks, "task_partition_sum": sum(task_counts.values()),
                  "unique_image_partition_sum": sum(unique.values()), "published_input_snapshots": len(snapshot),
                  "actual_wall_seconds": time.perf_counter() - start, "actual_cpu_seconds": time.process_time() - cpu_start,
                  "command": "python3 scripts/research/pallet_remaining_evidence_connection_20261006_v1/annotation_connection.py --handoff <extracted_handoff_or_annotations_inputs> --receiver-root <existing_repo> --output-root <result_repo>"}
    write_json(output / "VALIDATION.json", validation)
    docs.mkdir(parents=True, exist_ok=True)
    sensitivity_metrics = sensitivity["later_fixed72_saved_draft_coordinates_metrics"]
    sensitivity_rows = []
    for method in ("Base", "N3"):
        metric = sensitivity_metrics["methods"][method]
        sensitivity_rows.append(f"| {method} | {metric['conditional_error']['median_px']:.6f} | {metric['conditional_error']['p90_px']:.6f} | {metric['pck10_hit_count']}/72 ({metric['pck10_full_reference_percent']:.4f}%) | {metric['valid_matching_prediction_points']} / {metric['failed_reference_points']} |")
    source_archive_count = checks["receiver_existing_source_archive_members_exact"]
    image_archive_count = checks["receiver_existing_source_annotation_images_encoded_sha_exact"]
    local_copy_count = checks["receiver_identical_existing_annotation_files"]
    findings = f"""# 기존 주석과 인계 자료 연결 결과

완료했던 **12장·96점과 12개 실제 대상 확인을 그대로 재사용**했습니다. 다시 클릭할 작업을 열지 않았습니다. 원래 120장·반복 24장의 공식 가시 코너 평가 완료로 바꾸지는 않았습니다.

| 범위 | 완료 자료 재사용 | 사용자 제외 | 참조·실제 반복 기록 대기 | 합계 |
|---|---:|---:|---:|---:|
| 원래 1차 작업 | 12 | 5 | 103 | 120 |
| 원래 반복 작업 | 0 | 1 | 23 | 24 |
| 작업 합계 | 12 | 6 | 126 | 144 |
| 서로 다른 이미지 | 12 | 5 | 103 | 120 |

반복 24개는 24장의 새로운 이미지가 아닙니다. 완료 12장의 이미지 중 6장은 원래 반복 목록에도 들어 있지만, 그 1차 기록을 두 번째 실제 검수처럼 복사하지 않았습니다. 사용자 제외 5장 중 `174925:620`은 반복 목록에도 있어 작업 기준 제외는 6개, 이미지 기준 제외는 5장입니다.

기존 결과가 사용한 처음 저장한 점은 직접 클릭 66점 + PnP로 채운 30점입니다. 나중에 저장한 직접 좌표 72점 + 자체 가림 상태 24개는 **다른 버전**으로 보존했습니다. 13개 초안 중 이번 12장 밖의 초안 1개도 원래 평가에 넣지 않았습니다.

전달된 과거 주석 312개(같은 네 세션 이름의 67개, 다른 촬영의 245개)의 실제 JSON 크기·SHA와 저장된 점 출처를 검산했습니다. 같은 세션 이름의 67개는 로컬 Git blob과 바이트가 같습니다. 이번 수신 PC에서 실제 기존 ZIP 멤버와 바이트가 같았던 JSON은 {source_archive_count}개, 인계 SHA와 같았던 원사진은 {image_archive_count}개입니다. 로컬에 풀린 동일 JSON은 {local_copy_count}개였습니다. 원래 수신 PC의 두 ZIP 본체와 245개 멤버를 모두 검증한 실행 기록은 CONNECTION.json에 개별 경로·SHA와 함께 남깁니다. 기존 인벤토리의 점 출처 1,271개 직접 클릭 / 1,225개 PnP도 실제 저장 필드와 맞습니다. 원래 120장에 연결 가능한 과거 픽셀·코너 대응 확정은 0개입니다.

`173507:2910`, `173507:3210`은 파일명 기반 후보로 유지합니다. 이전 도구의 `camera_dynamic_0123_v4` 번호와 정확한 원사진 연결이 아직 확인되지 않아 정답으로 승격하지 않았습니다. 다른 촬영 245장은 인계된 CPU 픽셀 감사가 원래 촬영과 다름을 기록합니다. 이번 재실행은 영상이나 PNG를 다시 디코딩한 픽셀 감사가 아니라 실제 저장 바이트의 동일성 검산이며, 원본 ZIP이 없는 다른 PC에서는 그 검산을 수행했다고 표시하지 않습니다.

## 실제 수신 기록에서 확인한 공식 참조 조건

수신 PC의 `review/LIFTER_REFERENCE_REVIEWED.json`과 실제 일괄 승인 `APPROVAL_RECEIPT.json`이 없습니다. 현재 원본 snapshot의 이번 12개 record는 `status=draft`, `reviewer=null`, 최상위 `evaluation_use=false`입니다. 기존 코드의 `reference_gate`와 `_validate_reference`를 직접 호출하면 각각 `WAITING_HUMAN_REFERENCE_SUBMISSION`과 미승인 bundle 거절을 반환합니다. 72개 좌표와 각 코너 상태가 저장되지 않았다는 뜻이 아니라, 저장한 입력을 현재 계약의 공식 참조로 제출한 기록이 없다는 뜻입니다. 기존 실제 사람 이력 답변과 대상 판정 12개는 보존했습니다. 새로운 맹검 조건을 추가하지 않았습니다.

## 나중 저장한 72점에 대한 별도 민감도 계산

고정 12장·같은 Base/N3 원시 예측·선택 객체·결측 마스크·고정 ID를 그대로 사용했습니다. 24개 자체 가림 상태에는 좌표를 만들어 넣지 않았습니다. 원래 직접 클릭 좌표 66개는 완전히 같고, 원래 PnP 좌표였던 6개가 나중 저장된 좌표로 바뀌었습니다. 정확도가 좋은 버전을 선택하지 않고 원래 96점과 현재 72점 두 버전을 모두 공개합니다.

**아래 표는 승인 미제출 초안 좌표를 사용한 탐색적 참조 버전 민감도입니다. 공식 가시 코너 정확도나 독립 정답이 아닙니다.** 원래 원고의 96점 핵심 표를 교체하지 않습니다.

| 방법 | 중앙값(px) | P90(px) | PCK≤10px 전체72점 | 유효 / 실패 |
|---|---:|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

중앙값 차이 `median(N3)-median(Base)`는 {sensitivity_metrics['paired']['median_after_minus_median_before_px']:.6f}px, 짝지은 차이 중앙값 `median(N3-Base)`는 {sensitivity_metrics['paired']['median_of_paired_after_minus_before_px']:.6f}px입니다. 개선 {sensitivity_metrics['paired']['improved_points']} / 악화 {sensitivity_metrics['paired']['worsened_points']} / 동일 {sensitivity_metrics['paired']['tied_points']}점입니다. 이 버전에서도 중앙값 악화를 유지합니다. 원래 96점 수치 재계산은 원본 결과와 정확히 일치했습니다.

분모 변경 효과와 좌표 변경 효과를 구분하도록 같은 72개 ID에 원래 저장 좌표를 적용한 중간 패널도 JSON에 함께 저장했습니다. 72점과 96점의 결과 차이를 모델 성능 변화로 해석하지 않습니다.

원고에서 유지할 `x`: 원래 전체 120장·반복24장 공식 가시 코너 정확도, 반복 신뢰도, 독립 물리 T/R. 기존 12장 결과는 별도 보조 결과로 계속 쓸 수 있습니다.

- [144개 작업별 연결 CSV](../../../../data/pallet/results/{NAME}/annotations/FRAME_TASK_CONNECTION.csv)
- [312개 원본 주석별 연결 CSV](../../../../data/pallet/results/{NAME}/annotations/HANDOFF_SOURCE_CONNECTION.csv)
- [출처·버전·분모와 기존 수치](../../../../data/pallet/results/{NAME}/annotations/CONNECTION.json)
- [실행 검산 및 실제 CPU 비용](../../../../data/pallet/results/{NAME}/annotations/VALIDATION.json)
- [원래96점·같은72개ID 원래좌표·나중72개좌표 민감도](../../../../data/pallet/results/{NAME}/annotations/LATER_DRAFT72_SENSITIVITY.json)
- [나중72개 점별 오차와 원래 좌표 버전 비교](../../../../data/pallet/results/{NAME}/annotations/LATER_DRAFT72_POINT_ERRORS.csv)

재실행: 원본 인계 ZIP을 풀어 `--handoff`로 지정하고, 완료 자료가 있는 저장소를 `--receiver-root`로 지정합니다. 게시한 `annotations/inputs`도 동일 인계 입력으로 사용 가능합니다. 모델 실행과 새로운 주석 작성은 없습니다.

```bash
python3 scripts/research/{NAME}/annotation_connection.py \\
  --handoff /tmp/pallet-remaining-evidence-handoff-20261006 \\
  --receiver-root /home/minjae/Documents/github/pallet-pose \\
  --output-root /tmp/pallet-github-publication-20261006-v1 \\
  --archive-root /home/minjae/Downloads
```
"""
    (docs / "FINDINGS_KO.md").write_text(findings, encoding="utf-8")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--receiver-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--archive-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    result = run(args.handoff.resolve(), args.receiver_root.resolve(), args.output_root.resolve(),
                 [root.resolve() for root in args.archive_root])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
