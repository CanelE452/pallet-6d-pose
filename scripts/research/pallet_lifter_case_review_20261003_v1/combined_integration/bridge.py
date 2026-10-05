"""L4 evaluator bridge for the frozen lifter experiment.

This module never runs a model and never creates human decisions.  It keeps the
original JSONL immutable, derives a hash-bound evaluator copy, and joins only a
separately submitted human object-correspondence sidecar.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile


METHODS = ("Base", "N3")
LEGACY_METHODS = {"Base": "R0", "N3": "N3_seed1"}
QUEUE_SCHEMA = "lifter_object_match_queue_v1"
SIDECAR_SCHEMA = "lifter_object_match_review_v1"


class GateError(ValueError):
    """A frozen-input or human-evidence gate failed."""


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise GateError(f"blank JSONL row at {number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GateError(f"non-object JSONL row at {number}")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path | str, value) -> None:
    _atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2,
                                      sort_keys=False, allow_nan=False) + "\n")


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False) + "\n" for row in rows)
    _atomic_text(Path(path), text)


def finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def point_valid_mask(points, *, explicit_valid=None, missing_sentinels=()) -> list[bool]:
    """Return validity without conflating off-image coordinates with missingness.

    An explicit upstream boolean mask wins.  Otherwise None/NaN are missing and
    only explicitly configured *exact pairs* are sentinels.  A finite negative
    or otherwise off-image coordinate remains a valid model prediction.
    """
    if points is None:
        if explicit_valid not in (None, [], ()):
            raise GateError("explicit mask cannot make absent points valid")
        return []
    if not isinstance(points, (list, tuple)) or len(points) < 8:
        raise GateError("prediction must provide at least eight ordered point slots")
    if explicit_valid is not None:
        if (not isinstance(explicit_valid, (list, tuple)) or len(explicit_valid) < 8
                or any(type(value) is not bool for value in explicit_valid[:8])):
            raise GateError("upstream point-valid state must be eight booleans")
    sentinel_pairs = []
    for sentinel in missing_sentinels:
        if (not isinstance(sentinel, (list, tuple)) or len(sentinel) != 2
                or not all(finite(value) for value in sentinel)):
            raise GateError("a missing sentinel must be one exact finite x/y pair")
        sentinel_pairs.append(tuple(float(value) for value in sentinel))
    result = []
    for index, point in enumerate(points[:8]):
        coordinates_valid = (isinstance(point, (list, tuple)) and len(point) == 2
                             and all(finite(value) for value in point))
        is_sentinel = coordinates_valid and tuple(float(value) for value in point) in sentinel_pairs
        if explicit_valid is None:
            result.append(bool(coordinates_valid and not is_sentinel))
        else:
            # Explicit missingness remains missing even when a finite placeholder
            # is present; explicit validity still cannot legalize None/NaN/sentinel.
            result.append(bool(explicit_valid[index] and coordinates_valid and not is_sentinel))
    return result


def _upstream_validity(row: dict, method_name: str):
    raw = row.get("raw_shared_prediction", {}).get("methods", {}).get(
        LEGACY_METHODS[method_name], {})
    for key in ("keypoints_mask", "point_valid", "points_valid"):
        if key in raw:
            return raw[key], "upstream_explicit:" + key
    # The frozen Ultralytics 8.4.60 path stores raw xy and confidence but has no
    # declared missing-coordinate threshold.  Confidence is not silently turned
    # into a validity threshold here.
    return None, "frozen_xy_null_nan_contract_no_confidence_threshold"


def adapt_row_masks(row: dict, *, missing_sentinels=()) -> dict:
    """Create the evaluator row while preserving the frozen prediction values."""
    derived = copy.deepcopy(row)
    methods = derived.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(METHODS):
        raise GateError("both frozen Base and N3 methods are required")
    selected = []
    masks = {}
    for method_name in METHODS:
        method = methods[method_name]
        explicit, source = _upstream_validity(derived, method_name)
        mask = point_valid_mask(method.get("keypoints"), explicit_valid=explicit,
                                missing_sentinels=missing_sentinels)
        previous = method.get("keypoints_mask")
        if previous not in (None, []) and (not isinstance(previous, list)
                or len(previous) != len(mask) or any(type(value) is not bool for value in previous)):
            raise GateError("existing evaluator mask has an invalid schema")
        method["keypoints_mask"] = mask
        method["keypoints_mask_source"] = source
        method["missing_sentinel_contract"] = [list(pair) for pair in missing_sentinels]
        masks[method_name] = mask
        selected.append(method.get("selected_index"))
    if selected[0] != selected[1]:
        raise GateError("Base/N3 selected different objects")
    if masks["Base"] != masks["N3"]:
        raise GateError("Base/N3 point missingness changed")
    return derived


def _validate_corner_reference(reference: dict, manifest_path: Path) -> None:
    """Use the original review validator before exposing any selected box."""
    plan_path = manifest_path.parent.parent / "LIFTER_EVALUATION_PLAN.json"
    contract_path = manifest_path.parent / "CORNER_CONTRACT.json"
    validator_path = Path(__file__).resolve().parent.parent / "review" / "serve.py"
    for path in (plan_path, contract_path, validator_path):
        if not path.is_file():
            raise GateError("missing frozen corner-reference validator input: " + str(path))
    spec = importlib.util.spec_from_file_location("lifter_l4_original_review_validator", validator_path)
    if spec is None or spec.loader is None:
        raise GateError("cannot load original review validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        context = module.Context(manifest_path, plan_path, contract_path,
                                 Path(directory) / "unwritten_store.json")
        context.validate_bundle(reference, allow_drafts=False)


def _primary_targets(reference: dict) -> dict[str, dict]:
    if (reference.get("schema_version") != "lifter_reference_review_v1"
            or reference.get("source_kind") != "human_reviewed"):
        raise GateError("only a submitted human corner reference may seed object matching")
    result = {}
    for record in reference.get("records", []):
        if record.get("review_pass") != "primary" or record.get("status") != "reviewed":
            continue
        obj = record.get("object", {})
        if (obj.get("presence") != "present" or obj.get("target_identity_confirmed") is not True
                or not isinstance(obj.get("target_object_id"), str) or not obj["target_object_id"].strip()):
            continue
        if record.get("source_kind") != "human_reviewed":
            raise GateError("target reference is not human-reviewed")
        if record["frame_id"] in result:
            raise GateError("duplicate primary target frame")
        result[record["frame_id"]] = record
    return result


def _selection(row: dict) -> dict:
    methods = row.get("methods", {})
    if set(methods) != set(METHODS):
        raise GateError("prediction row lacks both frozen methods")
    base, n3 = methods["Base"], methods["N3"]
    if base.get("selected_index") != n3.get("selected_index"):
        raise GateError("Base/N3 candidate selection differs")
    a, b = base.get("selected_object"), n3.get("selected_object")
    if a != b:
        raise GateError("Base/N3 selected-object audit differs")
    if not isinstance(a, dict):
        raise GateError("selected-object audit is absent")
    if a.get("selected_index") != base.get("selected_index"):
        raise GateError("selected-object audit index differs from method output")
    return {"selected_index": base.get("selected_index"),
            "selected_box_xyxy": a.get("selected_box_xyxy"),
            "candidate_count": a.get("candidate_count"),
            "selection_rule": a.get("selection_rule")}


def build_object_match_queue(predictions_path: Path | str, reviewed_path: Path | str,
                             manifest_path: Path | str, output_path: Path | str) -> dict:
    predictions_path, reviewed_path = Path(predictions_path), Path(reviewed_path)
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    reference = read_json(reviewed_path)
    _validate_corner_reference(reference, manifest_path)
    source_rows = read_jsonl(predictions_path)
    rows = {row["frame_id"]: row for row in source_rows}
    if len(rows) != len(source_rows):
        raise GateError("duplicate prediction frame ID")
    targets = _primary_targets(reference)
    manifest = read_json(manifest_path)
    manifest_frames = {frame["frame_id"]: frame for frame in manifest.get("frames", [])}
    records = []
    not_required = {"no_selected_prediction": 0, "no_direct_visible_reference": 0,
                    "no_predicted_points": 0}
    for frame_id, target in targets.items():
        if frame_id not in rows or frame_id not in manifest_frames:
            raise GateError("reviewed target lacks a bound prediction or image: " + frame_id)
        frame = manifest_frames[frame_id]
        selection = _selection(rows[frame_id])
        if not any(corner.get("visibility") == "direct_visible"
                   for corner in target.get("corners", [])):
            not_required["no_direct_visible_reference"] += 1
            continue
        if selection["selected_index"] is None:
            not_required["no_selected_prediction"] += 1
            continue
        if all(rows[frame_id]["methods"][method].get("keypoints") is None
               for method in METHODS):
            not_required["no_predicted_points"] += 1
            continue
        records.append({"frame_id": frame_id, "session_id": frame["session_id"],
                        "target_object_id": target["object"]["target_object_id"],
                        "image_sha256": frame["image_sha256"],
                        "image_path": frame["image_path"], **selection})
    identity_path = predictions_path.parent / "RUN_IDENTITY.json"
    if not identity_path.is_file():
        raise GateError("prediction JSONL lacks sibling RUN_IDENTITY.json")
    queue = {"schema_version": QUEUE_SCHEMA, "source_kind": "machine_prepared_human_task",
             "status": "WAITING_HUMAN",
             "bindings": {"predictions_sha256": sha256(predictions_path),
                          "prediction_run_identity_sha256": sha256(identity_path),
                          "reviewed_reference_sha256": sha256(reviewed_path),
                          "manifest_sha256": sha256(manifest_path)},
             "policy": {"show_selected_box_only": True, "show_model_corners": False,
                        "show_method_name": False, "show_error_or_improvement": False,
                        "decision_values": ["same", "different", "undetermined"]},
             "human_decision_required_count": len(records),
             "human_decision_not_required_counts": not_required,
             "records": records}
    write_json(output_path, queue)
    return queue


def validate_sidecar(queue: dict, sidecar: dict) -> dict:
    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise GateError("bad object-match queue schema")
    if (sidecar.get("schema_version") != SIDECAR_SCHEMA
            or sidecar.get("source_kind") != "human_reviewed"):
        raise GateError("object-match decisions must be explicitly human-reviewed")
    if sidecar.get("queue_sha256") is None:
        raise GateError("object-match sidecar lacks its queue hash")
    expected = {record["frame_id"]: record for record in queue.get("records", [])}
    seen = set()
    for record in sidecar.get("records", []):
        frame_id = record.get("frame_id")
        if frame_id in seen or frame_id not in expected:
            raise GateError("unknown or duplicate object-match frame")
        seen.add(frame_id)
        frozen = expected[frame_id]
        for key in ("target_object_id", "selected_index", "selected_box_xyxy"):
            if record.get(key) != frozen.get(key):
                raise GateError("object-match selection binding changed: " + key)
        if record.get("decision") not in ("same", "different", "undetermined"):
            raise GateError("invalid object-match decision")
        if record.get("source_kind") != "human_reviewed":
            raise GateError("machine object-match decision rejected")
        reviewer = record.get("reviewer", {})
        if (not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip()
                or reviewer.get("entered_by") != "human" or reviewer.get("confirmation") is not True
                or reviewer.get("machine_assistance") is not True
                or reviewer.get("previous_prediction_exposure") is not True
                or not isinstance(reviewer.get("previous_annotation_exposure"), bool)):
            raise GateError("actual human reviewer and box exposure are required")
        timing = record.get("review_time", {})
        if (timing.get("clock_source") != "server_wall_and_monotonic"
                or not finite(timing.get("duration_seconds")) or timing["duration_seconds"] < 0
                or not timing.get("started_at") or not timing.get("finished_at")):
            raise GateError("actual object-match review time is required")
        try:
            started = datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))
            finished = datetime.fromisoformat(timing["finished_at"].replace("Z", "+00:00"))
            if started.tzinfo is None or finished.tzinfo is None or finished < started:
                raise ValueError
        except (AttributeError, TypeError, ValueError) as error:
            raise GateError("timezone-aware ordered object-review timestamps required") from error
    return sidecar


def apply_sidecar(predictions_path: Path | str, queue_path: Path | str,
                  sidecar_path: Path | str, output_path: Path | str,
                  receipt_path: Path | str, *, missing_sentinels=()) -> dict:
    predictions_path, queue_path = Path(predictions_path), Path(queue_path)
    sidecar_path, output_path, receipt_path = (Path(sidecar_path), Path(output_path),
                                               Path(receipt_path))
    queue, sidecar = read_json(queue_path), read_json(sidecar_path)
    if queue.get("bindings", {}).get("predictions_sha256") != sha256(predictions_path):
        raise GateError("queue is bound to a different raw prediction file")
    if sidecar.get("queue_sha256") != sha256(queue_path):
        raise GateError("sidecar is bound to a different object-match queue")
    validate_sidecar(queue, sidecar)
    decisions = {record["frame_id"]: record for record in sidecar["records"]}
    rows = []
    applied = {"same": 0, "different": 0, "undetermined": 0}
    for source in read_jsonl(predictions_path):
        row = adapt_row_masks(source, missing_sentinels=missing_sentinels)
        decision = decisions.get(row["frame_id"])
        if decision is not None:
            selection = _selection(row)
            for key in ("selected_index", "selected_box_xyxy"):
                if decision.get(key) != selection.get(key):
                    raise GateError("sidecar no longer matches frozen selected object: " + key)
            match = {"same": True, "different": False, "undetermined": None}[decision["decision"]]
            for method_name in METHODS:
                method = row["methods"][method_name]
                method["object_id"] = None
                method["object_match"] = match
                if match is None:
                    method.pop("object_match_source_kind", None)
                else:
                    method["object_match_source_kind"] = "human_reviewed"
                    method["object_match_evidence"] = {
                        "sidecar_sha256": sha256(sidecar_path),
                        "target_object_id": decision["target_object_id"],
                        "reviewer_id": decision["reviewer"]["id"]}
            applied[decision["decision"]] += 1
        rows.append(row)
    write_jsonl(output_path, rows)
    receipt = {"schema_version": "lifter_l4_evaluator_derivation_v1",
               "source_predictions": {"path": str(predictions_path),
                                      "sha256": sha256(predictions_path), "rows": len(rows)},
               "object_match_queue_sha256": sha256(queue_path),
               "human_sidecar_sha256": sha256(sidecar_path),
               "derived_predictions": {"path": str(output_path),
                                       "sha256": sha256(output_path), "rows": len(rows)},
               "applied_human_decisions": applied,
               "raw_prediction_overwritten": False,
               "method_specific_reselection": False,
               "machine_human_review_created": False}
    write_json(receipt_path, receipt)
    return receipt


def adapt_predictions(predictions_path: Path | str, output_path: Path | str,
                      receipt_path: Path | str, *, missing_sentinels=()) -> dict:
    """Derive mask-checked predictions without requiring human correspondence."""
    predictions_path, output_path, receipt_path = (Path(predictions_path),
                                                    Path(output_path), Path(receipt_path))
    source = read_jsonl(predictions_path)
    rows = [adapt_row_masks(row, missing_sentinels=missing_sentinels)
            for row in source]
    write_jsonl(output_path, rows)
    receipt = {"schema_version": "lifter_l4_mask_derivation_v1",
               "source_predictions": {"path": str(predictions_path),
                                      "sha256": sha256(predictions_path), "rows": len(source)},
               "derived_predictions": {"path": str(output_path),
                                       "sha256": sha256(output_path), "rows": len(rows)},
               "base_n3_mask_identity_checked": True,
               "missing_sentinels": [list(pair) for pair in missing_sentinels],
               "confidence_threshold_invented": False,
               "raw_prediction_overwritten": False}
    write_json(receipt_path, receipt)
    return receipt


def separated_status(*, preflight=None, inference=None, reference=None,
                     queue=None, sidecar=None, metrics=None) -> dict:
    pre = read_json(preflight) if preflight and Path(preflight).is_file() else {}
    inf = read_json(inference) if inference and Path(inference).is_file() else {}
    ref = read_json(reference) if reference and Path(reference).is_file() else {}
    task = read_json(queue) if queue and Path(queue).is_file() else {}
    side = read_json(sidecar) if sidecar and Path(sidecar).is_file() else {}
    met = read_json(metrics) if metrics and Path(metrics).is_file() else {}
    submitted = [r for r in ref.get("records", []) if r.get("review_pass") == "primary"
                 and r.get("status") in ("reviewed", "skipped")]
    match_records = side.get("records", []) if side.get("source_kind") == "human_reviewed" else []
    required_matches = (len(task.get("records", []))
                        if task.get("schema_version") == QUEUE_SCHEMA else None)
    correspondence_status = ("NOT_READY_CORNER_REFERENCE" if required_matches is None
                             else "VERIFIED_COMPLETE" if len(match_records) == required_matches
                             else "WAITING_HUMAN")
    return {"schema_version": "lifter_combined_partial_status_v1",
            "model_bindings": {"status": pre.get("status", "BLOCKED_CONTRACT"),
                               "missing": pre.get("missing_bindings", [])},
            "full_inference": {"status": inf.get("status", "NOT_STARTED"),
                               "inferred_frames": inf.get("inferred_frames", 0),
                               "expected_frames": 8910},
            "corner_reference": {"status": "VERIFIED_COMPLETE" if len(submitted) == 120 else "WAITING_HUMAN",
                                 "submitted_primary_frames": len(submitted), "expected_primary_frames": 120},
            "object_correspondence": {"status": correspondence_status,
                                      "reviewed_frames": len(match_records),
                                      "required_selected_object_frames": required_matches},
            "stop_intervals": met.get("reviewed_stop_variation", {"status": "WAITING_HUMAN"}),
            "independent_physical_reference": met.get("independent_physical_accuracy",
                {"status": "BLOCKED_REFERENCE"}),
            "metrics_statuses": met.get("statuses", {}),
            "note": "Completed inference is never relabelled BLOCKED_CONTRACT solely because human/reference branches wait."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    queue = sub.add_parser("prepare-object-review")
    queue.add_argument("--predictions", type=Path, required=True)
    queue.add_argument("--reviewed", type=Path, required=True)
    queue.add_argument("--manifest", type=Path, required=True)
    queue.add_argument("--output", type=Path, required=True)
    masks = sub.add_parser("adapt-masks")
    masks.add_argument("--predictions", type=Path, required=True)
    masks.add_argument("--output", type=Path, required=True)
    masks.add_argument("--receipt", type=Path, required=True)
    apply = sub.add_parser("apply-object-review")
    apply.add_argument("--predictions", type=Path, required=True)
    apply.add_argument("--queue", type=Path, required=True)
    apply.add_argument("--sidecar", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--receipt", type=Path, required=True)
    status = sub.add_parser("status")
    for name in ("preflight", "inference", "reference", "queue", "sidecar", "metrics"):
        status.add_argument("--" + name, type=Path)
    status.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-object-review":
            value = build_object_match_queue(args.predictions, args.reviewed,
                                             args.manifest, args.output)
        elif args.command == "adapt-masks":
            value = adapt_predictions(args.predictions, args.output, args.receipt)
        elif args.command == "apply-object-review":
            value = apply_sidecar(args.predictions, args.queue, args.sidecar,
                                  args.output, args.receipt)
        else:
            value = separated_status(preflight=args.preflight, inference=args.inference,
                                     reference=args.reference, queue=args.queue,
                                     sidecar=args.sidecar,
                                     metrics=args.metrics)
            write_json(args.output, value)
        print(json.dumps({"status": value.get("status", "VERIFIED_COMPLETE"),
                          "output": str(args.output)}, ensure_ascii=False))
        return 0
    except (GateError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_CONTRACT", "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
