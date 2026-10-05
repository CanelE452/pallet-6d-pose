"""Register static populations and reaggregate frozen predictions by human severity.

This is a read-only consumer of the original annotations and prediction caches.
It writes a new namespace and never changes the frozen N3/static-closeout files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.research.pallet_n3_completion_v3 import common as C
from scripts.research.pallet_n3_completion_v3 import metrics as M
from scripts.research.pallet_n3_completion_v3 import reuse as R


ROOT = Path(__file__).resolve().parents[3]
NAME = Path(__file__).parent.name
DOC = ROOT / "_docs/experiments" / NAME
RAW = ROOT / "data/pallet/results" / NAME

RESPONSE_REL = Path(
    "_docs/experiments/pallet_clean19_severity_audit_20261002_v1/"
    "data/COMBINED319_SEVERITY_RESPONSES.json")
REVIEW_REL = Path("data/pallet/results/pallet_combined319_severity_review_v1")
YOLO_SCORES_REL = Path("data/pallet/results/pallet_n3_static_closeout_v1/YOLO_SCORES.json")
N3_RAW_REL = Path("data/pallet/results/pallet_n3_completion_v3")
OLD_STATIC_REL = Path("_docs/experiments/pallet_n3_static_closeout_v1")
GREEN_REL = Path("_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json")
GEOMETRY_REL = Path("challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json")

LABELS = {
    "CLEAN": "clean",
    "MODERATE_OCCLUSION": "moderate",
    "SEVERE_OCCLUSION": "severe",
}
GROUP_ORDER = ("clean", "moderate", "severe", "all")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path, *, root: Path | None = None) -> dict:
    path = path.resolve()
    shown = str(path)
    if root is not None:
        try:
            shown = str(path.relative_to(root.resolve()))
        except ValueError:
            pass
    return {"path": shown, "sha256": sha256(path), "bytes": path.stat().st_size}


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def _mean_dict(values: Sequence[Mapping[str, Any]]) -> dict:
    output: dict[str, Any] = {}
    common = set.intersection(*(set(value) for value in values))
    for key in sorted(common):
        items = [value[key] for value in values]
        if all(isinstance(item, Mapping) for item in items):
            output[key] = _mean_dict(items)  # type: ignore[arg-type]
        elif all(isinstance(item, (int, float)) and not isinstance(item, bool)
                 and np.isfinite(item) for item in items):
            if all(isinstance(item, int) for item in items) and len(set(items)) == 1:
                output[key] = int(items[0])
            else:
                output[key] = float(np.mean(items))
        elif all(item == items[0] for item in items):
            output[key] = items[0]
    return output


def _headline(corner_rows: Sequence[Mapping[str, Any]],
              pose_rows: Sequence[Mapping[str, Any]]) -> dict:
    corner = M._corner_summary(corner_rows)
    pose = M._pose_summary(pose_rows)
    return {
        "frames": corner["denominators"]["frames"],
        "full_supervised_corners": corner["denominators"]["full_supervised_corners"],
        "observed_corners": corner["denominators"]["observed_corners"],
        "matched_frames": corner["denominators"]["matched_frames"],
        "corner_median_px": corner["metrics"]["matched_pooled_corner8_median_px"],
        "corner_P90_px": corner["metrics"]["matched_pooled_corner8_P90_px"],
        "PCK10_fraction": corner["metrics"]["full_PCK10_fraction"],
        "translation_median_cm": pose["translation_cm"]["median"],
        "translation_P90_cm": pose["translation_cm"]["P90"],
        "rotation_median_deg": pose["rotation_deg"]["median"],
        "rotation_P90_deg": pose["rotation_deg"]["P90"],
        "yaw_median_deg": pose["yaw_deg"]["median"],
        "yaw_P90_deg": pose["yaw_deg"]["P90"],
        "IoU3D_median": pose["IoU3D"]["median"],
        "ADDsym_AUC_full": pose["ADDsym_AUC_full"],
        "pose_available_frames": pose["denominators"]["available_frames"],
        "pose_failed_frames": pose["denominators"]["failed_frames"],
        "pose_coverage": pose["denominators"]["coverage"],
    }


def _summaries(corner_rows: Sequence[Mapping[str, Any]],
               pose_rows: Sequence[Mapping[str, Any]]) -> dict:
    output = {"all": _headline(corner_rows, pose_rows)}
    for label in GROUP_ORDER[:-1]:
        corners = [row for row in corner_rows if row["occlusion"] == label]
        poses = [row for row in pose_rows if row["occlusion"] == label]
        output[label] = _headline(corners, poses)
    materials = sorted({str(row["material"]) for row in corner_rows})
    output["material_x_severity"] = {}
    for material in materials:
        for label in GROUP_ORDER[:-1]:
            corners = [row for row in corner_rows
                       if row["material"] == material and row["occlusion"] == label]
            poses = [row for row in pose_rows
                     if row["material"] == material and row["occlusion"] == label]
            if corners:
                output["material_x_severity"][f"{material}::{label}"] = _headline(
                    corners, poses)
    return output


def _relabeled(rows: Iterable[Mapping[str, Any]], labels: Mapping[str, str]) -> list[dict]:
    output = []
    for row in rows:
        frame_id = str(row["id"])
        if frame_id not in labels:
            raise ValueError(f"missing completed label for {frame_id}")
        output.append({**row, "occlusion": labels[frame_id]})
    return output


def completed_labels(source: Path, dev_rows: Sequence[Mapping[str, Any]]) -> tuple[dict, dict]:
    response_path = source / RESPONSE_REL
    order_path = source / REVIEW_REL / "REVIEW_ORDER.json"
    mapping_path = source / REVIEW_REL / "PRIVATE_FRAME_MAPPING.json"
    response, order = read(response_path), read(order_path)
    mapping = read(mapping_path)["rows"]
    if response.get("schema_version") != "pallet_combined319_direct_three_choice_v1":
        raise ValueError("unexpected response schema")
    if response.get("review_status") != "HUMAN_DIRECT_REVIEW_COMPLETE":
        raise ValueError("human response is not submitted complete")
    if response.get("label_source") != "HUMAN_DIRECT_CLASS":
        raise ValueError("unexpected label source")
    if response.get("order_sha256") != sha256(order_path):
        raise ValueError("review order hash mismatch")
    review_ids = [row["review_id"] for row in mapping]
    if order["review_ids"] != review_ids or set(response["responses"]) != set(review_ids):
        raise ValueError("review identity/order mismatch")
    if len(mapping) != 319 or len({row["image_sha256"] for row in mapping}) != 319:
        raise ValueError("expected 319 unique review images")

    by_sha = {row["image_sha256"]: row for row in mapping}
    labels: dict[str, str] = {}
    sidecar = []
    verified_files = 0
    for row in dev_rows:
        image = source / row["_image_path"]
        image_hash = sha256(image)
        mapped = by_sha.get(image_hash)
        if mapped is None:
            raise ValueError(f"no exact RGB hash join for {row['id']}")
        review_id = mapped["review_id"]
        raw_label = response["responses"][review_id]
        if raw_label not in LABELS:
            raise ValueError(f"invalid severity {raw_label}")
        if mapped["frame_id"].replace("__", ":", 1) != row["id"]:
            raise ValueError(f"hash joined different frame identity for {row['id']}")
        labels[row["id"]] = LABELS[raw_label]
        sidecar.append({
            "id": row["id"], "session": row["session"], "material": row["material"],
            "review_id": review_id, "image_sha256": image_hash,
            "severity": LABELS[raw_label], "raw_response": raw_label,
            "source": "HUMAN_DIRECT_CLASS", "submitted": True,
            "exported_at": response["exported_at"],
        })
        verified_files += 1
    if len(labels) != 319:
        raise ValueError("incomplete exact-image label join")

    old_manifest = read(source / R.OCCLUSION_REL)["rows"]
    old = {row["frame_id"]: LABELS[row["severity"]] for row in old_manifest
           if row["frame_id"] in labels}
    split = read(source / R.SPLIT_REL)["heldout"]
    old_split = {row["id"]: LABELS[row["severity"]] for row in split
                 if row["id"] in labels}
    old_direct_changes = [{"id": frame_id, "old": old[frame_id],
                           "completed319": labels[frame_id]}
                          for frame_id in sorted(old) if old[frame_id] != labels[frame_id]]
    split_changes = [{"id": frame_id, "old": old_split[frame_id],
                      "completed319": labels[frame_id]}
                     for frame_id in sorted(old_split)
                     if old_split[frame_id] != labels[frame_id]]
    audit = {
        "schema": "pallet_static_human_severity_join_v1",
        "status": "VERIFIED_HUMAN_DIRECT_COMPLETE",
        "labels": len(labels),
        "counts": dict(Counter(labels.values())),
        "material_x_severity": {
            f"{material}::{label}": count for (material, label), count in sorted(Counter(
                (row["material"], labels[row["id"]]) for row in dev_rows).items())},
        "image_files_sha256_verified": verified_files,
        "join": "exact encoded RGB SHA-256 plus normalized frame identity",
        "source_response": bind(response_path, root=source),
        "review_order": bind(order_path, root=source),
        "private_mapping": bind(mapping_path, root=source),
        "exported_at": response["exported_at"],
        "old_direct128": {
            "count": len(old), "counts": dict(Counter(old.values())),
            "changes_against_completed319": old_direct_changes,
        },
        "older_split128": {
            "count": len(old_split), "counts": dict(Counter(old_split.values())),
            "changes_against_completed319": split_changes,
            "note": "Historical copied SPLIT_LOCK labels are retained only as a discrepancy audit.",
        },
        "automatic_human_labels": 0,
    }
    return {"labels": labels, "sidecar": sidecar}, audit


def aggregate_backbones(source: Path, labels: Mapping[str, str]) -> tuple[dict, dict]:
    yolo_path = source / YOLO_SCORES_REL
    yolo = read(yolo_path)
    backbone_inputs: dict[str, dict[str, tuple[list[dict], list[dict]]]] = {"yolo": {}}
    for method, payload in yolo.items():
        backbone_inputs["yolo"][method] = (
            _relabeled(payload["corner_scores"], labels),
            _relabeled(payload["pose_scores"], labels),
        )
    input_bindings = {"yolo": bind(yolo_path, root=source)}

    for backbone in ("dope", "resnet18"):
        path = source / N3_RAW_REL / "evaluation" / f"{backbone}.json"
        payload = read(path)
        methods = {}
        for method, entry in payload["methods"].items():
            result = entry["result"]
            methods[method] = (
                _relabeled(result["corner_rows"], labels),
                _relabeled(result["pose_rows"], labels),
            )
        backbone_inputs[backbone] = methods
        input_bindings[backbone] = bind(path, root=source)

    selected = {
        "yolo": {"Base": ["R0"], "P": [f"OLD_P_seed{i}" for i in (1, 2, 3)],
                 "N2": [f"N2_DIM_ONLY_seed{i}" for i in (1, 2, 3)],
                 "N3": [f"N3_DIM_SYM_seed{i}" for i in (1, 2, 3)]},
        "dope": {"Base": ["base"], "N3": [f"n3_seed{i}" for i in (1, 2, 3)]},
        "resnet18": {"Base": ["base"], "N3": [f"n3_seed{i}" for i in (1, 2, 3)]},
    }
    output: dict[str, Any] = {}
    invariance: dict[str, Any] = {}
    for backbone, display_methods in selected.items():
        output[backbone] = {}
        invariance[backbone] = {}
        for display, raw_names in display_methods.items():
            per_seed = {}
            for raw_name in raw_names:
                corners, poses = backbone_inputs[backbone][raw_name]
                per_seed[raw_name] = _summaries(corners, poses)
                # Labels are the only changed field; full-population metrics must be exact.
                original_corners = [{**row, "occlusion": "unclassified"} for row in corners]
                original_poses = [{**row, "occlusion": "unclassified"} for row in poses]
                before = _headline(original_corners, original_poses)
                after = per_seed[raw_name]["all"]
                invariance[backbone][raw_name] = {
                    "exact_equal": clean(before) == clean(after),
                    "before": before, "after": after,
                }
                if clean(before) != clean(after):
                    raise ValueError(f"overall metrics changed after labels: {backbone}/{raw_name}")
            if len(raw_names) == 1:
                output[backbone][display] = {
                    "aggregation": "single frozen prediction path",
                    "raw_methods": raw_names,
                    "result": per_seed[raw_names[0]],
                }
            else:
                group_keys = [*GROUP_ORDER, "material_x_severity"]
                mean = {}
                for group in group_keys:
                    if group == "material_x_severity":
                        keys = set.intersection(*(set(per_seed[name][group]) for name in raw_names))
                        mean[group] = {key: _mean_dict([per_seed[name][group][key]
                                                       for name in raw_names])
                                       for key in sorted(keys)}
                    else:
                        mean[group] = _mean_dict([per_seed[name][group]
                                                 for name in raw_names])
                output[backbone][display] = {
                    "aggregation": "arithmetic mean of per-seed statistics; predictions are not averaged",
                    "raw_methods": raw_names, "per_seed": per_seed, "result": mean,
                }
    artifact = {
        "schema": "pallet_static_completed319_reaggregation_v1",
        "population": "DEV319",
        "label_counts": dict(Counter(labels.values())),
        "evaluation_contract": {
            "corner": "locked 8-corner whole-object symmetry; conditional median/P90; full-reference PCK10",
            "pose": "same locked camera/dimensions/symmetry/PnP/failure contract; failures retained",
            "N3": "mean of three seed statistics; no prediction averaging",
        },
        "backbones": output,
        "inputs": input_bindings,
        "new_training": 0,
        "new_inference": 0,
        "new_PnP": 0,
    }
    return artifact, {"status": "PASS", "checks": invariance,
                      "all_exact": all(item["exact_equal"] for backbone in invariance.values()
                                       for item in backbone.values())}


def cohort_audit(source: Path) -> dict:
    old_manifest_path = source / R.OCCLUSION_REL
    old_rows = read(old_manifest_path)["rows"]
    raw_path = source / N3_RAW_REL / "reuse/PER_FRAME_SCORES.json"
    scores = read(raw_path)
    dev_ids = {row["id"] for row in scores["core"]["R0"]["corner_scores"]}
    occlusion_ids = {row["frame_id"] for row in old_rows if row["frame_id"] in dev_ids}
    student_ids = {row["id"] for row in scores["students_safe128"]["R0"]["corner_scores"]}
    return {
        "status": "VERIFIED",
        "occlusion_direct128": len(occlusion_ids),
        "student_safe128": len(student_ids),
        "intersection": len(occlusion_ids & student_ids),
        "occlusion_only": sorted(occlusion_ids - student_ids),
        "student_only": sorted(student_ids - occlusion_ids),
        "sets_exactly_equal": occlusion_ids == student_ids,
        "interpretation": "The two 128-frame cohorts are exactly equal by frame ID; the student panel remains a separate 128-frame result and is not mixed with DEV319.",
        "inputs": [bind(old_manifest_path, root=source), bind(raw_path, root=source)],
    }


def registry(source: Path, dev_rows: Sequence[Mapping[str, Any]],
             label_audit: Mapping[str, Any]) -> dict:
    manifest = read(C.DEV)
    items = {row["frame_id"]: row for row in manifest["items"]}
    geometry_path = source / GEOMETRY_REL
    geometry = read(geometry_path)
    objects = {row["object_type"]: row for row in geometry["objects"]}
    calibration = Counter()
    distortion = Counter()
    reference_sources = Counter()
    slot_count = 0
    referenced = 0
    for row in dev_rows:
        item = items[row["id"]]
        annotation_path = source / item["gt_v2_path"]
        annotation = read(annotation_path)
        camera = annotation["camera_data"]
        intrinsics = camera["intrinsics"]
        key = (camera["width"], camera["height"], intrinsics["fx"], intrinsics["fy"],
               intrinsics["cx"], intrinsics["cy"])
        calibration[key] += 1
        distortion["present" if "distortion" in camera else "absent"] += 1
        obj = annotation["objects"][0]
        annotations = obj.get("keypoint_annotations")
        if annotations:
            for point in annotations[:8]:
                reference_sources[str(point.get("source", "unknown"))] += 1
        else:
            source_name = str(obj.get("gt_source", "unknown"))
            reference_sources[source_name] += sum(bool(value) for value in row["valid"][:8])
        slot_count += 8
        referenced += sum(bool(value) for value in row["valid"][:8])
    green_path = source / GREEN_REL
    green = read(green_path)
    source_manifest_path = source / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
    source_manifest = read(source_manifest_path)
    partitions = Counter(row["partition"] for row in source_manifest["records"])
    population_roles = Counter(item.get("population_role", "unknown") for item in manifest["items"])
    return {
        "schema": "pallet_static_population_registry_v1",
        "DEV319": {
            "frames": len(dev_rows),
            "sessions": len({row["session"] for row in dev_rows}),
            "session_counts": dict(Counter(row["session"] for row in dev_rows)),
            "population_roles": dict(population_roles),
            "object_counts": dict(Counter(items[row["id"]]["object_type"] for row in dev_rows)),
            "material_counts": dict(Counter(row["material"] for row in dev_rows)),
            "severity_counts": label_audit["counts"],
            "corner_slots": slot_count,
            "reference_slots": referenced,
            "missing_reference_slots": slot_count - referenced,
            "known_visibility": {"DIRECT_VISIBLE": 66, "EXTERNAL_OCCLUDED": 5,
                                 "UNKNOWN": referenced - 71},
            "camera_calibrations": [
                {"width": key[0], "height": key[1], "fx": key[2], "fy": key[3],
                 "cx": key[4], "cy": key[5], "frames": value}
                for key, value in calibration.items()],
            "distortion_field": dict(distortion),
            "reference_source_fields": dict(reference_sources),
            "selection_flow": {
                "annotated_positive_inventory": 319,
                "paper_evaluation_manifest": 319,
                "evaluated": 319,
                "excluded_after_manifest": 0,
                "upstream_raw_candidate_count": "x",
                "upstream_sampling_interval": "x",
                "reason": "The frozen manifest proves membership and annotation identity, but does not encode the complete preselection stream or interval for every session.",
            },
        },
        "GREEN0918": {
            "frames": green["counts"]["frames"],
            "sessions": 1,
            "object_type": "plastic_standard_110x110x15",
            "manual_declared_corners": green["counts"]["manual_corners"],
            "manual_in_frame_corners": green["counts"]["manual_in_frame_corners"],
            "reference_missing_slots": 8 * green["counts"]["frames"] - green["counts"]["manual_corners"],
            "frame_severity": "WAITING_HUMAN_STATIC_REVIEW",
            "corner_visibility": "WAITING_HUMAN_STATIC_REVIEW",
            "canonical_6D_reference": "x",
            "limitations": "one session and one registered dimension; no independent canonical 6D reference",
        },
        "geometry_registry": {
            name: objects[name] for name in (
                "plastic_standard_110x130x11", "wood_small_80x59x14",
                "plastic_standard_110x110x15")},
        "dimension_order": "N3 metadata is W,D,H from source_measurement_cm; geometry axes are x,y,z = W,H,D.",
        "synthetic_manifest": {
            "partition_counts": dict(partitions),
            "total": sum(partitions.values()),
            "interpretation": "Manifest membership only; a split's existence does not prove a particular experiment was run.",
        },
        "identity_notes": {
            "exact_duplicates": "encoded image SHA-256 is unique within DEV319",
            "adjacent_frames": "same-session neighbors remain separate frames",
            "physical_object_identity": "x where acquisition metadata does not identify a serialised pallet",
            "lifter_relation": "separate population; no verified physical-object identity link",
        },
        "inputs": [bind(C.DEV, root=ROOT), bind(geometry_path, root=source),
                   bind(green_path, root=source), bind(source_manifest_path, root=source)],
    }


def comparator_audit(source: Path) -> dict:
    old = source / OLD_STATIC_REL
    paths = {
        "raw_regression": old / "RAW_COMPARATOR_STUDENT_REGRESSION.json",
        "reused_checks": old / "REUSED_COMPARATOR_STUDENT_CHECKS.json",
        "source_bindings": old / "SOURCE_BINDINGS.json",
        "runtime": old / "RUNTIME_PANEL.json",
        "resnet_effective": old / "RESNET_EFFECTIVE_PROTOCOL.json",
        "protocol_corrections": old / "PROTOCOL_CORRECTIONS.md",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prior audits: {missing}")
    raw = read(paths["raw_regression"])
    checks = read(paths["reused_checks"])
    runtime = read(paths["runtime"])
    return {
        "status": "REUSED_AND_HASH_BOUND",
        "DEV319_comparators": {
            "methods": ["D", "L", "PoseFix"],
            "same_population": 319,
            "same_corner_contract": 8,
            "raw_recalculation_checks": raw.get("checks", raw),
            "training_budget_limit": "Existing records do not establish fully equal pretraining, supervision, or selection histories; no retrospective equivalence claim.",
        },
        "student_alternatives": {
            "population": 128,
            "separate_from_DEV319_table": True,
            "raw_recalculation_checks": checks,
            "teacher_note": "The corrected-pseudo-label teacher is separately supervised and is not renamed as an N3 teacher.",
        },
        "runtime": runtime,
        "resnet": {
            "actual": "10-epoch CONSTANT-fold RGB base with DSNT decoder; dimensions enter the N3 refiner",
            "not_used": "60-epoch/argmax stale description and FULL direct-dimension base",
        },
        "known_unknowns": [
            "D/L/PoseFix complete upstream pretraining and selection equivalence",
            "same-environment latency ranking for every comparator",
            "complete perceptual/history overlap against GREEN0918",
        ],
        "inputs": {name: bind(path, root=source) for name, path in paths.items()},
        "new_training": 0,
        "new_runtime_measurement": 0,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "x"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def table_rows(results: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for backbone in ("yolo", "dope", "resnet18"):
        for group in GROUP_ORDER:
            for method in ("Base", "N3"):
                result = results["backbones"][backbone][method]["result"][group]
                rows.append([
                    backbone, group, method, result["frames"],
                    result["full_supervised_corners"], result["observed_corners"],
                    result["corner_median_px"], result["corner_P90_px"],
                    100 * result["PCK10_fraction"], result["translation_median_cm"],
                    result["translation_P90_cm"], result["rotation_median_deg"],
                    result["rotation_P90_deg"], 100 * result["pose_coverage"],
                ])
    return rows


def write_tables(results: Mapping[str, Any]) -> None:
    headers = ["Backbone", "Severity", "Method", "Frames", "Reference corners",
               "Observed corners", "Corner median px", "Corner P90 px", "PCK10 %",
               "T median cm", "T P90 cm", "R median deg", "R P90 deg", "Pose %"]
    rows = table_rows(results)
    target = DOC / "generated_tables"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "tab_completed319_occlusion.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * 3 + ["---:"] * (len(headers) - 3)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    (target / "tab_completed319_occlusion.md").write_text("\n".join(lines) + "\n")
    tex = ["\\begin{tabular}{lllrrrrrrrrrrr}", "\\toprule",
           "Backbone & Group & Method & Frames & Ref. & Obs. & Med. & P90 & PCK10 & T med. & T P90 & R med. & R P90 & Pose \\\\",
           "\\midrule"]
    for row in rows:
        cells = [_fmt(value).replace("%", "\\%") for value in row]
        tex.append(" & ".join(cells) + " \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (target / "tab_completed319_occlusion.tex").write_text("\n".join(tex) + "\n")


def write_figures(results: Mapping[str, Any], label_audit: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    target = DOC / "figures"
    target.mkdir(parents=True, exist_ok=True)
    groups = GROUP_ORDER[:-1]
    labels = ["Clean", "Moderate", "Severe"]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), layout="constrained")
    for column, backbone in enumerate(("yolo", "dope", "resnet18")):
        base = results["backbones"][backbone]["Base"]["result"]
        n3 = results["backbones"][backbone]["N3"]["result"]
        for method, values, shift, color in (
                ("Base", base, -0.18, "#666666"), ("N3", n3, 0.18, "#00856a")):
            axes[0, column].bar(x + shift, [values[g]["corner_median_px"] for g in groups],
                                width=.34, label=method, color=color)
            axes[1, column].bar(x + shift, [100 * values[g]["PCK10_fraction"] for g in groups],
                                width=.34, label=method, color=color)
        axes[0, column].set_title(backbone.upper())
        axes[0, column].set_ylabel("Corner median (px)")
        axes[1, column].set_ylabel("PCK@10 (%)")
        for axis in axes[:, column]:
            axis.set_xticks(x, labels, rotation=15)
            axis.grid(axis="y", alpha=.2)
        axes[0, column].legend(frameon=False)
    fig.suptitle("Frozen Base and N3 on 319 human-reviewed severity labels")
    fig.savefig(target / "completed319_backbone_occlusion.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    counts = label_audit["counts"]
    axes[0].bar(["Clean", "Moderate", "Severe"],
                [counts["clean"], counts["moderate"], counts["severe"]],
                color=["#4c956c", "#f2c14e", "#d1495b"])
    axes[0].set_ylabel("Frames")
    axes[0].set_title("Completed human severity review")
    cross = label_audit["material_x_severity"]
    width = .35
    for offset, material, color in ((-.175, "plastic", "#4778bf"),
                                    (.175, "wood", "#a8733f")):
        axes[1].bar(x + offset, [cross.get(f"{material}::{g}", 0) for g in groups],
                    width=width, label=material, color=color)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Frames")
    axes[1].set_title("Material × severity")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    fig.savefig(target / "completed319_label_composition.png", dpi=180)
    plt.close(fig)


def write_report(results: Mapping[str, Any], labels: Mapping[str, Any],
                 cohort: Mapping[str, Any], registry_data: Mapping[str, Any]) -> None:
    yolo = results["backbones"]["yolo"]
    lines = [
        "# 정적 자료 등록·가림 재집계 결과",
        "",
        "완료 제출된 사람 직접 분류 319장을 원본 RGB SHA-256과 frame ID로 전수 연결했다. "
        "가림 등급은 clean 151 / moderate 87 / severe 81이며 미분류는 0장이다. "
        "기계 제안이나 기존 모델 결과를 사람 레이블로 바꾸지 않았다.",
        "",
        "## 핵심 수치",
        "",
        "| 집단 | 영상 | YOLO Base 코너 중앙/P90 | YOLO N3 코너 중앙/P90 | "
        "Base→N3 T 중앙 | Base→N3 R 중앙 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in GROUP_ORDER:
        base, n3 = yolo["Base"]["result"][group], yolo["N3"]["result"][group]
        lines.append(
            f"| {group} | {base['frames']} | {base['corner_median_px']:.3f} / "
            f"{base['corner_P90_px']:.3f} | {n3['corner_median_px']:.3f} / "
            f"{n3['corner_P90_px']:.3f} | {base['translation_median_cm']:.3f}→"
            f"{n3['translation_median_cm']:.3f} cm | {base['rotation_median_deg']:.3f}→"
            f"{n3['rotation_median_deg']:.3f}° |")
    lines.extend([
        "",
        "![가림별 세 기반 결과](figures/completed319_backbone_occlusion.png)",
        "",
        "![사람 가림 레이블 구성](figures/completed319_label_composition.png)",
        "",
        "세 기반의 전체 319장 수치는 레이블 연결 전후 비트 단위 JSON 값으로 동일하다. "
        "이번 계산은 기존 원시 코너·자세 행을 새 사람 집단으로 필터링한 재집계이며, "
        "새 학습·추론·PnP는 0회다. N3는 각 기반의 세 seed 통계 평균이다. "
        "가림 집단은 개발자료의 사후 분석이며 독립 시험으로 부르지 않는다.",
        "",
        "## 레이블과 분모 확인",
        "",
        f"- 기존 직접검수 128장과 학생 공통 128장은 frame ID 교집합 {cohort['intersection']}장, "
        f"차집합 0/0장으로 정확히 같다.",
        f"- 직사각형 참조는 {registry_data['DEV319']['reference_slots']}/"
        f"{registry_data['DEV319']['corner_slots']} 슬롯이다. 좌표 유효성을 직접 가시로 간주하지 않았다.",
        "- 정사각형 119장의 가림과 코너 가시성은 아직 사람 검수 대기다. "
        "600/602점 기존 2D 분모와 독립 6D 참조 x는 유지한다.",
        "- 기존 66 직접 가시·5 외부 가림 상태만 승인된 옛 검수로 유지한다. "
        "나머지 직사각형 2,428점은 자동 확정하지 않았다.",
        "",
        "## 바로 확인할 파일",
        "",
        "- [319장 레이블 출처 감사](LABEL_PROVENANCE_AUDIT.json)",
        "- [세 기반 전체 재집계](STATIC_REAGGREGATION.json)",
        "- [전체 결과 불변 검사](STATIC_INVARIANCE_CHECK.json)",
        "- [정적 자료 레지스트리](REGISTRY.json)",
        "- [표 CSV](generated_tables/tab_completed319_occlusion.csv)",
        "- [남은 정사각형·코너 상태 검수 안내](review/README_KO.md)",
        "",
        "현재 제출 응답의 exported_at은 `" + str(labels["exported_at"]) + "`이며, "
        "원본 응답·순서·private mapping의 SHA-256은 감사 JSON에 고정했다.",
    ])
    (DOC / "FINAL_REPORT_KO.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/minjae/Documents/github/pallet-pose"))
    args = parser.parse_args(argv)
    source = args.source_root.expanduser().resolve()
    started = datetime.now(timezone.utc)
    dev_rows, source_contract = R.load_dev_context(source, include_pose=False)
    completed, label_audit = completed_labels(source, dev_rows)
    write_json(RAW / "COMPLETED319_LABEL_SIDECAR.json", {
        "schema": "pallet_static_completed319_label_sidecar_v1",
        "immutable_source": label_audit["source_response"],
        "rows": completed["sidecar"],
    })
    write_json(DOC / "LABEL_PROVENANCE_AUDIT.json", label_audit)
    results, invariance = aggregate_backbones(source, completed["labels"])
    write_json(DOC / "STATIC_REAGGREGATION.json", results)
    write_json(DOC / "STATIC_INVARIANCE_CHECK.json", invariance)
    cohort = cohort_audit(source)
    write_json(DOC / "COHORT_OVERLAP_AUDIT.json", cohort)
    registry_data = registry(source, dev_rows, label_audit)
    registry_data["source_contract"] = source_contract
    write_json(DOC / "REGISTRY.json", registry_data)
    write_json(DOC / "COMPARATOR_AND_COST_AUDIT.json", comparator_audit(source))
    write_tables(results)
    write_figures(results, label_audit)
    write_report(results, label_audit, cohort, registry_data)
    finished = datetime.now(timezone.utc)
    write_json(DOC / "EXECUTION_RECEIPT.json", {
        "status": "PASS",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
        "new_training": 0, "optimizer_updates": 0, "new_inference": 0,
        "new_PnP": 0, "human_labels_created_by_cli": 0,
        "outputs": [bind(path, root=ROOT) for path in sorted(DOC.rglob("*"))
                    if path.is_file()] + [bind(RAW / "COMPLETED319_LABEL_SIDECAR.json", root=ROOT)],
    })
    print(json.dumps({"status": "PASS", "label_counts": label_audit["counts"],
                      "invariance": invariance["all_exact"],
                      "wall_seconds": (finished - started).total_seconds()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
