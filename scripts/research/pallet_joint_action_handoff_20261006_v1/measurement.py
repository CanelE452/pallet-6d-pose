"""Import, validate and score independently measured physical occlusion pairs.

The newly defined v1 schema is described in measurement_packet/SCHEMA_KO.md.
Only final canonical pose outputs enter this offline evaluator; reference data
are never passed to a detector, candidate generator or prediction-only PnP.
The standard-library implementation needs no GPU or private historical data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

SCHEMA = "pallet_physical_pairs_20261006_v1"
PREDICTION_SCHEMA = "pallet_final_pose_outputs_20261006_v1"
COORDINATE_CONTRACT = "opencv_optical__canonical_object_xyz_WHD__metre"
STATES = {"success", "no_detection", "pnp_failure", "invalid_pose"}
SOURCES = {"direct-visible", "occluded-projected", "hidden-estimated", "unknown", "out-of-frame"}
REFERENCE_METHODS = {"optical_mocap", "laser_tracker", "surveyed_independent_tag_rig"}
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260917


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, length):
    return isinstance(value, list) and len(value) == length and all(finite(x) for x in value)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matvec(a, b):
    return [dot(row, b) for row in a]


def norm(v):
    return math.sqrt(dot(v, v))


def valid_rotation(value):
    if not isinstance(value, list) or len(value) != 3 or not all(vector(row, 3) for row in value):
        return False
    gram = matmul(value, transpose(value))
    if max(abs(gram[i][j] - (i == j)) for i in range(3) for j in range(3)) > 1e-6:
        return False
    a, b, c = value
    det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    return abs(det - 1) <= 1e-6


def rotation_error(a, b):
    rel = matmul(transpose(b), a)
    return math.degrees(math.acos(max(-1, min(1, (sum(rel[i][i] for i in range(3)) - 1) / 2))))


def rotations(order):
    # Same Y-axis proper group as locked pallet_dim_conditioned_p_v1/pose.py.
    return [[[math.cos(k * 2 * math.pi / order), 0, math.sin(k * 2 * math.pi / order)],
             [0, 1, 0],
             [-math.sin(k * 2 * math.pi / order), 0, math.cos(k * 2 * math.pi / order)]] for k in range(order)]


def pose_errors(prediction, reference, xyz, order):
    r, t = prediction["R_physical"], prediction["centroid_m"]
    g, gt = reference["R_physical"], reference["centroid_m"]
    corners = [[sx * xyz[0] / 2, sy * xyz[1] / 2, sz * xyz[2] / 2]
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    rotation, add = [], []
    for symmetry in rotations(order):
        target = matmul(g, symmetry)
        rotation.append(rotation_error(r, target))
        add.append(statistics.mean(norm([a + b - c - d for a, b, c, d in zip(matvec(r, x), t, matvec(target, x), gt)]) for x in corners))
    return {"translation_cm": 100 * norm([a - b for a, b in zip(t, gt)]),
            "rotation_deg": min(rotation), "ADDsym_m": min(add)}


def artifact(item, data_root, errors, label):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
        errors.append(f"{label}: artifact path required")
        return
    rel = Path(item["path"])
    if rel.is_absolute() or ".." in rel.parts:
        errors.append(f"{label}: path must be relative without parent traversal")
        return
    path = data_root / rel
    if not path.resolve().is_relative_to(data_root.resolve()):
        errors.append(f"{label}: resolved artifact is outside data_root")
        return
    expected = item.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        errors.append(f"{label}: explicit lowercase SHA-256 required")
    if not path.is_file():
        errors.append(f"{label}: missing file {rel}")
    elif sha256(path) != expected:
        errors.append(f"{label}: SHA-256 mismatch")


def stability_trace(item, data_root, pair, errors):
    """Check the independent relative-pose trace spans the complete pair interval."""
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        return
    rel = Path(item["path"])
    if rel.is_absolute() or ".." in rel.parts:
        return
    path = Path(data_root) / rel
    if not path.resolve().is_relative_to(Path(data_root).resolve()):
        return
    if not path.is_file():
        return
    pid = pair.get("pair_id")
    try:
        trace = read_json(path)
        if trace.get("schema") != "pallet_relative_pose_stability_20261006_v1" or trace.get("coordinate_contract") != COORDINATE_CONTRACT or trace.get("clock_id") != pair.get("camera", {}).get("clock_id"):
            raise ValueError("stability trace schema/coordinate/clock mismatch")
        records = trace.get("records", [])
        if len(records) < 2:
            raise ValueError("at least two independently measured stability records required")
        timestamps = [r.get("timestamp_ns") for r in records]
        if any(not isinstance(t, int) or isinstance(t, bool) for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("stability timestamps must be strictly increasing integer nanoseconds")
        endpoints = [pair.get(c, {}).get("timestamp_ns") for c in ("clean", "occluded")]
        if all(isinstance(t, int) for t in endpoints) and (timestamps[0] > min(endpoints) or timestamps[-1] < max(endpoints)):
            raise ValueError("stability trace does not cover the complete clean/occluded interval")
        tolerance = pair.get("same_pose_tolerance", {})
        gap = tolerance.get("max_trace_gap_ns")
        if not finite(gap) or gap <= 0 or max(b - a for a, b in zip(timestamps, timestamps[1:])) > gap:
            raise ValueError("stability sampling gap exceeds recorded max_trace_gap_ns")
        origin = pair.get("clean", {}).get("reference", {})
        for record in records:
            if not valid_rotation(record.get("R_physical")) or not vector(record.get("centroid_m"), 3) or not isinstance(record.get("source_record_id"), str) or not record["source_record_id"]:
                raise ValueError("invalid stability pose or missing raw source_record_id")
            if vector(origin.get("centroid_m"), 3) and finite(tolerance.get("translation_m")) and norm([a - b for a, b in zip(record["centroid_m"], origin["centroid_m"])]) > tolerance["translation_m"]:
                raise ValueError("independent relative translation drift exceeds tolerance")
            if valid_rotation(origin.get("R_physical")) and finite(tolerance.get("rotation_deg")) and rotation_error(record["R_physical"], origin["R_physical"]) > tolerance["rotation_deg"]:
                raise ValueError("independent relative rotation drift exceeds tolerance")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        errors.append(f"{pid}: {exc}")


def covariance_valid(matrix, size=6):
    # LDL-style Schur reduction checks semidefiniteness without a numpy dependency.
    if not isinstance(matrix, list) or len(matrix) != size or not all(vector(row, size) for row in matrix):
        return False
    if max(abs(matrix[i][j] - matrix[j][i]) for i in range(size) for j in range(size)) > 1e-12:
        return False
    a = copy.deepcopy(matrix)
    for i in range(size):
        if a[i][i] < -1e-12:
            return False
        if abs(a[i][i]) <= 1e-12:
            if any(abs(a[i][j]) > 1e-12 for j in range(i + 1, size)):
                return False
            continue
        for j in range(i + 1, size):
            for k in range(i + 1, size):
                a[j][k] -= a[j][i] * a[i][k] / a[i][i]
    return True


def validate_manifest(manifest, data_root, allow_fixture=False):
    """Return every contract violation; do not silently repair missing evidence."""
    data_root = Path(data_root)
    errors = []
    if manifest.get("schema") != SCHEMA:
        errors.append("unknown manifest schema")
    fixture = manifest.get("data_kind") == "format_fixture"
    if manifest.get("data_kind") not in ("physical_measurement", "format_fixture"):
        errors.append("data_kind must be physical_measurement or format_fixture")
    if fixture and not allow_fixture:
        errors.append("format_fixture requires --allow-fixture; it is never scientific evidence")
    if manifest.get("coordinate_contract") != COORDINATE_CONTRACT:
        errors.append("coordinate contract mismatch")
    pairs = manifest.get("pairs")
    models = manifest.get("models")
    if not isinstance(pairs, list) or not pairs:
        errors.append("pairs must be a nonempty fixed population")
        pairs = []
    if not isinstance(models, list) or len(models) < 2:
        errors.append("at least two locked models required")
        models = []
    model_ids = set()
    for model in models:
        name = model.get("method")
        if not isinstance(name, str) or not name or name in model_ids:
            errors.append("model method missing or duplicated")
        model_ids.add(name)
        for field in ("weights_sha256", "code_sha256", "pose_contract_sha256", "selection_rule_sha256"):
            value = model.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                errors.append(f"{name}: {field} required")
        if not isinstance(model.get("locked_at_ns"), int) or isinstance(model.get("locked_at_ns"), bool):
            errors.append(f"{name}: locked_at_ns required")
        for field in ("training_recording_ids", "training_pose_ids", "selection_recording_ids", "selection_pose_ids"):
            if not isinstance(model.get(field), list) or any(not isinstance(x, str) or not x for x in model[field]):
                errors.append(f"{name}: explicit {field} required, including empty lists")
    if manifest.get("baseline_method") not in model_ids:
        errors.append("baseline_method must name a locked model")
    groups, pair_ids, frame_ids, image_hashes, image_paths = {}, set(), set(), set(), set()
    for pair in pairs:
        pid = pair.get("pair_id", "<missing>")
        if not isinstance(pid, str) or not pid or pid in pair_ids:
            errors.append("pair_id missing or duplicated")
        pair_ids.add(pid)
        split = pair.get("split")
        if split not in ("train", "development", "independent_test"):
            errors.append(f"{pid}: invalid split")
        for field in ("recording_id", "physical_pose_id", "adjacent_interval_id", "pallet_id", "camera_id"):
            token = pair.get(field)
            if not isinstance(token, str) or not token:
                errors.append(f"{pid}: {field} required")
            if field in ("recording_id", "physical_pose_id", "adjacent_interval_id"):
                key = (field, token)
                if key in groups and groups[key] != split:
                    errors.append(f"{pid}: split leakage through {field}={token}")
                groups[key] = split
        if not vector(pair.get("dimensions_WDH_m"), 3) or any(x <= 0 for x in pair.get("dimensions_WDH_m", [])):
            errors.append(f"{pid}: positive dimensions_WDH_m required")
        artifact(pair.get("dimension_survey"), data_root, errors, f"{pid}/dimension_survey")
        if type(pair.get("symmetry_order")) is not int or pair["symmetry_order"] not in (1, 2, 4):
            errors.append(f"{pid}: symmetry_order must be 1, 2 or 4")
        artifact(pair.get("symmetry_evidence"), data_root, errors, f"{pid}/symmetry_evidence")
        if pair.get("occlusion_kind") != "physical_external":
            errors.append(f"{pid}: only physical_external occlusion qualifies")
        if not isinstance(pair.get("occluder_description"), str) or not pair["occluder_description"]:
            errors.append(f"{pid}: occluder_description required")
        if not isinstance(pair.get("used_for_selection"), bool):
            errors.append(f"{pid}: used_for_selection must be explicit")
        if type(pair.get("first_model_exposure_ns")) is not int:
            errors.append(f"{pid}: first_model_exposure_ns required")
        if split == "independent_test":
            if pair.get("used_for_selection") is not False:
                errors.append(f"{pid}: independent_test used for selection")
            for model in models:
                if isinstance(model.get("locked_at_ns"), int) and isinstance(pair.get("first_model_exposure_ns"), int) and model["locked_at_ns"] > pair["first_model_exposure_ns"]:
                    errors.append(f"{pid}: model {model['method']} locked after data exposure")
                for kind, field in (("recording", "recording_id"), ("pose", "physical_pose_id")):
                    for role in ("training", "selection"):
                        if pair.get(field) in (model.get(f"{role}_{kind}_ids") or []):
                            errors.append(f"{pid}: model {model['method']} {role} exposure leakage")
        camera = pair.get("camera", {})
        k = camera.get("K")
        if not isinstance(k, list) or len(k) != 3 or not all(vector(row, 3) for row in k) or k[0][0] <= 0 or k[1][1] <= 0 or k[2] != [0, 0, 1]:
            errors.append(f"{pid}: valid pixel camera K required")
        if not vector(camera.get("image_wh"), 2) or any(x <= 0 or int(x) != x for x in camera.get("image_wh", [])):
            errors.append(f"{pid}: integer positive image_wh required")
        if camera.get("distortion_model") not in ("opencv_brown_conrady", "none"):
            errors.append(f"{pid}: unsupported distortion_model")
        dc = camera.get("distortion_coefficients")
        if not isinstance(dc, list) or not all(finite(x) for x in dc) or (camera.get("distortion_model") == "none" and dc != []) or (camera.get("distortion_model") == "opencv_brown_conrady" and len(dc) not in (4, 5, 8, 12, 14)):
            errors.append(f"{pid}: explicit compatible distortion_coefficients required")
        for field in ("intrinsic_calibration", "time_sync_calibration", "reference_extrinsic_calibration"):
            artifact(camera.get(field), data_root, errors, f"{pid}/camera/{field}")
        if not isinstance(camera.get("clock_id"), str) or not camera["clock_id"]:
            errors.append(f"{pid}: camera clock_id required")
        tolerance = pair.get("same_pose_tolerance", {})
        for field in ("translation_m", "rotation_deg", "max_sync_error_ns", "max_trace_gap_ns"):
            if not finite(tolerance.get(field)) or tolerance[field] < 0:
                errors.append(f"{pid}: same_pose_tolerance/{field} must be nonnegative")
        artifact(tolerance.get("basis"), data_root, errors, f"{pid}/same_pose_tolerance/basis")
        artifact(pair.get("stability_record"), data_root, errors, f"{pid}/stability_record")
        uncertainty = pair.get("uncertainty", {})
        if uncertainty.get("tangent_order") != ["tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad"]:
            errors.append(f"{pid}: uncertainty tangent_order mismatch")
        for field in ("clean_covariance", "occluded_covariance"):
            if not covariance_valid(uncertainty.get(field)):
                errors.append(f"{pid}: invalid {field}")
        cross = uncertainty.get("clean_occluded_cross_covariance")
        if not isinstance(cross, list) or len(cross) != 6 or not all(vector(row, 6) for row in cross):
            errors.append(f"{pid}: explicit clean/occluded cross covariance required")
        elif covariance_valid(uncertainty.get("clean_covariance")) and covariance_valid(uncertainty.get("occluded_covariance")):
            joint = [uncertainty["clean_covariance"][i] + cross[i] for i in range(6)]
            joint += [[cross[j][i] for j in range(6)] + uncertainty["occluded_covariance"][i] for i in range(6)]
            if not covariance_valid(joint, 12):
                errors.append(f"{pid}: joint clean/occluded covariance is not positive semidefinite")
            # C_delta = C_clean + C_occ - C_cross - C_cross^T must be PSD.
            delta = [[uncertainty["clean_covariance"][i][j] + uncertainty["occluded_covariance"][i][j] - cross[i][j] - cross[j][i] for j in range(6)] for i in range(6)]
            if not covariance_valid(delta):
                errors.append(f"{pid}: difference covariance is not positive semidefinite")
        artifact(uncertainty.get("budget"), data_root, errors, f"{pid}/uncertainty/budget")
        for condition in ("clean", "occluded"):
            frame = pair.get(condition, {})
            fid = frame.get("frame_id")
            if not isinstance(fid, str) or not fid or fid in frame_ids:
                errors.append(f"{pid}/{condition}: frame_id missing or duplicated")
            frame_ids.add(fid)
            artifact(frame.get("image"), data_root, errors, f"{pid}/{condition}/image")
            image = frame.get("image", {})
            for item, seen, label in ((image.get("path"), image_paths, "image path"), (image.get("sha256"), image_hashes, "image bytes")):
                if item in seen:
                    errors.append(f"{pid}/{condition}: duplicate {label}")
                seen.add(item)
            if not isinstance(frame.get("timestamp_ns"), int) or isinstance(frame.get("timestamp_ns"), bool):
                errors.append(f"{pid}/{condition}: integer timestamp_ns required")
            if frame.get("clock_id") != camera.get("clock_id"):
                errors.append(f"{pid}/{condition}: clock_id mismatch")
            labels = frame.get("corner_sources")
            if not isinstance(labels, list) or len(labels) != 8 or any(x not in SOURCES for x in labels):
                errors.append(f"{pid}/{condition}: eight explicit corner_sources required")
            artifact(frame.get("label_review"), data_root, errors, f"{pid}/{condition}/label_review")
            ref = frame.get("reference", {})
            if ref.get("method") not in REFERENCE_METHODS or ref.get("independent_of_model_and_2d_labels") is not True or ref.get("derived_from") != []:
                errors.append(f"{pid}/{condition}: independently measured reference provenance required")
            if ref.get("coordinate_contract") != COORDINATE_CONTRACT or ref.get("clock_id") != camera.get("clock_id"):
                errors.append(f"{pid}/{condition}: reference clock/coordinate mismatch")
            if not valid_rotation(ref.get("R_physical")) or not vector(ref.get("centroid_m"), 3) or ref.get("centroid_m", [0, 0, -1])[2] <= 0:
                errors.append(f"{pid}/{condition}: finite proper reference pose in front of camera required")
            artifact(ref.get("source"), data_root, errors, f"{pid}/{condition}/reference/source")
            if not isinstance(ref.get("source_record_id"), str) or not ref["source_record_id"]:
                errors.append(f"{pid}/{condition}: raw source_record_id required")
            rt, ft = ref.get("timestamp_ns"), frame.get("timestamp_ns")
            if not isinstance(rt, int) or isinstance(rt, bool):
                errors.append(f"{pid}/{condition}: reference timestamp_ns required")
            elif isinstance(ft, int) and finite(tolerance.get("max_sync_error_ns")) and abs(rt - ft) > tolerance["max_sync_error_ns"]:
                errors.append(f"{pid}/{condition}: reference synchronization tolerance exceeded")
        clean, occ = pair.get("clean", {}).get("reference", {}), pair.get("occluded", {}).get("reference", {})
        if valid_rotation(clean.get("R_physical")) and valid_rotation(occ.get("R_physical")) and vector(clean.get("centroid_m"), 3) and vector(occ.get("centroid_m"), 3):
            drift_t = norm([a - b for a, b in zip(clean["centroid_m"], occ["centroid_m"])])
            drift_r = rotation_error(clean["R_physical"], occ["R_physical"])
            if finite(tolerance.get("translation_m")) and drift_t > tolerance["translation_m"]:
                errors.append(f"{pid}: same relative translation not preserved")
            if finite(tolerance.get("rotation_deg")) and drift_r > tolerance["rotation_deg"]:
                errors.append(f"{pid}: same relative rotation not preserved")
        stability_trace(pair.get("stability_record"), data_root, pair, errors)
    return errors


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def summary(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": quantile(values, .5), "P90": quantile(values, .9)}


def evaluate(manifest, predictions, split="independent_test", resamples=BOOTSTRAP_RESAMPLES):
    methods = [m["method"] for m in manifest["models"]]
    locks = {m["method"]: m for m in manifest["models"]}
    all_frames = {p[c]["frame_id"]: p[c] for p in manifest["pairs"] for c in ("clean", "occluded")}
    selected = [p for p in manifest["pairs"] if p["split"] == split]
    if not selected:
        raise ValueError(f"no fixed pairs in requested split {split}")
    by_key = {}
    for row in predictions:
        if row.get("schema") != PREDICTION_SCHEMA or row.get("coordinate_contract") != COORDINATE_CONTRACT:
            raise ValueError("prediction schema/coordinate contract mismatch")
        key = (row.get("method"), row.get("frame_id"))
        if key[0] not in methods or key[1] not in all_frames or key in by_key:
            raise ValueError(f"unknown/duplicate prediction key {key}")
        if row.get("image_sha256") != all_frames[key[1]]["image"]["sha256"]:
            raise ValueError(f"prediction image identity mismatch {key}")
        for field in ("weights_sha256", "code_sha256", "pose_contract_sha256", "selection_rule_sha256"):
            if row.get(field) != locks[key[0]][field]:
                raise ValueError(f"prediction lock mismatch {key}/{field}")
        if row.get("reference_access") is not False:
            raise ValueError(f"prediction reference_access must be false {key}")
        if row.get("state") not in STATES:
            raise ValueError(f"unknown prediction state {key}")
        if row["state"] == "success" and (not valid_rotation(row.get("R_physical")) or not vector(row.get("centroid_m"), 3) or row.get("centroid_m", [0, 0, -1])[2] <= 0):
            raise ValueError(f"malformed successful pose {key}")
        by_key[key] = row
    rows, errors, states = [], {}, {}
    for pair in selected:
        dims = pair["dimensions_WDH_m"]
        xyz = [dims[0], dims[2], dims[1]]
        for method in methods:
            state = {}
            for condition in ("clean", "occluded"):
                frame = pair[condition]
                pred = by_key.get((method, frame["frame_id"]))
                s = pred["state"] if pred else "missing_prediction"
                state[condition] = s
                value = pose_errors(pred, frame["reference"], xyz, pair["symmetry_order"]) if s == "success" else None
                errors[(pair["pair_id"], method, condition)] = value
                rows.append({"pair_id": pair["pair_id"], "recording_id": pair["recording_id"],
                             "method": method, "condition": condition, "frame_id": frame["frame_id"],
                             "state": s, **(value or {"translation_cm": None, "rotation_deg": None, "ADDsym_m": None})})
            states[(pair["pair_id"], method)] = state
    common = [p for p in selected if all(states[(p["pair_id"], m)][c] == "success" for m in methods for c in ("clean", "occluded"))]
    result = {"schema": "pallet_physical_pair_results_20261006_v1", "scientific_evidence": manifest["data_kind"] == "physical_measurement",
              "evidence_role": "independent_confirmation" if split == "independent_test" else "development_diagnostic",
              "split": split, "total_pairs": len(selected), "methods": {},
              "common_success_pair_ids": [p["pair_id"] for p in common], "common_success_pairs": len(common),
              "finite_delta_scope": "auxiliary common intersection: all locked methods and both conditions succeed",
              "reference_uncertainty": {"status": "recorded_not_propagated_to_metric_CI", "budgets": [p["uncertainty"]["budget"] for p in selected],
                                         "note": "session bootstrap describes sampling variation, not measurement uncertainty; no covariance components assumed independent"}}
    if not result["scientific_evidence"]:
        result["evidence_role"] = "format_fixture_not_scientific"
    delta_rows = []
    metric_fields = ("translation_cm", "rotation_deg", "ADDsym_m")
    for method in methods:
        counts = {"both_success": 0, "clean_only_failure": 0, "occluded_only_failure": 0, "both_failure": 0}
        failure_states = {c: {} for c in ("clean", "occluded")}
        for pair in selected:
            s = states[(pair["pair_id"], method)]
            cs, os = s["clean"] == "success", s["occluded"] == "success"
            counts["both_success" if cs and os else "clean_only_failure" if not cs and os else "occluded_only_failure" if cs and not os else "both_failure"] += 1
            for c in ("clean", "occluded"):
                failure_states[c][s[c]] = failure_states[c].get(s[c], 0) + 1
        values = {}
        for field in metric_fields:
            clean_values = [errors[(p["pair_id"], method, "clean")][field] for p in common]
            occ_values = [errors[(p["pair_id"], method, "occluded")][field] for p in common]
            deltas = [o - c for o, c in zip(occ_values, clean_values)]
            values[field] = {"clean": summary(clean_values), "occluded": summary(occ_values), "paired_occluded_minus_clean": summary(deltas),
                             "occlusion_worsened_pairs": sum(x > 1e-9 for x in deltas), "occlusion_improved_pairs": sum(x < -1e-9 for x in deltas)}
        result["methods"][method] = {"fixed_pair_failure_quadrants": counts, "frame_states": failure_states,
                                     "clean_coverage": (counts["both_success"] + counts["occluded_only_failure"]) / len(selected),
                                     "occluded_coverage": (counts["both_success"] + counts["clean_only_failure"]) / len(selected),
                                     "both_coverage": counts["both_success"] / len(selected), "common_pair_metrics": values}
        for p in common:
            delta_rows.append({"pair_id": p["pair_id"], "recording_id": p["recording_id"], "method": method,
                               **{field: errors[(p["pair_id"], method, "occluded")][field] - errors[(p["pair_id"], method, "clean")][field] for field in metric_fields}})
    baseline = manifest["baseline_method"]
    comparisons = {}
    for method in methods:
        if method == baseline:
            continue
        comparisons[method] = {}
        for condition in ("clean", "occluded"):
            success_to_failure, failure_to_success = 0, 0
            for pair in selected:
                b = states[(pair["pair_id"], baseline)][condition] == "success"
                m = states[(pair["pair_id"], method)][condition] == "success"
                success_to_failure += b and not m
                failure_to_success += not b and m
            comparisons[method][condition] = {"baseline_success_to_method_failure": success_to_failure,
                                             "baseline_failure_to_method_success": failure_to_success,
                                             "common_pair_error_difference": {f: summary([errors[(p["pair_id"], method, condition)][f] - errors[(p["pair_id"], baseline, condition)][f] for p in common]) for f in metric_fields},
                                             "common_pair_worsened": {f: sum(errors[(p["pair_id"], method, condition)][f] - errors[(p["pair_id"], baseline, condition)][f] > 1e-9 for p in common) for f in metric_fields}}
    result["baseline_method"] = baseline
    result["comparisons"] = comparisons
    recordings = sorted({p["recording_id"] for p in selected})
    boot = {"unit": "whole_recording", "resamples": resamples, "seed": BOOTSTRAP_SEED,
            "recordings": len(recordings), "shared_draws_across_methods": True, "multiplicity_adjustment": False}
    boot["common_success_recordings"] = len({p["recording_id"] for p in common})
    if boot["common_success_recordings"] < 2:
        boot.update(status="NOT_ESTIMATED", reason="fewer than two recording units in common success intersection")
    else:
        grouped = {g: [p for p in common if p["recording_id"] == g] for g in recordings}
        distributions = {(m, f): [] for m in methods for f in metric_fields}
        relative = {(m, f): [] for m in methods if m != baseline for f in metric_fields}
        rng = random.Random(BOOTSTRAP_SEED)
        skipped = 0
        for _ in range(resamples):
            sample = [p for g in rng.choices(recordings, k=len(recordings)) for p in grouped[g]]
            if not sample:
                skipped += 1
                continue
            sample_means = {}
            for m in methods:
                for f in metric_fields:
                    value = statistics.mean(errors[(p["pair_id"], m, "occluded")][f] - errors[(p["pair_id"], m, "clean")][f] for p in sample)
                    distributions[(m, f)].append(value)
                    sample_means[(m, f)] = value
            for m, f in relative:
                relative[(m, f)].append(sample_means[(m, f)] - sample_means[(baseline, f)])
        boot.update(status="DONE", empty_intersection_draws=skipped,
                    paired_mean_delta_95CI={m: {f: [quantile(distributions[(m, f)], .025), quantile(distributions[(m, f)], .975)] for f in metric_fields} for m in methods},
                    method_minus_baseline_paired_mean_delta_95CI={m: {f: [quantile(relative[(m, f)], .025), quantile(relative[(m, f)], .975)] for f in metric_fields} for m in methods if m != baseline})
    result["bootstrap"] = boot
    return result, rows, delta_rows


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def precision_source(scores_path, receipt_path, output):
    """Reuse hash-locked DEV319 pose scores solely to describe session variation."""
    source_hash = sha256(scores_path)
    receipt = read_json(receipt_path)
    bindings = [item for item in receipt["inputs"] if Path(item["path"]).name == "YOLO_SCORES.json"]
    if len(bindings) != 1 or source_hash != bindings[0]["sha256"]:
        raise ValueError("YOLO_SCORES does not match the existing paired-pose audit receipt")
    scores = read_json(scores_path)
    base = {row["id"]: row for row in scores["R0"]["pose_scores"]}
    sessions = sorted({row["session"] for row in base.values()})
    if len(base) != 319 or sessions != sorted(receipt["contract"]["sessions"]):
        raise ValueError("locked DEV319 session population mismatch")
    rows, per_session = [], {}
    for seed in (1, 2, 3):
        candidate = {row["id"]: row for row in scores[f"N3_DIM_SYM_seed{seed}"]["pose_scores"]}
        if set(candidate) != set(base) or any(candidate[fid]["session"] != base[fid]["session"] for fid in base):
            raise ValueError("pose scores are not paired by exact frame/session")
        for session in sessions:
            ids = [fid for fid, row in base.items() if row["session"] == session]
            common = [fid for fid in ids if base[fid]["available"] and candidate[fid]["available"]]
            row = {"session": session, "seed": seed, "full_frames": len(ids), "common_success_frames": len(common)}
            for field in ("translation_cm", "rotation_deg"):
                values = [candidate[fid][field] - base[fid][field] for fid in common]
                row[f"paired_mean_delta_{field}"] = statistics.mean(values) if values else None
                row[f"paired_median_delta_{field}"] = quantile(values, .5)
                per_session.setdefault((session, field), []).append(row[f"paired_mean_delta_{field}"])
            rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["session", "seed", "full_frames", "common_success_frames", "paired_mean_delta_translation_cm", "paired_median_delta_translation_cm", "paired_mean_delta_rotation_deg", "paired_median_delta_rotation_deg"]
    write_csv(output / "B_REUSED_DEV319_SESSION_VARIATION.csv", rows, fields)
    result = {"schema": "pallet_B_precision_source_20261006_v1", "status": "REUSED_VALIDATED", "frames": len(base), "sessions": len(sessions),
              "seeds": [1, 2, 3], "independent_units": "13 recorded development sessions; seeds are averaged within session",
              "reference": "reconstructed annotation/geometry pose; NOT independent physical metrology",
              "new_pose_solves": 0, "source_sha256": source_hash, "existing_receipt_sha256": sha256(receipt_path), "metrics": {},
              "B_sample_size": {"status": "BLOCKED_DATA", "reason": "these N3-minus-Base differences are not physical occluded-minus-clean pairs and cannot supply B pilot variance"}}
    for field in ("translation_cm", "rotation_deg"):
        values = [statistics.mean(per_session[(s, field)]) for s in sessions if all(x is not None for x in per_session[(s, field)])]
        result["metrics"][field] = {"seed_mean_session_mean_deltas": {s: statistics.mean(per_session[(s, field)]) for s in sessions if all(x is not None for x in per_session[(s, field)])},
                                    "between_session_sd": statistics.stdev(values) if len(values) > 1 else None,
                                    "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                                    "quantity": "within-session mean(N3 error - Base error), then mean of 3 seeds"}
    write_json(output / "B_REUSED_DEV319_SESSION_VARIATION.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reuse = sub.add_parser("precision-source", help="Reuse hash-locked DEV319 scores to describe session variation, not B variance")
    reuse.add_argument("--scores", required=True, type=Path)
    reuse.add_argument("--receipt", required=True, type=Path)
    reuse.add_argument("--output", required=True, type=Path)
    for command in ("import", "validate", "evaluate"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--manifest", required=True, type=Path)
        cmd.add_argument("--data-root", required=True, type=Path)
        cmd.add_argument("--allow-fixture", action="store_true", help="Explicit non-scientific format fixtures only")
        cmd.add_argument("--output", required=True, type=Path, help="JSON file for import/validate; directory for evaluate")
        if command == "evaluate":
            cmd.add_argument("--predictions", required=True, type=Path, help="JSONL final canonical poses; omitted frames count as failure")
            cmd.add_argument("--split", choices=("train", "development", "independent_test"), default="independent_test")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if args.command == "precision-source":
            result = precision_source(args.scores, args.receipt, args.output)
            print(json.dumps({"command": args.command, "status": result["status"], "frames": result["frames"], "sessions": result["sessions"], "B_sample_size": result["B_sample_size"]}, ensure_ascii=False))
            return 0
        manifest = read_json(args.manifest)
        errors = validate_manifest(manifest, args.data_root, args.allow_fixture)
        receipt = {"schema": "pallet_measurement_validation_20261006_v1", "status": "PASS" if not errors else "FAIL",
                   "errors": errors, "manifest_sha256": sha256(args.manifest), "pairs": len(manifest.get("pairs", [])),
                   "scientific_evidence": not errors and manifest.get("data_kind") == "physical_measurement", "command": args.command}
        if errors:
            write_json(args.output / "VALIDATION.json" if args.command == "evaluate" else args.output, receipt)
            print(json.dumps(receipt, ensure_ascii=False))
            return 2
        if args.command == "import":
            imported = copy.deepcopy(manifest)
            imported["import_receipt"] = receipt
            write_json(args.output, imported)
        elif args.command == "validate":
            receipt["elapsed_wall_seconds"] = time.perf_counter() - start
            write_json(args.output, receipt)
        else:
            predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
            result, rows, deltas = evaluate(manifest, predictions, args.split)
            args.output.mkdir(parents=True, exist_ok=True)
            result["manifest_sha256"] = sha256(args.manifest)
            result["predictions_sha256"] = sha256(args.predictions)
            result["elapsed_wall_seconds"] = time.perf_counter() - start
            write_json(args.output / "SUMMARY.json", result)
            write_json(args.output / "VALIDATION.json", receipt)
            write_csv(args.output / "FRAME_ERRORS.csv", rows, ["pair_id", "recording_id", "method", "condition", "frame_id", "state", "translation_cm", "rotation_deg", "ADDsym_m"])
            write_csv(args.output / "COMMON_PAIR_DELTAS.csv", deltas, ["pair_id", "recording_id", "method", "translation_cm", "rotation_deg", "ADDsym_m"])
        print(json.dumps({"command": args.command, "status": "DONE", "output": str(args.output), "scientific_evidence": receipt["scientific_evidence"]}, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"measurement {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
