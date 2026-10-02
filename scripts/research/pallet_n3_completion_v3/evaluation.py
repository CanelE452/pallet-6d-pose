"""DEV319 evaluation adapter for the DOPE/ResNet-18 N3 experiments.

The evaluator deliberately joins reference data from the frozen paper
manifest.  A prediction artifact may contain fields named ``gt`` for legacy
compatibility, but those fields are never read here.  Both the new combined
inference schema and the historical DOPE/ResNet refiner schemas are accepted.

All numeric work is delegated to :mod:`metrics`, which in turn reuses the
locked 8-corner and prediction-only pose contracts.  Missing seeds, incomplete
artifacts, and identity/preservation failures produce explicit ``null``/``x``
cells instead of copied or fabricated results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from challenge.evaluation_v2 import paper_real_eval
from challenge.evaluation_v2.real_dataset_contract import load_population_manifest
from scripts.research.pallet_n3_completion_v3 import common as C
from scripts.research.pallet_n3_completion_v3 import metrics as M


SCHEMA = "pallet_n3_completion_v3_evaluation_v1"
EXPECTED_DEV_FRAMES = 319
MATCH_IOU = .5
METHODS = ("base", "n3_seed1", "n3_seed2", "n3_seed3")
SYMMETRY_CONTRACT = (
    C.ROOT / "_docs/experiments/pallet_symmetry_three_line_v1/"
             "OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json")
POSE_ROOT = C.ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
POSE_AXIS = POSE_ROOT / "AXIS_REVIEW_MANIFEST.json"
POSE_GT = POSE_ROOT / "GEOMETRY_RESOLVED_POSE_GT.json"


def _canonical_method(value: Any, *, n3_context: bool = False) -> str | None:
    if value is None:
        return None
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if token in {"base", "baseline", "raw", "original", "unrefined", "r0"}:
        return "base"
    match = re.fullmatch(r"(?:n3(?:_dim_sym)?_?)?seed_?([123])", token)
    if match and (n3_context or token.startswith("n3")):
        return f"n3_seed{match.group(1)}"
    match = re.search(r"(?:^|_)n3(?:_dim_sym)?_?seed_?([123])(?:_|$)", token)
    if match:
        return f"n3_seed{match.group(1)}"
    return None


def _frame_id(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("frame_id"))
    if value is None or not str(value).strip():
        raise ValueError("Prediction row is missing id/frame_id")
    return str(value)


def _as_points(value: Any, valid: Any = None) -> list[list[float]] | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (9, 2) or np.isinf(points).any():
        raise ValueError("Prediction points must be finite/NaN [9,2]")
    if not np.array_equal(np.isfinite(points[:, 0]), np.isfinite(points[:, 1])):
        raise ValueError("Prediction has a partly missing x/y pair")
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (9,):
            raise ValueError("Prediction validity mask must have shape [9]")
        points = points.copy()
        points[~mask] = np.nan
    return points.tolist()


def _as_box(value: Any) -> list[float] | None:
    if value is None:
        return None
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Prediction bbox must be finite xyxy or null")
    return box.tolist()


def _prediction_block(value: Any, *, inherited: Mapping[str, Any] | None = None,
                      method: str | None = None) -> dict:
    inherited = {} if inherited is None else inherited
    if isinstance(value, Mapping):
        points = next((value[key] for key in (
            "points", "points_original", "keypoints_xy", "refined_points",
            "base_points") if key in value), None)
        valid = value.get("valid", value.get("point_valid", inherited.get("valid")))
        box_value = next((value[key] for key in (
            "bbox", "bbox_original", "box_original", "box_xyxy") if key in value),
            inherited.get("bbox"))
        score = value.get("score", inherited.get("score"))
        detected_value = value.get("detected")
        status = value.get("status", inherited.get("status"))
    else:
        points = value
        valid = inherited.get("valid")
        box_value = inherited.get("bbox")
        score = inherited.get("score")
        detected_value = None
        status = None
    normalized_points = _as_points(points, valid)
    normalized_box = _as_box(box_value)
    point_mask = (np.zeros(9, dtype=bool) if normalized_points is None else
                  np.isfinite(np.asarray(normalized_points, dtype=float)).all(-1))
    normalized_valid = (np.asarray(valid, dtype=bool) if valid is not None
                        else point_mask)
    if normalized_valid.shape != (9,):
        raise ValueError("Prediction validity mask must have shape [9]")
    if detected_value is None:
        detected = (bool(normalized_valid[:8].any())
                    if valid is not None or points is not None
                    else normalized_box is not None)
    else:
        detected = bool(detected_value)
    if status is not None and str(status).strip().lower() in {
            "missing", "no_detection", "not_detected", "empty"}:
        if detected:
            raise ValueError("Prediction status says missing while detected=true")
        detected = False
    return {
        "method": method,
        "points": normalized_points,
        "valid": normalized_valid.tolist(),
        "bbox": normalized_box,
        "score": None if score is None else float(score),
        "detected": detected,
        "status": None if status is None else str(status),
    }


def _selected_candidate(record: Mapping[str, Any]) -> Any:
    if "candidates" not in record:
        return None
    candidates = record["candidates"]
    selected = record.get("selected_index")
    if selected is None:
        return {"points": None, "bbox": None, "detected": False,
                "valid": [False] * 9}
    if (not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes))
            or not isinstance(selected, int) or not 0 <= selected < len(candidates)):
        raise ValueError(f"{_frame_id(record)}: invalid selected candidate")
    return candidates[selected]


def _record_methods(record: Mapping[str, Any], method_hint: str | None) -> dict[str, dict]:
    """Extract method predictions from one modern or legacy frame."""
    output: dict[str, dict] = {}

    # New combined inference: frame.base and frame.N3_seed1/2/3.
    for key, value in record.items():
        method = _canonical_method(key)
        if method is not None and (isinstance(value, Mapping) or value is None):
            output[method] = _prediction_block(value, method=method)

    # Generic per-frame prediction maps and historical `refined` maps.
    for container_key in ("predictions", "outputs", "refined"):
        container = record.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            method = _canonical_method(key)
            if method is not None:
                output[method] = _prediction_block(value, method=method)
    container = record.get("n3")
    if isinstance(container, Mapping):
        for key, value in container.items():
            method = _canonical_method(key, n3_context=True)
            if method is not None:
                output[method] = _prediction_block(value, method=method)

    # Historical DOPE/ResNet combined schema.  P*/D* keys are intentionally
    # ignored: they are comparison methods and cannot stand in for N3.
    if "base_points" in record:
        inherited = {
            "valid": record.get("point_valid"),
            "bbox": record.get("box_original", record.get("bbox_original")),
            "score": record.get("score"),
        }
        output["base"] = _prediction_block(
            {"points": record["base_points"], **inherited}, method="base")
        refined = record.get("refined")
        if isinstance(refined, Mapping):
            for key, value in refined.items():
                method = _canonical_method(key)
                if method is not None:
                    output[method] = _prediction_block(
                        value, inherited=inherited, method=method)

    # A method-specific artifact may expose a selected candidate or a single
    # prediction block rather than a combined frame.
    if method_hint is not None and method_hint not in output:
        selected = _selected_candidate(record)
        if selected is not None:
            output[method_hint] = _prediction_block(selected, method=method_hint)
        elif any(key in record for key in (
                "points", "points_original", "keypoints_xy", "bbox",
                "bbox_original", "box_xyxy")):
            output[method_hint] = _prediction_block(record, method=method_hint)
    return output


def _artifact_coordinate_status(payload: Mapping[str, Any]) -> tuple[bool, str | None]:
    value = payload.get("coordinate_system")
    if value is None:
        # The new schema fixes points/bboxes to original pixels by definition.
        return True, None
    token = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    accepted = {
        "original_pixels", "original_pixel", "original_unpadded_pixels",
        "original_image_pixels", "raw_pixels", "raw_image_pixels",
    }
    return (token in accepted,
            None if token in accepted else f"unsupported coordinate_system={value!r}")


def normalize_prediction_payload(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                                 *, method_hint: str | None = None) -> dict:
    """Normalize one raw inference artifact without consulting any GT fields."""
    if method_hint is not None:
        method_hint = _canonical_method(method_hint, n3_context=True)
        if method_hint is None:
            raise ValueError("Unknown method hint")
    if isinstance(payload, Mapping):
        if method_hint is None:
            method_hint = _canonical_method(
                payload.get("method"), n3_context=True)
        coordinate_ok, coordinate_error = _artifact_coordinate_status(payload)
        complete = payload.get("complete") is not False
        schema = payload.get("schema")
        records = payload.get("frames", payload.get("records"))
        if records is None and isinstance(payload.get("predictions"), Mapping):
            # Method-major generic schema: predictions[method] -> records.
            methods: dict[str, dict[str, dict]] = {}
            for key, rows in payload["predictions"].items():
                method = _canonical_method(key)
                if method is None or not isinstance(rows, Sequence):
                    continue
                nested = normalize_prediction_payload(
                    list(rows), method_hint=method)
                for nested_method, values in nested["methods"].items():
                    methods.setdefault(nested_method, {}).update(values)
            return {
                "schema": schema, "complete": complete,
                "coordinate_ok": coordinate_ok,
                "coordinate_error": coordinate_error, "methods": methods,
            }
        if records is None:
            raise ValueError("Prediction payload requires frames or records")
    else:
        coordinate_ok, coordinate_error = True, None
        complete, schema, records = True, None, payload
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Prediction frames/records must be a list")
    methods: dict[str, dict[str, dict]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("Prediction frame must be an object")
        frame_id = _frame_id(raw)
        frame_methods = _record_methods(raw, method_hint)
        for method, prediction in frame_methods.items():
            bucket = methods.setdefault(method, {})
            if frame_id in bucket:
                raise ValueError(f"Duplicate {method} prediction id: {frame_id}")
            prediction = dict(prediction)
            prediction["id"] = frame_id
            raw_hw = raw.get("raw_shape_hw", raw.get(
                "original_hw", raw.get("raw_hw")))
            if raw_hw is not None:
                hw = np.asarray(raw_hw, dtype=np.int64)
                if hw.shape != (2,) or np.any(hw <= 0):
                    raise ValueError(f"{frame_id}: invalid raw/original hw")
                prediction["hw"] = hw.tolist()
            session = raw.get("session_id", raw.get("session"))
            if session is not None:
                prediction["session"] = str(session)
            if raw.get("object_type") is not None:
                prediction["object_type"] = str(raw["object_type"])
            bucket[frame_id] = prediction
    return {
        "schema": schema, "complete": complete,
        "coordinate_ok": coordinate_ok,
        "coordinate_error": coordinate_error, "methods": methods,
    }


def _iou(first: Any, second: Any) -> float:
    if first is None or second is None:
        return 0.
    left, right = np.asarray(first, float), np.asarray(second, float)
    intersection = np.maximum(
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]), 0).prod()
    union = (np.maximum(left[2:] - left[:2], 0).prod()
             + np.maximum(right[2:] - right[:2], 0).prod() - intersection)
    return float(intersection / union) if union > 0 else 0.


def _rotation_y_quarter() -> np.ndarray:
    theta = np.pi / 2
    return np.array([[np.cos(theta), 0., np.sin(theta)],
                     [0., 1., 0.],
                     [-np.sin(theta), 0., np.cos(theta)]])


def _pose_specs(items, targets: Mapping[str, Any], groups: Mapping[str, dict]) -> dict:
    if not POSE_AXIS.is_file() or not POSE_GT.is_file():
        return {}
    axis = C.read(POSE_AXIS).get("frames_list", [])
    pose_gt = C.read(POSE_GT).get("frames", {})
    ids_by_image = {item.image: item.frame_id for item in items}
    specs = {}
    for row in axis:
        frame_id = ids_by_image.get(row.get("image"))
        reference = pose_gt.get(row.get("frame_id"))
        if frame_id is None or not isinstance(reference, Mapping) or not reference.get("resolved"):
            continue
        target = targets[frame_id]
        physical = target.physical_dimensions
        xyz = np.asarray(
            [physical.x_m, physical.y_m, physical.z_m], dtype=np.float64)
        dimensions = reference.get("physical_dimensions_m", {})
        try:
            body_xyz = np.asarray([
                dimensions["across"], dimensions["height"], dimensions["along"]],
                dtype=np.float64)
            body_rotation = np.asarray(reference["R_gt_representative"], dtype=np.float64)
            translation = np.asarray(reference["t_gt"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if (xyz.shape != (3,) or body_xyz.shape != (3,) or
                body_rotation.shape != (3, 3) or translation.shape != (3,)):
            continue
        quarter = np.eye(3) if abs(body_xyz[0] - xyz[0]) < 1e-6 else _rotation_y_quarter()
        order = int(groups[target.object_type]["group_order"])
        specs[frame_id] = {
            "K": np.asarray(target.camera_intrinsics, float).tolist(),
            "xyz": xyz.tolist(), "source": False,
            "truth": {
                "R": (body_rotation @ quarter).tolist(),
                "t": translation.tolist(), "xyz": xyz.tolist(),
                "body_R": body_rotation.tolist(),
                "body_xyz": body_xyz.tolist(), "order": order,
            },
        }
    return specs


def _material(object_type: str | None) -> str:
    # Material is explicit in the two frozen registry object types.  Unknown
    # types are not guessed from appearance or performance.
    if object_type == "plastic_standard_110x130x11":
        return "plastic"
    if object_type == "wood_small_80x59x14":
        return "wood"
    return M.UNCLASSIFIED


def _occlusion(value: Any) -> str:
    if value is None or str(value).strip().lower() in {"", "unknown", "unspecified"}:
        return M.UNCLASSIFIED
    return str(value).strip()


def load_dev319_truth(manifest_path: Path | str = C.DEV,
                      *, include_pose: bool = True) -> list[dict]:
    """Load frozen DEV319 labels without requiring local image bytes."""
    population = load_population_manifest(manifest_path, validate_files=False)
    if population.count != EXPECTED_DEV_FRAMES:
        raise ValueError(f"DEV population must contain {EXPECTED_DEV_FRAMES} frames")
    symmetry = C.read(SYMMETRY_CONTRACT)
    groups = {row["object_type"]: row for row in symmetry["objects"]}
    targets = {}
    annotations = {}
    for item in population.items:
        if item.object_type not in groups:
            raise ValueError(f"No approved symmetry for {item.object_type}")
        targets[item.frame_id] = paper_real_eval._legacy_forbidden_target(item)
        annotations[item.frame_id] = C.read(C.ROOT / item.label)
    pose = _pose_specs(population.items, targets, groups) if include_pose else {}
    rows = []
    for item in population.items:
        target = targets[item.frame_id]
        annotation = annotations[item.frame_id]
        camera = annotation.get("camera_data", {})
        obj = annotation.get("objects", [{}])[0]
        row = {
            "id": item.frame_id,
            "session": item.session_id,
            "hw": [int(camera["height"]), int(camera["width"])],
            "gt": np.asarray(target.keypoints_xy, float).tolist(),
            "valid": np.asarray(target.keypoint_supervision_mask, bool).tolist(),
            "box": np.asarray(target.box_xyxy, float).tolist(),
            "permutations": groups[item.object_type]["permutations"],
            "object_type": item.object_type,
            "material": _material(item.object_type),
            "occlusion": _occlusion(obj.get("occlusion_level")),
        }
        if item.frame_id in pose:
            row["pose"] = pose[item.frame_id]
        rows.append(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != EXPECTED_DEV_FRAMES:
        raise ValueError("DEV319 truth identity is not unique")
    return rows


def load_truth_payload(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict]:
    """Load a test/portable truth sidecar in the metrics row schema."""
    rows = payload.get("rows", payload.get("frames")) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Truth sidecar requires rows/frames list")
    output = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Truth row must be an object")
        value = dict(row)
        value["id"] = _frame_id(row)
        value.setdefault("material", M.UNCLASSIFIED)
        value.setdefault("occlusion", M.UNCLASSIFIED)
        output.append(value)
    if len({row["id"] for row in output}) != len(output):
        raise ValueError("Duplicate truth frame id")
    return output


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    first, second = np.asarray(left), np.asarray(right)
    try:
        return bool(np.array_equal(first, second, equal_nan=True))
    except TypeError:  # string/object provenance fields do not support isnan
        return bool(np.array_equal(first, second))


def _preservation_error(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> str | None:
    for field in ("detected", "status", "valid", "bbox", "score"):
        if not _same(base.get(field), candidate.get(field)):
            return f"{field} changed"
    base_points, candidate_points = base.get("points"), candidate.get("points")
    base_center = None if base_points is None else np.asarray(base_points, float)[8]
    candidate_center = None if candidate_points is None else np.asarray(candidate_points, float)[8]
    if not _same(base_center, candidate_center):
        return "center8 changed"
    return None


def _empty_method(status: str, reason: Any = None) -> dict:
    return {"status": status, "reason": reason, "result": None,
            "headline": None, "display": "x"}


def _headline(result: Mapping[str, Any]) -> dict:
    corner = result["corner"]
    values = dict(corner["metrics"])
    values.update(corner["denominators"])
    if result.get("pose") is not None:
        pose = result["pose"]
        values.update(
            pose_translation_cm_median=pose["translation_cm"]["median"],
            pose_translation_cm_P90=pose["translation_cm"]["P90"],
            pose_rotation_deg_median=pose["rotation_deg"]["median"],
            pose_rotation_deg_P90=pose["rotation_deg"]["P90"],
            pose_coverage=pose["denominators"]["coverage"],
            ADDsym_AUC_full=pose["ADDsym_AUC_full"],
        )
    else:
        values.update(
            pose_translation_cm_median=None, pose_translation_cm_P90=None,
            pose_rotation_deg_median=None, pose_rotation_deg_P90=None,
            pose_coverage=None, ADDsym_AUC_full=None,
        )
    return values


def _seed_aggregate(methods: Mapping[str, Mapping[str, Any]]) -> dict:
    names = [f"n3_seed{seed}" for seed in C.SEEDS]
    if any(methods[name]["status"] != "COMPLETE" for name in names):
        missing = [name for name in names if methods[name]["status"] != "COMPLETE"]
        return {"status": "INCOMPLETE", "missing": missing,
                "mean_of_seed_metrics": None, "display": "x"}
    headlines = [methods[name]["headline"] for name in names]
    numeric_keys = sorted(set.intersection(*(
        {key for key, value in row.items()
         if isinstance(value, (int, float)) and not isinstance(value, bool)}
        for row in headlines)))
    denominator_keys = {
        "frames", "evaluable_frames", "matched_frames", "detected_frames",
        "missing_detection_frames", "full_supervised_corners", "observed_corners",
    }
    means = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in headlines]
        if key in denominator_keys and len(set(values)) == 1:
            means[key] = int(values[0])
        elif all(np.isfinite(values)):
            means[key] = float(np.mean(values))
    return {"status": "COMPLETE", "seeds": list(C.SEEDS),
            "aggregation": "arithmetic mean of three independently reported seed statistics",
            "mean_of_seed_metrics": means, "display": means}


def _population_groups(rows: Sequence[Mapping[str, Any]], field: str) -> dict:
    def label_of(row):
        value = row.get(field)
        return (M.UNCLASSIFIED if value is None or not str(value).strip()
                else str(value).strip())

    labels = sorted({label_of(row) for row in rows})
    return {group: {
        "frames": sum(label_of(row) == group for row in rows),
        "sessions": len({str(row["session"]) for row in rows
                         if label_of(row) == group}),
        "full_supervised_corners": int(sum(
            np.asarray(row["valid"], bool)[:8].sum() for row in rows
            if label_of(row) == group)),
    } for group in labels}


def evaluate_payloads(payloads: Sequence[tuple[str | None, Any]],
                      truth_rows: Sequence[Mapping[str, Any]], *,
                      backbone: str | None = None,
                      include_pose: bool = True,
                      expected_frames: int | None = EXPECTED_DEV_FRAMES) -> dict:
    """Evaluate base and three N3 seeds on one frozen, ordered population.

    ``payloads`` contains ``(method_hint, payload)`` pairs.  A hint is useful
    for one-file-per-method artifacts; a combined new/legacy payload needs no
    hint.  The function never writes files.
    """
    truth = [dict(row) for row in truth_rows]
    truth_ids = [str(row.get("id")) for row in truth]
    if any(value in {"", "None"} for value in truth_ids) or len(set(truth_ids)) != len(truth_ids):
        raise ValueError("Truth rows require unique ids")
    if expected_frames is not None and len(truth) != expected_frames:
        raise ValueError(f"Expected {expected_frames} truth rows, found {len(truth)}")
    normalized = [normalize_prediction_payload(payload, method_hint=hint)
                  for hint, payload in payloads]
    merged: dict[str, dict[str, dict]] = {}
    source_state: dict[str, list[dict]] = {method: [] for method in METHODS}
    for index, artifact in enumerate(normalized):
        for method, records in artifact["methods"].items():
            state = {
                "artifact_index": index, "complete": artifact["complete"],
                "coordinate_ok": artifact["coordinate_ok"],
                "coordinate_error": artifact["coordinate_error"],
                "schema": artifact["schema"],
            }
            source_state.setdefault(method, []).append(state)
            bucket = merged.setdefault(method, {})
            for frame_id, prediction in records.items():
                if frame_id in bucket and not all(
                        _same(bucket[frame_id].get(field), prediction.get(field))
                        for field in ("points", "valid", "bbox", "score", "detected",
                                      "status", "hw", "session", "object_type")):
                    raise ValueError(f"Conflicting duplicate {method} prediction: {frame_id}")
                bucket[frame_id] = prediction

    entries: dict[str, dict] = {}
    ready: dict[str, dict[str, dict]] = {}
    expected = set(truth_ids)
    for method in METHODS:
        records = merged.get(method)
        if not records:
            entries[method] = _empty_method("MISSING", "no raw prediction artifact")
            continue
        states = source_state[method]
        if any(not state["coordinate_ok"] for state in states):
            entries[method] = _empty_method(
                "BLOCKED_COORDINATE_SYSTEM",
                [state["coordinate_error"] for state in states
                 if not state["coordinate_ok"]])
            continue
        if any(not state["complete"] for state in states):
            entries[method] = _empty_method("INCOMPLETE_ARTIFACT")
            continue
        actual = set(records)
        if actual != expected:
            entries[method] = _empty_method("BLOCKED_IDENTITY", {
                "missing_ids": sorted(expected - actual),
                "extra_ids": sorted(actual - expected),
            })
            continue
        metadata_errors = []
        truth_by_id = {str(row["id"]): row for row in truth}
        for frame_id in truth_ids:
            prediction = records[frame_id]
            reference = truth_by_id[frame_id]
            if (prediction.get("hw") is not None and
                    list(prediction["hw"]) != list(reference["hw"])):
                metadata_errors.append({"id": frame_id, "field": "hw"})
            if (prediction.get("session") is not None and
                    str(prediction["session"]) != str(reference["session"])):
                metadata_errors.append({"id": frame_id, "field": "session"})
            if (prediction.get("object_type") is not None and
                    reference.get("object_type") is not None and
                    prediction["object_type"] != reference["object_type"]):
                metadata_errors.append({"id": frame_id, "field": "object_type"})
        if metadata_errors:
            entries[method] = _empty_method("BLOCKED_METADATA", {
                "count": len(metadata_errors), "examples": metadata_errors[:20]})
            continue
        ready[method] = records

    # N3 is required to preserve the base detector branch, box, validity,
    # score, and center.  This check happens before any favorable metric exists.
    if "base" in ready:
        for method in METHODS[1:]:
            if method not in ready:
                continue
            violations = []
            for frame_id in truth_ids:
                error = _preservation_error(ready["base"][frame_id], ready[method][frame_id])
                if error is not None:
                    violations.append({"id": frame_id, "error": error})
            if violations:
                entries[method] = _empty_method(
                    "BLOCKED_PRESERVATION", {
                        "count": len(violations), "examples": violations[:20]})
                del ready[method]
    else:
        for method in METHODS[1:]:
            if method in ready:
                entries[method] = _empty_method(
                    "BLOCKED_BASE_MISSING",
                    "base is required for preservation and paired evaluation")
                del ready[method]

    rows = []
    for reference in truth:
        frame_id = str(reference["id"])
        row = {key: reference[key] for key in (
            "id", "session", "hw", "gt", "valid", "permutations")}
        row["material"] = reference.get("material", M.UNCLASSIFIED)
        row["occlusion"] = reference.get("occlusion", M.UNCLASSIFIED)
        if "pose" in reference:
            row["pose"] = reference["pose"]
        row["predictions"], row["matched"], row["detected"] = {}, {}, {}
        for method, records in ready.items():
            prediction = records[frame_id]
            row["predictions"][method] = prediction["points"]
            row["detected"][method] = prediction["detected"]
            row["matched"][method] = bool(
                prediction["detected"] and
                _iou(prediction["bbox"], reference.get("box")) >= MATCH_IOU)
        rows.append(row)

    pose_available = include_pose and bool(rows) and all("pose" in row for row in rows)
    scored_rows = {}
    for method in METHODS:
        if method not in ready:
            continue
        result = M.evaluate_method(rows, method, include_pose=pose_available)
        if include_pose and not pose_available:
            result["pose"] = None
            result["pose_subgroups"] = None
            result["pose_status"] = "MISSING_REFERENCE"
        headline = _headline(result)
        entries[method] = {
            "status": "COMPLETE", "reason": None, "result": result,
            "headline": headline, "display": headline,
        }
        scored_rows[method] = result["corner_rows"]

    comparisons = {}
    for method in METHODS[1:]:
        if "base" in scored_rows and method in scored_rows:
            comparisons[f"base_to_{method}"] = {
                "status": "COMPLETE",
                "result": M.paired_corner_analysis(
                    scored_rows["base"], scored_rows[method]),
            }
        else:
            comparisons[f"base_to_{method}"] = {
                "status": "MISSING", "result": None, "display": "x"}

    missing = [method for method in METHODS if entries[method]["status"] != "COMPLETE"]
    return {
        "schema": SCHEMA,
        "complete": not missing and (pose_available or not include_pose),
        "backbone": backbone,
        "population": {
            "name": "DEV319" if expected_frames == EXPECTED_DEV_FRAMES else "provided",
            "frames": len(truth), "frame_order": truth_ids,
            "sessions": len({str(row["session"]) for row in truth}),
            "full_supervised_corners": int(sum(
                np.asarray(row["valid"], bool)[:8].sum() for row in truth)),
            "subgroups": {
                "material": _population_groups(truth, "material"),
                "occlusion": _population_groups(truth, "occlusion"),
            },
        },
        "methods": entries,
        "comparisons": comparisons,
        "seed_aggregate": _seed_aggregate(entries),
        "missing_or_blocked_methods": missing,
        "pose_status": ("COMPLETE" if pose_available else
                        "DISABLED" if not include_pose else "MISSING_REFERENCE"),
        "contract": {
            "truth_source": "caller-provided frozen rows; raw prediction gt fields ignored",
            "corner_metric": "locked matched pooled 8-corner median/P90 plus full-population PCK10/E_sym/diagonal penalty",
            "match": "prediction bbox versus frozen GT bbox, IoU >= 0.5",
            "pose": "prediction-only frozen PnP; GT match does not gate pose",
            "pose_reference": "DEV pose reconstructed from image annotation and registered geometry; not independent physical 6D measurement",
            "subgroups": "explicit material and occlusion labels; absent/unknown is unclassified",
            "paired_bootstrap": {
                "unit": "session", "resamples": M.BOOTSTRAP_RESAMPLES,
                "seed": M.BOOTSTRAP_SEED,
            },
            "n3_identity": "N3 only; legacy P/D outputs are never aliases for N3",
            "missing_policy": "null result with display x; no imputation or favorable filtering",
        },
    }


def _summary_for_stdout(result: Mapping[str, Any]) -> dict:
    return {
        "schema": result["schema"], "complete": result["complete"],
        "backbone": result["backbone"], "population": result["population"]["frames"],
        "methods": {name: {
            "status": entry["status"], "headline": entry["headline"]}
            for name, entry in result["methods"].items()},
        "missing_or_blocked_methods": result["missing_or_blocked_methods"],
        "pose_status": result["pose_status"],
    }


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=tuple(C.CONFIGS))
    parser.add_argument("--predictions", action="append", default=[],
                        help="combined prediction JSON; may be repeated")
    parser.add_argument("--base")
    parser.add_argument("--seed1")
    parser.add_argument("--seed2")
    parser.add_argument("--seed3")
    parser.add_argument("--truth", help="optional portable truth rows JSON")
    parser.add_argument("--output", help="JSON destination under N3 DOC/RAW")
    parser.add_argument("--no-pose", action="store_true")
    arguments = parser.parse_args(argv)
    sources: list[tuple[str | None, Path]] = []
    sources.extend((None, Path(path)) for path in arguments.predictions)
    for method, path in (("base", arguments.base), ("n3_seed1", arguments.seed1),
                         ("n3_seed2", arguments.seed2), ("n3_seed3", arguments.seed3)):
        if path:
            sources.append((method, Path(path)))
    if not sources:
        parser.error("at least one prediction input is required")
    payloads = [(hint, C.read(path)) for hint, path in sources]
    if arguments.truth:
        truth_path = Path(arguments.truth)
        truth = load_truth_payload(C.read(truth_path))
        expected = None
    else:
        truth_path = C.DEV
        truth = load_dev319_truth(include_pose=not arguments.no_pose)
        expected = EXPECTED_DEV_FRAMES
    result = evaluate_payloads(
        payloads, truth, backbone=arguments.backbone,
        include_pose=not arguments.no_pose, expected_frames=expected)
    result["inputs"] = {
        "predictions": [dict(method_hint=hint, **C.binding(path))
                        for hint, path in sources],
        "truth": C.binding(truth_path),
    }
    destination = (Path(arguments.output) if arguments.output else
                   C.RAW / "evaluation" / f"{arguments.backbone}.json")
    C.write(destination, result)
    print(json.dumps(_summary_for_stdout(result), ensure_ascii=False, indent=2,
                     allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
