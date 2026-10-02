"""Frozen YOLO R0/OLD_P/N2/N3 transfer audit on GREEN0918_119.

Inference is deliberately separate from scoring: it reads images, the fixed
1.10 x 1.10 x 0.15 m registry entry, and frozen synthetic-selection locks,
but never annotations.  Scoring subsequently applies the already approved C4
whole-object evaluator to the two fixed manual-label denominators.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import numbers
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

from . import common as C


DATASET = C.ROOT / "_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json"
LEGACY_RAW = C.ROOT / "data/pallet/results/pallet_green0918_dimension_audit_v1/PREDICTIONS.json"
LEGACY_METRICS = C.ROOT / "data/pallet/results/pallet_green0918_dimension_audit_v1/METRICS.json"
OLD_SELECTION = C.ROOT / "_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/P_SELECTION.json"
DCP_SELECTION = C.ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json"
NORMALIZATION = C.ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"
SYMMETRY = C.ROOT / "_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
R0_WEIGHTS = C.ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
R0_SHA256 = "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7"

PREDICTION_OUTPUT = C.RAW / "SQUARE_YOLO_PREDICTIONS.json"
METRICS_OUTPUT = C.RAW / "SQUARE_YOLO_METRICS.json"
RESULT_OUTPUT = C.DOC / "SQUARE_YOLO_RESULTS.json"

DIMENSIONS_WDH_M = [1.1, 1.1, .15]
OBJECT_TYPE = "plastic_standard_110x110x15"
SEEDS = (1, 2, 3)
REUSED_METHODS = ("R0", *(f"N2_DIM_ONLY_seed{s}" for s in SEEDS))
NEW_METHODS = (*(f"OLD_P_seed{s}" for s in SEEDS),
               *(f"N3_DIM_SYM_seed{s}" for s in SEEDS))
METHODS = ("R0", *(f"OLD_P_seed{s}" for s in SEEDS),
           *(f"N2_DIM_ONLY_seed{s}" for s in SEEDS),
           *(f"N3_DIM_SYM_seed{s}" for s in SEEDS))
LEGACY_METHODS = ("R0", *(f"N0_BASE_REPLAY_seed{s}" for s in SEEDS),
                  *(f"N2_DIM_ONLY_seed{s}" for s in SEEDS))


def _verify_binding(binding: Mapping, path: Path | str) -> Path:
    """Verify a binding against an explicitly selected local file."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Bound input is missing: {path}")
    if C.sha256(path) != binding.get("sha256"):
        raise RuntimeError(f"Bound input hash changed: {path}")
    if "bytes" in binding and path.stat().st_size != binding["bytes"]:
        raise RuntimeError(f"Bound input size changed: {path}")
    return path


def _records(snapshot: Mapping, *, expected_count: int | None = 119) -> list[dict]:
    rows = snapshot.get("records")
    if not isinstance(rows, list) or (expected_count is not None and len(rows) != expected_count):
        raise RuntimeError("GREEN0918 membership/count drift")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) for value in ids):
        raise RuntimeError("GREEN0918 frame IDs are missing or duplicated")
    for row in rows:
        if row.get("canonical_WDH_m") != DIMENSIONS_WDH_M:
            raise RuntimeError(f"Square dimension drift: {row.get('id')}")
        hw = row.get("original_hw")
        if not (isinstance(hw, list) and len(hw) == 2 and min(hw) > 0):
            raise RuntimeError(f"Invalid raw shape: {row.get('id')}")
    return rows


def _candidate(candidate: Mapping, *, method: str, frame_id: str, index: int) -> None:
    required = {"candidate_index", "score", "box_xyxy", "keypoints_xy", "keypoints_conf"}
    if not required.issubset(candidate):
        raise RuntimeError(f"Incomplete candidate {method}/{frame_id}/{index}")
    if candidate["candidate_index"] != index:
        raise RuntimeError(f"Candidate index drift {method}/{frame_id}/{index}")
    box = np.asarray(candidate["box_xyxy"], dtype=np.float64)
    points = np.asarray(candidate["keypoints_xy"], dtype=np.float64)
    confidence = np.asarray(candidate["keypoints_conf"], dtype=np.float64)
    if box.shape != (4,) or points.shape != (9, 2) or confidence.shape != (9,):
        raise RuntimeError(f"Candidate tensor shape drift {method}/{frame_id}/{index}")
    if not np.isfinite(float(candidate["score"])):
        raise RuntimeError(f"Non-finite detection score {method}/{frame_id}/{index}")


def validate_prediction_rows(rows: Sequence[Mapping], records: Sequence[Mapping], method: str) -> None:
    if len(rows) != len(records):
        raise RuntimeError(f"Incomplete predictions for {method}: {len(rows)}/{len(records)}")
    for prediction, record in zip(rows, records):
        frame_id = record["id"]
        if prediction.get("id") != frame_id:
            raise RuntimeError(f"Prediction order drift for {method}: {prediction.get('id')} != {frame_id}")
        expected_hw = record.get("original_hw", record.get("hw"))
        if list(prediction.get("raw_hw", ())) != list(expected_hw):
            raise RuntimeError(f"Raw shape drift for {method}/{frame_id}")
        candidates = prediction.get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError(f"Candidates missing for {method}/{frame_id}")
        for index, candidate in enumerate(candidates):
            _candidate(candidate, method=method, frame_id=frame_id, index=index)
        selected = prediction.get("selected_index")
        expected = int(np.argmax([row["score"] for row in candidates])) if candidates else None
        if selected != expected:
            raise RuntimeError(f"Confidence selection drift for {method}/{frame_id}")


def validate_legacy(payload: Mapping, records: Sequence[Mapping], *, exact_methods: bool = True) -> None:
    if payload.get("complete") is not True:
        raise RuntimeError("Legacy GREEN0918 predictions are incomplete")
    if payload.get("GT_input") is not False or payload.get("camera_input") is not False:
        raise RuntimeError("Legacy GREEN0918 inference-input contract drift")
    if payload.get("dimensions_input") != DIMENSIONS_WDH_M:
        raise RuntimeError("Legacy GREEN0918 dimension vector drift")
    predictions = payload.get("predictions", {})
    expected = set(LEGACY_METHODS)
    wrong_inventory = (set(predictions) != expected if exact_methods else
                       not expected.issubset(predictions))
    if wrong_inventory:
        raise RuntimeError("Legacy GREEN0918 method inventory drift")
    for method in LEGACY_METHODS:
        validate_prediction_rows(predictions[method], records, method)
    for method in (f"N2_DIM_ONLY_seed{seed}" for seed in SEEDS):
        for base, refined in zip(predictions["R0"], predictions[method]):
            assert_detection_preserved(base, refined, method)


def load_legacy(prediction_path: Path | str = LEGACY_RAW,
                metrics_path: Path | str = LEGACY_METRICS,
                snapshot_path: Path | str = DATASET,
                *, expected_count: int | None = 119) -> tuple[dict, dict, list[dict]]:
    """Load R0/N2 only after the old metrics and dataset bindings verify."""
    prediction_path, metrics_path, snapshot_path = map(Path,
        (prediction_path, metrics_path, snapshot_path))
    metrics = C.read(metrics_path)
    _verify_binding(metrics.get("predictions", {}), prediction_path)
    payload = C.read(prediction_path)
    snapshot = C.read(snapshot_path)
    _verify_binding(payload.get("dataset", {}), snapshot_path)
    if "model" in payload:
        model_path = Path(payload["model"]["path"])
        if not model_path.is_absolute():
            model_path = C.ROOT / model_path
        _verify_binding(payload["model"], model_path)
    records = _records(snapshot, expected_count=expected_count)
    validate_legacy(payload, records)
    return payload, snapshot, records


def selection_contract(root: Path | str = C.ROOT, *, verify_checkpoints: bool = False) -> dict:
    """Read the frozen synthetic-only OLD_P and N3 selection decisions."""
    root = Path(root)
    old_path = root / OLD_SELECTION.relative_to(C.ROOT)
    dcp_path = root / DCP_SELECTION.relative_to(C.ROOT)
    old = C.read(old_path)
    dcp = C.read(dcp_path)
    if not (old.get("complete") and old.get("no_real_selection") and
            old.get("selection_population") == "synth_val" and
            old.get("heldout_accuracy_not_used") and old.get("real_accuracy_not_used")):
        raise RuntimeError("OLD_P synthetic-only selection contract drift")
    if not (dcp.get("complete") and dcp.get("real_access") is False and
            dcp.get("heldout_performance_access_before_lock") is False):
        raise RuntimeError("N3 synthetic-only calibration contract drift")
    expected_rule = {"lam": 1.0, "max_move_image_diagonal_fraction": .01}
    old_rule = {key: old["selected_rule"][key] for key in expected_rule}
    n3_rule = {key: dcp["rule"][key] for key in expected_rule}
    if old_rule != expected_rule or n3_rule != expected_rule:
        raise RuntimeError("Frozen decode rule drift")

    methods = {}
    for seed in SEEDS:
        name = f"OLD_P_seed{seed}"
        checkpoint = root / (
            f"data/pallet/results/pallet_final_ml_contribution_test_v1/"
            f"B_line_vs_point/runs/seed{seed}/last.pt")
        methods[name] = {
            "arm": "OLD_P", "seed": seed,
            "temperature": old["temperatures"][str(seed)]["temperature"],
            "rule": copy.deepcopy(old_rule), "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": old["checkpoints"][str(seed)],
        }
    for seed in SEEDS:
        name = f"N3_DIM_SYM_seed{seed}"
        selected = dcp["temperatures"][name]
        checkpoint = root / selected["checkpoint"]["path"]
        methods[name] = {
            "arm": "N3_DIM_SYM", "seed": seed,
            "temperature": selected["temperature"],
            "rule": copy.deepcopy(n3_rule), "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": selected["checkpoint"]["sha256"],
            "checkpoint_bytes": selected["checkpoint"].get("bytes"),
        }
    if set(methods) != set(NEW_METHODS):
        raise RuntimeError("New square inference inventory drift")
    if any(spec["temperature"] != 1.0 for spec in methods.values()):
        raise RuntimeError("Frozen synthetic temperature drift")
    if verify_checkpoints:
        for name, spec in methods.items():
            path = Path(spec["checkpoint_path"])
            if not path.is_file() or C.sha256(path) != spec["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint binding failed: {name}")
            if spec.get("checkpoint_bytes") is not None and path.stat().st_size != spec["checkpoint_bytes"]:
                raise RuntimeError(f"Checkpoint size binding failed: {name}")
    return {
        "methods": methods,
        "old_selection": C.binding(old_path), "n3_selection": C.binding(dcp_path),
        "selection_source": "synthetic calibration/validation only",
        "square_reselection_or_tuning": False,
    }


def _array_equal(left, right) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def assert_detection_preserved(base: Mapping, refined: Mapping, method: str = "refined") -> None:
    """Prove that a head changes only selected corners 0..7."""
    if base["id"] != refined["id"] or list(base["raw_hw"]) != list(refined["raw_hw"]):
        raise RuntimeError(f"Frame identity changed in {method}")
    if base["selected_index"] != refined["selected_index"]:
        raise RuntimeError(f"Selected detection changed in {method}/{base['id']}")
    before, after = base["candidates"], refined["candidates"]
    if len(before) != len(after):
        raise RuntimeError(f"Candidate count changed in {method}/{base['id']}")
    selected = base["selected_index"]
    for index, (left, right) in enumerate(zip(before, after)):
        if set(left) != set(right):
            raise RuntimeError(f"Candidate fields changed in {method}/{base['id']}")
        for key in left:
            if key != "keypoints_xy" and not _array_equal(left[key], right[key]):
                raise RuntimeError(f"Detection field {key} changed in {method}/{base['id']}")
        if index != selected:
            if not _array_equal(left["keypoints_xy"], right["keypoints_xy"]):
                raise RuntimeError(f"Unselected keypoints changed in {method}/{base['id']}")
        else:
            left_points = np.asarray(left["keypoints_xy"])
            right_points = np.asarray(right["keypoints_xy"])
            if not _array_equal(left_points[8], right_points[8]):
                raise RuntimeError(f"Center changed in {method}/{base['id']}")


def assert_captured_matches_r0(captured: Mapping, baseline: Mapping,
                               serial_fn: Callable[[Sequence], list]) -> None:
    replay = {
        "id": baseline["id"], "raw_hw": list(baseline["raw_hw"]),
        "candidates": serial_fn(captured["candidates"]),
        "selected_index": captured["selected_index"],
    }
    # With itself as the selected branch this checks every detection field and
    # every keypoint, including the eight refinable corners.
    assert_detection_preserved(baseline, replay, "R0 exact replay")
    for left, right in zip(baseline["candidates"], replay["candidates"]):
        if not _array_equal(left["keypoints_xy"], right["keypoints_xy"]):
            raise RuntimeError(f"R0 keypoints changed on replay: {baseline['id']}")


def infer_missing_rows(records: Sequence[Mapping], legacy: Mapping, *, extractor,
                       heads: Mapping[str, object], method_specs: Mapping[str, Mapping],
                       predict_fn: Callable, serial_fn: Callable, image_loader: Callable,
                       normalization: Mapping, dimensions: Sequence[float], order: int,
                       image_root: Path | str = C.ROOT,
                       verify_image: Callable[[Mapping, Path], None] | None = None,
                       progress: Callable[[int, int], None] | None = None) -> dict[str, list]:
    """Run one frozen R0 feature forward per image and only the six missing heads."""
    if set(heads) != set(NEW_METHODS) or set(method_specs) != set(NEW_METHODS):
        raise RuntimeError("Inference must contain exactly OLD_P and N3 seeds 1..3")
    if list(dimensions) != DIMENSIONS_WDH_M or order != 4:
        raise RuntimeError("Square registry/dimension contract drift")
    output = {name: [] for name in NEW_METHODS}
    image_root = Path(image_root)
    for index, record in enumerate(records):
        image_path = Path(record["image"]["path"])
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if verify_image is not None:
            verify_image(record["image"], image_path)
        image = image_loader(image_path)
        if image is None or list(image.shape[:2]) != list(record["original_hw"]):
            raise RuntimeError(f"Image decode/shape failure: {record['id']}")
        captured = extractor.predict(image)
        baseline = legacy["predictions"]["R0"][index]
        assert_captured_matches_r0(captured, baseline, serial_fn)
        for method in NEW_METHODS:
            spec = method_specs[method]
            predicted = predict_fn(
                heads[method], spec["arm"], captured, dimensions, order,
                spec["temperature"], spec["rule"], tuple(image.shape[:2]), normalization)
            prediction = predicted[0] if isinstance(predicted, tuple) else predicted
            row = {"id": record["id"], "raw_hw": list(image.shape[:2]), **prediction}
            assert_detection_preserved(baseline, row, method)
            output[method].append(row)
        if progress is not None:
            progress(index + 1, len(records))
    return output


def merge_predictions(legacy: Mapping, generated: Mapping[str, Sequence[Mapping]],
                      records: Sequence[Mapping]) -> dict[str, list]:
    if set(generated) != set(NEW_METHODS):
        raise RuntimeError("Generated method inventory is incomplete")
    merged = {name: copy.deepcopy(legacy["predictions"][name]) for name in REUSED_METHODS}
    merged.update({name: copy.deepcopy(list(generated[name])) for name in NEW_METHODS})
    merged = {name: merged[name] for name in METHODS}
    for method in METHODS:
        validate_prediction_rows(merged[method], records, method)
    for method in NEW_METHODS:
        for base, refined in zip(merged["R0"], merged[method]):
            assert_detection_preserved(base, refined, method)
    return merged


def validate_final_predictions(predictions: Mapping[str, Sequence[Mapping]],
                               records: Sequence[Mapping]) -> None:
    if set(predictions) != set(METHODS):
        raise RuntimeError("Square YOLO method inventory drift")
    for method in METHODS:
        validate_prediction_rows(predictions[method], records, method)
    for method in METHODS[1:]:
        for base, refined in zip(predictions["R0"], predictions[method]):
            assert_detection_preserved(base, refined, method)


def verify_payload_bindings(payload: Mapping) -> None:
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("Square YOLO source bindings are missing")
    for name in ("snapshot", "legacy_predictions", "legacy_metrics", "normalization",
                 "old_selection", "n3_selection", "R0_weights"):
        if name not in bindings:
            raise RuntimeError(f"Square YOLO source binding is missing: {name}")
        C.verify(bindings[name])
    checkpoints = bindings.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(NEW_METHODS):
        raise RuntimeError("Square YOLO checkpoint bindings are incomplete")
    for entry in checkpoints.values():
        C.verify(entry)


def _runtime_modules():
    directory = C.ROOT / "scripts/research/pallet_dim_conditioned_p_v1"
    sys.path.insert(0, str(directory))
    try:
        environment = importlib.import_module("dcp_env")
        inference = importlib.import_module("inference")
    finally:
        try:
            sys.path.remove(str(directory))
        except ValueError:
            pass
    expected = (directory / "inference.py").resolve()
    if Path(inference.__file__).resolve() != expected:
        raise RuntimeError(f"Wrong legacy inference module imported: {inference.__file__}")
    return environment, inference


def preflight(prediction_path: Path | str = LEGACY_RAW,
              metrics_path: Path | str = LEGACY_METRICS,
              snapshot_path: Path | str = DATASET) -> dict:
    payload, _, records = load_legacy(prediction_path, metrics_path, snapshot_path)
    contract = selection_contract(verify_checkpoints=True)
    if not R0_WEIGHTS.is_file() or C.sha256(R0_WEIGHTS) != R0_SHA256:
        raise RuntimeError("Frozen current YOLO R0 checkpoint binding failed")
    if not NORMALIZATION.is_file():
        raise RuntimeError("Dimension normalization lock is missing")
    return {
        "complete": True, "dataset": "GREEN0918_119", "frames": len(records),
        "dimensions_wdh_m": DIMENSIONS_WDH_M, "C4_order": 4,
        "reuse": list(REUSED_METHODS), "new_inference": list(NEW_METHODS),
        "legacy_predictions": C.binding(prediction_path),
        "legacy_metrics": C.binding(metrics_path), "snapshot": C.binding(snapshot_path),
        "R0": C.binding(R0_WEIGHTS), "normalization": C.binding(NORMALIZATION),
        "selections": {key: contract[key] for key in ("old_selection", "n3_selection")},
        "GT_input": payload["GT_input"], "square_reselection_or_tuning": False,
        "pose_3d": "x: no independent canonical pose reference",
    }


def infer(prediction_path: Path | str = LEGACY_RAW,
          metrics_path: Path | str = LEGACY_METRICS,
          snapshot_path: Path | str = DATASET,
          output_path: Path | str = PREDICTION_OUTPUT) -> dict:
    """Perform the minimum new GPU work: OLD_P and N3, three seeds each."""
    output_path = Path(output_path)
    legacy, snapshot, records = load_legacy(prediction_path, metrics_path, snapshot_path)
    contract = selection_contract(verify_checkpoints=True)
    if output_path.exists():
        existing = C.read(output_path)
        if (existing.get("complete") is not True or existing.get("GT_input") is not False or
                existing.get("annotation_input") is not False or
                existing.get("dimensions_input") != DIMENSIONS_WDH_M or
                existing.get("square_reselection_or_tuning") is not False):
            raise RuntimeError("Existing square YOLO output contract drift")
        verify_payload_bindings(existing)
        predictions = existing.get("predictions", {})
        validate_final_predictions(predictions, records)
        for method in REUSED_METHODS:
            if predictions[method] != legacy["predictions"][method]:
                raise RuntimeError(f"Existing output changed reused predictions: {method}")
        return existing

    start_gpu = C.gpu_snapshot(refuse_other_compute=True)
    environment, runtime = _runtime_modules()
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback would change the frozen protocol")
    dimensions, order = runtime.registry_input(OBJECT_TYPE)
    if dimensions.tolist() != DIMENSIONS_WDH_M or order != 4:
        raise RuntimeError("Square geometry registry drift")
    normalization = C.read(NORMALIZATION)
    heads = {}
    extractor = None
    try:
        for method in NEW_METHODS:
            spec = contract["methods"][method]
            head, path = runtime.load_head(spec["arm"], spec["seed"])
            if C.sha256(path) != spec["checkpoint_sha256"]:
                raise RuntimeError(f"Loaded checkpoint differs from selection lock: {method}")
            heads[method] = head
        extractor = environment.old("features").FrozenYoloFeatures(environment.R0)
        generated = infer_missing_rows(
            records, legacy, extractor=extractor, heads=heads,
            method_specs=contract["methods"], predict_fn=runtime.predict_captured,
            serial_fn=runtime.serial, image_loader=lambda path: cv2.imread(str(path)),
            normalization=normalization, dimensions=dimensions, order=order,
            verify_image=lambda binding, path: _verify_binding(binding, path),
            progress=lambda done, total: print(
                f"GREEN0918 OLD_P+N3 {done}/{total}", flush=True)
                if done == 1 or done % 20 == 0 or done == total else None)
        predictions = merge_predictions(legacy, generated, records)
    finally:
        if extractor is not None:
            extractor.close()
        heads.clear()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    end_gpu = C.gpu_snapshot(refuse_other_compute=True)
    output = {
        "schema": "pallet_n3_completion_v3_square_yolo_predictions_v1",
        "complete": True, "dataset_name": "GREEN0918_119",
        "dataset_status": "reused development 2D audit; not independent TEST",
        "frames": len(records), "GT_input": False, "annotation_input": False,
        "camera_input": False, "dimensions_input": DIMENSIONS_WDH_M,
        "object_type_input": OBJECT_TYPE, "C4_order": 4,
        "square_reselection_or_tuning": False,
        "reused_methods": list(REUSED_METHODS), "new_methods": list(NEW_METHODS),
        "new_R0_feature_forwards": len(records),
        "new_refiner_forwards": len(records) * len(NEW_METHODS),
        "bindings": {
            "snapshot": C.binding(snapshot_path),
            "legacy_predictions": C.binding(prediction_path),
            "legacy_metrics": C.binding(metrics_path),
            "normalization": C.binding(NORMALIZATION),
            "old_selection": contract["old_selection"],
            "n3_selection": contract["n3_selection"],
            "R0_weights": C.binding(environment.R0),
            "checkpoints": {name: C.binding(spec["checkpoint_path"])
                            for name, spec in contract["methods"].items()},
        },
        "predictions": predictions, "gpu_start": start_gpu, "gpu_end": end_gpu,
    }
    C.write(output_path, output, freeze=True)
    return output


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    intersection = np.maximum(np.minimum(left[2:], right[2:]) -
                              np.maximum(left[:2], right[:2]), 0).prod()
    union = (np.maximum(left[2:] - left[:2], 0).prod() +
             np.maximum(right[2:] - right[:2], 0).prod() - intersection)
    return float(intersection / max(union, 1e-12))


def _eval_math():
    path = C.ROOT / "scripts/research/pallet_dim_conditioned_p_v1/eval_math.py"
    return C.local_module("square_yolo_eval_math", path)


def _mean_tree(values: Sequence) -> object:
    """Arithmetic mean of like-shaped numeric leaves; never pool seed corners."""
    if all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in values):
        return float(np.mean(values))
    if all(isinstance(value, Mapping) for value in values):
        keys = set.intersection(*(set(value) for value in values))
        result = {}
        for key in sorted(keys):
            child = [value[key] for value in values]
            if all(item is not None for item in child):
                mean = _mean_tree(child)
                if mean is not None:
                    result[key] = mean
        return result
    return None


def _family(summary: Mapping[str, Mapping], prefix: str) -> dict:
    seeds = {str(seed): summary[f"{prefix}_seed{seed}"] for seed in SEEDS}
    return {
        "seeds": seeds,
        "mean_of_seed_summaries": _mean_tree(list(seeds.values())),
        "mean_definition": "arithmetic mean of three seed summaries; medians/P90 are not pooled across seeds",
    }


def _summary_delta(before: Mapping, after: Mapping) -> dict:
    fields = ("E_sym", "E_fixed", "matched_pooled_corner8_median_px",
              "matched_pooled_corner8_P90_px", "full_penalty_median_px",
              "full_penalty_P90_px", "gross20", "coverage")
    output = {}
    for field in fields:
        left, right = before.get(field), after.get(field)
        output[field] = None if left is None or right is None else float(right - left)
    for threshold in ("5", "10", "20"):
        left, right = before.get("PCK", {}).get(threshold), after.get("PCK", {}).get(threshold)
        output[f"PCK{threshold}"] = None if left is None or right is None else float(right - left)
    return output


def _comparison(before_name: str, after_name: str, rows: Mapping, summaries: Mapping,
                evaluator) -> dict:
    return {
        "before": before_name, "after": after_name,
        "delta_after_minus_before": _summary_delta(summaries[before_name], summaries[after_name]),
        "paired_frames": evaluator.damage(rows[before_name], rows[after_name]),
    }


def _comparisons(rows: Mapping, summaries: Mapping, evaluator) -> dict:
    definitions = {
        "R0_to_OLD_P": lambda seed: ("R0", f"OLD_P_seed{seed}"),
        "R0_to_N2_DIM_ONLY": lambda seed: ("R0", f"N2_DIM_ONLY_seed{seed}"),
        "R0_to_N3_DIM_SYM": lambda seed: ("R0", f"N3_DIM_SYM_seed{seed}"),
        "OLD_P_to_N3_DIM_SYM": lambda seed: (f"OLD_P_seed{seed}", f"N3_DIM_SYM_seed{seed}"),
        "N2_DIM_ONLY_to_N3_DIM_SYM": lambda seed: (f"N2_DIM_ONLY_seed{seed}", f"N3_DIM_SYM_seed{seed}"),
    }
    output = {}
    for name, pair in definitions.items():
        seeds = {}
        for seed in SEEDS:
            before, after = pair(seed)
            seeds[str(seed)] = _comparison(before, after, rows, summaries, evaluator)
        output[name] = {
            "seeds": seeds,
            "mean_of_seed_numeric_fields": _mean_tree(list(seeds.values())),
            "direction": "after minus before; negative error delta and positive PCK delta favor after",
        }
    return output


def evaluate_predictions(predictions: Mapping[str, Sequence[Mapping]],
                         truth_modes: Mapping[str, Sequence[Mapping]]) -> dict:
    """Pure exact C4 8-corner evaluation for tiny fixtures or GREEN0918."""
    if set(predictions) != set(METHODS):
        raise RuntimeError("Square YOLO evaluation method inventory drift")
    if set(truth_modes) != {"manual_declared", "manual_in_frame"}:
        raise RuntimeError("Both fixed manual denominators are required")
    evaluator = _eval_math()
    modes = {}
    for mode, truth in truth_modes.items():
        records = [{"id": row["id"], "original_hw": row["hw"]} for row in truth]
        by_id = {row["id"]: row for row in truth}
        if len(by_id) != len(truth):
            raise RuntimeError(f"Duplicate truth IDs in {mode}")
        scored = {}
        summaries = {}
        for method in METHODS:
            validate_prediction_rows(predictions[method], records, method)
            method_rows = []
            for prediction in predictions[method]:
                target = by_id[prediction["id"]]
                selected = prediction["selected_index"]
                candidate = None if selected is None else prediction["candidates"][selected]
                matched = candidate is not None and iou(candidate["box_xyxy"], target["box"]) >= .5
                points = (np.full((9, 2), np.nan) if candidate is None else
                          np.asarray(candidate["keypoints_xy"], dtype=np.float64))
                measured = evaluator.measure(
                    points, target["gt"], target["valid"], target["permutations"],
                    target["hw"], matched, selected is not None)
                method_rows.append({"id": target["id"],
                                    "session": target.get("session", "unclassified"), **measured})
            scored[method] = method_rows
            summaries[method] = evaluator.summary(method_rows)
        denominator = int(sum(np.asarray(row["valid"], dtype=bool)[:8].sum() for row in truth))
        modes[mode] = {
            "frames": len(truth), "manual_corner_denominator": denominator,
            "rows": scored, "summary": summaries,
            "families": {
                "R0": {"single": summaries["R0"]},
                "OLD_P": _family(summaries, "OLD_P"),
                "N2_DIM_ONLY": _family(summaries, "N2_DIM_ONLY"),
                "N3_DIM_SYM": _family(summaries, "N3_DIM_SYM"),
            },
            "comparisons": _comparisons(scored, summaries, evaluator),
        }
    return {
        "schema": "pallet_n3_completion_v3_square_yolo_evaluation_v1",
        "complete": True, "dataset": "GREEN0918_119",
        "dataset_status": "reused development 2D audit; not independent TEST",
        "dimensions_wdh_m": DIMENSIONS_WDH_M, "dimension_effect_identifiable": False,
        "dimension_effect_note": (
            "All frames have one fixed W,D,H vector; whole-model transfer is measured, "
            "but the causal benefit of dimension variation is not identifiable."),
        "symmetry": "exact approved whole-object C4 permutations; center fixed",
        "matching": "selected detector candidate; IoU >= 0.5 against all-known in-frame point box",
        "square_reselection_or_tuning": False,
        "confidence_interval": "x: one correlated capture session; not estimated",
        "pose_3d": {
            "translation": {"value": None, "display": "x",
                            "status": "BLOCKED_NO_INDEPENDENT_CANONICAL_POSE_REFERENCE"},
            "rotation": {"value": None, "display": "x",
                         "status": "BLOCKED_NO_INDEPENDENT_CANONICAL_POSE_REFERENCE"},
        },
        "modes": modes,
    }


def score(prediction_path: Path | str = PREDICTION_OUTPUT,
          metrics_path: Path | str = METRICS_OUTPUT,
          result_path: Path | str = RESULT_OUTPUT) -> dict:
    from . import square

    prediction_path, metrics_path, result_path = map(Path,
        (prediction_path, metrics_path, result_path))
    payload = C.read(prediction_path)
    if (payload.get("complete") is not True or payload.get("GT_input") is not False or
            payload.get("annotation_input") is not False or
            payload.get("dimensions_input") != DIMENSIONS_WDH_M or
            payload.get("square_reselection_or_tuning") is not False):
        raise RuntimeError("Square YOLO predictions are incomplete or not GT-free")
    verify_payload_bindings(payload)
    truth_modes = {
        "manual_declared": square.truth_rows(in_frame_only=False),
        "manual_in_frame": square.truth_rows(in_frame_only=True),
    }
    truth_records = [{"id": row["id"], "original_hw": row["hw"]}
                     for row in truth_modes["manual_declared"]]
    validate_final_predictions(payload["predictions"], truth_records)
    evaluation = evaluate_predictions(payload["predictions"], truth_modes)
    evaluation["predictions"] = C.binding(prediction_path)
    evaluation["snapshot"] = C.binding(DATASET)
    evaluation["symmetry_contract"] = C.binding(SYMMETRY)
    C.write(metrics_path, evaluation, freeze=True)

    result = {key: copy.deepcopy(value) for key, value in evaluation.items() if key != "modes"}
    result["metrics"] = C.binding(metrics_path)
    result["modes"] = {}
    for mode, value in evaluation["modes"].items():
        result["modes"][mode] = {key: copy.deepcopy(item) for key, item in value.items()
                                  if key != "rows"}
    C.write(result_path, result, freeze=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "infer", "score", "all"))
    parser.add_argument("--legacy-predictions", type=Path, default=LEGACY_RAW)
    parser.add_argument("--legacy-metrics", type=Path, default=LEGACY_METRICS)
    parser.add_argument("--snapshot", type=Path, default=DATASET)
    parser.add_argument("--predictions-output", type=Path, default=PREDICTION_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_OUTPUT)
    parser.add_argument("--result-output", type=Path, default=RESULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.stage == "preflight":
        result = preflight(arguments.legacy_predictions, arguments.legacy_metrics, arguments.snapshot)
    elif arguments.stage == "infer":
        result = infer(arguments.legacy_predictions, arguments.legacy_metrics,
                       arguments.snapshot, arguments.predictions_output)
    elif arguments.stage == "score":
        result = score(arguments.predictions_output, arguments.metrics_output,
                       arguments.result_output)
    else:
        preflight(arguments.legacy_predictions, arguments.legacy_metrics, arguments.snapshot)
        infer(arguments.legacy_predictions, arguments.legacy_metrics,
              arguments.snapshot, arguments.predictions_output)
        result = score(arguments.predictions_output, arguments.metrics_output,
                       arguments.result_output)
    print(json.dumps({key: result.get(key) for key in
                      ("complete", "dataset", "dataset_name", "frames", "reuse", "new_inference")
                      if key in result}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
