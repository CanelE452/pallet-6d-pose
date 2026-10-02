"""Unified same-frame runtime benchmark for YOLO, DOPE and ResNet-18.

Production measurement requires every accuracy artifact to be complete.  All
dependency, image-byte, rule and cached-output checks occur before this module
creates a protocol or raw-output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from . import common as C


YOLO_RUNTIME_PROTOCOL = C.ROOT / "_docs/experiments/pallet_sensors_submission_v1/RUNTIME_PROTOCOL.json"
YOLO_RUNTIME_AMENDMENT = C.ROOT / "_docs/experiments/pallet_sensors_submission_v1/RUNTIME_NUMERIC_AMENDMENT.json"
YOLO_RESULTS = C.ROOT / "_docs/experiments/pallet_sensors_submission_v1/UNIFIED_DEV_RESULTS.json"
YOLO_SELECTION = C.ROOT / "_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/P_SELECTION.json"
YOLO_IMPLEMENTATION_LOCK = C.ROOT / "_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/B_IMPLEMENTATION_LOCK.json"
YOLO_LINE_BINDING = C.ROOT / "_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/LINE_SOURCE_BINDING.json"
YOLO_SELECTION_CODE = C.ROOT / "scripts/research/pallet_final_ml_contribution_test_v1/select_point.py"
YOLO_LINE_RUNTIME = C.ROOT / "data/pallet/results/pallet_line_pose_v1/RUNTIME.json"
YOLO_BASE_CACHE = C.ROOT / "data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json"
YOLO_P1_PREDICTIONS = C.ROOT / "data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/evaluation/P1/IMAGE_PREDICTIONS.json"
YOLO_POSE = C.ROOT / "data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/evaluation/P1/POSE_PER_FRAME_BY_ARM.json"
YOLO_P1_CHECKPOINT = C.ROOT / "data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/runs/seed1/last.pt"
YOLO_R0_CHECKPOINT = (C.ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs"
                      / "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")

DOPE_DOC = C.ROOT / "_docs/experiments/pallet_dope_refiner_20261001_v1"
DOPE_RAW = C.ROOT / "data/pallet/results/pallet_dope_refiner_20261001_v1"

RESNET_DOC = C.ROOT / "_docs/experiments/pallet_resnet18_dim_refiner_20261002_v1"
RESNET_RAW = C.ROOT / "data/pallet/results/pallet_resnet18_dim_refiner_20261002_v1"

POSE_RAW = C.ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
POSE_CODE = C.ROOT / "scripts/paper/pose_metric_closure_v1"
POSE_CONTRACT = POSE_RAW / "POSE_EVAL_OBJECT_CONTRACT.json"
POSE_GT = POSE_RAW / "GEOMETRY_RESOLVED_POSE_GT.json"
POSE_EXECUTED_CODE = (
    POSE_CODE / "run_pose_evaluation.py",
    POSE_CODE / "pose_evaluation_paths.py",
    POSE_CODE / "symmetry_aware_pose_metrics.py",
    C.ROOT / "challenge/evaluation_v2/pnp_selector.py",
    C.ROOT / "challenge/evaluation_v2/oriented_iou3d.py",
)

HISTORICAL_RUNTIME_PROTOCOL = {
    "path": "_docs/experiments/pallet_sensors_submission_v1/RUNTIME_PROTOCOL.json",
    "sha256": "00f5c0434a50f72efdc4db4983681600c42718a0092b58ae470c019cc14336f7",
    "bytes": 3770,
}


def _same_binding(actual: dict, expected_path: Path) -> None:
    path = C.verify_binding(actual)
    if path != expected_path.resolve():
        raise ValueError(f"Receipt binds the wrong artifact: {path} != {expected_path}")


def _require_paths(paths) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError("Runtime dependencies/results are incomplete; no outputs created: "
                           + ", ".join(missing))


def _validated_resnet_dimensions(canonical_wdh, pose_dimensions, label: str) -> np.ndarray:
    """Keep registry W,D order for ResNet while auditing pose long/short dimensions."""
    canonical = np.asarray(canonical_wdh, np.float64)
    pose = np.asarray([
        pose_dimensions["long"], pose_dimensions["short"], pose_dimensions["height"]
    ], np.float64)
    if (canonical.shape != (3,) or pose.shape != (3,)
            or not np.isfinite(canonical).all() or not np.isfinite(pose).all()
            or not (canonical > 0).all() or not (pose > 0).all()):
        raise ValueError(f"Invalid ResNet/pose dimensions: {label}")
    # W,D are registry axes and need not be [long,short] in that order.
    if (not np.array_equal(np.sort(canonical[:2]), np.sort(pose[:2]))
            or canonical[2] != pose[2]):
        raise ValueError(f"ResNet W,D,H differs from canonical pose dimensions: {label}")
    return canonical


def _verify_hash_manifest(payload: dict, label: str) -> list[dict]:
    """Verify every frozen path/hash entry and return protocol bindings."""
    paths = payload.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError(f"{label} has no path/hash manifest")
    bindings = []
    for relative, expected in sorted(paths.items()):
        path = C.ROOT / relative
        if not path.is_file() or C.sha256(path) != expected:
            raise ValueError(f"{label} binding mismatch: {relative}")
        bindings.append(C.binding(path))
    return bindings


def _verify_dope() -> dict:
    protocol = C.read_json(DOPE_DOC / "PROTOCOL.json")
    if protocol.get("schema") != "dope_local_refiner_protocol_v1":
        raise ValueError("Wrong DOPE protocol")
    for value in protocol.get("bindings", []):
        C.verify_binding(value)
    training = C.require_complete(DOPE_DOC / "TRAIN_SEED1.json")
    if training.get("seed") != 1 or training.get("updates_per_arm") != 6000:
        raise ValueError("DOPE seed-1 training receipt drift")
    C.verify_binding(training["checkpoint"])
    selection = C.require_complete(DOPE_DOC / "SELECTION.json")
    if (selection.get("real_selection") is not False or set(selection.get("rules", {})) != {"P", "D"}
            or "P1" not in selection.get("temperatures", {})):
        raise ValueError("DOPE source-only selection drift")
    _same_binding(selection["protocol"], DOPE_DOC / "PROTOCOL.json")
    inference = C.require_complete(DOPE_DOC / "DEV_INFERENCE_COMPLETE.json")
    _same_binding(inference["predictions"], DOPE_RAW / "DEV_PREDICTIONS.json")
    predictions = C.require_complete(DOPE_RAW / "DEV_PREDICTIONS.json",
                                     schema="dope_refiner_dev_predictions_v1")
    if len(predictions.get("frames", [])) != 319:
        raise ValueError("DOPE DEV prediction provenance drift")
    _same_binding(predictions["selection"], DOPE_DOC / "SELECTION.json")
    results = C.require_complete(DOPE_DOC / "DEV_RESULTS.json",
                                 schema="dope_refiner_dev_results_v1")
    _same_binding(results["predictions"], DOPE_RAW / "DEV_PREDICTIONS.json")
    lock = C.require_complete(DOPE_DOC / "DEV_EVALUATION_LOCK.json")
    _same_binding(lock["predictions"], DOPE_RAW / "DEV_PREDICTIONS.json")
    poses = {}
    for method in ("DOPE", "P1"):
        path = C.verify_binding(results["pose_artifacts"][method])
        pose = C.require_complete(path)
        if pose.get("arm") != method or len(pose.get("all_frame_rows", [])) != 319:
            raise ValueError(f"DOPE pose artifact drift: {method}")
        poses[method] = pose
    return dict(protocol=protocol, training=training, selection=selection,
                predictions=predictions, results=results, lock=lock, poses=poses)


def _verify_resnet() -> dict:
    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1 import common as R
    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.inference import validate_prediction_payload
    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.selection import validate_selection_payload

    protocol = R.verify_protocol()
    training = C.require_complete(RESNET_DOC / "TRAIN_SEED1.json")
    if (training.get("seed") != 1 or training.get("updates_per_arm") != 6000
            or training.get("arms") != list(R.HEAD_ARMS)):
        raise ValueError("ResNet seed-1 training receipt drift")
    checkpoint = C.verify_binding(training["checkpoint"])
    if checkpoint != (RESNET_RAW / "runs/seed1/paired_last.pt").resolve():
        raise ValueError("ResNet seed-1 checkpoint path drift")
    selection = validate_selection_payload(C.require_complete(RESNET_DOC / "SELECTION.json"),
                                           protocol=protocol)
    inference = C.require_complete(RESNET_DOC / "DEV_INFERENCE_COMPLETE.json")
    _same_binding(inference["predictions"], RESNET_RAW / "DEV_PREDICTIONS.json")
    predictions = validate_prediction_payload(
        C.require_complete(RESNET_RAW / "DEV_PREDICTIONS.json",
                           schema="resnet18_full_dim_refiner_dev_predictions_v1"),
        protocol=protocol)
    results = C.require_complete(RESNET_DOC / "DEV_RESULTS.json",
                                 schema="resnet18_full_dim_refiner_dev_results_v1")
    _same_binding(results["predictions"], RESNET_RAW / "DEV_PREDICTIONS.json")
    lock = C.require_complete(RESNET_DOC / "DEV_EVALUATION_LOCK.json")
    _same_binding(lock["predictions"], RESNET_RAW / "DEV_PREDICTIONS.json")
    poses = {}
    for method in ("FULL", "P0_S1", "D0_S1", "P5_CONSTANT_S1", "P5_S1"):
        path = C.verify_binding(results["pose_artifacts"][method])
        pose = C.require_complete(path)
        if pose.get("method") != method or len(pose.get("all_frame_rows", [])) != 319:
            raise ValueError(f"ResNet pose artifact drift: {method}")
        poses[method] = pose
    return dict(protocol=protocol, training=training, selection=selection,
                predictions=predictions, results=results, lock=lock, poses=poses)


def _verify_yolo() -> dict:
    amendment = C.read_json(YOLO_RUNTIME_AMENDMENT)
    if amendment.get("original_protocol") != HISTORICAL_RUNTIME_PROTOCOL:
        raise ValueError("Historical 26-frame runtime protocol binding drift")
    _same_binding(amendment["original_protocol"], YOLO_RUNTIME_PROTOCOL)
    runtime = C.read_json(YOLO_RUNTIME_PROTOCOL)
    if (len(runtime.get("keys", [])) != C.FRAMES or runtime.get("warmup") != C.WARMUP
            or runtime.get("repeats") != C.REPEATS or runtime.get("batch") != C.BATCH):
        raise ValueError("Historical 26-frame runtime contract drift")
    results = C.require_complete(YOLO_RESULTS)
    if not {"R0", "P1"}.issubset(results.get("methods", {})):
        raise ValueError("YOLO R0/P1 accuracy results incomplete")
    selection = C.require_complete(YOLO_SELECTION)
    if (selection.get("no_real_selection") is not True
            or selection.get("selection_population") != "synth_val"):
        raise ValueError("YOLO P1 selection is not source-only")
    yolo_rule = selection.get("selected_rule", {})
    yolo_temperature = selection.get("temperatures", {}).get("1", {}).get("temperature")
    if (set(("lam", "max_move_image_diagonal_fraction")) - set(yolo_rule)
            or not np.isfinite(yolo_rule.get("lam", np.nan))
            or not np.isfinite(yolo_temperature) or yolo_temperature <= 0):
        raise ValueError("YOLO P1 selected lambda/cap/temperature is incomplete")
    if C.sha256(YOLO_P1_CHECKPOINT) != selection["checkpoints"]["1"]:
        raise ValueError("YOLO P1 checkpoint binding drift")
    if selection.get("selection_code_sha256") != C.sha256(YOLO_SELECTION_CODE):
        raise ValueError("YOLO P1 source-only selector code drift")
    implementation_bindings = _verify_hash_manifest(
        C.read_json(YOLO_IMPLEMENTATION_LOCK), "YOLO P1 implementation lock")
    line_bindings = _verify_hash_manifest(
        C.read_json(YOLO_LINE_BINDING), "YOLO baseline source binding")
    line_runtime = C.require_complete(YOLO_LINE_RUNTIME)
    if (line_runtime.get("PASS") is not True or line_runtime.get("parity_PASS") is not True
            or line_runtime.get("runtime", {}).get("cudnn_allow_tf32") is not True
            or line_runtime.get("runtime", {}).get("matmul_allow_tf32") is not False):
        raise ValueError("YOLO source CUDA numerical contract drift")
    base = C.require_complete(YOLO_BASE_CACHE)
    if base.get("weights_sha256") != C.sha256(YOLO_R0_CHECKPOINT):
        raise ValueError("YOLO R0 checkpoint/cache drift")
    p1 = C.require_complete(YOLO_P1_PREDICTIONS)
    if p1.get("selection_sha256") != C.sha256(YOLO_SELECTION):
        raise ValueError("YOLO P1 prediction/selection drift")
    pose = C.read_json(YOLO_POSE)
    if (pose.get("all_reproduced_exactly") is not True
            or not {"R0", "P1"}.issubset(pose.get("per_frame", {}))
            or any(len(pose["per_frame"][arm]) != 319 for arm in ("R0", "P1"))):
        raise ValueError("YOLO canonical pose parity artifact incomplete")
    return dict(runtime=runtime, amendment=amendment, results=results, selection=selection,
                base=base, p1=p1, pose=pose, line_runtime=line_runtime,
                source_bindings=implementation_bindings + line_bindings
                    + [C.binding(YOLO_SELECTION_CODE)])


def preflight() -> dict:
    """Validate everything needed for timing without creating any output."""
    required = [
        YOLO_RUNTIME_PROTOCOL, YOLO_RUNTIME_AMENDMENT, YOLO_RESULTS, YOLO_SELECTION,
        YOLO_IMPLEMENTATION_LOCK, YOLO_LINE_BINDING, YOLO_SELECTION_CODE,
        YOLO_LINE_RUNTIME, YOLO_BASE_CACHE,
        YOLO_P1_PREDICTIONS, YOLO_POSE, YOLO_P1_CHECKPOINT, YOLO_R0_CHECKPOINT,
        DOPE_DOC / "PROTOCOL.json", DOPE_DOC / "TRAIN_SEED1.json",
        DOPE_DOC / "SELECTION.json", DOPE_DOC / "DEV_INFERENCE_COMPLETE.json",
        DOPE_DOC / "DEV_RESULTS.json", DOPE_DOC / "DEV_EVALUATION_LOCK.json",
        DOPE_RAW / "DEV_PREDICTIONS.json",
        RESNET_DOC / "PROTOCOL.json", RESNET_DOC / "TRAIN_SEED1.json",
        RESNET_DOC / "SELECTION.json", RESNET_DOC / "DEV_INFERENCE_COMPLETE.json",
        RESNET_DOC / "DEV_RESULTS.json", RESNET_DOC / "DEV_EVALUATION_LOCK.json",
        RESNET_RAW / "DEV_PREDICTIONS.json", POSE_CONTRACT, POSE_GT,
        *POSE_EXECUTED_CODE,
    ]
    _require_paths(required)
    yolo = _verify_yolo()
    dope = _verify_dope()
    resnet = _verify_resnet()

    keys = list(yolo["runtime"]["keys"])
    if len(set(keys)) != C.FRAMES:
        raise ValueError("The runtime frame list contains duplicates")
    yolo_p1 = {row["image_key"]: row["prediction"] for row in yolo["p1"]["records"]}
    dope_rows = {row["image_key"]: row for row in dope["predictions"]["frames"]}
    resnet_rows = {row["image_key"]: row for row in resnet["predictions"]["frames"]}
    dope_map = {row["image_key"]: row for row in dope["lock"]["frame_mapping"]}
    resnet_map = {row["image_key"]: row for row in resnet["lock"]["frame_mapping"]}
    if not all(set(keys).issubset(source) for source in
               (yolo["base"]["frames"], yolo_p1, dope_rows, resnet_rows, dope_map, resnet_map)):
        raise ValueError("The exact historical 26 frames are not shared by all three estimators")

    yolo_pose = {arm: {row["frame_id"]: row for row in yolo["pose"]["per_frame"][arm]}
                 for arm in ("R0", "P1")}
    dope_pose = {method: {row["image_key"]: row for row in dope["poses"][method]["all_frame_rows"]}
                 for method in ("DOPE", "P1")}
    resnet_pose = {method: {row["image_key"]: row for row in resnet["poses"][method]["all_frame_rows"]}
                   for method in ("FULL", "P0_S1", "D0_S1", "P5_CONSTANT_S1", "P5_S1")}
    gt_payload = C.read_json(POSE_GT)
    targets = gt_payload["frames"]

    from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.inference import (
        canonical_dimensions,
    )

    frames = []
    sessions = Counter()
    for key in keys:
        path = C.ROOT / key
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        metadata = dope_map[key]
        rmeta = resnet_map[key]
        if any(metadata[field] != rmeta[field] for field in
               ("frame_id", "pose_frame_id", "session_id", "image_key", "object_type",
                "camera_intrinsics", "dimensions_m")):
            raise ValueError(f"DOPE/ResNet evaluation mapping differs: {key}")
        if (digest != yolo["base"]["frame_metadata"][key]["image_sha256"]
                or digest != dope_rows[key]["image_sha256"]
                or digest != resnet_rows[key]["image_sha256"]):
            raise ValueError(f"Shared frame byte hash differs: {key}")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[:2]) != metadata["original_hw"]:
            raise ValueError(f"Cannot decode exact runtime image: {key}")
        dims = metadata["dimensions_m"]
        dimensions = _validated_resnet_dimensions(
            resnet_rows[key]["canonical_WDH_m"], dims, key)
        registered = canonical_dimensions(metadata["object_type"])
        if not np.array_equal(dimensions, registered):
            raise ValueError(f"ResNet W,D,H differs from the bound object registry: {key}")
        pose_id = metadata["pose_frame_id"]
        if pose_id not in targets or pose_id not in yolo_pose["R0"] or pose_id not in yolo_pose["P1"]:
            raise ValueError(f"Pose parity target absent: {pose_id}")
        frame_pose = {
            "YOLO_R0": yolo_pose["R0"][pose_id],
            "YOLO_P1": yolo_pose["P1"][pose_id],
            "DOPE_BASE": dope_pose["DOPE"][key],
            "DOPE_P1": dope_pose["P1"][key],
            "RESNET_FULL": resnet_pose["FULL"][key],
            "RESNET_P0_S1": resnet_pose["P0_S1"][key],
            "RESNET_D0_S1": resnet_pose["D0_S1"][key],
            "RESNET_P5_CONSTANT_S1": resnet_pose["P5_CONSTANT_S1"][key],
            "RESNET_P5_S1": resnet_pose["P5_S1"][key],
        }
        frames.append(dict(
            key=key, image=image, image_binding=C.binding(path),
            frame_id=metadata["frame_id"], pose_frame_id=pose_id,
            session_id=metadata["session_id"], object_type=metadata["object_type"],
            camera=np.asarray(metadata["camera_intrinsics"], np.float64),
            dimensions=dimensions,
            spec=dict(long_m=float(dims["long"]), short_m=float(dims["short"]),
                      height_m=float(dims["height"])),
            target=targets[pose_id], pose_reference=frame_pose,
            output_reference=dict(yolo_r0=yolo["base"]["frames"][key], yolo_p1=yolo_p1[key],
                                  dope=dope_rows[key], resnet=resnet_rows[key]),
        ))
        sessions[metadata["session_id"]] += 1
    if len(sessions) != 13 or set(sessions.values()) != {2}:
        raise ValueError(f"Expected exactly two frames from each of thirteen sessions: {sessions}")

    dependencies = [C.binding(path) for path in required]
    dependencies += [
        C.binding(C.verify_binding(dope["training"]["checkpoint"])),
        C.binding(C.verify_binding(resnet["training"]["checkpoint"])),
        C.binding(C.verify_binding(resnet["protocol"]["baseline"])),
        C.binding(DOPE_RAW / "evaluation/POSE_EVALUATION_DOPE.json"),
        C.binding(DOPE_RAW / "evaluation/POSE_EVALUATION_P1.json"),
        *[C.binding(C.verify_binding(resnet["results"]["pose_artifacts"][method]))
          for method in ("FULL", "P0_S1", "D0_S1", "P5_CONSTANT_S1", "P5_S1")],
        *[C.binding(C.verify_binding(value)) for value in dope["protocol"]["bindings"]],
        *[C.binding(C.verify_binding(value)) for value in resnet["protocol"]["bindings"]],
        *yolo["source_bindings"],
    ]
    unique = {value["path"]: value for value in dependencies}
    return dict(frames=frames, yolo=yolo, dope=dope, resnet=resnet,
                dependencies=[unique[key] for key in sorted(unique)],
                pose_contract=C.binding(POSE_CONTRACT), pose_gt=C.binding(POSE_GT),
                GT_used_for_timing=False, GT_used_for_post_timing_parity=True)


def canonical_modules():
    for path in (C.ROOT, POSE_CODE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import importlib
    return (importlib.import_module("run_pose_evaluation"),
            importlib.import_module("pose_evaluation_paths"))


def solve_pose(points, has_detection, camera, spec, modules):
    """Canonical MAIN selector plus both SQPnP/RefineLM fits; no GT input."""
    M, G = modules
    if not has_detection:
        return dict(status="NO_DETECTION")
    value = np.asarray(points, np.float64)
    if value.shape != (9, 2) or np.isinf(value).any():
        raise ValueError("Invalid predicted points")
    valid = np.isfinite(value).all(-1)
    usable = valid[:8]
    if usable.sum() < 6:
        return dict(status="FEWER_THAN_SIX_FINITE_CORNERS")
    chosen = None
    try:
        selection = G.predict_pose_without_gt(value, camera, spec["long_m"], spec["short_m"],
                                              spec["height_m"])["selector_result"]
        for hypothesis in selection.hypotheses:
            if hypothesis.name == selection.selected_hypothesis and hypothesis.success:
                dims = hypothesis.camera_facing_dimensions.as_dict()
                chosen = (M.CF_WIDTH if abs(float(dims["width"]) - spec["long_m"]) < 1e-6
                          else M.CF_DEPTH)
    except Exception as exc:
        return dict(status="CANONICAL_SELECTOR_EXCEPTION",
                    reason=type(exc).__name__ + ": " + str(exc))
    models = {
        M.CF_WIDTH: M.cuboid(spec["long_m"], spec["height_m"], spec["short_m"]),
        M.CF_DEPTH: M.cuboid(spec["short_m"], spec["height_m"], spec["long_m"]),
    }
    try:
        fits = {key: M.solve(model, value[:8], camera, usable) for key, model in models.items()}
    except cv2.error as exc:
        return dict(status="CANONICAL_PNP_FAILED", reason=str(exc))
    if chosen is None or any(result is None for result in fits.values()):
        return dict(status="CANONICAL_MAIN_UNAVAILABLE")
    rotation, translation, residual = fits[chosen]
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all()
            and np.isfinite(residual)):
        raise FloatingPointError("Nonfinite canonical pose")
    return dict(status="OK", chosen=chosen, rotation=rotation, translation=translation,
                reprojection_mean_px=float(residual))


def _compare_array(actual, expected, label, atol=0.) -> float:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(f"{label}: missing-value mismatch")
        return 0.
    left, right = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if left.shape != right.shape or not np.array_equal(np.isnan(left), np.isnan(right)):
        raise AssertionError(f"{label}: shape/missing mask mismatch")
    if atol == 0. and (not np.array_equal(left, right, equal_nan=True)
                      or not np.array_equal(np.signbit(left), np.signbit(right))):
        raise AssertionError(f"{label}: exact value/sign mismatch")
    mask = np.isfinite(right)
    error = float(np.max(np.abs(left[mask] - right[mask]))) if mask.any() else 0.
    if error > atol:
        raise AssertionError(f"{label}: max abs {error} > {atol}")
    return error


def _compare_tree(actual, expected, label="root") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{label}: mapping keys differ")
        return max((_compare_tree(actual[key], expected[key], f"{label}.{key}")
                    for key in expected), default=0.)
    if isinstance(expected, list):
        try:
            array = np.asarray(expected, np.float64)
            if array.dtype != object:
                return _compare_array(actual, expected, label, 0.)
        except (TypeError, ValueError):
            pass
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(f"{label}: sequence differs")
        return max((_compare_tree(left, right, f"{label}[{index}]")
                    for index, (left, right) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, (int, float, bool)) or expected is None:
        if isinstance(expected, bool) or expected is None:
            if actual != expected:
                raise AssertionError(f"{label}: {actual!r} != {expected!r}")
            return 0.
        return _compare_array(actual, expected, label, 0.)
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    return 0.


def check_2d_parity(arm: str, prediction: dict, frame: dict,
                    point_atol: float = C.POINT_ATOL) -> float:
    reference = frame["output_reference"]
    native = prediction["native"]
    errors = []
    if arm == "YOLO_R0":
        actual, expected = native["candidates"], reference["yolo_r0"]
        if len(actual) != len(expected):
            raise AssertionError("YOLO R0 candidate count changed")
        for index, (left, right) in enumerate(zip(actual, expected)):
            errors += [_compare_array(left["score"], right["score"], f"R0/{index}/score"),
                       _compare_array(left["box_xyxy"], right["box_xyxy"], f"R0/{index}/box"),
                       _compare_array(left["keypoints_xy"], right["keypoints_xy"], f"R0/{index}/points")]
    elif arm == "YOLO_P1":
        errors.append(_compare_tree(native, reference["yolo_p1"], "YOLO_P1"))
    elif arm in ("DOPE_BASE", "DOPE_P1"):
        row = reference["dope"]; base = native["base"]
        errors += [
            _compare_array(base["points_original"], row["base_points"], "DOPE/base", point_atol),
            _compare_array(base["valid"], row["point_valid"], "DOPE/valid", 0.),
            _compare_array(base["bbox_original"], row["box_original"], "DOPE/box", point_atol),
            _compare_array(base["score"], row["score"], "DOPE/score", point_atol),
            _compare_array(base["confidence"], row["confidence"], "DOPE/confidence", point_atol),
            _compare_array(base["affine_input_to_net"], row["affine_input_to_net"], "DOPE/affine", 0.),
        ]
        expected = row["base_points"] if arm == "DOPE_BASE" else row["refined"]["P1"]
        errors.append(_compare_array(prediction["points"], expected, arm, point_atol))
    else:
        row = reference["resnet"]; base = native["base"]
        errors += [
            _compare_array(base["points_original"], row["base_points"], "RESNET/base", point_atol),
            _compare_array(base["valid"], row["point_valid"], "RESNET/valid", 0.),
            _compare_array(base["bbox_original"], row["box_original"], "RESNET/box", point_atol),
            _compare_array(base["score"], row["score"], "RESNET/score", point_atol),
            _compare_array(base["confidence"], row["confidence"], "RESNET/confidence", point_atol),
            _compare_array(base["affine_input_to_net"], row["affine_input_to_net"], "RESNET/affine", 0.),
        ]
        method = arm.removeprefix("RESNET_")
        expected = row["base_points"] if method == "FULL" else row["refined"][method]
        errors.append(_compare_array(prediction["points"], expected, arm, point_atol))
    return max(errors, default=0.)


def check_pose_parity(arm: str, pose: dict, frame: dict, modules,
                      pose_atol: float = C.POSE_ATOL) -> float:
    M, G = modules
    expected = frame["pose_reference"][arm]
    expected_status = expected.get("status", "OK")
    if pose["status"] != expected_status:
        raise AssertionError(f"{arm}/{frame['frame_id']}: pose status {pose['status']} != {expected_status}")
    if pose["status"] != "OK":
        return 0.
    target = frame["target"]
    expected_axis = expected.get("selected_axis")
    if expected_axis is None:
        truth_axis = target["physical_long_axis"]
        expected_axis = truth_axis if expected["axis_correct"] else (
            M.CF_DEPTH if truth_axis == M.CF_WIDTH else M.CF_WIDTH)
    if pose["chosen"] != expected_axis:
        raise AssertionError(f"{arm}/{frame['frame_id']}: selected axis changed")
    dims = target["physical_dimensions_m"]
    extents = (dims["across"], dims["height"], dims["along"])
    from symmetry_aware_pose_metrics import cuboid_model_points
    metrics = G.score_pose_against_gt(cuboid_model_points(extents),
        pose["rotation"], pose["translation"], np.asarray(target["R_gt_representative"]),
        np.asarray(target["t_gt"]), extents)
    fields = {
        "translation_error_cm": "translation_error_cm",
        "rotation_error_deg": "rotation_error_deg",
        "yaw_error_deg": "yaw_error_deg",
        "iou3d": "iou3d",
        "symmetry_aware_add_m": "add_sym_m",
    }
    errors = [_compare_array(metrics[left], expected[right], f"{arm}/pose/{right}", pose_atol)
              for left, right in fields.items()]
    if "reprojection_mean_px" in expected:
        errors.append(_compare_array(pose["reprojection_mean_px"],
                                     expected["reprojection_mean_px"],
                                     f"{arm}/pose/reprojection", pose_atol))
    return max(errors, default=0.)


def gpu_status(require_idle=True) -> dict:
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu",
        "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True).strip()
    foreign = []
    for row in processes.splitlines():
        cells = [value.strip() for value in row.split(",")]
        if cells and cells[0] != str(os.getpid()) and "/usr/share/rustdesk/rustdesk" not in row:
            foreign.append(row)
    if require_idle and foreign:
        raise RuntimeError(f"Foreign GPU compute is active; benchmark did not start: {foreign}")
    return dict(timestamp=C.now(), gpu=gpu, compute_processes=processes,
                foreign_compute=foreign)


def thread_state(require_one: bool = False) -> dict:
    """Return actual native and framework CPU pool sizes, optionally enforcing one."""
    pools = []
    for value in threadpool_info():
        row = {
            key: value.get(key)
            for key in ("user_api", "internal_api", "prefix", "filepath", "version", "num_threads")
        }
        pools.append(row)
    state = dict(
        torch_intraop=torch.get_num_threads(),
        torch_interop=torch.get_num_interop_threads(),
        opencv=cv2.getNumThreads(),
        environment={name: os.environ.get(name) for name in
                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        native_pools=pools,
    )
    counts = [state["torch_intraop"], state["torch_interop"], state["opencv"]]
    counts.extend(value["num_threads"] for value in pools if value["num_threads"] is not None)
    if require_one and (any(value != 1 for value in counts)
                        or set(state["environment"].values()) != {"1"}):
        raise RuntimeError(f"CPU thread contract could not be enforced: {state}")
    return state


def _selected_protocol(bundle: dict, models: dict, threads: dict) -> dict:
    selected = []
    for frame in bundle["frames"]:
        selected.append(dict(frame_id=frame["frame_id"], pose_frame_id=frame["pose_frame_id"],
            session_id=frame["session_id"], object_type=frame["object_type"], image=frame["image_binding"],
            original_hw=list(frame["image"].shape[:2]), camera_intrinsics=frame["camera"].tolist(),
            canonical_WDH_m=frame["dimensions"].tolist()))
    def operative(rule):
        return {
            "lam": rule["lam"],
            "max_move_image_diagonal_fraction": rule["max_move_image_diagonal_fraction"],
        }
    rules = {
        "YOLO_P1": dict(rule=operative(bundle["yolo"]["selection"]["selected_rule"]),
                        temperature=bundle["yolo"]["selection"]["temperatures"]["1"]["temperature"]),
        "DOPE_P1": dict(rule=operative(bundle["dope"]["selection"]["rules"]["P"]),
                        temperature=bundle["dope"]["selection"]["temperatures"]["P1"]),
    }
    for arm in ("P0", "D0", "P5_CONSTANT", "P5"):
        rules[f"RESNET_{arm}_S1"] = dict(
            rule=operative(bundle["resnet"]["selection"]["rules"][arm]),
            temperature=bundle["resnet"]["selection"]["temperatures"][f"{arm}_S1"])
    code = [C.binding(C.HERE / name) for name in
            ("common.py", "adapters.py", "benchmark.py", "report.py")]
    return dict(
        schema="pallet_three_backbone_runtime_protocol_v1", complete=True,
        purpose="Same-frame desktop latency for three frozen image estimators and their fixed selected correction heads.",
        primary_arms=list(C.PRIMARY_ARMS), auxiliary_arms=list(C.AUXILIARY_ARMS),
        baseline_pairs=C.BASELINE_FOR, models=models,
        frames=C.FRAMES, sessions=13, batch=C.BATCH, warmup_per_arm=C.WARMUP,
        repeats=C.REPEATS, measured_calls=len(C.measurement_schedule()),
        warmup_calls=len(C.warmup_schedule()), selected=selected,
        measurement_schedule=C.measurement_schedule(), warmup_schedule=C.warmup_schedule(),
        order="Repeat-level cyclic arm rotation; odd repeat blocks use reversed order; no fastest repeat selection.",
        threads=threads,
        software=dict(python=sys.version.split()[0], torch=torch.__version__,
                      cuda=torch.version.cuda, opencv=cv2.__version__, numpy=np.__version__,
                      ultralytics=package_version("ultralytics"),
                      threadpoolctl=package_version("threadpoolctl")),
        input_start="Already decoded native uint8 BGR in RAM; file read/decode excluded.",
        endpoint_2d="Preprocessing, backbone forward, decoder and selected correction rule through original-pixel 2D points.",
        endpoint_full="The 2D endpoint plus the same prediction-only MAIN axis selector and SQPnP/RefineLM.",
        excluded=["image file read/decode", "model construction/checkpoint load", "parity checks",
                  "ground-truth metric calculation", "JSON writes"],
        failure_policy="Every warmup/measured call is retained. Detection/PnP failures are timed and summarized by status.",
        parity=dict(
            two_d=(f"Saved DEV output replay tolerance <={C.POINT_ATOL:g}; "
                   "YOLO/DOPE were bit exact in the all-frame audit."),
            pose=(f"Saved status/axis must match exactly and stored pose metrics use "
                  f"tolerance <={C.POSE_ATOL:g}."),
            premeasurement_all_frame_audit=dict(
                frames=26, arms=9, outputs_created=False,
                maximum_2d_abs=C.PARITY_AUDIT_MAX_POINT,
                maximum_pose_metric_abs=C.PARITY_AUDIT_MAX_POSE),
            failure="Abort immediately; never keep a completed summary with failed parity."),
        resnet_head_policy="Only seed-1 P0/D0/P5_CONSTANT/P5 heads are resident; the twelve-head evaluation bank is forbidden.",
        selected_source_rules=rules, dependencies=bundle["dependencies"], code=code,
        GT_used_for_timing=False, GT_used_for_post_timing_parity=True,
        retries_automatic=0, fastest_trial_selection=False, device="single local desktop CUDA GPU")


def _append(handle, value) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _passed_row(raw: dict, error_2d: float, error_pose: float) -> dict:
    """Join a timed event with its successful post-timing parity event."""
    return dict(raw, parity_status="PASS", two_d_parity_max_abs_px=error_2d,
                pose_parity_max_abs=error_pose, parity_PASS=True)


def _measure_one(models, arm, frame, modules):
    # Global cuDNN flags differ in the frozen source experiments; restore them
    # outside the latency interval before every arm call.
    models.configure_backend(arm)
    torch.cuda.synchronize(models.device)
    start = time.perf_counter_ns()
    prediction = models.predict(arm, frame["image"], frame["dimensions"])
    torch.cuda.synchronize(models.device)
    two_d_end = time.perf_counter_ns()
    pose = solve_pose(prediction["points"], prediction["has_detection"], frame["camera"],
                      frame["spec"], modules)
    end = time.perf_counter_ns()
    values = dict(two_d_ms=(two_d_end - start) / 1e6,
                  pnp_ms=(end - two_d_end) / 1e6,
                  full_ms=(end - start) / 1e6,
                  pose_status=pose["status"],
                  finite_corners=0 if prediction["points"] is None else
                    int(np.isfinite(prediction["points"][:8]).all(-1).sum()),
                  has_detection=bool(prediction["has_detection"]))
    if not np.isfinite([values["two_d_ms"], values["pnp_ms"], values["full_ms"]]).all():
        raise FloatingPointError("Nonfinite runtime")
    if values["two_d_ms"] <= 0 or values["pnp_ms"] < 0 or values["full_ms"] < values["two_d_ms"]:
        raise AssertionError("Invalid runtime interval")
    return values, prediction, pose


def _write_report_and_receipt(result: dict, protocol: dict) -> None:
    """Deterministically finish, or validate, the report for a completed result."""
    from .report import render_report

    report = render_report(result, protocol)
    report_path = C.DOC / "REPORT_KO.md"
    if report_path.exists() and report_path.read_text() != report:
        raise ValueError("Frozen report differs")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        pending = report_path.with_name(report_path.name + ".pending")
        pending.write_text(report)
        pending.replace(report_path)
    C.write_frozen_json(C.DOC / "REPORT_COMPLETE.json", dict(
        complete=True, PASS=True, protocol=C.binding(C.DOC / "PROTOCOL.json"),
        results=C.binding(C.DOC / "RESULTS.json"), report=C.binding(report_path),
        raw_rows=C.binding(C.verify_binding(result["raw_rows"])),
        code=C.binding(C.HERE / "report.py")))


def _verify_protocol_bindings(protocol: dict) -> None:
    code = protocol.get("code", [])
    dependencies = protocol.get("dependencies", [])
    verified_code = {C.verify_binding(source) for source in code}
    expected_code = {(C.HERE / name).resolve() for name in
                     ("common.py", "adapters.py", "benchmark.py", "report.py")}
    if verified_code != expected_code or not dependencies:
        raise ValueError("Protocol runtime code/dependency bindings are incomplete")
    for source in dependencies:
        C.verify_binding(source)
    selected = protocol.get("selected", [])
    if len(selected) != C.FRAMES:
        raise ValueError("Protocol does not bind the exact 26 selected frames")
    actual_images = [C.verify_binding(frame["image"]) for frame in selected]
    expected_images = [(C.ROOT / key).resolve()
                       for key in C.read_json(YOLO_RUNTIME_PROTOCOL)["keys"]]
    if actual_images != expected_images:
        raise ValueError("Protocol selected-frame order differs from the frozen historical list")


def _recover_completed_result() -> dict:
    """Complete an interrupted report-only finalization without rerunning timing."""
    result = C.require_complete(C.DOC / "RESULTS.json",
                                schema="pallet_three_backbone_runtime_results_v1")
    protocol_path = C.verify_binding(result["protocol"])
    if protocol_path != (C.DOC / "PROTOCOL.json").resolve():
        raise ValueError("Completed result binds the wrong protocol")
    protocol = C.require_complete(protocol_path,
                                  schema="pallet_three_backbone_runtime_protocol_v1")
    _verify_protocol_bindings(protocol)
    C.verify_binding(result["raw_rows"])
    measurements = result.get("measurements", [])
    if (result.get("PASS") is not True or result.get("measured_calls") != 1170
            or result.get("warmup_calls") != 180
            or len(measurements) != len(C.measurement_schedule())
            or any(row.get("parity_status") != "PASS" or row.get("parity_PASS") is not True
                   for row in measurements)):
        raise ValueError("Completed result has incomplete parity rows")
    attempt_name = result.get("attempt")
    if (not isinstance(attempt_name, str) or not attempt_name.startswith("attempt_")
            or Path(attempt_name).name != attempt_name):
        raise ValueError("Completed result has an invalid attempt name")
    C.write_frozen_json(C.RAW / "runtime" / attempt_name / "COMPLETE.json", result)
    _write_report_and_receipt(result, protocol)
    receipt = C.require_complete(C.DOC / "REPORT_COMPLETE.json")
    for key in ("protocol", "results", "report", "raw_rows", "code"):
        C.verify_binding(receipt[key])
    return result


def run() -> dict:
    # Deliberately before directory creation or any output write.
    bundle = preflight()
    if (C.DOC / "RESULTS.json").exists():
        return _recover_completed_result()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Python imports NumPy/OpenCV/Torch before run(); constrain already-loaded
    # native pools as well as pools created later in this one-shot process.
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise RuntimeError("Run the benchmark in a fresh process so Torch inter-op threads can be fixed to 1") from exc
    cv2.setNumThreads(1)
    threads_before_models = thread_state(require_one=True)
    before = gpu_status(require_idle=True)
    from .adapters import UnifiedModels
    models = UnifiedModels("cuda")
    threadpool_limits(limits=1)
    threads_after_models = thread_state(require_one=True)
    modules = canonical_modules()

    # A real first-frame parity gate is excluded from timing and precedes every output.
    for arm in C.ARMS:
        models.configure_backend(arm)
        prediction = models.predict(arm, bundle["frames"][0]["image"],
                                    bundle["frames"][0]["dimensions"])
        pose = solve_pose(prediction["points"], prediction["has_detection"],
                          bundle["frames"][0]["camera"], bundle["frames"][0]["spec"], modules)
        check_2d_parity(arm, prediction, bundle["frames"][0])
        check_pose_parity(arm, pose, bundle["frames"][0], modules)

    protocol = _selected_protocol(bundle, models.identity(), dict(
        configured_before_model_init=threads_before_models,
        observed_after_model_init=threads_after_models))
    C.write_frozen_json(C.DOC / "PROTOCOL.json", protocol)
    runtime_root = C.RAW / "runtime"
    previous = sorted(runtime_root.glob("attempt_*")) if runtime_root.exists() else []
    prior = []
    for directory in previous:
        failure = directory / "FAILED.json"
        if not failure.is_file():
            raise RuntimeError(f"Unclosed previous attempt requires review: {directory}")
        prior.append(C.binding(failure))
    attempt = runtime_root / f"attempt_{len(previous) + 1:04d}"
    attempt.mkdir(parents=True)
    rows_path = attempt / "ROWS.jsonl"
    started = C.now(); warmup_rows = []; measured_rows = []
    gpu_checks = [dict(boundary="before_model_initialization", **before)]
    C.write_frozen_json(attempt / "START.json", dict(
        complete=False, started=started, protocol=C.binding(C.DOC / "PROTOCOL.json"),
        gpu=before, prior_failed_attempts=prior, model_identity=models.identity()))
    try:
        with rows_path.open("x") as handle:
            for index, job in enumerate(C.warmup_schedule()):
                frame = bundle["frames"][job["image_index"]]
                timing, prediction, pose = _measure_one(models, job["arm"], frame, modules)
                raw = dict(phase="warmup", call=index + 1, **job, frame_id=frame["frame_id"],
                           session_id=frame["session_id"], **timing, parity_status="PENDING")
                _append(handle, raw)
                error_2d = check_2d_parity(job["arm"], prediction, frame)
                error_pose = check_pose_parity(job["arm"], pose, frame, modules)
                row = _passed_row(raw, error_2d, error_pose)
                _append(handle, dict(phase="warmup_parity", call=index + 1,
                    arm=job["arm"], two_d_parity_max_abs_px=error_2d,
                    pose_parity_max_abs=error_pose, PASS=True))
                warmup_rows.append(row)
            for index, job in enumerate(C.measurement_schedule()):
                if job["arm_position"] == 0 and job["image_index"] == 0:
                    check = gpu_status(require_idle=True)
                    gpu_checks.append(dict(boundary=f"repeat_{job['repeat']}_start", **check))
                frame = bundle["frames"][job["image_index"]]
                timing, prediction, pose = _measure_one(models, job["arm"], frame, modules)
                raw = dict(phase="measured", call=index + 1, **job, frame_id=frame["frame_id"],
                           session_id=frame["session_id"], **timing, parity_status="PENDING")
                _append(handle, raw)
                error_2d = check_2d_parity(job["arm"], prediction, frame)
                error_pose = check_pose_parity(job["arm"], pose, frame, modules)
                row = _passed_row(raw, error_2d, error_pose)
                _append(handle, dict(phase="measured_parity", call=index + 1,
                    arm=job["arm"], two_d_parity_max_abs_px=error_2d,
                    pose_parity_max_abs=error_pose, PASS=True))
                measured_rows.append(row)
                if (index + 1) % (C.FRAMES * len(C.ARMS)) == 0:
                    print("THREE_BACKBONE_RUNTIME_BLOCK", index + 1, len(C.measurement_schedule()), flush=True)
        if len(warmup_rows) != 180 or len(measured_rows) != 1170:
            raise AssertionError("Runtime call count drift")
        _verify_protocol_bindings(protocol)
        summary = C.summarize(measured_rows)
        after = gpu_status(require_idle=True)
        gpu_checks.append(dict(boundary="after_measurement", **after))
        result = dict(
            schema="pallet_three_backbone_runtime_results_v1", complete=True, PASS=True,
            started=started, finished=C.now(), protocol=C.binding(C.DOC / "PROTOCOL.json"),
            attempt=attempt.name, raw_rows=C.binding(rows_path), prior_failed_attempts=prior,
            gpu_before=before, gpu_after=after, gpu_checks=gpu_checks,
            frames=C.FRAMES, sessions=13, repeats=C.REPEATS, batch=C.BATCH,
            warmup_calls=len(warmup_rows), measured_calls=len(measured_rows),
            primary_arms=list(C.PRIMARY_ARMS), auxiliary_arms=list(C.AUXILIARY_ARMS),
            summary=summary, measurements=measured_rows,
            all_2d_parity_PASS=True, all_pose_parity_PASS=True,
            maximum_2d_parity_abs_px=max(row["two_d_parity_max_abs_px"] for row in measured_rows),
            maximum_pose_parity_abs=max(row["pose_parity_max_abs"] for row in measured_rows),
            no_fastest_selection=True, failed_rows_discarded=False,
            GT_used_for_timing=False, GT_used_for_post_timing_parity=True,
            limitations=("One local desktop GPU; batch1; decoded-image start; 26 reused DEV frames; "
                         "five repeated blocks are technical latency samples, not accuracy replicates or Jetson timing."))
        C.write_frozen_json(C.DOC / "RESULTS.json", result)
        C.write_frozen_json(attempt / "COMPLETE.json", result)
        _write_report_and_receipt(result, protocol)
        print("THREE_BACKBONE_RUNTIME_COMPLETE", result["measured_calls"], flush=True)
        return result
    except BaseException as exc:
        failure = dict(
                complete=False, PASS=False, started=started, failed_at=C.now(),
                error_type=type(exc).__name__, error=str(exc),
                warmup_parities_completed=len(warmup_rows),
                measured_parities_completed=len(measured_rows),
                protocol=C.binding(C.DOC / "PROTOCOL.json"), no_automatic_retry=True)
        if rows_path.exists():
            failure["raw_rows"] = C.binding(rows_path)
        C.write_frozen_json(attempt / "FAILED.json", failure)
        raise
    finally:
        models.close()


def parity_audit() -> dict:
    """Read-only all-frame replay used to set a defensible numerical tolerance."""
    bundle = preflight()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise RuntimeError("Run parity-audit in a fresh process") from exc
    cv2.setNumThreads(1)
    gpu_status(require_idle=True)
    from .adapters import UnifiedModels
    models = UnifiedModels("cuda")
    modules = canonical_modules()
    errors = {arm: {"two_d": [], "pose": []} for arm in C.ARMS}
    try:
        for frame in bundle["frames"]:
            for arm in C.ARMS:
                models.configure_backend(arm)
                prediction = models.predict(arm, frame["image"], frame["dimensions"])
                pose = solve_pose(prediction["points"], prediction["has_detection"],
                                  frame["camera"], frame["spec"], modules)
                errors[arm]["two_d"].append(
                    check_2d_parity(arm, prediction, frame, point_atol=float("inf")))
                errors[arm]["pose"].append(
                    check_pose_parity(arm, pose, frame, modules, pose_atol=float("inf")))
    finally:
        models.close()
    return {
        "PASS": True,
        "frames": len(bundle["frames"]),
        "outputs_created": False,
        "maximum_by_arm": {
            arm: {
                "two_d_abs": max(value["two_d"], default=0.),
                "pose_metric_abs": max(value["pose"], default=0.),
            }
            for arm, value in errors.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selfcheck", "preflight", "parity-audit", "measure"))
    args = parser.parse_args()
    if args.command == "selfcheck":
        print(json.dumps(C.selfcheck(), indent=2))
    elif args.command == "preflight":
        bundle = preflight()
        print(json.dumps(dict(PASS=True, frames=len(bundle["frames"]), sessions=13,
                              outputs_created=False), indent=2))
    elif args.command == "parity-audit":
        print(json.dumps(parity_audit(), indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
