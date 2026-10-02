"""Locked CUDA runtime measurement for DOPE/ResNet-18 with N3 seed 1.

The fixed population comes from the already sealed 26-image runtime protocol.
Images are decoded into RAM before timing.  Two end-to-end paths include
preprocessing, the frozen base, prediction-only PnP, and (for the N3 path) the
fixed seed-1 correction.  A third path starts from a resident frozen-base
output and measures N3 conditioning, forward, decode, affine/cap conversion,
and preservation checks without PnP.

CUDA events are the primary clock and every boundary is synchronized.  A wall
clock is retained alongside it so the CPU portions of the scope remain
auditable.  Runtime values are descriptive only and are never used for model,
temperature, checkpoint, frame, or example selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from . import common as C


RAW_SCHEMA = "pallet_n3_completion_v3_runtime_raw_v1"
RECEIPT_SCHEMA = "pallet_n3_completion_v3_runtime_receipt_v1"
BACKBONES = tuple(C.CONFIGS)
PATHS = ("base_e2e", "n3_seed1_e2e", "n3_seed1_only")
SEED = 1
FRAMES = 26
SESSIONS = 13
WARMUP = 20
REPEATS = 5
BATCH = 1
TORCH_THREADS = 4
OPENCV_THREADS = 1
LEGACY_RUNTIME_PROTOCOL = (
    C.ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1/RUNTIME_PROTOCOL.json")
LEGACY_RUNTIME_PROTOCOL_SHA256 = (
    "f411f8654cc8cb4578f7a3fb676e7cf444d340d2b1cddc1353a60135d9977e85")
POSE_SOURCE = C.ROOT / "scripts/research/pallet_dim_conditioned_p_v1/pose.py"


def describe_ms(values: Sequence[float]) -> dict:
    """Return the locked linear median/P90 and batch-1 FPS."""
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all()
            or (array <= 0).any()):
        raise ValueError("Latency samples must be a nonempty positive finite vector")
    median = float(np.quantile(array, .5, method="linear"))
    p90 = float(np.quantile(array, .9, method="linear"))
    return {
        "count": int(len(array)),
        "median_ms": median,
        "p90_ms": p90,
        "fps_from_median": float(1000. / median),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
        "mean_ms": float(array.mean()),
        "quantile_method": "numpy linear",
    }


def schedule(frames: int = FRAMES, repeats: int = REPEATS) -> list[dict]:
    """Rotate the three paths at every image/repeat to balance order."""
    if frames <= 0 or repeats <= 0:
        raise ValueError("frames and repeats must be positive")
    jobs = []
    for repeat in range(repeats):
        for image_index in range(frames):
            offset = (repeat + image_index) % len(PATHS)
            for position in range(len(PATHS)):
                jobs.append({
                    "repeat": repeat,
                    "image_index": image_index,
                    "path": PATHS[(offset + position) % len(PATHS)],
                    "path_position": position,
                })
    return jobs


def warmup_schedule(frames: int = FRAMES, warmup: int = WARMUP) -> list[dict]:
    """Exactly ``warmup`` calls per path with rotating path order."""
    if frames <= 0 or warmup <= 0:
        raise ValueError("frames and warmup must be positive")
    jobs = []
    for warmup_index in range(warmup):
        image_index = warmup_index % frames
        for position in range(len(PATHS)):
            jobs.append({
                "warmup_index": warmup_index,
                "image_index": image_index,
                "path": PATHS[(warmup_index + position) % len(PATHS)],
                "path_position": position,
            })
    return jobs


def apply_preserving_delta(
    base_points: Any,
    base_valid: Any,
    delta: Any,
    support: Any,
) -> tuple[np.ndarray, dict]:
    """Apply eight corner deltas while proving center/missing preservation."""
    points = np.asarray(base_points, dtype=np.float64)
    valid = np.asarray(base_valid, dtype=bool)
    displacement = np.asarray(delta, dtype=np.float64)
    supported = np.asarray(support, dtype=bool)
    if (points.shape != (9, 2) or valid.shape != (9,)
            or displacement.shape != (8, 2) or supported.shape != (8,)):
        raise ValueError("Preservation inputs have invalid shapes")
    finite = np.isfinite(points).all(-1)
    if not np.array_equal(finite, valid):
        raise ValueError("Base validity and finite-point masks disagree")
    if not np.isfinite(displacement[supported]).all():
        raise ValueError("Supported deltas must be finite")
    before = points.copy()
    refined = points.copy()
    usable = valid[:8] & supported
    refined[:8][usable] += displacement[usable]
    preservation = {
        "center8_preserved": bool(np.array_equal(
            refined[8], before[8], equal_nan=True)),
        "missing_point_mask_preserved": bool(np.array_equal(
            np.isfinite(refined).all(-1), valid)),
        "point_order_preserved": True,
    }
    if not all(preservation.values()):
        raise AssertionError("N3 violated its preservation contract")
    if not np.array_equal(points, before, equal_nan=True):
        raise AssertionError("N3 mutated the base point array")
    return refined, preservation


def _timing_config() -> dict:
    return {
        "frames": FRAMES,
        "sessions": SESSIONS,
        "warmup_per_path": WARMUP,
        "repeats": REPEATS,
        "seed": SEED,
        "batch": BATCH,
        "paths": list(PATHS),
        "torch_threads": TORCH_THREADS,
        "opencv_threads": OPENCV_THREADS,
        "primary_clock": "torch.cuda.Event(enable_timing=True)",
        "synchronization": "torch.cuda.synchronize immediately before and after every call",
        "quantile": "numpy linear",
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict:
    rows = list(rows)
    output = {}
    for path in PATHS:
        selected = [row for row in rows if row.get("path") == path]
        if len(selected) != FRAMES * REPEATS:
            raise ValueError(f"{path}: expected {FRAMES * REPEATS} measured rows")
        output[path] = {
            "cuda_event_ms": describe_ms([row["cuda_event_ms"] for row in selected]),
            "synchronized_wall_ms": describe_ms([row["wall_ms"] for row in selected]),
            "peak_allocated_bytes": int(max(
                row["peak_allocated_bytes"] for row in selected)),
            "peak_reserved_bytes": int(max(
                row["peak_reserved_bytes"] for row in selected)),
            "maximum_incremental_allocated_bytes": int(max(
                row["incremental_peak_allocated_bytes"] for row in selected)),
        }
    return output


def _validate_selected(selected: Any, *, verify_files: bool) -> None:
    if not isinstance(selected, list) or len(selected) != FRAMES:
        raise ValueError("Runtime raw payload requires exactly 26 selected frames")
    frame_ids = [row.get("frame_id") for row in selected]
    sessions = [row.get("session_id") for row in selected]
    if len(set(frame_ids)) != FRAMES or set(Counter(sessions).values()) != {2}:
        raise ValueError("Runtime frame/session identity drift")
    if len(set(sessions)) != SESSIONS:
        raise ValueError("Runtime must contain 13 sessions")
    for row in selected:
        dimensions = np.asarray(row.get("dimensions_wdh_m"), dtype=float)
        camera = np.asarray(row.get("camera_intrinsics"), dtype=float)
        if (dimensions.shape != (3,) or not np.isfinite(dimensions).all()
                or (dimensions <= 0).any() or camera.shape != (3, 3)
                or not np.isfinite(camera).all() or camera[0, 0] <= 0
                or camera[1, 1] <= 0):
            raise ValueError("Runtime dimensions/camera contract drift")
        if verify_files:
            C.verify(row["image"])
            C.verify(row["camera_container"])


def _validate_rows(rows: Any, selected: list[dict], *, warmup: bool) -> None:
    expected_jobs = warmup_schedule() if warmup else schedule()
    if not isinstance(rows, list) or len(rows) != len(expected_jobs):
        raise ValueError("Runtime row count drift")
    fields = ("warmup_index", "image_index", "path", "path_position") if warmup else (
        "repeat", "image_index", "path", "path_position")
    for index, (row, expected) in enumerate(zip(rows, expected_jobs)):
        if any(row.get(key) != expected[key] for key in fields):
            raise ValueError(f"Runtime schedule drift at row {index}")
        if row.get("frame_id") != selected[expected["image_index"]]["frame_id"]:
            raise ValueError("Runtime frame order drift")
        timing = np.asarray([row.get("cuda_event_ms"), row.get("wall_ms")], float)
        if not np.isfinite(timing).all() or (timing <= 0).any():
            raise ValueError("Runtime latency must be positive and finite")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes",
                    "incremental_peak_allocated_bytes"):
            if not isinstance(row.get(key), int) or row[key] < 0:
                raise ValueError("Runtime memory values must be nonnegative integers")
        if row["path"].startswith("n3_"):
            preservation = row.get("output", {}).get("preservation")
            if not isinstance(preservation, Mapping) or not all(preservation.values()):
                raise ValueError("N3 runtime row failed preservation")


def validate_raw_payload(payload: Mapping[str, Any], backbone: str,
                         *, verify_files: bool = True) -> dict:
    """Strictly validate raw rows; the pure mode is used by CPU tests."""
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    required = {
        "schema", "complete", "backbone", "seed", "config", "hardware",
        "parameters", "selected", "warmup", "measurements", "bindings",
        "accuracy_values_read", "runtime_used_for_selection", "GT_keypoints_read",
        "decoded_images_in_RAM", "no_fastest_trial_selection",
    }
    if required - set(payload):
        raise ValueError(f"Runtime raw fields missing: {sorted(required - set(payload))}")
    if (payload["schema"] != RAW_SCHEMA or payload["complete"] is not True
            or payload["backbone"] != backbone or payload["seed"] != SEED
            or payload["config"] != _timing_config()
            or payload["accuracy_values_read"] is not False
            or payload["runtime_used_for_selection"] is not False
            or payload["GT_keypoints_read"] is not False
            or payload["decoded_images_in_RAM"] is not True
            or payload["no_fastest_trial_selection"] is not True):
        raise ValueError("Runtime raw identity/anti-selection contract drift")
    hardware = payload["hardware"]
    if not isinstance(hardware, Mapping) or "RTX" not in str(hardware.get("name", "")).upper():
        raise ValueError("Runtime requires an explicitly identified RTX GPU")
    parameters = payload["parameters"]
    if (not isinstance(parameters, Mapping)
            or any(not isinstance(parameters.get(key), int) or parameters[key] <= 0
                   for key in ("base_total", "n3_total", "base_plus_n3_total"))
            or parameters["base_plus_n3_total"] != (
                parameters["base_total"] + parameters["n3_total"])):
        raise ValueError("Runtime parameter counts are invalid")
    _validate_selected(payload["selected"], verify_files=verify_files)
    _validate_rows(payload["warmup"], payload["selected"], warmup=True)
    _validate_rows(payload["measurements"], payload["selected"], warmup=False)
    if verify_files:
        if not isinstance(payload["bindings"], list) or not payload["bindings"]:
            raise ValueError("Runtime requires immutable input/code bindings")
        for binding in payload["bindings"]:
            C.verify(binding)
    return dict(payload)


def validate_receipt_payload(payload: Mapping[str, Any], backbone: str,
                             *, raw_payload: Mapping[str, Any] | None = None,
                             verify_files: bool = True) -> dict:
    """Validate a reusable result and recompute every reported statistic."""
    if (payload.get("schema") != RECEIPT_SCHEMA or payload.get("complete") is not True
            or payload.get("backbone") != backbone or payload.get("seed") != SEED
            or payload.get("config") != _timing_config()
            or payload.get("runtime_used_for_selection") is not False
            or payload.get("same_RTX_required") is not True
            or payload.get("no_fastest_trial_selection") is not True):
        raise ValueError("Runtime receipt contract drift")
    if raw_payload is None:
        if not verify_files:
            raise ValueError("Pure receipt validation requires raw_payload")
        raw_path = C.verify(payload["raw"])
        raw_payload = C.read(raw_path)
    elif verify_files:
        raw_path = C.verify(payload["raw"])
        if C.read(raw_path) != raw_payload:
            raise ValueError("Supplied raw payload differs from bound raw file")
    raw = validate_raw_payload(raw_payload, backbone, verify_files=verify_files)
    expected = summarize(raw["measurements"])
    if payload.get("summary") != expected:
        raise ValueError("Runtime receipt summary is not reproducible from raw rows")
    if (payload.get("parameters") != raw["parameters"]
            or payload.get("hardware") != raw["hardware"]
            or payload.get("measured_calls") != FRAMES * REPEATS * len(PATHS)
            or payload.get("warmup_calls") != WARMUP * len(PATHS)):
        raise ValueError("Runtime receipt/raw identity drift")
    if verify_files:
        for binding in payload.get("bindings", []):
            C.verify(binding)
    return dict(payload)


def raw_path(backbone: str) -> Path:
    return C.RAW / "runtime" / f"{backbone}_seed1.json"


def receipt_path(backbone: str) -> Path:
    return C.DOC / f"RUNTIME_{backbone.upper()}_SEED1.json"


def _make_receipt(raw: Mapping[str, Any]) -> dict:
    backbone = str(raw["backbone"])
    path = raw_path(backbone)
    return {
        "schema": RECEIPT_SCHEMA,
        "complete": True,
        "backbone": backbone,
        "seed": SEED,
        "config": _timing_config(),
        "raw": C.binding(path),
        "summary": summarize(raw["measurements"]),
        "parameters": raw["parameters"],
        "hardware": raw["hardware"],
        "measured_calls": FRAMES * REPEATS * len(PATHS),
        "warmup_calls": WARMUP * len(PATHS),
        "bindings": list(raw["bindings"]),
        "same_RTX_required": True,
        "runtime_used_for_selection": False,
        "no_fastest_trial_selection": True,
        "historical_YOLO_11_408_15_350_ms_overwritten": False,
        "memory_scope": raw["memory_scope"],
        "timing_scope": raw["timing_scope"],
    }


def _load_or_recover(backbone: str) -> dict | None:
    receipt_file = receipt_path(backbone)
    raw_file = raw_path(backbone)
    if receipt_file.exists():
        return validate_receipt_payload(C.read(receipt_file), backbone)
    if raw_file.exists():
        raw = validate_raw_payload(C.read(raw_file), backbone)
        receipt = _make_receipt(raw)
        C.write(receipt_file, receipt, freeze=True)
        return validate_receipt_payload(receipt, backbone, raw_payload=raw)
    return None


def _hardware(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Runtime evidence requires an available CUDA device")
    properties = torch.cuda.get_device_properties(device)
    name = str(properties.name)
    if "RTX" not in name.upper():
        raise RuntimeError(f"Runtime contract requires the same RTX class device, got {name!r}")
    uuid = getattr(properties, "uuid", None)
    cudnn_version = torch.backends.cudnn.version()
    return {
        "name": name,
        "uuid": None if uuid is None else str(uuid),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "multiprocessors": int(properties.multi_processor_count),
        "device_index": int(device.index if device.index is not None
                            else torch.cuda.current_device()),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": None if cudnn_version is None else int(cudnn_version),
    }


def _hardware_key(value: Mapping[str, Any]) -> tuple:
    return tuple(json.dumps(value.get(key), sort_keys=True) for key in (
        "name", "uuid", "total_memory_bytes", "compute_capability",
        "multiprocessors", "torch", "torch_cuda", "cudnn"))


def _require_same_hardware(backbone: str, hardware: Mapping[str, Any]) -> None:
    other = "resnet18" if backbone == "dope" else "dope"
    path = receipt_path(other)
    if path.exists():
        receipt = validate_receipt_payload(C.read(path), other)
        if _hardware_key(receipt["hardware"]) != _hardware_key(hardware):
            raise RuntimeError("DOPE and ResNet runtime receipts use different hardware/software")


def _fixed_inputs():
    if C.sha256(LEGACY_RUNTIME_PROTOCOL) != LEGACY_RUNTIME_PROTOCOL_SHA256:
        raise RuntimeError("Inherited fixed-26 runtime protocol changed")
    inherited = C.read(LEGACY_RUNTIME_PROTOCOL)
    if (len(inherited.get("keys", [])) != FRAMES
            or inherited.get("seed") != SEED
            or inherited.get("warmup") != WARMUP
            or inherited.get("repeats") != REPEATS
            or inherited.get("threads") != TORCH_THREADS
            or inherited.get("opencv") != OPENCV_THREADS):
        raise RuntimeError("Inherited runtime constants changed")

    from .inference import dataset_records
    records, _ = dataset_records("DEV319")
    records_by_key = {row["image_key"]: row for row in records}
    manifest = C.read(C.DEV)
    items_by_key = {row["image_path"]: row for row in manifest["items"]}
    if set(inherited["keys"]) - set(records_by_key) or set(inherited["keys"]) - set(items_by_key):
        raise RuntimeError("A fixed runtime image left DEV319")

    selected, images, cameras, dimensions = [], [], [], []
    for key in inherited["keys"]:
        record = records_by_key[key]
        item = items_by_key[key]
        if (record["frame_id"] != item["frame_id"]
                or record["session_id"] != item["session_id"]):
            raise RuntimeError("Runtime frame identity join failed")
        image_path = C.ROOT / key
        encoded = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable runtime image: {image_path}")
        annotation_path = C.ROOT / item["gt_v2_path"]
        camera_container = C.read(annotation_path).get("camera_data", {})
        intrinsics = camera_container.get("intrinsics", {})
        camera = np.asarray([
            [intrinsics.get("fx"), 0., intrinsics.get("cx")],
            [0., intrinsics.get("fy"), intrinsics.get("cy")],
            [0., 0., 1.],
        ], dtype=np.float64)
        if (camera.shape != (3, 3) or not np.isfinite(camera).all()
                or [camera_container.get("height"), camera_container.get("width")]
                != list(image.shape[:2])):
            raise RuntimeError("Runtime camera/image geometry drift")
        dims = np.asarray(record["dimensions"], dtype=np.float64)
        selected.append({
            "frame_id": record["frame_id"],
            "session_id": record["session_id"],
            "image_key": key,
            "image": C.binding(image_path),
            "camera_container": C.binding(annotation_path),
            "camera_fields_consumed": ["width", "height", "intrinsics"],
            "camera_intrinsics": camera.tolist(),
            "dimensions_wdh_m": dims.tolist(),
            "object_type": record["object_type"],
            "original_hw": list(image.shape[:2]),
        })
        images.append(image)
        cameras.append(camera)
        dimensions.append(dims)
    _validate_selected(selected, verify_files=True)
    return selected, images, cameras, dimensions


def _point_digest(points: Any) -> str:
    value = np.asarray(points, dtype="<f8")
    return hashlib.sha256(value.tobytes()).hexdigest()


def _pose_output(module, base: Mapping[str, Any], points: np.ndarray,
                 camera: np.ndarray, dimensions_wdh: np.ndarray) -> dict:
    prediction = None if base.get("bbox_original") is None else points
    pose = module.infer(
        prediction, camera, dimensions_wdh[[0, 2, 1]], source=False)
    return {
        "available": bool(pose.get("available", False)),
        "status": "OK" if pose.get("available", False) else "UNAVAILABLE",
    }


def _base_signature(base: Mapping[str, Any], points: np.ndarray,
                    pose: Mapping[str, Any] | None) -> dict:
    valid = np.asarray(base["valid"], dtype=bool)
    return {
        "point_sha256": _point_digest(points),
        "finite_corners": int(valid[:8].sum()),
        "has_box": base.get("bbox_original") is not None,
        "pose": None if pose is None else dict(pose),
    }


@torch.no_grad()
def _refine_seed1(predictor, base: dict, dimensions: np.ndarray,
                  raw_hw: Sequence[int]) -> tuple[np.ndarray, dict]:
    from .train import KEYS
    box_before = (None if base.get("bbox_original") is None else
                  np.asarray(base["bbox_original"]).copy())
    score_before = float(base.get("score", 0.))
    batch = predictor._batch(base, dimensions)
    head = predictor.heads[SEED]
    output = head(*(batch[key] for key in KEYS),
                  dimension_context=batch["dimension_context"], lam=0.)
    if not torch.isfinite(output["logits"]).all():
        raise FloatingPointError("N3 seed1 produced nonfinite logits")
    temperature = float(predictor.selection["temperatures"][str(SEED)])
    delta, support = predictor._original_delta(
        output, base, raw_hw, temperature)
    points, preservation = apply_preserving_delta(
        base["points_original"], base["valid"], delta, support)
    box_after = (None if base.get("bbox_original") is None else
                 np.asarray(base["bbox_original"]))
    same_box = ((box_before is None and box_after is None)
                or (box_before is not None and box_after is not None
                    and np.array_equal(box_before, box_after, equal_nan=True)))
    preservation["selected_instance_box_score_preserved"] = bool(
        same_box and float(base.get("score", 0.)) == score_before)
    preservation["temperature_fixed_before_runtime"] = True
    preservation["seed1_final_checkpoint_fixed_before_runtime"] = True
    if not all(preservation.values()):
        raise AssertionError("N3 mutated a preserved base output")
    return points, preservation


def _operation(path: str, predictor, pose_module, image: np.ndarray,
               camera: np.ndarray, dimensions: np.ndarray,
               cached_base: dict | None = None) -> dict:
    if path == "base_e2e":
        base = predictor.adapter.infer(
            image, source_pre_padded=False, return_features=False)
        points = np.asarray(base["points_original"], dtype=np.float64)
        pose = _pose_output(pose_module, base, points, camera, dimensions)
        return _base_signature(base, points, pose)
    if path == "n3_seed1_e2e":
        base = predictor.adapter.infer(
            image, source_pre_padded=False, return_features=True)
        points, preservation = _refine_seed1(
            predictor, base, dimensions, image.shape[:2])
        pose = _pose_output(pose_module, base, points, camera, dimensions)
        result = _base_signature(base, points, pose)
        result["preservation"] = preservation
        return result
    if path == "n3_seed1_only":
        if cached_base is None:
            raise ValueError("N3-only timing requires an untimed resident base output")
        points, preservation = _refine_seed1(
            predictor, cached_base, dimensions, image.shape[:2])
        result = _base_signature(cached_base, points, None)
        result["preservation"] = preservation
        return result
    raise ValueError(path)


def _timed_call(function, device: torch.device) -> tuple[dict, dict]:
    """Measure one batch-1 call with synchronized CUDA event and wall clocks."""
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_start = int(torch.cuda.memory_allocated(device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output = function()
        end.record()
        torch.cuda.synchronize(device)
        wall_end = time.perf_counter_ns()
        event_ms = float(start.elapsed_time(end))
        wall_ms = float((wall_end - wall_start) / 1e6)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if (not math.isfinite(event_ms) or not math.isfinite(wall_ms)
            or event_ms <= 0 or wall_ms <= 0):
        raise RuntimeError("CUDA runtime clock returned a nonpositive value")
    return output, {
        "cuda_event_ms": event_ms,
        "wall_ms": wall_ms,
        "allocated_at_start_bytes": allocated_start,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - allocated_start),
    }


def _dependencies(backbone: str) -> list[dict]:
    paths = [
        C.DOC / "PROTOCOL.json",
        LEGACY_RUNTIME_PROTOCOL,
        C.DEV,
        C.NORMALIZATION,
        C.HERE / "runtime.py",
        C.HERE / "inference.py",
        C.HERE / "adapters.py",
        C.HERE / "model.py",
        C.HERE / "train.py",
        C.HERE / "selection.py",
        POSE_SOURCE,
        C.DOC / f"TRAIN_{backbone.upper()}_SEED1.json",
        C.DOC / f"TRAINING_{backbone.upper()}_COMPLETE.json",
        C.DOC / f"SELECTION_{backbone.upper()}.json",
    ]
    training = C.read(C.DOC / f"TRAIN_{backbone.upper()}_SEED1.json")
    paths.append(C.verify(training["checkpoint"]))
    paths.append(C.DOPE_WEIGHTS if backbone == "dope" else C.RESNET_CONSTANT)
    bindings = [C.binding(path) for path in paths]
    if len({entry["path"] for entry in bindings}) != len(bindings):
        raise RuntimeError("Duplicate runtime dependency binding")
    return bindings


def _parameter_counts(predictor, backbone: str) -> dict:
    base = sum(parameter.numel() for parameter in predictor.adapter.network.parameters())
    n3 = sum(parameter.numel() for parameter in predictor.heads[SEED].parameters())
    training = C.read(C.DOC / f"TRAIN_{backbone.upper()}_SEED1.json")
    if training.get("trainable_parameters") != n3:
        raise RuntimeError("N3 seed1 parameter receipt drift")
    if any(parameter.requires_grad for parameter in predictor.adapter.network.parameters()):
        raise RuntimeError("Runtime base is not frozen")
    if any(parameter.requires_grad for parameter in predictor.heads[SEED].parameters()):
        raise RuntimeError("Runtime N3 head is not frozen")
    return {
        "base_total": int(base),
        "n3_total": int(n3),
        "base_plus_n3_total": int(base + n3),
        "runtime_requires_grad": 0,
    }


def _one_row(job: Mapping[str, Any], *, phase: str, predictor, pose_module,
             selected: list[dict], images: list[np.ndarray],
             cameras: list[np.ndarray], dimensions: list[np.ndarray]) -> dict:
    index = int(job["image_index"])
    path = str(job["path"])
    image, camera, dims = images[index], cameras[index], dimensions[index]
    cached = None
    if path == "n3_seed1_only":
        # This is deliberately outside the timer.  The isolated segment begins
        # with the base output resident on the same CUDA device.
        cached = predictor.adapter.infer(
            image, source_pre_padded=False, return_features=True)
        torch.cuda.synchronize(predictor.device)
    output, timing = _timed_call(
        lambda: _operation(
            path, predictor, pose_module, image, camera, dims, cached),
        predictor.device)
    row = {
        "phase": phase,
        **dict(job),
        "frame_id": selected[index]["frame_id"],
        "session_id": selected[index]["session_id"],
        **timing,
        "output": output,
    }
    del cached
    return row


def measure(backbone: str, device: str = "cuda") -> dict:
    """Measure or hash-verify/reuse one completed backbone receipt."""
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    reused = _load_or_recover(backbone)
    if reused is not None:
        return reused
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("Published runtime measurement is CUDA-only")
    if raw_path(backbone).exists() or receipt_path(backbone).exists():
        raise RuntimeError("Existing runtime artifacts failed validation and were preserved")

    torch.set_num_threads(TORCH_THREADS)
    cv2.setNumThreads(OPENCV_THREADS)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    gpu_start = C.gpu_snapshot(refuse_other_compute=True)
    selected, images, cameras, dimensions = _fixed_inputs()

    from .inference import Predictor
    from .metrics import _pose_contract
    predictor = Predictor(backbone, device=str(torch_device))
    if predictor.device.type != "cuda":
        raise RuntimeError("Predictor did not load on CUDA")
    # Runtime is fixed to seed1.  Heads 2/3 are discarded before any warmup or
    # memory measurement; they cannot affect latency or resident-memory totals.
    for seed in tuple(predictor.heads):
        if seed != SEED:
            del predictor.heads[seed]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(predictor.device)
    if set(predictor.heads) != {SEED}:
        raise RuntimeError("Runtime predictor must retain seed1 only")
    pose_module = _pose_contract()
    hardware = _hardware(predictor.device)
    _require_same_hardware(backbone, hardware)
    parameters = _parameter_counts(predictor, backbone)
    resident_after_load = int(torch.cuda.memory_allocated(predictor.device))

    warmup_rows = []
    for call, job in enumerate(warmup_schedule(), 1):
        row = _one_row(
            job, phase="warmup", predictor=predictor, pose_module=pose_module,
            selected=selected, images=images, cameras=cameras, dimensions=dimensions)
        row["call"] = call
        warmup_rows.append(row)
    measured_rows = []
    for call, job in enumerate(schedule(), 1):
        row = _one_row(
            job, phase="measured", predictor=predictor, pose_module=pose_module,
            selected=selected, images=images, cameras=cameras, dimensions=dimensions)
        row["call"] = call
        measured_rows.append(row)
        if call % (FRAMES * len(PATHS)) == 0:
            print("N3_RUNTIME", backbone, call, FRAMES * REPEATS * len(PATHS), flush=True)

    dependencies = _dependencies(backbone)
    gpu_finish = C.gpu_snapshot(refuse_other_compute=True)
    raw = {
        "schema": RAW_SCHEMA,
        "complete": True,
        "backbone": backbone,
        "seed": SEED,
        "started_gpu": gpu_start,
        "finished_gpu": gpu_finish,
        "finished_at": C.now(),
        "config": _timing_config(),
        "hardware": hardware,
        "parameters": parameters,
        "selected": selected,
        "warmup": warmup_rows,
        "measurements": measured_rows,
        "bindings": dependencies,
        "resident_allocated_after_models_bytes": resident_after_load,
        "untimed_resident_base_calls_for_n3_only": WARMUP + FRAMES * REPEATS,
        "memory_scope": (
            "Process-scoped base plus seed1 N3 resident allocation; per-call CUDA "
            "peak reset after synchronization; N3-only begins with one resident base output."),
        "timing_scope": {
            "base_e2e": (
                "decoded image in RAM -> preprocessing -> frozen base -> CPU decode "
                "-> locked prediction-only PnP"),
            "n3_seed1_e2e": (
                "decoded image in RAM -> preprocessing -> frozen base -> CPU decode "
                "-> N3 seed1 conditioning/forward/decode/anisotropic inverse/cap/" 
                "preservation -> locked prediction-only PnP"),
            "n3_seed1_only": (
                "resident base features/points -> N3 seed1 conditioning/forward/decode/" 
                "anisotropic inverse/cap/preservation; excludes base and PnP"),
            "file_IO": "excluded; all 26 images decoded before model construction/timing",
        },
        "precision": {"base": "FP32", "N3": "FP32 with FP16 feature handoff",
                      "AMP": False, "TF32": False},
        "accuracy_values_read": False,
        "runtime_used_for_selection": False,
        "GT_keypoints_read": False,
        "camera_fields_only_from_annotation_containers": True,
        "decoded_images_in_RAM": True,
        "no_fastest_trial_selection": True,
        "failed_or_partial_attempts_pooled": False,
        "historical_YOLO_latency_preserved_separately": True,
        "Jetson_measurement": False,
    }
    validate_raw_payload(raw, backbone)
    C.write(raw_path(backbone), raw, freeze=True)
    receipt = _make_receipt(raw)
    C.write(receipt_path(backbone), receipt, freeze=True)
    receipt = validate_receipt_payload(receipt, backbone, raw_payload=raw)
    del predictor, pose_module, images
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def verify(backbone: str) -> dict:
    path = receipt_path(backbone)
    if not path.exists():
        raise FileNotFoundError(f"Runtime receipt is absent: {path}")
    return validate_receipt_payload(C.read(path), backbone)


def _run_many(command: str, backbone: str, device: str) -> dict:
    requested = BACKBONES if backbone == "all" else (backbone,)
    results = {
        value: (measure(value, device) if command == "measure" else verify(value))
        for value in requested
    }
    if len(results) == 2:
        keys = {_hardware_key(receipt["hardware"]) for receipt in results.values()}
        if len(keys) != 1:
            raise RuntimeError("DOPE and ResNet runtime were not measured on the same RTX")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure", "verify"))
    parser.add_argument("backbone", choices=(*BACKBONES, "all"))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    results = _run_many(arguments.command, arguments.backbone, arguments.device)
    print(json.dumps({
        backbone: {
            "complete": result["complete"],
            "raw": result["raw"],
            "summary": result["summary"],
            "parameters": result["parameters"],
            "hardware": result["hardware"],
        }
        for backbone, result in results.items()
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
