"""Fixed, read-only YOLO R0/N3 replay of the four usable lifter recordings.

This command is an offline case study, not a controller.  It first freezes the
full-session, one-second sensor-time sample identities and only then loads a
model.  Every sampled image has one frozen YOLO R0 forward whose candidates and
neck features are shared by the unmodified R0 path and the historical
``N3_DIM_SYM`` seed-1 head.  Both paths use the same recorded camera matrix,
the same square-pallet dimensions, and the same prediction-only PnP function.

The recordings contain deployed-model outputs but no independent pose or
corner reference.  Consequently the output reports coverage, fresh-output
coverage, sensor-time missing runs, and wrap-aware temporal jitter.  Accuracy,
position error, and yaw error remain explicit ``null``/``x`` fields.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import common as C
from . import lifter


PLAN_SCHEMA = "pallet_n3_completion_v3_lifter_plan_v1"
RAW_SCHEMA = "pallet_n3_completion_v3_lifter_raw_v1"
RECEIPT_SCHEMA = "pallet_n3_completion_v3_lifter_receipt_v1"
SAMPLE_INTERVAL_MS = 1000.0
ARM = "N3_DIM_SYM"
SEED = 1
OBJECT_TYPE = "plastic_standard_110x110x15"
METHODS = ("R0", "N3_seed1")
EXPECTED_SAMPLE_COUNTS = {
    "173507": 343,
    "174126": 72,
    "174342": 243,
    "174925": 188,
}
EXPECTED_DIMENSIONS_WDH_M = np.asarray([1.1, 1.1, .15], np.float64)
EXPECTED_IMAGE_HW = (480, 640)

PLAN_PATH = C.RAW / "lifter" / "FIXED_SENSOR_1000MS_PLAN.json"
RAW_PATH = C.RAW / "lifter" / "YOLO_R0_N3_SEED1_RAW.json"
RECEIPT_PATH = C.DOC / "LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json"

DCP_CODE = C.ROOT / "scripts/research/pallet_dim_conditioned_p_v1"
OLD_CODE = C.ROOT / "scripts/research/pallet_final_ml_contribution_test_v1"
LINE_CODE = C.ROOT / "scripts/research/pallet_line_pose_v1"
DCP_DOC = C.ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1"
DCP_RAW = C.ROOT / "data/pallet/results/pallet_dim_conditioned_p_v1"
SELECTION_PATH = DCP_DOC / "CALIBRATION_AND_SELECTION.json"
NORMALIZATION_PATH = DCP_DOC / "DIM_NORMALIZATION_LOCK.json"
FIT_RECEIPT_PATH = DCP_DOC / "fits/N3_DIM_SYM_seed1.json"
N3_CHECKPOINT_PATH = DCP_RAW / "runs/N3_DIM_SYM_seed1/last.pt"
R0_RELATIVE = Path(
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")


# These inputs are historical and immutable.  Hard-coding their observed hash
# and byte length prevents a same-name replacement from entering the case study.
LEGACY_BINDINGS = (
    ("dcp_env", Path("scripts/research/pallet_dim_conditioned_p_v1/dcp_env.py"),
     "ceff2bb8a9589a1c12e043804bc247bab016fb083d93f54a8fcc8e07e6aae1f5", 2467),
    ("inference", Path("scripts/research/pallet_dim_conditioned_p_v1/inference.py"),
     "15621cbdc946b762aa36cddc4001972778df24f4925ff1473f7cb940856b4f0a", 3399),
    ("refiner", Path("scripts/research/pallet_dim_conditioned_p_v1/refiner.py"),
     "d6ec32c7c31e1c4716873460cb4d5a1985260cdfebae27067acb72bcac3a2164", 4526),
    ("pose", Path("scripts/research/pallet_dim_conditioned_p_v1/pose.py"),
     "4e8c1e6b4c4e885fb671af233ea2d90c416fd7b232b63c75ce9c3ffdeb45b1d7", 7720),
    ("point_inference", Path(
        "scripts/research/pallet_final_ml_contribution_test_v1/point_inference.py"),
     "cef2977d2c9ef700f35267d64a899e1b2924d77f5296f11b689f908db5db42f1", 3338),
    ("generic_point_refiner", Path(
        "scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py"),
     "ec854fcb51f33f1bcbf0e336135b10b8b121c876b0ae4f7ad4d4065cebe658bd", 6153),
    ("features", Path("scripts/research/pallet_line_pose_v1/features.py"),
     "296d81b2adde5ce74192f0012e554ae546c95f309a82dd808024ee34105721ab", 6641),
    ("pnp_solver", Path("scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py"),
     "312aef156b6b099f6518a17290b7f86256d94ec22c73b1fc40ff924e4aaf9720", 10902),
    ("selection", Path(
        "_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json"),
     "6ff554de489758fd4a8836fab3242266b54f83fd38c2ac64b4c81e5981d0030c", 11571),
    ("normalization", Path(
        "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"),
     "0f30210ce310e651750197e3c72d2c61ed5aacad7194f7e8cbcc256631f2250a", 852),
    ("fit_receipt", Path(
        "_docs/experiments/pallet_dim_conditioned_p_v1/fits/N3_DIM_SYM_seed1.json"),
     "c7180d6cb846b25e2474a84215700b740e85ead11e7a7f83704fc52a0e0cda27", 36104),
    ("n3_checkpoint", Path(
        "data/pallet/results/pallet_dim_conditioned_p_v1/"
        "runs/N3_DIM_SYM_seed1/last.pt"),
     "ceea743b7b8467cf43eef66582b923dd980e5b2fb27f5575af5df185109f22dd", 305470),
    ("geometry_registry", Path(
        "challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"),
     "9ae991450c384f4fcb82cff74a71fe07d51c5dcc496a5e8fcb28f41b4420ff76", 2939),
    ("symmetry_contract", Path(
        "_docs/experiments/pallet_symmetry_three_line_v1/"
        "OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"),
     "e692b152658292bc2890f415e2ae612116a4dd03850774f1fda234447d2995a3", 9314),
    ("yolo_r0", R0_RELATIVE,
     "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7", 6552807),
)


def _json_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nullable(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim:
        return [_nullable(item) for item in array]
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    if np.issubdtype(array.dtype, np.floating):
        number = float(array)
        return number if math.isfinite(number) else None
    return value


def _checkout_roots() -> tuple[Path, ...]:
    """Return this worktree and, when applicable, its Git common checkout."""
    roots = [C.ROOT.resolve()]
    dot_git = C.ROOT / ".git"
    if dot_git.is_file():
        marker = dot_git.read_text().strip()
        if marker.startswith("gitdir:"):
            git_dir = Path(marker.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (C.ROOT / git_dir).resolve()
            common_marker = git_dir / "commondir"
            if common_marker.is_file():
                common_dir = Path(common_marker.read_text().strip())
                if not common_dir.is_absolute():
                    common_dir = (git_dir / common_dir).resolve()
                roots.append(common_dir.parent.resolve())
    return tuple(dict.fromkeys(roots))


def _resolve_bound(relative: Path, sha256: str, size: int) -> Path:
    problems = []
    for root in _checkout_roots():
        path = root / relative
        if not path.is_file():
            problems.append(f"missing:{path}")
            continue
        observed_size = path.stat().st_size
        observed_sha = C.sha256(path)
        if observed_size == size and observed_sha == sha256:
            return path
        problems.append(
            f"drift:{path}:{observed_size}:{observed_sha}")
    raise RuntimeError("bound historical input unavailable: " + "; ".join(problems))


def verify_legacy_bindings() -> dict[str, Path]:
    """Verify every historical source, lock, checkpoint, and base weight."""
    resolved: dict[str, Path] = {}
    for name, relative, digest, size in LEGACY_BINDINGS:
        resolved[name] = _resolve_bound(relative, digest, size)
    return resolved


def public_legacy_bindings() -> list[dict]:
    return [
        {"name": name, "path": str(path), "sha256": digest, "bytes": size}
        for name, path, digest, size in LEGACY_BINDINGS
    ]


def fixed_selection_contract() -> dict:
    """Load and validate the already selected synthetic-only N3 configuration."""
    selection = C.read(SELECTION_PATH)
    entry = selection.get("temperatures", {}).get(f"{ARM}_seed{SEED}")
    rule = selection.get("rule")
    if (selection.get("complete") is not True or not isinstance(entry, Mapping)
            or float(entry.get("temperature", float("nan"))) != 1.0
            or rule != {"lam": 1.0, "max_move_image_diagonal_fraction": .01}
            or selection.get("new_lambda_cap_sweep") is not False
            or selection.get("real_access") is not False
            or selection.get("heldout_performance_access_before_lock") is not False):
        raise RuntimeError("historical synthetic selection contract drift")
    checkpoint = entry.get("checkpoint", {})
    expected = next(row for row in public_legacy_bindings()
                    if row["name"] == "n3_checkpoint")
    if any(checkpoint.get(key) != expected[key]
           for key in ("path", "sha256", "bytes")):
        raise RuntimeError("N3 seed-1 checkpoint selection binding drift")
    normalization = C.read(NORMALIZATION_PATH)
    mean = np.asarray(normalization.get("mean"), np.float64)
    scale = np.asarray(normalization.get("scale"), np.float64)
    if (mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()
            or normalization.get("input") != [
                "logW", "logD", "logH", "log(W/D)", "log(H/sqrt(WD))"]):
        raise RuntimeError("dimension normalization lock drift")
    fit = C.read(FIT_RECEIPT_PATH)
    if (fit.get("complete") is not True or fit.get("arm") != ARM
            or fit.get("seed") != SEED or fit.get("final_step_only") is not True
            or fit.get("R0_hash_unchanged") is not True
            or fit.get("real_DEV_access") is not False
            or fit.get("checkpoint") != checkpoint):
        raise RuntimeError("N3 seed-1 completion receipt drift")
    return {
        "arm": ARM,
        "seed": SEED,
        "temperature": 1.0,
        "rule": dict(rule),
        "selection_source": "fixed synthetic calibration; no lifter outcome selection",
        "normalization": normalization,
        "checkpoint": dict(checkpoint),
    }


def camera_contract(meta: Mapping[str, Any]) -> dict:
    """Validate and expose only the dimensions/K recorded in a session meta file."""
    dimensions = meta.get("pallet_size_m")
    if not isinstance(dimensions, Mapping):
        raise ValueError("session meta lacks pallet_size_m")
    wdh = np.asarray([
        dimensions.get("width"), dimensions.get("length"),
        dimensions.get("height")], np.float64)
    if not np.array_equal(wdh, EXPECTED_DIMENSIONS_WDH_M):
        raise ValueError("lifter session is not the frozen 1.10 x 1.10 x 0.15 m pallet")
    intrinsics = meta.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("session meta lacks recorded intrinsics")
    values = np.asarray([
        intrinsics.get("fx"), intrinsics.get("fy"), intrinsics.get("ppx"),
        intrinsics.get("ppy")], np.float64)
    if (not np.isfinite(values).all() or (values[:2] <= 0).any()
            or (intrinsics.get("height"), intrinsics.get("width")) != EXPECTED_IMAGE_HW):
        raise ValueError("invalid recorded camera intrinsics")
    coefficients = np.asarray(intrinsics.get("coeffs"), np.float64)
    if (intrinsics.get("model") != "distortion.inverse_brown_conrady"
            or coefficients.shape != (5,) or not np.array_equal(
                coefficients, np.zeros(5, np.float64))):
        raise ValueError("PnP contract requires the recorded zero-distortion intrinsics")
    fx, fy, ppx, ppy = values
    matrix = np.asarray([[fx, 0., ppx], [0., fy, ppy], [0., 0., 1.]],
                        np.float64)
    return {
        "dimensions_wdh_m": wdh.tolist(),
        "pnp_xyz_m": wdh[[0, 2, 1]].tolist(),
        "K": matrix.tolist(),
        "intrinsics_source": "recorded session meta; ppx/ppy are principal point",
        "distortion_model": intrinsics["model"],
        "distortion_coefficients": coefficients.tolist(),
        "image_hw": list(EXPECTED_IMAGE_HW),
    }


def fixed_sample_rows(timing_rows: Sequence[Mapping[str, Any]],
                      *, session_id: str,
                      interval_ms: float = SAMPLE_INTERVAL_MS) -> list[dict]:
    """Select a first-frame-anchored absolute sensor-time grid."""
    if not math.isfinite(interval_ms) or interval_ms <= 0:
        raise ValueError("sample interval must be finite and positive")
    selected = []
    due: float | None = None
    previous: float | None = None
    for video_index, row in enumerate(timing_rows):
        timestamp = float(row[lifter.SENSOR_TIME_FIELD])
        if not math.isfinite(timestamp) or (
                previous is not None and timestamp < previous):
            raise ValueError("timing rows are not finite sensor-time order")
        previous = timestamp
        use, due = lifter._is_time_sample(timestamp, due, interval_ms)
        if not use:
            continue
        selected.append({
            "session_id": str(session_id),
            "video_index": int(video_index),
            "frame_i": int(row["frame_i"]),
            "camera_frame_number": int(row["camera_frame_number"]),
            lifter.SENSOR_TIME_FIELD: timestamp,
            "camera_timestamp_domain": str(row["camera_timestamp_domain"]),
        })
    return selected


def _capture_bindings() -> list[dict]:
    rows = []
    for session_id in lifter.SESSION_IDS:
        for role, (size, digest) in lifter.EXPECTED_FILES[session_id].items():
            rows.append({
                "session_id": session_id,
                "role": role,
                "file": f"{lifter.PREFIX}{session_id}{lifter.FILE_SUFFIXES[role]}",
                "bytes": size,
                "sha256": digest,
            })
    return rows


def build_plan(capture_root: Path | str | None = None, *, verify: bool = True) -> dict:
    """Build the deterministic selection before any prediction is evaluated."""
    root = lifter.discover_capture_root(capture_root)
    if verify:
        lifter.inventory_captures(root, strict=True)
        verify_legacy_bindings()
    sessions = {}
    camera_reference = None
    for session_id in lifter.USABLE_SESSION_IDS:
        paths = lifter.session_paths(root, session_id)
        timing = lifter._read_timing_rows(paths["timing"])
        mapped = timing[lifter.VIDEO_TIMING_ROW_OFFSET:]
        if len(mapped) != lifter.EXPECTED_FACTS[session_id]["video_frames"]:
            raise RuntimeError("fixed raw-video/timing offset contract drift")
        frames = fixed_sample_rows(mapped, session_id=session_id)
        if len(frames) != EXPECTED_SAMPLE_COUNTS[session_id]:
            raise RuntimeError(f"fixed sample membership drift for {session_id}")
        if not frames or frames[0]["video_index"] != 0 or frames[0]["frame_i"] != 36:
            raise RuntimeError("first eligible raw frame is not the sample anchor")
        meta = json.loads(paths["meta"].read_text(encoding="utf-8-sig"))
        camera = camera_contract(meta)
        if camera_reference is None:
            camera_reference = camera
        elif camera != camera_reference:
            raise RuntimeError("recorded K/dimensions differ across usable sessions")
        sessions[session_id] = {
            "raw_video_frames": len(mapped),
            "sampled_frames": len(frames),
            "first_sensor_timestamp_ms": frames[0][lifter.SENSOR_TIME_FIELD],
            "last_sensor_timestamp_ms": frames[-1][lifter.SENSOR_TIME_FIELD],
            "camera": camera,
            "frames": frames,
        }
    plan = {
        "schema": PLAN_SCHEMA,
        "complete": True,
        "config": {
            "sessions": list(lifter.USABLE_SESSION_IDS),
            "sample_interval_ms": SAMPLE_INTERVAL_MS,
            "timebase_field": lifter.SENSOR_TIME_FIELD,
            "anchor": "first eligible mapped raw frame independently in each session",
            "range": "entire usable raw-video span of every fixed session",
            "video_timing_row_offset": lifter.VIDEO_TIMING_ROW_OFFSET,
            "nominal_or_container_fps_used": False,
            "object_type": OBJECT_TYPE,
            "methods": list(METHODS),
            "tracking_or_smoothing": False,
            "retained_previous_pose": False,
        },
        "selection_locked_before_inference": True,
        "performance_or_accuracy_values_read_for_selection": False,
        "total_raw_video_frames": sum(
            lifter.EXPECTED_FACTS[sid]["video_frames"]
            for sid in lifter.USABLE_SESSION_IDS),
        "total_sampled_frames": sum(EXPECTED_SAMPLE_COUNTS.values()),
        "sessions": sessions,
        "capture_bindings": _capture_bindings(),
        "legacy_bindings": public_legacy_bindings(),
    }
    validate_plan(plan, verify_files=False)
    return plan


def validate_plan(plan: Mapping[str, Any], *, verify_files: bool = True,
                  capture_root: Path | str | None = None) -> dict:
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("complete") is not True
            or plan.get("selection_locked_before_inference") is not True
            or plan.get("performance_or_accuracy_values_read_for_selection") is not False
            or plan.get("total_raw_video_frames") != 8910
            or plan.get("total_sampled_frames") != sum(EXPECTED_SAMPLE_COUNTS.values())):
        raise ValueError("fixed lifter plan identity drift")
    config = plan.get("config", {})
    if (config.get("sessions") != list(lifter.USABLE_SESSION_IDS)
            or config.get("sample_interval_ms") != SAMPLE_INTERVAL_MS
            or config.get("timebase_field") != lifter.SENSOR_TIME_FIELD
            or config.get("nominal_or_container_fps_used") is not False
            or config.get("tracking_or_smoothing") is not False
            or config.get("retained_previous_pose") is not False):
        raise ValueError("fixed lifter plan config drift")
    sessions = plan.get("sessions")
    if not isinstance(sessions, Mapping) or list(sessions) != list(lifter.USABLE_SESSION_IDS):
        raise ValueError("fixed lifter session order drift")
    for session_id in lifter.USABLE_SESSION_IDS:
        session = sessions[session_id]
        frames = session.get("frames")
        if (not isinstance(frames, list)
                or len(frames) != EXPECTED_SAMPLE_COUNTS[session_id]
                or session.get("sampled_frames") != len(frames)):
            raise ValueError(f"fixed frame count drift for {session_id}")
        previous = None
        for index, row in enumerate(frames):
            timestamp = float(row[lifter.SENSOR_TIME_FIELD])
            if (row.get("session_id") != session_id
                    or not isinstance(row.get("video_index"), int)
                    or (previous is not None and timestamp <= previous)):
                raise ValueError(f"fixed sample identity/time drift for {session_id}")
            if index == 0 and (row["video_index"] != 0 or row["frame_i"] != 36):
                raise ValueError("sample plan is not first-frame anchored")
            previous = timestamp
        reconstructed_camera = camera_contract({
            "pallet_size_m": {
                "width": session["camera"]["dimensions_wdh_m"][0],
                "length": session["camera"]["dimensions_wdh_m"][1],
                "height": session["camera"]["dimensions_wdh_m"][2],
            },
            "intrinsics": {
                "fx": session["camera"]["K"][0][0],
                "fy": session["camera"]["K"][1][1],
                "ppx": session["camera"]["K"][0][2],
                "ppy": session["camera"]["K"][1][2],
                "height": session["camera"]["image_hw"][0],
                "width": session["camera"]["image_hw"][1],
                "model": session["camera"]["distortion_model"],
                "coeffs": session["camera"]["distortion_coefficients"],
            },
        })
        if reconstructed_camera != session["camera"]:
            raise ValueError("stored camera/PnP contract drift")
    if plan.get("capture_bindings") != _capture_bindings():
        raise ValueError("capture artifact bindings drift")
    if plan.get("legacy_bindings") != public_legacy_bindings():
        raise ValueError("legacy bindings drift")
    if verify_files:
        rebuilt = build_plan(capture_root, verify=True)
        if rebuilt != dict(plan):
            raise ValueError("plan no longer reproduces from bound raw inputs")
    return dict(plan)


def freeze_plan(capture_root: Path | str | None = None) -> tuple[dict, bool]:
    candidate = build_plan(capture_root, verify=True)
    reused = PLAN_PATH.is_file()
    if reused:
        existing = C.read(PLAN_PATH)
        validate_plan(existing, verify_files=False)
        if existing != candidate:
            raise RuntimeError("existing frozen lifter plan differs from bound inputs")
    else:
        C.write(PLAN_PATH, candidate, freeze=True)
    return candidate, reused


def _load_file_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_LEGACY_MODULE_CACHE: dict[str, ModuleType] | None = None


def load_legacy_modules() -> dict[str, ModuleType]:
    """Load the exact old functions under isolated names, without a GPU call."""
    global _LEGACY_MODULE_CACHE
    if _LEGACY_MODULE_CACHE is not None:
        return _LEGACY_MODULE_CACHE
    paths = verify_legacy_bindings()
    private = "pallet_n3_lifter_legacy_"
    aliases = ("common", "dcp_env", "generic_point_refiner",
               "point_inference", "refiner", "inference",
               "run_pose_evaluation")
    saved = {name: sys.modules.get(name) for name in aliases}
    before_path = list(sys.path)
    try:
        sys.path[:0] = [str(DCP_CODE), str(OLD_CODE), str(LINE_CODE), str(C.ROOT)]
        old_common = _load_file_module(
            private + "common", OLD_CODE / "common.py")
        sys.modules["common"] = old_common
        environment = _load_file_module(private + "dcp_env", paths["dcp_env"])
        sys.modules["dcp_env"] = environment
        generic = _load_file_module(
            private + "generic_point_refiner", paths["generic_point_refiner"])
        sys.modules["generic_point_refiner"] = generic
        point = _load_file_module(private + "point_inference", paths["point_inference"])
        sys.modules["point_inference"] = point
        refiner = _load_file_module(private + "refiner", paths["refiner"])
        sys.modules["refiner"] = refiner
        inference = _load_file_module(private + "inference", paths["inference"])
        sys.modules["inference"] = inference
        pnp_solver = _load_file_module(private + "pnp_solver", paths["pnp_solver"])
        sys.modules["run_pose_evaluation"] = pnp_solver
        pose = _load_file_module(private + "pose", paths["pose"])
    finally:
        sys.path[:] = before_path
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    expected_files = {
        "environment": paths["dcp_env"], "inference": paths["inference"],
        "refiner": paths["refiner"], "point": paths["point_inference"],
        "pose": paths["pose"], "pnp_solver": paths["pnp_solver"],
    }
    modules = {
        "environment": environment, "inference": inference,
        "refiner": refiner, "point": point, "pose": pose,
        "pnp_solver": pnp_solver,
    }
    for key, expected in expected_files.items():
        if Path(modules[key].__file__).resolve() != expected.resolve():
            raise RuntimeError(f"legacy module path drift: {key}")
    for name in ("load_head", "predict_captured", "registry_input", "serial"):
        if not callable(getattr(inference, name, None)):
            raise RuntimeError(f"historical inference function missing: {name}")
    if not callable(getattr(pose, "infer", None)):
        raise RuntimeError("historical PnP function missing")
    _LEGACY_MODULE_CACHE = modules
    return modules


@dataclass
class LegacyStack:
    extractor: Any
    head: Any
    inference: ModuleType
    pose: ModuleType
    dimensions_wdh: np.ndarray
    symmetry_order: int
    temperature: float
    rule: dict
    normalization: dict
    checkpoint: Path

    def close(self) -> None:
        self.extractor.close()


def load_legacy_stack(device: str = "cuda") -> LegacyStack:
    """Instantiate strict historical R0 + N3 seed 1 after all locks pass."""
    if device != "cuda":
        raise ValueError("historical load_head is CUDA-only; CPU is fixture-test only")
    paths = verify_legacy_bindings()
    selected = fixed_selection_contract()
    modules = load_legacy_modules()
    environment = modules["environment"]
    # In an isolated Git worktree large ignored files may live in the common
    # checkout.  Preserve the old loader while pointing its frozen roots at the
    # hash-identical resolved artifacts.
    environment.R0 = paths["yolo_r0"]
    environment.RAW = paths["n3_checkpoint"].parents[2]
    head, checkpoint = modules["inference"].load_head(ARM, SEED)
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != paths["n3_checkpoint"].resolve():
        raise RuntimeError("historical load_head resolved a different checkpoint")
    features = environment.old("features")
    if Path(features.__file__).resolve() != paths["features"].resolve():
        raise RuntimeError("historical YOLO feature extractor path drift")
    extractor = features.FrozenYoloFeatures(paths["yolo_r0"], device=device)
    dimensions, order = modules["inference"].registry_input(OBJECT_TYPE)
    dimensions = np.asarray(dimensions, np.float64)
    if not np.array_equal(dimensions, EXPECTED_DIMENSIONS_WDH_M) or order != 4:
        extractor.close()
        raise RuntimeError("square registry dimensions/symmetry drift")
    return LegacyStack(
        extractor=extractor, head=head, inference=modules["inference"],
        pose=modules["pose"], dimensions_wdh=dimensions,
        symmetry_order=int(order), temperature=float(selected["temperature"]),
        rule=dict(selected["rule"]), normalization=selected["normalization"],
        checkpoint=checkpoint,
    )


def _selected_points(result: Mapping[str, Any]) -> np.ndarray | None:
    selected = result.get("selected_index")
    candidates = result.get("candidates")
    if selected is None:
        return None
    if (not isinstance(selected, int) or not isinstance(candidates, Sequence)
            or not 0 <= selected < len(candidates)):
        raise ValueError("invalid selected candidate")
    points = np.asarray(candidates[selected]["keypoints_xy"], np.float64)
    if points.shape != (9, 2):
        raise ValueError("selected candidate is not nine keypoints")
    return points


def _pose_prediction(result: Mapping[str, Any], pose_module: Any,
                     camera: Mapping[str, Any]) -> dict:
    points = _selected_points(result)
    pose = pose_module.infer(
        points, np.asarray(camera["K"], np.float64),
        np.asarray(camera["pnp_xyz_m"], np.float64), source=False)
    available = bool(pose.get("available", False))
    output = {
        "available": available,
        "fresh": available,
        "detected": points is not None,
        "selected_index": result.get("selected_index"),
        "head_used": bool(result.get("head_used", False)),
        "points_xy": _nullable(points),
        "pos_x_m": None,
        "pos_y_m": None,
        "pos_z_m": None,
        "yaw_deg": None,
        "pose": {"available": False},
    }
    if not available:
        return output
    rotation = np.asarray(pose.get("R_physical"), np.float64)
    translation = np.asarray(pose.get("centroid"), np.float64)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or translation[2] <= 0):
        raise ValueError("historical PnP returned invalid available pose")
    yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
    output.update({
        "pos_x_m": float(translation[0]),
        "pos_y_m": float(translation[1]),
        "pos_z_m": float(translation[2]),
        "yaw_deg": float(yaw),
        "pose": _nullable(pose),
    })
    return output


def _movement_summary(before: np.ndarray | None,
                      after: np.ndarray | None) -> dict:
    if before is None or after is None:
        return {
            "comparable_corners": 0, "mean_move_px": None,
            "max_move_px": None, "center_preserved": before is None and after is None,
            "finite_mask_preserved": before is None and after is None,
        }
    valid_before = np.isfinite(before).all(-1)
    valid_after = np.isfinite(after).all(-1)
    common = valid_before[:8] & valid_after[:8]
    moves = np.linalg.norm(after[:8][common] - before[:8][common], axis=-1)
    return {
        "comparable_corners": int(common.sum()),
        "mean_move_px": float(moves.mean()) if len(moves) else None,
        "max_move_px": float(moves.max()) if len(moves) else None,
        "center_preserved": bool(np.array_equal(
            before[8], after[8], equal_nan=True)),
        "finite_mask_preserved": bool(np.array_equal(valid_before, valid_after)),
    }


def infer_shared_frame(image_bgr: np.ndarray, metadata: Mapping[str, Any],
                       camera: Mapping[str, Any], stack: LegacyStack) -> dict:
    """Run one base forward and derive both R0 and N3 prediction paths."""
    image = np.asarray(image_bgr)
    if image.shape != (*EXPECTED_IMAGE_HW, 3) or image.dtype != np.uint8:
        raise ValueError("raw lifter image must be uint8 BGR 480x640")
    if not np.array_equal(np.asarray(camera["dimensions_wdh_m"]),
                          stack.dimensions_wdh):
        raise ValueError("session dimensions differ from strict registry dimensions")
    captured = stack.extractor.predict(image)  # exactly one base forward
    base_result = {
        "candidates": stack.inference.serial(captured["candidates"]),
        "selected_index": captured["selected_index"],
        "head_used": False,
    }
    refined_result, diagnostic = stack.inference.predict_captured(
        stack.head, ARM, captured, stack.dimensions_wdh,
        stack.symmetry_order, stack.temperature, stack.rule,
        image.shape[:2], stack.normalization)
    base_points = _selected_points(base_result)
    refined_points = _selected_points(refined_result)
    output = {
        **dict(metadata),
        "in_view": None,
        "visible_state": "UNKNOWN_NOT_ANNOTATED",
        "base_forward_shared": True,
        "methods": {
            "R0": _pose_prediction(base_result, stack.pose, camera),
            "N3_seed1": _pose_prediction(refined_result, stack.pose, camera),
        },
        "n3": {
            "arm": ARM, "seed": SEED, "temperature": stack.temperature,
            "rule": dict(stack.rule),
            "movement": _movement_summary(base_points, refined_points),
            "diagnostic_present": diagnostic is not None,
        },
    }
    if not all(output["n3"]["movement"][key]
               for key in ("center_preserved", "finite_mask_preserved")):
        raise AssertionError("N3 violated point preservation")
    return output


def accuracy_x() -> dict:
    reason = "recordings have no independent synchronized pose/corner reference"
    return {
        "independent_ground_truth": False,
        "independent_accuracy": {"value": None, "status": "x", "reason": reason},
        "position_error_m": {"value": None, "status": "x", "reason": reason},
        "yaw_error_deg": {"value": None, "status": "x", "reason": reason},
        "deployed_csv_pose_used_as_reference": False,
    }


def _statistics_input(frames: Sequence[Mapping[str, Any]], method: str) -> list[dict]:
    return [{
        lifter.SENSOR_TIME_FIELD: row[lifter.SENSOR_TIME_FIELD],
        "in_view": None,
        "prediction": row["methods"][method],
    } for row in frames]


def _pooled_jitter(session_rows: Mapping[str, Sequence[Mapping[str, Any]]],
                   method: str, field: str, *, angular: bool,
                   fresh_only: bool) -> dict:
    values: list[float] = []
    steps: list[float] = []
    rates: list[float] = []
    zero_dt = 0
    for frames in session_rows.values():
        prepared = []
        for row in frames:
            prediction = row["methods"][method]
            available = bool(prediction.get("available", False))
            fresh = bool(prediction.get("fresh", available)) if available else False
            number = lifter._finite_number(prediction.get(field))
            prepared.append((float(row[lifter.SENSOR_TIME_FIELD]), available,
                             fresh, number))
        points = [(index, *entry) for index, entry in enumerate(prepared)
                  if entry[1] and (not fresh_only or entry[2]) and entry[3] is not None]
        values.extend(point[4] for point in points)
        for left, right in zip(points, points[1:]):
            if fresh_only:
                if any(not entry[1] for entry in prepared[left[0] + 1:right[0]]):
                    continue
            elif right[0] != left[0] + 1:
                continue
            dt_ms = right[1] - left[1]
            if dt_ms <= 0:
                zero_dt += 1
                continue
            delta = (lifter._angular_delta(left[4], right[4]) if angular
                     else right[4] - left[4])
            steps.append(abs(delta))
            rates.append(abs(delta) / (dt_ms / 1000.0))
    if angular and values:
        radians = np.radians(np.asarray(values, np.float64))
        sine = float(np.sin(radians).mean())
        cosine = float(np.cos(radians).mean())
        mean = math.degrees(math.atan2(sine, cosine))
        resultant = min(1., math.hypot(sine, cosine))
        variation = (math.degrees(math.sqrt(-2. * math.log(resultant)))
                     if 0. < resultant < 1. else (0. if resultant == 1. else None))
    elif values:
        mean = statistics.fmean(values)
        variation = statistics.pstdev(values)
    else:
        mean = variation = None
    return {
        "count": len(values), "mean": mean, "variation_std": variation,
        "adjacent_positive_dt_pairs": len(steps),
        "adjacent_zero_dt_pairs_skipped": zero_dt,
        "median_abs_step": lifter._percentile(steps, .5),
        "p90_abs_step": lifter._percentile(steps, .9),
        "median_abs_rate_per_s": lifter._percentile(rates, .5),
        "p90_abs_rate_per_s": lifter._percentile(rates, .9),
        "angular_wrap_applied": angular,
        "wrap_period_deg": 360.0 if angular else None,
        "session_boundaries_never_joined": True,
    }


def summarize(session_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict:
    """Summarize both methods without joining temporal pairs across sessions."""
    if list(session_rows) != list(lifter.USABLE_SESSION_IDS):
        raise ValueError("summary requires all four fixed sessions in order")
    output = {}
    for method in METHODS:
        per_session = {
            session_id: lifter.stability_statistics(
                _statistics_input(frames, method))
            for session_id, frames in session_rows.items()
        }
        total = sum(row["coverage"]["frames"] for row in per_session.values())
        available = sum(row["coverage"]["available_outputs"]
                        for row in per_session.values())
        fresh = sum(row["coverage"]["fresh_predictions"]
                    for row in per_session.values())
        held = sum(row["coverage"]["held_previous_outputs"]
                   for row in per_session.values())
        missing = sum(row["missing"]["frames"] for row in per_session.values())
        candidates = []
        for session_id, row in per_session.items():
            candidates.append({
                "session_id": session_id,
                "frames": row["missing"]["longest_run_frames"],
                "observed_span_s": row["missing"]["longest_observed_span_s"],
                "until_next_sample_s": row["missing"]["longest_until_next_sample_s"],
            })
        longest_observed = max(candidates, key=lambda row: (
            row["observed_span_s"], row["frames"], row["session_id"]))
        finite_until = [row for row in candidates
                        if row["until_next_sample_s"] is not None]
        longest_until = (max(finite_until, key=lambda row: (
            row["until_next_sample_s"], row["frames"], row["session_id"]))
                         if finite_until else None)
        output[method] = {
            "overall": {
                "coverage": {
                    "frames": total,
                    "available_outputs": available,
                    "available_fraction": available / total if total else None,
                    "fresh_predictions": fresh,
                    "fresh_fraction": fresh / total if total else None,
                    "held_previous_outputs": held,
                    "visibility_unknown_frames": total,
                },
                "missing": {
                    "frames": missing,
                    "fraction": missing / total if total else None,
                    "run_count": sum(row["missing"]["run_count"]
                                     for row in per_session.values()),
                    "longest_observed_sensor_time_run": longest_observed,
                    "longest_until_next_sample_sensor_time_run": longest_until,
                    "sampled_timeline_not_full_frame_timeline": True,
                },
                "yaw_wrap_jitter": {
                    "available_outputs_including_held": _pooled_jitter(
                        session_rows, method, "yaw_deg", angular=True,
                        fresh_only=False),
                    "fresh_predictions_only": _pooled_jitter(
                        session_rows, method, "yaw_deg", angular=True,
                        fresh_only=True),
                    "square_C4_representative_jumps_not_smoothed": True,
                },
            },
            "per_session": per_session,
        }
    return output


def _validate_accuracy_x(value: Mapping[str, Any]) -> None:
    if (value.get("independent_ground_truth") is not False
            or value.get("deployed_csv_pose_used_as_reference") is not False):
        raise ValueError("lifter accuracy reference contract drift")
    for key in ("independent_accuracy", "position_error_m", "yaw_error_deg"):
        row = value.get(key, {})
        if row.get("status") != "x" or row.get("value") is not None:
            raise ValueError(f"unsupported lifter accuracy field populated: {key}")


def _validate_completed_session(session_id: str, frames: Any,
                                plan: Mapping[str, Any]) -> None:
    expected = plan["sessions"][session_id]["frames"]
    if not isinstance(frames, list) or len(frames) != len(expected):
        raise ValueError(f"completed session length drift: {session_id}")
    for index, (row, identity) in enumerate(zip(frames, expected)):
        if any(row.get(key) != identity[key] for key in (
                "session_id", "video_index", "frame_i", "camera_frame_number",
                lifter.SENSOR_TIME_FIELD, "camera_timestamp_domain")):
            raise ValueError(f"frame identity drift: {session_id}:{index}")
        if (row.get("in_view") is not None
                or row.get("visible_state") != "UNKNOWN_NOT_ANNOTATED"
                or row.get("base_forward_shared") is not True
                or set(row.get("methods", {})) != set(METHODS)):
            raise ValueError(f"frame output contract drift: {session_id}:{index}")
        for method in METHODS:
            prediction = row["methods"][method]
            if not isinstance(prediction, Mapping):
                raise ValueError("method prediction must be a mapping")
            available = bool(prediction.get("available", False))
            if prediction.get("fresh") is not available:
                raise ValueError("offline fresh state must equal new pose availability")
            values = [prediction.get(key) for key in (
                "pos_x_m", "pos_y_m", "pos_z_m", "yaw_deg")]
            if available:
                numeric = np.asarray(values, np.float64)
                if (not np.isfinite(numeric).all() or numeric[2] <= 0
                        or prediction.get("pose", {}).get("available") is not True):
                    raise ValueError("available method pose is invalid")
            elif (any(value is not None for value in values)
                  or prediction.get("pose", {}).get("available") is not False):
                raise ValueError("unavailable method pose contains numeric output")
        if row["methods"]["R0"].get("head_used") is not False:
            raise ValueError("R0 path unexpectedly used a correction head")
        n3 = row.get("n3", {})
        movement = n3.get("movement", {})
        if (n3.get("arm") != ARM or n3.get("seed") != SEED
                or n3.get("temperature") != 1.0
                or n3.get("rule") != {
                    "lam": 1.0, "max_move_image_diagonal_fraction": .01}
                or movement.get("center_preserved") is not True
                or movement.get("finite_mask_preserved") is not True):
            raise ValueError("N3 frame contract/preservation drift")


def validate_raw(payload: Mapping[str, Any], plan: Mapping[str, Any],
                 *, verify_files: bool = True) -> dict:
    if (payload.get("schema") != RAW_SCHEMA or payload.get("complete") is not True
            or payload.get("plan_sha256") != _json_hash(plan)
            or payload.get("config") != plan["config"]
            or payload.get("methods") != list(METHODS)
            or payload.get("base_forward_count") != plan["total_sampled_frames"]
            or payload.get("shared_base_forward_count") != plan["total_sampled_frames"]
            or payload.get("n3_call_count") != plan["total_sampled_frames"]
            or payload.get("actual_control_invoked") is not False
            or payload.get("tracking_or_smoothing_applied") is not False):
        raise ValueError("lifter raw identity/execution contract drift")
    if payload.get("fixed_selection") != fixed_selection_contract():
        raise ValueError("lifter raw fixed selection drift")
    sessions = payload.get("sessions")
    if not isinstance(sessions, Mapping) or list(sessions) != list(lifter.USABLE_SESSION_IDS):
        raise ValueError("lifter raw sessions drift")
    for session_id in lifter.USABLE_SESSION_IDS:
        _validate_completed_session(session_id, sessions[session_id]["frames"], plan)
        if sessions[session_id].get("camera") != plan["sessions"][session_id]["camera"]:
            raise ValueError("session camera contract drift")
    expected_stats = summarize({sid: sessions[sid]["frames"]
                                for sid in lifter.USABLE_SESSION_IDS})
    if payload.get("statistics") != expected_stats:
        raise ValueError("stored lifter statistics do not reproduce from raw rows")
    _validate_accuracy_x(payload.get("accuracy", {}))
    if payload.get("legacy_bindings") != public_legacy_bindings():
        raise ValueError("raw legacy bindings drift")
    if verify_files:
        verify_legacy_bindings()
        validate_plan(plan, verify_files=True)
    return dict(payload)


def _progress_identity(plan: Mapping[str, Any]) -> dict:
    return {
        "schema": RAW_SCHEMA,
        "complete": False,
        "plan_sha256": _json_hash(plan),
        "config": plan["config"],
        "methods": list(METHODS),
        "legacy_bindings": public_legacy_bindings(),
        "fixed_selection": fixed_selection_contract(),
        "sessions": {},
        "actual_control_invoked": False,
        "tracking_or_smoothing_applied": False,
    }


def _load_progress(plan: Mapping[str, Any]) -> dict:
    expected = _progress_identity(plan)
    if not RAW_PATH.is_file():
        return expected
    value = C.read(RAW_PATH)
    if value.get("complete") is True:
        validate_raw(value, plan, verify_files=True)
        return value
    for key in expected:
        if key == "sessions":
            continue
        if value.get(key) != expected[key]:
            raise RuntimeError(f"partial lifter run identity drift: {key}")
    sessions = value.get("sessions")
    if not isinstance(sessions, Mapping):
        raise RuntimeError("partial lifter sessions are invalid")
    completed = list(sessions)
    if completed != list(lifter.USABLE_SESSION_IDS)[:len(completed)]:
        raise RuntimeError("partial lifter sessions are not a prefix")
    for session_id in completed:
        _validate_completed_session(session_id, sessions[session_id]["frames"], plan)
    return value


def run(capture_root: Path | str | None = None) -> tuple[dict, bool]:
    """Execute or resume the full fixed offline replay on the CUDA machine."""
    plan, _ = freeze_plan(capture_root)
    progress = _load_progress(plan)
    if progress.get("complete") is True:
        _write_receipt(progress)
        return progress, True
    snapshot = C.gpu_snapshot(refuse_other_compute=True)
    progress.setdefault("hardware", snapshot)
    root = lifter.discover_capture_root(capture_root)
    stack = load_legacy_stack("cuda")
    try:
        for session_id in lifter.USABLE_SESSION_IDS:
            if session_id in progress["sessions"]:
                continue
            camera = plan["sessions"][session_id]["camera"]
            expected = plan["sessions"][session_id]["frames"]
            frames = []
            iterator = lifter.iter_offline_frames(
                session_id, root, sample_interval_ms=SAMPLE_INTERVAL_MS,
                verify=False)
            for index, sample in enumerate(iterator):
                if index >= len(expected) or sample.metadata() != expected[index]:
                    raise RuntimeError(f"decoded sample membership drift: {session_id}:{index}")
                frames.append(infer_shared_frame(
                    sample.image_bgr, sample.metadata(), camera, stack))
            if len(frames) != len(expected):
                raise RuntimeError(f"decoded sample count drift: {session_id}")
            progress["sessions"][session_id] = {
                "camera": camera, "frames": frames,
            }
            # A completed session is the restart unit; no frame is selected or
            # discarded based on its predictions.
            C.write(RAW_PATH, progress)
    finally:
        stack.close()
    rows = {sid: progress["sessions"][sid]["frames"]
            for sid in lifter.USABLE_SESSION_IDS}
    total = plan["total_sampled_frames"]
    progress.update({
        "complete": True,
        "completed_at": C.now(),
        "base_forward_count": total,
        "shared_base_forward_count": total,
        "n3_call_count": total,
        "statistics": summarize(rows),
        "accuracy": accuracy_x(),
        "reference": {
            "visible_state": "UNKNOWN_NOT_ANNOTATED",
            "stationary_intervals": {"value": None, "status": "x"},
            "independent_pose_or_corner_reference": False,
            "deployed_csv_role": "operational output only; never accuracy reference",
        },
    })
    validate_raw(progress, plan, verify_files=True)
    C.write(RAW_PATH, progress)
    _write_receipt(progress)
    return progress, False


def _write_receipt(payload: Mapping[str, Any]) -> dict:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "complete": True,
        "raw": C.binding(RAW_PATH),
        "plan": C.binding(PLAN_PATH),
        "sampled_frames": int(payload["base_forward_count"]),
        "sessions": list(lifter.USABLE_SESSION_IDS),
        "methods": list(METHODS),
        "base_forward_shared_once_per_frame": True,
        "statistics": payload["statistics"],
        "accuracy": payload["accuracy"],
        "actual_control_invoked": False,
    }
    C.write(RECEIPT_PATH, receipt)
    return receipt


def verify(capture_root: Path | str | None = None) -> dict:
    plan = C.read(PLAN_PATH)
    validate_plan(plan, verify_files=True, capture_root=capture_root)
    raw = C.read(RAW_PATH)
    validate_raw(raw, plan, verify_files=True)
    receipt = C.read(RECEIPT_PATH)
    if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("complete") is not True
            or receipt.get("raw") != C.binding(RAW_PATH)
            or receipt.get("plan") != C.binding(PLAN_PATH)
            or receipt.get("sampled_frames") != sum(EXPECTED_SAMPLE_COUNTS.values())
            or receipt.get("actual_control_invoked") is not False
            or receipt.get("accuracy") != accuracy_x()
            or receipt.get("statistics") != raw["statistics"]):
        raise RuntimeError("lifter completion receipt drift")
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fixed offline R0/N3 replay of four recorded lifter sessions")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "verify"):
        command = sub.add_parser(name)
        command.add_argument(
            "--capture-root", type=Path, default=None,
            help="capture directory or repository root; default uses strict discovery")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "plan":
        plan, reused = freeze_plan(args.capture_root)
        print(json.dumps({
            "status": "REUSED" if reused else "CREATED",
            "path": str(PLAN_PATH),
            "sampled_frames": plan["total_sampled_frames"],
            "plan_sha256": _json_hash(plan),
        }, ensure_ascii=False, indent=2))
    elif args.command == "run":
        result, reused = run(args.capture_root)
        print(json.dumps({
            "status": "REUSED" if reused else "COMPLETE",
            "path": str(RAW_PATH),
            "sampled_frames": result["base_forward_count"],
            "receipt": str(RECEIPT_PATH),
        }, ensure_ascii=False, indent=2))
    else:
        receipt = verify(args.capture_root)
        print(json.dumps({"status": "VERIFIED", "receipt": receipt},
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
