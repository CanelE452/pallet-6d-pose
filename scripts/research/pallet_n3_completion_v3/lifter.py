"""Read-only, offline evaluation support for the recorded forklift sessions.

The recordings are evidence for temporal coverage and stability only.  Their
CSV pose columns are outputs of the deployed YOLO model, not an independent
reference.  Consequently this module deliberately has no accuracy evaluator
and exposes no vehicle-control surface.

The raw MP4 writers started after 35 timing rows.  For every usable recording,
raw video frame zero therefore maps to timing row 35 (``frame_i == 36``), and
the final video frame maps to the final timing row.  Sampling is based on the
recorded RealSense ``camera_sensor_timestamp_ms`` values; container FPS and the
30 FPS stream setting are inventory diagnostics only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from . import common as C


CAPTURE_RELATIVE = Path(
    "challenge/data/01_real/_live_captures/forklift_v4_20260901")
CAPTURE_ENV = "PALLET_LIFTER_CAPTURE_ROOT"
PREFIX = "forklift_v4_recording_20260901_"
SESSION_IDS = ("173507", "174126", "174342", "174925", "175419")
USABLE_SESSION_IDS = SESSION_IDS[:4]
CORRUPT_SESSION_IDS = ("175419",)

SENSOR_TIME_FIELD = "camera_sensor_timestamp_ms"
VIDEO_TIMING_ROW_OFFSET = 35
INDEPENDENT_GROUND_TRUTH = False
ALLOWED_OFFLINE_METRICS = frozenset({"coverage", "missing", "jitter"})

FILE_SUFFIXES = {
    "raw_mp4": "_raw.mp4",
    "meta": "_meta.json",
    "state": "_pallet_state.csv",
    "timing": "_inference_timing.csv",
    "control": "_control_seq.jsonl",
}

STATE_FIELDS = (
    "t_iso", "t_mono", "frame_i", "fsm_state", "align_sub", "det_ok",
    "pos_x", "pos_y", "pos_z", "yaw_deg", "yaw_model_raw_deg",
    "width_m", "fps", "rot_stop_command", "rot_stop_mode",
    "rot_stop_measured_rate_deg_s", "rot_stop_effective_rate_deg_s",
    "rot_stop_error_deg", "rot_post_stop_deg",
    "rot_post_stop_peak_abs_deg",
)
TIMING_FIELDS = (
    "t_iso", "frame_i", "fsm_state", "align_sub", "camera_frame_number",
    "camera_frame_delta", SENSOR_TIME_FIELD, "camera_sensor_interval_ms",
    "camera_timestamp_domain", "camera_input_host_mono_ms",
    "camera_input_interval_ms", "inference_ran",
    "inference_start_host_mono_ms", "inference_end_host_mono_ms",
    "model_inference_ms", "input_to_inference_end_ms",
    "pose_result_host_mono_ms", "input_to_pose_result_ms", "model_det_ok",
    "pnp_ok", "yaw_deg", "pos_x_m", "pos_z_m", "center_bearing_deg",
)
CONTROL_REQUIRED_FIELDS = frozenset(
    {"phase", "t_iso", "t_mono", "step"})


# Immutable bindings for the 25 artifacts used by the offline case study.
# The rendered/overlay MP4 is intentionally excluded: evaluation consumes raw RGB.
EXPECTED_FILES = {
    "173507": {
        "raw_mp4": (28387497, "5c552de6207ec5b959e1f4ea3fec7f826627835dc5c29fbedb353eabed41e80a"),
        "meta": (3896, "28161773188905f3324b09b3921dc040befe7b16e3826f77dd04d32746e48720"),
        "state": (351010, "58c1f6bafaec9d92ad744ea74eb4852e3d56a51d36cf7787cb9d32e82b4639d6"),
        "timing": (780529, "bce16e1bd9747dae98312c66be2d54a7e3bb2d36c5773ad3f0d52a22a67cbd29"),
        "control": (8761030, "d2154b388935324129ecb1c902574c1db109cd51e7f0c3ab0b61805e0331fe8e"),
    },
    "174126": {
        "raw_mp4": (5514064, "e58f35d026ee35a5bd012673412b58199a48af12b7cafeb0e26a7c60420ce8ad"),
        "meta": (3896, "540cdd52c103671e716a8eef44729827f86ba3b3f0c0f5546a4ef97fb48c8a45"),
        "state": (42839, "74ad8e110e9e1a7d5688140bc0890a18f5e9bfc55f4fb7d06958b2bb77d7506e"),
        "timing": (168980, "943e7405260117308cdf847830c2bf38c7afd28198f0136a09db266810b5e2ba"),
        "control": (1930374, "565ba623d178273808e035701909f14479a63a3008ecfde01c1acada8effa40f"),
    },
    "174342": {
        "raw_mp4": (12975291, "4e69cd2339bf06f69ad43fc60dbaac276818d3bcd8b6dbdf0bf52375eed62d6f"),
        "meta": (3896, "b228eadc62380f0ceac13fb08f81b8ce664d1bb91cbc19d38cc37fb62925dfc0"),
        "state": (225627, "596fba23fb9338ed9a09314fdc4c534ca45a276ced700ea00494b236ca992b9c"),
        "timing": (600223, "83de3c6f89160057da806cb1d998c2fa1bdd445bbcc3d8ec154a8226d18a267f"),
        "control": (6268148, "11d3e8c963ec3edf5d210b1c0901bdf1cbb32d3eb633ec575aa8ea7d3f113eeb"),
    },
    "174925": {
        "raw_mp4": (13110861, "1b796bd9cc10e76908d7132193897b3f7e23c0dd07f40b55360492bd1df8aa92"),
        "meta": (3896, "dcdac2d49e1e8df80bf4cc73c95fc7ceab26a819b436c8b2e3cd9f1b7d1dd543"),
        "state": (156059, "4a4e9544ec94bc121baad8db52e246ab3c984aec5792d366ed1c9733bfd7448a"),
        "timing": (458116, "0fed7181200508ba7e7eba04f0da8e78fc6e286a38dddf9278d82ac657eb90ad"),
        "control": (4806016, "c8598f19c97ff6248246ea5f543a4263b77ac83b47a32b2d697e05e2dbb61d94"),
    },
    "175419": {
        "raw_mp4": (3932204, "01c21db9159b189c85bbd9582cf7a164c83ebdab3c54681b6eec3056e0f50317"),
        "meta": (3896, "eca2457afe34f188c6f76b8250fd8e42b278090a01956d95164da8c88c9e3727"),
        "state": (44253, "aa55abb9402ab94479c9ddde7d5ac395d9065f199bf3106fcff29c133369a4a4"),
        "timing": (109519, "22331ba44f08d9c4990f0b478d07eaf8fdbca82e53e4e7afe77480f9b4365595"),
        "control": (1557054, "6cb900eff26545b83bd1e15758f9d4d8d39bac7b5ff1692d1c2757389652cb1a"),
    },
}

EXPECTED_FACTS = {
    "173507": dict(video_frames=3729, state_rows=2146, timing_rows=3764,
                   control_rows=18154, video_open=True),
    "174126": dict(video_frames=757, state_rows=437, timing_rows=792,
                   control_rows=4018, video_open=True),
    "174342": dict(video_frames=2501, state_rows=1465, timing_rows=2536,
                   control_rows=13014, video_open=True),
    "174925": dict(video_frames=1923, state_rows=1121, timing_rows=1958,
                   control_rows=9993, video_open=True),
    "175419": dict(video_frames=0, state_rows=355, timing_rows=630,
                   control_rows=3233, video_open=False),
}


def _candidate_capture_root(value: Path | str) -> Path:
    path = Path(value).expanduser().resolve()
    nested = path / CAPTURE_RELATIVE
    return nested if nested.is_dir() else path


def discover_capture_root(explicit: Path | str | None = None) -> Path:
    """Locate captures in either this worktree or its Git common checkout."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(_candidate_capture_root(explicit))
    configured = os.environ.get(CAPTURE_ENV)
    if configured:
        candidates.append(_candidate_capture_root(configured))
    candidates.append(C.ROOT / CAPTURE_RELATIVE)

    # Untracked large data lives in the main checkout and is absent from an
    # isolated worktree.  Resolve that checkout through .git/worktrees/.../commondir.
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
                candidates.append(common_dir.parent / CAPTURE_RELATIVE)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.is_dir():
            return candidate
        seen.add(candidate)
    rendered = ", ".join(str(value) for value in candidates)
    raise FileNotFoundError(f"forklift capture root not found; checked: {rendered}")


def session_paths(capture_root: Path | str, session_id: str) -> dict[str, Path]:
    if session_id not in SESSION_IDS:
        raise ValueError(f"session is outside the frozen set: {session_id!r}")
    root = Path(capture_root)
    stem = root / f"{PREFIX}{session_id}"
    return {key: Path(str(stem) + suffix)
            for key, suffix in FILE_SUFFIXES.items()}


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        observed = tuple(reader.fieldnames or ())
        if observed != tuple(expected_fields):
            raise ValueError(f"CSV schema drift in {path}: {observed!r}")
        return list(reader)


def _read_timing_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path, TIMING_FIELDS)
    timestamps: list[float] = []
    for index, row in enumerate(rows, 1):
        if int(row["frame_i"]) != index:
            raise ValueError(f"timing frame_i is not contiguous at row {index}")
        value = float(row[SENSOR_TIME_FIELD])
        if not math.isfinite(value):
            raise ValueError(f"non-finite sensor timestamp at row {index}")
        timestamps.append(value)
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("camera sensor timestamps move backwards")
    return rows


def _file_inventory(path: Path, expected: tuple[int, str]) -> tuple[dict, list[str]]:
    expected_bytes, expected_sha = expected
    if not path.is_file():
        return ({"path": str(path), "exists": False, "expected_bytes": expected_bytes,
                 "expected_sha256": expected_sha}, [f"missing file: {path}"])
    observed_bytes = path.stat().st_size
    observed_sha = C.sha256(path)
    record = {
        "path": str(path), "exists": True,
        "bytes": observed_bytes, "sha256": observed_sha,
        "expected_bytes": expected_bytes, "expected_sha256": expected_sha,
        "binding_matches": observed_bytes == expected_bytes and observed_sha == expected_sha,
    }
    issues = []
    if observed_bytes != expected_bytes:
        issues.append(f"byte-size drift: {path}")
    if observed_sha != expected_sha:
        issues.append(f"sha256 drift: {path}")
    return record, issues


def _probe_video(path: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    container_fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    first_decodes = False
    last_decodes = False
    if opened and frames > 0:
        first_decodes, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
        last_decodes, _ = capture.read()
    capture.release()
    return {
        "opened": opened, "reported_frame_count": frames,
        "width": width, "height": height,
        "container_fps_diagnostic_only": container_fps,
        "first_frame_decodes": bool(first_decodes),
        "last_frame_decodes": bool(last_decodes),
    }


def _control_inventory(path: Path) -> dict:
    phases: Counter[str] = Counter()
    schema: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for rows, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{rows}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at {path}:{rows}")
            missing = CONTROL_REQUIRED_FIELDS.difference(value)
            if missing:
                raise ValueError(f"control schema missing {sorted(missing)} at row {rows}")
            if not math.isfinite(float(value["t_mono"])):
                raise ValueError(f"non-finite control t_mono at row {rows}")
            schema.update(value)
            phases[str(value["phase"])] += 1
    return {
        "rows": rows, "required_fields": sorted(CONTROL_REQUIRED_FIELDS),
        "union_fields": sorted(schema), "phase_counts": dict(sorted(phases.items())),
    }


def inventory_session(capture_root: Path | str, session_id: str,
                      *, strict: bool = True) -> dict:
    """Hash and validate one frozen session without invoking any controller."""
    paths = session_paths(capture_root, session_id)
    facts = EXPECTED_FACTS[session_id]
    files: dict[str, dict] = {}
    issues: list[str] = []
    for key, path in paths.items():
        files[key], file_issues = _file_inventory(path, EXPECTED_FILES[session_id][key])
        issues.extend(file_issues)

    if issues:
        result = {"session_id": session_id, "files": files, "issues": issues,
                  "status": "INVALID_BINDING"}
        if strict:
            raise RuntimeError("; ".join(issues))
        return result

    state_rows = _read_csv(paths["state"], STATE_FIELDS)
    timing_rows = _read_timing_rows(paths["timing"])
    control = _control_inventory(paths["control"])
    meta = json.loads(paths["meta"].read_text(encoding="utf-8-sig"))
    video = _probe_video(paths["raw_mp4"])

    if len(state_rows) != facts["state_rows"]:
        issues.append(f"state row count drift: {len(state_rows)}")
    if len(timing_rows) != facts["timing_rows"]:
        issues.append(f"timing row count drift: {len(timing_rows)}")
    if control["rows"] != facts["control_rows"]:
        issues.append(f"control row count drift: {control['rows']}")
    if video["opened"] != facts["video_open"]:
        issues.append(f"raw video open-state drift: {video['opened']}")
    if video["reported_frame_count"] != facts["video_frames"]:
        issues.append(f"raw video frame-count drift: {video['reported_frame_count']}")

    intrinsics = meta.get("intrinsics", {})
    if video["opened"]:
        if (video["width"], video["height"]) != (640, 480):
            issues.append(f"raw video dimensions drift: {video['width']}x{video['height']}")
        if not video["first_frame_decodes"] or not video["last_frame_decodes"]:
            issues.append("raw video opens but boundary frame decode failed")
        if len(timing_rows) - video["reported_frame_count"] != VIDEO_TIMING_ROW_OFFSET:
            issues.append("raw-video/timing startup offset is not 35 rows")
        mapped_first_frame_i = int(timing_rows[VIDEO_TIMING_ROW_OFFSET]["frame_i"])
        mapped_last_frame_i = int(timing_rows[-1]["frame_i"])
    else:
        mapped_first_frame_i = None
        mapped_last_frame_i = None

    if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
        issues.append("meta intrinsics dimensions are not 640x480")
    timing_contract = meta.get("inference_timing", {})
    declared_timebase = str(timing_contract.get("rotation_rate_timebase", ""))
    if SENSOR_TIME_FIELD not in declared_timebase:
        issues.append("meta does not prefer camera_sensor_timestamp_ms")
    if meta.get("stream_fps") != 30:
        issues.append("unexpected nominal stream_fps")

    sensor_times = [float(row[SENSOR_TIME_FIELD]) for row in timing_rows]
    duplicate_times = sum(right == left
                          for left, right in zip(sensor_times, sensor_times[1:]))
    timebase = {
        "source": "RealSense per-frame sensor timestamp",
        "field": SENSOR_TIME_FIELD,
        "timestamp_domain_values": sorted(
            {row["camera_timestamp_domain"] for row in timing_rows}),
        "units": "ms", "rows": len(sensor_times),
        "first_ms": sensor_times[0], "last_ms": sensor_times[-1],
        "duplicate_adjacent_timestamps": duplicate_times,
        "backward_timestamps": 0,
        "nominal_stream_fps": meta["stream_fps"],
        "nominal_fps_used_for_sampling_or_metrics": False,
        "video_timing_row_offset": VIDEO_TIMING_ROW_OFFSET if video["opened"] else None,
        "mapped_first_frame_i": mapped_first_frame_i,
        "mapped_last_frame_i": mapped_last_frame_i,
    }

    expected_status = ("CORRUPT_RAW_MP4" if session_id in CORRUPT_SESSION_IDS
                       else "USABLE")
    observed_status = ("CORRUPT_RAW_MP4" if not video["opened"] else "USABLE")
    if expected_status != observed_status:
        issues.append(f"session status drift: {observed_status}")
    result = {
        "session_id": session_id,
        "expected_status": expected_status,
        "status": "INVALID" if issues else observed_status,
        "files": files,
        "video": video,
        "schemas": {
            "state": {"fields": list(STATE_FIELDS), "rows": len(state_rows)},
            "timing": {"fields": list(TIMING_FIELDS), "rows": len(timing_rows)},
            "control": control,
        },
        "camera": {
            "model": intrinsics.get("model"),
            "fx": intrinsics.get("fx"), "fy": intrinsics.get("fy"),
            "ppx": intrinsics.get("ppx"), "ppy": intrinsics.get("ppy"),
            "width": intrinsics.get("width"), "height": intrinsics.get("height"),
        },
        "dimensions_m": meta.get("pallet_size_m"),
        "recording_model": meta.get("model"),
        "timebase": timebase,
        "reference": {
            "independent_ground_truth": False,
            "accuracy_metrics": "X_NO_INDEPENDENT_GROUND_TRUTH",
            "csv_pose_role": "deployed_model_output_not_reference",
        },
        "issues": issues,
    }
    if strict and issues:
        raise RuntimeError("; ".join(issues))
    return result


def inventory_captures(capture_root: Path | str | None = None,
                       *, strict: bool = True) -> dict:
    """Return a JSON-serializable inventory for the fixed five sessions."""
    root = discover_capture_root(capture_root)
    sessions = [inventory_session(root, session_id, strict=strict)
                for session_id in SESSION_IDS]
    usable = [row["session_id"] for row in sessions if row["status"] == "USABLE"]
    corrupt = [row["session_id"] for row in sessions
               if row["status"] == "CORRUPT_RAW_MP4"]
    result = {
        "capture_root": str(root), "sessions": sessions,
        "usable_session_ids": usable, "corrupt_session_ids": corrupt,
        "usable_video_frames": sum(
            row["video"]["reported_frame_count"] for row in sessions
            if row["status"] == "USABLE"),
        "timebase_contract": {
            "field": SENSOR_TIME_FIELD,
            "units": "ms", "nominal_fps_used": False,
            "sampling_rule": "recorded_sensor_timestamp",
        },
        "reference_contract": {
            "independent_ground_truth": False,
            "accuracy_metrics": "X_NO_INDEPENDENT_GROUND_TRUTH",
            "allowed_metrics": sorted(ALLOWED_OFFLINE_METRICS),
        },
    }
    if strict:
        if usable != list(USABLE_SESSION_IDS) or corrupt != list(CORRUPT_SESSION_IDS):
            raise RuntimeError("frozen usable/corrupt session partition changed")
        if result["usable_video_frames"] != 8910:
            raise RuntimeError("frozen usable frame total changed")
    return result


@dataclass(frozen=True)
class OfflineFrame:
    session_id: str
    video_index: int
    frame_i: int
    camera_frame_number: int
    camera_sensor_timestamp_ms: float
    camera_timestamp_domain: str
    image_bgr: Any
    timing: Mapping[str, str]

    def metadata(self) -> dict:
        return {
            "session_id": self.session_id,
            "video_index": self.video_index,
            "frame_i": self.frame_i,
            "camera_frame_number": self.camera_frame_number,
            SENSOR_TIME_FIELD: self.camera_sensor_timestamp_ms,
            "camera_timestamp_domain": self.camera_timestamp_domain,
        }


def _is_time_sample(timestamp_ms: float, next_due_ms: float | None,
                    interval_ms: float | None) -> tuple[bool, float | None]:
    if interval_ms is None:
        return True, None
    if next_due_ms is None:
        return True, timestamp_ms + interval_ms
    if timestamp_ms < next_due_ms:
        return False, next_due_ms
    while next_due_ms <= timestamp_ms:
        next_due_ms += interval_ms
    return True, next_due_ms


def iter_offline_frames(
        session_id: str, capture_root: Path | str | None = None, *,
        sample_interval_ms: float | None = None,
        start_sensor_timestamp_ms: float | None = None,
        stop_sensor_timestamp_ms: float | None = None,
        limit: int | None = None, verify: bool = True) -> Iterator[OfflineFrame]:
    """Yield raw frames aligned to recorded sensor time for offline inference.

    ``sample_interval_ms`` is an absolute sensor-time grid anchored on the first
    eligible frame.  It never derives a stride from either MP4 FPS or meta
    ``stream_fps``.  Equal adjacent sensor timestamps are retained when every
    frame is requested and naturally coalesce under time-based sampling.
    """
    if session_id not in USABLE_SESSION_IDS:
        raise ValueError(f"no usable raw video for frozen session {session_id!r}")
    if sample_interval_ms is not None and (
            not math.isfinite(sample_interval_ms) or sample_interval_ms <= 0):
        raise ValueError("sample_interval_ms must be finite and positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    root = discover_capture_root(capture_root)
    if verify:
        inventory_session(root, session_id, strict=True)
    paths = session_paths(root, session_id)
    timing_rows = _read_timing_rows(paths["timing"])
    mapped_rows = timing_rows[VIDEO_TIMING_ROW_OFFSET:]
    if len(mapped_rows) != EXPECTED_FACTS[session_id]["video_frames"]:
        raise RuntimeError("video/timing mapping no longer satisfies the frozen offset")

    import cv2

    capture = cv2.VideoCapture(str(paths["raw_mp4"]))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"raw video did not open: {paths['raw_mp4']}")
    yielded = 0
    next_due_ms: float | None = None
    try:
        if limit == 0:
            return
        for video_index, timing in enumerate(mapped_rows):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"raw video decode stopped at frame {video_index}")
            timestamp_ms = float(timing[SENSOR_TIME_FIELD])
            if (start_sensor_timestamp_ms is not None and
                    timestamp_ms < start_sensor_timestamp_ms):
                continue
            if (stop_sensor_timestamp_ms is not None and
                    timestamp_ms > stop_sensor_timestamp_ms):
                break
            selected, next_due_ms = _is_time_sample(
                timestamp_ms, next_due_ms, sample_interval_ms)
            if not selected:
                continue
            yield OfflineFrame(
                session_id=session_id,
                video_index=video_index,
                frame_i=int(timing["frame_i"]),
                camera_frame_number=int(timing["camera_frame_number"]),
                camera_sensor_timestamp_ms=timestamp_ms,
                camera_timestamp_domain=timing["camera_timestamp_domain"],
                image_bgr=image,
                timing=timing,
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                break
    finally:
        capture.release()


def iter_offline_predictions(
        session_id: str,
        predictor: Callable[[Any, Mapping[str, Any]], Mapping[str, Any] | None],
        capture_root: Path | str | None = None, **frame_options) -> Iterator[dict]:
    """Run a caller-supplied predictor over offline frames and yield records.

    The callback receives ``(image_bgr, metadata)``.  This function does not
    import or call any CAN, FSM, actuator, or vehicle-control implementation.
    """
    if not callable(predictor):
        raise TypeError("predictor must be callable")
    for sample in iter_offline_frames(
            session_id, capture_root, **frame_options):
        metadata = sample.metadata()
        prediction = predictor(sample.image_bgr, metadata)
        if prediction is not None and not isinstance(prediction, Mapping):
            raise TypeError("predictor must return a mapping or None")
        yield {**metadata, "prediction": prediction}


def validate_requested_metrics(metric_names: Iterable[str]) -> tuple[str, ...]:
    """Reject accuracy/error requests because the recordings have no true GT."""
    names = tuple(metric_names)
    unsupported = set(names).difference(ALLOWED_OFFLINE_METRICS)
    if unsupported:
        raise ValueError(
            "recorded lifter sessions lack independent ground truth; "
            f"unsupported metrics: {sorted(unsupported)}")
    return names


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _angular_delta(left: float, right: float) -> float:
    return (right - left + 180.0) % 360.0 - 180.0


def _value_summary(values: Sequence[float], *, angular: bool) -> dict:
    if not values:
        return {"count": 0, "mean": None, "variation_std": None}
    if angular:
        radians = [math.radians(value) for value in values]
        mean_sin = statistics.fmean(math.sin(value) for value in radians)
        mean_cos = statistics.fmean(math.cos(value) for value in radians)
        mean = math.degrees(math.atan2(mean_sin, mean_cos))
        resultant = min(1.0, math.hypot(mean_sin, mean_cos))
        variation = (math.degrees(math.sqrt(-2.0 * math.log(resultant)))
                     if 0.0 < resultant < 1.0 else (0.0 if resultant == 1.0 else None))
    else:
        mean = statistics.fmean(values)
        variation = statistics.pstdev(values)
    return {"count": len(values), "mean": mean, "variation_std": variation}


def _jitter_for_field(entries: Sequence[dict], field: str, *, angular: bool,
                      fresh_only: bool) -> dict:
    points: list[tuple[int, float, float]] = []
    for index, entry in enumerate(entries):
        if not entry["available"] or (fresh_only and not entry["fresh"]):
            continue
        value = _finite_number(entry["prediction"].get(field))
        if value is not None:
            points.append((index, entry["timestamp_ms"], value))
    absolute_steps: list[float] = []
    absolute_rates: list[float] = []
    zero_dt_pairs = 0
    for left, right in zip(points, points[1:]):
        if fresh_only:
            # A held display value may sit between two new predictions.  Compare
            # those update events only while coverage remained available; never
            # bridge a missing interval and call the recovery jump "jitter".
            if any(not entry["available"]
                   for entry in entries[left[0] + 1:right[0]]):
                continue
        elif right[0] != left[0] + 1:
            continue
        delta = (_angular_delta(left[2], right[2]) if angular
                 else right[2] - left[2])
        dt_ms = right[1] - left[1]
        if dt_ms <= 0:
            zero_dt_pairs += 1
            continue
        absolute_steps.append(abs(delta))
        absolute_rates.append(abs(delta) / (dt_ms / 1000.0))
    summary = _value_summary([point[2] for point in points], angular=angular)
    summary.update({
        "adjacent_positive_dt_pairs": len(absolute_steps),
        "adjacent_zero_dt_pairs_skipped": zero_dt_pairs,
        "median_abs_step": _percentile(absolute_steps, .5),
        "p90_abs_step": _percentile(absolute_steps, .9),
        "median_abs_rate_per_s": _percentile(absolute_rates, .5),
        "p90_abs_rate_per_s": _percentile(absolute_rates, .9),
        "angular_wrap_applied": angular,
    })
    return summary


def _missing_runs(entries: Sequence[dict]) -> dict:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, entry in enumerate(entries):
        if entry["available"]:
            if start is not None:
                runs.append((start, index - 1))
                start = None
        elif start is None:
            start = index
    if start is not None:
        runs.append((start, len(entries) - 1))
    observed_spans = [
        (entries[end]["timestamp_ms"] - entries[start]["timestamp_ms"]) / 1000.0
        for start, end in runs]
    until_next_spans = [
        (entries[end + 1]["timestamp_ms"] - entries[start]["timestamp_ms"]) / 1000.0
        for start, end in runs if end + 1 < len(entries)]
    return {
        "run_count": len(runs),
        "longest_run_frames": max((end - start + 1 for start, end in runs), default=0),
        "longest_observed_span_s": max(observed_spans, default=0.0),
        "longest_until_next_sample_s": (
            max(until_next_spans) if until_next_spans else None),
    }


def stability_statistics(
        records: Iterable[Mapping[str, Any]], *,
        value_fields: Sequence[str] = ("pos_x_m", "pos_y_m", "pos_z_m", "yaw_deg"),
        angular_fields: Iterable[str] = ("yaw_deg",),
        stationary_key: str | None = None) -> dict:
    """Compute sensor-time coverage, missingness, and temporal jitter.

    Each record is the output of :func:`iter_offline_predictions`: timestamp
    metadata at the top level and a ``prediction`` mapping (or ``None``).
    A prediction may explicitly set ``available`` and ``fresh``.  Retained old
    values (``available=True, fresh=False``) count toward displayed coverage but
    are reported separately and excluded from fresh-output jitter.
    """
    validate_requested_metrics(("coverage", "missing", "jitter"))
    angular = set(angular_fields)
    prepared: list[dict] = []
    previous_time: float | None = None
    duplicate_timestamps = 0
    for input_index, record in enumerate(records):
        if stationary_key is not None and not bool(record.get(stationary_key, False)):
            continue
        timestamp = _finite_number(record.get(SENSOR_TIME_FIELD))
        if timestamp is None:
            raise ValueError(f"missing/non-finite {SENSOR_TIME_FIELD} at input {input_index}")
        if previous_time is not None:
            if timestamp < previous_time:
                raise ValueError("records are not ordered by camera sensor time")
            duplicate_timestamps += int(timestamp == previous_time)
        previous_time = timestamp
        raw_prediction = record.get("prediction")
        if raw_prediction is None:
            prediction: Mapping[str, Any] = {}
            available = False
            fresh = False
        elif isinstance(raw_prediction, Mapping):
            prediction = raw_prediction
            available = bool(prediction.get("available", True))
            fresh = bool(prediction.get("fresh", available)) if available else False
        else:
            raise TypeError(f"prediction at input {input_index} is not a mapping or None")
        prepared.append({
            "timestamp_ms": timestamp, "prediction": prediction,
            "available": available, "fresh": fresh,
            "in_view": record.get("in_view"),
        })

    total = len(prepared)
    available_count = sum(entry["available"] for entry in prepared)
    fresh_count = sum(entry["fresh"] for entry in prepared)
    held_count = sum(entry["available"] and not entry["fresh"] for entry in prepared)
    missing_count = total - available_count
    visible = [entry for entry in prepared if entry["in_view"] is True]
    out_of_view = sum(entry["in_view"] is False for entry in prepared)
    visibility_unknown = total - len(visible) - out_of_view
    duration_s = ((prepared[-1]["timestamp_ms"] - prepared[0]["timestamp_ms"]) / 1000.0
                  if total >= 2 else 0.0)

    fields = {}
    for field in value_fields:
        fields[field] = {
            "available_outputs_including_held": _jitter_for_field(
                prepared, field, angular=field in angular, fresh_only=False),
            "fresh_predictions_only": _jitter_for_field(
                prepared, field, angular=field in angular, fresh_only=True),
        }
    return {
        "timebase": {
            "field": SENSOR_TIME_FIELD, "units": "ms",
            "nominal_fps_used": False, "frames": total,
            "observed_duration_s": duration_s,
            "duplicate_adjacent_timestamps": duplicate_timestamps,
        },
        "coverage": {
            "frames": total,
            "available_outputs": available_count,
            "available_fraction": available_count / total if total else None,
            "fresh_predictions": fresh_count,
            "fresh_fraction": fresh_count / total if total else None,
            "held_previous_outputs": held_count,
            "visible_frames": len(visible),
            "visible_missing_outputs": sum(not entry["available"] for entry in visible),
            "out_of_view_frames": out_of_view,
            "visibility_unknown_frames": visibility_unknown,
        },
        "missing": {
            "frames": missing_count,
            "fraction": missing_count / total if total else None,
            **_missing_runs(prepared),
        },
        "jitter": fields,
        "scope": "stationary_only" if stationary_key is not None else "caller_supplied_frames",
        "reference": {
            "independent_ground_truth": False,
            "accuracy_metrics": "X_NO_INDEPENDENT_GROUND_TRUTH",
            "reported_metrics": sorted(ALLOWED_OFFLINE_METRICS),
        },
    }
