"""Pure contracts and write helpers for the unified runtime benchmark."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NAME = HERE.name
DOC = ROOT / "_docs/experiments" / NAME
RAW = ROOT / "data/pallet/results" / NAME

WARMUP = 20
REPEATS = 5
FRAMES = 26
BATCH = 1
# Fixed after a read-only replay of all 26 frames and all nine arms on the
# benchmark GPU.  YOLO/DOPE were bit exact; ResNet maxima were 1.893e-4 px and
# 2.323e-4 across stored pose metrics.  The 5e-4 gate is still negligible
# relative to reported pixel/pose errors and catches dimension-axis drift.
POINT_ATOL = 5e-4
POSE_ATOL = 5e-4
PARITY_AUDIT_MAX_POINT = 0.00018928624406555628
PARITY_AUDIT_MAX_POSE = 0.00023225626495104734

PRIMARY_ARMS = (
    "YOLO_R0",
    "YOLO_P1",
    "DOPE_BASE",
    "DOPE_P1",
    "RESNET_FULL",
    "RESNET_P0_S1",
)
AUXILIARY_ARMS = (
    "RESNET_D0_S1",
    "RESNET_P5_CONSTANT_S1",
    "RESNET_P5_S1",
)
ARMS = PRIMARY_ARMS + AUXILIARY_ARMS
BASELINE_FOR = {
    "YOLO_P1": "YOLO_R0",
    "DOPE_P1": "DOPE_BASE",
    "RESNET_P0_S1": "RESNET_FULL",
    "RESNET_D0_S1": "RESNET_FULL",
    "RESNET_P5_CONSTANT_S1": "RESNET_FULL",
    "RESNET_P5_S1": "RESNET_FULL",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path | str) -> dict:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    try:
        label = str(value.relative_to(ROOT.resolve()))
    except ValueError:
        label = str(value)
    return {"path": label, "sha256": sha256(value), "bytes": value.stat().st_size}


def verify_binding(value: dict) -> Path:
    if not isinstance(value, dict) or not {"path", "sha256"}.issubset(value):
        raise ValueError("A path/sha256 artifact binding is required")
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"Artifact binding mismatch: {path}")
    if "bytes" in value and path.stat().st_size != value["bytes"]:
        raise ValueError(f"Artifact size mismatch: {path}")
    return path.resolve()


def require_complete(path: Path | str, *, schema: str | None = None) -> dict:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"Required completed artifact is absent: {source}")
    value = read_json(source)
    if value.get("complete") is not True:
        raise RuntimeError(f"Required artifact is incomplete: {source}")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"Unexpected schema for {source}: {value.get('schema')!r}")
    return value


def write_frozen_json(path: Path | str, payload: dict) -> None:
    destination = Path(path).resolve()
    if not any(destination.is_relative_to(root.resolve()) for root in (DOC, RAW)):
        raise ValueError(f"Refusing output outside the benchmark namespace: {destination}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text() != text:
            raise ValueError(f"Frozen output differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending")
    pending.write_text(text)
    pending.replace(destination)


def measurement_schedule(arms: Iterable[str] = ARMS, frames: int = FRAMES,
                         repeats: int = REPEATS) -> list[dict]:
    """Repeat-level cyclic arm rotation, reversed on odd blocks."""
    names = tuple(arms)
    if len(names) != len(set(names)) or not names or frames < 1 or repeats < 1:
        raise ValueError("Unique arms and positive frame/repeat counts are required")
    rows = []
    for repeat in range(repeats):
        order = list(names[repeat % len(names):] + names[:repeat % len(names)])
        if repeat % 2:
            order.reverse()
        for arm_position, arm in enumerate(order):
            for image_index in range(frames):
                rows.append(dict(repeat=repeat, arm=arm, arm_position=arm_position,
                                 image_index=image_index))
    return rows


def warmup_schedule(arms: Iterable[str] = ARMS, frames: int = FRAMES,
                    rounds: int = WARMUP) -> list[dict]:
    names = tuple(arms)
    rows = []
    for warmup_index in range(rounds):
        order = list(names[warmup_index % len(names):] + names[:warmup_index % len(names)])
        if warmup_index % 2:
            order.reverse()
        for arm_position, arm in enumerate(order):
            rows.append(dict(warmup_index=warmup_index, arm=arm,
                             arm_position=arm_position,
                             image_index=warmup_index % frames))
    return rows


def describe(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("A finite nonempty one-dimensional sample is required")
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)),
                p90=float(np.quantile(array, .9)), minimum=float(array.min()),
                maximum=float(array.max()))


def summarize(rows: list[dict]) -> dict:
    expected = FRAMES * REPEATS
    result = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != expected:
            raise ValueError(f"{arm}: expected {expected} measurements, got {len(selected)}")
        result[arm] = {
            key: describe([row[key] for row in selected])
            for key in ("two_d_ms", "pnp_ms", "full_ms")
        }
        result[arm].update(
            pose_status_counts=dict(Counter(row["pose_status"] for row in selected)),
            max_2d_parity_abs_px=max(row["two_d_parity_max_abs_px"] for row in selected),
            max_pose_parity_abs=max(row["pose_parity_max_abs"] for row in selected),
            parity_PASS=all(row["parity_PASS"] for row in selected),
        )
    for arm, baseline in BASELINE_FOR.items():
        base = {(row["repeat"], row["image_index"]): row for row in rows
                if row["arm"] == baseline}
        current = [row for row in rows if row["arm"] == arm]
        result[arm]["paired_added_two_d_ms"] = describe([
            row["two_d_ms"] - base[(row["repeat"], row["image_index"])]["two_d_ms"]
            for row in current])
        result[arm]["paired_added_full_ms"] = describe([
            row["full_ms"] - base[(row["repeat"], row["image_index"])]["full_ms"]
            for row in current])
        result[arm]["paired_baseline"] = baseline
    return result


def selfcheck() -> dict:
    jobs = measurement_schedule()
    warmups = warmup_schedule()
    assert len(jobs) == len(ARMS) * FRAMES * REPEATS == 1170
    assert len(warmups) == len(ARMS) * WARMUP == 180
    assert Counter(row["arm"] for row in jobs) == {arm: 130 for arm in ARMS}
    assert Counter(row["arm"] for row in warmups) == {arm: 20 for arm in ARMS}
    first = [row["arm"] for row in jobs if row["repeat"] == 0 and row["image_index"] == 0]
    second = [row["arm"] for row in jobs if row["repeat"] == 1 and row["image_index"] == 0]
    assert first == list(ARMS)
    rotated = list(ARMS[1:] + ARMS[:1]); rotated.reverse()
    assert second == rotated
    rows = []
    for arm in ARMS:
        for repeat in range(REPEATS):
            for image_index in range(FRAMES):
                rows.append(dict(arm=arm, repeat=repeat, image_index=image_index,
                    two_d_ms=2., pnp_ms=1., full_ms=3., pose_status="OK",
                    two_d_parity_max_abs_px=0., pose_parity_max_abs=0., parity_PASS=True))
    summary = summarize(rows)
    assert summary["YOLO_R0"]["full_ms"]["median"] == 3.
    assert summary["RESNET_P5_S1"]["paired_added_full_ms"]["median"] == 0.
    return dict(PASS=True, GPU_calls=0, real_image_reads=0,
                measured_schedule_rows=len(jobs), warmup_schedule_rows=len(warmups))
