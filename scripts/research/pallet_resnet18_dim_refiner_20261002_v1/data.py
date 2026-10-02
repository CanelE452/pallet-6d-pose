"""Source-only dimension joins, deterministic paired orders and cache schema."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import (BATCH, DIMENSION_NORMALIZATION, DIMENSION_SIDECAR,
                     HEAD_ARMS, SOURCE_MANIFEST, STEPS)
from .dimensions import context_for_arm, load_normalization, normalized_context


EXPECTED_PARTITIONS = dict(train=55980, calibration=1004, selection=1031, heldout=1985)
CACHE_SPECS = {
    "points": ((9, 2), "float32"),
    "point_valid": ((9,), "bool"),
    "boxes": ((4,), "float32"),
    "input_shape": ((2,), "int32"),
    "scale_xy": ((2,), "float64"),
    "shift_xy": ((2,), "float64"),
    "gt_points": ((9, 2), "float32"),
    "gt_valid": ((9,), "bool"),
    "dimension_context": ((5,), "float32"),
    "matched": ((), "bool"),
    "iou": ((), "float64"),
    "detected": ((), "bool"),
    "score": ((), "float64"),
    "done": ((), "bool"),
}


class SourceDimensions:
    """Read only the sealed source manifest and canonical dimension sidecar."""

    def __init__(self, manifest: Path | str = SOURCE_MANIFEST,
                 sidecar: Path | str = DIMENSION_SIDECAR,
                 normalization: Path | str = DIMENSION_NORMALIZATION):
        payload = json.loads(Path(manifest).read_text())
        self.records = payload["records"]
        if len(self.records) != 60000:
            raise ValueError("Expected the fixed 60,000-row source manifest")
        for index, record in enumerate(self.records):
            if record.get("index") != index or record.get("source_kind") != "synthetic":
                raise ValueError(f"Source row identity mismatch at {index}")
        self.partitions = np.asarray([record["partition"] for record in self.records])
        counts = Counter(self.partitions.tolist())
        if counts != Counter(EXPECTED_PARTITIONS):
            raise ValueError(f"Partition drift: {counts}")
        values = np.load(sidecar)
        indices = np.asarray(values["record_index"])
        if not np.array_equal(indices, np.arange(len(self.records))):
            raise ValueError("Dimension sidecar is not a bijective source-record join")
        self.dimensions = np.asarray(values["dimensions"], np.float64)
        if self.dimensions.shape != (len(self.records), 3):
            raise ValueError("Expected one canonical W,D,H tuple per source record")
        self.normalization = load_normalization(normalization)
        self.contexts = normalized_context(self.dimensions, self.normalization)

    def rows(self, partition: str) -> np.ndarray:
        if partition not in EXPECTED_PARTITIONS:
            raise ValueError(f"Unknown source-only partition: {partition}")
        result = np.flatnonzero(self.partitions == partition)
        if len(result) != EXPECTED_PARTITIONS[partition]:
            raise ValueError("Partition row-count drift")
        return result

    def contexts_for(self, rows, arm: str):
        values = np.asarray(self.contexts[np.asarray(rows, np.int64)], np.float32)
        return context_for_arm(values, arm)


def paired_order(train_rows, seed: int, steps: int = STEPS, batch: int = BATCH) -> np.ndarray:
    rows = np.asarray(train_rows, np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("train_rows must be a nonempty unique 1D array")
    if seed not in (1, 2, 3) or steps < 1 or batch < 1:
        raise ValueError("Invalid fixed order request")
    rng = np.random.default_rng(seed)
    remaining = steps * batch
    parts = []
    while remaining:
        part = rng.permutation(rows)[:remaining]
        parts.append(part)
        remaining -= len(part)
    result = np.concatenate(parts).reshape(steps, batch)
    return result


def order_digest(order: np.ndarray) -> str:
    value = np.asarray(order, dtype="<i8")
    return hashlib.sha256(value.tobytes()).hexdigest()


def validate_cache_arrays(arrays: dict, rows: int) -> None:
    if set(arrays) != set(CACHE_SPECS):
        raise ValueError("Cache array names differ from the sealed schema")
    for name, (tail, dtype) in CACHE_SPECS.items():
        value = arrays[name]
        if value.shape != (rows, *tail) or str(value.dtype) != dtype:
            raise ValueError(f"Cache schema mismatch for {name}")
    context = np.asarray(arrays["dimension_context"])
    if not np.isfinite(context).all():
        raise ValueError("Nonfinite dimension context in source cache")


def batch_from_arrays(arrays: dict, rows, arm: str, device="cpu") -> dict:
    if arm not in HEAD_ARMS:
        raise ValueError(f"Unknown head arm: {arm}")
    indices = np.asarray(rows, np.int64)
    keys = ("points", "point_valid", "boxes", "input_shape", "gt_points", "gt_valid")
    result = {name: torch.from_numpy(np.array(arrays[name][indices], copy=True)).to(device)
              for name in keys}
    context = torch.from_numpy(np.array(arrays["dimension_context"][indices], copy=True)).to(device)
    if arm in ("P5", "P5_CONSTANT"):
        result["dimension_context"] = context_for_arm(context, arm)
    return result

