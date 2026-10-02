"""Compact FULL prediction cache and online frozen-prefix feature batches."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from . import common as C
from .data import CACHE_SPECS, EXPECTED_PARTITIONS, SourceDimensions, validate_cache_arrays


def box_iou(left, right) -> float:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0.).prod()
    union = np.maximum(a[2:] - a[:2], 0.).prod() + np.maximum(b[2:] - b[:2], 0.).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def source_target(record: dict, prepared: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source GT points/valid/box in frozen network pixels."""
    from .full_adapter import input_module
    I = input_module()
    target = I.targets(record, prepared)
    label = record["targets"][0]
    normalized = np.asarray(label["box_xywh_normalized"], np.float64)
    if normalized.shape != (4,) or not np.isfinite(normalized).all():
        raise ValueError(f"Invalid source GT box: {record['id']}")
    wh = np.asarray(record["prepared_shape_hw"][::-1], np.float64)
    center = normalized[:2] * wh
    size = normalized[2:] * wh
    corners = np.stack((center - size / 2., center + size / 2.))
    transformed = I.transform_points(corners, prepared["affine_input_to_net"])
    box = np.r_[transformed[0], transformed[1]]
    if not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
        raise ValueError(f"Degenerate source GT box: {record['id']}")
    return (np.asarray(target["points_net"], np.float32),
            np.asarray(target["target_valid"], bool), box.astype(np.float32))


def load_arrays(mode="r", directory: Path | str | None = None) -> dict:
    root = C.RAW / "cache" if directory is None else Path(directory)
    return {name: np.load(root / f"{name}.npy", mmap_mode=mode,
                          allow_pickle=False)
            for name in CACHE_SPECS}


def _cache_manifest_payload(rows: int, protocol: dict) -> dict:
    return dict(
        schema="resnet18_full_dim_refiner_source_cache_v1",
        rows=rows,
        protocol=C.binding(C.PROTOCOL),
        baseline=protocol["baseline"],
        source=C.binding(C.SOURCE_MANIFEST),
        dimension_sidecar=C.binding(C.DIMENSION_SIDECAR),
        arrays={name: dict(shape=[rows, *tail], dtype=dtype)
                for name, (tail, dtype) in CACHE_SPECS.items()},
        full_spatial_feature_cache=False,
    )


def _validate_cache_manifest(payload: dict, rows: int, protocol: dict) -> dict:
    expected = _cache_manifest_payload(rows, protocol)
    if payload != expected:
        raise ValueError("Cache manifest differs from the sealed schema and inputs")
    return payload


def _validate_cache_file_set(directory: Path | str) -> None:
    root = Path(directory)
    expected = {"CACHE_MANIFEST.json", *(f"{name}.npy" for name in CACHE_SPECS)}
    present = {path.name for path in root.iterdir()} if root.is_dir() else set()
    missing = expected - present
    extra = present - expected
    if missing or extra:
        raise ValueError(f"Inconsistent cache file set: missing={sorted(missing)}, extra={sorted(extra)}")
    invalid = sorted(path.name for path in root.iterdir()
                     if not path.is_file() or path.is_symlink())
    if invalid:
        raise ValueError(f"Cache entries must be regular non-symlink files: {invalid}")


def _initialize_cache(rows: int, manifest_payload: dict) -> dict:
    """Create the whole empty cache atomically or validate an existing set."""
    directory = C.RAW / "cache"
    pending = directory.with_name(directory.name + ".pending")
    if directory.exists():
        if pending.exists():
            raise ValueError("Both final and pending source caches exist")
        _validate_cache_file_set(directory)
        _validate_cache_manifest(C.read_json(directory / "CACHE_MANIFEST.json"),
                                 rows, C.verify_protocol())
        arrays = load_arrays("r+", directory)
        validate_cache_arrays(arrays, rows)
        return arrays
    if pending.exists():
        _validate_cache_file_set(pending)
        _validate_cache_manifest(C.read_json(pending / "CACHE_MANIFEST.json"),
                                 rows, C.verify_protocol())
        arrays = load_arrays("r+", pending)
        validate_cache_arrays(arrays, rows)
        del arrays
        pending.replace(directory)
        return load_arrays("r+", directory)
    pending.mkdir(parents=True, exist_ok=False)
    C.write_frozen_json(pending / "CACHE_MANIFEST.json", manifest_payload)
    for name, (tail, dtype) in CACHE_SPECS.items():
        path = pending / f"{name}.npy"
        value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype,
                                          shape=(rows, *tail))
        value[:] = False if dtype == "bool" else 0
        value.flush()
        del value
    _validate_cache_file_set(pending)
    arrays = load_arrays("r+", pending)
    validate_cache_arrays(arrays, rows)
    del arrays
    pending.replace(directory)
    return load_arrays("r+", directory)


def _cache_audit_summary(arrays: dict, source: SourceDimensions) -> tuple[dict, str]:
    partitions = source.partitions
    usable = arrays["matched"] & (
        arrays["gt_valid"][:, :8] & arrays["point_valid"][:, :8]).any(-1)
    partition_counts = {}
    for name, expected in EXPECTED_PARTITIONS.items():
        selected = partitions == name
        if int(selected.sum()) != expected:
            raise ValueError("Source partition drift after cache")
        partition_counts[name] = dict(
            total=int(selected.sum()),
            matched=int((arrays["matched"] & selected).sum()),
            usable=int((usable & selected).sum()),
            missing_all_corners=int((~arrays["detected"] & selected).sum()),
        )
    digest = hashlib.sha256(np.flatnonzero(
        (partitions == "train") & usable).astype("<i8").tobytes()).hexdigest()
    return partition_counts, digest


def _verify_completion(payload: dict, source: SourceDimensions | None = None) -> dict:
    protocol = C.verify_protocol()
    if (payload.get("complete") is not True or payload.get("PASS") is not True
            or payload.get("rows") != 60000
            or payload.get("full_feature_cache") is not False
            or payload.get("all_image_and_label_hashes_verified") is not True
            or payload.get("dimension_context_exact_source_join") is not True):
        raise ValueError("Incomplete FULL source cache receipt")
    if payload.get("protocol_sha256") != C.sha256(C.PROTOCOL):
        raise ValueError("Source cache protocol mismatch")
    if (payload.get("protocol") != C.binding(C.PROTOCOL)
            or payload.get("baseline") != protocol["baseline"]
            or payload.get("source") != C.binding(C.SOURCE_MANIFEST)):
        raise ValueError("Source cache receipt provenance mismatch")
    directory = C.RAW / "cache"
    if directory.with_name(directory.name + ".pending").exists():
        raise ValueError("Pending source cache remains beside completed cache")
    _validate_cache_file_set(directory)
    expected_arrays = [C.binding(directory / f"{name}.npy") for name in CACHE_SPECS]
    if payload.get("arrays") != expected_arrays:
        raise ValueError("Source cache receipt does not bind the exact ordered cache array set")
    if payload.get("cache_manifest") != C.binding(directory / "CACHE_MANIFEST.json"):
        raise ValueError("Source cache manifest binding drift")
    manifest_path = C.verify_binding(payload["cache_manifest"])
    if manifest_path != (C.RAW / "cache/CACHE_MANIFEST.json").resolve():
        raise ValueError("Unexpected cache manifest path")
    manifest = C.read_json(manifest_path)
    _validate_cache_manifest(manifest, 60000, protocol)
    arrays = load_arrays()
    validate_cache_arrays(arrays, 60000)
    if not arrays["done"].all():
        raise ValueError("Source cache receipt exists with unfinished rows")
    source = SourceDimensions() if source is None else source
    partition_counts, train_digest = _cache_audit_summary(arrays, source)
    if (payload.get("partition_counts") != partition_counts
            or payload.get("train_usable_rows_sha256") != train_digest):
        raise ValueError("Source cache audit summary differs from bound arrays")
    return payload


def cache_source(adapter, *, batch_size=8) -> dict:
    """Run the frozen FULL baseline only; this function performs no head update."""
    protocol = C.verify_protocol()
    if (not getattr(adapter, "trained_full_loaded", False)
            or adapter.checkpoint["sha256"] != protocol["baseline"]["sha256"]):
        raise ValueError("Cache requires the exact protocol-bound FULL adapter")
    completion = C.DOC / "SOURCE_CACHE_COMPLETE.json"
    if completion.exists():
        return _verify_completion(C.read_json(completion))
    source = SourceDimensions()
    count = len(source.records)
    manifest_payload = _cache_manifest_payload(count, protocol)
    arrays = _initialize_cache(count, manifest_payload)
    validate_cache_arrays(arrays, count)
    manifest = C.RAW / "cache/CACHE_MANIFEST.json"
    from .full_adapter import input_module
    I = input_module()
    todo = np.flatnonzero(~arrays["done"])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(0, len(todo), batch_size):
            rows = todo[begin:begin + batch_size]
            records = [source.records[int(row)] for row in rows]
            images = list(pool.map(I.load_image, records))
            for record in records:
                label_path = Path(record["label"])
                label = C.ROOT / label_path if not label_path.is_absolute() else label_path
                if C.sha256(label) != record["label_sha256"]:
                    raise ValueError(f"Source label binding mismatch: {record['id']}")
            dimensions = source.dimensions[rows]
            predictions = adapter.infer_batch(images, dimensions, source_pre_padded=True,
                                              return_features=False)
            for row, record, image, prediction in zip(rows, records, images, predictions):
                del image
                affine = np.asarray(prediction["affine_input_to_net"], np.float64)
                scale = np.diag(affine)[:2]
                shift = affine[:2, 2]
                if not np.all(scale > 0) or not np.allclose(affine[:2, :2], np.diag(scale)):
                    raise ValueError("Unexpected non-axis-aligned input transform")
                gt_points, gt_valid, gt_box = source_target(record, prediction)
                box = np.asarray(prediction["bbox_net"], np.float32)
                overlap = box_iou(box, gt_box)
                values = dict(
                    points=np.asarray(prediction["points_net"], np.float32),
                    point_valid=np.asarray(prediction["valid"], bool),
                    boxes=box,
                    input_shape=np.asarray(prediction["input_shape"][-2:], np.int32),
                    scale_xy=scale,
                    shift_xy=shift,
                    gt_points=gt_points,
                    gt_valid=gt_valid,
                    dimension_context=np.asarray(prediction["dimension_context"], np.float32),
                    matched=overlap >= .5,
                    iou=overlap,
                    detected=True,
                    score=prediction["score"],
                )
                for name, value in values.items():
                    arrays[name][row] = value
                if not np.array_equal(arrays["dimension_context"][row], source.contexts[row]):
                    raise ValueError("Adapter/source dimension context drift")
            for name, value in arrays.items():
                if name != "done":
                    value.flush()
            arrays["done"][rows] = True
            arrays["done"].flush()
            if begin == 0 or (begin + len(rows)) % 256 == 0 or begin + len(rows) == len(todo):
                progress = dict(complete_rows=int(arrays["done"].sum()), total=count,
                                elapsed_seconds=time.perf_counter() - started)
                path = C.RAW / "CACHE_PROGRESS.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".pending.json")
                temporary.write_text(__import__("json").dumps(progress, indent=2) + "\n")
                temporary.replace(path)
                print("FULL_DIM_REFINER_SOURCE_CACHE", progress["complete_rows"], count, flush=True)
    validate_cache_arrays(arrays, count)
    if not arrays["done"].all():
        raise RuntimeError("Cache loop ended with unfinished rows")
    partition_counts, train_digest = _cache_audit_summary(arrays, source)
    payload = dict(
        complete=True,
        PASS=True,
        created_at=C.now(),
        protocol_sha256=C.sha256(C.PROTOCOL),
        protocol=C.binding(C.PROTOCOL),
        baseline=protocol["baseline"],
        source=C.binding(C.SOURCE_MANIFEST),
        cache_manifest=C.binding(manifest),
        arrays=[C.binding(C.RAW / "cache" / f"{name}.npy") for name in CACHE_SPECS],
        rows=count,
        partition_counts=partition_counts,
        train_usable_rows_sha256=train_digest,
        full_feature_cache=False,
        all_image_and_label_hashes_verified=True,
        dimension_context_exact_source_join=True,
    )
    C.write_frozen_json(completion, payload)
    return payload


class FeatureDataset:
    """Cache metadata plus online frozen FULL layer2/layer3 features."""
    def __init__(self):
        source = SourceDimensions()
        self.records = source.records
        self.dimensions = source.dimensions
        self.contexts = source.contexts
        self.partitions = source.partitions
        self.arrays = load_arrays()
        _verify_completion(C.read_json(C.DOC / "SOURCE_CACHE_COMPLETE.json"), source=source)
        usable = self.arrays["matched"] & (
            self.arrays["gt_valid"][:, :8] & self.arrays["point_valid"][:, :8]).any(-1)
        self.train_rows = np.flatnonzero((self.partitions == "train") & usable)
        if not len(self.train_rows):
            raise RuntimeError("No usable FULL source TRAIN rows")
        self.validation_rows = np.flatnonzero(self.partitions != "train")
        if len(self.validation_rows) != sum(EXPECTED_PARTITIONS[name]
                                            for name in ("calibration", "selection", "heldout")):
            raise ValueError("Validation partitions drifted")
        self.pool = ThreadPoolExecutor(max_workers=4)

    def prepare(self, rows, adapter):
        selected = [self.records[int(row)] for row in np.asarray(rows, np.int64)]
        from .full_adapter import input_module
        I = input_module()
        images = list(self.pool.map(I.load_image, selected))
        return [adapter.prepare(image, source_pre_padded=True) for image in images]

    def batch_features(self, rows, adapter, prepared=None) -> dict:
        indices = np.asarray(rows, np.int64)
        prepared = self.prepare(indices, adapter) if prepared is None else prepared
        if len(prepared) != len(indices):
            raise ValueError("Prepared/source batch length mismatch")
        tensor = torch.stack([value["tensor"] for value in prepared]).to(adapter.device)
        p3, p4 = adapter.features_only(tensor)
        result = {name: torch.from_numpy(np.array(self.arrays[name][indices], copy=True)).to(adapter.device)
                  for name in ("points", "point_valid", "boxes", "input_shape", "gt_points", "gt_valid")}
        # Historical P protocol: frozen feature tensors make an explicit FP16 roundtrip.
        result["p3"] = p3.to(torch.float16)
        result["p4"] = p4.to(torch.float16)
        result["dimension_context"] = torch.from_numpy(
            np.array(self.arrays["dimension_context"][indices], copy=True)).to(adapter.device)
        for local, meta in enumerate(prepared):
            row = int(indices[local])
            affine = np.asarray(meta["affine_input_to_net"], np.float64)
            if not np.array_equal(np.diag(affine)[:2], self.arrays["scale_xy"][row]):
                raise ValueError("Cached scale differs from online preprocessing")
            if not np.array_equal(affine[:2, 2], self.arrays["shift_xy"][row]):
                raise ValueError("Cached shift differs from online preprocessing")
        return result

    def close(self):
        self.pool.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    from .full_adapter import FrozenFullAdapter
    cache_source(FrozenFullAdapter(args.device), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
