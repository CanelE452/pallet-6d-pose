"""Exact source join, reusable metadata cache, and frozen feature batches."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from . import common as C
from .model import normalize_dimensions


ARRAY_SPECS = {
    "points": ((9, 2), "float32"),
    "point_valid": ((9,), "bool"),
    "boxes": ((4,), "float32"),
    "input_shape": ((2,), "int32"),
    "scale_xy": ((2,), "float64"),
    "shift_xy": ((2,), "float64"),
    "gt_points": ((9, 2), "float32"),
    "gt_valid": ((9,), "bool"),
    "matched": ((), "bool"),
    "iou": ((), "float64"),
    "detected": ((), "bool"),
    "score": ((), "float64"),
    "done": ((), "bool"),
}
DOPE_CACHE = C.ROOT / "data/pallet/results/pallet_dope_refiner_20261001_v1/cache"
DOPE_CACHE_RECEIPT = C.ROOT / "_docs/experiments/pallet_dope_refiner_20261001_v1/SOURCE_CACHE_COMPLETE.json"


def load_image(record: dict) -> np.ndarray:
    path = Path(record["image"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["image_sha256"]:
        raise RuntimeError(f"Source image hash drift: {record['id']}")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or list(image.shape[:2]) != record["prepared_shape_hw"]:
        raise RuntimeError(f"Source image decode/shape drift: {record['id']}")
    return image


def iou(left, right) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return 0.
    intersection = np.maximum(
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]), 0).prod()
    union = (np.maximum(left[2:] - left[:2], 0).prod()
             + np.maximum(right[2:] - right[:2], 0).prod() - intersection)
    return float(intersection / union) if union > 0 else 0.


def target(record: dict, scale_xy, shift_xy):
    if len(record["targets"]) != 1:
        raise RuntimeError("N3 source contract requires one pallet target")
    item = record["targets"][0]
    width_height = np.asarray(record["prepared_shape_hw"][::-1], float)
    keypoints = np.asarray(item["keypoints_normalized"], float)
    valid = ((keypoints[:, 2] > 0) & np.isfinite(keypoints[:, :2]).all(-1)
             & ~(keypoints[:, :2] == -1).all(-1))
    points = keypoints[:, :2] * width_height * scale_xy + shift_xy
    points[~valid] = np.nan
    box = np.asarray(item["box_xywh_normalized"], float) * np.tile(width_height, 2)
    box = np.r_[box[:2] - box[2:] / 2, box[:2] + box[2:] / 2]
    box = box * np.tile(scale_xy, 2) + np.tile(shift_xy, 2)
    return points, valid, box


def _paths(backbone: str):
    if backbone == "dope":
        return DOPE_CACHE, DOPE_CACHE_RECEIPT
    if backbone == "resnet18":
        return C.RAW / "resnet18_constant_cache", C.DOC / "RESNET18_CONSTANT_CACHE_COMPLETE.json"
    raise ValueError(backbone)


def load_arrays(backbone: str, mode="r"):
    directory, _ = _paths(backbone)
    return {key: np.load(directory / f"{key}.npy", mmap_mode=mode)
            for key in ARRAY_SPECS}


def verify_dope_cache() -> dict:
    receipt = C.read(DOPE_CACHE_RECEIPT)
    if not receipt.get("complete") or receipt.get("rows") != 60000:
        raise RuntimeError("Historical DOPE source cache is incomplete")
    for entry in receipt["arrays"]:
        C.verify(entry)
    arrays = load_arrays("dope")
    if not arrays["done"].all():
        raise RuntimeError("Historical DOPE cache completion bitmap is incomplete")
    return receipt


def _create_arrays(directory: Path, rows: int):
    directory.mkdir(parents=True, exist_ok=True)
    for key, (shape, dtype) in ARRAY_SPECS.items():
        path = directory / f"{key}.npy"
        if path.exists():
            continue
        values = np.lib.format.open_memmap(
            path, mode="w+", dtype=dtype, shape=(rows, *shape))
        values[:] = False if dtype == "bool" else 0
        values.flush()


def cache_resnet18(adapter) -> dict:
    """Build only the missing zero-context ResNet metadata; no feature cache."""
    directory, completion = _paths("resnet18")
    records = C.read(C.SOURCE)["records"]
    if completion.exists():
        receipt = C.read(completion)
        for entry in receipt["arrays"]:
            C.verify(entry)
        if receipt["checkpoint"]["sha256"] != adapter.checkpoint_sha256:
            raise RuntimeError("ResNet constant cache checkpoint drift")
        if not load_arrays("resnet18")["done"].all():
            raise RuntimeError("ResNet constant cache completion bitmap is incomplete")
        return receipt
    _create_arrays(directory, len(records))
    arrays = load_arrays("resnet18", "r+")
    todo = np.flatnonzero(~arrays["done"])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(0, len(todo), 16):
            rows = todo[begin:begin + 16]
            selected = [records[int(index)] for index in rows]
            images = list(pool.map(load_image, selected))
            for record in selected:
                if C.sha256(record["label"]) != record["label_sha256"]:
                    raise RuntimeError(f"Source label hash drift: {record['id']}")
            predictions = adapter.infer_batch(
                images, source_pre_padded=True, return_features=False)
            for index, record, prediction in zip(rows, selected, predictions):
                affine = np.asarray(prediction["affine_input_to_net"], float)
                scale = np.diag(affine)[:2]
                shift = affine[:2, 2]
                if not (np.all(scale > 0) and
                        np.allclose(affine[:2, :2], np.diag(scale), rtol=0, atol=0)):
                    raise RuntimeError("Only exact axis-aligned affine is supported")
                points = np.asarray(prediction["points_net"], np.float32)
                valid = np.asarray(prediction["valid"], bool)
                box = (np.full(4, np.nan, np.float32)
                       if prediction["bbox_net"] is None
                       else np.asarray(prediction["bbox_net"], np.float32))
                gt_points, gt_valid, gt_box = target(record, scale, shift)
                overlap = iou(box, gt_box)
                values = dict(
                    points=points, point_valid=valid, boxes=box,
                    input_shape=prediction["input_shape"][-2:],
                    scale_xy=scale, shift_xy=shift,
                    gt_points=gt_points, gt_valid=gt_valid,
                    matched=overlap >= .5, iou=overlap,
                    detected=bool(valid[:8].any()), score=prediction["score"],
                )
                for key, value in values.items():
                    arrays[key][index] = value
            for key, values in arrays.items():
                if key != "done":
                    values.flush()
            arrays["done"][rows] = True
            arrays["done"].flush()
            if begin == 0 or (begin + len(rows)) % 256 == 0 or begin + len(rows) == len(todo):
                progress = dict(complete_rows=int(arrays["done"].sum()),
                                total=len(records),
                                elapsed_seconds=time.perf_counter() - started,
                                gpu=C.gpu_snapshot())
                C.write(C.RAW / "RESNET18_CACHE_PROGRESS.json", progress)
                print("N3_RESNET18_CACHE", progress["complete_rows"], len(records),
                      round(progress["elapsed_seconds"], 1), flush=True)
    if not arrays["done"].all():
        raise RuntimeError("ResNet constant cache did not finish")
    partitions = np.asarray([record["partition"] for record in records])
    usable = arrays["matched"] & (
        arrays["gt_valid"][:, :8] & arrays["point_valid"][:, :8]).any(-1)
    receipt = dict(
        schema="pallet_n3_resnet18_constant_cache_v1", complete=True,
        created_at=C.now(), rows=len(records), source=C.binding(C.SOURCE),
        checkpoint=C.binding(C.RESNET_CONSTANT),
        adapter=dict(checkpoint_sha256=adapter.checkpoint_sha256,
                     fixed_zero_context=True),
        arrays=[C.binding(directory / f"{key}.npy") for key in ARRAY_SPECS],
        partition_counts={partition: dict(
            total=int((partitions == partition).sum()),
            matched=int((arrays["matched"] & (partitions == partition)).sum()),
            usable=int((usable & (partitions == partition)).sum()))
            for partition in ("train", "calibration", "selection", "heldout")},
        full_feature_cache=False, all_image_and_label_hashes_verified=True,
        base_receives_variable_dimensions=False,
    )
    C.write(completion, receipt, freeze=True)
    return receipt


class N3Dataset:
    def __init__(self, backbone: str):
        self.backbone = backbone
        self.records = C.read(C.SOURCE)["records"]
        if backbone == "dope":
            self.receipt = verify_dope_cache()
        elif backbone == "resnet18":
            _, completion = _paths(backbone)
            self.receipt = C.read(completion)
            for entry in self.receipt["arrays"]:
                C.verify(entry)
        else:
            raise ValueError(backbone)
        self.arrays = load_arrays(backbone)
        for key, (shape, dtype) in ARRAY_SPECS.items():
            if (self.arrays[key].shape != (len(self.records), *shape)
                    or str(self.arrays[key].dtype) != dtype):
                raise RuntimeError(f"Cache array contract drift: {backbone}/{key}")
        sidecar = np.load(C.SIDECAR, allow_pickle=False)
        if not np.array_equal(sidecar["record_index"], np.arange(len(self.records))):
            raise RuntimeError("N3 sidecar row identity drift")
        self.dimensions = sidecar["dimensions"]
        self.permutations = sidecar["permutations"]
        self.group_valid = sidecar["group_valid"]
        normalization = C.read(C.NORMALIZATION)
        self.dimension_context = normalize_dimensions(
            torch.from_numpy(self.dimensions), normalization).numpy().astype(np.float32)
        self.partitions = np.asarray([record["partition"] for record in self.records])
        usable = self.arrays["matched"] & (
            self.arrays["gt_valid"][:, :8] & self.arrays["point_valid"][:, :8]).any(-1)
        self.train_rows = np.flatnonzero((self.partitions == "train") & usable)
        self.validation_rows = np.flatnonzero(self.partitions != "train")
        if not len(self.train_rows):
            raise RuntimeError(f"No usable {backbone} N3 source rows")
        self.pool = ThreadPoolExecutor(max_workers=4)

    def batch_meta(self, rows, device="cuda"):
        result = {key: torch.from_numpy(np.array(self.arrays[key][rows], copy=True)).to(device)
                  for key in ("points", "point_valid", "boxes", "input_shape",
                              "gt_points", "gt_valid")}
        result.update(
            dimension_context=torch.from_numpy(np.array(
                self.dimension_context[rows], copy=True)).to(device),
            permutations=torch.from_numpy(np.array(
                self.permutations[rows], copy=True)).to(device),
            group_valid=torch.from_numpy(np.array(
                self.group_valid[rows], copy=True)).to(device),
        )
        return result

    def prepare(self, rows, adapter):
        images = list(self.pool.map(load_image,
            [self.records[int(index)] for index in rows]))
        return [adapter.prepare(image, source_pre_padded=True) for image in images]

    def batch_features(self, rows, adapter, prepared=None):
        prepared = self.prepare(rows, adapter) if prepared is None else prepared
        groups = defaultdict(list)
        for index, item in enumerate(prepared):
            groups[tuple(item["tensor"].shape)].append(index)
        feature3 = [None] * len(rows)
        feature4 = [None] * len(rows)
        for _, indices in sorted(groups.items()):
            tensor = torch.stack([prepared[index]["tensor"] for index in indices]).to(adapter.device)
            left, right = adapter.features_only(tensor)
            for local, index in enumerate(indices):
                feature3[index] = left[local].detach().to(torch.float16)
                feature4[index] = right[local].detach().to(torch.float16)
            del tensor, left, right
        result = self.batch_meta(rows, str(adapter.device))
        for key, features in (("p3", feature3), ("p4", feature4)):
            height = max(value.shape[-2] for value in features)
            width = max(value.shape[-1] for value in features)
            output = torch.zeros((len(rows), features[0].shape[0], height, width),
                                 dtype=torch.float16, device=adapter.device)
            for index, value in enumerate(features):
                output[index, :, :value.shape[-2], :value.shape[-1]] = value
            result[key] = output
        for local, item in enumerate(prepared):
            affine = np.asarray(item["affine_input_to_net"])
            row = int(rows[local])
            if (not np.array_equal(affine[:2, 2], self.arrays["shift_xy"][row])
                    or not np.array_equal(np.diag(affine)[:2], self.arrays["scale_xy"][row])
                    or list(item["tensor"].shape[-2:]) != self.arrays["input_shape"][row].tolist()):
                raise RuntimeError("Cached affine/input shape drift")
        return result

    def order(self, seed: int):
        path = C.RAW / "orders" / f"{self.backbone}_seed{seed}.npy"
        random = np.random.default_rng(seed)
        pieces = []
        remaining = C.STEPS * C.BATCH
        while remaining:
            piece = random.permutation(self.train_rows)[:remaining]
            pieces.append(piece)
            remaining -= len(piece)
        result = np.concatenate(pieces).reshape(C.STEPS, C.BATCH)
        if path.exists():
            if not np.array_equal(np.load(path), result):
                raise RuntimeError(f"Batch order drift: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.save(stream, result)
        return result, C.binding(path)

    def close(self):
        self.pool.shutdown()

