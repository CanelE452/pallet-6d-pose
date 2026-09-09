"""Frozen synthetic source membership and GT-separated YOLO input transforms.

No model is loaded here. Images already contain the original 100px reflect101
border. The default preprocess matches batch-one YOLO.predict(imgsz=640,
rect=True): aspect-preserving LetterBox(auto=True, stride=32).

`load_inference_sample` deliberately returns no target. Read `load_loss_targets`
only after the highest-score predicted instance has been selected. Never use GT
to select a detection, crop features, initialize points, or choose a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch"
DATASET = REPO / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
DEFAULT_RUN = REPO / "data/pallet/results/pallet_line_pose_v1"
PAD = 100
SPLIT_SALT = "pallet_line_pose_v1:synthetic_val_scenario:25-25-50:v1"
PARTITIONS = ("train", "calibration", "selection", "heldout")
EDGES = ((1, 2), (3, 0), (5, 6), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def immutable_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to replace a frozen, different artifact: {path}")
        return
    # Exclusive creation: concurrent runs cannot silently overwrite provenance.
    with path.open("xb") as stream:
        stream.write(encoded)


@dataclass(frozen=True)
class LetterboxTransform:
    prepared_shape_hw: tuple[int, int]
    input_shape_hw: tuple[int, int]
    resized_shape_hw: tuple[int, int]
    gain: float
    # Actual integer raster/label padding; scale_coords uses half-padding below.
    pad_ltrb: tuple[int, int, int, int]
    scale_coords_pad_xy: tuple[float, float]
    reflect_pad_px: int = PAD

    @property
    def label_scale_xy(self) -> tuple[float, float]:
        return self.gain, self.gain

    @property
    def raster_scale_xy(self) -> tuple[float, float]:
        return (self.resized_shape_hw[1] / self.prepared_shape_hw[1],
                self.resized_shape_hw[0] / self.prepared_shape_hw[0])

    @property
    def scale_boxes_pad_xy(self) -> tuple[int, int]:
        return self.pad_ltrb[:2]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "label_scale_xy": self.label_scale_xy,
                "raster_scale_xy": self.raster_scale_xy,
                "scale_coords_gain": self.gain,
                "scale_boxes_pad_xy": self.scale_boxes_pad_xy,
                "point_output_clipping_is_noninvertible": True}

    def prepared_to_input(self, points: Any) -> np.ndarray:
        """GT/label affine, matching LetterBox.update_instances (no clipping)."""
        return np.asarray(points, np.float64) * self.gain + np.asarray(self.pad_ltrb[:2])

    def input_to_prepared(self, points: Any) -> np.ndarray:
        """Inverse of the label affine; use prediction method for YOLO results."""
        return (np.asarray(points, np.float64) - np.asarray(self.pad_ltrb[:2])) / self.gain

    def original_to_input(self, points: Any) -> np.ndarray:
        return self.prepared_to_input(np.asarray(points, np.float64) + self.reflect_pad_px)

    def input_to_original(self, points: Any) -> np.ndarray:
        return self.input_to_prepared(points) - self.reflect_pad_px

    def prediction_input_to_prepared(self, points: Any, *, clip: bool = True) -> np.ndarray:
        """Equivalent to installed ops.scale_coords(..., ratio_pad=None).

        Use this for raw-head predictions. It intentionally differs from label
        affine by 0.5 input pixel when a padding total is odd. No half-cell DHT
        correction belongs here.
        """
        xy = (np.asarray(points, np.float64) - self.scale_coords_pad_xy) / self.gain
        if clip:
            xy = np.clip(xy, (0, 0), self.prepared_shape_hw[::-1])
        return xy

    def prediction_prepared_to_input(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """Lift Results.keypoints.xy; return an interior/invertible mask too.

        Clipped points cannot recover their raw-head positions. For exact
        branch conditioning prefer raw-head predictions before scale_coords.
        """
        xy = np.asarray(points, np.float64)
        interior = np.all((xy > 0) & (xy < self.prepared_shape_hw[::-1]), axis=-1)
        return xy * self.gain + self.scale_coords_pad_xy, interior

    def prediction_boxes_input_to_prepared(self, boxes: Any, *, clip: bool = True) -> np.ndarray:
        box = np.asarray(boxes, np.float64).reshape(-1, 2, 2)
        box = (box - self.scale_boxes_pad_xy) / self.gain
        if clip:
            box = np.clip(box, (0, 0), self.prepared_shape_hw[::-1])
        return box.reshape(np.asarray(boxes).shape)


def transform(canvas_shape: Sequence[int], input_shape: Sequence[int], *,
              pad_px: int = PAD) -> LetterboxTransform:
    """Recover LetterBox affine from the *actual hooked* BCHW input's H/W.

    Both shapes are H,W (H,W,C is accepted). This also supports a square
    mixed-shape batch; the caller must supply the actual network input shape.
    """
    h, w = map(int, canvas_shape[:2])
    ih, iw = map(int, input_shape[:2])
    if min(h, w, ih, iw) <= 0:
        raise ValueError("positive image shapes required")
    gain = min(ih / h, iw / w)
    rh, rw = round(h * gain), round(w * gain)
    half_x, half_y = (iw - rw) / 2, (ih - rh) / 2
    left, right = round(half_x - .1), round(half_x + .1)
    top, bottom = round(half_y - .1), round(half_y + .1)
    if min(left, top, right, bottom) < 0 or (rh + top + bottom, rw + left + right) != (ih, iw):
        raise ValueError("incompatible aspect-preserving LetterBox shapes")
    return LetterboxTransform((h, w), (ih, iw), (rh, rw), gain,
                              (left, top, right, bottom), (half_x, half_y), pad_px)


def letterbox_transform(canvas_shape: Sequence[int], imgsz: int = 640, *,
                        auto: bool = True, stride: int = 32,
                        pad_px: int = PAD) -> LetterboxTransform:
    h, w = map(int, canvas_shape[:2])
    gain = min(imgsz / h, imgsz / w)
    rh, rw = round(h * gain), round(w * gain)
    dh, dw = imgsz - rh, imgsz - rw
    if auto:
        dh, dw = dh % stride, dw % stride
    return transform((h, w), (rh + dh, rw + dw), pad_px=pad_px)


def prepare_image(bgr: np.ndarray, imgsz: int = 640, *, auto: bool = True,
                  stride: int = 32) -> tuple[np.ndarray, LetterboxTransform]:
    """Already-padded BGR -> contiguous RGB CHW float32 /255, no GT access."""
    import cv2
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("expected uint8 BGR image with three channels")
    tf = letterbox_transform(bgr.shape[:2], imgsz, auto=auto, stride=stride)
    resized = cv2.resize(bgr, tf.resized_shape_hw[::-1], interpolation=cv2.INTER_LINEAR)
    left, top, right, bottom = tf.pad_ltrb
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    chw = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1), dtype=np.float32)
    chw /= 255.0
    return chw, tf


def load_inference_sample(record: dict[str, Any], *, imgsz: int = 640,
                          auto: bool = True, stride: int = 32,
                          verify_image_hash: bool = True) -> dict[str, Any]:
    """No GT fields in output; safe input to frozen prediction/feature capture."""
    import cv2
    path = Path(record["image"])
    content = path.read_bytes()
    if verify_image_hash and hashlib.sha256(content).hexdigest() != record["image_sha256"]:
        raise RuntimeError(f"source image bytes changed: {record['id']}")
    bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or list(bgr.shape[:2]) != record["prepared_shape_hw"]:
        raise RuntimeError(f"source image shape/read failure: {record['id']}")
    image, tf = prepare_image(bgr, imgsz, auto=auto, stride=stride)
    return {"id": record["id"], "image_bgr": bgr, "image_chw": image,
            "transform": tf, "input_shape_hw": tf.input_shape_hw}


def load_loss_targets(record: dict[str, Any], tf: LetterboxTransform) -> dict[str, np.ndarray]:
    """GT in actual input pixels, for loss and post-selection IoU matching only."""
    if tuple(record["prepared_shape_hw"]) != tf.prepared_shape_hw:
        raise ValueError("target canvas and transform disagree")
    objects = record["targets"]
    wh = np.asarray(tf.prepared_shape_hw[::-1], np.float64)
    kps = np.asarray([o["keypoints_normalized"] for o in objects], np.float64)
    valid = kps[..., 2] != 0
    points = tf.prepared_to_input(kps[..., :2] * wh)
    # NaN sentinel avoids treating unsupervised source zeros as true border points.
    points[~valid] = np.nan
    boxes = np.asarray([o["box_xywh_normalized"] for o in objects], np.float64)
    centres, extent = boxes[:, :2] * wh, boxes[:, 2:] * wh
    boxes_xyxy = np.concatenate([tf.prepared_to_input(centres - extent / 2),
                                 tf.prepared_to_input(centres + extent / 2)], axis=-1)
    edges = np.asarray(EDGES)
    return {"classes": np.asarray([o["class_id"] for o in objects], np.int64),
            "boxes_xyxy": boxes_xyxy, "kps": points, "kp_valid": valid,
            "kp_visibility": kps[..., 2].astype(np.int8),
            "edge_endpoints": points[:, edges],
            "edge_valid": valid[:, edges].all(axis=-1)}


def select_top1(scores: Any, *, class_ids: Any = None, class_id: int = 0) -> int | None:
    """Highest *predicted* score, stable first-index tie; no GT argument exists."""
    score = np.asarray(scores, np.float64).reshape(-1)
    keep = np.isfinite(score)
    if class_ids is not None:
        cls = np.asarray(class_ids).reshape(-1)
        if cls.shape != score.shape:
            raise ValueError("score/class lengths differ")
        keep &= cls == class_id
    indices = np.flatnonzero(keep)
    return None if not len(indices) else int(indices[np.argmax(score[indices])])


def match_loss_target(predicted_box_xyxy: Any, target_boxes_xyxy: Any, *,
                      minimum_iou: float) -> dict[str, Any]:
    """Match the already-selected predicted instance to a loss target only.

    Never call this to choose between predicted detections. Threshold is an
    explicit training-protocol choice, not tuned on real data here.
    """
    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be within [0,1]")
    boxes = np.asarray(target_boxes_xyxy, np.float64).reshape(-1, 4)
    pred = np.asarray(predicted_box_xyxy, np.float64).reshape(4)
    if not len(boxes) or not np.isfinite(pred).all() or not np.isfinite(boxes).all():
        return {"target_index": None, "iou": 0.0, "matched": False}
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], pred[2:]) -
                              np.maximum(boxes[:, :2], pred[:2])).prod(axis=1)
    union = (np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1) +
             np.maximum(0, pred[2:] - pred[:2]).prod() - intersection)
    ious = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    index = int(np.argmax(ious))
    matched = bool(ious[index] >= minimum_iou and ious[index] > 0)
    return {"target_index": index if matched else None, "iou": float(ious[index]),
            "matched": matched}


class SourceDataset:
    """Torch DataLoader-compatible dataset; targets are opt-in and separate.

    Rectangular images have different H/W. Use batch_size=1, shape-homogeneous
    batches, or list collation. Do not stack different shapes before prediction.
    """
    def __init__(self, manifest: str | Path | dict, partition: str = "train", *,
                 include_loss_targets: bool = False, verify_image_hash: bool = True):
        data = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
        if data["schema"] != "pallet_line_pose_source_v1" or partition not in PARTITIONS:
            raise ValueError("wrong source schema or partition")
        self.records = [data["records"][i] for i in data["partitions"][partition]]
        self.include_loss_targets = include_loss_targets
        self.verify_image_hash = verify_image_hash

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = load_inference_sample(record, verify_image_hash=self.verify_image_hash)
        if self.include_loss_targets:
            sample["loss_targets"] = load_loss_targets(record, sample["transform"])
        return sample


def _audit_record(meta: dict[str, Any]) -> dict[str, Any]:
    image = (REPO / meta["image_path"]).absolute()
    label = (REPO / meta["yolo_label_path"]).absolute()
    split, source = meta["split"], meta["source"]
    # Strict source allowlist; no generic arbitrary-path or real-data fallback.
    expected_image = DATASET / "images" / split / f"{meta['merged_stem']}.png"
    expected_label = DATASET / "labels" / split / f"{meta['merged_stem']}.txt"
    if image != expected_image or label != expected_label or source not in {"G38", "P0", "TEX"}:
        raise RuntimeError(f"source allowlist violation: {meta['merged_stem']}")
    raw_h, raw_w = meta["raw_shape_height_width"]
    digest = hashlib.sha256()
    with image.open("rb") as stream:
        header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise RuntimeError(f"unexpected image format: {image}")
        w, h = struct.unpack(">II", header[16:24])
        digest.update(header)
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if [h, w] != [raw_h + 200, raw_w + 200] or meta["prepared_pad_px"] != PAD:
        raise RuntimeError(f"prepared shape/padding mismatch: {image}")
    label_bytes = label.read_bytes()
    rows = [np.fromstring(s, sep=" ", dtype=np.float64) for s in label_bytes.decode().splitlines() if s.strip()]
    if len(rows) != 1 or rows[0].shape != (32,) or not np.isfinite(rows[0]).all():
        raise RuntimeError(f"expected one finite class+xywh+9x3 label: {label}")
    row = rows[0]
    box, kps = row[1:5], row[5:].reshape(9, 3)
    if (row[0] != 0 or np.any(box < 0) or np.any(box > 1) or np.any(box[2:] <= 0)
            or not np.isin(kps[:, 2], [0, 2]).all()
            or np.any(kps[:, :2] < 0) or np.any(kps[:, :2] > 1)
            or np.any(kps[kps[:, 2] == 0, :2] != 0) or not np.any(kps[:, 2] == 2)):
        raise RuntimeError(f"label contract violation: {label}")
    family = "G38" if source == "G38" else "P0_TEX"
    scenario = f"{family}:{meta['pair_group_id']}"
    return {"id": meta["merged_stem"], "source": source, "source_kind": "synthetic",
            "source_split": split, "scenario_id": scenario, "image": str(image),
            "label": str(label), "raw_shape_hw": [raw_h, raw_w],
            "prepared_shape_hw": [h, w], "reflect_pad_px": PAD,
            "image_sha256": digest.hexdigest(),
            "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
            "renderer_annotation_locator_provenance_only": meta["renderer_label_locator"],
            "targets": [{"class_id": 0, "box_xywh_normalized": box.tolist(),
                          "keypoints_normalized": kps.tolist()}]}


def _prior_dht_lineage(records: list[dict]) -> list[dict]:
    locations = ["hough_attention_transfer_v1", "dht_padding_view_v1",
                 "dht_pose_integration_v1/live"]
    target = {}
    for row in records:
        locator = row["renderer_annotation_locator_provenance_only"]
        key = str((REPO / locator).absolute()) if "::" not in locator else locator
        target[key] = row
    output = []
    for location in locations:
        path = REPO / "data/pallet/results" / location / "manifest.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        previous = data.get("records")
        if previous is None:
            previous = [r for rows in data["populations"].values() for r in rows]
        counts, examples, populations = Counter(), [], Counter()
        for row in previous:
            # Metadata only, and exclude real rows before reading target fields.
            if row.get("source_kind", "").startswith("real") or row.get("population", "").startswith("real"):
                continue
            populations[row["population"]] += 1
            annotation = str(Path(row.get("annotation", "")).absolute())
            if annotation in target:
                now = target[annotation]
                counts[f"{row['population']}->{now['partition']}"] += 1
                if len(examples) < 12:
                    examples.append({"prior_id": row["id"], "current_id": now["id"],
                                     "current_partition": now["partition"]})
        output.append({"manifest": str(path), "manifest_sha256": sha256(path),
                       "comparison": "exact normalized renderer annotation locator; no image/GT opened",
                       "prior_synthetic_population_counts": dict(populations),
                       "matched_counts": dict(counts), "matched_examples": examples,
                       "limitation": "A zero locator intersection is not proof of no shared assets or copied/render-related images."})
    return output


def build_manifest(run_dir: Path, *, workers: int = 8) -> dict[str, Any]:
    if not (run_dir / "PURPOSE.md").is_file():
        raise RuntimeError("run directory must already contain PURPOSE.md")
    metadata = SOURCE / "PROBE_METADATA_60K.jsonl"
    metadata_audit = SOURCE / "PROBE_METADATA_60K_AUDIT.json"
    audit = json.loads(metadata_audit.read_text())
    if sha256(metadata) != audit["manifest"]["sha256"] or audit["manifest"]["rows"] != 60000:
        raise RuntimeError("frozen source metadata binding differs")
    metas = [json.loads(line) for line in metadata.read_text().splitlines()]
    if len(metas) != 60000 or len({r["merged_stem"] for r in metas}) != 60000:
        raise RuntimeError("source count or ID uniqueness failed")
    expected = {"G38:train": 38002, "G38:val": 1998, "P0:train": 8989,
                "P0:val": 1011, "TEX:train": 8989, "TEX:val": 1011}
    if Counter(f"{r['source']}:{r['split']}" for r in metas) != Counter(expected):
        raise RuntimeError("source training/validation membership differs")
    for split in ("train", "val"):
        ids = {r["merged_stem"] for r in metas if r["split"] == split}
        if ({p.stem for p in (DATASET / "images" / split).glob("*.png")} != ids
                or {p.stem for p in (DATASET / "labels" / split).glob("*.txt")} != ids):
            raise RuntimeError(f"source image/label inventory differs: {split}")
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(_audit_record, metas):
            records.append(record)
            if len(records) % 10000 == 0:
                print(f"source content hashes and labels verified: {len(records)}/60000", flush=True)
    records.sort(key=lambda row: row["id"])
    by_scenario = defaultdict(list)
    for row in records:
        by_scenario[row["scenario_id"]].append(row)
    for scenario, rows in by_scenario.items():
        if len({r["source_split"] for r in rows}) != 1:
            raise RuntimeError(f"scenario crosses original train/val: {scenario}")
        expected_sources = {"G38"} if scenario.startswith("G38:") else {"P0", "TEX"}
        if len(rows) != len(expected_sources) or {r["source"] for r in rows} != expected_sources:
            raise RuntimeError(f"scenario membership differs: {scenario}")
    validation_groups = sorted(
        [key for key, rows in by_scenario.items() if rows[0]["source_split"] == "val"],
        key=lambda key: (hashlib.sha256(f"{SPLIT_SALT}\n{key}".encode()).hexdigest(), key))
    n = len(validation_groups)
    partition_for = {key: ("calibration" if i < n // 4 else "selection" if i < n // 2 else "heldout")
                     for i, key in enumerate(validation_groups)}
    partitions = {key: [] for key in PARTITIONS}
    for i, row in enumerate(records):
        row["index"] = i
        row["partition"] = "train" if row["source_split"] == "train" else partition_for[row["scenario_id"]]
        partitions[row["partition"]].append(i)
    bindings = {str(p): sha256(p) for p in (
        DATASET / "data.yaml", metadata, metadata_audit, SOURCE / "DATASET_BUILD.json",
        SOURCE / "build_dataset.py", SOURCE / "build_probe_metadata.py",
        REPO / "challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py")}
    payload = {"schema": "pallet_line_pose_source_v1", "dataset": str(DATASET),
               "source_bindings_sha256": bindings,
               "split_rule": {"salt": SPLIT_SALT, "ordering": "SHA256(salt + newline + namespaced scenario_id), ascending",
                              "source_train_unchanged": True, "validation_group_fractions": [.25, .25, .5],
                              "boundaries": [n // 4, n // 2], "validation_groups": n,
                              "no_gt_in_split": True,
                              "heldout_scope": "New line branch only; R0 and prior exploratory probes already used this original validation population."},
               "input_recipe": {"prepared_reflect101_pad_px": PAD, "add_reflect_padding": False,
                                "imgsz": 640, "rect": True, "auto": True, "stride": 32,
                                "scaleup": True, "center": True, "letterbox_value": 114,
                                "color": "RGB", "dtype": "float32", "normalization": "divide255",
                                "inference_batch": 1, "prediction_selection": "highest predicted score, no GT",
                                "dimensions_pose_parity_as_input": False},
               "side_edges": EDGES, "partitions": partitions, "records": records}
    immutable_json(run_dir / "SOURCE_MANIFEST.json", payload)
    visibility = Counter(int(v) for row in records for v in np.asarray(row["targets"][0]["keypoints_normalized"])[:, 2])
    group_counts = {part: len({records[i]["scenario_id"] for i in indices}) for part, indices in partitions.items()}
    summary = {"schema": "pallet_line_pose_source_audit_v1", "PASS": True,
               "source_data_py_sha256": sha256(Path(__file__)),
               "manifest_sha256": sha256(run_dir / "SOURCE_MANIFEST.json"),
               "n_records": len(records), "source_split_counts": expected,
               "partition_counts": {key: len(value) for key, value in partitions.items()},
               "partition_source_counts": {key: dict(Counter(records[i]["source"] for i in value)) for key, value in partitions.items()},
               "scenario_counts": group_counts, "scenario_partition_crossings": 0,
               "p0_tex_pair_groups": 10000, "p0_tex_val_pair_groups": 1011,
               "all_source_images_sha256_verified": True, "all_source_labels_sha256_verified": True,
               "all_prepared_png_headers_match_raw_plus200": True,
               "keypoint_visibility_counts": dict(visibility), "real_training_records": 0,
               "model_inference_executed": False,
               "prior_dht_lineage": _prior_dht_lineage(records)}
    immutable_json(run_dir / "SOURCE_DATA_AUDIT.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = build_manifest(args.run_dir.absolute(), workers=args.workers)
    print(json.dumps({k: v for k, v in report.items() if k != "prior_dht_lineage"}, indent=2))


if __name__ == "__main__":
    main()
