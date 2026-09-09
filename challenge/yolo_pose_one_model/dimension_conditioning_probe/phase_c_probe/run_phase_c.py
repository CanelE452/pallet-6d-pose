"""Frozen YOLO feature/geometry probes for camera-facing parity.

The script has two deliberately separated stages:

``extract``
    Discover the active end-to-end one2one pose head by capability (never by a
    hard-coded module path), attach read-only hooks, and cache the top-ranked
    candidate's classification penultimate feature and 2-D geometry.

``train``
    Train tiny Linear and two-layer MLP probes on the synthetic TRAIN split,
    select checkpoints only with synthetic DEV, lock that selection to disk,
    and only then open real DEV labels for diagnostics.

The YOLO model is always frozen.  No optimizer ever receives a YOLO parameter.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from challenge.evaluation_v2.pnp_selector import (  # noqa: E402
    SelectorConfig,
    select_pnp_hypotheses,
)
from scripts.annotate.object_geometry_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)


NS = REPO_ROOT / "challenge/yolo_pose_one_model/dimension_conditioning_probe"
PHASE_A_DIR = NS / "phase_a_oracle"
PHASE_B_DIR = NS / "phase_b_data"
OUT_DIR = NS / "phase_c_probe"
COMMON_PLASTIC_MANIFEST = (
    REPO_ROOT / "challenge/real_gt_v2/manifests/COMMON_DEV_PLASTIC_POS128.json"
)
COMMON_MULTISHAPE_MANIFEST = (
    REPO_ROOT / "challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json"
)
RAW_SYNTH = (
    REPO_ROOT
    / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
)
WEIGHTS = (
    REPO_ROOT
    / "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
    "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt"
)

SCHEMA_VERSION = "dimension_conditioning_phase_c_v1"
FEATURE_RECIPE_VERSION = "dimension_conditioning_top1_one2one_cls_pen_v2"
CHECKPOINT_SHA256 = "1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c"
CONFIDENCE_FLOOR = 0.001
IMGSZ = 640
PAD = 100
SYNTH_BATCH = 32
SHARD_SIZE = 512
TRAIN_EPOCHS = 40
TRAIN_BATCH = 512
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
SEEDS = (0, 1, 2)
ARCHITECTURES = ("LINEAR", "MLP")
SHORT_FRONT = 0
LONG_FRONT = 1
PARITY_NAME = {SHORT_FRONT: "short-face-front", LONG_FRONT: "long-face-front"}
SPLIT_CODE = {"TRAIN": 0, "DEV": 1, "TEST": 2}
SPLIT_NAME = {value: key for key, value in SPLIT_CODE.items()}

ASSET_SPLIT = {
    ("eur_pallet_bk_cc0.glb", "Pallet_3"): "TRAIN",
    ("woodpallet_block_jtoastie_ccby.glb", "Pallet_2"): "TRAIN",
    ("scene.usd", "Pallet_0"): "DEV",
    ("scene_1.usd", "Pallet_1"): "TEST",
}

WIDTH_EDGES = {
    frozenset((0, 1)),
    frozenset((2, 3)),
    frozenset((4, 5)),
    frozenset((6, 7)),
}
DEPTH_EDGES = {
    frozenset((0, 4)),
    frozenset((1, 5)),
    frozenset((2, 6)),
    frozenset((3, 7)),
}

ARM_FEATURES = {
    "B0_DIMS_ONLY": ("dims",),
    "B1_KP_ONLY": ("kp",),
    "B2_IMAGE_KP": ("image", "kp"),
    "B3_IMAGE_KP_DIMS": ("image", "kp", "dims"),
}


class PhaseCError(RuntimeError):
    """A frozen-contract or feature-provenance violation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseCError(f"JSON_UNREADABLE: {path}: {exc}") from exc


def _prerequisite_audit() -> dict[str, Any]:
    phase_a_path = PHASE_A_DIR / "DEV_DIMENSION_ORACLE_DIAGNOSTIC.json"
    phase_b_path = PHASE_B_DIR / "SYNTH_METADATA_AUDIT.json"
    split_path = PHASE_B_DIR / "SPLIT_MEMBERSHIP.json"
    phase_a = _read_json(phase_a_path)
    phase_b = _read_json(phase_b_path)
    if phase_a.get("oracle_gate", {}).get("verdict") != "ORACLE_PARITY_HEADROOM_PRESENT":
        raise PhaseCError("PHASE_A_ORACLE_GATE_NOT_GO")
    checkpoint = phase_a.get("checkpoint", {})
    if (
        checkpoint.get("sha256_before") != CHECKPOINT_SHA256
        or checkpoint.get("sha256_after") != CHECKPOINT_SHA256
        or checkpoint.get("unchanged") is not True
    ):
        raise PhaseCError("PHASE_A_CHECKPOINT_CONTRACT_MISMATCH")
    hard_block = phase_b.get("hard_block", {})
    if any(bool(value) for value in hard_block.values()):
        raise PhaseCError(f"PHASE_B_DIMENSION_SUPERVISION_HARD_BLOCK: {hard_block}")
    split_status = phase_b.get("asset_and_topology", {}).get("split_status")
    if split_status != "PROVISIONAL_ASSET_ID_DISJOINT":
        raise PhaseCError(f"PHASE_B_SPLIT_STATUS_UNEXPECTED: {split_status}")
    new_render = phase_b.get("new_render_assessment", {})
    return {
        "phase_a": {
            "path": _display(phase_a_path),
            "sha256": _sha256(phase_a_path),
            "verdict": "ORACLE_PARITY_HEADROOM_PRESENT",
        },
        "phase_b": {
            "path": _display(phase_b_path),
            "sha256": _sha256(phase_b_path),
            "hard_block": hard_block,
            "split_status": split_status,
            "asset_id_disjoint": bool(
                phase_b.get("asset_and_topology", {}).get(
                    "new_split_source_asset_overlap_zero"
                )
            ),
            "topology_certified": bool(
                phase_b.get("asset_and_topology", {}).get("topology_certified")
            ),
            "new_render_assessment": new_render,
        },
        "scope_lock": {
            "primary": "PLASTIC_STANDARD_ONLY",
            "plastic_selector_diagnostic_population": "DEV_PLASTIC_POS140",
            "plastic_model_comparison_population": "COMMON_DEV_PLASTIC_POS128",
            "synthetic_test_claim": "PROBE_HEAD_ASSET_ID_HELDOUT_ONLY_NOT_END_TO_END_HELDOUT",
            "synthetic_test_probe_head_split": "PROVISIONAL_ASSET_ID_DISJOINT_NOT_TOPOLOGY_CERTIFIED",
            "synthetic_test_frozen_feature_extractor_exposure": (
                "PRIMARY_YOLO_WAS_TRAINED_ON_ORIGINAL_G38_MEMBERSHIP_SPANNING_ALL_FOUR_ASSETS"
            ),
            "synthetic_test_original_g38_origin_counts": {
                "train": 9591,
                "val": 504,
                "total": 10095,
            },
            "synthetic_test_end_to_end_heldout_claim_allowed": False,
            "B4_real_control_population": "COMMON_DEV_MULTISHAPE_POS_128_PLUS_45",
            "wood": "OUT_OF_RANGE_EXTRAPOLATION_DIAGNOSTIC_ONLY",
            "multishape_claim_allowed": False,
            "multishape_blocker": (
                "DIM_PROBE_SYNTH_REQUIRED_FOR_MULTISHAPE_REGISTRY_COVERAGE"
                if bool(new_render.get("new_render_required"))
                else None
            ),
        },
        "split_membership": {
            "path": _display(split_path),
            "sha256": _sha256(split_path),
        },
    }


def _feature_recipe(prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    payload = {
        "version": FEATURE_RECIPE_VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "phase_a_sha256": prerequisites["phase_a"]["sha256"],
        "phase_b_sha256": prerequisites["phase_b"]["sha256"],
        "split_membership_sha256": prerequisites["split_membership"]["sha256"],
        "geometry_registry_sha256": registry.sha256,
        "inference": {
            "confidence_floor": CONFIDENCE_FLOOR,
            "imgsz": IMGSZ,
            "pad": PAD,
            "border": "BORDER_REFLECT_101",
            "candidate": "highest-confidence final detection",
        },
        "tap": {
            "head_discovery": (
                "unique module exposing one2one_cv3/one2one_cv4/get_topk_index/nl/stride"
            ),
            "feature": "immediate producer of one2one classification readout",
            "mapping": "captured get_topk_index final-source cell",
        },
        "kp_geometry": "bbox-center/diagonal 18xy + aspect + area + 6 edge features",
        "dimension_feature": "fixed-axis xyz logs reconstructed with perm_v4",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _fixed_dimensions(obj: Mapping[str, Any]) -> tuple[float, float, float]:
    dimensions = obj.get("dimensions_m")
    permutation = obj.get("perm_v4")
    if not isinstance(dimensions, Mapping):
        raise PhaseCError("SYNTH_DIMENSIONS_MISSING")
    if (
        not isinstance(permutation, list)
        or len(permutation) != 8
        or sorted(permutation) != list(range(8))
    ):
        raise PhaseCError("SYNTH_PERM_V4_INVALID")
    width = float(dimensions["width"])
    height = float(dimensions["height"])
    depth = float(dimensions["depth"])
    edge = frozenset((int(permutation[0]), int(permutation[1])))
    if edge in WIDTH_EDGES:
        return width, height, depth
    if edge in DEPTH_EDGES:
        return depth, height, width
    raise PhaseCError(f"SYNTH_FIXED_AXIS_NOT_RECOVERABLE: {permutation}")


def _load_synth_contract() -> dict[str, np.ndarray]:
    """Load renderer metadata only; no filename-derived target is accepted."""

    labels_dir = RAW_SYNTH / "labels"
    records: list[dict[str, Any]] = []
    for index in range(40_000):
        frame_id = f"f{index:04d}"
        label_path = labels_dir / f"{frame_id}_label.json"
        payload = _read_json(label_path)
        objects = payload.get("objects")
        if not isinstance(objects, list) or len(objects) != 1:
            raise PhaseCError(f"SYNTH_OBJECT_COUNT_INVALID: {frame_id}")
        obj = objects[0]
        asset = (str(obj.get("source_asset")), str(obj.get("name")))
        split = ASSET_SPLIT.get(asset)
        if split is None:
            raise PhaseCError(f"SYNTH_ASSET_NOT_IN_FROZEN_SPLIT: {frame_id}: {asset}")
        dimensions = obj.get("dimensions_m")
        width = float(dimensions["width"])
        depth = float(dimensions["depth"])
        if width == depth:
            raise PhaseCError(f"SYNTH_PARITY_TIE: {frame_id}")
        camera = payload.get("camera_data")
        if not isinstance(camera, Mapping):
            raise PhaseCError(f"SYNTH_CAMERA_DATA_MISSING: {frame_id}")
        resolution = camera.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            resolution = [camera.get("width"), camera.get("height")]
        raw_width, raw_height = int(resolution[0]), int(resolution[1])
        image_path = RAW_SYNTH / "rgb" / f"{frame_id}_rgb.png"
        if not image_path.is_file():
            raise PhaseCError(f"SYNTH_IMAGE_MISSING: {frame_id}")
        records.append(
            {
                "frame_id": frame_id,
                "image": str(image_path),
                "asset_id": f"{asset[0]}::{asset[1]}",
                "split": SPLIT_CODE[split],
                "dimensions": _fixed_dimensions(obj),
                "label": LONG_FRONT if width > depth else SHORT_FRONT,
                "raw_shape": (raw_height, raw_width),
            }
        )

    # The independently generated Phase-B membership is a required input and
    # must describe the same exact frame partition.  Accept either of the two
    # explicit replay schemas used by the audit builder, never infer from names.
    membership_path = PHASE_B_DIR / "SPLIT_MEMBERSHIP.json"
    if not membership_path.is_file():
        raise PhaseCError("PHASE_B_SPLIT_MEMBERSHIP_REQUIRED")
    membership = _read_json(membership_path)
    observed: dict[str, str] = {}
    split_payload = membership.get("splits") if isinstance(membership, Mapping) else None
    if isinstance(split_payload, Mapping):
        for split_name, value in split_payload.items():
            if isinstance(value, list):
                ids = value
            elif isinstance(value, Mapping) and isinstance(value.get("frame_ids"), list):
                ids = value["frame_ids"]
            else:
                continue
            for frame_id in ids:
                observed[str(frame_id)] = str(split_name).upper()
    mapping_payload = (
        membership.get("frame_to_split") if isinstance(membership, Mapping) else None
    )
    if isinstance(mapping_payload, Mapping):
        observed.update({str(k): str(v).upper() for k, v in mapping_payload.items()})
    if len(observed) != 40_000:
        raise PhaseCError(
            f"PHASE_B_SPLIT_MEMBERSHIP_SCHEMA_OR_COUNT_INVALID: {len(observed)}"
        )
    for record in records:
        expected = SPLIT_NAME[int(record["split"])]
        if observed.get(str(record["frame_id"])) != expected:
            raise PhaseCError(f"PHASE_B_SPLIT_MISMATCH: {record['frame_id']}")

    return {
        "frame_ids": np.asarray([row["frame_id"] for row in records], dtype=np.str_),
        "images": np.asarray([row["image"] for row in records], dtype=np.str_),
        "asset_ids": np.asarray([row["asset_id"] for row in records], dtype=np.str_),
        "splits": np.asarray([row["split"] for row in records], dtype=np.int8),
        "dimensions_xyz": np.asarray(
            [row["dimensions"] for row in records], dtype=np.float32
        ),
        "labels": np.asarray([row["label"] for row in records], dtype=np.int8),
        "raw_shapes": np.asarray([row["raw_shape"] for row in records], dtype=np.int32),
    }


def _dimension_raw(dimensions_xyz: np.ndarray) -> np.ndarray:
    values = np.asarray(dimensions_xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or np.any(values <= 0.0):
        raise PhaseCError("DIMENSION_FEATURE_INPUT_INVALID")
    geometric_mean = np.exp(np.mean(np.log(values), axis=1))
    x, y, z = values[:, 0], values[:, 1], values[:, 2]
    return np.stack(
        [
            np.log(x / geometric_mean),
            np.log(y / geometric_mean),
            np.log(z / geometric_mean),
            np.log(x / z),
        ],
        axis=1,
    ).astype(np.float32)


def _geometry_feature(
    box: np.ndarray,
    keypoints: np.ndarray,
    raw_height: int,
    raw_width: int,
) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if box.shape != (4,) or keypoints.shape != (9, 2):
        raise PhaseCError("KP_GEOMETRY_SHAPE_INVALID")
    width = max(float(box[2] - box[0]), 1.0e-6)
    height = max(float(box[3] - box[1]), 1.0e-6)
    diagonal = math.hypot(width, height)
    center = np.asarray(
        [(float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5]
    )
    normalized = ((keypoints - center) / diagonal).reshape(-1)

    width_edges = ((0, 1), (2, 3), (4, 5), (6, 7))
    height_edges = ((0, 3), (1, 2), (4, 7), (5, 6))
    depth_edges = ((0, 4), (1, 5), (2, 6), (3, 7))

    def mean_edge(edges: Sequence[tuple[int, int]]) -> float:
        return float(
            np.mean([np.linalg.norm(keypoints[first] - keypoints[second]) for first, second in edges])
        )

    edge_width = mean_edge(width_edges)
    edge_height = mean_edge(height_edges)
    edge_depth = mean_edge(depth_edges)
    eps = 1.0e-6
    structural = np.asarray(
        [
            edge_width / diagonal,
            edge_height / diagonal,
            edge_depth / diagonal,
            edge_width / max(edge_height, eps),
            edge_depth / max(edge_height, eps),
            edge_width / max(edge_depth, eps),
        ],
        dtype=np.float64,
    )
    scalars = np.asarray(
        [
            width / height,
            (width * height) / max(float(raw_width * raw_height), 1.0),
        ],
        dtype=np.float64,
    )
    result = np.concatenate([normalized, scalars, structural]).astype(np.float32)
    if result.shape != (26,) or not np.isfinite(result).all():
        raise PhaseCError("KP_GEOMETRY_FEATURE_INVALID")
    return result


class DynamicOne2OneTap:
    """Read the active one2one classification path without a module-name guess."""

    def __init__(self, weights: Path, device: str):
        import torch
        from ultralytics import YOLO

        if _sha256(weights) != CHECKPOINT_SHA256:
            raise PhaseCError("PRIMARY_CHECKPOINT_SHA_MISMATCH")
        self.torch = torch
        self.device = str(device)
        self.yolo = YOLO(str(weights), task="pose")
        self.model = self.yolo.model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        candidates: list[tuple[str, Any]] = []
        required = ("one2one_cv3", "one2one_cv4", "get_topk_index", "nl", "stride")
        for name, module in self.model.named_modules():
            if all(hasattr(module, attribute) for attribute in required):
                candidates.append((name, module))
        if len(candidates) != 1:
            raise PhaseCError(
                f"ONE2ONE_HEAD_DISCOVERY_NOT_UNIQUE: {[(name, type(m).__name__) for name, m in candidates]}"
            )
        self.head_path, self.head = candidates[0]
        self.level_count = int(self.head.nl)
        self.cap: dict[int, Any] = {}
        self.logit_cap: dict[int, Any] = {}
        self.flat_indices: Any | None = None
        self.handles: list[Any] = []
        self.penultimate_paths: list[str] = []
        self.logit_paths: list[str] = []
        qualified = {id(module): name for name, module in self.model.named_modules()}
        for level in range(self.level_count):
            tower = self.head.one2one_cv3[level]
            children = list(tower.children())
            if len(children) < 2:
                raise PhaseCError(f"ONE2ONE_CLS_TOWER_TOO_SHORT: level={level}")
            # The output/readout is discovered as the last direct child; the
            # feature tap is its immediate producer.  Paths are recorded from
            # named_modules identity, never assembled as an executable guess.
            penultimate = children[-2]
            readout = children[-1]
            if id(penultimate) not in qualified or id(readout) not in qualified:
                raise PhaseCError("ONE2ONE_TAP_QUALIFIED_PATH_NOT_FOUND")
            self.penultimate_paths.append(qualified[id(penultimate)])
            self.logit_paths.append(qualified[id(readout)])
            self.handles.append(
                penultimate.register_forward_hook(self._capture(level, logits=False))
            )
            self.handles.append(readout.register_forward_hook(self._capture(level, logits=True)))

        original = self.head.get_topk_index
        self._original_topk = original

        def wrapped(scores: Any, max_det: int) -> Any:
            output = original(scores, max_det)
            self.flat_indices = output[2].detach()
            return output

        self.head.get_topk_index = wrapped

    def _capture(self, level: int, *, logits: bool):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            target = self.logit_cap if logits else self.cap
            target[level] = output.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.head.get_topk_index = self._original_topk

    def _level_sizes(self) -> list[tuple[int, int]]:
        if set(self.cap) != set(range(self.level_count)):
            raise PhaseCError("ONE2ONE_FEATURE_HOOK_DID_NOT_FIRE")
        return [
            (int(self.cap[level].shape[-2]), int(self.cap[level].shape[-1]))
            for level in range(self.level_count)
        ]

    def _decode(self, flat: int) -> tuple[int, int, int]:
        offset = 0
        for level, (height, width) in enumerate(self._level_sizes()):
            count = height * width
            if flat < offset + count:
                relative = flat - offset
                return level, relative // width, relative % width
            offset += count
        raise PhaseCError(f"TOPK_FLAT_INDEX_OUT_OF_RANGE: {flat}")

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[list[Any], list[dict[str, Any] | None]]:
        self.cap.clear()
        self.logit_cap.clear()
        self.flat_indices = None
        results = self.yolo.predict(
            list(images),
            batch=len(images),
            conf=CONFIDENCE_FLOOR,
            imgsz=IMGSZ,
            device=self.device,
            verbose=False,
        )
        if self.flat_indices is None or int(self.flat_indices.shape[0]) != len(results):
            raise PhaseCError("TOPK_PROVENANCE_CAPTURE_MISSING")
        vectors: list[dict[str, Any] | None] = []
        for batch_index, result in enumerate(results):
            if result.boxes is None or len(result.boxes) == 0:
                vectors.append(None)
                continue
            top = int(result.boxes.conf.argmax().item())
            # End-to-end postprocess preserves the score-sorted top-k order
            # under confidence filtering.  We still independently verify the
            # mapped readout logit against the returned confidence.
            flat = int(self.flat_indices[batch_index, top, 0].item())
            level, row, column = self._decode(flat)
            feature = (
                self.cap[level][batch_index, :, row, column]
                .float()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            logit = float(
                self.logit_cap[level][batch_index, 0, row, column]
                .float()
                .detach()
                .cpu()
                .item()
            )
            confidence = float(result.boxes.conf[top].detach().cpu().item())
            sigmoid = 1.0 / (1.0 + math.exp(-logit))
            vectors.append(
                {
                    "feature": feature,
                    "level": level,
                    "flat": flat,
                    "top": top,
                    "logit_conf_abs_diff": abs(sigmoid - confidence),
                }
            )
        return list(results), vectors


def _read_padded(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise PhaseCError(f"IMAGE_DECODE_FAILED: {path}")
    raw_shape = (int(image.shape[0]), int(image.shape[1]))
    padded = cv2.copyMakeBorder(
        image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101
    )
    return padded, raw_shape


def _extract_result(
    result: Any,
    vector: Mapping[str, Any] | None,
    raw_shape: tuple[int, int],
) -> dict[str, Any]:
    if vector is None or result.boxes is None or len(result.boxes) == 0:
        return {
            "valid": False,
            "confidence": math.nan,
            "box": np.full(4, np.nan, dtype=np.float32),
            "keypoints": np.full((9, 2), np.nan, dtype=np.float32),
            "kp_feature": np.full(26, np.nan, dtype=np.float32),
            "image_feature": np.full(64, np.nan, dtype=np.float32),
            "level": -1,
            "provenance_diff": math.nan,
            "detection_count": 0,
        }
    top = int(vector["top"])
    if result.keypoints is None:
        raise PhaseCError("POSE_RESULT_KEYPOINTS_MISSING")
    box = result.boxes.xyxy[top].detach().cpu().numpy().astype(np.float32) - PAD
    keypoints = (
        result.keypoints.xy[top].detach().cpu().numpy().astype(np.float32) - PAD
    )
    if keypoints.shape != (9, 2):
        raise PhaseCError(f"POSE_KEYPOINT_COUNT_INVALID: {keypoints.shape}")
    image_feature = np.asarray(vector["feature"], dtype=np.float32)
    if image_feature.shape != (64,):
        raise PhaseCError(f"IMAGE_FEATURE_CHANNELS_INVALID: {image_feature.shape}")
    return {
        "valid": True,
        "confidence": float(result.boxes.conf[top].detach().cpu().item()),
        "box": box,
        "keypoints": keypoints,
        "kp_feature": _geometry_feature(box, keypoints, raw_shape[0], raw_shape[1]),
        "image_feature": image_feature,
        "level": int(vector["level"]),
        "provenance_diff": float(vector["logit_conf_abs_diff"]),
        "detection_count": int(len(result.boxes)),
    }


def _shard_fingerprint(
    recipe_sha256: str,
    indices: np.ndarray,
    synth: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256(recipe_sha256.encode("ascii"))
    contiguous_indices = np.ascontiguousarray(indices, dtype=np.int32)
    digest.update(contiguous_indices.tobytes())
    for field in (
        "frame_ids",
        "images",
        "asset_ids",
        "splits",
        "dimensions_xyz",
        "labels",
        "raw_shapes",
    ):
        values = np.asarray(synth[field])[contiguous_indices]
        if values.dtype.kind in {"U", "S"}:
            for value in values.tolist():
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\0")
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _shard_matches(path: Path, expected_fingerprint: str, indices: np.ndarray) -> bool:
    try:
        with np.load(path, allow_pickle=False) as source:
            return bool(
                "contract_sha256" in source.files
                and str(source["contract_sha256"].item()) == expected_fingerprint
                and np.array_equal(source["indices"], indices)
            )
    except (OSError, ValueError):
        return False


def _save_shard(
    path: Path,
    indices: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    contract_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=np.asarray(indices, dtype=np.int32),
        contract_sha256=np.asarray(contract_sha256),
        valid=np.asarray([row["valid"] for row in rows], dtype=np.bool_),
        confidences=np.asarray([row["confidence"] for row in rows], dtype=np.float32),
        boxes=np.asarray([row["box"] for row in rows], dtype=np.float32),
        keypoints=np.asarray([row["keypoints"] for row in rows], dtype=np.float32),
        kp_features=np.asarray([row["kp_feature"] for row in rows], dtype=np.float32),
        image_features=np.asarray(
            [row["image_feature"] for row in rows], dtype=np.float32
        ),
        source_levels=np.asarray([row["level"] for row in rows], dtype=np.int8),
        provenance_diffs=np.asarray(
            [row["provenance_diff"] for row in rows], dtype=np.float64
        ),
        detection_counts=np.asarray(
            [row["detection_count"] for row in rows], dtype=np.int16
        ),
    )


def _load_shards(
    paths: Sequence[Path],
    count: int,
    expected_contracts: Mapping[Path, str] | None = None,
) -> dict[str, np.ndarray]:
    output = {
        "valid": np.zeros(count, dtype=np.bool_),
        "confidences": np.full(count, np.nan, dtype=np.float32),
        "boxes": np.full((count, 4), np.nan, dtype=np.float32),
        "keypoints": np.full((count, 9, 2), np.nan, dtype=np.float32),
        "kp_features": np.full((count, 26), np.nan, dtype=np.float32),
        "image_features": np.full((count, 64), np.nan, dtype=np.float32),
        "source_levels": np.full(count, -1, dtype=np.int8),
        "provenance_diffs": np.full(count, np.nan, dtype=np.float64),
        "detection_counts": np.zeros(count, dtype=np.int16),
    }
    seen = np.zeros(count, dtype=np.bool_)
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            if expected_contracts is not None and (
                "contract_sha256" not in source.files
                or str(source["contract_sha256"].item())
                != expected_contracts[path]
            ):
                raise PhaseCError(f"FEATURE_SHARD_CONTRACT_MISMATCH: {path}")
            indices = np.asarray(source["indices"], dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= count) or np.any(seen[indices]):
                raise PhaseCError(f"FEATURE_SHARD_INDEX_INVALID: {path}")
            seen[indices] = True
            for name in output:
                output[name][indices] = source[name]
    if not seen.all():
        raise PhaseCError(f"FEATURE_SHARDS_INCOMPLETE: missing={int((~seen).sum())}")
    return output


def extract_features(device: str) -> dict[str, Any]:
    import cv2
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "FEATURE_CACHE.npz"
    if cache_path.exists():
        raise PhaseCError(f"FEATURE_CACHE_ALREADY_EXISTS: {cache_path}")
    if _sha256(WEIGHTS) != CHECKPOINT_SHA256:
        raise PhaseCError("PRIMARY_CHECKPOINT_SHA_MISMATCH")
    checkpoint_mtime_before = WEIGHTS.stat().st_mtime_ns
    prerequisites = _prerequisite_audit()
    feature_recipe = _feature_recipe(prerequisites)
    synth = _load_synth_contract()
    tap = DynamicOne2OneTap(WEIGHTS, device)
    shard_dir = OUT_DIR / "feature_shards"
    shard_paths: list[Path] = []
    shard_contracts: dict[Path, str] = {}
    started = time.perf_counter()

    # Shape-homogeneous batches preserve the exact rectangular letterbox path
    # that a single-image inference would use while retaining GPU throughput.
    by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (height, width) in enumerate(synth["raw_shapes"].tolist()):
        by_shape[(int(height), int(width))].append(index)
    processed = 0
    for (height, width), shape_indices in sorted(by_shape.items()):
        for shard_start in range(0, len(shape_indices), SHARD_SIZE):
            shard_indices = np.asarray(
                shape_indices[shard_start : shard_start + SHARD_SIZE], dtype=np.int32
            )
            shard_path = shard_dir / (
                f"synth_{height}x{width}_{shard_start:05d}_{shard_start + len(shard_indices):05d}.npz"
            )
            shard_paths.append(shard_path)
            shard_contract = _shard_fingerprint(
                str(feature_recipe["sha256"]), shard_indices, synth
            )
            shard_contracts[shard_path] = shard_contract
            if shard_path.is_file() and _shard_matches(
                shard_path, shard_contract, shard_indices
            ):
                processed += len(shard_indices)
                continue
            rows: list[dict[str, Any]] = []
            for offset in range(0, len(shard_indices), SYNTH_BATCH):
                batch_indices = shard_indices[offset : offset + SYNTH_BATCH]
                images: list[np.ndarray] = []
                raw_shapes: list[tuple[int, int]] = []
                for index in batch_indices.tolist():
                    image, raw_shape = _read_padded(Path(str(synth["images"][index])))
                    if raw_shape != (height, width):
                        raise PhaseCError("SYNTH_METADATA_IMAGE_SHAPE_MISMATCH")
                    images.append(image)
                    raw_shapes.append(raw_shape)
                results, vectors = tap.predict_batch(images)
                rows.extend(
                    _extract_result(result, vector, raw_shape)
                    for result, vector, raw_shape in zip(results, vectors, raw_shapes)
                )
            _save_shard(shard_path, shard_indices, rows, shard_contract)
            processed += len(shard_indices)
            print(
                f"[{time.strftime('%H:%M:%S')}] synth features {processed}/40000",
                flush=True,
            )

    synth_pred = _load_shards(shard_paths, 40_000, shard_contracts)

    # Real DEV is intentionally inferred one frame at a time: this must be
    # byte-identical to the immutable Phase-A prediction cache.
    phase_a_cache_path = PHASE_A_DIR / "PREDICTION_CACHE.npz"
    with np.load(phase_a_cache_path, allow_pickle=False) as source:
        phase_a = {name: np.asarray(source[name]) for name in source.files}
    real_rows: list[dict[str, Any]] = []
    max_diff = {"candidate_count": 0, "confidence": 0.0, "box": 0.0, "keypoints": 0.0}
    for index, image_rel in enumerate(phase_a["images"].tolist()):
        image, raw_shape = _read_padded(REPO_ROOT / str(image_rel))
        results, vectors = tap.predict_batch([image])
        row = _extract_result(results[0], vectors[0], raw_shape)
        real_rows.append(row)
        max_diff["candidate_count"] = max(
            max_diff["candidate_count"],
            abs(int(row["detection_count"]) - int(phase_a["detection_counts"][index])),
        )
        expected_valid = bool(np.isfinite(phase_a["confidences"][index]))
        if bool(row["valid"]) != expected_valid:
            max_diff["candidate_count"] = max(max_diff["candidate_count"], 1)
        if expected_valid and row["valid"]:
            max_diff["confidence"] = max(
                max_diff["confidence"],
                abs(float(row["confidence"]) - float(phase_a["confidences"][index])),
            )
            max_diff["box"] = max(
                max_diff["box"],
                float(
                    np.max(
                        np.abs(
                            np.asarray(row["box"], dtype=np.float64)
                            - np.asarray(phase_a["boxes_xyxy"][index], dtype=np.float64)
                        )
                    )
                ),
            )
            max_diff["keypoints"] = max(
                max_diff["keypoints"],
                float(
                    np.max(
                        np.abs(
                            np.asarray(row["keypoints"], dtype=np.float64)
                            - np.asarray(phase_a["keypoints_xy"][index], dtype=np.float64)
                        )
                    )
                ),
            )
        if (index + 1) % 25 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] real feature parity {index + 1}/185",
                flush=True,
            )

    real_pred = {
        "valid": np.asarray([row["valid"] for row in real_rows], dtype=np.bool_),
        "confidences": np.asarray(
            [row["confidence"] for row in real_rows], dtype=np.float32
        ),
        "boxes": np.asarray([row["box"] for row in real_rows], dtype=np.float32),
        "keypoints": np.asarray(
            [row["keypoints"] for row in real_rows], dtype=np.float32
        ),
        "kp_features": np.asarray(
            [row["kp_feature"] for row in real_rows], dtype=np.float32
        ),
        "image_features": np.asarray(
            [row["image_feature"] for row in real_rows], dtype=np.float32
        ),
        "source_levels": np.asarray(
            [row["level"] for row in real_rows], dtype=np.int8
        ),
        "provenance_diffs": np.asarray(
            [row["provenance_diff"] for row in real_rows], dtype=np.float64
        ),
        "detection_counts": np.asarray(
            [row["detection_count"] for row in real_rows], dtype=np.int16
        ),
    }
    if any(float(value) != 0.0 for value in max_diff.values()):
        raise PhaseCError(f"INSTRUMENTATION_PREDICTION_PARITY_FAILED: {max_diff}")

    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    dimensions_by_type = {
        PLASTIC_OBJECT_TYPE: registry.resolve(PLASTIC_OBJECT_TYPE).physical_dimensions,
        WOOD_OBJECT_TYPE: registry.resolve(WOOD_OBJECT_TYPE).physical_dimensions,
    }
    real_dimensions = np.asarray(
        [
            [
                dimensions_by_type[str(object_type)].x_m,
                dimensions_by_type[str(object_type)].y_m,
                dimensions_by_type[str(object_type)].z_m,
            ]
            for object_type in phase_a["object_types"].tolist()
        ],
        dtype=np.float32,
    )

    train_valid = (synth["splits"] == SPLIT_CODE["TRAIN"]) & synth_pred["valid"]
    dim_raw = _dimension_raw(synth["dimensions_xyz"])
    normalization: dict[str, dict[str, list[float]]] = {}
    for name, values in (
        ("image", synth_pred["image_features"]),
        ("kp", synth_pred["kp_features"]),
        ("dims", dim_raw),
    ):
        selected = np.asarray(values[train_valid], dtype=np.float64)
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std[std < 1.0e-8] = 1.0
        normalization[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source": "SYNTHETIC_TRAIN_VALID_ONLY",
        }

    provenance_values = np.concatenate(
        [
            synth_pred["provenance_diffs"][synth_pred["valid"]],
            real_pred["provenance_diffs"][real_pred["valid"]],
        ]
    )
    provenance_max = float(np.max(provenance_values))
    if provenance_max > 1.0e-5:
        raise PhaseCError(
            f"TOPK_FEATURE_CELL_PROVENANCE_FAILED: max_abs_diff={provenance_max}"
        )
    tap_audit = {
        "schema_version": "dimension_conditioning_feature_tap_audit_v1",
        "generated_at_utc": _now(),
        "discovery": {
            "rule": (
                "unique named_modules entry exposing one2one_cv3, one2one_cv4, "
                "get_topk_index, nl and stride"
            ),
            "resolved_head_path": tap.head_path,
            "resolved_head_type": type(tap.head).__name__,
            "level_count": tap.level_count,
            "penultimate_paths": tap.penultimate_paths,
            "logit_paths": tap.logit_paths,
            "hardcoded_module_path_used": False,
            "selected_feature": "one2one classification penultimate at final top-ranked source cell",
            "selection_reason": (
                "actual end-to-end inference branch; uniform 64 channels at P3/P4/P5; "
                "read-only candidate-aligned vector"
            ),
        },
        "instrumentation_prediction_parity": {
            "reference": _display(phase_a_cache_path),
            "n_frames": int(len(phase_a["frame_ids"])),
            "max_abs_diff": max_diff,
            "required_tolerance": 0.0,
            "passed": True,
        },
        "candidate_provenance": {
            "check": "sigmoid(mapped one2one readout logit) vs returned confidence",
            "n": int(len(provenance_values)),
            "max_abs_diff": provenance_max,
            "passed": True,
        },
        "feature_dimensions": {"image": 64, "kp_geometry": 26, "dimensions": 4},
        "candidate_scope": {
            "rule": "highest-confidence final detection per frame",
            "reason": (
                "matches the frozen Phase-A and deployment one-pose contract; "
                "parity is not trained on lower-ranked distractor detections"
            ),
            "synthetic": {
                "frames": 40_000,
                "total_final_detections": int(synth_pred["detection_counts"].sum()),
                "frames_with_multiple_detections": int(
                    np.sum(synth_pred["detection_counts"] > 1)
                ),
                "cached_top_candidate_rows": int(synth_pred["valid"].sum()),
            },
            "real": {
                "frames": int(len(real_pred["detection_counts"])),
                "total_final_detections": int(real_pred["detection_counts"].sum()),
                "frames_with_multiple_detections": int(
                    np.sum(real_pred["detection_counts"] > 1)
                ),
                "cached_top_candidate_rows": int(real_pred["valid"].sum()),
            },
        },
        "prerequisites": prerequisites,
        "feature_recipe": feature_recipe,
        "kp_geometry_definition": {
            "center": "predicted bbox center",
            "normalizer": "predicted bbox diagonal",
            "coordinates": 18,
            "scalars": ["bbox width/height", "bbox area/raw image area"],
            "structural": [
                "mean camera-facing width edge / bbox diagonal",
                "mean height edge / bbox diagonal",
                "mean depth edge / bbox diagonal",
                "width edge / height edge",
                "depth edge / height edge",
                "width edge / depth edge",
            ],
        },
        "normalization": normalization,
        "yolo_requires_grad_true": int(
            sum(int(parameter.requires_grad) for parameter in tap.model.parameters())
        ),
        "yolo_optimizer_membership": 0,
    }
    tap.close()
    del tap
    torch.cuda.empty_cache()

    metadata = {
        "schema_version": "dimension_conditioning_feature_cache_v1",
        "generated_at_utc": _now(),
        "checkpoint": {
            "path": _display(WEIGHTS),
            "sha256": CHECKPOINT_SHA256,
        },
        "phase_b_split_membership_sha256": _sha256(
            PHASE_B_DIR / "SPLIT_MEMBERSHIP.json"
        ),
        "phase_a_prediction_cache_sha256": _sha256(phase_a_cache_path),
        "geometry_registry_sha256": registry.sha256,
        "prerequisites": prerequisites,
        "feature_recipe": feature_recipe,
        "synthetic_count": 40_000,
        "real_count": int(len(phase_a["frame_ids"])),
        "synthetic_valid_count": int(synth_pred["valid"].sum()),
        "real_valid_count": int(real_pred["valid"].sum()),
        "real_gt_fields": [],
        "feature_source": "dynamic one2one classification penultimate",
        "normalization": normalization,
        "elapsed_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        cache_path,
        metadata_json=np.asarray(_json_text(metadata)),
        synth_frame_ids=synth["frame_ids"],
        synth_asset_ids=synth["asset_ids"],
        synth_splits=synth["splits"],
        synth_dimensions_xyz=synth["dimensions_xyz"],
        synth_labels=synth["labels"],
        synth_raw_shapes=synth["raw_shapes"],
        synth_valid=synth_pred["valid"],
        synth_confidences=synth_pred["confidences"],
        synth_boxes_xyxy=synth_pred["boxes"],
        synth_keypoints_xy=synth_pred["keypoints"],
        synth_kp_features=synth_pred["kp_features"],
        synth_image_features=synth_pred["image_features"],
        synth_source_levels=synth_pred["source_levels"],
        synth_detection_counts=synth_pred["detection_counts"],
        real_frame_ids=phase_a["frame_ids"],
        real_object_types=phase_a["object_types"],
        real_sessions=phase_a["sessions"],
        real_domains=phase_a["domains"],
        real_images=phase_a["images"],
        real_labels_paths=phase_a["labels"],
        real_dimensions_xyz=real_dimensions,
        real_valid=real_pred["valid"],
        real_confidences=real_pred["confidences"],
        real_boxes_xyxy=real_pred["boxes"],
        real_keypoints_xy=real_pred["keypoints"],
        real_kp_features=real_pred["kp_features"],
        real_image_features=real_pred["image_features"],
        real_source_levels=real_pred["source_levels"],
        real_detection_counts=real_pred["detection_counts"],
    )
    checkpoint_sha_after = _sha256(WEIGHTS)
    if (
        checkpoint_sha_after != CHECKPOINT_SHA256
        or WEIGHTS.stat().st_mtime_ns != checkpoint_mtime_before
    ):
        raise PhaseCError("YOLO_CHECKPOINT_CHANGED_DURING_EXTRACTION")
    tap_audit["feature_cache"] = {
        "path": _display(cache_path),
        "sha256": _sha256(cache_path),
    }
    tap_audit["checkpoint_unchanged"] = True
    (OUT_DIR / "FEATURE_TAP_AUDIT.json").write_text(
        _json_text(tap_audit), "utf-8"
    )
    return tap_audit


def _load_feature_cache() -> dict[str, np.ndarray | dict[str, Any]]:
    path = OUT_DIR / "FEATURE_CACHE.npz"
    if not path.is_file():
        raise PhaseCError("FEATURE_CACHE_REQUIRED")
    with np.load(path, allow_pickle=False) as source:
        arrays: dict[str, Any] = {name: np.asarray(source[name]) for name in source.files}
    arrays["metadata"] = json.loads(str(arrays["metadata_json"].item()))
    if arrays["metadata"].get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise PhaseCError("FEATURE_CACHE_CHECKPOINT_MISMATCH")
    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    expected_bindings = {
        "phase_b_split_membership_sha256": _sha256(
            PHASE_B_DIR / "SPLIT_MEMBERSHIP.json"
        ),
        "phase_a_prediction_cache_sha256": _sha256(
            PHASE_A_DIR / "PREDICTION_CACHE.npz"
        ),
        "geometry_registry_sha256": registry.sha256,
    }
    for field, expected in expected_bindings.items():
        if arrays["metadata"].get(field) != expected:
            raise PhaseCError(f"FEATURE_CACHE_{field.upper()}_MISMATCH")
    prerequisites = _prerequisite_audit()
    expected_recipe = _feature_recipe(prerequisites)
    if arrays["metadata"].get("feature_recipe", {}).get("sha256") != expected_recipe[
        "sha256"
    ]:
        raise PhaseCError("FEATURE_CACHE_RECIPE_FINGERPRINT_MISMATCH")
    return arrays


def _normalization(cache: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    payload = cache["metadata"]["normalization"][name]
    return np.asarray(payload["mean"], dtype=np.float32), np.asarray(
        payload["std"], dtype=np.float32
    )


def _normalized_features(
    cache: Mapping[str, Any], *, real: bool, dimensions_override: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    prefix = "real" if real else "synth"
    output: dict[str, np.ndarray] = {}
    for name, field in (("image", "image_features"), ("kp", "kp_features")):
        values = np.asarray(cache[f"{prefix}_{field}"], dtype=np.float32)
        mean, std = _normalization(cache, name)
        output[name] = (values - mean) / std
    dimensions = (
        np.asarray(dimensions_override, dtype=np.float32)
        if dimensions_override is not None
        else np.asarray(cache[f"{prefix}_dimensions_xyz"], dtype=np.float32)
    )
    raw = _dimension_raw(dimensions)
    mean, std = _normalization(cache, "dims")
    output["dims"] = (raw - mean) / std
    return output


def _arm_matrix(features: Mapping[str, np.ndarray], arm: str) -> np.ndarray:
    return np.concatenate([features[name] for name in ARM_FEATURES[arm]], axis=1).astype(
        np.float32
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid_mask = (
        np.isfinite(probabilities)
        if valid is None
        else np.asarray(valid, dtype=np.bool_) & np.isfinite(probabilities)
    )
    predicted = np.full(len(labels), -1, dtype=np.int8)
    predicted[valid_mask] = (probabilities[valid_mask] >= 0.5).astype(np.int8)
    confusion = np.zeros((2, 3), dtype=np.int64)
    for truth, prediction in zip(labels.tolist(), predicted.tolist()):
        confusion[int(truth), int(prediction) if prediction >= 0 else 2] += 1
    accuracy = float(np.mean(predicted == labels)) if len(labels) else None
    recalls: list[float | None] = []
    for truth in (SHORT_FRONT, LONG_FRONT):
        denominator = int(np.sum(labels == truth))
        recalls.append(
            float(np.sum((labels == truth) & (predicted == truth)) / denominator)
            if denominator
            else None
        )
    present_recalls = [value for value in recalls if value is not None]
    balanced = float(np.mean(present_recalls)) if present_recalls else None
    usable_labels = labels[valid_mask]
    usable_scores = probabilities[valid_mask]
    positives = int(np.sum(usable_labels == LONG_FRONT))
    negatives = int(np.sum(usable_labels == SHORT_FRONT))
    auroc: float | None = None
    if positives and negatives:
        ranks = _average_ranks(usable_scores)
        auroc = float(
            (
                ranks[usable_labels == LONG_FRONT].sum()
                - positives * (positives + 1) / 2.0
            )
            / (positives * negatives)
        )
    return {
        "N": int(len(labels)),
        "valid_n": int(valid_mask.sum()),
        "coverage": float(valid_mask.mean()) if len(labels) else None,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "auroc": auroc,
        "confusion_matrix_rows_true_short_long_cols_pred_short_long_abstain": confusion.tolist(),
        "recall_short": recalls[0] if recalls else None,
        "recall_long": recalls[1] if recalls else None,
        "balanced_accuracy_one_class_policy": (
            "mean over classes present in this subgroup"
        ),
    }


def _make_model(architecture: str, input_dim: int):
    import torch.nn as nn

    if architecture == "LINEAR":
        return nn.Linear(input_dim, 2)
    if architecture == "MLP":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )
    raise PhaseCError(f"UNKNOWN_PROBE_ARCHITECTURE: {architecture}")


def _predict_probabilities(model: Any, values: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    output = np.empty(len(values), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy(values[start : start + 4096]).to(device)
            output[start : start + len(batch)] = (
                torch.softmax(model(batch), dim=1)[:, LONG_FRONT].detach().cpu().numpy()
            )
    return output


def _train_run(
    *,
    arm: str,
    architecture: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    device: str,
    out_path: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = _make_model(architecture, int(train_x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    class_weights_np = len(train_y) / (2.0 * counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    sample_weights = class_weights_np[train_y]
    steps_per_epoch = int(math.ceil(len(train_y) / TRAIN_BATCH))
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    train_tensor = torch.from_numpy(train_x)
    label_tensor = torch.from_numpy(train_y.astype(np.int64))
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + epoch)
        indices = torch.multinomial(
            torch.from_numpy(sample_weights),
            num_samples=steps_per_epoch * TRAIN_BATCH,
            replacement=True,
            generator=generator,
        )
        losses: list[float] = []
        for step in range(steps_per_epoch):
            selected = indices[step * TRAIN_BATCH : (step + 1) * TRAIN_BATCH]
            values = train_tensor[selected].to(device, non_blocking=True)
            labels = label_tensor[selected].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        dev_probability = _predict_probabilities(model, dev_x, device)
        dev_metrics = _binary_metrics(dev_y, dev_probability)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "dev_accuracy": dev_metrics["accuracy"],
        }
        history.append(record)
        score = float(dev_metrics["balanced_accuracy"])
        if best is None or score > float(best["score"]):
            best = {
                "score": score,
                "epoch": epoch + 1,
                "state_dict": copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }
    if best is None:
        raise PhaseCError("PROBE_TRAINING_NO_CHECKPOINT")
    model.load_state_dict(best["state_dict"])
    dev_probability = _predict_probabilities(model, dev_x, device)
    payload = {
        "schema_version": "dimension_conditioning_probe_run_v1",
        "arm": arm,
        "architecture": architecture,
        "seed": seed,
        "input_dim": int(train_x.shape[1]),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "balanced_cross_entropy_weights": class_weights_np.tolist(),
        "class_balanced_sampler": True,
        "fixed_epochs_executed": TRAIN_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "fixed_optimizer_steps": TRAIN_EPOCHS * steps_per_epoch,
        "checkpoint_epoch_selected_by": "SYNTHETIC_DEV_BALANCED_ACCURACY_ONLY",
        "selected_epoch": int(best["epoch"]),
        "synthetic_dev": _binary_metrics(dev_y, dev_probability),
        "synthetic_test": {
            "status": "SEALED_UNTIL_SOURCE_ONLY_SELECTION_LOCK",
            "opened": False,
        },
        "history": history,
        "real_accessed_during_training_or_selection": False,
        "yolo_parameters_in_optimizer": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best["state_dict"],
            "architecture": architecture,
            "input_dim": int(train_x.shape[1]),
            "arm": arm,
            "seed": seed,
            "selected_epoch": int(best["epoch"]),
        },
        out_path,
    )
    payload["checkpoint"] = {
        "path": _display(out_path),
        "sha256": _sha256(out_path),
    }
    out_path.with_suffix(".json").write_text(_json_text(payload), "utf-8")
    return payload


def _load_probe(path: Path, device: str):
    import torch

    payload = torch.load(path, map_location="cpu")
    model = _make_model(str(payload["architecture"]), int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def _real_truth(cache: Mapping[str, Any]) -> np.ndarray:
    """This function is called only after SELECTION_LOCK.json exists."""

    lock_path = OUT_DIR / "SELECTION_LOCK.json"
    if not lock_path.is_file():
        raise PhaseCError("SOURCE_ONLY_SELECTION_LOCK_REQUIRED_BEFORE_REAL_GT")
    sys.path.insert(0, str(PHASE_A_DIR))
    import run_phase_a as phase_a

    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    symmetry_contract, _pose_contract_evidence = phase_a._plastic_pose_contract()
    labels: list[int] = []
    for frame_id, label_rel, object_type in zip(
        cache["real_frame_ids"].tolist(),
        cache["real_labels_paths"].tolist(),
        cache["real_object_types"].tolist(),
    ):
        spec = registry.resolve(str(object_type))
        if str(object_type) == PLASTIC_OBJECT_TYPE:
            truth = phase_a._truth(
                label_path=REPO_ROOT / str(label_rel),
                frame_id=str(frame_id),
                spec=spec,
                equivalent_rotations=symmetry_contract.rotations,
            )
            expected = str(truth["expected_parity"])
        else:
            # Wood symmetry/pose remains blocked, but its camera-facing parity
            # label is directly auditable from the GT-v2 physical dimensions.
            payload = _read_json(REPO_ROOT / str(label_rel))
            objects = payload.get("objects")
            if not isinstance(objects, list) or len(objects) != 1:
                raise PhaseCError(f"WOOD_GT_OBJECT_COUNT_INVALID: {frame_id}")
            obj = objects[0]
            observed = obj.get("physical_dimensions_m")
            if observed != spec.physical_dimensions_m:
                raise PhaseCError(f"WOOD_GT_REGISTRY_DIMENSION_MISMATCH: {frame_id}")
            camera_dimensions = obj.get("camera_facing_pnp", {}).get("dimensions_m")
            if not isinstance(camera_dimensions, Mapping):
                raise PhaseCError(f"WOOD_GT_CAMERA_DIMENSIONS_MISSING: {frame_id}")
            width = float(camera_dimensions["width"])
            depth = float(camera_dimensions["depth"])
            if width == depth:
                raise PhaseCError(f"WOOD_GT_PARITY_TIE: {frame_id}")
            expected = PARITY_NAME[LONG_FRONT if width > depth else SHORT_FRONT]
        if expected == PARITY_NAME[SHORT_FRONT]:
            labels.append(SHORT_FRONT)
        elif expected == PARITY_NAME[LONG_FRONT]:
            labels.append(LONG_FRONT)
        else:
            raise PhaseCError(f"REAL_PARITY_LABEL_INVALID: {frame_id}: {expected}")
    return np.asarray(labels, dtype=np.int8)


def _common_plastic_frame_ids(cache: Mapping[str, Any]) -> frozenset[str]:
    payload = _read_json(COMMON_PLASTIC_MANIFEST)
    items = payload.get("items")
    if payload.get("population_id") != "COMMON_DEV_PLASTIC_POS128":
        raise PhaseCError("COMMON_PLASTIC_POPULATION_ID_MISMATCH")
    if not isinstance(items, list) or len(items) != 128:
        raise PhaseCError("COMMON_PLASTIC_MEMBERSHIP_COUNT_MISMATCH")
    frame_ids = tuple(str(item.get("frame_id")) for item in items)
    if len(set(frame_ids)) != 128:
        raise PhaseCError("COMMON_PLASTIC_MEMBERSHIP_NOT_UNIQUE")
    if any(str(item.get("object_type")) != PLASTIC_OBJECT_TYPE for item in items):
        raise PhaseCError("COMMON_PLASTIC_OBJECT_TYPE_MISMATCH")
    cached = {
        str(frame_id): str(object_type)
        for frame_id, object_type in zip(
            cache["real_frame_ids"].tolist(), cache["real_object_types"].tolist()
        )
    }
    if any(cached.get(frame_id) != PLASTIC_OBJECT_TYPE for frame_id in frame_ids):
        raise PhaseCError("COMMON_PLASTIC_NOT_EXACT_CACHE_SUBSET")
    return frozenset(frame_ids)


def _common_multishape_indices(cache: Mapping[str, Any]) -> np.ndarray:
    payload = _read_json(COMMON_MULTISHAPE_MANIFEST)
    items = payload.get("items")
    if payload.get("population_id") != "COMMON_DEV_MULTISHAPE_POS":
        raise PhaseCError("COMMON_MULTISHAPE_POPULATION_ID_MISMATCH")
    if not isinstance(items, list) or len(items) != 173:
        raise PhaseCError("COMMON_MULTISHAPE_MEMBERSHIP_COUNT_MISMATCH")
    expected_ids = tuple(str(item.get("frame_id")) for item in items)
    if len(set(expected_ids)) != 173:
        raise PhaseCError("COMMON_MULTISHAPE_MEMBERSHIP_NOT_UNIQUE")
    cached_ids = tuple(str(frame_id) for frame_id in cache["real_frame_ids"].tolist())
    index_by_id = {frame_id: index for index, frame_id in enumerate(cached_ids)}
    if any(frame_id not in index_by_id for frame_id in expected_ids):
        raise PhaseCError("COMMON_MULTISHAPE_NOT_EXACT_CACHE_SUBSET")
    indices = np.asarray([index_by_id[frame_id] for frame_id in expected_ids], dtype=np.int64)
    cached_types = np.asarray(cache["real_object_types"])
    expected_types = tuple(str(item.get("object_type")) for item in items)
    if tuple(str(value) for value in cached_types[indices].tolist()) != expected_types:
        raise PhaseCError("COMMON_MULTISHAPE_OBJECT_TYPE_ORDER_MISMATCH")
    return indices


def _real_groups(cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    object_types = np.asarray(cache["real_object_types"])
    frame_ids = np.asarray(cache["real_frame_ids"])
    domains = np.asarray(cache["real_domains"])
    sessions = np.asarray(cache["real_sessions"])
    common_ids = _common_plastic_frame_ids(cache)
    common = np.asarray([str(frame_id) in common_ids for frame_id in frame_ids])
    groups: dict[str, np.ndarray] = {
        # PLASTIC:ALL is retained as the Phase-A selector-diagnostic DEV140
        # view.  Phase-C comparisons and gates use the explicit COMMON128 keys.
        "PLASTIC:ALL": object_types == PLASTIC_OBJECT_TYPE,
        "PLASTIC:DAY": (object_types == PLASTIC_OBJECT_TYPE) & (domains == "DAY"),
        "PLASTIC:NIGHT": (object_types == PLASTIC_OBJECT_TYPE) & (domains == "NIGHT"),
        "PLASTIC:COMMON128:ALL": common,
        "PLASTIC:COMMON128:DAY": common & (domains == "DAY"),
        "PLASTIC:COMMON128:NIGHT": common & (domains == "NIGHT"),
        "WOOD:ALL": object_types == WOOD_OBJECT_TYPE,
    }
    for session in sorted(set(sessions.tolist())):
        plastic = (object_types == PLASTIC_OBJECT_TYPE) & (sessions == session)
        wood = (object_types == WOOD_OBJECT_TYPE) & (sessions == session)
        if plastic.any():
            groups[f"PLASTIC:SESSION:{session}"] = plastic
        common_session = common & (sessions == session)
        if common_session.any():
            groups[f"PLASTIC:COMMON128:SESSION:{session}"] = common_session
        if wood.any():
            groups[f"WOOD:SESSION:{session}"] = wood
    return groups


def _evaluate_real_parity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        name: _binary_metrics(labels[mask], probabilities[mask], valid[mask])
        for name, mask in _real_groups(cache).items()
    }


def _shuffled_dimensions(dimensions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(dimensions))
    # Deterministically repair fixed points, so every frame truly receives a
    # different frame's metadata even when dimensions happen to be equal.
    fixed = np.flatnonzero(permutation == np.arange(len(dimensions)))
    if len(fixed) == 1:
        other = 0 if fixed[0] != 0 else 1
        permutation[fixed[0]], permutation[other] = permutation[other], permutation[fixed[0]]
    elif len(fixed) > 1:
        permutation[fixed] = np.roll(permutation[fixed], 1)
    return np.asarray(dimensions)[permutation], permutation.tolist()


def _wrong_type_dimensions(
    cache: Mapping[str, Any], dimensions: np.ndarray
) -> np.ndarray:
    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    plastic = registry.resolve(PLASTIC_OBJECT_TYPE).physical_dimensions
    wood = registry.resolve(WOOD_OBJECT_TYPE).physical_dimensions
    output = np.empty_like(dimensions, dtype=np.float32)
    for index, object_type in enumerate(cache["real_object_types"].tolist()):
        wrong = wood if str(object_type) == PLASTIC_OBJECT_TYPE else plastic
        output[index] = (wrong.x_m, wrong.y_m, wrong.z_m)
    return output


def _pose_diagnostic(
    cache: Mapping[str, Any],
    selected_by_arm: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(PHASE_A_DIR))
    import run_phase_a as phase_a

    registry = load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    plastic = registry.resolve(PLASTIC_OBJECT_TYPE)
    symmetry_contract, _pose_contract_evidence = phase_a._plastic_pose_contract()
    plastic_mask = np.asarray(cache["real_object_types"]) == PLASTIC_OBJECT_TYPE
    indices = np.flatnonzero(plastic_mask)
    with np.load(PHASE_A_DIR / "PREDICTION_CACHE.npz", allow_pickle=False) as source:
        phase_a_intrinsics = np.asarray(source["camera_intrinsics"], dtype=np.float64)
    all_rows: list[dict[str, Any]] = []
    common_ids = _common_plastic_frame_ids(cache)
    for arm, selected in selected_by_arm.items():
        for index in indices.tolist():
            frame_id = str(cache["real_frame_ids"][index])
            truth = phase_a._truth(
                label_path=REPO_ROOT / str(cache["real_labels_paths"][index]),
                frame_id=frame_id,
                spec=plastic,
                equivalent_rotations=symmetry_contract.rotations,
            )
            expected = str(truth["expected_parity"])
            points = np.asarray(cache["real_keypoints_xy"][index], dtype=np.float64)
            # Intrinsics remain in the immutable Phase-A cache; GT labels are
            # never passed into the selector or classifier.
            camera = phase_a_intrinsics[index]
            hypothesis = None
            selected_name: str | None = None
            if bool(cache["real_valid"][index]) and int(selected[index]) in PARITY_NAME:
                selected_name = PARITY_NAME[int(selected[index])]
                selection = select_pnp_hypotheses(
                    points, camera, plastic.physical_dimensions, SelectorConfig()
                )
                hypothesis = phase_a._hypothesis_by_name(selection, selected_name)
            row = phase_a._arm_row(
                arm=arm,
                frame_id=frame_id,
                domain=str(cache["real_domains"][index]),
                session=str(cache["real_sessions"][index]),
                object_type=PLASTIC_OBJECT_TYPE,
                detection_count=int(cache["real_detection_counts"][index]),
                confidence=(
                    float(cache["real_confidences"][index])
                    if bool(cache["real_valid"][index])
                    else None
                ),
                expected_parity=expected,
                selected_parity=selected_name,
                parity_correct=selected_name == expected,
                hypothesis=hypothesis,
                target_transforms=truth["target_transforms"],
                true_dimensions=plastic.physical_dimensions,
                paper_eligible=arm
                in {"B1_KP_ONLY", "B2_IMAGE_KP", "B3_IMAGE_KP_DIMS"},
                gt_used_for_parity=False,
                gt_used_for_selection=False,
                keypoint_source="FEATURE_CACHE.real_keypoints_xy",
            )
            all_rows.append(row)
    metrics: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for arm_index, arm in enumerate(selected_by_arm):
        arm_rows = [row for row in all_rows if row["arm"] == arm]
        groups = phase_a._subgroups(arm_rows)
        groups["COMMON128"] = [
            row for row in arm_rows if str(row["frame_id"]) in common_ids
        ]
        metrics[arm] = {
            name: phase_a.summarize_rows(group) for name, group in groups.items()
        }
        metrics[arm]["WOOD"] = {
            "status": "BLOCKED",
            "N": 45,
            "blocked_reasons": [
                "WOOD_SYMMETRY_UNREVIEWED",
                "WOOD_INTRINSICS_SENSOR_PROFILE_SCALED_NOT_APPROVED",
            ],
        }
        bootstrap[arm] = {}
        for group_index, (name, group) in enumerate(groups.items()):
            bootstrap[arm][name] = phase_a._bootstrap_one(
                group,
                seed=phase_a.BOOTSTRAP_SEED + arm_index * 100 + group_index,
            )
        bootstrap[arm]["WOOD"] = {
            "status": "BLOCKED",
            "iterations": 0,
            "metrics": {},
        }
    return metrics, bootstrap


def _phase_a_common_metrics(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Re-summarize frozen Phase-A rows on the model-comparison population."""

    sys.path.insert(0, str(PHASE_A_DIR))
    import run_phase_a as phase_a

    common_ids = _common_plastic_frame_ids(cache)
    wanted_arms = ("A0_CURRENT_SELECTOR", "A1_GT_PARITY_ORACLE")
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in wanted_arms}
    with (PHASE_A_DIR / "ORACLE_PER_FRAME.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for raw in csv.DictReader(handle):
            arm = str(raw["arm"])
            if arm not in wanted_arms or str(raw["frame_id"]) not in common_ids:
                continue
            row: dict[str, Any] = dict(raw)
            for field in ("parity_correct", "pnp_solved", "pose_valid"):
                row[field] = str(raw[field]).lower() == "true"
            for field in (
                "restricted_adds_normalized",
                "rotation_error_deg",
                "translation_error_m",
                "yaw_error_deg",
            ):
                row[field] = float(raw[field]) if raw[field] != "" else None
            by_arm[arm].append(row)
    if any(len(rows) != 128 for rows in by_arm.values()):
        raise PhaseCError("PHASE_A_COMMON128_ROWS_INCOMPLETE")
    return {
        "population_id": "COMMON_DEV_PLASTIC_POS128",
        "count": 128,
        "manifest": _display(COMMON_PLASTIC_MANIFEST),
        "manifest_sha256": _sha256(COMMON_PLASTIC_MANIFEST),
        "phase_a_per_frame": _display(PHASE_A_DIR / "ORACLE_PER_FRAME.csv"),
        "phase_a_per_frame_sha256": _sha256(PHASE_A_DIR / "ORACLE_PER_FRAME.csv"),
        "metrics": {
            arm: phase_a.summarize_rows(rows) for arm, rows in by_arm.items()
        },
    }


def _comparison_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "arm",
        "architecture",
        "seed",
        "selected_source_only",
        "synth_test_accuracy",
        "synth_test_balanced_accuracy",
        "synth_test_auroc",
        "plastic_model_comparison_population",
        "plastic_common128_accuracy",
        "plastic_dev140_diagnostic_accuracy",
        "plastic_night_accuracy",
        "wood_all_accuracy",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def train_and_evaluate(device: str) -> dict[str, Any]:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_lock_path = OUT_DIR / "SELECTION_LOCK.json"
    resume_after_source_lock = selection_lock_path.is_file()
    checkpoint_mtime_before = WEIGHTS.stat().st_mtime_ns
    cache = _load_feature_cache()
    prerequisites = _prerequisite_audit()
    scope_lock = prerequisites["scope_lock"]
    synth_features = _normalized_features(cache, real=False)
    synth_valid = np.asarray(cache["synth_valid"], dtype=np.bool_)
    splits = np.asarray(cache["synth_splits"], dtype=np.int8)
    labels = np.asarray(cache["synth_labels"], dtype=np.int64)
    masks = {
        split: synth_valid & (splits == code) for split, code in SPLIT_CODE.items()
    }
    run_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if resume_after_source_lock:
        for arm in ARM_FEATURES:
            for architecture in ARCHITECTURES:
                for seed in SEEDS:
                    run_path = OUT_DIR / arm / f"{architecture.lower()}_seed{seed}.json"
                    checkpoint_path = run_path.with_suffix(".pt")
                    if not run_path.is_file() or not checkpoint_path.is_file():
                        raise PhaseCError(f"POST_LOCK_RESUME_RUN_MISSING: {run_path}")
                    payload = _read_json(run_path)
                    if (
                        payload.get("arm") != arm
                        or payload.get("architecture") != architecture
                        or payload.get("seed") != seed
                        or payload.get("real_accessed_during_training_or_selection")
                        is not False
                        or payload.get("checkpoint", {}).get("sha256")
                        != _sha256(checkpoint_path)
                    ):
                        raise PhaseCError(f"POST_LOCK_RESUME_RUN_CONTRACT_MISMATCH: {run_path}")
                    run_payloads[arm].append(payload)
        selection_lock = _read_json(selection_lock_path)
        if (
            selection_lock.get("schema_version")
            != "dimension_conditioning_source_only_selection_lock_v1"
            or selection_lock.get("selection_source") != "SYNTHETIC_DEV_ONLY"
            or selection_lock.get("real_labels_opened_before_lock") is not False
        ):
            raise PhaseCError("POST_LOCK_RESUME_SELECTION_LOCK_INVALID")
    else:
        for arm in ARM_FEATURES:
            matrix = _arm_matrix(synth_features, arm)
            arm_dir = OUT_DIR / arm
            for architecture in ARCHITECTURES:
                for seed in SEEDS:
                    path = arm_dir / f"{architecture.lower()}_seed{seed}.pt"
                    payload = _train_run(
                        arm=arm,
                        architecture=architecture,
                        seed=seed,
                        train_x=matrix[masks["TRAIN"]],
                        train_y=labels[masks["TRAIN"]],
                        dev_x=matrix[masks["DEV"]],
                        dev_y=labels[masks["DEV"]],
                        device=device,
                        out_path=path,
                    )
                    run_payloads[arm].append(payload)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] trained {arm} {architecture} seed={seed} "
                        f"dev_bal={payload['synthetic_dev']['balanced_accuracy']:.4f}",
                        flush=True,
                    )
        selected_for_lock: dict[str, dict[str, Any]] = {}
        for arm, runs in run_payloads.items():
            ordered = sorted(
                runs,
                key=lambda row: (
                    -float(row["synthetic_dev"]["balanced_accuracy"]),
                    0 if row["architecture"] == "LINEAR" else 1,
                    int(row["seed"]),
                ),
            )
            selected_for_lock[arm] = ordered[0]
        selection_lock = {
            "schema_version": "dimension_conditioning_source_only_selection_lock_v1",
            "locked_at_utc": _now(),
            "selection_source": "SYNTHETIC_DEV_ONLY",
            "real_labels_opened_before_lock": False,
            "selection_rule": (
                "maximum synthetic DEV balanced accuracy; exact tie -> Linear -> lower seed"
            ),
            "selected": {
                arm: {
                    "architecture": row["architecture"],
                    "seed": row["seed"],
                    "synthetic_dev_balanced_accuracy": row["synthetic_dev"][
                        "balanced_accuracy"
                    ],
                    "checkpoint": row["checkpoint"],
                }
                for arm, row in selected_for_lock.items()
            },
        }
        selection_lock_path.write_text(_json_text(selection_lock), "utf-8")

    selected: dict[str, dict[str, Any]] = {}
    for arm, locked in selection_lock["selected"].items():
        matches = [
            row
            for row in run_payloads[arm]
            if row["architecture"] == locked["architecture"]
            and int(row["seed"]) == int(locked["seed"])
            and row["checkpoint"]["sha256"] == locked["checkpoint"]["sha256"]
        ]
        if len(matches) != 1:
            raise PhaseCError(f"SOURCE_SELECTION_LOCK_RUN_NOT_UNIQUE: {arm}")
        selected[arm] = matches[0]

    # TEST remains sealed until the architecture/seed lock is durable.  It is
    # now evaluated for reporting only; no subsequent model choice may change.
    test_all_indices = np.flatnonzero(splits == SPLIT_CODE["TEST"])
    test_valid_local = synth_valid[test_all_indices]
    if not resume_after_source_lock:
        for arm, runs in run_payloads.items():
            matrix = _arm_matrix(synth_features, arm)
            for row in runs:
                checkpoint_path = REPO_ROOT / str(row["checkpoint"]["path"])
                model, _ = _load_probe(checkpoint_path, device)
                probability = np.full(len(test_all_indices), np.nan, dtype=np.float64)
                probability[test_valid_local] = _predict_probabilities(
                    model, matrix[test_all_indices[test_valid_local]], device
                )
                row["synthetic_test"] = {
                    "opened_after_selection_lock": True,
                    "all_frame_with_abstention": _binary_metrics(
                        labels[test_all_indices], probability, test_valid_local
                    ),
                    "detected_only": _binary_metrics(
                        labels[test_all_indices][test_valid_local],
                        probability[test_valid_local],
                    ),
                }
                run_json_path = checkpoint_path.with_suffix(".json")
                persisted = _read_json(run_json_path)
                persisted["synthetic_test"] = row["synthetic_test"]
                persisted["selection_lock"] = {
                    "path": _display(selection_lock_path),
                    "sha256": _sha256(selection_lock_path),
                }
                run_json_path.write_text(_json_text(persisted), "utf-8")
    else:
        for arm, runs in run_payloads.items():
            for row in runs:
                if row.get("synthetic_test", {}).get("opened_after_selection_lock") is not True:
                    raise PhaseCError(f"POST_LOCK_RESUME_SYNTH_TEST_NOT_OPENED: {arm}")

    # Real labels are first opened below this durable lock.
    real_labels = _real_truth(cache)
    real_valid = np.asarray(cache["real_valid"], dtype=np.bool_)
    real_features = _normalized_features(cache, real=True)
    selected_probability: dict[str, np.ndarray] = {}
    real_results: dict[str, Any] = {}
    synth_selected_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for arm, row in selected.items():
        checkpoint_path = REPO_ROOT / str(row["checkpoint"]["path"])
        model, _model_payload = _load_probe(checkpoint_path, device)
        probability = np.full(len(real_labels), np.nan, dtype=np.float64)
        matrix = _arm_matrix(real_features, arm)
        probability[real_valid] = _predict_probabilities(
            model, matrix[real_valid], device
        )
        selected_probability[arm] = probability
        real_results[arm] = {
            "selected_run": selection_lock["selected"][arm],
            "subgroups": _evaluate_real_parity(
                real_labels, probability, real_valid, cache
            ),
        }
        synth_selected_results[arm] = row["synthetic_test"]

    # Counterfactual B4 reuses the exact selected B3 model.  Its real
    # derangement is restricted to the registered model-comparison union
    # COMMON128+Wood45; the twelve FT_EVAL_LEAK-excluded DEV140 rows cannot
    # influence this control.
    real_dimensions = np.asarray(cache["real_dimensions_xyz"], dtype=np.float32)
    control_indices = _common_multishape_indices(cache)
    shuffled_control_values, permutation_local = _shuffled_dimensions(
        real_dimensions[control_indices]
    )
    shuffled_dimensions = real_dimensions.copy()
    shuffled_dimensions[control_indices] = shuffled_control_values
    permutation_global = np.arange(len(real_dimensions), dtype=np.int64)
    permutation_global[control_indices] = control_indices[
        np.asarray(permutation_local, dtype=np.int64)
    ]
    permutation = permutation_global.tolist()
    shuffled_features = _normalized_features(
        cache, real=True, dimensions_override=shuffled_dimensions
    )
    b3_checkpoint = REPO_ROOT / str(selected["B3_IMAGE_KP_DIMS"]["checkpoint"]["path"])
    b3_model, _ = _load_probe(b3_checkpoint, device)

    synth_dimensions = np.asarray(cache["synth_dimensions_xyz"], dtype=np.float32)
    shuffled_synth_dimensions = synth_dimensions.copy()
    shuffled_test_values, synth_test_permutation_local = _shuffled_dimensions(
        synth_dimensions[test_all_indices]
    )
    shuffled_synth_dimensions[test_all_indices] = shuffled_test_values
    shuffled_synth_features = _normalized_features(
        cache, real=False, dimensions_override=shuffled_synth_dimensions
    )
    shuffled_synth_matrix = _arm_matrix(
        shuffled_synth_features, "B3_IMAGE_KP_DIMS"
    )
    shuffled_synth_probability = np.full(
        len(test_all_indices), np.nan, dtype=np.float64
    )
    shuffled_synth_probability[test_valid_local] = _predict_probabilities(
        b3_model,
        shuffled_synth_matrix[test_all_indices[test_valid_local]],
        device,
    )
    synth_selected_results["B4_SHUFFLED_DIMS"] = {
        "opened_after_selection_lock": True,
        "all_frame_with_abstention": _binary_metrics(
            labels[test_all_indices], shuffled_synth_probability, test_valid_local
        ),
        "detected_only": _binary_metrics(
            labels[test_all_indices][test_valid_local],
            shuffled_synth_probability[test_valid_local],
        ),
    }

    shuffled_matrix = _arm_matrix(shuffled_features, "B3_IMAGE_KP_DIMS")
    shuffled_probability = np.full(len(real_labels), np.nan, dtype=np.float64)
    shuffled_probability[real_valid] = _predict_probabilities(
        b3_model, shuffled_matrix[real_valid], device
    )
    selected_probability["B4_SHUFFLED_DIMS"] = shuffled_probability
    control_manifest = _read_json(COMMON_MULTISHAPE_MANIFEST)
    real_results["B4_SHUFFLED_DIMS"] = {
        "trained_model": "B3_IMAGE_KP_DIMS",
        "retrained": False,
        "shuffle_seed": 42,
        "control_population": {
            "population_id": "COMMON_DEV_MULTISHAPE_POS",
            "count": 173,
            "manifest": _display(COMMON_MULTISHAPE_MANIFEST),
            "manifest_sha256": _sha256(COMMON_MULTISHAPE_MANIFEST),
            "membership_sha256": control_manifest.get("membership_sha256"),
            "components": {"COMMON_DEV_PLASTIC_POS128": 128, "DEV_WOOD_POS45": 45},
        },
        "permutation_local": permutation_local,
        "permutation_global_cache_indices": permutation,
        "dev140_excluded_rows_left_unchanged_n": int(
            len(real_dimensions) - len(control_indices)
        ),
        "synthetic_test_permutation_local": synth_test_permutation_local,
        "synthetic_test": synth_selected_results["B4_SHUFFLED_DIMS"],
        "different_source_frame_for_every_control_row": bool(
            all(index != source for index, source in enumerate(permutation_local))
        ),
        "real_changed_dimension_triplet_n": int(
            np.sum(np.any(shuffled_dimensions != real_dimensions, axis=1))
        ),
        "synthetic_test_changed_dimension_triplet_n": int(
            np.sum(
                np.any(
                    shuffled_synth_dimensions[test_all_indices]
                    != synth_dimensions[test_all_indices],
                    axis=1,
                )
            )
        ),
        "plastic_rows_receiving_wood_dimensions": int(
            np.sum(
                (np.asarray(cache["real_object_types"]) == PLASTIC_OBJECT_TYPE)
                & (
                    np.asarray(cache["real_object_types"])[np.asarray(permutation)]
                    == WOOD_OBJECT_TYPE
                )
            )
        ),
        "subgroups": _evaluate_real_parity(
            real_labels, shuffled_probability, real_valid, cache
        ),
    }
    real_object_types = np.asarray(cache["real_object_types"])
    real_frame_ids = np.asarray(cache["real_frame_ids"])
    real_dimension_changed = np.any(shuffled_dimensions != real_dimensions, axis=1)
    plastic_mask = real_object_types == PLASTIC_OBJECT_TYPE
    common_ids = _common_plastic_frame_ids(cache)
    common_mask = np.asarray(
        [str(frame_id) in common_ids for frame_id in real_frame_ids], dtype=np.bool_
    )
    real_results["B4_SHUFFLED_DIMS"][
        "common128_changed_dimension_triplet_n"
    ] = int(np.sum(common_mask & real_dimension_changed))
    b3_probability = selected_probability["B3_IMAGE_KP_DIMS"]
    dimension_change_effect: dict[str, Any] = {}
    for subset_name, subset_mask in (
        ("PLASTIC_CHANGED_TRIPLET", plastic_mask & real_dimension_changed),
        ("PLASTIC_UNCHANGED_TRIPLET", plastic_mask & ~real_dimension_changed),
        ("COMMON128_CHANGED_TRIPLET", common_mask & real_dimension_changed),
        ("COMMON128_UNCHANGED_TRIPLET", common_mask & ~real_dimension_changed),
    ):
        subset_indices = np.flatnonzero(subset_mask)
        b3_subset = _binary_metrics(
            real_labels[subset_indices],
            b3_probability[subset_indices],
            real_valid[subset_indices],
        )
        b4_subset = _binary_metrics(
            real_labels[subset_indices],
            shuffled_probability[subset_indices],
            real_valid[subset_indices],
        )
        dimension_change_effect[subset_name] = {
            "B3_correct_dimensions": b3_subset,
            "B4_shuffled_dimensions": b4_subset,
            "B3_minus_B4_accuracy": float(b3_subset["accuracy"])
            - float(b4_subset["accuracy"]),
        }
    real_results["B4_SHUFFLED_DIMS"][
        "dimension_change_effect"
    ] = dimension_change_effect
    (OUT_DIR / "B4_SHUFFLED_DIMS").mkdir(parents=True, exist_ok=True)

    wrong_dimensions = _wrong_type_dimensions(cache, real_dimensions)
    wrong_features = _normalized_features(
        cache, real=True, dimensions_override=wrong_dimensions
    )
    wrong_matrix = _arm_matrix(wrong_features, "B3_IMAGE_KP_DIMS")
    wrong_probability = np.full(len(real_labels), np.nan, dtype=np.float64)
    wrong_probability[real_valid] = _predict_probabilities(
        b3_model, wrong_matrix[real_valid], device
    )
    selected_probability["B5_WRONG_TYPE"] = wrong_probability
    real_results["B5_WRONG_TYPE"] = {
        "trained_model": "B3_IMAGE_KP_DIMS",
        "retrained": False,
        "control_scope": (
            "wrong dimensions enter parity head only; pose evaluation retains the true "
            "registry dimensions to avoid an invalid wrong-scale pose claim"
        ),
        "subgroups": _evaluate_real_parity(
            real_labels, wrong_probability, real_valid, cache
        ),
    }
    (OUT_DIR / "B5_WRONG_TYPE").mkdir(parents=True, exist_ok=True)

    selected_by_arm = {
        arm: np.where(
            np.isfinite(probability), (probability >= 0.5).astype(np.int8), -1
        )
        for arm, probability in selected_probability.items()
    }
    pose_metrics, pose_bootstrap = _pose_diagnostic(cache, selected_by_arm)
    phase_a = _read_json(PHASE_A_DIR / "DEV_DIMENSION_ORACLE_DIAGNOSTIC.json")
    phase_a_common = _phase_a_common_metrics(cache)
    a0 = phase_a_common["metrics"]["A0_CURRENT_SELECTOR"]
    a1 = phase_a_common["metrics"]["A1_GT_PARITY_ORACLE"]
    denominator = float(a1["restricted_adds_auc"]) - float(
        a0["restricted_adds_auc"]
    )
    recovery: dict[str, Any] = {
        "definition": "(AUC_candidate - AUC_A0) / (AUC_A1 - AUC_A0)",
        "population": phase_a_common,
        "A0_restricted_adds_auc": a0["restricted_adds_auc"],
        "A1_oracle_restricted_adds_auc": a1["restricted_adds_auc"],
        "denominator": denominator,
        "dev140_selector_diagnostic_reference": {
            "A0_restricted_adds_auc": phase_a["metrics"]["A0_CURRENT_SELECTOR"][
                "ALL"
            ]["restricted_adds_auc"],
            "A1_oracle_restricted_adds_auc": phase_a["metrics"][
                "A1_GT_PARITY_ORACLE"
            ]["ALL"]["restricted_adds_auc"],
        },
        "arms": {},
    }
    for arm in ("B1_KP_ONLY", "B2_IMAGE_KP", "B3_IMAGE_KP_DIMS"):
        auc = float(pose_metrics[arm]["COMMON128"]["restricted_adds_auc"])
        recovery["arms"][arm] = {
            "restricted_adds_auc": auc,
            "oracle_recovery_ratio": (
                (auc - float(a0["restricted_adds_auc"])) / denominator
                if denominator != 0.0
                else None
            ),
        }

    # Populate all-run comparison rows.  Real values are reported for every
    # source-trained seed/architecture (robustness), but never feed selection.
    for arm, runs in run_payloads.items():
        real_matrix = _arm_matrix(real_features, arm)
        for row in runs:
            checkpoint_path = REPO_ROOT / str(row["checkpoint"]["path"])
            model, _ = _load_probe(checkpoint_path, device)
            probability = np.full(len(real_labels), np.nan, dtype=np.float64)
            probability[real_valid] = _predict_probabilities(
                model, real_matrix[real_valid], device
            )
            subgroups = _evaluate_real_parity(
                real_labels, probability, real_valid, cache
            )
            comparison_rows.append(
                {
                    "arm": arm,
                    "architecture": row["architecture"],
                    "seed": row["seed"],
                    "selected_source_only": bool(row is selected[arm]),
                    "synth_test_accuracy": row["synthetic_test"][
                        "all_frame_with_abstention"
                    ]["accuracy"],
                    "synth_test_balanced_accuracy": row["synthetic_test"][
                        "all_frame_with_abstention"
                    ]["balanced_accuracy"],
                    "synth_test_auroc": row["synthetic_test"][
                        "all_frame_with_abstention"
                    ]["auroc"],
                    "plastic_model_comparison_population": (
                        "COMMON_DEV_PLASTIC_POS128"
                    ),
                    "plastic_common128_accuracy": subgroups[
                        "PLASTIC:COMMON128:ALL"
                    ]["accuracy"],
                    "plastic_dev140_diagnostic_accuracy": subgroups["PLASTIC:ALL"][
                        "accuracy"
                    ],
                    "plastic_night_accuracy": subgroups[
                        "PLASTIC:COMMON128:NIGHT"
                    ]["accuracy"],
                    "wood_all_accuracy": subgroups["WOOD:ALL"]["accuracy"],
                }
            )

    b3_groups = real_results["B3_IMAGE_KP_DIMS"]["subgroups"]
    b2_groups = real_results["B2_IMAGE_KP"]["subgroups"]
    b4_groups = real_results["B4_SHUFFLED_DIMS"]["subgroups"]
    minimum_session = min(
        float(value["accuracy"])
        for name, value in b3_groups.items()
        if name.startswith("PLASTIC:COMMON128:SESSION:")
    )
    tap_audit = _read_json(OUT_DIR / "FEATURE_TAP_AUDIT.json")
    # The scope disclosure is reporting metadata, not part of the numerical
    # feature recipe.  Keep the extraction audit synchronized without touching
    # FEATURE_CACHE.npz or rerunning YOLO.
    if tap_audit.get("prerequisites", {}).get("scope_lock") != scope_lock:
        tap_audit.setdefault("prerequisites", {})["scope_lock"] = scope_lock
        tap_audit["scope_annotation_refresh"] = {
            "feature_cache_rewritten": False,
            "yolo_inference_rerun": False,
            "reason": "clarify probe-head split versus upstream frozen-YOLO exposure",
        }
        (OUT_DIR / "FEATURE_TAP_AUDIT.json").write_text(
            _json_text(tap_audit), "utf-8"
        )
    p1 = float(b3_groups["PLASTIC:COMMON128:ALL"]["accuracy"]) >= 0.95
    p2 = float(b3_groups["PLASTIC:COMMON128:NIGHT"]["accuracy"]) >= 0.90
    p3 = minimum_session >= 0.85
    p4_correct_delta_n = int(
        pose_metrics["B3_IMAGE_KP_DIMS"]["COMMON128"]["parity_correct_n"]
        - pose_metrics["B2_IMAGE_KP"]["COMMON128"]["parity_correct_n"]
    )
    p5_correct_delta_n = int(
        pose_metrics["B3_IMAGE_KP_DIMS"]["COMMON128"]["parity_correct_n"]
        - pose_metrics["B4_SHUFFLED_DIMS"]["COMMON128"]["parity_correct_n"]
    )
    # Express these thresholded differences from exact frame counts.  Direct
    # subtraction of JSON accuracies can place an exact rational threshold a
    # few ULPs below 0.05.  Threshold using exact COMMON128 frame counts.
    model_comparison_n = 128
    p4_delta = p4_correct_delta_n / model_comparison_n
    p5_delta = p5_correct_delta_n / model_comparison_n
    p4_pass = p4_correct_delta_n >= math.ceil(0.05 * model_comparison_n)
    p5_pass = p5_correct_delta_n >= math.ceil(0.05 * model_comparison_n)
    p6_value = recovery["arms"]["B3_IMAGE_KP_DIMS"]["oracle_recovery_ratio"]
    p7_delta = float(
        pose_metrics["B3_IMAGE_KP_DIMS"]["COMMON128"]["pose_valid_rate"]
    ) - float(a0["pose_valid_rate"])
    p8 = bool(
        tap_audit["instrumentation_prediction_parity"]["passed"]
        and tap_audit["candidate_provenance"]["passed"]
    )
    simpler_arm_gates: dict[str, Any] = {}
    for candidate_arm in ("B1_KP_ONLY", "B2_IMAGE_KP"):
        candidate_groups = real_results[candidate_arm]["subgroups"]
        candidate_min_session = min(
            float(value["accuracy"])
            for name, value in candidate_groups.items()
            if name.startswith("PLASTIC:COMMON128:SESSION:")
        )
        candidate_recovery = recovery["arms"][candidate_arm][
            "oracle_recovery_ratio"
        ]
        candidate_pose_valid_delta = float(
            pose_metrics[candidate_arm]["COMMON128"]["pose_valid_rate"]
        ) - float(a0["pose_valid_rate"])
        conditions = {
            "P1_overall": float(
                candidate_groups["PLASTIC:COMMON128:ALL"]["accuracy"]
            ) >= 0.95,
            "P2_night": float(
                candidate_groups["PLASTIC:COMMON128:NIGHT"]["accuracy"]
            ) >= 0.90,
            "P3_min_session": candidate_min_session >= 0.85,
            "P6_oracle_recovery": bool(
                candidate_recovery is not None and candidate_recovery >= 0.70
            ),
            "P7_pose_valid_no_harm": candidate_pose_valid_delta >= 0.0,
            "P8_prediction_exact": p8,
        }
        simpler_arm_gates[candidate_arm] = {
            "conditions": conditions,
            "values": {
                "population_id": "COMMON_DEV_PLASTIC_POS128",
                "overall_accuracy": candidate_groups[
                    "PLASTIC:COMMON128:ALL"
                ]["accuracy"],
                "night_accuracy": candidate_groups[
                    "PLASTIC:COMMON128:NIGHT"
                ]["accuracy"],
                "minimum_session_accuracy": candidate_min_session,
                "oracle_recovery_ratio": candidate_recovery,
                "pose_valid_rate_delta": candidate_pose_valid_delta,
            },
            "P9": {
                "status": "DEFERRED_UNLESS_SOURCE_PERFORMANCE_GATE_PASSES",
                "passed": False,
            },
            "source_performance_passed": all(conditions.values()),
            "passed": False,
        }
    pre_latency_pass = bool(
        p1 and p2 and p3 and p4_pass and p5_pass
        and p6_value is not None and float(p6_value) >= 0.70 and p7_delta >= 0.0 and p8
    )
    gate = {
        "population_id": "COMMON_DEV_PLASTIC_POS128",
        "population_n": model_comparison_n,
        "P1": {"passed": p1, "value": b3_groups["PLASTIC:COMMON128:ALL"]["accuracy"], "minimum": 0.95},
        "P2": {"passed": p2, "value": b3_groups["PLASTIC:COMMON128:NIGHT"]["accuracy"], "minimum": 0.90},
        "P3": {"passed": p3, "value": minimum_session, "minimum": 0.85},
        "P4": {
            "passed": p4_pass,
            "B3_minus_B2_absolute": p4_delta,
            "delta_percentage_points": 100.0 * p4_delta,
            "correct_frame_delta": {"numerator": p4_correct_delta_n, "denominator": model_comparison_n},
            "minimum": 0.05,
        },
        "P5": {
            "passed": p5_pass,
            "B3_minus_B4_absolute": p5_delta,
            "delta_percentage_points": 100.0 * p5_delta,
            "correct_frame_delta": {"numerator": p5_correct_delta_n, "denominator": model_comparison_n},
            "minimum": 0.05,
        },
        "P6": {"passed": bool(p6_value is not None and p6_value >= 0.70), "value": p6_value, "minimum": 0.70},
        "P7": {"passed": p7_delta >= 0.0, "pose_valid_rate_delta": p7_delta, "minimum": 0.0},
        "P8": {
            "passed": p8,
            "prediction_max_abs_diff": tap_audit[
                "instrumentation_prediction_parity"
            ]["max_abs_diff"],
            "mapped_logit_confidence_max_abs_diff": tap_audit[
                "candidate_provenance"
            ]["max_abs_diff"],
        },
        "P9": {
            "passed": False,
            "status": "DEFERRED_UNLESS_P1_TO_P8_PASS",
            "reason": None if pre_latency_pass else "NO_DEPLOYMENT_CANDIDATE_REACHED_LATENCY_GATE",
            "maximum_relative_increase": 0.05,
        },
        "P1_to_P8_passed": pre_latency_pass,
        "passed": False,
    }
    pre_latency_candidates = [
        arm
        for arm in ("B1_KP_ONLY", "B2_IMAGE_KP")
        if simpler_arm_gates[arm]["source_performance_passed"]
    ]
    if pre_latency_pass:
        pre_latency_candidates.append("B3_IMAGE_KP_DIMS")
    pre_latency_winner = pre_latency_candidates[0] if pre_latency_candidates else None

    real_payload = {
        "schema_version": "dimension_conditioning_real_parity_results_v1",
        "generated_at_utc": _now(),
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "paper_final_table_eligible": False,
        "selection_lock": _display(OUT_DIR / "SELECTION_LOCK.json"),
        "real_used_for_training": 0,
        "real_used_for_hyperparameter_or_checkpoint_selection": 0,
        "scope_lock": scope_lock,
        "population_views": {
            "PLASTIC:ALL": "DEV_PLASTIC_POS140_SELECTOR_DIAGNOSTIC",
            "PLASTIC:COMMON128:ALL": "COMMON_DEV_PLASTIC_POS128_MODEL_COMPARISON_AND_GATE",
            "WOOD:ALL": "DEV_WOOD_POS45_OUT_OF_RANGE_DIAGNOSTIC_ONLY",
        },
        "arms": real_results,
    }
    pose_payload = {
        "schema_version": "dimension_conditioning_real_pose_diagnostic_v1",
        "generated_at_utc": _now(),
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "paper_final_table_eligible": False,
        "scope_lock": scope_lock,
        "plastic_metrics": pose_metrics,
        "plastic_session_cluster_bootstrap_1000": pose_bootstrap,
        "wood_pose_status": "BLOCKED",
        "wood_pose_blocked_reasons": [
            "WOOD_SYMMETRY_UNREVIEWED",
            "WOOD_INTRINSICS_SENSOR_PROFILE_SCALED_NOT_APPROVED",
        ],
        "wrong_type_control_scope": real_results["B5_WRONG_TYPE"]["control_scope"],
    }
    recovery["phase_c_gate"] = gate
    recovery["simpler_arm_gates"] = simpler_arm_gates
    recovery["pre_latency_winner_by_priority"] = pre_latency_winner
    recovery["scope_lock"] = scope_lock
    (OUT_DIR / "REAL_PARITY_RESULTS.json").write_text(
        _json_text(real_payload), "utf-8"
    )
    (OUT_DIR / "REAL_POSE_DIAGNOSTIC.json").write_text(
        _json_text(pose_payload), "utf-8"
    )
    (OUT_DIR / "ORACLE_RECOVERY.json").write_text(_json_text(recovery), "utf-8")
    (OUT_DIR / "PROBE_COMPARISON.csv").write_text(
        _comparison_csv(comparison_rows), "utf-8"
    )
    (OUT_DIR / "B4_SHUFFLED_DIMS" / "RESULTS.json").write_text(
        _json_text(real_results["B4_SHUFFLED_DIMS"]), "utf-8"
    )
    (OUT_DIR / "B5_WRONG_TYPE" / "RESULTS.json").write_text(
        _json_text(real_results["B5_WRONG_TYPE"]), "utf-8"
    )

    if _sha256(WEIGHTS) != CHECKPOINT_SHA256 or WEIGHTS.stat().st_mtime_ns != checkpoint_mtime_before:
        raise PhaseCError("YOLO_CHECKPOINT_CHANGED_DURING_PROBE_TRAINING")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "checkpoint_sha256_before_after": [CHECKPOINT_SHA256, _sha256(WEIGHTS)],
        "yolo_parameter_updates": 0,
        "probe_training_runs": len(ARCHITECTURES) * len(SEEDS) * len(ARM_FEATURES),
        "full_yolo_training_runs": 0,
        "scope_lock": scope_lock,
        "selection": selection_lock,
        "synthetic_selected_test": synth_selected_results,
        "phase_c_gate": gate,
        "simpler_arm_gates": simpler_arm_gates,
        "pre_latency_winner_by_priority": pre_latency_winner,
        "deployment_wrapper_authorized": False,
        "cause": (
            (
                "LEARNED_2D_PARITY_SELECTOR_SUFFICIENT_PENDING_LATENCY"
                if pre_latency_winner == "B1_KP_ONLY"
                else (
                    "IMAGE_FEATURE_PARITY_HEAD_SUFFICIENT_PENDING_LATENCY"
                    if pre_latency_winner == "B2_IMAGE_KP"
                    else "DIMENSION_CONDITIONING_HAS_INCREMENTAL_VALUE_PENDING_LATENCY"
                )
            )
            if pre_latency_winner is not None
            else "PARITY_INFORMATION_NOT_RECOVERED_FROM_CURRENT_FEATURES"
        ),
    }
    (OUT_DIR / "PHASE_C_SUMMARY.json").write_text(_json_text(summary), "utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("extract", "train", "all"))
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"extract", "all"}:
        extract_features(str(args.device))
    if args.command in {"train", "all"}:
        train_and_evaluate(str(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
