"""Train the Phase-E S0/S1 probes from the locked corrected feature cache.

The extraction runner and its contract remain immutable.  This recovery runner
only changes the DEV abstention policy: TRAIN detections must all be valid,
while a DEV abstention is retained in the population and counted as an error by
the existing Phase-C metric.  It never updates YOLO or re-extracts features.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE_RUN = HERE / "runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42"
BASE_LAST = BASE_RUN / "weights/last.pt"
BASE_BEST = BASE_RUN / "weights/best.pt"
BASE_ARGS = BASE_RUN / "args.yaml"
BASE_RESULTS = BASE_RUN / "results.csv"
BASE_BINDING = HERE / "BASE_RUN_BINDING.json"
BASE_INTERRUPTION_EVENT = HERE / "BASE_INTERRUPTION_EVENT.json"
DATA_YAML = (
    REPO
    / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml"
)
INITIAL_WEIGHTS = REPO / "challenge/weights/pretrained_yolo/yolo26n-pose.pt"
METADATA = HERE / "PROBE_METADATA_60K.jsonl"
METADATA_AUDIT = HERE / "PROBE_METADATA_60K_AUDIT.json"
TRAINING_CONTRACT = HERE / "TRAINING_CONTRACT.json"
SPATIAL_CONTRACT = HERE / "SPATIAL_STAGE_CONTRACT.json"
BASE_LOCK = HERE / "BASE_CHECKPOINT_LOCK.json"
SHARD_DIR = HERE / "spatial_feature_shards_60k"
FEATURE_CACHE = HERE / "SPATIAL_FEATURE_CACHE_60K.npz"
FEATURE_AUDIT = HERE / "SPATIAL_FEATURE_CACHE_60K_AUDIT.json"
PROBE_DIR = HERE / "probe_runs"
SUMMARY = HERE / "SPATIAL_CONCAT_60K_SUMMARY.json"
REPORT = HERE / "SPATIAL_CONCAT_60K_REPORT.md"
DEV_PREDICTIONS = HERE / "SPATIAL_CONCAT_60K_DEV_PREDICTIONS.npz"

PHASE_C_RUNNER = (
    REPO
    / "challenge/yolo_pose_one_model/dimension_conditioning_probe/"
    "phase_c_probe/run_phase_c.py"
)
PHASE_E_RUNNER = (
    REPO
    / "challenge/yolo_pose_one_model/dimension_conditioning_probe/"
    "phase_e_spatial_fusion/run_spatial_fusion.py"
)
EXTRACTION_RUNNER = HERE / "run_spatial_concat.py"
PROBE_TRAINING_RECOVERY_CONTRACT = HERE / "PROBE_TRAINING_RECOVERY_CONTRACT.json"

SCHEMA = "spatial_concat_clean_start_60k_v1"
ARMS = ("S0_SPATIAL_NO_DIMS", "S1_SPATIAL_CONCAT")
SEEDS = (0, 1, 2)
PATCH_SIZE = 7
PATCH_TOKENS = 49
TOKEN_DIM = 64
KP_DIM = 26
DIM_DIM = 4
EXTRACT_BATCH = 32
SHARD_SIZE = 512
TRAIN_BATCH = 512
TRAIN_STEPS = 40
TRAIN_EPOCHS = 40
EVAL_BATCH = 1024
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
DATA_YAML_SHA256 = "19e0501c835df11e7d2356864bf7cd38da367618fd07e3822fad9ccc92417ec5"
INITIAL_WEIGHTS_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
BASE_CONTRACT_SHA256 = "eeee76970e75de672fdcc3e0ee38697a3d4afc7ab040b47d6de87dbe32715124"


class RunnerError(RuntimeError):
    pass


def normalize_device(value: str) -> str:
    """Use one device spelling accepted by both Ultralytics and torch.Tensor.to."""
    text = str(value).strip().lower()
    if text.isdigit():
        text = f"cuda:{text}"
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RunnerError(f"CUDA device requested but unavailable: {text}")
    try:
        torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise RunnerError(f"invalid torch device: {value}") from exc
    return text


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def publish_temporary_exclusive(temporary: Path, destination: Path) -> None:
    """Atomically publish a complete file without replacing an existing artifact."""
    os.link(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise RunnerError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_json_exclusive(path: Path, payload: Any) -> None:
    value = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text_exclusive(path, value)


def save_npz_exclusive(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as f:
            writer = np.savez_compressed if compressed else np.savez
            writer(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise RunnerError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def save_torch_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise RunnerError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_phase_c_module: Any | None = None
_phase_e_module: Any | None = None


def phase_c() -> Any:
    global _phase_c_module
    if _phase_c_module is None:
        _phase_c_module = load_module("cleanstart_phase_c", PHASE_C_RUNNER)
    return _phase_c_module


def phase_e() -> Any:
    global _phase_e_module
    if _phase_e_module is None:
        _phase_e_module = load_module("cleanstart_phase_e", PHASE_E_RUNNER)
    return _phase_e_module


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise RunnerError(f"invalid YAML: {path}")
    return value


def validate_spatial_contract() -> dict[str, Any]:
    payload = json.loads(SPATIAL_CONTRACT.read_text(encoding="utf-8"))
    expected_bindings = {
        "base_training_contract_sha256": sha256(TRAINING_CONTRACT),
        "base_interruption_event_sha256": sha256(BASE_INTERRUPTION_EVENT),
        "probe_metadata_sha256": sha256(METADATA),
        "probe_metadata_audit_sha256": sha256(METADATA_AUDIT),
        "phase_c_runner_sha256": sha256(PHASE_C_RUNNER),
        "phase_e_runner_sha256": sha256(PHASE_E_RUNNER),
        "runner_sha256": sha256(EXTRACTION_RUNNER),
    }
    expected_training = {
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "epochs": TRAIN_EPOCHS,
        "steps_per_epoch": TRAIN_STEPS,
        "batch": TRAIN_BATCH,
        "samples_per_epoch_with_replacement": TRAIN_STEPS * TRAIN_BATCH,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "class_balanced_sampler": True,
        "loss": "balanced_cross_entropy",
    }
    if (
        payload.get("schema_version") != "spatial_concat_stage_contract_v1"
        or payload.get("source_bindings") != expected_bindings
        or payload.get("architecture")
        != "exact post-hoc Phase-E S0/S1 7x7 selected-source spatial concat"
        or payload.get("dimension_input")
        != "fixed_renderer_dimensions_m_xyz_model_input"
        or payload.get("camera_facing_dimensions_model_input_forbidden") is not True
        or payload.get("feature_extraction", {}).get("already_padded_input") is not True
        or payload.get("feature_extraction", {}).get("two_pass_exactness_required")
        is not True
        or payload.get("feature_extraction", {}).get("batch") != EXTRACT_BATCH
        or payload.get("feature_extraction", {}).get("shard_size") != SHARD_SIZE
        or payload.get("probe_training") != expected_training
        or payload.get("real_used_for_training_or_selection") != 0
    ):
        raise RunnerError("spatial stage contract mismatch")
    return payload


def validate_probe_training_recovery_contract() -> dict[str, Any]:
    if not PROBE_TRAINING_RECOVERY_CONTRACT.is_file():
        raise RunnerError(
            f"probe training recovery contract missing: "
            f"{PROBE_TRAINING_RECOVERY_CONTRACT}"
        )
    payload = json.loads(
        PROBE_TRAINING_RECOVERY_CONTRACT.read_text(encoding="utf-8")
    )
    incident = HERE / "SPATIAL_EXTRACTION_FAILURE_20260830.json"
    expected_bindings = {
        "base_checkpoint_lock_sha256": sha256(BASE_LOCK),
        "probe_metadata_sha256": sha256(METADATA),
        "probe_metadata_audit_sha256": sha256(METADATA_AUDIT),
        "feature_cache_sha256": sha256(FEATURE_CACHE),
        "feature_cache_audit_sha256": sha256(FEATURE_AUDIT),
        "spatial_extraction_contract_sha256": sha256(SPATIAL_CONTRACT),
        "spatial_extraction_runner_sha256": sha256(EXTRACTION_RUNNER),
        "probe_training_runner_sha256": sha256(Path(__file__)),
        "phase_e_runner_sha256": sha256(PHASE_E_RUNNER),
        "failure_incident_sha256": sha256(incident),
    }
    expected_policy = {
        "train_invalid_detection_rows": "hard_fail",
        "dev_invalid_detection_rows": "retain_as_abstention",
        "dev_abstention_metric_treatment": (
            "prediction=-1; confusion abstain column; accuracy and per-class "
            "recall denominators include the abstention, so it counts as an error"
        ),
        "expected_dev_abstention_count": 1,
        "expected_dev_abstention_global_indices": [59507],
        "expected_dev_abstention_frame_ids": ["TEX__shard_04_f0110"],
        "checkpoint_selection_metric": (
            "DEV balanced accuracy including abstentions as recall errors"
        ),
    }
    if (
        payload.get("schema_version")
        != "spatial_probe_training_recovery_contract_v1"
        or payload.get("source_bindings") != expected_bindings
        or payload.get("policy") != expected_policy
        or payload.get("real_used_for_training_or_selection") != 0
        or payload.get("yolo_parameter_updates") != 0
    ):
        raise RunnerError("probe training recovery contract mismatch")
    return payload


def validate_base() -> dict[str, Any]:
    required = (
        BASE_LAST,
        BASE_BEST,
        BASE_ARGS,
        BASE_RESULTS,
        BASE_BINDING,
        BASE_INTERRUPTION_EVENT,
        DATA_YAML,
        INITIAL_WEIGHTS,
        TRAINING_CONTRACT,
        SPATIAL_CONTRACT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RunnerError(f"base run is incomplete: {missing}")
    validate_spatial_contract()
    args = load_yaml(BASE_ARGS)
    expected_args = {
        "task": "pose",
        "mode": "train",
        "epochs": 60,
        "batch": 32,
        "imgsz": 640,
        "workers": 2,
        "optimizer": "SGD",
        "seed": 42,
        "deterministic": True,
        "patience": 0,
        "single_cls": True,
        "rect": False,
        "cos_lr": True,
        "close_mosaic": 10,
        "amp": True,
        "pretrained": True,
        "resume": False,
        "compile": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.35,
        "scale": 0.25,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.3,
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise RunnerError(f"base arg mismatch {key}: {args.get(key)} != {expected}")
    if Path(str(args.get("model"))).resolve() != INITIAL_WEIGHTS.resolve():
        raise RunnerError("base initialization path mismatch")
    if Path(str(args.get("data"))).resolve() != DATA_YAML.resolve():
        raise RunnerError("base data path mismatch")
    if sha256(INITIAL_WEIGHTS) != INITIAL_WEIGHTS_SHA256:
        raise RunnerError("official generic initialization hash mismatch")
    if sha256(DATA_YAML) != DATA_YAML_SHA256:
        raise RunnerError("mixed dataset YAML hash mismatch")
    if sha256(TRAINING_CONTRACT) != BASE_CONTRACT_SHA256:
        raise RunnerError("base training contract changed after launch")
    binding = json.loads(BASE_BINDING.read_text(encoding="utf-8"))
    interruption_event = json.loads(
        BASE_INTERRUPTION_EVENT.read_text(encoding="utf-8")
    )
    expected_binding_sources = {
        str(DATA_YAML): DATA_YAML_SHA256,
        str(INITIAL_WEIGHTS): INITIAL_WEIGHTS_SHA256,
        str(TRAINING_CONTRACT): BASE_CONTRACT_SHA256,
    }
    if (
        binding.get("schema_version") != "mixed_base_runtime_binding_v1"
        or binding.get("dataset_counts") != {"train": 55980, "val": 4020}
        or binding.get("source_sha256") != expected_binding_sources
        or binding.get("prior_pallet_checkpoint_loaded") is not False
        or binding.get("random_initialization") is not False
        or binding.get("official_generic_pretrained_initialization") is not True
    ):
        raise RunnerError("base runtime binding mismatch")
    if (
        interruption_event.get("schema_version")
        != "mixed_base_interruption_event_v1"
        or interruption_event.get("interrupted_execution", {}).get(
            "last_complete_epoch"
        )
        != 1
        or interruption_event.get("temporary_resume", {}).get("checkpoint_sha256")
        != "70c3a11c4a34e458895cfde2d1b4243f364543ecabe68ee3b22e9dc554f34221"
        or interruption_event.get("final_recovery", {}).get(
            "prior_pallet_checkpoint_loaded"
        )
        is not False
    ):
        raise RunnerError("base interruption-event provenance mismatch")
    with BASE_RESULTS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(float(row["epoch"])) for row in rows]
    if len(rows) != 60 or epochs != list(range(1, 61)):
        raise RunnerError(f"base results are not a complete 60 epochs: {len(rows)}")
    payload = {
        "schema_version": "cleanstart_base_checkpoint_lock_v1",
        "created_at_utc": now(),
        "checkpoint_rule": "fixed-budget epoch-60 last.pt",
        "last_pt": {"path": str(BASE_LAST.relative_to(REPO)), "sha256": sha256(BASE_LAST)},
        "best_pt_comparison_only": {
            "path": str(BASE_BEST.relative_to(REPO)),
            "sha256": sha256(BASE_BEST),
        },
        "args_yaml_sha256": sha256(BASE_ARGS),
        "results_csv_sha256": sha256(BASE_RESULTS),
        "base_runtime_binding_sha256": sha256(BASE_BINDING),
        "base_interruption_event_sha256": sha256(BASE_INTERRUPTION_EVENT),
        "training_contract_sha256": sha256(TRAINING_CONTRACT),
        "spatial_stage_contract_sha256": sha256(SPATIAL_CONTRACT),
        "epochs_observed": len(rows),
        "yolo_parameter_updates_after_lock": 0,
    }
    if BASE_LOCK.exists():
        existing = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_at_utc", None)
        expected = dict(payload)
        expected.pop("created_at_utc", None)
        if comparable != expected:
            raise RunnerError("existing base checkpoint lock mismatch")
        return existing
    write_json_exclusive(BASE_LOCK, payload)
    return payload


def load_metadata() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if not METADATA.is_file() or not METADATA_AUDIT.is_file():
        raise RunnerError("run build_probe_metadata.py first")
    audit = json.loads(METADATA_AUDIT.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != "spatial_concat_probe_metadata_60k_v1"
        or audit.get("manifest", {}).get("rows") != 60000
        or audit.get("manifest", {}).get("sha256") != sha256(METADATA)
        or audit.get("allowed_dimension_input")
        != "fixed_renderer_dimensions_m_xyz_model_input"
        or audit.get("camera_facing_whd_parity_agreement", {}).get(
            "model_input_forbidden"
        )
        is not True
    ):
        raise RunnerError("60k metadata hash mismatch")
    rows = [json.loads(line) for line in METADATA.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 60000:
        raise RunnerError(f"metadata rows {len(rows)} != 60000")
    for row in rows:
        if (
            row.get("camera_facing_dimensions_are_forbidden_as_model_input") is not True
            or row.get("split") not in {"train", "val"}
            or row.get("dataset_split") != row.get("split")
            or row.get("source") not in {"G38", "P0", "TEX"}
            or row.get("front_face_class_short0_long1") not in {0, 1}
        ):
            raise RunnerError("camera-facing dimension input prohibition missing")
    arrays = {
        "frame_ids": np.asarray([r["merged_stem"] for r in rows]),
        "images": np.asarray([r["image_path"] for r in rows]),
        "sources": np.asarray([r["source"] for r in rows]),
        "splits": np.asarray([0 if r["split"] == "train" else 1 for r in rows], dtype=np.int8),
        "dimensions_xyz": np.asarray(
            [r["fixed_renderer_dimensions_m_xyz_model_input"] for r in rows],
            dtype=np.float32,
        ),
        "labels": np.asarray(
            [r["front_face_class_short0_long1"] for r in rows], dtype=np.int8
        ),
        "raw_shapes": np.asarray(
            [r["raw_shape_height_width"] for r in rows], dtype=np.int16
        ),
    }
    if len(set(arrays["frame_ids"].tolist())) != len(rows):
        raise RunnerError("metadata frame IDs not unique")
    if not np.isfinite(arrays["dimensions_xyz"]).all() or np.any(
        arrays["dimensions_xyz"] <= 0
    ):
        raise RunnerError("fixed dimension array invalid")
    if Counter(arrays["splits"].tolist()) != Counter({0: 55980, 1: 4020}):
        raise RunnerError("metadata split counts invalid")
    if Counter(arrays["sources"].tolist()) != Counter(
        {"G38": 40000, "P0": 10000, "TEX": 10000}
    ):
        raise RunnerError("metadata source counts invalid")
    if set(arrays["labels"].tolist()) != {0, 1} or np.any(arrays["raw_shapes"] <= 0):
        raise RunnerError("metadata labels/raw shapes invalid")
    return rows, arrays


def feature_recipe(base_sha: str, metadata_sha: str) -> dict[str, Any]:
    payload = {
        "schema_version": "cleanstart_s1_feature_recipe_v1",
        "base_checkpoint_sha256": base_sha,
        "metadata_sha256": metadata_sha,
        "confidence_floor": 0.001,
        "imgsz": 640,
        "input": "decode already-reflect-padded merged image; do not pad again",
        "inference_passes": (
            "two identical shape-homogeneous passes per batch; second-pass prediction "
            "and source feature must exactly reproduce the first pass before patch storage"
        ),
        "inference_options": {
            "batch": EXTRACT_BATCH,
            "half": False,
            "augment": False,
            "compile": False,
            "rect": True,
        },
        "shape_grouping": "raw height/width; stable metadata-index order",
        "shard_size": SHARD_SIZE,
        "input_content_binding": (
            "SHA-256 of every prepared image byte stream in each shard fingerprint; "
            "all hashes recomputed before final assembly"
        ),
        "raw_geometry": "subtract PAD=100 and normalize area by unpadded raw shape",
        "head_discovery": "unique module exposing one2one_cv3/one2one_cv4/get_topk_index/nl/stride",
        "candidate": "highest-confidence final detection",
        "feature": "immediate producer of active one2one classification readout",
        "patch": "7x7x64 zero-padded with boolean validity mask",
        "source_cell": (
            "captured postprocess get_topk_index top,flat,level,row,column; exact "
            "between the two inference passes"
        ),
        "kp_geometry": "Phase-C 26-D predicted bbox/keypoint geometry",
        "dimension_feature": "fixed-axis xyz scale-invariant 4-D logs; TRAIN normalization",
        "phase_c_runner_sha256": sha256(PHASE_C_RUNNER),
        "phase_e_runner_sha256": sha256(PHASE_E_RUNNER),
        "spatial_stage_contract_sha256": sha256(SPATIAL_CONTRACT),
        "runner_sha256": sha256(EXTRACTION_RUNNER),
    }
    return {**payload, "sha256": canonical_hash(payload)}


def make_tap(weights: Path, checkpoint_sha: str, device: str) -> Any:
    """Instantiate the exact Phase-C tap against the newly bound checkpoint."""
    base = phase_c()

    class BoundDynamicOne2OneTap(base.DynamicOne2OneTap):
        """Phase-C tap with the inference defaults made explicit and audited."""

        def predict_batch(
            self, images: Sequence[np.ndarray]
        ) -> tuple[list[Any], list[dict[str, Any] | None]]:
            self.cap.clear()
            self.logit_cap.clear()
            self.flat_indices = None
            results = self.yolo.predict(
                list(images),
                batch=len(images),
                conf=0.001,
                imgsz=640,
                device=self.device,
                half=False,
                augment=False,
                compile=False,
                rect=True,
                verbose=False,
            )
            if self.flat_indices is None or int(self.flat_indices.shape[0]) != len(
                results
            ):
                raise RunnerError("top-k provenance capture missing")
            if set(self.cap) != set(range(self.level_count)) or set(
                self.logit_cap
            ) != set(range(self.level_count)):
                raise RunnerError("one2one feature/readout hook did not fire for every level")
            for level in range(self.level_count):
                if int(self.cap[level].shape[1]) != TOKEN_DIM:
                    raise RunnerError(
                        f"one2one token channel mismatch at level {level}: "
                        f"{tuple(self.cap[level].shape)}"
                    )
                if int(self.logit_cap[level].shape[1]) != 1:
                    raise RunnerError(
                        f"one2one class count mismatch at level {level}: "
                        f"{tuple(self.logit_cap[level].shape)}"
                    )
            vectors: list[dict[str, Any] | None] = []
            for batch_index, result in enumerate(results):
                if result.boxes is None or len(result.boxes) == 0:
                    vectors.append(None)
                    continue
                top = int(result.boxes.conf.argmax().item())
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

    if sha256(weights) != checkpoint_sha:
        raise RunnerError("base checkpoint changed before tap construction")
    historical = base.CHECKPOINT_SHA256
    try:
        base.CHECKPOINT_SHA256 = checkpoint_sha
        tap = BoundDynamicOne2OneTap(weights, device)
    finally:
        base.CHECKPOINT_SHA256 = historical
    if int(tap.level_count) != 3 or int(getattr(tap.head, "nc", -1)) != 1:
        tap.close()
        raise RunnerError(
            f"unexpected pose head contract: nl={tap.level_count} "
            f"nc={getattr(tap.head, 'nc', None)}"
        )
    return tap


def read_prepared(row: Mapping[str, Any]) -> np.ndarray:
    image = cv2.imread(str(REPO / row["image_path"]))
    if image is None:
        raise RunnerError(f"image decode failed: {row['image_path']}")
    raw_h, raw_w = (int(x) for x in row["raw_shape_height_width"])
    expected = (raw_h + 200, raw_w + 200)
    if image.shape[:2] != expected:
        raise RunnerError(
            f"prepared shape mismatch: {row['merged_stem']} {image.shape[:2]} != {expected}"
        )
    return image


def shard_fingerprint(
    recipe_sha: str, indices: np.ndarray, rows: Sequence[Mapping[str, Any]]
) -> str:
    digest = hashlib.sha256(recipe_sha.encode("ascii"))
    digest.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
    for index in indices.tolist():
        digest.update(
            json.dumps(rows[index], sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
        image_path = REPO / str(rows[index]["image_path"])
        digest.update(bytes.fromhex(sha256(image_path)))
    return digest.hexdigest()


def validate_shard(path: Path, fingerprint: str, indices: np.ndarray) -> bool:
    n = len(indices)
    expected = {
        "indices": ((n,), np.dtype(np.int32)),
        "fingerprint": ((), None),
        "valid": ((n,), np.dtype(np.bool_)),
        "patches": ((n, PATCH_TOKENS, TOKEN_DIM), np.dtype(np.float16)),
        "patch_masks": ((n, PATCH_TOKENS), np.dtype(np.bool_)),
        "source_levels": ((n,), np.dtype(np.int8)),
        "source_rows": ((n,), np.dtype(np.int16)),
        "source_columns": ((n,), np.dtype(np.int16)),
        "kp_features": ((n, KP_DIM), np.dtype(np.float32)),
        "confidences": ((n,), np.dtype(np.float32)),
        "detection_counts": ((n,), np.dtype(np.int16)),
        "logit_conf_abs_diff": ((n,), np.dtype(np.float32)),
        "center_exact": ((n,), np.dtype(np.bool_)),
        "center_float16_quantization_diff": ((n,), np.dtype(np.float32)),
        "rerun_exact": ((n,), np.dtype(np.bool_)),
        "rerun_candidate_count_diff": ((n,), np.dtype(np.int16)),
        "rerun_confidence_diff": ((n,), np.dtype(np.float32)),
        "rerun_box_diff": ((n,), np.dtype(np.float32)),
        "rerun_keypoint_diff": ((n,), np.dtype(np.float32)),
        "rerun_center_diff": ((n,), np.dtype(np.float32)),
        "rerun_source_level_match": ((n,), np.dtype(np.bool_)),
        "rerun_source_cell_match": ((n,), np.dtype(np.bool_)),
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(expected):
                return False
            for name, (shape, dtype) in expected.items():
                if data[name].shape != shape:
                    return False
                if dtype is not None and data[name].dtype != dtype:
                    return False
            return bool(
                str(data["fingerprint"].item()) == fingerprint
                and np.array_equal(data["indices"], indices)
            )
    except (OSError, ValueError, KeyError):
        return False


def validate_feature_population(arrays: Mapping[str, np.ndarray], n: int) -> None:
    required_shapes = {
        "valid": (n,),
        "patches": (n, PATCH_TOKENS, TOKEN_DIM),
        "patch_masks": (n, PATCH_TOKENS),
        "source_levels": (n,),
        "source_rows": (n,),
        "source_columns": (n,),
        "kp_features": (n, KP_DIM),
        "confidences": (n,),
        "detection_counts": (n,),
        "logit_conf_abs_diff": (n,),
        "center_exact": (n,),
        "center_float16_quantization_diff": (n,),
        "rerun_exact": (n,),
        "rerun_candidate_count_diff": (n,),
        "rerun_confidence_diff": (n,),
        "rerun_box_diff": (n,),
        "rerun_keypoint_diff": (n,),
        "rerun_center_diff": (n,),
        "rerun_source_level_match": (n,),
        "rerun_source_cell_match": (n,),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or np.asarray(arrays[name]).shape != shape:
            raise RunnerError(
                f"feature population shape invalid for {name}: "
                f"{None if name not in arrays else np.asarray(arrays[name]).shape} != {shape}"
            )
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    invalid = ~valid
    patches = np.asarray(arrays["patches"])
    masks = np.asarray(arrays["patch_masks"], dtype=np.bool_)
    kp = np.asarray(arrays["kp_features"])
    confidence = np.asarray(arrays["confidences"])
    provenance = np.asarray(arrays["logit_conf_abs_diff"])
    levels = np.asarray(arrays["source_levels"])
    rows = np.asarray(arrays["source_rows"])
    columns = np.asarray(arrays["source_columns"])
    counts = np.asarray(arrays["detection_counts"])
    if not bool(valid.any()):
        raise RunnerError("feature population has no valid detections")
    if not np.isfinite(patches).all() or np.any(patches[~masks] != 0):
        raise RunnerError("feature patch finite/masked-zero invariant failed")
    if not np.array_equal(np.any(masks, axis=1), valid):
        raise RunnerError("feature patch-mask validity invariant failed")
    if np.any(~masks[valid, PATCH_TOKENS // 2]):
        raise RunnerError("valid feature patch center is masked")
    if (
        not np.isfinite(kp[valid]).all()
        or not np.isfinite(confidence[valid]).all()
        or not np.isfinite(provenance[valid]).all()
        or np.any(provenance[valid] > 1.0e-5)
    ):
        raise RunnerError("valid prediction feature/provenance invariant failed")
    if (
        np.any(~np.isnan(kp[invalid]))
        or np.any(~np.isnan(confidence[invalid]))
        or np.any(~np.isnan(provenance[invalid]))
    ):
        raise RunnerError("invalid prediction NaN sentinel invariant failed")
    if (
        np.any((levels[valid] < 0) | (levels[valid] >= 3))
        or np.any(rows[valid] < 0)
        or np.any(columns[valid] < 0)
        or np.any(counts[valid] <= 0)
        or np.any(levels[invalid] != -1)
        or np.any(rows[invalid] != -1)
        or np.any(columns[invalid] != -1)
        or np.any(counts[invalid] != 0)
    ):
        raise RunnerError("source-cell/detection-count sentinel invariant failed")
    exact_flags = (
        "center_exact",
        "rerun_exact",
        "rerun_source_level_match",
        "rerun_source_cell_match",
    )
    if any(not bool(np.asarray(arrays[name], dtype=np.bool_).all()) for name in exact_flags):
        raise RunnerError("feature extraction exactness flag failed")
    zero_diffs = (
        "rerun_candidate_count_diff",
        "rerun_confidence_diff",
        "rerun_box_diff",
        "rerun_keypoint_diff",
        "rerun_center_diff",
    )
    if any(np.any(np.asarray(arrays[name]) != 0) for name in zero_diffs):
        raise RunnerError("two-pass feature extraction difference is nonzero")
    quantization = np.asarray(arrays["center_float16_quantization_diff"])
    if not np.isfinite(quantization).all() or np.any(quantization < 0):
        raise RunnerError("float16 patch quantization audit invalid")


def save_shard(
    path: Path,
    fingerprint: str,
    indices: np.ndarray,
    extracted: Sequence[Mapping[str, Any]],
) -> None:
    if len(extracted) != len(indices):
        raise RunnerError(
            f"shard extraction count mismatch: {len(extracted)} != {len(indices)}"
        )
    save_npz_exclusive(
        path,
        indices=np.asarray(indices, dtype=np.int32),
        fingerprint=np.asarray(fingerprint),
        valid=np.asarray([r["valid"] for r in extracted], dtype=np.bool_),
        patches=np.asarray([r["patch"] for r in extracted], dtype=np.float16),
        patch_masks=np.asarray([r["mask"] for r in extracted], dtype=np.bool_),
        source_levels=np.asarray([r["level"] for r in extracted], dtype=np.int8),
        source_rows=np.asarray([r["row"] for r in extracted], dtype=np.int16),
        source_columns=np.asarray([r["column"] for r in extracted], dtype=np.int16),
        kp_features=np.asarray([r["kp_feature"] for r in extracted], dtype=np.float32),
        confidences=np.asarray([r["confidence"] for r in extracted], dtype=np.float32),
        detection_counts=np.asarray(
            [r["detection_count"] for r in extracted], dtype=np.int16
        ),
        logit_conf_abs_diff=np.asarray(
            [r["provenance_diff"] for r in extracted], dtype=np.float32
        ),
        center_exact=np.asarray([r["center_exact"] for r in extracted], dtype=np.bool_),
        center_float16_quantization_diff=np.asarray(
            [r["center_float16_quantization_diff"] for r in extracted],
            dtype=np.float32,
        ),
        rerun_exact=np.asarray([r["rerun_exact"] for r in extracted], dtype=np.bool_),
        rerun_candidate_count_diff=np.asarray(
            [r["rerun_candidate_count_diff"] for r in extracted], dtype=np.int16
        ),
        rerun_confidence_diff=np.asarray(
            [r["rerun_confidence_diff"] for r in extracted], dtype=np.float32
        ),
        rerun_box_diff=np.asarray(
            [r["rerun_box_diff"] for r in extracted], dtype=np.float32
        ),
        rerun_keypoint_diff=np.asarray(
            [r["rerun_keypoint_diff"] for r in extracted], dtype=np.float32
        ),
        rerun_center_diff=np.asarray(
            [r["rerun_center_diff"] for r in extracted], dtype=np.float32
        ),
        rerun_source_level_match=np.asarray(
            [r["rerun_source_level_match"] for r in extracted], dtype=np.bool_
        ),
        rerun_source_cell_match=np.asarray(
            [r["rerun_source_cell_match"] for r in extracted], dtype=np.bool_
        ),
    )


def extract_one(
    reference: Mapping[str, Any],
    result: Any,
    vector: Mapping[str, Any] | None,
    raw_shape: tuple[int, int],
    tap: Any,
    batch_index: int,
) -> dict[str, Any]:
    base = phase_c()
    spatial = phase_e()
    pred = base._extract_result(result, vector, raw_shape)
    patch = spatial._patch_from_source_cell(tap, vector, batch_index)
    if bool(reference["valid"]) != bool(pred["valid"]):
        raise RunnerError("two-pass prediction validity mismatch")
    if bool(pred["valid"]) != bool(vector is not None):
        raise RunnerError("prediction/vector validity mismatch")
    candidate_count_diff = abs(
        int(reference["detection_count"]) - int(pred["detection_count"])
    )
    confidence_diff = 0.0
    box_diff = 0.0
    keypoint_diff = 0.0
    center_diff = 0.0
    source_level_match = True
    source_cell_match = True
    if pred["valid"]:
        confidence_diff = abs(float(reference["confidence"]) - float(pred["confidence"]))
        box_diff = float(
            np.max(
                np.abs(
                    np.asarray(reference["box"], dtype=np.float32)
                    - np.asarray(pred["box"], dtype=np.float32)
                )
            )
        )
        keypoint_diff = float(
            np.max(
                np.abs(
                    np.asarray(reference["keypoints"], dtype=np.float32)
                    - np.asarray(pred["keypoints"], dtype=np.float32)
                )
            )
        )
        center_diff = float(
            np.max(
                np.abs(
                    np.asarray(reference["image_feature"], dtype=np.float32)
                    - np.asarray(pred["image_feature"], dtype=np.float32)
                )
            )
        )
        source_level_match = bool(
            int(reference["level"]) == int(pred["level"]) == int(patch["level"])
        )
        source_cell_match = bool(
            int(reference["source_flat"]) == int(vector["flat"])
            and int(reference["source_top"]) == int(vector["top"])
        )
        if int(pred["level"]) != int(patch["level"]):
            raise RunnerError("prediction/patch source-level mismatch")
        if max(float(reference["provenance_diff"]), float(pred["provenance_diff"])) > 1.0e-5:
            raise RunnerError("selected logit/confidence provenance mismatch")
        if not patch["patch_center_vector_exact"]:
            raise RunnerError("patch center/source vector mismatch")
    rerun_exact = bool(
        candidate_count_diff == 0
        and confidence_diff == 0.0
        and box_diff == 0.0
        and keypoint_diff == 0.0
        and center_diff == 0.0
        and source_level_match
        and source_cell_match
    )
    if not rerun_exact:
        raise RunnerError(
            "two-pass instrumentation mismatch: "
            f"candidate_count={candidate_count_diff} confidence={confidence_diff} "
            f"box={box_diff} keypoints={keypoint_diff} center={center_diff} "
            f"source_level_match={source_level_match} "
            f"source_cell_match={source_cell_match}"
        )
    return {
        **pred,
        **{key: patch[key] for key in ("patch", "mask", "row", "column")},
        "level": patch["level"],
        "center_exact": patch["patch_center_vector_exact"],
        "center_float16_quantization_diff": patch[
            "center_float16_quantization_diff"
        ],
        "rerun_exact": rerun_exact,
        "rerun_candidate_count_diff": candidate_count_diff,
        "rerun_confidence_diff": confidence_diff,
        "rerun_box_diff": box_diff,
        "rerun_keypoint_diff": keypoint_diff,
        "rerun_center_diff": center_diff,
        "rerun_source_level_match": source_level_match,
        "rerun_source_cell_match": source_cell_match,
    }


def extract(device: str) -> dict[str, Any]:
    if FEATURE_CACHE.exists() or FEATURE_AUDIT.exists():
        raise RunnerError("final feature cache/audit already exists")
    lock = validate_base()
    rows, metadata_arrays = load_metadata()
    base_sha = lock["last_pt"]["sha256"]
    metadata_sha = sha256(METADATA)
    recipe = feature_recipe(base_sha, metadata_sha)
    checkpoint_mtime = BASE_LAST.stat().st_mtime_ns
    tap = make_tap(BASE_LAST, base_sha, device)
    shard_paths: list[Path] = []
    fingerprints: dict[Path, str] = {}
    expected_indices: dict[Path, np.ndarray] = {}
    processed = 0
    started = time.perf_counter()
    try:
        by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, raw_shape in enumerate(metadata_arrays["raw_shapes"].tolist()):
            by_shape[tuple(int(x) for x in raw_shape)].append(index)
        for raw_shape, population in sorted(by_shape.items()):
            for offset in range(0, len(population), SHARD_SIZE):
                indices = np.asarray(population[offset : offset + SHARD_SIZE], dtype=np.int32)
                path = SHARD_DIR / (
                    f"h{raw_shape[0]}_w{raw_shape[1]}_"
                    f"{int(indices[0]):05d}_{int(indices[-1])+1:05d}.npz"
                )
                fingerprint = shard_fingerprint(recipe["sha256"], indices, rows)
                shard_paths.append(path)
                fingerprints[path] = fingerprint
                expected_indices[path] = indices.copy()
                if path.is_file():
                    if not validate_shard(path, fingerprint, indices):
                        raise RunnerError(f"existing shard contract mismatch: {path}")
                    processed += len(indices)
                    continue
                output: list[dict[str, Any]] = []
                for start in range(0, len(indices), EXTRACT_BATCH):
                    batch_indices = indices[start : start + EXTRACT_BATCH]
                    images = [read_prepared(rows[i]) for i in batch_indices.tolist()]
                    first_results, first_vectors = tap.predict_batch(images)
                    references = []
                    for index, result, vector in zip(
                        batch_indices.tolist(), first_results, first_vectors
                    ):
                        shape = tuple(
                            int(x) for x in rows[index]["raw_shape_height_width"]
                        )
                        references.append(phase_c()._extract_result(result, vector, shape))
                        references[-1]["source_flat"] = (
                            None if vector is None else int(vector["flat"])
                        )
                        references[-1]["source_top"] = (
                            None if vector is None else int(vector["top"])
                        )
                    del first_results, first_vectors
                    results, vectors = tap.predict_batch(images)
                    for local, (index, result, vector) in enumerate(
                        zip(batch_indices.tolist(), results, vectors)
                    ):
                        shape = tuple(
                            int(x) for x in rows[index]["raw_shape_height_width"]
                        )
                        output.append(
                            extract_one(
                                references[local], result, vector, shape, tap, local
                            )
                        )
                save_shard(path, fingerprint, indices, output)
                processed += len(indices)
                print(
                    f"[{time.strftime('%H:%M:%S')}] spatial extract {processed}/60000",
                    flush=True,
                )
    finally:
        tap.close()
        del tap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if sha256(BASE_LAST) != base_sha or BASE_LAST.stat().st_mtime_ns != checkpoint_mtime:
        raise RunnerError("base checkpoint changed during extraction")

    n = len(rows)
    assembled = {
        "valid": np.zeros(n, dtype=np.bool_),
        "patches": np.zeros((n, PATCH_TOKENS, TOKEN_DIM), dtype=np.float16),
        "patch_masks": np.zeros((n, PATCH_TOKENS), dtype=np.bool_),
        "source_levels": np.full(n, -1, dtype=np.int8),
        "source_rows": np.full(n, -1, dtype=np.int16),
        "source_columns": np.full(n, -1, dtype=np.int16),
        "kp_features": np.full((n, KP_DIM), np.nan, dtype=np.float32),
        "confidences": np.full(n, np.nan, dtype=np.float32),
        "detection_counts": np.zeros(n, dtype=np.int16),
        "logit_conf_abs_diff": np.full(n, np.nan, dtype=np.float32),
        "center_exact": np.zeros(n, dtype=np.bool_),
        "center_float16_quantization_diff": np.zeros(n, dtype=np.float32),
        "rerun_exact": np.zeros(n, dtype=np.bool_),
        "rerun_candidate_count_diff": np.zeros(n, dtype=np.int16),
        "rerun_confidence_diff": np.zeros(n, dtype=np.float32),
        "rerun_box_diff": np.zeros(n, dtype=np.float32),
        "rerun_keypoint_diff": np.zeros(n, dtype=np.float32),
        "rerun_center_diff": np.zeros(n, dtype=np.float32),
        "rerun_source_level_match": np.zeros(n, dtype=np.bool_),
        "rerun_source_cell_match": np.zeros(n, dtype=np.bool_),
    }
    filled = np.zeros(n, dtype=np.bool_)
    for path in shard_paths:
        observed_fingerprint = shard_fingerprint(
            recipe["sha256"], expected_indices[path], rows
        )
        if observed_fingerprint != fingerprints[path]:
            raise RunnerError(f"prepared image/metadata changed during extraction: {path}")
        if not validate_shard(path, fingerprints[path], expected_indices[path]):
            raise RunnerError(f"shard changed before assembly: {path}")
        with np.load(path, allow_pickle=False) as data:
            indices = np.asarray(data["indices"], dtype=np.int64)
            if (
                np.any(indices < 0)
                or np.any(indices >= n)
                or np.any(filled[indices])
            ):
                raise RunnerError("duplicate shard indices")
            for key in assembled:
                assembled[key][indices] = data[key]
            filled[indices] = True
    if not filled.all():
        raise RunnerError(f"feature shard coverage incomplete: {int(filled.sum())}/{n}")
    validate_feature_population(assembled, n)
    valid = assembled["valid"]
    if np.any(assembled["source_levels"][valid] < 0) or np.any(
        assembled["source_levels"][valid] >= 3
    ):
        raise RunnerError("valid source level invalid")
    if np.any(~assembled["patch_masks"][valid, PATCH_TOKENS // 2]):
        raise RunnerError("valid patch center masked")
    if np.any(assembled["patch_masks"][~valid]) or np.any(
        assembled["patches"][~valid] != 0
    ):
        raise RunnerError("invalid row patch sentinel invalid")
    if not np.all(assembled["center_exact"][valid]):
        raise RunnerError("valid center exactness failed")
    if not np.all(assembled["rerun_exact"]):
        raise RunnerError("two-pass inference exactness failed")
    if not np.all(assembled["rerun_source_level_match"]):
        raise RunnerError("two-pass source-level exactness failed")
    if not np.all(assembled["rerun_source_cell_match"]):
        raise RunnerError("two-pass source-cell exactness failed")

    train_valid = (metadata_arrays["splits"] == 0) & valid
    patch_mean, patch_std, patch_tokens = phase_e()._patch_normalization(
        assembled["patches"], assembled["patch_masks"], train_valid
    )
    dim_raw = phase_c()._dimension_raw(metadata_arrays["dimensions_xyz"])
    normalization: dict[str, dict[str, Any]] = {
        "patch": {
            "mean": patch_mean.tolist(),
            "std": patch_std.tolist(),
            "valid_token_count": int(patch_tokens),
            "source": "MIXED_TRAIN_VALID_TOKENS_ONLY",
        }
    }
    for name, values in (
        ("kp", assembled["kp_features"]),
        ("dims", dim_raw),
    ):
        selected = np.asarray(values[train_valid], dtype=np.float64)
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std[std < 1.0e-8] = 1.0
        normalization[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source": "MIXED_TRAIN_VALID_ONLY",
        }
    cache_metadata = {
        "schema_version": "cleanstart_spatial_feature_cache_60k_v1",
        "created_at_utc": now(),
        "base_checkpoint_sha256": base_sha,
        "probe_metadata_sha256": metadata_sha,
        "feature_recipe": recipe,
        "normalization": normalization,
        "row_count": n,
        "yolo_parameter_updates": 0,
    }
    save_npz_exclusive(
        FEATURE_CACHE,
        metadata_json=np.asarray(
            json.dumps(cache_metadata, sort_keys=True, separators=(",", ":"))
        ),
        **metadata_arrays,
        **assembled,
    )
    audit = {
        "schema_version": "cleanstart_spatial_feature_cache_60k_audit_v1",
        "created_at_utc": now(),
        "feature_cache": {
            "path": str(FEATURE_CACHE.relative_to(REPO)),
            "sha256": sha256(FEATURE_CACHE),
            "size_bytes": FEATURE_CACHE.stat().st_size,
        },
        "feature_recipe": recipe,
        "base_checkpoint_sha256_before_after": [base_sha, sha256(BASE_LAST)],
        "rows": n,
        "valid": int(valid.sum()),
        "invalid_abstentions": int((~valid).sum()),
        "train_valid": int(train_valid.sum()),
        "val_valid": int(((metadata_arrays["splits"] == 1) & valid).sum()),
        "all_valid_center_exact": True,
        "two_pass_inference_exact": bool(assembled["rerun_exact"].all()),
        "prepared_image_content_bound_in_shards_and_rechecked": True,
        "two_pass_max_abs_diff": {
            "candidate_count": int(assembled["rerun_candidate_count_diff"].max()),
            "confidence": float(assembled["rerun_confidence_diff"].max()),
            "box": float(assembled["rerun_box_diff"].max()),
            "keypoints": float(assembled["rerun_keypoint_diff"].max()),
            "float32_center": float(assembled["rerun_center_diff"].max()),
        },
        "max_center_float16_quantization_diff": float(
            assembled["center_float16_quantization_diff"].max()
        ),
        "max_logit_conf_abs_diff": float(
            np.nanmax(assembled["logit_conf_abs_diff"])
        ),
        "normalization": normalization,
        "shards": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "fingerprint": fingerprints[path],
            }
            for path in shard_paths
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    write_json_exclusive(FEATURE_AUDIT, audit)
    return audit


def load_cache() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    if not FEATURE_CACHE.is_file() or not FEATURE_AUDIT.is_file():
        raise RunnerError("feature cache/audit required")
    audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit["feature_cache"]["sha256"] != sha256(
        FEATURE_CACHE
    ):
        raise RunnerError("feature cache audit mismatch")
    lock = validate_base()
    with np.load(FEATURE_CACHE, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files if key != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"].item()))
    validate_feature_population(arrays, int(metadata.get("row_count", -1)))
    if metadata["base_checkpoint_sha256"] != lock["last_pt"]["sha256"]:
        raise RunnerError("feature cache/base checkpoint mismatch")
    if metadata["probe_metadata_sha256"] != sha256(METADATA):
        raise RunnerError("feature cache/probe metadata mismatch")
    expected_recipe = feature_recipe(
        lock["last_pt"]["sha256"], sha256(METADATA)
    )
    if (
        metadata.get("feature_recipe") != expected_recipe
        or audit.get("feature_recipe") != expected_recipe
        or audit.get("rows") != 60000
        or audit.get("base_checkpoint_sha256_before_after")
        != [lock["last_pt"]["sha256"], lock["last_pt"]["sha256"]]
    ):
        raise RunnerError("feature cache recipe/provenance mismatch")
    return arrays, metadata, audit


def bundle(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    norm = metadata["normalization"]
    dims_raw = phase_c()._dimension_raw(arrays["dimensions_xyz"])
    return {
        "patches": np.asarray(arrays["patches"], dtype=np.float16),
        "patch_masks": np.asarray(arrays["patch_masks"], dtype=np.bool_),
        "source_levels": np.asarray(arrays["source_levels"], dtype=np.int8),
        "kp": (
            np.asarray(arrays["kp_features"], dtype=np.float32)
            - np.asarray(norm["kp"]["mean"], dtype=np.float32)
        )
        / np.asarray(norm["kp"]["std"], dtype=np.float32),
        "dims": (
            dims_raw - np.asarray(norm["dims"]["mean"], dtype=np.float32)
        )
        / np.asarray(norm["dims"]["std"], dtype=np.float32),
        "patch_mean": np.asarray(norm["patch"]["mean"], dtype=np.float32),
        "patch_std": np.asarray(norm["patch"]["std"], dtype=np.float32),
    }


def batch_from_bundle(
    values: Mapping[str, np.ndarray], indices: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    indices = np.asarray(indices, dtype=np.int64)
    patches = np.asarray(values["patches"])[indices].astype(np.float32)
    patches = (
        patches
        - np.asarray(values["patch_mean"], dtype=np.float32)[None, None, :]
    ) / np.asarray(values["patch_std"], dtype=np.float32)[None, None, :]
    masks = np.asarray(values["patch_masks"], dtype=np.bool_)[indices]
    patches[~masks] = 0.0
    return (
        torch.from_numpy(patches).to(device, non_blocking=True),
        torch.from_numpy(masks).to(device, non_blocking=True),
        torch.from_numpy(
            np.asarray(values["source_levels"], dtype=np.int64)[indices]
        ).to(device, non_blocking=True),
        torch.from_numpy(np.asarray(values["kp"], dtype=np.float32)[indices]).to(
            device, non_blocking=True
        ),
        torch.from_numpy(np.asarray(values["dims"], dtype=np.float32)[indices]).to(
            device, non_blocking=True
        ),
    )


def predict(
    model: torch.nn.Module,
    values: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    output = np.empty(len(indices), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), EVAL_BATCH):
            selected = indices[start : start + EVAL_BATCH]
            logits = model(*batch_from_bundle(values, selected, device))
            output[start : start + len(selected)] = (
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
    return output


def model_config(arm: str) -> dict[str, Any]:
    return {
        "class": "Phase-E SpatialFusionProbe",
        "arm": arm,
        "patch_size": 7,
        "tokens": 49,
        "token_dim": 64,
        "pool": "masked_mean_plus_masked_max",
        "kp_dim": 26,
        "dimension_dim": 4,
        "classifier_hidden": [128, 32],
        "activation": "SiLU",
    }


def source_bindings(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        "base_checkpoint_sha256": sha256(BASE_LAST),
        "base_lock_sha256": sha256(BASE_LOCK),
        "base_interruption_event_sha256": sha256(BASE_INTERRUPTION_EVENT),
        "probe_metadata_sha256": sha256(METADATA),
        "probe_metadata_audit_sha256": sha256(METADATA_AUDIT),
        "feature_cache_sha256": sha256(FEATURE_CACHE),
        "feature_audit_sha256": sha256(FEATURE_AUDIT),
        "feature_recipe_sha256": metadata["feature_recipe"]["sha256"],
        "phase_e_runner_sha256": sha256(PHASE_E_RUNNER),
        "spatial_stage_contract_sha256": sha256(SPATIAL_CONTRACT),
        "spatial_extraction_runner_sha256": sha256(EXTRACTION_RUNNER),
        "probe_training_runner_sha256": sha256(Path(__file__)),
        "probe_training_recovery_contract_sha256": sha256(
            PROBE_TRAINING_RECOVERY_CONTRACT
        ),
        "training_contract_sha256": sha256(TRAINING_CONTRACT),
    }


def run_paths(arm: str, seed: int) -> tuple[Path, Path]:
    root = PROBE_DIR / arm
    return root / f"seed{seed}.pt", root / f"seed{seed}.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    return phase_c()._binary_metrics(labels, probabilities, valid)


def train_one(
    arm: str,
    seed: int,
    values: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    dev_indices: np.ndarray,
    device: str,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    checkpoint, run_json = run_paths(arm, seed)
    if checkpoint.exists() or run_json.exists():
        if not checkpoint.is_file() or not run_json.is_file():
            raise RunnerError(f"partial probe artifact: {arm} seed{seed}")
        existing = json.loads(run_json.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") != "cleanstart_spatial_probe_run_v1"
            or existing.get("arm") != arm
            or int(existing.get("seed", -1)) != seed
            or existing.get("model_config") != model_config(arm)
            or existing.get("source_bindings") != dict(bindings)
            or existing.get("optimizer") != "AdamW"
            or float(existing.get("learning_rate", -1.0)) != LR
            or float(existing.get("weight_decay", -1.0)) != WEIGHT_DECAY
            or existing.get("betas") != [0.9, 0.999]
            or float(existing.get("epsilon", -1.0)) != 1.0e-8
            or existing.get("scheduler") != "none"
            or existing.get("gradient_clipping") != "none"
            or existing.get("amp") is not False
            or int(existing.get("epochs", -1)) != TRAIN_EPOCHS
            or int(existing.get("steps_per_epoch", -1)) != TRAIN_STEPS
            or int(existing.get("samples_per_epoch_with_replacement", -1))
            != TRAIN_STEPS * TRAIN_BATCH
            or existing.get("real_accessed_during_training_or_selection") is not False
            or existing.get("checkpoint", {}).get("sha256") != sha256(checkpoint)
        ):
            raise RunnerError(f"existing probe binding mismatch: {arm} seed{seed}")
        payload = torch.load(checkpoint, map_location="cpu")
        if (
            payload.get("schema_version")
            != "cleanstart_spatial_probe_checkpoint_v1"
            or payload.get("arm") != arm
            or int(payload.get("seed", -1)) != seed
            or payload.get("model_config") != model_config(arm)
            or payload.get("source_bindings") != dict(bindings)
            or int(payload.get("selected_epoch", -1))
            != int(existing.get("selected_epoch", -2))
        ):
            raise RunnerError(f"existing probe checkpoint mismatch: {arm} seed{seed}")
        return existing

    labels = np.asarray(arrays["labels"], dtype=np.int64)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    seed_everything(seed)
    model = phase_e().SpatialFusionProbe(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    train_y = labels[train_indices]
    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    class_weights_np = len(train_y) / (2.0 * counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    sample_weights = class_weights_np[train_y]
    schedule_digest = hashlib.sha256()
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    dev_valid_indices = dev_indices[valid[dev_indices]]
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + epoch)
        local = torch.multinomial(
            torch.from_numpy(sample_weights),
            num_samples=TRAIN_STEPS * TRAIN_BATCH,
            replacement=True,
            generator=generator,
        ).numpy()
        selected = train_indices[local]
        schedule_digest.update(np.ascontiguousarray(selected, dtype=np.int32).tobytes())
        losses = []
        for step in range(TRAIN_STEPS):
            indices = selected[step * TRAIN_BATCH : (step + 1) * TRAIN_BATCH]
            target = torch.from_numpy(labels[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch_from_bundle(values, indices, device))
            loss = F.cross_entropy(logits, target, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_probability = np.full(len(dev_indices), np.nan, dtype=np.float64)
        position = {int(index): pos for pos, index in enumerate(dev_indices.tolist())}
        observed = predict(model, values, dev_valid_indices, device)
        for index, probability in zip(dev_valid_indices.tolist(), observed.tolist()):
            dev_probability[position[index]] = probability
        metrics = binary_metrics(
            labels[dev_indices], dev_probability, valid[dev_indices]
        )
        score = float(metrics["balanced_accuracy"])
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "dev_balanced_accuracy": score,
            "dev_accuracy": metrics["accuracy"],
            "dev_coverage": metrics["coverage"],
        }
        history.append(record)
        if best is None or score > float(best["score"]):
            best = {
                "score": score,
                "epoch": epoch + 1,
                "state_dict": copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                ),
                "dev_metrics": metrics,
            }
    if best is None:
        raise RunnerError("probe training produced no checkpoint")
    payload = {
        "schema_version": "cleanstart_spatial_probe_checkpoint_v1",
        "arm": arm,
        "seed": seed,
        "selected_epoch": best["epoch"],
        "model_config": model_config(arm),
        "source_bindings": dict(bindings),
        "state_dict": best["state_dict"],
    }
    save_torch_exclusive(checkpoint, payload)
    run = {
        "schema_version": "cleanstart_spatial_probe_run_v1",
        "created_at_utc": now(),
        "arm": arm,
        "seed": seed,
        "model_config": model_config(arm),
        "source_bindings": dict(bindings),
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "scheduler": "none",
        "gradient_clipping": "none",
        "amp": False,
        "epochs": TRAIN_EPOCHS,
        "steps_per_epoch": TRAIN_STEPS,
        "samples_per_epoch_with_replacement": TRAIN_STEPS * TRAIN_BATCH,
        "fixed_optimizer_steps": TRAIN_EPOCHS * TRAIN_STEPS,
        "class_balanced_sampler": True,
        "balanced_cross_entropy_weights": class_weights_np.tolist(),
        "sampler_schedule_sha256": schedule_digest.hexdigest(),
        "checkpoint_epoch_selected_by": "earliest maximum mixed synthetic DEV balanced accuracy",
        "selected_epoch": best["epoch"],
        "dev": best["dev_metrics"],
        "history": history,
        "real_accessed_during_training_or_selection": False,
        "yolo_parameters_in_optimizer": 0,
        "checkpoint": {
            "path": str(checkpoint.relative_to(REPO)),
            "sha256": sha256(checkpoint),
        },
    }
    write_json_exclusive(run_json, run)
    return run


def load_probe(path: Path, device: str, bindings: Mapping[str, str]) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu")
    arm = str(payload.get("arm"))
    if (
        payload.get("schema_version") != "cleanstart_spatial_probe_checkpoint_v1"
        or arm not in ARMS
        or payload.get("source_bindings") != dict(bindings)
        or payload.get("model_config") != model_config(arm)
    ):
        raise RunnerError(f"probe checkpoint bindings mismatch: {path}")
    model = phase_e().SpatialFusionProbe(arm)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def median_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(runs) != 3:
        raise RunnerError("median run requires three seeds")
    scores = sorted(float(run["dev"]["balanced_accuracy"]) for run in runs)
    median_score = scores[len(scores) // 2]
    matches = [
        run
        for run in runs
        if float(run["dev"]["balanced_accuracy"]) == median_score
    ]
    return min(matches, key=lambda run: int(run["seed"]))


def train(device: str) -> dict[str, Any]:
    recovery_contract = validate_probe_training_recovery_contract()
    existing_final = [
        path for path in (SUMMARY, REPORT, DEV_PREDICTIONS) if path.exists()
    ]
    if existing_final:
        raise RunnerError(f"final probe artifacts already exist: {existing_final}")
    arrays, cache_metadata, feature_audit = load_cache()
    values = bundle(arrays, cache_metadata)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    splits = np.asarray(arrays["splits"], dtype=np.int8)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    train_indices = np.flatnonzero(splits == 0)
    dev_indices = np.flatnonzero(splits == 1)
    if len(train_indices) != 55980 or len(dev_indices) != 4020:
        raise RunnerError("probe train/dev population invalid")
    if not bool(valid[train_indices].all()):
        raise RunnerError(
            f"mixed synthetic TRAIN contains "
            f"{int((~valid[train_indices]).sum())} invalid detection rows"
        )
    dev_abstention_indices = dev_indices[~valid[dev_indices]]
    expected_abstentions = recovery_contract["policy"]
    if (
        dev_abstention_indices.tolist()
        != expected_abstentions["expected_dev_abstention_global_indices"]
        or np.asarray(arrays["frame_ids"])[dev_abstention_indices].tolist()
        != expected_abstentions["expected_dev_abstention_frame_ids"]
    ):
        raise RunnerError(
            f"unexpected DEV abstention population: "
            f"{dev_abstention_indices.tolist()}"
        )
    bindings = source_bindings(cache_metadata)

    step_zero: dict[str, Any] = {}
    for seed in SEEDS:
        seed_everything(seed)
        s0 = phase_e().SpatialFusionProbe(ARMS[0]).to(device)
        seed_everything(seed)
        s1 = phase_e().SpatialFusionProbe(ARMS[1]).to(device)
        common_exact = all(
            torch.equal(s0.state_dict()[key], s1.state_dict()[key])
            for key in s0.state_dict()
        )
        sample = train_indices[: min(8, len(train_indices))]
        sample_batch = list(batch_from_bundle(values, sample, device))
        sample_batch[-1] = torch.zeros_like(sample_batch[-1])
        s0.eval()
        s1.eval()
        with torch.no_grad():
            max_diff = float(torch.max(torch.abs(s0(*sample_batch) - s1(*sample_batch))))
        if not common_exact or max_diff > 1.0e-7:
            raise RunnerError("S0/S1 step-zero common initialization failed")
        step_zero[str(seed)] = {"common_state_exact": True, "zero_dim_logit_max_diff": max_diff}
        del s0, s1

    runs_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            print(f"training {arm} seed={seed}", flush=True)
            runs_by_arm[arm].append(
                train_one(
                    arm,
                    seed,
                    values,
                    arrays,
                    train_indices,
                    dev_indices,
                    device,
                    bindings,
                )
            )

    schedule_exact_by_seed: dict[str, bool] = {}
    for seed_position, seed in enumerate(SEEDS):
        hashes = {
            str(runs_by_arm[arm][seed_position]["sampler_schedule_sha256"])
            for arm in ARMS
        }
        schedule_exact_by_seed[str(seed)] = len(hashes) == 1
        if len(hashes) != 1:
            raise RunnerError(f"S0/S1 sampler schedule mismatch for seed {seed}: {hashes}")

    locked: dict[str, Any] = {}
    for arm, runs in runs_by_arm.items():
        median = median_run(runs)
        locked[arm] = {
            "mean_dev_balanced_accuracy": float(
                np.mean([r["dev"]["balanced_accuracy"] for r in runs])
            ),
            "median_seed": int(median["seed"]),
            "median_seed_dev": median["dev"],
            "checkpoint": median["checkpoint"],
        }

    dev_valid = dev_indices[valid[dev_indices]]
    probabilities: dict[str, np.ndarray] = {}
    for arm in ARMS:
        model = load_probe(REPO / locked[arm]["checkpoint"]["path"], device, bindings)
        full = np.full(len(dev_indices), np.nan, dtype=np.float64)
        pos = {int(index): i for i, index in enumerate(dev_indices.tolist())}
        observed = predict(model, values, dev_valid, device)
        for index, probability in zip(dev_valid.tolist(), observed.tolist()):
            full[pos[index]] = probability
        probabilities[arm] = full

    s1_model = load_probe(
        REPO / locked["S1_SPATIAL_CONCAT"]["checkpoint"]["path"], device, bindings
    )
    shuffled_xyz, permutation = phase_e()._shuffled_dimensions(
        np.asarray(arrays["dimensions_xyz"])[dev_indices],
        seed=42,
        require_all_triplets_changed=True,
    )
    norm = cache_metadata["normalization"]["dims"]
    shuffled_raw = phase_c()._dimension_raw(shuffled_xyz)
    shuffled_dims = (
        shuffled_raw - np.asarray(norm["mean"], dtype=np.float32)
    ) / np.asarray(norm["std"], dtype=np.float32)
    shuffled_bundle = dict(values)
    shuffled_bundle["dims"] = np.asarray(values["dims"]).copy()
    shuffled_bundle["dims"][dev_indices] = shuffled_dims
    shuffled_full = np.full(len(dev_indices), np.nan, dtype=np.float64)
    positions = {int(index): i for i, index in enumerate(dev_indices.tolist())}
    shuffled_observed = predict(s1_model, shuffled_bundle, dev_valid, device)
    for index, probability in zip(dev_valid.tolist(), shuffled_observed.tolist()):
        shuffled_full[positions[index]] = probability

    metrics = {
        arm: binary_metrics(labels[dev_indices], probability, valid[dev_indices])
        for arm, probability in probabilities.items()
    }
    metrics["S1_SPATIAL_CONCAT_SHUFFLED_DIMS"] = binary_metrics(
        labels[dev_indices], shuffled_full, valid[dev_indices]
    )
    dev_sources = np.asarray(arrays["sources"])[dev_indices]
    metrics_by_source: dict[str, dict[str, Any]] = {}
    evaluated_probabilities = {
        **probabilities,
        "S1_SPATIAL_CONCAT_SHUFFLED_DIMS": shuffled_full,
    }
    for arm, probability in evaluated_probabilities.items():
        metrics_by_source[arm] = {}
        for source in ("G38", "P0", "TEX"):
            selected = dev_sources == source
            metrics_by_source[arm][source] = binary_metrics(
                labels[dev_indices][selected],
                probability[selected],
                valid[dev_indices][selected],
            )
    save_npz_exclusive(
        DEV_PREDICTIONS,
        source_bindings_json=np.asarray(
            json.dumps(bindings, sort_keys=True, separators=(",", ":"))
        ),
        global_indices=np.asarray(dev_indices, dtype=np.int32),
        frame_ids=np.asarray(arrays["frame_ids"])[dev_indices],
        sources=dev_sources,
        labels=np.asarray(labels[dev_indices], dtype=np.int8),
        valid=np.asarray(valid[dev_indices], dtype=np.bool_),
        s0_long_probability=np.asarray(
            probabilities["S0_SPATIAL_NO_DIMS"], dtype=np.float64
        ),
        s1_long_probability=np.asarray(
            probabilities["S1_SPATIAL_CONCAT"], dtype=np.float64
        ),
        s1_shuffled_dimensions_long_probability=np.asarray(
            shuffled_full, dtype=np.float64
        ),
        shuffled_dimension_source_local_indices=np.asarray(
            permutation, dtype=np.int32
        ),
    )
    result = {
        "schema_version": "cleanstart_spatial_concat_60k_summary_v1",
        "created_at_utc": now(),
        "scope": "challenge-track exploratory; mixed validation is not an independent test",
        "architecture": "exact post-hoc Phase-E S1; YOLO frozen",
        "source_bindings": bindings,
        "population": {
            "train_total": int(np.sum(splits == 0)),
            "train_valid": int(valid[train_indices].sum()),
            "dev_total": len(dev_indices),
            "dev_valid": len(dev_valid),
            "dev_abstentions": len(dev_abstention_indices),
            "dev_abstention_frame_ids": np.asarray(arrays["frame_ids"])[
                dev_abstention_indices
            ].tolist(),
        },
        "dev_abstention_policy": recovery_contract["policy"],
        "step_zero": step_zero,
        "sampler_schedule_exact_across_arms_by_seed": schedule_exact_by_seed,
        "runs": runs_by_arm,
        "locked_median_artifacts": locked,
        "dev_metrics": metrics,
        "dev_metrics_by_source": metrics_by_source,
        "s1_minus_s0_balanced_accuracy": (
            metrics["S1_SPATIAL_CONCAT"]["balanced_accuracy"]
            - metrics["S0_SPATIAL_NO_DIMS"]["balanced_accuracy"]
        ),
        "s1_minus_shuffled_balanced_accuracy": (
            metrics["S1_SPATIAL_CONCAT"]["balanced_accuracy"]
            - metrics["S1_SPATIAL_CONCAT_SHUFFLED_DIMS"]["balanced_accuracy"]
        ),
        "shuffle": {
            "seed": 42,
            "all_fixed_xyz_triplets_changed": True,
            "permutation_sha256": hashlib.sha256(
                np.asarray(permutation, dtype=np.int32).tobytes()
            ).hexdigest(),
        },
        "feature_audit_sha256": sha256(FEATURE_AUDIT),
        "dev_predictions": {
            "path": str(DEV_PREDICTIONS.relative_to(REPO)),
            "sha256": sha256(DEV_PREDICTIONS),
        },
        "yolo_parameter_updates_during_probe": 0,
        "real_used_for_training_or_selection": 0,
    }
    write_json_exclusive(SUMMARY, result)
    report = (
        "# Clean-start 60k spatial-concat result\n\n"
        "This is a challenge-track exploratory result. The mixed validation set is "
        "not an independent paper holdout.\n\n"
        f"- S0 balanced accuracy: {metrics['S0_SPATIAL_NO_DIMS']['balanced_accuracy']:.6f}\n"
        f"- S1 balanced accuracy: {metrics['S1_SPATIAL_CONCAT']['balanced_accuracy']:.6f}\n"
        f"- S1 shuffled-dimension balanced accuracy: "
        f"{metrics['S1_SPATIAL_CONCAT_SHUFFLED_DIMS']['balanced_accuracy']:.6f}\n"
        f"- S1 - S0: {result['s1_minus_s0_balanced_accuracy']:+.6f}\n"
        f"- S1 - shuffled: {result['s1_minus_shuffled_balanced_accuracy']:+.6f}\n"
        f"- DEV coverage: {metrics['S1_SPATIAL_CONCAT']['coverage']:.6f}\n"
        f"- DEV abstentions counted as errors: {len(dev_abstention_indices)}\n"
        f"- G38 S0/S1 balanced accuracy: "
        f"{metrics_by_source['S0_SPATIAL_NO_DIMS']['G38']['balanced_accuracy']:.6f} / "
        f"{metrics_by_source['S1_SPATIAL_CONCAT']['G38']['balanced_accuracy']:.6f}\n"
        f"- P0 S0/S1 balanced accuracy: "
        f"{metrics_by_source['S0_SPATIAL_NO_DIMS']['P0']['balanced_accuracy']:.6f} / "
        f"{metrics_by_source['S1_SPATIAL_CONCAT']['P0']['balanced_accuracy']:.6f}\n"
        f"- TEX S0/S1 balanced accuracy: "
        f"{metrics_by_source['S0_SPATIAL_NO_DIMS']['TEX']['balanced_accuracy']:.6f} / "
        f"{metrics_by_source['S1_SPATIAL_CONCAT']['TEX']['balanced_accuracy']:.6f}\n"
    )
    write_text_exclusive(REPORT, report)
    return result


def unit_smoke(device: str) -> dict[str, Any]:
    seed_everything(0)
    models = {arm: phase_e().SpatialFusionProbe(arm).to(device).eval() for arm in ARMS}
    patches = torch.randn(4, PATCH_TOKENS, TOKEN_DIM, device=device)
    masks = torch.ones(4, PATCH_TOKENS, dtype=torch.bool, device=device)
    levels = torch.tensor([0, 1, 2, 0], device=device)
    kp = torch.randn(4, KP_DIM, device=device)
    dims = torch.randn(4, DIM_DIM, device=device)
    with torch.no_grad():
        outputs = {arm: model(patches, masks, levels, kp, dims) for arm, model in models.items()}
        s0_changed = torch.max(
            torch.abs(
                models[ARMS[0]](patches, masks, levels, kp, torch.flip(dims, [0]))
                - outputs[ARMS[0]]
            )
        ).item()
    if any(tuple(value.shape) != (4, 2) for value in outputs.values()) or s0_changed != 0.0:
        raise RunnerError("unit smoke failed")
    return {"passed": True, "s0_dimension_invariance_max_diff": s0_changed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "unit-smoke"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    device = normalize_device(args.device)
    if args.command == "unit-smoke":
        print(json.dumps(unit_smoke(device), indent=2))
        return 0
    if args.command == "train":
        print(json.dumps(train(device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
