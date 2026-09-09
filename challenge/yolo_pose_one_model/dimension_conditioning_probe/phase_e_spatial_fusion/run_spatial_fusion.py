"""Pre-registered frozen-YOLO spatial fusion follow-up.

This runner deliberately lives outside the completed Phase-C directory.  It
has two state-changing commands:

``extract``
    Re-run the frozen primary YOLO read-only, capture a 7x7 neighbourhood at
    the exact final one2one classification source cell, and write a separate
    float16 spatial cache.  Every prediction and the float32 center token must
    exactly reproduce the completed Phase-C cache.

``train``
    Train the four pre-registered tiny spatial probes, write a synthetic-DEV
    source-selection lock, and only then evaluate synthetic TEST and real DEV.

The primary YOLO is never placed in an optimizer and no completed Phase-C
artifact is rewritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NS = REPO_ROOT / "challenge/yolo_pose_one_model/dimension_conditioning_probe"
PHASE_A_DIR = NS / "phase_a_oracle"
PHASE_B_DIR = NS / "phase_b_data"
PHASE_C_DIR = NS / "phase_c_probe"
OUT_DIR = NS / "phase_e_spatial_fusion"

PROTOCOL_PATH = OUT_DIR / "PRE_REGISTERED_PROTOCOL.md"
CONTRACT_PATH = OUT_DIR / "SPATIAL_FUSION_CONTRACT.json"
PROTOCOL_FREEZE_PATH = OUT_DIR / "PROTOCOL_FREEZE.json"
RUNNER_PATH = Path(__file__).resolve()
PHASE_C_RUNNER = PHASE_C_DIR / "run_phase_c.py"
PHASE_C_CACHE = PHASE_C_DIR / "FEATURE_CACHE.npz"
PHASE_C_SUMMARY = PHASE_C_DIR / "PHASE_C_SUMMARY.json"
PHASE_A_CACHE = PHASE_A_DIR / "PREDICTION_CACHE.npz"
PHASE_A_PER_FRAME = PHASE_A_DIR / "ORACLE_PER_FRAME.csv"
SPLIT_MEMBERSHIP = PHASE_B_DIR / "SPLIT_MEMBERSHIP.json"
PHASE_B_METADATA_AUDIT = PHASE_B_DIR / "SYNTH_METADATA_AUDIT.json"
PHASE_B_DIMENSION_DISTRIBUTION = PHASE_B_DIR / "DIMENSION_DISTRIBUTION.json"
DATA_CONTRACT = NS / "DATA_CONTRACT.json"
GEOMETRY_REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
GEOMETRY_REGISTRY_SNAPSHOT = NS / "OBJECT_GEOMETRY_REGISTRY_SNAPSHOT.json"
COMMON128_MANIFEST = (
    REPO_ROOT / "challenge/real_gt_v2/manifests/COMMON_DEV_PLASTIC_POS128.json"
)
COMMON_MULTISHAPE_MANIFEST = (
    REPO_ROOT / "challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json"
)
PRIMARY_WEIGHTS = (
    REPO_ROOT
    / "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
    "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt"
)

SPATIAL_CACHE = OUT_DIR / "SPATIAL_FEATURE_CACHE.npz"
SPATIAL_AUDIT = OUT_DIR / "SPATIAL_FEATURE_AUDIT.json"
SHARD_DIR = OUT_DIR / "spatial_feature_shards"
SELECTION_LOCK = OUT_DIR / "SOURCE_SELECTION_LOCK.json"
REAL_RESULTS = OUT_DIR / "REAL_SPATIAL_FUSION_RESULTS.json"
PNP_RESULTS = OUT_DIR / "SPATIAL_FUSION_PNP_DIAGNOSTIC.json"
CONTROL_RESULTS = OUT_DIR / "SPATIAL_FUSION_CONTROLS.json"
COMPARISON_CSV = OUT_DIR / "SPATIAL_FUSION_COMPARISON.csv"
SUMMARY_PATH = OUT_DIR / "SPATIAL_FUSION_SUMMARY.json"
REPORT_PATH = OUT_DIR / "SPATIAL_FUSION_REPORT.md"

SCHEMA_VERSION = "dimension_conditioning_spatial_fusion_v1"
CACHE_SCHEMA_VERSION = "dimension_conditioning_spatial_feature_cache_v1"
AUDIT_SCHEMA_VERSION = "dimension_conditioning_spatial_feature_audit_v1"
RUN_SCHEMA_VERSION = "dimension_conditioning_spatial_fusion_run_v1"
LOCK_SCHEMA_VERSION = "dimension_conditioning_spatial_source_selection_lock_v1"
CHECKPOINT_SCHEMA_VERSION = "dimension_conditioning_spatial_probe_checkpoint_v1"
SHARD_SCHEMA_VERSION = "dimension_conditioning_spatial_feature_shard_v1"

ARMS = (
    "S0_SPATIAL_NO_DIMS",
    "S1_SPATIAL_CONCAT",
    "S2_SPATIAL_FILM",
    "S3_SPATIAL_CROSS_ATTENTION",
)
DIMENSION_AWARE_ARMS = ARMS[1:]
ARM_PREFERENCE = {arm: index for index, arm in enumerate(DIMENSION_AWARE_ARMS)}
SEEDS = (0, 1, 2)
PATCH_SIZE = 7
PATCH_RADIUS = PATCH_SIZE // 2
PATCH_TOKENS = PATCH_SIZE * PATCH_SIZE
TOKEN_DIM = 64
KP_DIM = 26
DIMENSION_DIM = 4
LEVEL_COUNT = 3
ATTENTION_DIM = 32
ATTENTION_HEADS = 4
FILM_HIDDEN = 80
CLASSIFIER_INPUT_DIM = TOKEN_DIM * 2 + KP_DIM + DIMENSION_DIM
TRAIN_EPOCHS = 40
TRAIN_BATCH = 512
EVAL_BATCH = 1024
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
SHARD_SIZE = 512
EXTRACT_BATCH = 32
SHORT_FRONT = 0
LONG_FRONT = 1
SPLIT_CODE = {"TRAIN": 0, "DEV": 1, "TEST": 2}
STEP_ZERO_TOLERANCE = 1.0e-7
ADAPTER_RELATIVE_DIFFERENCE_MAX = 0.05


class SpatialFusionError(RuntimeError):
    """A pre-registration, provenance, immutability, or shape violation."""


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


def _canonical_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash named dense arrays independent of NPZ container metadata."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise SpatialFusionError(f"OBJECT_ARRAY_FORBIDDEN_IN_CONTENT_HASH: {name}")
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _runtime_binding_snapshot(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract_payload = _read_json(CONTRACT_PATH) if contract is None else contract
    return {
        "repo_head": _repo_head(),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "protocol_freeze_sha256": _sha256(PROTOCOL_FREEZE_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "source_bindings": _verify_source_bindings(contract_payload),
    }


def _assert_runtime_bindings_unchanged(expected: Mapping[str, Any]) -> None:
    observed = _runtime_binding_snapshot()
    if observed != dict(expected):
        raise SpatialFusionError(
            "RUNTIME_BINDINGS_CHANGED_DURING_OPERATION: "
            f"expected={expected} observed={observed}"
        )


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialFusionError(f"JSON_UNREADABLE: {path}: {exc}") from exc


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise SpatialFusionError(f"ARTIFACT_ALREADY_EXISTS: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _publish_temporary_exclusive(temporary: Path, destination: Path) -> None:
    """Atomically expose a fully fsynced inode without clobbering a peer."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise
    directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    _write_text_exclusive(path, _json_text(payload))


def _savez_exclusive(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise SpatialFusionError(f"ARTIFACT_ALREADY_EXISTS: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _torch_save_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise SpatialFusionError(f"ARTIFACT_ALREADY_EXISTS: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _repo_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SpatialFusionError("REPOSITORY_HEAD_UNAVAILABLE") from exc


def _contract() -> dict[str, Any]:
    payload = _read_json(CONTRACT_PATH)
    if (
        payload.get("schema_version")
        != "dimension_conditioning_spatial_fusion_contract_v2_exploratory"
    ):
        raise SpatialFusionError("SPATIAL_FUSION_CONTRACT_SCHEMA_MISMATCH")
    if payload.get("historical_contract_preserved") is not True:
        raise SpatialFusionError("HISTORICAL_PHASE_C_PRESERVATION_NOT_LOCKED")
    if tuple(payload.get("arms", ())) != ARMS:
        raise SpatialFusionError("SPATIAL_FUSION_ARM_CONTRACT_MISMATCH")
    if (
        payload.get("irrevocably_exploratory") is not True
        or payload.get("full_yolo_training_authorized_by_this_contract") is not False
        or payload.get("maximum_possible_verdict")
        != "EXPLORATORY_SPATIAL_FUSION_DIAGNOSTIC_COMPLETE"
    ):
        raise SpatialFusionError("SPATIAL_FUSION_EXPLORATORY_BOUNDARY_MISMATCH")
    if payload.get("repo_head") != _repo_head():
        raise SpatialFusionError("SPATIAL_FUSION_REPO_HEAD_MISMATCH")
    expected_training = {
        "seeds": list(SEEDS),
        "epochs": TRAIN_EPOCHS,
        "batch": TRAIN_BATCH,
        "steps_per_epoch": 40,
        "samples_per_epoch_with_replacement": 20_480,
        "drop_last": False,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "scheduler": "none",
        "gradient_clipping": "none",
        "amp": False,
        "loss": "balanced_cross_entropy",
        "class_balanced_sampler": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "decision_threshold": 0.5,
        "real_used_for_training_or_selection": 0,
    }
    training = payload.get("training", {})
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise SpatialFusionError(
                f"SPATIAL_FUSION_TRAINING_{field.upper()}_MISMATCH"
            )
    expected_population = {
        "synthetic_train_n": 20_281,
        "synthetic_dev_n": 9_624,
        "synthetic_test_n": 10_095,
        "real_historically_opened_before_phase_e": True,
        "synthetic_test_historically_opened_before_phase_e": True,
        "promotion_or_end_to_end_holdout_claim_allowed": False,
    }
    population = payload.get("population", {})
    for field, expected in expected_population.items():
        if population.get(field) != expected:
            raise SpatialFusionError(
                f"SPATIAL_FUSION_POPULATION_{field.upper()}_MISMATCH"
            )
    gates = payload.get("diagnostic_gates", {})
    expected_gate_scalars = {
        "E1_common128_correct_min": 122,
        "E2_night_correct_min": 26,
        "E4_primary_minus_no_dims_correct_frames_min": 7,
        "E5_current_synthetic_test_correct_minus_shuffled_frames_min": 505,
        "E5_current_synthetic_test_n": 10_095,
        "E6_candidate_auc_min": 0.306212109375,
        "E6_oracle_recovery_min": 0.7,
        "E8_prediction_and_float32_center_exact": True,
        "E8_tolerance": 0.0,
        "E9_batch": 1,
        "E9_warmup": 20,
        "E9_iterations": 200,
        "E9_relative_increase_max": 0.05,
    }
    for field, expected in expected_gate_scalars.items():
        if gates.get(field) != expected:
            raise SpatialFusionError(
                f"SPATIAL_FUSION_GATE_{field.upper()}_MISMATCH"
            )
    _verify_protocol_freeze()
    return payload


def _verify_protocol_freeze() -> dict[str, Any]:
    if not PROTOCOL_FREEZE_PATH.is_file():
        raise SpatialFusionError("PROTOCOL_FREEZE_REQUIRED")
    freeze = _read_json(PROTOCOL_FREEZE_PATH)
    if freeze.get("schema_version") != "dimension_conditioning_protocol_freeze_v1":
        raise SpatialFusionError("PROTOCOL_FREEZE_SCHEMA_MISMATCH")
    expected = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "repo_head": _repo_head(),
    }
    for field, observed in expected.items():
        if freeze.get(field) != observed:
            raise SpatialFusionError(
                f"PROTOCOL_FREEZE_{field.upper()}_MISMATCH: "
                f"expected={freeze.get(field)} observed={observed}"
            )
    if freeze.get("phase_e_results_existed_at_freeze") is not False:
        raise SpatialFusionError("PROTOCOL_FREEZE_RESULTS_EXISTED_AT_FREEZE_INVALID")
    test_sha = freeze.get("test_sha256")
    test_path = OUT_DIR / "test_spatial_fusion.py"
    if test_sha is not None and (
        not test_path.is_file() or str(test_sha) != _sha256(test_path)
    ):
        raise SpatialFusionError("PROTOCOL_FREEZE_TEST_SHA256_MISMATCH")
    return freeze


def _verify_source_bindings(contract: Mapping[str, Any]) -> dict[str, str]:
    expected = contract["source_bindings"]
    paths = {
        "primary_yolo_sha256": PRIMARY_WEIGHTS,
        "phase_c_feature_cache_sha256": PHASE_C_CACHE,
        "phase_c_runner_sha256": PHASE_C_RUNNER,
        "phase_c_summary_sha256": PHASE_C_SUMMARY,
        "phase_b_split_sha256": SPLIT_MEMBERSHIP,
        "phase_b_metadata_audit_sha256": PHASE_B_METADATA_AUDIT,
        "phase_b_dimension_distribution_sha256": PHASE_B_DIMENSION_DISTRIBUTION,
        "data_contract_sha256": DATA_CONTRACT,
        "phase_a_prediction_cache_sha256": PHASE_A_CACHE,
        "phase_a_per_frame_sha256": PHASE_A_PER_FRAME,
        "geometry_registry_sha256": GEOMETRY_REGISTRY,
        "geometry_registry_snapshot_sha256": GEOMETRY_REGISTRY_SNAPSHOT,
        "common128_manifest_sha256": COMMON128_MANIFEST,
        "common_multishape_manifest_sha256": COMMON_MULTISHAPE_MANIFEST,
    }
    if set(expected) != set(paths):
        raise SpatialFusionError(
            "BOUND_SOURCE_FIELD_SET_MISMATCH: "
            f"expected={sorted(paths)} observed={sorted(expected)}"
        )
    observed: dict[str, str] = {}
    for field, path in paths.items():
        if not path.is_file():
            raise SpatialFusionError(f"BOUND_SOURCE_MISSING: {field}: {path}")
        observed[field] = _sha256(path)
        if observed[field] != expected.get(field):
            raise SpatialFusionError(
                f"BOUND_SOURCE_SHA_MISMATCH: {field}: "
                f"expected={expected.get(field)} observed={observed[field]}"
            )
    return observed


_PHASE_C_MODULE: Any | None = None


def _phase_c() -> Any:
    global _PHASE_C_MODULE
    if _PHASE_C_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "dimension_conditioning_completed_phase_c", PHASE_C_RUNNER
        )
        if spec is None or spec.loader is None:
            raise SpatialFusionError("PHASE_C_RUNNER_IMPORT_SPEC_FAILED")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PHASE_C_MODULE = module
    return _PHASE_C_MODULE


def _feature_recipe(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "version": "dimension_conditioning_cls_pen_source_patch7_v1",
        "contract_sha256": _sha256(CONTRACT_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "protocol_freeze_sha256": _sha256(PROTOCOL_FREEZE_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "source_bindings": dict(contract["source_bindings"]),
        "candidate": "highest-confidence final detection",
        "tap": "immediate producer of active one2one classification readout",
        "anchor": "exact final source level,row,column",
        "layout": {
            "patch_size": PATCH_SIZE,
            "tokens": PATCH_TOKENS,
            "channels": TOKEN_DIM,
            "token_order": "row-major",
            "channel_order": "classification-penultimate channel order",
            "center_token_index": PATCH_TOKENS // 2,
            "padding": "zero_with_boolean_validity_mask",
            "storage_dtype": "float16",
        },
        "normalization": "per-channel synthetic TRAIN valid patch tokens only",
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _masked_mean_max(tokens: Tensor, mask: Tensor) -> Tensor:
    """Pool valid spatial tokens to masked mean plus masked max."""

    if tokens.ndim != 3 or tokens.shape[1:] != (PATCH_TOKENS, TOKEN_DIM):
        raise SpatialFusionError(
            f"MASKED_POOL_TOKEN_SHAPE_INVALID: {tuple(tokens.shape)}"
        )
    if mask.shape != tokens.shape[:2]:
        raise SpatialFusionError(
            f"MASKED_POOL_MASK_SHAPE_INVALID: {tuple(mask.shape)}"
        )
    mask = mask.to(dtype=torch.bool)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise SpatialFusionError("MASKED_POOL_EMPTY_ROW")
    expanded = mask.unsqueeze(-1)
    mean = (tokens * expanded.to(tokens.dtype)).sum(dim=1) / counts.unsqueeze(1).to(
        tokens.dtype
    )
    minimum = torch.finfo(tokens.dtype).min
    maximum = tokens.masked_fill(~expanded, minimum).max(dim=1).values
    return torch.cat([mean, maximum], dim=1)


class SpatialFusionProbe(nn.Module):
    """Shared spatial probe with one of the four frozen fusion contracts."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise SpatialFusionError(f"UNKNOWN_SPATIAL_FUSION_ARM: {arm}")
        self.arm = arm

        # All common modules are created before any adapter.  Resetting the
        # torch seed before constructing each arm therefore makes S1/S2/S3
        # common initialization byte-identical for a given seed.
        self.token_encoder = nn.Sequential(nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.SiLU())
        self.position_embedding = nn.Parameter(torch.empty(PATCH_TOKENS, TOKEN_DIM))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.source_level_embedding = nn.Embedding(LEVEL_COUNT, TOKEN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(CLASSIFIER_INPUT_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

        self.film_adapter: nn.Module | None = None
        self.attention_token_projection: nn.Module | None = None
        self.attention_query_projection: nn.Module | None = None
        self.attention: nn.Module | None = None
        self.attention_residual: nn.Module | None = None
        if arm == "S2_SPATIAL_FILM":
            self.film_adapter = nn.Sequential(
                nn.Linear(DIMENSION_DIM, FILM_HIDDEN),
                nn.SiLU(),
                nn.Linear(FILM_HIDDEN, 2 * TOKEN_DIM),
            )
            final = self.film_adapter[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        elif arm == "S3_SPATIAL_CROSS_ATTENTION":
            self.attention_token_projection = nn.Linear(TOKEN_DIM, ATTENTION_DIM)
            self.attention_query_projection = nn.Linear(DIMENSION_DIM, ATTENTION_DIM)
            self.attention = nn.MultiheadAttention(
                ATTENTION_DIM,
                ATTENTION_HEADS,
                dropout=0.0,
                batch_first=True,
            )
            self.attention_residual = nn.Linear(ATTENTION_DIM, 2 * TOKEN_DIM)
            nn.init.zeros_(self.attention_residual.weight)
            nn.init.zeros_(self.attention_residual.bias)

    def common_state_dict(self) -> dict[str, Tensor]:
        prefixes = (
            "token_encoder.",
            "position_embedding",
            "source_level_embedding.",
            "classifier.",
        )
        return {
            key: value
            for key, value in self.state_dict().items()
            if any(key == prefix or key.startswith(prefix) for prefix in prefixes)
        }

    def forward(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        source_levels: Tensor,
        kp_features: Tensor,
        dimension_features: Tensor,
    ) -> Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (PATCH_TOKENS, TOKEN_DIM):
            raise SpatialFusionError(f"MODEL_PATCH_SHAPE_INVALID: {tuple(patches.shape)}")
        batch = patches.shape[0]
        expected = {
            "patch_mask": (batch, PATCH_TOKENS),
            "source_levels": (batch,),
            "kp_features": (batch, KP_DIM),
            "dimension_features": (batch, DIMENSION_DIM),
        }
        observed = {
            "patch_mask": tuple(patch_mask.shape),
            "source_levels": tuple(source_levels.shape),
            "kp_features": tuple(kp_features.shape),
            "dimension_features": tuple(dimension_features.shape),
        }
        for name, shape in expected.items():
            if observed[name] != shape:
                raise SpatialFusionError(
                    f"MODEL_{name.upper()}_SHAPE_INVALID: expected={shape} "
                    f"observed={observed[name]}"
                )
        if torch.any((source_levels < 0) | (source_levels >= LEVEL_COUNT)):
            raise SpatialFusionError("MODEL_SOURCE_LEVEL_INVALID")
        mask = patch_mask.to(dtype=torch.bool)
        tokens = self.token_encoder(patches)
        tokens = tokens + self.position_embedding.unsqueeze(0)
        tokens = tokens + self.source_level_embedding(source_levels.long()).unsqueeze(1)

        if self.arm == "S2_SPATIAL_FILM":
            assert self.film_adapter is not None
            gamma, beta = self.film_adapter(dimension_features).chunk(2, dim=1)
            tokens = tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        pooled = _masked_mean_max(tokens, mask)
        if self.arm == "S3_SPATIAL_CROSS_ATTENTION":
            assert self.attention_token_projection is not None
            assert self.attention_query_projection is not None
            assert self.attention is not None
            assert self.attention_residual is not None
            key_value = self.attention_token_projection(tokens)
            query = self.attention_query_projection(dimension_features).unsqueeze(1)
            attended, _weights = self.attention(
                query,
                key_value,
                key_value,
                key_padding_mask=~mask,
                need_weights=False,
            )
            pooled = pooled + self.attention_residual(attended[:, 0, :])

        late_dimensions = (
            torch.zeros_like(dimension_features)
            if self.arm == "S0_SPATIAL_NO_DIMS"
            else dimension_features
        )
        classifier_input = torch.cat([pooled, kp_features, late_dimensions], dim=1)
        return self.classifier(classifier_input)


def _adapter_parameter_count(model: SpatialFusionProbe) -> int:
    if model.arm == "S2_SPATIAL_FILM":
        assert model.film_adapter is not None
        modules: tuple[nn.Module, ...] = (model.film_adapter,)
    elif model.arm == "S3_SPATIAL_CROSS_ATTENTION":
        assert model.attention_token_projection is not None
        assert model.attention_query_projection is not None
        assert model.attention is not None
        assert model.attention_residual is not None
        modules = (
            model.attention_token_projection,
            model.attention_query_projection,
            model.attention,
            model.attention_residual,
        )
    else:
        return 0
    return int(sum(parameter.numel() for module in modules for parameter in module.parameters()))


def _step_zero_audit(
    models: Mapping[str, SpatialFusionProbe],
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    tolerance: float = STEP_ZERO_TOLERANCE,
) -> dict[str, Any]:
    required = set(DIMENSION_AWARE_ARMS)
    if not required.issubset(models):
        raise SpatialFusionError(
            f"STEP_ZERO_MODELS_MISSING: {sorted(required - set(models))}"
        )
    common_reference = models["S1_SPATIAL_CONCAT"].common_state_dict()
    common_exact: dict[str, bool] = {}
    for arm in DIMENSION_AWARE_ARMS:
        observed = models[arm].common_state_dict()
        common_exact[arm] = bool(
            observed.keys() == common_reference.keys()
            and all(torch.equal(observed[key], common_reference[key]) for key in observed)
        )
    previous = {arm: model.training for arm, model in models.items()}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        logits = {arm: models[arm](*batch).detach().cpu() for arm in DIMENSION_AWARE_ARMS}
    for arm, state in previous.items():
        models[arm].train(state)
    differences: dict[str, float] = {}
    for first, second in (
        ("S1_SPATIAL_CONCAT", "S2_SPATIAL_FILM"),
        ("S1_SPATIAL_CONCAT", "S3_SPATIAL_CROSS_ATTENTION"),
        ("S2_SPATIAL_FILM", "S3_SPATIAL_CROSS_ATTENTION"),
    ):
        differences[f"{first}__{second}"] = float(
            torch.max(torch.abs(logits[first] - logits[second])).item()
        )
    film_count = _adapter_parameter_count(models["S2_SPATIAL_FILM"])
    attention_count = _adapter_parameter_count(
        models["S3_SPATIAL_CROSS_ATTENTION"]
    )
    relative = abs(film_count - attention_count) / max(film_count, attention_count)
    passed = bool(
        all(common_exact.values())
        and max(differences.values()) <= tolerance
        and relative <= ADAPTER_RELATIVE_DIFFERENCE_MAX
    )
    payload = {
        "common_state_exact": common_exact,
        "logit_max_abs_diff": differences,
        "required_logit_tolerance": tolerance,
        "adapter_parameter_counts": {
            "S2_SPATIAL_FILM": film_count,
            "S3_SPATIAL_CROSS_ATTENTION": attention_count,
        },
        "adapter_parameter_relative_difference": relative,
        "adapter_parameter_relative_difference_max": ADAPTER_RELATIVE_DIFFERENCE_MAX,
        "passed": passed,
    }
    if not passed:
        raise SpatialFusionError(f"STEP_ZERO_CONTRACT_FAILED: {payload}")
    return payload


def _shuffled_dimensions(
    dimensions: np.ndarray,
    *,
    seed: int = 42,
    require_all_triplets_changed: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Return a deterministic value-derangement when one exists."""

    values = np.asarray(dimensions)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise SpatialFusionError("SHUFFLE_DIMENSIONS_SHAPE_INVALID")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(values), dtype=np.int64)
    if not require_all_triplets_changed:
        # Preserve Phase-C's registered seed-42 control exactly for the
        # COMMON128+Wood45 real union.  That two-value population cannot have
        # every triplet changed, so only source-frame fixed points are repaired.
        permutation = rng.permutation(len(values))
        fixed = np.flatnonzero(permutation == indices)
        if len(fixed) == 1:
            other = 0 if int(fixed[0]) != 0 else 1
            permutation[fixed[0]], permutation[other] = (
                permutation[other],
                permutation[fixed[0]],
            )
        elif len(fixed) > 1:
            permutation[fixed] = np.roll(permutation[fixed], 1)
        return values[permutation], permutation.tolist()
    candidate: np.ndarray | None = None
    for _attempt in range(1024):
        permutation = rng.permutation(len(values))
        changed = np.any(values[permutation] != values, axis=1)
        if bool(changed.all()):
            candidate = permutation
            break
    if candidate is None:
        order = rng.permutation(len(values))
        for shift in range(1, len(values)):
            permutation = np.empty_like(order)
            permutation[order] = np.roll(order, shift)
            if bool(np.all(np.any(values[permutation] != values, axis=1))):
                candidate = permutation
                break
    if candidate is None:
        raise SpatialFusionError("SHUFFLE_100_PERCENT_VALUE_DERANGEMENT_IMPOSSIBLE")
    shuffled = values[candidate]
    if require_all_triplets_changed and not bool(
        np.all(np.any(shuffled != values, axis=1))
    ):
        raise SpatialFusionError("SHUFFLE_100_PERCENT_TRIPLET_CHANGE_FAILED")
    return shuffled, candidate.tolist()


def _patch_from_source_cell(
    tap: Any,
    vector: Mapping[str, Any] | None,
    batch_index: int,
) -> dict[str, Any]:
    if vector is None:
        return {
            "patch": np.zeros((PATCH_TOKENS, TOKEN_DIM), dtype=np.float16),
            "mask": np.zeros(PATCH_TOKENS, dtype=np.bool_),
            "row": -1,
            "column": -1,
            "level": -1,
            "patch_center_vector_diff": 0.0,
            "patch_center_vector_exact": True,
            "center_float16_quantization_diff": 0.0,
        }
    flat = int(vector["flat"])
    level, row, column = tap._decode(flat)
    if int(vector["level"]) != level:
        raise SpatialFusionError("PATCH_VECTOR_LEVEL_DECODE_MISMATCH")
    feature_map = tap.cap[level][batch_index].detach().float()
    if feature_map.ndim != 3 or int(feature_map.shape[0]) != TOKEN_DIM:
        raise SpatialFusionError(
            f"PATCH_FEATURE_MAP_SHAPE_INVALID: {tuple(feature_map.shape)}"
        )
    height, width = int(feature_map.shape[1]), int(feature_map.shape[2])
    padded = functional.pad(
        feature_map,
        (PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS),
        mode="constant",
        value=0.0,
    )
    patch_tensor = padded[
        :, row : row + PATCH_SIZE, column : column + PATCH_SIZE
    ]
    if tuple(patch_tensor.shape) != (TOKEN_DIM, PATCH_SIZE, PATCH_SIZE):
        raise SpatialFusionError(
            f"PATCH_TENSOR_SHAPE_INVALID: {tuple(patch_tensor.shape)}"
        )
    patch_float32 = (
        patch_tensor.permute(1, 2, 0)
        .contiguous()
        .reshape(PATCH_TOKENS, TOKEN_DIM)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    rows = np.arange(row - PATCH_RADIUS, row + PATCH_RADIUS + 1)
    columns = np.arange(column - PATCH_RADIUS, column + PATCH_RADIUS + 1)
    mask = (
        (rows[:, None] >= 0)
        & (rows[:, None] < height)
        & (columns[None, :] >= 0)
        & (columns[None, :] < width)
    ).reshape(PATCH_TOKENS)
    center = patch_float32[PATCH_TOKENS // 2]
    vector_feature = np.asarray(vector["feature"], dtype=np.float32)
    if vector_feature.shape != (TOKEN_DIM,):
        raise SpatialFusionError("PATCH_VECTOR_FEATURE_SHAPE_INVALID")
    center_diff = float(np.max(np.abs(center - vector_feature)))
    center_exact = bool(np.array_equal(center, vector_feature))
    if not center_exact:
        raise SpatialFusionError(
            f"PATCH_CENTER_CURRENT_VECTOR_MISMATCH: max_abs_diff={center_diff}"
        )
    if not bool(mask[PATCH_TOKENS // 2]):
        raise SpatialFusionError("PATCH_CENTER_TOKEN_MASKED")
    stored_patch = patch_float32.astype(np.float16)
    quantization_diff = float(
        np.max(
            np.abs(
                stored_patch[PATCH_TOKENS // 2].astype(np.float32)
                - vector_feature
            )
        )
    )
    return {
        "patch": stored_patch,
        "mask": mask.astype(np.bool_),
        "row": row,
        "column": column,
        "level": level,
        "patch_center_vector_diff": center_diff,
        "patch_center_vector_exact": center_exact,
        "center_float16_quantization_diff": quantization_diff,
    }


def _prediction_and_center_audit(
    *,
    base: Any,
    result: Any,
    vector: Mapping[str, Any] | None,
    raw_shape: tuple[int, int],
    tap: Any,
    batch_index: int,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    extracted = base._extract_result(result, vector, raw_shape)
    patch = _patch_from_source_cell(tap, vector, batch_index)
    expected_valid = bool(expected["valid"])
    valid_match = bool(extracted["valid"]) == expected_valid
    candidate_diff = abs(
        int(extracted["detection_count"]) - int(expected["detection_count"])
    )
    confidence_diff = 0.0
    box_diff = 0.0
    keypoint_diff = 0.0
    center_cache_diff = 0.0
    center_cache_exact = True
    level_match = int(patch["level"]) == int(expected["source_level"])
    if expected_valid and bool(extracted["valid"]):
        observed_confidence = np.float32(extracted["confidence"])
        expected_confidence = np.float32(expected["confidence"])
        confidence_diff = float(abs(observed_confidence - expected_confidence))
        observed_box = np.asarray(extracted["box"], dtype=np.float32)
        expected_box = np.asarray(expected["box"], dtype=np.float32)
        observed_keypoints = np.asarray(extracted["keypoints"], dtype=np.float32)
        expected_keypoints = np.asarray(expected["keypoints"], dtype=np.float32)
        box_diff = float(np.max(np.abs(observed_box - expected_box)))
        keypoint_diff = float(
            np.max(np.abs(observed_keypoints - expected_keypoints))
        )
        center_before_storage = np.asarray(vector["feature"], dtype=np.float32)
        expected_center = np.asarray(expected["image_feature"], dtype=np.float32)
        center_cache_diff = float(
            np.max(np.abs(center_before_storage - expected_center))
        )
        center_cache_exact = bool(
            np.array_equal(center_before_storage, expected_center)
        )
    elif expected_valid != bool(extracted["valid"]):
        confidence_diff = math.inf
        box_diff = math.inf
        keypoint_diff = math.inf
        center_cache_diff = math.inf
        center_cache_exact = False
    prediction_exact = bool(
        valid_match
        and candidate_diff == 0
        and confidence_diff == 0.0
        and box_diff == 0.0
        and keypoint_diff == 0.0
        and level_match
    )
    if not prediction_exact or not center_cache_exact:
        raise SpatialFusionError(
            "FROZEN_PHASE_C_REPRODUCTION_FAILED: "
            f"prediction_exact={prediction_exact} center_exact={center_cache_exact} "
            f"candidate={candidate_diff} confidence={confidence_diff} box={box_diff} "
            f"keypoints={keypoint_diff} center={center_cache_diff} "
            f"level_match={level_match}"
        )
    return {
        **patch,
        "prediction_exact": prediction_exact,
        "center_cache_exact": center_cache_exact,
        "valid_match": valid_match,
        "source_level_match": level_match,
        "candidate_count_diff": candidate_diff,
        "confidence_diff": confidence_diff,
        "box_diff": box_diff,
        "keypoint_diff": keypoint_diff,
        "center_cache_diff": center_cache_diff,
    }


def _expected_row(cache: Mapping[str, Any], prefix: str, index: int) -> dict[str, Any]:
    return {
        "valid": bool(cache[f"{prefix}_valid"][index]),
        "confidence": cache[f"{prefix}_confidences"][index],
        "box": cache[f"{prefix}_boxes_xyxy"][index],
        "keypoints": cache[f"{prefix}_keypoints_xy"][index],
        "image_feature": cache[f"{prefix}_image_features"][index],
        "source_level": cache[f"{prefix}_source_levels"][index],
        "detection_count": cache[f"{prefix}_detection_counts"][index],
    }


def _spatial_shard_fingerprint(
    recipe_sha256: str,
    population: str,
    indices: np.ndarray,
    cache: Mapping[str, Any],
) -> str:
    prefix = "synth" if population == "synth" else "real"
    digest = hashlib.sha256(recipe_sha256.encode("ascii"))
    digest.update(population.encode("ascii"))
    contiguous = np.ascontiguousarray(indices, dtype=np.int32)
    digest.update(contiguous.tobytes())
    for field in (
        "frame_ids",
        "valid",
        "confidences",
        "boxes_xyxy",
        "keypoints_xy",
        "image_features",
        "source_levels",
        "detection_counts",
    ):
        values = np.asarray(cache[f"{prefix}_{field}"])[contiguous]
        if values.dtype.kind in {"U", "S"}:
            for value in values.tolist():
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\0")
        else:
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _spatial_shard_matches(
    path: Path,
    expected_fingerprint: str,
    expected_indices: np.ndarray,
    expected_valid: np.ndarray,
    expected_source_levels: np.ndarray,
) -> bool:
    try:
        _read_validated_spatial_shard(
            path,
            expected_fingerprint=expected_fingerprint,
            expected_indices=expected_indices,
            expected_valid=expected_valid,
            expected_source_levels=expected_source_levels,
        )
        return True
    except (OSError, ValueError, KeyError, SpatialFusionError):
        return False


def _validate_spatial_population_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_valid: np.ndarray,
    expected_source_levels: np.ndarray,
    context: str,
) -> None:
    valid = np.asarray(expected_valid, dtype=np.bool_)
    levels = np.asarray(expected_source_levels, dtype=np.int8)
    count = len(valid)
    exact_shapes_and_dtypes = {
        "patches": ((count, PATCH_TOKENS, TOKEN_DIM), np.dtype(np.float16)),
        "patch_masks": ((count, PATCH_TOKENS), np.dtype(np.bool_)),
        "source_rows": ((count,), np.dtype(np.int16)),
        "source_columns": ((count,), np.dtype(np.int16)),
        "source_levels": ((count,), np.dtype(np.int8)),
        "prediction_exact": ((count,), np.dtype(np.bool_)),
        "center_exact": ((count,), np.dtype(np.bool_)),
        "source_level_match": ((count,), np.dtype(np.bool_)),
        "candidate_count_diffs": ((count,), np.dtype(np.int16)),
        "confidence_diffs": ((count,), np.dtype(np.float64)),
        "box_diffs": ((count,), np.dtype(np.float64)),
        "keypoint_diffs": ((count,), np.dtype(np.float64)),
        "center_diffs": ((count,), np.dtype(np.float64)),
        "center_float16_quantization_diffs": (
            (count,),
            np.dtype(np.float64),
        ),
    }
    for name, (shape, dtype) in exact_shapes_and_dtypes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != dtype:
            raise SpatialFusionError(
                f"{context}_{name.upper()}_SHAPE_OR_DTYPE_INVALID: "
                f"shape={value.shape} dtype={value.dtype}"
            )
    patches = np.asarray(arrays["patches"])
    masks = np.asarray(arrays["patch_masks"])
    if not bool(np.isfinite(patches).all()):
        raise SpatialFusionError(f"{context}_PATCH_NONFINITE")
    for name in (
        "confidence_diffs",
        "box_diffs",
        "keypoint_diffs",
        "center_diffs",
        "center_float16_quantization_diffs",
    ):
        if not bool(np.isfinite(arrays[name]).all()):
            raise SpatialFusionError(f"{context}_{name.upper()}_NONFINITE")
    if not np.array_equal(np.any(masks, axis=1), valid):
        raise SpatialFusionError(f"{context}_MASK_VALIDITY_MISMATCH")
    if bool(valid.any()) and not bool(masks[valid, PATCH_TOKENS // 2].all()):
        raise SpatialFusionError(f"{context}_VALID_CENTER_MASK_FALSE")
    if not bool(np.all(patches[~masks] == np.float16(0.0))):
        raise SpatialFusionError(f"{context}_PADDED_PATCH_VALUE_NONZERO")
    if bool((~valid).any()) and not bool(np.all(patches[~valid] == 0)):
        raise SpatialFusionError(f"{context}_INVALID_ROW_PATCH_NONZERO")
    observed_levels = np.asarray(arrays["source_levels"], dtype=np.int8)
    if not np.array_equal(observed_levels, levels):
        raise SpatialFusionError(f"{context}_SOURCE_LEVEL_BINDING_MISMATCH")
    rows = np.asarray(arrays["source_rows"], dtype=np.int16)
    columns = np.asarray(arrays["source_columns"], dtype=np.int16)
    if bool(valid.any()) and (
        np.any((observed_levels[valid] < 0) | (observed_levels[valid] >= LEVEL_COUNT))
        or np.any(rows[valid] < 0)
        or np.any(columns[valid] < 0)
    ):
        raise SpatialFusionError(f"{context}_VALID_SOURCE_COORDINATE_INVALID")
    if bool((~valid).any()) and not bool(
        np.all(observed_levels[~valid] == -1)
        and np.all(rows[~valid] == -1)
        and np.all(columns[~valid] == -1)
    ):
        raise SpatialFusionError(f"{context}_INVALID_SOURCE_SENTINEL_INVALID")
    if not bool(
        np.asarray(arrays["prediction_exact"]).all()
        and np.asarray(arrays["center_exact"]).all()
        and np.asarray(arrays["source_level_match"]).all()
    ):
        raise SpatialFusionError(f"{context}_EXACTNESS_FLAG_FALSE")
    if not bool(
        np.all(arrays["candidate_count_diffs"] == 0)
        and np.all(arrays["confidence_diffs"] == 0.0)
        and np.all(arrays["box_diffs"] == 0.0)
        and np.all(arrays["keypoint_diffs"] == 0.0)
        and np.all(arrays["center_diffs"] == 0.0)
        and np.all(arrays["center_float16_quantization_diffs"] >= 0.0)
    ):
        raise SpatialFusionError(f"{context}_PROVENANCE_DIFF_INVALID")


def _read_validated_spatial_shard(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_indices: np.ndarray,
    expected_valid: np.ndarray,
    expected_source_levels: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    data_fields = {
        "fingerprint_sha256",
        "indices",
        "patches",
        "patch_masks",
        "source_rows",
        "source_columns",
        "source_levels",
        "prediction_exact",
        "center_exact",
        "source_level_match",
        "candidate_count_diffs",
        "confidence_diffs",
        "box_diffs",
        "keypoint_diffs",
        "center_diffs",
        "center_float16_quantization_diffs",
    }
    if set(arrays) != data_fields | {"metadata_json"}:
        raise SpatialFusionError(f"SPATIAL_SHARD_FIELD_SET_INVALID: {path}")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if (
        metadata.get("schema_version") != SHARD_SCHEMA_VERSION
        or metadata.get("fingerprint_sha256") != expected_fingerprint
        or str(arrays["fingerprint_sha256"].item()) != expected_fingerprint
    ):
        raise SpatialFusionError(f"SPATIAL_SHARD_METADATA_INVALID: {path}")
    expected_indices = np.asarray(expected_indices, dtype=np.int32)
    if arrays["indices"].dtype != np.int32 or not np.array_equal(
        arrays["indices"], expected_indices
    ):
        raise SpatialFusionError(f"SPATIAL_SHARD_INDICES_MISMATCH: {path}")
    content = {name: arrays[name] for name in data_fields}
    observed_content_sha = _array_content_sha256(content)
    if metadata.get("array_content_sha256") != observed_content_sha:
        raise SpatialFusionError(f"SPATIAL_SHARD_CONTENT_SHA_MISMATCH: {path}")
    _validate_spatial_population_arrays(
        arrays,
        expected_valid=np.asarray(expected_valid, dtype=np.bool_),
        expected_source_levels=np.asarray(expected_source_levels, dtype=np.int8),
        context="SPATIAL_SHARD",
    )
    return arrays


def _save_spatial_shard(
    path: Path,
    indices: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    contract_sha256: str,
) -> None:
    arrays = {
        "fingerprint_sha256": np.asarray(contract_sha256),
        "indices": np.asarray(indices, dtype=np.int32),
        "patches": np.asarray([row["patch"] for row in rows], dtype=np.float16),
        "patch_masks": np.asarray([row["mask"] for row in rows], dtype=np.bool_),
        "source_rows": np.asarray([row["row"] for row in rows], dtype=np.int16),
        "source_columns": np.asarray(
            [row["column"] for row in rows], dtype=np.int16
        ),
        "source_levels": np.asarray([row["level"] for row in rows], dtype=np.int8),
        "prediction_exact": np.asarray(
            [row["prediction_exact"] for row in rows], dtype=np.bool_
        ),
        "center_exact": np.asarray(
            [
                bool(row["patch_center_vector_exact"])
                and bool(row["center_cache_exact"])
                for row in rows
            ],
            dtype=np.bool_,
        ),
        "source_level_match": np.asarray(
            [row["source_level_match"] for row in rows], dtype=np.bool_
        ),
        "candidate_count_diffs": np.asarray(
            [row["candidate_count_diff"] for row in rows], dtype=np.int16
        ),
        "confidence_diffs": np.asarray(
            [row["confidence_diff"] for row in rows], dtype=np.float64
        ),
        "box_diffs": np.asarray([row["box_diff"] for row in rows], dtype=np.float64),
        "keypoint_diffs": np.asarray(
            [row["keypoint_diff"] for row in rows], dtype=np.float64
        ),
        "center_diffs": np.asarray(
            [
                max(
                    float(row["patch_center_vector_diff"]),
                    float(row["center_cache_diff"]),
                )
                for row in rows
            ],
            dtype=np.float64,
        ),
        "center_float16_quantization_diffs": np.asarray(
            [row["center_float16_quantization_diff"] for row in rows],
            dtype=np.float64,
        ),
    }
    metadata = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "fingerprint_sha256": contract_sha256,
        "array_content_sha256": _array_content_sha256(arrays),
        "row_count": len(rows),
        "first_index": int(indices[0]) if len(indices) else None,
        "last_index": int(indices[-1]) if len(indices) else None,
    }
    _savez_exclusive(
        path, metadata_json=np.asarray(_json_text(metadata)), **arrays
    )


def _load_spatial_shards(
    paths: Sequence[Path],
    count: int,
    contracts: Mapping[Path, str],
    expected_valid: np.ndarray,
    expected_source_levels: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {
        "patches": np.zeros((count, PATCH_TOKENS, TOKEN_DIM), dtype=np.float16),
        "patch_masks": np.zeros((count, PATCH_TOKENS), dtype=np.bool_),
        "source_rows": np.full(count, -1, dtype=np.int16),
        "source_columns": np.full(count, -1, dtype=np.int16),
        "source_levels": np.full(count, -1, dtype=np.int8),
        "prediction_exact": np.zeros(count, dtype=np.bool_),
        "center_exact": np.zeros(count, dtype=np.bool_),
        "source_level_match": np.zeros(count, dtype=np.bool_),
        "candidate_count_diffs": np.full(count, -1, dtype=np.int16),
        "confidence_diffs": np.full(count, np.inf, dtype=np.float64),
        "box_diffs": np.full(count, np.inf, dtype=np.float64),
        "keypoint_diffs": np.full(count, np.inf, dtype=np.float64),
        "center_diffs": np.full(count, np.inf, dtype=np.float64),
        "center_float16_quantization_diffs": np.full(
            count, np.inf, dtype=np.float64
        ),
    }
    seen = np.zeros(count, dtype=np.bool_)
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            raw_indices = np.asarray(source["indices"])
        if raw_indices.dtype != np.int32:
            raise SpatialFusionError(f"SPATIAL_SHARD_INDEX_DTYPE_INVALID: {path}")
        indices = raw_indices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= count) or np.any(seen[indices]):
            raise SpatialFusionError(f"SPATIAL_SHARD_INDEX_INVALID: {path}")
        source = _read_validated_spatial_shard(
            path,
            expected_fingerprint=contracts[path],
            expected_indices=raw_indices,
            expected_valid=np.asarray(expected_valid)[indices],
            expected_source_levels=np.asarray(expected_source_levels)[indices],
        )
        seen[indices] = True
        for name in output:
            output[name][indices] = source[name]
    if not bool(seen.all()):
        raise SpatialFusionError(
            f"SPATIAL_SHARDS_INCOMPLETE: missing={int((~seen).sum())}"
        )
    _validate_spatial_population_arrays(
        output,
        expected_valid=np.asarray(expected_valid),
        expected_source_levels=np.asarray(expected_source_levels),
        context="ASSEMBLED_SPATIAL_SHARDS",
    )
    return output


def _patch_normalization(
    patches: np.ndarray, masks: np.ndarray, train_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    channel_sum = np.zeros(TOKEN_DIM, dtype=np.float64)
    channel_square_sum = np.zeros(TOKEN_DIM, dtype=np.float64)
    token_count = 0
    selected_indices = np.flatnonzero(train_rows)
    for start in range(0, len(selected_indices), 512):
        indices = selected_indices[start : start + 512]
        values = np.asarray(patches[indices], dtype=np.float64)
        mask = np.asarray(masks[indices], dtype=np.bool_)
        valid_values = values[mask]
        channel_sum += valid_values.sum(axis=0)
        channel_square_sum += np.square(valid_values).sum(axis=0)
        token_count += len(valid_values)
    if token_count == 0:
        raise SpatialFusionError("PATCH_NORMALIZATION_NO_VALID_TRAIN_TOKENS")
    mean = channel_sum / token_count
    variance = np.maximum(channel_square_sum / token_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1.0e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), token_count


def _extraction_audit(
    synth: Mapping[str, np.ndarray], real: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    combined = {
        name: np.concatenate([synth[name], real[name]])
        for name in (
            "prediction_exact",
            "center_exact",
            "source_level_match",
            "candidate_count_diffs",
            "confidence_diffs",
            "box_diffs",
            "keypoint_diffs",
            "center_diffs",
            "center_float16_quantization_diffs",
        )
    }
    max_diff = {
        "candidate_count": int(np.max(combined["candidate_count_diffs"])),
        "confidence": float(np.max(combined["confidence_diffs"])),
        "box": float(np.max(combined["box_diffs"])),
        "keypoints": float(np.max(combined["keypoint_diffs"])),
        "center_token_before_float16": float(np.max(combined["center_diffs"])),
    }
    passed = bool(
        combined["prediction_exact"].all()
        and combined["center_exact"].all()
        and combined["source_level_match"].all()
        and all(value == 0 for value in max_diff.values())
    )
    return {
        "n": int(len(combined["prediction_exact"])),
        "synthetic_n": int(len(synth["prediction_exact"])),
        "real_n": int(len(real["prediction_exact"])),
        "max_abs_diff": max_diff,
        "prediction_exact_n": int(combined["prediction_exact"].sum()),
        "center_exact_n": int(combined["center_exact"].sum()),
        "source_level_exact_n": int(combined["source_level_match"].sum()),
        "required_tolerance": 0.0,
        "float16_center_quantization_max_abs_diff_report_only": float(
            np.max(combined["center_float16_quantization_diffs"])
        ),
        "passed": passed,
    }


def extract_spatial_features(device: str) -> dict[str, Any]:
    if SPATIAL_CACHE.exists() or SPATIAL_AUDIT.exists():
        raise SpatialFusionError(
            "SPATIAL_EXTRACTION_FINAL_ARTIFACT_ALREADY_EXISTS: "
            f"cache={SPATIAL_CACHE.exists()} audit={SPATIAL_AUDIT.exists()}"
        )
    contract = _contract()
    bindings = _verify_source_bindings(contract)
    runtime_bindings_before = _runtime_binding_snapshot(contract)
    recipe = _feature_recipe(contract)
    base = _phase_c()
    cache = base._load_feature_cache()
    synth_contract = base._load_synth_contract()
    for base_field, source_field in (
        ("synth_frame_ids", "frame_ids"),
        ("synth_splits", "splits"),
        ("synth_dimensions_xyz", "dimensions_xyz"),
        ("synth_labels", "labels"),
        ("synth_raw_shapes", "raw_shapes"),
    ):
        if not np.array_equal(cache[base_field], synth_contract[source_field]):
            raise SpatialFusionError(
                f"PHASE_C_CACHE_SYNTH_CONTRACT_MISMATCH: {base_field}"
            )
    if _sha256(PRIMARY_WEIGHTS) != bindings["primary_yolo_sha256"]:
        raise SpatialFusionError("PRIMARY_CHECKPOINT_SHA_MISMATCH")
    checkpoint_mtime_before = PRIMARY_WEIGHTS.stat().st_mtime_ns
    tap = base.DynamicOne2OneTap(PRIMARY_WEIGHTS, device)
    started = time.perf_counter()
    synth_paths: list[Path] = []
    synth_contracts: dict[Path, str] = {}
    try:
        by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, raw_shape in enumerate(synth_contract["raw_shapes"].tolist()):
            by_shape[tuple(int(value) for value in raw_shape)].append(index)
        processed = 0
        for (height, width), shape_indices in sorted(by_shape.items()):
            for shard_start in range(0, len(shape_indices), SHARD_SIZE):
                indices = np.asarray(
                    shape_indices[shard_start : shard_start + SHARD_SIZE],
                    dtype=np.int32,
                )
                path = SHARD_DIR / (
                    f"synth_{height}x{width}_{shard_start:05d}_"
                    f"{shard_start + len(indices):05d}.npz"
                )
                fingerprint = _spatial_shard_fingerprint(
                    str(recipe["sha256"]), "synth", indices, cache
                )
                synth_paths.append(path)
                synth_contracts[path] = fingerprint
                if path.is_file():
                    if not _spatial_shard_matches(
                        path,
                        fingerprint,
                        indices,
                        np.asarray(cache["synth_valid"])[indices],
                        np.asarray(cache["synth_source_levels"])[indices],
                    ):
                        raise SpatialFusionError(
                            f"EXISTING_SPATIAL_SHARD_MISMATCH: {path}. "
                            "This runner will not overwrite it. After review, "
                            f"remove only this owned shard with: rm -- {shlex.quote(str(path))}"
                        )
                    processed += len(indices)
                    continue
                rows: list[dict[str, Any]] = []
                for offset in range(0, len(indices), EXTRACT_BATCH):
                    batch_indices = indices[offset : offset + EXTRACT_BATCH]
                    images: list[np.ndarray] = []
                    raw_shapes: list[tuple[int, int]] = []
                    for index in batch_indices.tolist():
                        image, raw_shape = base._read_padded(
                            Path(str(synth_contract["images"][index]))
                        )
                        if raw_shape != (height, width):
                            raise SpatialFusionError(
                                "SYNTH_METADATA_IMAGE_SHAPE_MISMATCH"
                            )
                        images.append(image)
                        raw_shapes.append(raw_shape)
                    results, vectors = tap.predict_batch(images)
                    for local_index, (index, result, vector, raw_shape) in enumerate(
                        zip(
                            batch_indices.tolist(),
                            results,
                            vectors,
                            raw_shapes,
                        )
                    ):
                        rows.append(
                            _prediction_and_center_audit(
                                base=base,
                                result=result,
                                vector=vector,
                                raw_shape=raw_shape,
                                tap=tap,
                                batch_index=local_index,
                                expected=_expected_row(cache, "synth", index),
                            )
                        )
                _save_spatial_shard(path, indices, rows, fingerprint)
                processed += len(indices)
                print(
                    f"[{time.strftime('%H:%M:%S')}] spatial synth "
                    f"{processed}/40000",
                    flush=True,
                )

        synth_spatial = _load_spatial_shards(
            synth_paths,
            len(cache["synth_frame_ids"]),
            synth_contracts,
            np.asarray(cache["synth_valid"]),
            np.asarray(cache["synth_source_levels"]),
        )

        real_indices = np.arange(len(cache["real_frame_ids"]), dtype=np.int32)
        real_path = SHARD_DIR / "real_00000_00185.npz"
        real_fingerprint = _spatial_shard_fingerprint(
            str(recipe["sha256"]), "real", real_indices, cache
        )
        if real_path.is_file():
            if not _spatial_shard_matches(
                real_path,
                real_fingerprint,
                real_indices,
                np.asarray(cache["real_valid"]),
                np.asarray(cache["real_source_levels"]),
            ):
                raise SpatialFusionError(
                    f"EXISTING_SPATIAL_SHARD_MISMATCH: {real_path}. "
                    "This runner will not overwrite it. After review, remove "
                    f"only this owned shard with: rm -- {shlex.quote(str(real_path))}"
                )
        else:
            real_rows: list[dict[str, Any]] = []
            for index, image_rel in enumerate(cache["real_images"].tolist()):
                image, raw_shape = base._read_padded(REPO_ROOT / str(image_rel))
                results, vectors = tap.predict_batch([image])
                real_rows.append(
                    _prediction_and_center_audit(
                        base=base,
                        result=results[0],
                        vector=vectors[0],
                        raw_shape=raw_shape,
                        tap=tap,
                        batch_index=0,
                        expected=_expected_row(cache, "real", index),
                    )
                )
                if (index + 1) % 25 == 0:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] spatial real "
                        f"{index + 1}/{len(real_indices)}",
                        flush=True,
                    )
            _save_spatial_shard(
                real_path, real_indices, real_rows, real_fingerprint
            )
        real_spatial = _load_spatial_shards(
            [real_path],
            len(real_indices),
            {real_path: real_fingerprint},
            np.asarray(cache["real_valid"]),
            np.asarray(cache["real_source_levels"]),
        )
    finally:
        tap.close()
        del tap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        _sha256(PRIMARY_WEIGHTS) != bindings["primary_yolo_sha256"]
        or PRIMARY_WEIGHTS.stat().st_mtime_ns != checkpoint_mtime_before
    ):
        raise SpatialFusionError("YOLO_CHECKPOINT_CHANGED_DURING_SPATIAL_EXTRACTION")
    _assert_runtime_bindings_unchanged(runtime_bindings_before)
    extraction_audit = _extraction_audit(synth_spatial, real_spatial)
    if not extraction_audit["passed"]:
        raise SpatialFusionError(
            f"SPATIAL_EXTRACTION_PROVENANCE_FAILED: {extraction_audit}"
        )
    train_rows = (
        np.asarray(cache["synth_valid"], dtype=np.bool_)
        & (np.asarray(cache["synth_splits"], dtype=np.int8) == SPLIT_CODE["TRAIN"])
    )
    patch_mean, patch_std, normalization_token_n = _patch_normalization(
        synth_spatial["patches"], synth_spatial["patch_masks"], train_rows
    )
    cache_arrays = {
        "synth_frame_ids": np.asarray(cache["synth_frame_ids"]),
        "synth_patches": synth_spatial["patches"],
        "synth_patch_masks": synth_spatial["patch_masks"],
        "synth_source_rows": synth_spatial["source_rows"],
        "synth_source_columns": synth_spatial["source_columns"],
        "synth_source_levels": synth_spatial["source_levels"],
        "real_frame_ids": np.asarray(cache["real_frame_ids"]),
        "real_patches": real_spatial["patches"],
        "real_patch_masks": real_spatial["patch_masks"],
        "real_source_rows": real_spatial["source_rows"],
        "real_source_columns": real_spatial["source_columns"],
        "real_source_levels": real_spatial["source_levels"],
    }
    cache_array_content_sha256 = _array_content_sha256(cache_arrays)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "contract": {
            "path": _display(CONTRACT_PATH),
            "sha256": _sha256(CONTRACT_PATH),
        },
        "protocol": {
            "path": _display(PROTOCOL_PATH),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "protocol_freeze": {
            "path": _display(PROTOCOL_FREEZE_PATH),
            "sha256": _sha256(PROTOCOL_FREEZE_PATH),
        },
        "runner": {"path": _display(RUNNER_PATH), "sha256": _sha256(RUNNER_PATH)},
        "source_bindings": bindings,
        "feature_recipe": recipe,
        "array_content_sha256": cache_array_content_sha256,
        "synthetic_count": int(len(cache["synth_frame_ids"])),
        "real_count": int(len(cache["real_frame_ids"])),
        "storage_dtype": "float16",
        "patch_shape_tokens_channels": [PATCH_TOKENS, TOKEN_DIM],
        "normalization": {
            "patch": {
                "mean": patch_mean.tolist(),
                "std": patch_std.tolist(),
                "valid_token_n": normalization_token_n,
                "source": "SYNTHETIC_TRAIN_VALID_PATCH_TOKENS_ONLY",
                "computed_from_stored_float16": True,
            },
            "kp_and_dimensions": "REUSE_PHASE_C_SYNTHETIC_TRAIN_ONLY_STATS",
        },
        "extraction_audit": extraction_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "yolo_parameter_updates": 0,
    }
    _assert_runtime_bindings_unchanged(runtime_bindings_before)
    _savez_exclusive(
        SPATIAL_CACHE,
        metadata_json=np.asarray(_json_text(metadata)),
        **cache_arrays,
    )
    audit_payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "feature_cache": {
            "path": _display(SPATIAL_CACHE),
            "sha256": _sha256(SPATIAL_CACHE),
            "array_content_sha256": cache_array_content_sha256,
            "size_bytes": SPATIAL_CACHE.stat().st_size,
        },
        "feature_recipe": recipe,
        "source_bindings": bindings,
        "prediction_and_center_provenance": extraction_audit,
        "normalization": metadata["normalization"],
        "checkpoint_sha256_before_after": [
            bindings["primary_yolo_sha256"],
            _sha256(PRIMARY_WEIGHTS),
        ],
        "checkpoint_mtime_unchanged": (
            PRIMARY_WEIGHTS.stat().st_mtime_ns == checkpoint_mtime_before
        ),
        "yolo_requires_grad_true": 0,
        "yolo_optimizer_membership": 0,
        "passed": True,
    }
    _write_json_exclusive(SPATIAL_AUDIT, audit_payload)
    _assert_runtime_bindings_unchanged(runtime_bindings_before)
    return audit_payload


def _validate_final_patch_population(
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str,
    expected_valid: np.ndarray,
    expected_levels: np.ndarray,
) -> None:
    count = len(expected_valid)
    required = {
        f"{prefix}_patches": ((count, PATCH_TOKENS, TOKEN_DIM), np.dtype(np.float16)),
        f"{prefix}_patch_masks": ((count, PATCH_TOKENS), np.dtype(np.bool_)),
        f"{prefix}_source_rows": ((count,), np.dtype(np.int16)),
        f"{prefix}_source_columns": ((count,), np.dtype(np.int16)),
        f"{prefix}_source_levels": ((count,), np.dtype(np.int8)),
    }
    for name, (shape, dtype) in required.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != dtype:
            raise SpatialFusionError(
                f"FINAL_CACHE_{name.upper()}_SHAPE_OR_DTYPE_INVALID"
            )
    patches = np.asarray(arrays[f"{prefix}_patches"])
    masks = np.asarray(arrays[f"{prefix}_patch_masks"])
    valid = np.asarray(expected_valid, dtype=np.bool_)
    levels = np.asarray(arrays[f"{prefix}_source_levels"])
    rows = np.asarray(arrays[f"{prefix}_source_rows"])
    columns = np.asarray(arrays[f"{prefix}_source_columns"])
    if not bool(np.isfinite(patches).all()):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_PATCH_NONFINITE")
    if not np.array_equal(np.any(masks, axis=1), valid):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_VALIDITY_MISMATCH")
    if not bool(np.all(patches[~masks] == 0)):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_PAD_NONZERO")
    if bool(valid.any()) and not bool(masks[valid, PATCH_TOKENS // 2].all()):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_CENTER_MASK_FALSE")
    if not np.array_equal(levels, np.asarray(expected_levels, dtype=np.int8)):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_LEVEL_MISMATCH")
    if bool(valid.any()) and (
        np.any(rows[valid] < 0)
        or np.any(columns[valid] < 0)
        or np.any((levels[valid] < 0) | (levels[valid] >= LEVEL_COUNT))
    ):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_COORDINATE_INVALID")
    if bool((~valid).any()) and not bool(
        np.all(rows[~valid] == -1)
        and np.all(columns[~valid] == -1)
        and np.all(levels[~valid] == -1)
        and np.all(patches[~valid] == 0)
    ):
        raise SpatialFusionError(f"FINAL_CACHE_{prefix.upper()}_SENTINEL_INVALID")


def _load_spatial_cache() -> tuple[dict[str, Any], dict[str, Any]]:
    if not SPATIAL_CACHE.is_file() or not SPATIAL_AUDIT.is_file():
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_AND_AUDIT_REQUIRED")
    contract = _contract()
    _verify_source_bindings(contract)
    audit = _read_json(SPATIAL_AUDIT)
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION or audit.get("passed") is not True:
        raise SpatialFusionError("SPATIAL_FEATURE_AUDIT_NOT_PASS")
    if audit.get("feature_cache", {}).get("sha256") != _sha256(SPATIAL_CACHE):
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_FILE_SHA_MISMATCH")
    with np.load(SPATIAL_CACHE, allow_pickle=False) as source:
        arrays: dict[str, Any] = {name: np.asarray(source[name]) for name in source.files}
    expected_fields = {
        "metadata_json",
        "synth_frame_ids",
        "synth_patches",
        "synth_patch_masks",
        "synth_source_rows",
        "synth_source_columns",
        "synth_source_levels",
        "real_frame_ids",
        "real_patches",
        "real_patch_masks",
        "real_source_rows",
        "real_source_columns",
        "real_source_levels",
    }
    if set(arrays) != expected_fields:
        raise SpatialFusionError(
            f"SPATIAL_FEATURE_CACHE_FIELDS_INVALID: {sorted(arrays)}"
        )
    metadata = json.loads(str(arrays["metadata_json"].item()))
    content_arrays = {
        name: arrays[name] for name in expected_fields if name != "metadata_json"
    }
    content_sha = _array_content_sha256(content_arrays)
    if (
        metadata.get("array_content_sha256") != content_sha
        or audit.get("feature_cache", {}).get("array_content_sha256") != content_sha
    ):
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_ARRAY_CONTENT_SHA_MISMATCH")
    arrays["metadata"] = metadata
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_SCHEMA_MISMATCH")
    expected_metadata = {
        "contract": _sha256(CONTRACT_PATH),
        "protocol": _sha256(PROTOCOL_PATH),
        "protocol_freeze": _sha256(PROTOCOL_FREEZE_PATH),
        "runner": _sha256(RUNNER_PATH),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field, {}).get("sha256") != expected:
            raise SpatialFusionError(
                f"SPATIAL_FEATURE_CACHE_{field.upper()}_SHA_MISMATCH"
            )
    if metadata.get("feature_recipe", {}).get("sha256") != _feature_recipe(contract)[
        "sha256"
    ]:
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_RECIPE_MISMATCH")
    current_sources = _verify_source_bindings(contract)
    if (
        metadata.get("source_bindings") != current_sources
        or audit.get("source_bindings") != current_sources
    ):
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_SOURCE_BINDINGS_MISMATCH")
    shapes = {
        "synth_patches": (40_000, PATCH_TOKENS, TOKEN_DIM),
        "synth_patch_masks": (40_000, PATCH_TOKENS),
        "synth_source_rows": (40_000,),
        "synth_source_columns": (40_000,),
        "synth_source_levels": (40_000,),
        "real_patches": (185, PATCH_TOKENS, TOKEN_DIM),
        "real_patch_masks": (185, PATCH_TOKENS),
        "real_source_rows": (185,),
        "real_source_columns": (185,),
        "real_source_levels": (185,),
    }
    for field, shape in shapes.items():
        if arrays[field].shape != shape:
            raise SpatialFusionError(
                f"SPATIAL_FEATURE_CACHE_{field.upper()}_SHAPE_MISMATCH: "
                f"{arrays[field].shape}"
            )
    if arrays["synth_patches"].dtype != np.float16 or arrays["real_patches"].dtype != np.float16:
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_PATCH_DTYPE_MISMATCH")
    if arrays["synth_patch_masks"].dtype != np.bool_ or arrays["real_patch_masks"].dtype != np.bool_:
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_MASK_DTYPE_MISMATCH")
    base_cache = _phase_c()._load_feature_cache()
    if not np.array_equal(
        arrays["synth_frame_ids"], base_cache["synth_frame_ids"]
    ) or not np.array_equal(arrays["real_frame_ids"], base_cache["real_frame_ids"]):
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_FRAME_ORDER_MISMATCH")
    _validate_final_patch_population(
        arrays,
        prefix="synth",
        expected_valid=np.asarray(base_cache["synth_valid"]),
        expected_levels=np.asarray(base_cache["synth_source_levels"]),
    )
    _validate_final_patch_population(
        arrays,
        prefix="real",
        expected_valid=np.asarray(base_cache["real_valid"]),
        expected_levels=np.asarray(base_cache["real_source_levels"]),
    )
    normalization = metadata.get("normalization", {}).get("patch", {})
    mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std = np.asarray(normalization.get("std", []), dtype=np.float32)
    if (
        mean.shape != (TOKEN_DIM,)
        or std.shape != (TOKEN_DIM,)
        or not bool(np.isfinite(mean).all() and np.isfinite(std).all())
        or np.any(std <= 0.0)
    ):
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_NORMALIZATION_INVALID")
    train_rows = np.asarray(base_cache["synth_valid"], dtype=np.bool_) & (
        np.asarray(base_cache["synth_splits"], dtype=np.int8) == SPLIT_CODE["TRAIN"]
    )
    recomputed_mean, recomputed_std, token_n = _patch_normalization(
        arrays["synth_patches"], arrays["synth_patch_masks"], train_rows
    )
    if (
        not np.array_equal(mean, recomputed_mean)
        or not np.array_equal(std, recomputed_std)
        or int(normalization.get("valid_token_n", -1)) != token_n
    ):
        raise SpatialFusionError("SPATIAL_FEATURE_CACHE_NORMALIZATION_MISMATCH")
    return arrays, audit


def _normalized_dimensions(base_cache: Mapping[str, Any], dimensions_xyz: np.ndarray) -> np.ndarray:
    base = _phase_c()
    raw = base._dimension_raw(np.asarray(dimensions_xyz, dtype=np.float32))
    mean, std = base._normalization(base_cache, "dims")
    return ((raw - mean) / std).astype(np.float32)


def _feature_bundle(
    base_cache: Mapping[str, Any],
    spatial_cache: Mapping[str, Any],
    *,
    real: bool,
    dimensions_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    prefix = "real" if real else "synth"
    base = _phase_c()
    normalized = base._normalized_features(
        base_cache,
        real=real,
        dimensions_override=dimensions_override,
    )
    normalization = spatial_cache["metadata"]["normalization"]["patch"]
    return {
        "patches": np.asarray(spatial_cache[f"{prefix}_patches"], dtype=np.float16),
        "patch_masks": np.asarray(
            spatial_cache[f"{prefix}_patch_masks"], dtype=np.bool_
        ),
        "source_levels": np.asarray(
            spatial_cache[f"{prefix}_source_levels"], dtype=np.int8
        ),
        "kp": np.asarray(normalized["kp"], dtype=np.float32),
        "dims": np.asarray(normalized["dims"], dtype=np.float32),
        "patch_mean": np.asarray(normalization["mean"], dtype=np.float32),
        "patch_std": np.asarray(normalization["std"], dtype=np.float32),
    }


def _bundle_with_dimensions(
    bundle: Mapping[str, np.ndarray], dimensions: np.ndarray
) -> dict[str, np.ndarray]:
    output = dict(bundle)
    output["dims"] = np.asarray(dimensions, dtype=np.float32)
    return output


def _batch_from_bundle(
    bundle: Mapping[str, np.ndarray], indices: np.ndarray, device: str
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    indices = np.asarray(indices, dtype=np.int64)
    patches_np = np.asarray(bundle["patches"])[indices].astype(np.float32)
    mean = np.asarray(bundle["patch_mean"], dtype=np.float32)
    std = np.asarray(bundle["patch_std"], dtype=np.float32)
    patches_np = (patches_np - mean[None, None, :]) / std[None, None, :]
    masks_np = np.asarray(bundle["patch_masks"], dtype=np.bool_)[indices]
    patches_np[~masks_np] = 0.0
    return (
        torch.from_numpy(patches_np).to(device, non_blocking=True),
        torch.from_numpy(masks_np).to(device, non_blocking=True),
        torch.from_numpy(
            np.asarray(bundle["source_levels"], dtype=np.int64)[indices]
        ).to(device, non_blocking=True),
        torch.from_numpy(np.asarray(bundle["kp"], dtype=np.float32)[indices]).to(
            device, non_blocking=True
        ),
        torch.from_numpy(np.asarray(bundle["dims"], dtype=np.float32)[indices]).to(
            device, non_blocking=True
        ),
    )


def _predict_probabilities(
    model: SpatialFusionProbe,
    bundle: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
) -> np.ndarray:
    model.eval()
    indices = np.asarray(indices, dtype=np.int64)
    output = np.empty(len(indices), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(indices), EVAL_BATCH):
            batch_indices = indices[start : start + EVAL_BATCH]
            logits = model(*_batch_from_bundle(bundle, batch_indices, device))
            output[start : start + len(batch_indices)] = (
                torch.softmax(logits, dim=1)[:, LONG_FRONT].detach().cpu().numpy()
            )
    return output


def _checkpoint_bindings(spatial_audit: Mapping[str, Any]) -> dict[str, str]:
    return {
        "contract_sha256": _sha256(CONTRACT_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "protocol_freeze_sha256": _sha256(PROTOCOL_FREEZE_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "spatial_feature_cache_sha256": _sha256(SPATIAL_CACHE),
        "spatial_feature_audit_sha256": _sha256(SPATIAL_AUDIT),
        "spatial_feature_recipe_sha256": str(
            spatial_audit["feature_recipe"]["sha256"]
        ),
        "primary_yolo_sha256": _sha256(PRIMARY_WEIGHTS),
        "phase_c_feature_cache_sha256": _sha256(PHASE_C_CACHE),
        "phase_b_split_sha256": _sha256(SPLIT_MEMBERSHIP),
    }


def _model_config(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "patch_size": PATCH_SIZE,
        "patch_tokens": PATCH_TOKENS,
        "token_dim": TOKEN_DIM,
        "kp_dim": KP_DIM,
        "dimension_dim": DIMENSION_DIM,
        "position_embedding": True,
        "source_level_embedding": True,
        "pool": "masked_mean_plus_masked_max",
        "classifier_input_dim": CLASSIFIER_INPUT_DIM,
        "classifier_hidden": [128, 32],
        "activation": "SiLU",
        "dropout": 0.0,
        "film_hidden": FILM_HIDDEN if arm == "S2_SPATIAL_FILM" else None,
        "attention_dim": ATTENTION_DIM
        if arm == "S3_SPATIAL_CROSS_ATTENTION"
        else None,
        "attention_heads": ATTENTION_HEADS
        if arm == "S3_SPATIAL_CROSS_ATTENTION"
        else None,
    }


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SpatialFusionError(f"PROBE_CHECKPOINT_UNREADABLE: {path}: {exc}") from exc


def _load_probe(
    path: Path,
    *,
    expected_sha256: str,
    expected_bindings: Mapping[str, str],
    device: str,
) -> tuple[SpatialFusionProbe, dict[str, Any]]:
    observed_sha = _sha256(path)
    if observed_sha != expected_sha256:
        raise SpatialFusionError(
            f"PROBE_CHECKPOINT_SHA_MISMATCH: {path}: "
            f"expected={expected_sha256} observed={observed_sha}"
        )
    payload = _load_checkpoint_payload(path)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SpatialFusionError("PROBE_CHECKPOINT_SCHEMA_MISMATCH")
    if payload.get("source_bindings") != dict(expected_bindings):
        raise SpatialFusionError("PROBE_CHECKPOINT_SOURCE_BINDINGS_MISMATCH")
    arm = str(payload.get("arm"))
    if payload.get("model_config") != _model_config(arm):
        raise SpatialFusionError("PROBE_CHECKPOINT_MODEL_CONFIG_MISMATCH")
    model = SpatialFusionProbe(arm)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def _run_paths(arm: str, seed: int) -> tuple[Path, Path]:
    stem = OUT_DIR / arm / f"seed{seed}"
    return stem.with_suffix(".pt"), stem.with_suffix(".json")


def _validate_existing_run(
    arm: str,
    seed: int,
    bindings: Mapping[str, str],
) -> dict[str, Any] | None:
    checkpoint_path, json_path = _run_paths(arm, seed)
    if not checkpoint_path.exists() and not json_path.exists():
        return None
    if not checkpoint_path.is_file() or not json_path.is_file():
        raise SpatialFusionError(
            f"PARTIAL_EXISTING_RUN_ARTIFACT: arm={arm} seed={seed}"
        )
    payload = _read_json(json_path)
    if (
        payload.get("schema_version") != RUN_SCHEMA_VERSION
        or payload.get("arm") != arm
        or int(payload.get("seed", -1)) != seed
        or payload.get("source_bindings") != dict(bindings)
        or payload.get("real_accessed_during_training_or_selection") is not False
        or payload.get("synthetic_test_opened_during_training_or_selection") is not False
        or payload.get("checkpoint", {}).get("sha256") != _sha256(checkpoint_path)
    ):
        raise SpatialFusionError(
            f"EXISTING_RUN_CONTRACT_MISMATCH: arm={arm} seed={seed}"
        )
    return payload


def _train_run(
    *,
    arm: str,
    seed: int,
    bundle: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    dev_indices: np.ndarray,
    device: str,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    existing = _validate_existing_run(arm, seed, bindings)
    if existing is not None:
        return existing
    _seed_everything(seed)
    model = SpatialFusionProbe(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    train_y = np.asarray(labels[train_indices], dtype=np.int64)
    dev_y = np.asarray(labels[dev_indices], dtype=np.int64)
    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    class_weights_np = len(train_y) / (2.0 * counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    sample_weights = class_weights_np[train_y]
    steps_per_epoch = int(math.ceil(len(train_indices) / TRAIN_BATCH))
    if steps_per_epoch != 40 or steps_per_epoch * TRAIN_BATCH != 20_480:
        raise SpatialFusionError(
            "TRAINING_STEP_CONTRACT_MISMATCH: "
            f"steps={steps_per_epoch} samples={steps_per_epoch * TRAIN_BATCH}"
        )
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    schedule_digest = hashlib.sha256()
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + epoch)
        local_indices = torch.multinomial(
            torch.from_numpy(sample_weights),
            num_samples=steps_per_epoch * TRAIN_BATCH,
            replacement=True,
            generator=generator,
        )
        selected_global = train_indices[local_indices.numpy()]
        schedule_digest.update(
            np.ascontiguousarray(selected_global, dtype=np.int32).tobytes()
        )
        losses: list[float] = []
        for step in range(steps_per_epoch):
            indices = selected_global[
                step * TRAIN_BATCH : (step + 1) * TRAIN_BATCH
            ]
            target = torch.from_numpy(
                np.asarray(labels[indices], dtype=np.int64)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*_batch_from_bundle(bundle, indices, device))
            loss = functional.cross_entropy(logits, target, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        dev_probability = _predict_probabilities(
            model, bundle, dev_indices, device
        )
        dev_metrics = _phase_c()._binary_metrics(dev_y, dev_probability)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "dev_accuracy": dev_metrics["accuracy"],
        }
        history.append(record)
        score = float(dev_metrics["balanced_accuracy"])
        # Strict greater-than deliberately preserves the earliest exact max.
        if best is None or score > float(best["score"]):
            best = {
                "score": score,
                "epoch": epoch + 1,
                "state_dict": copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                ),
                "dev_metrics": dev_metrics,
            }
    if best is None:
        raise SpatialFusionError("SPATIAL_PROBE_TRAINING_NO_CHECKPOINT")
    checkpoint_path, json_path = _run_paths(arm, seed)
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_dict": best["state_dict"],
        "arm": arm,
        "seed": seed,
        "selected_epoch": int(best["epoch"]),
        "model_config": _model_config(arm),
        "source_bindings": dict(bindings),
    }
    _torch_save_exclusive(checkpoint_path, checkpoint_payload)
    run_payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "arm": arm,
        "seed": seed,
        "model_config": _model_config(arm),
        "source_bindings": dict(bindings),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "scheduler": "none",
        "gradient_clipping": "none",
        "amp": False,
        "balanced_cross_entropy_weights": class_weights_np.tolist(),
        "class_balanced_sampler": True,
        "fixed_epochs_executed": TRAIN_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "fixed_optimizer_steps": TRAIN_EPOCHS * steps_per_epoch,
        "sampler_schedule_sha256": schedule_digest.hexdigest(),
        "checkpoint_epoch_selected_by": (
            "EARLIEST_MAXIMUM_SYNTHETIC_DEV_BALANCED_ACCURACY"
        ),
        "selected_epoch": int(best["epoch"]),
        "synthetic_dev": best["dev_metrics"],
        "synthetic_test": {"status": "SEALED_BEFORE_SOURCE_LOCK", "opened": False},
        "real_accessed_during_training_or_selection": False,
        "synthetic_test_opened_during_training_or_selection": False,
        "yolo_parameters_in_optimizer": 0,
        "checkpoint": {
            "path": _display(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "history": history,
    }
    _write_json_exclusive(json_path, run_payload)
    return run_payload


def _median_dev_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(runs) != len(SEEDS):
        raise SpatialFusionError("MEDIAN_DEV_REQUIRES_THREE_SEEDS")
    values = sorted(float(run["synthetic_dev"]["balanced_accuracy"]) for run in runs)
    median = values[len(values) // 2]
    matches = [
        run
        for run in runs
        if float(run["synthetic_dev"]["balanced_accuracy"]) == median
    ]
    return min(matches, key=lambda run: int(run["seed"]))


def _source_selection(
    runs_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    method_scores = {
        arm: float(
            np.mean(
                [
                    float(run["synthetic_dev"]["balanced_accuracy"])
                    for run in runs_by_arm[arm]
                ]
            )
        )
        for arm in ARMS
    }
    s1 = method_scores["S1_SPATIAL_CONCAT"]
    s2_eligible = method_scores["S2_SPATIAL_FILM"] >= s1 + 0.02
    s3_eligible = method_scores["S3_SPATIAL_CROSS_ATTENTION"] >= s1 + 0.02
    primary = "S1_SPATIAL_CONCAT"
    reason = "CONCAT_DEFAULT_COMPLEX_METHOD_NOT_2PP_BETTER"
    if s2_eligible and s3_eligible:
        if method_scores["S3_SPATIAL_CROSS_ATTENTION"] >= method_scores[
            "S2_SPATIAL_FILM"
        ] + 0.02:
            primary = "S3_SPATIAL_CROSS_ATTENTION"
            reason = "ATTENTION_2PP_ABOVE_CONCAT_AND_FILM"
        else:
            primary = "S2_SPATIAL_FILM"
            reason = "FILM_2PP_ABOVE_CONCAT_ATTENTION_NOT_2PP_ABOVE_FILM"
    elif s2_eligible:
        primary = "S2_SPATIAL_FILM"
        reason = "FILM_2PP_ABOVE_CONCAT"
    elif s3_eligible:
        primary = "S3_SPATIAL_CROSS_ATTENTION"
        reason = "ATTENTION_2PP_ABOVE_CONCAT"
    artifact_by_arm = {
        arm: _median_dev_run(runs_by_arm[arm]) for arm in ARMS
    }
    return {
        "method_scores_three_seed_dev_mean": method_scores,
        "complex_eligibility": {
            "minimum_absolute_improvement_over_concat": 0.02,
            "S2_SPATIAL_FILM": s2_eligible,
            "S3_SPATIAL_CROSS_ATTENTION": s3_eligible,
        },
        "primary_method": primary,
        "primary_selection_reason": reason,
        "artifact_rule": "median selected-epoch DEV balanced accuracy; tie lower seed",
        "artifact_by_arm": {
            arm: {
                "seed": int(run["seed"]),
                "synthetic_dev_balanced_accuracy": run["synthetic_dev"][
                    "balanced_accuracy"
                ],
                "checkpoint": run["checkpoint"],
            }
            for arm, run in artifact_by_arm.items()
        },
    }


def _resolve_artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _correct_count(metrics: Mapping[str, Any]) -> int:
    confusion = metrics[
        "confusion_matrix_rows_true_short_long_cols_pred_short_long_abstain"
    ]
    return int(confusion[SHORT_FRONT][SHORT_FRONT]) + int(
        confusion[LONG_FRONT][LONG_FRONT]
    )


def _prediction_vector(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    selected = np.full(len(values), -1, dtype=np.int8)
    finite = np.isfinite(values)
    selected[finite] = (values[finite] >= 0.5).astype(np.int8)
    return selected


def _evaluate_indices(
    *,
    model: SpatialFusionProbe,
    bundle: Mapping[str, np.ndarray],
    indices: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.asarray(indices, dtype=np.int64)
    valid = np.asarray(valid, dtype=np.bool_)
    if valid.shape != (len(indices),):
        raise SpatialFusionError("EVALUATION_VALID_MASK_SHAPE_MISMATCH")
    probability = np.full(len(indices), np.nan, dtype=np.float64)
    probability[valid] = _predict_probabilities(
        model, bundle, indices[valid], device
    )
    base = _phase_c()
    all_metrics = base._binary_metrics(labels, probability, valid)
    detected_metrics = base._binary_metrics(labels[valid], probability[valid])
    payload = {
        "all_frame_with_abstention": all_metrics,
        "detected_only": detected_metrics,
        "correct_n": _correct_count(all_metrics),
        "decision_threshold": 0.5,
    }
    return probability, payload


def _source_lock_payload(
    *,
    selection: Mapping[str, Any],
    runs_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, str],
    step_zero: Mapping[str, Any],
) -> dict[str, Any]:
    run_artifacts: dict[str, list[dict[str, Any]]] = {}
    sampler_hashes: dict[str, list[str]] = {}
    for arm in ARMS:
        ordered = sorted(runs_by_arm[arm], key=lambda row: int(row["seed"]))
        run_artifacts[arm] = []
        sampler_hashes[arm] = []
        for run in ordered:
            checkpoint = run["checkpoint"]
            json_path = _run_paths(arm, int(run["seed"]))[1]
            run_artifacts[arm].append(
                {
                    "seed": int(run["seed"]),
                    "selected_epoch": int(run["selected_epoch"]),
                    "synthetic_dev_balanced_accuracy": run["synthetic_dev"][
                        "balanced_accuracy"
                    ],
                    "checkpoint": checkpoint,
                    "run_json": {
                        "path": _display(json_path),
                        "sha256": _sha256(json_path),
                    },
                    "sampler_schedule_sha256": run[
                        "sampler_schedule_sha256"
                    ],
                }
            )
            sampler_hashes[arm].append(str(run["sampler_schedule_sha256"]))
    per_seed_schedule_exact = {
        str(seed): len(
            {
                str(runs_by_arm[arm][seed]["sampler_schedule_sha256"])
                for arm in ARMS
            }
        )
        == 1
        for seed in SEEDS
    }
    if not all(per_seed_schedule_exact.values()):
        raise SpatialFusionError(
            f"SAMPLER_SCHEDULE_NOT_IDENTICAL_ACROSS_ARMS: {sampler_hashes}"
        )
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "locked_at_utc": _now(),
        "repo_head": _repo_head(),
        "selection_source": "SYNTHETIC_DEV_ONLY",
        "method_score": (
            "MEAN_SELECTED_EPOCH_SYNTHETIC_DEV_BALANCED_ACCURACY_SEEDS_0_1_2"
        ),
        "artifact_by_arm_rule": (
            "MEDIAN_SELECTED_EPOCH_DEV_SCORE_TIE_LOWER_SEED"
        ),
        "synthetic_test_artifact_rule": "LOCKED_MEDIAN_ARTIFACT_PER_ARM",
        "real_and_gate_artifact_rule": (
            "LOCKED_PRIMARY_MEDIAN_PLUS_LOCKED_S0_MEDIAN_ONLY"
        ),
        "source_bindings": dict(bindings),
        "selection": dict(selection),
        "training_runs": run_artifacts,
        "step_zero_audit_by_seed": dict(step_zero),
        "sampler_schedule_exact_across_arms_by_seed": per_seed_schedule_exact,
        "synthetic_test_historically_opened_before_phase_e": True,
        "real_historically_opened_before_phase_e": True,
        "phase_e_synthetic_test_reaccessed_before_lock": False,
        "phase_e_real_reaccessed_before_lock": False,
        "real_labels_used_for_training_epoch_seed_method_or_threshold_selection": 0,
        "real_evaluation_scope_after_lock": (
            "PRIMARY_MEDIAN_PLUS_S0_MEDIAN_PLUS_PRIMARY_SHUFFLED_WRONG_CONTROLS_ONLY"
        ),
        "irrevocably_exploratory": True,
        "promotion_eligible": False,
    }


def _validate_source_lock(
    lock: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    runs_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    bindings: Mapping[str, str],
) -> None:
    required_exact = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "repo_head": _repo_head(),
        "selection_source": "SYNTHETIC_DEV_ONLY",
        "source_bindings": dict(bindings),
        "selection": dict(selection),
        "synthetic_test_historically_opened_before_phase_e": True,
        "real_historically_opened_before_phase_e": True,
        "phase_e_synthetic_test_reaccessed_before_lock": False,
        "phase_e_real_reaccessed_before_lock": False,
        "real_labels_used_for_training_epoch_seed_method_or_threshold_selection": 0,
        "irrevocably_exploratory": True,
        "promotion_eligible": False,
    }
    for field, expected in required_exact.items():
        if lock.get(field) != expected:
            raise SpatialFusionError(f"SOURCE_SELECTION_LOCK_{field.upper()}_MISMATCH")
    locked_runs = lock.get("training_runs")
    if not isinstance(locked_runs, Mapping):
        raise SpatialFusionError("SOURCE_SELECTION_LOCK_TRAINING_RUNS_INVALID")
    for arm in ARMS:
        records = locked_runs.get(arm)
        if not isinstance(records, list) or len(records) != len(SEEDS):
            raise SpatialFusionError(
                f"SOURCE_SELECTION_LOCK_RUN_COUNT_INVALID: {arm}"
            )
        observed_by_seed = {int(row["seed"]): row for row in records}
        for run in runs_by_arm[arm]:
            seed = int(run["seed"])
            record = observed_by_seed.get(seed)
            if record is None:
                raise SpatialFusionError(
                    f"SOURCE_SELECTION_LOCK_RUN_MISSING: {arm} seed={seed}"
                )
            checkpoint_path, json_path = _run_paths(arm, seed)
            if (
                record.get("checkpoint") != run["checkpoint"]
                or record.get("run_json", {}).get("path") != _display(json_path)
                or record.get("run_json", {}).get("sha256") != _sha256(json_path)
                or run["checkpoint"].get("sha256") != _sha256(checkpoint_path)
            ):
                raise SpatialFusionError(
                    f"SOURCE_SELECTION_LOCK_RUN_SHA_MISMATCH: {arm} seed={seed}"
                )
    for arm, artifact in selection["artifact_by_arm"].items():
        path = _resolve_artifact_path(str(artifact["checkpoint"]["path"]))
        if _sha256(path) != artifact["checkpoint"]["sha256"]:
            raise SpatialFusionError(
                f"SOURCE_SELECTION_LOCK_ARTIFACT_SHA_MISMATCH: {arm}"
            )


def _load_locked_model(
    arm: str,
    selection: Mapping[str, Any],
    bindings: Mapping[str, str],
    device: str,
) -> tuple[SpatialFusionProbe, dict[str, Any]]:
    artifact = selection["artifact_by_arm"][arm]
    checkpoint = artifact["checkpoint"]
    model, payload = _load_probe(
        _resolve_artifact_path(str(checkpoint["path"])),
        expected_sha256=str(checkpoint["sha256"]),
        expected_bindings=bindings,
        device=device,
    )
    if (
        payload.get("arm") != arm
        or int(payload.get("seed", -1)) != int(artifact["seed"])
    ):
        raise SpatialFusionError(f"LOCKED_ARTIFACT_IDENTITY_MISMATCH: {arm}")
    return model, payload


def _latency_one_model(
    model: SpatialFusionProbe,
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    *,
    warmup: int,
    iterations: int,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(*batch)
        torch.cuda.synchronize(batch[0].device)
        elapsed = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            torch.cuda.synchronize(batch[0].device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*batch)
            end.record()
            end.synchronize()
            torch.cuda.synchronize(batch[0].device)
            elapsed[index] = float(start.elapsed_time(end))
    return elapsed


def _latency_gate(
    *,
    pre_latency_passed: bool,
    s0_model: SpatialFusionProbe,
    primary_model: SpatialFusionProbe,
    real_bundle: Mapping[str, np.ndarray],
    real_valid: np.ndarray,
    device: str,
) -> dict[str, Any]:
    base_payload: dict[str, Any] = {
        "execution_rule": "ONLY_IF_E1_THROUGH_E8_PASS",
        "baseline": "S0_SPATIAL_NO_DIMS_PARITY_BRANCH",
        "scope": (
            "PARITY_BRANCH_ONLY_CACHED_NORMALIZED_DEVICE_TENSORS_"
            "EXCLUDES_YOLO_EXTRACTION_HOST_TRANSFER"
        ),
        "timer": (
            "CUDA_EVENTS_SYNCHRONIZE_BEFORE_AND_AFTER_EVERY_TIMED_ITERATION"
        ),
        "relative_increase_formula": "(primary_ms - S0_ms) / S0_ms",
        "warmup": 20,
        "iterations": 200,
        "batch": 1,
        "maximum_relative_increase": 0.05,
    }
    if not pre_latency_passed:
        return {
            **base_payload,
            "status": "NOT_RUN_E1_TO_E8_FAILED",
            "passed": False,
        }
    parsed_device = torch.device(device)
    if parsed_device.type != "cuda" or not torch.cuda.is_available():
        return {
            **base_payload,
            "status": "BLOCKED_PINNED_CUDA_HARDWARE_UNAVAILABLE",
            "passed": False,
        }
    index = parsed_device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    memory_mib = int(round(properties.total_memory / (1024 * 1024)))
    hardware = f"{properties.name} {memory_mib} MiB"
    if hardware != "NVIDIA GeForce RTX 3080 10240 MiB":
        return {
            **base_payload,
            "status": "BLOCKED_PINNED_CUDA_HARDWARE_MISMATCH",
            "observed_hardware": hardware,
            "required_hardware": "NVIDIA GeForce RTX 3080 10240 MiB",
            "passed": False,
        }
    valid_indices = np.flatnonzero(np.asarray(real_valid, dtype=np.bool_))
    if len(valid_indices) == 0:
        raise SpatialFusionError("LATENCY_GATE_NO_VALID_REAL_ROW")
    # Host-to-device transfer and normalization happen once before timing.
    batch = _batch_from_bundle(
        real_bundle, np.asarray([valid_indices[0]], dtype=np.int64), device
    )
    baseline_ms = _latency_one_model(
        s0_model, batch, warmup=20, iterations=200
    )
    primary_ms = _latency_one_model(
        primary_model, batch, warmup=20, iterations=200
    )
    baseline_stats = {
        "p50_ms": float(np.percentile(baseline_ms, 50)),
        "p95_ms": float(np.percentile(baseline_ms, 95)),
    }
    primary_stats = {
        "p50_ms": float(np.percentile(primary_ms, 50)),
        "p95_ms": float(np.percentile(primary_ms, 95)),
    }
    relative = {
        name: (primary_stats[f"{name}_ms"] - baseline_stats[f"{name}_ms"])
        / baseline_stats[f"{name}_ms"]
        for name in ("p50", "p95")
    }
    passed = bool(all(value <= 0.05 for value in relative.values()))
    return {
        **base_payload,
        "status": "COMPLETE",
        "hardware": hardware,
        "cached_frame_index": int(valid_indices[0]),
        "S0": baseline_stats,
        "primary": primary_stats,
        "relative_increase": relative,
        "passed": passed,
    }


def _build_gates_e1_to_e8(
    *,
    primary: str,
    synthetic_test: Mapping[str, Any],
    synthetic_controls: Mapping[str, Any],
    real_results: Mapping[str, Any],
    pose_metrics: Mapping[str, Any],
    spatial_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_groups = real_results[primary]["subgroups"]
    s0_groups = real_results["S0_SPATIAL_NO_DIMS"]["subgroups"]
    common = primary_groups["PLASTIC:COMMON128:ALL"]
    night = primary_groups["PLASTIC:COMMON128:NIGHT"]
    e1_value = _correct_count(common)
    e2_value = _correct_count(night)
    if int(common["N"]) != 128 or int(night["N"]) != 28:
        raise SpatialFusionError(
            "REAL_GATE_POPULATION_COUNT_MISMATCH: "
            f"common={common['N']} night={night['N']}"
        )
    session_thresholds = {
        "eval_outside": (9, 10),
        "eval_noapril": (11, 12),
        "eval_cad": (16, 18),
        "eval_pallet07": (23, 27),
        "eval_pallet09": (29, 33),
        "eval_night08": (11, 12),
        "eval_night09": (14, 16),
    }
    session_values: dict[str, Any] = {}
    for session, (minimum, expected_n) in session_thresholds.items():
        key = f"PLASTIC:COMMON128:SESSION:{session}"
        metrics = primary_groups[key]
        observed_n = int(metrics["N"])
        if observed_n != expected_n:
            raise SpatialFusionError(
                f"REAL_GATE_SESSION_COUNT_MISMATCH: {session}: "
                f"expected={expected_n} observed={observed_n}"
            )
        correct = _correct_count(metrics)
        session_values[session] = {
            "passed": correct >= minimum,
            "correct_n": correct,
            "N": observed_n,
            "minimum_correct_n": minimum,
        }
    primary_common_correct = e1_value
    s0_common_correct = _correct_count(
        s0_groups["PLASTIC:COMMON128:ALL"]
    )
    e4_delta = primary_common_correct - s0_common_correct
    current_test_correct = int(synthetic_test[primary]["correct_n"])
    shuffled_test_correct = int(
        synthetic_controls["arms"][primary]["correct_n"]
    )
    e5_delta = current_test_correct - shuffled_test_correct
    raw_changed = int(synthetic_controls["raw_xyz_triplet_changed_n"])
    normalized_changed = int(
        synthetic_controls["normalized_dimension_feature_changed_n"]
    )
    candidate_pose = pose_metrics[primary]["COMMON128"]
    candidate_auc = candidate_pose["restricted_adds_auc"]
    if candidate_auc is None:
        recovery = None
    else:
        recovery = (float(candidate_auc) - 0.27236328125) / (
            0.32071875 - 0.27236328125
        )
    pose_valid_n = int(candidate_pose["pose_valid_n"])
    provenance = spatial_audit["prediction_and_center_provenance"]
    max_diff = provenance["max_abs_diff"]
    e8_pass = bool(
        provenance.get("passed") is True
        and provenance.get("prediction_exact_n") == provenance.get("n")
        and provenance.get("center_exact_n") == provenance.get("n")
        and all(float(value) == 0.0 for value in max_diff.values())
    )
    gates = {
        "population_id": "COMMON_DEV_PLASTIC_POS128_ADAPTIVELY_REUSED_DEV_DIAGNOSTIC",
        "irrevocably_exploratory": True,
        "artifact_scope": (
            "LOCKED_PRIMARY_MEDIAN_AND_LOCKED_S0_MEDIAN_NO_ENSEMBLE"
        ),
        "E1": {
            "passed": e1_value >= 122,
            "correct_n": e1_value,
            "N": 128,
            "minimum_correct_n": 122,
        },
        "E2": {
            "passed": e2_value >= 26,
            "correct_n": e2_value,
            "N": 28,
            "minimum_correct_n": 26,
        },
        "E3": {
            "passed": all(row["passed"] for row in session_values.values()),
            "sessions": session_values,
        },
        "E4": {
            "passed": e4_delta >= 7,
            "primary_correct_n": primary_common_correct,
            "S0_correct_n": s0_common_correct,
            "primary_minus_S0_correct_n": e4_delta,
            "minimum_correct_delta_n": 7,
        },
        "E5": {
            "passed": bool(
                e5_delta >= 505
                and raw_changed == 10_095
                and normalized_changed == 10_095
            ),
            "population_n": 10_095,
            "current_correct_n": current_test_correct,
            "shuffled_correct_n": shuffled_test_correct,
            "current_minus_shuffled_correct_n": e5_delta,
            "minimum_correct_delta_n": 505,
            "raw_xyz_triplet_changed_n": raw_changed,
            "normalized_dimension_feature_changed_n": normalized_changed,
            "required_changed_n_each_representation": 10_095,
        },
        "E6": {
            "passed": bool(
                candidate_auc is not None
                and float(candidate_auc) >= 0.306212109375
                and recovery is not None
                and recovery >= 0.70
            ),
            "candidate_restricted_adds_auc": candidate_auc,
            "minimum_candidate_auc": 0.306212109375,
            "oracle_recovery_ratio": recovery,
            "minimum_oracle_recovery_ratio": 0.70,
            "A0_restricted_adds_auc": 0.27236328125,
            "A1_restricted_adds_auc": 0.32071875,
        },
        "E7": {
            "passed": pose_valid_n >= 126,
            "pose_valid_n": pose_valid_n,
            "N": 128,
            "minimum_pose_valid_n": 126,
        },
        "E8": {
            "passed": e8_pass,
            "prediction_and_float32_center_max_abs_diff": max_diff,
            "required_tolerance": 0.0,
            "float16_center_quantization_max_abs_diff_report_only": provenance[
                "float16_center_quantization_max_abs_diff_report_only"
            ],
        },
    }
    details = {
        "oracle_recovery": {
            "formula": (
                "(candidate_auc - 0.27236328125) / "
                "(0.32071875 - 0.27236328125)"
            ),
            "candidate_auc": candidate_auc,
            "ratio": recovery,
        }
    }
    return gates, details


def _spatial_comparison_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "arm",
        "method_score_three_seed_dev_mean",
        "artifact_seed",
        "artifact_dev_balanced_accuracy",
        "primary_method",
        "synth_test_correct_n",
        "synth_test_n",
        "synth_test_accuracy",
        "synth_test_balanced_accuracy",
        "synth_test_auroc",
        "synth_shuffled_correct_n",
        "synth_shuffled_accuracy",
        "real_evaluated",
        "real_common128_correct_n",
        "real_common128_accuracy",
        "real_night_correct_n",
        "real_night_accuracy",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _report_markdown(summary: Mapping[str, Any]) -> str:
    selection = summary["source_selection"]
    gate = summary["diagnostic_gates"]
    primary = selection["primary_method"]
    lines = [
        "# Phase-E spatial fusion report",
        "",
        "This run is irrevocably exploratory. It preserves the completed "
        "Phase-C result and cannot authorize promotion, full-YOLO training, "
        "wrapper integration, or deployment.",
        "",
        "## Source-only lock",
        "",
        f"- Locked primary method: `{primary}`",
        f"- Selection reason: `{selection['primary_selection_reason']}`",
        "- Method score: mean selected-epoch synthetic-DEV balanced accuracy "
        "over seeds 0/1/2.",
        "- Reported artifact per arm: median-DEV seed, tie lower seed.",
        "- Synthetic TEST and reused real DEV were re-accessed only after the "
        "durable Phase-E lock.",
        "",
        "## Frozen E-gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    for name in [f"E{index}" for index in range(1, 10)]:
        lines.append(f"| {name} | {str(bool(gate[name]['passed']))} |")
    lines.extend(
        [
            "",
            f"Overall E1-E9 pass: `{gate['passed']}`.",
            "",
            "Synthetic TEST uses each arm's locked median artifact. Real and "
            "E-gates use only the locked primary median and S0 median; only the "
            "primary receives shuffled/wrong-dimension real controls. Losing "
            "conditioned methods were not evaluated on real in Phase E.",
            "",
            "Wood pose remains blocked and no multishape-generalization claim "
            "is made.",
            "",
            "Maximum possible verdict: "
            "`EXPLORATORY_SPATIAL_FUSION_DIAGNOSTIC_COMPLETE`.",
            "",
        ]
    )
    return "\n".join(lines)


def train_and_evaluate(device: str) -> dict[str, Any]:
    final_artifacts = (
        REAL_RESULTS,
        PNP_RESULTS,
        CONTROL_RESULTS,
        COMPARISON_CSV,
        SUMMARY_PATH,
        REPORT_PATH,
    )
    existing_final = [path for path in final_artifacts if path.exists()]
    if existing_final:
        raise SpatialFusionError(
            "FINAL_ARTIFACT_ALREADY_EXISTS: "
            + ", ".join(_display(path) for path in existing_final)
        )
    contract = _contract()
    source_bindings = _verify_source_bindings(contract)
    checkpoint_sha_before = _sha256(PRIMARY_WEIGHTS)
    checkpoint_mtime_before = PRIMARY_WEIGHTS.stat().st_mtime_ns
    spatial_cache, spatial_audit = _load_spatial_cache()
    base = _phase_c()
    base_cache = base._load_feature_cache()
    if not np.array_equal(
        spatial_cache["synth_frame_ids"], base_cache["synth_frame_ids"]
    ) or not np.array_equal(
        spatial_cache["real_frame_ids"], base_cache["real_frame_ids"]
    ):
        raise SpatialFusionError("SPATIAL_AND_PHASE_C_CACHE_FRAME_ORDER_MISMATCH")
    synth_valid = np.asarray(base_cache["synth_valid"], dtype=np.bool_)
    real_valid = np.asarray(base_cache["real_valid"], dtype=np.bool_)
    if not np.array_equal(
        np.any(spatial_cache["synth_patch_masks"], axis=1), synth_valid
    ) or not np.array_equal(
        np.any(spatial_cache["real_patch_masks"], axis=1), real_valid
    ):
        raise SpatialFusionError("SPATIAL_PATCH_VALIDITY_PHASE_C_MISMATCH")
    splits = np.asarray(base_cache["synth_splits"], dtype=np.int8)
    labels = np.asarray(base_cache["synth_labels"], dtype=np.int64)
    split_indices = {
        name: np.flatnonzero(splits == code).astype(np.int64)
        for name, code in SPLIT_CODE.items()
    }
    expected_counts = {"TRAIN": 20_281, "DEV": 9_624, "TEST": 10_095}
    observed_counts = {name: len(values) for name, values in split_indices.items()}
    if observed_counts != expected_counts:
        raise SpatialFusionError(
            f"SYNTHETIC_SPLIT_COUNT_MISMATCH: {observed_counts}"
        )
    for name in ("TRAIN", "DEV"):
        if not bool(synth_valid[split_indices[name]].all()):
            raise SpatialFusionError(f"SYNTHETIC_{name}_INVALID_DETECTION_ROW")
    synth_bundle = _feature_bundle(
        base_cache, spatial_cache, real=False
    )
    bindings = _checkpoint_bindings(spatial_audit)

    step_zero_by_seed: dict[str, Any] = {}
    step_indices = split_indices["TRAIN"][:8]
    step_batch = _batch_from_bundle(synth_bundle, step_indices, device)
    for seed in SEEDS:
        models: dict[str, SpatialFusionProbe] = {}
        for arm in DIMENSION_AWARE_ARMS:
            _seed_everything(seed)
            models[arm] = SpatialFusionProbe(arm).to(device)
        step_zero_by_seed[str(seed)] = _step_zero_audit(models, step_batch)
        del models
    del step_batch

    runs_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arm in ARMS:
        for seed in SEEDS:
            run = _train_run(
                arm=arm,
                seed=seed,
                bundle=synth_bundle,
                labels=labels,
                train_indices=split_indices["TRAIN"],
                dev_indices=split_indices["DEV"],
                device=device,
                bindings=bindings,
            )
            runs_by_arm[arm].append(run)
            print(
                f"[{time.strftime('%H:%M:%S')}] spatial train {arm} "
                f"seed={seed} dev_bal="
                f"{float(run['synthetic_dev']['balanced_accuracy']):.6f}",
                flush=True,
            )
    for seed in SEEDS:
        hashes = {
            str(runs_by_arm[arm][seed]["sampler_schedule_sha256"])
            for arm in ARMS
        }
        if len(hashes) != 1:
            raise SpatialFusionError(
                f"SAMPLER_SCHEDULE_NOT_IDENTICAL_FOR_SEED: {seed}: {hashes}"
            )
    selection = _source_selection(runs_by_arm)
    if SELECTION_LOCK.is_file():
        selection_lock = _read_json(SELECTION_LOCK)
        _validate_source_lock(
            selection_lock,
            selection=selection,
            runs_by_arm=runs_by_arm,
            bindings=bindings,
        )
    else:
        selection_lock = _source_lock_payload(
            selection=selection,
            runs_by_arm=runs_by_arm,
            bindings=bindings,
            step_zero=step_zero_by_seed,
        )
        _write_json_exclusive(SELECTION_LOCK, selection_lock)
    selection_lock_sha = _sha256(SELECTION_LOCK)
    _validate_source_lock(
        selection_lock,
        selection=selection,
        runs_by_arm=runs_by_arm,
        bindings=bindings,
    )

    # Phase-E TEST re-access starts only after the durable source-only lock.
    test_indices = split_indices["TEST"]
    test_valid = synth_valid[test_indices]
    test_labels = labels[test_indices]
    synthetic_test: dict[str, Any] = {}
    for arm in ARMS:
        model, _ = _load_locked_model(arm, selection, bindings, device)
        _probability, metrics = _evaluate_indices(
            model=model,
            bundle=synth_bundle,
            indices=test_indices,
            labels=test_labels,
            valid=test_valid,
            device=device,
        )
        synthetic_test[arm] = {
            "artifact": selection["artifact_by_arm"][arm],
            **metrics,
        }
        del model

    test_raw_dimensions = np.asarray(
        base_cache["synth_dimensions_xyz"], dtype=np.float32
    )[test_indices]
    shuffled_test_raw, test_permutation = _shuffled_dimensions(
        test_raw_dimensions,
        seed=42,
        require_all_triplets_changed=True,
    )
    original_test_normalized = _normalized_dimensions(
        base_cache, test_raw_dimensions
    )
    shuffled_test_normalized = _normalized_dimensions(
        base_cache, shuffled_test_raw
    )
    raw_changed_mask = np.any(shuffled_test_raw != test_raw_dimensions, axis=1)
    normalized_changed_mask = np.any(
        shuffled_test_normalized != original_test_normalized, axis=1
    )
    if not bool(raw_changed_mask.all()) or not bool(
        normalized_changed_mask.all()
    ):
        raise SpatialFusionError(
            "SYNTHETIC_TEST_SHUFFLE_NOT_100_PERCENT_CHANGED_IN_BOTH_REPRESENTATIONS"
        )
    shuffled_synth_dimensions = np.asarray(
        synth_bundle["dims"], dtype=np.float32
    ).copy()
    shuffled_synth_dimensions[test_indices] = shuffled_test_normalized
    shuffled_synth_bundle = _bundle_with_dimensions(
        synth_bundle, shuffled_synth_dimensions
    )
    synthetic_control_arms: dict[str, Any] = {}
    for arm in DIMENSION_AWARE_ARMS:
        model, _ = _load_locked_model(arm, selection, bindings, device)
        _probability, metrics = _evaluate_indices(
            model=model,
            bundle=shuffled_synth_bundle,
            indices=test_indices,
            labels=test_labels,
            valid=test_valid,
            device=device,
        )
        synthetic_control_arms[arm] = {
            "artifact": selection["artifact_by_arm"][arm],
            **metrics,
        }
        del model
    synthetic_controls = {
        "population": "CURRENT_G38_SYNTHETIC_TEST_ADAPTIVELY_REUSED_DEV_DIAGNOSTIC",
        "N": len(test_indices),
        "seed": 42,
        "permutation_local": test_permutation,
        "different_source_frame_n": int(
            np.sum(np.asarray(test_permutation) != np.arange(len(test_indices)))
        ),
        "raw_xyz_triplet_changed_n": int(raw_changed_mask.sum()),
        "normalized_dimension_feature_changed_n": int(
            normalized_changed_mask.sum()
        ),
        "normalization_dtype": "float32",
        "arms": synthetic_control_arms,
    }

    # Real GT may first be re-accessed in Phase E below this point.  Only S0,
    # the locked primary, and controls of that primary are evaluated.
    if not SELECTION_LOCK.is_file() or _sha256(SELECTION_LOCK) != selection_lock_sha:
        raise SpatialFusionError("SOURCE_SELECTION_LOCK_CHANGED_BEFORE_REAL_ACCESS")
    real_labels = base._real_truth(base_cache)
    real_bundle = _feature_bundle(base_cache, spatial_cache, real=True)
    primary = str(selection["primary_method"])
    real_models: dict[str, SpatialFusionProbe] = {}
    real_probability: dict[str, np.ndarray] = {}
    real_results: dict[str, Any] = {}
    all_real_indices = np.arange(len(real_labels), dtype=np.int64)
    for arm in ("S0_SPATIAL_NO_DIMS", primary):
        model, _ = _load_locked_model(arm, selection, bindings, device)
        real_models[arm] = model
        probability, _metrics = _evaluate_indices(
            model=model,
            bundle=real_bundle,
            indices=all_real_indices,
            labels=real_labels,
            valid=real_valid,
            device=device,
        )
        real_probability[arm] = probability
        real_results[arm] = {
            "artifact": selection["artifact_by_arm"][arm],
            "subgroups": base._evaluate_real_parity(
                real_labels, probability, real_valid, base_cache
            ),
        }

    real_dimensions = np.asarray(
        base_cache["real_dimensions_xyz"], dtype=np.float32
    )
    control_indices = base._common_multishape_indices(base_cache)
    shuffled_control_values, real_permutation_local = _shuffled_dimensions(
        real_dimensions[control_indices], seed=42
    )
    shuffled_real_dimensions = real_dimensions.copy()
    shuffled_real_dimensions[control_indices] = shuffled_control_values
    real_permutation_global = np.arange(len(real_dimensions), dtype=np.int64)
    real_permutation_global[control_indices] = control_indices[
        np.asarray(real_permutation_local, dtype=np.int64)
    ]
    shuffled_real_bundle = _feature_bundle(
        base_cache,
        spatial_cache,
        real=True,
        dimensions_override=shuffled_real_dimensions,
    )
    shuffled_probability, _ = _evaluate_indices(
        model=real_models[primary],
        bundle=shuffled_real_bundle,
        indices=all_real_indices,
        labels=real_labels,
        valid=real_valid,
        device=device,
    )
    shuffled_name = f"{primary}__SHUFFLED_DIMS"
    real_probability[shuffled_name] = shuffled_probability
    real_results[shuffled_name] = {
        "trained_model": primary,
        "retrained": False,
        "artifact": selection["artifact_by_arm"][primary],
        "subgroups": base._evaluate_real_parity(
            real_labels, shuffled_probability, real_valid, base_cache
        ),
    }
    wrong_dimensions = base._wrong_type_dimensions(base_cache, real_dimensions)
    wrong_real_bundle = _feature_bundle(
        base_cache,
        spatial_cache,
        real=True,
        dimensions_override=wrong_dimensions,
    )
    wrong_probability, _ = _evaluate_indices(
        model=real_models[primary],
        bundle=wrong_real_bundle,
        indices=all_real_indices,
        labels=real_labels,
        valid=real_valid,
        device=device,
    )
    wrong_name = f"{primary}__WRONG_OBJECT_DIMS"
    real_probability[wrong_name] = wrong_probability
    real_results[wrong_name] = {
        "trained_model": primary,
        "retrained": False,
        "artifact": selection["artifact_by_arm"][primary],
        "subgroups": base._evaluate_real_parity(
            real_labels, wrong_probability, real_valid, base_cache
        ),
    }
    raw_real_changed = np.any(
        shuffled_real_dimensions != real_dimensions, axis=1
    )
    common_ids = base._common_plastic_frame_ids(base_cache)
    common_mask = np.asarray(
        [str(frame_id) in common_ids for frame_id in base_cache["real_frame_ids"]],
        dtype=np.bool_,
    )
    real_controls = {
        "evaluated_model": primary,
        "artifact": selection["artifact_by_arm"][primary],
        "shuffle": {
            "seed": 42,
            "population_id": "COMMON_DEV_MULTISHAPE_POS",
            "population_n": int(len(control_indices)),
            "permutation_local": real_permutation_local,
            "permutation_global_cache_indices": real_permutation_global.tolist(),
            "different_source_frame_for_every_control_row": bool(
                np.all(
                    np.asarray(real_permutation_local)
                    != np.arange(len(control_indices))
                )
            ),
            "changed_dimension_triplet_n": int(raw_real_changed.sum()),
            "common128_changed_dimension_triplet_n": int(
                np.sum(common_mask & raw_real_changed)
            ),
            "descriptive_not_gate": True,
        },
        "wrong_object_dimensions": {
            "changed_dimension_triplet_n": int(
                np.sum(np.any(wrong_dimensions != real_dimensions, axis=1))
            ),
            "pose_scale": "TRUE_REGISTRY_DIMENSIONS_ONLY",
            "descriptive_not_gate": True,
        },
    }

    selected_by_arm = {
        name: _prediction_vector(probability)
        for name, probability in real_probability.items()
    }
    pose_metrics, pose_bootstrap = base._pose_diagnostic(
        base_cache, selected_by_arm
    )
    phase_a_common = base._phase_a_common_metrics(base_cache)
    a0_auc = float(
        phase_a_common["metrics"]["A0_CURRENT_SELECTOR"][
            "restricted_adds_auc"
        ]
    )
    a1_auc = float(
        phase_a_common["metrics"]["A1_GT_PARITY_ORACLE"][
            "restricted_adds_auc"
        ]
    )
    if a0_auc != 0.27236328125 or a1_auc != 0.32071875:
        raise SpatialFusionError(
            f"FROZEN_PHASE_A_AUC_BINDING_MISMATCH: A0={a0_auc} A1={a1_auc}"
        )

    gates, gate_details = _build_gates_e1_to_e8(
        primary=primary,
        synthetic_test=synthetic_test,
        synthetic_controls=synthetic_controls,
        real_results=real_results,
        pose_metrics=pose_metrics,
        spatial_audit=spatial_audit,
    )
    pre_latency_pass = bool(
        all(gates[f"E{index}"]["passed"] for index in range(1, 9))
    )
    gates["E9"] = _latency_gate(
        pre_latency_passed=pre_latency_pass,
        s0_model=real_models["S0_SPATIAL_NO_DIMS"],
        primary_model=real_models[primary],
        real_bundle=real_bundle,
        real_valid=real_valid,
        device=device,
    )
    gates["E1_to_E8_passed"] = pre_latency_pass
    gates["passed"] = bool(
        pre_latency_pass and gates["E9"]["passed"]
    )

    comparison_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        test = synthetic_test[arm]
        row: dict[str, Any] = {
            "arm": arm,
            "method_score_three_seed_dev_mean": selection[
                "method_scores_three_seed_dev_mean"
            ][arm],
            "artifact_seed": selection["artifact_by_arm"][arm]["seed"],
            "artifact_dev_balanced_accuracy": selection["artifact_by_arm"][arm][
                "synthetic_dev_balanced_accuracy"
            ],
            "primary_method": arm == primary,
            "synth_test_correct_n": test["correct_n"],
            "synth_test_n": test["all_frame_with_abstention"]["N"],
            "synth_test_accuracy": test["all_frame_with_abstention"]["accuracy"],
            "synth_test_balanced_accuracy": test[
                "all_frame_with_abstention"
            ]["balanced_accuracy"],
            "synth_test_auroc": test["all_frame_with_abstention"]["auroc"],
            "real_evaluated": arm in {"S0_SPATIAL_NO_DIMS", primary},
        }
        if arm in synthetic_control_arms:
            row["synth_shuffled_correct_n"] = synthetic_control_arms[arm][
                "correct_n"
            ]
            row["synth_shuffled_accuracy"] = synthetic_control_arms[arm][
                "all_frame_with_abstention"
            ]["accuracy"]
        if row["real_evaluated"]:
            groups = real_results[arm]["subgroups"]
            row["real_common128_correct_n"] = _correct_count(
                groups["PLASTIC:COMMON128:ALL"]
            )
            row["real_common128_accuracy"] = groups[
                "PLASTIC:COMMON128:ALL"
            ]["accuracy"]
            row["real_night_correct_n"] = _correct_count(
                groups["PLASTIC:COMMON128:NIGHT"]
            )
            row["real_night_accuracy"] = groups[
                "PLASTIC:COMMON128:NIGHT"
            ]["accuracy"]
        comparison_rows.append(row)

    if (
        _sha256(PRIMARY_WEIGHTS) != checkpoint_sha_before
        or PRIMARY_WEIGHTS.stat().st_mtime_ns != checkpoint_mtime_before
    ):
        raise SpatialFusionError("YOLO_CHECKPOINT_CHANGED_DURING_SPATIAL_PROBE")
    final_source_bindings = _verify_source_bindings(contract)
    if final_source_bindings != source_bindings:
        raise SpatialFusionError("BOUND_SOURCES_CHANGED_DURING_SPATIAL_PROBE")
    if _verify_protocol_freeze().get("runner_sha256") != _sha256(RUNNER_PATH):
        raise SpatialFusionError("RUNNER_CHANGED_DURING_SPATIAL_PROBE")

    real_payload = {
        "schema_version": "dimension_conditioning_real_spatial_fusion_results_v1",
        "generated_at_utc": _now(),
        "role": "ADAPTIVELY_REUSED_DEV_DIAGNOSTIC_NOT_PROMOTION_ELIGIBLE",
        "irrevocably_exploratory": True,
        "selection_lock": {
            "path": _display(SELECTION_LOCK),
            "sha256": selection_lock_sha,
        },
        "real_used_for_training_or_selection": 0,
        "evaluated_arms": [
            "S0_SPATIAL_NO_DIMS",
            primary,
            shuffled_name,
            wrong_name,
        ],
        "losing_conditioned_methods_evaluated_on_real": [],
        "arms": real_results,
    }
    pose_payload = {
        "schema_version": "dimension_conditioning_spatial_fusion_pnp_v1",
        "generated_at_utc": _now(),
        "role": "ADAPTIVELY_REUSED_DEV_DIAGNOSTIC_NOT_PROMOTION_ELIGIBLE",
        "irrevocably_exploratory": True,
        "plastic_metrics": pose_metrics,
        "plastic_session_cluster_bootstrap_1000": pose_bootstrap,
        "phase_a_common128_reference": phase_a_common,
        "wrong_and_shuffled_controls_use_true_registry_dimensions_for_pnp": True,
        "wood_pose_status": "BLOCKED",
        "wood_pose_blocked_reasons": [
            "WOOD_SYMMETRY_UNREVIEWED",
            "WOOD_INTRINSICS_SENSOR_PROFILE_SCALED_NOT_APPROVED",
        ],
    }
    controls_payload = {
        "schema_version": "dimension_conditioning_spatial_fusion_controls_v1",
        "generated_at_utc": _now(),
        "source_selection_lock": {
            "path": _display(SELECTION_LOCK),
            "sha256": selection_lock_sha,
        },
        "synthetic_test": synthetic_controls,
        "real_primary_only": real_controls,
        "wrong_dimensions_enter_parity_branch_only_not_pnp_scale": True,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "verdict": "EXPLORATORY_SPATIAL_FUSION_DIAGNOSTIC_COMPLETE",
        "irrevocably_exploratory": True,
        "promotion_eligible": False,
        "paper_final_table_eligible": False,
        "full_yolo_training_authorized": False,
        "wrapper_integration_authorized": False,
        "deployment_authorized": False,
        "repo_head": _repo_head(),
        "source_selection_lock": {
            "path": _display(SELECTION_LOCK),
            "sha256": selection_lock_sha,
        },
        "source_selection": selection,
        "step_zero_audit_by_seed": step_zero_by_seed,
        "synthetic_test_locked_median_artifacts": synthetic_test,
        "real_evaluated_arms": real_payload["evaluated_arms"],
        "real_losing_conditioned_methods_evaluated": [],
        "diagnostic_gates": gates,
        "gate_details": gate_details,
        "yolo_parameter_updates": 0,
        "yolo_optimizer_membership": 0,
        "probe_training_runs": len(ARMS) * len(SEEDS),
        "full_yolo_training_runs": 0,
        "checkpoint_sha256_before_after": [
            checkpoint_sha_before,
            _sha256(PRIMARY_WEIGHTS),
        ],
        "checkpoint_mtime_unchanged": (
            PRIMARY_WEIGHTS.stat().st_mtime_ns == checkpoint_mtime_before
        ),
        "source_bindings_before_after_exact": True,
        "scope_limitations": [
            "SYNTHETIC_TEST_HISTORICALLY_OPENED_AND_UPSTREAM_EXPOSED",
            "COMMON128_HISTORICALLY_OPENED_IN_PHASE_C",
            "CURRENT_SPLITS_ADAPTIVELY_REUSED_DEV_DIAGNOSTIC",
            "WOOD_POSE_BLOCKED",
            "NO_MULTISHAPE_GENERALIZATION_CLAIM",
        ],
    }
    report = _report_markdown(summary)
    _write_json_exclusive(REAL_RESULTS, real_payload)
    _write_json_exclusive(PNP_RESULTS, pose_payload)
    _write_json_exclusive(CONTROL_RESULTS, controls_payload)
    _write_text_exclusive(COMPARISON_CSV, _spatial_comparison_csv(comparison_rows))
    _write_text_exclusive(REPORT_PATH, report)
    summary["artifacts"] = {
        "real_results": {
            "path": _display(REAL_RESULTS),
            "sha256": _sha256(REAL_RESULTS),
        },
        "pnp_diagnostic": {
            "path": _display(PNP_RESULTS),
            "sha256": _sha256(PNP_RESULTS),
        },
        "controls": {
            "path": _display(CONTROL_RESULTS),
            "sha256": _sha256(CONTROL_RESULTS),
        },
        "comparison_csv": {
            "path": _display(COMPARISON_CSV),
            "sha256": _sha256(COMPARISON_CSV),
        },
        "report": {"path": _display(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    }
    _write_json_exclusive(SUMMARY_PATH, summary)
    for model in real_models.values():
        del model
    return summary


def _unit_smoke(device: str) -> dict[str, Any]:
    _seed_everything(0)
    patches = torch.randn(4, PATCH_TOKENS, TOKEN_DIM, device=device)
    mask = torch.ones(4, PATCH_TOKENS, dtype=torch.bool, device=device)
    mask[:, :3] = False
    levels = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
    kp = torch.randn(4, KP_DIM, device=device)
    dims = torch.randn(4, DIMENSION_DIM, device=device)
    batch = (patches, mask, levels, kp, dims)
    models: dict[str, SpatialFusionProbe] = {}
    for arm in DIMENSION_AWARE_ARMS:
        _seed_everything(0)
        models[arm] = SpatialFusionProbe(arm).to(device)
    audit = _step_zero_audit(models, batch)
    raw = np.asarray(
        [[1.0, 0.1, 1.2], [0.8, 0.2, 0.6], [1.1, 0.1, 1.3], [0.9, 0.2, 0.7]],
        dtype=np.float32,
    )
    shuffled, permutation = _shuffled_dimensions(
        raw, seed=42, require_all_triplets_changed=True
    )
    return {
        "step_zero": audit,
        "shuffle_all_changed": bool(np.all(np.any(shuffled != raw, axis=1))),
        "permutation": permutation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("extract", "train", "all", "unit-smoke")
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"extract", "all"}:
        extract_spatial_features(str(args.device))
    if args.command in {"train", "all"}:
        train_and_evaluate(str(args.device))
    if args.command == "unit-smoke":
        print(_json_text(_unit_smoke(str(args.device))), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
