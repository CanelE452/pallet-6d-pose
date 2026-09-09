"""Immutable, frozen-cache architecture extension for the clean-start 60k probe.

This runner only trains tiny post-hoc branches over the already frozen 60k
feature cache.  It never re-extracts features, updates YOLO, opens an
independent TEST set, or writes a paper FINAL result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
REPO = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).resolve()
CONTRACT = HERE / "ARCHITECTURE_EXTENSION_CONTRACT.json"
BASE_RUNNER = BASE_DIR / "run_spatial_probe_from_cache.py"
PHASE_E_RUNNER = (
    REPO
    / "challenge/yolo_pose_one_model/dimension_conditioning_probe/"
    "phase_e_spatial_fusion/run_spatial_fusion.py"
)
FEATURE_CACHE = BASE_DIR / "SPATIAL_FEATURE_CACHE_60K.npz"
FEATURE_AUDIT = BASE_DIR / "SPATIAL_FEATURE_CACHE_60K_AUDIT.json"
PROBE_METADATA = BASE_DIR / "PROBE_METADATA_60K.jsonl"
PROBE_METADATA_AUDIT = BASE_DIR / "PROBE_METADATA_60K_AUDIT.json"
RECOVERY_CONTRACT = BASE_DIR / "PROBE_TRAINING_RECOVERY_CONTRACT.json"
BASE_SUMMARY = BASE_DIR / "SPATIAL_CONCAT_60K_SUMMARY.json"
BASE_PROBE_DIR = BASE_DIR / "probe_runs"

RUN_DIR = HERE / "runs"
PREDICTIONS = HERE / "ARCHITECTURE_EXTENSION_DEV_PREDICTIONS.npz"
SUMMARY = HERE / "ARCHITECTURE_EXTENSION_SUMMARY.json"
REPORT = HERE / "ARCHITECTURE_EXTENSION_REPORT.md"
LATENCY = HERE / "CUDA_BRANCH_LATENCY.json"

SCHEMA = "cleanstart_60k_architecture_extension_v1_exploratory"
RUN_SCHEMA = "cleanstart_60k_architecture_extension_run_v1"
CHECKPOINT_SCHEMA = "cleanstart_60k_architecture_extension_checkpoint_v1"
ARMS = (
    "S2_SPATIAL_FILM",
    "S3_SPATIAL_CROSS_ATTENTION",
    "D0_DIMS_ONLY",
    "K0_KP_ONLY",
    "C1_CENTER_CONCAT",
)
BASELINE_ARM = "S0_SPATIAL_NO_DIMS"
REFERENCE_ARM = "S1_SPATIAL_CONCAT"
FROZEN_REFERENCE_ARMS = (BASELINE_ARM, REFERENCE_ARM)
CONDITIONED_ARMS = (
    REFERENCE_ARM,
    "S2_SPATIAL_FILM",
    "S3_SPATIAL_CROSS_ATTENTION",
    "D0_DIMS_ONLY",
    "C1_CENTER_CONCAT",
)
SEEDS = (0, 1, 2)
TRAIN_BATCH = 512
TRAIN_STEPS = 40
TRAIN_EPOCHS = 40
EVAL_BATCH = 1024
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
PATCH_TOKENS = 49
TOKEN_DIM = 64
KP_DIM = 26
DIM_DIM = 4
CENTER_INDEX = 24
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260830
LATENCY_WARMUPS = 20
LATENCY_ITERATIONS = 200


class ExtensionError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExtensionError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base: Any | None = None


def base() -> Any:
    global _base
    if _base is None:
        # The frozen base runner imports OpenCV for its extraction-only path.
        # This extension never extracts images, so allow lightweight probe-only
        # environments to import it without silently enabling that path.
        try:
            import cv2 as _cv2  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("cv2")

            def unavailable_imread(*_args: Any, **_kwargs: Any) -> Any:
                raise ExtensionError("OpenCV unavailable; extraction is forbidden here")

            stub.imread = unavailable_imread  # type: ignore[attr-defined]
            sys.modules["cv2"] = stub
        _base = load_module("cleanstart_60k_frozen_base_runner", BASE_RUNNER)
    return _base


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtensionError(f"unreadable JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtensionError(f"JSON object required: {path}")
    return value


def publish_temporary_exclusive(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_text_exclusive(path: Path, text: str) -> None:
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
        publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise ExtensionError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_json_exclusive(path: Path, payload: Any) -> None:
    write_text_exclusive(
        path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def save_torch_exclusive(path: Path, payload: Any) -> None:
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
        publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise ExtensionError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def save_npz_exclusive(path: Path, **arrays: Any) -> None:
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
        publish_temporary_exclusive(temporary, path)
    except FileExistsError as exc:
        raise ExtensionError(f"artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def reference_run_path(arm: str, seed: int, suffix: str) -> Path:
    return BASE_PROBE_DIR / arm / f"seed{seed}.{suffix}"


def observed_source_bindings() -> dict[str, str]:
    paths: dict[str, Path] = {
        "feature_cache_sha256": FEATURE_CACHE,
        "feature_audit_sha256": FEATURE_AUDIT,
        "probe_metadata_sha256": PROBE_METADATA,
        "probe_metadata_audit_sha256": PROBE_METADATA_AUDIT,
        "probe_training_recovery_contract_sha256": RECOVERY_CONTRACT,
        "base_probe_runner_sha256": BASE_RUNNER,
        "phase_e_runner_sha256": PHASE_E_RUNNER,
        "base_spatial_summary_sha256": BASE_SUMMARY,
    }
    for arm in ("S0_SPATIAL_NO_DIMS", REFERENCE_ARM):
        for seed in SEEDS:
            paths[f"{arm}_seed{seed}_run_sha256"] = reference_run_path(
                arm, seed, "json"
            )
            paths[f"{arm}_seed{seed}_checkpoint_sha256"] = reference_run_path(
                arm, seed, "pt"
            )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ExtensionError(f"required frozen sources missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def validate_contract() -> dict[str, Any]:
    if not CONTRACT.is_file():
        raise ExtensionError(f"contract missing: {CONTRACT}")
    contract = read_json(CONTRACT)
    expected_training = {
        "seeds": list(SEEDS),
        "epochs": TRAIN_EPOCHS,
        "steps_per_epoch": TRAIN_STEPS,
        "batch": TRAIN_BATCH,
        "samples_per_epoch_with_replacement": TRAIN_STEPS * TRAIN_BATCH,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "scheduler": "none",
        "gradient_clipping": "none",
        "amp": False,
        "class_balanced_sampler": True,
        "loss": "balanced_cross_entropy",
        "checkpoint_selection": (
            "earliest maximum mixed synthetic DEV balanced accuracy"
        ),
    }
    expected_evaluation = {
        "population": "locked mixed synthetic DEV 4020 including one abstention as error",
        "artifact_lock": "median selected-epoch DEV balanced accuracy; tie lower seed",
        "conditioned_arm_controls": "correct and shared seed-42 all-triplets-changed shuffled fixed XYZ dimensions",
        "per_source": ["G38", "P0", "TEX"],
        "seed_summary": "mean and sample standard deviation over seeds 0,1,2",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired cluster percentile 95% CI for DEV balanced-accuracy delta vs S1",
            "clusters": "G38 pair_group_id singleton; matched P0/TEX pair_group_id joint cluster",
        },
        "optional_latency": {
            "scope": "branch-only CUDA batch=1",
            "warmups": LATENCY_WARMUPS,
            "iterations": LATENCY_ITERATIONS,
            "statistics": ["p50_ms", "p95_ms"],
        },
    }
    expected_schedule: dict[str, str] = {}
    for seed in SEEDS:
        s0 = read_json(reference_run_path("S0_SPATIAL_NO_DIMS", seed, "json"))
        s1 = read_json(reference_run_path(REFERENCE_ARM, seed, "json"))
        left = str(s0.get("sampler_schedule_sha256"))
        right = str(s1.get("sampler_schedule_sha256"))
        if left != right:
            raise ExtensionError(f"frozen S0/S1 schedule mismatch for seed {seed}")
        expected_schedule[str(seed)] = left
    if (
        contract.get("schema_version") != SCHEMA
        or tuple(contract.get("arms", ())) != ARMS
        or tuple(contract.get("frozen_reference_arms", ())) != FROZEN_REFERENCE_ARMS
        or contract.get("reference_arm") != REFERENCE_ARM
        or contract.get("training") != expected_training
        or contract.get("evaluation") != expected_evaluation
        or contract.get("reference_sampler_schedule_sha256_by_seed")
        != expected_schedule
        or contract.get("dimension_input")
        != "fixed_renderer_dimensions_m_xyz_transformed_to_frozen_4d_phase_c_feature"
        or contract.get("camera_facing_dimensions_model_input_forbidden") is not True
        or contract.get("feature_cache_frozen") is not True
        or contract.get("yolo_parameter_updates") != 0
        or contract.get("independent_test_opened") is not False
        or contract.get("paper_final_claim_allowed") is not False
        or contract.get("irrevocably_exploratory") is not True
        or contract.get("source_bindings") != observed_source_bindings()
        or contract.get("runner_sha256") != sha256(RUNNER)
    ):
        raise ExtensionError("architecture-extension contract mismatch")
    metadata_audit = read_json(PROBE_METADATA_AUDIT)
    if (
        metadata_audit.get("allowed_dimension_input")
        != "fixed_renderer_dimensions_m_xyz_model_input"
        or metadata_audit.get("camera_facing_whd_parity_agreement", {}).get(
            "model_input_forbidden"
        )
        is not True
        or metadata_audit.get("fixed_xyz_reconstructed_with_perm_v4") != 60_000
    ):
        raise ExtensionError("fixed-XYZ dimension provenance audit mismatch")
    return contract


def runtime_bindings(cache_metadata: Mapping[str, Any]) -> dict[str, str]:
    contract = validate_contract()
    return {
        **dict(contract["source_bindings"]),
        "architecture_extension_contract_sha256": sha256(CONTRACT),
        "architecture_extension_runner_sha256": sha256(RUNNER),
        "feature_recipe_sha256": str(cache_metadata["feature_recipe"]["sha256"]),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_device(value: str) -> str:
    text = str(value).strip().lower()
    if text.isdigit():
        text = f"cuda:{text}"
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise ExtensionError(f"CUDA requested but unavailable: {text}")
    try:
        torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise ExtensionError(f"invalid device: {value}") from exc
    return text


class ModalityOnlyProbe(nn.Module):
    """Exactly input -> 128 -> 32 -> 2 with SiLU between linear layers."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in ("D0_DIMS_ONLY", "K0_KP_ONLY"):
            raise ExtensionError(f"invalid modality-only arm: {arm}")
        self.arm = arm
        input_dim = DIM_DIM if arm == "D0_DIMS_ONLY" else KP_DIM
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        source_levels: Tensor,
        kp_features: Tensor,
        dimension_features: Tensor,
    ) -> Tensor:
        del patches, patch_mask, source_levels
        features = dimension_features if self.arm == "D0_DIMS_ONLY" else kp_features
        expected = DIM_DIM if self.arm == "D0_DIMS_ONLY" else KP_DIM
        if features.ndim != 2 or features.shape[1] != expected:
            raise ExtensionError(
                f"{self.arm} input shape invalid: {tuple(features.shape)}"
            )
        return self.classifier(features)


class CenterConcatProbe(nn.Module):
    """Phase-E S1 parameterization, but only the exact center token is pooled."""

    def __init__(self):
        super().__init__()
        reference = base().phase_e().SpatialFusionProbe(REFERENCE_ARM)
        self.token_encoder = reference.token_encoder
        self.position_embedding = reference.position_embedding
        self.source_level_embedding = reference.source_level_embedding
        self.classifier = reference.classifier

    def forward(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        source_levels: Tensor,
        kp_features: Tensor,
        dimension_features: Tensor,
    ) -> Tensor:
        batch_size = patches.shape[0]
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (
            PATCH_TOKENS,
            TOKEN_DIM,
        ):
            raise ExtensionError(f"C1 patch shape invalid: {tuple(patches.shape)}")
        if tuple(patch_mask.shape) != (batch_size, PATCH_TOKENS):
            raise ExtensionError("C1 patch-mask shape invalid")
        if not bool(torch.all(patch_mask[:, CENTER_INDEX])):
            raise ExtensionError("C1 center token must be valid")
        if tuple(source_levels.shape) != (batch_size,):
            raise ExtensionError("C1 source-level shape invalid")
        if tuple(kp_features.shape) != (batch_size, KP_DIM):
            raise ExtensionError("C1 keypoint-feature shape invalid")
        if tuple(dimension_features.shape) != (batch_size, DIM_DIM):
            raise ExtensionError("C1 dimension-feature shape invalid")
        token = self.token_encoder(patches[:, CENTER_INDEX, :])
        token = token + self.position_embedding[CENTER_INDEX].unsqueeze(0)
        token = token + self.source_level_embedding(source_levels.long())
        pooled = torch.cat([token, token], dim=1)
        return self.classifier(
            torch.cat([pooled, kp_features, dimension_features], dim=1)
        )


def make_model(arm: str) -> nn.Module:
    if arm in (
        *FROZEN_REFERENCE_ARMS,
        "S2_SPATIAL_FILM",
        "S3_SPATIAL_CROSS_ATTENTION",
    ):
        return base().phase_e().SpatialFusionProbe(arm)
    if arm in ("D0_DIMS_ONLY", "K0_KP_ONLY"):
        return ModalityOnlyProbe(arm)
    if arm == "C1_CENTER_CONCAT":
        return CenterConcatProbe()
    raise ExtensionError(f"unknown arm: {arm}")


def model_config(arm: str) -> dict[str, Any]:
    if arm in (
        *FROZEN_REFERENCE_ARMS,
        "S2_SPATIAL_FILM",
        "S3_SPATIAL_CROSS_ATTENTION",
    ):
        return {
            "implementation": "exact Phase-E SpatialFusionProbe",
            **base().phase_e()._model_config(arm),
        }
    if arm in ("D0_DIMS_ONLY", "K0_KP_ONLY"):
        input_dim = DIM_DIM if arm == "D0_DIMS_ONLY" else KP_DIM
        return {
            "arm": arm,
            "implementation": "modality-only MLP",
            "input": "fixed_xyz_4d" if arm == "D0_DIMS_ONLY" else "kp_26d",
            "layers": [input_dim, 128, 32, 2],
            "activation": "SiLU",
            "dropout": 0.0,
        }
    if arm == "C1_CENTER_CONCAT":
        return {
            "arm": arm,
            "implementation": "Phase-E common modules with center-cell-only forward",
            "center_index": CENTER_INDEX,
            "token_encoder": [TOKEN_DIM, TOKEN_DIM],
            "position_embedding": "Phase-E 49x64 bank; only row 24 is read",
            "source_level_embedding": True,
            "pool": "center_token_concatenated_with_itself",
            "classifier_input_dim": 2 * TOKEN_DIM + KP_DIM + DIM_DIM,
            "classifier_hidden": [128, 32],
            "activation": "SiLU",
            "dropout": 0.0,
        }
    raise ExtensionError(f"unknown model config arm: {arm}")


def parameter_counts(arm: str) -> dict[str, Any]:
    seed_everything(0)
    model = make_model(arm)
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    result: dict[str, Any] = {"total": total, "trainable": trainable}
    if arm in ("S2_SPATIAL_FILM", "S3_SPATIAL_CROSS_ATTENTION"):
        result["fusion_adapter"] = int(base().phase_e()._adapter_parameter_count(model))
    if arm == "C1_CENTER_CONCAT":
        result["unused_position_embedding_rows"] = PATCH_TOKENS - 1
        result["note"] = (
            "full Phase-E position bank retained for parameter/initialization matching; "
            "forward reads center row only"
        )
    return result


def run_paths(arm: str, seed: int) -> tuple[Path, Path]:
    root = RUN_DIR / arm
    return root / f"seed{seed}.pt", root / f"seed{seed}.json"


def expected_reference_schedules(contract: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(seed): str(value)
        for seed, value in contract["reference_sampler_schedule_sha256_by_seed"].items()
    }


def precompute_schedule_hash(
    seed: int, labels: np.ndarray, train_indices: np.ndarray
) -> str:
    train_y = np.asarray(labels[train_indices], dtype=np.int64)
    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    class_weights = len(train_y) / (2.0 * counts)
    sample_weights = class_weights[train_y]
    digest = hashlib.sha256()
    for epoch in range(TRAIN_EPOCHS):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + epoch)
        local = torch.multinomial(
            torch.from_numpy(sample_weights),
            num_samples=TRAIN_STEPS * TRAIN_BATCH,
            replacement=True,
            generator=generator,
        ).numpy()
        selected = train_indices[local]
        digest.update(np.ascontiguousarray(selected, dtype=np.int32).tobytes())
    return digest.hexdigest()


def validate_existing_run(
    arm: str, seed: int, bindings: Mapping[str, str]
) -> dict[str, Any] | None:
    checkpoint, run_json = run_paths(arm, seed)
    if not checkpoint.exists() and not run_json.exists():
        return None
    if run_json.exists() and not checkpoint.is_file():
        raise ExtensionError(f"run JSON without checkpoint: {arm} seed{seed}")
    if checkpoint.is_file() and not run_json.exists():
        # A power loss after the first atomic publish is recoverable because the
        # complete JSON body is embedded in the checkpoint.
        payload = torch.load(checkpoint, map_location="cpu")
        recovered = payload.get("run_payload_without_checkpoint")
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA
            or payload.get("source_bindings") != dict(bindings)
            or not isinstance(recovered, dict)
        ):
            raise ExtensionError(f"unrecoverable partial checkpoint: {checkpoint}")
        recovered = dict(recovered)
        recovered["checkpoint"] = {
            "path": str(checkpoint.relative_to(REPO)),
            "sha256": sha256(checkpoint),
        }
        write_json_exclusive(run_json, recovered)
    run = read_json(run_json)
    payload = torch.load(checkpoint, map_location="cpu")
    if (
        run.get("schema_version") != RUN_SCHEMA
        or run.get("arm") != arm
        or int(run.get("seed", -1)) != seed
        or run.get("model_config") != model_config(arm)
        or run.get("source_bindings") != dict(bindings)
        or run.get("checkpoint", {}).get("sha256") != sha256(checkpoint)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("arm") != arm
        or int(payload.get("seed", -1)) != seed
        or payload.get("model_config") != model_config(arm)
        or payload.get("source_bindings") != dict(bindings)
        or int(payload.get("selected_epoch", -1)) != int(run.get("selected_epoch", -2))
    ):
        raise ExtensionError(f"existing run binding mismatch: {arm} seed{seed}")
    return run


def predict(
    model: nn.Module,
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
            logits = model(*base().batch_from_bundle(values, selected, device))
            output[start : start + len(selected)] = (
                torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            )
    return output


def train_one(
    arm: str,
    seed: int,
    values: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    dev_indices: np.ndarray,
    device: str,
    bindings: Mapping[str, str],
    expected_schedule: str,
) -> dict[str, Any]:
    existing = validate_existing_run(arm, seed, bindings)
    if existing is not None:
        if existing.get("sampler_schedule_sha256") != expected_schedule:
            raise ExtensionError(f"existing schedule mismatch: {arm} seed{seed}")
        return existing

    labels = np.asarray(arrays["labels"], dtype=np.int64)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    precomputed = precompute_schedule_hash(seed, labels, train_indices)
    if precomputed != expected_schedule:
        raise ExtensionError(
            f"precomputed sampler schedule differs from frozen S0/S1: seed{seed}"
        )
    seed_everything(seed)
    model = make_model(arm).to(device)
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
    dev_valid_indices = dev_indices[valid[dev_indices]]
    dev_positions = {int(index): pos for pos, index in enumerate(dev_indices.tolist())}
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    schedule_digest = hashlib.sha256()
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
        losses: list[float] = []
        for step in range(TRAIN_STEPS):
            indices = selected[step * TRAIN_BATCH : (step + 1) * TRAIN_BATCH]
            target = torch.from_numpy(labels[indices]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*base().batch_from_bundle(values, indices, device))
            loss = F.cross_entropy(logits, target, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        dev_probability = np.full(len(dev_indices), np.nan, dtype=np.float64)
        observed = predict(model, values, dev_valid_indices, device)
        for index, probability in zip(dev_valid_indices.tolist(), observed.tolist()):
            dev_probability[dev_positions[index]] = probability
        metrics = base().binary_metrics(
            labels[dev_indices], dev_probability, valid[dev_indices]
        )
        score = float(metrics["balanced_accuracy"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "dev_balanced_accuracy": score,
                "dev_accuracy": metrics["accuracy"],
                "dev_coverage": metrics["coverage"],
            }
        )
        if best is None or score > float(best["score"]):
            best = {
                "score": score,
                "epoch": epoch + 1,
                "state_dict": copy.deepcopy(
                    {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
                ),
                "dev": metrics,
            }
    if best is None:
        raise ExtensionError(f"training produced no checkpoint: {arm} seed{seed}")
    observed_schedule = schedule_digest.hexdigest()
    if observed_schedule != expected_schedule:
        raise ExtensionError(f"executed schedule mismatch: {arm} seed{seed}")
    checkpoint, run_json = run_paths(arm, seed)
    run_without_checkpoint = {
        "schema_version": RUN_SCHEMA,
        "created_at_utc": now(),
        "scope": "exploratory frozen-cache DEV; no independent TEST or paper FINAL",
        "arm": arm,
        "seed": seed,
        "model_config": model_config(arm),
        "parameter_counts": parameter_counts(arm),
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
        "sampler_schedule_sha256": observed_schedule,
        "sampler_schedule_exact_to_frozen_s0_s1": True,
        "checkpoint_epoch_selected_by": (
            "earliest maximum mixed synthetic DEV balanced accuracy"
        ),
        "selected_epoch": int(best["epoch"]),
        "dev": best["dev"],
        "history": history,
        "real_accessed_during_training_or_selection": False,
        "independent_test_accessed": False,
        "yolo_parameters_in_optimizer": 0,
    }
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "arm": arm,
        "seed": seed,
        "selected_epoch": int(best["epoch"]),
        "model_config": model_config(arm),
        "source_bindings": dict(bindings),
        "state_dict": best["state_dict"],
        "run_payload_without_checkpoint": run_without_checkpoint,
    }
    save_torch_exclusive(checkpoint, checkpoint_payload)
    run = dict(run_without_checkpoint)
    run["checkpoint"] = {
        "path": str(checkpoint.relative_to(REPO)),
        "sha256": sha256(checkpoint),
    }
    write_json_exclusive(run_json, run)
    return run


def median_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(runs) != 3:
        raise ExtensionError("median lock requires exactly three seeds")
    values = sorted(float(run["dev"]["balanced_accuracy"]) for run in runs)
    median = values[1]
    matches = [run for run in runs if float(run["dev"]["balanced_accuracy"]) == median]
    return min(matches, key=lambda run: int(run["seed"]))


def load_extension_model(
    arm: str, checkpoint: Path, device: str, bindings: Mapping[str, str]
) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu")
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("arm") != arm
        or payload.get("model_config") != model_config(arm)
        or payload.get("source_bindings") != dict(bindings)
    ):
        raise ExtensionError(f"checkpoint binding mismatch: {checkpoint}")
    model = make_model(arm)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def load_frozen_reference_model(
    arm: str, device: str
) -> tuple[nn.Module, dict[str, Any]]:
    if arm not in FROZEN_REFERENCE_ARMS:
        raise ExtensionError(f"not a frozen reference arm: {arm}")
    summary = read_json(BASE_SUMMARY)
    locked = summary["locked_median_artifacts"][arm]
    checkpoint = REPO / locked["checkpoint"]["path"]
    if sha256(checkpoint) != locked["checkpoint"]["sha256"]:
        raise ExtensionError(f"frozen {arm} checkpoint hash mismatch")
    model = base().load_probe(checkpoint, device, summary["source_bindings"])
    return model, locked


def step_zero_audit(
    values: Mapping[str, np.ndarray] | None,
    train_indices: np.ndarray | None,
    device: str,
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for seed in SEEDS:
        models: dict[str, nn.Module] = {}
        for arm in (REFERENCE_ARM, "S2_SPATIAL_FILM", "S3_SPATIAL_CROSS_ATTENTION"):
            seed_everything(seed)
            models[arm] = make_model(arm).to(device)
        if values is None or train_indices is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(99)
            patches = torch.randn(8, PATCH_TOKENS, TOKEN_DIM, generator=generator).to(device)
            masks = torch.ones(8, PATCH_TOKENS, dtype=torch.bool, device=device)
            levels = torch.arange(8, device=device) % 3
            kp = torch.randn(8, KP_DIM, generator=generator).to(device)
            dims = torch.randn(8, DIM_DIM, generator=generator).to(device)
            batch = (patches, masks, levels, kp, dims)
        else:
            selected = train_indices[: min(8, len(train_indices))]
            batch = base().batch_from_bundle(values, selected, device)
        audits[str(seed)] = base().phase_e()._step_zero_audit(models, batch)
        del models
    return audits


def normalized_shuffled_bundle(
    arrays: Mapping[str, np.ndarray],
    cache_metadata: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
    dev_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    shuffled_xyz, permutation = base().phase_e()._shuffled_dimensions(
        np.asarray(arrays["dimensions_xyz"])[dev_indices],
        seed=42,
        require_all_triplets_changed=True,
    )
    dim_raw = base().phase_c()._dimension_raw(shuffled_xyz)
    norm = cache_metadata["normalization"]["dims"]
    normalized = (
        dim_raw - np.asarray(norm["mean"], dtype=np.float32)
    ) / np.asarray(norm["std"], dtype=np.float32)
    shuffled_values = dict(values)
    shuffled_values["dims"] = np.asarray(values["dims"]).copy()
    shuffled_values["dims"][dev_indices] = normalized
    return shuffled_values, np.asarray(permutation, dtype=np.int32)


def load_cluster_ids(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    rows: list[dict[str, Any]] = []
    with PROBE_METADATA.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if len(rows) != 60_000:
        raise ExtensionError("probe metadata row count changed")
    frame_ids = np.asarray(arrays["frame_ids"])
    observed = np.asarray([row.get("merged_stem") for row in rows])
    if not np.array_equal(frame_ids, observed):
        raise ExtensionError("probe metadata/cache row order mismatch")
    cluster_ids: list[str] = []
    for row in rows:
        source = str(row["source"])
        pair = str(row["pair_group_id"])
        prefix = "P0_TEX" if source in ("P0", "TEX") else source
        cluster_ids.append(f"{prefix}::{pair}")
    return np.asarray(cluster_ids)


def cluster_bootstrap_delta(
    labels: np.ndarray,
    valid: np.ndarray,
    cluster_ids: np.ndarray,
    probability: np.ndarray,
    reference_probability: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid, dtype=np.bool_)
    cluster_ids = np.asarray(cluster_ids)
    unique, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_count = len(unique)
    denominators = np.zeros((cluster_count, 2), dtype=np.int32)
    arm_correct = np.zeros((cluster_count, 2), dtype=np.int32)
    reference_correct = np.zeros((cluster_count, 2), dtype=np.int32)
    arm_prediction = np.full(len(labels), -1, dtype=np.int8)
    ref_prediction = np.full(len(labels), -1, dtype=np.int8)
    arm_usable = valid & np.isfinite(probability)
    ref_usable = valid & np.isfinite(reference_probability)
    arm_prediction[arm_usable] = (probability[arm_usable] >= 0.5).astype(np.int8)
    ref_prediction[ref_usable] = (
        reference_probability[ref_usable] >= 0.5
    ).astype(np.int8)
    for truth in (0, 1):
        truth_mask = labels == truth
        denominators[:, truth] = np.bincount(
            inverse[truth_mask], minlength=cluster_count
        )
        arm_correct[:, truth] = np.bincount(
            inverse[truth_mask & (arm_prediction == truth)], minlength=cluster_count
        )
        reference_correct[:, truth] = np.bincount(
            inverse[truth_mask & (ref_prediction == truth)], minlength=cluster_count
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    cursor = 0
    chunk = 128
    while cursor < BOOTSTRAP_REPLICATES:
        count = min(chunk, BOOTSTRAP_REPLICATES - cursor)
        draw = rng.integers(0, cluster_count, size=(count, cluster_count))
        denom = denominators[draw].sum(axis=1)
        arm_numerator = arm_correct[draw].sum(axis=1)
        ref_numerator = reference_correct[draw].sum(axis=1)
        if np.any(denom == 0):
            raise ExtensionError("bootstrap replicate omitted a label class")
        arm_ba = 0.5 * np.sum(arm_numerator / denom, axis=1)
        ref_ba = 0.5 * np.sum(ref_numerator / denom, axis=1)
        deltas[cursor : cursor + count] = arm_ba - ref_ba
        cursor += count
    arm_point = base().binary_metrics(labels, probability, valid)["balanced_accuracy"]
    ref_point = base().binary_metrics(labels, reference_probability, valid)[
        "balanced_accuracy"
    ]
    return {
        "metric": "balanced_accuracy_delta_vs_S1_SPATIAL_CONCAT",
        "point_delta": float(arm_point - ref_point),
        "replicates": BOOTSTRAP_REPLICATES,
        "valid_replicates": int(np.isfinite(deltas).sum()),
        "seed": BOOTSTRAP_SEED,
        "cluster_count": cluster_count,
        "cluster_size_counts": {
            str(size): int(count)
            for size, count in zip(*np.unique(np.bincount(inverse), return_counts=True))
        },
        "p0_tex_scenario_pairing": "joint pair_group_id cluster",
        "ci95_percentile": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "bootstrap_mean_delta": float(np.mean(deltas)),
    }


def full_dev_probability(
    model: nn.Module,
    values: Mapping[str, np.ndarray],
    dev_indices: np.ndarray,
    dev_valid: np.ndarray,
    device: str,
) -> np.ndarray:
    output = np.full(len(dev_indices), np.nan, dtype=np.float64)
    positions = {int(index): pos for pos, index in enumerate(dev_indices.tolist())}
    observed = predict(model, values, dev_valid, device)
    for index, probability in zip(dev_valid.tolist(), observed.tolist()):
        output[positions[index]] = probability
    return output


def prediction_key(name: str) -> str:
    return "probability__" + name


def save_or_load_predictions(
    *,
    bindings: Mapping[str, str],
    locked: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    dev_indices: np.ndarray,
    cluster_ids: np.ndarray,
    probabilities: Mapping[str, np.ndarray] | None,
    permutation: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    bindings_json = json.dumps(bindings, sort_keys=True, separators=(",", ":"))
    locked_json = json.dumps(locked, sort_keys=True, separators=(",", ":"))
    if PREDICTIONS.exists():
        with np.load(PREDICTIONS, allow_pickle=False) as source:
            if (
                str(source["source_bindings_json"].item()) != bindings_json
                or str(source["locked_artifacts_json"].item()) != locked_json
                or not np.array_equal(source["global_indices"], dev_indices)
                or not np.array_equal(source["cluster_ids"], cluster_ids)
            ):
                raise ExtensionError("existing prediction artifact binding mismatch")
            loaded = {
                key.removeprefix("probability__"): np.asarray(source[key])
                for key in source.files
                if key.startswith("probability__")
            }
            loaded_permutation = np.asarray(
                source["shuffled_dimension_source_local_indices"], dtype=np.int32
            )
        return loaded, loaded_permutation
    if probabilities is None or permutation is None:
        raise ExtensionError("prediction artifact absent and inference was not supplied")
    payload: dict[str, Any] = {
        "source_bindings_json": np.asarray(bindings_json),
        "locked_artifacts_json": np.asarray(locked_json),
        "global_indices": np.asarray(dev_indices, dtype=np.int32),
        "frame_ids": np.asarray(arrays["frame_ids"])[dev_indices],
        "sources": np.asarray(arrays["sources"])[dev_indices],
        "labels": np.asarray(arrays["labels"], dtype=np.int8)[dev_indices],
        "valid": np.asarray(arrays["valid"], dtype=np.bool_)[dev_indices],
        "cluster_ids": cluster_ids,
        "shuffled_dimension_source_local_indices": np.asarray(
            permutation, dtype=np.int32
        ),
    }
    for name, value in probabilities.items():
        payload[prediction_key(name)] = np.asarray(value, dtype=np.float64)
    save_npz_exclusive(PREDICTIONS, **payload)
    return dict(probabilities), np.asarray(permutation, dtype=np.int32)


def frozen_reference_seed_runs(arm: str) -> list[dict[str, Any]]:
    if arm not in FROZEN_REFERENCE_ARMS:
        raise ExtensionError(f"not a frozen reference arm: {arm}")
    return [
        read_json(reference_run_path(arm, seed, "json")) for seed in SEEDS
    ]


def seed_statistics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [float(run["dev"]["balanced_accuracy"]) for run in runs], dtype=np.float64
    )
    return {
        "n_seeds": len(values),
        "values_by_seed": {
            str(int(run["seed"])): float(run["dev"]["balanced_accuracy"])
            for run in runs
        },
        "mean_dev_balanced_accuracy": float(np.mean(values)),
        "sample_sd_dev_balanced_accuracy": float(np.std(values, ddof=1)),
    }


def report_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Clean-start 60k architecture extension",
        "",
        "Exploratory frozen-cache DEV only. This is not an independent TEST and must not be copied into a paper FINAL table.",
        "",
        "| Arm | DEV balanced accuracy | Shuffled dimensions | Mean +/- sample SD (3 seeds) | Params | Delta vs S1 (95% cluster CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = summary["dev_metrics"]
    for arm in (*FROZEN_REFERENCE_ARMS, *ARMS):
        shuffled = metrics.get(f"{arm}__SHUFFLED_DIMS", {}).get("balanced_accuracy")
        stats = summary["seed_statistics"][arm]
        ci = summary["paired_cluster_bootstrap_delta_vs_s1"].get(arm)
        ci_text = "reference"
        if ci is not None:
            ci_text = (
                f"{ci['point_delta']:+.6f} "
                f"[{ci['ci95_percentile'][0]:+.6f}, {ci['ci95_percentile'][1]:+.6f}]"
            )
        shuffled_text = "--" if shuffled is None else f"{shuffled:.6f}"
        lines.append(
            f"| {arm} | {metrics[arm]['balanced_accuracy']:.6f} | {shuffled_text} | "
            f"{stats['mean_dev_balanced_accuracy']:.6f} +/- "
            f"{stats['sample_sd_dev_balanced_accuracy']:.6f} | "
            f"{summary['parameter_counts'][arm]['total']} | {ci_text} |"
        )
    lines.extend(
        [
            "",
            "P0 and TEX rows with the same `pair_group_id` are resampled as one paired scenario cluster; each G38 `pair_group_id` is a singleton cluster.",
            "",
        ]
    )
    return "\n".join(lines)


def train_and_evaluate(device: str) -> dict[str, Any]:
    contract = validate_contract()
    if SUMMARY.exists():
        existing = read_json(SUMMARY)
        if (
            existing.get("schema_version") != SCHEMA
            or existing.get("source_bindings", {}).get(
                "architecture_extension_contract_sha256"
            )
            != sha256(CONTRACT)
            or not PREDICTIONS.is_file()
            or existing.get("dev_predictions", {}).get("sha256")
            != sha256(PREDICTIONS)
            or not REPORT.is_file()
            or existing.get("report", {}).get("sha256") != sha256(REPORT)
        ):
            raise ExtensionError("completed summary binding mismatch")
        return existing

    arrays, cache_metadata, feature_audit = base().load_cache()
    values = base().bundle(arrays, cache_metadata)
    splits = np.asarray(arrays["splits"], dtype=np.int8)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    train_indices = np.flatnonzero(splits == 0)
    dev_indices = np.flatnonzero(splits == 1)
    if len(train_indices) != 55_980 or len(dev_indices) != 4_020:
        raise ExtensionError("frozen train/DEV population mismatch")
    if not bool(valid[train_indices].all()) or int((~valid[dev_indices]).sum()) != 1:
        raise ExtensionError("frozen abstention population mismatch")
    bindings = runtime_bindings(cache_metadata)
    schedules = expected_reference_schedules(contract)
    for seed in SEEDS:
        if precompute_schedule_hash(seed, labels, train_indices) != schedules[seed]:
            raise ExtensionError(f"sampler schedule preflight mismatch: seed{seed}")

    step_zero = step_zero_audit(values, train_indices, device)
    runs_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            print(f"training/resuming {arm} seed={seed}", flush=True)
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
                    schedules[seed],
                )
            )
    schedule_equality = {
        str(seed): all(
            runs_by_arm[arm][seed]["sampler_schedule_sha256"] == schedules[seed]
            for arm in ARMS
        )
        for seed in SEEDS
    }
    if not all(schedule_equality.values()):
        raise ExtensionError("extension schedule hash differs from frozen S0/S1")

    locked: dict[str, Any] = {}
    for arm, runs in runs_by_arm.items():
        median = median_run(runs)
        locked[arm] = {
            "median_seed": int(median["seed"]),
            "median_seed_dev": median["dev"],
            "checkpoint": median["checkpoint"],
        }
    baseline_model, baseline_locked = load_frozen_reference_model(
        BASELINE_ARM, device
    )
    reference_model, reference_locked = load_frozen_reference_model(
        REFERENCE_ARM, device
    )
    locked_with_reference = {
        BASELINE_ARM: baseline_locked,
        REFERENCE_ARM: reference_locked,
        **locked,
    }

    all_cluster_ids = load_cluster_ids(arrays)
    dev_cluster_ids = all_cluster_ids[dev_indices]
    cluster_unique, cluster_sizes = np.unique(dev_cluster_ids, return_counts=True)
    if (
        len(cluster_unique) != 3_009
        or int(np.sum(cluster_sizes == 1)) != 1_998
        or int(np.sum(cluster_sizes == 2)) != 1_011
    ):
        raise ExtensionError("P0/TEX scenario cluster audit mismatch")

    probabilities: dict[str, np.ndarray] | None = None
    permutation: np.ndarray | None = None
    if not PREDICTIONS.exists():
        dev_valid = dev_indices[valid[dev_indices]]
        probabilities = {
            BASELINE_ARM: full_dev_probability(
                baseline_model, values, dev_indices, dev_valid, device
            ),
            REFERENCE_ARM: full_dev_probability(
                reference_model, values, dev_indices, dev_valid, device
            )
        }
        extension_models: dict[str, nn.Module] = {}
        for arm in ARMS:
            checkpoint = REPO / locked[arm]["checkpoint"]["path"]
            extension_models[arm] = load_extension_model(
                arm, checkpoint, device, bindings
            )
            probabilities[arm] = full_dev_probability(
                extension_models[arm], values, dev_indices, dev_valid, device
            )
        shuffled_values, permutation = normalized_shuffled_bundle(
            arrays, cache_metadata, values, dev_indices
        )
        probabilities[f"{REFERENCE_ARM}__SHUFFLED_DIMS"] = full_dev_probability(
            reference_model, shuffled_values, dev_indices, dev_valid, device
        )
        for arm in CONDITIONED_ARMS[1:]:
            probabilities[f"{arm}__SHUFFLED_DIMS"] = full_dev_probability(
                extension_models[arm], shuffled_values, dev_indices, dev_valid, device
            )
    probabilities, permutation = save_or_load_predictions(
        bindings=bindings,
        locked=locked_with_reference,
        arrays=arrays,
        dev_indices=dev_indices,
        cluster_ids=dev_cluster_ids,
        probabilities=probabilities,
        permutation=permutation,
    )

    dev_labels = labels[dev_indices]
    dev_valid_mask = valid[dev_indices]
    metrics = {
        name: base().binary_metrics(dev_labels, probability, dev_valid_mask)
        for name, probability in probabilities.items()
    }
    dev_sources = np.asarray(arrays["sources"])[dev_indices]
    per_source: dict[str, dict[str, Any]] = {}
    for name, probability in probabilities.items():
        per_source[name] = {}
        for source in ("G38", "P0", "TEX"):
            selected = dev_sources == source
            per_source[name][source] = base().binary_metrics(
                dev_labels[selected], probability[selected], dev_valid_mask[selected]
            )
    seed_stats = {
        arm: seed_statistics(frozen_reference_seed_runs(arm))
        for arm in FROZEN_REFERENCE_ARMS
    }
    seed_stats.update(
        {arm: seed_statistics(runs) for arm, runs in runs_by_arm.items()}
    )
    bootstraps = {
        arm: cluster_bootstrap_delta(
            dev_labels,
            dev_valid_mask,
            dev_cluster_ids,
            probabilities[arm],
            probabilities[REFERENCE_ARM],
        )
        for arm in (BASELINE_ARM, *ARMS)
    }
    counts = {
        arm: parameter_counts(arm) for arm in (*FROZEN_REFERENCE_ARMS, *ARMS)
    }
    s1_mean = seed_stats[REFERENCE_ARM]["mean_dev_balanced_accuracy"]
    complexity_eligibility = {
        arm: {
            "mean_dev_balanced_accuracy": seed_stats[arm][
                "mean_dev_balanced_accuracy"
            ],
            "required_s1_mean_plus_0_02": s1_mean + 0.02,
            "eligible": bool(
                seed_stats[arm]["mean_dev_balanced_accuracy"] >= s1_mean + 0.02
            ),
        }
        for arm in ("S2_SPATIAL_FILM", "S3_SPATIAL_CROSS_ATTENTION")
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": now(),
        "scope": "exploratory frozen-cache mixed DEV; no independent TEST; no paper FINAL",
        "paper_final_claim_allowed": False,
        "source_bindings": bindings,
        "population": {
            "train": len(train_indices),
            "dev": len(dev_indices),
            "dev_valid": int(dev_valid_mask.sum()),
            "dev_abstentions_counted_as_errors": int((~dev_valid_mask).sum()),
            "dev_clusters": len(cluster_unique),
            "g38_singleton_clusters": int(np.sum(cluster_sizes == 1)),
            "matched_p0_tex_clusters": int(np.sum(cluster_sizes == 2)),
        },
        "dimension_policy": {
            "model_input": (
                "fixed renderer XYZ transformed to frozen Phase-C 4D normalized feature"
            ),
            "camera_facing_dimensions_forbidden": True,
            "shared_shuffle_seed": 42,
            "all_fixed_xyz_triplets_changed": True,
            "shuffle_permutation_sha256": hashlib.sha256(
                np.asarray(permutation, dtype=np.int32).tobytes()
            ).hexdigest(),
        },
        "step_zero_s1_s2_s3": step_zero,
        "sampler_schedule_exact_to_existing_s0_s1_by_seed": schedule_equality,
        "runs": runs_by_arm,
        "locked_median_artifacts": locked_with_reference,
        "seed_statistics": seed_stats,
        "parameter_counts": counts,
        "dev_metrics": metrics,
        "dev_metrics_by_source": per_source,
        "paired_cluster_bootstrap_delta_vs_s1": bootstraps,
        "complexity_eligibility_vs_s1_mean_plus_0_02": complexity_eligibility,
        "feature_audit_sha256": sha256(FEATURE_AUDIT),
        "dev_predictions": {
            "path": str(PREDICTIONS.relative_to(REPO)),
            "sha256": sha256(PREDICTIONS),
        },
        "optional_branch_latency": {
            "status": "AVAILABLE" if LATENCY.is_file() else "NOT_REQUESTED",
            "path": str(LATENCY.relative_to(REPO)),
            "sha256": sha256(LATENCY) if LATENCY.is_file() else None,
        },
        "yolo_parameter_updates": 0,
        "independent_test_opened": False,
    }
    text = report_text(summary)
    if REPORT.exists():
        if REPORT.read_text(encoding="utf-8") != text:
            raise ExtensionError("existing report differs from resumed result")
    else:
        write_text_exclusive(REPORT, text)
    summary["report"] = {
        "path": str(REPORT.relative_to(REPO)),
        "sha256": sha256(REPORT),
    }
    write_json_exclusive(SUMMARY, summary)
    return summary


def latency_measurement(model: nn.Module, batch: tuple[Tensor, ...]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUPS):
            model(*batch)
        torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(LATENCY_ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*batch)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "minimum_ms": float(np.min(timings)),
        "maximum_ms": float(np.max(timings)),
    }


def measure_latency(device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        raise ExtensionError("latency command requires a CUDA device")
    contract = validate_contract()
    if not SUMMARY.is_file():
        raise ExtensionError("run training before optional latency measurement")
    summary = read_json(SUMMARY)
    if LATENCY.exists():
        existing = read_json(LATENCY)
        if (
            existing.get("contract_sha256") != sha256(CONTRACT)
            or existing.get("warmups") != LATENCY_WARMUPS
            or existing.get("iterations") != LATENCY_ITERATIONS
        ):
            raise ExtensionError("existing latency artifact binding mismatch")
        return existing
    arrays, cache_metadata, _audit = base().load_cache()
    values = base().bundle(arrays, cache_metadata)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    splits = np.asarray(arrays["splits"], dtype=np.int8)
    index = np.flatnonzero(valid & (splits == 1))[:1]
    batch = base().batch_from_bundle(values, index, device)
    baseline_model, _baseline_locked = load_frozen_reference_model(
        BASELINE_ARM, device
    )
    reference_model, _locked = load_frozen_reference_model(REFERENCE_ARM, device)
    models: dict[str, nn.Module] = {
        BASELINE_ARM: baseline_model,
        REFERENCE_ARM: reference_model,
    }
    bindings = runtime_bindings(cache_metadata)
    for arm in ARMS:
        locked = summary["locked_median_artifacts"][arm]
        models[arm] = load_extension_model(
            arm, REPO / locked["checkpoint"]["path"], device, bindings
        )
    results = {arm: latency_measurement(model, batch) for arm, model in models.items()}
    payload = {
        "schema_version": "cleanstart_60k_branch_latency_v1_exploratory",
        "created_at_utc": now(),
        "scope": "branch-only; excludes YOLO and feature extraction",
        "device": device,
        "batch": 1,
        "warmups": LATENCY_WARMUPS,
        "iterations": LATENCY_ITERATIONS,
        "statistics": results,
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(RUNNER),
        "summary_sha256": sha256(SUMMARY),
        "paper_final_claim_allowed": False,
    }
    write_json_exclusive(LATENCY, payload)
    return payload


def unit_smoke(device: str) -> dict[str, Any]:
    validate_contract()
    step_zero = step_zero_audit(None, None, device)
    seed_everything(7)
    patches = torch.randn(4, PATCH_TOKENS, TOKEN_DIM, device=device)
    masks = torch.ones(4, PATCH_TOKENS, dtype=torch.bool, device=device)
    levels = torch.tensor([0, 1, 2, 0], device=device)
    kp = torch.randn(4, KP_DIM, device=device)
    dims = torch.randn(4, DIM_DIM, device=device)
    batch = (patches, masks, levels, kp, dims)
    shapes: dict[str, list[int]] = {}
    with torch.no_grad():
        for arm in ARMS:
            seed_everything(7)
            model = make_model(arm).to(device).eval()
            output = model(*batch)
            shapes[arm] = list(output.shape)
            if tuple(output.shape) != (4, 2):
                raise ExtensionError(f"unit-smoke output shape failed: {arm}")
        seed_everything(7)
        d0 = make_model("D0_DIMS_ONLY").to(device).eval()
        d0_reference = d0(*batch)
        d0_changed_other = d0(
            torch.flip(patches, [0]), ~masks, torch.flip(levels, [0]), -kp, dims
        )
        seed_everything(7)
        k0 = make_model("K0_KP_ONLY").to(device).eval()
        k0_reference = k0(*batch)
        k0_changed_other = k0(
            -patches, ~masks, torch.flip(levels, [0]), kp, -dims
        )
        seed_everything(7)
        c1 = make_model("C1_CENTER_CONCAT").to(device).eval()
        c1_reference = c1(*batch)
        changed_noncenter = patches.clone()
        changed_noncenter[:, :CENTER_INDEX] *= -3.0
        changed_noncenter[:, CENTER_INDEX + 1 :] += 5.0
        c1_changed_noncenter = c1(
            changed_noncenter, masks, levels, kp, dims
        )
    invariance = {
        "D0_ignores_patch_level_kp_max_diff": float(
            torch.max(torch.abs(d0_reference - d0_changed_other)).item()
        ),
        "K0_ignores_patch_level_dims_max_diff": float(
            torch.max(torch.abs(k0_reference - k0_changed_other)).item()
        ),
        "C1_ignores_noncenter_patch_tokens_max_diff": float(
            torch.max(torch.abs(c1_reference - c1_changed_noncenter)).item()
        ),
    }
    if any(value != 0.0 for value in invariance.values()):
        raise ExtensionError(f"unit-smoke modality invariance failed: {invariance}")
    return {
        "passed": True,
        "output_shapes": shapes,
        "invariance": invariance,
        "step_zero_s1_s2_s3": step_zero,
        "parameter_counts": {
            arm: parameter_counts(arm) for arm in (*FROZEN_REFERENCE_ARMS, *ARMS)
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("validate-contract", "unit-smoke", "train", "latency")
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-contract":
        payload = validate_contract()
        print(
            json.dumps(
                {
                    "passed": True,
                    "contract_sha256": sha256(CONTRACT),
                    "runner_sha256": sha256(RUNNER),
                    "arms": payload["arms"],
                },
                indent=2,
            )
        )
        return 0
    device = normalize_device(args.device)
    if args.command == "unit-smoke":
        print(json.dumps(unit_smoke(device), indent=2))
    elif args.command == "train":
        print(json.dumps(train_and_evaluate(device), indent=2))
    elif args.command == "latency":
        print(json.dumps(measure_latency(device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
