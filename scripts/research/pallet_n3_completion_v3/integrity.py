"""Final integrity audit for the six-fit N3 completion experiment.

The verifier distinguishes execution integrity from performance.  Missing
artifacts remain explicit ``null``/``x`` records, while a present artifact with
a bad hash or a contradicted contract is ``BLOCKED_INTEGRITY``.  It never turns
an absent experiment into a numeric zero.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any

import numpy as np

from . import common as C


SCHEMA = "pallet_n3_completion_v3_verify_v1"
OUTPUT = C.DOC / "VERIFY_RESULTS.json"
BACKBONES = ("dope", "resnet18")
SEEDS = (1, 2, 3)
EXPECTED_FITS = tuple((backbone, seed) for backbone in BACKBONES for seed in SEEDS)
EXPECTED_STEPS = 6000
EXPECTED_BATCH = 16
EXPECTED_EXPOSURES_PER_FIT = 96_000
EXPECTED_TOTAL_EXPOSURES = 576_000
DIMENSION_A = [1.1, 1.3, .11]
DIMENSION_B = [.8, .59, .14]


ARTIFACT_SPECS = (
    ("environment_audit", "_docs/experiments/pallet_n3_completion_v3/ENVIRONMENT_AUDIT.json", "protocol", True),
    ("symmetry_activation_audit", "_docs/experiments/pallet_n3_completion_v3/SYMMETRY_ACTIVATION_AUDIT.json", "protocol", True),
    ("dimension_sensitivity_audit", "_docs/experiments/pallet_n3_completion_v3/DIMENSION_SENSITIVITY_AUDIT.json", "protocol", True),
    # GT-free prediction payloads and their receipts.
    ("inference_dope_dev", "data/pallet/results/pallet_n3_completion_v3/predictions/dope_DEV319.json", "inference", True),
    ("inference_resnet18_dev", "data/pallet/results/pallet_n3_completion_v3/predictions/resnet18_DEV319.json", "inference", True),
    ("inference_dope_square", "data/pallet/results/pallet_n3_completion_v3/predictions/dope_GREEN0918_119.json", "inference", True),
    ("inference_resnet18_square", "data/pallet/results/pallet_n3_completion_v3/predictions/resnet18_GREEN0918_119.json", "inference", True),
    ("inference_dope_dev_receipt", "_docs/experiments/pallet_n3_completion_v3/INFERENCE_DOPE_DEV319_COMPLETE.json", "inference", True),
    ("inference_resnet18_dev_receipt", "_docs/experiments/pallet_n3_completion_v3/INFERENCE_RESNET18_DEV319_COMPLETE.json", "inference", True),
    ("inference_dope_square_receipt", "_docs/experiments/pallet_n3_completion_v3/INFERENCE_DOPE_GREEN0918_119_COMPLETE.json", "inference", True),
    ("inference_resnet18_square_receipt", "_docs/experiments/pallet_n3_completion_v3/INFERENCE_RESNET18_GREEN0918_119_COMPLETE.json", "inference", True),
    # Exact DEV319 evaluation.
    ("evaluation_dope", "data/pallet/results/pallet_n3_completion_v3/evaluation/dope.json", "evaluation", True),
    ("evaluation_resnet18", "data/pallet/results/pallet_n3_completion_v3/evaluation/resnet18.json", "evaluation", True),
    # Separate GREEN0918 transfer and the frozen YOLO R0/P/N2/N3 audit.
    ("square_dope", "_docs/experiments/pallet_n3_completion_v3/SQUARE_DOPE_RESULTS.json", "square", True),
    ("square_resnet18", "_docs/experiments/pallet_n3_completion_v3/SQUARE_RESNET18_RESULTS.json", "square", True),
    ("square_yolo_predictions", "data/pallet/results/pallet_n3_completion_v3/SQUARE_YOLO_PREDICTIONS.json", "square", True),
    ("square_yolo_metrics", "data/pallet/results/pallet_n3_completion_v3/SQUARE_YOLO_METRICS.json", "square", True),
    ("square_yolo_result", "_docs/experiments/pallet_n3_completion_v3/SQUARE_YOLO_RESULTS.json", "square", True),
    # Same-machine fixed runtime protocol.
    ("runtime_dope_raw", "data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json", "runtime", True),
    ("runtime_resnet18_raw", "data/pallet/results/pallet_n3_completion_v3/runtime/resnet18_seed1.json", "runtime", True),
    ("runtime_dope_receipt", "_docs/experiments/pallet_n3_completion_v3/RUNTIME_DOPE_SEED1.json", "runtime", True),
    ("runtime_resnet18_receipt", "_docs/experiments/pallet_n3_completion_v3/RUNTIME_RESNET18_SEED1.json", "runtime", True),
    # Offline lifter replay. Accuracy is required to stay x without independent GT.
    ("lifter_plan", "data/pallet/results/pallet_n3_completion_v3/lifter/FIXED_SENSOR_1000MS_PLAN.json", "lifter", True),
    ("lifter_raw", "data/pallet/results/pallet_n3_completion_v3/lifter/YOLO_R0_N3_SEED1_RAW.json", "lifter", True),
    ("lifter_receipt", "_docs/experiments/pallet_n3_completion_v3/LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json", "lifter", True),
    # Hash-verified reuse/rescore package.
    ("reuse_summary", "_docs/experiments/pallet_n3_completion_v3/REUSE_RESULTS.json", "reuse", True),
    ("reuse_table", "_docs/experiments/pallet_n3_completion_v3/REUSE_RESULTS.csv", "reuse", False),
    ("reuse_rows", "data/pallet/results/pallet_n3_completion_v3/reuse/PER_FRAME_SCORES.json", "reuse", False),
    ("reuse_bindings", "data/pallet/results/pallet_n3_completion_v3/reuse/SOURCE_BINDINGS.json", "reuse", False),
    # Generated tables/report/figures.  PARTIAL with declared x is scientifically
    # valid and can pass integrity; presence/hash linkage is audited separately.
    ("report_tables", "_docs/experiments/pallet_n3_completion_v3/TABLES.json", "report", True),
    ("report_tables_md", "_docs/experiments/pallet_n3_completion_v3/TABLES.md", "report", False),
    ("report_remaining_x", "_docs/experiments/pallet_n3_completion_v3/REMAINING_X.md", "report", False),
    ("report_final_ko", "_docs/experiments/pallet_n3_completion_v3/FINAL_REPORT_KO.md", "report", False),
    ("report_figure_manifest", "_docs/experiments/pallet_n3_completion_v3/figures/SELECTION_MANIFEST.json", "report", True),
    ("report_training_figure", "_docs/experiments/pallet_n3_completion_v3/figures/training_curves.png", "report", False),
    ("report_backbone_figure", "_docs/experiments/pallet_n3_completion_v3/figures/backbone_comparison.png", "report", False),
    ("report_runtime_figure", "_docs/experiments/pallet_n3_completion_v3/figures/runtime.png", "report", False),
    ("report_dev_overlay", "_docs/experiments/pallet_n3_completion_v3/figures/dev_overlays.png", "report", False),
    ("report_lifter_figure", "_docs/experiments/pallet_n3_completion_v3/figures/lifter.png", "report", False),
    ("report_subgroups_figure", "_docs/experiments/pallet_n3_completion_v3/figures/subgroups_or_thresholds.png", "report", False),
    ("report_tex_backbone", "_docs/experiments/pallet_n3_completion_v3/table_fragments/backbone_dev_headline.tex", "report", False),
    ("report_tex_square", "_docs/experiments/pallet_n3_completion_v3/table_fragments/square_2d.tex", "report", False),
    ("report_tex_runtime", "_docs/experiments/pallet_n3_completion_v3/table_fragments/runtime.tex", "report", False),
    ("report_frame_csv", "data/pallet/results/pallet_n3_completion_v3/report/FRAME_METRICS.csv", "report", False),
    ("report_corner_csv", "data/pallet/results/pallet_n3_completion_v3/report/CORNER_METRICS.csv", "report", False),
)


class IntegrityError(RuntimeError):
    """A present artifact contradicts its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def binding(path: Path | str, root: Path | str = C.ROOT) -> dict:
    path = Path(path).resolve()
    root = Path(root).resolve()
    display = path.relative_to(root) if path.is_relative_to(root) else path
    return {"path": str(display), "sha256": C.sha256(path), "bytes": path.stat().st_size}


def checkout_roots(root: Path | str = C.ROOT) -> tuple[Path, ...]:
    """Return a worktree plus its common checkout for ignored large artifacts."""
    root = Path(root).resolve()
    roots = [root]
    marker = root / ".git"
    if marker.is_file():
        text = marker.read_text().strip()
        if text.startswith("gitdir:"):
            git_dir = Path(text.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
            common = git_dir / "commondir"
            if common.is_file():
                common_dir = Path(common.read_text().strip())
                if not common_dir.is_absolute():
                    common_dir = (git_dir / common_dir).resolve()
                roots.append(common_dir.parent.resolve())
    return tuple(dict.fromkeys(roots))


def resolve_binding(entry: Mapping[str, Any], roots: Sequence[Path]) -> Path:
    """Resolve and hash-check a standard {path,sha256,bytes} binding."""
    _require(isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str),
             "invalid binding schema")
    source = Path(entry["path"])
    candidates = [source] if source.is_absolute() else [root / source for root in roots]
    observed = []
    for path in candidates:
        if not path.is_file():
            observed.append(f"missing:{path}")
            continue
        digest = C.sha256(path)
        size = path.stat().st_size
        if digest == entry["sha256"] and ("bytes" not in entry or size == entry["bytes"]):
            return path
        observed.append(f"drift:{path}:{size}:{digest}")
    raise IntegrityError("bound file is missing or changed: " + "; ".join(observed))


def iter_bindings(value: Any, path: str = "$"):
    """Yield embedded standard bindings without mistaking ordinary paths for one."""
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield path, value
        for key, child in value.items():
            yield from iter_bindings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_bindings(child, f"{path}[{index}]")


def state_hash(state: Mapping[str, Any]) -> str:
    """Byte-identical implementation of :func:`train.state_hash`."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def order_hash(order: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(order).astype("<i8").tobytes()).hexdigest()


def _normalized_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalized_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalized_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized_json(item) for item in value]
    return value


def validate_protocol_payload(protocol: Mapping[str, Any]) -> dict:
    training = protocol.get("training", {})
    _require(protocol.get("schema") == "pallet_n3_completion_v3_protocol_v1",
             "protocol schema drift")
    _require(training.get("backbones") == list(BACKBONES), "protocol backbone drift")
    _require(training.get("seeds") == list(SEEDS), "protocol seed drift")
    _require(training.get("fits") == 6, "protocol must require six fits")
    _require(training.get("steps_per_fit") == EXPECTED_STEPS, "protocol step drift")
    _require(training.get("batch") == EXPECTED_BATCH, "protocol batch drift")
    _require(training.get("exposures_per_fit") == EXPECTED_EXPOSURES_PER_FIT,
             "protocol per-fit exposure drift")
    _require(training.get("total_updates") == 36_000, "protocol update drift")
    _require(training.get("total_exposures") == EXPECTED_TOTAL_EXPOSURES,
             "protocol total exposure drift")
    expected_optimizer = {"name": "AdamW", **C.OPTIMIZER}
    expected_optimizer["betas"] = list(expected_optimizer["betas"])
    _require(_normalized_json(training.get("optimizer")) == expected_optimizer,
             "protocol optimizer drift")
    _require(training.get("precision") == {"model": "FP32", "AMP": False, "TF32": False},
             "protocol precision drift")
    _require(training.get("selection") == "fixed final step6000 only",
             "protocol final-step selection drift")
    _require(training.get("outcome_driven_extension") is False,
             "outcome-driven extension is forbidden")
    _require(protocol.get("methods", {}).get("base_inputs") == ["RGB"],
             "base input is not RGB-only")
    for backbone in BACKBONES:
        row = protocol.get("backbones", {}).get(backbone, {})
        _require(row.get("frozen") is True and row.get("retrained") is False,
                 f"{backbone} base freeze contract drift")
        _require(row.get("base_dimension_input") is False,
                 f"{backbone} base unexpectedly receives dimensions")
    return {
        "status": "PASS", "fits": 6, "steps_per_fit": EXPECTED_STEPS,
        "batch": EXPECTED_BATCH, "total_exposures": EXPECTED_TOTAL_EXPOSURES,
        "precision": training["precision"], "optimizer": training["optimizer"],
    }


def _expected_base_sha(protocol: Mapping[str, Any], backbone: str) -> str:
    row = protocol["backbones"][backbone]
    key = "checkpoint" if backbone == "dope" else "source_checkpoint"
    return row[key]["sha256"]


def validate_fit_payload(receipt: Mapping[str, Any], checkpoint: Mapping[str, Any],
                         order: np.ndarray, protocol: Mapping[str, Any],
                         protocol_sha256: str, backbone: str, seed: int) -> dict:
    """Pure payload validation used by the CLI and small CPU fixtures."""
    import torch

    expected_optimizer = copy.deepcopy(C.OPTIMIZER)
    expected_optimizer["betas"] = list(expected_optimizer["betas"])
    _require(receipt.get("schema") == "pallet_n3_completion_v3_fit_v1",
             "fit receipt schema drift")
    _require(receipt.get("complete") is True and receipt.get("smoke") is False,
             "main fit is incomplete or is a smoke run")
    _require(receipt.get("backbone") == backbone and receipt.get("seed") == seed,
             "fit receipt identity drift")
    _require(receipt.get("steps") == EXPECTED_STEPS, "fit receipt step drift")
    _require(receipt.get("exposures") == EXPECTED_EXPOSURES_PER_FIT,
             "fit receipt exposure drift")
    _require(receipt.get("optimizer") == expected_optimizer, "fit optimizer receipt drift")
    _require(receipt.get("FP32") is True and receipt.get("AMP") is False,
             "fit is not locked FP32/no-AMP")
    _require(receipt.get("TF32_matmul") is False and receipt.get("TF32_cudnn") is False,
             "fit enabled TF32")
    for key, expected in (
            ("real_training_images", 0), ("base_retrained", False),
            ("symmetry_supervision", True), ("dimension_input_to_N3", True),
            ("base_receives_dimensions", False), ("final_step_only", True)):
        _require(receipt.get(key) == expected, f"fit contract drift: {key}")

    order = np.asarray(order)
    _require(order.shape == (EXPECTED_STEPS, EXPECTED_BATCH), "fit order shape drift")
    _require(np.issubdtype(order.dtype, np.integer), "fit order must contain integer rows")
    digest = order_hash(order)
    _require(receipt.get("order_prefix_sha256") == digest, "receipt order hash drift")
    _require(checkpoint.get("order_sha256") == digest, "checkpoint order hash drift")

    _require(checkpoint.get("complete") is True and checkpoint.get("smoke") is False,
             "checkpoint is incomplete or smoke")
    _require(checkpoint.get("backbone") == backbone and checkpoint.get("seed") == seed,
             "checkpoint identity drift")
    _require(checkpoint.get("step") == EXPECTED_STEPS and
             checkpoint.get("steps") == EXPECTED_STEPS, "checkpoint is not final step6000")
    _require(checkpoint.get("protocol_sha256") == protocol_sha256,
             "checkpoint protocol hash drift")
    _require(checkpoint.get("initial_state_sha256") == receipt.get("initial_state_sha256"),
             "initial-state hash drift")
    base_sha = _expected_base_sha(protocol, backbone)
    _require(checkpoint.get("base_checkpoint_sha256") == base_sha ==
             receipt.get("base_checkpoint_sha256"), "frozen base hash drift")

    model_state = checkpoint.get("model_state_dict")
    _require(isinstance(model_state, Mapping) and model_state, "model state is missing")
    for name, value in model_state.items():
        _require(torch.is_tensor(value) and value.dtype == torch.float32,
                 f"model tensor is not FP32: {name}")
        _require(torch.isfinite(value).all().item(), f"non-finite model tensor: {name}")
    final_hash = state_hash(model_state)
    _require(final_hash == receipt.get("final_state_sha256"), "final model-state hash drift")

    optimizer = checkpoint.get("optimizer_state_dict")
    _require(isinstance(optimizer, Mapping) and optimizer.get("state") and
             optimizer.get("param_groups"), "optimizer state is missing")
    _require(len(optimizer["param_groups"]) == 1, "unexpected optimizer group count")
    group = optimizer["param_groups"][0]
    _require(tuple(group.get("betas", ())) == tuple(C.OPTIMIZER["betas"]),
             "checkpoint AdamW betas drift")
    _require(float(group.get("weight_decay", -1)) == C.OPTIMIZER["weight_decay"],
             "checkpoint AdamW weight decay drift")
    _require(float(group.get("lr", -1)) == C.learning_rate(EXPECTED_STEPS),
             "checkpoint final learning rate drift")
    for parameter, values in optimizer["state"].items():
        _require(isinstance(values, Mapping), f"optimizer slot is invalid: {parameter}")
        for name in ("step", "exp_avg", "exp_avg_sq"):
            value = values.get(name)
            _require(torch.is_tensor(value), f"optimizer {name} is missing")
            _require(torch.isfinite(value).all().item(), f"optimizer {name} is non-finite")
        _require(int(values["step"].item()) == EXPECTED_STEPS,
                 "optimizer step does not equal checkpoint step")
        _require(values["exp_avg"].dtype == torch.float32 and
                 values["exp_avg_sq"].dtype == torch.float32,
                 "optimizer moments are not FP32")

    torch_rng = checkpoint.get("torch_rng_state")
    cuda_rng = checkpoint.get("cuda_rng_state")
    _require(torch.is_tensor(torch_rng) and torch_rng.dtype == torch.uint8 and
             torch_rng.ndim == 1 and torch_rng.numel() > 0, "CPU RNG state is missing")
    _require(isinstance(cuda_rng, list) and cuda_rng and all(
        torch.is_tensor(value) and value.dtype == torch.uint8 and value.ndim == 1
        and value.numel() > 0 for value in cuda_rng), "CUDA RNG state is missing")
    history = receipt.get("history")
    _require(isinstance(history, list) and history and history[0].get("step") == 1 and
             history[-1].get("step") == EXPECTED_STEPS, "fit history does not reach step6000")
    _require(receipt.get("first_step", {}).get("zero_effect_logits") is True and
             receipt.get("second_step", {}).get("metadata_encoder_connected") is True,
             "metadata gradient-connectivity receipt is incomplete")
    return {
        "status": "PASS", "backbone": backbone, "seed": seed,
        "steps": EXPECTED_STEPS, "exposures": EXPECTED_EXPOSURES_PER_FIT,
        "order_sha256": digest, "initial_state_sha256": receipt["initial_state_sha256"],
        "final_state_sha256": final_hash, "base_checkpoint_sha256": base_sha,
        "optimizer_state": "AdamW moments + final step verified",
        "rng_state": "CPU and CUDA states present",
        "precision": {"FP32": True, "AMP": False, "TF32": False},
    }


def _missing(reason: str, *, status: str = "MISSING") -> dict:
    return {"status": status, "value": None, "display": "x", "reason": reason}


def _find_relative(relative: Path, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        path = root / relative
        if path.is_file():
            return path
    return None


def inspect_training(root: Path | str, protocol: Mapping[str, Any],
                     protocol_path: Path) -> tuple[dict, dict, dict]:
    """Audit every fit independently and retain valid states for sensitivity."""
    import torch

    root = Path(root).resolve()
    roots = checkout_roots(root)
    protocol_sha = C.sha256(protocol_path)
    fits, states, receipts = {}, {}, {}
    for backbone, seed in EXPECTED_FITS:
        key = f"{backbone}_seed{seed}"
        receipt_rel = Path(
            f"_docs/experiments/pallet_n3_completion_v3/TRAIN_{backbone.upper()}_SEED{seed}.json")
        checkpoint_rel = Path(
            f"data/pallet/results/pallet_n3_completion_v3/runs/{backbone}/seed{seed}/last.pt")
        receipt_path = _find_relative(receipt_rel, roots)
        checkpoint_path = _find_relative(checkpoint_rel, roots)
        if receipt_path is None:
            if checkpoint_path is None:
                fits[key] = _missing("fit receipt and checkpoint do not exist")
                continue
            try:
                partial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if partial.get("complete") is True:
                    fits[key] = _missing(
                        "a complete checkpoint exists without its immutable fit receipt",
                        status="BLOCKED_INTEGRITY")
                else:
                    fits[key] = {
                        **_missing("fit is still in progress", status="IN_PROGRESS"),
                        "observed_step": partial.get("step"),
                        "expected_step": EXPECTED_STEPS,
                        "checkpoint": binding(checkpoint_path, root),
                    }
            except Exception as error:
                fits[key] = _missing(f"unreceipted checkpoint is unreadable: {error}",
                                     status="BLOCKED_INTEGRITY")
            continue
        if checkpoint_path is None:
            fits[key] = _missing("fit receipt exists but checkpoint is missing",
                                 status="BLOCKED_INTEGRITY")
            continue
        try:
            receipt = C.read(receipt_path)
            bound_checkpoint = resolve_binding(receipt["checkpoint"], roots)
            _require(bound_checkpoint.resolve() == checkpoint_path.resolve(),
                     "receipt points to a different checkpoint")
            order_path = resolve_binding(receipt["order"], roots)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            audit = validate_fit_payload(
                receipt, checkpoint, np.load(order_path), protocol,
                protocol_sha, backbone, seed)
            audit.update(receipt=binding(receipt_path, root),
                         checkpoint=binding(checkpoint_path, root),
                         order=binding(order_path, root))
            fits[key] = audit
            states[(backbone, seed)] = checkpoint["model_state_dict"]
            receipts[(backbone, seed)] = receipt
        except Exception as error:
            fits[key] = _missing(str(error), status="BLOCKED_INTEGRITY")

    statuses = [row["status"] for row in fits.values()]
    fits_complete = statuses == ["PASS"] * len(EXPECTED_FITS)
    blocked = any(status == "BLOCKED_INTEGRITY" for status in statuses)
    summaries = {}
    for backbone in BACKBONES:
        relative = Path(
            f"_docs/experiments/pallet_n3_completion_v3/TRAINING_{backbone.upper()}_COMPLETE.json")
        path = _find_relative(relative, roots)
        backbone_complete = all(fits[f"{backbone}_seed{seed}"]["status"] == "PASS"
                                for seed in SEEDS)
        if path is None:
            summaries[backbone] = (_missing("backbone training summary is absent",
                                             status=("BLOCKED_INTEGRITY" if backbone_complete
                                                     else "MISSING")))
            continue
        try:
            payload = C.read(path)
            _require(backbone_complete, "training summary exists before all three fits verify")
            expected_runs = [receipts[(backbone, seed)] for seed in SEEDS]
            _require(payload.get("schema") == "pallet_n3_completion_v3_training_v1" and
                     payload.get("complete") is True and payload.get("backbone") == backbone,
                     "training summary identity drift")
            _require(payload.get("fits") == 3 and payload.get("seeds") == list(SEEDS) and
                     payload.get("steps_per_fit") == EXPECTED_STEPS and
                     payload.get("total_updates") == 18_000 and
                     payload.get("total_exposures") == 288_000,
                     "training summary budget drift")
            _require(payload.get("runs") == expected_runs,
                     "training summary does not embed the three exact receipts")
            summaries[backbone] = {"status": "PASS", "binding": binding(path, root)}
        except Exception as error:
            summaries[backbone] = _missing(str(error), status="BLOCKED_INTEGRITY")
            blocked = True
    blocked = blocked or any(row["status"] == "BLOCKED_INTEGRITY"
                             for row in summaries.values())
    summaries_complete = all(row["status"] == "PASS" for row in summaries.values())
    complete = fits_complete and summaries_complete
    verified_exposures = sum(row.get("exposures", 0) for row in fits.values()
                             if row["status"] == "PASS")
    status = "BLOCKED_INTEGRITY" if blocked else "PASS" if complete else "INCOMPLETE"
    result = {
        "status": status, "expected_fits": 6,
        "verified_fits": sum(row["status"] == "PASS" for row in fits.values()),
        "expected_steps_per_fit": EXPECTED_STEPS,
        "expected_exposures_per_fit": EXPECTED_EXPOSURES_PER_FIT,
        "expected_total_exposures": EXPECTED_TOTAL_EXPOSURES,
        "verified_total_exposures": verified_exposures,
        "fits": fits, "summaries": summaries,
    }
    if not complete:
        result.update(value=None, display="x",
                      reason="all six final fit receipts have not yet passed")
    return result, states, receipts


def measure_dimension_sensitivity(backbone: str, state: Mapping[str, Any],
                                  normalization: Mapping[str, Any],
                                  checkpoint_sha256: str) -> dict:
    """Hold visual evidence fixed and vary only two registered W,D,H vectors."""
    import torch
    from .model import build_n3, normalize_dimensions

    head = build_n3(backbone).cpu().eval()
    head.load_state_dict(state, strict=True)
    config = C.CONFIGS[backbone]
    height = width = 64
    p3_shape = (1, config["c3"], height // config["stride3"], width // config["stride3"])
    p4_shape = (1, config["c4"], height // config["stride4"], width // config["stride4"])
    p3 = torch.linspace(-.25, .25, int(np.prod(p3_shape)), dtype=torch.float32).reshape(p3_shape)
    p4 = torch.linspace(.25, -.25, int(np.prod(p4_shape)), dtype=torch.float32).reshape(p4_shape)
    points = torch.tensor([[
        [20., 20.], [40., 20.], [40., 40.], [20., 40.],
        [24., 24.], [36., 24.], [36., 36.], [24., 36.], [30., 30.],
    ]], dtype=torch.float32)
    box = torch.tensor([[15., 15., 45., 45.]], dtype=torch.float32)
    valid = torch.ones((1, 9), dtype=torch.bool)
    input_shape = torch.tensor([[height, width]], dtype=torch.int64)
    dimensions = torch.tensor([DIMENSION_A, DIMENSION_B], dtype=torch.float32)
    contexts = normalize_dimensions(dimensions, normalization)
    with torch.no_grad():
        first = head(p3, p4, points, box, valid, input_shape,
                     dimension_context=contexts[:1], lam=0.)
        second = head(p3, p4, points, box, valid, input_shape,
                      dimension_context=contexts[1:], lam=0.)
    base_equal = torch.equal(first["base_logits"], second["base_logits"])
    difference = (first["logits"] - second["logits"]).abs()
    maximum = float(difference.max())
    changed = int((difference > 0).sum())
    finite = bool(torch.isfinite(first["logits"]).all() and
                  torch.isfinite(second["logits"]).all())
    result = {
        "status": "PASS" if base_equal and finite and changed > 0 and maximum > 0 else
                  "BLOCKED_INTEGRITY",
        "backbone": backbone, "checkpoint_sha256": checkpoint_sha256,
        "visual_inputs_bitwise_identical": True,
        "base_logits_bitwise_identical": base_equal,
        "dimension_a_wdh_m": DIMENSION_A, "dimension_b_wdh_m": DIMENSION_B,
        "normalized_context_a": contexts[0].tolist(),
        "normalized_context_b": contexts[1].tolist(),
        "changed_logit_entries": changed, "max_abs_logit_delta": maximum,
        "finite": finite,
    }
    if result["status"] != "PASS":
        result.update(value=None, display="x",
                      reason="trained N3 logits did not respond to W,D,H while visual inputs were fixed")
    return result


def inspect_sensitivity(training: Mapping[str, Any], states: Mapping,
                        normalization: Mapping[str, Any]) -> dict:
    fits = {}
    for backbone, seed in EXPECTED_FITS:
        key = f"{backbone}_seed{seed}"
        fit = training["fits"][key]
        if fit["status"] != "PASS":
            fits[key] = _missing("final trained checkpoint is not yet verified")
            continue
        try:
            fits[key] = measure_dimension_sensitivity(
                backbone, states[(backbone, seed)], normalization,
                fit["checkpoint"]["sha256"])
        except Exception as error:
            fits[key] = _missing(str(error), status="BLOCKED_INTEGRITY")
    statuses = [row["status"] for row in fits.values()]
    status = ("BLOCKED_INTEGRITY" if "BLOCKED_INTEGRITY" in statuses else
              "PASS" if statuses == ["PASS"] * 6 else "INCOMPLETE")
    output = {"status": status, "fits": fits,
              "contract": "same visual evidence and base logits; only registered W,D,H changes"}
    if status != "PASS":
        output.update(value=None, display="x",
                      reason="dimension sensitivity is not verified for all six final heads")
    return output


def class_method_arguments(source: str, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return [argument.arg for argument in
                            (*child.args.posonlyargs, *child.args.args, *child.args.kwonlyargs)]
    raise IntegrityError(f"method not found: {class_name}.{method_name}")


def predictor_base_call_contract(source: str) -> dict:
    """Prove Predictor sends dimensions to N3 batch, not adapter.infer."""
    tree = ast.parse(source)
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Predictor":
            method = next((child for child in node.body
                           if isinstance(child, ast.FunctionDef) and child.name == "predict"), None)
    _require(method is not None, "Predictor.predict is missing")
    base_calls, batch_calls = [], []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "infer":
            base_calls.append(node)
        if node.func.attr == "_batch":
            batch_calls.append(node)
    _require(len(base_calls) == 1 and len(batch_calls) == 1,
             "Predictor base/N3 call topology drift")
    base_names = {child.id for child in ast.walk(base_calls[0]) if isinstance(child, ast.Name)}
    batch_names = {child.id for child in ast.walk(batch_calls[0]) if isinstance(child, ast.Name)}
    _require("dimensions" not in base_names and "dimensions" in batch_names,
             "dimensions crossed the RGB-base boundary")
    return {"adapter_infer_receives_dimensions": False,
            "n3_batch_receives_dimensions": True}


def prove_base_rgb_only(root: Path | str, protocol: Mapping[str, Any],
                        training: Mapping[str, Any]) -> dict:
    root = Path(root).resolve()
    dope = root / "scripts/research/pallet_dope_refiner_20261001_v1/dope_adapter.py"
    resnet = root / "scripts/research/pallet_n3_completion_v3/resnet_constant_adapter.py"
    inference = root / "scripts/research/pallet_n3_completion_v3/inference.py"
    try:
        validate_protocol_payload(protocol)
        signatures = {
            "dope": class_method_arguments(dope.read_text(), "FrozenDopeAdapter", "infer"),
            "resnet18": class_method_arguments(
                resnet.read_text(), "FrozenConstantResnetAdapter", "infer"),
        }
        for backbone, arguments in signatures.items():
            _require(any(name in arguments for name in ("image", "images")),
                     f"{backbone} adapter lacks an RGB image argument")
            _require(not any("dimension" in name or "context" in name for name in arguments),
                     f"{backbone} adapter base API accepts dimensions")
        topology = predictor_base_call_contract(inference.read_text())
        receipt_flags = {
            key: row.get("status") == "PASS" and
                 training["fits"][key].get("precision", {}).get("FP32") is True
            for key, row in training["fits"].items()}
        return {
            "status": "PASS", "base_inputs": ["RGB"],
            "adapter_infer_arguments": signatures, **topology,
            "protocol_base_dimension_input": {
                backbone: protocol["backbones"][backbone]["base_dimension_input"]
                for backbone in BACKBONES},
            "verified_fit_flags": receipt_flags,
            "sources": [binding(dope, root), binding(resnet, root), binding(inference, root)],
        }
    except Exception as error:
        return _missing(str(error), status="BLOCKED_INTEGRITY")


_NULL_OUTCOME_KEYS = {
    "value", "result", "headline", "mean_of_seed_metrics", "pose",
    "pose_metrics", "translation", "rotation", "accuracy_metrics",
}


def _is_x_marker(value: Mapping[str, Any]) -> bool:
    # ``x`` marks missing/blocked evidence and ``NA`` marks a measured zero
    # denominator or protocol-defined non-applicability.  Both are explicit
    # report-schema explanations for an intentionally null machine value.
    if str(value.get("display", "")).strip().upper() in {"X", "NA", "N/A"}:
        return True
    status = str(value.get("status", "")).upper()
    return status in {"X", "NA", "N/A", "UNDEFINED", "NOT_APPLICABLE"} or status.startswith((
        "MISSING", "BLOCKED", "NOT_", "DISABLED", "UNAVAILABLE", "INCOMPLETE"))


def explicit_x_issues(value: Any, path: str = "$", inherited: bool = False) -> list[str]:
    """Find outcome nulls that lack a local/inherited missing-status marker."""
    issues = []
    if isinstance(value, Mapping):
        marker = inherited or _is_x_marker(value)
        if value.get("physical_pose_reference") is False:
            marker = True
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if child is None and key in _NULL_OUTCOME_KEYS and not marker:
                issues.append(child_path)
            elif isinstance(child, (Mapping, list)):
                issues.extend(explicit_x_issues(child, child_path, marker))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(explicit_x_issues(child, f"{path}[{index}]", inherited))
    return issues


def _validate_lifter_x(payload: Mapping[str, Any]) -> None:
    accuracy = payload.get("accuracy")
    if accuracy is None:
        return
    _require(accuracy.get("independent_ground_truth") is False,
             "lifter result claims an independent reference")
    for key in ("independent_accuracy", "position_error_m", "yaw_error_deg"):
        row = accuracy.get(key, {})
        _require(row.get("value") is None and str(row.get("status", "")).lower() == "x",
                 f"lifter unsupported accuracy is not explicit x: {key}")


def contract_remaining_x(name: str, category: str,
                         payload: Mapping[str, Any]) -> list[dict]:
    """Return truthful contract gaps that remain x despite valid artifacts.

    These are semantic gaps, not integrity failures.  A file can be complete,
    hash-valid and reproducible while still lacking labels or an independent
    reference required to replace a manuscript ``x`` with a number.
    """
    remaining: list[dict] = []
    if name == "reuse_summary":
        contracts = payload.get("contracts", {})
        d = contracts.get("D", {}) if isinstance(contracts, Mapping) else {}
        if d.get("status") == "PARTIAL_LABELS":
            counts = d.get("frame_counts", {})
            visibility = d.get("corner_visibility", {})
            unclassified = counts.get("unclassified")
            detail = visibility.get("reason")
            reason = (
                f"reuse contract D is PARTIAL_LABELS: {unclassified} DEV319 frames "
                "remain unclassified" if isinstance(unclassified, int) else
                "reuse contract D is PARTIAL_LABELS")
            if detail:
                reason += f"; {detail}"
            remaining.append({
                "id": "reuse_D_partial_labels", "artifact": name,
                "category": category, "status": "PARTIAL_LABELS",
                "value": None, "display": "x", "reason": reason,
                "required": "complete frozen frame/corner visibility labels",
            })
        i = contracts.get("I", {}) if isinstance(contracts, Mapping) else {}
        if i.get("status") == "PARTIAL_SAFE_COHORT":
            blocked = i.get("DEV319_table", {})
            reason = blocked.get("reason") or (
                "reuse contract I has only a partial safe common cohort")
            remaining.append({
                "id": "reuse_I_partial_safe_cohort", "artifact": name,
                "category": category, "status": "PARTIAL_SAFE_COHORT",
                "value": None, "display": "x", "reason": reason,
                "required": "a common non-exposed evaluation cohort for every compared student",
            })
    if category == "lifter":
        accuracy = payload.get("accuracy")
        if (isinstance(accuracy, Mapping)
                and accuracy.get("independent_ground_truth") is False):
            remaining.append({
                "id": "lifter_accuracy_position_yaw_no_independent_gt",
                "artifact": name, "category": category,
                "status": "BLOCKED_NO_INDEPENDENT_GT",
                "value": None, "display": "x",
                "reason": ("recorded lifter replay has no independent pose/corner ground "
                           "truth; accuracy, position error and yaw error remain x"),
                "required": "independent synchronized pose or corner ground truth",
            })
    if category == "square":
        missing_pose = (payload.get("physical_pose_reference") is False
                        and payload.get("pose_metrics") is None)
        pose_3d = payload.get("pose_3d")
        if isinstance(pose_3d, Mapping):
            missing_pose = missing_pose or all(
                isinstance(pose_3d.get(key), Mapping)
                and pose_3d[key].get("value") is None
                for key in ("translation", "rotation"))
        if missing_pose and name in {"square_dope", "square_resnet18", "square_yolo_result"}:
            remaining.append({
                "id": f"{name}_6d_no_reference", "artifact": name,
                "category": category,
                "status": "BLOCKED_NO_INDEPENDENT_CANONICAL_POSE_REFERENCE",
                "value": None, "display": "x",
                "reason": ("GREEN0918_119 supplies a reused 2D corner audit but no "
                           "independent canonical 6D pose reference; translation and "
                           "rotation metrics remain x"),
                "required": "independent canonical 6D pose reference for GREEN0918_119",
            })
    return remaining


def _validate_report_figure_manifest(payload: Mapping[str, Any], path: Path) -> None:
    _require(payload.get("schema") == "pallet_n3_completion_v3_figure_selection_v1",
             "report figure-manifest schema drift")
    figures = payload.get("figures")
    _require(isinstance(figures, list) and figures,
             "report figure manifest has no figure inventory")
    seen = set()
    for row in figures:
        _require(isinstance(row, Mapping) and isinstance(row.get("file"), str),
                 "invalid report figure inventory row")
        name = row["file"]
        _require(name not in seen, f"duplicate report figure: {name}")
        seen.add(name)
        figure = path.parent / name
        _require(figure.is_file(), f"bound report figure is missing: {name}")
        _require(row.get("sha256") == C.sha256(figure),
                 f"report figure hash drift: {name}")
        _require(row.get("bytes") == figure.stat().st_size,
                 f"report figure size drift: {name}")
    expected = {
        "training_curves.png", "backbone_comparison.png", "runtime.png",
        "dev_overlays.png", "lifter.png", "subgroups_or_thresholds.png"}
    _require(seen == expected, "report figure inventory drift")


def _validate_report_tex(path: Path, roots: Sequence[Path]) -> None:
    tables = _find_relative(
        Path("_docs/experiments/pallet_n3_completion_v3/TABLES.json"), roots)
    _require(tables is not None, "TABLES.json is missing for TeX verification")
    first = path.read_text().splitlines()[0] if path.stat().st_size else ""
    _require(C.sha256(tables) in first,
             f"TeX fragment does not bind current TABLES.json: {path.name}")


def _validate_count_summary(value: Mapping[str, Any], *, expected: int | None = None) -> None:
    total = value.get("rows_or_exposures")
    counts = value.get("branch_counts")
    identity = value.get("identity_count")
    non_identity = value.get("non_identity_count")
    fraction = value.get("non_identity_fraction")
    _require(isinstance(total, int) and total > 0,
             "symmetry audit has an invalid exposure count")
    if expected is not None:
        _require(total == expected, "symmetry audit exposure budget drift")
    _require(isinstance(counts, Mapping) and all(
        isinstance(count, int) and count >= 0 for count in counts.values()),
        "symmetry audit branch counts are invalid")
    _require(sum(counts.values()) == total,
             "symmetry audit branch counts do not sum to exposures")
    _require(isinstance(identity, int) and isinstance(non_identity, int)
             and identity + non_identity == total,
             "symmetry audit identity counts are inconsistent")
    _require(isinstance(fraction, (int, float)) and np.isfinite(fraction)
             and np.isclose(float(fraction), non_identity / total),
             "symmetry audit non-identity fraction drift")


def _validate_protocol_artifact(name: str, payload: Mapping[str, Any]) -> None:
    """Validate the three audit artifacts beyond generic complete/hash checks."""
    if name == "environment_audit":
        _require(payload.get("schema") ==
                 "pallet_n3_completion_v3_environment_audit_v1",
                 "environment audit schema drift")
        roles = payload.get("roles", {})
        primary = roles.get("dope_resnet_training_inference_runtime", {})
        yolo = roles.get("yolo26_square_and_offline_lifter", {})
        _require(payload.get("complete") is True
                 and primary.get("has_C3k2") is False
                 and yolo.get("has_C3k2") is True,
                 "environment interpreter capability boundary drift")
        _require(payload.get("boundary", {}).get(
            "single_process_environment_mixing") is False,
            "environment audit permits mixed interpreter state")
        return
    if name == "symmetry_activation_audit":
        _require(payload.get("schema") ==
                 "pallet_n3_completion_v3_symmetry_activation_audit_v1"
                 and payload.get("complete") is True,
                 "symmetry activation audit schema/completion drift")
        algorithm = payload.get("algorithm", {})
        _require(algorithm.get("whole_object_permutation") is True
                 and algorithm.get("identity_first_exact_tie") is True
                 and algorithm.get("inference_use") is False,
                 "symmetry selector contract drift")
        backbones = payload.get("backbones")
        _require(isinstance(backbones, Mapping)
                 and set(backbones) == set(BACKBONES),
                 "symmetry audit backbone inventory drift")
        for backbone in BACKBONES:
            row = backbones[backbone]
            _validate_count_summary(row.get("unique_usable_rows", {}))
            seeds = row.get("seeds")
            _require(isinstance(seeds, Mapping)
                     and set(seeds) == {str(seed) for seed in SEEDS},
                     f"{backbone} symmetry seed inventory drift")
            for seed in SEEDS:
                _validate_count_summary(
                    seeds[str(seed)], expected=EXPECTED_EXPOSURES_PER_FIT)
        _require(payload.get("interpretation", {}).get("objective_applied") is True,
                 "symmetry audit does not confirm objective application")
        return
    if name == "dimension_sensitivity_audit":
        _require(payload.get("schema") ==
                 "pallet_n3_completion_v3_dimension_sensitivity_audit_v1"
                 and payload.get("complete") is True,
                 "dimension sensitivity audit schema/completion drift")
        fits = payload.get("fits")
        expected = {f"{backbone}_seed{seed}"
                    for backbone, seed in EXPECTED_FITS}
        _require(isinstance(fits, Mapping) and set(fits) == expected,
                 "dimension audit fit inventory drift")
        for key, row in fits.items():
            _require(row.get("status") == "PASS"
                     and row.get("visual_inputs_bitwise_identical") is True
                     and row.get("base_logits_bitwise_identical") is True
                     and row.get("finite") is True
                     and isinstance(row.get("changed_logit_entries"), int)
                     and row["changed_logit_entries"] > 0,
                     f"dimension path is inactive or invalid: {key}")
        summary = payload.get("summary", {})
        _require(summary.get("status") == "PASS"
                 and summary.get("fits_verified") == len(EXPECTED_FITS)
                 and summary.get("all_six_changed") is True,
                 "dimension sensitivity summary drift")


def _validate_runtime_raw_nullable_pose(
        name: str, payload: Mapping[str, Any]) -> set[str]:
    """Return only schema-valid runtime ``output.pose`` null paths.

    Runtime raw rows time a fixed call even when prediction-only PnP is
    unavailable; the N3-only path deliberately excludes PnP.  A null in that
    exact field is therefore missing runtime output rather than an unexplained
    numeric result.  No null outside the two locked row lists is exempted.
    """
    _require(name in {"runtime_dope_raw", "runtime_resnet18_raw"},
             "runtime nullable-pose validator used for a non-runtime artifact")
    _require(payload.get("schema") == "pallet_n3_completion_v3_runtime_raw_v1"
             and payload.get("complete") is True,
             "runtime raw schema/completion drift")
    allowed: set[str] = set()
    valid_paths = {"base_e2e", "n3_seed1_e2e", "n3_seed1_only"}
    for collection in ("warmup", "measurements"):
        rows = payload.get(collection)
        _require(isinstance(rows, list), f"runtime raw {collection} rows are missing")
        for index, row in enumerate(rows):
            _require(isinstance(row, Mapping) and row.get("path") in valid_paths,
                     f"runtime raw {collection} row path is invalid")
            output = row.get("output")
            _require(isinstance(output, Mapping),
                     f"runtime raw {collection} row output is invalid")
            pose = output.get("pose")
            null_path = f"$.{collection}[{index}].output.pose"
            if pose is None:
                allowed.add(null_path)
                continue
            _require(isinstance(pose, Mapping)
                     and isinstance(pose.get("available"), bool)
                     and pose.get("status") in {"OK", "UNAVAILABLE"}
                     and ((pose["status"] == "OK") == pose["available"]),
                     f"runtime raw {collection} pose status is invalid")
    return allowed


def inspect_artifact(root: Path, roots: Sequence[Path], name: str, relative: str,
                     category: str, json_file: bool) -> dict:
    path = _find_relative(Path(relative), roots)
    if path is None:
        return {**_missing("required artifact does not exist"),
                "name": name, "path": relative, "category": category}
    audit = {"name": name, "path": relative, "category": category,
             "binding": binding(path, root), "embedded_bindings_verified": 0}
    if not json_file:
        if category == "report" and name.startswith("report_tex_"):
            try:
                _validate_report_tex(path, roots)
            except Exception as error:
                audit.update(_missing(str(error), status="BLOCKED_INTEGRITY"))
                return audit
        audit["status"] = "PASS"
        return audit
    try:
        payload = C.read(path)
        embedded = list(iter_bindings(payload))
        for _, entry in embedded:
            resolve_binding(entry, roots)
        allowed_nulls: set[str] = set()
        if name in {"runtime_dope_raw", "runtime_resnet18_raw"}:
            allowed_nulls = _validate_runtime_raw_nullable_pose(name, payload)
        issues = [issue for issue in explicit_x_issues(payload)
                  if issue not in allowed_nulls]
        _require(not issues, "unexplained numeric nulls: " + ", ".join(issues[:10]))
        if category == "protocol":
            _validate_protocol_artifact(name, payload)
        if category == "lifter":
            _validate_lifter_x(payload)
        if name == "report_figure_manifest":
            _validate_report_figure_manifest(payload, path)
        remaining = contract_remaining_x(name, category, payload)
        complete = payload.get("complete")
        status = str(payload.get("status", ""))
        if (category == "report" and complete is False
                and payload.get("overall_status") == "PARTIAL"):
            audit["status"] = "PASS"
            audit["declared_status"] = "PARTIAL_WITH_EXPLICIT_X"
        elif complete is False:
            # A result is allowed to remain partial only when every missing
            # value is already explicit x. It cannot support OVERALL_COMPLETE.
            audit["status"] = "INCOMPLETE_DECLARED_X"
            audit.update(value=None, display="x",
                         reason="artifact declares incomplete evaluation")
        elif complete is True:
            audit["status"] = "PASS"
        elif status:
            _require(status != "BLOCKED_INTEGRITY", "artifact reports BLOCKED_INTEGRITY")
            audit["status"] = "PASS"
            audit["declared_status"] = status
        else:
            # Raw per-frame/source-binding JSON files do not all carry a
            # completion flag; their parent receipt is a separate artifact.
            audit["status"] = "PASS"
        audit["embedded_bindings_verified"] = len(embedded)
        if allowed_nulls:
            audit["schema_valid_nullable_runtime_pose_count"] = len(allowed_nulls)
        if remaining:
            audit["remaining_x"] = remaining
        return audit
    except Exception as error:
        audit.update(_missing(str(error), status="BLOCKED_INTEGRITY"))
        return audit


def inspect_artifacts(root: Path | str = C.ROOT,
                      specs: Sequence[tuple[str, str, str, bool]] = ARTIFACT_SPECS) -> dict:
    root = Path(root).resolve()
    roots = checkout_roots(root)
    groups: dict[str, dict[str, dict]] = {}
    for name, relative, category, json_file in specs:
        groups.setdefault(category, {})[name] = inspect_artifact(
            root, roots, name, relative, category, json_file)
    output = {}
    for category, entries in groups.items():
        statuses = [row["status"] for row in entries.values()]
        status = ("BLOCKED_INTEGRITY" if "BLOCKED_INTEGRITY" in statuses else
                  "PASS" if all(value == "PASS" for value in statuses) else "INCOMPLETE")
        output[category] = {"status": status, "artifacts": entries}
        if status != "PASS":
            output[category].update(value=None, display="x",
                                    reason=f"{category} artifacts are not all complete and verified")
    return output


def _overall_status(protocol: Mapping, training: Mapping, sensitivity: Mapping,
                    base: Mapping, artifacts: Mapping[str, Mapping],
                    remaining_x: Sequence[Mapping[str, Any]] = ()) -> str:
    statuses = [protocol["status"], training["status"], sensitivity["status"],
                base["status"], *(value["status"] for value in artifacts.values())]
    if "BLOCKED_INTEGRITY" in statuses:
        return "BLOCKED_INTEGRITY"
    return ("OVERALL_COMPLETE" if all(value == "PASS" for value in statuses)
            and not remaining_x else "PARTIAL")


def _collect_remaining_x(artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """Collect and de-duplicate semantic x records from audited artifacts."""
    collected: dict[str, dict] = {}
    for group in artifacts.values():
        for artifact in group.get("artifacts", {}).values():
            for row in artifact.get("remaining_x", []):
                collected.setdefault(str(row["id"]), dict(row))
    return list(collected.values())


def verify(root: Path | str = C.ROOT, output_path: Path | str = OUTPUT) -> dict:
    root = Path(root).resolve()
    roots = checkout_roots(root)
    protocol_relative = Path("_docs/experiments/pallet_n3_completion_v3/PROTOCOL.json")
    protocol_path = _find_relative(protocol_relative, roots)
    if protocol_path is None:
        raise FileNotFoundError(protocol_relative)
    protocol_payload = C.read(protocol_path)
    protocol_audit = validate_protocol_payload(protocol_payload)
    protocol_audit["binding"] = binding(protocol_path, root)
    protocol_bindings = list(iter_bindings(protocol_payload.get("bindings", [])))
    for _, entry in protocol_bindings:
        resolve_binding(entry, roots)
    protocol_audit["embedded_bindings_verified"] = len(protocol_bindings)

    training, states, _ = inspect_training(root, protocol_payload, protocol_path)
    normalization_path = _find_relative(
        Path("_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"), roots)
    if normalization_path is None:
        sensitivity = _missing("dimension normalization lock is missing",
                               status="BLOCKED_INTEGRITY")
    else:
        sensitivity = inspect_sensitivity(training, states, C.read(normalization_path))
        sensitivity["normalization"] = binding(normalization_path, root)
    base = prove_base_rgb_only(root, protocol_payload, training)
    artifacts = inspect_artifacts(root)
    remaining_x = _collect_remaining_x(artifacts)
    component_statuses = [protocol_audit["status"], training["status"],
                          sensitivity["status"], base["status"],
                          *(value["status"] for value in artifacts.values())]
    integrity_status = ("BLOCKED_INTEGRITY" if "BLOCKED_INTEGRITY" in component_statuses
                        else "PASS" if all(value == "PASS" for value in component_statuses)
                        else "INCOMPLETE")
    status = _overall_status(
        protocol_audit, training, sensitivity, base, artifacts, remaining_x)
    result = {
        "schema": SCHEMA, "status": status,
        "integrity_status": integrity_status,
        "complete": status == "OVERALL_COMPLETE",
        "policy": {
            "missing": "numeric value null plus status/reason; display x",
            "zero": "a measured zero remains numeric 0",
            "undefined_or_empty": "NA, distinct from x and zero",
            "integrity_failure": "BLOCKED_INTEGRITY, never converted to x-only success",
        },
        "protocol": protocol_audit, "training": training,
        "dimension_context_sensitivity": sensitivity,
        "base_rgb_only": base, "artifacts": artifacts,
        "remaining_x": remaining_x,
        "expected_budget": {
            "fits": 6, "updates": 36_000, "sample_exposures": EXPECTED_TOTAL_EXPOSURES,
            "per_fit": {"steps": EXPECTED_STEPS, "batch": EXPECTED_BATCH,
                        "sample_exposures": EXPECTED_EXPOSURES_PER_FIT},
        },
    }
    if status != "OVERALL_COMPLETE":
        result.update(value=None, display="x",
                      reason="one or more required final checks remain incomplete or blocked")
    output_path = Path(output_path)
    if output_path.resolve().parent != (root / "_docs/experiments/pallet_n3_completion_v3").resolve():
        raise ValueError("VERIFY_RESULTS.json must stay in the N3 documentation directory")
    C.write(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    arguments = parser.parse_args(argv)
    result = verify()
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "integrity_status": result["integrity_status"],
        "remaining_x": len(result["remaining_x"]),
        "verified_fits": result["training"]["verified_fits"],
        "expected_fits": result["training"]["expected_fits"],
        "verified_exposures": result["training"]["verified_total_exposures"],
        "expected_exposures": result["training"]["expected_total_exposures"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "OVERALL_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
