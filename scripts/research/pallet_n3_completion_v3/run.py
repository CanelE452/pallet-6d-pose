"""Small hash-bound orchestrator for the frozen pallet N3 completion contract.

The orchestrator deliberately contains no training or evaluation logic.  It
only invokes CLIs that are present in this package, records a content-addressed
receipt, and resumes a stage only when every bound input, code file, config and
output still has the recorded bytes.  ``status`` performs file/process
inspection only and never initializes CUDA.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


PACKAGE = "scripts.research.pallet_n3_completion_v3"
BACKBONES = ("dope", "resnet18")
PER_BACKBONE = frozenset({
    "smoke", "train", "calibrate", "infer", "evaluate", "square", "runtime",
})
GLOBAL_STAGES = frozenset({
    "preflight", "tests", "square-yolo", "lifter", "reuse", "audit", "report",
    "verify", "manifest",
})
STAGES = (
    "status", "preflight", "tests", "smoke", "train", "calibrate", "infer",
    "evaluate", "square", "square-yolo", "runtime", "lifter", "reuse",
    "audit", "report", "verify", "manifest", "all",
)
ALL_ORDER = (
    "preflight", "tests", "smoke", "train", "calibrate", "infer", "evaluate",
    "square", "square-yolo", "runtime", "lifter", "reuse", "audit", "report",
    "verify", "manifest",
)
# Verification consumes the generated report.  The report deliberately does
# not consume VERIFY_RESULTS, which keeps the dependency graph acyclic.  The
# repeated report/verify pass is retained as an explicit convergence check;
# valid receipts make both repeated stages content-addressed skips.
ALL_EXECUTION_ORDER = (*ALL_ORDER[:-1], "report", "verify", "manifest")
SCHEMA = "pallet_n3_completion_v3_orchestrator_receipt_v1"
TEST_SCHEMA = "pallet_n3_completion_v3_test_results_v1"
DOC_REL = Path("_docs/experiments/pallet_n3_completion_v3")
RAW_REL = Path("data/pallet/results/pallet_n3_completion_v3")
HERE_REL = Path("scripts/research/pallet_n3_completion_v3")
REPORT_OUTPUTS = (
    DOC_REL / "TABLES.json",
    DOC_REL / "TABLES.md",
    DOC_REL / "REMAINING_X.md",
    DOC_REL / "FINAL_REPORT_KO.md",
)
REPORT_FIGURES = (
    "training_curves.png", "backbone_comparison.png", "runtime.png",
    "dev_overlays.png", "lifter.png", "subgroups_or_thresholds.png",
)
REPORT_TEX = ("backbone_dev_headline.tex", "square_2d.tex", "runtime.tex")
REPORT_RAW = ("FRAME_METRICS.csv", "CORNER_METRICS.csv")
STARTING_HEAD = "d1524b3"
_KNOWN_YOLO26_PYTHON = Path(
    "/home/minjae/anaconda3/envs/pallet-yolo26/bin/python")
DEFAULT_YOLO_PYTHON = os.environ.get(
    "PALLET_YOLO26_PYTHON",
    str(_KNOWN_YOLO26_PYTHON if _KNOWN_YOLO26_PYTHON.is_file()
        else Path(sys.executable)),
)


class StageError(RuntimeError):
    """A stage could not produce its declared, hashable outputs."""


class StageBlocked(StageError):
    """A declared prerequisite or intentionally external stage is absent."""


@dataclass(frozen=True)
class StagePlan:
    name: str
    backbone: str | None
    commands: tuple[tuple[str, ...], ...]
    inputs: tuple[str, ...]
    code: tuple[str, ...]
    outputs: tuple[str, ...]
    config: Mapping[str, Any]
    gpu: bool = False
    internal: str | None = None
    accepted_returncodes: tuple[int, ...] = (0,)

    @property
    def stage_id(self) -> str:
        suffix = f"_{self.backbone}" if self.backbone else ""
        return f"{self.name.replace('-', '_')}{suffix}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _checkout_roots(root: Path) -> tuple[Path, ...]:
    """Return this checkout plus the common checkout of a linked worktree."""
    roots = [root.resolve()]
    dotgit = root / ".git"
    try:
        if dotgit.is_file():
            line = dotgit.read_text().strip()
            if line.startswith("gitdir:"):
                gitdir = Path(line.split(":", 1)[1].strip()).resolve()
                common = (gitdir / "commondir").read_text().strip()
                common_git = (gitdir / common).resolve()
                candidate = common_git.parent
                if candidate.is_dir() and candidate not in roots:
                    roots.append(candidate)
    except (OSError, ValueError):
        pass
    return tuple(roots)


def _logical(value: Path | str) -> str:
    return Path(value).as_posix()


def _resolve(root: Path, logical: Path | str,
             *, output: bool = False, expected_sha: str | None = None) -> Path | None:
    path = Path(logical)
    candidates = (path,) if path.is_absolute() else tuple(
        checkout / path for checkout in ((root.resolve(),) if output else _checkout_roots(root)))
    for candidate in candidates:
        if candidate.is_file() and (expected_sha is None or _sha256(candidate) == expected_sha):
            return candidate
    return None


def file_binding(root: Path, logical: Path | str, *, output: bool = False) -> dict:
    logical_text = _logical(logical)
    path = _resolve(root, logical_text, output=output)
    if path is None:
        raise FileNotFoundError(logical_text)
    return {"path": logical_text, "sha256": _sha256(path), "bytes": path.stat().st_size}


def binding_valid(root: Path, entry: Mapping[str, Any], *, output: bool = False) -> bool:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        return False
    digest = entry.get("sha256")
    size = entry.get("bytes")
    if not isinstance(digest, str) or not isinstance(size, int):
        return False
    path = _resolve(root, entry["path"], output=output, expected_sha=digest)
    return path is not None and path.stat().st_size == size


def _manifest_hash(entries: Sequence[Mapping[str, Any]]) -> str:
    return _object_hash(list(entries))


def _unique(values: Iterable[Path | str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_logical(value) for value in values))


def _module_file(name: str) -> Path:
    return HERE_REL / f"{name}.py"


def _command(python: str, module: str, *arguments: str) -> tuple[str, ...]:
    return (python, "-m", f"{PACKAGE}.{module}", *arguments)


def _training_inputs(backbone: str) -> list[Path]:
    base = (Path("weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth")
            if backbone == "dope" else
            Path("data/pallet/results/pallet_resnet18_dimension_20261002_v1/DSNT_FULL_CONSTANT_epoch10.pt"))
    return [
        DOC_REL / "PROTOCOL.json", DOC_REL / "PREFLIGHT.json",
        Path("data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"),
        Path("data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"),
        Path("_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"),
        Path("challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"), base,
    ]


def _fit_outputs(backbone: str, *, smoke: bool) -> list[Path]:
    upper = backbone.upper()
    if smoke:
        return [
            DOC_REL / f"SMOKE_{upper}_SEED1.json",
            DOC_REL / f"SMOKE_{upper}_COMPLETE.json",
            RAW_REL / "smoke" / backbone / "seed1" / "last.pt",
        ]
    outputs: list[Path] = [DOC_REL / f"TRAINING_{upper}_COMPLETE.json"]
    for seed in (1, 2, 3):
        outputs.extend((
            DOC_REL / f"TRAIN_{upper}_SEED{seed}.json",
            RAW_REL / "runs" / backbone / f"seed{seed}" / "last.pt",
            RAW_REL / "orders" / f"{backbone}_seed{seed}.npy",
        ))
    return outputs


def _base_code(stage: str) -> list[Path]:
    names = {
        "preflight": ("preflight", "environment", "common"),
        "tests": ("run",),
        "smoke": ("train", "model", "data", "adapters", "common"),
        "train": ("train", "model", "data", "adapters", "common"),
        "calibrate": ("selection", "train", "model", "data", "adapters", "common"),
        "infer": ("inference", "selection", "model", "adapters", "common"),
        "evaluate": ("evaluation", "metrics", "common"),
        "square": ("square", "evaluation", "metrics", "common"),
        "square-yolo": ("square_yolo", "evaluation", "metrics", "common"),
        "runtime": ("runtime", "inference", "selection", "model", "adapters", "common"),
        "lifter": ("lifter_run", "common"),
        "reuse": ("reuse", "metrics", "common"),
        "audit": ("symmetry_activation_audit", "dimension_sensitivity_audit",
                  "integrity", "model", "data", "common"),
        "report": ("report", "run"),
        "verify": ("integrity", "common"),
        "manifest": ("run", "common"),
    }[stage]
    # The evaluator delegates prediction-only PnP to this frozen historical
    # source through metrics._pose_contract(); there is deliberately no local
    # pallet_n3_completion_v3/pose.py shim.
    external = {
        "evaluate": (
            Path("scripts/research/pallet_dim_conditioned_p_v1/pose.py"),
        ),
        "audit": (
            Path("scripts/research/pallet_dope_refiner_20261001_v1/refiner.py"),
        ),
    }
    # run.py is itself part of every stage's restart decision.
    return [
        *(_module_file(name) for name in dict.fromkeys((*names, "run"))),
        *external.get(stage, ()),
    ]


def _report_inputs() -> tuple[Path, ...]:
    values: list[Path] = [
        DOC_REL / "REUSE_RESULTS.json",
        RAW_REL / "reuse" / "PER_FRAME_SCORES.json",
        RAW_REL / "evaluation" / "dope.json",
        RAW_REL / "evaluation" / "resnet18.json",
        DOC_REL / "SQUARE_YOLO_RESULTS.json", DOC_REL / "SQUARE_DOPE_RESULTS.json",
        DOC_REL / "SQUARE_RESNET18_RESULTS.json",
        DOC_REL / "RUNTIME_DOPE_SEED1.json", DOC_REL / "RUNTIME_RESNET18_SEED1.json",
        RAW_REL / "runtime" / "dope_seed1.json",
        RAW_REL / "runtime" / "resnet18_seed1.json",
        DOC_REL / "LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json",
        RAW_REL / "lifter" / "YOLO_R0_N3_SEED1_RAW.json",
        RAW_REL / "predictions" / "dope_DEV319.json",
        RAW_REL / "predictions" / "resnet18_DEV319.json",
    ]
    for backbone in ("DOPE", "RESNET18"):
        values.extend(DOC_REL / f"TRAIN_{backbone}_SEED{seed}.json"
                      for seed in (1, 2, 3))
        values.extend((DOC_REL / f"TRAINING_{backbone}_COMPLETE.json",
                       DOC_REL / f"VALIDATION_{backbone}_COMPLETE.json",
                       DOC_REL / f"SELECTION_{backbone}.json"))
    return tuple(values)


def _report_outputs() -> tuple[Path, ...]:
    return (*REPORT_OUTPUTS,
            *(DOC_REL / "figures" / name for name in REPORT_FIGURES),
            DOC_REL / "figures" / "SELECTION_MANIFEST.json",
            *(DOC_REL / "table_fragments" / name for name in REPORT_TEX),
            *(RAW_REL / "report" / name for name in REPORT_RAW))


def _audit_inputs() -> tuple[Path, ...]:
    """Every file read by the two CPU audit generators.

    The environment audit is produced by preflight and is a prerequisite for
    this global gate.  Array/checkpoint paths are explicit so a receipt cannot
    be reused after a cache, fit, or interpreter-boundary audit changes.
    """
    array_keys = ("points", "point_valid", "boxes", "gt_points", "gt_valid")
    values: list[Path] = [
        DOC_REL / "PROTOCOL.json", DOC_REL / "ENVIRONMENT_AUDIT.json",
        Path("data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"),
        Path("_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"),
        Path("_docs/experiments/pallet_dope_refiner_20261001_v1/SOURCE_CACHE_COMPLETE.json"),
        DOC_REL / "RESNET18_CONSTANT_CACHE_COMPLETE.json",
    ]
    cache_roots = {
        "dope": Path("data/pallet/results/pallet_dope_refiner_20261001_v1/cache"),
        "resnet18": RAW_REL / "resnet18_constant_cache",
    }
    for backbone in BACKBONES:
        upper = backbone.upper()
        values.append(DOC_REL / f"TRAINING_{upper}_COMPLETE.json")
        values.extend(cache_roots[backbone] / f"{key}.npy" for key in array_keys)
        for seed in (1, 2, 3):
            values.extend((
                DOC_REL / f"TRAIN_{upper}_SEED{seed}.json",
                RAW_REL / "runs" / backbone / f"seed{seed}" / "last.pt",
                RAW_REL / "orders" / f"{backbone}_seed{seed}.npy",
            ))
    return tuple(values)


def _manifest_candidates() -> tuple[Path, ...]:
    values: list[Path] = [
        DOC_REL / "PROTOCOL.json", DOC_REL / "PREFLIGHT.json",
        DOC_REL / "ENVIRONMENT_AUDIT.json",
        DOC_REL / "SYMMETRY_ACTIVATION_AUDIT.json",
        DOC_REL / "DIMENSION_SENSITIVITY_AUDIT.json",
        DOC_REL / "REUSE_RESULTS.json", DOC_REL / "VERIFY_RESULTS.json",
        Path("data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"),
        Path("data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"),
        Path("_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"),
        Path("challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"),
        Path("challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"),
        Path("weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"),
        Path("data/pallet/results/pallet_resnet18_dimension_20261002_v1/DSNT_FULL_CONSTANT_epoch10.pt"),
        RAW_REL / "TRAIN_PROGRESS.json", *_report_outputs(),
    ]
    for backbone in BACKBONES:
        values.extend(_fit_outputs(backbone, smoke=False))
        values.extend((DOC_REL / f"SELECTION_{backbone.upper()}.json",
                       RAW_REL / "predictions" / f"{backbone}_DEV319.json",
                       RAW_REL / "predictions" / f"{backbone}_GREEN0918_119.json",
                       RAW_REL / "evaluation" / f"{backbone}.json",
                       RAW_REL / "runtime" / f"{backbone}_seed1.json"))
    values.extend((RAW_REL / "SQUARE_YOLO_PREDICTIONS.json",
                   RAW_REL / "SQUARE_YOLO_METRICS.json",
                   RAW_REL / "lifter" / "FIXED_SENSOR_1000MS_PLAN.json",
                   RAW_REL / "lifter" / "YOLO_R0_N3_SEED1_RAW.json",
                   RAW_REL / "reuse" / "PER_FRAME_SCORES.json",
                   RAW_REL / "reuse" / "SOURCE_BINDINGS.json"))
    return tuple(dict.fromkeys(values))


def build_plan(name: str, backbone: str | None = None, *,
               python: str | None = None, device: str = "cuda",
               yolo_python: str | None = None,
               capture_root: str | None = None, source_root: str | None = None,
               root: Path | None = None) -> StagePlan:
    """Build one concrete plan using only entry points present in this package."""
    if name not in set(ALL_ORDER):
        raise ValueError(name)
    if name in PER_BACKBONE and backbone not in BACKBONES:
        raise ValueError(f"{name} needs one backbone")
    if name in GLOBAL_STAGES:
        backbone = None
    python = str(Path(python or sys.executable).resolve())
    # YOLO26 checkpoints require Ultralytics 8.4.x (C3k2/Pose26), whereas the
    # DOPE/ResNet adapter environment intentionally retains Ultralytics 8.0.x.
    # Keep this interpreter boundary explicit and receipt-bound.
    yolo_python = str(Path(yolo_python or python).resolve())
    stage_python = yolo_python if name in {"square-yolo", "lifter"} else python
    root = Path(root or repository_root()).resolve()
    commands: list[tuple[str, ...]] = []
    inputs: list[Path | str] = []
    outputs: list[Path | str] = []
    gpu = False
    internal: str | None = None
    accepted_returncodes = (0,)
    config: dict[str, Any] = {
        "stage": name, "backbone": backbone,
        "python_executable": stage_python,
    }

    if name == "preflight":
        commands = [
            _command(python, "environment", "--primary-python", python,
                     "--yolo-python", yolo_python),
            _command(python, "preflight"),
        ]
        inputs = [
            Path("_docs/experiments/pallet_final_paper_tables_v1/TABLES.json"),
            Path("data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"),
            Path("data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"),
            Path("_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"),
            Path("challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"),
            Path("weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"),
            Path("data/pallet/results/pallet_resnet18_dimension_20261002_v1/DSNT_FULL_CONSTANT_epoch10.pt"),
        ]
        outputs = [DOC_REL / "ENVIRONMENT_AUDIT.json", DOC_REL / "PREFLIGHT.json"]
    elif name == "tests":
        commands = [(python, "-m", "pytest", "-q", str(HERE_REL))]
        inputs = [DOC_REL / "PROTOCOL.json"]
        outputs = [DOC_REL / "TEST_RESULTS.json"]
        internal = "tests"
    elif name == "smoke":
        commands = [_command(python, "train", "smoke", str(backbone))]
        inputs = _training_inputs(str(backbone))
        outputs = _fit_outputs(str(backbone), smoke=True)
        gpu = True
        config.update({"steps": 2, "seed": 1, "discarded": True, "device": device})
    elif name == "train":
        commands = [_command(python, "train", "train", str(backbone))]
        inputs = [*_training_inputs(str(backbone)),
                  DOC_REL / f"SMOKE_{str(backbone).upper()}_COMPLETE.json"]
        outputs = _fit_outputs(str(backbone), smoke=False)
        gpu = True
        config.update({"seeds": [1, 2, 3], "steps_per_fit": 6000,
                       "batch": 16, "exposures_per_fit": 96000,
                       "precision": "FP32", "TF32": False, "device": device})
    elif name == "calibrate":
        commands = [
            _command(python, "selection", "validation", str(backbone)),
            _command(python, "selection", "calibrate", str(backbone)),
        ]
        inputs = [DOC_REL / "PROTOCOL.json", *_fit_outputs(str(backbone), smoke=False)]
        outputs = [DOC_REL / f"VALIDATION_{str(backbone).upper()}_COMPLETE.json",
                   *(RAW_REL / "validation" / f"{backbone}_seed{seed}.npz"
                     for seed in (1, 2, 3)),
                   DOC_REL / f"SELECTION_{str(backbone).upper()}.json"]
        gpu = True
        config.update({"temperatures": [0.5, 1.0, 2.0, 4.0], "device": device})
    elif name == "infer":
        commands = [
            _command(python, "inference", str(backbone), dataset, "--device", device)
            for dataset in ("DEV319", "GREEN0918_119")]
        inputs = [DOC_REL / "PROTOCOL.json",
                  DOC_REL / f"TRAINING_{str(backbone).upper()}_COMPLETE.json",
                  DOC_REL / f"SELECTION_{str(backbone).upper()}.json",
                  Path("challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"),
                  Path("_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json")]
        outputs = []
        for dataset in ("DEV319", "GREEN0918_119"):
            outputs.extend((
                RAW_REL / "predictions" / f"{backbone}_{dataset}.json",
                DOC_REL / f"INFERENCE_{str(backbone).upper()}_{dataset}_COMPLETE.json",
            ))
        gpu = True
        config.update({"device": device, "datasets": ["DEV319", "GREEN0918_119"],
                       "GT_inputs": False})
    elif name == "evaluate":
        prediction = RAW_REL / "predictions" / f"{backbone}_DEV319.json"
        commands = [_command(python, "evaluation", str(backbone),
                             "--predictions", str(prediction))]
        inputs = [prediction, Path("challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json")]
        outputs = [RAW_REL / "evaluation" / f"{backbone}.json"]
        config.update({"dataset": "DEV319", "corners": 8, "pose": True,
                       "bootstrap_sessions": 10000, "bootstrap_seed": 20260917})
    elif name == "square":
        commands = [_command(python, "square", str(backbone))]
        inputs = [RAW_REL / "predictions" / f"{backbone}_GREEN0918_119.json",
                  Path("_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json")]
        outputs = [DOC_REL / f"SQUARE_{str(backbone).upper()}_RESULTS.json"]
        config.update({"dataset": "GREEN0918_119", "dimensions_wdh_m": [1.1, 1.1, 0.15],
                       "pose": None})
    elif name == "square-yolo":
        commands = [_command(stage_python, "square_yolo", "all")]
        inputs = [
            Path("_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json"),
            Path("data/pallet/results/pallet_green0918_dimension_audit_v1/PREDICTIONS.json"),
            Path("data/pallet/results/pallet_green0918_dimension_audit_v1/METRICS.json"),
        ]
        outputs = [RAW_REL / "SQUARE_YOLO_PREDICTIONS.json",
                   RAW_REL / "SQUARE_YOLO_METRICS.json", DOC_REL / "SQUARE_YOLO_RESULTS.json"]
        gpu = True
        config.update({"dataset": "GREEN0918_119", "fixed_square": True,
                       "reselection": False, "device": device})
    elif name == "runtime":
        commands = [_command(python, "runtime", "measure", str(backbone), "--device", device)]
        inputs = [DOC_REL / "PROTOCOL.json",
                  DOC_REL / f"TRAIN_{str(backbone).upper()}_SEED1.json",
                  DOC_REL / f"TRAINING_{str(backbone).upper()}_COMPLETE.json",
                  DOC_REL / f"SELECTION_{str(backbone).upper()}.json",
                  Path("challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json")]
        outputs = [RAW_REL / "runtime" / f"{backbone}_seed1.json",
                   DOC_REL / f"RUNTIME_{str(backbone).upper()}_SEED1.json"]
        gpu = True
        config.update({"device": device, "seed": 1, "Jetson": False})
    elif name == "lifter":
        root_arg = (() if capture_root is None else ("--capture-root", capture_root))
        commands = [
            _command(stage_python, "lifter_run", stage, *root_arg)
            for stage in ("plan", "run", "verify")]
        inputs = [DOC_REL / "PROTOCOL.json",
                  Path("_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json")]
        outputs = [RAW_REL / "lifter" / "FIXED_SENSOR_1000MS_PLAN.json",
                   RAW_REL / "lifter" / "YOLO_R0_N3_SEED1_RAW.json",
                   DOC_REL / "LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json"]
        gpu = True
        config.update({"capture_root": capture_root, "actual_control_invoked": False,
                       "device": device})
    elif name == "reuse":
        extra = (() if source_root is None else ("--source-root", source_root))
        commands = [_command(python, "reuse", *extra)]
        inputs = [DOC_REL / "PROTOCOL.json"]
        outputs = [DOC_REL / "REUSE_RESULTS.json", DOC_REL / "REUSE_RESULTS.csv",
                   RAW_REL / "reuse" / "PER_FRAME_SCORES.json",
                   RAW_REL / "reuse" / "SOURCE_BINDINGS.json"]
        config.update({"source_root": source_root, "contracts": ["A", "B", "C", "D", "E", "H", "I"]})
    elif name == "audit":
        commands = [
            _command(python, "symmetry_activation_audit"),
            _command(python, "dimension_sensitivity_audit"),
        ]
        inputs = list(_audit_inputs())
        outputs = [DOC_REL / "SYMMETRY_ACTIVATION_AUDIT.json",
                   DOC_REL / "DIMENSION_SENSITIVITY_AUDIT.json"]
        config.update({
            "CPU_only": True,
            "environment_audit_required": True,
            "symmetry_selector": "exact model.select_symmetric_target",
            "dimension_probe": "all six final step6000 N3 heads",
            "actual_accuracy_used": False,
        })
    elif name == "report":
        report_module = root / HERE_REL / "report.py"
        if report_module.is_file():
            commands = [_command(python, "report", "generate")]
        else:
            internal = "report_gate"
        expected_inputs = _report_inputs()
        inputs = [path for path in expected_inputs if _resolve(root, path) is not None]
        outputs = list(_report_outputs())
        config.update({"missing_values": "x", "language": "ko",
                       "implemented_report_cli": report_module.is_file(),
                       "input_presence": {
                           _logical(path): _resolve(root, path) is not None
                           for path in expected_inputs},
                       "output_inventory": [_logical(path) for path in outputs]})
    elif name == "verify":
        commands = [_command(python, "integrity", "verify")]
        inputs = [DOC_REL / "PROTOCOL.json", DOC_REL / "REUSE_RESULTS.json",
                  DOC_REL / "ENVIRONMENT_AUDIT.json",
                  DOC_REL / "SYMMETRY_ACTIVATION_AUDIT.json",
                  DOC_REL / "DIMENSION_SENSITIVITY_AUDIT.json",
                  *_report_outputs()]
        outputs = [DOC_REL / "VERIFY_RESULTS.json"]
        config.update({"strict": True, "missing_policy": "explicit_x"})
        accepted_returncodes = (0, 2)
    elif name == "manifest":
        expected = _manifest_candidates()
        orchestrator = root / DOC_REL / "orchestrator"
        receipt_files = tuple(
            path.relative_to(root) for path in sorted(orchestrator.glob("*.json"))
            if path.name != "manifest.json") if orchestrator.is_dir() else ()
        expected = (*expected, *receipt_files)
        inputs = [path for path in expected if _resolve(root, path) is not None]
        outputs = [DOC_REL / "RUN_MANIFEST.json"]
        internal = "manifest"
        config.update({
            "starting_head": STARTING_HEAD,
            "observed_git": {
                "head": _git_text(root, "rev-parse", "HEAD"),
                "branch": _git_text(root, "branch", "--show-current"),
                "status_sha256": _object_hash(
                    (_git_text(root, "status", "--short") or "").splitlines()),
            },
            "overall_status": "PARTIAL",
            "input_presence": {_logical(path): _resolve(root, path) is not None
                               for path in expected},
            "scope_exclusions": ["PDF generation", "actual lifter control",
                                 "new self-training"],
        })

    return StagePlan(
        name=name, backbone=backbone, commands=tuple(commands),
        inputs=_unique(inputs), code=_unique(_base_code(name)),
        outputs=_unique(outputs), config=config, gpu=gpu, internal=internal,
        accepted_returncodes=accepted_returncodes)


def expand_plans(stage: str, backbone: str, **options: Any) -> list[StagePlan]:
    selected = BACKBONES if backbone == "all" else (backbone,)
    names = ALL_ORDER if stage == "all" else (stage,)
    plans: list[StagePlan] = []
    for name in names:
        if name in PER_BACKBONE:
            plans.extend(build_plan(name, item, **options) for item in selected)
        else:
            plans.append(build_plan(name, None, **options))
    return plans


def receipt_path(root: Path, plan: StagePlan) -> Path:
    return root / DOC_REL / "orchestrator" / f"{plan.stage_id}.json"


def _bindings(root: Path, values: Sequence[str], *, output: bool = False) -> list[dict]:
    missing: list[str] = []
    bound: list[dict] = []
    for value in values:
        try:
            bound.append(file_binding(root, value, output=output))
        except FileNotFoundError:
            missing.append(value)
    if missing:
        raise StageBlocked("missing required files: " + ", ".join(missing))
    return bound


def _walk_bindings(value: Any) -> Iterable[dict]:
    if isinstance(value, Mapping):
        if (isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)
                and isinstance(value.get("bytes"), int)):
            yield {key: value[key] for key in ("path", "sha256", "bytes")}
        for child in value.values():
            yield from _walk_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_bindings(child)


def _embedded_inputs(root: Path, output_bindings: Sequence[Mapping[str, Any]]) -> list[dict]:
    output_paths = {str(entry["path"]) for entry in output_bindings}
    collected: dict[tuple[str, str, int], dict] = {}
    for output in output_bindings:
        path = _resolve(root, output["path"], output=True)
        if path is None or path.suffix.lower() != ".json":
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        for entry in _walk_bindings(payload):
            if entry["path"] in output_paths:
                continue
            if not binding_valid(root, entry):
                raise StageError(
                    f"output {output['path']} contains a missing or changed binding: "
                    f"{entry['path']}")
            key = (entry["path"], entry["sha256"], entry["bytes"])
            collected[key] = entry
    return [collected[key] for key in sorted(collected)]


def receipt_valid(root: Path, plan: StagePlan) -> tuple[bool, str]:
    path = receipt_path(root, plan)
    if not path.is_file():
        return False, "receipt missing"
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False, "receipt unreadable"
    if value.get("schema") != SCHEMA or value.get("complete") is not True:
        return False, "receipt incomplete"
    if (value.get("stage") != plan.name or value.get("backbone") != plan.backbone
            or value.get("commands") != [list(command) for command in plan.commands]
            or value.get("accepted_returncodes") != list(plan.accepted_returncodes)
            or value.get("config") != dict(plan.config)
            or value.get("config_sha256") != _object_hash(dict(plan.config))):
        return False, "plan/config changed"
    manifests = (("inputs", False), ("code", False), ("outputs", True))
    for key, output in manifests:
        entries = value.get(key)
        if not isinstance(entries, list) or not all(binding_valid(root, item, output=output) for item in entries):
            return False, f"{key} missing or hash changed"
        if value.get(f"{key}_sha256") != _manifest_hash(entries):
            return False, f"{key} manifest hash changed"
    if {entry["path"] for entry in value["code"]} != set(plan.code):
        return False, "code set changed"
    if not set(plan.inputs).issubset({entry["path"] for entry in value["inputs"]}):
        return False, "input set changed"
    if {entry["path"] for entry in value["outputs"]} != set(plan.outputs):
        return False, "output set changed"
    return True, "all input/code/config/output hashes match"


def _failure_receipt(root: Path, plan: StagePlan, *, started_at: str,
                     inputs: list[dict], code: list[dict], returncodes: list[int],
                     error: str) -> None:
    value = {
        "schema": SCHEMA, "complete": False, "status": "FAILED",
        "stage": plan.name, "backbone": plan.backbone,
        "started_at": started_at, "finished_at": _utc_now(),
        "commands": [list(command) for command in plan.commands],
        "accepted_returncodes": list(plan.accepted_returncodes),
        "returncodes": returncodes, "config": dict(plan.config),
        "config_sha256": _object_hash(dict(plan.config)),
        "inputs": inputs, "inputs_sha256": _manifest_hash(inputs),
        "code": code, "code_sha256": _manifest_hash(code),
        "outputs": [], "outputs_sha256": _manifest_hash([]), "error": error,
    }
    _write_json(receipt_path(root, plan), value)


def _git_text(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, timeout=10,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _optional_binding(root: Path, logical: Path | str) -> dict:
    try:
        return {"status": "PRESENT", **file_binding(root, logical)}
    except FileNotFoundError:
        return {"status": "MISSING", "path": _logical(logical),
                "value": None, "display": "x"}


def build_run_manifest(root: Path, *, now: Callable[[], str] = _utc_now) -> dict:
    """Build the replayable environment/receipt inventory; no GPU API is used."""
    root = root.resolve()
    progress_path = root / RAW_REL / "TRAIN_PROGRESS.json"
    try:
        progress = _read_json(progress_path) if progress_path.is_file() else None
    except (OSError, json.JSONDecodeError):
        progress = {"status": "UNREADABLE"}
    progress_mtime = progress_path.stat().st_mtime if progress_path.is_file() else None
    observed_epoch = time.time()
    training = [_training_status(
        root, backbone, seed, progress, (), progress_mtime=progress_mtime,
        observed_epoch=observed_epoch)
        for backbone in BACKBONES for seed in (1, 2, 3)]
    complete = [row for row in training if row["status"] == "COMPLETE"]
    current_head = _git_text(root, "rev-parse", "HEAD")
    branch = _git_text(root, "branch", "--show-current")
    porcelain = _git_text(root, "status", "--short") or ""
    receipt_rows = []
    orchestrator = root / DOC_REL / "orchestrator"
    if orchestrator.is_dir():
        for path in sorted(orchestrator.glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError):
                payload = {}
            receipt_rows.append({
                "stage": payload.get("stage"), "backbone": payload.get("backbone"),
                "complete": payload.get("complete") is True,
                "binding": file_binding(root, path.relative_to(root), output=True),
            })
    raw_rows = []
    for logical in _manifest_candidates():
        if not _logical(logical).startswith(_logical(RAW_REL) + "/"):
            continue
        path = _resolve(root, logical)
        if path is not None:
            raw_rows.append(file_binding(root, logical))
    raw_rows = list({(row["path"], row["sha256"], row["bytes"]): row
                     for row in raw_rows}.values())
    bindings = {
        "protocol": _optional_binding(root, DOC_REL / "PROTOCOL.json"),
        "environment_audit": _optional_binding(
            root, DOC_REL / "ENVIRONMENT_AUDIT.json"),
        "symmetry_activation_audit": _optional_binding(
            root, DOC_REL / "SYMMETRY_ACTIVATION_AUDIT.json"),
        "dimension_sensitivity_audit": _optional_binding(
            root, DOC_REL / "DIMENSION_SENSITIVITY_AUDIT.json"),
        "source_manifest": _optional_binding(
            root, "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"),
        "dimension_sidecar": _optional_binding(
            root, "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"),
        "dimension_normalization": _optional_binding(
            root, "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"),
        "dev319_manifest": _optional_binding(
            root, "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"),
        "geometry_registry": _optional_binding(
            root, "challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"),
        "dope_base": _optional_binding(
            root, "weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"),
        "resnet18_base": _optional_binding(
            root, "data/pallet/results/pallet_resnet18_dimension_20261002_v1/DSNT_FULL_CONSTANT_epoch10.pt"),
    }
    return {
        "schema": "pallet_n3_completion_v3_run_manifest_v1",
        "generated_at": now(),
        "git": {
            "starting_head": STARTING_HEAD,
            "current_pre_commit_head": current_head,
            "final_head": {
                "value": None, "display": "x",
                "status": "PENDING_REGENERATE_AFTER_FINAL_COMMIT",
                "reason": ("A committed manifest cannot contain its own resulting commit SHA; "
                           "rerun the manifest after the final commit/push to record observed HEAD."),
            },
            "branch": branch,
            "working_tree_status": porcelain.splitlines(),
        },
        "environment": {
            "python": sys.version.split()[0], "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in
                         ("torch", "numpy", "opencv-python", "matplotlib")},
            "GPU_queried": False,
        },
        "bindings": bindings,
        "scope": {
            "included": ("DOPE and ResNet-18 N3 training/inference/evaluation; "
                         "frozen YOLO reuse/square audit; environment/symmetry/dimension "
                         "audits; runtime; offline lifter replay; report"),
            "exclusions": [
                {"item": "PDF generation", "status": "EXCLUDED_BY_CONTRACT"},
                {"item": "actual lifter control", "status": "EXCLUDED_BY_CONTRACT"},
                {"item": "new self-training", "status": "EXCLUDED_BY_CONTRACT"},
            ],
        },
        "training": {
            "expected_fits": 6, "verified_fits": len(complete),
            "expected_updates": 36_000,
            "verified_updates": sum(int(row.get("steps") or 0) for row in complete),
            "expected_sample_exposures": 576_000,
            "verified_sample_exposures": sum(
                int(row.get("total_exposures") or 0) for row in complete),
            "fits": training,
        },
        "orchestrator_receipts": receipt_rows,
        "raw_output_bindings": raw_rows,
        "overall_status": "PARTIAL",
        "overall_complete": False,
        "overall_reason": (
            "Execution and integrity are tracked separately; contract-level x remains for "
            "partial labels/cohorts and absent independent square/lifter references."),
        "regenerate": (
            f"{sys.executable} -m {PACKAGE}.run manifest --backbone all"),
    }


def execute_plan(plan: StagePlan, *, root: Path | None = None,
                 run_command: Callable[..., Any] = subprocess.run,
                 now: Callable[[], str] = _utc_now) -> dict:
    root = Path(root or repository_root()).resolve()
    valid, reason = receipt_valid(root, plan)
    if valid:
        return {"stage": plan.stage_id, "status": "SKIPPED", "reason": reason,
                "receipt": str(receipt_path(root, plan))}
    started = now()
    inputs = _bindings(root, plan.inputs)
    code = _bindings(root, plan.code)
    if plan.internal == "report_gate":
        missing = [path for path in plan.outputs if _resolve(root, path, output=True) is None]
        if missing:
            message = ("report CLI is not implemented; artifact gate is blocked by: "
                       + ", ".join(missing))
            _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                             returncodes=[], error=message)
            raise StageBlocked(message)
    returncodes: list[int] = []
    for command in plan.commands:
        try:
            result = run_command(list(command), cwd=str(root), check=False)
        except OSError as exc:
            message = f"command could not start: {' '.join(command)}: {exc}"
            _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                             returncodes=returncodes, error=message)
            raise StageError(message) from exc
        returncode = int(getattr(result, "returncode", 0))
        returncodes.append(returncode)
        if returncode not in plan.accepted_returncodes:
            message = f"command failed ({returncode}): {' '.join(command)}"
            _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                             returncodes=returncodes, error=message)
            raise StageError(message)
    if plan.name == "verify":
        verify_output = root / plan.outputs[0]
        try:
            verified = _read_json(verify_output)
        except (OSError, json.JSONDecodeError) as exc:
            message = f"verify did not write a readable result: {exc}"
            _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                             returncodes=returncodes, error=message)
            raise StageError(message) from exc
        if verified.get("status") == "BLOCKED_INTEGRITY":
            message = "verify reported BLOCKED_INTEGRITY"
            _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                             returncodes=returncodes, error=message)
            raise StageError(message)
    if plan.internal == "tests":
        _write_json(root / plan.outputs[0], {
            "schema": TEST_SCHEMA, "complete": True, "finished_at": now(),
            "command": list(plan.commands[0]), "returncode": 0,
            "scope": "all pytest tests under scripts/research/pallet_n3_completion_v3",
        })
    elif plan.internal == "manifest":
        _write_json(root / plan.outputs[0], build_run_manifest(root, now=now))
    try:
        outputs = _bindings(root, plan.outputs, output=True)
    except StageBlocked as exc:
        _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                         returncodes=returncodes, error=str(exc))
        raise StageError(f"{plan.stage_id} exited successfully but {exc}") from exc
    try:
        dynamic = _embedded_inputs(root, outputs)
    except StageError as exc:
        _failure_receipt(root, plan, started_at=started, inputs=inputs, code=code,
                         returncodes=returncodes, error=str(exc))
        raise
    known = {(item["path"], item["sha256"], item["bytes"]) for item in inputs}
    inputs.extend(item for item in dynamic
                  if (item["path"], item["sha256"], item["bytes"]) not in known)
    value = {
        "schema": SCHEMA, "complete": True, "status": "COMPLETE",
        "stage": plan.name, "backbone": plan.backbone,
        "started_at": started, "finished_at": now(), "gpu_stage": plan.gpu,
        "commands": [list(command) for command in plan.commands],
        "accepted_returncodes": list(plan.accepted_returncodes),
        "returncodes": returncodes, "config": dict(plan.config),
        "config_sha256": _object_hash(dict(plan.config)),
        "inputs": inputs, "inputs_sha256": _manifest_hash(inputs),
        "code": code, "code_sha256": _manifest_hash(code),
        "outputs": outputs, "outputs_sha256": _manifest_hash(outputs),
        "restart_policy": "skip only when all input/code/config/output hashes match",
    }
    _write_json(receipt_path(root, plan), value)
    return {"stage": plan.stage_id, "status": "COMPLETE",
            "receipt": str(receipt_path(root, plan)),
            "outputs_sha256": value["outputs_sha256"]}


def _training_status(root: Path, backbone: str, seed: int,
                     progress: Mapping[str, Any] | None,
                     active_processes: Sequence[Mapping[str, Any]] = (),
                     progress_mtime: float | None = None,
                     observed_epoch: float | None = None) -> dict:
    receipt_logical = DOC_REL / f"TRAIN_{backbone.upper()}_SEED{seed}.json"
    checkpoint_logical = RAW_REL / "runs" / backbone / f"seed{seed}" / "last.pt"
    receipt_file = _resolve(root, receipt_logical, output=True)
    checkpoint_file = _resolve(root, checkpoint_logical, output=True)
    state: dict[str, Any] = {"backbone": backbone, "seed": seed}
    if receipt_file:
        try:
            receipt = _read_json(receipt_file)
            checkpoint = receipt.get("checkpoint")
            complete = (receipt.get("complete") is True and receipt.get("steps") == 6000
                        and isinstance(checkpoint, Mapping)
                        and binding_valid(root, checkpoint, output=True))
            state.update({"status": "COMPLETE" if complete else "INVALID_RECEIPT",
                          "receipt": _logical(receipt_logical),
                          "checkpoint_sha256": checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None,
                          "steps": receipt.get("steps"),
                          "total_exposures": receipt.get("exposures")})
            return state
        except (OSError, json.JSONDecodeError):
            state.update({"status": "INVALID_RECEIPT", "receipt": _logical(receipt_logical)})
            return state
    if checkpoint_file:
        state.update({"status": "CHECKPOINT_PRESENT_UNRECEIPTED",
                      "checkpoint": _logical(checkpoint_logical),
                      "checkpoint_bytes": checkpoint_file.stat().st_size})
    else:
        state["status"] = "MISSING"
    if (isinstance(progress, Mapping) and progress.get("backbone") == backbone
            and progress.get("seed") == seed and progress.get("complete") is not True):
        pid = progress.get("pid")
        gpu = progress.get("gpu")
        if pid is None and isinstance(gpu, Mapping):
            process_text = gpu.get("processes")
            if isinstance(process_text, str) and process_text.strip():
                candidate = process_text.split(",", 1)[0].strip()
                pid = int(candidate) if candidate.isdigit() else candidate
        active_pids = {str(row.get("pid")) for row in active_processes}
        pid_visible = pid is not None and str(pid) in active_pids
        module_command_visible = any(
            ("pallet_n3_completion_v3.train" in str(row.get("command", ""))
             or "pallet_n3_completion_v3/train.py" in str(row.get("command", "")))
            and backbone in str(row.get("command", ""))
            for row in active_processes)
        observed_epoch = time.time() if observed_epoch is None else observed_epoch
        progress_age = (None if progress_mtime is None else
                        max(0., observed_epoch - progress_mtime))
        advancing_record = (isinstance(progress.get("step"), int)
                            and progress.get("step", 0) > 0)
        recorded_compute = (isinstance(gpu, Mapping)
                            and isinstance(gpu.get("processes"), str)
                            and bool(gpu["processes"].strip()))
        fresh_progress = progress_age is not None and progress_age <= 180.
        running = (pid_visible or module_command_visible
                   or (fresh_progress and advancing_record and recorded_compute))
        state.update({
            "status": "RUNNING" if running else "INTERRUPTED_RESUMABLE",
            "step": progress.get("step"), "pid": pid,
            "elapsed_seconds": progress.get("elapsed_seconds"),
            "observed_at": (gpu.get("time") if isinstance(gpu, Mapping) else None),
            "progress_mtime_age_seconds": progress_age,
            "process_visible_in_this_pid_namespace": pid_visible or module_command_visible,
            "recorded_compute_entry": recorded_compute,
            "running_evidence": ("visible module process" if pid_visible or module_command_visible
                                 else "fresh advancing TRAIN_PROGRESS plus recorded compute entry"
                                 if running else "progress older than 180s and no visible module process"),
            "resume_command": (f"{sys.executable} -m {PACKAGE}.train train {backbone}"),
        })
    return state


def status_snapshot(*, root: Path | None = None, backbone: str = "all",
                    python: str | None = None, device: str = "cuda",
                    yolo_python: str | None = None,
                    capture_root: str | None = None, source_root: str | None = None,
                    process_reader: Callable[[], str] | None = None) -> dict:
    """Return a file/process snapshot without importing torch or querying a GPU."""
    root = Path(root or repository_root()).resolve()
    progress_path = root / RAW_REL / "TRAIN_PROGRESS.json"
    try:
        progress = _read_json(progress_path) if progress_path.is_file() else None
    except (OSError, json.JSONDecodeError):
        progress = {"status": "UNREADABLE"}
    selected = BACKBONES if backbone == "all" else (backbone,)
    options = dict(python=python, yolo_python=yolo_python, device=device,
                   capture_root=capture_root,
                   source_root=source_root, root=root)
    stages = []
    for plan in expand_plans("all", backbone, **options):
        valid, reason = receipt_valid(root, plan)
        missing = [path for path in plan.outputs if _resolve(root, path, output=True) is None]
        failure = None
        receipt = receipt_path(root, plan)
        if receipt.is_file() and not valid:
            try:
                failure = _read_json(receipt).get("error")
            except (OSError, json.JSONDecodeError):
                failure = "unreadable receipt"
        stages.append({"stage": plan.stage_id,
                       "status": "COMPLETE" if valid else "MISSING_OR_STALE",
                       "receipt_reason": reason, "missing_outputs": missing,
                       "last_error": failure, "gpu_stage": plan.gpu})
    if process_reader is None:
        def process_reader() -> str:
            try:
                return subprocess.check_output(
                    ["ps", "-eo", "pid=,etimes=,args="], text=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                return ""
    processes = []
    for line in process_reader().splitlines():
        if ("pallet_n3_completion_v3.train" in line
                or "pallet_n3_completion_v3/train.py" in line):
            pieces = line.strip().split(maxsplit=2)
            if len(pieces) == 3:
                processes.append({"pid": pieces[0], "elapsed_seconds": pieces[1],
                                  "command": pieces[2]})
    progress_mtime = progress_path.stat().st_mtime if progress_path.is_file() else None
    observed_epoch = time.time()
    training = [_training_status(
                    root, item, seed, progress, processes,
                    progress_mtime=progress_mtime, observed_epoch=observed_epoch)
                for item in selected for seed in (1, 2, 3)]
    return {
        "schema": "pallet_n3_completion_v3_orchestrator_status_v1",
        "observed_at": _utc_now(), "root": str(root), "GPU_queried": False,
        "selected_backbone": backbone, "training": training,
        "progress": progress, "matching_processes": processes, "stages": stages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash-bound restartable orchestrator for pallet N3 v3")
    parser.add_argument("stage", choices=STAGES,
                        help="status is read-only; all executes the frozen order")
    parser.add_argument("--backbone", choices=(*BACKBONES, "all"), default="all")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for stage subprocesses")
    parser.add_argument(
        "--yolo-python", default=DEFAULT_YOLO_PYTHON,
        help="Ultralytics 8.4.x Python used only for YOLO26 square/lifter stages")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capture-root",
                        help="recorded lifter capture root passed through verbatim")
    parser.add_argument("--source-root",
                        help="historical reuse source checkout passed through verbatim")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = repository_root()
    options = dict(python=arguments.python, yolo_python=arguments.yolo_python,
                   device=arguments.device,
                   capture_root=arguments.capture_root, source_root=arguments.source_root)
    if arguments.stage == "status":
        print(json.dumps(status_snapshot(root=root, backbone=arguments.backbone,
                                         **options), ensure_ascii=False, indent=2,
                         allow_nan=False), flush=True)
        return 0
    results = []
    try:
        names = ALL_EXECUTION_ORDER if arguments.stage == "all" else (arguments.stage,)
        for name in names:
            # Build each plan immediately before execution so a late manifest
            # observes receipts and outputs created by earlier stages.
            for plan in expand_plans(name, arguments.backbone, root=root, **options):
                result = execute_plan(plan, root=root)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    except StageBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc),
                          "completed": results}, ensure_ascii=False, indent=2),
              file=sys.stderr, flush=True)
        return 2
    except StageError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc),
                          "completed": results}, ensure_ascii=False, indent=2),
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
