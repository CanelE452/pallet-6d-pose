"""Shared paths, immutable bindings, and execution helpers for N3 v3."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NAME = HERE.name
DOC = ROOT / "_docs" / "experiments" / NAME
RAW = ROOT / "data" / "pallet" / "results" / NAME

SOURCE = ROOT / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
SIDECAR = ROOT / "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"
NORMALIZATION = ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"
DEV = ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"
DOPE_WEIGHTS = ROOT / "weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"
RESNET_CONSTANT = ROOT / "data/pallet/results/pallet_resnet18_dimension_20261002_v1/DSNT_FULL_CONSTANT_epoch10.pt"

SEEDS = (1, 2, 3)
STEPS = 6000
BATCH = 16
TEMPERATURES = (.5, 1., 2., 4.)
CAP_FRACTION = .01
LAMBDA = 1.
CONFIGS = {
    "dope": dict(c3=256, c4=128, stride3=4, stride4=8, hidden=16,
                 encoded=24, stencil_fraction=0.1310373991727829),
    "resnet18": dict(c3=128, c4=256, stride3=8, stride4=16, hidden=16,
                     encoded=24, stencil_fraction=0.1310373991727829),
}
OPTIMIZER = dict(lr=1e-3, weight_decay=1e-4, betas=(.9, .999),
                 warmup_steps=100, cosine_final_lr_fraction=.1,
                 gradient_clip_norm=5.)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path | str):
    return json.loads(Path(path).read_text())


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path | str) -> dict:
    path = Path(path).absolute()
    relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": str(relative), "sha256": sha256(path), "bytes": path.stat().st_size}


def verify(entry: dict) -> Path:
    path = Path(entry["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Bound input changed or is missing: {path}")
    if "bytes" in entry and path.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"Bound input size changed: {path}")
    return path


def write(path: Path | str, value, *, freeze: bool = False) -> None:
    path = Path(path)
    resolved_parent = path.parent.resolve()
    if not (resolved_parent.is_relative_to(DOC.resolve()) or
            resolved_parent.is_relative_to(RAW.resolve())):
        raise ValueError(f"N3 writer refuses path outside DOC/RAW: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if freeze and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Frozen output differs: {path}")
        return
    pending = path.with_name(path.name + ".pending")
    pending.write_text(text)
    pending.replace(path)


def local_module(key: str, path: Path | str):
    name = "pallet_n3_v3_" + key
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def learning_rate(step: int) -> float:
    if not 1 <= step <= STEPS:
        raise ValueError(step)
    warmup = OPTIMIZER["warmup_steps"]
    if step <= warmup:
        return OPTIMIZER["lr"] * step / warmup
    progress = (step - warmup) / (STEPS - warmup)
    final = OPTIMIZER["cosine_final_lr_fraction"]
    return OPTIMIZER["lr"] * (final + (1 - final) * .5 *
                              (1 + math.cos(math.pi * progress)))


def gpu_snapshot(*, refuse_other_compute: bool = True) -> dict:
    query = ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"]
    process_query = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader,nounits"]
    status = subprocess.check_output(query, text=True).strip()
    processes = subprocess.check_output(process_query, text=True).strip()
    unrelated = []
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] == str(os.getpid()):
            continue
        if len(fields) > 1 and fields[1] == "/usr/share/rustdesk/rustdesk":
            continue
        unrelated.append(line)
    if refuse_other_compute and unrelated:
        raise RuntimeError("Other GPU compute is active; nothing was killed: " + repr(unrelated))
    fields = [field.strip() for field in status.split(",")]
    if len(fields) >= 4 and float(fields[3]) >= 80:
        raise RuntimeError("GPU temperature guard reached 80 C")
    return {"time": now(), "status": status, "processes": processes,
            "other_compute": unrelated}

