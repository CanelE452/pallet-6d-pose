"""Shared constants and immutable-artifact helpers for the isolated experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NAME = HERE.name
DOC = ROOT / "_docs/experiments" / NAME
RAW = ROOT / "data/pallet/results" / NAME

DSNT_HERE = ROOT / "scripts/research/pallet_resnet18_dimension_20261002_v1"
DSNT_DOC = ROOT / "_docs/experiments/pallet_resnet18_dimension_20261002_v1"
DSNT_RAW = ROOT / "data/pallet/results/pallet_resnet18_dimension_20261002_v1"
FULL_COMPLETE = DSNT_DOC / "DSNT_FULL_FULL_COMPLETE.json"
FULL_PROTOCOL = DSNT_DOC / "DSNT_FULL_TRAIN_PROTOCOL.json"
PROTOCOL = DOC / "PROTOCOL.json"
DEV_POS = ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"
DEV_NEG = ROOT / "challenge/real_gt_v2/manifests/DEV_NEG2689.json"
POSE_RAW = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
POSE_CODE = ROOT / "scripts/paper/pose_metric_closure_v1"
GEOMETRY_REGISTRY = ROOT / "challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"

SOURCE_MANIFEST = ROOT / "data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json"
DIMENSION_SIDECAR = ROOT / "data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz"
DIMENSION_NORMALIZATION = ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"

SEEDS = (1, 2, 3)
HEAD_ARMS = ("D0", "P0", "P5", "P5_CONSTANT")
DIMENSION_ARMS = ("P5", "P5_CONSTANT")
STEPS = 6000
BATCH = 16
HEAD_CONFIG = dict(
    c3=128,
    c4=256,
    stride3=8,
    stride4=16,
    hidden=16,
    encoded=24,
    stencil_fraction=0.1310373991727829,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def binding(path: Path | str) -> dict:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    try:
        label = str(value.relative_to(ROOT.resolve()))
    except ValueError:
        label = str(value)
    return dict(path=label, sha256=sha256(value), bytes=value.stat().st_size)


def verify_binding(value: dict) -> Path:
    if set(value) < {"path", "sha256"}:
        raise ValueError("Binding requires path and sha256")
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"Artifact binding mismatch: {path}")
    if "bytes" in value and path.stat().st_size != value["bytes"]:
        raise ValueError(f"Artifact byte-size mismatch: {path}")
    return path.resolve()


def verify_protocol(path: Path | str = PROTOCOL) -> dict:
    """Load the sealed local protocol and verify every immutable input."""
    protocol_path = Path(path)
    if not protocol_path.is_file():
        raise RuntimeError("Refiner protocol is not sealed")
    payload = read_json(protocol_path)
    if payload.get("schema") != "resnet18_full_dim_local_refiner_protocol_v1":
        raise ValueError("Unexpected refiner protocol schema")
    for value in payload.get("bindings", []):
        verify_binding(value)
    verify_binding(payload["baseline"])
    verify_binding(payload["baseline_protocol"])
    verify_binding(payload["baseline_training_complete"])
    return payload


def arm_key(arm: str, seed: int) -> str:
    if arm not in HEAD_ARMS or seed not in SEEDS:
        raise ValueError("Unknown head arm/seed")
    return f"{arm}_S{seed}"


def write_frozen_json(path: Path | str, payload: dict) -> None:
    """Write once below this experiment's DOC/RAW roots; later bytes must match."""
    destination = Path(path).resolve()
    allowed = (DOC.resolve(), RAW.resolve())
    if not any(destination.is_relative_to(root) for root in allowed):
        raise ValueError(f"Refusing write outside experiment roots: {destination}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text() != text:
            raise ValueError(f"Frozen artifact already exists with different bytes: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending")
    temporary.write_text(text)
    temporary.replace(destination)
