"""Record the two interpreter environments used by the N3 execution.

DOPE/ResNet run in the historical ``pallet-pose`` environment.  YOLO26
checkpoints require the newer Ultralytics implementation in
``pallet-yolo26`` because their pickle contains C3k2/Pose26 classes.  This
small CPU-only audit makes that boundary machine-readable without treating
external conda files as portable repository bindings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence

from . import common as C


SCHEMA = "pallet_n3_completion_v3_environment_audit_v1"
OUTPUT = C.DOC / "ENVIRONMENT_AUDIT.json"
DEFAULT_YOLO_PYTHON = Path(
    "/home/minjae/anaconda3/envs/pallet-yolo26/bin/python")

_PROBE = r"""
import hashlib, json, platform, sys
from pathlib import Path
import cv2, numpy, torch, ultralytics
try:
    from ultralytics.nn.modules.block import C3k2
    has_c3k2 = C3k2 is not None
except (ImportError, AttributeError):
    has_c3k2 = False
module = Path(ultralytics.__file__).resolve()
digest = hashlib.sha256(module.read_bytes()).hexdigest()
print(json.dumps({
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "ultralytics": ultralytics.__version__,
    "ultralytics_module": str(module),
    "ultralytics_module_sha256": digest,
    "ultralytics_module_bytes": module.stat().st_size,
    "has_C3k2": has_c3k2,
}, sort_keys=True))
"""


def probe(python: str | Path, *,
          runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    executable = Path(python).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    completed = runner(
        [str(executable), "-c", _PROBE], text=True, capture_output=True,
        check=True, timeout=60)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty environment probe: {executable}")
    payload = json.loads(lines[-1])
    payload.update(
        executable=str(executable),
        executable_sha256=C.sha256(executable),
        executable_bytes=executable.stat().st_size,
    )
    return payload


def build(primary_python: str | Path, yolo_python: str | Path,
          *, probe_fn: Callable[[str | Path], dict] = probe) -> dict:
    primary = probe_fn(primary_python)
    yolo = probe_fn(yolo_python)
    if primary.get("has_C3k2") is not False:
        raise RuntimeError("primary pallet-pose environment unexpectedly exposes C3k2")
    if yolo.get("has_C3k2") is not True:
        raise RuntimeError("YOLO26 environment does not expose required C3k2")
    return {
        "schema": SCHEMA,
        "complete": True,
        "roles": {
            "dope_resnet_training_inference_runtime": primary,
            "yolo26_square_and_offline_lifter": yolo,
        },
        "boundary": {
            "reason": "YOLO26 checkpoint requires C3k2/Pose26 support",
            "single_process_environment_mixing": False,
            "same_GPU_for_new_measurements": "NVIDIA GeForce RTX 3080",
            "yolo_runtime_compared_in_new_fixed26_benchmark": False,
        },
    }


def generate(primary_python: str | Path = sys.executable,
             yolo_python: str | Path = DEFAULT_YOLO_PYTHON) -> dict:
    if OUTPUT.is_file():
        payload = C.read(OUTPUT)
        if payload.get("schema") != SCHEMA or payload.get("complete") is not True:
            raise RuntimeError("existing environment audit is invalid")
        return payload
    payload = build(primary_python, yolo_python)
    C.write(OUTPUT, payload, freeze=True)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-python", default=sys.executable)
    parser.add_argument("--yolo-python", default=str(DEFAULT_YOLO_PYTHON))
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.primary_python, args.yolo_python),
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
