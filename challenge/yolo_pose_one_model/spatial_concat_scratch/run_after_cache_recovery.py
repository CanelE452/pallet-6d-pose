"""Run probe training and real DEV evaluation after the locked cache recovery."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REQUIRED = (
    HERE / "SPATIAL_FEATURE_CACHE_60K.npz",
    HERE / "SPATIAL_FEATURE_CACHE_60K_AUDIT.json",
    HERE / "PROBE_TRAINING_RECOVERY_CONTRACT.json",
)


def stamp(message: str) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] {message}", flush=True)


def run_stage(label: str, argv: list[str]) -> None:
    stamp(f"starting {label}: {' '.join(argv)}")
    subprocess.run(argv, cwd=HERE, check=True)
    stamp(f"completed {label}")


def main() -> None:
    missing = [str(path) for path in REQUIRED if not path.is_file()]
    if missing:
        raise RuntimeError(f"cache recovery inputs missing: {missing}")
    python = sys.executable
    run_stage(
        "S0/S1 probe training with DEV abstention counted as error",
        [
            python,
            str(HERE / "run_spatial_probe_from_cache.py"),
            "train",
            "--device",
            "cuda:0",
        ],
    )
    run_stage(
        "real DEV base comparison",
        [python, str(HERE / "run_real_dev_base_comparison.py")],
    )
    stamp("all cache-recovery stages completed")


if __name__ == "__main__":
    main()
