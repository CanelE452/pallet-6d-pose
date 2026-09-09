"""Run the frozen spatial-concat stages after the persistent base job finishes.

This process is intended to run as a separate systemd user service.  It only
starts feature extraction once the base trainer has exited successfully and
all 60 epoch rows plus the final checkpoint are present.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_SERVICE = "pallet-pose-spatial-base.service"
RUN_DIR = HERE / "runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42"
RESULTS = RUN_DIR / "results.csv"
LAST = RUN_DIR / "weights/last.pt"
EXPECTED_EPOCHS = 60
POLL_SECONDS = 30
WATCH_STARTED_UNIX = time.time()
SYSTEMD_FAILURE_MESSAGE_ID = "d9b373ed55a64feb8242e02dbe79a49c"


def stamp(message: str) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] {message}", flush=True)


def service_properties() -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            BASE_SERVICE,
            "--property=LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    properties = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    properties.setdefault("ActiveState", f"unknown(rc={result.returncode})")
    return properties


def systemd_failure_after_watch_start() -> str | None:
    result = subprocess.run(
        [
            "journalctl",
            "--user",
            f"USER_UNIT={BASE_SERVICE}",
            f"MESSAGE_ID={SYSTEMD_FAILURE_MESSAGE_ID}",
            f"--since=@{WATCH_STARTED_UNIX:.6f}",
            "--output=cat",
            "--no-pager",
            "--quiet",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not inspect base service journal: {result.stderr.strip()}")
    message = result.stdout.strip()
    return message or None


def completed_epochs() -> list[int]:
    if not RESULTS.is_file():
        return []
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        return [int(float(row["epoch"])) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid epoch column in {RESULTS}") from exc


def wait_for_base() -> None:
    last_reported: tuple[str, int] | None = None
    while True:
        properties = service_properties()
        state = properties["ActiveState"]
        epochs = completed_epochs()
        status = (state, len(epochs))
        if status != last_reported:
            stamp(f"base service={state}; completed epochs={len(epochs)}/{EXPECTED_EPOCHS}")
            last_reported = status

        if state in {"active", "activating", "reloading", "deactivating"}:
            time.sleep(POLL_SECONDS)
            continue

        failure = systemd_failure_after_watch_start()
        if failure is not None:
            raise RuntimeError(f"base service failed according to systemd: {failure}")
        if properties.get("LoadState") != "not-found" and (
            properties.get("Result") != "success"
            or properties.get("ExecMainStatus") != "0"
        ):
            raise RuntimeError(f"base service did not exit successfully: {properties}")
        if epochs != list(range(1, EXPECTED_EPOCHS + 1)):
            raise RuntimeError(
                f"base service stopped in state={state} with epoch sequence {epochs}"
            )
        if not LAST.is_file() or LAST.stat().st_size == 0:
            raise RuntimeError(f"base service stopped without a valid checkpoint: {LAST}")
        stamp(f"base complete; final checkpoint bytes={LAST.stat().st_size}")
        return


def run_stage(label: str, argv: list[str]) -> None:
    stamp(f"starting {label}: {' '.join(argv)}")
    subprocess.run(argv, cwd=HERE, check=True)
    stamp(f"completed {label}")


def main() -> None:
    wait_for_base()
    python = sys.executable
    run_stage(
        "spatial feature extraction and S0/S1 probes",
        [python, str(HERE / "run_spatial_concat.py"), "all", "--device", "cuda:0"],
    )
    run_stage(
        "real DEV base comparison",
        [python, str(HERE / "run_real_dev_base_comparison.py")],
    )
    stamp("all post-base stages completed")


if __name__ == "__main__":
    main()
