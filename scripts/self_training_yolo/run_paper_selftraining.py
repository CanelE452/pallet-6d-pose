"""MAIN self-training arm 을 smoke → full → evaluate 까지 한 스크립트에서 돌린다.

전역 규칙: 드라이버를 "학습 완료" 에서 끝내지 않는다.  후속 단계(평가·판정)까지
이어지게 하고, 알림은 '완료' 가 아니라 **판정과 핵심 수치** 를 담는다.

완료 판정 재발방지 (§19):
  * `pgrep -f "<driver 이름>"` 같은 자기매칭 대기 루프를 쓰지 않는다.  과거에 그걸로
    GPU 를 12 시간 넘게 놀렸다.  여기서는 학습을 **동기 호출**하고, 끝난 뒤
    **산출물(checkpoint 존재 + 크기 > 0)** 로만 완료를 판정한다.
  * exit code 나 프로세스 존재를 완료 근거로 쓰지 않는다.

모든 arm 은 exposure lock 이 정한 같은 예산을 쓴다.  다른 것은 dataset 의
pseudo-label selection rule 하나뿐이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
LOCK = RESULTS / "SELFTRAIN_EXPOSURE_LOCK.json"
DATASETS = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v1"
RUNS = REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v1"
STATE = RESULTS / "DRIVER_STATE.json"
NOTIFY = Path.home() / ".claude" / "hooks" / "discord-notify.sh"

ARMS = ("R0_CONT", "R1_NAIVE", "R2_CONF", "R3_CONF_REPROJ", "R4_CONF_REMOVE", "R5_PROPOSED")

# R0 의 augmentation 을 그대로 쓰되, checkpoint 에서 이어가는 adaptation 이므로
# LR·warmup·mosaic 은 repo 의 adaptation 선례(ADAPT_N0/N1)를 따른다.
# fliplr 은 0.0 을 유지한다 — 켜면 flip consistency 가 학습이 강제한 항등식이 된다.
TRAIN_ARGS = dict(
    imgsz=640, batch=32, optimizer="SGD", lr0=0.002, lrf=0.01, cos_lr=True,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=1.0, patience=0,
    box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=0.0, translate=0.1, scale=0.25,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    mosaic=0.15, close_mosaic=3, mixup=0.0, copy_paste=0.0, erasing=0.4,
    seed=42, deterministic=True, val=False, plots=False, save_period=-1,
    # R0 와 같은 값.  지정하지 않으면 Ultralytics 기본 8 이 되는데, worker 마다
    # ~900MB 를 잡아 OOM killer 가 프로세스를 죽였다(EXIT 137, R1_NAIVE 2회).
    # I/O 설정이라 optimization 계약과 무관하고 모든 arm 에서 동일하다.
    workers=2, cache=False,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notify(message: str) -> None:
    """알림은 '완료' 가 아니라 판정과 수치를 담는다.  실패해도 학습을 막지 않는다."""

    if not NOTIFY.exists():
        return
    try:
        subprocess.run(["bash", str(NOTIFY), message], timeout=30, check=False)
    except Exception as error:  # noqa: BLE001
        print(f"discord 실패(무시): {error}", flush=True)


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"nvidia-smi unavailable: {error}"


def available_memory_mb() -> int:
    """MemAvailable.  OOM killer 가 남의 프로세스(VS Code 등)를 죽이는 걸 막는다."""

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:  # noqa: BLE001
        pass
    return -1


def preflight(minimum_mb: int = 6000) -> None:
    """메모리가 부족하면 시작하지 않는다.

    한 번 밟았다.  workers 기본 8 로 dataloader 를 띄우자 worker 마다 ~900MB 를
    잡았고, 죽은 실행의 worker 가 고아로 남아 누적되면서 가용 메모리가 1.7GB 까지
    떨어졌다.  OOM killer 는 우리 프로세스만 죽이지 않는다 — 사용자의 편집기까지
    같이 죽었다.  그래서 시작 전에 먼저 본다.
    """

    available = available_memory_mb()
    if available < 0:
        print("preflight: MemAvailable 을 못 읽었다 — 계속한다", flush=True)
        return
    print(f"preflight: MemAvailable {available} MB", flush=True)
    if available < minimum_mb:
        raise SystemExit(
            f"INSUFFICIENT_MEMORY: {available} MB < {minimum_mb} MB. "
            "고아 dataloader worker 가 남아 있는지 확인하라 "
            "(ps -eo pid,rss,cmd | grep selftraining)."
        )


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"stages": {}}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def train_arm(arm: str, epochs: int, tag: str, init: Path) -> dict:
    """한 arm 을 동기로 학습한다.  완료는 checkpoint 산출물로만 판정한다."""

    from ultralytics import YOLO

    name = f"{arm}__{tag}"
    run_dir = RUNS / name
    weights = run_dir / "weights" / "last.pt"
    results = run_dir / "results.csv"

    # last.pt 는 **매 epoch 갱신**된다.  존재만 보면 중단된 run 을 완료로 오인한다.
    # 완료의 증거는 results.csv 에 계획한 epoch 수만큼 행이 있는 것이다.
    if weights.exists() and weights.stat().st_size > 0 and results.exists():
        completed = max(len(results.read_text().strip().splitlines()) - 1, 0)
        if completed >= epochs:
            return {"status": "ALREADY_DONE", "epochs_completed": completed,
                    "weights": str(weights.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(weights)}
        print(f"{name}: {completed}/{epochs} epoch 에서 중단된 흔적 — 다시 돌린다",
              flush=True)

    started = time.time()
    print(f"\n{'=' * 70}\n{name}  epochs={epochs}  pid={os.getpid()}\n"
          f"GPU: {gpu_snapshot()}\n{'=' * 70}", flush=True)

    model = YOLO(str(init), task="pose")
    model.train(
        data=str(DATASETS / arm / "data.yaml"),
        epochs=epochs,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        **TRAIN_ARGS,
    )

    # 선언이 아니라 산출물로 판정한다.  epoch 수까지 본다.
    if not weights.exists() or weights.stat().st_size == 0:
        return {"status": "FAILED_NO_CHECKPOINT", "expected": str(weights)}
    completed = (
        max(len(results.read_text().strip().splitlines()) - 1, 0)
        if results.exists() else 0
    )
    if completed < epochs:
        return {"status": "FAILED_INCOMPLETE_EPOCHS",
                "epochs_completed": completed, "epochs_expected": epochs}
    return {
        "status": "OK",
        "epochs_completed": completed,
        "weights": str(weights.relative_to(REPO_ROOT)),
        "sha256": sha256_file(weights),
        "size_bytes": weights.stat().st_size,
        "elapsed_s": round(time.time() - started, 1),
        "gpu_after": gpu_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full", "all"), default="all")
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    preflight()

    lock = json.loads(LOCK.read_text())
    init = REPO_ROOT / lock["initialisation"]["checkpoint"]
    if not init.exists():
        raise SystemExit(f"INIT_CHECKPOINT_MISSING: {init}")
    if sha256_file(init) != lock["initialisation"]["sha256"]:
        raise SystemExit("INIT_CHECKPOINT_SHA_MISMATCH — exposure lock 과 다르다")
    full_epochs = int(lock["epochs"])

    state = load_state()
    state.setdefault("init_checkpoint", lock["initialisation"]["checkpoint"])
    state.setdefault("init_sha256", lock["initialisation"]["sha256"])
    state["total_optimizer_updates"] = lock["total_optimizer_updates"]
    state["train_args"] = TRAIN_ARGS

    stages = ("smoke", "full") if args.stage == "all" else (args.stage,)
    for stage in stages:
        epochs = 1 if stage == "smoke" else full_epochs
        for arm in args.arms:
            key = f"{stage}:{arm}"
            if state["stages"].get(key, {}).get("status") in ("OK", "ALREADY_DONE"):
                print(f"skip {key} (done)", flush=True)
                continue
            result = train_arm(arm, epochs, stage.upper(), init)
            state["stages"][key] = result
            save_state(state)
            print(f"{key}: {result['status']}", flush=True)
            if result["status"].startswith("FAILED"):
                print(f"\nSTOP — {key} 가 checkpoint 를 남기지 않았다.", flush=True)
                return 2
        if stage == "smoke":
            print("\nSMOKE 전 arm 통과 — FULL 로 넘어간다.", flush=True)

    print("\n" + "=" * 70)
    print("TRAINING DONE — 판정용 수치")
    for key, value in state["stages"].items():
        if key.startswith("full:"):
            print(f"  {key:28} {value['status']:12} "
                  f"{value.get('elapsed_s', '-')}s  sha {value.get('sha256', '')[:12]}")
    print("=" * 70)
    print("다음: scripts/self_training_yolo/evaluate_arms.py 로 PAPER_EVAL 평가")

    if args.notify:
        done = [k for k, v in state["stages"].items()
                if k.startswith("full:") and v["status"] in ("OK", "ALREADY_DONE")]
        notify(
            f"self-training FULL {len(done)}/{len(ARMS)} arm 완료 "
            f"({lock['total_optimizer_updates']} updates, lr {lock['learning_rate']}) — "
            "다음 단계는 PAPER_EVAL 평가"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
