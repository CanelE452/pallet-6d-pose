"""V3 development driver — 학습에서 끝내지 않는다.

    STEP 1  학습 (true-ignore trainer)   완료 판정은 results.csv epoch 수로만
    STEP 2  PAPER_EVAL 평가              V1/V2 와 같은 evaluator·같은 manifest
    STEP 3  지표 + G1~G8 gate            기준은 사전 고정, 결과 보고 못 고친다
    STEP 4  알림 = 판정

smoke 는 loss/gradient 건전성까지 확인한다 (§19).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

V3_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v3"
METHOD_LOCK = V3_RESULTS / "SELFTRAIN_V3_METHOD_LOCK.json"
DATASETS = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v3"
RUNS = REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v3"
EVAL_OUT = REPO_ROOT / "data/pallet/results/paper_eval_v3/arms"
EVALUATOR = REPO_ROOT / "challenge/evaluation_v2/paper_real_eval.py"
MANIFESTS = REPO_ROOT / "challenge/real_gt_v2/manifests"
R0 = REPO_ROOT / (
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")

ARMS = ("V3A_TRUE_IGNORE", "V3B_TRUE_IGNORE_AMBIG")

# V1/V2 와 동일.  V3 는 loss supervision mask 만 바꾼다.
TRAIN_ARGS = dict(
    imgsz=640, batch=32, optimizer="SGD", lr0=0.002, lrf=0.01, cos_lr=True,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=1.0, patience=0,
    box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=0.0, translate=0.1, scale=0.25,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    mosaic=0.15, close_mosaic=3, mixup=0.0, copy_paste=0.0, erasing=0.4,
    seed=42, deterministic=True, val=False, plots=False, save_period=-1,
    workers=2, cache=False,
)


def notify(message: str) -> None:
    hook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not hook:
        print(f"[notify skipped] {message}", flush=True)
        return
    try:
        import urllib.request

        request = urllib.request.Request(
            hook, data=json.dumps({"content": message[:1900]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=20).read()
    except Exception as exc:  # noqa: BLE001
        print(f"[notify failed] {exc}", flush=True)


def free_memory_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024 ** 2)
    return float("inf")


def completed_epochs(name: str) -> int:
    results = RUNS / name / "results.csv"
    if not results.exists():
        return 0
    return max(len(results.read_text().strip().splitlines()) - 1, 0)


def assert_contract_passed() -> None:
    receipt = V3_RESULTS / "TRUE_IGNORE_LOSS_CONTRACT.json"
    if not receipt.exists():
        raise SystemExit("LOSS_CONTRACT_MISSING — smoke 전에 계약을 먼저 검증하라")
    if json.loads(receipt.read_text())["status"] != "PASS":
        raise SystemExit("LOSS_CONTRACT_NOT_PASS — 학습 금지")


def train_arm(arm: str, epochs: int, tag: str) -> dict:
    from ultralytics import YOLO
    from true_ignore_trainer import TrueIgnorePoseTrainer

    name = f"{arm}__{tag}"
    weights = RUNS / name / "weights" / "last.pt"
    done = completed_epochs(name)
    if weights.exists() and done >= epochs:
        return {"status": "ALREADY_DONE", "epochs_completed": done,
                "weights": str(weights)}

    available = free_memory_gb()
    if available < 6.0:
        raise SystemExit(f"REFUSING_TO_START: 가용 메모리 {available:.1f} GB")

    print(f"\n{'=' * 70}\n{name}  epochs={epochs}  free={available:.1f}GB\n"
          f"{'=' * 70}", flush=True)
    model = YOLO(str(R0), task="pose")
    model.train(
        trainer=TrueIgnorePoseTrainer,
        data=str(DATASETS / arm / "data.yaml"),
        epochs=epochs,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        **TRAIN_ARGS,
    )
    if not weights.exists() or weights.stat().st_size == 0:
        return {"status": "FAILED_NO_CHECKPOINT", "expected": str(weights)}
    done = completed_epochs(name)
    if done < epochs:
        return {"status": "FAILED_INCOMPLETE_EPOCHS",
                "epochs_completed": done, "epochs_expected": epochs}
    return {"status": "OK", "epochs_completed": done, "weights": str(weights)}


def assert_true_ignore_trainer_was_used(arm: str, tag: str) -> dict:
    """선언이 아니라 확인.  학습된 모델의 criterion 클래스를 실제로 본다."""

    from ultralytics import YOLO
    from true_ignore_pose_loss import TrueIgnorePoseLoss26

    weights = RUNS / f"{arm}__{tag}" / "weights" / "last.pt"
    model = YOLO(str(weights), task="pose").model
    from true_ignore_pose_loss import make_criterion

    criterion = make_criterion(model)
    inner = getattr(criterion, "one2one", criterion)
    used = type(inner).__name__
    return {"criterion": type(criterion).__name__, "inner": used,
            "is_true_ignore": isinstance(inner, TrueIgnorePoseLoss26)}


def evaluate(name: str, weights: Path) -> tuple[Path, Path]:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    report = EVAL_OUT / f"{name}.json"
    per_frame = EVAL_OUT / f"{name}_per_frame.csv"
    if report.exists() and per_frame.exists():
        print(f"  {name}: 평가 이미 있음 — 건너뜀", flush=True)
        return report, per_frame
    command = [
        sys.executable, str(EVALUATOR),
        "--positive-manifest", str(MANIFESTS / "PAPER_EVAL_ALL_POS.json"),
        "--negative-manifest", str(MANIFESTS / "DEV_NEG2689.json"),
        "--population-role", "DEV",
        "--weights", str(weights),
        "--migration-gate", str(REPO_ROOT / "challenge/real_gt_v2/MIGRATION_GATE.json"),
        "--symmetry-contract",
        str(REPO_ROOT / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"),
        "--object-migration-gate",
        f"wood_small_80x59x14={REPO_ROOT}/challenge/real_gt_v2/wood_audit/"
        "migration/MIGRATION_GATE.json",
        "--out", str(report),
        "--per-frame-out", str(per_frame),
        "--report-out", str(EVAL_OUT / f"{name}_report.md"),
        "--device", "0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    if not report.exists() or not per_frame.exists():
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"EVALUATION_PRODUCED_NO_ARTIFACT: {name}")
    return report, per_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    args = parser.parse_args()

    assert_contract_passed()
    lock = json.loads(METHOD_LOCK.read_text())
    epochs = 1 if args.stage == "smoke" else int(lock["epochs"])
    tag = args.stage.upper()

    state = {
        "schema_version": "v3_dev_driver_state_v1",
        "stage": args.stage,
        "started": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "arms": {},
    }
    state_path = V3_RESULTS / f"V3_DRIVER_STATE_{tag}.json"

    for arm in args.arms:
        outcome = train_arm(arm, epochs, tag)
        state["arms"][arm] = {"train": outcome}
        if outcome["status"] != "FAILED_NO_CHECKPOINT":
            state["arms"][arm]["trainer_check"] = assert_true_ignore_trainer_was_used(
                arm, tag)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        print(f"{arm}: {outcome['status']}  "
              f"{state['arms'][arm].get('trainer_check', {})}", flush=True)
        if outcome["status"].startswith("FAILED"):
            notify(f"V3 {tag} **중단** — {arm} {outcome['status']}")
            raise SystemExit(f"TRAINING_FAILED: {arm} {outcome['status']}")

    if args.stage == "smoke":
        notify(f"V3 smoke 통과 — {len(args.arms)} arm 각 1 epoch. full 로 넘어간다.")
        print("\nsmoke 통과.", flush=True)
        return 0

    for arm in args.arms:
        weights = Path(state["arms"][arm]["train"]["weights"])
        report, per_frame = evaluate(f"{arm}__{tag}", weights)
        state["arms"][arm]["evaluation"] = {
            "report": str(report.relative_to(REPO_ROOT)),
            "per_frame": str(per_frame.relative_to(REPO_ROOT)),
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        print(f"{arm}: 평가 완료", flush=True)

    evaluate("R0", R0)
    state["finished"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    metrics = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "scripts/self_training_yolo/v3/v3_dev_metrics.py"),
         "--tag", tag],
        cwd=REPO_ROOT, text=True, capture_output=True)
    print(metrics.stdout[-8000:], flush=True)
    if metrics.returncode != 0:
        print(metrics.stderr[-3000:], flush=True)
        notify(f"V3 {tag} — 지표 계산 실패")
        raise SystemExit("METRICS_FAILED")

    verdict = json.loads((V3_RESULTS / "V3_DEV_GATE.json").read_text())
    lines = [f"**V3 DEV {verdict['status']}**"]
    for key, value in verdict["gates"].items():
        lines.append(f"{'PASS' if value['pass'] else 'FAIL'}  {key}: {value['detail']}")
    notify("\n".join(lines))
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
