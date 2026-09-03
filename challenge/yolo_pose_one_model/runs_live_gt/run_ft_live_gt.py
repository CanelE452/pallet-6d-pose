#!/usr/bin/env python3
"""수동 어노 402장으로 배포 모델을 이어서 미세조정하고, 그 자리에서 판정까지 낸다.

학습만 하고 끝내지 않는다.  base 를 같은 val 로 먼저 재고, FT 후 다시 재서 차이를
출력한다.  완료 판정은 프로세스 종료코드가 아니라 **산출물(weights/best.pt 와 결과 JSON)**
로 한다.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[3]
TRACK = REPO / "challenge/yolo_pose_one_model"
BASE = TRACK / "release/pallet-pose-yolo26n-ft/pallet_yolo26n_pose_ft.pt"
RUN_DIR = TRACK / "runs_live_gt"

EPOCHS = 40
BATCH = 32
IMGSZ = 640

# base(synthetic) 학습 계약 그대로.  소량 FT 를 ultralytics 기본 aug 로 돌리면
# base 가 배운 조건과 어긋난다 — mosaic 1.0 / fliplr 0.5 는 402장에 특히 공격적이다.
BASE_CONTRACT_AUG = dict(
    optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3.0,
    mosaic=0.3, close_mosaic=10, scale=0.25, fliplr=0.0, flipud=0.0,
    translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
)


def metrics_of(result) -> dict:
    """ultralytics 결과에서 비교에 쓸 값만 뽑는다."""
    box, pose = result.box, result.pose
    return {
        "box_mAP50": float(box.map50), "box_mAP50_95": float(box.map),
        "pose_mAP50": float(pose.map50), "pose_mAP50_95": float(pose.map),
        "box_precision": float(box.mp), "box_recall": float(box.mr),
    }


def drop_label_cache() -> None:
    """ultralytics 의 라벨 캐시를 지운다.

    라벨을 다시 만들어도 옛 ``labels/*.cache`` 를 그대로 읽는 일이 실제로 있었다.
    그때 pose mAP 가 0.945 대신 0.0157 로 나와, 규약이 깨진 것처럼 보였다.
    평가 전에 지우는 비용은 스캔 몇 초뿐이다.
    """
    for cache in (DATA.parent / "labels").glob("*.cache"):
        cache.unlink()
        print(f"   캐시 제거: {cache.name}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/live_gt_v2/data.yaml")
    ap.add_argument("--name", default="ft_live_gt_v2")
    ap.add_argument("--aug", choices=["base_contract", "ultralytics_default"],
                    default="base_contract")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args(argv)

    global DATA
    DATA = TRACK / args.data
    name = args.name
    print(f"base : {BASE.name}")
    drop_label_cache()
    print(f"data : {DATA}")
    print(f"aug  : {args.aug}   epochs {args.epochs}")

    print("\n── 1) base 를 val 로 먼저 잰다 (비교 기준) ──", flush=True)
    before = metrics_of(YOLO(str(BASE)).val(
        data=str(DATA), imgsz=IMGSZ, batch=BATCH, split="val",
        project=str(RUN_DIR), name=f"{name}__base_eval", exist_ok=True, verbose=False))
    for k, v in before.items():
        print(f"   {k:16} {v:.4f}")

    print(f"\n── 2) FT {args.epochs} epochs ──", flush=True)
    extra = BASE_CONTRACT_AUG if args.aug == "base_contract" else {}
    started = time.time()
    YOLO(str(BASE)).train(
        data=str(DATA), epochs=args.epochs, batch=BATCH, imgsz=IMGSZ,
        project=str(RUN_DIR), name=name, exist_ok=True,
        patience=0, seed=42, workers=4, val=True, plots=False, verbose=False,
        **extra)
    elapsed = time.time() - started

    weights = RUN_DIR / name / "weights/best.pt"
    if not weights.is_file():                     # 산출물로만 완료를 판정한다
        print(f"[FAIL] 가중치가 없다: {weights}")
        return 1

    print(f"\n── 3) FT 결과를 같은 val 로 잰다 ──", flush=True)
    after = metrics_of(YOLO(str(weights)).val(
        data=str(DATA), imgsz=IMGSZ, batch=BATCH, split="val",
        project=str(RUN_DIR), name=f"{name}__ft_eval", exist_ok=True, verbose=False))

    print(f"\n{'지표':18}{'base':>10}{'FT':>10}{'차이':>10}")
    print("─" * 48)
    for k in before:
        d = after[k] - before[k]
        print(f"{k:18}{before[k]:10.4f}{after[k]:10.4f}{d:+10.4f}")

    key = "pose_mAP50_95"
    delta = after[key] - before[key]
    verdict = "IMPROVED" if delta > 0.005 else ("REGRESSED" if delta < -0.005 else "FLAT")
    print(f"\n판정: {verdict}   ({key} {delta:+.4f})")
    print(f"학습 시간: {elapsed/60:.1f}분  ({elapsed/args.epochs:.1f}초/epoch)")

    (RUN_DIR / name / "VERDICT.json").write_text(json.dumps({
        "base": str(BASE.relative_to(REPO)), "data": str(DATA.relative_to(REPO)),
        "aug": args.aug, "aug_overrides": extra,
        "epochs": args.epochs, "batch": BATCH, "imgsz": IMGSZ,
        "train_seconds": elapsed, "seconds_per_epoch": elapsed / EPOCHS,
        "before": before, "after": after,
        "delta": {k: after[k] - before[k] for k in before},
        "verdict": verdict,
        "note": "real 수동 어노만. negative/synthetic 미포함 (사용자 지시).",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {RUN_DIR / name / 'VERDICT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
