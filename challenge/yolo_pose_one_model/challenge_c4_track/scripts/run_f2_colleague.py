#!/usr/bin/env python3
"""F2 — 동료 방식 재현: v4 checkpoint 에서 이어서 C4 FT.

F1 과 loss 는 같고 **출발점이 다르다**.

    동료   pallet_yolo26n_pose_ft.pt -> [indexed FT] -> v4 -> [C4 FT] -> c4
    F1     PAPER_GENERIC Stage A     -> [C4 FT]      -> F1
    F2     v4                        -> [C4 FT]      -> F2      ← 이 스크립트

데이터는 crop 을 빼고 원본 + flip/noise 만 쓴다 (사용자 지시).  **val 155 장은
F0/F1 과 동일**해야 세 모델을 같은 잣대로 비교할 수 있으므로 건드리지 않았다.

recipe 는 기억으로 재구성하지 않고 `runs_live_gt/ft_live_gt_v4/args.yaml` 에서 읽은
값을 쓴다 — `run_c4_arms.RECIPE` 를 그대로 import 한다(두 벌이 갈라지지 않게).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TRACK = HERE.parent
REPO = HERE.parents[3]   # scripts -> track -> one_model -> challenge -> repo
sys.path.insert(0, str(REPO))
ONE = REPO / "challenge/yolo_pose_one_model"

from run_c4_arms import RECIPE, sha256  # noqa: E402  recipe 를 공유한다

BASE = ONE / "runs_live_gt/ft_live_gt_v4/weights/best.pt"
BASE_SHA = "2c3b6286c9a1c669e2533a43f8700b01d30231ed5b836e1b262945ff08b226b2"
DATA = ONE / "datasets/live_gt_v5_nocrop/data.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="F2")
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--indexed", action="store_true",
                    help="C4 를 끄고 돌린다 (F2 의 control)")
    args = ap.parse_args(argv)

    if not BASE.is_file():
        print(f"[STOP] v4 checkpoint 없음: {BASE}")
        return 1
    got = sha256(BASE)
    if got != BASE_SHA:
        print(f"[STOP] base SHA 불일치\n  기대 {BASE_SHA}\n  실제 {got}")
        return 1
    print(f"base  v4  sha {got[:16]}…")

    cfg = TRACK / "c4_config_enabled.json"
    if args.indexed:
        os.environ.pop("C4_CONFIG", None)
    else:
        os.environ["C4_CONFIG"] = str(cfg)
    print(f"loss  {'indexed (control)' if args.indexed else 'C4 symmetry'}")
    print(f"data  {DATA}")

    # stale 캐시는 pose mAP 를 통째로 뒤집은 이력이 있다.
    for c in (DATA.parent / "labels").glob("*.cache"):
        c.unlink()
        print(f"   캐시 제거: {c.name}")

    from ultralytics import YOLO
    from pallet_yolo_loss.trainer import ChallengeC4Trainer

    out = TRACK / args.name
    started = time.time()
    YOLO(str(BASE)).train(data=str(DATA), project=str(TRACK), name=args.name,
                          trainer=ChallengeC4Trainer,
                          **{**RECIPE, "epochs": args.epochs})
    elapsed = time.time() - started

    w = out / "weights/best.pt"
    if not w.is_file():                       # 완료는 산출물로만 판정한다
        print(f"[FAIL] 가중치 없음: {w}")
        return 1
    print(f"\n완료 {elapsed/60:.1f}분   {w}\n  sha {sha256(w)[:16]}…")

    hist = out / "C4_BRANCH_HIST.csv"
    if hist.is_file():
        lines = hist.read_text(encoding="utf-8").strip().splitlines()
        print("  branch histogram (마지막 2줄):")
        for line in lines[-2:]:
            print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
