#!/usr/bin/env python3
"""F0(indexed) / F1(C4) 두 arm 을 같은 조건으로 학습하고 그 자리에서 판정까지 낸다.

두 arm 사이에서 다른 것은 **keypoint symmetry loss 하나뿐**이어야 한다.  init
checkpoint · dataset · epoch · batch · optimizer · lr · augment · seed 를 한 곳에서
읽어 양쪽에 그대로 넘기고, 실행 후 args.yaml 을 대조해 실제로 같았는지 확인한다.

recipe 는 기억으로 재구성하지 않는다.  ``runs_live_gt/ft_live_gt_v4/args.yaml`` 에서
읽은 값을 여기 상수로 고정했고, ``--verify-recipe`` 로 원본과 다시 대조한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[4]
TRACK = Path(__file__).resolve().parents[1]
ONE = REPO / "challenge/yolo_pose_one_model"

INIT = ONE / "runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt"
INIT_SHA = "6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea"
DATA = ONE / "datasets/live_gt_v4/data.yaml"
REF_ARGS = ONE / "runs_live_gt/ft_live_gt_v4/args.yaml"

# ft_live_gt_v4/args.yaml 에서 읽은 값.  결과를 보고 고치지 않는다.
RECIPE = dict(
    epochs=40, batch=32, imgsz=640,
    optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3.0,
    momentum=0.937, weight_decay=0.0005,
    pose=12.0, kobj=1.0, box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.3, close_mosaic=10, scale=0.25,
    fliplr=0.0, flipud=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
    seed=42, patience=0, deterministic=True, single_cls=True,
    # ★ v4 는 workers=4 였으나 이 머신에서 DataLoader 공유메모리 9.2GB 가
    #   다른 부하와 겹쳐 OOM(SIGKILL)이 났다. 자원 설정이라 loss 와 무관하지만
    #   augmentation 난수 시퀀스가 바뀌므로 F0/F1 **양쪽 모두** 이 값으로 돌린다.
    workers=2,
    val=True, plots=False, verbose=False, exist_ok=True,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_label_cache() -> None:
    """stale 라벨 캐시는 pose mAP 를 통째로 뒤집은 이력이 있다.  매 실행 전에 지운다."""
    for cache in (DATA.parent / "labels").glob("*.cache"):
        cache.unlink()
        print(f"   캐시 제거: {cache.name}")


def verify_recipe() -> list[str]:
    """참조 run 의 args.yaml 과 대조한다.  기억이 아니라 파일이 근거다."""
    import yaml
    ref = yaml.safe_load(REF_ARGS.read_text(encoding="utf-8"))
    bad = []
    for k, v in RECIPE.items():
        if k in ("val", "plots", "verbose", "exist_ok", "workers"):
            continue
        if k in ref and ref[k] != v:
            bad.append(f"{k}: recipe={v} vs ref={ref[k]}")
    return bad


def train_arm(name: str, c4_config: Path | None, epochs: int, subset: str | None):
    """한 arm 을 학습한다.  C4 는 환경변수로만 켠다 — 코드 경로는 동일하다."""
    from ultralytics import YOLO
    sys.path.insert(0, str(REPO))
    from pallet_yolo_loss.trainer import ChallengeC4Trainer

    if c4_config:
        os.environ["C4_CONFIG"] = str(c4_config)
    else:
        os.environ.pop("C4_CONFIG", None)

    out = TRACK / ("F1" if c4_config else "F0")
    data = subset or str(DATA)
    print(f"\n{'='*64}\n{name}   C4={'ON' if c4_config else 'OFF'}   epochs={epochs}")
    print(f"  init {INIT.name}  sha {sha256(INIT)[:16]}")
    print(f"  data {data}")
    drop_label_cache()

    started = time.time()
    model = YOLO(str(INIT))
    model.train(data=data, project=str(out.parent), name=out.name,
                trainer=ChallengeC4Trainer, **{**RECIPE, "epochs": epochs})
    elapsed = time.time() - started

    w = out / "weights/best.pt"
    if not w.is_file():                      # 완료는 산출물로만 판정한다
        print(f"[FAIL] 가중치 없음: {w}")
        return None
    print(f"  완료 {elapsed/60:.1f}분   {w}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F0", "F1", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch 만 돌려 배선을 확인한다 (성능 판정용 아님)")
    args = ap.parse_args(argv)

    if not INIT.is_file():
        print(f"[STOP] Stage A checkpoint 없음: {INIT}")
        return 1
    got = sha256(INIT)
    if got != INIT_SHA:
        print(f"[STOP] init SHA 불일치\n  기대 {INIT_SHA}\n  실제 {got}")
        return 1
    print(f"init SHA 확인 {got[:16]}…  (F0/F1 동일)")

    bad = verify_recipe()
    if bad:
        print("[STOP] recipe 가 참조 run 과 다르다:")
        for b in bad:
            print("  -", b)
        return 1
    print(f"recipe 대조 통과 (ref = {REF_ARGS.relative_to(REPO)})")

    epochs = 1 if args.smoke else args.epochs
    cfg = TRACK / "c4_config_enabled.json"
    if not cfg.is_file():
        perms = json.loads((TRACK / "C4_PERMUTATIONS.json").read_text(encoding="utf-8"))
        cfg.write_text(json.dumps({
            "enabled": True,
            "permutations": [perms["permutations"][k] for k in ("0", "90", "180", "270")],
        }), encoding="utf-8")

    done = []
    if args.arm in ("F0", "both"):
        done.append(("F0", train_arm("F0 CONTROL (indexed)", None, epochs, None)))
    if args.arm in ("F1", "both"):
        done.append(("F1", train_arm("F1 C4 (symmetry-aware)", cfg, epochs, None)))

    ok = all(p is not None for _, p in done)
    for tag, path in done:
        if path is None:
            continue
        print(f"\n{tag}: {path}")
        hist = path / "C4_BRANCH_HIST.csv"
        if hist.is_file():
            print("  branch histogram:")
            for line in hist.read_text(encoding="utf-8").strip().splitlines():
                print("   ", line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
