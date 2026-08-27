"""3-arm 10ep screen 실행 — HC / HM / HF.

세 arm 이 다른 것은 두 가지뿐이다.

    HC   dataset = hn_hc    (G38 + positive repeat 1,900)   loss = stock
    HM   dataset = hn_hard  (G38 + HARD_NEG1900)            loss = stock
    HF   dataset = hn_hard  (HM 과 동일)                     loss = focal-negative

HM 과 HF 가 같은 폴더를 쓰므로 "데이터가 같다" 를 증명할 필요가 없다 — 구조적으로 같다.
HC/HM 은 `HNTrainer.arm != "HF"` 라 criterion 을 건드리지 않으므로 stock 임이 보장된다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
HN = os.path.join(ROOT, "challenge/yolo_pose_one_model/hard_negative_v1")
DS = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets")
INIT = os.path.join(ROOT, "challenge/weights/pretrained_yolo/yolo26n-pose.pt")
sys.path.insert(0, HN)

ARMS = {
    "HC": ("hn_hc", "HC_POSREPEAT1900"),
    "HM": ("hn_hard", "HM_HARDNEG1900_STOCK"),
    "HF": ("hn_hard", "HF_HARDNEG1900_FOCALNEG"),
}
COMMON = dict(epochs=10, batch=32, imgsz=640, seed=42, optimizer="SGD",
              lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3, patience=0,
              single_cls=True, fliplr=0.0, flipud=0.0, deterministic=True,
              resume=False, mosaic=0.3, scale=0.25, erasing=0.4,
              close_mosaic=10, val=True, plots=False, save_period=-1,
              workers=8, verbose=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list(ARMS))
    args = ap.parse_args()

    lam = json.load(open(os.path.join(HN, "preflight/GRADIENT_CALIBRATION.json")))
    lambda_neg = lam["FIXED_LAMBDA_NEG"]

    import hn_trainer
    hn_trainer.HNTrainer.arm = args.arm
    hn_trainer.HNTrainer.lambda_neg = lambda_neg if args.arm == "HF" else 0.0

    ds_name, run_name = ARMS[args.arm]
    overrides = dict(COMMON)
    overrides.update(
        model=INIT, task="pose",
        data=os.path.join(DS, ds_name, "data.yaml"),
        project=os.path.join(HN, "runs"), name=run_name, exist_ok=True,
    )

    audit = {
        "arm": args.arm, "run": run_name, "dataset": ds_name,
        "init": os.path.relpath(INIT, ROOT),
        "init_sha256": hashlib.sha256(open(INIT, "rb").read()).hexdigest(),
        "lambda_neg": hn_trainer.HNTrainer.lambda_neg,
        "loss": "focal-negative (hn_loss)" if args.arm == "HF" else "stock PoseLoss26",
        "mosaic": "sample-type dependent — negative base sample 이면 OFF",
        "overrides": {k: v for k, v in overrides.items() if k != "model"},
    }
    os.makedirs(os.path.join(HN, "runs"), exist_ok=True)
    json.dump(audit, open(os.path.join(HN, "runs", f"RUNTIME_AUDIT_{args.arm}.json"),
                          "w"), indent=1, ensure_ascii=False)
    print(f"[{args.arm}] dataset={ds_name}  lambda_neg={audit['lambda_neg']}  "
          f"loss={audit['loss']}", flush=True)

    trainer = hn_trainer.HNTrainer(overrides=overrides)
    trainer.train()
    print(f"[{args.arm}] DONE -> {trainer.save_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
