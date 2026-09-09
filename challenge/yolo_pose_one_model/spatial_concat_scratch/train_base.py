"""Train the clean-start mixed-data YOLO26n-pose base for spatial concat.

"Clean-start" here means no G38, Y0E, or legacy-finetuned pallet checkpoint.
The run starts from the same official generic YOLO26n-pose initialization used
by the prior comparable fresh pallet-training runs.  It is intentionally not a
random-weight arm; that distinction is frozen in TRAINING_CONTRACT.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = REPO / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml"
PRETRAINED = REPO / "challenge/weights/pretrained_yolo/yolo26n-pose.pt"
PROJECT = HERE / "runs"
RUN_NAME = "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42"
RUN_DIR = PROJECT / RUN_NAME

EXPECTED_SHA256 = {
    DATA: "19e0501c835df11e7d2356864bf7cd38da367618fd07e3822fad9ccc92417ec5",
    PRETRAINED: "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    HERE / "TRAINING_CONTRACT.json": None,
}
EXPECTED_COUNTS = {"train": 55980, "val": 4020}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_count(split: str) -> int:
    root = DATA.parent
    images = {p.stem for p in (root / "images" / split).glob("*.png")}
    labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
    if images != labels:
        raise RuntimeError(
            f"{split} image/label mismatch: images_only={len(images-labels)}, "
            f"labels_only={len(labels-images)}"
        )
    return len(images)


def runtime_audit() -> dict:
    observed = {str(path): sha256(path) for path in EXPECTED_SHA256}
    for path, expected in EXPECTED_SHA256.items():
        if expected is not None and observed[str(path)] != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
    counts = {split: dataset_count(split) for split in ("train", "val")}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"dataset counts {counts} != {EXPECTED_COUNTS}")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "schema_version": "mixed_base_runtime_binding_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": observed,
        "dataset_counts": counts,
        "python_torch": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "ultralytics": ultralytics.__version__,
        "gpu": gpu,
        "run_dir": str(RUN_DIR),
        "prior_pallet_checkpoint_loaded": False,
        "random_initialization": False,
        "official_generic_pretrained_initialization": True,
    }


def train_new() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(
            f"run directory already exists; use --resume for an interrupted run: {RUN_DIR}"
        )
    audit = runtime_audit()
    HERE.joinpath("BASE_RUN_BINDING.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    init_seeds(42, deterministic=True)
    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA),
        epochs=60,
        patience=0,
        batch=32,
        imgsz=640,
        save=True,
        save_period=5,
        cache=False,
        device="0",
        workers=2,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        plots=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=0.0,
        translate=0.1,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        bgr=0.0,
        mosaic=0.3,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.4,
    )


def resume() -> None:
    checkpoint = RUN_DIR / "weights/last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    YOLO(str(checkpoint)).train(resume=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    resume() if args.resume else train_new()


if __name__ == "__main__":
    main()
