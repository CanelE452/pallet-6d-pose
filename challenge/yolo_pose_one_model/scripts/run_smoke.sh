#!/usr/bin/env bash
# Smoke test: 2 epochs on 512 train / 128 val at the real batch size.
# Its only job is to prove the pipeline runs at batch=32 without OOM or NaN.
# Metrics from 2 epochs on 512 frames mean nothing and must not be quoted.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26

cd "$(git rev-parse --show-toplevel)"
ROOT=challenge/yolo_pose_one_model
# project must be absolute: Ultralytics resolves a relative project against
# SETTINGS['runs_dir'] ("runs"), which would bury the run under runs/pose/<project>.
RUNS="$(pwd)/$ROOT/runs"

# Gate: do not start if another process is holding the GPU.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${FREE} MiB"
if [ "$FREE" -lt 7168 ]; then
  echo "ABORT: free VRAM ${FREE} MiB < 7168 MiB. Report the holding processes; do not lower batch."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  exit 1
fi

yolo pose train \
  model=challenge/weights/pretrained_yolo/yolo26n-pose.pt \
  data=$ROOT/configs/smoke.yaml \
  imgsz=640 \
  batch=32 \
  nbs=64 \
  epochs=2 \
  optimizer=SGD \
  lr0=0.01 \
  lrf=0.01 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=1.0 \
  cos_lr=True \
  amp=True \
  device=0 \
  workers=4 \
  cache=False \
  seed=42 \
  deterministic=True \
  single_cls=True \
  pretrained=True \
  resume=False \
  multi_scale=False \
  fliplr=0.0 \
  flipud=0.0 \
  degrees=0.0 \
  shear=0.0 \
  perspective=0.0 \
  translate=0.05 \
  scale=0.2 \
  mosaic=0.2 \
  close_mosaic=0 \
  mixup=0.0 \
  copy_paste=0.0 \
  save=True \
  plots=True \
  verbose=True \
  project="$RUNS" \
  name=smoke_b32_seed42 \
  exist_ok=False
