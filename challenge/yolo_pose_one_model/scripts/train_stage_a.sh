#!/usr/bin/env bash
# Stage A - general + target geometry pretraining on synthetic data only.
# batch is fixed at 32. If it OOMs, diagnose the GPU; do NOT lower the batch.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26

cd "$(git rev-parse --show-toplevel)"
ROOT=challenge/yolo_pose_one_model
# absolute: Ultralytics resolves a relative project against SETTINGS['runs_dir']
RUNS="$(pwd)/$ROOT/runs"

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${FREE} MiB"
if [ "$FREE" -lt 7168 ]; then
  echo "ABORT: free VRAM ${FREE} MiB < 7168 MiB."
  echo "Diagnose before doing anything else - do not change batch:"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  exit 1
fi

test -f $ROOT/configs/stage_a.yaml
test -f challenge/weights/pretrained_yolo/yolo26n-pose.pt

yolo pose train \
  model=challenge/weights/pretrained_yolo/yolo26n-pose.pt \
  data=$ROOT/configs/stage_a.yaml \
  imgsz=640 \
  batch=32 \
  nbs=64 \
  epochs=60 \
  patience=15 \
  optimizer=SGD \
  lr0=0.01 \
  lrf=0.01 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=3.0 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
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
  translate=0.10 \
  scale=0.25 \
  mosaic=0.30 \
  close_mosaic=10 \
  mixup=0.0 \
  copy_paste=0.0 \
  hsv_h=0.015 \
  hsv_s=0.50 \
  hsv_v=0.35 \
  save=True \
  save_period=5 \
  plots=True \
  verbose=True \
  project="$RUNS" \
  name=stage_a_synth_640_b32_seed42 \
  exist_ok=False

echo "STAGE_A_DONE"
