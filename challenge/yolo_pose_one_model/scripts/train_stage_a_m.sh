#!/usr/bin/env bash
# Stage A for yolo26m-pose. Same data, seed and schedule as the nano run; only the
# model and the batch differ.
#
# Why batch 16 and not 32 (2026-08-15):
#   A probe said batch 32 fit at 8.34 GiB, but the real run OOMed immediately and
#   Ultralytics silently stepped down 32 -> 16 -> 8. The probe was too optimistic: it
#   used ~300 frames with no val loader, and 8.34 / 8.9 GiB is 94% of what is free.
#   batch 16 is requested explicitly here so the run name matches what actually runs.
#
#   nbs=64 keeps the EFFECTIVE batch at 64 either way (accumulate = 64/batch), so the
#   optimiser sees the same thing as the nano run and lr0 stays 0.01.
#
# expandable_segments cuts allocator fragmentation - the repo hit the same class of
# failure on 2026-04-10 and this setting fixed it.
set -euo pipefail

BATCH="${1:-16}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26
cd "$(git rev-parse --show-toplevel)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=challenge/yolo_pose_one_model
RUNS="$(pwd)/$ROOT/runs"
NOTIFY="$HOME/.claude/hooks/discord-notify.sh"

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${FREE} MiB, requesting batch=${BATCH}"
if [ "$FREE" -lt 7168 ]; then
  echo "ABORT: free VRAM ${FREE} MiB < 7168 MiB"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  exit 1
fi

yolo pose train \
  model=challenge/weights/pretrained_yolo/yolo26m-pose.pt \
  data=$ROOT/configs/stage_a.yaml \
  imgsz=640 \
  batch=$BATCH \
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
  name=stage_a_m_640_b${BATCH}_seed42 \
  exist_ok=False
RC=$?

CSV=$ROOT/runs/stage_a_m_640_b${BATCH}_seed42/results.csv
EP=$(( $(wc -l < "$CSV" 2>/dev/null || echo 1) - 1 ))
SUM=$(tail -1 "$CSV" 2>/dev/null | awk -F, '{printf "epoch %s | Box mAP50 %.4f | Pose mAP50 %.4f mAP50-95 %.4f | %.1f h", $1, $11, $15, $16, $2/3600}')
if [ "$RC" = "0" ]; then
  "$NOTIFY" "[stage_a_m] medium 학습 완료 (요청 batch=${BATCH})
${SUM}" || true
else
  "$NOTIFY" "[stage_a_m] medium 학습 실패 (exit ${RC}, ${EP} epoch, 요청 batch=${BATCH})" || true
fi
echo "medium finished rc=$RC at ${EP} epochs"
