#!/usr/bin/env bash
# YOLO26N_PAPER_GENERIC_V1_SEED43 — seed42 와 seed 만 다르다.
# ★ seed42 가 STRONG_PASS 이기 전에는 실행하지 않는다. AUTORUN_NEXT = False.
set -uo pipefail
ROOT=/home/minjae/Documents/github/pallet-pose
YR=$ROOT/challenge/yolo_pose_one_model
NOTIFY=$HOME/.claude/hooks/discord-notify.sh
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pallet-yolo26
cd "$ROOT"

V=$(python -c "import json;print(json.load(open('$YR/evaluation/PAPER_YOLO_VERDICT.json'))['verdict'])" 2>/dev/null || echo NONE)
if [ "$V" != "STRONG_PASS" ]; then
  echo "차단: seed42 판정이 $V 다. STRONG_PASS 일 때만 실행한다."; exit 1
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$FREE" -ge 7168 ] || { echo "차단: free VRAM $FREE"; exit 1; }

yolo pose train \
  model=challenge/weights/pretrained_yolo/yolo26n-pose.pt \
  data=$YR/datasets/broad40k/data.yaml \
  imgsz=640 batch=32 nbs=64 epochs=60 patience=0 \
  optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  cos_lr=True amp=True device=0 workers=4 cache=False \
  seed=43 deterministic=True single_cls=True pretrained=True \
  multi_scale=False fliplr=0.0 flipud=0.0 degrees=0.0 shear=0.0 \
  perspective=0.0 translate=0.10 scale=0.25 mosaic=0.30 close_mosaic=10 \
  mixup=0.0 copy_paste=0.0 hsv_h=0.015 hsv_s=0.50 hsv_v=0.35 \
  save=True save_period=10 plots=True verbose=True \
  project="$YR/runs_paper" name=yolo26n_paper_generic_v1_seed43 exist_ok=False

python $YR/paper_generic_pipeline/seed_replicates/compare_args.py \
  $YR/runs_paper/yolo26n_paper_generic_v1_seed43/args.yaml \
  || { "$NOTIFY" "❌ seed43 args 가 seed42 와 다르다"; exit 1; }
"$NOTIFY" "seed43 60ep 완료 — args seed42 와 동일 확인" || true
