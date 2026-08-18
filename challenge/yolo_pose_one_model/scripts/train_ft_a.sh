#!/usr/bin/env bash
# FT-A — stage_a(합성 전용) 위에 real GT 157 + 배포환경 negative 259 를 얹는 finetuning.
#
# 목적/지표는 runs_ft/PURPOSE.md 에 고정돼 있다. 주효과는 FP 억제이고, real 157장은
# 합성 73,916 의 0.2% 라 도메인 적응은 부수적이다(과장하지 말 것).
#
# stage_a 와 다른 점만:
#   model  = stage_a best.pt        (scratch 아님)
#   lr0    = 0.01 -> 0.002          finetuning 이라 낮춘다. 그대로 두면 합성에서 배운 걸 흔든다
#   epochs = 60 -> 40, warmup 3 -> 1
#   mosaic = 0.30 -> 0.15           real 157장이 mosaic 으로 과하게 합성되면 negative 신호가 흐려진다
# 나머지(fliplr=0, seed, nbs, optimizer 등)는 stage_a 와 동일하게 둔다 — 비교 가능해야 한다.
set -euo pipefail

# 인자화: 같은 레시피를 patience/epochs/이름만 바꿔 다시 돌린다. 스크립트를 복사해
# 늘리면 설정이 두 벌로 갈라져 다음에 반드시 어긋난다.
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-15}"
RUN_NAME="${RUN_NAME:-ft_a_real157_neg259_synth12k}"
# BASE 도 인자화 — medium(yolo26m) finetuning 은 base 만 바꾸면 같은 레시피로 돈다.
BASE_DEFAULT=challenge/yolo_pose_one_model/runs/stage_a_synth_640_b32_seed42/weights/best.pt
DATA="${DATA:-challenge/yolo_pose_one_model/datasets/ft_a/data.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26

cd "$(git rev-parse --show-toplevel)"
ROOT=challenge/yolo_pose_one_model
RUNS="$(pwd)/$ROOT/runs_ft"        # absolute: Ultralytics 는 상대 project 를 runs_dir 기준으로 푼다
BASE="${BASE:-$BASE_DEFAULT}"

test -f "$RUNS/PURPOSE.md"         # 목적·지표 없이 학습하지 않는다
test -f "$BASE"
test -f "$DATA"

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${FREE} MiB"
if [ "$FREE" -lt 7168 ]; then
  echo "ABORT: free VRAM ${FREE} MiB < 7168 MiB. batch 를 낮추지 말고 원인을 찾아라."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  exit 1
fi

yolo pose train \
  model="$BASE" \
  data="$DATA" \
  imgsz=640 \
  batch=32 \
  nbs=64 \
  epochs=$EPOCHS \
  patience=$PATIENCE \
  optimizer=SGD \
  lr0=0.002 \
  lrf=0.01 \
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=1.0 \
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
  mosaic=0.15 \
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
  name=$RUN_NAME \
  exist_ok=False

echo "FT_A_DONE  ($RUN_NAME, epochs=$EPOCHS patience=$PATIENCE)"
