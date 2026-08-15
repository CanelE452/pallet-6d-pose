#!/usr/bin/env bash
# diffpnp3d_q1_train.sh — PAPER_S2 Phase 3 (Q1) lambda sweep, sequential.
# ALL runs share: fixed q1_split train (3000), scratch, batch12, sigma2, aspect,
# --diffpnp ON (aspect_resize + eligible-frame rotate-skip identical pipeline),
# same --manualseed (identical init + shuffle). ONLY variable = --diffpnp_lambda.
set -e
ROOT=/home/minjae/Documents/github/pallet-pose
SPLIT=$ROOT/data/pallet/results/paper_s2_scratch_diffpnp/q1_split
OUTBASE=$ROOT/weights/paper_s2/paper_s2_q1
LOGDIR=$ROOT/data/pallet/results/paper_s2_scratch_diffpnp/q1_logs
mkdir -p "$OUTBASE" "$LOGDIR"

export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pallet-pose
cd "$ROOT/Deep_Object_Pose/train"

LAMBDAS="${1:-0 0.003 0.005 0.008}"
SEED=42
EPOCHS=8

for LAM in $LAMBDAS; do
  TAG="lam${LAM}"
  OUTF="$OUTBASE/$TAG"
  LOG="$LOGDIR/${TAG}.log"
  if [ -f "$OUTF/final_net_epoch_0008.pth" ]; then
    echo "[skip] $TAG already has final_net_epoch_0008.pth"
    continue
  fi
  mkdir -p "$OUTF"
  echo "=== training $TAG (lambda=$LAM) -> $OUTF ==="
  python train.py \
    --data "$SPLIT/train" \
    --object pallet \
    --batchsize 12 \
    --sigma 2 \
    --epochs $EPOCHS \
    --lr 0.0001 \
    --imagesize 400 \
    --workers 6 \
    --manualseed $SEED \
    --outf "$OUTF" \
    --diffpnp \
    --diffpnp_lambda "$LAM" \
    --diffpnp_warmup 0 \
    --diffpnp_ramp 500 \
    --diffpnp_temp 0.1 \
    --diffpnp_index_dir "$SPLIT/train_index" \
    --diffpnp_root "$SPLIT" \
    > "$LOG" 2>&1
  echo "=== done $TAG ==="
done
echo "ALL DONE"
