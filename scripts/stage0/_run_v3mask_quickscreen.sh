#!/usr/bin/env bash
# v3 real-mask quick screen: M0/M1/M2 3-arm finetune from frozen base.
# ★ quick screen only — NOT full 10k. train=batch_000~007 subsample ~3000.
set -e
cd /home/minjae/Documents/github/pallet-pose
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pallet-pose
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1

BASE=weights/challenge0123/final_net_epoch_0060.pth
DATA="challenge/data/training/v3/batch_000 challenge/data/training/v3/batch_001 \
      challenge/data/training/v3/batch_002 challenge/data/training/v3/batch_003 \
      challenge/data/training/v3/batch_004 challenge/data/training/v3/batch_005 \
      challenge/data/training/v3/batch_006 challenge/data/training/v3/batch_007"
OUT=data/pallet/eval_results/stage10_v3mask
SUB=3000; EP=3; BS=12; LRV=0.1

common="--weights $BASE --data $DATA --subsample $SUB --epochs $EP \
        --batchsize $BS --workers 4 --lam_vec $LRV --lam_seg 1.0 \
        --warmup 200 --save_every 1 --log_every 25"

echo "=== M0 (heatmap only, control) ==="
python scripts/stage0/finetune_pvnet.py --vec_mode off $common \
  --outf $OUT/M0 --namefile M0 2>&1 | tail -5

echo "=== M1 (heatmap + seg mask aux, real mask_rle) ==="
python scripts/stage0/finetune_pvnet.py --vec_mode off --seg --pvnet_mask_rle $common \
  --outf $OUT/M1 --namefile M1 2>&1 | tail -5

echo "=== M2 (heatmap + unit-vec + seg, real mask_rle) ==="
python scripts/stage0/finetune_pvnet.py --vec_mode unit --seg --pvnet_mask_rle $common \
  --outf $OUT/M2 --namefile M2 2>&1 | tail -5

echo "=== ALL ARMS DONE ==="
