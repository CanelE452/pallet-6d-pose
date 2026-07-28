#!/usr/bin/env bash
# diffpnp3d_q1_eval_all.sh — eval every lambda's final (ep8) checkpoint on val 500.
set -e
ROOT=/home/minjae/Documents/github/pallet-pose
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pallet-pose
cd "$ROOT"
LAMBDAS="${1:-0 0.003 0.005 0.008}"
for LAM in $LAMBDAS; do
  CKPT="$ROOT/weights/paper_s2_q1/lam${LAM}/final_net_epoch_0008.pth"
  if [ ! -f "$CKPT" ]; then echo "[miss] $CKPT"; continue; fi
  python scripts/stage0/diffpnp3d_q1_eval.py --weights "$CKPT" --tag "lam${LAM}"
done
echo "EVAL ALL DONE"
