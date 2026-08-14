#!/usr/bin/env bash
# Waits for STAGE24 quick-screen to finish, then runs SEEN eval (v3 batch_000,
# which was in training) for the voting arm: heatmap-arm vs voting-seg vs
# voting-gtmask corner median. Gate input = does voting corner err improve on
# SEEN vs STAGE10's 53.8px, and vs its own heatmap-arm.
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1
L=data/pallet/eval_results/stage24_vec_newdata/logs
run () { conda run -n pallet-pose python "$@"; }

# wait for quick-screen
for i in $(seq 1 60); do
  s=$(cat "$L/STATUS" 2>/dev/null || echo "")
  [ "$s" = "QUICKSCREEN_DONE" ] && break
  case "$s" in *_FAIL*) echo "QUICKSCREEN FAILED: $s"; exit 1;; esac
  sleep 30
done
s=$(cat "$L/STATUS" 2>/dev/null || echo "")
[ "$s" != "QUICKSCREEN_DONE" ] && { echo "not done: $s"; exit 1; }
echo "===== quick-screen done, running SEEN eval $(date +%H:%M:%S) ====="

VW=weights/stage24_vec_newdata/voting/final_net_voting_unit.pth
# SEEN eval: v3 batch_000 (in training), real seg-head mask voting + gt mask
run scripts/stage0/eval_harness/eval_pvnet_heads.py --weights "$VW" --vec_mode unit \
  --evalset v3 --batch_dir challenge/data/02_synthetic/training/v3/batch_000 \
  --n_frames 60 --use_seg_mask --by_visibility --arm_tags S24seen \
  --tag s24_seen > "$L/seen_eval.log" 2>&1
echo "----- seen eval rc=$? -----"
echo "SEEN_EVAL_DONE" > "$L/STATUS2"
echo "===== SEEN EVAL DONE $(date +%H:%M:%S) ====="
