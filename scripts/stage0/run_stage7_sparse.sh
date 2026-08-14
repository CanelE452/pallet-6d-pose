#!/usr/bin/env bash
# STEP7 sparse edge 벡터 — 밤샘 무인. control vs sparse(같은 스케줄) → 3-arm eval.
# 스모크 게이트 → 본학습 2-arm → refine eval → ALL_DONE. 실패/NaN 즉시 중단.
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1

ROOT=data/pallet/eval_results/stage7_sparse
L=$ROOT/logs; mkdir -p "$L"
run () { conda run -n pallet-pose python "$@"; }
COMMON="--subsample 0 --epochs 10 --batchsize 16 --workers 6 --lam_edgevec 0.1 --warmup 2000 --save_every 1 --log_every 50"

echo "===== SMOKE $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_edgevec.py --edge_head --subsample 32 --epochs 1 --max_iters 2 \
  --lam_edgevec 0.1 --warmup 2000 --outf /tmp/stage7_smoke > "$L/smoke.log" 2>&1
if [ $? -ne 0 ]; then echo "SMOKE_FAIL" > "$L/STATUS"; echo SMOKE_FAIL; exit 1; fi
echo "----- smoke OK $(date +%H:%M:%S) -----"

echo "===== TRAIN control (heatmap only) $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_edgevec.py $COMMON --outf "$ROOT/run_control" > "$L/train_control.log" 2>&1
rc=$?; echo "----- control rc=$rc $(date +%H:%M:%S) -----"
if [ $rc -ne 0 ]; then echo "CONTROL_FAIL rc=$rc" > "$L/STATUS"; exit 1; fi

echo "===== TRAIN sparse (+edge_head) $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_edgevec.py --edge_head $COMMON --outf "$ROOT/run_sparse" > "$L/train_sparse.log" 2>&1
rc=$?; echo "----- sparse rc=$rc $(date +%H:%M:%S) -----"
if [ $rc -ne 0 ]; then echo "SPARSE_FAIL rc=$rc" > "$L/STATUS"; exit 1; fi

CW=$ROOT/run_control/final_net_edgevec_control.pth
SW=$ROOT/run_sparse/final_net_edgevec_sparse.pth
if [ ! -f "$CW" ] || [ ! -f "$SW" ]; then echo "NO_CKPT" > "$L/STATUS"; exit 1; fi

echo "===== EVAL 3-arm (control/sparse-hm/sparse-refine) $(date +%H:%M:%S) ====="
for ES in filter-val manual synthetic; do
  run scripts/stage0/eval_harness/eval_edgevec_refine.py --control_weights "$CW" --sparse_weights "$SW" \
    --evalset "$ES" --tag full > "$L/eval_$ES.log" 2>&1
  echo "----- eval $ES rc=$? -----"
done
echo "ALL_DONE" > "$L/STATUS"; echo "===== ALL DONE $(date +%H:%M:%S) ====="
