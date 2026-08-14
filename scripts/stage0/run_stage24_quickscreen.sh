#!/usr/bin/env bash
# STAGE24 quick-screen gate: NEW data (v3 000-008 + addon + trunc) + real mask_rle.
# 3 arms from challenge0123 base: voting(unit+seg+rle), sparse(edge), control(heatmap).
# Gate = convergence(L_vec/L_edge) + seen improvement. NOT full — screen only.
# Smoke(2 iter) → quick train each arm. NaN aborts via trainer sys.exit(2).
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1

BASE=weights/challenge0123/final_net_epoch_0060.pth
ROOT=data/pallet/eval_results/stage24_vec_newdata
L=$ROOT/logs; mkdir -p "$L"
WV=weights/stage24_vec_newdata/voting
WS=weights/stage24_vec_newdata/sparse
WC=weights/stage24_vec_newdata/control

DATA="challenge/data/02_synthetic/training/v3/batch_000 challenge/data/02_synthetic/training/v3/batch_001 \
challenge/data/02_synthetic/training/v3/batch_002 challenge/data/02_synthetic/training/v3/batch_003 \
challenge/data/02_synthetic/training/v3/batch_004 challenge/data/02_synthetic/training/v3/batch_005 \
challenge/data/02_synthetic/training/v3/batch_006 challenge/data/02_synthetic/training/v3/batch_007 \
challenge/data/02_synthetic/training/v3/batch_008 challenge/data/02_synthetic/training/addon_v1_train \
challenge/data/02_synthetic/training/truncation_addon_v1"

SUB=3000; EP=4; BS=12
run () { conda run -n pallet-pose python "$@"; }

echo "===== SMOKE voting $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_pvnet.py --vec_mode unit --seg --pvnet_mask_rle \
  --weights $BASE --data $DATA --subsample 32 --epochs 1 --max_iters 2 \
  --batchsize $BS --workers 4 --lam_vec 0.1 --lam_seg 1.0 --warmup 200 \
  --log_every 1 --outf /tmp/s24_smoke_v > "$L/smoke_voting.log" 2>&1
[ $? -ne 0 ] && { echo "SMOKE_VOTING_FAIL"; echo SMOKE_VOTING_FAIL > "$L/STATUS"; exit 1; }
echo "----- smoke voting OK -----"

echo "===== SMOKE sparse $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_edgevec.py --edge_head \
  --weights $BASE --data $DATA --subsample 32 --epochs 1 --max_iters 2 \
  --batchsize $BS --workers 4 --lam_edgevec 0.1 --warmup 200 \
  --log_every 1 --outf /tmp/s24_smoke_s > "$L/smoke_sparse.log" 2>&1
[ $? -ne 0 ] && { echo "SMOKE_SPARSE_FAIL"; echo SMOKE_SPARSE_FAIL > "$L/STATUS"; exit 1; }
echo "----- smoke sparse OK -----"

echo "===== TRAIN voting (unit+seg+rle, sub$SUB ep$EP) $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_pvnet.py --vec_mode unit --seg --pvnet_mask_rle \
  --weights $BASE --data $DATA --subsample $SUB --epochs $EP \
  --batchsize $BS --workers 6 --lam_vec 0.1 --lam_seg 1.0 --warmup 200 \
  --log_every 25 --save_every 1 --outf "$WV" --namefile voting > "$L/train_voting.log" 2>&1
rc=$?; echo "----- voting rc=$rc $(date +%H:%M:%S) -----"
[ $rc -ne 0 ] && { echo "VOTING_FAIL rc=$rc" > "$L/STATUS"; exit 1; }

echo "===== TRAIN sparse (edge, sub$SUB ep$EP) $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_edgevec.py --edge_head \
  --weights $BASE --data $DATA --subsample $SUB --epochs $EP \
  --batchsize $BS --workers 6 --lam_edgevec 0.1 --warmup 200 \
  --log_every 25 --save_every 1 --outf "$WS" --namefile sparse > "$L/train_sparse.log" 2>&1
rc=$?; echo "----- sparse rc=$rc $(date +%H:%M:%S) -----"
[ $rc -ne 0 ] && { echo "SPARSE_FAIL rc=$rc" > "$L/STATUS"; exit 1; }

echo "===== TRAIN control (heatmap only, sub$SUB ep$EP) $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_pvnet.py --vec_mode off \
  --weights $BASE --data $DATA --subsample $SUB --epochs $EP \
  --batchsize $BS --workers 6 --warmup 200 \
  --log_every 25 --save_every 1 --outf "$WC" --namefile control > "$L/train_control.log" 2>&1
rc=$?; echo "----- control rc=$rc $(date +%H:%M:%S) -----"
[ $rc -ne 0 ] && { echo "CONTROL_FAIL rc=$rc" > "$L/STATUS"; exit 1; }

echo "QUICKSCREEN_DONE" > "$L/STATUS"
echo "===== QUICKSCREEN ALL DONE $(date +%H:%M:%S) ====="
