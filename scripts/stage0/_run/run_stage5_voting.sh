#!/usr/bin/env bash
# STEP5 (A) voting 본판 — 밤샘 무인 실행.
# 스모크 게이트 → 본학습(unit+seg 전체데이터 긴학습, λ_vec 0.1 ramp) →
# voting 평가(seg 마스크) → ALL DONE. NaN/실패 시 즉시 중단(GPU 낭비 방지).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1

L=data/pallet/eval_results/stage5_voting/logs
mkdir -p "$L" weights/stage_screens/stage5_voting
OUT=weights/stage_screens/stage5_voting/unit_seg
OFF=weights/stage_screens/stage4_pvnet/off/final_net_pvnet_off.pth   # STEP4 control 참조군
NEW=$OUT/final_net_pvnet_unit.pth

run () { conda run -n pallet-pose python "$@"; }

echo "===== SMOKE $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_pvnet.py --vec_mode unit --seg \
  --subsample 32 --epochs 1 --max_iters 2 --batchsize 16 --workers 4 \
  --lam_vec 0.1 --lam_seg 1.0 --lam_diag 0.013 --lam_edge 0.0003 --warmup 2000 \
  --log_every 1 --save_every 1 --outf /tmp/stage5_smoke > "$L/smoke.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "SMOKE_FAIL rc=$rc"; echo "SMOKE_FAIL" > "$L/STATUS"; exit 1; fi
echo "----- smoke OK $(date +%H:%M:%S) -----"

echo "===== TRAIN unit+seg (full data, 10ep, bs16, lam_vec0.1 ramp2000) $(date +%H:%M:%S) ====="
run scripts/stage0/finetune_pvnet.py --vec_mode unit --seg \
  --subsample 0 --epochs 10 --batchsize 16 --workers 6 \
  --lam_vec 0.1 --lam_seg 1.0 --lam_diag 0.013 --lam_edge 0.0003 --warmup 2000 \
  --log_every 50 --save_every 1 --outf "$OUT" > "$L/train.log" 2>&1
rc=$?
echo "----- train rc=$rc $(date +%H:%M:%S) -----"
if [ $rc -ne 0 ]; then echo "TRAIN_FAIL rc=$rc (2=NaN abort)"; echo "TRAIN_FAIL rc=$rc" > "$L/STATUS"; exit 1; fi
if [ ! -f "$NEW" ]; then echo "NO_FINAL_CKPT"; echo "NO_FINAL_CKPT" > "$L/STATUS"; exit 1; fi

echo "===== EVAL voting (seg mask) vs control $(date +%H:%M:%S) ====="
for es in manual filter-val synthetic; do
  extra=""; [ "$es" = synthetic ] && extra="--n_frames 200"
  run scripts/stage0/eval_harness/eval_pvnet_heads.py --weights "$OFF" "$NEW" \
    --evalset "$es" --use_seg_mask $extra --tag stage5 > "$L/eval_$es.log" 2>&1
  echo "----- eval $es rc=$? -----"
done
echo "ALL_DONE" > "$L/STATUS"
echo "===== ALL DONE $(date +%H:%M:%S) ====="
