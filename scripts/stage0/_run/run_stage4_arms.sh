#!/usr/bin/env bash
# STEP 4 — 3-arm 짧은 finetune(control/offset/unit) + 평가 순차 실행.
# 백그라운드 실행용. 각 단계 로그를 stage4_pvnet/logs/ 에 남긴다.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1

LOG=data/pallet/eval_results/stage4_pvnet/logs
mkdir -p "$LOG" weights/stage4_pvnet

BS=${BS:-16}
SUB=${SUB:-6000}
EP=${EP:-8}
COMMON="--subsample $SUB --epochs $EP --batchsize $BS --workers 6 --lr 1e-4 \
  --lam_vec 0.5 --lam_diag 0.013 --lam_edge 0.0003 --warmup 400 \
  --log_every 50 --save_every 2"

run_arm () {
  local mode=$1
  echo "===== finetune $mode (bs=$BS sub=$SUB ep=$EP) $(date +%H:%M:%S) ====="
  conda run -n pallet-pose python scripts/stage0/finetune_pvnet.py \
    --vec_mode "$mode" $COMMON --outf "weights/stage4_pvnet/$mode" \
    > "$LOG/train_$mode.log" 2>&1
  echo "----- $mode done rc=$? $(date +%H:%M:%S) -----"
}

run_arm off
run_arm offset
run_arm unit

echo "===== eval (manual / filter-val / synthetic) $(date +%H:%M:%S) ====="
W="weights/stage4_pvnet/off/final_net_pvnet_off.pth \
   weights/stage4_pvnet/offset/final_net_pvnet_offset.pth \
   weights/stage4_pvnet/unit/final_net_pvnet_unit.pth"
for es in manual filter-val synthetic; do
  extra=""; [ "$es" = synthetic ] && extra="--n_frames 200"
  conda run -n pallet-pose python scripts/stage0/eval_harness/eval_pvnet_heads.py \
    --weights $W --evalset "$es" $extra --tag full \
    > "$LOG/eval_$es.log" 2>&1
  echo "----- eval $es done rc=$? -----"
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
