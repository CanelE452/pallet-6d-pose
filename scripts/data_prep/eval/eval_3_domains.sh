#!/bin/bash
# Phase 1 — 한 모델을 3 도메인 (indoor / outside / night) 에서 평가.
#
# 사용:
#   bash scripts/data_prep/eval/eval_3_domains.sh <weights> <tag>
#
# Output:
#   data/pallet/eval_results/phase1_<tag>/{indoor,outside,night}.txt

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <weights_path> <tag>"
    echo "Example: $0 weights/v8_ablation_A_coord/final_net_epoch_0065.pth R0"
    exit 1
fi

WEIGHTS="$1"
TAG="$2"
PALLET_PY="${PALLET_PY:-python}"

OUT_DIR="data/pallet/eval_results/phase1_$TAG"
mkdir -p "$OUT_DIR"

echo "============================================"
echo " Phase 1 eval — model: $WEIGHTS"
echo " Tag: $TAG"
echo " Output: $OUT_DIR"
echo "============================================"

# --- Indoor (F5 평가 셋) ---
echo ""
echo "[1/3] Indoor (capture0403middle)"
"$PALLET_PY" scripts/data_prep/eval/eval_nn_matching.py \
    --weights "$WEIGHTS" \
    --test_dir data/pallet/raw_data/capture0403middle \
    --gt_dir data/pallet/raw_data/capture0403middle/gt_final_isaac \
    2>&1 | tee "$OUT_DIR/indoor.txt"

# --- Outside (manual_gt 통합) ---
echo ""
echo "[2/3] Outside (manual_gt combined, 129 frame)"
"$PALLET_PY" scripts/data_prep/eval/eval_nn_matching.py \
    --weights "$WEIGHTS" \
    --test_dir data/_eval_sets/outside_combined \
    --gt_dir data/_eval_sets/outside_combined \
    2>&1 | tee "$OUT_DIR/outside.txt"

# --- Night (manual_gt 통합) ---
echo ""
echo "[3/3] Night (manual_gt combined, 90 frame)"
"$PALLET_PY" scripts/data_prep/eval/eval_nn_matching.py \
    --weights "$WEIGHTS" \
    --test_dir data/_eval_sets/night_combined \
    --gt_dir data/_eval_sets/night_combined \
    2>&1 | tee "$OUT_DIR/night.txt"

echo ""
echo "============================================"
echo " Eval done. Summary:"
for d in indoor outside night; do
    NN20=$(grep "<20px:" "$OUT_DIR/$d.txt" | tail -1 | awk '{print $2}')
    echo "  $d NN<20px: $NN20"
done
echo "============================================"
