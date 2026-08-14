#!/bin/bash
# Phase 1 — 한 모델을 3 도메인에서 6D pose metric (ADD/5cm5°) 평가.
#
# 사용:
#   bash scripts/data_prep/eval/eval_6d_3_domains.sh <weights> <tag>

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <weights_path> <tag>"
    exit 1
fi

WEIGHTS="$1"
TAG="$2"
PALLET_PY="${PALLET_PY:-python}"

OUT_DIR="data/pallet/eval_results/phase1_${TAG}_6d"
mkdir -p "$OUT_DIR"

echo "============================================"
echo " Phase 1 6D eval — model: $WEIGHTS"
echo " Tag: $TAG"
echo "============================================"

# D435i intrinsics (manual_gt 와 일치)
FX=605.91
FY=605.97
CX=317.60
CY=256.29

# --- Indoor ---
echo ""
echo "[1/3] Indoor (capture0403middle)"
"$PALLET_PY" scripts/data_prep/eval/evaluate_real.py \
    --weights "$WEIGHTS" \
    --test_dir data/pallet/raw_data/capture0403middle/gt_final_isaac \
    --image_dir data/pallet/raw_data/capture0403middle/rgb \
    --fx $FX --fy $FY --cx $CX --cy $CY \
    --output_dir "$OUT_DIR/indoor" \
    2>&1 | tee "$OUT_DIR/indoor.txt" | tail -50

# --- Outside ---
echo ""
echo "[2/3] Outside"
"$PALLET_PY" scripts/data_prep/eval/evaluate_real.py \
    --weights "$WEIGHTS" \
    --test_dir data/_eval_sets/outside_combined \
    --image_dir data/_eval_sets/outside_combined \
    --fx $FX --fy $FY --cx $CX --cy $CY \
    --output_dir "$OUT_DIR/outside" \
    2>&1 | tee "$OUT_DIR/outside.txt" | tail -50

# --- Night ---
echo ""
echo "[3/3] Night"
"$PALLET_PY" scripts/data_prep/eval/evaluate_real.py \
    --weights "$WEIGHTS" \
    --test_dir data/_eval_sets/night_combined \
    --image_dir data/_eval_sets/night_combined \
    --fx $FX --fy $FY --cx $CX --cy $CY \
    --output_dir "$OUT_DIR/night" \
    2>&1 | tee "$OUT_DIR/night.txt" | tail -50

echo ""
echo "============================================"
echo " 6D Eval done"
echo "============================================"
