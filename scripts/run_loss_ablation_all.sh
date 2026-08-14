#!/bin/bash
# 2026-04-16 — CoordDOPE loss ablation 전체 조합 (coord × edge × flip)
# 기존: A=coord, B=edge, C=coord+edge, D=flip
# 새로 학습: F=coord+flip, G=edge+flip, H=coord+edge+flip
#
# 공통 설정 (기존 A/B/C/D 와 완전히 일치):
#   anchor  = weights/mixed_v8/final_net_epoch_0060.pth
#   data    = data/pallet/training_data/mixed_v8_train
#   epochs  = 65 (61→65, 5 epoch ft)
#   lr      = 5e-5
#   batch   = 4
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda env 의 python 을 쓴다. 다른 인터프리터가 필요하면 PY 로 덮어쓸 것.
PY="${PY:-python}"
ANCHOR="../../weights/mixed_v8/final_net_epoch_0060.pth"
DATA="../../data/pallet/training_data/mixed_v8_train"
EPOCHS=65
LR=5e-05
BATCH=4
IMSIZE=448
SIGMA=4.0

LOG_DIR="logs/ablation_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
echo "Log dir: $LOG_DIR"

run_train() {
    local tag="$1"
    local coord="$2"
    local edge="$3"
    local flip="$4"
    local out="../../weights/v8_ablation_${tag}"
    echo ""
    echo "=============================================="
    echo " $tag: coord=$coord edge=$edge flip=$flip"
    echo "=============================================="
    mkdir -p "weights/v8_ablation_${tag}"
    (cd Deep_Object_Pose/train && \
        "$PY" train.py \
            --data "$DATA" \
            --object pallet \
            --epochs $EPOCHS \
            --batchsize $BATCH \
            --imagesize $IMSIZE \
            --lr $LR \
            --save_every 5 \
            --outf "$out" \
            --loginterval 500 \
            --workers 0 \
            --sigma $SIGMA \
            --net_path "$ANCHOR" \
            --struct_loss \
            --struct_lambda 1.0 \
            --struct_warmup 0 \
            --struct_coord "$coord" \
            --struct_edge "$edge" \
            --struct_flip "$flip" \
            --struct_delta 0.03) 2>&1 | tee "$LOG_DIR/${tag}.log"
}

run_train F_coord_flip      0.003  0.0    0.02
run_train G_edge_flip       0.0    0.003  0.02
run_train H_all             0.003  0.002  0.02

echo ""
echo "[DONE] All 3 runs finished."
