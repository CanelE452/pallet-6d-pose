#!/bin/bash
# 2026-04-16 — 7 ablation 조합 (A~D + F/G/H) 전부 NN matching 재평가
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda env 의 python 을 쓴다. 다른 인터프리터가 필요하면 PY 로 덮어쓸 것.
PY="${PY:-python}"
TEST_DIR="data/pallet/raw_data/capture0403middle"
OUT="data/pallet/eval_results/loss_ablation_all_nn.txt"

"$PY" scripts/data_prep/eval/eval_nn_matching.py \
    --weights \
        weights/mixed_v8/final_net_epoch_0060.pth \
        weights/v8_ablation_A_coord/final_net_epoch_0065.pth \
        weights/v8_ablation_B_edge/final_net_epoch_0065.pth \
        weights/v8_ablation_C_coord_edge/final_net_epoch_0065.pth \
        weights/v8_ablation_D_flip/final_net_epoch_0065.pth \
        weights/v8_ablation_F_coord_flip/final_net_epoch_0065.pth \
        weights/v8_ablation_G_edge_flip/final_net_epoch_0065.pth \
        weights/v8_ablation_H_all/final_net_epoch_0065.pth \
    --test_dir "$TEST_DIR" \
    2>&1 | tee "$OUT"

echo ""
echo "[DONE] Results saved to $OUT"
