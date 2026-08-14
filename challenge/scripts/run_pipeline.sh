#!/bin/bash
# challenge/scripts/run_pipeline.sh
#
# 사용자가 자리를 비울 때 한 번 실행해두면
#   1) 현재 manual GT 로 1차 ft (warmstart 모델)
#   2) 1차 모델로 pseudo-label 생성 (지정 시퀀스)
#   3) manual + pseudo 합쳐 2차 ft
#   4) 결과 weight + 로그 정리
#
# 사용:
#   bash challenge/scripts/run_pipeline.sh                # default settings
#   PSEUDO_SEQ=data/outside/capturepallet09 bash challenge/scripts/run_pipeline.sh
#   MANUAL_GT=challenge/data/01_real/manual_gt/capturepallet07_manual_gt bash challenge/scripts/run_pipeline.sh
#
# 환경:
#   conda env: pallet-pose 가 자동 활성화

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# ─ 설정 ────────────────────────────────────────────────────────────────────
MANUAL_GT="${MANUAL_GT:-challenge/data/01_real/manual_gt/capturepallet07_manual_gt}"
PSEUDO_SEQ="${PSEUDO_SEQ:-data/outside/capturepallet09}"
PSEUDO_OUT="${PSEUDO_OUT:-}"
BASELINE="${BASELINE:-challenge/weights/baseline_v8_A.pth}"
STAGE1_EXP="${STAGE1_EXP:-challenge_ft_stage1}"
STAGE2_EXP="${STAGE2_EXP:-challenge_ft_stage2}"
STAGE1_EP="${STAGE1_EP:-15}"        # manual 만으로 ft 라 짧게
STAGE2_EP="${STAGE2_EP:-15}"
STRIDE_PSEUDO="${STRIDE_PSEUDO:-3}" # pseudo 만들 때 frame stride
MAX_REPROJ_PSEUDO="${MAX_REPROJ_PSEUDO:-8}"  # 보수적

LOG_DIR="challenge/_docs/pipeline_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo " Challenge Pipeline — $(date)"
echo "=========================================="
echo " manual_gt    : $MANUAL_GT"
echo " baseline     : $BASELINE"
echo " pseudo_seq   : $PSEUDO_SEQ"
echo " stage1 exp   : $STAGE1_EXP  (epochs=$STAGE1_EP)"
echo " stage2 exp   : $STAGE2_EXP  (epochs=$STAGE2_EP)"
echo " log_dir      : $LOG_DIR"
echo "------------------------------------------"

# Conda 활성
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-pose
export PYTHONIOENCODING=utf-8

# manual GT 개수 확인
N_MANUAL=$(ls "$MANUAL_GT"/*.json 2>/dev/null | wc -l)
if [ "$N_MANUAL" -lt 5 ]; then
    echo "[ERROR] manual GT $N_MANUAL 개 < 5. 더 라벨링 후 다시 실행."
    exit 1
fi
echo "[OK] manual GT: $N_MANUAL frames"

# ─ STAGE 1: manual 만으로 1차 ft ──────────────────────────────────────────
echo ""
echo "### STAGE 1: 1차 finetune (manual 만) ###"
STAGE1_TRAIN="challenge/data/03_derived/_train_stage1"
python challenge/scripts/dataset/merge_dataset.py \
    --inputs "$MANUAL_GT" \
    --out "$STAGE1_TRAIN" \
    --val_fraction 0.2 \
    2>&1 | tee "$LOG_DIR/01_merge_stage1.log"

bash scripts/train_dope.sh --finetune \
    --net_path "$BASELINE" \
    --train_dir "$STAGE1_TRAIN/train" \
    --val_dir   "$STAGE1_TRAIN/val" \
    --exp_name "$STAGE1_EXP" \
    --symmetric_loss \
    --struct_loss --struct_coord 0.003 \
    2>&1 | tee "$LOG_DIR/02_ft_stage1.log" || {
        echo "[WARN] stage1 ft failed but continuing"
    }

STAGE1_LAST=$(ls weights/$STAGE1_EXP/final_net_epoch_*.pth 2>/dev/null | tail -1)
[ -z "$STAGE1_LAST" ] && STAGE1_LAST=$(ls weights/$STAGE1_EXP/net_epoch_*.pth 2>/dev/null | sort | tail -1)
if [ -z "$STAGE1_LAST" ] || [ ! -f "$STAGE1_LAST" ]; then
    echo "[ERROR] stage1 weight not found. abort."
    exit 1
fi
echo "[Stage1 weight] $STAGE1_LAST"

# ─ STAGE 2: pseudo-label 생성 ─────────────────────────────────────────────
echo ""
echo "### STAGE 2: pseudo-label 생성 (1차 모델로) ###"
SEQ_NAME=$(basename "$PSEUDO_SEQ")
PSEUDO_OUT="${PSEUDO_OUT:-challenge/data/01_real/pseudo_gt/${SEQ_NAME}_pseudo_gt}"
python challenge/scripts/dataset/make_pseudo_gt.py \
    --seq "$PSEUDO_SEQ" \
    --weights "$STAGE1_LAST" \
    --out_dir "$PSEUDO_OUT" \
    --stride "$STRIDE_PSEUDO" \
    --min_kp 6 --max_reproj "$MAX_REPROJ_PSEUDO" \
    2>&1 | tee "$LOG_DIR/03_pseudo.log"

N_PSEUDO=$(ls "$PSEUDO_OUT"/*.json 2>/dev/null | wc -l)
echo "[Pseudo GT] $N_PSEUDO frames generated"

# ─ STAGE 3: manual + pseudo 합쳐 2차 ft ───────────────────────────────────
echo ""
echo "### STAGE 3: 2차 finetune (manual + pseudo) ###"
STAGE2_TRAIN="challenge/data/03_derived/_train_stage2"

if [ "$N_PSEUDO" -gt 10 ]; then
    python challenge/scripts/dataset/merge_dataset.py \
        --inputs "$MANUAL_GT" "$PSEUDO_OUT" \
        --out "$STAGE2_TRAIN" \
        --val_fraction 0.15 \
        --manual_to_val_only \
        2>&1 | tee "$LOG_DIR/04_merge_stage2.log"

    bash scripts/train_dope.sh --finetune \
        --net_path "$STAGE1_LAST" \
        --train_dir "$STAGE2_TRAIN/train" \
        --val_dir   "$STAGE2_TRAIN/val" \
        --exp_name "$STAGE2_EXP" \
        --symmetric_loss \
    --struct_loss --struct_coord 0.003 \
        2>&1 | tee "$LOG_DIR/05_ft_stage2.log" || {
            echo "[WARN] stage2 ft failed"
        }
else
    echo "[SKIP] pseudo GT $N_PSEUDO 개 < 10 — stage2 건너뜀. stage1 결과를 최종으로 사용."
fi

# ─ 정리 ────────────────────────────────────────────────────────────────────
echo ""
echo "### 최종 가중치 → challenge/weights/finetuned/ ###"
mkdir -p challenge/weights/finetuned
for exp in "$STAGE1_EXP" "$STAGE2_EXP"; do
    SRC="weights/$exp"
    [ -d "$SRC" ] && cp -u "$SRC"/final_net_epoch_*.pth challenge/weights/finetuned/ 2>/dev/null || true
done

ls -la challenge/weights/finetuned/ | tee "$LOG_DIR/06_final_weights.log"

echo ""
echo "=========================================="
echo " Pipeline DONE — $(date)"
echo " Logs: $LOG_DIR"
echo " 검증: python challenge/scripts/live/run_live.py --seq $PSEUDO_SEQ --weights challenge/weights/finetuned/<file>.pth"
echo "=========================================="
