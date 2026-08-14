#!/bin/bash
# DOPE crop-aug 2단계 fine-tune
#   Stage 1: real 원본 GT 251장만, 높은 epoch (90)  — clean pose 정밀도 확보
#   Stage 2: real 원본 + crop 증강(485), 낮은 epoch (30) — truncation 강건성 보강
# base = crop-aug pretrain (mixed_v8 + synth crop, scratch 60ep)
#
# 주의: train.py는 net_path의 epoch에서 이어받고 EPOCHS=누적 목표로 해석.
#   pretrain 끝=60 → stage1 +90 = 목표 150 → stage2 +30 = 목표 180.
# env=pallet-pose. 사용법: bash scripts/ft_dope_cropaug.sh
set -e
source /home/minjae/anaconda3/etc/profile.d/conda.sh
conda activate pallet-pose
cd /home/minjae/Documents/github/pallet-pose

# real manual GT 251장 (빈 폴더 night01/03·pallet01·forklift_20260528 제외)
REAL_DIRS="\
challenge/data/01_real/manual_gt/capturepallet02_manual_gt \
challenge/data/01_real/manual_gt/capturepallet03_manual_gt \
challenge/data/01_real/manual_gt/capturepallet04_manual_gt \
challenge/data/01_real/manual_gt/capturepallet05_manual_gt \
challenge/data/01_real/manual_gt/capturepallet07_manual_gt \
challenge/data/01_real/manual_gt/capturepallet08_manual_gt \
challenge/data/01_real/manual_gt/capturepallet09_manual_gt \
challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt \
challenge/data/01_real/manual_gt/capturenight04_manual_gt \
challenge/data/01_real/manual_gt/capturenight05_manual_gt \
challenge/data/01_real/manual_gt/capturenight06_manual_gt \
challenge/data/01_real/manual_gt/capturenight07_manual_gt \
challenge/data/01_real/manual_gt/capturenight08_manual_gt \
challenge/data/01_real/manual_gt/capturenight09_manual_gt \
data/outside/forklift_raw_20260528_163408/gt_manual"

PRETRAIN_W="weights/dope_cropaug_pretrain/final_net_epoch_0060.pth"
S1_W="weights/dope_cropaug_ft_s1/final_net_epoch_0150.pth"

echo "########## Stage 1: real 원본 251장, 90ep ft (목표 누적 150) ##########"
EPOCHS=150 bash scripts/train_dope.sh --finetune \
  --net_path "$PRETRAIN_W" \
  --exp_name dope_cropaug_ft_s1 \
  --train_dirs "$REAL_DIRS" \
  2>&1 | tee weights/dope_cropaug_ft_s1_train.log

echo "########## Stage 2: real 원본 + crop, 30ep ft (목표 누적 180) ##########"
EPOCHS=180 bash scripts/train_dope.sh --finetune \
  --net_path "$S1_W" \
  --exp_name dope_cropaug_ft_s2 \
  --train_dirs "$REAL_DIRS challenge/data/03_derived/truncation_crops_dope/ft_real" \
  2>&1 | tee weights/dope_cropaug_ft_s2_train.log

echo "########## 완료 — best = weights/dope_cropaug_ft_s2/final_net_epoch_0180.pth ##########"
