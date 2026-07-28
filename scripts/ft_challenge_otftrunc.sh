#!/bin/bash
# challenge DOPE fine-tune (augmentation 없음, real GT 원본만)
#   base = challenge_pretrain_otftrunc (v1/v2 + on-the-fly truncation, scratch 60ep)
#   ft   = real GT 251장 원본만 (truncation/crop augmentation 없음)
#          → pretrain에서 truncation 강건성 학습했으므로 ft는 real 과적합 도메인 적응만.
# 누적 epoch: pretrain 60 → ft +90 = 목표 150.
# env=pallet-pose. 사용법: bash scripts/ft_challenge_otftrunc.sh
set -e
source /home/minjae/anaconda3/etc/profile.d/conda.sh
conda activate pallet-pose
cd /home/minjae/Documents/github/pallet-pose

# real manual GT 251장 (빈 폴더 night01/03·pallet01·forklift_20260528 제외)
REAL_DIRS="\
challenge/data/capturepallet02_manual_gt \
challenge/data/capturepallet03_manual_gt \
challenge/data/capturepallet04_manual_gt \
challenge/data/capturepallet05_manual_gt \
challenge/data/capturepallet07_manual_gt \
challenge/data/capturepallet08_manual_gt \
challenge/data/capturepallet09_manual_gt \
challenge/data/capturepalletcad_manual_gt \
challenge/data/capturenight04_manual_gt \
challenge/data/capturenight05_manual_gt \
challenge/data/capturenight06_manual_gt \
challenge/data/capturenight07_manual_gt \
challenge/data/capturenight08_manual_gt \
challenge/data/capturenight09_manual_gt \
data/outside/forklift_raw_20260528_163408/gt_manual"

PRETRAIN_W="weights/challenge_pretrain_otftrunc/final_net_epoch_0060.pth"

echo "########## challenge ft: real GT 251장, augmentation 없음, 90ep (누적 150) ##########"
EPOCHS=150 bash scripts/train_dope.sh --finetune \
  --net_path "$PRETRAIN_W" \
  --exp_name challenge_ft_otftrunc \
  --train_dirs "$REAL_DIRS" \
  2>&1 | tee weights/challenge_ft_otftrunc_train.log

echo "########## 완료 — weights/challenge_ft_otftrunc/final_net_epoch_0150.pth ##########"
