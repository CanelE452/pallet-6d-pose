#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh; conda activate pallet-pose
declare -A PLDIR=( [combined]=paper_s2_pl_reproj_flip [outside]=paper_s2_plrf_outside [night]=paper_s2_plrf_night [noapril]=paper_s2_plrf_noapril )
for K in combined outside night noapril; do
  PL=${PLDIR[$K]}; OUT="/home/minjae/Documents/github/pallet-pose/weights/paper_s2/paper_s2_reproj_flip/r1_$K"; mkdir -p "$OUT"
  echo "=== TRAIN rf R1_$K (PL=$PL) ==="
  cd "/home/minjae/Documents/github/pallet-pose/Deep_Object_Pose/train"
  python -u train.py \
    --data "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/mixed_v8_train" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/v4_split_base" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_squash_v2" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_trunc_v2" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_scale_v2" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/paper_4pallet_mask_v1" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/$PL" \
    --object pallet --batchsize 12 --sigma 2.0 --imagesize 400 \
    --lr 1e-05 --epochs 60 --epoch_size 6000 --workers 6 --manualseed 42 \
    --save_every 1 --nb_checkpoints 0 --namefile epoch \
    --net_path "/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth" --outf "$OUT" \
    --balance_groups "mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:57,paper_4pallet_mask_v1:38,$PL:5" \
    --mask_aux --mask_weight 0.01 --mask_warmup 0 --aspect_resize \
    --encoder_lr_scale 0.1 --encoder_freeze_steps 500 --trainable_scope all
  echo "=== DONE rf R1_$K ==="
done
echo "ALL RF R1 DONE"
