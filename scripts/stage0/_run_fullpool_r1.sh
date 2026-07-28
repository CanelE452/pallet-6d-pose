#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh; conda activate pallet-pose
cd "/home/minjae/Documents/github/pallet-pose/Deep_Object_Pose/train"
python -u train.py \
  --data "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/mixed_v8_train" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/v4_split_base" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_squash_v2" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_trunc_v2" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/aug_scale_v2" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/paper_4pallet_mask_v1" "/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/paper_s2_fullpool_r1" \
  --object pallet --batchsize 12 --sigma 2.0 --imagesize 400 \
  --lr 1e-05 --epochs 60 --epoch_size 6000 --workers 6 --manualseed 42 \
  --save_every 1 --nb_checkpoints 0 --namefile epoch \
  --net_path "/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth" --outf "/home/minjae/Documents/github/pallet-pose/weights/paper_s2_fullpool_selftrain/r1" \
  --balance_groups 'mixed_v8_train|v4_split_base|aug_squash_v2|aug_trunc_v2|aug_scale_v2:57,paper_4pallet_mask_v1:38,paper_s2_fullpool_r1:5' \
  --mask_aux --mask_weight 0.01 --mask_warmup 0 --aspect_resize \
  --encoder_lr_scale 0.1 --encoder_freeze_steps 500 --trainable_scope all
