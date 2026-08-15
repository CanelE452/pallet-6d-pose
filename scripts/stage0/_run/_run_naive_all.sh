#!/usr/bin/env bash
# Naïve ST(무필터) 체인: outside → night → noapril.
# h8 loo+flip 과 base/anchor/pool/rounds 전부 동일, filter_type=none 만 다름 (paired 비교).
set -e
cd /home/minjae/Documents/github/pallet-pose
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pallet-pose
export CUDA_MODULE_LOADING=LAZY PYTHONUNBUFFERED=1
BASE=weights/paper_s2_stageB/net_epoch_0057_noseg.pth
SYN=data/pallet/training_data/aug_squash_v2
RS=data/pallet/results/ralph/ralph_selftrain

run() {  # $1=name $2=real_dir
  local out=$RS/h9_s2_${1}_naive
  mkdir -p "$out"
  echo "=== START $1 $(date +%H:%M) ==="
  python -u scripts/self_training/self_train.py --config config/stage3_selftrain.yaml \
    --pretrained $BASE --synthetic_dir $SYN --real_dir $2 \
    --output_dir "$out" --num_rounds 2 --epochs_per_round 3 --filter_type none \
    > "$out/train.log" 2>&1
  echo "=== DONE $1 $(date +%H:%M) R1/R2: $(grep -E 'Round  [12]:' "$out/train.log" | tail -2 | tr '\n' ' ') ==="
}
run outside data/pallet/real_unlabeled_ralph_outside
run night   data/pallet/real_unlabeled_ralph_night
run noapril data/pallet/real_unlabeled_ralph_noapril
echo "=== ALL naive DONE $(date +%H:%M) ==="
