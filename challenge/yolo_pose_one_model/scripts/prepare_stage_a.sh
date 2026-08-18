#!/usr/bin/env bash
# Convert every G/T frame in the split manifests into padded YOLO-pose form.
# ~60k frames, roughly 48 GB. Safe to re-run: existing outputs are overwritten in place.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26

cd "$(git rev-parse --show-toplevel)"
P=challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py
W=${1:-10}

python $P --manifest manifests/generic_train.txt --out datasets/stage_a --split train --workers "$W"
python $P --manifest manifests/target_train.txt  --out datasets/stage_a --split train --workers "$W"
python $P --manifest manifests/generic_val.txt   --out datasets/stage_a --split val   --workers "$W"
python $P --manifest manifests/target_val.txt    --out datasets/stage_a --split val   --workers "$W"

echo "PREPARE_STAGE_A_DONE"
df -h . | tail -1
