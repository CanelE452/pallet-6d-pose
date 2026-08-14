#!/bin/bash
# A(padding) vs B(비패딩) holdout 재학습 — leakage 없는 엄밀 비교용
# frame-level 80/20 split (seed42, stratified). train 177 / holdout 42 (A·B 동일 frame).
# base/하이퍼파라미터 100% 동일. 차이 = data(padded vs nopad)뿐.
# env=pallet-yolo26 (ultralytics 8.4.60). 출력: runs/pose/challenge/weights/<name>/

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26
cd /home/minjae/Documents/github/pallet-pose

COMMON="model=challenge/weights/yolo26n_pose_v1/weights/best.pt \
optimizer=SGD lr0=0.0001 lrf=0.01 epochs=60 imgsz=640 batch=16 \
patience=30 momentum=0.937 weight_decay=0.0005 \
mosaic=1.0 close_mosaic=10 scale=0.5 translate=0.1 fliplr=0.5 \
hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 erasing=0.4 degrees=0.0 perspective=0.0 mixup=0.0 \
project=challenge/weights device=0 workers=4 save_period=10 exist_ok=True plots=True"

echo "########## A (padding) holdout 재학습 ##########"
yolo pose train data=challenge/yolo_pose/data_manual_pad_ho.yaml \
  name=yolo26n_pose_v1_ft_pad_ho $COMMON \
  2>&1 | tee challenge/weights/ft_pad_ho.log

echo "########## B (비패딩) holdout 재학습 ##########"
yolo pose train data=challenge/yolo_pose/data_manual_nopad_ho.yaml \
  name=yolo26n_pose_v1_ft_nopad_ho $COMMON \
  2>&1 | tee challenge/weights/ft_nopad_ho.log

echo "########## 둘 다 완료 ##########"
