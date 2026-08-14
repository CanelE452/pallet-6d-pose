#!/bin/bash
# B(비패딩) YOLO26n-pose fine-tune — A(yolo26n_pose_v1_ft_manual, padded)와 1:1 비교
#
# A와 동일 (A 학습 로그에서 추출):
#   base model = yolo26n_pose_v1/best.pt (합성 pretrain)
#   optimizer=SGD  lr0=1e-4  lrf=0.01  epochs=60  imgsz=640  batch=16
#   patience=30  momentum=0.937  weight_decay=0.0005
#   aug: mosaic=1.0 close_mosaic=10 scale=0.5 translate=0.1 fliplr=0.5
#        hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 erasing=0.4 degrees=0 perspective=0 mixup=0
#   val = yolo_pose/images/val (합성 holdout)
# 유일한 차이:
#   data = challenge/yolo_pose/data_manual.yaml  (비패딩 manual GT)
#   name = yolo26n_pose_v1_ft_manual_nopad
#
# env = pallet-yolo26 (yolo26 지원 ultralytics 신버전. 기존 pallet-pose 8.0.120 보존)

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26
cd /home/minjae/Documents/github/pallet-pose

yolo pose train \
  data=challenge/yolo_pose/data_manual.yaml \
  model=challenge/weights/yolo26n_pose_v1/weights/best.pt \
  optimizer=SGD lr0=0.0001 lrf=0.01 epochs=60 imgsz=640 batch=16 \
  patience=30 momentum=0.937 weight_decay=0.0005 \
  mosaic=1.0 close_mosaic=10 scale=0.5 translate=0.1 fliplr=0.5 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 erasing=0.4 degrees=0.0 perspective=0.0 mixup=0.0 \
  project=challenge/weights name=yolo26n_pose_v1_ft_manual_nopad \
  device=0 workers=4 save_period=10 exist_ok=True plots=True \
  2>&1 | tee challenge/weights/yolo26n_pose_v1_ft_manual_nopad_train.log
