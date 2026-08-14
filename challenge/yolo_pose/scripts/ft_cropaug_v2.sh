#!/bin/bash
# — A(padding) 설정 그대로, epoch만 충분히(120) + best.pt 채택
# data: data_cropaug_v2_pad.yaml (train 1007 = 원본177 + forklift32 + real crop404 + synth crop394, val 42=A와 동일)
# base/하이퍼파라미터는 ft_holdout_ab.sh의 A와 100% 동일, epochs 60→120 / patience 30→40 만 변경.
# env=pallet-yolo26 (ultralytics 8.4.60). 출력: challenge/weights/yolo26n_pose_v1_ft_cropaug_v2/

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pallet-yolo26
cd /home/minjae/Documents/github/pallet-pose

COMMON="model=challenge/weights/yolo26n_pose_v1/weights/best.pt \
optimizer=SGD lr0=0.0001 lrf=0.01 epochs=120 imgsz=640 batch=16 \
patience=40 momentum=0.937 weight_decay=0.0005 \
mosaic=1.0 close_mosaic=10 scale=0.5 translate=0.1 fliplr=0.5 \
hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 erasing=0.4 degrees=0.0 perspective=0.0 mixup=0.0 \
project=challenge/weights device=0 workers=4 save_period=10 exist_ok=True plots=True"

echo "########## ##########"
yolo pose train data=challenge/yolo_pose/data_cropaug_v2_pad.yaml \
  name=yolo26n_pose_v1_ft_cropaug_v2 $COMMON \
  2>&1 | tee challenge/weights/ft_cropaug_v2.log

echo "########## 완료 — best.pt = challenge/weights/yolo26n_pose_v1_ft_cropaug_v2/weights/best.pt ##########"
