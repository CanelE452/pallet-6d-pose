#!/usr/bin/env bash
# Build the TensorRT FP16 engine ON THE JETSON ORIN NANO.
#
# Do not run this on the RTX 3080 and copy the .engine over: a TensorRT engine is built
# for one GPU architecture and one TensorRT version, and will not load on the Jetson.
# Copy the .pt across, then run this script there.
set -euo pipefail

MODEL="challenge/yolo_pose_one_model/final/pallet_yolo26n_pose_640_b32_final.pt"
test -f "$MODEL"

yolo export \
  model="$MODEL" \
  format=engine \
  imgsz=640 \
  batch=1 \
  half=True \
  dynamic=False \
  device=0

SRC="${MODEL%.pt}.engine"
DST="challenge/yolo_pose_one_model/final/pallet_yolo26n_pose_640_b32_fp16.engine"
mv "$SRC" "$DST"
echo "engine: $DST"

# Runtime contract for the engine:
#   camera 640x480 -> 100 px reflect pad -> 840x680 -> YOLO imgsz 640 -> TensorRT FP16, batch 1
#   subtract 100 from predicted keypoints before PnP, and use the ORIGINAL 640x480 K.
