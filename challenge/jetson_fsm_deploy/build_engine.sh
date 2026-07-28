#!/bin/bash
# Jetson에서 .pt → TensorRT engine 빌드 (엔진은 빌드한 그 보드에서만 동작).
# 사용: bash build_engine.sh [imgsz] [precision]
#   imgsz     : 640(기본) 또는 320
#   precision : fp16(기본) 또는 int8
# 예) bash build_engine.sh 320 fp16
#     bash build_engine.sh 640 int8     (int8은 calibration 데이터 필요 — 아래 DATA 참고)
set -e
IMGSZ=${1:-640}
PREC=${2:-fp16}
HERE="$(cd "$(dirname "$0")" && pwd)"
PT="$HERE/models/pallet_pose_cropaug_v2.pt"

ARGS="model=$PT format=engine imgsz=$IMGSZ device=0"
if [ "$PREC" = "int8" ]; then
  # INT8은 calibration용 data yaml 필요. Jetson에 이미지 몇 장 두고 yaml 경로 지정하거나,
  # 간단히 fp16 사용 권장. 아래는 예시(data 경로는 본인 환경에 맞게).
  ARGS="$ARGS int8=True data=$HERE/calib.yaml"
else
  ARGS="$ARGS half=True"
fi

echo "[build] yolo export $ARGS"
yolo export $ARGS
echo "[done] engine: $HERE/models/pallet_pose_cropaug_v2.engine"
echo "  → infer_fps.py --model 로 이 .engine 을 지정해 추론하세요."
