# Pallet 6D Pose — Jetson 배포 패키지

YOLO26n-pose 팔레트 keypoint 추론 + SQPnP 6D pose + **inference FPS 측정**.
카메라 없이 **영상 파일(mp4)** 로도 FPS를 확인할 수 있습니다.

## 폴더 구성
```
pallet_jetson_deploy/
├── infer_fps.py      ← 추론 + FPS 측정 (자체완결, repo 의존 없음)
├── build_engine.sh   ← .pt → TensorRT engine 빌드 (Jetson에서 실행)
├── requirements.txt
├── models/
│   ├── pallet_pose_cropaug_v2.pt   ← 원본 (engine 빌드용)
│   ├── pallet_pose_640.onnx        ← portable (engine 직접 안 쓸 때)
│   └── pallet_pose_320.onnx        ← 저해상도(더 빠름)
└── data/
    ├── forklift_sample.mp4         ← 테스트 영상
    └── cam_K.txt                   ← 카메라 intrinsics (3x3)
```

## Jetson 셋업 (1회)
JetPack(= CUDA/cuDNN/TensorRT/torch 포함)이 깔린 Jetson 가정.
```bash
pip install -r requirements.txt          # ultralytics, opencv-python, numpy
# torch가 없다면 NVIDIA Jetson용 torch wheel을 별도 설치 (JetPack 버전에 맞게)
```

## 1) TensorRT engine 빌드 (Jetson에서, 1회)
> 엔진은 **빌드한 그 보드에서만** 동작합니다. 그래서 .pt/.onnx를 가져와 Jetson에서 빌드합니다.
```bash
bash build_engine.sh 640 fp16     # → models/pallet_pose_cropaug_v2.engine
# 더 빠르게: bash build_engine.sh 320 fp16
```

## 2) 추론 + FPS 측정
```bash
# 영상 파일로 (카메라 불필요) — 화면 표시 + 종료 시 FPS 요약
python infer_fps.py \
  --model models/pallet_pose_cropaug_v2.engine \
  --source data/forklift_sample.mp4 \
  --cam-k data/cam_K.txt \
  --imgsz 640 --conf 0.5 --show

# 화면 없이 FPS만 측정 (--show 빼면 headless, 순수 성능 측정)
python infer_fps.py --model models/pallet_pose_cropaug_v2.engine \
  --source data/forklift_sample.mp4 --cam-k data/cam_K.txt --imgsz 640

# engine 없이 onnx로 (느림, 빌드 전 동작확인용)
python infer_fps.py --model models/pallet_pose_640.onnx \
  --source data/forklift_sample.mp4 --cam-k data/cam_K.txt

# 실시간 카메라
python infer_fps.py --model models/pallet_pose_cropaug_v2.engine \
  --source 0 --cam-k data/cam_K.txt --show
```

### 주요 옵션
| 옵션 | 설명 | 기본 |
|---|---|---|
| `--model` | .pt / .onnx / .engine | (필수) |
| `--source` | mp4 / 이미지폴더 / 카메라id(정수) | sample mp4 |
| `--cam-k` | 카메라 K (3x3 txt) | 내장 fallback |
| `--imgsz` | 입력 크기 (640/320) | 640 |
| `--conf` | detection threshold (FP↑면 올리기) | 0.4 |
| `--kp-conf` | PnP용 keypoint conf | 0.5 |
| `--show` | 실시간 창 표시 | off(headless) |
| `--save` | 결과 mp4 저장 경로 | 미저장 |
| `--max-frames` | N프레임만 (0=전체) | 0 |

종료 시 **inference-only FPS / full-pipeline FPS / ms breakdown / 검출률 / PnP 성공률**이 표로 출력됩니다.

## 참고 (성능)
- **RTX 3080 실측**(참고): TRT FP16@640 ~360 FPS, @320 ~604 FPS.
- **Jetson 예상**: 구형 Nano는 TRT FP16 + imgsz320로 ~10~15 FPS 수준(보드/전력모드에 따라 다름), Orin Nano는 15~30+ FPS. `nvpmodel -m 0` + `jetson_clocks`로 최대 클럭 설정 권장.
- engine 실행 시 라이브러리 경로 문제가 나면(예: `libcudnn`/`libnvinfer` not found), 해당 lib 경로를 `LD_LIBRARY_PATH`에 추가하세요.

## 모델 정보
- keypoint 9점: 0~3 near face, 4~7 far face, 8 centroid (camera-facing convention)
- 팔레트 치수: 1.10 × 1.30 × 0.11 m
- 추론 시 100px reflect pad 적용(truncation 강건) → 코드에 내장(`--pad`)
- PnP: SQPnP (`cv2.SOLVEPNP_SQPNP`) + RefineLM
