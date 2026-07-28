# Jetson FSM 배포 — 팔레트 6D pose → 리프터 포크 정렬/삽입

Jetson에서 **FSM(정렬 상태머신) + 6D pose 추론(DOPE 또는 YOLO) + RealSense depth fusion + CAN 리프터 제어**를 돌리기 위한 셋업/실행 가이드.

> ⚠️ 이 폴더는 **번들이 아니라 셋업 가이드**입니다. 코드는 repo(`25y_automatic_lifter-master/.../depth_cam/`)에 있으므로, **Jetson에 repo를 git clone** 한 뒤 아래 절차를 따릅니다. (DOPE 가중치 + Deep_Object_Pose 모듈이 커서 USB 번들 대신 clone 방식이 현실적)

## 0. 실행 위치
```
<repo>/25y_automatic_lifter-master/25y_automatic_lifter-master/depth_cam/
python main_rec.py        # 여기서 실행
```

## 1. Jetson 환경 셋업 (한 번)

### (a) JetPack — torch/cuDNN/TensorRT 포함
Jetson에 맞는 JetPack 설치(보통 기본 OS). 그 위에:
```bash
# torch: pip 토치 ❌ → NVIDIA Jetson용 wheel 설치 (JetPack 버전에 맞게)
#   https://forums.developer.nvidia.com/t/pytorch-for-jetson  참고
# torchvision 도 Jetson 빌드/wheel 필요
pip install ultralytics opencv-python numpy pyrr
```

### (b) RealSense (librealsense + pyrealsense2)
pip `pyrealsense2` 는 ARM wheel이 없어서 **소스 빌드 필요**:
```bash
# librealsense 소스 빌드 (CUDA + python binding)
git clone https://github.com/IntelRealSense/librealsense
cd librealsense && mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_CUDA=ON
make -j4 && sudo make install
# PYTHONPATH 에 pyrealsense2 설치 경로 추가
```

### (c) CAN (리프터 제어)
현재 코드는 **Kvaser canlib** 기반(`calib/control.py`). Jetson(ARM)에서 Kvaser가 안 되면 **SocketCAN으로 포팅** 필요 → `CAN_PORTING.md` 참고. canlib 없으면 자동 **MOCK**(FSM 시각화만, 리프터 안 움직임).

## 2. 모델 준비

### YOLO backend (가벼움, Jetson 권장)
```bash
# .pt → TensorRT engine (Jetson에서 빌드, 그 보드 전용)
bash jetson_fsm_deploy/build_engine.sh 640 fp16
# → pallet_jetson_deploy/models/pallet_pose_cropaug_v2.engine
export MODEL_PATH_6D_YOLO=$(pwd)/pallet_jetson_deploy/models/pallet_pose_cropaug_v2.engine
```
### DOPE backend
`challenge/model/challengenight.pth` 가 repo에 있어야 함 (VGG-19, Jetson Nano엔 무거움 — YOLO 권장).

## 3. 실행 (preflight → run)
```bash
# (1) 환경 점검 — 무엇이 빠졌는지 먼저 확인
python jetson_fsm_deploy/preflight.py

# (2) 실행 — backend / 게이트 프로파일 선택
#   POSE_BACKEND : dope(기본) | yolo
#   GATE_PROFILE : demo(기본, 게이트 느슨) | real(실삽입용 강화)
POSE_BACKEND=yolo GATE_PROFILE=real python main_rec.py
```

## 4. 실삽입 전 필수 체크 (안전)
```
□ GATE_PROFILE=real         — 나쁜 프레임 거부 (demo는 게이트 거의 off)
□ pallet 실측 dims          — config PALLET_WIDTH/DEPTH/HEIGHT_M 를 줄자 실측값으로
                              (PnP scale·삽입 기하의 기준. 현재 1.10×1.30×0.12 가정)
□ depth fusion 동작 확인     — RealSense depth 유효(센서 연결, z_max 12m 내). 거리=depth 기반
□ CAM_TO_FORK_T/RPY 실측     — config 의 카메라→포크 extrinsic (현재 0=항등, 실측 필요)
□ CAN 실제 송신 확인         — is_mock()==False (MOCK이면 리프터 안 움직임)
□ 첫 시험은 빈 공간/저속      — 포크가 사람/구조물 없는 데서 정렬되는지 먼저
```

## 핵심 동작 (이미 구현/검증됨)
- 6D pose: DOPE/YOLO 둘 다 `infer_pose()`→`pose6d_to_align_vars()`→FSM (backend 무관 동일 경로)
- **거리/횡 = RealSense depth로 monocular PnP scale 보정** (monocular 단독은 ~0.5m 오차 → 삽입 실패). `pose6d_adapter.depth_scale_correct`
- 카메라→포크 extrinsic 변환 골격 (`apply_cam_to_fork`, 값 실측 후 config)
- camera intrinsics: RealSense에서 자동 read
- FSM: snapshot 방식(STOP_SEC=1.2)이라 낮은 FPS 허용

## 성능 (참고)
- YOLO26n TRT FP16: 3080 ~360fps. Jetson Nano(구형)는 imgsz320+INT8로 ~10~15fps 예상, Orin Nano는 30+.
- DOPE(VGG-19)는 Jetson Nano에 무거움 → **YOLO backend 권장**.
