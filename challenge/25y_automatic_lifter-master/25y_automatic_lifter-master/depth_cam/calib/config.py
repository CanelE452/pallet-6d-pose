# calib/config.py
# 공통 설정 / 상수 / 색상 팔레트 (FSM 다이어그램 및 main_rec.py 실행 흐름 기준)

from pathlib import Path as _Path
import os as _os

# config.py 가 위치한 디렉토리부터 repo root 까지 자동 계산 — 어느 PC 에서나 동작.
# parents: [0]=calib, [1]=depth_cam, [2]=25y_automatic_lifter-master(inner),
#          [3]=25y_automatic_lifter-master(outer), [4]=FoundationPose (repo root)
_CONFIG_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _CONFIG_DIR.parents[3]

# ===== 모델 & 감지 =====
MODEL_PATH = 'runs/segment/y11n_seg_finetune/weights/last.pt'  # 실제 가중치 경로에 맞춰 확인 필요
FRONT_CLASS_NAME = "front"
CONF_THR = 0.30  # 세그멘테이션/디텍션 confidence threshold

# ===== 6D pose (DOPE) 모드 =====
# True 면 main_rec.py 가 RealSense color 프레임을 DOPE 추론기에 넣어
# 9 keypoint → PnP → (ψ_pallet, d_lateral, d_forward) 를 계산하고
# fsm.step 의 6D 입력 kwargs (psi_pallet_deg, d_lateral_m, d_forward_m) 로 전달한다.
# False 면 기존 RGB-D RANSAC perception 만 사용.
USE_6D_MODE = True
# 6D pose backend 선택: "dope" (기본) 또는 "yolo" (YOLO26-pose + SQPnP).
# 둘 다 동일한 infer_pose() → pose6d_to_align_vars() 경로를 쓰므로 FSM 은 동일하게 동작.
# 환경변수 POSE_BACKEND 로 override 가능.
POSE_BACKEND = _os.environ.get("POSE_BACKEND", "dope").lower()
# YOLO backend 가중치 (.pt / .onnx / .engine). 환경변수 MODEL_PATH_6D_YOLO 로 override.
# .engine 사용 시 LD_LIBRARY_PATH 에 torch/lib + tensorrt_libs 추가 필요.
MODEL_PATH_6D_YOLO = _os.environ.get(
    "MODEL_PATH_6D_YOLO",
    str(_REPO_ROOT / "pallet_jetson_deploy" / "models" / "pallet_pose_cropaug_v2.pt"),
)
# True 면 YOLO segmentation (perception_yolo) + RGB-D RANSAC plane fit 도 추가로 동작
# (DOPE 검출 실패 시 fallback 또는 시각화 비교 용도). False (default) 면 DOPE 만.
# DOPE 만 쓰는 환경에선 False — YOLO 모델 로드 시간/메모리 절약 + HUD 가 깔끔.
USE_PERCEPTION_YOLO = False
# DOPE 가중치 경로 — challenge/model/ 안의 .pth 를 파일명 상관없이 자동 탐색.
# 환경변수 MODEL_PATH_6D 로 override 가능 (HuggingFace 등 다른 위치 사용 시).
def _resolve_model_path_6d():
    """challenge/model/ 폴더의 .pth 를 이름 상관없이 자동 선택.
    우선순위: ① 환경변수 MODEL_PATH_6D → ② 폴더 내 유일 .pth →
              ③ 여러 개면 challengenight.pth 우선(기존 호환), 없으면 이름순 첫 .pth."""
    env = _os.environ.get("MODEL_PATH_6D")
    if env:
        return env
    model_dir = _REPO_ROOT / "challenge" / "model"
    pths = sorted(model_dir.glob("*.pth")) if model_dir.is_dir() else []
    if not pths:
        # 폴더 비었으면 기존 기본 경로 반환 (런타임 load 시 명확한 에러)
        return str(model_dir / "challengenight.pth")
    if len(pths) == 1:
        return str(pths[0])
    for p in pths:
        if p.name == "challengenight.pth":
            return str(p)
    return str(pths[0])

MODEL_PATH_6D = _resolve_model_path_6d()
# DOPE 입력 height (run_live.py 와 동일하게 400). VGG stride 호환 위해 width 는 자동 정렬.
DOPE_INPUT_HEIGHT = 400
# belief map peak confidence 임계 — challenge/config/task.yaml 의 belief.threshold 와 동일.
DOPE_PEAK_THRESHOLD = 0.30
DOPE_PEAK_SIGMA = 3.0
# DOPE detector 의 보조 임계 (challenge/config/task.yaml inference.belief 와 동일).
DOPE_THRESH_MAP    = 0.30  # affinity grouping 시 영향력 임계
DOPE_THRESH_POINTS = 0.30  # peak 후보 임계
DOPE_THRESH_ANGLE  = 0.50  # affinity field cosine 임계
DOPE_SOFTMAX       = 1000  # affinity softmax temperature

# ===== DOPE sanity gate (challenge/config/task.yaml inference.gates 와 동일) =====
# PnP 결과를 FSM 으로 보내기 전 통과시켜야 하는 게이트들.
# snapshot FSM 은 한 번의 캡처가 잘못되면 즉시 잘못된 회전 cmd 가 전송되므로 게이트 필수.
# 2026-05-28: 시연 모드 — 모든 gate 거의 disable.
# 학습 데이터 (synthetic Blender + Isaac Sim) 와 실차 카메라의 PnP 정확도 차이로
# reproj 자체가 16~30px 일관되게 큼. CAN MOCK 모드라 실제 forklift 안 움직이므로
# 안전. snapshot FSM 의 STOP_SEC=1.2 가 safety net.
# GATE_PROFILE = "demo" (기본, 현재 시연값 그대로 — 회귀 없음) | "real" (실삽입용 강화).
# 실제 리프터가 포크를 끼우기 전(real)에는 나쁜 프레임이 포크 오작동을 일으키지
# 않도록 게이트를 다시 조인다. 환경변수 GATE_PROFILE 로 선택. 값은 rig 에서 튜닝 시작점.
GATE_PROFILE = _os.environ.get("GATE_PROFILE", "demo").lower()

if GATE_PROFILE == "real":
    DOPE_GATE_MIN_KP            = 6      # 신뢰 가능한 PnP 위해 6점 이상
    DOPE_GATE_MAX_REPROJ_PX     = 12.0   # reproj 큰(부정확) pose 거부
    DOPE_GATE_Z_MIN_M           = 0.30
    DOPE_GATE_Z_MAX_M           = 6.00
    DOPE_GATE_DEPTH_REL         = 0.25   # PnP z vs RealSense depth 괴리 크면 거부 (gross fail catch)
    DOPE_GATE_EDGE_RATIO_TOL    = 0.40
    DOPE_CONFIRM_FRAMES         = 3      # N 연속 통과해야 FSM 으로 (안정성)
else:  # "demo"
    DOPE_GATE_MIN_KP            = 4      # PnP 최소 4점 (SQPNP), 더 풀 수 없음
    DOPE_GATE_MAX_REPROJ_PX     = 50.0   # 사실상 reproj gate disable (실차 16~30px 흡수)
    DOPE_GATE_Z_MIN_M           = 0.10   # close-range 시연 (0.3m 이하 OK)
    DOPE_GATE_Z_MAX_M           = 8.00   # 멀어도 OK
    DOPE_GATE_DEPTH_REL         = 1.00   # depth_rel 사실상 disable
    DOPE_GATE_EDGE_RATIO_TOL    = 1.50   # edge_ratio 사실상 disable
    DOPE_CONFIRM_FRAMES         = 1      # 즉시 confirmed

# ===== 연산 디바이스 / 정밀도 =====
# CUDA가 사용 가능하고 USE_GPU=True면 'cuda:{CUDA_DEVICE}'로 동작.
# USE_HALF=True이면 FP16 추론을 시도합니다(지원되는 GPU에서 속도/메모리 이점).
USE_GPU = True
CUDA_DEVICE = 0
USE_HALF = True

# ===== 깊이 샘플링 / 강건화 =====
SAMPLE_STRIDE = 3          # 마스크 내 픽셀 stride 샘플링 간격
Z_INLIER_THRESH = 0.06     # z(inlier) 허용 오차(m)
MIN_POINTS = 120           # RANSAC에 투입할 최소 3D 포인트 개수

# ===== 평면 적합 (RANSAC) =====
PLANE_INLIER_THRESH = 0.01 # 평면 거리 허용(m)
PLANE_MAX_TRIALS = 200
# ===== EMA 스무딩 =====
EMA_ALPHA_OFFSET = 0.4  # 0.0(없음) ~ 1.0(즉시)
EMA_ALPHA_YAW    = 0.4
EMA_ALPHA_WIDTH  = 0.4

# ===== HUD 색상 (BGR for OpenCV) =====
COLOR_STATUS_OK   = (0, 220, 0)
COLOR_STATUS_TRK  = (0, 165, 255)
COLOR_ALERT       = (0, 0, 255)
COLOR_META        = (180, 180, 0)
COLOR_YAW         = (200, 100, 255)
COLOR_OFFSET      = (50, 200, 255)
COLOR_WIDTH       = (255, 170, 0)
COLOR_ZERR        = (0, 220, 255)
COLOR_BOX         = (0, 180, 255)
COLOR_CNT         = (255, 0, 255)
COLOR_CENTER      = (180, 180, 180)
COLOR_PANEL_BG    = (30, 30, 30)
COLOR_PANEL_EDGE  = (70, 70, 70)

# ===== 정렬 허용치 =====
YAW_TOL_DEG   = 2.00   # ±3 deg
OFF_TOL_M     = 0.12   # ±0.15 m

# ===== 폭(가로 길이) 기준 =====
# detected_length >= WIDTH_MIN_FULL -> ALIGN
# detected_length <  WIDTH_MIN_FULL -> RECOVER
WIDTH_MIN_FULL = 0.00  # m (탐지된 파렛트 최소 길이 임계값)

# ===== ALIGN 단계: 거리 밴드 제어 =====
# 6D pose (DOPE) 모드: ALIGN_DIST_M = 카메라에서 팔레트 centroid (무게중심) 까지 목표 거리 (m).
#   예: 전면 face 2.20m 앞에 정렬하려면 ALIGN_DIST_M = 2.20 + PALLET_DEPTH/2 ≈ 2.85.
# RGB-D + RANSAC 모드 (legacy): 전면 plane 기준이므로 ALIGN_DIST_M = 2.20 (전면 face).
# 두 모드 혼용 시 dist_z 와 d_forward_m 의 기준점이 다름에 유의.
ALIGN_DIST_M = 2.20    # 정렬 목표 거리 (m, 현재 값은 전면 기준)
ALIGN_BAND_M = 0.30    # 허용 밴드 (±m)

# 팔레트 실측 (scan_cleanup/pallet_full.obj). 어댑터 기본값과 동기.
PALLET_WIDTH_M  = 1.10
PALLET_DEPTH_M  = 1.30
PALLET_HEIGHT_M = 0.12

# ===== 카메라 → 포크(차량) frame 고정 extrinsic =====
# 카메라가 포크 중심에 없으므로 pose 는 camera frame. 포크 frame 으로 옮기는 고정
# 변환. ⚠ 실측 전이라 placeholder = 항등 (오프셋 0, 회전 0) → 현재 동작과 동일.
#   CAM_TO_FORK_T       : [x, y, z] (m) camera origin → fork origin 오프셋.
#   CAM_TO_FORK_RPY_DEG : [roll, pitch, yaw] (deg) camera axes → fork axes 회전.
# 실측 후 아래 값을 채울 것 (OpenCV X右 Y下 Z前 convention 유지).
CAM_TO_FORK_T       = [0.0, 0.0, 0.0]   # m
CAM_TO_FORK_RPY_DEG = [0.0, 0.0, 0.0]   # deg

# centroid 픽셀 depth scale 보정 유효 거리 (m). 이 범위 밖 depth 는 무시.
DEPTH_CORRECT_Z_MIN_M = 0.10
DEPTH_CORRECT_Z_MAX_M = 12.0

# ===== 포크 hardware 측정 (forklift 자체 spec) =====
# fork center → fork tip 까지 거리 (m). centerToP + forkLen.
FORK_CENTER_TO_TIP_M = 1.77   # 0.72 (center→P) + 1.05 (P→tip) = 1.77

# fork center → forklift body forward edge (mast / fork carriage 의 전면) 까지 거리 (m).
# centerToP - (forklift body 의 P 점 기준 frontY offset).
# 실제 forklift 의 mast 위치 측정값으로 조정. 기본값은 sim.js 의 frontY=0.05 와 일치.
FORK_CENTER_TO_BODY_FRONT_M = 0.67   # 0.72 (center→P) - 0.05 (P→body front) = 0.67

# ===== INSERT 안전 margin =====
# (deprecated) fork tip 이 pallet back face 에서 띄울 거리. 옛 fork-tip 기준 식.
INSERT_SAFETY_BACK_M = 0.10

# forklift body forward edge (mast/carriage 전면) 가 pallet entry face 에서 띄울
# 최소 거리 (m). 종료 시 body 가 entry face 의 INSERT_BODY_SAFETY_M 앞에 멈춤
# (mast 가 pallet 정면에 부딪히지 않도록). fork tip 은 자연히 안쪽 forkLen 만큼
# 들어가 pallet 의 ~85% 침투, back face 까지 충분한 여유.
INSERT_BODY_SAFETY_M = 0.05

# ===== 정렬 완료 판정 안정화 =====
CMD_STABLE_THR = 5     # 같은 판정이 연속 n프레임 유지돼야 상태 전이

# ===== 회전 목표(각도 기반) =====
# Snapshot 기반 새 다이어그램:
#   - YAW_CORRECT_* / LATERAL_ROTATE_* (첫 회전): snapshot 시점 |ψ_pallet| 만큼 회전.
#     실제 IMU 누적 종료 기준은 max(YAW_TURN_MIN_DEG, snapshot_psi_abs) 또는 그대로.
#   - LATERAL_ROTATE_*_BACK (복귀): LATERAL_BACK_YAW_DEG 만큼 회전.
# 하위 호환: 기존 코드 (구 align.py) 가 REL_YAW_TARGET_DEG 를 참조할 수 있으므로 유지.
REL_YAW_TARGET_DEG    = 85.0   # (legacy) 90° 체인 도달 조건
LATERAL_BACK_YAW_DEG  = 85.0   # *_BACK 복귀 회전 목표 (실측 IMU 노이즈/under-rotate 마진 고려)
YAW_TURN_MIN_DEG      = 1.0    # 첫 회전 최소 보장 (snapshot |ψ| 가 너무 작으면 무시될 수 있어 floor)

# ===== 전역 인터록(정지) =====
# 모든 제어 명령 상태 전환 사이에 STOP을 이 시간만큼 유지합니다.
STOP_SEC = 1.2

# (하위호환) 기존 코드에서 STOP_PAUSE_SEC을 참조하면 STOP_SEC을 사용하도록
STOP_PAUSE_SEC = STOP_SEC

# ===== 전진시간 피팅(가속→정속) 파라미터 =====
# 로그(t_monotonic & dist_z) 기반으로 적합된 파라미터.
# d_acc = 0.5 * FWD_A * FWD_T1^2,  vmax = FWD_A * FWD_T1
# t(d) =
#   d <= d_acc:  FWD_T0 + sqrt(2d/FWD_A)
#   d >  d_acc:  FWD_T0 + FWD_T1 + (d - d_acc)/vmax
USE_PIECEWISE_FWD_FIT = True

FWD_T0   = -0.0202    # s (명령→실이동 시작 지연; ≈0으로 봐도 무방)
FWD_T1   = 4.2780     # s (가속 구간 지속시간)
FWD_A    = 0.071565   # m/s^2 (초기 가속도)
FWD_SCALE = 1.0       # d_eff = FWD_SCALE*|offset| + FWD_BIAS
FWD_BIAS  = 0.0

# 안전 클램프(전진 명령 타이머)
FWD_MIN_SEC = 1.0
FWD_MAX_SEC = 15.0

