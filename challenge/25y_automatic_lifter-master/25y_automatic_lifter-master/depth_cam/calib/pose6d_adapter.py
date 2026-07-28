# calib/pose6d_adapter.py
# -----------------------------------------------------------------------------
# DOPE / FoundationPose 6D pose → ALIGN FSM 변수 (ψ_pallet, d_lateral, d_forward).
#
# 좌표계: OpenCV (+X right, +Y down, +Z forward / optical axis).
#
# DOPE 9 keypoint convention (v4):
#   0: front-top-LEFT   1: front-top-RIGHT
#   2: front-bot-RIGHT  3: front-bot-LEFT
#   4: rear-top-LEFT    5: rear-top-RIGHT
#   6: rear-bot-RIGHT   7: rear-bot-LEFT
#   8: centroid (무게중심)  ← ALIGN 의 기준점
#
# ALIGN 변수 정의 (모두 centroid 기준):
#   ψ_pallet (deg) : forklift heading 과 pallet forward 축 (centroid 를 지남)
#                    사이 각도. ψ=0 일 때 forklift.yaw == pallet.yaw.
#                    R 의 column (팔레트 +Z 축의 camera frame 표현) 으로 추출.
#                    회전이라 기준점에 무관 (centroid든 어디든 같은 값).
#   d_lateral (m)  : 카메라 +X (right) 방향 centroid 오프셋의 **부호 반전**값.
#                    부호 반전 이유: align.py 의 LATERAL 분기는 "d_lat<0 → ROT_RIGHT,
#                    d_lat>0 → ROT_LEFT" 인데, OpenCV +X right 정의 그대로 쓰면
#                    실차에서 "반대 방향 회전" 이 발생함 (2026-05-27 사용자 보고).
#                    시뮬에서 부호 반전으로 LATERAL chain 정상 진입 검증 완료.
#   d_forward (m)  : 카메라 +Z (forward) 방향 centroid 거리. 목표 ALIGN_DIST_M
#                    (centroid 기준).
#
# FSM 정렬 목표:
#   ψ_pallet ≈ 0 AND |d_lateral| ≤ OFF_TOL_M AND |d_forward - ALIGN_DIST_M| ≤ ALIGN_BAND_M
#   → READY_TO_DONE → INSERT (compute Δt_fwd from d_forward) → DONE
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Tuple, Optional, Sequence
import math

# 팔레트 실측 크기 (scan_cleanup/pallet_full.obj, KS T-11 변형)
PALLET_WIDTH_M  = 1.10
PALLET_DEPTH_M  = 1.30
PALLET_HEIGHT_M = 0.12


def _to_3x3(R) -> list:
    """numpy ndarray / list-of-lists 둘 다 받기."""
    try:
        return [[float(R[i][j]) for j in range(3)] for i in range(3)]
    except Exception:
        import numpy as np
        Rn = np.asarray(R, dtype=float).reshape(3, 3)
        return Rn.tolist()


def _to_3vec(t) -> Tuple[float, float, float]:
    try:
        return float(t[0]), float(t[1]), float(t[2])
    except Exception:
        import numpy as np
        tn = np.asarray(t, dtype=float).reshape(3)
        return float(tn[0]), float(tn[1]), float(tn[2])


def pose6d_to_align_vars(R, t, anchor: str = "entry_face") -> Tuple[float, float, float]:
    """OpenCV (R, t) → (psi_pallet_deg, d_lateral_m, d_forward_m).

    입력:
        R: 3x3, pallet local axes 의 camera frame 표현 (DOPE/FoundationPose 출력).
        t: 3-vec (m), 카메라 frame 에서 본 pallet model origin = centroid 위치.
        anchor:
          "entry_face"  : (default) d_lateral/d_forward 를 entry face 중심 기준.
                          centroid + R @ (0, 0, +depth/2) 변환. 기존 perception.py 와
                          호환되어 config 의 ALIGN_DIST_M, OFF_TOL_M 그대로 사용 가능.
          "centroid"    : centroid 그대로 (t).

    Returns:
        psi_pallet_deg : ψ_pallet (deg). + 이면 팔레트 정면이 카메라 우측.
        d_lateral_m   : anchor 의 카메라 +X 좌표 (m).
        d_forward_m   : anchor 의 카메라 +Z 좌표 (m).
    """
    Rm = _to_3x3(R)
    tx, _, tz = _to_3vec(t)

    # ψ_pallet — 카메라가 팔레트 정면을 바라볼 때 두 +Z 축이 마주봄 → atan2 가
    # ±180° 를 반환. align.py 의 YAW_TOL_DEG=2° (정렬 완료 시 ψ≈0° 기대) 와
    # 컨벤션이 어긋나므로 180° offset 을 흡수해 정렬 완료를 0° 로 정의.
    # 2026-05-27 실차 디버깅에서 ψ=-165.6°/+178.2° 측정 → fsm 이 -165° 회전
    # cmd 보내서 리프터 거의 안 움직임. wrap_to_180(psi + 180) 으로 fix.
    psi_rad = math.atan2(Rm[0][2], Rm[2][2])
    psi_deg = math.degrees(psi_rad) + 180.0
    psi_deg = ((psi_deg + 180.0) % 360.0) - 180.0   # wrap to [-180, 180]

    if anchor == "entry_face":
        # entry face = centroid + R @ (0, 0, +depth/2)
        offset = PALLET_DEPTH_M / 2.0
        ax = tx + Rm[0][2] * offset
        az = tz + Rm[2][2] * offset
        # d_lateral 부호 반전 — align.py LATERAL 분기 매칭 (시뮬 검증 완료)
        return psi_deg, float(-ax), float(az)
    # centroid anchor 도 동일하게 부호 반전
    return psi_deg, float(-tx), float(tz)


def keypoints9_to_align_vars(
    keypoints_2d: Sequence[Sequence[float]],
    camera_matrix,
    dist_coeffs=None,
    pallet_width_m: float = PALLET_WIDTH_M,
    pallet_depth_m: float = PALLET_DEPTH_M,
    pallet_height_m: float = PALLET_HEIGHT_M,
) -> Optional[Tuple[float, float, float]]:
    """DOPE 9 keypoint (image px) + 카메라 intrinsic → ALIGN 변수.

    내부에서 PnP (EPnP + iterative refine) 로 (R, t) 복원 후 centroid 기준 변환.
    centroid (keypoint 8) 는 PnP 입력에도 포함되어 추정 안정성을 높임.

    Args:
        keypoints_2d  : [(u, v)] x 9, v4 convention 순서.
                        invisible keypoint 는 (NaN, NaN) 또는 (-1, -1) 로 표기.
        camera_matrix : 3x3 K (fx, fy, cx, cy).
        dist_coeffs   : OpenCV distortion (k1, k2, p1, p2[, k3]). None=무왜곡.
        pallet_*_m    : 모델 실측 (default = pallet_full.obj 1.10×1.30×0.12).

    Returns:
        (psi_pallet_deg, d_lateral_m, d_forward_m) 또는 PnP 실패 시 None.
    """
    try:
        import numpy as np
        import cv2
    except Exception:
        return None

    # v4 convention 의 모델 좌표 (centroid 가 origin, 단위 m).
    # +X = pallet right(width), +Y = up(height), +Z = forward(depth).
    hw, hh, hd = pallet_width_m / 2.0, pallet_height_m / 2.0, pallet_depth_m / 2.0
    object_points = np.array([
        [-hw, +hh, +hd],   # 0 front-top-LEFT
        [+hw, +hh, +hd],   # 1 front-top-RIGHT
        [+hw, -hh, +hd],   # 2 front-bot-RIGHT
        [-hw, -hh, +hd],   # 3 front-bot-LEFT
        [-hw, +hh, -hd],   # 4 rear-top-LEFT
        [+hw, +hh, -hd],   # 5 rear-top-RIGHT
        [+hw, -hh, -hd],   # 6 rear-bot-RIGHT
        [-hw, -hh, -hd],   # 7 rear-bot-LEFT
        [0.0, 0.0, 0.0],   # 8 centroid
    ], dtype=np.float64)

    kps = np.array(keypoints_2d, dtype=np.float64)
    if kps.shape != (9, 2):
        return None

    # invisible keypoint 제거
    visible = ~np.isnan(kps).any(axis=1) & (kps[:, 0] >= 0) & (kps[:, 1] >= 0)
    n_vis = int(visible.sum())
    if n_vis < 4:
        return None

    obj_pts = object_points[visible].reshape(-1, 1, 3).astype(np.float64)
    img_pts = kps[visible].reshape(-1, 1, 2).astype(np.float64)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    # OpenCV solvePnP 는 dist_coeffs 가 2D (N,1) 또는 (1,N) 여야 함. 1D 면 m.dims>=2 assert fail.
    if dist_coeffs is None:
        D = np.zeros((5, 1), dtype=np.float64)
    else:
        D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    # EPnP: >=6 points 권장. SQPNP: 4+ points 지원 (OpenCV 4.5+). ITERATIVE: 초기값 필요 → skip.
    if n_vis >= 6:
        flag = cv2.SOLVEPNP_EPNP
    else:
        # 4~5 points: SQPNP 가 안정적. 없으면 fallback.
        flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)

    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flag)
    except cv2.error:
        # 4-5 points 에서 어떤 flag 든 fail 할 수 있음 — silent skip.
        return None
    if not ok:
        return None
    # iterative refine (>=6 일 때만; 미만이면 over-fit 가능)
    if n_vis >= 6:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)
        except cv2.error:
            pass

    R, _ = cv2.Rodrigues(rvec)
    return pose6d_to_align_vars(R, tvec.ravel())


def pose6d_to_align_vars_safe(R, t) -> Optional[Tuple[float, float, float]]:
    """예외 안전 wrapper. 변환 실패 시 None."""
    try:
        return pose6d_to_align_vars(R, t)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Depth scale 보정 — monocular PnP 의 scale 모호성을 RealSense depth 로 고정.
#
# 동기: monocular PnP 는 keypoint 비율만 보므로 절대 거리에 scale 오차가 있다
#       (YOLO backend 에서 ~0.6m bias 관측). centroid 픽셀의 RealSense depth(m)
#       는 광선 위 절대 거리이므로, t_m[2] 를 측정 depth 에 맞추도록 전체 t 를
#       동일 비율로 스케일하면 거리 정확도가 확보된다.
#
# 기하: centroid 는 카메라 광선 (원점→centroid) 위의 점이다. PnP 의 t_m 과 진짜
#       centroid 는 같은 광선을 공유하고 scale 만 다르므로, s = depth/t_m[2] 로
#       t 전체를 곱하면 z 는 측정 depth 와 일치하고 x,y 는 같은 광선 비율로
#       이동한다 (방향 R 은 불변, ψ 보존).
# -----------------------------------------------------------------------------
def depth_scale_correct(
    t_m,
    centroid_uv: Optional[Sequence[float]] = None,
    depth_m: Optional[float] = None,
    z_min_m: float = 0.10,
    z_max_m: float = 12.0,
) -> Tuple[Tuple[float, float, float], bool]:
    """RealSense depth 로 monocular PnP 의 t 를 scale 보정.

    Args:
        t_m         : 3-vec (m), PnP 추정 centroid 위치 (camera frame, OpenCV).
        centroid_uv : centroid 픽셀 (u, v). 로깅/유효성 표시용 (계산엔 미사용,
                      depth_m 은 호출측에서 이미 샘플링해 전달).
        depth_m     : centroid 픽셀의 RealSense depth (m). None/0/범위밖이면 무보정.
        z_min_m     : depth 유효 최소 거리 (m).
        z_max_m     : depth 유효 최대 거리 (m).

    Returns:
        (t_corrected, corrected_flag)
        depth 유효 시  : (t_m * s, True),  s = depth_m / t_m[2]
        depth 무효 시  : (t_m 원본, False)
    """
    tx, ty, tz = _to_3vec(t_m)
    # depth 유효성 — None/비양수/범위밖, 또는 PnP z 가 0 에 가까우면 무보정.
    valid = (
        depth_m is not None
        and depth_m > z_min_m
        and depth_m < z_max_m
        and abs(tz) > 1e-6
    )
    if not valid:
        return (tx, ty, tz), False
    s = float(depth_m) / float(tz)
    return (tx * s, ty * s, tz * s), True


# -----------------------------------------------------------------------------
# 카메라 → 포크(차량) frame 고정 변환 골격.
#
# 카메라는 포크 중심에 있지 않으므로 pose 는 camera frame 이다. 실차 정렬은 포크
# 기준이어야 하므로 고정 extrinsic 으로 옮긴다. 실측 전이므로 기본값은 항등
# (CAM_TO_FORK_T=0, RPY=0) → 현재 동작과 완전히 동일 (회귀 없음).
#
#   T_fork_cam = [[R_cf, t_cf], [0,0,0,1]]  (camera frame → fork frame)
#   pose_fork  = T_fork_cam @ pose_cam
#   R_fork = R_cf @ R_cam
#   t_fork = R_cf @ t_cam + t_cf
#
# ⚠ 실측 후 calib/config.py 의 CAM_TO_FORK_T / CAM_TO_FORK_RPY_DEG 값을 채울 것.
# -----------------------------------------------------------------------------
def _rpy_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list:
    """RPY(deg, OpenCV XYZ 고정축) → 3x3 회전행렬. R = Rz @ Ry @ Rx."""
    r, p, y = (math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz(y) @ Ry(p) @ Rx(r)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ]


def apply_cam_to_fork(
    R,
    t,
    cam_to_fork_t: Optional[Sequence[float]] = None,
    cam_to_fork_rpy_deg: Optional[Sequence[float]] = None,
) -> Tuple[list, Tuple[float, float, float]]:
    """camera frame pose (R, t) → fork frame pose (R_fork, t_fork).

    OpenCV convention 유지. 기본값(0)이면 입력=출력(항등) — 회귀 없음.

    Args:
        R                   : 3x3 (pallet local axes in camera frame).
        t                   : 3-vec (m), camera frame.
        cam_to_fork_t       : [x, y, z] (m) camera→fork 오프셋. None → [0,0,0].
        cam_to_fork_rpy_deg : [roll, pitch, yaw] (deg). None → [0,0,0].

    Returns:
        (R_fork (3x3 list), t_fork (3-vec)).
    """
    if cam_to_fork_t is None:
        cam_to_fork_t = (0.0, 0.0, 0.0)
    if cam_to_fork_rpy_deg is None:
        cam_to_fork_rpy_deg = (0.0, 0.0, 0.0)
    Rcf = _rpy_to_R(*[float(v) for v in cam_to_fork_rpy_deg])
    tcf = [float(v) for v in cam_to_fork_t]
    Rm = _to_3x3(R)
    tx, ty, tz = _to_3vec(t)
    # R_fork = Rcf @ Rm
    R_fork = [
        [sum(Rcf[i][k] * Rm[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    # t_fork = Rcf @ t + tcf
    tv = (tx, ty, tz)
    t_fork = tuple(
        sum(Rcf[i][k] * tv[k] for k in range(3)) + tcf[i] for i in range(3)
    )
    return R_fork, t_fork
