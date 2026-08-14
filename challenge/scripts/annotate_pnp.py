"""annotate.py — PnP 수학 모듈 (fix v9 — degenerate threshold 동적 스케일).

3D cuboid 모델, projection, PnP 풀이, manipulate 모드 pose 변환.

핵심 함수:
  make_pallet_keypoints_3d_diagram(W, D, H)  : 9-keypoint 3D 모델 (8 corner + centroid)
  project_3d(kp3d, R, t, K)                  : 3D → 2D projection
  solve_pose(kps_2d, K)                      : auto-dim PnP (110 vs 130 정면) — strict v6+v8
  pose_from_locked(state, K, dims)           : MANIPULATE 모드 pose 재구성
  apply_manip(state, dx/dy/dz/dyaw/dpitch/droll): MANIPULATE 모드 6DoF 미세조정
  line_intersection(p1, p2, p3, p4)          : TWO-LINE 모드 교점

좌표 convention: OpenCV (X=right, Y=down, Z=forward).

fix v6 (2026-05-24): 0~3 Z 부호 swap + pair-wise strict invariants + IPPE 6-face seed.
fix v7 (2026-05-24): _reproj_err_dict u<0 sentinel 수정 + degenerate bbox reject +
                     extrapolated_mask weight 0.3.
fix v8 (2026-05-25): Gravity-prior tilt penalty (R[1,1] hard/soft threshold).

fix v9 (2026-05-26) — Far-pallet degenerate threshold scaling
  (`diagnose_v8_night03.py` 케이스).

  증상: capturenight03 1779448848688752640 — pallet 멀리 (작게, ~6m) + oblique view,
  사용자 click 0~5 (6점), bbox 119×17 = 2023px². v8 enum 시 *정답* candidate
  (110front, reproj 3.61px, tilt 1.00, bbox 3068px²) 가 존재했으나
  `_solve_pose_single` 의 degenerate threshold (image area 1.5% = 4608px²) 가 3068
  < 4608 으로 *reject*. 결과: 더 큰 (4621px²) 부정확 candidate 가 채택 (reproj
  6.5px) → wireframe 이 사용자 click 보다 위/아래로 살짝 어긋남 (collapse 처럼 보임).

  원인: v7 의 fixed 1.5% threshold 는 click bbox 자체가 image 의 0.7% (멀리있을때)
  인 케이스를 고려 안 함. click 자체가 threshold 보다 작은데 candidate 만 더 크라고
  강제 → 정답 reject.

  v9 fix: degenerate threshold = max(click_bbox_area × 0.5, image_area × 0.5%).
  - 멀리 (small) click: click bbox 의 절반 정도까지만 허용 → 정답 candidate 살아남음.
  - 가까이 (large) click: 여전히 image area 의 0.5% (1536px²) bottom-floor →
    extreme collapse (한 점 몰림 ≤ 30×30px) 는 막힘.

  검증: night03 케이스에서 selected candidate 가 reproj 6.50 → 3.61 로 개선.
        pallet08 회귀 없음 (click_bbox=3912 → threshold=1956, 기존 selected 4621 살아남음).
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- 계열 폴더 탐색.
#     이 파일만 challenge/scripts 루트에 남는다: 해시가 고정된
#     scripts/stage0/paper_s2_frozen_diagnostic.py 가 이 경로로 import 하는데
#     그 파일은 한 글자만 바뀌어도 캐시가 무효가 되어 손댈 수 없다.
_CS = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d))
                         and not _d.startswith((".", "_"))]

import os
import sys
import numpy as np
import cv2

# v4 컨벤션 permutation 계산기 보존 (학습 데이터 변환에 사용중 — 의존성).
# 본 패치(v6) 의 strict scoring 은 학습 데이터 변환 로직과 무관 — compute_perm_v4 는
# 호환성을 위해 import 만 유지하고 solve_pose 내부 진단용으로만 사용.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from convert_to_camera_facing_v4 import compute_perm_v4 as _compute_perm_v4_z_height


# (width, depth, height) — 실측 사용자 plastic 팔레트 110 × 130 × 11 cm
PALLET_DIMS = (1.1, 1.3, 0.11)


def make_pallet_keypoints_3d_diagram(width=1.1, depth=1.3, height=0.11):
    """Camera-facing convention 9-keypoint 3D 모델 (fix v6, 2026-05-24).

    cuboid local frame: X=right (+), Y=down (OpenCV +y=bottom), Z=forward (+).
    Indices (R=I 가정 시):
      0: near-top-LEFT       (-w/2, -h/2, -d/2)    ★ near = Z_local 작은 쪽 = cam.z 작은쪽
      1: near-top-RIGHT      (+w/2, -h/2, -d/2)
      2: near-bottom-RIGHT   (+w/2, +h/2, -d/2)
      3: near-bottom-LEFT    (-w/2, +h/2, -d/2)
      4: far-top-LEFT        (-w/2, -h/2, +d/2)    ★ far = Z_local 큰 쪽
      5: far-top-RIGHT       (+w/2, -h/2, +d/2)
      6: far-bottom-RIGHT    (+w/2, +h/2, +d/2)
      7: far-bottom-LEFT     (-w/2, +h/2, +d/2)
      8: centroid

    v6 change: 이전 (v1~v5) 의 0~3 = +d/2 정의는 R=I + cam +Z forward 환경에서
    "0~3 cam.z 큰 = FAR" 라는 모순을 만들어, 어떤 proper rotation 으로도
    LR ∧ TB ∧ FR invariant 동시 만족 candidate 가 안 나옴 (반사 필요). Z 부호 flip
    으로 모순 제거. 학습 데이터 변환 로직 (`convert_to_camera_facing_v4`) 은 origin
    frame 3D coordinate 자체를 기준으로 동작하므로 영향 없음.
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-w, -h, -d],   # 0 near-top-LEFT
        [+w, -h, -d],   # 1 near-top-RIGHT
        [+w, +h, -d],   # 2 near-bottom-RIGHT
        [-w, +h, -d],   # 3 near-bottom-LEFT
        [-w, -h, +d],   # 4 far-top-LEFT
        [+w, -h, +d],   # 5 far-top-RIGHT
        [+w, +h, +d],   # 6 far-bottom-RIGHT
        [-w, +h, +d],   # 7 far-bottom-LEFT
    ], dtype=np.float64)
    centroid = corners.mean(axis=0, keepdims=True)
    return np.vstack([corners, centroid])


# Alias
make_pallet_keypoints_3d = make_pallet_keypoints_3d_diagram


# ── v6 pair-wise strict invariants ──────────────────────────────────────────
# fix v6 강제 — 세 그룹 모두 부등호 위반 (≥1) 인 candidate 즉시 reject.
LR_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]   # proj.u: a < b (left < right)
TB_PAIRS = [(0, 3), (1, 2), (4, 7), (5, 6)]   # proj.v: a < b (top < bot, OpenCV y=down)
FR_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]   # cam.z:  a < b (near < far)

# ── v8 gravity-prior thresholds ─────────────────────────────────────────────
# cuboid local Y axis (height direction) 의 cam-frame Y 성분 = |R[1, 1]|
# 1 = perfectly upright (pallet height axis 가 cam Y 와 정렬), 0 = 옆으로 누움
# 카메라가 RealSense D435i 가 pallet 을 바라보는 다양한 pitch 에 따라 R[1,1] 분포:
#   - 거의 수평 view (pallet 측면): R[1,1] ≈ 1.00
#   - 35° 아래로 (typical oblique top-down): R[1,1] ≈ 0.82
#   - 45° 아래로 (drone-like): R[1,1] ≈ 0.71
#   - 수직 아래 (overhead): R[1,1] ≈ 0.00
# saved frames (capturepalletcad set) 실측 tilt 분포: median=0.999, min=0.67.
# - V8_TILT_SOFT_THR = 0.60 (false-alarm 회피, 실측 min 0.67 보다 약간 낮게)
# - V8_TILT_HARD_THR = 0.30 (75°+ 누움 — 진짜 비정상)
# selection 에는 사용하지 않음 (단순 diagnostic flag + GUI 경고 trigger).
V8_TILT_SOFT_THR = 0.60   # < 0.60 → soft warning (53°+ tilt 의심)
V8_TILT_HARD_THR = 0.30   # < 0.30 → hard reject (75°+ 누움 — 거의 옆으로 누움)


def project_3d(kp3d, R, t, K):
    """3D points (N, 3) → 2D pixel (N, [u, v])."""
    pts_cam = (R @ kp3d.T).T + t
    proj = []
    for p in pts_cam:
        if p[2] <= 0:
            proj.append([-1.0, -1.0])
        else:
            u = K[0, 0] * p[0] / p[2] + K[0, 2]
            v = K[1, 1] * p[1] / p[2] + K[1, 2]
            proj.append([float(u), float(v)])
    return proj


def project_with_pose(R, t, K, dims):
    """주어진 dims 의 3D cuboid 를 R, t 로 화면에 projection."""
    return project_3d(make_pallet_keypoints_3d(*dims), R, t, K)


def _reproj_err_dict(proj_all, valid_idx, kps_2d, weights=None):
    """reproj mean. proj_all=(N,2) list, valid_idx=[i...], kps_2d=[[u,v] or None].

    fix v7 (2026-05-24): "u < 0" sentinel 버그 수정. 사용자가 image 밖 (u<0 또는
    u>W) 점을 t/x 외삽으로 정확히 클릭한 경우, projection 의 u 도 음수로 나올 수
    있는데 기존 코드는 이를 "behind camera" 로 오인 → 1e6 error 채택 → 모든
    candidate 가 잘못된 reproj 로 평가되어 selection 망가짐.

    project_3d 의 진짜 sentinel 은 (u, v) == (-1.0, -1.0). 그것만 1e6 처리하고
    그 외 음수 u/v 는 image 밖 valid projection 으로 취급해 normal 거리 계산.

    weights (선택): valid_idx 와 동일 길이의 [0..1] weight. 사용자 직접 클릭 = 1.0,
    extrapolated (t/x) = 0.3 식 가중. weighted mean 으로 반환. None 이면 equal.
    """
    errs = []
    ws = []
    for j, i in enumerate(valid_idx):
        u, v = proj_all[i]
        if u == -1.0 and v == -1.0:
            errs.append(1e6); ws.append(1.0); continue
        du, dv = u - kps_2d[i][0], v - kps_2d[i][1]
        errs.append(float(np.hypot(du, dv)))
        ws.append(float(weights[j]) if weights is not None else 1.0)
    if not errs:
        return 1e9
    if weights is None:
        return float(np.mean(errs))
    arr = np.array(errs, dtype=np.float64)
    wt  = np.array(ws,   dtype=np.float64)
    s = wt.sum()
    if s <= 1e-9:
        return float(np.mean(arr))
    return float(np.sum(arr * wt) / s)


def _refine_with_init(obj, img, K, R0, t0, weights=None):
    """LM refine from given init. Returns ``(R, t)`` or ``None``.

    ``cv2.solvePnP(..., SOLVEPNP_ITERATIVE)`` does not expose per-point
    weights.  The legacy, unweighted path therefore remains byte-for-byte the
    same when ``weights is None``.  When weights are supplied, a small
    weighted LM loop uses the analytic projection Jacobian returned by
    :func:`cv2.projectPoints`.  This lets a future corner-uncertainty head
    affect the actual refinement rather than only the final candidate score.
    """
    try:
        rvec0, _ = cv2.Rodrigues(R0)
        tvec0 = t0.reshape(3, 1).astype(np.float64)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(weights) != len(obj):
                raise ValueError("weights must match the PnP correspondences")
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("weights must be finite and non-negative")
            if np.count_nonzero(weights > 0) < 4:
                return None
            return _weighted_lm_refine(obj, img, K, rvec0, tvec0, weights)

        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, None,
            rvec=rvec0.copy(), tvec=tvec0.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()
    except cv2.error:
        return None


def _weighted_lm_refine(obj, img, K, rvec0, tvec0, weights,
                        max_iterations=20):
    """Weighted six-DoF LM using OpenCV's projection Jacobian.

    This intentionally has no SciPy dependency because ``annotate_pnp`` is
    also used by the standalone annotation GUI.  ``weights`` are reliability
    weights (larger = more trusted); the least-squares residual is multiplied
    by ``sqrt(weight)``.
    """
    params = np.concatenate([
        np.asarray(rvec0, dtype=np.float64).reshape(3),
        np.asarray(tvec0, dtype=np.float64).reshape(3),
    ])
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    sqrt_w = np.repeat(np.sqrt(np.asarray(weights, dtype=np.float64)), 2)
    damping = 1e-3

    def evaluate(p, need_jacobian):
        projected, jacobian = cv2.projectPoints(
            obj, p[:3].reshape(3, 1), p[3:].reshape(3, 1), K, None)
        residual = projected.reshape(-1, 2) - img
        weighted_residual = residual.reshape(-1) * sqrt_w
        if not need_jacobian:
            return weighted_residual, None
        # OpenCV columns 0:3 and 3:6 are derivatives wrt rvec and tvec.
        weighted_jacobian = jacobian[:, :6] * sqrt_w[:, None]
        return weighted_residual, weighted_jacobian

    residual, jacobian = evaluate(params, True)
    if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
        return None
    cost = float(residual @ residual)

    for _ in range(max_iterations):
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        scale = np.maximum(np.diag(normal), 1.0)
        try:
            delta = np.linalg.solve(
                normal + damping * np.diag(scale), -gradient)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(delta).all():
            return None

        trial = params + delta
        trial_residual, _ = evaluate(trial, False)
        trial_cost = float(trial_residual @ trial_residual)
        if np.isfinite(trial_cost) and trial_cost < cost:
            params = trial
            cost = trial_cost
            damping = max(damping * 0.3, 1e-9)
            residual, jacobian = evaluate(params, True)
            if np.linalg.norm(delta) < 1e-8:
                break
        else:
            damping = min(damping * 10.0, 1e12)
            if damping >= 1e12:
                break

    R, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    return R, params[3:].copy()


# 24 proper rotations of a cube (octahedral group) — face-flip ambiguity resolver
_CUBE_FLIPS_DEG = [
    (0, 0, 0), (90, 0, 0), (180, 0, 0), (270, 0, 0),
    (0, 90, 0), (0, 180, 0), (0, 270, 0),
    (0, 0, 90), (0, 0, 180), (0, 0, 270),
    (90, 90, 0), (90, 180, 0), (90, 270, 0),
    (180, 90, 0), (180, 270, 0),
    (270, 90, 0), (270, 180, 0), (270, 270, 0),
    (90, 0, 90), (90, 0, 180), (90, 0, 270),
    (270, 0, 90), (270, 0, 180), (270, 0, 270),
]


def _rot_axis_angle(axis, deg):
    v = np.array(axis, dtype=np.float64) * np.deg2rad(deg)
    return cv2.Rodrigues(v)[0]


def _eval_pair_invariants(R, t, K, kp3d):
    """v6 pair-wise invariants 평가.

    Returns:
      lr_viol, tb_viol, fr_viol  : 각 그룹 위반 pair 개수 (0..4)
      proj_all                   : project_3d(kp3d, R, t, K)  (length 9)
      pts_cam                    : (R @ kp3d.T).T + t          (shape 9x3)
    """
    pts_cam = (R @ kp3d.T).T + t
    proj_all = project_3d(kp3d, R, t, K)
    proj = np.array(proj_all[:8], dtype=np.float64)

    lr_viol = sum(1 for (a, b) in LR_PAIRS if not (proj[a, 0] < proj[b, 0]))
    tb_viol = sum(1 for (a, b) in TB_PAIRS if not (proj[a, 1] < proj[b, 1]))
    fr_viol = sum(1 for (a, b) in FR_PAIRS if not (pts_cam[a, 2] < pts_cam[b, 2]))
    return lr_viol, tb_viol, fr_viol, proj_all, pts_cam


def _eval_v8_tilt(R):
    """v8 gravity-prior tilt 평가.

    cuboid local Y axis (height direction, palletheight=0.11m) 의 cam-frame Y 성분.
    = R @ (0,1,0) 의 Y 성분 = R[1, 1].
    |R[1, 1]| 이 1 에 가까울수록 pallet 이 upright (수평면 위), 0 에 가까울수록 누움.

    Returns: float in [0, 1] — 1 = perfectly upright, 0 = lying on side.
    """
    return float(abs(R[1, 1]))


def _seed_from_ippe_face(kps_2d, K, kp3d, face_indices):
    """face_indices (planar 4 점) 으로 IPPE PnP → 0/1/2 개 seed (cheirality OK 만)."""
    seeds = []
    if not all(i < len(kps_2d) and kps_2d[i] is not None for i in face_indices):
        return seeds
    obj = np.array([kp3d[i] for i in face_indices], dtype=np.float64)
    img = np.array([kps_2d[i] for i in face_indices], dtype=np.float64)
    try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            for rv, tv in zip(rvecs, tvecs):
                if tv[2, 0] > 0:
                    R, _ = cv2.Rodrigues(rv)
                    seeds.append((R, tv.flatten()))
    except cv2.error:
        pass
    return seeds


def _eval_click_lr_viol(kps_2d):
    """사용자 클릭 u 부등호 위반 (LR_PAIRS, 양 쪽 모두 클릭된 pair 만)."""
    n = 0
    for (a, b) in LR_PAIRS:
        if (a < len(kps_2d) and b < len(kps_2d)
                and kps_2d[a] is not None and kps_2d[b] is not None):
            if kps_2d[a][0] >= kps_2d[b][0] - 1.0:
                n += 1
    return n


def _eval_click_tb_viol(kps_2d):
    """사용자 클릭 v 부등호 위반 (TB_PAIRS, 양 쪽 모두 클릭된 pair 만)."""
    n = 0
    for (a, b) in TB_PAIRS:
        if (a < len(kps_2d) and b < len(kps_2d)
                and kps_2d[a] is not None and kps_2d[b] is not None):
            if kps_2d[a][1] >= kps_2d[b][1] - 1.0:
                n += 1
    return n


# 6 faces of cuboid — IPPE seed source (v6 + truncation fix 2026-05-24)
# v6 convention: 0~3 near (Z=-d/2), 4~7 far (Z=+d/2)
_CUBOID_FACES = [
    ("FRONT",  (0, 1, 2, 3)),   # near face (-Z) — fork pocket side
    ("BACK",   (4, 5, 6, 7)),   # far face  (+Z)
    ("TOP",    (0, 1, 5, 4)),   # top    (-Y)
    ("BOTTOM", (3, 2, 6, 7)),   # bottom (+Y)
    ("LEFT",   (0, 3, 7, 4)),   # left   (-X)
    ("RIGHT",  (1, 2, 6, 5)),   # right  (+X)
]


def _correspondence_weights(valid_idx, extrapolated_mask=None,
                            keypoint_weights=None,
                            keypoint_uncertainties=None):
    """Build normalized PnP reliability weights for ``valid_idx``.

    ``keypoint_weights`` are direct confidences (larger is better).
    ``keypoint_uncertainties`` are sigma-like values (smaller is better) and
    are converted with inverse variance.  They are mutually exclusive.
    Extrapolated annotation points retain their historical 0.3 multiplier.

    Returns ``None`` only when no weighting input was supplied, preserving the
    exact legacy OpenCV refinement path.
    """
    if keypoint_weights is not None and keypoint_uncertainties is not None:
        raise ValueError(
            "pass either keypoint_weights or keypoint_uncertainties, not both")
    if (keypoint_weights is None and keypoint_uncertainties is None
            and extrapolated_mask is None):
        return None

    reliability = []
    for i in valid_idx:
        value = 1.0
        if keypoint_weights is not None:
            if i < len(keypoint_weights) and keypoint_weights[i] is not None:
                value = float(keypoint_weights[i])
        elif keypoint_uncertainties is not None:
            if (i < len(keypoint_uncertainties)
                    and keypoint_uncertainties[i] is not None):
                sigma = float(keypoint_uncertainties[i])
                if not np.isfinite(sigma) or sigma < 0:
                    raise ValueError(
                        "keypoint uncertainties must be finite and non-negative")
                value = 1.0 / max(sigma * sigma, 1e-6)
        if not np.isfinite(value) or value < 0:
            raise ValueError("keypoint weights must be finite and non-negative")
        if (extrapolated_mask is not None and i < len(extrapolated_mask)
                and extrapolated_mask[i]):
            value *= 0.3
        reliability.append(value)

    arr = np.asarray(reliability, dtype=np.float64)
    if np.count_nonzero(arr > 0) < 4:
        raise ValueError("at least four detected keypoints need positive weight")

    # Extreme inverse-variance ratios make the normal equations fragile and
    # effectively turn soft uncertainty into accidental hard rejection.
    positive = arr[arr > 0]
    reference = float(np.median(positive))
    if reference > 0:
        arr = np.clip(arr / reference, 0.02, 50.0)
    arr *= len(arr) / max(float(arr.sum()), 1e-12)  # mean reliability = 1
    return arr.tolist()


def _solve_pose_single(kps_2d, K, dims, extrapolated_mask=None, img_shape=None,
                       keypoint_weights=None, keypoint_uncertainties=None):
    """단일 dim PnP — fix v7 weighted scoring + degenerate reject.

    Init candidates:
      (a) IPPE 6 faces (FRONT/BACK/TOP/BOTTOM/LEFT/RIGHT) — coplanar 4 점 클릭된
          모든 face. truncation 시 0/3 외삽 어려운 케이스 (012456) 에서
          RIGHT (1,2,6,5) face 가 핵심 seed. 각 face 당 2 해.
      (b) EPNP / SQPNP — 각 1 해
      (c) IPPE all-valid (n>=4) — 2 해
      (d) Rx180 + manual t / Identity + manual t

    Selection (v7):
      (1) 각 init 마다 24 cube symmetry × LM refine.
      (2) 각 candidate (R, t) 에 pair-wise (LR/TB/FR) 위반 카운트.
      (3) Degenerate reject: cuboid 8-corner screen bbox area 가 image area 의
          1.5% 미만 (≈ 96×72 / 640×480) 이면 candidate 제외 — wireframe 작게
          몰림 버그 (v6 이전 보고된 케이스: 7 missing + 0/3 외삽 → z 4m 채택).
      (4) extrapolated_mask 가 주어지면 weighted reproj 계산: 직접 click = 1.0,
          외삽 = 0.3. 외삽 점은 click 정확도가 낮으므로 selection 시 영향 ↓.
      (5) 사용자 click LR/TB pair 모순 (click_lr_viol ≥ 1 or click_tb_viol ≥ 1) 시
          strict disable, reproj 최소 채택 (사용자 의도 우선).
      (6) 그 외 strict-pass (viol_sum == 0) candidate 중 weighted reproj 최소.
          strict-pass 없으면 fallback (weighted reproj + 100000 * viol_sum 최소).
    """
    kp3d = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    if len(valid_idx) < 4:
        return None
    obj = np.array([kp3d[i] for i in valid_idx], dtype=np.float64)
    img = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)

    # v7 extrapolation weight plus optional learned uncertainty/confidence.
    # With no weighting inputs this is None, retaining the exact legacy path.
    weights = _correspondence_weights(
        valid_idx,
        extrapolated_mask=extrapolated_mask,
        keypoint_weights=keypoint_weights,
        keypoint_uncertainties=keypoint_uncertainties,
    )
    # Backward compatibility: extrapolated_mask historically affected only
    # candidate scoring.  Actual weighted LM is enabled exclusively by one of
    # the new learned reliability inputs.
    refine_weights = weights if (
        keypoint_weights is not None or keypoint_uncertainties is not None
    ) else None

    # v9 (2026-05-26): degenerate cuboid reject threshold —
    #   v7 fixed 1.5% image area (4608px² @ 640x480) 는 far-pallet (small click) 에서
    #   정답 candidate (e.g. 3068px²) 도 reject. v9 = image_area * 0.5% (1536px² @
    #   640x480) — 한 점 collapse (≤30x30=900px²) 는 막으면서 정답 small cuboid 살림.
    #   Dynamic per-click threshold 는 partial-occlusion 케이스 (pallet07, click_bbox
    #   45760px²) 에서 cuboid 가 click 보다 작거나 비슷할 때 정답을 reject 할 수 있어
    #   채택 안 함.
    if img_shape is not None:
        img_area = float(img_shape[0] * img_shape[1])
    else:
        img_area = float(4.0 * K[0, 2] * K[1, 2])
    min_bbox_area = 0.005 * img_area

    inits = []
    # (a) IPPE 6 face seeds (planar) — truncation 시 임의 face seed 활용
    for _name, face in _CUBOID_FACES:
        inits.extend(_seed_from_ippe_face(kps_2d, K, kp3d, list(face)))
    # (b) EPNP / SQPNP on all valid clicks
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R, _ = cv2.Rodrigues(rvec)
                inits.append((R, tvec.flatten()))
        except cv2.error:
            pass
    # (c) IPPE on all valid points (≥ 4) — planar 가정이라 비평면이면 noisy 해
    try:
        ok_n, rvec_list, tvec_list, _ = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for rv, tv in zip(rvec_list, tvec_list):
                if tv[2, 0] > 0:
                    R_ippe, _ = cv2.Rodrigues(rv)
                    inits.append((R_ippe, tv.flatten()))
    except cv2.error:
        pass

    # (d) Rx180 + manual t init / Identity + manual t init
    cx_K, cy_K = K[0, 2], K[1, 2]
    fx_K = K[0, 0]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    img_w = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(img_w, 50.0))
    t_manual = np.array([(mean_u - cx_K) * z_guess / fx_K,
                         (mean_v - cy_K) * z_guess / fx_K,
                         z_guess], dtype=np.float64)
    Rx180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    inits.append((Rx180.copy(), t_manual.copy()))
    inits.append((np.eye(3), t_manual.copy()))

    if not inits:
        return None

    # 24 cube symmetry flip
    flips = []
    for ax_rot_deg in _CUBE_FLIPS_DEG:
        rx = _rot_axis_angle((1, 0, 0), ax_rot_deg[0])
        ry = _rot_axis_angle((0, 1, 0), ax_rot_deg[1])
        rz = _rot_axis_angle((0, 0, 1), ax_rot_deg[2])
        flips.append(rz @ ry @ rx)

    # Click span (degeneracy 가드용)
    click_pts = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    click_bbox_w = float(click_pts[:, 0].max() - click_pts[:, 0].min())
    click_bbox_h = float(click_pts[:, 1].max() - click_pts[:, 1].min())
    click_span = max(click_bbox_w, click_bbox_h, 50.0)
    z_far_limit = 50.0 * fx_K * max(dims) / click_span

    click_lr_viol = _eval_click_lr_viol(kps_2d)
    click_tb_viol = _eval_click_tb_viol(kps_2d)

    candidates = []
    for R0, t0 in inits:
        for F in flips:
            R_init = R0 @ F
            res = _refine_with_init(
                obj, img, K, R_init, t0, weights=refine_weights)
            if res is None:
                continue
            R, t = res
            if t[2] <= 0:
                continue
            if t[2] > z_far_limit:
                continue
            pts_cam_check = (R @ kp3d.T).T + t
            if (pts_cam_check[:, 2] <= 0).any():
                continue
            lrv, tbv, frv, proj_all, _pts_cam = _eval_pair_invariants(R, t, K, kp3d)
            # v7: degenerate cuboid reject — 8 corner screen bbox area 너무 작으면
            # PnP 가 z 큰 작은 cube 채택한 케이스 (wireframe 한 점에 몰림 버그).
            proj_8 = np.array(proj_all[:8], dtype=np.float64)
            bbox_w = float(proj_8[:, 0].max() - proj_8[:, 0].min())
            bbox_h = float(proj_8[:, 1].max() - proj_8[:, 1].min())
            if bbox_w * bbox_h < min_bbox_area:
                continue
            err = _reproj_err_dict(proj_all, valid_idx, kps_2d, weights=weights)
            viol_sum = lrv + tbv + frv
            tilt = _eval_v8_tilt(R)
            candidates.append({
                "err": err,
                "lr_viol": lrv, "tb_viol": tbv, "fr_viol": frv,
                "viol_sum": viol_sum,
                "R": R, "t": t, "proj_all": proj_all,
                "tilt": tilt,
            })

    if not candidates:
        return None

    # v8: gravity-prior hard reject — pallet 이 거의 옆으로 누운 candidate 만 제외.
    # SOFT threshold 는 selection 에는 사용하지 않음 (legit oblique view 도 tilt 낮음).
    # SOFT 위반은 diagnostic flag (v4_warning=True) 로만 GUI 에 노출.
    # 단, 모든 candidate 가 hard reject 면 (사용자가 정말 누운 pallet 클릭) 원본 살림.
    v8_filtered = [c for c in candidates if c["tilt"] >= V8_TILT_HARD_THR]
    if v8_filtered:
        cand_use = v8_filtered
    else:
        cand_use = candidates  # 모두 hard reject — fallback (경고만)

    strict_ok = [c for c in cand_use if c["viol_sum"] == 0]
    if click_lr_viol >= 1 or click_tb_viol >= 1:
        # 사용자 click 자체가 LR/TB 모순 — invariant 강제하면 사용자 클릭과 충돌.
        # reproj 최소 채택, GUI 빨간 경고로 재클릭 안내.
        best = min(cand_use, key=lambda c: c["err"])
        strict_passed = False
    elif strict_ok:
        best = min(strict_ok, key=lambda c: c["err"])
        strict_passed = True
    else:
        # strict-pass candidate 없음 — fallback (viol heavy penalty).
        best = min(cand_use, key=lambda c: c["err"] + 100000.0 * c["viol_sum"])
        strict_passed = False

    rvec, _ = cv2.Rodrigues(best["R"])
    return {
        "R": best["R"], "t": best["t"],
        "rvec": rvec, "tvec": best["t"].reshape(3, 1),
        "reproj_error_px": best["err"],
        "projected_all": best["proj_all"],
        "dims": dims,
        "_v6_lr_viol": best["lr_viol"],
        "_v6_tb_viol": best["tb_viol"],
        "_v6_fr_viol": best["fr_viol"],
        "_v6_viol_sum": best["viol_sum"],
        "_v6_strict_passed": strict_passed,
        "_v6_click_lr_viol": click_lr_viol,
        "_v6_click_tb_viol": click_tb_viol,
        "_v6_n_candidates": len(candidates),
        "_v6_n_strict_ok": len(strict_ok),
        "_v8_tilt": best["tilt"],
        "_v8_n_after_hard_reject": len(cand_use),
        "_v8_hard_reject_fallback": not bool(v8_filtered),
        "_weighted_pnp": refine_weights is not None,
    }


def _compute_perm_v4_local(kp3d_local, proj_all, img_w=None, img_h=None):
    """v4 컨벤션 permutation 계산 (cuboid local frame 기반) — 진단용으로만 보존.

    fix v6 의 solve_pose 는 strict invariant 로 (R, t) 자체를 v4 정합으로 강제하므로
    perm 은 identity 가 정상. 진단용으로 mismatch 발생시 _v4_perm 필드에 기록.

    compute_perm_v4 는 origin frame 의 z 축이 height (gravity-up, 큰 값 = top) 라고
    가정. annotate 의 cuboid local frame 은 Y=down → z_height := -Y_local 로 재배열.
    """
    origin_v4 = np.column_stack([
        kp3d_local[:8, 0],   # x
        kp3d_local[:8, 2],   # z (forward) — pairing 용
        -kp3d_local[:8, 1],  # z_height = -Y_local
    ])
    proj_8 = np.array(proj_all[:8], dtype=np.float64)
    if img_w is not None and img_h is not None and img_w > 0 and img_h > 0:
        proj_8 = proj_8.copy()
        proj_8[:, 0] = np.clip(proj_8[:, 0], 0.0, float(img_w - 1))
        proj_8[:, 1] = np.clip(proj_8[:, 1], 0.0, float(img_h - 1))
    return _compute_perm_v4_z_height(origin_v4, proj_8)


def _check_v4_warning(kps_2d, proj_all_post_perm, pose=None):
    """v6 컨벤션 경고 — 사용자 click LR/TB 모순 OR strict invariant 미통과 OR v8 tilt 누움.

    True 인 케이스:
      - click_lr_viol ≥ 1 또는 click_tb_viol ≥ 1  (사용자가 v6 정의와 어긋나게 클릭)
      - viol_sum > 0                                 (strict mode 실패, fallback 사용)
      - v8 tilt < V8_TILT_SOFT_THR                  (pallet 이 32° 이상 누움 — 비정상)

    False = strict pair-wise invariant 모두 통과 + 사용자 click 도 일관 + upright.
    """
    if pose is None:
        return False
    tilt = pose.get("_v8_tilt", 1.0)
    return bool(pose.get("_v6_click_lr_viol", 0)
                or pose.get("_v6_click_tb_viol", 0)
                or pose.get("_v6_viol_sum", 0) > 0
                or tilt < V8_TILT_SOFT_THR)


def _apply_perm_to_projected(proj_all, perm):
    """projected_all (9, [u,v]) 에 perm 적용. perm[new]=old, idx 8(centroid) 보존."""
    result = []
    for i in range(9):
        old = perm[i]
        if 0 <= old < len(proj_all):
            result.append(list(proj_all[old]))
        else:
            result.append([-1.0, -1.0])
    return result


_CUBOID_AXIS_EDGES = {
    "width": tuple(LR_PAIRS),
    "height": tuple(TB_PAIRS),
    "depth": tuple(FR_PAIRS),
}


def assess_keypoint_topology(kps_2d, min_corners=7,
                             reject_missing_structural_edge=True,
                             min_complete_edges_per_axis=2):
    """Assess whether detected cuboid corners safely constrain PnP.

    Only corner channels 0..7 count; centroid channel 8 cannot replace a
    missing structural corner.  The conservative defaults accept 7/8 or 8/8
    corners.  Lowering ``min_corners`` to 6 is supported for experiments, but
    a pair of missing endpoints that removes an entire cuboid edge and weak
    axis coverage are still reported separately.

    Returns a JSON-friendly diagnostic dictionary with stable reason codes:
    ``insufficient_corners``, ``missing_structural_edge``, and
    ``insufficient_axis_coverage``.
    """
    detected = {
        i for i in range(min(8, len(kps_2d))) if kps_2d[i] is not None
    }
    missing = sorted(set(range(8)) - detected)
    complete_by_axis = {}
    missing_edges = []
    for axis, edges in _CUBOID_AXIS_EDGES.items():
        complete_by_axis[axis] = sum(
            1 for a, b in edges if a in detected and b in detected)
        for a, b in edges:
            if a not in detected and b not in detected:
                missing_edges.append({"axis": axis, "edge": [a, b]})

    insufficient_axes = sorted(
        axis for axis, count in complete_by_axis.items()
        if count < int(min_complete_edges_per_axis))
    reasons = []
    if len(detected) < int(min_corners):
        reasons.append("insufficient_corners")
    if reject_missing_structural_edge and missing_edges:
        reasons.append("missing_structural_edge")
    if insufficient_axes:
        reasons.append("insufficient_axis_coverage")

    return {
        "accepted": not reasons,
        "reason": reasons[0] if reasons else "ok",
        "reasons": reasons,
        "n_corners": len(detected),
        "detected_corners": sorted(detected),
        "missing_corners": missing,
        "missing_structural_edges": missing_edges,
        "complete_edges_per_axis": complete_by_axis,
        "insufficient_axes": insufficient_axes,
        "min_corners": int(min_corners),
        "min_complete_edges_per_axis": int(min_complete_edges_per_axis),
    }


def _validated_dims(dims):
    """Return a positive finite ``(W, D, H)`` tuple."""
    values = np.asarray(dims, dtype=np.float64).reshape(-1)
    if len(values) != 3 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("dims must contain three positive finite values (W, D, H)")
    return tuple(float(v) for v in values)


def _finalize_pose_candidate(pose, kps_2d, K, extrapolated_mask):
    """Attach public projection/reprojection diagnostics to one W/D pose."""
    pose = dict(pose)
    pose["_selection_reproj_error_px"] = float(pose["reproj_error_px"])
    R, t = pose["R"], pose["t"]
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    proj_all = project_3d(kp3d, R, t, K)

    img_w_est = int(round(2.0 * K[0, 2]))
    img_h_est = int(round(2.0 * K[1, 2]))
    try:
        perm = _compute_perm_v4_local(kp3d, proj_all, img_w_est, img_h_est)
    except Exception:
        perm = None

    real_valid = [
        i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None
    ]
    if extrapolated_mask is not None:
        click_only = [
            i for i in real_valid
            if not (i < len(extrapolated_mask) and extrapolated_mask[i])
        ]
        report_idx = click_only if click_only else real_valid
    else:
        report_idx = real_valid
    errors = []
    for i in report_idx:
        u, v = proj_all[i]
        if u == -1.0 and v == -1.0:
            continue
        errors.append(float(np.hypot(
            u - kps_2d[i][0], v - kps_2d[i][1])))
    if errors:
        pose["reproj_error_px"] = float(np.mean(errors))
    pose["projected_all"] = proj_all
    pose["v4_perm"] = perm
    pose["v4_warning"] = _check_v4_warning(kps_2d, proj_all, pose=pose)
    return pose


def solve_pose_candidates(kps_2d, K, dims=None, extrapolated_mask=None,
                          img_shape=None, keypoint_weights=None,
                          keypoint_uncertainties=None, auto_swap_dims=True):
    """Return the complete as-given and W/D-swapped PnP candidates.

    Unlike historical ``solve_pose``, an explicit ``dims=(W,D,H)`` is now
    honored.  Omitting it still reads the module-level ``PALLET_DIMS`` at call
    time, preserving ``annotate_wood.py`` and GUI behavior that intentionally
    override that global.

    ``keypoint_weights`` (larger is better) and
    ``keypoint_uncertainties`` (sigma, smaller is better) are mutually
    exclusive and affect both LM refinement and candidate scoring.
    """
    base_dims = _validated_dims(PALLET_DIMS if dims is None else dims)
    hypotheses = [("as_given", base_dims)]
    swapped_dims = (base_dims[1], base_dims[0], base_dims[2])
    if auto_swap_dims and not np.allclose(base_dims, swapped_dims):
        hypotheses.append(("swapped", swapped_dims))

    candidates = []
    for hypothesis, candidate_dims in hypotheses:
        pose = _solve_pose_single(
            kps_2d, K, candidate_dims,
            extrapolated_mask=extrapolated_mask,
            img_shape=img_shape,
            keypoint_weights=keypoint_weights,
            keypoint_uncertainties=keypoint_uncertainties,
        )
        if pose is None:
            continue
        pose["_wd_hypothesis"] = hypothesis
        candidates.append(_finalize_pose_candidate(
            pose, kps_2d, K, extrapolated_mask))
    return candidates


def _pose_rank_key(pose):
    """Historical candidate preference expressed as a sortable key."""
    strict = bool(pose.get("_v6_strict_passed", False))
    error = float(pose.get("_selection_reproj_error_px",
                           pose["reproj_error_px"]))
    if strict:
        return (0, 0, error)
    return (1, int(pose.get("_v6_viol_sum", 0)), error)


def _pose_candidate_summary(pose):
    return {
        "hypothesis": pose.get("_wd_hypothesis", "as_given"),
        "dims": tuple(float(v) for v in pose["dims"]),
        "selection_reproj_error_px": float(
            pose.get("_selection_reproj_error_px", pose["reproj_error_px"])),
        "reported_reproj_error_px": float(pose["reproj_error_px"]),
        "strict_passed": bool(pose.get("_v6_strict_passed", False)),
        "violation_sum": int(pose.get("_v6_viol_sum", 0)),
        "tilt": float(pose.get("_v8_tilt", 0.0)),
    }


def _select_pose_candidate(candidates, wd_ambiguity_abs_px=0.5,
                           wd_ambiguity_rel=0.05,
                           wd_as_given_prob=None,
                           wd_prior_min_confidence=0.65):
    """Select a pose and attach explicit W/D ambiguity/prior diagnostics.

    A learned W/D head is permitted to break a tie only after reprojection and
    invariant ranking establish that both candidates occupy the same quality
    tier *and* their score gap is inside the ambiguity threshold.  It can
    therefore never overturn a geometrically clear solution.
    """
    if not candidates:
        return None
    ordered = sorted(candidates, key=_pose_rank_key)
    legacy_best = ordered[0]
    selected = legacy_best
    summaries = [_pose_candidate_summary(pose) for pose in candidates]

    ambiguous = False
    gap = None
    ratio = None
    threshold = None
    competing = False
    if len(ordered) >= 2:
        alternative = ordered[1]
        best_strict = bool(legacy_best.get("_v6_strict_passed", False))
        alt_strict = bool(alternative.get("_v6_strict_passed", False))
        competing = best_strict == alt_strict
        if not best_strict:
            competing = competing and (
                int(legacy_best.get("_v6_viol_sum", 0))
                == int(alternative.get("_v6_viol_sum", 0)))
        best_error = float(legacy_best.get(
            "_selection_reproj_error_px", legacy_best["reproj_error_px"]))
        alt_error = float(alternative.get(
            "_selection_reproj_error_px", alternative["reproj_error_px"]))
        gap = max(0.0, alt_error - best_error)
        ratio = (1.0 if abs(best_error) < 1e-12 and abs(alt_error) < 1e-12
                 else alt_error / max(best_error, 1e-12))
        threshold = max(
            float(wd_ambiguity_abs_px),
            float(wd_ambiguity_rel) * max(best_error, 1.0),
        )
        ambiguous = bool(competing and gap <= threshold)

    prior_probability = None
    prior_confidence = None
    prior_used = False
    if wd_as_given_prob is not None:
        prior_probability = float(wd_as_given_prob)
        if not np.isfinite(prior_probability) or not 0.0 <= prior_probability <= 1.0:
            raise ValueError("wd_as_given_prob must be finite and in [0, 1]")
        min_confidence = float(wd_prior_min_confidence)
        if not 0.5 <= min_confidence <= 1.0:
            raise ValueError("wd_prior_min_confidence must be in [0.5, 1.0]")
        prior_confidence = max(prior_probability, 1.0 - prior_probability)
        if (ambiguous and prior_confidence >= min_confidence
                and abs(prior_probability - 0.5) > 1e-12):
            preferred = "as_given" if prior_probability > 0.5 else "swapped"
            preferred_candidates = [
                pose for pose in ordered
                if pose.get("_wd_hypothesis") == preferred
            ]
            if preferred_candidates:
                selected = preferred_candidates[0]
                prior_used = True

    best = dict(selected)
    best["_wd_ambiguous"] = ambiguous
    best["_wd_competing_quality_tier"] = competing
    best["_wd_score_gap_px"] = gap
    best["_wd_error_ratio"] = ratio
    best["_wd_ambiguity_threshold_px"] = threshold
    best["_wd_candidates"] = summaries
    best["_wd_n_candidates"] = len(candidates)
    best["_wd_as_given_prob"] = prior_probability
    best["_wd_prior_confidence"] = prior_confidence
    best["_wd_prior_min_confidence"] = float(wd_prior_min_confidence)
    best["_wd_prior_used"] = prior_used
    best["_wd_prior_resolved_ambiguity"] = bool(ambiguous and prior_used)
    best["_wd_legacy_hypothesis"] = legacy_best.get("_wd_hypothesis")
    return best


def solve_pose(kps_2d, K, dims=None, extrapolated_mask=None, img_shape=None,
               keypoint_weights=None, keypoint_uncertainties=None,
               auto_swap_dims=True, wd_ambiguity_abs_px=0.5,
               wd_ambiguity_rel=0.05, wd_as_given_prob=None,
               wd_prior_min_confidence=0.65):
    """Backward-compatible PnP with explicit W/D and uncertainty diagnostics.

    Existing callers still receive the selected pose dictionary (or ``None``)
    and are not rejected for ambiguity.  New callers that need fail-closed
    behavior should use :func:`solve_pose_safe` and check ``result["accepted"]``.
    """
    candidates = solve_pose_candidates(
        kps_2d, K, dims=dims,
        extrapolated_mask=extrapolated_mask,
        img_shape=img_shape,
        keypoint_weights=keypoint_weights,
        keypoint_uncertainties=keypoint_uncertainties,
        auto_swap_dims=auto_swap_dims,
    )
    return _select_pose_candidate(
        candidates,
        wd_ambiguity_abs_px=wd_ambiguity_abs_px,
        wd_ambiguity_rel=wd_ambiguity_rel,
        wd_as_given_prob=wd_as_given_prob,
        wd_prior_min_confidence=wd_prior_min_confidence,
    )


def _finite_convex_hull_area(points):
    """Return a positive finite 2-D convex-hull area, otherwise ``None``."""
    finite = []
    for point in points:
        if point is None:
            continue
        try:
            xy = np.asarray(point, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(xy) < 2 or not np.isfinite(xy[:2]).all():
            continue
        finite.append(xy[:2])
    if len(finite) < 3:
        return None
    try:
        hull = cv2.convexHull(np.asarray(finite, dtype=np.float32))
        area = float(cv2.contourArea(hull))
    except cv2.error:
        return None
    if not np.isfinite(area) or area <= 0.0:
        return None
    return area


def _projection_to_raw_area_ratio(pose, kps_2d):
    """Compare the selected PnP footprint with pre-PnP detected corners.

    The centroid channel is deliberately excluded.  Six finite raw cuboid
    corners are required so a sparse/degenerate detection cannot become a new
    rejection path merely because its hull is under-constrained.
    """
    if pose is None:
        return None

    raw_corners = []
    for point in list(kps_2d)[:8]:
        if point is None:
            continue
        try:
            xy = np.asarray(point, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(xy) >= 2 and np.isfinite(xy[:2]).all():
            raw_corners.append(xy[:2])
    if len(raw_corners) < 6:
        return None

    projected_all = pose.get("projected_all")
    if projected_all is None:
        return None
    raw_area = _finite_convex_hull_area(raw_corners)
    projected_area = _finite_convex_hull_area(list(projected_all)[:8])
    if raw_area is None or projected_area is None:
        return None
    ratio = projected_area / raw_area
    return float(ratio) if np.isfinite(ratio) and ratio > 0.0 else None


def solve_pose_safe(kps_2d, K, dims=None, extrapolated_mask=None,
                    img_shape=None, keypoint_weights=None,
                    keypoint_uncertainties=None, auto_swap_dims=True,
                    min_corners=7, reject_missing_structural_edge=True,
                    min_complete_edges_per_axis=2, require_strict=True,
                    reject_wd_ambiguity=True, wd_ambiguity_abs_px=0.5,
                    wd_ambiguity_rel=0.05, wd_as_given_prob=None,
                    wd_prior_min_confidence=0.65,
                    wd_prior_resolves_ambiguity=True,
                    max_reproj_error_px=None,
                    projection_contraction_threshold=0.75):
    """Fail-closed single-image PnP wrapper with structured reason codes.

    The return value is always a dictionary.  Consumers **must** check
    ``accepted`` before using ``pose``; rejected W/D cases retain the selected
    pose and both full candidates for debugging, but are not safe outputs.
    """
    contraction_threshold = float(projection_contraction_threshold)
    if (not np.isfinite(contraction_threshold)
            or not 0.0 <= contraction_threshold <= 1.0):
        raise ValueError(
            "projection_contraction_threshold must be finite and in [0, 1]")

    topology = assess_keypoint_topology(
        kps_2d,
        min_corners=min_corners,
        reject_missing_structural_edge=reject_missing_structural_edge,
        min_complete_edges_per_axis=min_complete_edges_per_axis,
    )
    if not topology["accepted"]:
        return {
            "accepted": False,
            "reason": topology["reason"],
            "reasons": list(topology["reasons"]),
            "pose": None,
            "candidates": [],
            "topology": topology,
            "_projection_to_raw_area_ratio": None,
            "_projection_contraction_threshold": contraction_threshold,
        }

    candidates = solve_pose_candidates(
        kps_2d, K, dims=dims,
        extrapolated_mask=extrapolated_mask,
        img_shape=img_shape,
        keypoint_weights=keypoint_weights,
        keypoint_uncertainties=keypoint_uncertainties,
        auto_swap_dims=auto_swap_dims,
    )
    pose = _select_pose_candidate(
        candidates,
        wd_ambiguity_abs_px=wd_ambiguity_abs_px,
        wd_ambiguity_rel=wd_ambiguity_rel,
        wd_as_given_prob=wd_as_given_prob,
        wd_prior_min_confidence=wd_prior_min_confidence,
    )
    area_ratio = _projection_to_raw_area_ratio(pose, kps_2d)
    if pose is not None:
        pose["_projection_to_raw_area_ratio"] = area_ratio
        pose["_projection_contraction_threshold"] = contraction_threshold

    reasons = []
    if pose is None:
        reasons.append("pnp_failed")
    else:
        if require_strict and not pose.get("_v6_strict_passed", False):
            reasons.append("invariant_violation")
        ambiguity_resolved = bool(
            wd_prior_resolves_ambiguity
            and pose.get("_wd_prior_resolved_ambiguity", False))
        if (reject_wd_ambiguity and pose.get("_wd_ambiguous", False)
                and not ambiguity_resolved):
            reasons.append("wd_ambiguous")
        if (max_reproj_error_px is not None
                and pose["reproj_error_px"] > float(max_reproj_error_px)):
            reasons.append("reprojection_error")
        if (area_ratio is not None
                and area_ratio < contraction_threshold):
            reasons.append("projection_contraction")

    return {
        "accepted": not reasons,
        "reason": reasons[0] if reasons else "ok",
        "reasons": reasons,
        "pose": pose,
        "candidates": candidates,
        "topology": topology,
        "_projection_to_raw_area_ratio": area_ratio,
        "_projection_contraction_threshold": contraction_threshold,
    }


# ─── MANIPULATE 모드 ──────────────────────────────────────────────────────────

def euler_to_R(yaw_deg, pitch_deg, roll_deg):
    """yaw(Y) → pitch(X) → roll(Z) 순서 회전 행렬."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def apply_manip(state, dx=0, dy=0, dz=0, dyaw=0, dpitch=0, droll=0):
    """locked_pose 에 translation/rotation 추가."""
    if state.locked_pose is None:
        return
    R = state.locked_pose["R"]
    t = state.locked_pose["t"]
    t = t + np.array([dx, dy, dz], dtype=np.float64)
    dR = euler_to_R(dyaw, dpitch, droll)
    R = R @ dR
    state.locked_pose["R"] = R
    state.locked_pose["t"] = t
    state.dirty = True


def pose_from_locked(state, K, dims=PALLET_DIMS):
    """locked_pose 로부터 pose dict 재구성 (projected_all + reproj_error 포함).
    fix v6 strict invariants 진단 동시 계산 — 위반시 GUI 경고."""
    if state.locked_pose is None:
        return None
    R = state.locked_pose["R"]
    t = state.locked_pose["t"]
    kp3d = make_pallet_keypoints_3d(*dims)
    proj_all = project_3d(kp3d, R, t, K)
    rvec, _ = cv2.Rodrigues(R)

    img_w_est = int(round(2.0 * K[0, 2]))
    img_h_est = int(round(2.0 * K[1, 2]))
    try:
        perm = _compute_perm_v4_local(kp3d, proj_all, img_w_est, img_h_est)
    except Exception:
        perm = None

    # v6 pair-wise invariants (진단)
    lrv, tbv, frv, _proj_all2, _pts_cam = _eval_pair_invariants(R, t, K, kp3d)
    click_lr_viol = _eval_click_lr_viol(state.kps_2d)
    click_tb_viol = _eval_click_tb_viol(state.kps_2d)
    viol_sum = lrv + tbv + frv
    tilt = _eval_v8_tilt(R)

    diag = {
        "_v6_lr_viol": lrv,
        "_v6_tb_viol": tbv,
        "_v6_fr_viol": frv,
        "_v6_viol_sum": viol_sum,
        "_v6_click_lr_viol": click_lr_viol,
        "_v6_click_tb_viol": click_tb_viol,
        "_v8_tilt": tilt,
    }
    v4_warning = _check_v4_warning(state.kps_2d, proj_all, pose=diag)

    errs = []
    for i in range(min(9, len(state.kps_2d))):
        if state.kps_2d[i] is None:
            continue
        # v7: project_3d sentinel = (-1, -1) — 그 외 u<0 은 valid image-out projection
        if proj_all[i][0] == -1.0 and proj_all[i][1] == -1.0:
            continue
        du = proj_all[i][0] - state.kps_2d[i][0]
        dv = proj_all[i][1] - state.kps_2d[i][1]
        errs.append(float(np.hypot(du, dv)))
    return {
        "R": R, "t": t, "rvec": rvec, "tvec": t.reshape(3, 1),
        "reproj_error_px": float(np.mean(errs)) if errs else 0.0,
        "projected_all": proj_all,
        "dims": dims,
        "v4_perm": perm,
        "v4_warning": v4_warning,
        "_v6_lr_viol": lrv,
        "_v6_tb_viol": tbv,
        "_v6_fr_viol": frv,
        "_v6_viol_sum": viol_sum,
        "_v6_click_lr_viol": click_lr_viol,
        "_v6_click_tb_viol": click_tb_viol,
        "_v6_strict_passed": (viol_sum == 0 and click_lr_viol == 0 and click_tb_viol == 0),
        "_v8_tilt": tilt,
    }


# ─── TWO-LINE 모드 ────────────────────────────────────────────────────────────

# ─── PARALLELOGRAM 외삽 ──────────────────────────────────────────────────────

# v6 cuboid 6 face — 각 face 의 4 corner (반시계/시계 무관, "대각선" 만 일관).
# face 의 임의 3 corner 알면 4 번째 = parallelogram law 로 외삽:
#   4th = corner_opp_to_missing + (corner_adj1 - corner_diag) + (corner_adj2 - corner_diag)
# 더 간단히: face = (a, b, c, d) 가 cyclic 순서 (a→b→c→d→a) 인 경우
#   d = a + (c - b)   (b 의 대각선은 d)
#   c = b + (d - a)
#   b = a + (c - d)
#   a = b + (d - c)
# v6 face 정의가 cyclic 순서임 (FRONT=(0,1,2,3), TOP=(0,1,5,4) 등) — 위 공식 그대로 적용.
_PARALLELOGRAM_FACES_CYCLIC = [
    ("FRONT",  (0, 1, 2, 3)),
    ("BACK",   (4, 5, 6, 7)),
    ("TOP",    (0, 1, 5, 4)),
    ("BOTTOM", (3, 2, 6, 7)),
    ("LEFT",   (0, 3, 7, 4)),
    ("RIGHT",  (1, 2, 6, 5)),
]


def parallelogram_extrapolate(kps_2d, missing_idx):
    """missing_idx (0..7) 의 위치를 어떤 face 의 3 corner 로부터 parallelogram 외삽.

    kps_2d: length≥8 list, 각 [u,v] or None.
    missing_idx 가 포함된 face 중 나머지 3 corner 가 모두 클릭된 face 가 있으면
    그 face 의 cyclic 순서로 외삽하여 [u, v] 반환. 후보 face 가 여러 개면 평균.
    가능한 face 없으면 None.

    Cyclic 가정 (FRONT=(0,1,2,3) → a-b-c-d-a 사이클):
      missing == a:  a = b + (d - c)
      missing == b:  b = a + (c - d)
      missing == c:  c = b + (d - a)   = d + (b - a)
      missing == d:  d = a + (c - b)

    반환:
      result_uv (list of 2 float) — 외삽 좌표
      face_name (str) — 사용한 face (debug)
      face_indices (tuple) — 사용한 4 corner (debug)
    실패 시 (None, None, None).
    """
    candidates = []
    for fname, face in _PARALLELOGRAM_FACES_CYCLIC:
        if missing_idx not in face:
            continue
        # 나머지 3 corner 모두 클릭됐는지
        other_3 = [i for i in face if i != missing_idx]
        if not all(i < len(kps_2d) and kps_2d[i] is not None for i in other_3):
            continue
        # cyclic 위치 찾기 — face = (a, b, c, d)
        a, b, c, d = face
        kp = lambda i: np.array(kps_2d[i], dtype=np.float64)
        if missing_idx == a:
            pt = kp(b) + (kp(d) - kp(c))
        elif missing_idx == b:
            pt = kp(a) + (kp(c) - kp(d))
        elif missing_idx == c:
            pt = kp(b) + (kp(d) - kp(a))
        elif missing_idx == d:
            pt = kp(a) + (kp(c) - kp(b))
        else:
            continue
        candidates.append((pt, fname, face))
    if not candidates:
        return None, None, None
    # 여러 face 가능시 평균 + 첫 face 라벨/idx 리턴
    avg = np.mean([c[0] for c in candidates], axis=0)
    return [float(avg[0]), float(avg[1])], candidates[0][1], candidates[0][2]


def line_intersection(p1, p2, p3, p4):
    """두 line (P1-P2, P3-P4) 의 교점. None 이면 평행."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return [float(ix), float(iy)]
