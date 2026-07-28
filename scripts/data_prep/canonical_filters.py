"""Canonical Geometric Filters — 무차원 비율 기반, 데이터셋/해상도 불변.

Core 3 filters:
  A: Flip-equivariant 2D keypoint consistency (normalized)
  B: Visibility-aware structural coverage + non-collinearity
  C: Normalized leave-one-out PnP stability

Optional priors:
  D1: Depth range
  D2: Tilt range

모든 임계값은 무차원 비율이므로 카메라 거리/해상도/시점과 무관.
"""

import cv2
import numpy as np


# ── Utilities ───────────────────────────────────────────────────────────

def projected_diagonal(reproj_8pts):
    """재투영된 8개 점의 대각선 길이 (max pairwise distance)."""
    max_dist = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(reproj_8pts[i] - reproj_8pts[j])
            if d > max_dist:
                max_dist = d
    return max_dist


def convex_hull_area(pts_2d):
    """2D 점들의 convex hull 면적. 3점 미만이면 0."""
    if len(pts_2d) < 3:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        return ConvexHull(pts_2d).volume  # 2D에서 volume = area
    except Exception:
        return 0.0


def covariance_anisotropy(pts_2d):
    """2D 점들의 공분산 이방성 (λ_min / λ_max). 1=등방, 0=일직선."""
    if len(pts_2d) < 3:
        return 0.0
    pts = np.array(pts_2d, dtype=np.float64)
    cov = np.cov(pts.T)  # (2, 2)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.abs(eigvals))
    if eigvals[1] < 1e-8:
        return 0.0
    return float(eigvals[0] / eigvals[1])


# ── Filter A: Normalized Flip-Equivariant Consistency ──────────────────

FLIP_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]


def filter_A(model, img_bgr, pred_kps_belief, device, pnp_solver, R, t,
             tau_A=0.05):
    """Flip consistency normalized by projected diagonal.

    s_A = (1/D) * median_i ||p_i - T^{-1}(p̂_i^flip)||
    D = projected cuboid diagonal

    Args:
        tau_A: 무차원 임계값 (default 0.05 = cuboid 대각선의 5%)
    Returns:
        passed: bool, score: float (낮을수록 좋음)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Projected diagonal
    reproj = pnp_solver.reproject(R, t)[:8]
    D = projected_diagonal(reproj)
    if D < 1e-6:
        return False, float("inf")

    # Flip inference
    import torch
    img_flip = cv2.flip(img_bgr, 1)
    img_rgb = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (400, 400))  # 학습 입력(400×400)과 일치
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out_bel, _ = model(tensor)
    belief_flip = out_bel[-1][0].cpu().numpy()

    from visualize_inference import extract_keypoints
    kps_flip = extract_keypoints(belief_flip, 0.1)
    bh, bw = belief_flip.shape[1], belief_flip.shape[2]

    # Original keypoints (belief coords)
    kp_orig = []
    for kp in pred_kps_belief[:8]:
        kp_orig.append(np.array([kp[0], kp[1]]) if kp is not None else None)

    # Flip → unflip + symmetric remap
    kp_flip_back = [None] * 8
    for idx, kp in enumerate(kps_flip[:8]):
        if kp is not None:
            kp_flip_back[idx] = np.array([bw - kp[0], kp[1]])

    kp_flip_remapped = [None] * 8
    for a, b in FLIP_PAIRS:
        kp_flip_remapped[a] = kp_flip_back[b]
        kp_flip_remapped[b] = kp_flip_back[a]

    # Compute normalized errors
    errors = []
    for i in range(8):
        if kp_orig[i] is None or kp_flip_remapped[i] is None:
            continue
        # belief coords → pixel coords (scale by img size / belief size)
        err_belief = np.linalg.norm(kp_orig[i] - kp_flip_remapped[i])
        # Normalize: belief space diagonal ≈ D * (bw / img_w)
        errors.append(err_belief)

    if len(errors) < 3:
        return False, float("inf")

    # Scale belief-space error to pixel space, then normalize by D
    # belief map is 50x50, image is 448x448, but D is in original pixel space
    # belief_to_pixel ≈ img_size / belief_size
    scale = 448.0 / bw
    median_err_px = float(np.median(errors)) * scale
    # Now normalize by projected diagonal (in original image space)
    # D is from reproj in original resolution, need to scale to 448
    h_orig, w_orig = img_bgr.shape[:2]
    D_448 = D * 448.0 / max(w_orig, 1)

    score = median_err_px / D_448 if D_448 > 1e-6 else float("inf")
    return score < tau_A, score


# ── Filter B: Visible Structural Support ───────────────────────────────
#
# B = B1(span) ∧ B2(endpoint support) ∧ B3(non-collinearity)
#
# "검출된 점이 pose 전체를 지지하는가"를 판단.
# 핵심: keypoint 개수가 아니라, 양쪽에 증거가 있는지.

def _visible_principal_axis(reproj_vis):
    """Visible reprojected points의 장축 방향 벡터."""
    pts = np.array(reproj_vis, dtype=np.float64)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 장축 = 최대 고유값에 대응하는 고유벡터
    return eigvecs[:, np.argmax(eigvals)]


def _is_in_frame(pt, img_w, img_h, margin=10):
    """점이 이미지 프레임 안에 있는지 (margin 포함)."""
    return margin < pt[0] < img_w - margin and margin < pt[1] < img_h - margin


def filter_B(kps_orig, pnp_solver, R, t,
             tau_span=0.35, tau_end=0.10, tau_nc=0.02,
             min_kps=4, img_size=(640, 480)):
    """Visible structural support: span + endpoint + non-collinearity.

    B1. Visible span ratio: 검출 점이 장축 방향으로 얼마나 커버하는지
    B2. Two-sided endpoint support: 양쪽 끝이 검출 점으로 지지되는지
    B3. Non-collinearity: 일직선 퇴화 방지

    모든 metric은 무차원 비율 — 데이터셋/해상도 불변.

    Args:
        tau_span: 최소 span ratio (default 0.35)
        tau_end: 최대 endpoint 거리 비율 (default 0.10)
        tau_nc: 최소 non-collinearity (default 0.02)
        min_kps: 최소 감지 keypoint 수
        img_size: (W, H) 이미지 크기 (endpoint 프레임 검사용)
    Returns:
        passed: bool, details: dict
    """
    detected_idx = []
    detected_2d = []
    for i, p in enumerate(kps_orig[:8]):
        if p is not None:
            detected_idx.append(i)
            detected_2d.append([float(p[0]), float(p[1])])

    if len(detected_idx) < min_kps:
        return False, {"span": 0, "d_left": 1, "d_right": 1, "nc": 0}

    detected_2d = np.array(detected_2d, dtype=np.float64)
    reproj_all = pnp_solver.reproject(R, t)[:8]
    D_vis = projected_diagonal(reproj_all)
    if D_vis < 1e-6:
        return False, {"span": 0, "d_left": 1, "d_right": 1, "nc": 0}

    # ── B1: Visible span ratio ──
    # 장축 방향으로 detected vs reproj의 span 비율
    u = _visible_principal_axis(reproj_all)  # 장축 단위벡터

    proj_reproj = reproj_all @ u  # (8,) scalar projections
    proj_det = detected_2d @ u    # (N,)

    span_reproj = proj_reproj.max() - proj_reproj.min()
    span_det = proj_det.max() - proj_det.min() if len(proj_det) > 1 else 0

    span_ratio = span_det / span_reproj if span_reproj > 1e-6 else 0

    # ── B2: Two-sided endpoint support ──
    # 재투영 장축의 양 끝점에서 가장 가까운 detected kp 거리
    idx_left = np.argmin(proj_reproj)
    idx_right = np.argmax(proj_reproj)
    e_left = reproj_all[idx_left]   # 장축 왼쪽 끝 재투영점
    e_right = reproj_all[idx_right]  # 장축 오른쪽 끝 재투영점

    img_w, img_h = img_size

    # 각 끝점에서 가장 가까운 detected kp 거리 (D_vis로 normalize)
    d_left = float(np.min(np.linalg.norm(detected_2d - e_left, axis=1))) / D_vis
    d_right = float(np.min(np.linalg.norm(detected_2d - e_right, axis=1))) / D_vis

    # 프레임 밖 끝점은 면제
    left_in_frame = _is_in_frame(e_left, img_w, img_h)
    right_in_frame = _is_in_frame(e_right, img_w, img_h)

    endpoint_ok = True
    if left_in_frame and d_left > tau_end:
        endpoint_ok = False
    if right_in_frame and d_right > tau_end:
        endpoint_ok = False

    # ── B3: Non-collinearity ──
    nc = covariance_anisotropy(detected_2d)

    details = {
        "span": float(span_ratio),
        "d_left": float(d_left),
        "d_right": float(d_right),
        "left_in_frame": left_in_frame,
        "right_in_frame": right_in_frame,
        "nc": float(nc),
    }

    passed = (span_ratio > tau_span and endpoint_ok and nc > tau_nc)
    return passed, details


# ── Filter C: Normalized LOO PnP Stability ─────────────────────────────

def filter_C(kps_orig, pnp_solver, R, t,
             tau_C=0.05, min_kps=5, sigmas=None):
    """Normalized leave-one-out PnP stability.

    한 점씩 빼고 PnP → 빠진 점의 reproj error → projected diagonal로 normalize.
    s_C = median(LOO reproj errors) / D

    If sigmas provided (weighted mode):
      - LOO error weighted by 1/sigma^2 (confident points' errors matter more)
      - score = weighted_median / D

    Args:
        tau_C: 무차원 임계값 (default 0.05 = cuboid 대각선의 5%)
        min_kps: LOO를 위한 최소 감지 keypoint 수 (최소 5: 4로 PnP + 1 검증)
        sigmas: optional list of 9 per-keypoint uncertainties.
    Returns:
        passed: bool, score: float (낮을수록 좋음)
    """
    # Collect detected keypoints
    detected_idx = []
    detected_2d = []
    detected_sigma = []
    for i, p in enumerate(kps_orig[:8]):
        if p is not None:
            detected_idx.append(i)
            detected_2d.append([float(p[0]), float(p[1])])
            if sigmas is not None and i < len(sigmas) and sigmas[i] is not None:
                detected_sigma.append(float(sigmas[i]))
            else:
                detected_sigma.append(1.0)

    if len(detected_idx) < min_kps:
        return False, float("inf")

    detected_2d = np.array(detected_2d, dtype=np.float64)
    detected_sigma = np.array(detected_sigma, dtype=np.float64)
    kp3d_all = pnp_solver.keypoints_3d[:8].astype(np.float64)
    kp3d_detected = kp3d_all[detected_idx]

    # Projected diagonal for normalization
    reproj_all = pnp_solver.reproject(R, t)[:8]
    D = projected_diagonal(reproj_all)
    if D < 1e-6:
        return False, float("inf")

    # LOO: 감지된 점 중에서만 LOO
    loo_errors = []
    loo_weights = []
    for leave_idx in range(len(detected_idx)):
        # Remaining points
        mask = [j for j in range(len(detected_idx)) if j != leave_idx]
        if len(mask) < 4:
            continue

        pts_2d_remain = detected_2d[mask]
        pts_3d_remain = kp3d_detected[mask]

        success, rvec, tvec = cv2.solvePnP(
            pts_3d_remain, pts_2d_remain,
            pnp_solver.camera_matrix, None, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            continue

        # Reproject the left-out point
        left_3d = kp3d_detected[leave_idx].reshape(1, 3)
        reproj_left, _ = cv2.projectPoints(
            left_3d, rvec, tvec, pnp_solver.camera_matrix, None
        )
        reproj_left = reproj_left.reshape(2)
        actual_left = detected_2d[leave_idx]

        err = np.linalg.norm(reproj_left - actual_left)
        loo_errors.append(err)
        # Weight: confident point (small sigma) gets higher weight
        w = 1.0 / (detected_sigma[leave_idx] ** 2 + 1e-4)
        loo_weights.append(w)

    if not loo_errors:
        return False, float("inf")

    if sigmas is not None:
        # Weighted median: sort by error, accumulate weights, find 50% crossing
        errors_arr = np.array(loo_errors)
        weights_arr = np.array(loo_weights)
        sorted_idx = np.argsort(errors_arr)
        cum_w = np.cumsum(weights_arr[sorted_idx])
        half = cum_w[-1] / 2.0
        median_idx = np.searchsorted(cum_w, half)
        median_idx = min(median_idx, len(errors_arr) - 1)
        score = float(errors_arr[sorted_idx[median_idx]]) / D
    else:
        score = float(np.median(loo_errors)) / D

    return score < tau_C, score


# ── Filter D: Conditional Diagonal Incidence ──────────────────────────

# Cuboid face diagonals (pairs of opposite corners on each face)
FACE_DIAG_PAIRS = [
    # front: (0,2), (1,3)
    ((0, 2), (1, 3)),
    # rear: (4,6), (5,7)
    ((4, 6), (5, 7)),
    # top: (0,5), (1,4)
    ((0, 5), (1, 4)),
    # bottom: (2,7), (3,6)
    ((2, 7), (3, 6)),
]


def filter_D(kps_orig, pnp_solver, R, t, tau_D=0.08):
    """Conditional diagonal incidence filter.

    Available diagonal pair가 있을 때만 검사.
    같은 face의 두 대각선 중점이 일치하는지 확인 (cuboid property).
    s_D = midpoint distance / projected diagonal

    "있는 점들만으로" incidence를 검사 — 9점 전부를 요구하지 않음.

    Args:
        tau_D: 무차원 임계값 (default 0.08)
    Returns:
        passed: bool, score: float (낮을수록 좋음), n_checked: int
    """
    detected = {}
    for i, p in enumerate(kps_orig[:8]):
        if p is not None:
            detected[i] = np.array([float(p[0]), float(p[1])], dtype=np.float64)

    reproj_all = pnp_solver.reproject(R, t)[:8]
    D = projected_diagonal(reproj_all)
    if D < 1e-6:
        return False, float("inf"), 0

    errors = []
    for (a, c), (b, d) in FACE_DIAG_PAIRS:
        # Both diagonals of this face must have all 4 corners detected
        if a in detected and c in detected and b in detected and d in detected:
            mid_ac = (detected[a] + detected[c]) / 2.0
            mid_bd = (detected[b] + detected[d]) / 2.0
            err = np.linalg.norm(mid_ac - mid_bd) / D
            errors.append(err)

    if not errors:
        # No complete diagonal pair available — pass by default (can't check)
        return True, 0.0, 0

    score = float(np.mean(errors))
    return score < tau_D, score, len(errors)


# ── Optional Priors ────────────────────────────────────────────────────

def prior_depth(t, depth_min=0.5, depth_max=15.0):
    """Depth range check."""
    z = float(t[2])
    return depth_min < z < depth_max, z


def prior_tilt(R, tau_tilt=45.0):
    """Pallet tilt check (Y-axis should be roughly vertical)."""
    y_cam = R @ np.array([0, 1, 0], dtype=np.float64)
    cos_angle = abs(y_cam[1])
    tilt = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))
    return tilt < tau_tilt, tilt
