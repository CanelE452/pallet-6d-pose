# calib/geometry.py
# 3D 포인트 구성, 평면 적합으로 yaw 계산, bbox 클램프

import numpy as np
from sklearn.linear_model import RANSACRegressor
import pyrealsense2 as rs
from typing import Tuple, Optional
from .config import PLANE_INLIER_THRESH, PLANE_MAX_TRIALS, MIN_POINTS

def compute_yaw_deg_from_plane(a: float, b: float) -> float:
    # z = a x + b y + c → 법선(nx,ny,nz)=(-a,-b,1), 여기서 yaw는 x-z 투영을 사용
    nx, nz = -a, 1.0
    return float(np.degrees(np.arctan2(nx, nz)))

def robust_points_from_mask_or_roi(depth_frame, depth_intrin, mask_or_roi,
                                   stride=3, z_inlier_thresh=0.06, min_points=120) -> Tuple[bool, Optional[np.ndarray]]:
    ys, xs = np.where(mask_or_roi > 0)
    if len(xs) == 0:
        return False, None
    xs = xs[::stride].astype(np.int32)
    ys = ys[::stride].astype(np.int32)

    dists = np.array([depth_frame.get_distance(int(u), int(v)) for u, v in zip(xs, ys)],
                     dtype=np.float32)
    valid = np.isfinite(dists) & (dists > 0)
    if not np.any(valid):
        return False, None
    xs_v, ys_v, d_v = xs[valid], ys[valid], dists[valid]
    if len(d_v) < min_points:
        return False, None

    pts = np.array([
        rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], float(d))
        for u, v, d in zip(xs_v, ys_v, d_v)
    ], dtype=np.float32)

    z_med = np.median(pts[:, 2])
    inliers = np.abs(pts[:, 2] - z_med) <= float(z_inlier_thresh)
    pts_in = pts[inliers]
    if len(pts_in) < min_points:
        return False, None
    return True, pts_in

def fit_plane_yaw_from_points(pts_in: np.ndarray):
    X = pts_in[:, :2]; Z = pts_in[:, 2]
    ransac = RANSACRegressor(residual_threshold=PLANE_INLIER_THRESH,
                             max_trials=PLANE_MAX_TRIALS, random_state=0)
    ransac.fit(X, Z)
    if ransac.inlier_mask_ is None or ransac.inlier_mask_.sum() < max(30, MIN_POINTS//2):
        return False, None, None, None
    a, b = ransac.estimator_.coef_
    yaw = compute_yaw_deg_from_plane(a, b)
    return True, yaw, a, b

def clamp_bbox(x1,y1,x2,y2,W,H):
    x1 = max(0, min(W-1, int(x1)))
    y1 = max(0, min(H-1, int(y1)))
    x2 = max(0, min(W-1, int(x2)))
    y2 = max(0, min(H-1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1,y1,x2,y2

# ----------------------------------------------------------------------
# [추가] offset/width 계산 유틸
#  - 요구사항: offset은 중앙값(median) 기반으로 계산
#  - width는 기존 관행 유지(min-max 범위)
# ----------------------------------------------------------------------
def compute_offset_and_width(pts_in: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    pts_in: (N,3) ndarray of 3D points in camera coords (x, y, z)
    return:
      ok: bool
      offset_x: float (중앙값 기반)
      width_x:  float (min-max 기반)
    """
    if pts_in is None or len(pts_in) == 0:
        return False, None, None

    X = pts_in[:, 0].astype(np.float32)

    # ---- offset: 중앙값(median) ----
    offset_x = float(np.median(X))

    # ---- width: 기존대로 min-max ----
    if len(X) > 1:
        width_x = float(np.max(X) - np.min(X))
    else:
        width_x = 0.0

    return True, offset_x, width_x
