"""run_live.py — Sanity gate 모듈.

PnP 결과에 적용하는 sanity gate 들. 각 함수는 단순/명확하게 분리.

_kp_count(raw_points)           : None 아닌 keypoint 갯수
_reproj_error(raw, proj)         : raw 와 PnP projection 의 평균 reproj px
_edge_ratio_ok(proj, dim_cm, tol): cuboid edge bbox 종횡비 sanity
evaluate_result(...)             : 위 gate 들 + z range + depth 일치 종합 평가
                                    반환 (ok: bool, reason: str, info: dict)
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import numpy as np


def _kp_count(raw_points):
    return sum(1 for p in raw_points if p is not None)


def _reproj_error(raw_points, proj_points):
    if proj_points is None:
        return float("inf")
    errs = []
    for r, p in zip(raw_points, proj_points):
        if r is None or p is None:
            continue
        errs.append(np.hypot(r[0] - p[0], r[1] - p[1]))
    return float(np.mean(errs)) if errs else float("inf")


def _edge_ratio_ok(proj_points, cuboid_dim_cm, tol):
    """proj_points 0~7 의 X/Y bbox 종횡비가 cuboid 가능 범위인가."""
    if proj_points is None:
        return True
    pts = [p for p in proj_points[:8] if p is not None]
    if len(pts) < 8:
        return True
    arr = np.array(proj_points[:8], dtype=np.float32)
    xs, ys = arr[:, 0], arr[:, 1]
    w_px = xs.max() - xs.min()
    h_px = ys.max() - ys.min()
    if w_px < 5 or h_px < 5:
        return False
    ratio_px = max(w_px, h_px) / max(min(w_px, h_px), 1.0)
    # KS T-11 1.1×0.15×1.1 → 시점에 따라 1~8 범위 정상. 그 이상은 비정상.
    return 0.8 <= ratio_px <= (8.0 * (1.0 + tol))


def evaluate_result(result, cfg_gates, depth_cm, K_proc):
    """PnP 결과에 모든 gate 적용. (ok, reason, info) 반환."""
    raw = result.get("raw_points")
    if raw is None:
        return False, "no_raw_points", {}
    n_kp = _kp_count(raw)
    if n_kp < cfg_gates["min_detected_keypoints"]:
        return False, f"kp={n_kp}<{cfg_gates['min_detected_keypoints']}", {"n_kp": n_kp}

    loc = result.get("location")
    if loc is None:
        return False, "pnp_failed", {"n_kp": n_kp}
    z_m = float(loc[2]) / 100.0   # location 은 cm
    if z_m < cfg_gates["z_min_m"] or z_m > cfg_gates["z_max_m"]:
        return False, f"z={z_m:.2f}m out of range", {"n_kp": n_kp, "z_m": z_m}

    proj = result.get("projected_points")
    reproj = _reproj_error(raw, proj)
    if reproj > cfg_gates["max_reproj_error_px"]:
        return False, f"reproj={reproj:.1f}px", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj}

    if not _edge_ratio_ok(proj, None, cfg_gates["cuboid_edge_ratio_tol"]):
        return False, "edge_ratio", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj}

    if depth_cm is not None:
        z_cm = z_m * 100.0
        rel = abs(depth_cm - z_cm) / max(z_cm, 1e-6)
        if rel > cfg_gates["depth_pnp_z_max_rel"]:
            return False, f"depth_z_diff={rel:.2f}", {
                "n_kp": n_kp, "z_m": z_m, "reproj": reproj, "depth_rel": rel
            }
    return True, "ok", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj}
