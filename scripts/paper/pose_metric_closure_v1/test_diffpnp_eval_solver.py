"""diffpnp_eval_solver 게이트 1 — 이게 통과하지 못하면 solver_swap 수치는 전부 무효다.

    python3 -m pytest scripts/paper/pose_metric_closure_v1/test_diffpnp_eval_solver.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffpnp_eval_solver import (exp_so3, jacobian, project,  # noqa: E402
                                 solve_pnp_gn)

CAMERA = np.array([[600.0, 0.0, 320.0],
                   [0.0, 600.0, 240.0],
                   [0.0, 0.0, 1.0]], np.float64)


def _cuboid(across=1.1, height=0.12, along=1.3):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], np.float64)


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=3) * 0.35
    R = exp_so3(torch.as_tensor(w, dtype=torch.float64)[None])[0].numpy()
    t = np.array([rng.normal(0, 0.15), rng.normal(0, 0.15), 3.0 + rng.uniform(0, 2)])
    return R, t


def _project_np(R, t, X):
    P = X @ R.T + t
    uv = (P @ CAMERA.T)
    return uv[:, :2] / uv[:, 2:3]


def _rot_err_deg(A, B):
    """atan2 형태 — 0 근방에서 안정.

    ``arccos((tr-1)/2)`` 는 tr→3 에서 sqrt(eps) 로 뭉개져 float64 에서 1.2e-6 deg
    미만을 못 잰다. 그 눈금을 solver 오차로 오독한 이력이 있어 여기서만 안정형을 쓴다.
    """
    D = A @ B.T
    v = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(v) / 2.0,
                                       (np.trace(D) - 1.0) / 2.0)))


def test_recovers_known_pose_from_noiseless_correspondences():
    """게이트 1 — 무잡음이면 알려진 pose 를 회복해야 한다 (rot 1e-6도, t 1e-8m)."""
    X = _cuboid()
    for seed in range(8):
        R_gt, t_gt = _truth(seed)
        uv = _project_np(R_gt, t_gt, X)
        R, t, info = solve_pnp_gn(X, uv, CAMERA, init="epnp")
        assert R is not None, info
        assert not info["fallback"], info
        assert _rot_err_deg(R, R_gt) < 1e-6, (seed, _rot_err_deg(R, R_gt))
        assert np.linalg.norm(t - t_gt) < 1e-8, (seed, np.linalg.norm(t - t_gt))


def test_jacobian_matches_finite_difference():
    """좌측 섭동 규약이 실제로 맞는지 — Jacobian 을 수치미분과 대조한다."""
    X = _cuboid()
    R_gt, t_gt = _truth(3)
    uv = _project_np(R_gt, t_gt, X)
    Xt = torch.as_tensor(X, dtype=torch.float64)[None]
    Kt = torch.as_tensor(CAMERA, dtype=torch.float64)[None]
    Rt = torch.as_tensor(R_gt, dtype=torch.float64)[None]
    tt = torch.as_tensor(t_gt, dtype=torch.float64).reshape(1, 3)

    J = jacobian(Rt, tt, Xt, Kt)[0].numpy()
    base = project(Rt, tt, Xt, Kt)[0][0].numpy().reshape(-1)
    eps = 1e-7
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        Rp = exp_so3(torch.as_tensor(d[:3], dtype=torch.float64)[None]) @ Rt
        tp = tt + torch.as_tensor(d[3:], dtype=torch.float64).reshape(1, 3)
        pert = project(Rp, tp, Xt, Kt)[0][0].numpy().reshape(-1)
        num = (pert - base) / eps
        assert np.allclose(num, J[:, k], atol=1e-4), (k, num[:4], J[:4, k])
    del uv


def test_agrees_with_opencv_on_clean_data():
    """무잡음에서는 SQPnP+LM 과 같은 답이어야 한다 (같은 최소점)."""
    import cv2
    X = _cuboid()
    for seed in range(5):
        R_gt, t_gt = _truth(seed + 20)
        uv = _project_np(R_gt, t_gt, X)
        ok, rvec, tvec = cv2.solvePnP(X, uv, CAMERA, None, flags=cv2.SOLVEPNP_SQPNP)
        assert ok
        rvec, tvec = cv2.solvePnPRefineLM(X, uv, CAMERA, None, rvec, tvec)
        R_cv, _ = cv2.Rodrigues(rvec)
        R, t, _ = solve_pnp_gn(X, uv, CAMERA, init="epnp")
        assert _rot_err_deg(R, R_cv) < 1e-5
        assert np.linalg.norm(t - tvec.reshape(3)) < 1e-6


def test_huber_downweights_a_gross_outlier():
    """Huber arm 이 실제로 이상점을 눌러야 한다 — 안 그러면 D2/D4 는 D1 과 같다."""
    X = _cuboid()
    R_gt, t_gt = _truth(7)
    uv = _project_np(R_gt, t_gt, X)
    bad = uv.copy()
    bad[2] += np.array([120.0, -90.0])          # 코너 하나만 크게 틀림

    R_ls, t_ls, _ = solve_pnp_gn(X, bad, CAMERA, init="epnp")
    R_hu, t_hu, _ = solve_pnp_gn(X, bad, CAMERA, init="epnp", huber_delta=12.0)
    assert _rot_err_deg(R_hu, R_gt) < _rot_err_deg(R_ls, R_gt)
    assert np.linalg.norm(t_hu - t_gt) < np.linalg.norm(t_ls - t_gt)


def test_is_actually_differentiable_wrt_image_points():
    """'미분가능' 주장이 말뿐이 아닌지 — 2D 입력으로 grad 가 흐르는지 확인한다."""
    X = _cuboid()
    R_gt, t_gt = _truth(11)
    uv = _project_np(R_gt, t_gt, X) + np.random.default_rng(0).normal(0, 1.5, (8, 2))
    R, t, info = solve_pnp_gn(X, uv, CAMERA, init="epnp", requires_grad=True)
    assert "t_t" in info
    loss = info["t_t"].sum()
    loss.backward()
    grad = info["obs_t"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
