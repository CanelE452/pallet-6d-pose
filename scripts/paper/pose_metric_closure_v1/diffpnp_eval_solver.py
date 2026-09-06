"""미분가능 Gauss-Newton PnP — 논문 pose 평가의 read-out solver 교체용.

`cv2.SOLVEPNP_SQPNP + solvePnPRefineLM` 자리에 그대로 끼울 수 있는 solver 다.
투영식과 Jacobian 은 `Deep_Object_Pose/train/diffpnp3d_loss.py` 의 검증된 규약
(`_project_batch` / `_jac_batch`, geo_loss_bpnp 계보)을 그대로 옮겼고, 회전 갱신만
Jacobian 의 좌측 섭동 규약에 정확히 맞춰 ``R <- Exp(w) R`` 로 바꿨다.

학습 loss 가 아니라 **평가 경로**용이므로 accept/reject 분기 없이 고정 step 수의
damped GN 을 편다(unrolled). 그래야 입력 2D 좌표에 대한 미분이 끝까지 흐른다.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

GN_STEPS = 10
GN_DAMPING = 1e-3
GN_DELTA_CLIP = 0.5
HUBER_DELTA_PX = 12.0


def _skew(w):
    """(B,3) -> (B,3,3)"""
    B = w.shape[0]
    O = torch.zeros(B, dtype=w.dtype, device=w.device)
    return torch.stack([
        torch.stack([O, -w[:, 2], w[:, 1]], dim=-1),
        torch.stack([w[:, 2], O, -w[:, 0]], dim=-1),
        torch.stack([-w[:, 1], w[:, 0], O], dim=-1),
    ], dim=-2)


def exp_so3(w):
    """axis-angle (B,3) -> SO(3) (B,3,3). theta=0 근방에서 안전하다."""
    theta = w.norm(dim=1).clamp(min=1e-12)
    K = _skew(w / theta.unsqueeze(-1))
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(w.shape[0], -1, -1)
    s = torch.sin(theta).view(-1, 1, 1)
    c = torch.cos(theta).view(-1, 1, 1)
    return eye + s * K + (1.0 - c) * torch.bmm(K, K)


def project(R, t, X, K):
    """R (B,3,3), t (B,3), X (B,N,3), K (B,3,3) -> uv (B,N,2), P_cam (B,N,3)."""
    P_cam = torch.bmm(X, R.transpose(1, 2)) + t.unsqueeze(1)
    uvw = torch.bmm(P_cam, K.transpose(1, 2))
    z = uvw[:, :, 2:3].clamp(min=1e-6)
    return uvw[:, :, :2] / z, P_cam


def jacobian(R, t, X, K):
    """reprojection 의 (w, t) Jacobian. (B,2N,6). w 는 좌측 섭동 R <- Exp(w) R."""
    B, N, _ = X.shape
    P_cam = torch.bmm(X, R.transpose(1, 2)) + t.unsqueeze(1)
    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    Xc, Yc = P_cam[:, :, 0], P_cam[:, :, 1]
    Zc = P_cam[:, :, 2].clamp(min=1e-6)
    Z2 = Zc * Zc

    dpdP = torch.zeros(B, N, 2, 3, dtype=X.dtype, device=X.device)
    dpdP[:, :, 0, 0] = fx / Zc
    dpdP[:, :, 0, 2] = -fx * Xc / Z2
    dpdP[:, :, 1, 1] = fy / Zc
    dpdP[:, :, 1, 2] = -fy * Yc / Z2

    RX = P_cam - t.unsqueeze(1)                     # = X R^T
    dPdw = torch.zeros(B, N, 3, 3, dtype=X.dtype, device=X.device)
    dPdw[:, :, 0, 1] = RX[:, :, 2];  dPdw[:, :, 0, 2] = -RX[:, :, 1]
    dPdw[:, :, 1, 0] = -RX[:, :, 2]; dPdw[:, :, 1, 2] = RX[:, :, 0]
    dPdw[:, :, 2, 0] = RX[:, :, 1];  dPdw[:, :, 2, 1] = -RX[:, :, 0]
    dPdt = torch.eye(3, dtype=X.dtype, device=X.device).view(1, 1, 3, 3).expand(B, N, -1, -1)

    J = torch.cat([torch.matmul(dpdP, dPdw), torch.matmul(dpdP, dPdt)], dim=-1)
    return J.reshape(B, 2 * N, 6)


def _init_pose(object_points, image_points, camera, flag):
    """비미분 초기화. BPnP 계열이 쓰는 방식 그대로 — 초기값만 OpenCV 에서 받는다."""
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera, None, flags=flag)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.reshape(3).astype(np.float64)


def solve_pnp_gn(object_points, image_points, camera, *,
                 init="epnp", init_pose=None, weights=None,
                 huber_delta=None, steps=GN_STEPS, damping=GN_DAMPING,
                 delta_clip=GN_DELTA_CLIP, requires_grad=False):
    """미분가능 GN PnP.

    object_points (N,3) / image_points (N,2) / camera (3,3)
    init          "epnp" | "sqpnp" | "given"(init_pose=(R,t) 필요)
    weights       (N,) 점별 가중치 (예: keypoint confidence). None 이면 균일.
    huber_delta   px. None 이면 순수 제곱오차, 값이 있으면 IRLS Huber.
    requires_grad True 면 image_points 를 leaf tensor 로 받아 grad 를 흘린다.

    반환 (R (3,3) ndarray, t (3,) ndarray, info dict). 실패 시 R,t = None.
    """
    X_np = np.asarray(object_points, np.float64)
    uv_np = np.asarray(image_points, np.float64)
    K_np = np.asarray(camera, np.float64)

    if init == "given":
        if init_pose is None:
            raise ValueError("init='given' 이면 init_pose 가 필요하다")
        seed = (np.asarray(init_pose[0], np.float64),
                np.asarray(init_pose[1], np.float64).reshape(3))
    else:
        flag = cv2.SOLVEPNP_EPNP if init == "epnp" else cv2.SOLVEPNP_SQPNP
        seed = _init_pose(X_np, uv_np, K_np, flag)
    if seed is None:
        return None, None, {"fallback": True, "reason": "init_failed"}

    dev = "cpu"
    X = torch.as_tensor(X_np, dtype=torch.float64, device=dev)[None]
    obs = torch.as_tensor(uv_np, dtype=torch.float64, device=dev)[None]
    if requires_grad:
        obs = obs.clone().requires_grad_(True)
    Km = torch.as_tensor(K_np, dtype=torch.float64, device=dev)[None]
    R = torch.as_tensor(seed[0], dtype=torch.float64, device=dev)[None]
    t = torch.as_tensor(seed[1], dtype=torch.float64, device=dev).reshape(1, 3)

    if weights is None:
        w_base = torch.ones(1, X.shape[1], dtype=torch.float64, device=dev)
    else:
        w_base = torch.as_tensor(np.asarray(weights, np.float64),
                                 dtype=torch.float64, device=dev)[None]
        w_base = w_base.clamp(min=1e-6)

    eye6 = torch.eye(6, dtype=torch.float64, device=dev)[None]
    info = {"fallback": False, "reason": None, "steps_taken": 0,
            "observed_before": None, "observed_after": None}

    with torch.no_grad():
        uv0, _ = project(R, t, X, Km)
        info["observed_before"] = float(torch.linalg.norm(uv0 - obs, dim=-1).mean())

    for _ in range(int(steps)):
        uv, P_cam = project(R, t, X, Km)
        res = uv - obs                                        # (1,N,2)

        w = w_base
        if huber_delta is not None:
            with torch.no_grad():
                rn = torch.linalg.norm(res, dim=-1).clamp(min=1e-12)
                w = w_base * torch.where(rn <= huber_delta,
                                         torch.ones_like(rn), huber_delta / rn)
        sw = torch.sqrt(w).unsqueeze(-1)                       # (1,N,1)

        J = jacobian(R, t, X, Km)                              # (1,2N,6)
        Jw = J * sw.repeat_interleave(2, dim=1)
        rw = (res * sw).reshape(1, -1, 1)

        A = torch.bmm(Jw.transpose(1, 2), Jw) + damping * eye6
        g = torch.bmm(Jw.transpose(1, 2), rw)
        try:
            delta = torch.linalg.solve(A, -g).reshape(1, 6)
        except Exception:
            info["fallback"], info["reason"] = True, "solve_failed"
            break
        if not torch.isfinite(delta).all():
            info["fallback"], info["reason"] = True, "non_finite_delta"
            break

        norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-12)
        delta = delta * (delta_clip / norm).clamp(max=1.0)

        R = torch.bmm(exp_so3(delta[:, :3]), R)
        t = t + delta[:, 3:]
        info["steps_taken"] += 1

    if info["fallback"]:
        return seed[0], seed[1], info

    with torch.no_grad():
        uv1, P_cam = project(R, t, X, Km)
        info["observed_after"] = float(torch.linalg.norm(uv1 - obs, dim=-1).mean())
        if not torch.isfinite(R).all() or not torch.isfinite(t).all() \
                or float(P_cam[..., 2].min()) <= 0.0 \
                or abs(float(torch.linalg.det(R[0])) - 1.0) > 1e-6:
            info["fallback"], info["reason"] = True, "invalid_pose"
            return seed[0], seed[1], info

    if requires_grad:
        info["R_t"] = R
        info["t_t"] = t
        info["obs_t"] = obs
    return (R[0].detach().cpu().numpy(), t[0].detach().cpu().numpy(), info)
