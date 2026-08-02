"""diffpnp3d_loss.py — dimensions-aware DiffPnP3D 3D-corner geometry loss.

heatmap keypoint(DOPE belief) 위에 붙이는 regularizer. inference 시 완전 제거 가능.

구성:
  LocalSoftArgmax2D   : peak 주변 7x7 window softmax (미분가능 sub-pixel keypoint)
                        + confidence(peak / peak-to-second ratio / local sharpness)
  DiffPnP3DLoss       : GT-pose init unrolled Gauss-Newton → T_pred, 3D corner
                        Huber(||T_pred·X − T_gt·X|| / diag). pnp_valid_3d∧V8 gating,
                        NaN / positive-depth / condition / grad-norm guards.
  build_diffpnp3d_targets : frame JSON + audit index → {X_i(camera-facing 8x3),
                        K, R_gt, t_gt, diag, pnp_valid, V8}. audit 기하 재사용.

기하 규약(재검증 말고 사용 — pnp_valid_3d_audit.py 에서 확정 [확인]):
  pose_transform = object→camera, OpenCV z-forward.
  X_i = canonical_box(W,D,H) 에 best_sym(48-sym) 적용 = camera-facing 0123 순서.
  K 는 프레임별. belief(50x50) ↔ 원본(640x480) 은 축별(anisotropic) 선형 스케일.

좌표 공간(scale factor 명시):
  soft-argmax 는 belief(H_b,W_b) 픽셀에서 sub-pixel peak 를 잡고,
  scale_x = W_orig/W_bel, scale_y = H_orig/H_bel 로 원본(640x480) 픽셀로 환산한다.
  DiffPnP3D 는 원본 픽셀 pred_xy 와 원본 K(=JSON intrinsics)를 함께 쓴다
  (projected_cuboid/K 와 동일 공간).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_loss_bpnp import rodrigues_batch  # noqa: E402  (재사용, 검증됨)


# ======================================================================
# 1. Local 7x7 soft-argmax (미분가능 keypoint + confidence)
# ======================================================================
class LocalSoftArgmax2D(nn.Module):
    """peak 주변 window softmax 로 sub-pixel keypoint 를 낸다(global soft-argmax 아님).

    peak 위치는 argmax(비미분, index 선택)로 잡되, 좌표는 window softmax 가중합이라
    belief 값에 대해 미분가능(detach 없음). coords 는 원본(orig_size) 픽셀로 환산해 반환.
    """

    def __init__(self, window=7, temperature=1.0,
                 orig_size=(640, 480), belief_size=(50, 50)):
        super().__init__()
        assert window % 2 == 1, "window must be odd"
        self.r = window // 2
        self.temperature = temperature
        self.orig_w, self.orig_h = orig_size
        self.bel_w, self.bel_h = belief_size
        # scale factor: belief pixel -> orig pixel (축별 anisotropic)
        self.scale_x = self.orig_w / self.bel_w
        self.scale_y = self.orig_h / self.bel_h

    def forward(self, belief):
        """belief (B,C,H,W) → coords_orig (B,C,2) [x,y in orig px], conf(dict)."""
        B, C, H, W = belief.shape
        dev = belief.device
        flat = belief.reshape(B, C, H * W)

        peak_val, peak_idx = flat.max(dim=2)          # (B,C)  argmax = 비미분
        py = (peak_idx // W).long()
        px = (peak_idx % W).long()

        # 7x7 window 절대좌표 (정수 상수) — clamp 로 경계 처리
        off = torch.arange(-self.r, self.r + 1, device=dev)
        dy, dx = torch.meshgrid(off, off, indexing="ij")
        dy = dy.reshape(-1)                            # (K,)
        dx = dx.reshape(-1)
        wy = (py.unsqueeze(-1) + dy).clamp(0, H - 1)   # (B,C,K)
        wx = (px.unsqueeze(-1) + dx).clamp(0, W - 1)
        gidx = (wy * W + wx).long()                    # (B,C,K)

        vals = torch.gather(flat, 2, gidx)             # (B,C,K) 미분 통과
        w = F.softmax(vals / self.temperature, dim=2)  # (B,C,K)

        cx = (w * wx.float()).sum(2)                   # belief px (B,C)
        cy = (w * wy.float()).sum(2)
        coords_orig = torch.stack(
            [cx * self.scale_x, cy * self.scale_y], dim=-1)  # (B,C,2) orig px

        # --- confidence (진단, grad 불필요) ---
        with torch.no_grad():
            # local sharpness: window softmax 2차 모멘트 (belief px)
            var_x = (w * (wx.float() - cx.unsqueeze(-1)) ** 2).sum(2)
            var_y = (w * (wy.float() - cy.unsqueeze(-1)) ** 2).sum(2)
            cov_xy = (w * (wx.float() - cx.unsqueeze(-1))
                      * (wy.float() - cy.unsqueeze(-1))).sum(2)
            sigma = torch.sqrt((var_x + var_y).clamp(min=1e-8))
            # peak-to-second: window 영역을 -inf 로 지우고 남은 최대
            flat_masked = flat.clone()
            flat_masked.scatter_(2, gidx, float("-inf"))
            second_val, _ = flat_masked.max(dim=2)
            ratio = peak_val / (second_val.clamp(min=0) + 1e-6)
        conf = {"peak": peak_val, "second": second_val, "ratio": ratio,
                "sigma": sigma, "var_x": var_x, "var_y": var_y, "cov_xy": cov_xy}
        return coords_orig, conf


# ======================================================================
# batched projection + Gauss-Newton Jacobian (per-frame X, per-frame K)
#   geo_loss_bpnp.project_points/projection_jacobian 와 동일 부호규약 —
#   공유 kp3d 가정을 per-frame X_i(B,N,3), per-frame K(B,3,3) 로 확장.
# ======================================================================
def _project_batch(rvec, tvec, X, K):
    """rvec,tvec (B,3), X (B,N,3), K (B,3,3) → uv (B,N,2), P_cam (B,N,3)."""
    R = rodrigues_batch(rvec)                          # (B,3,3)
    P_cam = torch.bmm(X, R.transpose(1, 2)) + tvec.unsqueeze(1)
    uvw = torch.bmm(P_cam, K.transpose(1, 2))
    z = uvw[:, :, 2:3].clamp(min=1e-6)
    return uvw[:, :, :2] / z, P_cam


def _jac_batch(rvec, tvec, X, K):
    """residual(proj) 의 (rvec,tvec) Jacobian. → (B,2N,6). geo_loss_bpnp 규약."""
    B, N, _ = X.shape
    dev, dt = X.device, X.dtype
    R = rodrigues_batch(rvec)
    P_cam = torch.bmm(X, R.transpose(1, 2)) + tvec.unsqueeze(1)
    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    Xc, Yc = P_cam[:, :, 0], P_cam[:, :, 1]
    Zc = P_cam[:, :, 2].clamp(min=1e-6)
    Z2 = Zc * Zc

    dpdP = torch.zeros(B, N, 2, 3, device=dev, dtype=dt)
    dpdP[:, :, 0, 0] = fx / Zc
    dpdP[:, :, 0, 2] = -fx * Xc / Z2
    dpdP[:, :, 1, 1] = fy / Zc
    dpdP[:, :, 1, 2] = -fy * Yc / Z2

    RP = P_cam - tvec.unsqueeze(1)                     # = X R^T
    dPdr = torch.zeros(B, N, 3, 3, device=dev, dtype=dt)
    dPdr[:, :, 0, 1] = RP[:, :, 2];  dPdr[:, :, 0, 2] = -RP[:, :, 1]
    dPdr[:, :, 1, 0] = -RP[:, :, 2]; dPdr[:, :, 1, 2] = RP[:, :, 0]
    dPdr[:, :, 2, 0] = RP[:, :, 1];  dPdr[:, :, 2, 1] = -RP[:, :, 0]
    dPdt = torch.eye(3, device=dev, dtype=dt).view(1, 1, 3, 3).expand(B, N, -1, -1)

    dpdr = torch.matmul(dpdP, dPdr)                    # (B,N,2,3)
    dpdt = torch.matmul(dpdP, dPdt)
    J = torch.cat([dpdr, dpdt], dim=-1)               # (B,N,2,6)
    return J.reshape(B, 2 * N, 6)


def _huber(x, delta):
    """element-wise Huber (bounded slope → grad-norm guard 역할). x>=0 가정."""
    a = x.abs()
    quad = 0.5 * a * a
    lin = delta * (a - 0.5 * delta)
    return torch.where(a <= delta, quad, lin)


def _pnp_fit_coverage(uv_pred, observed_xy, margin, huber_delta,
                      min_observed_span=1.0):
    """One-sided footprint coverage of a PnP projection over its observations.

    ``observed_xy`` is the decoded heatmap geometry used by PnP.  Its centre,
    PCA axes and spans are deliberately detached: they are a fixed target for
    this loss, not a second route by which the network can reduce the target.
    Both point sets are centred independently, so the term is invariant to an
    image-plane translation and measures footprint size only.

    Returns per-frame loss, the minimum valid-axis span ratio and a frame-valid
    flag.  Degenerate/non-finite observed axes are ignored rather than creating
    a large or undefined ratio.
    """
    B, N, _ = observed_xy.shape
    del N  # The normalization below is independent of the number of corners.

    observed = observed_xy.detach()
    observed_centered = (
        observed - observed.mean(dim=1, keepdim=True)).detach()
    covariance = torch.bmm(
        observed_centered.transpose(1, 2), observed_centered
    ) / float(observed_xy.shape[1])
    _eigenvalues, observed_axes = torch.linalg.eigh(covariance.detach())
    observed_axis = torch.bmm(observed_centered, observed_axes).detach()
    observed_span = (
        observed_axis.amax(dim=1) - observed_axis.amin(dim=1)).detach()

    projected_centered = uv_pred - uv_pred.mean(dim=1, keepdim=True)
    projected_axis = torch.bmm(projected_centered, observed_axes)
    projected_span = projected_axis.amax(dim=1) - projected_axis.amin(dim=1)

    axis_ok = (
        torch.isfinite(observed_span)
        & (observed_span >= float(min_observed_span))
        & torch.isfinite(projected_span)
    )
    safe_observed_span = torch.where(
        axis_ok, observed_span, torch.ones_like(observed_span))
    safe_projected_span = torch.where(
        axis_ok, projected_span, safe_observed_span)
    span_ratio = (
        safe_projected_span.clamp_min(1.0e-6)
        / safe_observed_span.clamp_min(1.0e-6)
    )
    margin_t = torch.as_tensor(
        margin, device=uv_pred.device, dtype=uv_pred.dtype)
    span_deficit = torch.relu(
        torch.log(margin_t) - torch.log(span_ratio.clamp_min(1.0e-6)))
    axis_loss = torch.where(
        axis_ok, _huber(span_deficit, huber_delta),
        torch.zeros_like(span_deficit))
    axis_count = axis_ok.sum(dim=1)
    per_frame = axis_loss.sum(dim=1) / axis_count.clamp(min=1).to(
        uv_pred.dtype)

    # Invalid axes are neutral (ratio 1) for diagnostics.  A frame with no
    # valid image-plane extent is explicitly marked invalid by ``frame_ok``.
    diagnostic_ratio = torch.where(
        axis_ok, span_ratio, torch.ones_like(span_ratio))
    min_span_ratio = diagnostic_ratio.amin(dim=1)
    frame_ok = axis_count > 0
    return per_frame, min_span_ratio, frame_ok


# ======================================================================
# 2. DiffPnP3D loss
# ======================================================================
class DiffPnP3DLoss(nn.Module):
    """GT-pose init unrolled Gauss-Newton PnP + 3D corner Huber(/diag).

    L = mean_corner Huber( ||T_pred·X_i − T_gt·X_i|| / diag ).  (2D reproj 아님)
    pnp_valid_3d∧V8 프레임에만 적용, 나머지 0. batch 평균은 valid 프레임 수로 정규화.
    """

    def __init__(self, n_gn=4, gn_damping=1e-3, huber_delta=0.05,
                 delta_clip=0.5, cond_max=1e8, temperature=1.0,
                 geometry_weight=1.0, undercoverage_weight=0.0,
                 span_under_weight=1.0, depth_under_weight=1.0,
                 span_margin=1.0, depth_margin=1.0,
                 hard_span_threshold=0.90, hard_depth_threshold=1.10,
                 hard_example_gain=0.0, fit_coverage_weight=0.0,
                 fit_span_margin=1.0, fit_hard_span_threshold=0.90,
                 fit_min_observed_span=1.0):
        super().__init__()
        if (geometry_weight < 0 or undercoverage_weight < 0
                or fit_coverage_weight < 0):
            raise ValueError("DiffPnP loss weights must be non-negative")
        if span_under_weight < 0 or depth_under_weight < 0:
            raise ValueError("span/depth undercoverage weights must be non-negative")
        if (undercoverage_weight > 0
                and span_under_weight == 0 and depth_under_weight == 0):
            raise ValueError(
                "enabled undercoverage loss needs a span or depth component")
        if not 0 < span_margin <= 1.0:
            raise ValueError("span_margin must be in (0, 1]")
        if depth_margin < 1.0:
            raise ValueError("depth_margin must be >= 1")
        if not 0 < hard_span_threshold <= 1.0:
            raise ValueError("hard_span_threshold must be in (0, 1]")
        if hard_depth_threshold < 1.0:
            raise ValueError("hard_depth_threshold must be >= 1")
        if hard_example_gain < 0:
            raise ValueError("hard_example_gain must be non-negative")
        if not 0 < fit_span_margin <= 1.0:
            raise ValueError("fit_span_margin must be in (0, 1]")
        if not 0 < fit_hard_span_threshold <= 1.0:
            raise ValueError("fit_hard_span_threshold must be in (0, 1]")
        if fit_min_observed_span <= 0:
            raise ValueError("fit_min_observed_span must be positive")
        self.n_gn = n_gn
        self.gn_damping = gn_damping
        self.huber_delta = huber_delta
        self.delta_clip = delta_clip          # GN trust-region (grad 폭주 방지)
        self.cond_max = cond_max
        self.temperature = temperature
        self.geometry_weight = float(geometry_weight)
        self.undercoverage_weight = float(undercoverage_weight)
        self.span_under_weight = float(span_under_weight)
        self.depth_under_weight = float(depth_under_weight)
        self.span_margin = float(span_margin)
        self.depth_margin = float(depth_margin)
        self.hard_span_threshold = float(hard_span_threshold)
        self.hard_depth_threshold = float(hard_depth_threshold)
        self.hard_example_gain = float(hard_example_gain)
        self.fit_coverage_weight = float(fit_coverage_weight)
        self.fit_span_margin = float(fit_span_margin)
        self.fit_hard_span_threshold = float(fit_hard_span_threshold)
        self.fit_min_observed_span = float(fit_min_observed_span)

    @staticmethod
    def rvec_from_R(R):
        """(B,3,3) → (B,3) rvec (scipy, GT init 전용 — 상수, grad 불필요)."""
        from scipy.spatial.transform import Rotation
        return torch.from_numpy(
            Rotation.from_matrix(R.detach().cpu().numpy()).as_rotvec()
        ).to(device=R.device, dtype=R.dtype)

    def forward(self, pred_xy, X, K, R_gt, t_gt, diag, mask):
        """pred_xy (B,8,2) orig px, X (B,8,3), K (B,3,3), R_gt (B,3,3),
        t_gt (B,3), diag (B,), mask (B,) bool(pnp_valid_3d∧V8).
        return: loss(scalar), diag(dict)."""
        B, N, _ = X.shape
        dev = pred_xy.device
        # Guard before the batched solve/eigendecomposition.  Checking only the
        # final loss is too late: one NaN observation can make eigvalsh raise and
        # abort the entire otherwise-valid batch.  ``nan_to_num`` is identical
        # on finite observations and has zero gradient at replaced entries; the
        # original finite flag below keeps those frames out of the loss.
        pred_finite = torch.isfinite(pred_xy).reshape(B, -1).all(dim=1)
        pred_xy_safe = torch.nan_to_num(
            pred_xy, nan=0.0, posinf=0.0, neginf=0.0)
        obs = pred_xy_safe.reshape(B, 2 * N, 1)

        # --- GT-pose init (Option A) ---
        rvec = self.rvec_from_R(R_gt).clone()
        tvec = t_gt.detach().clone()
        eye6 = torch.eye(6, device=dev, dtype=pred_xy.dtype).unsqueeze(0)

        cond = torch.full((B,), float("nan"), device=dev)
        for _ in range(self.n_gn):
            uv, _ = _project_batch(rvec, tvec, X, K)
            r = (uv.reshape(B, 2 * N, 1) - obs)        # proj - obs (obs=pred_xy)
            J = _jac_batch(rvec, tvec, X, K)           # (B,2N,6)
            JtJ = torch.bmm(J.transpose(1, 2), J)
            A = JtJ + self.gn_damping * eye6
            Jtr = torch.bmm(J.transpose(1, 2), r)      # (B,6,1)
            delta = torch.linalg.solve(A, Jtr).squeeze(-1)  # (B,6)
            # trust-region clip (grad-norm guard)
            dn = delta.norm(dim=1, keepdim=True).clamp(min=1e-9)
            scale = (self.delta_clip / dn).clamp(max=1.0)
            delta = delta * scale
            rvec = rvec - delta[:, :3]
            tvec = tvec - delta[:, 3:]
            with torch.no_grad():
                ev = torch.linalg.eigvalsh(A)
                cond = ev[:, -1] / ev[:, 0].clamp(min=1e-12)

        # --- T_pred·X vs T_gt·X (3D corner distance) ---
        R_pred = rodrigues_batch(rvec)
        P_pred = torch.bmm(X, R_pred.transpose(1, 2)) + tvec.unsqueeze(1)   # (B,N,3)
        P_gt = (torch.bmm(X, R_gt.transpose(1, 2)) + t_gt.unsqueeze(1)).detach()
        d3d = torch.norm(P_pred - P_gt, dim=2) / diag.view(B, 1).clamp(min=1e-6)
        per_frame_geometry = _huber(
            d3d, self.huber_delta).mean(dim=1)                              # (B,)

        # One-sided PnP footprint/depth term.  This is intentionally distinct
        # from a raw heatmap span loss: it penalizes the *pose selected by the
        # differentiable PnP solve* only when that pose is farther away or its
        # projected W/D footprint is smaller than GT.  GT PCA axes are detached
        # constants, keeping the gradient exclusively on the predicted pose.
        uv_pred, _ = _project_batch(rvec, tvec, X, K)
        gt_uvw = torch.bmm(P_gt, K.transpose(1, 2))
        uv_gt = gt_uvw[:, :, :2] / gt_uvw[:, :, 2:3].clamp(min=1e-6)
        gt_centered = uv_gt - uv_gt.mean(dim=1, keepdim=True)
        pred_centered = uv_pred - uv_pred.mean(dim=1, keepdim=True)
        covariance = torch.bmm(
            gt_centered.transpose(1, 2), gt_centered) / float(N)
        _eigenvalues, axes = torch.linalg.eigh(covariance.detach())
        gt_axis = torch.bmm(gt_centered, axes)
        pred_axis = torch.bmm(pred_centered, axes)
        gt_span = gt_axis.amax(dim=1) - gt_axis.amin(dim=1)
        pred_span = pred_axis.amax(dim=1) - pred_axis.amin(dim=1)
        span_ratio = pred_span.clamp_min(1.0e-6) / gt_span.clamp_min(1.0e-6)
        span_under = torch.relu(
            torch.log(torch.as_tensor(
                self.span_margin, device=dev, dtype=pred_xy.dtype))
            - torch.log(span_ratio.clamp_min(1.0e-6)))

        tz_ratio = tvec[:, 2].clamp_min(1.0e-6) / t_gt[:, 2].detach().clamp_min(
            1.0e-6)
        depth_under = torch.relu(
            torch.log(tz_ratio.clamp_min(1.0e-6))
            - torch.log(torch.as_tensor(
                self.depth_margin, device=dev, dtype=pred_xy.dtype)))
        under_denominator = max(
            2.0 * self.span_under_weight + self.depth_under_weight, 1.0e-12)
        per_frame_undercoverage = (
            self.span_under_weight
            * _huber(span_under, self.huber_delta).sum(dim=1)
            + self.depth_under_weight
            * _huber(depth_under, self.huber_delta)
        ) / under_denominator
        min_span_ratio = span_ratio.amin(dim=1)
        hard_example = (
            ((self.span_under_weight > 0)
             & (min_span_ratio.detach() < self.hard_span_threshold))
            | ((self.depth_under_weight > 0)
               & (tz_ratio.detach() > self.hard_depth_threshold))
        )
        hard_weight = 1.0 + self.hard_example_gain * hard_example.to(
            pred_xy.dtype)
        per_frame_base = (
            self.geometry_weight * per_frame_geometry
            + self.undercoverage_weight * hard_weight
            * per_frame_undercoverage
        )

        # PnP-fit coverage is relative to the *observed heatmap footprint*,
        # unlike the GT-relative term above.  It therefore detects the failure
        # mode where an otherwise valid rigid fit projects smaller than the
        # visible keypoint support.  Keep the disabled path numerically
        # identical to the legacy loss; detached diagnostics remain available.
        if self.fit_coverage_weight > 0:
            (per_frame_fit_coverage, fit_min_span_ratio,
             fit_frame_ok) = _pnp_fit_coverage(
                    uv_pred, pred_xy_safe, self.fit_span_margin, self.huber_delta,
                self.fit_min_observed_span)
            per_frame = (
                per_frame_base
                + self.fit_coverage_weight * per_frame_fit_coverage)
        else:
            with torch.no_grad():
                (per_frame_fit_coverage, fit_min_span_ratio,
                 fit_frame_ok) = _pnp_fit_coverage(
                    uv_pred.detach(), pred_xy_safe.detach(), self.fit_span_margin,
                    self.huber_delta, self.fit_min_observed_span)
            per_frame = per_frame_base
        fit_hard_example = (
            fit_frame_ok
            & (fit_min_span_ratio.detach() < self.fit_hard_span_threshold))

        # --- guards ---
        depth_ok = (P_pred[:, :, 2] > 0).all(dim=1)
        nan_ok = (
            pred_finite
            & torch.isfinite(per_frame)
            & torch.isfinite(span_ratio).all(dim=1)
            & torch.isfinite(tz_ratio)
        )
        if self.fit_coverage_weight > 0:
            nan_ok = (
                nan_ok
                & torch.isfinite(per_frame_fit_coverage)
                & torch.isfinite(fit_min_span_ratio)
            )
        cond_ok = torch.isfinite(cond) & (cond < self.cond_max)
        valid = mask.bool() & depth_ok & nan_ok & cond_ok
        per_frame_safe = torch.where(valid, per_frame,
                                     torch.zeros_like(per_frame))
        n_valid = valid.sum().clamp(min=1)
        loss = per_frame_safe.sum() / n_valid

        info = {
            "valid_frac": valid.float().mean().item(),
            "n_valid": int(valid.sum().item()),
            "mean_L": (per_frame_safe.sum() / n_valid).item(),
            "mean_geometry_L": (
                torch.where(valid, per_frame_geometry,
                            torch.zeros_like(per_frame_geometry)).sum()
                / n_valid).item(),
            "mean_undercoverage_L": (
                torch.where(valid, per_frame_undercoverage,
                            torch.zeros_like(per_frame_undercoverage)).sum()
                / n_valid).item(),
            "mean_min_span_ratio": (
                torch.where(valid, min_span_ratio,
                            torch.zeros_like(min_span_ratio)).sum()
                / n_valid).item(),
            "mean_tz_ratio": (
                torch.where(valid, tz_ratio, torch.zeros_like(tz_ratio)).sum()
                / n_valid).item(),
            "hard_fraction": (
                (hard_example & valid).float().sum() / n_valid).item(),
            "mean_fit_coverage_L": (
                torch.where(valid & fit_frame_ok, per_frame_fit_coverage,
                            torch.zeros_like(per_frame_fit_coverage)).sum()
                / (valid & fit_frame_ok).sum().clamp(min=1)).item(),
            "mean_fit_min_span_ratio": (
                torch.where(valid & fit_frame_ok, fit_min_span_ratio,
                            torch.zeros_like(fit_min_span_ratio)).sum()
                / (valid & fit_frame_ok).sum().clamp(min=1)).item(),
            "fit_hard_fraction": (
                (fit_hard_example & valid).float().sum()
                / (valid & fit_frame_ok).sum().clamp(min=1)).item(),
            "cond_max": float(cond[torch.isfinite(cond)].max().item())
            if torch.isfinite(cond).any() else float("nan"),
            "skip_depth": int((mask.bool() & ~depth_ok).sum().item()),
            "skip_nan": int((mask.bool() & depth_ok & ~nan_ok).sum().item()),
            "skip_cond": int((mask.bool() & depth_ok & nan_ok & ~cond_ok).sum().item()),
            "gated_out": int((~mask.bool()).sum().item()),
            "per_frame_L": per_frame_safe.detach(),
        }
        return loss, info


# ======================================================================
# 3. 프레임 로더 헬퍼 (audit 기하 재사용)
# ======================================================================
_AUDIT = None


def _audit_mod():
    global _AUDIT
    if _AUDIT is None:
        here = os.path.dirname(os.path.abspath(__file__))
        audit_dir = os.path.abspath(os.path.join(
            here, "..", "..", "scripts", "data_prep", "validate"))
        sys.path.insert(0, audit_dir)
        import pnp_valid_3d_audit as A  # noqa
        _AUDIT = A
    return _AUDIT


def build_diffpnp3d_targets(frame_json_path, index_entry):
    """frame JSON + audit index entry → dict.

    index_entry: {"best_sym", "dims":{W,D,H,diag}, "pnp_valid_3d", "V8", ...}
    best_sym 으로 camera-facing 3D corner 를 재탐색 없이 복원(audit SYMS/canonical_box).
    returns {X_i(8,3), K(3,3), R_gt(3,3), t_gt(3,), diag, pnp_valid, V8}.
    """
    A = _audit_mod()
    K, R, t, cub, pc, dm, kind, Wi, Hi = A.load_frame(frame_json_path)
    dims = index_entry["dims"]
    W, D, H = dims["W"], dims["D"], dims["H"]
    X0 = A.canonical_box(W, D, H)                       # (8,3)
    # audit: Xs = einsum("sij,nj->sni", SYMS, X0) → X_i[n] = SYMS[s] @ X0[n]
    X_i = X0 @ A.SYMS[int(index_entry["best_sym"])].T   # (8,3)
    return {
        "X_i": X_i.astype(np.float64),
        "K": K.astype(np.float64),
        "R_gt": R.astype(np.float64),
        "t_gt": t.astype(np.float64),
        "diag": float(dims["diag"]),
        "pnp_valid": bool(index_entry["pnp_valid_3d"]),
        "V8": bool(index_entry["V8"]),
        "projected_cuboid": pc.astype(np.float64),      # (8,2) orig px (belief gen용)
        "img_wh": (Wi, Hi),
    }


# ============================================================================
# Deployment-aligned refinement: predicted seed, predicted observations
# ============================================================================
# The training-time loss above starts its Gauss-Newton from the GT pose and
# reads a local 7x7 soft-argmax.  Neither is available at inference, so this
# helper takes the pose the canonical OpenCV PnP actually returned and the same
# nine predicted coordinates it was given, and never touches ground truth.
GN_STEPS = 4
GN_DAMPING = 1e-3
GN_DELTA_CLIP = 0.5
GN_COND_MAX = 1e8


def refine_pose_from_predicted_seed(observed_xy, object_points, K,
                                    R_seed, t_seed,
                                    steps=GN_STEPS, damping=GN_DAMPING,
                                    delta_clip=GN_DELTA_CLIP,
                                    cond_max=GN_COND_MAX):
    """Gauss-Newton on the predicted correspondences from the predicted pose.

    observed_xy   (N,2) canonical decoder output in original image pixels
    object_points (N,3) canonical 3D points, centroid included
    K             (3,3) original intrinsics
    R_seed,t_seed the pose canonical OpenCV PnP returned; detached, never GT

    Returns the refined pose plus a health record.  A step whose observed
    reprojection does not fall is rejected and the previous pose kept, and any
    guard failure returns the seed unchanged with the reason recorded.
    """
    device = observed_xy.device if torch.is_tensor(observed_xy) else "cpu"
    obs = torch.as_tensor(observed_xy, dtype=torch.float64, device=device)[None]
    X = torch.as_tensor(object_points, dtype=torch.float64, device=device)[None]
    Km = torch.as_tensor(K, dtype=torch.float64, device=device)[None]
    R0 = torch.as_tensor(R_seed, dtype=torch.float64, device=device).detach()
    t0 = torch.as_tensor(t_seed, dtype=torch.float64, device=device).detach()

    rvec = _rotation_to_rodrigues(R0)[None].clone()
    tvec = t0.reshape(1, 3).clone()

    def residual_norm(r, t):
        uv, cam = _project_batch(r, t, X, Km)
        return float(torch.linalg.norm(uv - obs, dim=-1).mean()), cam

    start_residual, _ = residual_norm(rvec, tvec)
    health = {"accepted": 0, "rejected": 0, "fallback": False,
              "fallback_reason": None, "cond_max": 0.0,
              "rotation_update_norm": 0.0, "translation_update_norm": 0.0,
              "observed_before": start_residual, "observed_after": start_residual}

    best_r, best_t, best_res = rvec.clone(), tvec.clone(), start_residual
    for _ in range(int(steps)):
        uv, _ = _project_batch(best_r, best_t, X, Km)
        residual = (uv - obs).reshape(1, -1, 1)
        J = _jac_batch(best_r, best_t, X, Km)
        JtJ = torch.bmm(J.transpose(1, 2), J)
        eye6 = torch.eye(6, dtype=JtJ.dtype, device=JtJ.device)[None]
        A = JtJ + damping * eye6
        cond = torch.linalg.cond(A)
        health["cond_max"] = max(health["cond_max"], float(cond))
        if not torch.isfinite(cond) or float(cond) > cond_max:
            health["fallback"], health["fallback_reason"] = True, "condition_number"
            break
        try:
            delta = torch.linalg.solve(A, -torch.bmm(J.transpose(1, 2), residual))
        except Exception:
            health["fallback"], health["fallback_reason"] = True, "solve_failed"
            break
        delta = delta.reshape(1, 6)
        if not torch.isfinite(delta).all():
            health["fallback"], health["fallback_reason"] = True, "non_finite_delta"
            break
        norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-12)
        delta = delta * (delta_clip / norm).clamp(max=1.0)
        trial_r = best_r + delta[:, :3]
        trial_t = best_t + delta[:, 3:]
        trial_res, cam = residual_norm(trial_r, trial_t)
        if not np.isfinite(trial_res) or float(cam[..., 2].min()) <= 0.0:
            health["rejected"] += 1
            continue
        if trial_res >= best_res:
            health["rejected"] += 1
            continue
        health["accepted"] += 1
        health["rotation_update_norm"] += float(torch.linalg.norm(delta[:, :3]))
        health["translation_update_norm"] += float(torch.linalg.norm(delta[:, 3:]))
        best_r, best_t, best_res = trial_r, trial_t, trial_res

    R = rodrigues_batch(best_r)[0]
    if health["fallback"] or not torch.isfinite(R).all() \
            or abs(float(torch.linalg.det(R)) - 1.0) > 1e-3:
        health["fallback"] = True
        health["fallback_reason"] = health["fallback_reason"] or "invalid_rotation"
        return {"R": R0.cpu().numpy(), "t": t0.reshape(3).cpu().numpy()}, health
    health["observed_after"] = best_res
    return ({"R": R.cpu().numpy(), "t": best_t.reshape(3).cpu().numpy()}, health)


def _rotation_to_rodrigues(R):
    """SO(3) -> axis-angle, differentiable-friendly and torch-only."""
    trace = torch.clamp((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.arccos(trace)
    if float(angle) < 1e-9:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * torch.sin(angle)) * angle
