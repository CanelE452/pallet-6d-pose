"""DiffPnP 3D-corner loss — YOLO pose head 위에서 미분가능 PnP 로 pose 를 읽는다.

평가가 pose 를 읽는 연산(예측 2D → PnP)을 학습 안으로 가져온다. 예측 keypoint 로
Gauss-Newton PnP 를 풀고, 그 pose 가 만드는 카메라 좌표계 코너와 참조 pose 의 코너를
비교한다. 참조 pose 는 **GT 2D 로 푼 PnP** 라서 규약 오프셋(perm_v4 180도)이 예측·참조
양쪽에서 상쇄된다.

증강 대응: mosaic·affine 로 이미지가 변형되면 원본 K 를 못 쓴다. 영상면 affine A 는
투영에 그대로 곱해지므로 (``A K X / z``) **K' = A K** 가 정확한 보정이다. A 는 소스
projected_cuboid 와 배치 안의 GT keypoint 대응에서 최소제곱으로 복원한다.

``lambda_dp = 0`` 이면 부모 loss 와 구성적으로 동일하다 — 항을 아예 계산하지 않는다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils.loss import PoseLoss26

EPS = 1e-9

# data.yaml 의 flip_idx 앞 8개 — 좌우 반전 시 코너 인덱스가 이렇게 바뀐다
FLIP8 = (1, 0, 3, 2, 5, 4, 7, 6)


@dataclass
class DiffPnPConfig:
    enabled: bool = False
    lambda_dp: float = 0.0
    huber_delta_norm: float = 0.10     # 대각선 정규화 후 값
    gn_steps: int = 5
    damping: float = 1e-3
    delta_clip: float = 0.5
    min_visible: int = 6               # PnP 에 필요한 최소 가시 코너
    affine_residual_max_px: float = 1.0
    warmup_steps: int = 0
    index_dir: str = "data/pallet/results/diffpnp_yolo_v1"
    log_path: str | None = None

    @classmethod
    def from_env(cls):
        p = os.environ.get("DIFFPNP_CONFIG")
        if not p or not os.path.exists(p):
            return cls()
        d = json.load(open(p))
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class DiffPnPIndex:
    """stem -> (K, X_centered, uv_src, R_ref, t_ref_centered)."""

    def __init__(self, index_dir: str | Path):
        d = Path(index_dir)
        self.stems = json.loads((d / "diffpnp_index_stems.json").read_text())
        z = np.load(d / "diffpnp_index.npz")
        K = z["K"].astype(np.float64)
        X = z["X"].astype(np.float64)
        R = z["R"].astype(np.float64)
        t = z["t"].astype(np.float64)
        # world 좌표가 원점에서 멀어 GN 조건수가 나쁘다. 코너 중심으로 옮기고
        # t 를 그만큼 보정한다 — R X + t 는 불변이다.
        centre = X.mean(axis=1, keepdims=True)
        self.X = X - centre
        self.t = t + np.einsum("nij,nj->ni", R, centre[:, 0, :])
        self.K = K
        self.R = R
        self.uv = z["uv"].astype(np.float64)
        self.diag = np.linalg.norm(self.X.max(1) - self.X.min(1), axis=1)

    def lookup(self, keys):
        rows = [self.stems.get(k, -1) for k in keys]
        return np.asarray(rows, np.int64)


def exp_so3(w):
    theta = w.norm(dim=1).clamp(min=1e-12)
    k = w / theta.unsqueeze(-1)
    O = torch.zeros_like(theta)
    K = torch.stack([
        torch.stack([O, -k[:, 2], k[:, 1]], dim=-1),
        torch.stack([k[:, 2], O, -k[:, 0]], dim=-1),
        torch.stack([-k[:, 1], k[:, 0], O], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(w.shape[0], -1, -1)
    return (eye + torch.sin(theta).view(-1, 1, 1) * K
            + (1.0 - torch.cos(theta)).view(-1, 1, 1) * torch.bmm(K, K))


def project(R, t, X, K):
    P = torch.bmm(X, R.transpose(1, 2)) + t.unsqueeze(1)
    uvw = torch.bmm(P, K.transpose(1, 2))
    z = uvw[:, :, 2:3].clamp(min=1e-6)
    return uvw[:, :, :2] / z, P


def jacobian(R, t, X, K):
    B, N, _ = X.shape
    P = torch.bmm(X, R.transpose(1, 2)) + t.unsqueeze(1)
    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    sk = K[:, 0, 1].view(B, 1)
    Xc, Yc = P[:, :, 0], P[:, :, 1]
    Zc = P[:, :, 2].clamp(min=1e-6)
    Z2 = Zc * Zc
    dpdP = torch.zeros(B, N, 2, 3, dtype=X.dtype, device=X.device)
    dpdP[:, :, 0, 0] = fx / Zc
    dpdP[:, :, 0, 1] = sk / Zc
    dpdP[:, :, 0, 2] = -(fx * Xc + sk * Yc) / Z2
    dpdP[:, :, 1, 1] = fy / Zc
    dpdP[:, :, 1, 2] = -fy * Yc / Z2
    RX = P - t.unsqueeze(1)
    dPdw = torch.zeros(B, N, 3, 3, dtype=X.dtype, device=X.device)
    dPdw[:, :, 0, 1] = RX[:, :, 2];  dPdw[:, :, 0, 2] = -RX[:, :, 1]
    dPdw[:, :, 1, 0] = -RX[:, :, 2]; dPdw[:, :, 1, 2] = RX[:, :, 0]
    dPdw[:, :, 2, 0] = RX[:, :, 1];  dPdw[:, :, 2, 1] = -RX[:, :, 0]
    dPdt = torch.eye(3, dtype=X.dtype, device=X.device).view(1, 1, 3, 3).expand(B, N, -1, -1)
    J = torch.cat([torch.matmul(dpdP, dPdw), torch.matmul(dpdP, dPdt)], dim=-1)
    return J.reshape(B, 2 * N, 6)


def recover_affine(uv_src, uv_dst, weight):
    """uv_src (M,N,2) -> uv_dst (M,N,2) 의 2D affine A (M,2,3) 를 가중 최소제곱으로.

    반환 (A, residual_px, ok).  영상면 affine 은 투영에 그대로 곱해지므로 K'=A K.
    """
    M, N, _ = uv_src.shape
    # Hartley 정규화 — 정규방정식은 조건수를 제곱하므로 원 픽셀 좌표(수백 단위)를
    # 그대로 넣으면 유효자리가 날아간다. 풀고 나서 되돌린다.
    wsum = weight.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    mu = (uv_src * weight.unsqueeze(-1)).sum(dim=1) / wsum               # (M,2)
    scale = (((uv_src - mu.unsqueeze(1)).norm(dim=-1) * weight).sum(dim=1)
             / wsum.squeeze(-1)).clamp(min=EPS)                          # (M,)
    src_n = (uv_src - mu.unsqueeze(1)) / scale.view(-1, 1, 1)

    p = torch.cat([src_n, torch.ones(M, N, 1, dtype=uv_src.dtype,
                                     device=uv_src.device)], dim=-1)     # (M,N,3)
    # 대상 좌표도 같은 이유로 정규화한다 (편향 없이 되돌릴 수 있는 상수 변환).
    mu_d = (uv_dst * weight.unsqueeze(-1)).sum(dim=1) / wsum
    scale_d = (((uv_dst - mu_d.unsqueeze(1)).norm(dim=-1) * weight).sum(dim=1)
               / wsum.squeeze(-1)).clamp(min=EPS)
    dst_n = (uv_dst - mu_d.unsqueeze(1)) / scale_d.view(-1, 1, 1)

    w = weight.unsqueeze(-1)
    G = torch.einsum("mni,mnj->mij", p * w, p)                          # (M,3,3)
    b = torch.einsum("mni,mnj->mij", p * w, dst_n)                      # (M,3,2)
    eye = torch.eye(3, dtype=uv_src.dtype, device=uv_src.device).expand(M, -1, -1)
    # 절대 릿지는 해에 편향을 남긴다. 특이한 배치만 막도록 상대값으로 준다.
    G = G + 1e-12 * G.diagonal(dim1=1, dim2=2).sum(dim=1).view(-1, 1, 1) * eye
    ok = torch.isfinite(G).flatten(1).all(dim=1) & (weight.sum(dim=1) >= 3)
    G = torch.where(ok.view(-1, 1, 1), G, eye)
    sol = torch.linalg.solve(G, b)                                      # (M,3,2)
    An = sol.transpose(1, 2)                                            # (M,2,3) 정규화 좌표계
    # 되돌리기:  dst = scale_d * (A' [(u-mu)/s, 1]) + mu_d
    lin = An[:, :, :2] * (scale_d / scale).view(-1, 1, 1)
    off = (An[:, :, 2] * scale_d.view(-1, 1) + mu_d
           - torch.einsum("mij,mj->mi", lin, mu))
    A = torch.cat([lin, off.unsqueeze(-1)], dim=-1)                     # (M,2,3)

    p_raw = torch.cat([uv_src, torch.ones(M, N, 1, dtype=uv_src.dtype,
                                          device=uv_src.device)], dim=-1)
    pred = torch.einsum("mij,mnj->mni", A, p_raw)
    err = (pred - uv_dst).norm(dim=-1)
    residual = (err * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
    ok = ok & torch.isfinite(residual) & torch.isfinite(A).flatten(1).all(dim=1)
    return A, residual, ok


def diffpnp_corner_loss(pred_px, vis, K, X, R_ref, t_ref, diag, cfg):
    """예측 2D 로 GN PnP 를 풀고 참조 pose 의 카메라 좌표 코너와 비교한다.

    pred_px (M,8,2) · vis (M,8) · K (M,3,3) · X (M,8,3) · R_ref (M,3,3)
    · t_ref (M,3) · diag (M,)
    반환 (scalar loss, valid mask, stats)
    """
    M = pred_px.shape[0]
    R = R_ref.clone()
    t = t_ref.clone()
    eye6 = torch.eye(6, dtype=pred_px.dtype, device=pred_px.device).expand(M, -1, -1)
    w = vis.to(pred_px.dtype)
    sw = torch.sqrt(w).unsqueeze(-1)

    for _ in range(int(cfg.gn_steps)):
        uv, _ = project(R, t, X, K)
        res = (uv - pred_px) * sw
        J = jacobian(R, t, X, K) * sw.repeat_interleave(2, dim=1)
        A = torch.bmm(J.transpose(1, 2), J) + cfg.damping * eye6
        g = torch.bmm(J.transpose(1, 2), res.reshape(M, -1, 1))
        delta = torch.linalg.solve(A, -g).reshape(M, 6)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-12)
        delta = delta * (cfg.delta_clip / norm).clamp(max=1.0)
        R = torch.bmm(exp_so3(delta[:, :3]), R)
        t = t + delta[:, 3:]

    P_gn = torch.bmm(X, R.transpose(1, 2)) + t.unsqueeze(1)
    P_ref = torch.bmm(X, R_ref.transpose(1, 2)) + t_ref.unsqueeze(1)
    d = (P_gn - P_ref).norm(dim=-1) / diag.view(-1, 1).clamp(min=EPS)

    delta_h = cfg.huber_delta_norm
    huber = torch.where(d <= delta_h, 0.5 * d * d, delta_h * (d - 0.5 * delta_h))
    per_frame = huber.mean(dim=1)

    valid = (torch.isfinite(per_frame)
             & torch.isfinite(P_gn).flatten(1).all(dim=1)
             & (P_gn[:, :, 2].min(dim=1).values > 0.0))
    stats = {"n_valid": int(valid.sum().item()),
             "mean_corner_norm": float(d[valid].mean().item()) if valid.any() else 0.0}
    if not valid.any():
        return pred_px.sum() * 0.0, valid, stats
    return per_frame[valid].mean(), valid, stats


class DiffPnPPoseLoss26(PoseLoss26):
    """PoseLoss26 + DiffPnP 3D-corner 항.  lambda_dp=0 이면 부모와 동일하다."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.dp = DiffPnPConfig.from_env()
        self.dp_index = None
        self.dp_stats = {"n_valid": 0, "n_lookup_miss": 0, "n_affine_reject": 0,
                         "n_flipped": 0, "last_dp": 0.0, "mean_corner_norm": 0.0}
        self._dp_files = None
        self._dp_step = 0
        if self.dp.enabled and self.dp.lambda_dp != 0.0:
            self.dp_index = DiffPnPIndex(self.dp.index_dir)

    # 배치의 파일 경로를 저장해 둔다 — keypoint loss 단계에선 batch 가 안 넘어온다.
    # ★ end2end 경로에서 E2ELoss 는 __call__ 이 아니라 loss() 를 부른다. 둘 다 덮지
    #   않으면 one2many/one2one 학습에서 항이 조용히 빠진다 (실제로 한 번 겪었다).
    def loss(self, preds, batch):
        self._dp_files = batch.get("im_file")
        return super().loss(preds, batch)

    def __call__(self, preds, batch):
        self._dp_files = batch.get("im_file")
        return super().__call__(preds, batch)

    @staticmethod
    def _keys(files):
        out = []
        for f in files:
            p = Path(f)
            out.append(f"{p.parent.name}/{p.stem}")
        return out

    def diffpnp_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                     stride_tensor, pred_kpts):
        zero = torch.zeros(1, device=pred_kpts.device,
                           dtype=pred_kpts.dtype).squeeze()
        if self.dp_index is None or self._dp_files is None or not masks.any():
            return None
        sel = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        gt_px = sel[masks][..., :2]                       # (M,9,2) 네트워크 입력 px
        has_v = sel.shape[-1] == 3
        vis9 = (sel[masks][..., 2] > 0) if has_v else torch.ones_like(gt_px[..., 0],
                                                                     dtype=torch.bool)
        stride = stride_tensor.view(1, -1, 1, 1).expand_as(pred_kpts[..., :1])[masks]
        pred_px = pred_kpts[masks][..., :2] * stride     # (M,9,2)

        image_of = masks.nonzero(as_tuple=False)[:, 0].tolist()
        keys = self._keys([self._dp_files[b] for b in image_of])
        rows = self.dp_index.lookup(keys)
        hit = rows >= 0
        self.dp_stats["n_lookup_miss"] = int((~hit).sum())
        if not hit.any():
            return None

        idx = torch.as_tensor(rows[hit], device=pred_kpts.device)
        keep = torch.as_tensor(np.nonzero(hit)[0], device=pred_kpts.device)
        dev, dt = pred_kpts.device, torch.float64

        def take(a):
            return torch.as_tensor(a, dtype=dt, device=dev)[idx]

        K = take(self.dp_index.K)
        X = take(self.dp_index.X)
        uv_src = take(self.dp_index.uv)
        R_ref = take(self.dp_index.R)
        t_ref = take(self.dp_index.t)
        diag = take(self.dp_index.diag)

        gt8 = gt_px[keep][:, :8].to(dt)
        vis8 = vis9[keep][:, :8].to(dt)
        pred8 = pred_px[keep][:, :8].to(dt)

        # 좌우 반전 증강은 keypoint 인덱스를 flip_idx 로 바꿔버린다. 그러면 소스와의
        # 대응이 어긋나 affine 이 안 맞는다. 두 대응을 다 풀고 잔차가 낮은 쪽을 쓴다.
        flip = torch.as_tensor(FLIP8, device=dev)
        # 가중치 vis8 은 예측·GT 인덱스 k 기준이라 순열을 적용하지 않는다.
        # 뒤집힌 경우 pred[k] 가 가리키는 3D 점이 X[flip[k]] 이므로 X 만 permute 한다.
        A0, r0, ok0 = recover_affine(uv_src, gt8, vis8)
        A1, r1, ok1 = recover_affine(uv_src[:, flip], gt8, vis8)
        use_flip = (r1 < r0) & ok1
        A = torch.where(use_flip.view(-1, 1, 1), A1, A0)
        residual = torch.where(use_flip, r1, r0)
        ok_affine = torch.where(use_flip, ok1, ok0)
        X = torch.where(use_flip.view(-1, 1, 1), X[:, flip], X)
        self.dp_stats["n_flipped"] = int(use_flip.sum().item())

        ok = (ok_affine & (residual <= self.dp.affine_residual_max_px)
              & (vis8.sum(dim=1) >= self.dp.min_visible))
        self.dp_stats["n_affine_reject"] = int((~ok).sum().item())
        if not ok.any():
            return None

        row = torch.zeros(A.shape[0], 1, 3, dtype=dt, device=dev)
        row[:, 0, 2] = 1.0
        K_eff = torch.bmm(torch.cat([A, row], dim=1), K)

        sub = ok.nonzero(as_tuple=True)[0]
        loss, valid, stats = diffpnp_corner_loss(
            pred8[sub], vis8[sub], K_eff[sub], X[sub], R_ref[sub], t_ref[sub],
            diag[sub], self.dp)
        self.dp_stats.update({"n_valid": stats["n_valid"],
                              "mean_corner_norm": stats["mean_corner_norm"],
                              "last_dp": float(loss.detach().item())})
        return loss.to(pred_kpts.dtype)

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        dp = None
        if self.dp.enabled and self.dp.lambda_dp != 0.0:
            self._dp_step += 1
            if self._dp_step > self.dp.warmup_steps:
                dp = self.diffpnp_loss(masks, target_gt_idx, keypoints, batch_idx,
                                       stride_tensor, pred_kpts)
        kpts_loss, kpts_obj_loss, rle_loss = super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
            target_bboxes, pred_kpts)
        if dp is not None:
            if os.environ.get("DIFFPNP_PROBE") == "1":
                self._probe_grad_band(kpts_loss, dp, pred_kpts)
            kpts_loss = kpts_loss + self.dp.lambda_dp * dp
        return kpts_loss, kpts_obj_loss, rle_loss

    def _probe_grad_band(self, base, dp, pred_kpts):
        """두 항이 pred_kpts 를 미는 힘의 비율. lambda 를 정하는 근거다 (Q0)."""
        try:
            gb = torch.autograd.grad(base, pred_kpts, retain_graph=True,
                                     allow_unused=True)[0]
            gd = torch.autograd.grad(dp, pred_kpts, retain_graph=True,
                                     allow_unused=True)[0]
        except RuntimeError:
            return
        if gb is None or gd is None:
            return
        nb = float(gb.norm())
        nd = float(gd.norm())
        self.dp_stats["grad_base"] = nb
        self.dp_stats["grad_dp_at_lambda1"] = nd
        self.dp_stats["lambda_for_5pct"] = (0.05 * nb / nd) if nd > 0 else 0.0
