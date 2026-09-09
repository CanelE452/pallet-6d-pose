"""A2b — LC_DERIVED_STABLE_MECHANISM_BASELINE.

Linear-Covariance (Liu, Hu, Salzmann, ICCV 2023) 의 pose-covariance loss 를
YOLO26 pose head 에 붙인다.  `ours` / `proposed` / `novel` 이 아니고, LC 의 **정확한 재현도 아니다** —
수치 퇴화를 막는 안정화 가드가 하나 들어가 있다(아래 lc_covariance_term 참조).

무엇을 그대로 따르고 무엇을 단순화했는지는
`_docs/experiments/pallet_translation_loss_v1/METHOD_LOCK_A2B.md` §3~§5 에 있다.
공식 저장소(github.com/fulliu/lc)에는 LICENSE 파일이 없어 코드를 복사하지 않았고,
수식만 읽고 여기서 다시 구현했다.

lambda = 0 이면 이 클래스는 부모 `PoseLoss26` 과 **구성적으로 같은 경로**를 탄다 —
LC 항을 계산하지도 않는다.  CONTROL arm 은 그 상태로 돌린다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils.loss import PoseLoss26

EPS = 1e-9
REPO = Path(__file__).resolve().parents[3]
SIDETABLE = Path(__file__).resolve().parent / "GEOMETRY_SIDETABLE.npz"


@dataclass
class A2BConfig:
    enabled: bool = False
    lambda_geo: float = 0.0
    max_err_len: float = 32.0      # LC 기본값
    rel_thresh: float = 3.0        # LC 기본값
    w_e_thresh: float = 4.0        # LC 기본값
    min_visible: int = 6
    fit_residual_tol_px: float = 0.5   # letterbox 적합 잔차가 이보다 크면 그 표본은 버린다
    ridge: float = 1e-8
    train_stem_list: str | None = None   # §17 leakage guard. 없으면 lookup 을 막지 않는다
    calibration: bool = False            # §20 lambda calibration 모드 (optimizer step 없음)
    log_path: str | None = None

    @classmethod
    def from_env(cls):
        p = os.environ.get("A2B_CONFIG")
        if not p or not os.path.exists(p):
            return cls()
        d = json.load(open(p))
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


def twice_huber(val_abs: torch.Tensor, delta) -> torch.Tensor:
    """LC 의 twice_huber.  delta 는 항상 detach 된 채로 들어온다."""
    if isinstance(delta, torch.Tensor):
        delta = delta.detach()
    return torch.where(val_abs > delta, delta * (2 * val_abs - delta), val_abs ** 2)


def clamp_error(error: torch.Tensor, max_err_len: float) -> torch.Tensor:
    """(..., N, 2) 잔차의 벡터 길이를 max_err_len 으로 자른다.  배율은 detach."""
    with torch.no_grad():
        n = torch.linalg.vector_norm(error, dim=-1) + 1e-6
        f = ((n - max_err_len) / n).unsqueeze(-1)
        delta = f * error * (f > 0)
    return error - delta


def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3)"""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def pose_jacobians(K4: torch.Tensor, R: torch.Tensor, t: torch.Tensor, X: torch.Tensor):
    """(J_2d, J_alt3d) at the GT pose.

    규약은 `scripts/research/pallet_translation_loss_v1/pnp_jacobian.py` 와 같다
    (finite difference 로 검증됨):
        R(xi) = expm(skew(dr)) R,  t(xi) = t + dt,  둘 다 카메라 프레임.
        dP/d(dr) = -skew(R X),  dP/d(dt) = I.

    J_2d   (M, 2N, 6)   투영 좌표의 Jacobian, prepared 픽셀 단위
    J_alt  (M, 3N, 6)   3D corner 위치의 Jacobian (LC 의 xform_3d)
    """
    RX = torch.einsum("mij,mnj->mni", R, X)          # (M, N, 3)
    P = RX + t[:, None, :]
    fx, fy = K4[:, 0:1], K4[:, 1:2]
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    iz = 1.0 / z

    dPdxi = torch.cat([-skew(RX),
                       torch.eye(3, device=X.device, dtype=X.dtype)
                       .expand(RX.shape[0], RX.shape[1], 3, 3)], dim=-1)   # (M,N,3,6)

    dudP = torch.zeros(P.shape[0], P.shape[1], 2, 3, device=X.device, dtype=X.dtype)
    dudP[..., 0, 0] = fx * iz
    dudP[..., 0, 2] = -fx * x * iz * iz
    dudP[..., 1, 1] = fy * iz
    dudP[..., 1, 2] = -fy * y * iz * iz

    J2d = torch.einsum("mnij,mnjk->mnik", dudP, dPdxi).reshape(P.shape[0], -1, 6)
    Jalt = dPdxi.reshape(P.shape[0], -1, 6)
    return J2d, Jalt


def project(K4: torch.Tensor, R: torch.Tensor, t: torch.Tensor, X: torch.Tensor):
    P = torch.einsum("mij,mnj->mni", R, X) + t[:, None, :]
    return torch.stack([K4[:, 0:1] * P[..., 0] / P[..., 2] + K4[:, 2:3],
                        K4[:, 1:2] * P[..., 1] / P[..., 2] + K4[:, 3:4]], dim=-1)


def _grouped_sqrt_mean(diag: torch.Tensor, dim: int) -> torch.Tensor:
    """LC 의 loss_cov_3d: 대각을 점 단위로 묶어 합 -> sqrt -> 점 평균."""
    g = diag.reshape(diag.shape[0], -1, dim)
    good = (diag > 0).all(dim=-1, keepdim=True)
    return torch.where(good, g.sum(-1), torch.ones_like(g.sum(-1))).sqrt().mean(-1)


def lc_covariance_term(K4, R, t, X, gt_px, pred_px, vis, inv_std, cfg,
                       scale=None, stats=None):
    """LC pose-covariance term.  모델 없이 부를 수 있게 분리해 둔다 (테스트용).

    K4      (M,4)   fx, fy, cx, cy   -- prepared 좌표계
    R,t     (M,3,3) (M,3)            -- GT pose, 카메라 프레임
    X       (M,8,3) camera-facing 0..7 순서의 3D 코너
    gt_px   (M,8,2) 입력 이미지 픽셀
    pred_px (M,8,2) 입력 이미지 픽셀 (gradient 는 여기로만 들어간다)
    vis     (M,8)   bool
    inv_std (M,8,2) 예측 1/sigma.  없으면 1 로 채운다
    scale   (M,2)   prepared -> 입력 픽셀 **축별** 배율 (sx, sy).  None 이면 1.

    ★축별이어야 한다.  이 파이프라인은 letterbox 가 아니라 640x640 으로
    비등방 stretch(squash) 한다 -- 등방으로 가정하면 잔차가 0.22 px 로 남고
    축별로 두면 0.0005 px (라벨 양자화 바닥)까지 떨어진다.  stretch 후에도
    fx' = sx fx, fy' = sy fy 인 정상 핀홀이라 PnP 기하는 보존된다.
    """
    # ★ AMP 아래에서는 tensor 가 fp16 로 들어온다.  공분산·역행렬은 fp16 에서
    # 계산할 수 없고(linalg.inv 미지원), 되더라도 PnP 계열은 fp16 에서 뒤집힌 이력이
    # 있다.  autocast 를 끄고 fp32 로 올린다 -- cast 는 미분 가능하므로 gradient 는
    # 그대로 pred 로 돌아간다.
    if pred_px.dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type=pred_px.device.type, enabled=False):
            return lc_covariance_term(
                K4.float(), R.float(), t.float(), X.float(), gt_px.float(),
                pred_px.float(), vis, inv_std.float(), cfg,
                scale=None if scale is None else scale.float(), stats=stats)

    dev, dt = pred_px.device, pred_px.dtype
    J2d, Jalt = pose_jacobians(K4, R, t, X)
    if scale is not None:
        M, N2, _ = J2d.shape
        J2d = (J2d.reshape(M, N2 // 2, 2, 6) * scale[:, None, :, None]).reshape(M, N2, 6)

    err = clamp_error(pred_px - gt_px, cfg.max_err_len)
    vmask = vis.to(dt)[..., None]

    ae = err.abs()
    with torch.no_grad():
        mean_abs = (ae * vmask).sum(1) / vmask.sum(1).clamp_min(1.0)
    cov = twice_huber(ae, mean_abs[:, None, :] * cfg.rel_thresh)

    with torch.no_grad():
        w_e = (inv_std ** 2) * cov
        mean_we = (w_e * vmask).sum(1) / vmask.sum(1).clamp_min(1.0)
        delta_inv = torch.sqrt((mean_we[:, None, :] * cfg.w_e_thresh) / (cov + 1e-6))
        # ★ 퇴화 방지.  잔차가 전부 0 이면 mean_we = 0 -> delta_inv = 0 이고
        # twice_huber(a, 0) = 0 이라 가중치가 통째로 사라진다.  그러면 H = ridge*I 가
        # 되어 prior 가 폭발하고, **잘 맞춘 표본일수록 loss 가 커진다.**
        # Huber 화의 목적은 큰 가중치를 자르는 것이므로, 임계가 퇴화하면
        # 자르지 않는 쪽(delta = inf, 즉 a^2)으로 되돌린다.
        degenerate = mean_we[:, None, :] <= 1e-12
        delta_inv = torch.where(degenerate, torch.full_like(delta_inv, float("inf")),
                                delta_inv)
    weights = twice_huber(inv_std, delta_inv) * vmask

    wf = weights.reshape(weights.shape[0], -1)
    cf = (cov * vmask).reshape(cov.shape[0], -1)
    rf = (err * vmask).reshape(err.shape[0], -1)

    JtW = J2d.transpose(1, 2) * wf[:, None, :]
    H_raw = JtW @ J2d
    H = H_raw + cfg.ridge * torch.eye(6, device=dev, dtype=dt)
    Hinv = torch.linalg.inv(H)
    A = Hinv @ JtW
    if stats is not None:
        with torch.no_grad():
            sv = torch.linalg.svdvals(H_raw.float())
            cond = (sv[:, 0] / sv[:, -1].clamp_min(1e-30))
            stats["cond"] = cond.cpu().numpy()
            stats["rank_deficient"] = int((sv[:, -1] <= 1e-12 * sv[:, 0]).sum())
            stats["ridge_active"] = int((sv[:, -1] < cfg.ridge).sum())

    Sigma = A @ (cf[:, :, None] * A.transpose(1, 2))
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))

    cov_err = _grouped_sqrt_mean(((Jalt @ Sigma) * Jalt).sum(-1), 3)
    prior_err = _grouped_sqrt_mean(((Jalt @ Hinv) * Jalt).sum(-1), 3).clamp_min(1e-12)
    delta = (Jalt @ (A @ rf.detach()[:, :, None])).squeeze(-1)
    lin_err = delta.reshape(delta.shape[0], -1, 3).norm(dim=-1).mean(-1)

    per = prior_err.log() + 0.5 * (cov_err + lin_err) / prior_err
    if stats is not None:
        stats["cov_err"] = float(cov_err.mean().detach())
        stats["lin_err"] = float(lin_err.mean().detach())
        stats["prior_err"] = float(prior_err.mean().detach())
    return per


class A2BPoseLoss26(PoseLoss26):
    """PoseLoss26 + LC pose-covariance term.  lambda_geo=0 이면 부모와 동일."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.a2b = A2BConfig.from_env()
        self._stems = None
        self.a2b_stats = {"n_used": 0, "n_skipped_visibility": 0,
                          "n_skipped_fit": 0, "last_term": 0.0, "fit_p99_px": 0.0}
        # §17 leakage · §21 conditioning
        self.lookup_stats = {"train_lookup_count": 0, "val_lookup_count_during_train": 0,
                             "missing_train_lookup": 0, "unexpected_sample": 0}
        self.calib_rows = []        # §20 calibration: per-instance gradient norm
        self.cond_log = []          # cond(J^T W J) 표본. 집계용, 매 스텝 저장 아님
        self.rank_deficient = 0
        self.ridge_fallback = 0
        self._table = None
        self._train_stems = None
        self.batch_stem_log = []     # §19 — 두 arm 의 batch 순서 비교용 (CONTROL 포함)
        if self.a2b.train_stem_list:
            with open(self.a2b.train_stem_list, encoding="utf-8") as f:
                self._train_stems = {ln.strip() for ln in f if ln.strip()}
        if self.a2b.enabled and self.a2b.lambda_geo != 0.0:
            self._load_table()

    def _load_table(self):
        d = np.load(SIDETABLE, allow_pickle=True)
        self._index = {s: i for i, s in enumerate(d["stems"].tolist())}
        dev = self.device
        self._table = {
            "K": torch.as_tensor(d["K"], dtype=torch.float32, device=dev),
            "R": torch.as_tensor(d["R"], dtype=torch.float32, device=dev),
            "t": torch.as_tensor(d["t"], dtype=torch.float32, device=dev),
            "X": torch.as_tensor(d["Xcf"], dtype=torch.float32, device=dev),
        }

    # ------------------------------------------------------------------ hook --
    def loss(self, preds, batch):
        # im_file 은 여기서만 보인다.  mosaic=0 이라 이미지 1 장 = 표본 1 개다.
        self._stems = [Path(f).stem for f in batch["im_file"]] if "im_file" in batch else None
        # §19 — 첫 100 training batch 의 sample_id 목록.  lambda 와 무관하게 기록하므로
        # CONTROL 과 A2b 의 batch 순서를 직접 비교할 수 있다.
        if (self._stems and len(self.batch_stem_log) < 100
                and self._train_stems is not None
                and all(st_ in self._train_stems for st_ in self._stems)):
            self.batch_stem_log.append(list(self._stems))
        return super().loss(preds, batch)

    # ------------------------------------------------------------------- LC --
    def _lc_term(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, pred_kpts):
        cfg = self.a2b
        zero = pred_kpts.new_zeros(())
        if self._table is None or self._stems is None or not masks.any():
            return zero

        # §17 — loss lookup 은 TRAIN manifest 안에서만 일어나야 한다.
        # validation batch(전부 non-train)는 LC 항을 **계산하지 않고** 건너뛴다.
        # val 의 GT pose/K 는 어떤 loss 에도 들어가지 않는다.
        if self._train_stems is not None:
            in_train = [st_ in self._train_stems for st_ in self._stems]
            if not any(in_train):
                self.lookup_stats["val_batches_skipped"] = \
                    self.lookup_stats.get("val_batches_skipped", 0) + 1
                return zero
            if not all(in_train):
                raise RuntimeError(
                    "A2b: a batch mixes train and non-train samples — "
                    "the leakage guard cannot separate them")
            for st_ in self._stems:
                if st_ in self._index:
                    self.lookup_stats["train_lookup_count"] += 1
                else:
                    self.lookup_stats["unexpected_sample"] += 1
                    raise RuntimeError(f"A2b: sample {st_!r} is not in the side table")

        rows = [self._index.get(s, -1) for s in self._stems]
        rows_t = torch.as_tensor(rows, device=pred_kpts.device)
        img = torch.nonzero(masks, as_tuple=False)[:, 0]          # (M,) 이미지 인덱스
        keep_known = rows_t[img] >= 0
        if not keep_known.any():
            return zero

        sel = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        gt = sel[masks]                                            # (M, K, 2 or 3) 입력 픽셀
        pk = pred_kpts[masks]                                      # (M, K, C) stride 단위
        st = stride_tensor.view(1, -1, 1).expand(masks.shape[0], -1, 1)[masks]   # (M,1)

        gt_px = gt[..., :2][:, :8, :]
        pred_px = pk[..., :2][:, :8, :] * st[:, None, :]
        vis = (gt[..., 2] > 0)[:, :8] if gt.shape[-1] == 3 else torch.ones_like(gt_px[..., 0], dtype=torch.bool)

        ridx = rows_t[img].clamp_min(0)
        K4 = self._table["K"][ridx]
        R = self._table["R"][ridx]
        t = self._table["t"][ridx]
        X = self._table["X"][ridx]

        # letterbox 는 등방 배율 + 평행이동이다.  변환을 역추적하지 않고
        # GT 로 직접 적합한다 -- 잔차가 곧 배선 검증이다.
        with torch.no_grad():
            u_ref = project(K4, R, t, X)                           # (M, 8, 2) prepared
            w = vis.to(u_ref.dtype)[..., None]
            n = w.sum(1).clamp_min(1.0)
            um = (u_ref * w).sum(1) / n
            gm = (gt_px * w).sum(1) / n
            du, dg = (u_ref - um[:, None]) * w, (gt_px - gm[:, None]) * w
            # 축별 배율.  등방으로 두면 stretch 를 흡수하지 못한다.
            s = (du * dg).sum(1) / (du * du).sum(1).clamp_min(EPS)   # (M, 2)
            T = gm - s * um
            fit = ((s[:, None, :] * u_ref + T[:, None] - gt_px).norm(dim=-1) * vis).max(1).values

        enough = vis.sum(1) >= cfg.min_visible
        ok = keep_known & enough & (fit <= cfg.fit_residual_tol_px)
        self.a2b_stats["n_skipped_visibility"] = int((~enough).sum())
        self.a2b_stats["n_skipped_fit"] = int((enough & (fit > cfg.fit_residual_tol_px)).sum())
        self.a2b_stats["fit_p99_px"] = float(torch.quantile(fit.float(), 0.99)) if fit.numel() else 0.0
        if not ok.any():
            return zero

        idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
        K4, R, t, X, s = K4[idx], R[idx], t[idx], X[idx], s[idx]
        gt_px, pred_px, vis = gt_px[idx], pred_px[idx], vis[idx]
        pk_sel = pk[idx]

        if pk_sel.shape[-1] >= 4:
            inv_std = 1.0 / (pk_sel[..., 2:4][:, :8, :].sigmoid() + 1e-6)
        else:
            inv_std = torch.ones_like(gt_px)

        st_ = {}
        term = lc_covariance_term(K4, R, t, X, gt_px, pred_px, vis, inv_std,
                                  cfg, scale=s, stats=st_).mean()
        if "cond" in st_ and len(self.cond_log) < 200000:
            self.cond_log.extend(st_["cond"].tolist())
            self.rank_deficient += st_["rank_deficient"]
            self.ridge_fallback += st_["ridge_active"]
        self.a2b_stats["n_used"] = int(ok.sum())
        self.a2b_stats["last_term"] = float(term.detach())
        return term

    # -------------------------------------------------------------- override --
    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        if self.a2b.calibration:
            return self._calibration_pass(masks, target_gt_idx, keypoints, batch_idx,
                                          stride_tensor, target_bboxes, pred_kpts)
        kpts_loss, kpts_obj_loss, rle_loss = super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
            target_bboxes, pred_kpts)
        if self.a2b.enabled and self.a2b.lambda_geo != 0.0:
            lc = self._lc_term(masks, target_gt_idx, keypoints, batch_idx,
                               stride_tensor, pred_kpts)
            kpts_loss = kpts_loss + self.a2b.lambda_geo * lc
        return kpts_loss, kpts_obj_loss, rle_loss

    # ------------------------------------------------------------ calibration --
    def _calibration_pass(self, masks, target_gt_idx, keypoints, batch_idx,
                          stride_tensor, target_bboxes, pred_kpts):
        """§20 — 같은 pred_kpts tensor 에 대한 base 와 LC 의 gradient norm.

        optimizer step 은 하지 않는다.  lambda 는 여기서 나온 비율로 정한다.
        """
        pk = pred_kpts if pred_kpts.requires_grad else pred_kpts.detach().requires_grad_(True)
        base, obj, rle = super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
            target_bboxes.clone(), pk)
        g_base = torch.autograd.grad(base, pk, retain_graph=True, allow_unused=True)[0]
        lc = self._lc_term(masks, target_gt_idx, keypoints, batch_idx, stride_tensor, pk)
        if lc.requires_grad:
            g_lc = torch.autograd.grad(lc, pk, retain_graph=False, allow_unused=True)[0]
        else:
            g_lc = None
        if g_base is None:
            g_base = torch.zeros_like(pk)
        if g_lc is None:
            g_lc = torch.zeros_like(pk)
        # per positive instance: keypoint xy 성분만
        n = int(masks.sum())
        gb = g_base[masks][..., :2].reshape(n, -1).norm(dim=1)
        gl = g_lc[masks][..., :2].reshape(n, -1).norm(dim=1)
        img = torch.nonzero(masks, as_tuple=False)[:, 0].cpu().tolist()
        stems = self._stems or []
        for k, (a, b) in enumerate(zip(gb.detach().cpu().tolist(), gl.detach().cpu().tolist())):
            self.calib_rows.append(
                (stems[img[k]] if img[k] < len(stems) else "?", a, b))
        z = pk.new_zeros(())
        return z, z, z
