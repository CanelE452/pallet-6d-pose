"""A1 — symmetry / role-aware keypoint objective.

★ CASE 를 코드에 굳히지 않는다. `SYMMETRY_MANIFEST` 의 asset 별 class 가
  분기를 정한다.  contract 가 도착해야 CASE 1(전부 SYM) 인지 CASE 2(혼재)인지
  결정되고, UNRESOLVED 가 하나라도 있으면 build 단계에서 막는다.

★ A2 의 projective term 은 여기 없다.  A1 config 는 lambda_pc 를 강제로 0 으로
  두고, PSPCPoseLoss26 의 PC 경로는 `enabled=False` 로 꺼진다 (test 로 실증).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from ultralytics.utils.ops import xyxy2xywh
from .loss import PSPCPoseLoss26, EPS


# ASC — 학습 중 epoch 을 loss 로 전달하는 유일한 통로. trainer callback 이 채운다.
CURRENT_EPOCH = {"e": 0}
ROLE_CALLS = {"n": 0}


def normalized_residual(pred, gt, area):
    """r = ||pred-gt||_2 / sqrt(area+eps).  stride 정규화 좌표에서 계산하므로 scale-free."""
    d = (pred[..., :2] - gt[..., :2]).pow(2).sum(-1).clamp_min(0).sqrt()
    return d / area.clamp_min(EPS).sqrt()


def smooth_l1_to_zero(x, beta):
    """SmoothL1(x, target=0, beta).  beta<=0 이면 L1."""
    a = x.abs()
    if beta <= 0:
        return a
    return torch.where(a < beta, 0.5 * a.pow(2) / beta, a - 0.5 * beta)


def takl_tail(pred, gt, mask, area, tau, z_clip=0.0):
    """tail auxiliary.  r<=tau 인 점은 정확히 0 을 기여한다."""
    r = normalized_residual(pred, gt, area)
    z = torch.relu(r - tau)
    if z_clip > 0:
        z = z.clamp(max=z_clip)
    v = mask.float()
    return (smooth_l1_to_zero(z, tau) * v).sum() / v.sum().clamp_min(1.0)


def pevl_loss(pred, gt, mask, area, edges, alpha_len, resid_q95):
    """12 edge 의 방향(cosine) + 상대 길이(log ratio).  centroid 는 제외.

    catastrophic edge(정규화 잔차 > q95 인 코너를 포함)는 auxiliary 를 끈다 —
    standard loss 는 그대로 두고 PEVL 만 OFF.
    """
    r = normalized_residual(pred, gt, area)          # (M, K)
    bad = r > resid_q95 if resid_q95 > 0 else torch.zeros_like(r, dtype=torch.bool)
    ldir = pred.new_zeros(())
    llen = pred.new_zeros(())
    n = 0
    for i, j in edges:
        v = mask[:, i] & mask[:, j] & (~bad[:, i]) & (~bad[:, j])
        if not bool(v.any()):
            continue
        pe = pred[v][:, j, :2] - pred[v][:, i, :2]
        ge = gt[v][:, j, :2] - gt[v][:, i, :2]
        pn = pe.norm(dim=1).clamp_min(EPS)
        gn = ge.norm(dim=1).clamp_min(EPS)
        cos = (pe * ge).sum(1) / (pn * gn)
        ldir = ldir + (1.0 - cos).mean()
        llen = llen + smooth_l1_to_zero(torch.log((pn + EPS) / (gn + EPS)), 0.1).mean()
        n += 1
    if n == 0:
        return pred.new_zeros(())
    return ldir / n + alpha_len * (llen / n)


def nrl_coord(pred, gt, mask, area, beta):
    """coordinate localization 을 축별 SmoothL1 로 교체.  bbox 크기로 정규화."""
    sx = area.clamp_min(EPS).sqrt()
    dx = (pred[..., 0] - gt[..., 0]) / sx
    dy = (pred[..., 1] - gt[..., 1]) / sx
    v = mask.float()
    t = smooth_l1_to_zero(dx, beta) + smooth_l1_to_zero(dy, beta)
    return (t * v).sum() / v.sum().clamp_min(1.0)


def asc_beta(epoch, full_end, ramp_end):
    """epoch < full_end : 1.0 / [full_end, ramp_end) : 선형 하강 / >= ramp_end : 0.0

    epoch 은 ultralytics 의 self.epoch (0-based).  checkpoint 파일명과 같은 색인이라
    epoch0.pt = 첫 epoch 종료 시점이다.
    """
    if epoch < full_end:
        return 1.0
    if epoch >= ramp_end:
        return 0.0
    return (ramp_end - epoch) / float(ramp_end - full_end)


@dataclass
class A1Config:
    enabled: bool = False
    mode: str = "exact_min"          # exact_min | softmin  (사전등록으로 고정)
    softmin_tau: float = 0.0
    lambda_role: float = 0.0
    margin: float = 0.0
    p180: tuple = (5, 4, 7, 6, 1, 0, 3, 2)
    centroid_index: int = 8
    sym_assets: tuple = ()
    asym_assets: tuple = ()
    stem_asset_map: str = ""
    role_ramp: tuple = (5, 20)       # epoch 0~5 = 0, 5~20 선형, 이후 고정
    asc_enabled: bool = False        # ASC — 초기 symmetry, 후기 identity 복귀
    asc_full_end: int = 20           # 0..19 beta=1.0
    asc_ramp_end: int = 30           # 20..29 선형 하강, 30.. beta=0.0
    # TAKL — standard 를 **제거하지 않고** tail auxiliary 를 더한다
    takl_enabled: bool = False
    takl_tau: float = 0.0            # train A0 residual q75 (사전 고정)
    takl_lambda: float = 0.0         # gradient-norm 매칭으로 사전 고정
    takl_z_clip: float = 0.0         # >0 이면 winsorize (train q95, 사전 고정)
    # NRL — coordinate localization term 을 **교체**한다 (kobj/visibility 는 불변)
    nrl_enabled: bool = False
    nrl_beta: float = 0.0
    nrl_lambda: float = 0.0
    # PEVL — cuboid 12 edge 의 방향/상대길이 보존 (auxiliary, standard 는 그대로)
    pevl_enabled: bool = False
    pevl_edges: tuple = ()
    pevl_alpha_len: float = 0.0
    pevl_lambda: float = 0.0
    pevl_resid_q95: float = 0.0      # catastrophic edge 는 auxiliary OFF
    pevl_ramp: tuple = (10, 20)      # 0~9 = 0, 10~19 선형, 20.. 고정

    @classmethod
    def from_env(cls):
        p = os.environ.get("A1_CONFIG")
        if not p or not os.path.exists(p):
            return cls()
        d = json.load(open(p))
        for k in ("p180", "sym_assets", "asym_assets", "role_ramp"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


def per_instance_kpt_loss(keypoint_loss, pred, gt, mask, area):
    """KeypointLoss 와 **같은 수식**을, 마지막 .mean() 만 instance 축으로 남긴다.

    원본:  (factor * (1-exp(-e)) * mask).mean()
    여기:  ... .mean(dim=1)     -> (M,)
    K 가 균일하므로 mean(per_instance) == 원본 scalar (test 로 확인).
    """
    d = (pred[..., 0] - gt[..., 0]).pow(2) + (pred[..., 1] - gt[..., 1]).pow(2)
    factor = mask.shape[1] / (torch.sum(mask != 0, dim=1) + 1e-9)
    e = d / ((2 * keypoint_loss.sigmas).pow(2) * (area + 1e-9) * 2)
    return (factor.view(-1, 1) * ((1 - torch.exp(-e)) * mask)).mean(dim=1)


class A1SymmetryPoseLoss(PSPCPoseLoss26):
    """symmetry-aware positive + (혼재 시) role separation.  PC 는 꺼져 있다."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.a1 = A1Config.from_env()
        self.pspc.enabled = False          # ★ A2 projective term 강제 OFF
        self.pspc.lambda_pc = 0.0
        self._stem_class = {}
        if self.a1.enabled and self.a1.stem_asset_map:
            m = json.load(open(self.a1.stem_asset_map))
            S, A = set(self.a1.sym_assets), set(self.a1.asym_assets)
            for stem, asset in m.items():
                self._stem_class[stem] = 1 if asset in S else (2 if asset in A else 0)
        self._batch = None
        self.a1_stats = {"n_sym": 0, "n_asym": 0, "last_pos": 0.0, "last_role": 0.0}

    def loss(self, preds, batch):
        self._batch = batch                # im_file 을 쓰려고 잠시 보관
        return super().loss(preds, batch)

    def _instance_class(self, masks):
        """선택된 anchor 각각의 대칭 class. 이미지당 객체 1개임을 전수 확인했다."""
        bs = masks.shape[0]
        img = torch.arange(bs, device=masks.device)[:, None].expand_as(masks)[masks]
        files = (self._batch or {}).get("im_file") or []
        out = torch.zeros(img.shape[0], dtype=torch.long, device=masks.device)
        for n, i in enumerate(img.tolist()):
            if i < len(files):
                stem = os.path.splitext(os.path.basename(files[i]))[0]
                out[n] = self._stem_class.get(stem, 0)
        return out

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        if not self.a1.enabled or not masks.any():
            return super().calculate_keypoints_loss(
                masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
                target_bboxes, pred_kpts)
        # ★ sym/asym asset 이 하나도 없으면 fixed-object asset map 을 아예 읽지 않는다.
        #   camera-facing 트랙에서 fixed 자산 의존을 0 으로 만들기 위함 (T14).
        if self._stem_class:
            cls = self._instance_class(masks)
        else:
            cls = torch.zeros(int(masks.sum()), dtype=torch.long, device=masks.device)
        # ★ 부모(PoseLoss26)와 **같은 in-place 연산**을 쓴다.  out-of-place 로 바꾸면
        #   forward 는 같지만 backward 가 달라진다(측정: grad 5.6e-3 vs 잡음 1.7e-4).
        #   A0 parity 가 목적이므로 부모 동작을 그대로 복제한다.
        sel = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        sel[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        target_bboxes /= stride_tensor
        gt = sel[masks]
        pk = pred_kpts[masks]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        m = gt[..., 2] != 0 if gt.shape[-1] == 3 else torch.full_like(gt[..., 0], True)
        is_sym = cls == 1
        is_asym = cls == 2
        # ★ base 는 **부모와 같은 단일 reduction** 을 쓴다.  2단계 reduction
        #   (mean(dim=1).mean()) 으로 바꾸면 forward 는 비트단위 같지만 backward 가
        #   달라진다 (실측: grad 5.6e-3 vs 잡음 1.9e-4).  A0 가 control 이므로
        #   beta=0 에서 A0 와 그래프가 **동일**해야 한다.
        if self.a1.nrl_enabled:
            # ★ REPLACE — 기존 coordinate localization term 을 더하지 않는다
            base = self.a1.nrl_lambda * nrl_coord(pk, gt, m, area, self.a1.nrl_beta)
        else:
            base = self.keypoint_loss(pk, gt, m, area)
        beta = asc_beta(CURRENT_EPOCH["e"], self.a1.asc_full_end,
                        self.a1.asc_ramp_end) if self.a1.asc_enabled else 1.0
        need_sym = beta > 0.0 and bool(is_sym.any())
        need_role = self.a1.lambda_role > 0.0 and bool(is_asym.any())
        d_id = d_180 = soft = None
        if need_sym or need_role:
            # ★ 필요할 때만 계산한다.  beta=0 인데 d_180/soft 를 만들면 쓰이지도 않으면서
            #   GPU workspace 를 바꿔 cuDNN algo 선택이 흔들린다(측정: grad 1.5e-4).
            perm = list(self.a1.p180) + [self.a1.centroid_index]
            d_id = per_instance_kpt_loss(self.keypoint_loss, pk, gt, m, area)
            d_180 = per_instance_kpt_loss(self.keypoint_loss, pk, gt[:, perm, :],
                                          m[:, perm], area)
            if self.a1.mode == "softmin" and self.a1.softmin_tau > 0:
                t = self.a1.softmin_tau
                soft = -t * torch.logsumexp(torch.stack([-d_id / t, -d_180 / t]), 0)
            else:
                soft = torch.minimum(d_id, d_180)
        if not need_sym:
            kpts_loss = base            # 보정항을 만들지 않는다 -> 그래프가 A0 와 동일
            pos = base
        else:
            # 보정항: SYM 인 instance 에서만 d_id -> min(d_id,d_180) 로 이동
            corr = torch.where(is_sym, soft - d_id, torch.zeros_like(d_id)).mean()
            kpts_loss = base + beta * corr
            pos = torch.where(is_sym, (1.0 - beta) * d_id + beta * soft, d_id)
        role = torch.zeros((), device=kpts_loss.device, dtype=kpts_loss.dtype)
        if need_role:
            ROLE_CALLS["n"] += 1
            r = torch.relu(self.a1.margin + d_id - d_180)
            role = torch.where(is_asym, r, torch.zeros_like(r)).sum() / is_asym.sum()
            kpts_loss = kpts_loss + self._ramp() * self.a1.lambda_role * role
        self.a1_stats = {"n_sym": int(is_sym.sum()), "n_asym": int(is_asym.sum()),
                         "n_total": int(pos.numel()), "beta": float(beta),
                         "epoch": int(CURRENT_EPOCH["e"]),
                         "d_id": float(d_id.mean()) if d_id is not None else None,
                         "d_180": float(d_180.mean()) if d_180 is not None else None,
                         "sym_min": float(soft.mean()) if soft is not None else None,
                         "last_pos": float(pos.mean()), "last_role": float(role)}
        if self.a1.pevl_enabled and self.a1.pevl_lambda != 0.0:
            a, b = self.a1.pevl_ramp
            e = CURRENT_EPOCH["e"]
            w = 0.0 if e < a else (1.0 if e >= b else (e - a) / max(b - a, 1))
            if w > 0:
                kpts_loss = kpts_loss + w * self.a1.pevl_lambda * pevl_loss(
                    pk, gt, m, area, [tuple(x) for x in self.a1.pevl_edges],
                    self.a1.pevl_alpha_len, self.a1.pevl_resid_q95)
        if self.a1.takl_enabled and self.a1.takl_lambda != 0.0:
            # ★ ADD — standard 를 제거하지 않는다
            kpts_loss = kpts_loss + self.a1.takl_lambda * takl_tail(
                pk, gt, m, area, self.a1.takl_tau, self.a1.takl_z_clip)
        # obj / rle 는 부모 정의를 그대로 쓴다
        kpts_obj_loss = torch.zeros((), device=kpts_loss.device, dtype=kpts_loss.dtype)
        rle_loss = torch.zeros((), device=kpts_loss.device, dtype=kpts_loss.dtype)
        if pk.shape[-1] in (3, 5):
            kpts_obj_loss = self.bce_pose(pk[..., 2], m.float())
        if self.rle_loss is not None and pk.shape[-1] in (4, 5):
            rle_loss = self.calculate_rle_loss(pk, gt, m).clamp(min=0)
        return kpts_loss, kpts_obj_loss, rle_loss

    def _ramp(self):
        a, b = self.a1.role_ramp
        e = getattr(self, "_epoch", None)
        if e is None:
            return 1.0
        if e < a:
            return 0.0
        if e >= b:
            return 1.0
        return (e - a) / max(b - a, 1)
