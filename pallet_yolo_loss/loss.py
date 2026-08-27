"""PSPC loss — 이번 범위는 A2 (Projective Cuboid Consistency) 뿐이다.

★ 설계상 중요한 두 가지

1. 기준선은 **GT projected centroid** 다.  pred centroid 를 쓰면 centroid 만
   움직여 loss 를 낮추는 shortcut 이 생긴다.  GT 를 쓰면 gradient 가 corner
   endpoint 로만 들어간다.  pred_kpts[..., 8, :] 는 이 항에서 **참조조차 하지
   않으므로** centroid 의 PC gradient 는 구조적으로 정확히 0 이다.

2. base term 은 super() 를 그대로 호출해 얻는다.  복제하지 않으므로
   lambda_pc=0 일 때 parity 가 **구성적으로** 보장된다.

주의: 부모 `calculate_keypoints_loss` 는 `target_bboxes /= stride_tensor` 로
호출자 텐서를 **in-place 수정**한다.  그래서 PC 를 먼저 계산하고 super() 를
나중에 부른다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import torch

from ultralytics.utils.loss import PoseLoss26
from ultralytics.utils.ops import xyxy2xywh

EPS = 1e-9


@dataclass
class PSPCConfig:
    enabled: bool = False
    lambda_pc: float = 0.0
    diagonal_pairs: tuple = ((0, 6), (1, 7), (2, 4), (3, 5))
    centroid_index: int = 8
    degenerate_eps: float = 1e-3      # stride 단위. 이보다 짧은 대각선은 제외
    log_path: str | None = None

    @classmethod
    def from_env(cls):
        p = os.environ.get("PSPC_CONFIG")
        if not p or not os.path.exists(p):
            return cls()
        d = json.load(open(p))
        d["diagonal_pairs"] = tuple(tuple(x) for x in d.get("diagonal_pairs",
                                                            cls.diagonal_pairs))
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class PSPCPoseLoss26(PoseLoss26):
    """PoseLoss26 + projective cuboid consistency.  lambda_pc=0 이면 부모와 동일."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.pspc = PSPCConfig.from_env()
        self.pc_stats = {"n_valid": 0, "n_degenerate": 0, "last_pc": 0.0}

    # ---------------------------------------------------------------- PC ----
    def projective_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                        stride_tensor, target_bboxes, pred_kpts):
        """mean_(i,j) dist(GT centroid, line(pred_i, pred_j)) / sqrt(area).

        target_bboxes 를 **수정하지 않는다** (부모가 나중에 in-place 로 나눈다).
        """
        cfg = self.pspc
        zero = torch.zeros(1, device=pred_kpts.device, dtype=pred_kpts.dtype).squeeze()
        if not masks.any():
            return zero
        sel = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        sel = sel.clone()
        sel[..., :2] = sel[..., :2] / stride_tensor.view(1, -1, 1, 1)
        gt = sel[masks]                              # (M, K, 2 or 3)
        pk = pred_kpts[masks]                        # (M, K, >=2)
        # 부모와 같은 convention: stride 단위 bbox 의 w*h.
        # 나눗셈을 먼저 하고 mask 를 적용한다 (in-place 아님 — 부모가 나중에 나눈다).
        tb = target_bboxes / stride_tensor
        area = xyxy2xywh(tb[masks])[:, 2:].prod(1)
        scale = area.clamp_min(EPS).sqrt()            # (M,)
        has_v = gt.shape[-1] == 3
        ci = cfg.centroid_index
        c = gt[:, ci, :2]                             # GT centroid, gradient 없음
        c_ok = (gt[:, ci, 2] != 0) if has_v else torch.ones_like(c[:, 0], dtype=torch.bool)
        total = torch.zeros_like(scale)
        cnt = torch.zeros_like(scale)
        ndeg = 0
        for i, j in cfg.diagonal_pairs:
            a = pk[:, i, :2]
            b = pk[:, j, :2]
            ab = b - a
            L = ab.norm(dim=1)
            cross = (ab[:, 0] * (c[:, 1] - a[:, 1]) - ab[:, 1] * (c[:, 0] - a[:, 0])).abs()
            d = cross / (L + EPS)
            ok = c_ok.clone()
            if has_v:
                ok = ok & (gt[:, i, 2] != 0) & (gt[:, j, 2] != 0)
            deg = L <= cfg.degenerate_eps
            ndeg += int((ok & deg).sum().item())
            ok = ok & (~deg)
            total = total + torch.where(ok, d / (scale + EPS), torch.zeros_like(d))
            cnt = cnt + ok.to(cnt.dtype)
        per = total / cnt.clamp_min(1.0)
        used = cnt > 0
        self.pc_stats = {"n_valid": int(used.sum().item()), "n_degenerate": ndeg,
                         "last_pc": float(per[used].mean().item()) if used.any() else 0.0}
        return per[used].mean() if used.any() else zero

    # ------------------------------------------------------------ override --
    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        pc = None
        if self.pspc.enabled and self.pspc.lambda_pc != 0.0:
            # 부모가 target_bboxes 를 in-place 로 나누기 **전에** 계산한다
            pc = self.projective_loss(masks, target_gt_idx, keypoints, batch_idx,
                                      stride_tensor, target_bboxes, pred_kpts)
        kpts_loss, kpts_obj_loss, rle_loss = super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
            target_bboxes, pred_kpts)
        if pc is not None:
            kpts_loss = kpts_loss + self.pspc.lambda_pc * pc
        return kpts_loss, kpts_obj_loss, rle_loss
