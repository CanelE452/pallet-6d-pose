"""Y1 — pose-quality-aware classification objective for YOLO26 Pose.

architecture / assigner / bbox / DFL / keypoint-coord / RLE / augmentation 전부 불변.
**classification objective 하나만** 바꾼다.

    q_pose_j = mean over valid kpts of exp(-e_j)      e 는 stock KeypointLoss 의 것 그대로
    t_pose   = detach(t_det) * detach(q_pose)
    L_align  = |t_pose - sigmoid(logit)|^gamma * BCEWithLogits(logit, t_pose)
    L_cls_Y1 = L_cls_standard + lambda * L_align      (같은 classification scale)

구현 원칙 — stock 코드를 복사하지 않는다.  `self.keypoint_loss` 와 `self.bce` 를 감싸서
**stock 이 실제로 쓰는 바로 그 텐서**를 훔쳐본다.  upstream 이 바뀌어도 수식이 갈라지지
않고, loss 경로 자체는 한 줄도 건드리지 않는다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import PoseLoss26

CALLS = {"align": 0, "qpose": 0}          # runtime audit 용


class PoseAwareClsLoss26(PoseLoss26):
    """PoseLoss26 + Quality-Focal style pose-quality alignment on the class head."""

    LAMBDA = 0.25
    GAMMA = 2.0

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self._reset()

    def _reset(self):
        self._q_pose = None
        self._pred_scores = None
        self._target_scores = None
        self._fg_mask = None
        self._tss = None

    # -- stock 텐서 가로채기 -------------------------------------------------
    def get_assigned_targets_and_loss(self, preds, batch):
        bce = self.bce

        def spy(pred_scores, target_scores):
            # stock 은 이 조합으로 딱 한 번 호출한다 (loss.py:434)
            self._pred_scores = pred_scores
            self._target_scores = target_scores
            self._tss = target_scores.sum().clamp_min(1.0)
            return bce(pred_scores, target_scores)

        self.bce = spy
        try:
            out = super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.bce = bce
        self._fg_mask = out[0][0]
        return out

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        kl = self.keypoint_loss

        def spy(pred_kpt, gt_kpt, kpt_mask, area):
            with torch.no_grad():
                # stock KeypointLoss.forward 와 동일한 e (loss.py:327-330)
                d = ((pred_kpt[..., 0] - gt_kpt[..., 0]).pow(2)
                     + (pred_kpt[..., 1] - gt_kpt[..., 1]).pow(2))
                e = d / ((2 * kl.sigmas).pow(2) * (area + 1e-9) * 2)
                q = torch.exp(-e) * kpt_mask
                self._q_pose = (q.sum(1) / kpt_mask.sum(1).clamp_min(1.0)).detach()
                CALLS["qpose"] += 1
            return kl(pred_kpt, gt_kpt, kpt_mask, area)

        self.keypoint_loss = spy
        try:
            return super().calculate_keypoints_loss(masks, target_gt_idx, keypoints,
                                                    batch_idx, stride_tensor,
                                                    target_bboxes, pred_kpts)
        finally:
            self.keypoint_loss = kl

    # -- alignment term ------------------------------------------------------
    def _posealign(self):
        if (self._q_pose is None or self._pred_scores is None
                or self._fg_mask is None or not self._fg_mask.any()):
            return None
        logits = self._pred_scores[self._fg_mask]              # (N_fg, nc)
        t_det = self._target_scores[self._fg_mask].detach()    # (N_fg, nc)
        q = self._q_pose.to(logits.dtype).view(-1, 1)          # (N_fg, 1) 이미 detach
        if q.shape[0] != logits.shape[0]:
            # fg 순서/개수가 어긋나면 조용히 틀리는 대신 끈다
            return None
        t_pose = (t_det * q).detach()
        p = logits.sigmoid()
        weight = (t_pose - p).abs().pow(self.GAMMA)
        bce = F.binary_cross_entropy_with_logits(logits, t_pose, reduction="none")
        CALLS["align"] += 1
        return (weight * bce).sum() / self._tss

    def loss(self, preds, batch):
        self._reset()
        total, detached = super().loss(preds, batch)
        align = self._posealign()
        if align is not None:
            batch_size = preds["kpts"].shape[0]
            add = self.hyp.cls * self.LAMBDA * align       # 표준 cls 와 동일 scale
            bump = torch.zeros_like(total)
            bump[3] = add * batch_size                    # stock 이 loss*batch_size 를 반환
            total = total + bump
            detached = detached.clone()
            detached[3] = detached[3] + add.detach()
        return total, detached


class PoseAwareClsLoss26_Lambda0(PoseAwareClsLoss26):
    """parity test 전용 — lambda=0 이면 stock 과 수치가 같아야 한다."""
    LAMBDA = 0.0
