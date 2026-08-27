"""HF arm 전용 — negative image 에서만 classification 을 hard-focused 로 바꾼다.

architecture 는 vanilla YOLO26n-Pose 그대로다.  바뀌는 것은 loss 한 항뿐이다.

## 어디를 건드리는가

`PoseLoss26` 은 `v8DetectionLoss.get_assigned_targets_and_loss` 에서 cls 를 낸다:

    target_scores_sum = max(target_scores.sum(), 1)
    bce_loss = self.bce(pred_scores, target_scores)      # (bs, A, nc)
    loss[1] = bce_loss.sum() / target_scores_sum

pure negative image 는 `target_scores` 가 전부 0 이라 분모에 기여하지 않으면서
분자에는 anchor 8,400 개의 합을 얹는다.  negative 를 늘릴수록 cls loss scale 이
커지는 구조이고, 이것이 YN 에서 positive confidence 까지 눌린 경로다.

## 무엇으로 바꾸는가

    POSITIVE image  ->  stock 그대로.  단 한 줄도 바꾸지 않는다.
    NEGATIVE image  ->  stock all-anchor 기여를 **제거**하고
                        lambda_neg * mean_over_anchors( p^gamma * BCE(z, 0) ) 로 대체

제거와 대체를 같이 하므로 double-count 가 없다.  image-wise mean 으로 먼저
정규화해 negative 장수가 늘어도 scale 이 폭증하지 않는다(spec 12).

`E2ELoss` 가 one2many/one2one 두 갈래를 각각 `PoseLoss26` 으로 돌리므로 이 클래스는
양쪽에 동일하게 적용된다 — 한쪽만 바꾸면 두 갈래가 다른 것을 학습한다.

lambda_neg 는 `hyp.cls` gain 앞에서 곱한다.  positive 항과 같은 gain 을 받게 해
"positive 경로 무변경" 을 유지하기 위해서다.
"""
from __future__ import annotations

import torch

from ultralytics.utils.loss import PoseLoss26

GAMMA = 2.0          # spec 11 — 첫 screen 에서 고정.  grid search 금지.


class HardFocalPoseLoss26(PoseLoss26):
    """negative image 한정 focal-negative classification."""

    lambda_neg = 1.0     # GRADIENT_CALIBRATION 이 계산한 값으로 학습 전에 덮어쓴다
    gamma = GAMMA

    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device,
                             dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1),
                             batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))   # (bs, A, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        # ---- 여기가 유일한 변경점 -------------------------------------------
        neg_img = (mask_gt.sum(dim=(1, 2)) == 0)          # (bs,) pure negative
        if neg_img.any():
            keep = (~neg_img).view(-1, 1, 1).to(bce_loss.dtype)
            cls_pos = (bce_loss * keep).sum() / target_scores_sum   # stock, positive 만
            # focal 계수 p^gamma 는 detach 하지 않는다 — 표준 focal loss 정의대로
            # gradient 가 계수를 통해서도 흐른다.  spec 11 이 detach 를 요구하지 않았다.
            p = pred_scores.sigmoid()
            focal = (p ** self.gamma) * bce_loss                     # negative 는 target=0
            neg_term = focal[neg_img].mean()                         # image-wise -> batch mean
            # calibration 이 두 항의 gradient 를 따로 재야 한다.  norm 의 차로는
            # 못 잰다 — 거의 같은 두 norm 을 빼면 gradient 가 달라도 0 이 나온다.
            self.last_cls_parts = (cls_pos, neg_term)
            loss[1] = cls_pos + self.lambda_neg * neg_term
        else:
            loss[1] = bce_loss.sum() / target_scores_sum             # 전부 positive = stock
            self.last_cls_parts = (loss[1], None)
        # --------------------------------------------------------------------

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss, loss.detach())


def make_criterion(model, lambda_neg, gamma=GAMMA):
    """`E2ELoss` 를 HardFocalPoseLoss26 으로 구성한다 (one2many/one2one 양쪽).

    ★로컬 클래스를 만들지 않는다.  ultralytics 는 학습 중 `model.criterion` 을
    모델에 붙여 두고 체크포인트를 통째로 pickle 하는데, 로컬 클래스/함수는
    pickle 이 안 된다(실제로 HF arm 이 epoch 1 저장에서 죽었다).
    lambda/gamma 는 **인스턴스 속성**으로 얹는다 — 클래스 속성을 가린다.
    """
    from ultralytics.utils.loss import E2ELoss

    crit = E2ELoss(model, HardFocalPoseLoss26)
    for branch in (crit.one2many, crit.one2one):
        branch.lambda_neg = float(lambda_neg)
        branch.gamma = float(gamma)
    return crit
