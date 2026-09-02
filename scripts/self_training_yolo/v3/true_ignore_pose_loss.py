"""V3 — pseudo keypoint 를 loss level 에서 **진짜로** 무시한다.

V2 의 발견: Ultralytics 8.4.60 에서 `visibility = 0` 은 pure ignore 가 아니다.
location 과 RLE 는 사라지지만 keypoint objectness BCE 가 살아 있어 "이 keypoint 는
보이지 않는다" 는 **negative supervision** 을 준다.  그 결과 kp_conf 의 꼬리가
눌렸다 (p05 0.976 -> 0.75~0.83, `<0.5` 비율 0.1% -> 2~3%).

V3 가 원하는 것은

    UNRELIABLE KEYPOINT -> NO GRADIENT

이지 `NOT VISIBLE TARGET` 이 아니다.

## 어떻게 신호를 나르나

부가 텐서를 batch 에 넣는 방법은 **쓰지 않았다**.  mosaic(0.15)이 4 장을 섞고
RandomPerspective 가 instance 를 재배열·제거하므로, keypoint 와 따로 실리는
per-instance 텐서는 동기화가 깨진다.  ultralytics 의 augmentation 은 auxiliary
텐서를 `Instances` 와 함께 변환해 주지 않는다.

대신 **visibility 채널 자체**를 쓴다.  그건 keypoints 텐서의 일부라 모든
augmentation 을 정확히 따라간다.

    2  supervised   location + RLE + keypoint objectness
    1  TRUE IGNORE  아무 gradient 도 주지 않는다        <- V3 sentinel
    0  invisible    stock 의미 (location 없음, objectness 는 target 0 으로 학습)

`1` 은 이 저장소의 라벨에서 **한 번도 쓰이지 않는다** — synthetic replay 라벨과
V2 pseudo 라벨 모두 0 과 2 만 쓴다(실측).  그래서 sentinel 로 안전하고, synthetic
배치에서는 `1` 이 아예 없으므로 stock 과 **정확히 동일한 손실**이 나온다.

augmentation 은 화면 밖으로 나간 점만 `visible[out_mask] = 0` 으로 바꾼다
(`data/augment.py`).  즉 sentinel 이 0 으로 강등될 수 있는데, 그건 "이 점은 이제
화면 밖이라 정말 안 보인다" 는 뜻이므로 의미상 맞다.  빈도는 계약 검증에서 잰다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.utils.loss import PoseLoss26
from ultralytics.utils.ops import xyxy2xywh

SUPERVISED = 2.0
TRUE_IGNORE = 1.0
INVISIBLE = 0.0


class TrueIgnorePoseLoss26(PoseLoss26):
    """`visibility == 1` 을 어느 keypoint 항에도 넣지 않는다.

    box / cls / dfl 은 건드리지 않는다.  그쪽은 pseudo keypoint mask 와 무관하게
    stock 그대로다.
    """

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ):
        selected_keypoints = self._select_target_keypoints(
            keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes = target_bboxes / stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]

            if gt_kpt.shape[-1] == 3:
                visibility = gt_kpt[..., 2]
                supervise = visibility == SUPERVISED
                ignore = visibility == TRUE_IGNORE
            else:
                supervise = torch.full_like(gt_kpt[..., 0], True, dtype=torch.bool)
                ignore = torch.zeros_like(supervise)

            # location — supervised 인 점만.  stock 은 `!= 0` 인데 라벨이 0/2 만
            # 쓰므로 synthetic 에서는 동일하고, pseudo 의 sentinel 1 만 빠진다.
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, supervise, area)

            if self.rle_loss is not None and pred_kpt.shape[-1] in (4, 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, supervise)
                rle_loss = rle_loss.clamp(min=0)

            if pred_kpt.shape[-1] in (3, 5):
                keep = ~ignore
                if keep.any():
                    # sum_i m_i * BCE_i / sum_i m_i  (§7).  ignore 인 점은 분자에도
                    # 분모에도 없다 — 그래서 gradient 가 정확히 0 이다.
                    elementwise = nn.functional.binary_cross_entropy_with_logits(
                        pred_kpt[..., 2], supervise.to(pred_kpt.dtype),
                        reduction="none")
                    kpts_obj_loss = (elementwise * keep).sum() / keep.sum()
                else:
                    kpts_obj_loss = pred_kpt.sum() * 0.0

        return kpts_loss, kpts_obj_loss, rle_loss


def make_criterion(model):
    """`PoseModel.init_criterion` 을 대신한다 — E2E wrapper 는 그대로 둔다."""

    from ultralytics.utils.loss import E2ELoss, v8PoseLoss

    if getattr(model, "end2end", False):
        return E2ELoss(model, TrueIgnorePoseLoss26)
    return v8PoseLoss(model)
