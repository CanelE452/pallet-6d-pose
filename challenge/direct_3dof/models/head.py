"""YOLO26 Pose head 에 direct yaw 회귀를 형제로 붙인다.

최종 산출은 여전히 3DoF ``(x, z, yaw)`` 지만 그 중 **yaw 만** 이 head 가 낸다.
``x, z`` 는 기존 keypoint → PnP 경로를 그대로 쓴다.  이유는 두 가지다.

* 실제로 흔들리는 값은 yaw 다.  x/z 는 PnP 가 이미 안정적으로 낸다.
* 현재 synthetic 은 팔레트 치수가 프레임마다 2.5배 범위로 랜덤이라
  ``실제크기 / z`` 만 관측된다 — z 를 direct 로 배우는 것이 원리적으로 불가능하다.
  yaw 는 크기의 절대값이 아니라 형상 사영에서 나오므로 그 제약을 받지 않는다.

설계 근거 (ultralytics 8.4.60 `nn/modules/head.py` 실측):

* ``Pose26.cv4`` 는 읽기 헤드가 없는 2-Conv trunk (출력 ``c4`` 채널) 이고, 그 위에
  ``cv4_kpts`` (27ch) 와 ``cv4_sigma`` (18ch) 가 1x1 로 병렬로 붙는다.
  yaw 도 **같은 trunk 위의 세 번째 형제** 로 붙이면 backbone/neck/trunk 를 전부
  공유하면서 추가 비용이 1x1 conv 세 개(레벨당 하나)뿐이다.
* ``end2end=True`` 라 ``one2one_*`` 사본이 존재한다.  새 head 도 같은 규칙으로 복제한다.
* head 는 anchor-free dense 다.  8,400 anchor 각각이 자기가 담당하는 object 의 yaw 를
  낸다.  object 대응은 학습 때 TAL 의 ``fg_mask``/``target_gt_idx``, 추론 때
  ``get_topk_index`` 가 맡는다 — detection 과 pose 의 association 이 유지된다.

``Pose26.forward_head`` 를 super() 로 부르지 않고 재구현하는 이유는 하나다.  상위 구현이
trunk 출력(``features``) 을 지역변수로 쓰고 버려서, super() 를 쓰면 trunk 를 두 번
돌리게 된다.  연산량이 두 배가 되는 대신 코드 몇 줄을 아끼는 거래는 하지 않는다.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from ultralytics.nn.modules.head import Detect, Pose, Pose26

# (sin 4ψ, cos 4ψ).  4배각인 이유는 3DOF_CONTRACT.md §3 — 현장 팔레트가 4방향 진입이라
# 90° 회전이 등가다.  (sin ψ, cos ψ) 로 두면 등가인 네 회전이 서로 다른 타깃을 받아
# 학습 신호가 상쇄된다.
N_YAW_OUTPUTS = 2


class PoseDirectYaw26(Pose26):
    """Pose26 + per-anchor direct yaw regression.

    keypoint branch 는 구조적으로 그대로 남는다.  variant 는 loss weight 로 가른다:

    * ``direct_yaw``       — kp weight 0.  yaw 만 배운다.
    * ``direct_yaw_auxkp`` — kp weight > 0.  synthetic keypoint geometry 를
      regularizer 로 함께 쓴다.

    구조를 같게 두어야 두 variant 비교에서 파라미터 수가 교란변수로 끼지 않는다.
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16,
                 end2end=False, ch: tuple = ()):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.n_yaw = N_YAW_OUTPUTS
        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4_yaw = nn.ModuleList(nn.Conv2d(c4, self.n_yaw, 1) for _ in ch)
        if end2end:
            self.one2one_cv4_yaw = copy.deepcopy(self.cv4_yaw)

    # ── head 컴포넌트 묶음 ──────────────────────────────────────────────────
    @property
    def one2many(self) -> dict:
        heads = dict(super().one2many)
        heads["yaw_head"] = self.cv4_yaw
        return heads

    @property
    def one2one(self) -> dict:
        heads = dict(super().one2one)
        heads["yaw_head"] = self.one2one_cv4_yaw
        return heads

    # ── forward ────────────────────────────────────────────────────────────
    def forward_head(self, x, box_head, cls_head, pose_head, kpts_head,
                     kpts_sigma_head, yaw_head=None) -> dict:
        """Pose26.forward_head 와 같되 trunk 출력을 yaw 에도 나눠 쓴다."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is None:
            return preds
        bs = x[0].shape[0]
        features = [pose_head[i](x[i]) for i in range(self.nl)]
        preds["kpts"] = torch.cat(
            [kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        if self.training:
            preds["kpts_sigma"] = torch.cat(
                [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1)
                 for i in range(self.nl)], 2)
        if yaw_head is not None:
            preds["yaw"] = torch.cat(
                [yaw_head[i](features[i]).view(bs, self.n_yaw, -1) for i in range(self.nl)], 2)
        return preds

    def _inference(self, x: dict) -> torch.Tensor:
        """추론 텐서 뒤에 (sin4ψ, cos4ψ) 를 그대로 잇는다.

        keypoint 와 달리 anchor 상대 offset 이 아니라 각도 인코딩이라
        anchor/stride 로 디코드하지 않는다.  각도 복원은 소비자 쪽에서 한다.
        """
        preds = Pose._inference(self, x)
        return torch.cat([preds, x["yaw"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """[x1,y1,x2,y2, conf, cls, kpts…, sin4ψ, cos4ψ] 로 top-k object 를 뽑는다."""
        boxes, scores, kpts, yaw = preds.split(
            [4, self.nc, self.nk, self.n_yaw], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        yaw = yaw.gather(dim=1, index=idx.repeat(1, 1, self.n_yaw))
        return torch.cat([boxes, scores, conf, kpts, yaw], dim=-1)

    def fuse(self) -> None:
        """one2many 제거.  yaw head 도 같은 규칙을 따른다."""
        super().fuse()
        self.cv4_yaw = None
