"""PSPCPoseModel — init_criterion 만 override 한다.

★ E2ELoss(model, loss_fn) 는 loss_fn 으로 one2many / one2one **두 벌**을 만든다.
   loss_fn 을 PSPCPoseLoss26 으로 넘기면 두 경로 모두에 적용된다.
   한쪽에만 들어가는 실수를 test T10/T11 로 검증한다.
"""
from __future__ import annotations

from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import E2ELoss

from .loss import PSPCPoseLoss26


class PSPCPoseModel(PoseModel):
    def init_criterion(self):
        if getattr(self, "end2end", False):
            return E2ELoss(self, PSPCPoseLoss26)
        return PSPCPoseLoss26(self)


class A1SymmetryPoseModel(PoseModel):
    """A1 — symmetry/role-aware. PC term 없음."""

    def init_criterion(self):
        from .symmetry import A1SymmetryPoseLoss
        if getattr(self, "end2end", False):
            return E2ELoss(self, A1SymmetryPoseLoss)
        return A1SymmetryPoseLoss(self)
