"""HC / HM / HF 3-arm 공용 trainer.

두 가지만 stock 과 다르고, 둘 다 METHOD_SPEC 에 방법의 일부로 적혀 있다.

1. **sample-type dependent mosaic** (spec 15)
   YN 에서 negative 의 19.7% 가 mosaic 으로 positive 와 합성돼 "pure negative" 라는
   의미가 깨졌다.  여기서는 base sample 이 negative 면 mosaic 을 건너뛴다.
   positive 의 augmentation 은 기존 recipe 그대로다 — 한쪽만 바꾸면 arm 간
   비교가 아니라 두 실험이 된다.

2. **HF arm 의 focal-negative loss** (spec 11~13)
   `--arm HF` 일 때만 criterion 을 `hn_loss` 로 바꾼다.  HC/HM 은 손대지 않으므로
   stock 임이 구조적으로 보장된다.
"""
from __future__ import annotations

import os
import sys

from ultralytics.data.augment import Mosaic
from ultralytics.models.yolo.pose import PoseTrainer

HN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HN)


class NegativeAwareMosaic(Mosaic):
    """base sample 에 instance 가 없으면 mosaic 을 적용하지 않는다."""

    def __call__(self, labels):
        inst = labels.get("instances", None)
        if inst is not None and len(inst) == 0:
            return labels          # pure negative — 그대로 통과
        return super().__call__(labels)


def _patch_mosaic():
    """`v8_transforms` 가 만드는 Mosaic 을 negative-aware 버전으로 바꾼다."""
    import ultralytics.data.augment as A
    if getattr(A, "_hn_patched", False):
        return
    A.Mosaic = NegativeAwareMosaic
    A._hn_patched = True


def _hn_init_criterion(self):
    """HF arm 의 criterion.  모듈 레벨이라 pickle 가능하다."""
    import hn_loss
    return hn_loss.make_criterion(self, lambda_neg=HNTrainer.lambda_neg)


class HNTrainer(PoseTrainer):
    """arm 에 따라 loss 만 갈아끼운다.  architecture 는 건드리지 않는다."""

    arm = "HC"
    lambda_neg = 0.0

    def build_dataset(self, img_path, mode="train", batch=None):
        _patch_mosaic()
        return super().build_dataset(img_path, mode, batch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if self.arm == "HF":
            # ★인스턴스에 클로저를 붙이지 않는다 — 체크포인트를 pickle 할 때 죽는다.
            #   모듈 레벨 함수를 **클래스 메서드로** 갈아끼운다.  pickle 은 클래스를
            #   이름으로 저장하므로 메서드 교체는 문제가 없다.
            type(model).init_criterion = _hn_init_criterion
            model.criterion = None      # 다음 forward 에서 새로 만들어지게 한다
        return model
