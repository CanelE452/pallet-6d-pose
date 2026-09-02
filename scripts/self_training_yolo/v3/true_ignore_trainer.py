"""V3 trainer — 손실만 바꾼다.  site-packages 의 ultralytics 는 수정하지 않는다.

architecture · forward · optimizer · schedule · augmentation 은 전부 stock 이다.
`PoseModel.init_criterion` 만 `TrueIgnorePoseLoss26` 을 돌려주게 바꾼다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import PoseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from true_ignore_pose_loss import make_criterion  # noqa: E402


class TrueIgnorePoseModel(PoseModel):
    """stock PoseModel.  criterion 만 다르다."""

    def init_criterion(self):
        return make_criterion(self)


class TrueIgnorePoseTrainer(PoseTrainer):
    """stock PoseTrainer.  모델 클래스만 갈아끼운다."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TrueIgnorePoseModel(
            cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and self.args.rank == -1 if hasattr(self.args, "rank")
            else verbose,
        )
        if weights:
            model.load(weights)
        return model
