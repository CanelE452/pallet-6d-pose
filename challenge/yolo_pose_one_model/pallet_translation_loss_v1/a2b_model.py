"""A2b model/trainer — CONTROL 과 A2b 가 **같은 코드 경로**를 쓴다 (§18).

두 arm 의 유일한 차이는 `A2B_CONFIG` 의 `lambda_geo` 다.  lambda_geo=0 이면
`A2BPoseLoss26` 은 LC 항을 계산조차 하지 않으므로 부모 `PoseLoss26` 과
구성적으로 동일하다.  따라서 `CUSTOM TRAINER 효과` 와 `LC TERM 효과` 가 분리된다.
"""
from __future__ import annotations

import hashlib
import json
import os

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import E2ELoss

from a2b_loss import A2BPoseLoss26


class A2BPoseModel(PoseModel):
    def init_criterion(self):
        if getattr(self, "end2end", False):
            return E2ELoss(self, A2BPoseLoss26)
        return A2BPoseLoss26(self)


def _inner(crit):
    """E2ELoss 는 one2many/one2one 두 벌을 감싼다.  둘 다 돌려준다."""
    out = []
    for name in ("one2many", "one2one"):
        m = getattr(crit, name, None)
        if m is not None:
            out.append((name, m))
    return out or [("single", crit)]


class A2BTrainer(PoseTrainer):
    """batch 순서 해시 · LC 통계 · conditioning 을 남긴다 (§19, §21)."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = A2BPoseModel(cfg, nc=self.data["nc"],
                             data_kpt_shape=self.data["kpt_shape"],
                             ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._batch_hash_rows = []

        def _on_epoch_end(trainer):
            path = os.path.join(self.save_dir, "A2B_EPOCH_STATS.jsonl")
            rows = []
            for name, m in _inner(getattr(trainer.model, "criterion", None)):
                st = dict(getattr(m, "a2b_stats", {}) or {})
                st.update(getattr(m, "lookup_stats", {}) or {})
                cond = getattr(m, "cond_log", []) or []
                if cond:
                    import numpy as np
                    c = np.asarray(cond)
                    st.update({"cond_p50": float(np.percentile(c, 50)),
                               "cond_p90": float(np.percentile(c, 90)),
                               "cond_p99": float(np.percentile(c, 99)),
                               "cond_n": int(c.size)})
                    m.cond_log = []
                st.update({"rank_deficient": getattr(m, "rank_deficient", 0),
                           "ridge_fallback": getattr(m, "ridge_fallback", 0),
                           "criterion": name, "epoch": int(trainer.epoch),
                           "lambda_geo": float(getattr(m, "a2b").lambda_geo)})
                rows.append(st)
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")

        def _on_train_end(trainer):
            # trainer 는 batch 를 self 에 저장하지 않는다 (ultralytics 8.4.60 확인).
            # loss 가 배치마다 stem 을 이미 보고 있으므로 거기서 가져온다.
            rows = []
            for _, m in _inner(getattr(trainer.model, "criterion", None)):
                log = getattr(m, "batch_stem_log", None)
                if log:
                    rows = [{"i": i, "stems": st} for i, st in enumerate(log)]
                    break
            self._batch_hash_rows = rows
            payload = json.dumps(self._batch_hash_rows, sort_keys=True)
            out = {"n_batches_hashed": len(self._batch_hash_rows),
                   "first_100_batch_sha256": hashlib.sha256(payload.encode()).hexdigest()}
            with open(os.path.join(self.save_dir, "A2B_BATCH_ORDER.json"), "w") as f:
                json.dump(out, f, indent=2)

        self.add_callback("on_train_epoch_end", _on_epoch_end)
        self.add_callback("on_train_end", _on_train_end)
