"""PSPC trainer — get_model 만 갈아끼운다."""
from __future__ import annotations

from ultralytics.models.yolo.pose import PoseTrainer

from .model import PSPCPoseModel


class PSPCPoseTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PSPCPoseModel(cfg, nc=self.data["nc"],
                              data_kpt_shape=self.data["kpt_shape"],
                              ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


class A1SymmetryTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        from .model import A1SymmetryPoseModel
        model = A1SymmetryPoseModel(cfg, nc=self.data["nc"],
                                    data_kpt_shape=self.data["kpt_shape"],
                                    ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


class ASCTrainer(A1SymmetryTrainer):
    """ASC — epoch 을 loss 로 전달하고 epoch 별 진단을 남긴다."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import os
        from .symmetry import CURRENT_EPOCH

        log = os.path.join(self.save_dir, "CONVERGENCE.csv")

        def _start(trainer):
            CURRENT_EPOCH["e"] = int(trainer.epoch)

        def _end(trainer):
            crit = getattr(trainer.model, "criterion", None)
            inner = getattr(crit, "one2many", crit)
            st = getattr(inner, "a1_stats", None) or {}
            new_file = not os.path.exists(log)
            with open(log, "a") as f:
                if new_file:
                    f.write("epoch,beta,d_id,d_180,sym_min,asc_pos,n_sym,n_total\n")
                f.write(f"{trainer.epoch},{st.get('beta','')},{st.get('d_id','')},"
                        f"{st.get('d_180','')},{st.get('sym_min','')},"
                        f"{st.get('last_pos','')},{st.get('n_sym','')},"
                        f"{st.get('n_total','')}\n")

        self.add_callback("on_train_epoch_start", _start)
        self.add_callback("on_train_epoch_end", _end)
