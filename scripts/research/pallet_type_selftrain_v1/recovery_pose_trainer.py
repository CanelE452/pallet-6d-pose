"""Train only independent pose branches; keep detector/backbone and BN stats exact."""
import torch
from scripts.self_training_yolo.v3.true_ignore_trainer import TrueIgnorePoseTrainer


def pose_parameter(name):
    return name.startswith(('model.23.cv4.','model.23.cv4_kpts.','model.23.cv4_sigma.',
                            'model.23.one2one_cv4.','model.23.one2one_cv4_kpts.',
                            'model.23.one2one_cv4_sigma.','model.23.flow_model.'))


class PoseOnlyTrainer(TrueIgnorePoseTrainer):
    def _setup_train(self):
        super()._setup_train()
        for name,param in self.model.named_parameters():param.requires_grad_(pose_parameter(name))
        self.recovery_trainable=[n for n,p in self.model.named_parameters() if p.requires_grad]
        assert self.recovery_trainable
        allowed=set(self.recovery_trainable)
        # Every buffer is frozen, including pose BN. Trainable BN affine parameters stay trainable.
        self.recovery_fixed={n:v.detach().clone() for n,v in self.model.state_dict().items() if n not in allowed}
        self._model_train()

    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

    def check_frozen(self):
        actual=self.model.state_dict()
        assert all(torch.equal(actual[n],v) for n,v in self.recovery_fixed.items()),'Frozen state changed'
        return len(self.recovery_fixed)

    def save_model(self):
        self.check_frozen()
        # EMA multiply/add can round unchanged tensors. Frozen tensors must remain exact in saved inference model.
        ema=self.ema.ema.state_dict()
        for name,value in self.recovery_fixed.items():ema[name].copy_(value)
        return super().save_model()
