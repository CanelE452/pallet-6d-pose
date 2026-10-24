"""Pure matched-capacity native-input residual factory; no files or data reads."""
import torch
from torch import nn
from torch.nn import functional as F


def validate_native(native, n, device):
    if not isinstance(native, torch.Tensor) or native.dtype != torch.int64:
        raise ValueError('Stored native must be torch.int64')
    if native.shape != (n,) or native.device != device:
        raise ValueError('Native batch shape/device must match the image features')
    if not bool(((native >= 0) & (native < 8)).all()):
        raise ValueError('Native slot must be exactly0..7')
    return native


def make_native_conditioned_head_class(existing_head_module, candidate_module):
    """Append after the original base initialization; only direct logit is changed.

    Both arms store every one of the116407 parameters. The native condition flag
    is fixed by the protocol and checkpoint metadata, not a learned state tensor.
    The final residual layer is zero, so all original logits initially match.
    """
    original = candidate_module.make_candidate_head_class(existing_head_module)

    class NativeConditionedDirectEndHead(original):
        def __init__(self, *, native_condition):
            if not isinstance(native_condition, bool):
                raise ValueError('Native condition flag must be bool')
            super().__init__()
            self.native_condition = native_condition
            self.native_residual = nn.Sequential(nn.Linear(64 + 8, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.native_residual[2].weight)
            nn.init.zeros_(self.native_residual[2].bias)

        def residual_from_features(self, features, native):
            if features.dtype != torch.float32 or features.ndim != 2 or features.shape[1] != 64 or not bool(torch.isfinite(features).all()):
                raise ValueError('Original shared features must be finiteFP32[N,64]')
            validate_native(native, len(features), features.device)
            code = F.one_hot(native, num_classes=8).to(torch.float32)
            if not self.native_condition:
                code = torch.zeros_like(code)
            return self.native_residual(torch.cat((features, code), dim=1))[:, 0]

        def forward(self, x, native):
            if x.dtype != torch.float32 or x.ndim != 4 or tuple(x.shape[1:]) != (8, 96, 96) or not bool(torch.isfinite(x).all()):
                raise ValueError('Same original finiteFP32[N,8,96,96] input required')
            validate_native(native, len(x), x.device)
            features = self.shared(self.pool(self.encoder(x)))
            return dict(support_logits=self.support_head(features),
                        owner_logits=self.owner_head(features),
                        direct_end_external_logit=self.direct_end_head(features)[:, 0] + self.residual_from_features(features, native))

    return NativeConditionedDirectEndHead
