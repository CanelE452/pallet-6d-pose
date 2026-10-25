"""Pure V223 expansion of a strictly loaded C65. No I/O or source initialization.

Both arms contain the same POINT and PART heads; routing is external to forward.
The original encoder/shared64 is evaluated once. Cloning consumes no RNG.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def make_split_target_head_class(spatial_module):
    class SplitPointPartHead(nn.Module):
        def __init__(self, loaded_C65):
            super().__init__()
            require(loaded_C65.spatial_q8 is True, 'Both arms retain actual q8')
            require(sum(p.numel() for p in loaded_C65.parameters()) == 118359, 'Strict original C65 before expansion')
            for name in ('encoder', 'pool', 'shared', 'support_head', 'owner_head',
                         'direct_end_head', 'spatial_residual'):
                setattr(self, name, getattr(loaded_C65, name))
            self.spatial_q8 = True
            self.part_direct_end_head = copy.deepcopy(self.direct_end_head)
            self.part_spatial_residual = copy.deepcopy(self.spatial_residual)
            require(sum(p.numel() for p in self.parameters()) == 121561, 'Identical expanded capacity')
            for point, part in ((self.direct_end_head, self.part_direct_end_head),
                                (self.spatial_residual, self.part_spatial_residual)):
                for a, b in zip(point.parameters(), part.parameters()):
                    require(torch.equal(a, b) and a.data_ptr() != b.data_ptr(), 'Exact independently trainable head clone')

        def forward(self, x, native, other_mask, other_xy):
            require(x.dtype == torch.float32 and x.ndim == 4 and tuple(x.shape[1:]) == (8, 96, 96), 'Unchanged RAW8')
            require(torch.isfinite(x).all(), 'Finite RAW8')
            spatial_module.validate_context(native, other_mask, other_xy, len(x), x.device)
            features = self.shared(self.pool(self.encoder(x)))
            category = F.one_hot(native, num_classes=8).to(features.dtype)
            conditioned = torch.cat((features, category, other_mask, other_xy), dim=1)
            point = self.direct_end_head(features)[:, 0] + self.spatial_residual(conditioned)[:, 0]
            part = self.part_direct_end_head(features)[:, 0] + self.part_spatial_residual(conditioned)[:, 0]
            return dict(support_logits=self.support_head(features), owner_logits=self.owner_head(features),
                        point_logit=point, part_logit=part)

    return SplitPointPartHead
