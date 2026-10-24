"""Frozen V78 pooled DINO query-support candidate; four input tensors only.

The caller gathers the target frame embedding for each query.  Whole-frame
F x 8 inputs are also accepted; target F x 384 is broadcast over its 8 slots.
No file, backbone, annotation or dataset operations occur in this module.
"""
import torch
from torch import nn
from torch.nn import functional as F

FORWARD_FIELDS = ('local_features', 'target_features', 'context42',
                  'validity_fractions')


class DinoPooledQuerySupportHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(nn.LayerNorm(384, elementwise_affine=False),
                                       nn.Linear(384, 64), nn.GELU())
        self.context = nn.Sequential(nn.Linear(42, 32), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(356, 64), nn.GELU(),
                                    nn.Linear(64, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, local_features, target_features, context42,
                validity_fractions):
        assert local_features.shape[-2:] == (2, 384)
        shape = local_features.shape[:-2]
        assert len(shape) in (1, 2)
        assert context42.shape == (*shape, 42)
        assert validity_fractions.shape == (*shape, 3)
        if len(shape) == 2:
            assert shape[1] == 8 and target_features.shape == (shape[0], 384)
            target_features = target_features[:, None, :].expand(-1, 8, -1)
        else:
            assert target_features.shape == (*shape, 384)
        # Storage FP16 becomes FP32 before all head arithmetic; no autocast.
        local = local_features.float().reshape(-1, 2, 384)
        target = self.projector(target_features.float().reshape(-1, 384))
        local = self.projector(local)
        center, global_local = local[:, 0], local[:, 1]
        geometry = self.context(context42.float().reshape(-1, 42))
        # Cosine uses the projected 64-D vectors, matching the frozen fusion.
        cosine = F.cosine_similarity(center, target, dim=-1, eps=1e-8)[:, None]
        fused = torch.cat((center, global_local, target, center * target,
                           (center - target).abs(), cosine,
                           validity_fractions.float().reshape(-1, 3),
                           geometry), dim=-1)
        assert fused.shape[-1] == 356
        return {'support_logit': self.fusion(fused).reshape(shape)}
