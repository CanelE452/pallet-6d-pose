"""V227 고정 SOURCE 대조: fixed-Q read 대 learned query-conditioned read."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def unit_rows(x):
    """통계 학습 없는 FP32 마지막 축 unit-L2. 영벡터는 영벡터로 유지."""
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-12)


class ContextReadProbe(nn.Module):
    def __init__(self, use_attention, seed=17):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.query = nn.Linear(32, 8)
        self.key = nn.Linear(388, 8)
        self.classifier = nn.Linear(1184, 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for module in (self.query, self.key):
                module.weight.normal_(0.0, 0.02, generator=generator)
                module.bias.zero_()
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()

    def attention_read(self, tokens, condition32, normalized_active_q):
        tokens = unit_rows(tokens)
        axis = (torch.arange(16, device=tokens.device, dtype=torch.float32) + 0.5) / 16
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack((xx, yy), dim=-1).reshape(1, 256, 2)
        relative = xy - normalized_active_q.float()[:, None, :]
        key_input = torch.cat((tokens, relative, relative.square()), dim=-1)
        query = self.query(condition32.float())
        keys = self.key(key_input)
        attention = torch.softmax((keys * query[:, None, :]).sum(-1) / math.sqrt(8.0), dim=1)
        descriptor = unit_rows((attention[:, :, None] * tokens).sum(dim=1))
        return descriptor, attention

    def forward(self, localCLS, fixedQ, contextCLS, tokens, condition32, normalized_active_q):
        if self.use_attention:
            descriptor, _ = self.attention_read(tokens, condition32, normalized_active_q)
        else:
            descriptor = unit_rows(fixedQ)
        x = torch.cat((unit_rows(localCLS), descriptor, unit_rows(contextCLS), condition32.float()), dim=1)
        return self.classifier(x).squeeze(1)

    def active_l2(self):
        parameters = list(self.classifier.parameters())
        if self.use_attention:
            parameters += list(self.query.parameters()) + list(self.key.parameters())
        return sum(parameter.square().sum() for parameter in parameters)


def balanced_bce(logits, targets):
    targets = targets.float()
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert (targets == 0).any() and (targets == 1).any(), "D/E 두 class가 필요합니다"
    return 0.5 * (losses[targets == 0].mean() + losses[targets == 1].mean())
