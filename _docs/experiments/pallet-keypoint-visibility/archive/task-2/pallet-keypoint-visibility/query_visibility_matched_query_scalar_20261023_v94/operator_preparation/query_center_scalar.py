"""Generated-only V94 candidate: one query-conditioned scalar, three image inputs.

No cue reference, region, semantic ID or target annotation enters this module.
The future source contract must give the binary target its meaning and establish
target/cue association. This code does not establish native8 or human DES truth.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

FORWARD_FIELDS = ('p3_roi', 'local_rgb96', 'local_valid96')


class QueryCenterScalar(nn.Module):
    """V91-size encoders with GT-free, centre-conditioned global pooling.

    The single scalar always belongs to the unchanged input q/crop. Attention
    permits evidence anywhere in that crop; it never supplies a new coordinate
    or changes the output site. Attention is diagnostic, not a GT readout map.
    """

    def __init__(self):
        super().__init__()
        self.p3_encoder = nn.Sequential(nn.Conv2d(65, 24, 3, padding=1), nn.GELU())
        self.local_encoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Conv2d(44, 32, 1), nn.GELU(),
            nn.Conv2d(32, 24, 3, padding=1), nn.GELU())
        self.query = nn.Linear(24, 8)
        self.key = nn.Conv2d(24, 8, 1)
        self.readout = nn.Sequential(nn.Linear(49, 16), nn.GELU(), nn.Linear(16, 1))
        yy, xx = torch.meshgrid(torch.arange(9), torch.arange(9), indexing='ij')
        self.register_buffer('native_offsets', torch.stack((xx - 4, yy - 4)).float()[None] / 9)
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, p3_roi, local_rgb96, local_valid96):
        tensors = (p3_roi, local_rgb96, local_valid96)
        if not all(isinstance(x, torch.Tensor) for x in tensors):
            raise TypeError('Exactly three tensors; metadata/GT dictionaries are forbidden')
        if p3_roi.ndim != 4:
            raise ValueError('Flat query batch N65x9x9 required; no implicit slot identity')
        n = len(p3_roi)
        if (not n or tuple(p3_roi.shape) != (n, 65, 9, 9)
                or tuple(local_rgb96.shape) != (n, 3, 96, 96)
                or tuple(local_valid96.shape) != (n, 1, 96, 96)):
            raise ValueError('Unexpected input shapes')
        if not p3_roi.is_floating_point() or local_rgb96.dtype != torch.uint8:
            raise TypeError('Floating original P3 and unmarked uint8 RGB required')
        if not all(x.device == p3_roi.device for x in tensors):
            raise ValueError('Devices must agree')
        if not torch.isfinite(p3_roi).all():
            raise ValueError('P3 must be finite')
        native = p3_roi[:, 64:65]
        if (not torch.all((native == 0) | (native == 1))
                or not torch.all((local_valid96 == 0) | (local_valid96 == 1))):
            raise ValueError('Only factual binary input support masks are allowed')
        dtype = self.readout[-1].weight.dtype
        native = native.to(dtype)
        p3 = p3_roi.to(dtype)
        valid = local_valid96.to(dtype)
        pf = self.p3_encoder(torch.cat((p3[:, :64] * native, native), 1))
        rf = self.local_encoder(torch.cat((local_rgb96.to(dtype) / 255 * valid, valid), 1))
        rf = F.interpolate(rf, size=(9, 9), mode='bilinear', align_corners=False)
        valid9 = F.adaptive_avg_pool2d(valid, (9, 9))
        fused = self.fusion(torch.cat((
            pf, rf, valid9, native,
            self.native_offsets.to(dtype).expand(n, -1, -1, -1)), 1))

        factual = (valid9[:, 0] > 0) & native[:, 0].bool()
        # RGB96 has an even number of samples. These four cells bracket nominal q;
        # this is a conservative tensor-support rule, not a semantic visibility rule.
        centre_supported = (native[:, 0, 4, 4].bool()
                            & local_valid96[:, 0, 47:49, 47:49].bool().flatten(1).all(1))
        available = centre_supported & factual.flatten(1).any(1)
        centre = fused[:, :, 4, 4]
        query = self.query(centre)
        scores = (self.key(fused) * query[:, :, None, None]).sum(1) / math.sqrt(8)
        scores = scores.masked_fill(~factual, -torch.finfo(dtype).max)
        attention = torch.softmax(scores.flatten(1), 1).reshape(n, 9, 9)
        attention = attention * factual.to(dtype)
        attention = attention / attention.flatten(1).sum(1)[:, None, None].clamp_min(1e-12)
        pooled = (fused * attention[:, None]).flatten(2).sum(2)
        fraction = factual.to(dtype).flatten(1).mean(1, keepdim=True)
        raw = self.readout(torch.cat((centre, pooled, fraction), 1))[:, 0]
        # Unavailable rows remain present with a safe finite scalar plus explicit U.
        scalar = torch.where(available, raw, raw * 0)
        return {'query_logit': scalar, 'query_available': available,
                'attention_diagnostic_only': attention}


def known_binary_loss(output, labels, known):
    """Loss reads the same scalar used at inference; labels are loss-only.

    Available and known rows are averaged. Empty batches yield graph zero. The
    future trainer must report all known/unavailable rows rather than drop them
    from evaluation or treat their safe zero scalar as a binary prediction.
    """
    logits = output['query_logit']
    available = output['query_available']
    if labels.shape != logits.shape or known.shape != logits.shape or known.dtype != torch.bool:
        raise ValueError('One label and explicit known bit per unchanged query')
    if labels.device != logits.device or known.device != logits.device:
        raise ValueError('Loss devices must agree')
    if not torch.all((labels[known] == 0) | (labels[known] == 1)):
        raise ValueError('Known binary labels must be exactly 0 or 1')
    target = torch.where(known, labels, torch.zeros_like(labels)).to(logits.dtype)
    admitted = known & available
    per_row = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    weights = admitted.to(logits.dtype)
    loss = (per_row * weights).sum() / weights.sum().clamp_min(1)
    return {'loss': loss, 'admitted': admitted,
            'known_count': known.sum(), 'available_known_count': admitted.sum(),
            'known_unavailable_count': (known & ~available).sum()}


def decode_binary(output):
    """Authoritative sign decision, independent of rounded sigmoid display."""
    z = output['query_logit']
    available = output['query_available']
    return torch.where(available, (z >= 0).to(torch.int8), torch.full_like(z, -1, dtype=torch.int8))
