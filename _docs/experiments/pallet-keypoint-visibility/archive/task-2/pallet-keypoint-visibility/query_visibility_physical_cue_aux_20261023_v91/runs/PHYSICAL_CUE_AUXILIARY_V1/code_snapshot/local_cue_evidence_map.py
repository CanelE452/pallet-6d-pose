"""Conditional physical-cue observation evidence, not pixel membership or DES.

Image-only auxiliary: the three forward tensors carry no target instance ID,
native8 identity, reference, uncertainty, pose, depth or annotations. A native
initializer's target association is a separate condition of later transfer.
"""
import torch
from torch import nn
from torch.nn import functional as F

FORWARD_FIELDS = ('p3_roi', 'local_rgb96', 'local_valid96')


class LocalCueEvidenceMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.p3_encoder = nn.Sequential(nn.Conv2d(65, 24, 3, padding=1), nn.GELU())
        self.local_encoder = nn.Sequential(nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(44, 32, 1), nn.GELU(), nn.Conv2d(32, 24, 3, padding=1), nn.GELU())
        self.evidence = nn.Conv2d(24, 1, 1)
        yy, xx = torch.meshgrid(torch.arange(9), torch.arange(9), indexing='ij')
        self.register_buffer('native_offsets', torch.stack((xx-4, yy-4)).float()[None] / 9)
        nn.init.zeros_(self.evidence.weight)
        nn.init.zeros_(self.evidence.bias)

    def forward(self, p3_roi, local_rgb96, local_valid96):
        if not all(isinstance(x, torch.Tensor) for x in (p3_roi, local_rgb96, local_valid96)):
            raise TypeError('Exactly three Tensor inputs; no metadata/GT dictionaries')
        if p3_roi.ndim != 4:
            raise ValueError('P3 input is a flat arbitrary-grid batch N65x9x9')
        n = p3_roi.shape[0]
        if not n or tuple(p3_roi.shape) != (n, 65, 9, 9) or tuple(local_rgb96.shape) != (n, 3, 96, 96) or tuple(local_valid96.shape) != (n, 1, 96, 96):
            raise ValueError('Invalid local input shapes')
        if not p3_roi.is_floating_point() or local_rgb96.dtype != torch.uint8:
            raise TypeError('Floating P3 and raw uint8 RGB are required')
        if not all(x.device == p3_roi.device for x in (local_rgb96, local_valid96)):
            raise ValueError('Input devices must agree')
        if not torch.isfinite(p3_roi).all():
            raise ValueError('P3 must be finite')
        native = p3_roi[:, 64:65]
        if not torch.all((native == 0) | (native == 1)) or not torch.all((local_valid96 == 0) | (local_valid96 == 1)):
            raise ValueError('Input validity is factual binary, never semantic visibility')
        dtype = self.evidence.weight.dtype
        native = native.to(dtype); p3 = p3_roi.to(dtype); valid = local_valid96.to(dtype)
        p3_features = self.p3_encoder(torch.cat((p3[:, :64] * native, native), 1))
        rgb_features = self.local_encoder(torch.cat((local_rgb96.to(dtype) / 255 * valid, valid), 1))
        rgb_features = F.interpolate(rgb_features, size=(9, 9), mode='bilinear', align_corners=False)
        valid9 = F.adaptive_avg_pool2d(valid, (9, 9))
        fused = self.fusion(torch.cat((p3_features, rgb_features, valid9, native, self.native_offsets.to(dtype).expand(n, -1, -1, -1)), 1))
        factual = (valid9[:, 0] > 0) & native[:, 0].bool()
        return {'physical_external_logit': self.evidence(fused)[:, 0], 'map_valid': factual, 'input_available': factual.flatten(1).any(1)}


def initialize_shared_encoders(model, source_state):
    own = model.state_dict()
    keys = [key for key in own if key.startswith(('p3_encoder.', 'local_encoder.'))]
    for key in keys:
        if key not in source_state or source_state[key].shape != own[key].shape:
            raise ValueError('Missing/incompatible source encoder: ' + key)
        own[key] = source_state[key]
    model.load_state_dict(own)
    nn.init.zeros_(model.evidence.weight); nn.init.zeros_(model.evidence.bias)
    return {'copied': keys, 'excluded_source': [key for key in source_state if key not in keys]}


def region_logit(logits, factual_valid, uncertainty9_normalized):
    """GT-only downstream region operation; never called inside model.forward.

Each known uncertainty region yields ONE averaged evidence logit and BCE unit.
The nine locations are not nine independently labeled material/pixel samples.
Invalid interpolation support cannot become a semantic negative.
    """
    if logits.ndim != 3 or logits.shape[1:] != (9, 9) or factual_valid.shape != logits.shape:
        raise ValueError('Spatial logit/validity shape')
    if uncertainty9_normalized.shape != (len(logits), 9, 2):
        raise ValueError('Nine downstream GT-only region sample locations required')
    grid = uncertainty9_normalized.to(logits.dtype)[:, None]
    valid = factual_valid.to(logits.dtype)[:, None]
    denominator = F.grid_sample(valid, grid, mode='bilinear', padding_mode='border', align_corners=False)[:, 0, 0]
    numerator = F.grid_sample(logits[:, None] * valid, grid, mode='bilinear', padding_mode='border', align_corners=False)[:, 0, 0]
    available = (denominator > 0).all(1)
    pooled = (numerator / denominator.clamp_min(1e-12)).mean(1)
    return pooled, available
