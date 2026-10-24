"""C5: exact frozen C1 plus a query-centred local RGB residual scalar.

Only the original three image tensors enter forward. This module neither moves
q nor supplies native-part semantics; the source label contract remains external.
"""
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
sys.dont_write_bytecode = True

import torch
from torch import nn
from torch.nn import functional as F

P = Path(__file__).resolve().parents[2]
BASE_CODE = P / 'query_visibility_matched_query_scalar_20261023_v94/operator_preparation/query_center_scalar.py'
BASE_CODE_SHA256 = 'd62e1610ca45f24e3cbf90e66dff3a7d954f87713afea4f5ae7bdc404411b0cb'
C1_CHECKPOINT = P / 'query_visibility_matched_query_scalar_20261023_v94/runs/C1_NATURAL_KNOWNSET_SOURCE_V1/BEST_QUERY_CENTER_SCALAR.pt'
C1_CHECKPOINT_SHA256 = '1a21d8bda252c203a3f54f891e452a6d4bedac6489f495c86b0d382de2c0fd79'
FORWARD_FIELDS = ('p3_roi', 'local_rgb96', 'local_valid96')
RESIDUAL_CROP = (32, 64)
QUERY_FEATURE_INDEX = 15.5 / 8
QUERY_GRID_ALIGN_TRUE = 7 / 24


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_base_code():
    if file_sha256(BASE_CODE) != BASE_CODE_SHA256:
        raise RuntimeError('Immutable V94 C1 model code changed')
    spec = importlib.util.spec_from_file_location('v101_exact_v94_base', BASE_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_code()
known_binary_loss = _BASE.known_binary_loss
decode_binary = _BASE.decode_binary


def state_sha256(state):
    """Value digest includes named parameters AND buffers, dtype and shape."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(str(value.dtype).encode('ascii') + b'\0')
        digest.update(str(tuple(value.shape)).encode('ascii') + b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sample_query_feature(feature4):
    """Nominal q: local32 index15.5 -> strided-feature index1.9375.

    Three k3/s2/p1 convolutions have feature centre index0 at input index0.
    For a4-cell map align_corners=True requires grid2*1.9375/3-1=7/24.
    This is the fixed input query phase, never a learned/GT coordinate.
    """
    if feature4.ndim != 4 or tuple(feature4.shape[1:]) != (16, 4, 4):
        raise ValueError('Expected N16x4x4 central RGB features')
    grid = feature4.new_full((len(feature4), 1, 1, 2), QUERY_GRID_ALIGN_TRUE)
    return F.grid_sample(feature4, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=True)[:, :, 0, 0]


class LocalRGBResidualC5(nn.Module):
    """29794 total parameters: frozen26573 base + trainable3221 residual."""

    def __init__(self, c1_state_dict):
        super().__init__()
        self.base = _BASE.QueryCenterScalar()
        # Strict loading, before copying encoder. Source C1 is never reinitialized.
        self.base.load_state_dict(c1_state_dict, strict=True)
        if not all(torch.isfinite(x).all() for x in self.base.state_dict().values()):
            raise ValueError('Base parameters and buffers must be finite')
        self.base.requires_grad_(False)
        self.base.eval()
        self.residual_encoder = copy.deepcopy(self.base.local_encoder)
        # deepcopy would also copy the frozen requires_grad=False flags.
        self.residual_encoder.requires_grad_(True)
        self.residual_readout = nn.Sequential(nn.Linear(17, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.residual_readout[-1].weight)
        nn.init.zeros_(self.residual_readout[-1].bias)
        self.loaded_base_state_sha256 = state_sha256(self.base.state_dict())
        self.train(True)
        if sum(p.numel() for p in self.base.parameters()) != 26573:
            raise RuntimeError('Base architecture mismatch')
        if sum(p.numel() for p in self.parameters() if p.requires_grad) != 3221:
            raise RuntimeError('Residual architecture mismatch')

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Remains eval even during an outer model.train().
        return self

    def assert_base_unchanged(self):
        current = state_sha256(self.base.state_dict())
        if current != self.loaded_base_state_sha256:
            raise RuntimeError('Frozen C1 state changed')
        return current

    def _residual_raw(self, local_rgb96, local_valid96):
        dtype = self.residual_readout[-1].weight.dtype
        valid = local_valid96[:, :, 32:64, 32:64].to(dtype)
        rgb = local_rgb96[:, :, 32:64, 32:64].to(dtype) / 255.
        features = self.residual_encoder(torch.cat((rgb * valid, valid), 1))
        query_features = sample_query_feature(features)
        support_fraction = valid.mean(dim=(2, 3))  # one factual central32 support fraction
        return self.residual_readout(torch.cat((query_features, support_fraction), 1))[:, 0]

    def forward(self, p3_roi, local_rgb96, local_valid96):
        # C1 performs the exact original input shape/type/finiteness/support checks.
        # Base values are never differentiated and no source/GT/identity enters it.
        with torch.no_grad():
            base_output = self.base(p3_roi, local_rgb96, local_valid96)
        available = base_output['query_available']
        base_logit = base_output['query_logit']
        raw_delta = self._residual_raw(local_rgb96, local_valid96)
        delta = torch.where(available, raw_delta, raw_delta * 0)
        combined = base_logit + delta
        # Exactly one predictive scalar. Components are loss/diagnostic only.
        return {'query_logit': combined, 'query_available': available,
                'base_logit': base_logit, 'delta_logit': delta}


def load_frozen_c1(checkpoint_path=C1_CHECKPOINT, expected_sha256=C1_CHECKPOINT_SHA256,
                   map_location='cpu'):
    """Explicit strict C1 load; callers bind the default official source checkpoint.

    Generated tests pass a separate invented C1-format file and its exact hash.
    No function loads source data, input caches, labels or current artifacts.
    """
    checkpoint_path = Path(checkpoint_path)
    actual_sha = file_sha256(checkpoint_path)
    if actual_sha != expected_sha256:
        raise RuntimeError('C1 checkpoint SHA mismatch')
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise ValueError('C1 payload requires state_dict')
    model = LocalRGBResidualC5(payload['state_dict'])
    receipt = {'checkpoint': {'path': str(checkpoint_path.resolve()), 'sha256': actual_sha},
               'base_code': {'path': str(BASE_CODE), 'sha256': BASE_CODE_SHA256},
               'loaded_base_state_sha256': model.loaded_base_state_sha256,
               'checkpoint_update': payload.get('update'),
               'parameter_count': 29794, 'trainable_parameter_count': 3221,
               'base_frozen': True, 'base_eval': not model.base.training}
    return model, receipt
