"""Inert C8 deployment composition. Weight I/O occurs only in load_selected_c8()."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import importlib.util
import torch

ROOT = Path(__file__).resolve().parent.parent
FEATURE_CODE = ROOT / 'frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'
FEATURE_CODE_SHA = 'cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578'
HEAD_CODE = ROOT / 'train_frozen_rgb_head.py'
HEAD_CODE_SHA = '93436ffc1046f33cb21c95ce42b549c667b753d37d28d86e4b081170a153eb28'
BATCH_SIZE = 32


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def _module(path, expected, name):
    assert sha(path) == expected, str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenRGBOperator(torch.nn.Module):
    """Exact RGB feature operator plus C8 head; P3 values are availability-only."""
    def __init__(self, feature_encoder, head):
        super().__init__()
        self.feature_encoder = feature_encoder.eval().requires_grad_(False)
        self.head = head.eval().requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        self.feature_encoder.eval()
        self.head.eval()
        return self

    @torch.no_grad()
    def forward(self, p3_roi, rgb96, crop_valid96):
        n = len(rgb96)
        if p3_roi.ndim != 4 or tuple(p3_roi.shape) != (n, 65, 9, 9):
            raise ValueError('Expected P3 ROI [N,65,9,9].')
        if rgb96.dtype != torch.uint8 or tuple(rgb96.shape) != (n, 3, 96, 96):
            raise ValueError('Expected uint8 RGB [N,3,96,96].')
        if crop_valid96.dtype != torch.bool or tuple(crop_valid96.shape) != (n, 1, 96, 96):
            raise ValueError('Expected bool validity [N,1,96,96].')
        if not (p3_roi.device == rgb96.device == crop_valid96.device):
            raise ValueError('All three inputs must share a device.')
        available = (p3_roi[:, 64, 4, 4] > 0) & crop_valid96[:, 0, 47:49, 47:49].flatten(1).all(dim=1)
        values = []
        for start in range(0, n, BATCH_SIZE):
            features = self.feature_encoder(rgb96[start:start+BATCH_SIZE],
                                            crop_valid96[start:start+BATCH_SIZE])
            values.append(self.head(features).float())
        z = torch.cat(values) if values else torch.empty(0, dtype=torch.float32, device=rgb96.device)
        if z.shape != (n,) or not torch.isfinite(z).all():
            raise ValueError('C8 head must return finite scalar logits.')
        return dict(query_logit=torch.where(available, z, torch.zeros_like(z)),
                    query_available=available)


def load_selected_c8(head_checkpoint_path, expected_sha256, map_location='cpu'):
    """Explicit selected head load; caller separately verifies freeze/admission/root entry."""
    path = Path(head_checkpoint_path).resolve()
    if len(expected_sha256) != 64 or sha(path) != expected_sha256:
        raise ValueError('Head checkpoint SHA mismatch.')
    feature = _module(FEATURE_CODE, FEATURE_CODE_SHA, 'c8_exact_rgb_feature_code')
    head_module = _module(HEAD_CODE, HEAD_CODE_SHA, 'c8_exact_head_code')
    head = head_module.FrozenRGBHead()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    head.load_state_dict(payload['state_dict'], strict=True)
    assert sum(p.numel() for p in head.parameters()) == 65793
    # The factory loads only public ImageNet f37072fd encoder weights and selected C8 head.
    encoder = feature.TwoScaleFrozenResNet18(feature.load_imagenet_backbone())
    operator = FrozenRGBOperator(encoder, head).to(map_location).eval()
    receipt = dict(completed=True, head_checkpoint=binding(path), selected_update=payload['update'],
                   encoder_weights=binding(feature.PRETRAINED), feature_code=binding(FEATURE_CODE),
                   head_code=binding(HEAD_CODE), operator_code=binding(__file__), batch_size=BATCH_SIZE,
                   forward_fields=['p3_roi', 'rgb96', 'crop_valid96'],
                   P3_values_are_classifier_features=False, C1_prior=False,
                   threshold_logit=0, unavailable_logit=0, source_or_current_forward_performed=False)
    return operator, receipt
