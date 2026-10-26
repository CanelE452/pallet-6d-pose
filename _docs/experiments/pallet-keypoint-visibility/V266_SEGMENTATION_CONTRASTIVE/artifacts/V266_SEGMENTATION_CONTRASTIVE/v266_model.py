"""SYN-only spatial ownership residual over the frozen V264 heatmap readout.

Class order is TARGET_PALLET, FOREIGN_OBJECT, BACKGROUND.  The two new arms
have exactly the same parameters.  Their only difference is the training loss.
This module accepts frozen features and frozen baseline logits; it does not
load images, keypoint references, a backbone, or real annotations.
"""
from pathlib import Path
import hashlib
import io
import json
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('SEGMENTATION', 'SEGMENTATION_CONTRAST')
OWNER_CLASSES = ('TARGET_PALLET', 'FOREIGN_OBJECT', 'BACKGROUND')
INPUT_DIMENSION = 22


def safe(path):
    p = Path(path)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink: ' + str(a))
    return p


def binding(path):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size}


def checked(bound):
    got = binding(bound['path'])
    if got != bound:
        raise ValueError('Binding mismatch: ' + str(bound['path']))
    return safe(bound['path'])


def read_bound(bound):
    return checked(bound).read_bytes()


def load_code(bound, name):
    p = checked(bound)
    module = types.ModuleType(name)
    module.__file__ = str(p)
    sys.modules[name] = module
    exec(compile(p.read_bytes(), str(p), 'exec'), module.__dict__)
    return module


def write_json(path, value):
    p = safe(path)
    if p.exists():
        raise FileExistsError(str(p))
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def state_sha(state):
    h = hashlib.sha256()
    for key in sorted(state):
        x = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def runtime(require_cuda=True):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('The fixed training run requires the existing CUDA runtime.')
    return 'cuda:0' if require_cuda else 'cpu'


class OwnerVisibilityResidual(nn.Module):
    """A dense owner branch and a zero-initialized scalar visibility residual.

    Input tokens: B x 256 x 384, support: B x 16 x 16, auxiliary: B x 12,
    uv: B x 2 in [0,1], base_E_logit: B.  q is the original crop centre.
    Visibility sees [frozen baseline logit, owner probabilities at q / frozen
    heatmap location / global support, unchanged native/location auxiliary12].
    The frozen baseline is always added to the learned residual.
    """
    def __init__(self):
        super().__init__()
        self.token_norm = nn.LayerNorm(384)
        self.token_linear = nn.Linear(384, 32)
        self.spatial = nn.Conv2d(32, 32, 3, padding=1)
        self.owner = nn.Conv2d(32, 3, 1)
        self.residual = nn.Sequential(nn.Linear(INPUT_DIMENSION, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def pool_weights(support, uv):
        valid = support.flatten(1).float()
        yy, xx = torch.meshgrid(torch.arange(16, device=uv.device),
                                torch.arange(16, device=uv.device), indexing='ij')
        grid = torch.stack((xx, yy), -1).reshape(1, 256, 2).float() + .5
        q = torch.full_like(uv, .5)
        weights = []
        for centre in (q, uv):
            d2 = (grid - centre[:, None] * 16).square().sum(-1)
            w = torch.exp(-d2 / (2 * 1.5 ** 2)) * valid
            weights.append(w / w.sum(1, keepdim=True).clamp_min(1e-12))
        weights.append(valid / valid.sum(1, keepdim=True).clamp_min(1))
        return torch.stack(weights, 1)  # B x (q, heatmap, global) x 256

    def forward(self, tokens, support, auxiliary, uv, base_E_logit):
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (256, 384):
            raise ValueError('Expected Bx256x384 frozen tokens.')
        if tuple(support.shape[1:]) != (16, 16) or auxiliary.shape[1] != 12 or uv.shape[1] != 2:
            raise ValueError('Invalid frozen geometry/auxiliary shape.')
        valid = support.flatten(1).to(tokens.dtype)
        hidden = F.relu(self.token_linear(self.token_norm(tokens))) * valid[..., None]
        hidden = hidden.transpose(1, 2).reshape(-1, 32, 16, 16)
        embedding = F.relu(self.spatial(hidden)) * support[:, None].to(hidden.dtype)
        owner_logits = self.owner(embedding)
        owner_probability = owner_logits.softmax(1)
        weights = self.pool_weights(support, uv)
        pooled = torch.einsum('brp,bcp->brc', weights, owner_probability.flatten(2)).flatten(1)
        features = torch.cat((base_E_logit[:, None], pooled, auxiliary), 1)
        residual = self.residual(features)[:, 0]
        return {'E_logit': base_E_logit + residual, 'residual_E_logit': residual,
                'owner_logits': owner_logits, 'owner_probability': owner_probability,
                'dense_probabilities': owner_probability, 'descriptor': features,
                'owner_summary': pooled, 'embedding': embedding}


def segmentation_loss(owner_logits, soft_target, known_mass, support, class_weights=None):
    """Fractional area labels, unknown pixels ignored; equal weight per crop.

    soft_target sums to one at known tokens; known_mass is the fraction with
    known rendered ownership, before multiplication by token support.
    """
    weight = known_mass.float() * support.float()
    target = soft_target.float()
    if class_weights is None:
        class_weights = torch.ones(3, dtype=torch.float32, device=owner_logits.device)
    weighted_target = target * class_weights[None, :, None, None]
    ce = -(weighted_target * F.log_softmax(owner_logits.float(), 1)).sum(1)
    # Preserve CE scale while increasing each rare class's relative influence.
    denom = (weight * weighted_target.sum(1)).flatten(1).sum(1)
    per_row = (ce * weight).flatten(1).sum(1) / denom.clamp_min(1e-12)
    active = denom > 0
    if not bool(active.any()):
        return owner_logits.sum() * 0
    return per_row[active].mean()


def contrast_candidates(soft_target, known_mass, support, fit):
    """Build lists only for FIT rows, using fixed coverage/purity criteria."""
    target = np.asarray(soft_target)
    mass = np.asarray(known_mass)
    if target.ndim != 4 or target.shape[1:] != (3, 16, 16):
        raise ValueError('Expected soft_target Nx3x16x16.')
    result = {}
    for row in np.flatnonzero(fit):
        cls = target[row].argmax(0)
        ok = (target[row].max(0) >= .8) & (mass[row] >= .95) & np.asarray(support[row], dtype=bool)
        result[int(row)] = tuple(np.flatnonzero(ok.flatten() & (cls.flatten() == c)) for c in range(3))
    return result


def sample_contrast_pixels(batch, candidates, rng):
    rows, pixels, classes = [], [], []
    for batch_row, source_row in enumerate(batch):
        for c, pool in enumerate(candidates[int(source_row)]):
            if len(pool):
                chosen = rng.choice(pool, size=min(4, len(pool)), replace=False)
                rows.extend([batch_row] * len(chosen))
                pixels.extend(chosen.tolist())
                classes.extend([c] * len(chosen))
    return tuple(np.asarray(v, dtype=np.int64) for v in (rows, pixels, classes))


def pixel_contrast_loss(embedding, batch_base_ids, sample, temperature=.1):
    """Same-class positives must come from different synthetic base scenes.

    Different-class tokens are negatives.  Same-base same-class pairs are
    excluded entirely, including the anchor itself.  No real or DEV pixels
    enter candidate construction or this loss.
    """
    rows, pixels, classes = sample
    n = len(rows)
    zero = embedding.sum() * 0
    if n < 2:
        return zero, {'sampled_pixels': n, 'active_anchors': 0, 'positive_pairs': 0}
    device = embedding.device
    row_t = torch.as_tensor(rows, device=device)
    pixel_t = torch.as_tensor(pixels, device=device)
    labels = torch.as_tensor(classes, device=device)
    groups = batch_base_ids[row_t]
    vectors = embedding.permute(0, 2, 3, 1).reshape(-1, 256, 32)[row_t, pixel_t]
    vectors = F.normalize(vectors.float(), p=2, dim=1, eps=1e-8)
    same_class = labels[:, None] == labels[None, :]
    positive = same_class & (groups[:, None] != groups[None, :])
    denominator = positive | ~same_class
    n_positive = positive.sum(1)
    active = n_positive > 0
    stats = {'sampled_pixels': n, 'active_anchors': int(active.sum()),
             'positive_pairs': int(positive.sum())}
    if not bool(active.any()):
        return zero, stats
    similarity = (vectors @ vectors.T) / temperature
    log_denominator = torch.logsumexp(similarity.masked_fill(~denominator, -1e9), 1)
    log_probability = similarity - log_denominator[:, None]
    per_anchor = -(log_probability * positive).sum(1) / n_positive.clamp_min(1)
    return per_anchor[active].mean(), stats


def as_batch(tokens, support, packet, indices, device):
    def tensor(array, dtype):
        return torch.as_tensor(np.array(array[indices], copy=True), dtype=dtype, device=device)
    return {'tokens': tensor(tokens, torch.float32), 'support': tensor(support, torch.bool),
            'auxiliary': tensor(packet['auxiliary'], torch.float32),
            'uv': tensor(packet['uv'], torch.float32),
            'base_E_logit': tensor(packet['base_E_logit'], torch.float32)}


def infer(model, tokens, support, packet, indices, device='cpu', batch_size=64):
    """CPU/GPU replay API; no target labels or ground-truth coordinates read."""
    model.eval()
    outputs = {key: [] for key in ('E_logit', 'residual_E_logit', 'owner_probability', 'owner_summary', 'descriptor')}
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            sl = np.asarray(indices[start:start + batch_size])
            result = model(**as_batch(tokens, support, packet, sl, device))
            for key in outputs:
                outputs[key].append(result[key].cpu().numpy())
    result = {key: np.concatenate(parts) for key, parts in outputs.items()}
    available = np.asarray(packet['available'])[indices] & np.asarray(support[indices]).reshape(len(indices), -1).any(1)
    result['pred'] = np.where(available, np.where(result['E_logit'] > 0, 1, 0), -1).astype(np.int8)
    result['available'] = available
    return result


def segmentation_summary(probability, soft_target, known_mass, support):
    """SYN-only 16x16 owner diagnostics, explicitly fractional token labels."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(soft_target, dtype=np.float64)
    mass = np.asarray(known_mass, dtype=np.float64) * np.asarray(support, dtype=bool)
    predicted = probability.argmax(1)
    cm = np.zeros((3, 3), dtype=np.float64)
    for truth in range(3):
        for prediction in range(3):
            cm[truth, prediction] = (target[:, truth] * mass * (predicted == prediction)).sum()
    ce = -(target * np.log(np.clip(probability, 1e-30, 1))).sum(1)
    mass_per_row = mass.reshape(len(mass), -1).sum(1)
    known = mass_per_row > 0
    per_row = (ce * mass).reshape(len(mass), -1).sum(1) / np.maximum(mass_per_row, 1e-30)
    purity = (target.max(1) >= .8) & (np.asarray(known_mass) >= .95) & np.asarray(support, dtype=bool)
    dominant = target.argmax(1)
    pure_cm = [[int((purity & (dominant == a) & (predicted == b)).sum()) for b in range(3)] for a in range(3)]
    per_class = {}
    for c, name in enumerate(OWNER_CLASSES):
        tp, truth, prediction = cm[c, c], cm[c].sum(), cm[:, c].sum()
        per_class[name] = {'precision': float(tp / prediction) if prediction else None,
                           'recall': float(tp / truth) if truth else None,
                           'IoU': float(tp / (truth + prediction - tp)) if truth + prediction - tp else None}
    return {'domain': 'SYN_DEV_ONLY', 'resolution': '16x16 tokens, fractional rendered ownership labels',
            'classes': list(OWNER_CLASSES), 'fractional_CM_true_rows_predicted_columns': cm.tolist(),
            'known_token_mass': float(mass.sum()), 'rows_with_known_mass': int(known.sum()),
            'mean_per_crop_weighted_CE': float(per_row[known].mean()) if known.any() else None,
            'fractional_accuracy': float(np.trace(cm) / cm.sum()) if cm.sum() else None,
            'per_class': per_class, 'pure_token_CM_true_rows_predicted_columns': pure_cm,
            'pure_token_count': int(purity.sum()), 'purity_rule': 'max_soft_target>=.8 and known_mass>=.95 and token_support'}


def summary(y, pred):
    known = y >= 0
    cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1, -1)] for a in (0, 1)]
    result = {'known': int(known.sum()), 'CM_rows_D_E_cols_D_E_U': cm}
    for c, name in enumerate(('D', 'E')):
        tp, fp, total = cm[c][c], cm[1-c][c], sum(cm[c])
        result[name] = {'precision': tp / (tp + fp) if tp + fp else None,
                        'recall': tp / total if total else None,
                        'F1': 2 * tp / (total + tp + fp) if total + tp + fp else None}
    recalls = [result[k]['recall'] for k in ('D', 'E')]
    result['balanced_recall'] = float(np.mean(recalls)) if all(v is not None for v in recalls) else None
    result['U_rate'] = sum(r[2] for r in cm) / max(int(known.sum()), 1)
    return result
