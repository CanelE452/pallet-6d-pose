"""Full-frame frozen CLIP token decoder; no q/native/pose/baseline input."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('PLAIN_CLIP', 'CLIP_SURGERY_ARCH')
FEATURE_KEYS = {'PLAIN_CLIP': 'tokensPLAIN', 'CLIP_SURGERY_ARCH': 'tokensCS'}


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(path):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size}


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        x = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


class FullMaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.linear = nn.Linear(512, 32)
        self.spatial = nn.Conv2d(32, 32, 3, padding=1)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, tokens):
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (1024, 512):
            raise ValueError('Full uncropped frame token input must be Nx1024x512')
        h = F.relu(self.linear(self.norm(tokens)))
        h = h.transpose(1, 2).reshape(-1, 32, 32, 32)
        return self.output(F.relu(self.spatial(h)))


def fullmask_loss(logits, target, positive_weight, negative_weight, eps=1e-6):
    """Soft-label weighted pixel BCE + unweighted per-frame soft Dice loss."""
    target = target.float()
    assert logits.shape == target.shape and tuple(target.shape[1:]) == (1,32,32)
    positive = positive_weight * target
    negative = negative_weight * (1-target)
    numerator = -(positive*F.logsigmoid(logits.float()) + negative*F.logsigmoid(-logits.float()))
    bce = numerator.flatten(1).sum(1) / (positive+negative).flatten(1).sum(1).clamp_min(1e-12)
    prob = logits.float().sigmoid()
    intersection = (prob*target).flatten(1).sum(1)
    denominator = (prob+target).flatten(1).sum(1)
    dice = 1 - (2*intersection+eps)/(denominator+eps)
    return bce.mean()+dice.mean(), {'weighted_pixel_BCE': bce.mean(), 'soft_Dice_loss': dice.mean()}


def infer_mask(model, tokens, indices=None, device='cpu', batch_size=32):
    if indices is None:
        indices = np.arange(len(tokens))
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ix = np.asarray(indices[start:start+batch_size])
            x = torch.as_tensor(np.array(tokens[ix], copy=True), dtype=torch.float32, device=device)
            outputs.append(model(x).cpu().numpy())
    logits = np.concatenate(outputs)
    probability = 1/(1+np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
    return {'mask_logits': logits, 'probability': probability.astype(np.float32)}


def segmentation_summary(probability, target):
    """All metrics at fixed32x32 target-grid resolution, no GT crop."""
    p = np.asarray(probability, np.float64)
    t = np.asarray(target, np.float64)
    assert p.shape == t.shape and p.shape[1:] == (1,32,32)
    hard_pred = p > .5
    result = {'resolution': '32x32 full-frame token grid; not original-resolution pixel IoU',
              'frames': len(t), 'threshold': .5, 'soft_target': 'M4 full-frame512 bilinear ->avgpool16'}
    for name, truth in [('fractional_target', t), ('hard_target_gt_half', (t > .5).astype(np.float64))]:
        tp = (truth*hard_pred).sum((1,2,3)); fp = ((1-truth)*hard_pred).sum((1,2,3))
        fn = (truth*(~hard_pred)).sum((1,2,3)); tn = ((1-truth)*(~hard_pred)).sum((1,2,3))
        union = tp+fp+fn
        dice_denominator = 2*tp+fp+fn
        iou = np.where(union > 0, tp/np.maximum(union, 1e-12), 1.0)
        dice = np.where(dice_denominator > 0, 2*tp/np.maximum(dice_denominator, 1e-12), 1.0)
        total_tp, total_fp, total_fn = float(tp.sum()), float(fp.sum()), float(fn.sum())
        result[name] = {'CM_TP_FP_FN_TN': [total_tp,total_fp,total_fn,float(tn.sum())],
            'micro_IoU': total_tp/max(total_tp+total_fp+total_fn, 1e-12),
            'micro_Dice': 2*total_tp/max(2*total_tp+total_fp+total_fn, 1e-12),
            'mean_frame_IoU': float(iou.mean()), 'mean_frame_Dice': float(dice.mean()),
            'empty_GT_frames': int((truth.sum((1,2,3)) == 0).sum()),
            'empty_union_frames': int((union == 0).sum()),
            'empty_union_per_frame_policy': 'Both empty ->IoU1/Dice1; micro counts unchanged.'}
    prob_intersection = (p*t).sum((1,2,3))
    result['mean_frame_soft_probability_Dice'] = float(((2*prob_intersection+1e-6)/(p.sum((1,2,3))+t.sum((1,2,3))+1e-6)).mean())
    result['pixel_BCE_unweighted'] = float(-(t*np.log(np.clip(p,1e-30,1))+(1-t)*np.log(np.clip(1-p,1e-30,1))).mean())
    return result
