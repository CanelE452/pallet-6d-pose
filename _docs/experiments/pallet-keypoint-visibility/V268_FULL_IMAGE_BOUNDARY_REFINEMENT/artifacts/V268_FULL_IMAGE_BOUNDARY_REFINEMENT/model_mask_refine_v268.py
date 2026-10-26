"""Full-frame 256 RGB/CLIP residual mask refiner; no query/native/pose inputs."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('HIGHRES_REFINE', 'BOUNDARY_REFINE')
SIDE = 256
DELTA_BOUND = 6.
BOUNDARY_RADIUS = 2


def safe(value):
    p = Path(value)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink(), str(a)
    return p


def binding(path):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def state_sha(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        x = tensor.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


class FullFrameRefiner(nn.Module):
    """Inputs are whole-frame RGB[0,1], frozen CS tokens and frozen CS logits.

    RGB preprocessing: original frame -> PIL bicubic512 -> avgpool2 /255;
    storage is float16 and the model uses float32 CLIP normalization.  Output
    is the same fixed coarse32 map upsampled256 plus 6*tanh(raw_delta/6).
    At initialization the output equals the upsampled coarse map exactly.
    """
    def __init__(self):
        super().__init__()
        self.context_norm = nn.LayerNorm(512)
        self.context_linear = nn.Linear(512, 16)
        self.rgb_stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(33, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.output = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, rgb256, tokens, coarse_logits):
        assert rgb256.ndim == 4 and tuple(rgb256.shape[1:]) == (3, SIDE, SIDE)
        assert tokens.ndim == 3 and tuple(tokens.shape[1:]) == (1024, 512)
        assert tuple(coarse_logits.shape) == (len(rgb256), 1, 32, 32)
        rgb = rgb256.float()
        mean = rgb.new_tensor((.48145466, .4578275, .40821073))[None, :, None, None]
        std = rgb.new_tensor((.26862954, .26130258, .27577711))[None, :, None, None]
        high = self.rgb_stem((rgb-mean)/std)
        context = F.relu(self.context_linear(self.context_norm(tokens.detach().float())))
        context = context.transpose(1, 2).reshape(-1, 16, 32, 32)
        context = F.interpolate(context, (SIDE, SIDE), mode='bilinear', align_corners=False)
        coarse = F.interpolate(coarse_logits.detach().float(), (SIDE, SIDE), mode='bilinear', align_corners=False)
        hidden = self.decoder(torch.cat((high, context, coarse), 1))
        raw_delta = self.output(hidden)
        delta = DELTA_BOUND * torch.tanh(raw_delta / DELTA_BOUND)
        return dict(mask_logits=coarse+delta, residual_logits=delta, bounded_delta=delta, raw_delta=raw_delta,
                    coarse_logits256=coarse)


def boundary_band(target, radius=BOUNDARY_RADIUS):
    """GT-derived two-sided band, including holes; target itself is untouched.

    Morphology is only used to assign training-loss weights.  Outside-frame
    labels are zero for band construction.  No predicted mask is dilated.
    """
    assert tuple(target.shape[1:]) == (1, SIDE, SIDE) and radius == BOUNDARY_RADIUS
    hard = (target.float() >= .5).float()
    padded = F.pad(hard, (radius, radius, radius, radius), value=0)
    kernel = 2*radius+1
    outer = F.max_pool2d(padded, kernel, stride=1) > 0
    inner = -F.max_pool2d(-padded, kernel, stride=1) >= 1
    return outer & ~inner


def refine_loss(logits, target, band, positive_weight, negative_weight, boundary_weight, eps=1e-6):
    assert logits.shape == target.shape == band.shape and tuple(target.shape[1:]) == (1, SIDE, SIDE)
    assert boundary_weight in (0., .5)
    y = target.float()
    logp, logn = F.logsigmoid(logits.float()), F.logsigmoid(-logits.float())
    pos, neg = positive_weight*y, negative_weight*(1-y)
    bce = (-(pos*logp+neg*logn)).flatten(1).sum(1)/(pos+neg).flatten(1).sum(1).clamp_min(1e-12)
    p = logits.float().sigmoid()
    dice = 1-(2*(p*y).flatten(1).sum(1)+eps)/((p+y).flatten(1).sum(1)+eps)
    bp, bn = band.float()*y, band.float()*(1-y)
    mp, mn = bp.flatten(1).sum(1), bn.flatten(1).sum(1)
    ep = -(bp*logp).flatten(1).sum(1)/mp.clamp_min(1e-12)
    en = -(bn*logn).flatten(1).sum(1)/mn.clamp_min(1e-12)
    activep, activen = mp > 0, mn > 0
    nclasses = activep.float()+activen.float()
    boundary = (ep*activep+en*activen)/nclasses.clamp_min(1)
    active = nclasses > 0
    boundary_mean = boundary[active].mean() if bool(active.any()) else logits.sum()*0
    parts = dict(weighted_pixel_BCE=bce.mean(), soft_Dice_loss=dice.mean(), boundary_balanced_BCE=boundary_mean)
    return parts['weighted_pixel_BCE']+parts['soft_Dice_loss']+boundary_weight*boundary_mean, parts


def infer_refiner(model, rgb256, tokens, coarse_logits, indices, device='cpu', batch_size=8):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ix = np.asarray(indices[start:start+batch_size])
            arrays = [torch.as_tensor(np.array(a[ix], copy=True), dtype=torch.float32, device=device)
                      for a in (rgb256, tokens, coarse_logits)]
            outputs.append(model(*arrays)['mask_logits'].cpu().numpy())
    return dict(mask_logits=np.concatenate(outputs))


def mask_summary(logits, target):
    """Soft M4 and half-threshold M4 at256; original pixel IoU is separate."""
    z, t = np.asarray(logits, np.float64), np.asarray(target, np.float64)
    assert z.shape == t.shape and tuple(z.shape[1:]) == (1, SIDE, SIDE)
    pred = z >= 0
    result = dict(frames=len(z), resolution='256x256 full-frame target grid, not original pixel resolution', threshold_logit_ge=0)
    for name, truth in [('fractional_target', t), ('hard_target_gt_half', (t >= .5).astype(np.float64))]:
        tp = (truth*pred).sum((1, 2, 3)); fp = ((1-truth)*pred).sum((1, 2, 3))
        fn = (truth*~pred).sum((1, 2, 3)); tn = ((1-truth)*~pred).sum((1, 2, 3))
        den = tp+fp+fn; dd = 2*tp+fp+fn
        result[name] = dict(CM_TP_FP_FN_TN=[float(tp.sum()), float(fp.sum()), float(fn.sum()), float(tn.sum())],
            micro_IoU=float(tp.sum()/max(float(den.sum()), 1e-12)),
            micro_Dice=float(2*tp.sum()/max(float(dd.sum()), 1e-12)),
            mean_frame_IoU=float(np.where(den > 0, tp/np.maximum(den, 1e-12), 1).mean()),
            mean_frame_Dice=float(np.where(dd > 0, 2*tp/np.maximum(dd, 1e-12), 1).mean()),
            empty_union_frames=int((den == 0).sum()))
    return result
