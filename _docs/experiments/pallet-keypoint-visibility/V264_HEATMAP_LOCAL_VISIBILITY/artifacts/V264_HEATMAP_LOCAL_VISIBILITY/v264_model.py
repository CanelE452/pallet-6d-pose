"""Two matched SYN-only D/E readouts over frozen V263 DINO/heatmap features."""
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

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
ARMS = ('GLOBAL_MATCHED', 'HEATMAP_LOCAL')


def safe(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(path):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size}


def checked(path, sha=None, size=None):
    b = binding(path)
    if sha is not None and b['sha256'] != sha:
        raise ValueError('SHA mismatch: ' + str(path))
    if size is not None and b['size_bytes'] != size:
        raise ValueError('Size mismatch: ' + str(path))
    return safe(path)


def read_bound(b):
    return checked(b['path'], b['sha256'], b['size_bytes']).read_bytes()


def load_code(path, name, sha=None):
    p = checked(path, sha)
    code = p.read_bytes()
    m = types.ModuleType(name)
    m.__file__ = str(p)
    sys.modules[name] = m
    exec(compile(code, str(p), 'exec'), m.__dict__)
    return m


def write_json(path, value):
    p = safe(path)
    assert not p.exists(), p
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        x = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def runtime():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'This fixed run uses the existing CUDA runtime.'
    return 'cuda:0'


def frozen_heatmap_model(device):
    s = load_code(ROOT / 'V263_SPATIAL_HEAD_C/spatial_head_v2.py', 'v264_frozen_spatial',
                  'e630763c487e649bd47e47048a35269a61ff7101e59153b403abc9e5ffc9b1f0')
    p = checked(ROOT / 'V263_SPATIAL_HEAD_C/JOINT_selected_v1/FINAL1000.pt',
                'f1ba62b8329df0ab46d3c5380e2e3ccd028be16e1283c04c9bfab6f5e320c6ce')
    cp = torch.load(io.BytesIO(p.read_bytes()), map_location='cpu')
    model = s.SpatialPointHead()
    model.load_state_dict(cp['state_dict'], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pool_features(model, tokens, support, native, device):
    """No labels or reference coordinates accepted. Fixed sigma1.5 patch cells.

    GLOBAL gets [G,G,A]; LOCAL gets [G,L,A], same dimension/classifier budget.
    G/L are frozen DINO tokens pooled globally or around predicted softargmax.
    A is identical native one-hot8 + shift2 + peak1 + normalized entropy1.
    Per-descriptor layer normalization uses no fitted statistics.
    """
    n = len(native)
    result = {k: [] for k in ('global_features', 'local_features', 'auxiliary', 'uv', 'available')}
    before = state_sha(model.state_dict())
    with torch.inference_mode():
        for start in range(0, n, 64):
            sl = slice(start, start + 64)
            x = torch.as_tensor(np.array(tokens[sl], copy=True), device=device).float()
            valid = torch.as_tensor(np.array(support[sl], copy=True), device=device).bool()
            slot = torch.as_tensor(native[sl], device=device).long()
            o = model(x, valid, slot)
            p = o['probability'].flatten(1)
            v = valid.flatten(1).float()
            denom = v.sum(1, keepdim=True).clamp_min(1)
            g = (x * v[..., None]).sum(1) / denom
            yy, xx = torch.meshgrid(torch.arange(16, device=device), torch.arange(16, device=device), indexing='ij')
            grid = torch.stack((xx, yy), -1).reshape(1, 256, 2).float() + .5
            uv = o['uv_subpixel']
            d2 = (grid - uv[:, None] * 16).square().sum(-1)
            w = torch.exp(-d2 / (2 * 1.5 ** 2)) * v
            w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
            local = (x * w[..., None]).sum(1)
            entropy = -(p * p.clamp_min(1e-30).log()).sum(1)
            entropy = entropy / denom[:, 0].clamp_min(2).log()
            aux = torch.cat((F.one_hot(slot, 8).float(), (uv - .5) * 2,
                             p.max(1).values[:, None], entropy[:, None]), 1)
            for k, a in [('global_features', F.layer_norm(g, (384,))),
                         ('local_features', F.layer_norm(local, (384,))),
                         ('auxiliary', aux), ('uv', uv), ('available', v.any(1))]:
                result[k].append(a.cpu().numpy())
    assert before == state_sha(model.state_dict())
    return {k: np.concatenate(v) for k, v in result.items()}


def design(packet, arm):
    assert arm in ARMS
    local = packet['global_features'] if arm == 'GLOBAL_MATCHED' else packet['local_features']
    return np.concatenate((packet['global_features'], local, packet['auxiliary']), 1).astype(np.float32)


class VisibilityReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(780, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def infer(model, x, available, device):
    model.eval()
    with torch.inference_mode():
        logits = np.concatenate([model(torch.as_tensor(x[s:s+256], device=device)).cpu().numpy()
                                 for s in range(0, len(x), 256)])
    pred = np.where(available, np.where(logits > 0, 1, 0), -1).astype(np.int8)
    return logits, pred


def summary(y, logits, pred):
    known = y >= 0
    cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1, -1)] for a in (0, 1)]
    out = {'known': int(known.sum()), 'CM_rows_D_E_cols_D_E_U': cm}
    for a, name in [(0, 'D'), (1, 'E')]:
        tp = cm[a][a]; fp = cm[1-a][a]; n = sum(cm[a])
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / n if n else None
        out[name] = {'precision': precision, 'recall': recall,
                     'F1': 2*tp/(n+tp+fp) if n+tp+fp else None}
    out['balanced_recall'] = np.mean([out[k]['recall'] for k in ('D', 'E')])
    out['U_rate'] = sum(r[2] for r in cm) / max(int(known.sum()), 1)
    return out
