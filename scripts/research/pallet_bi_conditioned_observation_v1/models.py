"""Identical local-quality MLPs, differing solely in the two condition values."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .features import FEATURE_DIMENSION, FEATURE_NAMES, MODEL_INPUT_DIMENSION

SEEDS = (20261018, 20261019, 20261020)
ARMS = ('COMMON', 'BI')


class Scorer(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError('Only the two fixed scorer families are authorized')
        self.arm = arm
        self.layers = nn.Sequential(nn.Linear(MODEL_INPUT_DIMENSION + 2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x, conditions):
        if x.ndim != 2 or x.shape[1] != MODEL_INPUT_DIMENSION or conditions.shape != (len(x), 2):
            raise ValueError('Explicit normalized numeric/validity and two-condition shapes required')
        c = torch.full_like(conditions, .5) if self.arm == 'COMMON' else conditions
        return self.layers(torch.cat((x, c), 1)).squeeze(-1)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def normalizer(frames):
    """Train-only valid observations; missingness never becomes observed zero."""
    if any(row.get('split') != 'train' for row in frames): raise ValueError('Normalizer accepts TRAIN only')
    values = [[] for _ in range(FEATURE_DIMENSION)]
    for frame in frames:
        for row in frame['corner_rows']:
            x, valid = np.asarray(row['x_values'], float), np.asarray(row['x_validity'], float)
            validate_features(x, valid)
            for j in range(FEATURE_DIMENSION): values[j].extend(x[valid[:, j] == 1, j].tolist())
    means = [float(np.mean(v)) if v else 0. for v in values]
    stds = [float(np.std(v)) if v else 1. for v in values]
    stds = [s if s > 0 else 1. for s in stds]
    return dict(schema='OBSERVATION_TRAIN_ONLY_NORMALIZER_V1', feature_names=list(FEATURE_NAMES),
        counts=[len(v) for v in values], mean=means, standard_deviation=stds, ddof=0,
        frame_ids=[f['id'] for f in frames], missing='zero after normalization, separate unscaled validity')


def validate_features(x, valid):
    if x.ndim != 2 or x.shape[1] != FEATURE_DIMENSION or valid.shape != x.shape:
        raise ValueError('Candidate numeric feature shape differs from frozen allowlist')
    if not np.isfinite(x).all() or not np.isfinite(valid).all() or not np.isin(valid, (0., 1.)).all():
        raise ValueError('Features must be finite with binary validity')
    if np.any(x[valid == 0] != 0): raise ValueError('Missing numeric features must be explicit zero')


def normalized(row, norm):
    if norm['feature_names'] != list(FEATURE_NAMES): raise ValueError('Normalizer feature order changed')
    x, valid = np.asarray(row['x_values'], float), np.asarray(row['x_validity'], float)
    validate_features(x, valid)
    mean, std = np.asarray(norm['mean']), np.asarray(norm['standard_deviation'])
    if not np.isfinite(np.r_[mean, std]).all() or np.any(std <= 0): raise ValueError('Invalid normalizer')
    out = np.concatenate(((x - mean) / std * valid, valid), axis=1).astype(np.float32)
    if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite normalized feature')
    return out


def prepare_frame(frame, targets, norm):
    if frame['id'] != targets['id'] or frame['split'] != targets['split']: raise ValueError('Label frame join changed')
    labels = {c['native_corner']: c for c in targets['corner_rows']}
    result = dict(id=frame['id'], split=frame['split'], corners=[])
    for row in frame['corner_rows']:
        target = labels[row['native_corner']]
        if row['candidate_ids'] != target['candidate_ids']: raise ValueError('Label candidate join changed')
        valid = np.asarray(target['valid_target'], bool)
        y = np.array([0. if t is None else float(t) for t in target['target_log1p_error']], dtype=np.float32)
        if len(valid) != len(row['candidate_ids']) or y.shape != valid.shape or not np.isfinite(y).all():
            raise ValueError('Target availability/shape changed')
        if any(v and t is None for v, t in zip(valid, target['target_log1p_error'])):
            raise ValueError('Missing reference is not a zero-error target')
        partial = bool(row['partial_replaces_full_likelihood'])
        if partial and valid.any(): raise ValueError('Fixed partial paths have no neural point supervision')
        if row['state'] not in ('B', 'I'): raise ValueError('Only frozen automatic visible anchors enter scorer')
        x = normalized(row, norm)
        condition = np.tile([1., 0.] if row['state'] == 'B' else [0., 1.], (len(x), 1)).astype(np.float32)
        result['corners'].append(dict(native_corner=row['native_corner'], candidate_ids=list(row['candidate_ids']),
            x=x, c=condition, y=y, valid=valid, partial=partial))
    return result


def collate(items):
    xs, cs, ys, corner_ids, corner_frames = [], [], [], [], []
    usable_frames = 0; corners = 0
    for item in items:
        present = [c for c in item['corners'] if c['valid'].any()]
        if not present: continue
        for corner in present:
            valid = corner['valid']; n = int(valid.sum())
            xs.append(corner['x'][valid]); cs.append(corner['c'][valid]); ys.append(corner['y'][valid])
            corner_ids.extend([corners] * n); corner_frames.append(usable_frames); corners += 1
        usable_frames += 1
    if not usable_frames: return None
    return dict(x=torch.from_numpy(np.concatenate(xs)), c=torch.from_numpy(np.concatenate(cs)),
        y=torch.from_numpy(np.concatenate(ys)), candidate_corner=torch.tensor(corner_ids, dtype=torch.int64),
        corner_frame=torch.tensor(corner_frames, dtype=torch.int64), frame_count=usable_frames,
        corner_count=corners, candidate_count=len(corner_ids))


def loss_value(model, batch):
    if batch is None: return None
    point = F.smooth_l1_loss(model(batch['x'], batch['c']), batch['y'], beta=1., reduction='none')
    cc = batch['candidate_corner']; cf = batch['corner_frame']
    sums = torch.zeros(batch['corner_count'], dtype=point.dtype).index_add(0, cc, point)
    corners = sums / torch.bincount(cc, minlength=len(sums))
    frames = torch.zeros(batch['frame_count'], dtype=point.dtype).index_add(0, cf, corners)
    return (frames / torch.bincount(cf, minlength=len(frames))).mean()


def evaluate_loss(model, items):
    model.eval(); total = 0.; frames = corners = candidates = 0
    with torch.no_grad():
        for i in range(0, len(items), 32):
            batch = collate(items[i:i + 32])
            if batch is None: continue
            total += float(loss_value(model, batch)) * batch['frame_count']
            frames += batch['frame_count']; corners += batch['corner_count']; candidates += batch['candidate_count']
    return dict(loss=total / frames if frames else None, supervised_frames=frames, full_frame_denominator=len(items),
                supervised_corners=corners, supervised_candidates=candidates)


def choose(model, feature_row, norm, condition_overrides=None):
    """One actual candidate; same order is q0-first then fixed numeric order."""
    out = []; model.eval()
    with torch.no_grad():
        for row in feature_row['corner_rows']:
            if row['partial_replaces_full_likelihood']:
                out.append(dict(native_corner=row['native_corner'], selected_candidate_id=row['candidate_ids'][0],
                    scores=None, selection_reason='FIXED_SHARED_PARTIAL_PATH')); continue
            state = (condition_overrides or {}).get(row['native_corner'], row['state'])
            if state not in ('B', 'I'): raise ValueError('State diagnostic can only replace B/I condition inside fixed V')
            x = normalized(row, norm); c = np.tile([1., 0.] if state == 'B' else [0., 1.], (len(x), 1)).astype(np.float32)
            scores = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
            if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite scorer output')
            j = int(np.argmin(scores))
            out.append(dict(native_corner=row['native_corner'], selected_candidate_id=row['candidate_ids'][j],
                scores=scores.tolist(), selection_reason='ARGMIN_LOG_POSITION_ERROR', scorer_condition_state=state))
    return out
