"""Frozen H0/H1 architectures and frame-balanced losses (no GT I/O)."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .action_data import INPUT_DIM, inputs, source_aliases

ARMS = {
    'D0_H0_MLP': ('D0', 'H0', 'MLP', False),
    'D0_H1_MLP': ('D0', 'H1', 'MLP', False),
    'D1_H0_MLP': ('D1', 'H0', 'MLP', False),
    'D1_H1_MLP': ('D1', 'H1', 'MLP', False),
    'D1_H1_ROLE_AFFINE': ('D1', 'H1', 'ROLE_AFFINE', False),
    'D1_H1_PAIR_MLP': ('D1', 'H1', 'MLP', True),
    'D1_H1_PAIR_ROLE_AFFINE': ('D1', 'H1', 'ROLE_AFFINE', True),
}


class RoleAffine(nn.Module):
    """Two187-dimensional affine scores, routed by the same is_N0 bit as MLP.

    2*(187+1)=376 parameters. The routing bit is not a redundant within-role
    regressor. Candidate-vs-candidate context still cancels within their role.
    """
    def __init__(self):
        super().__init__(); self.other = nn.Linear(INPUT_DIM - 1, 1); self.n0 = nn.Linear(INPUT_DIM - 1, 1)

    def forward(self, x):
        return torch.where(x[..., -1:] > .5, self.n0(x[..., :-1]), self.other(x[..., :-1]))


def make_model(architecture):
    if architecture == 'MLP': return nn.Sequential(nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    if architecture == 'ROLE_AFFINE': return RoleAffine()
    if architecture == 'LEGACY_AFFINE': return nn.Linear(INPUT_DIM, 1)
    raise ValueError(architecture)


def parameter_count(architecture): return sum(p.numel() for p in make_model(architecture).parameters())


def prepared(frame, labels, normalizer, objective, pair=False):
    x, entries = inputs(frame, normalizer, objective, pair)
    if objective == 'H0':
        original = {m['candidate_id']: (i, m) for i, m in enumerate(labels['candidate_metrics'])}
        targets = [labels['h0_target_distribution'][original[c['candidate_id']][0]] for c in entries[:-1]] + [labels['h0_target_distribution'][-1]]
        metrics = [original[c['candidate_id']][1] for c in entries[:-1]] + [labels['n0_metric']]
        if not np.isclose(sum(targets), 1): raise ValueError('H0 eligibility/target mismatch')
    else:
        by_id = {m['candidate_id']: m for m in labels['action_metrics']}
        metrics = [by_id[c['candidate_id']] for c in entries]
        targets = [float(m['success_5cm5deg']) for m in metrics]
    return dict(id=frame['id'], x=x, target=np.asarray(targets, np.float32), metrics=metrics, entries=entries)


def collate(items):
    width = max(1, max(len(r['x']) for r in items)); batch = torch.zeros((len(items), width, INPUT_DIM))
    mask = torch.zeros((len(items), width), dtype=torch.bool); target = torch.zeros((len(items), width))
    for i, r in enumerate(items):
        n = len(r['x']); batch[i, :n] = torch.from_numpy(r['x']); mask[i, :n] = True; target[i, :n] = torch.from_numpy(r['target'])
    return batch, mask, target


def loss_value(model, batch, objective):
    x, mask, target = batch; logits = model(x).squeeze(-1)
    if objective == 'H0': return -(target * logits.masked_fill(~mask, -1e9).log_softmax(-1)).sum(-1).mean()
    counts = mask.sum(-1); valid = counts > 0
    per_action = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    per_frame = (per_action * mask).sum(-1) / counts.clamp_min(1)
    if not valid.any(): return logits.sum() * 0.
    return per_frame[valid].mean()


def choose(logits, entries, objective):
    if len(entries) == 0: return None
    scores = np.asarray(logits, float)
    if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite action logits')
    # H1 entries are ordered N0 first, then physical numerical key. Exact ties
    # therefore prefer N0 then ID-independent stable physical order.
    return int(np.argmax(scores))


def infer(frame, model, normalizer, objective, pair=False):
    from copy import deepcopy
    x, entries = inputs(frame, normalizer, objective, pair)
    if len(entries) == 0:
        return dict(actual_pose=dict(available=False), fallback=False, prediction_status='NO_POSE', selected_candidate_id=None,
            action_role='NO_POSE', selected_source_aliases=[], logits=[], actions=[], eligible_candidates=0, archived_candidates=len(frame['candidates']))
    with torch.inference_mode(): logits = model(torch.from_numpy(x)).squeeze(-1); scores = logits.numpy(); probabilities = logits.softmax(0) if objective == 'H0' else logits.sigmoid()
    selected = choose(scores, entries, objective); c = entries[selected]; none = c is None
    pose = frame.get('fallback', dict(available=False)) if none else c['actual_pose']
    aliases = ['N0'] if none else sorted(source_aliases(c)); role = 'N0' if 'N0' in aliases else ('E0' if 'E0' in aliases else 'OTHER')
    catalog = []
    for i, entry in enumerate(entries):
        catalog.append(dict(candidate_id='H0_NONE' if entry is None else entry['candidate_id'], role='H0_NONE_TO_N0' if entry is None else ('N0' if 'N0' in source_aliases(entry) else 'E0' if 'E0' in source_aliases(entry) else 'OTHER'),
            source_aliases=['N0'] if entry is None else sorted(source_aliases(entry)), score=float(scores[i]), sigmoid_or_softmax=float(probabilities[i])))
    return dict(actual_pose=deepcopy(pose), fallback=none, prediction_status='H0_NONE_TO_N0' if none else 'ACTUAL_ACTION_SELECTED',
        selected_candidate_id='N0' if none else c['candidate_id'], action_role=role if pose.get('available') else 'NO_POSE',
        selected_source_aliases=aliases, none_selected=none, selected_score=float(scores[selected]), probability=float(probabilities[selected]), probability_is_calibrated_confidence=False,
        eligible_candidates=len(entries) - int(objective == 'H0'), archived_candidates=len(frame['candidates']), logits=scores.tolist(), actions=catalog)
