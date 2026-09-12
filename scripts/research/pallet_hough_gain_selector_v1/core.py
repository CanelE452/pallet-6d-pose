"""Frozen-action gain selection. Observation-only inference; separate GT targets."""
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import OBSERVATION_KEYS


def assigned_target(p, q, valid, y, yvalid, permutations, diagonal):
    """Select ONE whole-object assignment using P only; use it for both actions."""
    p, q, y = (np.asarray(a, dtype=np.float64) for a in (p, q, y))
    valid, yvalid = np.asarray(valid, bool), np.asarray(yvalid, bool)
    candidates = []
    for index, perm in enumerate(permutations):
        gt, mask = y[np.asarray(perm)][:8], yvalid[np.asarray(perm)][:8]
        errors = []
        coverage = []
        for action in (p, q):
            usable = mask & valid[:8] & np.isfinite(action[:8]).all(-1)
            e = np.full(8, float(diagonal))
            e[usable] = np.linalg.norm(action[:8][usable] - gt[usable], axis=-1)
            errors.append(e); coverage.append(int(usable.sum()))
        mean = lambda e: float(e[mask].mean()) if mask.any() else float(diagonal)
        candidates.append(dict(choice=index, p_error=errors[0], q_error=errors[1], mask=mask,
                               p_mean=mean(errors[0]), q_mean=mean(errors[1]), coverage=coverage))
    result = min(candidates, key=lambda x: x['p_mean'])
    result['gain_px'] = result['p_mean'] - result['q_mean']
    result['gain_normalized'] = result['gain_px'] / diagonal
    return result


def choose(p, q, predicted_gain_normalized, diagonal, tau_px, force_point=False):
    """Exact whole-layout P-or-Q action, never blending; no target arguments."""
    if not torch.equal(p[:, 8], q[:, 8]):
        raise ValueError('center8 contract violated upstream')
    pick = torch.isfinite(predicted_gain_normalized) & (predicted_gain_normalized * diagonal > tau_px)
    if force_point:
        pick = torch.zeros_like(pick)
    return torch.where(pick[:, None, None], q, p), pick


@torch.no_grad()
def action_features(observation, proposal):
    """Allowlist is enforced: no GT, ID, source, session, or symmetry metadata."""
    if set(observation) != set(OBSERVATION_KEYS):
        raise ValueError('inference accepts observation allowlist only')
    required = {'points', 'raw_line', 'utility', 'mode_mass', 'ambiguity'}
    if set(proposal) != required:
        raise ValueError('proposal allowlist mismatch')
    p, q = observation['base_points'][:, :8], proposal['points'][:, :8]
    box = observation['box']; origin = box[:, None, :2]
    size = (box[:, 2:] - box[:, :2]).clamp_min(1)[:, None]
    d = observation['image_hw'].norm(dim=-1)[:, None, None]
    delta = q-p; mag = delta.norm(dim=-1, keepdim=True)
    groups = []
    def add(name, tensor):
        groups.append((name, tensor.flatten(1)))
    add('P_roi', (p-origin)/size); add('Q_roi', (q-origin)/size)
    add('P_diagonal', p/d); add('Q_diagonal', q/d)
    add('point_valid', observation['point_valid'][:, :8].float())
    add('point_sigma_diagonal', observation['point_sigma'][:, :8]/d[:, :, 0])
    add('delta_diagonal', delta/d); add('delta_magnitude', mag/d)
    add('delta_direction', delta/mag.clamp_min(1e-8))
    lines = proposal['raw_line']; n, c = lines[..., :2], lines[..., 2]
    add('line_normal', n); add('line_offset_diagonal', c/d[:, :, 0])
    add('line_angle_over_pi', torch.atan2(n[..., 1], n[..., 0])/torch.pi)
    for key in ('utility', 'mode_mass', 'ambiguity'):
        add(key, proposal[key])
    relations = []
    for e, (a, b) in enumerate(EDGES):
        before = (p[:, [a,b]]*n[:, e, None]).sum(-1)+c[:, e, None]
        after = (q[:, [a,b]]*n[:, e, None]).sum(-1)+c[:, e, None]
        along = (delta[:, [a,b]]*n[:, e, None]).sum(-1)
        tangent = torch.stack((-n[:, e, 1], n[:, e, 0]), -1)
        across = (delta[:, [a,b]]*tangent[:, None]).sum(-1)
        relations.append(torch.stack((before, after, before.abs()-after.abs(), along, across), -1)/d)
    add('endpoint_relationships', torch.stack(relations, 1))
    incident = []
    for k in range(8):
        edges = [e for e, pair in enumerate(EDGES) if k in pair]
        angles = [(n[:, a]*n[:, b]).sum(-1).abs().clamp(0,1).acos()/torch.pi for a,b in itertools.combinations(edges,2)]
        count = (proposal['utility'][:, edges]>0).float().sum(-1)/len(edges)
        incident.append(torch.stack([*angles, count],-1))
    add('incident_angles_active_counts', torch.stack(incident,1))
    # Fixed channel grouping, not a learned backbone. Preserve spatial evidence.
    f = observation['features']; b, channels, h, w = f.shape
    reduced = f.reshape(b, 12, channels//12, h, w).mean(2)
    content = observation['content'].to(f.dtype)
    add('image_global', (reduced*content).sum((-2,-1))/content.sum((-2,-1)).clamp_min(1))
    add('image_spatial_2x2', F.adaptive_avg_pool2d(reduced*content,(2,2)))
    edge_mid = torch.stack([(p[:,a]+p[:,b])/2 for a,b in EDGES],1)
    sites = torch.cat((p,q,edge_mid),1)
    grid = 2*(sites-origin)/size-1
    local = F.avg_pool2d(reduced*content,3,stride=1,padding=1)
    sampled = F.grid_sample(local,grid[:, :, None],align_corners=False,padding_mode='zeros')
    add('image_local_P_Q_edge', sampled)
    offsets = {}; start = 0
    for name, value in groups:
        offsets[name] = [start,start+value.shape[1]]; start += value.shape[1]
    result = torch.cat([v for _,v in groups],1).detach()
    if not torch.isfinite(result).all():
        raise ValueError('non-finite selector evidence')
    return result, offsets


class GainSelector(nn.Module):
    def __init__(self, dimension, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.detach().clone())
        self.register_buffer('std', std.detach().clone().clamp_min(1e-5))
        self.net = nn.Sequential(nn.Linear(dimension,64),nn.SiLU(),nn.Dropout(.1),
                                 nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # Output in 1000 * normalized gain for stable regression conditioning.
        return self.net(((x.detach()-self.mean)/self.std).clamp(-10,10)).squeeze(-1)


def metrics(rows, selected):
    pool=[]; values=[]; damage=good=covered=catastrophic=targets=0
    for row, pick in zip(rows, selected):
        pe=np.asarray(row['p_error']); qe=np.asarray(row['q_error'])
        e=qe if pick else pe; mask=np.asarray(row['mask'],bool)
        mean=row['q_mean'] if pick else row['p_mean']
        values.append(mean/row['diagonal'])
        pool.extend(e[mask] if mask.any() else [row['diagonal']]*8)
        g=mask & (pe<=5); good+=int(g.sum()); damage+=int((g & (e>10)).sum())
        covered+=row['coverage'][int(pick)]; targets+=int(mask.sum())
        catastrophic+=int(row['p_mean']<=10 and mean>50)
    return dict(primary=float(np.mean(values)),median_px=float(np.median(pool)),p90_px=float(np.percentile(pool,90)),
                coverage=covered,target_count=targets,coverage_rate=covered/max(targets,1),
                good_point_count=good,good_point_damage_count=damage,good_point_damage_rate=damage/max(good,1),
                catastrophic=catastrophic)


def gate(baseline, method):
    checks=dict(primary_gain_ge_1pct=(baseline['primary']-method['primary'])/baseline['primary']>=.01,
                median_nonworse=method['median_px']<=baseline['median_px'],p90_nonworse=method['p90_px']<=baseline['p90_px'],
                coverage_nonworse=method['coverage']>=baseline['coverage'],good_damage_le_0p5pct=method['good_point_damage_rate']<=.005,
                catastrophic_zero=method['catastrophic']==0)
    return dict(PASS=all(checks.values()),checks=checks)
