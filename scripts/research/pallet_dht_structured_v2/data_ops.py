"""Prediction-only perturbations/proposals and separate synthetic supervision."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from .model import EDGES, INPUT_KEYS
from .proposals import C4, build_proposals

STATES = ('clean', 'point_c4', 'point_and_line_c4', 'local_deformation')
STATE_PROBABILITIES = (.5, .2, .2, .1)
FACES = np.asarray(((0,1,2,3),(4,5,6,7),(0,1,5,4),(3,2,6,7),(0,3,7,4),(1,2,6,5)))


def fixed_states(indices, state, seed=20260909):
    """Deterministic corruption, independent of labels and model/arm."""
    if state not in STATES:
        raise ValueError(state)
    variants = []
    for index in indices:
        digest = hashlib.sha256(f'pallet_dht_structured_v2:fixed:{seed}:{state}:{int(index)}'.encode()).digest()
        values = np.frombuffer(digest, dtype='<u8').astype(np.float64) / 2.**64
        variants.append(np.minimum(values, np.nextafter(1., 0.)))
    return [state]*len(indices), np.asarray(variants, np.float64).reshape(-1,4)


def prepare_batch(inputs, targets, states, variants):
    """Return model inputs, proposal mask, targets. Never consult target values.

    Corruption affects predictions only; GT stays in its original semantic IDs.
    Accept CPU or CUDA input tensors and retain their device.
    """
    b = len(states)
    variants = np.asarray(variants, np.float64)
    if variants.shape != (b,4) or not ((variants >= 0) & (variants < 1)).all():
        raise ValueError('Expected four [0,1) variates per observation')
    arrays = {key: inputs[key].detach().cpu().numpy().copy() for key in
              ('baseline_points','point_conf','point_valid','intersections','intersection_valid',
               'line_h','peak_logits','peak_valid','diagonal')}
    edge_lookup = {tuple(sorted(edge)): r for r,edge in enumerate(EDGES)}
    layouts, masks = [], []
    for j,state in enumerate(states):
        if state not in STATES:
            raise ValueError(state)
        # Missing predictions always retain the unmodified original layout.
        if arrays['point_valid'][j,:8].all():
            if state in ('point_c4','point_and_line_c4'):
                perm = C4[1+int(variants[j,0]*3)]
                for key in ('baseline_points','point_conf','point_valid'):
                    arrays[key][j,:8] = arrays[key][j,:8][perm].copy()
                if state == 'point_and_line_c4':
                    roles = [edge_lookup[tuple(sorted((int(perm[u]),int(perm[v]))))] for u,v in EDGES]
                    for key in ('line_h','peak_logits','peak_valid'):
                        arrays[key][j] = arrays[key][j][roles].copy()
                    for key in ('intersections','intersection_valid'):
                        arrays[key][j] = arrays[key][j][perm].copy()
            elif state == 'local_deformation':
                face = FACES[int(variants[j,1]*6)]
                angle = 2*np.pi*variants[j,2]
                magnitude = (.01+.03*variants[j,3])*arrays['diagonal'][j]
                arrays['baseline_points'][j,face] += magnitude*np.asarray([np.cos(angle),np.sin(angle)])
        proposal = build_proposals(arrays['baseline_points'][j], arrays['intersections'][j],
            arrays['intersection_valid'][j], float(arrays['diagonal'][j]), point_valid=arrays['point_valid'][j])
        layouts.append(proposal['layouts'].astype(np.float32))
        masks.append(proposal['valid'])
    device = inputs['p4'].device
    model_inputs = {key: inputs[key] for key in INPUT_KEYS if key != 'layouts'}
    for key in arrays:
        if key in model_inputs:
            model_inputs[key] = torch.from_numpy(arrays[key]).to(device)
    model_inputs['layouts'] = torch.from_numpy(np.stack(layouts)).to(device)
    return model_inputs, torch.from_numpy(np.stack(masks)).to(device), targets


def supervised_errors(batch, targets):
    error = torch.linalg.vector_norm(batch['layouts'][:,:,:8]-targets['points'][:,None,:8], dim=-1)
    mask = targets['loss_valid'][:,:8].bool()
    count = mask.sum(-1)
    mean = (error*mask[:,None]).sum(-1)/count[:,None].clamp_min(1)
    scale = .01*batch['diagonal'][:,None]
    return dict(error_px=error, mask=mask, frame_valid=count>0,
                target_cost=torch.log1p(mean/scale),
                target_corner=torch.log1p(error/scale[:,:,None]))


def training_loss(cost, diagnostics, batch, candidate_valid, targets, cfg):
    values = supervised_errors(batch, targets)
    frames = values['frame_valid']
    if not bool(frames.any()):
        raise ValueError('No supervised synthetic corners in optimizer batch')
    temperature = cfg['listwise_temperature']
    predicted_logprob = F.log_softmax((-cost/temperature).masked_fill(~candidate_valid, -1e9), -1)
    target_prob = F.softmax((-values['target_cost']/temperature).masked_fill(~candidate_valid, -1e9), -1)
    ranking = (-(target_prob*predicted_logprob).sum(-1))[frames].mean()
    active = candidate_valid & frames[:,None]
    regression = F.smooth_l1_loss(cost[active], values['target_cost'][active], beta=cfg['regression_beta'])
    corner_active = candidate_valid[:,:,None] & values['mask'][:,None]
    corner = F.smooth_l1_loss(diagnostics['percorner_error'][corner_active],
        values['target_corner'][corner_active], beta=cfg['regression_beta'])
    total = ranking+cfg['regression_weight']*regression+cfg['corner_regression_weight']*corner
    return total, dict(ranking=float(ranking.detach()), regression=float(regression.detach()),
        corner=float(corner.detach()), supervised_corners=int(values['mask'].sum()))
