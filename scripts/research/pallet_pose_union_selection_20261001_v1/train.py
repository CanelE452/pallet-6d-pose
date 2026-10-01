"""Bounded, source-TRAIN-only whole-pose linear ranking.

Importing this module is pure: scoring is shared with the separately guarded VAL
process. Actual fitting installs stricter reference guards and requires both the
sealed feasibility gate and a separately sealed training protocol. ``selfcheck``
uses analytical arrays only; it performs no data fit or artifact read.
"""
from . import common as C
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ARMS = ('R0_ONLY', 'UNION')
SEEDS = (1, 2, 3)
HYP = ('long-face-front', 'short-face-front')
DIM, ROWS, EPOCHS, BATCH, UPDATES = 94, 2598, 30, 256, 330
PROTOCOL = C.DOC / 'TRAIN_PROTOCOL.json'
READS = None
_AUTHORIZED_INPUTS = object()


def install_training_guard():
    global READS
    if READS is not None:
        return
    READS = C.source_guard(allow_source_targets=False)
    allowed_npz = {str((C.RAW / name).resolve()) for name in
                   ('SOURCE_FEATURES.npz', 'SOURCE_TRAIN_LABELS.npz')}

    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(C.ROOT) or path.suffix == '.py':
            return
        # These checks apply to reads and writes: fitting never creates VAL or
        # real-quality artifacts either. Code bindings remain readable.
        assert not any(s in path.name.upper() for s in
                       ('SOURCE_VAL', 'VAL_LABEL', 'VAL_QUALITY', 'POSE_METRICS', 'RESULTS')), ('VAL_OR_REAL_QUALITY_DENIED', str(path))
        if path.suffix == '.npz':
            assert str(path) in allowed_npz, ('UNAPPROVED_ARRAY_CONTAINER', str(path))
    sys.addaudithook(hook)


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(array_sha(state[name].detach().cpu().numpy()).encode())
    return digest.hexdigest()


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(DIM, 1)

    def forward(self, value):
        return self.net(value).squeeze(-1)


def initialized(seed):
    torch.manual_seed(seed)
    model = Scorer().cpu()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 95
    return model


def candidate_names(arm, seed):
    assert arm in ARMS and seed in SEEDS
    names = ['R0:' + h for h in HYP]
    if arm == 'UNION':
        names += [f'DIVERSE251_s{seed}:' + h for h in HYP]
    return names


def tie_key(name):
    expert, hyp = name.split(':')
    assert hyp in HYP
    return expert != 'R0', hyp, expert


def targets(errors, valid, scale, names):
    """One whole-pose target; exact cost, Pareto, R0, hypothesis name."""
    errors = np.asarray(errors, np.float64)
    valid = np.asarray(valid, bool)
    scale = np.asarray(scale, np.float64)
    assert errors.shape == (*valid.shape, 2) and errors.ndim == 3
    assert len(names) == valid.shape[1] and len(set(names)) == len(names)
    assert scale.shape == (2,) and np.isfinite(scale).all() and (scale > 0).all()
    assert np.isfinite(errors[valid]).all() and (errors[valid] >= 0).all()
    assert np.isposinf(errors[~valid]).all()
    out = np.full(len(errors), -1, np.int64)
    for i, row in enumerate(errors):
        usable = np.flatnonzero(valid[i])
        if not len(usable):
            continue
        cost = np.max(row[usable] / scale, axis=1)
        assert np.isfinite(cost).all()
        ties = usable[cost == cost.min()].tolist()
        frontier = [j for j in ties if not any(
            np.all(row[k] <= row[j]) and np.any(row[k] < row[j]) for k in ties)]
        out[i] = min(frontier, key=lambda j: tie_key(names[j]))
    return out


def masked_loss(score, valid, target):
    """CE(-score); 0/1-candidate rows contribute zero without dropping rows."""
    assert score.shape == valid.shape and target.shape == (len(score),)
    assert valid.dtype == torch.bool and torch.isfinite(score).all()
    count = valid.sum(1)
    assert torch.equal(target >= 0, count > 0)
    present = count > 0
    if present.any():
        assert valid[present, target[present]].all()
    active = count >= 2
    if not active.any():
        return score.sum() * 0.
    logits = (-score[active]).masked_fill(~valid[active], float('-inf'))
    return nn.functional.cross_entropy(logits, target[active], reduction='sum') / len(score)


def score_candidates(checkpoint, raw_features, valid):
    """Same Linear94 checkpoint format as the historical GEO_LINEAR scorer."""
    assert checkpoint['variant'] == 'GEO_LINEAR' and checkpoint['d'] == DIM
    raw_features, valid = np.asarray(raw_features), np.asarray(valid)
    assert valid.dtype == bool and raw_features.shape == (*valid.shape, DIM)
    assert raw_features.ndim == 3 and np.isfinite(raw_features[valid]).all()
    mean, std = np.asarray(checkpoint['mean']), np.asarray(checkpoint['std'])
    assert mean.shape == std.shape == (DIM,) and np.isfinite(mean).all()
    assert np.isfinite(std).all() and (std >= 1e-6).all()
    x = np.zeros_like(raw_features, dtype=np.float32)
    x[valid] = (raw_features[valid] - mean) / std
    assert np.isfinite(x).all()
    model = Scorer()
    model.load_state_dict(checkpoint['state'], strict=True)
    model.eval()
    with torch.no_grad():
        score = model(torch.from_numpy(x)).numpy()
    assert np.isfinite(score).all()
    return np.where(valid, score, np.inf)


def select_candidates(score, valid, names):
    """GT-free tie policy; -1 explicitly means no usable candidate."""
    score, valid = np.asarray(score), np.asarray(valid)
    assert valid.dtype == bool and score.shape == valid.shape and score.ndim == 2
    assert len(names) == score.shape[1] and len(set(names)) == len(names)
    assert np.isfinite(score[valid]).all()
    picks = []
    for row, mask in zip(score, valid):
        usable = np.flatnonzero(mask)
        picks.append(min(usable, key=lambda j: (float(row[j]), tie_key(names[j]))) if len(usable) else -1)
    return np.asarray(picks, np.int64)


def orders(seed, n=ROWS):
    rng = np.random.default_rng(seed)
    return np.asarray([rng.permutation(n) for _ in range(EPOCHS)], np.int64)


def source_median(values):
    """Bitwise operation order used by the frozen source-gate quantile."""
    values = np.sort(np.asarray(values, float))
    assert len(values) and not np.isnan(values).any() and not np.isneginf(values).any()
    rank = (len(values) - 1) * .5
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return float(values[lo])
    if np.isposinf(values[hi]):
        return float('inf')
    return float(values[lo] + (rank - lo) * (values[hi] - values[lo]))


def checked_binding(binding, path):
    assert binding['path'] == str(Path(path).relative_to(C.ROOT)), (binding, str(path))
    C.verify(binding)
    return C.read(path) if Path(path).suffix == '.json' else None


def load_training_inputs():
    install_training_guard()
    seal = C.read(C.DOC / 'TRAIN_PROTOCOL_SHA.json')
    protocol = checked_binding(seal, PROTOCOL)
    expected = dict(schema='pallet_pose_union_train_v1', arms=list(ARMS), seeds=list(SEEDS),
                    train_rows=ROWS, feature_dim=DIM, epochs=EPOCHS, batch_size=BATCH,
                    updates_per_fit=UPDATES, device='cpu', max_fits=6)
    for key, value in expected.items():
        assert protocol[key] == value, (key, protocol.get(key), value)
    assert protocol['optimizer'] == dict(name='AdamW', lr=.001, weight_decay=.0001,
                                        betas=[.9, .999], eps=1e-8)
    assert protocol['normalization'] == dict(source='R0_eligible_train_valid_candidates', std_floor=1e-6)
    code = {b['path']: b for b in protocol['codes']}
    for path in (Path(__file__), Path(C.__file__)):
        assert str(path.resolve().relative_to(C.ROOT)) in code
    for binding in code.values():
        C.verify(binding)
    inputs = protocol['inputs']
    for binding in inputs.values():
        C.verify(binding)
    features = C.RAW / 'SOURCE_FEATURES.npz'
    labels = C.RAW / 'SOURCE_TRAIN_LABELS.npz'
    checked_binding(inputs['features'], features)
    checked_binding(inputs['train_labels'], labels)
    gate = checked_binding(inputs['train_gate'], C.DOC / 'SOURCE_TRAIN_GATE.json')
    feasible = checked_binding(inputs['feasibility_protocol'], C.DOC / 'TRAIN_FEASIBILITY_PROTOCOL.json')
    checked_binding(C.read(C.DOC / 'TRAIN_FEASIBILITY_PROTOCOL_SHA.json'), C.DOC / 'TRAIN_FEASIBILITY_PROTOCOL.json')
    predlock = checked_binding(inputs['source_predictions_lock'], C.DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert gate['complete'] and gate['PASS'] and gate['scope'] == 'C2_TRAIN_ONLY'
    assert not gate['VAL_quality_scored'] and not gate['C1_quality_scored']
    assert gate['frames'] == ROWS and gate['fits'] == 0
    assert gate['labels'] == inputs['train_labels'] and gate['protocol'] == inputs['feasibility_protocol']
    assert feasible['scope'] == 'C2_TRAIN_ONLY' and feasible['eligible_counts'] == dict(TRAIN=ROWS, VAL=1024)
    assert feasible['scalar_cost'] == 'max(T/sT,R/sR)'
    assert feasible['tie_rule'] == 'exact_cost_then_pareto_then_R0_then_hypothesis'
    assert feasible['scale'] == 'median_R0_GEO_C2_TRAIN'
    assert feasible['median_strict'] and feasible['failure_no_increase'] and feasible['p90_ratio_max'] == 1.05
    featurelock = checked_binding(gate['feature_lock'], C.DOC / 'SOURCE_FEATURE_LOCK.json')
    assert gate['feature_lock'] == inputs['feature_lock']
    assert featurelock['complete'] and not featurelock['source_targets_read']
    assert not featurelock['real_targets_read'] and not featurelock['VAL_quality_scored']
    assert featurelock['features'] == inputs['features'] and featurelock['predictions'] == inputs['source_predictions_lock']
    assert featurelock['models'] == list(C.MODELS) and featurelock['hypothesis_names'] == list(HYP)
    assert len(featurelock['feature_names']) == DIM
    assert predlock['complete'] and predlock['models'] == list(C.MODELS) and predlock['frames'] == 5120
    assert predlock['protocol'] == featurelock['protocol'] == feasible['source_protocol']
    checked_binding(feasible['source_protocol'], C.DOC / 'SOURCE_PROTOCOL.json')
    amendment = checked_binding(inputs['runtime_amendment'], C.DOC / 'SOURCE_RUNTIME_AMENDMENT_01.json')
    checked_binding(C.read(C.DOC / 'SOURCE_RUNTIME_AMENDMENT_01_SHA.json'), C.DOC / 'SOURCE_RUNTIME_AMENDMENT_01.json')
    assert amendment['complete'] and amendment['original_protocol'] == predlock['protocol']
    assert predlock['runtime_amendment'] == inputs['runtime_amendment']
    contract = checked_binding(gate['source_contract'], C.DOC / 'SOURCE_CONTRACT.json')
    assert gate['source_contract'] == feasible['source_contract'] == inputs['source_contract']
    assert contract['complete'] and contract['status'] == 'PASS'
    eligible = contract['fit_eligibility']
    assert eligible['eligible_counts'] == dict(TRAIN=ROWS, VAL=1024)
    # Do not recursively verify the gate/contract: their raw source-reference
    # bindings contain broader-label NPZs forbidden to this training process.
    fkeys = {'ids', 'split', 'hypothesis_names'} | {m + s for m in C.MODELS for s in ('_geo', '_valid', '_GEO_index')}
    lkeys = {'ids', 'source_index', 'eligible_train_mask', 'hypothesis_names', 'sT_cm', 'sR_deg'} | {
        m + s for m in C.MODELS for s in ('_T_cm', '_R_deg', '_GEO_T_cm', '_GEO_R_deg')}
    with np.load(features, allow_pickle=False) as data:
        assert set(data.files) == fkeys
        f = {k: data[k] for k in data.files}
    with np.load(labels, allow_pickle=False) as data:
        assert set(data.files) == lkeys
        label = {k: data[k] for k in data.files}
    assert len(f['ids']) == len(set(f['ids'].tolist())) == 5120
    assert f['ids'].ndim == f['split'].ndim == 1 and len(f['split']) == 5120
    assert int(np.sum(f['split'] == 'TRAIN')) == 4096 and int(np.sum(f['split'] == 'VAL')) == 1024
    assert f['hypothesis_names'].tolist() == label['hypothesis_names'].tolist() == list(HYP)
    idx, mask = label['source_index'], label['eligible_train_mask']
    assert idx.shape == (ROWS,) and np.issubdtype(idx.dtype, np.integer)
    assert mask.shape == (5120,) and mask.dtype == bool
    np.testing.assert_array_equal(idx, np.flatnonzero(mask))
    np.testing.assert_array_equal(f['ids'][idx], label['ids'])
    assert np.all(f['split'][idx] == 'TRAIN') and len(set(label['ids'].tolist())) == ROWS
    assert set(label['ids'].tolist()) == set(eligible['eligible_ids']['TRAIN'])
    assert set(label['ids'].tolist()).isdisjoint(eligible['eligible_ids']['VAL'])
    scale = np.array([label['sT_cm'].item(), label['sR_deg'].item()], np.float64)
    assert np.isfinite(scale).all() and (scale > 0).all()
    errors, feature, valid = {}, {}, {}
    for model in C.MODELS:
        x, v = f[model + '_geo'], f[model + '_valid']
        assert x.shape == (5120, 2, DIM) and x.dtype == np.float32
        assert v.shape == (5120, 2) and v.dtype == bool and np.isfinite(x[v]).all()
        feature[model], valid[model] = x[idx], v[idx]
        err = np.stack([label[model + '_T_cm'], label[model + '_R_deg']], axis=-1)
        assert err.shape == (ROWS, 2, 2)
        assert np.isfinite(err[v[idx]]).all() and (err[v[idx]] >= 0).all()
        assert np.isposinf(err[~v[idx]]).all()
        errors[model] = err
        op = np.stack([label[model + '_GEO_T_cm'], label[model + '_GEO_R_deg']], axis=-1)
        assert op.shape == (ROWS, 2) and not np.isnan(op).any() and not np.isneginf(op).any()
        assert (op >= 0).all() and np.all(np.isfinite(op).all(1) == np.isfinite(op).any(1))
        if model == 'R0':
            np.testing.assert_array_equal(scale, [source_median(op[:, j]) for j in range(2)])
    vals = feature['R0'][valid['R0']]
    assert vals.shape[1] == DIM and len(vals) > 1
    mean, std = vals.mean(0), np.maximum(vals.std(0), np.float32(1e-6))
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    normalization_sha = array_sha(np.stack([mean, std]))
    return dict(_authorized=_AUTHORIZED_INPUTS, protocol=protocol, binding=C.bind(PROTOCOL), ids=label['ids'], source_index=idx,
                features=feature, valid=valid, errors=errors, scale=scale, mean=mean, std=std,
                normalization_sha=normalization_sha)


def fit_paths(arm, seed):
    stem = f'{arm}_s{seed}'
    folder = C.RAW / 'fits' / stem
    return folder, C.DOC / f'FIT_{stem}.json'


def verify_completed(arm, seed, data):
    folder, receipt_path = fit_paths(arm, seed)
    receipt = C.read(receipt_path)
    assert receipt['complete'] and receipt['arm'] == arm and receipt['seed'] == seed
    assert receipt['protocol'] == data['binding'] and receipt['updates'] == UPDATES
    assert receipt['normalization_sha'] == data['normalization_sha']
    assert receipt['order_sha'] == array_sha(orders(seed))
    for name, path in [('START', folder / 'START.json'), ('trace', folder / 'TRACE.jsonl'), ('checkpoint', folder / 'final.pt')]:
        checked_binding(receipt[name], path)
    start = C.read(folder / 'START.json')
    assert start['protocol'] == data['binding'] and start['arm'] == arm and start['seed'] == seed
    assert start['initial_state_sha'] == receipt['initial_state_sha'] == state_sha(initialized(seed).state_dict())
    assert start['normalization_sha'] == receipt['normalization_sha'] and start['order_sha'] == receipt['order_sha']
    with (folder / 'TRACE.jsonl').open() as handle:
        trace = [json.loads(line) for line in handle]
    assert len(trace) == UPDATES and [r['step'] for r in trace] == list(range(1, UPDATES + 1))
    checkpoint = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['arm'] == arm and checkpoint['seed'] == seed and checkpoint['protocol'] == data['binding']
    assert checkpoint['candidate_names'] == candidate_names(arm, seed)
    assert state_sha(checkpoint['state']) == receipt['final_state_sha']
    assert trace[-1]['state_sha'] == receipt['final_state_sha']
    np.testing.assert_array_equal(checkpoint['mean'], data['mean'])
    np.testing.assert_array_equal(checkpoint['std'], data['std'])
    expected_rows = [array_sha(batch) for epoch in orders(seed) for batch in
                     (epoch[j:j + BATCH] for j in range(0, ROWS, BATCH))]
    assert [r['rows_sha'] for r in trace] == expected_rows
    assert all(np.isfinite(r['loss']) for r in trace)
    return receipt


def fit_one(arm, seed, data):
    assert arm in ARMS and seed in SEEDS
    install_training_guard()
    assert data.get('_authorized') is _AUTHORIZED_INPUTS, 'LOAD_SEALED_PASS_INPUTS_BEFORE_FIT'
    C.verify(data['binding'])
    for binding in list(data['protocol']['inputs'].values()) + data['protocol']['codes']:
        C.verify(binding)
    folder, receipt_path = fit_paths(arm, seed)
    if receipt_path.exists():
        return verify_completed(arm, seed, data)
    if folder.exists():
        assert not any(folder.iterdir()), ('PARTIAL_FIT_STOP_NO_AUTOMATIC_REPEAT', str(folder))
    fit_root = C.RAW / 'fits'
    allowed = {f'{a}_s{s}' for a in ARMS for s in SEEDS}
    if fit_root.exists():
        assert all(p.name in allowed and p.is_dir() for p in fit_root.iterdir()), 'UNEXPECTED_FIT_BUDGET_ENTRY'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert EPOCHS * math.ceil(ROWS / BATCH) == UPDATES
    names = candidate_names(arm, seed)
    models = ['R0'] + ([f'DIVERSE251_s{seed}'] if arm == 'UNION' else [])
    raw = np.concatenate([data['features'][m] for m in models], axis=1)
    valid = np.concatenate([data['valid'][m] for m in models], axis=1)
    errors = np.concatenate([data['errors'][m] for m in models], axis=1)
    y = targets(errors, valid, data['scale'], names)
    x = np.zeros_like(raw, dtype=np.float32)
    x[valid] = (raw[valid] - data['mean']) / data['std']
    assert np.isfinite(x).all()
    xx, vv, yy = torch.from_numpy(x), torch.from_numpy(valid), torch.from_numpy(y)
    order = orders(seed)
    model = initialized(seed)
    initial_hash = state_sha(model.state_dict())
    folder.mkdir(parents=True, exist_ok=True)
    C.save(folder / 'START.json', dict(created_at=C.now(), arm=arm, seed=seed, protocol=data['binding'],
           initial_state_sha=initial_hash, normalization_sha=data['normalization_sha'], order_sha=array_sha(order),
           candidate_names=names, labels_sha=array_sha(y), ids_sha=array_sha(data['ids']),
           source_index_sha=array_sha(data['source_index']), rows=ROWS, epochs=EPOCHS, batch_size=BATCH,
           planned_updates=UPDATES, trainable_parameters=95, device='cpu', torch_threads=1,
           torch_version=str(torch.__version__), numpy_version=np.__version__,
           zero_valid_rows=int(np.sum(valid.sum(1) == 0)), one_valid_rows=int(np.sum(valid.sum(1) == 1)),
           policy='Final epoch only; no VAL reads, no early stopping, no retry of any partial fit.'))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
    started = time.monotonic()
    step = 0
    try:
        with (folder / 'TRACE.jsonl').open('x') as trace:
            for epoch, rows in enumerate(order, start=1):
                for offset in range(0, ROWS, BATCH):
                    batch = rows[offset:offset + BATCH]
                    opt.zero_grad(set_to_none=True)
                    score = model(xx[batch])
                    loss = masked_loss(score, vv[batch], yy[batch])
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    opt.step()
                    assert all(torch.isfinite(p).all() for p in model.parameters())
                    step += 1
                    entry = dict(step=step, epoch=epoch, batch_size=len(batch), rows_sha=array_sha(batch),
                                 source_rows_sha=array_sha(data['source_index'][batch]), targets_sha=array_sha(y[batch]),
                                 loss=float(loss.detach()), active_rows=int(np.sum(valid[batch].sum(1) >= 2)),
                                 state_sha=state_sha(model.state_dict()))
                    trace.write(json.dumps(entry, allow_nan=False) + '\n')
                    trace.flush()
                os.fsync(trace.fileno())
                print('UNION_LINEAR_FIT', arm, seed, epoch, step, entry['loss'], flush=True)
        assert step == UPDATES
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_hash = state_sha(state)
        assert final_hash != initial_hash
        checkpoint = dict(variant='GEO_LINEAR', d=DIM, state=state, mean=data['mean'], std=data['std'],
                          arm=arm, seed=seed, candidate_names=names, protocol=data['binding'],
                          epochs=EPOCHS, updates=step, selection='final_epoch_only',
                          normalization_sha=data['normalization_sha'], order_sha=array_sha(order))
        with (folder / 'final.pt').open('xb') as handle:
            torch.save(checkpoint, handle)
        C.save(receipt_path, dict(complete=True, created_at=C.now(), arm=arm, seed=seed,
               protocol=data['binding'], START=C.bind(folder / 'START.json'), trace=C.bind(folder / 'TRACE.jsonl'),
               checkpoint=C.bind(folder / 'final.pt'), updates=step, initial_state_sha=initial_hash,
               final_state_sha=final_hash, normalization_sha=data['normalization_sha'], order_sha=array_sha(order),
               source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
               wall_seconds=time.monotonic() - started, read_paths=sorted(set(READS))))
    except BaseException as exc:
        failed = folder / 'FAILED.json'
        if not failed.exists():
            C.save(failed, dict(created_at=C.now(), arm=arm, seed=seed, completed_updates=step,
                   error_type=type(exc).__name__, error=str(exc), automatic_resume=False))
        raise
    return verify_completed(arm, seed, data)


def complete_all(data):
    receipts = [verify_completed(arm, seed, data) for seed in SEEDS for arm in ARMS]
    pairs = {}
    for seed in SEEDS:
        a, b = [r for r in receipts if r['seed'] == seed]
        keys = ('initial_state_sha', 'normalization_sha', 'order_sha')
        assert all(a[k] == b[k] for k in keys)
        pairs[str(seed)] = {k: a[k] for k in keys}
        paths = [C.ROOT / r['trace']['path'] for r in (a, b)]
        traces = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
        assert all((a['rows_sha'], a['source_rows_sha'], a['batch_size']) ==
                   (b['rows_sha'], b['source_rows_sha'], b['batch_size']) for a, b in zip(*traces))
    assert len({r['initial_state_sha'] for r in receipts}) == len(SEEDS)
    final = C.DOC / 'TRAINING_COMPLETE.json'
    result = dict(complete=True, created_at=C.now(), protocol=data['binding'],
                  fits=[C.bind(fit_paths(a, s)[1]) for s in SEEDS for a in ARMS],
                  fit_count=6, total_updates=6 * UPDATES, paired_checks=pairs,
                  all_final_epoch_only=True, source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False)
    if final.exists():
        previous = C.read(final)
        for key in result.keys() - {'created_at'}:
            assert previous[key] == result[key], key
    else:
        C.save(final, result)
    return result


def selfcheck():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hashes = []
    for seed in SEEDS:
        a, b = initialized(seed), initialized(seed)
        assert state_sha(a.state_dict()) == state_sha(b.state_dict())
        hashes.append(state_sha(a.state_dict()))
        assert array_sha(orders(seed)) == array_sha(orders(seed))
    assert len(set(hashes)) == 3 and len({array_sha(orders(s)) for s in SEEDS}) == 3
    assert EPOCHS * math.ceil(ROWS / BATCH) == UPDATES
    assert source_median([14.424519675836176, 94.86545821925301]) == 54.64498894754459
    v = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    y = torch.tensor([0, 0, -1])
    score = torch.tensor([[0., 1., -1e6], [2., -1e6, 5.], [8., 9., 10.]], requires_grad=True)
    loss = masked_loss(score, v, y)
    np.testing.assert_allclose(loss.item(), np.log1p(np.exp(-1)) / 3., rtol=1e-6)
    loss.backward()
    assert torch.isfinite(score.grad).all() and torch.equal(score.grad[~v], torch.zeros_like(score.grad[~v]))
    assert torch.equal(score.grad[1:], torch.zeros_like(score.grad[1:]))
    swapped = score.detach().clone()
    swapped[0, :2] = torch.tensor([1., 0.])
    assert masked_loss(swapped, v, y) > loss.detach()
    empty_score = torch.randn(2, 2, requires_grad=True)
    zero = masked_loss(empty_score, torch.tensor([[False, False], [True, False]]), torch.tensor([-1, 0]))
    zero.backward()
    assert zero.item() == 0. and torch.equal(empty_score.grad, torch.zeros_like(empty_score))
    names = candidate_names('UNION', 1)
    examples = np.array([[[2., 2.], [2., 1.], [2., .5], [9., 9.]],
                         [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                         [[1., 4.], [4., 1.], [3., 3.], [9., 9.]],
                         [[np.inf, np.inf]] * 4])
    valid = np.isfinite(examples).all(-1)
    chosen = targets(examples, valid, [1., 1.], names)
    np.testing.assert_array_equal(chosen, [2, 0, 2, -1])
    perm = np.array([3, 2, 1, 0])
    other = targets(examples[:, perm], valid[:, perm], [1., 1.], [names[j] for j in perm])
    np.testing.assert_array_equal(np.where(other >= 0, perm[np.maximum(other, 0)], -1), chosen)
    score = np.array([[0., 0., 0., -100.], [np.inf] * 4])
    picks = select_candidates(score, np.array([[True, True, True, False], [False] * 4]), names)
    np.testing.assert_array_equal(picks, [0, -1])
    model = initialized(1)
    checkpoint = dict(variant='GEO_LINEAR', d=DIM, state=model.state_dict(), mean=np.zeros(DIM, np.float32), std=np.ones(DIM, np.float32))
    x = np.random.default_rng(10).normal(size=(3, 4, DIM)).astype(np.float32)
    v = np.array([[True, True, False, True], [False] * 4, [True] * 4])
    scored = score_candidates(checkpoint, x, v)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(scored[v], expected[v], rtol=0, atol=1e-7)
    assert np.isposinf(scored[~v]).all()
    print('SELF_CHECK_PASS: paired init/orders; 330 updates; masked CE sign/denominator/grad; zero/one candidate; exact cost/Pareto/R0/name ties; whole-pose tradeoff; scoring parity. No data fit.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'fit', 'all'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == 'selfcheck':
        selfcheck()
        return
    install_training_guard()
    selfcheck()
    data = load_training_inputs()
    if args.stage == 'fit':
        assert args.arm is not None and args.seed is not None
        fit_one(args.arm, args.seed, data)
    else:
        assert args.arm is None and args.seed is None
        for seed in SEEDS:
            for arm in ARMS:
                fit_one(arm, seed, data)
        complete_all(data)


if __name__ == '__main__':
    main()
