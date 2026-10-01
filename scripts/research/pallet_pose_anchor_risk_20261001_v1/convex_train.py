"""Certified fixed189 scorer with TRAIN-only log1p anchor-risk margins.

The explicit CE + lambda/2 * ||w||^2 objective differs from the original
decoupled AdamW decay. Bias is fixed at zero because it cancels in ranking.
The previous anchored TRAIN targets and fixed189 feature map are unchanged.
Only TRAIN logits receive a frozen log1p normalized anchor-excess margin.
No margin, label, or reference error is used by the shared runtime scorer.
The original validity mask remains in CE and GT-free runtime selection.
Imports and ``selfcheck`` read no artifacts. No fit before a sealed PASS gate.
"""
from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_selector_convergence_20261001_v1 import convex_train as PREV
from scripts.research.pallet_pose_pareto_anchor_20261001_v1 import convex_train as ANCHOR
from scripts.research.pallet_pose_anchor_context_20261001_v1 import convex_train as CONTEXT
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch
from threadpoolctl import threadpool_limits

LAMBDA = 1e-4
MAX_CALLS = 2000
MAX_ITER = 1000
GAP_MAX = 1e-6
PROTOCOL = C.DOC / 'TRAIN_PROTOCOL.json'
TARGET_RULE = 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
FEATURE_MAP = 'normalized94_abs_anchor_delta94_identity1'
FEATURE_DIM = 189
RAW_FEATURE_DIM = 94
CHECKPOINT_SCHEMA = 'pallet_pose_anchor_risk_linear189_v1'
LOSS_RULE = 'TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE'
_AUTHORIZED = object()


class ClosureBudgetExceeded(RuntimeError):
    pass


def candidate_names(model):
    assert model in C.MODEL_NAMES
    return OLD.candidate_names('R0_ONLY', 1) if model == 'R0_ONLY' else OLD.candidate_names('UNION', int(model[-1]))


def normalized_inputs(raw, valid, mean, std):
    """Retain the original float32 arithmetic, then promote exactly to float64."""
    raw, valid = np.asarray(raw), np.asarray(valid)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    assert raw.dtype == np.float32 and valid.dtype == bool
    assert raw.shape == (*valid.shape, 94) and raw.ndim == 3
    assert mean.shape == std.shape == (94,) and np.isfinite(mean).all()
    assert np.isfinite(std).all() and (std >= np.float32(1e-6)).all()
    assert np.isfinite(raw[valid]).all()
    x = np.zeros_like(raw, dtype=np.float32)
    x[valid] = (raw[valid] - mean) / std
    assert np.isfinite(x).all()
    return x.astype(np.float64)


def context_inputs(raw, valid, anchor_index, mean, std):
    """Input-only anchor context after exactly the previous FP32 normalization.

    The anchor must be one of the two original R0 candidate slots, never a
    reference-derived choice. A missing anchor with valid candidates stops.
    """
    valid, anchor_index = np.asarray(valid), np.asarray(anchor_index)
    assert valid.ndim == 2 and valid.dtype == bool and valid.shape[1] in (2, 4)
    assert anchor_index.shape == (len(valid),) and np.issubdtype(anchor_index.dtype, np.integer)
    assert np.isin(anchor_index, [-1, 0, 1]).all()
    present = valid.any(1)
    assert np.array_equal(anchor_index >= 0, present), 'ANCHOR_CANDIDATE_SUPPORT_CONTRACT'
    rows = np.flatnonzero(present)
    assert valid[rows, anchor_index[rows]].all(), 'INVALID_OPERATIONAL_ANCHOR'
    z = normalized_inputs(raw, valid, mean, std)
    anchor = np.zeros((len(valid), RAW_FEATURE_DIM), np.float64)
    anchor[rows] = z[rows, anchor_index[rows]]
    identity = np.zeros((*valid.shape, 1), np.float64)
    identity[rows, anchor_index[rows], 0] = 1.
    out = np.concatenate([z, np.abs(z - anchor[:, None, :]), identity], axis=2)
    out[~valid] = 0.
    assert out.shape == (*valid.shape, FEATURE_DIM) and np.isfinite(out).all()
    return out


def score_candidates(checkpoint, raw_features, valid, anchor_index):
    assert checkpoint['schema'] == CHECKPOINT_SCHEMA
    assert checkpoint['loss_rule'] == LOSS_RULE and checkpoint['runtime_uses_margin'] is False
    assert checkpoint['feature_map'] == FEATURE_MAP
    assert checkpoint['feature_dim'] == FEATURE_DIM and checkpoint['raw_feature_dim'] == RAW_FEATURE_DIM
    assert checkpoint['normalization'] == 'old_float32_then_float64'
    assert checkpoint['names'][:2] == OLD.candidate_names('R0_ONLY', 1)
    assert len(checkpoint['names']) == np.asarray(valid).shape[1]
    assert len(set(checkpoint['names'])) == len(checkpoint['names'])
    assert checkpoint['bias'] == 0. and checkpoint['lambda_l2'] == LAMBDA
    weight = np.asarray(checkpoint['weight'], np.float64)
    assert weight.shape == (FEATURE_DIM,) and np.isfinite(weight).all()
    x = context_inputs(raw_features, valid, anchor_index, checkpoint['mean'], checkpoint['std'])
    score = np.einsum('nkd,d->nk', x, weight)
    assert np.isfinite(score).all()
    return np.where(valid, score, np.inf)


select_candidates = OLD.select_candidates


def margin_arrays(errors, valid, anchor_errors, anchor_index, scale, target):
    """Frozen TRAIN excess/margin, without evaluating inf-inf on invalid rows."""
    errors, valid = np.asarray(errors, np.float64), np.asarray(valid)
    anchor_errors, anchor_index = np.asarray(anchor_errors, np.float64), np.asarray(anchor_index)
    scale, target = np.asarray(scale, np.float64), np.asarray(target)
    assert errors.ndim == 3 and errors.shape == (*valid.shape, 2)
    assert valid.dtype == bool and valid.shape[1] in (2, 4)
    assert anchor_errors.shape == (len(errors), 2) and anchor_index.shape == target.shape == (len(errors),)
    assert np.issubdtype(anchor_index.dtype, np.integer) and np.issubdtype(target.dtype, np.integer)
    assert np.isin(anchor_index, [-1, 0, 1]).all()
    assert scale.shape == (2,) and np.isfinite(scale).all() and (scale > 0).all()
    assert np.isfinite(errors[valid]).all() and (errors[valid] >= 0).all()
    assert np.isposinf(errors[~valid]).all()
    present = valid.any(1)
    assert np.array_equal(anchor_index >= 0, present) and np.array_equal(target >= 0, present)
    assert np.all(target[~present] == -1)
    assert np.isfinite(anchor_errors[present]).all() and (anchor_errors[present] >= 0).all()
    assert np.isposinf(anchor_errors[~present]).all()
    rows = np.flatnonzero(present)
    assert valid[rows, anchor_index[rows]].all() and valid[rows, target[rows]].all()
    np.testing.assert_array_equal(errors[rows, anchor_index[rows]], anchor_errors[rows])
    # Select finite pairs before subtracting; all-invalid +inf anchors never
    # participate in a ufunc, and unavailable candidates have exact zero risk.
    risk = np.zeros(valid.shape, np.float64)
    margin = np.zeros(valid.shape, np.float64)
    frame_index = np.nonzero(valid)[0]
    excess = (errors[valid] - anchor_errors[frame_index]) / scale
    assert np.isfinite(excess).all()
    risk[valid] = np.maximum(excess.max(1), 0.)
    margin[valid] = np.log1p(risk[valid])
    assert np.isfinite(risk).all() and np.isfinite(margin).all()
    assert (risk >= 0).all() and (margin >= 0).all()
    assert not risk[~valid].any() and not margin[~valid].any()
    assert np.all(risk[rows, target[rows]] == 0.) and np.all(margin[rows, target[rows]] == 0.), 'ANCHORED_TARGET_MARGIN_MUST_BE_ZERO'
    return risk, margin


def objective(weight, x, valid, target, margin, ridge=LAMBDA):
    """All-frame mean margin-adjusted CE, ridge, and analytic gradient.

    CE = mean(logsumexp(-score + margin) + score_target), with target
    margin exactly zero. unadjusted_CE is diagnostic and is not optimized.
    """
    weight, x = np.asarray(weight, np.float64), np.asarray(x, np.float64)
    valid, target = np.asarray(valid), np.asarray(target)
    assert x.ndim == 3 and x.shape[:2] == valid.shape and x.shape[-1] == len(weight)
    assert valid.dtype == bool and target.shape == (len(x),)
    assert np.issubdtype(target.dtype, np.integer)
    assert np.isfinite(weight).all() and np.isfinite(x).all() and ridge > 0
    margin = np.asarray(margin, np.float64)
    assert margin.shape == valid.shape and np.isfinite(margin).all() and (margin >= 0).all()
    assert not margin[~valid].any()
    present = valid.any(1)
    assert np.array_equal(target >= 0, present)
    assert np.all(target[~present] == -1)
    assert np.all(valid[np.flatnonzero(present), target[present]])
    assert np.all(margin[np.flatnonzero(present), target[present]] == 0.)
    scores = np.einsum('nkd,d->nk', x, weight)
    base_logits = -scores
    base_logits[~valid] = -np.inf
    logits = base_logits + margin
    probability = np.zeros(valid.shape, np.float64)
    ce_rows = np.zeros(len(x), np.float64)
    unadjusted_rows = np.zeros(len(x), np.float64)
    if present.any():
        normalizer = logsumexp(logits[present], axis=1)
        probability[present] = np.exp(logits[present] - normalizer[:, None])
        ce_rows[present] = normalizer + scores[present, target[present]]
        unadjusted_rows[present] = logsumexp(base_logits[present], axis=1) + scores[present, target[present]]
    target_x = np.zeros((len(x), len(weight)), np.float64)
    target_x[present] = x[np.flatnonzero(present), target[present]]
    gradient = (target_x - np.einsum('nk,nkd->nd', probability, x)).mean(0) + ridge * weight
    ce, penalty = float(ce_rows.mean()), float(.5 * ridge * np.dot(weight, weight))
    assert np.isfinite(gradient).all() and np.isfinite(ce + penalty)
    return dict(value=ce + penalty, CE=ce, unadjusted_CE=float(unadjusted_rows.mean()), penalty=penalty,
                gradient=gradient, probability=probability)


def hessian(x, probability, ridge=LAMBDA):
    expectation = np.einsum('nk,nkd->nd', probability, x)
    delta = x - expectation[:, None, :]
    out = np.einsum('nk,nkd,nke->de', probability, delta, delta, optimize=True) / len(x)
    return out + ridge * np.eye(x.shape[-1])


def certificate(result, final, objective_calls, iterations):
    gradient_l2 = float(np.linalg.norm(final['gradient']))
    gap = gradient_l2 ** 2 / (2 * LAMBDA)
    passed = bool(result.success and np.isfinite(gap) and gap <= GAP_MAX
                  and objective_calls <= MAX_CALLS and iterations <= MAX_ITER)
    return dict(PASS=passed, optimizer_success=bool(result.success),
                gradient_l2=gradient_l2, gradient_l2_squared_over_2lambda=gap,
                max_gap_upper_bound=GAP_MAX, lambda_l2=LAMBDA,
                objective_value=final['value'], CE=final['CE'], L2_penalty=final['penalty'],
                unadjusted_CE=final['unadjusted_CE'], loss_rule=LOSS_RULE,
                CE_definition='Margin-adjusted TRAIN CE; unadjusted_CE is diagnostic only.',
                objective_calls=objective_calls, iterations=iterations,
                theorem='For lambda-strongly-convex differentiable J, J(w)-min J <= ||grad J(w)||^2/(2*lambda).',
                scope='Numerical certificate for full-TRAIN fixed-log1p-margin CE+ridge; no T/R performance guarantee.')


def load_anchor(parent):
    """Read frozen TRAIN-only anchor errors and their input-only R0 indices."""
    OLD.install_training_guard()
    bindings = {key: parent['protocol']['inputs'][key] for key in ('features', 'train_labels')}
    for binding in bindings.values():
        C.verify(binding)
    with np.load(C.ROOT / bindings['features']['path'], allow_pickle=False) as feature:
        index = feature['R0_GEO_index'][parent['source_index']]
        np.testing.assert_array_equal(feature['ids'][parent['source_index']], parent['ids'])
        np.testing.assert_array_equal(feature['R0_valid'][parent['source_index']], parent['valid']['R0'])
    with np.load(C.ROOT / bindings['train_labels']['path'], allow_pickle=False) as labels:
        np.testing.assert_array_equal(labels['ids'], parent['ids'])
        errors = np.stack([labels['R0_GEO_T_cm'], labels['R0_GEO_R_deg']], axis=-1)
    assert index.dtype == np.int64 and index.shape == (2598,)
    assert errors.dtype == np.float64 and errors.shape == (2598, 2)
    finite = np.isfinite(errors).all(1)
    assert np.array_equal(finite, np.isfinite(errors).any(1))
    assert np.isposinf(errors[~finite]).all() and (errors >= 0).all()
    assert np.isin(index, [-1, 0, 1]).all() and np.array_equal(index >= 0, finite)
    rows = np.flatnonzero(finite)
    assert parent['valid']['R0'][rows, index[rows]].all()
    np.testing.assert_array_equal(parent['errors']['R0'][rows, index[rows]], errors[rows])
    # The current locked pool has one all-model failure; do not silently invent
    # a target rule for an operational anchor absent from the scored candidates.
    any_candidate = np.logical_or.reduce([parent['valid'][model].any(1) for model in C.U.MODELS])
    assert np.array_equal(any_candidate, finite), 'ANCHOR_CANDIDATE_SUPPORT_CONTRACT'
    return dict(errors=errors, index=index, finite=finite,
                errors_sha=OLD.array_sha(errors), index_sha=OLD.array_sha(index),
                finite_sha=OLD.array_sha(finite), bindings=bindings)


def anchored_targets(errors, valid, anchor_errors, anchor_index, scale, names):
    """Safe mask affects targets only; return original-index whole-pose labels.

    Every present row must have a finite valid R0 anchor in candidate slots0/1.
    Missing anchor support is a contract failure, not a resampling decision.
    """
    errors, valid = np.asarray(errors, np.float64), np.asarray(valid)
    anchor_errors, anchor_index = np.asarray(anchor_errors, np.float64), np.asarray(anchor_index)
    assert valid.dtype == bool and errors.shape == (*valid.shape, 2) and errors.ndim == 3
    assert valid.shape[1] in (2, 4) and len(names) == valid.shape[1]
    assert names[:2] == OLD.candidate_names('R0_ONLY', 1)
    assert anchor_errors.shape == (len(errors), 2) and anchor_index.shape == (len(errors),)
    assert np.issubdtype(anchor_index.dtype, np.integer) and np.isin(anchor_index, [-1, 0, 1]).all()
    assert np.isfinite(errors[valid]).all() and (errors[valid] >= 0).all()
    assert np.isposinf(errors[~valid]).all()
    finite = np.isfinite(anchor_errors).all(1)
    assert np.array_equal(finite, np.isfinite(anchor_errors).any(1))
    assert np.isposinf(anchor_errors[~finite]).all() and (anchor_errors >= 0).all()
    assert np.array_equal(anchor_index >= 0, finite)
    assert np.array_equal(finite, valid.any(1)), 'ANCHOR_CANDIDATE_SUPPORT_CONTRACT'
    rows = np.flatnonzero(finite)
    assert valid[rows, anchor_index[rows]].all()
    np.testing.assert_array_equal(errors[rows, anchor_index[rows]], anchor_errors[rows])
    safe = valid & np.all(errors <= anchor_errors[:, None, :], axis=2)
    assert safe[rows, anchor_index[rows]].all()
    selected_errors = np.where(safe[:, :, None], errors, np.inf)
    target = OLD.targets(selected_errors, safe, scale, names)
    assert np.array_equal(target >= 0, valid.any(1))
    assert safe[rows, target[rows]].all()
    return target, safe


def load_inputs():
    OLD.install_training_guard()
    parent = OLD.load_training_inputs()
    protocol = C.protocol('TRAIN_PROTOCOL')
    expected = dict(feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
                    feature_map=FEATURE_MAP, train_rows=2598, lambda_l2=LAMBDA, bias=0.,
                    normalization='old_float32_then_float64', max_fits=4, target_rule=TARGET_RULE,
                    loss_rule=LOSS_RULE, runtime_uses_margin=False)
    for key, value in expected.items():
        assert protocol[key] == value, (key, protocol.get(key))
    assert protocol['solver'] == dict(method='L-BFGS-B', maxiter=MAX_ITER, maxfun=MAX_CALLS,
                                     ftol=1e-15, gtol=1e-8, bounds=None)
    assert protocol['certificate'] == dict(optimizer_success=True, gap_upper_bound_max=GAP_MAX)
    assert protocol['models'] == list(C.MODEL_NAMES)
    prefit_path = C.DOC / 'PREFIT_REVIEW.json'
    assert protocol['inputs']['prefit_review'] == C.bind(prefit_path)
    prefit = C.read(prefit_path)
    assert prefit['complete'] and prefit['PASS'], 'PREFIT_REVIEW_MUST_PASS'
    assert prefit['loss_rule'] == LOSS_RULE and prefit['runtime_uses_margin'] is False
    assert set(prefit['models']) == set(C.MODEL_NAMES)
    bound = {b['path']: b for b in protocol['inputs'].values()}
    for binding in [parent['binding']] + list(parent['protocol']['inputs'].values()):
        assert bound.get(binding['path']) == binding, ('UNBOUND_PARENT_TRAIN_INPUT', binding['path'])
    code = {b['path']: b for b in protocol['codes']}
    for path in (Path(__file__), Path(C.__file__), Path(OLD.__file__), Path(PREV.__file__),
                 Path(ANCHOR.__file__), Path(CONTEXT.__file__)):
        assert str(path.resolve().relative_to(C.ROOT)) in code
    feasibility_path = C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json'
    assert protocol['inputs']['source_feasibility'] == C.bind(feasibility_path)
    feasibility = C.read(feasibility_path)
    assert feasibility['complete'] and feasibility['PASS'] and feasibility['source_TRAIN_only']
    assert feasibility['frames'] == 2598 and set(feasibility['models']) == set(C.MODEL_NAMES)
    assert not feasibility['VAL_quality_read'] and not feasibility['real_targets_read']
    assert feasibility['protocol'] == protocol['inputs']['source_protocol']
    assert feasibility['protocol'] == C.bind(C.ANCHOR_DOC / 'SOURCE_PROTOCOL.json')
    C.verify(feasibility['protocol'])
    anchor = load_anchor(parent)
    return dict(_authorized=_AUTHORIZED, parent=parent, anchor=anchor, feasibility=feasibility, prefit=prefit,
                protocol=protocol, binding=C.bind(PROTOCOL))


def prepared(model, data):
    parent = data['parent']
    names = candidate_names(model)
    models = ['R0'] + ([f'DIVERSE251_s{model[-1]}'] if model != 'R0_ONLY' else [])
    raw = np.concatenate([parent['features'][m] for m in models], axis=1)
    valid = np.concatenate([parent['valid'][m] for m in models], axis=1)
    errors = np.concatenate([parent['errors'][m] for m in models], axis=1)
    target, safe = anchored_targets(errors, valid, data['anchor']['errors'], data['anchor']['index'], parent['scale'], names)
    locked = data['feasibility']['models'][model]
    for key, value in dict(target_sha=OLD.array_sha(target), safe_mask_sha=OLD.array_sha(safe),
                           anchor_errors_sha=data['anchor']['errors_sha'], anchor_index_sha=data['anchor']['index_sha']).items():
        assert locked[key] == value, ('SOURCE_TARGET_AUDIT_MISMATCH', model, key)
    x = context_inputs(raw, valid, data['anchor']['index'], parent['mean'], parent['std'])
    assert x.shape == (2598, len(names), FEATURE_DIM)
    risk, margin = margin_arrays(errors, valid, data['anchor']['errors'], data['anchor']['index'], parent['scale'], target)
    locked_margin = data['prefit']['models'][model]
    for key, array in (('risk_sha', risk), ('margin_sha', margin), ('target_sha', target),
                       ('original_valid_sha', valid), ('context_sha', x)):
        assert locked_margin[key] == OLD.array_sha(array), ('PREFIT_MARGIN_OR_INPUT_MISMATCH', model, key)
    return names, x, valid, target, safe, risk, margin


def paths(model):
    assert model in C.MODEL_NAMES
    return C.RAW / 'fits' / model, C.DOC / f'FIT_{model}.json'


def verify_completed(model, data):
    folder, receipt_path = paths(model)
    receipt = C.read(receipt_path)
    assert receipt['complete'] and receipt['model'] == model and receipt['protocol'] == data['binding']
    assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
    for key, path in [('START', folder / 'START.json'), ('trace', folder / 'TRACE.jsonl'), ('checkpoint', folder / 'final.json')]:
        assert receipt[key]['path'] == str(path.relative_to(C.ROOT))
        C.verify(receipt[key])
    ck = C.read(folder / 'final.json')
    assert ck['schema'] == CHECKPOINT_SCHEMA
    assert ck['model'] == model and ck['names'] == candidate_names(model)
    assert ck['protocol'] == data['binding'] and ck['bias'] == 0. and ck['lambda_l2'] == LAMBDA
    assert ck['target_rule'] == receipt['target_rule'] == TARGET_RULE
    assert ck['loss_rule'] == receipt['loss_rule'] == LOSS_RULE
    assert ck['runtime_uses_margin'] is receipt['runtime_uses_margin'] is False
    assert ck['runtime_safe_mask'] is False and ck['loss_uses_original_valid_mask'] is True
    for key, val in dict(feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM).items():
        assert ck[key] == receipt[key] == val
    assert ck['normalization'] == 'old_float32_then_float64'
    assert ck['anchor_index_sha'] == receipt['anchor_index_sha'] == data['anchor']['index_sha']
    assert receipt['source_feasibility'] == data['protocol']['inputs']['source_feasibility']
    start = C.read(folder / 'START.json')
    assert start['protocol'] == data['binding'] and start['model'] == model and start['target_rule'] == TARGET_RULE
    assert start['source_feasibility'] == receipt['source_feasibility']
    assert start['loss_rule'] == LOSS_RULE and start['runtime_uses_margin'] is False
    for key in ('risk_sha', 'margin_sha'):
        assert ck[key] == start[key] == receipt[key] == data['prefit']['models'][model][key]
    for key in ('feature_map', 'feature_dim', 'raw_feature_dim'):
        assert start[key] == receipt[key]
    for key in ('target_sha', 'safe_mask_sha'):
        assert start[key] == receipt[key] == data['feasibility']['models'][model][key]
    assert start['anchor_errors_sha'] == data['anchor']['errors_sha']
    assert start['anchor_index_sha'] == data['anchor']['index_sha']
    assert ck['certificate'] == receipt['certificate'] and ck['certificate']['PASS']
    assert ck['certificate']['optimizer_success'] and ck['certificate']['gradient_l2_squared_over_2lambda'] <= GAP_MAX
    assert ck['certificate']['objective_calls'] <= MAX_CALLS and ck['certificate']['iterations'] <= MAX_ITER
    assert receipt['normalization_sha'] == ck['normalization_sha'] == data['parent']['normalization_sha']
    np.testing.assert_array_equal(np.asarray(ck['mean'], np.float32), data['parent']['mean'])
    np.testing.assert_array_equal(np.asarray(ck['std'], np.float32), data['parent']['std'])
    weight = np.asarray(ck['weight'], np.float64)
    assert weight.shape == (FEATURE_DIM,) and np.isfinite(weight).all()
    assert OLD.array_sha(weight) == receipt['final_weight_sha']
    trace = [json.loads(line) for line in (folder / 'TRACE.jsonl').read_text().splitlines()]
    calls = [row for row in trace if row['event'] == 'objective']
    assert [row['call'] for row in calls] == list(range(1, receipt['objective_calls'] + 1))
    assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
    assert calls[-1]['objective'] == ck['certificate']['objective_value']
    for row in trace:
        assert row['feature_map'] == FEATURE_MAP and row['anchor_index_sha'] == data['anchor']['index_sha']
        assert row['loss_rule'] == LOSS_RULE and row['margin_sha'] == receipt['margin_sha']
    assert calls[-1]['unadjusted_CE'] == ck['certificate']['unadjusted_CE']
    assert ck['certificate']['loss_rule'] == LOSS_RULE
    return receipt


def fit_one(model, data):
    OLD.install_training_guard()
    assert data.get('_authorized') is _AUTHORIZED
    assert model in C.MODEL_NAMES
    C.verify(data['binding'])
    for b in data['protocol']['codes'] + list(data['protocol']['inputs'].values()):
        C.verify(b)
    folder, receipt_path = paths(model)
    if receipt_path.exists():
        return verify_completed(model, data)
    if folder.exists():
        assert not any(folder.iterdir()), ('PARTIAL_FIT_STOP_NO_RETRY', str(folder))
    fit_root = C.RAW / 'fits'
    if fit_root.exists():
        assert all(p.is_dir() and p.name in C.MODEL_NAMES for p in fit_root.iterdir())
    names, x, valid, target, safe, risk, margin = prepared(model, data)
    risk_sha, margin_sha = OLD.array_sha(risk), OLD.array_sha(margin)
    parent = data['parent']
    folder.mkdir(parents=True, exist_ok=True)
    C.save(folder / 'START.json', dict(created_at=C.now(), model=model, protocol=data['binding'],
           initialization='all_zero_float64', initial_weight_sha=OLD.array_sha(np.zeros(FEATURE_DIM, np.float64)),
           feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
           loss_rule=LOSS_RULE, risk_sha=risk_sha, margin_sha=margin_sha, runtime_uses_margin=False,
           bias=0., source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
           names=names, train_rows=len(x), source_ids_sha=OLD.array_sha(parent['ids']),
           normalized_features_sha=OLD.array_sha(x), valid_mask_sha=OLD.array_sha(valid),
           target_sha=OLD.array_sha(target), safe_mask_sha=OLD.array_sha(safe),
           anchor_errors_sha=data['anchor']['errors_sha'], anchor_index_sha=data['anchor']['index_sha'],
           source_feasibility=data['protocol']['inputs']['source_feasibility'], target_rule=TARGET_RULE,
           normalization_sha=parent['normalization_sha'],
           lambda_l2=LAMBDA, objective='full_frame_mean_logsumexp_negative_scores_plus_TRAIN_log1p_risk_margin_plus_target_score_plus_half_lambda_weight_norm_squared',
           CE_definition='CE is margin-adjusted; unadjusted_CE is diagnostic only.',
           solver=data['protocol']['solver'], certificate_rule=data['protocol']['certificate'],
           cpu_threads=1, torch_version=str(torch.__version__), numpy_version=np.__version__, scipy_version=scipy.__version__,
           no_valid_rows=int((~valid.any(1)).sum()), one_valid_rows=int((valid.sum(1) == 1).sum()),
           distinction='Only the TRAIN log1p anchor-excess margin changes relative to context189. Features, anchored targets, candidate masks, normalization, solver and runtime remain unchanged; no runtime margin or GT-safe filtering.'))
    started = time.monotonic()
    calls, iterations, last = 0, 0, None
    try:
        with (folder / 'TRACE.jsonl').open('x') as trace:
            def closure(weight):
                nonlocal calls, last
                if calls >= MAX_CALLS:
                    raise ClosureBudgetExceeded('Hard stop before objective closure 2001; no extension or restart.')
                calls += 1
                evaluated = objective(weight, x, valid, target, margin)
                last = dict(evaluated, weight=np.asarray(weight, np.float64).copy())
                row = dict(event='objective', call=calls, objective=evaluated['value'], CE=evaluated['CE'],
                           unadjusted_CE=evaluated['unadjusted_CE'], loss_rule=LOSS_RULE, margin_sha=margin_sha,
                           L2_penalty=evaluated['penalty'], gradient_l2=float(np.linalg.norm(evaluated['gradient'])),
                           feature_map=FEATURE_MAP, anchor_index_sha=data['anchor']['index_sha'],
                           weight_sha=OLD.array_sha(weight))
                trace.write(json.dumps(row, allow_nan=False) + '\n')
                trace.flush()
                return evaluated['value'], evaluated['gradient']

            def callback(weight):
                nonlocal iterations
                iterations += 1
                assert iterations <= MAX_ITER
                assert last is not None and np.array_equal(weight, last['weight'])
                trace.write(json.dumps(dict(event='iteration', iteration=iterations, objective_calls=calls,
                    loss_rule=LOSS_RULE, margin_sha=margin_sha,
                    objective=last['value'], gradient_l2=float(np.linalg.norm(last['gradient'])),
                    feature_map=FEATURE_MAP, anchor_index_sha=data['anchor']['index_sha'],
                    weight_sha=OLD.array_sha(weight)), allow_nan=False) + '\n')
                trace.flush()
                if iterations % 25 == 0:
                    print('ANCHOR_RISK_CONVEX', model, iterations, calls, last['value'], flush=True)

            result = minimize(closure, np.zeros(FEATURE_DIM, np.float64), jac=True, method='L-BFGS-B', bounds=None,
                              callback=callback, options=dict(maxiter=MAX_ITER, maxfun=MAX_CALLS, ftol=1e-15, gtol=1e-8))
        assert last is not None and np.array_equal(result.x, last['weight'])
        assert result.nfev == calls and result.nit == iterations
        np.testing.assert_array_equal(result.jac, last['gradient'])
        assert float(result.fun) == last['value']
        cert = certificate(result, last, calls, iterations)
        curvature = hessian(x, last['probability'])
        eig = np.linalg.eigvalsh(curvature)
        assert eig[0] >= LAMBDA - 1e-10 and np.isfinite(eig).all()
        solver = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
                      iterations=int(result.nit), objective_calls=int(result.nfev),
                      final_hessian_min=float(eig[0]), final_hessian_max=float(eig[-1]))
        if not cert['PASS']:
            C.save(C.DOC / f'REJECTED_{model}.json', dict(complete=True, accepted=False, model=model,
                   protocol=data['binding'], certificate=cert, solver=solver,
                   START=C.bind(folder / 'START.json'), trace=C.bind(folder / 'TRACE.jsonl'),
                   final_weight=result.x, final_weight_sha=OLD.array_sha(result.x), automatic_retry=False))
            raise RuntimeError('SOLVER_OR_CERTIFICATE_REJECTED; no extra iterations or new fit authorized.')
        ck = dict(schema=CHECKPOINT_SCHEMA, model=model, names=names, weight=result.x,
                  loss_rule=LOSS_RULE, risk_sha=risk_sha, margin_sha=margin_sha, runtime_uses_margin=False,
                  feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
                  anchor_index_sha=data['anchor']['index_sha'],
                  bias=0., mean=parent['mean'], std=parent['std'], protocol=data['binding'],
                  normalization_sha=parent['normalization_sha'], lambda_l2=LAMBDA,
                  certificate=cert, solver=solver, normalization='old_float32_then_float64',
                  CE_definition='CE is TRAIN margin-adjusted; inference uses original scores without any margin.',
                  target_rule=TARGET_RULE, runtime_safe_mask=False, loss_uses_original_valid_mask=True,
                  anchor_rule='TRAIN R0 GEO, exact T<=anchorT AND R<=anchorR; fixed TRAIN minmax among eligible candidates only for the target.',
                  context_rule='[z94, abs(z-z_R0_GEO)94, is_R0_GEO1]; input-only anchor index, no extra normalization; invalid candidates zeroed.')
        C.save(folder / 'final.json', ck)
        C.save(receipt_path, dict(complete=True, model=model, protocol=data['binding'],
               START=C.bind(folder / 'START.json'), trace=C.bind(folder / 'TRACE.jsonl'),
               checkpoint=C.bind(folder / 'final.json'), certificate=cert,
               normalization_sha=parent['normalization_sha'], final_weight_sha=OLD.array_sha(result.x),
               objective_calls=calls, iterations=iterations, source_TRAIN_only=True,
               loss_rule=LOSS_RULE, risk_sha=risk_sha, margin_sha=margin_sha, runtime_uses_margin=False,
               feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
               anchor_index_sha=data['anchor']['index_sha'],
               target_rule=TARGET_RULE, source_feasibility=data['protocol']['inputs']['source_feasibility'],
               safe_mask_sha=OLD.array_sha(safe), target_sha=OLD.array_sha(target),
               VAL_quality_read=False, real_targets_read=False, fits_executed=1,
               created_at=C.now(), wall_seconds=time.monotonic() - started, read_paths=sorted(set(OLD.READS))))
    except BaseException as exc:
        if not (folder / 'FAILED.json').exists():
            C.save(folder / 'FAILED.json', dict(created_at=C.now(), model=model, protocol=data['binding'],
                   objective_calls=calls, iterations=iterations, error_type=type(exc).__name__,
                   error=str(exc), automatic_retry=False, further_iterations_authorized=False))
        raise
    return verify_completed(model, data)


def complete_all(data):
    receipts = [verify_completed(model, data) for model in C.MODEL_NAMES]
    assert len({r['normalization_sha'] for r in receipts}) == 1
    result = dict(complete=True, models=list(C.MODEL_NAMES), protocol=data['binding'],
                  fit_count=4, fits=[C.bind(paths(model)[1]) for model in C.MODEL_NAMES],
                  all_certified=True, max_objective_calls_per_fit=MAX_CALLS,
                  total_objective_calls=sum(r['objective_calls'] for r in receipts),
                  source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
                  loss_rule=LOSS_RULE, runtime_uses_margin=False,
                  margin_sha_by_model={r['model']: r['margin_sha'] for r in receipts},
                  risk_sha_by_model={r['model']: r['risk_sha'] for r in receipts},
                  feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
                  anchor_index_sha=data['anchor']['index_sha'],
                  target_rule=TARGET_RULE, source_feasibility=data['protocol']['inputs']['source_feasibility'])
    path = C.DOC / 'TRAINING_COMPLETE.json'
    if path.exists():
        assert C.read(path) == result
    else:
        C.save(path, result)
    return result



def selfcheck():
    global margin_arrays
    # All fixtures are invented. Reuse the prior pure test as a control for
    # normalization/context, while independently checking this module below.
    CONTEXT.selfcheck()
    names = candidate_names('UNION_s1')
    errors = np.array([[[2., 2.], [3., 1.], [1., 1.], [2., 5.]],
                       [[np.inf, np.inf], [1., 3.], [4., 4.], [.5, 3.]],
                       [[1., 1.], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
                       [[np.inf, np.inf]] * 4])
    valid = np.isfinite(errors).all(2)
    anchor_errors = np.array([[2., 2.], [1., 3.], [1., 1.], [np.inf, np.inf]])
    anchor_index = np.array([0, 1, 0, -1], np.int64)
    scale = np.array([2., 1.])
    target, safe = anchored_targets(errors, valid, anchor_errors, anchor_index, scale, names)
    np.testing.assert_array_equal(target, [2, 3, 0, -1])
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        risk, margin = margin_arrays(errors, valid, anchor_errors, anchor_index, scale, target)
    np.testing.assert_array_equal(risk, [[0., .5, 0., 3.], [0., 0., 1.5, 0.], [0.] * 4, [0.] * 4])
    np.testing.assert_array_equal(margin, np.log1p(risk))
    assert risk.dtype == margin.dtype == np.float64
    assert np.all(margin[safe] == 0.) and np.all(margin[~valid] == 0.)
    try:
        margin_arrays(errors, valid, anchor_errors, anchor_index, scale, np.array([1, 3, 0, -1]))
    except AssertionError:
        pass
    else:
        raise AssertionError('Unsafe target margin must be rejected')
    try:
        margin_arrays(errors, valid, anchor_errors, np.array([-1, 1, 0, -1]), scale, target)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing finite anchor support must be rejected')
    # Entirely missing data is valid as a retained failure, without inf-inf.
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        empty_risk, empty_margin = margin_arrays(errors[-1:], valid[-1:], anchor_errors[-1:],
                                                anchor_index[-1:], scale, target[-1:])
    assert not empty_risk.any() and not empty_margin.any()

    rng = np.random.default_rng(7521)
    raw = rng.normal(size=(4, 4, 94)).astype(np.float32)
    raw[~valid] = np.nan
    mean = rng.normal(size=94).astype(np.float32)
    std = rng.uniform(.2, 2., size=94).astype(np.float32)
    x = context_inputs(raw, valid, anchor_index, mean, std)
    np.testing.assert_array_equal(x, CONTEXT.context_inputs(raw, valid, anchor_index, mean, std))
    weight = rng.normal(0., .03, FEATURE_DIM)
    zeros = np.zeros(valid.shape, np.float64)
    old = CONTEXT.objective(weight, x, valid, target)
    zero = objective(weight, x, valid, target, zeros)
    for key in ('value', 'CE', 'penalty'):
        assert zero[key] == old[key]
    assert zero['unadjusted_CE'] == old['CE']
    np.testing.assert_array_equal(zero['gradient'], old['gradient'])
    np.testing.assert_array_equal(zero['probability'], old['probability'])
    np.testing.assert_array_equal(hessian(x, zero['probability']), CONTEXT.hessian(x, old['probability']))
    actual = objective(weight, x, valid, target, margin)
    assert actual['CE'] > actual['unadjusted_CE']
    assert actual['unadjusted_CE'] == old['CE']
    np.testing.assert_array_equal(actual['probability'][~valid], 0.)
    np.testing.assert_array_equal(margin, np.log1p(risk))
    # Independent Torch CE/autograd on the full frozen-margin formula.
    wt = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    xt = torch.tensor(x, dtype=torch.float64)
    vv = torch.tensor(valid, dtype=torch.bool)
    mt = torch.tensor(margin, dtype=torch.float64)
    yy = torch.tensor(target, dtype=torch.int64)
    present = vv.any(1)
    scores = torch.einsum('nkd,d->nk', xt, wt)
    logits = (-scores[present] + mt[present]).masked_fill(~vv[present], -torch.inf)
    ce = (torch.logsumexp(logits, dim=1) + scores[present, yy[present]]).sum() / len(x)
    total = ce + .5 * LAMBDA * wt.square().sum()
    gradient, = torch.autograd.grad(total, wt)
    np.testing.assert_allclose(actual['CE'], float(ce.detach()), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual['value'], float(total.detach()), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual['gradient'], gradient.detach().numpy(), atol=1e-12, rtol=1e-12)
    # Exercise every189 weight and Hessian column, including identity1.
    eps = 1e-5
    fd = np.empty(FEATURE_DIM)
    finite_hessian = np.empty((FEATURE_DIM, FEATURE_DIM))
    for j in range(FEATURE_DIM):
        delta = np.zeros(FEATURE_DIM); delta[j] = eps
        plus = objective(weight + delta, x, valid, target, margin)
        minus = objective(weight - delta, x, valid, target, margin)
        fd[j] = (plus['value'] - minus['value']) / (2 * eps)
        finite_hessian[:, j] = (plus['gradient'] - minus['gradient']) / (2 * eps)
    np.testing.assert_allclose(actual['gradient'], fd, atol=2e-9, rtol=2e-6)
    hh = hessian(x, actual['probability'])
    np.testing.assert_allclose(hh, finite_hessian, atol=2e-9, rtol=2e-6)
    np.testing.assert_allclose(hh, hh.T, atol=1e-12, rtol=1e-12)
    assert np.linalg.eigvalsh(hh)[0] >= LAMBDA - 1e-12
    # Sign and full-row denominator: positive TRAIN risk pushes the competing
    # candidate's score up (less likely), without scaling or deleting rows.
    toy_x = np.array([[[0.], [1.]], [[0.], [0.]]])
    toy_valid = np.array([[True, True], [False, False]])
    toy_y = np.array([0, -1])
    toy_m = np.array([[0., np.log(4.)], [0., 0.]])
    toy = objective(np.zeros(1), toy_x, toy_valid, toy_y, toy_m)
    np.testing.assert_allclose(toy['CE'], np.log(5.) / 2, atol=1e-15)
    np.testing.assert_allclose(toy['gradient'], [-.4], atol=1e-15)
    assert toy['unadjusted_CE'] == np.log(2.) / 2
    # If all features are equal, only ridge depends on w. The certificate's
    # gap bound is an equality even with fixed nonzero TRAIN margins.
    same = np.zeros((2, 2, 3))
    ww = np.array([.2, -.3, .4])
    quadratic = objective(ww, same, toy_valid, toy_y, toy_m)
    np.testing.assert_allclose(quadratic['value'] - np.log(5.) / 2,
                               np.dot(quadratic['gradient'], quadratic['gradient']) / (2 * LAMBDA), atol=1e-16)
    # Runtime consumes only frozen189 features, weights, and input-only anchor;
    # its function signature has no margin or reference-error argument.
    checkpoint = dict(schema=CHECKPOINT_SCHEMA, bias=0., lambda_l2=LAMBDA,
                      loss_rule=LOSS_RULE, runtime_uses_margin=False, feature_map=FEATURE_MAP,
                      feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
                      names=names, normalization='old_float32_then_float64', weight=weight.tolist(),
                      mean=mean.tolist(), std=std.tolist())
    expected = CONTEXT.score_candidates(dict(checkpoint, schema=CONTEXT.CHECKPOINT_SCHEMA), raw, valid, anchor_index)
    runtime = score_candidates(checkpoint, raw, valid, anchor_index)
    np.testing.assert_array_equal(runtime, expected)
    np.testing.assert_array_equal(select_candidates(runtime, valid, names), CONTEXT.select_candidates(expected, valid, names))
    original_margin_function = margin_arrays
    def forbidden_margin(*args, **kwargs):
        raise AssertionError('Runtime attempted to compute TRAIN margins')
    try:
        margin_arrays = forbidden_margin
        np.testing.assert_array_equal(score_candidates(checkpoint, raw, valid, anchor_index), runtime)
    finally:
        margin_arrays = original_margin_function
    try:
        score_candidates(dict(checkpoint, runtime_uses_margin=True), raw, valid, anchor_index)
    except AssertionError:
        pass
    else:
        raise AssertionError('A runtime margin contract must be rejected')
    assert OLD.READS is None, 'Selfcheck must not load TRAIN or evaluation artifacts.'
    print('ANCHOR_RISK_SELF_CHECK_PASS: safe finite-pair risk/log1p, invalid inf-inf prevention, target-zero margin, exact zero-margin prior reduction, Torch objective/gradient,189 finite differences/Hessian, strong-convex certificate, and margin-free runtime parity. No data fit or artifact reads.', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'fit', 'all'])
    parser.add_argument('--model', choices=C.MODEL_NAMES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.stage == 'selfcheck':
            selfcheck()
            return
        assert args.model is None if args.stage == 'all' else args.model is not None
        data = load_inputs()
        if args.stage == 'fit':
            fit_one(args.model, data)
        else:
            for model in C.MODEL_NAMES:
                fit_one(model, data)
            complete_all(data)


if __name__ == '__main__':
    main()
