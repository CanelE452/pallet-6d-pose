"""Certified scorer with a fixed nonlinear R0-anchor context feature map.

The explicit CE + lambda/2 * ||w||^2 objective differs from the original
decoupled AdamW decay. Bias is fixed at zero because it cancels in ranking.
The previous anchored TRAIN targets and the CE/ridge objective are unchanged.
Only the fixed input map adds absolute anchor deltas and anchor identity.
The original validity mask remains in CE and GT-free runtime selection.
Imports and ``selfcheck`` read no artifacts. No fit before a sealed PASS gate.
"""
from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_selector_convergence_20261001_v1 import convex_train as PREV
from scripts.research.pallet_pose_pareto_anchor_20261001_v1 import convex_train as ANCHOR
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
CHECKPOINT_SCHEMA = 'pallet_pose_anchor_context_linear189_v1'
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


def objective(weight, x, valid, target, ridge=LAMBDA):
    """Exact full-frame mean CE and analytic gradient; no optimizer state."""
    weight, x = np.asarray(weight, np.float64), np.asarray(x, np.float64)
    valid, target = np.asarray(valid), np.asarray(target)
    assert x.ndim == 3 and x.shape[:2] == valid.shape and x.shape[-1] == len(weight)
    assert valid.dtype == bool and target.shape == (len(x),)
    assert np.issubdtype(target.dtype, np.integer)
    assert np.isfinite(weight).all() and np.isfinite(x).all() and ridge > 0
    present = valid.any(1)
    assert np.array_equal(target >= 0, present)
    assert np.all(valid[np.flatnonzero(present), target[present]])
    logits = -np.einsum('nkd,d->nk', x, weight)
    logits[~valid] = -np.inf
    probability = np.zeros(valid.shape, np.float64)
    ce_rows = np.zeros(len(x), np.float64)
    if present.any():
        normalizer = logsumexp(logits[present], axis=1)
        probability[present] = np.exp(logits[present] - normalizer[:, None])
        ce_rows[present] = normalizer - logits[present, target[present]]
    target_x = np.zeros((len(x), len(weight)), np.float64)
    target_x[present] = x[np.flatnonzero(present), target[present]]
    gradient = (target_x - np.einsum('nk,nkd->nd', probability, x)).mean(0) + ridge * weight
    ce, penalty = float(ce_rows.mean()), float(.5 * ridge * np.dot(weight, weight))
    assert np.isfinite(gradient).all() and np.isfinite(ce + penalty)
    return dict(value=ce + penalty, CE=ce, penalty=penalty,
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
                objective_calls=objective_calls, iterations=iterations,
                theorem='For lambda-strongly-convex differentiable J, J(w)-min J <= ||grad J(w)||^2/(2*lambda).',
                scope='Numerical certificate for the explicit full-TRAIN CE+ridge objective; no T/R performance guarantee.')


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
                    normalization='old_float32_then_float64', max_fits=4, target_rule=TARGET_RULE)
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
    bound = {b['path']: b for b in protocol['inputs'].values()}
    for binding in [parent['binding']] + list(parent['protocol']['inputs'].values()):
        assert bound.get(binding['path']) == binding, ('UNBOUND_PARENT_TRAIN_INPUT', binding['path'])
    code = {b['path']: b for b in protocol['codes']}
    for path in (Path(__file__), Path(C.__file__), Path(OLD.__file__), Path(PREV.__file__),
                 Path(ANCHOR.__file__)):
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
    return dict(_authorized=_AUTHORIZED, parent=parent, anchor=anchor, feasibility=feasibility,
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
    return names, x, valid, target, safe


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
    assert ck['runtime_safe_mask'] is False and ck['loss_uses_original_valid_mask'] is True
    for key, val in dict(feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM).items():
        assert ck[key] == receipt[key] == val
    assert ck['normalization'] == 'old_float32_then_float64'
    assert ck['anchor_index_sha'] == receipt['anchor_index_sha'] == data['anchor']['index_sha']
    assert receipt['source_feasibility'] == data['protocol']['inputs']['source_feasibility']
    start = C.read(folder / 'START.json')
    assert start['protocol'] == data['binding'] and start['model'] == model and start['target_rule'] == TARGET_RULE
    assert start['source_feasibility'] == receipt['source_feasibility']
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
    names, x, valid, target, safe = prepared(model, data)
    parent = data['parent']
    folder.mkdir(parents=True, exist_ok=True)
    C.save(folder / 'START.json', dict(created_at=C.now(), model=model, protocol=data['binding'],
           initialization='all_zero_float64', initial_weight_sha=OLD.array_sha(np.zeros(FEATURE_DIM, np.float64)),
           feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
           bias=0., source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
           names=names, train_rows=len(x), source_ids_sha=OLD.array_sha(parent['ids']),
           normalized_features_sha=OLD.array_sha(x), valid_mask_sha=OLD.array_sha(valid),
           target_sha=OLD.array_sha(target), safe_mask_sha=OLD.array_sha(safe),
           anchor_errors_sha=data['anchor']['errors_sha'], anchor_index_sha=data['anchor']['index_sha'],
           source_feasibility=data['protocol']['inputs']['source_feasibility'], target_rule=TARGET_RULE,
           normalization_sha=parent['normalization_sha'],
           lambda_l2=LAMBDA, objective='full_frame_mean_CE_negative_scores_plus_half_lambda_weight_norm_squared',
           solver=data['protocol']['solver'], certificate_rule=data['protocol']['certificate'],
           cpu_threads=1, torch_version=str(torch.__version__), numpy_version=np.__version__, scipy_version=scipy.__version__,
           no_valid_rows=int((~valid.any(1)).sum()), one_valid_rows=int((valid.sum(1) == 1).sum()),
           distinction='Only the fixed 189-dimensional input map changes relative to the anchored-target control. Original targets, candidate masks, normalization and solver remain; no runtime anchor-error or GT-safe filtering.'))
    started = time.monotonic()
    calls, iterations, last = 0, 0, None
    try:
        with (folder / 'TRACE.jsonl').open('x') as trace:
            def closure(weight):
                nonlocal calls, last
                if calls >= MAX_CALLS:
                    raise ClosureBudgetExceeded('Hard stop before objective closure 2001; no extension or restart.')
                calls += 1
                evaluated = objective(weight, x, valid, target)
                last = dict(evaluated, weight=np.asarray(weight, np.float64).copy())
                row = dict(event='objective', call=calls, objective=evaluated['value'], CE=evaluated['CE'],
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
                    objective=last['value'], gradient_l2=float(np.linalg.norm(last['gradient'])),
                    feature_map=FEATURE_MAP, anchor_index_sha=data['anchor']['index_sha'],
                    weight_sha=OLD.array_sha(weight)), allow_nan=False) + '\n')
                trace.flush()
                if iterations % 25 == 0:
                    print('ANCHOR_CONTEXT_CONVEX', model, iterations, calls, last['value'], flush=True)

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
                  feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
                  anchor_index_sha=data['anchor']['index_sha'],
                  bias=0., mean=parent['mean'], std=parent['std'], protocol=data['binding'],
                  normalization_sha=parent['normalization_sha'], lambda_l2=LAMBDA,
                  certificate=cert, solver=solver, normalization='old_float32_then_float64',
                  target_rule=TARGET_RULE, runtime_safe_mask=False, loss_uses_original_valid_mask=True,
                  anchor_rule='TRAIN R0 GEO, exact T<=anchorT AND R<=anchorR; fixed TRAIN minmax among eligible candidates only for the target.',
                  context_rule='[z94, abs(z-z_R0_GEO)94, is_R0_GEO1]; input-only anchor index, no extra normalization; invalid candidates zeroed.')
        C.save(folder / 'final.json', ck)
        C.save(receipt_path, dict(complete=True, model=model, protocol=data['binding'],
               START=C.bind(folder / 'START.json'), trace=C.bind(folder / 'TRACE.jsonl'),
               checkpoint=C.bind(folder / 'final.json'), certificate=cert,
               normalization_sha=parent['normalization_sha'], final_weight_sha=OLD.array_sha(result.x),
               objective_calls=calls, iterations=iterations, source_TRAIN_only=True,
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
    # The cost-best candidate can trade lower R for worse T than the anchor.
    # Anchoring changes its target, but it remains a full CE competitor.
    names = candidate_names('UNION_s1')
    errors = np.array([[[1., 3.], [2., 2.], [1., 2.], [1., 2.]],
                       [[2., 3.], [1., 4.], [3., 1.], [4., 4.]],
                       [[1., 1.], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
                       [[np.inf, np.inf]] * 4])
    valid_anchor = np.isfinite(errors).all(2)
    anchor_errors = np.array([[1., 3.], [2., 3.], [1., 1.], [np.inf, np.inf]])
    anchor_index = np.array([0, 0, 0, -1], np.int64)
    target_anchor, safe = anchored_targets(errors, valid_anchor, anchor_errors, anchor_index, [1., 1.], names)
    np.testing.assert_array_equal(target_anchor, [2, 0, 0, -1])
    np.testing.assert_array_equal(safe[1], [True, False, False, False])
    np.testing.assert_array_equal(safe[3], [False] * 4)
    assert safe.dtype == bool and target_anchor.dtype == np.int64
    fixture_x = np.zeros((4, 4, 2))
    fixture_x[0, 1] = [1., 0.]
    fixture_x[1, 1] = [0., 1.]
    unrestricted = objective(np.zeros(2), fixture_x, valid_anchor, target_anchor)
    wrongly_masked = objective(np.zeros(2), fixture_x, safe, target_anchor)
    assert unrestricted['CE'] > wrongly_masked['CE']
    assert not np.array_equal(unrestricted['gradient'], wrongly_masked['gradient'])
    # Shared inference keeps the unsafe candidate eligible; there is no claim
    # that this learned rule can enforce reference nonregression at runtime.
    picked = select_candidates(np.array([[1., -1., 2., 3.]]), valid_anchor[:1], names)
    assert picked[0] == 1 and not safe[0, picked[0]]
    try:
        anchored_targets(errors[:1], valid_anchor[:1], [[np.inf, np.inf]], np.array([-1]), [1., 1.], names)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing operational anchor with available candidates must stop')
    rng = np.random.default_rng(37)
    x = rng.normal(size=(5, 3, 4))
    valid = np.array([[True, True, False], [False, False, False], [False, True, False],
                      [True, True, True], [True, False, True]])
    target = np.array([1, -1, 1, 0, 2], np.int64)
    w = rng.normal(size=4)
    value = objective(w, x, valid, target)
    original = PREV.objective(w, x, valid, target)
    for key in ('value', 'CE', 'penalty'):
        assert value[key] == original[key]
    np.testing.assert_array_equal(value['gradient'], original['gradient'])
    np.testing.assert_array_equal(value['probability'], original['probability'])
    eps = 1e-5
    finite_gradient = []
    finite_hessian = []
    for j in range(4):
        delta = np.zeros(4)
        delta[j] = eps
        plus, minus = objective(w + delta, x, valid, target), objective(w - delta, x, valid, target)
        finite_gradient.append((plus['value'] - minus['value']) / (2 * eps))
        finite_hessian.append((plus['gradient'] - minus['gradient']) / (2 * eps))
    np.testing.assert_allclose(value['gradient'], finite_gradient, atol=2e-10, rtol=1e-7)
    H = hessian(x, value['probability'])
    np.testing.assert_array_equal(H, PREV.hessian(x, original['probability']))
    np.testing.assert_allclose(H, np.asarray(finite_hessian).T, atol=2e-10, rtol=1e-7)
    assert np.linalg.eigvalsh(H)[0] >= LAMBDA
    np.testing.assert_array_equal(value['probability'][~valid], np.zeros((~valid).sum()))
    # With identical candidate features, CE is constant; ridge is exactly a
    # quadratic, making the gradient gap certificate an equality, not a mirror
    # of an arbitrary implementation threshold.
    same = np.zeros((3, 2, 4))
    v = np.array([[True, True], [False, False], [True, False]])
    y = np.array([0, -1, 0])
    q = objective(w, same, v, y)
    best = math.log(2) / 3
    np.testing.assert_allclose(q['value'] - best, np.dot(q['gradient'], q['gradient']) / (2 * LAMBDA), atol=1e-16)
    assert objective(np.zeros(4), same, v, y)['value'] == best
    # Operational FP32 normalization is retained before the fixed FP64 map.
    raw = rng.normal(size=(5, 4, RAW_FEATURE_DIM)).astype(np.float32)
    mean = rng.normal(size=94).astype(np.float32)
    std = rng.uniform(.1, 2, size=94).astype(np.float32)
    mask = np.array([[True] * 4, [False, True, True, False], [False] * 4,
                     [True, False, False, False], [True] * 4])
    anchor = np.array([0, 1, -1, 0, 1], np.int64)
    raw[~mask] = np.nan  # Invalid payloads are never normalized or propagated.
    z = normalized_inputs(raw, mask, mean, std)
    np.testing.assert_array_equal(z, ANCHOR.normalized_inputs(raw, mask, mean, std))
    mapped = context_inputs(raw, mask, anchor, mean, std)
    scalar = np.zeros((5, 4, FEATURE_DIM), np.float64)
    for i in range(len(raw)):
        if anchor[i] < 0:
            continue
        for j in range(4):
            if mask[i, j]:
                scalar[i, j, :94] = z[i, j]
                for k in range(94):
                    scalar[i, j, 94 + k] = abs(z[i, j, k] - z[i, anchor[i], k])
                scalar[i, j, -1] = float(j == anchor[i])
    np.testing.assert_array_equal(mapped, scalar)
    np.testing.assert_array_equal(mapped[~mask], 0.)
    present = np.flatnonzero(mask.any(1))
    np.testing.assert_array_equal(mapped[present, anchor[present], 94:188], 0.)
    np.testing.assert_array_equal(mapped[:, :, -1].sum(1), mask.any(1).astype(np.float64))
    weight94 = rng.normal(size=94)
    embedded = np.concatenate([weight94, np.zeros(95)])
    ck = dict(schema=CHECKPOINT_SCHEMA, bias=0., lambda_l2=LAMBDA,
              feature_map=FEATURE_MAP, feature_dim=FEATURE_DIM, raw_feature_dim=RAW_FEATURE_DIM,
              normalization='old_float32_then_float64', names=candidate_names('UNION_s1'),
              weight=embedded.tolist(), mean=mean.tolist(), std=std.tolist())
    score = score_candidates(ck, raw, mask, anchor)
    prior_ck = dict(schema='pallet_pose_convex_linear94_v1', bias=0., lambda_l2=LAMBDA,
                    weight=weight94.tolist(), mean=mean.tolist(), std=std.tolist())
    prior_score = ANCHOR.score_candidates(prior_ck, raw, mask)
    np.testing.assert_allclose(score[mask], prior_score[mask], atol=1e-12, rtol=1e-12)
    assert np.isposinf(score[~mask]).all()
    embedded_target = np.array([2, 2, -1, 0, 3], np.int64)
    old_obj = ANCHOR.objective(weight94, z, mask, embedded_target)
    embedded_obj = objective(embedded, mapped, mask, embedded_target)
    for key in ('value', 'CE', 'penalty'):
        np.testing.assert_allclose(embedded_obj[key], old_obj[key], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(embedded_obj['gradient'][:94], old_obj['gradient'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(embedded_obj['probability'], old_obj['probability'], atol=1e-12, rtol=1e-12)
    # Candidate permutation, including swapping the two R0 slots, requires the
    # corresponding anchor-index remap; the mathematical map is equivariant.
    perm = np.array([1, 0, 3, 2])
    remapped_anchor = np.where(anchor >= 0, np.argsort(perm)[np.maximum(anchor, 0)], -1)
    np.testing.assert_array_equal(context_inputs(raw[:, perm], mask[:, perm], remapped_anchor, mean, std), mapped[:, perm])
    row_perm = np.array([4, 2, 0, 3, 1])
    np.testing.assert_array_equal(context_inputs(raw[row_perm], mask[row_perm], anchor[row_perm], mean, std), mapped[row_perm])
    np.testing.assert_array_equal(score_candidates(ck, raw[row_perm], mask[row_perm], anchor[row_perm]), score[row_perm])
    for bad_anchor in [np.array([-1, 1, -1, 0, 1]), np.array([0, 0, -1, 0, 1]),
                       np.array([0, 1, 0, 0, 1]), np.array([2, 1, -1, 0, 1])]:
        try:
            context_inputs(raw, mask, bad_anchor, mean, std)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid or absent operational anchor must stop')
    try:
        score_candidates(dict(ck, names=ck['names'][::-1]), raw, mask, anchor)
    except AssertionError:
        pass
    else:
        raise AssertionError('R0 hypothesis slots must be checked')
    # Identical raw candidate features but distinct operational anchors can
    # reverse the ranking using absolute deltas; common linear subtraction
    # would cancel and cannot create this fixture's conditional preference.
    toy_raw = np.zeros((2, 2, 94), np.float32)
    toy_raw[:, 1, 0] = 2.
    toy_mask = np.ones((2, 2), bool)
    toy_anchor = np.array([0, 1], np.int64)
    toy_weight = np.zeros(FEATURE_DIM); toy_weight[94] = 1.
    toy_ck = dict(ck, names=candidate_names('R0_ONLY'), mean=np.zeros(94).tolist(),
                  std=np.ones(94).tolist(), weight=toy_weight.tolist())
    toy_score = score_candidates(toy_ck, toy_raw, toy_mask, toy_anchor)
    np.testing.assert_array_equal(toy_score, [[0., 2.], [2., 0.]])
    np.testing.assert_array_equal(select_candidates(toy_score, toy_mask, toy_ck['names']), [0, 1])
    toy_weight[:] = 0.; toy_weight[-1] = -1.
    np.testing.assert_array_equal(score_candidates(dict(toy_ck, weight=toy_weight.tolist()), toy_raw, toy_mask, toy_anchor), [[-1., 0.], [0., -1.]])
    # Finite differences exercise all 189 trainable weights on the actual map,
    # including the identity coordinate and retained all-invalid denominator.
    weight189 = rng.normal(0., .04, FEATURE_DIM)
    full = objective(weight189, mapped, mask, embedded_target)
    fd = np.empty(FEATURE_DIM)
    for j in range(FEATURE_DIM):
        delta = np.zeros(FEATURE_DIM); delta[j] = eps
        fd[j] = (objective(weight189 + delta, mapped, mask, embedded_target)['value'] -
                 objective(weight189 - delta, mapped, mask, embedded_target)['value']) / (2 * eps)
    np.testing.assert_allclose(full['gradient'], fd, atol=2e-9, rtol=2e-6)
    hh = hessian(mapped, full['probability'])
    np.testing.assert_allclose(hh, hh.T, atol=1e-12, rtol=1e-12)
    assert np.linalg.eigvalsh(hh)[0] >= LAMBDA - 1e-12
    choices = select_candidates(np.array([[0., 0., -100., 0.], [np.inf] * 4]),
                                np.array([[True, True, False, True], [False] * 4]), candidate_names('UNION_s1'))
    np.testing.assert_array_equal(choices, [0, -1])
    assert OLD.READS is None, 'Selfcheck must not load source training artifacts.'
    print('ANCHOR_CONTEXT_SELF_CHECK_PASS: unchanged anchored targets/original-mask CE, 94-to-189 embedding score/objective parity, scalar feature-map parity, absolute-delta ranking flip, identity/invalid/permutation checks, missing-anchor stops, 189-coordinate finite-difference gradient, Hessian and strong-convex gap checks. No artifact reads or data fit.', flush=True)


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
