"""Certified convex scorer control; no data fit without a sealed protocol.

The explicit CE + lambda/2 * ||w||^2 objective differs from the original
decoupled AdamW decay. Bias is fixed at zero because it cancels in ranking.
Imports and ``selfcheck`` read no artifacts. Shared scoring is GT-free.
"""
from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
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
PROTOCOL = C.DOC / 'PROTOCOL.json'
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


def score_candidates(checkpoint, raw_features, valid):
    assert checkpoint['schema'] == 'pallet_pose_convex_linear94_v1'
    assert checkpoint['bias'] == 0. and checkpoint['lambda_l2'] == LAMBDA
    weight = np.asarray(checkpoint['weight'], np.float64)
    assert weight.shape == (94,) and np.isfinite(weight).all()
    x = normalized_inputs(raw_features, valid, checkpoint['mean'], checkpoint['std'])
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


def load_inputs():
    OLD.install_training_guard()
    parent = OLD.load_training_inputs()
    protocol = C.protocol()
    expected = dict(feature_dim=94, train_rows=2598, lambda_l2=LAMBDA, bias=0.,
                    normalization='old_float32_then_float64', max_fits=4)
    for key, value in expected.items():
        assert protocol[key] == value, (key, protocol.get(key))
    assert protocol['solver'] == dict(method='L-BFGS-B', maxiter=MAX_ITER, maxfun=MAX_CALLS,
                                     ftol=1e-15, gtol=1e-8, bounds=None)
    assert protocol['certificate'] == dict(optimizer_success=True, gap_upper_bound_max=GAP_MAX)
    bound = {b['path']: b for b in protocol['inputs'].values()}
    for binding in [parent['binding']] + list(parent['protocol']['inputs'].values()):
        assert bound.get(binding['path']) == binding, ('UNBOUND_PARENT_TRAIN_INPUT', binding['path'])
    code = {b['path']: b for b in protocol['codes']}
    for path in (Path(__file__), Path(C.__file__)):
        assert str(path.resolve().relative_to(C.ROOT)) in code
    return dict(_authorized=_AUTHORIZED, parent=parent, protocol=protocol, binding=C.bind(PROTOCOL))


def prepared(model, data):
    parent = data['parent']
    names = candidate_names(model)
    models = ['R0'] + ([f'DIVERSE251_s{model[-1]}'] if model != 'R0_ONLY' else [])
    raw = np.concatenate([parent['features'][m] for m in models], axis=1)
    valid = np.concatenate([parent['valid'][m] for m in models], axis=1)
    errors = np.concatenate([parent['errors'][m] for m in models], axis=1)
    target = OLD.targets(errors, valid, parent['scale'], names)
    x = normalized_inputs(raw, valid, parent['mean'], parent['std'])
    assert x.shape == (2598, len(names), 94)
    return names, x, valid, target


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
    assert ck['schema'] == 'pallet_pose_convex_linear94_v1'
    assert ck['model'] == model and ck['names'] == candidate_names(model)
    assert ck['protocol'] == data['binding'] and ck['bias'] == 0. and ck['lambda_l2'] == LAMBDA
    assert ck['certificate'] == receipt['certificate'] and ck['certificate']['PASS']
    assert ck['certificate']['optimizer_success'] and ck['certificate']['gradient_l2_squared_over_2lambda'] <= GAP_MAX
    assert ck['certificate']['objective_calls'] <= MAX_CALLS and ck['certificate']['iterations'] <= MAX_ITER
    assert receipt['normalization_sha'] == ck['normalization_sha'] == data['parent']['normalization_sha']
    np.testing.assert_array_equal(np.asarray(ck['mean'], np.float32), data['parent']['mean'])
    np.testing.assert_array_equal(np.asarray(ck['std'], np.float32), data['parent']['std'])
    weight = np.asarray(ck['weight'], np.float64)
    assert weight.shape == (94,) and np.isfinite(weight).all()
    assert OLD.array_sha(weight) == receipt['final_weight_sha']
    trace = [json.loads(line) for line in (folder / 'TRACE.jsonl').read_text().splitlines()]
    calls = [row for row in trace if row['event'] == 'objective']
    assert [row['call'] for row in calls] == list(range(1, receipt['objective_calls'] + 1))
    assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
    assert calls[-1]['objective'] == ck['certificate']['objective_value']
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
    names, x, valid, target = prepared(model, data)
    parent = data['parent']
    folder.mkdir(parents=True, exist_ok=True)
    C.save(folder / 'START.json', dict(created_at=C.now(), model=model, protocol=data['binding'],
           initialization='all_zero_float64', initial_weight_sha=OLD.array_sha(np.zeros(94, np.float64)),
           bias=0., source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
           names=names, train_rows=len(x), source_ids_sha=OLD.array_sha(parent['ids']),
           normalized_features_sha=OLD.array_sha(x), valid_mask_sha=OLD.array_sha(valid),
           target_sha=OLD.array_sha(target), normalization_sha=parent['normalization_sha'],
           lambda_l2=LAMBDA, objective='full_frame_mean_CE_negative_scores_plus_half_lambda_weight_norm_squared',
           solver=data['protocol']['solver'], certificate_rule=data['protocol']['certificate'],
           cpu_threads=1, torch_version=str(torch.__version__), numpy_version=np.__version__, scipy_version=scipy.__version__,
           no_valid_rows=int((~valid.any(1)).sum()), one_valid_rows=int((valid.sum(1) == 1).sum()),
           distinction='Explicit ridge differs from original AdamW decoupled decay. Same fixed cost-best target and original float32 normalization.'))
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
                    weight_sha=OLD.array_sha(weight)), allow_nan=False) + '\n')
                trace.flush()
                if iterations % 25 == 0:
                    print('CONVEX_CONTROL', model, iterations, calls, last['value'], flush=True)

            result = minimize(closure, np.zeros(94, np.float64), jac=True, method='L-BFGS-B', bounds=None,
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
        ck = dict(schema='pallet_pose_convex_linear94_v1', model=model, names=names, weight=result.x,
                  bias=0., mean=parent['mean'], std=parent['std'], protocol=data['binding'],
                  normalization_sha=parent['normalization_sha'], lambda_l2=LAMBDA,
                  certificate=cert, solver=solver, normalization='old_float32_then_float64')
        C.save(folder / 'final.json', ck)
        C.save(receipt_path, dict(complete=True, model=model, protocol=data['binding'],
               START=C.bind(folder / 'START.json'), trace=C.bind(folder / 'TRACE.jsonl'),
               checkpoint=C.bind(folder / 'final.json'), certificate=cert,
               normalization_sha=parent['normalization_sha'], final_weight_sha=OLD.array_sha(result.x),
               objective_calls=calls, iterations=iterations, source_TRAIN_only=True,
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
                  source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False)
    path = C.DOC / 'TRAINING_COMPLETE.json'
    if path.exists():
        assert C.read(path) == result
    else:
        C.save(path, result)
    return result


def selfcheck():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(5, 3, 4))
    valid = np.array([[True, True, False], [False, False, False], [False, True, False],
                      [True, True, True], [True, False, True]])
    target = np.array([1, -1, 1, 0, 2], np.int64)
    w = rng.normal(size=4)
    value = objective(w, x, valid, target)
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
    # Operational float32 normalization followed by float64 score is explicit.
    raw = rng.normal(size=(2, 4, 94)).astype(np.float32)
    mean = rng.normal(size=94).astype(np.float32)
    std = rng.uniform(.1, 2, size=94).astype(np.float32)
    mask = np.array([[True, True, False, True], [False] * 4])
    weight = rng.normal(size=94)
    ck = dict(schema='pallet_pose_convex_linear94_v1', bias=0., lambda_l2=LAMBDA,
              weight=weight.tolist(), mean=mean.tolist(), std=std.tolist())
    score = score_candidates(ck, raw, mask)
    expected = ((raw - mean) / std).astype(np.float64) @ weight
    np.testing.assert_allclose(score[mask], expected[mask], atol=1e-13, rtol=1e-13)
    assert np.isposinf(score[~mask]).all()
    choices = select_candidates(np.array([[0., 0., -100., 0.], [np.inf] * 4]), mask, candidate_names('UNION_s1'))
    np.testing.assert_array_equal(choices, [0, -1])
    assert OLD.READS is None, 'Selfcheck must not load source training artifacts.'
    print('CONVEX_SELF_CHECK_PASS: finite-difference gradient/Hessian, strong convexity, exact quadratic gap bound, masks/0-1 candidates, float32-to-float64 scoring/ties. No data fit.', flush=True)


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
