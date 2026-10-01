"""Bounded pairwise supervision of frozen whole-pose candidates.

Imports and synthetic selfcheck read no data. Source TRAIN access is guarded at
execution. The certified R0 binary objective is reused without an optimizer run.
"""
from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_selector_convergence_20261001_v1 import convex_train as PREV
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits

LAMBDA, MAX_CALLS, MAX_ITER, GAP_MAX = 1e-4, 2000, 1000, 1e-6
SCHEMA = 'pallet_pose_pairwise_linear94_v1'
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))
PROTOCOL = C.DOC / 'PROTOCOL.json'
_AUTHORIZED = object()
normalized_inputs = PREV.normalized_inputs
select_candidates = OLD.select_candidates
candidate_names = PREV.candidate_names
ClosureBudgetExceeded = PREV.ClosureBudgetExceeded


def score_candidates(checkpoint, raw_features, valid):
    """GT-free float32 normalization, float64 scores, same whole-pose argmin."""
    assert checkpoint['schema'] == SCHEMA
    assert checkpoint['bias'] == 0. and checkpoint['lambda_l2'] == LAMBDA
    weight = np.asarray(checkpoint['weight'], np.float64)
    assert weight.shape == (94,) and np.isfinite(weight).all()
    x = normalized_inputs(raw_features, valid, checkpoint['mean'], checkpoint['std'])
    score = np.einsum('nkd,d->nk', x, weight)
    assert np.isfinite(score).all()
    return np.where(valid, score, np.inf)


def global_ranks(errors, valid, scale, names):
    """Cost, then successive Pareto fronts, then the original name tie key."""
    errors, valid = np.asarray(errors, np.float64), np.asarray(valid)
    scale = np.asarray(scale, np.float64)
    original = OLD.targets(errors, valid, scale, names)  # validates the contract
    ranks = np.full(valid.shape, -1, np.int64)
    for i, row in enumerate(errors):
        usable = np.flatnonzero(valid[i])
        costs = np.max(row[usable] / scale, axis=1)
        ordered = []
        for cost in np.unique(costs):
            remaining = usable[costs == cost].tolist()
            while remaining:
                front = [j for j in remaining if not any(
                    np.all(row[k] <= row[j]) and np.any(row[k] < row[j])
                    for k in remaining)]
                assert front
                ordered.extend(sorted(front, key=lambda j: OLD.tie_key(names[j])))
                remaining = [j for j in remaining if j not in front]
        if ordered:
            assert ordered[0] == original[i]
            ranks[i, ordered] = np.arange(len(ordered))
        else:
            assert original[i] == -1
    return ranks, original


def edge_design(x, valid, ranks):
    """Fixed coefficients; an unavailable edge contributes exactly zero."""
    x, valid, ranks = np.asarray(x, np.float64), np.asarray(valid), np.asarray(ranks)
    assert x.ndim == 3 and x.shape[:2] == valid.shape == ranks.shape
    assert valid.dtype == bool and np.array_equal(ranks >= 0, valid)
    edges = ((0, 1),) if valid.shape[1] == 2 else EDGES
    assert valid.shape[1] in (2, 4) and np.isfinite(x).all()
    active = np.stack([valid[:, a] & valid[:, b] for a, b in edges], axis=1)
    winner = np.full(active.shape, -1, np.int64)
    delta = np.zeros((*active.shape, x.shape[-1]), np.float64)
    for e, (a, b) in enumerate(edges):
        rows = np.flatnonzero(active[:, e])
        w = np.where(ranks[rows, a] < ranks[rows, b], a, b)
        loser = np.where(w == a, b, a)
        winner[rows, e] = w
        delta[rows, e] = x[rows, w] - x[rows, loser]
    coefficient = np.ones(1) if len(edges) == 1 else np.full(4, .25)
    return dict(delta=delta, active=active, winner=winner, coefficient=coefficient,
                edges=edges)


def objective(weight, design, ridge=LAMBDA):
    weight = np.asarray(weight, np.float64)
    delta, active, coefficient = (design[k] for k in ('delta', 'active', 'coefficient'))
    assert delta.shape[:2] == active.shape and delta.shape[-1] == len(weight)
    assert active.dtype == bool and coefficient.shape == (active.shape[1],)
    assert np.isfinite(weight).all() and np.isfinite(delta).all() and ridge > 0
    z = np.einsum('ned,d->ne', delta, weight)
    probability = expit(z)
    factor = active * coefficient[None, :]
    row_loss = (factor * np.logaddexp(0., z)).sum(1)
    gradient = np.einsum('ne,ned->d', factor * probability, delta) / len(delta) + ridge * weight
    loss, penalty = float(row_loss.mean()), float(.5 * ridge * np.dot(weight, weight))
    assert np.isfinite(gradient).all() and np.isfinite(loss + penalty)
    return dict(value=loss + penalty, pairwise_loss=loss, penalty=penalty,
                gradient=gradient, probability=probability)


def hessian(design, probability, ridge=LAMBDA):
    delta = design['delta']
    factor = design['active'] * design['coefficient'][None, :] * probability * (1 - probability)
    return (np.einsum('ne,ned,nef->df', factor, delta, delta, optimize=True) / len(delta)
            + ridge * np.eye(delta.shape[-1]))


def certificate(success, final, calls, iterations, reused=False):
    gradient_l2 = float(np.linalg.norm(final['gradient']))
    gap = gradient_l2 ** 2 / (2 * LAMBDA)
    return dict(PASS=bool(success and np.isfinite(gap) and gap <= GAP_MAX
                          and 0 <= calls <= MAX_CALLS and 0 <= iterations <= MAX_ITER),
        optimizer_success=bool(success), optimizer_success_source=(
            'verified_origin_R0_ONLY_fit' if reused else 'current_L-BFGS-B_fit'),
        gradient_l2=gradient_l2, gradient_l2_squared_over_2lambda=gap,
        max_gap_upper_bound=GAP_MAX, lambda_l2=LAMBDA, objective_value=final['value'],
        pairwise_loss=final['pairwise_loss'], L2_penalty=final['penalty'],
        objective_calls=calls, iterations=iterations,
        theorem='For lambda-strongly-convex differentiable J, J(w)-min J <= ||grad J(w)||^2/(2*lambda).',
        scope='Full-TRAIN pairwise logistic+ridge objective only; no T/R or global-argmin performance guarantee.')


def load_inputs():
    OLD.install_training_guard()
    parent = OLD.load_training_inputs()
    p = C.protocol()
    expected = dict(feature_dim=94, train_rows=2598, lambda_l2=LAMBDA, bias=0.,
        normalization='old_float32_then_float64', max_fits=3, model_count=4,
        objective='equal_group_pairwise_logistic', new_models=list(C.NEW_MODELS),
        reused_models=['R0_ONLY'], edge_groups={'WD': [[0, 1], [2, 3]], 'expert': [[0, 2], [1, 3]]},
        group_weights={'WD': .5, 'expert': .5}, fixed_edge_denominators=2)
    for key, value in expected.items():
        assert p[key] == value, (key, p.get(key))
    assert p['solver'] == dict(method='L-BFGS-B', maxiter=MAX_ITER, maxfun=MAX_CALLS,
                               ftol=1e-15, gtol=1e-8, bounds=None)
    assert p['certificate'] == dict(optimizer_success=True, gap_upper_bound_max=GAP_MAX)
    bound = {b['path']: b for b in p['inputs'].values()}
    for b in [parent['binding']] + list(parent['protocol']['inputs'].values()):
        assert bound.get(b['path']) == b, ('UNBOUND_PARENT_TRAIN_INPUT', b['path'])
    for path in (Path(__file__), Path(C.__file__), Path(OLD.__file__), Path(PREV.__file__)):
        assert str(path.resolve().relative_to(C.ROOT)) in {b['path'] for b in p['codes']}
    assert p['inputs']['target_audit'] == C.bind(C.DOC / 'TARGET_CONTRACT_AUDIT.json')
    audit = C.read(C.DOC / 'TARGET_CONTRACT_AUDIT.json')
    assert audit['PASS'] and audit['complete']
    return dict(_authorized=_AUTHORIZED, parent=parent, protocol=p, binding=C.bind(PROTOCOL), audit=audit)


def prepared(model, data):
    parent = data['parent']
    names = candidate_names(model)
    models = ['R0'] + ([f'DIVERSE251_s{model[-1]}'] if model != 'R0_ONLY' else [])
    raw = np.concatenate([parent['features'][m] for m in models], axis=1)
    valid = np.concatenate([parent['valid'][m] for m in models], axis=1)
    errors = np.concatenate([parent['errors'][m] for m in models], axis=1)
    x = normalized_inputs(raw, valid, parent['mean'], parent['std'])
    ranks, target = global_ranks(errors, valid, parent['scale'], names)
    design = edge_design(x, valid, ranks)
    assert x.shape == (2598, len(names), 94)
    assert OLD.array_sha(parent['ids']) == data['audit']['training_input_ids_sha']
    if model != 'R0_ONLY':
        audited = data['audit']['seeds'][model[-1]]['hashes']
        for key, value in dict(global_rank=ranks, original_whole_best=target,
                               edge_winners=design['winner'], edge_active=design['active']).items():
            assert OLD.array_sha(value) == audited[key], ('INDEPENDENT_TARGET_AUDIT_MISMATCH', model, key)
    return dict(names=names, x=x, valid=valid, errors=errors, ranks=ranks, target=target, design=design)


def paths(model):
    assert model in C.MODEL_NAMES
    return C.RAW / 'fits' / model, C.DOC / f'FIT_{model}.json'


def origin(data, model='R0_ONLY'):
    path = C.CONV_DOC / f'FIT_{model}.json'
    receipt = C.read(path)
    assert receipt['complete'] and receipt['model'] == model and receipt['certificate']['PASS']
    assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
    for key in ('protocol', 'checkpoint', 'START', 'trace'):
        C.verify(receipt[key])
    ck = C.read(C.ROOT / receipt['checkpoint']['path'])
    assert ck['schema'] == 'pallet_pose_convex_linear94_v1' and ck['model'] == model
    assert ck['names'] == candidate_names(model) and ck['protocol'] == receipt['protocol']
    assert ck['certificate'] == receipt['certificate'] and ck['certificate']['optimizer_success']
    assert ck['certificate']['gradient_l2_squared_over_2lambda'] <= GAP_MAX
    assert ck['bias'] == 0. and ck['lambda_l2'] == LAMBDA
    assert ck['normalization_sha'] == receipt['normalization_sha'] == data['parent']['normalization_sha']
    for key in ('mean', 'std'):
        np.testing.assert_array_equal(np.asarray(ck[key], np.float32), data['parent'][key])
    weight = np.asarray(ck['weight'], np.float64)
    assert weight.shape == (94,) and np.isfinite(weight).all()
    assert OLD.array_sha(weight) == receipt['final_weight_sha']
    if model == 'R0_ONLY':
        for key, binding in dict(R0_origin_receipt=C.bind(path), R0_origin_checkpoint=receipt['checkpoint'],
            convergence_protocol=receipt['protocol'], R0_origin_START=receipt['START'], R0_origin_trace=receipt['trace']).items():
            assert data['protocol']['inputs'][key] == binding
    return ck, receipt, C.bind(path)


def binary_equivalence(weight, prepared_data, old_certificate):
    """Independent torch CE/autograd comparison, no optimizer invocation."""
    q = prepared_data
    assert q['valid'].shape[1] == 2
    parameter = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    x = torch.from_numpy(q['x'])
    score = x @ parameter
    loss = OLD.masked_loss(score, torch.from_numpy(q['valid']), torch.from_numpy(q['target']))
    loss = loss + .5 * LAMBDA * parameter.square().sum()
    loss.backward()
    binary_grad = parameter.grad.detach().numpy()
    value = objective(weight, q['design'])
    diff = abs(float(loss.detach()) - value['value'])
    grad_diff = float(np.max(np.abs(binary_grad - value['gradient'])))
    old_objective_diff = abs(value['value'] - old_certificate['objective_value'])
    old_gradient_norm_diff = abs(float(np.linalg.norm(value['gradient'])) - old_certificate['gradient_l2'])
    passed = max(diff, grad_diff, old_objective_diff, old_gradient_norm_diff) <= 1e-12
    assert passed, ('R0_OBJECTIVE_EQUIVALENCE_FAIL', diff, grad_diff, old_objective_diff, old_gradient_norm_diff)
    return value, dict(PASS=True, objective_abs_diff=diff, gradient_max_abs_diff=grad_diff,
        origin_objective_abs_diff=old_objective_diff, origin_gradient_l2_abs_diff=old_gradient_norm_diff,
        max_abs_tolerance=1e-12, check='Independent float64 torch masked two-class CE+ridge and autograd on all2598 TRAIN rows.',
        optimizer_steps=0, targets_identical=True)


def train_diagnostic(weight, q, scale):
    score = np.einsum('nkd,d->nk', q['x'], weight)
    choice = select_candidates(score, q['valid'], q['names'])
    target, present = q['target'], q['target'] >= 0
    idx = np.flatnonzero(present)
    costs = np.max(q['errors'] / scale, axis=2)
    selected, best = costs[idx, choice[idx]], costs[idx, target[idx]]
    regret = selected - best
    assert np.isfinite(regret).all() and (regret >= -1e-12).all()
    edge = {}
    for e, (a, b) in enumerate(q['design']['edges']):
        active = q['design']['active'][:, e]
        picked = select_candidates(score[:, [a, b]], q['valid'][:, [a, b]], [q['names'][a], q['names'][b]])
        actual = np.asarray([a, b])[picked[active]]
        correct = actual == q['design']['winner'][active, e]
        edge[f'{a}:{b}'] = dict(active=int(active.sum()), correct=int(correct.sum()), accuracy=float(correct.mean()))
    return dict(rows=len(choice), no_valid_rows=int((~present).sum()),
        whole_best_correct=int((choice[present] == target[present]).sum()),
        whole_best_accuracy=float((choice[present] == target[present]).mean()), edge_accuracy=edge,
        cost_defined_rows=int(present.sum()), cost_unavailable_rows=int((~present).sum()),
        mean_selected_cost=float(selected.mean()), mean_best_cost=float(best.mean()),
        mean_regret=float(regret.mean()), regret_P90=float(np.quantile(regret, .9)),
        interpretation='TRAIN-only descriptive costs on available candidates; no-valid rows explicitly counted, not imputed correct. Not a stop criterion.')


def verify_completed(model, data):
    folder, receipt_path = paths(model)
    r = C.read(receipt_path)
    assert r['complete'] and r['model'] == model and r['protocol'] == data['binding']
    assert r['source_TRAIN_only'] and not r['VAL_quality_read'] and not r['real_targets_read']
    reused = model == 'R0_ONLY'
    assert r['reused'] is reused and r['new_fit'] is (not reused)
    for key, path in [('START', folder / 'START.json'), ('trace', folder / 'TRACE.jsonl'), ('checkpoint', folder / 'final.json')]:
        assert r[key]['path'] == str(path.relative_to(C.ROOT))
        C.verify(r[key])
    ck = C.read(folder / 'final.json')
    assert ck['schema'] == SCHEMA and ck['model'] == model and ck['names'] == candidate_names(model)
    assert ck['protocol'] == data['binding'] and ck['bias'] == 0. and ck['lambda_l2'] == LAMBDA
    cert = ck['certificate']
    assert cert == r['certificate'] and cert['PASS'] and cert['optimizer_success']
    assert cert['gradient_l2_squared_over_2lambda'] <= GAP_MAX
    assert r['objective_calls'] == cert['objective_calls'] <= MAX_CALLS
    assert r['iterations'] == cert['iterations'] <= MAX_ITER
    assert r['normalization_sha'] == ck['normalization_sha'] == data['parent']['normalization_sha']
    for key in ('mean', 'std'):
        np.testing.assert_array_equal(np.asarray(ck[key], np.float32), data['parent'][key])
    weight = np.asarray(ck['weight'], np.float64)
    assert weight.shape == (94,) and np.isfinite(weight).all()
    assert OLD.array_sha(weight) == r['final_weight_sha']
    trace = [json.loads(line) for line in (folder / 'TRACE.jsonl').read_text().splitlines()]
    if reused:
        old_ck, old_receipt, old_receipt_binding = origin(data)
        assert r['objective_calls'] == r['iterations'] == 0
        assert cert['optimizer_success_source'] == 'verified_origin_R0_ONLY_fit'
        assert r['origin_checkpoint'] == ck['origin_checkpoint'] == old_receipt['checkpoint']
        assert r['origin_receipt'] == ck['origin_receipt'] == old_receipt_binding
        assert r['origin_protocol'] == ck['origin_protocol'] == old_receipt['protocol']
        assert r['equivalence'] == ck['equivalence'] and r['equivalence']['PASS']
        assert r['equivalence']['max_abs_tolerance'] == 1e-12
        np.testing.assert_array_equal(weight, np.asarray(old_ck['weight'], np.float64))
        for key in ('mean', 'std'):
            np.testing.assert_array_equal(np.asarray(ck[key], np.float32), np.asarray(old_ck[key], np.float32))
        assert len(trace) == 1 and trace[0]['event'] == 'reuse_equivalence'
    else:
        calls = [row for row in trace if row['event'] == 'objective']
        assert 0 < r['objective_calls'] <= MAX_CALLS
        assert [row['call'] for row in calls] == list(range(1, r['objective_calls'] + 1))
        assert calls[-1]['weight_sha'] == r['final_weight_sha']
        assert calls[-1]['objective'] == cert['objective_value']
    return r


def execute_one(model, data):
    OLD.install_training_guard()
    assert data.get('_authorized') is _AUTHORIZED and model in C.MODEL_NAMES
    C.verify(data['binding'])
    for b in data['protocol']['codes'] + list(data['protocol']['inputs'].values()):
        C.verify(b)
    folder, receipt_path = paths(model)
    if receipt_path.exists():
        return verify_completed(model, data)
    if folder.exists():
        assert not any(folder.iterdir()), ('PARTIAL_FIT_STOP_NO_RETRY', str(folder))
    if (C.RAW / 'fits').exists():
        assert all(p.is_dir() and p.name in C.MODEL_NAMES for p in (C.RAW / 'fits').iterdir())
    q, parent = prepared(model, data), data['parent']
    reused = model == 'R0_ONLY'
    folder.mkdir(parents=True, exist_ok=True)
    C.save(folder / 'START.json', dict(created_at=C.now(), model=model, protocol=data['binding'],
        initialization='reuse_certified_R0_parameters' if reused else 'all_zero_float64',
        reused=reused, new_fit=not reused, bias=0., source_TRAIN_only=True, VAL_quality_read=False,
        real_targets_read=False, names=q['names'], train_rows=len(q['x']), source_ids_sha=OLD.array_sha(parent['ids']),
        normalized_features_sha=OLD.array_sha(q['x']), valid_mask_sha=OLD.array_sha(q['valid']),
        target_sha=OLD.array_sha(q['target']), rank_sha=OLD.array_sha(q['ranks']),
        edge_winner_sha=OLD.array_sha(q['design']['winner']), edge_active_sha=OLD.array_sha(q['design']['active']),
        normalization_sha=parent['normalization_sha'], lambda_l2=LAMBDA,
        objective='equal_group_pairwise_logistic_plus_half_lambda_weight_norm_squared',
        edges=q['design']['edges'], edge_coefficients=q['design']['coefficient'],
        solver=data['protocol']['solver'], certificate_rule=data['protocol']['certificate'], cpu_threads=1,
        torch_version=str(torch.__version__), numpy_version=np.__version__, scipy_version=scipy.__version__,
        no_valid_rows=int((~q['valid'].any(1)).sum())))
    started, calls, iterations, last = time.monotonic(), 0, 0, None
    extra = {}
    try:
        with (folder / 'TRACE.jsonl').open('x') as trace:
            if reused:
                old_ck, old_receipt, old_receipt_binding = origin(data)
                weight = np.asarray(old_ck['weight'], np.float64)
                last, equivalence = binary_equivalence(weight, q, old_ck['certificate'])
                cert = certificate(old_ck['certificate']['optimizer_success'], last, 0, 0, True)
                extra = dict(origin_checkpoint=old_receipt['checkpoint'], origin_receipt=old_receipt_binding,
                    origin_protocol=old_receipt['protocol'], equivalence=equivalence)
                trace.write(json.dumps(dict(event='reuse_equivalence', equivalence=equivalence,
                    objective=last['value'], weight_sha=OLD.array_sha(weight), optimizer_steps=0), allow_nan=False) + '\n')
                solver = dict(reused=True, success=True, objective_calls=0, iterations=0,
                    origin_solver=old_ck['solver'], no_new_optimizer_steps=True)
            else:
                def closure(weight):
                    nonlocal calls, last
                    if calls >= MAX_CALLS:
                        raise ClosureBudgetExceeded('Hard stop before objective closure2001; no extension/restart.')
                    calls += 1
                    evaluated = objective(weight, q['design'])
                    last = dict(evaluated, weight=np.asarray(weight, np.float64).copy())
                    trace.write(json.dumps(dict(event='objective', call=calls, objective=evaluated['value'],
                        pairwise_loss=evaluated['pairwise_loss'], L2_penalty=evaluated['penalty'],
                        gradient_l2=float(np.linalg.norm(evaluated['gradient'])), weight_sha=OLD.array_sha(weight)), allow_nan=False) + '\n')
                    trace.flush()
                    return evaluated['value'], evaluated['gradient']

                def callback(weight):
                    nonlocal iterations
                    iterations += 1
                    assert iterations <= MAX_ITER and last is not None and np.array_equal(weight, last['weight'])
                    trace.write(json.dumps(dict(event='iteration', iteration=iterations, objective_calls=calls,
                        objective=last['value'], gradient_l2=float(np.linalg.norm(last['gradient'])),
                        weight_sha=OLD.array_sha(weight)), allow_nan=False) + '\n')
                    trace.flush()
                    if iterations % 25 == 0:
                        print('PAIRWISE', model, iterations, calls, last['value'], flush=True)

                result = minimize(closure, np.zeros(94, np.float64), jac=True, method='L-BFGS-B', bounds=None,
                    callback=callback, options=dict(maxiter=MAX_ITER, maxfun=MAX_CALLS, ftol=1e-15, gtol=1e-8))
                assert last is not None and np.array_equal(result.x, last['weight'])
                assert result.nfev == calls and result.nit == iterations
                np.testing.assert_array_equal(result.jac, last['gradient'])
                assert float(result.fun) == last['value']
                weight = result.x
                cert = certificate(result.success, last, calls, iterations)
                eig = np.linalg.eigvalsh(hessian(q['design'], last['probability']))
                assert eig[0] >= LAMBDA - 1e-10 and np.isfinite(eig).all()
                solver = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
                    iterations=int(result.nit), objective_calls=int(result.nfev),
                    final_hessian_min=float(eig[0]), final_hessian_max=float(eig[-1]))
        if not cert['PASS']:
            C.save(C.DOC / f'REJECTED_{model}.json', dict(complete=True, accepted=False, model=model,
                protocol=data['binding'], certificate=cert, solver=solver, START=C.bind(folder / 'START.json'),
                trace=C.bind(folder / 'TRACE.jsonl'), final_weight=weight, final_weight_sha=OLD.array_sha(weight), automatic_retry=False))
            raise RuntimeError('SOLVER_OR_CERTIFICATE_REJECTED; no extra fit authorized.')
        ck = dict(schema=SCHEMA, model=model, names=q['names'], weight=weight, bias=0.,
            mean=parent['mean'], std=parent['std'], protocol=data['binding'],
            normalization_sha=parent['normalization_sha'], lambda_l2=LAMBDA, certificate=cert,
            solver=solver, normalization='old_float32_then_float64', reused=reused, new_fit=not reused,
            fits_executed=int(not reused), optimizer_steps=iterations, **extra)
        C.save(folder / 'final.json', ck)
        C.save(receipt_path, dict(complete=True, model=model, protocol=data['binding'],
            START=C.bind(folder / 'START.json'), trace=C.bind(folder / 'TRACE.jsonl'), checkpoint=C.bind(folder / 'final.json'),
            certificate=cert, normalization_sha=parent['normalization_sha'], final_weight_sha=OLD.array_sha(weight),
            objective_calls=calls, iterations=iterations, source_TRAIN_only=True, VAL_quality_read=False,
            real_targets_read=False, reused=reused, new_fit=not reused, fits_executed=int(not reused),
            optimizer_steps=iterations,
            TRAIN_diagnostic=train_diagnostic(weight, q, parent['scale']),
            created_at=C.now(), wall_seconds=time.monotonic() - started, read_paths=sorted(set(OLD.READS)), **extra))
    except BaseException as exc:
        if not (folder / 'FAILED.json').exists():
            C.save(folder / 'FAILED.json', dict(created_at=C.now(), model=model, protocol=data['binding'],
                objective_calls=calls, iterations=iterations, error_type=type(exc).__name__, error=str(exc),
                automatic_retry=False, further_iterations_authorized=False))
        raise
    return verify_completed(model, data)


def complete_all(data):
    receipts = [verify_completed(model, data) for model in C.MODEL_NAMES]
    assert len({r['normalization_sha'] for r in receipts}) == 1
    result = dict(complete=True, models=list(C.MODEL_NAMES), protocol=data['binding'], model_count=4,
        fit_count=3, new_fit_count=3, reused_count=1, fits=[C.bind(paths(m)[1]) for m in C.MODEL_NAMES],
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
    rng = np.random.default_rng(51)
    names = candidate_names('UNION_s1')
    errors = np.array([[[.9, 1], [1, .5], [.8, 1], [.7, 1]]])
    valid = np.ones((1, 4), bool)
    ranks, target = global_ranks(errors, valid, [1., 1.], names)
    np.testing.assert_array_equal(np.argsort(ranks[0]), [1, 3, 2, 0])
    assert target[0] == 1
    old_winner = []
    for a, b in EDGES:
        local = OLD.targets(errors[:, [a, b]], valid[:, [a, b]], [1., 1.], [names[a], names[b]])[0]
        old_winner.append((a, b)[local])
    assert old_winner == [0, 3, 2, 1]  # 0->1->3->2->0: old local cycle
    x = rng.normal(size=(5, 4, 4))
    valid = np.array([[True] * 4, [False] * 4, [True, False, False, False],
                      [True, True, False, False], [True, False, True, True]])
    errors = rng.uniform(.01, 3, size=(5, 4, 2))
    errors[~valid] = np.inf
    ranks, target = global_ranks(errors, valid, [1., 1.], names)
    d = edge_design(x, valid, ranks)
    w = rng.normal(size=4)
    value, eps = objective(w, d), 1e-5
    finite_gradient, finite_hessian = [], []
    for j in range(4):
        delta = np.zeros(4)
        delta[j] = eps
        plus, minus = objective(w + delta, d), objective(w - delta, d)
        finite_gradient.append((plus['value'] - minus['value']) / (2 * eps))
        finite_hessian.append((plus['gradient'] - minus['gradient']) / (2 * eps))
    np.testing.assert_allclose(value['gradient'], finite_gradient, atol=2e-10, rtol=1e-7)
    H = hessian(d, value['probability'])
    np.testing.assert_allclose(H, np.asarray(finite_hessian).T, atol=2e-10, rtol=1e-7)
    assert np.linalg.eigvalsh(H)[0] >= LAMBDA
    expected_zero_loss = d['active'].sum() * .25 * math.log(2) / len(x)
    np.testing.assert_allclose(objective(np.zeros(4), d)['value'], expected_zero_loss, atol=1e-16)
    same = edge_design(np.zeros_like(x), valid, ranks)
    quad = objective(w, same)
    np.testing.assert_allclose(quad['value'] - expected_zero_loss,
        np.dot(quad['gradient'], quad['gradient']) / (2 * LAMBDA), atol=1e-16)
    # Two-candidate masks exercise two/one/zero valid rows independently.
    xb, vb, eb = x[:, :2], valid[:, :2], errors[:, :2]
    rb, tb = global_ranks(eb, vb, [1., 1.], candidate_names('R0_ONLY'))
    pair = objective(w, edge_design(xb, vb, rb))
    ce = PREV.objective(w, xb, vb, tb)
    np.testing.assert_allclose(pair['value'], ce['value'], atol=1e-15)
    np.testing.assert_allclose(pair['gradient'], ce['gradient'], atol=1e-15)
    raw = rng.normal(size=(2, 4, 94)).astype(np.float32)
    mean, std = np.zeros(94, np.float32), np.ones(94, np.float32)
    std[0] = np.float32(1e-6)
    mask = np.array([[True, True, False, True], [False] * 4])
    ck = dict(schema=SCHEMA, bias=0., lambda_l2=LAMBDA, weight=rng.normal(size=94), mean=mean, std=std)
    score = score_candidates(ck, raw, mask)
    expected = ((raw - mean) / std).astype(np.float64) @ ck['weight']
    np.testing.assert_allclose(score[mask], expected[mask], atol=1e-9, rtol=1e-13)
    assert np.isposinf(score[~mask]).all()
    np.testing.assert_array_equal(select_candidates(np.array([[0., 0., -100., 0.], [np.inf] * 4]), mask, names), [0, -1])
    assert OLD.READS is None, 'Synthetic checks must not load TRAIN artifacts.'
    print('PAIRWISE_SELF_CHECK_PASS: global tie-cycle correction/top parity, analytic gradient/Hessian, strong convexity/gap, fixed edge denominators, binary CE equivalence, float32 normalization/scoring. No data fit.', flush=True)


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
            execute_one(args.model, data)
        else:
            for model in C.MODEL_NAMES:
                execute_one(model, data)
            complete_all(data)


if __name__ == '__main__':
    main()
