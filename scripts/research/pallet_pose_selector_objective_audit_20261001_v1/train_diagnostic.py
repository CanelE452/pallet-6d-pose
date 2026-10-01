"""Read-only fixed-scorer diagnosis: no optimizer, image model, or new labels."""
from scripts.research.pallet_pose_union_selection_20261001_v1 import common as C
READS = C.source_guard(allow_source_targets=False)
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as T
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import torch

NAME = 'pallet_pose_selector_objective_audit_20261001_v1'
DOC = C.ROOT / '_docs/experiments' / NAME


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def describe(values):
    values = np.asarray(values, float).reshape(-1)
    assert not np.isnan(values).any() and not np.isneginf(values).any()
    finite = values[np.isfinite(values)]
    ordered = np.sort(values)
    quantiles = {}
    for q in (0, 50, 90, 95, 99, 100):
        if not len(values):
            quantiles[f'P{q}'] = None
            continue
        rank = (len(values) - 1) * q / 100
        lo, hi = math.floor(rank), math.ceil(rank)
        quantiles[f'P{q}'] = float(ordered[lo]) if lo == hi else float('inf') if np.isposinf(ordered[hi]) else float(ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo]))
    return dict(n=len(values), finite=len(finite), positive_infinity=int(np.isposinf(values).sum()),
                mean=float(values.mean()) if len(values) else None,
                finite_mean=float(finite.mean()) if len(finite) else None,
                finite_quantiles={f'P{q}': float(np.quantile(finite, q / 100)) if len(finite) else None for q in (0, 50, 90, 95, 99, 100)},
                **quantiles)


def error_summary(errors, scale):
    errors = np.asarray(errors, float)
    return dict(frames=len(errors), failures=int((~np.isfinite(errors).all(1)).sum()),
                T_cm=describe(errors[:, 0]), R_deg=describe(errors[:, 1]),
                scalar_cost=describe(np.max(errors / scale, axis=1)))


def checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def statistics(ck, raw, valid, errors, scale, names, fallback, ids, detailed=True):
    target = T.targets(errors, valid, scale, names)
    score = T.score_candidates(ck, raw, valid)
    pick = T.select_candidates(score, valid, names)
    x = np.zeros_like(raw, dtype=np.float32)
    x[valid] = (raw[valid] - ck['mean']) / ck['std']
    rows = np.arange(len(raw))
    present = target >= 0
    # Preserve actual float32 choices above. Differentiate the corresponding
    # mathematical linear CE in float64, on the exact same float32 inputs and
    # stored weight values, to make curvature/descent statements well-defined.
    w64 = ck['state']['net.weight'].detach().numpy().reshape(-1).astype(float)
    b64 = float(ck['state']['net.bias'].item())
    score64 = np.einsum('nkd,d->nk', x.astype(float), w64) + b64
    score64[~valid] = np.inf
    logits = -score64
    probabilities = np.zeros_like(logits)
    ce = np.zeros(len(raw))
    if present.any():
        stable = logits[present] - np.max(logits[present], axis=1, keepdims=True)
        exponent = np.exp(stable)
        probabilities[present] = exponent / exponent.sum(1, keepdims=True)
        ce[present] = np.log(exponent.sum(1)) - stable[np.arange(present.sum()), target[present]]
    selected = np.array(fallback, copy=True)
    selected[pick >= 0] = errors[rows[pick >= 0], pick[pick >= 0]]
    cost = np.max(errors / scale, axis=-1)
    best = cost.min(1)
    actual = np.max(selected / scale, axis=-1)
    regret = np.zeros(len(raw))
    regret[present] = actual[present] - best[present]
    assert np.all(regret[present] >= 0)
    margin = np.zeros(len(raw))
    multiple = valid.sum(1) >= 2
    ordered_cost = np.sort(cost[multiple], axis=1)
    margin[multiple] = ordered_cost[:, 1] - ordered_cost[:, 0]
    disagreement = present & (pick != target)
    suboptimal = present & (regret > 0)
    result = dict(frames=len(raw), valid_candidates=dict(Counter(valid.sum(1).tolist())),
                  target_distribution=dict(Counter(names[j] if j >= 0 else 'NO_VALID' for j in target)),
                  selected_distribution=dict(Counter(names[j] if j >= 0 else 'R0_GEO_FALLBACK' for j in pick)),
                  exact_target_accuracy=float(np.mean(pick[present] == target[present])),
                  disagreement=int(disagreement.sum()), strict_suboptimal_cost=int(suboptimal.sum()),
                  mean_CE_all_frames=float(ce.mean()), summary=error_summary(selected, scale),
                  analytical_score64_vs_operational_score32_maxdiff=float(np.max(np.abs(score64[valid]-score[valid]))),
                  oracle_cost=describe(best), regret=describe(regret[present]), wrong_regret=describe(regret[disagreement]),
                  second_best_cost_margin=describe(margin[multiple]),
                  regret_gt1=int(np.sum(regret > 1)), regret_gt10=int(np.sum(regret > 10)),
                  invalid_softmax_probability=float(probabilities[~valid].sum()))
    confusion = np.zeros((len(names), len(names)), np.int64)
    for a, b in zip(target[present], pick[present]):
        confusion[a, b] += 1
    result['confusion_target_rows_prediction_columns'] = confusion
    result['candidate_names'] = names
    if not detailed:
        return result
    top = np.argsort(-regret, kind='stable')[:max(1, math.ceil(len(raw) * .1))]
    result['top10percent_regret_frames'] = dict(n=len(top), regret_share=float(regret[top].sum() / regret.sum()) if regret.sum() else 0,
        CE_share=float(ce[top].sum() / ce.sum()) if ce.sum() else 0,
        examples=[dict(id=str(ids[j]), target=names[target[j]], selected=names[pick[j]],
                       selected_error=selected[j], target_error=errors[j, target[j]],
                       regret=float(regret[j]), CE=float(ce[j]), second_cost_margin=float(margin[j])) for j in top[:20]])
    result['margin_bins'] = {}
    for label, lower, upper in [('zero', -.1, 0), ('0_to_0.1', 0, .1), ('0.1_to_1', .1, 1), ('1_to_10', 1, 10), ('above10', 10, np.inf)]:
        subset = multiple & (margin > lower) & (margin <= upper)
        result['margin_bins'][label] = dict(frames=int(subset.sum()), errors=int(disagreement[subset].sum()),
            accuracy=float(np.mean(~disagreement[subset])) if subset.any() else None,
            regret_sum=float(regret[subset].sum()), CE_sum=float(ce[subset].sum()))
    wrong_hyp = disagreement & ((pick % 2) != (target % 2))
    same_hyp_wrong_expert = disagreement & ~wrong_hyp
    result['error_decomposition'] = {}
    for label, subset in [('wrong_hypothesis', wrong_hyp), ('same_hypothesis_wrong_expert', same_hyp_wrong_expert), ('correct', present & ~disagreement)]:
        result['error_decomposition'][label] = dict(frames=int(subset.sum()),
            regret_sum=float(regret[subset].sum()), regret_share=float(regret[subset].sum()/regret.sum()) if regret.sum() else 0,
            CE_sum=float(ce[subset].sum()), CE_share=float(ce[subset].sum()/ce.sum()) if ce.sum() else 0,
            selected_T_cm=describe(selected[subset, 0]), selected_R_deg=describe(selected[subset, 1]))
    # Analytical derivatives at frozen weights only. No optimizer or update.
    target_x = np.zeros((len(raw), 94))
    target_x[present] = x[rows[present], target[present]]
    expectation = np.einsum('nk,nkd->nd', probabilities, x.astype(float))
    gradient = np.mean(target_x - expectation, axis=0)
    candidate_regret = np.zeros_like(cost)
    candidate_regret[valid] = (cost - np.where(present, best, 0)[:, None])[valid]
    soft_regret = np.sum(probabilities * candidate_regret, axis=1)
    regret_gradient = np.mean(soft_regret[:, None] * expectation - np.einsum('nk,nkd->nd', probabilities * candidate_regret, x.astype(float)), axis=0)
    centered = x.astype(float) - expectation[:, None, :]
    hessian = np.einsum('nk,nkd,nke->de', probabilities, centered, centered, optimize=True) / len(raw)
    eigen, eigenvectors = np.linalg.eigh(hessian)
    positive = eigen > max(0, eigen[-1]) * 1e-10
    projected_gradient = eigenvectors.T @ gradient
    decrement_squared = float(np.sum(projected_gradient[positive] ** 2 / eigen[positive]))
    # For any softmax probabilities, covariance = sum_{a<b} p_a*p_b
    # (x_a-x_b)(x_a-x_b)^T and p_a*p_b <= 1/4. Hence this matrix is
    # a global Hessian upper bound, with no model update or line search.
    upper = np.zeros((94, 94))
    for a in range(x.shape[1]):
        for b in range(a + 1, x.shape[1]):
            available = valid[:, a] & valid[:, b]
            delta = x[available, a].astype(float) - x[available, b].astype(float)
            upper += delta.T @ delta / (4 * len(raw))
    lipschitz_upper = float(np.linalg.eigvalsh(upper)[-1])
    assert np.linalg.eigvalsh(upper - hessian)[0] >= -1e-10
    weight = ck['state']['net.weight'].detach().numpy().reshape(-1).astype(float)
    result['frozen_objective_geometry'] = dict(gradient_l2=float(np.linalg.norm(gradient)),
        gradient_linf=float(np.abs(gradient).max()), hessian_eigenvalues=eigen,
        positive_rank_relative1e_10=int(np.sum(eigen > max(0, eigen[-1]) * 1e-10)),
        weight_l2=float(torch.linalg.norm(ck['state']['net.weight']).item()),
        soft_expected_regret=float(soft_regret.mean()), expected_regret_gradient_l2=float(np.linalg.norm(regret_gradient)),
        CE_expected_regret_gradient_cosine=float(np.dot(gradient,regret_gradient)/(np.linalg.norm(gradient)*np.linalg.norm(regret_gradient))),
        CE_stationarity=dict(newton_decrement_squared=decrement_squared, newton_decrement=math.sqrt(decrement_squared),
            local_quadratic_gap_estimate=decrement_squared / 2,
            local_quadratic_estimate_is_not_a_global_bound=True,
            positive_eigen_min=float(eigen[positive][0]), positive_eigen_max=float(eigen[-1]),
            positive_condition_number=float(eigen[-1] / eigen[positive][0]),
            nullspace_gradient_l2=float(np.linalg.norm(projected_gradient[~positive])),
            global_Hessian_spectral_upper_bound=lipschitz_upper,
            guaranteed_possible_CE_decrease_lower_bound=float(np.dot(gradient, gradient)/(2*lipschitz_upper)),
            bound_note='Descent lemma for the mathematical linear softmax CE: a hypothetical -gradient/L step decreases CE by at least ||gradient||^2/(2L). No step was applied or scored. This does not bound T/R or median improvement.',
            AdamW_weight_decay=.0001, AdamW_lr=.001,
            per_step_decoupled_decay_weight_norm=float(1e-7*np.linalg.norm(weight)),
            wd_times_weight_norm_over_CE_gradient=float(.0001*np.linalg.norm(weight)/np.linalg.norm(gradient)),
            explicit_loss_penalty=0.,
            L2_surrogate_penalty_if_one_used_lambda_equal_wd=float(.5*.0001*np.dot(weight,weight)),
            penalty_note='AdamW weight decay was decoupled; the hypothetical L2 value is scale context only and was not the trained objective.'),
        within_frame_constant_feature_indices=np.flatnonzero(np.ptp(x, axis=1).max(0) == 0),
        interpretation='Frozen analytical derivative, no optimization. Small CE gradients do not prove low pose regret; rank threshold is descriptive only.')
    result['per_frame'] = dict(ids=ids, target=target, selected=pick, selected_T_cm=selected[:, 0],
        selected_R_deg=selected[:, 1], scalar_cost=actual, best_cost=best,
        regret_defined=present, regret=np.where(present, regret, np.nan), CE=ce, second_cost_margin=margin)
    return result


def main():
    torch.set_num_threads(1)
    C.verify(C.read(C.DOC / 'TRAIN_PROTOCOL_SHA.json'))
    protocol = C.read(C.DOC / 'TRAIN_PROTOCOL.json')
    for b in list(protocol['inputs'].values()) + protocol['codes']:
        C.verify(b)
    done = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert done['complete'] and done['fit_count'] == 6 and done['total_updates'] == 1980
    assert done['protocol'] == C.bind(C.DOC / 'TRAIN_PROTOCOL.json')
    bindings = [C.bind(C.DOC / 'TRAIN_PROTOCOL.json'), C.bind(C.DOC / 'TRAINING_COMPLETE.json')]
    with np.load(C.RAW / 'SOURCE_FEATURES.npz', allow_pickle=False) as f:
        features = {k: f[k] for k in f.files}
    with np.load(C.RAW / 'SOURCE_TRAIN_LABELS.npz', allow_pickle=False) as f:
        labels = {k: f[k] for k in f.files}
    ix, ids = labels['source_index'], labels['ids']
    assert len(ids) == 2598 and np.all(features['split'][ix] == 'TRAIN')
    np.testing.assert_array_equal(features['ids'][ix], ids)
    scale = np.array([labels['sT_cm'].item(), labels['sR_deg'].item()])
    assert np.isfinite(scale).all() and (scale > 0).all()
    feat = {m: features[m + '_geo'][ix] for m in C.MODELS}
    valid = {m: features[m + '_valid'][ix] for m in C.MODELS}
    errors = {m: np.stack([labels[m + '_T_cm'], labels[m + '_R_deg']], axis=-1) for m in C.MODELS}
    old = {m: np.stack([labels[m + '_GEO_T_cm'], labels[m + '_GEO_R_deg']], axis=-1) for m in C.MODELS}
    receipts, checkpoints, traces = {}, {}, {}
    for binding in done['fits']:
        C.verify(binding)
        r = C.read(C.ROOT / binding['path'])
        name = f"{r['arm']}_s{r['seed']}"
        assert name not in receipts and r['complete'] and r['updates'] == 330 and r['protocol'] == done['protocol']
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(r[key])
        ck = checkpoint(C.ROOT / r['checkpoint']['path'])
        assert T.state_sha(ck['state']) == r['final_state_sha']
        trace = [json.loads(line) for line in (C.ROOT / r['trace']['path']).read_text().splitlines()]
        assert len(trace) == 330 and trace[-1]['state_sha'] == r['final_state_sha']
        receipts[name], checkpoints[name], traces[name] = r, ck, trace
        bindings += [binding, r['checkpoint'], r['trace']]
    assert set(receipts) == {f'{a}_s{s}' for s in T.SEEDS for a in T.ARMS}
    lock = C.read(C.DOC / 'SOURCE_FEATURE_LOCK.json')
    result = dict(complete=True, created_at=C.now(), scope='FROZEN_TRAIN_SCORER_DIAGNOSTIC',
                  frames=2598, scale=dict(sT_cm=scale[0], sR_deg=scale[1]), fits={},
                  old_GEO={m: error_summary(e, scale) for m, e in old.items()},
                  original_gates_unchanged=True, new_fits=0, new_image_forwards=0,
                  real_routes=0, real_references_read=False, new_VAL_quality_computed=False,
                  quantile_note='P-values use full population and the original extended-real interpolation; finite_quantiles are explicitly conditional. Infinity serializes null with positive_infinity count. No denominator or primary gate is altered.',
                  derivative_note='Actual choices retain original float32 scorer. CE derivatives use the mathematical float64 linear function of the same frozen float32 normalized inputs and weight values; recorded score32/score64 discrepancies disclose rounding.',
                  normalization={}, feature_names=lock['feature_names'])
    common_ck = checkpoints['R0_ONLY_s1']
    r0 = feat['R0'][valid['R0']]
    np.testing.assert_array_equal(common_ck['mean'], r0.mean(0))
    np.testing.assert_array_equal(common_ck['std'], np.maximum(r0.std(0), np.float32(1e-6)))
    for ck in checkpoints.values():
        np.testing.assert_array_equal(ck['mean'], common_ck['mean'])
        np.testing.assert_array_equal(ck['std'], common_ck['std'])
    for model in C.MODELS:
        normalized = (feat[model][valid[model]] - common_ck['mean']) / common_ck['std']
        assert np.isfinite(normalized).all()
        mx = np.max(np.abs(normalized), axis=0)
        top = np.argsort(-mx)[:10]
        result['normalization'][model] = dict(valid_vectors=len(normalized), nonfinite=0,
            abs_feature_values=describe(np.abs(normalized)), coordinates_abs_gt10=int(np.sum(np.abs(normalized) > 10)),
            coordinates_abs_gt100=int(np.sum(np.abs(normalized) > 100)),
            vectors_any_abs_gt10=int(np.sum(np.max(np.abs(normalized), axis=1) > 10)),
            top_columns=[dict(index=int(j), name=lock['feature_names'][j], absmax=float(mx[j]),
                             mean=float(normalized[:, j].mean()), std=float(normalized[:, j].std())) for j in top])
    for seed in T.SEEDS:
        for arm in T.ARMS:
            name = f'{arm}_s{seed}'
            models = ['R0'] + ([f'DIVERSE251_s{seed}'] if arm == 'UNION' else [])
            x, v, e = [np.concatenate([source[m] for m in models], axis=1) for source in (feat, valid, errors)]
            ck = checkpoints[name]
            stat = statistics(ck, x, v, e, scale, T.candidate_names(arm, seed), old['R0'], ids)
            initial = dict(ck, state=T.initialized(seed).state_dict())
            assert T.state_sha(initial['state']) == receipts[name]['initial_state_sha']
            ini = statistics(initial, x, v, e, scale, T.candidate_names(arm, seed), old['R0'], ids, detailed=False)
            stat['initial'] = {k: ini[k] for k in ('mean_CE_all_frames', 'exact_target_accuracy', 'summary', 'regret')}
            stat['parameter_displacement_initial_to_final_l2'] = float(torch.linalg.norm(ck['state']['net.weight']-initial['state']['net.weight']).item())
            stat['intermediate_parameter_update_norm_available'] = False
            stat['old_GEO_comparison_same_pool'] = {}
            best = np.max(e / scale, axis=-1).min(1)
            for baseline in models:
                available = np.isfinite(best) & np.isfinite(old[baseline]).all(1)
                regret_old = np.max(old[baseline][available] / scale, axis=1) - best[available]
                stat['old_GEO_comparison_same_pool'][baseline] = dict(frames_with_defined_regret=int(available.sum()),
                    no_valid_oracle_or_failed_baseline=int((~available).sum()), regret=describe(regret_old),
                    scalar_cost_better_than_learned=int(np.sum(np.max(old[baseline] / scale, axis=1) < np.asarray(stat['per_frame']['scalar_cost']))),
                    scalar_cost_worse_than_learned=int(np.sum(np.max(old[baseline] / scale, axis=1) > np.asarray(stat['per_frame']['scalar_cost']))))
            epoch = []
            for number in range(1, 31):
                batch = [row for row in traces[name] if row['epoch'] == number]
                assert len(batch) == 11 and sum(row['batch_size'] for row in batch) == 2598
                epoch.append(dict(epoch=number, frames=2598,
                    online_loss_frame_weighted=sum(row['loss'] * row['batch_size'] for row in batch) / 2598,
                    online_loss_batch_unweighted=float(np.mean([row['loss'] for row in batch])),
                    supervised_rows=sum(row['active_rows'] for row in batch)))
            stat['epoch_online_losses'] = epoch
            stat['last5_online_loss_relative_change'] = epoch[-1]['online_loss_frame_weighted'] / epoch[-6]['online_loss_frame_weighted'] - 1
            result['fits'][name] = stat
            print('DIAGNOSED', name, 'accuracy', stat['exact_target_accuracy'], 'CE', stat['mean_CE_all_frames'], 'regret_mean', stat['regret']['mean'], flush=True)
    # Cross-apply already frozen UNION weights to TRAIN candidate pools only.
    # This does not identify initialization causally: the weights themselves
    # were jointly affected by each original pool and optimizer seed.
    result['frozen_weight_by_TRAIN_pool'] = {}
    for ws in T.SEEDS:
        for ps in T.SEEDS:
            model = f'DIVERSE251_s{ps}'
            x, v, e = [np.concatenate([source['R0'], source[model]], axis=1) for source in (feat, valid, errors)]
            stat = statistics(checkpoints[f'UNION_s{ws}'], x, v, e, scale, T.candidate_names('UNION', ps), old['R0'], ids, detailed=False)
            result['frozen_weight_by_TRAIN_pool'][f'weights{ws}_pool{ps}'] = stat
    result['cross_apply_limitation'] = 'Read-only TRAIN probe of frozen weights and pools. Not a crossed training design, not a causal separation of initialization/refiner effects, not a new model selection.'
    # Reuse completed VAL summaries only; no new VAL selection or label math.
    val_path = C.DOC / 'SOURCE_VAL_GATE.json'
    val = C.read(val_path)
    assert val['complete'] and val['protocol'] == done['protocol']
    C.verify(val['routing_lock'])
    C.verify(val['metrics'])
    result['already_scored_VAL'] = {k: val[k] for k in ('frames', 'PASS', 'status', 'failed_checks', 'summaries')}
    bindings += [C.bind(val_path), val['metrics'], protocol['inputs']['features'], protocol['inputs']['train_labels'], C.bind(Path(__file__))]
    result['bindings'] = bindings
    result['read_paths'] = sorted(set(READS))
    DOC.mkdir(parents=True, exist_ok=True)
    # Only this new diagnostic is regenerated; every original input is read-only.
    with (DOC / 'TRAIN_DIAGNOSTIC.json').open('w') as handle:
        json.dump(clean(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
