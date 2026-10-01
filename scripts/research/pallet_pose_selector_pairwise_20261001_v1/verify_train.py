"""Read-only independent TRAIN audit after all pairwise fits are complete.

No optimizer is imported or invoked here. Actual execution requires completion;
selfcheck is purely analytic. Frozen source TRAIN labels are the only targets.
"""
from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))
LAMBDA = 1e-4


def independent_ranks(errors, valid, scale, names):
    ranks = np.full(valid.shape, -1, np.int64)
    for i in range(len(valid)):
        costs = {int(j): float(np.max(errors[i, j] / scale)) for j in np.flatnonzero(valid[i])}
        rank = 0
        while costs:
            smallest = min(costs.values())
            group = {j for j, cost in costs.items() if cost == smallest}
            while group:
                nondominated = set(group)
                for j in group:
                    for k in group:
                        if np.all(errors[i, k] <= errors[i, j]) and np.any(errors[i, k] < errors[i, j]):
                            nondominated.discard(j)
                assert nondominated
                for j in sorted(nondominated, key=lambda k: OLD.tie_key(names[k])):
                    ranks[i, j] = rank
                    rank += 1
                    del costs[j]
                group.difference_update(nondominated)
    return ranks


def prepare(parent, model):
    seed = 1 if model == 'R0_ONLY' else int(model[-1])
    names = OLD.candidate_names('R0_ONLY' if model == 'R0_ONLY' else 'UNION', seed)
    models = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{seed}'])
    raw, valid, errors = [np.concatenate([parent[key][m] for m in models], axis=1)
                          for key in ('features', 'valid', 'errors')]
    x32 = np.zeros_like(raw, np.float32)
    x32[valid] = (raw[valid] - parent['mean']) / parent['std']
    x = x32.astype(np.float64)
    ranks = independent_ranks(errors, valid, parent['scale'], names)
    target = OLD.targets(errors, valid, parent['scale'], names)
    present = valid.any(1)
    assert np.array_equal(np.argmin(np.where(valid[present], ranks[present], 100), axis=1), target[present])
    edges = ((0, 1),) if model == 'R0_ONLY' else EDGES
    active = np.stack([valid[:, a] & valid[:, b] for a, b in edges], axis=1)
    winners = np.full(active.shape, -1, np.int64)
    for e, (a, b) in enumerate(edges):
        rows = np.flatnonzero(active[:, e])
        winners[rows, e] = np.where(ranks[rows, a] < ranks[rows, b], a, b)
    return dict(x=x, valid=valid, errors=errors, ranks=ranks, target=target,
                names=names, edges=edges, active=active, winners=winners)


def independent_objective(weight, q):
    """Torch score differences/softplus and autograd, not NumPy train gradient."""
    w = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    scores = torch.einsum('nkd,d->nk', torch.from_numpy(q['x']), w)
    total = scores.sum() * 0.
    contributions = []
    coefficient = 1. if len(q['edges']) == 1 else .25
    for e, (a, b) in enumerate(q['edges']):
        rows = np.flatnonzero(q['active'][:, e])
        winners = q['winners'][rows, e]
        losers = np.where(winners == a, b, a)
        edge_sum = F.softplus(scores[rows, winners] - scores[rows, losers]).sum()
        term = coefficient * edge_sum / len(scores)
        total = total + term
        contributions.append(float(term.detach()))
    penalty = .5 * LAMBDA * w.square().sum()
    value = total + penalty
    gradient, = torch.autograd.grad(value, w)
    gradient = gradient.detach().numpy()
    present = q['valid'].any(1)
    logits = (-scores[present]).masked_fill(~torch.from_numpy(q['valid'][present]), -torch.inf)
    ce = F.cross_entropy(logits, torch.from_numpy(q['target'][present]), reduction='sum') / len(scores)
    norm = float(np.linalg.norm(gradient))
    return dict(objective=float(value.detach()), pairwise_loss=float(total.detach()),
        L2_penalty=float(penalty.detach()), gradient_l2=norm,
        gradient_linf=float(np.max(np.abs(gradient))), certified_gap_upper_bound=norm ** 2 / (2 * LAMBDA),
        whole_best_CE=float(ce.detach()), edge_loss_contributions=contributions,
        gradient=gradient, score=scores.detach().numpy())


def metrics(recomputed, q, scale):
    scores = recomputed['score']
    picked = OLD.select_candidates(scores, q['valid'], q['names'])
    target, present = q['target'], q['target'] >= 0
    rows = np.flatnonzero(present)
    edge_metrics = {}
    edge_correct, edge_counts = [], []
    for e, (a, b) in enumerate(q['edges']):
        active = q['active'][:, e]
        pair_pick = OLD.select_candidates(scores[:, [a, b]], q['valid'][:, [a, b]], [q['names'][a], q['names'][b]])
        actual = np.asarray([a, b])[pair_pick[active]]
        right = int((actual == q['winners'][active, e]).sum())
        count = int(active.sum())
        edge_correct.append(right)
        edge_counts.append(count)
        edge_metrics[f'{a}:{b}'] = dict(correct=right, active=count, unavailable=len(picked) - count,
            accuracy=right / count if count else None,
            weighted_loss_over_all_rows=recomputed['edge_loss_contributions'][e])
    group_metrics = {}
    for name, positions in [('WD', [0])] if len(q['edges']) == 1 else [('WD', [0, 1]), ('expert', [2, 3])]:
        correct, count = sum(edge_correct[j] for j in positions), sum(edge_counts[j] for j in positions)
        group_metrics[name] = dict(correct=correct, active=count, accuracy=correct / count if count else None,
            weighted_loss_over_all_rows=sum(recomputed['edge_loss_contributions'][j] for j in positions))
    costs = np.max(q['errors'] / scale, axis=2)
    selected, best = costs[rows, picked[rows]], costs[rows, target[rows]]
    regret = selected - best
    assert np.isfinite(regret).all() and (regret >= -1e-12).all()
    err = np.full((len(picked), 2), np.inf)
    err[rows] = q['errors'][rows, picked[rows]]
    selected_hyp = np.asarray([n.split(':')[1] for n in q['names']])[picked[rows]]
    target_hyp = np.asarray([n.split(':')[1] for n in q['names']])[target[rows]]
    wrong = picked[rows] != target[rows]
    diff_hyp = selected_hyp != target_hyp
    return dict(frames=len(picked), no_candidate_rows=int((~present).sum()),
        defined_cost_rows=len(rows), full_denominator_rows=len(picked),
        whole_best_correct=int((~wrong).sum()), whole_best_accuracy=float((~wrong).mean()),
        global_choice_different_hypothesis_from_best=int(diff_hyp.sum()),
        global_choice_same_hypothesis_wrong_expert=int((wrong & ~diff_hyp).sum()),
        edges=edge_metrics, groups=group_metrics,
        mean_selected_cost_defined_rows=float(selected.mean()), mean_best_cost_defined_rows=float(best.mean()),
        mean_regret_defined_rows=float(regret.mean()), P90_regret_defined_rows=float(np.quantile(regret, .9)),
        total_regret_defined_rows=float(regret.sum()),
        T_median_cm_full=float(np.quantile(err[:, 0], .5)), R_median_deg_full=float(np.quantile(err[:, 1], .5)),
        T_P90_cm_full=float(np.quantile(err[:, 0], .9)), R_P90_deg_full=float(np.quantile(err[:, 1], .9)),
        cost_mean_scope='Available-candidate TRAIN rows only; unavailable rows explicitly counted and retained as +inf for full-frame T/R quantiles.',
        hypothesis_error_scope='Disagreement with cost-best whole pose, not a separate ground-truth width/depth classifier.')


def public_objective(value):
    return {key: item for key, item in value.items() if key not in ('gradient', 'score')}


def verify_old_complete():
    seal = C.read(C.CONV_DOC / 'PROTOCOL_SHA.json')
    C.verify(seal)
    assert seal == C.bind(C.CONV_DOC / 'PROTOCOL.json')
    complete = C.read(C.CONV_DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['protocol'] == seal and complete['models'] == list(C.MODEL_NAMES)
    bindings = {}
    for binding in complete['fits']:
        C.verify(binding)
        receipt = C.read(C.ROOT / binding['path'])
        assert receipt['model'] not in bindings
        assert receipt['complete'] and receipt['protocol'] == seal
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(receipt[key])
        bindings[receipt['model']] = (binding, receipt)
    assert set(bindings) == set(C.MODEL_NAMES)
    return bindings


def main():
    torch.set_num_threads(1)
    OLD.install_training_guard()
    parent = OLD.load_training_inputs()
    protocol = C.protocol()
    binding = C.bind(C.DOC / 'PROTOCOL.json')
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['protocol'] == binding
    assert complete['models'] == list(C.MODEL_NAMES) and complete['model_count'] == 4
    assert complete['fit_count'] == complete['new_fit_count'] == 3 and complete['reused_count'] == 1
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    audit = C.read(C.DOC / 'TARGET_CONTRACT_AUDIT.json')
    assert audit['PASS'] and audit['complete'] and protocol['inputs']['target_audit'] == C.bind(C.DOC / 'TARGET_CONTRACT_AUDIT.json')
    old_receipts = verify_old_complete()
    records, bound = {}, [binding, C.bind(Path(__file__)), C.bind(C.DOC / 'TRAINING_COMPLETE.json'),
        C.bind(C.CONV_DOC / 'TRAINING_COMPLETE.json'), C.bind(C.CONV_DOC / 'PROTOCOL_SHA.json')]
    for fit_binding in complete['fits']:
        C.verify(fit_binding)
        receipt = C.read(C.ROOT / fit_binding['path'])
        model = receipt['model']
        assert model in C.MODEL_NAMES and model not in records
        assert receipt['complete'] and receipt['protocol'] == binding
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(receipt[key])
        ck, start = [C.read(C.ROOT / receipt[key]['path']) for key in ('checkpoint', 'START')]
        assert ck['schema'] == 'pallet_pose_pairwise_linear94_v1'
        assert ck['protocol'] == start['protocol'] == binding
        assert ck['model'] == start['model'] == model and ck['bias'] == 0. and ck['lambda_l2'] == LAMBDA
        assert ck['normalization_sha'] == receipt['normalization_sha'] == parent['normalization_sha']
        for key in ('mean', 'std'):
            np.testing.assert_array_equal(np.asarray(ck[key], np.float32), parent[key])
        q = prepare(parent, model)
        assert ck['names'] == q['names']
        for key, value in dict(normalized_features_sha=q['x'], target_sha=q['target'], valid_mask_sha=q['valid'],
            rank_sha=q['ranks'], edge_winner_sha=q['winners'], edge_active_sha=q['active'], source_ids_sha=parent['ids']).items():
            assert OLD.array_sha(value) == start[key], ('START_INPUT_HASH', model, key)
        if model != 'R0_ONLY':
            for key, value in dict(global_rank=q['ranks'], edge_winners=q['winners'], edge_active=q['active'], original_whole_best=q['target']).items():
                assert OLD.array_sha(value) == audit['seeds'][model[-1]]['hashes'][key]
        weight = np.asarray(ck['weight'], np.float64)
        assert weight.shape == (94,) and np.isfinite(weight).all() and OLD.array_sha(weight) == receipt['final_weight_sha']
        value = independent_objective(weight, q)
        saved = ck['certificate']
        assert saved == receipt['certificate'] and saved['PASS'] and saved['optimizer_success']
        assert value['certified_gap_upper_bound'] <= 1e-6
        assert abs(value['objective'] - saved['objective_value']) < 1e-12
        assert abs(value['pairwise_loss'] - saved['pairwise_loss']) < 1e-12
        assert abs(value['gradient_l2'] - saved['gradient_l2']) < 1e-12
        assert saved['gradient_l2_squared_over_2lambda'] <= 1e-6
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls = [r for r in trace if r['event'] == 'objective']
        iterations = [r for r in trace if r['event'] == 'iteration']
        assert len(calls) == receipt['objective_calls'] == saved['objective_calls'] <= 2000
        assert len(iterations) == receipt['iterations'] == saved['iterations'] <= 1000
        assert receipt['optimizer_steps'] == ck['optimizer_steps'] == len(iterations)
        assert [r['call'] for r in calls] == list(range(1, len(calls) + 1))
        assert [r['iteration'] for r in iterations] == list(range(1, len(iterations) + 1))
        old_receipt_binding, old_receipt = old_receipts[model]
        old_ck = C.read(C.ROOT / old_receipt['checkpoint']['path'])
        assert old_ck['schema'] == 'pallet_pose_convex_linear94_v1' and old_ck['model'] == model
        assert old_ck['certificate'] == old_receipt['certificate'] and old_ck['certificate']['PASS']
        assert old_ck['names'] == q['names'] and old_ck['normalization_sha'] == parent['normalization_sha']
        old_start = C.read(C.ROOT / old_receipt['START']['path'])
        assert old_start['normalized_features_sha'] == start['normalized_features_sha']
        assert old_start['valid_mask_sha'] == start['valid_mask_sha'] and old_start['target_sha'] == start['target_sha']
        old_weight = np.asarray(old_ck['weight'], np.float64)
        assert OLD.array_sha(old_weight) == old_receipt['final_weight_sha']
        for key in ('mean', 'std'):
            np.testing.assert_array_equal(np.asarray(old_ck[key], np.float32), parent[key])
        old_value = independent_objective(old_weight, q)
        assert value['objective'] <= old_value['objective'] + 1e-6
        reuse_checks = None
        if model == 'R0_ONLY':
            assert receipt['reused'] and not receipt['new_fit'] and receipt['fits_executed'] == 0
            assert ck['fits_executed'] == ck['optimizer_steps'] == receipt['optimizer_steps'] == 0
            assert len(trace) == 1 and trace[0]['event'] == 'reuse_equivalence'
            for key, expected in dict(origin_checkpoint=old_receipt['checkpoint'], origin_receipt=old_receipt_binding,
                                       origin_protocol=old_receipt['protocol']).items():
                assert receipt[key] == ck[key] == expected
            for key, dtype in [('weight', np.float64), ('mean', np.float32), ('std', np.float32)]:
                assert np.asarray(ck[key], dtype).tobytes() == np.asarray(old_ck[key], dtype).tobytes()
            assert receipt['equivalence'] == ck['equivalence'] and ck['equivalence']['PASS']
            assert saved['optimizer_success_source'] == 'verified_origin_R0_ONLY_fit'
            assert abs(value['objective'] - old_ck['certificate']['objective_value']) <= 1e-12
            assert abs(value['pairwise_loss'] - value['whole_best_CE']) <= 1e-12
            reuse_checks = dict(PASS=True, numeric_weight_and_normalization_bitexact=True, optimizer_steps=0,
                pairwise_CE_abs_diff=abs(value['pairwise_loss'] - value['whole_best_CE']),
                origin_objective_abs_diff=abs(value['objective'] - old_ck['certificate']['objective_value']))
        else:
            assert not receipt['reused'] and receipt['new_fit'] and receipt['fits_executed'] == ck['fits_executed'] == 1
            assert calls and calls[-1]['weight_sha'] == receipt['final_weight_sha']
            assert calls[-1]['objective'] == saved['objective_value']
        new_metrics, old_metrics = metrics(value, q, parent['scale']), metrics(old_value, q, parent['scale'])
        for key in ('WD', 'expert'):
            if key in new_metrics['groups']:
                assert new_metrics['groups'][key]['active'] == old_metrics['groups'][key]['active']
        records[model] = dict(independent_recompute=public_objective(value), saved_certificate=saved,
            objective_abs_difference=abs(value['objective'] - saved['objective_value']),
            gradient_norm_abs_difference=abs(value['gradient_l2'] - saved['gradient_l2']),
            objective_calls=len(calls), iterations=len(iterations), reuse_checks=reuse_checks,
            new_TRAIN_metrics=new_metrics, old_converged_same_pool=dict(objective=public_objective(old_value),
                metrics=old_metrics, receipt=old_receipt_binding, checkpoint=old_receipt['checkpoint']),
            new_minus_old=dict(pairwise_objective=value['objective'] - old_value['objective'],
                whole_best_CE=value['whole_best_CE'] - old_value['whole_best_CE'],
                mean_regret_defined_rows=new_metrics['mean_regret_defined_rows'] - old_metrics['mean_regret_defined_rows'],
                global_different_hypothesis=new_metrics['global_choice_different_hypothesis_from_best'] - old_metrics['global_choice_different_hypothesis_from_best']))
        bound += [fit_binding, receipt['checkpoint'], receipt['trace'], receipt['START'], old_receipt_binding,
                  old_receipt['checkpoint'], old_receipt['trace'], old_receipt['START']]
    assert set(records) == set(C.MODEL_NAMES)
    assert sum(r['objective_calls'] for r in records.values()) == complete['total_objective_calls']
    out = dict(complete=True, independent_check_PASS=True, created_at=C.now(), models=records,
        protocol=binding, bindings=bound, source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
        new_fits=0, optimizer_steps=0, implementation='Independent torch.float64 per-edge softplus/autograd; independently reconstructed global ranks and exact original float32 normalization.',
        comparison='Old converged CE and new pairwise weights evaluated on identical frozen TRAIN inputs, candidate pools and target order. Diagnostic only, no new gate.',
        caveat='Numerical convergence and TRAIN differences do not establish source VAL or real T/R improvement. Four fixed edges need not uniquely determine whole-best argmin.',
        read_paths=sorted(set(OLD.READS)))
    C.save(C.DOC / 'TRAIN_CONVERGENCE.json', out)
    write_markdown(out)
    print('INDEPENDENT_PAIRWISE_TRAIN_PASS_3_NEW_1_REUSED', flush=True)
    for model, record in records.items():
        print(model, record['independent_recompute'], record['new_minus_old'], flush=True)


def write_markdown(out):
    lines = ['# Pairwise TRAIN 수렴과 선택 결과의 독립 검증', '',
        '**새 UNION 세 모델의 수렴 인증과 R0 기준 모델 재사용을 독립 계산으로 검증했다.** 이 문서는 TRAIN 진단이며 T/R 개선 판정이 아니다.', '',
        '[검증 JSON](TRAIN_CONVERGENCE.json)은 checkpoint·START·TRACE·protocol과 이전 모델의 원본 receipt를 연결한다. 별도 PyTorch float64 `softplus`와 자동미분을 사용했다. 검증 과정의 새 학습·optimizer step·VAL 품질·실사 GT 읽기는 모두0회다.', '',
        '| 모델 | pairwise loss | L2 penalty | objective | gradient L2 | gap 상한 | 새 iterations / calls |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model, record in out['models'].items():
        a = record['independent_recompute']
        lines.append(f"| {model} | {a['pairwise_loss']:.8f} | {a['L2_penalty']:.8f} | {a['objective']:.8f} | {a['gradient_l2']:.3g} | {a['certified_gap_upper_bound']:.3g} | {record['iterations']} / {record['objective_calls']} |")
    lines += ['', 'UNION은 WD 쌍2개와 같은 WD의 expert 쌍2개를 사용한다. 각 그룹은 고정된 두 edge의 평균이며 두 그룹 비중은 각각0.5다. 불가능한 edge는0으로 두고 전체2598행으로 나눈다. `0.5×1e−4×||w94||²`를 더한 목적함수에 대해 `J(w)−J* ≤ ||∇J(w)||²/(2λ)` 상한1e−6과 solver success를 함께 확인했다. trace 번호·1000 iterations/2000 closures 상한·최종 weight hash도 검증했다.', '',
        'R0는 edge1개의 logistic loss와 원래 두 후보 CE가 같은 함수다. 기존 수렴 weight·float32 평균/표준편차를 바이트 단위로 유지했다. 기존 optimizer 성공을 명시적으로 참조하고 이번 R0 optimizer step과 closure는0회다. 재사용도 저장된 숫자에서 목적함수와 gradient를 다시 계산했다.', '',
        '| 모델 | 이전 → 새 pairwise objective | 이전 → 새 WD edge 정확도 | 이전 → 새 expert edge 정확도 | 이전 → 새 whole-best 정확도 | 이전 → 새 평균 regret |',
        '|---|---:|---:|---:|---:|---:|']
    for model, record in out['models'].items():
        if model == 'R0_ONLY':
            continue
        a, b = record['old_converged_same_pool'], record['new_TRAIN_metrics']
        m = a['metrics']
        lines.append(f"| {model} | {a['objective']['objective']:.6f} → {record['independent_recompute']['objective']:.6f} | {m['groups']['WD']['accuracy']:.2%} → {b['groups']['WD']['accuracy']:.2%} | {m['groups']['expert']['accuracy']:.2%} → {b['groups']['expert']['accuracy']:.2%} | {m['whole_best_accuracy']:.2%} → {b['whole_best_accuracy']:.2%} | {m['mean_regret_defined_rows']:.6f} → {b['mean_regret_defined_rows']:.6f} |")
    lines += ['', '비교 양쪽에 동일한 후보·TRAIN 정규화·2598행과 동일한 `max(T/sT,R/sR)` 비용을 적용했다. 후보가 없는1행은 수렴 목적함수 분모에서 유지한다. 위 정확도와 평균 regret은 후보가 있는2597행에서 정의하며, 누락1행을 정확한 선택이나 비용0으로 간주하지 않는다. JSON의 전체 T/R 분위수에는 이 실패를+∞로 유지했다.', '',
        '| 모델 | 이전 → 새 whole-best와 다른 WD 선택 수 | 이전 → 새 같은 WD의 다른 expert 선택 수 | 이전 → 새 whole-best CE |',
        '|---|---:|---:|---:|']
    for model, record in out['models'].items():
        if model == 'R0_ONLY':
            continue
        a, b = record['old_converged_same_pool'], record['new_TRAIN_metrics']
        m = a['metrics']
        lines.append(f"| {model} | {m['global_choice_different_hypothesis_from_best']} → {b['global_choice_different_hypothesis_from_best']} | {m['global_choice_same_hypothesis_wrong_expert']} → {b['global_choice_same_hypothesis_wrong_expert']} | {a['objective']['whole_best_CE']:.6f} → {record['independent_recompute']['whole_best_CE']:.6f} |")
    max_obj = max(r['objective_abs_difference'] for r in out['models'].values())
    max_grad = max(r['gradient_norm_abs_difference'] for r in out['models'].values())
    lines += ['', f'저장 certificate와 독립 계산의 최대 objective 차이는{max_obj:.3g}, gradient L2 차이는{max_grad:.3g}다. 모든 새 fit은 동일한 사전 solver·상한·수렴 기준을 사용했다.', '',
        '이번 변경은 기존 CE의 전체 최선 후보 타깃을 네 개의 고정 pair 비교로 바꾼 것이다. 원래 학습 목적함수와 같은 손실이라고 주장하지 않는다. 비용 격차 가중치나 새 입력 특징은 추가하지 않았다. WD/expert 정확도는 해당 TRAIN의 물리적 비용으로 만든 후보 순서와의 일치율이다.', '',
        '네 edge의 방향이 모두 맞더라도 누락된 대각 edge 때문에 최선 후보를 유일하게 정하지 못하는 경우가 있다. 독립 타깃 감사에서 seed별4/5/2행이 해당한다. 따라서 낮은 pairwise loss·높은 edge 정확도를 whole-pose 선택 성공으로 바꾸어 해석할 수 없다. 세 refiner 결과를 모두 보존하며 이 문서로 seed·threshold·checkpoint를 선택하지 않는다.', '',
        '**최종 판단에는 별도의 원래 source VAL 기준과, 그 통과 이후에만 허용되는 실사 기준이 필요하다.** TRAIN 비용 감소는 T/R 중앙값·tail·recording 안정성 개선을 보장하지 않는다.', '']
    C.save(C.DOC / 'TRAIN_CONVERGENCE_KO.md', '\n'.join(lines))


def selfcheck():
    names = OLD.candidate_names('UNION', 1)
    err = np.array([[[.9, 1.], [1., .5], [.8, 1.], [.7, 1.]]])
    valid = np.ones((1, 4), bool)
    ranks = independent_ranks(err, valid, np.ones(2), names)
    np.testing.assert_array_equal(np.argsort(ranks[0]), [1, 3, 2, 0])
    rng = np.random.default_rng(722)
    x = rng.normal(size=(2, 4, 3))
    q = dict(x=x, valid=np.array([[True]*4, [False]*4]), target=np.array([1, -1]),
        edges=EDGES, active=np.array([[True]*4, [False]*4]), winners=np.array([[1, 3, 2, 1], [-1]*4]))
    w = rng.normal(size=3)
    actual = independent_objective(w, q)
    grad = []
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = 1e-5
        grad.append((independent_objective(w+delta, q)['objective'] - independent_objective(w-delta, q)['objective']) / 2e-5)
    np.testing.assert_allclose(actual['gradient'], grad, atol=1e-10, rtol=1e-7)
    zero = independent_objective(np.zeros(3), q)
    np.testing.assert_allclose(zero['pairwise_loss'], .5*np.log(2), atol=1e-15)
    assert OLD.READS is None
    print('INDEPENDENT_PAIRWISE_VERIFY_SELFCHECK_PASS; invented arrays only; no artifact reads.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'verify'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        (selfcheck if args.stage == 'selfcheck' else main)()
