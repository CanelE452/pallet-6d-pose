"""Independently audit one rejected optimizer state; never choose/evaluate poses."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import common as C
from . import verify_train as I

OLD = I.OLD
V = I.V


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    OLD.install_training_guard()
    from . import convex_train as T

    protocol = C.protocol('TRAIN_PROTOCOL')
    binding = C.bind(C.DOC / 'TRAIN_PROTOCOL.json')
    assert T.metadata() == I.METADATA
    for key, value in I.METADATA.items():
        assert protocol[key] == value
    assert protocol['solver'] == dict(method='L-BFGS-B', maxiter=1000,
        maxfun=2000, ftol=1e-15, gtol=1e-8, bounds=None)
    assert protocol['certificate'] == dict(optimizer_success=True,
        gap_upper_bound_max=1e-6)
    assert protocol['lambda_l2'] == 1e-4 and protocol['bias'] == 0.
    assert protocol['train_rows'] == 2598
    model = 'R0_ONLY'
    folder = C.RAW / 'fits' / model
    assert {p.name for p in (C.RAW / 'fits').iterdir()} == {model}
    assert {p.name for p in folder.iterdir()} == {'START.json', 'TRACE.jsonl', 'FAILED.json'}
    absent = [C.DOC / 'TRAINING_COMPLETE.json']
    absent += [C.DOC / f'FIT_{m}.json' for m in C.MODEL_NAMES]
    absent += [C.RAW / 'fits' / m / 'final.json' for m in C.MODEL_NAMES]
    absent += [C.DOC / f'REJECTED_{m}.json' for m in C.MODEL_NAMES if m != model]
    assert all(not p.exists() for p in absent)
    rejected = C.read(C.DOC / f'REJECTED_{model}.json')
    start = C.read(folder / 'START.json')
    failed = C.read(folder / 'FAILED.json')
    assert rejected['complete'] and rejected['accepted'] is False
    assert rejected['automatic_retry'] is False
    assert failed['automatic_retry'] is False and failed['further_iterations_authorized'] is False
    assert failed['error_type'] == 'RuntimeError'
    assert failed['error'] == 'SOLVER_OR_CERTIFICATE_REJECTED; no extra iterations or new fit authorized.'
    assert rejected['START'] == C.bind(folder / 'START.json')
    assert rejected['trace'] == C.bind(folder / 'TRACE.jsonl')
    assert start['initialization'] == 'all_zero_float64'
    assert start['solver'] == protocol['solver'] and start['certificate_rule'] == protocol['certificate']
    assert start['source_TRAIN_only'] and not start['VAL_quality_read'] and not start['real_targets_read']
    assert start['loss_uses_original_valid_mask'] and start['lambda_l2'] == 1e-4 and start['bias'] == 0.
    assert start['names'] == ['R0:long-face-front', 'R0:short-face-front']

    basis_binding = C.bind(C.RBF_DOC / 'RBF_BASIS.json')
    assert protocol['inputs']['rbf_basis'] == basis_binding
    basis_artifact = C.read(C.RBF_DOC / 'RBF_BASIS.json')
    assert basis_artifact['complete'] and basis_artifact['PASS'] and not basis_artifact['labels_read']
    assert basis_artifact['protocol'] == protocol['inputs']['basis_protocol']
    basis = basis_artifact['basis']
    prefit = C.read(C.DOC / 'PREFIT_REVIEW.json')
    assert prefit['complete'] and prefit['PASS'] and prefit['basis_SHA_bind'] == basis_binding
    assert protocol['inputs']['prefit_review'] == C.bind(C.DOC / 'PREFIT_REVIEW.json')
    codebindings = {b['path']: b for b in protocol['codes']}
    for module in (C, T, OLD, T.B):
        actual = C.bind(module.__file__)
        assert codebindings[actual['path']] == actual

    parent = OLD.load_training_inputs()
    anchor, index = V.load_anchor(parent)
    assert len(parent['ids']) == 2598 and parent['ids'][index < 0].tolist() == ['TEX__shard_04_f0110']
    assert OLD.array_sha(index) == basis_artifact['anchor_index_sha']
    assert OLD.array_sha(parent['ids']) == start['source_ids_sha'] == basis_artifact['source_ids_sha']
    assert OLD.array_sha(parent['source_index']) == basis_artifact['source_index_sha']
    assert OLD.array_sha(anchor) == start['anchor_errors_sha']
    assert parent['normalization_sha'] == start['normalization_sha'] == basis_artifact['normalization_sha']
    assert start['axis_order'] == ['translation_cm', 'rotation_deg']
    np.testing.assert_array_equal(start['signed_target_scales'], parent['scale'])
    raw, valid, errors = (parent[k]['R0'] for k in ('features', 'valid', 'errors'))
    phi189 = V.independent_context(raw, valid, index, parent['mean'], parent['std'])
    phi = V.independent_rbf(phi189, valid, basis)
    x = I.independent_difference(phi, valid, index)
    with np.errstate(all='raise'):
        scaled, target = I.independent_targets(errors, valid, anchor, index, parent['scale'])
    hashes = dict(signed_target_sha=OLD.array_sha(target), input_difference_sha=OLD.array_sha(x),
        base_context_sha=OLD.array_sha(phi), errors_sha=OLD.array_sha(errors),
        scaled_excess_sha=OLD.array_sha(scaled), original_valid_sha=OLD.array_sha(valid))
    for obj in (start, rejected, failed):
        assert obj['model'] == model and obj['protocol'] == binding
        assert obj['basis_SHA_bind'] == basis_binding
        assert obj['anchor_index_sha'] == OLD.array_sha(index)
        for key, value in I.METADATA.items():
            assert obj[key] == value
        for key in I.HASH_KEYS:
            assert obj[key] == hashes[key] == prefit['models'][model][key]
    assert start['train_rows'] == 2598 and start['no_valid_rows'] == int((~valid.any(1)).sum()) == 1
    assert start['valid_candidates'] == int(valid.sum()) == 5194
    assert not x[~valid].any() and not target[~valid].any()
    np.testing.assert_array_equal(x, T.difference_inputs(raw, valid, index, parent['mean'], parent['std'], basis))
    scaled_native, target_native = T.signed_targets(errors, valid, anchor, index, parent['scale'])
    np.testing.assert_array_equal(scaled, scaled_native)
    np.testing.assert_array_equal(target, target_native)

    weight = np.asarray(rejected['final_weight'], np.float64)
    assert weight.shape == (253, 2) and np.isfinite(weight).all()
    assert OLD.array_sha(weight) == rejected['final_weight_sha']
    zero_weight = np.zeros((253, 2), np.float64)
    assert OLD.array_sha(zero_weight) == start['initial_weight_sha']
    actual = I.independent_objective(weight, x, valid, target, with_hessian=True)
    native = T.objective(weight, x, valid, target)
    native_H = T.hessian(x, valid, native['residual'])
    np.testing.assert_allclose(actual['gradient'], native['gradient'], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(actual['prediction'], native['prediction'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual['Hessian'], native_H, rtol=1e-9, atol=1e-10)
    H = actual['Hessian']
    np.testing.assert_array_equal(H[0::2, 1::2], np.zeros((253, 253)))
    np.testing.assert_array_equal(H[1::2, 0::2], np.zeros((253, 253)))
    np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-12)
    eig = np.linalg.eigvalsh(H)
    assert eig[0] >= 1e-4 - 1e-10 and np.isfinite(eig).all()
    cert = rejected['certificate']
    solver = rejected['solver']
    assert cert['PASS'] is False and cert['optimizer_success'] is False
    assert cert['lambda_l2'] == 1e-4 and cert['max_gap_upper_bound'] == 1e-6
    assert cert['loss_rule'] == I.METADATA['loss_rule'] and cert['huber_delta'] == 1. and cert['output_dim'] == 2
    differences = {}
    for key, recorded in (('objective', 'objective_value'), ('Huber', 'Huber'),
            ('L2_penalty', 'L2_penalty'), ('gradient_l2', 'gradient_l2'),
            ('certified_gap_upper_bound', 'gradient_l2_squared_over_2lambda')):
        differences[key] = abs(actual[key] - cert[recorded])
        assert differences[key] <= 1e-10
    assert actual['certified_gap_upper_bound'] > 1e-6
    assert actual['gradient_linf'] > protocol['solver']['gtol']
    assert solver['success'] is False and solver['status'] == 1
    assert solver['message'] == 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
    assert solver['weight_flattening'] == 'C_order_feature_then_axis'
    assert solver['exact_huber_kink_axes'] == actual['exact_Huber_kink_scalar_count']
    np.testing.assert_allclose([eig[0], eig[-1]],
        [solver['final_hessian_min'], solver['final_hessian_max']], rtol=1e-8, atol=1e-10)

    trace = [json.loads(line) for line in (folder / 'TRACE.jsonl').read_text().splitlines()]
    calls, iterations = [], []
    for row in trace:
        for key, value in I.METADATA.items():
            assert row[key] == value
        for key in I.HASH_KEYS:
            assert row[key] == hashes[key]
        assert row['basis_SHA_bind'] == basis_binding and row['anchor_index_sha'] == OLD.array_sha(index)
        assert all(np.isfinite(row[k]) and row[k] >= 0 for k in ('objective', 'Huber', 'gradient_l2'))
        if row['event'] == 'objective':
            assert row['call'] == len(calls) + 1
            assert np.isfinite(row['L2_penalty']) and row['L2_penalty'] >= 0
            assert abs(row['objective'] - row['Huber'] - row['L2_penalty']) < 1e-12
            calls.append(row)
        else:
            assert row['event'] == 'iteration' and calls
            assert row['iteration'] == len(iterations) + 1 and row['objective_calls'] == len(calls)
            assert all(row[k] == calls[-1][k] for k in ('weight_sha', 'objective', 'Huber', 'gradient_l2'))
            iterations.append(row)
    assert len(calls) == cert['objective_calls'] == solver['objective_calls'] == failed['objective_calls'] == 1108
    assert len(iterations) == cert['iterations'] == solver['iterations'] == failed['iterations'] == 1000
    assert len(calls) < protocol['solver']['maxfun'] and len(iterations) == protocol['solver']['maxiter']
    assert calls[0]['weight_sha'] == start['initial_weight_sha']
    assert calls[-1]['weight_sha'] == iterations[-1]['weight_sha'] == rejected['final_weight_sha']
    assert calls[-1]['objective'] == cert['objective_value'] and calls[-1]['Huber'] == cert['Huber']
    zero = I.independent_objective(zero_weight, x, valid, target)
    for key, recorded in (('objective', 'objective'), ('Huber', 'Huber'), ('gradient_l2', 'gradient_l2')):
        assert abs(zero[key] - calls[0][recorded]) <= 1e-10
    accepted_increases = sum(b['objective'] > a['objective'] + 1e-12 for a, b in zip(iterations, iterations[1:]))
    assert accepted_increases == 0
    # No argmin, score_candidates, physical selected errors, or optimizer calls.
    source_bindings = [binding, C.bind(C.DOC / 'TRAIN_PROTOCOL_SHA.json'),
        C.bind(C.DOC / f'REJECTED_{model}.json'), C.bind(folder / 'START.json'),
        C.bind(folder / 'TRACE.jsonl'), C.bind(folder / 'FAILED.json'),
        C.bind(C.DOC / 'PREFIT_REVIEW.json'), basis_binding,
        C.bind(__file__), C.bind(I.__file__), C.bind(V.__file__), C.bind(T.__file__)]
    out = dict(complete=True, PASS=True, verification_PASS=True,
        optimizer_certificate_PASS=False, accepted_checkpoint=False,
        status='INDEPENDENT_REJECTED_STATE_VERIFIED_OPTIMIZER_CERTIFICATE_FAIL',
        created_at=C.now(), model=model, protocol=binding, **I.METADATA,
        independent_recompute=I.public_objective(actual), original_certificate=cert,
        original_solver=solver, absolute_differences=differences,
        gradient_vector_max_abs_difference=float(np.abs(actual['gradient'] - native['gradient']).max()),
        Hessian=dict(dimension=506, min=float(eig[0]), max=float(eig[-1]),
            condition_number=float(eig[-1] / eig[0]),
            maximum_absolute_difference=float(np.abs(H - native_H).max()),
            exact_kink_count=actual['exact_Huber_kink_scalar_count'],
            cross_axis_blocks_exact_zero=True, positive_definite=True,
            convention='Zero data curvature at |residual|==1, plus lambda I. C-order feature/axis flattening.'),
        hashes=hashes, basis_SHA_bind=basis_binding,
        parameters=dict(shape=[253, 2], count=506, final_weight_sha=rejected['final_weight_sha'],
            role='Rejected optimizer diagnostic state; not an accepted or operational checkpoint.'),
        trace=dict(rows=len(trace), objective_calls=len(calls), iterations=len(iterations),
            max_objective_calls=2000, max_iterations=1000, call_budget_reached=False,
            iteration_budget_reached=True, accepted_objective_increase_count=accepted_increases,
            initial_objective=zero['objective'], final_objective=actual['objective'],
            objective_at_iteration_900=iterations[899]['objective'],
            last_100_iterations_objective_decrease=iterations[899]['objective'] - iterations[-1]['objective'],
            iteration_rows_match_preceding_objective=True,
            intermediate_values_scope='Metadata/count/order/hash/log arithmetic checked for every trace row; only initial and terminal numerical objectives independently recomputed because intermediate weights are not stored.'),
        ledger=dict(attempted_fits=1, rejected_fits=1, accepted_fits=0,
            union_fits=0, retries=0, certified_model_count=0, TRAINING_COMPLETE_exists=False,
            absent_artifacts=[str(p.relative_to(C.ROOT)) for p in absent]),
        training_rows=2598, available_rows=2597, all_invalid_rows=1, valid_candidates=5194,
        full_denominator_retained=True, source_TRAIN_only=True,
        new_fits=0, optimizer_steps=0, selected_pose_computations=0,
        selected_T_R_quality_computations=0, VAL_quality_read=False, real_targets_read=False,
        raw_reference_reads=0, performance_claim=False, method_success=False, goal_complete=False,
        interpretation=dict(gap_bound_is_upper_bound=True,
            failed_upper_bound_does_not_prove_actual_gap_exceeds_tolerance=True,
            high_condition_number_is_observed_not_causal_proof=True,
            source_and_real_quality_not_measured=True),
        next_numerical_intervention=dict(status='PROPOSAL_ONLY_NOT_EXECUTED',
            separate_namespace_required=True, solver='Block generalized Newton with fixed Armijo backtracking',
            unchanged=['objective', 'lambda1e-4', '253 features', 'signed targets', 'full2598 denominator',
                'zero initialization', '1000 accepted iterations', '2000 objective calls', 'source45 and real5 criteria'],
            direction='Solve each SPD253 axis block H_a p_a=-gradient_a; no change to ridge or targets.',
            line_search=dict(start_alpha=1., halving=.5, armijo_c1=1e-4),
            success='Gradient linf<=1e-8 AND strong-convex gap upper bound<=1e-6; use the already evaluated accepted point.',
            accounting='Initial evaluation and every line-search trial count toward the unchanged2000 objective-call cap; stop on exhausted budget, non-descent/nonfinite state, or failed solve. No warm start from this rejected state.',
            rationale='Observed large terminal Hessian condition and continuing logged objective decrease justify a curvature-aware numerical control; they do not establish the unique cause of failure or predict T/R success.'),
        bindings=source_bindings, read_paths=sorted(set(OLD.READS)),
        wall_seconds=time.monotonic() - started)
    C.save(C.DOC / 'REJECTED_FIT_VERIFICATION_KO.md', report(out))
    C.save(C.DOC / 'REJECTED_FIT_VERIFICATION.json', out)
    print('SIGNED_AXES_REJECTED_STATE_VERIFICATION_PASS_CERTIFICATE_FAIL',
        json.dumps(dict(objective=actual['objective'], gradient_l2=actual['gradient_l2'],
            gap=actual['certified_gap_upper_bound'], condition=out['Hessian']['condition_number'],
            seconds=out['wall_seconds'])), flush=True)


def report(x):
    q=x['independent_recompute']; h=x['Hessian']; t=x['trace']
    return '\n'.join([
        '# 중단된 첫 signed-axis 학습의 독립 검산', '',
        '**수치·기록 검산 PASS, optimizer 수렴 인증 FAIL.** 첫 R0_ONLY 학습은 사전 1,000회 반복 한도에서 중단됐다. UNION 학습은 시작하지 않았고, 승인된 checkpoint/FIT/TRAINING_COMPLETE가 없다. 실패 JSON의 506개 가중치는 중단 상태를 검산하기 위한 값이며 운영 모델이나 최종 성능 checkpoint가 아니다.', '',
        '| 항목 | 독립 계산 |', '|---|---:|',
        f"| 전체 목적함수 J | {q['objective']:.17g} |",
        f"| Huber 항 | {q['Huber']:.17g} |",
        f"| L2 항 | {q['L2_penalty']:.17g} |",
        f"| gradient L2 | {q['gradient_l2']:.17g} |",
        f"| gradient Linf | {q['gradient_linf']:.17g} |",
        f"| 강볼록 gap 상한 | {q['certified_gap_upper_bound']:.17g} |",
        f"| Hessian 최소/최대 고유값 | {h['min']:.12g} / {h['max']:.12g} |",
        f"| Hessian 조건수 | {h['condition_number']:.12g} |",
        f"| 반복 / objective 호출 | {t['iterations']} / {t['objective_calls']} |", '',
        '고정 TRAIN 2,598행의 raw94→context189→RBF253→anchor 차분과 signed-log1p 두 축 target을 독립 재구성했다. 원래 유효 후보를 유지하여 각 frame에서 후보×두 축 Huber(delta1)를 평균한 뒤 전체 2,598행으로 평균하고 λ=1e−4 ridge를 더했다. 실패 1행은 유효 후보가 없어 데이터 손실0이지만 전체 분모에 남는다. Torch64 autograd와 독립 가중 design-matrix의 506×506 block Hessian을 실행 코드와 대조했다.', '',
        f"목적함수 기록 차이는 {x['absolute_differences']['objective']:.3g}, gradient 벡터 최대 차이는 {x['gradient_vector_max_abs_difference']:.3g}, Hessian 최대 차이는 {h['maximum_absolute_difference']:.3g}이다. 정확한 Huber kink(|residual|=1)는 {h['exact_kink_count']}개였다. kink가 있으면 사전 선언된 데이터 곡률0을 사용하되 λI는 유지한다. gradient-gap 상한은 Huber의 C1 강볼록성에 근거한다.", '',
        '## 중단 판정과 감사 범위', '',
        f"강볼록 gap 상한 {q['certified_gap_upper_bound']:.9g}은 계약의 1e−6 이하 조건을 만족하지 못한다. optimizer success도 false이며, 종료 사유는 `TOTAL NO. of ITERATIONS REACHED LIMIT`이다. 2,000회 objective 호출 한도에는 도달하지 않았지만 1,000회 반복 한도에 도달했으므로 그대로 종료했다. gap **상한** 초과만으로 실제 optimality gap이 1e−6보다 크다고 증명되는 것은 아니다. 이번 판정은 정해진 수렴 인증을 얻지 못했다는 뜻이다.", '',
        f"START/REJECTED/FAILED/PREFIT와 전체 {t['rows']}개 trace 행의 6개 데이터 SHA, basis·anchor SHA, 사건 순서, 호출 수와 반복 수를 대조했다. 각 iteration은 직전 objective 기록과 일치한다. 초기·중단 가중치의 목적함수는 독립 재계산했으며, 중간 가중치는 저장되지 않았으므로 중간 수치의 재계산을 주장하지 않는다. 900→1,000번째 반복의 기록상 J 감소는 {t['last_100_iterations_objective_decrease']:.12g}이다.", '',
        '실패 가중치의 argmin 선택이나 T/R 성능은 계산하지 않았다. 추가 fit/optimizer step, source VAL 또는 실사 품질 열람, raw 참조 열람은 모두0이다. source45 및 실사 양쪽 5개 조건은 미평가 상태이고, 이전 CE와 현재 Huber를 직접 비교하지 않았다.', '',
        '## 다음 수치 개입 제안 — 아직 실행하지 않음', '',
        '별도 namespace에서 **동일 목적식의 block generalized-Newton + 고정 Armijo** solver 한 가지만 바꾸는 수치 실험을 제안한다. λI가 포함된 두 253×253 SPD block을 풀어 방향을 구하고, alpha=1에서 시작해 0.5씩 줄이며 c1=1e−4 Armijo 조건을 적용한다. 비영 gradient에서 SPD Newton 방향은 descent 방향이며, Huber의 C1 성질로 backtracking을 정의할 수 있다. 기존 zero 초기화, 특징·target·λ·분모·1,000회 반복/2,000회 objective 호출 한도를 유지하고 실패 가중치로 warm start하지 않는다. 초기 평가와 모든 line-search trial을 호출 예산에 포함한다. 채택점의 기존 gradient로 Linf≤1e−8 및 gap 상한≤1e−6을 모두 확인하며, 한도 소진·비정상 수치·선형계 실패 시 중단한다.', '',
        '큰 terminal 조건수와 계속 감소한 목적함수는 곡률을 직접 사용하는 solver를 검토할 관측 근거다. 이것이 현재 실패의 유일 원인이라는 결론이나 다음 solver의 수렴·T/R 개선 보장은 아니다. 이번 namespace에서는 재시작하지 않았고, 제안의 실제 실행도 없다.', '',
        '[검산 JSON](REJECTED_FIT_VERIFICATION.json) · [중단 상태](REJECTED_R0_ONLY.json) · [고정 학습 계약](TRAIN_PROTOCOL.json) · [학습 전 검산](PREFIT_REVIEW_KO.md)', ''
    ])


def selfcheck():
    # The existing independent verifier's analytic fixtures never load artifacts.
    I.selfcheck()
    print('REJECTED_AUDIT_INDEPENDENT_HELPER_TOYS_PASS_NO_FIT', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('selfcheck', 'verify'))
    with threadpool_limits(limits=1):
        (selfcheck if parser.parse_args().stage == 'selfcheck' else main)()
