"""Independent torch-autograd check of frozen convex TRAIN fits; never fits."""
from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits


def independent_objective(weight, x, valid, target):
    # Separate implementation from convex_train's NumPy analytic derivatives.
    w = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    xx, vv = torch.tensor(x, dtype=torch.float64), torch.tensor(valid, dtype=torch.bool)
    yy = torch.tensor(target, dtype=torch.int64)
    present = vv.any(1)
    scores = torch.einsum('nkd,d->nk', xx, w)
    ce = F.cross_entropy((-scores[present]).masked_fill(~vv[present], -torch.inf), yy[present], reduction='sum') / len(xx)
    penalty = .5 * 1e-4 * torch.sum(w * w)
    value = ce + penalty
    gradient, = torch.autograd.grad(value, w)
    gradient = gradient.detach().numpy()
    picks = OLD.select_candidates(scores.detach().numpy(), valid, ['R0:long-face-front', 'R0:short-face-front'] +
                                 (['DIVERSE:long-face-front', 'DIVERSE:short-face-front'] if x.shape[1] == 4 else []))
    norm = float(np.linalg.norm(gradient))
    return dict(CE=float(ce.detach()), L2_penalty=float(penalty.detach()), objective=float(value.detach()),
                gradient_l2=norm, gradient_linf=float(np.abs(gradient).max()),
                certified_gap_upper_bound=norm ** 2 / (2e-4),
                target_accuracy_valid_rows=float(np.mean(picks[present.numpy()] == target[present.numpy()])),
                frames=len(xx), valid_target_rows=int(present.sum()), no_candidate_rows=int((~present).sum()))


def main():
    torch.set_num_threads(1)
    OLD.install_training_guard()
    parent = OLD.load_training_inputs()
    protocol = C.protocol()
    binding = C.bind(C.DOC / 'PROTOCOL.json')
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['protocol'] == binding and complete['models'] == list(C.MODEL_NAMES)
    bound = [binding, C.bind(C.DOC / 'TRAINING_COMPLETE.json'), C.bind(Path(__file__))]
    records = {}
    for fit_binding in complete['fits']:
        C.verify(fit_binding)
        receipt = C.read(C.ROOT / fit_binding['path'])
        model = receipt['model']
        assert model in C.MODEL_NAMES and model not in records
        assert receipt['complete'] and receipt['protocol'] == binding
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(receipt[key])
        ck = C.read(C.ROOT / receipt['checkpoint']['path'])
        start = C.read(C.ROOT / receipt['START']['path'])
        assert ck['protocol'] == start['protocol'] == binding
        assert ck['model'] == start['model'] == model and ck['bias'] == 0. and ck['lambda_l2'] == 1e-4
        assert ck['normalization_sha'] == receipt['normalization_sha'] == parent['normalization_sha']
        mean, std = np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)
        np.testing.assert_array_equal(mean, parent['mean'])
        np.testing.assert_array_equal(std, parent['std'])
        seed = 1 if model == 'R0_ONLY' else int(model[-1])
        arm = 'R0_ONLY' if model == 'R0_ONLY' else 'UNION'
        names = OLD.candidate_names(arm, seed)
        assert ck['names'] == names
        sources = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{seed}'])
        raw, valid, errors = [np.concatenate([parent[key][source] for source in sources], axis=1)
                              for key in ('features', 'valid', 'errors')]
        target = OLD.targets(errors, valid, parent['scale'], names)
        normalized32 = np.zeros_like(raw, dtype=np.float32)
        normalized32[valid] = (raw[valid] - mean) / std
        x = normalized32.astype(np.float64)
        assert OLD.array_sha(x) == start['normalized_features_sha']
        assert OLD.array_sha(target) == start['target_sha']
        assert OLD.array_sha(valid) == start['valid_mask_sha']
        weight = np.asarray(ck['weight'], np.float64)
        assert OLD.array_sha(weight) == receipt['final_weight_sha']
        actual = independent_objective(weight, x, valid, target)
        saved = ck['certificate']
        assert saved == receipt['certificate'] and saved['PASS'] and saved['optimizer_success']
        assert actual['certified_gap_upper_bound'] <= 1e-6
        assert abs(actual['objective'] - saved['objective_value']) < 1e-10
        assert abs(actual['gradient_l2'] - saved['gradient_l2']) < 1e-10
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls = [r for r in trace if r['event'] == 'objective']
        iterations = [r for r in trace if r['event'] == 'iteration']
        assert len(calls) == receipt['objective_calls'] == saved['objective_calls'] <= 2000
        assert len(iterations) == receipt['iterations'] == saved['iterations'] <= 1000
        assert [r['call'] for r in calls] == list(range(1, len(calls) + 1))
        assert [r['iteration'] for r in iterations] == list(range(1, len(iterations) + 1))
        assert calls[-1]['weight_sha'] == receipt['final_weight_sha']
        assert calls[-1]['objective'] == saved['objective_value']
        old = {}
        for old_seed in (1, 2, 3) if model == 'R0_ONLY' else (seed,):
            old_name = f'{arm}_s{old_seed}'
            old_receipt_path = C.PARENT_DOC / f'FIT_{old_name}.json'
            old_receipt = C.read(old_receipt_path)
            C.verify(old_receipt['checkpoint'])
            old_ck = torch.load(C.ROOT / old_receipt['checkpoint']['path'], map_location='cpu', weights_only=False)
            old_weight = old_ck['state']['net.weight'].numpy().reshape(-1).astype(np.float64)
            old[old_name] = independent_objective(old_weight, x, valid, target)
            old[old_name]['CE_change_new_minus_old'] = actual['CE'] - old[old_name]['CE']
            old[old_name]['ridge_objective_change_new_minus_old'] = actual['objective'] - old[old_name]['objective']
            bound += [C.bind(old_receipt_path), old_receipt['checkpoint']]
        records[model] = dict(independent_recompute=actual, original_certificate=saved,
            objective_abs_difference=abs(actual['objective'] - saved['objective_value']),
            gradient_norm_abs_difference=abs(actual['gradient_l2'] - saved['gradient_l2']),
            objective_calls=len(calls), iterations=len(iterations), old_same_pool_weights=old,
            last10_iterations=iterations[-10:])
        bound += [fit_binding, receipt['checkpoint'], receipt['trace'], receipt['START']]
    assert set(records) == set(C.MODEL_NAMES)
    out = dict(complete=True, independent_check_PASS=True, created_at=C.now(), models=records,
               protocol=binding, bindings=bound, source_TRAIN_only=True, VAL_quality_read=False,
               real_targets_read=False, new_fits=0, optimizer_steps=0,
               implementation='PyTorch float64 cross_entropy/autograd, independently checked against NumPy analytic training objective/gradient.',
               normalization='Exact original float32 normalization, then cast to float64; old weights evaluated on same input and target. Common bias omitted because it cancels.',
               caveat='Explicit ridge objective differs from old AdamW decoupled decay; convergence is not proof of lower T/R or source/real gate PASS.',
               read_paths=sorted(set(OLD.READS)))
    C.save(C.DOC / 'TRAIN_CONVERGENCE.json', out)
    lines = ['# 凸 scorer의 TRAIN 수렴 독립 검증', '',
             '**네 fit 모두 명시적 CE+ridge objective의 수렴 인증을 독립 재계산으로 통과했다.** 이는 T/R 개선 판정이 아니다.', '',
             '[검증 JSON](TRAIN_CONVERGENCE.json)은 원래 입력·각 checkpoint·START·TRACE·protocol hash를 연결한다. 저장 weight를 수정하지 않고 별도 PyTorch float64 cross-entropy/autograd로 objective와 gradient를 계산했다. 새 fit, optimizer step, VAL quality 또는 실사 GT 읽기는0회다.', '',
             '| 모델 | CE | L2 penalty | objective | gradient L2 | objective gap 상한 | iterations / calls |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for model, record in records.items():
        a = record['independent_recompute']
        lines.append(f"| {model} | {a['CE']:.8f} | {a['L2_penalty']:.8f} | {a['objective']:.8f} | {a['gradient_l2']:.3g} | {a['certified_gap_upper_bound']:.3g} | {record['iterations']} / {record['objective_calls']} |")
    lines += ['', 'objective는2598행 전체의 CE 평균에 `0.5×1e−4×||w94||²`를 더한다. 모든 fit에서 후보 없는1행도 원래 분모에 남는다. bias는 순위에서 상쇄되어0으로 고정했다. 명시적 강볼록 계수λ=1e−4에 대해 `J(w)−J* ≤ ||∇J(w)||²/(2λ)`이며, 네 결과 모두 사전 기준1e−6보다 작다. optimizer success와1000 iterations/2000 objective closure 상한도 함께 확인했다.', '',
              '| 원래 AdamW weight → 대응 수렴 모델 | 동일 입력의 old CE | new CE | CE 변화 | 명시적 ridge objective 변화 |',
              '|---|---:|---:|---:|---:|']
    for model, record in records.items():
        for old_name, old in record['old_same_pool_weights'].items():
            new = record['independent_recompute']
            lines.append(f"| {old_name} → {model} | {old['CE']:.8f} | {new['CE']:.8f} | {old['CE_change_new_minus_old']:+.8f} | {old['ridge_objective_change_new_minus_old']:+.8f} |")
    max_obj = max(r['objective_abs_difference'] for r in records.values())
    max_grad = max(r['gradient_norm_abs_difference'] for r in records.values())
    lines += ['', f'저장 certificate와 독립 계산의 최대 objective 차이는{max_obj:.3g}, gradient norm 차이는{max_grad:.3g}다. 모든 trace call/iteration 번호가 연속이고 최종 weight hash가 checkpoint와 일치한다.', '',
              '이 비교에서 old weight도 정확히 같은 float32 정규화 입력을 float64로 올린 함수에 적용했다. 원래 실사 예측을 수정하거나 다른 seed를 선택하지 않았다. R0_ONLY는 하나의 결정론적 공통 control이고, UNION의 세 경우는 기존 동결 refiner3개를 각각 유지한다.', '',
              '이전6개 모델의 CE가 더 낮아질 수 있었음은 확인된다. 따라서 원래 실패를 Linear94의 표현력 한계만으로 설명해서는 안 된다. 다만 이번 통제는 optimizer·precision·초기값과 명시적 ridge를 바꾸므로 순수하게 optimizer 하나만의 인과 효과를 식별한 실험도 아니다. 기존 AdamW의 decoupled weight decay와 새로운 ridge objective를 같은 목적함수라고 부르면 안 된다.', '',
              '**실질적인 T/R 판단은 별도 source VAL과, 그 통과 이후에만 허용되는 기존 실사 gate에서 해야 한다.** 수렴 인증이나 TRAIN CE 감소가 중앙값·tail·recording 안정성 개선을 대신하지 않는다.', '']
    C.save(C.DOC / 'TRAIN_CONVERGENCE_KO.md', '\n'.join(lines))
    print('INDEPENDENT_TRAIN_CONVERGENCE_PASS_ALL4', flush=True)
    for model, record in records.items():
        print(model, record['independent_recompute'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
