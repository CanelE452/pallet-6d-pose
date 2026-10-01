"""Independent TRAIN189 audit; no fitting, VAL quality, or real references."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from . import common as C
from scripts.research.pallet_pose_pareto_anchor_20261001_v1.source_feasibility import independent_targets, summary
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD

FEATURE_MAP = 'normalized94_abs_anchor_delta94_identity1'
FEATURE_DIM, RAW_FEATURE_DIM = 189, 94
CHECKPOINT_SCHEMA = 'pallet_pose_anchor_context_linear189_v1'


def independent_context(raw, valid, anchor_index, mean, std):
    """Separate scalar-row construction; never call the trainer's feature map."""
    raw, valid, anchor_index = np.asarray(raw), np.asarray(valid), np.asarray(anchor_index)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    assert raw.dtype == np.float32 and valid.dtype == bool
    assert raw.shape == (*valid.shape, RAW_FEATURE_DIM) and valid.shape[1] in (2, 4)
    assert mean.shape == std.shape == (RAW_FEATURE_DIM,) and np.isfinite(mean).all()
    assert np.isfinite(std).all() and (std >= np.float32(1e-6)).all()
    assert anchor_index.shape == (len(raw),) and np.issubdtype(anchor_index.dtype, np.integer)
    out = np.zeros((*valid.shape, FEATURE_DIM), np.float64)
    for row in range(len(raw)):
        available = np.flatnonzero(valid[row])
        if not len(available):
            assert anchor_index[row] == -1
            continue
        anchor = int(anchor_index[row])
        assert anchor in (0, 1) and valid[row, anchor]
        normalized = {}
        for candidate in available:
            assert np.isfinite(raw[row, candidate]).all()
            # Both ufunc inputs are float32, exactly as in the frozen94 path.
            value32 = np.divide(np.subtract(raw[row, candidate], mean), std)
            assert value32.dtype == np.float32 and np.isfinite(value32).all()
            normalized[int(candidate)] = value32.astype(np.float64)
        for candidate in available:
            z = normalized[int(candidate)]
            out[row, candidate, :94] = z
            out[row, candidate, 94:188] = np.abs(z - normalized[anchor])
            out[row, candidate, 188] = float(candidate == anchor)
    assert np.isfinite(out).all() and not np.any(out[~valid])
    return out


def select(scores, valid, names):
    output = np.full(len(scores), -1, np.int64)
    for row in range(len(scores)):
        indices = np.flatnonzero(valid[row])
        if len(indices):
            output[row] = min(indices, key=lambda j: (scores[row, j], names[j].split(':')[0] != 'R0',
                                                     names[j].split(':')[1], names[j].split(':')[0]))
    return output


def independent_objective(weight, x, valid, target):
    """Separate Torch CE/autograd implementation; all frames in denominator."""
    weight = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    xx, vv = torch.tensor(x, dtype=torch.float64), torch.tensor(valid, dtype=torch.bool)
    yy = torch.tensor(target, dtype=torch.int64)
    present = vv.any(1)
    assert torch.equal(yy >= 0, present)
    scores = torch.einsum('nkd,d->nk', xx, weight)
    ce = scores.sum() * 0.
    if present.any():
        logits = (-scores[present]).masked_fill(~vv[present], -torch.inf)
        ce = F.cross_entropy(logits, yy[present], reduction='sum') / len(xx)
    penalty = .5e-4 * weight.square().sum()
    objective = ce + penalty
    gradient, = torch.autograd.grad(objective, weight)
    grad = gradient.detach().numpy()
    norm = float(np.linalg.norm(grad))
    return dict(CE=float(ce.detach()), L2_penalty=float(penalty.detach()), objective=float(objective.detach()),
                gradient_l2=norm, gradient_linf=float(np.abs(grad).max()),
                certified_gap_upper_bound=norm * norm / (2e-4), scores=scores.detach().numpy(), gradient=grad)


def score_statistics(scores, valid, target, safe, errors, anchor, names):
    picked = select(scores, valid, names)
    present = valid.any(1)
    assert np.array_equal(picked >= 0, present) and np.array_equal(target >= 0, present)
    rows = np.flatnonzero(present)
    selected_errors = np.full((len(errors), 2), np.inf)
    selected_errors[rows] = errors[rows, picked[rows]]
    assert np.isfinite(anchor[rows]).all() and np.isfinite(selected_errors[rows]).all()
    above = selected_errors[rows] > anchor[rows]
    assert np.array_equal(~safe[rows, picked[rows]], above.any(1))
    total = len(rows)
    violations = {}
    for name, mask in (('T', above[:, 0]), ('R', above[:, 1]), ('either', above.any(1)), ('both', above.all(1))):
        count = int(mask.sum())
        violations[name] = dict(count=count, rate=count / len(errors),
                               rate_available_rows=count / total if total else None)
    return dict(frames=len(errors), available_anchor_rows=total, failed_rows=int((~present).sum()),
                denominators=dict(full_population=len(errors), available_anchor_rows=total),
                target_correct=int((picked[rows] == target[rows]).sum()),
                target_accuracy_full_population=float((picked[rows] == target[rows]).sum() / len(errors)),
                target_accuracy_available_rows=float(np.mean(picked[rows] == target[rows])) if total else None,
                anchor_violations=violations, selected_error_summary=summary(selected_errors),
                selected_index_sha=OLD.array_sha(picked),
                denominator='Primary accuracy/violation fractions retain all2598 TRAIN rows; invalid target-1 is not counted correct. Conditional rates also reported on2597 available anchors; one failure is explicit and retained as+inf in full-frame T/R summaries.'), picked


def public_objective(value):
    return {k: v for k, v in value.items() if k not in ('scores', 'gradient')}


def load_anchor(parent):
    """Independent selection of frozen TRAIN-only fields; no raw GT load."""
    bindings = {k: parent['protocol']['inputs'][k] for k in ('features', 'train_labels')}
    for binding in bindings.values():
        C.verify(binding)
    with np.load(C.ROOT / bindings['features']['path'], allow_pickle=False) as z:
        indices = z['R0_GEO_index'][parent['source_index']].copy()
        np.testing.assert_array_equal(z['ids'][parent['source_index']], parent['ids'])
    with np.load(C.ROOT / bindings['train_labels']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], parent['ids'])
        errors = np.stack([z['R0_GEO_T_cm'], z['R0_GEO_R_deg']], -1)
    assert indices.dtype == np.int64 and errors.dtype == np.float64
    finite = np.isfinite(errors).all(1)
    assert indices.shape == (2598,) and errors.shape == (2598, 2)
    assert np.array_equal(indices >= 0, finite) and np.isin(indices, [-1, 0, 1]).all()
    assert finite.sum() == 2597 and np.isposinf(errors[~finite]).all()
    rows = np.flatnonzero(finite)
    np.testing.assert_array_equal(parent['errors']['R0'][rows, indices[rows]], errors[rows])
    assert parent['valid']['R0'][rows, indices[rows]].all()
    return errors, indices


def main():
    torch.set_num_threads(1)
    OLD.install_training_guard()
    protocol = C.protocol('TRAIN_PROTOCOL')
    binding = C.bind(C.DOC / 'TRAIN_PROTOCOL.json')
    assert protocol['raw_feature_dim'] == RAW_FEATURE_DIM and protocol['feature_dim'] == FEATURE_DIM
    assert protocol['feature_map'] == FEATURE_MAP and protocol['normalization'] == 'old_float32_then_float64'
    assert protocol['lambda_l2'] == 1e-4 and protocol['bias'] == 0 and protocol['max_fits'] == 4
    assert protocol['solver'] == dict(method='L-BFGS-B', maxiter=1000, maxfun=2000, ftol=1e-15, gtol=1e-8, bounds=None)
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['protocol'] == binding and complete['models'] == list(C.MODEL_NAMES)
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    for key in ('raw_feature_dim', 'feature_dim', 'feature_map'):
        assert complete[key] == protocol[key]
    assert len(complete['fits']) == 4
    parent = OLD.load_training_inputs()
    anchor, anchor_index = load_anchor(parent)
    feasibility = C.read(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json')
    assert feasibility['complete'] and feasibility['PASS'] and feasibility['frames'] == 2598
    assert protocol['inputs']['source_feasibility'] == C.bind(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json')
    assert feasibility['protocol'] == protocol['inputs']['source_protocol']
    assert feasibility['anchor']['errors_sha'] == OLD.array_sha(anchor)
    assert feasibility['anchor']['index_sha'] == OLD.array_sha(anchor_index)
    assert complete['anchor_index_sha'] == OLD.array_sha(anchor_index)
    previous_complete = C.read(C.ANCHOR_DOC / 'TRAINING_COMPLETE.json')
    assert previous_complete['complete'] and previous_complete['all_certified'] and previous_complete['fit_count'] == 4
    C.verify(previous_complete['protocol'])
    assert previous_complete['protocol'] == C.bind(C.ANCHOR_DOC / 'TRAIN_PROTOCOL.json')
    assert previous_complete['source_feasibility'] == protocol['inputs']['source_feasibility']
    assert previous_complete['target_rule'] == protocol['target_rule']
    old_receipts = {}
    for b in previous_complete['fits']:
        C.verify(b)
        receipt = C.read(C.ROOT / b['path'])
        assert receipt['complete'] and receipt['protocol'] == previous_complete['protocol']
        assert b == C.bind(C.ANCHOR_DOC / f"FIT_{receipt['model']}.json")
        old_receipts[receipt['model']] = (b, receipt)
    assert set(old_receipts) == set(C.MODEL_NAMES)
    from . import convex_train as T
    codebindings = {v['path']: v for v in protocol['codes']}
    for module in (T, C, OLD):
        actual_binding = C.bind(Path(module.__file__))
        assert codebindings[actual_binding['path']] == actual_binding
    bounds = [binding, C.bind(C.DOC / 'TRAINING_COMPLETE.json'), C.bind(Path(__file__)),
              C.bind(Path(T.__file__)), C.bind(Path(independent_targets.__code__.co_filename)),
              C.bind(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json'), C.bind(C.ANCHOR_DOC / 'TRAINING_COMPLETE.json'),
              previous_complete['protocol']]
    records = {}
    for fit_binding in complete['fits']:
        C.verify(fit_binding)
        receipt = C.read(C.ROOT / fit_binding['path'])
        model = receipt['model']
        assert model in C.MODEL_NAMES and model not in records
        assert fit_binding == C.bind(C.DOC / f'FIT_{model}.json')
        assert receipt['complete'] and receipt['protocol'] == binding
        assert receipt['fits_executed'] == 1
        assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(receipt[key])
        assert receipt['checkpoint'] == C.bind(C.RAW / 'fits' / model / 'final.json')
        ck, start = [C.read(C.ROOT / receipt[k]['path']) for k in ('checkpoint', 'START')]
        assert ck['schema'] == CHECKPOINT_SCHEMA
        for key in ('raw_feature_dim', 'feature_dim', 'feature_map'):
            assert start[key] == ck[key] == receipt[key] == protocol[key]
        assert ck['anchor_index_sha'] == receipt['anchor_index_sha'] == start['anchor_index_sha'] == complete['anchor_index_sha']
        assert ck['protocol'] == start['protocol'] == binding
        assert ck['model'] == start['model'] == model and ck['bias'] == 0 and ck['lambda_l2'] == 1e-4
        assert not ck['runtime_safe_mask'] and ck['loss_uses_original_valid_mask']
        assert ck['target_rule'] == receipt['target_rule'] == start['target_rule'] == protocol['target_rule']
        assert start['source_feasibility'] == receipt['source_feasibility'] == protocol['inputs']['source_feasibility']
        seed = 1 if model == 'R0_ONLY' else int(model[-1])
        names = OLD.candidate_names('R0_ONLY' if model == 'R0_ONLY' else 'UNION', seed)
        sources = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{seed}'])
        raw, valid, errors = [np.concatenate([parent[key][m] for m in sources], axis=1)
                              for key in ('features', 'valid', 'errors')]
        target, safe = independent_targets(errors, valid, anchor, anchor_index, parent['scale'], names)
        assert ck['names'] == names and len(raw) == 2598
        assert np.count_nonzero(~valid.any(1)) == 1 and np.array_equal(target >= 0, valid.any(1))
        for key, array in (('target_sha', target), ('safe_mask_sha', safe)):
            assert OLD.array_sha(array) == start[key] == receipt[key] == feasibility['models'][model][key]
        for key, array in (('anchor_errors_sha', anchor), ('anchor_index_sha', anchor_index)):
            assert OLD.array_sha(array) == start[key] == feasibility['models'][model][key]
        assert OLD.array_sha(valid) == start['valid_mask_sha'] == feasibility['models'][model]['original_valid_sha']
        assert OLD.array_sha(errors) == feasibility['models'][model]['unscaled_errors_sha']
        assert start['no_valid_rows'] == 1 and start['initial_weight_sha'] == OLD.array_sha(np.zeros(FEATURE_DIM, np.float64))
        assert start['train_rows'] == 2598 and start['source_ids_sha'] == OLD.array_sha(parent['ids'])
        assert start['initialization'] == 'all_zero_float64' and start['lambda_l2'] == 1e-4 and start['bias'] == 0
        assert start['solver'] == protocol['solver'] and start['certificate_rule'] == protocol['certificate']
        mean, std = np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)
        np.testing.assert_array_equal(mean, parent['mean'])
        np.testing.assert_array_equal(std, parent['std'])
        assert ck['normalization_sha'] == receipt['normalization_sha'] == parent['normalization_sha']
        assert mean.shape == std.shape == (RAW_FEATURE_DIM,) and ck['normalization'] == 'old_float32_then_float64'
        x = independent_context(raw, valid, anchor_index, mean, std)
        assert OLD.array_sha(x) == start['normalized_features_sha']
        np.testing.assert_array_equal(x, T.context_inputs(raw, valid, anchor_index, mean, std))
        assert not np.any(x[~valid]) and np.all(x[np.flatnonzero(valid.any(1)), anchor_index[valid.any(1)], 94:188] == 0)
        assert np.array_equal(x[:, :, 188].sum(1), valid.any(1).astype(np.float64))
        weight = np.asarray(ck['weight'], np.float64)
        assert weight.shape == (FEATURE_DIM,) and np.isfinite(weight).all()
        assert OLD.array_sha(weight) == receipt['final_weight_sha']
        actual = independent_objective(weight, x, valid, target)
        native = T.objective(weight, x, valid, target)
        gradient_max_difference = float(np.abs(actual['gradient'] - native['gradient']).max())
        assert gradient_max_difference <= 1e-10
        cert = ck['certificate']
        assert cert == receipt['certificate'] and cert['PASS'] and cert['optimizer_success']
        assert cert['lambda_l2'] == 1e-4 and cert['max_gap_upper_bound'] == 1e-6
        assert actual['certified_gap_upper_bound'] <= 1e-6
        assert abs(actual['objective'] - cert['objective_value']) <= 1e-10
        assert abs(actual['gradient_l2'] - cert['gradient_l2']) <= 1e-10
        assert abs(actual['CE'] - cert['CE']) <= 1e-10
        assert abs(actual['L2_penalty'] - cert['L2_penalty']) <= 1e-10
        assert abs(actual['certified_gap_upper_bound'] - cert['gradient_l2_squared_over_2lambda']) <= 1e-12
        runtime = T.score_candidates(ck, raw, valid, anchor_index)
        assert np.isposinf(runtime[~valid]).all()
        np.testing.assert_allclose(runtime[valid], actual['scores'][valid], rtol=1e-12, atol=1e-12)
        stats, chosen = score_statistics(runtime, valid, target, safe, errors, anchor, names)
        np.testing.assert_array_equal(chosen, select(actual['scores'], valid, names))
        np.testing.assert_array_equal(chosen, T.select_candidates(runtime, valid, names))
        trace = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        calls = [r for r in trace if r['event'] == 'objective']
        iterations = [r for r in trace if r['event'] == 'iteration']
        assert 0 < len(calls) == receipt['objective_calls'] == cert['objective_calls'] <= 2000
        assert len(iterations) == receipt['iterations'] == cert['iterations'] <= 1000
        assert [r['call'] for r in calls] == list(range(1, len(calls) + 1))
        assert [r['iteration'] for r in iterations] == list(range(1, len(iterations) + 1))
        assert calls[-1]['weight_sha'] == receipt['final_weight_sha'] and calls[-1]['objective'] == cert['objective_value']
        assert abs(calls[0]['objective'] - np.log(valid.sum(1)[valid.any(1)]).sum() / len(valid)) <= 1e-12
        assert calls[0]['weight_sha'] == start['initial_weight_sha']
        for row in trace:
            assert row['feature_map'] == FEATURE_MAP and row['anchor_index_sha'] == complete['anchor_index_sha']
        old_binding, old_receipt = old_receipts[model]
        C.verify(old_receipt['checkpoint'])
        assert old_receipt['checkpoint'] == C.bind(C.ANCHOR_RAW / 'fits' / model / 'final.json')
        old_ck = C.read(C.ROOT / old_receipt['checkpoint']['path'])
        assert old_ck['model'] == model and old_ck['names'] == names
        assert old_ck['protocol'] == previous_complete['protocol'] and old_ck['bias'] == 0 and old_ck['lambda_l2'] == 1e-4
        assert old_ck['schema'] == 'pallet_pose_convex_linear94_v1' and old_ck['target_rule'] == ck['target_rule']
        assert old_receipt['target_sha'] == receipt['target_sha'] and old_receipt['safe_mask_sha'] == receipt['safe_mask_sha']
        assert old_ck['certificate']['PASS'] and old_ck['certificate'] == old_receipt['certificate']
        for key, dtype in (('mean', np.float32), ('std', np.float32)):
            assert np.asarray(old_ck[key], dtype).tobytes() == np.asarray(ck[key], dtype).tobytes()
        old_weight = np.asarray(old_ck['weight'], np.float64)
        assert old_weight.shape == (RAW_FEATURE_DIM,)
        padded_weight = np.concatenate([old_weight, np.zeros(FEATURE_DIM - RAW_FEATURE_DIM, np.float64)])
        previous = independent_objective(padded_weight, x, valid, target)
        old_native = independent_objective(old_weight, x[:, :, :RAW_FEATURE_DIM], valid, target)
        np.testing.assert_allclose(previous['scores'][valid], old_native['scores'][valid], rtol=1e-12, atol=1e-12)
        assert abs(previous['objective'] - old_native['objective']) <= 1e-12
        assert abs(previous['objective'] - old_ck['certificate']['objective_value']) <= 1e-10
        assert abs(old_native['gradient_l2'] - old_ck['certificate']['gradient_l2']) <= 1e-10
        from scripts.research.pallet_pose_pareto_anchor_20261001_v1 import convex_train as PRIOR
        old_runtime = PRIOR.score_candidates(old_ck, raw, valid)
        np.testing.assert_allclose(old_runtime[valid], previous['scores'][valid], rtol=1e-12, atol=1e-12)
        old_stats, old_chosen = score_statistics(old_runtime, valid, target, safe, errors, anchor, names)
        np.testing.assert_array_equal(old_chosen, select(previous['scores'], valid, names))
        records[model] = dict(independent_recompute=public_objective(actual), original_certificate=cert,
            objective_abs_difference=abs(actual['objective'] - cert['objective_value']),
            gradient_norm_abs_difference=abs(actual['gradient_l2'] - cert['gradient_l2']),
            gradient_vector_max_abs_difference=gradient_max_difference,
            objective_calls=len(calls), iterations=len(iterations), statistics=stats,
            previous_converged_same_target=dict(objective=public_objective(previous), statistics=old_stats,
                padded_weight_sha=OLD.array_sha(padded_weight), native94_objective=public_objective(old_native),
                embedding_objective_abs_difference=abs(previous['objective'] - old_native['objective']),
                expanded189_gradient_is_not_the_old94_certificate=True,
                receipt=old_binding, checkpoint=old_receipt['checkpoint']),
            changes_new_minus_old=dict(objective=actual['objective'] - previous['objective'],
                CE=actual['CE'] - previous['CE'],
                target_accuracy=stats['target_accuracy_full_population'] - old_stats['target_accuracy_full_population'],
                anchor_violation_rates={k: stats['anchor_violations'][k]['rate'] - old_stats['anchor_violations'][k]['rate']
                                        for k in ('T', 'R', 'either', 'both')}),
            original_valid_used_in_loss_and_runtime=True, target_safe_mask_hash=OLD.array_sha(safe),
            scalar_target_reconstruction_PASS=True, scalar_context_reconstruction_PASS=True,
            expanded_context_sha=OLD.array_sha(x), original_normalized94_sha=OLD.array_sha(x[:, :, :94]),
            runtime_score_and_selection_parity=True)
        bounds += [fit_binding, *(receipt[k] for k in ('START', 'trace', 'checkpoint')), old_binding, old_receipt['checkpoint']]
    assert set(records) == set(C.MODEL_NAMES)
    assert sum(r['objective_calls'] for r in records.values()) == complete['total_objective_calls']
    output = dict(complete=True, PASS=True, independent_check_PASS=True, created_at=C.now(), protocol=binding,
        models=records, bindings=bounds, source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
        new_fits=0, optimizer_steps=0, targets_reconstructed_by='source_feasibility.independent_targets scalar loop',
        implementation='Separate scalar-row float32 normalize then float64 context189, Torch float64 CE/autograd on all2598 rows, original valid CE competitors and runtime candidates.',
        feature_map=FEATURE_MAP, raw_feature_dim=RAW_FEATURE_DIM, feature_dim=FEATURE_DIM,
        comparison='Immediately preceding pareto_anchor94 weights padded with95 zeros and new context189 weights on identical anchored targets; native94 objective/score parity independently checked. Expanded189 gradient is not the old94 convergence certificate. TRAIN-only description, no tuning or model selection.',
        caveat='Reference-safe training targets do not enforce reference-safe inference. Violation rates are measured from unrestricted original-valid runtime choices; no runtime GT mask is used.',
        all_invalid_rows=1, failed_rows_retained_in_full_denominator=True, read_paths=sorted(set(OLD.READS)))
    C.save(C.DOC / 'TRAIN_CONVERGENCE.json', output)
    lines = ['# Anchor context189 학습의 독립 수렴·TRAIN 진단', '',
        '**독립 검산 PASS**. 수렴 인증은 실제 T/R 개선 판정을 대신하지 않는다.', '',
        '정답은 기존 독립 scalar-loop 구현으로 복원하고 target/safe/anchor/원래 valid hash를 START 및 이전 source feasibility와 대조했다. 94차원 float32 정규화→float64 승격→절댓값 anchor 차이94·identity1 연결은 새 독립 행별 loop로 계산했다. PyTorch float64 CE/autograd로 목적함수·gradient 벡터·수렴 상한을 검산했다. 후보 없는 1행도 전체 2,598행 분모에 남으며, CE 및 runtime에는 원래 valid 후보 전체가 참여한다.', '',
        '| 모델 | 새 CE | 새 objective | gradient L2 | gap 상한 | 호출 수 |', '|---|---:|---:|---:|---:|---:|']
    for name, record in records.items():
        a = record['independent_recompute']
        lines.append(f"| {name} | {a['CE']:.9f} | {a['objective']:.9f} | {a['gradient_l2']:.3g} | {a['certified_gap_upper_bound']:.3g} | {record['objective_calls']} |")
    lines += ['', '바로 이전 pareto_anchor의94차원 weight 뒤에95개의0을 붙여 같은 target·context189에서 비교했다. 원래94 score/objective와의 동등성도 검증했다. 추가95축에서의 gradient는 기존94 수렴 인증과 다른 공간이므로 동일시하지 않는다. 아래 정확도와 위반 비율의 분모는 전체 TRAIN 2,598행이다. 실패1행은 target 정답으로 세지 않으며, 위반 여부를 평가할 수 없는 실패로 별도 보고한다. 유효 anchor 2,597행 기준의 조건부 비율도 JSON에 함께 제공한다.', '',
        '| 모델 | target 정확도 이전→현재 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 |', '|---|---:|---:|---:|---:|']
    for name, record in records.items():
        a, b = record['previous_converged_same_target']['statistics'], record['statistics']
        cols = [f"{100*a['target_accuracy_full_population']:.3f}% → {100*b['target_accuracy_full_population']:.3f}%"]
        cols += [f"{100*a['anchor_violations'][k]['rate']:.3f}% → {100*b['anchor_violations'][k]['rate']:.3f}%" for k in ('T', 'R', 'either')]
        lines.append(f"| {name} | " + ' | '.join(cols) + ' |')
    lines += ['', '위반은 실제 unrestricted scorer가 선택한 후보의 T/R가 같은 프레임 R0 anchor를 초과하는지로 집계했다. 학습 target의 safe mask를 inference에 적용하지 않았다. 따라서 oracle의 pointwise 보존은 학습된 scorer에 자동으로 전달되는 보장이 아니다.', '',
        '이 비교는 TRAIN 설명용이며 추가 fit·optimizer step·VAL 품질·실사 GT 읽기·선택 정책 변경은 없었다. 실질 성능 판정은 사전 고정 source VAL과 그 통과 후에만 허용되는 실사 평가로 분리한다.', '',
        '[검산 JSON](TRAIN_CONVERGENCE.json), [재사용 source target 진단](../pallet_pose_pareto_anchor_20261001_v1/SOURCE_FEASIBILITY.json), [학습 계약](TRAIN_PROTOCOL.json)', '']
    C.save(C.DOC / 'TRAIN_CONVERGENCE_KO.md', '\n'.join(lines))
    print('INDEPENDENT_ANCHOR_CONTEXT_TRAIN_VERIFICATION_PASS_ALL4', flush=True)


def selfcheck():
    names = ['R0:long-face-front', 'R0:short-face-front', 'DIVERSE251_s1:long-face-front', 'DIVERSE251_s1:short-face-front']
    errors = np.array([[[2., 2.], [1., 4.], [1., 1.], [3., 3.]], [[np.inf, np.inf]] * 4])
    valid = np.isfinite(errors).all(2)
    anchor = errors[:, 0].copy()
    target, safe = independent_targets(errors, valid, anchor, np.array([0, -1]), [1., 1.], names)
    assert target.tolist() == [2, -1] and safe.tolist() == [[True, False, True, False], [False] * 4]
    x = np.array([[[0., 0.], [-1., 0.], [1., 0.], [0., 1.]], [[0., 0.]] * 4])
    value = independent_objective(np.array([1., 0.]), x, valid, target)
    stats, picked = score_statistics(value['scores'], valid, target, safe, errors, anchor, names)
    assert picked.tolist() == [1, -1] and stats['anchor_violations']['R']['count'] == 1
    assert stats['failed_rows'] == 1 and stats['frames'] == 2
    assert stats['target_accuracy_full_population'] == 0. and stats['anchor_violations']['R']['rate'] == .5
    assert stats['anchor_violations']['R']['rate_available_rows'] == 1.
    expected_ce = float(F.cross_entropy(torch.tensor([[0., 1., -1., 0.]], dtype=torch.float64), torch.tensor([2]))) / 2
    assert abs(value['CE'] - expected_ce) < 1e-12
    # Independently construct189 columns, including the all-invalid -1 row.
    from . import convex_train as T
    rng = np.random.default_rng(6719)
    raw = rng.normal(size=(3, 4, RAW_FEATURE_DIM)).astype(np.float32)
    vv = np.array([[True, True, False, True], [True, True, True, True], [False] * 4])
    raw[~vv] = np.nan
    ai = np.array([1, 0, -1], np.int64)
    mean = rng.normal(size=RAW_FEATURE_DIM).astype(np.float32)
    std = rng.uniform(.1, 2., size=RAW_FEATURE_DIM).astype(np.float32)
    xx = independent_context(raw, vv, ai, mean, std)
    np.testing.assert_array_equal(xx, T.context_inputs(raw, vv, ai, mean, std))
    np.testing.assert_array_equal(xx[:, :, -1], [[0., 1., 0., 0.], [1., 0., 0., 0.], [0.] * 4])
    np.testing.assert_array_equal(xx[~vv], 0.)
    ww = rng.normal(size=RAW_FEATURE_DIM)
    padded = np.concatenate([ww, np.zeros(95)])
    yy = np.array([3, 1, -1])
    a = independent_objective(padded, xx, vv, yy)
    b = independent_objective(ww, xx[:, :, :94], vv, yy)
    np.testing.assert_allclose(a['scores'], b['scores'], rtol=1e-12, atol=1e-12)
    assert abs(a['objective'] - b['objective']) < 1e-12
    np.testing.assert_allclose(a['gradient'][:94], b['gradient'], rtol=1e-12, atol=1e-12)
    complete_weight = rng.normal(size=FEATURE_DIM)
    autograd = independent_objective(complete_weight, xx, vv, yy)
    for column in (0, 93, 94, 187, 188):
        delta = np.zeros(FEATURE_DIM); delta[column] = 1e-5
        plus = independent_objective(complete_weight + delta, xx, vv, yy)['objective']
        minus = independent_objective(complete_weight - delta, xx, vv, yy)['objective']
        assert abs((plus - minus) / 2e-5 - autograd['gradient'][column]) < 1e-7
    try:
        independent_context(raw, vv, np.array([-1, 0, -1]), mean, std)
    except AssertionError:
        pass
    else:
        raise AssertionError('MISSING_ANCHOR_NOT_REJECTED')
    assert OLD.READS is None, 'Selfcheck must not read artifacts'
    print('ANCHOR_CONTEXT_TRAIN_VERIFIER_SELFCHECK_PASS_NO_ARTIFACT_READS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (selfcheck if args.stage == 'selfcheck' else main)()
