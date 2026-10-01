"""Describe existing TRAIN choices only; no policy probe, fit, VAL or real data."""
import argparse
from pathlib import Path
import numpy as np

from . import common as C
from . import convex_train as T
from . import verify_train as V


def describe(values):
    values = np.asarray(values, np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    return dict(n=len(values), median=float(np.quantile(values, .5)) if len(values) else None,
                P90=float(np.quantile(values, .9)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None,
                mean=float(values.mean()) if len(values) else None)


def classify(indices, valid, errors, anchor, anchor_index, scale):
    """Classify a supplied, already fixed choice; never choose a new candidate."""
    n = len(indices)
    present = valid.any(1)
    assert np.array_equal(indices >= 0, present)
    rows = np.flatnonzero(present)
    assert valid[rows, indices[rows]].all()
    selected = errors[rows, indices[rows]]
    reference = anchor[rows]
    assert np.isfinite(selected).all() and np.isfinite(reference).all()
    delta = (selected - reference) / np.asarray(scale, np.float64)
    excess = np.maximum(delta.max(1), 0.)
    is_anchor = indices[rows] == anchor_index[rows]
    safe = np.all(selected <= reference, axis=1)
    strict_gain = safe & np.any(selected < reference, axis=1)
    equal = np.all(selected == reference, axis=1)
    masks = dict(anchor=is_anchor, safe_improvement=~is_anchor & strict_gain,
                 safe_equal=~is_anchor & equal, unsafe=~safe)
    assert np.all(sum(mask.astype(int) for mask in masks.values()) == 1)
    assert np.all(safe[is_anchor]) and np.all(excess[is_anchor] == 0.)
    classes = {k: dict(count=int(v.sum()), fraction_full_population=float(v.sum()/n),
                       fraction_available_rows=float(v.mean())) for k, v in masks.items()}
    classes['failed'] = dict(count=int((~present).sum()), fraction_full_population=float((~present).mean()),
                             fraction_available_rows=None)
    violates = dict(T=delta[:, 0] > 0., R=delta[:, 1] > 0.,
                    T_only=(delta[:, 0] > 0.) & (delta[:, 1] <= 0.),
                    R_only=(delta[:, 1] > 0.) & (delta[:, 0] <= 0.),
                    both=np.all(delta > 0., axis=1), either=np.any(delta > 0., axis=1))
    severity = {}
    for name, mask in violates.items():
        severity[name] = dict(count=int(mask.sum()), fraction_full_population=float(mask.sum()/n),
                             fraction_available_rows=float(mask.mean()),
                             normalized_max_excess=describe(excess[mask]),
                             normalized_T_positive_excess=describe(np.maximum(delta[mask, 0], 0.)),
                             normalized_R_positive_excess=describe(np.maximum(delta[mask, 1], 0.)))
    return dict(frames=n, available_rows=len(rows), failed_rows=n-len(rows), classes=classes,
                violations=severity, normalized_max_excess_all_available=describe(excess),
                normalized_max_excess_definition='max((T-T_anchor)/sT, (R-R_anchor)/sR, 0)',
                undefined_failed_severity=n-len(rows),
                denominator='Fractions report both all2598 rows and available2597. Failure severity is undefined, never zero or a safe outcome.')


def run():
    T.OLD.install_training_guard()
    protocol = C.protocol('TRAIN_PROTOCOL')
    binding = C.bind(C.DOC / 'TRAIN_PROTOCOL.json')
    audit_path = C.DOC / 'TRAIN_CONVERGENCE.json'
    audit = C.read(audit_path)
    assert audit['complete'] and audit['PASS'] and audit['protocol'] == binding
    assert audit['source_TRAIN_only'] and not audit['VAL_quality_read'] and not audit['real_targets_read']
    assert audit['feature_map'] == T.FEATURE_MAP and set(audit['models']) == set(C.MODEL_NAMES)
    locked = {b['path']: b for b in audit['bindings']}
    for b in locked.values():
        C.verify(b)
    parent = T.OLD.load_training_inputs()
    anchor, anchor_index = V.load_anchor(parent)
    assert len(parent['ids']) == 2598 and np.sum(anchor_index >= 0) == 2597
    bindings = [binding, C.bind(audit_path), C.bind(Path(__file__)), C.bind(Path(V.__file__))]
    models = {}
    for model in C.MODEL_NAMES:
        checkpoint_path = C.RAW / 'fits' / model / 'final.json'
        checkpoint_binding = C.bind(checkpoint_path)
        assert locked[checkpoint_binding['path']] == checkpoint_binding
        ck = C.read(checkpoint_path)
        assert ck['protocol'] == binding and ck['model'] == model and ck['certificate']['PASS']
        seed = 1 if model == 'R0_ONLY' else int(model[-1])
        names = T.candidate_names(model)
        parents = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{seed}'])
        raw, valid, errors = [np.concatenate([parent[key][p] for p in parents], axis=1)
                              for key in ('features', 'valid', 'errors')]
        targets, safe = V.independent_targets(errors, valid, anchor, anchor_index, parent['scale'], names)
        score = T.score_candidates(ck, raw, valid, anchor_index)
        picked = V.select(score, valid, names)
        assert T.OLD.array_sha(picked) == audit['models'][model]['statistics']['selected_index_sha']
        assert T.OLD.array_sha(safe) == audit['models'][model]['target_safe_mask_hash']
        receipt = C.read(C.DOC / f'FIT_{model}.json')
        assert T.OLD.array_sha(targets) == receipt['target_sha']
        target_info = classify(targets, valid, errors, anchor, anchor_index, parent['scale'])
        selected_info = classify(picked, valid, errors, anchor, anchor_index, parent['scale'])
        assert target_info['classes']['unsafe']['count'] == 0
        prior = audit['models'][model]['statistics']
        assert selected_info['classes']['unsafe']['count'] == prior['anchor_violations']['either']['count']
        for key in ('T', 'R', 'both', 'either'):
            assert selected_info['violations'][key]['count'] == prior['anchor_violations'][key]['count']
        target_correct = int(np.sum(picked[valid.any(1)] == targets[valid.any(1)]))
        assert target_correct == prior['target_correct']
        models[model] = dict(checkpoint=checkpoint_binding, target=target_info, selected=selected_info,
            exact_target_correct=target_correct, target_index_sha=T.OLD.array_sha(targets),
            unchanged_selected_index_sha=T.OLD.array_sha(picked), original_choice_replay_PASS=True)
        bindings += [checkpoint_binding, C.bind(C.DOC / f'FIT_{model}.json')]
    output = dict(complete=True, PASS=True, created_at=C.now(), protocol=binding,
        scope='Existing frozen TRAIN choices only. No changed scores, feature map, policy, threshold, routing, or loss trial.',
        frames=2598, available_anchor_rows=2597, failed_rows=1, scale_sT_cm_sR_deg=parent['scale'], models=models,
        source_TRAIN_only=True, VAL_quality_read=False, real_targets_read=False,
        new_fits=0, optimizer_steps=0, new_image_forwards=0, alternative_policy_probes=0,
        target_definition=protocol['target_rule'], bindings=bindings, read_paths=sorted(set(T.OLD.READS)),
        method_success=False, goal_complete=False,
        interpretation='Uniform CE uses target identity and score probabilities, not physical excess magnitude once that target is fixed. Correlation through features remains possible; this does not prove an expressivity limit or that a risk loss will pass source/real gates.')
    C.save(C.DOC / 'TRAIN_RISK_DIAGNOSTIC.json', output)
    write_report(output)
    print('TRAIN_RISK_DIAGNOSTIC_PASS_FROZEN_CHOICES_ONLY', flush=True)


def write_report(output):
    lines = ['# 고정 TRAIN 선택의 anchor 위험 진단', '',
        '현재189 모델의 기존 TRAIN 선택만 다시 재현했다. 모든 선택 index SHA는 독립 TRAIN 검산의 기존 값과 일치한다. 새로운 후보 선택 규칙, 임계값, loss 또는 fit을 실행하지 않았다. VAL 배열·실사 GT는 읽지 않았다.', '',
        '전체2,598행 중 유효 anchor는2,597행이다. 실패1행은 별도 유지하며 위험을0이나 안전한 선택으로 간주하지 않는다. JSON에는 전체·유효행 두 분모의 비율을 모두 기록했다. 아래 표는 행 수다. `safe improvement`는 두 축 모두 anchor 이하이면서 적어도 한 축이 엄격히 작은 non-anchor pose다. 두 오차가 정확히 같은 non-anchor는 별도 `safe equal`이다.', '',
        '| 모델 | target anchor | target non-anchor safe | 선택 anchor | 선택 safe improvement | 선택 safe equal | 선택 unsafe | 실패 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in output['models'].items():
        a, b = value['target']['classes'], value['selected']['classes']
        lines.append(f"| {name} | {a['anchor']['count']} | {a['safe_improvement']['count']+a['safe_equal']['count']} | {b['anchor']['count']} | {b['safe_improvement']['count']} | {b['safe_equal']['count']} | {b['unsafe']['count']} | {b['failed']['count']} |")
    lines += ['', '위험은 같은 frame의 R0 operational GEO에 대한 정규화 초과량 `r=max((T-Ta)/sT,(R-Ra)/sR,0)`이다. 기존 TRAIN 고정 scale을 그대로 사용한다. 다음 분위수의 분모는 실제 해당 위반을 선택한 행만이며 실패1행은 정의되지 않는다. T·R 위반 집합은 두 축 모두 위반한 경우 겹친다.', '',
        '| 모델 | 위반 | 행 수 | r 중앙값 | r P90 | r 최대 |', '|---|---|---:|---:|---:|---:|']
    for name, value in output['models'].items():
        for axis in ('T', 'R', 'either'):
            v = value['selected']['violations'][axis]
            d = v['normalized_max_excess']
            def fmt(x):
                return 'NA' if x is None else f'{x:.6f}'
            lines.append(f"| {name} | {axis} | {v['count']} | {fmt(d['median'])} | {fmt(d['P90'])} | {fmt(d['maximum'])} |")
    lines += ['', '현재 CE는 target y가 고정되면 `logsumexp(-score)+score_y`다. 같은 target·valid·score라면 오선택의 물리 초과량이 달라도 loss와 gradient가 같다. score 확률이나 입력 특징을 통한 간접 구분은 가능하므로, 이 사실만으로 현재 표현력이 부족하거나 CE가 위험을 전혀 학습할 수 없다고 결론내리지 않는다.', '',
        '## 다음 개입 후보 하나: TRAIN 위험으로 경쟁 logit을 보정하는 CE', '',
        '실행하지 않은 후보로, 고정 target·189특징·후보·scale은 유지하고 TRAIN에서 각 후보의 anchor 위험 r_c를 구해 `m_c=log1p(r_c)`를 정한다. 손실만 `logsumexp(-score_c+m_c)+score_y+(lambda/2)||w||²`로 바꾼다. anchored target은 r_y=0이므로 target 항에는 추가 보정이 없다. 이 식은 unsafe 경쟁 후보의 softmax 질량에1+r_c를 곱하며, 안전한 경쟁 후보에는 기존 CE와 같은0 margin을 적용한다.', '',
        '이는 TRAIN label에서 정한 상수 margin을 쓰는 log-sum-exp이므로 고정 선형189 weight에 대해 convex이고, 기존 모든 weight의 L2를 유지하면 lambda-strongly-convex 수렴 인증을 그대로 적용할 수 있다. 원시 r를 직접 margin으로 쓰면 지수적으로 큰 경쟁 가중이 생기므로, 제안은 추가 scale이나 sweep 없이 `log1p(r)` 한 형태만 지정한다. 현재 수치로 가중치·절단값을 최적화한 것은 아니다. 런타임은 기존 input-only score와 전체 valid 후보의 argmin을 유지하며 GT 위험·margin을 입력하지 않는다.', '',
        '이 개입은 단순 row별 class weighting이나 local4-edge pairwise가 아니라, 현재 whole-pose target 하나와 모든 경쟁 후보의 physical anchor 초과량을 같은 listwise loss에 넣는다. 따라서 모든 오선택을 같은 비용으로 다루는 현재 감독과 차이가 명확하다. 그러나 위험 가중의 감소가 실제 두 중앙값·P90을 함께 개선한다는 보장은 없고, anchor 쪽으로 과도하게 보수적으로 이동해 strict gain을 잃을 수도 있다.', '',
        '## 선행과 중단 조건', '',
        '비용·보존 감독 자체는 새 개념이 아니다. [기존 목적함수 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)에 DHT의2D soft-cost CE/회귀, whole-layout gain 회귀, RGB utility 회귀, 좌표 헤드의 baseline-relative squared regret 실패가 정리돼 있다. [고정4-edge pairwise 실험](../pallet_pose_selector_pairwise_20261001_v1/REPORT_KO.md)도 수렴 후 전체 pose 선택과 위험이 악화했다. 이들은 현재 physical T/R anchor 위험을 TRAIN-only logit margin으로 쓰는 global189 CE와 같지 않지만, 위험/비용 정보를 추가하면 전이가 해결된다는 주장에 대한 반례다. 저장소 제한 검색에서 이 정확한 조합의 실행은 찾지 못했으며 학술적 신규성 주장은 하지 않는다.', '',
        '다음 실제 실험을 택하더라도 단일 고정 loss, 같은4모델·zero init·L2·solver 예산·모든2,598행을 유지한다. 실패한 수렴을 연장하거나 좋은 seed만 선택하지 않는다. 기존 source45개 조건을 전부 통과하기 전 실사 routing은 금지하며, 통과 후에도 원래 실사5개 안정성 조건과 모든 seed 조건을 그대로 적용한다. 이 문서는 제안과 TRAIN 진단이며 새 실험 protocol이나 T/R 개선 결과가 아니다.', '',
        '[전체 수치와 해시](TRAIN_RISK_DIAGNOSTIC.json) · [기존 독립 TRAIN 검산](TRAIN_CONVERGENCE_KO.md)', '']
    C.save(C.DOC / 'TRAIN_RISK_DIAGNOSTIC_KO.md', '\n'.join(lines))


def selfcheck():
    err = np.array([[[1., 1.], [2., 3.]], [[1., 1.], [.5, 1.]], [[1., 1.], [1., 1.]], [[np.inf, np.inf]]*2])
    valid = np.isfinite(err).all(2)
    anchor = err[:, 0]
    out = classify(np.array([1, 1, 1, -1]), valid, err, anchor, np.array([0, 0, 0, -1]), [1., 2.])
    assert {k: v['count'] for k, v in out['classes'].items()} == dict(anchor=0, safe_improvement=1, safe_equal=1, unsafe=1, failed=1)
    assert out['violations']['both']['count'] == 1 and out['violations']['either']['normalized_max_excess']['maximum'] == 1.
    assert out['classes']['unsafe']['fraction_full_population'] == .25
    assert out['classes']['unsafe']['fraction_available_rows'] == 1/3
    assert T.OLD.READS is None
    print('TRAIN_RISK_SELFCHECK_PASS_INVENTED_ONLY', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'run'])
    args = parser.parse_args()
    (selfcheck if args.stage == 'selfcheck' else run)()
