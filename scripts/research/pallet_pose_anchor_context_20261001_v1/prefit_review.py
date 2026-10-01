"""Independent pre-fit input-map review; no labels, optimization, or VAL quality.

The invented fixtures exercise the actual public map/scorer/objective API.
The real TRAIN check selects input-only feature rows from existing eligibility
metadata, then connects unchanged target hashes to the previous sealed audit.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
OLD_DOC = ROOT / '_docs/experiments/pallet_pose_pareto_anchor_20261001_v1'
MODELS = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
READS = set()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def verify(binding):
    assert bind(ROOT / binding['path']) == binding, binding['path']


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def guard():
    permitted = {DOC / 'PREFIT_REVIEW.json', DOC / 'PREFIT_REVIEW_KO.md'}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(ROOT):
            return
        mode = args[1]
        if isinstance(mode, str) and any(c in mode for c in 'wax+'):
            assert p in permitted, ('WRITE_SCOPE', str(p))
            return
        s = str(p)
        assert not any(token in s for token in (
            '/data/evaluation/', '/real_gt_v2/', '/annotations/', 'SOURCE_TRAIN_LABELS.npz',
            'SOURCE_VAL_GATE.json', 'SOURCE_VAL_METRICS.npz', 'REAL_FEATURE', 'REAL_CHOICES',
            'GEOMETRY_SIDETABLE.npz', '/SYNTH_RECORDS.json', '/SYNTH_LABELS.npz',
            'POSE_METRICS.json', 'CANDIDATE_BOUNDS', 'GEOMETRY_RESOLVED_POSE_GT')), ('FORBIDDEN_DATA', s)
        assert p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.onnx'), s
        READS.add(str(p.relative_to(ROOT)))
    sys.addaudithook(hook)


def row_map(raw, valid, anchor_index, mean, std):
    """Scalar candidate loop, deliberately independent of production broadcast."""
    n, k, d = raw.shape
    assert d == 94
    out = np.zeros((n, k, 189), np.float64)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    for i in range(n):
        usable = np.flatnonzero(valid[i]).tolist()
        anchor = int(anchor_index[i])
        if not usable:
            assert anchor == -1
            continue
        assert anchor in (0, 1) and anchor in usable
        reference = ((raw[i, anchor] - mean) / std).astype(np.float64)
        for j in usable:
            z = ((raw[i, j] - mean) / std).astype(np.float64)
            out[i, j, :94] = z
            out[i, j, 94:188] = np.array([abs(float(z[q]) - float(reference[q])) for q in range(94)])
            out[i, j, 188] = float(j == anchor)
    return out


def row_objective(w, x, valid, target, ridge=1e-4):
    value = 0.
    gradient = np.zeros(len(w), np.float64)
    for i in range(len(x)):
        keep = np.flatnonzero(valid[i]).tolist()
        if not keep:
            assert target[i] == -1
            continue
        logits = np.array([-sum(float(a) * float(b) for a, b in zip(w, x[i, j])) for j in keep])
        maximum = max(logits)
        exps = np.exp(logits - maximum)
        probability = exps / sum(exps)
        value += maximum + math.log(sum(exps)) - logits[keep.index(int(target[i]))]
        gradient += x[i, target[i]]
        for j, p in zip(keep, probability):
            gradient -= p * x[i, j]
    return value / len(x) + ridge * sum(w * w) / 2., gradient / len(x) + ridge * w


def expect_rejection(callback):
    try:
        callback()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Invalid anchor support silently accepted')


def synthetic_review(T):
    rng = np.random.default_rng(2026100107)
    raw = rng.normal(size=(5, 4, 94)).astype(np.float32)
    valid = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1]], bool)
    anchor = np.array([0, 1, 0, -1, 1], np.int64)
    raw[~valid] = np.nan  # Missing values may not leak into any context column.
    mean = rng.normal(size=94).astype(np.float32)
    std = (rng.random(94) + .3).astype(np.float32)
    actual = T.context_inputs(raw, valid, anchor, mean, std)
    independent = row_map(raw, valid, anchor, mean, std)
    np.testing.assert_array_equal(actual, independent)
    assert np.count_nonzero(actual[~valid]) == 0
    np.testing.assert_array_equal(actual[:, :, :94], T.ANCHOR.normalized_inputs(raw, valid, mean, std))
    for i in np.flatnonzero(valid.any(1)):
        assert not actual[i, anchor[i], 94:188].any()
        assert actual[i, :, 188].sum() == 1.
    for bad in (np.array([-1, 1, 0, -1, 1]), np.array([2, 1, 0, -1, 1]),
                np.array([0, 0, 0, -1, 1]), np.array([0, 1, 0, 0, 1])):
        expect_rejection(lambda: T.context_inputs(raw, valid, bad, mean, std))
    # Candidate identity is operational, not feature equality.
    duplicate = np.zeros((1, 2, 94), np.float32)
    duplicate_map = T.context_inputs(duplicate, np.ones((1, 2), bool), np.array([1]), np.zeros(94), np.ones(94))
    np.testing.assert_array_equal(duplicate_map[0, :, 188], [0., 1.])
    w = rng.normal(size=189) * .03
    ck = dict(schema=T.CHECKPOINT_SCHEMA, feature_map=T.FEATURE_MAP, feature_dim=189,
        raw_feature_dim=94, normalization='old_float32_then_float64', names=T.candidate_names('UNION_s1'),
        bias=0., lambda_l2=1e-4, mean=mean.tolist(), std=std.tolist(), weight=w.tolist())
    score = T.score_candidates(ck, raw, valid, anchor)
    reference_score = np.where(valid, np.sum(independent * w, axis=2), np.inf)
    np.testing.assert_allclose(score, reference_score, rtol=1e-14, atol=1e-13)
    w94 = rng.normal(size=94)
    embedded = {**ck, 'weight': np.r_[w94, np.zeros(95)].tolist()}
    legacy = dict(schema='pallet_pose_convex_linear94_v1', bias=0., lambda_l2=1e-4,
                  weight=w94.tolist(), mean=mean.tolist(), std=std.tolist())
    new_score = T.score_candidates(embedded, raw, valid, anchor)
    old_score = T.ANCHOR.score_candidates(legacy, raw, valid)
    np.testing.assert_allclose(new_score, old_score, rtol=1e-14, atol=1e-13)
    target = np.array([2, 2, 0, -1, 0], np.int64)
    embedded_objective = T.objective(np.r_[w94, np.zeros(95)], actual, valid, target)
    legacy_objective = T.ANCHOR.objective(w94, actual[:, :, :94], valid, target)
    assert abs(embedded_objective['value'] - legacy_objective['value']) < 1e-12
    np.testing.assert_allclose(embedded_objective['gradient'][:94], legacy_objective['gradient'], rtol=1e-12, atol=1e-13)
    out = T.objective(w, actual, valid, target)
    manual, gradient = row_objective(w, independent, valid, target)
    assert abs(out['value'] - manual) <= 1e-13
    np.testing.assert_allclose(out['gradient'], gradient, rtol=1e-12, atol=1e-13)
    differences = []
    for j in range(189):
        step = np.zeros(189); step[j] = 1e-5
        numerical = (row_objective(w + step, independent, valid, target)[0] -
                     row_objective(w - step, independent, valid, target)[0]) / 2e-5
        differences.append(abs(numerical - out['gradient'][j]))
    assert max(differences) < 1e-8
    assert out['probability'][0, 0] > 0 and not out['probability'][~valid].any()
    # Fixed abs context can change a ranking; signed z-zanchor alone cancels
    # the common anchor under a shared linear score.
    toy = np.zeros((2, 2, 94), np.float32); toy[:, 0, 0] = -2.; toy[:, 1, 0] = 1.
    toy_map = T.context_inputs(toy, np.ones((2, 2), bool), np.array([0, 1]), np.zeros(94), np.ones(94))
    np.testing.assert_array_equal(toy_map[:, :, 94], [[0., 3.], [3., 0.]])
    assert ast.dump(ast.parse(inspect.getsource(T.anchored_targets))) == ast.dump(ast.parse(inspect.getsource(T.ANCHOR.anchored_targets)))
    assert ast.dump(ast.parse(inspect.getsource(T.objective))) == ast.dump(ast.parse(inspect.getsource(T.ANCHOR.objective)))
    assert T.MAX_CALLS == 2000 and T.MAX_ITER == 1000 and T.GAP_MAX == 1e-6 and T.LAMBDA == 1e-4
    return dict(PASS=True, map_row_loop_bit_exact=True, invalid_columns_zero=True,
        rejected_bad_anchor_contracts=4, anchor_identity_not_feature_equality=True,
        raw94_embedding_score_max_difference=float(np.abs(new_score[valid]-old_score[valid]).max()),
        raw94_embedding_objective_difference=abs(embedded_objective['value']-legacy_objective['value']),
        raw94_embedding_gradient_max_difference=float(np.abs(embedded_objective['gradient'][:94]-legacy_objective['gradient']).max()),
        scorer_max_difference=float(np.abs(score[valid]-reference_score[valid]).max()),
        independent_gradient_max_difference=float(np.abs(gradient-out['gradient']).max()),
        all189_finite_difference_gradient_max_difference=max(differences),
        full_row_denominator_keeps_all_invalid_row=True, original_valid_candidates_remain_CE_competitors=True,
        target_function_AST_identical=True, CE_ridge_objective_AST_identical=True,
        absolute_context_changes_toy_ranking=True, new_optimizations=0)


def actual_inputs(T):
    old_protocol_path = OLD_DOC / 'TRAIN_PROTOCOL.json'
    verify(read(OLD_DOC / 'TRAIN_PROTOCOL_SHA.json'))
    protocol = read(old_protocol_path)
    for key in ('features', 'feature_lock', 'source_contract', 'source_feasibility'):
        verify(protocol['inputs'][key])
    contract = read(ROOT / protocol['inputs']['source_contract']['path'])
    assert contract['complete'] and contract['status'] == 'PASS'
    feasibility = read(ROOT / protocol['inputs']['source_feasibility']['path'])
    assert feasibility['complete'] and feasibility['PASS'] and feasibility['source_TRAIN_only']
    assert not feasibility['VAL_quality_read'] and not feasibility['real_targets_read']
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    assert len(eligible) == 2598 and eligible.isdisjoint(contract['fit_eligibility']['eligible_ids']['VAL'])
    feature_lock = read(ROOT / protocol['inputs']['feature_lock']['path'])
    assert not feature_lock['source_targets_read'] and not feature_lock['real_targets_read']
    assert feature_lock['features'] == protocol['inputs']['features']
    with np.load(ROOT / protocol['inputs']['features']['path'], allow_pickle=False) as data:
        all_ids, split = data['ids'], data['split']
        idx = np.array([i for i, fid in enumerate(all_ids) if fid in eligible], np.int64)
        ids = all_ids[idx]
        assert len(idx) == 2598 and (split[idx] == 'TRAIN').all()
        assert array_sha(ids) == feasibility['source_ids_sha']
        assert array_sha(idx) == feasibility['source_index_sha']
        anchor = data['R0_GEO_index'][idx]
        features = {m: data[m+'_geo'][idx] for m in ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')}
        masks = {m: data[m+'_valid'][idx] for m in features}
    assert anchor.dtype == np.int64 and Counter(anchor.tolist())[-1] == 1
    absent_ids = ids[anchor < 0].tolist()
    assert absent_ids == ['TEX__shard_04_f0110']
    vals = features['R0'][masks['R0']]
    mean, std = vals.mean(0), np.maximum(vals.std(0), np.float32(1e-6))
    normsha = array_sha(np.stack([mean, std]))
    complete = read(OLD_DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['protocol'] == bind(old_protocol_path)
    receipts = {}
    for binding in complete['fits']:
        verify(binding)
        receipt = read(ROOT / binding['path'])
        assert receipt['normalization_sha'] == normsha
        assert receipt['source_feasibility'] == protocol['inputs']['source_feasibility']
        receipts[receipt['model']] = receipt
    checked = {}
    for model in MODELS:
        parents = ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw = np.concatenate([features[m] for m in parents], axis=1)
        valid = np.concatenate([masks[m] for m in parents], axis=1)
        raw_before, valid_before = array_sha(raw), array_sha(valid)
        assert np.array_equal(valid.any(1), anchor >= 0)
        expected = feasibility['models'][model]
        assert valid_before == expected['original_valid_sha']
        assert array_sha(anchor) == expected['anchor_index_sha']
        assert receipts[model]['target_sha'] == expected['target_sha']
        assert receipts[model]['safe_mask_sha'] == expected['safe_mask_sha']
        current = T.context_inputs(raw, valid, anchor, mean, std)
        independent = row_map(raw, valid, anchor, mean, std)
        np.testing.assert_array_equal(current, independent)
        np.testing.assert_array_equal(current[:, :, :94], T.ANCHOR.normalized_inputs(raw, valid, mean, std))
        assert not current[~valid].any()
        assert array_sha(raw) == raw_before and array_sha(valid) == valid_before
        checked[model] = dict(shape=list(current.shape), context_sha=array_sha(current), raw_features_sha=raw_before,
            original_valid_sha=valid_before, anchor_index_sha=array_sha(anchor),
            previous_target_sha=expected['target_sha'], previous_safe_mask_sha=expected['safe_mask_sha'],
            prior_fit_target_safe_hashes_match=True, independent_row_map_bit_exact=True,
            normalized94_preserved_exact=True, valid_candidates=int(valid.sum()),
            all_invalid_rows=int((~valid.any(1)).sum()), source_rows_retained=2598)
    return dict(PASS=True, source_TRAIN_only=True, rows=2598, ids_sha=array_sha(ids), source_index_sha=array_sha(idx),
        normalization_sha=normsha, anchor_index_sha=array_sha(anchor), anchor_index_counts=dict(Counter(anchor.tolist())),
        absent_anchor_ids=absent_ids, models=checked, source_contract=protocol['inputs']['source_contract'],
        features=protocol['inputs']['features'], source_feasibility=protocol['inputs']['source_feasibility'],
        previous_train_protocol=bind(old_protocol_path), previous_training_complete=bind(OLD_DOC/'TRAINING_COMPLETE.json'),
        source_label_values_read=False, new_target_values_computed=False, VAL_feature_rows_selected=0)


def prior_review():
    specs = [
        ('Stage4 expert router', 'scripts/research/pallet_selector_recovery_v1/router_features.py',
         '이미 expert box/keypoint/confidence/pose 차이와 선택 margin, 선택적 GAP448 문맥을 사용했다. 두 expert의 W/D를 먼저 선택한 뒤 ADDnorm 기반 MLP routing이며, 4 whole-pose 후보의 고정 189 선형 map이 아니다.'),
        ('Model-conditioned selector', 'scripts/research/pallet_model_conditioned_selector_v1/train_scorers.py',
         'model별 기존 94차원 GEO_LINEAR를 따로 학습한다. 공유 per-candidate 선형 점수, source parity BCE, VAL early-stop이며 고정 R0 GEO 기준 abs94+identity1은 없다.'),
        ('N2/Replay utility', 'scripts/research/pallet_posefix_utility_selector_v1/model.py',
         '후보 좌표 차이와 corner/partner/global RGB·수치 문맥을 사용한다. corner utility로 vertical pair를 교체하는 비선형 CNN이며 whole-pose T/R anchored CE와 다르다.'),
        ('DHT local no-harm', 'scripts/research/pallet_symmetry_dht_local_v2/symdht_local_v2/objective.py',
         'baseline 대비 2D 오차의 ReLU no-harm 및 line utility를 학습하는 연속 좌표 교정 목적식이다. 이 목적식의 baseline 비교와 이번 입력-only abs-normalized pose-feature context는 같은 연산이 아니다.'),
    ]
    prior = [dict(name=name, code=bind(ROOT/path), distinction_ko=description) for name,path,description in specs]
    return dict(scope='Bounded repository implementation comparison, not a literature or global novelty claim.',
        relative_context_precedent_exists=True, exact189_map_found_in_these_prior_codes=False,
        new_intervention='Fixed [z94, abs(z-z_R0_GEO)94, is_operational_anchor1]; shared linear189 with previous anchored targets, original valid CE, ridge and solver.',
        no_guaranteed_gain=True, previous_negative_audit=bind(OLD_DOC/'PRIOR_METHOD_AUDIT.json'), methods=prior)


def run(write):
    sys.dont_write_bytecode = True
    guard()
    from . import convex_train as T
    assert T.OLD.READS is None
    code_before = bind(T.__file__)
    synthetic = synthetic_review(T)
    if not write:
        print(json.dumps(synthetic, indent=2)); return
    actual = actual_inputs(T)
    priors = prior_review()
    assert bind(T.__file__) == code_before, 'Trainer changed during review; retry after final freeze.'
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    result = dict(complete=True, PASS=True, created_at=datetime.now(timezone.utc).isoformat(),
        scope='Independent pre-fit synthetic numerics and existing TRAIN input-only feature map; no performance evaluation.',
        feature_map=T.FEATURE_MAP, feature_dim=189, raw_feature_dim=94,
        trainer=code_before, reviewer=bind(__file__), synthetic=synthetic, actual_TRAIN_inputs=actual,
        prior_method_audit=priors, source_TRAIN_only=True, source_label_values_read=False,
        VAL_quality_read=False, real_targets_read=False, fits=0, image_forwards=0, new_PnP_calls=0,
        new_target_calculations=0, input_only_feature_transform=True, read_paths=sorted(READS),
        method_success=False, goal_complete=False)
    md = f'''# 189차원 anchor 문맥의 학습 전 독립 검토

**구현·입력 계약 검산 PASS. 성능 개선을 뜻하지 않는다.** 실제 학습이나 source VAL/실사 성능 평가는 실행하지 않았다.

고정 map은 `[z94, abs(z−z_R0_GEO)94, anchor_identity1]`이다. z는 기존 float32 정규화를 끝낸 뒤 float64로 변환한다. 추가 정규화나 학습되는 feature extractor는 없다. anchor는 입력만으로 저장된 R0 GEO의 0/1 index이며 참조 T/R에서 선택하지 않는다. identity는 숫자 특징이 같은 다른 후보가 아니라 그 index 하나에만 1이다.

독립 NumPy 행 반복 구현과 합성/실제 TRAIN map이 bit 단위로 일치했다. invalid 후보의 189열은 모두 0이며 score는 +∞다. anchor가 없는 all-invalid 한 행은 그대로 남고, 유효 후보가 있는데 anchor가 없거나 잘못된 index인 4종 계약 위반은 STOP한다. 기존 94 scorer를 추가 95가중치 0으로 넣을 수 있음을 확인했다. 점수 차이는 수치 합산 순서에 따른 최대 {synthetic['raw94_embedding_score_max_difference']:.3g}였으며, 절대 차이 map이 실제로 순위를 바꿀 수 있는 합성 예도 확인했다.

CE+ridge와 anchored target 함수 AST는 이전 구현과 같다. 독립 행별 CE/기울기 및 189개 좌표 중앙차분으로 검산했으며 최대 차분 오차는 {synthetic['all189_finite_difference_gradient_max_difference']:.3g}다. 원래 valid 후보가 모두 CE 경쟁자로 남고 all-invalid 행도 전체 분모에 포함된다. 고정 map에 대한 선형 점수이므로 가중치에 대한 볼록성은 유지된다. 이는 새 특징이 일반화한다는 증거는 아니다.

기존 source eligibility로 TRAIN **2,598개**의 순서·ID·source-index SHA를 복원했다. anchor 유효 **2,597개**, anchor=-1은 기존 `TEX__shard_04_f0110` 한 행으로, 모든 모델에 그대로 포함된다. 4모델 원본 valid SHA, anchor index SHA, 기존 float32 정규화 SHA 및 94열 보존을 확인했다. 이전 SOURCE_FEASIBILITY와 4개 fit 영수증의 target/safe-mask SHA를 연결했다. **TRAIN label NPZ와 새 target 값은 읽거나 계산하지 않았다.** VAL은 컨테이너에 있지만 선택·변환한 행은 TRAIN만이다.

선행과의 차이:

'''
    for p in priors['methods']:
        md += '- **' + p['name'] + '**: ' + p['distinction_ko'] + '\n'
    md += '''
상대 후보 문맥과 baseline 대비 학습의 선행은 이미 있다. 이번 좁은 변경은 고정 operational R0 anchor의 abs94+identity1을 이전 anchored-target 볼록 선형 문제에 넣는 것이다. 새 정보/후보/이미지 forward/GT를 더하지 않으며, 과거 실패가 해결됐다는 주장은 별도 고정 source VAL gate 이후에만 평가할 수 있다. 입력은 이미지 한 장·치수·기존 K 계약을 유지한다.

[기계 검산과 SHA](PREFIT_REVIEW.json). 선행 코드의 정확한 경로·SHA와 범위 한계도 JSON에 기록했다.
'''
    DOC.mkdir(parents=True, exist_ok=True)
    with (DOC/'PREFIT_REVIEW_KO.md').open('x') as f:
        f.write(md)
    result['note'] = bind(DOC/'PREFIT_REVIEW_KO.md')
    with (DOC/'PREFIT_REVIEW.json').open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('PREFIT_REVIEW_PASS', bind(DOC/'PREFIT_REVIEW.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'review'])
    run(parser.parse_args().stage == 'review')
