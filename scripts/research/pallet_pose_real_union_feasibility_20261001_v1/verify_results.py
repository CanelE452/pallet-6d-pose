"""Independent cached-result audit; no experiment helper or raw GT import."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_real_union_feasibility_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
KEYS = ('translation_cm', 'rotation_deg')
DIAGNOSTICS = ('AXIS_LOWER_BOUND', 'FIXED_TRAIN_COST_ORACLE')
FIXED = ('R0', 'PRIOR1', 'FULL125')
SEEDS = (1, 2, 3)
READS = set()


def audit_open(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    path = str(p)
    assert not any(token in path for token in (
        '/annotations/', 'GEOMETRY_RESOLVED_POSE_GT', 'TRUTH_FOR_DISPLAY',
        'AXIS_REVIEW_MANIFEST', 'GEOMETRY_SIDETABLE', '/real_gt_v2/')), path
    assert p.suffix.lower() not in ('.pt', '.pth', '.onnx', '.png', '.jpg', '.jpeg', '.bmp'), path
    mode, flags = args[1], args[2]
    writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
               or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    if writing:
        assert p.parent == DOC and p.name in ('VERIFICATION.json', 'VERIFICATION_KO.md'), path
    elif p.is_relative_to(ROOT):
        READS.add(str(p.relative_to(ROOT)))


sys.dont_write_bytecode = True
sys.addaudithook(audit_open)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def verify(binding):
    assert bind(ROOT / binding['path']) == binding, binding['path']


def q(values, probability):
    ordered = sorted(float(v) for v in values)
    assert ordered and all(not math.isnan(v) and v >= 0 or math.isfinite(v) for v in ordered)
    rank = (len(ordered) - 1) * probability
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    if ordered[high] == math.inf:
        return math.inf
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def scalar(value):
    return dict(value=float(value) if math.isfinite(value) else None,
                status='FINITE' if math.isfinite(value) else 'UNDEFINED' if math.isnan(value)
                else 'POSITIVE_INFINITY' if value > 0 else 'NEGATIVE_INFINITY')


def per_seed_quantiles(values, probability=.5, conditional=True):
    assert values.ndim == 3 and values.shape[-1] == 2 and not np.isnan(values).any()
    assert (values >= 0).all()
    assert np.array_equal(np.isfinite(values[..., 0]), np.isfinite(values[..., 1]))
    output = []
    for seed in values:
        rows = seed[np.isfinite(seed).all(1)] if conditional else seed
        output.append([q(rows[:, k], probability) if len(rows) else float('nan') for k in range(2)])
    return np.asarray(output)


def average_quantiles(values, probability=.5, conditional=True):
    return per_seed_quantiles(values, probability, conditional).mean(0)


def mean_summary(values):
    out = {}
    for mode, conditional in (('conditional', True), ('full_population', False)):
        out[mode] = {name: {key: scalar(v) for key, v in zip(KEYS, average_quantiles(values, p, conditional))}
                     for name, p in (('median', .5), ('P90', .9))}
    failures = np.isposinf(values[..., 0]).sum(1)
    out['failure_counts_by_seed'] = failures.tolist()
    out['mean_failure_count'] = float(failures.mean())
    return out


def nested_equal(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        return sum(nested_equal(actual[k], v, path + '.' + k) for k, v in expected.items())
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        return sum(nested_equal(a, b, path + f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected)))
    if isinstance(expected, float):
        assert isinstance(actual, (float, int)) and math.isfinite(actual) and math.isfinite(expected), path
        assert abs(actual - expected) <= 1e-12, (path, actual, expected)
    else:
        assert actual == expected and type(actual) is type(expected), (path, actual, expected)
    return 1


def bootstrap(before, after, recordings):
    rng = np.random.default_rng(20261001)
    names = sorted(set(recordings))
    groups = [np.flatnonzero(np.asarray(recordings) == name) for name in names]
    differences = []
    for _ in range(2000):
        seed_draw = rng.integers(0, before.shape[0], before.shape[0])
        group_draw = rng.integers(0, len(names), len(names))
        frame_draw = np.concatenate([groups[k] for k in group_draw])
        differences.append(average_quantiles(after[seed_draw][:, frame_draw])
                           - average_quantiles(before[seed_draw][:, frame_draw]))
    differences = np.asarray(differences)
    assert np.isfinite(differences).all(), 'Finite-support proof requires every bootstrap draw finite'
    point = average_quantiles(after) - average_quantiles(before)
    out = {}
    for j, key in enumerate(KEYS):
        interval = [q(differences[:, j], p) for p in (.025, .975)]
        out[key] = dict(point_estimate=scalar(point[j]), CI95=interval, finite_draws=2000,
                        undefined_draws=0, status='FINITE_ALL_DRAWS', upper95_below_zero=interval[1] < 0,
                        interval_interpretation='Finite-draw percentile interval; undefined draws explicitly retained in counts and cannot yield gate PASS.')
    return out


def loro(before, after, recordings):
    out = {}
    for name in sorted(set(recordings)):
        keep = np.asarray(recordings) != name
        delta = average_quantiles(after[:, keep]) - average_quantiles(before[:, keep])
        out[name] = dict(frames=int(keep.sum()),
                         mean_seed_median_difference={k: scalar(v) for k, v in zip(KEYS, delta)},
                         both_negative=bool(np.isfinite(delta).all() and (delta < 0).all()))
    return out


def guard(before, after, quantiles):
    checks = {}
    for name, probability in quantiles:
        b, a = average_quantiles(before, probability), average_quantiles(after, probability)
        for j, key in enumerate(KEYS):
            checks[f'{name}:{key}'] = dict(before=scalar(b[j]), after=scalar(a[j]), ratio_limit=1.05,
                                           limit=scalar(b[j] * 1.05),
                                           pass_guard=bool(np.isfinite([a[j], b[j]]).all() and a[j] <= b[j] * 1.05))
    bf, af = np.isposinf(before[..., 0]).sum(1), np.isposinf(after[..., 0]).sum(1)
    checks['pose_failures'] = dict(before_by_seed=bf.tolist(), after_by_seed=af.tolist(),
                                   pass_guard=bool((af <= bf).all()), rule='No seed increases full-population failure count.')
    return dict(PASS=all(v['pass_guard'] for v in checks.values()), checks=checks)


def original_gates(tensors, positions, metadata, reported_hierarchy):
    natural = positions['NATURAL99']
    clean = positions['CLEAN29']
    after = tensors['DIVERSE251'][:, natural]
    recordings = [metadata[i]['recording'] for i in natural]
    three, uncertainty, sensitivity, tails = {}, {}, {}, {}
    check_leaves = 0
    for name in ('SINGLE251', *FIXED):
        before = tensors[name][:, natural]
        delta = per_seed_quantiles(after) - per_seed_quantiles(before)
        three[name] = bool(np.isfinite(delta).all() and (delta < 0).all())
        recorded = reported_hierarchy['NATURAL99'][f'DIVERSE251-minus-{name}']
        check_leaves += nested_equal(mean_summary(before), recorded['before'])
        check_leaves += nested_equal(mean_summary(after), recorded['after'])
        check_leaves += nested_equal([{k: scalar(v) for k, v in zip(KEYS, row)} for row in delta], recorded['per_seed_median_difference'])
        check_leaves += nested_equal({k: scalar(v) for k, v in zip(KEYS, delta.mean(0))}, recorded['mean_seed_median_difference'])
        assert three[name] == recorded['all_seeds_both_medians_smaller']
        leave = loro(before, after, recordings)
        check_leaves += nested_equal(leave, recorded['LORO'])
        if name in ('SINGLE251', 'R0'):
            intervals = bootstrap(before, after, recordings)
            b = recorded['hierarchical_bootstrap']
            assert b['repeats'] == 2000 and b['seed'] == 20261001 and b['recording_count'] == 6 and b['training_seed_count'] == 3
            assert b['paired_recording_and_training_seed'] and b['same_draws_all_comparisons']
            check_leaves += nested_equal(intervals, b['metrics'])
            uncertainty[name] = all(v['upper95_below_zero'] for v in intervals.values())
            sensitivity[name] = all(v['both_negative'] for v in leave.values())
            tails[name] = guard(before, after, [('P90', .9)])
    clean_guard = guard(tensors['R0'][:, clean], tensors['DIVERSE251'][:, clean], [('median', .5), ('P90', .9)])
    gates = dict(all_three_seeds_joint_gain=dict(PASS=all(three.values()), comparisons=three),
                 joint_uncertainty=dict(PASS=all(uncertainty.values()), comparisons=uncertainty),
                 recording_sensitivity=dict(PASS=all(sensitivity.values()), comparisons=sensitivity),
                 natural_tails=dict(PASS=all(v['PASS'] for v in tails.values()), comparisons=tails),
                 clean_preservation=clean_guard)
    return gates, check_leaves


def main():
    protocol = read(DOC / 'PROTOCOL.json')
    verify(read(DOC / 'PROTOCOL_SHA.json'))
    result = read(DOC / 'RESULTS.json')
    assert result['complete'] and result['diagnostic_only'] and not result['method_success'] and not result['goal_complete']
    assert result['protocol'] == bind(DOC / 'PROTOCOL.json')
    assert result['inputs'] == protocol['inputs'] and result['codes'] == protocol['codes']
    frozen_bindings = [*protocol['inputs'].values(), *protocol['codes'], *result['artifacts']]
    for binding in frozen_bindings:
        verify(binding)
    values = {key: read(ROOT / b['path']) for key, b in protocol['inputs'].items() if key != 'candidate_rows'}
    assert values['input_audit']['PASS'] and values['input_audit']['complete']
    assert protocol['inputs']['candidate_rows'] in values['candidate_bounds']['artifacts']
    assert protocol['inputs']['operational_metrics'] in values['stable_results']['artifacts']
    assert values['stable_results']['protocol'] == protocol['inputs']['stable_protocol']
    assert values['stable_protocol_sha'] == protocol['inputs']['stable_protocol']
    metadata = values['eval_metadata']
    ids = [r['id'] for r in metadata]
    assert len(ids) == len(set(ids)) == 173
    byid = {fid: i for i, fid in enumerate(ids)}
    groups = values['eval_groups']
    assert {p: len(groups[p]) for p in ('NATURAL99', 'CLEAN29', 'WOOD45')} == dict(NATURAL99=99, CLEAN29=29, WOOD45=45)
    assert len({metadata[byid[i]]['recording'] for i in groups['NATURAL99']}) == 6
    assert set(groups['NATURAL99']).isdisjoint(groups['CLEAN29'])
    assert set(groups['FULL128']) == set(groups['NATURAL99']) | set(groups['CLEAN29'])
    assert set(groups['FULL128']).isdisjoint(groups['WOOD45']) and set(groups['FULL128']) | set(groups['WOOD45']) == set(ids)
    positions = {name: [byid[fid] for fid in fids] for name, fids in groups.items()}
    positions['ALL173'] = list(range(173))
    operations = values['operational_metrics']
    arrays = {m: np.asarray([[v[k] if v['available'] else np.inf for k in KEYS]
                             for fid in ids for v in [records[fid]]]) for m, records in operations.items()}
    assert all(a.shape == (173, 2) and np.isfinite(a).all() and (a >= 0).all() for a in arrays.values())
    hypotheses = values['stable_pose_candidates']
    for model, records in hypotheses.items():
        assert set(records) == set(ids)
        for record in records.values():
            hs = record['hypotheses']
            assert len(hs) == 2 and {h['name'] for h in hs} == {'long-face-front', 'short-face-front'}
            assert all(h['pose']['available'] for h in hs)
    with (ROOT / protocol['inputs']['candidate_rows']['path']).open(newline='') as f:
        cached = list(csv.DictReader(f))
    assert len(cached) == 9 * 173 * 2
    pools, seen = {}, set()
    for row in cached:
        model, fid, axis, hyp = (row[k] for k in ('model', 'id', 'oracle', 'selected_whole_pose'))
        assert model in arrays and fid in byid and axis in ('T_best', 'R_best') and row['diagnostic_only'] == 'True'
        assert (model, fid, axis) not in seen
        seen.add((model, fid, axis))
        assert hyp in ('long-face-front', 'short-face-front') and row['recording'] == metadata[byid[fid]]['recording']
        error = (float(row['T_cm']), float(row['R_deg']))
        assert np.isfinite(error).all() and min(error) >= 0
        old = pools.setdefault((model, fid), {}).setdefault((model, hyp), error)
        assert old == error
    scale = np.asarray(protocol['scale'], np.float64)
    assert scale.tolist() == [values['source_train_gate']['source_scale'][k] for k in ('sT_cm', 'sR_deg')]
    assert scale.tolist() == [2.4636887551191258, 1.113474019956766]
    with (DOC / 'FRAME_RESULTS.csv').open(newline='') as f:
        frame_rows = list(csv.DictReader(f))
    assert len(frame_rows) == 1038
    reported_rows = {(r['diagnostic'], int(r['seed']), r['id']): r for r in frame_rows}
    assert len(reported_rows) == 1038
    selected, trace_checks = {}, 0
    for diagnostic in DIAGNOSTICS:
        selected[diagnostic] = np.empty((3, 173, 2))
        for seed in SEEDS:
            for i, fid in enumerate(ids):
                pool = dict(pools['R0', fid], **{})
                pool.update(pools[f'DIVERSE251_s{seed}', fid])
                names = list(pool)
                # Independent vectorized dominance matrix; no evaluator import.
                errors = np.asarray([pool[n] for n in names])
                dominates = (errors[:, None] <= errors[None]).all(-1) & (errors[:, None] < errors[None]).any(-1)
                keep = np.flatnonzero(~dominates.any(0))
                key = lambda j: (names[j][0] != 'R0', names[j][1], names[j][0])
                if diagnostic == 'AXIS_LOWER_BOUND':
                    index = [min(keep, key=lambda j: (errors[j, k], key(j))) for k in range(2)]
                    expected_error = np.asarray([errors[index[k], k] for k in range(2)])
                    chosen = None
                else:
                    cost = np.max(errors / scale, axis=1)
                    best = cost[keep].min()
                    j = min([j for j in keep if cost[j] == best], key=key)
                    index = [j, j]
                    expected_error = errors[j]
                    chosen = names[j]
                selected[diagnostic][seed - 1, i] = expected_error
                row = reported_rows[diagnostic, seed, fid]
                assert row['available'] == row['GT_derived_diagnostic_only'] == 'True'
                assert row['physical_complete_pose'] == str(chosen is not None)
                assert np.array_equal(np.asarray([float(row['T_cm']), float(row['R_deg'])]), expected_error)
                for k, prefix in enumerate(('T', 'R')):
                    assert (row[f'{prefix}_source_model'], row[f'{prefix}_source_hypothesis']) == names[index[k]]
                if chosen is None:
                    assert row['selected_model'] == row['selected_hypothesis'] == row['fixed_TRAIN_cost'] == ''
                else:
                    assert (row['selected_model'], row['selected_hypothesis']) == chosen
                    assert float(row['fixed_TRAIN_cost']) == best
                assert int(row['deduplicated_union_candidates']) == len(pool)
                assert int(row['nondominated_union_candidates']) == len(keep)
                trace_checks += 1
    assert (selected[DIAGNOSTICS[0]] <= selected[DIAGNOSTICS[1]]).all()
    names = dict(SINGLE251=[f'SINGLE251_s{s}' for s in SEEDS], DIVERSE251=[f'DIVERSE251_s{s}' for s in SEEDS])
    names.update({m: [m] * 3 for m in FIXED})
    original_tensors = {name: np.stack([arrays[m] for m in seq]) for name, seq in names.items()}
    original_gate, leaves = original_gates(original_tensors, positions, metadata, values['stable_results']['hierarchy'])
    leaves += nested_equal(original_gate, values['stable_results']['stability']['gates'])
    diagnostic_checks = {}
    for diagnostic in DIAGNOSTICS:
        reported = result['diagnostics'][diagnostic]
        tensors = dict(original_tensors, DIVERSE251=selected[diagnostic])
        gates, count = original_gates(tensors, positions, metadata, reported['hierarchy'])
        leaves += count + nested_equal(gates, reported['stability']['gates'])
        passed = all(v['PASS'] for v in gates.values())
        assert reported['stability']['PASS'] == passed and reported['stability']['diagnostic_only']
        assert not reported['method_success'] and not reported['goal_complete']
        for population, pos in positions.items():
            for arm, tensor in tensors.items():
                leaves += nested_equal(mean_summary(tensor[:, pos]), reported['summaries'][population][arm])
            for model in arrays:
                tensor = selected[diagnostic][[int(model[-1]) - 1]][:, pos] if model.startswith('DIVERSE251_s') else arrays[model][None, pos]
                leaves += nested_equal(mean_summary(tensor), reported['per_seed_summaries'][population][model])
        for population, recs in reported['by_recording'].items():
            for rec, values_rec in recs.items():
                pos = [i for i in positions[population] if metadata[i]['recording'] == rec]
                assert values_rec['frames'] == len(pos)
                for arm, tensor in tensors.items():
                    leaves += nested_equal(mean_summary(tensor[:, pos]), values_rec['summaries'][arm])
        diagnostic_checks[diagnostic] = dict(PASS=True, original_gate_PASS=passed,
                                             gates={k: v['PASS'] for k, v in gates.items()},
                                             all_error_pairs_finite=True, frame_choices_checked=519)
    lb, whole = [result['diagnostics'][d] for d in DIAGNOSTICS]
    assert result['finite_contract']['necessary_exclusion_logic_valid']
    assert lb['interpretation']['necessary_exclusion_logic_valid'] and not lb['interpretation']['PASS_is_joint_feasibility_proof']
    assert lb['interpretation']['fixed_union_goal_excluded'] == (not lb['stability']['PASS'])
    assert whole['interpretation']['gate_passing_whole_pose_witness'] == whole['stability']['PASS']
    assert not whole['interpretation']['FAIL_excludes_all_whole_pose_selections'] and not whole['interpretation']['deployable']
    assert all(not whole['stability']['gates'][k]['PASS'] or lb['stability']['gates'][k]['PASS'] for k in original_gate)
    assert 'R0_ONLY' not in arrays and 'R0_ONLY' in protocol['unmeasured_comparator']
    for key in ('image_forwards', 'new_fits', 'optimizer_steps', 'new_PnP_calls', 'raw_GT_reads', 'learned_checkpoints_read'):
        assert result[key] == 0
    for binding in frozen_bindings:
        verify(binding)
    out = dict(complete=True, PASS=True, created_at=datetime.now(timezone.utc).isoformat(),
        diagnostic_only=True, method_success=False, goal_complete=False,
        implementation='No experiment helper imported. Independent dominance matrix, fixed minmax selector, sorted-rank quantiles, paired recording/seed bootstrap and original five gates.',
        results=bind(DOC / 'RESULTS.json'), frame_rows=bind(DOC / 'FRAME_RESULTS.csv'),
        protocol=bind(DOC / 'PROTOCOL.json'), verifier=bind(Path(__file__)),
        input_code_artifact_bindings=frozen_bindings, binding_count=len(frozen_bindings),
        frame_rows_checked=trace_checks, scalar_leaves_checked=leaves,
        baseline_models_unchanged=True, historical_five_gate_parity=True,
        diagnostics=diagnostic_checks, allfinite_necessary_bound_logic_verified=True,
        fixed_cost_target_limit=dict(
            perfect_target_selection_meets_original_goal=whole['stability']['PASS'],
            natural_T_P90_vs_R0=whole['stability']['gates']['natural_tails']['comparisons']['R0']['checks']['P90:translation_cm'],
            all_other_union_selections_excluded=lb['interpretation']['fixed_union_goal_excluded']),
        learned_R0_ONLY_real_comparator_absent=True, expanded_UNION_method_success_not_established=True,
        raw_GT_reads=0, new_image_forwards=0, new_fits=0, new_PnP_calls=0,
        numeric_atol=1e-12, all_gate_booleans_exact=True, read_paths=sorted(READS))
    with (DOC / 'VERIFICATION.json').open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    text = ['# 고정 real union 가능성 진단의 독립 검산', '',
            '**검산 PASS**. 이것은 후보 가능성 진단의 재현 확인이며 실제 모델 개선이나 전체 목표 달성이 아니다.', '',
            f'- 원래 5개 gate와 두 진단의 5개 gate를 독립 계산했다. {trace_checks:,}개 프레임/seed/진단 행, {leaves:,}개 수치·상태 leaf를 확인했다.',
            '- repository 평가 helper를 import하지 않았다. 독립 dominance matrix와 정렬 순위 보간으로 후보 선택·median/P90을 계산하고, recording×seed bootstrap 2,000회(seed 20261001)와 6회 recording 제외 검사를 재현했다.',
            '- 원래 baseline 오차와 gate, 각 모집단·seed·recording 요약이 저장 결과와 일치했다. Boolean 판정은 정확히 같고 부동소수점 값의 검산 허용치는 1e-12이다.', '',
            '| 진단 | 원래 5개 gate 전체 | 확인 가능한 결론 |', '|---|---|---|']
    for diagnostic in DIAGNOSTICS:
        passed = diagnostic_checks[diagnostic]['original_gate_PASS']
        if diagnostic == DIAGNOSTICS[0]:
            conclusion = '필요조건 통과이며 공동 달성 증명은 아님' if passed else '동일 후보·동일 원래 gate에서 불가능성을 보이는 필요조건 실패'
        else:
            conclusion = '원래 gate를 만족하는 GT 기반 whole-pose 선택 조합의 존재' if passed else '고정 TRAIN cost 규칙 실패이며 모든 다른 선택 조합을 배제하지 않음'
        text.append(f"| {diagnostic} | {'PASS' if passed else 'FAIL'} | {conclusion} |")
    tail = out['fixed_cost_target_limit']['natural_T_P90_vs_R0']
    text += ['', f"고정 TRAIN-cost target을 매 프레임 완벽하게 선택해도 자연 T P90의 seed 평균은 {tail['after']['value']:.12f} cm로 R0 기준 상한 {tail['limit']['value']:.12f} cm를 넘는다. 따라서 현재 고정 target을 완벽히 모방하는 것 자체로 원래 전체 목표가 달성되지 않는다. 이 실패는 다른 whole-pose 선택 조합의 불가능성을 증명하지 않으며, 축별 낙관적 하한이 통과했다는 결과와 모순되지 않는다.", '',
             '모든 기존 운영 모델 및 union 진단의 오류가 유한하고, 원래 실패 수 증가 금지 조건을 유지했다. 따라서 같은 분모·bootstrap draw에서 축별 하한의 단조 필요조건 해석이 유효하다. 축별 최솟값은 서로 다른 pose에서 올 수 있으므로 물리적 pose로 해석하지 않는다.', '',
             '고정 cost oracle은 T와 R를 같은 완전한 후보 pose에서 가져온다. source TRAIN의 기존 scale, 정확한 cost 동률·Pareto 제거·R0/hypothesis/expert 우선순위를 그대로 사용했다. 이미지·checkpoint·raw GT를 읽거나 새 학습·PnP·learned real routing을 수행하지 않았다.', '',
             '**learned R0_ONLY real 비교 결과는 없다.** 이 진단의 원래 SINGLE251/R0/PRIOR1/FULL125 비교는 후속 UNION 실험의 확장 비교 계약을 대체하지 않는다. Oracle 통과는 배포 가능성·학습 가능성·새 recording 일반화·전체 목표 달성의 증거가 아니다.', '',
             '[독립 검산 JSON](VERIFICATION.json), [진단 결과](RESULTS.json), [동결 프로토콜](PROTOCOL.json), [검산 코드](../../../scripts/research/pallet_pose_real_union_feasibility_20261001_v1/verify_results.py)', '']
    with (DOC / 'VERIFICATION_KO.md').open('x') as f:
        f.write('\n'.join(text))
    print('INDEPENDENT_REAL_UNION_DIAGNOSTIC_VERIFICATION_PASS', diagnostic_checks, 'leaves', leaves)


if __name__ == '__main__':
    main()
