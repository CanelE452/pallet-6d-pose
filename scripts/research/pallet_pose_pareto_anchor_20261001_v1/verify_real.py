"""Independent cached-result audit; no experiment helper or raw GT import."""
import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_pareto_anchor_20261001_v1'
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
        assert p.parent == DOC and p.name in ('REAL_VERIFICATION.json', 'REAL_VERIFICATION_KO.md'), path
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
    result = read(DOC / 'REAL_FEASIBILITY.json')
    assert result['complete'] and result['diagnostic_only'] and not result['method_success'] and not result['goal_complete']
    assert result['protocol'] == bind(DOC / 'PROTOCOL.json')
    assert result['inputs'] == protocol['inputs'] and result['codes'] == protocol['codes']
    bindings = [*protocol['inputs'].values(), *protocol['codes'], *result['artifacts']]
    for b in bindings:
        verify(b)
    values = {k: read(ROOT / b['path']) for k,b in protocol['inputs'].items() if k != 'candidate_rows'}
    assert values['input_audit']['complete'] and values['input_audit']['PASS']
    assert values['previous_feasibility']['protocol'] == protocol['inputs']['previous_protocol']
    for k,b in values['previous_protocol']['inputs'].items():
        assert protocol['inputs'][k] == b
    assert protocol['inputs']['candidate_rows'] in values['candidate_bounds']['artifacts']
    assert protocol['inputs']['operational_metrics'] in values['stable_results']['artifacts']
    assert values['stable_results']['protocol'] == protocol['inputs']['stable_protocol']
    assert values['stable_protocol_sha'] == protocol['inputs']['stable_protocol']
    metadata = values['eval_metadata']; ids = [r['id'] for r in metadata]
    assert len(ids) == len(set(ids)) == 173
    indices = {fid:i for i,fid in enumerate(ids)}
    groups = values['eval_groups']
    assert {p:len(groups[p]) for p in ('NATURAL99','CLEAN29','WOOD45')} == dict(NATURAL99=99,CLEAN29=29,WOOD45=45)
    assert len({metadata[indices[i]]['recording'] for i in groups['NATURAL99']}) == 6
    assert set(groups['NATURAL99']).isdisjoint(groups['CLEAN29'])
    assert set(groups['FULL128']) == set(groups['NATURAL99']) | set(groups['CLEAN29'])
    assert set(groups['FULL128']).isdisjoint(groups['WOOD45']) and set(groups['FULL128']) | set(groups['WOOD45']) == set(ids)
    positions = {p:[indices[i] for i in fids] for p,fids in groups.items()}
    positions['ALL173'] = list(range(173))
    operations = values['operational_metrics']
    arrays = {m:np.asarray([[records[fid][k] if records[fid]['available'] else np.inf for k in KEYS] for fid in ids]) for m,records in operations.items()}
    assert len(arrays) == 9
    assert all(a.shape == (173,2) and np.isfinite(a).all() and (a>=0).all() for a in arrays.values())
    pose = values['stable_pose_candidates']
    for m in arrays:
        assert set(pose[m]) == set(ids)
        for r in pose[m].values():
            assert len(r['hypotheses']) == 2 and {h['name'] for h in r['hypotheses']} == {'long-face-front','short-face-front'}
            assert all(h['pose']['available'] for h in r['hypotheses'])
    with (ROOT/protocol['inputs']['candidate_rows']['path']).open(newline='') as f:
        records = list(csv.DictReader(f))
    assert len(records) == 9*173*2
    pools, seen = {},set()
    for row in records:
        model,fid,axis,hyp = (row[k] for k in ('model','id','oracle','selected_whole_pose'))
        assert model in arrays and fid in indices and axis in ('T_best','R_best') and row['diagnostic_only']=='True'
        assert (model,fid,axis) not in seen;seen.add((model,fid,axis))
        assert hyp in ('long-face-front','short-face-front') and row['recording']==metadata[indices[fid]]['recording']
        e=(float(row['T_cm']),float(row['R_deg']));assert np.isfinite(e).all() and min(e)>=0
        previous=pools.setdefault((model,fid),{}).setdefault((model,hyp),e)
        assert previous==e
    scale=np.asarray(protocol['scale'])
    assert scale.tolist()==[values['source_train_gate']['source_scale'][k] for k in ('sT_cm','sR_deg')]
    assert scale.tolist()==[2.4636887551191258,1.113474019956766]
    with (DOC/'REAL_FEASIBILITY_ROWS.csv').open(newline='') as f:
        frames=list(csv.DictReader(f))
    assert len(frames)==519
    lookup={(int(r['seed']),r['id']):r for r in frames};assert len(lookup)==519
    selected=np.empty((3,173,2));by_seed={};total=Counter();anchor_parity=0
    for seed in SEEDS:
        tally=Counter()
        for i,fid in enumerate(ids):
            anchor_record=pose['R0'][fid];anchor_name=('R0',anchor_record['GEO_name'])
            matches=[h['pose'] for h in anchor_record['hypotheses'] if h['name']==anchor_name[1]]
            assert len(matches)==1 and matches[0]==anchor_record['GEO_pose'] and matches[0]['available']
            anchor=tuple(arrays['R0'][i])
            pool=dict(pools['R0',fid]);pool.update(pools[f'DIVERSE251_s{seed}',fid])
            compressed_count=len(pool);restored=anchor_name not in pool
            if not restored:
                assert pool[anchor_name]==anchor
                anchor_parity+=1
            pool[anchor_name]=anchor
            names=list(pool);errors=np.asarray([pool[n] for n in names])
            eligible=np.flatnonzero((errors<=anchor).all(1))
            assert eligible.size and anchor_name in [names[j] for j in eligible]
            admissible=errors[eligible]
            dominance=(admissible[:,None]<=admissible[None]).all(-1)&(admissible[:,None]<admissible[None]).any(-1)
            frontier=eligible[~dominance.any(0)]
            cost=np.max(errors/scale,axis=1);mincost=cost[frontier].min()
            ties=[j for j in frontier if cost[j]==mincost]
            j=min(ties,key=lambda k:(names[k][0]!='R0',names[k][1],names[k][0]))
            selected[seed-1,i]=errors[j]
            assert (errors[j]<=anchor).all()
            counts=dict(compressed_candidates=compressed_count,anchor_augmented_candidates=len(pool),eligible_candidates=len(eligible),rejected_candidates=len(pool)-len(eligible),anchor_restored=restored,anchor_returned=names[j]==anchor_name,anchor_only_eligible=len(eligible)==1,strictly_dominating_eligible=int((errors[eligible]<anchor).any(1).sum()),T_strict_improvement=bool(errors[j,0]<anchor[0]),R_strict_improvement=bool(errors[j,1]<anchor[1]),both_strict_improvement=bool((errors[j]<anchor).all()))
            row=lookup[seed,fid]
            assert row['diagnostic']=='PARETO_ANCHOR_ORACLE' and row['recording']==metadata[i]['recording']
            assert row['available']==row['physical_complete_pose']==row['GT_derived_diagnostic_only']=='True'
            assert row['T_nonincrease']==row['R_nonincrease']=='True'
            assert np.array_equal(np.asarray([float(row['T_cm']),float(row['R_deg'])]),errors[j])
            assert (row['anchor_model'],row['anchor_hypothesis'])==anchor_name
            assert (float(row['anchor_T_cm']),float(row['anchor_R_deg']))==anchor
            assert (row['selected_model'],row['selected_hypothesis'])==names[j]
            for axis in ('T','R'):
                assert (row[axis+'_source_model'],row[axis+'_source_hypothesis'])==names[j]
            assert float(row['fixed_TRAIN_cost'])==mincost
            assert int(row['deduplicated_union_candidates'])==compressed_count
            assert int(row['nondominated_union_candidates'])==len(frontier)
            for key,value in counts.items():
                assert row[key]==str(value),(seed,fid,key,row[key],value)
            for key in ('anchor_restored','anchor_returned','anchor_only_eligible','T_strict_improvement','R_strict_improvement','both_strict_improvement'):
                tally[key]+=int(counts[key])
            for key in ('eligible_candidates','rejected_candidates','anchor_augmented_candidates','strictly_dominating_eligible'):
                tally[f'{key}:{counts[key]}']+=1
            tally['frames']+=1;tally['T_nonincrease']+=1;tally['R_nonincrease']+=1
        by_seed[f's{seed}']=dict(tally);total.update(tally)
    assert np.isfinite(selected).all() and (selected<=arrays['R0'][None]).all()
    assert result['pointwise_anchor']['PASS'] and result['pointwise_anchor']['T_nonincrease']==result['pointwise_anchor']['R_nonincrease']==519
    assert result['pointwise_anchor']['counts']==dict(total) and result['pointwise_anchor']['by_seed']==by_seed
    assert result['fallback_counts']['anchor_returned']==total['anchor_returned']
    assert result['fallback_counts']['anchor_only_eligible']==total['anchor_only_eligible']
    names=dict(SINGLE251=[f'SINGLE251_s{s}' for s in SEEDS],DIVERSE251=[f'DIVERSE251_s{s}' for s in SEEDS])
    names.update({m:[m]*3 for m in FIXED})
    original={name:np.stack([arrays[m] for m in seq]) for name,seq in names.items()}
    oldgates,leaves=original_gates(original,positions,metadata,values['stable_results']['hierarchy'])
    leaves+=nested_equal(oldgates,values['stable_results']['stability']['gates'])
    tensors=dict(original,DIVERSE251=selected)
    gates,count=original_gates(tensors,positions,metadata,result['hierarchy'])
    leaves+=count+nested_equal(gates,result['stability']['gates'])
    passed=all(v['PASS'] for v in gates.values())
    assert result['PASS']==result['stability']['PASS']==passed
    assert result['stability']['diagnostic_only'] and not result['stability']['method_success'] and not result['stability']['goal_complete']
    for population,pos in positions.items():
        for arm,tensor in tensors.items():
            leaves+=nested_equal(mean_summary(tensor[:,pos]),result['summaries'][population][arm])
        for model in arrays:
            tensor=selected[[int(model[-1])-1]][:,pos] if model.startswith('DIVERSE251_s') else arrays[model][None,pos]
            leaves+=nested_equal(mean_summary(tensor),result['per_seed_summaries'][population][model])
    for population,recordings in result['by_recording'].items():
        for recording,rv in recordings.items():
            pos=[i for i in positions[population] if metadata[i]['recording']==recording]
            assert rv['frames']==len(pos)
            for arm,tensor in tensors.items():
                leaves+=nested_equal(mean_summary(tensor[:,pos]),rv['summaries'][arm])
    previous=values['previous_feasibility']['diagnostics']['FIXED_TRAIN_COST_ORACLE']
    assert result['previous_fixed_cost_oracle']['binding']==protocol['inputs']['previous_feasibility']
    assert result['previous_fixed_cost_oracle']['PASS']==previous['stability']['PASS']==False
    assert result['previous_fixed_cost_oracle']['stability']==previous['stability']
    assert result['previous_fixed_cost_oracle']['summaries']==previous['summaries']
    assert 'R0_ONLY' not in arrays
    for key in ('image_forwards','new_fits','optimizer_updates','new_PnP','new_reference_metric_calls','raw_GT_reads','learned_checkpoints_read','new_learned_real_routing'):
        assert result[key]==0
    for b in bindings:verify(b)
    natural=mean_summary(selected[:,positions['NATURAL99']])
    out=dict(complete=True,PASS=True,created_at=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,method_success=False,goal_complete=False,implementation='Independent cached CSV/anchor restoration, vectorized Pareto dominance, exact minmax tie, sorted-rank quantiles and original 5-gate recording/seed bootstrap; no repository helper import.',results=bind(DOC/'REAL_FEASIBILITY.json'),rows=bind(DOC/'REAL_FEASIBILITY_ROWS.csv'),protocol=bind(DOC/'PROTOCOL.json'),verifier=bind(Path(__file__)),bindings=bindings,frame_seed_rows_checked=519,scalar_leaves_checked=leaves,original_gate_parity=True,diagnostic_gate_PASS=passed,gates={k:v['PASS'] for k,v in gates.items()},natural_summary=natural,pointwise_R0_nonincrease519=True,anchor_counts=dict(total),anchor_counts_by_seed=by_seed,existing_anchor_identity_error_pairs_checked=anchor_parity,all173_operational_and_selected_errors_finite=True,missing_contract_unexercised=True,candidate_count_scope='Compressed stored error candidates plus restored anchor, not all original candidate identity counts.',expanded_UNION_contract_success_not_established=True,learned_R0_ONLY_real_comparator_absent=True,previous_fixed_cost_oracle_FAIL_preserved=True,raw_GT_reads=0,new_fits=0,new_image_forwards=0,new_PnP_calls=0,new_learned_real_routing=0,numeric_atol=1e-12,all_gate_booleans_exact=True,read_paths=sorted(READS))
    with (DOC/'REAL_VERIFICATION.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    t=natural['conditional'];med=t['median'];tail=t['P90']
    md=['# Pareto anchor 실사 진단의 독립 검산','',f'**검산 PASS**. 원래 5개 안정성 gate의 진단 판정은 **{"PASS" if passed else "FAIL"}**이며, 실제 배포 가능한 모델이나 전체 목표 성공을 뜻하지 않는다.','',f'519개 frame×seed 선택을 독립 복원하고 {leaves:,}개 통계·상태 leaf를 확인했다. 원래 기준 gate, 필요한 2,000회 recording×seed bootstrap(seed 20261001), 6회 recording 제외, 모집단·seed·recording 요약이 저장 결과와 일치한다. Boolean은 정확히 같으며 float 허용치는 1e-12다.','',f"자연99의 seed별 quantile 평균은 T median {med['translation_cm']['value']:.12f} cm, R median {med['rotation_deg']['value']:.12f}°, T P90 {tail['translation_cm']['value']:.12f} cm, R P90 {tail['rotation_deg']['value']:.12f}°다.",'',f"R0 operational GEO anchor를 명시적으로 복원하고, 같은 identity가 캐시에 존재한 {anchor_parity}건의 오류 쌍이 정확히 같음을 확인했다. 519/519 선택은 T와 R 모두 R0 anchor 이하이며 완전한 단일 pose다. Anchor 복원 {total['anchor_restored']}건, anchor identity 반환 {total['anchor_returned']}건, 보존한 후보 집합에서 anchor만 허용된 경우 {total['anchor_only_eligible']}건이다. 이 후보 수는 압축 캐시 + anchor 기준이며 원래 네 후보 identity 전체의 census가 아니다.",'','원래 고정 cost oracle의 실패 기록을 그대로 대조했다. 여기서 바뀐 것은 같은 frame의 R0 T/R를 모두 보존하는 허용 집합이다. 이것은 기존 target의 한계를 보완하는 GT 기반 후보 가능성 증거이며, 새 학습 결과나 실제 runtime 개선으로 바꾸어 말하지 않는다.','', '현재 모든173 입력이 유효하므로 결측 anchor 분기는 실행되지 않았다. 프로토콜의 실패 유지 서술과 구현의 예상 밖 결측 STOP 차이는 [구현 검토](REAL_METHOD_REVIEW_KO.md)에 명시했다. learned R0_ONLY real 비교는 없고 원래 SINGLE251/R0/PRIOR1/FULL125 기준만 검사하므로 후속 UNION 확장 계약의 전체 성공도 아니다.','', '[검산 JSON](REAL_VERIFICATION.json), [진단 결과](REAL_FEASIBILITY.json), [독립 검산 코드](../../../scripts/research/pallet_pose_pareto_anchor_20261001_v1/verify_real.py)','']
    with (DOC/'REAL_VERIFICATION_KO.md').open('x') as f:f.write('\n'.join(md))
    print('INDEPENDENT_PARETO_ANCHOR_REAL_VERIFICATION_PASS',passed,'rows519','leaves',leaves,flush=True)


if __name__=='__main__':
    main()
