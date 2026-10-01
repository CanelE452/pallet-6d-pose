"""Describe frozen real choices versus an existing oracle; no new routing."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import numpy as np
from . import common as C

POPS = ('NATURAL99', 'CLEAN29', 'WOOD45')
CLASSES = ('anchor', 'safe_improvement', 'safe_equal', 'unsafe_T_only', 'unsafe_R_only', 'unsafe_both', 'failed')


def numbers(values):
    values = np.asarray(values, np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    return dict(n=len(values), median=float(np.quantile(values, .5)) if len(values) else None,
                P90=float(np.quantile(values, .9)) if len(values) else None,
                minimum=float(values.min()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def category(name, anchor_name, error, anchor):
    if error is None:
        return 'failed'
    delta = np.asarray(error) - np.asarray(anchor)
    if name == anchor_name:
        assert np.array_equal(error, anchor)
        return 'anchor'
    t, r = bool(delta[0] > 0), bool(delta[1] > 0)
    if t or r:
        return 'unsafe_both' if t and r else 'unsafe_T_only' if t else 'unsafe_R_only'
    return 'safe_improvement' if np.any(delta < 0) else 'safe_equal'


def transition(name, anchor_name):
    if name is None:
        return 'failed'
    expert, hyp = name.split(':')
    _, anchor_hyp = anchor_name.split(':')
    if name == anchor_name:
        return 'anchor'
    return ('R0' if expert == 'R0' else 'refiner') + ('_same_WD' if hyp == anchor_hyp else '_WD_switch')


def summarize(rows):
    classes = Counter(r['class'] for r in rows)
    present = [r for r in rows if r['available']]
    oracle = [r for r in rows if r['oracle'] is not None]
    opportunities = [r for r in oracle if r['oracle']['strict_safe_opportunity']]
    misses = [r for r in opportunities if not r['safe_improvement']]
    out = dict(frames=len(rows), available=len(present), failed=len(rows)-len(present),
        classes={k: classes[k] for k in CLASSES},
        class_fractions_full_population={k: classes[k]/len(rows) for k in CLASSES},
        transitions_from_anchor=dict(Counter(r['transition'] for r in rows)),
        selected_experts=dict(Counter(r['parent'] for r in rows)),
        exact_best_score_ties=sum(r['best_score_tie'] for r in rows),
        error_equal_nonanchor=classes['safe_equal'],
        T_cm=numbers([r['T_cm'] for r in present]), R_deg=numbers([r['R_deg'] for r in present]),
        delta_T_cm=numbers([r['delta_T_cm'] for r in present]),
        delta_R_deg=numbers([r['delta_R_deg'] for r in present]),
        unsafe_normalized_excess=numbers([r['normalized_excess'] for r in present if r['class'].startswith('unsafe')]),
        paired_R0_T_cm=numbers([r['anchor_T_cm'] for r in rows]),
        paired_R0_R_deg=numbers([r['anchor_R_deg'] for r in rows]))
    if oracle:
        out['oracle_comparison'] = dict(frames=len(oracle), opportunities=len(opportunities),
            exact_identity_hits=sum(r['oracle']['exact_identity_hit'] for r in oracle),
            opportunities_captured=sum(r['safe_improvement'] for r in opportunities),
            opportunities_missed=len(misses),
            opportunity_capture_fraction=sum(r['safe_improvement'] for r in opportunities)/len(opportunities) if opportunities else None,
            miss_classes=dict(Counter(r['class'] for r in misses)),
            miss_oracle_transitions=dict(Counter(r['oracle']['transition'] for r in misses)),
            oracle_transitions=dict(Counter(r['oracle']['transition'] for r in oracle)),
            oracle_vs_actual_transition_matrix=dict(Counter(r['oracle']['transition']+' -> '+r['transition'] for r in oracle)),
            expert_mismatch=sum(r['oracle']['expert_mismatch'] for r in oracle),
            WD_mismatch=sum(r['oracle']['WD_mismatch'] for r in oracle),
            different_but_safe_improvement=sum(r['safe_improvement'] and not r['oracle']['exact_identity_hit'] for r in oracle),
            oracle_T_cm=numbers([r['oracle']['T_cm'] for r in oracle]),
            oracle_R_deg=numbers([r['oracle']['R_deg'] for r in oracle]))
    else:
        out['oracle_comparison'] = None
    return out


def load():
    results_path = C.DOC/'REAL_RESULTS.json'
    result = C.read(results_path)
    assert result['complete']
    for b in [result['protocol'], result['routing_lock'], *result['artifacts']]:
        C.verify(b)
    lock = C.read(C.ROOT/result['routing_lock']['path'])
    assert lock['complete'] and lock['frames'] == 173 and lock['models'] == list(C.MODEL_NAMES)
    assert lock['protocol'] == result['protocol'] and not lock['real_reference_values_read']
    assert lock['runtime_uses_margin'] is False
    C.verify(lock['choices'])
    choices = C.read(C.ROOT/lock['choices']['path'])
    assert choices['models'] == list(C.MODEL_NAMES) and len(choices['ids']) == len(set(choices['ids'])) == 173
    metrics_binding = next(b for b in result['artifacts'] if b['path'].endswith('/POSE_METRICS.json'))
    metrics = C.read(C.ROOT/metrics_binding['path'])
    for key in ('metadata', 'groups'):
        C.verify(lock['inputs'][key])
    metadata = C.read(C.ROOT/lock['inputs']['metadata']['path'])
    groups = C.read(C.ROOT/lock['inputs']['groups']['path'])
    assert [r['id'] for r in metadata] == choices['ids']
    assert {p: len(groups[p]) for p in POPS} == dict(NATURAL99=99, CLEAN29=29, WOOD45=45)
    assert set().union(*(set(groups[p]) for p in POPS)) == set(choices['ids'])
    assert sum(len(groups[p]) for p in POPS) == 173
    oracle_path = C.ANCHOR_DOC/'REAL_FEASIBILITY.json'
    oracle = C.read(oracle_path)
    C.verify(oracle['protocol'])
    assert oracle['complete'] and oracle['diagnostic_only'] and oracle['frame_seed_rows'] == 519
    pairs = [('eval_metadata','metadata'),('eval_groups','groups'),('stable_pose_candidates','poses'),
             ('stable_pose_lock','pose_lock'),('stable_prediction_lock','prediction_lock'),('stable_protocol','stable_protocol')]
    for old, new in pairs:
        assert oracle['inputs'][old] == lock['inputs'][new]
        C.verify(oracle['inputs'][old])
    csv_binding = next(b for b in oracle['artifacts'] if b['path'].endswith('/REAL_FEASIBILITY_ROWS.csv'))
    C.verify(csv_binding)
    with (C.ROOT/csv_binding['path']).open() as f:
        oracle_rows = {(int(r['seed']),r['id']): r for r in csv.DictReader(f)}
    assert set(oracle_rows) == {(s,i) for s in (1,2,3) for i in choices['ids']}
    train_path = C.DOC/'TRAIN_CONVERGENCE.json'
    train = C.read(train_path)
    assert train['complete'] and train['PASS'] and train['source_TRAIN_only']
    previous_train_path = C.CONTEXT_DOC/'TRAIN_RISK_DIAGNOSTIC.json'
    previous_train = C.read(previous_train_path)
    assert previous_train['complete'] and previous_train['PASS']
    for model in C.MODEL_NAMES:
        fb = next(b for b in train['bindings'] if b['path'] == str((C.DOC/f'FIT_{model}.json').relative_to(C.ROOT)))
        C.verify(fb); receipt = C.read(C.ROOT/fb['path'])
        assert receipt['target_sha'] == previous_train['models'][model]['target_index_sha']
    C.verify(lock['source_val_gate'])
    source_gate = C.read(C.ROOT/lock['source_val_gate']['path'])
    assert source_gate['complete'] and source_gate['PASS'] and source_gate['checks_total'] == source_gate['checks_passed'] == 45
    bindings = [C.bind(results_path), result['protocol'], result['routing_lock'], lock['choices'], metrics_binding,
                lock['inputs']['metadata'], lock['inputs']['groups'], C.bind(oracle_path), oracle['protocol'], csv_binding,
                C.bind(train_path), C.bind(previous_train_path), lock['source_val_gate'], C.bind(Path(__file__))]
    return result, choices, metrics, metadata, groups, oracle, oracle_rows, train, previous_train, bindings


def run():
    result, choices, metrics, metadata, groups, oracle, oracle_rows, train, previous_train, bindings = load()
    scale = np.asarray(oracle['scale'], np.float64)
    assert np.isfinite(scale).all() and (scale > 0).all()
    population = {fid: p for p in POPS for fid in groups[p]}
    rows = []
    for model in C.MODEL_NAMES:
        for meta in metadata:
            fid = meta['id']
            choice, metric, base = choices['records'][model][fid], metrics[model][fid], metrics['R0'][fid]
            assert base['available'] and metric['available'] and choice['pose_available']
            anchor = np.array([base['translation_cm'],base['rotation_deg']],np.float64)
            error = np.array([metric['translation_cm'],metric['rotation_deg']],np.float64)
            assert np.isfinite(anchor).all() and np.isfinite(error).all()
            assert choice['anchor_name'] == choices['records']['R0_ONLY'][fid]['anchor_name']
            cls = category(choice['candidate_name'], choice['anchor_name'], error, anchor)
            valid_scores = [s for s,v in zip(choice['scores'],choice['valid']) if v]
            assert valid_scores and all(v is not None and np.isfinite(v) for v in valid_scores)
            trans = transition(choice['candidate_name'],choice['anchor_name'])
            row = dict(model=model, seed=None if model=='R0_ONLY' else int(model[-1]), id=fid,
                population=population[fid], recording=meta['recording'], available=True,
                candidate_name=choice['candidate_name'], anchor_name=choice['anchor_name'], parent=choice['parent'],
                hypothesis=choice['hypothesis'], **{'class':cls}, safe_improvement=cls=='safe_improvement',
                transition=trans, best_score_tie=sum(s==min(valid_scores) for s in valid_scores)>1,
                T_cm=float(error[0]),R_deg=float(error[1]),anchor_T_cm=float(anchor[0]),anchor_R_deg=float(anchor[1]),
                delta_T_cm=float(error[0]-anchor[0]),delta_R_deg=float(error[1]-anchor[1]),
                normalized_excess=float(max(0.,*((error-anchor)/scale))), oracle=None)
            if model!='R0_ONLY':
                old = oracle_rows[int(model[-1]),fid]
                np.testing.assert_array_equal(anchor, [float(old['anchor_T_cm']),float(old['anchor_R_deg'])])
                assert choice['anchor_name'] == 'R0:'+old['anchor_hypothesis']
                chosen_name = old['selected_model']+':'+old['selected_hypothesis']
                oracle_error = np.array([float(old['T_cm']),float(old['R_deg'])])
                assert np.all(oracle_error<=anchor)
                opportunity = bool(np.any(oracle_error<anchor))
                assert opportunity == (int(old['strictly_dominating_eligible'])>0)
                hit = chosen_name==choice['candidate_name']
                if hit:
                    np.testing.assert_array_equal(oracle_error,error)
                row['oracle'] = dict(candidate_name=chosen_name,T_cm=float(oracle_error[0]),R_deg=float(oracle_error[1]),
                    strict_safe_opportunity=opportunity, exact_identity_hit=hit,
                    transition=transition(chosen_name,choice['anchor_name']),
                    expert_mismatch=choice['parent']!=old['selected_model'],
                    WD_mismatch=choice['hypothesis']!=old['selected_hypothesis'],
                    opportunity_missed=opportunity and cls!='safe_improvement')
            rows.append(row)
    assert len(rows)==692 and len({(r['model'],r['id']) for r in rows})==692
    populations = {}
    for pop in (*POPS,'ALL173'):
        subset = [r for r in rows if pop=='ALL173' or r['population']==pop]
        populations[pop] = dict(models={m:summarize([r for r in subset if r['model']==m]) for m in C.MODEL_NAMES})
        populations[pop]['by_recording'] = {rec:{m:summarize([r for r in subset if r['recording']==rec and r['model']==m])
                                                       for m in C.MODEL_NAMES} for rec in sorted({r['recording'] for r in subset})}
    comparison = {model:dict(TRAIN_selected=train['models'][model]['risk_statistics'],
                       TRAIN_target=previous_train['models'][model]['target'],
                       real_populations={p:populations[p]['models'][model] for p in POPS}) for model in C.MODEL_NAMES}
    output = dict(complete=True,PASS=True,diagnostic_only=True,method_success=False,goal_complete=False,
        created_at=C.now(),source_gate_summary=dict(passed=45,total=45),real_gate_PASS=result['stability']['PASS'],
        real_gate_categories={k:v['PASS'] for k,v in result['stability']['gates'].items()},
        actual_choice_rows=692,paired_UNION_oracle_rows=519,populations=populations,TRAIN_comparison=comparison,rows=rows,
        oracle_scope='Existing GT-derived anchored whole-pose oracle reused without reselection. R0_ONLY has a smaller pool and is not graded against the four-candidate UNION oracle.',
        miss_definition='Oracle has a strict pointwise-safe improvement, but the actual choice has no strict pointwise-safe improvement; another safe improvement counts as captured even when not the exact oracle identity.',
        ties_definition='Exact minimum-score ties and exact error-equal non-anchor poses are separate. No epsilon or threshold is introduced.',
        interpretation_limits='Observed frozen choice/metric transitions, not proof of the reason for domain transfer. TRAIN labels/source errors and real geometry-derived errors differ in provenance; real results are reused DEV, not an independent population.',
        scale_sT_cm_sR_deg=scale,bindings=bindings,raw_GT_reads=0,new_metric_calls=0,new_PnP=0,image_forwards=0,new_fits=0,
        new_selector_routes=0,threshold_sweeps=0,source_VAL_metric_arrays_read=False,cached_reference_errors_used=True,
        input_contract='One RGB image, supplied physical dimensions, existing calibrated K. Diagnostic references are not runtime inputs.')
    C.save(C.DOC/'SELECTOR_TRANSFER_DIAGNOSTIC.json',output)
    report(output)
    print('SELECTOR_TRANSFER_DIAGNOSTIC_COMPLETE_692_FROZEN_CHOICES',flush=True)


def report(x):
    lines=['# 고정 selector의 source→실사 전환 진단','',
        'source45/45 통과 뒤 실사 원래 조건과 개입 대조군을 합친5개 항목 중2개만 통과했다. 새 성능이나 선택 규칙을 시험하지 않고 실제 고정692개 선택과 기존519개 anchored oracle 행을 대조했다. raw GT 재열람·metric 재계산·이미지 forward·PnP·fit은 모두0회다. 이미 저장된 GT 유래 오차를 쓰는 사후 진단이며 배포 정책이 아니다.','',
        '| 모집단 | 모델 | anchor | safe 개선 | 동오차 non-anchor | T만 악화 | R만 악화 | 양축 악화 | oracle 개선 기회 | 놓친 기회 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in POPS:
        for m,v in x['populations'][p]['models'].items():
            c=v['classes'];o=v['oracle_comparison']
            vals=[c[k] for k in ('anchor','safe_improvement','safe_equal','unsafe_T_only','unsafe_R_only','unsafe_both')]
            lines.append(f"| {p} | {m} | "+' | '.join(map(str,vals))+f" | {o['opportunities'] if o else 'NA'} | {o['opportunities_missed'] if o else 'NA'} |")
    lines+=['','Anchor는 기존 R0 operational GEO의 같은 후보 identity다. Safe 개선은 두 오차 모두 anchor 이하이면서 한 축 이상 엄격히 작을 때다. R0_ONLY는2후보이므로4후보 UNION oracle 기회 회수율을 적용하지 않는다. JSON은 모든 모집단·recording·seed별 집계와692행을 제공한다.','',
        '## 후보·expert·W/D 전환','',
        '| 모집단 | UNION seed | 실제 refiner 선택 | 실제 W/D 전환 | oracle refiner 선택 | oracle W/D 전환 | 정확 oracle identity 일치 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in POPS:
        for s in (1,2,3):
            v=x['populations'][p]['models'][f'UNION_s{s}'];o=v['oracle_comparison'];a=v['transitions_from_anchor'];b=o['oracle_transitions']
            lines.append(f"| {p} | {s} | {sum(n for k,n in a.items() if k.startswith('refiner'))} | {sum(n for k,n in a.items() if k.endswith('WD_switch'))} | {sum(n for k,n in b.items() if k.startswith('refiner'))} | {sum(n for k,n in b.items() if k.endswith('WD_switch'))} | {o['exact_identity_hits']} |")
    lines+=['','## NATURAL99 recording별 놓친 기회','',
        '| recording | seed | frame 수 | anchor | safe 개선 | unsafe | oracle 기회 | 놓친 기회 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for rec,models in x['populations']['NATURAL99']['by_recording'].items():
        for s in (1,2,3):
            v=models[f'UNION_s{s}'];c=v['classes'];o=v['oracle_comparison']
            lines.append(f"| {rec} | {s} | {v['frames']} | {c['anchor']} | {c['safe_improvement']} | {sum(n for k,n in c.items() if k.startswith('unsafe'))} | {o['opportunities']} | {o['opportunities_missed']} |")
    lines+=['','## TRAIN과 실사에서 관측한 차이','',
        '| 모델 | TRAIN target anchor/유효행 | TRAIN 선택 anchor/유효행 | NATURAL99 oracle anchor/frame | NATURAL99 선택 anchor/frame |',
        '|---|---:|---:|---:|---:|']
    for m,v in x['TRAIN_comparison'].items():
        t=v['TRAIN_target'];s=v['TRAIN_selected'];r=v['real_populations']['NATURAL99'];o=r['oracle_comparison']
        lines.append(f"| {m} | {t['classes']['anchor']['count']}/{t['available_rows']} | {s['classes']['anchor']['count']}/{s['available_rows']} | {o['oracle_transitions'].get('anchor',0) if o else 'NA'}/{r['frames']} | {r['classes']['anchor']}/{r['frames']} |")
    lines+=['','전체 TRAIN 분모는2,598행이고 실패1행을 유지했다. 위 anchor 비율만 정의 가능한2,597행을 표시한다. 실사173행은 모두 available이었다. 실제 선택/오차의 exact tie는 JSON에 별도 집계하며 임계값은 사용하지 않았다.','',
        '**관측:** natural에서 UNION은 대부분 anchor에 머물며 oracle의 pointwise-safe 개선 기회를 대부분 놓친다. 이번 실사 실패 항목은 공동 중앙값·불확실성·recording 민감도이고 tail/clean 보호 항목은 통과했다. 따라서 이번 결과를 많은 교정으로 tail이 무너진 실패라고 요약하면 틀리다. R0_ONLY의 W/D 교환에는 실제 악화도 있어 단순히 가설 전환을 늘리는 정책 역시 근거가 없다.','',
        '**추정과 한계:** TRAIN에서도 target보다 anchor를 더 자주 선택하며, 실사 oracle의 개선 기회 구성도 TRAIN과 다르다. 이는 보수적 supervision/의사결정과 domain shift를 검토할 근거다. 그러나 이 집계만으로 class 빈도, risk margin, 기하 특징, RGB 보정 품질 중 무엇이 원인인지 분리할 수 없다. 189특징의 표현력 부족이나 합성학습의 불가능성을 확정하지 않는다. source45는 해당 고정 합성 VAL의 비교 조건이며 실사 recording에 대한 보증이 아니었다.','',
        '## 다음 구조 변경 하나의 설계 제안 — 고정 RBF context64 추가','',
        '현재189특징은 후보 기하94·anchor 절댓값 차이94·identity1이며 직접 RGB appearance를 담지 않는다. 그 위의 공유 선형 결합만으로 서로 다른 기하·confidence·anchor 관계를 충분히 구분하는지와, 합성→실사 전이 또는 이미지 정보 부족인지는 현재 결과로 분리되지 않는다. 다음 한 가지 구조 실험은 기존189에 TRAIN에서 고정한 RBF64를 이어 붙인253차원 scorer다. 현재 target·risk margin·후보·원래 정규화·loss 계수·추론 argmin은 유지하고 고정 특징 map만 바꾼다.','',
        '센터 후보는 eligible TRAIN2,598행의 R0 및 세 frozen refiner의 유효 후보에서 얻은 기존189 context다. 반복된 R0 항목은(frame ID, expert, hypothesis) identity로 한 번만 세고, 이 identity의 SHA256 순으로 서로 다른 벡터64개를 고른다. label·오차·VAL·실사는 센터 선정에 쓰지 않는다. 양의 센터 쌍별 제곱거리의 median을 bandwidth 제곱값으로 한 번 고정하고 `exp(-||x-center||²/(2*bandwidth²))`64개를 추가한다. 센터64개나 양의 유한 bandwidth를 얻지 못하면 중단하며 다른개수·scale을 시험하지 않는다. 원래189 이후 추가 정규화는 없고 invalid 후보는253개 모두0으로 둔다.','',
        '이는 새 RGB 정보를 만드는 방법이 아니라 기존 특징들의 비선형 상호작용을 추가하는 최소 검증이다. 센터·bandwidth·특징을 target 품질 평가 전에 동결한다. 기존189weight 뒤에0을64개 붙인 score/objective 동등성을 확인하고, 동일한 margin CE+L2로 네 모델 각1회·zero init·같은solver 예산과 강볼록 인증을 적용한다. source45와 복구된 실사 원래/개입5개 AND 조건, 모든 seed·실패행·대조군은 유지한다. 센터 선정과 loss는 TRAIN만 사용하고, 런타임에는 기존 RGB에서 나온 후보기하·치수·K와 고정센터만 필요하며 GT나risk margin을 전달하지 않는다.','',
        '원래 gate 하나라도 실패하면 중단하며 threshold·margin배율·센터수·bandwidth sweep이나 좋은 seed 선택으로 구제하지 않는다. 이 구조가 TRAIN 적합을 개선해도 실사에서 RBF가 source support 밖으로 벗어나거나 직접RGB 단서가 부족할 수 있다. 따라서 실패는 모든 비선형 방법의 불가능성 증명이 아니고, 성공도 현재 표현력 부족만이 원인이었다는 인과 증명이 아니다.','',
        '[기존 방법 감사](../pallet_pose_union_selection_20261001_v1/PRIOR_AND_METHOD_AUDIT_KO.md)의 Stage4 관계기하+RGB context MLP 음성 결과, [objective 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)의 utility·보존·soft-cost 실패, [pairwise 실패](../pallet_pose_selector_pairwise_20261001_v1/REPORT_KO.md)를 그대로 유지한다. 일반 비선형 선택기는 이미 시험됐으므로 발명이나 해결책으로 주장하지 않는다. 제한된 관련 코드 RBF/Nyström 검색에서는 현재 고정 pool·target·risk loss와 같은 이 구조의 실행을 찾지 못했다. 새 fit은 아직 실행하지 않았으며 별도 사전 protocol이 필요하다.','',
        '[전체 집계·692행·해시](SELECTOR_TRANSFER_DIAGNOSTIC.json) · [실사 고정 결과](REAL_RESULTS.json) · [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [기존 oracle 진단](../pallet_pose_pareto_anchor_20261001_v1/REAL_FEASIBILITY.json)','']
    C.save(C.DOC/'SELECTOR_TRANSFER_DIAGNOSTIC_KO.md','\n'.join(lines))


def selfcheck():
    assert category('R0:a','R0:a',[1.,2.],[1.,2.])=='anchor'
    assert category('D:a','R0:a',[.5,2.],[1.,2.])=='safe_improvement'
    assert category('D:a','R0:a',[1.,2.],[1.,2.])=='safe_equal'
    assert category('D:a','R0:a',[2.,1.],[1.,2.])=='unsafe_T_only'
    assert category('D:b','R0:a',[.5,3.],[1.,2.])=='unsafe_R_only'
    assert category('D:b','R0:a',[2.,3.],[1.,2.])=='unsafe_both'
    assert transition('D:a','R0:a')=='refiner_same_WD' and transition('R0:b','R0:a')=='R0_WD_switch'
    print('TRANSFER_SELFCHECK_PASS_INVENTED_ONLY')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['selfcheck','run']);args=parser.parse_args()
    (selfcheck if args.stage=='selfcheck' else run)()
