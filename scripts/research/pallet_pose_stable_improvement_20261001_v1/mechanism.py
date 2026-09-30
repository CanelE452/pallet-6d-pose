"""Supplementary held-R0-W/D diagnosis after the full primary result is frozen.

No new candidate, coordinate, fit, image forward, reference-based selection,
or change to the primary verdict. The supplemental protocol is written before
new outcomes are read and the score entry point requires completed results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import evaluate as E

POPS = ('NATURAL99', 'CLEAN29', 'WOOD45')
TOL = 1e-7  # Existing diagnosis numerical parity/direction tolerance.


def prepare():
    path = C.DOC / 'MECHANISM_PROTOCOL.json'
    if path.exists():
        E.verify_file(C.read(C.DOC / 'MECHANISM_PROTOCOL_SHA.json'))
        print('MECHANISM_PROTOCOL_ALREADY_LOCKED', flush=True)
        return
    assert not (C.DOC / 'RESULTS.json').exists(), 'Supplement must be locked before primary outcomes exist'
    protocol = C.protocol()
    old_path = C.D.RAW / 'E2_BOX_DIAGNOSTIC.json'
    C.save(path, dict(created_at=C.now(), type='Supplemental diagnostic fixed before primary scoring; six-fit primary contract unchanged.',
        primary_effective_protocol=C.bind(C.DOC / 'PROTOCOL_EFFECTIVE.json'),
        primary_criteria_unchanged=True, models=E.model_names(), populations=list(POPS),
        purpose='Separate observed coordinate/continuous-pose changes under fixed R0 W/D from operational coordinate-plus-GEO changes.',
        held_rule='For every model/frame, use the existing candidate with name equal to frozen R0 GEO_name. Do not recompute/select by reference or mix T/R minima.',
        unavailable='If the R0 branch is undefined, absent, or its current-model candidate has unavailable pose, retain failed row; do not replace with current GEO.',
        preserved='Same corner8 final SQPnP/LM candidate, K/dimensions, physical-frame C2 T/R, original confidence/box, full denominator.',
        caveat='Held W/D does not hold continuous R/t or internal PnP root fixed. This is diagnostic; no new deployment rule is promoted.',
        direction_tolerance=TOL, direction_tolerance_role='Existing numeric tolerance, not meaningful-effect threshold.',
        summaries=['Each model/seed conditional and failure-inclusive median/P90', 'Operational-minus-held paired directions',
                   'Both held and operational versus R0', 'R0-relative direction transition counts',
                   'W/D switch counts versus R0', 'Per-recording descriptive summaries'],
        old_detection_strata=dict(binding=C.bind(old_path) if old_path.exists() else None,
            rule='Use unchanged old E2 identity categories for all99, and original R0 top ceil(0.1*99)=10 translation errors; descending T then ID tie.',
            observational_only=True, no_population_filter_for_primary=True,
            no_new_threshold_or_topcase_selection=True),
        entry_guard='Require complete PREDICTIONS_LOCK, POSE_PREDICTIONS_LOCK and RESULTS, verify their artifact hashes and historical reference hashes before scoring.',
        no_model_seed_hyperparameter_selection=True, new_fits=0, image_forwards=0,
        original_goal_unaltered=protocol['objective'], code=C.bind(Path(__file__))))
    C.save(C.DOC / 'MECHANISM_PROTOCOL_SHA.json', C.bind(path))
    print('MECHANISM_PROTOCOL_LOCKED_BEFORE_RESULTS', flush=True)


def held_pose(record, r0_name):
    if r0_name is None:
        return dict(available=False), 'R0_BRANCH_UNDEFINED'
    candidates = [h for h in record['hypotheses'] if h['name'] == r0_name]
    assert len(candidates) <= 1
    if not candidates:
        return dict(available=False), 'R0_BRANCH_ABSENT_IN_MODEL'
    pose = candidates[0]['pose']
    return pose, 'OK' if pose['available'] else 'HELD_CANDIDATE_POSE_FAILED'


def direction(before, after):
    if not before['available']:
        return 'BASELINE_UNAVAILABLE'
    if not after['available']:
        return 'MODEL_UNAVAILABLE'
    def sign(delta):
        return 'IMPROVE' if delta < -TOL else 'WORSEN' if delta > TOL else 'TIE'
    return 'T_'+sign(after['translation_cm']-before['translation_cm'])+'__R_'+sign(after['rotation_deg']-before['rotation_deg'])


def paired(before, after, ids):
    value = C.D.M.paired(before, after, ids)
    value['direction_counts_tolerance_1e_7'] = dict(Counter(direction(before[i], after[i]) for i in ids))
    return value


def summarize_group(ids, operational, held, candidates):
    r0 = operational['R0']
    output = {}
    for model in E.model_names():
        switches = sum(candidates[model][i]['GEO_name'] != candidates['R0'][i]['GEO_name'] for i in ids)
        transitions = Counter(direction(r0[i], held[model][i])+' -> '+direction(r0[i], operational[model][i]) for i in ids)
        output[model] = dict(frames=len(ids), branch_switches_from_R0=switches,
            operational=C.D.summarize(operational[model][i] for i in ids),
            held_R0_WD=C.D.summarize(held[model][i] for i in ids),
            operational_minus_held=paired(held[model], operational[model], ids),
            held_minus_R0=paired(r0, held[model], ids),
            operational_minus_R0=paired(r0, operational[model], ids),
            R0_relative_direction_transition_counts=dict(transitions))
    return output


def old_detection_strata(spec, ids):
    binding = spec['binding']
    if binding is None:
        return dict(status='UNAVAILABLE_AT_PREREGISTRATION', groups={})
    E.verify_file(binding)
    old = C.read(C.ROOT / binding['path'])['identity']
    assert set(ids) <= set(old)
    strata = {'ALL_NATURAL99': list(ids)}
    for category in sorted({old[i]['category'] for i in ids}):
        strata['BOX_CATEGORY:'+category] = [i for i in ids if old[i]['category'] == category]
    n_tail = math.ceil(.1*len(ids))
    def old_t(fid):
        row = old[fid]
        selected = next((b for b in row['boxes'] if b['index'] == row['selected_index']), None)
        return selected['GEO']['translation_cm'] if selected and selected['GEO']['available'] else float('inf')
    tail = sorted(ids, key=lambda i: (-old_t(i), i))[:n_tail]
    strata['OLD_R0_T_TOP10'] = tail
    assert len(ids) == 99 and len(tail) == 10
    for category in sorted({old[i]['category'] for i in tail}):
        strata['OLD_R0_T_TOP10:'+category] = [i for i in tail if old[i]['category'] == category]
    return dict(status='FIXED_HISTORICAL_OBSERVATIONAL_STRATA', groups=strata,
        full_categories=dict(Counter(old[i]['category'] for i in ids)),
        tail_categories=dict(Counter(old[i]['category'] for i in tail)),
        selection_source=binding, scope='Historical R0 categories/tail fixed across every new model; no new GT-conditioned runtime selection or primary exclusion.')


def number(value):
    return 'NA' if value is None else f'{value:.3f}'


def report(result):
    lines = ['# W/D 고정 기전 진단', '',
        '이 분석은 전체 예측과 주 결과를 동결한 뒤 수행한 보조 진단이다. 원래 안정성 기준과 판정을 바꾸지 않는다. '
        '각 보정 좌표에서 기존 R0의 GEO가 선택한 W/D 이름을 유지한 pose와 운영 GEO가 선택한 pose를 비교한다. '
        'GT로 후보를 고르거나 다른 후보의 최소 T·R을 합치지 않았다.', '',
        '**W/D 고정은 연속 R/t나 PnP 내부 해까지 고정한다는 뜻이 아니다.** '
        '좌표 변화의 결과를 고정된 두 W/D 분기 중 같은 분기에서 읽는 진단이다. 새로운 배포 gate나 최종 모델 선택으로 사용하지 않는다.', '',
        '표의 중앙값/P90은 유효 pose 조건부이며 실패 수를 함께 적었다. 실패 포함 +∞ 분포, 프레임별 방향, '
        'recording별 결과는 [MECHANISM_RESULTS.json](MECHANISM_RESULTS.json)에 있다.', '']
    for population in POPS:
        lines += [f'## {population}', '',
            '| 모델 | 운영 T/R 중앙값 | 고정 W/D T/R 중앙값 | 운영 T/R P90 | 고정 W/D T/R P90 | 운영/고정 실패 | W/D 전환 |',
            '|---|---|---|---|---|---|---|']
        for model, value in result['populations'][population]['models'].items():
            op, he = value['operational'], value['held_R0_WD']
            def pair(summary, statistic):
                return number(summary['conditional']['translation_cm'][statistic])+' / '+number(summary['conditional']['rotation_deg'][statistic])
            lines.append(f'| {model} | {pair(op,"median")} | {pair(he,"median")} | {pair(op,"P90")} | {pair(he,"P90")} | {op["failed_pose"]}/{he["failed_pose"]} | {value["branch_switches_from_R0"]}/{value["frames"]} |')
        lines += ['', 'T 단위 cm, R 단위 °. 모든 seed를 표시하며 최상의 seed를 고르지 않았다.', '']
    lines += ['## 고정된 검출 꼬리', '',
        '이전 진단의 R0 자연99 박스 분류와 원래 T 상위10장을 그대로 사용했다. 새 모델의 큰 오차부터 다시 사례를 선택하지 않았다. '
        '각 집단은 관찰용 분해이며 주 평가99장에서는 한 장도 제외하지 않았다.', '',
        '| 집단 | N |', '|---|---:|']
    for name, value in result['detection_strata']['summaries'].items():
        lines.append(f'| {name} | {value["frames"]} |')
    lines += ['', '각 집단의 모델별 운영/고정 T·R, 실패 수, 원래 R0 대비 개선·악화 수는 JSON의 `detection_strata.summaries`에 저장했다. '
        '박스 IoU 분류를 실제 객체 identity의 독립 확인 또는 배포용 검출 선택 규칙으로 해석하지 않는다.', '',
        '## 해석의 범위', '',
        '운영 GEO와 고정 W/D의 차이는 같은 좌표에서의 후보 선택 기여다. 고정 W/D가 좋아져도 운영 GEO가 나빠지면 선택기 호환성이 남는다. '
        '고정 W/D와 운영 모두 나빠지면 현재 자료만으로 후보 선택 하나를 원인으로 삼을 수 없다. '
        '중앙값 차이는 프레임별 변화의 중앙값과 다르며 두 효과를 더해 원인 비율로 만들지 않는다.', '',
        '주 실험의 안정성 판정은 [RESULTS.json](RESULTS.json)에 동결돼 있다. '
        '이 진단으로 성공 기준을 낮추거나 현재 GT를 사용하는 routing을 만들지 않는다. '
        '재사용 DEV·geometry reference·독립 촬영 확인 부재라는 한계도 유지된다.', '',
        '[결과 전 보조 분석 계약](MECHANISM_PROTOCOL.json) · [주 보고서](REPORT_KO.md)', '']
    C.save(C.DOC / 'MECHANISM_RESULTS_KO.md', '\n'.join(lines))


def score():
    start = time.monotonic()
    protocol_path = C.DOC / 'MECHANISM_PROTOCOL.json'
    E.verify_file(C.read(C.DOC / 'MECHANISM_PROTOCOL_SHA.json'))
    spec = C.read(protocol_path)
    for key in ('code', 'primary_effective_protocol'):
        E.verify_file(spec[key])
    assert spec['models'] == E.model_names()
    result_path = C.DOC / 'RESULTS.json'
    assert result_path.exists(), 'No mechanism outcome access before complete primary RESULTS'
    original_result_binding = C.bind(result_path)
    primary = C.read(result_path)
    assert primary['complete'] is True
    assert spec['created_at'] < primary['created_at']
    prediction_lock, rows, _, protocol = E.locked_inputs()
    assert primary['prediction_lock'] == C.bind(C.DOC / 'PREDICTIONS_LOCK.json')
    assert primary['pose_lock'] == C.bind(C.DOC / 'POSE_PREDICTIONS_LOCK.json')
    pose_lock = C.read(C.DOC / 'POSE_PREDICTIONS_LOCK.json')
    for binding in list(E.bindings_in(pose_lock)) + primary['artifacts']:
        E.verify_file(binding)
    references = E.reference_bindings(protocol, rows)
    assert references == C.read(C.DOC / 'REFERENCE_BINDINGS.json')
    if (C.DOC / 'MECHANISM_RESULTS.json').exists():
        previous = C.read(C.DOC / 'MECHANISM_RESULTS.json')
        assert previous['primary_results'] == original_result_binding
        for binding in previous['artifacts']:
            E.verify_file(binding)
        print('MECHANISM_ALREADY_SCORED', flush=True)
        return
    candidates = C.read(C.RAW / 'POSE_CANDIDATES.json')
    operational = C.read(C.RAW / 'POSE_METRICS.json')
    populations = C.read(C.RAW / 'EVAL_GROUPS.json')
    _, truth = C.D.O.D.Pose.metadata('REAL_DEV')
    held, frame_rows = {}, []
    no_switch_parity = 0
    for model in E.model_names():
        held[model] = {}
        for row in rows:
            fid = row['id']; name = candidates['R0'][fid]['GEO_name']
            pose, status = held_pose(candidates[model][fid], name)
            metric = C.D.metric(fid, pose, truth[fid])
            held[model][fid] = metric
            if candidates[model][fid]['GEO_name'] == name:
                C.D.O.D.close(metric, operational[model][fid]); no_switch_parity += 1
            frame_rows.append(dict(model=model,id=fid,recording=row['recording'],severity=row['severity'],
                R0_GEO_name=name,operational_GEO_name=candidates[model][fid]['GEO_name'],held_GEO_name=name,
                branch_switched=candidates[model][fid]['GEO_name'] != name,held_status=status,
                held_available=metric['available'],held_T_cm=metric.get('translation_cm'),held_R_deg=metric.get('rotation_deg'),
                operational_available=operational[model][fid]['available'],operational_T_cm=operational[model][fid].get('translation_cm'),
                operational_R_deg=operational[model][fid].get('rotation_deg'),
                held_vs_R0=direction(operational['R0'][fid],metric),operational_vs_R0=direction(operational['R0'][fid],operational[model][fid]),
                operational_vs_held=direction(metric,operational[model][fid])))
        E.error_tensor(held,[model],[r['id'] for r in rows])
        print('HELD_R0_WD_SCORED',model,len(held[model]),flush=True)
    output = {}
    metadata = {r['id']: r for r in rows}
    for population in POPS:
        ids = populations[population]
        output[population] = dict(frames=len(ids), models=summarize_group(ids,operational,held,candidates), by_recording={})
        for recording in sorted({metadata[i]['recording'] for i in ids}):
            selected = [i for i in ids if metadata[i]['recording'] == recording]
            output[population]['by_recording'][recording] = dict(frames=len(selected),models=summarize_group(selected,operational,held,candidates))
    strata = old_detection_strata(spec['old_detection_strata'],populations['NATURAL99'])
    strata['summaries'] = {name:dict(frames=len(ids),models=summarize_group(ids,operational,held,candidates)) for name,ids in strata['groups'].items()}
    metric_path = C.RAW / 'MECHANISM_HELD_METRICS.json'
    csv_path = C.DOC / 'MECHANISM_FRAME_RESULTS.csv'
    C.save(metric_path,held); E.write_csv(csv_path,frame_rows)
    result = dict(complete=True,created_at=C.now(),protocol=C.bind(protocol_path),primary_results=original_result_binding,
        primary_verdict_unchanged=primary['stability']['verdict'],populations=output,detection_strata=strata,
        no_switch_metric_parity_count=no_switch_parity,frame_rows=len(frame_rows),
        branch_rule='Only existing candidate matching R0 GEO_name; no reference selection; absent/failed retained.',
        primary_model_or_seed_selection=False,new_fits=0,image_forwards=0,
        artifacts=[C.bind(metric_path),C.bind(csv_path)],wall_seconds=time.monotonic()-start)
    assert C.bind(result_path) == original_result_binding
    C.save(C.DOC / 'MECHANISM_RESULTS.json',result)
    report(result)
    print('MECHANISM_COMPLETE_PRIMARY_UNCHANGED',len(frame_rows),flush=True)


def self_check():
    good = dict(available=True,token='existing')
    record = dict(hypotheses=[dict(name='A',pose=good),dict(name='B',pose=dict(available=False))])
    assert held_pose(record,'A') == (good,'OK')
    assert held_pose(record,'B')[1] == 'HELD_CANDIDATE_POSE_FAILED'
    assert held_pose(record,'C')[1] == 'R0_BRANCH_ABSENT_IN_MODEL'
    assert held_pose(record,None)[1] == 'R0_BRANCH_UNDEFINED'
    base = dict(available=True,translation_cm=10.,rotation_deg=5.)
    after = dict(available=True,translation_cm=9.,rotation_deg=6.)
    assert direction(base,after) == 'T_IMPROVE__R_WORSEN'
    assert direction(base,base) == 'T_TIE__R_TIE'
    assert direction(base,dict(available=False)) == 'MODEL_UNAVAILABLE'
    print('MECHANISM_SELF_CHECK_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','score','self-check']);args=parser.parse_args()
    C.D.torch.set_num_threads(1);C.D.cv2.setNumThreads(1)
    dict(prepare=prepare,score=score,self_check=self_check)[args.stage.replace('-','_')]()


if __name__ == '__main__':
    main()
