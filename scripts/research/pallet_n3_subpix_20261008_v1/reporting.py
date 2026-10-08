"""Publish row-derived statistics and a short Korean report; inference/F calls zero."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import time

from .common import DOC, ROOT, finite, read, sha, write
from . import statistics as S

VISIBILITY = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'


def fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def mean_std(stat):
    return fmt(stat['mean']) + ' ± ' + fmt(stat['sample_std'])


def delta_ci(stat):
    ci = stat['CI95']
    return fmt(stat['mean_paired_difference']) + (' [' + fmt(ci[0]) + ', ' + fmt(ci[1]) + ']'
                                                if ci is not None else ' [NA]')


def verdict(result):
    comparisons = result['paired_comparisons']
    signs = {}
    for base in ('N3', 'SUBPIX'):
        stats = comparisons['N3_SUBPIX_minus_' + base]['statistics']
        marginal = comparisons['N3_SUBPIX_minus_' + base]['difference_of_marginal_statistics']
        dt, dr = (marginal[k]['mean'] for k in ('translation_cm', 'rotation_deg'))
        if dt is None or dr is None:
            state = '무결성 또는 실행 미완료'
        elif dt < 0 and dr < 0:
            state = '개선 신호'
        elif dt > 0 and dr > 0:
            state = '추가 이득 없음'
        elif dt * dr < 0:
            state = '절충'
        else:
            state = '추가 이득 없음'
        signs[base] = dict(state=state, translation_mean_delta_cm=dt, rotation_mean_delta_deg=dr,
                          translation_paired_mean_delta_cm=stats['translation_cm']['mean_paired_difference'],
                          rotation_paired_mean_delta_deg=stats['rotation_deg']['mean_paired_difference'])
    n3 = signs['N3']
    if n3['state'] == '개선 신호':
        opening = 'N3 뒤에 cornerSubPix를 붙여 평균 위치·회전 오차가 모두 줄었다. 탐색적 개선 신호이며 구간·손상·시간을 함께 판단한다.'
    elif n3['state'] == '절충':
        if n3['translation_mean_delta_cm'] < 0:
            opening = 'N3 뒤에 cornerSubPix를 붙여 평균 위치 오차는 줄었지만 회전 오차는 늘었다. 결과는 절충이다.'
        else:
            opening = 'N3 뒤에 cornerSubPix를 붙여 평균 회전 오차는 줄었지만 위치 오차는 늘었다. 결과는 절충이다.'
    elif n3['state'] == '무결성 또는 실행 미완료':
        opening = 'N3 뒤에 cornerSubPix를 붙여 평균 위치·회전 오차가 실제로 줄었는지 완료된 평가로 확인하지 못했다.'
    else:
        opening = 'N3 뒤에 cornerSubPix를 붙여 평균 위치·회전 오차가 함께 줄지는 않았다. 평균 기준 추가 이득이 없다.'
    return dict(opening=opening, comparisons=signs,
                overall='개선 신호' if all(v['state'] == '개선 신호' for v in signs.values())
                else '무결성 또는 실행 미완료' if any(v['state'] == '무결성 또는 실행 미완료' for v in signs.values())
                else '절충' if any(v['state'] == '절충' for v in signs.values()) else '추가 이득 없음')


def runtime_statistics(runtime):
    output = {}
    for name, value in runtime.get('summaries', {}).items():
        normalized = name.replace('_seed1', '')
        if normalized not in S.ARMS:
            continue
        full = value.get('full_pipeline', value)
        def item(*keys):
            return next((full[k] for k in keys if k in full), None)
        output[normalized] = dict(n=item('n', 'count', 'samples'),
            mean=item('mean', 'mean_ms'), sample_variance=item('sample_variance', 'variance_ms2', 'sample_variance_ms2'),
            sample_std=item('sample_std', 'std_ms', 'sample_std_ms'),
            median=item('median', 'median_ms'), P90=item('P90', 'p90_ms', 'P90_ms'), unit='ms')
    return output


def render(result):
    summaries = result['summaries']; v = result['verdict']
    lines = ['# N3 뒤 cornerSubPix 고정 결합 평가', '', v['opening'], '',
        'cornerSubPix 단독 대비 판정: ' + v['comparisons']['SUBPIX']['state'] +
        ' (평균 위치 Δ=' + fmt(v['comparisons']['SUBPIX']['translation_mean_delta_cm']) +
        ' cm, 평균 회전 Δ=' + fmt(v['comparisons']['SUBPIX']['rotation_mean_delta_deg']) + ' degree).', '',
        '동일 REAL_DEV319장·13세션·N3 seed1이다. 원래 Base 코너에서 원영상 대각선 1% 총이동 제한, '
        'win=(5,5), zeroZone=(-1,-1), count40, epsilon0.001을 고정했다. 가중치·온도·입출력 계약을 유지했고 새 학습·추가 seed·튜닝은 0이다.', '',
        '객체·박스·점수·중심점·지원 마스크는 보존했다. 실제 객체 선택은 모든 경로의 fixed_metadata.selected_index로 기록한다. '
        '기존 Base/N3 원행의 최상위 selected_index는 과거 action/provenance 필드로 그대로 보존하며 실제 검출 객체 선택과 혼용하지 않는다.', '',
        '| 경로 | 코너 평균 ± SD px (n) | 위치 평균 ± SD cm (n) | 회전 평균 ± SD degree (n) | ADDsym 평균 ± SD cm (n) | 매칭 | 실제 자세/319 | 실패율 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in S.ARMS:
        s = summaries[name]
        cells = [mean_std(s['metrics'][k]) + f" ({s['metrics'][k]['n']})" for k in S.METRICS]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) +
                     f" | {s['corner']['matched_frames']} | {s['pose']['available']}/319 | {fmt(100*s['pose']['failure_rate'],2)}% |")
    lines += ['', 'SD는 원행 오차의 표본 표준편차(ddof=1)다. 코너는 조건부 관측 코너 풀링, 자세는 성공 프레임당 '
        '비음수 참조 오차다. SD는 신뢰구간·평균 표준오차·시간 흔들림·학습 seed 간 변동이 아니다. '
        '큰 유한 오류를 제거하지 않았고 실패를 0이나 임의의 큰 값으로 대체하지 않았다.', '',
        '| 경로 | 코너 분산 px² | 위치 분산 cm² | 회전 분산 degree² | ADDsym 분산 cm² |',
        '|---|---:|---:|---:|---:|']
    for name in S.ARMS:
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(summaries[name]['metrics'][k]['sample_variance'])
                                                     for k in S.METRICS) + ' |')
    lines += ['', 'ADDsym는 기존 8개 cuboid 코너의 proper-group 대칭 대응 오차다. m→cm 평균·SD 100배, 분산 10000배를 검산했다.', '',
        '| 경로 | 코너 median/P90 px | 위치 median/P90 cm | 회전 median/P90 degree | ADDsym median/P90 cm | PCK@10 | gross20 (수) | 전체 실패 포함 ADDsym AUC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in S.ARMS:
        s = summaries[name]
        cells = [fmt(s['metrics'][k]['median']) + '/' + fmt(s['metrics'][k]['P90']) for k in S.METRICS]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) +
                     f" | {fmt(s['corner']['PCK']['10'],6)} | {fmt(s['corner']['gross20_rate'],6)} ({s['corner']['gross20_count']}) | {fmt(s['ADDsym_AUC']['full'],6)} |")
    lines += ['', 'PCK/gross20는 전체 참조2499코너와 기존 결측 페널티를 유지했다. AUC는 기존 정규화 ADDsym '
        '0..0.1/1001 임계값 적분이며 실패를 inf로 포함한다. 중앙값만 좋아진 경우 평균 개선으로 해석하지 않는다.', '',
        '| 짝비교 (음수=오차 감소) | 코너 평균 Δ [95% 구간] px | 위치 평균 Δ [95% 구간] cm | 회전 평균 Δ [95% 구간] degree | ADDsym 평균 Δ [95% 구간] cm | 공통 자세/제외 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, comparison in result['paired_comparisons'].items():
        cells = [delta_ci(comparison['statistics'][k]) for k in S.METRICS]
        common = comparison['coverage']['common_success']
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + f' | {common}/{319-common} |')
    b = result['bootstrap']
    n3ci = result['paired_comparisons']['N3_SUBPIX_minus_N3']['statistics']
    subci = result['paired_comparisons']['N3_SUBPIX_minus_SUBPIX']['statistics']
    def contains_zero(stat):
        ci = stat['CI95']
        return ci is not None and ci[0] <= 0 <= ci[1]
    interval_note = ('결합−N3 위치·회전 구간은 모두0을 포함해 이 자료에서의 관측 평균 감소가 확증된 것은 아니다. '
                     if all(contains_zero(n3ci[k]) for k in ('translation_cm', 'rotation_deg')) else '')
    if contains_zero(subci['translation_cm']) and subci['rotation_deg']['CI95'][1] < 0:
        interval_note += '결합−SUBPIX 위치 구간은0을 포함하고 회전 구간은 전부 음수다. '
    lines += ['', interval_note, '', f"기존13세션 단위 paired cluster bootstrap10000회, seed20260917, 모든 지표·비교에서 동일 draw를 재사용했다 (SHA256 `{b['draw_sha256']}`). "
        '세션이 중복 추출되면 그 안의 원코너/원프레임도 같은 횟수로 포함하며 원관측 수로 가중한다. '
        '구간에0이 있으면 차이의 불확실성이 남으며 동등성 증거가 아니다. 반복 개발자료의 단일 seed 탐색적 비교다. '
        '원행의 주변 평균차이와 공통 유효 짝 평균차이를 JSON에 함께 보존했다.', '',
        '| 기준→경로 | good<5px→bad>10px | bad>20px→good<10px | 좋아진/나빠진/같은 프레임 |',
        '|---|---:|---:|---:|']
    for reference, group in (('BASE', result['RAW_canonical_damage']), ('N3', result['N3_canonical_damage'])):
        for name in S.ARMS:
            if name == reference:
                continue
            d = group[name]
            lines.append(f"| {reference}→{name} | {d['good5_to_bad10']} | {d['bad20_to_good10']} | {d['improved_frames']}/{d['harmed_frames']}/{d['unchanged_frames']} |")
    ns, cs = summaries['N3'], summaries['N3_SUBPIX']
    lines += ['', 'N3 대비 대가: 코너 평균은 '+fmt(ns['metrics']['corner_px']['mean'])+'→'+
        fmt(cs['metrics']['corner_px']['mean'])+' px로 낮지만 중앙값·P90·SD는 증가했다. PCK@10은 '+
        fmt(ns['corner']['PCK']['10'],6)+'→'+fmt(cs['corner']['PCK']['10'],6)+', gross20은 '+
        str(ns['corner']['gross20_count'])+'→'+str(cs['corner']['gross20_count'])+'코너다. '+
        'N3 양호 코너'+str(result['N3_canonical_damage']['N3_SUBPIX']['good5_to_bad10'])+
        '개와 Base 양호 코너'+str(result['RAW_canonical_damage']['N3_SUBPIX']['good5_to_bad10'])+
        '개가 bad>10px로 손상됐다. ADDsym P90도 '+
        fmt(ns['metrics']['ADDsym_cm']['P90'])+'→'+fmt(cs['metrics']['ADDsym_cm']['P90'])+
        ' cm로 늘었다. 네 경로 모두 실제 자세319/319이며 실패율0%다.']
    lines += ['', '| 결합 비교 | 위치·회전 모두 감소 | 위치 감소·회전 증가 | 위치 증가·회전 감소 | 모두 증가 | 하나 이상 정확 불변 | 실패 제외 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for base in ('N3', 'SUBPIX'):
        counts = result['paired_comparisons']['N3_SUBPIX_minus_'+base]['pose_quadrants']['counts']
        lines.append('| 결합−'+base+' | '+' | '.join(str(counts[k]) for k in (
            'BOTH_DECREASE','T_DECREASE_R_INCREASE','T_INCREASE_R_DECREASE',
            'BOTH_INCREASE','ONE_OR_BOTH_EXACTLY_UNCHANGED','UNAVAILABLE'))+' |')
    m = result['motion']
    lines += ['', '결합 SUBPIX 추가이동(qS−qN): 평균 '+fmt(m['SUBPIX_additional_move_from_N3_px']['mean'])+
        ' ± '+fmt(m['SUBPIX_additional_move_from_N3_px']['sample_std'])+' px. 총상한 작동 '+
        str(m['cap_active_corners'])+'코너/'+str(m['cap_active_frames'])+'프레임. '+
        'Base와 N3 시작점 때문에 SUBPIX 원형 좌표가 달라진 코너 '+str(m['native_SUBPIX_result_different_from_BASE_start_corners'])+
        '개, 최종 결합이 N3와 완전 동일한 프레임 '+str(m['final_exactly_N3_frames'])+
        '개, SUBPIX와 완전 동일한 프레임 '+str(m['final_exactly_SUBPIX_frames'])+'개다.', '',
        '결합 실제 OpenCV 코너 호출 '+str(m['OpenCV_corner_calls'])+'회. 실패 이유 '+str(m['fallback_counts'])+
        '. SUBPIX 실패 반환 qS는 N3 좌표를 유지했다. 최종 엄격 상한에서의 수치 예외는 아래에 명시한다. '
        '세부 상태·이동과 가시성별 평균/SD/손상은 SUMMARY.json에 보존했다.', '',
        '| 결합 가설 비교 | 가설 전환/유지/실패 | 전환 위치/회전 평균 Δ | 유지 위치/회전 평균 Δ |',
        '|---|---:|---:|---:|']
    for base in ('N3', 'BASE'):
        g = result['hypothesis']['N3_SUBPIX_minus_'+base]['groups']
        cells = [str(g[k]['frames']) for k in ('SWITCH','NO_SWITCH','UNAVAILABLE')]
        diffs = ['/'.join(fmt(g[state]['mean_error_change'][k]['mean']) for k in ('translation_cm','rotation_deg'))
                 for state in ('SWITCH','NO_SWITCH')]
        lines.append('| 결합−'+base+' | '+'/'.join(cells)+' | '+' | '.join(diffs)+' |')
    lines += ['', '가설·가시성 분해는 저장 결과의 사후 진단이며 GT로 적용을 고르는 게이트가 아니다. 기존 참조는 '
        '2D/등록치수 재구성 참조이며 독립 물리 실측 확증으로 확대하지 않는다.', '']
    rt = result['runtime']; times = result['runtime_statistics']
    lines.append('전체시간 상태: '+rt.get('status', 'RUNTIME_PENDING')+'. RAM 원영상→검출/객체선택→보정→F 한 번을 '
                 '실제로 측정하며 파일 디코딩·모델 로딩을 제외한다. 캐시 정확도 재생시간이나 기존 시간 합으로 대체하지 않는다.')
    if rt.get('complete') and len(times) == 4:
        lines += ['', '| 경로 | 평균 ± SD ms | 분산 ms² | median/P90 ms | n |',
                  '|---|---:|---:|---:|---:|']
        for name in S.ARMS:
            t = times[name]
            lines.append('| '+name+' | '+mean_std(t)+' | '+fmt(t['sample_variance'])+' | '+fmt(t['median'])+'/'+fmt(t['P90'])+' | '+str(t['n'])+' |')
        combo = times['N3_SUBPIX']
        lines += ['', '결합 전체경로 평균 추가시간: '+', '.join(
            base+' 대비 '+fmt(combo['mean']-times[base]['mean'])+' ms' for base in ('N3','SUBPIX','BASE'))+'. '
            '20회 준비+고정26프레임×5반복, 네 경로의 사전 고정 교차 순서다. 원시간·환경·간섭검사는 RUNTIME.json에 있다.']
    else:
        lines += ['', '정상 완료된 네 경로 전체 시간표가 없어 추가시간을 확정하지 않는다. '+rt.get('reason','')]
    notes = result['implementation_notes']; rounding = notes['historical_N3_cap_rounding']
    fallback = notes['fallback']; adjustments = fallback.get('strict_cap_rounding_adjustments', [])
    if rounding:
        maximum_adjustment = max((r['final_minus_N3_px'] for r in adjustments), default=0.)
        lines += ['', '수치 계약 예외: 과거 float32 N3 캐시는 '+str(rounding['exceeding_corners'])+
            '코너에서 1% 상한을 최대 '+fmt(rounding['maximum_excess_px'],10)+' px 넘었다. '
            '캐시 자체를 수정하지 않고 원래 Base 기준 엄격 총상한을 결합에 적용했다. 이 때문에 SUBPIX가 실패한 '+
            str(len(adjustments))+'코너의 최종 qFinal은 qN에서 최대 '+fmt(maximum_adjustment,10)+
            ' px 미세 조정됐다. 실패 반환 qS=qN은 보존되나 최종 실패 fallback의 비트 단위 동일성에는 이 예외가 있다. '
            'Base 좌표로 되돌린 것은 아니다. 코너별 예외는 CHECKS.fallback에 보존했다.']
    repair = notes['precheck_repair']
    if repair:
        lines += ['', '입력 무결성 검사에서 NumPy 메타데이터의 직접 비교가 예외를 냈다. F0·모델0 상태에서 '
            '동일 값을 정규화한 stable digest 비교로 수리하고 재검사했다. 결과 성능에 따른 재실행이나 허용오차 완화는 없으며 '
            'PRECHECK_REPAIR.json에 원인·수리와 실행량을 보존했다.']
    accuracy = result['accuracy_execution']; execution = rt.get('execution', {})
    models = rt.get('execution_model_forwards', {})
    accuracy_f = accuracy.get('combined_accuracy_F', 0)
    parity_f = accuracy.get('parity_F', 0)
    runtime_f = execution.get('final_F_calls_complete', 0)
    solve = accuracy.get('accuracy_PnP_counts', {}).get('solvePnP', 0) + accuracy.get('parity_PnP_counts', {}).get('solvePnP', 0) + execution.get('solvePnP', 0)
    refine = accuracy.get('accuracy_PnP_counts', {}).get('solvePnPRefineLM', 0) + accuracy.get('parity_PnP_counts', {}).get('solvePnPRefineLM', 0) + execution.get('solvePnPRefineLM', 0)
    lines += ['', f'실제 실행량: 결합 정확도 F {accuracy_f}회 + 세 경로 parity F {parity_f}회 + 시간측정 F {runtime_f}회 = '+
        str(accuracy_f+parity_f+runtime_f)+'회. 시간측정 전체경로 '+str(execution.get('pipeline_calls_complete',0))+
        '회(측정행'+str(execution.get('full_measured_rows',0))+'개; 예정은 네 경로 각20회 준비+130회 측정). '
        '정확도는 검출기/N3 재추론0, 기존 세 경로 원행'+str(accuracy.get('cached_F_rows_reused',0))+'개 재사용이다. '
        '검출기 실제 경로'+str(execution.get('detector_calls',0))+'회와 내부 초기화 준비'+
        str(models.get('detector_internal_initialization_warmup',0))+'회, N3 head'+str(models.get('N3',0))+
        '회, 중복 N3 백본'+str(models.get('duplicate_N3_backbone_forwards',0))+'회다. '
        f'내부 solvePnP {solve}회+정밀화 {refine}회. OpenCV cornerSubPix는 정확도 '+
        str(accuracy.get('combined_OpenCV_corner_calls',0))+'회+parity '+str(accuracy.get('parity_OpenCV_corner_calls',0))+
        '회+시간측정 '+str(execution.get('cornerSubPix',0))+'회다. 새 학습/optimizer update0, '
        '통계/보고 자체의 추론0·F0이다. 실행량 상세는 CHECKS/RUNTIME에 보존했다.', '',
        '게시 대상 브랜치 `research/n3-subpix-20261008`. local/remote SHA 일치는 게시 후 최종 응답과 게시 검증에 기록한다. '
        '입력 RGB·비공개 주석·가중치를 새 공개물에 넣지 않았고 기존 결과·논문·LaTeX·PDF·참고문헌을 수정하지 않았다.', '',
        '[SUMMARY](SUMMARY.json) · [RESULTS.csv](RESULTS.csv) · [PAIRED_COMPARISON](PAIRED_COMPARISON.json) · '
        '[PREDICTIONS](PREDICTIONS.jsonl.gz) · [PROTOCOL](PROTOCOL.json) · [RUNTIME](RUNTIME.json) · [CHECKS](CHECKS.json)', '']
    return '\n'.join(line.rstrip() for line in lines)


def report():
    started = time.monotonic()
    predictions = DOC / 'PREDICTIONS.jsonl.gz'
    with gzip.open(predictions, 'rt', encoding='utf-8') as stream:
        flat = [json.loads(line) for line in stream if line.strip()]
    assert len(flat) == 1276 and set(r['method'] for r in flat) == set(S.ARMS)
    by_method = {name: [r for r in flat if r['method'] == name] for name in S.ARMS}
    ids = [r['id'] for r in by_method['BASE']]
    methods = {}
    for name, rows in by_method.items():
        lookup = {r['id']: r for r in rows}
        assert len(rows) == len(lookup) == 319 and set(lookup) == set(ids)
        methods[name] = [lookup[fid] for fid in ids]
    labels = None
    if VISIBILITY.exists():
        labels = {(r['population'], r['frame_id'], r['corner_id']): r['category']
                  for r in read(VISIBILITY)['rows']}
    result = S.aggregate(methods, labels)
    rt_path = DOC / 'RUNTIME.json'
    runtime = read(rt_path) if rt_path.exists() else dict(status='RUNTIME_PENDING', complete=False,
                                                        reason='No new complete matched runtime artifact')
    receipt_path = DOC / 'PREDICTIONS.json'
    receipt = read(receipt_path) if receipt_path.exists() else read(DOC / 'CHECKS.json')
    checks = read(DOC / 'CHECKS.json'); protocol = read(DOC / 'PROTOCOL.json')
    repair_path = DOC / 'PRECHECK_REPAIR.json'
    result.update(schema='pallet_n3_subpix_20261008_v1', runtime=runtime,
                  runtime_statistics=runtime_statistics(runtime),
                  accuracy_execution=receipt.get('execution', dict(status='MISSING_EXECUTION_RECEIPT')),
                  implementation_notes=dict(historical_N3_cap_rounding=protocol.get('historical_N3_cap_rounding', {}),
                                            fallback=checks.get('fallback', {}),
                                            precheck_repair=read(repair_path) if repair_path.exists() else {}),
                  prediction_rows=1276, prediction_sha256=sha(predictions),
                  source_bindings=dict(predictions=sha(predictions),
                                       visibility=sha(VISIBILITY) if VISIBILITY.exists() else None,
                                       statistics_code=sha(Path(S.__file__)), reporting_code=sha(Path(__file__))))
    result['verdict'] = verdict(result)
    result['execution']['seconds'] = time.monotonic() - started
    write(DOC / 'SUMMARY.json', result)
    write(DOC / 'PAIRED_COMPARISON.json', dict(schema=result['schema'],
        bootstrap=result['bootstrap'], comparisons=result['paired_comparisons'],
        interpretation=result['interpretation']))
    fieldnames = ('method','metric','unit','n','mean','sample_variance','sample_std','median','P90','min','max','ddof',
                  'matched_frames','pose_available','total_frames','pose_failures','pose_failure_rate',
                  'PCK10','gross20_rate','gross20_count','full_ADDsym_AUC')
    temporary = DOC / 'RESULTS.csv.pending'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for name in S.ARMS:
            s = result['summaries'][name]
            for metric in S.METRICS:
                writer.writerow(dict(method=name, metric=metric, **s['metrics'][metric],
                    matched_frames=s['corner']['matched_frames'], pose_available=s['pose']['available'],
                    total_frames=319, pose_failures=s['pose']['failures'], pose_failure_rate=s['pose']['failure_rate'],
                    PCK10=s['corner']['PCK']['10'], gross20_rate=s['corner']['gross20_rate'],
                    gross20_count=s['corner']['gross20_count'], full_ADDsym_AUC=s['ADDsym_AUC']['full']))
    temporary.replace(DOC / 'RESULTS.csv')
    text_temp = DOC / 'RESULT_KO.md.pending'
    text_temp.write_text(render(result), encoding='utf-8')
    text_temp.replace(DOC / 'RESULT_KO.md')
    print('REPORT_DONE', finite(dict(verdict=result['verdict'], rows=len(flat),
                                    seconds=result['execution']['seconds'], runtime=runtime.get('status'))), flush=True)
    return result


if __name__ == '__main__':
    report()
