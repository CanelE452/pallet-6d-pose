"""Render completed numerical evidence without choosing/tuning an inference arm."""
from __future__ import annotations

import csv
from . import common as C

LABELS = {
    'BASE': 'Base', 'N3': 'N3 seed1', 'SUBPIX': 'Base → SubPix(5)',
    'N3_SUBPIX': 'N3 → SubPix(5)',
    **{f'{start}_{kind}_{limit}': f'{start} → {name} / {cap}'
       for start in ('BASE', 'N3') for kind, name in (('WIDE', 'SubPix(25)'), ('BOUNDARY', '공유 경계'))
       for limit, cap in (('NATIVE', '외부 상한 없음'), ('CAP1', 'Base 기준 1%'))},
}


def csv_file(name, fields, rows):
    with (C.DOC / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    s, runtime = C.read(C.DOC/'SUMMARY.json'), C.read(C.DOC/'RUNTIME.json')
    assert runtime['complete'] and runtime['status'] == 'DONE'
    assert C.read(C.DOC/'CHECKS.json')['status'] == C.read(C.DOC/'CHECKS_STATISTICS.json')['status'] == 'PASS'
    all_metrics = []
    for arm in C.ARMS:
        for metric, values in s['summaries'][arm]['metrics'].items():
            all_metrics.append(dict(arm=arm, metric=metric, **{k:values[k] for k in
                ('unit','n','mean','sample_variance','sample_std','median','P90','min','max','ddof')}))
    csv_file('METRICS.csv', list(all_metrics[0]), all_metrics)
    groups=[]
    for group, by_arm in s['point_groups'].items():
        for arm, data in by_arm.items():
            for denominator, key in (('observed','observed_error_px'),('all_references','full_reference_error_px')):
                values=data[key]
                groups.append(dict(group=group,arm=arm,denominator=denominator,
                    reference_corners=data['reference_corners'],unobserved_corners=data['unobserved_corners'],
                    **{k:values[k] for k in ('n','mean','sample_variance','sample_std','median','P90')},
                    PCK5=data['observed_PCK' if denominator=='observed' else 'full_PCK']['5'],
                    PCK10=data['observed_PCK' if denominator=='observed' else 'full_PCK']['10'],
                    gross20=data['observed_gross20_count' if denominator=='observed' else 'full_gross20_count']))
    csv_file('VISIBILITY_METRICS.csv',list(groups[0]),groups)
    differences=[]
    for name, comparison in s['comparisons'].items():
        for metric, values in comparison['statistics'].items():
            differences.append(dict(comparison=name,metric=metric,unit=values['unit'],
                delta_mean=values['mean_paired_difference'],CI95_low=values['CI95'][0],CI95_high=values['CI95'][1],
                observations=values['common_eligible_observations'],frames=values['common_eligible_frames'],
                eligible_sessions=values['eligible_sessions'],master_sessions=13,
                resamples=values['resamples'],bootstrap_seed=values['seed']))
    csv_file('MEAN_DIFFERENCES.csv',list(differences[0]),differences)
    damage=[]
    for group, by_arm in s['point_groups'].items():
        for arm,data in by_arm.items():
            for base,values in data['damage_from'].items():
                damage.append(dict(group=group,arm=arm,baseline=base,**{k:v for k,v in values.items() if k!='definition'}))
    csv_file('CORNER_DAMAGE.csv',list(damage[0]),damage)
    timings=[]
    for arm in C.ARMS:
        for stage,values in runtime['summaries'][arm].items():
            timings.append(dict(arm=arm,stage=stage,**{k:v for k,v in values.items()
                if k not in ('std_definition','quantile_method')}))
    csv_file('TIMING.csv',list(timings[0]),timings)

    lines=[
        '이번 제안의 실제 평가: 넓은 검색·공유 경계 보정 모두 N3보다 평균 위치·회전이 좋아졌다고 볼 수 없다.',
        '사용자가 직접 보임으로 표시한 관측 KP 1,759개만 평가해도 모든 새 경로가 N3보다 평균 코너 오차가 높았다.',
        '보이는 KP만 보정한 실험은 아니다: 모든 유효 예측점에 시도하고, 사람의 가시성 표시는 결과를 나누는 데만 썼다.',
        '기존 319장/13세션. N3 seed1의 가중치·온도·입출력은 유지했다. 새 학습·추가 seed·새 어노테이션은 0회다.',
        '새 좌표 2,552행을 정답·가림 표시 없이 먼저 고정한 뒤 각 최종 좌표로 실제 F를 다시 계산했다.',
        '기존 네 경로 1,276행의 수치·좌표·자세는 재사용했다. 출처 설명 문자열만 현재 작업에 맞게 갱신했다.',
        '',
        '단 한 개의 넓은 설정(winSize=25,25 / 40회 / epsilon=.001)과 한 개의 비학습 공유 경계 방법을 시험했다.',
        '공유 경계: 원영상 Sobel, 예측 변의 법선 ±32px 탐색, 강건 TLS/IRLS 선 적합, (0,3) 등 네 쌍을 원자적으로 보정.',
        '쌍의 공유선 또는 한쪽 교차점 근거가 부족하면 두 점 모두 시작 좌표를 유지한다. 정답으로 적용 여부를 고르지 않는다.',
        '창·상한 탐색이나 결과를 본 뒤 설정 변경은 하지 않았다. 이 DEV는 이전에 예시를 본 탐색 평가이며 새 독립 시험셋이 아니다.',
        'NATIVE는 기존 외부 1% 상한을 푼 별도 진단 경로다. 방법 내부 검색/이동 제한은 유지한다.',
        'CAP1은 원래 Base에서 원영상 대각선 1% 이내다. N3에서 추가 1%를 허용하지 않는다.',
        '개별 점의 상한 투영은 공유선 교차 구조를 깨뜨릴 수 있다. 투영 뒤 재적합을 하지 않았고 native/final 선 잔차를 저장했다.',
        '',
        '평균 결과 (직접 가시 코너 px / 전체 관측 코너 px / 위치 cm / 회전 degree / ADDsym cm, 자세 실패 수, 실제 전체 경로 ms):',
        '경로 | 보이는 코너 | 전체 관측 코너 | 위치 | 회전 | ADDsym | 실패/319 | 전체 평균 ms',
    ]
    for arm in C.ARMS:
        data=s['summaries'][arm];m=data['metrics'];t=runtime['summaries'][arm]['full_pipeline']
        visible=s['point_groups']['DIRECT_VISIBLE'][arm]['observed_error_px']['mean']
        lines.append(f"{LABELS[arm]} | {visible:.3f} | " + ' | '.join(f"{m[k]['mean']:.3f}" for k in
            ('corner_px','translation_cm','rotation_deg','ADDsym_cm')) +
            f" | {data['pose']['failures']}/319 | {t['mean_ms']:.3f}")
    lines += [
        '',
        '코너 평균은 동일한 관측 코너 2,445개, 자세 평균은 해당 경로의 성공 영상 기준이다.',
        '미관측 54코너도 숨기지 않았다: 전체 2,499개에 기존 영상 대각선 벌점을 넣은 통계/PCK/큰 오류를 SUMMARY에 별도 저장했다.',
        'Wide NATIVE 두 경로는 같은 1장에서 자세 실패: eval_night09:1779449575470221824.',
        '그 영상의 N3 위치 오차는 2,586.949cm다. 실패로 이 큰 값이 빠져 전체 성공 위치 평균이 낮아 보인다.',
        '공통 318장에서는 N3 위치 평균 31.917cm → Base Wide 37.659cm / N3 Wide 36.973cm로 오히려 악화했다.',
        '따라서 이 두 경로의 표에 나온 낮은 위치 평균을 정확도 개선이라고 해석하지 않는다.',
        '',
        'N3에 추가한 방법 − N3의 짝지은 평균 차이 [13세션 재표집 95% 구간]:',
        '경로 | 위치 cm | 회전 degree | 보이는 코너 px | 보이는 0·3 px',
    ]
    for arm in C.NEW_ARMS:
        if not arm.startswith('N3_'): continue
        statistics=s['comparisons'][f'{arm}_minus_N3']['statistics']
        def formatted(k):
            v=statistics[k];return f"{v['mean_paired_difference']:+.3f} [{v['CI95'][0]:+.3f}, {v['CI95'][1]:+.3f}]"
        lines.append(LABELS[arm]+' | '+' | '.join(formatted(k) for k in
            ('translation_cm','rotation_deg','DIRECT_VISIBLE_corner_px','DIRECT_VISIBLE_0_3_corner_px')))
    lines += [
        '같은 기존 13세션을 10,000회 짝지어 재표집(seed 20260917)했다. 음수는 개선, 양수는 악화다.',
        '관측점/성공영상의 원행을 합쳐 계산했다. 표준편차는 원행의 퍼짐이고 신뢰구간이나 학습 seed 변동이 아니다.',
        'Base/N3/SubPix/기존 결합 각각 대비한 모든 새 경로의 C/T/R/ADD/가시성 평균 차이·구간은 MEAN_DIFFERENCES.csv에 있다.',
        '중앙값만 낮아진 결과를 평균 개선으로 부르지 않았다. 예: N3 공유경계 CAP1 위치 중앙값은 낮아졌지만 평균은 높아졌다.',
        '',
        '사람이 직접 보인다고 표시한 코너 1,776개(관측 1,759)의 손상:',
        '경로 | 보이는 평균 px | 보이는 0·3 평균 px(542개) | N3 양호<5 → 오류>10 | Base 양호<5 → 오류>10 | N3 >20 → <=10',
    ]
    for arm in C.ARMS:
        v=s['point_groups']['DIRECT_VISIBLE'][arm];z=s['point_groups']['DIRECT_VISIBLE_0_3'][arm]
        lines.append(f"{LABELS[arm]} | {v['observed_error_px']['mean']:.3f} | {z['observed_error_px']['mean']:.3f} | "
            f"{v['damage_from']['N3']['good5_to_bad10']}/895 | {v['damage_from']['BASE']['good5_to_bad10']}/742 | "
            f"{v['damage_from']['N3']['bad20_to_good10']}/210")
    lines += [
        '외부 가림·자체 가림·화면 밖은 기존 키포인트별 사람 표시로도 분리 집계했다. 이 표시는 추론에 넣지 않았다.',
        '',
        '개별 0·3 사진과 정답 근처 초기점 진단:',
        'figures/PHOTO1/2/3_REFINEMENT.png는 같은 원영상/공통 배율과 저장된 최종 좌표를 사용한다. 공유선과 CAP1을 구별 표시했다.',
        '사진 2/3의 참조 좌표는 원래 좌표 출처가 unknown이다. 보이는 실제 코너와 일치하는 새 수동 GT로 간주하지 않는다.',
        'DIAGNOSTICS.json에는 참조 좌표 + 고정 5가지 미소 이동으로 기존 SubPix(5)를 실행한 40회 진단과 코너별 오차가 있다.',
        '이는 정답 도움을 받은 원인 진단이며 운영 보정·319장 경로 결과에 합치지 않았다. 새 어노테이션을 만들지 않았다.',
        '강한 일관된 바닥/다른 사각형도 선택될 수 있다. 공유선 일관성 자체가 파렛트의 의미 경계를 인증하지 않는다.',
        '이번 숫자는 넓은 창이나 공유선만으로 충분하다는 근거를 주지 못했다. 경계 의미를 구별하는 새 학습은 이번에 하지 않았다.',
        '',
        '실행량과 검증:',
    ]
    a=C.read(C.DOC/'CHECKS.json')['execution'];i=C.read(C.DOC/'COORDINATES_LOCK.json')['execution'];r=runtime['execution']
    lines += [
        f"정확도: 원영상 319장 × Base/N3 × 2방법 = {sum(i[k] for k in ('BASE_WIDE_image_calls','BASE_BOUNDARY_image_calls','N3_WIDE_image_calls','N3_BOUNDARY_image_calls'))} 보정, SubPix {i['cornerSubPix_calls']}회.",
        f"새 최종 좌표 실제 F {a['accuracy_F_completed']}회 + Base 동일성 F {a['parity_F']}회. 캐시 재사용 {a['cached_rows_reused']}행. 전체 원행 {a['rows_written']}행.",
        f"시간: 실제 전체 경로 {r['pipeline_calls_complete']}회(12경로 × 준비20+측정130), 실제 F {r['final_F_calls_complete']}회.",
        '동일 detector/N3를 실제 실행하고 GPU 동기화한 전체 벽시계 시간이다. 기존 시간 합산·좌표 캐시 재생 시간이 아니다.',
        '26장/13세션, 경로마다 측정130회. 영상 읽기/모델 초기화/정답 채점/사후 parity/저널 기록은 시간에서 제외했다.',
        f"시간 측정에서 실제 detector {runtime['model_forwards']['detector']}회(초기화 {runtime['model_forwards']['initialization_detector_calls']}회 포함), 고정 N3 {runtime['model_forwards']['N3']}회 실행했다. 중복 backbone은 0회다.",
        'N3 대비 전체 경로 평균 추가시간(ms): '+', '.join(
            f"{LABELS[arm]} {runtime['summaries'][arm]['full_pipeline']['mean_ms']-runtime['summaries']['N3']['full_pipeline']['mean_ms']:+.3f}"
            for arm in ('N3_SUBPIX','N3_WIDE_NATIVE','N3_WIDE_CAP1','N3_BOUNDARY_NATIVE','N3_BOUNDARY_CAP1'))+'.',
        '다른 벤치마크 없이 순차 실행, 경쟁 작업/온도 검사와 모든 좌표·실제 예측 자세 parity를 원행에 저장했다.',
        '최소 검사14개 PASS. 기존 4경로 통계96개 원래 수치와 일치. 좌표·이미지·모델·코드의 SHA를 기록했다.',
        'CODE_AMENDMENT는 시간 실행 전 DONE 상태를 스냅샷 앞에서 기록하도록 한 bookkeeping 한 줄 수정이다. 보정 설정/정확도 재실행은 없다.',
        '기존 사용자 파일은 보존하고 연구 브랜치 research/visible-boundary-refine-20261009에 정상 commit/push한다.',
        '6D 참조는 기존 좌표에서 재구성한 기하 기준이며 독립적으로 물리 측정한 실제 자세 GT는 아니다.',
        '',
        '파일: PROTOCOL,INPUTS,COORDINATES_LOCK,COORDINATES/PREDICTIONS 원행, SUMMARY/PAIRED_COMPARISON, RUNTIME/ROWS, DIAGNOSTICS, CHECKS*.',
        'METRICS.csv: 원행 평균·표본 분산·표준편차·중앙값·P90. VISIBILITY_METRICS.csv: 사람 표시별 결과. CORNER_DAMAGE.csv: 손상/큰 오류 회복.',
        'MEAN_DIFFERENCES.csv: 모든 짝지은 차이·95% 구간. TIMING.csv: 실제 전체 경로/보정 시간 원행 통계.',
        '재현은 PALLET_SOURCE_ROOT를 모델·원영상·기존 데이터가 있는 checkout으로 지정한 별도 출력 checkout에서 수행한다.',
        'prepare → infer → score → statistics → diagnostics → benchmark → report 순서이며 이미 완료한 결과는 덮어쓰지 않는다.',
    ]
    (C.DOC/'REPORT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_COMPLETE',len(lines),'lines')


if __name__ == '__main__':
    main()
