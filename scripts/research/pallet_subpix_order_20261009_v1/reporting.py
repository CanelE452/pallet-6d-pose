"""Existing four-route grade replay first, then the five-route saved-row report."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import time

from . import common as C
from . import statistics as S


def fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def existing_methods():
    flat = read_rows(C.PRIOR / 'PREDICTIONS.jsonl.gz')
    assert len(flat) == 1276 and {r['method'] for r in flat} == set(S.EXISTING_ARMS)
    methods = {name: [r for r in flat if r['method'] == name] for name in S.EXISTING_ARMS}
    ids = [r['id'] for r in methods['BASE']]
    assert len(ids) == len(set(ids)) == 319
    for name, rows in methods.items():
        lookup = {r['id']: r for r in rows}
        assert len(rows) == len(lookup) == 319 and set(lookup) == set(ids)
        methods[name] = [lookup[i] for i in ids]
    return methods


def grade_bindings(methods):
    audit = C.read(C.GRADE_PATH)
    assert audit['status'] == 'PASS' and all(audit['checks'].values())
    assert audit['classification_semantics_status'] == 'NOT_CONFIRMED'
    rows = [r for r in audit['rows'] if r['population'] == 'DEV319']
    grades = {r['id']: r for r in rows}
    assert len(grades) == len(rows) == 319
    protocol = C.read(C.PRIOR / 'PROTOCOL.json')
    inputs = {r['id']: r for r in protocol['input_manifest']}
    ids = [r['id'] for r in methods['BASE']]
    groups = S.member_ids(ids, grades)
    assert set(ids) == set(inputs) == set(grades)
    bindings = []
    for row in methods['BASE']:
        grade = grades[row['id']]; bound = inputs[row['id']]
        assert row['session'] == grade['session'] == bound['session']
        assert grade['image']['sha256'] == bound['image_sha256']
        for name in ('image', 'annotation'):
            path = C.ROOT / grade[name]['path']
            assert path.stat().st_size == grade[name]['bytes'] and C.sha(path) == grade[name]['sha256']
        bindings.append(dict(id=row['id'], session=row['session'], severity=grade['severity'],
                             image_sha256=grade['image']['sha256'],
                             annotation_sha256=grade['annotation']['sha256'],
                             classification_semantics='NOT_CONFIRMED'))
    output = dict(status='PASS', full_frames=319, unique_ids=319,
                  grade_counts=S.GRADE_COUNTS,
                  grade_session_counts={g: len({grades[i]['session'] for i in groups[g]}) for g in S.GRADES},
                  image_and_reference_annotation_hashes_verified=319,
                  group_union_is_master=True, group_intersections_empty=True,
                  source_bindings=dict(grade_audit_sha256=C.sha(C.GRADE_PATH),
                      prior_protocol_sha256=C.sha(C.PRIOR/'PROTOCOL.json'),
                      prior_predictions_sha256=C.sha(C.PRIOR/'PREDICTIONS.jsonl.gz')),
                  frame_bindings=bindings, labels_used_only_after_predictions=True,
                  grade_source_field='DEV319 rows severity; never prior_approved.status',
                  private_images_annotations_copied=False,
                  execution=dict(model_forwards=0, final_F_calls=0, optimizer_updates=0, new_annotations=0))
    return grades, output


def stats_csv(path, result):
    fieldnames = ('group','method','metric','unit','n','mean','sample_variance','sample_std','median','P90','min','max','ddof',
                  'group_frames','group_sessions','reference_corners','observed_corners','matched_frames','pose_available',
                  'pose_failures','pose_failure_rate','PCK10','gross20_count','gross20_rate','full_ADDsym_AUC')
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator='\n'); writer.writeheader()
        for group in S.GROUPS:
            entry = result['groups'][group]; pop = entry['population']
            for name, summary in entry['summaries'].items():
                for metric in S.METRICS:
                    writer.writerow(dict(group=group,method=name,metric=metric,**summary['metrics'][metric],
                        group_frames=pop['frames'],group_sessions=pop['sessions'],
                        reference_corners=summary['corner']['reference_corners'],
                        observed_corners=summary['corner']['observed_corners'],
                        matched_frames=summary['corner']['matched_frames'],pose_available=summary['pose']['available'],
                        pose_failures=summary['pose']['failures'],pose_failure_rate=summary['pose']['failure_rate'],
                        PCK10=summary['corner']['PCK']['10'],gross20_count=summary['corner']['gross20_count'],
                        gross20_rate=summary['corner']['gross20_rate'],full_ADDsym_AUC=summary['ADDsym_AUC']['full']))
    temporary.replace(path)


def mean_std(stat):
    return fmt(stat['mean'])+' ± '+fmt(stat['sample_std'])


def comparison_cell(entry, metric):
    stat=entry['statistics'][metric]; ci=stat['CI95']
    return fmt(stat['mean_paired_difference']) + (
        ' ['+fmt(ci[0])+', '+fmt(ci[1])+']' if ci is not None else ' [NA]')


def reduction_cell(entry, metric):
    stat=entry['marginal_change'][metric]['mean']
    return fmt(stat['absolute_reduction_before_minus_after'])+' / '+fmt(stat['relative_reduction_percent'],2)+'%'


def runtime_stats(runtime):
    result={}
    for name, value in runtime.get('summaries',{}).items():
        full=value.get('full_pipeline',value)
        if name not in S.ARMS:
            continue
        result[name]=dict(n=full.get('n'),mean=full.get('mean_ms',full.get('mean')),
            sample_std=full.get('sample_std_ms',full.get('sample_std')),
            sample_variance=full.get('sample_variance_ms2',full.get('sample_variance')),
            median=full.get('median_ms',full.get('median')),P90=full.get('p90_ms',full.get('P90')))
    return result


def first_answers(result, final):
    if not final:
        return ['기존 네 경로의 영상 등급 재집계부터 실제로 완료했다. 이 보고서는 모델/F 호출0이며 역순은 아직 실행하지 않은 선행 결과다.']
    stats=result['comparisons']['SUBPIX_N3_minus_N3_SUBPIX']['ALL']['statistics']
    dt,dr=(stats[k]['mean_paired_difference'] for k in ('translation_cm','rotation_deg'))
    if dt is None or dr is None:
        first='역순이 정방향보다 평균 위치·회전을 줄였는지 공통 유효 결과로 확인하지 못했다.'
    elif dt<0 and dr<0:
        first='역순은 정방향보다 평균 위치·회전 오차를 모두 줄였다. 반복 개발자료의 탐색적 개선 신호이며 구간·손상·시간을 함께 판단한다.'
    elif dt>0 and dr>0:
        first='역순은 정방향보다 관측 평균 위치·회전 오차를 모두 늘렸다. 위치 Δ='+fmt(dt)+' cm, 회전 Δ='+fmt(dr)+' degree다. 전체 평균 위치·회전을 함께 낮추는 추가 이득은 관측되지 않았다. 두 차이의95% 구간은0을 포함한다.'
    elif dt*dr<0:
        first='역순은 정방향 대비 위치·회전 사이의 절충이다. 평균 위치 Δ='+fmt(dt)+' cm, 회전 Δ='+fmt(dr)+' degree다.'
    else:
        first='역순은 정방향보다 평균 위치·회전 오차를 함께 줄이지 못했다.'
    rankings=result['grade_effect_ranking']['N3_SUBPIX_minus_BASE']
    second=[]
    for metric,label in (('translation_cm','위치'),('rotation_deg','회전')):
        r=rankings[metric]['mean']
        second.append(label+' 최종평균 최소='+','.join(r['smallest_final_error_grades'])+
            ', 절대 감소량 최대='+','.join(r['largest_absolute_reduction_grades'])+
            ', 상대 감소율 최대='+','.join(r['largest_relative_reduction_grades']))
    whole=[];extra=[]
    for grade in S.GRADES:
        a=result['comparisons']['N3_SUBPIX_minus_BASE'][grade]['marginal_change']
        b=result['comparisons']['N3_SUBPIX_minus_N3'][grade]['marginal_change']
        whole.append(grade+' '+fmt(a['translation_cm']['mean']['after_minus_before'])+' cm/'+fmt(a['rotation_deg']['mean']['after_minus_before'])+' degree')
        extra.append(grade+' '+fmt(b['translation_cm']['mean']['after_minus_before'])+' cm/'+fmt(b['rotation_deg']['mean']['after_minus_before'])+' degree')
    times=runtime_stats(result.get('runtime',{}))
    if len(times)==5 and dt>0 and dr>0:
        fourth='시간까지 고려하면 정방향을 정확도 우선의 탐색적 후보로 조건부 유지하고 역순 채택은 보류한다. 역순은 평균 위치·회전이 더 크면서 정방향보다 '+fmt(times['SUBPIX_N3']['mean']-times['N3_SUBPIX']['mean'])+' ms 느렸다. '
        fourth+='정방향의 N3 대비 추가 '+fmt(times['N3_SUBPIX']['mean']-times['N3']['mean'])+' ms와 양호 코너 손상을 감수하기 어려운 목적에는 기존 N3/저비용 SUBPIX를 유지할 근거가 남는다. 이는 전체 경로 선택 권고이며 새 등급 게이트가 아니다.'
    elif len(times)==5:
        fourth='시간·손상과 평균 차이를 함께 판단하는 조건부 경로 선택이다. 이번 개발자료의 관측 차이를 독립 확증이나 새 등급 게이트로 확대하지 않는다.'
    else:
        fourth='시간 측정이 미완료여서 비용을 고려한 최종 경로 선택은 보류한다.'
    return [first,'Base 대비 정방향 T/R 변화(after−before): '+'; '.join(whole)+'. 정방향−N3의 SUBPIX 추가 효과: '+'; '.join(extra)+'. 각 단독/역순의 Base 대비 효과는 별도 표에 있다.',
            '클린 최종오차와 개선폭을 구분한 정방향−Base 결과: '+'; '.join(second)+'. '
            '클린은 정방향 평균 위치 최솟값 집단이 아니며, 개선폭도 지표·기준에 따라 다르다. 각 경로의 평균·중앙값·P90 순위는 SUMMARY.grade_effect_ranking에 있다.',
            fourth]


def render(result, final):
    lines=['# 고정 SUBPIX/N3 순서와 기존 영상 등급 비교','','\n\n'.join(first_answers(result,final)),'',
        '등급은 기존 영상 주석의 없음(clean)153장·중간(moderate)92장·어려움(severe)74장이다. '
        '외부 가림의 물리적 강도라는 의미는 NOT_CONFIRMED이며 거리·시점·재질·세션 차이에 의한 교란이 가능하다. '
        '코너별 가시성이나 별도 clean29/자연가림99/플라스틱128 집합을 섞지 않았다. 단일 N3 seed1의 저장 원행이다.','',
        '| 집합 | 영상 | 세션 | 참조/관측 코너 | 매칭 |', '|---|---:|---:|---:|---:|']
    for group in S.GROUPS:
        p=result['groups'][group]['population']
        lines.append(f"| {group} | {p['frames']} | {p['sessions']} | {p['reference_corners']}/{p['observed_corners']} | {p['matched_frames']} |")
    lines += ['', '| 등급 | 경로 | 코너 평균±SD px | 위치 평균±SD cm | 회전 평균±SD degree | ADDsym 평균±SD cm | 자세 산출/전체 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for group in S.GROUPS:
        for name,s in result['groups'][group]['summaries'].items():
            lines.append('| '+group+' | '+name+' | '+' | '.join(mean_std(s['metrics'][k]) for k in S.METRICS)+
                f" | {s['pose']['available']}/{s['total_frames']} |")
    lines += ['', 'SD는 원코너/원프레임 간 비음수 오차의 표본 산포(ddof1)이며 CI·표준오차·학습 seed 변동이 아니다. '
        '큰 유한 오차를 잘라내지 않았고 미산출 자세는 실패로 따로 세며 평균에0/임의 큰 값을 넣지 않는다. '
        '모든 n·분산·median·P90·최댓값은 RESULTS.csv와 SUMMARY에 보존했다. ADDsym는 대응8코너 proper-group 대칭 오차이며 표면 ADD-S가 아니다.', '',
        '| 등급 | 경로−Base: 전체 보정 효과 | 코너 감소 px / % | 위치 감소 cm / % | 회전 감소 degree / % | ADDsym 감소 cm / % |',
        '|---|---|---:|---:|---:|---:|']
    for group in S.GRADES:
        for name in result['methods']:
            if name=='BASE':continue
            c=result['comparisons'][name+'_minus_BASE'][group]
            lines.append('| '+group+' | '+name+' | '+' | '.join(reduction_cell(c,k) for k in S.METRICS)+' |')
    lines += ['', '| 등급 | 정방향−N3: SUBPIX 추가 효과 | 코너 감소 px / % | 위치 감소 cm / % | 회전 감소 degree / % | ADDsym 감소 cm / % |',
              '|---|---|---:|---:|---:|---:|']
    for group in S.GRADES:
        c=result['comparisons']['N3_SUBPIX_minus_N3'][group]
        lines.append('| '+group+' | N3_SUBPIX−N3 | '+' | '.join(reduction_cell(c,k) for k in S.METRICS)+' |')
    lines += ['', '위 두 표의 양수는 감소량(before−after)과 상대 감소율100*(before−after)/before다. '
        '분모0이면 상대 감소율은 NA다. 중앙값·P90의 감소량/율도 SUMMARY.comparisons.marginal_change에 보존했다. '
        '아래 짝비교는 after−before여서 음수가 개선 방향이다.', '',
        '| 비교 | 등급 | 코너 평균 Δ [95% 구간] px | 위치 평균 Δ [95% 구간] cm | 회전 평균 Δ [95% 구간] degree | ADDsym 평균 Δ [95% 구간] cm | T 기여세션/빈draw |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,groups in result['comparisons'].items():
        for group,c in groups.items():
            t=c['statistics']['translation_cm']
            lines.append('| '+name+' | '+group+' | '+' | '.join(comparison_cell(c,k) for k in S.METRICS)+
                f" | {t['eligible_sessions']}/{t['empty_resamples']} |")
    lines += ['', '기존 원13세션의 공유 paired bootstrap10000회·seed20260917·draw SHA256 `'+
        result['bootstrap']['draw_sha256']+'`를 모든 경로/지표/등급에서 재사용했다. '
        '원세션을 뽑은 뒤 해당 등급의 원코너/원프레임을 남기고 관측 수로 가중한다. '
        '기여세션·빈draw·공통 유효 n·실패 제외는 지표별 JSON에 있으며 기여세션1이면 CI는 NA다. '
        '0을 포함하는 구간은 불확실성이 남으며 동등성/효과없음 증거가 아니다. 이 개발자료는 독립 확증이 아니다.', '',
        '| 등급 | 경로 | Base good<5→bad>10 / bad>20→good<10 | N3 good<5→bad>10 / bad>20→good<10 | PCK10 | gross20 수/율 | 실패 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for group in S.GROUPS:
        entry=result['groups'][group]
        for name,s in entry['summaries'].items():
            a,b=entry['RAW_canonical_damage'][name],entry['N3_canonical_damage'][name]
            lines.append(f"| {group} | {name} | {a['good5_to_bad10']}/{a['bad20_to_good10']} | {b['good5_to_bad10']}/{b['bad20_to_good10']} | {fmt(s['corner']['PCK']['10'],6)} | {s['corner']['gross20_count']}/{fmt(s['corner']['gross20_rate'],6)} | {s['pose']['failures']} |")
    lines += ['', '손상은 저장된 대칭 분기 뒤의 동일 canonical identity를 맞췄다. PCK/gross는 전체 참조 분모, '
        '코너 평균/분산은 조건부 관측 코너 풀링이다. 실제 최종 가설 전환/유지별 오차 변화와 '
        'T/R 모두 감소·반대·모두 증가 수는 모든 비교/등급의 JSON에 보존했다. '
        '이 분석으로 등급별 새 적용 게이트나 프레임별 GT oracle 혼합을 만들지 않는다.','']
    runtime=result.get('runtime',{});times=runtime_stats(runtime)
    lines.append('이번 동일환경 전체경로 시간 상태: '+runtime.get('status','NOT_STARTED')+'. '
                 '과거 시간과 이번 시간의 차이를 경로 성능 차이로 해석하지 않는다. 캐시 재생시간/시간 단순 합은 배포 지연시간이 아니다.')
    if runtime.get('complete') and len(times)==5:
        lines += ['', '| 경로 | 평균±SD ms | 분산 ms² | median/P90 ms | n |','|---|---:|---:|---:|---:|']
        for name in S.ARMS:
            t=times[name];lines.append('| '+name+' | '+mean_std(t)+' | '+fmt(t['sample_variance'])+' | '+fmt(t['median'])+'/'+fmt(t['P90'])+' | '+str(t['n'])+' |')
        lines += ['', '역순 평균 추가시간: '+', '.join(base+' 대비 '+fmt(times['SUBPIX_N3']['mean']-times[base]['mean'])+' ms'
            for base in ('N3_SUBPIX','N3','SUBPIX'))+'. 원영상 RAM→초기검출/선택→보정→F 한 번이며 디코딩/모델 로딩은 제외했다.']
        grade_times=runtime.get('grade_summaries',{})
        if grade_times:
            lines += ['', '시간의 등급별 표는 전체153/92/74장이 아닌 고정26프레임 패널 부분집합이다. '
                'clean12장×5회=n60, moderate8장×5회=n40, severe6장×5회=n30를 경로마다 동일하게 측정했다. '
                '등급별 원시간 평균·분산·SD·median·P90은 RUNTIME.grade_summaries에 보존했다.']
    elif final:
        lines += ['', '시간 미완료: '+runtime.get('reason','정상 완료된 다섯 경로 전체 측정이 없다.')]
    lines += ['', '순서는 재학습하지 않은 고정 N3에 대해 비교했다. N3 입력 qS에서 실제 후보/맥락/표본을 다시 만들며 '
        '과거 N3 이동벡터를 단순 더하는 방법이 아니다. 마지막 총상한은 원래 Base 기준1%이고 중간 자세는 계산하지 않는다. '
        '관련 기능/캐시/최종F/수치 상한 검사는 CHECKS와 원행에 남긴다.','',
        '통계 실행량: 모델 forward0·최종F0·optimizer update0·새 어노테이션0. 이번 결과는 전용 브랜치 '
        '`research/subpix-n3-order-20261009`에 게시하며 main에 합치지 않는다.','']
    if final:
        motion=result['reverse_motion'];counts=motion['diagnostic_counts']
        lines += ['역순 이동/상한: SUBPIX 입력 제한 '+str(counts['SUBPIX_input_cap_active_corners'])+'코너/'+str(counts['SUBPIX_input_cap_active_frames'])+
            '프레임, 실제 N3 내부 제한 '+str(counts['N3_internal_cap_active_corners'])+'코너/'+str(counts['N3_internal_cap_active_frames'])+
            '프레임, Base 기준 최종 제한 '+str(counts['final_cap_active_corners'])+'코너/'+str(counts['final_cap_active_frames'])+
            '프레임이다. SUBPIX/N3/최종제한 단계 이동 평균은 '+
            '/'.join(fmt(motion['stage_displacements'][k]['mean']) for k in ('SUBPIX','N3','FINAL'))+
            ' px다. SUBPIX fallback='+str(motion['fallback_counts'])+
            '; N3 head 미사용 '+str(counts.get('N3_head_skipped_frames',0))+'프레임, 출력지원 없는 코너 '+
            str(counts.get('N3_output_unsupported_corners',0))+'개다. 입력 feature support와 좌표에 따라 바뀌는 표본 범위를 분리했다.','']
        comp=result['comparisons']['SUBPIX_N3_minus_N3_SUBPIX']['ALL'];quads=comp['pose_quadrants']['counts']
        hypothesis=comp['hypothesis']['groups']
        lines += ['역순−정방향 T/R 공동 변화: 모두 감소 '+str(quads['BOTH_DECREASE'])+
            ', 모두 증가 '+str(quads['BOTH_INCREASE'])+', 위치 감소·회전 증가 '+str(quads['T_DECREASE_R_INCREASE'])+
            ', 위치 증가·회전 감소 '+str(quads['T_INCREASE_R_DECREASE'])+'프레임이다. 최종 가설 전환 '+
            str(hypothesis['SWITCH']['frames'])+', 유지 '+str(hypothesis['NO_SWITCH']['frames'])+'프레임이며 세부 평균 변화는 JSON에 있다.','']
        execution=result['accuracy_execution'];rt=runtime.get('execution',{});models=runtime.get('execution_model_forwards',{})
        accuracy_f=execution['reverse_accuracy_F_calls'];runtime_f=rt.get('final_F_calls_complete',0)
        lines += ['실제 실행량: 역순 정확도 head '+str(execution['reverse_accuracy_head_calls'])+'/F '+str(accuracy_f)+
            '회, 후보 특징 재표본화 '+str(execution['actual_candidate_sample_calls'])+'회, 정확도 backbone0회. '
            '기존 네 경로 parity는 저장된 검증을 재사용해 새 control F0회다. 시간측정 '+str(rt.get('pipeline_calls_complete',0))+
            '회(100회 준비+650회 측정), 총 새 F '+str(accuracy_f+runtime_f)+'회. 검출기 '+str(rt.get('detector_calls',0))+
            '회+내부 초기화 '+str(models.get('detector_internal_initialization_warmup',0))+'회, 시간측정 N3 head '+
            str(models.get('N3',0))+'회, 중복 backbone0회다. 새 학습/optimizer0, 정확도·시간 활성 실행 '+
            fmt(execution['elapsed_seconds']+rt.get('elapsed_seconds',0),3)+'초다.','',
            '입력 resolver 수리: legacy checkpoint 검색이 불변 baseline worktree의 누락 경로를 찾아 '
            '모델/F0에서 실패했다. 동일 SHA의 기존 source 체크포인트를 같은 생성자/state dict로 직접 읽도록 수정했다. '
            '설정 변경·성능에 따른 재실행은 없으며 PRECHECK_REPAIR.json에 기록했다.','',
            'runtime 메타데이터 해석: 실행 중 snapshot인 내부 execution.status는 STARTED지만 외부 status=DONE, '
            '완료 journal, 750회 완료 및 active_job=None으로 완료를 확인했다. 일부 inherited binding.origin 표시는 '
            '실제 source/baseline 위치와 다르지만 경로·SHA·bytes는 독립 검산에서 일치했다. sealed CHECKS/RUNTIME를 고쳐 숨기지 않았다.','']
    prefix='' if final else 'EXISTING_GRADE_'
    lines += ['['+prefix+'SUMMARY]('+prefix+'SUMMARY.json) · ['+prefix+'RESULTS.csv]('+prefix+'RESULTS.csv) · '
              '['+('PAIRED_COMPARISON' if final else 'EXISTING_GRADE_PAIRED')+']('+('PAIRED_COMPARISON' if final else 'EXISTING_GRADE_PAIRED')+'.json) · [GRADE_BINDINGS](GRADE_BINDINGS.json)','']
    return '\n'.join(lines)


def report(stage='existing',reverse_rows=None):
    started=time.monotonic();methods=existing_methods()
    grades,bindings=grade_bindings(methods)
    final=stage=='final'
    if final:
        path=Path(reverse_rows) if reverse_rows else C.DOC/'PREDICTIONS.jsonl.gz'
        rows=read_rows(path)
        rows=[r for r in rows if r['method']=='SUBPIX_N3']
        lookup={r['id']:r for r in rows}
        ids=[r['id'] for r in methods['BASE']]
        assert len(rows)==len(lookup)==319 and set(lookup)==set(ids)
        methods['SUBPIX_N3']=[lookup[i] for i in ids]
    result=S.aggregate(methods,grades)
    runtime=C.DOC/'RUNTIME.json'
    result.update(stage='ALL_FIVE_FINAL' if final else 'EXISTING_FOUR_BEFORE_REVERSE',
        grade_binding_sha256=C.digest(bindings),
        source_bindings=dict(bindings['source_bindings']),
        statistics_code_sha256=C.sha(Path(S.__file__)),reporting_code_sha256=C.sha(Path(__file__)),
        prior_statistics_code_sha256=C.sha(Path(S.S.__file__)),
        runtime=C.read(runtime) if final and runtime.exists() else dict(status='NOT_STARTED',complete=False))
    if final:
        checks=C.read(C.DOC/'CHECKS.json')
        result['accuracy_execution']=checks['execution']
        repair=C.DOC/'PRECHECK_REPAIR.json'
        result['implementation_notes']=dict(precheck_repair=C.read(repair) if repair.exists() else {},
            runtime_snapshot_status='execution.status STARTED is the prefinal snapshot; outer DONE, final journal and counts establish completion',
            inherited_binding_origin='Some producer origin labels are inherited; actual paths/hashes/bytes independently verified')
        result['source_bindings'].update(reverse_predictions_sha256=C.sha(path),
            checks_sha256=C.sha(C.DOC/'CHECKS.json'),runtime_sha256=C.sha(runtime) if runtime.exists() else None,
            precheck_repair_sha256=C.sha(repair) if repair.exists() else None)
    result['execution']['elapsed_seconds']=time.monotonic()-started
    prefix='' if final else 'EXISTING_GRADE_'
    if not (C.DOC/'GRADE_BINDINGS.json').exists():
        C.write(C.DOC/'GRADE_BINDINGS.json',bindings)
    else:
        assert C.digest(C.read(C.DOC/'GRADE_BINDINGS.json'))==C.digest(bindings)
    C.write(C.DOC/(prefix+'SUMMARY.json'),result)
    C.write(C.DOC/('PAIRED_COMPARISON.json' if final else 'EXISTING_GRADE_PAIRED.json'),
        dict(bootstrap=result['bootstrap'],comparisons=result['comparisons'],contract=result['contract']))
    stats_csv(C.DOC/(prefix+'RESULTS.csv'),result)
    path=C.DOC/('RESULT_KO.md' if final else 'EXISTING_GRADE_KO.md')
    temporary=path.with_suffix(path.suffix+'.pending');temporary.write_text(render(result,final),encoding='utf-8');temporary.replace(path)
    print('GRADE_REPORT_DONE',C.finite(dict(stage=result['stage'],methods=result['methods'],
        grade_counts=result['grade_counts'],seconds=result['execution']['elapsed_seconds'],
        calls=result['execution'],summary_sha256=C.sha(C.DOC/(prefix+'SUMMARY.json')))),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('existing','final'),required=True)
    parser.add_argument('--reverse-rows',type=Path)
    args=parser.parse_args();report(args.stage,args.reverse_rows)
