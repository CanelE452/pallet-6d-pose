"""Korean closure tables and evidence figures from sealed measurements."""
from pathlib import Path
from . import common as C


def number(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def table(groups, names):
    lines = ['| 집합 | 방법 | T 중앙값/P90 cm | C2 R 중앙값/P90 ° | valid/전체 | ADDsym AUC(보조) |',
             '|---|---|---:|---:|---:|---:|']
    for group in names:
        for arm, row in groups[group].items():
            f = row['full_population']; t, r = f['translation_cm'], f['rotation_deg']
            lines.append(f'| {group} | {arm} | {number(t["median"])}/{number(t["P90"])} | '
                         f'{number(r["median"])}/{number(r["P90"])} | {row["valid_pose"]}/{row["frames"]} | {number(row["ADDsym_AUC"])} |')
    return lines


def plot(groups, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prefixes = ['R0', 'OLD_RAW', 'OLD_REF', 'RAW_CLEAR_S42', 'REF_CLEAR_S42',
                'RAW_OCC_S42', 'REF_OCC_S42', 'RAW_CLEAR_S43', 'REF_CLEAR_S43', 'RAW_OCC_S43', 'REF_OCC_S43']
    natural = groups['NATURAL99']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for column, selector in enumerate(('D9', 'GEO', 'NEWGEO')):
        for line, key in enumerate(('translation_cm', 'rotation_deg')):
            ax = axes[line, column]
            values = [natural[p + '_' + selector]['full_population'][key]['median'] for p in prefixes]
            bars = ax.barh(prefixes, values, color=['#89949c' if p.startswith('R0') else '#d88d44' if p.startswith('OLD')
                                               else '#448dab' if 'RAW' in p else '#5d956b' for p in prefixes])
            ax.invert_yaxis(); ax.set_xlim(0, max(values) * 1.22)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            ax.set_title(selector + ' — ' + ('T median (cm)' if line == 0 else 'C2 R median (deg)'))
            ax.grid(axis='x', alpha=.2)
    fig.suptitle('Natural occlusion99 / same frozen data and pose contract / lower is better\nAll selectors and both actual training streams retained; reused DEV, not independent TEST')
    fig.tight_layout(rect=(0, 0, 1, .94)); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150); plt.close(fig)


def main():
    result = C.read(C.DOC / 'FINAL_RESULTS.json')
    decision = C.read(C.DOC / 'FINAL_DECISION.json')
    robust = C.read(C.DOC / 'FINAL_ROBUSTNESS.json')
    for binding in result['inputs'] + result['private_artifacts'] + robust['inputs']:
        C.verify(binding)
    groups = result['groups']
    C.save(C.DOC / 'TABLE_ALL_CONDITIONS.md', '\n'.join(['# 전체 조건과 촬영 기록 결과', '',
        'T=팔레트 중심 cm, R=C2 full rotation °. 실패를 삭제하지 않은 전체 분모. NEWGEO는 추가 학습한 단일 합성 선택기.', ''] + table(groups, list(groups))) + '\n', True)
    figure = C.DOC / 'figures/final_same_selector_comparison.png'; plot(groups, figure)
    lines = ['# 자연 가림 위치·회전: 최소 구성 후속 실험 최종 보고', '', decision['headline_ko'], '',
        '기준 main `6ba5604a`의 완료 결과를 보존한 후속 배치다. 누락 대조와 이번 실험은 기존 DEV 결과를 본 뒤 추가했다. 소급 사전등록 또는 독립 평가라고 부르지 않는다.', '',
        '## 남기는 구성과 제외한 변경', '', *decision['component_conclusions_ko'], '',
        '![같은 선택기별 전체 반복 비교](figures/final_same_selector_comparison.png)', '',
        '## 자연 가림 99장의 전체 비교', '',
        '자연99는 Moderate21+Severe78의 실제 프레임을 합쳐 계산했다. 두 난도 중앙값을 평균하지 않았다. OLD는 기존217장 학생이며 clean78과 membership·노출·post-affine support가 달라 clean 여부만의 효과가 아니다. 42/43은 동일 R0에서 실제 다른 데이터·증강 흐름이며, 독립 초기화가 아니다.', '',
        *table(groups, ['NATURAL99']), '',
        'Clean29/Moderate21/Severe78/Full128 및 모든 촬영 기록의 T/R median·P90·valid/전체·AUC는 [전체 표](TABLE_ALL_CONDITIONS.md)에 생략 없이 보존했다. 기존16개, 재보정 선택기8개, CLEAR43 6개, 과거RAW217 3개로 총33개 고정 구성이다.', '',
        '## 구성별 matched delta: 음수는 오차 감소', '',
        '| 비교(after−before) | Δ T 중앙값 cm (mm) | Δ R 중앙값 ° | paired ΔT/ΔR 중앙값 | 둘다 개선/둘다 악화 |',
        '|---|---:|---:|---:|---:|']
    for before, after in decision['key_pairs']:
        pair = result['paired']['NATURAL99'][after + '-minus-' + before]
        delta = pair['difference_of_conditional_medians']; pm = pair['median_of_common_frame_differences']; counts = pair['paired_direction_counts']
        lines.append(f'| {after} − {before} | {number(delta["translation_cm"])} ({number(10*delta["translation_cm"])}) | '
            f'{number(delta["rotation_deg"])} | {number(pm["translation_cm"])}/{number(pm["rotation_deg"])} | '
            f'{counts["T_IMPROVE__R_IMPROVE"]}/{counts["T_WORSEN__R_WORSEN"]} |')
    lines += ['', '중앙값의 차이와 같은 프레임 차이의 중앙값은 다른 통계다. 위 표는 둘을 따로 썼다. 성능에 유리한 seed·선택기를 섞어 하나의 반복 결과로 만들지 않았다.', '',
        '## 기록 의존성과 불확실성', '',
        '자연99의6개 recording(33/2/16/27/12/9장)을 cluster 단위로 복원추출한2000회 설명용95%구간이다. 99개 독립 표본 검정이 아니며 작은 불균형 DEV를 독립 검증으로 승격하지 않는다. 전수 leave-one-recording-out도 JSON에 보존했다.', '',
        '| 비교 | T Δmedian 95% 구간 cm | R Δmedian 95% 구간 ° |', '|---|---:|---:|']
    for name, row in robust['comparisons'].items():
        ci = row['cluster_bootstrap']['intervals']
        interval = lambda key: ' / '.join(number(x) for x in ci[key]['percentile95'])
        lines.append(f'| {name} | {interval("translation_cm")} | {interval("rotation_deg")} |')
    lines += ['', *decision['tail_and_recording_ko'], '',
        '## 실행 경과와 다음 가설 판단', '', *decision['execution_and_stop_ko'], '',
        '선행 원문·공식 구현과 저장소의 과거 실험은 [선행 대조](PRIOR_AND_STANDARD_AUDIT_KO.md), 남은 가설과 추가학습 판단은 [다음 가설 감사](NEXT_HYPOTHESIS_AUDIT.md)에 구분했다. box/pose 분야의 방법을 이 팔레트에서 성공했다고 가정하지 않았다.', '',
        '## 실제 개선·악화·큰 오류 사례', '',
        '각 비교의 공동 개선/악화는 T 변화량 순, 최종 T/R 큰 오류는 해당 오차 순으로 각각 상위2개를 고른다(동률ID). 공개 이미지는 기존 공개승인ID만 재사용하며 그 승인집합 내 순위다. 전수99장의 최악 사례는 private gallery와 JSON에 별도 보존하고, 공개 사례를 전체 최악이라고 부르지 않는다. 청록은 native2D, 주황 점선은 저장된 최종PnP투영, 초록은 legacy reference다.', '',
        '[사례 전체와 기준](CASES_FINAL.md)', '', *decision.get('case_embeds_ko', []), '',
        '## 감독·평가·해석의 한계', '',
        '- 추가RGB·수동좌표0개. clean78은 assistant RGB-only 검토와 마커판 제외 기준이며 사람의 전코너 검수가 아니다. 제외137장을 전부 자연 심한가림이라고 부르지 않는다.',
        '- 현재교사9장/38코너, 기존GEO의 간접 감독 포함 추적 가능 합집합19장/86코너. NEWGEO 자체는 현재 RAW/REF donor를 사용하므로 R0+NEWGEO도 연구 전체에서 self-training을 제거한 방법이 아니다.',
        '- 평가128장/자연99장은 반복DEV. 6D참조는 annotation 기반 기하 재구성이며 독립 물리 측정이 아니다. 예측 잠금과 두 학습 흐름은 독립 확인을 만들지 않는다.',
        '- 보정 의사 좌표 추종과 물리적 정확도는 다르다. 좋은 train fit, 안정적confidence, 합성 선택기 정확도만으로 자연가림 정확성을 보장하지 않는다.',
        '- 세션별 상쇄, tail손상, 미검증 가설은 남겼다. 모든머신러닝방법이 불가능하다는 결론이 아니다.', '',
        '## 비용·재현·검증', '',
        f'누적 학생 {decision["resource_totals"]["student_fits"]}회 / 학생 update {decision["resource_totals"]["optimizer_updates"]}회 / 신규 선택기 {decision["resource_totals"]["selector_fits"]}회 / GPU 학습 {decision["resource_totals"]["GPU_training_seconds"]:.3f}초. 선택기416 update는 학생update와 분리한다. 추론 시간은 별도 lock에 기록했다.', '',
        '[최종 구성·해시](FINAL_CONFIGURATION.json) · [재현 명령](REPRODUCE.md) · [최종 판정](FINAL_DECISION.json) · [검증](FINAL_AUDIT_KO.md)', '',
        '이번 배치는 종료한다. 백그라운드 자동재개는 설정하지 않았다. 원본RGB·비공개 좌표·체크포인트는 Git에 추가하지 않는다.', '']
    C.save(C.DOC / 'REPORT_KO.md', '\n'.join(lines), True)
    C.save(C.DOC / 'REPORT_BUILD.json', dict(inputs=[C.bind(C.DOC / n) for n in ('FINAL_RESULTS.json', 'FINAL_DECISION.json', 'FINAL_ROBUSTNESS.json')],
        outputs=[C.bind(C.DOC / n) for n in ('REPORT_KO.md', 'TABLE_ALL_CONDITIONS.md', 'figures/final_same_selector_comparison.png')],
        source=C.bind(Path(__file__))), True)
    print('FINAL_REPORT_BUILT')


if __name__ == '__main__':
    main()
