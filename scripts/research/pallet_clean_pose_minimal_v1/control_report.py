"""Publish all stronger controls; no fit, prediction, or checkpoint choice."""
from pathlib import Path
from . import common as C


def plot(groups, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=[m for m in groups['NATURAL99'] if m.endswith('_GEO')]
    labels=[m.replace('_GEO','').replace('_S42',' 42').replace('_S43',' 43').replace('_',' ') for m in models]
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,(key,stat,title) in zip(axes.flat,(
        ('translation_cm','median','T median (cm)'),('rotation_deg','median','C2 full R median (deg)'),
        ('translation_cm','P90','T P90 (cm)'),('rotation_deg','P90','C2 full R P90 (deg)'))):
        vals=[groups['NATURAL99'][m]['full_population'][key][stat] for m in models]
        bars=ax.barh(labels,vals,color=['#5b6570','#b76930','#619bb7','#336890','#82ae87','#497c51','#aaa65c','#79742c'])
        ax.invert_yaxis();ax.set_title(title+' — lower is better');ax.grid(axis='x',alpha=.2)
        ax.set_xlim(0,max(vals)*1.18)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    fig.suptitle('Same frozen GEO / natural occlusion99 / posthoc reused DEV\nBoth medians and tails retained; no inferred independent confirmation')
    fig.tight_layout(rect=(0,0,1,.94));path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    target=C.DOC/'STRONGER_CONTROLS_KO.md'
    buildpath=C.DOC/'CONTROL_REPORT_BUILD.json'
    if buildpath.exists():
        for b in C.read(buildpath)['inputs']+C.read(buildpath)['outputs']:C.verify(b)
        print('CONTROL_REPORT_REUSED');return
    result=C.read(C.DOC/'CONTROL_RESULTS.json')
    robust=C.read(C.DOC/'CONTROL_ROBUSTNESS.json')
    for b in result['sources']+result['private_artifacts']+robust['inputs']:C.verify(b)
    figure=C.DOC/'figures/same_geo_stronger_controls.png'
    plot(result['groups'],figure)
    lines=['# 같은 GEO의 더 강한 단순 대조: 무학습 후속 분석','',
        '기존 clean78 결과를 본 뒤 빠진 대조를 추가했다. 새 student/selector fit은0이다. 이 문서는 다음 실험 선택 전 근거이며 배치의 최종 결론은 아니다.','',
        '**핵심:** 기존217장 REF+GEO는 11.4101cm/4.9553°로 clean78 REF CLEAR+GEO보다 위치·회전 중앙값 모두 낮다. clean78 REF OCC42/43은 기존217 REF보다 위치는 약0.405/1.458mm 낮지만 회전은 약0.308/0.119° 높다. 따라서 기존 보고서의 REF OCC+GEO vs R0+D9 이득을 가장 강한 동일선택기 대조에 대한 공동 이득으로 승격하지 않는다.','',
        '![동일 GEO의 T/R median 및 P90](figures/same_geo_stronger_controls.png)','',
        '## 모든 고정 모델과 선택기','',
        'T는 팔레트 중심의 위치 오차cm, R는 기존 C2 대칭의 전체 회전 오차°다. 자연99=중간21+심함78 원프레임을 직접 합쳤다. 실패를 제외하지 않은 전체 분모 quantile과valid 수를 보고한다.','',
        '| 집합 | 방법 | T median/P90 cm | R median/P90 ° | valid/전체 |',
        '|---|---|---:|---:|---:|']
    for group in ('NATURAL99','CLEAN29','MOD21','SEV78','FULL128'):
        for arm,row in result['groups'][group].items():
            f=row['full_population']
            lines.append(f'| {group} | {arm} | {f["translation_cm"]["median"]:.4f}/{f["translation_cm"]["P90"]:.4f} | {f["rotation_deg"]["median"]:.4f}/{f["rotation_deg"]["P90"]:.4f} | {row["valid_pose"]}/{row["frames"]} |')
    lines+=['','## 필요성 질문별 결과','',
        '- 자기학습: OLD_REF217+GEO는 R0+GEO보다 median T −0.9931cm(−9.931mm), R −0.2623°다. 선택기만 바꾸는 R0+GEO와 구분한다. 이것만으로 현재 clean78 학습이 필요하다고 할 수는 없다.',
        '- 보정: CLEAR42에서 REF−RAW는 T −0.2303cm/R +0.0063°로 trade-off. OCC42/43에서는 두 median이 낮지만, 가장 강한 OLD_REF217 대조를 공동으로 넘지 못한다.',
        '- 입력 가림: RAW42는 CLEAR가 OCC보다 두 median 모두 낮다. REF42의 OCC−CLEAR는 T −0.1302cm/R +0.0609°로 trade-off다. CLEAR43은 아직 없으며 효과 전체의 반복을 주장하지 않는다.',
        '- 선택기: 같은 frozen keypoint/두 후보에서 선택만 달라진다. GEO가 모든 모델·지표에서 D9보다 좋지는 않다. R0의 위치 median은 D9보다 악화하고 회전은 개선한다.',
        '- clean78: OLD_REF217과 이미지 구성·고유 개수·반복 노출 및 post-affine 공통support 계약도 다르므로 clean 여부만의 인과 효과라고 쓰지 않는다.','',
        '## 촬영 기록 의존성과 paired 변화','',
        '6개 자연가림 recording을 cluster 단위로 복원추출한 2000회 설명용95%구간이다. 각 recording의 표본은33/2/16/27/12/9장으로 작고 불균형하다. 독립TEST 또는 유의성 합격선이 아니다.','',
        '| 비교(after−before) | Δmedian T cm/R ° | paired차이 median T/R | T 구간 cm | R 구간 ° | 둘다 개선/악화 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,item in robust['comparisons'].items():
        pair=item['NATURAL99'];dm=pair['difference_of_conditional_medians'];pm=pair['median_of_common_frame_differences'];ci=item['cluster_bootstrap']['intervals'];count=pair['paired_direction_counts']
        ints=lambda key:' / '.join(f'{v:.4f}' for v in ci[key]['percentile95'])
        lines.append(f'| {name} | {dm["translation_cm"]:.4f}/{dm["rotation_deg"]:.4f} | {pm["translation_cm"]:.4f}/{pm["rotation_deg"]:.4f} | {ints("translation_cm")} | {ints("rotation_deg")} | {count["T_IMPROVE__R_IMPROVE"]}/{count["T_WORSEN__R_WORSEN"]} |')
    lines+=['','전체 recording 및 leave-one-recording-out은 [CONTROL_RESULTS.json](CONTROL_RESULTS.json)과 [CONTROL_ROBUSTNESS.json](CONTROL_ROBUSTNESS.json)에 보존했다.','',
        '## 증거 범위와 다음 검증','',
        '새GT/새RGB/새manual은0. oldGEO의 간접감독 때문에 R0+GEO도 zero-human-supervision이 아니다(과거Plastic10/48). 현재 보정학생+oldGEO 전체 합집합은19장/86코너이며, 현재교사9장/38점과 다르다. 원래 FINAL/STOP과모든과거수치를 보존했다.','',
        '이 대조만으로 안정적 실용 개선 구성을 확보했다고 선언하지 않는다. 다음 변경은 기존GEO의 합성 feature 생성모델과 현재학생의 차이에 직접 연결할 수 있는지 감사한 뒤, 같은architecture/훈련규칙의단일source-onlyselector 재보정 여부를 사전에 결정한다. 새student를 즉시학습하거나DEV로threshold를조정하지않는다.','']
    C.save(target,'\n'.join(lines),True)
    C.save(buildpath,dict(passed=True,inputs=[C.bind(p) for p in (C.DOC/'CONTROL_RESULTS.json',C.DOC/'CONTROL_ROBUSTNESS.json',Path(__file__))],
        outputs=[C.bind(target),C.bind(figure)],new_fits=0),True)
    print('CONTROL_REPORT_WRITTEN')


if __name__=='__main__':main()
