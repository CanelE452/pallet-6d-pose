"""Render the completed diagnostic bounds and exact score attribution."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .selector_attribution import ROOT, DOC, PREV, read, save, bind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-presentation', action='store_true')
    args = parser.parse_args()
    def text_output(path, value):
        if args.refresh_presentation and path.exists():
            assert path.name in ('REPORT_KO.md','FIGURE_BINDINGS.json')
            path.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        else:
            save(path,value)
    bounds = read(DOC/'CANDIDATE_BOUNDS.json')
    attribution = read(DOC/'SELECTOR_ATTRIBUTION.json')
    models = ['R0','PRIOR1','FULL125','SINGLE251_mean3seeds','DIVERSE251_mean3seeds']
    labels = ['R0','PRIOR1','FULL125','SINGLE251\n3-seed mean','DIVERSE251\n3-seed mean']
    figures = DOC/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
    for ax,key,limit,title in zip(axes,
        ['clean_T_median_lower_bound_cm','natural_T_P90_lower_bound_cm'],
        ['clean_T_limit_cm','natural_T_P90_limit_cm'],
        ['Clean29: best possible T median','Natural99: best possible T P90']):
        values = [bounds['bounds'][m][key] for m in models]
        cap = bounds['bounds']['R0'][limit]
        ax.bar(labels, values, color=['#587c98','#6c98b5','#6c98b5','#58a39c','#bd7078'])
        ax.axhline(cap,color='#a52b35',linestyle='--',label=f'Original R0 × 1.05 ceiling: {cap:.6f}')
        ax.set(title=title,ylabel='Translation error (cm); lower is better',ylim=(0,max(values)*1.18))
        ax.tick_params(axis='x',labelsize=9)
        for j,v in enumerate(values):
            ax.text(j,v+max(values)*.018,f'{v:.3f}',ha='center',fontsize=10)
        ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Fixed coordinates + fixed detection: pointwise T-best candidate bounds\nGT used only for diagnosis; no deployable selector is evaluated',fontsize=13)
    p = figures/'01_candidate_bounds.png'
    assert args.refresh_presentation or not p.exists();fig.savefig(p,dpi=150);plt.close(fig)
    families = ['other_residuals','reprojection_rmse','shape_and_orientation','explicit_center_residual','rotation_matrix','translation']
    flabels = ['Other residual terms','Reprojection RMSE','Shape/orientation','Explicit center residual','Rotation matrix','Translation']
    amodels = ['FULL125','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3']
    fig,ax = plt.subplots(figsize=(11,5.8),layout='constrained')
    x = np.arange(len(families))
    for j,model in enumerate(amodels):
        data = attribution['populations']['NATURAL99'][model]['families']
        ax.bar(x+(j-1.5)*.19,[data[f]['mean_abs_contribution_change'] for f in families],width=.19,label=model)
    ax.set_xticks(x,flabels,rotation=15,ha='right')
    ax.set(yscale='log',ylabel='Mean |group contribution change| in long-minus-short score',
        title='Exact linear GEO score decomposition: changes from R0 on natural99\nAlgebraic attribution, not causal importance or a feature-selection result')
    ax.legend(fontsize=9)
    p = figures/'02_selector_contributions.png'
    assert args.refresh_presentation or not p.exists();fig.savefig(p,dpi=150);plt.close(fig)
    lines = ['# T·R 개선 상태와 다음 변경의 가능 범위', '',
        '**현재 T·R의 안정적인 동시 개선은 달성되지 않았습니다.** 새 6회 학습은 완료됐지만 기존 FULL125보다 낫지 않았고, 큰 T 오류가 늘었습니다. 이번 후속 분석은 그 실패 원인과 다음 변경의 한계를 좁힌 결과이며 개선 모델을 새로 얻은 결과가 아닙니다.', '',
        '| 자연 가림99 | T 중앙값 cm ↓ | R 중앙값 ° ↓ | T P90 cm ↓ |',
        '|---|---:|---:|---:|',
        '| R0 | 12.403 | 5.218 | 120.471 |',
        '| 기존 FULL125 | 11.986 | 4.411 | 120.824 |',
        '| DIVERSE251, seed별 통계의 평균 | 12.004 | 4.567 | 129.503 |', '',
        'R0 대비 일부 수치는 낮아졌으나 세 seed의 공동 개선, recording 재표본·제외 민감도와 꼬리 보존 조건을 통과하지 못했습니다. '
        '[기존 6회 실험의 전체 결과](../pallet_pose_stable_improvement_20261001_v1/REPORT_KO.md)와 [natural99 전체 비교 이미지](../pallet_pose_stable_improvement_20261001_v1/GALLERY_NATURAL99.md)를 유지합니다.', '',
        '## 이번에 추가로 확인한 내용', '',
        '1. 검출 오류 8건을 원본 이미지와 모든 저장 박스로 확인했습니다. 선택된 박스는 다른 팔레트보다 콘 받침·콘의 사각 부위에 걸렸습니다. 저장 pool에 팔레트 참조와 맞는 박스가 있는 것은 3건이고 5건은 없습니다. [박스·실제 이미지 감사](ASSOCIATION_AUDIT_KO.md).',
        '2. 모든 9모델×173장의 선택 점수를 94개 특징의 합으로 다시 계산했습니다. 기존 선택과 1,557/1,557건 일치했습니다. 점수 변화는 주로 재투영 잔차 항에 나타났고, 중심점 항 하나만의 문제라는 근거는 얻지 못했습니다.',
        '3. 정답을 알고 기존 W/D 후보 중 T가 가장 작은 pose를 골라도 FULL125의 clean 기준과 DIVERSE 3-seed 평균의 T 꼬리 기준을 통과할 수 없음을 계산했습니다. 따라서 이 구성들의 모든 문제를 선택기 재학습 하나로 해결한다는 다음 계획은 성립하지 않습니다.', '',
        '## 후보 선택만 바꿔서 도달할 수 있는 하한', '',
        '프레임 i의 고정된 후보 집합을 Cᵢ라 하면 min꜀∈Cᵢ T(i,c) ≤ T(i,선택결과)입니다. 같은 분모에서 프레임별 하한의 중앙값·P90도 어떤 선택 결과의 해당 통계보다 클 수 없습니다. seed별 분위수의 평균도 같은 순서를 보존합니다. 이 하한부터 허용값을 넘으면 그 후보 집합 안에서 선택만 바꾸는 방법으로 해당 기준을 통과할 수 없습니다.', '',
        '현재 natural99와 clean29는 모든 모델에서 pose 실패가 0건입니다. 원래 실패 수 비증가 기준까지 유지해야 하므로, 어려운 프레임을 실패로 바꿔 조건부 통계에서 빼는 선택은 통과 방법이 될 수 없습니다. 아래 3-seed 평균 행으로 집단 평균 기준을 판단하며, 개별 seed의 한계와 혼동하지 않습니다.', '',
        '**T-best는 참조 정답을 사용하는 진단입니다.** 하나의 전체 (R,t)를 선택하며 R-best와 별도로 계산했습니다. 서로 다른 후보의 최저 T·최저 R를 하나의 성능으로 합치지 않았고 실행 시 사용하는 예측으로 배포하지 않았습니다.', '',
        '| 고정 출력 | clean T 중앙값 하한 cm | 자연99 T P90 하한 cm | 이 두 하한으로 제외되는가 |',
        '|---|---:|---:|---|']
    for model,label in zip(models,['R0','PRIOR1','FULL125','SINGLE251, 3-seed 평균','DIVERSE251, 3-seed 평균']):
        b=bounds['bounds'][model]
        reasons=[]
        if b['clean_T_guard_impossible']: reasons.append('clean T')
        if b['natural_T_tail_guard_impossible']: reasons.append('자연 T P90')
        lines.append(f'| {label} | {b["clean_T_median_lower_bound_cm"]:.6f} | {b["natural_T_P90_lower_bound_cm"]:.6f} | '+(' / '.join(reasons) if reasons else '제외되지 않음; 성공은 미입증')+' |')
    lines += ['', '원래 허용값은 clean T 중앙값 **3.076355407cm**, 자연99 T P90 **126.494938608cm**입니다. DIVERSE clean 초과량은 0.000016957cm로 물리적으로 의미 있는 차이라고 해석하지 않습니다. 자연 T P90 하한 129.502732cm는 별도로 기준을 초과합니다. 기존 기준을 사후 완화하지 않습니다.', '',
        '![고정 후보 집합에서 가능한 T 하한](figures/01_candidate_bounds.png)', '',
        '범위는 **같은 좌표·같은 선택 검출 박스·같은 solver·같은 W/D 후보 두 개**입니다. 좌표 보정, 검출기 수정, 다른 solver, 모델 간 후보 합집합에는 이 하한을 그대로 적용할 수 없습니다. R0와 SINGLE 평균이 이 두 하한에 걸리지 않는다고 배포 가능한 선택기가 존재하거나 다른 안정성 조건을 통과한다는 뜻도 아닙니다. 잘 나온 seed 하나를 선택하지 않았습니다.', '',
        f'기존 R0/PRIOR1/FULL125 oracle과 수치 {bounds["old_oracle_numeric_parity_checks"]}개가 1e−7 절대 허용 범위 안에서 일치했습니다. 새 6모델까지 확장한 [전체 후보 요약](CANDIDATE_BOUNDS.json)과 [전체 3,114행](CANDIDATE_BOUND_ROWS.csv)을 제공합니다.', '',
        '## 왜 GEO 선택이 바뀌는가', '',
        '기존 선택기는 두 후보에 동일한 Linear94를 적용하고 작은 점수를 고릅니다. 두 점수 차이에서는 bias와 평균 정규화 항이 상쇄됩니다. 따라서 각 특징의 `weight × (long−short) / std`를 합하면 실제 결정 점수 차이를 재구성할 수 있습니다.', '',
        f'원래 점수와 최대 차이는 {attribution["score_parity_max_absolute_error"]:.3g}, 특징 합으로 재구성한 margin의 최대 차이는 {attribution["additive_margin_reconstruction_max_absolute_error"]:.3g}였습니다. 모든 1,557건에서 선택 가설 이름은 일치합니다. 같은 후보 쌍에 공통인 confidence·bbox 직접 특징은 점수 차이에 기여하지 않습니다. bbox 정규화와 confidence 가중 residual처럼 다른 항을 통한 영향은 남으므로 confidence 전체가 무관하다고 해석하지 않습니다.', '',
        '![GEO 점수 변화의 항별 분해](figures/02_selector_contributions.png)', '',
        'natural99의 FULL125에서 R0 대비 residual 그룹 기여 변화의 평균 절댓값은 0.919, 명시적 중심 residual 항은 0.0128입니다. DIVERSE에서는 각각 1.157–1.220과 0.0155–0.0196입니다. 명시적 중심 항은 4개의 직접 중심 residual 특징만 포함하며 집계 residual에는 중심의 간접 영향이 있습니다. 그래프의 그룹별 절댓값은 서로 상쇄될 수 있어 전체 변화에 대한 비율이나 인과적 중요도가 아닙니다. 특징 삭제·가중치 조절을 실제 성능 개선으로 시험한 결과도 아닙니다.', '',
        '[점수 분해 JSON](SELECTOR_ATTRIBUTION.json) · [모든 모델·프레임 margin CSV](SELECTOR_MARGIN_ROWS.csv)', '',
        '## 다음 변경에 대한 판단', '',
        '- 현재 FULL125 또는 DIVERSE 좌표를 유지한 채 GEO만 다시 학습하는 방안은 전체 목표를 해결하는 단독 변경으로 채택하지 않습니다. 회전 선택 일부를 개선할 여지는 별도로 남아 있습니다.',
        '- 현재 R0의 실제 합성 출력으로 GEO를 재보정하는 방안은 위 하한으로 제외되지 않습니다. 다만 이 경우 개선 대상은 R0+선택기이며, FULL/ST 보정기가 해결됐다고 말할 수 없습니다. 과거 출력별 선택기 재학습의 음성 결과도 있습니다. [정확한 cache·split·실행 비용 감사](SELECTOR_FEASIBILITY_KO.md).',
        '- 연속 T 문제를 다루려면 좌표 또는 검출 경로를 바꾸는 별도 근거가 필요합니다. pose/Jacobian loss와 여러 검출 변형은 이미 시도됐으므로 이름만 바꿔 반복하지 않습니다. 현재 출력의 무학습 pose-mode 복구 검사는 검토 가능한 다음 진단이며 학습 성공을 보장하지 않습니다. [과거 연속 pose 변경·중복 감사](CONTINUOUS_POSE_AUDIT_KO.md).', '',
        '아직 새 개선 모델은 채택하지 않았습니다. 전체 목표는 계속 미달성 상태입니다. 이 분석은 재사용 DEV에서 결과를 확인한 뒤 수행한 탐색 분석입니다. 모델·seed·threshold를 고르는 독립 TEST로 사용할 수 없습니다.', '',
        '## 실행과 보존', '',
        '새 fit 0회, 이미지 신경망 forward 0회입니다. 저장 좌표의 PnP 특징 재계산과 저장 후보의 참조 오차 평가만 수행했습니다. 이전 주 결과 SHA는 실행 전후 동일합니다. 참조는 기존 2D·K·치수에서 유도한 geometry reference이며 독립 실측 pose가 아닙니다. 새 실사 정답을 학습에 추가하지 않았고 이전 교사의 실사 수동 코너 사용 이력도 지우지 않았습니다.', '',
        '재현 명령은 저장소 루트의 기존 `pallet-yolo26` 환경에서 실행합니다. 출력이 있으면 불필요하게 재실행하지 않도록 새 파일 생성을 제한했습니다. 원본 이미지·checkpoint·일부 선행 cache는 로컬 경로와 SHA로 연결되고 이 문서만으로 다운로드되지는 않습니다.', '',
        '```bash',
        'python -m scripts.research.pallet_pose_joint_recovery_20261001_v1.selector_attribution',
        'python -m scripts.research.pallet_pose_joint_recovery_20261001_v1.candidate_bounds',
        'python -m scripts.research.pallet_pose_joint_recovery_20261001_v1.make_report',
        '```', '',
        '[후속 분석 코드](../../../scripts/research/pallet_pose_joint_recovery_20261001_v1/) · [게시 파일 목록과 SHA](PUBLICATION_MANIFEST.json)', '']
    text_output(DOC/'REPORT_KO.md','\n'.join(lines))
    text_output(DOC/'FIGURE_BINDINGS.json',dict(figures=[bind(p) for p in sorted(figures.glob('0*.png'))],
        source_results=[bind(DOC/'CANDIDATE_BOUNDS.json'),bind(DOC/'SELECTOR_ATTRIBUTION.json')],
        charts_describe_existing_outputs_only=True))


if __name__ == '__main__':
    main()
