"""공개 집계만으로 과거99장 결과의 두 지표/프레임별 상쇄 그림을 생성한다."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .old_reference import DOC, bind, save


def main():
    source = DOC / 'OLD_CLEAN19_POSE_REFERENCE.json'
    result = json.loads(source.read_text())
    rows = result['groups']['MODERATE_PLUS_SEVERE99']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1))
    colors = {'R0':'#666666','OLD_REF':'#a94442','S0':'#3b78ac','S1':'#248a52','S2':'#a47412'}
    for arm, row in rows.items():
        x, y = [row['conditional'][k]['median'] for k in ('translation_cm','rotation_deg')]
        axes[0].scatter(x, y, color=colors[arm], s=100, marker='x' if arm in ('R0','OLD_REF') else 'o')
        offset = (-43,-19) if arm=='S2' else (7,7)
        axes[0].annotate(arm, (x,y), xytext=offset, textcoords='offset points', fontsize=11)
    for arm in ('S1','S2'):
        a, b = rows['S0']['conditional'], rows[arm]['conditional']
        axes[0].annotate('', xy=(b['translation_cm']['median'],b['rotation_deg']['median']),
            xytext=(a['translation_cm']['median'],a['rotation_deg']['median']),
            arrowprops=dict(arrowstyle='->', color=colors[arm], alpha=.5))
    axes[0].set(xlabel='Translation median (cm), lower is better', ylabel='Full rotation median (deg), lower is better',
        title='Same 99 natural-occlusion frames; original D9')
    axes[0].set_xlim(10.,14.1)
    axes[0].set_ylim(3.5,9.5)
    axes[0].grid(alpha=.2)
    comparisons = result['contrasts']['MODERATE_PLUS_SEVERE99']
    segments = [('both_improve','Both improve','#248a52'),('T_only','T improves / R worsens','#5a93c8'),
        ('R_only','R improves / T worsens','#e9b34d'),('both_worsen','Both worsen','#bf5656')]
    bottom = np.zeros(2)
    for key,label,color in segments:
        vals=np.array([comparisons[a][key] for a in ('S1-minus-S0','S2-minus-S0')])
        axes[1].barh([0,1], vals, left=bottom, color=color, height=.52, label=label)
        for i, value in enumerate(vals):
            axes[1].text(bottom[i]+value/2, i, str(value), ha='center', va='center', color='white', fontweight='bold')
        bottom += vals
    assert list(bottom) == [99,99]
    axes[1].set(yticks=[0,1],yticklabels=['S1 minus S0','S2 minus S0'],xlim=(0,99),
        xlabel='Paired frames (N=99; 0 exact ties)',title='Aggregate gain does not mean every frame improves')
    axes[1].invert_yaxis()
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5,-.18), frameon=False, ncol=2,fontsize=9)
    fig.suptitle('Old Clean19: frozen-cache reaggregation, no new fit',fontweight='bold',fontsize=15)
    fig.text(.025,.01,'S0/S1/S2: historical matched arms. R0/OLD_REF: same-population reference, not a matched training-contract comparison. Reused DEV.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    path=DOC/'figures/old_clean19_pose_reference.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)
    save(DOC/'OLD_CLEAN19_FIGURE_MANIFEST.json',dict(input=bind(source),figure=bind(path),
        content='공개 aggregate T/R와paired개수만 사용; RGB/좌표/개별ID 없음',implementation=bind(Path(__file__))))
    save(DOC/'OLD_CLEAN19_VISUAL_SUMMARY.md','''# 과거 Clean19의 동시 개선 신호: 그림 요약

![동일 자연 가림99장의 과거 T/R 및 프레임별 변화](figures/old_clean19_pose_reference.png)

같은 자연 Moderate21+Severe78에서 S0의 **13.2197cm / 5.0326°**가 S1에서 **10.8778cm / 4.4412°**, S2에서 **10.5676cm / 4.6673°**로 감소했다. 동일99장의 R0 **12.1479cm / 6.0379°**보다도 두 median이 낮다.

하지만 이 결과는 과거 학습 계약에서의 신호다. 현재 Replay9/38+217 조건에서 같은 효과가 난다는 증거는 아니다. S1은 Clean29 T/R median이4.2969cm/1.5131°에서4.7072cm/1.5727°로 악화했다. Severe78에서 S2의 T median은 낮아졌지만 R median은12.5418°에서12.8392°로 높아졌다. 과거 Severe81의 R=7.2555°와 분모를 혼동하면 안 된다.

S1의 자연99 T/R P90은105.5464cm/88.4300°에서76.1457cm/87.9433°로 낮아졌다. S2 T P90은73.6958cm로 감소하지만 R P90은88.6276°로 높아져 모든 꼬리 오류 개선은 아니다. 정확한 수치는 [집계 JSON](OLD_CLEAN19_POSE_REFERENCE.json)에 저장했다.

프레임별 둘 다 개선된 것은 S1 40/99, S2 37/99다. 둘 다 악화한20/99도 모두 유지했다. 원본결과·모델·GT는 변경하지 않았고 새 fit/추론/PnP 실행은0회다.

[전체 난도·recording·paired 표](OLD_CLEAN19_POSE_REFERENCE.md)
''')


if __name__ == '__main__':
    main()
