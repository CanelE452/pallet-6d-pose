from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

O = Path(__file__).resolve().parent
data = json.loads((O / 'CROSS40_AGREEMENT_BEFORE_HISTORICAL_LABEL_JOIN.json').read_text())
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={'width_ratios': [1.5, 1]})
fig.suptitle('공개 학습용 고정 40사례: 부위·query 대응 독립 판독', fontproperties=font, fontsize=18, y=.98)
native = data['per_native']
ns = [native[str(k)]['N'] for k in range(8)]
agree = [native[str(k)]['part_visibility_unanimous'] for k in range(8)]
axes[0].bar(range(8), agree, color='#168888', label='4개 판독 일치')
axes[0].bar(range(8), [n-a for n,a in zip(ns,agree)], bottom=agree, color='#e2b972', label='불일치 보존')
for k,n in enumerate(ns):
    axes[0].text(k, n+.1, f'{agree[k]}/{n}', ha='center', fontsize=11)
axes[0].set_xticks(range(8), [str(k) for k in range(8)])
axes[0].set_xlabel('native 슬롯', fontproperties=font)
axes[0].set_ylabel('사례 수', fontproperties=font)
axes[0].set_ylim(0, 9)
axes[0].set_title('의도한 부위 가시성: 판독 일치 여부', fontproperties=font, fontsize=13)
axes[0].legend(prop=font, loc='upper left')
axes[1].barh([1, 0], [13, 1], color=['#3877b5', '#8a6bb0'])
axes[1].set_yticks([1, 0], ['부위는 보임\nquery는 배경', '부위는 보임\nquery는 다른 부위'], fontproperties=font)
for y,x in [(1,13),(0,1)]:
    axes[1].text(x+.25, y, f'{x}/40', va='center', fontsize=13)
axes[1].set_xlim(0, 17)
axes[1].set_title('두 축의 차이에 모두 동의한 사례', fontproperties=font, fontsize=13)
axes[1].set_xlabel('사례 수', fontproperties=font)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x' if ax is axes[1] else 'y', alpha=.15)
    ax.set_axisbelow(True)
fig.text(.02, .045, '같은 계열 assistant의 약관찰이며 사람 정답·모델 정확도가 아닙니다. native 균형 표본이므로 전체 발생률로 해석하지 않습니다.', fontproperties=font, fontsize=10)
fig.text(.02, .012, '기존 라벨·모델 예측을 가리고 판독. 불일치는 다수결로 해소하지 않음. native6은 원래 학습 자료에 2개뿐. 전체390 판독은 별도 집계.', fontproperties=font, fontsize=10)
fig.tight_layout(rect=(0,.1,1,.93))
dest = O / 'CROSS40_AGREEMENT_AND_QUERY_RELATION.png'
assert not dest.exists()
fig.savefig(dest, dpi=150, facecolor='white')
print(dest)
