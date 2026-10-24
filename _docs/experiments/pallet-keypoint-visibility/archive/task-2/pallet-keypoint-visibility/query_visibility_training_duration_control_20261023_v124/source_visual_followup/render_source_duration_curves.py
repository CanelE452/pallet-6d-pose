from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V=Path(__file__).resolve().parents[1]
RUN=V/'runs/C16_EXACT_C15_DURATION_CONTROL_8000'
h=[json.loads(x) for x in (RUN/'HISTORY.jsonl').read_text().splitlines()]
font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family']='NanumGothic'
plt.rcParams['axes.unicode_minus']=False
steps=[x['update'] for x in h]
means=[x['criterion_fourteen_VAL_cohorts_BCE'] for x in h]
part=[x['VAL_cohorts']['TOMIE_DEVELOPMENT_E']['mean_BCE'] for x in h]
fig,ax=plt.subplots(1,2,figsize=(14,4.8),layout='constrained')
for a in ax:
    a.spines[['top','right']].set_visible(False)
    a.grid(alpha=.2)
    a.set_xlabel('C15와 동일한 조건의 총 업데이트')
    a.set_xticks([0,1000,3000,4500,6000,8000])
ax[0].plot(steps,means,color='#286a99',lw=2,label='공개 검증 14항목 평균 BCE')
ax[0].scatter([1000,4500,8000],[means[10],means[45],means[80]],
              color=['#666666','#c55a2c','#9b3b3b'],s=55,zorder=4)
ax[0].set_title('공개 검증은 4,500회 모델을 선택',fontsize=15)
ax[0].set_ylabel('동일한 14항목 평균 BCE')
ax[0].legend(frameon=False)
for s,y,offset in [(1000,means[10],(8,14)),(4500,means[45],(8,22)),(8000,means[80],(-70,18))]:
    ax[0].annotate(f'{s:,}회\n{y:.5f}',(s,y),xytext=offset,
                   textcoords='offset points',fontsize=10)
ax[1].plot(steps[1:],part[1:],color='#b86037',lw=2,
           label='TOMIE 외부 가림 E 항목 (100회 이후)')
ax[1].set_ylim(.24,.35)
ax[1].set_ylabel('TOMIE 외부 가림 E의 BCE')
ax[1].set_title('잔여 가림 항목: 소폭 개선 뒤 악화',fontsize=15)
ax[1].legend(frameon=False,loc='upper left')
for a in ax:
    a.axvline(1000,color='#888888',ls=':',lw=1)
    a.axvline(4500,color='#c55a2c',ls='--',lw=1)
fig.suptitle('C16 학습 기간 대조: 데이터·구조·학습률·문턱 유지, 현재 실사로 선택하지 않음',fontsize=15)
fig.savefig(Path(__file__).parent/'SOURCE_DURATION_CURVES.png',dpi=150)
plt.close(fig)
