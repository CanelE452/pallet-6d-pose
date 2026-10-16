from run import ROOT, read, write, sha, clean
from pathlib import Path
import argparse, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
TITLE='Hough 의사 라벨 품질 검증'
LABELS={'baseline':'기존 좌표','line_uncertainty_only':'선 불확실성 보정','unconditional_lambda1':'무조건 선 보정'}

def main(out):
    s=read(out/'SUMMARY.json'); rows=read(out/'ROWS.json'); (out/'figures').mkdir(exist_ok=True)
    assert read(out/'AUDIT.json')['status']=='PASS'
    primary=s['results']['F4_accepted']; b=primary['baseline']; p=primary['line_uncertainty_only']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axes[0].bar(['Original','Weak Hough','Full Hough'],[primary[a]['mean_px'] for a in LABELS],color=['#376a9d','#e88726','#9ba4ae'])
    axes[0].set(ylabel='Mean error (px), lower is better',title='Same 35 accepted pseudo-label candidates')
    groups=['eval_cad','eval_noapril','eval_outside']; x=np.arange(3)
    for j,a in enumerate(['baseline','line_uncertainty_only']):
        axes[1].bar(x+(j-.5)*.35,[s['results']['accepted/'+g][a]['mean_px'] for g in groups],width=.35,label=['Original','Weak Hough'][j],color=['#376a9d','#e88726'][j])
    axes[1].set(xticks=x,xticklabels=['CAD (18)','No-April (12)','Outside (5)'],ylabel='Mean error (px)',title='Average improvement hides session harm');axes[1].legend()
    fig.savefig(out/'figures/quality.png',dpi=150); plt.close(fig)
    accepted=[r for r in rows if r['accepted']]
    def diff(r):
        mask=np.asarray(r['mask'],bool)
        return float((np.asarray(r['errors']['1']['line_uncertainty_only'])-np.asarray(r['errors']['1']['baseline']))[mask].mean())
    helped=sorted([r for r in accepted if diff(r)<0],key=diff); harmed=sorted([r for r in accepted if diff(r)>0],key=diff)
    cases=[('help',helped[len(helped)//2]),('harm',harmed[len(harmed)//2]),('remaining_bad',max(accepted,key=lambda r:max(np.asarray(r['errors']['1']['line_uncertainty_only'])[r['mask']])))]
    metadata=[]
    for name,r in cases:
        fig,axes=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
        im=np.asarray(Image.open(r['image']).convert('RGB')); gt=np.asarray(r['gt']); mask=np.asarray(r['mask'],bool); base=np.asarray(r['predictions']['1']['baseline']); corrected=np.asarray(r['predictions']['1']['line_uncertainty_only'])
        for j,ax in enumerate(axes):
            ax.imshow(im); ax.set(xlim=(-.5,im.shape[1]-.5),ylim=(im.shape[0]-.5,-.5)); ax.axis('off')
            ax.scatter(gt[mask,0],gt[mask,1],s=80,marker='x',c='#00e860',linewidths=2,label='GT (scored)')
            points=base if j==0 else corrected
            ax.scatter(points[:,0],points[:,1],s=70,marker='o' if j==0 else '+',facecolors='none' if j==0 else '#ff9500',edgecolors='#398aff' if j==0 else '#ff9500',linewidths=1.8,label='Original' if j==0 else 'Weak Hough')
            for k in np.flatnonzero(mask): ax.text(gt[k,0]+4,gt[k,1]-4,str(k),color='#00e860',fontsize=9,weight='bold',bbox=dict(facecolor='black',alpha=.5,pad=1,edgecolor='none'))
            if j:
                for k,line in enumerate(r['lines']['1']):
                    a,z=np.asarray(line); ax.plot([a[0],z[0]],[a[1],z[1]],'--',color='#ffce58',alpha=.55,lw=1,label='DHT predicted lines' if k==0 else None)
            e=np.asarray(r['errors']['1']['baseline' if j==0 else 'line_uncertainty_only'])[mask]
            ax.set_title(('Original' if j==0 else 'Weak Hough')+f' | mean {e.mean():.2f}px | max {e.max():.2f}px')
            ax.legend(loc='upper right',fontsize=8)
        fig.suptitle(r['id']+' | seed 1 | '+name)
        fig.savefig(out/f'figures/{name}.png',dpi=140);plt.close(fig)
        metadata.append(dict(name=name,id=r['id'],seed=1,mean_change_px=diff(r),selection='median improved / median harmed / largest remaining error among accepted; descriptive, never used for tuning'))
    write(out/'CASES.json',metadata)
    def table_lines():
        text=['| 기존 필터 통과 35장·307점 | 평균 오차 px | 중앙값 px | P90 px | 20px 초과 오류가 있는 장수 |','|---|---:|---:|---:|---|']
        for a,label in LABELS.items():
            z=primary[a]; text.append(f"| {label} | {z['mean_px']:.3f} | {z['median_px']:.3f} | {z['p90_px']:.3f} | "+'/'.join(str(v['bad_frames']['20']) for v in z['per_seed'])+' |')
        return '\n'.join(text)
    text=f'''# {TITLE}

**평균 좌표 품질은 개선됐지만, 안정적인 의사 라벨 정제로 채택할 기준은 통과하지 못했다.**

기존 F4 필터와 YOLO 예측을 고정한 뒤, 기존 DHT의 선 불확실성을 반영한 약한 보정(λ=0.0625)을 적용했다. 새 학습·새 문턱값 탐색·GT 좌표 수정은 없다. 무조건 보정 λ=1은 실패 원인을 비교하는 고정 대조다.

{table_lines()}

수치는 3 DHT 시드별 지표의 평균이며 마지막 열은 시드 1/2/3이다. 같은 35장의 반복이므로 105개의 독립 이미지가 아니다. 평균 개선은 {100*(1-p['mean_px']/b['mean_px']):.2f}%, 시드별 개선율은 {', '.join(f"{v['mean_gain_percent']:.2f}%" for v in p['per_seed'])}다. 평균 오차 변화의 조건부 95% 구간은 [{p['difference_ci95'][0]:.3f}, {p['difference_ci95'][1]:.3f}]px다.

![전체와 세션별 라벨 오차](figures/quality.png)

## 무엇이 좋아졌고, 무엇이 남았나

- 좌표 오차와 P90은 세 시드 모두 감소했다. 원래 20px 초과 점은 20개였고, 보정 후 각 시드 19개였다.
- **20px 초과 오류가 하나라도 있는 라벨은 15/35장으로 그대로다.** 세 시드 모두 잘못된 라벨→양호 라벨, 양호 라벨→잘못된 라벨 전환은 0장이다. 해당 기준의 양호 라벨 비율은 20/35=57.14%로 유지된다. 선이 라벨을 조금 다듬었지만 큰 오류 라벨을 복구하지는 못했다.
- 기존 오차≤10px인 199점을 보면 seed3에서 평균 4.834→{p['per_seed'][2]['easy_mean_px']:.3f}px, 약 {p['per_seed'][2]['easy_worsening_percent']:.2f}% 악화했다. 약 0.10px의 작은 변화이며 대형 붕괴라고 해석하지 않는다. 하지만 사전에 정한 1% 보존 기준을 넘었다.
- 촬영 세션별 방향도 다르다. CAD 18장은 평균 12.030→11.140px, outside 통과 5장은 7.087→6.707px로 좋아졌다. **noapril 12장은 4.099→4.378px로 세 시드 모두 악화**했다. 이 부분은 사후 설명이며 세션별 보정 사용 규칙으로 채택하지 않는다.
- 10px 기준에서는 시드별 양호로 복구된 라벨 3/2/1장과 새로 잘못된 라벨 1/1/1장이 함께 존재한다. 한 픽셀 이상 악화한 감독 점도 33/31/49개다. 평균 향상이 개별 라벨의 안전성을 보장하지 않는다.
- 주 비교에 35개의 중심점도 포함하며 중심은 보정하지 않는다. 8코너만 평가한 수치도 SUMMARY.json에 따로 남겼다. 중심점까지 포함한 M4 정의와 과거 DHT52장의 8코너 수치를 섞지 않는다.

## 데이터와 검증 범위

기존 DHT DEV52장과 기존 M4 plastic 평가 기록의 교집합은 40장이다. 나머지 12장은 M4 비교 기록에 없어 제외했다. 교집합은 eval_cad18/eval_noapril12/eval_outside10이며, outside에서 5장이 기존 F4에 거절돼 주 비교는 35장이다. 원래 좌표·유효점·GT 좌표·원본 이미지 SHA가 일치함을 검사했다. 정본 EVAL_CANONICAL의 objects[0].split=eval도 확인했다.

GT는 기존 M4의 gt_xy와 gt_supervised를 그대로 사용했다. 총 307개 감독·관측 점(코너272+중심35)을 점 오차 분모로 삼는다. 수동/재구성 정답의 물리적 정확성을 새로 인증한 실험은 아니다. GT는 오차 계산에만 쓰고 보정·기존 필터에는 입력하지 않았다.

**실제 무라벨 학습 pool의 품질이나 학생 재학습 성능을 검증한 결과는 아니다.** 평가 이미지에서 의사 라벨 후보의 품질을 측정한 진단이다. 이 평가 이미지는 학습용 의사 라벨로 내보내지 않는다. 이미 본 DEV이며, 기존 방법의 선택도 이 DEV 결과를 알고 있다. 3개 촬영 세션 내 프레임 bootstrap 구간은 이 표본에 조건부이고 새 장면 일반화·프레임 시간 상관·GT 주석 오차를 모두 반영하지 않는다.

## 판정과 종료 범위

사전 기준: 평균 1% 이상 개선(각 시드), 조건부 차이 구간 상한<0, P90 비악화(각 시드), 20px 초과 오류 라벨 수 비증가(각 시드), 기존≤10px 점의 평균 악화≤1%(각 시드).

판정: `{s['decision']}`. 앞의 네 기준은 통과했고 마지막 쉬운 점 보존 기준은 실패했다. 개선 효과가 없다는 뜻은 아니다. **현재 규칙을 전체 의사 라벨에 일괄 적용해 학생을 학습시킬 근거는 부족하다.** 이번 작업에서는 학생 학습을 시작하지 않았고, 새 계수나 필터를 찾아 같은 표본에 맞추지 않았다. 앞서 종료한 고정 특징 보정 아키텍처 실험도 재개하지 않았다.

기존 함수를 이용한 보정 출력을 독립적인 절대좌표 가중 최소제곱 풀이로 확인했고, 원래 저장 예측과도 일치했다. 원 필터 통과 여부와 원 GT 오차를 재계산했다. 자세한 수치와 재현 코드는 [SUMMARY](SUMMARY.json), [AUDIT](AUDIT.json), [PROTOCOL](PROTOCOL.json), [개별 점·선·오차](ROWS.json)를 따른다.

## 실제 개선·악화 사례

고정 seed1에서 평균 개선 사례의 중앙, 평균 악화 사례의 중앙, 보정 후 최대 오차가 가장 큰 통과 라벨을 각각 표시했다. 사례는 설명용이며 규칙 선택에 사용하지 않았다. 녹색 ×는 평가에 쓰인 GT, 파란 원은 기존 좌표, 주황 +는 보정 좌표다. 노란 점선은 실제 DHT가 예측한 무한 직선이며 attention 또는 관측 가능한 유한 물리 선분이 아니다.

![중앙 개선 사례](figures/help.png)

![중앙 악화 사례](figures/harm.png)

![큰 오류가 남은 통과 라벨](figures/remaining_bad.png)
'''
    (out/'REPORT_KO.md').write_text(text)
    import markdown
    body=markdown.markdown(text,extensions=['tables'])
    (out/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+TITLE+'</title><style>body{max-width:1180px;margin:36px auto;padding:0 24px;font:17px/1.75 sans-serif;color:#24354b}h1{font-size:32px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:16px}td,th{border:1px solid #bbcadb;padding:10px}th{background:#edf4fa}img{width:100%;height:auto}a{color:#1762a2}code{overflow-wrap:anywhere}</style>'+body+'</html>')
    note=ROOT/'_docs/notes/supervision_and_distillation/hough-pseudolabel-quality-v1.md'
    # Convert report links to the actual result directory, relative to the note.
    import re,os
    note_text=re.sub(r'\]\(([^)]+)\)',lambda m:']('+os.path.relpath(out/m.group(1),note.parent)+')',text)
    note.write_text(note_text)
    write(out/'REPORT_COMPLETE.json',dict(complete=True,summary_sha256=sha(out/'SUMMARY.json'),markdown_sha256=sha(out/'REPORT_KO.md'),html_sha256=sha(out/'index.html'),figures=4,note=str(note.relative_to(ROOT))))
    print('Report complete',note)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'data/pallet/results/hough_pseudolabel_quality_v1'); main(p.parse_args().out)
