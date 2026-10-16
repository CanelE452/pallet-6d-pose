from common import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-role-selection-mpl')
import numpy as np,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
NOTE=ROOT/'_docs/notes/pose_geometry/role-selection-and-residual.md'

def main():
    assert read(OUT/'FINAL_AUDIT.json')['PASS']
    folder=OUT/'foreground_matched';s=read(folder/'SUMMARY.json');cond=read(folder/'CONDITIONAL_ANALYSIS.json')['populations']
    initial=read(OUT/'SUMMARY.json');mean=read(OUT/'mean_risk_followup'/'SUMMARY.json');training=read(OUT/'TRAINING_DIAGNOSIS.json')
    pred=np.load(folder/'predictions.npz');rows=read(folder/'test_records.json');frames=read(folder/'PER_FRAME.json')
    labels={'stock':'기존 YOLO','canonical':'규칙 기반 ID 재정렬','unmatched_training':'대상 연결 수정 전','direct':'점 보정만 · 대상 연결 수정','selected':'순서 선택 + 보정 · 대상 연결 수정'}
    methods=list(labels)
    def values(method,pop,key,state=None):
        tags=[method] if method in ['stock','canonical'] else [f'{method}_s{i}' for i in [1,2,3]]
        return np.array([s['results'][t][pop]['overall'][key] if state is None else s['results'][t][pop]['conditions'][state][key] for t in tags])
    def fmt(v):return f'{v.mean():.3f}'+(f' ± {v.std(ddof=1):.3f}' if len(v)>1 else '')
    header=['방법','본 종류 평균 px','새 종류 평균 px','새 종류 주 지표↓','새 종류 P90 px','이동 오차 cm']
    table=[[labels[m],fmt(values(m,'seen','matched_mean_px')),fmt(values(m,'unseen','matched_mean_px')),
            f'{values(m,"unseen","primary").mean():.6f}',f'{values(m,"unseen","matched_p90_px").mean():.2f}',f'{values(m,"unseen","translation_median_cm").mean():.3f}'] for m in methods]
    seedtable=[]
    for seed in [1,2,3]:
        q=s['selector'][f'selected_s{seed}']
        seedtable.append([str(seed),f'{s["results"][f"selected_s{seed}"]["seen"]["overall"]["matched_mean_px"]:.3f}',
            f'{s["results"][f"selected_s{seed}"]["unseen"]["overall"]["matched_mean_px"]:.3f}',
            f'{100*s["selected_vs_stock_gain_per_seed"][seed-1]:+.3f}%',f'{q["useful"]}/{q["harmful"]}',str(s['new_good_to_bad_all_assets'][seed-1])])
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for ax,pop in zip(axes,['seen','unseen']):
        selected=['stock','direct','selected'];v=[values(m,pop,'matched_mean_px') for m in selected]
        ax.bar(range(3),[x.mean() for x in v],yerr=[x.std(ddof=1) if len(x)>1 else 0 for x in v],capsize=5,color=['#64748b','#38bdf8','#f97316'])
        ax.set_xticks(range(3),['YOLO','Direct correction','Role + correction']);ax.set_ylabel('Matched mean corner error (px)')
        ax.set_title(('Seen pallet types' if pop=='seen' else 'Unseen pallet types')+' / new scenes');ax.grid(axis='y',alpha=.2)
    fig.savefig(figdir/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,pop in zip(axes,['seen','unseen']):
        for method,color in [('stock','#64748b'),('direct','#38bdf8'),('selected','#f97316')]:
            v=[values(method,pop,'matched_mean_px',state).mean() for state in ['clean','panel','crop25']]
            ax.plot(range(3),v,'o-',label=method,color=color)
        ax.set_xticks(range(3),['Clean','Occlusion','Crop25']);ax.set_title(pop+' types');ax.set_ylabel('Matched mean error (px)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(figdir/'conditions.png',dpi=160);plt.close(fig)
    gt=np.array([r['gt'] for r in rows]);b=np.linalg.norm(pred['stock'][:,:8]-gt,axis=-1).mean(-1)
    c=np.linalg.norm(pred['selected_s1'][:,:8]-gt,axis=-1).mean(-1)
    valid=[i for i,r in enumerate(rows) if frames['stock'][i]['matched']]
    candidates=[('seen_gain','Seen-type improvement',max([i for i in valid if rows[i]['seen']],key=lambda i:b[i]-c[i])),
                ('unseen_gain','Unseen-type improvement',max([i for i in valid if not rows[i]['seen']],key=lambda i:b[i]-c[i])),
                ('unseen_harm','Unseen-type failure',max([i for i in valid if not rows[i]['seen']],key=lambda i:c[i]-b[i]))]
    gallery=[]
    for name,label,i in candidates:
        fig,axes=plt.subplots(1,3,figsize=(16,5));im=Image.open(ROOT/rows[i]['image'])
        for ax,tag,title in zip(axes,['stock','direct_s1','selected_s1'],['YOLO','Direct correction','Role selection + correction']):
            p=pred[tag][i,:8];ax.imshow(im);ax.scatter(gt[i,:,0],gt[i,:,1],s=65,facecolors='none',edgecolors='#00ff99',linewidths=2);ax.scatter(p[:,0],p[:,1],c='#ff4388',s=24)
            for j in range(8):
                ax.text(gt[i,j,0]-8,gt[i,j,1]-7,str(j),c='#00ff99',fontsize=10,weight='bold',bbox=dict(facecolor='black',alpha=.5,pad=.2))
                ax.text(p[j,0]+5,p[j,1]+8,str(j),c='#ff4388',fontsize=10,weight='bold',bbox=dict(facecolor='black',alpha=.5,pad=.2))
            ax.set_title(f'{title}\n{np.linalg.norm(p-gt[i],axis=-1).mean():.2f}px');ax.axis('off')
        fig.suptitle(label+' | '+rows[i]['id']+' | seed1 | Green GT / pink prediction',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(figdir/(name+'.jpg'),dpi=130);plt.close(fig)
        gallery.append(dict(label=label,id=rows[i]['id'],src='figures/'+name+'.jpg'))
    def mdtable(h,t):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in t)
    def htable(h,t):return '<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in t)+'</tbody></table>'
    seedheader=['시드','본 종류 px','새 종류 px','새 종류 주 지표 개선','유익/유해 순서 변경','좋음→큰 오류 · 전체 종류']
    seen=s['results']['stock']['seen']['overall'];unseen=s['results']['stock']['unseen']['overall']
    initialtable=[]
    for label,obj,names in [('1차 새 장면',initial['results']['fresh'],['stock','A_direct','C_select_residual']),('손실 대조 새 장면',mean['results'],['stock','direct','mean'])]:
        vv=[]
        for name in names:
            v=obj['stock']['unseen']['overall']['matched_mean_px'] if name=='stock' else np.mean([obj[f'{name}_s{i}']['unseen']['overall']['matched_mean_px'] for i in [1,2,3]])
            vv.append(f'{v:.3f}')
        initialtable.append([label]+vv)
    text=f'''# 코너 대응 선택과 작은 보정 분리 — 실제 검증

2026-09-11. **본 팔레트 종류의 새 장면에서는 평균 개선이 재현됐지만, 처음 보는 종류에서 안정적인 추가 개선은 실패했다.** 사용자가 승인한 순서 선택+보정 구조를 구현하고, 발견한 손실·학습 대상 연결 문제까지 대조했다. 이번 결과를 보편적인 아키텍처 불가능성이나 논문용 안정적 개선으로 확대하지 않는다.

[HTML·이미지 보고서](../../../../data/pallet/results/pallet_role_selection_v2/index.html) · [최종 수치](../../../../data/pallet/results/pallet_role_selection_v2/foreground_matched/SUMMARY.json) · [실행 코드](../../../../scripts/research/pallet_role_selection_v2/README.md)

## 최종 비교: GT 대응 감독을 수정한 뒤 새로 만든576관측

평균·P90은 모든 군에 공통인 검출 프레임의8개 코너 오차다. 본 종류 **{seen['matched']}/288**, 새 종류 **{unseen['matched']}/288**가 검출·매칭됐다. ±는3시드 표준편차. 주 지표는 실패를 이미지 대각선 오차로 포함한 전체 프레임 정규화 오차다.

{mdtable(header,table)}

- 본 종류: 기존 **23.916→17.823px**, 평균 **25.48% 감소**. 직접 점 보정22.495px보다도 세 시드 모두 좋았다. 새 장면 그룹 bootstrap의 C−직접보정 주 지표 구간은 **[{cond['seen']['selected_vs_direct_ci']['low']:.7f}, {cond['seen']['selected_vs_direct_ci']['high']:.7f}]**로0미만이다. 이는 사전 주 목표인 새 종류 일반화와 별도의 조건부 결과다.
- 새 종류: 기존 **25.140→24.396px**, 평균 **2.96% 감소**지만, **직접 점 보정23.736px보다 나빴다.** 검출 실패 포함 주 지표에서는1시드 개선·2시드 악화다. 원시 픽셀 평균과 정규화 주 지표는 crop 해상도에 따라 가중치가 달라 결론의 부호도 다를 수 있다.
- 새 종류 C−기존 주 지표95%구간은 **[{s['selected_minus_stock_CI']['low']:.7f}, {s['selected_minus_stock_CI']['high']:.7f}]**로0을 포함한다. 사전 안정성 기준은 통과하지 못했다.
- 새 종류는 scene_2.usd/scene_3.usd, 본 종류는 scene.usd/scene_1.usd다. 새 종류는 해당 YOLO와 이번 보정 학습에 없지만 과거 다른 연구 풀에는 있었다. 두 메시·12장면 그룹에 대한 합성 결과이며 독립 실사/임의 새 종류 일반화 신뢰구간이 아니다.
- 이동 오차는 각 시드의 PnP 이동 오차 중앙값을 평균낸 진단이다. 물리적 방향 의미가 확정된 회전 정확도·완전한6D 자세 성능은 이번에 입증하지 않았다. 규칙 재정렬은 좌표 집합을 바꾸지 않으므로 큰 ID 오차 감소가 그대로 물리 자세 개선을 뜻하지 않는다.

![최종 비교](../../../../data/pallet/results/pallet_role_selection_v2/figures/comparison.png)

{mdtable(seedheader,seedtable)}

유익/유해 순서 변경은 새 종류 검출 프레임에서 작은 보정 전 순서 선택만의 효과다. 새 종류에서 선택을 바꾼13~14개 중6~7개가 잘못된 방향이었다. 마지막 열은 본/새 종류 전체에서 기존10px이하가20px초과로 바뀐 수다. 평균이 좋아진 경우도 정상 프레임 보호가 보장되지 않는다.

![가림·잘림](../../../../data/pallet/results/pallet_role_selection_v2/figures/conditions.png)

## 실제로 바뀐 아키텍처

```mermaid
flowchart LR
 I[RGB] --> Y[기존 YOLO 동결]
 Y --> P[기존 점과 관측 특징]
 P --> H[객체 전체 코너 ID 후보4개]
 D[알려진 XYZ 치수] --> S[후보별 오차 예측 선택기]
 H --> S
 S --> C[한 후보 선택]
 C --> R[독립된 작은 좌표 보정]
 R --> O[최종 점과 동일 치수 PnP]
```

추가한 두 학습 모듈을 기존 YOLO 출력 경로에 연결했다. Backbone·검출 head는 동결했고, 전 네트워크 재학습은 아니다. 후보마다 점 좌표·점 특징·가시성 값을 같은 순서로 재배치하고 중심점8은 유지한다. 후보의 좌표를 평균내지 않고 하나를 선택하며, 선택 후 보정은 각 좌표±16px다. 초기에는 기존 순서와 원래 점을 정확히 보존한다.

네 후보는 **카메라 기준 코너 ID가 잘못 배정됐을 가능성**이며, 실제 팔레트를90도 돌려도 같다는 C4 대칭 가정이 아니다. 물리적인 전면 의미를 식별하도록 새로 정의한6D head도 아니다. 선택기와 보정기는 입력을 공유하지만 학습 파라미터를 공유하지 않는다. hard argmin을 통과하는 gradient 대신 각각의 감독을 사용한다. gradient surgery를 사용하지 않았으며, 두 분기의 직접적인 gradient 충돌이 이번 실패의 원인이라는 설명은 맞지 않는다.

최종25.48% 조건부 개선은 순서 선택 구조와 후보별 보정 감독을 함께 적용한 방법의 결과다. 최종 GT 대응 수정 단계에서 치수 제거군이나 선택만 학습하는 군을 다시 훈련하지 않았으므로, 이를 치수 입력 단독의 기여라고 부르지 않는다. 처음4군 대조와 최종 두 군 대조의 범위를 구분한다.

추론에는 RGB에서 얻은 점/특징과 알려진 치수만 들어간다. GT 박스·GT 점·정답 후보 번호를 넣지 않는다. 학습에서는 실제 GT를 이용해 후보 오차와 가장 좋은 후보의 보정량을 감독한다. 이것은 정답 감독 학습이며 pseudo-label selftraining이 아니다. Hough/선 분기는 이번 코너 대응 원인 실험에 포함하지 않았다.

## 확인한 두 실험 구성 문제와 수정

1. **손실과 목표의 불일치.** 최초 선택기는 평균적으로 낮은 오차 후보를 고르는 것이 목적이었으나 Huber 회귀는 조건부 평균 오차 추정과 같지 않다. 제곱오차로 바꾸고 좌표 보정 가중치는 정확히 고정해 대조했다. 학습의 큰 수정 가능 사례193개 중 실제 고친 수는7/14/14→36/27/25개로 늘었다. 그러나 새 장면에서는 시드마다 이득과 손해가 달랐다. 손실이 일부 원인이지만 단독 해법은 아니었다.
2. **학습 후보와 GT 객체 연결.** 처음에는 가장 높은 점수 후보 전부에 코너 감독을 줬다. 그중136/1536개는 GT 박스 IoU<.5였다. GT IoU≥.5인1400개에만 코너 감독을 주도록 수정하고, 직접 보정과 선택+보정을 동일 예산으로 다시 학습했다. 이 과정에서 confidence 기준은 쓰지 않았다. 평가의 검출 실패와 기존 GT는 그대로 유지했다. 최종 표가 이 수정을 반영한 비교다.

두 수정마다 다음 평가를 보기 전에 protocol을 저장하고 새 장면576개를 따로 만들었다. 아래 표는 **서로 다른 평가셋**이므로 행 사이 수치를 개선량으로 계산하면 안 된다.

{mdtable(['단계별 별도 평가','YOLO px','직접 보정 px','당시 선택+보정 px'],initialtable)}

총21개 체크포인트·25,200업데이트, 새 합성 관측1,728개를 생성했다. 첫4군×3시드 비교, 선택 손실만 바꾼3시드, GT 대상 연결을 수정한2군×3시드다. 한 데이터셋에 가중치·임계값을 반복 맞추는 탐색은 하지 않았다.

## 해결된 것과 남은 것

| 항목 | 현재 판단 |
|---|---|
| ±64px로 큰 ID 이동을 못 하던 표현 제한 | 순서 후보를 통해 큰 이동을 표현할 수 있게 수정·검사 |
| 선택과 보정의 직접 gradient 충돌 | 파라미터 분리, 선택만/B와 C의 보정 제거 출력이 정확히 같음 |
| 평균 오차 선택 목적과 Huber 회귀 차이 | 제곱오차 대조 완료; 학습 내 개선, 새 장면 안정성 미해결 |
| GT와 다른 객체 후보에 코너 감독 | 학습의 GT 박스 대응으로 수정; 원본 GT는 수정하지 않음 |
| 본 종류의 새 시점 대응 | 조건부 평균 개선은3시드 재현 |
| 새 종류에서의 후보 선택 신뢰성 | 유해 순서 변경과 시드 변동이 남음 |
| 검출 실패 | 동결 검출기 공유로 개선하지 못함 |
| 실제 새로운 팔레트·촬영 세션 | 데이터 미확보, 평가하지 않음 |

따라서 “두 모듈이 공존할 수 없다”는 결론은 아니다. **본 종류에서 쓰일 수 있는 조건부 효과는 있고, 새 종류의 올바른 순서 선택이 남은 병목이다.** 다만 어느 입력 특징이 부족한지, 학습량/구조 중 무엇이 지배적인지는 이번 결과만으로 단정할 수 없다. 단순히 gradient 수술이나 보정 cap 확대로 해결된다고도 말할 수 없다.

## 이미지와 다음 판단

![본 종류에서 개선](../../../../data/pallet/results/pallet_role_selection_v2/figures/seen_gain.jpg)
![새 종류에서 악화](../../../../data/pallet/results/pallet_role_selection_v2/figures/unseen_harm.jpg)

그림은 모두 실제 모델 추론이며 초록GT·분홍예측이다. 시드1의 개선/악화 예를 명시적으로 선택해 설명했고 체크포인트 선정에는 쓰지 않았다.

이번 구조의 사전 안정성 기준은 실패로 종료한다. 같은 후보 점수·전환 임계값 탐색을 추가하지 않는다. 다음 설계 판단은 camera-facing 면적 기준 ID와 고정 물리 역할을 구분하고, 새 종류에서도 관측 가능한 역할 증거를 학습하는 데 초점을 둬야 한다. 현재의 단순 cuboid ID 재배치만으로 팔레트 전용 논문 신규성과 안정적인 실제6D 개선을 주장하기는 어렵다. 그 이후의 새 구조·새 실사 학습은 이번 완료 범위에 포함하지 않는다.

## 재현

`FINAL_AUDIT.json`에서 새 장면 중복 없음, 동일 검출 평가 모집단, 모델 출력 재현, 지표 독립 재계산, 작은 보정 범위, GT 재투영과 이전 결과 보존을 검사했다. GPU는 기존 프로세스 전용 라이브러리 경로를 사용했고 재부팅하지 않았다. 실행시간은1차 `SUMMARY.json`의 CPU2스레드 보정 분기 진단이며 최종 전체 네트워크 지연을 측정한 것은 아니다.
'''
    NOTE.write_text(text)
    title='코너 순서 선택과 작은 보정 — 검증 결과'
    content=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>{title}</title><style>
    body{{margin:0;background:#0b1220;color:#dce5f3;font:16px/1.7 system-ui}}main{{max-width:1220px;margin:auto;padding:38px 24px}}h1{{font-size:34px;line-height:1.3}}h2{{margin-top:38px;color:white}}.card{{padding:24px;background:#18253b;border:1px solid #334155;border-radius:14px;margin:20px 0}}.tag{{color:#7dd3fc;font-weight:bold}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:11px;text-align:right;border-bottom:1px solid #334155}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;border-radius:10px;margin:12px 0}}a{{color:#7dd3fc}}select{{background:#26364c;color:white;padding:12px;border-radius:8px;border:1px solid #64748b}}.small{{color:#afbed1;font-size:14px}}.flow{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}.flow span{{padding:12px;background:#263a51;border-radius:8px}}</style>
    <main><p class="tag">구조 구현 · 원인 대조 · 3회 새 장면 평가 · 독립 재현 검사 통과</p><h1>{title}</h1>
    <div class="card"><strong>본 종류의 새 구도에서는 효과가 있지만, 새 종류의 안정적 개선은 아직 아닙니다.</strong><p>본 종류:23.92→17.82px (평균25.48% 감소). 새 종류:25.14→24.40px (평균2.96% 감소), 직접 점 보정23.74px보다 나쁩니다. 새로운 순서를 잘못 고르는 사례가 남았습니다.</p></div>
    <h2>최종 대조: 같은 새 평가셋</h2><p>본 종류{seen['matched']}/288, 새 종류{unseen['matched']}/288검출·매칭. 평균·P90은 이 공통 프레임 기준이며, 주 지표는 검출 실패도 포함합니다.</p>{htable(header,table)}
    <p class="small">±는3시드 표준편차. 새 종류 주 지표에서는1시드 개선·2시드 악화입니다. 이동 오차는 PnP 진단이며 의미가 확정된 회전·실사6D 성능은 검증하지 않았습니다.</p><img src="figures/comparison.png">
    <h2>세 시드에서 어떻게 달랐나</h2>{htable(seedheader,seedtable)}<p class="small">유익/유해는 새 종류의 순서 변경만 평가. 마지막 열은 전체 종류에서 원래10px이하가20px초과로 바뀐 프레임 수입니다.</p><img src="figures/conditions.png">
    <h2>실제 구조</h2><div class="flow"><span>동결 YOLO 점·특징</span>→<span>코너 ID 후보4개</span>→<span>치수와 후보 오차로1개 선택</span>→<span>독립된±16px 보정</span></div>
    <p>후보 전체를 평균내지 않습니다. 추론에 GT가 들어가지 않으며, 선택과 보정은 학습 파라미터를 공유하지 않습니다. 이는 추가 모듈을 연결한 구조이고 전체 YOLO 재학습은 아닙니다. Hough나 물리적 C4 대칭을 가정한 실험도 아닙니다.</p>
    <h2>초기 구성에서 고친 두 문제</h2><div class="card"><p><b>오차 예측 손실:</b> Huber를 평균 오차 목적에 맞는 제곱오차로 대조했습니다. 학습 내 큰 오류 수정은 늘었지만 새 장면 안정성은 해결되지 않았습니다.</p><p><b>학습 대상 연결:</b> GT 박스와 맞지 않는136개 후보를 코너 감독 대상으로 삼았던 구성을 수정했습니다. GT에 대응하는1400개로 학습하고 평가 실패는 모두 유지했습니다.</p></div>
    <h2>성공·실패 사례</h2><p>모두 실제 시드1 모델 출력입니다. 초록은GT, 분홍은예측이며 원인 진단용 정답 선택 oracle은 이 그림에 없습니다.</p><select id="case"></select><img id="caseimg"><p id="caseid" class="small"></p>
    <h2>범위와 판단</h2><p>총21개 체크포인트와 새 합성 관측1,728개를 확인했습니다. 본 종류 조건부 평균 개선은 재현됐지만, 새 종류와 정상 프레임 보호를 포함한 안정성 기준은 실패했습니다. 같은 임계값 탐색은 여기서 종료합니다. 새 실제 팔레트·촬영 세션 평가와 물리 역할 감독은 아직 남아 있습니다.</p>
    <p><a href="{NOTE.as_uri()}">전체 분석 MD</a> · <a href="foreground_matched/SUMMARY.json">최종 수치</a> · <a href="foreground_matched/PROTOCOL.json">최종 감독 규칙</a> · <a href="FINAL_AUDIT.json">재현 검사</a></p>
    <script>const cases={json.dumps(gallery)};const sel=document.getElementById('case');cases.forEach((c,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=c.label;sel.appendChild(o)}});function show(){{let c=cases[sel.value];document.getElementById('caseimg').src=c.src;document.getElementById('caseid').textContent=c.id}}sel.onchange=show;show();</script></main></html>'''
    (OUT/'index.html').write_text(content)
    write(OUT/'REPORT_COMPLETE.json',dict(title=title,note=str(NOTE.relative_to(ROOT)),table_cells=sum(table,[]),gallery=gallery,summary_sha256=sha(folder/'SUMMARY.json')))
    print('Report and note written')
if __name__=='__main__':main()
