from common import *
import html,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

LABELS={'stock':'기존 YOLO','A_points':'A 점 보정','B_concat':'B 단순 치수 결합','C_geometry':'C 기하 관계','D_no_relations':'D 관계 제거'}
ENGLISH=['YOLO','Point residual','Dimensions concat','Geometry relations','Relations removed']
NOTE=ROOT/'_docs/notes/pose_geometry/unseen-dimensions-screen.md'

def main():
    assert read(OUT/'AUDIT.json')['PASS']
    s=read(OUT/'SUMMARY.json');cause=read(OUT/'CAUSE_DIAGNOSTIC.json');data=read(OUT/'DATA_AUDIT.json')
    cfg=read(OUT/'PROTOCOL.json');pred=np.load(OUT/'predictions.npz');records=read(OUT/'test_records.json')
    canvas=Image.new('RGB',(1280,1000),'white');draw=ImageDraw.Draw(canvas)
    for ai,asset in enumerate(ASSETS):
        x=(ai%2)*640;y=(ai//2)*500
        canvas.paste(Image.open(OUT/'images'/f'a{ai}_test_g000_v1_clean.png'),(x,y+20))
        draw.text((x+12,y+4),asset+(' (seen)' if ai<2 else ' (unseen)'),fill='black')
    canvas.save(OUT/'mesh_preview.jpg')
    frames=read(OUT/'PER_FRAME.json');audit=read(OUT/'AUDIT.json')
    methods=['stock']+ARMS;table=[];figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def vals(method,pop,field,condition=None):
        tags=['stock'] if method=='stock' else [f'{method}_s{i}' for i in [1,2,3]]
        return np.array([s['results'][t][pop]['conditions'][condition][field] if condition else s['results'][t][pop]['overall'][field] for t in tags])
    def fmt(v):return f'{v.mean():.3f}'+(f' ± {v.std(ddof=1):.3f}' if len(v)>1 else '')
    for m in methods:
        table.append([LABELS[m],fmt(vals(m,'seen','matched_mean_px')),fmt(vals(m,'unseen','matched_mean_px')),
                      f'{vals(m,"unseen","primary").mean():.6f}',f'{vals(m,"unseen","matched_p90_px").mean():.2f}',
                      f'{vals(m,"unseen","translation_median_cm").mean():.3f}'])
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,pop in zip(axes,['seen','unseen']):
        v=[vals(m,pop,'matched_mean_px') for m in methods]
        ax.bar(np.arange(5),[x.mean() for x in v],yerr=[x.std(ddof=1) if len(x)>1 else 0 for x in v],capsize=4,color=['#64748b','#94a3b8','#06b6d4','#f97316','#a78bfa'])
        ax.set_xticks(range(5),ENGLISH,rotation=22,ha='right');ax.set_ylabel('Matched-frame mean corner error (px)')
        ax.set_title(pop.upper()+' meshes / same renderer');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'architecture_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(3)
    for i,m in enumerate(methods):
        y=[vals(m,'unseen','matched_mean_px',c).mean() for c in cfg['conditions']]
        ax.bar(x+(i-2)*.15,y,.15,label=ENGLISH[i])
    ax.set_xticks(x,['Clean','Physical occlusion','25% image crop']);ax.set_ylabel('Matched-frame mean corner error (px)')
    ax.legend(ncol=3,fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figdir/'conditions.png',dpi=160);plt.close(fig)
    # Cases explain ID failure and ordinary successes; not used to select checkpoints.
    stockrows=cause['rows']['stock'];valid=[i for i,r in enumerate(records) if not r['seen'] and frames['stock'][i]['matched']]
    oracle_case=max(valid,key=lambda i:stockrows[i]['original_px']-stockrows[i]['four_role_GT_oracle_px'])
    cg=pred['C_geometry_s1'];sg=pred['stock'];gt=np.array([r['gt'] for r in records])
    e0=np.linalg.norm(sg[:,:8]-gt,axis=-1).mean(-1);e1=np.linalg.norm(cg[:,:8]-gt,axis=-1).mean(-1)
    cases=[('role_error','Corner ID error',oracle_case),('improved','Geometry residual improves',max(valid,key=lambda i:e0[i]-e1[i])),
           ('worsened','Geometry residual worsens',max(valid,key=lambda i:e1[i]-e0[i])),
           ('ordinary','Ordinary low error',min(valid,key=lambda i:abs(e0[i]-5.37)))]
    signs=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])
    perms=[]
    for a in [0,90,180,270]:
        c=np.cos(np.deg2rad(a));sn=np.sin(np.deg2rad(a));rot=np.array([[c,0,sn],[0,1,0],[-sn,0,c]])
        perms.append(np.linalg.norm((signs@rot.T)[:,None]-signs[None],axis=-1).argmin(-1))
    gallery=[]
    for name,label,i in cases:
        im=Image.open(ROOT/records[i]['image']);fig,axes=plt.subplots(1,3,figsize=(16,4.7))
        branch=stockrows[i]['four_role_GT_oracle_branch'];oracle=sg[i,:8][np.argsort(perms[branch])]
        for ax,p,title in zip(axes,[sg[i,:8],cg[i,:8],oracle],['YOLO','Geometry / seed 1','GT role oracle / NOT deployable']):
            ax.imshow(im);ax.scatter(gt[i,:,0],gt[i,:,1],s=65,facecolors='none',edgecolors='#00ff99',linewidths=2)
            ax.scatter(p[:,0],p[:,1],s=22,c='#ff4388')
            for j in range(8):
                ax.text(gt[i,j,0]-9,gt[i,j,1]-7,str(j),color='#00ff99',fontsize=10,weight='bold',bbox=dict(facecolor='black',alpha=.5,pad=.2))
                ax.text(p[j,0]+5,p[j,1]+7,str(j),color='#ff4388',fontsize=10,weight='bold',bbox=dict(facecolor='black',alpha=.5,pad=.2))
            ax.set_title(f'{title}\n{np.linalg.norm(p-gt[i],axis=-1).mean():.2f}px');ax.axis('off')
        fig.suptitle(f'{label} | {records[i]["id"]} | Green: GT, pink: prediction',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(figdir/(name+'.jpg'),dpi=130);plt.close(fig)
        gallery.append(dict(label=label,id=records[i]['id'],src='figures/'+name+'.jpg'))
    d=cause['summary']['stock']['unseen'];cn=s['results']['stock']['unseen']['overall'];seen=s['results']['stock']['seen']['overall']
    conditiontable=[]
    for m in methods:conditiontable.append([LABELS[m]]+[f'{vals(m,"unseen","matched_mean_px",c).mean():.3f}' for c in cfg['conditions']])
    sensitivity=[]
    for arm in ARMS[1:]:
        v=[]
        for suffix in ['', '_shuffle', '_noise5']:
            v.append(f'{np.mean([s["results"][f"{arm}_s{i}"+suffix]["unseen"]["overall"]["matched_mean_px"] for i in [1,2,3]]):.3f}')
        latency=np.mean([s['latency'][f'{arm}_s{i}']['cpu_branch_batch1_p50_ms'] for i in [1,2,3]])
        sensitivity.append([LABELS[arm]]+v+[f'{latency:.3f}'])
    def mdtable(header,body):return '| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in body)
    headers=['방법','본 자산 평균 px','새 자산 평균 px','새 자산 주 지표↓','새 자산 P90 px','이동 오차 중앙값 cm']
    text=f'''# 새 팔레트 치수·기하 구조 검증 — 2026-09-11

**결론: 이번 기하 관계 보정 구조의 안정적인 추가 이득은 확인되지 않았다.** 단순 치수 결합과 기하 관계 구조 모두 기존 YOLO를 일관되게 넘지 못했다. 이는 고정 YOLO 뒤의 이번 보정 모듈·학습 예산에 대한 결과이며, 치수 입력 아키텍처 전체의 불가능성 판정은 아니다.

[대화형 HTML 보고서](../../../../data/pallet/results/pallet_unseen_dimensions_v1/index.html) · [수치 원본](../../../../data/pallet/results/pallet_unseen_dimensions_v1/SUMMARY.json) · [실행 코드](../../../../scripts/research/pallet_unseen_dimensions_v1/README.md)

## 무엇을 새로 검증했나

기존 YOLO가 학습한 자산은 scene.usd, scene_1.usd, EUR GLB, block GLB 네 종류다. P0/TEX는 기록상 모두 scene.usd다. 이번 학습에는 앞의 두 USD만 사용하고, **해당 YOLO가 보지 않은 scene_2.usd와 scene_3.usd**를 평가했다. 이 두 자산은 다른 과거 연구 풀에는 존재하므로, 프로젝트 전체에서 미개봉인 논문 FINAL 데이터라고 부르지 않는다.

같은 렌더러·카메라·크기·조명 분포로 본 자산과 새 자산을 함께 평가했다. 이미지 2,304개: 학습1,536 / 본 자산 검증192 / 평가576(본 자산288 + 새 자산288). 데이터와 조건을 결과 전에 동결했다. 검증192개로 체크포인트를 고르지 않고 최종 업데이트를 사용했다. 그룹별 크기 변형은 다양화이며 새 구조로 세지 않았다.

![실제 네 메시](../../../../data/pallet/results/pallet_unseen_dimensions_v1/mesh_preview.jpg)

## 구조와 비교의 범위

```mermaid
flowchart LR
 I[RGB] --> Y[동결된 기존 YOLO]
 Y --> P[기존 9개 점]
 Y --> F[점 위치 특징 + ROI 7x7 특징]
 D[알려진 고정 XYZ 치수] --> H[폭·깊이 두 배치 가설]
 H --> G[코너 역할과 상대 3D 관계]
 F --> A[영상 attention과 코너 간 attention]
 G --> A
 A --> R[초기값 0인 좌표 보정]
 P --> O[보정된 점]
 R --> O
 O --> S[동일 치수와 K로 PnP 진단]
```

A는 점·영상 특징만, B는 동일 구조에 단순 치수 결합, C는 코너 역할과 상대 기하를 attention에 추가, D는 C에서 상대 관계를 제거한 대조군이다. 각 모델57,560개 파라미터를 할당했지만 입력을 끈 분기의 실효 자유도까지 같다고 주장하지 않는다. 출력이0으로 시작해 모든 군이 초기 YOLO 점을 정확히 보존했다. 이후 동일한 초기 상태·배치 순서로4군×3시드×1,200업데이트=14,400업데이트를 수행했다. 각 좌표 보정은±64px로 제한했다.

**실제 학습한 것은 추가 보정 모듈이다. YOLO backbone·검출 head 전체 재학습, Hough 분기, 물리적 면 분류 감독은 이번 실험에 없다.** C의 관계 분기 gradient와 초기 보존 검사를 통과했다. 이 cuboid 관계 자체가 팔레트 논문 신규성을 뜻하지는 않는다.

## 비교 결과

아래 평균·P90은 동일하게 검출·매칭된 프레임의8코너 오차다. 평균의±는3시드 표준편차이며 신뢰구간이 아니다. 주 지표는 **검출 실패를 이미지 대각선 오차로 포함**한 전체 프레임 정규화 오차다. 실패를 표에서 숨기지 않는다.

{mdtable(headers,table)}

- 본 자산 검출·매칭: **{seen['matched']}/{seen['n']}**, 새 자산: **{cn['matched']}/{cn['n']}**. 동결 검출기를 공유하므로 모든 군이 같다.
- 새 자산 기존 YOLO의 점 오차 중앙값은 **{np.median(e0[[i for i in valid]]):.3f}px**, 평균은 **{d['original_px']:.3f}px**다. 큰 코너 대응 오류가 평균을 끌어올린다.
- C 대 B 주 지표 개선률: **{', '.join(f'{x*100:+.4f}%' for x in s['C_over_B_gain_per_seed'])}**. 사전 기준인 모든 시드2% 개선에 미달했다.
- C−B 정규화 오차 차이의 장면 그룹 bootstrap95%구간: **[{s['C_minus_B_bootstrap']['low']:.7f}, {s['C_minus_B_bootstrap']['high']:.7f}]**.0을 포함한다.12개 장면 그룹에 대한 구간이며 두 메시 밖의 자산 모집단 일반화 신뢰구간은 아니다.
- 이동 오차는 각 시드의 중앙값을 평균낸 진단이다. GT를 넣지 않은 두 cuboid 대응 가설 PnP에 동일한 올바른 치수를 사용했다. **물리적 의미가 확정된 회전 정확도와 완전한6D 성능은 입증하지 않았다.**
- 이 표는 새 렌더링 평가셋이다. 이전 실사5px·8px 또는 합성5.05px 표와 숫자만 직접 비교하지 않는다.

![아키텍처 비교](../../../../data/pallet/results/pallet_unseen_dimensions_v1/figures/architecture_comparison.png)

### 새 자산 조건별 점 오차

{mdtable(['방법','깨끗함 px','가림판 px','이미지25% 잘림 px'],conditiontable)}

가림판은 평가에서 물체 픽셀 평균31.1%를 가렸다. 잘림192개 중 실제 화면 밖 코너가 있는 프레임은123개다. 모든 crop 프레임이 실제 팔레트를 자른다고 분류하지 않는다.

![조건별 비교](../../../../data/pallet/results/pallet_unseen_dimensions_v1/figures/conditions.png)

### 치수 민감도와 보정 모듈 실행시간

{mdtable(['방법','올바른 치수 px','치수 섞기 px','치수5% 잡음 px','CPU 보정 p50 ms'],sensitivity)}

신경망 입력 치수만 다른 장면 그룹의 치수로 바꾸거나 독립 로그정규5% 잡음을 넣었다. PnP에는 계속 올바른 치수를 사용해 신경망 효과와 solver의 스케일 오류를 섞지 않았다. 실행시간은 CPU2스레드·batch1·고정 특징 입력의 모듈만이며, GPU 전체 YOLO 지연이나 실시간 배포 성능이 아니다. 표의 지연은 시드별 p50의 평균이다.

## 안 된 이유: 확인과 추정을 구분

1. **검출 실패는 보정 모듈로 해결되지 않는다.** 새 자산91/288프레임이 검출·매칭 기준에 실패했다. 본 자산도81/288프레임이 실패해, 새 메시 효과뿐 아니라 렌더링·시점 난도가 함께 있다. 따라서 현재 화면에서 치수 head만 바꿔 전체 성능을 크게 개선하는 데 구조적 한계가 있다.
2. **큰 오류에는 코너 ID 대응 문제가 있다.** 새 자산197검출 프레임 중28개는 네 가지 수평 코너 역할 배치 중 정답을 보고 고른 배치가 원래 배치보다 좋았다. 그 oracle 평균은 **{d['four_role_GT_oracle_px']:.3f}px**다. 완전히 자유로운 점 매칭 oracle은 **{d['unconstrained_assignment_GT_oracle_px']:.3f}px**다. 둘 다 실제 모델 성능이 아니고,90도 재배치를 물체의 물리적 C4 대칭으로 인정한 것도 아니다.
3. **작은 보정 경로로 큰 ID 이동을 맡긴 한계가 확인됐다.** 정답 보정 방향을 직접 주어도 각 좌표±64px 범위에서 가능한 최저 평균은 **{d['capped_residual_GT_oracle_px']:.3f}px**다. 이는 표현 범위의 하한 진단이지 더 좋은 학습 결과가 아니다. cap을 키우면 자동 해결된다는 증거도 아니다.
4. **기존 GT의 좌표 계산은 맞지만 ID 선택은 경계에서 민감하다.** 모든 GT가 기존 camera-facing-v4 규칙의 재정렬에서 자기 자신으로 돌아왔고,3D 재투영·GT PnP 이동 재현도 통과했다. 반면 GT에 독립1px 잡음을100회 더한 진단에서는 새 자산 검출 프레임에서 평균 **{100*d['GT_1px_jitter_relabel_rate']:.2f}%**가 다른 ID 순서로 재정렬됐다. 이는 기록된 GT가 틀렸다는 뜻이 아니라, 면의 투영 면적에 따른 ID 정의가 작은 변화에 불연속적으로 바뀔 수 있다는 뜻이다.
5. **단순 재정렬도 완전한 해법은 아니다.** 예측점만 기존 규칙으로 재정렬하면 평균 **{d['original_px']:.3f}→{d['observation_only_canonical_px']:.3f}px**로 좋아졌지만, 원래10px이하였던2개 프레임이20px초과로 나빠졌다. 사후 후처리 진단이며 아키텍처 기여로 세지 않는다.

해석: 기하 attention에 학습 신호가 전달되지 않는 것이 주된 원인이라는 증거는 없다. 현재 근거는 **검출·코너 역할 선택·미세 좌표 보정을 구분해야 한다**는 쪽이다. 다만 oracle만으로 학습 가능한 역할 판별기가 존재한다고 증명되지는 않는다. gradient surgery는 이산 ID 변경이나 보정 표현 범위를 직접 바꾸지 않는다.

![코너 ID 실패의 예](../../../../data/pallet/results/pallet_unseen_dimensions_v1/figures/role_error.jpg)

## 다음 판단과 논문 기여

이번 C의 관계 가중치·보정 cap을 같은 평가셋에 맞춰 다시 탐색하는 작업은 진행하지 않았다. 이 결과를 논문에서 안정적인 개선의 증거로 사용하기 어렵다. 현재의 현실적인 다음 구조는 기존 점 좌표를 보존하면서 **코너 대응 가설을 별도로 선택·유지하고, 선택 후 작은 잔차를 보정하는 방식**이다. camera-facing 면적 경계의 GT 역할 변화를 먼저 정의해야 하며, W/D/H만으로 관측 불가능한 면까지 결정할 수 있다고 가정하면 안 된다. 이번에 그 구조를 새로 학습한 것은 아니다.

논문용으로 확대하기 전에는 (1) 고정 물리 역할과 카메라 기준 ID를 분리한 감독 계약, (2) 기존 단순 치수 분류/결합 대비 역할 선택 효과, (3) 충분한 검출 커버리지와 학습에서 제외한 실제 새 팔레트·촬영 세션을 확보해야 한다. 새 실사 경로와 실측 치수는 사용자에게 확인 중이며 현재 실사 독립 평가는 수행하지 않았다. 추가 분석 몇 번이면 논문 기여가 확정된다고 말할 근거는 없다.

## 재현과 검증

`PROTOCOL.json`, `TRAIN_LOCK.json`, `DATA_AUDIT.json`, `PREFLIGHT.json`, `POSE_GT_AUDIT.json`, `AUDIT.json`을 저장했다. 데이터 분리·동일 초기화·모델 입력에 GT 없음·실제 공개 YOLO 출력과 캐시 일치·점 예측 재현·지표 독립 재계산을 확인했다. GPU는 프로세스 전용 NVIDIA580.173.02 경로로 사용했으며 재부팅하지 않았다. 실행시간 표는 CPU2스레드 보정 모듈만 측정했고 전체 카메라→결과 지연으로 주장하지 않는다.
'''
    NOTE.write_text(text)
    def htmltable(header,body):return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in header)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in body)+'</tbody></table>'
    title='새 팔레트 치수·기하 구조 검증'
    content=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>{title}</title><style>
    body{{margin:0;background:#0b1220;color:#dce5f3;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:40px 26px}}h1{{font-size:34px;line-height:1.25}}h2{{margin-top:38px;color:#fff}}p{{max-width:1050px}}.card{{padding:22px;border:1px solid #334155;border-radius:14px;background:#162237;margin:18px 0}}.tag{{color:#fb923c;font-weight:700}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px;text-align:right;border-bottom:1px solid #334155}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;border-radius:10px;margin-top:12px}}a{{color:#7dd3fc}}select{{padding:12px;background:#26364c;color:white;border:1px solid #64748b;border-radius:8px;max-width:100%}}.small{{font-size:14px;color:#adbed5}}code{{color:#a5f3fc}}</style>
    <main><p class="tag">4개 구조 × 3시드 · 학습에서 제외한 두 메시 · 재현 검사 통과</p><h1>{title}</h1>
    <div class="card"><strong>안정적인 기하 구조 추가 이득은 확인되지 않았습니다.</strong><p>큰 코너 ID 오류와 검출 실패가 남았습니다. 이번 결과는 동결 YOLO 뒤의 보정 모듈에 대한 검증이며, 치수 기반 아키텍처 전체의 불가능성 판정은 아닙니다.</p></div>
    <h2>같은 평가셋에서 비교</h2><p>새 자산288프레임 중197개 검출·매칭. 본 자산288프레임 중207개 검출·매칭. 평균과 P90은 이 공통 프레임의 점 오차입니다. 주 지표는 검출 실패도 포함합니다.</p>
    {htmltable(headers,table)}<p class="small">±는3시드 표준편차. 이동 오차는 cuboid 대응 PnP 진단이며 의미가 확정된 회전 정확도는 검증하지 않았습니다. 이전 실사·합성 평가와 직접 수치 비교하지 않습니다.</p>
    <img src="figures/architecture_comparison.png"><h2>가림과 잘림</h2>{htmltable(['방법','깨끗함 px','가림판 px','이미지25% 잘림 px'],conditiontable)}<img src="figures/conditions.png">
    <h2>무엇이 막고 있나</h2><div class="card"><p><b>코너 대응:</b> 새 자산 평균30.156px → 정답으로 네 역할 중 선택한 oracle8.684px. 이 차이는 진단용이며 실제 성능 개선이 아닙니다.</p><p><b>보정 범위:</b> 정답 방향을 알아도±64px 보정의 최저 평균은14.787px. 작은 좌표 보정으로 큰 ID 이동을 다루는 한계가 있습니다.</p><p><b>단순 재정렬:</b> 예측점만 사용한 후처리는24.173px로 개선됐지만, 원래 좋던2개 프레임이20px초과로 악화됐습니다. 새로운 학습 아키텍처로 세지 않습니다.</p></div>
    <h2>실제 이미지와 점 ID</h2><p>초록은 GT, 분홍은 예측입니다. 오른쪽은 정답으로 순서를 고른 진단용 oracle이며 배포 가능한 결과가 아닙니다. Hough를 학습하거나 표시한 실험은 아닙니다.</p>
    <select id="case"></select><img id="caseimg"><p id="caseid" class="small"></p>
    <h2>새로 사용한 메시</h2><img src="mesh_preview.jpg"><p>위 두 자산으로 보정 모듈을 학습하고, 아래 두 자산을 학습에서 제외했습니다. 아래 자산은 현재 YOLO에도 없지만 다른 과거 연구 풀에는 있었습니다. 실제 새 팔레트 촬영 검증은 아닙니다.</p>
    <h2>다음 판단</h2><p>이 보정 모듈의 가중치 탐색을 반복하지 않습니다. 코너 역할 가설과 미세 좌표 보정을 분리하는 감독·구조가 다음 후보입니다. 역할 선택이 관측 가능한지 검증하고 독립 새 실사를 확보해야 논문 기여를 판단할 수 있습니다.</p>
    <p><a href="{NOTE.as_uri()}">전체 분석 MD</a> · <a href="SUMMARY.json">수치 원본</a> · <a href="CAUSE_DIAGNOSTIC.json">원인 진단</a> · <a href="AUDIT.json">독립 재현 검사</a> · <a href="PROTOCOL.json">사전 규칙</a></p>
    <script>const cases={json.dumps(gallery)};const sel=document.getElementById('case');cases.forEach((c,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=c.label;sel.appendChild(o)}});function show(){{let c=cases[sel.value];document.getElementById('caseimg').src=c.src;document.getElementById('caseid').textContent=c.id}}sel.onchange=show;show();</script></main></html>'''
    (OUT/'index.html').write_text(content)
    write(OUT/'REPORT_COMPLETE.json',dict(title=title,note=str(NOTE.relative_to(ROOT)),table_cells=sum(table,[]),
        summary_sha256=sha(OUT/'SUMMARY.json'),gallery=gallery))
    print('Created',NOTE,OUT/'index.html')
if __name__=='__main__':main()
