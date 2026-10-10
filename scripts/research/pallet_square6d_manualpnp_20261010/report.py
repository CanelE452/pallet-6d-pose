"""Korean report, complete manuscript table and numeric PNGs; no RGB/model/F."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import common as C
NAMES={'yolo':'YOLO','dope':'DOPE','resnet18':'ResNet-18'}
LABELS={'BASE':'Base','N3_DIM_SYM':'N3','SUBPIX':'SubPix','N3_THEN_SUBPIX':'N3→SubPix'}
def f(v):return '—' if v is None else f'{v:.4f}'
def mean_sd(x,scale=1):return f"{f(x['mean']*scale) if x['mean'] is not None else '—'} ± {f(x['std']*scale) if x['std'] is not None else '—'}"
def ci(v,scale=1):return '—' if v is None else '['+', '.join(f(x*scale) for x in v)+']'
def main():
 assert not (C.DOC/'REPORT_KO.md').exists()
 m=C.read(C.DOC/'METRICS.json');p=C.read(C.DOC/'PAIRED.json');r=C.read(C.DOC/'SQUARE_REFERENCE_POSES.json');ex=C.read(C.DOC/'EXECUTION.json')
 figdir=C.DOC/'figures';figdir.mkdir(exist_ok=True);figs=[]
 def save(fig,name):
  fig.tight_layout();fig.savefig(figdir/name,dpi=180,bbox_inches='tight');plt.close(fig);figs.append(dict(path='figures/'+name,sha256=C.sha(figdir/name)))
 refs=[x for x in r['frames'] if x['eligible']];good=[x for x in refs if x['available']]
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 counts={i:sum(x['manual_count']==i for x in refs) for i in (4,5,6,7)}
 axes[0].bar([str(i) for i in counts],list(counts.values()),color='#4378a1');axes[0].set(xlabel='In-frame directly manual corners',ylabel='Frames',title='Eligible118: 4/5/6/7 manual corners')
 for j,(key,n) in enumerate(counts.items()):axes[0].text(j,n+.7,str(n),ha='center')
 values=sorted(x['residual_mean_px'] for x in good);axes[1].plot(np.arange(1,len(values)+1),values,'.',color='#4378a1')
 axes[1].axhline(3,color='#a14f43',ls='--',label='median gate3px');axes[1].axhline(5,color='#a18243',ls=':',label='per-frame sensitivity5px')
 axes[1].set(xlabel='Sorted frame rank (all118 retained)',ylabel='Mean manual reprojection (px)',title=f"Median{r['median_mean_residual_px']:.3f}px; >5px:1/118")
 axes[1].annotate('026399:21.001px',(118,values[-1]),xytext=(64,17),arrowprops={'arrowstyle':'->'});axes[1].legend(fontsize=8)
 save(fig,'01_reference_residuals.png')
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
 for ax,b in zip(axes,C.BACKBONES):
  packet=m['backbones'][b]['ALL118'];rate=[packet['seed_mean'][a]['success_rate']['rate']*100 for a in C.METHODS]
  lows=[packet['seed_mean'][a]['success_rate']['CI95'][0]*100 for a in C.METHODS];highs=[packet['seed_mean'][a]['success_rate']['CI95'][1]*100 for a in C.METHODS]
  x=np.arange(4);ax.bar(x,rate,color=['#76838c','#579b82','#d6ad59','#477fa8'],alpha=.85)
  ax.errorbar(x,rate,yerr=[np.array(rate)-lows,np.array(highs)-rate],fmt='none',ecolor='black',capsize=4,label='Descriptive frame95%CI')
  for seed in C.SEEDS:ax.plot(x,[packet['by_seed'][str(seed)][a]['success_rate']['rate']*100 for a in C.METHODS],'.-',alpha=.5,lw=.8,label=f'Seed{seed}')
  ax.set_xticks(x,[LABELS[a] for a in C.METHODS],rotation=20);ax.set_title(f"{NAMES[b]} available{packet['seed_mean']['BASE']['available']}/118")
  ax.set_ylim(0,42);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('5cm / 5deg success (% of ALL118)');axes[2].legend(fontsize=7)
 fig.suptitle('Single-session manual-PnP reference: descriptive only; all seeds shown',fontsize=11)
 save(fig,'02_success_all_backbones.png')
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for col,b in enumerate(C.BACKBONES):
  for row,(key,label) in enumerate([('T_cm','Centroid error (cm)'),('R_deg','Proper C4 rotation error (deg)')]):
   aa=[m['backbones'][b]['ALL118']['seed_mean'][a]['metrics'][key] for a in C.METHODS];x=np.arange(4)
   axes[row,col].bar(x,[a['mean'] for a in aa],yerr=[a['std'] for a in aa],capsize=4,color=['#76838c','#579b82','#d6ad59','#477fa8'])
   axes[row,col].set_xticks(x,[LABELS[a] for a in C.METHODS],rotation=20);axes[row,col].set_title(NAMES[b]);axes[row,col].grid(axis='y',alpha=.2)
   if col==0:axes[row,col].set_ylabel(label+'; mean ± sampleSD')
 save(fig,'03_error_means_sd.png')
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
 scopes=[('ALL118','All118'),('reference_residual_mean_le5','Residual≤5 (117)'),('manual4','Manual4 (30)'),('manual_ge5','Manual≥5 (88)')]
 for ax,b in zip(axes,C.BACKBONES):
  x=np.arange(4)
  for offset,a in enumerate(C.METHODS):
   y=[m['backbones'][b][s]['seed_mean'][a]['success_rate']['rate']*100 for s,_ in scopes]
   ax.plot(x,y,'o-',label=LABELS[a],lw=1)
  ax.set_xticks(x,[label for _,label in scopes],rotation=25,ha='right');ax.set_title(NAMES[b]);ax.grid(alpha=.2)
 axes[0].set_ylabel('Success (%; group denominator fixed)');axes[-1].legend(fontsize=8)
 save(fig,'04_fixed_sensitivity_groups.png')
 table=['# GREEN0918 수동 PnP 참조 기준 원고용 표','',
 '[확인] 단일 세션의 118장, 같은 영상에서 seed별 오차를 평균한 기술통계이다. T/R/ADD/IoU는 자세 산출 영상만의 n을 명시하고 성공률은 항상 해당 범위의 전체 분모를 유지한다. 직사각형 결과와 합산하거나 직접 순위를 비교하지 않는다.','',
 '| 기반 | 경로 | 자세 n/118 | T cm 평균±SD | R° 평균±SD | ADDsym cm 평균±SD | IoU3D 평균±SD | 5cm·5° % |',
 '| --- | --- | --- | --- | --- | --- | --- | --- |']
 csvrows=[]
 for b in C.BACKBONES:
  for a in C.METHODS:
   v=m['backbones'][b]['ALL118']['seed_mean'][a];n=v['available'];s=v['metrics']
   table.append(f"| {NAMES[b]} | {LABELS[a]} | {n}/118 | {mean_sd(s['T_cm'])} | {mean_sd(s['R_deg'])} | {mean_sd(s['ADDsym_m'],100)} | {mean_sd(s['IoU3D'])} | {f(v['success_rate']['rate']*100)} |")
   csvrows.append([b,a,n,118,*[s[k][j] for k in ('T_cm','R_deg','ADDsym_m','IoU3D') for j in ('mean','std','median','P90')],v['success_rate']['rate']])
 (C.DOC/'PAPER_TABLE_KO.md').write_text('\n'.join(table)+'\n')
 with (C.DOC/'PAPER_TABLE.csv').open('x',newline='') as stream:
  w=csv.writer(stream);w.writerow(['backbone','method','available_n','frames',*[k+'_'+j for k in ('T_cm','R_deg','ADDsym_m','IoU3D') for j in ('mean','std','median','P90')],'success_rate']);w.writerows(csvrows)
 text=['[확인] GREEN0918 수동점 PnP 참조 118장 평가를 완료했다: 잔차 게이트 PASS, 결론은 단일 세션의 FEASIBILITY_ONLY다.','',
 '# 정사각형 GREEN0918 수동 키포인트 PnP 참조 6D 평가','',
 '[확인] 새 사용자 계약은 화면 안 직접 수동 코너≥4, 기존 SQPnP→RefineLM, 등록 W/D/H=[1.1,1.1,0.15]m를 cuboid의 W/H/D=[1.1,0.15,1.1]m로 전달하며 proper C4를 사용한다. 직사각형의 6점 최소·LOO QA 필터·축 승인 절차를 이번 평가에 적용하지 않았다. 참조는 수동점과 K·치수만으로 생성하며 예측 코너는 입력하지 않는다. 이는 독립적인 물리 측정 GT가 아닌 **manual_keypoint_PnP_reference**다.','',
 '## 입력과 실제 재현 게이트','',
 '[확인] 119장·1세션 0918_dataset, 원영상 480×640의 RGB/JSON SHA와 K·치수·ID를 확인했다. 수동 코너 수는 3점1장/4점30장/5점52장/6점35장/7점1장이다. 029844는 3점으로 제외하고 주분모118을 유지했다. 적격118장은 모두 한 면에만 몰린 점 집합이 아니다. 선언 수동점602개, 화면 안600개를 그대로 유지했다. [입력 감사](INPUT_AUDIT.json).','',
 '[확인] 기존 YOLO·DOPE·ResNet-18 Base/N3 예측과 checkpoint를 재사용했다. 기존 2D Square 평가의 두 분모와 모든 seed를 실제 다시 채점하여 39,004개 수치 비교의 최대 절대차0.0을 확인했다. 모델을 다시 추론하거나 학습하지 않았다. [2D parity](TWO_D_PARITY.json).','',
 '![참조 잔차](figures/01_reference_residuals.png)','',
 f"[확인] 참조118/118개를 산출했다. 평균 재투영 잔차의 영상별 중앙값은 {f(r['median_mean_residual_px'])}px, mean>5px는 1/118={f(r['fraction_mean_residual_over5_px']*100)}%로 사전 게이트(중앙값≤3px, >5px비율≤10%)를 통과했다. 026399의 mean21.0013px를 주결과에서 제외하지 않았다. 민감도 범위 mean≤5px는117장이다. [참조 자세](SQUARE_REFERENCE_POSES.json), [잔차 게이트](REFERENCE_GATE.json).",
 '', '[확인] ≥5점88장의477회 LOO를 추가 계산했고 최대 proper C4 R 변화와 centroid T 변화 및 LOO 실패를 영상별로 기록했다. 4점30장에는 LOO를 적용하지 않았으며 LOO 수치를 필터나 승인 조건으로 사용하지 않았다. 원 RGB 참조 검수 sheet6개는 저장소 외부의 지정 `--private-dir`에만 저장했다. [private 파일명·SHA](PRIVATE_REVIEW_RECEIPT.json).','',
 '## 세 기반의 고정 경로와 모든 seed','',
 '![성공률](figures/02_success_all_backbones.png)','',
 '\n'.join(table[4:]),'',
 '[확인] seed-mean은 같은 ID의 세 seed 오차 또는 먼저 판정한 이진 성공 여부를 평균한 뒤 영상 통계를 계산한다. 영상 수를 3배로 간주하지 않는다. SD는 영상별 표본산포(ddof=1)이며 seed 산포나 평균의 CI가 아니다. 아래 raw와 METRICS에는 표본분산·중앙값·P90·최대·유효 n 및 기술용 95%CI를 모두 남겼다.','',
 '![오차 평균과 SD](figures/03_error_means_sd.png)','',
 '<details><summary>seed 1·2·3 각각의 결과</summary>','',
 '| 기반 | seed | 경로 | 자세 n/118 | T cm 평균±SD; 중앙값/P90 | R° 평균±SD; 중앙값/P90 | 성공률 % [기술CI] |',
 '| --- | --- | --- | --- | --- | --- | --- |']
 for b in C.BACKBONES:
  for seed in C.SEEDS:
   for a in C.METHODS:
    v=m['backbones'][b]['ALL118']['by_seed'][str(seed)][a];t=v['metrics']['T_cm'];rot=v['metrics']['R_deg']
    text.append(f"| {NAMES[b]} | {seed} | {LABELS[a]} | {v['available']}/118 | {mean_sd(t)}; {f(t['median'])}/{f(t['P90'])} | {mean_sd(rot)}; {f(rot['median'])}/{f(rot['P90'])} | {f(v['success_rate']['rate']*100)} {ci(v['success_rate']['CI95'],100)} |")
 text+=['','</details>','','## 같은 영상의 paired 기술 비교','',
 '| 기반 | 비교 | Δ성공 %p [기술95%CI] | ΔT cm [CI] | ΔR° [CI] | seed별 Δ성공 %p |','| --- | --- | --- | --- | --- | --- |']
 for b in C.BACKBONES:
  for name,v in p['backbones'][b]['ALL118'].items():
   z=v['seed_mean'];rate=z['success_rate']
   text.append(f"| {NAMES[b]} | {name} | {f(rate['delta']*100)} {ci(rate['CI95'],100)} | {f(z['T_cm']['delta'])} {ci(z['T_cm']['CI95'])} | {f(z['R_deg']['delta'])} {ci(z['R_deg']['CI95'])} | {', '.join(f(x*100) for x in rate['per_seed_delta'])} |")
 text+=['','[확인] 모든 paired CI는 고정118영상의 같은 multinomial frame draw10,000회, seed20260917을 공유한다. 민감도·수동점 집단은 같은 master draw에서 해당 ID만 취한다. 한 세션이므로 cluster bootstrap을 하지 않았으며 frame CI는 세션 내 상관을 해결하지 않는 기술용 수치다. SUPPORTED 또는 독립 확증을 주장하지 않는다. [PAIRED](PAIRED.json), [VERDICT](VERDICT.json).','',
 '[확인] YOLO의 N3 단독 성공률은 Base보다 낮았고(15.2542%→13.5593%), DOPE SubPix의 T 평균도 Base보다 높았다. 불리한 경로·seed·큰 유한 오차를 제거하거나 clipping하지 않았다. DOPE의 미산출5개 ID와 모든 경로의 손상·회복 ID는 [FAILURES.json](FAILURES.json)에 보존했다.','',
 '## 고정 민감도와 4점 불안정성','',
 '![고정 집단](figures/04_fixed_sensitivity_groups.png)','',
 '| 범위 | 기반 | 경로 | 자세 n/전체 | T cm 평균±SD | R° 평균±SD | 성공률 % |','| --- | --- | --- | --- | --- | --- | --- |']
 for scope in ('reference_residual_mean_le5','manual4','manual_ge5'):
  for b in C.BACKBONES:
   for a in C.METHODS:
    v=m['backbones'][b][scope]['seed_mean'][a]
    text.append(f"| {scope} | {NAMES[b]} | {LABELS[a]} | {v['available']}/{v['frames']} | {mean_sd(v['metrics']['T_cm'])} | {mean_sd(v['metrics']['R_deg'])} | {f(v['success_rate']['rate']*100)} |")
 text+=['','[추정] 4점에서는 작은 영상 잔차가 참조의 깊이·회전 안정성이나 물리적 정확성을 보장하지 않을 수 있다. 4점과≥5점을 나누어 보고했고, LOO는 기록 가능한≥5점에서만 계산했다. 수동 PnP 참조와 예측이 같은 영상 기하를 사용하므로 독립 참조로 일반화하지 않는다.','',
 '## 실행량·검산·재현','',
 f"[확인] 실제 F {ex['actual_F_calls']:,}회, SQPnP {ex['actual_PnP_calls']['solvePnP']:,}회, RefineLM {ex['actual_PnP_calls']['solvePnPRefineLM']:,}회다. 원 참조 solve118회와 LOO477회는 이 F 횟수와 구분한다. cornerSubPix 코너 호출은 {ex['actual_subpix_corner_calls']:,}회다. 새 학습·모델/N3 forward·합성 생성·촬영·주석 수정은0이다. 실제 캐시 기반 실행 시간은 배포 latency가 아니다. [EXECUTION](EXECUTION.json).",'',
 '[확인] 정사각형의90° W/D 교환은 proper C4 등가여서 회전오차나 이 표로 직사각형 뒤바뀜을 측정할 수 없다. 직사각형과는 최소점·참조 생성·분모가 다르므로 두 결과를 합치거나 직접 순위를 비교하지 않았다.','',
 '[확인] 독립 검산은 저장된 R/t에서 T/R/ADD를 다시 계산하고, 참조 재투영·잔차 gate·마스크/중심 보존·모든 통계/paired frame CI를 확인한다. IoU의 raw 통계는 검산하며 intersection geometry는 독립 재구현하지 않았다. [VERIFICATION](VERIFICATION.json), [방법](METHOD_KO.md), [재현](REPRODUCE.md), [원고 표](PAPER_TABLE_KO.md).']
 (C.DOC/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
 (C.DOC/'METHOD_KO.md').write_text('''# 고정 방법과 참조의 범위

[확인] `references.reference_fit(manual_xy,K,corner_indices)`에는 직접 수동점·K·코너 인덱스만 입력된다. 3D 모델은 기존 cuboid(1.1,0.15,1.1)이고 기존 solve의 SQPnP→RefineLM을 그대로 호출한다. 중심8·PnP 파생점·예측점은 fit하지 않는다. no solution 또는 tz≤0는 참조 실패로 기록하고 다른 참조로 대체하지 않는다.

[확인] CF0123는 앞면0–3, 위{0,1,4,5}, 아래{2,3,6,7}이다. 입력119장의 direct manual_in_frame 집합은 그대로 고정했다. 잔차 중앙값≤3px이고 mean>5px 비율≤10%면 자동 진행한다. 사람 승인이나 기존 직사각형6점/LOO 필터는 필요하지 않은 새 계약이다. LOO는≥5점에서 하나씩 제거하여 proper C4 최소회전과 centroid 변화만 기록한다.

[확인] Base/N3는 기존 선택 객체·box·score·confidence·support·중심8·좌표를 그대로 재사용한다. N3를 다시 실행하지 않는다. SubPix는 원영상 grayscale uint8에서 각 지원 코너0–7에 cornerSubPix(win=(5,5),zeroZone=(-1,-1),EPS|COUNT40,.001)를 호출한다. 원영상 밖 초기점·오류·비유한/밖 반환은 시작점을 유지한다. 최종 새 SubPix/직렬점에 원래 Base 기준 raw diagonal1% cap을 마지막에 한 번 적용한다. N3 기존 내부cap을 재해석하거나 Base/N3 자체를 다시 이동하지 않는다.

[확인] 자세는 SHA를 확인한 기존 pose.infer/metric을 그대로 사용한다. W=D 분기는 SQUARE_IDENTICAL_WD이며≥6개 유한 예측 코너가 있어야 F 자세를 산출한다. 이것은 manual reference의≥4 조건과 다르다. 참조와 예측 실패를 숨기지 않고 주분모118을 보존한다. proper C4는 중심8을 고정하며90° W/D 교환이 등가다.

[확인] 5cm·5°는 strict T<5cm 및 proper C4 최소R<5°이다. seed 이진 결과를 먼저 만들고 ID별 평균한다. numerical error는 available-only의 n을 명시한다. 표본분산/SD는ddof1, median/P90은linearquantile이다. paired master frame resampling10,000회 seed20260917, 모든 scope는 같은118개 frame draw에서 subset한다. 단일세션 기술값이며 cluster CI와 확증판정을 하지 않는다.

[확인] 코드: [참조](../../../scripts/research/pallet_square6d_manualpnp_20261010/references.py), [평가](../../../scripts/research/pallet_square6d_manualpnp_20261010/evaluate.py), [집계](../../../scripts/research/pallet_square6d_manualpnp_20261010/summarize.py), [독립 검산](../../../scripts/research/pallet_square6d_manualpnp_20261010/verify.py). 고정 계약과 원F SHA는 [METHOD_LOCK.json](METHOD_LOCK.json), 실제 source 봉인은 [COORDINATES_SEAL.json](COORDINATES_SEAL.json)에 있다.
''')
 (C.DOC/'REPRODUCE.md').write_text('''# 재현

[확인] 기존 환경과 비공개 원본 입력을 읽는다. 새 학습·모델 forward 없이 fresh output에만 쓴다. 원본 RGB 참조 sheet는 저장소 밖 private-dir만 허용한다.

```bash
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/input-checkout"
export PALLET_BASELINE_ROOT="/path/to/existing/baseline-checkout"
export PALLET_SQUARE_OUTPUT="/path/to/nonexistent/fresh-square-output"
export PALLET_PRIVATE_SQUARE="/path/to/private/review-output"
export PYTHONDONTWRITEBYTECODE=1
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.preflight
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.references --private-dir "$PALLET_PRIVATE_SQUARE"
# 2D parity 및 REFERENCE_GATE.proceed_to_evaluation=true일 때만 계속한다.
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.evaluate
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.summarize
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.report
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.verify --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"
```

[확인] 각 스크립트는 기존 출력이 있으면 멈춘다. source/annotation/checkpoint는 수정하지 않는다. 공개 수치 검산은 원 RGB나 checkpoint 없이도 가능하다:

```bash
unset PALLET_SQUARE_OUTPUT
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.verify --output /path/to/nonexistent/public-recheck.json
```

[확인] 2D parity는 기존 두 manual 분모의 결과를 다시 계산하고 비교한다. reference는 예측을 열지 않는 별도 함수이며 잔차 gate를 통과해야 평가가 실행된다. 독립 verifier는 PnP·모델·F를 호출하지 않는다. 다른 사용자 재현에서는 원세션의 private 보존 snapshot을 만들어 낸 것처럼 표시하지 않는다.
''')
 C.write('FIGURE_INDEX.json',dict(status='COMPLETE',figures=figs,numeric_only=True,raw_RGB_published=False))
 (C.DOC/'README.md').write_text('[확인] GREEN0918 118장 수동 PnP 참조 기반6D 평가 완료; FEASIBILITY_ONLY.\n\n[한국어 보고서](REPORT_KO.md) · [방법](METHOD_KO.md) · [재현](REPRODUCE.md) · [원고 표](PAPER_TABLE_KO.md) · [검산](VERIFICATION.json) · [그림](FIGURE_INDEX.json)\n')
 print('SQUARE_REPORT_COMPLETE',len(figs),flush=True)
if __name__=='__main__':main()
