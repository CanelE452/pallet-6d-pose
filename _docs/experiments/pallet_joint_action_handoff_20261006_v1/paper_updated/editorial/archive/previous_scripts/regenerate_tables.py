"""Regenerate manuscript tables from verbatim, Git-blob-verified published CSVs.
No model inference, statistical refitting, or experiment is performed here.
"""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/github_tables'; OUT=ROOT/'tables'
manifest=json.loads((ROOT/'evidence/SOURCE_MANIFEST.json').read_text())
MAP=[]
for name,m in manifest.items():
 b=(SRC/name).read_bytes()
 assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==m['git_blob_sha1']
def data(name):
 return list(csv.DictReader((SRC/name).open(encoding='utf-8',newline='')))
def n(v,d=3):
 if v in ('',None): return r'\notrun'
 if v=='NA': return r'\NA'
 return f'{float(v):.{d}f}'
def integer(v): return f'{int(float(v)):,}'
def method(x):
 return {'N0 replay P':'N0','N1 symmetry':'N1','N2 dimensions':'N2','N3 dimensions + symmetry':'N3','R0_plus_N3':'R0+N3','source_only_update':'합성 추가 학습','raw_pseudo_student':'원래 의사 레이블 학생','corrected_pseudo_student':'보정 의사 레이블 학생'}.get(x,x)
def backbone(x):
 return {'yolo':'YOLO','YOLO':'YOLO','dope':'DOPE','resnet18':r'ResNet-18$^{\dagger}$','ResNet18 RGB 10ep CONSTANT-fold':r'ResNet-18$^{\dagger}$'}.get(x,x)
def register(dest,source,row,cols):
 for k in cols:
  if k in row:
   MAP.append(dict(table_file=dest,source_file=source,source_column=k,source_row_identity={i:row[i] for i in ('Method','Path','Group','Status','Backbone','Metric','Seed') if i in row},value=row[k],commit=manifest[source]['commit']))
def table(dest,label,caption,headers,rows,spec,note='',wide=False,sep='3pt'):
 headers=[r'\shortstack{'+h.replace(r'\newline',r'\\')+'}' if r'\newline' in h else h for h in headers]
 env='table*' if wide else 'table'; width=r'\textwidth' if wide else r'\columnwidth'
 lines=[f'% Generated from verified GitHub CSV at 7e136fc. See evidence/SOURCE_MANIFEST.json.',f'\\begin{{{env}}}[t]',r'\centering\footnotesize',f'\\caption{{{caption}}}',f'\\label{{{label}}}',f'\\setlength{{\\tabcolsep}}{{{sep}}}',f'\\begin{{tabularx}}{{{width}}}{{{spec}}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
 for r in rows:
  if isinstance(r,str): lines.append(r)
  else: lines.append(' & '.join(r)+r' \\')
 lines += [r'\bottomrule',r'\end{tabularx}']
 if note: lines.append(r'\tabnote{'+note+'}')
 lines += [f'\\end{{{env}}}','']
 (OUT/dest).write_text('\n'.join(lines))

# Data composition. Zero here means zero ASSIGNED labels, not absence of occlusion.
s='tab_composition.csv';rows=[]
labels=['직사각형 / 플라스틱','직사각형 / 목재','기존 직사각형 합계','정사각형 GREEN0918']
for label,r in zip(labels,data(s)):
 cols=['Labeled clean','Labeled moderate','Labeled severe','Unknown','Total']; rows.append([label]+[integer(r[k]) for k in cols]);register('composition.tex',s,r,cols)
table('composition.tex','tab:composition','형상·재질별 가림 레이블 구성. 단위는 영상 수이다. 미분류는 전체 분모에 유지한다.',['집단','없음','중간','심함','미분류','전체'],rows,'Yrrrrr',r'목재와 정사각형의 0은 해당 등급으로 \emph{확정 분류된 영상 수가 0}이라는 뜻이며, 가림이 실제로 없다는 뜻이 아니다. 정사각형은 기존 319장에 합산하지 않는다.',sep='2.3pt')

s='tab_pose.csv';rows=[]
for r in data(s):
 cols=['T median cm','T P90 cm','R median deg','R P90 deg','Yaw median deg','Yaw P90 deg','IoU3D median','ADDsym AUC full','Pose frames']; register('pose_results.tex',s,r,cols)
 rows.append([method(r['Method']),n(r['T median cm'])+' / '+n(r['T P90 cm']),n(r['R median deg'])+' / '+n(r['R P90 deg']),n(r['Yaw median deg'])+' / '+n(r['Yaw P90 deg']),n(r['IoU3D median'],4),n(r['ADDsym AUC full'],4),integer(r['Pose frames'])+'/319'])
table('pose_results.tex','tab:pose','동일 319장의 기하 재구성 참조에 대한 YOLO 계열 자세 평가. N0/N1의 누락 자세 계산을 포함한다.',['방법',r'이동(cm)\newline 중앙값 / P90',r'회전(도)\newline 중앙값 / P90',r'방향각(도)\newline 중앙값 / P90',r'$\mathrm{IoU}_{3D}$',r'$\mathrm{ADD}_{sym}$ AUC','자세 산출'],rows,'lYYYrrr',r'학습한 방법은 세 seed별 통계의 평균이다. P는 과거 국소 보정, N0는 통제 재현 구성으로 서로 다른 가중치이다. 자세 산출은 정답 성공이 아니며, 모든 실패는 전체 AUC 분모에 남긴다. 이 참조는 독립 물리 계측이 아니다.',wide=True)

s='tab_ablation_results.csv';rows=[]
for r in data(s):
 cols=['Median px','P90 px','PCK10 %','E_sym'];register('ablation_results.tex',s,r,cols)
 rows.append([method(r['Method'])]+[n(r[k],6 if k=='E_sym' else 3) for k in cols])
table('ablation_results.tex','tab:ablation_results','동일 319장·8코너 규약의 통제 절제 결과. N0/N1은 원시 예측 재평가 값이다.',['구성','중앙값(px)','P90(px)',r'PCK10(\%)',r'$E_{\rm sym}$'],rows,'Yrrrr',r'N0: 국소 보정, N1: 대칭만, N2: 치수만, N3: 치수와 대칭. 세 seed 통계의 평균이며, 자세 결과는 표~\ref{tab:pose}에 함께 제시한다. 추가 파라미터와 치수 내용 자체의 효과가 완전히 분리된 것은 아니다.',sep='2.2pt')

s='tab_type_results.csv';rows=[]
for i,r in enumerate(data(s)):
 if i==4: rows.append(r'\midrule')
 cols=['Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T med cm','R med deg'];register('type_results.tex',s,r,cols)
 group='직사각형 / '+('플라스틱' if 'plastic' in r['Group'] else '목재') if i%4==0 else ''
 rows.append([group,method(r['Method']),r['Frames'],r['Full corners']+'/'+r['Observed corners']]+[n(r[k]) for k in cols[3:]])
table('type_results.tex','tab:type_results','기존 직사각형의 재질별 보정 효과. 각 재질 안에서 같은 영상·참조의 전후를 비교한다.',['집단','방법','영상',r'코너\newline 전체/관측','중앙값(px)','P90(px)',r'PCK10(\%)','이동(cm)','회전(도)'],rows,'Ylrrrrrrr',r'플라스틱 194장과 목재 125장의 합은 319장이다. 중앙 코너 오차·P90은 매칭된 유효 코너, PCK10은 전체 참조 코너 분모를 사용한다. 이동·회전은 중앙값이다. 서로 다른 재질의 난도를 동일하다고 가정하지 않는다.',wide=True)

rows=[]
for mode,title in [('declared','(a) 선언된 전체 수동 코너: 602점'),('in_frame','(b) 영상 내부 수동 코너: 600점')]:
 s='tab_square_manual_'+mode+'.csv'
 rows.append(r'\multicolumn{9}{l}{\textbf{'+title+r'}} \\')
 for r in data(s):
  cols=['Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T cm','R deg'];register('square_results.tex',s,r,cols)
  rows.append([backbone(r['Backbone']),method(r['Method']),integer(r['Frames']),integer(r['Full corners'])+'/'+integer(r['Observed corners']),n(r['Median px']),n(r['P90 px']),n(r['PCK10 %']),r'\notrun',r'\notrun'])
 if mode=='declared':rows.append(r'\midrule')
table('square_results.tex','tab:square_results','정사각형 GREEN0918의 두 평가 모드. 두 패널은 동일 119장과 동일 예측에서 참조 코너 포함 규칙만 다르다.',['기반','방법','영상',r'코너\newline 전체/관측','중앙값(px)','P90(px)',r'PCK10(\%)','이동(cm)','회전(도)'],rows,'Ylrrrrrrr',r'602점 중 영상 밖 좌표 2점을 제외하면 600점이다. 두 모드의 수치를 혼합하지 않는다. $\dagger$: 10-epoch CONSTANT-fold RGB 모델. 단일 촬영 세션·한 치수 $[1.1,1.1,0.15]$m이며, 독립 표준 자세 참조가 없어 이동·회전 정확도는 x이다. 세션 간 일반화 신뢰구간은 정의할 수 없다. N3의 세 seed 통계는 앙상블이 아니라 산술평균이다.',wide=True)

s='tab_occlusion_results.csv';rows=[];last=None
groups={'all':'전체','clean':'가림 없음','moderate':'중간 가림','severe':'심한 가림','unclassified':'미분류'}
for r in data(s):
 if last is not None and last!=r['Group']: rows.append(r'\midrule')
 first=last!=r['Group']; last=r['Group']
 cols=['Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T med cm','T P90 cm','R med deg','R P90 deg','Pose %'];register('occlusion_results.tex',s,r,cols)
 rows.append([groups[r['Group']] if first else '',method(r['Method']),r['Frames'],r['Full corners']+'/'+r['Observed corners'],n(r['Median px']),n(r['P90 px']),n(r['PCK10 %']),n(r['T med cm'])+' / '+n(r['T P90 cm']),n(r['R med deg'])+' / '+n(r['R P90 deg'])])
table('occlusion_results.tex','tab:occlusion_results','외부 가림 정도별 YOLO 보정 전후. 전체 결과와 미분류 191장을 모두 유지한다.',['집단','방법','영상',r'코너\newline 전체/관측','중앙값(px)','P90(px)',r'PCK10(\%)',r'이동(cm)\newline 중앙값 / P90',r'회전(도)\newline 중앙값 / P90'],rows,'llrrrrrYY',r'모든 행의 자세 산출률은 100\%이며 정확한 자세를 뜻하지 않는다. 직접 가림 등급이 확인된 것은 128장이다. 중간 20장·심함 79장은 이번에 연결한 직접 검수 버전이며, 과거 21장·78장과 다른 한 프레임의 변경 사유는 미확인이다. 코너 지표의 관측 분모와 전체 분모를 구분한다.',wide=True,sep='2.5pt')

s='tab_visibility.csv'; rows=[]; last=None;labels={'DIRECT_VISIBLE':'직접 가시','EXTERNAL_OCCLUDED':'외부 가림','UNKNOWN':'가시성 미확인'}
for r in data(s):
 if last is not None and last!=r['Status']:rows.append(r'\midrule')
 first=last!=r['Status']; last=r['Status']
 cols=['Full corners','Observed corners','Median px','P90 px','PCK10 %'];register('visibility_results.tex',s,r,cols)
 rows.append([labels[last] if first else '',method(r['Method']),integer(r['Full corners'])+'/'+integer(r['Observed corners']),n(r['Median px']),n(r['P90 px']),n(r['PCK10 %'])])
table('visibility_results.tex','tab:visibility','기존 수동 검수의 가시성 상태를 연결한 YOLO 평가. 좌표는 기존 DEV 참조를 유지하였다.',['상태','방법',r'코너\newline 전체/관측','중앙값(px)','P90(px)',r'PCK10(\%)'],rows,'Ylrrrr',r'직접 가시 66점·외부 가림 5점의 상태만 재사용하였다. 재클릭 좌표로 참조를 교체하지 않았으며, PnP 보조 검수 이력과 사전 예측 노출 미확인을 가진 제한된 부분집합이다. 가시성 미확인 2,428점의 오차는 계산할 수 있지만 그 상태가 새로 검증됐다는 뜻은 아니다. 5점 집단의 PCK10=0은 실제 관측값이다.',wide=True)

s='tab_tail.csv';raw=data(s);rows=[]
for title,col in [('초기 오차가 이동 한계 이내','Inside count'),('초기 오차가 이동 한계 밖','Outside count'),('한계 안: 개선','Inside better'),('한계 안: 무차이','Inside same'),('한계 안: 악화','Inside worse'),('한계 밖: 개선','Outside better'),('한계 밖: 무차이','Outside same'),('한계 밖: 악화','Outside worse'),(r'양호($<5$px)$\to$악화($>10$px)','Good5-bad10'),(r'큰 오차($>20$px)$\to$복구($<10$px)','Bad20-good10'),('2D·이동·회전 모두 개선 (영상)','2D+T+R better frames'),('2D 개선, 이동 또는 회전 악화 (영상)','2D better T/R worse frames')]:
 cells=[title]
 for arm in ['P','N3 dimensions + symmetry']:
  rr=[r for r in raw if r['Method']==arm]
  cells.append(' / '.join(integer(r[col]) for r in rr))
  for r in rr:register('tail_results.tex',s,r,[col])
 rows.append(cells)
table('tail_results.tex','tab:tail','동일 참조 대응을 고정한 큰 오류·손상·복구 분석. 각 칸은 seed 1 / 2 / 3의 실제 개수이다.',['진단 항목','R0 → P','R0 → N3'],rows,'Yrr',r'보정 범위는 원본 영상 대각선의 1\%이다. 코너 집계는 2,445점, 공동 변화 분석은 매칭된 311장이다. 마지막 두 행만 영상 수이고 그 위는 코너 수이다. 대칭 평가의 최적 대응 변경으로 이동 한계의 기하적 하한을 바꾸지 않았다. 무차이·악화·복구 없음도 그대로 보고한다.',wide=True)

s='tab_backbones.csv';rows=[];last=None
for r in data(s):
 if last is not None and last!=r['Backbone']:rows.append(r'\midrule')
 first=last!=r['Backbone'];last=r['Backbone']
 cols=['Median px','P90 px','PCK10 %','T med cm','R med deg','Matched frames','Observed corners','Pose frames','Pose %'];register('backbone_results.tex',s,r,cols)
 rows.append([backbone(last) if first else '',r['Path']]+[n(r[k]) for k in cols[:5]]+[r['Matched frames']+'/319',integer(r['Observed corners']),n(r['Pose %'],2)])
table('backbone_results.tex','tab:backbones','세 기반 추정기의 동일 N3 보정 전후. 전체 평가 영상은 모두 319장, 전체 참조 코너는 2,499점이다.',['기반','경로','중앙값(px)','P90(px)',r'PCK10(\%)','이동(cm)','회전(도)','매칭','관측 코너',r'자세 산출(\%)'],rows,'Ylrrrrrrrr',r'N3는 세 seed별 통계의 평균이다. $\dagger$: 10-epoch 합성 CONSTANT 가중치를 고정 변환과 합친 RGB-only ResNet-18. 60-epoch 모델이나 기본 신경망이 실제 치수를 받는 모델이 아니다. 각 기반의 보정 전후 집단은 고정되지만 기반 간 조건부 매칭 집단이 달라 조건부 중앙값만으로 절대 순위를 만들지 않는다. 동일 가중치 전이 실험도 아니다.',wide=True,sep='2.5pt')

s='tab_cost.csv';raw=data(s);rows=[]
for env in ['pallet-yolo26','pallet-pose']:
 rows.append(r'\multicolumn{7}{l}{\textbf{환경: '+env+r'}} \\')
 for r in raw:
  if r['Environment']!=env:continue
  cols=['Added params','CUDA med ms','CUDA P90 ms','Wall med ms','N3 only med ms','Peak allocated MiB'];register('cost_results.tex',s,r,cols)
  rows.append([backbone(r['Backbone'])+' '+r['Path'],integer(r['Added params']),n(r['CUDA med ms']),n(r['CUDA P90 ms']),n(r['Wall med ms']),n(r['N3 only med ms']),n(r['Peak allocated MiB'],2)])
 if env=='pallet-yolo26':rows.append(r'\midrule')
table('cost_results.tex','tab:cost','고정 26프레임·seed 1의 처리 시간과 추가 파라미터. 같은 RTX 3080에서 실행 환경을 분리하여 표시하였다.',['기반·경로','추가 파라미터','전체 중앙(ms)','전체 P90(ms)','벽시계 중앙(ms)','보정 중앙(ms)','최대 메모리(MiB)'],rows,'Yrrrrrr',r'경로별 준비 20회와 프레임당 5반복, 배치 1; 파일 읽기는 계측에서 제외한다. 전체 중앙/P90 및 보정 중앙은 CUDA event 시간이며 동기화 벽시계도 함께 남겼다. YOLO는 이번 추가 측정이고 DOPE·ResNet은 같은 고정 계약의 기존 측정이다. 환경 차이를 무시한 순수 하드웨어 비교 또는 Jetson 성능으로 해석하지 않는다. 보정 없는 경로의 보정 시간은 해당 없음이다.',wide=True,sep='2.5pt')

s='tab_comparators.csv';rows=[]
for r in data(s):
 cols=['Median px','P90 px','PCK10 %','T med cm','R med deg'];register('comparator_results.tex',s,r,cols)
 names={'R0':'R0: 보정 없음','P':'P: 국소 분포','D':'D: 직접 회귀','L':'L: 선 구조','PoseFix':'PoseFix 기반','N3':'N3: 제안 구성'}
 rows.append([names[r['Method']]]+[n(r[k]) for k in cols])
table('comparator_results.tex','tab:comparators','동일 319장·8코너 규약의 보정 방법 비교. 모든 수치는 같은 원시 예측 평가 규약으로 재검산하였다.',['방법','중앙값(px)','P90(px)',r'PCK10(\%)','이동(cm)','회전(도)'],rows,'Yrrrrr',r'입력·구조·학습 손실 또는 감독 예산이 다른 전체 방법 비교이며 구성 요소 하나의 인과 효과를 뜻하지 않는다. PoseFix 기반 비교군의 중앙값이 더 낮은 지표를 보존한다. 다른 논문 데이터의 요약 성능을 이 표에 가져오지 않았다.',wide=True)

s='tab_update_alternative.csv';rows=[]
for r in data(s):
 cols=['Median px','P90 px','PCK10 %','T med cm','R med deg'];register('update_results.tex',s,r,cols)
 rows.append([method(r['Method'])]+[n(r[k]) for k in cols])
table('update_results.tex','tab:update_alternative','공통 비노출 조건을 확인한 플라스틱 128장의 업데이트 대안 비교. 이 표의 R0와 N3도 모두 같은 128장의 결과이다.',['경로','중앙값(px)','P90(px)',r'PCK10(\%)','이동(cm)','회전(도)'],rows,'Yrrrrr',r'전체 참조 985점, 관측 931점이며 반복 사용한 개발 부분집합이다. 독립 TEST가 아니다. 별도 교사를 사용한 보정 의사 레이블 학생을 N3 학생으로 명명하지 않는다. 원래 319장 학생 비교는 공통 노출 계약이 성립하지 않아 대체 수치를 넣지 않았다. 학습·감독 예산 차이가 있는 전체 방법 비교이다.',wide=True)

s='tab_pose_uncertainty.csv';rows=[];last=None
metric_names={'translation_cm_median':'이동 중앙값 (cm)','translation_cm_P90':'이동 P90 (cm)','rotation_deg_median':'회전 중앙값 (도)','rotation_deg_P90':'회전 P90 (도)','yaw_deg_median':'방향각 중앙값 (도)','yaw_deg_P90':'방향각 P90 (도)'}
for r in data(s):
 if last is not None and last!=r['Backbone']:rows.append(r'\midrule')
 first=last!=r['Backbone'];last=r['Backbone']
 cols=['Mean delta','CI95 low','CI95 high','Seed SD','Seed min','Seed max'];register('pose_uncertainty.tex',s,r,cols)
 rows.append([backbone(last) if first else '',metric_names[r['Metric']],n(r['Mean delta']),'['+n(r['CI95 low'])+', '+n(r['CI95 high'])+']',n(r['Seed SD']),'['+n(r['Seed min'])+', '+n(r['Seed max'])+']'])
table('pose_uncertainty.tex','tab:pose_uncertainty','N3−Base 자세 통계 차이의 사후 불확실성. 음의 차이는 오차 감소를 뜻한다.',['기반','지표','평균 차이',r'95\% 구간','seed 표준편차','seed 최솟값 / 최댓값'],rows,'lYrrrr',r'13개 촬영 세션을 단위로 동일한 10,000회 짝지은 재표집을 사용하고 각 재표집 안에서 세 seed의 통계 차이를 평균했다. 초기 난수 seed는 20260917이며 다중비교 보정은 없다. 통계량(after)−통계량(before)이지 프레임 차이의 중앙값이 아니다. ResNet 이동 중앙값처럼 구간이 0을 포함하는 결과를 확증적 개선으로 바꾸지 않는다.',wide=True)

(ROOT/'evidence/PAPER_CELL_MAP.json').write_text(json.dumps(MAP,ensure_ascii=False,indent=2))
print(f'Generated {len(set(x["table_file"] for x in MAP))} tables; {len(MAP)} numeric source links.')
