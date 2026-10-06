"""Rebuild publication tables from the frozen, unmodified CSV snapshot.
No training, prediction, data relabeling, or metric recomputation is performed.
Only presentation, selection of existing rows and decimal rounding change.
"""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
MAP=[]

def read(stem):
    file=ROOT/'evidence/github_tables'/f'tab_{stem}.csv'
    with file.open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
    for n,r in enumerate(rows,2):r['_file']=file.relative_to(ROOT).as_posix();r['_line']=n
    return rows

def num(row,key,d=3):
    v=row[key]
    if v in ('','x','NA'):return r'\notrun' if v!='NA' else r'\NA'
    val=float(v);s=f'{val:,.0f}' if d==0 else f'{val:.{d}f}'
    MAP.append({'file':row['_file'],'csv_line':row['_line'],'field':key,'raw':v,'display':s})
    return s

def write(name,caption,label,cols,header,rows,note='',wide=True):
    env='table*' if wide else 'table';w=r'\textwidth' if wide else r'\columnwidth'
    s='\\begin{'+env+'}[t]\n\\centering\\footnotesize\n'
    s+=r'\setlength{\tabcolsep}{4pt}'+'\n'+r'\caption{'+caption+'}\\label{'+label+'}\n'
    s+=r'\begin{tabularx}{'+w+'}{'+cols+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'
    for r in rows:s+=r+'\n'
    s+='\\bottomrule\n\\end{tabularx}\n'
    if note:s+=r'\tabnote{'+note+'}\n'
    s+='\\end{'+env+'}\n'
    (ROOT/name).write_text(s,encoding='utf8')

def row(cells):return ' & '.join(cells)+r' \\'

def method(m):
    return {'R0':'Base','P':'영상 보정 P','N0 replay P':'N0: 영상 보정','N1 symmetry':'N1: 영상+회전 대응',
    'N2 dimensions':'N2: 영상+치수','N3 dimensions + symmetry':'N3: 제안 보정',
    'D':'직접 회귀 D','L':'선 구조 L','PoseFix':'PoseFix 기반','N3':'제안 보정 N3',
    'source_only_update':'합성 추가 학습','raw_pseudo_student':'원래 의사 레이블 학생',
    'corrected_pseudo_student':'보정 의사 레이블 학생','R0_plus_N3':'Base+N3'}.get(m,m)

def base(b): return {'yolo':'YOLO','YOLO':'YOLO','dope':'DOPE','resnet18':r'ResNet-18$^\dagger$','ResNet18 RGB 10ep CONSTANT-fold':r'ResNet-18$^\dagger$'}.get(b,b)

# Dataset counts are source-verified constants, with roles kept explicit.
write('tables/data.tex','학습·평가 자료의 역할과 확인 범위. 실사 집합들은 하나의 독립 시험집합으로 합산하지 않는다.','tab:data',r'p{.28\columnwidth}rY',r'자료 & 영상 수 & 역할',[
row(['합성 학습','55,980','보정기 합성 감독']),row(['합성 calibration','1,004','추론 온도 선택']),
row(['합성 selection','1,031','등록 분할; 추가 사용 이력 확인 필요']),row(['합성 held-out','1,985','등록 분할; 별도 평가 완료를 뜻하지 않음']),r'\midrule',
row(['직사각형 실사','319','13세션; 재사용 개발 평가']),row(['플라스틱 / 목재','194 / 125','319장의 하위집단']),
row(['정사각형 실사','119','1세션; 별도 2D 평가']),row(['업데이트 대안','128','공통 비노출 개발 부분집합'])],
'합성 네 분할 합계는 60,000장이다. 초기 추정기의 사전학습과 보정기의 추가 학습을 구분한다. 개체 수·모집·제외 이력 및 두 128장 집합의 ID 관계는 추가 확인 대상이다.',False)
rs=read('composition'); label={'Rectangular plastic':'직사각형 / 플라스틱','Rectangular wood':'직사각형 / 목재','DEV total':'직사각형 합계','Square GREEN0918':'정사각형'}
write('tables/composition.tex','실사 영상의 가림 레이블 확인 현황(영상 수).','tab:composition','Yrrrrr',r'집단 & 없음 & 중간 & 심함 & \shortstack{등급\\미확인} & 전체',[
row([label[r['Group']]]+[num(r,k,0) for k in ['Labeled clean','Labeled moderate','Labeled severe','Unknown','Total']]) for r in rs],
'0은 해당 등급으로 확정된 영상이 0장이라는 뜻이다. 목재 125장 전체와 정사각형 119장의 가림 등급은 미확인이다. 이 현황을 실제 운용의 가림 발생 빈도로 해석하지 않는다.',False)
write('tables/ablation_contract.tex','구성 요소 대조. 모든 경로의 PnP에는 같은 치수를 사용한다.','tab:ablation_contract',r'lYcc',r'구성 & 보정 & \shortstack{보정기의\\치수 입력} & \shortstack{학습 시\\회전 대응}',[
row(['Base','없음','--','--']),row(['N0','영상 기반 국소 보정','없음','고정 번호']),row(['N1','영상 기반 국소 보정','없음','동등성 고려']),row(['N2','영상 기반 국소 보정','사용','고정 번호']),row(['N3','영상 기반 국소 보정','사용','동등성 고려'])],
'N0--N3가 구성 요소 비교이다. 별도 실행의 영상 보정 P를 N0로 재명명하지 않는다.',False)
rs=read('backbones')
write('tables/backbone_results.tex','세 기반 추정기의 보정 전후. 같은 기반 안의 동일 초기 출력에 N3를 적용한다.','tab:backbones','Ylrrrrrrrr',r'기반 & 경로 & \shortstack{코너 중앙값\\(px)} & \shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & \shortstack{위치 중앙값\\(cm)} & \shortstack{회전 중앙값\\(도)} & \shortstack{매칭\\영상} & \shortstack{유효 예측\\코너} & \shortstack{자세 산출\\(\%)}',[
row([base(r['Backbone']),r['Path']]+[num(r,k) for k in ['Median px','P90 px','PCK10 %','T med cm','R med deg']]+[num(r,'Matched frames',0)+'/319',num(r,'Observed corners',0),num(r,'Pose %',2)]) for r in rs],
'전체 319장·참조 2,499코너. 중앙값·P90은 유효 코너 또는 산출된 자세에 조건부이다. N3는 세 seed의 통계 평균이며 앙상블이 아니다. $^\\dagger$10-epoch CONSTANT-fold RGB 모델. 기반별 매칭·자세 산출 집합은 다르므로 조건부 오차만으로 기반의 절대 순위를 정하지 않는다.')
rs=read('pose_uncertainty');chosen=[r for r in rs if r['Metric'] in ['translation_cm_median','rotation_deg_median']]
write('tables/pose_uncertainty_compact.tex','N3−Base 위치·회전 중앙값 변화의 사후 95\% 구간.','tab:pose_uncertainty_main','llrr',r'기반 & 지표 & 차이 & 95\% 구간',[
row([base(r['Backbone']), '위치(cm)' if r['Metric'].startswith('translation') else '회전(도)',num(r,'Mean delta'), '['+num(r,'CI95 low')+', '+num(r,'CI95 high')+']']) for r in chosen],
'13세션의 10,000회 짝지은 재표집. 음수는 오차 감소이다. ResNet 위치 구간은 0을 포함한다. P90·방향각·seed 편차는 보충자료에 제시한다.',False)
ps={r['Method']:r for r in read('pose')};ars=read('ablation_results')
write('tables/ablation_results.tex','동일 YOLO의 구성 요소 절제. 국소 보정에 치수·회전 대응을 추가한 효과를 분리한다.','tab:ablation_results','Yrrrrrrr',r'구성 & \shortstack{코너 중앙값\\(px)} & \shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & \shortstack{위치 중앙값\\(cm)} & \shortstack{위치 P90\\(cm)} & \shortstack{회전 중앙값\\(도)} & \shortstack{회전 P90\\(도)}',[
row([method(r['Method'])]+[num(r,k) for k in ['Median px','P90 px','PCK10 %']]+[num(ps[r['Method']],k) for k in ['T median cm','T P90 cm','R median deg','R P90 deg']]) for r in ars],
'319장·8코너 평가, 세 seed의 통계 평균. 각 구성의 자세 산출은 319/319이다. N3−N2의 변화는 지표별로 다르며, 치수 정보와 추가 파라미터 효과도 완전히 분리된 것은 아니다. 세부 방향각·겹침·정규화 영상 오차는 보충자료에 보존한다.')
rs=read('comparators')
write('tables/comparator_results.tex','동일 319장·8코너 평가에서의 보정 방법 비교.','tab:comparators','Yrrrrr',r'방법 & \shortstack{코너 중앙값\\(px)} & \shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & \shortstack{위치 중앙값\\(cm)} & \shortstack{회전 중앙값\\(도)}',[
row([method(r['Method'])]+[num(r,k) for k in ['Median px','P90 px','PCK10 %','T med cm','R med deg']]) for r in rs],
'Base는 보정 없음, P는 별도 실행의 영상 기반 보정으로 N0와 다른 가중치이다. 같은 평가 집합이 입력·손실·사전학습·선택 예산의 동등성을 의미하지 않는다. PoseFix 기반 방법이 유리한 중앙값을 유지한다.')
# Square keep both source inclusion modes and x pose, not a new selection.
sqrows=[]
for stem,title in [('square_manual_declared','(a) 전체 수동 참조 602코너'),('square_manual_in_frame','(b) 영상 내부 수동 참조 600코너')]:
    sqrows+=[r'\multicolumn{9}{l}{\textbf{'+title+r'}} \\']
    for r in read(stem):
        if r['Method']=='P':continue # historical P data preserved in supplementary
        sqrows.append(row([base(r['Backbone']),method(r['Method']).replace(' dimensions','')]+[num(r,k) for k in ['Median px','P90 px','PCK10 %']]+[num(r,'Observed corners',0),num(r,'Full corners',0),num(r,'T cm'),num(r,'R deg')]))
    sqrows.append(r'\midrule')
write('tables/square_results.tex','정사각형 119장에 대한 고정 모델 평가. 동일 예측의 두 참조 포함 규칙을 함께 제시한다.','tab:square_results','lYrrrrrrr',r'기반 & 방법 & \shortstack{코너 중앙값\\(px)} & \shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & \shortstack{유효 예측\\코너} & \shortstack{참조\\코너} & \shortstack{위치\\(cm)} & \shortstack{회전\\(도)}',sqrows[:-1],
'한 세션·한 등록 치수 [1.1, 1.1, 0.15]m. 두 모드는 영상 밖 참조 2개 포함 여부만 다르며 어느 쪽도 결과에 따라 선택하지 않았다. 위치·회전은 독립 표준 자세 참조 부재로 x이다. $^\\dagger$ResNet은 10-epoch CONSTANT-fold RGB 모델. 영상 보정 P의 두 모드는 보충자료에 보존한다.')
rs=read('occlusion_results');gd={'all':'전체','clean':'가림 없음','moderate':'중간 가림','severe':'심한 가림','unclassified':r'\shortstack[l]{가림 등급\\미확인}'}
write('tables/occlusion_results.tex','가림 정도별 YOLO Base와 N3. 확정 등급 128장은 모두 플라스틱이며 전체와 등급 미확인 191장도 남긴다.','tab:occlusion_results','Ylrrrrrrr',r'집단 & 방법 & 영상 & \shortstack{코너 중앙값\\(px)} & \shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & \shortstack{위치 중앙/P90\\(cm)} & \shortstack{회전 중앙/P90\\(도)} & \shortstack{참조/유효\\예측 코너}',[
row([gd[r['Group']],method(r['Method']),num(r,'Frames',0)]+[num(r,k) for k in ['Median px','P90 px','PCK10 %']]+[num(r,'T med cm')+' / '+num(r,'T P90 cm'),num(r,'R med deg')+' / '+num(r,'R P90 deg'),num(r,'Full corners',0)+' / '+num(r,'Observed corners',0)]) for r in rs if r['Method'] in ('R0','N3 dimensions + symmetry')],
'모든 행의 자세 산출률은 100\\%이며 정답률이 아니다. 등급 미확인은 네 번째 가림 난도가 아니다. 목재의 가림 분류는 미완료이다. P·N2의 같은 집단 결과와 전체 분모는 보충자료에 제시한다.')
rs=read('tail');groups=[rs[:3],rs[3:]]
# Actual seed triplets, never an invented representative run.
def trip(g,k):return ' / '.join(num(r,k,0) for r in g)
tr=[]
for key,lab in [('Inside count','초기 오차가 이동 상한 이내'),('Outside count','초기 오차가 이동 상한 밖'),('Good5-bad10',r'양호($<5$px)$\to$악화($>10$px)'),('Bad20-good10',r'큰 오차($>20$px)$\to$복구($<10$px)')]:tr.append(row([lab]+[trip(g,key) for g in groups]))
tr.insert(0,r'\multicolumn{3}{l}{\textbf{(a) 코너 수: 유효 평가 예측 2,445개}} \\')
tr += [r'\midrule',r'\multicolumn{3}{l}{\textbf{(b) 영상 수: 매칭 영상 311장}} \\']
for key,lab in [('2D+T+R better frames','코너·위치·회전 모두 개선'),('2D better T/R worse frames','코너 개선, 위치 또는 회전 악화')]:tr.append(row([lab]+[trip(g,key) for g in groups]))
write('tables/tail_results.tex','국소 보정의 복구 범위와 손상. 각 칸은 seed 1 / 2 / 3의 실제 개수이다.','tab:tail','Yrr',r'항목 & Base$\to$P & Base$\to$N3',tr,
'참조 대응을 초기 예측 기준으로 고정한 진단이다. 각 코너의 이동 상한은 원본 영상 대각선의 1\\%이다. 한계 안팎의 모든 개선·무차이·악화 개수는 보충자료에 함께 남긴다.')
rs=read('cost');cost=[]
for env in ['pallet-yolo26','pallet-pose']:
    cost+=[r'\multicolumn{6}{l}{환경: \texttt{'+env+r'}} \\']
    for r in rs:
        if r['Environment']!=env:continue
        cost.append(row([base(r['Backbone'])+' '+r['Path'],num(r,'Added params',0),num(r,'CUDA med ms'),num(r,'CUDA P90 ms'),num(r,'N3 only med ms'),num(r,'Peak allocated MiB',2)]))
write('tables/cost_results.tex','고정 표본에서의 추가 계산 비용. 동일 RTX 3080의 서로 다른 소프트웨어 환경을 구분한다.','tab:cost','Yrrrrr',r'기반·경로 & \shortstack{추가\\파라미터} & \shortstack{전체 중앙\\(ms)} & \shortstack{전체 P90\\(ms)} & \shortstack{보정 단독\\(ms)} & \shortstack{최대 메모리\\(MiB)}',cost,
'26프레임·seed 1·batch 1, 경로별 준비 20회와 프레임당 5반복. 시간은 CUDA event 기준이고 파일 읽기는 제외한다. 보정 단독 시간과 전체 중앙값의 차이는 다른 계측량이다. 동기화 벽시계 시간은 보충자료에 제시한다. Base의 보정 시간은 해당 없음이다.')
rs=read('update_alternative')
write('tables/update_results.tex','플라스틱 공통 128장에서 추정기 업데이트와 출력 보정의 비교.','tab:update_alternative','Yrrrrr',r'경로 & \shortstack{코너 중앙값\\(px)} & \shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & \shortstack{위치 중앙값\\(cm)} & \shortstack{회전 중앙값\\(도)}',[
row([method(r['Method'])]+[num(r,k) for k in ['Median px','P90 px','PCK10 %','T med cm','R med deg']]) for r in rs],
'모든 행의 집합은 128장·전체 참조 985코너·유효 예측 931코너이다. 반복 개발 부분집합이며 독립 시험이 아니다. 별도 교사와 감독을 사용한 학생을 N3 학생으로 부르지 않는다. 학습·감독 예산이 같은 단일 요인 비교는 아니다.')
(ROOT/'audit/TABLE_VALUE_TRANSFORMS.json').write_text(json.dumps(MAP,ensure_ascii=False,indent=2),encoding='utf8')
print('Generated',len(MAP),'source-to-display numeric bindings')
