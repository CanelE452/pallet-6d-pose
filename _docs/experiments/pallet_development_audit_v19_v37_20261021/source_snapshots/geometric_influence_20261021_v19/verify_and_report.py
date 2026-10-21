"""Independent numerical selected-pose audit; no model, solver or observation call."""
from pathlib import Path
import csv,gzip,json,hashlib,math,time
import numpy as np
R=Path(__file__).resolve().parent;T=R.parent;B=Path('${DATA_ROOT}')
def read(p):return json.loads(p.read_text())
def rows(p):
 with gzip.open(p,'rt')as h:return [json.loads(s)for s in h]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cw(p,z):
 with p.open('x',newline='')as h:w=csv.DictWriter(h,fieldnames=list(z[0]));w.writeheader();w.writerows(z)
syn={x['id']:x for x in rows(B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')};refs=read(B/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references'];n=0;errT=[];errR=[];allrows=[];record=[];cost=[];swaps=[]
def angle(a,b):return math.degrees(math.acos(max(-1.,min(1.,(math.fsum(float(x)*float(y)for x,y in zip(np.asarray(a).ravel(),np.asarray(b).ravel()))-1)/2))))
syms=[np.diag([1,1,1]),np.diag([-1,1,-1])];quarters=[np.array([[0,0,1],[0,1,0],[-1,0,0]]),np.array([[0,0,-1],[0,1,0],[1,0,0]])]
for pop in ['SYNTHETIC_VAL','CAL46']:
 folder=R/f'{pop}_s20261018';ev=read(folder/'EVALUATION.json');assert sha(folder/'PREDICTIONS.jsonl.gz')==read(folder/'SEAL.json')['prediction_sha256'];ps=rows(folder/'PREDICTIONS.jsonl.gz');by={x['id']:x for x in csv.DictReader((folder/'PAIRED_ROWS.csv').open())}
 for arm in ['BASELINE','CONTROL','INFLUENCE']:
  vals=[];rec={}
  for p in ps:
   g=syn[p['id']]if pop=='SYNTHETIC_VAL'else refs[p['partition']][p['id']];G=np.array(g['R_fixed']if pop=='SYNTHETIC_VAL'else g['R']);t=np.array(g['t']);q=p['methods'][arm]['pose'];tc=100*math.sqrt(math.fsum(float(x)**2 for x in np.asarray(q['centroid'])-t));rr=min(angle(G@s,q['R_physical'])for s in syms);ss=tc<=5 and rr<=5;row=by[p['id']];errT.append(abs(tc-float(row[arm+'_translation_cm'])));errR.append(abs(rr-float(row[arm+'_rotation_deg'])));assert ss==(row[arm+'_success']=='True');n+=1;vals.append((tc,rr,ss));rec.setdefault(p['recording'],[]).append((tc,rr,ss));quarter=min(angle(G@s,q['R_physical'])for s in quarters);swaps.append(dict(population=pop,method=arm,id=p['id'],R_C2_deg=rr,R_quarter_turn_deg=quarter,quarter_turn_compatible=rr>45 and quarter<=15))
  a=ev['methods'][arm];assert len(vals)==a['denominator']and len(vals)==a['T_n']==a['R_n']and a['no_pose']==0;assert sum(z[2]for z in vals)==a['successes'];out=dict(population=pop,seed=20261018,method=arm,**a);out['quarter_turn_compatible']=sum(s['quarter_turn_compatible']for s in swaps if s['population']==pop and s['method']==arm);allrows.append(out)
  for rr,z in rec.items():record.append(dict(population=pop,seed=20261018,method=arm,recording=rr,denominator=len(z),available=len(z),no_pose=0,successes=sum(x[2]for x in z),T_mean=float(np.mean([x[0]for x in z])),R_mean=float(np.mean([x[1]for x in z]))))
for arm in ['CONTROL','INFLUENCE']:cost.append(read(R/f'{arm}_s20261018/RECEIPT.json'))
pres=read(R/'PRESERVATION_BEFORE.json');assert all(sha(Path(p))==s for p,s in pres.items());assert max(errT)<1e-7 and max(errR)<1e-7
cw(R/'ALL_METRICS.csv',allrows);cw(R/'RECORDING_METRICS.csv',record);cw(R/'QUARTER_TURN_DIAGNOSTIC.csv',swaps);cw(R/'FIT_COST.csv',cost)
check=dict(status='PASS',independent_pose_metric_checks=n,max_T_error_cm=max(errT),max_R_error_deg=max(errR),preserved_reports=len(pres),new_optimizer_fits=2,training_fit_seconds=sum(c['fit_seconds']for c in cost),feature_seconds=sum(read(R/p/'SEAL.json')['seconds']for p in ['SYNTHETIC','CAL46']),new_other_seed=0,new_PRIMARY=0,new_PnP=0,new_RGB_queries=0,independent_test_n=0)
(R/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(check,indent=2)+'\n')
text='''v19: 저장 관측의 기하 정보·corner 삭제 영향 실험

사전 조건은 PROTOCOL.json에 고정했습니다. 원188특징·후보·관측·CAL 생산은 유지하고, 두 family의 실제 점/부분선 Jacobian 정보행렬 및 관측 corner를 하나씩 뺀 선형화 영향48입력을 추가했습니다. 기존 multimodal profile을 새로 만들지 않았습니다. 이 값은 현재 관측 대응을 가정한 민감도이며 안정성이 정답이나 독립 physical uncertainty라는 뜻이 아닙니다. 선 공유 상관·유한 지지 범위·모델 owner 오류·비선형 basin·불등식·prior·outline을 완전한 posterior로 모델링하지 않습니다.

원v5 C2 head와 동일 초기값의188입력 계속학습 대조군/236입력 확장군을 seed18 각각1fit했습니다. hidden64/64, frame BCE, AdamW.001/decay.0001/batch32/max50/patience5/strict lower validation BCE를 고정했습니다. 추가열 초기값은0입니다. 각각6epoch/best1로 선택됐습니다. train-only 정규화이며 CAL은 평가에만 사용했습니다. 데이터와 CAL은 반복 노출 개발자료입니다.

합성 validation235: v5 공동101/T17.872301/R9.206962 → 계속학습101/17.834802/9.199530 → 확장101/17.515345/9.174503. 확장 대 v5 회복0/손상0/선택변경7, T>50 26→25, T P90 54.870281→53.777377, P95 81.342719→77.751670. 나머지 안전지표 악화 없이 양쪽 대조 gate를 통과했습니다. bank 성공포함133은 고정이며 새 후보 회복이 아닙니다.

CAL46: 세 방법의 실제 선택 candidate가46개 모두 동일했습니다. 성공39/T2.583822/R5.242126, 회복0/손상0, 미산출0, R>45=2 모두 같습니다. 따라서 엄격 gate의 no_mean_improvement로 미통과입니다. 손상하지 않았다는 것과 개선을 확인했다는 것을 구분합니다. 나머지 seed/PRIMARY는 실행하지 않았습니다. 이 동일 결과 하나를 모든 경로의 종료로 보지 않고, 다음 비중복 실험 v20에서 기존 profile의 순서/family 정보를 보존하는 가설을 검토합니다. v19 문턱·손실·gate는 조정하지 않습니다.

비용: 신규 fit2, CPU1/GPU0, 각 학습+epoch validation/checkpoint 약1.1초이며 파싱/준비는 별도 FIT_COST.csv total_seconds에 기록했습니다. 저장 특징 생성 synthetic1003 약66초/CAL46 약5초, 새 PnP/관측/픽셀 추출/SAM0입니다. 이 구간은 전체 영상 추론 가속이 아닙니다. 전체/paired/recording/90도 호환 진단과 실제 로그를 보존했습니다. 재현하려면 기존 원자료/런타임이 필요하며 exclusive-create 출력 폴더를 재사용하지 않습니다.

실행 명령(새 출력 root로 복사·binding 확인 후):
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 [PRIVATE_PATH_OMITTED] -B geometric_influence_20261021_v19/experiment.py features --population SYNTHETIC
동일 prefix로 train --arm CONTROL, train --arm INFLUENCE, predict --population SYNTHETIC_VAL, evaluate --population SYNTHETIC_VAL, features --population CAL46, predict --population CAL46, evaluate --population CAL46을 실행했습니다.

원 저장소 HEAD7e92fdcefb0a37bdee0aef95d3e86c572e23b967/과학 소스base93c24d143fa3901de1fa127b3156e8250d81dba8와 dirty 변경, v5–v18, 이전 최종 보고서와 원고는 보존했습니다. 독립 owner/COMMON/pre-CAL 미해결은 그대로이며 새 실사 정답/독립 검증은0입니다.
'''
(R/'REPORT_KO.md').write_text(text);print(json.dumps(check,indent=2))
