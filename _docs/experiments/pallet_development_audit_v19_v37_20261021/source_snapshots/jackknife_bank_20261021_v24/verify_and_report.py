"""Independent final audit of completed seed18 union experiment, without fitting."""
from pathlib import Path
import json,csv,gzip,hashlib,math
import numpy as np
R=Path(__file__).resolve().parent;T=R.parent;B=Path('${DATA_ROOT}/pallet-boundary-action-repair-20261018-v1')
def rd(p):return json.loads(p.read_text())
def rows(p):
 with gzip.open(p,'rt')as h:return [json.loads(x)for x in h]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cw(p,z):
 with p.open('x',newline='')as h:w=csv.DictWriter(h,fieldnames=list(z[0]));w.writeheader();w.writerows(z)
GT={z['id']:z for z in rows(B.parent/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')};refs=rd(B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references'];S=[np.eye(3),np.diag([-1,1,-1])];QUARTER=[np.array([[0,0,1],[0,1,0],[-1,0,0]]),np.array([[0,0,-1],[0,1,0],[1,0,0]])]
def angle(a,b):return math.degrees(math.acos(max(-1.,min(1.,(math.fsum(float(x)*float(y)for x,y in zip(a.ravel(),b.ravel()))-1)/2))))
def metric(p,g):
 Rg=np.array(g.get('R_fixed',g.get('R')));rot=np.array(p['R_physical']);t=100*math.sqrt(math.fsum(float(x)**2 for x in np.array(p['centroid'])-g['t']));r=min(angle(Rg@s,rot)for s in S);return t,r,t<=5 and r<=5,r>45 and min(angle(Rg@s,rot)for s in QUARTER)<=15
n=0;maxT=0;maxR=0;metrics=[];recs=[];turns=[]
for pop in ['SYNTHETIC_VAL','CAL46']:
 out=R/f'{pop}_s20261018';ps=rows(out/'PREDICTIONS.jsonl.gz');assert sha(out/'PREDICTIONS.jsonl.gz')==rd(out/'SEAL.json')['prediction_sha256'];by={x['id']:x for x in csv.DictReader((out/'PAIRED_ROWS.csv').open())};ev=rd(out/'EVALUATION.json')
 for kind in ['before','after']:
  buckets={}
  for p in ps:
   g=GT[p['id']]if pop=='SYNTHETIC_VAL'else refs[p['partition']][p['id']];t,r,s,q=metric(p[kind+'_pose'],g);z=by[p['id']];dt=abs(t-float(z[kind+'_translation_cm']));dr=abs(r-float(z[kind+'_rotation_deg']));assert dt<1e-7 and dr<1e-7 and s==(z[kind+'_success']=='True');n+=1;maxT=max(maxT,dt);maxR=max(maxR,dr);turns.append(dict(population=pop,id=p['id'],method=kind,quarter_turn_compatible=q));buckets.setdefault(p['recording'],[]).append((t,r,s))
  a=ev[kind];metrics.append(dict(population=pop,seed=20261018,method=kind,**a,quarter_turn_compatible=sum(x['quarter_turn_compatible']for x in turns if x['population']==pop and x['method']==kind)))
  for rec,z in buckets.items():recs.append(dict(population=pop,method=kind,recording=rec,n=len(z),available=len(z),no_pose=0,successes=sum(x[2]for x in z),T_mean=float(np.mean([x[0]for x in z])),R_mean=float(np.mean([x[1]for x in z]))))
old={f['id']:f for f in rows(B/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz')};union=rows(R/'SYNTHETIC_v3/HEAD_FEATURES.jsonl.gz');preserved=0
for f in union:
 now={c['candidate_id']:c for c in f['candidates']}
 for c in old[f['id']]['candidates']:
  d=now[c['candidate_id']];assert d['actual_pose']==c['actual_pose']and d['eligible']==c['eligible']and d['features'][:26]==c['features'][:26]and d['features'][31:57]==c['features'][31:57];preserved+=1
norm=rd(R/'SYNTHETIC_v3/NORMALIZER.json');assert len(norm['training_frame_ids'])==768 and all(old[i]['split']=='train'for i in norm['training_frame_ids'])
cw(R/'ALL_METRICS.csv',metrics);cw(R/'RECORDING_METRICS.csv',recs);cw(R/'QUARTER_TURN_DIAGNOSTIC.csv',turns)
check=dict(status='PASS',independent_selected_pose_checks=n,max_T_error_cm=maxT,max_R_error_deg=maxR,all1003_old_candidate_poses_eligibility_first26_scalars_and_validity_bitwise_preserved_n=preserved,score_gaps_intentionally_recomputed=True,full_old_exporter_context_parity_independent_frames=8,not_all1003_full_exporter_equivalence=True,own_normalizer_TRAIN768_only=True,actual_new_H1fits=1,otherseed_fits=0,PRIMARYnew=0)
(R/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(check,indent=2)+'\n')
text='''v24: 원 forecast jackknife 후보 확장 + 원 H1 재학습

검증한 평균 격차: 원v5 C2와 같은235 합성 validation에서 성공101->104, T17.872301->15.383988cm/R9.206962->8.849067도, 회복4/손상1입니다. T>50 26->23/T>100 7->5, P90/P95/max 악화 없이 사전 gate를 통과했습니다. CAL46에서는 성공39유지, 회복0/손상0, T2.583822->2.454579cm, T P95 13.107490->11.603072cm/max17.766490->14.489192cm로 낮아졌지만 R5.242126->5.261341도/P90 3.937353->4.094960도로 올라 엄격 gate를 실패했습니다. tiny R악화라고 기준을 완화하지 않았으며 다른seed/PRIMARY는0입니다. 실사 동시평균 개선 및 외부9cm목표를 달성했다고 주장하지 않습니다.

원인 근거: v23은 원8 forecast를 한점씩 제외한7점SQPnP+LM으로 rawbank 성공포함133->153을 만들었지만 단순trim선택은99성공/24손상입니다. 이번은 원 hardadmission을 그대로 적용하고 실제확장 bank 특징·gap·mean/max/count·정규화를 재계산한 뒤 H1을 다시 학습했습니다. train768 성공포함434->502(추가68), VAL235 133->153(추가20)이 admission뒤에도 유지됩니다. VAL의 실현성공104/selection실패49와 bank oracle153을 분리합니다. 이 자료의 후보 부재를 줄일 경로는 실제 확인됐지만 새 실사192후보 성능은 아직 평가하지 않았습니다.

기존과 구분: 기존 CTRL_RANSAC은 corrected fullpoint와EPnP/원고정G13점수였고, v19는 선형화삭제였습니다. 이번 제안은 같은 원N0forecast의<=8leave-one-outSQPnP/LM, 고정물리family/Q/K/치수, RGB/SAM/검출기/상태/관측/CAL생산 재실행0입니다. 가려진 forecast를 실제 관측이나 owner정답으로 만들지 않았습니다. 이전v5–v23/원고/dirtysource는 보존했습니다. 새 후보를 원H1출력필터만 바꿔 넣은 민감도 검사가 아니며 새bankownTRAINnormalizer로 fresh동일seed188->64->64->1 C2binary frameBCE/AdamW.001 decay.0001/batch32/max50/patience5/strictlowerVALBCE를1fit했습니다. 원v5 actualcode 역시 freshinitializer임을 확인했고 matched기준선은 재사용했습니다. seed18 fit23epoch/best18 약5.6초입니다.

기술 검산: pilot8TRAIN fulloldexport가 scalar/validity/eligibility 모두정확0차이였습니다. 고정큰score-gap signedlog 역변환의 cancellation은1TRAIN/1CAL에서 검산을 실패해 원고정scorer로 해당프레임의 oldbankmin을 정확히 재계산했습니다. 오차허용을 완화하지 않았습니다. v1/v2 실패소스·로그/TECHNICAL_ABORT_V2.json은 보존했고 v3/SYNTHETIC_v3를 분리했습니다. 모든1003 oldcandidate의 pose/eligibility/first26scalar 및validity가 bitwise보존됨을 별도 확인했으며 gap26..30은 새bank에 맞게 바뀝니다. pilot8은 전체1003의 full exporter 독립 동치 검증은 아닙니다. 후보등록에는 exactphysicaldedup과 원모든mode중 feasible 규칙을 유지했습니다. 원래mask·관측·각family비용의 다른단위를 임의빼지 않았고 기존원exporter정의를 유지합니다.

비용: CPU1/GPU0. pilot oldfull검산14.98초, 반복실행구간8장 별도측정전체추정285초, 실제1003 unionexport306.05초/CAL46 22.02초였습니다. 실패한 v2prefix도비용에포함해야하며 TECHNICAL_ABORT의 TRAIN_PnP상한944입니다. v23 VAL2115쌍중후보1880을 재사용했고 v24추가 TRAIN PnP상한6144/CAL368/기술pilot128쌍입니다. 위값은 SQPnP+LM쌍이며 전체state/DNN추론시간이 아닙니다. 실제solver function evaluation160회 같은주장은하지않습니다. 기존fits/negative원장과 새fit1을혼동하지않습니다.

다음: bankheadroom을없다고종료하지않고 v25에서 후보생성원자료인 원8forecast의 개별signed residual구조가 선택기에 없는지검증합니다. v24의CAL사후선택6행을확인한적응적개발이력이며 독립일반화주장이아닙니다. 주석/COMMON/pre-CAL차단은해결됐다고하지않습니다.

실행: 공통prefix env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 [PRIVATE_PATH_OMITTED] -B . experiment.py pilot; experiment_v2.py cost; experiment_v2.py features(검산중단); experiment_v3.py features; experiment_v3.py targets; learn.py train; learn.py predict; learn.py evaluate; experiment_v3.py features --population CAL46; learn.py predict --population CAL46; learn.py evaluate --population CAL46. 기존출력은exclusive-create입니다. 새분리root와원자료/runtime/binding확인후재현합니다.

원저장소 HEAD7e92fdcefb0a37bdee0aef95d3e86c572e23b967, 과학소스base93c24d143fa3901de1fa127b3156e8250d81dba8. 새PRIMARY/독립실사/수동label0, 원고/main/기존실험숫자변경0입니다.
'''
(R/'REPORT_KO.md').write_text(text);print(json.dumps(check,indent=2))
