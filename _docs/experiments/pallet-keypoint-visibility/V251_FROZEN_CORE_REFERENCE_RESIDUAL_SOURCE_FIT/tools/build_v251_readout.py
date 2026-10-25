"""Plot only saved SOURCE metrics; optional posthoc scalar interpretation is separate."""
from pathlib import Path
import json,csv,html,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).absolute().parent
H=R/'V251_ROOT_REFERENCE_RESIDUAL'
D=H/'matched_reference_residual_v1'
MODELS=('BASE_FROZEN_V239','CONTROL_NO_REFERENCE','TREAT_REFERENCE')
LABELS=('Frozen core','Query residual','Reference residual')
DOMAINS=('SYN intended-native MAIN','YCB associated-r AUX')

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(x in str(p).lower() for x in ('v112','cache','.pyc')):raise ValueError(str(p))
 for a in reversed([p,*p.parents]):
  if a.is_symlink():raise ValueError('Artifact symlink')
 return p
def load(p):return json.loads(safe(p).read_bytes())
def write(p,s):
 with safe(p).open('x') as f:f.write(s)
def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def htable(headers,rows):
 return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def fmt(v):return 'NA' if v is None else f'{100*v:.2f}%'
def csvwrite(p,rows):
 with safe(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
 s=load(D/'SOURCE_RESULTS.json');assert s['completed'] and s['PASS']
 metrics={MODELS[0]:s[MODELS[0]],**s['arms']};rows=[]
 for model in MODELS:
  for family in (0,1):
   for role in ('FIT','DEV'):
    m=metrics[model][f'family{family}_{role}'];cm=m['CM_D_E_by_D_E_U'];n=sum(map(sum,cm))
    rows.append({'model':model,'family':family,'label_namespace':DOMAINS[family],'role':role,'original_rows':n,'D':sum(cm[0]),'E':sum(cm[1]),'D_to_D':cm[0][0],'D_to_E':cm[0][1],'D_to_U':cm[0][2],'E_to_D':cm[1][0],'E_to_E':cm[1][1],'E_to_U':cm[1][2],'precision_E':m['precisionE'],'recall_D':m['recallD'],'recall_E':m['recallE'],'balanced_recall':m['BR'],'E_logit_AUROC':m['AUC']['auc'],'model_U':m['U']})
 g=s['SOURCE_gate'];old=g['historical_SOURCE_TREAT_vs_BASE_FROZEN_V239']
 tests=[('TREAT−BASE mean family BR gain (pp)',old['mean_family_DEV_balanced_recall_gain_pp'],'>=',3.),('Maximum class recall drop vs BASE (pp)',old['max_family_class_recall_drop_pp'],'<=',2.),('YCB DEV E recall',old['YCB_DEV_E_recall'],'>=',.5),('SYN DEV D recall',old['SYN_DEV_D_recall'],'>=',.9),('SYN DEV E recall',old['SYN_DEV_E_recall'],'>=',.9),('TREAT−CONTROL mean family BR gain (pp)',g['matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp'],'>=',1.)]
 gate=[{'criterion':name,'actual':v,'operator':op,'threshold':t,'PASS':v is not None and (v>=t if op=='>=' else v<=t)} for name,v,op,t in tests]
 assert all(r['PASS'] for r in gate)==g['PASS']
 csvwrite(H/'SOURCE_COMPARISON.csv',rows);csvwrite(H/'SOURCE_GATE.csv',gate)
 colors=('#82919d','#397da5','#d47d44');fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
 for family,ax in enumerate(axes):
  for k,model in enumerate(MODELS):
   m=metrics[model][f'family{family}_DEV'];values=[m['recallD'],m['recallE']];x=np.arange(2)+(k-1)*.24
   ax.bar(x,values,width=.23,label=LABELS[k],color=colors[k])
   for xx,v in zip(x,values):ax.text(xx,v+.018,f'{100*v:.1f}%',ha='center',fontsize=9)
  cm=metrics[MODELS[0]][f'family{family}_DEV']['CM_D_E_by_D_E_U'];ax.set_title(f'{DOMAINS[family]}\nDEV D={sum(cm[0])}, E={sum(cm[1])}');ax.set_xticks([0,1],['Visible D recall','External E recall']);ax.set_ylim(0,1.14);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axes[0].set_ylabel('Recall at fixed raw E-logit >= 0');axes[1].legend(fontsize=9,loc='center right');fig.suptitle('V251 actual matched SOURCE fit: fixed threshold, all original DEV rows')
 fig.savefig(safe(H/'MODEL_DEV_RECALL.png'),dpi=165);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,7.1),constrained_layout=True)
 for family in (0,1):
  vmax=max(max(map(max,metrics[model][f'family{family}_DEV']['CM_D_E_by_D_E_U'])) for model in MODELS)
  for k,model in enumerate(MODELS):
   ax=axes[family,k];cm=np.asarray(metrics[model][f'family{family}_DEV']['CM_D_E_by_D_E_U']);ax.imshow(cm,vmin=0,vmax=vmax,cmap='Blues')
   for i in range(2):
    for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>.5*vmax else '#1f2933',fontsize=13)
   ax.set_xticks(range(3),['D','E','U']);ax.set_yticks(range(2),['D','E']);ax.set_xlabel('Prediction');ax.set_ylabel('Truth');ax.set_title(LABELS[k]+'\n'+DOMAINS[family])
 fig.suptitle('SOURCE DEV confusion matrices: SYN / YCB labels remain separate')
 fig.savefig(safe(H/'SOURCE_CONFUSION_MATRICES.png'),dpi=165);plt.close(fig)
 hdr=['모델','라벨','분할','원 D/E','E precision','D recall','E recall','BR','E AUROC']
 tab=[[r['model'],r['label_namespace'],r['role'],f"{r['D']}/{r['E']}",fmt(r['precision_E']),fmt(r['recall_D']),fmt(r['recall_E']),fmt(r['balanced_recall']),'NA' if r['E_logit_AUROC'] is None else f"{r['E_logit_AUROC']:.4f}"] for r in rows]
 gh=['조건','실측','비교','기준','결과'];gt=[[r['criterion'],f"{r['actual']:.6f}" if r['actual'] is not None else 'NA',r['operator'],r['threshold'],'PASS' if r['PASS'] else 'FAIL'] for r in gate]
 lead='고정 문턱의 SOURCE 판정은 개선되지 않았습니다. 참조 모델과 대조군의 DEV 혼동행렬이 동일하고, 공개 실사 YCB 외부 가림은 두 모델 모두 0/57개를 검출했습니다. 합성 DEV 가림은 50/56개(89.29%)로 사전 기준90%에 못 미칩니다. 전체 SOURCE gate는 FAIL이며 현재 평가 진입과 모델 채택은 없습니다.'
 method='기존 V239 core 86,054개 파라미터를 동결하고, 같은 query hidden64에 작은 residual MLP(73→32→2, 학습 파라미터2,434개)를 연결했습니다. CONTROL은 참조9값을0으로, TREAT는 세 특징의 선택/대안 참조 cosine6개와 지원3개를 사용했습니다. 같은 초기값·PCG17 draw ledger·AdamW·1,000회 업데이트로 두 모델을 실제 학습했습니다. 두 최종 checkpoint·로그·bank를 모두 봉인한 뒤 DEV를 평가했습니다.'
 banks='합성 참조는 SOURCE FIT-D의 기존 native 입력 주소0..7별 평균 특징이며 목재/정사각형 class나 CAD ID를 가정하지 않았습니다. 주소는 깨끗한 물리 부위 template이라는 인증이 아닙니다. 다음 주소는 고정 비교 참조이며 가림 음성 정답이 아닙니다. YCB는 기존 FIT-D 감독 bank4개를 재사용하고 다른 q8 노드로 참조를 선택했습니다. query의 정답 object ID는 선택·head 입력에 넣지 않았습니다.'
 finding='기존 core는 YCB FIT의 D841/E283개를 이미 모두 맞혔지만 DEV에서는 E1/57개만 맞혔습니다. 새 참조 residual도 FIT100%와 DEV 실패가 이어집니다. 학습점의 기존 logit이 이미 정답을 맞히는 구조에서는 새로운 참조를 활용하도록 학습할 유인이 부족했을 가능성이 있습니다. 이는 원인 가설이며, 저장 모델의 참조 입력 개입 결과는 아래 별도 posthoc 진단으로 구분합니다.'
 limits='SYN intended-native-address MAIN과 YCB associated-r3×3 AUX는 다른 라벨 권한입니다. YCB native·SELF 상태는 UNKNOWN이고, q는 정답 r 주변≤4px jitter를 모사했습니다. 반복 노출 SOURCE DEV와 이미 본 네 YCB class에서 얻은 결과이며 실제 팔레트 전이를 입증하지 않습니다. 기존 S/U7,423행은 binary loss에서만 제외되며 context의 입력 지원 노드를 의미 라벨로 지우지 않았습니다. 보류는 원 입력 가용성으로만 결정했고 이번 known SOURCE의 모델 U와 참조 UNKNOWN은0입니다.'
 scope='새 DINO encoder 학습·forward·공식 backbone 가중치 읽기, 새 RGB/depth/annotation archive 읽기, 현재/보호 평가 자료 사용, PnP·좌표 교정은 없습니다. 기존 SOURCE core checkpoint는 실제 읽었고, 합성 FIT-D 참조 통계와 두 residual은 실제 학습했습니다. SOURCE gate의 기준이나 문턱0을 결과에 맞춰 바꾸지 않았습니다.'
 links=[('학습 전 고정 명세','artifacts/V251_ROOT_REFERENCE_RESIDUAL/PROTOCOL_BEFORE_REFERENCE_FIT_PAYLOADS.json'),('모든 원 점수·checkpoint·로그의 COMPLETE','artifacts/V251_ROOT_REFERENCE_RESIDUAL/matched_reference_residual_v1/COMPLETE.json'),('Source 결과 원 JSON','artifacts/V251_ROOT_REFERENCE_RESIDUAL/matched_reference_residual_v1/SOURCE_RESULTS.json'),('두 checkpoint의 DEV 이전 봉인','artifacts/V251_ROOT_REFERENCE_RESIDUAL/matched_reference_residual_v1/BOTH_FINAL1000_LOGS_BANKS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json'),('독립 수학·로그·bank 검산','artifacts/V251_ACTUAL_REFERENCE_FIT_QA_C/actual_review_v1/REVIEW.json'),('학습 코드','artifacts/V251_FROZEN_CORE_REFERENCE_TRAINER_B/train_reference_residual_v2.py'),('공개 저장 모델 추론 재현','artifacts/V251_PUBLIC_REPLAY_A/verify_saved_fit.py'),('모든 FIT/DEV 수치 CSV','SOURCE_COMPARISON.csv'),('고정 gate CSV','SOURCE_GATE.csv')]
 post=load(H/'POSTHOC_PUBLIC_READOUT.json')
 ph=post['headers'];pr=post['rows'];pp=post['explanation_ko']
 links.append(('Posthoc 참조 입력 개입 원 결과','artifacts/V251_ROOT_REFERENCE_RESIDUAL/reference_effect_posthoc_v1/SUMMARY.json'))
 paras=[lead,method,banks,finding,limits,scope]
 md='# V251 — 동결 core에 참조 residual을 학습한 실제 SOURCE 비교\n\n'+'\n\n'.join(paras)+'\n\n## 모든 FIT·DEV 결과\n\n'+table(hdr,tab)+'\n![고정 문턱 recall 비교](MODEL_DEV_RECALL.png)\n\n![원 분모 혼동행렬](SOURCE_CONFUSION_MATRICES.png)\n\n## 사전에 고정한 SOURCE gate\n\n'+table(gh,gt)+'\n## 별도 posthoc 저장 모델 진단\n\n'+pp+'\n\n'+table(ph,pr)+'\n## 검토·재현 자료\n\n'+'\n'.join(f'- [{a}]({b})' for a,b in links)+'\n'
 write(H/'PUBLIC_README_KO.md',md)
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V251 SOURCE 참조 head 학습 결과</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:35px auto;padding:22px;color:#1b3038;background:#f7f9fb}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px;border:1px solid #d8e1e5;white-space:nowrap}th{background:#e5edf2}h2{margin-top:35px}a{color:#126a86}</style><h1>V251: 작은 참조 head의 실제 학습 결과 — SOURCE gate FAIL</h1>'+''.join('<p>'+html.escape(p)+'</p>' for p in paras)+'<h2>모든 FIT·DEV 결과</h2>'+htable(hdr,tab)+'<img src="MODEL_DEV_RECALL.png" alt="고정 문턱 recall"><img src="SOURCE_CONFUSION_MATRICES.png" alt="SOURCE 혼동행렬"><h2>고정 SOURCE gate</h2>'+htable(gh,gt)+'<h2>별도 posthoc 참조 개입</h2><p>'+html.escape(pp)+'</p>'+htable(ph,pr)+'<h2>검토·재현 자료</h2><ul>'+''.join('<li><a href="'+b+'">'+html.escape(a)+'</a></li>' for a,b in links)+'</ul></html>'
 write(H/'ROOT_READOUT_KO.html',page)
 print(json.dumps({'completed':True,'PASS':True,'primary_rows':len(rows),'SOURCE_gate_PASS':g['PASS'],'new_model_training_by_readout':0}))

if __name__=='__main__':run()
