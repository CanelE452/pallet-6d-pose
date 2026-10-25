"""Render saved SOURCE results; no model, threshold or data selection."""
from pathlib import Path
import json, csv, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).absolute().parent
H=R/'V252_ROOT_DIRECT_REFERENCE'
D=H/'matched_direct_reference_v1'
MODELS=('BASE_FROZEN_V239','CONTROL_GENERIC_REFERENCE','TREAT_TARGET_REFERENCE')
LABELS=('Saved V239 baseline','Generic reference direct','Target reference direct')
DOMAINS=('SYN intended-native MAIN','YCB associated-r AUX')

def safe(p):
    p=Path(p)
    if not p.is_absolute() or '..' in p.parts or any(t in str(p).lower() for t in ('v112','cache','.pyc')): raise ValueError(str(p))
    for a in reversed([p,*p.parents]):
        if a.is_symlink(): raise ValueError('Artifact symlink')
    return p

def load(p): return json.loads(safe(p).read_bytes())
def write(p,b):
    with safe(p).open('xb') as f: f.write(b.encode() if isinstance(b,str) else b)
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def htable(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(t))+'</th>' for t in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def pct(v): return 'NA' if v is None else f'{100*v:.2f}%'
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
                pd=cm[0][0]/(cm[0][0]+cm[1][0]) if cm[0][0]+cm[1][0] else None
                rows.append(dict(model=model,family=family,label_namespace=DOMAINS[family],role=role,original_rows=n,D=sum(cm[0]),E=sum(cm[1]),D_to_D=cm[0][0],D_to_E=cm[0][1],D_to_U=cm[0][2],E_to_D=cm[1][0],E_to_E=cm[1][1],E_to_U=cm[1][2],precision_D=pd,precision_E=m['precisionE'],recall_D=m['recallD'],recall_E=m['recallE'],balanced_recall=m['BR'],E_logit_AUROC=m['AUC']['auc'],model_U=m['U']))
    g=s['SOURCE_gate'];old=g['historical_SOURCE_TREAT_vs_BASE_FROZEN_V239']
    tests=[('TREAT−BASE mean family BR gain (pp)',old['mean_family_DEV_balanced_recall_gain_pp'],'>=',3.),('Maximum class recall drop vs BASE (pp)',old['max_family_class_recall_drop_pp'],'<=',2.),('YCB DEV E recall',old['YCB_DEV_E_recall'],'>=',.5),('SYN DEV D recall',old['SYN_DEV_D_recall'],'>=',.9),('SYN DEV E recall',old['SYN_DEV_E_recall'],'>=',.9),('TREAT−CONTROL mean family BR gain (pp)',g['matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp'],'>=',1.)]
    gate=[dict(criterion=name,actual=v,operator=op,threshold=t,PASS=v is not None and (v>=t if op=='>=' else v<=t)) for name,v,op,t in tests]
    assert all(x['PASS'] for x in gate)==g['PASS']
    csvwrite(H/'SOURCE_COMPARISON.csv',rows);csvwrite(H/'SOURCE_GATE.csv',gate)
    colors=('#82919d','#397da5','#d47d44')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
    for family,ax in enumerate(axes):
        for k,model in enumerate(MODELS):
            m=metrics[model][f'family{family}_DEV'];values=[m['recallD'],m['recallE']];xx=np.arange(2)+(k-1)*.24
            ax.bar(xx,values,width=.23,label=LABELS[k],color=colors[k])
            for x,v in zip(xx,values):ax.text(x,v+.018,f'{100*v:.1f}%',ha='center',fontsize=9)
        cm=metrics[MODELS[0]][f'family{family}_DEV']['CM_D_E_by_D_E_U']
        ax.set_title(f'{DOMAINS[family]}\nDEV D={sum(cm[0])}, E={sum(cm[1])}');ax.set_xticks([0,1],['Visible D recall','External E recall']);ax.set_ylim(0,1.14);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Recall at fixed raw E-logit >= 0');fig.legend(*axes[1].get_legend_handles_labels(),fontsize=9,loc='lower center',bbox_to_anchor=(.5,-.06),ncol=3);fig.suptitle('V252 actual direct-reference SOURCE fit: all original DEV rows')
    fig.savefig(safe(H/'MODEL_DEV_RECALL.png'),dpi=165,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7.1),constrained_layout=True)
    for family in (0,1):
        vmax=max(max(map(max,metrics[model][f'family{family}_DEV']['CM_D_E_by_D_E_U'])) for model in MODELS)
        for k,model in enumerate(MODELS):
            ax=axes[family,k];cm=np.asarray(metrics[model][f'family{family}_DEV']['CM_D_E_by_D_E_U']);ax.imshow(cm,vmin=0,vmax=vmax,cmap='Blues')
            for i in range(2):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>.5*vmax else '#1f2933',fontsize=13)
            ax.set_xticks(range(3),['D','E','U']);ax.set_yticks(range(2),['D','E']);ax.set_xlabel('Prediction');ax.set_ylabel('Truth');ax.set_title(LABELS[k]+'\n'+DOMAINS[family])
    fig.suptitle('V252 SOURCE DEV confusion matrices: separate label namespaces')
    fig.savefig(safe(H/'SOURCE_CONFUSION_MATRICES.png'),dpi=165);plt.close(fig)
    hdr=['모델','라벨','분할','원 D/E','D precision','E precision','D recall','E recall','BR','E AUROC']
    tab=[[r['model'],r['label_namespace'],r['role'],f"{r['D']}/{r['E']}",pct(r['precision_D']),pct(r['precision_E']),pct(r['recall_D']),pct(r['recall_E']),pct(r['balanced_recall']),'NA' if r['E_logit_AUROC'] is None else f"{r['E_logit_AUROC']:.4f}"] for r in rows]
    gh=['조건','실측','비교','기준','결과'];gt=[[r['criterion'],f"{r['actual']:.6f}" if r['actual'] is not None else 'NA',r['operator'],r['threshold'],'PASS' if r['PASS'] else 'FAIL'] for r in gate]
    tr=metrics[MODELS[2]];sy=tr['family0_DEV'];yc=tr['family1_DEV']
    lead=f"고정 조건의 direct head 학습을 완료했습니다. 대상 참조 모델의 SYN DEV 보임 recall은 {pct(sy['recallD'])}, 외부 가림 recall은 {pct(sy['recallE'])}입니다. 공개 실사 YCB AUX의 보임 recall은 {pct(yc['recallD'])}, 외부 가림 recall은 {pct(yc['recallE'])}입니다. 사전 SOURCE gate는 {'PASS' if g['PASS'] else 'FAIL'}이며, 현재 팔레트 평가 결과를 뜻하지 않습니다."
    cy=metrics[MODELS[1]]['family1_DEV'];bc=metrics[MODELS[0]]['family0_DEV'];by=metrics[MODELS[0]]['family1_DEV']
    finding=f"공개 실사 YCB DEV의 E 검출은 저장 BASE {by['CM_D_E_by_D_E_U'][1][1]}/57에서 TREAT {yc['CM_D_E_by_D_E_U'][1][1]}/57로 늘었습니다. 그러나 SYN DEV의 보이는 D를 E로 오판한 수는 {bc['CM_D_E_by_D_E_U'][0][1]}/1288에서 {sy['CM_D_E_by_D_E_U'][0][1]}/1288로 늘었고, TREAT E precision은 SYN {pct(sy['precisionE'])}, YCB {pct(yc['precisionE'])}입니다. 일반 참조 CONTROL의 YCB E recall은 {pct(cy['recallE'])}, E precision은 {pct(cy['precisionE'])}로 대상 참조보다 높습니다. TREAT−CONTROL 평균 family DEV BR 차이는 {g['matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp']:.3f}pp입니다. 이번 구성에서 공개 실사 가림 신호를 더 읽었지만 대상 참조의 추가 효과나 보임·가림 동시 개선은 입증되지 않았습니다."
    method='기존 모델의 hidden·logit·checkpoint를 입력이나 초기값에 쓰지 않았습니다. 참조 비교9값과 원 condition32값에서 MLP41→32→2(학습 파라미터1,410개)가 두 라벨 권한의 출력을 직접 예측합니다. 일반 참조 CONTROL과 대상 참조 TREAT는 같은 초기값·최종층0·PCG17 draw ledger·AdamW·1,000회 업데이트를 사용합니다. 두 최종 checkpoint와 로그·bank·정책을 모두 봉인한 뒤 새 DEV 추론과 기존 BASE CSV 비교를 시작했습니다.'
    bank='CONTROL도 감독 학습 참조입니다. SOURCE FIT-D의 원 특징을 각 family·세 view별로 동등 가중 평균(FP64, 원 행 순서)한 뒤 L2 정규화했습니다. 선택/대안 슬롯에 같은 일반 참조 cosine을 넣습니다. TREAT는 V251의 고정 대상 참조9값을 재사용합니다. SYN은 기존 native 입력 주소별 참조이고, YCB는 정답 object ID 없이 다른 초기 q8 노드로 선택한 참조입니다. 세 view를 함께 쓰며 view·문턱·epoch 선택은 없습니다.'
    inclusion='두 참조의 FIT 통계에는 그 FIT-D 학습점 자체가 포함될 수 있습니다. DEV는 참조 통계와 loss에 기여하지 않습니다. 따라서 이 비교는 두 학습된 표현의 효과를 비교하며 참조 대상 정체성만의 순수 인과 효과를 뜻하지 않습니다. 합성 주소별 참조가 깨끗한 물리 부위 template이라는 인증도 아닙니다.'
    limits='SYN intended-native-address MAIN과 YCB associated-r3×3 AUX의 라벨 정의를 합치지 않습니다. YCB native·SELF는 UNKNOWN이며 q는 정답 주변 jitter를 모사합니다. 자기 가림은 upstream 기하로 처리한다는 현재 목표를 유지합니다. 기존 S/U7,423행은 binary loss에서만 제외되며 E 또는 성공 보류로 계산하지 않습니다. 이번 known SOURCE7,617행의 모델 보류 U는0개(0%)이고 두 모델 모두 세 참조 view를100% 지원합니다. 반복 노출 SOURCE DEV 성능으로 새 실사 팔레트 전이를 단정할 수 없습니다.'
    scope='이번에는 합성·공개 실사에서 저장된 특징만 사용했습니다. 새 encoder·backbone·RGB/depth/annotation archive·현재 또는 보호 평가 데이터·PnP·좌표 교정은 사용하지 않았습니다. 기존 BASE는 저장 CSV를 최종 봉인 뒤 읽어 비교했습니다. 모델 채택과 현재 평가 진입은 고정 입력·라벨의 배포 가능성을 감사한 뒤 판단할 다음 단계입니다.'
    links=[('학습 전 고정 명세','artifacts/V252_ROOT_DIRECT_REFERENCE/PROTOCOL_BEFORE_DIRECT_HEAD_FIT.json'),('전체 산출물 COMPLETE','artifacts/V252_ROOT_DIRECT_REFERENCE/matched_direct_reference_v1/COMPLETE.json'),('원 SOURCE 결과','artifacts/V252_ROOT_DIRECT_REFERENCE/matched_direct_reference_v1/SOURCE_RESULTS.json'),('DEV 이전 두 모델 봉인','artifacts/V252_ROOT_DIRECT_REFERENCE/matched_direct_reference_v1/BOTH_FINAL1000_LOGS_BANKS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json'),('모든 FIT·DEV 수치 CSV','SOURCE_COMPARISON.csv'),('고정 gate CSV','SOURCE_GATE.csv')]
    extra=load(H/'READOUT_VERIFIED_LINKS.json');links+=extra['links']
    paras=[lead,finding,method,bank,inclusion,limits,scope]
    md='# V252 — 일반 참조와 대상 참조에서 직접 보임·가림 판정\n\n'+'\n\n'.join(paras)+'\n\n## 모든 FIT·DEV 결과\n\n'+table(hdr,tab)+'\n![고정 문턱 recall 비교](MODEL_DEV_RECALL.png)\n\n![원 분모 혼동행렬](SOURCE_CONFUSION_MATRICES.png)\n\n## 사전 SOURCE gate\n\n'+table(gh,gt)+'\n## 검토·재현 자료\n\n'+'\n'.join(f'- [{a}]({b})' for a,b in links)+'\n'
    write(H/'PUBLIC_README_KO.md',md)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V252 direct reference head</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:35px auto;padding:22px;color:#1b3038;background:#f7f9fb}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px;border:1px solid #d8e1e5;white-space:nowrap}th{background:#e5edf2}h2{margin-top:35px}a{color:#126a86}</style><h1>V252: 참조에서 직접 판정한 SOURCE 결과</h1>'+''.join('<p>'+html.escape(p)+'</p>' for p in paras)+'<h2>모든 FIT·DEV 결과</h2>'+htable(hdr,tab)+'<img src="MODEL_DEV_RECALL.png" alt="고정 문턱 recall"><img src="SOURCE_CONFUSION_MATRICES.png" alt="SOURCE 혼동행렬"><h2>고정 SOURCE gate</h2>'+htable(gh,gt)+'<h2>검토·재현 자료</h2><ul>'+''.join('<li><a href="'+b+'">'+html.escape(a)+'</a></li>' for a,b in links)+'</ul></html>'
    write(H/'ROOT_READOUT_KO.html',page)
    print(json.dumps(dict(completed=True,PASS=True,primary_rows=len(rows),SOURCE_gate_PASS=g['PASS'],new_model_training_by_readout=0)))

if __name__=='__main__': run()
