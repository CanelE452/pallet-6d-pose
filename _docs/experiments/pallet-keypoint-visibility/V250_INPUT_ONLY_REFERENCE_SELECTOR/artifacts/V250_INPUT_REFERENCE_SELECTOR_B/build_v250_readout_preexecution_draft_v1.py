#!/usr/bin/env python3
"""Render saved V250 SOURCE scalar results. No SOURCE arrays/model/RGB inputs."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

R=Path(__file__).absolute().parent
VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')
NAMES=('PUBLIC_README_KO.md','ROOT_READOUT_KO.html','AUC_COMPARISON.csv','REFERENCE_SELECTION.csv','AUTO_ORACLE_AUC.png','REFERENCE_SELECTION_CONFUSION.png')


def guard(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any('v112' in a.lower() or 'cache' in a.lower() or a.lower().endswith('.pyc') for a in p.parts):raise ValueError('Forbidden path before FS')
    for a in reversed([p,*p.parents]):
        if a.is_symlink():raise ValueError('Symlink artifact/ancestor')
    return p


def sha(path):
    p=guard(path);h=hashlib.sha256();size=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':size}


def jread(path):return json.loads(guard(path).read_text(encoding='utf-8'))
def pct(v):return 'NA' if v is None else f'{100*v:.2f}%'
def dec(v):return 'NA' if v is None else f'{v:.4f}'
def div(n,d):return n/d if d else None


def csvwrite(path,rows):
    with guard(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def table_md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def table_html(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def build(saved_dir,output_dir):
    src=guard(saved_dir);out=guard(output_dir)
    if not out.is_dir():raise ValueError('Existing Root metadata output directory required')
    if any(guard(out/n).exists() for n in NAMES):raise ValueError('Six report leaves must all be fresh; preserve existing metadata')
    # These are saved scalar outputs only, following no original SOURCE pointers.
    summary=jread(src/'SUMMARY.json');tables=jread(src/'DESCRIPTIVE_TABLES.json')['tables']
    with guard(src/'DEV1191_AUTO_ORACLE_COMPARISON.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if summary['counts']['YCB_prediction_queries']!=1521 or summary['counts']['input_only_view_scores']!=4563 or summary['counts']['DEV_queries']!=397 or len(rows)!=1191:raise ValueError('Fixed 1521/4563/397/1191 population')
    if summary['current_admission'] is not False:raise ValueError('No current admission')
    by_view={v:[r for r in rows if r['view']==v] for v in VIEWS}
    for v,rs in by_view.items():
        if len(rs)!=397 or len({r['sample_index'] for r in rs})!=397 or sum(r['target']=='0' for r in rs)!=340 or sum(r['target']=='1' for r in rs)!=57 or any(r['role']!='DEV' for r in rs):raise ValueError('Saved fixed DEV composition')
    first=by_view[VIEWS[0]]
    identities={r['sample_index']:(r['row_id'],r['target'],r['object_id'],r['selected_bank_index'],r['selection_correct']) for r in first}
    for rs in by_view.values():
        if any(identities[r['sample_index']]!=(r['row_id'],r['target'],r['object_id'],r['selected_bank_index'],r['selection_correct']) for r in rs):raise ValueError('Selection is shared across descriptor views')
    auc_rows=[]
    for st in ('all','<=3px'):
        for view in VIEWS:
            t=next(t for t in tables if t['view']==view and t['q_r_stratum']==st and t['object_id'] is None)
            for metric in ('negative_cosine','negative_margin'):
                z=t[metric]
                r={'view':view,'q_r_stratum':st,'E_direction_fixed':metric,'original_rows':t['rows'],'original_D':t['D'],'original_E':t['E'],'query_UNKNOWN':t['query_UNKNOWN'],'same_support_rows':z['same_support_rows']}
                for name in ('auto','same_support_auto','same_support_oracle','full_oracle'):
                    r[name+'_AUROC']=z[name]['auc'];r[name+'_D_supported']=z[name]['D_supported'];r[name+'_E_supported']=z[name]['E_supported']
                auc_rows.append(r)
    selection=[]
    for view,rs in by_view.items():
        for state in ('ALL','D','E'):
            rr=rs if state=='ALL' else [r for r in rs if r['target']==('0' if state=='D' else '1')]
            sp=[r for r in rr if r['selected_bank_index']!=''];correct=sum(r['selection_correct']=='True' for r in sp);unknown=len(rr)-len(sp);qu=sum(r['auto_supported']!='True' for r in rr)
            selection.append({'view':view,'teacher_state_subset':state,'original_queries':len(rr),'selector_supported':len(sp),'selector_correct':correct,'selector_accuracy_supported':div(correct,len(sp)),'selector_correct_over_original_denominator':div(correct,len(rr)),'selector_UNKNOWN':unknown,'selector_UNKNOWN_rate':div(unknown,len(rr)),'query_score_UNKNOWN':qu,'query_score_UNKNOWN_rate':div(qu,len(rr)),'authority':'CLOSED4BANK_NOMINAL_OBJECT_SELECTION_NOT_VISIBILITY_ACCURACY'})
    overall=selection[0]
    if overall['selector_correct']!=summary['selection_pooled']['correct'] or overall['selector_UNKNOWN']!=summary['selection_pooled']['selector_UNKNOWN']:raise ValueError('Selection arithmetic vs summary')
    objects=sorted({int(r['object_id']) for r in first})
    if len(objects)!=4:raise ValueError('Expected closed4 nominal bank objects')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(16,6.7),sharey=True)
    colors=('#176b87','#df8354','#765899')
    for ax,st in zip(axes,('all','<=3px')):
        rr=[r for r in auc_rows if r['q_r_stratum']==st];x=np.arange(6)
        for off,key,label,color in zip((-.25,0,.25),('same_support_auto','same_support_oracle','full_oracle'),('Auto: same support','Oracle: same support','Oracle: full denominator'),colors):
            vals=[np.nan if r[key+'_AUROC'] is None else r[key+'_AUROC'] for r in rr]
            ax.bar(x+off,vals,width=.23,label=label,color=color)
        ax.set_xticks(x,[r['view']+'\n'+('−cosine' if r['E_direction_fixed']=='negative_cosine' else '−margin') for r in rr],rotation=22,ha='right')
        ax.set_ylim(0,1.04);ax.axhline(.5,color='#777',lw=.7,ls='--');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
        ax.set_title('DEV '+st+f" | N={rr[0]['original_rows']} D={rr[0]['original_D']} E={rr[0]['original_E']}")
    axes[0].set_ylabel('E AUROC (ranking; no threshold or classifier)')
    fig.suptitle('V250: all three predeclared views; no best-view selection',fontsize=14,y=.98)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015));fig.subplots_adjust(bottom=.29,top=.87,wspace=.12)
    fig.savefig(out/'AUTO_ORACLE_AUC.png',dpi=170,facecolor='white');plt.close(fig)
    cm=np.zeros((4,5),np.int64)
    for r in first:
        col=4 if r['selected_object_id']=='' else objects.index(int(r['selected_object_id']))
        cm[objects.index(int(r['object_id'])),col]+=1
    fig,ax=plt.subplots(figsize=(9,6.7));im=ax.imshow(cm,cmap='Blues',vmin=0)
    ax.set_xticks(range(5),[str(o) for o in objects]+['UNKNOWN']);ax.set_yticks(range(4),list(map(str,objects)))
    for i in range(4):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()*.55 else '#19252b')
    ax.set_xlabel('Selected frozen bank nominal object ID');ax.set_ylabel('Late teacher nominal object ID')
    ax.set_title(f"Closed4-bank selection | {overall['selector_correct']}/397 correct ({pct(overall['selector_accuracy_supported'])})\nNot native/physical-instance or visibility accuracy",pad=16)
    fig.colorbar(im,ax=ax,label='DEV query count',shrink=.75);fig.tight_layout(pad=1.8);fig.savefig(out/'REFERENCE_SELECTION_CONFUSION.png',dpi=170,facecolor='white');plt.close(fig)
    csvwrite(out/'AUC_COMPARISON.csv',auc_rows);csvwrite(out/'REFERENCE_SELECTION.csv',selection)
    ah=['분모','view','E 점수 방향','Auto 동일지원','Oracle 동일지원','Oracle 전체','D/E (원분모)','지원 행']
    ar=[[r['q_r_stratum'],r['view'],r['E_direction_fixed'],dec(r['same_support_auto_AUROC']),dec(r['same_support_oracle_AUROC']),dec(r['full_oracle_AUROC']),f"{r['original_D']}/{r['original_E']}",r['same_support_rows']] for r in auc_rows]
    sh=['view','부분집합','원분모','선택 정답/지원','선택 정확도','선택 UNKNOWN','선택 UNKNOWN율','query 점수 UNKNOWN/율']
    sr=[[r['view'],r['teacher_state_subset'],r['original_queries'],f"{r['selector_correct']}/{r['selector_supported']}",pct(r['selector_accuracy_supported']),r['selector_UNKNOWN'],pct(r['selector_UNKNOWN_rate']),f"{r['query_score_UNKNOWN']} / {pct(r['query_score_UNKNOWN_rate'])}"] for r in selection]
    # Public artifact paths only. Exact existing metadata links are checked at Root execution;
    # never follow any absolute pointer contained in an artifact.
    result_prefix='artifacts/V250_ROOT_INPUT_REFERENCE/input_reference_selector_v1/'
    links=[('전체 strata·object별 표',result_prefix+'DESCRIPTIVE_TABLES.json'),('저장 요약',result_prefix+'SUMMARY.json'),('DEV1191 late join',result_prefix+'DEV1191_AUTO_ORACLE_COMPARISON.csv'),('예측 선봉인',result_prefix+'PREDICTIONS_FROZEN_BEFORE_TARGET_ROLE_ID_OR_ORACLE_READ.json'),('생성 완료·접근 범위',result_prefix+'COMPLETE.json'),('AUC CSV','AUC_COMPARISON.csv'),('선택·UNKNOWN CSV','REFERENCE_SELECTION.csv')]
    metadata=[('사전 고정 protocol','V250_ROOT_INPUT_REFERENCE/PROTOCOL_BEFORE_SELECTOR_PAYLOADS.json'),('선택기 CODE','V250_INPUT_REFERENCE_SELECTOR_B/select_input_reference.py'),('독립 CODE QA','V250_SELECTOR_CODE_QA_C/REVIEW.json'),('독립 저장값 QA','V250_ACTUAL_SELECTOR_QA_C/actual_review_v1/REVIEW.json'),('입력 경계·dependency closure','V250_REFERENCE_SELECTOR_BOUNDARY_A/dependency_closure.json'),('공개 재현 wrapper','V250_PUBLIC_REPLAY_A/replay_public_selector.py'),('공개 dependency map','V250_PUBLIC_REPLAY_A/public_dependencies.json')]
    for label,relative in metadata:
        if guard(R/relative).is_file():links.append((label,'artifacts/'+relative))
    ec=Counter(int(r['object_id']) for r in first if r['target']=='1');obj,n=ec.most_common(1)[0]
    limit=f'네 개의 이미 본 SOURCE 명목 object bank를 비교한 진단입니다. E {n}/57개가 object {obj}에 집중되어 있으며, 이는 독립 물리 instance {n}개나 자연 가림 일반화를 뜻하지 않습니다. q는 associated-r 주변 GT-jitter emulation이고 physical pallet native·SELF 권한은 UNKNOWN입니다. SOURCE DEV는 반복 노출되었습니다.'
    lead=f"입력의 다른 q8 RGB 노드로 동결 reference bank를 선택했습니다. DEV397(D340/E57)에서 명목 object 선택은 {overall['selector_correct']}/397, 지원 행 정확도 {pct(overall['selector_accuracy_supported'])}, 선택 UNKNOWN {overall['selector_UNKNOWN']}/397입니다. 이것은 closed4-bank 선택 정확도이며 D/E visibility 정확도가 아닙니다. 전체 입력 예측1521개·세 view 점수4563개를 정답/역할/object ID/oracle 판독 전에 봉인하고 DEV1191개 view 행을 후결합했습니다."
    method='active 슬롯을 제외하고 original_available8 & node_available & 유한한 비영 full-CLS만 사용했습니다. 최소 두 다른 노드가 필요하며, 네 bank별 top2 cosine 평균을 비교하고 수치 동률은 가장 작은 bank index로 정했습니다. query의 실제 명목 object ID는 선택 API에 들어가지 않았습니다. bank 자체는 기존 SOURCE FIT-D 감독 통계이며 이번에는 재적합하지 않았습니다.'
    interpretation='E 방향은 사전 고정된 −cosine 및 −margin입니다. 표의 Auto/Oracle 동일지원은 정확히 같은 지원 행을 비교하고 Oracle 전체는 원래 고정 분모를 보존합니다. 모든 세 view와 all/≤3px를 함께 표시하며 다른 strata·object별 빈 클래스/NA는 연결된 JSON에 보존합니다. 기존 centre-CLS oracle margin AUROC 약0.9837은 비교용 reference이며 채택한 view·문턱·분류기가 아닙니다.'
    boundary='새 optimizer/encoder/head 학습, reference 재적합, neural forward, 원RGB·weight·annotation archive 읽기, PnP·좌표 교정은 없습니다. 저장 SOURCE feature와 감독 bank 및 late teacher 메타데이터로 기술 통계를 계산했습니다. UNKNOWN을 E로 바꾸지 않았고 성공적인 보류라고 해석하지 않습니다. threshold·best-view·모델 선택이나 현재 실사223 평가·성능 향상·배포 채택을 주장하지 않습니다.'
    md='# V250 입력 기반 SOURCE reference 선택 진단\n\n'+'\n\n'.join([lead,method,interpretation,limit,boundary])+'\n\n## 고정 AUROC 비교\n\n'+table_md(ah,ar)+'\n![세 view의 AUROC 비교](AUTO_ORACLE_AUC.png)\n\n## 선택 분모와 UNKNOWN\n\n'+table_md(sh,sr)+'\n![명목 object reference 선택 혼동표](REFERENCE_SELECTION_CONFUSION.png)\n\n## 저장 근거와 재현\n\n'+'\n'.join(f'- [{a}]({b})' for a,b in links)+'\n'
    guard(out/'PUBLIC_README_KO.md').write_text(md,encoding='utf-8')
    body=''.join('<p>'+html.escape(p)+'</p>' for p in [lead,method,interpretation,limit,boundary])+'<h2>고정 AUROC 비교</h2>'+table_html(ah,ar)+'<img src="AUTO_ORACLE_AUC.png" alt="All3 views auto and oracle AUC"><h2>선택 분모와 UNKNOWN</h2>'+table_html(sh,sr)+'<img src="REFERENCE_SELECTION_CONFUSION.png" alt="Closed4bank selection confusion"><h2>저장 근거와 재현</h2><ul>'+''.join('<li><a href="'+html.escape(b,quote=True)+'">'+html.escape(a)+'</a></li>' for a,b in links)+'</ul>'
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V250 SOURCE 입력 reference 진단</title><style>body{max-width:1200px;margin:36px auto;padding:0 22px;font:16px/1.7 system-ui;color:#18313a;background:#f6f9fa}h1{font-size:30px}h2{margin-top:38px}p{max-width:1100px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid #d7e2e6;text-align:left;white-space:nowrap}th{background:#e4edf1}.scroll{overflow:auto}img{max-width:100%;height:auto;background:white;margin:22px 0}a{color:#176b87}</style><body><h1>V250 입력 기반 SOURCE reference 선택 진단</h1>'+body+'</body></html>'
    guard(out/'ROOT_READOUT_KO.html').write_text(page,encoding='utf-8')
    print(json.dumps({'completed':True,'PASS':True,'source_scalar_files_only':[str(src/n) for n in ('SUMMARY.json','DESCRIPTIVE_TABLES.json','DEV1191_AUTO_ORACLE_COMPARISON.csv')],'outputs':{n:sha(out/n) for n in NAMES},'new_model_forward_or_training':0},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--saved-dir',default=str(R/'V250_ROOT_INPUT_REFERENCE/input_reference_selector_v1'));p.add_argument('--output-dir',default=str(R/'V250_ROOT_INPUT_REFERENCE'));a=p.parse_args();build(a.saved_dir,a.output_dir)


if __name__=='__main__':main()
