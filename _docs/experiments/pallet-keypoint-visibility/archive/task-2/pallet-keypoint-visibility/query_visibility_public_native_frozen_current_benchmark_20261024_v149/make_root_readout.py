"""Render completed, saved seven-model results only; no models or RGB."""
from pathlib import Path
import datetime, hashlib, html, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V=Path(__file__).resolve().parent
R=V/'saved_current_benchmark_evaluation_v1'
def read(p):return json.loads(p.read_text())
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def ratio(a,b):return a/b if b else None
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'

if __name__=='__main__':
    a=read(R/'CURRENT_FROZEN_BENCHMARK_RESULTS.json')
    seal=read(R/'EVALUATION_SEAL.json')
    review=read(V/'independent_saved_evaluation_review/REVIEW.json')
    assert a['completed'] and a['PASS'] and a['selected_model'] is None
    assert seal['completed'] and seal['PASS'] and review['completed'] and review['PASS']
    T=a['tables'];models=a['models'];assert models==['v61','legacy','C16','C29','C32','C33','C34']
    names={'v61':'v61','legacy':'기존 실사 CAL46','C16':'이전 C16','C29':'C29','C32':'시작 C32','C33':'대조 C33','C34':'공개 실사 추가 C34'}
    gn={'WOOD_PRIMARY105':'목재','SQUARE118':'정사각형'}
    aggregate=[]
    for model in models:
        rows=[r for r in T['metrics']if r['cohort']=='paired_same_query_RGB' and r['task']=='D_vs_E' and r['model']==model]
        keys=['N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']
        c={k:sum(r[k]for r in rows)for k in keys}
        dd,de,ed,ee=[c[k]for k in ['TP','FN_decided','FP','TN']]
        du=sum(c[k]for k in ['U_visible','out_of_task_S_visible','other_unresolved_visible']);eu=sum(c[k]for k in ['U_occluded','out_of_task_S_occluded','other_unresolved_occluded'])
        assert c['N']==1237 and c['visible_support']==1147 and c['occluded_support']==90
        aggregate.append(dict(model=model,CM_D_E_by_pred_D_E_U=[[dd,de,du],[ed,ee,eu]],D_precision=ratio(dd,dd+ed),D_recall=ratio(dd,1147),E_precision=ratio(ee,ee+de),E_recall=ratio(ee,90),U=ratio(du+eu,1237),counts=c))
    near={m:sum(r['pred_binary_hidden']for r in T['manual_near_D']if r['model']==m)for m in models}
    comparisons=[]
    for new in ['C33','C34']:
        for old in ['C16','C32']+(['C33']if new=='C34'else[]):
            rows=[r for r in T['paired_recovery_and_damage']if r['cohort']=='paired_same_query_RGB' and r['baseline']==old and r['new_model']==new]
            comparisons.append(dict(baseline=old,new_model=new,recovered=sum(r['baseline_wrong_new_correct']for r in rows),damaged=sum(r['baseline_correct_new_wrong']for r in rows)))
    c34=next(x for x in aggregate if x['model']=='C34');c33=next(x for x in aggregate if x['model']=='C33')
    outcome=f"공개 실사 추가 효과가 보임 보존에는 일부 전이됐지만 외부 가림 검출은 악화됐다. C34: 보임 {c34['counts']['TP']}/1147, 외부 가림 {c34['counts']['TN']}/90. 대조 C33: 보임 {c33['counts']['TP']}/1147, 외부 가림 {c33['counts']['TN']}/90. 즉 보임25개 회복과 외부 가림7개 손상이 함께 발생했다. 이전 C16 대비 보임40개 손실·외부 가림1개 추가 검출에 그쳐 전체 개선이나 해결 근거가 아니다."
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(17,5.2),layout='constrained')
    colors=['#9ca3af','#ba8c50','#2563eb','#8b5cf6','#db6674','#7591ac','#168778']
    labels=['v61','실사\nCAL46','C16','C29','C32','C33','C34']
    for ax,key,title in zip(axes[:2],['D_recall','E_recall'],['보이는 점을 보존 · D recall','외부 가림을 검출 · E recall']):
        vals=[100*r[key]for r in aggregate];bars=ax.bar(range(7),vals,color=colors)
        ax.bar_label(bars,labels=[f'{x:.1f}%'for x in vals],padding=3,fontsize=9);ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_title(title,pad=12);ax.set_xticks(range(7),labels);ax.spines[['top','right']].set_visible(False)
    bars=axes[2].bar(range(7),[near[m]for m in models],color=colors);axes[2].bar_label(bars,padding=3);axes[2].set_xticks(range(7),labels);axes[2].set_title('오차 ≤3px · 수동 보임376점\n가림으로 잘못 버린 개수',pad=12);axes[2].set_ylim(0,max(near.values())*1.15+2);axes[2].spines[['top','right']].set_visible(False)
    fig.suptitle('V149 · 공개 실사 학습 추가 후 현재 평가223장 · 문턱과 두 후보 고정',fontsize=17)
    fig.text(.5,-.03,'D/E 동일 query 1237점(D1147/E90). CAL46은 평가 일부와 학습 중복. 현재223은 개발 노출 평가이며 blind test가 아님.',ha='center',fontsize=10)
    png=V/'REAL_TRANSFER_READOUT_KO.png';fig.savefig(png,dpi=130,bbox_inches='tight');plt.close(fig)
    result=dict(completed=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active',fully_solved=False,outcome_KO=outcome,
        aggregate_primary=aggregate,manual_near_D=T['manual_near_D'],paired_recovery_and_damage=comparisons,all_group_CMs=T['D_vs_external_E_2x3'],
        actual_execution=dict(head_session=98847,head_exit_code=0,saved_evaluation_exit_code=0,new_training=0,new_encoder_RGB_initializer=0,PnP_or_q_L_correction=0,C16_original_byteparity=True),
        selected_model=None,source_qualification_unchanged=True,source_native35_pass_each=22,source_native35_total=35,
        source_public_HALL=dict(D_inputvalid=33,E_inputvalid=6,original_D_input_U=2,C32_correct_D=29,C33_correct_D=28,C34_correct_D=33,E_all_three_correct=6,assistant_weak_labels_not_human_gold=True),
        development_exposed_not_blind=True,current_used_for_training_or_threshold_checkpoint_model_choice=False,
        limitations_KO=a['limits_KO']+['기존 실사 CAL46은 평가 일부와 학습 중복이 있는 과거 참고다.','공개 실사 HALL은 작은 개발 노출 자료이며 assistant 구조적 weak 라벨이다.','공개 실사 E6은 시작부터 정답이므로 신규 효과는 D 보존이다.','현재223 수치는 외부/자기 원인 분류나 query 단일 픽셀 gold를 뜻하지 않는다.'],
        bindings=dict(evaluation=bind(R/'EVALUATION_SEAL.json'),full_results=bind(R/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'),prediction=bind(V/'frozen_current_head_benchmark_v1/PREDICTION_SEAL.json'),independent_review=bind(V/'independent_saved_evaluation_review/REVIEW.json'),plot=bind(png)))
    write(V/'ROOT_RESULTS_KO.json',result)
    s='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V149 공개 실사 학습 후 전이 비교</title><style>body{font:16px/1.7 sans-serif;max-width:1250px;margin:32px auto;padding:0 22px;color:#172033;background:#f7f9fd}h1{font-size:29px}h2{margin-top:34px}.lead{background:#e6f3f1;border-left:5px solid #168778;padding:18px}.scroll{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;background:white;font-size:14px}td,th{border:1px solid #dae1eb;padding:9px 12px;text-align:right}th{background:#eaf0f8}td:first-child,td:nth-child(2){text-align:left}img{max-width:100%}a{color:#1456b8}.muted{color:#596476}</style><body><h1>V149 · 공개 실사 학습 후 사용자 평가223장</h1>'''
    s+='<p class="lead">'+html.escape(outcome)+'</p><p>초기 좌표·crop·입력 순서·원래 보류45점을 그대로 유지했다. C33/C34 두 후보와 2000회 학습·raw 문턱0을 평가 전에 고정했다. 원 C16 출력의 byte 일치를 통과한 뒤 동결 추론하고, 예측 봉인 후 라벨로 지표를 계산했다. 이번 단계의 새 학습·PnP·좌표 보정은 없다.</p>'
    s+='<img src="REAL_TRANSFER_READOUT_KO.png" alt="일곱 모델의 보임 보존과 외부 가림 검출 및 3px 이하 보임 오판 비교">'
    s+='<h2>동일 query 1237점</h2><p>D=보임, E=외부 가림, U=보류. 실제 D1147/E90. recall은 보류를 포함한 실제 상태 전체 분모다. 전체 입력 U45/1784와 이 표의 분모는 다르다.</p>'
    s+=table(['모델','실제 D → [D,E,U]','실제 E → [D,E,U]','D precision','D recall','E precision','E recall','보류율'],[[names[r['model']],r['CM_D_E_by_pred_D_E_U'][0],r['CM_D_E_by_pred_D_E_U'][1],pct(r['D_precision']),pct(r['D_recall']),pct(r['E_precision']),pct(r['E_recall']),pct(r['U'])]for r in aggregate])
    headers=['재질','모델','실제 D → [D,E,U]','실제 E → [D,E,U]','D precision','D recall','E precision','E recall','보류율']
    for cohort,title in [('paired_same_query_RGB','목재/정사각형 · 동일 query'),('same_slot_pipeline','기존 동일 slot · 1248점')]:
        rows=[r for r in T['D_vs_external_E_2x3']if r['cohort']==cohort]
        s+='<h2>'+title+'</h2>'+table(headers,[[gn[r['group']],names[r['model']],[r['D_pred_D'],r['D_pred_hidden'],r['D_pred_U']],[r['E_pred_D'],r['E_pred_hidden'],r['E_pred_U']],pct(r['D_precision']),pct(r['D_recall']),pct(r['external_E_response_precision']),pct(r['external_E_response_recall']),pct(r['unresolved_rate'])]for r in rows])
    s+='<h2>오차 ≤3px · 수동 보임376점</h2><p>학습·문턱·모델 선택 필터가 아닌 진단이다.</p>'+table(['재질','모델','N','보임 정답','가림 오판','보류'],[[gn[r['group']],names[r['model']],r['N'],r['pred_visible'],r['pred_binary_hidden'],r['unresolved']]for r in T['manual_near_D']])
    s+='<h2>같은 행의 회복과 손상</h2>'+table(['기준','후보','기준 오답 → 정답','기준 정답 → 오답'],[[r['baseline'],r['new_model'],r['recovered'],r['damaged']]for r in comparisons])
    s+='<h2>자기 가림 · 별도 진단</h2><p>자기 가림은 upstream 기하 처리 대상이다. binary hidden 응답은 외부/자기 원인을 식별했다는 의미가 아니다.</p>'+table(['재질','모델','사람 S 수','hidden 응답','보임 응답','미해결'],[[gn[r['group']],names[r['model']],r['human_S_support'],r['binary_hidden_response'],r['visible_response'],r['unresolved']]for r in T['human_self_binary_response']if r['cohort']=='paired_same_query_RGB'])
    s+='<h2>공개 실사에서 확인한 이득과 한계</h2><p>학습은 LAB의 입력 유효43점(D35/E8), 검증은 HALL의 입력 유효39점(D33/E6)이다. C32는 D29/E6, C33은 D28/E6, C34는 D33/E6 정답이다. 외부 가림 E6은 시작부터 모두 정답이어서 이번 이득은 보이는 점 보존이다. 공개 라벨은 하부 두 역할의 assistant weak 판정이고 사람 gold가 아니다. 원래 입력 불가2점은 보류로 유지했다. 합성 native35 기준은 두 후보 모두22/35이며 채택 조건을 통과하지 못했다.</p>'
    s+='<h2>평가의 한계</h2><p class="muted">'+html.escape(' '.join(result['limitations_KO']))+'</p><h2>저장 근거</h2><ul>'
    for label,path in [('전체 수치',R/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'),('1784행 원장',R/'ALL1784_NATIVE_LEDGER.csv'),('외부 가림 혼동행렬',R/'D_VS_EXTERNAL_E_2X3.csv'),('독립 검산',V/'independent_saved_evaluation_review/REVIEW.json'),('공개 실사 crop 저장 진단',V.parent/'query_visibility_public_native_supervision_control_20261024_v148/source_visual_diagnosis/INDEX.html')]:s+='<li><a href="'+html.escape(str(path))+'">'+label+'</a></li>'
    s+='</ul></body></html>'; (V/'ROOT_READOUT_KO.html').write_text(s)
    print(json.dumps(dict(results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(V/'ROOT_READOUT_KO.html'),plot=bind(png),aggregate=aggregate,comparisons=comparisons),ensure_ascii=False))
