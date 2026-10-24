"""Render saved aggregate benchmark results; no models or image inputs."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, html, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V=Path(__file__).resolve().parent
R=V/'saved_current_benchmark_evaluation_v1'
a=json.loads((R/'CURRENT_FROZEN_BENCHMARK_RESULTS.json').read_text())
assert a['completed'] and a['PASS'] and a['selected_model'] is None
T=a['tables']; models=a['models']; groups=['WOOD_PRIMARY105','SQUARE118']
names={'v61':'v61','legacy':'기존 실사 CAL46','C16':'이전 C16','C29':'합성 보강 C29','C32':'AUX gradient 제외 C32'}
gn={'WOOD_PRIMARY105':'목재','SQUARE118':'정사각형'}
primary=[x for x in T['metrics'] if x['cohort']=='paired_same_query_RGB' and x['task']=='D_vs_E']
aggregate=[]
for model in models:
    rows=[x for x in primary if x['model']==model]
    count={k:sum(r[k] for r in rows) for k in ['N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']}
    D,E=count['visible_support'],count['occluded_support'];dd,de,ed,ee=[count[k] for k in ['TP','FN_decided','FP','TN']]
    du=count['U_visible']+count['out_of_task_S_visible']+count['other_unresolved_visible'];eu=count['U_occluded']+count['out_of_task_S_occluded']+count['other_unresolved_occluded']
    aggregate.append(dict(model=model,CM_D_E_by_pred_D_E_U=[[dd,de,du],[ed,ee,eu]],D_precision=dd/(dd+ed),D_recall=dd/D,E_precision=ee/(ee+de),E_recall=ee/E,U=(du+eu)/count['N'],counts=count))
def binding(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def metricrow(x):
    return [gn.get(x['group'],x['group']),names[x['model']],f"[{x['D_pred_D']}, {x['D_pred_hidden']}, {x['D_pred_U']}]",f"[{x['E_pred_D']}, {x['E_pred_hidden']}, {x['E_pred_U']}]",pct(x['D_precision']),pct(x['D_recall']),pct(x['external_E_response_precision']),pct(x['external_E_response_recall']),pct(x['unresolved_rate'])]

font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
colors=['#9ca3af','#bd8d4b','#2563eb','#8b5cf6','#dc4b54']
for ax,key,title in zip(axes[:2],['D_recall','E_recall'],['보이는 점을 보인다고 판단','외부 가림을 가림으로 판단']):
    values=[r[key]*100 for r in aggregate]
    bars=ax.bar(range(5),values,color=colors)
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4,fontsize=10)
    ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('정답 / 해당 실제 상태 전체 (%)');ax.set_title(title,pad=12)
    ax.set_xticks(range(5),['v61','실사\nCAL46','C16','C29','C32'])
    ax.spines[['top','right']].set_visible(False)
near=[]
for m in models:
    rows=[r for r in T['manual_near_D'] if r['model']==m]
    near.append(sum(r['pred_binary_hidden'] for r in rows))
bars=axes[2].bar(range(5),near,color=colors)
axes[2].bar_label(bars,padding=4);axes[2].set_ylim(0,335);axes[2].set_title('오차 ≤3px · 수동 D 376개\n보이는데 가림으로 오판한 수',pad=12)
axes[2].set_xticks(range(5),['v61','실사\nCAL46','C16','C29','C32']);axes[2].set_ylabel('오판 개수 (보류 별도)');axes[2].spines[['top','right']].set_visible(False)
fig.suptitle('실사 동일 query 비교: 합성 개선이 실사 개선으로 이어지지 않음',fontsize=17)
fig.text(.5,-.03,'D/E 주 비교 1237개(D1147/E90). CAL46은 평가 영상 일부와 학습이 겹친 과거 참고 모델. 현재 자료는 개발 노출 평가셋.',ha='center',fontsize=10)
png=V/'REAL_TRANSFER_READOUT_KO.png';fig.savefig(png,dpi=130,bbox_inches='tight');plt.close(fig)

outcome='새 합성 검증의 작은 개선은 실사 전체 개선으로 이어지지 않았다. C32는 C16 대비 E 정답 +3/90이지만 D 정답 −67/1147이며, 수동 D·오차≤3px에서도 19/376 오판이 생겼다. 해결 또는 모델 채택 근거가 아니다.'
result=dict(completed=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active',fully_solved=False,outcome_KO=outcome,
    aggregate_primary=aggregate,all_group_CMs=T['D_vs_external_E_2x3'],manual_near_D=T['manual_near_D'],paired_recovery_and_damage=T['paired_recovery_and_damage'],
    actual_execution=dict(head_session=44629,head_exit_code=0,saved_evaluation_exit_code=0,new_training=0,new_encoder_RGB_initializer=0,PnP_or_q_L_correction=0,C16_original_byteparity=True),
    selected_model=None,source_qualification_unchanged=True,development_exposed_not_blind=True,
    limitations_KO=a['limits_KO']+['기존 semantic reference D/E는 실사 단일 픽셀 presence의 gold가 아니다.','실사 CAL46은 현재 자료 일부와 학습이 겹친 과거 참고이며 독립 일반화 비교가 아니다.'],
    bindings=dict(evaluation=binding(R/'EVALUATION_SEAL.json'),full_results=binding(R/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'),prediction=binding(V/'frozen_current_head_benchmark_v1/PREDICTION_SEAL.json'),plot=binding(png)))
write(V/'ROOT_RESULTS_KO.json',result)
headers=['재질','모델','실제 D → [D,E,U]','실제 E → [D,E,U]','D precision','D recall','E precision','E recall','보류율']
s='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V143 실사 전이 고정 비교</title><style>body{font:16px/1.65 sans-serif;max-width:1250px;margin:35px auto;padding:0 24px;color:#172033;background:#fafbff}h1{font-size:30px}h2{margin-top:34px}p{max-width:1050px}.lead{background:#fff0ee;border-left:5px solid #db5549;padding:18px}table{border-collapse:collapse;background:white;font-size:14px;white-space:nowrap}td,th{padding:9px 13px;border:1px solid #dfe3ec;text-align:right}th{background:#eaf0fa}td:first-child,td:nth-child(2){text-align:left}.scroll{overflow-x:auto}img{max-width:100%}a{color:#1456b8}.muted{color:#596476}</style><body><h1>V143 · 최근 두 모델의 실사 전이 확인</h1>'''
s+='<p class="lead">'+outcome+'</p><p>초기 키포인트의 위치·crop 크기·순서와 기존 보류 45개를 그대로 사용했다. C16 저장 출력 비트 일치를 먼저 통과한 후 C29와 C32를 동결 평가했다. 새 학습·문턱 조정·encoder·영상 입력·PnP·좌표 수정은 없다. 두 후보 모두 원래 source qualification 실패 상태이며 채택하지 않았다.</p>'
s+='<img src="REAL_TRANSFER_READOUT_KO.png" alt="실사 D/E recall 및 오차 3px 이하 D 오판 비교">'
s+='<h2>같은 query 주 비교 · 1237개</h2><p>D=보임, E=외부 가림, U=보류/미해결. 실제 D1147개/E90개. recall은 U를 포함한 실제 상태 전체 분모다. U가 0%인 이 표와 전체 입력의 U45/1784는 분모가 다르다.</p>'
s+=table(['모델','실제 D → [D,E,U]','실제 E → [D,E,U]','D precision','D recall','E precision','E recall','보류율'],[[names[r['model']],str(r['CM_D_E_by_pred_D_E_U'][0]),str(r['CM_D_E_by_pred_D_E_U'][1]),pct(r['D_precision']),pct(r['D_recall']),pct(r['E_precision']),pct(r['E_recall']),pct(r['U'])]for r in aggregate])
s+='<h2>목재 / 정사각형 · 같은 query</h2>'+table(headers,[metricrow(r)for r in T['D_vs_external_E_2x3']if r['cohort']=='paired_same_query_RGB'])
s+='<h2>기존 같은 slot 비교 · 1248개</h2><p>가시성 점수와 기존 기하 gate의 query 동일성까지 주장하는 분모가 아니다. 원래 cohort를 그대로 유지한다.</p>'+table(headers,[metricrow(r)for r in T['D_vs_external_E_2x3']if r['cohort']=='same_slot_pipeline'])
s+='<h2>수동 보임 D · 좌표 오차 ≤3px</h2><p>목재153개/정사각형223개. 전체 평가의 필터나 학습·문턱 선택 기준으로 사용하지 않은 보조 진단이다.</p>'+table(['재질','모델','N','보임 정답','가림 오판','보류'],[[gn[r['group']],names[r['model']],r['N'],r['pred_visible'],r['pred_binary_hidden'],r['unresolved']]for r in T['manual_near_D']])
s+='<h2>기존 C16 대비 같은 행의 회복 / 손상</h2>'+table(['재질','후보','둘 다 정답','기존 정답 → 오답','기존 오답 → 정답','둘 다 오답'],[[gn[r['group']],r['new_model'],r['both_correct'],r['baseline_correct_new_wrong'],r['baseline_wrong_new_correct'],r['both_wrong']]for r in T['paired_recovery_and_damage']if r['cohort']=='paired_same_query_RGB'and r['baseline']=='C16'])
s+='<h2>자기 가림 S · 별도 응답 진단</h2><p>자기 가림은 사용자의 upstream 기하 처리 대상이다. 아래 E/hidden 응답은 학습 모델이 외부/자기 원인을 구별했다는 의미가 아니다. 정사각형의 human S support=0도 정답 증거가 아니다.</p>'+table(['재질','모델','사람 S 수','binary hidden 응답','보임 응답','미해결'],[[gn[r['group']],names[r['model']],r['human_S_support'],r['binary_hidden_response'],r['visible_response'],r['unresolved']]for r in T['human_self_binary_response']if r['cohort']=='paired_same_query_RGB'])
s+='<h2>전체 1784 slot · 누락과 U 유지</h2><p>정답 상태는 D1168/E92/S136/화면 밖24/미주석356/매핑 미상8이다. 전부 D/E 평가 정답은 아니다. 기존 실사 모델의 기하 S는 외부 E로 강제하지 않고 미해결에 합산하되 별도 표시한다.</p>'+table(['그룹','모델','전체','D 응답','binary E 응답','전체 U','U율','legacy U','기하 S'],[[gn.get(r['group'],'전체'),names[r['model']],r['all_registered_slots'],r['D'],r['E_binary_response'],r['U'],pct(r['U_rate']),r['legacy_abstain'],r['out_of_task_geometry_S']]for r in T['all1784_model_inventory']])
s+='<h2>해석과 다음 확인</h2><p>새 합성 VAL에서는 C29→C32 외부 가림 정답이112→116/130으로 늘었지만, 실사 D 오판이 크게 증가했다. 목표 혼합을 줄이는 것만으로 전이 문제가 풀리지 않았다. 초기 좌표 오차만으로 설명하기도 어렵다. 공개 실사에서 실제 초기 추론 대상·native 부위와 가림 라벨의 대응을 맞추는 작업, 작은 틈과 부위를 읽는 표현의 source 진단은 기존 계획대로 남아 있다. 실사 결과로 모델·문턱을 고르지 않는다.</p><p class="muted">현재223은 반복 개발에 노출된 평가 자료이며 독립 blind test가 아니다. 기존 실사 CAL46 모델은 학습 중복이 있는 과거 참고다. 사람의 reference 가림 라벨은 query 단일 픽셀 mask gold가 아니다. 성공 가능성의 수치를 추정할 근거는 없으며, 이번 결과는 해결을 입증하지 못한다.</p>'
s+='<h2>저장 근거</h2><ul>'
for label,path in [('전체 평가 결과',R/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'),('좌표 오차 구간별 지표',R/'COORDINATE_ERROR_BINS.csv'),('1784행 원래 라벨·예측 원장',R/'ALL1784_NATIVE_LEDGER.csv'),('이전 합성 진단 및 실제 crop',V.parent/'query_visibility_native_target_gradient_control_20261024_v142/ROOT_READOUT_KO.html'),('예측 검산',V/'independent_saved_prediction_review/REVIEW.json')]:
    s+=f'<li><a href="{html.escape(str(path))}">{label}</a></li>'
s+='</ul></body></html>'
(V/'ROOT_READOUT_KO.html').write_text(s)
print(json.dumps({'root_results':binding(V/'ROOT_RESULTS_KO.json'),'readout':binding(V/'ROOT_READOUT_KO.html'),'plot':binding(png),'aggregate':aggregate},ensure_ascii=False))
