"""Render a source-only scientific readout from frozen V160 outputs; no model calls."""
from pathlib import Path
import datetime, hashlib, html, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def bind(p): return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
summary=read(V/'factorial_source_readout/ARM_DATASET_SUMMARY.json')
full=read(V/'factorial_source_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json')
review=read(V/'independent_saved_review/SAVED_SOURCE_EXECUTION_REVIEW.json')
assert review['PASS'] and review['C39_C38_source_scale_control_exact']
control=read(V/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000/CONTROL_COMPLETE.json')
execution=read(V/'ROOT_ACTUAL_SOURCE_FIT_COMPLETED.json')
assert control['completed'] and control['PASS'] and execution['exec_exit_code']==0
arms=list(summary['arms'])
labels=['C39 baseline','C40 +slot','C41 +owner','C42 +both']
ko=['기준','키포인트 번호 추가','중심 표면 보조 추가','둘 다 추가']
folds={'OLD6144':'VALIDATION','NEW20480':'VALIDATION','PALLOX512':'VAL_HALL','MR1840':'SOURCE_VALIDATION'}
families=list(folds)
def m(arm,fam,group=None):return summary['arms'][arm][fam][group or folds[fam]]['all_original_rows']
def pct(x):return '—' if x is None else f'{x*100:.2f}%'
def cmtext(cm):return '['+', '.join(str(row) for row in cm)+']'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
x=np.arange(4);colors=['#4f647d','#4992c7','#35a58c','#9b78c4']
for i,a in enumerate(arms):
    train=m(a,'MR1840','SOURCE_DEVELOPMENT');held=m(a,'MR1840')
    for target,metric in [(ax[0],train),(ax[1],held)]:
        y=metric['E_recall']*100;target.bar(i,y,color=colors[i],width=.65)
        cm=metric['CM_D_E_rows_D_E_U_columns'];target.text(i,y+2,f'{cm[1][1]}/{sum(cm[1])}',ha='center',fontsize=11)
    passed=summary['arms'][a]['gates']['native35_passed'];ax[2].bar(i,passed,color=colors[i],width=.65);ax[2].text(i,passed+.5,f'{passed}/35',ha='center')
for axis in ax[:2]:axis.set_ylim(0,112);axis.set_ylabel('External occlusion recall (%)');axis.set_yticks([0,20,40,60,80,100]);axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
ax[0].set_title('MR TRAIN weak labels: all 49/49')
ax[1].set_title('MR held scene: no visibility gain')
ax[2].set_title('Original native source checks');ax[2].set_ylim(0,38);ax[2].set_ylabel('Checks passed');ax[2].set_yticks([0,7,14,21,28,35]);ax[2].grid(axis='y',alpha=.15);ax[2].set_axisbelow(True)
for axis in ax:axis.set_xticks(x,labels,rotation=18,ha='right')
fig.suptitle('V160: added slot / owner supervision did not resolve held-scene failure',fontsize=16)
fig.savefig(V/'ROOT_SOURCE_RESULT_OVERVIEW.png',dpi=160);plt.close(fig)

header='<tr><th>조건</th><th>혼동행렬<br>행 D/E · 열 D/E/U</th><th>D precision</th><th>D recall</th><th>E precision</th><th>E recall</th><th>보류 / D·E</th></tr>'
def metricrow(label,metric):
    return '<tr><td>'+html.escape(label)+'</td><td class="mono">'+html.escape(cmtext(metric['CM_D_E_rows_D_E_U_columns']))+'</td>'+''.join('<td>'+pct(metric[k])+'</td>' for k in ['D_precision','D_recall','E_precision','E_recall'])+f'<td>{metric["prediction_U_known"]}/{metric["known_DE"]} ({pct(metric["prediction_U_rate_known"])})</td></tr>'
def table(rows):return '<div class="scroll"><table>'+header+''.join(rows)+'</table></div>'
sections=[]
for fam in families:
    first=m(arms[0],fam)
    title={'OLD6144':'기존 합성 검증','NEW20480':'추가 합성 검증','PALLOX512':'Pallox 공개 실사 검증','MR1840':'MR 공개 실사 · 분리 장면'}[fam]
    text=f'<p>원래 {first["all_original_rows"]}개 입력 · D {first["known_D"]} / E {first["known_E"]} · 외부 가림 이진 정답 밖 {first["unknown_truth"]}개. 원래 입력 보류 {first["factual_input_U_all"]}개 중 D/E 정답이 있는 보류 {first["factual_input_U_known"]}개를 아래 분모에 포함했습니다.</p>'
    valid_first=summary['arms'][arms[0]][fam][folds[fam]]['input_valid_diagnostic']
    diagnostic='<details><summary>원래 입력 유효 조건부 진단 · '+str(valid_first['known_DE'])+'개 D/E</summary><p>입력이 원래 유효한 D '+str(valid_first['known_D'])+' / E '+str(valid_first['known_E'])+'개만의 별도 진단입니다. 위 전체 정답 분모와 보류율을 대체하지 않습니다.</p>'+table([metricrow(ko[i],summary['arms'][a][fam][folds[fam]]['input_valid_diagnostic']) for i,a in enumerate(arms)])+'</details>'
    sections.append('<section><h2>'+title+'</h2>'+text+table([metricrow(ko[i],m(a,fam)) for i,a in enumerate(arms)])+diagnostic+'</section>')
native_sections=[]
for fam in families:
    fold=folds[fam];rows=[]
    for i,a in enumerate(arms):
        for slot in range(8):
            key=fold+f'::native{slot}';rows.append(metricrow(ko[i]+f' · native{slot}',full['arms'][a][fam][key]['all_original_rows']))
    native_sections.append('<details><summary>'+html.escape(fam)+' · native0–7 전체</summary>'+table(rows)+'</details>')
for fam in families[:2]:
    keys=[k for k in full['arms'][arms[0]][fam] if k.startswith('VALIDATION::material_aspect::') and '::native' not in k]
    rows=[metricrow(ko[i]+' · '+key.split('material_aspect::')[1],full['arms'][a][fam][key]['all_original_rows']) for i,a in enumerate(arms) for key in keys]
    native_sections.append('<details><summary>'+fam+' · 목재/폴리머 × 직사각형/정사각형</summary>'+table(rows)+'</details>')

manifest=read(V/'source_crop_readout/rendered_fixed12/FIXED12_RENDER_MANIFEST.json')
assert manifest['completed'] and len(manifest['records'])==12
crop_html=[]
for r in manifest['records']:
    p=Path(r['card']['path']);assert bind(p)==r['card']
    case=r['case'];title=f'{case["card_index"]:02d} · {case["family"]} · native{case["native_slot"]} · 기존 약라벨 {case["weak_role_state"]}'
    crop_html.append('<details><summary>'+title+'</summary><a href="'+str(p.relative_to(V))+'"><img loading="lazy" src="'+str(p.relative_to(V))+'" alt="'+title+'"></a></details>')

result={'schema':'V160_ROOT_FOUR_ARM_SOURCE_RESULT_UNSOLVED','completed':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_execution':bind(V/'ROOT_ACTUAL_SOURCE_FIT_COMPLETED.json'),'source_summary':bind(V/'factorial_source_readout/ARM_DATASET_SUMMARY.json'),'full_native_material_tables':bind(V/'factorial_source_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json'),'factorial_effects':bind(V/'factorial_source_readout/FACTORIAL_EFFECTS_DIAGNOSTIC.json'),'independent_review':bind(V/'independent_saved_review/SAVED_SOURCE_EXECUTION_REVIEW.json'),'fixed12_manifest':bind(V/'source_crop_readout/rendered_fixed12/FIXED12_RENDER_MANIFEST.json'),'overview_figure':bind(V/'ROOT_SOURCE_RESULT_OVERVIEW.png'),'all_source_rows_each':28976,'all_inputU_each':535,'known_DE_each':12949,'unknown_truth_each':16027,'all4_actual_updates_each':8000,'all4_actual_updates_total':32000,'MR_held_full':[dict(arm=a,condition_KO=ko[i],**m(a,'MR1840')) for i,a in enumerate(arms)],'native35_passed':[summary['arms'][a]['gates']['native35_passed'] for a in arms],'MR_train_available_D341_E49_all_correct_each':True,'trained_owner_weak253_all_correct_C41_C42':True,'held_owner_gold_available':False,'held_owner_accuracy':None,'C39_previous_C38_source_outputs_and_draws_exact':True,'all_source_full_qualification_PASS':False,'interpretation_KO':['이 고정 설계·seed1550·8000회 조건에서 키포인트 번호와 owner253 약감독은 공개 MR held 외부 가림 일반화를 개선하지 못했다. 다른 구현·규제·더 다양한 데이터의 효과까지 부정하지는 않는다.','TRAIN 주 가림 및 owner 보조 분류 적합은 성공했지만, 이것은 held 가림 성공 증거가 아니다.','Pallox 카드보드 가림은 원래 기준도 맞혔다. MR 바구니/목재상자에 같은 개선이 전이되지 않았다.','정확한 코드 대조군·draw·원래 입력과 보류 보존이 확인돼, 이 결과를 단순한 새 브랜치 초기화/평가 어댑터 오류로 설명할 근거는 없다.','중심에 보이는 재료와 의도한 native 부위의 가시성은 여전히 별도다. owner를 E로 자동 변환하거나 GT 좌표 오차로 평가 표본을 빼지 않았다.'],'limitations_KO':['공개 실사 D/E는 기존 구조적 약라벨이며 공식 사람이 찍은 키포인트 가시성 gold가 아니다.','한 seed의 고정 대조실험이며 효과가 모든 학습법에서 없다고 일반화하지 않는다.','별도 entity 출력의 기존 한계와 실사 좌표 품질 gold 부재가 남아 있다.','이번 단계는 current223를 읽거나 새 평가·학습·문턱·조건 선택에 사용하지 않았다. 최신 실제 current 평가 기준은 V156으로 유지한다.'],'selected_model':None,'adopted':False,'fully_solved':False,'goal_status':'active_unsolved','current223_reads':0,'V112_reads':0,'PnP_q_L_changes':0,'new_encoder_R0_calls':0}
write(V/'ROOT_RESULTS_KO.json',result)

css='body{font:16px/1.65 system-ui,sans-serif;color:#203147;background:#f4f6f9;margin:0}main{max-width:1180px;margin:auto;padding:30px 22px}h1{font-size:30px;line-height:1.3}h2{font-size:23px}section,.box,details{background:white;border:1px solid #dce2e9;border-radius:12px;padding:20px;margin:20px 0}.lead{font-size:21px}.negative{border-left:5px solid #b94c40}.muted{color:#596b80;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}td,th{padding:9px 12px;text-align:right;border-bottom:1px solid #e3e8ef}th:first-child,td:first-child{text-align:left}.mono{font-family:monospace}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:650}a{color:#1b66a0}code{background:#eaf0f5;padding:2px 5px;border-radius:4px}'
body='''<h1>가림 판단 대조실험 V160</h1><p class="lead">번호 정보와 중심 표면 보조 학습을 추가했지만, 다른 장면의 가림 판단은 개선되지 않았습니다.</p><div class="box negative"><strong>실제 네 학습 완료 · 총 32,000회 · 새 모델 채택 없음</strong><p>기준 / 번호 추가 / 표면 보조 추가 / 둘 다 추가를 같은 초기값·데이터·8,000회로 비교했습니다. 네 모델을 모두 동결한 뒤 저장 출력으로 계산했습니다. MR 검증 가림은 각각 <b>2/52 · 2/52 · 2/52 · 1/52</b>입니다. 보이는 점 D는 모두 93/102를 유지했고, 나머지 9개는 원래 입력 보류입니다.</p></div>'''
body+='<img src="ROOT_SOURCE_RESULT_OVERVIEW.png" alt="학습 자료 적합과 분리 장면 일반화, 원래 source 조건 통과 수 비교">'
body+='''<p class="muted">D = 기존 native 부위 보임 약라벨, E = 외부 가림 약라벨, U = 원래 입력 보류. 실사 약라벨은 공식 per-keypoint human gold가 아닙니다. 보조 owner의 대상 팔레트/외부 물체/배경은 이 D/E와 별개입니다.</p><section><h2>이번에 확인한 점</h2><p>학습에 쓴 MR D341/E49는 네 조건 모두 맞혔습니다. 표면 보조를 학습한 두 조건은 대상 53 / 외부 물체 52 / 배경 148의 약감독 253개도 모두 맞혔습니다. 그러나 이 학습 적합이 분리 장면의 native 가림 판단으로 이어지지 않았습니다. 검증 장면의 owner 정답은 전부 UNKNOWN이므로 owner 일반화 정확도를 주장하지 않습니다.</p><p>추가 기능을 끈 C39는 이전 C38의 28,976개 원래 세 출력과 예측·표본 draw를 정확하게 재현했습니다. 독립 검증은 모든 혼동행렬·35개 기준·보류·분할 보존을 확인했습니다. 결과 읽기 어댑터의 두 스키마 오류는 원본 실패를 보존하고 읽기 부분만 고쳤으며, 모델 학습을 다시 실행하지 않았습니다.</p><p>전체 native source 기준 통과는 25/35 → 23/35 → 23/35 → 21/35입니다. 기존 entity 한계와 무관하게 주 visibility 조건도 통과하지 못했습니다. 기준·문턱을 낮추거나 일부 조건만 골라 성공으로 바꾸지 않았습니다.</p></section>'''
body+='<p class="muted">원래 전체 28,976개 중 D/E 정답 12,949개, 외부 가림 이진 정답 밖 16,027개입니다. 원래 입력 보류 535개는 D/E 정답이 있는 132개와 그 밖 403개로 구성됩니다. 정답 밖에는 자기 가림·화면 밖·미확정이 포함되며 E로 바꾸지 않았습니다.</p>'
body+=''.join(sections)
body+='<section><h2>목재·정사각형·native8 전체 상세</h2><p>정답이 없는 native/자기 가림은 별도 집계하며 외부 가림 E로 바꾸지 않았습니다. 아래 표는 원래 정답·입력 보류를 포함합니다. 모든 native×재질×형상 조합과 학습/검증 분할은 <a href="factorial_source_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json">전체 JSON</a>에 있습니다.</p>'+''.join(native_sections)+'</section>'
body+='''<section><h2>고정한 실제 crop 12개</h2><p>점수와 RGB를 보기 전에 입력 메타데이터로 정한 12개입니다. 각 카드는 원래 전체 RGB, 실제 저장 RGB96, 중앙32 확대와 네 모델의 출력을 보여줍니다. q/L과 crop은 이동하지 않았습니다. 가시성/owner 재라벨이나 검증 자료의 학습 편입은 없습니다.</p><p>Pallox의 두 골판지 가림 예시는 네 모델 모두 E로 판단합니다. MR의 바구니·목재상자 예시는 네 모델 모두 D로 남았습니다. 특히 나무상자 모서리가 들어온 사례에서 보조 출력이 대상 팔레트 쪽에 약 0.996을 주는 모습은, 재료와 소유 개체를 구별하는 문제가 남아 있음을 시사합니다. 이 정성 관찰을 새로운 검증 owner 정답으로 등록하지 않았습니다.</p>'''+''.join(crop_html)+'</section>'
body+='''<section><h2>남은 문제와 다음 작업</h2><p>이번 고정 설계에서는 번호 정보나 253개 표면 보조 감독만 더하는 것으로는 부족했습니다. 학습한 장면·가림물 범위를 넘어서는 데이터와, 원래 q 주변을 직접 구분하도록 하는 감독이 더 필요한지 다음 source 단계에서 확인해야 합니다. 공개 TRAIN의 확실한 표면 사례에 대해 같은 q의 원본/불투명 덮기/옆 배치 대조 자료를 만들 수 있는지 경로·의미·변경된 전체 이미지 입력의 일관성을 먼저 확인하고 있습니다. 이는 owner 보조 감독 준비이며, native 가림이나 좌표 품질의 정답으로 자동 복사하지 않습니다.</p><p>평가용 실사 223장 접근 0 · 새 current 평가 0 · 새 encoder/R0 실행 0 · PnP/좌표 보정 0. 최신 실제 current 평가 기준은 V156이며 이번 결과로 선택한 모델은 없습니다. 전체 목표는 미해결 상태입니다.</p></section>'''
body+='<p class="muted"><a href="ROOT_RESULTS_KO.json">결과 JSON</a> · <a href="factorial_source_readout/FACTORIAL_EFFECTS_DIAGNOSTIC.json">요인별 변화</a> · <a href="independent_saved_review/SAVED_SOURCE_EXECUTION_REVIEW.json">독립 검증</a> · <a href="ROOT_ACTUAL_SOURCE_FIT_COMPLETED.json">실제 실행 기록</a></p>'
(V/'ROOT_READOUT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V160 가림 대조실험 결과</title><style>'+css+'</style><main>'+body+'</main></html>')
print(json.dumps({'results':bind(V/'ROOT_RESULTS_KO.json'),'report':bind(V/'ROOT_READOUT_KO.html'),'figure':bind(V/'ROOT_SOURCE_RESULT_OVERVIEW.png')},ensure_ascii=False))
