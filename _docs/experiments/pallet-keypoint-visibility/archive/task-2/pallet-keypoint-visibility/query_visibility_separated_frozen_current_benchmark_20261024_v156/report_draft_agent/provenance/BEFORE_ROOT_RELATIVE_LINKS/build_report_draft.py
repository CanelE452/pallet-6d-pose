"""Aggregate-only Korean report draft. Never reads row/RGB/feature/model bodies."""
from pathlib import Path
import json, hashlib, datetime, html

OUT = Path(__file__).resolve().parent
V = OUT.parent
P = V.parent
SUMMARY = V/'current_result_summary_agent/CURRENT_RESULTS_SUMMARY_KO.json'
EVAL = V/'saved_seven_current_evaluation_v1/EVALUATION_SEAL.json'
PRED = V/'frozen_four_head_current_benchmark_v1/PREDICTION_SEAL.json'
COMMIT = P/'query_visibility_independent_auxiliary_source_control_20261024_v155/ROOT_NEXT_ACTION_COMMITMENT_BEFORE_ANY_NEW_PRIVATE_READS.json'
AUDIT = V/'independent_saved_evaluation_review/REVIEW.json'
PLOT_QA = V/'current_result_summary_agent/FINAL_SUMMARY_QA.json'

def read(p): return json.loads(p.read_text())
def bind(p): return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def esc(x): return html.escape(str(x))
def pct(x): return '—' if x is None else f'{100*x:.1f}%'
def dump(name,x):
    with (OUT/name).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')

s,es,ps,c,a,pq = [read(p) for p in (SUMMARY,EVAL,PRED,COMMIT,AUDIT,PLOT_QA)]
assert bind(SUMMARY)['sha256']=='a565ab034fdd4aade4aaea4d7ba7561c9782807e1dc61a4b0412bee2a09d1d63'
assert bind(EVAL)['sha256']=='52fb0fdaaeb1839afaea1cac4425379d0d488b75b81880a8f28aa469ad87654b'
assert bind(PRED)['sha256']=='255bca46874f95fe7837f7140024c456541e4d53c2a470673b134f2e0c815c4f'
assert bind(AUDIT)['sha256']=='191a3bdfc2b6355020dd9383ce4218f03ae2dfba49f09a48299c52ca28aabe0c'
assert all(x['completed'] and x['PASS'] for x in (s,es,ps,a,pq))
assert s['all_registered_slots']==1784 and s['original_input_U']==45
assert s['selected_model'] is None and c['selected_model'] is None
models=s['models']; groups=('ALL','WOOD_PRIMARY105','SQUARE118')
names={'ALL':'전체 paired1237','WOOD_PRIMARY105':'목재 paired645','SQUARE118':'정사각형 paired592'}
main=s['visibility']['paired_same_query_RGB']
for m in models:
    assert main['ALL'][m]['N']==1237 and main['ALL'][m]['D_N']==1147 and main['ALL'][m]['E_N']==90
plots=['CURRENT_ALL1237_CONFUSIONS_KO.png','CURRENT_MATERIAL_AND_NEAR_DIAGNOSTICS_KO.png','CURRENT_AUXILIARY_AUTHORITY_DIAGNOSTICS_KO.png']
plotfiles=[V/'current_result_summary_agent'/n for n in plots]
pair=[r for r in s['paired_recovery_damage_ALL'] if r['cohort']=='paired_same_query_RGB' and r['baseline']=='C16' and r['new_model'] in ('C35','C36','C37','C38') and r['truth_class'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')]
quality=s['secondary_fixed_reference_quality']
quality_compact={m:{k:z for k,z in quality['metrics']['ALL_ORIGINAL']['ALL'][m].items() if k in ('CM_rows_far_near_columns_far_near_U','far_recall','near_recall','far_precision','near_precision','U_known','untrained_neutral_auxiliary')} for m in models[3:]}
lead='외부가림을 검출하는 신호는 실사에서도 나타났지만, 보이는 점을 유지하는 성능과 함께 높이지는 못했습니다. 진전은 있으나 아직 성공이나 채택을 뒷받침하는 결과는 아닙니다.'
objective='초기 팔레트 추론이 낸 원래 키포인트와 crop을 고정하고, 그 native 역할이 외부 물체에 가려졌는지를 판단합니다. 자기 가림은 상위 기하 단계에서 우선 처리한다는 목표를 유지합니다. 이번 비교는 좌표 보정·PnP·자세 개선을 하지 않았습니다.'
limitations=[
 '현재223장은 평가 전용으로 유지했지만 이전 반복 평가로 개발에 노출되었습니다. 독립적인 blind test나 최종 일반화 성능으로 해석하지 않습니다.',
 'legacy는 과거 private 실사 학습·CAL 개발과 평가 계보가 겹치는 모델입니다. 새 모델과 독립된 공개자료 전용 기준선이라고 주장하지 않습니다.',
 'C35/C36은2000회·seed1540의 shared-hidden 계열이고 C37/C38은8000회·seed1550의 독립 보조 경로입니다. 이 둘 사이의 차이를 보조 경로 분리만의 인과효과로 해석할 수 없습니다.',
 'C37 warm/C38 fresh는 같은8000회 설정의 대조이지만 양쪽 모두 source 기준을 통과하지 못했습니다. 초기화 비교만으로 표현력 한계나 성공 가능성이 증명된 것은 아닙니다.',
 '보조 entity 출력에 대한 현재 실사의 독립 정답이 없습니다. 음수 반응과 조건부 gate 진단을 entity 정확도·오거부율로 부르지 않습니다.',
 '새 실사 위치 품질 gold는0개입니다. 원 reference와의 정확한 byte 연결은 실제 click 이력과 물리 native 대응의 권위를 모두 증명하지 않습니다.',
 '공개실사의 구조적 약한 라벨은 개체·역할·가림 범위에 한계가 있으며 human gold가 아닙니다. source 성능이 현재 실사 전이 성공을 대신하지 않습니다.'
]
mechanisms=[
 '공유 보조 손실을 더한 C36에서 보임 손상이 커졌지만, 이것만으로 라벨 충돌이나 gradient 간섭이 유일 원인이라고 단정할 수 없습니다.',
 '보조 경로를 분리한 C37/C38도 실사에서 보임 손상과 외부가림 누락을 함께 해결하지 못했습니다. 경로 분리 자체가 해결책이라는 주장도 성립하지 않습니다.',
 '이번 수치로 모든 모델의 실패를 좌표 오차 하나에 돌리지 않습니다. near376에서도 새 모델의 false E가 남습니다. 이 집계는 고정 reference 기반 진단이며 새3px gold를 선언한 결과는 아닙니다.'
]
future={
 'commitment':bind(COMMIT),'chosen_before_current_outcomes':True,
 'action_KO':'기존 공개·합성 source만으로 TRAIN과 held source의 대상·context·crop 범위, frame 단위 entity 오류, known E 실패를 점검합니다. 추가 공개자료의 이용 가능성은 공식 manifest로 확인합니다.',
 'fixed_limits_KO':'기존 held45 라벨이나 split을 바꾸지 않으며 학습으로 옮기지 않습니다. 새 source-only protocol이 봉인되기 전 새 fit은 하지 않습니다. 현재223 결과로 다음 구조·손실·문턱·checkpoint를 고르지 않습니다.',
 'original_scope':c['next_source_diagnostic'],'private_not_used_for_next_training':c['private_not_used_for_next_training'],
 'new_fit_authorized_by_this_report':False
}
draft={
 'schema':'V156_ROOT_REPORT_DRAFT_AGGREGATE_ONLY','draft_only':True,'completed':True,'PASS':True,
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lead_KO':lead,'objective_KO':objective,
 'inputs':[bind(p) for p in (SUMMARY,EVAL,PRED,COMMIT,AUDIT,PLOT_QA)],
 'primary':{'cohort':s['primary_cohort'],'all7_global_and_material':main},
 'same_slot_pipeline1248':s['visibility']['same_slot_pipeline'],
 'full_registered_labelled_DE':s['full_registered_labelled_DE'],
 'full_inventory':{'slots':1784,'original_available':1739,'original_input_U':45,'human_states':s['teacher_inventory'],'model_inventory':s['full_registered_model_inventory'],'human_self_binary_response':s['human_self_binary_response']},
 'manual_near_D376':{'scope':s['manual_near_D_scope'],'metrics':s['manual_near_D']},
 'paired_C16_to_candidates':pair,
 'quality_authority':{'new_gold_known':0,'new_gold_unknown':1784,'secondary_near':376,'secondary_far':39,'secondary_known':415,'secondary_unknown':1369,'known_original_input_U':3,'traceback_counts':quality['authority']['manual_raw_xy_traceback_counts'],'physical_native_authority_certified':False,'secondary_metrics':quality_compact},
 'auxiliary_response_diagnostic':s['auxiliary_response_diagnostic'],
 'mechanism_limits_KO':mechanisms,'interpretation_limits_KO':limitations,
 'source_qualification_unchanged':s['source_qualification_unchanged'],'selected_model':None,'adopted_model':None,'goal_solved':False,
 'visual_status':{'scientific_plots_completed_and_actually_viewed':True,'plots':[bind(p) for p in plotfiles],'crop_gallery_status':'PENDING_RENDER_REPAIR','crop_gallery_full_visual_complete':False,'reason_KO':'crop 갤러리 렌더 오류의 복구와 별도 QA가 남았습니다. 아래 초안은 실제 crop 전수 시각검토 완료를 주장하지 않습니다.'},
 'next_action_unchanged':future,
 'access':{'only_completed_aggregate_JSON_and_seals_read':True,'new_model_training_RGB_feature_weight_or_V112_access':0,'root_reports_or_parent_pointers_modified':False}
}
dump('ROOT_RESULTS_DRAFT.json',draft)

def metric_table(group):
    rows=[]
    for m in models:
        z=main[group][m];cm=z['CM_rows_D_E_columns_D_E_U']
        rows.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in (m,f'{cm[0][0]} / {cm[0][1]} / {cm[0][2]}',f'{cm[1][0]} / {cm[1][1]} / {cm[1][2]}',pct(z['D_precision']),pct(z['D_recall']),pct(z['E_precision']),pct(z['E_recall']),pct(z['U_rate'])))+'</tr>')
    return '<table><thead><tr><th>모델</th><th>사람 D → D/E/U</th><th>사람 E → D/E/U</th><th>D precision</th><th>D recall</th><th>E precision</th><th>E recall</th><th>U 비율</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
nearrows=[]
for m in models:
    z=s['manual_near_D']['ALL'][m];w=s['manual_near_D']['WOOD_PRIMARY105'][m];q=s['manual_near_D']['SQUARE118'][m]
    nearrows.append('<tr>'+''.join('<td>'+esc(x)+'</td>'for x in(m,z['pred_visible'],z['pred_binary_hidden'],z['unresolved'],w['pred_binary_hidden'],q['pred_binary_hidden']))+'</tr>')
qualrows=[]
for m,z in quality_compact.items():
    cm=z['CM_rows_far_near_columns_far_near_U']
    qualrows.append('<tr>'+''.join('<td>'+esc(x)+'</td>'for x in(m+'*'if m=='C35'else m,f'{cm[0][0]}/39',f'{cm[1][1]}/376',pct(z['far_precision']),z['U_known']))+'</tr>')
inventory=''.join('<tr><td>'+esc(name)+'</td><td>'+str(s['teacher_inventory']['ALL'][k])+'</td></tr>'for k,name in [('DIRECT_VISIBLE','D 보임'),('EXTERNAL_OCCLUDED','E 외부가림'),('SELF_OCCLUDED','S 자기 가림'),('OUT_OF_FRAME','화면 밖'),('UNANNOTATED','미주석'),('UNKNOWN_MAPPING','매핑 미확정')])
listitems=lambda xs:'<ul>'+''.join('<li>'+esc(x)+'</li>'for x in xs)+'</ul>'
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>초기 키포인트 외부가림 평가 · 보고서 초안</title>
<style>body{{max-width:1180px;margin:34px auto;padding:0 22px 60px;background:#f6f7f9;color:#192535;font-family:"Noto Sans KR","NanumGothic",sans-serif;line-height:1.7}}h1{{font-size:29px;line-height:1.4}}h2{{font-size:22px;margin:34px 0 12px}}h3{{font-size:18px;margin:25px 0 8px}}p{{margin:12px 0}}.card{{background:white;border:1px solid #d8dfe6;border-radius:12px;padding:22px;margin:18px 0}}.lead{{font-size:20px;font-weight:700}}.muted{{color:#526174;font-size:14px}}.flag{{background:#fff5da;border-left:5px solid #c28c17;padding:16px}}.draft{{font-size:13px;font-weight:700;color:#8b5210}}table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff}}th,td{{border:1px solid #d5dee7;padding:8px 7px;text-align:right;white-space:nowrap}}th{{background:#edf2f7}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow-x:auto}}img{{max-width:100%;height:auto;border:1px solid #d8dfe6;background:white}}figcaption{{font-size:13px;color:#526174}}li{{margin:7px 0}}a{{color:#235d99}}code{{background:#edf1f5;padding:2px 5px}}.smalltable{{max-width:650px}}</style>
<div class="draft">검토용 초안 · root 보고서와 포인터는 변경하지 않음 · crop 시각화 복구 대기</div>
<h1>외부가림 신호는 늘었지만<br>보이는 점 보존까지 함께 해결되지는 않았습니다</h1>
<div class="card"><p class="lead">{esc(lead)}</p><p>{esc(objective)}</p><p>성공 가능성이 없다고 결론낼 단계는 아닙니다. 다만 현재 결과는 두 요구를 함께 만족하는 방법을 찾았다는 증거가 아닙니다. 새 모델은 source 기준 실패 상태를 유지하며 채택하지 않았습니다.</p></div>
<h2>무엇을 같은 조건에서 비교했나</h2>
<p>현재 실사223장의 원래8개 native 슬롯, 전체1784개를 유지했습니다. 원입력은1739개 사용 가능·45개 U입니다. 공통 paired1237개(D1147/E90)와 원래 same-slot1248개는 서로 다른 집계이며 정의를 바꾸지 않았습니다. 아래 주표는 같은 원 q/crop에서 비교할 수 있는 paired1237개입니다.</p>
<p>D는 사람 주석의 보임, E는 외부가림, U는 보류입니다. 사람 S·화면 밖·미주석·매핑 미확정은 외부가림 정답으로 바꾸지 않았습니다. U도 recall 분모에 유지했습니다. 기존 baseline 판정 규칙과 새 head의 고정 raw0를 사용했고 문턱은 고르지 않았습니다.</p>
<div class="flag">현재223은 평가 전용이지만 이미 개발에 노출된 자료입니다. legacy에는 과거 private 실사 학습·CAL 개발과 평가 계보의 중첩이 있습니다. 이 비교를 독립 blind test나 공정한 공개자료 전용 모델 간 대회 결과로 해석하지 않습니다.</div>
<h2>7모델 전체 및 재질별 결과</h2>
'''
for group in groups:
    body+=f'<h3>{esc(names[group])}</h3><div class="scroll">{metric_table(group)}</div>'
body+=f'''
<p>C16은 보임1137/1147·외부가림35/90입니다. C37은 보임991/1147·외부가림66/90으로 외부가림은 늘었지만 보임은 줄었습니다. 같은 항목끼리 보면 C16 대비 외부가림31개를 회복하는 동안 원래 맞던 D148개를 손상했고, 다른 D2개를 회복했습니다. 따라서 외부가림 recall 증가만으로 성공을 선언할 수 없습니다.</p>
<figure><img src="../current_result_summary_agent/{plots[0]}" alt="전체1237에 대한7모델 D/E/U 혼동행렬"><figcaption>실제 생성·view QA 완료. 색은 행별 비율, 숫자는 원래 개수입니다.</figcaption></figure>
<h2>좌표가 가까웠던 기존376개 진단에서도 손상이 남습니다</h2>
<p>원래 manual/D/reference-error≤3 진단376개(목재153·정사각형223)를 그대로 보존했습니다. primary 가시성 평가를 이 조건으로 거르지 않았습니다. 이376개를 새로운 물리 native·pixel gold로 승격하지 않았습니다.</p>
<div class="scroll"><table><thead><tr><th>모델</th><th>D 정답 /376</th><th>false E /376</th><th>U /376</th><th>목재 false E /153</th><th>정사각형 false E /223</th></tr></thead><tbody>{''.join(nearrows)}</tbody></table></div>
<figure><img src="../current_result_summary_agent/{plots[1]}" alt="목재 정사각형 지표와 고정near376 오판"><figcaption>재질별 precision/recall/U와 기존 near 진단을 함께 표시했습니다.</figcaption></figure>
<h2>위치 품질·대상 판별의 보조 출력은 어디까지 알 수 있나</h2>
<p>새 실사 좌표 품질 gold는 <strong>0개, 미확정1784개</strong>입니다. 기존 visibility 검토는 좌표를 새로 클릭하는 작업이 아니었습니다. 실제 원 manual xy와 byte가 연결되더라도 독립적인 click 사건과 물리 native 역할까지 모두 인증되지는 않습니다.</p>
<p>별도로 DIRECT-visible·명시된 원 manual xy 연결 조건에 한정한 고정 reference 거리 진단415개(near≤3:376, far&gt;20:39, 원입력 U3)를 제공합니다. 나머지1369개는 이 진단의 정답이 미확정입니다. 이는 실제 위치 정확도 gold나 가려진 점의 amodal 좌표 평가가 아닙니다.</p>
<div class="scroll smalltable"><table><thead><tr><th>모델</th><th>far 정답 /39</th><th>near 정답 /376</th><th>far precision</th><th>known 중 U</th></tr></thead><tbody>{''.join(qualrows)}</tbody></table></div>
<p class="muted">* C35 보조 손실0의 중립0 출력은 학습된 품질 모델로 해석하지 않습니다. C37/C38의 far precision50%도 far1개 정답·예측2개라는 작은 분모와 함께 봐야 합니다.</p>
<p>현재 실사에는 독립 entity 정답이 없어 보조 대상 판별의 정확도는 계산하지 않았습니다. C37/C38은 paired의 사람 D에서644/1147, 사람 E에서57/90에 entity 음수를 냈습니다. 이는 예측 반응 진단이며 entity 오류율이 아닙니다. 보조 점수로 primary visibility를 gate하거나 좌표 오차로 표본을 제외하지 않았습니다.</p>
<figure><img src="../current_result_summary_agent/{plots[2]}" alt="조건부 reference 거리 및 entity 음수 반응과 전체 주석 분모"><figcaption>주 가시성 결과, 참조 거리 진단, entity 반응의 권위를 구분했습니다.</figcaption></figure>
<h2>전체1784개와 보류를 보존했습니다</h2><div class="scroll smalltable"><table><thead><tr><th>사람 역할</th><th>개수</th></tr></thead><tbody>{inventory}</tbody></table></div>
<p>전체 원입력 U45와 paired1237에서의 U는 같은 값이 아닙니다. 예를 들어 paired에서 새 head의 U가0이어도 전체1784개에서 원입력 U45는 그대로입니다. legacy의 RGB 보류와 기존 기하 S 응답도 원래 의미를 유지합니다. S에 대한 이진 hidden 반응을 자기 가림 학습 정확도로 주장하지 않습니다.</p>
<h2>실제 crop 확인은 아직 완료되지 않았습니다</h2>
<div class="flag"><strong>crop 갤러리 렌더 오류 복구 대기</strong><p>현재 초안은 저장 수치와 위3개 과학 그림의 QA까지 완료한 상태입니다. 원래 crop 갤러리의 복구·실제 시각 QA가 남아 있으므로 전체 시각검증 완료로 표시하지 않습니다. 복구된 별도 완료 seal과 그림은 root 최종본에서 연결할 예정입니다.</p></div>
<h2>원인을 어디까지 말할 수 있나</h2>{listitems(mechanisms)}
<p>C35/C36의2000회 설정과 C37/C38의8000회 설정을 단순 비교해 경로 분리만의 효과라고 주장하지 않습니다. 공개실사 약한 라벨과 대상·역할 범위, source held 일반화, 실제 초기 q의 품질을 함께 고려해야 하지만, 이 보고서로 하나의 실패 원인을 확정하거나 새 구조를 선택하지 않습니다.</p>
<h2>다음 행동은 평가 전에 정한 source 점검을 유지합니다</h2>
<p>{esc(future['action_KO'])}</p><p>{esc(future['fixed_limits_KO'])}</p>
<p>이번 보고서는 새 학습을 시작하지 않습니다. C35/C36/C37/C38 모두 source 자격 실패·미채택 상태를 유지하며 selected model은 없습니다. 다음 학습 구조·손실·문턱을 현재223 결과로 고르지 않습니다.</p>
<details><summary>해석 범위와 근거 파일</summary>{listitems(limitations)}<p><a href="ROOT_RESULTS_DRAFT.json">이 초안의 구조화된 결과</a> · <a href="../current_result_summary_agent/CURRENT_RESULTS_SUMMARY_KO.json">저장 지표 요약</a> · <a href="../saved_seven_current_evaluation_v1/EVALUATION_SEAL.json">평가 완료 seal</a> · <a href="../frozen_four_head_current_benchmark_v1/PREDICTION_SEAL.json">고정 추론 완료 seal</a> · <a href="../../query_visibility_independent_auxiliary_source_control_20261024_v155/ROOT_NEXT_ACTION_COMMITMENT_BEFORE_ANY_NEW_PRIVATE_READS.json">평가 전 source 다음 행동 약속</a></p></details>
<p class="muted">초안 생성은 완료된 집계 JSON·seal만 읽었습니다. 새 모델 호출·학습·원RGB·feature·weight·V112 접근0. 원본 보고서와 부모 포인터는 변경하지 않았습니다.</p></html>'''
with (OUT/'ROOT_READOUT_DRAFT.html').open('x') as f: f.write(body)
dump('DRAFT_STRUCTURE_QA.json',{'completed':True,'PASS':True,'draft_only':True,'outputs':[bind(OUT/n)for n in('ROOT_RESULTS_DRAFT.json','ROOT_READOUT_DRAFT.html')],'inputs':draft['inputs'],'all7_global_material_near':True,'full1784_U45_distinct_from1237':True,'quality_primary_gold0_secondary415':True,'crop_visualization_pending_explicit':True,'no_new_future_fit_choices_from_current':True,'root_reports_or_pointers_changed':False,'browser_render_or_screenshot_performed':False})
print(json.dumps({'completed':True,'files':[bind(OUT/n)for n in('ROOT_RESULTS_DRAFT.json','ROOT_READOUT_DRAFT.html','DRAFT_STRUCTURE_QA.json')]},ensure_ascii=False))
