"""Consolidate completed source evidence; preserve model/evaluation state."""
from pathlib import Path
import datetime,hashlib,json,html
from collections import Counter
V=Path(__file__).resolve().parent;P=V.parent
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def bind(p):
 p=Path(p).resolve();assert '_v112'not in str(p).lower()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 definition=read(V/'native_definition_audit/DEFINITION_AND_FEASIBILITY.json')
 mask=read(V/'native_mask_support_audit/COMPLETE.json')
 bridge=read(V/'owner_supervision_bridge/COMPLETE.json')
 visual=read(V/'ROOT_ACTUAL_NATIVE_SUPPORT_VISUAL_QA.json')
 assert definition['completed']and mask['PASS']and bridge['PASS']and visual['PASS']
 assert mask['all390_retained']and mask['original_roles']=={'D':341,'E':49}
 assert bridge['input_rows']==1840 and bridge['owner_target_available_rows']==253 and bridge['source_held_rows_all_UNKNOWN']==360
 assert not bridge['training_admitted']and not definition['goal_solved']
 nearest=read(V/'native_mask_support_audit/NEAREST_PIXEL_DOMAIN_CLARIFICATION.json')
 assert nearest['nearest_pixel_counts']['ALL']['nearest_pixel_available']==390
 evidence={k:bind(V/p)for k,p in dict(definition='native_definition_audit/DEFINITION_AND_FEASIBILITY.json',native8_parts='native_definition_audit/NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl',mask='native_mask_support_audit/COMPLETE.json',mask_nearest_domain='native_mask_support_audit/NEAREST_PIXEL_DOMAIN_CLARIFICATION.json',owner_bridge='owner_supervision_bridge/COMPLETE.json',visual='ROOT_ACTUAL_NATIVE_SUPPORT_VISUAL_QA.json',native_information='ROOT_INPUT_NATIVE_HINT_INFORMATION_AUDIT.json').items()}
 coverage={k:{j:mask['counts'][k][j]for j in ('rows','nominal_ref_inside96','nominal_ref_inside32')}for k in ('ALL','ROLE_D','ROLE_E','OWNER_TARGET_VISIBLE_SUPPORTED')}
 next_action=dict(source_only=True,evidence=evidence,action_KO='별도로 고정할 다음 source 통제실험에서 중심 표면 관찰을 native D/E와 구분하는 보조 감독으로만 사용하고, 해당 부위의 정체성 정보와 시각 특징의 역할을 분리해 검증한다.',
  owner_data='All1840 sidecar retained; T53/F52/BG148 are weak categorical owner observations only, UNKNOWN1587 remains masked only for owner loss.',
  native_roles='Original native D/E, entity and coordinate-quality tables remain unchanged. No owner-to-D/E or owner-to-quality relabel.',
  qualification='Full existing WOOD/SQUARE/native8 source checks and fixed public held native visibility evaluation remain required; no owner-only success criterion.',
  held_policy='Held360 owner targets remainUNKNOWN and never move to TRAIN.',
  current_policy='Private current223 remains evaluation-only and cannot select training, threshold, model, arm or architecture.',
  no_PnP_or_qL_correction=True,new_model_selected=False,new_fit_authorized_by_this_file=False)
 write(V/'ROOT_NEXT_SOURCE_EXPERIMENT_REQUIREMENTS.json',next_action)
 result=dict(schema='V159_NATIVE_PART_FEASIBILITY_AND_SEPARATE_OWNER_DATA_COMPLETE',completed=True,stage_complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn_classification='progress',goal_status='active_unsolved',fully_solved=False,selected_model=None,
  lead_KO='실제 중심 표면과 원하는 native 부위는 다르다는 것을 공개 crop에서 확인했다. 공개 CAD/자세/마스크만으로 실사 정밀 native 가림 정답을 새로 인증할 수는 없었다. 대신 중심 표면을 별도 보조 감독으로 사용할 수 있는 전체1840 슬롯 sidecar를 실제 준비했다.',
  source_rows=390,source_frames=88,original_D_E=[341,49],all_original_qL_native_splits_roles_preserved=True,
  nominal_CAD_support='All8 AABB extrema exactly coincide with provided obj1 CAD vertices/surface; not empty unsupported CAD virtual corners. Real-photo point support remains uncertified.',
  native_authority='Camera-dynamic front/rear native8 names and correspondence derive from task V74 contract, not official MR6D per-keypoint visibility annotations.',
  pose_uncertainty={'paper':'https://arxiv.org/html/2508.13775v1#S3.SS2','dimension_variation':'few centimeters','expected_static_accuracy':'low-centimeter range','per_frame_or_3px_bound_supplied':False},
  coverage=coverage,mask_support=dict(nearest_q_visible=95,nearest_nominal_visible=154,total390=390,nominal_bilinear_any_touch=243,nominal_bilinear_all_touch=59,globally_overlapping_visible_instance_frames=42,total_frames=88,mask_ownership_gold=False,physical_native_gold=False,original_nearest_available390_preserved=True),
  actual_visual_scope=dict(fixed_cases=12,contacts_viewed=3,additional_full_panels_viewed=[1075,94,93,1613],native_D_slots=list(range(8)),E_slots=[4,5]),
  visual_findings_KO=[
   'Q1075: 중심은 카펫이어도 바로 옆 native 모서리 블록은 원래 crop에 보인다. 배경 중심을 E로 바꿀 수 없다.',
   'Q94: 중심에 같은 팔레트 목재가 있지만 원하는 native6 명목 부위와는 떨어져 있다. 목재가 보인다는 사실이 정확한 부위라는 뜻은 아니다.',
   'Q93: 원래 중심은 뒤 검은 지지대이고 명목 native5는 오른쪽 목재 끝 근처이다. native D와 q 중심의 소유는 구분해야 한다.',
   'Q1613: 중심에 목재가 보여도 명목 native5 부근은 외부 화물 쪽이다. 중심 TARGET과 native 역할 E는 양립할 수 있다.'
  ],
  concrete_prepared_owner_supervision=dict(all_original_slots=1840,source_TRAIN_slots=1480,source_held_slots=360,known_weak_owner=253,owner_counts=dict(TARGET=53,FOREIGN=52,BACKGROUND=148,UNKNOWN=1587),held_owner_all_UNKNOWN=True,new_judgments=False,training_admitted=False,original_visibility_quality_untouched=True),
  native_information_KO='현재 C37/C38 입력1928에는 명시적인 native 순번이 없고 G42는0이다. 선행 V133 G42에는 순번과 상대기하가 이미 포함됐으므로, native 정보를 처음 시험한다는 주장은 하지 않는다. 새 owner 감독과 현재 source 구성에서 입력/목표를 구분할 근거만 확인했다.',
  remaining_limits_KO=['MR6D obj1은 Euro WOOD이며 실사 SQUARE/다양한 WOOD 설계와 native6 범위가 부족하다. 기존 full WOOD/SQUARE/native8 목표는 유지한다.','마스크는 팔레트 인스턴스만 포함하고 외부 화물 전체를 주지 않는다. 비배타적 겹침과 raster 경계 민감성이 있어 마스크 밖을 배경/가림 정답으로 볼 수 없다.','53개 중심 TARGET도 native 좌표/부위 정답53개가 아니다. 6/53개는 명목 참조가 원래96 범위 밖,21/53개는 중앙32 범위 밖이다. 명목 crop 포함 자체도 실사 정확도 인증은 아니다.','이번 단계 새 모델 성능 향상이나 실용 성공은 입증하지 않았다.'],
  evidence=evidence,next_source_requirements=bind(V/'ROOT_NEXT_SOURCE_EXPERIMENT_REQUIREMENTS.json'),new_native_visibility_or_real_coordinate_gold=0,new_model_fit_encoder_R0_private_V112_PnP_qL_calls=0,live_processes=[])
 write(V/'ROOT_RESULTS_KO.json',result)
 rows=''.join(f'<tr><th>{lab}</th><td>{coverage[k]["rows"]}</td><td>{coverage[k]["nominal_ref_inside96"]}</td><td>{coverage[k]["nominal_ref_inside32"]}</td></tr>'for k,lab in [('ROLE_D','원래 약한 D'),('ROLE_E','원래 약한 E'),('OWNER_TARGET_VISIBLE_SUPPORTED','중심 TARGET'),('ALL','전체')])
 cards=[('1075','CASE_04_Q1075.png','중심은 바닥이어도 모서리는 보임'),('94','CASE_07_Q0094.png','목재가 보여도 원하는 부위와 다름'),('93','CASE_11_Q0093.png','보이는 native 부위와 초기 중심이 다름'),('1613','CASE_12_Q1613.png','중심 목재와 native E가 양립')]
 gallery=''.join(f'<article><h3>Q{q}: {caption}</h3><a href="native_support_visual12/{f}"><img src="native_support_visual12/{f}"></a></article>'for q,f,caption in cards)
 doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>native 부위와 초기 crop 대조 · V159</title><style>body{margin:0;background:#f3f6fa;color:#20344a;font:17px/1.7 system-ui,sans-serif}main{max-width:1200px;margin:35px auto;padding:0 24px 70px}section,article{padding:24px;background:white;border:1px solid #dce4ee;border-radius:12px;margin:22px 0}h1{font-size:34px;line-height:1.4}h2{font-size:26px}h3{font-size:21px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:right}th:first-child{text-align:left}.tag{color:#a54422;font-weight:bold}.note{font-size:15px;color:#53657a}a{color:#176baa}</style><main><p class="tag">V159 · source 준비 완료 · 목표 미해결 · 새 모델 채택 없음</p><h1>중심에 무엇이 보이는가와<br>원하는 키포인트 부위가 보이는가는 다릅니다</h1><p>공개 TRAIN 390개를 같은 native/선택 인스턴스의 제공 CAD·자세·마스크와 대조했습니다. 중심 표면 정보는 별도 보조 감독으로 쓸 수 있도록 전체1840 슬롯에 연결했습니다. native D/E나 좌표 정답으로 바꾸지는 않았습니다.</p><section><h2>실제 crop에서 확인한 네 가지 차이</h2><p>자홍색은 원래 q와 crop, 노란 원은 제공 pose/CAD의 명목 native 위치입니다. 오른쪽 명목 위치 확대는 진단용 새 창이며 모델 입력이 아닙니다. 원래 좌표·crop은 그대로입니다.</p>'''+gallery+'''<p class="note">사전에 고정한12개 중 네 사례입니다. 나머지까지 포함한 <a href="native_support_visual12/CONTACT_01.png">1페이지</a> · <a href="native_support_visual12/CONTACT_02.png">2페이지</a> · <a href="native_support_visual12/CONTACT_03.png">3페이지</a>를 실제 확인했습니다. 이 관찰은 약한 장면 근거이며 정확한 native/물리 가림 정답이 아닙니다.</p></section><section><h2>명목 부위가 원래 crop에 들어오는 범위</h2><table><thead><tr><th>범주</th><th>전체</th><th>96 crop 안</th><th>중앙32 안</th></tr></thead><tbody>'''+rows+'''</tbody></table><p>중심에 대상 목재가 있다고 관찰한53개도, 명목 native 위치는6개에서 원래 crop 밖이고21개에서 중앙32 밖입니다. 대상 목재 판별을 native 좌표/보임 정답으로 그대로 쓰면 안 되는 이유입니다.</p><p class="note">포함 여부는 제공 CAD/자세에 따른 명목 계산입니다. 실사 좌표 오차나 물리 부위의 포함 여부를 인증하는 수치는 아닙니다.</p></section><section><h2>CAD와 마스크가 주는 근거의 한계</h2><p>제공 CAD에서는8개 nominal corner가 모두 실제 mesh 꼭짓점/표면에 있습니다. 빈 공간의 가상 CAD 점으로 확인된 것이 아닙니다. 다만 카메라에 따른 앞·뒤 native 이름은 과제의 V74 계약에서 온 것이며 MR6D 공식 점별 가림 어노테이션은 아닙니다.</p><p>선택 팔레트의 visible mask를 최근접 픽셀로 읽으면 q에서95/390, 명목 위치에서154/390이 닿습니다. 명목 위치의 bilinear 이웃 중 하나라도 닿는 경우243/390과 모두 닿는 경우59/390이 크게 다릅니다. 모서리와 raster 경계의 민감성으로, 마스크를 점별 정답으로 쓰기 어렵습니다.</p><p>88프레임 중42개는 visible instance mask끼리 겹칩니다. 마스크는 외부 화물 전체를 포함하지 않습니다. 마스크 밖을 배경이나 외부 가림으로 판정할 근거가 아닙니다.</p><p class="note">원래390개 availability는 유지했습니다. Q1043은 최근접 픽셀을 읽을 수 있지만 원래 crop 중심4표본 중2개만 유효하여 owner UNKNOWN입니다. bilinear 이웃 전체가 영상 안인389개와 원래 available390개를 구분합니다.</p><p>MR6D 논문도 팔레트 치수의 cm 단위 차이와 정적 장면의 cm 수준 예상 정확도를 설명합니다. 이로부터 프레임별3px 보장이나 임의의 픽셀 오차 한계를 만들 수는 없습니다. <a href="https://arxiv.org/html/2508.13775v1#S3.SS2">원 논문</a></p></section><section><h2>실제로 준비한 학습자료</h2><p>전체1840 슬롯을 보존한 별도 owner sidecar를 만들었습니다. 대상53 / 다른 물체52 / 배경148은253개의 약한 범주형 관찰이고, 나머지1587개는 UNKNOWN입니다. 공개 held360개도 모두 owner UNKNOWN으로 남겼습니다. 기존 native D/E·entity·좌표 품질 테이블은 변경하지 않았습니다.</p><p>390개의 원래 관찰은350개 단일 약검토,30개 세 관찰 일치,10개 불일치 정책 적용 이력을 그대로 유지합니다. 새 사람/정밀 native 좌표 정답을 만들거나 자동으로 학습에 편입한 것은 아닙니다.</p><p>현재 C37/C38의 시각 입력에는 native 순번이 명시적으로 들어가지 않고 G42는0입니다. 앞선 V133에서는 순번과 상대기하를 묶은 G42를 이미 다뤘습니다. 새로운 owner 관찰과 native 부위 판정을 분리해 시험할 근거가 준비된 상태입니다.</p><p class="tag">실용 성능 향상은 아직 미입증입니다. 다음 source 프로토콜에서 원래 목재·정사각형·native8 평가 범위를 유지해 검증합니다.</p></section><section><h2>기록</h2><p><a href="ROOT_RESULTS_KO.json">전체 결과</a> · <a href="owner_supervision_bridge/ALL1840_SEPARATE_OWNER_AUXILIARY_SUPERVISION.jsonl">1840개 별도 owner 감독</a> · <a href="native_definition_audit/NATIVE8_EXPECTED_PART_AND_PROOF_REQUIREMENTS.jsonl">native8 부위와 근거 요구사항</a> · <a href="native_mask_support_audit/COMPLETE.json">390개 마스크/기하 감사</a> · <a href="ROOT_INPUT_NATIVE_HINT_INFORMATION_AUDIT.json">입력 정보와 선행 시험</a></p><p class="note">이번 단계는 공개 source만 사용했습니다. 새 학습/모델 추론/PnP/좌표 변경은 없습니다. 최근 실제 학습 V155와 현재 평가 V156은 유지됩니다. 전체 목표는 미해결입니다.</p></section></main></html>'''
 with(V/'ROOT_READOUT_KO.html').open('x')as f:f.write(doc)
 phase=dict(completed=True,stage_complete=True,goal_status='active_unsolved',fully_solved=False,selected_model=None,results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(V/'ROOT_READOUT_KO.html'),evidence=evidence,all390_preserved=True,all1840_owner_sidecar_prepared=True,training_admitted=False,new_model_fit_private_PnP_qL_calls=0,HTML_browser_render_confirmed=False,actual_source_PNG_QA=evidence['visual'],live_processes=[])
 write(V/'ROOT_PHASE_RESULT.json',phase)
 publish=[]
 for name in('LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json'):
  q=P/name;before=q.read_bytes();(V/'provenance'/('BEFORE_PUBLICATION_'+name)).write_bytes(before);d=json.loads(before)
  protected={k:d.get(k)for k in('latest_current_evaluation','latest_source_training','latest_actual_current','latest_phase_result_V156')}
  d.update(schema='PALLET_QUERY_VISIBILITY_V159_SOURCE_NATIVE_FEASIBILITY_COMPLETE_UNSOLVED',updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active_unsolved',fully_solved=False,selected_model=None,active_work=str(V),active_work_state='V159_SOURCE_NATIVE_FEASIBILITY_AND_OWNER_BRIDGE_COMPLETE',latest_completed_work=str(V),latest_report=str(V/'ROOT_READOUT_KO.html'),latest_report_kind='V159_NATIVE_PART_FEASIBILITY_AND_SEPARATE_OWNER_DATA_NOT_MODEL_SUCCESS',latest_source_report=bind(V/'ROOT_READOUT_KO.html'),latest_source_report_QA=evidence['visual'],latest_report_QA=evidence['visual'],latest_results=bind(V/'ROOT_RESULTS_KO.json'),latest_index=bind(V/'ROOT_RESULTS_KO.json'),latest_source_diagnostic=bind(V/'ROOT_RESULTS_KO.json'),latest_phase_result=bind(V/'ROOT_PHASE_RESULT.json'),latest_phase_seal=bind(V/'ROOT_PHASE_RESULT.json'),latest_source_phase_seal=bind(V/'ROOT_PHASE_RESULT.json'),V159_state='COMPLETE_NATIVE8_CAD_MASK390_AND_OWNER1840_PREPARATION',V159_phase_result=bind(V/'ROOT_PHASE_RESULT.json'),V159_results=bind(V/'ROOT_RESULTS_KO.json'),latest_source_owner_supervision_bridge=evidence['owner_bridge'],live_processes=[])
  assert all(d.get(k)==x for k,x in protected.items())
  tmp=q.with_suffix('.v159.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(q);publish.append(dict(parent=bind(q),latest_actual_current_V156_and_source_fit_V155_preserved=True))
 write(V/'ROOT_PARENT_POINTER_PUBLICATION_QA.json',dict(completed=True,PASS=True,parents=publish,phase=bind(V/'ROOT_PHASE_RESULT.json')))
 print(json.dumps(dict(completed=True,report=bind(V/'ROOT_READOUT_KO.html'),phase=bind(V/'ROOT_PHASE_RESULT.json')),ensure_ascii=False))
if __name__=='__main__':main()
