import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
V=Path(__file__).resolve().parent;P=V.parent
N=P/'query_visibility_consistent_query_context_20261023_v119'
RUN=V/'runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY'
CUR=V/'private_evaluation/C13_FROZEN_CURRENT223';EV=CUR/'evaluation'
def b(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def r(p):return json.loads(Path(p).read_text())
def w(p,d):
 with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
stamp=datetime.now(timezone.utc).isoformat()
assert r(V/'saved_source_review/REVIEW.json')['PASS'] and r(V/'saved_current_review/REVIEW.json')['PASS']
assert r(V/'report_preparation/REPORT_QA.json')['PASS']
qa=dict(completed=True,PASS=True,created_UTC=stamp,report=b(V/'ROOT_READOUT_KO.html'),renderer_QA=b(V/'report_preparation/REPORT_QA.json'),
 root_actually_viewed=[b(V/'report_preparation/qa'/n)for n in ['HTML_TOP_FINAL.png','HTML_CURRENT_FINAL.png','HTML_PROBE_FINAL.png']],
 root_actual_source_contact_view=b(V/'source_visual_followup/CONTACT_FIXED3_SOURCE_TRIPLES.png'),
 observations_KO=['상단에서 공개 source 개선과 실사 외부가림 미개선을 명확히 구분한다.','실사 CM·다섯 모델 precision/recall/U와 SQ 개별1개 개선/1개 퇴행이 저장 결과와 일치한다.','공개 probe 완료 수치와 비물리 feature hybrid 한계, S/좌표 분리 표가 읽힌다.'],
 goal_solved=False,root_results_point_in_time_preserved=True)
w(V/'ROOT_REPORT_VISUAL_QA.json',qa)
final=dict(completed=True,created_UTC=stamp,source_fit_session=81199,current_head_session=74373,source_feature_session=32421,all_terminal_exit_code=0,
 source_fit_complete=b(RUN/'FIT_COMPLETE.json'),current_inference=b(CUR/'INFERENCE_SEAL.json'),current_evaluation=b(EV/'EVALUATION_SEAL.json'),
 source_saved_review=b(V/'saved_source_review/REVIEW.json'),current_saved_review=b(V/'saved_current_review/REVIEW.json'),
 source_probe_completed=b(V/'public_query_grounding_probe/CONCISE_INTERPRETATION_AND_EXECUTION_CLOSURE.json'),
 context_provenance_audit=b(V/'public_context_provenance_audit/RESULT.json'),
 completion_audit=r(V/'ROOT_RESULTS_KO.json')['completion_audit'],
 conclusion_KO='C13의 공개 목재 가림 개선은 현재 촬영 자료의 외부가림 재현율로 전이되지 않았다. 전체 목표는 미완료이며 모델·문턱을 현재 자료로 고르지 않았다. 다음 입력 정의 통제는 공개 metadata/code provenance 감사에 근거한다.',
 no_new_C14_fit=True,goal_status='active',fully_solved=False)
w(V/'ROOT_ACTUAL_CURRENT_AND_DIAGNOSTIC_CLOSURE.json',final)
files=[V/n for n in ['ROOT_C13_DESIGN_BEFORE_SOURCE_FORWARD.json','ROOT_ACTUAL_C13_SOURCE_FIT_ENTRY.json','ROOT_ACTUAL_C13_SOURCE_EXECUTION.json','ROOT_ACTUAL_C13_FROZEN_CURRENT_HEAD_ONLY_ENTRY.json','ROOT_ACTUAL_C13_SAVED_CURRENT_EVALUATION_ENTRY.json','ROOT_RESULTS_KO.json','ROOT_READOUT_KO.html','ROOT_REPORT_VISUAL_QA.json','ROOT_ACTUAL_CURRENT_AND_DIAGNOSTIC_CLOSURE.json','public_source_comparison/RESULT.json','source_visual_followup/RESULT.json','public_query_grounding_probe/RESULT.json','public_query_grounding_probe/CONCISE_INTERPRETATION_AND_EXECUTION_CLOSURE.json','public_context_provenance_audit/RESULT.json','saved_source_review/REVIEW.json','saved_current_review/REVIEW.json']]
files += [RUN/'FIT_COMPLETE.json',RUN/'MODEL_FREEZE.json',CUR/'INFERENCE_SEAL.json',EV/'EVALUATION_SEAL.json']
w(V/'ARTIFACT_INDEX_PHASE_RESULTS.json',dict(completed=True,scope='C13 augmentation fit, source/current comparison and source-only diagnostics; not overall goal success',artifacts=[b(p)for p in files]))
w(V/'ROOT_PHASE_RESULT_SEAL.json',dict(completed=True,created_UTC=stamp,phase='C13_SOURCE_CURRENT_AND_DIAGNOSTICS_COMPLETE',index=b(V/'ARTIFACT_INDEX_PHASE_RESULTS.json'),root_closure=b(V/'ROOT_ACTUAL_CURRENT_AND_DIAGNOSTIC_CLOSURE.json'),report=b(V/'ROOT_READOUT_KO.html'),QA=b(V/'ROOT_REPORT_VISUAL_QA.json'),current_source_selected_update=1000,current_external_E={'WOOD':[8,29],'SQUARE':[6,61]},goal_status='active',fully_solved=False,next_work=str(N)))
backup=V/'provenance/PARENT_POINTERS_BEFORE_C13_CURRENT_COMPLETE';backup.mkdir(parents=True)
new_current=dict(version='V118_C13',report=str(V/'ROOT_READOUT_KO.html'),report_binding=b(V/'ROOT_READOUT_KO.html'),results=b(V/'ROOT_RESULTS_KO.json'),report_QA=b(V/'ROOT_REPORT_VISUAL_QA.json'),
 source_freeze=b(RUN/'MODEL_FREEZE.json'),source_fit_complete=b(RUN/'FIT_COMPLETE.json'),independent_source_saved_review=b(V/'saved_source_review/REVIEW.json'),source_selected_update=1000,threshold_logit=0,D_rule='z<0',source_gates_PASS=15,source_gates_total=15,
 original_native_slots=1784,valid=1739,factual_U=45,extra_operator_U=0,
 paired_D_E_rows_D_E_cols_visible_hidden_U={'WOOD':[[599,17,0],[21,8,0]],'SQUARE':[[531,0,0],[55,6,0]]},
 manual_DIRECT_error_at_most3px={'WOOD':{'visible':153,'N':153},'SQUARE':{'visible':223,'N':223}},manual_near_D_is_coordinate_correctness_certificate=False,
 inference=b(CUR/'INFERENCE_SEAL.json'),evaluation=b(EV/'EVALUATION_SEAL.json'),independent_saved_review=b(V/'saved_current_review/REVIEW.json'),root_phase_seal=b(V/'ROOT_PHASE_RESULT_SEAL.json'),artifact_index=b(V/'ARTIFACT_INDEX_PHASE_RESULTS.json'),
 limitation='Latest chronological source-frozen head, not chosen as best on current. Public woodVAL27/35→34/35 but currentE remains8/29and6/61; WOODD601→599 andSQ531unchanged. Reliable external visibility goal remains unsolved. No current training/normalizer/threshold/checkpoint/arm choice; no V112 use, q/L correction or PnP. InitialgeometryS separate, no learnedS; coordinate/native correspondence not certified. Current/source development exposed, not new blind holdouts.',
 latest_does_not_mean_best_or_selected_on_current=True,current_used_for_training_or_model_selection=False)
for name in ['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']:
 path=P/name;assert not(backup/name).exists();shutil.copy2(path,backup/name);d=r(path)
 previous=d['latest_current_evaluation'];history=d.setdefault('older_current_evaluation_history',[])
 if isinstance(history,list)and not any(isinstance(x,dict)and x.get('version')==previous.get('version')for x in history):history.append(previous)
 d.update(schema='PALLET_QUERY_VISIBILITY_C13_CURRENT_UNRESOLVED_SOURCE_DIAGNOSTICS_COMPLETE_V119_CONTEXT_FEASIBILITY',updated_UTC=stamp,goal_status='active',fully_solved=False,
 latest_current_evaluation=new_current,latest_actual_current=new_current,latest_current_model_candidate='V118_C13_SOURCE_SELECTED_STEP1000_LOGIT0_NOT_PRIVATE_SELECTED_BEST',
 latest_report=str(V/'ROOT_READOUT_KO.html'),latest_results=b(V/'ROOT_RESULTS_KO.json'),latest_report_kind='C13_SOURCE_AUGMENTATION_FROZEN_CURRENT_AND_PUBLIC_GROUNDING_DIAGNOSIS_GOAL_UNSOLVED',latest_phase_seal=b(V/'ROOT_PHASE_RESULT_SEAL.json'),latest_index=b(V/'ARTIFACT_INDEX_PHASE_RESULTS.json'),
 latest_source_report=b(V/'public_source_comparison/RESULT.json'),latest_source_report_QA=b(V/'saved_source_review/REVIEW.json'),latest_source_diagnostic=b(V/'public_query_grounding_probe/CONCISE_INTERPRETATION_AND_EXECUTION_CLOSURE.json'),latest_source_context_audit=b(V/'public_context_provenance_audit/RESULT.json'),
 V118_state='COMPLETE_C13_FIT_SOURCE_CURRENT_COMPARISON_EXTERNAL_E_NO_NET_GAIN_PUBLIC_PROBES_COMPLETE',V118_source_saved_review=b(V/'saved_source_review/REVIEW.json'),V118_current_saved_review=b(V/'saved_current_review/REVIEW.json'),V118_current_inference_seal=b(CUR/'INFERENCE_SEAL.json'),V118_current_evaluation_seal=b(EV/'EVALUATION_SEAL.json'),V118_report_QA=b(V/'ROOT_REPORT_VISUAL_QA.json'),
 V118_current_E_recall={'WOOD':[8,29],'SQUARE':[6,61]},V118_current_used_for_training_normalizer_threshold_checkpoint_arm_choice=False,V118_PnP_or_q_correction=0,
 active_work=str(N),active_work_state='V119_INPUT_ONLY_CONSISTENT_Q8_CONTEXT_METADATA_NATIVE_RGB_FEASIBILITY_NO_C14_ENCODER_OR_FIT',V119_scope=b(N/'ROOT_FEASIBILITY_SCOPE_BEFORE_PIXELS.json'),
 next_required_actions=['Finish exact q8 source-box input-only manifest and native RGB replay feasibility; keep all46108 rows and originalq/L/teacher/folds.','Review feasibility then freeze one prospective source-only C14 feature/head design. Source boxes may reuse context only if exact nativeRGB and box match; no selectedbox/GT fallback or private-driven choice.','Continue toward reliable external visibility on current; source proxy gates do not replace the objective. Keep initialgeometryS separate, coordinate uncertainty explicit and no PnP.'],
 model_candidate=dict(state='V118_C13_COMPLETE_SOURCE_FROZEN_CURRENT_EXTERNAL_E_UNIMPROVED_GOAL_INCOMPLETE',selected_step=1000,source_gates_PASS=True,source_gates_pass_count=15,latest_current_model='V118_C13',source_freeze_binding=b(RUN/'MODEL_FREEZE.json'),current_used_for_model_choice=False,latest_does_not_mean_best_or_selected_on_current=True),
 known_limit=new_current['limitation'],previous_goal_turn_classification='PROGRESS_ACTUAL124_FRESH_FEATURES_2000UPDATE_C13_FROZEN_CURRENT_EVALUATION_NEGATIVE_TRANSFER_PUBLIC_FEATURE_AND_INPUT_DIAGNOSIS',previous_parent_pointers=str(backup),no_user_approval_pending=True)
 path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(completed=True,phase=b(V/'ROOT_PHASE_RESULT_SEAL.json'),report=b(V/'ROOT_READOUT_KO.html'),goal_status='active',fully_solved=False)))
