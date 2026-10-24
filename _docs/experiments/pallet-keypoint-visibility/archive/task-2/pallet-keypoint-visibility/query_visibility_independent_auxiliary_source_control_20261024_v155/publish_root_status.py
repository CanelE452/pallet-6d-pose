"""Publish completed V155 source metadata, preserving all private pointers."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

V=Path(__file__).resolve().parent
P=V.parent
R=V/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000'


def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p)
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


assert read(V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json')['PASS']
phase=read(V/'ROOT_PHASE_RESULT.json')
assert phase['stage_complete'] and not phase['fully_solved']
control=read(R/'CONTROL_COMPLETE.json')
names=['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']
private_keys=['latest_current_evaluation','latest_actual_current','active_prediction_seal']
snapshot=V/'provenance/PARENT_POINTERS_BEFORE_V155_COMPLETE'
snapshot.mkdir(parents=True,exist_ok=False)
for name in names:shutil.copy2(P/name,snapshot/name)
now=datetime.now(timezone.utc).isoformat()
state='V155_COMPLETE_FROZEN_HIDDEN_DIAGNOSIS_AND_TWO_INDEPENDENT_AUX_FIXED8000_FITS_BOTH_SOURCE_INELIGIBLE'
training={
 'version':'V155_C37_C38_INDEPENDENT_AUXILIARY_WARM_FRESH_FIXED8000',
 'directory':str(V),'control_complete':bind(R/'CONTROL_COMPLETE.json'),
 'root_execution':bind(V/'ROOT_ACTUAL_INDEPENDENT_SOURCE_FIT_EXECUTION_RECEIPT.json'),
 'independent_saved_review':bind(V/'independent_saved_fit_review/REVIEW.json'),
 'updates_each':8000,'actual_total_updates':16000,'fixed_arms':list(control['arms']),
 'model_freezes':{a:bind(R/a/'MODEL_FREEZE.json') for a in control['arms']},
 'fixed_held_available_D_E':[93,52],'C37_held_correct_D_E':[91,8],
 'C38_held_correct_D_E':[93,2],'native35_pass_C37_C38':[22,25],
 'native35_total':35,'all_source_qualification_pass':False,
 'same_final_auxiliary_states_and_all_outputs':True,
 'selected_model':None,'current_allowed':False,'new_private_inference':0,
}
commit=bind(V/'ROOT_NEXT_ACTION_COMMITMENT_BEFORE_ANY_NEW_PRIVATE_READS.json')
status='C37/C38 각8000회 실제학습·전체28976행 독립검산 완료. 가림/보조 gradient·optimizer·clip 분리와 최종보조출력동일성 확인. sourceheld C37 D91/E8,C38 D93/E2 (D93/E52분모), C38AUC.892이나 native35는22/35·25/35로 둘다미달. 단일global문턱으로 전체기준동시충족불가. 합성quality far60/66·near1152/1156·farprecision93.8%, real품질미증명. 현재223신규평가0·최신V149유지, 다음4모델동결평가초안은inert.'
checks={}
for name in names:
 d=read(snapshot/name)
 private_before={k:d.get(k) for k in private_keys}
 previous_training=d.get('latest_source_training')
 d.update({
  'schema':'PALLET_QUERY_VISIBILITY_V155_INDEPENDENT_AUX_SOURCE_FITS_COMPLETE_UNSOLVED',
  'updated_UTC':now,'goal_status':'active_unsolved','fully_solved':False,
  'selected_model':None,'model_candidate':None,'active_work':str(V),
  'active_work_state':state,'V155_state':state,'latest_completed_work':str(V),
  'latest_report_kind':'V155_SOURCE_FROZEN_MECHANISM_AND_MATCHED_WARM_FRESH_FITS_NOT_CURRENT_SUCCESS',
  'latest_report':str(V/'ROOT_READOUT_KO.html'),
  'latest_source_report':bind(V/'ROOT_READOUT_KO.html'),
  'latest_source_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
  'latest_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
  'latest_results':bind(V/'ROOT_RESULTS_KO.json'),'latest_index':bind(V/'ROOT_RESULTS_KO.json'),
  'latest_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),
  'latest_phase_result':bind(V/'ROOT_PHASE_RESULT.json'),
  'latest_source_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),
  'previous_source_training_before_V155':previous_training,
  'latest_source_training':training,
  'latest_source_diagnostic':bind(V/'independent_saved_fit_review/GLOBAL_THRESHOLD_FEASIBILITY_DIAGNOSTIC.json'),
  'latest_source_hidden_diagnostic':bind(V/'frozen_hidden_probe_agent/SAVED_PROBE_REVIEW.json'),
  'latest_source_context_audit':bind(V/'source_result_summary_agent/FINAL_SUMMARY_QA.json'),
  'next_safe_source_action':{**commit,'state':'FROZEN_FOUR_MODEL_CURRENT_BENCHMARK_FUTURE_PHASE_DRAFT_ONLY_AND_SOURCE_CROP_GENERALIZATION_DIAGNOSIS'},
  'next_source_representation_design':{**commit,'state':'NO_NEW_TRAINING_DESIGN_SELECTED_FROM_PRIVATE;SOURCE_ONLY_DIAGNOSTIC_NEXT'},
  'next_required_actions':[
   '별도 단계에서 모든C35/C36/C37/C38 최종가중치·readout을 먼저동결하고, 기존C16 ALL1784x2 byteparity후 같은전체입력으로동결평가한다. 원래1237/1248/nearD376과45U를보존한다.',
   '실제품질평가의DIRECT수동2D·native권위를먼저코드/근거로확인한다. 숨은점·중간오차·미확인mapping은qualityunknown으로유지한다.',
   '다음학습·모델·문턱·checkpoint·arm선택은현재223결과를쓰지않는다. 새source설계전공개학습/검증context·crop범위·E/개체오류를공개자료안에서진단한다.',
   '기존held45재라벨/학습편입·PnP·q/L/xy수정과V112접근은하지않는다. WOOD/SQUARE/native8/full35범위를유지한다.',
  ],
  'current_turn_status_KO':status,
  'known_limit':'C38공개held AUC.892와합성quality신호는있으나fixedraw0외부가림2/52, native35 25/35이다. C37도E8/52·22/35로미달. entity는held대상132/244를거부하며단일global문턱해법도기존전체기준에불가능하다. 실사품질및현재223개선미증명·채택없음·목표미해결.',
  'latest_progress':{'full_goal_solved':False,'source_phase':bind(V/'ROOT_PHASE_RESULT.json'),
    'actual_source_optimizer_updates':16000,'frozen_source_replay_rows':28976,
    'frozen_probe_original_output_byteexact':True,'new_encoder_R0_private_PnP_xy_calls':0,
    'scientific_source_qualification_pass':False,'next_action_commitment':commit},
  'previous_goal_turn_classification':'PROGRESS','last_goal_turn_classification':'PROGRESS',
  'V155_results':bind(V/'ROOT_RESULTS_KO.json'),'V155_phase_result':bind(V/'ROOT_PHASE_RESULT.json'),
  'V155_source_optimizer_updates':16000,'V155_selected_model':None,
  'V155_report_review':bind(V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json'),
  'V155_future_current_preparation':bind(V/'future_current_preparation/FUTURE_PROTOCOL_DRAFT_NO_CURRENT_AUTHORIZATION.json'),
  'live_processes':[],'active_parallel_work':[],'no_user_approval_pending':True,
  'previous_parent_pointers':str(snapshot),
 })
 assert all(d.get(k)==private_before[k] for k in private_keys)
 tmp=P/(name+'.v155.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(P/name)
 after=read(P/name)
 checks[name]={k:after.get(k)==private_before[k] for k in private_keys}
 assert all(checks[name].values())
q=V/'ROOT_PARENT_POINTER_PUBLICATION_QA.json'
with q.open('x') as f:
 json.dump({'completed':True,'PASS':True,'UTC':now,'snapshot':str(snapshot),
   'updated_parent_files':{n:bind(P/n) for n in names},
   'private_evaluation_prediction_pointers_exactly_preserved':checks,
   'latest_source_training':training['version'],'goal_status':'active_unsolved','selected_model':None,
   'new_private_reads_or_model_training_calls_by_publisher':0},f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'receipt':bind(q),'private_pointers_preserved':checks},ensure_ascii=False))
