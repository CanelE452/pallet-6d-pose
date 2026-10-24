"""Publish V154 metadata after independent report review; preserve private pointers."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

V=Path(__file__).resolve().parent
P=V.parent
RUN=V/'runs/C35_C36_SEPARATE_QUERY_FIXED2000'


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


review=read(V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json')
assert review.get('PASS') is True, 'Independent final report review must pass first.'
phase=read(V/'ROOT_PHASE_RESULT.json')
assert phase['stage_complete'] and not phase['fully_solved']
control=read(RUN/'CONTROL_COMPLETE.json')
snapshot=V/'provenance/PARENT_POINTERS_BEFORE_V154_COMPLETE'
assert not snapshot.exists(), 'Do not overwrite a previous pointer snapshot.'
snapshot.mkdir(parents=True)
private_keys=['latest_current_evaluation','latest_actual_current','active_prediction_seal']
names=['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']
before={name:read(P/name) for name in names}
for name in names:
    shutil.copy2(P/name,snapshot/name)
now=datetime.now(timezone.utc).isoformat()
state='V154_COMPLETE_PUBLIC_QUERY_SUPERVISION_AND_TWO_FIXED2000_FITS_BOTH_SOURCE_INELIGIBLE'
commit=bind(V/'ROOT_NEXT_SOURCE_COMMITMENT_BEFORE_FUTURE_PRIVATE_READS.json')
next_actions=[
    '고정 source28976행에서 C32/C35/C36의 저장 출력 일치를 먼저 확인하고 hidden 결합을 진단한다. 새 encoder·R0·private·학습은 없다.',
    '대상 일치/좌표 품질의 독립 내부 경로를 구현하고, 보조 gradient와 공통 clipping이 가림 경로를 바꾸지 않는 source 대조를 사전 고정한다.',
    '원래query와 full native8·목재/정사각형·native35 범위를 유지한다. quality/GT거리로 원래 visibility 평가행을 제거하지 않는다.',
    'C35/C36 현재223 비교를 수행한다면 두 모델·모든readout을 먼저 동결하고 평가 전용으로 쓴다. 결과로 이미 고정한 구조·손실·문턱·checkpoint·arm을 선택하지 않는다.',
]
training={
    'version':'V154_MR6D_QUERY_SUPERVISION_C35_C36_FIXED2000',
    'directory':str(V),'control_complete':bind(RUN/'CONTROL_COMPLETE.json'),
    'root_execution':bind(V/'ROOT_ACTUAL_SOURCE_FIT_EXECUTION_RECEIPT.json'),
    'independent_saved_review':bind(V/'independent_saved_fit_review/REVIEW.json'),
    'updates_each':2000,'actual_total_updates':4000,
    'fixed_arms':list(control['arms']),
    'model_freezes':{a:bind(RUN/a/'MODEL_FREEZE.json') for a in control['arms']},
    'fixed_held_available_D_E':[93,52],
    'C35_held_correct_D_E':[89,13],'C36_held_correct_D_E':[64,31],
    'native35_pass_C35_C36':[22,17],'native35_total':35,
    'all_source_qualification_pass':False,'selected_model':None,
    'current_allowed':False,'new_private_inference':0,
}
status_KO='공개230장 원래query1840의 개체·역할 약한 감독을 확정하고 C35/C36 각2000회 실제학습 완료. 같은 공개held의 C35 AUC .610→.785 개선. C36 보조학습은 가림13→31/52·보임89→64/93 충돌. 둘 다source기준 미달·채택없음. 실제223 신규평가0, 다음source독립보조경로 검증은 미실행.'
checks={}
for name in names:
    d=before[name]
    previous_training=d.get('latest_source_training')
    d.update({
        'schema':'PALLET_QUERY_VISIBILITY_V154_MATCHED_SOURCE_FITS_COMPLETE_UNSOLVED',
        'updated_UTC':now,'goal_status':'active_unsolved','fully_solved':False,
        'selected_model':None,'model_candidate':None,
        'active_work':str(V),'active_work_state':state,'V154_state':state,
        'latest_completed_work':str(V),
        'latest_report_kind':'V154_PUBLIC_SOURCE_MATCHED_FITS_AND_SAVED_CAUSE_DIAGNOSIS_NOT_CURRENT_SUCCESS',
        'latest_report':str(V/'ROOT_READOUT_KO.html'),
        'latest_source_report':bind(V/'ROOT_READOUT_KO.html'),
        'latest_source_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
        'latest_report_QA':bind(V/'ROOT_REPORT_VISUAL_QA.json'),
        'latest_results':bind(V/'ROOT_RESULTS_KO.json'),
        'latest_index':bind(V/'ROOT_RESULTS_KO.json'),
        'latest_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),
        'latest_phase_result':bind(V/'ROOT_PHASE_RESULT.json'),
        'latest_source_phase_seal':bind(V/'ROOT_PHASE_RESULT.json'),
        'latest_source_training':training,
        'previous_source_training_before_V154':previous_training,
        'latest_public_native_truth_candidate':{
            'version':'V154_PUBLIC_QUERY_ENTITY_AND_ROLE_WEAK_SUPERVISION',
            'directory':str(V),'seal':bind(V/'ROOT_SUPERVISION_SEAL.json'),
            'input_binding_appendix':bind(V/'ROOT_SUPERVISION_INPUT_BINDING_APPENDIX.json'),
            'source_human_gold':False,'exact_real_pixel_quality_gold':False,
            'all1840_original_query_rows_retained':True,
        },
        'latest_source_diagnostic':bind(V/'saved_output_cause_diagnosis/CAUSE_ASSESSMENT_COMPLETE.json'),
        'latest_source_alignment_diagnostic':bind(V/'ALIGNMENT_FIGURE_VISUAL_QA.json'),
        'next_safe_source_action':{**commit,'state':'NEXT_SOURCE_FROZEN_HIDDEN_DIAGNOSTIC_THEN_INDEPENDENT_AUXILIARY_CONTROL_NOT_EXECUTED'},
        'next_source_representation_design':{**commit,'state':'SOURCE_COMMITTED_DESIGN_ONLY_NOT_YET_IMPLEMENTED_OR_FIT'},
        'next_required_actions':next_actions,
        'current_turn_status_KO':status_KO,
        'known_limit':'C35/C36 공개held 추가학습의 rank개선과 공동보조손실 충돌을 확인했지만 native35는22/35·17/35로 미달이다. 가림·보임 공동성공 및 실제223 개선은 미증명. Euro목재 weak감독과 정사각형 실사 차이·정확한 실사좌표품질 정답 공백도 남는다. 최신223 실제평가는V149 그대로다.',
        'latest_progress':{
            'full_goal_solved':False,'source_phase':bind(V/'ROOT_PHASE_RESULT.json'),
            'actual_source_optimizer_updates':4000,
            'public_frames_reviewed':230,'query_supervision_rows':1840,
            'all_source_rows_preserved':28976,
            'new_encoder_R0_private_or_PnP_xy_calls':0,
            'scientific_source_qualification_pass':False,
            'next_action_KO':next_actions[0],
        },
        'previous_goal_turn_classification':'PROGRESS',
        'last_goal_turn_classification':'PROGRESS',
        'V154_results':bind(V/'ROOT_RESULTS_KO.json'),
        'V154_phase_result':bind(V/'ROOT_PHASE_RESULT.json'),
        'V154_actual_executions':{'source_head_fit_updates':4000,'updates_each':2000,
                                  'encoder_R0_private_PnP_xy':0,'exit_code':0},
        'V154_selected_model':None,'V154_adopted':False,
        'V154_source_semantics_train_ready':'assistant weak only; fitted development split; labels immutable after fit',
        'V154_report_review':bind(V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json'),
        'live_processes':[],'active_parallel_work':[],
        'no_user_approval_pending':True,
        'previous_parent_pointers':str(snapshot),
    })
    for key in private_keys:
        assert d.get(key)==before[name].get(key)
    tmp=P/(name+'.v154.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(P/name)
    after=read(P/name); old=read(snapshot/name)
    checks[name]={key:after.get(key)==old.get(key) for key in private_keys}
    assert all(checks[name].values())
receipt={'completed':True,'PASS':True,'UTC':now,'snapshot':str(snapshot),
         'updated_parent_files':{n:bind(P/n) for n in names},
         'current_evaluation_prediction_pointers_exactly_preserved':checks,
         'latest_source_training':training['version'],
         'goal_status':'active_unsolved','selected_model':None,
         'new_private_body_reads_or_model_training_calls_by_publisher':0}
out=V/'ROOT_PARENT_POINTER_PUBLICATION_QA.json'
assert not out.exists()
out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'PASS':True,'publication_receipt':bind(out),
                  'private_pointers_preserved':checks},ensure_ascii=False))
