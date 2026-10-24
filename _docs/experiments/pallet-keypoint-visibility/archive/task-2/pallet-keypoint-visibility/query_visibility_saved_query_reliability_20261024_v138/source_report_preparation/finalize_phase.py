"""Record completed saved-data diagnosis and preserve current-model pointers."""
from pathlib import Path
import datetime, hashlib, json, shutil
V=Path(__file__).resolve().parents[1];P=V.parent
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
r=read(V/'ROOT_RESULTS_KO.json');qa=read(V/'ROOT_REPORT_VISUAL_QA.json')
assert r['completed']and not r['fully_solved']and qa['PASS']
for s in ('input_only_reliability_v1/INPUT_SEAL.json','saved_observability_diagnosis_v1/DIAGNOSIS_SEAL.json','independent_design_review/SAVED_DIAGNOSIS_REVIEW.json'):
    a=read(V/s);assert a['completed']and a['PASS']
scope=dict(completed=True,assessment_only=True,execution_policy_for_next_model=False,goal_status='active',fully_solved=False,
    source_evidence=r['bindings'],next_actions_KO=[
        '합성·공개 자료 안에서 영상 증거와 초기 입력 조건을 함께 쓰는 통제 실험을 구체화한다. 이전 G42 실패와 이번 단일 신호의 한계를 반영한다.',
        '위치 신뢰도와 외부 가림의 목표를 분리하고 원본 q/L, 모든 known-DE·보류 분모, 기존 64개 source 조건을 보존한다.',
        '새 학습 전 비교군·입력·예산·고정 문턱과 선택 절차를 별도로 정한다. 실제 평가 결과를 보고 새 모델을 고르지 않는다.'],
    no_fit_model_private_or_new_threshold_executed=True,phase_result_does_not_certify_real_transfer=True)
write(V/'ROOT_NEXT_SOURCE_SCOPE.json',scope)
phase=dict(completed=True,PASS=True,phase='V138_SAVED_QUERY_INPUT_RELIABILITY',goal_status='active',fully_solved=False,goal_turn_classification='PROGRESS',
    actual_jobs=[dict(stage='input_only_table',session_id=96411,exit_code=0,rows=26624),dict(stage='saved_GT_only_diagnosis',session_id=16879,exit_code=0,records=15168)],
    all_jobs_terminal=True,new_model_forward=0,training_updates=0,new_current_image_label_prediction_body_reads=0,PnP=0,q_or_L_correction=0,new_threshold_or_model_selection=0,
    input_seal=bind(V/'input_only_reliability_v1/INPUT_SEAL.json'),binning_before_GT=bind(V/'saved_observability_diagnosis_v1/BINNING_SEAL_BEFORE_GT.json'),
    diagnosis_seal=bind(V/'saved_observability_diagnosis_v1/DIAGNOSIS_SEAL.json'),independent_input_review=bind(V/'independent_design_review/SAVED_INPUT_TABLE_REVIEW.json'),
    independent_diagnosis_review=bind(V/'independent_design_review/SAVED_DIAGNOSIS_REVIEW.json'),results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(V/'ROOT_READOUT_KO.html'),QA=bind(V/'ROOT_REPORT_VISUAL_QA.json'),next_scope=bind(V/'ROOT_NEXT_SOURCE_SCOPE.json'),
    source_improvement_is_historical_V137=True,current_latest_model='V124_C16',current_D_correct_total=[1137,1147],current_E_correct_total=[35,90],no_model_adopted=True)
write(V/'ROOT_PHASE_RESULT.json',phase)
backup=V/'provenance/PARENT_POINTERS_BEFORE_V138_COMPLETE';backup.mkdir(parents=True,exist_ok=False)
for name in ('LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json'):
    p=P/name;shutil.copy2(p,backup/name);d=read(p);old_current=d.get('latest_current_evaluation');old_model=d.get('model_candidate')
    d.update(schema='PALLET_QUERY_VISIBILITY_V138_SAVED_SOURCE_OBSERVABILITY_COMPLETE_APPLICATION_UNSOLVED',updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active',fully_solved=False,
        latest_report=str(V/'ROOT_READOUT_KO.html'),latest_source_report=bind(V/'ROOT_RESULTS_KO.json'),latest_source_report_QA=bind(V/'ROOT_REPORT_VISUAL_QA.json'),
        latest_phase_seal=bind(V/'ROOT_PHASE_RESULT.json'),latest_index=bind(V/'ROOT_RESULTS_KO.json'),active_work=str(V),V138_state='COMPLETE_SAVED_26624_INPUT_TABLE_AND_15168_FIXED_SIGN_DIAGNOSES;NO_NEW_FIT_MODEL_OR_CURRENT_FORWARD',
        V138_input_seal=bind(V/'input_only_reliability_v1/INPUT_SEAL.json'),V138_diagnosis_seal=bind(V/'saved_observability_diagnosis_v1/DIAGNOSIS_SEAL.json'),V138_results=bind(V/'ROOT_RESULTS_KO.json'),V138_phase=bind(V/'ROOT_PHASE_RESULT.json'),
        V138_next_scope=bind(V/'ROOT_NEXT_SOURCE_SCOPE.json'),V138_all_jobs_terminal=True,no_user_approval_pending=True,
        known_limit='V137 synthetic coverage improves source D/E but C29 remains ineligible44/64. V138 confidence/edge signals show modest location/visible-mistake association and weak direct external visibility; no simple confidence/geometry rule adopted. Current C16 real D1137/1147,E35/90 unchanged.',
        next_required_actions=scope['next_actions_KO'])
    assert d.get('latest_current_evaluation')==old_current and d.get('model_candidate')==old_model
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(phase=bind(V/'ROOT_PHASE_RESULT.json'),goal_status='active',fully_solved=False,all_jobs_terminal=True)))
