"""Seal completed public-source preparation, preserving old model benchmarks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

V = Path(__file__).resolve().parent
P = V.parent

def bind(p):
    p = Path(p)
    if not p.is_absolute():
        p = V / p
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def save(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write('\n')

review = json.loads((V/'ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json').read_text())
assert review['completed'] and review['PASS']
results = json.loads((V/'ROOT_RESULTS_KO.json').read_text())
assert results['completed'] and results['goal_status']=='active_unsolved'
assert results['selected_model'] is None and not results['fully_solved']
report, summary, visual_qa = [bind(p) for p in ['ROOT_READOUT_KO.html','ROOT_RESULTS_KO.json','ROOT_REPORT_VISUAL_QA.json']]
phase = dict(completed=True, UTC=datetime.now(timezone.utc).isoformat(),
    classification='PROGRESS_PUBLIC230_ACTUAL1840_INPUTS_FEATURES_AND_SEPARATE_SUPERVISION_PREPARATION',
    phase_only_complete=True, goal_status='active_unsolved', fully_solved=False, selected_model=None,
    outcome_KO=results['outcome_KO'], success_assessment_KO=results['success_assessment_KO'],
    report=report,results=summary,visual_QA=visual_qa,
    final_independent_report_review=bind('ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json'),
    bindings=results['bindings'], actual_executions=results['actual_executions'],
    new_source_data_and_prototype_not_model_success=True,
    source_role_proposals_train_ready=False,
    latest_private_evaluation_unchanged='V149',latest_actual_head_training_unchanged='V148',
    all_original_queries_and_full_wood_square_scope_retained=True,
    no_user_approval_pending=True,next_required_actions=results['next_required_actions'])
save(V/'ROOT_PHASE_RESULT.json',phase)
phase_binding = bind('ROOT_PHASE_RESULT.json')
backup = V/'provenance/PARENT_POINTERS_BEFORE_V153_COMPLETE'
backup.mkdir(parents=True, exist_ok=False)
updates=[]
for name in ['LATEST_STATUS_KO.json','ACTIVE_HANDOFF_KO.json']:
    p=P/name
    original=bind(p)
    shutil.copy2(p,backup/name)
    d=json.loads(p.read_text())
    unchanged={k:json.dumps(d.get(k),ensure_ascii=False,sort_keys=True) for k in
        ['latest_current_evaluation','latest_actual_current','latest_source_training','active_prediction_seal']}
    assert d['latest_current_evaluation']['version']=='V149_FROZEN_C33_C34_DIAGNOSTIC'
    assert d['latest_source_training']['version']=='V148_PUBLIC_NATIVE_WEIGHT_CONTROL'
    d.update(schema='PALLET_QUERY_VISIBILITY_V153_FULL_PUBLIC_INPUT_PREPARATION_COMPLETE_UNSOLVED',
        updated_UTC=datetime.now(timezone.utc).isoformat(),goal_status='active_unsolved',fully_solved=False,
        selected_model=None,active_work=str(V),
        active_work_state='V153_COMPLETE_FULL_PUBLIC_INPUTS_FEATURES_AND_UNTRAINED_SEPARATE_HEAD',
        V153_state='COMPLETE_230_PUBLIC_RGB_1840_ORIGINAL_Q_DINO_AND_SOURCE_VALIDATION_WEAK_REVIEW_NO_NEW_FIT',
        latest_report=str(V/'ROOT_READOUT_KO.html'),latest_source_report=report,
        latest_source_report_QA=visual_qa,latest_report_QA=visual_qa,
        latest_phase_seal=phase_binding,latest_index=summary,
        V153_results=summary,V153_phase_result=phase_binding,
        V153_source_input_counts=results['source_input_counts'],
        V153_new_separated_head=results['new_separated_head'],
        V153_source_GT_role_proposals=results['new_source_GT_role_proposals'],
        V153_actual_executions=results['actual_executions'],
        V153_selected_model=None,V153_source_semantics_train_ready=False,
        known_limit='V149 저장평가에서 C16 D1137/1147 E35/90, C33 D1072/1147 E43/90, C34 D1097/1147 E36/90. 보임 보존/외부가림 검출을 함께 해결하지 못했다. V153 공개230장 입력·feature와 미학습3출력head를 준비했으나 실제역할/개체 대응·개발감독·새학습은 미완료다. Euro목재 공개실사가 정사각형실사 공백을 채우지는 못한다.',
        active_parallel_work='V153 공개230/R01840/DINO1840/전체원crop230/validation역할360weak 검토와 독립 검산 완료. 현재 장기실행중인 모델·학습 작업은 없다. 다음은 실제 선택개체·역할 연결과 development185 감독을 준비해 source-only 학습을 고정·검증하는 작업이다.',
        live_processes=[],next_required_actions=results['next_required_actions'],
        next_safe_source_action=dict(**summary,state='NEXT_SOURCE_SUPERVISION_ASSOCIATION_BEFORE_FIT'),
        next_source_representation_design=dict(**bind('head_preparation/SEPARATED_HEAD_PROTOTYPE_CHECKS.json'),
            state='IMPLEMENTED_UNIT_CHECKED_UNTRAINED;NO_NEW_CURRENT_OR_COORDINATE_REGRESSION'),
        no_user_approval_pending=True)
    for k,val in unchanged.items():
        assert json.dumps(d.get(k),ensure_ascii=False,sort_keys=True)==val
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    updates.append(dict(parent_after=bind(p),parent_before=original,
        previous_snapshot=bind(backup/name),latest_private_and_training_unchanged=True))
save(V/'PARENT_POINTER_UPDATE_RECEIPT.json',dict(completed=True,PASS=True,phase=phase_binding,
    updates=updates,goal_status='active_unsolved',selected_model=None,
    current_model_evaluation_or_training_performed_by_pointer_update=False))
print(json.dumps(dict(phase=phase_binding,pointers=bind('PARENT_POINTER_UPDATE_RECEIPT.json'))))
