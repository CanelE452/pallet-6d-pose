import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).resolve().parent
P = V.parent
S = P / 'query_visibility_same_material_foreground_source_20261023_v116'
def binding(p):
    p = Path(p)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
stamp = datetime.now(timezone.utc).isoformat()
lock = json.loads((V/'SELECTION_LOCK_BEFORE_PIXELS.json').read_text())
receipt = json.loads((V/'RENDER_RECEIPTS.json').read_text())
assert [r['original_native_row'] for r in lock['selected']] == [200,840,259,1416,696,325,996,256,1021]
assert receipt['all9_rawPNG_exact_saved_RGB96'] and receipt['all_full_RGB_file_and_decoded_hashes_exact']
assert [r['original_native_row'] for r in receipt['cases']] == [r['original_native_row'] for r in lock['selected']]
assert lock['selected'][2]['full_RGB_binding'] == lock['selected'][7]['full_RGB_binding']
assert lock['selected'][2]['frozen_context_box'] == lock['selected'][7]['frozen_context_box']
assert lock['selected'][2]['original_L'] == lock['selected'][7]['original_L']

observations = [
 {'rows':[200,840], 'observed_KO':'노출된 모서리·구조의 꺾임이 보이고 C11/C12가 D로 응답한다. 840은 근처에 콘도 있다.', 'inference_limit_KO':'crop 안에 외부 물체가 있다는 사실만으로 E가 되지 않는다. 모델이 초기 좌표나 native 대응의 정확성을 검증했다는 의미는 아니다.'},
 {'rows':[259,1416], 'observed_KO':'콘이 질의 부근을 차지하고 네 모델 모두 hidden으로 응답한다.', 'inference_limit_KO':'가림 분류 성공이다. 가려진 reference의 좌표 출처는 미지정이며 정확한 위치 gold가 아니다.'},
 {'rows':[325,996], 'observed_KO':'콘과 남은 목재·초록 팔레트 재질이 함께 보이는 경계에서 C11/C12는 D, 인간 라벨은 E다. 996의 q는 남아 있는 초록 테두리 부근이다.', 'hypothesis_KO':'주변 대상 재질의 존재와 기대한 부품 단서의 가시성을 혼동할 수 있다.', 'inference_limit_KO':'시각적 공통점이며 원인으로 확정하지 않는다. E의 작은 추정 q-r로 q나 부품 대응의 정확성을 확정하지 않는다. 기존 라벨은 변경하지 않았다.'},
 {'rows':[256,1021], 'observed_KO':'콘이 crop 중심 또는 대부분을 차지하지만 C12는 D로 퇴행했다. C11은 두 사례 모두 hidden이었다.', 'inference_limit_KO':'경계나 재질 혼합만으로 모든 실패를 설명할 수 없다. 256/259는 같은 원본·context·L에서 q/슬롯이 다르고 C12 판정도 다르므로 global context 하나로도 설명할 수 없다.'},
 {'rows':[696], 'observed_KO':'야간·흐림·작은 L16.37 원본픽셀. v61/C11 hidden에서 C12 D로 바뀌어 기존 인간 D와 일치한다.', 'inference_limit_KO':'이 성공은 정보가 적은 crop의 일반적인 해결이나 독립 좌표 검증을 뜻하지 않는다.'}
]
source_refs = [
 P/'query_visibility_material_augmented_query_head_20261023_v114/public_source_regression_inventory/COMPACT_RESULT.json',
 P/'query_visibility_material_augmented_query_head_20261023_v114/ROOT_PUBLIC_REGRESSION_ACTUAL_VISUAL_RECONCILIATION.json',
 S/'corpus_build_v1/CORPUS_OUTPUT_SEAL.json', S/'independent_saved_corpus_review/REVIEW.json',
 S/'independent_source_visual_review/OBSERVATIONS_AND_RECOMMENDATION.json'
]
result = {
 'schema':'V117_ROOT_ACTUAL_SAVED_VISUAL_INTERPRETATION', 'completed':True, 'created_UTC':stamp,
 'root_actually_viewed': [binding(V/'INLINE_FIXED4_SUMMARY.png')] + [binding(V/'figures'/f'CONTACT_{i:02d}.png') for i in [1,2,3]] + [binding(V/'figures'/f'CASE_{i:02d}.png') for i in [7,8]],
 'all9_cases_actually_seen_in_contact_sheets':True,
 'selection_lock':binding(V/'SELECTION_LOCK_BEFORE_PIXELS.json'), 'render_receipt':binding(V/'RENDER_RECEIPTS.json'),
 'renderer_actual_review':binding(V/'ACTUAL_VISUAL_QA_AND_OBSERVATIONS.json'),
 'independent_actual_review':binding(V/'independent_visual_review/ACTUAL_LOCKED9_VISUAL_OBSERVATIONS_KO.json'),
 'root_observations':observations,
 'goal_assessment_KO':'초기 q에서 대상의 기대한 국소 단서가 직접 보이는지/외부 물체에 가려졌는지를 판별하는 목표 자체가 과도하다는 증거는 없다. 성공도 실제로 있지만 현재 방법은 해결되지 않았다. 모든 작은 crop의 100% 판단 가능성도 입증되지 않았다. 초기 좌표 검증·정확한 native 대응·원인 분류까지 동시에 해결한 것으로 해석하지 않는다.',
 'scope':{'fixed_initial_q_L':True,'learned_task':'conditional D versus external E after separate initial-geometry self route','new_xy_regression':False,'self_teacher_learned':False,'coordinate_certificate':False},
 'latest_C12_full_paired_saved_metrics':{'WOOD':{'D_correct':601,'D_N':616,'E_correct':8,'E_N':29},'SQUARE':{'D_correct':531,'D_N':531,'E_correct':6,'E_N':61}},
 'preexisting_public_source_evidence': [binding(p) for p in source_refs],
 'prospective_validation_directions_not_a_private_selected_model':[],
 'current_cases_are_outcome_selected_diagnostic_not_frequency_or_causal_proof':True,
 'current223_historically_development_exposed_not_fresh_blind_holdout':True,
 'new_model_encoder_forward_fit_threshold_arm_or_checkpoint_selection':0,'q_L_GT_reference_changed':False,'PnP':0,
 'overall_model_goal_solved':False
}
# Keep all prose readable to the Korean project owner.
result['prospective_validation_directions_not_a_private_selected_model'] = [
 '기존 공개 source에서 확인한 목재 foreground TRAIN 누락을 보완하는 동일 재질 cover/sham62쌍을 조건부 AUX로 준비했다. 자연 영상 전이 개선은 아직 측정하지 않았다.',
 '공개 source에서는 cover-sham 점수 차가 충분해도 cover 자체를 D로 판단하는 실패가 있었다. 상대적 점수 차와 함께 절대 D/E 판정·precision/recall·보류율을 검증해야 한다.',
 '공개 SAM 실험에서는 초기 추론 대상과 어노테이션 대상이 다른 사례가 있었다. 대상이 연결되지 않은 mask를 자동 E 교사로 사용하지 않는다.',
 '공개 source에서 경계에 남은 재질·동일 재질 전경·저조도·작은 L을 나눠 확인한다. 이번 평가9사례로 가중치·문턱·checkpoint·arm을 선택하지 않는다.'
]
save(V/'ROOT_ACTUAL_VISUAL_INTERPRETATION.json', result)

html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보임·외부가림: 성공, 실패와 해결 방향</title><style>body{margin:0;background:#eef2f6;color:#182a39;font:17px/1.8 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:24px}section{background:white;border:1px solid #d4dfe7;padding:24px;margin:18px 0;border-radius:12px}h1{font-size:30px;line-height:1.5}h2{font-size:23px}img{width:100%;height:auto}a{color:#155d90}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d4dfe7;padding:12px;text-align:left;vertical-align:top}.note{border-left:5px solid #356e98;background:#eaf3f9;padding:16px}.warn{border-left:5px solid #b66537;background:#fff0e7;padding:16px}.scroll{overflow:auto}</style><main>
<section><h1>성공과 실패를 실제 crop으로 확인했습니다</h1><p><b>현재 목표가 너무 높아서 안 된다는 증거는 없습니다. 다만 현재 방법은 아직 외부가림을 잘 구별하지 못합니다.</b> C12는 보이는 점을 목재601/616(97.6%)·정사각형531/531(100%) 살리지만, 외부가림은 각각8/29(27.6%)·6/61(9.8%)만 잡습니다.</p><p>아래는 저장된 초기 q·L·판정을 유지한 사례입니다. 이번 진단의 새 학습·모델 추론·좌표 교정·PnP는 모두0입니다. <a href="ROOT_READOUT_KO.html">전체9사례·네 모델의 지표와 원본 보기</a></p></section>
<section><h2>왼쪽 위: 보임 성공 / 오른쪽 위: 외부가림 성공<br>왼쪽 아래: 외부가림 실패 / 오른쪽 아래: 야간 보임 개선</h2><a href="INLINE_FIXED4_SUMMARY.png"><img src="INLINE_FIXED4_SUMMARY.png" alt="실제 저장 crop의 성공과 실패 네 사례"></a><p>노랑＋는 초기 q, 하늘색 원은 기존 reference r입니다. D=보임 응답, H=hidden 응답입니다. 무표시 crop과 좌표 표시 패널을 분리했습니다. E의 r은 가려진 위치의 추정/미검증 좌표이므로 작은 q-r도 정확한 위치의 증명은 아닙니다.</p></section>
<section><h2>공통되는 문제와 아직 확정하지 못한 부분</h2><div class="scroll"><table><tr><th>실제 관찰</th><th>의미와 한계</th><th>공개 source에서 검증할 방향</th></tr><tr><td>성공200/840은 모서리의 꺾임과 구조가 드러납니다. 840은 콘도 가까이 있습니다.</td><td>주변에 외부 물체가 있다는 사실만으로 해당 점이 가려진 것은 아닙니다.</td><td>같은 q에서 실제 단서를 가린 경우와 옆에 물체만 있는 경우를 대조합니다.</td></tr><tr><td>실패325/996은 콘과 남은 목재·초록 테두리가 함께 보입니다.</td><td>대상 재질이 남아 있는 것과 기대한 구조 단서가 보이는 것을 혼동할 가능성이 있습니다. 996의 부품 대응은 그림만으로 확정되지 않습니다.</td><td>재질·물체량을 맞춘 cover/sham 대조, 기대 부품과 대상의 연결을 확인합니다.</td></tr><tr><td>퇴행256/1021은 콘이 중심 또는 crop 대부분인데 C12가 보임입니다.</td><td>경계 문제 하나로 설명할 수 없습니다. 보임 쪽 점수 이동·국소 특징 사용·학습 범위가 추가 가설입니다.</td><td>가렸을 때 점수만 상승하는지보다, 가린 사례가 실제 E 판정 문턱을 넘는지도 검증합니다.</td></tr><tr><td>야간696은 어둡고 흐리며 원본 crop 한 변이16.37px입니다.</td><td>C12가 기존 인간 D에 맞췄지만 시각 정보가 적습니다. 한 성공이 저조도 전체 해결을 뜻하지 않습니다.</td><td>공개 데이터에서 조도·흐림·원본 크기를 나눠 성능과 보류율을 확인합니다.</td></tr></table></div><p class="warn">9사례는 성공·실패 조건별 진단 표본입니다. 공통 모양을 전체 빈도나 유일한 원인으로 단정하지 않았습니다. 현재223평가는 과거 개발에 노출된 자료이며 새 blind holdout이 아닙니다.</p></section>
<section><h2>콘이 충분히 보이는데도 놓친 사례</h2><a href="figures/CASE_08.png"><img src="figures/CASE_08.png" alt="콘 중심의 가림을 C11은 숨김, C12는 보임으로 응답한 실제 사례1021"></a><p>이 사례는 v61·기존 실사 모델·C11은 hidden, C12만 D입니다. 새 모델이 항상 좋아진 것은 아닙니다. 같은 원본·context·L의 다른 q인259/256도 C12 판정이 갈리므로 전체 배경 하나만으로 원인을 확정할 수 없습니다.</p></section>
<section><h2>이미 공개 데이터에서 확인한 보완점</h2><p>현재 평가를 보기 전 공개 source 진단에서 <b>목재를 가림 물체로 쓰는 TRAIN 역할 누락</b>과 <b>cover-sham 차이는 충분한데 가린 사례 자체는 D인 문제</b>가 확인됐습니다. 이를 검증할 동일 재질 대조 자료를 준비했습니다.448개 시도 중62쌍이 기하·픽셀 조건을 통과했고, 실패386쌍은 U로 남겼습니다. 원 q를 옮기지 않았습니다.</p><a href="../query_visibility_same_material_foreground_source_20261023_v116/fixed_source_visual_review_v1/cards/ATTEMPT_049_ACCEPTED.png"><img src="../query_visibility_same_material_foreground_source_20261023_v116/fixed_source_visual_review_v1/cards/ATTEMPT_049_ACCEPTED.png" alt="공개 source의 같은 q에서 목재 단서를 가린 cover와 보존한 sham"></a><p>위 대조는 원래 보이던 목재 단서를 같은 재질로 가리거나 옆으로 옮긴 통제된 보조 학습 후보입니다. 붙여넣기 경계·밝기·그림자 차이가 있고62쌍 모두 PALLOX라 자연 가림의 정답이나 실제 성능 개선으로 해석할 수 없습니다. 새 head에는 아직 학습하지 않았습니다.</p><p>공개 SAM 실험에서는 어노테이션의 팔레트와 초기 추론의 팔레트가 다른 사례도 확인했습니다. 대상이 연결되지 않은 mask를 자동 가림 정답으로 사용하지 않는 것이 필요합니다.</p></section>
<section><h2>목표의 범위</h2><p class="note"><b>초기 q를 고정한 채 해당 대상의 기대한 국소 단서가 직접 보이는지·외부 물체에 가려졌는지를 판단합니다.</b> 자기 가림은 말씀하신 초기 기하 경로로 먼저 다룹니다. 가시성 응답을 좌표 정답 검증으로 해석하지 않습니다.</p><p>이 범위에서 성공 사례가 있으므로 목표를 포기할 근거는 없습니다. 모든 흐린 작은 crop에서100% 답을 내는 가능성은 입증되지 않았습니다. 현재 필요한 것은 목표를 키우는 일이 아니라 대상·부품·라벨을 일치시키고, 공개 source에서 D와 E의 절대 판정 및 보류율을 함께 검증하는 일입니다. 이번 평가 그림으로 모델·문턱·checkpoint를 고르지 않습니다.</p><p><a href="ROOT_READOUT_KO.html">전체9사례와 저장 지표</a> · <a href="ROOT_ACTUAL_VISUAL_INTERPRETATION.json">root의 실제 관찰·해석 기록</a></p></section></main></html>'''
(V/'ROOT_INTERPRETATION_KO.html').write_text(html)

summary=json.loads((S/'corpus_build_v1/SUMMARY.json').read_text())
audit=json.loads((S/'independent_saved_corpus_review/REVIEW.json').read_text())
assert summary['accepted_pairs']==62 and summary['failures_U']==386 and audit['PASS']
source_record={
 'schema':'V116_ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION', 'completed':True,'created_UTC':stamp,
 'actual_construction_session':52786,'actual_construction_terminal_exit_code':0,'fixed_visual_session':26730,'fixed_visual_terminal_exit_code':0,
 'corpus_summary':binding(S/'corpus_build_v1/SUMMARY.json'),'corpus_seal':binding(S/'corpus_build_v1/CORPUS_OUTPUT_SEAL.json'),
 'independent_saved_replay':binding(S/'independent_saved_corpus_review/REVIEW.json'),
 'independent_actual_visual_review':binding(S/'independent_source_visual_review/OBSERVATIONS_AND_RECOMMENDATION.json'),
 'root_actually_viewed':[binding(S/'fixed_source_visual_review_v1/CONTACT_ALL_FIXED_CASES.png')]+[binding(S/f'fixed_source_visual_review_v1/cards/ATTEMPT_{i:03d}_ACCEPTED.png') for i in [49,53,54]],
 'counts':{'bases':28,'donors':16,'attempts':448,'accepted_pairs':62,'failed_U_pairs':386,'occurrences':924,'D':90,'E':62,'U':772},
 'root_visual_observations_KO':['49/53/54는 같은 PALLOX base의 목재 단서가 cover에서 가려지고 sham에서 보존되는 대조다. 독립 대상3개는 아니다.','실패0/1/2는 LOCO 원본 그대로이며 U 유지가 맞다.','공개 RGB와 실제 alpha의 통제된 편집이나 이질적 밝기·떠 있는 패치·그림자 부재·일부 얇은 꼬리와 경계가 보인다. 자연 가림 gold가 아니다.'],
 'conditional_AUX_admission':{'admitted':True,'scope':'Only original28D and accepted62 coverE/shamD under frozen PUBLIC TRAIN query/cue protocol; all772U excluded from semantic D/E loss. No human/native gold promotion. No support shrink, donor replacement, query shift or new relabeling.','new_fit_authorized_by_this_record':False,'training_has_started':False,'natural_transfer_established':False,'all62_PALLOX_LOCO0':True,'new_role_not_new_independent_RGB_holdout':True},
 'public_source_rationale_predates_V117_current_visual_diagnosis':True,'new_features_model_forward_or_fit':0,'current223_or_V112_used':False,'PnP':0,
 'next':'Prospective source-only head/feature design and validation required before any new fit; no training during user-requested visual review.'
}
save(S/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json',source_record)
print(json.dumps({'root_interpretation':binding(V/'ROOT_ACTUAL_VISUAL_INTERPRETATION.json'),'report':binding(V/'ROOT_INTERPRETATION_KO.html'),'source_execution':binding(S/'ROOT_ACTUAL_EXECUTION_AND_FIXED_VISUAL_DECISION.json')},ensure_ascii=False))
