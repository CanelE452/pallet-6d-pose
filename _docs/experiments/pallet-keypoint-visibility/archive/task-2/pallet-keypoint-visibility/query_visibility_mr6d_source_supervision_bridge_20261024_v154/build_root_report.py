"""Saved source metadata -> V154 report. No model/data feature/private reads."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import os

V = Path(__file__).resolve().parent
P = V.parent
RUN = V / 'runs/C35_C36_SEPARATE_QUERY_FIXED2000'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path)}


def write(name, obj):
    path = V / name
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    return bind(path)


def href(path, text):
    rel = os.path.relpath(path, V)
    return f'<a href="{escape(rel, quote=True)}">{escape(text)}</a>'


def figure(rel, caption):
    return f'<figure><a href="{rel}"><img src="{rel}" alt="{escape(caption, quote=True)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


summary_path = V / 'source_result_summary_agent/SOURCE_RESULTS_SUMMARY_KO.json'
ranking_path = V / 'independent_saved_fit_review/RANKING_DIAGNOSTIC.json'
cause_path = V / 'saved_output_cause_diagnosis/CAUSE_ASSESSMENT_KO.json'
summary = read(summary_path)
ranking = read(ranking_path)
cause = read(cause_path)
seal = read(V / 'ROOT_SUPERVISION_SEAL.json')
control = read(RUN / 'CONTROL_COMPLETE.json')
execution = read(V / 'ROOT_ACTUAL_SOURCE_FIT_EXECUTION_RECEIPT.json')
parent = read(P / 'LATEST_STATUS_KO.json')
assert control['completed'] and control['PASS']
assert execution['root_observed_terminal_exit_code'] == 0
assert execution['actual_source_optimizer_updates'] == 4000
assert summary['selected_model'] is None and not summary['adopted']
assert seal['original_q_L_flags_source_ids_and_splits_unchanged']
assert 'V149' in parent['latest_current_evaluation']['version']
val = summary['MR']['SOURCE_VALIDATION']
models = val['models']
expected = {'C32_initial': [[86, 7, 9], [45, 7, 0]],
            'C35': [[89, 4, 9], [39, 13, 0]],
            'C36': [[64, 29, 9], [21, 31, 0]]}
for name, cm in expected.items():
    assert models[name]['primary_all']['CM_rows_D_E_columns_D_E_U'] == cm
assert models['C36']['secondary_entity_gated_all']['CM_rows_D_E_columns_D_E_U'] == [[64,18,20],[21,15,16]]
assert summary['synthetic_point_quality_held']['C36']['combined_held_CM_rows_far_near_columns_far_near_U'] == [[63,3,0],[72,1084,0]]

evidence_files = [
    'ROOT_POLICY_BEFORE_SOURCE_BRIDGE_REVIEW.json',
    'actual_query_association_agent/ASSOCIATION_REVIEW_COMPLETE.json',
    'development_native_review_scene0_2_agent/SOURCE_ROLE_REVIEW_COMPLETE.json',
    'development_native_review_scene5_6_agent/SEMANTIC_REVIEW_COMPLETE.json',
    'ROOT_DEVELOPMENT_WEAK_ROLE_ADMISSION.json',
    'ROOT_SOURCE_VALIDATION_WEAK_ROLE_ADJUDICATION.json',
    'ROOT_SUPERVISION_SEAL.json', 'ROOT_SUPERVISION_INPUT_BINDING_APPENDIX.json',
    'independent_supervision_review/REVIEW.json',
    'MATCHED_EXPERIMENT_DESIGN_BEFORE_LABEL_COUNTS.json',
    'head_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json',
    'independent_prepared_fit_review/REVIEW.json',
    'development_native_review_scene5_6_agent/INDEPENDENT_MATCHED_TRAINER_CODE_REVIEW.json',
    'ROOT_APPROVED_SOURCE_HEAD_FIT_ENTRY.json',
    'ROOT_ACTUAL_SOURCE_FIT_EXECUTION_RECEIPT.json',
    'runs/C35_C36_SEPARATE_QUERY_FIXED2000/CONTROL_COMPLETE.json',
    'independent_saved_fit_review/REVIEW.json',
    'independent_saved_fit_review/STATE_AND_CONTROL_SCOPE_APPENDIX.json',
    'independent_saved_fit_review/RANKING_DIAGNOSTIC.json',
    'source_result_summary_agent/SOURCE_RESULTS_SUMMARY_KO.json',
    'source_result_summary_agent/FINAL_SUMMARY_QA.json',
    'saved_output_cause_diagnosis/CAUSE_ASSESSMENT_COMPLETE.json',
    'saved_output_cause_diagnosis/CAUSE_ASSESSMENT_KO.json',
    'ALIGNMENT_FIGURE_VISUAL_QA.json',
    'ROOT_NEXT_SOURCE_COMMITMENT_BEFORE_FUTURE_PRIVATE_READS.json',
]
evidence = {n: bind(V/n) for n in evidence_files}
rank_compact = {n: {k: x[k] for k in (
    'ROC_AUC_E_positive_tie_half', 'raw0_original_point_unchanged',
    'E_recall_capacity_at_D_recall_floor')} for n,x in ranking['results'].items()}
now = datetime.now(timezone.utc).isoformat()
results = {
    'schema': 'V154_SOURCE_WEAK_QUERY_SUPERVISION_MATCHED_HEAD_RESULTS',
    'completed': True, 'UTC': now, 'stage_complete': True,
    'goal_status': 'active_unsolved', 'fully_solved': False,
    'selected_model': None, 'adopted': False,
    'execution_QA_PASS_is_not_scientific_qualification': True,
    'scientific_source_qualification_PASS': False,
    'latest_work_KO': '公開230장 원래 query와 개체/역할 감독을 연결하고 작은3출력head 두 개를 각각2000회 실제 학습했다. 두 모델 모두 source 기준 미달이며 채택하지 않았다.',
    'actual_execution': execution,
    'source_data': {
        'MR_frames': 230, 'MR_development_frames': 185, 'MR_held_frames': 45,
        'original_queries': 1840, 'GT_role_references': 2424,
        'all_source_rows_preserved': 28976, 'synthetic_rows': 26624,
        'Pallox_rows': 512, 'MR_rows': 1840,
        'GT_and_query_supervision_are_separate': True,
        'source_human_gold': False, 'exact_real_pixel_quality_gold': False,
        'counts': {k: seal['counts'][k] for k in ('ALL','SOURCE_DEVELOPMENT','SOURCE_VALIDATION')},
        'association_frame_states': {'supported':129,'wrong_entity':49,'ambiguous':49,'no_query':3},
        'partial_clear_rear_rows_retained':44,
        'wrong_entity_never_assigned_E':True,
    },
    'design': {**summary['design'], 'parameters_each':224643,
               'outputs':['external_occlusion_E_vs_D','target_entity_supported','coarse_point_quality'],
               'shared_hidden_dimensions':64,'new_encoder_or_R0_calls':0,
               'point_quality_training':'Exact synthetic <=3px / >20px only, quality unknown never removes visibility rows',
               'coordinates_predicted_or_corrected':False},
    'MR': summary['MR'], 'MR_paired_primary':summary['MR_paired_primary'],
    'source_ranking':rank_compact,
    'ranking_operating_point_or_threshold_selected':False,
    'synthetic_point_quality_held':summary['synthetic_point_quality_held'],
    'synthetic_quality_far_precision':63/(63+72),
    'entity_held_available':{'correct_entity_total':244,'correct_entity_kept':187,
                             'correct_entity_false_rejected':57,
                             'wrong_entity_rejected':8,'wrong_entity_total':8,
                             'wrong_entity_independent_frames':1,
                             'wrong_entity_precision':8/(8+57)},
    'native35':summary['native35'],'Pallox_HALL':summary['Pallox_HALL'],
    'cause_findings_KO':cause['key_findings_KO'],
    'interpretation_KO':[
        '같은 공개 held의 C32→C35 AUC .610→.785, 보임86→89/93, 외부가림7→13/52는 유용한 진전이다. common native/Pallox/MR 손실이 함께 계속되어 MR 데이터만의 인과효과는 아니다.',
        'C35→C36은 외부가림18개 회복과 보임25개 손상이 함께 발생했다. AUC도 .785→.702이며 두 손실 추가의 공동 효과이지 각 보조손실의 독립 효과는 아니다.',
        '가림 점수를 정렬하는 능력은 일부 개선됐으나 보임90%를 유지할 때 최대 외부가림 recall도 C35 16/52(30.8%)이다. 단순 문턱 조정으로 현재 목표를 해결한다는 근거는 없다. 이 수는 사후 진단이며 문턱을 선택하지 않았다.',
        '대상 entity gate는 맞게 가림으로 판단한31개 중16개를 보류시켜15개만 남긴다. real quality 정답은 없으며 합성 quality 결과를 실사 성과로 바꾸지 않는다.',
        '실제 crop의 foreign bin/crate와 target role visibility는 서로 다른 질문이다. D 손상 사례의 명목 좌표 편차가144/193px인 경우 이를 정확한 위치의 순수 가림 전이 실패라고 부를 수 없다.',
        '세 출력의 강한 상관과 shared hidden은 확인됐으나 hidden rank 붕괴나 특정 보조손실의 인과 원인은 아직 증명하지 않았다.',
        '성공 가능성을 뒷받침하는 신호는 있으나 배치 가능한 모델은 없다. Euro 목재 weak source를 정사각형 실사 및 사용자223 성능의 증명으로 확대하지 않는다.',
    ],
    'limits_KO':[
        '공개 역할 감독은 assistant 시각 검토 기반 약한 라벨이다. 제공 pose는 cm 수준 오차 가능성이 있어 실사3px 정답이 아니다.',
        'held는 장면을 분리했지만 동일 물리 물체 재사용과 장면내 반복 촬영이 가능하다. 52개 가림 query가52개 독립 물체는 아니다.',
        'held entity 음성8개는 한 프레임의8개 query이다. 범용 wrong-entity 성능을 증명하지 않는다.',
        'known D/E154 외206행(S9/OOF77/U120)과 원입력 불가D9를 모두 보존했다. 입력가능145행 지표와 전체154행 지표를 구분한다.',
        'native35는 모든 기준 통과가 아니며 기존 AUX29를 새로 재검증했다는 주장도 아니다. 목재/정사각형과 native8 전체 범위는 유지한다.',
        '사용자223은 과거 반복 평가로 이미 개발에 노출된 자료이며 blind test가 아니다. 이번 단계에서는 private body를 새로 읽지 않았고 학습·선택에도 쓰지 않았다.',
    ],
    'private_current_evaluation_unchanged_metadata':parent['latest_current_evaluation'],
    'new_private_current223_reads_or_inference':0,
    'private_used_for_training_threshold_checkpoint_arm_selection':False,
    'PnP_q_L_xy_changes':0, 'live_processes':[],
    'next_source_commitment':evidence['ROOT_NEXT_SOURCE_COMMITMENT_BEFORE_FUTURE_PRIVATE_READS.json'],
    'evidence':evidence,
}
result_binding = write('ROOT_RESULTS_KO.json', results)

rows=''
for name, label in [('C32_initial','학습 전 C32'),('C35','C35 가림 판단 학습'),('C36','C36 대상·좌표 품질 보조학습 추가')]:
    m=models[name]['primary_input_valid']; full=models[name]['primary_all']
    rankname={'C32_initial':'C32_INITIAL','C35':'C35_FINAL','C36':'C36_FINAL'}[name]
    auc=rank_compact[rankname]['ROC_AUC_E_positive_tie_half']
    d=expected[name][0][0]; e=expected[name][1][1]
    rows+=f'<tr><th>{label}</th><td>{d}/93 · {m["D_recall"]:.1%}</td><td>{e}/52 · {m["E_recall"]:.1%}</td><td>{m["E_precision"]:.1%}</td><td>{auc:.3f}</td><td>{full["U_rate_known"]:.1%} (9/154)</td></tr>'
evidence_links=''.join(f'<li>{href(V/n,n)}</li>' for n in evidence_files)
limits=''.join(f'<li>{escape(x)}</li>' for x in results['limits_KO'])
html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V154 — 공개 실사 추가 학습의 진전과 남은 충돌</title>
<style>body{{font-family:"Noto Sans CJK KR","Noto Sans KR",sans-serif;max-width:1160px;margin:34px auto;padding:0 24px;background:#f5f7fa;color:#182235;line-height:1.75}}h1{{font-size:30px;line-height:1.4}}h2{{margin-top:36px;font-size:23px}}p,li{{max-width:1040px}}.card{{background:white;border:1px solid #dbe1ea;border-radius:12px;padding:22px;margin:20px 0}}.lead{{font-size:20px}}.flag{{color:#873519;font-weight:700}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{border:1px solid #d4dce8;padding:10px;text-align:left}}th{{background:#edf2f8}}figure{{margin:28px 0}}img{{display:block;max-width:100%;height:auto;border:1px solid #cfd7e4;background:white}}figcaption{{font-size:14px;color:#45546c;margin:8px 0}}a{{color:#135da0;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}.small{{font-size:14px;color:#45546c}}@media(max-width:760px){{body{{padding:0 12px}}table{{font-size:13px}}td,th{{padding:6px}}}}</style>
<h1>진전은 있다. 아직 보임 보존과 외부 가림 검출을 함께 해결하지 못했다.</h1>
<p class="lead">공개 팔레트 실사와 합성으로 작은 head 두 개를 실제 학습했다. 같은 공개 검증 장면의 가림/보임 점수 구분은 <b>AUC 0.610→0.785</b>로 좋아졌다. 보조 판단까지 함께 학습하면 가림 검출은 늘지만 보이는 점 손상이 커졌다.</p>
<p class="flag">V154 단계 완료 · 전체 목표 미해결 · 채택 모델 없음 · 사용자223 신규 평가 없음 · PnP 및 좌표 수정 없음</p>
<p class="small">{escape(now)} · JSON {href(V/'ROOT_RESULTS_KO.json','전체 수치·근거')} · {href(V/'ROOT_ACTUAL_SOURCE_FIT_EXECUTION_RECEIPT.json','실제 학습 종료 기록')}</p>
<div class="card"><b>이번에 완료한 일</b><p>MR6D 공개230장(학습185·검증45)의 원래 추론 query1840개를 그대로 두고, 대상 개체 일치와 의미상 코너 역할을 따로 검토했다. 다른 물체를 잡은49프레임은 외부 가림 정답으로 만들지 않았다. 같은 물체여도 역할 대응이 불확실하면 U로 남겼다. 합성26624행·기존 공개 Pallox512행·MR1840행을 보존한 채 C35/C36을 각각2000 update 학습했다.</p><p>입력 encoder와 초기 추론은 기존 저장 결과를 재사용했다. 새로 학습한 것은224643개 파라미터의 head이며, 좌표를 예측하거나 옮기지 않는다. 가림, 대상 일치, 거친 좌표 품질의 세 출력은64차원 hidden을 공유한다. C35는 보조 손실0, C36은 두 보조 손실을 함께 추가한 대조다.</p></div>
<h2>같은 공개 검증에서 무엇이 좋아졌나</h2>
<p>다음 recall은 입력 가능한 보임D93개·외부 가림E52개에서 계산했다. 표의 보류율은 입력 불가D9개까지 포함한 전체 알려진D/E154개 기준이다. 전체45장360행은 D102/E52/S9/화면밖77/U120으로 유지했다. S·화면밖·미확정206행을 가림 오답으로 재분류하지 않았다.</p>
<table><thead><tr><th>고정 모델</th><th>보이는 점 보존 recall</th><th>외부 가림 recall</th><th>외부 가림 precision</th><th>AUC</th><th>전체 알려진 D/E 보류율</th></tr></thead><tbody>{rows}</tbody></table>
<p>C32→C35에서는 보임 정답이3개, 가림 정답이6개 늘었다. 공통 합성·Pallox·MR 손실을 함께 계속한 결과로, MR 데이터만의 효과를 분리한 실험은 아니다. C35→C36은 두 보조 손실 추가의 공동 효과를 비교할 수 있으며, <b>가림18개 회복과 보임25개 손상</b>이 함께 생겼다.</p>
{figure('source_result_summary_agent/MR_PRE_POST_COMPARISON_KO.png','같은 공개 학습·검증 행의 전후 결과. 아래 혼동행렬은 입력 불가D9를 U로 보존한다. 이 그림은 사용자223 평가가 아니다.')}
<h2>문턱 하나를 바꾸면 해결되는가</h2>
<p>현재 점수 순서로 보임 recall90% 이상을 유지할 때 외부 가림 recall의 사후 최대치는 C32 12/52(23.1%), C35 16/52(30.8%), C36 15/52(28.8%)였다. 따라서 현재 source에서 보임90%·가림80%를 동시에 만족시키는 단일 문턱은 없다. 문턱 수치는 내보내거나 선택하지 않았으며, 모든 주 결과는 사전 고정 raw logit0 그대로다. {href(ranking_path,'점수 순서·가능 범위 진단')}</p>
<h2>보조 판단에서 생긴 충돌</h2>
<p>C36은 다른 개체인 검증8개 query를 모두 거부했지만, 이8개는 <b>단 한 프레임</b>에서 나온 점들이다. 맞는 대상244개 중57개(23.4%)도 잘못 거부했다. 이 점수로 가림 판단을 보류시키면 맞게 검출한 가림31개 중16개가 빠져15개만 남는다. 전체 알려진D/E 보류는9/154→36/154로 늘어난다.</p>
<p>합성의 좌표 품질 검증에서는 큰 오차63/66, 작은 오차1084/1156을 맞혔다. 하지만 큰 오차로 분류한 것의 precision은63/135=46.7%이며, 나머지4922행은 품질 정답 미확정이다. 실제 공개 실사에는 정확한3px 좌표 품질 정답이 없다. 이 결과를 실사 위치 판별 성공으로 해석하지 않는다.</p>
<p>source native35개 기준 통과 수는 C35 22/35, C36 17/35로 둘 다 미달이다. 별도 Pallox HALL의 입력가능 보임도32/33→29/33으로 줄었다. 최종 채택은 없다.</p>
{figure('source_result_summary_agent/AUXILIARY_AND_SOURCE_GATES_KO.png','대상 일치·합성 좌표 품질·기존 source 기준 결과. entity 검증 음성8개는 한 프레임이며 범용 검증으로 볼 수 없다.')}
<h2>실제 crop에서 보이는 공통 문제</h2>
<p>아래 첫 두 줄은 target 코너가 외부 물체에 가려졌다는 약한E 라벨을 가지며, crop에는 상자·금속통이 크게 보인다. 가림 출력은 맞아도 대상 일치 출력이 이들을 다른 개체로 거부한다. 학습E49개는 모두 반복 추출되었고 거부된14개도 각각81–117회 노출됐다. 단순 누락은 아니며, 가림E가 entity 양성638개 중49개인 하위집단 불균형은 남아 있다.</p>
<p>세 번째 줄의 보임D 손상 두 사례는 원래 query가 기준 역할에서 명목144/193px 벗어나 있다. 역할의 보임과 query 중심 픽셀의 대상 존재가 서로 다를 수 있음을 보여준다. 제공 pose 자체가 부정확할 수 있으므로 이 거리로 실사 정답을 만들거나3px 평가를 대체하지 않는다.</p>
{figure('saved_output_cause_diagnosis/FIXED8_SAVED_CROP_DIAGNOSIS.png','고정8개 원래96×96 crop과 중앙32×32. 점수는 저장 출력이며 명목 거리는 정확한 실사 pixel gold가 아니다.')}
<p>전체 개체를 잘못 잡은 사례와, 같은 팔레트지만 native 역할 대응이 불확실한 사례도 따로 보존했다. GT와 가장 가까운 점으로 바꾸거나 순서를 재배열해 성능을 높이지 않았다.</p>
{figure('ROLE_AND_ENTITY_ALIGNMENT_FAILURES_KO.png','개체 오류와 같은 개체 내 역할 대응 문제. 원래 좌표·crop·역할 번호를 유지한 진단.')}
<p>가림·대상·품질 출력은 강하게 결합되어 있고 hidden도 공유한다. 두 보조 손실을 함께 더했을 때의 손상은 확인했지만, 특정 손실 하나가 원인인지 hidden 차원이 붕괴한 것인지는 아직 검증하지 않았다. {href(V/'saved_output_cause_diagnosis/CAUSE_ASSESSMENT_KO.html','저장 출력 원인 진단 상세')}</p>
<h2>성공 가능성과 다음 검증</h2>
<p><b>시도할 근거는 있다.</b> 같은 공개 held에서 추가 학습 후 점수 구분과 정답 수가 함께 개선됐다. 다만 아직 실사용 가능하다는 증거는 없고 성공 확률을 수치로 말할 근거도 없다. 공개 Euro 목재와 assistant 약한 라벨의 개선을 사용자 목재·정사각형 전체 성능으로 확대하지 않는다.</p>
<p>다음에는 저장 source 특징과 동결 head로 hidden·출력 결합을 확인하고, 대상 일치·좌표 품질의 내부 경로를 가림 판단 경로와 분리한 대조를 진행한다. 보조 손실의 gradient와 공통 clipping이 가림 경로를 바꾸지 않게 하는 것이 검증할 가설이다. 이 설계는 공개 source 결과에 근거해 미리 고정했고 아직 구현·학습하지 않았다. {href(V/'ROOT_NEXT_SOURCE_COMMITMENT_BEFORE_FUTURE_PRIVATE_READS.json','다음 source 검증의 사전 약속')}</p>
<div class="card"><b>현재223 평가 상태</b><p>이번 C35/C36을 사용자223에 새로 평가하지 않았다. 최신 실제223 평가 포인터는 기존 V149 그대로다. 이번 공개 수치와 과거223 수치를 다른 분모인 채 직접 우열 비교하지 않는다. 사용자의223으로 학습·문턱·checkpoint·모델 선택은 하지 않는다. {href(Path(parent['latest_current_evaluation']['report']),'기존 V149 평가 보고서')}</p></div>
<h2>해석의 한계와 근거 파일</h2><ul>{limits}</ul>
<details><summary>라벨·학습·독립 검산 근거 목록</summary><ul>{evidence_links}</ul></details>
<p class="small">이미지는 실제 PNG를 확인했다. HTML은 로컬 링크와 구조를 검산했으며 브라우저 렌더링을 확인했다고 주장하지 않는다. 학습 종료의 PASS는 실행 무결성 검산이며 성능 기준 통과와 다르다.</p>
</html>'''
report_path=V/'ROOT_READOUT_KO.html'
assert not report_path.exists()
report_path.write_text(html)


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        for k,value in attrs:
            if k in ('href','src') and value:
                self.links.append(value)


parser=LocalLinks(); parser.feed(html)
missing=[]
for link in parser.links:
    if '://' not in link and not (V/link.split('#')[0]).exists():
        missing.append(link)
assert not missing, missing
figs=['source_result_summary_agent/MR_PRE_POST_COMPARISON_KO.png',
      'source_result_summary_agent/AUXILIARY_AND_SOURCE_GATES_KO.png',
      'saved_output_cause_diagnosis/FIXED8_SAVED_CROP_DIAGNOSIS.png',
      'ROLE_AND_ENTITY_ALIGNMENT_FAILURES_KO.png']
qa=write('ROOT_REPORT_VISUAL_QA.json',{
    'completed':True,'PASS':True,'UTC':now,
    'report':bind(report_path),'results':result_binding,
    'local_links_checked':len(parser.links),'missing_local_links':[],
    'figures':[bind(V/n) for n in figs],
    'root_actual_image_views':[
        'All four final figures viewed in root tool outputs during V154 work.',
        'Main MR plot and fixed8 crop board re-viewed after context continuation before this report.',
    ],
    'visual_observations_KO':'한글·표·분모와 고정8crop의 점수 표기 확인. 보임 손상과 가림 회복을 함께 표시. 명목 거리의 real pixel gold 오해를 경고.',
    'browser_render_checked':False,'HTML_QA_scope':'Static local-link existence/structure and image QA only.',
    'numeric_checks':'Saved source aggregate exact CM checks, held original154/145 distinction, fixed quality CM, root actual exit0.',
    'wrong_entity_held8_queries_one_frame_explicit':True,
    'private_current_body_reads_for_report':0,'model_feature_or_fit_calls_for_report':0,
    'selected_model':None,'goal_status':'active_unsolved',
})
phase=write('ROOT_PHASE_RESULT.json',{
    'schema':'V154_COMPLETE_SOURCE_SUPERVISION_AND_MATCHED_HEAD_FITS_NOT_SOLVED',
    'completed':True,'stage_complete':True,'UTC':now,
    'goal_status':'active_unsolved','fully_solved':False,
    'classification':'PROGRESS','selected_model':None,'adopted':False,
    'results':result_binding,'report':bind(report_path),'report_QA':qa,
    'supervision_seal':evidence['ROOT_SUPERVISION_SEAL.json'],
    'actual_fit_control':evidence['runs/C35_C36_SEPARATE_QUERY_FIXED2000/CONTROL_COMPLETE.json'],
    'actual_execution':evidence['ROOT_ACTUAL_SOURCE_FIT_EXECUTION_RECEIPT.json'],
    'actual_source_optimizer_updates':4000,'updates_each':2000,
    'scientific_source_qualification_PASS':False,
    'C35_native35':22,'C36_native35':17,'native35_total':35,
    'source_weak_labels_frozen_before_fit':True,
    'new_encoder_R0_private_PnP_or_xy_calls':0,
    'latest_private_evaluation_unchanged_version':parent['latest_current_evaluation']['version'],
    'next_source_commitment':evidence['ROOT_NEXT_SOURCE_COMMITMENT_BEFORE_FUTURE_PRIVATE_READS.json'],
    'live_processes':[],
})
print(json.dumps({'completed':True,'results':result_binding,'report':bind(report_path),
                  'QA':qa,'phase':phase,'checked_local_links':len(parser.links)},ensure_ascii=False))
