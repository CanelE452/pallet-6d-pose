"""Saved-evidence status report only. No models, training, or private inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

V = Path(__file__).resolve().parent
P = V.parent

def read(rel):
    return json.loads((V / rel).read_text())

def binding(path):
    p = Path(path)
    if not p.is_absolute():
        p = V / p
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def save(name, data):
    with (V / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')

benchmark_path = P / 'query_visibility_public_native_frozen_current_benchmark_20261024_v149/ROOT_RESULTS_KO.json'
benchmark = json.loads(benchmark_path.read_text())
intake = read('full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json')
r0 = read('actual_full230_initializer_v1/RESULT.json')
feat = read('full230_native_DINO_v1/FEATURE_COMPLETE.json')
quality = read('head_preparation/saved_synthetic_point_quality_v1/RESULTS.json')
prototype = read('head_preparation/SEPARATED_HEAD_PROTOTYPE_CHECKS.json')
weak = read('native_reference_preparation/source_validation_visual_review_agent/VALIDATION_STRUCTURAL_REVIEW_COMPLETE.json')
evidence = read('original_query_evidence_full230/EVIDENCE_RENDER_COMPLETE.json')
assert intake['completed'] and intake['frames'] == 230
assert r0['completed'] and r0['native_slots'] == 1840
assert feat['completed'] and feat['rows'] == 1840
assert prototype['actual_training_updates'] == 0
assert weak['validation_roles'] == 360 and not weak['train_ready'] and not weak['human_gold']
assert benchmark['selected_model'] is None
assert quality['total_rows'] == 26624

font = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False})
rows = benchmark['aggregate_primary']
fig, ax = plt.subplots(figsize=(12, 6.3), dpi=150)
y = list(range(len(rows)))
ax.barh([v-.18 for v in y], [r['D_recall']*100 for r in rows], height=.32,
        color='#168575', label='보이는 점 보존율 D recall (1147개)')
ax.barh([v+.18 for v in y], [r['E_recall']*100 for r in rows], height=.32,
        color='#dd8e32', label='외부 가림 검출율 E recall (90개)')
for i, r in enumerate(rows):
    cm = r['CM_D_E_by_pred_D_E_U']
    for dy, n, den, recall in [(-.18, cm[0][0], 1147, r['D_recall']), (.18, cm[1][1], 90, r['E_recall'])]:
        ax.text(recall*100+1.0, i+dy, f'{n}/{den} ({recall*100:.1f}%)', va='center', fontsize=9)
ax.set_yticks(y, [r['model'] + (' *' if r['model']=='legacy' else '') for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 121)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel('동일 평가 대상에서의 recall (%) · 큰 값이 좋음')
ax.grid(axis='x', alpha=.18)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, fontsize=10)
fig.suptitle('성능의 현재 한계: 보임 보존과 외부 가림 검출을 함께 해결하지 못함', y=.99, fontsize=15)
fig.text(.07, .035, 'V149 저장 집계 재표시 · 새 230장 학습 결과 아님 · legacy는 평가 일부와 학습 중복, 보류율 14.3%\n'
         '목재·정사각형 합계 N=1237 (D=1147/E=90), self/OOF 제외 · 나머지 6모델의 동일 대상 보류율 0% · 채택 모델 없음', fontsize=9)
fig.tight_layout(rect=(0, .10, 1, .94))
fig.savefig(V / 'CURRENT_PROGRESS_AND_LIMITS_KO.png')
plt.close(fig)

actual_examples = [
    dict(frame_index=73, source_id='val/000001:0',
         observation_KO='원래8q가 팔레트 옆 검은 금속 받침대에 모여 있다. 8개 crop의 입력이 정상이어도 대상 개체가 맞다는 뜻은 아니다.',
         evidence='original_query_evidence_full230/frames/FRAME_073_val_000001_0.png'),
    dict(frame_index=95, source_id='val/000001:22',
         observation_KO='전면·하부 q와 후면·상부 q의 위치 품질이 서로 다르고 일부는 영상 밖이다. 바구니·목재상자·배경이 섞여 있다.',
         evidence='original_query_evidence_full230/frames/FRAME_095_val_000001_22.png'),
    dict(frame_index=109, source_id='val/000001:36',
         observation_KO='일부 전면 q는 팔레트 주변에 있지만 후면 crop은 바구니·통·배경까지 포함한다. 이 픽셀 종류만으로 해당 역할의 E를 정할 수 없다.',
         evidence='original_query_evidence_full230/frames/FRAME_109_val_000001_36.png'),
]
for e in actual_examples:
    e['visual'] = binding(e['evidence'])
    e['root_actually_viewed'] = True
    e['model_visibility_prediction_or_human_gold'] = False

root_hold = dict(completed=True, human_gold=False, train_ready=False,
    proposal_receipt=binding('native_reference_preparation/source_validation_visual_review_agent/VALIDATION_STRUCTURAL_REVIEW_COMPLETE.json'),
    original_proposals_preserved=True, proposal_counts=weak['counts'],
    root_actual_source_overlay_frames=[91,109,113], root_actual_query_crop_frames=[73,95,109],
    reason_KO='원본 역할 제안은 실제 R0의 선택 개체/역할과 아직 연결되지 않았다. 후방 D/S 제안은 명목 reference의 cm급 오차와 가림 경계 때문에 root 확정 대기다. 177은 GT 역할 D/E 제안 수이며 실제 query 학습 가능 개수가 아니다.',
    source_validation_training_allowed=False, labels_overwritten=False,
    real_coordinate_accuracy_labels_created=0)
save('ROOT_SOURCE_PROPOSAL_HOLD.json', root_hold)

bindings = {k: binding(p) for k,p in {
    'policy':'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json',
    'source_intake':'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json',
    'download':'full_source_v1/FULL_SOURCE_DOWNLOAD_COMPLETE.json',
    'failure_diagnosis':'full_intake_preparation/FIRST_DOWNLOAD_FAILURE_DIAGNOSIS.json',
    'explicit_exact4_recovery':'ROOT_EXPLICIT_REMAINING4_CONTINUATION_AFTER_ZERO_RESPONSE.json',
    'initializer':'actual_full230_initializer_v1/PREDICTION_SEAL.json',
    'initializer_independent_review':'independent_saved_pipeline_review/INITIALIZER_REVIEW.json',
    'features':'full230_native_DINO_v1/FEATURE_COMPLETE.json',
    'features_independent_review':'initializer_feature_preparation/FULL230_SAVED_FEATURE_REVIEW.json',
    'source_references':'native_reference_preparation/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json',
    'source_validation_proposals':'native_reference_preparation/source_validation_visual_review_agent/FINAL_REVIEW_DELIVERY.json',
    'root_proposal_hold':'ROOT_SOURCE_PROPOSAL_HOLD.json',
    'synthetic_quality_labels':'head_preparation/saved_synthetic_point_quality_v1/RESULTS.json',
    'small_head_prototype':'head_preparation/SEPARATED_HEAD_PROTOTYPE_CHECKS.json',
    'head_quality_independent_review':'initializer_feature_preparation/SEPARATED_HEAD_AND_QUALITY_REVIEW.json',
    'query_evidence':'original_query_evidence_full230/EVIDENCE_RENDER_COMPLETE.json',
    'independent_status_review':'ROOT_STATUS_INDEPENDENT_REVIEW.json',
    'semantic_metadata_appendix':'SEMANTIC_PROPOSAL_METADATA_APPENDIX.json',
}.items()}
bindings['latest_private_aggregate_preserved'] = binding(benchmark_path)
next_actions = [
    '전체230의 원래 query를 유지하면서 development185의 역할과 실제 선택 개체를 검토·연결한다. 다른 개체는 외부 가림 E로 바꾸지 않는다.',
    '고정 validation45의 weak 제안을 독립적으로 확정하고, 불명확한 후방·자기 가림·OOF는 별도로 유지한다. 3px 실측 정답으로 쓰지 않는다.',
    'source-only로 세 출력의 loss 가중치·sampling·고정 학습 길이·선택 규칙·보류 및 결합 규칙을 학습 전에 고정한다. 작은 head의 실제 학습과 기존 visibility 경로 대조를 진행한다.',
    '합성 native35와 목재/정사각형 전체 source 검증을 유지한다. 공개 Euro 목재만의 개선을 전체 해결로 취급하지 않는다.',
    'source 후보가 고정된 뒤 현재223은 최종 비교에만 쓴다. 현재223으로 학습·문턱·checkpoint·모델을 선택하지 않는다. PnP/xy 수정은 하지 않는다.',
]
results = dict(completed=True, created_UTC=datetime.now(timezone.utc).isoformat(),
    phase='V153_FULL_PUBLIC_INPUTS_FEATURES_AND_SEPARATE_HEAD_PREPARATION_COMPLETE',
    goal_status='active_unsolved', fully_solved=False, selected_model=None,
    outcome_KO='공개 실사230장·원래query1840개·동결 feature를 실제 확보하고, 작은 세 출력 head와 합성 위치품질 감독을 준비했다. 새 head 학습 또는 현재223 성능 개선은 아직 없다.',
    success_assessment_KO='실패 원인과 다음 검증 방법은 구체화됐다. 그러나 보임 보존과 외부 가림 검출을 동시에 만족하는 결과가 없고, 이번230장 기반 모델은 미학습이므로 성공 가능성이 높다고 수치화하거나 보장할 근거가 없다.',
    latest_current_evaluation='V149; stored aggregates only in this phase',
    aggregate_primary=rows, current_manual_near_D=benchmark['manual_near_D'],
    source_native35_C33_C34=dict(passed=22,total=35,qualified=False),
    current_legacy_limit_KO='legacy CAL46은 평가 일부와 학습이 겹치는 과거 참고이며 독립 일반화 비교로 인증하지 않는다.',
    current_evaluation_development_exposed_not_blind=True,
    source_input_counts=dict(frames=230,development=185,validation=45,GT_observations=303,
        native_reference_roles=2424,original_queries=1840,files=1067,original_bytes=178909419,
        continuous_input_valid=1538,nearest_input_valid=1535,finite_OOB=278,no_detection_rows=24,
        feature_constructed=1816,all_original_rows_retained=True),
    actual_executions=dict(R0_primary_frames=230,R0_explicit_predict_calls=232,R0_total_forward_hooks=233,
        DINO_primary_encoder_calls=230,DINO_repeat_calls=10,DINO_total_calls=240,
        DINO_views_per_call=17,visibility_classifier_forward_calls=0,new_head_training_updates=0,
        new_private_model_calls=0,private_data_input_or_tuning_calls=0,PnP_or_xy_correction_calls=0),
    new_source_GT_role_proposals=dict(validation_frames=45,validation_roles=360,counts=weak['counts'],
        human_gold=False,train_ready=False,GT_D_E_proposals_before_query_bridge=177,
        development_roles_U_UNREVIEWED=2064,actual_R0_query_bridge_complete=False,
        source_validation_not_fit=True,root_uncertain_rear_proposals_pending=True),
    new_separated_head=dict(parameters=prototype['parameters'],trained=False,
        outputs=['external_occlusion_E_vs_D','target_entity_supported_vs_wrong','coarse_point_quality'],
        predicts_or_corrects_coordinates=False,learned_deployment_fusion_fixed=False,
        input_encoder_frozen=True,unit_checks_are_not_model_performance=True),
    synthetic_quality_supervision=dict(all_rows=26624,**quality['all_counts'],
        error_positive_le3=True,error_negative_gt20=True,middle_quality_unknown=True,
        original_visibility_labels_and_loss_eligibility_not_filtered_by_quality=True,
        real_MR_exact_coordinate_accuracy_labels=0),
    actual_query_visual_observations=actual_examples,
    limits_KO=[
        '새 데이터 준비와 prototype 검산은 실제 모델 성능 개선의 증거가 아니다.',
        '현재 약점은 외부 가림의 누락과 보이는 점의 보존을 함께 개선하지 못한 것이다. 기존 모델보다 전체적으로 낫다고 채택한 모델은 없다.',
        '공개 source의 제공 pose/CAD는 cm급 명목 reference이며, 겹치는 mask는 배타적 owner/native 가림 gold가 아니다.',
        '실제 query가 다른 물체나 배경을 가리키는 경우와 대상 팔레트의 가림을 분리해야 한다. 정상 crop만으로 대상·좌표가 맞다는 뜻은 아니다.',
        '공개 scene 분리 검증이며 같은 물리 물체의 반복 가능성이 있다. 공식 MR6D blind test나 독립 평가 인증은 아니다.',
        'Euro 목재와 몇 가지 적재물의 공개 실사는 정사각형 실사·다양한 가림의 감독 공백을 모두 채우지 못한다.',
    ], next_required_actions=next_actions,bindings=bindings)
save('ROOT_RESULTS_KO.json', results)

table_rows = ''.join('<tr><td>'+html.escape(r['model'])+'</td>' +
    ''.join(f'<td>{r[k]*100:.1f}%</td>' for k in ['D_recall','E_recall','E_precision','U'])+'</tr>' for r in rows)
example_html = ''.join(f'<article><h3>{html.escape(e["source_id"])}</h3><p>{html.escape(e["observation_KO"])}</p><img src="{e["evidence"]}" alt="원래 q와 실제 crop 입력"><p class="muted">입력 증거. 가림 모델 예측 또는 사람 정답 아님.</p></article>' for e in actual_examples)
link_list = ''.join(f'<li><a href="{html.escape(str(Path(b["path"]).relative_to(V)) if Path(b["path"]).is_relative_to(V) else "../"+str(Path(b["path"]).relative_to(P)))}">{html.escape(k)}</a></li>' for k,b in bindings.items())
doc = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V153 현재 진전과 한계</title><style>
body{{font-family:system-ui,'NanumGothic',sans-serif;margin:36px auto;padding:0 22px;max-width:1120px;line-height:1.65;color:#233240;background:#f8fafc}}
h1,h2,h3{{line-height:1.3}}h2{{margin-top:38px}}.lead{{font-size:21px;font-weight:650}}.card{{padding:20px;background:#fff;border:1px solid #d9e2e9;border-radius:10px;margin:18px 0}}
img{{display:block;max-width:100%;height:auto}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #d9e2e9}}th:first-child,td:first-child{{text-align:left}}
.muted{{font-size:14px;color:#596b7b}}article{{background:white;border:1px solid #d9e2e9;padding:18px;margin:20px 0}}article img{{width:800px}}a{{color:#156d91}}li{{margin:7px 0}}
</style><h1>진전은 있다. 아직 성공 모델은 없다.</h1>
<p class="lead">보이는 점을 잘 보존하는 모델은 만들었지만, 외부 가림까지 함께 잘 잡는 균형은 아직 확보하지 못했다.</p>
<p>이번 V153은 공개 실사와 실제 초기 query에 대한 데이터·입력 준비를 완료한 단계다. 새230장 기반 head 학습 결과가 아니며, 최종 목표는 미해결이다.</p>
<div class="card"><b>완료:</b> 공개 실사230장, 원래 초기query1840개, 원본crop와 동결 DINO feature 전량, 작은 세 출력 head의 미학습 prototype, 합성 위치품질 라벨.<br>
<b>남음:</b> 실제 선택 개체와 역할 라벨 연결, development185 감독 검토, source 검증 후 실제 새 head 학습. 채택 모델 없음.</div>
<h2>현재 성능: V149의 같은 평가 대상</h2>
<p>D=보임1147개, E=외부가림90개, 합계1237개. 아래는 모두 기존 저장 결과다. 자기 가림·OOF는 이 이진 비교에서 별도이며, 현재223은 개발 과정에 노출된 평가 자료다.</p>
<img src="CURRENT_PROGRESS_AND_LIMITS_KO.png" alt="현재 recall tradeoff 전체7모델">
<table><thead><tr><th>모델</th><th>보임 recall</th><th>외부가림 recall</th><th>외부가림 precision</th><th>보류율</th></tr></thead><tbody>{table_rows}</tbody></table>
<p class="muted">legacy CAL46은 평가 일부와 학습 중복이 있어 과거 참고다. C33/C34는 source native35 검증22/35로 자격 미달이며, 현재 수치로 checkpoint·문턱·모델을 선택하지 않았다.</p>
<h2>이번에 실제 진행한 일</h2><ul>
<li>MR6D 공개 Euro 팔레트 원본230장을 확보하고 원본1067개 파일의 pinned checksum·크기를 확인했다. development185 / validation45를 입력 열기 전에 고정했다.</li>
<li>동일 초기 모델의 첫 argmax 선택과 native8 순서를 그대로 사용했다. 1840행 중 continuous 유효1538, finite OOB278, 미검출24를 모두 보존했다. nearest 유효1535는 별도 필드다.</li>
<li>원q 중심96crop와 동결 DINO feature1840행을 실제 추출하고 저장 pixel·좌표·ID·flags를 독립 검산했다. GT 중심으로 crop이나 좌표를 옮기지 않았다.</li>
<li>검증45장의360개 GT 역할을 실제 원본으로 검토했다. D116/E61/S9/OOF93/U81은 assistant 약한 제안이며 사람 정답이나 실제query의177개 학습 라벨이 아니다. 후방 D/S와 실제 개체 연결은 확정 대기다.</li>
<li>세 출력 head: 외부가림, 대상 개체 적합성, 좌표의 거친 품질을 분리했다. 좌표를 찾거나 수정하지 않는다. 224643개 parameter의 prototype이며 아직 미학습이다.</li>
<li>저장 합성26624행에서 원래 가림 라벨을 유지한 채, 위치품질만5245개 근접/260개 큰오차/21119개 미확정으로 나눴다. 공개 실사의 명목 좌표를3px 정답으로 만들지 않았다.</li></ul>
<h2>실제 초기 crop에서 확인한 문제</h2><p>아래3장은 root가 직접 확인했다. 입력이 유효한지, 대상 개체가 맞는지, 역할이 가려졌는지는 서로 다른 문제다. 3개 사례로 전체230장의 비율을 추정하지 않는다.</p>{example_html}
<p><a href="original_query_evidence_full230/index.html">전체230장의 원래query·crop 보기</a> · <a href="native_reference_preparation/source_reference_contact_v1/INDEX.html">전체230장의 명목 native reference 보기</a></p>
<h2>성공 가능성을 어떻게 판단하나</h2>
<p>보이는 점의 보존은 이미 높은 수준까지 가능했기 때문에 목표 전체를 무리하다고 볼 근거는 없다. 다만 외부 가림 누락이 여전히 많고, 새로운 개체·가림 종류와 정사각형 실사에 대한 전이는 검증되지 않았다. 지금은 성공 가능성이 높다고 보장할 단계가 아니다.</p>
<p>다음 실험은 개체·위치 품질과 역할 가림을 분리해 학습하되, 기존 visibility 경로와 source-only로 대조해야 한다. 이 변화가 목재/정사각형 전체에서 실제로 개선되는지를 보여야 진전으로 인정할 수 있다. 자기 가림은 상류 기하 필터와 별도로 유지하고 PnP나 좌표 보정은 하지 않는다.</p>
<h2>남은 작업</h2><ol>{''.join('<li>'+html.escape(s)+'</li>' for s in next_actions)}</ol>
<h2>실행·증거 기록</h2><p>이번 단계의 새 학습0 / 새 current223 추론0 / PnP0. 초기 모델과 동결 encoder 입력 추출은 실제 실행했다. 첫 다운로드의 빈응답4건은 실패 기록을 보존한 뒤 동일 pinned4파일만 복구했다. HTML은 정적 로컬 readout이며 브라우저 렌더를 검증했다고 주장하지 않는다.</p><ul>{link_list}</ul>
<p class="muted">원본 출처: <a href="https://huggingface.co/datasets/anas-gouda/mr6d">MR6D · Gouda 외 · CC-BY-4.0</a>, pinned commit3c4ed8453312655499c3d3a3cf38b4929fae77ee. source scene 분리는 공식 blind benchmark가 아니며 물리 물체가 반복될 수 있다. Euro 목재만으로 정사각형 감독 공백이 채워지지 않는다.</p></html>'''
(V / 'ROOT_READOUT_KO.html').write_text(doc)
refs = re.findall(r'(?:src|href)="([^"]+)"', doc)
local = [r for r in refs if not r.startswith(('https://','http://','#'))]
missing = [r for r in local if not (V / r).resolve().exists()]
assert not missing, missing
save('ROOT_READOUT_STATIC_QA.json',dict(completed=True,PASS=True,
    report=binding('ROOT_READOUT_KO.html'),results=binding('ROOT_RESULTS_KO.json'),
    plot=binding('CURRENT_PROGRESS_AND_LIMITS_KO.png'),local_references_checked=len(local),
    missing_references=missing,HTML_browser_render_claim=False,
    benchmark_counts_and_percentages_derived_from_unchanged_saved_V149=True,
    visual_QA_pending_plot=True,root_actually_viewed_query_boards=[73,95,109]))
print(json.dumps(dict(results=binding('ROOT_RESULTS_KO.json'),report=binding('ROOT_READOUT_KO.html'),
    static_QA=binding('ROOT_READOUT_STATIC_QA.json')),ensure_ascii=False))
