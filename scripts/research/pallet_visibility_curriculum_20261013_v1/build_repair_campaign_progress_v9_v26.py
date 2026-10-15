"""Append-only v26 accounting after all238 prediction, audit and score.

Source is prepared without reading new predictions or metrics. Execution uses
sealed small score rows only; large prediction journals are hashed, not decoded.
The previous23 uniform methods are copied unchanged and six new controls added.
No image/reference decode, fit, inference or per-frame best-method routing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path

from . import build_repair_campaign_progress_v8_v25 as PREVIOUS
from . import build_repair_campaign_progress_v6_v23 as COMMON
from . import evaluate_prepartial_RGB_preservation_v4_26 as V


P, C = PREVIOUS.P, PREVIOUS.C
SOURCE24, SOURCE25 = V.V, PREVIOUS.V
OUTPUT = PREVIOUS.OUTPUT
NAMES = ('PROGRESS_V9_V26.json', 'RESULTS_V9_V26.html', 'PROGRESS_V9_V26_SEAL.json')
LABELS = {
    V.ARMS[1]: 'v26 partial 이전 RGB q 복구 · noP 영상선',
    V.ARMS[2]: 'v26 partial 이전 RGB q 복구 · noP 영상선 · e WD 유지',
    V.ARMS[3]: 'v26 partial 이전 RGB q 복구 · 위치 prior · 영상선',
    V.ARMS[4]: 'v26 partial 이전 RGB q 복구 · 위치 prior · 영상선 · e WD 유지',
    V.ARMS[5]: 'v26 partial 이전 RGB q 복구 · 위치 prior · 포함 제약',
    V.ARMS[6]: 'v26 partial 이전 RGB q 복구 · 위치 prior · 포함 제약 · e WD 유지',
}


def _profile_contracts():
    """Explicit noP/TP/C controls; FULL here is only the zero-weight schema."""
    expected = (
        ('NATIVE_ZERO_COARSE', 'FULL', False, V.ARMS[1:3]),
        ('TRANSLATION_NATIVE', 'TRANSLATION', False, V.ARMS[3:5]),
        ('TRANSLATION_CONTAINMENT', 'TRANSLATION', True, V.ARMS[5:7]),
    )
    if V.PROFILES != expected or len(V.ARMS) != 7 or len(set(V.ARMS)) != 7:
        raise ValueError('Exact noP, TP, TP+C pairs and six distinct new controls required')
    profiles = {}
    for name, schema, containment, arms in expected:
        for index, arm in enumerate(arms):
            profiles[arm] = dict(restore_prepartial_nonfallback_RGB_or_unknown_q=True,
                new_CAL_edge_noise_generation='V26', new_CAL_edge_noise_fits=1,
                shared_v24_TCAL_gamma=True, coarse_prior_schema_mode=schema,
                coarse_prior_weight=0. if schema == 'FULL' else 1.,
                active_rotation_prior=False,
                noP_FULL_schema_is_zero_weight_placeholder=schema == 'FULL',
                translation_prior_uses_existing_marginal_covariance=schema == 'TRANSLATION',
                coarse_bias_strength=1., bias_subtracted_once_before_whitening=True,
                containment_enabled=containment, containment_weight=1. if containment else 0.,
                fixed_old_containment_configuration=True,
                old_symmetric_mask_cost_or_numeric_tie_break_used=False,
                e_predicted_family_constraint=index == 1, e_numeric_pose_target_or_fallback=False,
                source_profile=name)
    return profiles


def _paired_definitions():
    """Eight fixed comparisons; none is a GT-selected per-frame mixture."""
    result = [
        dict(before_run='SOURCE24', after_run='V26', before=SOURCE24.ARMS[3], after=V.ARMS[1],
            intervention='PREPARTIAL_RESTORE_NEW_NOISE_REMOVE_OLD_MASK_TIE',
            label='v24 SOURCE native → v26 noP native',
            interpretation='partial 이전 RGB/unknown q 복구 + 별도 CAL edge noise + 기존 numeric whole-mask tie 제거의 동시 변경; 단일 요소 효과 아님'),
        dict(before_run='SOURCE25', after_run='V26', before=SOURCE25.ARMS[1], after=V.ARMS[3],
            intervention='PREPARTIAL_RESTORE_AND_NEW_NOISE_TP_NATIVE',
            label='v25 TP native → v26 TP native',
            interpretation='같은 TP/기존 O-S/시작점 정책에서 q 복구와 적용집단별 새 CAL noise가 함께 변경'),
        dict(before_run='SOURCE25', after_run='V26', before=SOURCE25.ARMS[3], after=V.ARMS[5],
            intervention='PREPARTIAL_RESTORE_AND_NEW_NOISE_TP_CONTAINMENT',
            label='v25 TP+C → v26 TP+C',
            interpretation='같은 TP/C/기존 O-S/시작점 정책에서 q 복구와 적용집단별 새 CAL noise가 함께 변경'),
        dict(before_run='V26', after_run='V26', before=V.ARMS[1], after=V.ARMS[3],
            intervention='ADD_TRANSLATION_PRIOR_SAME_RESTORED_INPUTS_NOISE',
            label='v26 noP native → TP native',
            interpretation='같은 복구 입력·CAL noise·cached 후보·v19 start 이름에서 TP 비용 추가; 연속 최적화 endpoint는 달라질 수 있음'),
        dict(before_run='V26', after_run='V26', before=V.ARMS[3], after=V.ARMS[5],
            intervention='ADD_CONTAINMENT_RETAIN_SAME_MODE_NATIVE_ENDPOINTS',
            label='v26 TP native → TP+C',
            interpretation='같은 TP native endpoint를 후보로 보존하고 one-sided C를 최적화·선택 총비용에 한 번 추가'),
    ]
    for before, after, profile in ((V.ARMS[1],V.ARMS[2],'noP'),
            (V.ARMS[3],V.ARMS[4],'TP'),(V.ARMS[5],V.ARMS[6],'TP+C')):
        result.append(dict(before_run='V26', after_run='V26', before=before, after=after,
            intervention='PREDICTED_E_FAMILY_CONSTRAINT', label='v26 '+profile+' unlocked → e WD 유지',
            interpretation='같은 전체 raw-feasible 후보를 e 예측 family로 AUTH 전에 제한; e 수치 pose/fallback이나 GT family 선택 없음'))
    if len(result) != 8 or len({(r['before'],r['after']) for r in result}) != 8:
        raise ValueError('Eight distinct predeclared paired controls required')
    return result


def _require_same_reference_bindings(before, after):
    for part in P.COUNTS:
        a, b = before[part]['reference_binding'], after[part]['reference_binding']
        if (a['path'],a['sha256']) != (b['path'],b['sha256']):
            raise ValueError('Unchanged fixed reference binding required for ' + part)


def _require_same_input_bytes(protocol, folder, previous_folder):
    current=C.read(folder/'INPUT_SEAL.json')
    previous=C.read(previous_folder/'INPUT_SEAL.json')
    if (protocol.get('input_bytes_identical_to_v25') is not True
            or current['rows']!=238 or previous['rows']!=238
            or current['input_sha256']!=previous['input_sha256']
            or C.sha(folder/'INPUTS.jsonl.gz')!=current['input_sha256']
            or C.sha(previous_folder/'INPUTS.jsonl.gz')!=previous['input_sha256']):
        raise ValueError('Byte-identical original v25 all238 inputs required')


def _same_previous_methods(previous, methods):
    if (previous['method_count'] != 23 or len(previous['methods']) != 23
            or methods[:23] != previous['methods']
            or len(methods) != 29 or len({m['method'] for m in methods}) != 29):
        raise ValueError('Keep all previous23 rows unchanged and add exactly six distinct uniform methods')
    if any(m['current_default'] for m in methods[23:]):
        raise ValueError('No new default adoption')


def _render(data):
    lines = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>v26 전체192 비교</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1700px;margin:32px auto;padding:0 20px;color:#192536}'
        'table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{border-bottom:1px solid #ccd4df;'
        'padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef4fa}'
        '.scroll{overflow-x:auto}.new{background:#f3faf5}small{font-size:12px}</style>',
        '<p>기본 e는 유지합니다. 이전 23개 고정 방법을 그대로 남기고 v26의 6개 방법을 추가했습니다. '
        '각 방법은 TEST49와 추가143장 모두에 동일하게 적용했습니다. 사진별 최선 방법을 섞지 않고 '
        '모든 실패·fallback·native 대응 플래그를 포함합니다. severe 제외, CAL46은 별도입니다.</p>',
        '<p>v24/v25 SOURCE는 당시 complete RGB/unknown q만 보존했습니다. v26은 이미 partial로 전환된 '
        'B/I의 partial 이전에 보관된 RGB/unknown q·원 공분산·reliability도 그대로 2D 비용으로 복구합니다. '
        '정확한 VISIBLE_BACKBONE_FALLBACK만 기존 partial 정책을 유지합니다. 복구한 owner의 중복 normal·접선·edge 비용은 '
        '제거하되 다른 owner가 남은 공유 edge는 한 번 유지합니다. 기존 partial hard guard는 그대로입니다.</p>',
        '<p>활성 edge 집단이 달라 별도 CAL noise를 한 번 적합했습니다. support 총 prior를 같게 두고 '
        '서로 다른 fit variant에 나누는 기존 NLL·조건부 floor·3 starts·4 LOGO를 유지했습니다. '
        'v24의 접선 γ·O/S·coarse 모델은 다시 적합하지 않았습니다. 복구 q와 공유 영상선의 통계적 독립을 주장하지 않습니다.</p>',
        '<p>noP는 coarse prior 가중치 0입니다. 내부 FULL schema는 0 가중치 placeholder이며 회전 prior가 아닙니다. '
        'TP는 첫 N3_SUBPIX 모델 중심의 ray/log-depth와 기존 3×3 주변 공분산을 쓰고 λ=1·편향 강도=1입니다. '
        'C는 고정 관측 전역 hull 표본이 전체 투영 모델 hull 밖에 있을 때만 주는 soft 포함 비용이며 λ=1로 한 번만 합산합니다. '
        '기존 대칭 전체 마스크 비용은 native 수치 동률에서도 쓰지 않습니다.</p>',
        '<p>e WD 유지는 raw-feasible 후보를 e가 예측한 family로 먼저 제한한 별도 선택기입니다. '
        '그 family는 과거 마스크에 의존하며 물리 축의 인증이 아닙니다. e 수치 pose를 target이나 fallback으로 사용하지 않습니다. '
        'S는 자세 투영만 출력하고 S 영상·2D·영역 관측은 없습니다. 제한된 초기 모델 S prior와 기존 O/S 비용·U 영역은 유지합니다.</p>']
    for part,title in (('PRIMARY192','전체192 = TEST49 + 추가143'),('TEST','기존 TEST49'),
            ('REMAINDER','추가143'),('CAL','별도 CAL46')):
        lines.append('<p><b>'+title+'</b> — T≤5cm는 위치 오차, 공동은 5cm·5° 모두 만족한 수입니다.</p>'
            '<div class="scroll"><table><tr>'+''.join('<th>'+s+'</th>' for s in (
                '고정 방법','수','T≤5cm','공동','평균 T(cm)','최대 T(cm)','최대 T 사례',
                '평균 R(°)','최대 R(°)','R 미확정','fallback','기존49 보존'))+'</tr>')
        for method in data['methods']:
            q=method['populations'][part]
            numbers=[q['count'],q['translation_le5'],q['successes_5cm_5deg'],
                q['translation_cm']['mean'],q['translation_cm']['max'],q['max_translation_id'],
                q['rotation_deg']['mean'],q['rotation_deg']['max'],q['rotation_unresolved'],
                q['fallback_count'],'예' if method['original_TEST49_all_translation_le5'] else '아니오']
            cells=['미확정' if x is None else f'{x:.3f}' if isinstance(x,float) else html.escape(str(x)) for x in numbers]
            lines.append('<tr'+(' class="new"' if method['method'] in V.ARMS[1:] else '')+'><td>'+
                html.escape(method['label_KO'])+'</td>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
        lines.append('</table></div>')
    lines.extend(['<p>위 29개×4집단=116행은 모두 공개한 균일 방법입니다. 새 8개 paired 비교는 JSON에 '
        '원래 행별 gain/loss와 최대 사례를 남깁니다. v24 SOURCE→v26 noP 차이는 q 복구·새 CAL noise·옛 '
        'numeric mask tie 제거를 함께 포함하며, v25→v26도 q 복구와 새 noise를 분리한 단일 요인 실험이 아닙니다.</p>',
        '<p>추가143장의 참조는 저장 annotation/PnP 유도값이며 독립 물리 계측이 아닙니다. '
        '같은 TEST를 반복 사용한 탐색 결과입니다. accepted·비용 감소·family 제한·최적화 수렴이 '
        '정확도나 유일성을 보장하지 않습니다. 전체192 목표 완료나 새 기본 방법 채택을 이 문서에서 선언하지 않습니다.</p>',
        '<p><a href="PROGRESS_V9_V26.json">전체 수치·8개 대응 비교·원본 봉인 근거</a> · '
        '<a href="RESULTS_V8_V25.html">이전 V8 문서</a></p></html>'])
    return '\n'.join(lines)


def build(*, root_postscore_GO=False):
    if root_postscore_GO is not True:
        raise ValueError('Wait for all7-arm prediction, zero-failure independent audit, score and root accounting GO')
    paths=[OUTPUT/name for name in NAMES]
    if any(path.exists() for path in paths):
        raise FileExistsError('Append-only V9: preserve all previous campaign documents')
    profiles=_profile_contracts()
    definitions=_paired_definitions()
    previouspath=OUTPUT/'PROGRESS_V8_V25.json'
    previoussealpath=OUTPUT/'PROGRESS_V8_V25_SEAL.json'
    previous,previousseal=C.read(previouspath),C.read(previoussealpath)
    for key in ('source','json','html'):
        binding=previousseal[key]
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:
            raise ValueError('Frozen V8 source/document changed')
    if previous['method_count']!=23 or len(previous['methods'])!=23:
        raise ValueError('Exactly previous23 uniform methods required')
    protocol,reports=P._load_run(V.OUTPUT)
    old25protocol,old25=P._load_run(SOURCE25.OUTPUT)
    old24protocol,old24=P._load_run(SOURCE24.OUTPUT)
    COMMON._require_declared_arms(V.OUTPUT,V.ARMS)
    predseal=C.read(V.OUTPUT/'PREDICTION_SEAL.json')
    if (predseal.get('all7_arms_complete') is not True
            or protocol['ids']!=old25protocol['ids'] or protocol['ids']!=old24protocol['ids']):
        raise ValueError('Seven complete arms on the exact original ordered238 required')
    _require_same_input_bytes(protocol,V.OUTPUT,SOURCE25.OUTPUT)
    for before in (old25,old24):
        COMMON._same_e(before,reports)
        _require_same_reference_bindings(before,reports)
    auditpaths=COMMON._require_audit(V.OUTPUT)
    methods=list(previous['methods'])
    for arm in V.ARMS[1:]:
        populations=COMMON._population_summaries(reports,arm)
        methods.append(dict(method=arm,label_KO=LABELS[arm],populations=populations,
            source_output=str(V.OUTPUT.relative_to(C.ROOT)),current_default=False,
            original_TEST49_all_translation_le5=populations['TEST']['translation_le5']==49,
            numerical_primary192_target_met=populations['PRIMARY192']['translation_le5']==192,
            intervention_contract=profiles[arm]))
    _same_previous_methods(previous,methods)
    runs={'SOURCE24':old24,'SOURCE25':old25,'V26':reports}
    paired=[]
    for definition in definitions:
        before,after=runs[definition['before_run']],runs[definition['after_run']]
        a,b=definition['before'],definition['after']
        parts={part:COMMON.PAIR._paired(before[part]['rows'],after[part]['rows'],a,b) for part in P.COUNTS}
        parts['PRIMARY192']=COMMON.PAIR._paired(before['TEST']['rows']+before['REMAINDER']['rows'],
            after['TEST']['rows']+after['REMAINDER']['rows'],a,b)
        paired.append(dict(**definition,populations=parts))
    evidence=[Path(__file__),Path(PREVIOUS.__file__),Path(COMMON.__file__),Path(P.__file__),
        Path(COMMON.PAIR.__file__),previouspath,previoussealpath,*auditpaths,V.PLAN]
    for folder in (V.OUTPUT,SOURCE24.OUTPUT,SOURCE25.OUTPUT):
        evidence.extend(folder/name for name in ('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json',
            'SCORE_SEAL.json','CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json'))
    for module in (V.RESTORE_CAL,SOURCE24.TCAL):
        evidence.extend(module.OUTPUT/name for name in ('MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'))
    evidence.extend(V.RESTORE_CAL.OUTPUT/name for name in ('PROTOCOL.json','PREPARATION_SEAL.json',
        'OWNER_EXECUTION_RECEIPT_V1.json','CAL_MIXTURE_ERRORS.jsonl.gz'))
    evidence.extend(V.RESTORE_CAL.PREPARED/name for name in ('PREPARATION_SEAL.json',
        'CAL_PREFLIGHT_V1.json','CAL_PREFLIGHT_SEAL_V1.json','CAL_PREPARED_GROUPS.jsonl.gz'))
    for relative in ('prepartial_RGB_source_audit_v4_26/AUDIT.json',
            'prepartial_RGB_source_audit_v4_26/AUDIT_SEAL.json',
            'prepartial_RGB_source_audit_v4_26/PATH_CLOSURE_V1.json',
            'prepartial_RGB_preservation_peer_review_v4_26/CAL_SOURCE_AND_METADATA_REVIEW_V1.json',
            'independent_CAL_prepartial_RGB_review_v4_26/AUDIT.json'):
        evidence.append(V.OUTPUT.parent/relative)
    active=C.read(OUTPUT/'ACTIVE_GOAL_PROTOCOL_V1.json')
    P._verify(active['protected_user_sha256'])
    bindings=[P._binding(path) for path in dict.fromkeys(evidence)]
    data=dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V9_PREPARTIAL_RGB_V26',
        generated_UTC=datetime.now(timezone.utc).isoformat(),goal_status='active',goal_completion_declared=False,
        default_changed=False,default_method=P.BASELINE,methods=methods,method_count=29,
        previous_method_count=23,new_method_count=6,population_table_rows=116,
        paired_controls=paired,paired_control_count=8,exact_baseline_consistency_verified=True,
        original_V8_methods_retained_exactly=True,primary=dict(TEST49=49,REMAINDER143=143,total=192,severe=0),separate_CAL=46,
        v26_inputs_byte_identical_to_v25=True,
        numerical_target_met_any_new_method=any(m['numerical_primary192_target_met'] for m in methods[23:]),
        original49_preserved_by_new_method={m['method']:m['original_TEST49_all_translation_le5'] for m in methods[23:]},
        all_methods_uniform_full_population=True,no_per_frame_best_or_GT_ID_partition_routing=True,
        all238_no_frame_fallback_or_native_flag_filter=True,S_image_2D_or_regions=0,U_regions_only=True,
        original_hard_guards_unchanged=True,original_v19_unregularized_start_names_policy_unchanged=True,
        new_v26_controls_old_symmetric_mask_not_used_even_at_ties=True,no_numeric_e_pose_prior_or_fallback=True,
        new_v26_contract_scope='six new methods only; inherited23 retain exact original recipes and metrics',
        restore_nonfallback_prepartial_q_source_covariance_reliability_exact=True,
        removed_restored_owner_old_normal_tangent_and_edge_costs_only=True,shared_other_owner_edge_once=True,
        new_v26_CAL_edge_noise_fit_count=1,new_v26_TCAL_O_S_or_coarse_prior_fit_count=0,
        coarse_prior_model_reused=True,noP_weight=0.,TP_weight=1.,C_weight=1.,
        noP_FULL_schema_is_zero_weight_placeholder=True,translation_prior_uses_existing_marginal_covariance=True,
        coarse_bias_subtracted_once=True,one_sided_containment_is_soft_conditional_sample_cue_not_subset_certificate=True,
        old_e_predicted_family_depends_on_historical_whole_mask=True,
        RGB_q_and_retained_image_edges_not_claimed_independent=True,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,
        reused_exploratory_TEST_not_new_holdout=True,numeric_reference_bindings_unchanged=True,
        no_new_inference_pose_fit_images_or_reference_decode=True,
        independent_root_audits=[P._binding(auditpaths[0])],evidence=bindings,
        protected_user_sha256=active['protected_user_sha256'])
    for binding in bindings:
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:
            raise ValueError('Evidence changed during aggregation')
    C.write(paths[0],data)
    paths[1].write_text(_render(data),encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2],dict(source=P._binding(Path(__file__)),json=P._binding(paths[0]),html=P._binding(paths[1]),
        evidence=bindings,exact_e_preserved=True,original_V8_methods_retained_exactly=True,
        method_count=29,population_table_rows=116,paired_control_count=8,
        no_new_inference_pose_fit_images_or_reference_decode=True))
    print(dict(methods=29,table_rows=116,paired_controls=8,json_sha256=C.sha(paths[0]),
        html_sha256=C.sha(paths[1])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-v26-score-and-root-audit-go',action='store_true',required=True)
    args=parser.parse_args()
    build(root_postscore_GO=args.after_v26_score_and_root_audit_go)
