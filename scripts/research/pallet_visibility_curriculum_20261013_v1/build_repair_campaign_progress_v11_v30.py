"""Append four uniform V30 methods only after complete audit and fixed scores.

Source preparation uses no actual scores, observations or predictions. At an
explicit postscore GO, only small score rows are decoded; large journals are
hash-bound. The previous33 methods, fixed references and default e are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path

from . import build_repair_campaign_progress_v10_v29 as PREVIOUS
from . import evaluate_geometry_guided_edges_v4_30 as V

C, P, COMMON = PREVIOUS.C, PREVIOUS.P, PREVIOUS.COMMON
OUTPUT = PREVIOUS.OUTPUT
NAMES = ('PROGRESS_V11_V30.json', 'RESULTS_V11_V30.html', 'PROGRESS_V11_V30_SEAL.json')
LABELS = {
    V.ARMS[1]: 'v30 원 B/I 대응 검사 · B/I 방향 교체',
    V.ARMS[2]: 'v30 원 B/I 대응 검사 · B/I 방향 교체 + 포함 제약',
    V.ARMS[3]: 'v30 원 B/I 대응 검사 · 기존 RGB q 보존',
    V.ARMS[4]: 'v30 원 B/I 대응 검사 · 기존 RGB q 보존 + 포함 제약',
}


def _profile_contracts():
    if (len(V.ARMS) != 5 or len(set(V.ARMS)) != 5 or V.ARMS[0] != P.BASELINE
            or set(V.POLICY_ARMS) != set(V.G.SOURCE_POLICIES)):
        raise ValueError('Exact e plus four distinct uniform V30 methods required')
    result = {}
    for policy in V.G.SOURCE_POLICIES:
        arms = V.POLICY_ARMS[policy]
        if len(arms) != 2:
            raise ValueError('Each policy must declare native then containment')
        for index, arm in enumerate(arms):
            result[arm] = dict(source_policy=policy, containment_weight=float(index),
                selection_profile=V.G.SELECTION_PROFILE,
                noise_observation_family=V.G.OBSERVATION_FAMILY,
                raw_observation_generation=V.G.RAW_OBSERVATION_GENERATION,
                original_named_BI_to_actual_infinite_line_normal_reach_px=43.,
                raw_RGB_chains_guides_indices_q_covariance_states_and_regions_unchanged=True,
                new_RGB_NN_NMS_or_pose_guide_fits=0,
                fresh_policy_specific_CAL_line_and_actual_axis_tangent_noise=True,
                old_V29_noise_relabelled=False, guide_physical_Q_transport_unchanged=True,
                same_v19_native_six_diverse_plus_coarse_start_policy=True,
                full_pose_TP_topnormal_prior_weight=0., bounded_O_S_regularizers_unchanged=True,
                old_symmetric_mask_cost_or_numeric_tie_break_used=False,
                e_numeric_pose_target_prior_or_fallback=False,
                S_image_2D_or_regions=0, U_approximate_regions_only=True)
    if list(result) != list(V.ARMS[1:]):
        raise ValueError('Policy and selector positions must match the declared arms')
    return result


def _paired_definitions():
    """Four matched V29→V30 pairs plus four within-V30 fixed controls."""
    result = []
    old = PREVIOUS.V
    if set(old.POLICY_ARMS) != set(V.POLICY_ARMS):
        raise ValueError('Identical source policies required across generations')
    for policy in V.G.SOURCE_POLICIES:
        before, after = old.POLICY_ARMS[policy], V.POLICY_ARMS[policy]
        if len(before) != 2 or len(after) != 2:
            raise ValueError('Exact native/containment selector pairs required')
        for index, selector in enumerate(('native', 'containment')):
            result.append(dict(before_run='V29', after_run='V30', before=before[index],
                after=after[index], source_policy=policy, selector=selector, selector_index=index,
                label_KO='v29 → v30: ' + LABELS[after[index]],
                intervention='ORIGINAL_BI_ASSOCIATION_GATE_AND_NEW_POPULATION_CAL_NOISE',
                interpretation_KO='같은 실제 선 보관자료에 원 B/I 대응 검사를 추가하고 적용 집단별 CAL 선·접선 잡음을 다시 적합한 묶음 변경; 단일 요소 효과 아님'))
    for before, after, label in (
            (V.ARMS[1], V.ARMS[2], 'B/I 교체에서 포함 제약 추가'),
            (V.ARMS[3], V.ARMS[4], 'RGB q 보존에서 포함 제약 추가'),
            (V.ARMS[1], V.ARMS[3], 'native에서 RGB q 보존 정책 비교'),
            (V.ARMS[2], V.ARMS[4], '포함 제약에서 RGB q 보존 정책 비교')):
        result.append(dict(before_run='V30', after_run='V30', before=before, after=after,
            label_KO='v30 ' + label, intervention='FIXED_WITHIN_V30_CONTROL',
            interpretation_KO='선언된 두 균일 방법의 대응 비교; source 정책 비교에는 정책별 CAL 잡음 차이도 포함'))
    if len(result) != 8 or len({(p['before'], p['after']) for p in result}) != 8:
        raise ValueError('Eight distinct fixed paired controls required')
    return result


def _load_new_run(folder):
    protocol = C.read(folder / 'PROTOCOL.json')
    pred, score = C.read(folder / 'PREDICTION_SEAL.json'), C.read(folder / 'SCORE_SEAL.json')
    if (pred['rows'] != 238 or pred['arms'] != list(V.ARMS) or pred['arm_count'] != 5
            or pred['all_declared_arms_complete'] is not True
            or pred['before_new_control_metric_read'] is not True
            or C.sha(folder / 'PREDICTIONS.jsonl.gz') != pred['predictions_sha256']
            or C.sha(folder / 'PROTOCOL.json') != pred['protocol_sha256']
            or C.sha(folder / 'INPUT_SEAL.json') != pred['input_seal_sha256']
            or C.sha(folder / 'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']
            or protocol['selection_profile'] != V.G.SELECTION_PROFILE
            or protocol['noise_observation_family'] != V.G.OBSERVATION_FAMILY
            or protocol['raw_observation_generation'] != V.G.RAW_OBSERVATION_GENERATION):
        raise ValueError('Complete exact five-arm V30 prediction, profile and score seals required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        P._verify(protocol[key])
    reports = {}
    for part, count in P.COUNTS.items():
        path = folder / (part + '_SCORE.json')
        if C.sha(path) != score['score_sha256'][part]:
            raise ValueError('Frozen score changed')
        report = C.read(path)
        if (report['count'] != count or len(report['rows']) != count
                or len({r['id'] for r in report['rows']}) != count
                or any(set(r['metrics']) != set(V.ARMS) for r in report['rows'])
                or any(set(r['accepted']) != set(V.ARMS) for r in report['rows'])):
            raise ValueError('Keep every original row, declared method and fallback')
        reports[part] = report
    if [r['id'] for part in P.COUNTS for r in reports[part]['rows']] != protocol['ids']:
        raise ValueError('Exact ordered238 inventory required')
    return protocol, reports


def _same_previous_methods(previous, methods):
    if (previous['method_count'] != 33 or len(previous['methods']) != 33
            or methods[:33] != previous['methods'] or len(methods) != 37
            or len({m['method'] for m in methods}) != 37):
        raise ValueError('Keep all previous33 methods exact and append four distinct controls')
    if any(m['current_default'] for m in methods[33:]):
        raise ValueError('No new default adoption')


def _render(data):
    out = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>v30 전체192 비교</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1700px;margin:30px auto;padding:0 20px}'
        'table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ccd4df;text-align:right}'
        'th:first-child,td:first-child{text-align:left}th{background:#eef4fa}.new{background:#eff8ef}.scroll{overflow:auto}</style>',
        '<p>기본 e를 포함한 기존 33개 방법은 그대로 유지합니다. 새 4개 방법도 TEST49와 추가143 모두에 동일하게 적용하며 '
        '실패·fallback·제로 관측 행을 포함합니다. severe 제외 후 정량 평가 가능한 추가143이며 CAL46은 별도입니다.</p>',
        '<p>v30은 기존 v29 실제 RGB 선·마스크·guide·q·공분산·상태를 보존하고, 원래 이름이 붙은 B/I 끝점과 '
        '실제 무한 TLS 선의 법선 거리가 기존 탐색 범위 43px 안인지 확인한 뒤 같은 ordered graph를 다시 평가합니다. '
        '새 픽셀 탐색이나 NN·선 후보 생성은 없습니다. 물리 Q guide 변환과 WD 가설은 v29와 같습니다.</p>',
        '<p>새로 선택된 집단에 맞춰 정책별 CAL 선 잡음과 실제 선 접선축의 보관 q 잡음을 다시 적합했습니다. '
        'v29→v30 대응 비교는 대응 검사와 새 CAL 잡음이 함께 바뀐 대조이며 단일 요인 효과가 아닙니다. '
        '기존 RGB q 보존 정책과 전체 B/I 교체 정책은 별도로 표시합니다.</p>',
        '<p>기존 native 6개 시작점과 coarse 정책, O/S 제한 벌점, U 영역 및 hard guard는 유지합니다. '
        'full pose·translation·top-normal prior는 0이며 선택적 포함 비용 C는 λ1입니다. '
        'e 수치 자세를 prior나 fallback으로 넣지 않습니다. S 영상 점·국소 질의·2D 비용은 없고 '
        '모델/PnP 투영과 기존 coarse 유래 bounded S 항은 유지합니다. U는 대략적인 영역만 사용합니다. '
        '별도 다른 물체 가림 마스크는 없으며 target mask가 없는 구간은 unknown입니다.</p>']
    for part, label in (('PRIMARY192', '전체192'), ('TEST', '기존 TEST49'),
            ('REMAINDER', '추가143'), ('CAL', '별도 CAL46')):
        out.append('<p><b>' + label + '</b> — 공동은 위치 5cm 및 회전 5°를 모두 만족한 수입니다.</p>'
            '<div class="scroll"><table><tr>' + ''.join('<th>' + x + '</th>' for x in (
                '고정 방법', '수', '위치≤5cm', '공동', '평균 위치 cm', '최대 위치 cm', '최대 사례',
                '평균 회전 °', '최대 회전 °', '회전 미확정', 'fallback', '기존49 보존')) + '</tr>')
        for method in data['methods']:
            q = method['populations'][part]
            values = [q['count'], q['translation_le5'], q['successes_5cm_5deg'],
                q['translation_cm']['mean'], q['translation_cm']['max'], q['max_translation_id'],
                q['rotation_deg']['mean'], q['rotation_deg']['max'], q['rotation_unresolved'],
                q['fallback_count'], '예' if method['original_TEST49_all_translation_le5'] else '아니오']
            cells = ['미확정' if x is None else f'{x:.3f}' if isinstance(x, float) else str(x) for x in values]
            out.append('<tr' + (' class="new"' if method['method'] in LABELS else '') + '><td>'
                + html.escape(method['label_KO']) + '</td>'
                + ''.join('<td>' + html.escape(x) + '</td>' for x in cells) + '</tr>')
        out.append('</table></div>')
    out.extend(['<p>37개 방법×4집단=148행입니다. JSON에는 분포·최대 사례·8개 고정 대응 비교도 보존합니다. '
        '같은 TEST를 반복 사용한 탐색 결과이고 추가143 참조는 annotation/PnP 유도값입니다. '
        '독립 물리 계측·새 holdout 일반화·전역 최적해·자세 유일성·새 기본 방법 채택을 주장하지 않습니다.</p>',
        '<p><a href="PROGRESS_V11_V30.json">전체 수치와 대응 비교</a> · '
        '<a href="RESULTS_V10_V29.html">이전 결과</a></p></html>'])
    return '\n'.join(out)


def build(*, root_postscore_GO=False):
    if root_postscore_GO is not True:
        raise ValueError('Wait for complete V30 prediction, independent audit, score and root accounting GO')
    paths = [OUTPUT / name for name in NAMES]
    if any(p.exists() for p in paths):
        raise FileExistsError('Append-only V11 campaign accounting')
    profiles, definitions = _profile_contracts(), _paired_definitions()
    oldpath, oldseal = OUTPUT / 'PROGRESS_V10_V29.json', OUTPUT / 'PROGRESS_V10_V29_SEAL.json'
    previous, previousseal = C.read(oldpath), C.read(oldseal)
    for key in ('source', 'json', 'html'):
        binding = previousseal[key]
        if C.sha(C.ROOT / binding['path']) != binding['sha256']:
            raise ValueError('Previous frozen campaign changed')
    protocol, reports = _load_new_run(V.OUTPUT)
    oldprotocol, oldreports = PREVIOUS._load_new_run(PREVIOUS.V.OUTPUT)
    if protocol['ids'] != oldprotocol['ids']:
        raise ValueError('Original ordered238 shared by both methods required')
    COMMON._require_declared_arms(V.OUTPUT, V.ARMS)
    COMMON._same_e(oldreports, reports)
    PREVIOUS.PREVIOUS._require_same_reference_bindings(oldreports, reports)
    auditpaths = COMMON._require_audit(V.OUTPUT)
    methods = list(previous['methods'])
    for arm in V.ARMS[1:]:
        populations = COMMON._population_summaries(reports, arm)
        methods.append(dict(method=arm, label_KO=LABELS[arm], populations=populations,
            source_output=str(V.OUTPUT.relative_to(C.ROOT)), current_default=False,
            original_TEST49_all_translation_le5=populations['TEST']['translation_le5'] == 49,
            numerical_primary192_target_met=populations['PRIMARY192']['translation_le5'] == 192,
            intervention_contract=profiles[arm]))
    _same_previous_methods(previous, methods)
    runs, paired = {'V29': oldreports, 'V30': reports}, []
    for definition in definitions:
        before, after = runs[definition['before_run']], runs[definition['after_run']]
        a, b = definition['before'], definition['after']
        parts = {part: COMMON.PAIR._paired(before[part]['rows'], after[part]['rows'], a, b) for part in P.COUNTS}
        parts['PRIMARY192'] = COMMON.PAIR._paired(before['TEST']['rows'] + before['REMAINDER']['rows'],
            after['TEST']['rows'] + after['REMAINDER']['rows'], a, b)
        paired.append(dict(**definition, populations=parts))
    active = C.read(OUTPUT / 'ACTIVE_GOAL_PROTOCOL_V1.json')
    P._verify(active['protected_user_sha256'])
    prep, cal = V._modules()
    evidence = [Path(__file__), Path(PREVIOUS.__file__), Path(COMMON.__file__),
        Path(P.__file__), Path(COMMON.PAIR.__file__), oldpath, oldseal, *auditpaths, V.PLAN]
    for folder in (V.OUTPUT, PREVIOUS.V.OUTPUT):
        evidence.extend(folder / name for name in ('PROTOCOL.json', 'INPUT_SEAL.json', 'PREDICTION_SEAL.json',
            'SCORE_SEAL.json', 'CAL_SCORE.json', 'TEST_SCORE.json', 'REMAINDER_SCORE.json'))
    evidence.extend(cal.OUTPUT / name for name in ('PROTOCOL.json', 'PREPARATION_SEAL.json',
        'MODEL_SEAL.json', 'CAL_LOGO_REPORT.json'))
    evidence.extend(prep.OBS / name for name in ('FINAL_PROTOCOL.json', 'OBSERVATION_SEAL.json',
        'CAL_OBSERVATION_SEAL.json', 'REST_OBSERVATION_SEAL.json', 'FINAL_COMPLETION_RECEIPT_V30B.json'))
    bindings = [P._binding(path) for path in dict.fromkeys(evidence)]
    data = dict(schema='ACTIVE_REPAIR_CAMPAIGN_PROGRESS_V11_OBSERVED_BI_ASSOCIATED_LINES_V30',
        generated_UTC=datetime.now(timezone.utc).isoformat(), goal_status='active', goal_completion_declared=False,
        default_changed=False, default_method=P.BASELINE, method_count=37, previous_method_count=33,
        new_method_count=4, population_table_rows=148, methods=methods,
        paired_control_count=8, paired_controls=paired, original_V10_methods_retained_exactly=True,
        primary=dict(TEST49=49, REMAINDER143=143, total=192, severe=0), separate_CAL=46,
        exact_baseline_consistency_verified=True, all_methods_uniform_full_population=True,
        numerical_target_met_any_new_method=any(m['numerical_primary192_target_met'] for m in methods[33:]),
        original49_preserved_by_new_method={m['method']: m['original_TEST49_all_translation_le5'] for m in methods[33:]},
        no_GT_ID_error_partition_runtime_routing=True, no_primary_native_compatibility_or_fallback_filter=True,
        original_v19_start_names_verified=True, no_e_numeric_pose_prior_or_fallback=True,
        raw_V29_measurements_and_physical_guide_transport_preserved=True,
        V30_association_gate_and_population_CAL_noise_are_bundled_changes=True,
        S_image_2D_or_regions=0, U_regions_only=True, new_precise_2D_corners=0,
        actual_other_instance_occluder_masks_in_current_trial=0, targetmask_absence_is_unknown=True,
        old_global_symmetric_mask_or_tie_used_in_new_methods=False,
        no_new_reference_numeric_binding_changes=True, reused_exploratory_TEST_not_new_holdout=True,
        additional_reference_is_annotation_PnP_derived_not_independent_metrology=True,
        no_per_frame_best_method_mixture=True,
        no_new_images_labels_inference_or_optimization_in_accounting=True,
        independent_root_audits=[P._binding(auditpaths[0])], evidence=bindings,
        protected_user_sha256=active['protected_user_sha256'])
    C.write(paths[0], data)
    paths[1].write_text(_render(data), encoding='utf-8')
    P._verify(active['protected_user_sha256'])
    C.write(paths[2], dict(source=P._binding(Path(__file__)), json=P._binding(paths[0]), html=P._binding(paths[1]),
        evidence=bindings, method_count=37, population_table_rows=148, paired_control_count=8,
        original33_methods_retained_exactly=True, exact_e_preserved=True))
    print(dict(methods=37, table_rows=148, paired_controls=8, json_sha256=C.sha(paths[0])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-v30-score-and-root-audit-go', action='store_true', required=True)
    build(root_postscore_GO=parser.parse_args().after_v30_score_and_root_audit_go)
